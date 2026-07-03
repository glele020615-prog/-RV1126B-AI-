#include "tracker.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

// ===================== KalmanFilter =====================
KalmanFilter::KalmanFilter() {
    std::memset(mean, 0, sizeof(mean));
    std::memset(cov, 0, sizeof(cov));
}

void KalmanFilter::init(float cx, float cy, float s, float r) {
    mean[0] = cx;
    mean[1] = cy;
    mean[2] = s;
    mean[3] = r;
    mean[4] = mean[5] = mean[6] = 0.0f;

    std::memset(cov, 0, sizeof(cov));
    cov[0][0] = 2.0f * s * r;
    cov[1][1] = 2.0f * s / r;
    cov[2][2] = 2.0f * s;
    cov[3][3] = 10.0f * r / (s + 1e-6f);
    cov[4][4] = 1e4f;
    cov[5][5] = 1e4f;
    cov[6][6] = 1e4f;
}

void KalmanFilter::predict() {
    // height = sqrt(s / r)
    float h = std::sqrt(mean[2] / (mean[3] + 1e-6f));

    const float std_pos = 1.0f / 20.0f;
    const float std_vel = 1.0f / 160.0f;

    float q[7];
    q[0] = std_pos * h;
    q[1] = std_pos * h;
    q[2] = std_pos * 1e-2f;
    q[3] = std_pos * h;
    q[4] = std_vel * h;
    q[5] = std_vel * h;
    q[6] = std_vel * 1e-5f;
    // Square for variance
    for (int i = 0; i < 7; i++) q[i] = q[i] * q[i];

    // x = F * x
    mean[0] += mean[4];
    mean[1] += mean[5];
    mean[2] += mean[6];
    // mean[3] unchanged (aspect ratio constant in prediction)

    // P = F * P * F^T + Q
    // FPF[i][j] = sum over a,b of F[i][a] * P[a][b] * F[j][b]
    // Using the simplified form derived above
    float new_cov[7][7];
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            float val;
            if (i < 3 && j < 3) {
                val = cov[i][j] + cov[i+4][j] + cov[i][j+4] + cov[i+4][j+4];
            } else if (i < 3) {
                val = cov[i][j] + cov[i+4][j];
            } else if (j < 3) {
                val = cov[i][j] + cov[i][j+4];
            } else {
                val = cov[i][j];
            }
            new_cov[i][j] = val + (i == j ? q[i] : 0.0f);
        }
    }
    std::memcpy(cov, new_cov, sizeof(cov));

    // Clamp
    if (mean[3] < 0.01f) mean[3] = 0.01f;
    if (mean[2] < 1.0f) mean[2] = 1.0f;
}

void KalmanFilter::update(float cx, float cy, float s, float r) {
    float z[4] = {cx, cy, s, r};

    // y = z - H * x  (innovation; H = [I_4 | 0_4x3])
    float y[4];
    for (int i = 0; i < 4; i++) y[i] = z[i] - mean[i];

    // S = H * P * H^T + R = P[:4,:4] + diag(R)
    float S[4][4];
    const float R_diag = 1.0f;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            S[i][j] = cov[i][j] + (i == j ? R_diag : 0.0f);

    // S^{-1}
    float S_inv[4][4];
    if (!mat_4x4_inv(S, S_inv)) {
        // Fallback: skip update
        return;
    }

    // K = P * H^T * S^{-1} = P[:4, :] * S^{-1}
    // K[i][j] = sum_k P[i][k] * S_inv[k][j] for k in 0..3
    float K[7][4];
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 4; j++) {
            K[i][j] = 0;
            for (int k = 0; k < 4; k++)
                K[i][j] += cov[i][k] * S_inv[k][j];
        }

    // x = x + K * y
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 4; j++)
            mean[i] += K[i][j] * y[j];

    // P = (I - KH) * P = P - K * (first 4 rows of P)
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += K[i][k] * cov[k][j];
            cov[i][j] -= sum;
        }

    // Regularize
    if (mean[3] < 0.01f) mean[3] = 0.01f;
    if (mean[2] < 1.0f) mean[2] = 1.0f;

    // Copy K to member for potential external use
    std::memcpy(K_, K, sizeof(K_));
}

void KalmanFilter::get_bbox(float& x1, float& y1, float& x2, float& y2) const {
    float w = std::sqrt(mean[2] * mean[3] + 1e-6f);
    float h = std::sqrt(mean[2] / mean[3] + 1e-6f);
    x1 = mean[0] - w / 2.0f;
    y1 = mean[1] - h / 2.0f;
    x2 = mean[0] + w / 2.0f;
    y2 = mean[1] + h / 2.0f;
}

bool KalmanFilter::mat_4x4_inv(const float m[4][4], float out[4][4]) const {
    // Cofactor method
    float det = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i % 2 == 0) ? 0 : 1; // sign
        // Minor matrix for m[0][i]
        float minor[3][3];
        int mi = 0, mj;
        for (int r = 1; r < 4; r++) {
            mj = 0;
            for (int c = 0; c < 4; c++) {
                if (c == i) continue;
                minor[mi][mj++] = m[r][c];
            }
            mi++;
        }
        // 3x3 determinant
        float d3 = minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1])
                 - minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0])
                 + minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]);
        det += (i % 2 == 0 ? 1.0f : -1.0f) * m[0][i] * d3;
    }

    if (std::fabs(det) < 1e-12f) return false;
    float inv_det = 1.0f / det;

    // Compute adjugate matrix
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float minor[3][3];
            int mi = 0, mj;
            for (int i = 0; i < 4; i++) {
                if (i == r) continue;
                mj = 0;
                for (int j = 0; j < 4; j++) {
                    if (j == c) continue;
                    minor[mi][mj++] = m[i][j];
                }
                mi++;
            }
            float d3 = minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1])
                     - minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0])
                     + minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]);
            out[c][r] = ((r + c) % 2 == 0 ? 1.0f : -1.0f) * d3 * inv_det;
        }
    }
    return true;
}

// ===================== ByteTracker =====================
ByteTracker::ByteTracker(int max_lost, float track_thresh,
                         float high_thresh, float match_thresh)
    : max_lost_(max_lost), track_thresh_(track_thresh),
      high_thresh_(high_thresh), match_thresh_(match_thresh) {}

Track ByteTracker::create_track(const DetectBox& det) {
    Track t;
    t.track_id = next_id_++;
    t.cls = det.cls;
    t.score = det.score;
    t.x1 = det.x1; t.y1 = det.y1;
    t.x2 = det.x2; t.y2 = det.y2;

    float cx = (det.x1 + det.x2) / 2.0f;
    float cy = (det.y1 + det.y2) / 2.0f;
    float w = det.x2 - det.x1;
    float h = det.y2 - det.y1;
    float s = w * h;
    float r = w / (h + 1e-6f);

    t.kf.init(cx, cy, s, r);
    t.is_activated = true;
    t.conf_history.push_back(det.score);
    return t;
}

void ByteTracker::compute_iou_matrix(
    const std::vector<Track>& tracks,
    const std::vector<size_t>& dets_idx,
    const std::vector<DetectBox>& dets,
    float* iou_matrix, int n_track, int n_det) const
{
    for (int ti = 0; ti < n_track; ti++) {
        const Track& trk = tracks[ti];
        float tx1 = trk.x1, ty1 = trk.y1, tx2 = trk.x2, ty2 = trk.y2;
        float ta = (tx2 - tx1) * (ty2 - ty1);

        for (int di = 0; di < n_det; di++) {
            const DetectBox& det = dets[dets_idx[di]];
            float dx1 = det.x1, dy1 = det.y1, dx2 = det.x2, dy2 = det.y2;

            float xx1 = std::max(tx1, dx1);
            float yy1 = std::max(ty1, dy1);
            float xx2 = std::min(tx2, dx2);
            float yy2 = std::min(ty2, dy2);

            float inter_w = std::max(0.0f, xx2 - xx1);
            float inter_h = std::max(0.0f, yy2 - yy1);
            float inter = inter_w * inter_h;

            float da = (dx2 - dx1) * (dy2 - dy1);
            float uni = ta + da - inter + 1e-6f;
            iou_matrix[ti * n_det + di] = inter / uni;
        }
    }
}

void ByteTracker::greedy_match(
    const float* iou_matrix, int n, int m,
    std::vector<int>& matched_tracks,
    std::vector<int>& matched_dets,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_dets,
    float thresh) const
{
    matched_tracks.assign(n, -1);
    matched_dets.assign(m, -1);

    // Build pair list (iou, track_idx, det_idx)
    struct Pair { float iou; int t; int d; };
    std::vector<Pair> pairs;
    pairs.reserve(n * m);
    for (int t = 0; t < n; t++)
        for (int d = 0; d < m; d++) {
            float iou = iou_matrix[t * m + d];
            if (iou >= thresh)
                pairs.push_back({iou, t, d});
        }

    std::sort(pairs.begin(), pairs.end(),
        [](const Pair& a, const Pair& b) { return a.iou > b.iou; });

    for (const auto& p : pairs) {
        if (matched_tracks[p.t] < 0 && matched_dets[p.d] < 0) {
            matched_tracks[p.t] = p.d;
            matched_dets[p.d] = p.t;
        }
    }

    for (int t = 0; t < n; t++)
        if (matched_tracks[t] < 0) unmatched_tracks.push_back(t);
    for (int d = 0; d < m; d++)
        if (matched_dets[d] < 0) unmatched_dets.push_back(d);
}

void ByteTracker::kalman_predict() {
    for (auto& t : tracks_) {
        t.kf.predict();
        t.kf.get_bbox(t.x1, t.y1, t.x2, t.y2);
    }
}

void ByteTracker::kalman_update(Track& track, const DetectBox& det) {
    float cx = (det.x1 + det.x2) / 2.0f;
    float cy = (det.y1 + det.y2) / 2.0f;
    float w = det.x2 - det.x1;
    float h = det.y2 - det.y1;
    float s = w * h;
    float r = w / (h + 1e-6f);

    track.kf.update(cx, cy, s, r);
    track.kf.get_bbox(track.x1, track.y1, track.x2, track.y2);
    track.score = det.score;
    track.cls = det.cls;
    track.missed = 0;
    track.age++;
    track.conf_history.push_back(det.score);
    if ((int)track.conf_history.size() > 30)
        track.conf_history.erase(track.conf_history.begin());
}

std::vector<Track> ByteTracker::update(const std::vector<DetectBox>& dets) {
    frame_id_++;
    activated_.clear();

    // Step 1: Predict all existing tracks
    kalman_predict();

    // Step 2: Partition detections into high and low score
    std::vector<size_t> high_dets, low_dets;
    for (size_t i = 0; i < dets.size(); i++) {
        if (dets[i].score >= high_thresh_)
            high_dets.push_back(i);
        else if (dets[i].score >= track_thresh_)
            low_dets.push_back(i);
    }

    // Step 3: First matching round ? high-score detections with all tracks
    std::vector<int> trk_match(tracks_.size(), -1);
    std::vector<int> det_match_high(high_dets.size(), -1);

    if (!tracks_.empty() && !high_dets.empty()) {
        int nt = (int)tracks_.size();
        int nh = (int)high_dets.size();
        std::vector<float> iou_mat(nt * nh);
        compute_iou_matrix(tracks_, high_dets, dets, iou_mat.data(), nt, nh);

        std::vector<int> unmatched_t, unmatched_d;
        greedy_match(iou_mat.data(), nt, nh,
                     trk_match, det_match_high,
                     unmatched_t, unmatched_d, match_thresh_);
    }

    // Step 4: Second matching round ? low-score detections with unmatched tracks
    std::vector<int> unmatched_tracks;
    for (size_t t = 0; t < tracks_.size(); t++) {
        if (trk_match[t] < 0) unmatched_tracks.push_back((int)t);
    }

    std::vector<int> det_match_low(low_dets.size(), -1);
    if (!unmatched_tracks.empty() && !low_dets.empty()) {
        int nu = (int)unmatched_tracks.size();
        int nl = (int)low_dets.size();

        // Extract only the unmatched tracks data
        std::vector<Track> unmatched_trk_data;
        for (int t : unmatched_tracks)
            unmatched_trk_data.push_back(tracks_[t]);

        std::vector<float> iou_mat2(nu * nl);
        compute_iou_matrix(unmatched_trk_data, low_dets, dets, iou_mat2.data(), nu, nl);

        std::vector<int> trk_match2(nu, -1);
        std::vector<int> det_match2;
        std::vector<int> unused_u, unused_d;
        greedy_match(iou_mat2.data(), nu, nl,
                     trk_match2, det_match_low,
                     unused_u, unused_d, 0.5f);

        // Transfer matches to full track matrix
        for (int u = 0; u < nu; u++) {
            if (trk_match2[u] >= 0) {
                int track_idx = unmatched_tracks[u];
                trk_match[track_idx] = trk_match2[u] + (int)high_dets.size();  // offset to avoid overlap
            }
        }
    }

    // Step 5: Apply updates from both rounds
    // Mark all tracks as missed initially
    for (auto& t : tracks_) t.missed++;

    for (size_t t = 0; t < tracks_.size(); t++) {
        if (trk_match[t] >= 0) {
            int det_idx;
            // Was it matched to high or low?
            if ((size_t)trk_match[t] < high_dets.size()) {
                det_idx = high_dets[trk_match[t]];
            } else {
                // It's a low-score match
                int low_idx = trk_match[t] - (int)high_dets.size();
                if (low_idx < (int)low_dets.size())
                    det_idx = low_dets[low_idx];
                else
                    continue;
            }
            kalman_update(tracks_[t], dets[det_idx]);
        }
    }

    // Step 6: Create new tracks from unmatched high-score detections
    for (size_t d = 0; d < high_dets.size(); d++) {
        if (det_match_high[d] < 0) {
            tracks_.push_back(create_track(dets[high_dets[d]]));
        }
    }

    // Step 7: Remove lost tracks
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [this](const Track& t) { return t.missed > max_lost_; }),
        tracks_.end());

    // Step 8: Generate output (only activated tracks)
    for (const auto& t : tracks_) {
        if (t.is_activated)
            activated_.push_back(t);
    }

    return activated_;
}
