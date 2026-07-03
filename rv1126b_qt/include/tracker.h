#pragma once
#include "detector.h"
#include <vector>

// ==================== KalmanFilter (7-state constant velocity) ====================
class KalmanFilter {
public:
    // State: [cx, cy, s(area), r(aspect), d_cx, d_cy, d_s]
    float mean[7];
    float cov[7][7];  // 7x7 covariance

    KalmanFilter();

    void init(float cx, float cy, float s, float r);
    void predict();
    void update(float cx, float cy, float s, float r);

    // Convert state to [x1, y1, x2, y2]
    void get_bbox(float& x1, float& y1, float& x2, float& y2) const;

private:
    // Pre-allocated scratch buffers
    float tmp7_[7];
    float tmp7x7_[7][7];
    float tmp4_[4];
    float tmp4x7_[4][7];
    float tmp7x4_[7][4];
    float tmp4x4_[4][4];
    float K_[7][4];  // Kalman gain 7x4

    void mat_7x7_mul_7x1(const float a[7][7], const float b[7], float out[7]) const;
    void mat_7x7_mul_7x7(const float a[7][7], const float b[7][7], float out[7][7]) const;
    void mat_7x7_mul_4x7T(const float a[7][7], const float b[4][7], float out[7][4]) const;
    void mat_4x7_mul_7x7(const float a[4][7], const float b[7][7], float out[4][7]) const;
    void mat_4x7_mul_7x4(const float a[4][7], const float b[7][4], float out[4][4]) const;
    void mat_7x4_mul_4x7(const float a[7][4], const float b[4][7], float out[7][7]) const;
    bool mat_4x4_inv(const float m[4][4], float out[4][4]) const;
    void mat_7x4_mul_4x4(const float a[7][4], const float b[4][4], float out[7][4]) const;
    void mat_7x4_mul_4x1(const float a[7][4], const float b[4], float out[7]) const;
    void mat_4x7_mul_7x1(const float a[4][7], const float b[7], float out[4]) const;
    void vec7_sub(float a[7], const float b[7]) const;
    void vec4_sub(float a[4], const float b[4]) const;
    void eye_7x7(float out[7][7]) const;
};

// ==================== Track ====================
struct Track {
    int track_id = -1;
    KalmanFilter kf;
    float x1, y1, x2, y2;      // Current bbox in image coords
    float score = 0;
    int cls = 0;
    int age = 0;                // Total frames since birth
    int missed = 0;             // Consecutive missed frames
    bool is_activated = false;
    std::vector<float> conf_history;  // For confidence smoothing
};

// ==================== ByteTracker ====================
class ByteTracker {
public:
    ByteTracker(int max_lost = 30,
                float track_thresh = 0.5f,
                float high_thresh = 0.6f,
                float match_thresh = 0.8f);

    // Main update: takes detections, returns tracked objects
    std::vector<Track> update(const std::vector<DetectBox>& dets);

private:
    int next_id_ = 1;
    int frame_id_ = 0;
    int max_lost_;
    float track_thresh_;
    float high_thresh_;
    float match_thresh_;

    std::vector<Track> tracks_;
    std::vector<Track> activated_;  // result of current frame

    // Matching helpers
    void compute_iou_matrix(const std::vector<Track>& tracks,
                            const std::vector<size_t>& dets_idx,
                            const std::vector<DetectBox>& dets,
                            float* iou_matrix, int n_track, int n_det) const;

    void greedy_match(const float* iou_matrix, int n, int m,
                      std::vector<int>& matched_tracks,
                      std::vector<int>& matched_dets,
                      std::vector<int>& unmatched_tracks,
                      std::vector<int>& unmatched_dets,
                      float thresh) const;

    void kalman_predict();
    void kalman_update(Track& track, const DetectBox& det);
    Track create_track(const DetectBox& det);
};
