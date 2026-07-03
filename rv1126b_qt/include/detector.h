#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

struct DetectBox {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

struct LetterboxInfo {
    float ratio;
    int pad_x, pad_y;
    int new_w, new_h;
};

inline LetterboxInfo calc_letterbox(int src_w, int src_h, int dst_size) {
    LetterboxInfo info{};
    float r = std::min((float)dst_size / src_w, (float)dst_size / src_h);
    info.new_w = (int)std::round(src_w * r);
    info.new_h = (int)std::round(src_h * r);
    info.pad_x = (dst_size - info.new_w) / 2;
    info.pad_y = (dst_size - info.new_h) / 2;
    info.ratio = r;
    return info;
}

// static constexpr int CAM_W = 1920;
// static constexpr int CAM_H = 1080;
static constexpr int IMG_SIZE = 1088;
static constexpr int NUM_CLASSES = 2;
static constexpr const char* CLASS_NAMES[2] = {"Fire", "Smoke"};
