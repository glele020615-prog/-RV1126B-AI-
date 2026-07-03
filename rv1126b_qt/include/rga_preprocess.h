#pragma once
#include <cstdint>
#include "detector.h"

// 模型输入：NV12 摄像头帧 -> RGB888 letterbox 1088x1088
bool rga_nv12_to_rgb_letterbox(
    void* nv12, int cam_w, int cam_h,
    unsigned char* rgb_out, int out_size,
    const LetterboxInfo& lb
);

// Qt 显示：NV12 摄像头帧 -> RGB888 原始分辨率
bool rga_nv12_to_rgb_full(
    void* nv12, int src_w, int src_h,
    unsigned char* rgb_out, int dst_w, int dst_h
);
