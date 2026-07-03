#pragma once
#include <cstdint>
#include <cstddef>

#define V4L2_BUF_COUNT 4
#define CAMERA_V4L2_H

// 根据你的实际摄像头分辨率设置
#define CAM_W 1920
#define CAM_H 1080

class V4L2Camera {
public:
    int fd = -1;

    struct CameraBuffer {
        void* start = nullptr;
        size_t length = 0;
    };

    CameraBuffer buffers[V4L2_BUF_COUNT];

    bool open(const char* dev);
    bool get_frame(void** data, int* index);
    void release_frame(int index);
    void close();
};
