#include "rga_preprocess.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include "im2d.h"
#include "RgaUtils.h"

bool rga_nv12_to_rgb_letterbox(
    void* nv12, int cam_w, int cam_h,
    unsigned char* rgb_out, int out_size,
    const LetterboxInfo& lb)
{
    std::memset(rgb_out, 114, out_size);

    rga_buffer_t src = wrapbuffer_virtualaddr(nv12, cam_w, cam_h,
                                              RK_FORMAT_YCbCr_420_SP);

    static std::vector<unsigned char> tmp_buf;
    size_t tmp_size = (size_t)lb.new_w * lb.new_h * 3;
    if (tmp_buf.size() != tmp_size) {
        tmp_buf.resize(tmp_size);
    }

    rga_buffer_t dst_tmp = wrapbuffer_virtualaddr(
        tmp_buf.data(), lb.new_w, lb.new_h, RK_FORMAT_RGB_888);

    int ret = imresize(src, dst_tmp);
    if (ret != IM_STATUS_SUCCESS) {
        printf("RGA imresize NV12->RGB letterbox failed: %d\n", ret);
        return false;
    }

    for (int y = 0; y < lb.new_h; ++y) {
        unsigned char* dst_line = rgb_out +
            ((y + lb.pad_y) * IMG_SIZE + lb.pad_x) * 3;
        unsigned char* src_line = tmp_buf.data() + y * lb.new_w * 3;
        std::memcpy(dst_line, src_line, (size_t)lb.new_w * 3);
    }

    return true;
}

bool rga_nv12_to_rgb_full(
    void* nv12, int src_w, int src_h,
    unsigned char* rgb_out, int dst_w, int dst_h)
{
    rga_buffer_t src = wrapbuffer_virtualaddr(nv12, src_w, src_h,
                                              RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t dst = wrapbuffer_virtualaddr(rgb_out, dst_w, dst_h,
                                              RK_FORMAT_RGB_888);

    int ret = imresize(src, dst);

    if (ret != IM_STATUS_SUCCESS) {
        printf("RGA NV12->RGB full frame failed: %d\n", ret);
        return false;
    }
    return true;
}
