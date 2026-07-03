#pragma once

#include <vector>
#include "detector.h"
#include "rknn_api.h"


#define NUM_CLASSES 2   // 火和烟
#define IMG_SIZE 1088  // 模型输入尺寸

struct RKNNEngine {
    rknn_context ctx = 0;
    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> output_attrs;
};

int rknn_init_engine(RKNNEngine& engine, const char* model_path);

std::vector<DetectBox> rknn_infer(
    RKNNEngine& engine,
    unsigned char* rgb_input,
    const LetterboxInfo& lb
);

void rknn_release_engine(RKNNEngine& engine);
