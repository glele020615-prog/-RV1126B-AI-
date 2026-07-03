#include "rknn_engine.h"
#include "camera_v4l2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static std::vector<unsigned char> load_file(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen model");
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    std::vector<unsigned char> data(size);
    fread(data.data(), 1, size, fp);
    fclose(fp);

    return data;
}

static float sigmoid(float x) {
    if (x > 50) x = 50;
    if (x < -50) x = -50;
    return 1.0f / (1.0f + std::exp(-x));
}

static float iou(const DetectBox& a, const DetectBox& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area_a = std::max(0.0f, a.x2 - a.x1) *
                   std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) *
                   std::max(0.0f, b.y2 - b.y1);

    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<DetectBox> nms_class_aware(
    std::vector<DetectBox>& boxes,
    float nms_thres)
{
    std::sort(boxes.begin(), boxes.end(),
        [](const DetectBox& a, const DetectBox& b) {
            return a.score > b.score;
        });

    std::vector<DetectBox> result;
    std::vector<int> removed(boxes.size(), 0);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            if (boxes[i].cls != boxes[j].cls) continue;
            if (iou(boxes[i], boxes[j]) > nms_thres) {
                removed[j] = 1;
            }
        }
    }

    return result;
}

static std::vector<DetectBox> postprocess_two_outputs(
    float* boxes_data,
    int boxes_count,
    float* scores_data,
    int scores_count,
    const LetterboxInfo& lb)
{
    std::vector<DetectBox> candidates;

    int num_preds = boxes_count / 4;
    if (scores_count / NUM_CLASSES < num_preds) {
        num_preds = scores_count / NUM_CLASSES;
    }

    const float conf_thres[2] = {0.20f, 0.30f};  // Fire, Smoke

    for (int i = 0; i < num_preds; ++i) {
        float x = boxes_data[i * 4 + 0];
        float y = boxes_data[i * 4 + 1];
        float w = boxes_data[i * 4 + 2];
        float h = boxes_data[i * 4 + 3];

        float s0 = scores_data[i * NUM_CLASSES + 0];
        float s1 = scores_data[i * NUM_CLASSES + 1];

        if (s0 < 0 || s0 > 1) s0 = sigmoid(s0);
        if (s1 < 0 || s1 > 1) s1 = sigmoid(s1);

        int cls = s0 >= s1 ? 0 : 1;
        float score = cls == 0 ? s0 : s1;

        if (score < conf_thres[cls]) continue;
        if (w <= 1 || h <= 1) continue;

        DetectBox box;
        box.x1 = x - w / 2.0f;
        box.y1 = y - h / 2.0f;
        box.x2 = x + w / 2.0f;
        box.y2 = y + h / 2.0f;
        box.score = score;
        box.cls = cls;

        if (box.x2 <= box.x1 || box.y2 <= box.y1) continue;

        candidates.push_back(box);
    }

    const int PRE_NMS_TOPK = 300;
    if ((int)candidates.size() > PRE_NMS_TOPK) {
        std::sort(candidates.begin(), candidates.end(),
            [](const DetectBox& a, const DetectBox& b) {
                return a.score > b.score;
            });
        candidates.resize(PRE_NMS_TOPK);
    }

    auto dets = nms_class_aware(candidates, 0.45f);

    const int MAX_DET = 20;
    if ((int)dets.size() > MAX_DET) dets.resize(MAX_DET);

    for (auto& d : dets) {
        d.x1 = (d.x1 - lb.pad_x) / lb.ratio;
        d.y1 = (d.y1 - lb.pad_y) / lb.ratio;
        d.x2 = (d.x2 - lb.pad_x) / lb.ratio;
        d.y2 = (d.y2 - lb.pad_y) / lb.ratio;

        d.x1 = std::max(0.0f, std::min((float)CAM_W - 1, d.x1));
        d.y1 = std::max(0.0f, std::min((float)CAM_H - 1, d.y1));
        d.x2 = std::max(0.0f, std::min((float)CAM_W - 1, d.x2));
        d.y2 = std::max(0.0f, std::min((float)CAM_H - 1, d.y2));
    }

    return dets;
}

int rknn_init_engine(RKNNEngine& engine, const char* model_path) {
    std::vector<unsigned char> model_data = load_file(model_path);

    int ret = rknn_init(&engine.ctx, model_data.data(),
                        model_data.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        printf("rknn_init failed: %d\n", ret);
        return -1;
    }

    ret = rknn_query(engine.ctx, RKNN_QUERY_IN_OUT_NUM,
                     &engine.io_num, sizeof(engine.io_num));
    if (ret != RKNN_SUCC) {
        printf("RKNN_QUERY_IN_OUT_NUM failed: %d\n", ret);
        return -1;
    }

    printf("RKNN input num=%d output num=%d\n",
           engine.io_num.n_input, engine.io_num.n_output);

    engine.output_attrs.resize(engine.io_num.n_output);

    for (int i = 0; i < engine.io_num.n_output; ++i) {
        memset(&engine.output_attrs[i], 0, sizeof(rknn_tensor_attr));
        engine.output_attrs[i].index = i;

        rknn_query(engine.ctx, RKNN_QUERY_OUTPUT_ATTR,
                   &engine.output_attrs[i], sizeof(rknn_tensor_attr));

        printf("output[%d]: name=%s n_dims=%d dims=[",
               i, engine.output_attrs[i].name,
               engine.output_attrs[i].n_dims);

        for (int j = 0; j < engine.output_attrs[i].n_dims; ++j) {
            printf("%d%s", engine.output_attrs[i].dims[j],
                   j == engine.output_attrs[i].n_dims - 1 ? "" : ",");
        }
        printf("] type=%d qnt_type=%d\n",
               engine.output_attrs[i].type,
               engine.output_attrs[i].qnt_type);
    }

    return 0;
}

std::vector<DetectBox> rknn_infer(
    RKNNEngine& engine,
    unsigned char* rgb_input,
    const LetterboxInfo& lb)
{
    std::vector<DetectBox> dets;

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].buf = rgb_input;
    inputs[0].size = IMG_SIZE * IMG_SIZE * 3;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    int ret = rknn_inputs_set(engine.ctx, 1, inputs);
    if (ret != RKNN_SUCC) {
        printf("rknn_inputs_set failed: %d\n", ret);
        return dets;
    }

    ret = rknn_run(engine.ctx, nullptr);
    if (ret != RKNN_SUCC) {
        printf("rknn_run failed: %d\n", ret);
        return dets;
    }

    std::vector<rknn_output> outputs(engine.io_num.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output) * engine.io_num.n_output);

    for (int i = 0; i < engine.io_num.n_output; ++i) {
        outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(engine.ctx, engine.io_num.n_output,
                           outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        printf("rknn_outputs_get failed: %d\n", ret);
        return dets;
    }

    if (engine.io_num.n_output >= 2) {
        float* boxes_data = (float*)outputs[0].buf;
        float* scores_data = (float*)outputs[1].buf;

        int boxes_count = outputs[0].size / sizeof(float);
        int scores_count = outputs[1].size / sizeof(float);

        dets = postprocess_two_outputs(boxes_data, boxes_count,
                                       scores_data, scores_count, lb);
    } else {
        printf("Model has %d outputs; expected >=2 (boxes + scores)\n",
               engine.io_num.n_output);
    }

    rknn_outputs_release(engine.ctx, engine.io_num.n_output, outputs.data());
    return dets;
}

void rknn_release_engine(RKNNEngine& engine) {
    if (engine.ctx) {
        rknn_destroy(engine.ctx);
        engine.ctx = 0;
    }
}
