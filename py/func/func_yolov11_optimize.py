import cv2
import numpy as np

CLASS_NAMES = ['Fire', 'Smoke']
IMG_SIZE = 1088
CLASS_CONF_THRES = {
    0: 0.6,   # Fire
    1: 0.7,   # Smoke
}
PRE_NMS_TOPK = 300
NMS_THRES = 0.45
MAX_DET = 20
CLASS_COLORS = {
    'Fire': (255, 0, 0),
    'Smoke': (255, 255, 0),
}


def letterbox(im, new_shape=(1088, 1088), color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def xywh2xyxy(box):
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def nms_boxes(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep

def class_aware_nms(boxes, scores, class_ids, iou_thres=0.45):
    final_keep = []

    class_ids_np = np.array(class_ids)

    for cls in np.unique(class_ids_np):
        idxs = np.where(class_ids_np == cls)[0]

        cls_boxes = [boxes[i] for i in idxs]
        cls_scores = [scores[i] for i in idxs]

        keep = nms_boxes(cls_boxes, cls_scores, iou_thres)

        for k in keep:
            final_keep.append(idxs[k])

    final_keep = sorted(final_keep, key=lambda i: scores[i], reverse=True)
    return final_keep


def scale_coords(box, ratio, pad, orig_shape):
    left, top = pad
    x1, y1, x2, y2 = box

    x1 = (x1 - left) / ratio
    y1 = (y1 - top) / ratio
    x2 = (x2 - left) / ratio
    y2 = (y2 - top) / ratio

    h, w = orig_shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    return [x1, y1, x2, y2]


def prepare_prediction_array(output):
    pred = np.asarray(output)

    if pred.ndim == 3:
        pred = pred[0]

    if pred.ndim != 2:
        raise ValueError(f'Unexpected output shape: {pred.shape}')

    if pred.shape[1] == 4 + len(CLASS_NAMES):
        return pred

    if pred.shape[0] == 4 + len(CLASS_NAMES):
        return pred.transpose(1, 0)

    raise ValueError(f'Unexpected prediction layout: {pred.shape}')


def build_input_variants(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized, ratio, pad = letterbox(frame_rgb, (IMG_SIZE, IMG_SIZE))

    # 方案 A：uint8 NHWC，很多 RKNN 量化模型更常见
    tensor_uint8_nhwc = resized.astype(np.uint8)[None, ...]

    # # 方案 B：int8 NCHW，你现在的写法
    # tensor_int8_nchw = resized.astype(np.int16) - 128
    # tensor_int8_nchw = np.clip(tensor_int8_nchw, -128, 127).astype(np.int8)
    # tensor_int8_nchw = np.ascontiguousarray(
    #     np.transpose(tensor_int8_nchw, (2, 0, 1))[None, ...]
    # )

    return [
        ('rgb_uint8_nhwc', tensor_uint8_nhwc, ratio, pad),
        # ('rgb_nchw_int8', tensor_int8_nchw, ratio, pad),
    ]


def decode_detections(pred, debug=False):
    boxes = []
    scores = []
    class_ids = []

    cls_scores_all = pred[:, 4:4 + len(CLASS_NAMES)]

    for det in pred:
        x, y, w, h = det[:4]
        class_scores = det[4:4 + len(CLASS_NAMES)].astype(np.float32)

        if class_scores.size == 0:
            continue

        if np.any(class_scores < 0) or np.any(class_scores > 1):
            class_scores = sigmoid(class_scores)

        cls_id = int(np.argmax(class_scores))
        conf = float(class_scores[cls_id])

        conf_thres = CLASS_CONF_THRES.get(cls_id, 0.25)
        if conf < conf_thres:
            continue

        # 过滤非法框：宽高异常会导致后面画不出来
        if w <= 1 or h <= 1:
            continue

        # 过滤明显越界/异常大框，可按你的模型输出范围调整
        if not np.isfinite([x, y, w, h]).all():
            continue

        box = xywh2xyxy([x, y, w, h])

        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append(box)
        scores.append(conf)
        class_ids.append(cls_id)

    if debug:
        print(
            "decode:",
            "raw preds =", len(pred),
            "decoded boxes =", len(boxes),
            "fire max =", float(cls_scores_all[:, 0].max()),
            "smoke max =", float(cls_scores_all[:, 1].max())
        )

        if len(scores) > 0:
            top_idx = np.argsort(scores)[::-1][:5]
            for i in top_idx:
                print(
                    "top det:",
                    CLASS_NAMES[class_ids[i]],
                    "score =", scores[i],
                    "box =", boxes[i]
                )

    if len(scores) > PRE_NMS_TOPK:
        top_idx = np.argsort(scores)[::-1][:PRE_NMS_TOPK]
        boxes = [boxes[i] for i in top_idx]
        scores = [scores[i] for i in top_idx]
        class_ids = [class_ids[i] for i in top_idx]

    return boxes, scores, class_ids


def draw_detections(frame, boxes, scores, class_ids, ratio, pad):
    if boxes is None or len(boxes) == 0:
        return frame, None

    detection = None

    for i in range(len(boxes)):
        box = scale_coords(boxes[i], ratio, pad, frame.shape)
        score = scores[i]
        cls_id = class_ids[i]
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        x1, y1, x2, y2 = box
        
        # 适当扩大检测框，以完全框住火源（外扩 10%-15%）
        bw = x2 - x1
        bh = y2 - y1
        x1 = max(0, x1 - int(bw * 0.1))
        y1 = max(0, y1 - int(bh * 0.15))
        x2 = min(frame.shape[1] - 1, x2 + int(bw * 0.1))
        y2 = min(frame.shape[0] - 1, y2 + int(bh * 0.15))

        color = CLASS_COLORS.get(label, (255, 0, 0))

        label_text = f'{label} {score:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_bottom = max(0, y1 - 6)
        text_top = max(0, text_bottom - text_height - baseline - 6)
        text_right = min(frame.shape[1] - 1, x1 + text_width + 8)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(frame, (x1, text_top), (text_right, text_bottom), color, -1)
        cv2.putText(frame, label_text, (x1 + 4, text_bottom - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if detection is None:
            detection = label

    return frame, detection

def to_2d_channels(output, channels):
    arr = np.asarray(output)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f'Unexpected output shape: {arr.shape}')

    if arr.shape[0] == channels:
        return arr.transpose(1, 0)

    if arr.shape[1] == channels:
        return arr

    raise ValueError(f'Unexpected output layout: {arr.shape}, channels={channels}')


def prepare_rknn_outputs(outputs):
    # 新模型：两个输出 boxes_out + scores_out
    if len(outputs) >= 2:
        boxes = to_2d_channels(outputs[0], 4)
        scores = to_2d_channels(outputs[1], len(CLASS_NAMES))

        # print(
        #     "boxes shape:", boxes.shape,
        #     "scores shape:", scores.shape,
        #     "scores min:", scores.min(),
        #     "scores max:", scores.max()
        # )

        return np.concatenate([boxes, scores], axis=1)

    # 旧模型：一个 output0
    return prepare_prediction_array(outputs[0])


def myFunc(rknn_lite, frame):
    original = frame.copy()
    DEBUG = False

    for variant_name, tensor, ratio, pad in build_input_variants(original):
        outputs = rknn_lite.inference(inputs=[tensor])
        if outputs is None or len(outputs) == 0:
            continue

        pred = prepare_rknn_outputs(outputs)
        boxes, scores, class_ids = decode_detections(pred, debug=DEBUG)

        if len(boxes) == 0:
            continue

        keep = class_aware_nms(boxes, scores, class_ids, NMS_THRES)
        keep = keep[:MAX_DET]

        # 如果 NMS 之后没有框，直接保留 decode 原始前几框
        if len(keep) == 0 and len(boxes) > 0:
            keep = list(range(min(MAX_DET, len(boxes))))

        if len(keep) > 0:
            filtered_boxes = [boxes[i] for i in keep]
            filtered_scores = [scores[i] for i in keep]
            filtered_cls = [class_ids[i] for i in keep]
            return (filtered_boxes, filtered_scores, filtered_cls, ratio, pad)

    return None
