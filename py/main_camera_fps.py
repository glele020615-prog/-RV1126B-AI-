import cv2
import time
import sys
import re

from rknnpool.rknnpool_ld import rknnPoolExecutor
from func.func_yolov11_optimize import myFunc, draw_detections

out_win = "output_style_full_screen"
modelPath = "best_nano_111_rv1126b_hybrid.rknn"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def open_camera(preferred_device="/dev/video23"):
    candidates = [preferred_device, "/dev/video23", "/dev/video25", "/dev/video0", "/dev/video24", "/dev/video26", "/dev/video30"]
    tried = []

    for device in candidates:
        if device in tried:
            continue
        tried.append(device)

        gstreamer_pipeline = (
            f"v4l2src device={device} ! "
            f"video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
            # f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
            f"appsink max-buffers=1 drop=true sync=false"
        )

        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"Using GStreamer camera pipeline: {device}")
            return cap
        cap.release()

        device_match = re.search(r'\d+$', device)
        if device_match:
            cap = cv2.VideoCapture(int(device_match.group()), cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Using V4L2 camera device: {device}")
            return cap
        cap.release()

    raise RuntimeError(f"Unable to open camera device. Tried: {', '.join(tried)}")

# 线程数, 过大容易造成画面积压和延迟
TPEs = 1

preferred_device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video23"
cap = open_camera(preferred_device)

# 初始化rknn池
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc
)

cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(out_win, DISPLAY_WIDTH, DISPLAY_HEIGHT)

# 初始化异步所需要的帧
last_processed_data = None
infer_frames = 0
infer_loop_time = time.time()

frames, loopTime, initTime=0, time.time(), time.time()
ret, frame = cap.read()
print(ret, frame.shape, frame.dtype)
while(cap.isOpened()):
    frames+=1
    ret,frame=cap.read()
    if not ret:
        break


    pool.put(frame)

    processed_data, flag = pool.get()
    if flag:
        infer_frames += 1
        last_processed_data = processed_data

    display_frame = frame
    if last_processed_data is not None:
        display_frame, _ = draw_detections(display_frame, *last_processed_data)
        
    cv2.imshow(out_win, display_frame)



    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    if frames % 30 == 0:
        now = time.time()
        print("主循环/显示FPS:\t", 30 / (now - loopTime), "帧")
        loopTime = now

        print("实际推理FPS:\t", infer_frames / (now - infer_loop_time), "帧")
        infer_frames = 0
        infer_loop_time = now
print("总平均帧率\t", frames/(time.time()-initTime))

#释放cap和rknn线程池
cap.release()
cv2.destroyAllWindows()
pool.release()