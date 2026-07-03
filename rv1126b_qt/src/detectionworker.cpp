#include "detectionworker.h"

#include "camera_v4l2.h"
#include "controller.h"
#include "detector.h"
#include "rga_preprocess.h"
#include "rknn_engine.h"
#include "tracker.h"

#include <QMutexLocker>
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QFont>
#include <QFontMetrics>
#include <QThread>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

DetectionWorker::DetectionWorker(QObject *parent)
    : QObject(parent)
{
}

DetectionWorker::~DetectionWorker()
{
    stopWork();
}

void DetectionWorker::stopWork()
{
    QMutexLocker locker(&mutex_);
    stopRequested_ = true;
}

bool DetectionWorker::isStopRequested()
{
    QMutexLocker locker(&mutex_);
    return stopRequested_;
}

QString DetectionWorker::phaseToString(int phase) const
{
    switch (static_cast<ControlPhase>(phase)) {
    case ControlPhase::IDLE:
        return "IDLE";
    case ControlPhase::CONFIRMING:
        return "CONFIRMING";
    case ControlPhase::ACTIVE:
        return "ACTIVE";
    case ControlPhase::RELEASING:
        return "RELEASING";
    default:
        return "UNKNOWN";
    }
}

static void drawDetectionOverlay(QImage& frame,
                                 const std::vector<Track>& tracks,
                                 double fps)
{
    QPainter painter(&frame);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // ================= FPS floating box =================
    QFont fpsFont("Arial", 18, QFont::Bold);
    painter.setFont(fpsFont);

    QRectF fpsRect(18, 18, 125, 48);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(0, 0, 0, 175));
    painter.drawRoundedRect(fpsRect, 8, 8);

    painter.setPen(Qt::white);
    painter.drawText(fpsRect,
                     Qt::AlignCenter,
                     QString("FPS: %1").arg(fps, 0, 'f', 1));

    // ================= Detection boxes =================
    for (const auto& t : tracks) {
        if (!t.is_activated || t.missed > 0) {
            continue;
        }

        const bool isFire = (t.cls == 0);
        const QColor boxColor = isFire
                ? QColor(255, 35, 35)
                : QColor(255, 205, 0);

        const QColor textColor = isFire
                ? QColor(255, 255, 255)
                : QColor(0, 0, 0);

        QRectF boxRect(QPointF(t.x1, t.y1), QPointF(t.x2, t.y2));
        boxRect = boxRect.normalized();

        if (boxRect.width() < 2 || boxRect.height() < 2) {
            continue;
        }

        boxRect.setLeft(std::max(0.0, boxRect.left()));
        boxRect.setTop(std::max(0.0, boxRect.top()));
        boxRect.setRight(std::min(static_cast<double>(frame.width() - 1),
                                  boxRect.right()));
        boxRect.setBottom(std::min(static_cast<double>(frame.height() - 1),
                                   boxRect.bottom()));

        // Main rectangle, similar to the screenshot
        QPen boxPen(boxColor);
        boxPen.setWidth(4);
        painter.setPen(boxPen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(boxRect);

        // Label text: Fire ID:13 0.93 / Smoke ID:25 0.87
        QString label = QString("%1 ID:%2  %3")
                .arg(isFire ? "Fire" : "Smoke")
                .arg(t.track_id)
                .arg(t.score, 0, 'f', 2);

        QFont labelFont("Arial", 17, QFont::Bold);
        painter.setFont(labelFont);

        QFontMetrics fm(labelFont);
        int labelW = fm.horizontalAdvance(label) + 18;
        int labelH = 34;

        double labelX = boxRect.left();
        double labelY = boxRect.top() - labelH;

        if (labelY < 0) {
            labelY = boxRect.top();
        }

        if (labelX + labelW > frame.width()) {
            labelX = frame.width() - labelW - 1;
        }

        if (labelX < 0) {
            labelX = 0;
        }

        QRectF labelRect(labelX, labelY, labelW, labelH);

        painter.setPen(Qt::NoPen);
        painter.setBrush(boxColor);
        painter.drawRoundedRect(labelRect, 5, 5);

        painter.setPen(textColor);
        painter.drawText(labelRect.adjusted(8, 0, -8, 0),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         label);
    }

    painter.end();
}

void DetectionWorker::startWork(const QString& cameraDev,
                                const QString& modelPath)
{
    {
        perf_.initCSV("perf_log.csv");
        
        QMutexLocker locker(&mutex_);
        stopRequested_ = false;
    }

    emit logMessage("DetectionWorker starting...");

    const QByteArray devBytes = cameraDev.toLocal8Bit();
    const QByteArray modelBytes = modelPath.toLocal8Bit();

    const char* dev = devBytes.constData();
    const char* model = modelBytes.constData();

    LetterboxInfo lb = calc_letterbox(CAM_W, CAM_H, IMG_SIZE);

    V4L2Camera cam;
    if (!cam.open(dev)) {
        emit logMessage("FATAL: camera open failed");
        emit finished();
        return;
    }

    RKNNEngine rknn;
    if (rknn_init_engine(rknn, model) != 0) {
        emit logMessage("FATAL: RKNN init failed");
        cam.close();
        emit finished();
        return;
    }

    ByteTracker tracker(30, 0.35f, 0.65f, 0.50f);

    ControlConfig ctrlCfg;
    ctrlCfg.confirm_frames   = 5;
    ctrlCfg.release_frames   = 15;
    ctrlCfg.ramp_up_frames   = 10;
    ctrlCfg.ramp_down_frames = 20;
    ctrlCfg.intensity_min    = 0.15f;
    ctrlCfg.intensity_max    = 1.0f;
    ctrlCfg.fire_boost       = 1.3f;

    ctrlCfg.use_pwm          = false;
    ctrlCfg.use_gpio         = true;

    ctrlCfg.pwm_chip         = 0;
    ctrlCfg.pwm_channel      = 0;
    ctrlCfg.pwm_period_ns    = 50000;

    // 原工程里使用 GPIO182 控制输出，保持不变 GPIO5_C6
    ctrlCfg.gpio_pin         = 182;

    FireController controller;
    if (!controller.init(ctrlCfg)) {
        emit logMessage("WARN: FireController init failed; continuing");
    }

    std::vector<unsigned char> modelRgb(IMG_SIZE * IMG_SIZE * 3);
    std::vector<unsigned char> displayRgb(CAM_W * CAM_H * 3);

    int frames = 0;
    auto fpsTime = std::chrono::steady_clock::now();
    double fps = 0.0;

    // ---------- 新增：性能累计变量 ----------
    double sum_pre = 0.0, sum_inf = 0.0, sum_track = 0.0, sum_ctrl = 0.0, sum_lat = 0.0;
    int frames_since_log = 0;
    // --------------------------------------

    emit logMessage("DetectionWorker running");

    while (!isStopRequested()) {
        auto t_frame_start = std::chrono::steady_clock::now();   // 端到端开始

        void* nv12 = nullptr;
        int bufIdx = -1;

        if (!cam.get_frame(&nv12, &bufIdx)) {
            QThread::msleep(1);
            continue;
        }

        ++frames;
        ++frames_since_log;

        // ---- 1. 预处理计时 ----
        auto t_pre_begin = std::chrono::steady_clock::now();
        const bool prepOk = rga_nv12_to_rgb_letterbox(
            nv12,
            CAM_W,
            CAM_H,
            modelRgb.data(),
            static_cast<int>(modelRgb.size()),
            lb
        );
        auto t_pre_end = std::chrono::steady_clock::now();
        sum_pre += std::chrono::duration<double, std::milli>(t_pre_end - t_pre_begin).count();

        // ---- 2. 推理计时 ----
        auto t_inf_begin = std::chrono::steady_clock::now();
        std::vector<DetectBox> dets;
        if (prepOk) {
            dets = rknn_infer(rknn, modelRgb.data(), lb);
        }
        auto t_inf_end = std::chrono::steady_clock::now();
        sum_inf += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_begin).count();

        // ---- 3. 跟踪计时 ----
        auto t_track_begin = std::chrono::steady_clock::now();
        std::vector<Track> tracks = tracker.update(dets);
        auto t_track_end = std::chrono::steady_clock::now();
        sum_track += std::chrono::duration<double, std::milli>(t_track_end - t_track_begin).count();

        // ---- 4. 控制计时 ----
        auto t_ctrl_begin = std::chrono::steady_clock::now();
        controller.update(tracks, frames);
        auto t_ctrl_end = std::chrono::steady_clock::now();
        sum_ctrl += std::chrono::duration<double, std::milli>(t_ctrl_end - t_ctrl_begin).count();

        // 5. Count valid detections
        int fireCount = 0;
        int smokeCount = 0;

        for (const auto& t : tracks) {
            if (!t.is_activated || t.missed > 0) {
                continue;
            }

            if (t.cls == 0) {
                ++fireCount;
            } else {
                ++smokeCount;
            }
        }

        // 6. 计算端到端延迟（从帧开始到此刻）
        auto t_frame_end = std::chrono::steady_clock::now();
        sum_lat += std::chrono::duration<double, std::milli>(t_frame_end - t_frame_start).count();

        // 7. 每10帧计算一次 FPS 并记录性能数据
        if (frames % 10 == 0) {
            auto now = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(now - fpsTime).count();
            if (sec > 0.0) {
                fps = 30.0 / sec;
            }
            fpsTime = now;

            // 只有当这30帧都成功处理了（frames_since_log == 30）时才记录
            if (frames_since_log == 30) {
                cur_.fps_capture = fps;                       // 捕获帧率 ≈ 端到端帧率
                cur_.fps_end2end = fps;
                cur_.fps_infer = (sum_inf > 0.0) ? (1000.0 / (sum_inf / 30.0)) : 0.0;

                cur_.preprocess_ms = sum_pre / 30.0;
                cur_.infer_ms      = sum_inf / 30.0;
                cur_.track_ms      = sum_track / 30.0;
                cur_.control_ms    = sum_ctrl / 30.0;
                cur_.latency_ms    = sum_lat / 30.0;

                cur_.fire_count  = fireCount;
                cur_.smoke_count = smokeCount;
                cur_.tracks      = static_cast<int>(tracks.size());
                // id_switch_100f, CPU, Power 暂不填充（需额外实现）

                perf_.log(cur_);

                // 重置累加器
                sum_pre = sum_inf = sum_track = sum_ctrl = sum_lat = 0.0;
                frames_since_log = 0;
            }
        }

        // 8. NV12 -> full RGB frame for Qt display
        if (rga_nv12_to_rgb_full(nv12,
                                 CAM_W,
                                 CAM_H,
                                 displayRgb.data(),
                                 CAM_W,
                                 CAM_H)) {
            QImage img(displayRgb.data(),
                       CAM_W,
                       CAM_H,
                       CAM_W * 3,
                       QImage::Format_RGB888);

            QImage frame = img.copy();

            // 9. Draw high-quality overlay by QPainter
            drawDetectionOverlay(frame, tracks, fps);

            emit frameReady(frame);
        }

        cam.release_frame(bufIdx);

        emit statusReady(
            phaseToString(static_cast<int>(controller.get_phase())),
            controller.has_active_target()
                ? controller.get_active_target_id()
                : -1,
            controller.get_output_intensity(),
            fireCount,
            smokeCount,
            fps,
            "Running",
            controller.has_active_target() ? "ON" : "OFF"
        );
    }

    // 关闭日志文件，确保数据写入磁盘
    perf_.close();

    controller.shutdown();
    rknn_release_engine(rknn);
    cam.close();

    emit statusReady("IDLE",
                     -1,
                     0.0f,
                     0,
                     0,
                     fps,
                     "Stopped",
                     "OFF");

    emit logMessage("DetectionWorker stopped");
    emit finished();
}