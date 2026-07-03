#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QGraphicsScene>
#include <QImage>
#include <QMainWindow>
#include <QThread>
#include <QTimer>

#include "detectionworker.h"
#include <QThread>
#include <QElapsedTimer>
#include "dht11reader.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void startDetection();
    void stopDetection();
    void onFrameReady(const QImage& image);
    void onStatusReady(const QString& phase,
                       int targetId,
                       float outputIntensity,
                       int fireCount,
                       int smokeCount,
                       double fps,
                       const QString& runningState,
                       const QString& gpioState);
    void updateClock();
    void onWorkerFinished();

private:
    void setupUiState();
    void appendLog(const QString& msg);

private:
    Ui::MainWindow *ui = nullptr;
    QGraphicsScene *scene_ = nullptr;
    QThread *workerThread_ = nullptr;
    DetectionWorker *worker_ = nullptr;
    QTimer clockTimer_;
    bool running_ = false;

    QString cameraDev_ = "/dev/video23";
    QString modelPath_ = "best_nano_111_rv1126b_hybrid.rknn";


private slots:
    void onDht11Data(float temperature, float humidity);
    void onDht11Error(const QString& msg);

private:
    void startDht11();
    void stopDht11();
    QString formatRunTime() const;

private:
    QThread* dhtThread_ = nullptr;
    Dht11Reader* dhtReader_ = nullptr;
    QElapsedTimer runTimer_;
};

#endif // MAINWINDOW_H
