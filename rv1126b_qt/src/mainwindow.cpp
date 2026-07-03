#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDateTime>
#include <QGraphicsPixmapItem>
#include <QMetaObject>
#include <QPixmap>
#include <QPushButton>
#include <QStatusBar>
#include <QCoreApplication>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , scene_(new QGraphicsScene(this))
{
    ui->setupUi(this);

    modelPath_ = QDir(QCoreApplication::applicationDirPath())
                     .absoluteFilePath("../model/best_nano_111_rv1126b_hybrid.rknn");

    ui->graphicsView_2->setScene(scene_);
    ui->graphicsView_2->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView_2->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    setupUiState();

    connect(ui->pushButton_2, &QPushButton::clicked,
            this, &MainWindow::startDetection);
    connect(ui->pushButton_7, &QPushButton::clicked,
            this, &MainWindow::stopDetection);
    connect(ui->pushButton_6, &QPushButton::clicked,
            this, &MainWindow::stopDetection);

    connect(&clockTimer_, &QTimer::timeout,
            this, &MainWindow::updateClock);
    clockTimer_.start(1000);
    updateClock();

    runTimer_.start();
    startDht11();
}

MainWindow::~MainWindow()
{
    stopDetection();
    stopDht11();
    delete ui;
}

void MainWindow::setupUiState()
{
    setWindowTitle("Fire & Smoke Detection System");

    ui->label_32->setText("IDLE");
    ui->label_34->setText("00:00:00");
    ui->label_36->setText("None");
    ui->label_38->setText("0 %");
    ui->label_40->setText("Stopped");
    ui->label_42->setText("OFF");

    ui->label_44->setText("0");
    ui->label_46->setText("0");

    ui->label_48->setText(QString("Model: %1").arg(QFileInfo(modelPath_).fileName()));
    ui->label_49->setText("Resolution: 1920 x 1080");
    ui->label_51->setText("Disconnected");

    ui->label_53->setText("Temperature / Humidity");
    ui->label_54->setText("-- °C / -- %RH");

    ui->label_18->setText("FPS: 0.0");

    ui->horizontalSlider_2->setRange(0, 100);
    ui->horizontalSlider_2->setValue(0);

    ui->pushButton_7->setEnabled(false);
    ui->pushButton_8->setEnabled(false);
}

void MainWindow::startDetection()
{
    if (running_) return;

    running_ = true;
    runTimer_.restart();

    ui->pushButton_2->setEnabled(false);
    ui->pushButton_7->setEnabled(true);

    ui->label_40->setText("Starting");
    ui->label_51->setText("Connecting");

    workerThread_ = new QThread(this);
    worker_ = new DetectionWorker();

    worker_->moveToThread(workerThread_);

    connect(workerThread_, &QThread::finished,
            worker_, &QObject::deleteLater);

    connect(workerThread_, &QThread::started, this, [this]() {
        QMetaObject::invokeMethod(worker_, "startWork",
                                  Qt::QueuedConnection,
                                  Q_ARG(QString, cameraDev_),
                                  Q_ARG(QString, modelPath_));
    });

    connect(worker_, &DetectionWorker::frameReady,
            this, &MainWindow::onFrameReady);

    connect(worker_, &DetectionWorker::statusReady,
            this, &MainWindow::onStatusReady);

    connect(worker_, &DetectionWorker::logMessage,
            this, &MainWindow::appendLog);

    connect(worker_, &DetectionWorker::finished,
            this, &MainWindow::onWorkerFinished);

    workerThread_->start();
}

void MainWindow::stopDetection()
{
    if (!workerThread_) return;

    if (worker_) {
        worker_->stopWork();
    }

    workerThread_->quit();
    workerThread_->wait(3000);

    if (workerThread_->isRunning()) {
        workerThread_->terminate();
        workerThread_->wait();
    }

    workerThread_->deleteLater();
    workerThread_ = nullptr;
    worker_ = nullptr;

    running_ = false;

    ui->pushButton_2->setEnabled(true);
    ui->pushButton_7->setEnabled(false);

    ui->label_32->setText("IDLE");
    ui->label_36->setText("None");
    ui->label_38->setText("0 %");
    ui->label_40->setText("Stopped");
    ui->label_42->setText("OFF");
    ui->label_51->setText("Disconnected");
    ui->horizontalSlider_2->setValue(0);
}

void MainWindow::onWorkerFinished()
{
    if (workerThread_) {
        workerThread_->quit();
    }

    running_ = false;
    ui->pushButton_2->setEnabled(true);
    ui->pushButton_7->setEnabled(false);
    ui->label_40->setText("Stopped");
    ui->label_51->setText("Disconnected");
}

void MainWindow::onFrameReady(const QImage& image)
{
    scene_->clear();

    QPixmap pixmap = QPixmap::fromImage(image);
    QGraphicsPixmapItem *item = scene_->addPixmap(pixmap);
    item->setTransformationMode(Qt::SmoothTransformation);

    scene_->setSceneRect(pixmap.rect());
    ui->graphicsView_2->fitInView(scene_->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::onStatusReady(const QString& phase,
                               int targetId,
                               float outputIntensity,
                               int fireCount,
                               int smokeCount,
                               double fps,
                               const QString& runningState,
                               const QString& gpioState)
{
    QString phaseText = phase;
    QString stateText = runningState;
    QString gpioText = gpioState;

    if (phaseText == "运行中") phaseText = "ACTIVE";
    if (phaseText == "空闲") phaseText = "IDLE";
    if (phaseText == "确认中") phaseText = "CONFIRMING";
    if (phaseText == "释放中") phaseText = "RELEASING";

    if (stateText == "运行中") stateText = "Running";
    if (stateText == "已停止") stateText = "Stopped";
    if (stateText == "启动中") stateText = "Starting";

    if (gpioText == "开") gpioText = "ON";
    if (gpioText == "关") gpioText = "OFF";

    ui->label_32->setText(phaseText);

    if (targetId >= 0) {
        ui->label_36->setText(QString("ID: %1").arg(targetId));
    } else {
        ui->label_36->setText("None");
    }

    ui->label_38->setText(QString("%1 %")
                          .arg(outputIntensity * 100.0f, 0, 'f', 1));

    ui->label_40->setText(stateText);
    ui->label_42->setText(gpioText);

    ui->label_44->setText(QString::number(fireCount));
    ui->label_46->setText(QString::number(smokeCount));

    ui->label_18->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));

    int pwmValue = static_cast<int>(outputIntensity * 100.0f);
    pwmValue = qBound(0, pwmValue, 100);
    ui->horizontalSlider_2->setValue(pwmValue);

    ui->label_51->setText(stateText == "Running" ? "Connected" : "Disconnected");
    ui->label_34->setText(formatRunTime());
}

void MainWindow::updateClock()
{
    ui->label_30->setText("System Time: " +
                          QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));

    if (running_) {
        ui->label_34->setText(formatRunTime());
    }
}

QString MainWindow::formatRunTime() const
{
    qint64 sec = runTimer_.elapsed() / 1000;

    int h = static_cast<int>(sec / 3600);
    int m = static_cast<int>((sec % 3600) / 60);
    int s = static_cast<int>(sec % 60);

    return QString("%1:%2:%3")
        .arg(h, 2, 10, QChar('0'))
        .arg(m, 2, 10, QChar('0'))
        .arg(s, 2, 10, QChar('0'));
}

void MainWindow::appendLog(const QString& msg)
{
    statusBar()->showMessage(msg, 5000);
}

void MainWindow::startDht11()
{
    if (dhtThread_) return;

    dhtThread_ = new QThread(this);

    // D21 -> GPIO5_C5 -> gpiochip5 line 21
    dhtReader_ = new Dht11Reader("/dev/gpiochip5", 21);

    dhtReader_->moveToThread(dhtThread_);

    connect(dhtThread_, &QThread::started,
            dhtReader_, &Dht11Reader::start);

    connect(dhtReader_, &Dht11Reader::dataReady,
            this, &MainWindow::onDht11Data);

    connect(dhtReader_, &Dht11Reader::errorMessage,
            this, &MainWindow::onDht11Error);

    connect(dhtThread_, &QThread::finished,
            dhtReader_, &QObject::deleteLater);

    dhtThread_->start();
}

void MainWindow::stopDht11()
{
    if (!dhtThread_) return;

    if (dhtReader_) {
        QMetaObject::invokeMethod(dhtReader_,
                                  "stop",
                                  Qt::BlockingQueuedConnection);
    }

    dhtThread_->quit();
    dhtThread_->wait();

    dhtThread_->deleteLater();
    dhtThread_ = nullptr;
    dhtReader_ = nullptr;
}

void MainWindow::onDht11Data(float temperature, float humidity)
{
    ui->label_53->setText("Temperature / Humidity");

    ui->label_54->setText(QString("%1 °C / %2 %RH")
                          .arg(temperature, 0, 'f', 1)
                          .arg(humidity, 0, 'f', 1));

    ui->label_51->setText("Connected");
}

void MainWindow::onDht11Error(const QString& msg)
{
    ui->label_54->setText("-- °C / -- %RH");
    statusBar()->showMessage(msg, 3000);
}