#ifndef DETECTIONWORKER_H
#define DETECTIONWORKER_H

#include <QObject>
#include <QImage>
#include <QString>
#include <QMutex>
#include "perf_monitor.h"


class DetectionWorker : public QObject
{
    Q_OBJECT
public:
    explicit DetectionWorker(QObject *parent = nullptr);
    ~DetectionWorker() override;

public slots:
    void startWork(const QString& cameraDev, const QString& modelPath);
    void stopWork();

signals:
    void frameReady(const QImage& image);
    void statusReady(const QString& phase,
                     int targetId,
                     float outputIntensity,
                     int fireCount,
                     int smokeCount,
                     double fps,
                     const QString& runningState,
                     const QString& gpioState);
    void logMessage(const QString& msg);
    void finished();

private:
    bool isStopRequested();
    QString phaseToString(int phase) const;

private:
    QMutex mutex_;
    bool stopRequested_ = false;

private:
    PerfMonitor perf_;
    PerfSample cur_;
};

#endif // DETECTIONWORKER_H
