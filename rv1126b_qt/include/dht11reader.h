#ifndef DHT11READER_H
#define DHT11READER_H

#include <QObject>
#include <QTimer>

class Dht11Reader : public QObject
{
    Q_OBJECT

public:
    explicit Dht11Reader(QString chipPath,
                         unsigned int lineOffset,
                         QObject *parent = nullptr);

public slots:
    void start();
    void stop();

signals:
    void dataReady(float temperature, float humidity);
    void errorMessage(const QString& msg);

private slots:
    void readOnce();

private:
    bool readDht11(float& temperature, float& humidity);

private:
    QString chipPath_;
    unsigned int lineOffset_;
    // QTimer timer_;
    bool running_ = false;
};

#endif