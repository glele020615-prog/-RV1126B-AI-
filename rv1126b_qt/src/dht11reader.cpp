#include "dht11reader.h"

#include <gpiod.h>
#include <QThread>
#include <QElapsedTimer>
#include <unistd.h>
#include <vector>
#include <cstdio>

static inline long long nowUs()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

Dht11Reader::Dht11Reader(QString chipPath,
                         unsigned int lineOffset,
                         QObject *parent)
    : QObject(parent),
      chipPath_(std::move(chipPath)),
      lineOffset_(lineOffset)
{
}

void Dht11Reader::start()
{
    running_ = true;

    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &Dht11Reader::readOnce);
    timer->start(2000);

    readOnce();
}

void Dht11Reader::stop()
{
    running_ = false;

    for (QTimer *t : findChildren<QTimer*>()) {
        t->stop();
    }
}

void Dht11Reader::readOnce()
{
    if (!running_) return;

    float t = 0.0f;
    float h = 0.0f;

    if (readDht11(t, h)) {
        emit dataReady(t, h);
    } else {
        emit errorMessage("DHT11 read failed");
    }
}

bool Dht11Reader::readDht11(float& temperature, float& humidity)
{
    gpiod_chip* chip = gpiod_chip_open(chipPath_.toUtf8().constData());
    if (!chip) return false;

    gpiod_line* line = gpiod_chip_get_line(chip, lineOffset_);
    if (!line) {
        gpiod_chip_close(chip);
        return false;
    }

    // =========================
    // 1. 发送起始信号
    // =========================
    if (gpiod_line_request_output(line, "dht11", 1) < 0) {
        gpiod_chip_close(chip);
        return false;
    }

    gpiod_line_set_value(line, 0);
    usleep(20000); // 20ms start signal

    gpiod_line_set_value(line, 1);
    usleep(40);

    gpiod_line_release(line);

    // =========================
    // 2. 切换为输入读取
    // =========================
    line = gpiod_chip_get_line(chip, lineOffset_);
    if (!line) {
        gpiod_chip_close(chip);
        return false;
    }

    if (gpiod_line_request_input(line, "dht11") < 0) {
        gpiod_chip_close(chip);
        return false;
    }

    // =========================
    // 3. 等待DHT11响应
    // =========================
    auto waitLevel = [&](int level, int timeoutUs) -> bool {
        long long start = nowUs();
        while ((nowUs() - start) < timeoutUs) {
            if (gpiod_line_get_value(line) == level)
                return true;
        }
        return false;
    };

    // DHT11 response: 80us LOW + 80us HIGH
    if (!waitLevel(0, 100)) {
        gpiod_line_release(line);
        gpiod_chip_close(chip);
        return false;
    }

    if (!waitLevel(1, 100)) {
        gpiod_line_release(line);
        gpiod_chip_close(chip);
        return false;
    }

    // =========================
    // 4. 读取40bit数据
    // =========================
    std::vector<int> bits;
    bits.reserve(40);

    for (int i = 0; i < 40; ++i) {

        // 等待 LOW start (50us)
        long long t0 = nowUs();
        while (gpiod_line_get_value(line) == 1) {
            if (nowUs() - t0 > 100) break;
        }

        // 等待 HIGH start
        t0 = nowUs();
        while (gpiod_line_get_value(line) == 0) {
            if (nowUs() - t0 > 100) break;
        }

        long long rise = nowUs();

        // 测量 HIGH duration
        while (gpiod_line_get_value(line) == 1) {
            if (nowUs() - rise > 200) break;
        }

        long long width = nowUs() - rise;

        // DHT11: 26-28us = 0, ~70us = 1
        bits.push_back(width > 50 ? 1 : 0);
    }

    gpiod_line_release(line);
    gpiod_chip_close(chip);

    if (bits.size() < 40) {
        printf("bit capture failed\n");
        return false;
    }

    // =========================
    // 5. decode
    // =========================
    unsigned char data[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < 40; ++i) {
        data[i / 8] <<= 1;
        data[i / 8] |= bits[i];
    }

    unsigned char checksum =
        data[0] + data[1] + data[2] + data[3];

    if (checksum != data[4]) {
        //printf("checksum error: %u vs %u\n", checksum, data[4]);
        return false;
    }

    humidity = data[0] + data[1] * 0.1f;
    temperature = data[2] + data[3] * 0.1f;

    printf("DHT11 OK: T=%.1f H=%.1f\n", temperature, humidity);
    return true;
}