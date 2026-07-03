#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <mutex>

struct PerfSample {
    double fps_capture = 0;
    double fps_infer = 0;
    double fps_end2end = 0;

    double preprocess_ms = 0;
    double infer_ms = 0;
    double track_ms = 0;
    double control_ms = 0;

    double latency_ms = 0;

    int fire_count = 0;
    int smoke_count = 0;
    int tracks = 0;
    int id_switch_100f = 0;

    double cpu = 0;
    double power = 0;
};

class PerfMonitor {
public:
    void initCSV(const std::string& path);
    void log(const PerfSample& s);
    void close();

private:
    std::ofstream file_;
    std::mutex mtx_;
};