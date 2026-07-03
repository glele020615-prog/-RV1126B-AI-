#include "perf_monitor.h"
#include <ctime>

void PerfMonitor::initCSV(const std::string& path) {
    file_.open(path);
    file_ << "time,fps_cap,fps_inf,fps_e2e,"
             "pre_ms,inf_ms,track_ms,ctrl_ms,"
             "latency,fire,smoke,tracks,idsw,CPU,Power\n";
}

void PerfMonitor::log(const PerfSample& s) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!file_.is_open()) return;

    file_ << time(nullptr) << ","
          << s.fps_capture << ","
          << s.fps_infer << ","
          << s.fps_end2end << ","
          << s.preprocess_ms << ","
          << s.infer_ms << ","
          << s.track_ms << ","
          << s.control_ms << ","
          << s.latency_ms << ","
          << s.fire_count << ","
          << s.smoke_count << ","
          << s.tracks << ","
          << s.id_switch_100f << ","
          << s.cpu << ","
          << s.power << "\n";
}

void PerfMonitor::close() {
    if (file_.is_open()) file_.close();
}