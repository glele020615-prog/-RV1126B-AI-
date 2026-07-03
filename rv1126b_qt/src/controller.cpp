#include "controller.h"
#include "camera_v4l2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <fcntl.h>

// ===================== PwmController =====================
PwmController::PwmController() {}
PwmController::~PwmController() { if (enabled_) enable(false); }

bool PwmController::init(int chip, int channel, int period_ns) {
    chip_ = chip;
    channel_ = channel;
    period_ns_ = period_ns;
    initialized_ = false;

    // Export PWM channel
    char buf[128];
    snprintf(buf, sizeof(buf), "/sys/class/pwm/pwmchip%d/export", chip);
    if (!write_sysfs(buf, std::to_string(channel))) {
        // Might already be exported; continue
    }

    usleep(10000);  // 10 ms for device to appear

    // Set path
    snprintf(buf, sizeof(buf), "/sys/class/pwm/pwmchip%d/pwm%d", chip, channel);
    pwm_path_ = buf;

    // Set period
    snprintf(buf, sizeof(buf), "%s/period", pwm_path_.c_str());
    if (!write_sysfs(buf, std::to_string(period_ns))) {
        fprintf(stderr, "PWM: failed to set period\n");
        return false;
    }

    // Set initial duty to 0
    snprintf(buf, sizeof(buf), "%s/duty_cycle", pwm_path_.c_str());
    write_sysfs(buf, "0");

    current_duty_ = 0.0f;
    initialized_ = true;
    printf("PWM init: chip%d channel%d period=%dns\n", chip, channel, period_ns);
    return true;
}

void PwmController::set_duty_cycle(float ratio) {
    if (!initialized_) return;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    current_duty_ = ratio;

    int duty_ns = (int)(period_ns_ * ratio + 0.5f);
    char buf[256];
    snprintf(buf, sizeof(buf), "%s/duty_cycle", pwm_path_.c_str());
    write_sysfs(buf, std::to_string(duty_ns));
}

void PwmController::enable(bool on) {
    if (!initialized_) return;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s/enable", pwm_path_.c_str());
    if (write_sysfs(buf, on ? "1" : "0")) {
        enabled_ = on;
        if (!on) {
            // Reset duty to 0 when disabling
            set_duty_cycle(0.0f);
        }
    }
}

bool PwmController::write_sysfs(const std::string& file, const std::string& value) {
    int fd = open(file.c_str(), O_WRONLY);
    if (fd < 0) return false;
    bool ok = (write(fd, value.c_str(), value.size()) == (ssize_t)value.size());
    close(fd);
    return ok;
}

// ===================== GpioOutput =====================
GpioOutput::GpioOutput() {}
GpioOutput::~GpioOutput() { set(false); }

bool GpioOutput::init(int gpio_num) {
    gpio_ = gpio_num;

    // Export
    char buf[128];
    snprintf(buf, sizeof(buf), "/sys/class/gpio/export");
    write_sysfs(buf, std::to_string(gpio_));
    usleep(50000);  // 50 ms for device to appear

    // Direction
    snprintf(buf, sizeof(buf), "/sys/class/gpio/gpio%d/direction", gpio_);
    if (!write_sysfs(buf, "out")) {
        fprintf(stderr, "GPIO: direction out failed\n");
        return false;
    }

    snprintf(buf, sizeof(buf), "/sys/class/gpio/gpio%d/value", gpio_);
    gpio_path_ = buf;

    set(false);
    exported_ = true;
    printf("GPIO init: pin%d\n", gpio_);
    return true;
}

void GpioOutput::set(bool high) {
    if (gpio_ < 0) return;
    write_sysfs(gpio_path_, high ? "1" : "0");
}

bool GpioOutput::write_sysfs(const std::string& file, const std::string& value) {
    int fd = open(file.c_str(), O_WRONLY);
    if (fd < 0) return false;
    bool ok = (write(fd, value.c_str(), value.size()) == (ssize_t)value.size());
    close(fd);
    return ok;
}

// ===================== FireController =====================
FireController::FireController() {}
FireController::~FireController() { shutdown(); }

bool FireController::init(const ControlConfig& config) {
    cfg_ = config;
    phase_ = ControlPhase::IDLE;
    confirm_counter_ = 0;
    release_counter_ = 0;
    ramp_counter_ = 0;
    active_target_id_ = -1;
    target_intensity_ = 0.0f;
    output_intensity_ = 0.0f;

    if (cfg_.use_pwm) {
        if (!pwm_.init(cfg_.pwm_chip, cfg_.pwm_channel, cfg_.pwm_period_ns)) {
            fprintf(stderr, "PWM init failed; disabling\n");
            cfg_.use_pwm = false;
        }
    }

    if (cfg_.use_gpio) {
        if (!gpio_.init(cfg_.gpio_pin)) {
            fprintf(stderr, "GPIO init failed; disabling\n");
            cfg_.use_gpio = false;
        }
    }

    printf("FireController init: confirm=%d release=%d ramp_up=%d ramp_down=%d\n",
           cfg_.confirm_frames, cfg_.release_frames,
           cfg_.ramp_up_frames, cfg_.ramp_down_frames);
    printf("  PWM=%s GPIO=%s\n",
           cfg_.use_pwm ? "enabled" : "disabled",
           cfg_.use_gpio ? "enabled" : "disabled");
    return true;
}

void FireController::shutdown() {
    if (cfg_.use_pwm) {
        pwm_.enable(false);
    }
    if (cfg_.use_gpio) {
        gpio_.set(false);
    }
}

void FireController::update(const std::vector<Track>& tracks, int frame) {
    (void)frame;

    // Step 1: Select best target
    int best_id = select_best_target(tracks);
    float intensity = 0.0f;

    if (best_id >= 0) {
        // Find the track
        for (const auto& t : tracks) {
            if (t.track_id == best_id) {
                intensity = calc_intensity(t);
                break;
            }
        }
    }

    // Step 2: Anti-shake state machine
    switch (phase_) {
    case ControlPhase::IDLE:
        if (best_id >= 0) {
            confirm_counter_++;
            if (confirm_counter_ >= cfg_.confirm_frames) {
                phase_ = ControlPhase::CONFIRMING;
                confirm_counter_ = 0;
                active_target_id_ = best_id;
                target_intensity_ = intensity;
                ramp_counter_ = 0;
                printf("CONTROL: IDLE -> CONFIRMING (target=%d, intensity=%.3f)\n",
                       best_id, intensity);
            }
        } else {
            confirm_counter_ = 0;
        }
        break;

    case ControlPhase::CONFIRMING:
        if (best_id >= 0) {
            active_target_id_ = best_id;
            target_intensity_ = intensity;
            ramp_counter_++;

            // Ramp up smoothly
            float progress = std::min(1.0f, (float)ramp_counter_ / cfg_.ramp_up_frames);
            output_intensity_ = target_intensity_ * progress;

            if (ramp_counter_ >= cfg_.ramp_up_frames) {
                phase_ = ControlPhase::ACTIVE;
                printf("CONTROL: CONFIRMING -> ACTIVE (target=%d, output=%.3f)\n",
                       active_target_id_, output_intensity_);
            }
        } else {
            // Lost target during ramp-up, go back to IDLE
            phase_ = ControlPhase::IDLE;
            confirm_counter_ = 0;
            ramp_counter_ = 0;
            target_intensity_ = 0.0f;
            output_intensity_ = 0.0f;
            active_target_id_ = -1;
            printf("CONTROL: CONFIRMING -> IDLE (target lost)\n");
        }
        break;

    case ControlPhase::ACTIVE:
        if (best_id >= 0) {
            active_target_id_ = best_id;
            target_intensity_ = intensity;
            release_counter_ = 0;

            // Update output with slight smoothing
            output_intensity_ = output_intensity_ * 0.7f + target_intensity_ * 0.3f;
        } else {
            release_counter_++;
            if (release_counter_ >= cfg_.release_frames) {
                phase_ = ControlPhase::RELEASING;
                release_counter_ = 0;
                ramp_counter_ = 0;
                printf("CONTROL: ACTIVE -> RELEASING (target lost)\n");
            } else {
                // While waiting, gradually decrease
                float decay = 1.0f - (float)release_counter_ / cfg_.release_frames;
                output_intensity_ *= std::max(0.1f, decay);
            }
        }
        break;

    case ControlPhase::RELEASING:
        if (best_id >= 0) {
            // Target reappeared, go back to ACTIVE
            active_target_id_ = best_id;
            target_intensity_ = intensity;
            phase_ = ControlPhase::ACTIVE;
            release_counter_ = 0;
            printf("CONTROL: RELEASING -> ACTIVE (target=%d)\n", best_id);
        } else {
            ramp_counter_++;
            float progress = std::min(1.0f, (float)ramp_counter_ / cfg_.ramp_down_frames);
            output_intensity_ = target_intensity_ * std::max(0.0f, 1.0f - progress);

            if (ramp_counter_ >= cfg_.ramp_down_frames) {
                phase_ = ControlPhase::IDLE;
                confirm_counter_ = 0;
                ramp_counter_ = 0;
                output_intensity_ = 0.0f;
                target_intensity_ = 0.0f;
                active_target_id_ = -1;
                printf("CONTROL: RELEASING -> IDLE\n");
            }
        }
        break;
    }

    // Step 3: Apply to hardware
    if (cfg_.use_pwm) {
        pwm_.set_duty_cycle(output_intensity_);
        if (output_intensity_ > 0.01f && !pwm_.is_enabled()) {
            pwm_.enable(true);
        } else if (output_intensity_ <= 0.01f && pwm_.is_enabled()) {
            pwm_.enable(false);
        }
    }

    if (cfg_.use_gpio) {
        gpio_.set(output_intensity_ > 0.01f);
    }
}

int FireController::select_best_target(const std::vector<Track>& tracks) const {
    if (tracks.empty()) return -1;

    int best_id = -1;
    float best_score = 0;

    for (const auto& t : tracks) {
        // Average confidence from history
        float avg_conf = 0;
        if (!t.conf_history.empty()) {
            for (float c : t.conf_history) avg_conf += c;
            avg_conf /= t.conf_history.size();
        }

        float w = t.x2 - t.x1;
        float h = t.y2 - t.y1;
        float area = w * h;

        // Fire > Smoke priority
        float cls_bonus = (t.cls == 0) ? 1.3f : 1.0f;

        // Score: confidence * class_bonus * area_factor (prefer larger targets)
        float area_factor = std::min(1.0f, area / (CAM_W * CAM_H * 0.5f));
        float score = avg_conf * cls_bonus * (0.7f + 0.3f * area_factor);

        // Boost currently active target to reduce jitter
        if (t.track_id == active_target_id_) {
            score *= 1.5f;
        }

        if (score > best_score) {
            best_score = score;
            best_id = t.track_id;
        }
    }

    return best_id;
}

float FireController::calc_intensity(const Track& track) const {
    // Average confidence from history
    float avg_conf = 0;
    if (!track.conf_history.empty()) {
        for (float c : track.conf_history) avg_conf += c;
        avg_conf /= track.conf_history.size();
    }

    float w = track.x2 - track.x1;
    float h = track.y2 - track.y1;
    float area = w * h;
    float area_ratio = area / (CAM_W * CAM_H);  // 0~1

    // Base intensity from confidence
    float intensity = avg_conf;

    // Boost by area (larger fires need more water)
    intensity *= (1.0f + area_ratio * 0.5f);

    // Fire class boost
    if (track.cls == 0) {
        intensity *= cfg_.fire_boost;
    }

    // Clamp
    intensity = std::max(0.0f, std::min(1.0f, intensity));

    // Apply min/max constraints
    if (intensity > 0 && intensity < cfg_.intensity_min)
        intensity = cfg_.intensity_min;
    if (intensity > cfg_.intensity_max)
        intensity = cfg_.intensity_max;

    return intensity;
}
