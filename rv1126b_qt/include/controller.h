#pragma once
#include "detector.h"
#include "tracker.h"
#include <vector>
#include <string>

// ==================== PWM ?? (sysfs) ====================
class PwmController {
public:
    PwmController();
    ~PwmController();

    bool init(int chip = 0, int channel = 0,
              int period_ns = 50000);  // 20 kHz default

    void set_duty_cycle(float ratio);  // 0.0 ~ 1.0
    void enable(bool on);
    bool is_enabled() const { return enabled_; }
    float get_duty_cycle() const { return current_duty_; }

private:
    int chip_ = 0;
    int channel_ = 0;
    int period_ns_ = 50000;
    float current_duty_ = 0.0f;
    bool enabled_ = false;
    bool initialized_ = false;

    std::string pwm_path_;
    bool write_sysfs(const std::string& file, const std::string& value);
};

// ==================== GPIO ?? (sysfs) ====================
class GpioOutput {
public:
    GpioOutput();
    ~GpioOutput();

    bool init(int gpio_num);
    void set(bool high);

private:
    int gpio_ = -1;
    bool exported_ = false;
    std::string gpio_path_;
    bool write_sysfs(const std::string& file, const std::string& value);
};

// ==================== ????? ====================
enum class ControlPhase {
    IDLE,
    CONFIRMING,
    ACTIVE,
    RELEASING
};

struct ControlConfig {
    int confirm_frames = 5;      // ????????
    int release_frames = 15;     // ???????
    int ramp_up_frames = 10;     // PWM ????
    int ramp_down_frames = 20;   // PWM ????
    float intensity_min = 0.15f; // ?? PWM ??
    float intensity_max = 1.0f;  // ?? PWM ??
    float fire_boost = 1.2f;     // Fire ?????
    bool use_pwm = true;
    bool use_gpio = false;
    int pwm_chip = 0;
    int pwm_channel = 0;
    int pwm_period_ns = 50000;   // 20 kHz
    int gpio_pin = 0;            // GPIO number for pump on/off
};

// ==================== ????? (?? + PWM/GPIO ??) ====================
class FireController {
public:
    FireController();
    ~FireController();

    bool init(const ControlConfig& config = ControlConfig());
    void shutdown();

    // ???????????
    // ??????????? PWM/GPIO
    void update(const std::vector<Track>& tracks, int frame);

    // ??????
    ControlPhase get_phase() const { return phase_; }
    float get_target_intensity() const { return target_intensity_; }
    float get_output_intensity() const { return output_intensity_; }
    int get_active_target_id() const { return active_target_id_; }
    bool has_active_target() const { return active_target_id_ >= 0; }

private:
    ControlConfig cfg_;
    ControlPhase phase_ = ControlPhase::IDLE;

    // ?????
    int confirm_counter_ = 0;
    int release_counter_ = 0;
    int ramp_counter_ = 0;

    // ????
    int active_target_id_ = -1;
    float target_intensity_ = 0.0f;  // ??????
    float output_intensity_ = 0.0f;  // ???????????

    // ????
    PwmController pwm_;
    GpioOutput gpio_;

    // ????????????
    int select_best_target(const std::vector<Track>& tracks) const;

    // ??????
    float calc_intensity(const Track& track) const;
};
