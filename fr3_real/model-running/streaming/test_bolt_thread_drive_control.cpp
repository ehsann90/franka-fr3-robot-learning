#include "bolt_thread_drive_control.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

int main() {
  using namespace bolt_thread_drive;
  assert(std::abs(drive_angle(0.0)) < 1e-12);
  assert(std::abs(drive_angle(kDriveDurationS) - kDriveAngleRad) < 1e-12);
  assert(std::abs(drive_angular_speed(0.5 * kDriveDurationS) -
                  kPeakAngularSpeedRadS) < 1e-12);
  assert(std::abs(force_command(1.0) - 0.5) < 1e-12);
  assert(std::abs(force_command(4.0) - 1.5) < 1e-12);
  assert(std::abs(interpolate({0.0, 1.0}, {0.0, 2.0}, 0.25) - 0.5) < 1e-12);

  Guard nominal;
  nominal.baseline_fz_N = -1.0;
  nominal.axis = Eigen::Vector3d(0.0, 0.0, -1.0);
  nominal.dynamic_model_min_q7_rad = 0.0;
  nominal.dynamic_model_max_q7_rad = 2.0;
  nominal.target_q7_rad = 1.6;
  nominal.start_q.fill(0.0);
  std::array<double, 7> q{};
  q[6] = 0.5;
  nominal.update(1.0, q, 0.0, -0.1, {0.0, 0.0, -0.0004}, nominal.axis);
  assert(nominal.stop_reason == nullptr);

  Guard torque_stop = nominal;
  torque_stop.stop_reason = nullptr;
  torque_stop.update(1.1, q, 0.0, -0.6, {0.0, 0.0, 0.0}, nominal.axis);
  assert(std::string(torque_stop.stop_reason) == "hard_residual_torque_cutoff");

  Guard completed = nominal;
  completed.stop_reason = nullptr;
  completed.update(kDriveDurationS, q, 0.0, -0.1, {0.0, 0.0, 0.0}, nominal.axis);
  assert(std::string(completed.stop_reason) == "drive_window_complete");
  return 0;
}
