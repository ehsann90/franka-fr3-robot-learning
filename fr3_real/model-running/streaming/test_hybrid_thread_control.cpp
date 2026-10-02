#include "hybrid_thread_control.h"

#include <cassert>
#include <cmath>
#include <string>

int main() {
  using namespace hybrid_thread;
  assert(std::abs(proportional_force_command_target(0.0) - 0.30009) < 1e-12);
  assert(std::abs(proportional_force_command_target(kMeasuredAxialTargetN)) < 1e-12);
  assert(std::abs(slew_limited_force_command(0.0, 0.3, 0.1) - 0.05) < 1e-12);
  assert(std::abs(tracking_progress_scale(0.0, 0.0) - 1.0) < 1e-12);
  assert(std::abs(tracking_progress_scale(kPauseLateralErrorM, 0.0)) < 1e-12);
  assert(tracking_progress_scale(0.000175, 0.0) > 0.49);
  assert(tracking_progress_scale(0.000175, 0.0) < 0.51);
  const Eigen::Vector3d axis(0.0, 0.0, -1.0);
  assert((lateral_component({1.0, 2.0, 3.0}, axis) - Eigen::Vector3d(1.0, 2.0, 0.0)).norm() < 1e-12);
  const Eigen::Matrix3d desired =
      Eigen::AngleAxisd(5.0 * kPi / 180.0, Eigen::Vector3d::UnitZ())
          .toRotationMatrix();
  const Eigen::Vector3d rotational_error =
      orientation_error(desired, Eigen::Matrix3d::Identity());
  assert(std::abs(orientation_error_angle_rad(rotational_error) -
                  5.0 * kPi / 180.0) < 1e-12);
  Guard nominal;
  nominal.baseline_fz_N = -1.0;
  nominal.axis = axis;
  nominal.dynamic_model_min_q7_rad = 0.0;
  nominal.dynamic_model_max_q7_rad = 2.0;
  std::array<double, 7> q{}; q[6] = 1.0;
  nominal.update(q, 0.0, -0.1, {0, 0, -0.0002}, 0.0001, 0.01);
  assert(nominal.stop_reason == nullptr);
  Guard lateral = nominal; lateral.stop_reason = nullptr;
  lateral.update(q, 0.0, -0.1, {0, 0, 0}, 0.00031, 0.01);
  assert(std::string(lateral.stop_reason) == "maximum_lateral_tracking_error");
  return 0;
}
