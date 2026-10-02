#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace hybrid_thread {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMeasuredAxialTargetN = 1.200360;
constexpr double kProportionalGain = 0.25;
constexpr double kForceCommandCeilingN = 1.5;
constexpr double kForceCommandSlewNs = 0.5;
constexpr double kForceAcquisitionDurationS = 3.0;
constexpr double kHardAbsoluteAxialDeltaN = 2.0;
constexpr double kHardNegativeResidualTzNm = -0.5383051567216975;
constexpr double kMaximumAxialTravelM = 0.003;
constexpr double kMaximumLateralTrackingErrorM = 0.0003;
constexpr double kMaximumOrientationTrackingErrorRad = 5.0 * kPi / 180.0;
constexpr double kFullSpeedLateralErrorM = 0.00010;
constexpr double kPauseLateralErrorM = 0.00025;
constexpr double kFullSpeedOrientationErrorRad = 1.0 * kPi / 180.0;
constexpr double kPauseOrientationErrorRad = 4.0 * kPi / 180.0;
constexpr double kMaximumTotalControlDurationS = 90.0;
constexpr double kLateralStiffnessNm = 2000.0;
constexpr double kLateralDampingNsM = 2.0 * std::sqrt(kLateralStiffnessNm);
constexpr double kRotationalStiffnessNmRad = 200.0;
constexpr double kRotationalDampingNmsRad = 2.0 * std::sqrt(kRotationalStiffnessNmRad);
constexpr double kQ7CableMinRad = -0.15688785910606384;
constexpr double kQ7CableMaxRad = 2.998908281326294;
constexpr double kDynamicModelLowAllowanceRad = 0.02;

inline double proportional_force_command_target(double measured_axial_delta_N) {
  return std::clamp(
      kProportionalGain * (kMeasuredAxialTargetN - measured_axial_delta_N),
      0.0, kForceCommandCeilingN);
}

inline double slew_limited_force_command(double previous_command_N,
                                         double target_command_N,
                                         double period_s) {
  const double maximum_step = kForceCommandSlewNs * std::max(0.0, period_s);
  return previous_command_N + std::clamp(
      target_command_N - previous_command_N, -maximum_step, maximum_step);
}

inline double tracking_progress_scale(double lateral_error_m,
                                      double orientation_error_rad) {
  const auto scale = [](double error, double full_speed, double pause) {
    if (error <= full_speed) return 1.0;
    if (error >= pause) return 0.0;
    return (pause - error) / (pause - full_speed);
  };
  return std::min(
      scale(lateral_error_m, kFullSpeedLateralErrorM, kPauseLateralErrorM),
      scale(orientation_error_rad, kFullSpeedOrientationErrorRad,
            kPauseOrientationErrorRad));
}

inline Eigen::Vector3d lateral_component(const Eigen::Vector3d& vector,
                                         const Eigen::Vector3d& axis) {
  return vector - axis * axis.dot(vector);
}

inline Eigen::Vector3d orientation_error(const Eigen::Matrix3d& desired,
                                         const Eigen::Matrix3d& measured) {
  Eigen::Quaterniond desired_q(desired);
  Eigen::Quaterniond measured_q(measured);
  if (desired_q.coeffs().dot(measured_q.coeffs()) < 0.0) measured_q.coeffs() *= -1.0;
  const Eigen::Quaterniond difference = measured_q.inverse() * desired_q;
  return -measured * Eigen::Vector3d(difference.x(), difference.y(), difference.z());
}

inline double orientation_error_angle_rad(
    const Eigen::Vector3d& quaternion_vector_error) {
  return 2.0 *
         std::asin(std::clamp(quaternion_vector_error.norm(), 0.0, 1.0));
}

struct Guard {
  double baseline_fz_N = 0.0;
  Eigen::Vector3d start_xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  double dynamic_model_min_q7_rad = 0.0;
  double dynamic_model_max_q7_rad = 0.0;
  double minimum_residual_tz_Nm = std::numeric_limits<double>::infinity();
  double minimum_axial_delta_N = std::numeric_limits<double>::infinity();
  double maximum_axial_delta_N = -std::numeric_limits<double>::infinity();
  double maximum_axial_travel_m = 0.0;
  double maximum_lateral_error_m = 0.0;
  double maximum_orientation_error_rad = 0.0;
  const char* stop_reason = nullptr;

  void update(const std::array<double, 7>& q, double measured_fz_N,
              double residual_tz_Nm, const Eigen::Vector3d& xyz,
              double lateral_error_m, double orientation_error_rad) {
    const double axial_delta = measured_fz_N - baseline_fz_N;
    const double axial_travel = std::abs(axis.dot(xyz - start_xyz));
    minimum_residual_tz_Nm = std::min(minimum_residual_tz_Nm, residual_tz_Nm);
    minimum_axial_delta_N = std::min(minimum_axial_delta_N, axial_delta);
    maximum_axial_delta_N = std::max(maximum_axial_delta_N, axial_delta);
    maximum_axial_travel_m = std::max(maximum_axial_travel_m, axial_travel);
    maximum_lateral_error_m = std::max(maximum_lateral_error_m, lateral_error_m);
    maximum_orientation_error_rad = std::max(maximum_orientation_error_rad, orientation_error_rad);
    if (stop_reason != nullptr) return;
    if (!std::isfinite(measured_fz_N) || !std::isfinite(residual_tz_Nm) ||
        !xyz.allFinite() || !std::isfinite(lateral_error_m) ||
        !std::isfinite(orientation_error_rad)) {
      stop_reason = "invalid_control_state";
    } else if (std::abs(axial_delta) >= kHardAbsoluteAxialDeltaN) {
      stop_reason = "hard_absolute_axial_force_cutoff";
    } else if (residual_tz_Nm <= kHardNegativeResidualTzNm) {
      stop_reason = "hard_residual_torque_cutoff";
    } else if (axial_travel >= kMaximumAxialTravelM) {
      stop_reason = "maximum_axial_travel";
    } else if (lateral_error_m >= kMaximumLateralTrackingErrorM) {
      stop_reason = "maximum_lateral_tracking_error";
    } else if (orientation_error_rad >= kMaximumOrientationTrackingErrorRad) {
      stop_reason = "maximum_orientation_tracking_error";
    } else if (q[6] < kQ7CableMinRad || q[6] > kQ7CableMaxRad) {
      stop_reason = "q7_cable_limit";
    } else if (q[6] < dynamic_model_min_q7_rad - kDynamicModelLowAllowanceRad ||
               q[6] > dynamic_model_max_q7_rad) {
      stop_reason = "q7_dynamic_model_boundary";
    }
  }
};

}  // namespace hybrid_thread
