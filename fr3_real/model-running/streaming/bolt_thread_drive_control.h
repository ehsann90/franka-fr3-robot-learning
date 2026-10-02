#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace bolt_thread_drive {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDriveAngleRad = 89.0 * kPi / 180.0;
constexpr double kPeakAngularSpeedRadS = 0.09;
constexpr double kDriveDurationS =
    kDriveAngleRad * kPi / (2.0 * kPeakAngularSpeedRadS);
constexpr double kForceCommandCeilingN = 1.5;
constexpr double kForceCommandSlewNs = 0.5;
constexpr double kHardAbsoluteAxialDeltaN = 2.0;
constexpr double kHardNegativeResidualTzNm = -0.5383051567216975;
constexpr double kMaximumAxialTravelM = 0.003;
constexpr double kMaximumLateralTravelM = 0.0003;
constexpr double kMaximumToolAxisTiltRad = 5.0 * kPi / 180.0;
constexpr double kMaximumOtherJointDeviationRad = 0.12;
constexpr double kQ7CableMinRad = -0.15688785910606384;
constexpr double kQ7CableMaxRad = 2.998908281326294;
constexpr double kCommissionedStartQ7Rad = 0.043015;
constexpr double kStartQ7ToleranceRad = 0.03;
constexpr double kQ7OvershootAllowanceRad = 0.01;
constexpr double kDynamicModelLowEndpointAllowanceRad = 0.02;

constexpr double kLateralStiffnessNm = 2000.0;
constexpr double kRotationalStiffnessNmRad = 200.0;
constexpr double kLateralDampingNsM = 2.0 * std::sqrt(kLateralStiffnessNm);
constexpr double kRotationalDampingNmsRad =
    2.0 * std::sqrt(kRotationalStiffnessNmRad);

inline double drive_angle(double elapsed_s) {
  const double phase = std::clamp(elapsed_s / kDriveDurationS, 0.0, 1.0);
  return 0.5 * kDriveAngleRad * (1.0 - std::cos(kPi * phase));
}

inline double drive_angular_speed(double elapsed_s) {
  const double phase = std::clamp(elapsed_s / kDriveDurationS, 0.0, 1.0);
  if (phase <= 0.0 || phase >= 1.0) return 0.0;
  return 0.5 * kDriveAngleRad * kPi / kDriveDurationS *
         std::sin(kPi * phase);
}

inline double force_command(double elapsed_s) {
  return std::clamp(kForceCommandSlewNs * elapsed_s, 0.0,
                    kForceCommandCeilingN);
}

inline double interpolate(const std::vector<double>& x,
                          const std::vector<double>& y,
                          double query) {
  if (x.size() != y.size() || x.size() < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (query <= x.front()) return y.front();
  if (query >= x.back()) return y.back();
  const auto upper = std::upper_bound(x.begin(), x.end(), query);
  const std::size_t hi = static_cast<std::size_t>(upper - x.begin());
  const std::size_t lo = hi - 1;
  const double fraction = (query - x[lo]) / (x[hi] - x[lo]);
  return y[lo] + fraction * (y[hi] - y[lo]);
}

inline Eigen::Vector3d lateral_component(const Eigen::Vector3d& vector,
                                         const Eigen::Vector3d& unit_axis) {
  return vector - unit_axis * unit_axis.dot(vector);
}

inline Eigen::Vector3d orientation_error(const Eigen::Matrix3d& desired,
                                         const Eigen::Matrix3d& measured) {
  Eigen::Quaterniond desired_q(desired);
  Eigen::Quaterniond measured_q(measured);
  if (desired_q.coeffs().dot(measured_q.coeffs()) < 0.0) {
    measured_q.coeffs() *= -1.0;
  }
  const Eigen::Quaterniond difference = measured_q.inverse() * desired_q;
  return -measured *
         Eigen::Vector3d(difference.x(), difference.y(), difference.z());
}

struct Guard {
  double baseline_fz_N = 0.0;
  Eigen::Vector3d start_xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  std::array<double, 7> start_q{};
  double target_q7_rad = 0.0;
  double dynamic_model_min_q7_rad = 0.0;
  double dynamic_model_max_q7_rad = 0.0;
  double minimum_residual_tz_Nm = std::numeric_limits<double>::infinity();
  double maximum_axial_delta_N = -std::numeric_limits<double>::infinity();
  double minimum_axial_delta_N = std::numeric_limits<double>::infinity();
  double maximum_axial_travel_m = 0.0;
  double maximum_lateral_travel_m = 0.0;
  double maximum_tool_axis_tilt_rad = 0.0;
  double maximum_other_joint_deviation_rad = 0.0;
  const char* stop_reason = nullptr;

  void update(double elapsed_s, const std::array<double, 7>& q,
              double measured_fz_N, double residual_tz_Nm,
              const Eigen::Vector3d& xyz,
              const Eigen::Vector3d& current_axis) {
    const double axial_delta = measured_fz_N - baseline_fz_N;
    const Eigen::Vector3d displacement = xyz - start_xyz;
    const double axial_travel = std::abs(axis.dot(displacement));
    const double lateral_travel = lateral_component(displacement, axis).norm();
    const double tool_axis_tilt =
        std::acos(std::clamp(axis.dot(current_axis), -1.0, 1.0));
    double other_joint_deviation = 0.0;
    for (std::size_t index = 0; index < 6; ++index) {
      other_joint_deviation =
          std::max(other_joint_deviation, std::abs(q[index] - start_q[index]));
    }
    minimum_residual_tz_Nm = std::min(minimum_residual_tz_Nm, residual_tz_Nm);
    maximum_axial_delta_N = std::max(maximum_axial_delta_N, axial_delta);
    minimum_axial_delta_N = std::min(minimum_axial_delta_N, axial_delta);
    maximum_axial_travel_m = std::max(maximum_axial_travel_m, axial_travel);
    maximum_lateral_travel_m =
        std::max(maximum_lateral_travel_m, lateral_travel);
    maximum_tool_axis_tilt_rad =
        std::max(maximum_tool_axis_tilt_rad, tool_axis_tilt);
    maximum_other_joint_deviation_rad =
        std::max(maximum_other_joint_deviation_rad, other_joint_deviation);

    if (stop_reason != nullptr) return;
    if (!std::isfinite(elapsed_s) || !std::isfinite(measured_fz_N) ||
        !std::isfinite(residual_tz_Nm) || !xyz.allFinite() ||
        !current_axis.allFinite()) {
      stop_reason = "invalid_control_state";
    } else if (std::abs(axial_delta) >= kHardAbsoluteAxialDeltaN) {
      stop_reason = "hard_absolute_axial_force_cutoff";
    } else if (residual_tz_Nm <= kHardNegativeResidualTzNm) {
      stop_reason = "hard_residual_torque_cutoff";
    } else if (axial_travel >= kMaximumAxialTravelM) {
      stop_reason = "maximum_axial_travel";
    } else if (lateral_travel >= kMaximumLateralTravelM) {
      stop_reason = "maximum_lateral_translation";
    } else if (tool_axis_tilt >= kMaximumToolAxisTiltRad) {
      stop_reason = "maximum_tool_axis_tilt";
    } else if (other_joint_deviation >= kMaximumOtherJointDeviationRad) {
      stop_reason = "maximum_other_joint_deviation";
    } else if (q[6] < kQ7CableMinRad || q[6] > kQ7CableMaxRad ||
               q[6] > target_q7_rad + kQ7OvershootAllowanceRad) {
      stop_reason = "q7_cable_or_target_guard";
    } else if (q[6] < dynamic_model_min_q7_rad -
                            kDynamicModelLowEndpointAllowanceRad ||
               q[6] > dynamic_model_max_q7_rad) {
      stop_reason = "q7_dynamic_model_guard";
    } else if (elapsed_s >= kDriveDurationS) {
      stop_reason = "drive_window_complete";
    }
  }
};

}  // namespace bolt_thread_drive
