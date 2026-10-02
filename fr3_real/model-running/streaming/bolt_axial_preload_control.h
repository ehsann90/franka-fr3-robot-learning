#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <string>

namespace bolt_preload {

constexpr double kTargetAxialForceN = 1.200360;
constexpr double kRampDurationS = 3.0;
constexpr double kHoldDurationS = 1.0;
constexpr double kTestDurationS = kRampDurationS + kHoldDurationS;
constexpr double kProportionalGain = 0.0;
constexpr double kIntegralGain = 0.0;
constexpr double kMaximumCommandedForceN = 1.500;
constexpr double kMaximumCommandSlewNs = 0.500;
constexpr double kHardAbsoluteAxialDeltaN = 2.0;
constexpr double kWrongDirectionThresholdN = 0.5;
constexpr double kWrongDirectionPersistenceS = 0.1;
constexpr double kMaximumAxialTravelM = 0.003;
constexpr double kMaximumLateralTravelM = 0.0003;

// These are the same stiffness values used in the already completed guarded
// Franky preload trial. Axial stiffness and damping are intentionally zero.
constexpr double kLateralStiffnessNm = 2000.0;
constexpr double kRotationalStiffnessNmRad = 200.0;
constexpr double kLateralDampingNsM = 2.0 * std::sqrt(kLateralStiffnessNm);
constexpr double kRotationalDampingNmsRad = 2.0 * std::sqrt(kRotationalStiffnessNmRad);

inline double ceiling_characterization_force(double elapsed_s) {
  return std::clamp(kMaximumCommandSlewNs * elapsed_s,
                    0.0, kMaximumCommandedForceN);
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
  return -measured * Eigen::Vector3d(difference.x(), difference.y(), difference.z());
}

struct Guard {
  double baseline_fz_N = 0.0;
  Eigen::Vector3d start_xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  double wrong_direction_start_s = -1.0;
  double min_axial_delta_N = 0.0;
  double max_axial_delta_N = 0.0;
  double max_axial_travel_m = 0.0;
  double max_lateral_travel_m = 0.0;
  std::string stop_reason;

  void update(double elapsed_s, double measured_fz_N, const Eigen::Vector3d& xyz) {
    const double axial_delta = measured_fz_N - baseline_fz_N;
    const Eigen::Vector3d displacement = xyz - start_xyz;
    const double axial_travel = std::abs(axis.dot(displacement));
    const double lateral_travel = lateral_component(displacement, axis).norm();
    min_axial_delta_N = std::min(min_axial_delta_N, axial_delta);
    max_axial_delta_N = std::max(max_axial_delta_N, axial_delta);
    max_axial_travel_m = std::max(max_axial_travel_m, axial_travel);
    max_lateral_travel_m = std::max(max_lateral_travel_m, lateral_travel);

    if (axial_delta <= -kWrongDirectionThresholdN) {
      if (wrong_direction_start_s < 0.0) wrong_direction_start_s = elapsed_s;
    } else {
      wrong_direction_start_s = -1.0;
    }

    if (!stop_reason.empty()) return;
    if (!std::isfinite(elapsed_s) || !std::isfinite(measured_fz_N) || !xyz.allFinite()) {
      stop_reason = "invalid_control_state";
    } else if (std::abs(axial_delta) >= kHardAbsoluteAxialDeltaN) {
      stop_reason = "hard_absolute_axial_force_cutoff";
    } else if (wrong_direction_start_s >= 0.0 &&
               elapsed_s - wrong_direction_start_s >= kWrongDirectionPersistenceS) {
      stop_reason = "persistent_wrong_direction_force";
    } else if (axial_travel >= kMaximumAxialTravelM) {
      stop_reason = "maximum_axial_travel";
    } else if (lateral_travel >= kMaximumLateralTravelM) {
      stop_reason = "maximum_lateral_translation";
    } else if (elapsed_s >= kTestDurationS) {
      stop_reason = "ceiling_hold_complete";
    }
  }
};

}  // namespace bolt_preload
