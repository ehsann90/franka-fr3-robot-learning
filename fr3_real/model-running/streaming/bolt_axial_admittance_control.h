#pragma once

#include <algorithm>
#include <cmath>

namespace bolt_axial_admittance {

constexpr double kApprovedTargetForceN = 1.200360;
constexpr double kMaximumAxialSpeedMS = 0.000500;
constexpr double kForceToVelocityGainMSPerN =
    kMaximumAxialSpeedMS / kApprovedTargetForceN;
constexpr double kTargetRampDurationS = 3.0;
constexpr double kDerivedVelocityAccelerationMS2 =
    kMaximumAxialSpeedMS / kTargetRampDurationS;
constexpr double kMaximumAxialTravelM = 0.003;
#ifndef FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M
#define FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M 0.0003
#endif
constexpr double kMaximumLateralTravelM = FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M;
constexpr double kHardAbsoluteAxialDeltaN = 2.0;
constexpr double kWrongDirectionDeltaN = -0.5;
constexpr double kWrongDirectionPersistenceS = 0.1;
constexpr double kStateTimeoutS = 8.0;

inline double target_force(double elapsed_s) {
  return kApprovedTargetForceN *
         std::clamp(elapsed_s / kTargetRampDurationS, 0.0, 1.0);
}

inline double requested_speed(double target_force_N,
                              double measured_delta_force_N) {
  return std::clamp(
      kForceToVelocityGainMSPerN *
          (target_force_N - measured_delta_force_N),
      -kMaximumAxialSpeedMS, kMaximumAxialSpeedMS);
}

inline double slew_speed(double previous_speed_m_s,
                         double requested_speed_m_s,
                         double period_s) {
  const double maximum_step =
      kDerivedVelocityAccelerationMS2 * std::max(0.0, period_s);
  return previous_speed_m_s +
         std::clamp(requested_speed_m_s - previous_speed_m_s,
                    -maximum_step, maximum_step);
}

}  // namespace bolt_axial_admittance
