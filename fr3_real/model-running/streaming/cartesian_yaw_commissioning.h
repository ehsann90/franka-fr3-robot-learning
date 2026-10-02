#pragma once

#include <algorithm>
#include <cmath>

namespace cartesian_yaw {

constexpr double kPi = 3.14159265358979323846;
#ifndef FR3_CARTESIAN_YAW_TEST_ANGLE_DEG
#define FR3_CARTESIAN_YAW_TEST_ANGLE_DEG 5.0
#endif
constexpr double kTestAngleDeg = FR3_CARTESIAN_YAW_TEST_ANGLE_DEG;
constexpr double kTestAngleRad = kTestAngleDeg * kPi / 180.0;
// The production controller remains 0.03 rad/s. Separately named accelerated
// binaries opt into 0.06, 0.09, or 0.12 rad/s at compile time so a config cannot
// silently change the speed of an already-passed binary.
#ifndef FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S
#define FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S 0.03
#endif
constexpr double kPeakAngularSpeedRadS = FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S;
constexpr double kDurationS = kPi * kTestAngleRad / (2.0 * kPeakAngularSpeedRadS);
// A guard may become true while the half-cosine profile still commands close
// to peak speed.  Dropping that command directly to zero in one 1 kHz cycle
// asks for an acceleration discontinuity in an accelerated controller and
// is itself rejected by libfranka as a motion-generator discontinuity.  This
// bounded stop adds at most 0.0144 rad (0.825 deg) at 0.12 rad/s. An 85-degree
// window starting at q7=0.1 retains about 0.0209 rad of calibrated model margin.
constexpr double kGuardStopAngularDecelerationRadS2 = 0.5;
constexpr double kMaximumGuardStopExtraAngleRad =
    kPeakAngularSpeedRadS * kPeakAngularSpeedRadS /
    (2.0 * kGuardStopAngularDecelerationRadS2);
constexpr double kMaximumTranslationM = 0.0003;
constexpr double kHardAbsoluteAxialDeltaN = 2.0;
constexpr double kStationaryDqThresholdRadS = 0.01;
constexpr double kMinimumZeroCommandSettleS = 0.5;
constexpr int kRequiredStationarySamples = 100;
constexpr double kMaximumZeroCommandSettleS = 5.0;
constexpr double kQ7CableMinRad = -0.15688785910606384;
constexpr double kQ7CableMaxRad = 2.998908281326294;

inline double angular_speed(double elapsed_s) {
  if (elapsed_s <= 0.0 || elapsed_s >= kDurationS) return 0.0;
  return kTestAngleRad * kPi / (2.0 * kDurationS) *
         std::sin(kPi * elapsed_s / kDurationS);
}

inline double angle(double elapsed_s) {
  const double time = std::clamp(elapsed_s, 0.0, kDurationS);
  return 0.5 * kTestAngleRad * (1.0 - std::cos(kPi * time / kDurationS));
}

inline double slew_angular_speed(double previous_speed_rad_s,
                                 double requested_speed_rad_s,
                                 double period_s) {
  const double maximum_step =
      kGuardStopAngularDecelerationRadS2 * std::max(0.0, period_s);
  return previous_speed_rad_s +
         std::clamp(requested_speed_rad_s - previous_speed_rad_s,
                    -maximum_step, maximum_step);
}

inline bool settle_complete(double settle_elapsed_s, int stationary_samples) {
  return settle_elapsed_s >= kMinimumZeroCommandSettleS &&
         stationary_samples >= kRequiredStationarySamples;
}

}  // namespace cartesian_yaw
