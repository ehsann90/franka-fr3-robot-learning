#include "cartesian_yaw_commissioning.h"

#include <cassert>
#include <cmath>

int main() {
  using namespace cartesian_yaw;
  assert(std::abs(angular_speed(0.0)) < 1e-12);
  assert(std::abs(angular_speed(kDurationS)) < 1e-12);
  assert(std::abs(angular_speed(kDurationS / 2.0) - kPeakAngularSpeedRadS) < 1e-12);
  assert(std::abs(angle(0.0)) < 1e-12);
  assert(std::abs(angle(kDurationS) - kTestAngleRad) < 1e-12);
  assert(std::abs(kMaximumGuardStopExtraAngleRad -
                  kPeakAngularSpeedRadS * kPeakAngularSpeedRadS /
                      (2.0 * kGuardStopAngularDecelerationRadS2)) < 1e-12);
  const double stopped_once = slew_angular_speed(0.06, 0.0, 0.001);
  assert(std::abs(stopped_once - 0.0595) < 1e-12);
  double stopped = 0.06;
  for (int i = 0; i < 120; ++i)
    stopped = slew_angular_speed(stopped, 0.0, 0.001);
  assert(std::abs(stopped) < 1e-12);
  assert(std::abs(slew_angular_speed(0.0, 0.01, 0.001) - 0.0005) < 1e-12);
  assert(!settle_complete(0.49, kRequiredStationarySamples));
  assert(!settle_complete(0.5, kRequiredStationarySamples - 1));
  assert(settle_complete(0.5, kRequiredStationarySamples));
  return 0;
}
