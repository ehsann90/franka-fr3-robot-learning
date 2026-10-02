#include "bolt_axial_admittance_control.h"

#include <cmath>

int main() {
  using namespace bolt_axial_admittance;
  if (target_force(0.0) != 0.0) return 1;
  if (std::abs(target_force(1.5) - 0.5 * kApprovedTargetForceN) >= 1e-12) return 2;
  if (std::abs(target_force(3.0) - kApprovedTargetForceN) >= 1e-12) return 3;
  if (requested_speed(kApprovedTargetForceN, 0.0) != kMaximumAxialSpeedMS) return 4;
  if (requested_speed(0.0, kApprovedTargetForceN) != -kMaximumAxialSpeedMS) return 5;
  if (std::abs(requested_speed(kApprovedTargetForceN,
                               kApprovedTargetForceN)) >= 1e-12) return 6;
  const double first = slew_speed(0.0, kMaximumAxialSpeedMS, 0.001);
  if (std::abs(first - kDerivedVelocityAccelerationMS2 * 0.001) >= 1e-12) return 7;
  const double back = slew_speed(first, 0.0, 0.001);
  if (std::abs(back) >= 1e-12) return 8;
  return 0;
}
