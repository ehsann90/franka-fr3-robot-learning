#include "bolt_axial_preload_control.h"

#include <Eigen/Core>

#include <cassert>
#include <cmath>

int main() {
  using namespace bolt_preload;
  assert(std::abs(ceiling_characterization_force(0.0)) < 1e-12);
  assert(std::abs(ceiling_characterization_force(1.5) - 0.75) < 1e-12);
  assert(std::abs(ceiling_characterization_force(3.0) - 1.5) < 1e-12);
  assert(std::abs(ceiling_characterization_force(4.0) - 1.5) < 1e-12);

  const Eigen::Vector3d axis(0.0, 0.0, -1.0);
  const Eigen::Vector3d lateral = lateral_component({1.0, 2.0, 3.0}, axis);
  assert((lateral - Eigen::Vector3d(1.0, 2.0, 0.0)).norm() < 1e-12);

  Guard nominal;
  nominal.baseline_fz_N = -1.5;
  nominal.start_xyz.setZero();
  nominal.axis = axis;
  nominal.update(1.0, -0.3, {0.0, 0.0, -0.0001});
  assert(nominal.stop_reason.empty());

  Guard force_stop = nominal;
  force_stop.stop_reason.clear();
  force_stop.update(1.1, 0.6, {0.0, 0.0, 0.0});
  assert(force_stop.stop_reason == "hard_absolute_axial_force_cutoff");

  Guard lateral_stop;
  lateral_stop.axis = axis;
  lateral_stop.update(0.1, 0.0, {0.00031, 0.0, 0.0});
  assert(lateral_stop.stop_reason == "maximum_lateral_translation");

  Guard completed;
  completed.axis = axis;
  completed.update(4.0, 0.0, {0.0, 0.0, 0.0});
  assert(completed.stop_reason == "ceiling_hold_complete");
  return 0;
}
