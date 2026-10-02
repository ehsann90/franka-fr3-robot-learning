// Read-only continuous IK preview for a camera-outboard bracket trajectory.
//
// The input contains recorded q seeds and desired O_T_EE transforms.  This
// utility loads libfranka's robot model, solves those targets, and writes a
// candidate joint path.  It never starts control or issues a robot command.

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kJoints = 7;
// Nominal FR3 limits from the installed fr3.urdf.  The preview keeps an
// additional margin inside them; hand guiding can enter a wider soft-limit
// region that must not be copied directly into an autonomous FCI replay.
constexpr std::array<double, kJoints> kJointMin{{
    -2.7501, -1.7918, -2.9065, -3.0481, -2.8101, 0.54092, -3.0196}};
constexpr std::array<double, kJoints> kJointMax{{
    2.7501, 1.7918, 2.9065, -0.1458, 2.8101, 4.5205, 3.0196}};
// Match the conservative autonomous replay envelope (approximately 0.12 rad
// inside the nominal FR3 limits), rather than the wider soft-limit region that
// Desk permits during hand guiding.
constexpr double kJointMargin = 0.125;
constexpr double kPositionToleranceM = 5e-5;
constexpr double kOrientationToleranceRad = 2e-4;
constexpr double kMaximumAdjacentJumpRad = 0.08;
constexpr int kMaximumIterations = 500;

using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;

struct Target {
  double time_s = 0.0;
  Vector7d source_q = Vector7d::Zero();
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
};

struct Solution {
  Vector7d q = Vector7d::Zero();
  double position_error_m = std::numeric_limits<double>::infinity();
  double orientation_error_rad = std::numeric_limits<double>::infinity();
  bool converged = false;
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream input(line);
  std::string field;
  while (std::getline(input, field, ',')) fields.push_back(field);
  return fields;
}

std::vector<Target> load_targets(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open IK target CSV");
  std::string line;
  if (!std::getline(input, line) || split(line).size() != 24)
    throw std::runtime_error("IK target header must have 24 columns");
  std::vector<Target> targets;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() != 24) throw std::runtime_error("IK target row must have 24 columns");
    Target target;
    std::size_t field = 0;
    target.time_s = std::stod(fields[field++]);
    for (Eigen::Index joint = 0; joint < 7; ++joint)
      target.source_q(joint) = std::stod(fields[field++]);
    for (Eigen::Index row = 0; row < 4; ++row)
      for (Eigen::Index column = 0; column < 4; ++column)
        target.pose(row, column) = std::stod(fields[field++]);
    if (!std::isfinite(target.time_s) || !target.source_q.allFinite() ||
        !target.pose.allFinite())
      throw std::runtime_error("IK target contains a non-finite value");
    targets.push_back(target);
  }
  if (targets.size() < 2) throw std::runtime_error("IK target CSV has too few rows");
  return targets;
}

Vector7d clamp_to_limits(const Vector7d& value) {
  Vector7d result = value;
  for (Eigen::Index joint = 0; joint < 7; ++joint) {
    result(joint) = std::clamp(
        result(joint), kJointMin[static_cast<std::size_t>(joint)] + kJointMargin,
        kJointMax[static_cast<std::size_t>(joint)] - kJointMargin);
  }
  return result;
}

double limit_margin(const Vector7d& q) {
  double margin = std::numeric_limits<double>::infinity();
  for (Eigen::Index joint = 0; joint < 7; ++joint) {
    const auto index = static_cast<std::size_t>(joint);
    margin = std::min(margin, q(joint) - kJointMin[index]);
    margin = std::min(margin, kJointMax[index] - q(joint));
  }
  return margin;
}

Eigen::Matrix4d model_pose(const franka::Model& model, const Vector7d& q,
                           const std::array<double, 16>& F_T_EE,
                           const std::array<double, 16>& EE_T_K) {
  std::array<double, 7> q_array{};
  Eigen::Map<Vector7d>(q_array.data()) = q;
  const auto pose = model.pose(franka::Frame::kEndEffector, q_array, F_T_EE, EE_T_K);
  return Eigen::Map<const Eigen::Matrix<double, 4, 4>>(pose.data());
}

Matrix67d model_jacobian(const franka::Model& model, const Vector7d& q,
                         const std::array<double, 16>& F_T_EE,
                         const std::array<double, 16>& EE_T_K) {
  std::array<double, 7> q_array{};
  Eigen::Map<Vector7d>(q_array.data()) = q;
  const auto jacobian =
      model.zeroJacobian(franka::Frame::kEndEffector, q_array, F_T_EE, EE_T_K);
  return Eigen::Map<const Matrix67d>(jacobian.data());
}

Eigen::Matrix<double, 6, 1> pose_error(const Eigen::Matrix4d& current,
                                       const Eigen::Matrix4d& target) {
  Eigen::Matrix<double, 6, 1> error;
  error.head<3>() = target.topRightCorner<3, 1>() - current.topRightCorner<3, 1>();
  Eigen::AngleAxisd rotation_error(
      target.topLeftCorner<3, 3>() * current.topLeftCorner<3, 3>().transpose());
  error.tail<3>() = rotation_error.axis() * rotation_error.angle();
  if (!error.allFinite()) error.tail<3>().setZero();
  return error;
}

double weighted_cost(const Eigen::Matrix<double, 6, 1>& error) {
  return error.head<3>().squaredNorm() + 0.08 * error.tail<3>().squaredNorm();
}

Solution solve_target(const franka::Model& model, const Target& target,
                      const Vector7d& initial, const Vector7d& preference,
                      const std::array<double, 16>& F_T_EE,
                      const std::array<double, 16>& EE_T_K,
                      int max_iterations = kMaximumIterations) {
  Vector7d q = clamp_to_limits(initial);
  Solution best;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const Eigen::Matrix4d current = model_pose(model, q, F_T_EE, EE_T_K);
    const auto error = pose_error(current, target.pose);
    const double position_error = error.head<3>().norm();
    const double orientation_error = error.tail<3>().norm();
    if (!best.converged && position_error + 0.1 * orientation_error <
                               best.position_error_m +
                                   0.1 * best.orientation_error_rad) {
      best.q = q;
      best.position_error_m = position_error;
      best.orientation_error_rad = orientation_error;
    }
    if (position_error <= kPositionToleranceM &&
        orientation_error <= kOrientationToleranceRad) {
      const Matrix67d jacobian = model_jacobian(model, q, F_T_EE, EE_T_K);
      const Eigen::JacobiSVD<Matrix67d> svd(jacobian, Eigen::ComputeFullV);
      const Vector7d null_direction = svd.matrixV().col(6);
      const double projected_preference = null_direction.dot(preference - q);

      // Record the best exact-pose point reached on this redundancy manifold.
      // A seven-joint arm has one local null-space direction for a six-DoF
      // Cartesian target.  Following that direction is what lets the arm move
      // away from the literal q7-pi branch without moving the tool target.
      const double posture_cost = (preference - q).squaredNorm();
      const double best_posture_cost = (preference - best.q).squaredNorm();
      if (!best.converged || posture_cost < best_posture_cost) {
        best.q = q;
        best.position_error_m = position_error;
        best.orientation_error_rad = orientation_error;
        best.converged = true;
      }
      if (std::abs(projected_preference) < 2e-4) return best;

      const double signed_step = std::clamp(0.25 * projected_preference, -0.02, 0.02);
      const Vector7d candidate = clamp_to_limits(q + signed_step * null_direction);
      if ((candidate - q).cwiseAbs().maxCoeff() < 1e-7) return best;
      q = candidate;
      continue;
    }

    const Matrix67d jacobian = model_jacobian(model, q, F_T_EE, EE_T_K);
    constexpr double damping = 0.015;
    const Eigen::Matrix<double, 6, 6> task_matrix =
        jacobian * jacobian.transpose() +
        damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::Matrix<double, 7, 6> pseudo_inverse =
        jacobian.transpose() * task_matrix.ldlt().solve(
                                   Eigen::Matrix<double, 6, 6>::Identity());
    Vector7d delta = pseudo_inverse * error;
    const Eigen::Matrix<double, 7, 7> nullspace =
        Eigen::Matrix<double, 7, 7>::Identity() - pseudo_inverse * jacobian;
    delta += 0.08 * nullspace * (preference - q);
    const double maximum = delta.cwiseAbs().maxCoeff();
    if (maximum > 0.12) delta *= 0.12 / maximum;

    const double old_cost = weighted_cost(error);
    bool accepted = false;
    for (double scale : {1.0, 0.5, 0.25, 0.1, 0.05}) {
      const Vector7d candidate = clamp_to_limits(q + scale * delta);
      const double candidate_cost = weighted_cost(
          pose_error(model_pose(model, candidate, F_T_EE, EE_T_K), target.pose));
      if (candidate_cost < old_cost) {
        q = candidate;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      q = clamp_to_limits(q + 0.01 * nullspace * (preference - q));
    }
  }
  return best;
}

// Solves the six non-redundant joints for an exact pose match with q7 held at
// a directly scheduled value, instead of leaving q7 to a weak null-space bias
// toward a soft preference. For the manually recorded path, the local
// null-space direction at a converged posture was found to have almost no
// q7 component -- nudging a "preference" through it barely moved q7 across
// hundreds of frames, so the solved path stayed on its initial branch while
// the recorded path kept moving, and the two eventually could not be
// reconciled without a jump. Fixing q7 to a smooth, safety-clamped schedule
// and solving the remaining six joints as a square (locally well-posed) IK
// problem avoids that weak-coupling failure mode entirely.
Solution solve_target_fixed_q7(const franka::Model& model, const Target& target,
                               const Vector7d& initial, double q7,
                               const std::array<double, 16>& F_T_EE,
                               const std::array<double, 16>& EE_T_K,
                               int max_iterations = kMaximumIterations) {
  Vector7d q = clamp_to_limits(initial);
  q(6) = std::clamp(q7, kJointMin[6] + kJointMargin, kJointMax[6] - kJointMargin);
  Solution best;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const Eigen::Matrix4d current = model_pose(model, q, F_T_EE, EE_T_K);
    const auto error = pose_error(current, target.pose);
    const double position_error = error.head<3>().norm();
    const double orientation_error = error.tail<3>().norm();
    if (!best.converged && position_error + 0.1 * orientation_error <
                               best.position_error_m +
                                   0.1 * best.orientation_error_rad) {
      best.q = q;
      best.position_error_m = position_error;
      best.orientation_error_rad = orientation_error;
    }
    if (position_error <= kPositionToleranceM &&
        orientation_error <= kOrientationToleranceRad) {
      best.q = q;
      best.position_error_m = position_error;
      best.orientation_error_rad = orientation_error;
      best.converged = true;
      return best;
    }

    const Matrix67d jacobian = model_jacobian(model, q, F_T_EE, EE_T_K);
    const Eigen::Matrix<double, 6, 6> jacobian6 = jacobian.leftCols<6>();
    constexpr double damping = 0.015;
    const Eigen::Matrix<double, 6, 6> task_matrix =
        jacobian6 * jacobian6.transpose() +
        damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::Matrix<double, 6, 1> delta6 =
        jacobian6.transpose() *
        task_matrix.ldlt().solve(error);
    Vector7d delta = Vector7d::Zero();
    delta.head<6>() = delta6;
    const double maximum = delta.cwiseAbs().maxCoeff();
    if (maximum > 0.12) delta *= 0.12 / maximum;

    const double old_cost = weighted_cost(error);
    bool accepted = false;
    for (double scale : {1.0, 0.5, 0.25, 0.1, 0.05}) {
      Vector7d candidate = clamp_to_limits(q + scale * delta);
      candidate(6) = q(6);
      const double candidate_cost = weighted_cost(
          pose_error(model_pose(model, candidate, F_T_EE, EE_T_K), target.pose));
      if (candidate_cost < old_cost) {
        q = candidate;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      Vector7d candidate = clamp_to_limits(q + 0.05 * delta);
      candidate(6) = q(6);
      q = candidate;
    }
  }
  return best;
}

std::vector<Vector7d> initial_seeds(const Target& target) {
  std::vector<Vector7d> seeds;
  seeds.push_back(clamp_to_limits(target.source_q));
  Vector7d reverse = target.source_q;
  reverse(6) -= M_PI;
  seeds.push_back(clamp_to_limits(reverse));
  for (double q7 : {-2.70, -2.25, -1.80, -1.35, -0.90, -0.45}) {
    for (double q1_offset : {0.0, -0.45, 0.45}) {
      for (double q3_offset : {0.0, -0.45, 0.45}) {
        Vector7d seed = target.source_q;
        seed(0) += q1_offset;
        seed(2) += q3_offset;
        seed(6) = q7;
        seeds.push_back(clamp_to_limits(seed));
      }
    }
  }
  return seeds;
}

Vector7d safe_preference(const Target& target, bool target_matches_source_orientation) {
  Vector7d preference = target.source_q;
  if (target_matches_source_orientation) {
    // Track the recorded q7 exactly while it is inside the replay-safe
    // envelope, and only pull it toward the boundary once the demonstration
    // itself would exceed it. A constant preference away from the source
    // trace forces the solver to fight a naturally continuous recorded path
    // from frame one, so by the time the source trace actually reaches the
    // limit the solved path is already far from it -- that mismatch is what
    // produces a branch discontinuity right at the crossing point. Hugging
    // the source trace until the last safe moment keeps the redundant-joint
    // walk minimal and leaves it maximal runway to absorb the roll that the
    // clamp can no longer track.
    constexpr double kSafeQ7Min =
        kJointMin[6] + kJointMargin;
    constexpr double kSafeQ7Max =
        kJointMax[6] - kJointMargin;
    preference(6) = std::clamp(target.source_q(6), kSafeQ7Min, kSafeQ7Max);
    return clamp_to_limits(preference);
  }
  // The desired Cartesian orientation already encodes the reverse 180-degree
  // roll. Keep q7 away from its lower stop and let the redundant arm joints
  // distribute that roll rather than tracking source_q7-pi literally.
  preference(6) = -1.70;
  return clamp_to_limits(preference);
}

// Dispatches to the fixed-q7 solve for the manually recorded exact-pose path
// (preference(6) already holds that frame's scheduled q7, from
// safe_preference), or to the original preference-nudged solve for the
// changed-orientation path, which was already working and is left alone.
Solution solve_frame(const franka::Model& model, const Target& target, const Vector7d& initial,
                     const Vector7d& preference, bool target_matches_source_pose,
                     const std::array<double, 16>& F_T_EE, const std::array<double, 16>& EE_T_K,
                     int max_iterations = kMaximumIterations) {
  if (target_matches_source_pose)
    return solve_target_fixed_q7(model, target, initial, preference(6), F_T_EE, EE_T_K,
                                 max_iterations);
  return solve_target(model, target, initial, preference, F_T_EE, EE_T_K, max_iterations);
}

std::vector<Vector7d> continuation_recovery_seeds(const Target& target,
                                                   const Vector7d& previous) {
  std::vector<Vector7d> seeds;
  seeds.push_back(previous);
  for (Eigen::Index joint : {0, 2, 4, 6}) {
    for (double offset : {-0.20, -0.10, 0.10, 0.20}) {
      Vector7d seed = previous;
      seed(joint) += offset;
      seeds.push_back(clamp_to_limits(seed));
    }
  }
  const auto global = initial_seeds(target);
  seeds.insert(seeds.end(), global.begin(), global.end());
  return seeds;
}

Eigen::Matrix4d interpolate_pose(const Eigen::Matrix4d& start, const Eigen::Matrix4d& end,
                                 double t) {
  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.topRightCorner<3, 1>() =
      (1.0 - t) * start.topRightCorner<3, 1>() + t * end.topRightCorner<3, 1>();
  const Eigen::Quaterniond qa(Eigen::Matrix3d(start.topLeftCorner<3, 3>()));
  const Eigen::Quaterniond qb(Eigen::Matrix3d(end.topLeftCorner<3, 3>()));
  result.topLeftCorner<3, 3>() = qa.slerp(t, qb).toRotationMatrix();
  return result;
}

constexpr int kSubdivisionSteps = 8;
constexpr int kSubdivisionMaxIterations = 150;

// Walks a small, fixed number of Cartesian waypoints between two
// already-accepted poses so the redundant-arm null-space branch is tracked
// continuously instead of re-solved from a seed that is far away in
// configuration space. This is a bounded-cost sequential walk (at most
// kSubdivisionSteps solves), not a recursive bisection, so a stuck target
// cannot blow up into an exponential amount of solver work.
bool solve_via_subdivision(const franka::Model& model, const Vector7d& start_q,
                           const Eigen::Matrix4d& start_pose, const Eigen::Matrix4d& end_pose,
                           const Vector7d& preference, bool target_matches_source_pose,
                           const std::array<double, 16>& F_T_EE,
                           const std::array<double, 16>& EE_T_K, Solution* out) {
  Vector7d q = start_q;
  Solution step_solution;
  for (int step = 1; step <= kSubdivisionSteps; ++step) {
    const double t = static_cast<double>(step) / kSubdivisionSteps;
    Target waypoint;
    waypoint.pose = interpolate_pose(start_pose, end_pose, t);
    waypoint.source_q = q;
    step_solution = solve_frame(model, waypoint, q, preference, target_matches_source_pose,
                                F_T_EE, EE_T_K, kSubdivisionMaxIterations);
    const double step_jump = step_solution.converged
                                  ? (step_solution.q - q).cwiseAbs().maxCoeff()
                                  : std::numeric_limits<double>::infinity();
    if (!step_solution.converged || step_jump > kMaximumAdjacentJumpRad) {
      *out = step_solution;
      return false;
    }
    q = step_solution.q;
  }
  *out = step_solution;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " ROBOT_IP TARGETS.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    const auto targets = load_targets(argv[2]);
    std::cout << "=== FR3 BRACKET OUTER-WRIST IK PREVIEW ===\n"
              << "mode:                       READ_ONLY_MODEL_SOLVE\n"
              << "robot IP:                   " << argv[1] << "\n"
              << "targets:                    " << targets.size() << "\n"
              << "target orientation:         supplied exactly by the input poses\n"
              << "control/motion commands:    DISABLED BY DESIGN\n"
              << "Hand commands:              DISABLED BY DESIGN\n"
              << "collision settings:         UNCHANGED\n"
              << "Press Enter to connect read-only (Ctrl+C cancels): ";
    std::string confirmation;
    std::getline(std::cin, confirmation);

    franka::Robot robot(argv[1]);
    robot.readOnce();
    const franka::RobotState state = robot.readOnce();
    const franka::Model model = robot.loadModel();

    double maximum_source_position_error = 0.0;
    double maximum_source_orientation_error = 0.0;
    for (const Target& target : targets) {
      const auto error = pose_error(
          model_pose(model, target.source_q, state.F_T_EE, state.EE_T_K), target.pose);
      maximum_source_position_error =
          std::max(maximum_source_position_error, error.head<3>().norm());
      maximum_source_orientation_error =
          std::max(maximum_source_orientation_error, error.tail<3>().norm());
    }
    // A new cell starts at the recorded staging pose, then translates the
    // target later in the path. Classify the *whole* trace, not just frame 0.
    // Small, gradual wall-pose corrections should remain on the source q7
    // branch. The reverse-roll target differs by about pi and stays separate.
    const bool target_matches_source_orientation =
        maximum_source_orientation_error < 0.20;
    const bool target_matches_source_pose =
        maximum_source_orientation_error < 1e-3 && maximum_source_position_error < 1e-4;
    std::cout << "IK target type:              "
              << (target_matches_source_pose
                      ? "recorded pose with scheduled q7"
                      : target_matches_source_orientation
                            ? "adapted pose with source-q7 preference"
                            : "changed-orientation pose")
              << "\n";

    std::vector<Solution> solutions;
    solutions.reserve(targets.size());
    Vector7d preference = safe_preference(targets.front(), target_matches_source_orientation);

    Solution first_best;
    for (const Vector7d& seed : initial_seeds(targets.front())) {
      const Solution candidate = solve_frame(
          model, targets.front(), seed, preference, target_matches_source_pose,
          state.F_T_EE, state.EE_T_K);
      if (candidate.converged) {
        // Prefer a branch with room to traverse the complete trajectory, not
        // merely the first reverse-q7 solution returned by local IK.
        if (!first_best.converged || limit_margin(candidate.q) > limit_margin(first_best.q))
          first_best = candidate;
        continue;
      }
      if (!first_best.converged &&
          candidate.position_error_m + 0.1 * candidate.orientation_error_rad <
          first_best.position_error_m + 0.1 * first_best.orientation_error_rad)
        first_best = candidate;
    }
    if (!first_best.converged)
      throw std::runtime_error("no joint-limit-safe IK solution for the first target");
    solutions.push_back(first_best);

    double maximum_jump = 0.0;
    double maximum_position_error = first_best.position_error_m;
    double maximum_orientation_error = first_best.orientation_error_rad;
    double minimum_limit_margin = limit_margin(first_best.q);
    for (std::size_t index = 1; index < targets.size(); ++index) {
      preference = safe_preference(targets[index], target_matches_source_orientation);
      Solution solution = solve_frame(
          model, targets[index], solutions.back().q, preference, target_matches_source_pose,
          state.F_T_EE, state.EE_T_K);
      double jump = solution.converged
                        ? (solution.q - solutions.back().q).cwiseAbs().maxCoeff()
                        : std::numeric_limits<double>::infinity();
      // A converged solve can still land on a different redundancy branch
      // than the previous frame, producing a large joint jump even though
      // the Cartesian pose matches. Recovery must trigger on that case too,
      // not only on outright non-convergence, or a branch-hop is reported as
      // an unrecoverable discontinuity when a continuous solution exists.
      if (!solution.converged || jump > kMaximumAdjacentJumpRad) {
        Solution recovery_best;
        bool recovery_within_jump = false;
        for (const Vector7d& seed :
             continuation_recovery_seeds(targets[index], solutions.back().q)) {
          const Solution candidate = solve_frame(
              model, targets[index], seed, preference, target_matches_source_pose,
              state.F_T_EE, state.EE_T_K);
          if (!candidate.converged) {
            // Preserve the best failed attempt for an actionable diagnostic.
            // Previously recovery_best stayed at infinity, concealing whether
            // the target was nearly reachable or fundamentally incompatible.
            const double candidate_cost =
                candidate.position_error_m + 0.1 * candidate.orientation_error_rad;
            const double best_cost =
                recovery_best.position_error_m + 0.1 * recovery_best.orientation_error_rad;
            if (!recovery_best.converged && candidate_cost < best_cost)
              recovery_best = candidate;
            continue;
          }
          const double candidate_jump =
              (candidate.q - solutions.back().q).cwiseAbs().maxCoeff();
          const bool candidate_within_jump = candidate_jump <= kMaximumAdjacentJumpRad;
          if (candidate_within_jump && !recovery_within_jump) {
            recovery_best = candidate;
            recovery_within_jump = true;
            // First seed that lands within the jump budget wins; stop paying
            // for further seeds instead of exhaustively scoring all ~70 of
            // them on every recovery (that scan was the main driver of the
            // runaway solve time on a stuck stretch of frames).
            break;
          }
          if (candidate_within_jump != recovery_within_jump) continue;
          const double best_jump = recovery_best.converged
                                       ? (recovery_best.q - solutions.back().q)
                                             .cwiseAbs()
                                             .maxCoeff()
                                       : std::numeric_limits<double>::infinity();
          if (!recovery_best.converged || candidate_jump < best_jump)
            recovery_best = candidate;
        }
        if (!recovery_within_jump) {
          Solution subdivided;
          if (solve_via_subdivision(model, solutions.back().q, targets[index - 1].pose,
                                    targets[index].pose, preference, target_matches_source_pose,
                                    state.F_T_EE, state.EE_T_K, &subdivided)) {
            recovery_best = subdivided;
            recovery_within_jump = true;
          }
        }
        solution = recovery_best;
        jump = solution.converged
                   ? (solution.q - solutions.back().q).cwiseAbs().maxCoeff()
                   : std::numeric_limits<double>::infinity();
      }
      if (!solution.converged) {
        std::ostringstream message;
        message << "IK failed at target " << index << " time="
                << targets[index].time_s << " s; best position/orientation error="
                << solution.position_error_m << "/" << solution.orientation_error_rad
                << "; best q7=" << solution.q(6)
                << "; hard-limit margin=" << limit_margin(solution.q);
        throw std::runtime_error(message.str());
      }
      if (jump > kMaximumAdjacentJumpRad)
        throw std::runtime_error("IK joint discontinuity at target " +
                                 std::to_string(index));
      maximum_jump = std::max(maximum_jump, jump);
      maximum_position_error = std::max(maximum_position_error, solution.position_error_m);
      maximum_orientation_error =
          std::max(maximum_orientation_error, solution.orientation_error_rad);
      minimum_limit_margin = std::min(minimum_limit_margin, limit_margin(solution.q));
      solutions.push_back(solution);
      if (index % 100 == 0 || index + 1 == targets.size()) {
        std::cout << "[IK] solved " << (index + 1) << "/" << targets.size()
                  << " targets; q7=" << std::fixed << std::setprecision(4)
                  << solution.q(6) << " rad; minimum hard-limit margin="
                  << minimum_limit_margin << " rad\n" << std::flush;
      }
    }

    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("cannot open IK output CSV");
    output << "time_s";
    for (std::size_t joint = 0; joint < kJoints; ++joint) output << ",q" << (joint + 1);
    output << ",position_error_m,orientation_error_rad\n";
    output << std::setprecision(17);
    for (std::size_t index = 0; index < targets.size(); ++index) {
      output << targets[index].time_s;
      for (Eigen::Index joint = 0; joint < 7; ++joint) output << ',' << solutions[index].q(joint);
      output << ',' << solutions[index].position_error_m << ','
             << solutions[index].orientation_error_rad << '\n';
    }
    std::cout << std::fixed << std::setprecision(9)
              << "maximum position error:     " << maximum_position_error << " m\n"
              << "maximum orientation error:  " << maximum_orientation_error << " rad\n"
              << "maximum adjacent q jump:    " << maximum_jump << " rad\n"
              << "minimum hard-limit margin:  " << minimum_limit_margin << " rad\n"
              << "[PASS] continuous read-only IK candidate written: " << argv[3] << "\n"
              << "No control session, arm, Hand, or collision-setting command was issued.\n";
  } catch (const franka::Exception& exception) {
    std::cerr << "libfranka error: " << exception.what() << '\n';
    return 1;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}
