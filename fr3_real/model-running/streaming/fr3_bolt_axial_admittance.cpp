#include "bolt_axial_admittance_control.h"
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
#include "cartesian_yaw_commissioning.h"
#endif

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/robot.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kBaselineDurationS = 3.0;
#ifndef FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N
#define FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N 0.1
#endif
constexpr double kMaximumBaselineFzStdN = FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N;
constexpr double kMaximumBaselineDqRadS = 0.01;
constexpr double kJointMarginRad = 0.12;
constexpr std::array<double, 7> kJointMin{{
    -2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159}};
constexpr std::array<double, 7> kJointMax{{
    2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159}};
constexpr std::array<double, 3> kWorkspaceMin{{0.57, 0.30, 0.08}};
constexpr std::array<double, 3> kWorkspaceMax{{0.63, 0.35, 0.15}};
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
// The longest explicitly compiled drive is 85 degrees: about 80.7 s of
// acquisition plus motion, followed by a bounded stationary termination.
// Keep telemetry capacity above the complete control-session timeout so
// logging can never become the reason a commissioned drive terminates.
constexpr std::size_t kMaximumSamples = 120000;
constexpr double kMotionCompletionS =
    bolt_axial_admittance::kTargetRampDurationS + cartesian_yaw::kDurationS;
constexpr double kDerivedStateTimeoutS = kMotionCompletionS + 5.0;
constexpr double kHardNegativeResidualTzNm = -0.5383051567216975;
constexpr double kDynamicModelLowAllowanceRad = 0.02;
#else
constexpr std::size_t kMaximumSamples = 10000;
constexpr double kMotionCompletionS =
    bolt_axial_admittance::kTargetRampDurationS;
constexpr double kDerivedStateTimeoutS =
    bolt_axial_admittance::kStateTimeoutS;
#endif
constexpr double kStationaryDqThresholdRadS = 0.01;
constexpr double kMinimumZeroCommandSettleS = 0.5;
constexpr int kRequiredStationarySamples = 100;
constexpr double kMaximumZeroCommandSettleS = 5.0;
constexpr double kQ7CableMinRad = -0.15688785910606384;
constexpr double kQ7CableMaxRad = 2.998908281326294;

std::atomic<bool> g_abort{false};
void signal_handler(int) { g_abort.store(true, std::memory_order_relaxed); }

Eigen::Matrix4d pose_matrix(const std::array<double, 16>& pose) {
  return Eigen::Map<const Eigen::Matrix<double, 4, 4>>(pose.data());
}

bool in_workspace(const Eigen::Vector3d& xyz) {
  for (Eigen::Index i = 0; i < 3; ++i) {
    if (xyz(i) < kWorkspaceMin[static_cast<std::size_t>(i)] ||
        xyz(i) > kWorkspaceMax[static_cast<std::size_t>(i)]) return false;
  }
  return true;
}

bool joints_within_guard(const std::array<double, 7>& q) {
  for (std::size_t i = 0; i < q.size(); ++i) {
    if (q[i] <= kJointMin[i] + kJointMarginRad ||
        q[i] >= kJointMax[i] - kJointMarginRad) return false;
  }
  return true;
}

struct Sample {
  std::int64_t wall_time_ns = 0;
  double time_s = 0.0;
  double target_force_N = 0.0;
  double measured_delta_force_N = 0.0;
  double requested_speed_m_s = 0.0;
  double commanded_speed_m_s = 0.0;
  double axial_travel_m = 0.0;
  double lateral_travel_m = 0.0;
  double maximum_abs_dq_rad_s = 0.0;
  double commanded_angular_speed_rad_s = 0.0;
  double measured_yaw_rad = 0.0;
  double residual_tz_Nm = 0.0;
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 3> xyz{};
};

void write_csv(const std::filesystem::path& path,
               const std::vector<Sample>& samples) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open telemetry output");
  output << "wall_time_ns,time_s,target_force_N,measured_delta_force_N,requested_speed_m_s,commanded_speed_m_s,axial_travel_m,lateral_travel_m,maximum_abs_dq_rad_s,commanded_angular_speed_rad_s,measured_yaw_rad,residual_tz_Nm";
  for (int i = 0; i < 7; ++i) output << ",q" << i;
  for (int i = 0; i < 7; ++i) output << ",dq" << i;
  output << ",x,y,z\n" << std::setprecision(12);
  for (const auto& sample : samples) {
    output << sample.wall_time_ns << ',' << sample.time_s << ',' << sample.target_force_N << ','
           << sample.measured_delta_force_N << ','
           << sample.requested_speed_m_s << ',' << sample.commanded_speed_m_s
           << ',' << sample.axial_travel_m << ',' << sample.lateral_travel_m
           << ',' << sample.maximum_abs_dq_rad_s << ','
           << sample.commanded_angular_speed_rad_s << ','
           << sample.measured_yaw_rad << ',' << sample.residual_tz_Nm;
    for (double value : sample.q) output << ',' << value;
    for (double value : sample.dq) output << ',' << value;
    for (double value : sample.xyz) output << ',' << value;
    output << '\n';
  }
}

#ifdef FR3_AXIAL_ADMITTANCE_THREAD
struct ModelPoint { double q7 = 0.0; double tz = 0.0; };

std::vector<ModelPoint> load_model(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open dynamic model");
  std::string line;
  std::getline(input, line);
  std::vector<ModelPoint> model;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream stream(line);
    std::string q7, tz;
    if (!std::getline(stream, q7, ',') || !std::getline(stream, tz, ','))
      throw std::runtime_error("invalid dynamic model row");
    model.push_back({std::stod(q7), std::stod(tz)});
  }
  if (model.size() < 2) throw std::runtime_error("dynamic model is too short");
  for (std::size_t i = 1; i < model.size(); ++i)
    if (!(model[i].q7 > model[i - 1].q7))
      throw std::runtime_error("dynamic model q7 is not increasing");
  return model;
}

double expected_tz(const std::vector<ModelPoint>& model, double q7) {
  if (q7 <= model.front().q7) return model.front().tz;
  if (q7 >= model.back().q7) return model.back().tz;
  const auto upper = std::upper_bound(
      model.begin(), model.end(), q7,
      [](double value, const ModelPoint& point) { return value < point.q7; });
  const std::size_t hi = static_cast<std::size_t>(upper - model.begin());
  const std::size_t lo = hi - 1;
  const double fraction =
      (q7 - model[lo].q7) / (model[hi].q7 - model[lo].q7);
  return model[lo].tz + fraction * (model[hi].tz - model[lo].tz);
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
  constexpr int kExpectedArgc = 6;
#else
  constexpr int kExpectedArgc = 5;
#endif
  if (argc != kExpectedArgc) {
    std::cerr << "Usage: " << argv[0]
              << " read-only-preview|execute ROBOT_IP TELEMETRY_CSV CONFIRMATION"
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
              << " MODEL_CSV"
#endif
              << "\n";
    return 2;
  }
  const std::string mode(argv[1]);
  const bool execute = mode == "execute";
  if (!execute && mode != "read-only-preview") return 2;
  try {
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
    const auto dynamic_model = load_model(argv[5]);
#endif
    std::cout << "=== DIRECT LIBFRANKA "
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
              << "COMBINED AXIAL-ADMITTANCE THREAD DRIVE"
#else
              << "STATIONARY AXIAL ADMITTANCE"
#endif
              << " ===\n"
              << "mode:                     " << (execute ? "EXECUTE" : "READ_ONLY_PREVIEW") << "\n"
              << "robot IP:                 " << argv[2] << "\n"
              << "required setup:           upright, self-supported, head-held, unloaded bolt\n"
              << "approved force target:    +1.200360 N relative to live baseline\n"
              << "target ramp:              3.000 s\n"
              << "force-to-velocity gain:   "
              << bolt_axial_admittance::kForceToVelocityGainMSPerN * 1000.0
              << " mm/(s*N)\n"
              << "maximum axial speed:      0.500 mm/s\n"
              << "maximum axial/lateral:    "
              << bolt_axial_admittance::kMaximumAxialTravelM * 1000.0 << " / "
              << bolt_axial_admittance::kMaximumLateralTravelM * 1000.0 << " mm\n"
              << "hard axial cutoff:        +/-2.000 N\n"
              << "wrong-direction cutoff:   -0.500 N for 0.100 s\n"
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
              << "rotation:                 +" << cartesian_yaw::kTestAngleDeg
              << " deg after 3 s force acquisition\n"
              << "hard residual cutoff:     -0.538305 Nm\n"
              << "derived state timeout:    " << kDerivedStateTimeoutS << " s\n"
              << "Hand/ratchet/seated:      DISABLED\n"
#else
              << "rotation/Hand/threading:  DISABLED\n"
#endif
              << "collision/reflex settings:UNCHANGED\n"
              << "Press Enter to connect " << (execute ? "and perform preflight" : "read-only")
              << " (Ctrl+C cancels): ";
    std::string input;
    std::getline(std::cin, input);
    franka::Robot robot(argv[2]);
    robot.readOnce();
    franka::RobotState state = robot.readOnce();
    double count = 0.0, mean_fz = 0.0, m2_fz = 0.0, maximum_dq = 0.0;
    const auto baseline_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - baseline_start).count()
           < kBaselineDurationS) {
      state = robot.readOnce();
      const double fz = state.K_F_ext_hat_K[2];
      count += 1.0;
      const double delta = fz - mean_fz;
      mean_fz += delta / count;
      m2_fz += delta * (fz - mean_fz);
      for (double dq : state.dq) maximum_dq = std::max(maximum_dq, std::abs(dq));
    }
    const double std_fz = count > 1.0 ? std::sqrt(m2_fz / count) : 0.0;
    const Eigen::Matrix4d start_pose = pose_matrix(state.O_T_EE);
    const Eigen::Vector3d start_xyz = start_pose.topRightCorner<3, 1>();
    const Eigen::Matrix3d start_rotation = start_pose.topLeftCorner<3, 3>();
    const Eigen::Vector3d axis = start_pose.topLeftCorner<3, 3>().col(2).normalized();
    std::string blockers;
    if (!in_workspace(start_xyz)) blockers += "  - live TCP is outside the local workspace\n";
    if (!joints_within_guard(state.q)) blockers += "  - live joints violate guarded mechanical ranges\n";
    if (state.q[6] < kQ7CableMinRad || state.q[6] > kQ7CableMaxRad)
      blockers += "  - q7 is outside the cable-safe interval\n";
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
    if (state.q[6] < dynamic_model.front().q7 - kDynamicModelLowAllowanceRad ||
        state.q[6] + cartesian_yaw::kTestAngleRad +
                cartesian_yaw::kMaximumGuardStopExtraAngleRad >
            dynamic_model.back().q7)
      blockers += "  - insufficient calibrated dynamic-model q7 range for requested rotation\n";
#endif
    if (std_fz > kMaximumBaselineFzStdN) blockers += "  - force baseline is not stationary\n";
    if (maximum_dq > kMaximumBaselineDqRadS) blockers += "  - arm moved during baseline\n";
    std::cout << std::fixed << std::setprecision(6)
              << "\nlive xyz:                 " << start_xyz.transpose() << "\n"
              << "live +Kz in base:         " << axis.transpose() << "\n"
              << "live q7:                  " << state.q[6] << " rad\n"
              << "baseline F_Kz/std:        " << mean_fz << " / " << std_fz << " N\n"
              << "maximum baseline |dq|:    " << maximum_dq << " rad/s\n";
    if (!blockers.empty()) std::cout << "\nExecution blockers:\n" << blockers;
    if (!execute) {
      std::cout << "\nPreview complete. No control session, arm, Hand, or collision-setting command was issued.\n";
      return blockers.empty() ? 0 : 3;
    }
    if (!blockers.empty()) return 3;
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
    if (std::string(argv[4]) != "COMBINED_THREAD_AUTHORIZED") return 3;
    std::cout << "Keep one hand near emergency stop. Press Enter for the approved combined +"
              << cartesian_yaw::kTestAngleDeg << "-degree test: ";
#else
    if (std::string(argv[4]) != "AXIAL_ADMITTANCE_AUTHORIZED") return 3;
    std::cout << "Keep one hand near emergency stop. Press Enter for the approved stationary no-rotation test: ";
#endif
    std::getline(std::cin, input);

    std::signal(SIGINT, signal_handler);
    g_abort.store(false, std::memory_order_relaxed);
    std::vector<Sample> samples;
    samples.reserve(kMaximumSamples);
    double elapsed_s = 0.0;
    double commanded_speed = 0.0;
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
    double commanded_angular_speed = 0.0;
#endif
    double minimum_delta = 0.0, maximum_delta = 0.0;
    double maximum_axial = 0.0, maximum_lateral = 0.0;
    double minimum_residual_tz = std::numeric_limits<double>::infinity();
    double wrong_direction_since = -1.0;
    const char* stop_reason = nullptr;
    double zero_start_s = 0.0;
    int stationary_samples = 0;
    bool normal_finishing = false;
    bool hard_stopping = false;
    const auto control_start_wall_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto callback = [&](const franka::RobotState& current,
                        franka::Duration period) -> franka::CartesianVelocities {
      const double dt = period.toSec();
      elapsed_s += dt;
      const Eigen::Vector3d xyz =
          pose_matrix(current.O_T_EE).topRightCorner<3, 1>();
      const Eigen::Matrix3d current_rotation =
          pose_matrix(current.O_T_EE).topLeftCorner<3, 3>();
      const Eigen::Matrix3d relative_rotation =
          start_rotation.transpose() * current_rotation;
      const double measured_yaw =
          std::atan2(relative_rotation(1, 0), relative_rotation(0, 0));
      const Eigen::Vector3d displacement = xyz - start_xyz;
      const double axial_travel = axis.dot(displacement);
      const double lateral_travel = (displacement - axis * axial_travel).norm();
      const double measured_delta = current.K_F_ext_hat_K[2] - mean_fz;
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
      const double residual_tz =
          current.K_F_ext_hat_K[5] - expected_tz(dynamic_model, current.q[6]);
#else
      const double residual_tz = 0.0;
#endif
      minimum_residual_tz = std::min(minimum_residual_tz, residual_tz);
      minimum_delta = std::min(minimum_delta, measured_delta);
      maximum_delta = std::max(maximum_delta, measured_delta);
      maximum_axial = std::max(maximum_axial, std::abs(axial_travel));
      maximum_lateral = std::max(maximum_lateral, lateral_travel);

      if (measured_delta <= bolt_axial_admittance::kWrongDirectionDeltaN) {
        if (wrong_direction_since < 0.0) wrong_direction_since = elapsed_s;
      } else {
        wrong_direction_since = -1.0;
      }
      const char* abnormal_reason = nullptr;
      if (g_abort.load(std::memory_order_relaxed)) abnormal_reason = "operator_abort";
      else if (std::abs(measured_delta) >= bolt_axial_admittance::kHardAbsoluteAxialDeltaN)
        abnormal_reason = "hard_absolute_axial_force_cutoff";
      else if (wrong_direction_since >= 0.0 &&
               elapsed_s - wrong_direction_since >= bolt_axial_admittance::kWrongDirectionPersistenceS)
        abnormal_reason = "persistent_wrong_direction_force";
      else if (std::abs(axial_travel) >= bolt_axial_admittance::kMaximumAxialTravelM)
        abnormal_reason = "maximum_axial_travel";
      else if (lateral_travel >= bolt_axial_admittance::kMaximumLateralTravelM)
        abnormal_reason = "maximum_lateral_travel";
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
      else if (residual_tz <= kHardNegativeResidualTzNm)
        abnormal_reason = "hard_residual_torque_cutoff";
      else if (current.q[6] < dynamic_model.front().q7 - kDynamicModelLowAllowanceRad ||
               current.q[6] > dynamic_model.back().q7)
        abnormal_reason = "q7_dynamic_model_boundary";
#endif
      else if (!in_workspace(xyz)) abnormal_reason = "workspace_guard";
      else if (!joints_within_guard(current.q)) abnormal_reason = "joint_guard";
      else if (elapsed_s >= kDerivedStateTimeoutS)
        abnormal_reason = "state_timeout";

      if (abnormal_reason != nullptr) {
        stop_reason = abnormal_reason;
        normal_finishing = false;
        hard_stopping = true;
        commanded_speed = 0.0;
      } else if (hard_stopping) {
        commanded_speed = 0.0;
      } else if (elapsed_s >= kMotionCompletionS) {
        if (stop_reason == nullptr) stop_reason =
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
            "motion_completed";
#else
            "target_ramp_complete";
#endif
        normal_finishing = true;
        commanded_speed = bolt_axial_admittance::slew_speed(
            commanded_speed, 0.0, dt);
      } else {
        const double target = bolt_axial_admittance::target_force(elapsed_s);
        const double requested = bolt_axial_admittance::requested_speed(target, measured_delta);
        commanded_speed = bolt_axial_admittance::slew_speed(
            commanded_speed, requested, dt);
      }

      double maximum_live_dq = 0.0;
      for (double dq : current.dq) maximum_live_dq = std::max(maximum_live_dq, std::abs(dq));
      const double target = bolt_axial_admittance::target_force(elapsed_s);
      const double requested = normal_finishing || hard_stopping ? 0.0 :
          bolt_axial_admittance::requested_speed(target, measured_delta);
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
      const double rotation_elapsed =
          elapsed_s - bolt_axial_admittance::kTargetRampDurationS;
      const double requested_angular_speed =
          normal_finishing || hard_stopping
              ? 0.0
              : cartesian_yaw::angular_speed(rotation_elapsed);
      commanded_angular_speed = cartesian_yaw::slew_angular_speed(
          commanded_angular_speed, requested_angular_speed, dt);
#else
      const double commanded_angular_speed = 0.0;
#endif
      if (samples.size() < kMaximumSamples) {
        Sample sample;
        sample.wall_time_ns = control_start_wall_time_ns +
            static_cast<std::int64_t>(std::llround(elapsed_s * 1.0e9));
        sample.time_s = elapsed_s;
        sample.target_force_N = target;
        sample.measured_delta_force_N = measured_delta;
        sample.requested_speed_m_s = requested;
        sample.commanded_speed_m_s = commanded_speed;
        sample.axial_travel_m = axial_travel;
        sample.lateral_travel_m = lateral_travel;
        sample.maximum_abs_dq_rad_s = maximum_live_dq;
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
        sample.commanded_angular_speed_rad_s = commanded_angular_speed;
#else
        sample.commanded_angular_speed_rad_s = 0.0;
#endif
        sample.measured_yaw_rad = measured_yaw;
        sample.residual_tz_Nm = residual_tz;
        sample.q = current.q;
        sample.dq = current.dq;
        sample.xyz = {{xyz.x(), xyz.y(), xyz.z()}};
        samples.push_back(sample);
      } else if (stop_reason == nullptr) {
        stop_reason = "telemetry_capacity";
        commanded_speed = 0.0;
      }

      const bool zero_command = std::abs(commanded_speed) < 1e-8 &&
                                std::abs(commanded_angular_speed) < 1e-8;
      if (stop_reason != nullptr && zero_command) {
        if (zero_start_s == 0.0) zero_start_s = elapsed_s;
        if (maximum_live_dq < kStationaryDqThresholdRadS) ++stationary_samples;
        else stationary_samples = 0;
      }
      std::array<double, 6> command{{
          axis.x() * commanded_speed,
          axis.y() * commanded_speed,
          axis.z() * commanded_speed,
          axis.x() * commanded_angular_speed,
          axis.y() * commanded_angular_speed,
          axis.z() * commanded_angular_speed}};
      franka::CartesianVelocities output(command);
      if (zero_start_s > 0.0) {
        const double settling = elapsed_s - zero_start_s;
        if ((settling >= kMinimumZeroCommandSettleS &&
             stationary_samples >= kRequiredStationarySamples) ||
            settling >= kMaximumZeroCommandSettleS)
          return franka::MotionFinished(output);
      }
      return output;
    };

    try {
      robot.control(callback);
    } catch (const franka::Exception& error) {
      // Preserve the samples and the controller's guard reason even when
      // libfranka rejects or reflex-aborts the motion.  Previously this path
      // discarded the only evidence that could distinguish a force/position
      // guard from a command-generation fault.
      write_csv(argv[3], samples);
      std::cerr << "[exception] controller stop reason: "
                << (stop_reason ? stop_reason : "none_before_libfranka_exception")
                << '\n'
                << "[saved] " << argv[3] << '\n';
      throw;
    }
    const Eigen::Matrix3d final_rotation =
        pose_matrix(robot.readOnce().O_T_EE).topLeftCorner<3, 3>();
    const Eigen::Matrix3d final_relative =
        start_rotation.transpose() * final_rotation;
    const double final_yaw =
        std::atan2(final_relative(1, 0), final_relative(0, 0));
    write_csv(argv[3], samples);
    std::cout << "[result] stop reason:      " << (stop_reason ? stop_reason : "unknown") << "\n"
              << "[result] axial delta range:" << minimum_delta << " .. " << maximum_delta << " N\n"
              << "[result] max axial travel: " << maximum_axial * 1000.0 << " mm\n"
              << "[result] max lateral:      " << maximum_lateral * 1000.0 << " mm\n"
              << "[result] measured yaw:     " << final_yaw * 180.0 / M_PI << " deg\n"
              << "[result] minimum residual:" << minimum_residual_tz << " Nm\n"
              << "[saved] " << argv[3] << "\n"
#ifdef FR3_AXIAL_ADMITTANCE_THREAD
              << "The Hand remains closed. No ratchet, release, seated classification, camera, or collision-setting command was issued.\n";
#else
              << "The Hand remains closed. No rotation, release, ratchet, threading, or collision-setting command was issued.\n";
#endif
  } catch (const franka::Exception& error) {
    std::cerr << "libfranka error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
