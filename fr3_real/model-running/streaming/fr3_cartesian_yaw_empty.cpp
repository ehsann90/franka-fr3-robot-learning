#include "cartesian_yaw_commissioning.h"

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/robot.h>

#include <Eigen/Core>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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
constexpr double kMaximumBaselineFzStdN = 0.1;
constexpr double kMaximumBaselineDqRadS = 0.01;
constexpr double kJointMarginRad = 0.12;
constexpr std::array<double, 7> kJointMin{{
    -2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159}};
constexpr std::array<double, 7> kJointMax{{
    2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159}};
constexpr std::array<double, 3> kWorkspaceMin{{0.45, 0.25, 0.07}};
constexpr std::array<double, 3> kWorkspaceMax{{0.65, 0.38, 0.20}};
// 45-degree profile (41.1 s) + guarded settling tail at 1 kHz.
constexpr std::size_t kMaximumSamples = 60000;
#ifdef FR3_CARTESIAN_YAW_HELD
constexpr double kHardNegativeResidualTzNm = -0.5383051567216975;
constexpr double kDynamicModelLowAllowanceRad = 0.02;
#endif

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
  double time_s = 0.0;
  double commanded_speed_rad_s = 0.0;
  double measured_yaw_rad = 0.0;
  double translation_m = 0.0;
  double axial_delta_N = 0.0;
  double maximum_abs_dq_rad_s = 0.0;
  int stopping = 0;
  double residual_tz_Nm = 0.0;
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 3> xyz{};
};

void write_csv(const std::filesystem::path& path,
               const std::vector<Sample>& samples) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open telemetry output");
  output << "time_s,commanded_speed_rad_s,measured_yaw_rad,translation_m,axial_delta_N,maximum_abs_dq_rad_s,stopping,residual_tz_Nm";
  for (int i = 0; i < 7; ++i) output << ",q" << i;
  for (int i = 0; i < 7; ++i) output << ",dq" << i;
  output << ",x,y,z\n" << std::setprecision(12);
  for (const auto& sample : samples) {
    output << sample.time_s << ',' << sample.commanded_speed_rad_s << ','
           << sample.measured_yaw_rad << ',' << sample.translation_m << ','
           << sample.axial_delta_N << ',' << sample.maximum_abs_dq_rad_s << ','
           << sample.stopping << ',' << sample.residual_tz_Nm;
    for (double value : sample.q) output << ',' << value;
    for (double value : sample.dq) output << ',' << value;
    for (double value : sample.xyz) output << ',' << value;
    output << '\n';
  }
}

#ifdef FR3_CARTESIAN_YAW_HELD
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
#ifdef FR3_CARTESIAN_YAW_HELD
  constexpr int kExpectedArgc = 6;
#else
  constexpr int kExpectedArgc = 5;
#endif
  if (argc != kExpectedArgc) {
    std::cerr << "Usage: " << argv[0]
              << " read-only-preview|execute ROBOT_IP TELEMETRY_CSV CONFIRMATION"
#ifdef FR3_CARTESIAN_YAW_HELD
              << " MODEL_CSV"
#endif
              << "\n";
    return 2;
  }
  const std::string mode(argv[1]);
  const bool execute = mode == "execute";
  if (!execute && mode != "read-only-preview") return 2;
  try {
#ifdef FR3_CARTESIAN_YAW_HELD
    const auto dynamic_model = load_model(argv[5]);
#endif
    std::cout << "=== "
#ifdef FR3_CARTESIAN_YAW_HELD
              << "BOLT-HELD FREE CARTESIAN YAW COMMISSIONING"
#else
              << "EMPTY-HAND CARTESIAN VELOCITY YAW COMMISSIONING"
#endif
              << " ===\n"
              << "mode:                     " << (execute ? "EXECUTE" : "READ_ONLY_PREVIEW") << "\n"
              << "robot IP:                 " << argv[2] << "\n"
              << "required setup:           "
#ifdef FR3_CARTESIAN_YAW_HELD
              << "bolt upright, head-held, self-supported, NOT engaged, unloaded\n"
#else
              << "EMPTY HAND; NO BOLT; clear local path\n"
#endif
              << "motion generator:         official libfranka CartesianVelocities\n"
              << "command:                  +" << cartesian_yaw::kTestAngleDeg
              << " deg about live tool axis\n"
              << "profile/peak speed:       half-cosine / "
              << cartesian_yaw::kPeakAngularSpeedRadS << " rad/s\n"
              << "expected duration:        " << cartesian_yaw::kDurationS << " s\n"
              << "termination:              zero command + measured stationary hold\n"
              << "telemetry capacity:       60.0 s at 1 kHz\n"
              << "linear velocity:          EXACTLY ZERO\n"
              << "axial force command:      DISABLED\n"
#ifdef FR3_CARTESIAN_YAW_HELD
              << "hard residual cutoff:     -0.538305 Nm\n"
#else
              << "contact/threading:        DISABLED\n"
#endif
              << "max TCP translation:      0.300 mm\n"
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
    const Eigen::Vector3d axis = start_rotation.col(2).normalized();
    std::string blockers;
    if (!in_workspace(start_xyz)) blockers += "  - live TCP is outside the commissioning workspace\n";
    if (!joints_within_guard(state.q)) blockers += "  - live joints violate guarded mechanical ranges\n";
    if (state.q[6] < cartesian_yaw::kQ7CableMinRad ||
        state.q[6] > cartesian_yaw::kQ7CableMaxRad) blockers += "  - q7 is outside the cable-safe interval\n";
#ifdef FR3_CARTESIAN_YAW_HELD
    if (state.q[6] < dynamic_model.front().q7 - kDynamicModelLowAllowanceRad ||
        state.q[6] > dynamic_model.back().q7)
      blockers += "  - q7 is outside the calibrated dynamic torque model\n";
#endif
    if (std_fz > kMaximumBaselineFzStdN) blockers += "  - force baseline is not stationary\n";
    if (maximum_dq > kMaximumBaselineDqRadS) blockers += "  - arm moved during baseline\n";
    std::cout << std::fixed << std::setprecision(6)
              << "\nlive xyz:                 " << start_xyz.transpose() << "\n"
              << "live tool axis:           " << axis.transpose() << "\n"
              << "live q7:                  " << state.q[6] << " rad\n"
              << "baseline F_Kz/std:        " << mean_fz << " / " << std_fz << " N\n"
              << "maximum baseline |dq|:    " << maximum_dq << " rad/s\n";
    if (!blockers.empty()) std::cout << "\nExecution blockers:\n" << blockers;
    if (!execute) {
      std::cout << "\nPreview complete. No control session, arm, Hand, or collision-setting command was issued.\n";
      return blockers.empty() ? 0 : 3;
    }
    if (!blockers.empty()) return 3;
#ifdef FR3_CARTESIAN_YAW_HELD
    if (std::string(argv[4]) != "HELD_CARTESIAN_YAW_AUTHORIZED") return 3;
    std::cout << "Keep one hand near emergency stop. Press Enter for the bolt-held +"
              << cartesian_yaw::kTestAngleDeg << "-degree free-yaw test: ";
#else
    if (std::string(argv[4]) != "EMPTY_CARTESIAN_YAW_AUTHORIZED") return 3;
    std::cout << "Keep one hand near emergency stop. Press Enter for the empty-Hand +"
              << cartesian_yaw::kTestAngleDeg << "-degree test: ";
#endif
    std::getline(std::cin, input);
    std::signal(SIGINT, signal_handler);
    g_abort.store(false, std::memory_order_relaxed);
    std::vector<Sample> samples;
    samples.reserve(kMaximumSamples);
    double elapsed_s = 0.0;
    double maximum_translation = 0.0;
    double maximum_abs_axial_delta = 0.0;
    double minimum_residual_tz = std::numeric_limits<double>::infinity();
    const char* requested_stop_reason = nullptr;
    double stop_request_time_s = 0.0;
    int stationary_samples = 0;
    auto callback = [&](const franka::RobotState& current,
                        franka::Duration period) -> franka::CartesianVelocities {
      elapsed_s += period.toSec();
      const Eigen::Matrix4d pose = pose_matrix(current.O_T_EE);
      const Eigen::Vector3d xyz = pose.topRightCorner<3, 1>();
      const Eigen::Matrix3d relative = start_rotation.transpose() * pose.topLeftCorner<3, 3>();
      const double measured_yaw = std::atan2(relative(1, 0), relative(0, 0));
      const double translation = (xyz - start_xyz).norm();
      const double axial_delta = current.K_F_ext_hat_K[2] - mean_fz;
#ifdef FR3_CARTESIAN_YAW_HELD
      const double residual_tz =
          current.K_F_ext_hat_K[5] - expected_tz(dynamic_model, current.q[6]);
#else
      const double residual_tz = 0.0;
#endif
      maximum_translation = std::max(maximum_translation, translation);
      maximum_abs_axial_delta = std::max(maximum_abs_axial_delta, std::abs(axial_delta));
      minimum_residual_tz = std::min(minimum_residual_tz, residual_tz);
      if (g_abort.load(std::memory_order_relaxed) && requested_stop_reason == nullptr)
        requested_stop_reason = "operator_abort";
      if (translation >= cartesian_yaw::kMaximumTranslationM && requested_stop_reason == nullptr)
        requested_stop_reason = "maximum_translation";
      if (std::abs(axial_delta) >= cartesian_yaw::kHardAbsoluteAxialDeltaN && requested_stop_reason == nullptr)
        requested_stop_reason = "hard_absolute_axial_force_cutoff";
#ifdef FR3_CARTESIAN_YAW_HELD
      if (residual_tz <= kHardNegativeResidualTzNm && requested_stop_reason == nullptr)
        requested_stop_reason = "hard_residual_torque_cutoff";
#endif
      if (!in_workspace(xyz) && requested_stop_reason == nullptr) requested_stop_reason = "workspace_guard";
      if (!joints_within_guard(current.q) && requested_stop_reason == nullptr) requested_stop_reason = "joint_guard";
      if ((current.q[6] < cartesian_yaw::kQ7CableMinRad ||
           current.q[6] > cartesian_yaw::kQ7CableMaxRad) && requested_stop_reason == nullptr)
        requested_stop_reason = "q7_cable_limit";
#ifdef FR3_CARTESIAN_YAW_HELD
      if ((current.q[6] < dynamic_model.front().q7 - kDynamicModelLowAllowanceRad ||
           current.q[6] > dynamic_model.back().q7) && requested_stop_reason == nullptr)
        requested_stop_reason = "q7_dynamic_model_boundary";
#endif
      if (elapsed_s >= cartesian_yaw::kDurationS && requested_stop_reason == nullptr)
        requested_stop_reason = "motion_completed";
      if (requested_stop_reason != nullptr && stop_request_time_s == 0.0)
        stop_request_time_s = elapsed_s;
      double maximum_live_dq = 0.0;
      for (double dq : current.dq) maximum_live_dq = std::max(maximum_live_dq, std::abs(dq));
      const bool stopping = requested_stop_reason != nullptr;
      if (stopping && maximum_live_dq < cartesian_yaw::kStationaryDqThresholdRadS)
        ++stationary_samples;
      else if (stopping)
        stationary_samples = 0;
      const double speed = stopping ? 0.0 : cartesian_yaw::angular_speed(elapsed_s);
      if (samples.size() < kMaximumSamples) {
        Sample sample;
        sample.time_s = elapsed_s;
        sample.commanded_speed_rad_s = speed;
        sample.measured_yaw_rad = measured_yaw;
        sample.translation_m = translation;
        sample.axial_delta_N = axial_delta;
        sample.maximum_abs_dq_rad_s = maximum_live_dq;
        sample.stopping = stopping ? 1 : 0;
        sample.residual_tz_Nm = residual_tz;
        sample.q = current.q;
        sample.dq = current.dq;
        sample.xyz = {{xyz.x(), xyz.y(), xyz.z()}};
        samples.push_back(sample);
      }
      else if (requested_stop_reason == nullptr)
        requested_stop_reason = "telemetry_capacity";
      std::array<double, 6> command{{0.0, 0.0, 0.0,
          axis.x() * speed, axis.y() * speed, axis.z() * speed}};
      franka::CartesianVelocities output(command);
      const double settle_elapsed_s = stopping ? elapsed_s - stop_request_time_s : 0.0;
      if (stopping && cartesian_yaw::settle_complete(
                          settle_elapsed_s, stationary_samples))
        return franka::MotionFinished(output);
      if (stopping && settle_elapsed_s >= cartesian_yaw::kMaximumZeroCommandSettleS)
        return franka::MotionFinished(output);
      return output;
    };
    robot.control(callback);
    const auto final_state = robot.readOnce();
    const Eigen::Matrix3d final_rotation = pose_matrix(final_state.O_T_EE).topLeftCorner<3, 3>();
    const Eigen::Matrix3d final_relative = start_rotation.transpose() * final_rotation;
    const double final_yaw = std::atan2(final_relative(1, 0), final_relative(0, 0));
    write_csv(argv[3], samples);
    std::cout << "[result] stop reason:      " << (requested_stop_reason ? requested_stop_reason : "unknown") << "\n"
              << "[result] measured yaw:     " << final_yaw * 180.0 / cartesian_yaw::kPi << " deg\n"
              << "[result] max translation: " << maximum_translation * 1000.0 << " mm\n"
              << "[result] max axial delta: " << maximum_abs_axial_delta << " N\n"
              << "[result] minimum residual:" << minimum_residual_tz << " Nm\n"
              << "[saved] " << argv[3] << "\n"
#ifdef FR3_CARTESIAN_YAW_HELD
              << "The Hand remains closed. No axial-force, release, ratchet, threading, or collision-setting command was issued.\n";
#else
              << "No bolt, Hand, axial-force, contact, threading, or collision-setting command was issued.\n";
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
