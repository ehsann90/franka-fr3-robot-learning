#include "bolt_axial_preload_control.h"

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

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
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t kMaximumSamples = 5000;
constexpr double kBaselineDurationS = 3.0;
constexpr double kMaximumBaselineFzStdN = 0.1;
constexpr double kMaximumBaselineDqRadS = 0.01;
constexpr double kJointMarginRad = 0.12;
constexpr std::array<double, 7> kJointMin{
    {-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159}};
constexpr std::array<double, 7> kJointMax{
    {2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159}};
constexpr std::array<double, 3> kWorkspaceMin{{0.57, 0.30, 0.08}};
constexpr std::array<double, 3> kWorkspaceMax{{0.63, 0.35, 0.15}};
constexpr std::array<double, 3> kThreadReadyReferenceXyz{
    {0.5941215006415239, 0.32814065294113465, 0.102263430055387}};
constexpr std::array<double, 3> kThreadReadyReferenceAxis{
    {-0.039941575876372065, 0.023253877017044736, -0.9989313932949911}};
constexpr double kMaximumStartAxialErrorM = 0.003;
constexpr double kMaximumStartLateralErrorM = 0.005;
constexpr double kMaximumStartToolAxisTiltRad = 5.0 * M_PI / 180.0;

std::atomic<bool> g_abort{false};
void signal_handler(int) { g_abort.store(true, std::memory_order_relaxed); }

struct Telemetry {
  double time_s = 0.0;
  double approved_measured_target_N = 0.0;
  double commanded_force_N = 0.0;
  double measured_delta_fz_N = 0.0;
  double axial_travel_m = 0.0;
  double lateral_travel_m = 0.0;
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 7> tau_command{};
  std::array<double, 6> wrench_K{};
  std::array<double, 3> xyz{};
};

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

void write_csv(const std::filesystem::path& path,
               const std::array<Telemetry, kMaximumSamples>& samples,
               std::size_t count) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open telemetry output: " + path.string());
  output << "time_s,approved_measured_target_N,commanded_force_N,measured_delta_fz_N,axial_travel_m,"
            "lateral_travel_m";
  for (int i = 0; i < 7; ++i) output << ",q" << i;
  for (int i = 0; i < 7; ++i) output << ",dq" << i;
  for (int i = 0; i < 7; ++i) output << ",tau_command" << i;
  for (int i = 0; i < 6; ++i) output << ",K_wrench" << i;
  output << ",x,y,z\n";
  output << std::setprecision(12);
  for (std::size_t row = 0; row < count; ++row) {
    const auto& sample = samples[row];
    output << sample.time_s << ',' << sample.approved_measured_target_N << ','
           << sample.commanded_force_N << ','
           << sample.measured_delta_fz_N << ',' << sample.axial_travel_m << ','
           << sample.lateral_travel_m;
    for (double value : sample.q) output << ',' << value;
    for (double value : sample.dq) output << ',' << value;
    for (double value : sample.tau_command) output << ',' << value;
    for (double value : sample.wrench_K) output << ',' << value;
    for (double value : sample.xyz) output << ',' << value;
    output << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " read-only-preview|execute ROBOT_IP [TELEMETRY_CSV]" << std::endl;
    return 2;
  }
  const std::string mode(argv[1]);
  const bool execute = mode == "execute";
  if (!execute && mode != "read-only-preview") {
    std::cerr << "mode must be read-only-preview or execute" << std::endl;
    return 2;
  }
  if (execute && argc != 4) {
    std::cerr << "execute mode requires a telemetry CSV path" << std::endl;
    return 2;
  }

  try {
    std::cout << "=== DIRECT LIBFRANKA BOLT AXIAL PRELOAD ===\n"
              << "mode:                     " << (execute ? "EXECUTE" : "READ_ONLY_PREVIEW") << "\n"
              << "robot IP:                 " << argv[2] << "\n"
              << "linked controller:        repository libfranka 0.21.2 path\n"
              << "approved measured target: +" << std::fixed << std::setprecision(6)
              << bolt_preload::kTargetAxialForceN << " N along live +Kz\n"
              << "command schedule:         0 -> +" << bolt_preload::kMaximumCommandedForceN
              << " N at " << bolt_preload::kMaximumCommandSlewNs << " N/s\n"
              << "ceiling hold duration:    " << bolt_preload::kHoldDurationS << " s\n"
              << "total test duration:      " << bolt_preload::kTestDurationS << " s\n"
              << "force feedback:           DISABLED; Kp=" << bolt_preload::kProportionalGain
              << "; Ki=" << bolt_preload::kIntegralGain << "\n"
              << "axial stiffness/damping:  0 / 0\n"
              << "lateral/rotation hold:    " << bolt_preload::kLateralStiffnessNm << " N/m / "
              << bolt_preload::kRotationalStiffnessNmRad << " Nm/rad\n"
              << "hard axial cutoff:        +/-" << bolt_preload::kHardAbsoluteAxialDeltaN << " N\n"
              << "max axial/lateral travel: " << bolt_preload::kMaximumAxialTravelM * 1000.0
              << " / " << bolt_preload::kMaximumLateralTravelM * 1000.0 << " mm\n"
              << "collision/reflex settings:UNCHANGED\n"
              << "rotation/threading/Hand:   DISABLED\n"
              << "Press Enter to connect " << (execute ? "and perform preflight" : "read-only")
              << " (Ctrl+C cancels): ";
    std::string input;
    std::getline(std::cin, input);

    franka::Robot robot(argv[2]);
    robot.readOnce();
    franka::RobotState state = robot.readOnce();
    franka::Model model = robot.loadModel();

    double count = 0.0;
    double mean_fz = 0.0;
    double m2_fz = 0.0;
    double maximum_dq = 0.0;
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

    const Eigen::Matrix4d O_T_EE = pose_matrix(state.O_T_EE);
    const Eigen::Matrix4d EE_T_K = pose_matrix(state.EE_T_K);
    const Eigen::Matrix4d O_T_K = O_T_EE * EE_T_K;
    const Eigen::Vector3d axis = O_T_K.topLeftCorner<3, 3>().col(2).normalized();
    const Eigen::Vector3d xyz = O_T_EE.topRightCorner<3, 1>();
    const Eigen::Vector3d reference_xyz(kThreadReadyReferenceXyz.data());
    const Eigen::Vector3d reference_axis =
        Eigen::Vector3d(kThreadReadyReferenceAxis.data()).normalized();
    const Eigen::Vector3d start_error = xyz - reference_xyz;
    const double start_axial_error = axis.dot(start_error);
    const double start_lateral_error =
        bolt_preload::lateral_component(start_error, axis).norm();
    const double start_tool_axis_tilt =
        std::acos(std::clamp(reference_axis.dot(axis), -1.0, 1.0));
    const double ee_t_k_identity_error =
        (EE_T_K - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff();
    const auto jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    Eigen::Matrix<double, 6, 1> peak_wrench = Eigen::Matrix<double, 6, 1>::Zero();
    peak_wrench.head<3>() = axis * bolt_preload::kMaximumCommandedForceN;
    const Eigen::Matrix<double, 7, 1> peak_task_tau = jacobian.transpose() * peak_wrench;

    std::cout << "\nlive xyz:                 " << xyz.transpose() << "\n"
              << "live +Kz in base:         " << axis.transpose() << "\n"
              << "baseline F_Kz:            " << mean_fz << " N (std=" << std_fz << ")\n"
              << "maximum baseline |dq|:    " << maximum_dq << " rad/s\n"
              << "THREAD_READY error:       axial " << start_axial_error * 1000.0
              << " mm; lateral " << start_lateral_error * 1000.0
              << " mm; tilt " << start_tool_axis_tilt * 180.0 / M_PI << " deg\n"
              << "EE_T_K identity error:    " << ee_t_k_identity_error << "\n"
              << "Jacobian row norms:       " << jacobian.rowwise().norm().transpose() << "\n"
              << "ceiling task torque:      " << peak_task_tau.transpose() << " Nm\n"
              << "ceiling torque norm:      " << peak_task_tau.norm() << " Nm\n";

    std::string blockers;
    if (!in_workspace(xyz)) blockers += "  - live TCP is outside local workspace\n";
    if (!joints_within_guard(state.q)) blockers += "  - live joints violate guarded ranges\n";
    if (std::abs(start_axial_error) > kMaximumStartAxialErrorM)
      blockers += "  - live axial position is too far from THREAD_READY\n";
    if (start_lateral_error > kMaximumStartLateralErrorM)
      blockers += "  - live lateral position is too far from THREAD_READY\n";
    if (start_tool_axis_tilt > kMaximumStartToolAxisTiltRad)
      blockers += "  - live tool axis is too tilted from THREAD_READY\n";
    if (ee_t_k_identity_error > 1e-8)
      blockers += "  - EE_T_K is not identity; direct wrench mapping is not commissioned\n";
    if (std_fz > kMaximumBaselineFzStdN) blockers += "  - baseline axial force is not stationary\n";
    if (maximum_dq > kMaximumBaselineDqRadS) blockers += "  - arm moved during baseline\n";
    if (!jacobian.allFinite() || jacobian.cwiseAbs().maxCoeff() < 0.1)
      blockers += "  - direct libfranka Jacobian is invalid\n";
    if (!peak_task_tau.allFinite() || peak_task_tau.norm() < 0.1)
      blockers += "  - peak task torque is invalid\n";
    if (!blockers.empty()) std::cout << "\nExecution blockers:\n" << blockers;

    if (!execute) {
      std::cout << "\nPreview complete. No control session, motion, Hand, or collision-setting command was issued.\n";
      return blockers.empty() ? 0 : 3;
    }
    if (!blockers.empty()) {
      std::cerr << "[SAFETY] direct preload execution refused" << std::endl;
      return 3;
    }
    std::cout << "Type RUN_DIRECT_AXIAL_RAMP to execute the approved stationary test; anything else aborts: ";
    std::getline(std::cin, input);
    if (input != "RUN_DIRECT_AXIAL_RAMP") {
      std::cout << "Aborted. No control session was started." << std::endl;
      return 0;
    }

    std::signal(SIGINT, signal_handler);
    const Eigen::Vector3d start_xyz = xyz;
    const Eigen::Matrix3d desired_rotation = O_T_EE.topLeftCorner<3, 3>();
    bolt_preload::Guard guard;
    guard.baseline_fz_N = mean_fz;
    guard.start_xyz = start_xyz;
    guard.axis = axis;
    std::array<Telemetry, kMaximumSamples> telemetry{};
    std::size_t telemetry_count = 0;
    double elapsed_s = 0.0;

    auto callback = [&](const franka::RobotState& current,
                        franka::Duration period) -> franka::Torques {
      elapsed_s += period.toSec();
      const Eigen::Matrix4d current_pose = pose_matrix(current.O_T_EE);
      const Eigen::Vector3d current_xyz = current_pose.topRightCorner<3, 1>();
      guard.update(elapsed_s, current.K_F_ext_hat_K[2], current_xyz);
      if (g_abort.load(std::memory_order_relaxed) && guard.stop_reason.empty())
        guard.stop_reason = "operator_abort";
      if (!in_workspace(current_xyz) && guard.stop_reason.empty())
        guard.stop_reason = "workspace_guard";
      if (!joints_within_guard(current.q) && guard.stop_reason.empty())
        guard.stop_reason = "joint_guard";

      const auto current_jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, current);
      const auto coriolis_array = model.coriolis(current);
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> current_jacobian(
          current_jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(current.dq.data());
      const Eigen::Matrix<double, 6, 1> twist = current_jacobian * dq;
      const Eigen::Vector3d position_error = current_xyz - start_xyz;
      const Eigen::Vector3d lateral_error =
          bolt_preload::lateral_component(position_error, axis);
      const Eigen::Vector3d lateral_velocity =
          bolt_preload::lateral_component(twist.head<3>(), axis);
      const Eigen::Vector3d rotation_error = bolt_preload::orientation_error(
          desired_rotation, current_pose.topLeftCorner<3, 3>());

      Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
      const double measured_force_delta_N = current.K_F_ext_hat_K[2] - mean_fz;
      const double force_command_N =
          bolt_preload::ceiling_characterization_force(elapsed_s);
      if (!std::isfinite(force_command_N) && guard.stop_reason.empty())
        guard.stop_reason = "invalid_force_command";
      wrench.head<3>() = axis * force_command_N
          - bolt_preload::kLateralStiffnessNm * lateral_error
          - bolt_preload::kLateralDampingNsM * lateral_velocity;
      wrench.tail<3>() = -bolt_preload::kRotationalStiffnessNmRad * rotation_error
          - bolt_preload::kRotationalDampingNmsRad * twist.tail<3>();
      Eigen::Matrix<double, 7, 1> tau_desired = current_jacobian.transpose() * wrench + coriolis;
      std::array<double, 7> tau_desired_array{};
      Eigen::Map<Eigen::Matrix<double, 7, 1>>(tau_desired_array.data()) = tau_desired;
      const std::array<double, 7> tau_limited =
          franka::limitRate(franka::kMaxTorqueRate, tau_desired_array, current.tau_J_d);

      if (telemetry_count < telemetry.size()) {
        auto& sample = telemetry[telemetry_count++];
        sample.time_s = elapsed_s;
        sample.approved_measured_target_N = bolt_preload::kTargetAxialForceN;
        sample.commanded_force_N = force_command_N;
        sample.measured_delta_fz_N = measured_force_delta_N;
        const Eigen::Vector3d displacement = current_xyz - start_xyz;
        sample.axial_travel_m = axis.dot(displacement);
        sample.lateral_travel_m = bolt_preload::lateral_component(displacement, axis).norm();
        sample.q = current.q;
        sample.dq = current.dq;
        sample.tau_command = tau_limited;
        sample.wrench_K = current.K_F_ext_hat_K;
        sample.xyz = {{current_xyz.x(), current_xyz.y(), current_xyz.z()}};
      } else if (guard.stop_reason.empty()) {
        guard.stop_reason = "telemetry_capacity";
      }

      franka::Torques command(tau_limited);
      if (!guard.stop_reason.empty()) return franka::MotionFinished(command);
      return command;
    };

    robot.control(callback);
    write_csv(argv[3], telemetry, telemetry_count);
    std::cout << "[result] stop reason:      " << guard.stop_reason << "\n"
              << "[result] axial delta range:" << guard.min_axial_delta_N << " .. "
              << guard.max_axial_delta_N << " N\n"
              << "[result] max axial travel: " << guard.max_axial_travel_m * 1000.0 << " mm\n"
              << "[result] max lateral:      " << guard.max_lateral_travel_m * 1000.0 << " mm\n"
              << "[saved] " << argv[3] << "\n"
              << "No rotation, threading, Hand, or collision-setting command was issued.\n";
  } catch (const franka::Exception& error) {
    std::cerr << "libfranka error: " << error.what() << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
