#include "bolt_thread_drive_control.h"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMaximumSamples = 35000;
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
constexpr double kMaximumStartToolAxisTiltRad =
    5.0 * bolt_thread_drive::kPi / 180.0;

std::atomic<bool> g_abort{false};
void signal_handler(int) { g_abort.store(true, std::memory_order_relaxed); }

struct Telemetry {
  double time_s = 0.0;
  double desired_angle_rad = 0.0;
  double desired_angular_speed_rad_s = 0.0;
  double commanded_axial_force_N = 0.0;
  double measured_axial_delta_N = 0.0;
  double expected_K_Tz_Nm = 0.0;
  double residual_K_Tz_Nm = 0.0;
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

std::vector<double> parse_csv_numbers(const std::string& text) {
  std::vector<double> values;
  std::istringstream input(text);
  std::string token;
  while (std::getline(input, token, ',')) {
    if (token.empty()) throw std::runtime_error("empty dynamic-model value");
    std::size_t consumed = 0;
    const double value = std::stod(token, &consumed);
    if (consumed != token.size() || !std::isfinite(value)) {
      throw std::runtime_error("invalid dynamic-model value");
    }
    values.push_back(value);
  }
  return values;
}

bool valid_dynamic_model(const std::vector<double>& q7,
                         const std::vector<double>& tz) {
  if (q7.size() != tz.size() || q7.size() < 2) return false;
  for (std::size_t index = 0; index < q7.size(); ++index) {
    if (!std::isfinite(q7[index]) || !std::isfinite(tz[index])) return false;
    if (index > 0 && q7[index] <= q7[index - 1]) return false;
  }
  return true;
}

bool in_workspace(const Eigen::Vector3d& xyz) {
  for (Eigen::Index index = 0; index < 3; ++index) {
    if (xyz(index) < kWorkspaceMin[static_cast<std::size_t>(index)] ||
        xyz(index) > kWorkspaceMax[static_cast<std::size_t>(index)]) return false;
  }
  return true;
}

bool joints_within_guard(const std::array<double, 7>& q) {
  for (std::size_t index = 0; index < q.size(); ++index) {
    if (q[index] <= kJointMin[index] + kJointMarginRad ||
        q[index] >= kJointMax[index] - kJointMarginRad) return false;
  }
  return true;
}

void write_csv(const std::filesystem::path& path,
               const std::vector<Telemetry>& samples, std::size_t count) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open telemetry output: " + path.string());
  output << "time_s,desired_angle_rad,desired_angular_speed_rad_s,commanded_axial_force_N,"
            "measured_axial_delta_N,expected_K_Tz_Nm,residual_K_Tz_Nm,axial_travel_m,"
            "lateral_travel_m";
  for (int index = 0; index < 7; ++index) output << ",q" << index;
  for (int index = 0; index < 7; ++index) output << ",dq" << index;
  for (int index = 0; index < 7; ++index) output << ",tau_command" << index;
  for (int index = 0; index < 6; ++index) output << ",K_wrench" << index;
  output << ",x,y,z\n" << std::setprecision(12);
  for (std::size_t row = 0; row < count; ++row) {
    const auto& sample = samples[row];
    output << sample.time_s << ',' << sample.desired_angle_rad << ','
           << sample.desired_angular_speed_rad_s << ','
           << sample.commanded_axial_force_N << ','
           << sample.measured_axial_delta_N << ',' << sample.expected_K_Tz_Nm << ','
           << sample.residual_K_Tz_Nm << ',' << sample.axial_travel_m << ','
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
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " read-only-preview|execute ROBOT_IP TELEMETRY_CSV MODEL_Q7_CSV MODEL_TZ_CSV\n";
    return 2;
  }
  const std::string mode(argv[1]);
  const bool execute = mode == "execute";
  if (!execute && mode != "read-only-preview") {
    std::cerr << "mode must be read-only-preview or execute\n";
    return 2;
  }

  try {
    const std::vector<double> model_q7 = parse_csv_numbers(argv[4]);
    const std::vector<double> model_tz = parse_csv_numbers(argv[5]);
    if (!valid_dynamic_model(model_q7, model_tz)) {
      throw std::runtime_error("dynamic torque model is invalid");
    }

    std::cout << "=== DIRECT LIBFRANKA GUARDED BOLT THREAD DRIVE ===\n"
              << "mode:                     " << (execute ? "EXECUTE" : "READ_ONLY_PREVIEW") << "\n"
              << "robot IP:                 " << argv[2] << "\n"
              << "linked controller:        repository libfranka 0.21.2 path\n"
              << "planned rotation/profile: +89.0 deg / half-cosine\n"
              << "peak angular speed:       " << bolt_thread_drive::kPeakAngularSpeedRadS << " rad/s\n"
              << "planned duration:         " << bolt_thread_drive::kDriveDurationS << " s\n"
              << "axial command:            0 -> " << bolt_thread_drive::kForceCommandCeilingN
              << " N at " << bolt_thread_drive::kForceCommandSlewNs << " N/s\n"
              << "hard axial cutoff:        +/-" << bolt_thread_drive::kHardAbsoluteAxialDeltaN << " N\n"
              << "hard residual cutoff:     " << bolt_thread_drive::kHardNegativeResidualTzNm << " Nm\n"
              << "q7 cable-safe interval:   " << bolt_thread_drive::kQ7CableMinRad << " .. "
              << bolt_thread_drive::kQ7CableMaxRad << " rad\n"
              << "dynamic model interval:   " << model_q7.front() << " .. " << model_q7.back() << " rad\n"
              << "automatic ratchet/Hand:   DISABLED\n"
              << "collision/reflex settings:UNCHANGED\n"
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
        bolt_thread_drive::lateral_component(start_error, axis).norm();
    const double start_tool_axis_tilt =
        std::acos(std::clamp(reference_axis.dot(axis), -1.0, 1.0));
    const double ee_t_k_identity_error =
        (EE_T_K - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff();
    const double target_q7 = state.q[6] + bolt_thread_drive::kDriveAngleRad;
    const auto jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    Eigen::Matrix<double, 6, 1> envelope_wrench = Eigen::Matrix<double, 6, 1>::Zero();
    envelope_wrench.head<3>() = axis * bolt_thread_drive::kForceCommandCeilingN;
    envelope_wrench.tail<3>() = axis *
        (bolt_thread_drive::kRotationalDampingNmsRad *
         bolt_thread_drive::kPeakAngularSpeedRadS);
    const Eigen::Matrix<double, 7, 1> envelope_tau =
        jacobian.transpose() * envelope_wrench;

    std::cout << std::fixed << std::setprecision(6)
              << "\nlive/target q7:           " << state.q[6] << " / " << target_q7 << " rad\n"
              << "live xyz:                 " << xyz.transpose() << "\n"
              << "live +Kz in base:         " << axis.transpose() << "\n"
              << "baseline F_Kz:            " << mean_fz << " N (std=" << std_fz << ")\n"
              << "maximum baseline |dq|:    " << maximum_dq << " rad/s\n"
              << "THREAD_READY error:       axial " << start_axial_error * 1000.0
              << " mm; lateral " << start_lateral_error * 1000.0
              << " mm; tilt " << start_tool_axis_tilt * 180.0 / bolt_thread_drive::kPi << " deg\n"
              << "EE_T_K identity error:    " << ee_t_k_identity_error << "\n"
              << "Jacobian row norms:       " << jacobian.rowwise().norm().transpose() << "\n"
              << "planned wrench envelope:  " << envelope_wrench.transpose() << " [N,Nm]\n"
              << "planned task-tau envelope:" << envelope_tau.transpose() << " Nm\n";

    std::string blockers;
    if (!in_workspace(xyz)) blockers += "  - live TCP is outside local workspace\n";
    if (!joints_within_guard(state.q)) blockers += "  - live joints violate guarded ranges\n";
    if (std::abs(state.q[6] - bolt_thread_drive::kCommissionedStartQ7Rad) >
        bolt_thread_drive::kStartQ7ToleranceRad)
      blockers += "  - live q7 is not near the commissioned drive start\n";
    if (target_q7 > bolt_thread_drive::kQ7CableMaxRad)
      blockers += "  - target q7 leaves the cable-safe interval\n";
    if (state.q[6] < model_q7.front() - bolt_thread_drive::kDynamicModelLowEndpointAllowanceRad ||
        target_q7 > model_q7.back())
      blockers += "  - planned q7 interval leaves the calibrated dynamic model\n";
    if (std::abs(start_axial_error) > kMaximumStartAxialErrorM)
      blockers += "  - live axial position is too far from THREAD_READY\n";
    if (start_lateral_error > kMaximumStartLateralErrorM)
      blockers += "  - live lateral position is too far from THREAD_READY\n";
    if (start_tool_axis_tilt > kMaximumStartToolAxisTiltRad)
      blockers += "  - live tool axis is too tilted from THREAD_READY\n";
    if (ee_t_k_identity_error > 1e-8)
      blockers += "  - EE_T_K is not identity; wrench mapping is not commissioned\n";
    if (std_fz > kMaximumBaselineFzStdN)
      blockers += "  - baseline axial force is not stationary\n";
    if (maximum_dq > kMaximumBaselineDqRadS)
      blockers += "  - arm moved during baseline\n";
    if (!jacobian.allFinite() || jacobian.cwiseAbs().maxCoeff() < 0.1)
      blockers += "  - direct libfranka Jacobian is invalid\n";
    if (!envelope_tau.allFinite())
      blockers += "  - planned task-torque envelope is invalid\n";
    if (!blockers.empty()) std::cout << "\nExecution blockers:\n" << blockers;

    if (!execute) {
      std::cout << "\nPreview complete. No control session, arm, Hand, or collision-setting command was issued.\n";
      return blockers.empty() ? 0 : 3;
    }
    if (!blockers.empty()) {
      std::cerr << "[SAFETY] guarded thread drive refused\n";
      return 3;
    }
    std::cout << "Keep one hand near the emergency stop. Press Enter to execute the one authorized "
                 "89-degree drive; Ctrl+C aborts: ";
    std::getline(std::cin, input);

    std::signal(SIGINT, signal_handler);
    g_abort.store(false, std::memory_order_relaxed);
    const Eigen::Vector3d start_xyz = xyz;
    const Eigen::Matrix3d start_rotation = O_T_EE.topLeftCorner<3, 3>();
    bolt_thread_drive::Guard guard;
    guard.baseline_fz_N = mean_fz;
    guard.start_xyz = start_xyz;
    guard.axis = axis;
    guard.start_q = state.q;
    guard.target_q7_rad = target_q7;
    guard.dynamic_model_min_q7_rad = model_q7.front();
    guard.dynamic_model_max_q7_rad = model_q7.back();
    std::vector<Telemetry> telemetry(kMaximumSamples);
    std::size_t telemetry_count = 0;
    double elapsed_s = 0.0;

    auto callback = [&](const franka::RobotState& current,
                        franka::Duration period) -> franka::Torques {
      elapsed_s += period.toSec();
      const Eigen::Matrix4d current_pose = pose_matrix(current.O_T_EE);
      const Eigen::Vector3d current_xyz = current_pose.topRightCorner<3, 1>();
      const Eigen::Vector3d current_axis =
          current_pose.topLeftCorner<3, 3>().col(2).normalized();
      const double expected_tz = bolt_thread_drive::interpolate(
          model_q7, model_tz, current.q[6]);
      const double residual_tz = current.K_F_ext_hat_K[5] - expected_tz;
      guard.update(elapsed_s, current.q, current.K_F_ext_hat_K[2], residual_tz,
                   current_xyz, current_axis);
      if (g_abort.load(std::memory_order_relaxed) && guard.stop_reason == nullptr)
        guard.stop_reason = "operator_abort";
      if (!in_workspace(current_xyz) && guard.stop_reason == nullptr)
        guard.stop_reason = "workspace_guard";
      if (!joints_within_guard(current.q) && guard.stop_reason == nullptr)
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
          bolt_thread_drive::lateral_component(position_error, axis);
      const Eigen::Vector3d lateral_velocity =
          bolt_thread_drive::lateral_component(twist.head<3>(), axis);
      const double desired_angle = bolt_thread_drive::drive_angle(elapsed_s);
      const double desired_speed =
          bolt_thread_drive::drive_angular_speed(elapsed_s);
      const Eigen::Matrix3d desired_rotation =
          start_rotation *
          Eigen::AngleAxisd(desired_angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
      const Eigen::Vector3d rotation_error = bolt_thread_drive::orientation_error(
          desired_rotation, current_pose.topLeftCorner<3, 3>());
      const Eigen::Vector3d desired_angular_velocity = axis * desired_speed;

      Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
      const double axial_force = bolt_thread_drive::force_command(elapsed_s);
      wrench.head<3>() = axis * axial_force
          - bolt_thread_drive::kLateralStiffnessNm * lateral_error
          - bolt_thread_drive::kLateralDampingNsM * lateral_velocity;
      wrench.tail<3>() = -bolt_thread_drive::kRotationalStiffnessNmRad * rotation_error
          + bolt_thread_drive::kRotationalDampingNmsRad *
                (desired_angular_velocity - twist.tail<3>());
      Eigen::Matrix<double, 7, 1> tau_desired =
          current_jacobian.transpose() * wrench + coriolis;
      std::array<double, 7> tau_desired_array{};
      Eigen::Map<Eigen::Matrix<double, 7, 1>>(tau_desired_array.data()) = tau_desired;
      const std::array<double, 7> tau_limited =
          franka::limitRate(franka::kMaxTorqueRate, tau_desired_array,
                            current.tau_J_d);

      if (telemetry_count < telemetry.size()) {
        auto& sample = telemetry[telemetry_count++];
        sample.time_s = elapsed_s;
        sample.desired_angle_rad = desired_angle;
        sample.desired_angular_speed_rad_s = desired_speed;
        sample.commanded_axial_force_N = axial_force;
        sample.measured_axial_delta_N = current.K_F_ext_hat_K[2] - mean_fz;
        sample.expected_K_Tz_Nm = expected_tz;
        sample.residual_K_Tz_Nm = residual_tz;
        const Eigen::Vector3d displacement = current_xyz - start_xyz;
        sample.axial_travel_m = axis.dot(displacement);
        sample.lateral_travel_m =
            bolt_thread_drive::lateral_component(displacement, axis).norm();
        sample.q = current.q;
        sample.dq = current.dq;
        sample.tau_command = tau_limited;
        sample.wrench_K = current.K_F_ext_hat_K;
        sample.xyz = {{current_xyz.x(), current_xyz.y(), current_xyz.z()}};
      } else if (guard.stop_reason == nullptr) {
        guard.stop_reason = "telemetry_capacity";
      }

      franka::Torques command(tau_limited);
      if (guard.stop_reason != nullptr) return franka::MotionFinished(command);
      return command;
    };

    robot.control(callback);
    const franka::RobotState final_state = robot.readOnce();
    write_csv(argv[3], telemetry, telemetry_count);
    std::cout << "[result] stop reason:      "
              << (guard.stop_reason != nullptr ? guard.stop_reason : "unknown") << "\n"
              << "[result] minimum residual: " << guard.minimum_residual_tz_Nm << " Nm\n"
              << "[result] axial delta range:" << guard.minimum_axial_delta_N << " .. "
              << guard.maximum_axial_delta_N << " N\n"
              << "[result] max axial travel: " << guard.maximum_axial_travel_m * 1000.0 << " mm\n"
              << "[result] max lateral:      " << guard.maximum_lateral_travel_m * 1000.0 << " mm\n"
              << "[result] final q7:         " << final_state.q[6] << " rad\n"
              << "[saved] " << argv[3] << "\n"
              << "The Hand remains closed. No ratchet, release, seated classification, camera, or collision-setting command was issued.\n";
  } catch (const franka::Exception& error) {
    std::cerr << "libfranka error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
