#include "hybrid_thread_control.h"

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

constexpr std::size_t kMaximumSamples = 100000;
constexpr double kBaselineDurationS = 3.0;
constexpr double kMaximumBaselineFzStdN = 0.1;
constexpr double kMaximumBaselineDqRadS = 0.01;
constexpr double kStartJointToleranceRad = 0.005;
constexpr double kJointMarginRad = 0.12;
constexpr std::array<double, 7> kJointMin{{
    -2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159}};
constexpr std::array<double, 7> kJointMax{{
    2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159}};
constexpr std::array<double, 7> kExpectedStartQ{{
    -0.18192104995250702, 0.9287270307540894, 0.8591495156288147,
    -1.5973801612854004, -0.7637084722518921, 2.0962517261505127,
    0.06202800199389458}};
constexpr std::array<double, 3> kWorkspaceMin{{0.58, 0.315, 0.095}};
constexpr std::array<double, 3> kWorkspaceMax{{0.605, 0.335, 0.115}};

std::atomic<bool> g_abort{false};
void signal_handler(int) { g_abort.store(true, std::memory_order_relaxed); }

struct PlanPoint {
  double time_s = 0.0;
  Eigen::Vector3d local_lateral = Eigen::Vector3d::Zero();
  Eigen::Quaterniond relative_rotation = Eigen::Quaterniond::Identity();
  double source_q7_rad = 0.0;
};

struct ModelPoint { double q7 = 0.0; double tz = 0.0; };

struct Telemetry {
  double time_s = 0.0;
  double plan_time_s = 0.0;
  double plan_progress_scale = 0.0;
  double source_q7_rad = 0.0;
  double commanded_axial_force_N = 0.0;
  double measured_axial_delta_N = 0.0;
  double residual_tz_Nm = 0.0;
  double axial_travel_m = 0.0;
  double lateral_error_m = 0.0;
  double orientation_error_rad = 0.0;
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 6> wrench{};
  std::array<double, 3> xyz{};
};

Eigen::Matrix4d pose_matrix(const std::array<double, 16>& pose) {
  return Eigen::Map<const Eigen::Matrix<double, 4, 4>>(pose.data());
}

std::vector<double> csv_values(const std::string& line) {
  std::vector<double> values;
  std::istringstream input(line);
  std::string token;
  while (std::getline(input, token, ',')) values.push_back(std::stod(token));
  return values;
}

std::vector<PlanPoint> load_plan(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open hybrid plan");
  std::string line; std::getline(input, line);
  std::vector<PlanPoint> plan;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto v = csv_values(line);
    if (v.size() != 9) throw std::runtime_error("hybrid plan row must have 9 columns");
    PlanPoint point;
    point.time_s = v[0];
    point.local_lateral = {v[1], v[2], v[3]};
    point.relative_rotation = Eigen::Quaterniond(v[7], v[4], v[5], v[6]).normalized();
    point.source_q7_rad = v[8];
    plan.push_back(point);
  }
  if (plan.size() < 2) throw std::runtime_error("hybrid plan is too short");
  for (std::size_t i = 1; i < plan.size(); ++i)
    if (!(plan[i].time_s > plan[i - 1].time_s)) throw std::runtime_error("hybrid plan time is invalid");
  return plan;
}

std::vector<ModelPoint> load_model(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open dynamic model");
  std::string line; std::getline(input, line);
  std::vector<ModelPoint> model;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto v = csv_values(line);
    if (v.size() != 2) throw std::runtime_error("dynamic model row must have 2 columns");
    model.push_back({v[0], v[1]});
  }
  if (model.size() < 2) throw std::runtime_error("dynamic model is too short");
  for (std::size_t i = 1; i < model.size(); ++i)
    if (!(model[i].q7 > model[i - 1].q7)) throw std::runtime_error("dynamic model q7 is invalid");
  return model;
}

PlanPoint interpolate_plan(const std::vector<PlanPoint>& plan, double time_s) {
  if (time_s <= plan.front().time_s) return plan.front();
  if (time_s >= plan.back().time_s) return plan.back();
  const auto upper = std::upper_bound(
      plan.begin(), plan.end(), time_s,
      [](double value, const PlanPoint& point) { return value < point.time_s; });
  const std::size_t hi = static_cast<std::size_t>(upper - plan.begin());
  const std::size_t lo = hi - 1;
  const double f = (time_s - plan[lo].time_s) / (plan[hi].time_s - plan[lo].time_s);
  PlanPoint result;
  result.time_s = time_s;
  result.local_lateral = plan[lo].local_lateral + f * (plan[hi].local_lateral - plan[lo].local_lateral);
  result.relative_rotation = plan[lo].relative_rotation.slerp(f, plan[hi].relative_rotation).normalized();
  result.source_q7_rad = plan[lo].source_q7_rad + f * (plan[hi].source_q7_rad - plan[lo].source_q7_rad);
  return result;
}

double expected_tz(const std::vector<ModelPoint>& model, double q7) {
  if (q7 <= model.front().q7) return model.front().tz;
  if (q7 >= model.back().q7) return model.back().tz;
  const auto upper = std::upper_bound(
      model.begin(), model.end(), q7,
      [](double value, const ModelPoint& point) { return value < point.q7; });
  const std::size_t hi = static_cast<std::size_t>(upper - model.begin());
  const std::size_t lo = hi - 1;
  const double f = (q7 - model[lo].q7) / (model[hi].q7 - model[lo].q7);
  return model[lo].tz + f * (model[hi].tz - model[lo].tz);
}

bool in_workspace(const Eigen::Vector3d& xyz) {
  for (Eigen::Index i = 0; i < 3; ++i)
    if (xyz(i) < kWorkspaceMin[static_cast<std::size_t>(i)] ||
        xyz(i) > kWorkspaceMax[static_cast<std::size_t>(i)]) return false;
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
               const std::vector<Telemetry>& data, std::size_t count) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open telemetry output");
  out << "time_s,plan_time_s,plan_progress_scale,source_q7_rad,commanded_axial_force_N,measured_axial_delta_N,residual_tz_Nm,axial_travel_m,lateral_error_m,orientation_error_rad";
  for (int i = 0; i < 7; ++i) out << ",q" << i;
  for (int i = 0; i < 7; ++i) out << ",dq" << i;
  for (int i = 0; i < 6; ++i) out << ",w" << i;
  out << ",x,y,z\n" << std::setprecision(12);
  for (std::size_t row = 0; row < count; ++row) {
    const auto& s = data[row];
    out << s.time_s << ',' << s.plan_time_s << ',' << s.plan_progress_scale << ','
        << s.source_q7_rad << ',' << s.commanded_axial_force_N << ','
        << s.measured_axial_delta_N << ',' << s.residual_tz_Nm << ',' << s.axial_travel_m << ','
        << s.lateral_error_m << ',' << s.orientation_error_rad;
    for (double v : s.q) out << ',' << v;
    for (double v : s.dq) out << ',' << v;
    for (double v : s.wrench) out << ',' << v;
    for (double v : s.xyz) out << ',' << v;
    out << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0] << " read-only-preview|execute ROBOT_IP TELEMETRY_CSV PLAN_CSV MODEL_CSV CONFIRMATION\n";
    return 2;
  }
  const bool execute = std::string(argv[1]) == "execute";
  if (!execute && std::string(argv[1]) != "read-only-preview") return 2;
  try {
    const auto plan = load_plan(argv[4]);
    const auto model_points = load_model(argv[5]);
    std::cout << "=== DIRECT LIBFRANKA HYBRID THREAD REPLAY ===\n"
              << "mode:                     " << (execute ? "EXECUTE" : "READ_ONLY_PREVIEW") << "\n"
              << "robot IP:                 " << argv[2] << "\n"
              << "plan points/duration:     " << plan.size() << " / " << plan.back().time_s << " s\n"
              << "tool-axis position replay:REMOVED; AXIALLY COMPLIANT\n"
              << "lateral/roll/pitch:       HELD AT LIVE START\n"
              << "tightening yaw:           successful Pilot profile\n"
              << "axial regulation:         measured target +1.200360 N; P-only Kp=0.25\n"
              << "command ceiling/slew:     +1.5 N / 0.5 N/s\n"
              << "force acquisition:        3.0 s before source motion\n"
              << "target pacing:            slow/pause before tracking guard\n"
              << "hard axial/residual stop: +/-2.0 N / -0.538305 Nm\n"
              << "dynamic q7 boundary:      " << model_points.back().q7 << " rad\n"
              << "Franka protections:       UNCHANGED\n"
              << "Press Enter to connect " << (execute ? "and perform preflight" : "read-only") << ": ";
    std::string input; std::getline(std::cin, input);
    franka::Robot robot(argv[2]); robot.readOnce();
    franka::RobotState state = robot.readOnce();
    franka::Model model = robot.loadModel();
    double count = 0.0, mean_fz = 0.0, m2 = 0.0, max_dq = 0.0;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < kBaselineDurationS) {
      state = robot.readOnce(); const double fz = state.K_F_ext_hat_K[2]; count += 1.0;
      const double delta = fz - mean_fz; mean_fz += delta / count; m2 += delta * (fz - mean_fz);
      for (double dq : state.dq) max_dq = std::max(max_dq, std::abs(dq));
    }
    const double std_fz = count > 1 ? std::sqrt(m2 / count) : 0.0;
    const Eigen::Matrix4d pose = pose_matrix(state.O_T_EE);
    const Eigen::Matrix4d ee_t_k = pose_matrix(state.EE_T_K);
    const Eigen::Vector3d xyz = pose.topRightCorner<3, 1>();
    const Eigen::Matrix3d start_rotation = pose.topLeftCorner<3, 3>();
    const Eigen::Matrix4d O_T_K = pose * ee_t_k;
    const Eigen::Vector3d axis =
        O_T_K.topLeftCorner<3, 3>().col(2).normalized();
    double start_q1_q6_error = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
      start_q1_q6_error =
          std::max(start_q1_q6_error, std::abs(state.q[i] - kExpectedStartQ[i]));
    }
    double maximum_local_lateral = 0.0;
    bool planned_path_in_workspace = true;
    for (const auto& point : plan) {
      maximum_local_lateral =
          std::max(maximum_local_lateral, point.local_lateral.norm());
      planned_path_in_workspace = planned_path_in_workspace &&
          in_workspace(xyz + start_rotation * point.local_lateral);
    }
    const auto jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    std::cout << std::fixed << std::setprecision(6)
              << "\nlive xyz:                 " << xyz.transpose() << "\n"
              << "live q7/q1-q6 error:      " << state.q[6] << " / " << start_q1_q6_error << " rad\n"
              << "baseline F_Kz/std:        " << mean_fz << " / " << std_fz << " N\n"
              << "maximum baseline |dq|:    " << max_dq << " rad/s\n"
              << "maximum planned lateral:  " << maximum_local_lateral * 1000.0 << " mm\n"
              << "Jacobian row norms:       " << jacobian.rowwise().norm().transpose() << "\n";
    std::string blockers;
    if (start_q1_q6_error > kStartJointToleranceRad) blockers += "  - live q1-q6 are not near the recorded Drive 1 start\n";
    if (state.q[6] < model_points.front().q7 - hybrid_thread::kDynamicModelLowAllowanceRad ||
        state.q[6] > model_points.back().q7) blockers += "  - live q7 is outside the dynamic torque model\n";
    if (!in_workspace(xyz)) blockers += "  - live TCP is outside the local workspace\n";
    if (!planned_path_in_workspace) blockers += "  - planned compliant-axis path leaves the local workspace\n";
    if (!joints_within_guard(state.q)) blockers += "  - live joints violate guarded mechanical ranges\n";
    if ((ee_t_k - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff() > 1e-8) blockers += "  - EE_T_K is not identity\n";
    if (std_fz > kMaximumBaselineFzStdN) blockers += "  - force baseline is not stationary\n";
    if (max_dq > kMaximumBaselineDqRadS) blockers += "  - arm moved during baseline\n";
    if (!jacobian.allFinite() || jacobian.cwiseAbs().maxCoeff() < 0.1) blockers += "  - direct Jacobian is invalid\n";
    if (!blockers.empty()) std::cout << "\nExecution blockers:\n" << blockers;
    if (!execute) {
      std::cout << "\nPreview complete. No control session, arm, Hand, or collision-setting command was issued.\n";
      return blockers.empty() ? 0 : 3;
    }
    if (!blockers.empty()) return 3;
    std::cout << "Press Enter for the one authorized hybrid replay; Ctrl+C aborts: "; std::getline(std::cin, input);
    if (std::string(argv[6]) != "HYBRID_EXECUTION_AUTHORIZED") return 3;
    std::signal(SIGINT, signal_handler); g_abort.store(false);
    hybrid_thread::Guard guard;
    guard.baseline_fz_N = mean_fz; guard.start_xyz = xyz; guard.axis = axis;
    guard.dynamic_model_min_q7_rad = model_points.front().q7;
    guard.dynamic_model_max_q7_rad = model_points.back().q7;
    std::vector<Telemetry> telemetry(kMaximumSamples); std::size_t telemetry_count = 0;
    double elapsed_s = 0.0;
    double plan_time_s = 0.0;
    double plan_progress_scale = 0.0;
    double axial_force_command_N = 0.0;
    auto callback = [&](const franka::RobotState& current, franka::Duration period) -> franka::Torques {
      elapsed_s += period.toSec();
      const auto target = interpolate_plan(plan, plan_time_s);
      const Eigen::Matrix4d current_pose = pose_matrix(current.O_T_EE);
      const Eigen::Vector3d current_xyz = current_pose.topRightCorner<3, 1>();
      const Eigen::Vector3d desired_xyz = xyz + start_rotation * target.local_lateral;
      const Eigen::Matrix3d desired_rotation = start_rotation * target.relative_rotation.toRotationMatrix();
      const Eigen::Vector3d position_error = current_xyz - desired_xyz;
      const Eigen::Vector3d lateral_error = hybrid_thread::lateral_component(position_error, axis);
      const Eigen::Vector3d rotation_error = hybrid_thread::orientation_error(desired_rotation, current_pose.topLeftCorner<3, 3>());
      const double rotation_error_angle =
          hybrid_thread::orientation_error_angle_rad(rotation_error);
      plan_progress_scale = elapsed_s < hybrid_thread::kForceAcquisitionDurationS
          ? 0.0
          : hybrid_thread::tracking_progress_scale(
                lateral_error.norm(), rotation_error_angle);
      plan_time_s = std::min(
          plan.back().time_s,
          plan_time_s + period.toSec() * plan_progress_scale);
      const double residual_tz = current.K_F_ext_hat_K[5] - expected_tz(model_points, current.q[6]);
      guard.update(current.q, current.K_F_ext_hat_K[2], residual_tz, current_xyz,
                   lateral_error.norm(), rotation_error_angle);
      if (g_abort.load() && guard.stop_reason == nullptr) guard.stop_reason = "operator_abort";
      if (!in_workspace(current_xyz) && guard.stop_reason == nullptr) guard.stop_reason = "workspace_guard";
      if (!joints_within_guard(current.q) && guard.stop_reason == nullptr) guard.stop_reason = "joint_guard";
      if (elapsed_s >= hybrid_thread::kMaximumTotalControlDurationS &&
          guard.stop_reason == nullptr) guard.stop_reason = "state_timeout";
      if (plan_time_s >= plan.back().time_s && guard.stop_reason == nullptr) guard.stop_reason = "hybrid_plan_complete";
      const auto j_array = model.zeroJacobian(franka::Frame::kEndEffector, current);
      const auto c_array = model.coriolis(current);
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> J(j_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(c_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(current.dq.data());
      const Eigen::Matrix<double, 6, 1> twist = J * dq;
      const Eigen::Vector3d lateral_velocity = hybrid_thread::lateral_component(twist.head<3>(), axis);
      Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
      const double measured_axial_delta_N = current.K_F_ext_hat_K[2] - mean_fz;
      const double axial_force_target_N =
          hybrid_thread::proportional_force_command_target(measured_axial_delta_N);
      axial_force_command_N = hybrid_thread::slew_limited_force_command(
          axial_force_command_N, axial_force_target_N, period.toSec());
      wrench.head<3>() = axis * axial_force_command_N - hybrid_thread::kLateralStiffnessNm * lateral_error - hybrid_thread::kLateralDampingNsM * lateral_velocity;
      wrench.tail<3>() = -hybrid_thread::kRotationalStiffnessNmRad * rotation_error - hybrid_thread::kRotationalDampingNmsRad * twist.tail<3>();
      Eigen::Matrix<double, 7, 1> tau = J.transpose() * wrench + coriolis;
      std::array<double, 7> desired_tau{}; Eigen::Map<Eigen::Matrix<double, 7, 1>>(desired_tau.data()) = tau;
      const auto limited = franka::limitRate(franka::kMaxTorqueRate, desired_tau, current.tau_J_d);
      if (telemetry_count < telemetry.size()) {
        auto& s = telemetry[telemetry_count++]; s.time_s = elapsed_s;
        s.plan_time_s = plan_time_s; s.plan_progress_scale = plan_progress_scale;
        s.source_q7_rad = target.source_q7_rad;
        s.commanded_axial_force_N = axial_force_command_N; s.measured_axial_delta_N = measured_axial_delta_N;
        s.residual_tz_Nm = residual_tz; s.axial_travel_m = axis.dot(current_xyz - xyz);
        s.lateral_error_m = lateral_error.norm(); s.orientation_error_rad = rotation_error_angle;
        s.q = current.q; s.dq = current.dq; s.wrench = current.K_F_ext_hat_K;
        s.xyz = {{current_xyz.x(), current_xyz.y(), current_xyz.z()}};
      } else if (guard.stop_reason == nullptr) guard.stop_reason = "telemetry_capacity";
      franka::Torques command(limited);
      if (guard.stop_reason != nullptr) return franka::MotionFinished(command);
      return command;
    };
    robot.control(callback);
    write_csv(argv[3], telemetry, telemetry_count);
    std::cout << "[result] stop reason:      " << (guard.stop_reason ? guard.stop_reason : "unknown") << "\n"
              << "[result] axial range:      " << guard.minimum_axial_delta_N << " .. " << guard.maximum_axial_delta_N << " N\n"
              << "[result] minimum residual: " << guard.minimum_residual_tz_Nm << " Nm\n"
              << "[result] max axial travel: " << guard.maximum_axial_travel_m * 1000.0 << " mm\n"
              << "[result] max lateral error:" << guard.maximum_lateral_error_m * 1000.0 << " mm\n"
              << "[saved] " << argv[3] << "\n";
  } catch (const franka::Exception& e) { std::cerr << "libfranka error: " << e.what() << '\n'; return 1; }
    catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
  return 0;
}
