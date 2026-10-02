// Guarded 1 kHz executor for a clamped-cubic joint trajectory.
//
// The plan is fitted offline through the exact commissioned 15 Hz position
// knots.  This controller evaluates that spline continuously and sends
// rate-limited JointVelocities with a small position-error correction.  It
// avoids the per-waypoint Ruckig boundaries that stopped the legacy 2x replay.

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
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
constexpr double kStartToleranceRad = 0.005;
constexpr double kTerminalToleranceRad = 0.002;
constexpr double kTrackingCutoffRad = 0.050;
constexpr double kPositionCorrectionGain = 4.0;
constexpr double kStationaryDqRadS = 0.010;
constexpr int kRequiredStationarySamples = 100;
constexpr double kMinimumSettleS = 0.5;
constexpr double kMaximumSettleS = 5.0;
constexpr double kBaselineDurationS = 1.0;
constexpr double kMaximumBaselineDqRadS = 0.010;
constexpr double kJointMarginRad = 0.12;
constexpr std::array<double, kJoints> kJointMin{{
    -2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159}};
constexpr std::array<double, kJoints> kJointMax{{
    2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159}};
#ifndef FR3_RETIMED_VELOCITY_CAP_RAD_S
#define FR3_RETIMED_VELOCITY_CAP_RAD_S 1.20
#endif
#ifndef FR3_RETIMED_ACCELERATION_CAP_RAD_S2
#define FR3_RETIMED_ACCELERATION_CAP_RAD_S2 4.0
#endif
#ifndef FR3_RETIMED_JERK_CAP_RAD_S3
#define FR3_RETIMED_JERK_CAP_RAD_S3 40.0
#endif
// Defaults above exactly match the original always-shared caps; only a
// distinct target (see CMakeLists.txt) overrides them via -D, so the
// original fr3_retimed_joint_trajectory binary used by every bolt stage
// (and bracket stages before 2026-08-16) is byte-for-byte unaffected.
constexpr std::array<double, kJoints> kVelocityCaps{{
    FR3_RETIMED_VELOCITY_CAP_RAD_S, FR3_RETIMED_VELOCITY_CAP_RAD_S, FR3_RETIMED_VELOCITY_CAP_RAD_S,
    FR3_RETIMED_VELOCITY_CAP_RAD_S, FR3_RETIMED_VELOCITY_CAP_RAD_S, FR3_RETIMED_VELOCITY_CAP_RAD_S,
    FR3_RETIMED_VELOCITY_CAP_RAD_S}};
constexpr std::array<double, kJoints> kAccelerationCaps{{
    FR3_RETIMED_ACCELERATION_CAP_RAD_S2, FR3_RETIMED_ACCELERATION_CAP_RAD_S2, FR3_RETIMED_ACCELERATION_CAP_RAD_S2,
    FR3_RETIMED_ACCELERATION_CAP_RAD_S2, FR3_RETIMED_ACCELERATION_CAP_RAD_S2, FR3_RETIMED_ACCELERATION_CAP_RAD_S2,
    FR3_RETIMED_ACCELERATION_CAP_RAD_S2}};
constexpr std::array<double, kJoints> kJerkCaps{{
    FR3_RETIMED_JERK_CAP_RAD_S3, FR3_RETIMED_JERK_CAP_RAD_S3, FR3_RETIMED_JERK_CAP_RAD_S3,
    FR3_RETIMED_JERK_CAP_RAD_S3, FR3_RETIMED_JERK_CAP_RAD_S3, FR3_RETIMED_JERK_CAP_RAD_S3,
    FR3_RETIMED_JERK_CAP_RAD_S3}};
std::array<double, 3> g_workspace_min{{0.21, -0.50, 0.005}};
std::array<double, 3> g_workspace_max{{0.70, 0.35, 0.27}};

std::atomic<bool> g_abort{false};
void signal_handler(int) { g_abort.store(true, std::memory_order_relaxed); }

struct Segment {
  double t0 = 0.0;
  double t1 = 0.0;
  std::array<std::array<double, 4>, kJoints> coefficient{};
};

struct Reference {
  std::array<double, kJoints> q{};
  std::array<double, kJoints> dq{};
  std::array<double, kJoints> ddq{};
  std::array<double, kJoints> jerk{};
};

struct Sample {
  std::int64_t wall_time_ns = 0;
  double time_s = 0.0;
  double maximum_tracking_error_rad = 0.0;
  double maximum_abs_dq_rad_s = 0.0;
  int stopping = 0;
  std::array<double, kJoints> q{};
  std::array<double, kJoints> dq{};
  std::array<double, kJoints> q_reference{};
  std::array<double, 3> xyz{};
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream input(line);
  std::string field;
  while (std::getline(input, field, ',')) fields.push_back(field);
  return fields;
}

std::vector<Segment> load_plan(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open retimed trajectory plan");
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("trajectory plan is empty");
  if (split(line).size() != 30) throw std::runtime_error("trajectory plan header is invalid");
  std::vector<Segment> segments;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() != 30) throw std::runtime_error("trajectory plan row is invalid");
    Segment segment;
    segment.t0 = std::stod(fields[0]);
    segment.t1 = std::stod(fields[1]);
    std::size_t field = 2;
    for (auto& joint : segment.coefficient) {
      for (double& value : joint) value = std::stod(fields[field++]);
    }
    if (!std::isfinite(segment.t0) || !std::isfinite(segment.t1) ||
        segment.t1 <= segment.t0)
      throw std::runtime_error("trajectory plan has invalid segment time");
    for (const auto& joint : segment.coefficient)
      for (double value : joint)
        if (!std::isfinite(value))
          throw std::runtime_error("trajectory plan contains a non-finite coefficient");
    if (segments.empty()) {
      if (std::abs(segment.t0) > 1e-9)
        throw std::runtime_error("trajectory plan must begin at zero");
    } else if (std::abs(segment.t0 - segments.back().t1) > 1e-8) {
      throw std::runtime_error("trajectory plan segment times are not contiguous");
    }
    segments.push_back(segment);
  }
  if (segments.empty()) throw std::runtime_error("trajectory plan has no segments");
  return segments;
}

Reference evaluate(const Segment& segment, double time_s) {
  const double u = std::clamp(time_s, segment.t0, segment.t1) - segment.t0;
  Reference result;
  for (std::size_t joint = 0; joint < kJoints; ++joint) {
    const auto& c = segment.coefficient[joint];
    result.q[joint] = c[0] + u * (c[1] + u * (c[2] + u * c[3]));
    result.dq[joint] = c[1] + u * (2.0 * c[2] + 3.0 * u * c[3]);
    result.ddq[joint] = 2.0 * c[2] + 6.0 * u * c[3];
    result.jerk[joint] = 6.0 * c[3];
  }
  return result;
}

Reference evaluate(const std::vector<Segment>& plan, double time_s,
                   std::size_t* hint = nullptr) {
  std::size_t index = hint == nullptr ? 0 : *hint;
  index = std::min(index, plan.size() - 1);
  while (index + 1 < plan.size() && time_s > plan[index].t1) ++index;
  while (index > 0 && time_s < plan[index].t0) --index;
  if (hint != nullptr) *hint = index;
  return evaluate(plan[index], time_s);
}

bool joints_safe(const std::array<double, kJoints>& q) {
  for (std::size_t joint = 0; joint < kJoints; ++joint)
    if (q[joint] <= kJointMin[joint] + kJointMarginRad ||
        q[joint] >= kJointMax[joint] - kJointMarginRad)
      return false;
  return true;
}

bool workspace_safe(const std::array<double, 16>& pose) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double value = pose[12 + axis];
    if (value < g_workspace_min[axis] || value > g_workspace_max[axis]) return false;
  }
  return true;
}

double maximum_abs(const std::array<double, kJoints>& values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::abs(value));
  return result;
}

double maximum_error(const std::array<double, kJoints>& a,
                     const std::array<double, kJoints>& b) {
  double result = 0.0;
  for (std::size_t joint = 0; joint < kJoints; ++joint)
    result = std::max(result, std::abs(a[joint] - b[joint]));
  return result;
}

void write_csv(const std::filesystem::path& path, const std::vector<Sample>& samples) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open trajectory telemetry output");
  output << "wall_time_ns,time_s,maximum_tracking_error_rad,maximum_abs_dq_rad_s,stopping";
  for (int joint = 0; joint < 7; ++joint) output << ",q" << joint;
  for (int joint = 0; joint < 7; ++joint) output << ",dq" << joint;
  for (int joint = 0; joint < 7; ++joint) output << ",q_reference" << joint;
  output << ",x,y,z\n" << std::setprecision(12);
  for (const auto& sample : samples) {
    output << sample.wall_time_ns << ',' << sample.time_s << ','
           << sample.maximum_tracking_error_rad << ','
           << sample.maximum_abs_dq_rad_s << ',' << sample.stopping;
    for (double value : sample.q) output << ',' << value;
    for (double value : sample.dq) output << ',' << value;
    for (double value : sample.q_reference) output << ',' << value;
    for (double value : sample.xyz) output << ',' << value;
    output << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 13) {
    std::cerr << "Usage: " << argv[0]
              << " read-only-preview|execute ROBOT_IP PLAN_CSV TELEMETRY_CSV"
                 " CONFIRMATION STAGE_NAME MIN_X MIN_Y MIN_Z MAX_X MAX_Y MAX_Z\n";
    return 2;
  }
  const std::string mode(argv[1]);
  const bool execute = mode == "execute";
  if (!execute && mode != "read-only-preview") return 2;
  try {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      g_workspace_min[axis] = std::stod(argv[7 + axis]);
      g_workspace_max[axis] = std::stod(argv[10 + axis]);
      if (!std::isfinite(g_workspace_min[axis]) ||
          !std::isfinite(g_workspace_max[axis]) ||
          g_workspace_min[axis] >= g_workspace_max[axis])
        throw std::runtime_error("invalid stage workspace bounds");
    }
    const auto plan = load_plan(argv[3]);
    const double duration_s = plan.back().t1;
    const auto start_reference = evaluate(plan, 0.0);
    const auto final_reference = evaluate(plan, duration_s);
    double peak_velocity = 0.0;
    double peak_acceleration = 0.0;
    double peak_jerk = 0.0;
    bool planned_joints_safe = true;
    for (double time_s = 0.0; time_s <= duration_s + 1e-12; time_s += 0.001) {
      const auto reference = evaluate(plan, std::min(time_s, duration_s));
      peak_velocity = std::max(peak_velocity, maximum_abs(reference.dq));
      peak_acceleration = std::max(peak_acceleration, maximum_abs(reference.ddq));
      peak_jerk = std::max(peak_jerk, maximum_abs(reference.jerk));
      planned_joints_safe = planned_joints_safe && joints_safe(reference.q);
    }
    std::string blockers;
    if (peak_velocity > maximum_abs(kVelocityCaps))
      blockers += "  - planned velocity exceeds the controller cap\n";
    if (peak_acceleration > maximum_abs(kAccelerationCaps))
      blockers += "  - planned acceleration exceeds the controller cap\n";
    if (peak_jerk > maximum_abs(kJerkCaps))
      blockers += "  - planned jerk exceeds the controller cap\n";
    if (!planned_joints_safe) blockers += "  - planned joints leave guarded ranges\n";

    std::cout << "=== GUARDED RETIMED JOINT TRAJECTORY ===\n"
              << "mode:                     " << (execute ? "EXECUTE" : "READ_ONLY_PREVIEW") << "\n"
              << "robot IP:                 " << argv[2] << "\n"
              << "stage:                    " << argv[6] << "\n"
              << "motion generator:         1 kHz clamped-cubic reference + rate-limited JointVelocities\n"
              << "planned duration:         " << duration_s << " s\n"
              << "planned peak |dq|:        " << peak_velocity << " rad/s\n"
              << "planned peak |ddq|:       " << peak_acceleration << " rad/s^2\n"
              << "planned peak |jerk|:      " << peak_jerk << " rad/s^3\n"
              << "velocity/accel/jerk caps: " << maximum_abs(kVelocityCaps) << " / "
              << maximum_abs(kAccelerationCaps) << " / " << maximum_abs(kJerkCaps) << "\n"
              << "tracking/terminal guards: 0.050 / 0.002 rad\n"
              << "stage workspace min:      " << g_workspace_min[0] << ' '
              << g_workspace_min[1] << ' ' << g_workspace_min[2] << " m\n"
              << "stage workspace max:      " << g_workspace_max[0] << ' '
              << g_workspace_max[1] << ' ' << g_workspace_max[2] << " m\n"
              << "collision/reflex settings:UNCHANGED\n"
              << "Hand commands:            DISABLED\n"
              << "Press Enter to connect " << (execute ? "and perform preflight" : "read-only")
              << " (Ctrl+C cancels): ";
    std::string input;
    std::getline(std::cin, input);
    franka::Robot robot(argv[2]);
    const franka::Model model = robot.loadModel();
    franka::RobotState state = robot.readOnce();
    double maximum_baseline_dq = 0.0;
    const auto baseline_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(
               std::chrono::steady_clock::now() - baseline_start).count() <
           kBaselineDurationS) {
      state = robot.readOnce();
      maximum_baseline_dq = std::max(maximum_baseline_dq, maximum_abs(state.dq));
    }
    const double start_error = maximum_error(state.q, start_reference.q);
    if (start_error > kStartToleranceRad)
      blockers += "  - live arm is not at the retimed stage start\n";
    if (maximum_baseline_dq > kMaximumBaselineDqRadS)
      blockers += "  - arm moved during the stationary baseline\n";
    if (!joints_safe(state.q)) blockers += "  - live joints leave guarded ranges\n";
    if (!workspace_safe(state.O_T_EE)) blockers += "  - live TCP is outside the guarded workspace\n";

    bool predicted_workspace_safe = true;
    std::array<double, 3> xyz_min{{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()}};
    std::array<double, 3> xyz_max{{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}};
    for (double time_s = 0.0; time_s <= duration_s + 1e-12; time_s += 0.010) {
      const auto reference = evaluate(plan, std::min(time_s, duration_s));
      const auto pose = model.pose(
          franka::Frame::kEndEffector, reference.q, state.F_T_EE, state.EE_T_K);
      predicted_workspace_safe = predicted_workspace_safe && workspace_safe(pose);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        xyz_min[axis] = std::min(xyz_min[axis], pose[12 + axis]);
        xyz_max[axis] = std::max(xyz_max[axis], pose[12 + axis]);
      }
    }
    if (!predicted_workspace_safe)
      blockers += "  - predicted retimed path leaves the guarded workspace\n";
    std::cout << std::fixed << std::setprecision(6)
              << "\nlive start error:         " << start_error << " rad\n"
              << "maximum baseline |dq|:    " << maximum_baseline_dq << " rad/s\n"
              << "predicted path xyz min:   " << xyz_min[0] << ' ' << xyz_min[1] << ' ' << xyz_min[2] << "\n"
              << "predicted path xyz max:   " << xyz_max[0] << ' ' << xyz_max[1] << ' ' << xyz_max[2] << "\n";
    if (!blockers.empty()) std::cout << "\nExecution blockers:\n" << blockers;
    if (!execute) {
      std::cout << "\nPreview complete. No control session, arm, Hand, or collision-setting command was issued.\n";
      return blockers.empty() ? 0 : 3;
    }
    if (!blockers.empty()) return 3;
    if (std::string(argv[5]) != "RUN_RETIMED_2X_STAGE" &&
        std::string(argv[5]) != "RUN_RETIMED_ACCELERATED_STAGE") return 3;
    std::cout << "Keep the empty/object path clear and one hand near emergency stop. "
                 "Press Enter for this commissioned stage: ";
    std::getline(std::cin, input);

    std::signal(SIGINT, signal_handler);
    g_abort.store(false, std::memory_order_relaxed);
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>((duration_s + 7.0) * 1000.0));
    std::array<double, kJoints> commanded_velocity = state.dq_d;
    std::array<double, kJoints> commanded_acceleration = state.ddq_d;
    std::array<double, kJoints> lower_velocity{};
    for (std::size_t joint = 0; joint < kJoints; ++joint)
      lower_velocity[joint] = -kVelocityCaps[joint];
    const auto wall_start = std::chrono::system_clock::now();
    const auto wall_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        wall_start.time_since_epoch()).count();
    double elapsed_s = 0.0;
    double stop_request_s = 0.0;
    int stationary_samples = 0;
    std::size_t segment_hint = 0;
    const char* stop_reason = nullptr;
    double maximum_live_tracking_error = 0.0;

    auto callback = [&](const franka::RobotState& current,
                        franka::Duration period) -> franka::JointVelocities {
      elapsed_s += period.toSec();
      const double reference_time = std::min(elapsed_s, duration_s);
      const Reference reference = evaluate(plan, reference_time, &segment_hint);
      const double tracking_error = maximum_error(current.q, reference.q);
      maximum_live_tracking_error = std::max(maximum_live_tracking_error, tracking_error);
      if (g_abort.load(std::memory_order_relaxed) && stop_reason == nullptr)
        stop_reason = "operator_abort";
      if (tracking_error > kTrackingCutoffRad && stop_reason == nullptr)
        stop_reason = "maximum_joint_tracking_error";
      if (!joints_safe(current.q) && stop_reason == nullptr)
        stop_reason = "joint_guard";
      if (!workspace_safe(current.O_T_EE) && stop_reason == nullptr)
        stop_reason = "workspace_guard";
      const double final_error = maximum_error(current.q, final_reference.q);
      if (elapsed_s >= duration_s && final_error <= kTerminalToleranceRad &&
          stop_reason == nullptr)
        stop_reason = "motion_completed";
      if (elapsed_s >= duration_s + kMaximumSettleS && stop_reason == nullptr)
        stop_reason = "state_timeout";
      if (stop_reason != nullptr && stop_request_s == 0.0) stop_request_s = elapsed_s;

      std::array<double, kJoints> target_velocity{};
      if (stop_reason == nullptr) {
        for (std::size_t joint = 0; joint < kJoints; ++joint) {
          target_velocity[joint] = std::clamp(
              reference.dq[joint] +
                  kPositionCorrectionGain * (reference.q[joint] - current.q[joint]),
              -kVelocityCaps[joint], kVelocityCaps[joint]);
        }
      }
      const auto previous = commanded_velocity;
      commanded_velocity = franka::limitRate(
          kVelocityCaps, lower_velocity, kAccelerationCaps, kJerkCaps,
          target_velocity, commanded_velocity, commanded_acceleration);
      for (std::size_t joint = 0; joint < kJoints; ++joint)
        commanded_acceleration[joint] =
            (commanded_velocity[joint] - previous[joint]) / franka::kDeltaT;

      const double maximum_live_dq = maximum_abs(current.dq);
      if (stop_reason != nullptr && maximum_live_dq < kStationaryDqRadS &&
          maximum_abs(commanded_velocity) < 1e-3)
        ++stationary_samples;
      else if (stop_reason != nullptr)
        stationary_samples = 0;
      Sample sample;
      sample.wall_time_ns = wall_start_ns + static_cast<std::int64_t>(
          std::llround(elapsed_s * 1.0e9));
      sample.time_s = elapsed_s;
      sample.maximum_tracking_error_rad = tracking_error;
      sample.maximum_abs_dq_rad_s = maximum_live_dq;
      sample.stopping = stop_reason == nullptr ? 0 : 1;
      sample.q = current.q;
      sample.dq = current.dq;
      sample.q_reference = reference.q;
      sample.xyz = {{current.O_T_EE[12], current.O_T_EE[13], current.O_T_EE[14]}};
      samples.push_back(sample);

      franka::JointVelocities output(commanded_velocity);
      const double settling = stop_reason == nullptr ? 0.0 : elapsed_s - stop_request_s;
      if (stop_reason != nullptr &&
          ((settling >= kMinimumSettleS &&
            stationary_samples >= kRequiredStationarySamples) ||
           settling >= kMaximumSettleS))
        return franka::MotionFinished(output);
      return output;
    };
    robot.control(callback);
    const auto final_state = robot.readOnce();
    const double terminal_error = maximum_error(final_state.q, final_reference.q);
    write_csv(argv[4], samples);
    std::cout << "[result] stop reason:      "
              << (stop_reason == nullptr ? "unknown" : stop_reason) << "\n"
              << "[result] maximum tracking:" << maximum_live_tracking_error << " rad\n"
              << "[result] terminal error:   " << terminal_error << " rad\n"
              << "[saved] " << argv[4] << "\n"
              << "No Hand or collision-setting command was issued.\n";
    return stop_reason != nullptr && std::string(stop_reason) == "motion_completed" ? 0 : 4;
  } catch (const franka::Exception& error) {
    std::cerr << "libfranka error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
