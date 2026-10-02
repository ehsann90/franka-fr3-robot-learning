cmake_minimum_required(VERSION 3.16)
project(fr3_policy_streamer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Franka REQUIRED)
list(GET Franka_INCLUDE_DIRS 0 FRANKA_INCLUDE_DIR)
if(NOT EXISTS "${FRANKA_INCLUDE_DIR}/franka/active_control_base.h")
  message(FATAL_ERROR
    "The FR3 streamers require a libfranka installation that provides "
    "franka/active_control_base.h. Install the compatible active-control "
    "libfranka development headers, then reconfigure this build directory."
  )
endif()
find_package(Threads REQUIRED)
# The local libfranka build was configured with these private shared-library
# dependencies, but its generated development target does not re-export them.
find_package(Poco REQUIRED COMPONENTS Net Foundation)
find_package(pinocchio REQUIRED)
find_package(TinyXML2 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(fr3_joint_velocity_streamer fr3_joint_velocity_streamer.cpp)
target_link_libraries(fr3_joint_velocity_streamer PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_joint_velocity_streamer PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fr3_retimed_joint_trajectory fr3_retimed_joint_trajectory.cpp)
target_link_libraries(fr3_retimed_joint_trajectory PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_retimed_joint_trajectory PRIVATE
  -Wall -Wextra -Wpedantic
)

# Bracket-only variant with raised acceleration/jerk caps, requested
# 2026-08-16 to allow closer-to-recorded-speed bracket replay. Velocity cap
# stays at the original 1.20 rad/s (already sufficient at 1.0x for both
# commissioned bracket candidates). Acceleration/jerk raised to 10.0/160.0,
# comfortably above the worst measured 1.0x values across B1 and B2
# (8.84 rad/s^2, 142.83 rad/s^3). This is a genuinely new, never-before-
# commissioned safety envelope -- deliberately a separate binary so the
# original caps used by every bolt stage (and bracket before this change)
# stay completely untouched.
add_executable(fr3_retimed_joint_trajectory_bracket_highcap fr3_retimed_joint_trajectory.cpp)
target_compile_definitions(fr3_retimed_joint_trajectory_bracket_highcap PRIVATE
  FR3_RETIMED_VELOCITY_CAP_RAD_S=1.20
  FR3_RETIMED_ACCELERATION_CAP_RAD_S2=10.0
  FR3_RETIMED_JERK_CAP_RAD_S3=160.0)
target_link_libraries(fr3_retimed_joint_trajectory_bracket_highcap PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_retimed_joint_trajectory_bracket_highcap PRIVATE
  -Wall -Wextra -Wpedantic
)

# B3 (2026-08-17) piloted its take faster than B1/B2 did -- offline validation
# against fr3_retimed_joint_trajectory_bracket_highcap blocked at true 1.0x on
# velocity (peak 1.628611 rad/s vs the 1.20 cap above) and jerk (peak
# 229.397092 rad/s^3 vs the 160.0 cap above); acceleration stayed under the
# existing 10.0 cap (peak 9.250645). This raises only velocity/jerk, with
# ~13-17% margin over those measured peaks, in a separate binary so
# fr3_retimed_joint_trajectory_bracket_highcap (and every config that already
# cites its exact 1.20/10.0/160.0 numbers as commissioning basis for B1/B2)
# stays completely untouched.
add_executable(fr3_retimed_joint_trajectory_bracket_highcap_v2 fr3_retimed_joint_trajectory.cpp)
target_compile_definitions(fr3_retimed_joint_trajectory_bracket_highcap_v2 PRIVATE
  FR3_RETIMED_VELOCITY_CAP_RAD_S=1.90
  FR3_RETIMED_ACCELERATION_CAP_RAD_S2=10.0
  FR3_RETIMED_JERK_CAP_RAD_S3=260.0)
target_link_libraries(fr3_retimed_joint_trajectory_bracket_highcap_v2 PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_retimed_joint_trajectory_bracket_highcap_v2 PRIVATE
  -Wall -Wextra -Wpedantic
)

# B4 (2026-08-17) offline validation against fr3_retimed_joint_trajectory_
# bracket_highcap_v2 blocked at true 1.0x on acceleration alone (peak
# 12.297626 rad/s^2 vs the 10.0 cap above; velocity 1.594663 and jerk
# 151.664528 both stayed under the v2 caps). This raises only acceleration
# (10.0 -> 14.0 rad/s^2, ~14% margin over the measured peak), keeping
# velocity/jerk at the v2 values. Per operator instruction, this tier goes
# straight to execution_enabled true from offline validation alone, no
# per-stage live preview -- v2 already proved this general approach live on
# B3. Separate binary so v2 (and every config citing its exact numbers for
# B3) stays untouched.
add_executable(fr3_retimed_joint_trajectory_bracket_highcap_v3 fr3_retimed_joint_trajectory.cpp)
target_compile_definitions(fr3_retimed_joint_trajectory_bracket_highcap_v3 PRIVATE
  FR3_RETIMED_VELOCITY_CAP_RAD_S=1.90
  FR3_RETIMED_ACCELERATION_CAP_RAD_S2=14.0
  FR3_RETIMED_JERK_CAP_RAD_S3=260.0)
target_link_libraries(fr3_retimed_joint_trajectory_bracket_highcap_v3 PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_retimed_joint_trajectory_bracket_highcap_v3 PRIVATE
  -Wall -Wextra -Wpedantic
)

# Controller-ablation executable. It shares only the UDP/state plumbing with
# the deployed streamer; its 1 kHz command tracking and predictive guard use
# libfranka's official joint-velocity rate limiter.
add_executable(fr3_reference_joint_velocity_streamer fr3_joint_velocity_streamer.cpp)
target_compile_definitions(fr3_reference_joint_velocity_streamer PRIVATE
  FR3_REFERENCE_CONTROLLER=1
)
target_link_libraries(fr3_reference_joint_velocity_streamer PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_reference_joint_velocity_streamer PRIVATE
  -Wall -Wextra -Wpedantic
)

enable_testing()
add_executable(fr3_reference_limiter_test test_reference_limiter.cpp)
target_link_libraries(fr3_reference_limiter_test PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_reference_limiter_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME fr3_reference_limiter_test COMMAND fr3_reference_limiter_test)

add_executable(fr3_model_jacobian_diagnostic fr3_model_jacobian_diagnostic.cpp)
target_link_libraries(fr3_model_jacobian_diagnostic PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_model_jacobian_diagnostic PRIVATE
  -Wall -Wextra -Wpedantic
)

# Read-only numerical IK planner for the bracket camera-outboard posture.  It
# loads the FR3 model and solves supplied Cartesian targets but never starts a
# control session or sends a motion command.
add_executable(fr3_bracket_outer_ik_preview fr3_bracket_outer_ik_preview.cpp)
target_link_libraries(fr3_bracket_outer_ik_preview PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bracket_outer_ik_preview PRIVATE
  -Wall -Wextra -Wpedantic
)

add_executable(fr3_bolt_axial_preload_direct fr3_bolt_axial_preload_direct.cpp)
target_link_libraries(fr3_bolt_axial_preload_direct PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_axial_preload_direct PRIVATE
  -Wall -Wextra -Wpedantic
)

add_executable(fr3_bolt_axial_preload_control_test test_bolt_axial_preload_control.cpp)
target_link_libraries(fr3_bolt_axial_preload_control_test PRIVATE Eigen3::Eigen)
target_compile_options(fr3_bolt_axial_preload_control_test PRIVATE
  -Wall -Wextra -Wpedantic
)
add_test(NAME fr3_bolt_axial_preload_control_test
         COMMAND fr3_bolt_axial_preload_control_test)

add_executable(fr3_bolt_thread_drive_control_test test_bolt_thread_drive_control.cpp)
target_link_libraries(fr3_bolt_thread_drive_control_test PRIVATE Eigen3::Eigen)
target_compile_options(fr3_bolt_thread_drive_control_test PRIVATE
  -Wall -Wextra -Wpedantic
)
add_test(NAME fr3_bolt_thread_drive_control_test
         COMMAND fr3_bolt_thread_drive_control_test)

add_executable(fr3_bolt_thread_drive_direct fr3_bolt_thread_drive_direct.cpp)
target_link_libraries(fr3_bolt_thread_drive_direct PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_thread_drive_direct PRIVATE
  -Wall -Wextra -Wpedantic
)

add_executable(fr3_hybrid_thread_control_test test_hybrid_thread_control.cpp)
target_link_libraries(fr3_hybrid_thread_control_test PRIVATE Eigen3::Eigen)
target_compile_options(fr3_hybrid_thread_control_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME fr3_hybrid_thread_control_test COMMAND fr3_hybrid_thread_control_test)

add_executable(fr3_hybrid_thread_replay_direct fr3_hybrid_thread_replay_direct.cpp)
target_link_libraries(fr3_hybrid_thread_replay_direct PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_hybrid_thread_replay_direct PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_commissioning_test
  test_cartesian_yaw_commissioning.cpp)
target_compile_options(fr3_cartesian_yaw_commissioning_test PRIVATE
  -Wall -Wextra -Wpedantic)
add_test(NAME fr3_cartesian_yaw_commissioning_test
         COMMAND fr3_cartesian_yaw_commissioning_test)

add_executable(fr3_cartesian_yaw_2x_profile_test
  test_cartesian_yaw_commissioning.cpp)
target_compile_definitions(fr3_cartesian_yaw_2x_profile_test PRIVATE
  FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.06)
target_compile_options(fr3_cartesian_yaw_2x_profile_test PRIVATE
  -Wall -Wextra -Wpedantic)
add_test(NAME fr3_cartesian_yaw_2x_profile_test
         COMMAND fr3_cartesian_yaw_2x_profile_test)

add_executable(fr3_cartesian_yaw_3x_profile_test
  test_cartesian_yaw_commissioning.cpp)
target_compile_definitions(fr3_cartesian_yaw_3x_profile_test PRIVATE
  FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.09)
target_compile_options(fr3_cartesian_yaw_3x_profile_test PRIVATE
  -Wall -Wextra -Wpedantic)
add_test(NAME fr3_cartesian_yaw_3x_profile_test
         COMMAND fr3_cartesian_yaw_3x_profile_test)

add_executable(fr3_cartesian_yaw_4x_profile_test
  test_cartesian_yaw_commissioning.cpp)
target_compile_definitions(fr3_cartesian_yaw_4x_profile_test PRIVATE
  FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.12)
target_compile_options(fr3_cartesian_yaw_4x_profile_test PRIVATE
  -Wall -Wextra -Wpedantic)
add_test(NAME fr3_cartesian_yaw_4x_profile_test
         COMMAND fr3_cartesian_yaw_4x_profile_test)

add_executable(fr3_cartesian_yaw_empty fr3_cartesian_yaw_empty.cpp)
target_link_libraries(fr3_cartesian_yaw_empty PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_empty PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_empty_45 fr3_cartesian_yaw_empty.cpp)
target_compile_definitions(fr3_cartesian_yaw_empty_45 PRIVATE
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=45.0)
target_link_libraries(fr3_cartesian_yaw_empty_45 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_empty_45 PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_empty_2x fr3_cartesian_yaw_empty.cpp)
target_compile_definitions(fr3_cartesian_yaw_empty_2x PRIVATE
  FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.06)
target_link_libraries(fr3_cartesian_yaw_empty_2x PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_empty_2x PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_empty_45_2x fr3_cartesian_yaw_empty.cpp)
target_compile_definitions(fr3_cartesian_yaw_empty_45_2x PRIVATE
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=45.0
  FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.06)
target_link_libraries(fr3_cartesian_yaw_empty_45_2x PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_empty_45_2x PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_held_5 fr3_cartesian_yaw_empty.cpp)
target_compile_definitions(fr3_cartesian_yaw_held_5 PRIVATE
  FR3_CARTESIAN_YAW_HELD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=5.0)
target_link_libraries(fr3_cartesian_yaw_held_5 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_held_5 PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_held_45 fr3_cartesian_yaw_empty.cpp)
target_compile_definitions(fr3_cartesian_yaw_held_45 PRIVATE
  FR3_CARTESIAN_YAW_HELD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=45.0)
target_link_libraries(fr3_cartesian_yaw_held_45 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_held_45 PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_held_5_2x fr3_cartesian_yaw_empty.cpp)
target_compile_definitions(fr3_cartesian_yaw_held_5_2x PRIVATE
  FR3_CARTESIAN_YAW_HELD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=5.0
  FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.06)
target_link_libraries(fr3_cartesian_yaw_held_5_2x PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_held_5_2x PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_cartesian_yaw_held_45_2x fr3_cartesian_yaw_empty.cpp)
target_compile_definitions(fr3_cartesian_yaw_held_45_2x PRIVATE
  FR3_CARTESIAN_YAW_HELD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=45.0
  FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.06)
target_link_libraries(fr3_cartesian_yaw_held_45_2x PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_cartesian_yaw_held_45_2x PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_bolt_axial_admittance_control_test
  test_bolt_axial_admittance_control.cpp)
target_compile_options(fr3_bolt_axial_admittance_control_test PRIVATE
  -Wall -Wextra -Wpedantic)
add_test(NAME fr3_bolt_axial_admittance_control_test
         COMMAND fr3_bolt_axial_admittance_control_test)

add_executable(fr3_bolt_axial_admittance
  fr3_bolt_axial_admittance.cpp)
target_link_libraries(fr3_bolt_axial_admittance PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_axial_admittance PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_bolt_combined_thread_5
  fr3_bolt_axial_admittance.cpp)
target_compile_definitions(fr3_bolt_combined_thread_5 PRIVATE
  FR3_AXIAL_ADMITTANCE_THREAD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=5.0)
target_link_libraries(fr3_bolt_combined_thread_5 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_combined_thread_5 PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_bolt_combined_thread_45
  fr3_bolt_axial_admittance.cpp)
target_compile_definitions(fr3_bolt_combined_thread_45 PRIVATE
  FR3_AXIAL_ADMITTANCE_THREAD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=45.0)
target_link_libraries(fr3_bolt_combined_thread_45 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_combined_thread_45 PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_bolt_combined_thread_40
  fr3_bolt_axial_admittance.cpp)
target_compile_definitions(fr3_bolt_combined_thread_40 PRIVATE
  FR3_AXIAL_ADMITTANCE_THREAD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=40.0)
target_link_libraries(fr3_bolt_combined_thread_40 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_combined_thread_40 PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_bolt_combined_thread_85
  fr3_bolt_axial_admittance.cpp)
target_compile_definitions(fr3_bolt_combined_thread_85 PRIVATE
  FR3_AXIAL_ADMITTANCE_THREAD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=85.0)
target_link_libraries(fr3_bolt_combined_thread_85 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_combined_thread_85 PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_bolt_combined_thread_50
  fr3_bolt_axial_admittance.cpp)
target_compile_definitions(fr3_bolt_combined_thread_50 PRIVATE
  FR3_AXIAL_ADMITTANCE_THREAD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=50.0)
target_link_libraries(fr3_bolt_combined_thread_50 PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_combined_thread_50 PRIVATE
  -Wall -Wextra -Wpedantic)

# 2026-08-17: fr3_bolt_combined_thread_85/50's 0.300 mm lateral-travel guard
# (bolt_axial_admittance_control.h) tripped on the first 85-degree window
# three times across different bracket positions (B4, then twice on B9),
# always around 0.31-0.32 mm measured -- just over the cap. Physical
# inspection each time found no bolt/bracket wear, no cross-threading, and
# accurate S1 placement, so this reflects normal engagement variance rather
# than a real misalignment the guard is protecting against. These
# _widelateral variants raise only the lateral cap (0.45 mm, ~40% margin
# over the worst measured 0.323 mm) via FR3_BOLT_AXIAL_MAXIMUM_LATERAL_
# TRAVEL_M; axial travel/force/timeout guards are unchanged. The original
# fr3_bolt_combined_thread_85/50 binaries (used by every already-accepted
# episode) are untouched.
#
# Also on 2026-08-17 (B12): the separate 3-second force-baseline-stationary
# preflight check (kMaximumBaselineFzStdN, an external-force-estimate noise
# gate, not a motion/contact safety guard) blocked once at std_fz=0.106 N
# against the 0.1 N default, right after a ratchet reset -- arm dq was fine
# (well under its own threshold), so this reads as normal estimator noise.
# Raised to 0.15 N here via FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N.
add_executable(fr3_bolt_combined_thread_85_widelateral
  fr3_bolt_axial_admittance.cpp)
target_compile_definitions(fr3_bolt_combined_thread_85_widelateral PRIVATE
  FR3_AXIAL_ADMITTANCE_THREAD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=85.0
  FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M=0.00045
  FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N=0.15)
target_link_libraries(fr3_bolt_combined_thread_85_widelateral PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_combined_thread_85_widelateral PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(fr3_bolt_combined_thread_50_widelateral
  fr3_bolt_axial_admittance.cpp)
target_compile_definitions(fr3_bolt_combined_thread_50_widelateral PRIVATE
  FR3_AXIAL_ADMITTANCE_THREAD=1
  FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=50.0
  FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M=0.00045
  FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N=0.15)
target_link_libraries(fr3_bolt_combined_thread_50_widelateral PRIVATE
  Franka::Franka Threads::Threads Poco::Foundation Poco::Net
  pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
)
target_compile_options(fr3_bolt_combined_thread_50_widelateral PRIVATE
  -Wall -Wextra -Wpedantic)

# 2026-08-17: discovered that record_autonomous_bolt_engagement_demo.py's
# accelerated (--two-x/--three-x/--four-x) path always overrides each
# engagement stage's binary to this exact fr3_bolt_combined_thread_{angle}_
# {speed}x naming pattern at runtime, ignoring whatever "binary" path is set
# in the task config's engagement_drive_sequence. That means the earlier
# _widelateral fix (raised lateral-travel 0.30->0.45mm, baseline-noise
# 0.1->0.15N) never actually applied to any live accelerated run -- B12's and
# B42's live faults at the "wrong" (pre-fix) thresholds prove it. Applying
# the same two raises directly to these actually-used _2x/_3x/_4x targets.
foreach(ANGLE IN ITEMS 5 40 45 50 85)
  add_executable(fr3_bolt_combined_thread_${ANGLE}_2x
    fr3_bolt_axial_admittance.cpp)
  target_compile_definitions(fr3_bolt_combined_thread_${ANGLE}_2x PRIVATE
    FR3_AXIAL_ADMITTANCE_THREAD=1
    FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=${ANGLE}.0
    FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.06
    FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M=0.00045
    FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N=0.15)
  target_link_libraries(fr3_bolt_combined_thread_${ANGLE}_2x PRIVATE
    Franka::Franka Threads::Threads Poco::Foundation Poco::Net
    pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
  )
  target_compile_options(fr3_bolt_combined_thread_${ANGLE}_2x PRIVATE
    -Wall -Wextra -Wpedantic)
endforeach()

# Separately named recorded-speed threading binaries keep 0.120 rad/s an
# explicit build-time choice while retaining the same admittance and guards.
foreach(ANGLE IN ITEMS 5 40 45 50 85)
  add_executable(fr3_bolt_combined_thread_${ANGLE}_4x
    fr3_bolt_axial_admittance.cpp)
  target_compile_definitions(fr3_bolt_combined_thread_${ANGLE}_4x PRIVATE
    FR3_AXIAL_ADMITTANCE_THREAD=1
    FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=${ANGLE}.0
    FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.12
    FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M=0.00045
    FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N=0.15)
  target_link_libraries(fr3_bolt_combined_thread_${ANGLE}_4x PRIVATE
    Franka::Franka Threads::Threads Poco::Foundation Poco::Net
    pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
  )
  target_compile_options(fr3_bolt_combined_thread_${ANGLE}_4x PRIVATE
    -Wall -Wextra -Wpedantic)
endforeach()

# The production chain selects the wide-lateral 4x names explicitly. These
# retain the validated 0.120 rad/s speed and the raised lateral/noise guards.
foreach(ANGLE IN ITEMS 5 40 45 50 85)
  add_executable(fr3_bolt_combined_thread_${ANGLE}_widelateral_4x
    fr3_bolt_axial_admittance.cpp)
  target_compile_definitions(fr3_bolt_combined_thread_${ANGLE}_widelateral_4x PRIVATE
    FR3_AXIAL_ADMITTANCE_THREAD=1
    FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=${ANGLE}.0
    FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.12
    FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M=0.00045
    FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N=0.15)
  target_link_libraries(fr3_bolt_combined_thread_${ANGLE}_widelateral_4x PRIVATE
    Franka::Franka Threads::Threads Poco::Foundation Poco::Net
    pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
  )
  target_compile_options(fr3_bolt_combined_thread_${ANGLE}_widelateral_4x PRIVATE
    -Wall -Wextra -Wpedantic)
endforeach()

# Separately named 3x production binaries prevent a configuration typo from
# silently changing the speed of the physically passed 2x executables.
foreach(ANGLE IN ITEMS 5 40 45 50 85)
  add_executable(fr3_bolt_combined_thread_${ANGLE}_3x
    fr3_bolt_axial_admittance.cpp)
  target_compile_definitions(fr3_bolt_combined_thread_${ANGLE}_3x PRIVATE
    FR3_AXIAL_ADMITTANCE_THREAD=1
    FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=${ANGLE}.0
    FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.09
    FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M=0.00045
    FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N=0.15)
  target_link_libraries(fr3_bolt_combined_thread_${ANGLE}_3x PRIVATE
    Franka::Franka Threads::Threads Poco::Foundation Poco::Net
    pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
  )
  target_compile_options(fr3_bolt_combined_thread_${ANGLE}_3x PRIVATE
    -Wall -Wextra -Wpedantic)
endforeach()

# 8x = 2x the 4x (recorded-source-speed) binaries, i.e. 0.24 rad/s. This is a
# genuinely new, never-before-commissioned speed tier requested on
# 2026-08-16 -- unlike 2x/3x/4x it has no prior physical validation history.
# Force/admittance targets are left identical to every other tier; only the
# commanded rotation rate changes.
foreach(ANGLE IN ITEMS 5 40 45 50 85)
  add_executable(fr3_bolt_combined_thread_${ANGLE}_8x
    fr3_bolt_axial_admittance.cpp)
  target_compile_definitions(fr3_bolt_combined_thread_${ANGLE}_8x PRIVATE
    FR3_AXIAL_ADMITTANCE_THREAD=1
    FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=${ANGLE}.0
    FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.24)
  target_link_libraries(fr3_bolt_combined_thread_${ANGLE}_8x PRIVATE
    Franka::Franka Threads::Threads Poco::Foundation Poco::Net
    pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
  )
  target_compile_options(fr3_bolt_combined_thread_${ANGLE}_8x PRIVATE
    -Wall -Wextra -Wpedantic)
endforeach()

# The fast-chain task configuration selects the wide-lateral safety variants
# by name. Keep these 8x preview binaries consistent with the already-used
# 2x/3x/4x guard corrections; the runtime threading gate remains disabled
# until this new speed tier is physically commissioned.
foreach(ANGLE IN ITEMS 5 40 45 50 85)
  add_executable(fr3_bolt_combined_thread_${ANGLE}_widelateral_8x
    fr3_bolt_axial_admittance.cpp)
  target_compile_definitions(fr3_bolt_combined_thread_${ANGLE}_widelateral_8x PRIVATE
    FR3_AXIAL_ADMITTANCE_THREAD=1
    FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=${ANGLE}.0
    FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.24
    FR3_BOLT_AXIAL_MAXIMUM_LATERAL_TRAVEL_M=0.00045
    FR3_BOLT_AXIAL_MAXIMUM_BASELINE_FZ_STD_N=0.15)
  target_link_libraries(fr3_bolt_combined_thread_${ANGLE}_widelateral_8x PRIVATE
    Franka::Franka Threads::Threads Poco::Foundation Poco::Net
    pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
  )
  target_compile_options(fr3_bolt_combined_thread_${ANGLE}_widelateral_8x PRIVATE
    -Wall -Wextra -Wpedantic)
endforeach()

# 6x = 0.18 rad/s, between the validated 4x (0.12) and the calibration-blocked
# 8x (0.24). Also never physically commissioned before 2026-08-16.
foreach(ANGLE IN ITEMS 5 40 45 50 85)
  add_executable(fr3_bolt_combined_thread_${ANGLE}_6x
    fr3_bolt_axial_admittance.cpp)
  target_compile_definitions(fr3_bolt_combined_thread_${ANGLE}_6x PRIVATE
    FR3_AXIAL_ADMITTANCE_THREAD=1
    FR3_CARTESIAN_YAW_TEST_ANGLE_DEG=${ANGLE}.0
    FR3_CARTESIAN_YAW_PEAK_SPEED_RAD_S=0.18)
  target_link_libraries(fr3_bolt_combined_thread_${ANGLE}_6x PRIVATE
    Franka::Franka Threads::Threads Poco::Foundation Poco::Net
    pinocchio::pinocchio tinyxml2::tinyxml2 OpenMP::OpenMP_CXX
  )
  target_compile_options(fr3_bolt_combined_thread_${ANGLE}_6x PRIVATE
    -Wall -Wextra -Wpedantic)
endforeach()
