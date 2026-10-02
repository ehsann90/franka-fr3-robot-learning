# FR3 Bracket + Bolt/Threading Project Handoff

> Historical snapshot: this file records the August 2026 commissioning state.
> Its episode counts, latest-capture claims, IK blockers and recommended next
> steps are not the current wall-cell collection status. Start with the
> [current bracket/bolt workflow guide](../docs/BRACKET_BOLT_WORKFLOWS.md)
> and the [FR3 directory map](README.md). Keep this handoff as the detailed
> account of the earlier bracket, bolt and threading setup.

This document is the operational handoff for the FR3 bracket-placement and 3D-printed bolt/threading project. It is written so another engineer or LLM can continue from the repository and recordings without relying on chat history.

Repository root:

```text
/home/salaar/Job/franka-fr3-robot-learning/fr3_real
```

Robot:

```text
Franka Research 3 (FR3)
robot IP: 172.16.0.2
camera exterior: RealSense serial 336222074819
camera wrist:    RealSense serial 243322074869
```

Use the `fr3-recording` environment for Python work. The system/base Python is missing some recording dependencies such as OpenCV.

```bash
conda activate fr3-recording
# or use this interpreter explicitly:
/home/salaar/miniforge3/envs/fr3-recording/bin/python
```

## Project goal

The physical task has two objects and two useful policy boundaries:

1. Bracket task: pick the 3D-printed bracket from B1, place it onto the frame/wall alignment pins, release it, and retreat.
2. Bolt task: pick the bolt from S1, insert it into the mounted bracket, make it upright/self-supported, regrasp the head, and perform thread engagement/tightening.

The threading controller is a contact-rich force/admittance controller, not ordinary visual joint-position replay. The intended policy variants are:

- Full-policy dataset: policy controls bracket placement, bolt pickup/insertion, and possibly the entire threading sequence.
- Hybrid dataset: policy controls bracket + bolt to a `THREAD_READY` state; a deterministic guarded thread controller performs tightening.

The DROID-style policy action is 8-dimensional: seven achieved-next absolute joint positions plus achieved-next gripper closedness. A pure policy cannot directly emit a named “trigger threading” action unless a convention or external phase controller is added. The hybrid design uses the policy endpoint/event/state as the trigger and keeps the guarded thread controller outside the policy action space.

## August source data snapshot

### Previously accepted full-assembly setup episode 9

```text
recordings/full_bracket_bolt_autonomous_raw_v1/20260814_194920_full_bracket_bolt_autonomous_0009_B1_S1_CENTER
```

At the time of this handoff, this was the accepted full bracket plus bolt
setup capture. The named August 14 folder is not present in the current
checkout, so do not use this path as an input without locating and auditing
the actual source. The repository now contains later bracket and assembly
captures with their own metadata and event boundaries.

Important derived boundaries used by the converter:

- bracket motion ends at the bracket release/retreat boundary;
- bolt `THREAD_READY` prefix ends before the deterministic threading controller;
- `training_motion_completed` is emitted after the final release and one stationary second;
- the final operator visual-review period is excluded from derived policy data.

Raw files inside an episode normally include:

```text
camera_0_336222074869_rgb.mp4       # exterior/wrist role depends on metadata order
camera_1_243322074869_rgb.mp4
*_timestamps.jsonl
robot_state.jsonl
actions.jsonl
events.jsonl
metadata.json
episode_arrays.npz
direct_drive_*.csv / retimed_*.csv  # when controller telemetry was recorded
```

At the time of this handoff the raw directory was described as containing
episode 9 only. That statement is no longer true for the current checkout.

### Latest manual bracket episode

```text
recordings/bracket_manual_raw_v1/20260816_133045_bracket_manual_0001_B1_CENTER
```

Metadata result:

```text
success: true
accepted_by_operator: true
bc_training_eligible: true
wrist_camera_side: OUTER
gripper state source: operator_marked_discrete_state
```

Measured capture audit:

```text
duration:                 ~64.5 s
robot states:             1898
exterior frames:          1936
wrist frames:             1937
state max gap:            ~40.9 ms
camera max gap:           ~35.4 ms
```

Events/actions are correctly ordered:

```text
manual_demo_started
bracket_grasp_pose_reached
bracket_gripper_closed
bracket_place_pose_reached
bracket_gripper_opened
training_motion_completed
manual_demo_finished
```

The long recording contains operator pauses. Edge-idle trimming leaves about 22.8 seconds of active arm movement before safe retiming. The recorded q7 trace reaches about `-3.0332 rad` while the arm is in the outer-side grasp posture. That is inside the Desk/Pilot soft-limit region but outside the autonomous replay guard used by `fr3_retimed_joint_trajectory`; it must not be copied directly into FCI replay.

### Current outer-wrist IK status

The read-only exact-pose rebalancer is:

```text
model-running/prepare_bracket_outer_ik_preview.py
model-running/streaming/fr3_bracket_outer_ik_preview.cpp
```

The command for the latest manual episode is:

```bash
python model-running/prepare_bracket_outer_ik_preview.py \
  --read-only-preview \
  --manual-episode recordings/bracket_manual_raw_v1/20260816_133045_bracket_manual_0001_B1_CENTER
```

It preserves the exact recorded TCP poses and uses the FR3 redundant joints to move q7 inward. It never starts a control session or commands the Hand. The current solver reaches 201/1898 targets and then reports a branch discontinuity at target 234. This is unresolved. Do not execute autonomous bracket replay until this candidate passes continuity, joint-limit, workspace, and dynamic audits.

## Safety rules

- A `READ_ONLY` or `PREVIEW` label is meaningful only if the program does not create a control session. Inspect the final output for that statement.
- Never bypass a controller blocker by weakening joint, workspace, velocity, acceleration, jerk, force, residual, or gripper guards without a new read-only audit.
- The Hand must be empty/open before automatic arm resets unless an object-aware stage explicitly says otherwise.
- Keep one hand near the physical emergency stop for every live arm trial.
- Keep operators outside both camera views during data capture where possible.
- Do not manually pull, push, or rotate a bolt during a guarded thread stage.
- Collision/reflex settings are intentionally unchanged throughout this project.
- After a direct controller process exits with an error, assume the Hand may still hold an object until visually and physically verified.
- Manual Desk/Pilot capture must not open a persistent `franky.Gripper` state client. The manual bracket recorder now follows the working manual bolt recorder pattern and leaves Pilot Hand control available.

## Recording workflows

### Manual bracket recording

The recorder is camera/state only. All arm and Hand motion comes from supported Franka Desk/Pilot controls.

```bash
python franka-tests/record_manual_bracket_demo.py \
  --record \
  --episode-index 1 \
  --slot B1 \
  --grasp CENTER \
  --wrist-side OUTER
```

Prompt sequence:

1. Type `RECORD_READ_ONLY`.
2. Put the bracket at B1, leave the bolt at S1, and choose a safe outer-side wrist posture with Pilot.
3. Type `START_RECORDING`.
4. Move to the center grasp pose and hold still; press Enter.
5. Use Pilot’s top Hand button to close; hold the arm still and press Enter.
6. Pilot lift/transport/place; hold the seated bracket and arm still; press Enter.
7. Use Pilot to open the Hand without moving the bracket; press Enter.
8. Retreat, hold still one second, type `TASK_COMPLETE`.
9. Inspect only after recording stops; type `SUCCESS` or `FAIL`.

`--wrist-side OUTER` is a metadata label and operator instruction. It does not apply an automatic q7 offset. Choose a comfortable collision-free whole-arm posture; exact 180-degree q7 shifts are not assumed safe.

### Manual bolt recording

```bash
python franka-tests/record_manual_bolt_demo.py \
  --record \
  --episode-index 1 \
  --slot S1 \
  --grasp CENTER
```

This recorder intentionally has no Hand client. It records explicit operator markers for open, pickup closed, head/thread-ready closed, ratchet transitions, release, and retreat. It is suitable when Desk/Pilot must retain exclusive Hand control.

### Autonomous commissioned full assembly

The main orchestrator is:

```text
model-running/record_autonomous_bolt_engagement_demo.py
```

Read-only recording preview:

```bash
python model-running/record_autonomous_bolt_engagement_demo.py \
  --recording-preview --full-assembly --two-x --episode-index N
```

Accepted setup/full-assembly recording (only after the relevant commissioned stages pass):

```bash
python model-running/record_autonomous_bolt_engagement_demo.py \
  --record-demo --full-assembly --two-x --episode-index N
```

Speed flags:

```text
--two-x    smooth 2x arm profile; threading peak about 0.060 rad/s
--three-x  cap-limited 3x target; threading peak about 0.090 rad/s
--four-x   cap-limited 4x target; threading peak about 0.120 rad/s
```

The speed name is a target, not a promise. Per-stage velocity/acceleration/jerk caps can stretch a stage. Never assume a faster label means the raw path is safe.

## Threading sequence

The commissioned 270-degree setup sequence was:

```text
pickup -> preinsert -> upright/self-supported -> open -> head regrasp
-> +85 deg -> stationary open/unload -> q7 reset/regrasp
-> +85 deg -> stationary open/unload -> q7 reset/regrasp
-> +50 deg -> stationary open/unload -> q7 reset/regrasp
-> +50 deg -> release
```

Each drive uses combined axial admittance and yaw:

```text
force target:             +1.200360 N relative to measured baseline
force-to-velocity gain:   0.416542 mm/(s*N)
maximum axial speed:      0.500 mm/s
maximum axial/lateral:    3.000 / 0.300 mm
hard axial cutoff:        +/-2.000 N
wrong-direction cutoff:   -0.500 N for 0.100 s
hard residual cutoff:      -0.538305 Nm
```

The 270-degree chain completed all six drives in episode 9, but the visual endpoint remains the authority for whether the printed screw is visibly engaged. Threading must not be inferred from yaw alone.

Useful direct-controller commands (all require their own setup/read-only checks):

```bash
python model-running/preview_bolt_thread_drive.py --help
python model-running/run_bolt_combined_thread_5.py --help
python model-running/run_bolt_thread_drive_direct.py --help
python model-running/run_bolt_ratchet_reset.py --help
python model-running/run_bolt_hybrid_thread_replay.py --help
```

The C++ contact/force executors are in `model-running/streaming/` and are built through its CMake file.

## Arm retiming and idle trimming

The smooth arm controller is:

```text
model-running/streaming/fr3_retimed_joint_trajectory.cpp
expert_demo/retimed_trajectory.py
```

It fits a clamped cubic through joint-position knots and runs a 1 kHz rate-limited `JointVelocities` controller. It audits:

```text
velocity cap:      1.2 rad/s
acceleration cap:  4.0 rad/s^2
jerk cap:          40.0 rad/s^3
tracking cutoff:   0.050 rad
terminal cutoff:   0.002 rad
```

Idle-edge trimming removes only stationary knots before the first and after the last moving interval, retains a short boundary buffer, and preserves interior pauses. It does not remove grasp/release transitions.

Existing smooth 2x commissioning:

```bash
python model-running/commission_full_assembly_arm_2x.py \
  --offline-validation

python model-running/commission_full_assembly_arm_2x.py \
  --read-only-preview --stage STAGE_NAME

python model-running/commission_full_assembly_arm_2x.py \
  --prepare-controller-preview --stage STAGE_NAME

python model-running/commission_full_assembly_arm_2x.py \
  --execute --stage STAGE_NAME
```

The seven commissioned arm stage names are:

```text
bracket_approach_open
bracket_transport_place_closed
bracket_retreat_open
bolt_approach_open
bolt_transport_preinsert_closed
bolt_lower_held
bolt_reposition_regrasp_open
```

## Data conversion and policy views

Raw capture validation:

```bash
python model-validation/validate_raw_recordings.py \
  recordings/full_bracket_bolt_autonomous_raw_v1
```

Convert accepted raw data to LeRobot/OpenPI-compatible DROID joint-position data:

```bash
python data-pipeline/convert_raw_recordings_to_lerobot.py \
  --raw_dir recordings/full_bracket_bolt_autonomous_raw_v1 \
  --output_dir datasets/fr3_full_policy \
  --repo_id local/fr3-full-policy \
  --schema droid_joint_position \
  --fps 15 \
  --derived_view whole \
  --overwrite
```

Hybrid prefix view (bracket + bolt through `THREAD_READY`, excluding deterministic threading):

```bash
python data-pipeline/convert_raw_recordings_to_lerobot.py \
  --raw_dir recordings/full_bracket_bolt_autonomous_raw_v1 \
  --output_dir datasets/fr3_hybrid_prefix \
  --repo_id local/fr3-hybrid-prefix \
  --schema droid_joint_position \
  --fps 15 \
  --derived_view two_task_hybrid \
  --overwrite
```

Full-policy derived view:

```bash
python data-pipeline/convert_raw_recordings_to_lerobot.py \
  --raw_dir recordings/full_bracket_bolt_autonomous_raw_v1 \
  --output_dir datasets/fr3_two_task_full_policy \
  --repo_id local/fr3-two-task-full-policy \
  --schema droid_joint_position \
  --fps 15 \
  --derived_view two_task_full_policy \
  --overwrite
```

The converter excludes failed and pending-review captures by default. Do not use `--include_failed` for training. `--include_source_trajectories` is for explicitly marked pilot source data and should be used only when intentionally building a mixed dataset.

Relevant converter behavior:

- reads `events.jsonl`, `robot_state.jsonl`, camera videos, and timestamps;
- emits 15 Hz DROID-style frames;
- uses achieved-next absolute joint positions for `droid_joint_position`;
- derives gripper closedness from measured or explicitly operator-marked width;
- uses `training_motion_completed` to exclude post-task operator inspection;
- supports whole, full-policy two-task, and hybrid `THREAD_READY` views.

## File-by-file map

### Recorders and hardware-facing Python

| File | Role |
|---|---|
| `franka-tests/realsense_recorder.py` | Shared RealSense video + robot-state recorder. Supports measured gripper state, operator-marked gripper state, timestamped events, MP4/timestamp output, and episode metadata. |
| `franka-tests/record_manual_bracket_demo.py` | Manual bracket capture. No arm or Hand commands; no persistent Hand client; explicit stationary open/closed markers; supports `OUTER` wrist label. |
| `franka-tests/record_manual_bolt_demo.py` | Manual bolt/thread capture. No Hand client; operator-marked pickup/head/thread/release states. |
| `franka-tests/record_piloted_bracket_trajectory.py` | Camera-free pilot source trajectory capture used to commission bracket replay. |
| `franka-tests/record_piloted_bolt_preinsert_trajectory.py` | Camera-free source capture for bolt pickup/preinsert. |
| `franka-tests/record_piloted_bolt_drop_regrasp.py` | Camera-free source capture for upright/self-supported drop and head regrasp. |
| `franka-tests/record_piloted_bolt_thread_engagement.py` | Manual/piloted threading source capture. |
| `franka-tests/record_bolt_axial_load_calibration.py` | Records axial force baseline/calibration data. |
| `franka-tests/record_bolt_preinsert_wrench_baseline.py` | Records preinsert force/torque baseline. |
| `franka-tests/calibrate_bolt_held_width.py` | Measures accepted head-held gripper width. |
| `franka-tests/calibrate_bolt_ratchet_limits.py` | Calibrates ratchet/wrist limits. |
| `franka-tests/monitor_bolt_contact_read_only.py` | Read-only contact/force monitoring. |
| `franka-tests/preview_realsense.py` | Camera smoke/preview. |
| `franka-tests/test_realsense_recording.py` | Camera recording tests. |
| `franka-tests/test_connection.py` | Basic robot connection/read-only checks. |
| `franka-tests/teach_expert_poses.py` | Guided expert pose collection. |
| `franka-tests/pick_and_place.py` | Older general pick-and-place experiments; not the canonical bracket/bolt pipeline. |

### Planning, state, safety, and replay math (`expert_demo/`)

| File | Role |
|---|---|
| `expert_demo/replay_plan.py` | Reads source trajectory/event boundaries and creates stage plans. |
| `expert_demo/retimed_trajectory.py` | Clamped cubic retiming, idle-edge trimming, dense velocity/acceleration/jerk audit, CSV serialization. |
| `expert_demo/replay_execution.py` | Replay command/velocity helpers and smoothing. |
| `expert_demo/object_episode_execution.py` | Static blockers, reset checks, gripper/workspace/velocity validation for object episodes. |
| `expert_demo/bracket_expert.py` | Bracket-specific expert trajectory/state definitions. |
| `expert_demo/bolt_preinsert.py` | Bolt pickup/preinsert stage construction. |
| `expert_demo/bolt_drop_regrasp.py` | Upright/self-supported and regrasp stage construction. |
| `expert_demo/full_bolt_replay_plan.py` | Full bolt replay stage composition. |
| `expert_demo/hybrid_thread_plan.py` | Hybrid bolt-to-thread-ready plan. |
| `expert_demo/piloted_thread_replay.py` | Thread replay stage planning. |
| `expert_demo/piloted_thread_guard.py` | Guard conditions for manual/piloted thread replay. |
| `expert_demo/thread_engagement.py` | Thread engagement state/endpoint logic. |
| `expert_demo/thread_drive_preview.py` | Read-only thread-drive preview calculations. |
| `expert_demo/ratchet_limits.py` | Ratchet reset/rotation limits. |
| `expert_demo/axial_load_calibration.py` | Axial target/baseline calculations. |
| `expert_demo/contact_detection.py` | Contact/residual detection. |
| `expert_demo/contact_search.py` | Contact-search logic. |
| `expert_demo/wrench_baseline.py` | Wrench baseline statistics. |
| `expert_demo/wrist_commissioning.py` | Wrist dynamic/reset commissioning helpers. |
| `expert_demo/calibration.py` | Calibration file loading/hashing. |
| `expert_demo/safety.py` | Generic safety checks. |
| `expert_demo/snapshots.py` | Robot/Hand snapshot serialization. |
| `expert_demo/transforms.py` | Pose/transform utilities. |
| `expert_demo/episode_spec.py` | Episode metadata/specification helpers. |
| `expert_demo/trajectory_recording.py` | Older camera-free pilot recording support. |

### Main model-running entry points

| File | Role |
|---|---|
| `model-running/record_autonomous_bolt_engagement_demo.py` | Main commissioned autonomous bolt or full bracket+bolt recorder, including retimed stages, Hand actions, cameras, and threading chain. |
| `model-running/replay_piloted_bracket_episode.py` | Legacy object-aware bracket replay with Hand grasp/release. Uses existing commissioned source trajectory/config. |
| `model-running/replay_piloted_bracket_stage.py` | Single bracket stage replay. |
| `model-running/prepare_piloted_bracket_replay.py` | Offline bracket source preparation. |
| `model-running/check_piloted_bracket_replay_start.py` | Read-only start-state check. |
| `model-running/reset_to_piloted_trajectory_start.py` | Guarded reset to source trajectory start. |
| `model-running/prepare_bracket_outer_ik_preview.py` | Read-only exact-pose outer/manual bracket IK candidate preparation. Current manual candidate stops at target 234 due branch discontinuity. |
| `model-running/commission_full_assembly_arm_2x.py` | Offline/read-only/prepare/execute smooth 2x arm stage commissioning. |
| `model-running/replay_piloted_bolt_preinsert_episode.py` | Object-aware bolt pickup/preinsert replay. |
| `model-running/replay_piloted_bolt_drop_regrasp_episode.py` | Object-aware upright/drop/regrasp replay. |
| `model-running/replay_piloted_*_arm_only.py` | Arm-only variants with Hand disabled. |
| `model-running/replay_piloted_thread_drive_guarded.py` | Guarded piloted thread replay. |
| `model-running/run_bolt_combined_thread_5.py` | Combined axial-admittance + fixed yaw test launcher. |
| `model-running/run_bolt_thread_drive_direct.py` | Direct libfranka threading controller launcher. |
| `model-running/run_bolt_hybrid_thread_replay.py` | Hybrid thread replay path. |
| `model-running/run_bolt_ratchet_reset.py` | Stationary open -> empty q7 reset -> head regrasp. |
| `model-running/run_bolt_axial_admittance.py` | Axial admittance launcher. |
| `model-running/run_bolt_axial_preload.py` / `_direct.py` | Axial preload variants. |
| `model-running/run_bolt_contact_search.py` | Contact search launcher. |
| `model-running/run_bolt_free_spin.py` | Free-spin/rotation experiment. |
| `model-running/preview_bolt_thread_drive.py` | Read-only force/yaw preview. |
| `model-running/commission_bolt_*` | Read-only/commissioning tools for Cartesian yaw, wrist reset, and wrist rotation. |
| `model-running/guarded_fr3_policy_arm_rollout.py` | Policy arm rollout with guards. |
| `model-running/commission_fr3_policy_motion.py` | Policy motion commissioning. |
| `model-running/run_fr3_policy_eval_trial.py` | Policy evaluation trial. |
| `model-running/manual_fr3_policy_grasp.py` | Manual Hand grasp helper. |
| `model-running/recover_held_cube_to_basket.py` | Recovery utility for another task; not bracket/bolt canonical flow. |

### C++ streaming/controllers

| File | Role |
|---|---|
| `model-running/streaming/fr3_retimed_joint_trajectory.cpp` | 1 kHz clamped-cubic joint trajectory executor with velocity/acceleration/jerk limiting, tracking, workspace, and joint guards. |
| `model-running/streaming/fr3_joint_velocity_streamer.cpp` | Generic direct joint-velocity streamer. |
| `model-running/streaming/fr3_bolt_axial_admittance.cpp` | Direct axial-admittance controller. |
| `model-running/streaming/fr3_bolt_axial_preload_direct.cpp` | Direct axial preload controller. |
| `model-running/streaming/fr3_bolt_thread_drive_direct.cpp` | Direct combined axial/yaw thread drive. |
| `model-running/streaming/fr3_hybrid_thread_replay_direct.cpp` | Hybrid thread controller. |
| `model-running/streaming/fr3_cartesian_yaw_empty.cpp` | Empty-wrist Cartesian yaw commissioning. |
| `model-running/streaming/fr3_model_jacobian_diagnostic.cpp` | Model/Jacobian diagnostic. |
| `model-running/streaming/fr3_bracket_outer_ik_preview.cpp` | Read-only FR3 model IK solver for exact TCP pose and joint-limit branch rebalancing. |
| `model-running/streaming/*_control.h` | Header-only control/limit logic for axial preload, admittance, thread, hybrid, and Cartesian yaw controllers. |
| `model-running/streaming/CMakeLists.txt` | Builds all direct controllers and tests, including the outer IK preview. |
| `model-running/streaming/test_*.cpp` | Native unit tests for force/thread/yaw/control limit logic. |
| `model-running/streaming/stream_protocol.py` | Python streaming protocol definitions/tests. |
| `model-running/streaming/replay_recorded_velocity_chunks.py` | Recorded velocity chunk replay utility. |
| `model-running/streaming/stream_policy_chunks.py` | Policy chunk streaming utility. |
| `model-running/streaming/warm_policy_server.py` | Policy server warm-up. |

### Configuration files

Canonical task configs:

| File | Role |
|---|---|
| `configs/bracket_manual_recording_v1.json` | Manual bracket camera/state recording, B1, grasp variations, operator-marked Hand widths. |
| `configs/bolt_manual_recording_v1.json` | Manual bolt/thread recording, S1, operator-marked Hand states. |
| `configs/autonomous_demo_collection_v1.json` | Autonomous bracket/bolt/full-assembly task definitions, stage paths, gripper settings, dataset policy. |
| `configs/piloted_bracket_trajectory_v1.json` | Camera-free bracket source trajectory configuration. |
| `configs/piloted_bolt_preinsert_v1.json` | Bolt preinsert source trajectory configuration. |
| `configs/piloted_bolt_drop_regrasp_v1.json` | Bolt drop/regrasp source configuration. |
| `configs/full_assembly_2x_commissioning_v1.json` | Smooth 2x arm stages, idle trimming, caps, reset and workspace guards. |
| `configs/full_assembly_3x_production_v1.json` | 3x target production configuration. |
| `configs/full_assembly_4x_production_v1.json` | 4x target/source-speed production configuration. |
| `configs/bolt_combined_thread_{5,40,45,50,85}_*.json` | Per-angle combined axial/yaw drive parameters and speed variants. |
| `configs/bolt_threading_v1.json` | Threading/engagement task parameters. |
| `configs/bolt_hybrid_thread_replay_v1.json` | Hybrid policy-to-thread-controller configuration. |
| `configs/bolt_full_replay_v1.json` | Full bolt replay configuration. |
| `configs/bolt_piloted_thread_replay_v1.json` | Piloted thread replay configuration. |
| `configs/bolt_ratchet_reset_v1.json` | Guarded stationary wrist reset/regrasp. |
| `configs/bolt_initial_prewind_reset_2x_v1.json` | Initial empty-wrist prewind/reset. |
| `configs/bolt_axial_admittance_v1.json` | Axial admittance values. |
| `configs/bolt_axial_preload_v1.json` | Axial preload values. |
| `configs/bolt_contact_v1.json` | Contact-search values. |
| `configs/bolt_free_spin_v1.json` | Free-spin values. |
| `configs/bolt_wrist_dynamic_baseline_v1.json` | Wrist dynamic baseline. |
| `configs/bolt_wrist_reset_commissioning_v1.json` | Wrist reset commissioning. |
| `configs/bolt_wrist_rotation_commissioning_v1.json` | Wrist rotation commissioning. |
| `configs/bracket_expert_v1.json` | Bracket expert/calibration values. |

The many `fr3_*cell*`, spatial-validation, tracker, and paper configs are part of the broader FR3 cell/paper workflow and are not required for this bracket/bolt handoff.

## Build and test

Build direct controllers:

```bash
cmake --build model-running/streaming/build -j2
```

Compile Python task scripts:

```bash
python -m py_compile \
  franka-tests/record_manual_bracket_demo.py \
  franka-tests/record_manual_bolt_demo.py \
  model-running/record_autonomous_bolt_engagement_demo.py \
  model-running/prepare_bracket_outer_ik_preview.py
```

Relevant Python tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

If the base interpreter lacks OpenCV, use:

```bash
/home/salaar/miniforge3/envs/fr3-recording/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The current targeted marker/conversion tests pass in `fr3-recording`:

```text
10 tests OK
```

## Recommended next steps

The steps below were written for the August outer-wrist candidate. They are
historical, not a prerequisite for the later bracket wall-cell loop. The
current loop uses translated per-cell candidates and a bracket-only capture;
see `model-running/record_bracket_workspace_cell_loop.py` and the current
workflow guide before choosing a new cell.

1. Rerun the manual exact-pose IK preview after the latest branch-continuity change.
2. Require a candidate audit with no discontinuity, hard-limit violations, and acceptable dense path errors.
3. Build a three-stage trimmed replay from that candidate: approach/open, transport/closed, retreat/open.
4. Perform read-only live start/reset/workspace validation.
5. Run one guarded autonomous bracket trial with the Hand sequence enabled: open -> grasp -> transport -> release -> retreat.
6. Review the autonomous result visually before using the bracket capture for policy training.
7. Convert episode 9 to full-policy and hybrid views, then add accepted bracket-only episodes to the bracket dataset.
8. Keep deterministic threading separate for the hybrid policy unless a separate policy-threading dataset has enough varied successful examples.

## Do not lose these facts

- The August 14 episode 9 path above is historical and is not present in the current checkout. Do not rely on it for conversion without locating the actual source.
- The latest manual bracket capture is accepted as data, but its raw q7 outer-side posture is not directly replay-safe.
- `--wrist-side OUTER` never means “add pi to q7”; it means the operator selected an outer camera posture.
- The Hand-client-disabled change is required for Pilot’s top gripper button to work during manual capture.
- A successful read-only preview is not a successful physical trial.
- A successful physical arm trial is not proof of thread engagement; visual endpoint review remains required.
