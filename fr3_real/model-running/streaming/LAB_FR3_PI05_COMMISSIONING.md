# FR3 lab π0.5 streaming commissioning

The selected profile is [`../../configs/fr3_lab_pi05_v1.json`](../../configs/fr3_lab_pi05_v1.json). It records robot `192.170.10.101`, exterior D435i `244222076317`, wrist D435i `342222073510`, server `127.0.0.1:8000`, 15 Hz, checkpoint `msbutt1/pi05-fr3-v3-12000`, and `pi05_droid`. The profile is deliberately marked uncommissioned. The launch wrapper and feeder require a separate approved record; neither changes the original workcell defaults.

## Read-only checks before setting bounds

From the repository root, activate the repository Python environment. Stop the `fr3_up` launch in its terminal with Ctrl-C and check `pgrep -af fr3_up` until it reports no active launch. A standalone FCI controller must be the only controller. Keep the workcell clear and E-stop reachable. Then record the current robot state with the existing read-only probe:

```sh
cd fr3_real
../.venv/bin/python franka-tests/diagnose_fr3_kinematics.py --robot_ip 192.170.10.101
```

The probe prints seven `q` values (rad), `O_T_EE` translation (m in robot base frame), `F_T_EE`, and `EE_T_K`. `O_T_EE` is the end-effector frame translation used by the streamer's guard; it is not necessarily the physical tool tip if an additional tool offset is present. Inspect the physical table, fixtures, camera placement, reachable task region, and tool geometry. Establish minimum and maximum **`O_T_EE` translations in the robot base frame** with clearance for the entire path, including stop travel. The original author's X `[0.18, 0.75]`, Y `[-0.50, 0.45]`, Z `[0.03, 0.65]` m are unvalidated for this laboratory and must not be copied as an approval by default. The guard margins remain `[0.005, 0.005, 0.002]` m.

Create a local record, for example `/tmp/fr3_lab_workspace_review.json`, after measurement and human review. Use this exact schema with measured values and reviewed bounds (the numbers below are **field descriptions, not suggested values**):

```json
{
  "workstation": "fr3_lab_pi05_v1",
  "approved": true,
  "reviewed_by": "operator name and date",
  "tcp_xyz_m": [0, 0, 0],
  "joint_positions_rad": [0, 0, 0, 0, 0, 0, 0],
  "workspace": {"min": [0, 0, 0], "max": [0, 0, 0]}
}
```

Replace **every** placeholder with actual readings and reviewed limits. The loader rejects the placeholders when the measured pose lies outside the bounds. It checks the current pose against the streamer's joint and workspace entry guards. A true approval is an operator decision, not evidence that the full path or stop tail is safe. Preserve the completed record with the session logs.

## Offline and observation checks

From repository root:

```sh
cmake -S fr3_real/model-running/streaming -B .local/build/fr3-streaming-ros-libfranka -DCMAKE_PREFIX_PATH="/opt/ros/jazzy;$HOME/franka_ros2_ws/install/libfranka" -DFranka_DIR="$HOME/franka_ros2_ws/install/libfranka/lib/cmake/Franka"
cmake --build .local/build/fr3-streaming-ros-libfranka --target fr3_reference_joint_velocity_streamer fr3_reference_limiter_test -j2
.venv/bin/python -m unittest discover -s fr3_real/model-running/streaming -p 'test_*.py'
ctest --test-dir .local/build/fr3-streaming-ros-libfranka -R fr3_reference_limiter_test --output-on-failure
.venv/bin/python fr3_real/model-validation/test_fr3_policy_server.py --host 127.0.0.1 --port 8000
```

The server must already be running with `pi05_droid`, the specified checkpoint, and its normalization statistics. Check its startup output and metadata: the profile stores the expected checkpoint label but cannot establish server identity when metadata omits it. Synthetic inference should return finite `(15, 8)` actions. Do not send synthetic actions to the robot.

For a robot-free check of four-step completion and STOP, start the reference binary with `dummy --dry-run --command-port 51022 --state-port 51023` in one terminal, then run `.venv/bin/python fr3_real/model-running/streaming/test_stream_protocol.py --port 51022 --state-port 51023 --bounded-four-step` in another. Expect `four-step plan completed; sending STOP` and `STOP acknowledged reason=command`.

With the stationary reference streamer publishing state, this read-only diagnostic receives one 176-byte protocol-v3 state and captures both cameras without sending any command:

```sh
.venv/bin/python fr3_real/model-running/streaming/diagnose_lab_observation.py --workstation-config fr3_real/configs/fr3_lab_pi05_v1.json
```

It reports the current joint and original workspace guard result, while always labeling the original workspace uncommissioned. Inspect camera orientation and scene content with the feeder's logged PNGs before motion. The live observation keys are `observation/exterior_image_1_left` and `observation/wrist_image_left`, RGB `uint8`, resized with padding to 224×224, plus seven joint positions in rad, one gripper closedness scalar, and the prompt. The dataset converter duplicates exterior video into `exterior_image_2_left` for the DROID dataset; the existing inference client omits that duplicate. No camera reordering, image flip, normalization change, or action conversion was made. Action channels 0–6 are joint velocities in rad/s; channel 7 is gripper closedness and is not commanded by the reference streamer. The feeder's default observation closedness is 0.0, so verify the actual gripper is open or explicitly provide a measured `--gripper-closedness`.

## Supervised one-chunk trial

After completing the record and observation review, stop `fr3_up` and verify it is gone again. Use separate terminals from the repository root. **Do not run these commands until the approved bounds and tool-frame assumptions have been checked on the robot.**

Terminal 1, with operator at the E-stop:

```sh
.venv/bin/python fr3_real/model-running/streaming/run_lab_streamer.py --workstation-config fr3_real/configs/fr3_lab_pi05_v1.json --commissioning-record /tmp/fr3_lab_workspace_review.json
```

Inspect the printed profile and bounds. Type `START` only when the workcell and robot are ready. The process starts the persistent reference FCI controller and prints protocol, state rate, limiter, plan acceptance/rejection, and final stop reason.

Terminal 2, after stationary state packets are visible and policy inference has been warmed:

```sh
.venv/bin/python fr3_real/model-running/streaming/stream_policy_chunks.py --workstation-config fr3_real/configs/fr3_lab_pi05_v1.json --commissioning-record /tmp/fr3_lab_workspace_review.json --max-chunks 1 --execute-steps 4 --replan-at-step 2 --gripper-closedness 0.0
```

Replace `0.0` with measured closedness if the gripper is not fully open. The feeder captures RGB images, logs the exact state timestamp, requests one plan, sends at most four joint velocity actions, waits for the **accepted** plan to finish, sends STOP, and waits up to two seconds for a state with nonzero stop reason. `--replan-at-step 2` applies only if another chunk is allowed; one-chunk mode does not replan. A guard may shorten or reject the plan. An inference failure or feeder interruption also enters the STOP path. The streamer holds a zero target after plan exhaustion and would stop on its 0.75 s watchdog if STOP is lost; its rate-limited stop tail must settle before FCI ends. The expected terminal lines are `[FEEDER] final requested plan completed`, a `stop_acknowledged` event in the feeder log, and `[STREAM] stopped cleanly reason=command`. A guard stop, stale plan, watchdog, controller fault, missing acknowledgement, or nonzero feeder exit requires investigation before another trial.

The feeder logs events and raw PNGs under `fr3_real/eval_logs/stream_sessions` unless `--log-dir` is set. Retain the exact checkpoint/server metadata, commissioning record, state trace, accepted step count, stop reason, and actual arm motion. Physical policy movement remains unvalidated until this supervised trial succeeds.

The selected `.local/build/fr3-streaming-ros-libfranka` build should be retained until validation is complete. The older `.local/build/fr3-streaming` cache references `.local/libfranka-0.17.0`, whereas the selected build resolves `~/franka_ros2_ws/install/libfranka`. After the physical trial and a dependency review, the older `.local/build/fr3-streaming`, `.local/build/libfranka-0.17.0`, `.local/libfranka-0.17.0`, and `.local/sources/libfranka-0.17.0` may be candidates for removal. No package or generated build is removed by these instructions.
