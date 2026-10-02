#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/run_bracket_policy_eval.py --help
"""One guarded, autonomous bracket pick-and-place policy evaluation trial.

Derived from the proven cube runner (run_fr3_policy_eval_trial.py) with the
policy loop kept structurally identical, but every task-specific constant
replaced by values measured from the 50 bracket training captures and loaded
from configs/bracket_policy_eval_v1.json. Carrying the cube defaults over would
silently look like policy failure -- most importantly its 0.10 rad/s velocity
cap, which would clip 24% of this policy's timesteps.

The human places the bracket, approves the single grasp and the single release,
and scores the trial. Ctrl+C calls robot.stop().

Run model-validation/bracket_eval_preflight.py first. It proves the observation
pipeline reproduces training, so a failure here is attributable to the policy.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from openpi_client import image_tools, websocket_client_policy

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(FR3_ROOT / "model-validation"))
sys.path.insert(0, str(FR3_ROOT / "model-running"))

from commission_fr3_policy_motion import JOINT_MAX, JOINT_MIN  # noqa: E402
from shadow_fr3_policy import read_rgb, start_camera  # noqa: E402


def vec(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(-1)


def require_server(host: str, port: int, timeout: float = 5.0) -> None:
    """Fail fast if the policy server is down.

    openpi_client's WebsocketClientPolicy retries a refused connection forever
    with a 5 s sleep, so without this probe an unstarted server looks like a
    hang rather than an error.
    """
    import socket
    with socket.socket() as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, int(port)))
        except OSError as exc:
            raise SystemExit(
                f"\nPolicy server unreachable at {host}:{port}  ({exc})\n\n"
                "Start it on the training machine (uv run fails on evdev; use the venv python):\n"
                f"  ssh intern2@{host}\n"
                "  cd ~/franka_r3/openpi\n"
                f"  .venv/bin/python scripts/serve_policy.py --port {port} policy:checkpoint \\\n"
                "      --policy.config=pi05_fr3_bracket_full \\\n"
                "      --policy.dir=$HOME/franka_r3/openpi/checkpoints/pi05_fr3_bracket_full/"
                "fr3_bracket_full_20260818/<STEP>\n"
            ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=FR3_ROOT / "configs/bracket_policy_eval_v1.json")
    ap.add_argument("--trial_id", required=True, help="e.g. E01_pos_left_far")
    ap.add_argument("--checkpoint_label", required=True, help="which checkpoint is being served, e.g. 6000")
    ap.add_argument("--robot_ip", default="172.16.0.2")
    ap.add_argument("--max_chunks", type=int, default=120)
    ap.add_argument("--reset_dynamics_factor", type=float, default=0.05)
    ap.add_argument("--reset_max_joint_delta", type=float, default=0.5)
    ap.add_argument("--match_training_resize", action="store_true",
                    help="Reproduce the converter's 640x480->320x240 INTER_AREA step before "
                         "resize_with_pad. Only set this if the preflight reports it scores better.")
    ap.add_argument("--throttle_ratio_warn", type=float, default=0.70,
                    help="warn if achieved joint displacement falls below this fraction of commanded")
    ap.add_argument("--log_dir", type=Path, default=FR3_ROOT / "eval_logs")
    ap.add_argument("--skip_reset", action="store_true", help="assume the arm is already at the start pose")
    ap.add_argument("--reset_only", action="store_true",
                    help="reset to the trained start pose, open the hand, and exit. Use this before a "
                         "streaming trial, which drives the arm through fr3_joint_velocity_streamer instead.")
    ap.add_argument("--no_continuous", action="store_true",
                    help="issue every chunk as a separate blocking motion (the pre-2026-08-18 behaviour). "
                         "This ramps the arm to a full stop at every chunk boundary and is a train/eval "
                         "mismatch; use it only to reproduce an old trial.")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    pol, grip_cfg, mot, guard, start, cams = (cfg["policy"], cfg["gripper"], cfg["motion"],
                                              cfg["workspace_guard"], cfg["start_pose"], cfg["cameras"])
    caps = np.asarray(mot["joint_velocity_caps_rad_s"], dtype=float)
    q_start = np.asarray(start["q"], dtype=float)
    lo = np.asarray(guard["min_xyz"], float)
    hi = np.asarray(guard["max_xyz"], float)
    margin = float(mot["joint_margin_rad"])
    ws_scale = float(mot.get("workspace_displacement_scale", 1.0))
    train_caps = mot["training_executor_caps"]
    continuous = bool(mot.get("continuous_streaming", True)) and not args.no_continuous
    lead_s = float(mot.get("continuous_lead_s", 0.03))
    j_lo, j_hi = JOINT_MIN + margin, JOINT_MAX - margin
    dt = 1.0 / float(pol["control_hz"])
    exec_steps = int(pol["execute_steps"])
    gmax = float(grip_cfg["max_width_m"])

    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_dir / f"bracket_eval_{args.trial_id}_{stamp}.jsonl"
    res_path = log_path.with_suffix(".result.json")

    print("=" * 78)
    print("BRACKET POLICY EVALUATION -- one guarded autonomous trial")
    print("=" * 78)
    print(f"trial       : {args.trial_id}")
    print(f"checkpoint  : {args.checkpoint_label}")
    print(f"server      : {pol['server_host']}:{pol['server_port']}")
    print(f"prompt      : {pol['prompt']!r}")
    print(f"control     : {pol['control_hz']} Hz, execute {exec_steps}/{pol['action_shape'][0]} per chunk")
    print(f"vel caps    : {np.array2string(caps, precision=2)} rad/s")
    print(f"dynamics    : matched to the training executor -- vel<={train_caps['velocity_rad_s']} rad/s, "
          f"acc<={train_caps['acceleration_rad_s2']} rad/s^2, jerk<={train_caps['jerk_rad_s3']} rad/s^3")
    print(f"streaming   : {'continuous (chunks overlap, arm never stops mid-reach)' if continuous else 'blocking per chunk'}")
    print(f"gripper     : close>={grip_cfg['close_threshold']} release<={grip_cfg['release_threshold']}")
    print(f"log         : {log_path}")
    print("=" * 78)

    from franky import (  # noqa: PLC0415
        Duration, Frame, Gripper, JointMotion, JointState,
        JointVelocityStopMotion, JointVelocityWaypoint, JointVelocityWaypointMotion,
        RelativeDynamicsFactor, Robot,
    )

    robot = Robot(args.robot_ip)
    gripper = Gripper(args.robot_ip)

    def killswitch(signum, frame):
        print("\n[KILLSWITCH] Ctrl+C -- calling robot.stop() now.")
        try:
            robot.stop()
        finally:
            sys.exit(130)

    signal.signal(signal.SIGINT, killswitch)

    result = {"status": "not_started", "trial_id": args.trial_id,
              "checkpoint": args.checkpoint_label, "log": str(log_path)}
    exterior = wrist = None
    throttle_flags = 0
    infer_total = 0.0
    motion_total = 0.0
    chunks_done = 0
    motion_end = None
    loop_t0 = None
    gap_total = 0.0
    try:
        # ---------- guarded reset to the fixed policy start pose ----------
        if not args.skip_reset:
            q_now = vec(robot.state.q)
            delta = float(np.max(np.abs(q_start - q_now)))
            segments = max(1, int(np.ceil(delta / args.reset_max_joint_delta)))
            print(f"\n[RESET] {delta:.3f} rad from start pose; {segments} guarded segment(s)")
            if np.any(q_start < j_lo) or np.any(q_start > j_hi):
                raise RuntimeError("configured start pose violates guarded joint limits")
            input("[RESET] Confirm the workspace is clear of the bracket and hands, then press Enter: ")
            robot.relative_dynamics_factor = args.reset_dynamics_factor
            for s in range(1, segments + 1):
                target = q_now + (s / segments) * (q_start - q_now)
                robot.move(JointMotion(JointState(target)))
                err = float(np.max(np.abs(vec(robot.state.q) - target)))
                print(f"  [reset {s}/{segments}] terminal error {err:.6f} rad")
            final_err = float(np.max(np.abs(vec(robot.state.q) - q_start)))
            if final_err > 0.01:
                raise RuntimeError(f"reset did not reach start pose (error {final_err:.4f} rad)")
            print(f"[RESET] at start pose (error {final_err:.6f} rad)")

        # Hand must start open; the policy's first frames were all recorded open.
        gripper.open(float(grip_cfg["open_speed_m_s"]))
        w = float(gripper.state.width)
        if w < float(grip_cfg["minimum_open_width_m"]):
            raise RuntimeError(f"Hand did not open (width {w:.4f} m)")
        print(f"[RESET] Hand open at {w:.4f} m")

        if args.reset_only:
            result.update(status="reset_only")
            print("[RESET] reset_only: at the trained start pose with the hand open. "
                  "Start the streamer and feeder now.")
            return

        print("\n[SETUP] Place the bracket at the position under test.")
        print("[SETUP] It must lie flat on the table in its recorded orientation, inside")
        print(f"[SETUP] the trained pick envelope x[0.25,0.48] y[-0.47,0.39] (metres, TCP frame).")
        input("[SETUP] Press Enter when the bracket is placed and you are clear of both camera views: ")

        # Reproduce the training executor's dynamics envelope exactly.
        #
        # A single relative_dynamics_factor scales velocity, acceleration and
        # jerk together, so the 0.9 used before 2026-08-18 matched training
        # velocity while permitting far more jerk than any training motion ever
        # used -- the arm slammed at every chunk boundary. franky takes the
        # three factors independently, so each is derived from this robot's live
        # limit and the cap the trajectory binaries were compiled with.
        def limit_array(limit) -> np.ndarray:
            for attr in ("max", "value", "upper"):
                if hasattr(limit, attr):
                    return np.asarray(getattr(limit, attr), dtype=float).reshape(-1)
            return np.asarray(limit, dtype=float).reshape(-1)

        live = {
            "velocity": (limit_array(robot.joint_velocity_limit), float(train_caps["velocity_rad_s"])),
            "acceleration": (limit_array(robot.joint_acceleration_limit), float(train_caps["acceleration_rad_s2"])),
            "jerk": (limit_array(robot.joint_jerk_limit), float(train_caps["jerk_rad_s3"])),
        }
        rdf = {}
        for axis, (values, cap) in live.items():
            binding = float(values.min())
            if binding <= 0:
                raise RuntimeError(f"robot reported a non-positive {axis} limit: {values}")
            rdf[axis] = min(cap / binding, 1.0)
            print(f"[DYNAMICS] {axis:<12} live min {binding:9.3f} -> factor {rdf[axis]:.4f} "
                  f"(reproduces the training cap {cap:g})")
        if float(np.min(live["velocity"][0])) * rdf["velocity"] < float(caps.max()) - 1e-9:
            raise RuntimeError(
                f"velocity factor {rdf['velocity']:.4f} would throttle the policy: it permits "
                f"{float(np.min(live['velocity'][0])) * rdf['velocity']:.3f} rad/s but the caps are "
                f"{float(caps.max()):.3f} rad/s")
        robot.relative_dynamics_factor = RelativeDynamicsFactor(
            rdf["velocity"], rdf["acceleration"], rdf["jerk"])
        result["dynamics_factors"] = {k: round(v, 5) for k, v in rdf.items()}

        require_server(pol["server_host"], pol["server_port"])
        client = websocket_client_policy.WebsocketClientPolicy(pol["server_host"], pol["server_port"])
        exterior = start_camera(cams["exterior_serial"], cams["width"], cams["height"], cams["fps"])
        wrist = start_camera(cams["wrist_serial"], cams["width"], cams["height"], cams["fps"])
        for _ in range(15):
            read_rgb(exterior); read_rgb(wrist)

        waypoint_ms = max(1, round(1000.0 * dt))
        chunk_period = exec_steps * dt
        holding = False
        # Streaming state. motion_end is when the currently-issued chunk is due
        # to finish; prep_ema is a running estimate of how long capture +
        # inference + guarding takes, so the next chunk can be prepared during
        # the tail of the current one instead of after it.
        prep_ema = 0.20
        q_at_issue = None
        prev_cmd_disp = None

        with log_path.open("w") as log_file:
            for chunk in range(1, args.max_chunks + 1):
                if continuous and motion_end is not None:
                    # Sleep through the part of the chunk we do not need, so the
                    # observation is still only ~prep_ema old when it takes effect.
                    prep_start = motion_end - lead_s - prep_ema
                    while time.perf_counter() < prep_start:
                        time.sleep(0.002)
                t_prep = time.perf_counter()
                ext_rgb, wri_rgb = read_rgb(exterior), read_rgb(wrist)
                if args.match_training_resize:
                    import cv2  # noqa: PLC0415
                    ext_rgb = cv2.resize(ext_rgb, (320, 240), interpolation=cv2.INTER_AREA)
                    wri_rgb = cv2.resize(wri_rgb, (320, 240), interpolation=cv2.INTER_AREA)
                state = robot.state
                q = vec(state.q)
                closed = float(np.clip(1.0 - float(gripper.state.width) / gmax, 0.0, 1.0))
                observation = {
                    "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(ext_rgb, 224, 224)),
                    "observation/wrist_image_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wri_rgb, 224, 224)),
                    "observation/joint_position": q.astype(np.float32),
                    "observation/gripper_position": np.asarray([closed], dtype=np.float32),
                    "prompt": pol["prompt"],
                }
                t0 = time.perf_counter()
                actions = np.asarray(client.infer(observation)["actions"], dtype=float)
                latency = time.perf_counter() - t0
                infer_total += latency
                chunks_done += 1
                if tuple(actions.shape) != tuple(pol["action_shape"]) or not np.isfinite(actions).all():
                    raise RuntimeError(f"invalid policy actions {actions.shape}")

                grip_plan = actions[:exec_steps, 7]
                close_req = float(grip_plan.max()) >= float(grip_cfg["close_threshold"])
                release_req = holding and float(grip_plan.max()) <= float(grip_cfg["release_threshold"])

                if release_req:
                    if continuous and motion_end is not None:
                        robot.move(JointVelocityStopMotion()); motion_end = None
                    d = input("[POLICY] Release requested. Type RELEASE only if the bracket is visibly "
                              "seated on the mount pins; anything else leaves the arm stationary: ").strip().upper()
                    if d == "RELEASE":
                        gripper.open(float(grip_cfg["open_speed_m_s"]))
                        verdict = input("[EVAL] Bracket released. Type PASS or FAIL: ").strip().upper()
                        result.update(status="released", chunk=chunk,
                                      assessment=verdict if verdict in {"PASS", "FAIL"} else "UNSCORED")
                    else:
                        result.update(status="release_not_approved", chunk=chunk)
                        print("[POLICY] Release declined; arm stationary.")
                    break

                if close_req and not holding:
                    if continuous and motion_end is not None:
                        robot.move(JointVelocityStopMotion()); motion_end = None
                    d = input("[POLICY] Grasp requested. Type CLOSE to let it grasp; "
                              "anything else stops the trial: ").strip().upper()
                    if d != "CLOSE":
                        result.update(status="grasp_declined", chunk=chunk)
                        break
                    accepted = gripper.grasp(float(grip_cfg["grasp_target_width_m"]),
                                             float(grip_cfg["grasp_speed_m_s"]),
                                             float(grip_cfg["grasp_force_n"]),
                                             float(grip_cfg["grasp_epsilon_inner_m"]),
                                             float(grip_cfg["grasp_epsilon_outer_m"]))
                    gw = float(gripper.state.width)
                    banded = float(grip_cfg["held_width_min_m"]) <= gw <= float(grip_cfg["held_width_max_m"])
                    holding = bool((accepted and gripper.is_grasped) or banded)
                    print(f"[GRIPPER] accepted={accepted} is_grasped={gripper.is_grasped} "
                          f"width={gw:.4f} banded={banded} -> holding={holding}")
                    if not holding:
                        result.update(status="grasp_failed", chunk=chunk, gripper_width=gw)
                        break

                raw = actions[:exec_steps, :7]
                vel = np.clip(raw, -caps, caps)
                clipped = float(np.mean(np.abs(raw) > caps))
                step_q = np.cumsum(vel * dt, axis=0)
                pred_q = q + step_q * float(mot["guard_displacement_scale"])
                if np.any(pred_q < j_lo) or np.any(pred_q > j_hi):
                    raise RuntimeError("predicted chunk violates guarded joint limits")
                tcp = np.asarray(state.O_T_EE.matrix, float)
                j7 = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, float)
                j7_to_tcp = np.linalg.solve(j7, tcp)
                # The workspace box is extrapolated at its own (1x) scale. The 2x
                # joint-limit lookahead predicts a sub-table z during any normal
                # grasp descent, which aborted trial T00_B2_aligned at chunk 6
                # with the TCP still 93 mm above the floor.
                ws_q = q + step_q * ws_scale
                pred_xyz = np.asarray([
                    (np.asarray(robot.model.pose(Frame.Joint7, s, state.F_T_EE, state.EE_T_K).matrix,
                                float) @ j7_to_tcp)[:3, 3] for s in ws_q])
                if np.any(pred_xyz < lo) or np.any(pred_xyz > hi):
                    raise RuntimeError(f"predicted chunk leaves guarded workspace: "
                                       f"min={pred_xyz.min(0)} max={pred_xyz.max(0)}")

                cmd_disp = float(np.abs(vel * dt).sum())
                motion = JointVelocityWaypointMotion(
                    [JointVelocityWaypoint(v, minimum_time=Duration(waypoint_ms)) for v in vel])

                prep_ema = 0.7 * prep_ema + 0.3 * (time.perf_counter() - t_prep)

                if continuous:
                    # Issue just before the running chunk ends so the arm is never
                    # commanded to rest mid-reach. Waiting past motion_end would
                    # reintroduce exactly the stop-start profile this replaces.
                    if motion_end is not None:
                        while time.perf_counter() < motion_end - lead_s:
                            time.sleep(0.001)
                    q_before = vec(robot.state.q)
                    t_issue = time.perf_counter()
                    if motion_end is not None:
                        gap_total += max(0.0, t_issue - motion_end)
                    robot.move(motion, asynchronous=True)
                    motion_end = t_issue + chunk_period
                    if loop_t0 is None:
                        loop_t0 = t_issue
                    # The chunk just issued has not executed yet, so throttle is
                    # scored against the PREVIOUS chunk, whose full period elapsed
                    # while this one was being prepared.
                    act_disp = float(np.abs(q_before - q_at_issue).sum()) if q_at_issue is not None else 0.0
                    scored_cmd = prev_cmd_disp
                    q_at_issue = q_before
                    prev_cmd_disp = cmd_disp
                else:
                    q_before = vec(robot.state.q)
                    t_issue = time.perf_counter()
                    if motion_end is not None:
                        gap_total += max(0.0, t_issue - motion_end)
                    if loop_t0 is None:
                        loop_t0 = t_issue
                    robot.move(motion)
                    motion_end = time.perf_counter()
                    motion_total += motion_end - t_issue
                    act_disp = float(np.abs(vec(robot.state.q) - q_before).sum())
                    scored_cmd = cmd_disp

                # Throttle detection: if franky's dynamics limits are binding, the
                # arm silently under-executes and the policy looks slow/incompetent.
                if scored_cmd is None or scored_cmd <= 1e-6:
                    ratio, throttled = 1.0, False
                else:
                    ratio = act_disp / scored_cmd
                    throttled = scored_cmd > 1e-3 and ratio < args.throttle_ratio_warn
                if throttled:
                    throttle_flags += 1

                end_xyz = np.asarray(robot.state.O_T_EE.translation, float)
                if np.any(end_xyz < lo) or np.any(end_xyz > hi):
                    raise RuntimeError(f"actual TCP outside guarded workspace: {end_xyz}")

                rec = {"chunk": chunk, "latency_s": latency, "q": q.tolist(),
                       "tcp": tcp[:3, 3].tolist(), "raw_peak": float(np.abs(raw).max()),
                       "commanded_peak": float(np.abs(vel).max()), "clipped_fraction": clipped,
                       "commanded_disp": cmd_disp, "achieved_disp": act_disp,
                       "throttle_ratio": ratio, "throttled": throttled,
                       "gripper_plan": grip_plan.tolist(), "holding": holding}
                log_file.write(json.dumps(rec) + "\n"); log_file.flush()
                flag = "  <<THROTTLED" if throttled else ""
                clip = f" clip={clipped*100:.0f}%" if clipped > 0 else ""
                print(f"[{chunk:03d}/{args.max_chunks}] lat={latency:.3f}s "
                      f"tcp={np.array2string(tcp[:3,3], precision=3)} "
                      f"peak={rec['raw_peak']:.2f}{clip} exec={ratio:.2f} "
                      f"grip={grip_plan.min():.2f}..{grip_plan.max():.2f} hold={holding}{flag}")
            else:
                result.update(status="max_chunks_reached", chunk=args.max_chunks)

        robot.move(JointVelocityStopMotion())

    except Exception as exc:
        result.update(status="aborted", error=str(exc))
        print(f"\n[ABORT] {exc}")
        try:
            robot.stop()
        except Exception:
            pass
    finally:
        for cam in (exterior, wrist):
            if cam is not None:
                try:
                    cam.stop()
                except Exception:
                    pass
        result["throttled_chunks"] = throttle_flags
        result["streaming"] = "continuous" if continuous else "blocking"
        if loop_t0 is not None and motion_end is not None and motion_end > loop_t0:
            wall = motion_end - loop_t0
            result["motion_duty_cycle"] = round(max(0.0, 1.0 - gap_total / wall), 3)
            result["command_gap_s"] = round(gap_total, 3)
            result["chunks_executed"] = chunks_done
            result["mean_infer_s"] = round(infer_total / max(chunks_done, 1), 4)
        result["finished_at"] = datetime.now().isoformat()
        res_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n[RESULT] {json.dumps({k: v for k, v in result.items() if k != 'log'})}")
        if result.get("motion_duty_cycle") is not None:
            d = result["motion_duty_cycle"]
            if continuous:
                print(f"[TIMING] motion duty cycle {d:.1%} -- {result['command_gap_s']:.2f} s total with no "
                      "command active. Chunks overlap, so inference no longer stalls the arm and it is "
                      "never ramped to rest mid-reach.")
            else:
                print(f"[TIMING] motion duty cycle {d:.1%} (the rest is inference stall between chunks). "
                      "Blocking mode also ramps the arm to a full stop at every chunk boundary, which is a "
                      "motion profile absent from the training data. Drop --no_continuous to remove both.")
        if result.get("throttled_chunks"):
            print(f"[WARNING] {result['throttled_chunks']} chunk(s) under-executed. The arm did not "
                  "follow commanded velocity -- raise relative_dynamics_factor before trusting this trial.")
        print(f"[RESULT] {res_path}")


if __name__ == "__main__":
    main()
