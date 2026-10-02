#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/print_bracket_streaming_commands.py
"""Emit the exact streamer + feeder commands for a bracket policy trial.

The bracket eval must run on fr3_joint_velocity_streamer, not on the franky
chunked runner. The streamer holds a persistent 1 kHz libfranka loop and swaps
each new 16-step plan in at the action index implied by its age, so the arm is
never ramped to rest between chunks. Every cap below comes from
configs/bracket_policy_eval_v1.json rather than the streamer's built-in
defaults, which are the CUBE task's values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FR3_ROOT = Path(__file__).resolve().parent.parent


def joined(values) -> str:
    return " ".join(f"{float(v):g}" for v in values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=FR3_ROOT / "configs/bracket_policy_eval_v1.json")
    ap.add_argument("--robot_ip", default="172.16.0.2")
    ap.add_argument("--run_id", default="B2_stream_01")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    st, pol, cams, guard, grip = (cfg["streaming"], cfg["policy"], cfg["cameras"],
                                  cfg["workspace_guard"], cfg["gripper"])
    lo, hi = list(guard["min_xyz"]), guard["max_xyz"]
    # The streamer enforces min-z + guard_margin[z] + braking_buffer (+ a
    # speed-dependent braking distance), so the value handed to --min-z must be
    # pre-compensated or the effective floor lands above the 0.0121 m training
    # grasp height and the guard fires at the moment of the grasp.
    floor_offset = float(st.get("guard_margin_z_m", 0.002)) + float(st.get("braking_buffer_m", 0.002))
    streamer_min_z = round(lo[2] - floor_offset, 6)
    if streamer_min_z < 0:
        raise SystemExit(f"workspace_guard.min_xyz[2]={lo[2]} is below the streamer's own "
                         f"{floor_offset} m margin; the floor cannot be expressed.")

    streamer = " ".join([
        f"taskset -c {st['streamer_cpu']}", f"./{Path(st['streamer_binary']).relative_to('model-running')}"
        if str(st["streamer_binary"]).startswith("model-running/") else st["streamer_binary"],
        args.robot_ip,
        f"--command-port {st['command_port']}", f"--state-port {st['state_port']}",
        f"--policy-hz {pol['control_hz']:g}",
        f"--velocity-caps {joined(st['velocity_caps_rad_s'])}",
        f"--acceleration-caps {joined(st['acceleration_caps_rad_s2'])}",
        f"--jerk-caps {joined(st['jerk_caps_rad_s3'])}",
        f"--min-x {lo[0]:g} --max-x {hi[0]:g}",
        f"--min-y {lo[1]:g} --max-y {hi[1]:g}",
        f"--min-z {streamer_min_z:g} --max-z {hi[2]:g}",
    ])

    def feeder_cmd(phase: str) -> str:
        parts = [
            f"taskset -c {st['feeder_cpus']}", "python model-running/streaming/stream_policy_chunks.py",
            f"--server-host {pol['server_host']} --server-port {pol['server_port']}",
            f"--command-port {st['command_port']} --state-port {st['state_port']}",
            f"--exterior-serial {cams['exterior_serial']} --wrist-serial {cams['wrist_serial']}",
            f"--camera-width {cams['width']} --camera-height {cams['height']} --camera-fps {cams['fps']}",
            f"--policy-hz {pol['control_hz']:g} --execute-steps {st['execute_steps']}",
        ]
        if phase == "approach":
            parts.append(f"--gripper-closedness 0.0")
            parts.append(f"--stop-on-gripper-threshold {grip['close_threshold']:g}")
        else:
            parts.append(f"--holding --gripper-closedness {grip['held_observation_closedness']:g}")
            parts.append(f"--stop-on-release-threshold {grip['release_threshold']:g}")
        parts.append(f"--run-id {args.run_id}_{phase}")
        parts.append(f"--prompt {json.dumps(pol['prompt'])}")
        return " ".join(parts)

    print("=" * 78)
    print("BRACKET STREAMING EVAL")
    print("=" * 78)
    print("""
The streamer exits when the feeder stops at a gripper decision, so a trial is
FOUR phases, not one: approach -> grasp -> transport -> release. The streamer
must be restarted for the transport phase. This mirrors run_streaming_validation_
trial.py, which is how the cube evals were actually run.
""".strip())

    print("\n[0] policy server -- on the training machine, once per checkpoint:")
    print(f"  ssh intern2@{pol['server_host']}")
    print("  cd ~/franka_r3/openpi && .venv/bin/python scripts/serve_policy.py --port "
          f"{pol['server_port']} policy:checkpoint \\")
    print("      --policy.config=pi05_fr3_bracket_full \\")
    print("      --policy.dir=$HOME/franka_r3/openpi/checkpoints/pi05_fr3_bracket_full/"
          "fr3_bracket_full_20260818/<STEP>")

    print("\n[1] reset the arm to the trained start pose, then place the bracket:")
    print("  python model-running/run_bracket_policy_eval.py --trial_id reset "
          "--checkpoint_label na --reset_only")
    print("  python model-validation/align_bracket_to_trained_pose.py --slot B2")

    print(f"\n    (--min-z {streamer_min_z:g} = configured floor {lo[2]:g} minus the streamer's own "
          f"{floor_offset:g} m margin+buffer, so the EFFECTIVE floor is {lo[2]:g} m at rest -- "
          f"{1000*(0.0121 - lo[2]):.1f} mm below the 0.0121 m training grasp height.)")

    print("\n[2] APPROACH -- terminal A (streamer, from fr3_real/model-running/), type START at its prompt:")
    print(f"  {streamer}")
    print("    terminal B (feeder, from fr3_real/):")
    print(f"  {feeder_cmd('approach')}")
    print("    the feeder stops itself and prints 'policy requested gripper closure'.")

    print("\n[3] GRASP -- no arm motion; run after the streamer has exited:")
    print(f"  python model-running/manual_fr3_policy_grasp.py --width {grip['grasp_target_width_m']:g} "
          f"--speed {grip['grasp_speed_m_s']:g} --force {grip['grasp_force_n']:g}")
    print(f"    a real grasp reads {grip['held_width_min_m']:g}-{grip['held_width_max_m']:g} m "
          "(training measured 0.0383 m). Anything else: score the trial FAIL here.")

    print("\n[4] TRANSPORT -- restart the streamer in terminal A (same command as [2]), then:")
    print(f"  {feeder_cmd('transport')}")
    print("    the feeder stops itself and prints 'policy requested gripper release'.")

    print("\n[5] RELEASE -- only if the bracket is visibly seated on the mount pins:")
    print(f"  python model-running/manual_fr3_policy_grasp.py --release "
          f"--open-speed {grip['open_speed_m_s']:g}")

    print("\nCtrl+C in either the feeder or the streamer stops arm motion.")
    print("=" * 78)


if __name__ == "__main__":
    main()
