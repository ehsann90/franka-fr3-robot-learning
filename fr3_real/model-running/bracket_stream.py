#!/usr/bin/env python3
# Usage: from anywhere, run: python <path>/model-running/bracket_stream.py --help
"""Launch the bracket streamer or either feeder leg, with every flag from config.

The hand-typed invocations are 200-300 characters and were repeatedly broken by
terminal line-wrapping on paste -- a split in the wrong place silently dropped
--acceleration-caps or --min-z, which are the two flags that took the longest to
get right. Every value here comes from configs/bracket_policy_eval_v1.json, so
the launched command cannot drift from the commissioned one.

  bracket_stream.py streamer
  bracket_stream.py approach  --run_id B2_v2_ck6000
  bracket_stream.py transport --run_id B2_v2_ck6000

Paths are resolved from this file, so the working directory does not matter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

FR3_ROOT = Path(__file__).resolve().parent.parent


def nums(values) -> list[str]:
    return [f"{float(v):g}" for v in values]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=["streamer", "approach", "transport"])
    ap.add_argument("--run_id", help="required for approach/transport; the leg name is appended")
    ap.add_argument("--config", type=Path, default=FR3_ROOT / "configs/bracket_policy_eval_v1.json")
    ap.add_argument("--robot_ip", default="172.16.0.2")
    ap.add_argument("--dry_run", action="store_true", help="print the command instead of running it")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    st, pol, cams, guard, grip = (cfg["streaming"], cfg["policy"], cfg["cameras"],
                                  cfg["workspace_guard"], cfg["gripper"])
    lo, hi = guard["min_xyz"], guard["max_xyz"]

    if args.role == "streamer":
        binary = FR3_ROOT / st["streamer_binary"]
        if not binary.exists():
            raise SystemExit(f"streamer binary not built: {binary}")
        # The streamer's floor is min-z PLUS its own guard margin and braking
        # buffer, so the configured floor must be pre-compensated here.
        offset = float(st.get("guard_margin_z_m", 0.002)) + float(st.get("braking_buffer_m", 0.002))
        min_z = round(float(lo[2]) - offset, 6)
        if min_z < 0:
            raise SystemExit(f"configured floor {lo[2]} is below the streamer's own {offset} m margin")
        cmd = ["taskset", "-c", str(st["streamer_cpu"]), str(binary), args.robot_ip,
               "--command-port", str(st["command_port"]), "--state-port", str(st["state_port"]),
               "--policy-hz", f"{float(pol['control_hz']):g}",
               "--velocity-caps", *nums(st["velocity_caps_rad_s"]),
               "--acceleration-caps", *nums(st["acceleration_caps_rad_s2"]),
               "--jerk-caps", *nums(st["jerk_caps_rad_s3"]),
               "--min-x", f"{float(lo[0]):g}", "--max-x", f"{float(hi[0]):g}",
               "--min-y", f"{float(lo[1]):g}", "--max-y", f"{float(hi[1]):g}",
               "--min-z", f"{min_z:g}", "--max-z", f"{float(hi[2]):g}"]
        print(f"[bracket_stream] effective floor {float(lo[2]):g} m "
              f"({min_z:g} + {offset:g} margin), "
              f"{1000 * (0.0121 - float(lo[2])):.1f} mm below the training grasp height")
        print("[bracket_stream] type START at the prompt\n")
    else:
        if not args.run_id:
            raise SystemExit(f"--run_id is required for the {args.role} leg")
        cmd = ["taskset", "-c", str(st["feeder_cpus"]),
               sys.executable, str(FR3_ROOT / "model-running/streaming/stream_policy_chunks.py"),
               "--server-host", str(pol["server_host"]), "--server-port", str(pol["server_port"]),
               "--command-port", str(st["command_port"]), "--state-port", str(st["state_port"]),
               "--exterior-serial", str(cams["exterior_serial"]),
               "--wrist-serial", str(cams["wrist_serial"]),
               "--camera-width", str(cams["width"]), "--camera-height", str(cams["height"]),
               "--camera-fps", str(cams["fps"]),
               "--policy-hz", f"{float(pol['control_hz']):g}",
               "--execute-steps", str(st["execute_steps"])]
        if args.role == "approach":
            cmd += ["--gripper-closedness", "0.0",
                    "--stop-on-gripper-threshold", f"{float(grip['close_threshold']):g}"]
        else:
            cmd += ["--holding",
                    "--gripper-closedness", f"{float(grip['held_observation_closedness']):g}",
                    "--stop-on-release-threshold", f"{float(grip['release_threshold']):g}"]
        cmd += ["--run-id", f"{args.run_id}_{args.role}", "--prompt", str(pol["prompt"])]

    if args.dry_run:
        print(" ".join(repr(c) if " " in c else c for c in cmd))
        return
    os.chdir(FR3_ROOT)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
