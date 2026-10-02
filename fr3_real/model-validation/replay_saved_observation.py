#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/replay_saved_observation.py --help
"""Ask the running policy server what it predicts for observations already on disk.

stream_policy_chunks.py saves every observation it sent: both images, the joint
positions, and the gripper closedness. Replaying one removes the robot, the
streamer, the cameras and the guards from the loop entirely, so a difference in
the answer can only come from the server -- a different checkpoint, a stale
process still holding the port, or genuine policy stochasticity.

The motivating case: checkpoint 4000 grasped at 12:44 (gripper 0.473) and then
refused to grasp on three later runs of the same checkpoint, with the bracket
provably unmoved (ROI diff 2.9-3.6 of 255).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image

FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = "Place the 3D-printed bracket onto the wall alignment pins."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True,
                    help="directory under eval_logs/stream_sessions, e.g. B2_stream_01_approach")
    ap.add_argument("--index", type=int, default=-1,
                    help="1-based observation index; -1 (default) replays the LAST one, "
                         "which is the frame at which the grasp decision was made")
    ap.add_argument("--repeat", type=int, default=5,
                    help="queries to run. pi0.5 is a flow-matching model and samples, so a single "
                         "query cannot distinguish a dead gripper channel from an unlucky draw.")
    ap.add_argument("--server-host", default="10.6.38.133")
    ap.add_argument("--server-port", type=int, default=8000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--close-threshold", type=float, default=0.35)
    args = ap.parse_args()

    root = FR3_ROOT / "eval_logs/stream_sessions" / args.session
    if not root.is_dir():
        raise SystemExit(f"no such session: {root}")

    obs_rows = [json.loads(line) for line in (root / "events.jsonl").open()]
    obs_rows = [row for row in obs_rows if row.get("event") == "observation"]
    if not obs_rows:
        raise SystemExit(f"{args.session} recorded no observation events")
    row = obs_rows[args.index if args.index < 0 else args.index - 1]

    def rgb(path: str) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    observation = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb(row["exterior_image"]), 224, 224)),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb(row["wrist_image"]), 224, 224)),
        "observation/joint_position": np.asarray(row["q"], dtype=np.float32),
        "observation/gripper_position": np.asarray([row["gripper_closedness"]], dtype=np.float32),
        "prompt": args.prompt,
    }

    print(f"session      : {args.session}")
    print(f"observation  : #{row['observation_index']} of {len(obs_rows)}")
    print(f"images       : {Path(row['exterior_image']).name}, {Path(row['wrist_image']).name}")
    print(f"server       : {args.server_host}:{args.server_port}")
    print(f"close thresh : {args.close_threshold}")
    print("-" * 70)

    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    fired = 0
    peaks = []
    for attempt in range(1, args.repeat + 1):
        actions = np.asarray(client.infer(observation)["actions"], dtype=float)
        grip = actions[:, 7]
        peak = float(grip.max())
        peaks.append(peak)
        crossed = peak >= args.close_threshold
        fired += crossed
        print(f"  query {attempt}: gripper {grip.min():+.3f} .. {peak:+.3f}   "
              f"arm |v|max {np.abs(actions[:, :7]).max():.3f}   "
              f"{'CLOSE' if crossed else '-'}")

    print("-" * 70)
    print(f"gripper peak over {args.repeat} queries: "
          f"min {min(peaks):.3f}  median {float(np.median(peaks)):.3f}  max {max(peaks):.3f}")
    print(f"crossed {args.close_threshold} in {fired}/{args.repeat} queries")
    if fired == 0 and max(peaks) < 0.05:
        print("\nThe gripper channel is flat at this observation. The policy is not "
              "\nnear the decision -- this is not a threshold or sampling problem.")
    elif 0 < fired < args.repeat:
        print("\nThe grasp is STOCHASTIC at this observation. A single live rollout can "
              "\nmiss it purely by sampling; that alone explains a run-to-run difference.")


if __name__ == "__main__":
    main()
