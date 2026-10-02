#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/screen_checkpoint_grasp.py --help
"""Does the served checkpoint still command a grasp where the demonstrations do?

Replaying an observation saved from an EARLIER policy's rollout only asks
"would you grasp from the pose that other policy chose". A better-trained policy
may legitimately want to keep moving there, so a flat gripper channel on that
frame is not proof the channel is broken.

This asks the unambiguous question instead. It rebuilds frames straight out of
the training captures -- the same frame selection, resize and action convention
the converter used -- picks the frames where the DEMONSTRATION commanded a
close, and reports whether the policy commands one too. These are frames the
model was trained on, so a checkpoint that will not grasp here has genuinely
lost the gripper channel; there is no distribution-shift excuse available.

Costs one server restart per checkpoint and no robot time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(FR3_ROOT / "model-validation"))

import cv2  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

from bracket_eval_preflight import (  # noqa: E402
    build_observation,
    closedness,
    nearest_indices,
    read_jsonl,
    require_server,
)


def grasp_frames(ep_dir: Path, horizon: int, close_threshold: float, max_frames: int):
    """Frames whose ground-truth action chunk contains the demonstration's grasp.

    Mirrors the converter: 60->15 Hz stride, nearest state per camera frame,
    action = dq_d now plus the NEXT sample's gripper command.
    """
    meta = json.loads((ep_dir / "metadata.json").read_text())
    states = read_jsonl(ep_dir / "robot_state.jsonl")
    state_times = np.asarray([r["time_ns"] for r in states], dtype=np.int64)
    events = read_jsonl(ep_dir / "events.jsonl")
    start_ns = int(meta["record_start_time_ns"])
    end_ns = [int(r["time_ns"]) for r in events
              if r.get("event") == "bracket_retreat_open_completed"][0]

    videos = sorted(ep_dir.glob("camera_*_rgb.mp4"))
    ts_rows = [read_jsonl(ep_dir / f"{v.stem.removesuffix('_rgb')}_timestamps.jsonl") for v in videos]
    stride = max(1, round(int(meta.get("fps", 60)) / 15))
    frame_count = min(len(r) for r in ts_rows)
    selected = [i for i in range(0, frame_count, stride)
                if start_ns <= int(ts_rows[0][i]["time_ns"]) <= end_ns]
    if len(selected) <= horizon:
        raise RuntimeError(f"{ep_dir.name}: too few frames")
    ref_times = np.asarray([ts_rows[0][i]["time_ns"] for i in selected], dtype=np.int64)
    state_idx = nearest_indices(state_times, ref_times)

    gt_grip = []
    for k in range(len(selected)):
        nxt = min(int(state_idx[k]) + 1, len(states) - 1)
        gt_grip.append(closedness(states[nxt].get("gripper_command_width")))
    gt_grip = np.asarray(gt_grip)

    usable = len(selected) - horizon
    peaks = np.asarray([gt_grip[i:i + horizon].max() for i in range(usable)])
    closing = np.flatnonzero(peaks >= close_threshold)
    if len(closing) == 0:
        raise RuntimeError(f"{ep_dir.name}: no ground-truth grasp in the bracket window")

    # The onset matters most: the first frames from which the demo commits to a
    # close. Deep inside the closed phase the Hand is already shut, which is a
    # much easier and less informative question.
    picks = closing[:max_frames]
    wanted = {selected[p] for p in picks}

    caps = [cv2.VideoCapture(str(v)) for v in videos]
    images: dict[int, list] = {}
    try:
        for fi in range(max(wanted) + 1):
            frames = [c.read() for c in caps]
            if not all(ok for ok, _ in frames):
                break
            if fi in wanted:
                images[fi] = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in frames]
    finally:
        for c in caps:
            c.release()

    out = []
    for p in picks:
        fi = selected[p]
        if fi not in images:
            continue
        si = int(state_idx[p])
        out.append({
            "exterior": images[fi][0],
            "wrist": images[fi][1],
            "q": np.asarray(states[si]["q"], dtype=np.float64),
            "grip": closedness(states[si].get("gripper_width")),
            "gt_peak": float(gt_grip[p:p + horizon].max()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=FR3_ROOT / "configs/bracket_policy_eval_v1.json")
    ap.add_argument("--raw_dir", type=Path,
                    default=FR3_ROOT / "recordings/full_bracket_bolt_autonomous_raw_v1")
    ap.add_argument("--episodes", type=int, default=4, help="training episodes to probe")
    ap.add_argument("--frames_per_episode", type=int, default=3,
                    help="grasp-onset frames per episode")
    ap.add_argument("--repeats", type=int, default=4,
                    help="policy samples per frame; pi0.5 is stochastic")
    ap.add_argument("--label", default="", help="free text for the report, e.g. the step number")
    ap.add_argument("--training_style_resize", action="store_true",
                    help="reproduce the converter's 640x480->320x240 INTER_AREA step first")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    pol, grip = cfg["policy"], cfg["gripper"]
    horizon = int(pol["action_shape"][0])
    close_threshold = float(grip["close_threshold"])

    eps = sorted(p for p in args.raw_dir.iterdir()
                 if (p / "events.jsonl").exists() and (p / "metadata.json").exists())
    eps = eps[:: max(1, len(eps) // max(args.episodes, 1))][: args.episodes]

    require_server(pol["server_host"], pol["server_port"])
    client = websocket_client_policy.WebsocketClientPolicy(pol["server_host"], pol["server_port"])

    print("=" * 78)
    print(f"GRASP-CHANNEL SCREEN{('  ' + args.label) if args.label else ''}")
    print("=" * 78)
    print(f"server        : {pol['server_host']}:{pol['server_port']}")
    print(f"close thresh  : {close_threshold}")
    print(f"frames        : training frames where the DEMO commands a close")
    print("-" * 78)

    fired = total = 0
    peaks = []
    for ep in eps:
        try:
            frames = grasp_frames(ep, horizon, close_threshold, args.frames_per_episode)
        except Exception as exc:
            print(f"  {ep.name[:44]:46s} skipped: {exc}")
            continue
        for n, fr in enumerate(frames, 1):
            obs = build_observation(fr["exterior"], fr["wrist"], fr["q"], fr["grip"],
                                    pol["prompt"], args.training_style_resize)
            got = []
            for _ in range(args.repeats):
                actions = np.asarray(client.infer(obs)["actions"], dtype=float)
                got.append(float(actions[:, 7].max()))
            hit = sum(g >= close_threshold for g in got)
            fired += hit
            total += len(got)
            peaks.extend(got)
            print(f"  {ep.name[:40]:42s} frame {n}  gt {fr['gt_peak']:.3f}  "
                  f"policy {min(got):.3f}..{max(got):.3f}  fired {hit}/{len(got)}")

    if total == 0:
        raise SystemExit("no probe frames were built; nothing was measured")

    print("-" * 78)
    peaks = np.asarray(peaks)
    print(f"probes           : {total} across {len(eps)} episodes")
    print(f"policy peak      : min {peaks.min():.3f}  median {np.median(peaks):.3f}  max {peaks.max():.3f}")
    print(f"crossed {close_threshold}     : {fired}/{total}  ({100 * fired / total:.0f}%)")
    print()
    if fired == 0 and peaks.max() < 0.05:
        print("VERDICT: gripper channel is DEAD. On frames it was trained on, where the")
        print("         demonstration closes the Hand, this checkpoint predicts open. It")
        print("         will not grasp on the robot. Screen a different checkpoint.")
    elif fired >= 0.8 * total:
        print("VERDICT: gripper channel is HEALTHY. Worth a live trial.")
    else:
        print("VERDICT: gripper channel is WEAK/STOCHASTIC. It may grasp on some rollouts")
        print("         and stall on others; expect run-to-run variation on the robot.")


if __name__ == "__main__":
    main()
