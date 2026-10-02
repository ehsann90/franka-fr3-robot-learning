#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/bracket_eval_preflight.py --help
"""Prove the bracket policy evaluation environment is not the thing under test.

Every check here exists so that a failed eval trial can be attributed to the
policy rather than to the rig. Nothing in this script commands robot or Hand
motion; --with-robot only reads state and kinematic limits.

  A  policy server handshake, action shape, finiteness, latency
  B  OBSERVATION-REPLAY FIDELITY -- the decisive check. Recorded training
     frames are rebuilt into live-format observations and sent to the running
     policy; the returned action chunk is compared against the ground-truth
     action recorded at that instant. If the eval observation pipeline differs
     from the training pipeline in colourspace, resize geometry, state layout,
     gripper scaling or prompt, this correlation collapses.
  C  cameras present, correct serials, expected resolution, distinct devices
  D  live robot kinematic limits vs the configured velocity caps and dynamics
     factor, and reachability of the configured start pose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
# JOINT_MIN/JOINT_MAX live in model-running/, and shadow_fr3_policy in
# model-validation/; both are imported lazily further down.
sys.path.insert(0, str(FR3_ROOT / "model-running"))
sys.path.insert(0, str(FR3_ROOT / "model-validation"))

GRIPPER_MAX_WIDTH = 0.08


# --------------------------------------------------------------------------
# shared helpers -- these mirror the converter and the live client exactly
# --------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def closedness(width: float | None) -> float:
    if width is None:
        return 0.0
    return float(np.clip(1.0 - float(width) / GRIPPER_MAX_WIDTH, 0.0, 1.0))


def build_observation(exterior_rgb, wrist_rgb, q, grip_closedness, prompt, training_style: bool):
    """Exactly the live-client observation.

    training_style additionally reproduces the converter's intermediate
    640x480 -> 320x240 INTER_AREA downsample before the shared
    resize_with_pad(224). Both paths are 4:3 so the letterbox geometry is
    identical; only the resampling kernel differs.
    """
    if training_style:
        exterior_rgb = cv2.resize(exterior_rgb, (320, 240), interpolation=cv2.INTER_AREA)
        wrist_rgb = cv2.resize(wrist_rgb, (320, 240), interpolation=cv2.INTER_AREA)
    return {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(exterior_rgb, 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_rgb, 224, 224)
        ),
        "observation/joint_position": np.asarray(q, dtype=np.float32),
        "observation/gripper_position": np.asarray([grip_closedness], dtype=np.float32),
        "prompt": prompt,
    }


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


def nearest_indices(sorted_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left, right = sorted_times[idx - 1], sorted_times[idx]
    return np.where(query - left <= right - query, idx - 1, idx)


def reconstruct_training_frames(ep_dir: Path, samples: int, horizon: int, motion_threshold: float = 0.05):
    """Rebuild (image, state, ground-truth action chunk) exactly as the converter did."""
    meta = json.loads((ep_dir / "metadata.json").read_text())
    states = read_jsonl(ep_dir / "robot_state.jsonl")
    state_times = np.asarray([r["time_ns"] for r in states], dtype=np.int64)
    events = read_jsonl(ep_dir / "events.jsonl")
    start_ns = int(meta["record_start_time_ns"])
    end_ns = [int(r["time_ns"]) for r in events if r.get("event") == "bracket_retreat_open_completed"][0]

    videos = sorted(ep_dir.glob("camera_*_rgb.mp4"))
    ts_rows = [read_jsonl(ep_dir / f"{v.stem.removesuffix('_rgb')}_timestamps.jsonl") for v in videos]
    stride = max(1, round(int(meta.get("fps", 60)) / 15))
    frame_count = min(len(r) for r in ts_rows)

    selected = [i for i in range(0, frame_count, stride)
                if start_ns <= int(ts_rows[0][i]["time_ns"]) <= end_ns]
    ref_times = np.asarray([ts_rows[0][i]["time_ns"] for i in selected], dtype=np.int64)
    state_idx = nearest_indices(state_times, ref_times)

    # ground-truth action per selected frame: dq_d now + gripper command next
    gt = []
    for k in range(len(selected)):
        si = int(state_idx[k])
        ai = min(si + 1, len(states) - 1)
        s = states[si]
        gt.append(np.asarray(list(s.get("dq_d", s["dq"])) +
                             [closedness(states[ai].get("gripper_command_width"))], dtype=np.float64))
    gt = np.asarray(gt)

    # Probe frames that actually contain motion. Roughly half of a bracket
    # episode is near-stationary (stage handoffs, grasp settling); a chunk of
    # all-zero ground truth makes cosine similarity degenerate and would make
    # this check look like noise rather than measuring the pipeline.
    usable = len(selected) - horizon
    if usable < samples:
        raise RuntimeError(f"{ep_dir.name}: only {usable} usable frames for horizon {horizon}")
    chunk_peak = np.asarray([np.abs(gt[i:i + horizon, :7]).max() for i in range(usable)])
    moving = np.flatnonzero(chunk_peak >= motion_threshold)
    if len(moving) < samples:
        moving = np.argsort(chunk_peak)[::-1][:max(samples, 1)]
        moving.sort()
    probes = moving[np.linspace(0, len(moving) - 1, samples).astype(int)]
    wanted = {selected[p] for p in probes}

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
    for p in probes:
        fi = selected[p]
        if fi not in images:
            continue
        si = int(state_idx[p])
        out.append({
            "exterior": images[fi][0],
            "wrist": images[fi][1],
            "q": np.asarray(states[si]["q"], dtype=np.float64),
            "grip": closedness(states[si].get("gripper_width")),
            "gt_chunk": gt[p:p + horizon],
        })
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=FR3_ROOT / "configs/bracket_policy_eval_v1.json")
    ap.add_argument("--raw_dir", type=Path, default=FR3_ROOT / "recordings/full_bracket_bolt_autonomous_raw_v1")
    ap.add_argument("--episodes", type=int, default=3, help="training episodes to replay through the policy")
    ap.add_argument("--samples", type=int, default=6, help="probe frames per episode")
    ap.add_argument("--motion_threshold", type=float, default=0.05,
                    help="rad/s; probe only frames whose ground-truth chunk actually moves")
    ap.add_argument("--repeats", type=int, default=4,
                    help="samples per probe; pi0.5 is stochastic, so its self-agreement "
                         "across samples is the ceiling any comparison can reach")
    ap.add_argument("--with-cameras", action="store_true")
    ap.add_argument("--with-robot", action="store_true")
    ap.add_argument("--robot_ip", default="172.16.0.2")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    pol, grip, mot, guard, start = (cfg["policy"], cfg["gripper"], cfg["motion"],
                                    cfg["workspace_guard"], cfg["start_pose"])
    caps = np.asarray(mot["joint_velocity_caps_rad_s"], dtype=float)
    failures: list[str] = []
    warnings: list[str] = []

    def ok(msg):   print(f"  \033[32mok  \033[0m {msg}")
    def bad(msg):  print(f"  \033[31mFAIL\033[0m {msg}"); failures.append(msg)
    def warn(msg): print(f"  \033[33mwarn\033[0m {msg}"); warnings.append(msg)

    print("=" * 78)
    print("BRACKET POLICY EVAL PREFLIGHT   (no robot or Hand motion is commanded)")
    print("=" * 78)
    print(f"config : {args.config}")
    print(f"server : {pol['server_host']}:{pol['server_port']}")
    print(f"prompt : {pol['prompt']!r}")

    # ---------------- A: policy server ----------------
    print("\n--- A. policy server ---")
    require_server(pol["server_host"], pol["server_port"])
    client = websocket_client_policy.WebsocketClientPolicy(pol["server_host"], pol["server_port"])
    probe = build_observation(np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640, 3), np.uint8),
                              np.zeros(7), 0.0, pol["prompt"], training_style=False)
    lat = []
    try:
        for _ in range(5):
            t0 = time.perf_counter()
            a = np.asarray(client.infer(probe)["actions"], dtype=float)
            lat.append(time.perf_counter() - t0)
    except Exception as exc:
        bad(f"policy server unreachable or erroring: {exc}")
        print("\nCannot continue without the server. Start it on the training machine first.")
        raise SystemExit(1)
    if tuple(a.shape) == tuple(pol["action_shape"]):
        ok(f"action shape {a.shape}")
    else:
        bad(f"action shape {a.shape}, expected {tuple(pol['action_shape'])}")
    ok("actions finite") if np.isfinite(a).all() else bad("actions contain NaN/Inf")
    lat = np.asarray(lat)
    budget = pol["execute_steps"] / pol["control_hz"]
    msg = f"latency mean={lat.mean()*1000:.0f}ms max={lat.max()*1000:.0f}ms (chunk covers {budget*1000:.0f}ms)"
    ok(msg) if lat.max() < budget else warn(msg + "  -> inference slower than the chunk it replans; motion will stutter")

    # ---------------- B: observation-replay fidelity ----------------
    print("\n--- B. observation-replay fidelity  (THE decisive check) ---")
    eps = sorted(p for p in args.raw_dir.iterdir() if (p / "metadata.json").exists())
    if not eps:
        bad(f"no training episodes under {args.raw_dir}")
    else:
        pick = [eps[i] for i in np.linspace(0, len(eps) - 1, min(args.episodes, len(eps))).astype(int)]
        horizon = pol["action_shape"][0]

        def cos(a, b):
            a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
            d = np.linalg.norm(a) * np.linalg.norm(b)
            return float(a @ b / d) if d > 1e-9 else np.nan

        probes = []
        for ep in pick:
            try:
                frames = reconstruct_training_frames(ep, args.samples, horizon, args.motion_threshold)
            except Exception as exc:
                warn(f"{ep.name}: could not reconstruct ({exc})")
                continue
            for fr in frames:
                if fr["gt_chunk"].shape[0] < horizon:
                    continue
                obs_d = build_observation(fr["exterior"], fr["wrist"], fr["q"], fr["grip"],
                                          pol["prompt"], training_style=False)
                reps = np.stack([np.asarray(client.infer(obs_d)["actions"], float)
                                 for _ in range(args.repeats)])
                obs_t = build_observation(fr["exterior"], fr["wrist"], fr["q"], fr["grip"],
                                          pol["prompt"], training_style=True)
                tr = np.asarray(client.infer(obs_t)["actions"], float)
                probes.append({"gt": fr["gt_chunk"], "reps": reps, "train": tr,
                               "peak": float(np.abs(fr["gt_chunk"][:, :7]).max())})

        if not probes:
            bad("no frames could be replayed")
        else:
            n = len(probes)
            print(f"  replayed {n} moving training frames from {len(pick)} episodes, "
                  f"{horizon}-step chunks, {args.repeats} samples each\n")

            # pi0.5 is a stochastic flow-matching policy: it cannot agree with the
            # recording better than it agrees with itself. Measure that ceiling
            # rather than comparing against an arbitrary absolute threshold.
            self_c = [cos(pr["reps"][i][:, :7], pr["reps"][j][:, :7])
                      for pr in probes for i in range(args.repeats) for j in range(i + 1, args.repeats)]
            ceiling = float(np.nanmean(self_c)) if self_c else np.nan
            chance = float(np.nanmean([cos(probes[i]["gt"][:, :7], probes[j]["gt"][:, :7])
                                       for i in range(n) for j in range(n) if i != j]))
            mean_pred = [pr["reps"][:, :, :7].mean(0) for pr in probes]
            agree = float(np.nanmean([cos(m, pr["gt"][:, :7]) for m, pr in zip(mean_pred, probes)]))
            # Per-SAMPLE ratio, not the norm of the sample mean: eval commands one
            # sampled chunk, and averaging stochastic samples shrinks the norm in a
            # way that would understate the magnitude actually sent to the robot.
            ratio = float(np.nanmean([
                np.linalg.norm(pr["reps"][k][:, :7]) / np.linalg.norm(pr["gt"][:, :7])
                for pr in probes for k in range(args.repeats)
                if np.linalg.norm(pr["gt"][:, :7]) > 1e-9]))
            gmae = float(np.nanmean([np.mean(np.abs(pr["reps"][:, :, 7].mean(0) - pr["gt"][:, 7]))
                                     for pr in probes]))
            tr_agree = float(np.nanmean([cos(pr["train"][:, :7], pr["gt"][:, :7]) for pr in probes]))
            fast = [c for c, pr in zip([cos(m, pr["gt"][:, :7]) for m, pr in zip(mean_pred, probes)], probes)
                    if pr["peak"] >= 0.5]
            exec_steps = int(pol["execute_steps"])
            near = float(np.nanmean([cos(m[:exec_steps], pr["gt"][:exec_steps, :7])
                                     for m, pr in zip(mean_pred, probes)]))

            print(f"  chance baseline (unrelated chunks)   : {chance:+.3f}   <- a broken pipeline looks like this")
            print(f"  policy self-consistency (its ceiling): {ceiling:+.3f}   <- stochastic sampling limit")
            print(f"  agreement with recorded ground truth : {agree:+.3f}")
            print(f"  ... over the {exec_steps} steps actually executed : {near:+.3f}")
            if fast:
                print(f"  ... on fast segments (>0.5 rad/s)    : {np.nanmean(fast):+.3f}  (n={len(fast)})")
            print(f"  magnitude ratio  ‖pred‖/‖gt‖ (per sample): {ratio:.3f}")
            print(f"  gripper channel MAE                  : {gmae:.4f}")
            frac = agree / ceiling if ceiling and np.isfinite(ceiling) and ceiling > 1e-6 else np.nan
            print(f"  fraction of achievable ceiling       : {frac:.2f}\n")

            if not np.isfinite(agree) or agree < max(0.15, 5 * abs(chance)):
                bad(f"policy output is not meaningfully related to the recorded actions "
                    f"(agreement {agree:.3f} vs chance {chance:.3f}). Observation pipeline, served "
                    f"checkpoint, or prompt is wrong.")
            elif np.isfinite(frac) and frac >= 0.70:
                ok(f"agreement is {frac:.0%} of the policy's own sampling ceiling and {agree/max(abs(chance),1e-3):.0f}x "
                   f"chance -> observation pipeline matches training")
            else:
                warn(f"agreement {agree:.3f} is only {frac:.0%} of the policy's ceiling ({ceiling:.3f}); "
                     "confirm the served checkpoint and prompt")
            if 0.5 <= ratio <= 2.0:
                ok(f"action magnitude matches recorded scale ({ratio:.3f}) -> units and normalization correct")
            else:
                bad(f"predicted magnitude is {ratio:.2f}x recorded -- normalization or checkpoint mismatch")
            if gmae <= 0.10:
                ok(f"gripper channel reproduces the recorded command (MAE {gmae:.4f})")
            else:
                bad(f"gripper channel MAE {gmae:.4f} -- gripper scaling mismatch")
            if tr_agree - agree > 0.05:
                warn(f"training-style downsample agrees better ({tr_agree:.3f} vs {agree:.3f}); "
                     "pass --match_training_resize during eval")
            else:
                ok(f"resampling path is not material (direct {agree:.3f} vs training-style {tr_agree:.3f})")

    # ---------------- C: cameras ----------------
    if args.with_cameras:
        print("\n--- C. cameras ---")
        from shadow_fr3_policy import read_rgb, start_camera  # noqa: PLC0415
        cams = cfg["cameras"]
        pipes = {}
        try:
            for role, serial in (("exterior", cams["exterior_serial"]), ("wrist", cams["wrist_serial"])):
                try:
                    pipes[role] = start_camera(serial, cams["width"], cams["height"], cams["fps"])
                    ok(f"{role} camera {serial} opened")
                except Exception as exc:
                    bad(f"{role} camera {serial} failed to open: {exc}")
            imgs = {}
            for role, pipe in pipes.items():
                for _ in range(10):
                    read_rgb(pipe)
                img = read_rgb(pipe)
                imgs[role] = img
                if img.shape[:2] == (cams["height"], cams["width"]):
                    ok(f"{role} frame {img.shape}")
                else:
                    bad(f"{role} frame {img.shape}, expected {(cams['height'], cams['width'], 3)}")
                if float(img.std()) < 5.0:
                    bad(f"{role} image is nearly uniform (std={img.std():.1f}) -- lens cap or dead feed?")
            if len(imgs) == 2:
                diff = float(np.mean(np.abs(imgs["exterior"].astype(float) - imgs["wrist"].astype(float))))
                ok(f"exterior and wrist views differ (mean abs diff {diff:.1f})") if diff > 5.0 else \
                    bad(f"exterior and wrist frames nearly identical (diff {diff:.1f}) -- serials likely swapped/duplicated")
        finally:
            for p in pipes.values():
                try:
                    p.stop()
                except Exception:
                    pass

    # ---------------- D: robot limits, dynamics, start pose ----------------
    if args.with_robot:
        print("\n--- D. robot limits / dynamics / start pose ---")
        from franky import Frame, Robot  # noqa: PLC0415
        robot = Robot(args.robot_ip)
        vlim = np.asarray(robot.joint_velocity_limit.max, dtype=float) \
            if hasattr(robot.joint_velocity_limit, "max") else np.asarray(robot.joint_velocity_limit, dtype=float)
        print(f"  live joint_velocity_limit: {np.array2string(vlim, precision=3)}")
        rdf = float(mot["relative_dynamics_factor"])
        effective = rdf * vlim
        print(f"  effective @ rdf={rdf}:      {np.array2string(effective, precision=3)}")
        print(f"  configured caps:           {np.array2string(caps, precision=3)}")
        if np.all(effective >= caps):
            ok(f"relative_dynamics_factor {rdf} permits the configured caps on every joint")
        else:
            need = float(np.max(caps / vlim))
            bad(f"rdf {rdf} THROTTLES the policy: joints {np.where(effective < caps)[0] + 1} are limited."
                f" Need relative_dynamics_factor >= {need:.3f}, or lower the caps.")
        state = robot.state
        q_now = np.asarray(state.q, dtype=float)
        q_start = np.asarray(start["q"], dtype=float)
        from commission_fr3_policy_motion import JOINT_MAX, JOINT_MIN  # noqa: PLC0415
        margin = float(mot["joint_margin_rad"])
        if np.all(q_start > JOINT_MIN + margin) and np.all(q_start < JOINT_MAX - margin):
            ok("configured start pose is inside guarded joint limits")
        else:
            bad("configured start pose violates guarded joint limits")
        lo = np.asarray(guard["min_xyz"], float); hi = np.asarray(guard["max_xyz"], float)
        tcp = np.asarray(state.O_T_EE.matrix, float)
        j7 = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, float)
        j7_to_tcp = np.linalg.solve(j7, tcp)
        xyz_start = (np.asarray(robot.model.pose(Frame.Joint7, q_start, state.F_T_EE, state.EE_T_K).matrix,
                                float) @ j7_to_tcp)[:3, 3]
        print(f"  start pose TCP: {np.array2string(xyz_start, precision=4)}")
        if np.all(xyz_start >= lo) and np.all(xyz_start <= hi):
            ok("start pose TCP is inside the guarded workspace")
        else:
            bad(f"start pose TCP outside guarded workspace {lo}..{hi}")
        d = float(np.max(np.abs(q_now - q_start)))
        print(f"  current pose is {d:.3f} rad from the start pose (per-joint max)")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    if failures:
        print(f"PREFLIGHT FAILED -- {len(failures)} blocking issue(s):")
        for f in failures:
            print(f"  - {f}")
        print("\nDo NOT run eval trials. A failure here would be misread as policy error.")
        raise SystemExit(1)
    if warnings:
        print(f"PREFLIGHT PASSED with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("PREFLIGHT PASSED -- no issues.")
    print("The evaluation environment is verified. Remaining failures are attributable to the policy.")
    print("=" * 78)


if __name__ == "__main__":
    main()
