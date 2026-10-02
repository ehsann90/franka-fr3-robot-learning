#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/convert_raw_recordings_to_lerobot.py --help
"""Convert FR3 raw recording episodes into a LeRobotDataset.

This converter expects episode folders produced by realsense_recorder.py:
metadata.json, robot_state.jsonl, actions.jsonl, and camera_*_rgb.mp4 files.
It uses LeRobot's writer API so the output is a normal LeRobot dataset with
Parquet state/action data and MP4 camera observations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


FR3_FULL_STATE_NAMES = (
    [f"joint_{i}.pos" for i in range(7)]
    + [f"joint_{i}.vel" for i in range(7)]
    + ["ee.x", "ee.y", "ee.z"]
    + ["ee.qx", "ee.qy", "ee.qz", "ee.qw"]
    + ["gripper.width"]
)
FR3_FULL_ACTION_NAMES = ["target.ee.x", "target.ee.y", "target.ee.z", "target.ee.qx", "target.ee.qy", "target.ee.qz", "target.ee.qw", "target.gripper.width"]
DROID_JOINT_STATE_NAMES = [f"joint_{i}.pos" for i in range(7)]
DROID_GRIPPER_STATE_NAMES = ["gripper.closedness"]
DROID_ACTION_NAMES = [f"joint_{i}.velocity" for i in range(7)] + ["gripper.closedness"]
DROID_JOINT_POSITION_ACTION_NAMES = [f"joint_{i}.position" for i in range(7)] + ["gripper.closedness"]

BRACKET_TASK_PROMPT = "Place the 3D-printed bracket onto the wall alignment pins."
FULL_BOLT_TASK_PROMPT = (
    "Pick up the 3D-printed bolt, insert it into the mounted bracket, and engage the threads."
)
HYBRID_BOLT_TASK_PROMPT = (
    "Pick up the 3D-printed bolt, insert it into the mounted bracket, and securely grasp its head for threading."
)


class EpisodeView(NamedTuple):
    name: str
    task: str
    start_time_ns: int | None = None
    end_time_ns: int | None = None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_dirs(
    raw_dir: Path,
    include_failed: bool,
    include_source_trajectories: bool = False,
    latest_take_per_episode: bool = False,
    force_include: tuple[str, ...] = (),
) -> list[Path]:
    dirs = []
    unmatched = set(force_include)
    for path in sorted(raw_dir.iterdir()):
        metadata_path = path / "metadata.json"
        if not path.is_dir() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        # Resolve "forced" first: a capture named explicitly by the operator must
        # not be dropped by the eligibility filters below, and its name must be
        # marked seen so the typo check cannot fire on a capture that does exist.
        forced = path.name in force_include
        unmatched.discard(path.name)
        source_only = metadata.get("bc_training_eligible") is False
        if source_only and not include_source_trajectories and not forced:
            continue
        if metadata.get("success", False) or include_failed or forced:
            dirs.append(path)
    if unmatched:
        # A typo here would silently drop the episode the operator meant to add.
        raise RuntimeError(f"--force_include named captures that do not exist: {sorted(unmatched)}")
    if latest_take_per_episode:
        # Directory names begin with a sortable YYYYmmdd_HHMMSS stamp, so the last
        # surviving entry for an episode_index is its newest take.
        newest: dict[object, Path] = {}
        for path in dirs:
            index = json.loads((path / "metadata.json").read_text()).get("episode_index")
            key = index if index is not None else path.name
            if key not in newest or path.name > newest[key].name:
                newest[key] = path
        dirs = sorted(newest.values())
    return dirs


def _unique_event_time(events: list[dict], name: str, ep_dir: Path) -> int:
    matches = [int(row["time_ns"]) for row in events if row.get("event") == name]
    if len(matches) != 1:
        raise RuntimeError(
            f"{ep_dir}: expected exactly one {name!r} event, found {len(matches)}"
        )
    return matches[0]


def _training_motion_end(ep_dir: Path, events: list[dict]) -> int:
    """Return the pre-review endpoint, including one second of terminal context."""

    explicit = [
        int(row["time_ns"])
        for row in events
        if row.get("event") == "training_motion_completed"
    ]
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        raise RuntimeError(
            f"{ep_dir}: expected at most one 'training_motion_completed' event"
        )

    # Episode 9 predates the explicit boundary by one collection run. Its raw
    # achieved release action is authoritative; retain the same one-second
    # terminal context without altering any source file.
    actions = read_jsonl(ep_dir / "actions.jsonl")
    releases = [
        int(row["time_ns"])
        for row in actions
        if row.get("action") == "engaged_bolt_released"
    ]
    if len(releases) != 1:
        raise RuntimeError(
            f"{ep_dir}: no unique terminal training boundary or engaged-bolt release"
        )
    return releases[0] + 1_000_000_000


BRACKET_PHASE_EVENTS = (
    "bracket_approach_open_started",
    "bracket_approach_open_completed",
    "bracket_transport_place_closed_started",
    "bracket_transport_place_closed_completed",
    "bracket_retreat_open_started",
    "bracket_retreat_open_completed",
)
# Measured over the bracket window of every accepted capture: the commanded
# gripper closedness is bimodal at 0.000 (open) and 0.521-0.625 (closed on the
# 38.3 mm bracket). Anything above this floor is unambiguously a real grasp.
MINIMUM_GRASP_CLOSEDNESS = 0.40


def _require_verified_bracket_capture(ep_dir: Path, metadata: dict) -> None:
    """Prove a setup-validation capture is a real bracket demonstration.

    Checks the two things the setup_validation_only flag would otherwise let
    through: that the whole bracket sequence ran, and that the Hand actually
    closed on the bracket during it. A dry run fails the second.
    """
    events = read_jsonl(ep_dir / "events.jsonl")
    present = {row.get("event") for row in events}
    missing = [name for name in BRACKET_PHASE_EVENTS if name not in present]
    if missing:
        raise RuntimeError(f"{ep_dir}: bracket-only capture is missing {missing}")

    record_start = int(metadata["record_start_time_ns"])
    bracket_end = _unique_event_time(events, "bracket_retreat_open_completed", ep_dir)
    gripper_max_width = float(metadata.get("gripper_max_width_m", 0.08))
    peak = 0.0
    for row in read_jsonl(ep_dir / "robot_state.jsonl"):
        if not record_start <= int(row["time_ns"]) <= bracket_end:
            continue
        width = row.get("gripper_command_width")
        if width is None:
            continue
        peak = max(peak, 1.0 - float(width) / gripper_max_width)
    if peak < MINIMUM_GRASP_CLOSEDNESS:
        raise RuntimeError(
            f"{ep_dir}: bracket-only capture never grasped "
            f"(peak commanded closedness {peak:.3f} < {MINIMUM_GRASP_CLOSEDNESS}); "
            f"this is a dry run, not a demonstration")


def derived_episode_views(
    ep_dir: Path,
    metadata: dict,
    view: str,
    default_task: str,
    allow_bracket_only_captures: bool = False,
    force_include: tuple[str, ...] = (),
) -> list[EpisodeView]:
    """Return immutable time views of one accepted synchronized raw capture."""

    if view == "whole":
        return [EpisodeView("whole", default_task)]
    if metadata.get("setup_validation_only") is True:
        # The recorder flags a run that stops after the bracket phase as
        # setup_validation_only, which is right for a dry run but also catches the
        # bracket-only demonstrations captured on 2026-08-18. Those are real
        # demonstrations: complete bracket event sequence, real grasp, operator
        # accepted. Admit them for bracket_only ONLY, and only after verifying
        # both of those facts from the capture itself rather than trusting a flag.
        if view != "bracket_only" or not (allow_bracket_only_captures or ep_dir.name in force_include):
            raise RuntimeError(
                f"{ep_dir}: setup-validation captures cannot produce training views "
                f"(pass --allow-bracket-only-captures to admit verified bracket-only demos "
                f"into the bracket_only view)")
        _require_verified_bracket_capture(ep_dir, metadata)
    if metadata.get("bc_training_eligible") is not True or metadata.get("success") is not True:
        # success=false on a full-assembly capture usually means a LATER stage
        # faulted. The bracket phase can still be a clean demonstration, which is
        # exactly why bracket_only resolves no bolt/threading boundaries. Admitting
        # one is an operator judgement, so it must be named explicitly -- and it is
        # still verified the same way a bracket-only capture is.
        if view != "bracket_only" or ep_dir.name not in force_include:
            raise RuntimeError(
                f"{ep_dir}: derived training views require accepted successful raw data "
                f"(name it in --force_include to admit its bracket phase alone)")
        _require_verified_bracket_capture(ep_dir, metadata)
    if metadata.get("task_name") != "full_bracket_bolt_assembly":
        raise RuntimeError(f"{ep_dir}: {view} requires a full-assembly raw capture")

    events = read_jsonl(ep_dir / "events.jsonl")
    bracket_end = _unique_event_time(events, "bracket_retreat_open_completed", ep_dir)
    record_start = int(metadata["record_start_time_ns"])
    if not record_start < bracket_end:
        raise RuntimeError(f"{ep_dir}: bracket task boundary is not time ordered")

    bracket_view = EpisodeView("bracket", BRACKET_TASK_PROMPT, record_start, bracket_end)

    # bracket_only deliberately resolves no bolt/threading boundaries. That keeps
    # a capture whose bolt or threading phase later faulted still usable for its
    # bracket episode, and it is the only view that trains a pure pick-and-place
    # policy with no contact-rich threading data mixed in.
    if view == "bracket_only":
        return [bracket_view]

    review_end = _unique_event_time(events, "autonomous_bolt_episode_finished", ep_dir)
    bolt_end = _training_motion_end(ep_dir, events)
    if not bracket_end < bolt_end <= review_end:
        raise RuntimeError(f"{ep_dir}: full-policy task boundaries are not time ordered")

    views = [bracket_view]
    if view == "two_task_full_policy":
        views.append(EpisodeView("bolt_full", FULL_BOLT_TASK_PROMPT, bracket_end, bolt_end))
        return views
    if view == "two_task_hybrid":
        thread_ready = _unique_event_time(events, "thread_ready_completed", ep_dir)
        if not bracket_end < thread_ready < bolt_end:
            raise RuntimeError(f"{ep_dir}: hybrid THREAD_READY boundary is not time ordered")
        views.append(
            EpisodeView("bolt_thread_ready", HYBRID_BOLT_TASK_PROMPT, bracket_end, thread_ready)
        )
        return views
    raise ValueError(f"unknown derived view: {view}")


def camera_files(ep_dir: Path) -> list[Path]:
    return sorted(ep_dir.glob("camera_*_rgb.mp4"))


def camera_key(video_path: Path) -> str:
    return "observation.images." + video_path.stem.removesuffix("_rgb").replace("-", "_")


def normalize_quaternion(q: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / norm
    if q[3] < 0:
        q = -q
    return q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(q)
    return np.asarray([-x, -y, -z, w], dtype=np.float64)


def quaternion_delta_to_rotvec(q_current: list[float], q_next: list[float]) -> np.ndarray:
    q_current = normalize_quaternion(q_current)
    q_next = normalize_quaternion(q_next)
    q_delta = normalize_quaternion(quaternion_multiply(q_next, quaternion_inverse(q_current)))
    xyz = q_delta[:3]
    w = float(np.clip(q_delta[3], -1.0, 1.0))
    norm_xyz = float(np.linalg.norm(xyz))
    if norm_xyz < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(norm_xyz, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (xyz / norm_xyz * angle).astype(np.float32)


def gripper_closedness(sample: dict, gripper_max_width: float) -> float:
    width = float(sample.get("gripper_width", gripper_max_width))
    return float(np.clip(1.0 - width / gripper_max_width, 0.0, 1.0))


def gripper_command_closedness(sample: dict, gripper_max_width: float) -> float:
    width = sample.get("gripper_command_width")
    if width is None:
        return gripper_closedness(sample, gripper_max_width)
    return float(np.clip(1.0 - float(width) / gripper_max_width, 0.0, 1.0))


def fr3_full_state_vector(sample: dict) -> np.ndarray:
    return np.asarray(
        sample["q"]
        + sample["dq"]
        + sample["ee_translation"]
        + sample["ee_quaternion"]
        + [float(sample.get("gripper_width", 0.0))],
        dtype=np.float32,
    )


def fr3_full_action_vector(sample: dict) -> np.ndarray:
    return np.asarray(
        sample["ee_translation"]
        + sample["ee_quaternion"]
        + [float(sample.get("gripper_width", 0.0))],
        dtype=np.float32,
    )


def droid_joint_state_vector(sample: dict) -> np.ndarray:
    return np.asarray(sample["q"], dtype=np.float32)


def droid_joint_velocity_action_vector(sample: dict, next_sample: dict, gripper_max_width: float) -> np.ndarray:
    """DROID-compatible action: 7 measured joint velocities and next gripper state.

    Franky's dq_d is libfranka's desired joint velocity at the sampled control
    instant, which is a closer match to the original DROID velocity command than
    measured dq. The gripper channel is the explicit persisted command.
    """
    return np.asarray(
        list(sample.get("dq_d", sample["dq"])) + [gripper_command_closedness(next_sample, gripper_max_width)],
        dtype=np.float32,
    )


def droid_joint_position_action_vector(next_sample: dict, gripper_max_width: float) -> np.ndarray:
    """Absolute achieved-next joint pose and measured achieved-next gripper state.

    Manual Pilot demonstrations do not expose an arm or Franka Hand command.
    The target therefore comes from the robot state synchronized to the next
    dataset frame. This is the same achieved-next joint-position convention as
    the repository's existing pi0.5 joint-position LeRobot datasets. The raw
    recording remains authoritative and records that no gripper command was
    available.
    """

    return np.asarray(
        list(next_sample["q"]) + [gripper_closedness(next_sample, gripper_max_width)],
        dtype=np.float32,
    )


def nearest_indices(sorted_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_times, query_times)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = idx - 1
    choose_left = np.abs(query_times - sorted_times[left]) <= np.abs(sorted_times[idx] - query_times)
    return np.where(choose_left, left, idx)


def read_video_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_features(
    first_ep: Path, schema: str, image_width: int | None, image_height: int | None
) -> tuple[dict, list[tuple[str, int]]]:
    video_paths = camera_files(first_ep)
    if not video_paths:
        raise RuntimeError(f"No camera videos found in {first_ep}")

    if schema in {"droid_joint_velocity", "droid_joint_position"}:
        state_names = DROID_JOINT_STATE_NAMES
        action_names = (
            DROID_ACTION_NAMES if schema == "droid_joint_velocity" else DROID_JOINT_POSITION_ACTION_NAMES
        )
    else:
        state_names = FR3_FULL_STATE_NAMES
        action_names = FR3_FULL_ACTION_NAMES

    if schema in {"droid_joint_velocity", "droid_joint_position"}:
        # These exact names are consumed by OpenPI's LeRobotDROIDDataConfig.
        features = {
            "joint_position": {"dtype": "float32", "shape": (7,), "names": DROID_JOINT_STATE_NAMES},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": DROID_GRIPPER_STATE_NAMES},
            "actions": {"dtype": "float32", "shape": (8,), "names": action_names},
        }
        required_cameras = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    else:
        features = {
            "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": list(state_names)},
            "action": {"dtype": "float32", "shape": (len(action_names),), "names": list(action_names)},
        }
        required_cameras = []

    camera_sources: list[tuple[str, int]] = []
    if schema not in {"droid_joint_velocity", "droid_joint_position"}:
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Could not read first frame from {video_path}")
            height, width = frame.shape[:2]
            if image_width is not None and image_height is not None:
                width, height = image_width, image_height
            key = camera_key(video_path)
            camera_sources.append((key, len(camera_sources)))
            features[key] = {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
    if schema in {"droid_joint_velocity", "droid_joint_position"}:
        if len(video_paths) < 2:
            raise RuntimeError("pi05_droid conversion requires the two RealSense camera videos.")
        # OpenPI's DROID adapter requires three image keys. The second FR3 view
        # fills the wrist slot; exterior_image_2_left duplicates camera 0.
        camera_sources = [
            (required_cameras[0], 0),
            (required_cameras[1], 0),
            (required_cameras[2], 1),
        ]
        for key, source_idx in camera_sources:
            source_path = video_paths[source_idx]
            cap = cv2.VideoCapture(str(source_path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Could not read first frame from {source_path}")
            height, width = frame.shape[:2]
            if image_width is not None and image_height is not None:
                width, height = image_width, image_height
            features[key] = {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
    return features, camera_sources


def convert_episode(
    dataset,
    ep_dir: Path,
    camera_sources: list[tuple[str, int]],
    task: str,
    max_frames: int | None,
    schema: str,
    gripper_max_width: float,
    image_width: int | None,
    image_height: int | None,
    output_fps: int,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> int:
    states = read_jsonl(ep_dir / "robot_state.jsonl")
    if len(states) < 2:
        raise RuntimeError(f"{ep_dir} has too few robot_state samples")
    state_times = np.asarray([row["time_ns"] for row in states], dtype=np.int64)

    video_paths = camera_files(ep_dir)
    if schema not in {"droid_joint_velocity", "droid_joint_position"} and len(video_paths) != len(camera_sources):
        raise RuntimeError(f"{ep_dir} camera count changed: expected {len(camera_sources)}, got {len(video_paths)}")

    timestamp_rows = []
    for video_path in video_paths:
        name = video_path.stem.removesuffix("_rgb")
        rows = read_jsonl(ep_dir / f"{name}_timestamps.jsonl")
        if not rows:
            raise RuntimeError(f"Missing timestamps for {video_path}")
        timestamp_rows.append(rows)

    source_fps = int(json.loads((ep_dir / "metadata.json").read_text()).get("fps", 60))
    frame_stride = (
        max(1, round(source_fps / output_fps))
        if schema in {"droid_joint_velocity", "droid_joint_position"}
        else 1
    )
    frame_count = min(len(rows) for rows in timestamp_rows)
    if frame_count < 2:
        raise RuntimeError(f"{ep_dir} has too few synchronized frames")

    selected_frame_indices = list(range(0, frame_count, frame_stride))
    if start_time_ns is not None:
        selected_frame_indices = [
            idx
            for idx in selected_frame_indices
            if int(timestamp_rows[0][idx]["time_ns"]) >= start_time_ns
        ]
    if end_time_ns is not None:
        selected_frame_indices = [
            idx
            for idx in selected_frame_indices
            if int(timestamp_rows[0][idx]["time_ns"]) <= end_time_ns
        ]
    if max_frames is not None:
        selected_frame_indices = selected_frame_indices[:max_frames]
    if len(selected_frame_indices) < 2:
        raise RuntimeError(f"{ep_dir} has too few frames after resampling")
    ref_times = np.asarray([timestamp_rows[0][idx]["time_ns"] for idx in selected_frame_indices], dtype=np.int64)
    state_indices = nearest_indices(state_times, ref_times)

    # Joint-position labels require the state synchronized to the following
    # selected camera frame. Do not emit a terminal self-target frame.
    saved_frame_indices = (
        selected_frame_indices[:-1] if schema == "droid_joint_position" else selected_frame_indices
    )

    caps = [cv2.VideoCapture(str(path)) for path in video_paths]
    try:
        saved_frames = 0
        selected_set = set(saved_frame_indices)
        for frame_idx in range(frame_count):
            images = [read_video_frame(cap) for cap in caps]
            if any(image is None for image in images):
                break
            if frame_idx not in selected_set:
                continue
            if image_width is not None and image_height is not None:
                images = [cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA) for image in images]
            state_idx = int(state_indices[saved_frames])
            action_idx = min(state_idx + 1, len(states) - 1)
            if schema == "droid_joint_velocity":
                frame = {
                    "task": task,
                    "joint_position": droid_joint_state_vector(states[state_idx]),
                    "gripper_position": np.asarray([gripper_closedness(states[state_idx], gripper_max_width)], dtype=np.float32),
                    "actions": droid_joint_velocity_action_vector(states[state_idx], states[action_idx], gripper_max_width),
                }
            elif schema == "droid_joint_position":
                next_state_idx = int(state_indices[saved_frames + 1])
                frame = {
                    "task": task,
                    "joint_position": droid_joint_state_vector(states[state_idx]),
                    "gripper_position": np.asarray(
                        [gripper_closedness(states[state_idx], gripper_max_width)], dtype=np.float32
                    ),
                    "actions": droid_joint_position_action_vector(
                        states[next_state_idx], gripper_max_width
                    ),
                }
            else:
                state = fr3_full_state_vector(states[state_idx])
                action = fr3_full_action_vector(states[action_idx])
                frame = {"task": task, "observation.state": state, "action": action}
            for key, source_idx in camera_sources:
                frame[key] = images[source_idx]
            dataset.add_frame(frame)
            saved_frames += 1
    finally:
        for cap in caps:
            cap.release()
    if saved_frames == 0:
        raise RuntimeError(
            f"{ep_dir}: OpenCV decoded zero synchronized frames from "
            f"{', '.join(path.name for path in video_paths)}"
        )
    dataset.save_episode()
    return saved_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot repo id, e.g. msbutt1/fr3-pick-place")
    parser.add_argument("--task", type=str, default="Pick up the blue cube and place it in the basket.")
    parser.add_argument(
        "--derived_view",
        choices=["whole", "bracket_only", "two_task_full_policy", "two_task_hybrid"],
        default="whole",
        help=(
            "Derive immutable task episodes from each accepted full-assembly raw capture. "
            "bracket_only emits exactly one bracket pick-and-place episode per capture and "
            "no bolt/threading data at all; the full-policy view contains bracket + complete "
            "bolt/threading episodes; the hybrid view replaces complete threading with a "
            "bolt-to-THREAD_READY episode."
        ),
    )
    parser.add_argument("--fps", type=int, default=15, help="Output dataset FPS. pi05_droid expects 15 Hz.")
    parser.add_argument("--include_failed", action="store_true")
    parser.add_argument(
        "--allow_bracket_only_captures",
        action="store_true",
        help=(
            "Admit captures flagged setup_validation_only into the bracket_only view, after "
            "verifying from the capture itself that the full bracket event sequence ran and "
            "that the Hand actually closed on the bracket. The 2026-08-18 bracket-only "
            "demonstrations carry that flag because the recorder sets it for any run that "
            "stops after the bracket phase; they are real demonstrations. A dry run fails "
            "the grasp check and is still rejected."
        ),
    )
    parser.add_argument(
        "--force_include",
        nargs="+",
        default=[],
        metavar="EPISODE_DIR_NAME",
        help=(
            "Admit these captures by exact directory name even though metadata says "
            "success=false, and emit their bracket phase into the bracket_only view. For a "
            "full-assembly capture whose BOLT or THREADING stage faulted after a clean "
            "bracket placement -- bracket_only resolves no bolt boundaries, so that data is "
            "still good. Each named capture is verified exactly like a bracket-only one "
            "(complete bracket event sequence plus a real grasp); naming a capture cannot "
            "skip that check. A name that matches nothing is an error, not a silent no-op."
        ),
    )
    parser.add_argument(
        "--latest_take_per_episode",
        action="store_true",
        help=(
            "When a capture was retried, keep only the newest directory for each "
            "episode_index. Without this, every successful retake is emitted as its own "
            "episode, silently duplicating that bracket position in the dataset."
        ),
    )
    parser.add_argument(
        "--include_source_trajectories",
        action="store_true",
        help="Include successful Pilot source trajectories explicitly marked bc_training_eligible=false.",
    )
    parser.add_argument("--max_frames_per_episode", type=int, default=None)
    parser.add_argument(
        "--schema",
        choices=["droid_joint_velocity", "droid_joint_position", "fr3_full"],
        default="droid_joint_velocity",
        help=(
            "droid_joint_position stores achieved-next absolute joint targets for "
            "LeRobotDROIDDataConfig(joint_position_actions=True); droid_joint_velocity "
            "preserves the existing velocity pipeline; fr3_full is diagnostic only."
        ),
    )
    parser.add_argument("--gripper_max_width", type=float, default=0.08,
                        help="Meters, used to normalize gripper closedness for DROID-style schema.")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Converted image width. Default preserves 640x480's 4:3 aspect at half size.")
    parser.add_argument("--image_height", type=int, default=240,
                        help="Converted image height. OpenPI performs the final model resize.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    # OpenPI's pinned legacy LeRobot resolves this at import time. Treat
    # --output_dir as the cache root, exactly as the existing VM converters do.
    os.environ["HF_LEROBOT_HOME"] = str(args.output_dir.expanduser().resolve())

    legacy_root = None
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            # This is the API used by the LeRobot revision pinned by OpenPI.
            from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            legacy_root = Path(HF_LEROBOT_HOME)
        except ImportError as exc:
            raise SystemExit(
                "Could not import LeRobot. Run this from the OpenPI environment, "
                "which installs the compatible LeRobot revision."
            ) from exc
    except Exception as exc:
        raise SystemExit(
            f"Could not import LeRobot: {exc}"
        ) from exc

    episodes = episode_dirs(
        args.raw_dir,
        args.include_failed,
        args.include_source_trajectories,
        args.latest_take_per_episode,
        tuple(args.force_include),
    )
    if not episodes:
        raise SystemExit(f"No usable episodes found under {args.raw_dir}")
    create_parameters = inspect.signature(LeRobotDataset.create).parameters
    supports_root = "root" in create_parameters
    # Both the VM's legacy LeRobot and its existing conversion scripts use
    # --output_dir as a cache parent, with the actual dataset at <root>/<repo>.
    destination = args.output_dir / args.repo_id
    if destination.exists():
        if not args.overwrite:
            raise SystemExit(f"{destination} exists. Pass --overwrite to replace it.")
        shutil.rmtree(destination)

    features, camera_sources = build_features(episodes[0], args.schema, args.image_width, args.image_height)
    create_kwargs = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        # DROID/OpenPI uses Panda kinematic conventions. FR3 shares the
        # relevant seven-joint Franka arm interface for this data adapter.
        "robot_type": (
            "panda"
            if args.schema in {"droid_joint_velocity", "droid_joint_position"}
            else "franka_fr3"
        ),
        "features": features,
    }
    if legacy_root is not None:
        create_kwargs.update({
            "root": str(destination),
            # LeRobot 0.1's multiprocess image writer can report completion
            # before its first PNG is visible on a local workstation filesystem.
            # That leaves a sparse episode image directory and video encoding
            # then fails on frame_000000.png. Synchronous writes are slower but
            # deterministic for this one-time conversion.
            "image_writer_threads": 0,
            "image_writer_processes": 0,
        })
    elif supports_root:
        create_kwargs.update({"root": destination, "use_videos": True})
    dataset = LeRobotDataset.create(**create_kwargs)

    converted = 0
    for ep_dir in episodes:
        metadata = json.loads((ep_dir / "metadata.json").read_text())
        views = derived_episode_views(
            ep_dir,
            metadata,
            args.derived_view,
            args.task,
            args.allow_bracket_only_captures,
            tuple(args.force_include),
        )
        for view in views:
            print(f"[convert] {ep_dir.name} -> {view.name}")
            n_frames = convert_episode(
                dataset,
                ep_dir,
                camera_sources,
                view.task,
                args.max_frames_per_episode,
                args.schema,
                args.gripper_max_width,
                args.image_width,
                args.image_height,
                args.fps,
                view.start_time_ns,
                view.end_time_ns,
            )
            print(f"          saved {n_frames} frames; task={view.task!r}")
            converted += 1
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    print(f"[done] converted {converted} episodes to {destination}")


if __name__ == "__main__":
    main()
