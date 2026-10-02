"""Build the compliant-axis Cartesian intent used by hybrid thread replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from expert_demo.piloted_thread_replay import PilotedThreadDrive


@dataclass(frozen=True)
class HybridThreadPlan:
    time_s: np.ndarray
    local_lateral_m: np.ndarray
    relative_quaternion_xyzw: np.ndarray
    source_q7_rad: np.ndarray
    source_trim_start_s: float


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    # Eigen-compatible branch-stable matrix-to-quaternion conversion.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x, y, z = 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x, y, z = (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x, y, z = (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def build_hybrid_thread_plan(
    drive: PilotedThreadDrive,
    *,
    playback_time_scale: float,
    trim_stationary_prefix: bool = False,
    stationary_lead_in_s: float = 0.25,
    replay_lateral_translation: bool = True,
    replay_yaw_only: bool = False,
) -> HybridThreadPlan:
    if not np.isfinite(playback_time_scale) or playback_time_scale <= 0.0:
        raise ValueError("playback time scale must be finite and positive")
    if not np.isfinite(stationary_lead_in_s) or stationary_lead_in_s < 0.0:
        raise ValueError("stationary lead-in must be finite and nonnegative")
    poses = np.asarray(drive.pose_targets, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("drive pose targets must be Nx4x4")
    start_rotation = poses[0, :3, :3]
    start_xyz = poses[0, :3, 3]
    local = np.asarray(
        [start_rotation.T @ (pose[:3, 3] - start_xyz) for pose in poses]
    )
    local[:, 2] = 0.0  # critical: never position-replay demonstrated screw-axis descent
    if not replay_lateral_translation:
        local[:, :2] = 0.0
    relative_rotations = np.asarray(
        [start_rotation.T @ pose[:3, :3] for pose in poses]
    )
    if replay_yaw_only:
        yaw = np.unwrap(
            np.arctan2(relative_rotations[:, 1, 0], relative_rotations[:, 0, 0])
        )
        relative_rotations = np.asarray(
            [
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
                for angle in yaw
            ],
            dtype=np.float64,
        )
    quaternions = np.asarray(
        [rotation_to_quaternion_xyzw(rotation) for rotation in relative_rotations]
    )
    # Enforce quaternion hemisphere continuity for interpolation.
    for index in range(1, len(quaternions)):
        if float(quaternions[index - 1] @ quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    time_s = np.linspace(
        0.0, drive.source_duration_s * playback_time_scale, len(poses)
    )
    source_trim_start_s = 0.0
    if trim_stationary_prefix and len(poses) > 2:
        q7_motion = np.abs(np.asarray(drive.q_targets[:, 6]) - drive.q_targets[0, 6]) > 0.001
        lateral_motion = np.linalg.norm(local, axis=1) > 0.0001
        orientation_angle = 2.0 * np.arccos(
            np.clip(np.abs(quaternions[:, 3]), 0.0, 1.0)
        )
        orientation_motion = orientation_angle > np.deg2rad(0.1)
        moving = np.flatnonzero(q7_motion | lateral_motion | orientation_motion)
        if len(moving):
            first_moving = int(moving[0])
            source_trim_start_s = float(time_s[first_moving])
            # Preserve the exact recorded start as a short stationary boundary,
            # then interpolate into the first source-motion point. This removes
            # operator prompt dwell without creating a discontinuous target.
            selection = np.r_[0, np.arange(first_moving, len(poses))]
            local = local[selection]
            quaternions = quaternions[selection]
            source_q7 = np.asarray(drive.q_targets[:, 6], dtype=np.float64)[selection]
            tail_time = time_s[first_moving:] - time_s[first_moving] + stationary_lead_in_s
            time_s = np.r_[0.0, tail_time]
        else:
            source_q7 = np.asarray(drive.q_targets[:, 6], dtype=np.float64)
    else:
        source_q7 = np.asarray(drive.q_targets[:, 6], dtype=np.float64)
    return HybridThreadPlan(
        time_s=time_s,
        local_lateral_m=local,
        relative_quaternion_xyzw=quaternions,
        source_q7_rad=source_q7,
        source_trim_start_s=source_trim_start_s,
    )
