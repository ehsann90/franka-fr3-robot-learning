"""Read-only, non-real-time capture of kinesthetically piloted FR3 trajectories."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import validate_transform


TRAJECTORY_SCHEMA_VERSION = 1


def _vector(value: Any, size: int, name: str) -> list[float]:
    if hasattr(value, "matrix"):
        value = value.matrix
    elif hasattr(value, "linear") and hasattr(value, "angular"):
        value = np.concatenate(
            [np.asarray(value.linear, dtype=np.float64), np.asarray(value.angular, dtype=np.float64)]
        )
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array.tolist()


def _robot_time_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to_sec"):
        value = value.to_sec()
    result = float(value)
    return result if np.isfinite(result) else None


def robot_state_record(state: Any, latest_gripper: dict[str, Any] | None) -> dict[str, Any]:
    pose_value = state.O_T_EE.matrix if hasattr(state.O_T_EE, "matrix") else state.O_T_EE
    T_base_ee = validate_transform(pose_value, name="state.O_T_EE")
    record: dict[str, Any] = {
        "host_time_ns": time.time_ns(),
        "monotonic_time_ns": time.perf_counter_ns(),
        "robot_time_s": _robot_time_seconds(getattr(state, "time", None)),
        "q": _vector(state.q, 7, "q"),
        "dq": _vector(state.dq, 7, "dq"),
        "T_base_ee": T_base_ee.tolist(),
        "translation_xyz_m": T_base_ee[:3, 3].tolist(),
    }
    for name in ("F_T_EE", "EE_T_K"):
        if hasattr(state, name) and getattr(state, name) is not None:
            value = getattr(state, name)
            matrix = value.matrix if hasattr(value, "matrix") else value
            record[name] = validate_transform(matrix, name=f"state.{name}").tolist()
    for name, size in (
        ("q_d", 7),
        ("dq_d", 7),
        ("ddq_d", 7),
        ("tau_J", 7),
        ("tau_J_d", 7),
        ("tau_ext_hat_filtered", 7),
        ("O_F_ext_hat_K", 6),
        ("K_F_ext_hat_K", 6),
    ):
        if hasattr(state, name) and getattr(state, name) is not None:
            record[name] = _vector(getattr(state, name), size, name)
    record["gripper"] = dict(latest_gripper) if latest_gripper is not None else {"available": False}
    return record


def gripper_state_record(state: Any) -> dict[str, Any]:
    return {
        "available": True,
        "host_time_ns": time.time_ns(),
        "monotonic_time_ns": time.perf_counter_ns(),
        "robot_time_s": _robot_time_seconds(getattr(state, "time", None)),
        "width_m": float(state.width),
        "max_width_m": float(state.max_width),
        "is_grasped": bool(state.is_grasped),
        "temperature_c": int(state.temperature),
    }


class PilotedTrajectoryRecorder:
    """Samples measured state in background threads and never commands hardware."""

    def __init__(self, robot: Any, gripper: Any | None, robot_hz: float, gripper_hz: float) -> None:
        if robot_hz <= 0 or gripper_hz <= 0:
            raise ValueError("sampling rates must be positive")
        self.robot = robot
        self.gripper = gripper
        self.robot_hz = float(robot_hz)
        self.gripper_hz = float(gripper_hz)
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self._latest_gripper: dict[str, Any] | None = None
        self._gripper_lock = threading.Lock()
        self._stop = threading.Event()
        self._robot_thread: threading.Thread | None = None
        self._gripper_thread: threading.Thread | None = None
        self.started_host_time_ns: int | None = None
        self.stopped_host_time_ns: int | None = None

    def preflight(self) -> dict[str, Any]:
        if self.gripper is None:
            return robot_state_record(self.robot.state, None)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                latest_gripper = gripper_state_record(self.gripper.state)
                break
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.1)
        else:
            raise RuntimeError(f"gripper read-only preflight failed after 3 attempts: {last_error}") from last_error
        with self._gripper_lock:
            self._latest_gripper = latest_gripper
        return robot_state_record(self.robot.state, latest_gripper)

    def start(self, *, start_gripper_thread: bool = True) -> None:
        if self._robot_thread is not None:
            raise RuntimeError("trajectory recorder has already been started")
        self.samples = []
        self.events = []
        self.errors = []
        self._stop.clear()
        self.started_host_time_ns = time.time_ns()
        self._robot_thread = threading.Thread(target=self._robot_loop, daemon=True)
        self._robot_thread.start()
        if start_gripper_thread and self.gripper is not None:
            self._gripper_thread = threading.Thread(target=self._gripper_loop, daemon=True)
            self._gripper_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._robot_thread, self._gripper_thread):
            if thread is not None:
                thread.join(timeout=3.0)
        self.stopped_host_time_ns = time.time_ns()

    def _wait_period(self, deadline: float, period: float) -> float:
        deadline += period
        delay = deadline - time.perf_counter()
        if delay > 0:
            self._stop.wait(delay)
            return deadline
        return time.perf_counter()

    def _robot_loop(self) -> None:
        period = 1.0 / self.robot_hz
        deadline = time.perf_counter()
        while not self._stop.is_set():
            try:
                with self._gripper_lock:
                    latest_gripper = (
                        None if self._latest_gripper is None else dict(self._latest_gripper)
                    )
                self.samples.append(robot_state_record(self.robot.state, latest_gripper))
            except Exception as error:
                self.errors.append(
                    {
                        "host_time_ns": time.time_ns(),
                        "source": "robot_state",
                        "error": repr(error),
                    }
                )
            deadline = self._wait_period(deadline, period)

    def _gripper_loop(self) -> None:
        period = 1.0 / self.gripper_hz
        deadline = time.perf_counter()
        while not self._stop.is_set():
            try:
                sample = gripper_state_record(self.gripper.state)
                with self._gripper_lock:
                    self._latest_gripper = sample
            except Exception as error:
                self.errors.append(
                    {
                        "host_time_ns": time.time_ns(),
                        "source": "gripper_state",
                        "error": repr(error),
                    }
                )
            deadline = self._wait_period(deadline, period)

    def poll_gripper_once(self) -> bool:
        """Poll on the caller's thread for bindings whose socket is thread-affine."""

        if self.gripper is None:
            raise RuntimeError("no gripper state client is attached")
        try:
            sample = gripper_state_record(self.gripper.state)
            with self._gripper_lock:
                self._latest_gripper = sample
            return True
        except Exception as error:
            self.errors.append(
                {
                    "host_time_ns": time.time_ns(),
                    "source": "gripper_state",
                    "error": repr(error),
                }
            )
            return False

    def latest_gripper_snapshot(self) -> dict[str, Any] | None:
        with self._gripper_lock:
            return None if self._latest_gripper is None else dict(self._latest_gripper)

    def add_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "host_time_ns": time.time_ns(),
                "monotonic_time_ns": time.perf_counter_ns(),
                "event": event,
                "payload": payload or {},
            }
        )

    def write(self, trajectory_dir: Path, metadata: dict[str, Any], accepted: bool) -> Path:
        if self.started_host_time_ns is None or self.stopped_host_time_ns is None:
            raise RuntimeError("trajectory must be started and stopped before writing")
        trajectory_dir = trajectory_dir.expanduser().resolve()
        trajectory_dir.mkdir(parents=True, exist_ok=False)
        duration_s = (self.stopped_host_time_ns - self.started_host_time_ns) / 1e9
        document = {
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            **metadata,
            "accepted_by_operator": bool(accepted),
            "record_start_time_ns": self.started_host_time_ns,
            "record_stop_time_ns": self.stopped_host_time_ns,
            "duration_s": duration_s,
            "robot_sample_hz_requested": self.robot_hz,
            "gripper_sample_hz_requested": self.gripper_hz,
            "robot_sample_count": len(self.samples),
            "sample_error_count": len(self.errors),
            "event_count": len(self.events),
            "replay_commissioned": False,
        }
        (trajectory_dir / "metadata.json").write_text(json.dumps(document, indent=2) + "\n")
        with (trajectory_dir / "trajectory.jsonl").open("w") as stream:
            for sample in self.samples:
                stream.write(json.dumps(sample) + "\n")
        with (trajectory_dir / "sample_errors.jsonl").open("w") as stream:
            for error in self.errors:
                stream.write(json.dumps(error) + "\n")
        with (trajectory_dir / "events.jsonl").open("w") as stream:
            for event in self.events:
                stream.write(json.dumps(event) + "\n")

        if self.samples:
            arrays: dict[str, np.ndarray] = {}
            for name in (
                "host_time_ns",
                "monotonic_time_ns",
                "q",
                "dq",
                "T_base_ee",
                "F_T_EE",
                "EE_T_K",
                "translation_xyz_m",
                "q_d",
                "dq_d",
                "ddq_d",
                "tau_J",
                "tau_J_d",
                "tau_ext_hat_filtered",
                "O_F_ext_hat_K",
                "K_F_ext_hat_K",
            ):
                if all(name in sample for sample in self.samples):
                    arrays[name] = np.asarray([sample[name] for sample in self.samples])
            if all(sample["gripper"].get("available") for sample in self.samples):
                arrays["gripper_width_m"] = np.asarray(
                    [sample["gripper"]["width_m"] for sample in self.samples]
                )
                arrays["gripper_is_grasped"] = np.asarray(
                    [sample["gripper"]["is_grasped"] for sample in self.samples], dtype=bool
                )
                arrays["gripper_state_host_time_ns"] = np.asarray(
                    [sample["gripper"]["host_time_ns"] for sample in self.samples], dtype=np.int64
                )
            np.savez_compressed(trajectory_dir / "trajectory_arrays.npz", **arrays)
        return trajectory_dir
