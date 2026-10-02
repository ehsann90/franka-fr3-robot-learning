"""Versioned calibration storage for taught FR3 expert poses."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import validate_transform


SCHEMA_VERSION = 1
TRANSFORM_CONVENTION = "T_A_B is pose of frame B expressed in frame A; p_A = T_A_B @ p_B"
POSE_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,79}$")


class CalibrationError(ValueError):
    """Raised for invalid, incompatible, or corrupt calibration data."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_pose_name(name: str) -> str:
    normalized = name.strip().upper().replace("-", "_").replace(" ", "_")
    if not POSE_NAME_PATTERN.fullmatch(normalized):
        raise CalibrationError(
            "pose name must start with A-Z and contain only A-Z, 0-9, and underscore (maximum 80 characters)"
        )
    return normalized


def _hash_payload(document: dict[str, Any]) -> str:
    payload = deepcopy(document)
    payload.pop("content_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    required = {
        "snapshot_kind",
        "timestamp_utc",
        "host_time_ns",
        "q",
        "dq",
        "T_base_ee",
        "translation_xyz_m",
        "orientation",
        "gripper",
        "notes",
    }
    missing = sorted(required - snapshot.keys())
    if missing:
        raise CalibrationError(f"snapshot is missing fields: {missing}")

    validated = deepcopy(snapshot)
    validated["T_base_ee"] = validate_transform(snapshot["T_base_ee"], name="T_base_ee").tolist()
    for field_name, size in (("q", 7), ("dq", 7), ("translation_xyz_m", 3)):
        array = np.asarray(snapshot[field_name], dtype=np.float64).reshape(-1)
        if array.size != size or not np.isfinite(array).all():
            raise CalibrationError(f"{field_name} must contain {size} finite values")
        validated[field_name] = array.tolist()
    expected_translation = np.asarray(validated["T_base_ee"], dtype=np.float64)[:3, 3]
    if not np.allclose(validated["translation_xyz_m"], expected_translation, atol=1e-9, rtol=0.0):
        raise CalibrationError("translation_xyz_m does not match T_base_ee")
    quaternion = np.asarray(snapshot["orientation"].get("quaternion_xyzw"), dtype=np.float64).reshape(-1)
    if quaternion.size != 4 or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1.0):
        raise CalibrationError("orientation.quaternion_xyzw must be a normalized finite quaternion")
    return validated


class CalibrationStore:
    def __init__(self, path: str | Path, robot_ip: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.robot_ip = robot_ip
        self.document = self._load_or_create()

    def _new_document(self) -> dict[str, Any]:
        now = _utc_now()
        document = {
            "schema_version": SCHEMA_VERSION,
            "calibration_id": str(uuid.uuid4()),
            "robot_ip": self.robot_ip,
            "transform_convention": TRANSFORM_CONVENTION,
            "created_at_utc": now,
            "updated_at_utc": now,
            "poses": {},
        }
        document["content_sha256"] = _hash_payload(document)
        return document

    def _load_or_create(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new_document()
        try:
            document = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"cannot read calibration {self.path}: {error}") from error
        if document.get("schema_version") != SCHEMA_VERSION:
            raise CalibrationError(
                f"unsupported calibration schema {document.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        if document.get("robot_ip") != self.robot_ip:
            raise CalibrationError(
                f"calibration robot_ip={document.get('robot_ip')!r} does not match requested {self.robot_ip!r}"
            )
        expected_hash = _hash_payload(document)
        if document.get("content_sha256") != expected_hash:
            raise CalibrationError("calibration content hash mismatch; file may be truncated or edited incorrectly")
        if not isinstance(document.get("poses"), dict):
            raise CalibrationError("calibration poses must be an object")
        for name, snapshot in document["poses"].items():
            if normalize_pose_name(name) != name:
                raise CalibrationError(f"calibration contains non-canonical pose name {name!r}")
            validate_snapshot(snapshot)
        return document

    @property
    def calibration_hash(self) -> str:
        return str(self.document["content_sha256"])

    @property
    def poses(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.document["poses"])

    def list_pose_names(self) -> list[str]:
        return sorted(self.document["poses"])

    def add(self, name: str, snapshot: dict[str, Any]) -> str:
        pose_name = normalize_pose_name(name)
        if pose_name in self.document["poses"]:
            raise CalibrationError(f"pose {pose_name!r} already exists; use the explicit overwrite operation")
        self.document["poses"][pose_name] = validate_snapshot(snapshot)
        self._write()
        return pose_name

    def overwrite(self, name: str, snapshot: dict[str, Any]) -> Path:
        pose_name = normalize_pose_name(name)
        if pose_name not in self.document["poses"]:
            raise CalibrationError(f"pose {pose_name!r} does not exist")
        backup = self._backup()
        self.document["poses"][pose_name] = validate_snapshot(snapshot)
        self._write()
        return backup

    def _backup(self) -> Path:
        if not self.path.exists():
            raise CalibrationError("cannot back up a calibration that has not yet been written")
        backup_dir = self.path.parent / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup_path = backup_dir / f"{self.path.stem}.{timestamp}{self.path.suffix}.bak"
        shutil.copy2(self.path, backup_path)
        return backup_path

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.document["updated_at_utc"] = _utc_now()
        self.document["content_sha256"] = _hash_payload(self.document)
        contents = json.dumps(self.document, indent=2, sort_keys=False) + "\n"
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(self.path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
