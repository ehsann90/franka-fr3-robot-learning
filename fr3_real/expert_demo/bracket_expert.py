"""Discrete Stage-1 bracket expert planning with no hardware side effects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .episode_spec import EpisodeSpec
from .safety import SafetyError, execution_readiness_blockers, workspace_from_config
from .transforms import make_transform, validate_transform


BRACKET_CONFIG_SCHEMA_VERSION = 1


class BracketConfigError(ValueError):
    """Raised when the Stage-1 bracket configuration is inconsistent."""


class BracketState(str, Enum):
    IDLE = "IDLE"
    PREGRASP = "PREGRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSFER = "TRANSFER"
    PREPLACE = "PREPLACE"
    ALIGN = "ALIGN"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    ABORT = "ABORT"


@dataclass(frozen=True)
class BracketStep:
    state: BracketState
    operation: str
    calibration_name: str | None = None
    target_T_base_ee: np.ndarray | None = None
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.calibration_name is None or self.target_T_base_ee is not None


@dataclass(frozen=True)
class BracketPlan:
    spec: EpisodeSpec
    steps: tuple[BracketStep, ...]
    missing_calibrations: tuple[str, ...]
    safety_blockers: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return not self.missing_calibrations and not self.safety_blockers


def load_bracket_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BracketConfigError(f"cannot read bracket config {config_path}: {error}") from error
    if config.get("schema_version") != BRACKET_CONFIG_SCHEMA_VERSION:
        raise BracketConfigError(
            f"unsupported bracket config schema {config.get('schema_version')!r}; "
            f"expected {BRACKET_CONFIG_SCHEMA_VERSION}"
        )
    if config.get("task") != "bracket":
        raise BracketConfigError("bracket config task must be 'bracket'")
    for style in ("HIGH", "CENTER", "LOW"):
        if style not in config.get("grasp_styles", {}):
            raise BracketConfigError(f"bracket config is missing grasp style {style}")
    return config


def required_pose_names(grasp_style: str) -> tuple[str, ...]:
    style = grasp_style.upper()
    if style not in {"HIGH", "CENTER", "LOW"}:
        raise BracketConfigError(f"unknown Stage-1 grasp style {style!r}")
    return (
        f"BRACKET_{style}_PREGRASP",
        f"BRACKET_{style}_GRASP",
        f"BRACKET_{style}_LIFT",
        f"BRACKET_{style}_TRANSFER_SAFE",
        f"WALL_PREPLACE_{style}",
        f"WALL_PLACE_{style}",
        f"WALL_RETREAT_{style}",
    )


def all_stage1_pose_names() -> tuple[str, ...]:
    return tuple(name for style in ("HIGH", "CENTER", "LOW") for name in required_pose_names(style))


def _resolved_target(
    poses: dict[str, dict[str, Any]],
    name: str,
    offset_xyz_m: tuple[float, float, float] | None = None,
) -> np.ndarray | None:
    snapshot = poses.get(name)
    if snapshot is None:
        return None
    target = validate_transform(snapshot["T_base_ee"], name=name)
    if offset_xyz_m is not None:
        target = validate_transform(make_transform(offset_xyz_m) @ target, name=f"{name}_offset")
    return target


def build_bracket_plan(
    spec: EpisodeSpec,
    config: dict[str, Any],
    calibration_poses: dict[str, dict[str, Any]],
) -> BracketPlan:
    """Resolve one explicit state-machine plan without commanding hardware."""

    style = spec.bracket_grasp_style
    slot_config = config["slots"].get(spec.bracket_slot)
    if not isinstance(slot_config, dict) or not slot_config.get("enabled"):
        raise BracketConfigError(f"slot {spec.bracket_slot!r} is not enabled")
    slot_offset = slot_config.get("offset_xyz_m")
    if slot_offset is None:
        raise BracketConfigError(f"slot {spec.bracket_slot!r} has no calibrated offset")
    pickup_offset = tuple(
        float(slot_offset[index]) + spec.approach_offset_xyz_m[index] for index in range(3)
    )

    names = required_pose_names(style)
    pregrasp, grasp, lift, transfer_safe, wall_preplace, wall_place, wall_retreat = names
    steps = (
        BracketStep(BracketState.IDLE, "await_operator_confirmation", detail=f"place bracket in {spec.bracket_slot}"),
        BracketStep(BracketState.PREGRASP, "move_pose", pregrasp, _resolved_target(calibration_poses, pregrasp, pickup_offset)),
        BracketStep(BracketState.GRASP, "move_pose_slow", grasp, _resolved_target(calibration_poses, grasp, pickup_offset)),
        BracketStep(BracketState.GRASP, "close_gripper_validate", detail=f"grasp style {style}"),
        BracketStep(BracketState.LIFT, "move_pose", lift, _resolved_target(calibration_poses, lift, pickup_offset)),
        BracketStep(
            BracketState.TRANSFER,
            "move_pose",
            transfer_safe,
            _resolved_target(calibration_poses, transfer_safe),
            detail="manually taught collision-free transfer waypoint",
        ),
        BracketStep(BracketState.PREPLACE, "move_pose", wall_preplace, _resolved_target(calibration_poses, wall_preplace)),
        BracketStep(BracketState.PREPLACE, "settle_and_validate_grasp", detail="hold at safe wall preplace"),
        BracketStep(BracketState.ALIGN, "move_pose_slow", wall_place, _resolved_target(calibration_poses, wall_place)),
        BracketStep(BracketState.PLACE, "settle", detail="bracket seated on alignment pins"),
        BracketStep(BracketState.RELEASE, "open_gripper"),
        BracketStep(BracketState.RETREAT, "move_pose", wall_retreat, _resolved_target(calibration_poses, wall_retreat)),
        BracketStep(BracketState.SUCCESS, "mark_success"),
    )

    missing = tuple(sorted({step.calibration_name for step in steps if not step.resolved and step.calibration_name}))
    blockers = execution_readiness_blockers(config)
    workspace = workspace_from_config(config)
    if workspace is not None:
        for step in steps:
            if step.target_T_base_ee is None:
                continue
            try:
                workspace.validate_pose(step.target_T_base_ee, label=f"{step.state.value}:{step.calibration_name}")
            except SafetyError as error:
                blockers.append(str(error))
    blockers.extend(f"missing calibration pose {name}" for name in missing)
    return BracketPlan(spec, steps, missing, tuple(blockers))
