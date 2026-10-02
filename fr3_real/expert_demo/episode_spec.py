"""Reproducible episode specifications for FR3 expert demonstrations."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any


class EpisodeSpecError(ValueError):
    """Raised when an episode requests a disabled or unknown variation."""


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    task: str
    language_prompt: str
    bracket_slot: str
    bracket_grasp_style: str
    robot_start_pose: str | None
    approach_offset_xyz_m: tuple[float, float, float]
    random_seed: int

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata["approach_offset_xyz_m"] = list(self.approach_offset_xyz_m)
        return metadata


def _enabled_names(config: dict[str, Any], section: str) -> list[str]:
    values = config.get(section)
    if not isinstance(values, dict):
        raise EpisodeSpecError(f"config section {section!r} must be an object")
    enabled = sorted(name for name, value in values.items() if isinstance(value, dict) and value.get("enabled"))
    if not enabled:
        raise EpisodeSpecError(f"config section {section!r} has no enabled entries")
    return enabled


def generate_bracket_episode_spec(
    config: dict[str, Any],
    *,
    episode_index: int,
    random_seed: int,
    bracket_slot: str | None = None,
    bracket_grasp_style: str | None = None,
) -> EpisodeSpec:
    """Select a deterministic discrete bracket slot and grasp style."""

    if episode_index < 1:
        raise EpisodeSpecError("episode_index must be at least 1")
    enabled_slots = _enabled_names(config, "slots")
    enabled_grasps = _enabled_names(config, "grasp_styles")
    if bracket_slot is not None:
        bracket_slot = bracket_slot.strip().upper()
        if bracket_slot not in enabled_slots:
            raise EpisodeSpecError(
                f"bracket slot {bracket_slot!r} is not enabled; enabled slots: {enabled_slots}"
            )
    if bracket_grasp_style is not None:
        bracket_grasp_style = bracket_grasp_style.strip().upper()
        if bracket_grasp_style not in enabled_grasps:
            raise EpisodeSpecError(
                f"grasp style {bracket_grasp_style!r} is not enabled; enabled styles: {enabled_grasps}"
            )

    generator = random.Random(random_seed)
    selected_slot = bracket_slot or generator.choice(enabled_slots)
    selected_grasp = bracket_grasp_style or generator.choice(enabled_grasps)

    randomization = config.get("randomization", {})
    if randomization.get("continuous_jitter_enabled", False):
        raise EpisodeSpecError("continuous jitter is disabled for Stage 1 and is not implemented")

    return EpisodeSpec(
        episode_index=episode_index,
        task="bracket",
        language_prompt=str(config["language_prompt"]),
        bracket_slot=selected_slot,
        bracket_grasp_style=selected_grasp,
        robot_start_pose=config.get("robot_start_pose"),
        approach_offset_xyz_m=(0.0, 0.0, 0.0),
        random_seed=random_seed,
    )
