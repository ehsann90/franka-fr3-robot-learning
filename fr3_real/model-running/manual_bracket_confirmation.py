"""Confirm a checked bracket Hand action without changing its preflight guards."""

from __future__ import annotations

from collections.abc import Callable


PHRASES = {"grasp": "CLOSE_ON_BRACKET", "release": "OPEN_AND_RELEASE"}


def confirm_hand_action(action: str, config: dict, input_fn: Callable[[str], str] = input) -> bool:
    if action not in PHRASES:
        raise ValueError(f"unknown bracket Hand action: {action}")
    if config.get("calibration_only") is True:
        verb = "grasp the bracket" if action == "grasp" else "release the bracket"
        return input_fn(f"Press Enter to {verb} now; type anything to abort: ") == ""
    phrase = PHRASES[action]
    verb = "grasp the real bracket" if action == "grasp" else "release the bracket"
    return input_fn(f"Type {phrase} to {verb} now; anything else aborts: ") == phrase
