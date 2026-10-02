#!/usr/bin/env python3
"""Guarded threading engagement chain from THREAD_READY, at an overridable
speed (default 4x / 0.12 rad/s, double the originally commissioned 2x /
0.06 rad/s), using the already-commissioned per-angle direct force
controllers.

Requires THREAD_READY to already be true (bolt upright, head-held, unloaded)
-- e.g. via run_bolt_dropregrasp_to_thread_ready_fast.py. Runs the same six
drive windows (85+85+85+50+50+50 degrees) as
record_autonomous_bolt_engagement_demo.py's full orchestrator, with the same
stationary open/empty-wrist-ratchet/regrasp before every window after the
first, and the same mandatory visual confirmations at every Hand action --
just extracted into a standalone bolt-only, non-recording run.

Force/admittance targets are identical across speeds (only the commanded
rotation rate changes) and each speed has its own separately commissioned
gate (threading_2x/3x/4x.combined_production_chain_enabled in the relevant
full_assembly_*_production_v1.json), so this does not weaken any guard --
it only selects which already-commissioned speed's binaries to run.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from record_autonomous_bolt_engagement_demo import (  # noqa: E402
    DEFAULT_RECORDING_CONFIG,
    dynamic_model_csv,
    load_inputs,
)
from run_bolt_dropregrasp_to_thread_ready_fast import check_head_grasp  # noqa: E402

SPEED_TO_LABEL = {"2x": "2x", "3x": "3x", "4x": "4x", "8x": "8x"}
SPEED_TO_RAD_S = {"2x": 0.06, "3x": 0.09, "4x": 0.12, "8x": 0.24}
# 8x has no prior physical commissioning history at all (see
# configs/full_assembly_8x_production_v1.json) -- flag it distinctly so the
# operator-facing prompts can't be mistaken for a routine, previously-tested tier.
UNPRECEDENTED_SPEEDS = {"8x"}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def build_sequence(task: dict, speed_label: str) -> list[dict]:
    sequence = [dict(stage) for stage in task["engagement_drive_sequence"]]
    for stage in sequence:
        base = str(stage["binary"])
        stage["binary"] = base if speed_label == "1x" else f"{base}_{speed_label}"
    return sequence


def run_direct_drive(
    stage: dict,
    robot_ip: str,
    task: dict,
    output_dir: Path,
    dynamic_model_path: Path | None = None,
) -> None:
    angle_deg = float(stage["angle_deg"])
    binary = project_path(stage["binary"])
    if not binary.is_file():
        raise SystemExit(f"missing compiled threading controller: {binary}")
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / f"direct_{stage['name']}.csv"
    with tempfile.TemporaryDirectory(prefix=f"fr3_threading_fast_{stage['name']}_") as temp:
        model_path = Path(temp) / "dynamic_model.csv"
        dynamic_model_csv(
            model_path,
            dynamic_model_path or project_path(task["dynamic_model_calibration"]),
        )
        print(f"Keep one hand near emergency stop. The commissioned +{angle_deg:.0f}-degree "
              f"drive begins now ({binary.name}).")
        completed = subprocess.run(
            [str(binary), "execute", robot_ip, str(telemetry_path),
             "COMBINED_THREAD_AUTHORIZED", str(model_path)],
            input="\n\n", text=True, capture_output=True, check=False,
        )
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0 or "[result] stop reason:      motion_completed" not in completed.stdout:
        raise RuntimeError(f"direct +{angle_deg:.0f}-degree drive ({stage['name']}) did not complete normally")
    print(f"[DONE] {stage['name']} telemetry saved: {telemetry_path}")


def ratchet_regrasp_head(robot, gripper, task: dict, drop: dict, head_grasp: dict, event_prefix: str) -> None:
    print(f"[{event_prefix}] opening the Hand while the bolt remains stationary")
    gripper.open(float(drop["gripper"]["open_speed_m_s"]))
    open_width = float(gripper.state.width)
    if open_width < float(drop["gripper"]["minimum_open_width_m"]):
        raise RuntimeError(f"{event_prefix} failed to open the Hand")
    if input(
        "Confirm the bolt stayed upright/fixed and the open-finger sweep is clear. "
        "Press Enter to continue; type anything to abort: "
    ):
        raise SystemExit(f"[threading] operator rejected {event_prefix} upright-bolt check")

    from franky import JointMotion, JointState  # noqa: PLC0415
    target = np.array(robot.state.q, dtype=np.float64, copy=True)
    start_q7 = float(target[6])
    target[6] = float(task["engagement_start_q7_rad"])
    robot.move(JointMotion(
        JointState(target),
        relative_dynamics_factor=float(task["empty_wrist_prewind_relative_dynamics_factor"]),
    ))
    final_q = np.asarray(robot.state.q, dtype=np.float64)
    error = float(np.max(np.abs(final_q - target)))
    print(f"[{event_prefix}] empty q7 reset {start_q7:+.6f} -> {final_q[6]:+.6f} rad; error={error:.6f}")
    if error > 0.005:
        raise RuntimeError(f"{event_prefix} empty-wrist reset tolerance failed")

    accepted = gripper.grasp(
        float(head_grasp["target_width_m"]), float(head_grasp["speed_m_s"]),
        float(head_grasp["force_N"]), float(head_grasp["epsilon_inner_m"]),
        float(head_grasp["epsilon_outer_m"]),
    )
    held_width = float(gripper.state.width)
    held = bool(gripper.is_grasped)
    strict, physical_evidence = check_head_grasp(
        head_grasp, accepted=bool(accepted), is_grasped=held, width=held_width
    )
    print(f"[{event_prefix}] accepted={accepted} is_grasped={held} width={held_width:.6f} m "
          f"strict={strict} physical_evidence={physical_evidence}")
    if not (strict or physical_evidence):
        raise RuntimeError(
            f"[SAFETY] {event_prefix} regrasp did not pass either check; Hand was NOT "
            "automatically reopened -- visually verify and clear manually"
        )
    if input(
        "Confirm the bolt is upright, securely head-held, and unloaded. Press Enter to "
        "continue; type anything to abort: "
    ):
        raise SystemExit(f"[threading] operator rejected {event_prefix} head regrasp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="offline + read-only live checks only")
    modes.add_argument("--execute", action="store_true", help="run the complete commissioned chain")
    modes.add_argument(
        "--commission-one",
        action="store_true",
        help=(
            "run exactly one supervised 8x window for commissioning; this never runs "
            "the complete chain and never enables the production gate"
        ),
    )
    modes.add_argument(
        "--commission-transition",
        action="store_true",
        help=(
            "run the supervised 8x ratchet/regrasp transition before a selected window; "
            "this never runs a rotation window or enables the production gate"
        ),
    )
    parser.add_argument("--speed", choices=["2x", "3x", "4x", "8x"], default="4x")
    parser.add_argument(
        "--commission-stage",
        default="drive_85_first",
        help="stage name for --commission-one (default: drive_85_first)",
    )
    parser.add_argument(
        "--dynamic-model",
        type=Path,
        default=None,
        help="review-only dynamic-model JSON override, intended for one-window commissioning",
    )
    parser.add_argument("--recording-config", type=Path, default=DEFAULT_RECORDING_CONFIG)
    args = parser.parse_args()
    if (args.commission_one or args.commission_transition) and args.speed != "8x":
        raise SystemExit("8x commissioning modes are only available for the uncommissioned 8x tier")

    collection, task, preinsert, drop, engagement, commissioning, plans = load_inputs(
        args.recording_config
    )
    del preinsert, engagement, commissioning, plans
    robot_ip = collection["robot_ip"]
    head_grasp = task["thread_ready_head_grasp"]
    dynamic_model_path = (
        project_path(args.dynamic_model)
        if args.dynamic_model is not None
        else project_path(task["dynamic_model_calibration"])
    )
    if not dynamic_model_path.is_file():
        raise SystemExit(f"dynamic model does not exist: {dynamic_model_path}")
    speed_config_path = FR3_ROOT / "configs" / f"full_assembly_{args.speed}_production_v1.json"
    speed_config = json.loads(speed_config_path.read_text())
    threading_key = f"threading_{args.speed}"
    threading_config = speed_config.get(threading_key, {})

    sequence = build_sequence(task, args.speed)
    output_dir = project_path(f"recordings/bolt_threading_fast_{args.speed}_trials_v1")

    print("=== FAST BOLT THREADING ENGAGEMENT CHAIN (FROM THREAD_READY) ===")
    mode_name = (
        "COMMISSION_ONE" if args.commission_one
        else "COMMISSION_TRANSITION" if args.commission_transition
        else "EXECUTE" if args.execute
        else "CHECK"
    )
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {robot_ip}")
    print(f"requested speed:          {args.speed} ({SPEED_TO_RAD_S[args.speed]:.3f} rad/s peak)")
    print(f"total rotation:           {float(task['total_engagement_rotation_deg']):.1f} deg "
          f"across {len(sequence)} windows")
    print(f"threading gate enabled:   {threading_config.get('combined_production_chain_enabled')}")
    print(f"threading gate status:    {threading_config.get('combined_production_chain_status')}")
    if args.speed in UNPRECEDENTED_SPEEDS:
        print(f"\n*** {args.speed} ({SPEED_TO_RAD_S[args.speed]:.3f} rad/s) has NO prior physical "
              "commissioning history. Every other tier here was individually validated before "
              "use; this one has not been. ***\n")

    blockers = []
    if (
        threading_config.get("combined_production_chain_enabled") is not True
        and (args.execute or args.commission_one or args.commission_transition)
    ):
        if args.execute:
            blockers.append(f"{threading_key}.combined_production_chain_enabled is not true "
                             f"in {speed_config_path.name}")
        else:
            print(f"[COMMISSION] {threading_key} remains disabled; supervised commissioning "
                  "is allowed without changing the production gate")
    elif threading_config.get("combined_production_chain_enabled") is not True:
        print(
            f"[CHECK] {threading_key} remains disabled; continuing with read-only checks only"
        )
    selected_stage = None
    transition_stage = None
    if args.commission_one:
        selected_stage = next(
            (stage for stage in sequence if stage.get("name") == args.commission_stage),
            None,
        )
        if selected_stage is None:
            valid = ", ".join(str(stage["name"]) for stage in sequence)
            blockers.append(
                f"unknown --commission-stage {args.commission_stage!r}; choose one of: {valid}"
            )
    if args.commission_transition:
        transition_name = args.commission_stage
        transition_stage = next(
            (stage for stage in sequence if stage.get("name") == transition_name),
            None,
        )
        if transition_stage is None:
            valid = ", ".join(str(stage["name"]) for stage in sequence)
            blockers.append(
                f"unknown transition target {transition_name!r}; choose one of: {valid}"
            )
        elif not (
            transition_stage.get("empty_wrist_ratchet_before")
            or transition_stage.get("stationary_open_regrasp_before")
        ):
            blockers.append(
                f"{transition_name} does not define a ratchet/regrasp transition"
            )

    stages_to_validate = (
        sequence if not (args.commission_one or args.commission_transition)
        else ([selected_stage] if selected_stage else [])
    )
    for stage in stages_to_validate:
        binary = project_path(stage["binary"])
        if not binary.is_file():
            blockers.append(f"missing compiled controller for {stage['name']}: {binary}")
        else:
            print(f"  {stage['name']:16s} +{stage['angle_deg']:.0f} deg  {binary.name}")
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] fast threading chain refused")

    input("Press Enter to connect and inspect live THREAD_READY state (Ctrl+C cancels; no command issued yet): ")
    from franky import Gripper, Robot  # noqa: PLC0415

    robot = Robot(robot_ip)
    gripper = Gripper(robot_ip)
    live_q7 = float(robot.state.q[6])
    dynamic_model = json.loads(
        dynamic_model_path.read_text()
    )
    dynamic_knots = np.asarray(dynamic_model["knots"], dtype=np.float64)
    selected_angle = float(selected_stage["angle_deg"]) if args.commission_one else None
    width = float(gripper.state.width)
    is_grasped = bool(gripper.is_grasped)
    strict, physical_evidence = check_head_grasp(
        head_grasp, accepted=is_grasped, is_grasped=is_grasped, width=width
    )
    print(f"\nlive Hand width/grasped: {width:.6f} m / {is_grasped}")
    print(f"head-hold strict/bounded: {strict} / {physical_evidence}")
    print(
        f"dynamic-model q7 interval: {dynamic_knots[0, 0]:+.6f} .. "
        f"{dynamic_knots[-1, 0]:+.6f} rad"
    )
    print(f"live q7:                  {live_q7:+.6f} rad")
    if selected_angle is not None:
        selected_target_q7 = live_q7 + np.deg2rad(selected_angle)
        print(f"requested endpoint q7:     {selected_target_q7:+.6f} rad")
        if selected_target_q7 > dynamic_knots[-1, 0]:
            print(
                "\nBlockers:\n"
                "  - insufficient calibrated dynamic-model q7 range for requested rotation\n"
                f"  - live q7 + {selected_angle:.0f} deg = {selected_target_q7:+.6f} rad, "
                f"but calibrated maximum is {dynamic_knots[-1, 0]:+.6f} rad\n"
                "  - reposition THREAD_READY to the calibrated start window or collect "
                "new dynamic-model data; do not extend the JSON bound by hand"
            )
            raise SystemExit(3)
    if (args.commission_one or args.commission_transition) and not strict:
        print(
            "\nBlockers:\n"
            "  - 8x commissioning requires strict calibrated head-hold confirmation; "
            "bounded evidence is insufficient"
        )
        raise SystemExit(3)
    if not (strict or physical_evidence):
        print("\nBlockers:\n  - THREAD_READY not confirmed live (Hand does not show evidence of holding the head)")
        raise SystemExit(3 if args.check else 1)
    if args.check:
        print("\nCheck complete. No arm or Hand command was issued.")
        return

    if args.commission_transition:
        print(
            f"\n[COMMISSION] 8x ratchet/regrasp transition before "
            f"{transition_stage['name']}"
        )
        print(
            "[COMMISSION] no rotation window will run; the production gate remains disabled"
        )
        if input(
            "Confirm the completed 85-degree window is physically stable and the bolt is "
            "still head-held. Press Enter to begin; type anything to abort: "
        ):
            raise SystemExit("[commissioning] operator aborted before the transition")
        ratchet_regrasp_head(
            robot, gripper, task, drop, head_grasp,
            f"commission_before_{transition_stage['name']}_ratchet",
        )
        print(
            "[COMMISSION] transition complete. Robot is at the next THREAD_READY start "
            "pose; no 8x rotation window was executed."
        )
        return

    if args.commission_one:
        stage = selected_stage
        print(
            f"\n[COMMISSION] one-window 8x trial: {stage['name']} "
            f"(+{float(stage['angle_deg']):.0f} deg)"
        )
        print("[COMMISSION] automatic ratchet, regrasp, and remaining windows are disabled")
        if input(
            "Confirm the bolt is unloaded, head-held, and positioned for this exact window. "
            "Press Enter to begin; type anything to abort: "
        ):
            raise SystemExit("[commissioning] operator aborted before the one-window drive")
        del robot, gripper
        gc.collect()
        time.sleep(0.5)
        commission_dir = project_path("recordings/bolt_threading_8x_commissioning_v1")
        run_direct_drive(stage, robot_ip, task, commission_dir, dynamic_model_path)
        print(
            "[COMMISSION] one-window trial complete. Inspect telemetry and the physical "
            "threading result before commissioning another window."
        )
        return

    if input(
        f"Confirm THREAD_READY one more time immediately before threading at {args.speed} "
        f"({SPEED_TO_RAD_S[args.speed]:.3f} rad/s). Press Enter to begin; type anything to abort: "
    ):
        raise SystemExit("[threading] operator aborted before the first drive")

    for index, stage in enumerate(sequence):
        if stage.get("empty_wrist_ratchet_before") or stage.get("stationary_open_regrasp_before"):
            ratchet_regrasp_head(robot, gripper, task, drop, head_grasp, f"before_{stage['name']}_ratchet")
        del robot, gripper
        gc.collect()
        time.sleep(0.5)
        run_direct_drive(stage, robot_ip, task, output_dir)
        robot = Robot(robot_ip)
        gripper = Gripper(robot_ip)
        print(f"[threading] {index + 1}/{len(sequence)} windows complete")

    input(
        f"The complete {float(task['total_engagement_rotation_deg']):.0f}-degree chain finished. "
        "Press Enter to open the stationary Hand and inspect engagement: "
    )
    gripper.open(float(drop["gripper"]["open_speed_m_s"]))
    release_width = float(gripper.state.width)
    if release_width < float(drop["gripper"]["minimum_open_width_m"]):
        raise RuntimeError("Hand failed to open after engagement")
    print(f"\n[DONE] chain complete, Hand open (width={release_width:.6f} m).")
    print("Visual endpoint review is required -- threading must not be inferred from yaw alone.")


if __name__ == "__main__":
    main()
