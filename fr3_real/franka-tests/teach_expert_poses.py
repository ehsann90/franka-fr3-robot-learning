#!/usr/bin/env python3
"""Interactively save read-only FR3 state snapshots as named calibrations.

This utility does not command motion, guiding mode, error recovery, collision
behavior, impedance, or the gripper. The operator must enable and use Franka's
supported hand-guiding mode independently through Desk/pilot controls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.calibration import CalibrationError, CalibrationStore, normalize_pose_name  # noqa: E402
from expert_demo.snapshots import SnapshotReadError, capture_robot_snapshot  # noqa: E402


DEFAULT_CALIBRATION = FR3_ROOT / "calibration" / "expert_poses.json"


def print_pose_list(store: CalibrationStore) -> None:
    poses = store.poses
    if not poses:
        print("[calibration] no saved poses")
        return
    print(f"[calibration] {len(poses)} saved pose(s); hash={store.calibration_hash}")
    for name in sorted(poses):
        snapshot = poses[name]
        xyz = snapshot["translation_xyz_m"]
        gripper = snapshot["gripper"]
        width = gripper.get("width_m") if gripper.get("available") else None
        width_text = "unavailable" if width is None else f"{float(width):.4f} m"
        print(
            f"  {name:<32} kind={snapshot['snapshot_kind']:<14} "
            f"xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}) "
            f"gripper={width_text} saved={snapshot['timestamp_utc']}"
        )


def prompt_name(prompt: str = "pose name: ") -> str:
    while True:
        try:
            return normalize_pose_name(input(prompt))
        except CalibrationError as error:
            print(f"[invalid] {error}")


def save_new_snapshot(store: CalibrationStore, robot, gripper, snapshot_kind: str) -> None:
    name = prompt_name()
    if name in store.list_pose_names():
        print(f"[refused] {name} already exists. Use menu option 5 to overwrite it explicitly.")
        return
    notes = input("notes (optional): ").strip()
    snapshot = capture_robot_snapshot(robot, gripper, snapshot_kind=snapshot_kind, notes=notes)
    store.add(name, snapshot)
    xyz = snapshot["translation_xyz_m"]
    print(f"[saved] {name}: xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})")
    print(f"        file={store.path}")
    print(f"        calibration_hash={store.calibration_hash}")


def overwrite_snapshot(store: CalibrationStore, robot, gripper) -> None:
    if not store.list_pose_names():
        print("[calibration] there are no poses to overwrite")
        return
    print_pose_list(store)
    name = prompt_name("existing pose name to overwrite: ")
    poses = store.poses
    if name not in poses:
        print(f"[refused] {name} does not exist")
        return
    confirmation = input(f"Type OVERWRITE {name} to replace it: ").strip()
    if confirmation != f"OVERWRITE {name}":
        print("[cancelled] overwrite confirmation did not match")
        return
    notes = input("replacement notes (optional): ").strip()
    previous_kind = str(poses[name]["snapshot_kind"])
    snapshot = capture_robot_snapshot(robot, gripper, snapshot_kind=previous_kind, notes=notes)
    backup = store.overwrite(name, snapshot)
    print(f"[overwritten] {name}")
    print(f"              backup={backup}")
    print(f"              calibration_hash={store.calibration_hash}")


def run_menu(store: CalibrationStore, robot, gripper) -> None:
    menu = """
[1] save current EE pose
[2] save current joint pose
[3] save current gripper state
[4] list saved poses
[5] overwrite a pose with confirmation
[6] exit
"""
    while True:
        print(menu)
        choice = input("choice: ").strip()
        try:
            if choice == "1":
                save_new_snapshot(store, robot, gripper, "ee_pose")
            elif choice == "2":
                save_new_snapshot(store, robot, gripper, "joint_pose")
            elif choice == "3":
                if gripper is None:
                    print("[refused] gripper state is unavailable because --no-gripper was selected")
                else:
                    save_new_snapshot(store, robot, gripper, "gripper_state")
            elif choice == "4":
                print_pose_list(store)
            elif choice == "5":
                overwrite_snapshot(store, robot, gripper)
            elif choice == "6":
                print("[teach] exiting; no robot or gripper command was issued")
                return
            else:
                print("[invalid] choose 1 through 6")
        except (CalibrationError, SnapshotReadError, ValueError, OSError) as error:
            print(f"[error] snapshot was not saved: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--teach-only",
        action="store_true",
        help="Required acknowledgement that this process is snapshot-only and will not execute motion.",
    )
    parser.add_argument("--no-gripper", action="store_true", help="Do not connect to/read the Franka Hand.")
    parser.add_argument("--list-only", action="store_true", help="List the calibration without connecting to hardware.")
    args = parser.parse_args()

    try:
        store = CalibrationStore(args.calibration, args.robot_ip)
    except CalibrationError as error:
        raise SystemExit(f"[calibration] {error}") from error

    print("=== FR3 EXPERT POSE TEACHER ===")
    print(f"mode:             {'LIST_ONLY' if args.list_only else 'TEACH_ONLY / READ_ONLY'}")
    print(f"robot IP:         {args.robot_ip}")
    print(f"calibration file: {store.path}")
    print(f"schema version:   {store.document['schema_version']}")
    print("motion commands:  DISABLED BY DESIGN")
    print("guide-mode calls: DISABLED BY DESIGN")
    print("gripper commands: DISABLED BY DESIGN")

    if args.list_only:
        print_pose_list(store)
        return
    if not args.teach_only:
        raise SystemExit("Refusing hardware connection without explicit --teach-only")

    print("\nUse Franka Desk/pilot controls for supported hand guiding.")
    print("This program will only snapshot measured state when you choose a save operation.")
    confirmation = input("Type CONNECT_READ_ONLY to connect without commanding motion: ").strip()
    if confirmation != "CONNECT_READ_ONLY":
        raise SystemExit("[teach] cancelled before hardware connection")

    from franky import Gripper, Robot  # noqa: PLC0415

    print(f"[teach] connecting read-only state client to {args.robot_ip} ...")
    robot = Robot(args.robot_ip)
    gripper = None
    if not args.no_gripper:
        try:
            gripper = Gripper(args.robot_ip)
            _ = gripper.state
        except Exception as error:
            raise SystemExit(
                f"[teach] Franka Hand state connection failed: {error}. "
                "Fix the hand connection or deliberately rerun with --no-gripper."
            ) from error
    # Read once before presenting the menu so incompatible/missing state fields
    # fail without creating a calibration pose.
    try:
        preflight = capture_robot_snapshot(robot, gripper, snapshot_kind="preflight")
    except SnapshotReadError as error:
        raise SystemExit(f"[teach] read-only preflight failed: {error}") from error
    xyz = preflight["translation_xyz_m"]
    print(f"[teach] read-only preflight OK; q={len(preflight['q'])} dq={len(preflight['dq'])}")
    print(f"[teach] current EE xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})")

    try:
        run_menu(store, robot, gripper)
    except (KeyboardInterrupt, EOFError):
        print("\n[teach] input interrupted; exiting without commanding the robot")


if __name__ == "__main__":
    main()
