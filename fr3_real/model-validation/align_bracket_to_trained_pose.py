#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-validation/align_bracket_to_trained_pose.py --help
"""Place the physical bracket exactly where a chosen training episode had it.

A "trained position" placed by eye is not the trained position. A lateral error
of a centimetre changes the wrist view enough that the policy may approach to the
right height yet never see the pre-grasp appearance it learned to close on --
which looks like a grasp failure but is really a setup mismatch.

This shows the chosen episode's first exterior frame against the live camera so
the bracket can be nudged until they coincide. Read-only: no robot, no Hand.

  [r] select the bracket region   [b] blend   [s] side-by-side
  [d] difference   [q] quit

The arm is deliberately NOT expected to match: evaluation always starts from the
fixed medoid start pose, not from this episode's own start pose. The arm fills a
large part of the exterior view, so a whole-frame difference score is dominated
by arm pixels and says nothing useful. Press [r] once and drag a box around the
bracket; the score is then computed only inside that box.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(FR3_ROOT), str(FR3_ROOT / "model-validation")]
from shadow_fr3_policy import read_rgb, start_camera  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=FR3_ROOT / "configs/bracket_policy_eval_v1.json")
    ap.add_argument("--raw_dir", type=Path, default=FR3_ROOT / "recordings/full_bracket_bolt_autonomous_raw_v1")
    ap.add_argument("--slot", help="bracket slot to reproduce, e.g. B2 (default: the medoid start episode)")
    ap.add_argument("--ref_frame", type=int, default=60,
                    help="reference frame index. NOT 0: RealSense auto-exposure needs ~20-30 frames "
                         "to converge, and frame 0 is ~70%% darker than the settled image, which "
                         "swamps the alignment score with a brightness difference.")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    cams = cfg["cameras"]

    target = None
    for d in sorted(args.raw_dir.iterdir()):
        mp = d / "metadata.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        if args.slot:
            if str(meta.get("bracket_slot", "")).upper() == args.slot.upper():
                target = d
                break
        elif d.name == cfg["start_pose"]["source_episode"]:
            target = d
            break
    if target is None:
        raise SystemExit(f"no episode found for slot {args.slot!r}")
    meta = json.loads((target / "metadata.json").read_text())
    print(f"reference episode : {target.name}")
    print(f"bracket slot      : {meta.get('bracket_slot')}")

    vid = sorted(target.glob("camera_*_rgb.mp4"))[0]
    cap = cv2.VideoCapture(str(vid))
    ref_bgr = None
    try:
        for i in range(args.ref_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            ref_bgr = frame
    finally:
        cap.release()
    if ref_bgr is None:
        raise SystemExit(f"could not read frames from {vid}")
    print(f"reference frame   : {args.ref_frame} (mean brightness {ref_bgr.mean():.1f})")

    pipe = start_camera(cams["exterior_serial"], cams["width"], cams["height"], cams["fps"])
    mode = "d"
    roi = None  # (x, y, w, h); until set, the score covers the whole frame
    print("\nPress [r] and drag a box around the BRACKET, then nudge it to minimise the score.")
    print("Ignore the arm -- it is expected to differ (eval starts from the fixed medoid pose).")
    print("  [r] select bracket region   [b] blend   [s] side-by-side   [d] difference   [q] done\n")
    try:
        for _ in range(10):
            read_rgb(pipe)
        while True:
            live_bgr = cv2.cvtColor(read_rgb(pipe), cv2.COLOR_RGB2BGR)
            if mode == "s":
                view = np.hstack([ref_bgr, live_bgr])
                cv2.putText(view, "REFERENCE (trained)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(view, "LIVE", (ref_bgr.shape[1] + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            elif mode == "d":
                view = cv2.absdiff(ref_bgr, live_bgr)
                if roi is not None:
                    x, y, w, h = roi
                    score = float(view[y:y + h, x:x + w].mean())
                    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = (f"BRACKET REGION  mean={score:.1f}  (relative only -- minimise by nudging)")
                else:
                    score = float(view.mean())
                    label = f"whole frame mean={score:.1f} -- press [r], the arm dominates this"
                cv2.putText(view, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                view = cv2.addWeighted(ref_bgr, 0.5, live_bgr, 0.5, 0)
                cv2.putText(view, "blend: green=trained, live overlaid", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("bracket alignment", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("r"):
                sel = cv2.selectROI("bracket alignment", live_bgr, showCrosshair=True, fromCenter=False)
                roi = tuple(int(v) for v in sel) if sel[2] > 0 and sel[3] > 0 else None
                if roi:
                    print(f"bracket region set: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
                mode = "d"
            elif k in (ord("b"), ord("s"), ord("d")):
                mode = chr(k)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    if roi is not None:
        print(f"final bracket-region difference: {score:.1f}")
    print("Alignment window closed. Run the eval trial without disturbing the bracket.")


if __name__ == "__main__":
    main()
