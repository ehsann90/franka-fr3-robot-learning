# Bracket, wall-cell, and bolt workflows

This is the current navigation guide for the FR3 bracket-and-bolt workcell.
Commands are relative to `fr3_real/`. The [cube recording guide](RECORDING_GUIDE.md)
documents a separate task; do not use its cube tracker, schema, or prompts for
bracket/bolt collection.

## Task and data boundaries

| Workflow | Physical start → end | Dataset boundary |
| --- | --- | --- |
| Bracket placement | Bracket at a selected table cell → seated on fixed wall pins; Hand open and arm retreated | Record the cell-to-wall placement. The wall-to-cell transfer is setup, not a training episode. |
| Bolt placement | Bolt at S1 → upright/self-supported in the mounted bracket, head regrasped | A `THREAD_READY` prefix can train a bolt-placement policy. |
| Threading | Head-held bolt → visually engaged/self-supported bolt | A guarded axial-admittance/yaw controller runs the contact motion in the hybrid workflow; a full-policy dataset includes it only when explicitly selected. |

The full-assembly recorder also supports a bracket-only capture. Its camera
order is exterior then wrist (serials in
`configs/autonomous_demo_collection_v1.json`), robot state is sampled at 30 Hz,
and conversion targets 15 Hz. The bracket/bolt collection config specifies
`droid_joint_position`: seven achieved-next absolute joint positions plus
achieved-next gripper closedness. The older cube pipeline and bracket policy
evaluation use `droid_joint_velocity`; do not mix these schemas in one dataset.

## Inputs and source of truth

- `configs/autonomous_demo_collection_v1.json` defines the full-assembly task,
  camera roles, Hand settings, and threading stages. Speed-specific arm and
  threading configs live alongside it in `configs/`.
- `configs/bracket_manual_replay_1x_highcap_b93.json` is the recorded wall-pose
  reference for the wall-cell path generator. A translated per-cell config and
  candidate path are written below `diagnostics/bracket_workspace_cell_loop_v1/`.
- `calibration/bracket_workspace_corners_v2/` and
  `calibration/bracket_wall_keepout_v1/` contain measured geometry and candidate
  previews. Proposals are not commissioned merely because they appear in a grid.
- `configs/bracket_workspace_cell_tracker.xlsx` is the bracket-specific workbook;
  the loop's JSON ledger is under `diagnostics/bracket_workspace_cell_loop_v1/`.
  Cube trackers are separate.
- `recordings/full_bracket_bolt_autonomous_raw_v1/` holds raw camera/state
  episodes. Do not infer acceptance from the directory name or the final
  controller message; inspect metadata and the physical endpoint.

## Bracket wall-cell loop

The loop prepares a translated target path, checks inverse kinematics and
offline limits, performs the unrecorded wall-to-cell setup, then records the
cell-to-wall placement with both cameras. It stops on a blocked target or a
failed physical cycle. The fixed wall pose is reusable, but each new cell
still needs a valid path and a correctly placed bracket.

```bash
cd /path/to/franka-fr3-robot-learning/fr3_real
conda activate fr3-recording

# Rebuild/view the bracket-specific workbook; no robot connection.
python model-running/record_bracket_workspace_cell_loop.py --preview-only

# After checking the selected proposal, current setup and recorded start:
python model-running/record_bracket_workspace_cell_loop.py \
  --execute-supervised-trials \
  --start-proposal Pxx_yy --count 1 --start-episode N
```

Replace `Pxx_yy` and `N` with an eligible proposal and unused episode index.
`--start-proposal` chooses the first eligible, `UNTESTED`, in-hull row at or
after that ID; it does not force that exact row. For a blocked/failed row use
`--retry-proposal Pxx_yy` after resolving its cause. If setup already succeeded
and the bracket remains at that cell, combine one `--retry-proposal` with
`--record-placement-only --count 1` to record placement without repeating
wall-to-cell transfer. Never use placement-only after the bracket has moved.

The physical loop can reject a cell even when it lies inside the tabletop
rectangle: the translated arm/held-object path also has to clear the measured
wall keep-out. A controller-mode rejection or reset miss is not evidence of a
valid recording. Check the tracker and episode directory before assigning the
next episode index. The setup transfer speed can differ from the recorded
placement speed; the generated config is the authority for each stage.

## Manual source recording and full assembly

For a new human-piloted bracket source, use
`franka-tests/record_manual_bracket_demo.py`. The `OUTER` wrist-side label
describes the camera viewpoint; it does not rotate the wrist automatically.
For bolt source capture use `franka-tests/record_manual_bolt_demo.py`.
Both are separate from cube collection. See their `--help` and
[PROJECT_HANDOFF.md](../fr3_real/PROJECT_HANDOFF.md) for the Pilot marker
sequence and earlier commissioning history.

The autonomous entry point is
`model-running/record_autonomous_bolt_engagement_demo.py` with
`--recording-preview` or `--record-demo`, `--full-assembly`, an approved speed
profile, and `--episode-index`. Add `--bracket-only` to stop after the bracket
placement. `--training-data` is a separate eligibility switch; recording a
setup run does not make it training data. Speed names are targets: controller
caps can stretch stage durations. The thread chain is currently configured
as three 85-degree and three 50-degree windows with stationary open/reset/
regrasp between windows and a relative axial-force target; rotation count or
measured yaw alone does not prove engagement.

## Review and conversion

Review the raw videos, event boundaries, robot/Hand state, and the physical
result before admitting an episode. The converter is
`data-pipeline/convert_raw_recordings_to_lerobot.py`; it supports
`--schema droid_joint_position` and `--derived_view whole`,
`two_task_full_policy`, or `two_task_hybrid`. The hybrid view ends the bolt
policy prefix at `THREAD_READY`, leaving threading to the guarded controller.
The bracket policy training/evaluation runbooks in `deploy/` describe dated
experiments and may use a different action schema; do not assume their
checkpoint, episode counts, or remote host are current.

Before any live command, inspect its `--help`, selected config, start pose,
physical objects, workspace, robot mode, and E-stop access. Do not weaken
collision/reflex or force/trajectory limits to get past a blocker.
