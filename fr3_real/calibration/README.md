# Expert-demonstration calibration

This directory also contains the bracket workcell measurements used by the
[current wall-cell workflow](../../docs/BRACKET_BOLT_WORKFLOWS.md):
`bracket_workspace_corners_v2/` records the probed tabletop corners and
candidate-cell preview; `bracket_wall_keepout_v1/` records the measured wall
face used by the translated-path keep-out audit. Their `probed_points.md`
files are measurement records, not instructions or robot motion targets.
Do not replace them with the older cube `probed_points` or extrapolate a
commissioned trajectory from a rectangular grid alone.

The expert-pose JSONs described below are a separate calibration family.

`teach_expert_poses.py` writes workcell- and robot-specific JSON calibration
files here by default. Those JSON files are intentionally ignored by Git.

The authoritative pose representation is the nested 4x4 `T_base_ee` matrix.
The stored XYZ and XYZW quaternion are descriptive, redundant fields used for
inspection. The convention is:

```text
T_A_B = pose of frame B expressed in frame A
p_A = T_A_B @ p_B
```

Existing pose names are never silently overwritten. An explicit overwrite
creates a timestamped copy of the complete prior calibration under
`.backups/` before the new snapshot is written atomically.

For each discrete bracket grasp style, the Stage-1 plan also requires a
`BRACKET_<STYLE>_TRANSFER_SAFE` pose. Teach it while the bracket is securely
held, at a clear intermediate waypoint between the pickup lift and wall
preplace. This prevents the planner from treating the full pickup-to-wall
transfer as one unreviewed Cartesian segment.
