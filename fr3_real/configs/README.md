# FR3 configuration map

Run commands from `fr3_real/`. Paths inside these JSON files are consumed by
live and offline scripts. Do not move or rename a config, trajectory, or
calibration file without checking its callers and generated per-cell copies.

| Family | Examples | Purpose |
| --- | --- | --- |
| Full assembly | `autonomous_demo_collection_v1.json`, `full_assembly_*_production_v1.json` | Camera, task, speed, reset and stage settings for bracket plus bolt. |
| Bracket source and replay | `bracket_manual_recording_v1.json`, `bracket_manual_replay_1x_highcap_b*.json` | Manual source and commissioned bracket stage paths. B-cell files are distinct physical starts. |
| New bracket cells | `bracket_new_cell_b97_trial_v1.json`, `bracket_workspace_cell_tracker.xlsx` | Wall-cell template and bracket-only status workbook. Per-proposal outputs are in `diagnostics/bracket_workspace_cell_loop_v1/`. |
| Bolt | `piloted_bolt_*.json`, `bolt_combined_thread_*.json`, `bolt_ratchet_reset_v1.json` | Bolt source, insertion and guarded threading settings. |
| Cube and older FR3 studies | `fr3_*cell*.json`, `fr3_spatial_validation*.json`, `fr3_*tracker*.xlsx` | Separate cube/spatial-evaluation campaigns. Do not use for bracket collection. |

The [bracket/bolt workflow guide](../../docs/BRACKET_BOLT_WORKFLOWS.md)
explains which file is authoritative for each stage. The
[FR3 Recording Guide](../../docs/RECORDING_GUIDE.md) covers the cube data.
Calibration measurements live in `../calibration/`, not in a generic JSON
configuration that can be reused on another robot.
