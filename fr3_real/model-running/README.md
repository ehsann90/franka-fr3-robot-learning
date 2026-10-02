# Live FR3 entry points

This directory contains live-motion scripts as well as read-only preparation
tools. A filename that includes `prepare` or `preview` is not a safety claim
for every flag. Read the selected script's `--help`, its mode banner and its
config before use. Keep the E-stop reachable for physical runs.

| Task | Start here |
| --- | --- |
| New bracket workspace cells | `record_bracket_workspace_cell_loop.py` |
| Wall-to-cell setup for one translated candidate | `run_bracket_wall_to_cell.py` |
| Full bracket and bolt capture or bracket-only capture | `record_autonomous_bolt_engagement_demo.py` |
| Bolt contact and tightening | `run_bolt_thread_drive_direct.py` and the commissioned combined-drive configs |
| Earlier arm commissioning | `commission_full_assembly_arm_2x.py`, `commission_manual_bracket_arm.py` |
| Policy evaluation | `run_bracket_policy_eval.py`, `run_fr3_policy_eval_trial.py` |

Planning and static checks are shared through `../expert_demo/`. The direct
libfranka controllers and local stream protocol are under `streaming/`.
Camera/state source recording is under `../franka-tests/`, conversion under
`../data-pipeline/`, and offline tests under `../tests/`.

See the [current bracket/bolt workflow](../../docs/BRACKET_BOLT_WORKFLOWS.md)
and the [script guide](../../docs/SCRIPT_GUIDE.md) for task boundaries and
commands. Generated controller binaries under `streaming/build/` are not
source files and are ignored by Git.
