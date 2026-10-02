# FR3 workcell

Run the commands in this directory from `fr3_real/` unless a guide says otherwise.
The repository-root [README](../README.md) covers installation and the older
cube pick-and-place pipeline. The current bracket, wall-cell, and bolt workflows
are in the [bracket/bolt guide](../docs/BRACKET_BOLT_WORKFLOWS.md).

| Directory | Purpose | Data or code? |
| --- | --- | --- |
| `franka-tests/` | Robot/camera checks and human-piloted source recording | Hardware-facing code |
| `model-running/` | Commissioned replay, wall-cell loop, full assembly, and policy evaluation | Hardware-facing code |
| `model-running/streaming/` | Direct libfranka controllers and local stream protocol | C++/Python code; `build/` is generated |
| `expert_demo/` | Replay plans, retiming, calibration and safety calculations | Shared code |
| `model-validation/` | Offline audits and read-only validation | Code |
| `data-pipeline/` | Raw-to-LeRobot conversion and tracker utilities | Code |
| `training/`, `deploy/` | Training launchers and remote-deployment handoffs | Code and dated runbooks |
| `configs/` | Workcell-specific JSON configuration and tracker workbooks | Inputs; keep their current paths |
| `calibration/` | Measured workcell geometry and calibration notes | Inputs; do not substitute between workcells |
| `recordings/`, `diagnostics/`, `eval_logs/` | Raw episodes, commissioning evidence, evaluations | Generated evidence, not source code |
| `tests/` | Offline regression tests | Code |

`recordings/` and `eval_logs/` are ignored by Git. A successful controller
trace is not, by itself, an accepted training episode: use episode metadata,
event boundaries, camera/state validation, and the operator's endpoint review.
Do not move or rename raw episodes, calibration files, or commissioned configs
without updating every consumer and retesting the resulting workflow.

For a script catalog, see the [script guide](../docs/SCRIPT_GUIDE.md). For the
older project history and file-by-file map, see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).
The [config map](configs/README.md) and [live entry-point map](model-running/README.md)
help navigate the two largest source directories without changing their paths.
