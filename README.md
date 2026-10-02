# franka-fr3-robot-learning

Tools for Franka Research 3 robot learning: cube pick-and-place, bracket
placement on wall pins, bolt insertion/threading, recording and dataset
conversion, and policy training/evaluation. These are distinct tasks with
different action schemas and workcell assumptions.

Start with the [FR3 directory map](fr3_real/README.md). For current bracket,
wall-cell, and bolt collection, use the
[bracket/bolt workflow guide](docs/BRACKET_BOLT_WORKFLOWS.md). The
[recording guide](docs/RECORDING_GUIDE.md) remains the cube-specific workflow.
The [documentation index](docs/README.md) separates current guides from dated
runbooks and paper drafts.

## Released Artifacts

- Model checkpoint: [`pi0.5-fr3-v3-pick-place`](https://huggingface.co/msbutt1/pi05-fr3-v3-12000)
- LeRobot dataset: [`fr3-pick-place-lerobot`](https://huggingface.co/datasets/msbutt1/fr3-pick-place-lerobot)
- Live Evaluations: [`demos`](https://github.com/msbutt1/franka-fr3-robot-learning/wiki/Live-Policy-Evaluations)

The released cube policy and the dated bracket evaluation use pi0.5-DROID
joint-velocity actions: seven FR3 joint velocities plus gripper closedness at
15 Hz. The bracket/bolt autonomous collection config instead specifies
`droid_joint_position` (achieved-next absolute joints and gripper closedness).
Choose the schema for the intended policy; do not combine them silently.

## Repository Layout

The `fr3_real/` tools are grouped by operational purpose:

- `franka-tests/`: robot connection, kinematics, camera, gripper, probing, and hardware commissioning checks.
- `model-validation/`: policy-server smoke tests, shadow inference, offline replay, recording audits, and ranking tools that do not command robot motion.
- `model-running/`: guarded live policy execution, streaming validation, gripper control, recovery, reset, and the local streaming controller.
- `data-pipeline/`: recording conversion, dataset staging, cell tracking, manifest generation, and normalization statistics.
- `training/`: OpenPI/pi0.5 training launchers, dataset preparation, checkpoint preparation, and cluster/Slurm scripts.
- `configs/`: workcell-specific task configs, trackers, and run-selection metadata.
- `calibration/`: measured geometry and calibration evidence for this workcell.
- `recordings/`, `diagnostics/`, `eval_logs/`: generated episodes and trial evidence; not source code.

Shared modules such as `franka_motion.py` and `grid_utils.py` remain at the
`fr3_real/` root because multiple groups import them directly.

For operational entry points, see [the FR3 Script Guide](docs/SCRIPT_GUIDE.md).
The bracket/wall-cell tools remain in their existing directories because
scripts and configs refer to those paths; the new guides organize them without
renaming commissioned trajectories or raw data.

## Dataset Collection

### Bracket and bolt

The bracket task records placement from a selected table cell onto fixed wall
pins. Its wall-to-cell transfer is unrecorded setup. The bolt task starts at
S1, places the bolt into the mounted bracket, and can end at `THREAD_READY` for
a hybrid policy or include guarded threading for a full-policy view. The
bracket-specific workbook, wall keep-out, raw episodes, conversion views, and
current commands are documented in the
[bracket/bolt workflow guide](docs/BRACKET_BOLT_WORKFLOWS.md). A physical
controller completion is not a training-data acceptance decision.

### Cube pick-and-place

The released cube dataset is built from successful real-robot demonstrations. For each
demonstration, the collection script moves a cube from the basket to a selected
table cell, returns the arm to a consistent start pose, then records the robot
as it picks that cube and places it back in the basket. Two RealSense cameras,
robot joint and end-effector state, gripper state, and commanded actions are
saved together in one raw episode folder. Failed or incomplete attempts remain
separate and are excluded from normal training conversion.

After collection, the raw episodes are checked for complete state and camera
data, then converted into the LeRobot format. Training uses those synchronized
images, observations, actions, and task labels. The collection script controls
the demonstrated motion; the learned policy is not involved during recording.

For the complete operator workflow, including calibration, safe recording,
quality checks, conversion, and publishing, see the
[FR3 Recording Guide](docs/RECORDING_GUIDE.md).

## Python Setup

For the FR3 control and validation workstation, create the provided Conda
environment and install the repository's Python dependencies:

```bash
conda env create -f fr3_real/configs/environment-fr3-recording.yml
conda activate fr3-recording
python -m pip install -r requirements/fr3-recording.txt
python -m pip install -e ./openpi-client
```

The bootstrap command below performs the same local `openpi-client` install
only when it is not already importable. Real-robot use still requires
host-specific prerequisites outside Python: a compatible `libfranka`,
RealSense/librealsense access and USB permissions, and Linux headers when a
package needs to build native input bindings.

The local FR3 streamers specifically require the active-control libfranka
development headers, including `franka/active_control_base.h`. A normal
`franky` Python import is not sufficient to prove that this C++ requirement is
installed. The readiness checker reports this separately before live use.

For a fresh workstation, use the non-destructive bootstrap and then inspect
the readiness report before attempting any live command:

```bash
bash setup/bootstrap_fr3_environment.sh
conda run -n fr3-recording python setup/verify_fr3_setup.py
```

The same commands also work from `fr3_real/`, where compatibility launchers
are provided under `fr3_real/setup/`.

`verify_fr3_setup.py` never moves the robot, commands the gripper, or opens a
camera. Add `--check-policy-server` only to perform a TCP reachability check
against a running policy server.

## From Scratch To Evaluation

Use two machines when possible: an inference machine with the OpenPI checkout
and GPU, and the robot-control workstation with this repository, Franka FCI
access, the RealSense cameras, and the locally built streamer. Set the shell
variables below to locations appropriate for the host.

### 1. Prepare The Robot Workstation

Clone this repository and run the bootstrap from `fr3_real/`:

```bash
git clone https://github.com/msbutt1/franka-fr3-robot-learning.git
cd franka-fr3-robot-learning/fr3_real
bash setup/bootstrap_fr3_environment.sh
conda run -n fr3-recording python setup/verify_fr3_setup.py
```

Before a live evaluation, the workstation must have a compatible active-control
`libfranka` development installation, FCI enabled in Franka Desk, RealSense USB
permissions, the two camera serials configured in the scripts, a built local
streamer, and a calibrated
`configs/probed_points.json`. The calibration file is robot/workcell-specific:
do not reuse it blindly on another physical setup.

Build the local streamer once after installing its C++ prerequisites:

```bash
cd model-running/streaming
cmake -S . -B build
cmake --build build -j2
```

### 2. Download The Dataset

Install the Hugging Face CLI on the machine that will train or inspect data,
authenticate if the repository is private, then download the dataset snapshot:

```bash
hf auth login
hf download msbutt1/fr3-pick-place-lerobot \
  --repo-type dataset \
  --local-dir "$HOME/fr3_data/fr3-pick-place-lerobot"
```

The download is a LeRobot dataset. For OpenPI training, place or link it below
the configured LeRobot cache using the repository name expected by the chosen
training config, then use the scripts in `fr3_real/training/`.

### 3. Download And Serve The Policy

On the GPU/OpenPI machine, clone the OpenPI repository, create its supported
environment, and download the checkpoint:

```bash
hf auth login
hf download msbutt1/pi05-fr3-v3-12000 \
  --repo-type model \
  --local-dir "$HOME/checkpoints/pi05-fr3-v3-12000"

cd /path/to/openpi
uv run --no-sync scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir="$HOME/checkpoints/pi05-fr3-v3-12000"
```

Keep this server running. The robot workstation must be able to reach it on
TCP port `8000`; change the port consistently on both sides if needed.

### 4. Validate Without Robot Motion

First verify only the policy-server protocol from the robot workstation:

```bash
cd fr3_real
conda activate fr3-recording
python model-validation/test_fr3_policy_server.py \
  --host <policy-server-ip> --port 8000
```

Then run the setup checker with a TCP-only server check:

```bash
python setup/verify_fr3_setup.py \
  --server-host <policy-server-ip> \
  --server-port 8000 \
  --check-policy-server
```

### 5. Run A Guarded Live Evaluation

Start with a known manifest position and a clear workspace. The trial runner
uses the local reference streamer; the policy selects arm actions and gripper
transition timing, while the runner executes and verifies physical close/open
commands and performs scripted turnover after a verified release.

```bash
cd fr3_real
conda activate fr3-recording

python run_streaming_validation_trial.py \
  --eval-id I01 \
  --trial 1 \
  --checkpoint v3_12000 \
  --manifest configs/fr3_spatial_validation_v3.json \
  --points configs/probed_points.json \
  --robot-ip 172.16.0.2 \
  --server-host <policy-server-ip> \
  --server-port 8000 \
  --min-z 0.020 \
  --controller reference \
  --auto-start-streamer \
  --auto-gripper \
  --run
```

Keep the E-stop reachable. Do not disable workspace guards or use a different
calibration/workcell without first commissioning the recorded replay and a
known training cell. The result ledger is written under `fr3_real/eval_logs/`.

Before running a live script, check its `--help` output, robot IP, policy
server, checkpoint, workspace guards, E-stop access, and whether it commands
the arm or gripper.
