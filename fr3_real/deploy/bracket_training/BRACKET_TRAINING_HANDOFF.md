# FR3 Bracket Pick-and-Place — pi0.5-DROID full fine-tune

> Dated training-run handoff (August 2026). Its remote host, episode counts,
> checkpoint names and launch settings describe that run, not the current
> wall-cell collection loop. For new bracket placement captures and the
> bracket/bolt action-schema split, see the
> [current workflow guide](../../../docs/BRACKET_BOLT_WORKFLOWS.md).

Bracket pick-and-place **only**. No bolt pickup, no insertion, no threading —
every episode is cut at the `bracket_retreat_open_completed` event.

---

## What gets trained

| | |
|---|---|
| Task prompt | `Place the 3D-printed bracket onto the wall alignment pins.` |
| Source captures | **80** episodes, 80 distinct bracket positions |
| — 50 of them | full-assembly captures, cut at `bracket_retreat_open_completed` |
| — 29 of them | bracket-only captures recorded 2026-08-18 |
| — 1 of them | episode 52, admitted by name after its bolt stage faulted |
| Bracket span | 23.0 s – 62.0 s, mean 32.4 s |
| Dataset size | ~38,800 frames @ 15 Hz ≈ **43.2 minutes** of motion (1.56x the first run) |
| Schema | `droid_joint_velocity` (from libfranka `dq_d`), matches `pi05_droid` |
| Base model | `pi05_droid_finetune` → pi0.5-DROID checkpoint |
| Method | **Full-parameter** fine-tune (not LoRA) |
| Steps / batch | 13000 × 32 = 416,000 samples ≈ **10.7 epochs** |
| LR schedule | cosine, warmup 500, 1e-5 → 1e-6 (decay matched to the run length) |
| Checkpoints | every 500 steps (26 total); every 1000 kept permanently (13) |
| Norm stats | **skipped** — inherits pi05-DROID assets, matching the proven cube launcher |
| Env | `WANDB_MODE=offline`, `CUDA_VISIBLE_DEVICES=0`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` |

Verified before shipping, against the real converter code:

```
--force_include absent  : 79 dirs -> 79 episodes, 0 rejected
--force_include ep52    : 80 dirs -> 80 episodes, 0 rejected
--allow_bracket_only_captures=False : 50 episodes, 29 rejected
```

The 50/79/80 progression is the control: each flag admits exactly the captures it
names and nothing else. Both guards were re-tested after wiring and still bite --
a misspelled `--force_include` name raises rather than silently dropping the
episode, and a genuine dry run (take 2 of episode 78) is still refused for a
missing bracket event sequence even when named explicitly.

### Why the two new converter flags exist

`--allow_bracket_only_captures` — the recorder sets `setup_validation_only: true`
on any run that stops after the bracket phase, which is correct for a dry run but
also catches all 29 real bracket-only demonstrations. The converter used to raise
on that flag unconditionally. It now admits such a capture into `bracket_only`
**only** after verifying from the capture itself that all six bracket events are
present and that the Hand actually closed (peak commanded closedness ≥ 0.40;
measured range is 0.000 open / 0.521–0.625 closed on the 38.3 mm bracket). A true
dry run has no grasp and is still rejected. This is a verification, not a bypass.

`--latest_take_per_episode` — episode indices 52, 69 and 78 were retried. Index 78
has **two** successful takes, so without dedup that bracket position would be
duplicated in the dataset. Directory names begin with a sortable timestamp, so the
newest directory per `episode_index` wins.

`--force_include` — episode 52 is included by explicit name. All four of its takes
have `success: false`, and its newest take is additionally flagged
`bc_training_eligible: false` and `setup_validation_only: true`, so it was being
dropped by three separate filters. The fault was `direct +85-degree drive did not
complete normally` — the **bolt** stage, after a clean bracket placement — and the
operator confirmed the bracket half on 2026-08-18. Rather than loosen any of those
three filters (which would have admitted unrelated captures), the flag names this
one directory and still runs the full bracket-phase verification on it.

The take used is `20260818_144139_..._0052_B52_S1_CENTER`. Note that episode 52's
two earliest takes are labelled slot **B2**, not B52; only the two newest are at
B52, and dedup selects the newest, so the dataset gets B52.

---

## Tonight — launch (3 commands)

```bash
# 1. from the laptop: push data + scripts (prompts for the intern2 password)
cd ~/Job/franka-fr3-robot-learning/fr3_real/deploy/bracket_training
./push_to_intern2.sh

# 2. on the training machine: 2-second sanity check, no work performed
ssh intern2@10.6.38.133
cd ~/franka_r3 && ./run_bracket_pipeline.sh --check

# 3. if --check passes, launch. survives logout.
cd ~/franka_r3 && setsid nohup ./run_bracket_pipeline.sh >/dev/null 2>&1 &
```

Do **not** skip step 2. It catches every missing path, a broken interpreter,
a wrong episode count, and whether `scripts/train.py --help` runs — in seconds
rather than after an hour of video conversion.

---

## In the morning — one command

Conversion and training write to **one shared log**, so a single view follows
the whole run:

```bash
ssh intern2@10.6.38.133
cd ~/franka_r3 && ./watch_bracket_pipeline.sh
```

Prints state, phase, elapsed, phase timeline, checkpoints written, and GPU —
then follows the live log. `Ctrl-C` only detaches the viewer; it never touches
the run.

```bash
./watch_bracket_pipeline.sh --status     # summary only, no follow
cat ~/franka_r3/bracket_pipeline_status.txt   # single-line answer: SUCCESS / FAILED / RUNNING
```

The status file's `state` is the fastest answer to "did it work":
`RUNNING`, `SUCCESS`, or `FAILED` (with the failing phase and line).

---

## Phase order (fail-fast)

1. **preflight** — paths, interpreter, `cv2`/`lerobot` imports, episode count, trainer CLI
2. **register config** — writes `pi05_fr3_bracket_full` into OpenPI's `config.py` *(~1 s, deliberately before the slow phase)*
3. **convert** — raw → LeRobot, bracket portion only *(the long one)*
4. **verify dataset** — size, episode/frame counts, warns if ≠ 50
5. **norm stats** — `compute_norm_stats.py`
6. **train** — 6000 steps

Any phase failing stops everything after it. Training can never start on a
half-converted dataset.

---

## Aligned with the proven cube launcher

Checked against `run_fr3_real_dell_full.sh` (the launcher that actually worked
for the cube task). Three things were corrected as a result:

| | cube launcher | fixed here |
|---|---|---|
| `compute_norm_stats` | not run at all | **skipped by default** (`SKIP_NORM_STATS=1`) |
| `WANDB_MODE` | `offline` | now exported — without it training can block on wandb |
| `CUDA_VISIBLE_DEVICES` | `0` | now exported |

The remaining difference is intentional: the cube script calls
`uv run --no-sync`, this one calls `openpi/.venv/bin/python` directly. Both
bypass uv's dependency re-sync (the thing that fails on `evdev`); the direct
interpreter was verified working in preflight on this machine.

`scripts/train.py` is invoked as
`"$PY" scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite`.
`--overwrite` is the only flag the cube launcher does not pass — it is there so
a re-run with the same `EXP_NAME` (for instance after an OOM, with
`SKIP_CONVERT=1`) does not abort on an existing checkpoint directory.

---

## Tuning without editing files

Every knob is an environment variable:

```bash
STEPS=4000 BATCH_SIZE=16 setsid nohup ./run_bracket_pipeline.sh >/dev/null 2>&1 &
```

| Variable | Default | Use |
|---|---|---|
| `STEPS` | `13000` | shorter/longer run |
| `BATCH_SIZE` | `48` | lower if the GPU OOMs |
| `SAVE_INTERVAL` | `1000` | checkpoint frequency |
| `SKIP_CONVERT` | `0` | `1` to reuse an already-converted dataset |
| `SKIP_NORM_STATS` | `1` | `0` to force computing norm stats |
| `EXP_NAME` | `fr3_bracket_full_<date>` | separate runs |
| `CONFIG_NAME` | `pi05_fr3_bracket_full` | OpenPI config name |
| `REPO_ID` | `local/fr3_bracket_droid` | LeRobot dataset id |

---

## If something failed overnight

```bash
cat ~/franka_r3/bracket_pipeline_status.txt        # which phase, which line
grep -n "PHASE:\|FAILED\|Error\|Traceback" ~/franka_r3/bracket_pipeline_*.log | tail -40
```

**GPU OOM during training** — conversion is already done, so skip it:
```bash
SKIP_CONVERT=1 BATCH_SIZE=16 setsid nohup ./run_bracket_pipeline.sh >/dev/null 2>&1 &
```

**Config got mangled** — every run backs up OpenPI's config first:
```bash
ls ~/franka_r3/openpi/src/openpi/training/config.py.bak_*
cp ~/franka_r3/openpi/src/openpi/training/config.py.bak_<stamp> \
   ~/franka_r3/openpi/src/openpi/training/config.py
```

**Wrong hyperparameters** — the config generator is re-runnable:
```bash
cd ~/franka_r3/openpi
.venv/bin/python ~/franka_r3/prepare_pi05_fr3_bracket_full.py --replace --steps 4000
```

**`uv run` fails on `evdev`** — expected, and already avoided. `evdev` needs
kernel headers that aren't installed and no sudo is available. It's only pulled
in via `lerobot → pynput → evdev` for live teleop input, which nothing here
uses. Every command in the pipeline calls `openpi/.venv/bin/python` directly,
bypassing uv's re-sync entirely. Don't "fix" this by switching back to `uv run`.

---

## Expectations for this dataset

Set against the earlier research: Physical Intelligence's own guidance is
1–20 hours of data for a fine-tune, and their DROID docs assume a custom
dataset "<10s of hours". At **27.6 minutes**, this sits below that band.

Bracket placement is the favourable case for it — free-space, vision-driven,
one atomic skill, no force/contact reasoning (which is exactly what a
vision-only VLA like pi0.5 handles worst), and 80 genuinely distinct bracket
positions rather than 80 repeats of one.

Judge it on eval success rate at held-out positions, not on training loss.

### On 13000 steps / batch 32 — and what the first run taught us

The first run was 8000 steps at batch 32 over 50 episodes (~10.3 epochs), with
checkpoints kept only every 2000. Live evaluation on 2026-08-18 found:

| checkpoint | reach accuracy at the grasp pose | emitted a grasp? |
|---|---|---|
| 4000 | 4.3 mm lateral | **yes** (gripper 0.473) |
| 6000 | 7.1 mm lateral | no (flat ~0.000) |
| 7999 | **2.6 mm** lateral | no (flat ~0.000) |

Every checkpoint located the bracket to within ~7 mm, and the *best* localisation
came from the *final* one. What died with training was the discrete gripper
channel: 6000 and 7999 both drove to the grasp pose and then sat there
indefinitely with the gripper output pinned near zero. Confirmed offline by
replaying a saved observation — the exact frame that produced 0.473 live returned
a flat channel across 10 samples from the later checkpoints.

The grasp is roughly one frame in 500 per episode, so a model can lower its loss
by predicting the dominant "open" mode everywhere and paying for that one frame.
The continuous channels keep improving while the discrete one collapses, and
**training loss will not show it**.

13000 steps at batch 32 over ~38,800 frames is **10.7 epochs** — near-identical
to the first run's 10.3, but over 1.56x the data, so each position is seen fewer
times at the same total compute. That is the reason the longer run is defensible
here rather than a repeat of the same mistake:

| run | data | steps x batch | epochs |
|---|---|---|---|
| first | 24,900 frames | 8000 x 32 | 10.3 |
| **this** | **38,845 frames** | **13000 x 32** | **10.7** |

The real mitigation is checkpoint density. `SAVE_INTERVAL=500` gives **26**
checkpoints instead of the first run's four coarse samples, so the point where
the gripper channel dies is measurable rather than guessed. Expect the best
checkpoint to be well before the end.

Screen checkpoints cheaply before spending robot time. `model-validation/
replay_saved_observation.py` replays a saved observation against whichever
checkpoint is being served and reports whether the gripper channel crosses the
0.35 close threshold — seconds per checkpoint, no robot, no reflex recovery:

```bash
python model-validation/replay_saved_observation.py \
    --session B2_stream_01_approach --repeat 10
```

A checkpoint whose gripper channel is flat on that frame will not grasp on the
robot either. Only take the survivors to a live trial.
