# Bracket policy evaluation — operator guide

> Dated evaluation runbook (August 2026). The checkpoint, remote host, limits
> and trial counts below describe that evaluation setup; they are not defaults
> for new wall-cell dataset collection. See the
> [current bracket/bolt workflow guide](../../../docs/BRACKET_BOLT_WORKFLOWS.md)
> for collection and schema boundaries.

Evaluating `pi05_fr3_bracket_full` (trained 2026-08-18, 50 bracket-only episodes,
checkpoints 2000 / 4000 / 6000 / 7999).

The goal of this rig is narrow and specific: **make the evaluation environment
provably not the thing under test**, so a failed trial is the policy's fault.

---

## 1. Start the policy server (on intern2)

Use `.venv/bin/python`, **not** `uv run` — uv re-syncs and dies on `evdev`.

```bash
ssh intern2@10.6.38.133
cd ~/franka_r3/openpi

CKPT=6000        # evaluate a spread: 4000, 6000, 7999
.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_fr3_bracket_full \
  --policy.dir=$HOME/franka_r3/openpi/checkpoints/pi05_fr3_bracket_full/fr3_bracket_full_20260818/$CKPT
```

Serve one checkpoint at a time; restart with a different `$CKPT` to switch.
The server binds `0.0.0.0:8000`, so the robot laptop reaches it directly.

Leave it running (`setsid nohup … &` if you want it detached).

---

## 2. Preflight — run this before every session

From the **robot laptop**, in `fr3_real/`:

```bash
# observation pipeline only (no robot, no cameras) — start here
python model-validation/bracket_eval_preflight.py

# full check, once the robot and cameras are powered
python model-validation/bracket_eval_preflight.py --with-cameras --with-robot
```

### What check B actually proves

It rebuilds **real recorded training frames** into live-format observations,
sends them to the running policy, and compares the returned 16-step chunk to
the ground-truth action recorded at that instant.

If the eval pipeline differed from training in colourspace, resize geometry,
state layout, gripper scaling, prompt text, or served checkpoint, that
correlation collapses. A high cosine similarity with a magnitude ratio near
1.0 means the observation contract is intact end-to-end.

π0.5 is a **stochastic flow-matching** policy: sampled twice on the *same*
observation it only agrees with itself at cos ≈ 0.6. So an absolute cosine
threshold is meaningless. Check B therefore measures three things and judges
agreement **relative to the policy's own ceiling**:

| metric | measured 2026-08-18 (ckpt 6000) | meaning |
|---|---|---|
| chance baseline | **+0.018** | what a *broken* pipeline looks like |
| policy self-consistency | **+0.625** | the ceiling any comparison can reach |
| agreement with ground truth | **+0.528** | **84% of ceiling, 29× chance** |
| over the 8 executed steps | +0.589 | near-term is what actually gets commanded |
| on fast segments (>0.5 rad/s) | **+0.959** | near-perfect where motion is decisive |
| magnitude ratio (per sample) | 0.709 | units and normalization correct |
| gripper channel MAE | 0.0039 | gripper scaling correct |

Verdict logic:

| result | meaning |
|---|---|
| agreement < max(0.15, 5×chance) | **do not evaluate** — pipeline, checkpoint or prompt is wrong |
| ≥ 70% of ceiling | pipeline verified; failures are the policy's |
| below that | confirm served checkpoint and prompt |
| magnitude ratio outside 0.5–2.0 | normalization or checkpoint mismatch |

The magnitude ratio is computed **per sample**, not from the mean of samples —
eval commands one sampled chunk, and averaging stochastic samples would shrink
the norm and understate what is actually sent to the robot.

A ratio of ~0.71 says the policy commands about 71% of the demonstrated velocity
magnitude. That is regression-to-the-mean, normal for behaviour cloning, and is
a **policy property, not a rig fault** — expect it to move somewhat more
deliberately than the demonstrations.

Probe frames are deliberately drawn from **moving** segments; roughly half a
bracket episode is near-stationary and zero-action chunks would make the metric
meaningless.

Check D reads the robot's **live** `joint_velocity_limit` and fails if
`relative_dynamics_factor × limit < velocity caps` — i.e. if franky would
silently throttle the policy.

---

## 3. Run a trial

```bash
python model-running/run_bracket_policy_eval.py \
  --trial_id E01_left_far \
  --checkpoint_label 6000
```

Sequence: guarded segmented reset to the fixed start pose → Hand opens → you
place the bracket → policy runs closed-loop → you approve the single grasp
(`CLOSE`) and the single release (`RELEASE`) → you score `PASS`/`FAIL`.

`Ctrl+C` calls `robot.stop()` at any point.

Per-chunk console line:
```
[012/120] lat=0.148s tcp=[0.51 -0.12 0.18] peak=0.83 exec=0.97 grip=0.00..0.00 hold=False
```
`exec` is achieved ÷ commanded joint displacement. Sustained values below 0.70
print `<<THROTTLED` and are flagged in the result JSON — that means the arm did
not follow the commanded velocity and the trial should not be counted against
the policy.

---

## 4. Suggested protocol

The task is **vary-the-pick / fixed-the-place**: the mount point is effectively
constant (15 mm spread) while the bracket pickup varies enormously (859 mm in y).
So vary the **bracket**, never the mount.

- Pick **6–8 held-out bracket positions** inside the trained envelope
  (x ≈ 0.25–0.48, y ≈ −0.47–0.39 m), including positions *between* trained ones.
- **3 trials each**, same positions for every checkpoint.
- Evaluate **4000, 6000, 7999** — the loss curve had flattened well before the
  end, so the final checkpoint is the most likely to have memorised the 50
  training positions.
- Score `PASS` only if the bracket ends seated on the mount pins.
- Do not tune anything between checkpoints; identical positions and order.

Results land in `eval_logs/bracket_eval_<trial>_<stamp>.result.json`.

---

## 5. Verified interfaces (why this rig is trustworthy)

Each of these was checked against source, not assumed:

| item | verification |
|---|---|
| image keys | `DroidInputs` reads **only** `exterior_image_1_left` + `wrist_image_left`; the third slot is always `zeros_like`, `mask=False`. Our converter's duplicated `exterior_image_2_left` is never read — no train/eval asymmetry. |
| colourspace | converter `cvtColor(BGR2RGB)`; client `[..., ::-1]` — both RGB |
| resize geometry | both paths call the **same** `image_tools.resize_with_pad`; 640×480 and 320×240 are both 4:3, so letterboxing is identical |
| action units | 7 × joint velocity (rad/s, trained on libfranka `dq_d`) + gripper closedness; returned **unnormalized**, so franky is commanded directly in rad/s |
| state layout | `concat(joint_position[7], gripper_position[1])`, padded to 32 server-side |
| control rate | 15 Hz, execute 8 of 16 — matches OpenPI's DROID convention |
| prompt | exact `BRACKET_TASK_PROMPT` string the dataset was written with |

### Cube defaults that would have broken this eval

The cube runner's defaults are wrong here in ways that look like policy failure:

| parameter | cube default | correct for bracket | consequence if left alone |
|---|---|---|---|
| `joint_velocity_caps` | **0.10** rad/s | **1.9** | clips **24.1%** of timesteps; peak throttled 16× |
| `relative_dynamics_factor` | **0.085** | **0.9** | franky caps motion near 0.2 rad/s, ~8× too slow |
| `prompt` | cube text | bracket text | out-of-distribution language conditioning |
| `close_threshold` | 0.50 | 0.35 | only +0.125 above the 0.625 closed mode |
| workspace bounds | cube envelope | measured bracket envelope | spurious guard aborts |

---

## 6. Known, quantified artifact

Inference is synchronous: the arm executes an 8-step chunk (~533 ms), then
pauses during the next `infer()`. The runner reports `motion_duty_cycle` in
the result JSON.

This is inherent to chunked inference and is exactly what OpenPI's reference
DROID client does. `franky` does support `move(..., asynchronous=True)`, which
would remove the stall, but it was deliberately **not** enabled: the safety
guards would then be evaluated against stale joint state. If the preflight
reports inference latency approaching 533 ms, reduce it server-side rather than
overlapping execution.

Images are resized to 224×224 **on the laptop** before transmission, which is
OpenPI's own recommended latency measure.
