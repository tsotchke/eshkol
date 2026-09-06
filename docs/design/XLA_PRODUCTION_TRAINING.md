# S9: checkpoint authority, preemption survival, and inference from a checkpoint

Status: single-device checkpoint authority, preemption survival, and
corrupt-checkpoint refusal implemented and measured (Stage 9 of the
XLA-to-TPU program, criterion `xla_tpu_production_ready` in
`.icc/completion-oracles.yaml`). The multi-device checkpoint row is
**unmeasured**: it depends on `xla_multidevice_step` (S8, a sibling lane not
yet landed on this branch when this stage was built), and a sharded
checkpoint's shard-to-host reassembly has no single-device analogue to
substitute. The gate (`scripts/run_xla_gate.sh --production`) is therefore a
**forced FAIL** until S8 lands and that row is measured — see section 5.

## 0. What "checkpoint authority" means here

The brief's phrase, taken literally: the checkpoint file, not the process, is
the source of truth for training state. A process that crashes, is
preempted, or is deliberately killed loses nothing that was already
checkpointed, and a resumed process reproduces the EXACT continuation an
uninterrupted process would have produced — not an approximately-similar
one. Section 3 states what "exact" covers and what it does not.

## 1. What already existed, and why nothing about the ESKM binary format changed

`lib/core/model_io.cpp` already had an ESKM container (magic, format version,
a named-tensor-record list, trailing CRC-32) used by `(model-save)` /
`(model-load)`, and `lib/ml/mixed_curvature_step.h` (S6) already had a full
description of the mixed-curvature training step's state:

- four parameter tensors (`w`, `p_hyp`, `p_sph`, `p_euc`)
- eight optimizer-moment tensors (`m_*`/`v_*`, one pair per parameter)
- one shared step count
- one curvature hyperparameter
- **the training batch is drawn once, deterministically, from one seed**
  (`eshkol_mixed_curvature_init`), and no step consumes any further
  randomness — `docs/design/XLA_TRAINING_STEP.md` section 2 states this, and
  `tests/xla/training_step_parity_test.cpp` relies on it (the "free" model
  is never re-seeded).

That last point is what makes checkpointing this training state tractable
without extending the ESKM format: every piece of state that determines the
trajectory already has a natural representation as a named f64 tensor
(including the seed, stored as a bit-pattern in a length-1 tensor), and the
container was not missing a field for this job. So `lib/ml/training_checkpoint.cpp`
writes the training state as one more ESKM tensor list, through the exact
same wire format `model_io.cpp` reads — a training checkpoint loads through
the generic `(model-load path)` builtin like any other model checkpoint.

## 2. The two files a checkpoint is

`eshkol_training_checkpoint_save(path, meta, params, moments)` writes two
files, both published atomically (see section 4):

- **`<path>`** — the ESKM payload: 15 named tensors (`w`, `p_hyp`, `p_sph`,
  `p_euc`, the eight moments, `step`, `curvature`, `seed`), magic `ESKM`,
  format version 1, trailing CRC-32. Unmodified format.
- **`<path>.manifest`** — a small text sidecar: `shape_n`/`shape_d`/`shape_c`
  (the shape signature), `dtype_policy` (`f32` or `bf16-mixed` — which S7
  numeric policy produced the values; the ESKM payload itself is always f64,
  this records what generated it), `curvature`, `step`, `seed`, and
  `content_crc32` — the SAME CRC-32 the payload's own trailing footer
  carries, restated here so a resume can validate a checkpoint from the
  small manifest file alone before ever opening the (potentially large)
  payload, and so a manifest/payload disagreement alone is grounds for
  refusal without parsing the body.

Plain `key=value` text rather than JSON, for the same reason the ESKM
container is a fixed binary layout rather than a generic format: no
dependency beyond `snprintf`/`sscanf`.

## 3. Validation order, and what "refuse" means

`eshkol_training_checkpoint_load()` / `_is_valid()` check, in order, any of
which is a refusal (return `false`, no output written, existing in-memory
state untouched):

1. the manifest parses and has all 8 fields;
2. the manifest's shape matches the caller's expected shape (a resuming
   process's own compiled-in shape, not something the checkpoint can assert
   over the caller);
3. the payload's own trailing CRC-32 matches its bytes (the ESKM container's
   existing integrity check, run independently of the manifest);
4. the payload's CRC-32 equals the manifest's `content_crc32` (the two files
   agree with each other, not just each with itself);
5. every expected named tensor is present with the expected element count;
6. the payload's `step` tensor equals the manifest's `step` field, and the
   payload's `seed` tensor equals the manifest's `seed` field (belt-and-braces
   against a manifest/payload pairing that split across two different saves).

A refusal is silent to the caller only in the sense that it returns `false`
rather than crashing or partially loading; `training_checkpoint_driver`
(section 5) logs which file it refused and which one it fell back to.

## 4. Atomicity

Both files go through `lib/core/model_io_atomic.h` (Gabriel Kahen's
atomic-save helper, PR #600, reused as-is rather than reimplemented): a
uniquely and exclusively created temporary file in the destination
directory, published by `rename()` only after write, flush and close all
succeed; on any failure the temporary file is removed and the existing
destination is untouched. `model_io.cpp`'s and `vm_model_io.c`'s own
`(model-save)`/`(tensor-save)` writers were changed to go through the same
helper in this stage (they previously used a bare `fopen(path, "wb")`), so
every ESKM writer in the codebase — not just the training checkpoint — now
publishes atomically. SIGHUP/SIGINT/SIGQUIT/SIGTERM are blocked for the
duration of the write and unblocked immediately after commit or abort, so a
signal cannot interrupt a write mid-flight; this does not, and is not
claimed to, protect against SIGKILL or a power loss (no fsync of the file or
its parent directory — same limitation PR #600 states for the general case).

## 5. Preemption survival: `training_checkpoint_driver`

`tests/xla/training_checkpoint_driver.cpp` is a standalone process (not a
library call) taking `<ckpt_dir> <total_steps> <checkpoint_every> <seed>`:

- On start, lists `ckpt_dir` for files named `ckpt.<step>.eskm`, newest
  step first, and tries to load each in turn until one validates (section
  3); a refused file is logged and the next-newest is tried. If none
  validate, it trains from step 0.
- Installs a `SIGTERM` handler that finishes the in-flight step, checkpoints,
  and exits 0 — the graceful "PJRT device loss" / preemption-notice path.
- Checkpoints every `checkpoint_every` steps to `ckpt.<step>.eskm`.
- **Cannot** do anything about `SIGKILL` — by construction, since `SIGKILL`
  is not catchable. That is the actual preemption-survival property under
  test: correctness after an ungraceful kill has to come from what was
  already durably on disk, not from the process reacting.

### Measured rows (`tests/xla/training_checkpoint_test.cpp`, `scripts/run_xla_gate.sh --production`)

Run on this machine (arm64 macOS, host CPU, single device) as part of
building this stage:

| row | what it does | result |
|---|---|---|
| `round_trip_parity` | save mid-training state, load into a fresh struct, `memcmp`-equivalent compare (every param/moment/step/seed/curvature) | PASS — byte-identical |
| `trajectory_parity` | K=5 steps, checkpoint, K=5 more vs. an uninterrupted K=10 run from the same seed | PASS — bit-identical loss sequence from the checkpoint onward |
| `kill_and_resume` | fork/exec the real driver binary, `SIGKILL` it mid-run (~400ms into a multi-second run, a real OS kill), relaunch against the same checkpoint directory, diff the final step/loss against an uninterrupted reference run | PASS — identical final step and loss |
| `corrupt_refusal` | flip one byte in the newest checkpoint's payload, require `eshkol_training_checkpoint_is_valid()` to go from true to false, then relaunch the driver and require it to name the refused file and fall back to the previous checkpoint | PASS — refused, fell back, resumed, finished |

These are FAIL-then-PASS by construction, not merely asserted: `kill_and_resume`
observes the `SIGKILL` was actually delivered (`WIFSIGNALED` + `SIGTERM`→`SIGKILL`
check) and that the relaunched process exited 0; `corrupt_refusal` observes
validity flip from true to false across the byte-flip and greps the relaunch's
own stderr for the refusal and the fallback, rather than assuming either.

`training_checkpoint_test` prints `SUMMARY: rows_passed=N rows_failed=M`,
which `scripts/run_xla_gate.sh`'s `stage_production` parses the same way
every other XLA-program stage's test binary is parsed.

## 6. Inference from a checkpoint

Not built in this stage. `eshkol_mixed_curvature_loss()` (S6, already in
`lib/ml/mixed_curvature_step.h`) is the forward-only entry point a host
inference path would call after `eshkol_training_checkpoint_load()`; wiring
that to the device (PJRT) forward-only program and an executable-cache-hit
check on a second batch needs `training_step_lowering.cpp`'s device
executor and TPU node time this stage did not have. Left as the next build
item under this criterion rather than claimed done.

## 7. What is bit-exact across a restore, and what is not

Bit-exact: every parameter and moment value (raw f64, byte-for-byte through
the ESKM payload), the step count, the seed, and the curvature. The
`round_trip_parity` and `trajectory_parity` rows measure exactly this.

Not claimed bit-exact: reproducing the SAME trajectory on a DIFFERENT
process whose libm differs from the one that generated the checkpoint (a
different platform, a different libc). `eshkol_mixed_curvature_train_step`
performs the same floating-point operations in the same order every call
given the same inputs, so on one platform / one libm the post-restore
trajectory equals the uninterrupted one exactly; the golden-corpus drift
already on record for this program (1-3 ULP differences regenerating
`tests/qllm_oracle/golden/*.json` between Apple libm and glibc, noted in
`.swarm/runtime/CODEX_HANDOFF.md`) is evidence of the same class of gap and
is not re-litigated here — a resume on the SAME libm is what this stage's
measured rows exercise and claim.

## 8. Launching a training run on a TPU VM

No infrastructure identifiers (hostnames, zones, project ids, hardware
models) appear below or belong in this file; substitute the node's own
values when running this.

1. Build with the XLA gate's baseline + production stages:
   `scripts/run_xla_gate.sh --baseline --pjrt-cpu --training-step --production`
   (each stage requires the previous to have built; `--production` builds
   and runs `training_checkpoint_driver` / `training_checkpoint_test`).
2. Launch training: `training_checkpoint_driver <ckpt_dir> <total_steps>
   <checkpoint_every> <seed>`. Choose `checkpoint_every` against the node's
   preemption notice window — the driver's `SIGTERM` handler finishes the
   in-flight step and checkpoints before exiting, so `checkpoint_every` only
   bounds the SIGKILL-loss window (the steps since the last checkpoint that
   a preemption with no notice at all would lose), not the graceful path.
3. On restart after any exit (graceful or killed), invoke the driver again
   with the SAME `<ckpt_dir>`: it resumes from the newest valid checkpoint
   automatically (section 5), or trains from step 0 if none validate.
4. `total_steps` is the target step count, not a step delta — a resumed
   invocation with the same `total_steps` runs only the remaining steps.
