# Eshkol public benchmarks — wave 1

Every performance and rigor claim Eshkol has published, up to this point, has been
self-reported: numbers in `CHANGELOG.md`, `RELEASE_NOTES.md`, and design docs, with no
way for a stranger to reproduce them. This is the fix. It is a benchmark suite that a
person who has never touched this repository can clone, build, run with **one command**,
and get a machine-readable result plus a human-readable table — on their own hardware,
compared against the numbers we publish here from ours.

This is wave 1: four axes, chosen deliberately. See ["What this suite does NOT benchmark,
and why"](#what-this-suite-does-not-benchmark-and-why) before assuming a gap is an
oversight.

## Run it

```sh
# 1. Build (see "Build flags" below for exactly what wave 1 needs)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DESHKOL_QUANTUM_ENABLED=ON -DESHKOL_BUILD_TESTS=ON -DESHKOL_BUILD_AGENT_FFI=ON
cmake --build build --parallel

# 2. Run every axis
bench/run_public_benchmarks.sh --build-dir build
```

That's the whole reproduction recipe. It writes `bench/results/<UTC timestamp>/results.json`
and `results.md`. Every result — every field in the JSON — is produced fresh by this run;
nothing is copied from a prior invocation or from this document.

Useful flags:

- `--smoke` — a fast harness-correctness subset (seconds, not minutes). This is what CI
  runs; it proves the harness still works, it is **not a measurement run** and its numbers
  must never be quoted as performance data (tiny sweeps, a couple of noisy samples — see
  the `smoke_mode: true` field in its `results.json`).
- `--only 1,3` — run a subset of axes (1=exact-AD, 2=Ozaki-II GEMM, 3=flat-RSS,
  4=quantum kernels).
- `--out-dir DIR` — where results land (default `bench/results/<timestamp>/`).
- `--work-root DIR` — scratch space for generated `.esk` sources and compiled binaries.
  Defaults to a `mktemp` directory under `TMPDIR`, matching every other harness in this
  repo (see `benchmarks/gpu_matmul_bench.sh`); every axis script disk-caps its own scratch
  dir (`BENCH_DISK_CAP_MB`, default 2048MB per axis) and deletes it on exit unless
  `--keep-work` is passed.

### Build flags

| Axis | Needs |
|---|---|
| 1. exact-AD cost curves | nothing beyond the ordinary build |
| 2. Ozaki-II GEMM | a Metal (macOS) or CUDA (Linux) GPU; `-DESHKOL_GPU_ENABLED=ON` (default ON) |
| 3. flat-RSS | `-DESHKOL_BUILD_TESTS=ON` for the VM curve (`eshkol-vm-standalone-test`); native curve needs nothing extra |
| 4. quantum kernels | H2 Hessian needs nothing extra; VQE needs `-DESHKOL_QUANTUM_ENABLED=ON -DESHKOL_BUILD_AGENT_FFI=ON` (pulls in Moonlab) |

Missing a build flag never fails the run — the affected row is marked unavailable and
says why (`grep -q 'unavailable' results.md` finds every such row). This suite reports
what it could measure, honestly, rather than requiring one maximal build or silently
skipping without saying so.

## The reference run published in this repo

`bench/reference-run/` holds the actual `results.json` / `results.md` this suite produced
on the machine below — this is not an illustration, it is the numbers this PR's commit
message and description quote. Read them there, not copied into this file, so they never
drift out of sync with the JSON.

- **Machine:** Mac Studio (Mac14,14), Apple M2 Ultra, 24 cores (16P+8E), 192GB unified
  memory, macOS 15.1.
- **Compiler/toolchain:** Homebrew LLVM/Clang 21.1.7, `-DCMAKE_BUILD_TYPE=Release`.
- **BLAS:** Apple Accelerate (vecLib/AMX) — the vendor baseline axis 2 compares against.
- **GPU:** Apple M2 Ultra integrated GPU via Metal — the device axis 2's Ozaki-II rows
  measure.
- Full detail (exact compiler version string, git SHA, load average at capture time, and
  every other field) is in `bench/reference-run/results.json`'s `environment` object —
  that JSON is the source of truth, this bullet list is a human-readable pointer to it.

**Never publish a number you did not measure.** If you rerun this suite on different
hardware, your `results.json`'s `environment` block will say so — do not edit ours, and do
not represent your numbers as ours or vice versa.

### Two honest caveats about THIS reference run

Both surfaced BY running the suite, not hidden after the fact — which is the entire point
of building a real harness instead of hand-picking numbers:

1. **This machine was under heavy, unrelated concurrent load while the reference run was
   captured** (`environment.load_average_at_capture` in the JSON: load averages around
   150-170 on a 24-core machine — other work was running on it at the time). Axis 1's
   per-k/per-d timings show visible non-monotonicity from this
   (`bench/reference-run/axis1.json`), and the fitted power-law exponents are noisier than
   a quiescent-machine run would produce. The qualitative claim (exact-rational cost grows
   much faster than float cost as order/dimension increase) is unaffected; do not read the
   fitted exponents as precise without rerunning on a quiet machine. This is exactly the
   noise-control gap flagged above ("no dedicated quiescent runner, no bootstrap CI") —
   the JSON says so rather than presenting a clean number a shared laptop cannot honestly
   produce.
2. **Axis 3's native curve is flat at the originally-published scale but NOT flat when
   swept further.** At 100,000 ticks — the exact point RELEASE_NOTES.md's "flat at 34MB"
   claim was made at — this run measures 36MB, consistent with that claim. Extending the
   sweep to 400,000 ticks (4x further than anything previously published) shows renewed
   growth: 11MB at 10k ticks up to 119MB at 400k ticks, a 10.8x memory increase over a 40x
   work increase — outside this suite's own 1.5x flatness allowance
   (`bench/reference-run/axis3.json`, `flatness.flat_within_allowance: false`). The VM
   curve (with-region evacuator) stays flat across the same relative sweep range (26MB to
   28MB, 1000 to 64000 iterations). This is a genuine finding this suite's curve surfaced
   that the original single-point measurement could not have — it is reported here, not
   fixed here (fixing it is separate compiler work, out of scope for a benchmark-harness
   PR), and is worth a v1.3.5 follow-up investigation into what grows in the native
   resident-loop path beyond ~100-200k ticks (candidates: hash-table internal resize
   policy counting insertions rather than distinct keys, or per-region backing-chunk
   allocation not being reclaimed across chunk boundaries — neither confirmed, both
   plausible, worth investigating rather than guessing further here).

## Result format

One `results.json` per run:

```jsonc
{
  "schema": "eshkol-public-benchmarks-v1",
  "smoke_mode": false,
  "started_at": "...", "finished_at": "...",
  "environment": { /* OS, CPU/GPU model, compiler/LLVM/BLAS versions, git SHA,
                      build flags, load average at capture — see
                      bench/lib/fingerprint.sh */ },
  "axes": {
    "1_exact_ad_cost_curves": { /* ... */ },
    "2_ozaki_ii_gemm": { /* ... */ },
    "3_flat_rss_under_resident_load": { /* ... */ },
    "4_differentiable_quantum_kernels": { /* ... */ }
  }
}
```

`results.md` is generated FROM `results.json` (never hand-edited) and is the
human-readable rendering of the same data — every number in it traces back to a field in
the JSON. See `bench/combine_results.py`.

### On `scripts/lib/build_fingerprint.sh`

At the time this suite was written, PR #465 (which adds `scripts/lib/build_fingerprint.sh`)
had not merged. That file solves a narrower, complementary problem — proving a test
harness's evidence talks about the exact binary still on disk (a sha256/mtime staleness
check), not describing the machine/toolchain that produced it. `bench/lib/fingerprint.sh`
solves the provenance problem this suite needs: a full environment description embedded
once per `results.json`. See the header comment in `bench/lib/fingerprint.sh` for the
planned integration once #465 lands.

## Noise controls (ADR-0007)

The methodology here is a deliberately smaller subset of
[`docs/design/adr/0007-performance-pgo-wpo.md`](../docs/design/adr/0007-performance-pgo-wpo.md)'s
"Result protocol" and "Noise controls" sections — that ADR designs a full PGO-training
release gate with dedicated quiescent runners and bootstrap confidence intervals; wave 1 is
a public reproducibility suite anyone can run on a shared laptop, so it borrows the parts
of that discipline that travel:

- **Every raw sample is kept**, not collapsed to a mean before it reaches the JSON
  (`raw_ns_samples` / `ns_samples` arrays throughout).
- **Warmup is explicit and separated from measurement** — axis 1's calibration loop and
  axis 2's throughput fixture both run an untimed warmup call before any timed sample
  (JIT/shader-pipeline compilation must not contaminate a "cost of one call" number).
- **Iteration counts are chosen so each sample takes a floor amount of wall time**
  (axis 1's `TARGET_NS` calibration), rather than a fixed iteration count that means
  different things on different hardware.
- **Thread counts are pinned, not ambient** — `bench_pin_single_thread` in
  `bench/lib/common.sh` sets `OMP_NUM_THREADS=VECLIB_MAXIMUM_THREADS=MKL_NUM_THREADS=
  OPENBLAS_NUM_THREADS=1` and pins `ESHKOL_JIT_CACHE`; every axis is a single-call-latency
  or single-process measurement, not a throughput-under-parallelism one (wave 1 has no
  scaling lane — a fair multi-threaded comparison is future work, not silently assumed).
- **Every result records its own environment**, including load average at capture time —
  this repo's benchmarks were measured on a shared development machine, not a dedicated
  quiescent runner, and the `results.json` says so rather than pretending otherwise.
- **What this suite does NOT do**, honestly: it does not pin CPU affinity (no portable
  `taskset` equivalent is used — macOS has no simple userspace affinity API), does not
  isolate the worker from unrelated jobs, and does not compute bootstrap confidence
  intervals or A/B/B/A blocks. Those are real gaps against ADR-0007's full discipline,
  appropriate for a release-gate CI runner and not yet built for this public suite. A
  future wave should close them before this suite is used as a regression gate rather
  than a reproducibility artifact.

## What this suite does NOT benchmark, and why

Per the maintainer-ratified strategy: **benchmark ONLY on the axes where Eshkol claims
superiority.** This is a reproducibility suite, not a competition entry, and adding
comparisons on someone else's turf would undermine the point — it would look like trying
to win a fight nobody picked with us.

Explicitly ruled out for wave 1:

- **ResNet/ImageNet-style large model training throughput.** Eshkol does not claim to be
  a training-throughput competitor to XLA/PyTorch/JAX on dense-float32 vision workloads,
  and has no dedicated data-loading/mixed-precision/distributed-training stack to make
  that a fair fight. Benchmarking it would measure our absence of infrastructure, not our
  actual claims.
- **Large dense float64 GEMM throughput as a bare speed contest against XLA.** Axis 2
  benchmarks Ozaki-II GEMM against the **vendor BLAS baseline on the same machine**
  (Accelerate/AMX), which is the comparison our own accuracy/throughput claim is about.
  It does not benchmark against XLA's GEMM, which is a different tool solving a different
  problem (dense ML training throughput, not CRT-exact f64 reconstruction) — putting them
  side by side would invite exactly the apples-to-oranges reading this suite exists to
  avoid.
- **The CUDA Ozaki-II INT8 tensor-core numbers already in `CHANGELOG.md`** (RTX 3090,
  RTX PRO 6000 Blackwell) are **not re-measured here** — this reference machine has no
  NVIDIA GPU. `bench/axes/02_ozaki_gemm.sh`'s JSON output cites them explicitly as prior
  published measurements on that other, named hardware; it never re-quotes them as
  something this suite measured on this machine.
- **General-purpose language/interpreter benchmark suites** (e.g. the classic
  fibonacci/nqueens/binary-trees shootout set). Eshkol's claims are about AD exactness,
  CRT-exact GEMM, resident-memory flatness, and differentiable quantum kernels — not raw
  interpreter throughput against Python/Node/Ruby, which is not a claim this project makes.

If a future wave wants to add a comparison outside these four axes, it should go through
the same ratification the original four did, not be added quietly.

## Repository layout

```
bench/
  run_public_benchmarks.sh    the one entry point
  combine_results.py          merges fingerprint + 4 axis JSONs -> results.json/.md
  lib/
    common.sh                 logging, disk cap, RSS measurement, JSON helpers
    fingerprint.sh             environment/build provenance capture
  axes/
    01_exact_ad.sh / _reduce.py
    02_ozaki_gemm.sh / _reduce.py
    03_flat_rss.sh / _reduce.py
    04_quantum_kernels.sh / _reduce.py
  reference-run/               this repo's own measured reference numbers (committed)
  results/                     default --out-dir for ad-hoc runs (gitignored)
  generate_large_single_file.py     synthetic large-single-file .esk generator
  large_single_file_compile_bench.sh  continuous AOT compile-time ceiling gate
                                       for that shape — see its header comment;
                                       NOT part of the wave-1 axis suite above
                                       (it measures a compile-time cost, not a
                                       runtime performance claim), wired into
                                       .github/workflows/adversarial-nightly.yml
```

See `docs/reference/benchmarks/INDEX.md` for the doc-site version of this page, and each
`bench/axes/*.sh` file's header comment for the exact claim it substantiates and the
methodology behind it.

## CI

`.github/workflows/ci.yml`'s `bench-smoke` job runs `bench/run_public_benchmarks.sh
--smoke` on every PR that touches `bench/**` — it is a harness-correctness check (does the
suite still run, produce valid JSON, exit cleanly), not a performance gate. It is
explicitly named `bench-smoke (not a measurement)` in the CI job list and must never be
added to branch-protection's required list as if it verified a performance claim; a
shared, non-dedicated CI runner cannot produce a trustworthy timing number, and this job
does not pretend to.
