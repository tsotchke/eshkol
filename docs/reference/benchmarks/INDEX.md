# Public Benchmarks — Reference

Eshkol's performance and rigor claims were, until v1.3.5's "benchmarks wave 1", entirely
self-reported. This page indexes the fix: a public, reproducible benchmark suite that
measures only the axes where Eshkol claims something distinctive, that a stranger can run
with one command and get numbers on their own hardware to compare against the reference
numbers published here.

The suite lives at [`bench/`](../../../bench/README.md) — that README is the full
reproduction guide (build flags, noise-control methodology, result JSON schema, and an
explicit list of what is deliberately NOT benchmarked and why). This page is the doc-site
pointer into it, in the same fan-out shape as the other `docs/reference/*/INDEX.md` pages.

## Run it

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DESHKOL_QUANTUM_ENABLED=ON -DESHKOL_BUILD_TESTS=ON -DESHKOL_BUILD_AGENT_FFI=ON
cmake --build build --parallel
bench/run_public_benchmarks.sh --build-dir build
```

Writes `bench/results/<timestamp>/results.json` (machine-readable) and `results.md`
(human-readable), both derived from the same measurement run — see
[`bench/README.md#result-format`](../../../bench/README.md#result-format).

## The four axes (wave 1)

| # | Axis | Substantiates | Script |
|---|------|----------------|--------|
| 1 | Exact-AD cost curves | `derivative-n` is O(k²) in derivative order (truncated-Taylor recurrence, not 2ᵏ nested duals); exact-at-exact-points returns exact rationals | [`bench/axes/01_exact_ad.sh`](../../../bench/axes/01_exact_ad.sh) |
| 2 | Ozaki-II CRT exact f64 GEMM | Accuracy vs an exact-rational reference AND throughput, against the vendor BLAS baseline (Accelerate/AMX) on the SAME machine | [`bench/axes/02_ozaki_gemm.sh`](../../../bench/axes/02_ozaki_gemm.sh) |
| 3 | Flat-RSS under resident load | The native resident-loop and bytecode-VM `with-region` evacuator (PR #461) both hold flat peak RSS as a sweep of tick/iteration counts grows | [`bench/axes/03_flat_rss.sh`](../../../bench/axes/03_flat_rss.sh) |
| 4 | Differentiable quantum kernels | H2 vibrational frequency from an exact 2nd-order AD Hessian; VQE H2 energy + native adjoint gradient vs Moonlab's exact oracle | [`bench/axes/04_quantum_kernels.sh`](../../../bench/axes/04_quantum_kernels.sh) |

Each script's header comment names the exact claim it tests, the fixture it uses (existing
shipped tests/examples where one already existed — `examples/h2_vibrational.esk`,
`tests/quantum/vqe_test.esk` — otherwise a small generated `.esk` sweep modeled on the
closest existing test in this repo), and the methodology behind the numbers.

## What this is not

This suite does not compare Eshkol against XLA/PyTorch/JAX on their own turf (dense
float32 training throughput, ResNet-class model benchmarks, or a bare GEMM speed contest
against XLA's kernels). See
[`bench/README.md`'s "What this suite does NOT benchmark, and why"](../../../bench/README.md#what-this-suite-does-not-benchmark-and-why)
for the explicit, ratified list. It is a reproducibility artifact for claims this project
actually makes, not a competitive benchmark suite.

## CI

A `bench-smoke (not a measurement)` job in `.github/workflows/ci.yml` runs
`bench/run_public_benchmarks.sh --smoke` on every PR touching `bench/**` — a fast
harness-correctness check, never a performance gate. See `bench/README.md#ci`.

## See also

- [`docs/design/adr/0007-performance-pgo-wpo.md`](../../design/adr/0007-performance-pgo-wpo.md) —
  the "Result protocol" / "Noise controls" sections this suite's methodology is a public
  subset of.
- [`docs/breakdown/BENCHMARKING.md`](../../breakdown/BENCHMARKING.md) — the pre-existing
  `benchmarks/*.esk` micro-benchmarks (matmul, activations, convolution, GPU vs CPU) this
  suite does not replace; those remain useful development-time perf probes, this suite is
  the public/reproducible claim-substantiation layer.
- [`docs/reference/ad/architecture.md`](../ad/architecture.md) — the AD architecture axis 1
  measures against.
- [`docs/breakdown/GPU_ACCELERATION.md`](../../breakdown/GPU_ACCELERATION.md) — the
  Ozaki-II CRT GEMM design axis 2 measures against.
