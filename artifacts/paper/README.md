# SDNC Paper Artifact

Reproducibility package for **"The Self-Differentiating Neural Computer:
Computable Transformers via Analytical Weight Construction"**
(tsotchke 2026).

## Historical Pinned Artifact

- **Repository:** this repository (`tsotchke/eshkol`)
- **Commit SHA:** `8235d9987d70086e6e62083d120f3cf51fac9e48`
- **Git tag:**  `v1.1.13-accelerate-166-g8235d99`
- **Paper PDF:** see Noesis companion repo `docs/paper-computable-transformers/`
- **Build environment:** LLVM 21, CMake 3.14+, C17/C++20 (see `docker/Dockerfile.paper`)

To reproduce the historical pinned artifact:

```bash
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
git checkout 8235d9987d70086e6e62083d120f3cf51fac9e48
docker build -f docker/Dockerfile.paper -t eshkol-paper .
docker run --rm -v "$(pwd):/work" -w /work eshkol-paper \
    scripts/paper/run_paper_suite.sh
```

## One-command rerun

```bash
scripts/paper/run_paper_suite.sh
```

Expected wall time on a 2023 M2 Max: under five minutes.

Output: populated `artifacts/paper/outputs/` with:
- `weights.qlmw` — regenerated weight matrices (SHA-256 checksum below)
- `vm-traces.jsonl` — per-step state traces from the reference C interpreter
- `transformer-traces.jsonl` — per-step state traces from the compiled transformer
- `comparison-report.json` — fieldwise agreement report (123/123 expected)
- `opcode-coverage.json` — 82/83 canonical opcodes weight-implemented in the
  exercised bounded suite; `OP_NATIVE_CALL` remains the explicit external
  boundary
- `tables/*.tex` — regenerated LaTeX for every table in the paper

## Expected Checksums

```
SHA-256  weights.qlmw              381599e7a5607b4047ede0d6c8e6d270cb81dbdebfdb0bf0c0eba38758aa3f0c
SHA-256  vm-traces.jsonl           4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  transformer-traces.jsonl  4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  comparison-report.json    80aa6fed4db40bca521217ae8777677173fe7eeb239baa69847111e7ac674105
SHA-256  opcode-coverage.json      152a4bacc483d8985abeb08bc0d44112144f536ed663274bc7b1eeccbdd2dfe4
```

The current regeneration records these hashes; every subsequent regeneration
from the same source tree on an IEEE 754 float32 platform should produce
bit-identical outputs. Platform divergence is a bug; file an issue with
the platform details.

### What the agreement metrics mean

The artifact contract is **bit-identical agreement at every step**
between the reference C interpreter and the matrix forward pass. The
`comparison-report.json` carries two metrics:

  - `output_agreeing_programs` — programs whose first PRINT result
    matches.
  - `fully_agreeing_programs` — programs whose entire per-step state
    vector matches bitwise (PC, SP, TOS, SOS, registers, memory, tape,
    flags).

**Both metrics must equal `total_programs` (currently 123/123).** If a regression
ever introduces a divergence, it must be fixed — not documented as
acceptable drift.

The matrix path is a `W·x + b` SQUARE-FFN transformer, so cross
products are computed via the polarisation identity
`a·b = ½·(a+b)² − ½·a² − ½·b²` (Layer 1). For bit-identity, the
reference VM (`ad_backward_step`) uses the same polarisation arithmetic
where it would otherwise compute `grad · saved` directly — direct
multiplication and polarisation are mathematically equal but differ by
1–13 ULPs in float32 due to operation order, and the matrix path has
no architectural way to compute direct multiplication.

## Script breakdown

| Script | Purpose |
|--------|---------|
| `scripts/paper/run_paper_suite.sh`        | Top-level runner; calls the others in order |
| `scripts/paper/export_weights.sh`         | Regenerates QLMW weight matrices from the ISA spec |
| `scripts/paper/dump_vm_trace.sh`          | Runs the reference C VM on the traced program suite; emits per-step traces |
| `scripts/paper/dump_transformer_trace.sh` | Runs the compiled transformer on the same traced programs; emits per-step traces |
| `scripts/paper/compare_traces.py`         | Fieldwise exact comparison; emits `comparison-report.json` |
| `scripts/paper/gen_paper_tables.py`       | Regenerates LaTeX tables from measurement outputs |

## What the paper claims the artifact proves

1. The weight generation is deterministic — same ISA, same weights, bit-identical across platforms.
2. Three-way agreement: reference C VM = simulated transformer (C code mirroring the weight-implemented opcodes) = matrix forward pass (actual W @ x + b matmul). The current repository artifact verifies 126/126 inline programs and 123/123 traced programs.
3. The strict bounded-artifact scope covers 82/83 canonical opcodes in weights. `OP_NATIVE_CALL` remains the explicit external boundary for host services and high-level runtime calls.
4. The AD tape (8 nodes in the state vector) correctly computes gradients on the reported toy scalar/vector programs; gradient-check vs. dual numbers within 1e-6 relative error.

## What this artifact does NOT claim

- No claim that `OP_NATIVE_CALL` host services are themselves encoded as closed-form VM weights.
- No claim on float64 or other precisions.
- No claim on state-vector capacities beyond $D=256$.
- No benchmark claims against frontier LLMs (see the Noesis companion paper for empirical comparisons).

## Current Implementation Status

The scripts in `scripts/paper/` run the current SDNC verification path end to
end. `scripts/paper/run_paper_suite.sh` builds `weight_matrices`, exports the
QLMW artifact, dumps both reference-VM and matrix-forward traces, compares them
fieldwise, and regenerates the paper tables.

The current bounded artifact verifies 126/126 inline programs and 123/123 traced
programs. Opcode coverage is 82 weight-implemented / 0 native-delegated / 0
transformer-native-assisted opcodes in the exercised suite. `OP_NATIVE_CALL`
remains the intentional host-service boundary.

## Issues and questions

File issues at https://github.com/tsotchke/eshkol/issues with tag
`artifact-repro`.
