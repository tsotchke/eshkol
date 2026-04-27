# SDNC Paper Artifact

Reproducibility package for **"The Self-Differentiating Neural Computer:
Computable Transformers via Analytical Weight Construction"**
(tsotchke 2026).

## Pinned artifact

- **Repository:** this repository (`tsotchke/eshkol`)
- **Commit SHA:** `8235d9987d70086e6e62083d120f3cf51fac9e48`
- **Git tag:**  `v1.1.13-accelerate-166-g8235d99`
- **Paper PDF:** see Noesis companion repo `docs/paper-computable-transformers/`
- **Build environment:** LLVM 21, CMake 3.14+, C17/C++20 (see `docker/Dockerfile.paper`)

To reproduce against the pinned artifact:

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
- `comparison-report.json` — fieldwise agreement report (74/74 expected)
- `opcode-coverage.json` — 57/83 weight-implemented, 26/83 delegated
- `tables/*.tex` — regenerated LaTeX for every table in the paper

## Expected checksums (post-regeneration)

```
SHA-256  weights.qlmw                    c8525f133ee1de3c67b1b56bd948fda24db56a3c89aacf29a8c9e9c9dc046759
SHA-256  vm-traces.jsonl                 <fill at first regen>
SHA-256  transformer-traces.jsonl        <fill at first regen>
SHA-256  comparison-report.json          <fill at first regen>
```

The first regeneration at the pinned commit records these hashes; every
subsequent regeneration on an IEEE 754 float32 platform should produce
bit-identical outputs. Platform divergence is a bug; file an issue with
the platform details.

## Script breakdown

| Script | Purpose |
|--------|---------|
| `scripts/paper/run_paper_suite.sh`        | Top-level runner; calls the others in order |
| `scripts/paper/export_weights.sh`         | Regenerates QLMW v3 weight matrices from the ISA spec |
| `scripts/paper/dump_vm_trace.sh`          | Runs the reference C VM on the 74-program suite; emits per-step traces |
| `scripts/paper/dump_transformer_trace.sh` | Runs the compiled transformer on the same 74 programs; emits per-step traces |
| `scripts/paper/compare_traces.py`         | Fieldwise exact comparison; emits `comparison-report.json` |
| `scripts/paper/gen_paper_tables.py`       | Regenerates LaTeX tables from measurement outputs |

## What the paper claims the artifact proves

1. The weight generation is deterministic — same ISA, same weights, bit-identical across platforms.
2. Three-way agreement: reference C VM = simulated transformer (C code mirroring the weight-implemented opcodes) = matrix forward pass (actual W @ x + b matmul). 74/74 programs agree.
3. The scope of exactness is the 57 weight-implemented opcodes (37 base + 15 AD forward + 5 AD control). The remaining 26 opcodes delegate to C runtime and are explicitly labeled in the opcode-coverage report.
4. The AD tape (8 nodes in the state vector) correctly computes gradients on the reported toy scalar/vector programs; gradient-check vs. dual numbers within 1e-6 relative error.

## What this artifact does NOT claim

- No claim on opcodes beyond the 57 weight-implemented ones.
- No claim on float64 or other precisions.
- No claim on state-vector capacities beyond $D=128$.
- No benchmark claims against frontier LLMs (see the Noesis companion paper for empirical comparisons).

## Current implementation status

The scripts in `scripts/paper/` are functional skeletons as of commit
`8235d99`. Each runs the existing upstream infrastructure where it
exists and documents its remaining TODO items inline. The three-way
verification harness is the priority path; weight export and trace
dumps build on the standalone VM compile path documented in
`CONTRIBUTING.md`.

See each script's header for its specific status.

## Issues and questions

File issues at https://github.com/tsotchke/eshkol/issues with tag
`artifact-repro`.
