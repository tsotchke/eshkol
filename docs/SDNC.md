# The Self-Differentiating Neural Computer

**Paper**: *"The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction"* (tsotchke, 2026).
Companion repo: `noesis` at `docs/paper-computable-transformers/`.

**Headline claim** (paper): a six-layer transformer with
$d_{\text{model}} = 128$ and 2.8 million analytically-constructed
parameters is bit-identical to an 83-instruction bytecode VM, with
native reverse-mode automatic differentiation in the ISA. No training
is involved.

**Status in this repository**: reproducibility artifact ships with
Eshkol v1.2.0-scale. One command regenerates every weight matrix and
trace, verifies bit-identical agreement on the 71-program test suite
pinned at commit `8235d99`.

---

## What is the SDNC?

The Self-Differentiating Neural Computer is a fixed-weight neural
network that *executes programs and computes their gradients* through
its own forward passes — the gradient tape lives in the state vector
and the backward pass runs through the same weight matrices that
implement the forward semantics.

Concretely:

- **6-layer transformer**, $d_{\text{model}} = 128$, feed-forward
  width 1024, 16 attention heads, 2 head dimensions, 2.8M parameters.
- **83-instruction ISA**: 64 base opcodes (arithmetic, control flow,
  memory, data structures, function calls, type predicates) and 19
  reverse-mode AD opcodes (forward recording + backward propagation).
- **57 opcodes weight-implemented** (37 base + 15 AD forward + 5 AD
  control): execute entirely through `W·x + b` matmul-plus-bias.
- **26 opcodes delegated** to the C runtime via an `IS_NATIVE` flag —
  heap operations, closures, exception handling, continuations. The
  transformer emits a correctly-tagged boundary marker; a runtime
  dispatcher honours it.
- **Self-differentiating**: the AD tape resides in the state vector;
  the backward pass dispatches through the same weight matrices that
  carry the forward semantics. Programs are tensors, running them is a
  matmul, differentiating their output with respect to inputs is a
  backward pass through the same network.

The paper is a constructive — not statistical — proof that a
transformer can *be* an interpreter when its weights are derived
analytically from the ISA specification rather than fit by gradient
descent. The artifact in this repository is the empirical confirmation
of the proof.

## The artifact contract

The artifact at the pinned commit `8235d99` ships **71 test programs**
(52 base + 19 AD) with a three-way verification harness:

1. **Reference C interpreter** — direct `switch` over the 83 opcodes;
   the ground truth.
2. **Simulated transformer** — C functions that mirror the six layers
   (Gaussian-attention instruction fetch, polarisation-identity
   product, address-resolution preprocessing, gated opcode dispatch,
   tape write + parent load, backward gradient dispatch + write-back).
3. **Matrix-based forward pass** — explicit weight matrices generated
   by `generate_weights`, applied via the generic
   $\mathbf{y} = W_\text{down}\,(\sigma(W_g\mathbf{x} + b_g) \odot
   (W_u\mathbf{x} + b_u)) + b_d$ formula at each layer.

Agreement across all three modes constitutes the verification chain:
mode 1 is the spec by construction, mode 2 confirms the analytical
layer functions match the spec, mode 3 confirms the weight matrices
reproduce the layer functions when applied via standard matrix
multiplication.

For every program, every step, every dimension of the 128-dimensional
state vector is compared:

```json
{
  "status": "ok",
  "total_programs":          71,
  "output_agreeing_programs": 71,
  "fully_agreeing_programs":  71
}
```

`output_agreeing` = the final program output matches; `fully_agreeing`
= every intermediate step's state vector matches bitwise (PC, SP, TOS,
SOS, registers, memory, tape, flags). 71/71 on both metrics is the
agreement contract: no program disagrees on its result, no program
disagrees at any step.

> **Note on test suite size.** The paper PDF reports 74 programs
> (55 base + 19 AD). The artifact at `8235d99` pins to a 71-program
> snapshot (52 base + 19 AD). The three additional base programs in
> the paper post-date the artifact pin; they will fold into a future
> artifact revision.

## Reproducing the result

The reproducibility package lives at `artifacts/paper/`:

```text
artifacts/paper/
├── README.md
└── outputs/
    ├── weights.qlmw                 # regenerated weight matrices
    ├── vm-traces.jsonl              # per-step VM state traces
    ├── transformer-traces.jsonl     # per-step matrix-fwd traces
    ├── comparison-report.json       # 71/71 fieldwise agreement
    ├── opcode-coverage.json         # per-opcode test coverage
    └── tables/                      # LaTeX tables for the paper
```

### One-command rerun

```bash
scripts/paper/run_paper_suite.sh
```

Expected wall time on a 2023 M2 Max: under five minutes.

### Pinned-artifact rerun

```bash
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
git checkout 8235d9987d70086e6e62083d120f3cf51fac9e48
docker build -f docker/Dockerfile.paper -t eshkol-paper .
docker run --rm -v "$(pwd):/work" -w /work eshkol-paper \
    scripts/paper/run_paper_suite.sh
```

### Expected SHA-256 checksums

On any IEEE 754 float32 platform, regeneration produces these
exact hashes:

```text
SHA-256  weights.qlmw              638376aab6d49e829da2c54d22b545d86c50aa1c2d508e8ec029d2a6d3f1e77d
SHA-256  vm-traces.jsonl           564fbe1fa4dba5793db0c0e54d402932061f2c82b94da470c7541a5c421584f3
SHA-256  transformer-traces.jsonl  5cc01b2a17e87d88628b13ef5f7602bd7bcd6380e407a0aac5c39b35a9570715
SHA-256  comparison-report.json    8a7917d2b56254f9fad71a4cd5e59284504313e73f99c2b668b461a74e154aab
SHA-256  opcode-coverage.json      aa0c666ad3c2b7a1034e4a69deee6e271e53261bddb12e07dce52fb55218438f
```

Platform divergence is a bug; please file an issue with the platform
details (CPU, libc, FP environment).

## Per-opcode coverage

`opcode-coverage.json` records, for every opcode in the 83-opcode
ISA, which test programs exercise it. **43 distinct opcodes are
exercised by the 71-program suite**; the remaining weight-implemented
opcodes from the paper's 57-opcode total are exercised only by
extended programs not in the artifact (loop control variants, deeper
stack ops, additional symbolic forms).

Coverage table per opcode is regenerated by `compare_traces.py`
during the suite run.

## Implementation note: float32 saturation margin

The paper proves the gated indicator function is exact in float32
arithmetic for any scale parameter $S > 33.2$ (the bound from
$S/2 > 24 \ln 2$), and uses $S = 100$ as the conservative working
constant. This repository's weight generator (`lib/backend/weight_matrices.c`)
ships with `SCALE = 300` rather than 100. The reason is empirical:
at $S = 100$ the softmax score gap between the peak position and its
adjacent positions is $\approx 35.4$, so the off-peak residue is
$e^{-35.4} \approx 4.6 \times 10^{-16}$ — perfectly representable in
float32, and observed to accumulate as `tos = 4.4e-16` in
`tail sum(100)` at step 1206 versus exactly 0 in the reference.
Raising $S$ to 300 pushes the gap to $\approx 106 > 87$ so $e^{-\text{gap}}$
underflows to literal float32 zero. The proof's bound is tight only
for the indicator gates; full bit-identical agreement at the
attention-softmax level requires the larger margin. This is documented
in the comment block at `weight_matrices.c:59-84`.

## See also

- [docs/breakdown/COMPUTABLE_TRANSFORMER.md](breakdown/COMPUTABLE_TRANSFORMER.md) —
  the high-level architecture of programs-as-matrices.
- [docs/breakdown/BYTECODE_VM.md](breakdown/BYTECODE_VM.md) —
  the 83-opcode ISA the transformer implements.
- [lib/backend/weight_matrices.c](../lib/backend/weight_matrices.c) — the
  4248-line analytical weight constructor.
- [scripts/paper/](../scripts/paper/) — the trace-dump,
  comparison, and table-generation pipeline.
- [artifacts/paper/](../artifacts/paper/) — the reproducibility
  package and outputs.
