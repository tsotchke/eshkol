# The Self-Differentiating Neural Computer

**Paper**: *"The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction"* (tsotchke, 2026).
Companion repo: `noesis` at `docs/paper-computable-transformers/`.

**Headline claim**: a six-layer transformer with
$d_{\text{model}} = 256$ and 12.22 million analytically-constructed
parameters is bit-identical to a bounded 83-instruction bytecode VM,
with reverse-mode automatic differentiation in the ISA. No training is
involved.

**Status in this repository**: the current artifact regenerates the
weight matrices and traces in one command, then verifies 126/126 inline
programs and 123/123 traced programs. The strict weight artifact covers
82 of the 83 canonical opcodes; the only remaining external boundary is
the deliberate `OP_NATIVE_CALL` dispatch point.

---

## What is the SDNC?

The Self-Differentiating Neural Computer is a fixed-weight neural
network that *executes programs and computes their gradients* through
its own forward passes — the gradient tape lives in the state vector
and the backward pass runs through the same weight matrices that
implement the forward semantics.

Concretely:

- **6-layer transformer**, $d_{\text{model}} = 256$, feed-forward
  width 2304, 16 attention heads, 2 head dimensions, 12.22M parameters.
- **83-instruction ISA**: 64 base opcodes (arithmetic, control flow,
  memory, data structures, function calls, type predicates) and 19
  reverse-mode AD opcodes (forward recording + backward propagation).
- **82 opcodes weight-implemented**: execute entirely through analytical
  attention and feed-forward weights. The bounded artifact includes
  arithmetic, control flow, arena memory, data structures, closures,
  continuations, type predicates, and the AD tape operations exercised by
  the verifier.
- **1 deliberate external boundary**: `OP_NATIVE_CALL` carries runtime
  builtin IDs for host services and library calls. It is not counted as a
  native-delegated VM semantic opcode in the strict artifact.
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

The current artifact ships a three-way verification harness:

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

For every traced program, every step, every dimension of the 256-dimensional
state vector is compared:

```json
{
  "status": "ok",
  "total_programs":          123,
  "output_agreeing_programs": 123,
  "fully_agreeing_programs":  123
}
```

`output_agreeing` = the final program output matches; `fully_agreeing`
= every intermediate step's state vector matches bitwise (PC, SP, TOS,
SOS, registers, arena cells, tape, flags). 123/123 on both metrics is
the traced agreement contract: no traced program disagrees on its
result, no traced program disagrees at any step.

## Reproducing the result

The reproducibility package lives at `artifacts/paper/`:

```text
artifacts/paper/
├── README.md
└── outputs/
    ├── weights.qlmw                 # regenerated weight matrices
    ├── vm-traces.jsonl              # per-step VM state traces
    ├── transformer-traces.jsonl     # per-step matrix-fwd traces
    ├── comparison-report.json       # 123/123 fieldwise agreement
    ├── opcode-coverage.json         # per-opcode test coverage
    └── tables/                      # LaTeX tables for the paper
```

### One-command rerun

```bash
scripts/paper/run_paper_suite.sh
```

Expected wall time on a 2023 M2 Max: under five minutes.

### Current SHA-256 checksums

The suite prints the regenerated SHA-256 values at the end of each run.
A current successful run in this repository produced:

```text
SHA-256  weights.qlmw              381599e7a5607b4047ede0d6c8e6d270cb81dbdebfdb0bf0c0eba38758aa3f0c
SHA-256  vm-traces.jsonl           4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  transformer-traces.jsonl  4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  comparison-report.json    80aa6fed4db40bca521217ae8777677173fe7eeb239baa69847111e7ac674105
SHA-256  opcode-coverage.json      152a4bacc483d8985abeb08bc0d44112144f536ed663274bc7b1eeccbdd2dfe4
```

Platform divergence is a bug; please file an issue with the platform
details (CPU, libc, FP environment).

## Per-opcode coverage

`opcode-coverage.json` records, for every opcode in the 83-opcode
ISA, which test programs exercise it. The current strict artifact
weight-covers **82/83 canonical opcodes**. `OP_NATIVE_CALL` is the one
intentional external boundary because it names host runtime services
and library calls rather than a closed VM semantic rule.

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
- [docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md](breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md) —
  design and verification notes for the arena-memory, continuation, and
  AD opcode slices now encoded as weight matrices.
- [lib/backend/weight_matrices.c](../lib/backend/weight_matrices.c) — the
  analytical weight constructor and verifier.
- [scripts/paper/](../scripts/paper/) — the trace-dump,
  comparison, and table-generation pipeline.
- [artifacts/paper/](../artifacts/paper/) — the reproducibility
  package and outputs.
