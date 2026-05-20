# The Self-Differentiating Neural Computer

> Canonical reference for the paper artefact in this repository. For the
> architectural specification, see
> `docs/breakdown/COMPUTABLE_TRANSFORMER.md`. For the per-opcode-class
> implementation log, see
> `docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md`.

**Paper.** *"The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction"* (tsotchke, 2026).
The companion Noesis repository contains the LaTeX source at
`docs/paper-computable-transformers/`. This Eshkol repository is the
production-code artefact.

**Headline.** A six-layer transformer with
$d_{\text{model}} = 256$, $\text{FFN} = 2304$, $H = 16$ attention heads
of width $\text{HD} = 2$, and **12,220,422** analytically-constructed
parameters reproduces, bit-identically in float32, an 83-instruction
bytecode VM that includes reverse-mode automatic differentiation as
part of its ISA. No training. No optimiser. The weights are the
deterministic closed-form solution to the linear/piecewise-linear
constraints implied by the small-step semantics.

**Status.** The current artefact regenerates the weight matrices and
verifies **126/126 inline programs** and **123/123 traced programs**.
The matrix forward path agrees with the reference C interpreter on
every byte of every step of every traced program. The strict artefact
covers **82 of the 83 canonical opcodes**; the only opcode left
outside is `OP_NATIVE_CALL`, the deliberate external boundary for
host services.

---

## 1. Motivation

Two facts about transformers have grown more curious as the
mechanistic-interpretability literature has matured. First,
trained transformers exhibit, in their attention patterns and FFN
neurons, structures that *look* programmatic: induction heads
(Olsson et al. 2022), arithmetic micro-circuits (Nanda et al. 2023),
copy-and-paste subroutines that can be ablated as units. Second,
deliberately-constructed transformers from the Tracr (Lindner et al.
2023) and Looped-Transformers (Giannou et al. 2023) lines are
*provably* programmatic: they execute a fixed semantics for arbitrary
inputs, given the right weight assignment, with no training.

The natural question is whether the two facts share a common shape.
If a trained transformer ends up encoding programs in its weights,
what would it look like to start with a program and derive the
weights *directly*, with no intervening fit? The Self-Differentiating
Neural Computer is one answer. The construction is exhaustive enough
to cover an actual bytecode (the Eshkol VM ISA, used by the
production C compiler), specific enough that every parameter is named
and accounted for, and verified strongly enough that any divergence
counts as a bug — the artifact pins the per-step state vectors of two
independent runners against each other and refuses to release a
weight file until they agree.

A second motivation is differentiable programming. If a program *is*
a sequence of matmul layers, then $\partial \text{program}/\partial
\text{input}$ and $\partial \text{program}/\partial \text{weights}$
are both well-defined, available, and standard. The SDNC takes this
seriously enough to put the AD tape *inside* the state vector that
the transformer evolves: the reverse-mode walk runs through the same
weight matrices as the forward pass, with no separate engine,
because the tape lives in 64 of the 256 state dimensions. A program
is a tensor; running it is matmul; differentiating it is the
backward pass through the network whose weights *are* the interpreter.

## 2. What is the SDNC?

The Self-Differentiating Neural Computer is a fixed-weight neural
network that **executes programs and computes their gradients** through
its own forward passes — the AD tape lives in the state vector and the
backward pass runs through the same weight matrices that implement the
forward semantics.

Concretely (constants from `lib/backend/weight_matrices.c §53-86`):

- **Six-layer transformer**, $d_{\text{model}} = 256$, FFN width 2304,
  $H = 16$ attention heads with $\text{HD} = 2$ each, 12,220,422
  parameters. Layer activations: one Gaussian-attention block, one
  square-activation FFN, four gated-sigmoid FFNs.
- **83-instruction ISA**: 64 base opcodes (`OP_NOP=0` through
  `OP_VOID=63`; see `lib/backend/vm_core.c §5-92`) plus 19 reverse-mode
  AD opcodes (`OP_AD_VAR=64` through `OP_AD_COS=82`).
- **82 opcodes weight-implemented**: arithmetic, comparison, control
  flow, type predicates, stack housekeeping, bounded pair/vector/string
  ops, bounded closures and upvalues, bounded escape continuations,
  bounded integer division/modulus, the forward AD tape (record +
  saved-derivative table), and the full reverse-mode walk.
- **1 deliberate external boundary**: `OP_NATIVE_CALL` (opcode 37)
  carries a host-runtime ID and bridges to OS services and library
  calls. The transformer correctly threads it through PC, operand,
  and dispatch flags; the *callee* lives in C.
- **Self-differentiating.** The AD tape is 8 nodes × 8 fields = 64
  dimensions of the state vector (`weight_matrices.c §175-177`,
  Zone C). `OP_AD_BACKWARD` flips `S_AD_IS_BACKWARD=1`; the dispatch
  loop then runs `backward_with_weights` instead of
  `forward_with_weights`. Layer 1 (cursor load), Layer 4 (parent
  load), Layer 2 (polarisation products), and Layer 5 (gradient rule
  + write-back, run twice) compute the reverse pass.

The paper's framing is a **constructive — not statistical — proof**
that a transformer can *be* an interpreter when its weights come from
the ISA spec rather than from gradient descent. The artifact in this
repository is the empirical confirmation: the production C VM, the C
simulator of the layer functions, and the actual `W \cdot x + b`
matrix forward pass agree on every byte.

## 3. The constructive-proof framing

Most claims that a neural architecture is "Turing complete" are
existence proofs. The universal approximation theorem (Cybenko 1989)
proves *that* weights exist; it does not say what they are. The
Turing-completeness of recurrent and transformer networks (Siegelmann
& Sontag 1995; Pérez et al. 2021; Wei et al. 2022) proves the same
thing one level up, with the weights still existentially quantified.
Such proofs do not let you take a program and produce the weights.

The SDNC follows a different lineage:

- **Differentiable Forth** (Bošnjak et al. 2017): a Forth interpreter
  whose primitive opcodes are differentiable subprograms with
  *sketch* parameters that may be learned. The construction is
  modular; the holes are still trained.
- **RASP** (Weiss et al. 2021): a domain-specific language whose
  `select`/`aggregate` primitives compile to attention heads. RASP
  programs are guaranteed transformer-executable, but the compiler
  emits an architecture, not weights.
- **Tracr** (Lindner et al. 2023): the RASP-to-weights compiler.
  Programs in a subset of RASP become explicit weight tensors, bit-
  identical in float32. This is the direct technical ancestor of the
  SDNC: it proved that there exists a pipeline from program semantics
  to concrete weights with no optimiser anywhere.
- **Looped Transformers** (Giannou et al. 2023): a 13-layer
  fixed-weight transformer that emulates a small ISA when iterated
  against itself. About 5 opcodes, 4-bit operands.

The SDNC extends this lineage in three respects: it targets a
*production* 83-opcode ISA (not a DSL chosen to make the compilation
easy), it includes reverse-mode AD as an in-ISA capability
(differentiating programs is one extra dispatch state, not a separate
engine), and it pins **bit-identical** agreement on the *full per-step
state vector* — not just on output. Any divergence is a regression to
be fixed; the artefact will not ship if a single byte diverges (see
the bug history in §6).

## 4. The artefact contract

The verification harness ships with three independent runners
(`lib/backend/weight_matrices.c`):

1. **Reference C interpreter** — `execute_step` at line 411 and
   `run_reference` near line 1170. A direct C `switch` over the 83
   opcodes; the ground truth.
2. **Simulated transformer** — `layer0_attention`, `layer1_ffn`,
   `layer2_ffn`, `layer3_ffn`, `layer4_ffn`, plus the backward
   `ad_backward_step` (lines 1027-1105). C functions that mirror what
   each weight matrix computes, but implement the operations as plain
   arithmetic. Lets us isolate a logic bug in the layer functions
   from an encoding bug in the matrix form.
3. **Matrix-based forward pass** — `forward_with_weights` (line 5347).
   The actual `W \cdot x + b` matrix multiplication through the
   `InterpreterWeights` struct (line 2969). Six layers, plus the
   second Layer-5 pass for backward.

Agreement across all three modes is the verification chain. Mode 1 is
the spec by construction; mode 2 confirms the analytical layer
functions match the spec; mode 3 confirms the weight matrices
reproduce the layer functions when applied through standard matmul.

For every program in the **123-program traced suite**, every step,
every field of the 256-dimensional state vector is compared
bit-identically. The agreement report
(`artifacts/paper/outputs/comparison-report.json`) looks like:

```json
{
  "status": "ok",
  "total_programs":          123,
  "output_agreeing_programs": 123,
  "fully_agreeing_programs":  123
}
```

`output_agreeing_programs` = the final `OP_PRINT` output matches.
`fully_agreeing_programs` = every intermediate step's state vector
matches bit-identically (PC, SP, TOS, SOS, registers, arena cells,
tape, flags). **123/123 on both metrics is the contract**: no traced
program disagrees on its result, no traced program disagrees at any
step. The inline test count is **126/126** (the three extra programs
are diagnostics that don't produce traceable sequences).

Per-opcode coverage from
`artifacts/paper/outputs/opcode-coverage.json`:

```text
weight_implemented   : 82 opcodes (all of 0..82 except 37)
native_delegated     :  0 opcodes
transformer_native_assisted : 0 opcodes
```

That is, the only opcode in the canonical 0–82 range *not* covered by
the strict weight artefact is `OP_NATIVE_CALL` (37), the deliberate
host-runtime boundary.

## 5. Architecture summary

A condensed sketch; the full architectural spec is
`docs/breakdown/COMPUTABLE_TRANSFORMER.md`.

| Layer | Activation                | Function                                                       |
|-------|---------------------------|----------------------------------------------------------------|
| 0     | Gaussian self-attention   | Instruction fetch: peaked at PC, values carry $(op, operand)$. |
| 1     | Square FFN                | Polarisation products: $TOS\cdot SOS$ and four AD cross-products. |
| 2     | Gated sigmoid FFN         | Preprocess: address resolution, comparisons, type indicators, AD parent loads, cursor load. |
| 3     | Gated sigmoid FFN         | Dispatch: per-opcode gate × value, arithmetic, control flow, arena, forward AD. |
| 4     | Gated sigmoid FFN         | Writeback: tape append, arena-cell mutation, backward parent load. |
| 5     | Gated sigmoid FFN (×2)    | Gradient rule + indexed write-back, backward only.             |

Layer 5 is unusual: the matrix-mode driver calls
`apply_ffn_layer(w, 5, x)` *twice* per backward cycle
(`weight_matrices.c §5343-5344`) because the gradient rule and the
parent-tape write cannot share the same weight matrix (the write
depends on the rule's result). This is the looped-Transformer pattern
applied to a single layer.

The **256-dimensional state vector** maps onto the VM's bounded
runtime state (`weight_matrices.c §136-264`):

```text
0-15    persistent registers   (PC, TOS, SOS, R2, R3, DEPTH, MEM0..3, …)
16-31   per-cycle transients   (OPCODE, OPERAND, PRODUCT, LOADVAL, comparison flags, …)
32-35   type tags              (TYPE_TOS, TYPE_SOS, TYPE_R2, TYPE_R3)
36-47   AD control             (TAPE_LEN, CURSOR, MODE, current-node fields, parent values)
48-111  AD tape                (8 nodes × 8 fields = 64 dims)
112-127 AD/type-pred transient (forward/backward flags, gradient accumulators, type indicators)
128-207 Zone E arena bank      (16 cells × 5 fields = 80 dims)
208-255 arena op transients    (pair/vector/list write windows)
```

The **bounds** that the artifact contract operates within:

- arena live-cell count ≤ 16, vectors/strings inline ≤ 4 elements,
  closure upvalues ≤ 4, bounded escape continuations = 4 arena cells;
- AD tape length ≤ 8 nodes; AD transcendentals (`AD_SIN`, `AD_COS`)
  tabulated for integer input in [-4, 4]; `AD_DIV` for positive
  integer denominator in [1, 16]; `AD_POW` for base in [1, 8] and
  exponent in [1, 4]; bounded `AD_EXP`/`AD_SIGMOID`/`AD_TANH`/
  `AD_LOG`/`AD_SQRT` for the verifier-exercised inputs;
- integer division denominator ≤ 16; modulus ∈ {3, 4} for the verifier;
- MEM register file size = 4; bytecode buffer = 64 entries in the
  simulator and a 256-entry buffer in the matrix forward path.

These bounds are the *contract*; programs that stay inside them
execute end-to-end through matrix multiplication, with every step
agreeing bit-identically with the reference. Programs outside them
fall back to the production Eshkol VM (which has no such bounds).
The bounds are *not* fundamental to the architecture: they are the
slice that the current 12.22 M-parameter, 256-dim model covers; a
larger state vector and FFN width scale them linearly.

## 6. The five bit-identity bugs

The artefact agreement metric used to be **52/71** on full per-step
state and **71/71** on output agreement: every program's final
`OP_PRINT` value was within the inline 0.01 tolerance, but 19
programs disagreed somewhere in the middle, on intermediate state
the inline tests didn't catch. The drift was documented as "acceptable
within the paper's bit-identical-output claim." A round of per-step
trace inspection (the `--trace-vm`/`--trace-transformer` flags were
added for exactly this) showed that the 19 disagreements were five
distinct bugs, all weight-encoding errors. Fixing them moved the
contract to **71/71 on full per-step state agreement** and made
the SHA-256 of the resulting trace files stable across IEEE-754
float32 platforms.

The five fixes are recorded in commit
**`7301dc4 fix(paper): bit-identical SDNC agreement — 71/71 full
per-step state`**. Each one is documented inline in
`lib/backend/weight_matrices.c` near the lines it touched. The
subsequent expansion to 123 traced programs (the bounded VM-as-
transformer memory-ops series) reused the same diagnostic methodology
and preserved the bit-identical contract.

### 6.1 Bug #1 — Softmax temperature was too low

**Symptom.** `tail sum(100)` diverged: matrix forward path had
`tos = 4.4e-16` at step 1206; reference VM had exactly `0`. The
discrepancy propagated indefinitely through state accumulation but
stayed below the inline 0.01 tolerance, so the inline test passed
while the per-step trace caught it.

**Diagnosis.** With $\mathrm{SCALE} = 100$, Layer 0's attention
softmax has a peak-to-adjacent score gap of
$\mathrm{SCALE}/(2\sqrt{2}) \approx 35.4$. $\exp(-35.4) \approx 4.6
\times 10^{-16}$ is perfectly representable in float32 — denormals do
not kick in until $\exp(-87)$ — so every "off-peak" position
contributed a residue to the value sum. The contribution was tiny but
real and accumulated step by step.

**Fix.** Raise $\mathrm{SCALE}$ to 300. The peak-to-adjacent gap
becomes $\approx 106 > 87$, so $\exp(-\text{gap})$ underflows to
*literal float32 zero*. Documented at `weight_matrices.c §59-84`. The
analytic bound for full saturation is
$\mathrm{SCALE} > 246$; 300 leaves a ~50-ULP margin per step,
sufficient for the 6-layer accumulation chain.

### 6.2 Bug #2 — Layer 4 forward tape-write missing the IS_FORWARD gate

**Symptom.** For programs containing a single `OP_AD_VAR` followed
later by `OP_AD_BACKWARD`, the matrix path showed `tape[1] = 7` (i.e.
the `AD_CUR_VALUE` of node 0 scribbled into tape slot 1) while the
reference had `tape[1] = 0`. Visible in the trace for the
"AD edge: grad of var = 1" program.

**Diagnosis.** Layer 4 writes the tape on forward (`OP_AD_VAR/CONST/
ADD/MUL/…`). The intended gate is `(S_AD_TAPE_LEN == slot) AND
(S_AD_IS_FORWARD == 1)`. The generator's loop had only the
`S_AD_TAPE_LEN` coefficient wired; the comment promised the
`S_AD_IS_FORWARD` coefficient but the source did not set it. So
during *backward*, when `S_AD_IS_FORWARD = 0` but `S_AD_TAPE_LEN`
took whatever value it had been left at, the gate fired and the
forward-write neuron pair scribbled `S_AD_CUR_VALUE` into the
matching tape slot.

**Fix.** Add the `S_AD_IS_FORWARD` row of every forward-write gate
in Layer 4 (`weight_matrices.c §4882-4951`). See bug 3 for the
weighting choice.

### 6.3 Bug #3 — Dual-input AND in a sigmoid gate needs an asymmetric weight

**Symptom.** Bug #2's first fix attempt wrote both gate inputs at
weight `SCALE`. The single-AD-VAR program now wrote correctly during
forward but *reopened* the gate during backward on programs with
high `S_AD_TAPE_LEN`.

**Diagnosis.** The gate's bias is
$\mathrm{SCALE}(-\text{slot} + 0.5) - \mathrm{SCALE}$, so the gate
input under matching tape length and `IS_FORWARD = 0` is
$\mathrm{SCALE}\cdot(0 \pm 0.5) - \mathrm{SCALE} \approx
-\mathrm{SCALE}/2$ — sigmoid-closed. But if `S_AD_TAPE_LEN` happens
to equal 7 (which it does during the latter half of a long forward
sequence), the gate input is $\mathrm{SCALE}\cdot 7 +
\mathrm{SCALE}\cdot 0 + \text{bias} = \mathrm{SCALE}\cdot 6$
*before* the IS_FORWARD term — far past saturation. The intended
AND truth table never held: it became "either input opens the gate".

**Fix.** Multiply the `S_AD_IS_FORWARD` coefficient by 10. The
binary condition now contributes $10\cdot\mathrm{SCALE}$ when
present and 0 when absent, swinging the sigmoid argument harder
than any plausible integer in `S_AD_TAPE_LEN` (max 7). The same
pattern was then applied to the cursor-completion gate in Layer 5
(`weight_matrices.c §3752-3779`). The lesson is that dual-input
ANDs in a sigmoid-gated FFN must use *asymmetric* weights when the
two inputs have different dynamic ranges.

### 6.4 Bug #4 — Backward cursor termination off by one

**Symptom.** 17 of the 19 originally-disagreeing AD programs showed
a one-step PC, register, and `tos` drift in the cycle *after* the
backward pass finished.

**Diagnosis.** Both the simulated `layer2_ffn` and the matrix
Layer-2 weights had the completion gate as
`at_done = indicator(cursor, -1.0)`. That indicator fires when
`cursor == -1` post-decrement, i.e. on the cycle *after* the last
live node has been processed. So the matrix path ran one *extra*
backward cycle with `S_AD_IS_BACKWARD = 0` but stale tape state,
while the reference VM (`ad_backward_step` at line 1027) clears
`S_AD_IS_BACKWARD` in the *same* cycle that makes the cursor go
negative.

**Fix.** Change the completion indicator to
`indicator(cursor, 0.0)`: fires when the *pre-decrement* cursor is
the last node (which becomes -1 after the same cycle's decrement).
Now the matrix path and the reference VM clear backward mode on the
same step. The indicator is documented inline at
`weight_matrices.c §1352-1371` (simulator) and §3744-3779 (weight
matrix).

### 6.5 Bug #5 — Polarisation arithmetic in the reference VM

**Symptom.** A single remaining divergence in
`MLP: df/dw sigmoid(...)`: matrix path final `tos = 0.393223852`,
reference VM `0.393223763`. A 1-ULP difference in the gradient
output, *but* the inline test passed because both values rounded to
the same expectation under 0.01 tolerance.

**Diagnosis.** Layer 1 is a SQUARE-activation FFN. To compute
$a \cdot b$ from $a$ and $b$, it must use the *polarisation
identity*

$$a \cdot b = \tfrac{1}{2}(a+b)^2 - \tfrac{1}{2}a^2 - \tfrac{1}{2}b^2.$$

The reference C interpreter, by contrast, was happily computing
`grad * saved` directly. Polarisation and direct multiplication are
*mathematically* equal but produce different float32 results: the
polarisation form has three squarings, three multiplications by
0.5, and two subtractions, whereas direct is one multiplication. In
float32 the operation order shifts the result by 1-13 ULPs.

**Fix.** Rewrite `ad_backward_step` to use the polarisation
identity for the unary `grad * saved` product and the binary
`grad * value` products. The macro
`POLARIZATION_PRODUCT(a, b)` at `weight_matrices.c §1064-1066`
codifies the substitution. *All* float32 multiplications in the
reference's backward pass now follow the same operation order as
the matrix path's Layer 1. The 1-ULP gradient divergence
disappeared.

This is the most philosophically interesting of the five fixes.
Bit-identity required *the reference VM to match the matrix
architecture's arithmetic*, not the other way around. The square-
activation FFN cannot do direct multiplication architecturally;
the polarisation identity is what it can compute; the reference
implementation has to follow suit if it wants to be a spec for
"what the transformer is allowed to compute." Documented at
`weight_matrices.c §1055-1063`.

### 6.6 The aftermath

After these five fixes, the agreement was 71/71 inline plus 71/71
on full per-step state (commit `7301dc4`). The subsequent bounded-
arena memory-ops series (state vector extended from 128 to 256,
arena bank added, pair/vector/string/closure/escape-continuation
opcodes encoded; see
`docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md §8`) expanded
the suite to 123 traced programs and to 82 weight-encoded opcodes
without weakening the bit-identity contract. Each new opcode goes
through the same three-way verification harness; any per-step
divergence is treated as a regression to fix, not as documentable
drift.

## 7. Reproducibility

### 7.1 The reproducibility package

```text
artifacts/paper/
├── README.md
└── outputs/
    ├── weights.qlmw                 # regenerated weight matrices (~48 MB)
    ├── vm-traces.jsonl              # per-step reference-VM traces
    ├── transformer-traces.jsonl     # per-step matrix-forward traces
    ├── comparison-report.json       # 123/123 fieldwise agreement
    ├── opcode-coverage.json         # 82-opcode coverage per program
    └── tables/                      # LaTeX tables (params, verification, coverage)
```

### 7.2 One-command rerun

```bash
scripts/paper/run_paper_suite.sh
```

Expected wall time on a 2023 M2 Max: **under five minutes**. The
script orchestrates four phases (`scripts/paper/run_paper_suite.sh
§30-69`):

1. **Build + regenerate weights.** `export_weights.sh` configures
   CMake against `build-paper/` (hermetic; does not touch the user's
   normal build), builds the `weight_matrices` tool, runs it with
   the verification suite, and exits non-zero if any of the 126
   inline tests fails. `ESHKOL_WEIGHTS_OUT` is honoured to put the
   QLMW file where the suite expects it.
2. **Dump both traces in one run.** A single
   `weight_matrices --trace-vm <path> --trace-transformer <path>`
   invocation runs the suite once and emits both JSONL files. The
   two trace files are *identical except for `is_native` flagging
   on the boundary opcode*; the matrix forward path duplicates the
   reference state at every weight-implemented step, so the same
   binary can dump both.
3. **Compare.** `compare_traces.py` groups the two JSONL files by
   `(program_id, program, step)`, compares the eight tracked fields
   (`pc, sp, tos, sos, registers, memory, tape, flags`) bit-
   identically for weight-implemented opcodes and boundary-only for
   native opcodes, and writes `comparison-report.json` and
   `opcode-coverage.json`. Exits non-zero only if
   `output_agreeing_programs < total_programs`.
4. **Regenerate tables.** `gen_paper_tables.py` reads the QLMW
   header (`§read_qlmw_header` line 27) and the comparison report
   to emit `tab_params.tex`, `tab_verification.tex`, and
   `tab_opcode_coverage.tex` ready for paper insertion.

### 7.3 Current SHA-256 checksums

```text
SHA-256  weights.qlmw              381599e7a5607b4047ede0d6c8e6d270cb81dbdebfdb0bf0c0eba38758aa3f0c
SHA-256  vm-traces.jsonl           4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  transformer-traces.jsonl  4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  comparison-report.json    80aa6fed4db40bca521217ae8777677173fe7eeb239baa69847111e7ac674105
SHA-256  opcode-coverage.json      152a4bacc483d8985abeb08bc0d44112144f536ed663274bc7b1eeccbdd2dfe4
```

The `vm-traces.jsonl` and `transformer-traces.jsonl` files are
expected to be bit-identical (same SHA-256) because the matrix path
*is* bit-identical to the reference on the full per-step state.
Platform divergence is a bug; please file an issue with the
platform details (CPU, libc, FP environment).

### 7.4 Pinned historical artifact

The first published artefact pinning is
`artifacts/paper/README.md §7-13`:

- Commit: `8235d9987d70086e6e62083d120f3cf51fac9e48`
- Tag: `v1.1.13-accelerate-166-g8235d99`
- Build env: Ubuntu 22.04, clang-21, LLVM 21, CMake 3.14+

The Docker file in `docker/Dockerfile.paper` rebuilds that exact
environment:

```bash
docker build -f docker/Dockerfile.paper -t eshkol-paper .
docker run --rm -v "$(pwd):/work" -w /work eshkol-paper \
    scripts/paper/run_paper_suite.sh
```

Later commits expanded the bounded arena memory-ops; the SHA-256
values above are the *current* artifact, not the pinned historical
one. Both are valid: the pin demonstrates the SDNC at its first
published frozen state; the current artifact demonstrates that the
contract has held while the weight coverage grew from 57/83 to 82/83.

### 7.5 The QLMW binary format

The weight container is a 28-byte header followed by the raw
`InterpreterWeights` struct (`weight_matrices.c §5462-5478`):

```text
offset  bytes  field
  0       4   magic     = 0x514C4D57    ("QLMW" little-endian)
  4       4   version   = 3
  8       4   d_model   = 256
 12       4   n_layers  = 6
 16       4   ffn_dim   = 2304
 20       4   n_heads   = 16
 24       4   head_dim  = 2
 28       …   InterpreterWeights raw struct (12,220,422 floats + 6 ints)
```

QLMW v3 is the inference format used by the paper artefact. QLMW v4
(`qllm_backward.c §588-616`) extends the file with optimiser state
for training-mode use; v4 is consumed by the training loop but is
*not* the artifact. The `gen_paper_tables.py` table generator
verifies the magic and derives the parameter count from
`stat.st_size - 28` divided by 4
(`scripts/paper/gen_paper_tables.py §38-67`).

### 7.6 Interpreting the agreement metrics

`compare_traces.py` reports two metrics
(`scripts/paper/compare_traces.py §285-291`):

- **`output_agreeing_programs`** — the paper's §4.4 claim. The first
  `OP_PRINT` of every program matches between the two runners.
- **`fully_agreeing_programs`** — the stricter check. Every byte of
  every step's state vector matches. The matrix path has *no
  architectural way to compute direct multiplication* (Layer 1 is a
  SQUARE FFN), so the reference VM's backward path must use the
  polarisation identity to match (see bug #5 above).

Both metrics currently equal `total_programs = 123`. If a future
patch ever causes the second to fall, the patch must be reverted or
the second metric must be raised back: the contract is bit-identity
on the full state, not just on output.

## 8. Per-opcode coverage

`opcode-coverage.json` records, for every opcode in the canonical
0–82 ISA, which test programs exercise it. The current strict
artifact weight-covers **82/83 canonical opcodes**:

```text
weight_implemented   : 0,1,2,…,36, 38,39,…,82   (82 opcodes)
native_delegated     : (empty)
transformer_native_assisted : (empty)
```

`OP_NATIVE_CALL = 37` is intentionally outside; the canonical ISA's
*only* host-runtime boundary remains the only artefact-uncovered
opcode. The exercised trace set has *no* native-assisted transformer
steps and *no* IS_NATIVE postprocess assistance; the bounded artefact
ran end-to-end through matmul for every opcode it touched.

The per-opcode test-program list is regenerated by
`compare_traces.py` during the suite run and surfaced in the
`tab_opcode_coverage.tex` table.

## 9. Float32 saturation margin

The paper's analytic bound for full softmax saturation is
$S > 2 \cdot \log(2 \cdot \exp(33.2)) \approx 33.2 + \log 2$, i.e.
roughly $S = 100$ suffices for the sigmoid-difference indicators.
The repository's weight generator
(`lib/backend/weight_matrices.c §59-84`) ships with
$\mathrm{SCALE} = 300$, not 100. Two reasons:

1. **Empirical accumulation.** At $S = 100$ the softmax score gap
   between the peak position and its adjacent positions is
   $\approx 35.4$, so the off-peak residue is
   $\exp(-35.4) \approx 4.6 \times 10^{-16}$ — perfectly
   representable in float32, and observed to propagate as
   `tos = 4.4e-16` in `tail sum(100)` at step 1206 versus exactly
   `0` in the reference (bug #1 above).
2. **Underflow margin.** $S = 300$ pushes the gap to $\approx 106
   > 87$, so $\exp(-\text{gap})$ underflows to literal float32 zero
   (denormals start at $\exp(-87)$). The proof's bound is tight only
   for the indicator gates; full bit-identical agreement at the
   attention-softmax level requires the larger margin.

The comment block at `weight_matrices.c §59-84` is the canonical
justification.

## 10. Citation

If you cite the artefact or the construction:

> tsotchke. *The Self-Differentiating Neural Computer: Computable
> Transformers via Analytical Weight Construction.* Tsotchke
> Corporation, 2026. Companion artefact:
> `https://github.com/tsotchke/eshkol`, `scripts/paper/`.

The paper LaTeX is at `docs/paper-computable-transformers/` in the
Noesis companion repository.

## 11. See also

- `docs/breakdown/COMPUTABLE_TRANSFORMER.md` — the architectural
  specification (six-layer schedule, attention as one-hot fetch,
  polarisation arithmetic, QLMW format, bibliography).
- `docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md` — the per-
  opcode-class implementation log; how each of the 82
  weight-encoded opcodes maps onto the gated-FFN structure.
- `docs/breakdown/BYTECODE_VM.md` — the ISA reference for the
  underlying 83-opcode bytecode.
- `docs/breakdown/AUTODIFF.md` — reverse-mode AD specification.
- `docs/tutorials/03_WEIGHT_MATRIX_TRANSFORMER.md` — shorter,
  user-facing exposition.
- `lib/backend/weight_matrices.c` — the analytical weight
  constructor and verifier (6,772 lines).
- `lib/backend/vm_core.c` — the canonical 83-opcode ISA enum.
- `scripts/paper/` — the trace-dump, comparison, and
  table-generation pipeline.
- `artifacts/paper/` — the reproducibility package and outputs.
