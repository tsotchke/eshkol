# The Computable Transformer: Programs as Neural Network Weights

> Companion architectural specification for the Self-Differentiating Neural
> Computer (`docs/SDNC.md`). The implementation log of which opcodes are
> encoded into which weights, and on what bounded preconditions, lives in
> `docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md`.

## 1. The claim, in one paragraph

A six-layer, 12-million-parameter transformer can be made bit-identical to a
canonical 83-instruction bytecode interpreter — including its 19-opcode
reverse-mode automatic-differentiation extension — by writing its weights
out analytically from the instruction-set specification, rather than fitting
them by gradient descent. The state vector $x\in\mathbb{R}^{256}$ is the VM
state. One forward pass is one bytecode cycle: $x_{t+1} = F(x_t)$, where $F$
is a stack of attention and feed-forward blocks with linear/sigmoid/square
non-linearities. The backward pass through the *same* weights computes
reverse-mode gradients on the in-state AD tape. No layer of $F$ contains a
single trained parameter; every weight is the constructively determined
solution to a linear (or piecewise-linear) system derived from the ISA's
small-step semantics.

This is a *constructive proof of Turing-completeness modulo bounded heap*:
within a fixed live-cell budget, the transformer evaluates any term of the
canonical ISA exactly. It does so on a single set of weights, regardless of
program. The program is data; the network is the universal interpreter.

## 2. The constructive-proof argument

Most claims that a particular neural architecture is "Turing complete" are
existence proofs: a universal approximation theorem or a
machine-construction argument that takes parameters as free variables. The
SDNC takes the constructive route that has been pioneered by RASP (Weiss
et al. 2021), Tracr (Lindner et al. 2023), Looped Transformers
(Giannou et al. 2023), and Differentiable Forth (Bošnjak et al. 2017): one
exhibits a concrete weight assignment and proves that the resulting
forward pass evaluates a specified semantics. The novelties here, relative
to the prior art, are:

1. **The semantics is a *real* bytecode**, not a DSL chosen to make the
   compilation easy. The 83-opcode ISA comes from the production Eshkol
   compiler (`lib/backend/vm_core.c`) and is the same one consumed by the
   reference C interpreter (`lib/backend/vm_run.c`, computed-goto dispatch).
2. **Reverse-mode AD is in the ISA**. Nineteen opcodes
   (`OP_AD_VAR=64 … OP_AD_COS=82`) record nodes onto a 64-dimensional
   in-state tape, walk it backwards, and accumulate gradients via the same
   weight matrices. Differentiating a program is one extra dispatch state
   (`S_AD_IS_BACKWARD=1`), not a separate engine.
3. **Bit-identity in float32**, not approximate agreement. Every step of
   every program in the verifier set has every byte of its 256-float state
   vector equal across three implementations (reference C, simulated
   weights, and matrix matmul). The proof is empirical at this point but
   is supported by an analytic gate-saturation argument
   (`lib/backend/weight_matrices.c §SCALE comment lines 59-84`).
4. **One transformer block iterated against itself.** Layers 0-5 are fixed;
   the dispatch loop runs `forward_with_weights` (forward mode) or
   `backward_with_weights` (backward mode) repeatedly until `S_HALT=1`.
   This puts the construction in the same family as the Looped
   Transformers result of Giannou et al., but with the loop unrolled at
   the bytecode level rather than buried in a learned position counter.

The intellectual lineage is documented in
`docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md §10` and at the head
of `lib/backend/weight_matrices.c`; this document focuses on the
architecture itself.

## 3. The transformer

### 3.1 Architectural constants

All five constants are defined together in `weight_matrices.c §53-58`:

```c
#define D 256        /* state-vector dimension, transformer d_model */
#define H 16         /* attention heads */
#define HD 2         /* head dimension */
#define N_LAYERS 6
#define FFN_DIM 2304 /* per-layer FFN hidden width */
#define SCALE 300.0f /* softmax/sigmoid saturation temperature */
```

A direct count from the `InterpreterWeights` struct
(`weight_matrices.c §2969-2982`) gives, per layer:

- four $D\times D$ attention projection matrices $W_Q, W_K, W_V, W_O$:
  $4 \cdot 256^2 = 262\,144$ parameters
- one $D$-dimensional attention bias $b_Q$: $256$
- two $D \times FFN$ FFN matrices `ff_up`, `ff_gate`: $2 \cdot 256 \cdot 2304
  = 1\,179\,648$
- one $FFN \times D$ FFN matrix `ff_down`: $2304 \cdot 256 = 589\,824$
- two $FFN$-dimensional FFN biases: $2 \cdot 2304 = 4\,608$
- one $D$-dimensional residual bias: $256$
- a single integer `ff_type[L]` selecting the layer's activation

Per-layer total: $2\,036\,736$. Across six layers and the six `ff_type`
integers: $\boldsymbol{12\,220\,422}$ parameters. This matches the
`tab_params.tex` cell emitted by `gen_paper_tables.py §38-67` from the
exported QLMW file, and the comment at the top of
`weight_matrices.c §13`.

### 3.2 The six-layer schedule

`weight_matrices.c §10-19` documents the schedule; the explicit
forward-pass driver is `forward_with_weights` at line 5347 and the
backward pass at `backward_with_weights` line 5339.

| Layer | Activation                | Purpose                                                                                                   |
|-------|---------------------------|-----------------------------------------------------------------------------------------------------------|
| 0     | Gaussian self-attention   | **Instruction fetch.** Query $q=\text{SCALE}\cdot\mathrm{onehot}(PC)$; values carry $(opcode,operand)$.    |
| 1     | Square FFN                | **Polarisation products.** Computes $TOS \cdot SOS$ and the four AD cross-products that backward needs.   |
| 2     | Gated sigmoid FFN         | **Preprocess.** Address resolution; comparison flags; type-tag indicators; AD parent loads; cursor load. |
| 3     | Gated sigmoid FFN         | **Dispatch.** Per-opcode gate × value pairs. The arithmetic, control flow, arena and forward AD live here. |
| 4     | Gated sigmoid FFN         | **Writeback.** Tape append; arena-cell mutation; parent-load for backward.                                |
| 5     | Gated sigmoid FFN         | **Gradient writeback.** Backward-only; runs *twice* per backward cycle (gradient compute → write-back).   |

Layer 5 is unusual: the matrix-mode driver calls `apply_ffn_layer(w, 5, x)`
twice in succession (`weight_matrices.c §5343-5344`) so that the gradient
rule dispatch (pass 1) and the indexed write-back (pass 2) share weights
but execute as separate compute steps. This is the only place where the
flat six-layer schedule departs from a strict feedforward stack, and it
is the looped-transformer pattern (Giannou et al. 2023) applied to one
layer rather than the whole network.

### 3.3 Layer-by-layer specification

**Layer 0 — Attention as one-hot positional lookup.**  The instruction at
position $p$ is embedded by `embed_instruction` (line 338) as

$$
e_p = \big(p,\ -p^2/2,\ 0,\ \dots,\ 0,\ op_p,\ operand_p,\ 0,\dots\big)\in
\mathbb{R}^{256}
$$

with the opcode and operand placed at fixed lanes (`S_OPCODE=16`,
`S_OPERAND=17`). The query projection multiplies the state's `S_PC` by
SCALE; the key projection passes positional features through unchanged.
A two-dimensional head with embedding $(p, -p^2/2)$ and query
$(\mathrm{SCALE}\cdot PC, \mathrm{SCALE})$ gives the inner product

$$
\langle q,\, e_p\rangle = \mathrm{SCALE}\cdot p\cdot PC - \tfrac{\mathrm{SCALE}}{2}\cdot p^2
= -\tfrac{\mathrm{SCALE}}{2}\,(p-PC)^2 + \tfrac{\mathrm{SCALE}}{2}\cdot PC^2.
$$

The unwanted $PC^2$ term is constant across keys, so the softmax peaks at
$p=PC$ with a *Gaussian* falloff. SCALE = 300 makes the off-peak
contribution underflow to literal float32 zero (peak-to-adjacent gap
$\approx 106$, denormals start at $\exp(-87)$). The attention is
bit-exactly one-hot, and the value $W_V e_{PC} = (op_{PC}, operand_{PC})$
is written into the residual stream at `S_OPCODE`/`S_OPERAND`.

**Layer 1 — Polarisation products.** The matrix-mode forward path cannot
multiply two state dimensions directly: every layer is either linear or
gate-times-linear with no quadratic up-projection. The trick is the
polarisation identity

$$
a\cdot b = \tfrac{1}{2}(a+b)^2 - \tfrac{1}{2}a^2 - \tfrac{1}{2}b^2,
$$

which a square-activation FFN computes natively. `layer1_ffn` at
line 1244 produces four AD cross-products ($g\cdot r$, $g\cdot \ell$,
$\ell\cdot r$, $g\cdot s$) plus the plain $TOS\cdot SOS$ product needed
by `OP_MUL` and the comparison ops. The reference C interpreter
(`ad_backward_step` at line 1027) was originally written with direct
`grad * saved` multiplications; bit-identity required rewriting it to
use polarisation arithmetic (`POLARIZATION_PRODUCT` macro, lines
1064-1096) because direct multiplication and polarisation are equal
mathematically but differ by 1-13 ULPs in float32. This is the most
subtle of the five bit-identity fixes; it is documented in commit
`7301dc4` and in `docs/SDNC.md §5`.

**Layer 2 — Preprocess.** `layer2_ffn` (line 1258) is the workhorse. It
emits:

- one-hot indicator gates for the seven type tags
  (`S_TYPE_IS_NUM…S_TYPE_IS_NIL`, lines 1297-1303), so that Layer 3 can
  implement type predicates as a single `add_gated_pair` rather than a
  two-layer composition;
- the four-slot `MEM`-indexed load for `OP_GET_LOCAL` (lines 1261-1262)
  and the four-slot store delta for `OP_SET_LOCAL` (lines 1264-1265);
- `S_CMP_EQ`, `S_CMP_LT`, `S_ABS_DELTA` precomputes (lines 1288-1291);
- exact-integer DIV and MOD lookups (lines 1273-1286) for the bounded
  denominator/numerator ranges $1\le d \le 16$, $0\le v \le 21$ with
  $d\in\{3,4\}$ for MOD;
- the AD cursor load (lines 1310-1320): when `S_AD_IS_BACKWARD>0.5`,
  the eight tape fields at slot `S_AD_CURSOR` are copied to the
  `S_AD_CUR_*` register lanes via one-hot indicators;
- the AD forward parent load (lines 1340-1348): when the opcode is a
  binary AD op, the values at tape slots `S_TOS` and `S_SOS` are loaded
  into `S_AD_LEFT_VALUE` / `S_AD_RIGHT_VALUE` for Layer 1's polarisation
  products to consume next cycle;
- the cursor decrement and completion check
  (lines 1352-1371): if `S_AD_CURSOR == 0` pre-decrement,
  `S_AD_IS_BACKWARD` and `S_AD_MODE` are cleared in the same cycle.

The cursor-completion check is what fix (4) in the bit-identity bug
sequence resolved (commit `7301dc4`): the original indicator was
`indicator(cursor, -1.0)`, which fires the cycle *after* the cursor
has gone negative; the corrected `indicator(cursor, 0.0)` fires on the
cycle that processes the last live node, matching the reference VM's
`ad_backward_step` exactly.

**Layer 3 — Dispatch.** `layer3_ffn` (line 1374) is one big block of
gated indicator pairs, one per opcode-operand combination. The
mechanical primitive is `add_gated_pair` (`weight_matrices.c §2988-3009`):

```c
static int add_gated_pair(InterpreterWeights* w, int L, int n,
                          int op_id,
                          int ud1, float us1, int ud2, float us2,
                          int ud3, float us3, int ud4, float us4,
                          float ubias,
                          int out_dim, float coeff);
```

`op_id` is the opcode discriminator: the gate row places SCALE at the
`S_OPCODE` column, $-\text{SCALE}$ at `S_HALT`, and a bias of
$\text{SCALE}\cdot(-\text{op\_id}\pm 0.5)$. The two sign-conjugate
neurons combine to form a difference-of-sigmoids approximation of
$\mathbf{1}[opcode = \text{op\_id}]$ that saturates exactly in float32 as
long as $\mathrm{SCALE}>40$ (sigmoid clamp at $|x|>20$). The up
projection draws a linear combination of state dimensions into the gate
value; the down projection writes the residual delta.

For the four-dimensional bounded arithmetic family
(`OP_ADD` through `OP_GE`), each opcode is exactly *one* pair: gate on
`S_OPCODE==op_id`, up-project `TOS + SOS` or `TOS - SOS`, write residual
delta into `S_TOS`. Forty-three of the 64 base opcodes fit this shape
directly. The remainder use the family extensions
`add_gated_opcode_index` (additional integer state in the gate, e.g.
`OP_POPN` with operand $\in \{1,2,3\}$) and
`add_gated_opcode_two_indices` (two integer states, e.g. bounded
`AD_POW` indexed by base × exponent).

**Layer 4 — Writeback.** Layer 3 mutates only the working stack
registers (`S_TOS…S_R3`) and the comparison flags; the AD tape and the
Zone E arena need a separate write step because their target slot is a
function of `S_AD_TAPE_LEN` or `S_ARENA_NEXT`, not of the opcode. Layer
4 implements both: for each tape slot $i\in[0,8)$ and each of the eight
tape fields, two sign-conjugate neurons fire when
`S_AD_TAPE_LEN == i` *and* `S_AD_IS_FORWARD == 1`. The dual-input AND is
the central trick (`weight_matrices.c §4880-4951`): the gate
contribution from `S_AD_IS_FORWARD` is $10\cdot\text{SCALE}$, not
$\text{SCALE}$, so that high values of `S_AD_TAPE_LEN` (up to 7)
cannot reopen the gate when `S_AD_IS_FORWARD = 0`. Fixes (2) and (3)
in the bit-identity sequence (commit `7301dc4`) corrected exactly this:
the original code omitted `S_AD_IS_FORWARD` from the gate, so the
forward tape-write fired during backward whenever `TL == slot`,
scribbling `AD_CUR_VALUE` into `tape[1]` every backward cycle for
single-`AD_VAR` programs.

**Layer 5 — Gradient writeback.** Backward-only. Runs in two passes
because the gradient rule (which determines `S_AD_LEFT_GRAD_NEW` and
`S_AD_RIGHT_GRAD_NEW`) and the parent-tape write (which accumulates
those deltas into `tape[S_AD_CUR_LEFT]` and `tape[S_AD_CUR_RIGHT]`)
cannot share the same weight matrix without a separator: the rule
depends on `S_AD_CUR_OP` and the write depends on the *result* of the
rule. The two-pass solution is structurally identical to the looped
attention head pattern. The cursor decrement and the
`indicator(cursor, 0.0)` completion gate also live in Layer 5
(`weight_matrices.c §3737-3779`), with the dual-input AND pattern from
Layer 4 used again to prevent re-opening the cursor gate when the
backward flag is cleared.

## 4. The state vector ($d_{\text{model}} = 256$)

Layout from the enum declaration in `weight_matrices.c §136-264`:

| Dimensions   | Zone | Content                                                                                                          |
|--------------|------|------------------------------------------------------------------------------------------------------------------|
| 0–15         | A    | Persistent: `PC, TOS, SOS, R2, R3, DEPTH, OUTPUT, HALT, MEM0..MEM3, SP, FP, HAS_OUT, CUR_CLOSURE`                  |
| 16–31        | A'   | Per-cycle transients: `OPCODE, OPERAND, PRODUCT, LOADVAL, STORED0..3, ZOPER, ZPC1, CMP_EQ, CMP_LT, IS_CALL, IS_RET, IS_NATIVE, ABS_DELTA` |
| 32–35        | A_t  | Type tags for `TOS, SOS, R2, R3`                                                                                  |
| 36–47        | B    | AD control: `TAPE_LEN, CURSOR, MODE`, current-node six fields, left/right parent values                          |
| 48–111       | C    | AD tape: 8 nodes × 8 fields                                                                                       |
| 112–127      | D    | AD transient + type-predicate indicators                                                                          |
| 128–207      | E    | Bounded arena bank: 16 cells × 5 fields                                                                           |
| 208–255      | E'   | Arena operation transients (write/read scratch, vector write window, list write window)                          |

The "two state vectors" framing — that `x` is *both* a VM state and a
transformer residual stream — is more than a notational trick.
Persistent dimensions (Zones A, A_t, B, C, E) must survive one
forward pass; transient dimensions (A', D, E') must be zeroed at each
cycle. The zeroing is mechanised by Layer 3's universal reset pass at
lines 1390-1394: each transient dimension $d$ has its current value
$x[d]$ subtracted off in the residual update, so the post-cycle state
has those dimensions at zero. This is the same "clear with residual"
pattern that classical Tracr uses (Lindner et al. 2023 §3.2), but
applied to a much larger transient slice (Eshkol's per-cycle scratch
spans 96 of 256 dimensions).

The Zone E arena, added when the artifact moved from $d_{\text{model}}=128$
to $d_{\text{model}}=256$, is the construction that makes `cons`, `car`,
`cdr`, vector and string reads, and bounded closures into matrix
operations. Each of 16 cells stores `[kind, car_value, cdr_value,
car_type, cdr_type]` (16 × 5 = 80 dims). The free-list pointer
`S_ARENA_NEXT` (dimension 208) is a bump allocator: `OP_CONS` writes
into `state[S_ARENA_NEXT..S_ARENA_NEXT+4]` and increments by 5.
`OP_CAR`/`OP_CDR` are content-addressed reads via a SCALE-saturated
indicator over the cell index in `S_TOS`. The full encoding scheme is
the subject of
`docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md §5`.

## 5. The 83-opcode ISA

The canonical numbering — used by the production C compiler, the
reference interpreter, the weight constructor, and the qLLM loader — is
the enum at `lib/backend/vm_core.c §5-92` (`OP_NOP=0` through
`OP_VOID=63`, `OP_COUNT=64` for the base set) and at
`lib/backend/weight_matrices.c §102-131` (the 19-opcode AD extension
`OP_AD_VAR=64` through `OP_AD_COS=82`, `OP_COUNT=83`).

The opcode families and the count per family:

| Family            | Range          | Count | Examples                                              |
|-------------------|----------------|-------|-------------------------------------------------------|
| Stack & literals  | 0–6            | 7     | `NOP, CONST, NIL, TRUE, FALSE, POP, DUP`               |
| Arithmetic        | 7–13           | 7     | `ADD, SUB, MUL, DIV, MOD, NEG, ABS`                    |
| Comparison        | 14–19          | 6     | `EQ, LT, GT, LE, GE, NOT`                              |
| Variables         | 20–23          | 4     | `GET_LOCAL, SET_LOCAL, GET_UPVALUE, SET_UPVALUE`       |
| Functions         | 24–27          | 4     | `CLOSURE, CALL, TAIL_CALL, RETURN`                     |
| Control flow      | 28–30          | 3     | `JUMP, JUMP_IF_FALSE, LOOP`                            |
| Pairs/lists       | 31–34          | 4     | `CONS, CAR, CDR, NULL_P`                               |
| I/O & halt        | 35–37          | 3     | `PRINT, HALT, NATIVE_CALL` ← *only external boundary*  |
| Closure storage   | 38, 54         | 2     | `CLOSE_UPVALUE, OPEN_CLOSURE`                          |
| Vectors           | 39–42          | 4     | `VEC_CREATE, VEC_REF, VEC_SET, VEC_LEN`                |
| Strings           | 43–44          | 2     | `STR_REF, STR_LEN`                                     |
| Type predicates   | 45–50          | 6     | `PAIR_P, NUM_P, STR_P, BOOL_P, PROC_P, VEC_P`          |
| Set mutations     | 51–52          | 2     | `SET_CAR, SET_CDR`                                     |
| Stack n-pop       | 53             | 1     | `POPN`                                                 |
| Continuations     | 55–56          | 2     | `CALLCC, INVOKE_CC` (bounded escape only)              |
| Exceptions        | 57–59          | 3     | `PUSH_HANDLER, POP_HANDLER, GET_EXN`                   |
| Rest-args         | 60             | 1     | `PACK_REST`                                            |
| Dynamic-wind      | 61–62          | 2     | `WIND_PUSH, WIND_POP`                                  |
| Void              | 63             | 1     | `VOID`                                                 |
| AD leaf           | 64–65          | 2     | `AD_VAR, AD_CONST`                                     |
| AD binary         | 66–68, 79–80   | 5     | `AD_ADD, AD_SUB, AD_MUL, AD_DIV, AD_POW`               |
| AD unary          | 69–76, 81–82   | 10    | `AD_NEG…AD_SQRT, AD_SIN, AD_COS`                       |
| AD control        | 77–78          | 2     | `AD_BACKWARD, AD_GRAD`                                 |
| **Total**         | **0–82**       | **83** | (= 64 base + 19 AD)                                  |

Of these 83, the current artifact weight-encodes 82.
`opcode-coverage.json` from `artifacts/paper/outputs/` lists exactly
opcodes $\{0,1,\ldots,36,38,39,\ldots,82\}$ as weight-implemented and
zero opcodes as native-delegated; the missing index is 37,
`OP_NATIVE_CALL`. This is the deliberate external boundary for host
runtime services (the Eshkol runtime uses native IDs in the 300+ range
for cons, display, tensor ops, the consciousness engine, geometric
manifolds, and AD subsystem entry points; see
`lib/backend/eshkol_compiler.c` and the `vm_native.c` dispatcher).

## 6. Two tiers of execution

### 6.1 Tier 1 — Weight-encoded

The arithmetic family, comparison family, control flow, type
predicates, stack housekeeping, arena pair ops, bounded vector ops,
bounded string reads, bounded closures, bounded escape continuations,
the forward AD tape (record + unary saved-derivative table), and the
full reverse-mode walk are computed by matrix multiplication alone.
"Bounded" means: live-cell count ≤ 16, vector/string inline length ≤ 4,
tape length ≤ 8, MEM register count = 4, escape-continuation arena cells
= 4, integer divisor ≤ 16, modulus ∈ {3,4}, AD trig input ∈ [-4, 4],
AD power base ≤ 8, exponent ≤ 4. The artifact's verifier suite is
constructed within these bounds and exercises every weight-implemented
opcode.

### 6.2 Tier 2 — Native dispatch boundary

`OP_NATIVE_CALL`, opcode 37, takes a native function ID as its
operand. The Eshkol runtime maps these IDs to host C functions:

| ID range | Subsystem (see `lib/backend/vm_native.c`)                   |
|----------|-------------------------------------------------------------|
| 300–369  | Core library: full list/string/I-O routines                 |
| 370–409  | Automatic differentiation: dual numbers, AD tape, gradients |
| 410–470  | Tensor operations: matmul, reshape, reduce, dataloader      |
| 500–549  | Consciousness engine: logic, inference, workspace          |
| 800–803  | Model I/O: tensor/checkpoint save-load                      |
| 804–843  | Geometric / Riemannian manifold operations                  |

These are *not* counted as failures of the constructive-equivalence
claim. The claim is for the canonical VM/AD semantics, not the host
runtime: `OP_NATIVE_CALL` carries an *opaque integer*, and the
transformer correctly threads that integer through PC, operand
fields, and the native-dispatch postprocess that runs in C. From the
transformer's perspective `OP_NATIVE_CALL` is a non-mutating opcode
whose effect on the bounded state vector is determined entirely by
the host runtime — the same way a real CPU treats a syscall.

## 7. Three-way verification

The verification harness lives in `lib/backend/weight_matrices.c` and is
invoked by `scripts/paper/run_paper_suite.sh`. Three independent
runners execute every program:

1. **`run_reference`** (`weight_matrices.c §execute_step` at line 411
   and `run_reference` near line 1170): a direct C `switch` over the
   83 opcodes, the same one used by the production VM.
2. **`run_simulated`** (around line 2570): C functions
   (`layer0_attention`, `layer1_ffn`, `layer2_ffn`, `layer3_ffn`,
   `layer4_ffn`, `layer5_ffn` for backward) that mirror what each
   weight-matrix layer computes, but execute it as plain arithmetic
   rather than going through matrix multiplication. Used as a
   diagnostic to isolate whether a divergence is in the *logic* of the
   layer functions or in the *encoding* of those logic functions into
   weights.
3. **`run_with_weights`** (line 5347, `forward_with_weights`): the
   real $W\cdot x + b$ matrix forward pass through the
   `InterpreterWeights` struct. Calls `apply_ffn_layer` and the
   Layer-0 attention block.

For each test program, `test()` (line 5487) executes all three runners
from a freshly reset state and compares the printed output to a hand-
coded expectation. The pass count after a clean run is **126 inline
tests, 0 failed** (from `scripts/paper/run_paper_suite.sh` log: see
`build-paper.suite_trace.log` line 174).

For per-step state agreement, the same binary writes JSONL trace files
via `--trace-vm` and `--trace-transformer` flags. `compare_traces.py`
(at `scripts/paper/compare_traces.py`) groups records by
`(program_id, program, step)`, compares the eight fields
`{pc, sp, tos, sos, registers, memory, tape, flags}` bit-identically
for each weight-implemented step, and reports two metrics:

```json
{
  "status": "ok",
  "total_programs": 123,
  "output_agreeing_programs": 123,
  "fully_agreeing_programs": 123
}
```

The 123-program traced suite is a subset of the 126 inline tests; the
three excluded are inline-only diagnostics that do not produce a
traceable program sequence.

## 8. QLMW binary format

After the verifier passes, the same binary serialises the
`InterpreterWeights` struct to the path in `ESHKOL_WEIGHTS_OUT` via
`export_weights_binary` (`weight_matrices.c §5462-5478`). The on-disk
layout is fixed-endian little-endian, no padding:

```text
offset  bytes  field
  0       4   magic     = 0x514C4D57   ("QLMW", little-endian "WMLQ")
  4       4   version   = 3
  8       4   d_model   = 256
 12       4   n_layers  = 6
 16       4   ffn_dim   = 2304
 20       4   n_heads   = 16
 24       4   head_dim  = 2
 28       …   raw InterpreterWeights struct (12,220,422 × 4 bytes + 6×4)
```

The total file size is $28 + 6 \times 2\,036\,736 \times 4 + 6 \times 4
= 28 + 48\,881\,664 + 24 = 48\,881\,716$ bytes
(verified by `gen_paper_tables.py §read_qlmw_header` at line 27 and the
`stat.st_size - 28` derivation at line 46).

The QLMW format is consumed by `qllm_interpreter.c` (which loads the
weights into a runtime tensor engine for Metal/CUDA-accelerated
execution; see `qllm_interpreter.c §207` for the header parse) and by
`qllm_backward.c` for training-mode use (QLMW v4 extends the file with
optimiser state; see `qllm_backward.c §591-593`). Version 3 is the
inference-only weight-export format used by the paper artifact.

ESKB (`lib/backend/eskb_format.h`) is the *bytecode* binary container,
not the weight container — magic `0x45534B42` ("ESKB"), version 1,
section-based with CRC32 integrity check. ESKB feeds the bytecode
*compiler* path (`eshkol-run -B program.eskb`) and is independent of
the QLMW weight artifact. The two formats coexist because the
transformer needs both: ESKB to load the program tokens into Layer 0's
attention buffer, QLMW to know what the layers compute.

## 9. Why a constructive proof matters

There is a long-running tension in machine learning between
*existence* and *construction*. The universal approximation theorem
(Cybenko 1989, Hornik 1991) is an existence proof — for any continuous
function, there exists a wide-enough network that approximates it.
Turing-completeness of recurrent networks (Siegelmann & Sontag 1995)
and of transformers (Pérez et al. 2021, Wei et al. 2022) are existence
proofs — for any computable function, there exist weights that compute
it. None of these constructions are usable: the weights are either
free variables in a proof, asymptotic in a width parameter, or known
only up to a fitting procedure.

A constructive equivalence is qualitatively different. It says: *here
is a network, here is a program, here is the trace; you can run it.*
The closest precedents are:

- **Differentiable Forth** (Bošnjak et al. 2017): a Forth interpreter
  whose primitive opcodes are encoded as differentiable subprograms
  with slot-and-sketch holes for the learned bits. The construction is
  modular, but the parameters in the holes are still trained.
- **RASP** (Weiss et al. 2021): a language whose `select`/`aggregate`
  primitives map onto attention heads. Programs in RASP are
  guaranteed-compilable into transformer layers, but the compiler
  produces an architecture, not a frozen weight tensor.
- **Tracr** (Lindner et al. 2023): the RASP-to-weights compiler.
  Programs in a domain-specific subset of RASP become explicit
  transformer weights — bit-identical in float32. Tracr is the direct
  technical ancestor of the SDNC: it proved that there exists a
  pipeline from program semantics to concrete weights with no
  intervening optimiser.
- **Looped Transformers** (Giannou et al. 2023): a 13-layer
  fixed-weight transformer that, run in a loop, emulates a small
  instruction-set computer including memory load/store. This is the
  closest existing analogue to the SDNC, but Giannou et al. work with
  a ~5-opcode ISA and 4-bit operands.

The SDNC pushes the construction to (a) a *production* ISA with 83
opcodes, (b) reverse-mode AD as an in-network capability, and (c) bit-
identical agreement on the full per-step state vector rather than
just on output. The bibliography in
`docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md §10` records the
papers consulted for each design decision.

## 10. Bibliography (cached locally)

Each entry below corresponds to a paper consulted for a specific
architectural decision. The bibliography file
`research/vm-transformer-bib.bib` and the cached PDF store under
`research/papers/content/objects/<sha>/` are referenced from
`docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md §10`; both are
gitignored (≈47 MB of cached PDF) but are reconstructible by re-fetching
via the listed DOIs.

⚠ unverified: the `research/` tree was not located in the current
working copy (the directory does not exist at the repository root).
The 33-paper cache and bib file are referenced by
`VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md §10` and by the
`project_vm_transformer_memory_design.md` memory entry, but the
directory may live in the `feature/vm-transformer-memory` worktree
referenced at the top of that document rather than on master. A future
editor should locate the worktree and re-anchor the citations.

The papers most directly cited by the construction:

- Graves, Wayne & Danihelka, *Neural Turing Machines*, 2014.
  Content + location-addressable read/write heads (Layer 0 attention,
  Zone E arena reads).
- Joulin & Mikolov, *Inferring Algorithmic Patterns with
  Stack-Augmented Recurrent Networks*, 2015. Differentiable
  push/pop scalars (the bump-allocator pattern in Zone E).
- Grefenstette et al., *Learning to Transduce with Unbounded Memory*,
  2015. Bit-stable scalar stack dynamics.
- Bošnjak et al., *Programming with a Differentiable Forth
  Interpreter*, 2017. Slot-and-sketch construction model; the
  *every opcode is a fixed function on the state* mental model used
  in `weight_matrices.c`.
- Dehghani et al., *Universal Transformer*, 2018. The shared-block
  iteration that lets Layer 5 run twice per backward cycle.
- Weiss, Goldberg & Yahav, *Thinking Like Transformers* (RASP), 2021.
  `select` + `aggregate` map directly to attention heads; the type
  predicates and arena read attention are RASP aggregates.
- Lindner et al., *Tracr: Compiled Transformers as a Laboratory for
  Interpretability*, 2023. The direct ancestor compilation pipeline.
- Giannou et al., *Looped Transformers as Programmable Computers*,
  2023. Precedent for the 6→6+looped-Layer-5 schedule.
- Liu et al., *Transformers Learn Shortcuts to Automata*, 2022.
  Krohn-Rhodes decomposition justification for O(1) dispatch depth.

## 11. What this architecture does *not* claim

- It does **not** claim Turing-completeness for arbitrary unbounded
  programs. The arena bank caps live-cell count at 16; the AD tape
  caps at 8 nodes; the bytecode buffer caps at MAX_PROG instructions
  in the simulator and at the static buffer in the matrix forward
  path. Programs that exceed these bounds fall back to the production
  Eshkol VM, which has no bound but is not the SDNC.
- It does **not** claim that the transformer accelerates execution.
  The matrix forward pass is roughly $10^3\times$ slower than the C
  switch dispatcher per opcode cycle; the point is constructive
  equivalence, not throughput.
- It does **not** claim that `OP_NATIVE_CALL` is itself encoded as
  weights. Host runtime calls remain runtime calls.
- It does **not** claim bit-identical agreement on hosts with
  non-IEEE float32, non-FTZ denormal handling, or compilers that
  reorder floating point. The artifact's CI matrix is x86_64 Linux
  with glibc and macOS arm64 with libSystem; both pass the SHA-256
  pinning.

## 12. Cross-references

- `docs/SDNC.md` — the readable entrypoint to the paper artifact;
  what the SDNC is, how to reproduce it, and the bit-identity bug
  history.
- `docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md` — the
  per-opcode-class encoding spec; what each of the 82 weight-encoded
  opcodes computes inside the gated FFN.
- `docs/breakdown/BYTECODE_VM.md` — the ISA reference, with each
  opcode's small-step semantics.
- `docs/breakdown/AUTODIFF.md` — the reverse-mode AD design that the
  19 AD opcodes implement.
- `docs/tutorials/03_WEIGHT_MATRIX_TRANSFORMER.md` — a shorter,
  user-facing version of this document.

## 13. Source files

| File                                          | Lines  | Purpose                                                       |
|-----------------------------------------------|-------:|---------------------------------------------------------------|
| `lib/backend/vm_core.c`                       |    120 | Canonical opcode enum, value representation, `MEM_SIZE`, etc. |
| `lib/backend/vm_run.c`                        |    ~80 | Computed-goto VM dispatch (production reference)              |
| `lib/backend/weight_matrices.c`               |  6,772 | Weight construction, simulator, matrix forward pass, verifier |
| `lib/backend/qllm_interpreter.c`              |  1,232 | qLLM tensor-engine loader for the QLMW artifact               |
| `lib/backend/qllm_backward.c`                 |    736 | Reverse pass + AdamW + QLMW v4 checkpoint                      |
| `lib/backend/eskb_format.h`                   |    223 | ESKB bytecode container header                                |
| `lib/backend/eskb_writer.c`                   |    142 | ESKB serialiser                                               |
| `lib/backend/eskb_reader.c`                   |    271 | ESKB deserialiser                                             |
| `inc/eshkol/bridge/qllm_bridge.h`             |    260 | The Eshkol↔qLLM AD-aware tensor surface                       |
| `scripts/paper/run_paper_suite.sh`            |     86 | End-to-end reproducibility harness                            |
| `scripts/paper/compare_traces.py`             |    340 | Bit-identity comparator                                       |
| `scripts/paper/gen_paper_tables.py`           |    156 | LaTeX table regenerator                                       |
| `scripts/paper/export_weights.sh`             |     77 | QLMW regenerator (paper §4.5)                                 |
| `scripts/paper/dump_vm_trace.sh`              |     65 | Reference-VM trace dump                                       |
| `scripts/paper/dump_transformer_trace.sh`     |     56 | Matrix-forward trace dump                                     |
| `docker/Dockerfile.paper`                     |     48 | Pinned build environment (Ubuntu 22.04, clang-21)             |
| `artifacts/paper/README.md`                   |    125 | Reproducibility-package overview                              |
| `artifacts/paper/outputs/weights.qlmw`        | ~48 MB | Regenerated QLMW                                              |
| `artifacts/paper/outputs/vm-traces.jsonl`     |  ~50K  | Per-step reference traces                                     |
| `artifacts/paper/outputs/transformer-traces.jsonl` | ~50K | Per-step matrix traces                                        |
| `artifacts/paper/outputs/comparison-report.json` | 4K    | Agreement report (123/123)                                    |
| `artifacts/paper/outputs/opcode-coverage.json`   | 8K    | Per-opcode test coverage                                      |
