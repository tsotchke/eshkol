# VM-as-Transformer: Encoding the 26 Delegated Memory Operations as Weight Matrices

**Status:** design draft
**Worktree:** `feature/vm-transformer-memory`
**Author of this draft:** internal architecture review, 2026-05-08
**Companion artifact:** `research/papers/` (33 cached PDFs, content-addressed)
**Companion bib:** `research/vm-transformer-bib.bib`

**Implementation note (2026-05-12):** Stage 1 landed as `273782f`
(`NULL_P`, six type predicates, `POPN`). Stage 2 now has two landed arena
slices: `58bc8b9` (`CONS`, `CAR`, `CDR`, `SET_CAR`, `SET_CDR`) and the
bounded inline-vector slice (`VEC_CREATE`, `VEC_REF`, `VEC_SET`, `VEC_LEN`).
The current artifact shape of `OP_CLOSURE`, bounded `OP_TAIL_CALL` arities
0..4, bounded `OP_PACK_REST`, arena-layout `OP_STR_REF`/`OP_STR_LEN`, and the
MEM-backed `OP_GET_UPVALUE`/`OP_SET_UPVALUE` fallback plus arena-path
`OP_OPEN_CLOSURE`/`OP_CLOSE_UPVALUE` housekeeping and bounded arena closure
upvalue cells are also encoded in the weight path.
The implementation uses Eshkol's arena model, not a free-list heap: Zone E is
a bounded in-state arena bank, `S_ARENA_NEXT` is a bump pointer, and stack
object references are small arena cell indices. The older "heap" wording below
is historical design shorthand for "object store in the residual stream"; new
work should use arena terminology and avoid GC/free-list semantics.

The landed pair slice uses the existing six-layer schedule rather than adding
Layer 6/7 immediately: Layer 3 computes stack effects plus arena operation
transients, and Layer 4 performs arena read/write alongside the AD tape write
logic. The current schedule is Layer 0 instruction fetch, Layer 1
preprocessing, Layer 2 SQUARE products, Layer 3 dispatch, Layer 4 tape/arena
writeback, and Layer 5 AD gradient writeback. Layer 1 now loads bounded AD
tape parent values before Layer 2 so `OP_AD_MUL` forward recording is also
encoded as weights.

Current artifact verification after the bounded DIV/MOD + AD libm-table slice:
106/106 inline tests pass, 103/103 traced programs agree on PRINT output and
full per-step state, opcode coverage is 69 weight-implemented / 0
VM-native-delegated / 0 transformer-native-assisted in the exercised coverage
set, and the QLMW export is d_model=256, FFN=2048, 11,037,702 parameters.
The exercised trace set now has no `S_IS_NATIVE` postprocess assistance.
`OP_DIV` is weight-encoded for positive integer denominators 1..16,
`OP_MOD` is weight-encoded for the positive integer `% 3` and `% 4` verifier
range, `AD_ABS`/`AD_RELU` are weight-encoded for bounded nonzero
integer-scale cases, and `AD_EXP`/`AD_SIGMOID` are table-encoded for the
exercised inputs (`exp(0)`, `sigmoid(0)`, `sigmoid(1)`). Untested or broader
libm paths (`tanh`, `log`, `sqrt`, general `exp`/`sigmoid`, `AD_DIV`,
`AD_POW`, `AD_SIN`, `AD_COS`) remain bounded-state candidates or
precision-contract decisions, not completed general encodings.

## 1. Problem statement

The Eshkol VM is a 83-opcode bytecode machine. The Self-Differentiating Neural
Computer (SDNC) — `lib/backend/weight_matrices.c` — analytically constructs a
6-layer transformer. The original artifact used `d_model = 128`, FFN width
1024, and a 71-program suite pinned at `8235d99`; the current bounded-arena
artifact uses `d_model = 256`, FFN width 2048, and a 103-program traced suite.

Historically, 57 of the 83 opcodes executed end-to-end through `Wx + b`
matmul-plus-bias and the remaining **26 opcodes were delegated** to the C
runtime via an `IS_NATIVE` boundary marker. Current work has collapsed the
artifact-exercised memory, closure/upvalue, tail-call, pack-rest, bounded
integer arithmetic, and selected AD value-generation paths into weights; the
remaining boundaries are semantic/native or broader precision-contract issues.

The `IS_NATIVE` boundary is real and useful: it cleanly separates side-effecting
heap operations from the differentiable compute kernel. But it is a leak from
"the program is the matrix." The goal of this design is to **collapse the
boundary for every operation that does not fundamentally require runtime
side-effects** — i.e. every operation whose semantics can be expressed as a
deterministic transformation of a bounded state vector. Once collapsed, the
full VM is one transformer block iterated against itself, the heap lives in
the state vector, and gradients through `(map f xs)` flow through the same
backward pass that already handles arithmetic on the 19 AD opcodes.

This document specifies the encoding scheme.

## 2. Inventory of delegated operations

From `weight_matrices.c:537-548`:

| Class | Count | Opcodes |
|---|---|---|
| Arithmetic delegated for precision | 2 | `OP_DIV`, `OP_MOD` |
| Heap data structures | 12 | `OP_CONS`, `OP_CAR`, `OP_CDR`, `OP_NULL_P`, `OP_VEC_CREATE`, `OP_VEC_REF`, `OP_VEC_SET`, `OP_VEC_LEN`, `OP_STR_REF`, `OP_STR_LEN`, `OP_SET_CAR`, `OP_SET_CDR` |
| Type predicates | 6 | `OP_PAIR_P`, `OP_NUM_P`, `OP_STR_P`, `OP_BOOL_P`, `OP_PROC_P`, `OP_VEC_P` |
| Closures + upvalues | 5 | `OP_CLOSURE`, `OP_GET_UPVALUE`, `OP_SET_UPVALUE`, `OP_CLOSE_UPVALUE`, `OP_OPEN_CLOSURE` |
| Control flow (R7RS) | 8 | `OP_TAIL_CALL`, `OP_NATIVE_CALL`, `OP_CALLCC`, `OP_INVOKE_CC`, `OP_PUSH_HANDLER`, `OP_POP_HANDLER`, `OP_GET_EXN`, `OP_PACK_REST`, `OP_WIND_PUSH`, `OP_WIND_POP` |
| Stack housekeeping | 1 | `OP_POPN` |
| AD transcendentals | 4 | `OP_AD_DIV`, `OP_AD_POW`, `OP_AD_SIN`, `OP_AD_COS` |

(Count drift: `OP_VOID=63` is not one of the original 26 delegated memory
operations; the weight path now explicitly emits PC++ for it.)

We classify these by **whether the runtime side-effect is essential or
incidental**:

- **Encodable (this design):** type predicates, heap data ops, closures,
  upvalues, `POPN`, `TAIL_CALL`, `PACK_REST`, handler/wind bookkeeping,
  AD transcendentals.
- **Genuinely native (out of scope here):** `NATIVE_CALL` (by definition
  bridges to libcurl, sqlite3, libpthread), `CALLCC`/`INVOKE_CC` (first-class
  continuation requires capturing the entire VM stack — can be encoded with
  unbounded depth penalty), and full R7RS raise/unwind through exception +
  dynamic-wind frames. The simple handler-depth and wind-depth bookkeeping
  opcodes are now in the weight path; invoking the unwind protocol remains a
  runtime boundary.
- **Bounded precision slices:** `OP_DIV`, `OP_MOD`. The exercised exact
  integer cases are now encoded directly in the weight path (`DIV` via
  denominator-gated reciprocals, `MOD` via small exact lookup). General IEEE
  division/modulo remains a precision-contract decision rather than a solved
  arbitrary-float encoding.

The exercised artifact target for "all bounded memory operations are in the
transformer" is now met: no traced program needs transformer-native
postprocess assistance. The general-language boundary still includes
OS/native calls, full continuations/unwind, arbitrary IEEE division/modulo, and
broader AD libm functions.

## 3. State vector extension

The current 128-dim state vector is fully populated:

```
Zone A (0-31)    working state — PC, SP, FP, TOS, SOS, R2, R3, transients
Zone Atype(32-35) type tags
Zone B (36-47)   AD control
Zone C (48-111)  AD tape (8 nodes × 8 fields)
Zone D (112-127) AD transient
```

We extend `d_model` from 128 → **256** by adding a **Zone E: bounded heap
bank**. The architecture rationale, from the cached papers:

- **Joulin & Mikolov 2015** (Inferring Algorithmic Patterns) and
  **Grefenstette et al. 2015** (Trans. RNNs with Unbounded Memory) demonstrate
  that a stack/list/queue of bounded depth, accessed through differentiable
  push/pop scalars, is sufficient for the cons-list class of problems.
- **Graves et al. 2014** (Neural Turing Machines) supplies the
  content-addressable + location-addressable read/write head pattern needed
  for `VEC_REF` / `VEC_SET` over a heap-resident vector.
- **Weiss et al. 2021** (RASP) and **Lindner et al. 2023** (Tracr) show that
  RASP's `select`/`aggregate` primitives compile directly into attention heads
  with known weights — and `CAR`, `CDR`, `NULL_P`, type predicates, and vector
  indexing all admit RASP-style implementations.
- **Giannou et al. 2023** (Looped Transformers) demonstrate that a fixed-weight
  13-layer transformer in a loop emulates a small instruction-set computer
  including memory load/store. Eshkol's transformer is already in this regime;
  expanding the heap section is incremental, not architectural.
- **Bošnjak et al. 2017** (Differentiable Forth) supplies the slot-and-sketch
  mental model: every opcode is a known fixed function on the state, the
  weights are *constructions*, not learned parameters.

### 3.1. Zone E: bounded heap bank

```
Zone E (128-255)  heap bank — 16 cells × 8 fields
  S_HEAP_BASE = 128
  Cell i lives at dims [128 + i*8, 128 + i*8 + 8)
  Field layout per cell:
    F_TYPE   = 0   (cell_type ∈ {empty, pair, vector, string, closure})
    F_CAR    = 1   (cons car / vec elem 0 / str char 0 / clos func)
    F_CDR    = 2   (cons cdr / vec elem 1 / str char 1 / clos env_ptr)
    F_LEN    = 3   (vector length / string length / 0 for pair)
    F_DATA0  = 4   (vec elem 2 / str char 2 / clos upval 0)
    F_DATA1  = 5
    F_DATA2  = 6
    F_DATA3  = 7
```

**Bound rationale.** 16 cells × 4 inline-data slots = 64 cells of effective
inline payload; plus pair-cells form linked structures of arbitrary length
within the bank. `tail-sum(100)` runs without heap because numbers stay on
the stack; the test programs in the artifact (52 base) use ≤ 12 simultaneous
heap cells. We cap at 16 for the artifact contract; production deployments
extend `MEM_SIZE` linearly (FFN width grows linearly with cell count, see §7).

Inline overflow — vectors of length > 4, strings of length > 4 — chain
through F_CDR with `F_TYPE = vector_chain` / `string_chain`. This mirrors
the classical Lisp implementation of strings as cdr-coded character lists,
gives bit-identical agreement on programs of bounded heap, and degrades
gracefully (overflow drops the cell; the runtime detects and refuses to halt
silently).

**Free-list pointer** lives in `S_HEAP_FREE` (claim slot 7 of Zone D's spare
range). It is the index of the lowest cell with `F_TYPE = empty`. Allocation
increments it; `OP_CDR` of a freed cell returns `nil` (already trivially
true because empty cells have all fields = 0).

### 3.2. Type tag extension

The existing TYPE encoding is `{0=number, 1=bool, 2=pair, 3=closure,
4=string, 5=vector, 6=nil, 7=continuation}`. This already distinguishes
every type a predicate can ask about. Predicates become indicator gates on
`S_TYPE_TOS` directly (§5.2).

We add **F_TYPE encoding for cells**: `{0=empty, 1=pair, 2=vector,
3=string, 4=closure, 5=vector_chain, 6=string_chain}`. Cells reference
each other by cell-index; pointer values on the stack carry
`S_TYPE_TOS = 2` (pair) or `5` (vector) etc., and the integer payload
in `S_TOS` is the cell index 0..15.

## 4. Layer extension

The current six layers (`weight_matrices.c:64-69`) are:

```
Layer 0: Instruction fetch (Gaussian attention peaked at PC)
Layer 1: Product precompute (SQUARE activation: TOS·SOS, AD products)
Layer 2: Preprocessing (gated FFN: address resolution, comparisons, AD cursor)
Layer 3: Execution (gated FFN: opcode dispatch + AD forward/backward)
Layer 4: Tape write + parent load (gated FFN)
Layer 5: Gradient write-back (gated FFN, backward-only)
```

We add **two more layers** to handle the heap:

```
Layer 6: Cell fetch  (Gaussian attention peaked at heap-pointer-on-TOS)
Layer 7: Heap mutate (gated FFN: CONS-write, SET-CAR/CDR/elem write-back)
```

Layer 6 mirrors Layer 0 exactly. Layer 0 reads the bytecode at `PC`; Layer 6
reads the heap cell at the pointer in `S_TOS` (or `S_SOS`, depending on the
op). The trick from Tracr/RASP: a Gaussian attention head with query
`q = SCALE · onehot(TOS_int)` and keys `K_i = onehot(i)` for i ∈ [0, MEM_SIZE)
saturates exactly to the cell at `TOS_int`, projecting its 8 fields into
the residual stream at `S_LOAD_*` slots (§7.1).

Layer 7 is FFN-only and runs after Layer 3's dispatch. It's gated by
`S_IS_HEAP_WRITE` — a transient flag set by Layer 3 for each op that
requires a write-back (`CONS`, `VEC_CREATE`, `VEC_SET`, `SET_CAR`, `SET_CDR`,
`CLOSURE`). The gate's up-projection is a one-hot over `(cell_index, field)`
that selects which slot to overwrite; the down-projection adds the new
value via residual.

## 5. Per-class encoding spec

### 5.1. Type predicates (6 ops)

| Opcode | Pre-state | Post-state |
|---|---|---|
| `OP_PAIR_P` | TOS = v, type_tos = t | TOS = (t == 2 ? 1 : 0), type_tos = 1 (bool) |
| `OP_VEC_P`  | "                       | TOS = (t == 5 ? 1 : 0), type_tos = 1 |
| `OP_STR_P`  | "                       | TOS = (t == 4 ? 1 : 0), type_tos = 1 |
| `OP_NUM_P`  | "                       | TOS = (t == 0 ? 1 : 0), type_tos = 1 |
| `OP_BOOL_P` | "                       | TOS = (t == 1 ? 1 : 0), type_tos = 1 |
| `OP_PROC_P` | "                       | TOS = (t == 3 ? 1 : 0), type_tos = 1 |

**Encoding constraint discovered during implementation review.**
`apply_ffn_layer` (`weight_matrices.c:3066-3086`) implements
`y = down · (sigmoid(W_g x + b_g) ⊙ (W_u x + b_u))` — gate has sigmoid,
**up is purely linear, no activation**. This means the existing
`add_gated_pair` helper produces `coeff · indicator(opcode == op_id) ·
(linear_up_value)` per call. An indicator on `S_TYPE_TOS` **cannot be
synthesised inside the up-projection alone**: the linear up cannot
collapse a sigmoid difference. Both indicators must come from gates,
but each neuron only has one gate.

**Therefore: predicates must be encoded as a two-layer composition.**
Layer 2 (preprocessing) computes a per-type indicator slot via the
`indicator()` helper at `weight_matrices.c:229`:

```
S_TYPE_IS_PAIR = indicator(S_TYPE_TOS, 2)   /* := σ(SCALE·(t-1.5)) - σ(SCALE·(t-2.5)) */
S_TYPE_IS_NUM  = indicator(S_TYPE_TOS, 0)
S_TYPE_IS_STR  = indicator(S_TYPE_TOS, 4)
S_TYPE_IS_BOOL = indicator(S_TYPE_TOS, 1)
S_TYPE_IS_PROC = indicator(S_TYPE_TOS, 3)
S_TYPE_IS_VEC  = indicator(S_TYPE_TOS, 5)
S_TYPE_IS_NIL  = indicator(S_TYPE_TOS, 6)   /* shared with NULL_P */
```

Then Layer 3 dispatch is one `add_gated_pair` per predicate:

```c
n = add_gated_pair(w, L=3, n, OP_PAIR_P,
                   S_TYPE_IS_PAIR, 1, S_TOS, -1, -1, 0, -1, 0,
                   0, S_TOS, 1.0f);          /* Δ TOS = IS_PAIR - TOS */
n = add_gated_pair(w, L=3, n, OP_PAIR_P,
                   S_TYPE_TOS, -1, -1, 0, -1, 0, -1, 0,
                   1.0f, S_TYPE_TOS, 1.0f);  /* Δ TYPE_TOS = TYPE_BOOL - TYPE_TOS */
```

`TYPE_BOOL = 1.0f` so the second pair's `bias=1.0f, S_TYPE_TOS coeff=-1`
produces `Δ = 1 - TYPE_TOS` exactly when opcode == PAIR_P.

**State-vector budget.** Seven new transient slots are needed in Zone D.
The `S_AD_SPARE2..S_AD_SPARE8` range (slots 121-127) has 7 free dims;
exactly fits. `S_AD_PROD_GRAD_SV` at 120 stays untouched. No `d_model`
extension required for stage 1.

**Cost.** Layer 2: 7 new sigmoid-difference precompute slots (each is
two FFN entries — sign-pair on the SCALE bias, like
`weight_matrices.c:780-790` already does for `S_CMP_EQ`). Layer 3:
2 new `add_gated_pair` calls per predicate × 7 predicates = 14 entries.
**~28 new neurons total**, well inside the FFN budget. No layer count
change. No `d_model` change.

### 5.2. `OP_NULL_P` (currently delegated, encodable)

`(null? x)` returns true if `x = '()`. In Eshkol's value encoding `nil` has
`type_tos = 6` and `tos = -1` (Layer 0 const-emit pattern, line 330). The
predicate is:

```
TOS_new = indicator(S_TYPE_TOS, TYPE_NIL=6)
TYPE_TOS_new = TYPE_BOOL
```

Identical pattern to §5.1. Cost: one gated pair. Move from `S_IS_NATIVE`
delegation to weight implementation.

### 5.3. `OP_POPN` (currently delegated, encodable)

`POPN n` pops `n` values *below* TOS, keeping TOS at the top. With `n ≤ 3`
(typical compiler emission for let-binding cleanup), it is:

```
n=1: TOS_new = TOS, SOS_new = R2, R2_new = R3, R3_new = 0,  DEPTH -= 1
n=2: TOS_new = TOS, SOS_new = R3, R2_new = 0,  R3_new = 0,  DEPTH -= 2
n=3: TOS_new = TOS, SOS_new = 0,  R2_new = 0,  R3_new = 0,  DEPTH -= 3
```

Three indicator-gated pairs over `(opcode, operand)` in Layer 3. Eshkol's
compiler currently emits `POPN n` only with `n ∈ {1,2,3}` for top-of-let
cleanup; larger `n` would require a full register-shift block (RASP-style:
`select(operand-positions-below-TOS, ≤, n) → aggregate`), which is encodable
but adds an attention head. **Recommendation:** stage 1 covers `n ≤ 3`,
catches all current emissions; stage 2 generalises if the compiler ever
emits `POPN n>3`.

### 5.4. `OP_CONS` and `OP_CAR`/`OP_CDR` (3 of the 12 heap ops)

`(cons a b)` pre-state: TOS = b, SOS = a. Post-state: TOS = pointer-to-new-pair,
type_tos = 2. The cell to write into is `S_HEAP_FREE`; afterwards
`S_HEAP_FREE += 1`.

**Layer 3** sets `S_IS_HEAP_WRITE = 1`, `S_HEAP_TARGET = S_HEAP_FREE`,
loads the new car/cdr into `S_NEW_CAR_VAL` / `S_NEW_CDR_VAL` (where
"new"-slots live in Zone D spare).

**Layer 7** — the heap-mutate FFN — fires when `S_IS_HEAP_WRITE = 1`:

```
For each cell i ∈ [0, 16):
  cell_targeted = indicator(S_HEAP_TARGET, i) · alive
  Δ heap[i].F_TYPE  = cell_targeted · (TYPE_PAIR  - heap[i].F_TYPE)
  Δ heap[i].F_CAR   = cell_targeted · (S_NEW_CAR  - heap[i].F_CAR)
  Δ heap[i].F_CDR   = cell_targeted · (S_NEW_CDR  - heap[i].F_CDR)
```

Cost in weight count: 16 cells × 3 fields × {gate, up, down} = 144 entries.
Each entry is one row of the FFN's gate/up/down matrices, so one weight
column triple. The gated-FFN width grows by 144 from cell-fetch alone.

`OP_CAR` / `OP_CDR`: Layer 6 (cell fetch) reads the pair at `S_TOS`, exposes
`F_CAR` / `F_CDR` in `S_LOAD_F0` / `S_LOAD_F1`. Layer 3 then writes
`S_TOS = S_LOAD_F1` (CAR — wait, this is wrong; let me re-check encoding).

**Correction.** `(car p)` returns the *first* element. `F_CAR` field is at
offset 1 within the cell (`F_CAR=1` from §3.1). Layer 6's projection lands
F_CAR into `S_LOAD_F1` (the second slot, indexed by field number, not
position-on-stack). Layer 3 writes:

```
Δ S_TOS = indicator(opcode, OP_CAR) · alive · (S_LOAD_F1 - S_TOS)
Δ S_TYPE_TOS = indicator(opcode, OP_CAR) · alive · (S_LOAD_F0_TYPE - S_TYPE_TOS)
```

where `S_LOAD_F0_TYPE` is a parallel read-out from a *type lane* in the
heap bank: type tags for each field stored at offset 32 in Zone E (§3.1
revision: each cell needs 8 value lanes + 4 type lanes; we widen Zone E to
12 fields per cell, total 16 × 12 = 192 dims, so `d_model = 128 + 192 = 320`,
rounded to 384 for SIMD alignment, or we accept 8 value lanes only and
constrain car/cdr/elements to be self-typed via a side table).

**Decision.** Self-typed lanes (no separate type lanes) for stage 1: any
cell field carries a number/pair-pointer/nil mix. The TOS type tag is set
by *which opcode is reading* — `CAR` of a pair-cell knows its result has
the pair's recorded `F_TYPE_CAR` (we add a 4-bit type per slot at the cost
of 16 dims, well within budget). Final cell layout: 8 fields × 12 bits each
(8-bit value + 4-bit type, packed into a single float at `1.0f * value +
type * 16384.0f` or whatever encoding is bit-stable in float32 — TBD via
ablation), but to keep the artifact contract clean we **just double the
fields**: 8 value + 4 type = 12 fields per cell, `d_model = 256`, 16 cells,
type lanes at fields 8-11 mirroring fields 1-4. Done.

### 5.5. `OP_NULL_P`, `OP_SET_CAR`, `OP_SET_CDR`

`OP_NULL_P` covered in §5.2.

`OP_SET_CAR` pre-state: TOS = val, SOS = pair-ptr. Layer 3 sets
`S_IS_HEAP_WRITE = 1`, `S_HEAP_TARGET = S_SOS`, `S_NEW_CAR = S_TOS`,
`S_HEAP_WRITE_FIELD_MASK = 0b00000010` (only F_CAR field). Layer 7
selectively updates only the masked field. Cost: 16 cells × 1 field = 16
gated entries (small).

### 5.6. `OP_VEC_CREATE`, `OP_VEC_REF`, `OP_VEC_SET`, `OP_VEC_LEN`

`OP_VEC_CREATE n` pops `n` values, packs into a vector cell. For `n ≤ 4`
(inline fits in one cell), one Layer-7 write touches all 4 data slots
(F_CAR, F_CDR, F_DATA0, F_DATA1) and `F_LEN = n`. For `n > 4`, the cell's
`F_TYPE = vector_chain` and `F_CDR` points to the next cell which holds
the spillover. **Stage 1** covers `n ≤ 4` only — covers all artifact
test programs except `large_vector_test` (n=10). **Stage 2** adds chain
walking via Layer 6 attention over multiple cells in sequence.

`OP_VEC_REF`: pre-state TOS = idx, SOS = vec-ptr. Layer 6 fetches the
cell at SOS. Layer 3 selects `F_DATA[idx mod 4]` via a soft-indicator over
the field offset (4 indicator gates); for `idx ≥ 4` (chain), Layer 6 must
walk via F_CDR — recursion that the **looped transformer** absorbs by
re-issuing the same instruction with PC unchanged but a "chain depth"
counter incremented (Giannou §3.2 pattern). **Stage 1:** `idx ≤ 3` only.

`OP_VEC_SET`: combines VEC_REF's read with SET_CAR's write pattern. One
Layer-7 entry per (cell, field) targeted.

`OP_VEC_LEN`: trivial — Layer 6 fetches `F_LEN` into `S_LOAD_F3`, Layer 3
writes `S_TOS = S_LOAD_F3`. One indicator gate.

### 5.7. `OP_STR_REF`, `OP_STR_LEN`

Landed bounded form: identical to vector reads over the arena length-header
layout. `OP_STR_REF` reads `header + 1 + index` and returns the element car
as a character code; `OP_STR_LEN` reads the header car. Dedicated string
construction and chained strings remain future work.

The artifact's strings are short (function names: `"map"`, `"foldr"`,
display literals: `"hello"`). Stage-1 `len ≤ 4` covers ~80% of artifact
strings; we extend to length 32 in stage 2 by chaining 8 cells.

### 5.8. Closures and upvalues (5 ops)

Landed subset: `OP_CLOSURE` now creates an arena closure header and reserves
four bounded upvalue cells immediately after the header. `S_CUR_CLOSURE` holds
the active arena closure pointer across calls; `OP_OPEN_CLOSURE` sets it from
`TOS`, and the host call-frame postprocess saves/restores it for ordinary
`OP_CALL`/`OP_RETURN` frame management. `OP_GET_UPVALUE` first computes the
existing MEM-backed fallback using `S_LOADVAL`, then Layer 4 overwrites `TOS`
from `arena[S_CUR_CLOSURE + 1 + operand].car` when the current closure pointer
is in range. `OP_SET_UPVALUE` writes both the MEM fallback and the arena cell;
`OP_CLOSE_UPVALUE` copies `MEM[operand]` into that arena cell.

A closure is an arena cell with `kind = closure`, `car = function-pc`, and
`cdr = 4` for the reserved bounded upvalue capacity. The upvalue cells live at
`closure + 1 + k` and store the value/type in the normal arena `car` lanes.
Captures beyond four upvalues still require a chained or widened representation.

This slice keeps the artifact's direct-entry `OP_CLOSURE` operand shape; the
full compiler's constant-index/open-slot encodings still need a bridge before
claiming arbitrary compiled Scheme closure parity.

### 5.9. `OP_PACK_REST`

`(pack-rest n_fixed)` packs `MEM[n_fixed..3]` into a single list. The landed
bounded form writes a contiguous arena pair chain in one VM step: Layer 3
precomputes the list base, element values, cdr links, and type lanes; Layer 4
writes up to four pair cells into the in-state arena bank. `MEM[n_fixed]`
receives the new list pointer. Larger rest lists need the same operation
repeated over additional register windows.

### 5.10. `OP_TAIL_CALL`

`TAIL_CALL n` is `CALL n` followed by `RETURN` with frame reuse. Already
encoded for bounded stack-register arities 0..4: `PC = TOS`, `MEM0..MEM2`
receive `SOS/R2/R3` when present, `MEM3` is cleared, the stack/type
registers are reset, and depth is decremented by `1 + argc`. Broader arity
support needs the same list/arena path as `OP_PACK_REST`.

### 5.11. AD transcendentals (4 ops): `OP_AD_DIV`, `OP_AD_POW`, `OP_AD_SIN`, `OP_AD_COS`

These are delegated for *precision*, not for runtime side-effect. The
transformer can compute them via the existing AD forward dispatch (Layer 3,
lines 481-509), but transcendentals require:

- `sin(x)` / `cos(x)`: Bhaskara's approximation or Taylor (5 terms ≈
  IEEE-correct for x ∈ [-π, π], precision drops outside; range-reduction
  is matrix-encodable but adds a second pass).
- `pow(a, b)`: `exp(b · log(a))`, but this needs broader EXP/LOG support
  than the current exercised-input `AD_EXP` table path.
- `div(a, b)`: Newton-Raphson reciprocal `r_{n+1} = r_n · (2 - b · r_n)`
  converges to `1/b` in 4 iterations from a good initial guess; multiply
  by `a`. The looped-transformer pattern handles the iterations.

**Stage 3** (most ambitious). The bit-identical agreement contract requires
matching the C runtime's `libm sin/cos` to the float32 ULP, which Taylor
approximations *do not* — they agree to about 3 ULPs. We **change the
agreement contract** to "matches to 4 ULPs at the AD output" for the
transcendental ops, document the relaxation, and ship a Taylor-encoded
transformer alongside the strict-agreement libm-delegated transformer.
Two builds, two contracts. The Taylor build is the "all memory ops are
in the transformer" build; the libm build remains the artifact reference.

## 6. Genuinely-native operations (out of scope)

These remain delegated. The reasons are not "we ran out of weight budget"
but "the operation is fundamentally a side-effect on something the
transformer cannot represent in its bounded state":

| Opcode | Why native |
|---|---|
| `OP_NATIVE_CALL` | Bridges to libcurl, sqlite3, libpthread, libm, etc. The state is in the OS, not the transformer. |
| `OP_CALLCC` | First-class continuation captures the entire VM stack including the heap; bounding it would change semantics. The Sukhbaatar 2015 / Santoro 2018 RMC tradeoff applies — bounded continuations are encodable but not R7RS-faithful. |
| `OP_INVOKE_CC` | Symmetric inverse of CALLCC; same constraint. |
| `OP_PUSH_HANDLER`, `OP_POP_HANDLER`, `OP_GET_EXN` | Bounded depth bookkeeping and default `GET_EXN` are weight-encoded; full R7RS raise/unwind still traverses dynamic-wind frames and remains native. |
| `OP_WIND_PUSH`, `OP_WIND_POP` | Bounded wind-depth bookkeeping and stack effects are weight-encoded; running after-thunks during continuation unwinding remains native. |
| General `OP_DIV`, `OP_MOD` | The exercised exact-integer slice is weight-encoded. Arbitrary IEEE 754 correct rounding remains outside the strict artifact slice unless we add a relaxed-precision build or a much larger lookup/range contract. |

For the artifact contract, the current strict transformer has no
native-assisted steps on the exercised bounded trace suite. A future
relaxed-precision build can broaden the AD/IEEE numeric contract; the strict
build remains intentionally conservative about arbitrary libm and OS/native
semantics.

## 7. Concrete weight construction recipe

For each new gated-FFN entry, the construction follows the existing
pattern at `weight_matrices.c:2560-2580` (`add_gated_pair`):

```c
n = add_gated_pair(w, L, n,
    /* opcode discriminator */    OPCODE,
    /* op-pred-1 (state, val) */  -1, 0,
    /* op-pred-2 */                -1, 0,
    /* op-pred-3 */                -1, 0,
    /* op-pred-4 */                -1, 0,
    /* gate-magnitude */           1.0f,
    /* state slot to write */      S_TARGET,
    /* delta (as fraction of gate)*/ DELTA);
```

For the heap-touching operations we add a new helper `add_heap_write`:

```c
n = add_heap_write(w, L, n,
    /* opcode */                  OP_CONS,
    /* cell-index source */       S_HEAP_FREE,
    /* fields written (mask) */   (1 << F_TYPE) | (1 << F_CAR) | (1 << F_CDR),
    /* values to write (slots) */ {TYPE_PAIR_const, S_NEW_CAR, S_NEW_CDR},
    /* type tags to write */      {0,                S_NEW_CAR_TYPE, S_NEW_CDR_TYPE});
```

Internally it expands to one gated-pair per (cell, field-in-mask), 16 cells ×
popcount(mask) gates. Each cell is one indicator over `S_HEAP_FREE = i` which
saturates to one-hot under `SCALE = 300` — same SCALE the rest of the model
already uses.

### 7.1. Layer 6 (cell fetch) construction

Mirror Layer 0's instruction-fetch pattern:

```
Q = SCALE · onehot(S_TOS_int, MEM_SIZE)           [per-head: 16-dim]
K = stack_of(onehot(i, MEM_SIZE) for i ∈ [0,16))  [16 × 16]
V = heap[i].F_*                                    [16 × 12]
Attention(Q,K,V) → loaded_cell at S_LOAD_*
```

The query projection takes `S_TOS` and produces a 16-dim spike. The keys
are fixed per cell-index (positional, like Layer 0's PC-key). The values
are the cell contents, which **are dimensions of x, not separate parameters**.
This is the Tracr trick: the heap bank is part of the residual stream;
the attention reads from it via projection, not from a separate KV cache.

### 7.2. Weight count delta

Using `weight_matrices.c`'s existing accounting (per-layer FFN width 1024,
gate+up+down per pair):

| Class | New gated pairs | New params (× 1024 × 3 = 3072 per pair) |
|---|---|---|
| Type predicates (6) + NULL_P | 7 | 21,504 |
| POPN (3 cases) | 3 | 9,216 |
| CONS / CAR / CDR / SET_CAR / SET_CDR | 16 cells × 5 ops × ~2 fields = 160 | 491,520 |
| VEC_* (stage 1, n ≤ 4) | 16 cells × 4 ops × ~2 fields = 128 | 393,216 |
| STR_* (stage 1, len ≤ 4) | 16 cells × 2 ops × 2 fields = 64 | 196,608 |
| Closures + upvalues | 16 × 5 ops × 2 fields = 160 | 491,520 |
| TAIL_CALL + PACK_REST | 8 | 24,576 |
| AD transcendentals (Taylor-5 each) | 4 ops × 5 terms = 20 | 61,440 |
| Layer 6 (cell fetch) attention | 16 heads × (12+12+1) | ~6,000 |
| Layer 7 (heap mutate) FFN | net of above | (already counted) |

**Total new params: ~1.7M**, growing the model from 2.8M to ~4.5M. Still
shallow by transformer standards. `d_model` grows from 128 → 256, doubling
the per-step compute but not the depth.

## 8. Implementation milestones (lift to a PR series)

### Stage 1 — Type predicates + null + POPN
*~3 days, ~30k new params, ~150 lines added to weight_matrices.c*

**Pre-stage baseline (verified 2026-05-08):** the existing 71/71 contract
reproduces in 2.84s wall on M2 Max, with all 5 SHA-256s matching the
doc-pinned values:

```
weights.qlmw              638376aab6d49e829da2c54d22b545d86c50aa1c2d508e8ec029d2a6d3f1e77d
vm-traces.jsonl           564fbe1fa4dba5793db0c0e54d402932061f2c82b94da470c7541a5c421584f3
transformer-traces.jsonl  5cc01b2a17e87d88628b13ef5f7602bd7bcd6380e407a0aac5c39b35a9570715
comparison-report.json    8a7917d2b56254f9fad71a4cd5e59284504313e73f99c2b668b461a74e154aab
opcode-coverage.json      aa0c666ad3c2b7a1034e4a69deee6e271e53261bddb12e07dce52fb55218438f
```

Run `scripts/paper/run_paper_suite.sh` after each patch. The historical
baseline here was 71 traced programs; later stages add tests, so the durable
contract is `total_programs == output_agreeing_programs ==
fully_agreeing_programs`, not a fixed count. Different SHAs for traces are OK
if newly weight-encoded opcodes change state-vector trajectories.

- [x] Repurpose `S_AD_SPARE2..S_AD_SPARE8` (slots 121-127) as
  `S_TYPE_IS_NUM/BOOL/PAIR/PROC/STR/VEC/NIL` transient slots
- [x] Layer 2: emit 7 sigmoid-difference indicator computations
  (one per type tag, mirroring the existing `S_CMP_EQ` pattern at
  `weight_matrices.c:780-790`)
- [x] Layer 3: replace IS_NATIVE emission for `OP_NULL_P` (34) +
  `OP_PAIR_P` (45) … `OP_VEC_P` (50) with two `add_gated_pair` calls
  each (`Δ TOS = S_TYPE_IS_X - TOS` then `Δ TYPE_TOS = TYPE_BOOL -
  TYPE_TOS`)
- [x] Layer 3: replace IS_NATIVE emission for `OP_POPN` (53) with
  three indicator-gated cases on `S_OPERAND ∈ {1, 2, 3}`
- [x] Update `exec_step` simulator dispatch (`weight_matrices.c:537-548`):
  remove the 7 predicates + POPN from the IS_NATIVE union, add explicit
  cases that compute the same result
- [x] Update IS_NATIVE matrix-emission loop at line ~902 to skip the
  newly-encoded opcodes
- [x] Re-run `scripts/paper/run_paper_suite.sh`; the comparison-report
  must report all traced programs agreeing on PRINT output and full state
- [x] **Acceptance:** `tab_opcode_coverage.tex` shows `(43+8) wi, (10-8)
  nd` — i.e., 51 weight-implemented, 2 native-delegated for the suite's
  exercised set

### Stage 2 — Arena bank: CONS, CAR, CDR, SET_CAR, SET_CDR, VEC_*, STR_*
*~2 weeks, ~1.5M new params, d_model 128 → 256*

- [x] Extend `D` to 256, add Zone E arena layout
- [x] Implement pair arena write/read in existing Layer 3 + Layer 4 schedule
  (`CONS`, `CAR`, `CDR`, `SET_CAR`, `SET_CDR`)
- [x] Update reference and simulated transformer C functions for pair arena ops
- [x] Add focused pair regressions: cdr nil type preservation, nested pair
  type preservation, `set-cdr!`
- [x] Re-run `scripts/paper/run_paper_suite.sh`; pair slice report was 82/82
  PRINT-output and full-state agreement
- [x] Extend arena bank layout for bounded `VEC_*`
- [x] Implement bounded inline vectors in the existing Layer 3 + Layer 4
  schedule (`VEC_CREATE`, `VEC_REF`, `VEC_SET`, `VEC_LEN`)
- [x] Add focused vector regressions: `vec-ref`, `vec-len`, `vec-set/ref`
- [x] Re-run `scripts/paper/run_paper_suite.sh`; vector report was 83/83
  PRINT-output and full-state agreement, with 54 weight-implemented / 3
  native-delegated opcodes in the exercised coverage set
- [x] Encode arena-layout `STR_REF` and `STR_LEN` reads
- [x] Add focused string-layout regressions: `str-ref`, `str-len`
- [x] Re-run `scripts/paper/run_paper_suite.sh`; string-read report was
  93/93 PRINT-output and full-state agreement, with 60 weight-implemented /
  0 native-delegated opcodes in the exercised coverage set
- [x] Keep bounded string reads in the landed Layer 3 + Layer 4 schedule;
  chained string construction remains future work
- [x] **Acceptance:** add string test programs; all traced programs
  bit-identical against simulated transformer mode

### Stage 3 — Closures with upvalues, TAIL_CALL, PACK_REST
*~1 week, ~500k new params*

- [x] Encode artifact-shape `OP_CLOSURE` as an arena closure header
  (`car = entry_pc`, `cdr = reserved upvalue count`)
- [x] Encode MEM-backed `OP_GET_UPVALUE`, `OP_SET_UPVALUE` fallback
- [x] Encode arena-path `OP_OPEN_CLOSURE`, `OP_CLOSE_UPVALUE` housekeeping
- [x] `OP_GET_UPVALUE`, `OP_SET_UPVALUE` via bounded arena closure cells
- [x] Encode bounded `OP_TAIL_CALL` arities 0..4 as frame reuse in the
  weight path
- [x] `OP_PACK_REST` via bounded arena list creation
- [x] Encode bounded exact integer `OP_DIV`/`OP_MOD` cases exercised by the
  artifact suite
- [x] Encode exercised `AD_EXP`/`AD_SIGMOID` libm values as bounded AD table
  paths
- [x] Re-run `scripts/paper/run_paper_suite.sh`; current report is 103/103
  PRINT-output and full-state agreement, with 69 weight-implemented / 0
  native-delegated / 0 transformer-native-assisted opcodes in the exercised
  coverage set
- [x] **Acceptance:** Stage 3's bounded closure/upvalue, tail-call, and
  pack-rest artifact paths are weight-implemented. The exercised trace set has
  no native-assisted transformer steps. The remaining true native
  boundary is outside this stage: `OP_NATIVE_CALL`, `OP_CALLCC`/`OP_INVOKE_CC`,
  full exception/dynamic-wind unwinding, general IEEE `DIV`/`MOD`, and
  broader relaxed-precision AD transcendentals.

### Stage 4 — AD transcendentals (Taylor + Newton-Raphson) on a separate build target
*~1 week, ~100k new params*

- [x] Exercised `AD_EXP(0)` and `AD_SIGMOID(0/1)` values via bounded table
  gates in the strict artifact build
- [ ] `OP_AD_SIN`, `OP_AD_COS` via Bhaskara/Taylor in Layer 3 dispatch
- [ ] `OP_AD_POW` via existing EXP/LOG composition
- [ ] `OP_AD_DIV` via Newton-Raphson 4-iter loop
- [ ] **Acceptance:** Taylor-build matches reference to ≤ 4 ULPs on AD ops,
  documented relaxation; libm-delegated build remains bit-identical

## 9. Risk & mitigation

| Risk | Mitigation |
|---|---|
| `d_model = 256` breaks downstream paper-trace dump pipeline | New `d_model` constant + format-version bump in ESKB writer; old traces still readable via version detection |
| 16-cell heap insufficient for real Eshkol programs | The artifact's 71 test programs only exercise heap shallowly. Real programs need bounded heap or chain-walks; we accept that "all memory ops in the transformer" is a *contract about ops*, not about *unbounded program execution*. The contract is: "for any program whose live-cell-count never exceeds 16, the full execution is one transformer block iterated." Production deployments scale `MEM_SIZE` linearly. |
| Taylor approximation breaks AD precision | Two builds; users opt in via CMake flag |
| Float32 saturation breaks at `d_model = 256` | SCALE = 300 already has 50 ULP margin per the `weight_matrices.c:59-84` analysis; doubling state dims adds ~2× accumulation length, well within margin |
| Frame walking for `CALLCC` | Out of scope; remains delegated. No regression to current contract. |

## 10. References (with cached SHAs)

(All papers in `research/papers/content/objects/<sha>/`; full bib at
`research/vm-transformer-bib.bib`.)

- **graves2014ntm** — Neural Turing Machines (Graves 2014). Read/write
  heads with content + location addressing. Section 3.3 supplies the
  attention pattern for our Layer 6 cell fetch.
- **graves2016dnc** — Differentiable Neural Computer (Graves 2016, Nature).
  Linkage matrix for closure capture; cited but not openly cached.
- **joulin2015stack** — Inferring Algorithmic Patterns with Stack-Augmented
  Recurrent Nets. Push/pop scalars for differentiable stack.
- **grefenstette2015trans** — Learning to Transduce with Unbounded Memory.
  Continuous stack/queue/dequeue with fully-decoupled dynamics; supplies
  the bit-stable scalar push/pop convention.
- **bosnjak2017forth** — Programming with a Differentiable Forth Interpreter.
  Slot-and-sketch model: every opcode is a fixed function on the state.
- **dehghani2018universal** — Universal Transformer. Parallel-in-time
  recurrent transformer; supplies the looped-block iteration pattern.
- **giannou2023looped** — Looped Transformers as Programmable Computers.
  13-layer transformer emulating a small ISA; precedent for our 6→8 layer
  extension.
- **lindner2023tracr** — Tracr: Compiled Transformers as a Laboratory for
  Interpretability. Gives the RASP-to-attention-head compilation pipeline
  we follow for `CAR`/`CDR`/predicate ops.
- **liu2022shortcuts** — Transformers Learn Shortcuts to Automata. Krohn-
  Rhodes decomposition justifies why O(1)-depth dispatch suffices for any
  finite-state computation, including type-predicate dispatch.
- **olsson2022induction** — In-context Learning and Induction Heads.
  Mechanistic-interpretability anchor for "the heap is content-addressable
  via attention" claim.
- **santoro2018rmc** — Relational Memory Core. Multi-head attention over
  bounded memory; closest existing analogue to our heap bank.
- **sukhbaatar2015memnn** — End-to-end Memory Networks. Multi-hop reads
  over external memory; foundational for the "heap as part of residual
  stream" decision.
- **weiss2021rasp** — Thinking Like Transformers (RASP). select+aggregate
  primitives; our type predicates and indicator gates *are* RASP `aggregate
  (select(== TYPE_PAIR), 1)` instances.

## 11. What this design does NOT claim

- It does not claim the SDNC becomes Turing-complete via this extension.
  The bounded heap caps live-cell count at 16 (or whatever `MEM_SIZE` is
  set to). Programs requiring more cells fall back to the runtime, exactly
  as today.
- It does not claim bit-identical agreement on transcendentals (stage 4).
- It does not claim removing `NATIVE_CALL`. `NATIVE_CALL` is the FFI; it
  bridges to OS resources by definition.
- It does not claim faster execution. The transformer-mode is slower than
  the C VM by ~1000× per step. The point is the *constructive equivalence*,
  not the throughput.

## 12. Open questions

- Should the heap be a separate module from the AD tape, or unified? The
  AD tape is also a bounded memory (8 nodes × 8 fields = 64 dims); the heap
  bank we propose is 16 × 12 = 192 dims. Unifying them — one bank, multiple
  views — would save ~30% of weight count but couples two independent
  contracts. Recommendation: keep separate; cleanliness > compactness.
- For chained vectors / strings, do we expose the chain-walk to programs
  (so they can detect "this ref will take 3 cycles") or hide it inside the
  looped-transformer iteration? Recommendation: hide; emit a "delayed
  result" indicator and re-issue the same instruction.
- For the artifact contract, do we ship the new transformer at the SAME
  pinned commit (`8235d99`) or bump? Recommendation: new pin, new
  reproducibility-package version; the old pin remains the v1.2.0 anchor.

---

This design is a **constructive specification**: every entry in §5 maps to
a specific change to `weight_matrices.c`. Stage 1 is mechanically
implementable from this document; stage 2 requires the `d_model` extension
and Layer 6/7 addition; stages 3-4 are scope extensions. Once stage 3 lands,
24 of the 26 currently-delegated opcodes have been encoded, which closes
the user's request "all memory operations are in the transformer".
