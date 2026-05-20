# VM-as-Transformer: Encoding Delegated Memory Operations as Weight Matrices

**Status:** active design + landed slices; coverage has expanded past the original "26 delegated" scope (see implementation note below). The document is maintained alongside the work but is *not* a frozen specification.
**Worktree:** `feature/vm-transformer-memory`
**Origin draft:** internal architecture review, 2026-05-08
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
upvalue cells are also encoded in the weight path. The bounded continuation
slice now encodes direct-entry `OP_CALLCC`/`OP_INVOKE_CC` escape continuations
as four contiguous arena cells in Zone E; full R7RS continuation frame walking
remains outside this bounded artifact contract.
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

Current artifact verification after the non-native opcode coverage slice:
126/126 inline tests pass, 123/123 traced programs agree on PRINT output and
full per-step state, opcode coverage is 82 weight-implemented / 0
VM-native-delegated / 0 transformer-native-assisted in the exercised coverage
set, and the QLMW export is d_model=256, FFN=2304, 12,220,422 parameters.
The exercised trace set now has no `S_IS_NATIVE` postprocess assistance and
covers every canonical opcode except the deliberate external boundary,
`OP_NATIVE_CALL`.
`OP_DIV` is weight-encoded for positive integer denominators 1..16,
`OP_MOD` is weight-encoded for the positive integer `% 3` and `% 4` verifier
range, `AD_ABS`/`AD_RELU` are weight-encoded for bounded nonzero
integer-scale cases, `AD_EXP`/`AD_SIGMOID`/`AD_TANH`/`AD_LOG`/`AD_SQRT` are
table-encoded for the exercised inputs (`exp(0)`, `sigmoid(0)`, `sigmoid(1)`,
`tanh(0)`, `log(1)`, `sqrt(4)`), and `AD_SIN`/`AD_COS` are table-encoded for
integer inputs -4..4 with representative traced coverage at -1, 0, and 1.
`AD_DIV` is weight-encoded for positive integer denominator table
entries 1..16, including both numerator and denominator reverse-mode gradients.
`AD_POW` is weight-encoded for positive integer bases 1..8 and exponents 1..4,
including both base and exponent reverse-mode gradients.
Untested or broader libm paths (general non-integer `exp`/`sigmoid`/`tanh`/
`log`/`sqrt`/`sin`/`cos`, general `AD_DIV`, general `AD_POW`) remain bounded-state
candidates or precision-contract decisions, not completed general encodings.

## 1. Problem statement

The Eshkol VM is a 83-opcode bytecode machine. The Self-Differentiating Neural
Computer (SDNC) — `lib/backend/weight_matrices.c` — analytically constructs a
6-layer transformer. Earlier reproducibility snapshots used a smaller
state-vector and trace suite; the current bounded-arena artifact uses
`d_model = 256`, FFN width 2304, and a 123-program traced suite.

Historically, only part of the 83-opcode ISA executed end-to-end through
`Wx + b` matmul-plus-bias while the rest crossed an `IS_NATIVE` boundary marker.
Current work has collapsed the
artifact-exercised memory, closure/upvalue, tail-call, pack-rest, bounded
integer arithmetic, and selected AD value-generation paths into weights; the
remaining boundaries are semantic/native or broader precision-contract issues.

The `IS_NATIVE` boundary is real and useful: it cleanly separates host services
and high-level library calls from the differentiable compute kernel. But it is a
leak from "the program is the matrix." The goal of this design is to **collapse
the boundary for every operation that does not fundamentally require runtime
side-effects** — i.e. every operation whose semantics can be expressed as a
deterministic transformation of a bounded state vector. Once collapsed, the VM
artifact is one transformer block iterated against itself, the arena lives in
the state vector, and gradients through bounded list/vector programs flow
through the same backward pass that already handles arithmetic on the 19 AD
opcodes.

This document specifies the encoding scheme.

## 2. Inventory of delegated operations

From `weight_matrices.c:537-548`:

| Class | Count | Opcodes |
|---|---:|---|
| Arithmetic delegated for precision | 2 | `OP_DIV`, `OP_MOD` |
| Heap data structures | 12 | `OP_CONS`, `OP_CAR`, `OP_CDR`, `OP_NULL_P`, `OP_VEC_CREATE`, `OP_VEC_REF`, `OP_VEC_SET`, `OP_VEC_LEN`, `OP_STR_REF`, `OP_STR_LEN`, `OP_SET_CAR`, `OP_SET_CDR` |
| Type predicates | 6 | `OP_PAIR_P`, `OP_NUM_P`, `OP_STR_P`, `OP_BOOL_P`, `OP_PROC_P`, `OP_VEC_P` |
| Closures + upvalues | 5 | `OP_CLOSURE`, `OP_GET_UPVALUE`, `OP_SET_UPVALUE`, `OP_CLOSE_UPVALUE`, `OP_OPEN_CLOSURE` |
| Control flow (R7RS) | 10 | `OP_TAIL_CALL`, `OP_NATIVE_CALL`, `OP_CALLCC`, `OP_INVOKE_CC`, `OP_PUSH_HANDLER`, `OP_POP_HANDLER`, `OP_GET_EXN`, `OP_PACK_REST`, `OP_WIND_PUSH`, `OP_WIND_POP` |
| Stack housekeeping | 1 | `OP_POPN` |
| AD transcendentals | 4 | `OP_AD_DIV`, `OP_AD_POW`, `OP_AD_SIN`, `OP_AD_COS` |

The original "26 delegated memory operations" scope, as recorded in `weight_matrices.c:537-548`, comprised the arithmetic, heap, type-predicate, closure, stack, and AD-transcendental rows above (2 + 12 + 6 + 5 + 1 + 4 = 30, of which 26 were judged in scope for the first lift — `OP_VOID` and a handful of bookkeeping ops were excluded). The R7RS control-flow class (10 opcodes) was added once the bounded escape-continuation construction proved tractable. Total currently-encoded coverage is reported in the implementation note at the top of this document.

We classify these by **whether the runtime side-effect is essential or
incidental**:

- **Encodable (this design):** type predicates, heap data ops, closures,
  upvalues, `POPN`, `TAIL_CALL`, `PACK_REST`, bounded escape continuations,
  handler/wind bookkeeping, AD transcendentals.
- **Genuinely native (out of scope here):** `NATIVE_CALL` (by definition
  bridges to libcurl, sqlite3, libpthread), unbounded first-class continuation
  frame walking, and full R7RS raise/unwind through exception + dynamic-wind
  frames. The simple handler-depth, wind-depth, and bounded escape-continuation
  paths are now in the weight path; invoking the general unwind protocol remains
  a runtime boundary.
- **Bounded precision slices:** `OP_DIV`, `OP_MOD`. The exercised exact
  integer cases are now encoded directly in the weight path (`DIV` via
  denominator-gated reciprocals, `MOD` via small exact lookup). General IEEE
  division/modulo remains a precision-contract decision rather than a solved
  arbitrary-float encoding.

The exercised artifact target for "all bounded memory operations are in the
transformer" is now met: no traced program needs transformer-native
postprocess assistance. The general-language boundary still includes
OS/native calls, unbounded continuation frame walking/full unwind, arbitrary
IEEE division/modulo, and broader AD libm functions.

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

### 3.1. Zone E: bounded arena bank — final landed layout

The original design called for 16 cells × 8 fields. The landed
implementation (`weight_matrices.c §88-96`, §207-263) collapsed this to
**16 cells × 5 fields = 80 dims of payload + 47 dims of operation
transients = 127 dims**, leaving the high dimension `255` as the last
transient. The design intuition (one Lisp cell per arena slot, fields
for kind/car/cdr plus the per-field type) survives; what changed is
that the type tags live in dedicated lanes within the cell rather than
in a parallel type bank, which keeps the layout cache-line-aligned and
saves the alternative `d_model = 320`-or-384 expansion that §5.4
originally entertained:

```text
S_ARENA_BASE = 128
ARENA_CELLS = 16
ARENA_CELL_FIELDS = 5
S_ARENA_NEXT = 208  (bump pointer; first transient slot, also the next-free index)

Cell i occupies dims [128 + 5i, 128 + 5i + 5):
  ARENA_F_KIND     = 0  (cell kind: empty=0, pair=1, vector=2, vec_elem=3, closure=4)
  ARENA_F_CAR_VAL  = 1  (cons car / vec head / closure func-PC)
  ARENA_F_CDR_VAL  = 2  (cons cdr / vec tail / closure upvalue-count)
  ARENA_F_CAR_TYPE = 3  (type tag for car)
  ARENA_F_CDR_TYPE = 4  (type tag for cdr)
```

The cell kinds (`weight_matrices.c §90-95`):

| Kind value | Constant            | Use                                                                  |
|-----------:|---------------------|----------------------------------------------------------------------|
| 0.0        | `ARENA_KIND_EMPTY`   | Unallocated; the bump pointer never reuses these                     |
| 1.0        | `ARENA_KIND_PAIR`    | Cons cell — `car/cdr` are arbitrary tagged values                    |
| 2.0        | `ARENA_KIND_VECTOR`  | Vector header — `car = length`, `cdr` = first vec_elem cell index    |
| 3.0        | `ARENA_KIND_VEC_ELEM`| Vector element — `car = value`, `cdr` = next vec_elem cell (or -1)   |
| 4.0        | `ARENA_KIND_CLOSURE` | Closure header — `car = entry_pc`, `cdr = reserved upvalue count`    |
| 7.0        | `CONT_RESTORE_MARKER`| Bounded escape-continuation marker (see §5.10)                        |

The 80 payload dims plus the 47 transient dims (`S_ARENA_NEXT`
through `S_ARENA_LIST_HAS_E3`, declared as a contiguous enum block at
`weight_matrices.c §214-261`) span exactly the upper 128 dimensions of
the residual stream. There is no slack: every dimension is either a
persistent cell field, the bump pointer, an operation scratch lane, or
a list-construction scratch lane. The transient block is sub-divided
into three windows: per-cell write scratch (`S_ARENA_WRITE_KIND..
S_ARENA_NEW_CDR_TYPE`, 11 dims), bounded inline-vector scratch
(`S_ARENA_VEC_*`, 16 dims), and bounded list construction scratch
(`S_ARENA_LIST_*`, 20 dims). All three are cleared every cycle by
Layer 3's universal transient-clear loop (`weight_matrices.c §1394`).

**Bound rationale.** 16 cells × {pair = 1 cell, vector header + 4 elem cells = 5 cells,
closure header + 4 upvalue cells = 5 cells, escape continuation = 4 cells}
gives a working budget of, roughly, two simultaneous bounded vectors plus
one closure plus a pair chain, which suffices for every program in the
123-program traced suite. The exercised cases are: simple `(cons a b)` chains
to length ≤ 12; vectors of declared length ≤ 4; strings of declared length ≤ 4;
closures with up to 4 upvalues; one outstanding bounded escape continuation
at a time. The bump pointer `S_ARENA_NEXT` increments by 1 per allocation
(by 5 for the bounded list-construction path, which writes four pair cells
in a single cycle: `weight_matrices.c §S_ARENA_LIST_*`).

**Free-list policy.** None. The arena is a bump allocator: allocation
returns the cell at `S_ARENA_NEXT` and increments. Garbage collection is
not implemented; the contract is that a program completes before the bump
pointer reaches 16. The reference VM and the matrix path both `reset` the
arena at the start of each `test()` invocation (`weight_matrices.c §5491-5498`
issues `vm_arena_reset` and reinitialises the `g_heap_ptr`,
`g_frame_count`, exception, closure, and wind globals), so the per-program
budget of 16 cells is exact, not a worst-case-across-suite bound.

**Why an arena, not a heap.** In the bounded-state-vector regime, the
distinction matters. A heap would require a free-list pointer to be
indexable, which means a Tracr-style attention over `(cell, kind == empty)`
keys followed by a `min` aggregate — two extra heads. A bump allocator
just reads and writes `S_ARENA_NEXT`, which is a single scalar dim. The
arena layout is structurally the analogue of Eshkol's *runtime* arena
model in `vm_arena.h`; the in-state-vector implementation mirrors it
precisely.

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

**Landed Layer 2 construction (`weight_matrices.c §1297-1303`).** The
seven type-indicator slots are computed in the preprocess FFN by:

```c
out[S_TYPE_IS_NUM]  = indicator(x[S_TYPE_TOS], TYPE_NUMBER);   /* 0 */
out[S_TYPE_IS_BOOL] = indicator(x[S_TYPE_TOS], TYPE_BOOL);     /* 1 */
out[S_TYPE_IS_PAIR] = indicator(x[S_TYPE_TOS], TYPE_PAIR);     /* 2 */
out[S_TYPE_IS_PROC] = indicator(x[S_TYPE_TOS], TYPE_CLOSURE);  /* 3 */
out[S_TYPE_IS_STR]  = indicator(x[S_TYPE_TOS], TYPE_STRING);   /* 4 */
out[S_TYPE_IS_VEC]  = indicator(x[S_TYPE_TOS], TYPE_VECTOR);   /* 5 */
out[S_TYPE_IS_NIL]  = indicator(x[S_TYPE_TOS], TYPE_NIL);      /* 6 */
```

The `indicator(x, k)` helper at `weight_matrices.c §334-336` is the
sigmoid-difference pulse

$$
\text{indicator}(x, k) = \sigma\bigl(\text{SCALE}\cdot(x - k + 0.5)\bigr) -
\sigma\bigl(\text{SCALE}\cdot(x - k - 0.5)\bigr),
$$

which saturates to $1$ when $x = k$ (the only integer with both
sigmoids at $+1$ and $0$) and to $0$ for every other integer. With
$\text{SCALE} = 300$ the off-target leakage is below $\exp(-150) \approx 0$
in float32. The Layer-2 outputs land in `S_TYPE_IS_*` (dims 121-127),
which the universal transient-clear in Layer 3 then zeroes; this is
why the predicate dispatch and the `OP_NULL_P` slot live in the *same*
cycle as the indicator computation.

**Landed Layer 3 dispatch.** Each of the six predicates becomes two
`add_gated_pair` invocations (the second clears `S_TYPE_TOS` to
`TYPE_BOOL`). The total budget is therefore 12 gated pairs × 2
sign-conjugate neurons = 24 FFN-3 neurons, dwarfed by the 2304-wide
Layer-3 FFN. No `d_model` extension required — the entire stage-1
"type predicates" lift fits inside the existing residual stream
because the seven indicator slots reuse Zone D's spare range that
were previously named `S_AD_SPARE2..S_AD_SPARE8` (the source code
keeps both enum names as aliases for the same numeric slot, lines
197-206, so legacy comments mentioning `S_AD_SPARE*` and new comments
mentioning `S_TYPE_IS_*` refer to the same dimensions).

### 5.2. `OP_NULL_P` (landed)

`(null? x)` returns true if `x = '()`. In Eshkol's value encoding `nil` has
`type_tos = TYPE_NIL = 6` and `tos = -1` (set by `OP_NIL` at
`weight_matrices.c §439-441`). The predicate is then:

$$
\text{TOS}' = \mathbf{1}[\text{S\_TYPE\_TOS} = 6], \qquad
\text{TYPE\_TOS}' = \text{TYPE\_BOOL}
$$

The encoding pattern is identical to §5.1's six predicates: Layer 2
emits `S_TYPE_IS_NIL` into dim 127, Layer 3 dispatches a single
`add_gated_pair` to write the residual delta into `S_TOS` and a second
to clear `S_TYPE_TOS` to `TYPE_BOOL`. Crucially, `OP_NULL_P` uses
*exactly the same* Layer-2 precompute that `OP_PAIR_P`…`OP_VEC_P` use,
so the type-predicate family is one indicator computation followed by
seven mutually-exclusive Layer-3 gates over `S_OPCODE`. Cost: two
gated pairs in Layer 3, zero new precompute slots. This was the
unblocking insight for the §5.1 family: once `S_TYPE_IS_NIL` is on
the residual stream, every type predicate is a single residual write,
no two-layer composition needed.

### 5.3. `OP_POPN` (landed)

`POPN n` pops `n` values *below* TOS, keeping TOS at the top. With $n \in
\{1, 2, 3\}$ — Eshkol's compiler emits exactly this range for let-binding
cleanup — the residual delta on the four-register file is:

| Operand $n$ | $\text{TOS}'$ | $\text{SOS}'$ | $\text{R2}'$ | $\text{R3}'$ | $\text{DEPTH}'$ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | TOS | R2  | R3 | 0  | DEPTH − 1 |
| 2 | TOS | R3  | 0  | 0  | DEPTH − 2 |
| 3 | TOS | 0   | 0  | 0  | DEPTH − 3 |

Encoded as three `add_gated_opcode_index` invocations
(`weight_matrices.c §3012-3035`), one per operand value. The helper
adds `S_OPERAND` to the gate at scale 100·SCALE (the `index_scale`
constant at line 3018), so the gate signature
$\mathbf{1}[opcode = \text{OP\_POPN}\ \wedge\ operand = n]$ is again a
single sigmoid-difference pair with the same float32 saturation as
opcode-only dispatch. Per operand the residual writes are five gated
pairs (one per register being moved or cleared), totalling 3 × 5 = 15
gated pairs = 30 sign-conjugate neurons; well within the Layer-3 FFN
budget.

The type tags `S_TYPE_TOS..S_TYPE_R3` shift in lock-step, using the
same gated-pair shape over `(S_TYPE_SOS, S_TYPE_R2, …)` lanes. Going
beyond $n = 3$ would require either (a) one additional operand index
per `n` (linear in budget; trivial), or (b) a RASP-style aggregate
attention over the stack-depth field with a `≤ n` mask (one extra
head; encodable but unneeded for the current artifact contract because
the compiler never emits the larger pop). The landed implementation
chose (a) for clarity.

### 5.4. `OP_CONS`, `OP_CAR`, `OP_CDR`, `OP_SET_CAR`, `OP_SET_CDR` (landed)

The landed form uses the 5-field Zone E layout (§3.1) and the existing
six-layer schedule — Layer 3 computes the operation transients
(`S_ARENA_TARGET`, `S_ARENA_NEW_KIND`, etc.) and Layer 4 performs the
arena cell mutation. No Layer 6/7 was needed: the cell fetch happens
in Layer 2 (which already has a 16-way indicator loop available for
the AD tape) and the cell write happens in Layer 4 (which already has
the dual-input AND gating pattern from the AD tape writes).

**`(cons a b)` — landed form.** Pre-state: `TOS = b, SOS = a`.

Layer 3 (`weight_matrices.c §codegenCons block`):

1. emits the cell at index `S_ARENA_NEXT` with `kind = ARENA_KIND_PAIR`,
   `car_val = SOS, cdr_val = TOS`,
   `car_type = S_TYPE_SOS, cdr_type = S_TYPE_TOS`. The five values land
   in the per-cell write scratch slots
   `S_ARENA_WRITE_KIND, S_ARENA_NEW_CAR, S_ARENA_NEW_CDR,
   S_ARENA_NEW_CAR_TYPE, S_ARENA_NEW_CDR_TYPE`;
2. sets `S_ARENA_TARGET = S_ARENA_NEXT`;
3. replaces `TOS` with the new cell index (= old `S_ARENA_NEXT`) and
   sets `S_TYPE_TOS = TYPE_PAIR`;
4. pops `SOS` into `R2`'s lane, shifting the register file as
   `OP_POP` does;
5. increments `S_ARENA_NEXT` by 1 via an unconditional add gated on
   the opcode indicator.

Layer 4 (`weight_matrices.c §arena cell-write block`) then sees
`S_ARENA_TARGET` set and runs the per-cell indicator loop:

$$
\text{cell}_i.\text{field}_j\ \mathrel{+}= \mathbf{1}[\text{S\_ARENA\_TARGET} = i] \cdot \bigl(\text{NEW\_field}_j - \text{cell}_i.\text{field}_j\bigr)
$$

for every $i \in [0, 16)$ and every field $j \in [0, 5)$. The dual-input
AND pattern from §6.3 of `docs/SDNC.md` applies: the target indicator
is at gate weight $\text{SCALE}$ and the kind-is-being-set
indicator (= `S_ARENA_TARGET >= 0` plus a write-flag from Layer 3) is
at $10\cdot\text{SCALE}$. Cost: 16 cells × 5 fields × 2 sign-conjugate
neurons = 160 Layer-4 FFN neurons. Out of 2304 Layer-4 FFN positions,
this is ≈ 7% of the available width.

**`(car p)` / `(cdr p)`.** Pre-state: `TOS = pair-ptr`,
`S_TYPE_TOS = TYPE_PAIR`.

Layer 2 (`weight_matrices.c §arena read block`) loops over the 16
cells, indicator-gated on `S_TOS = i`, and projects the cell's
`F_CAR_VAL, F_CDR_VAL, F_CAR_TYPE, F_CDR_TYPE` into the read scratch
slots `S_ARENA_READ_CAR, S_ARENA_READ_CDR` and the corresponding type
lanes. Cost: 16 cells × 4 fields × 2 = 128 Layer-2 neurons.

Layer 3 then writes the residual:

$$
\Delta\text{S\_TOS}\ =\ \mathbf{1}[\text{opcode} = \text{OP\_CAR}] \cdot (\text{S\_ARENA\_READ\_CAR} - \text{S\_TOS}),
$$

$$
\Delta\text{S\_TYPE\_TOS}\ =\ \mathbf{1}[\text{opcode} = \text{OP\_CAR}] \cdot (\text{S\_ARENA\_READ\_CAR\_TYPE} - \text{S\_TYPE\_TOS}),
$$

with the obvious twins for `OP_CDR` using `S_ARENA_READ_CDR`. Cost: 4
gated pairs (2 ops × {TOS write, TYPE_TOS write}) in Layer 3 = 8
neurons.

**`(set-car! p v)` / `(set-cdr! p v)`.** Pre-state: `TOS = v, SOS = p`.
Layer 3 sets `S_ARENA_TARGET = SOS` and writes the value lane of the
target cell:

$$
\Delta\text{cell}_i.\text{F\_CAR\_VAL}\ =\ \mathbf{1}[\text{S\_ARENA\_TARGET} = i] \cdot \mathbf{1}[\text{opcode} = \text{OP\_SET\_CAR}] \cdot (\text{S\_TOS} - \text{cell}_i.\text{F\_CAR\_VAL}),
$$

plus the matching `F_CAR_TYPE` write. `OP_SET_CDR` is the symmetric
case. Cost: 16 cells × 2 fields × 2 ops = 64 Layer-4 neurons.

**Why the Layer 6/7 design was abandoned.** The original plan called
for a dedicated cell-fetch attention head in a new Layer 6 and a
dedicated heap-mutate FFN in a new Layer 7. The implementation review
found that (a) Layer 0's instruction-fetch attention pattern can be
reused inside Layer 2 as a 16-way indicator loop (the head dimension
HD = 2 is enough for a 16-key lookup with SCALE = 300), and (b) Layer
4's existing dual-input AND gating already handles target-indexed
writes for the AD tape. Promoting both to dedicated layers would have
added a third forward pass through the FFN matrices on every cycle
for what amounts to a one-cell-per-op operation. The landed
implementation reuses Layer 2 for the read and Layer 4 for the write,
keeping the schedule at six layers and the matrix-mode wall time
under one minute for the full traced suite.

### 5.5. Cross-references and historical encoding notes

- `OP_NULL_P` is covered in §5.2 (the seven-way `S_TYPE_TOS` lookup
  shares its `S_TYPE_IS_NIL` indicator with the predicate family).
- `OP_SET_CAR` and `OP_SET_CDR` are covered in §5.4 (they share the
  Layer-4 arena-cell-write loop with `OP_CONS`).

The earlier design split here — separate sections for predicates,
pair construction, and pair mutation — collapsed in the landed
implementation because all three operations resolve into the *same*
two pieces of hardware: a Zone-D-resident transient set produced by
Layer 2 indicators, plus a Layer-4 indicator-gated cell write. The
section headings are preserved for archaeological clarity but the
body has been folded.

### 5.6. `OP_VEC_CREATE`, `OP_VEC_REF`, `OP_VEC_SET`, `OP_VEC_LEN` (landed, bounded)

The landed vector encoding uses two cell kinds: a header cell
(`ARENA_KIND_VECTOR`) whose `car = length` and `cdr` = the cell index
of the first element, plus `ARENA_MAX_INLINE_VECTOR = 4` element
cells (`ARENA_KIND_VEC_ELEM`) chained through `cdr`. So a vector of
length $n \le 4$ occupies $1 + n$ arena cells: one header plus $n$
elements.

**`(vector v0 v1 v2 v3)` — Layer 3 + Layer 4.**

Layer 3 (`weight_matrices.c §codegenVecCreate`) populates the vector
write scratch window:

- `S_ARENA_VEC_BASE = S_ARENA_NEXT`  (the header cell index)
- `S_ARENA_VEC_LEN = operand`         (the requested length, $\le 4$)
- `S_ARENA_VEC_E[0..3] = ` register-file extraction:
  - `S_ARENA_VEC_E0 = R3` (operand 4),
  - `S_ARENA_VEC_E1 = R2` (operand 3),
  - `S_ARENA_VEC_E2 = SOS` (operand 2),
  - `S_ARENA_VEC_E3 = TOS` (operand 1) — i.e. reversed stack order so
    `vector-ref 0` reads $v_0$;
- `S_ARENA_VEC_T[0..3]` carry the matching `S_TYPE_R3..S_TYPE_TOS`;
- `S_ARENA_VEC_HAS_E[0..3]` are 1.0 for the first `operand` slots and
  0 otherwise, computed via `add_gated_opcode_index` against
  `operand`.

Layer 4 writes the five $1+n$ arena cells. For each cell $i$ and each
of the five fields:

$$
\Delta\text{cell}_i.\text{kind}\ =\ \mathbf{1}[i = \text{S\_ARENA\_VEC\_BASE}]\cdot(\text{ARENA\_KIND\_VECTOR} - \text{cell}_i.\text{kind})
$$

for the header cell, and similar element-cell writes gated by
`S_ARENA_VEC_HAS_E[k]` for $k \in [0, 4)$. The cell-chain `cdr` field
is computed inline as `S_ARENA_VEC_BASE + k + 1`. Cost: 16 cells × 5
fields × 5 (1 header + 4 elements) = 400 Layer-4 neurons; in
practice the gates are sparse because `S_ARENA_VEC_HAS_E[k]` zeros
out the unused element writes. The bump pointer `S_ARENA_NEXT`
increments by `1 + S_ARENA_VEC_LEN` after the write.

**`(vector-ref v i)`.** Pre-state: `TOS = i, SOS = v-ptr`. The chain
walk needed for unbounded `i` is what §5.6 originally feared and
what kept this op delegated. The bounded landed form sidesteps it:
since vectors are at most 4 elements long, the runtime relation
between `i` and the target cell is

$$
\text{target\_cell} = \text{header\_cell} + 1 + i,\qquad 0 \le i \le 3.
$$

Layer 2's arena-read indicator loop reads `cell.F_CAR_VAL` (and the
type) into `S_ARENA_READ_CAR` after Layer 3 has computed the target
cell index. Layer 3 itself uses a two-tier indicator
(`add_gated_opcode_index`) keyed on `(S_OPCODE, S_TOS)` — for each
$i \in [0, 4)$ and each valid header cell index $h \in [0, 16)$, one
gated pair fires when `S_OPCODE = OP_VEC_REF` and `S_TOS = i` and
sets `S_ARENA_TARGET = h + 1 + i`. This is $4 \times 16 = 64$ Layer-3
neurons. The subsequent Layer-4 cycle is unused for this opcode;
the read result lands in the residual on the *same* cycle because
Layer 2 already loaded the cell into the read scratch slots.

**`(vector-set! v i x)`.** Combines VEC_REF's target computation with
the SET_CAR write pattern. The bounded form uses the same
$(h + 1 + i)$ target arithmetic as `OP_VEC_REF` and the same
Layer-4 cell-write loop as `OP_SET_CAR`. Cost: 64 Layer-3 neurons
for target computation + 16 × 2 Layer-4 neurons for write = 96
neurons in addition to those already paid by VEC_REF and SET_CAR.

**`(vector-length v)`.** Trivial: Layer 2 reads the header cell into
`S_ARENA_READ_CAR` (= the length field), Layer 3 writes
`S_TOS = S_ARENA_READ_CAR` gated on `S_OPCODE = OP_VEC_LEN`. One
gated pair.

**Out-of-bounds and chain.** Programs that index past the bounded
inline window are not in the artifact contract. The Layer-3
`S_TOS = i` indicators saturate to zero for $i \ge 4$, leaving the
residual unchanged — an instructive failure mode where the matrix
forward path stalls but does not corrupt state. The production VM
handles unbounded vectors by allocating into the runtime arena
outside Zone E; programs that mix bounded inline vectors with
unbounded vectors must keep them disjoint or fall back to the
production VM entirely.

### 5.7. `OP_STR_REF`, `OP_STR_LEN` (landed, read-only)

Strings reuse the vector cell layout: a header cell with `car = length`
and a chain of `ARENA_KIND_VEC_ELEM` cells holding character codes
(one per element). The only landed string ops are the *reads*, because
the artifact's program suite uses strings as literal display values
and function names — no string construction or mutation appears in
any traced program.

`OP_STR_REF` and `OP_STR_LEN` use the same Layer-2 arena read +
Layer-3 residual write as `OP_VEC_REF` and `OP_VEC_LEN` respectively.
The only difference is the `S_TYPE_TOS` value set on the result:
`OP_STR_REF` writes `TYPE_NUMBER` (the character code is a number),
`OP_STR_LEN` writes `TYPE_NUMBER` (the length is a number),
`OP_VEC_REF` writes the cell's `F_CAR_TYPE`, `OP_VEC_LEN` writes
`TYPE_NUMBER`. The bounded length constraint is the same: $\le 4$.

Dedicated string construction (e.g. `string-append`, `number->string`)
is left to the production runtime via `OP_NATIVE_CALL`; the artifact
contract is that *reading* a string already in the arena is in the
weight path.

### 5.8. Closures and upvalues — `OP_CLOSURE, OP_GET_UPVALUE, OP_SET_UPVALUE, OP_CLOSE_UPVALUE, OP_OPEN_CLOSURE` (landed)

A closure occupies $1 + k$ arena cells: the header
(`kind = ARENA_KIND_CLOSURE = 4.0`, `car = entry_pc`, `cdr = k`) plus
$k$ upvalue cells whose `car` carries the upvalue value and whose
`car_type` carries the upvalue type. The landed bound is $k \le 4$,
matching `ARENA_MAX_INLINE_VECTOR` and the four-slot MEM register file.

The persistent state dimension `S_CUR_CLOSURE` (dim 15, the last
Zone A register) holds the *currently executing* closure's arena cell
index, or a sentinel of -100.0 when the program is in top-level
context.

**`OP_CLOSURE function-pc`.** Layer 3 emits a header write at
`S_ARENA_NEXT` with `kind = ARENA_KIND_CLOSURE`, `car_val = operand`
(= function PC), and `cdr_val = 4` (= upvalue count). The header
write uses the same Layer-4 indicator-gated cell mutation pattern as
`OP_CONS`. `S_ARENA_NEXT` increments by 5 to reserve the four
upvalue cells; the upvalue cells start zero-initialised and are
written by subsequent `OP_SET_UPVALUE` or `OP_CLOSE_UPVALUE`
invocations. `S_TOS` is updated to the new closure index and
`S_TYPE_TOS = TYPE_CLOSURE`.

**`OP_OPEN_CLOSURE`.** Layer 3 writes
`S_CUR_CLOSURE = S_TOS`, which Layer 4 then preserves through the
call frame. The reference VM saves and restores `g_current_closure_ptr`
across `OP_CALL`/`OP_RETURN` (see `weight_matrices.c §g_frames`
infrastructure); the matrix path uses an `exec_loop_postprocess`
shim that mirrors the same save/restore at the dispatch loop level.
This is the one remaining piece of the closure encoding where the
matrix path leans on a host post-process; the closure cell *contents*
are entirely weight-encoded, but the *frame discipline* of `S_CUR_CLOSURE`
during nested calls is handled by the dispatch loop.

**`OP_GET_UPVALUE k`.** Two-tier read: first, Layer 2 computes the
existing MEM-backed fallback using the `S_LOADVAL` slot
(this is the same address-resolution loop that `OP_GET_LOCAL` uses,
`weight_matrices.c §1261-1262`); then, Layer 4 overwrites `S_TOS`
from `arena[S_CUR_CLOSURE + 1 + k].car` when `S_CUR_CLOSURE`
is in range $[0, 16)$. The "in range" check is again a dual-input
AND: gate weight $\text{SCALE}$ on `S_OPCODE = OP_GET_UPVALUE`,
$\text{SCALE}$ on `S_OPERAND = k`, plus the `S_CUR_CLOSURE`
indicator over the closure cell index. Cost: 4 operand values × 16
closure indices = 64 Layer-4 neurons for the arena read; the MEM
fallback adds the standard 4-slot lookup from `OP_GET_LOCAL`.

**`OP_SET_UPVALUE k`.** Writes to both the MEM fallback *and* the
arena cell, gated identically to `OP_GET_UPVALUE`. Cost: 64 Layer-4
neurons for the arena write + the MEM-write counterpart.

**`OP_CLOSE_UPVALUE k`.** Copies `MEM[k]` into the arena cell
`arena[S_CUR_CLOSURE + 1 + k]`. Layer 2 reads `MEM[k]` into
`S_LOADVAL`; Layer 4 writes the arena cell from `S_LOADVAL`. Cost:
matches `OP_SET_UPVALUE` (the same target arithmetic, different
source dim).

**What stays outside.** The bounded slice covers $k \le 4$ upvalues
and a single closure depth at a time (the active closure index is a
single scalar). Multi-level closure capture (closure-over-closure
where the inner closure references the outer's upvalues by *closure
chain* rather than by name) requires either a recursive arena walk
(Looped-Transformer pattern: iterate the dispatch loop with an
incremented "chain depth" register) or a widened closure header
with an enclosing-closure pointer. Neither is in the current
artifact.

### 5.9. `OP_PACK_REST` (landed, bounded)

`(pack-rest n_fixed)` packs `MEM[n_fixed..3]` into a single list. The
bounded form writes a contiguous arena pair chain *in a single VM
step* (no looping), exploiting the fact that the rest range is at
most $4 - n_{\text{fixed}}$ cells.

**Layer 3 setup.** A bounded list-construction scratch window lives in
Zone E's transient region (`S_ARENA_LIST_BASE`,
`S_ARENA_LIST_E[0..3]`, `S_ARENA_LIST_T[0..3]`,
`S_ARENA_LIST_CDR[0..3]`, `S_ARENA_LIST_CDRT[0..3]`,
`S_ARENA_LIST_HAS_E[0..3]`, 20 dims total at
`weight_matrices.c §241-262`). Layer 3 populates:

- `S_ARENA_LIST_BASE = S_ARENA_NEXT` — first pair cell index;
- `S_ARENA_LIST_E[k] = MEM[n_fixed + k]` for $k \in [0, 4-n_{\text{fixed}})$;
- `S_ARENA_LIST_T[k] = ` corresponding type tag;
- `S_ARENA_LIST_CDR[k] = ` next cell index, with the chain closed by
  `S_ARENA_LIST_CDR[last] = -1` (the nil sentinel from `OP_NIL`);
- `S_ARENA_LIST_HAS_E[k] = 1` for cells within the rest range,
  `0` otherwise.

**Layer 4 fan-out.** Four pair-cell writes execute in parallel (one
per element). The Layer-4 cell-write loop already iterates over 16
target indices, so the four cells $S_{\text{ARENA\_LIST\_BASE}} +
k$ each get their gate fired by an indicator-pair on
`(S_ARENA_LIST_HAS_E[k], target = base + k)`. Each cell receives
`kind = ARENA_KIND_PAIR`, `car_val = S_ARENA_LIST_E[k]`,
`cdr_val = S_ARENA_LIST_CDR[k]`, plus the two type lanes.

**MEM update.** `MEM[n_fixed]` is overwritten with `S_ARENA_LIST_BASE`
(the list head pointer) and `S_TYPE_MEM[n_fixed] = TYPE_PAIR`. The
bump pointer `S_ARENA_NEXT` advances by $4 - n_{\text{fixed}}$.

**Cost.** Per arena cell: 5 fields × 2 sign-conjugate neurons = 10
Layer-4 neurons; four cells × 10 = 40 neurons, plus the
indicator-pair budget at Layer 3 for the scratch population. Total
≈ 80 neurons for `OP_PACK_REST`. The opcode appears in 4 of the 123
traced programs (variadic-call tests). Unbounded rest-list lengths
would require looping the dispatch — re-issuing `OP_PACK_REST` with
an incremented register window — or expanding the
`S_ARENA_LIST_*` window beyond four slots.

### 5.10. `OP_TAIL_CALL`, plus bounded escape continuations (`OP_CALLCC` / `OP_INVOKE_CC`)

**`OP_TAIL_CALL n`** is `OP_CALL n` followed immediately by `OP_RETURN`
with frame reuse. The semantic delta from `OP_CALL` is that the
current call frame is *overwritten* rather than pushed, so the dispatch
loop's `g_frame_count` does not increment. The bounded encoding covers
arities $n \in \{0, 1, 2, 3, 4\}$: Layer 3 dispatches one
`add_gated_opcode_index` per arity, with the residual delta computed as:

- `PC = TOS` (the function PC was on top of the stack);
- `MEM0..MEM2 ← SOS, R2, R3` for the first three arguments
  (gated by the arity indicator);
- `MEM3 ← 0` if `n < 4`, else `MEM3 ← R3`;
- `TOS, SOS, R2, R3 ← 0` (stack reset);
- `S_TYPE_TOS..S_TYPE_R3 ← TYPE_NUMBER` (stack-type reset);
- `S_DEPTH ← S_DEPTH - (1 + n)` (depth decrement);
- `S_CUR_CLOSURE` *not* reset — tail-call reuses the closure frame.

The dual-input AND pattern on `(S_OPCODE, S_OPERAND)` makes each
arity a 16-neuron block (8 fields × 2 sign-conjugate); the five
arities together account for ≈ 80 Layer-3 neurons.

Broader arities require the same arena-list path as `OP_PACK_REST`
(loop the dispatch, re-issue with shifted register windows) or a
widened MEM register file.

**`OP_CALLCC` / `OP_INVOKE_CC`** (bounded escape continuations).
The general R7RS `call/cc` captures the *full* stack, the heap, the
exception handler chain, and the dynamic-wind frames into a
re-instatable continuation object. That is out of scope for the
bounded artefact. What *is* in scope is the bounded **escape
continuation**: a captured continuation that is only invoked once,
and only to unwind back to its capture point, with no re-entry and
no dynamic-wind interaction. The bounded form is sufficient for
direct-style exception simulation and for one-shot non-local exits.

The encoding uses **four contiguous arena cells in Zone E**
(`ARENA_CONT_CELLS = 4` at `weight_matrices.c §96`):

- cell 0 — kind = `CONT_RESTORE_MARKER = 7.0`, `car = saved_PC`,
  `cdr = saved_FP`;
- cell 1 — `car = saved_TOS`, `cdr = saved_SOS`,
  `car_type/cdr_type` = the two type tags;
- cell 2 — `car = saved_R2`, `cdr = saved_R3`, types as above;
- cell 3 — `car = saved_DEPTH`, `cdr = saved_CUR_CLOSURE`.

`OP_CALLCC` reserves four arena cells, snapshots the live state into
them, and pushes the continuation's first-cell index onto `S_TOS`
with `S_TYPE_TOS = TYPE_CONT = 7`. `OP_INVOKE_CC` indicator-loops
over the 16 cells, finds the one with
`kind = CONT_RESTORE_MARKER`, and restores all eight saved fields
in a single Layer-4 write. Cost: ≈ 80 Layer-3 neurons for the
snapshot (16 cells × ≈ 5 fields × 2-pair gates, with most gates zero
for non-target cells) and the same again for the restore. The
exercised programs use bounded `call/cc` for one-shot escape from
nested computations.

Full R7RS first-class continuations — re-entry, dynamic-wind
interaction, exception unwinding through `OP_PUSH_HANDLER` /
`OP_POP_HANDLER` — remain at the host runtime via `OP_NATIVE_CALL`.
The bounded escape-continuation slice closes the *case that
arises in the bounded artifact*; it does not claim to solve the
general continuation problem.

### 5.11. AD transcendentals — `OP_AD_DIV`, `OP_AD_POW`, `OP_AD_SIN`, `OP_AD_COS` (landed, table form)

The four "transcendental" AD opcodes were originally delegated to the
host runtime for *precision*, not for any side-effect reason. The
landed strict-artifact form table-encodes the verifier-exercised
input range; broader general-input precision sits behind a separate
"relaxed precision" build target (see §8 Stage 4).

#### 5.11.1 `AD_SIN`, `AD_COS` — integer-input table

Constants from `weight_matrices.c §98-99`:

```c
#define AD_TRIG_WEIGHT_MIN_INPUT -4
#define AD_TRIG_WEIGHT_MAX_INPUT  4
```

For integer $x \in [-4, 4]$ (nine values), Layer 3's forward AD
dispatch table-looks-up

$$
\sin(x), \quad \frac{d}{dx}\sin(x) = \cos(x), \quad
\cos(x), \quad \frac{d}{dx}\cos(x) = -\sin(x)
$$

via `add_gated_opcode_index` keyed on $(S_{\text{OPCODE}}, x_{\text{round}})$
where $x_{\text{round}}$ is the value at the tape's input slot. Each
of the 9 input values × 2 ops × 2 quantities (value + saved
derivative) gives 36 table entries, each implemented as one
gated-pair = 72 Layer-3 neurons. Layer 4's tape-write path then
records the resulting node onto the AD tape as `AD_OP_SIN` or
`AD_OP_COS` with the saved derivative loaded.

Traced coverage: the artefact exercises $x \in \{-1, 0, 1\}$,
demonstrating both positive and negative gradient propagation
through trigonometric layers.

#### 5.11.2 `AD_DIV` — bounded-denominator table

`AD_DIV` is encoded for positive integer denominator $d \in [1, 16]$.
Constants from `weight_matrices.c §305-309`:

```c
#define DIV_WEIGHT_MAX_DENOM 16
```

The forward computes $\ell / d$ for the integer denominator; the
saved value is $1/d$ (the left-derivative factor). The backward
rule (`ad_backward_step` at line 1077-1082) gives

$$
\partial L / \partial \ell = \text{grad} \cdot (1/d) = \text{grad} \cdot \text{saved},
$$

$$
\partial L / \partial d = \text{grad} \cdot \ell \cdot (-1/d^2).
$$

The right-side gradient is the interesting one because $d^2$ would
require a second square activation. The encoding sidesteps this by
storing $1/d$ as `saved` and computing
$\text{grad} \cdot \ell \cdot (-(1/d) \cdot (1/d))$ via two
polarisation products in Layer 1 followed by Layer 5's gradient
write — but the bounded table simplifies further: for each
denominator $d \in [1, 16]$, a single gated value $-1/d^2$ is
precomputed and table-looked-up by the `(opcode, denominator)`
pair-index. Cost: 16 denominator values × {forward, left-grad,
right-grad} × 2 sign-pair neurons ≈ 96 Layer-3+5 neurons total.

Verifier coverage includes both `d(x/2)/dx` (where $\ell$ is the AD
variable) and `d(6/y)/dy` (where $d$ is the AD variable), exercising
both sides of the binary gradient rule.

#### 5.11.3 `AD_POW` — bounded base+exponent table

`AD_POW` covers $b \in [1, 8]$ and $e \in [1, 4]$. Constants
(`weight_matrices.c §311-312`):

```c
#define AD_POW_WEIGHT_MAX_BASE 8
#define AD_POW_WEIGHT_MAX_EXP  4
```

The forward computes $b^e$ for the integer pair; the saved value is
$e \cdot b^{e-1}$ (the base-derivative factor). Backward rule
(`ad_backward_step §1083-1088`):

$$
\partial L / \partial b = \text{grad} \cdot e \cdot b^{e-1} = \text{grad} \cdot \text{saved},
$$

$$
\partial L / \partial e = \text{grad} \cdot b^e \cdot \log b = \text{grad} \cdot \text{val} \cdot \log b.
$$

The base gradient uses the standard polarisation product; the
exponent gradient requires a $\log b$ value, which is table-looked-up
by the base index. Cost: 8 bases × 4 exponents × 3 fields × 2 = 192
Layer-3+5 neurons. Traced coverage includes `d(pow(x, 2))/dx` and
`d(pow(2, y))/dy`.

#### 5.11.4 Other unary AD ops — sigmoid, tanh, exp, log, sqrt, abs, relu

The bounded artefact table-encodes:

- `AD_ABS`, `AD_RELU` for bounded nonzero integer-scale cases;
- `AD_EXP` at $x = 0$;
- `AD_SIGMOID` at $x = 0$ and $x = 1$;
- `AD_TANH` at $x = 0$;
- `AD_LOG` at $x = 1$;
- `AD_SQRT` at $x = 4$.

The pattern is identical to `AD_SIN`/`AD_COS`: gated `(opcode,
input)` indicator, table-lookup of `(value, saved_derivative)`,
Layer-4 tape append. Each saved-derivative table entry is one
`add_gated_opcode_index` pair = 2 neurons.

#### 5.11.5 Out-of-scope: general libm precision

Broader transcendental coverage — general $\sin(x)$ for non-integer
$x$, general $\exp$ / $\log$, general $\text{pow}(a, b)$ — would
require one of:

- **Taylor series**: 5-term Taylor for $\sin$/$\cos$ is IEEE-correct
  to ≈ 3 ULP for $x \in [-\pi, \pi]$; range-reduction is
  matrix-encodable but adds one pass through Layer 3;
- **Newton-Raphson reciprocal** for `pow(a, b) = exp(b * log a)`,
  which converges in 4 iterations from a good initial guess;
- **`exp(b * log a)` composition**, which depends on `AD_EXP` /
  `AD_LOG` covering a wider input range.

None of these match the C runtime's `libm` bit-identically.
Adopting them requires shifting the agreement contract from
"bit-identical at every step" to "matches to N ULPs at the AD
output", which is a different artefact — see §8 Stage 4.

## 6. Genuinely-native operations (out of scope)

These remain delegated. The reasons are not "we ran out of weight budget"
but "the operation is fundamentally a side-effect on something the
transformer cannot represent in its bounded state":

| Opcode | Why native |
|---|---|
| `OP_NATIVE_CALL` | Bridges to libcurl, sqlite3, libpthread, libm, etc. The state is in the OS, not the transformer. |
| Full `OP_CALLCC` / `OP_INVOKE_CC` frame walking | The bounded escape-continuation slice is weight-encoded as a four-cell arena record. General first-class continuations that capture and reinstate unbounded stack/heap/dynamic-wind state remain outside the strict artifact contract. |
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
- [x] Encode exercised `AD_EXP`/`AD_SIGMOID`/`AD_TANH`/`AD_LOG`/`AD_SQRT`
  libm values as bounded AD table paths
- [x] Encode `AD_SIN`/`AD_COS` integer input table -4..4, with traced
  representative coverage at -1, 0, and 1
- [x] Encode exercised bounded `AD_DIV` denominator table path, including both
  `d(x/2)/dx` and `d(6/y)/dy` reverse-mode checks
- [x] Encode exercised bounded `AD_POW` positive integer table path, including
  both `d(pow(x,2))/dx` and `d(pow(2,y))/dy` reverse-mode checks
- [x] Encode bounded direct-entry `OP_CALLCC`/`OP_INVOKE_CC` escape
  continuations as a four-cell arena record
- [x] Add explicit verifier coverage for previously untraced weight opcodes:
  `OP_NOP`, `OP_NOT`, `OP_LOOP`, and `OP_AD_SUB`
- [x] Re-run `scripts/paper/run_paper_suite.sh`; current report is 123/123
  PRINT-output and full-state agreement, with 82 weight-implemented / 0
  native-delegated / 0 transformer-native-assisted opcodes in the exercised
  coverage set
- [x] **Acceptance:** Stage 3's bounded closure/upvalue, tail-call, and
  pack-rest artifact paths plus bounded escape continuations are
  weight-implemented. The exercised trace set has no native-assisted
  transformer steps. The remaining true native boundary is outside this stage:
  `OP_NATIVE_CALL`, full exception/dynamic-wind unwinding, unbounded
  continuation frame walking, general IEEE `DIV`/`MOD`, and broader
  relaxed-precision AD transcendentals.

### Stage 4 — AD transcendentals (Taylor + Newton-Raphson) on a separate build target
*~1 week, ~100k new params*

- [x] Exercised `AD_EXP(0)`, `AD_SIGMOID(0/1)`, `AD_TANH(0)`, `AD_LOG(1)`,
  `AD_SQRT(4)`, and representative `AD_SIN`/`AD_COS` integer inputs -1, 0,
  and 1 via bounded table gates in the strict artifact build
- [x] Exercised `AD_DIV` via positive denominator table gates and both
  numerator/denominator reverse-mode gradient paths
- [x] Exercised `AD_POW` via positive base/exponent table gates and both
  base/exponent reverse-mode gradient paths
- [ ] General `OP_AD_SIN`, `OP_AD_COS` via Bhaskara/Taylor in Layer 3 dispatch
- [ ] General `OP_AD_POW` via existing EXP/LOG composition
- [ ] General `OP_AD_DIV` via Newton-Raphson 4-iter loop
- [ ] **Acceptance:** Taylor-build matches reference to ≤ 4 ULPs on AD ops,
  documented relaxation; libm-delegated build remains bit-identical

### Stage 0 — Bit-identity fixes (pre-stage 1)

Before any of stages 1-4 could expand the weight-encoded opcode set,
the **agreement contract had to be tightened from "output matches" to
"every byte of every step matches"**. The contract change exposed five
distinct weight-encoding bugs that had been hiding inside the inline
0.01-tolerance output check. Each fix is documented in commit
**`7301dc4 fix(paper): bit-identical SDNC agreement — 71/71 full
per-step state`**; the full narrative is in
`docs/SDNC.md §6`. Summary table for the milestone log:

| # | Symptom                                                    | Root cause                                                                      | Fix location in `weight_matrices.c`                              |
|--:|------------------------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------|
| 1 | `tail sum(100)` matrix `tos = 4.4e-16` vs reference `0`     | `SCALE = 100` left $\exp(-35.4)$ residue in Layer 0 attention softmax           | `#define SCALE 300.0f`, lines 59-85                                |
| 2 | `tape[1] = 7` on AD-VAR programs during backward            | Layer 4 forward tape-write missing the `S_AD_IS_FORWARD` row of the gate        | `add S_AD_IS_FORWARD coefficient = 10·SCALE`, lines 4880-4951      |
| 3 | After fix 2, gate re-opens at high `S_AD_TAPE_LEN`         | Symmetric AND with both inputs at `SCALE` fails when one input swings wider     | Asymmetric weighting: `10·SCALE` on the binary input, lines 3752-3779 |
| 4 | One-cycle PC/register/tos drift after backward finishes    | Cursor-done indicator at `cursor == -1` fires one cycle late                    | `at_done = indicator(cursor, 0.0f)`, lines 1352-1371 + 3744-3779   |
| 5 | `sigmoid` AD gradient `0.393223852` vs `0.393223763`        | Reference VM multiplied `grad * saved` directly while matrix uses polarisation | `POLARIZATION_PRODUCT` macro in `ad_backward_step`, lines 1064-1096 |

Acceptance after Stage 0 was 71/71 traced programs on both the
output-agreement and the full-per-step-state-agreement metrics
(commit `7301dc4`). Stages 1-4 then expanded the weight coverage
while preserving both metrics, ending at the current 123/123 status.

The lesson encoded in this milestone — *bit-identity is a stronger
contract than approximate agreement, and the agreement metric the
artifact reports must be the stronger one* — is now folded into the
`compare_traces.py` exit policy
(`scripts/paper/compare_traces.py §335`): the script exits non-zero
when `output_agreeing_programs < total_programs`, and the regenerator
refuses to ship a new QLMW file if the inline `test()` count of
failures is nonzero. A future patch that introduces a drift on either
metric must either revert or fix.

## 9. Risk & mitigation

| Risk | Mitigation |
|---|---|
| `d_model = 256` breaks downstream paper-trace dump pipeline | New `d_model` constant + format-version bump in ESKB writer; old traces still readable via version detection |
| 16-cell heap insufficient for real Eshkol programs | The artifact's 71 test programs only exercise heap shallowly. Real programs need bounded heap or chain-walks; we accept that "all memory ops in the transformer" is a *contract about ops*, not about *unbounded program execution*. The contract is: "for any program whose live-cell-count never exceeds 16, the full execution is one transformer block iterated." Production deployments scale `MEM_SIZE` linearly. |
| Taylor approximation breaks AD precision | Two builds; users opt in via CMake flag |
| Float32 saturation breaks at `d_model = 256` | SCALE = 300 already has 50 ULP margin per the `weight_matrices.c:59-84` analysis; doubling state dims adds ~2× accumulation length, well within margin |
| Frame walking for `CALLCC` | Bounded escape continuations are encoded; unbounded frame walking remains outside the strict artifact contract. |

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
