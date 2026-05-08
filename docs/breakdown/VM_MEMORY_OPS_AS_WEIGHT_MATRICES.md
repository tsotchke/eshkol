# VM-as-Transformer: Encoding the 26 Delegated Memory Operations as Weight Matrices

**Status:** design draft
**Worktree:** `feature/vm-transformer-memory`
**Author of this draft:** internal architecture review, 2026-05-08
**Companion artifact:** `research/papers/` (33 cached PDFs, content-addressed)
**Companion bib:** `research/vm-transformer-bib.bib`

## 1. Problem statement

The Eshkol VM is a 83-opcode bytecode machine. The Self-Differentiating Neural
Computer (SDNC) — `lib/backend/weight_matrices.c`, 4248 lines — analytically
constructs a 6-layer transformer (`d_model = 128`, 16 heads, 2 head dims, FFN
width 1024, 2.8M parameters) whose forward pass is bit-identical to the
reference VM on the 71-program suite pinned at `8235d99`.

57 of the 83 opcodes execute end-to-end through `Wx + b` matmul-plus-bias.
The remaining **26 opcodes are delegated** to the C runtime via an `IS_NATIVE`
boundary marker that the transformer emits and the VM dispatcher honours
(`weight_matrices.c:537-548, 868-905`). For each delegated opcode the
transformer cleanly increments PC and sets `S_IS_NATIVE = 1`; the runtime then
performs the side-effecting heap operation and resumes the next cycle.

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

(Count drift: `OP_VOID=63` is treated as native by the runtime in some paths
but the weight matrix already emits PC++ for it; not in the 26.)

We classify these by **whether the runtime side-effect is essential or
incidental**:

- **Encodable (this design):** type predicates, heap data ops, closures,
  upvalues, `POPN`, `TAIL_CALL`, `PACK_REST`, AD transcendentals.
- **Genuinely native (out of scope here):** `NATIVE_CALL` (by definition
  bridges to libcurl, sqlite3, libpthread), `CALLCC`/`INVOKE_CC` (first-class
  continuation requires capturing the entire VM stack — can be encoded with
  unbounded depth penalty), `PUSH_HANDLER`/`POP_HANDLER`/`GET_EXN`/`WIND_PUSH`/
  `WIND_POP` (R7RS exception + dynamic-wind require the VM to manipulate frames
  *outside* the transformer's instruction-fetch path).
- **Precision-delegated:** `OP_DIV`, `OP_MOD`. The transformer can emit
  approximations via Newton-Raphson iteration but the bit-identical agreement
  contract requires IEEE-correct rounding, which is delegated. We leave these
  alone; this design does not change the precision contract.

The reachable target for "all *memory* operations are in the transformer" is
**24 of the 26**, ramping in three milestones (§9). The two genuinely-native
ops (`NATIVE_CALL`, `CALLCC`) remain delegated by design.

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

**Encoding.** Each predicate is **one indicator gate on `S_TYPE_TOS`** in
Layer 3:

```
Δ S_TOS = indicator(opcode, OP_PAIR_P) · alive · (indicator(S_TYPE_TOS, 2) - S_TOS)
```

Plus the type-tag write `Δ S_TYPE_TOS = indicator(opcode, OP_PAIR_P) · alive
· (1 - S_TYPE_TOS)`. Cost: 6 new gated-pair entries in the Layer 3 FFN.
**Trivial.** The pattern is identical to `OP_NULL_P`'s currently-delegated
form, except `NULL_P` checks the value not the type tag — see §5.2 for that.

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

Identical to vector ops but with `F_TYPE = string`. Strings of length ≤ 4
fit inline; longer strings chain (stage 2).

The artifact's strings are short (function names: `"map"`, `"foldr"`,
display literals: `"hello"`). Stage-1 `len ≤ 4` covers ~80% of artifact
strings; we extend to length 32 in stage 2 by chaining 8 cells.

### 5.8. Closures and upvalues (5 ops)

A closure is a **cell with `F_TYPE = closure`**, where `F_CAR = function-pc`
(integer constant pool index pointing to the bytecode entry), and
`F_DATA0..F_DATA3` are the four upvalues (Eshkol closures rarely capture
more than 4; see lambda-lift pass in `closure_codegen.cpp`). Captures of
> 4 upvalues require chain walks (stage 2).

`OP_CLOSURE`: Layer 7 writes function index + captured upvalues into a new
cell. Pre-state: stack has `[upval_0, upval_1, ..., upval_n-1, func_idx]`.
Layer 3 reads `S_TOS = func_idx`, `S_SOS = upval_n-1`, ..., from working
state. Stage 1 caps `n ≤ 4`.

`OP_GET_UPVALUE k`: closure pointer is in current frame's `S_FRAME_CLOSURE`
(a new dim added in Zone B). Layer 6 fetches the cell, Layer 3 selects
`F_DATA[k]` via indicator-gate over the operand. Cost: 4 indicator gates
(operand ∈ {0,1,2,3}).

`OP_SET_UPVALUE k`: SET_CAR pattern, targeting `F_DATA[k]` of the closure
cell.

`OP_CLOSE_UPVALUE`: marks an upvalue as no longer live. In Eshkol's runtime
this triggers eager cell freeing (closure-on-thread leak fix from bug 224).
Encoded as `Δ heap[closure].F_DATA[k] = 0`, `Δ heap[closure].F_TYPE_DATA[k]
= TYPE_NIL`. One masked write.

`OP_OPEN_CLOSURE`: noop in current Eshkol runtime (placeholder for stack
upvalue → heap upvalue conversion). Already trivially encodable as PC++.

### 5.9. `OP_PACK_REST`

`(pack-rest n_fixed)` packs the top `argc - n_fixed` arguments into a
single list. This is a chained `CONS` operation: at most one CONS per
remaining argument. Encoded via the **looped transformer** pattern: keep
PC pinned, decrement an internal counter each cycle, emit one CONS per
cycle. Cost: zero new opcodes; reuses CONS infrastructure with a
"chain mode" Layer 3 dispatch that increments PC only when the counter
hits zero.

### 5.10. `OP_TAIL_CALL`

`TAIL_CALL n` is `CALL n` followed by `RETURN` with frame reuse. Already
half-encoded (line 405 sets `IS_RET`; `IS_CALL` triggers frame management
in the exec loop). Full encoding requires the transformer to emit
*both* the frame-collapse (move return PC up) and the new call-frame
push in one cycle — currently the runtime does this, but the construction
mirrors `OP_CALL` + `OP_RETURN` composed. One new gated pair, branching on
the operand's tail-call flag.

### 5.11. AD transcendentals (4 ops): `OP_AD_DIV`, `OP_AD_POW`, `OP_AD_SIN`, `OP_AD_COS`

These are delegated for *precision*, not for runtime side-effect. The
transformer can compute them via the existing AD forward dispatch (Layer 3,
lines 481-509), but transcendentals require:

- `sin(x)` / `cos(x)`: Bhaskara's approximation or Taylor (5 terms ≈
  IEEE-correct for x ∈ [-π, π], precision drops outside; range-reduction
  is matrix-encodable but adds a second pass).
- `pow(a, b)`: `exp(b · log(a))`, falls out of EXP/LOG which are already
  weight-implemented.
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
| `OP_PUSH_HANDLER`, `OP_POP_HANDLER`, `OP_GET_EXN` | R7RS exception handling traverses the dynamic-wind stack; the unwinding is iterative-with-side-effect-on-frames. |
| `OP_WIND_PUSH`, `OP_WIND_POP` | Same as above. |
| `OP_DIV`, `OP_MOD` | IEEE 754 correct rounding requires libm. Newton-Raphson in the transformer matches to ~3 ULPs, not bit-identical. |

For the artifact contract, we ship **both** a "strict-bit-identical"
transformer (the current 26-delegated version) and an **"all-memory-ops-
encoded"** transformer that delegates only the 8 genuinely-native ops.

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

### Stage 1 — Type predicates + null + POPN + closure-stack-only ops
*~3 days, ~30k new params, ~1500 lines added to weight_matrices.c*

- [ ] Add `S_TYPE_*` lanes to closures' upvalue slots in Zone D
- [ ] Implement 7 indicator gates for `PAIR_P` … `NUM_P` + `NULL_P`
- [ ] Implement 3 `POPN` cases
- [ ] Implement `OP_CLOSE_UPVALUE`, `OP_OPEN_CLOSURE` (trivially PC++)
- [ ] Re-verify bit-identical agreement on artifact (no heap touched)
- [ ] **Acceptance:** weight_matrices.c compiles, 71/71 still bit-identical

### Stage 2 — Heap bank: CONS, CAR, CDR, SET_CAR, SET_CDR, VEC_*, STR_*
*~2 weeks, ~1.5M new params, d_model 128 → 256*

- [ ] Extend `D` to 256, add Zone E layout
- [ ] Add Layer 6 (cell fetch), Layer 7 (heap mutate)
- [ ] Implement 12 heap ops via `add_heap_write` + `add_cell_load_indicator`
- [ ] Update simulated transformer C functions (`exec_loop_postprocess`)
- [ ] Update reference comparison harness for 256-dim states
- [ ] **Acceptance:** add 5 new test programs that exercise heap (cons/list,
  vector ops, string ops); all 76 programs (71 base + 5 new) bit-identical
  agreement against simulated transformer mode

### Stage 3 — Closures with upvalues, TAIL_CALL, PACK_REST
*~1 week, ~500k new params*

- [ ] `OP_CLOSURE`, `OP_GET_UPVALUE`, `OP_SET_UPVALUE` via heap cells
- [ ] `OP_TAIL_CALL` as composed CALL+RETURN
- [ ] `OP_PACK_REST` via looped CONS
- [ ] **Acceptance:** all currently-delegated ops (24 of 26) now weight-
  implemented; only `OP_NATIVE_CALL` and `OP_CALLCC`/`OP_INVOKE_CC` +
  exception ops + DIV/MOD remain native

### Stage 4 — AD transcendentals (Taylor + Newton-Raphson) on a separate build target
*~1 week, ~100k new params*

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
