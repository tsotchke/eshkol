---
kind: explanation
status: current
owner-area: runtime
since: v1.3.5
sources:
  - lib/core/runtime_vector_mutation.cpp
  - tests/vm_parity/corpus/85_container_slot_store.esk
  - inc/eshkol/backend/codegen_context.h
  - .icc/ledger/entries/SW-179.yaml
---
# ADR-0020: One store boundary for every container slot

**Status:** Accepted
**Ledger:** SW-179
**Scope:** Native LLVM code generation (JIT, AOT and the REPL JIT), the native
runtime, and the bytecode VM.

## Context

Eshkol exposes two sequence representations through the R7RS vector API.

| | Scheme vector | Tensor |
|---|---|---|
| Built by | `vector`, `make-vector`, `list->vector`, `vector-map`, `vector-append`, a quoted `'#(...)`, a `#(...)` literal with a non-numeric, rational or bignum element | `tensor`, `make-tensor`, and a numeric `#(...)` literal (the language specification, section 3.2, makes that the tensor-literal syntax) |
| Heap subtype | `HEAP_SUBTYPE_VECTOR` | `HEAP_SUBTYPE_TENSOR` |
| Slot | an inline 16-byte tagged value | 8 bytes in a separate element buffer, interpreted through the tensor's dtype |

Both answer `vector?`, and every `vector-*` procedure accepts both. The read
side has always dispatched on the heap subtype. The write side did not have a
single rule, and each mutator had grown its own copy of the store:

- `vector-set!` dispatched on the subtype but stored `unpackDouble(value)` —
  the payload bits of whatever value arrived. An exact integer, a string
  pointer or a boolean was written as if it were a double. `vector-ref` then
  read a small integer payload back as a differentiation-node pointer.
- `vector-fill!` had no tensor path; it read the tensor header as a length.
- `tensor-set!` converted numbers correctly but stored `0.0` for a value that
  is not a number, and accepted a Scheme vector operand by updating a coerced
  copy that was then discarded.
- `vector-copy!` lived in the runtime and validated before writing, but its
  numeric conversion did not know exact rationals or bignums.
- The VM's tensor writers stored `0.0` for a value that is not a number.

`docs/reference/tensors/creation.md` already states the rule — one conversion,
"applied uniformly at every chokepoint that builds or mutates a tensor from a
Scheme value". The implementation had several chokepoints and no shared
facility, so the rule held only where someone had remembered it.

ICC discovery for this decision: `icc adr list` shows no prior ADR covering slot
representation (ADR-0012 governs the object *header* layout, not slot
contents); `icc trace-callers` on `CollectionCodegen::vectorSet` reaches it only
through `codegenCallTask`, and `icc trace-callees` shows it stored through
`TaggedValueCodegen::unpackDouble` with no conversion step;
`icc duplicate-implementations` over `lib/backend` and `lib/core` reports no
function-level duplicates of the store, because every copy was emitted inline
inside a larger codegen function — which is exactly why the copies diverged
unnoticed.

## Decision

**Invariant.** A value stored into a container slot is a value of the slot's
declared representation. A store never reinterprets payload bits.

- A **Scheme vector slot** declares "any tagged value". The value is stored
  unchanged, behind the region write barrier.
- A **tensor slot** declares "a real number at this tensor's dtype". A real
  number of any exactness — fixnum, flonum, exact rational, bignum — is
  converted exactly as `inexact` converts it and reduced to the tensor's dtype.
  This is the conversion tensor construction already applies to each element.
- A reverse-mode differentiation node keeps the in-tensor carrier encoding the
  differentiation operators already read back, so a store performed inside a
  differentiated function stays on the tape. A dual tensor's slot is a tagged
  value and takes a dual number or a real number.
- Any other value has **no representation** in a tensor slot. The store is
  refused with a catchable error, and the refusal happens before the
  destination is modified — for a range operation, before the first slot is
  written.

**One boundary.** The runtime half is `lib/core/runtime_vector_mutation.cpp`.
`encode_tensor_slot()` is the only code that writes a Scheme value into a tensor
slot, and `tensor_slot_accepts()` the only code that decides whether it may.
They are reached through four entry points that share one status vocabulary
(`eshkol_slot_store_status_t`):

| Entry point | Used by |
|---|---|
| `eshkol_sequence_slot_store` | `vector-set!` on any operand not proven to be a Scheme vector |
| `eshkol_tensor_slot_store` | `tensor-set!` |
| `eshkol_sequence_fill` | `vector-fill!` |
| `eshkol_vector_copy_mutating` | `vector-copy!` |

The compiled half is three emitters on `CodegenContext`
(`emitSequenceSlotStore`, `emitTensorSlotStore`, `emitSequenceFill`) plus
`emitSlotStoreStatusCheck`, the single failure block that turns a status into
the diagnostic for that mutator. Compiled code may store inline on exactly two
paths, both of which have already proven the representation: a tagged value
into a `HEAP_SUBTYPE_VECTOR` slot, and a `DOUBLE` into an f64 tensor slot.
Everything else calls the runtime.

The VM half is `vm_tensor_slot_value()` in `lib/backend/vm_core.c`, used by the
VM's `vector-set!` tensor path and both `tensor-set!` forms.

A mutator written in Eshkol (the VM prelude's `vector-copy!`, library code) is
inside the boundary by construction, because it bottoms out in `vector-set!`.

An operand that is written through in place must be a real container.
`tensor-set!` therefore takes its operand in `RequireTensor` mode, the existing
facility for optimizer parameters: a Scheme vector is rejected instead of being
coerced to a temporary that absorbs the write.

## Alternatives considered

**Materialise numeric `#(...)` literals as Scheme vectors.** Rejected. The
language specification defines a numeric `#(...)` literal as the tensor-literal
syntax, `#(#(1 2) #(3 4))` as a rank-2 tensor, and the differentiation and
tensor references use the literal as a point and as an operand throughout.
Changing the literal's representation would remove a documented feature to
repair a store that was simply not converting its operand.

**Promote the container in place when a value of another kind is stored.**
Adopted for the vector API in the amendment below; see it for why the original
objection did not survive contact with the engine difference it left behind.

**Store `0.0` for a value that is not a number** (the previous `tensor-set!`
behaviour). Rejected: it is a silent wrong answer.

## Consequences

- `(vector-set! v 0 99)` on a numeric literal stores 99; the slot reads back as
  the inexact `99.0`, as every element of a tensor does.
- Storing a string, boolean, character, symbol, pair, vector or procedure into a
  tensor-backed vector raises `<mutator>: value has no representation in a
  numeric tensor slot`. R7RS leaves mutation of a literal constant undefined; a
  program that needs a heterogeneous mutable vector builds it with `vector`,
  `make-vector` or `vector-copy` of a quoted literal, all of which are Scheme
  vectors. The bytecode VM materialises `#(...)` as a Scheme vector and so
  accepts such a store; that is the pre-existing literal-representation
  difference recorded in `tests/vm_parity/found/tensor_predicate_on_literal.esk`,
  not a new one, and it is loud on the side that differs.
- A refused `tensor-set!`, a `tensor-set!` on a Scheme vector, and a
  `vector-fill!` on a non-sequence are now errors rather than silent no-ops.
- The hot paths are unchanged in shape: a Scheme vector store is still an
  inline bounds check, barrier and store; an f64 tensor store of a double is
  still an inline store, preceded by one dtype load and compare.
- The tensor *load* side still recognises a differentiation-node pointer by its
  bit range. That decoder belongs to the differentiation carriers work and is
  not changed here; with stores no longer able to write a non-number's payload,
  the only values it can see are the ones it was written for.

## Verification

`tests/core/container_slot_store_test.esk` sweeps every mutator against every
value kind on both representations and asserts exact results, under the
in-process JIT, the cached run path and AOT. `tests/vm_parity/corpus/` carries
the subset on which the VM and the native engines must agree. The tutorial
example that exposed SW-179 runs unmarked in the documentation example gate.

---

## Amendment 1 (v1.3.5): the vector API promotes the carrier

The decision above made a non-numeric store through the vector API a catchable
error. That is honest, but it left the two engines disagreeing about a program
R7RS defines: the bytecode VM materialises `#(10 20 30)` as a heterogeneous
Scheme vector and stores `"x"` happily, while the native engines materialise it
as a tensor and raised. R7RS vectors hold any object, so the error was a
native-only restriction, not a language rule.

The original objection to promotion was that a tensor descriptor cannot become
a Scheme vector at the same address. It cannot — but it does not have to. The
tensor object already supports a *tagged* element buffer: `ESHKOL_TENSOR_DTYPE_DUAL`
stores 16-byte tagged values for the forward-mode Hessian sweep. Promotion
reuses that shape under a new dtype, `ESHKOL_TENSOR_DTYPE_BOXED`:

- `promote_tensor_to_boxed()` in `lib/core/runtime_vector_mutation.cpp`
  allocates a tagged-value buffer, carries every existing element over as the
  number it was, and re-points `elements` and `dtype` **on the descriptor
  itself**, so every alias of the vector sees the promotion.
- It runs only from the vector API — `vector-set!`, `vector-fill!`,
  `vector-copy!` — and only for a value the numeric carrier cannot hold. The
  tensor API (`tensor-set!`) keeps a tensor numeric and still refuses, because
  a tensor is a numeric carrier by definition.
- A promoted carrier is no longer a numeric tensor. `tensor?` answers `#f` for
  it, and every tensor kernel refuses it through the one operand check
  (`eshkol_tensor_operand_checked` and its destination/matrix variants), so no
  kernel can read a tagged slot as an f64.
- Every reader of tensor slots learned the dtype: `vector-ref`, `vector->list`,
  `vector-copy`, `vector-append`, `vector-map`/`vector-for-each` (through one
  shared element loader that replaced two copies), `display`, `equal?`, and
  region evacuation, which walks the tagged slots so a promoted vector cannot
  dangle into a popped region.

**Consequence.** `(define v #(10 20 30)) (vector-set! v 0 "x") (display v)`
prints `#(x 20 30)` on the JIT, on an AOT binary and on the VM.
`tests/vm_parity/corpus/85_container_slot_store.esk` proves the agreement for a
string, a boolean, a pair, a character, a symbol and a nested vector, and for
every reader above.

**Residual difference.** A *constructed* `(tensor 1.0 2.0)` promotes natively
through the vector API but raises on the VM, whose tensors have no promotion
path: the VM would have to re-dispatch every alias through its heap box. The
difference is loud on the side that differs and is checked per engine
(`tests/core/container_slot_store_test.esk`, `tests/vm/tensor_slot_store_test.esk`)
rather than in the parity corpus. Numeric `#(...)` literals — the case the
engines actually disagreed about — agree exactly.
