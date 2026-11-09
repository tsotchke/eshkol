---
kind: explanation
status: current
owner-area: numeric
since: v1.4.0
sources:
  - lib/core/arena_memory.h
  - lib/core/runtime_exact_tensor.cpp
  - lib/core/runtime_vector_mutation.cpp
  - lib/core/runtime_tensor_alloc.cpp
  - lib/core/runtime_tensor_dtype.cpp
  - lib/core/runtime_tensor_math.cpp
  - lib/core/runtime_list_helpers.cpp
  - lib/backend/codegen_context.cpp
  - lib/backend/tensor_codegen.cpp
  - lib/backend/tensor_linalg_codegen.cpp
  - lib/backend/tensor_creation_codegen.cpp
  - lib/backend/vm_native.c
  - lib/math.esk
  - lib/core/exact_linalg.esk
  - tests/stdlib/exact_tensor_test.esk
---
# ADR-0055: The exact tensor element type

**Status:** Proposed
**Scope:** The tensor descriptor's `dtype` field, the tensor slot store, the
tensor operand checks, the tagged tensor kernels, the native linear-algebra
operators (`tensor-det`, `tensor-inverse`, `tensor-solve`, `linear-solve`),
`tensor-cast` / `make-tensor :dtype`, `lib/math.esk` (`det`, `inv`, `solve`)
and `core.exact_linalg`.

## Context

The numeric tower is exact at the scalar level: fixnums, bignums and
bignum-capable rationals, with R7RS contagion. Tensor storage was not: every
numeric tensor held f64 bit patterns, `(tensor 1/3)` stored the nearest
double, and `lib/core/linear_solve.cpp`, the BLAS entry points and the LU
kernels behind `tensor-det` / `tensor-inverse` / `tensor-solve` read doubles
only. `core.exact_linalg` (v1.3.5) worked around this on Scheme vectors of row
vectors, and `lib/math.esk`'s `det` / `inv` seeded `1.0` and `0.0`, so an exact
matrix came back inexact. An exact system over tensor storage had to be
written out by hand on Scheme vectors (docs/design/NAVIER_STOKES_RESIDUAL_MECHANIZATION.md,
steps 7, 22, 24, 25, 34, 59 and 68).

The runtime already had a second slot layout. A jet tensor (dtype `DUAL`,
ADR-0020 amendment 2) and a boxed carrier (dtype `BOXED`, ADR-0020) store
16-byte tagged values in `elements`; region evacuation, display, `equal?`,
list conversion and the vector API already handle that layout through
`eshkol_tensor_dtype_is_tagged`. The jet kernels for elementwise arithmetic,
matmul and sum already apply the language's own scalar operators slot by slot
(`eshkol_taylor_binary_tagged`), and on two exact numbers that operator is the
exact ring.

## Decision

1. **One new dtype code, `ESHKOL_TENSOR_DTYPE_EXACT = 66`.** Its `elements` are
   16-byte tagged values, each an exact real of the scalar tower. The
   descriptor layout (40 bytes, five fields) does not change; the new code is
   an additional value of field 4 and joins `eshkol_tensor_dtype_is_tagged`.
   An exact tensor is a numeric tensor: `tensor?` answers `#t` and
   `(tensor-dtype t)` answers `exact`. `eshkol_tensor_dtype_is_tagged_numeric`
   (DUAL or EXACT) names the tensors that take the generic-scalar kernels; its
   IR spelling is `CodegenContext::emitDtypeIsTaggedNumeric`, and
   `emitDtypeHasTaggedSlots` replaces the per-site `BOXED || DUAL` tests so
   every slot-width decision agrees with the runtime.

2. **Construction is explicit.** `(tensor-cast X 'exact)` takes a tensor, a
   list, a vector or a rectangular nest (the shape rules of `(tensor X)`):
   exact elements are kept whole and a finite double becomes the exact rational
   it denotes, as `exact` gives it; an infinity or NaN is refused.
   `(make-tensor shape fill :dtype 'exact)` stores an exact fill whole and
   refuses an inexact one. `(tensor ...)` and numeric `#(...)` literals keep
   building f64 tensors, so no existing program changes representation.
   `(tensor-cast t 'f64)` is the explicit conversion out.

3. **The slot store keeps the type honest.** `tensor_slot_accepts` admits only
   exact reals into an exact tensor; `tensor-set!` of an inexact number is
   refused, never rounded into the slot and never allowed to make the
   declared exactness false. The vector API may relabel an exact tensor as a
   boxed carrier for a non-numeric value (its slots are already tagged), as it
   does for an f64 tensor.

4. **Kernels with an exact rule, and a refusing default.** The operand checks
   take an explicit policy (`eshkol_tensor_operand_policy_checked` /
   `eshkol_tensor_destination_policy_checked`, accept mask DUAL | EXACT), and
   `TensorCodegen::unpackTensorOperandChecked` derives the mask from two lists:
   the dual carriers (unchanged) and the exact carriers: `tensor-get`,
   `tensor-length`, `tensor-shape`, `reshape`, `tensor-sum`, `matmul`,
   `transpose`, `tensor-set!` and `tensor-add` / `-sub` / `-mul` / `-div`
   (with the generic `+ - * /` on tensors, which lower to them). Element
   access (`tensor-ref`, `vector-ref`, `vref`, `tensor-get`) reads the tagged
   slot. Every other kernel, including BLAS-backed ones, the f64 LU / QR /
   Cholesky / SVD family and the transcendental maps, refuses an exact tensor
   by name with a message naming `(tensor-cast t 'f64)`.

5. **Linear algebra has an exact path.** `tensor-det`, `tensor-inverse`,
   `tensor-solve` and `linear-solve` test their operands at run time
   (`eshkol_linalg_operands_exact`). When an operand is an exact tensor,
   `eshkol_exact_linalg_tagged` answers: the determinant by Bareiss
   fraction-free elimination; solve and inverse by fraction-free elimination
   of `[A | B]` and exact back substitution. A singular matrix is decided from
   an exact zero pivot column, never from a tolerance. The f64 kernels
   (`linear_solve.cpp`, the LU routines) are never handed an exact tensor.

6. **Contagion decides representation from operand dtypes.** In every tagged
   kernel (`finish_tagged_result`, `eshkol_exact_linalg_tagged`): two exact
   operands give an exact result; a jet operand gives a jet tensor; an exact
   operand meeting an inexact numeric tensor gives an inexact tensor of that
   operand's dtype, computed on the projections (for solve, by the f64 LU
   routine). A list or vector operand converts as `(tensor X)` converts it
   (f64), so passing exact data to an exact kernel means passing an exact
   tensor.

7. **`lib/math.esk` follows contagion.** `det`, `inv` and `solve` seed exact
   constants (`1`, `0`, the exact identity); a pivot is zero exactly when it is
   exact and by `epsilon` only when inexact. An exact input gives an exact
   answer, an inexact input the same doubles as before.

8. **`core.exact_linalg` accepts exact tensors.** Every public function reads
   a rank-2 (or rank-1 right-hand-side) tensor entry by entry with
   `tensor-ref`; a tensor operand gets a tensor answer, exact when every entry
   is exact. The native kernels and this module are independent
   implementations and are cross-checked against each other in
   `tests/stdlib/exact_tensor_test.esk`.

9. **The bytecode VM refuses the dtype explicitly.** VM tensors are
   double-backed. `tensor-cast` and `make-tensor` with `'exact` raise a
   catchable error naming the native engines; VM exact tensors are planned for
   v1.4.1.

## Consequences

- The object ABI gains one dtype value; the descriptor and the object header
  are unchanged, so every existing GEP and every existing tensor keeps its
  meaning. Code that decides slot width must ask the shared predicate, not
  compare against `DUAL` and `BOXED` by hand.
- An exact system is written once over tensor storage and cross-checked
  against `core.exact_linalg`; the NS mechanization steps that solved exact
  systems on Scheme vectors can use exact tensors directly.
- The tagged matmul kernel gained PEP 465 promotion of rank-1 operands, which
  jet tensors share.
- Exact kernels are O(n^3) ring operations on heap rationals; they are for
  certification-sized systems, not for bulk numerics.
- Planned for v1.4.1: exact tensors on the bytecode VM, and exact rules for
  further kernels (`tensor-apply`, reductions along an axis, `einsum`).
