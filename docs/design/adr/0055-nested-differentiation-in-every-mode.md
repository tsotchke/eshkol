---
kind: explanation
status: current
owner-area: ad
since: v1.4.0
sources:
  - inc/eshkol/eshkol.h
  - lib/core/runtime_taylor.c
  - lib/core/runtime_autodiff.cpp
  - lib/backend/autodiff_codegen.cpp
  - lib/backend/vm_numeric.h
  - lib/backend/vm_dual.c
  - lib/backend/vm_native.c
  - tests/ad/nested_depth_matrix_test.esk
---
# ADR-0055: Nested differentiation in every mode

**Status:** Proposed
**Builds on:** [ADR-0027](0027-recursive-taylor-level-carrier.md) (one recursive
Taylor level carrier)
**Scope:** Native LLVM code generation (JIT and AOT), the bytecode VM, and the
browser VM, which is the same VM compiled to WebAssembly.

## Context

ADR-0027 made every forward pass a *level*: a truncated series in its own
perturbation, whose coefficients are numbers of the enclosing levels. Forward
passes then nest to any depth and order. Three things were still outside that
rule.

1. **Perturbation tags were 16 bits.** A level's epoch was drawn from a
   process-wide counter that wrapped at 65536 and was stored in bits 16..31 of
   `esh_taylor_t::flags`; the nested-pass route packed it into bits 8..23 of
   an `i32`. The level path chooses the *largest* epoch among the operands as
   the innermost level, which is true only while tags are issued in opening
   order. After 65536 passes in one process a level opened after the wrap had
   a smaller tag than the level enclosing it, and a program that ran a nested
   derivative in a loop stopped answering after about 16000 iterations.
2. **A vector-point `gradient` inside a live pass took the reverse tape.** The
   tape stores one raw double per node, so neither an enclosing forward
   perturbation nor a carrier inside the point has anywhere to live on it.
   `jacobian` and the nested `hessian` already switched to forward columns in
   that situation; `gradient` did not, so a forward pass over a forward pass
   over a vector `gradient`, or a third reverse level, lost the enclosing
   dependence.
3. **An inner pass that met an enclosing reverse pass's tape nodes linearised
   against the active seed only.** The forward-over-reverse arm of the
   runtime-closure `gradient` wrote `d(grad_i)/d(seed)` back onto the outer
   tape, with the seed found by pointer identity. A point whose components
   were *computed* from the outer variables (`(g (vector (* 2.0 (vref v 0))))`)
   had no component equal to the seed, so the arm raised; a point mixing the
   seed with values computed from it dropped the computed components'
   contribution.

## Decision

### 1. Tags are 64-bit and never reused

- `esh_taylor_t` holds the tag in its own field, `uint64_t epoch`, in place of
  the three retired companion-lane tags. The struct stays 32 bytes, and
  `exact_c` and `c[]` keep their offsets. `flags` no longer carries a tag;
  `ESH_TAYLOR_MK_FLAGS(coeff)` builds only the coefficient field.
- `eshkol_taylor_next_epoch()` returns `uint64_t` and never wraps in practice
  (2^64 passes). `VmDual::epoch` and `vm_dual_next_taylor_epoch()` are 64-bit
  on the VM.
- The nested-pass route is an `i64`: `0` for an un-nested pass, otherwise
  `(epoch << 1) | inexact`, built and read with `ESH_AD_NEST_ROUTE`,
  `ESH_AD_NEST_ROUTE_EPOCH` and `ESH_AD_NEST_ROUTE_INEXACT`.
  `eshkol_ad_nested_seed` returns it and `eshkol_ad_nested_extract` takes it.

The invariant of ADR-0027 section 1 -- a level opened later has a larger tag
than every level enclosing it -- now holds for the life of the process, so the
"largest tag is innermost" rule is exact and perturbation confusion cannot
arise from tag reuse.

### 2. A vector-point operator inside a live pass is forward columns

One predicate, `emitPassLiveAt(point)`, holds when a forward pass is live
(the forward level is above 0), a tower or level pass is open, or the point
carries a carrier. When it holds and the point is a collection, `gradient`
computes component `j` as one forward pass on coordinate `j` through the
shared seed/extract protocol, so each component is a level of the enclosing
passes. This is the route `jacobian` and the nested `hessian` already take.
The statically resolved `gradient` and the runtime-closure `gradient` (a
whole-point callable) both use it. The gradient is a rank-1 tensor of tagged
numbers, given the point's shape by `eshkol_gradient_in_point_shape`.

Un-nested passes do not reach the predicate's true arm, so the reverse tape
and its one-pass sweep remain the fast path.

### 3. Forward-over-reverse linearises against the components

When the runtime-closure `gradient` meets a point some of whose components are
the enclosing reverse pass's tape nodes, it runs forward and records, for each
output component `i`,

    grad_i  ~  g_i + sum_k H[i][k] (x_k - value_k)

over every tape-node component `k`, with `H[i][k]` the `e1·ep` coefficient of
one forward evaluation seeded `e1` on `i` and `ep` on `k`.
`eshkol_ad_linear_record` writes it onto the outer tape as one `MUL` edge per
component. The enclosing backward pass carries each edge on through whatever
computed that component, so the record does not depend on which variable the
enclosing pass is seeding. A Scheme-vector point keeps its tape-node elements
as node pointers, the encoding a tensor of tape nodes already uses, so both
spellings reach the same arm.

### 4. Retired refusals

- `eshkol_ad_nested_unsupported` and the `ESH_AD_NEST_UNSUPPORTED` route. With
  the level carrier the route was reachable only when the carrier could not be
  allocated; the seed now raises an allocation error itself.
- `eshkol_ad_curried_gradient_unsupported`: a point computed from the enclosing
  pass's variables is linearised like any other (section 3).

## Consequences

- Nesting depth, per-level order and the number of passes in a process are
  limited only by memory, in every combination of forward and reverse passes.
- The route and the tag field are an internal ABI between the code generator
  and the runtime of the same build. A module compiled before this change
  declares the `i32` route and must be recompiled; the precompiled standard
  library is rebuilt with the compiler, as for every runtime change.
- The forward-over-reverse arm costs one forward evaluation per (output,
  tape-node component) pair, the cost of the direct Hessian's forward-over-
  forward sweep.
- `tests/ad/nested_depth_matrix_test.esk` is the acceptance matrix on native
  JIT, AOT and the VM.

## Alternatives rejected

- **Wider tags in the flags word.** Sixteen more bits postpone the wrap
  rather than remove it.
- **Tags from the nesting depth.** Unique among live passes, but a carrier that
  escapes its pass would share the tag of a later sibling.
- **A forward tangent sweep over the outer tape.** Computing `d(node)/d(seed)`
  for computed components needs a forward rule for every tape node type, a
  second copy of the reverse rules; linearising against the components reuses
  the existing backward pass instead.
- **Making the outer reverse pass forward whenever its body differentiates.**
  Correct, but the body is only known to differentiate after it has run once,
  and the reverse tape's one-pass sweep is the point of `gradient` at large
  points.
