---
kind: explanation
status: current
owner-area: ad
since: v1.3.5
sources:
  - inc/eshkol/eshkol.h
  - lib/core/runtime_taylor.c
  - lib/backend/autodiff_codegen.cpp
  - lib/core/runtime_regions.cpp
  - lib/backend/vm_numeric.h
  - lib/backend/vm_dual.c
  - lib/backend/vm_native.c
  - lib/backend/vm_region_evac.c
  - scripts/build-wasm-repl.sh
  - .icc/ledger/entries/SW-154.yaml
  - .icc/ledger/entries/SW-193.yaml
  - .icc/ledger/entries/SW-194.yaml
---
# ADR-0027: Nested differentiation uses one recursive Taylor level carrier

**Status:** Accepted
**Ledger:** SW-154; SW-193 and SW-194 are closed against the same carrier, and SW-206..SW-209 and SW-213 bring every operator onto it
**Scope:** Native LLVM code generation (JIT and AOT), the bytecode VM, and the
browser VM, which is the same VM compiled to WebAssembly.

## Context

`derivative`, `derivative-n` and `taylor` each open a *differentiation pass*.
Programs nest passes in two ways: through the evaluation point
(`(derivative (lambda (t) (derivative-n f (+ 1 t) 2)) 0.0)`) and through a
captured variable (`(derivative-n (lambda (a) (derivative-n (lambda (b) (* a b))
1.0 2)) 2.0 2)`). Mathematically every such program is a truncated series in
several independent perturbations.

Before this decision each engine represented nesting with a fixed set of
special cases on top of a flat carrier:

- The native Taylor tower held one value series and at most one first-order
  companion series (`ESH_TAYLOR_TANGENT_FLAG`, `carry_epoch`), plus a
  hyper-dual second companion for exact arithmetic. The routes `RIDE`,
  `CARRY_TWR` and `CARRY_JET` of `eshkol_ad_nested_seed` chose which pass rode
  the companion.
- The native 8-jet (`eshkol_dual_number_t`) holds three nilpotent directions
  (`e1`, `e2`, `ep`) and nests only with itself.
- The VM `VmDual` mirrored the tower with `tangent_coeff`, `tangent2_coeff`
  and `mixed_coeff` lanes and the same ride/carry seeds.

Every shape outside those cases either raised ("both passes have order >= 2",
"through a captured carrier") or answered 0 with exit status 0: two enclosing
first-order levels around an order-2 pass (SW-154), nested holomorphic
complex derivatives (SW-193), and the browser VM's scalar Hessian of `expt`
(SW-194). The limit was the representation, not the mathematics.

## Decision

### 1. The level carrier

A pass that is nested inside another live pass runs as a **level**. A level is
identified by an **epoch**, a tag drawn from the process-wide epoch counter
when the pass seeds, so a pass opened later always has a larger epoch than
every pass enclosing it.

A **level carrier** of epoch `E` and order `K` is the truncated series

    c[0] + c[1] t_E + ... + c[K] t_E^K

in its own perturbation `t_E`, whose coefficients `c[i]` are *any number*:

- an exact integer, bignum or rational, or a flonum;
- a carrier of a strictly enclosing level (an epoch smaller than `E`);
- a native 8-jet or VM scalar dual, which is how the outermost first-order
  `derivative` carries its perturbation;
- a classic (non-level) Taylor tower of an enclosing pass.

Invariant: a level carrier of epoch `E` never contains, at any depth, a
carrier of epoch `>= E`. Because coefficients are numbers of the enclosing
levels, depth and per-level order are unbounded, and two perturbations are
combined only when their epochs are equal, so perturbation confusion cannot
occur.

### 2. Arithmetic

Every numeric primitive on carriers reaches one entry point per engine:
`eshkol_taylor_binary_tagged` / `eshkol_taylor_unary_tagged` on native and the
`vm_dual_*` operations on the VM. Each takes the **level path** when an operand
is a level carrier, or when two operands are towers of different epochs. The
level path:

1. picks the active epoch `E`, the largest epoch among the operands;
2. reads an operand of epoch `E` as its coefficient series, and any other
   operand as the constant series `(operand, 0, 0, ...)` — the operand is kept
   whole, never reduced to its primal;
3. runs the standard Taylor recurrences (Cauchy product, division, `exp`,
   `log`, coupled `sin`/`cos`, the constant-power recurrence, and the
   compositions `tan`, `sqrt`, `sinh`, `cosh`, `tanh`, `sigmoid`, `relu`,
   `abs`, `expt`) with every coefficient operation dispatched recursively
   through the same generic arithmetic.

Exact arithmetic follows the numeric tower's contagion: a coefficient is
exact while every value it was computed from is exact. `expt` with an exact
integer exponent is computed by repeated multiplication, so it stays exact and
is defined at a zero base. Branches (`abs`, `relu`, comparisons) read the
primal: `c[0]` taken recursively down to a plain number.

A native 8-jet coefficient uses the 8-jet's own algebra, implemented in the
runtime: the product over subsets of `{e1, e2, ep}`, and a unary function as
`f(x0 + n) = sum_k f^(k)(x0) / k! * n^k` for the nilpotent part `n`, with the
`f^(k)(x0) / k!` taken from the same double recurrences. The VM scalar dual is
already a `VmDual` and uses `vm_dual_*`.

A complex value stays the outermost structure: its real and imaginary parts
are carriers (ADR-0025), and complex arithmetic is carried out on the parts
with the generic real arithmetic above.

### 3. When a pass runs as a level

At the seed site the engine decides:

- a `derivative-n`/`taylor` pass (and the exact-tier or chain-routed
  `derivative`, which run as order-`k` tower passes) runs as a level when any
  forward pass is live — the forward perturbation level is above 0 or a tower
  pass is open — or when its point is already a carrier;
- a first-order `derivative` pass runs as an order-1 level when a tower or
  level pass is open, its point is a tower or level carrier, or two jet levels
  are already live. Otherwise it keeps the 8-jet on native and the scalar dual
  on the VM, which is also what nests `derivative` inside `derivative`. The
  8-jet holds two forward directions (`e1`, `e2`) and the reverse-seed slot
  `ep`; a third live jet level has no slot of its own, so it is a level.

Every other operator reaches the same protocol:

- `gradient` and `hessian` at a scalar point are the order-1 and order-2 tower
  passes whenever the point is exact or a pass is live; nested, they are levels.
- `jacobian` met while a pass is live, or at a point carrying a carrier,
  computes each column as a forward pass of its own (seed one coordinate,
  extract the derivative of every output) and assembles a tensor of tagged
  numbers; the reverse tape, which carries one raw double per node, is used
  only when nothing encloses it.
- Each operator is its own pass: the tower mode of an enclosing pass, set while
  its differentiand is emitted, is cleared for the operator's own emission.

An un-nested pass seeds exactly as before: the classic F64 or exact tower on
native, the jet, or the VM scalar dual. The fast paths are unchanged.

The seed of a level of order `K` at point `p` is `(p, 1, 0, ..., 0)` with the
unit exact when the primal of `p` is exact and `1.0` otherwise. Order 0 is
evaluation: the point is passed through unchanged.

### 4. Extraction

When the level pass returns, the result `r` is read in the level's own epoch:

- `derivative-n` (and `derivative`) of order `k`: `k! * c[k]`, itself a number
  of the enclosing levels, so the enclosing pass keeps its perturbations;
- `taylor` of order `K`: the list `(c[0] ... c[K])`;
- when `r` is not a carrier of epoch `E`, the function does not depend on this
  level: order 0 gives `r`, higher orders give 0 (exact when the point was
  exact, `0.0` otherwise).

A structural zero produced by the recurrences is converted the same way, so
exactness follows the point, as it does for an un-nested pass.

### 5. Native interface

- `esh_taylor_t` with `ESH_TAYLOR_COEFF_CARRIER` (2) in the coefficient field
  is a level carrier: `c[]` holds `order_k + 1` `eshkol_tagged_value_t`, and
  `exact_c` points at the same storage. It carries no companion lanes.
- `int32_t eshkol_ad_nested_seed(arena, point, order, pert_level, tower_pass,
  tower_depth, out)` returns `ESH_AD_NEST_NONE` (0) for an un-nested pass, or a
  packed route: `ESH_AD_NEST_LEVEL` (2) in bits 0-7, the level's epoch in bits
  8-23, and bit 24 set when the seed unit is inexact. Codegen passes the
  runtime forward level and the tower depth it maintains, and pushes the tower
  context for every level pass, jet arm included, so a pass nested inside a
  level sees it.
- `void eshkol_ad_nested_extract(arena, result, route, order, want_list, out)`
  performs section 4; `want_list` selects the `taylor` form.
- Region evacuation walks a level carrier's `c[]` like an exact tower's.

### 6. VM and browser interface

- `VmDual` gains the kind `VM_DUAL_KIND_LEVEL` (2) with `epoch`, `order` and
  `VmDual** lcoeff` (`order + 1` entries). Each coefficient is a `VmDual` of any
  kind; a plain number is a scalar `VmDual` with zero tangent, carrying its
  exact value in `eprimal`. The `primal`/`eprimal` fields of a level hold its
  primal, so every existing reader of a dual's primal (comparisons,
  `as_number_vm`) sees the right value.
- Every `vm_dual_*` arithmetic entry dispatches to the level path first.
- A VM-wide count of live forward passes (`derivative`, `derivative-n`,
  `taylor`, and the operators built on them) drives the section 3 rule.
  `derivative` at a point that is already a dual, or while a pass is live,
  runs as an order-1 level instead of raising.
- Extraction returns a VM `Value`: a coefficient that is a scalar constant
  becomes its number, anything else a `VAL_DUAL`.
- Region evacuation walks `lcoeff` recursively.
- The browser VM is rebuilt from the same sources with
  `scripts/build-wasm-repl.sh`; `site/static/eshkol-vm.{js,wasm}` are those
  build outputs, and the browser differs from the VM in no carrier code.

### 7. Retired special cases

The level carrier replaces, on both engines: the `RIDE`, `CARRY_TWR` and
`CARRY_JET` routes; the capture companion (`carry_epoch`,
`eshkol_ad_tower_carry_result`, `eshkol_ad_jet_extract_tower`,
`eshkol_ad_nested_capture_unsupported`); the hyper-dual second companion used
for two foreign epochs; and the VM ride/carry seeds and the VM refusal of a
nested `derivative`. The first-order companion series stays for exactly one
purpose, reverse-over-Taylor (the reverse-seed tangent of
`docs/design/AD_TAYLOR_TOWER.md` section 8), which is not a forward level.

## Consequences

- Nesting depth and the order of every level are limited only by memory, on
  every engine, with the same answers on native JIT, AOT, the VM and the
  browser.
- A nested pass allocates one tagged series per operation and its
  coefficient arithmetic is generic; un-nested passes keep their fast paths.
- `tests/ad/nested_towers_matrix_test.esk` and
  `tests/ad/nested_operator_matrix_test.esk` are the acceptance matrices on
  native JIT, AOT and the VM.

## v1.3.5 carrier WIP reconciliation

An earlier fixed-lane carrier design predates this carrier. Its behaviors are
provided by the following current facilities. Each listed repro passed on the native JIT and bytecode VM;
`tests/ad/dense_tensor_operand_boundary_test.esk` additionally passed 29/29
native checks after the local standard library was built.

| Earlier design element | Current facility and repro |
| --- | --- |
| `autodiff_codegen.h/.cpp` complex seed and extract; `runtime_taylor.c` coefficient zipping | Per-level native seeding and extraction preserve each complex component's enclosing carrier; `tests/ad/complex_nested_carrier_test.esk` passes 30 checks. |
| `llvm_codegen.cpp` duplicate math dispatch removal | The shared numeric builtin dispatch remains the sole native path; `tests/ad/complex_nested_carrier_test.esk` and `tests/ad/vm_complex_intermediate_taylor_test.esk` cover composed complex math. |
| `vm_complex.c`, `vm_numeric.h`, `vm_native.c` complex carrier ingress, arithmetic and projection | `VmComplex` keeps component carriers and `VM_DUAL_KIND_LEVEL` represents nested passes; the same two complex tests and all 118 checks in `tests/ad/complex_carrier_math_test.esk` pass on the VM. |
| `vm_dual.c` hyper-Taylor allocation, algebra, projection and unary rules | Recursive `lcoeff` arithmetic and `vm_dual_level_coefficient` replace fixed companion lanes; `tests/ad/site_scalar_hessian_test.esk` and `tests/ad/nested_carrier_orderings_test.esk` exercise exact order and nesting. |
| `vm_native.c` scalar Hessian path | Shared level application and extraction preserve the exact point; `tests/ad/site_scalar_hessian_test.esk` returns exact `4/3` for `(expt x 4)` at `1/3`. |
| `vm_parallel.c`, `vm_region_evac.c` carrier ownership | Worker cloning and region evacuation walk recursive level coefficients; `tests/vm_parity/corpus/parallel_ad_carrier_lifetime.esk` and `tests/memory/vm_region_evac_level_carrier_test.esk` pass. |
| CMake registration, two AD test files, and the SW-194 ledger entry | Both tests are registered, SW-193/SW-194 are closed, and the ledger entry is generated. |

The focused parity repro is `(taylor (lambda (x) (* x 1)) 1/3 3)` =
`(1/3 1 0 0)` and `(derivative (lambda (x) (tensor-ref (tensor x) 0))
1/3)` = `1` on both engines. No WIP runtime hunk needs transplanting because
its fixed-width carrier representation is superseded by the level carrier.

## Alternatives rejected

- **More companion lanes.** Each additional lane covers one more shape and the
  next nesting still fails; this is the path that produced SW-154.
- **Nested hyper-duals (2^n components).** Exponential in depth, and cannot
  carry orders above one per direction.
- **Pseudo-epochs for jet slots.** Restating 8-jet slots as synthetic levels
  and back at extraction ties the carrier to the jet's slot layout; keeping the
  jet as a coefficient with its own algebra needs no restatement.
