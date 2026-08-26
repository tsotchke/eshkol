# Automatic Differentiation — Architecture

How Eshkol computes exact derivatives. This complements
[../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) (which covers the AD
node opcodes, the tensor backward pass, and the numeric-type boundary) with the
forward-jet / reverse-tape / perturbation-level machinery that governs the
operator behavior documented in [operators.md](operators.md).

Primary source: [`lib/backend/autodiff_codegen.cpp`](../../../lib/backend/autodiff_codegen.cpp)
and [`lib/core/runtime_autodiff.cpp`](../../../lib/core/runtime_autodiff.cpp).

---

## Two engines

Eshkol runs **forward mode** for `derivative` and **reverse mode** for
`gradient`/`jacobian` and the vector-calculus operators built on them
(`hessian`, `laplacian`, `divergence`, `curl`, `directional-derivative`).
`diff` is a third, purely compile-time symbolic engine.

| Mode | Used by | Data structure | Cost | Best when |
|------|---------|----------------|------|-----------|
| Forward | `derivative` | 4-component Taylor jet | O(n) forward pass | 1 scalar input → 1 scalar output (higher-order via nesting) |
| Reverse | `gradient`, `jacobian`, `hessian`, … | tape of `ad_node_t` | O(n) forward + O(n) backward | many inputs, 1 output |
| Symbolic | `diff` | AST rewrite | O(1) at compile time | closed-form derivatives |

---

## Substrates and carriers

The table above describes the **native** engine. Eshkol also differentiates on
the bytecode VM, and the two substrates reach the same numbers through
different carriers. Which carrier answers which operator is declared, per
operator and per substrate, in
[`.icc/ad-carrier-manifest.yaml`](../../../.icc/ad-carrier-manifest.yaml) and
enforced by
[`scripts/gate_ad_shared_node_model.py`](../../../scripts/gate_ad_shared_node_model.py),
which re-derives every declaration from the source rather than believing it.

| Carrier | Where | Vocabulary | Substrates |
|---------|-------|------------|------------|
| `ad_node_t` reverse tape | `inc/eshkol/eshkol.h`, emitted by `autodiff_codegen.cpp` | ~80 typed nodes incl. `AD_NODE_CUSTOM` | native |
| forward jet | `autodiff_codegen.cpp` (`seedForwardAndPush`) | e1/e2/ep slots + Taylor tower | native |
| `VmDual {primal, tangent}` | `vm_dual.c` | 16 flat forward-dual ops | VM |
| `VmHyperDual {f, f1, f2, f12}` | `vm_hyperdual.c` | second-order forward | VM |
| `AdTape`/`AdNode` Wengert tape | `vm_autodiff.c` | 17 ops, int-indexed | **both** — the Scheme-visible `ad-*` primitives |

Every carrier in that table is **exact**: each propagates derivatives by the
chain rule. No default AD path on either substrate uses a finite difference.
The one deliberate difference quotient in the tree is `record-fd-op!` in
[`lib/core/ad/tape.esk`](../../../lib/core/ad/tape.esk) — the pure-Scheme
escape hatch for an opaque forward function that supplies no analytic adjoint,
named at its call site and ledgered in the carrier manifest's `fd_allowlist`.

The forward carriers are scalar. A vector field whose components are packed
into a `VmTensor` loses its tangents at construction, because `VmTensor::data`
is a bare `double*`. `divergence` and `curl` therefore accept a field that
returns a **list or a vector** and raise a named diagnostic for one that
returns a tensor, rather than approximating it.

### Build item — `AD_NODE_CUSTOM` on the VM

`AD_NODE_CUSTOM` carries an externally supplied vector-Jacobian product (see
below) and is reachable only from the native `ad_node_t` tape. Four things
stand between it and the VM's shared Wengert tape:

1. **Vocabulary.** `AdOpType` (`lib/backend/vm_autodiff.c`) has 17 members and
   no `CUSTOM`.
2. **Node shape.** `eshkol_custom_vjp_t` lives in `ad_node_t::saved_tensors[0]`.
   `AdNode` has a single `double saved` slot — no `saved_tensors`, no
   `num_saved`.
3. **Input handles.** `eshkol_custom_vjp_t::inputs` is `ad_node**`. `AdNode`
   addresses parents by index into an array that is relocated on growth, so a
   stored node pointer would dangle.
4. **Reverse driver.** `eshkol_ad_node_custom_backward`
   (`lib/core/runtime_autodiff.cpp`) is hard-typed to `ad_node_t*`, and the
   VM's `ad_backward` switches on `AdOpType` with no default hook.

The work, in order:

- Make the VM tape's node storage **chunked and never relocated**, so an
  `AdNode*` stays valid for the tape's lifetime. This also removes a latent
  hazard for any future pointer-holding node type.
- Add `AD_CUSTOM` to `AdOpType` and `void** saved; int num_saved;` to `AdNode`.
- Factor the universal reverse rule
  (`input_i->gradient += upstream * dy/dx_i`) out of
  `eshkol_ad_node_custom_backward` into one function parameterised over the
  accumulate step, and call it from both `ad_backward`'s new `case AD_CUSTOM:`
  and the native driver — so the two substrates cannot disagree about what a
  custom VJP means.
- Route the VM's `(gradient f x)` to the tape when `f` is reverse-mode-marked,
  so a custom node is reachable at all: today the VM's `gradient` is the
  forward `VmDual` carrier and builds no tape.

---

### Build item — `AD_NODE_SQUARED_DISTANCE` has no Scheme surface and no VM path

`d²(x,y)` on the space forms and their products
(`lib/bridge/space_form_ad.cpp`, `AD_NODE_SQUARED_DISTANCE`) is a C bridge
entry point. It records onto the native `ad_node_t` tape and is answered by the
`native-ad-node` carrier, which `.icc/ad-carrier-manifest.yaml` declares
`shared: false`. There is no `(squared-distance …)` builtin, no native call id,
and no row in `tests/vm_parity/PARITY.tsv` — so the structural carrier gate has
nothing to grade, which is correct rather than convenient, and this section is
here so the absence is recorded rather than inferred.

Two things gate a VM path, and neither is a wiring change:

1. **The VM has no `ad_node_t`.** `lib/backend/vm_autodiff.c` carries its own
   scalar `AdNode` with a single `double saved` slot. This op keeps its factor
   list — form, dimension, curvature and weight per factor — in
   `saved_tensors[0]`, which has no counterpart there. The chunked-storage and
   `void** saved` work listed for `AD_NODE_CUSTOM` above is the same
   prerequisite.
2. **The VM's `gradient` builds no tape at all.** It is the forward `VmDual`
   carrier, which is scalar and cannot carry a tangent through a `VmTensor`
   (`fork_debt.blocked_on`). A point on a manifold is a vector by construction,
   so there is nothing for the forward carrier to seed.

Until both land, adding a VM opcode for this op would mean a *second*
implementation of the geometry answering the same name — the exact fork
`gate_ad_shared_node_model.py` was written to size and cap. The honest
intermediate state is the one recorded here: native only, declared as such, and
no parity row claiming otherwise.

What is worth stating alongside that, because it is the reason the op exists:
this node deliberately does **not** inherit `ad_hyperbolic_distance`'s
coincidence refusal. `d` has a cone point at `x == y` and refusing there is
right. `d²` is smooth on the whole injectivity ball, with
`grad_x d² = -2 log_x(y)` vanishing exactly at coincidence — and it cannot be
obtained from `d` by squaring, because `2·d·grad d` is `0·(no limit)` at the
diagonal. The two ops are kept in separate translation units with separate
dispatch arms for that reason: a shared implementation would eventually
propagate one op's refusal to the other, and the diagonal is not an edge case
for a scoring function, it is every self-attention row.

---

## Forward mode — the 4-component jet

A "dual number" in Eshkol is **not** the classic 2-component `{value,
derivative}`. It is a truncated **bivariate** Taylor jet with two independent
perturbation symbols `e1`, `e2` (`e1² = e2² = 0`):

```
v = f0 + f1·e1 + f2·e2 + f3·e1·e2
```

stored as the LLVM struct `{primal, d1, d2, d12}`. A single-level derivative
only touches `f0`/`f1`, so it is exactly backward-compatible with a plain dual
`{primal, tangent, 0, 0}`.

Why two slots: **each nesting level gets its own perturbation symbol**. This is
the standard cure for *perturbation confusion* — when you nest `derivative`
inside `derivative` (or differentiate a Hessian w.r.t. two arguments), level 0
seeds `e1` (field 1) and level 1 seeds `e2` (field 2). The mixed `e1·e2`
coefficient (`d12`) carries the exact second-order term. Every arithmetic op
propagates all four components in closed form (see `dualUnaryChain` in
`autodiff_codegen.cpp`) — no finite differences, no recursion.

**Two levels are exact; three is not.** A 4-component jet cannot carry a third
independent perturbation, so a third nested `derivative` *aliases* onto `e2` and
the compiler emits a `nested derivative depth N exceeds exact 2-level forward
AD` warning rather than silently returning a wrong-but-plausible number.

---

## Perturbation levels — `__ad_pert_level`

The perturbation level is a **runtime** counter, not a compile-time lexical
depth. It lives in thread-local storage:

```c
// lib/core/runtime_autodiff.cpp
thread_local uint64_t __ad_pert_level = 0;
```

and is exposed to codegen as a global loaded/stored around every forward-mode
call (`seedForwardAndPush` / `popAndExtractForward`). The counter is pushed
before a `derivative`/`gradient` evaluates its body and popped afterward.

This runtime push/pop is what makes nesting correct **across a function-call or
named-let TCO boundary**. A compile-time lexical depth could not see across the
call boundary, so a `derivative` reached *through* a called function would
clobber the outer perturbation (this was the ESH-0070 class of bug). A runtime
counter is invariant under TCO re-entry by construction: the inner call reads
the level the outer one left live (level 1 → slot `e2`) and therefore seeds a
distinct slot instead of overwriting the outer's `e1`.

---

## Reverse mode — the tape

`gradient`/`jacobian` build a computational graph of `ad_node_t` records during
the forward pass, then propagate gradients backward from the output. The tape,
node structure, opcode enum, and the tensor backward-pass dispatch are
documented in
[../../breakdown/AUTODIFF.md#reverse-mode-ad-computational-graph](../../breakdown/AUTODIFF.md#reverse-mode-ad-computational-graph).

The tape stack supports **32 levels** of nesting and is **per-thread**
(`thread_local __ad_tape_stack[32]`), so `parallel-map` of a gradient function
is tape-safe. (Two shared globals — `__current_ad_tape`, `__ad_mode_active` —
are not thread-local; see the AUTODIFF.md "Parallel Tape Management" note.)

---

## Mixed mode — reverse-over-forward (v1.3, ESH-0093)

The headline v1.3 AD change (#113) makes an **outer vector `gradient` (reverse
tape) over an inner `derivative` (forward jet)** propagate the dependency on
captured tape parameters. Mechanism, from
[`runtime_autodiff.cpp`](../../../lib/core/runtime_autodiff.cpp) and
`autodiff_codegen.cpp`:

1. While a forward pass is live (`__ad_pert_level > 0`), reverse-tape nodes that
   flow into scalar arithmetic are **jet-lifted** to dual numbers: `value =
   node->value`, and the `e2` slot is seeded with `1.0` iff the node *is* the
   published seed (`eshkol_ad_seed_flag`).
2. The forward 4-jet then carries the mixed `e1·e2` coefficient through the
   inner computation — no new arithmetic rules needed.
3. At the `derivative` return site, `eshkol_ad_mixed_record` records the result
   back onto the outer tape with a backward edge `a12 = d(result)/d(seed)`, so
   the outer reverse pass sees the correct sensitivity.

This is exercised end-to-end by
[`tests/ad/mixed_mode_ad_test.esk`](../../../tests/ad/mixed_mode_ad_test.esk)
(15/15 on this build), including nonlinear captures, tensor-literal points
(reverse-tape path) vs `vector` points, and a 1000-iteration stability loop.

```scheme
;; f(x;p0)=p0·x²  ->  ∂/∂p0 [ d/dx f @2 ] = 4
(gradient (lambda (p) (derivative (lambda (x) (* (vref p 0) (* x x))) 2.0))
          (vector 3.0))
;; => #(4)
```

Reverse-**over-reverse** is covered as well. `gradient` of `gradient` at a
vector point (formerly **ESH-0096**) and `gradient` of a *named* inner function
(formerly **ESH-0078**) both return the true second-order value on this build,
where each used to return zeros:

```scheme
(gradient (lambda (v)
            (vref (gradient (lambda (w) (* (vref w 0) (vref w 0) (vref w 0))) v) 0))
          (vector 2.0))
;; => #(12)
```

This holds for the *direct* nested form shown above. The **curried** route —
`(define g (gradient f))` then `(jacobian g point)` — raises
`unsupported nested differentiation` rather than answering (a loud refusal,
not a silent zero; SW-05); see KNOWN_ISSUES.md. Use `(hessian f point)` for
exact second order.

See [support-matrix.md](support-matrix.md) for the per-cell evidence.

---

## Callable-arity recovery — `gradient` through wrappers and curried forms

`gradient` is exact reverse-mode AD **regardless of how the callable is reached**
— named directly, passed in through a function parameter, wrapped, or applied in
curried form. There is **no finite-difference fallback** anywhere in the gradient
path; a claim of "FD" for any of these forms is stale.

The direct-call path always unpacked an N-element point into N scalar arguments
using the callable's arity. A first-class-tensor-loss path added later
(reverse-mode element seeding) unconditionally captured every vector/list/tensor
point and invoked the closure with a **single tensor argument**, ignoring the
callable's real arity. That shadowed the correct forward path, so when the
callable was reached *indirectly* — `(gradient f point)` where `f` came through a
parameter, or the curried `((gradient f) point)` — a multi-parameter scalar loss
such as `(loss x y)` was invoked as `loss(<tensor>)` and its scalar body
misdispatched.

The fix recovers the callable's arity from its **closure metadata** (no closure
ABI change) and unpacks the point accordingly, so the indirect and curried forms
are byte-identical to the direct call for scalar multi-argument, vector, and
non-polynomial losses, on both the JIT and AOT paths. A 25-check suite pins the
direct/indirect/curried equivalence.

```scheme
;; direct, indirect, and curried all agree exactly:
(define (loss x y) (+ (* x x) (* y y)))
(gradient loss (vector 3.0 4.0))            ; direct
(define (apply-grad f pt) (gradient f pt))
(apply-grad loss (vector 3.0 4.0))          ; through a parameter
((gradient loss) (vector 3.0 4.0))          ; curried
;; => #(6 8) in every case
```

A related fix follows transitive closure captures in a **custom vector-Jacobian
product**: a custom-VJP backward closure that reached a captured value through an
intermediate closure previously had its contribution silently dropped (zero
gradient); the capture walk now follows transitive references so the node
contributes its full sensitivity.

## Numeric boundary (summary)

The AD engine operates on `double` and the jet/dual structs only. Bignums,
rationals, and complex numbers do **not** carry derivatives — see
[../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) ("Numeric Type
Interactions with AD") for the exact conversion behavior at the boundary.
Convert exotic numeric inputs to `double` before entering an AD context.

---

## Performance (measured, this build, arm64/Apple Silicon, AOT)

| Workload | Result | Timing |
|----------|--------|--------|
| 2nd-order scalar derivative of a cubic, 10⁶ calls | `1.8e7` (18/call) | 0.41 s user / 0.74 s wall → ≈ **0.4 µs/call** |
| 7168-dim `gradient` of Σ xᵢ² (single call) | `#(3 … 3)` | 5.56 s user / **6.30 s wall** |

Method: AOT-compiled (`eshkol-run f.esk -o bin`), timed with `/usr/bin/time`.
The 2nd-order figure is the effective per-call cost including the accumulation
loop. Forward mode is O(1) space (just the jet); reverse mode is O(n) space in
the tape.

---

## See also

- [operators.md](operators.md) — per-operator API, capture rules, nesting table
- [support-matrix.md](support-matrix.md) — oracle matrix and open cells
- [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) — opcodes, tensor backward, tape internals
