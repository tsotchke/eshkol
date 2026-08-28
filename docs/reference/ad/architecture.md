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
### Build item — `AD_NODE_TENSOR_*` producers on the compiled and VM paths

The tensor node family (`AD_NODE_TENSOR_MATMUL` … `AD_NODE_FRECHET_MEAN`) is
recorded today only by the C entry points of the external-tensor bridge,
`lib/bridge/qllm_bridge.cpp`. Every one of those nodes now has a producer, so
its backward rule is reachable and gradchecked *through* that producer rather
than through a hand-built fixture. What remains is reachability from the two
other paths, and neither is a wiring change.

**Compiled Eshkol (JIT and AOT).** No compiled program can create one of these
nodes at all. `AutodiffCodegen::recordADNodeTensor` exists and has exactly one
call site, dead behind `kDenseTensorADNodesEnabled` in
`lib/backend/llvm_codegen.cpp`; the block comment there records that flipping
the flag SIGSEGVs rather than yielding a slower-but-correct gradient, for three
independent reasons:

1. `recordADNodeTensor` stores NULL into `tensor_gradient`, while the reverse
   pass *selects* the tensor backward by testing that field non-null —
   constructor and consumer each wait for the other;
2. the node it builds is dropped: the function goes on to return a plain
   tagged tensor, so nothing downstream can find it;
3. under AD the scalarizing path leaves AD-node *pointers* in the result
   tensor's elements, which is what `tensor-sum` and friends consume, so a
   dense node would sever the chain at the next tensor op.

That is ADR-0002 Position A (the dense resident tape), scheduled for v1.6. Until
it lands, `(embedding …)` codegen emits a plain gather with no tape node, and
`(gradient (lambda (W) … (embedding idx W)))` records nothing for the lookup.

**The VM.** `lib/backend/vm_autodiff.c` has its own scalar `AdNode`
representation; no `vm_*.c` file references `ad_node_t` or any `AD_NODE_*`
constant. `frechet-mean` (native call id 817) therefore cannot record an
`AD_NODE_FRECHET_MEAN`, and the same chunked-storage and shared-reverse-rule
work listed for `AD_NODE_CUSTOM` above is the prerequisite. Its forward is
nonetheless already shared with the bridge producer —
`inc/eshkol/backend/frechet_mean_core.h` is included by both
`lib/backend/vm_geometric.c` and `lib/bridge/qllm_bridge.cpp` — so the VM opcode
and the differentiable path cannot drift apart on what the mean *is* while the
tape work is pending. That matters more here than for most ops: the Fréchet
backward is implicit differentiation of a stationarity condition and refuses
above a residual bar, so a forward that drifted would hand the derivative means
it rejects.

---

## The AD node registry

Every AD node type is declared **once**, in
[`inc/eshkol/ad_node_registry.def`](../../../inc/eshkol/ad_node_registry.def), as
an X-macro row:

```c
ESHKOL_AD_NODE(NAME, VALUE, PAYLOAD, TENSOR_BACKWARD, BRIDGE_FN)
```

There are **83 rows**, values `0`–`82`, dense. `VALUE` is explicit and asserted
equal to the row's ordinal, because these values are an ABI: emitted LLVM IR
compares `node->type` against integer literals and serialized tapes carry them.

The enum (`AD_NODE_##NAME` in `inc/eshkol/eshkol.h`), the tensor backward dispatch
table and the dispatcher's `switch` are all generated from this one file, so
"registered" is a compile-time fact rather than an intention. Adding a node type
means writing its row; there is no way to add one and have its gradient silently
vanish.

### The five dispositions

`TENSOR_BACKWARD` says where — or whether — the node's adjoint lives.

| Disposition | Rows | Meaning |
|---|---:|---|
| `SCALAR_ADJOINT` | 44 | the adjoint is computed by the scalar reverse sweep |
| `BRIDGE` | 18 | an exact tensor backward, named by `BRIDGE_FN` |
| `INLINE` | 14 | the backward is emitted inline at the recording site |
| `UNREGISTERED` | 4 | an **explicit registered refusal** |
| `LEAF` | 2 | a tape leaf; nothing to propagate |
| `CUSTOM_VJP` | 1 | the user supplied the vector-Jacobian product |

`UNREGISTERED` is the disposition that makes this a registry rather than a table.
It means: no exact backward exists for this node in the tensor dispatcher, and
rather than return a plausible zero the dispatcher **aborts, naming the node
type**. It is a declaration, not an oversight — the row had to be written.

```
AD backward dispatch: no exact backward is registered for tensor node type N
(NAME). ad_node_registry.def declares it UNREGISTERED, which means this gap is
known and stated, not discovered here. Refusing to return a gradient of zero for
an operation that has one.
```

The `BRIDGE` arm carries the mirror-image guard: a row that says `BRIDGE` but
whose named function is missing from the generated table refuses rather than
drops the gradient — and because the table's initializer names the symbol
directly, a `BRIDGE` row naming a function that does not exist **does not
compile**.

### Why there is no `default:`

The dispatcher's `switch` has an explicit `case AD_NODE_TYPE_COUNT:` arm and no
`default:`, and the translation unit is built with `-Werror=switch-enum`. A
`default:` would accept any future enum member into "nothing to do", which is
precisely how four tensor ops came to return zero gradients in silence. With the
default removed, a member that says nothing is a compile error rather than a
wrong number.

Three independent totality checks back the registry up: a compile-time assertion
that the coverage table spans every declared row, a startup assertion that no row
is `NULL` (a designated-initializer table silently zero-fills any index nobody
wrote, and a zero row reads as "walk this, I know how"), and a runtime
out-of-range guard. A static assertion in `inc/eshkol/eshkol.h` pins
`ESHKOL_AD_NODE_REGISTRY_ROWS == AD_NODE_TYPE_COUNT` — no gaps, no duplicates.

### Node blocks

| Values | Block |
|---|---|
| 0–11 | core arithmetic |
| 12–18 | activations |
| 19–28 | tensor ops (all `INLINE`) |
| 29–32 | transformer (all `INLINE`) |
| **33–40** | **qLLM geometric** |
| 41–45 | additional math |
| 46–53 | Phase-4 activations |
| 54–66 | complete math |
| **67–80** | **qLLM bridge tensor nodes** (all `BRIDGE`) |
| 81 | `ATAN2` |
| 82 | `CUSTOM` |

## Exact geometric backwards

The 33–40 block is the one the registry was built for: four of its node types
reached an asserted-impossible `default:` and returned zero. The numeric band that
default reasoned from — "tensor ops are 19–32 and 67–80" — never covered 33–40.

| Value | Node | Disposition | Backward |
|---:|---|---|---|
| 33 | `HYPERBOLIC_DISTANCE` | `BRIDGE` | `tensor_hyperbolic_distance_backward` |
| 34 | `POINCARE_EXP_MAP` | `BRIDGE` | `tensor_poincare_exp_map_backward` |
| 35 | `POINCARE_LOG_MAP` | `BRIDGE` | `tensor_poincare_log_map_backward` |
| 36 | `TANGENT_PROJECT` | `UNREGISTERED` | — |
| 37 | `GEODESIC_ATTENTION` | `BRIDGE` | `tensor_geodesic_attention_backward` |
| 38 | `MOBIUS_ADD` | `UNREGISTERED` | — |
| 39 | `MOBIUS_MATMUL` | `UNREGISTERED` | — |
| 40 | `GYROVECTOR_SPACE` | `UNREGISTERED` | — |

The four exact rules live in `lib/bridge/tensor_backward.cpp`; their producers are
`ad_hyperbolic_distance`, `ad_poincare_exp_map`, `ad_poincare_log_map` and
`ad_geodesic_attention` in `lib/bridge/qllm_bridge.cpp`.

Two of the rules are compositions this file already differentiates for the Fréchet
mean, and they **reuse** that machinery — `FrechetGeometry::mobius_add_with_jacobians`
and `FrechetGeometry::log_map_with_jacobians` — rather than re-deriving it: deriving
the same Möbius Jacobian twice is how a sign error survives a gradient check on the
other copy.

The remaining four have **no producer anywhere in the tree** — the emitted scalar
sweep reads them, nothing writes them. They stay `UNREGISTERED` so that a future
producer which records one as a tensor node inherits the abort rather than the
silence.

### Six backwards, two different kinds

The v1.3.5-evolve wave is often summarised as "six exact geometric backwards". The
distinction is worth keeping:

- **Four newly written** — the `BRIDGE` rules above (ledger SW-65), which did not
  exist and whose nodes returned zero.
- **Two newly reachable** — `TENSOR_EMBEDDING` (78) and `FRECHET_MEAN` (80) already
  had exact rules, but no producer existed anywhere in the tree, so both were
  validated only against `ad_node_t` structures the tests assembled by hand. Adding
  `ad_tensor_embedding` and `ad_frechet_mean` made them reachable and gradchecked
  through a real producer.

### Where these rules refuse, and why that is not conservatism

The Riemannian distance `d(x, y)` behaves like `|x − y|` near coincidence: at
`x = y` it has no derivative, only a subgradient set. Any number returned there is
invented. So each of these rules refuses rather than picking one:

- **`hyperbolic-distance`** — the two points coincide, or one lies outside the
  Poincaré ball. "It is a cone point, and every value a rule could return there is
  invented. Refusing."
- **`poincare-log-map`** — no finite `log_x(y)` exists at the operands. The forward
  clamps `artanh`'s argument and returns a value anyway; "that value is fabricated,
  and differentiating it would launder the fabrication into a gradient."
- **`geodesic-attention`** — a query row coincides exactly with a key row. This has
  a consequence worth stating outright: **scoring by distance makes the op
  non-differentiable whenever `Q` and `K` are the same tensor**, which is the
  ordinary self-attention case. The message names the row, column, batch and head.
- **`poincare-exp-map`** — uses a series expansion below a small-tangent threshold
  of `1e-6`, and differentiates the *mathematical* map rather than the forward's
  `|v| < 1e-15 → return x` shortcut branch.

### The Fréchet stationarity gate

`FRECHET_MEAN`'s backward is implicit differentiation of the stationarity condition
`Σᵢ wᵢ log_μ(xᵢ) = 0`. Those formulas are the derivative **at** the fixed point;
away from it they return a plausible but wrong gradient. So the rule measures the
residual and refuses above a bar.

- Default relative tolerance `kFrechetResidualTol = 1e-9`
  (`lib/bridge/tensor_backward.cpp`), overridable per node through `params` slot 3
  (a non-finite or non-positive value falls back to the default).
- The forward's own bar is `ESHKOL_FRECHET_RESID_TOL = 1e-9` with
  `ESHKOL_FRECHET_MAX_ITERS = 256` (`inc/eshkol/backend/frechet_mean_core.h`),
  deliberately matched, so a forward that returns successfully produces a mean the
  derivative will accept.
- The residual is measured in **Riemannian units**, not ambient ball coordinates —
  scaled by the conformal factor `λ = 2/(1 − c·⟨μ,μ⟩)` and normalised by the weight
  sum. This is not cosmetic: with the ambient scale the forward accepted means wrong
  by `8.8e-8` and `7.6e-6` as converged, and the rule would then have differentiated
  them and returned exactly the plausible wrong gradient the gate exists to prevent.
- The refusal names the absolute residual, the relative residual, the tolerance, the
  point count and the dimension, and tells you the two ways forward: tighten the
  forward iteration, or raise `params` slot 3 deliberately.

This is also why both the VM opcode and the AD producer compute the mean in **f64**.
An fp32 mean carries `|μ − μ*| ≈ 1e-7`, so its residual sits around `1e-7` relative
and can never satisfy a `1e-9` gate: an fp32 forward makes the exact derivative
unavailable by construction.

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
- [tape.md](tape.md) — the explicit `ad-*` tape builtins and the AD instrumentation counters
- [`inc/eshkol/ad_node_registry.def`](../../../inc/eshkol/ad_node_registry.def) — the 83-row node registry itself
- [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) — opcodes, tensor backward, tape internals
