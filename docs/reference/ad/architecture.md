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

**Compiled Eshkol (JIT and AOT) — COMPLETE for `matmul`, dense elementwise
arithmetic, `tensor-sum` and `tensor-mean`.** This was, through v1.3.4, the single largest gap in the AD
architecture: no compiled program could create one of these nodes at all.
`AutodiffCodegen::recordADNodeTensor` existed and had exactly one call site,
dead behind `kDenseTensorADNodesEnabled` in `lib/backend/llvm_codegen.cpp`, and
flipping that flag SIGSEGV'd rather than yielding a slower-but-correct
gradient, for three independent reasons:

1. `recordADNodeTensor` stores NULL into `tensor_gradient`, while the reverse
   pass *selected* the tensor backward by testing that field non-null —
   constructor and consumer each waiting for the other;
2. the node it built was dropped: the function went on to return a plain
   tagged tensor, so nothing downstream could find it;
3. under AD the scalarizing path leaves AD-node *pointers* in the result
   tensor's elements, which is what `tensor-sum` and friends consume, so a
   dense node would sever the chain at the next tensor op.

All three are now closed, and the dense path is what a compiled program takes
by default (ADR-0002 Position A, `.icc/silent-wrong-ledger.yaml` SW-48):

1. **Selection.** The reverse pass recognises a tensor node by its
   `tensor_value`, not by `tensor_gradient`. `tensor_value` is set at record
   time and is documented as NULL for scalar nodes, so the test is decidable
   the moment the node exists; `tensor_gradient` is still accepted as well,
   because the qLLM bridge's C entry points seed it directly on nodes they did
   not build here. A tensor node can therefore no longer fall into the scalar
   dispatch and dereference the `input1`/`input2` a tensor node legitimately
   leaves null — which is what the SIGSEGV was. A one-element tensor node whose
   gradient arrived on the SCALAR side (`(tensor-sum …)` feeding ordinary
   arithmetic) is bridged into its tensor gradient by
   `eshkol_tensor_backward_dispatch` rather than silently dropped.
2. **The node is returned.** Under AD, `matmul` returns the `AD_NODE_MATMUL`
   node itself, tagged `CALLABLE` with the AD-node subtype the allocator
   stamps — the same shape `extractTensorAndADNode` already read on the operand
   side. Outside AD mode it returns the plain tensor it always returned, so
   nothing changes for non-differentiated code.
3. **The consumers understand it.** `tensor-sum` and `tensor-mean` reduce the
   node's dense buffer directly and record one `AD_NODE_SUM` / `AD_NODE_MEAN`
   node; `matmul` reuses a dense operand's node rather than re-packing it, so a
   chain of dense ops stays one node per op. Where an operand is *not* dense —
   the tensor of scalar AD nodes `(gradient f x)` seeds, or the result of a
   still-scalarizing tensor op — it is bridged by an `AD_NODE_TENSOR_PACK`
   node, whose backward is the identity scatter from the dense gradient onto
   the scalar nodes. A pack node performs no arithmetic, so it can change the
   *representation* of a gradient and not its value.

The cost claim is measured, not asserted.
[`tests/ad/matmul_tape_node_count_test.esk`](../../../tests/ad/matmul_tape_node_count_test.esk)
is a shrink-only ratchet on the tape size, and
[`scripts/run_dense_tensor_ad_gate.sh`](../../../scripts/run_dense_tensor_ad_gate.sh)
compiles
[`tests/ad/dense_tensor_ad_gradcheck_test.esk`](../../../tests/ad/dense_tensor_ad_gradcheck_test.esk)
under **both** lowerings — `ESHKOL_DENSE_TENSOR_AD_NODES=1` and `=0` — and
requires their parsed numeric gradients to agree within tolerance while the tape shrinks. The
scalarizing lowering is retained precisely so that it can go on serving as that
oracle; `ESHKOL_DENSE_TENSOR_AD_NODES=0` selects it, and the choice is made at
codegen time, so the two are two emitted programs rather than one program with
a runtime branch.

#### Dense-versus-scalar routing

The compiled backend's dense resident-tape route is shape-specific:

| Operation and shapes | Route in AD mode | Contract |
|---|---|---|
| `tensor-add`, `tensor-sub`, `tensor-mul`, `tensor-div` with equal ranks and dimensions | dense | one node; both numeric operands come from the densified f64 views |
| The same elementwise operations with broadcast-compatible shapes, including leading rank promotion | dense broadcast | one node; output is preflighted and allocated at the exact broadcast total |
| `matmul` with operand ranks 1 or 2, matching inner dimension, and overflow-safe products | dense | one `AD_NODE_MATMUL`; 1-D contraction follows PEP-465 result shape |
| `batch-matmul` with rank-3 operands `[batch,M,K]` and `[batch,K,N]` | dense batched | one `AD_NODE_BATCH_MATMUL`; each batch has an independent VJP |
| `transpose` of a dense rank-2 producer | dense consumer | one `AD_NODE_TRANSPOSE`; backward swaps the two matrix axes |
| whole-tensor `tensor-sum`, `tensor-mean`, and `tensor-max` of a dense producer | dense reduction | one tensor node; max uses the documented last-winner subgradient at ties |
| `matmul` rank greater than 2, incompatible shapes, or overflowed products | loud error | no `AD_NODE_MATMUL` is recorded |
| convolution, attention, norm layers, and unsupported shape-changing consumers | scalarized or loud error at the call site | not admitted to the dense node contract |

Dense operand buffers, result buffers, saved operands, packed scalar slots, and
shape arrays retained by a tensor node are allocated from
`eshkol_ad_home_arena`, the active tape's owner arena. This lifetime boundary
makes a region exit unable to reclaim a live dense node payload. The dense max
tie rule is a subgradient convention, not an assertion that the ordinary
derivative exists at a tie.

**Still scalarizing** (unchanged, and correct — the scalar decomposition has
always produced exact gradients; what it costs is tape size): `conv2d`,
`attention` and the norm layers. `(embedding …)`
codegen still emits a plain gather with no tape node, so
`(gradient (lambda (W) … (embedding idx W)))` records nothing for the lookup.
An op that has not learned the dense representation and is handed a dense
AD-node handle raises a catchable type error at its own call site rather than
returning a wrong number.

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

The **curried** route — `(define g (gradient f))` then `(jacobian g point)` —
is exact as well, by a different mechanism (SW-05). Reverse-over-reverse is not
representable on this tape, because `ad_node_t::value` is a single double, so
when the runtime-closure gradient is handed a point whose components are the
enclosing pass's tape nodes it evaluates the inner gradient **forward** instead:
for output component `i`, component `k` of the argument becomes the 8-jet
`x_k = value_k + [k == i]·e1 + [k is the active seed]·ep`. The returned jet's
`e1` coefficient is `grad_i` and the surviving `e1·ep` coefficient is
`d(grad_i)/d(seed) = H[i][seed]` — the same two fields `popAndExtractForward`
reads at level 0 — and `eshkol_ad_mixed_record` writes that exact local
linearization back onto the outer tape. This is the same forward-over-reverse
composition `(hessian f point)` performs, and it agrees with it entry-for-entry:

```scheme
(define (f v) (* (vref v 0) (vref v 0) (vref v 1)))
(define g (gradient f))
(jacobian g (vector 2.0 3.0))   ;; => #(#(6 4) #(4 0))
(hessian  f (vector 2.0 3.0))   ;; => #(#(6 4) #(4 0))
```

The one shape that still refuses is a point *computed* from the enclosing pass's
variables — `(jacobian (lambda (v) (g (vector (* 2.0 (vref v 0))))) point)` —
where no component IS the published active seed, so no edge can be threaded
back. That **raises** (a loud refusal, not a silent zero): the `(vector …)`
spelling is caught by the point coercion, `evaluation point is not a number`,
before the tensor arm is reached; a tensor of non-seed nodes is caught by the
pre-scan itself, `unsupported nested differentiation`. See KNOWN_ISSUES.md.

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
