# Automatic Differentiation — Support Matrix

This is the authoritative, machine-verified statement of what Eshkol's AD
system does and does not do in v1.3.5-evolve. It mirrors the **AD composition
oracle**
([`tests/ad_oracle/`](../../../tests/ad_oracle/)), which enumerates the whole AD
surface as a matrix and checks **every cell against in-language central finite
differences** — ground truth with no hand computation.

The values below were produced by running `scripts/run_ad_oracle.sh` on this
build (JIT `-r` and AOT both):

```
ad_oracle summary: total=60 passed=60 xknown=0 failed=0 crashed=0 hung=0
ad_oracle gate: PASS
```

`total` counts each of the 30 probe files under two modes (JIT + AOT); JIT and
AOT verdicts are identical. The corpus is **235 probes / 490 checks in 30
files**. `passed` = agrees with finite differences; `xknown` = a tracked open
bug (expected); `failed`/`crashed`/`hung` = 0, so the gate is green — and with
`xknown = 0`, every enumerated cell is now a genuine pass.

---

## The matrix axes

The oracle sweeps the Cartesian product of:

| axis | values |
|------|--------|
| operator | `derivative` `gradient` `jacobian` `hessian` `divergence` `curl` `laplacian` |
| point | scalar, 2-vector, 3-vector, 2/3-tensor (`(tensor …)`), multi-param + `(list …)` |
| shape | polynomial, product-of-linears, with-subtraction, rational `1/(1+x²)`, exp/sin composite, let-bound reuse, named-let accumulation |
| binding | inline lambda, named `define`, lambda-in-variable |
| capture | none, global scalar, local param scalar, `vref` of outer param |
| nesting | none, derivative-of-derivative, gradient-of-derivative (scalar+vector), gradient-of-gradient (scalar+vector), AD-in-loop |

Tolerance: `|ad - fd| ≤ atol + rtol·|fd|`, `rtol = 1e-4`. First-order stencils
`h = 1e-5` (atol 1e-6); second-order stencils `h = 1e-4` (atol 1e-5).

---

## What passes (PASS cells)

- **All first-order operators on all point types**: `gradient`, `jacobian`,
  `divergence`, `curl` accept `vector`, `#(…)`/`tensor`, scalar, and
  `(list …)` points across every shape.
- **`derivative`** including vector-valued output and 2-level nesting
  (derivative-of-derivative, exact via the two jet slots).
- **`hessian` / `laplacian` on `vector`, `#(…)`/`tensor` and scalar points**,
  all shapes.
- **Mixed reverse-over-forward** — outer vector `gradient` over inner
  `derivative` with captured parameters (v1.3, ESH-0093). See
  [`tests/ad/mixed_mode_ad_test.esk`](../../../tests/ad/mixed_mode_ad_test.esk).
- **Gradient of gradient** at a scalar *and* a vector param, through an inline
  lambda and through a named function alike.
- **Global captures** in every mode; **local captures** in every mode, under
  `derivative` and under every reverse-mode operator.
- **AD reused inside a bounded loop** (stable over 1000+ iterations).
- **Exact-element vectors**: integer/rational elements are tag-coerced by value
  before vector gradients; they do not become heap-address doubles. The Keller
  map regression checks the exact and inexact representations.

---

## Execution model — how a tensor gradient is recorded

The matrix above is about *answers*. This is about what the tape costs to get
them, which ADR-0002 treats as a separate axis because a correct gradient
computed through 2·M·N·K scalar tape nodes and the same gradient computed
through one dense node are indistinguishable to the oracle and very different
to a training loop.

| Operation | Under AD, records | Status |
|-----------|-------------------|--------|
| `matmul` / `tensor-matmul` | ONE `AD_NODE_MATMUL`, backward by `eshkol_backward_matmul` | COMPLETE |
| `tensor-sum` (whole tensor, dense operand) | ONE `AD_NODE_SUM` | COMPLETE |
| `tensor-mean` (whole tensor, dense operand) | ONE `AD_NODE_MEAN` | COMPLETE |
| `tensor-max` (whole tensor, dense operand) | ONE `AD_NODE_TENSOR_MAX_DENSE`; last-winner subgradient at ties | COMPLETE |
| dense/scalar boundary | ONE `AD_NODE_TENSOR_PACK` per operand that arrives scalarized; identity scatter backward | COMPLETE |
| elementwise `tensor-add/sub/mul/div` | ONE dense node (`AD_NODE_TENSOR_*_DENSE`) | COMPLETE |
| broadcast elementwise variants | ONE dense node (`AD_NODE_TENSOR_BROADCAST_*_DENSE`), summed VJP over broadcast axes | COMPLETE |
| `batch-matmul` with rank-3 `[batch,M,K]` and `[batch,K,N]` operands | ONE `AD_NODE_BATCH_MATMUL`, independent batched VJP | COMPLETE |
| `transpose` of a dense rank-2 producer | ONE `AD_NODE_TRANSPOSE` | COMPLETE |
| `conv2d` | one scalar node per scalar operation | Scalarizing — dense kernels exist in `lib/backend/tensor_backward.cpp`, producer not yet routed |
| `attention`, `layer-norm` | one scalar node per scalar operation for reverse mode; dtype-DUAL tensor for first-order `derivative` inputs | Reverse mode remains scalarizing; first-order dual-vector support is complete for native (rank-2/rank-3 attention) and VM (rank-2 attention) (`tests/ad/issue_551_tensor_transformer_dual_test.esk`, `tests/vm_parity/corpus/551_tensor_transformer_dual.esk`) |
| `embedding` | nothing (plain gather) | Build item, see [architecture.md](architecture.md) |
| VM (`eshkol-vm-standalone-test`) | scalar `AdNode` for explicit tape primitives; `VmDual` plus parallel dual tensor carrier for first-order transformer derivatives | First-order `layer-norm`/`scaled-dot-attention` dual support; no VM reverse tensor-node tape |

Both lowerings are kept and are differentially gated against each other:

```
scripts/run_dense_tensor_ad_gate.sh      # both lowerings, numeric gradients must agree
ESHKOL_DENSE_TENSOR_AD_NODES=0           # select the scalarizing lowering
```

The variable is read at **codegen** time, so it selects which program is
emitted rather than which branch a program takes. The gate compiles
`tests/ad/dense_tensor_ad_gradcheck_test.esk` both ways and requires the parsed
numeric gradients to agree within tolerance, across square and non-square
shapes, either operand, the PEP-465 1-D contraction, `tensor-sum` and
`tensor-mean`, nested elementwise and dense→dense chains, transposes, batched
matmul, and max subgradients — while the 6×6 tape has exactly four nodes.

---

## Nesting ceiling (SW-154)

Differentiation passes nest, and the shapes below the ceiling are exact. The
ceiling itself is a property of the v1.3.5 forward **carrier**, not of the
mathematics: a pass rides on one value series plus one first-order companion
series, which is exactly enough for one enclosing level at first order.

| Shape | v1.3.5 |
|---|---|
| Any depth of **first-order** passes (`derivative` inside `derivative` inside `derivative`, …) | Supported, exact |
| One pass of order ≥ 2 with **one** enclosing first-order pass, in either position | Supported, exact |
| Two passes both of order ≥ 2 | **Raises** `unsupported nested differentiation (both passes order >= 2)` |
| **Two enclosing levels** over a pass of order ≥ 2 | **Not supported (SW-154).** On the exact tier it raises; on the inexact tier it currently answers `0` |

Verified on this build:

```scheme
;; depth-3, all first order: d/dx d/dy d/dz (x*y*z) = 1
(display (derivative (lambda (x) (derivative (lambda (y) (derivative (lambda (z) (* x y z)) 1.0)) 1.0)) 1.0)) (newline)
;; depth-3, all first order: d/dx d/dy d/dz (x^2 y^2 z^2) at (2,3,4) = 2x*2y*2z
(display (derivative (lambda (x) (derivative (lambda (y) (derivative (lambda (z) (* x x y y z z)) 4.0)) 3.0)) 2.0)) (newline)
;; one order-2 pass under one first-order pass: d/da [d2/db2 (a^2 b^3)] at b=1/2, a=1/3
(display (derivative (lambda (a) (derivative-n (lambda (b) (* a a b b b)) 1/2 2)) 1/3)) (newline)
;; and the mirror position: d2/dx2 [d/dy (x^3 y)] at 1.0
(display (derivative-n (lambda (x) (derivative (lambda (y) (* x x x y)) 1.0)) 1.0 2)) (newline)
```
```
1
192
2
6
```

The refusal, and the shape that is **not** supported:

```scheme
(define (h r) (* r r r r))
(display (taylor (lambda (t) (* t (derivative-n h (+ 1 t) 2))) 0 2)) (newline)
```
```
     ERROR: unsupported nested differentiation: an order-2 `derivative-n`/`taylor` pass inside another differentiation of order 2 or higher. Eshkol's forward carriers compose when at least one of the two passes is first order; rewrite the inner or outer pass as a first-order `derivative`, or compute the higher-order term with a single `(derivative-n f x k)`.
Unhandled exception: unsupported nested differentiation (both passes order >= 2)
```

```scheme
;; TWO enclosing levels over an order-2 pass. The analytic answer is 2.
(display (derivative (lambda (a)
           (derivative (lambda (b)
             (derivative-n (lambda (c) (* a b c c)) 1.0 2)) 1.0)) 1.0)) (newline)
```
```
0
```

Do not rely on that shape. It is pinned, unregistered, in
[`tests/ad/nested_towers_matrix_test.esk`](../../../tests/ad/nested_towers_matrix_test.esk),
which cannot pass until the carrier rewrite (ESH-0413) lands; that rewrite
replaces the carrier the v1.3.5 exact-coefficient tier is built on, so the two
cannot both be in force, and it is **v1.4 work**. The same program written with
an exact seed raises instead of answering `0`:

```scheme
(display (derivative (lambda (a)
           (derivative (lambda (b)
             (derivative-n (lambda (c) (* a b c c)) 1/2 2)) 1/3)) 1/5)) (newline)
```
```
     ERROR: Taylor hyperdual propagation requires exact +, -, *, or / operands
Unhandled exception: unsupported inexact/non-rational Taylor hyperdual operation
```

The nesting shapes that **do** hold are gated by
`nested_operator_matrix_*` (the captured-variable matrix, JIT + AOT) and
`ad_carrier_nesting_*` (the point matrix).

---

## Open cells (XKNOWN)

**None on this build.** Every cell the oracle enumerates agrees with finite
differences in both the JIT and AOT lanes: `xknown=0`.

The five cells that were open through v1.3.0–v1.3.3 are all closed. Their
minimal repros are kept in
[`tests/ad_oracle/found/`](../../../tests/ad_oracle/found/) as the acceptance
tests of the fixes, and they now print the correct answers:

| Task | Cells | Was | Now |
|------|-------|-----|-----|
| **ESH-0072** | `grad.*.s.caplocal` (scalar point) | Reverse-mode lambda capturing a **local scalar** failed LLVM verification (`PtrToInt source must be pointer`) at compile time. | Compiles and differentiates; `(define (mk a) (gradient (lambda (x) (* a x x)) 3.0))`, `(mk 2.0)` → `12`. |
| **ESH-0097** | `{grad,jac,hess,div,curl,lap}.*.v*.caplocal / .capvrefout` | Same `PtrToInt` failure for any **vector-param** reverse-mode operator capturing a local param or a `vref` of an outer param. | Compiles; [`found/esh0097_…`](../../../tests/ad_oracle/found/esh0097_local_capture_vector_ad_ptrtoint.esk) prints its expected `#(4.42 0)`. |
| **ESH-0095** | `hess.poly.t2/t3`, `lap.poly.t2/t3` | `hessian`/`laplacian` **SIGSEGV** at a `tensor`/`#(…)` point. | Points are classified by runtime value, not AST node kind (#343); every point form gives the same result. |
| **ESH-0096** | `nest.gofg.*.v1/v2` | `gradient` of `gradient` at a **vector** param silently returned zeros. | Returns the true second derivative — `#(12)` for the 1-D case, `#(8 6)` for the 2-D one. |
| **ESH-0078** | `nest.gofg.*.s.named/lamvar` | Second-order gradient through a **named** inner function returned `0`. | Returns `18`, matching the inline-lambda form. |

Verified on this build:

```scheme
;; ESH-0078 — inline and named forms now agree
(define (L z) (* z (* z z)))
(gradient (lambda (y) (gradient (lambda (z) (L z)) y)) 3.0)  ;; => 18
(gradient (lambda (y) (gradient L y)) 3.0)                   ;; => 18

;; ESH-0096 — vector-param gradient-of-gradient
(gradient (lambda (v) (vref (gradient (lambda (w) (* (vref w 0) (vref w 0) (vref w 0))) v) 0))
          (vector 2.0))                                       ;; => #(12)

;; ESH-0095 — second-order operator at a tensor point
(hessian (lambda (v) (let ((x (vref v 0)) (y (vref v 1))) (+ (* x x) (* x y))))
         (tensor 1.0 2.0))                                    ;; => #((2 1) (1 0))

;; ESH-0072 / ESH-0097 — local capture under a reverse-mode operator
(define (mk a) (gradient (lambda (x) (* a x x)) 3.0))
(mk 2.0)                                                      ;; => 12
```

---

## Running the oracle

```
scripts/run_ad_oracle.sh            # full sweep, JIT + AOT
scripts/run_ad_oracle.sh --quick    # CI subset (first file of each section/task)
scripts/run_ad_oracle.sh --no-aot   # JIT lane only
scripts/run_ad_oracle.sh --regen    # regenerate the (deterministic) corpus first
```

Point it at a build dir with `BUILD_DIR=…`. Per-file verdicts: `PASS` / `FAIL`
(an *untracked* cell diverged from finite differences) / `XKNOWN` (tracked open
bug) / `CRASH` / `HANG`. The gate is green iff there are no FAIL/CRASH/HANG.
Verdicts stream to `scripts/icc_traces/ad_oracle.jsonl` as `kind:"ad_oracle"`
events (consumed by `.icc/completion-oracles.yaml::ad-oracle`).

When a task is fixed, its probes flip `XKNOWN → PASS` automatically — no oracle
edit needed. The generator ([`gen_ad_oracle.py`](../../../tests/ad_oracle/gen_ad_oracle.py))
is deterministic; regenerating reproduces the corpus byte-for-byte.

---

## Native AD bridge status

The Scheme composition oracle covers the Scheme operators above. The native
bridge has one additional registered AD primitive:

| Node / entry point | Status | Coverage | VM status |
|---|---|---|---|
| `AD_NODE_SQUARED_DISTANCE` / `ad_squared_distance` and `ad_product_squared_distance` | **COMPLETE** | `squared_distance_gradcheck`: 48 exact, identity, audit-counterexample, boundary, golden-vector, and finite-difference checks through the real producer and reverse sweep | Native-only, justified in `tests/vm_parity/PARITY.tsv`: the VM has no tensor-valued `ad_node_t` carrier or matching opcode |

The node is not a Scheme builtin, so it does not add a row to the Scheme
operator axes. Its registered row in `inc/eshkol/ad_node_registry.def` is the
source of truth for its tensor payload and bridge backward function.

## See also

- [operators.md](operators.md) — per-operator API, capture rules, nesting
- [architecture.md](architecture.md) — forward jet, reverse tape, mixed mode, and the
  [83-row AD node registry](architecture.md#the-ad-node-registry): what `UNREGISTERED`
  means, why the dispatcher has no `default:`, and the
  [exact geometric backwards](architecture.md#exact-geometric-backwards) for node
  types 33-40, which this oracle's scalar corpus does not reach
- [tape.md](tape.md) — the explicit `ad-*` tape builtins and the instrumentation
  counters, including `(ad-finite-difference-evals)` and its negative control
- [`tests/ad_oracle/README.md`](../../../tests/ad_oracle/README.md) — oracle design
