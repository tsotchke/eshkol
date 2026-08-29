# Automatic Differentiation — Support Matrix

This is the authoritative, machine-verified statement of what Eshkol's AD
system does and does not do in v1.3.4. It mirrors the **AD composition oracle**
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
| `conv2d`, `attention`, norm layers | one scalar node per scalar operation | Scalarizing — dense kernels exist in `lib/backend/tensor_backward.cpp`, producers not yet routed |
| `embedding` | nothing (plain gather) | Build item, see [architecture.md](architecture.md) |
| VM (`eshkol-vm-standalone-test`) | scalar `AdNode` only; no `ad_node_t`, no tensor node types | Not implemented — see `tests/vm_parity/PARITY.tsv` |

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

## See also

- [operators.md](operators.md) — per-operator API, capture rules, nesting
- [architecture.md](architecture.md) — forward jet, reverse tape, mixed mode
- [`tests/ad_oracle/README.md`](../../../tests/ad_oracle/README.md) — oracle design
