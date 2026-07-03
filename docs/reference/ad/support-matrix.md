# Automatic Differentiation — Support Matrix

This is the authoritative, machine-verified statement of what Eshkol's AD
system does and does not do in v1.3.0. It mirrors the **AD composition oracle**
([`tests/ad_oracle/`](../../../tests/ad_oracle/)), which enumerates the whole AD
surface as a matrix and checks **every cell against in-language central finite
differences** — ground truth with no hand computation.

The values below were produced by running `scripts/run_ad_oracle.sh` on this
build (JIT `-r` and AOT both):

```
ad_oracle summary: total=80 passed=46 xknown=34 failed=0 crashed=0 hung=0
ad_oracle gate: PASS
```

`total` counts each of the 40 probe files under two modes (JIT + AOT); JIT and
AOT verdicts are identical. The corpus is **214 probes / 440 checks in 40
files**. `passed` = agrees with finite differences; `xknown` = a tracked open
bug (expected); `failed`/`crashed`/`hung` = 0, so the gate is green.

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
- **`hessian` / `laplacian` on `vector` points**, all shapes.
- **Mixed reverse-over-forward** — outer vector `gradient` over inner
  `derivative` with captured parameters (v1.3, ESH-0093). See
  [`tests/ad/mixed_mode_ad_test.esk`](../../../tests/ad/mixed_mode_ad_test.esk).
- **Global captures** in every mode; **local captures** under `derivative`.
- **AD reused inside a bounded loop** (stable over 1000+ iterations).

---

## Open cells (XKNOWN)

Each references a task in [`.swarm/tasks/`](../../../.swarm/tasks/). These are
tracked, reproduced, and expected — they do **not** fail the gate. Minimal
repros live in [`tests/ad_oracle/found/`](../../../tests/ad_oracle/found/).

| Task | Cells | Symptom | Repro |
|------|-------|---------|-------|
| **ESH-0072** | `grad.*.s.caplocal` (scalar point) | Reverse-mode lambda capturing a **local scalar** → LLVM `PtrToInt source must be pointer (%eshkol_tagged_value)` at compile time, both `-r` and AOT. | — |
| **ESH-0097** | `{grad,jac,hess,div,curl,lap}.*.v*.caplocal / .capvrefout` (12 files) | Same `PtrToInt` failure for **any vector-param** reverse-mode operator capturing a local param, or a `vref` of an outer param. `derivative` and global captures are unaffected. | [`found/esh0097_local_capture_vector_ad_ptrtoint.esk`](../../../tests/ad_oracle/found/esh0097_local_capture_vector_ad_ptrtoint.esk) |
| **ESH-0095** | `hess.poly.t2/t3`, `lap.poly.t2/t3` (4 files) | `hessian` / `laplacian` **SIGSEGV** when the point is a `tensor`/`#(…)` literal (works on `(vector …)`). Second-order paths read the point through the 16-byte tagged layout; tensors are 8-byte doubles. | [`found/esh0095_hessian_tensor_point_sigsegv.esk`](../../../tests/ad_oracle/found/esh0095_hessian_tensor_point_sigsegv.esk), [`…_laplacian_…`](../../../tests/ad_oracle/found/esh0095_laplacian_tensor_point_sigsegv.esk) |
| **ESH-0096** | `nest.gofg.*.v1/v2` | `gradient` of `gradient` with a **vector** param silently returns zeros (`#(0)` for a 1-D case that should give `#(12)`). The scalar-point form is correct. | [`found/esh0096_gradient_of_gradient_vector_param_zeros.esk`](../../../tests/ad_oracle/found/esh0096_gradient_of_gradient_vector_param_zeros.esk) |
| **ESH-0078** | `nest.gofg.*.s.named/lamvar` | Second-order gradient through a **named** inner function returns `0`; the inline-lambda form is correct (`18`). | — |

> **Documentation caveat.** [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md)
> "Higher-Order Derivatives" shows `(gradient (lambda (v) (gradient … v)) (vector 2.0))`
> returning `#(12.0)`. On this build that vector-param gradient-of-gradient
> returns `#(0)` (**ESH-0096**). The scalar-point second derivative is correct.

Verified open-cell behavior on this build:

```scheme
;; ESH-0078
(define (L z) (* z (* z z)))
(gradient (lambda (y) (gradient (lambda (z) (L z)) y)) 3.0)  ;; => 18  (inline, correct)
(gradient (lambda (y) (gradient L y)) 3.0)                   ;; =>  0  (named, WRONG)

;; ESH-0096
(gradient (lambda (v) (vref (gradient (lambda (w) (* (vref w 0) (vref w 0) (vref w 0))) v) 0))
          (vector 2.0))                                       ;; => #(0)  (should be #(12))

;; ESH-0095  -> SIGSEGV
(hessian (lambda (v) (let ((x (vref v 0)) (y (vref v 1))) (+ (* x x) (* x y))))
         (tensor 1.0 2.0))

;; ESH-0072 / ESH-0097  -> compile-time PtrToInt verification failure
(define (mk a) (gradient (lambda (x) (* a x x)) 3.0))
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
