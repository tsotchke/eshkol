# Automatic Differentiation — Operator Reference

Every operator, signature, accepted point type, binding form, and capture rule
below is verified by running it on the v1.3.0 compiler. Outputs are pasted
exactly as printed by `eshkol-run` (JIT `-r` and AOT agree unless noted). Open
cells are marked with their ledger id — see
[support-matrix.md](support-matrix.md).

For the underlying machinery (forward 4-jet, reverse tape, perturbation
levels) see [architecture.md](architecture.md). For the numeric-boundary and
tensor-backward internals see
[../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md).

---

## Operator summary

| Operator | Field type | Signature | Result |
|----------|-----------|-----------|--------|
| `derivative` | ℝ → ℝ | `(derivative f x)` | scalar |
| `gradient` | ℝⁿ → ℝ | `(gradient f point)` | vector `∇f` |
| `jacobian` | ℝⁿ → ℝᵐ | `(jacobian f point)` | matrix `J` (vector of rows) |
| `hessian` | ℝⁿ → ℝ | `(hessian f point)` | matrix `H` (vector of rows) |
| `laplacian` | ℝⁿ → ℝ | `(laplacian f point)` | scalar `∇²f` = tr(H) |
| `directional-derivative` | ℝⁿ → ℝ | `(directional-derivative f point dir)` | scalar `∇f · dir` |
| `divergence` | ℝⁿ → ℝⁿ | `(divergence f point)` | scalar `∇·F` |
| `curl` | ℝ³ → ℝ³ | `(curl f point)` | vector `∇×F` |
| `diff` | AST → AST | `(diff f 'x)` | symbolic derivative (compile time) |

`f` is a one-argument function (an inline `lambda`, a named `define`, or a
variable bound to a lambda). `vref` is the tensor/vector element accessor and
is an alias for `vector-ref` in AD bodies.

---

## `derivative` — forward-mode scalar derivative

```
(derivative f x)   ;; f : ℝ → ℝ, x a real; returns f'(x)
```

Forward mode (4-component Taylor jet). Best for one input / many outputs.

```scheme
(derivative (lambda (x) (* x x)) 3.0)       ;; => 6
(derivative (lambda (x) (sin (* x x))) 1.0) ;; => 1.0806
```

`display` prints doubles at reduced precision (≈5 significant figures), so the
second result shows as `1.0806` (the full value is 2·cos(1) ≈ 1.08060461).

> **Not supported (returns `0`).** A **vector- or tensor-valued** function
> (ℝ → ℝⁿ) does *not* differentiate componentwise under `derivative` on this
> build — it silently returns `0`:
>
> ```scheme
> (derivative (lambda (x) (vector (* x x) (* x x x))) 2.0)  ;; => 0  (NOT #(4 12))
> ```
>
> Take per-component derivatives with separate scalar `derivative` calls, or
> use `jacobian` for a vector-valued map of a vector point.

Second derivative by nesting two `derivative` calls (two perturbation slots,
exact — see architecture):

```scheme
(derivative (lambda (y) (derivative (lambda (z) (* z z z)) y)) 3.0)  ;; => 18
```

---

## `gradient` — reverse-mode gradient

```
(gradient f point)   ;; f : ℝⁿ → ℝ ; point a vector or tensor; returns ∇f
```

`f` takes the whole point vector and unpacks components with `vref`. Reverse
mode; best for many inputs / one output (loss functions, training).

```scheme
(gradient (lambda (v)
            (let ((x (vref v 0)) (y (vref v 1)))
              (+ (* x x) (* y y))))
          (vector 3.0 4.0))
;; => #(6 8)
```

A scalar point is also accepted (treated as a 1-D gradient, returning a
scalar):

```scheme
(gradient (lambda (x) (* x x)) 3.0)   ;; => 6
```

---

## `jacobian` — reverse-mode Jacobian

```
(jacobian f point)   ;; f : ℝⁿ → ℝᵐ ; returns the m×n matrix J[i][j] = ∂fᵢ/∂xⱼ
```

```scheme
(jacobian (lambda (v)
            (let ((x (vref v 0)) (y (vref v 1)))
              (vector (* x x) (* y y))))
          (vector 3.0 4.0))
;; => #((6 0) (0 8))
```

The result is a vector of row vectors. Note the display form: the outer level
prints with `#(…)` and inner rows print as bare `(…)` — `#((6 0) (0 8))` is a
2×2 matrix.

---

## `hessian` — second-order (matrix of second partials)

```
(hessian f point)   ;; f : ℝⁿ → ℝ ; returns H[i][j] = ∂²f/(∂xᵢ∂xⱼ)
```

```scheme
;; f(x) = x⁴, f''(2) = 12·2² = 48
(hessian (lambda (v) (let ((x (vref v 0))) (* x (* x (* x x)))))
         (vector 2.0))
;; => #((48))

;; f(x,y) = x² + xy
(hessian (lambda (v) (let ((x (vref v 0)) (y (vref v 1))) (+ (* x x) (* x y))))
         (vector 1.0 2.0))
;; => #((2 1) (1 0))
```

> Open cell **ESH-0095**: `hessian` (and `laplacian`) **SIGSEGV** when the
> point is a `tensor`/`#(…)` literal instead of a `vector`. Use `(vector …)`
> points for second-order operators. See support-matrix.md.

---

## `laplacian` — trace of the Hessian

```
(laplacian f point)   ;; f : ℝⁿ → ℝ ; returns ∇²f = Σᵢ ∂²f/∂xᵢ²
```

```scheme
(laplacian (lambda (v) (let ((x (vref v 0)) (y (vref v 1))) (+ (* x x) (* y y))))
           (vector 3.0 4.0))
;; => 4
```

> Same open cell **ESH-0095**: SIGSEGV on `tensor`/`#(…)` points; use vectors.

---

## `directional-derivative` — gradient projected onto a direction

```
(directional-derivative f point dir)   ;; returns ∇f(point) · dir
```

```scheme
(directional-derivative
   (lambda (v) (let ((x (vref v 0)) (y (vref v 1))) (+ (* x x) (* y y))))
   (vector 3.0 4.0) (vector 1.0 0.0))
;; => 6
```

---

## `divergence` — scalar divergence of a vector field

```
(divergence f point)   ;; f : ℝⁿ → ℝⁿ ; returns Σᵢ ∂Fᵢ/∂xᵢ
```

```scheme
(divergence (lambda (v)
              (vector (* (vref v 0) (vref v 0))
                      (* (vref v 1) (vref v 1))))
            (vector 3.0 4.0))
;; => 14           ; 2x + 2y = 6 + 8
```

---

## `curl` — curl of a 3-D vector field

```
(curl f point)   ;; f : ℝ³ → ℝ³ ; returns ∇×F
```

```scheme
(curl (lambda (v)
        (vector (* (vref v 1) (vref v 2))    ; y·z
                (* (vref v 0) (vref v 2))     ; x·z
                (* (vref v 0) (vref v 1))))   ; x·y
      (vector 1.0 2.0 3.0))
;; => #(0 0 0)
```

---

## `diff` — symbolic differentiation (compile time)

`diff` rewrites the function AST at compile time using 12 rules (constant,
variable, sum, difference, product, quotient, power, sin, cos, exp, log,
chain). It returns a function, not a value. See
[../../breakdown/AUTODIFF.md#symbolic-differentiation](../../breakdown/AUTODIFF.md#symbolic-differentiation)
for the full rule table.

```scheme
(diff (lambda (x) (* x x)) 'x)   ;; AST equivalent to (lambda (x) (* 2 x))
```

---

## Accepted point types

| Point form | Layout | `gradient`/`jacobian`/`divergence`/`curl` | `hessian`/`laplacian` |
|------------|--------|-------------------------------------------|-----------------------|
| `(vector 1.0 2.0)` | 16-byte tagged values | ✅ | ✅ |
| `#(1.0 2.0)` literal (tensor) | 8-byte doubles | ✅ | ❌ SIGSEGV (**ESH-0095**) |
| `(tensor 1.0 2.0)` | 8-byte doubles | ✅ | ❌ SIGSEGV (**ESH-0095**) |
| scalar `3.0` | double | ✅ (1-D) | — |
| multi-param via `(list …)` | cons list | ✅ (first-order ops) | — |

First-order operators accept both `vector` and `tensor` points (verified
against the [AD oracle](support-matrix.md) matrix). Second-order operators are
only safe on `vector` points until ESH-0095 lands. See
[../tensors/creation.md](../tensors/creation.md) for the vector-vs-tensor
distinction.

---

## Binding forms

The function argument may be supplied three ways. All three work for
first-order operators:

```scheme
;; 1. inline lambda
(gradient (lambda (v) (* (vref v 0) (vref v 0))) (vector 3.0))   ;; => #(6)

;; 2. named define
(define (sq v) (* (vref v 0) (vref v 0)))
(gradient sq (vector 3.0))                                        ;; => #(6)

;; 3. lambda bound to a variable
(define sqv (lambda (v) (* (vref v 0) (vref v 0))))
(gradient sqv (vector 3.0))                                       ;; => #(6)
```

> **Nesting caveat (ESH-0078).** For a *nested* second-order gradient, the
> inner `gradient`/`derivative` must currently receive an **inline lambda**.
> A named function or lambda-variable as the inner differentiand silently
> returns `0`:
>
> ```scheme
> (define (L z) (* z (* z z)))
> (gradient (lambda (y) (gradient (lambda (z) (L z)) y)) 3.0)  ;; => 18  ✅ inline
> (gradient (lambda (y) (gradient L y)) 3.0)                   ;; =>  0  ❌ named (ESH-0078)
> ```

---

## Capture rules

What the differentiated lambda may close over depends on the mode:

| Captured value | `derivative` (forward) | `gradient`/`jacobian`/`hessian`/`divergence`/`curl`/`laplacian` (reverse) |
|----------------|------------------------|---------------------------------------------------------------------------|
| top-level (global) scalar | ✅ | ✅ |
| top-level (global) vector, via `vref` | ✅ | ✅ |
| **local** / parameter scalar | ✅ | ❌ LLVM `PtrToInt` verification failure (**ESH-0072** scalar point, **ESH-0097** vector point) |
| `vref` of an outer **local** vector param | ✅ | ❌ same failure (**ESH-0097** `capvrefout`) |

Verified examples:

```scheme
;; GLOBAL capture — works in both modes
(define a 8.0) (define t 5.0)
(define (loss x) (let ((d (- x t))) (* a (* d d))))
(derivative loss 7.0)                                    ;; => 32   ✅

(define g 1.7)
(gradient (lambda (v) (* g (vref v 0) (vref v 0))) (vector 1.3 -0.7))
;; => #(4.42 0)   ✅

;; LOCAL capture under `derivative` — works (forward mode)
(define (step a t w0) (derivative (lambda (x) (loss2 a t x)) w0))
;; (step 8.0 5.0 7.0) => 32   ✅

;; LOCAL capture under `gradient` — FAILS at compile time (ESH-0072 / ESH-0097)
(define (mk a) (gradient (lambda (x) (* a x x)) 3.0))
;; ERROR: LLVM module verification failed: PtrToInt source must be pointer
;;        %N = ptrtoint %eshkol_tagged_value %a to i64
```

**Workaround until ESH-0072/ESH-0097 land:** lift captured scalars to
top-level `define`s (or pass them *inside* the point vector and `vref` them),
so the reverse-mode lambda only closes over globals.

---

## Composition / nesting

| Composition | Status | Note |
|-------------|--------|------|
| `derivative` of `derivative` (scalar 2nd order) | ✅ | exact, 2 perturbation slots |
| `gradient` of scalar `derivative`, scalar point | ✅ | forward-fast-path |
| `gradient` (vector point) over inner `derivative` — **mixed reverse-over-forward** | ✅ | fixed in v1.3 (#113, ESH-0093); see [tests/ad/mixed_mode_ad_test.esk](../../../tests/ad/mixed_mode_ad_test.esk), 15/15 |
| `gradient` of `gradient`, **scalar** point | ✅ | e.g. `L''` returns correct value |
| `gradient` of `gradient`, **vector** point | ❌ returns zeros (**ESH-0096**) |
| `gradient` of a **named** inner function | ❌ returns 0 (**ESH-0078**) |
| AD inside a bounded loop (reuse) | ✅ | stable over 1000+ iterations |

Mixed reverse-over-forward (an outer vector `gradient` over an inner
`derivative` that depends on captured tape parameters) is the headline v1.3 AD
fix. Verified:

```scheme
;; f(x;p0)=p0·x², ∂/∂p0 [ d/dx f @2 ] = 4
(gradient (lambda (p) (derivative (lambda (x) (* (vref p 0) (* x x))) 2.0))
          (vector 3.0))
;; => #(4)
```

---

## See also

- [architecture.md](architecture.md) — forward 4-jet, reverse tape, `__ad_pert_level`, mixed-mode recording
- [support-matrix.md](support-matrix.md) — full oracle matrix, open cells, running the oracle
- [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) — modes, node opcodes, tensor backward, numeric boundary
- [../tensors/INDEX.md](../tensors/INDEX.md) — tensor operations that AD flows through
