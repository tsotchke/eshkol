# Automatic Differentiation in Eshkol

*A user guide to the v1.3.0-evolve Taylor-tower AD system.*

Eshkol differentiates programs, not just formulas. `derivative`, `gradient`,
`jacobian` and friends are **compiler primitives** — you write ordinary Scheme
and the compiler emits the exact derivative alongside the value. There is no
tracer, no graph-building DSL, and no separate "differentiable" dialect: the
same `(* x x)` you evaluate is the one that gets differentiated.

What makes Eshkol's AD unusual:

- **Arbitrary order.** Not just first and second derivatives — the *n*-th
  derivative for any *n*, computed by truncated-Taylor recurrences that cost
  O(*n*²), never the 2ⁿ blow-up of nested dual numbers.
- **Exact when it can be.** Differentiate a polynomial or rational function at
  an exact point and the derivative comes back as an exact `bignum` or
  `rational` — zero floating-point error. No double-only framework (JAX,
  PyTorch, Zygote) offers this.
- **Validated when you need a guarantee.** Taylor models return a Taylor
  polynomial *plus a rigorous interval remainder*, so you get a proven
  enclosure of a function over a whole domain, not just a point estimate.
- **A property of the language.** Derivatives flow through `if`/`cond`, loops,
  recursion, closures, and `map`/`fold`; through tensors (matmul/conv2d/
  activations); through the reverse tape; and they nest without perturbation
  confusion.

This guide is example-driven. **Every snippet below was run with
`./build/eshkol-run -r <file>` and the output shown is the real output** — no
invented numbers. Snippets that `(require core.ad.*)` a library module were run
with a library search path (`-L build`) and with the JIT object cache disabled
(`ESHKOL_JIT_CACHE=0`); this is noted where it applies. The design and proofs
behind all of this live in
[`docs/design/AD_TAYLOR_TOWER.md`](../design/AD_TAYLOR_TOWER.md).

---

## Table of contents

1. [The classic operators](#1-the-classic-operators)
2. [Arbitrary order — the tower](#2-arbitrary-order--the-tower)
3. [Exact coefficients (bignum & rational)](#3-exact-coefficients-bignum--rational)
4. [Multivariate mixed partials (GUW)](#4-multivariate-mixed-partials-guw)
5. [Tensor AD — towers of tensors](#5-tensor-ad--towers-of-tensors)
6. [Validated AD — Taylor models](#6-validated-ad--taylor-models)
7. [Sparse high-order — sparse Hessians](#7-sparse-high-order--sparse-hessians)
8. [Reverse mode & checkpointing](#8-reverse-mode--checkpointing)
9. [Differentiable control flow](#9-differentiable-control-flow)
10. [Tower numerics — ODEs, roots, inversion](#10-tower-numerics--odes-roots-inversion)
11. [Perturbation safety & how it works](#11-perturbation-safety--how-it-works)
12. [API quick reference](#12-api-quick-reference)

---

## 1. The classic operators

These are the first-class AD builtins — always available, no `require` needed.
They cover scalar derivatives, gradients, and the standard vector-calculus
operators.

### `derivative` — ℝ → ℝ

```scheme
(display (derivative (lambda (x) (* x x)) 3.0)) (newline)

;; Called with just the function, it returns the derivative *function*:
(define df (derivative (lambda (x) (sin x))))
(display (df 0.0)) (newline)
```

Output:

```
6
1
```

`(derivative f x)` gives `f'(x)` directly; `(derivative f)` gives a new function
you can apply at many points.

### `gradient` — ℝⁿ → ℝ

```scheme
(define (f v) (+ (* (vector-ref v 0) (vector-ref v 0))
                 (* (vector-ref v 1) (vector-ref v 1))))
(display (gradient f #(2.0 3.0))) (newline)
```

Output:

```
#(4 6)
```

∇(x²+y²) = (2x, 2y) = (4, 6). `gradient` uses reverse mode, so the whole
gradient costs one backward pass regardless of the input dimension — the right
tool for scalar loss functions of many parameters.

### `jacobian` — ℝⁿ → ℝᵐ

```scheme
(define (f v) (vector (* 2 (vector-ref v 0))
                      (+ (vector-ref v 0) (vector-ref v 1))))
(display (jacobian f #(3.0 4.0))) (newline)
```

Output:

```
#((2 0) (1 1))
```

Row *i* is ∇fᵢ. Here f = (2x, x+y) so J = [[2,0],[1,1]].

### `hessian` — second derivatives, ℝⁿ → ℝ

```scheme
(define (f v) (+ (* (vector-ref v 0) (vector-ref v 0))
                 (* (vector-ref v 1) (vector-ref v 1))))
(display (hessian f #(1.0 1.0))) (newline)
```

Output:

```
#((2 0) (0 2))
```

### `divergence` and `curl`

```scheme
(define (F v) v)                                  ; identity field
(display (divergence F #(1.0 2.0 3.0))) (newline) ; ∇·F = 3

(define (G v) (vector (vector-ref v 1) (- (vector-ref v 0)) 0.0))
(display (curl G #(1.0 2.0 0.0))) (newline)       ; ∇×G
```

Output:

```
3
#(0 0 -2)
```

### `laplacian` and `directional-derivative`

```scheme
(define (f v) (+ (* (vector-ref v 0) (vector-ref v 0))
                 (* (vector-ref v 1) (vector-ref v 1))))
(display (laplacian f #(1.0 2.0))) (newline)             ; ∇²f = 2+2 = 4

(define (g v) (* (vector-ref v 0) (vector-ref v 1)))
(display (directional-derivative g #(2.0 3.0) #(1.0 0.0))) (newline) ; ∂/∂x = y = 3
```

Output:

```
4
3
```

**When to use:** these are the everyday operators — optimization gradients,
Jacobians of vector maps, Hessians for Newton steps, and the vector-calculus
trio (`divergence`/`curl`/`laplacian`) for physics/PDE work.

---

## 2. Arbitrary order — the tower

The headline capability. `(derivative-n f x k)` returns the exact *k*-th
derivative `f⁽ᵏ⁾(x)` for **any** `k`, and `(taylor f x k)` returns the whole
truncated Taylor series — the coefficients `c[0..k]` of
`f(x + t) = Σ cₖ·tᵏ`. The two are linked by `f⁽ⁿ⁾(x) = n!·cₙ`.

Under the hood this is Taylor-mode AD: a coefficient array propagated by closed
recurrences (Cauchy convolution for `*`, coupled recurrences for `sin`/`cos`,
etc.). Cost is O(k²) in the order — never the 2ᵏ explosion of stacked dual
numbers.

### `derivative-n` — the *n*-th derivative

```scheme
(define (f x) (* x x x x x))          ; x^5

(display (derivative-n f 2.0 3)) (newline)   ; f'''(2)
(display (derivative-n f 2.0 5)) (newline)   ; f'''''(2)
(display (derivative-n (lambda (x) (sin x)) 0.0 8)) (newline)   ; d^8/dx^8 sin, at 0
(display (derivative-n (lambda (x) (exp x)) 0.5 8)) (newline)   ; d^8/dx^8 exp, at 0.5
```

Output:

```
240
120
0
1.64872
```

For x⁵: f‴ = 60x² = 240 at x=2; f⁽⁵⁾ = 120 (constant); f⁽⁶⁾ and up are 0.
The 8th derivative of sin at 0 cycles back to 0; every derivative of exp is exp,
so d⁸exp(0.5) = e^{0.5} ≈ 1.64872. Note `derivative-n` reaches order 8 and
beyond, where the old fixed 4-component jet returned a flat 0.

### `taylor` — the coefficient series

```scheme
(define e05 (exp 0.5))
(define cs (taylor (lambda (x) (exp x)) 0.5 4))   ; c[0..4]
(display cs) (newline)

;; f^(n) = n! . c_n
(display (* (list-ref cs 4) 24.0)) (newline)      ; 4! . c4 = f''''(0.5)
(display e05) (newline)
```

Output:

```
(1.64872 1.64872 0.824361 0.274787 0.0686967)
1.64872
1.64872
```

The coefficients of exp about 0.5 are `e^{0.5}/n!`: `1.64872, 1.64872,
0.824361, 0.274787, 0.0686967`. Multiplying c₄ by 4! recovers the 4th
derivative, confirming `f⁽ⁿ⁾ = n!·cₙ`.

**Call form:** both are `(derivative-n f x k)` and `(taylor f x k)` — plain
positional arguments, `k` an integer order, `x` the evaluation point. (There is
no `#:order` keyword form; the design sketch's keyword syntax was dropped for
implementation reasons — see the `taylor_numerics` module header.)

**When to use:** any high-order derivative — series expansions, stiff-ODE
coefficients, higher-order optimization, physics requiring `f⁗` and beyond.

---

## 3. Exact coefficients (bignum & rational)

If you seed the tower at an **exact** point (an integer or rational, not a
double) and the function uses only exact-preserving operations (`+ - * /` and
non-negative integer `expt`), the derivatives come back **exact** — arbitrary
precision, zero floating error. This is a capability JAX and PyTorch simply do
not have.

```scheme
(define (x30 x) (expt x 30))
(display (derivative-n x30 7 0)) (newline)    ; 7^30 -- exact bignum
(display (exact? (derivative-n x30 7 0))) (newline)
(display (derivative-n x30 7 1)) (newline)    ; 30 . 7^29 -- exact bignum

(define (geom x) (/ 1 (- 1 x)))
(define half (/ 1 2))
(display (derivative-n geom half 5)) (newline) ; 5! . 2^6 via exact rational algebra
(display (exact? (derivative-n geom half 5))) (newline)
```

Output:

```
22539340290692258087863249
#t
96597172674395391805128210
7680
#t
```

`7³⁰` is a 26-digit integer, far past `INT64_MAX` — this genuinely exercises the
bignum substrate, not just int64. `1/(1-x)` at `x = 1/2` differentiates through
exact **rational** division (`w₀ = 1 - 1/2 = 1/2`) and lands on the exact
integer `5!·2⁶ = 7680`. `(exact? …)` confirms the result carries the exact tag.

The moment a transcendental (`exp`/`log`/`sin`/`cos`/…) enters, the tower
gracefully demotes to the ordinary double series — R7RS exactness contagion,
applied to arbitrary-order AD.

**When to use:** exact high-order derivatives of polynomials and rational
functions — symbolic-quality coefficients (combinatorial generating functions,
exact Taylor tables, verified numerics) without a computer-algebra system.

---

## 4. Multivariate mixed partials (GUW)

For mixed partials of order ≥ 3 in several variables, load `core.ad.guw`. It
implements the Griewank–Utke–Walther directional-interpolation method: it
propagates univariate Taylor towers along a set of directions and solves a small
linear system to recover the full symmetric derivative tensor — no new AD
primitive, just orchestration over `taylor`.

```scheme
(require core.ad.guw)

(define (f v)
  (let ((x (vector-ref v 0)) (y (vector-ref v 1)))
    (+ (* x x x y y) (sin (* x y)))))

(define xs (vector 1.0 0.5))

;; order-3 mixed partial: d^3f / dx^2 dy
(display (mixed-partial f xs (list 0 0 1))) (newline)

;; order-3 mixed partial: d^3f / dx dy^2
(display (mixed-partial f xs (list 0 1 1))) (newline)

;; the full symmetric order-3 tensor, as (multi-index . value) pairs
(display (gradient-n f xs 3)) (newline)
```

Run with `ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r ex.esk -L build`. Output:

```
5.30118
4.60236
(((0 3) . -0.877583) ((1 2) . 4.60236) ((2 1) . 5.30118) ((3 0) . 1.3903))
```

`mixed-partial` takes a list of variable indices *with repetition*: `(0 0 1)` is
∂³f/∂x²∂y. `gradient-n f xs 3` returns the whole order-3 tensor as
`(β . value)` pairs, where β is the multi-index `(dx dy)`: `(3 0)` is ∂³/∂x³,
`(2 1)` is ∂³/∂x²∂y (matching the `mixed-partial` call above), and so on.

Both `mixed-partial` and `gradient-n` require order ≥ 3; use `gradient`/
`hessian` for orders 1 and 2, which stay on the fast jet path.

**When to use:** third- and higher-order sensitivities across multiple
variables — higher-order optimization, moment expansions, sensitivity analysis.

---

## 5. Tensor AD — towers of tensors

High-order AD is not just for scalars. `core.ad.tensor_tower` generalizes the
Taylor coefficients from scalars to **tensors**, so you can take high-order
derivatives of tensor computations (elementwise chains, matmul, conv2d,
sigmoid/tanh activations). The series index and the tensor axes are orthogonal
and compose cleanly.

```scheme
(require stdlib)
(require core.ad.tensor_tower)

;; f(t) = sum(A .* t^3)  =>  h(t) = sum(A)*t^3
(define A (tensor 2 2 2.0 3.0 -1.5 4.0))
(define t0 1.7)
(define sc (tt-sum (tt-from-scalar-series A (taylor (lambda (t) (* t t t)) t0 4))))

(display (tt-nth-derivative sc 0)) (newline)   ; h(t0)  = sum(A)*t0^3
(display (tt-nth-derivative sc 1)) (newline)   ; h'(t0) = 3*sum(A)*t0^2
(display (tensor-sum A)) (newline)             ; sum(A)
```

Run with `ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r ex.esk -L build`. Output:

```
36.8475
65.025
7.5
```

sum(A) = 2+3−1.5+4 = 7.5. h(t) = 7.5·t³ so h(1.7) = 7.5·4.913 = 36.8475 and
h′(1.7) = 22.5·1.7² = 65.025 — exactly the tensor-tower results. The module
also provides `tt-matmul-cauchy` and `tt-conv2d-cauchy` for bilinear ops
(matmul, conv2d) and `tt-sigmoid`/`tt-tanh` for activations, so the same
machinery differentiates the ML path to arbitrary order.

**When to use:** high-order derivatives through tensor/neural-network
computations — curvature of losses over conv/attention layers, higher-order
tensor sensitivities.

---

## 6. Validated AD — Taylor models

A **Taylor model** is a Taylor polynomial plus a rigorous interval remainder
that provably encloses the truncation error over a whole domain box. Instead of
"the value at a point," you get a *guaranteed enclosure* of the function over an
interval — the foundation of validated numerics and verified global
optimization. Load `core.ad.taylor_models`.

```scheme
(require core.ad.taylor_models)

;; a guaranteed enclosure of exp(x) over x in [-0.5, 0.5]
(define tm (taylor-model (lambda (x) (exp x)) 0.0 0.5 6))
(define rng (tm-range tm))
(display (interval-lo rng)) (newline)
(display (interval-hi rng)) (newline)

;; the enclosure at a point always contains the true value
(display (interval-contains? (tm-eval tm 0.25) (exp 0.25))) (newline)
```

Run with `ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r ex.esk -L build`. Output:

```
0.35127
1.64873
#t
```

`(taylor-model f center radius k)` builds an order-`k` model over
`[center−radius, center+radius]`. `tm-range` returns a guaranteed
`[lo, hi]` enclosing every value of exp on `[-0.5, 0.5]` — and indeed
`0.35127 ≤ e^{-0.5}=0.6065` and `e^{0.5}=1.6487 ≤ 1.64873`, with the enclosure
sound (it contains the true range). `tm-eval` gives a point enclosure that
provably contains the true value. Enclosures tighten as the order `k` grows and
as the radius shrinks.

**When to use:** when you need a *proof*, not an estimate — rigorous error
bounds, verified global optimization, guaranteed ODE enclosures, robust handling
of catastrophic cancellation.

---

## 7. Sparse high-order — sparse Hessians

Many functions are *partially separable*: their Hessian is mostly zero (banded,
block, or stencil structure). `core.ad.sparse_guw` recovers such a Hessian using
distance-2 graph coloring, so the number of AD passes tracks the *bandwidth*,
not the dimension — recovering an m×m Hessian in a handful of passes instead of
m of them.

```scheme
(require core.ad.sparse_guw)

;; banded f = sum(x_i^2) + sum(x_i * x_{i+1}): only 3 AD passes needed, not m
(define (fq v)
  (let ((m (vector-length v)))
    (let loop ((i 0) (acc 0.0))
      (if (< i m)
          (let ((x (vector-ref v i)))
            (loop (+ i 1)
                  (+ acc (* x x)
                     (if (< i (- m 1)) (* x (vector-ref v (+ i 1))) 0.0))))
          acc))))

(define xs (vector 0.3 0.5 0.7 0.9 1.1))
(define sp (sparse-hessian fq xs))

(display (sparse-hessian-ref sp 0 0)) (newline)   ; H_00 = 2
(display (sparse-hessian-ref sp 0 1)) (newline)   ; H_01 = 1
(display (sparse-hessian-ref sp 0 2)) (newline)   ; H_02 = 0 (structural zero)
(display (sparse-hessian-colors sp)) (newline)    ; AD passes used
```

Run with `ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r ex.esk -L build`. Output:

```
2
1
0
3
```

The Hessian is `H_ii = 2`, `H_{i,i+1} = 1`, everything else 0. `sparse-hessian`
probes the structure automatically, recovers it in **3 colors (AD passes)** —
independent of the dimension `m` — and gives CSR accessors
(`sparse-hessian-row-ptr`/`-col-idx`/`-values`). A fully-dense function is
detected and falls back to the dense path. (An explicit
`sparse-hessian-pat` lets you supply a known sparsity pattern instead of
probing.)

**When to use:** large partially-separable Hessians — banded/block problems,
finite-element stencils, structured ML second-order methods — where a dense
Hessian would be prohibitively expensive.

---

## 8. Reverse mode & checkpointing

### Reverse-over-Taylor: high-order gradients

`gradient` (reverse mode) composes with `derivative-n` (the high-order forward
tower): you can take the gradient, with respect to outer parameters, of a
quantity that is itself a high-order derivative.

```scheme
;; d/dv [ d^3/dt^3 sin(v*t) ]  at t0 = 0.4, then differentiated in v at v0 = 0.6
(define (g v) (derivative-n (lambda (t) (sin (* v t))) 0.4 3))
(display (g 0.6)) (newline)          ; f'''(t) = -v^3 cos(v t)
(display (gradient g 0.6)) (newline) ; d/dv of that
```

Output:

```
-0.209809
-1.02851
```

`g(v)` is the 3rd `t`-derivative of `sin(v·t)`, which analytically is
`−v³cos(v·t)`; at v=0.6, t=0.4 that is `−0.209809`. Wrapping it in `gradient`
differentiates *that* with respect to `v`, giving `−1.02851`. Before v1.3 this
returned a flat 0 (the reverse tape "swallowed" the tower); the seed-tangent
dual tower fixes it. This is exact, not finite-difference.

### Checkpointed reverse for deep graphs

A naive reverse pass over a deep, high-order graph stores O(N·K) tape data.
`core.ad.checkpoint` implements Griewank binomial checkpointing over
tower-valued tapes, so peak memory scales like O(√N·K) instead — rematerializing
segments during the backward sweep rather than storing them all.

```scheme
(require core.ad.checkpoint)

(define t0 0.3) (define v0 0.8)
(define order 3) (define korder 3) (define n 100)

;; dense: one tape spans all N layers -- O(N) peak tape memory
(define dv-dense (dense-gradient t0 v0 n order korder))

;; checkpointed: Griewank sqrt(N) schedule -- O(sqrt(N)) peak memory
(define dv-ckpt  (checkpointed-gradient t0 v0 n order korder))

(display dv-dense) (newline)
(display dv-ckpt) (newline)
(display (checkpoint-block-size n)) (newline)   ; ~sqrt(100)
```

Run with `ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r ex.esk -L build`. Output:

```
-3.31007e-08
-3.31007e-08
10
```

The checkpointed gradient is **identical** to the dense one, but its block size
is √100 = 10, and (as the module's own test measures) peak tape memory grows
sub-linearly — for an N=200 chain, dense peaks at ~9600 tape nodes while
checkpointed peaks at ~740.

**When to use:** high-order gradients through deep chains / long loops / ML-scale
graphs where storing the full tape is too expensive.

---

## 9. Differentiable control flow

Towers flow through ordinary Scheme control flow — `if`/`cond`/`case` branches,
named-let loops, recursion (including mutual tail recursion), and `map`/`fold`
over closures — because a tower is just a tagged heap value that the control-flow
machinery threads through unchanged.

```scheme
;; f(x) = x^3 for x > 0, -x otherwise (a kink at x=0)
(define (branch-f x) (if (> x 0) (* x x x) (- x)))

;; x0=2.0 selects the (* x x x) arm -- AD differentiates the executed branch
(display (derivative-n branch-f 2.0 0)) (newline)  ; f    = 8
(display (derivative-n branch-f 2.0 1)) (newline)  ; f'   = 3x^2 = 12
(display (derivative-n branch-f 2.0 2)) (newline)  ; f''  = 6x   = 12
(display (derivative-n branch-f 2.0 3)) (newline)  ; f''' = 6
```

Output:

```
8
12
12
6
```

The predicate `(> x 0)` works on a tower-valued `x` (branching on its primal
coefficient), and AD differentiates whichever arm the primal actually selects.

**The kink policy:** at a branch point, Eshkol differentiates *the arm the
primal predicate selects* — the standard forward-mode subderivative convention
used by JAX, PyTorch autograd, Stan, and Julia's ForwardDiff. At an exact kink
(a predicate boundary) this yields one of the one-sided derivatives,
deterministically; it is never an average and never the wrong arm's derivative
smuggled in. Differentiate strictly away from kinks and you get the true
derivative of the locally smooth piece.

**When to use:** differentiating real programs — data-dependent branches, loops,
recursive definitions — not just straight-line kernels.

---

## 10. Tower numerics — ODEs, roots, inversion

Once the tower is first-class, the same kernel powers general numerical methods.
`core.ad.taylor_numerics` exposes three: a Taylor-series ODE integrator,
Householder-family root finding, and local functional inversion — all built
purely on `taylor`.

```scheme
(require core.ad.taylor_numerics)

;; ODE: y' = -y, y(0) = 1  =>  y(1) = exp(-1)
(display (taylor-ode-solve (lambda (t y) (* -1.0 y)) 1.0 0.0 1.0 8 10)) (newline)
(display (exp -1.0)) (newline)

;; root-finding: solve x^2 - 2 = 0 (order 1 = Newton's method)
(display (taylor-root (lambda (x) (- (* x x) 2.0)) 1.0 1)) (newline)
(display (sqrt 2.0)) (newline)

;; local functional inverse of f(x) = x + x^2 around x0 = 0
(display (taylor-inverse-series (lambda (x) (+ x (* x x))) 0.0 5)) (newline)
```

Run with `ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r ex.esk -L build`. Output:

```
0.367879
0.367879
1.41421
1.41421
(0 1 -1 2 -5 14)
```

- **`taylor-ode-solve f y0 t0 t1 k n`** integrates `y' = f(t,y)` from `t0` to
  `t1` in `n` steps of Taylor order `k`. For `y' = -y, y(0)=1` it matches
  `exp(-1) = 0.367879` at order 8 / 10 steps.
- **`taylor-root f x0 k`** refines a root by series reversion: `k=1` is Newton
  (quadratic), `k=2` is Halley (cubic), general `k` gives order-(k+1)
  convergence. Solving `x²−2=0` from `x0=1` lands on `√2 = 1.41421`.
- **`taylor-inverse-series f x0 k`** returns the Taylor coefficients of the local
  inverse `f⁻¹` about `f(x0)`. For `f(x)=x+x²` the inverse series coefficients
  are `(0 1 -1 2 -5 14)` — the Catalan-number-flavored reversion of `x+x²`.

**When to use:** high-accuracy ODE stepping, fast high-order root polishing, and
series inversion / analytic continuation — all reusing the AD kernel.

---

## 11. Perturbation safety & how it works

### Nested differentiation is safe

Nested `derivative`/`derivative-n` is the classic *perturbation-confusion* trap
(Siskind–Pearlmutter): a naive implementation lets an inner differentiation leak
into an outer one, silently producing the wrong answer. Eshkol tags each
differentiation context with a distinct epoch, so this cannot happen.

```scheme
;; The inner derivative (w.r.t. y, at y=3) is a CONSTANT wrt the outer x.
;; Perturbation confusion would let the inner epoch leak into the outer one.
(display
  (derivative-n
    (lambda (x) (+ (* x x)
                   (derivative-n (lambda (y) (* y y y)) 3.0 1)))
    4.0 1))
(newline)
```

Output:

```
8
```

The inner `(derivative-n (lambda (y) (* y y y)) 3.0 1)` is `3y²|_{y=3} = 27`, a
constant with respect to `x`. So the outer function is `x² + 27` and its
derivative at `x=4` is `2·4 = 8` — exactly what comes back. A confusion bug
would corrupt this to something else.

### How it works (in one paragraph)

The single computational kernel is **truncated-Taylor arithmetic**: a function's
behavior near `x₀` is represented by its Taylor coefficients `c[0..K]`, and each
primitive operation has a closed recurrence that maps input coefficient arrays to
output ones — Cauchy convolution for multiplication, coupled recurrences for
`sin`/`cos`, divided recurrences for `/` and `log`, and so on. Because these
recurrences are O(K²), high-order AD is *polynomial* in the order, not the 2ᴷ
blow-up of stacking dual numbers. When the order `K` is a literal at the call
site (the common case in a compiler), the entire tower is emitted as unrolled,
stack-allocated, branch-free IR — no heap allocation in the AD hot loop. Each
active differentiation context carries a distinct **epoch tag** in the tower's
header so nested derivatives never cross-contaminate. Order ≤ 2 keeps the
existing fast 4-component jet byte-for-byte; the tower only appears when order
≥ 3 is requested.

For the full design — the recurrence table, the compile-time monomorphization,
the FP-contraction policy that makes `mono ≡ runtime` bit-exact, the exact and
tensor coefficient layouts, and the Taylor-model remainder arithmetic — see
[`docs/design/AD_TAYLOR_TOWER.md`](../design/AD_TAYLOR_TOWER.md).

---

## 12. API quick reference

### Built-in operators (no `require`)

| Operator | Field | Signature | Notes |
|---|---|---|---|
| `derivative` | ℝ→ℝ | `(derivative f x)` / `(derivative f)` | forward mode; 2nd form returns a function |
| `derivative-n` | ℝ→ℝ | `(derivative-n f x k)` | *k*-th derivative, any `k` (Taylor tower) |
| `taylor` | ℝ→ℝ | `(taylor f x k)` | coefficient list `c[0..k]`; `f⁽ⁿ⁾ = n!·cₙ` |
| `gradient` | ℝⁿ→ℝ | `(gradient f pt)` / `(gradient f)` | reverse mode |
| `jacobian` | ℝⁿ→ℝᵐ | `(jacobian f pt)` | rows are ∇fᵢ |
| `hessian` | ℝⁿ→ℝ | `(hessian f pt)` | matrix of 2nd partials |
| `divergence` | ℝⁿ→ℝⁿ | `(divergence F pt)` | ∇·F |
| `curl` | ℝ³→ℝ³ | `(curl F pt)` | ∇×F |
| `laplacian` | ℝⁿ→ℝ | `(laplacian f pt)` | ∇²f |
| `directional-derivative` | ℝⁿ→ℝ | `(directional-derivative f pt dir)` | ∇f·dir |

Points are vectors (`#(…)` or `(vector …)`). Exact seeds (integer/rational)
yield exact `derivative-n`/`taylor` results for polynomial/rational functions.

### Library modules (`(require core.ad.<name>)`)

| Module | Public API | Purpose |
|---|---|---|
| `core.ad.guw` | `taylor-propagate`, `mixed-partial`, `gradient-n` | multivariate mixed partials, order ≥ 3 |
| `core.ad.tensor_tower` | `tt-from-scalar-series`, `tt-matmul-cauchy`, `tt-conv2d-cauchy`, `tt-sigmoid`, `tt-tanh`, `tt-sum`, `tt-nth-derivative`, … | high-order AD through tensor computations |
| `core.ad.taylor_models` | `taylor-model`, `tm-range`, `tm-eval`, `tm-coeffs`, `tm-remainder`, … | validated AD (rigorous enclosures) |
| `core.ad.interval` | `make-interval`, `interval-lo/-hi`, `interval-contains?`, `interval-width`, … | outward-rounded interval arithmetic (used by Taylor models) |
| `core.ad.sparse_guw` | `sparse-hessian`, `sparse-hessian-pat`, `sparse-hessian-ref`, `sparse-hessian-colors`, `sparse-hessian-row-ptr/-col-idx/-values`, `sparse-hessian-dense?` | colored sparse-Hessian recovery |
| `core.ad.checkpoint` | `dense-gradient`, `checkpointed-gradient`, `dense-tower-reverse`, `checkpointed-tower-reverse`, `checkpoint-block-size` | Griewank-checkpointed high-order reverse |
| `core.ad.taylor_numerics` | `taylor-ode-solve`, `taylor-root`, `taylor-inverse-series`, `taylor-eval` | series-based ODE / root / inversion |
| `core.ad.tape` | `make-tape`, `tape-input`, `tape-mul`/`-add`/…, `tape-gradient`, `tape-node-count`, … | explicit reverse-mode tape (Scheme level) |

**Running module examples:** because these modules are loaded via `require`,
run them with the build directory on the library path and the JIT object cache
disabled:

```
ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r your-file.esk -L build
```

The built-in-operator examples in §1–§2 need only `./build/eshkol-run -r
your-file.esk`.

### Further reading

- [`docs/design/AD_TAYLOR_TOWER.md`](../design/AD_TAYLOR_TOWER.md) — the full
  Taylor-tower design (P0–P12): representation, recurrences, monomorphization,
  exact/tensor/validated coefficient tiers, and every phase's correctness gate.
- [`docs/API_REFERENCE.md`](../API_REFERENCE.md) — the built-in AD operators in
  the context of the whole language surface.
- [`docs/reference/ad/INDEX.md`](../reference/ad/INDEX.md) — the machine-verified
  AD operator reference and support matrix.
