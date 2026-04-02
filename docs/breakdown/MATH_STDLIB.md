# Eshkol Mathematics Standard Library

**Status:** Stable — v1.1-accelerate
**Module:** `math`, `math.constants`, `math.special`, `math.ode`, `math.statistics`, `random`
**Sources:** `lib/math/constants.esk`, `lib/math/special.esk`, `lib/math/ode.esk`,
`lib/math/statistics.esk`, `lib/random/random.esk`

---

## Overview

The Eshkol mathematics standard library provides a comprehensive suite of numerical
computing tools suitable for scientific, engineering, and machine learning applications.
The library is organized into five sub-modules, each independently importable:

| Sub-module           | Import form                    | Contents                               |
|----------------------|-------------------------------|----------------------------------------|
| `math.constants`     | `(require math.constants)`    | Mathematical and physical constants    |
| `math.special`       | `(require math.special)`      | Special functions (gamma, Bessel, etc) |
| `math.ode`           | `(require math.ode)`          | ODE solvers (Euler through RK45)       |
| `math.statistics`    | `(require math.statistics)`   | Descriptive statistics                 |
| `random`             | `(require random)`            | PRNG, quantum RNG, distributions       |

All five modules are imported together with `(require math)`.

All computations use IEEE 754 double precision (64-bit) floating point. The ODE solvers
and special functions handle both scalar and vector (Eshkol `vector`) state representations.

---

## 1. Mathematical Constants

### 1.1 Fundamental Constants

| Binding        | Value                           | Description                              |
|----------------|---------------------------------|------------------------------------------|
| `pi`           | 3.14159265358979323846...       | Ratio of circumference to diameter, π    |
| `e`            | 2.71828182845904523536...       | Euler's number, base of natural log      |
| `phi`          | 1.61803398874989484820...       | Golden ratio, (1+√5)/2                   |
| `euler-gamma`  | 0.57721566490153286060...       | Euler-Mascheroni constant γ              |
| `tau`          | 6.28318530717958647692...       | Full circle, 2π                          |
| `pi/2`         | 1.57079632679489661923...       | Quarter circle                           |
| `pi/4`         | 0.78539816339744830961...       | Eighth circle                            |
| `sqrt-pi`      | 1.77245385090551602729...       | √π                                       |
| `sqrt-2pi`     | 2.50662827463100050241...       | √(2π), Gaussian normalization factor     |

### 1.2 Logarithmic Constants

| Binding    | Value                    | Description                  |
|------------|--------------------------|------------------------------|
| `ln2`      | 0.693147180559945309...  | ln 2                         |
| `ln10`     | 2.302585092994045684...  | ln 10                        |
| `log2e`    | 1.442695040888963407...  | log₂ e                       |
| `log10e`   | 0.434294481903251827...  | log₁₀ e                      |

### 1.3 Square Root Constants

| Binding      | Value                    | Description    |
|--------------|--------------------------|----------------|
| `sqrt2`      | 1.41421356237309504880...| √2             |
| `sqrt3`      | 1.73205080756887729352...| √3             |
| `sqrt5`      | 2.23606797749978969640...| √5             |
| `inv-sqrt2`  | 0.70710678118654752440...| 1/√2 = √2/2   |

### 1.4 Machine Precision Constants

| Binding            | Value                   | Description                                         |
|--------------------|-------------------------|-----------------------------------------------------|
| `machine-epsilon`  | 2.220446049250313e-16   | Unit roundoff: smallest x with 1.0+x ≠ 1.0 (IEEE 754) |
| `double-min`       | 2.2250738585072014e-308 | Smallest positive normalized double                |
| `double-max`       | 1.7976931348623157e+308 | Largest finite double                               |
| `epsilon`          | 1e-15                   | Default tolerance for numerical comparisons         |

Machine epsilon equals 2⁻⁵², consistent with the IEEE 754 binary64 standard. For
algorithms that need guaranteed convergence, tolerances should be set several orders of
magnitude above `machine-epsilon` to avoid perpetual refinement.

### 1.5 Physical Constants — CODATA 2019

All physical constants use SI units and the 2019 CODATA recommended values. Constants
marked "exact" have zero uncertainty under the 2019 SI redefinition.

| Binding              | Symbol | Value               | Units      | Notes         |
|----------------------|--------|---------------------|------------|---------------|
| `c`                  | c      | 299792458.0         | m/s        | Exact         |
| `h`                  | h      | 6.62607015e-34      | J·s        | Exact         |
| `hbar`               | ℏ      | 1.054571817e-34     | J·s        | h/(2π)        |
| `elementary-charge`  | e      | 1.602176634e-19     | C          | Exact         |
| `k-boltzmann`        | k_B    | 1.380649e-23        | J/K        | Exact         |
| `avogadro`           | N_A    | 6.02214076e23       | mol⁻¹      | Exact         |
| `G`                  | G      | 6.67430e-11         | m³/(kg·s²) | Measured      |
| `m-electron`         | m_e    | 9.1093837015e-31    | kg         |               |
| `m-proton`           | m_p    | 1.67262192369e-27   | kg         |               |
| `m-neutron`          | m_n    | 1.67492749804e-27   | kg         |               |
| `alpha`              | α      | 7.2973525693e-3     | —          | Fine-structure|
| `a0`                 | a₀     | 5.29177210903e-11   | m          | Bohr radius   |
| `eV`                 | eV     | 1.602176634e-19     | J          | = e (exact)   |

The fine-structure constant α = e²/(4πε₀ℏc) characterizes the strength of
electromagnetic interaction. The Bohr radius a₀ = ℏ/(m_e·c·α) is the most probable
electron-nucleus distance in a ground-state hydrogen atom.

### 1.6 Angle Conversion Utilities

```scheme
(require math.constants)

;; Conversion factors
deg->rad  ;; π/180 = 0.01745329251994329...
rad->deg  ;; 180/π = 57.29577951308232...

;; Conversion functions
(degrees->radians deg)    ;; (* deg deg->rad)
(radians->degrees rad)    ;; (* rad rad->deg)

;; Approximate equality
(approx= a b tolerance)   ;; (< (abs (- a b)) tolerance)
(approx-zero? x tolerance);; (< (abs x) tolerance)

;; Example
(degrees->radians 45.0)   ;; => 0.7853981633974483  (= pi/4)
(approx= pi 3.14159 1e-5) ;; => #t
```

---

## 2. Special Functions

### 2.1 Gamma Function and Related

#### `(gamma z)` — Euler Gamma Function

**Mathematical definition:**

    Γ(z) = ∫₀^∞  t^(z-1) · e^(-t)  dt,    Re(z) > 0

Extended to the negative real axis (excluding non-positive integers) via the reflection formula:

    Γ(z) = π / (sin(πz) · Γ(1-z))

**Implementation:** Lanczos approximation with g=7 and 9 Lanczos coefficients:

    Γ(z+1) ≈ √(2π) · (z + g + 1/2)^(z+1/2) · e^(-(z+g+1/2)) · A_g(z)

    A_g(z) = c₀ + Σ_{k=1}^{8}  c_k / (z + k)

Coefficients (Spouge/Lanczos form):

    c₀ = 0.99999999999980993
    c₁ = 676.5203681218851
    c₂ = −1259.1392167224028
    c₃ = 771.32342877765313
    c₄ = −176.61502916214059
    c₅ = 12.507343278686905
    c₆ = −0.13857109526572012
    c₇ = 9.9843695780195716×10⁻⁶
    c₈ = 1.5056327351493116×10⁻⁷

For z < 0.5 the reflection formula redirects to a half-plane where the approximation
converges. Returns incorrect results for non-positive integer arguments (poles of Γ).

```scheme
(gamma 5.0)     ;; => 24.0   (= 4!)
(gamma 0.5)     ;; => 1.7724538509055159  (= sqrt(pi))
(gamma 1.0)     ;; => 1.0
```

#### `(lgamma z)` — Log-Gamma Function

Computes ln|Γ(z)| using the same Lanczos scheme in logarithmic form:

    ln Γ(z) ≈ 1/2·ln(2π) + (z-1/2)·ln(t) - t + ln(A_g(z-1))
    where t = z - 1 + g + 1/2

Numerically superior to `(log (gamma z))` for large z where `gamma` would overflow the
double range (Γ overflows around z ≈ 172).

```scheme
(lgamma 172.0)  ;; => 711.714...   (gamma would overflow)
(lgamma 1.0)    ;; => 0.0
```

#### `(factorial n)` — Factorial

Computes n! via `(gamma (+ n 1.0))`. Returns `#f` for n < 0.

```scheme
(factorial 10)  ;; => 3628800.0
(factorial 0)   ;; => 1.0
(factorial -1)  ;; => #f
```

#### `(beta a b)` — Beta Function

    B(a, b) = Γ(a)Γ(b) / Γ(a+b)

Computed in log-space as `(exp (+ (lgamma a) (lgamma b) (- (lgamma (+ a b)))))` to
prevent intermediate overflow.

```scheme
(beta 2.0 3.0)  ;; => 0.08333...  (= 1/12)
```

#### `(digamma x)` — Digamma (Psi) Function

    ψ(x) = d/dx ln Γ(x) = Γ'(x) / Γ(x)

**Algorithm:** Upward recursion for x < 6 using the identity ψ(x) = ψ(x+1) − 1/x,
then asymptotic expansion for x ≥ 6:

    ψ(x) ≈ ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸)

Convergence of the asymptotic series requires x to be large; the recursion ensures
the argument is ≥ 6 before the expansion is applied.

```scheme
(digamma 1.0)   ;; => -0.5772...  (= -euler-gamma)
(digamma 2.0)   ;; => 0.4227...   (= 1 - euler-gamma)
```

### 2.2 Error Functions

#### `(erf x)` — Error Function

    erf(x) = (2/√π) · ∫₀^x  e^(-t²)  dt

**Implementation:** Abramowitz & Stegun approximation 7.1.26, maximum error 1.5×10⁻⁷.
Uses Horner evaluation of a degree-5 polynomial in t = 1/(1 + 0.3275911·|x|):

    erf(x) = sign(x) · (1 − (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵) · e^(-x²))

Coefficients: a₁=0.254829592, a₂=−0.284496736, a₃=1.421413741,
              a₄=−1.453152027, a₅=1.061405429.

```scheme
(erf 0.0)   ;; => 0.0
(erf 1.0)   ;; => 0.8427007929497148
(erf +inf.0);; => 1.0
```

#### `(erfc x)` — Complementary Error Function

    erfc(x) = 1 − erf(x) = (2/√π) · ∫_x^∞  e^(-t²)  dt

Note: For large positive x, precision degrades because erfc approaches 0 and the
subtraction `(- 1.0 (erf x))` loses significant digits. Applications requiring high
precision for x > 3 should use a dedicated continued-fraction implementation.

#### `(erfinv y)` — Inverse Error Function

Computes x such that erf(x) = y for y ∈ (−1, 1). Returns `+inf.0` for y=1, `-inf.0`
for y=−1, `+nan.0` for |y| > 1. Uses a rational polynomial approximation.

#### `(normcdf x)` — Standard Normal CDF

    Φ(x) = (1/2) · (1 + erf(x/√2))

The CDF of the standard normal distribution N(0,1).

#### `(normpdf x)` — Standard Normal PDF

    φ(x) = (1/√(2π)) · e^(-x²/2)

```scheme
(normpdf 0.0)   ;; => 0.3989422804014327
(normcdf 0.0)   ;; => 0.5
(normcdf 1.96)  ;; => 0.9750021048517796  (~97.5th percentile)
```

### 2.3 Bessel Functions

#### `(besselj0 x)` — J₀(x)

Bessel function of the first kind, order zero, defined as the solution to:

    x² y'' + x y' + x² y = 0

**Algorithm (|x| < 8):** Rational Chebyshev approximation using 6-term polynomials
in y = x² for numerator and denominator, providing machine-precision accuracy.

**Algorithm (|x| ≥ 8):** Asymptotic expansion:

    J₀(x) ≈ √(2/πx) · (P₀(x)·cos(x − π/4) − Q₀(x)·sin(x − π/4))

where P₀, Q₀ are 4-term polynomial corrections in (8/x)².

#### `(besselj1 x)` — J₁(x)

Bessel function of the first kind, order one. Same dual-regime approximation as J₀,
asymptotic phase shift is 3π/4:

    J₁(x) ≈ √(2/πx) · (P₁(x)·cos(x − 3π/4) − Q₁(x)·sin(x − 3π/4))

Satisfies J₁(−x) = −J₁(x).

#### `(besseljn n x)` — Jₙ(x)

Integer-order Bessel functions for any n ≥ 0. Dispatches to `besselj0` or `besselj1`
for n=0,1. For n < 0 applies the symmetry relation Jₙ(x) = (−1)ⁿ J₋ₙ(x).

**Algorithm:** Miller's downward recurrence. Starting at an index m much larger than n
(m = 2(n + ⌊√(40n)⌋)), downward recurrence is numerically stable:

    J_{k-1}(x) = (2k/x) · J_k(x) − J_{k+1}(x)

The result is normalized using the Neumann identity Σ Jₙ(x) = 1 via accumulated
even-order partial sums.

```scheme
(besselj0 2.4048)  ;; => ~0.0  (first zero of J₀)
(besseljn 3 5.0)   ;; => 0.3648...
```

### 2.4 Incomplete Gamma Functions

#### `(gammainc-lower a x)` — Lower Incomplete Gamma γ(a,x)

    γ(a, x) = ∫₀^x  t^(a-1) · e^(-t)  dt

Returns `#f` for x < 0 or a ≤ 0.

**Algorithm:** Series expansion:

    γ(a, x) = e^(a·ln x − x − ln Γ(a)) · Σ_{n=0}^∞  x^n / (a · (a+1) · ... · (a+n))

Terminated when relative term size drops below ε=10⁻¹⁵ or after 100 iterations.
Underflow protection: returns 0 when the log-prefactor falls below −88.

#### `(gammainc-upper a x)` — Upper Incomplete Gamma Γ(a,x)

    Γ(a, x) = Γ(a) − γ(a, x) = ∫_x^∞  t^(a-1) · e^(-t)  dt

Note: For large x relative to a, numerically superior continued fraction algorithms
exist. This implementation uses the direct difference formula.

### 2.5 Riemann Zeta Function

#### `(zeta s)` — ζ(s)

    ζ(s) = Σ_{n=1}^∞  1/n^s,    s > 1

Returns `+inf.0` for s=1 (pole), `#f` for s < 1.

**Algorithm:** Euler-Maclaurin summation formula. Computes 10 direct terms, then adds:
- Integral approximation: n^(1−s) / (s−1)
- Endpoint correction: 1/(2·n^s)
- First Bernoulli correction: s / (12·n^(s+1))

Accuracy is moderate (a few digits of relative error); high-precision applications
should use the full Euler-Maclaurin expansion with more Bernoulli terms.

```scheme
(zeta 2.0)   ;; => ~1.6449340668...  (= π²/6)
(zeta 4.0)   ;; => ~1.0823232337...  (= π⁴/90)
```

### 2.6 Exponential Integral

#### `(expint-e1 x)` — E₁(x)

    E₁(x) = ∫_x^∞  e^(-t)/t  dt,    x > 0

Returns `+inf.0` for x ≤ 0.

**Algorithm (x < 1):** Power series with Euler-Mascheroni constant:

    E₁(x) = −γ − ln x − Σ_{n=1}^∞  (−x)^n / (n · n!)

**Algorithm (x ≥ 1):** Lentz continued fraction algorithm:

    E₁(x) = e^(-x) · cf(x)

where cf(x) is the continued fraction 1/(x+1/(1+1/(x+2/(1+2/(x+...))))), evaluated
with the modified Lentz method to machine precision.

---

## 3. ODE Solvers

### 3.1 Background

An ordinary differential equation initial value problem (IVP) takes the form:

    dy/dt = f(t, y),    y(t₀) = y₀

where y may be scalar or vector-valued. The goal is to approximate y on [t₀, t_f].

All Eshkol ODE solvers accept the same calling convention:
- `f`: a procedure `(lambda (t y) ...)` returning the derivative dy/dt
- `y` can be a scalar (number) or a heterogeneous `vector`
- Solvers return a list of `(t y)` pairs; `*-final` variants return only the final value

**Vector arithmetic helpers** (available in `math.ode`):

```scheme
(vec-add v1 v2)        ;; Element-wise sum
(vec-sub v1 v2)        ;; Element-wise difference
(vec-scale s v)        ;; Scalar-vector product
(vec-max-abs v)        ;; max |vᵢ|  (for error norms)
(state-vector? y)      ;; #t if y is a vector
```

### 3.2 Euler Method (Order 1, Explicit)

**Formula:**

    y_{n+1} = y_n + h · f(t_n, y_n)

**Local truncation error:** O(h²); global error O(h).

**Stability:** Conditionally stable. For the test equation y' = λy, requires
|1 + hλ| ≤ 1. For purely imaginary eigenvalues the method is always unstable.

**API:**

```scheme
(euler-step f t y h)              ;; Single step -> y_{n+1}
(euler    f t0 y0 tf h)           ;; Full trajectory -> ((t y) ...)
(euler-final f t0 y0 tf h)        ;; Final value only
```

**Example — exponential decay (y' = -y, y(0) = 1):**

```scheme
(require math.ode)
(define (decay t y) (- y))
(euler-final decay 0.0 1.0 1.0 0.01)
;; => ~0.3660  (exact: e^-1 = 0.3679)
```

### 3.3 Heun's Method (Order 2, Explicit)

Also called the improved Euler or explicit trapezoidal method.

**Formula:**

    k₁ = f(t_n, y_n)
    y* = y_n + h · k₁                     (predictor)
    k₂ = f(t_{n+1}, y*)
    y_{n+1} = y_n + (h/2) · (k₁ + k₂)    (corrector)

**Local truncation error:** O(h³); global error O(h²).

**Stability region:** Larger than Euler; includes portions of the imaginary axis.

**API:**

```scheme
(heun-step f t y h)
(heun      f t0 y0 tf h)
```

### 3.4 Midpoint Method (Order 2, Explicit)

Equivalent in accuracy to Heun but uses a midpoint evaluation:

**Formula:**

    k₁ = f(t_n, y_n)
    y_{mid} = y_n + (h/2) · k₁
    y_{n+1} = y_n + h · f(t_n + h/2, y_{mid})

**API:**

```scheme
(midpoint-step f t y h)
(midpoint      f t0 y0 tf h)
```

### 3.5 Classical Runge-Kutta (RK4, Order 4, Explicit)

The workhorse of scientific computing. Four function evaluations per step.

**Butcher tableau / formula:**

    k₁ = f(t_n,       y_n)
    k₂ = f(t_n + h/2, y_n + h·k₁/2)
    k₃ = f(t_n + h/2, y_n + h·k₂/2)
    k₄ = f(t_n + h,   y_n + h·k₃)
    y_{n+1} = y_n + (h/6) · (k₁ + 2k₂ + 2k₃ + k₄)

**Local truncation error:** O(h⁵); global error O(h⁴).

**Stability:** For y' = λy, stable when |hλ| lies within the RK4 stability region,
which extends to approximately −2.8 on the negative real axis.

**API:**

```scheme
(rk4-step  f t y h)
(rk4       f t0 y0 tf h)    ;; -> list of (t y) pairs
(rk4-final f t0 y0 tf h)    ;; -> final y
```

**Example — harmonic oscillator (y'' + y = 0 as system):**

```scheme
(require math.ode)

;; State: #(position velocity)
(define (oscillator t state)
  (let ((pos (vref state 0))
        (vel (vref state 1)))
    (vector vel (- pos))))    ;; d/dt [x, v] = [v, -x]

(define traj
  (rk4 oscillator 0.0 (vector 1.0 0.0) (* 2 3.14159265) 0.01))

;; Final position should be ~1.0 (full period)
(vref (cadr (car (reverse traj))) 0)   ;; => ~1.0
```

### 3.6 Runge-Kutta-Fehlberg RK45 (Adaptive, Orders 4/5)

Embeds a 4th-order method inside a 5th-order method for automatic step size control.
Uses Cash-Karp coefficients.

**Six-stage Butcher tableau (Cash-Karp variant):**

Stage | c (time node) | Coupling coefficients a_{ij}
------|---------------|------------------------------
k₁   | 0             | —
k₂   | 1/5           | 1/5
k₃   | 3/10          | 3/40, 9/40
k₄   | 3/5           | 3/10, −9/10, 6/5
k₅   | 1             | −11/54, 5/2, −70/27, 35/27
k₆   | 7/8           | 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096

5th-order weights b: 37/378, 0, 250/621, 125/594, 0, 512/1771
4th-order weights b*: 2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4

**Step control:**

    err = max |y₅ − y₄|
    If err ≤ tol:   accept step,  h_new = min(5h,  0.9·h·(err/tol)^{-0.2})
    If err > tol:   reject step,  h_new = max(h/10, 0.9·h·(err/tol)^{-0.25})

Minimum step: 10⁻¹², maximum step: |t_f − t₀|.

**API:**

```scheme
(rk45-step  f t y h tol)       ;; -> (y-new h-new accepted?)
(rk45       f t0 y0 tf h0 tol) ;; -> list of (t y) pairs (variable spacing)
(rk45-final f t0 y0 tf h0 tol) ;; -> final y
```

**Example — stiff-ish decay with adaptive stepping:**

```scheme
(require math.ode)
(define (fast-decay t y) (* -10.0 y))
(rk45-final fast-decay 0.0 1.0 2.0 0.1 1e-8)
;; => ~e^{-20} ≈ 2.06e-9  (solver auto-reduces h for stiff region)
```

### 3.7 Backward Euler (Order 1, Implicit, A-Stable)

Solves the implicit equation:

    y_{n+1} = y_n + h · f(t_{n+1}, y_{n+1})

**A-stability:** Stable for all hλ with Re(λ) < 0, making it appropriate for stiff
systems where the explicit methods require prohibitively small step sizes.

**Algorithm:**

- **Scalar:** Newton iteration with numerical Jacobian.
  Residual: g(y_guess) = y_guess − y_n − h·f(t_{n+1}, y_guess)
  Numerical derivative: dg/dy ≈ (g(y+ε) − g(y)) / ε  with ε = 10⁻⁸
  Initial guess from explicit Euler; terminates when |Δy| < newton_tol = 10⁻¹⁰
  or after max_newton_iters = 10 iterations.

- **Vector:** Fixed-point iteration (no Jacobian computation).
  Iterate: y_new = y_n + h · f(t_{n+1}, y_guess) until ‖y_new − y_guess‖_∞ < tol.

**API:**

```scheme
(backward-euler-step f t y h max-iters tol)
(backward-euler      f t0 y0 tf h)    ;; uses defaults: max=10, tol=1e-10
```

**When to prefer backward Euler:** Stiff ODEs arising from, e.g., reaction-diffusion
systems, RC circuit transients, or semi-discretized PDEs with small spatial mesh.

---

## 4. Statistics

### 4.1 Data Input

All statistical functions accept tensors, vectors, and (possibly nested) lists via
an internal `flatten-to-list` dispatcher:

    tensor -> tensor->vector -> vector->list
    vector -> vector->list
    list   -> flatten (recursive)
    number -> (list x)

### 4.2 Central Tendency and Order Statistics

#### `(median data)` — Sample Median

Sorts data and returns the middle value (n odd) or average of the two middle values (n even).

#### `(percentile data p)` — p-th Percentile

Linear interpolation between sorted order statistics. For n data points, the index for
percentile p is i = (p/100)·(n−1); the result is a weighted average of floor(i) and
ceil(i) elements.

    P(p) = (1−frac) · x_{⌊i⌋} + frac · x_{⌊i⌋+1},    frac = i − ⌊i⌋

#### `(quartiles data)` — Q1, Q2, Q3

Returns `(list Q1 Q2 Q3)` — the 25th, 50th, and 75th percentiles.

#### `(iqr data)` — Interquartile Range

    IQR = Q3 − Q1

Robust measure of spread, insensitive to outliers. Standard Tukey fence for outlier
detection: values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] are flagged.

### 4.3 Measures of Spread

#### `(variance data)` — Population Variance

    σ² = (1/n) · Σᵢ (xᵢ − x̄)²

Note: This is the **population** variance (divisor n), not the sample variance
(divisor n−1). For unbiased sample estimation, multiply by n/(n−1).

#### `(std-dev data)` — Standard Deviation

    σ = √(σ²)

```scheme
(require math.statistics)
(define data '(2 4 4 4 5 5 7 9))
(variance data)  ;; => 4.0
(std-dev data)   ;; => 2.0
(median data)    ;; => 4.5
```

### 4.4 Bivariate Statistics

#### `(covariance xs ys)` — Sample Covariance

    Cov(X, Y) = (1/n) · Σᵢ (xᵢ − x̄)(yᵢ − ȳ)

When n(xs) ≠ n(ys), uses the minimum length.

#### `(correlation xs ys)` — Pearson Correlation Coefficient

    r = Cov(X, Y) / (σ_X · σ_Y)

r ∈ [−1, 1]. Returns 0.0 if either variable has zero standard deviation.

```scheme
(correlation '(1 2 3 4 5) '(2 4 6 8 10))  ;; => 1.0  (perfect linear)
(correlation '(1 2 3 4 5) '(5 4 3 2 1))   ;; => -1.0 (perfect inverse)
```

### 4.5 Histograms

#### `(histogram data num-bins)` — Frequency Histogram

Partitions [min, max] into `num-bins` equal-width bins. Values at the maximum are
placed in the last bin (closed on both ends). Returns an association list:

    ((bin-center₀ . count₀) (bin-center₁ . count₁) ... )

where each `bin-center` is the midpoint of its bin.

#### `(bin-data data num-bins)` — Bin Assignment

Returns a list of 0-based bin indices, one per data point.

```scheme
(histogram '(1 1 2 3 3 3 4 5) 3)
;; => ((1.5 . 2) (3.0 . 4) (4.5 . 2))
```

### 4.6 Normalization and Summary

#### `(zscore data)` — Z-Score Normalization

    z_i = (x_i − x̄) / σ

Returns a list of z-scores. Returns all-zeros if σ = 0.

#### `(describe data)` — Summary Statistics

Returns an association list with keys: `count`, `mean`, `std`, `min`, `q1`,
`median`, `q3`, `max`.

```scheme
(describe '(2 4 4 4 5 5 7 9))
;; => ((count . 8) (mean . 5.0) (std . 2.0)
;;     (min . 2) (q1 . 4.0) (median . 4.5)
;;     (q3 . 5.5) (max . 9))

(cdr (assq 'mean (describe data)))   ;; => 5.0
```

---

## 5. Random Numbers

### 5.1 PRNG vs Quantum RNG

Eshkol provides two random number sources:

| Source     | Basis              | Speed | Use case                          |
|------------|--------------------|-------|-----------------------------------|
| PRNG       | drand48 (LCG-48)   | Fast  | Simulations, Monte Carlo, ML      |
| Quantum    | Hardware entropy   | Slow  | Cryptography, security, key gen   |

The PRNG is deterministic given a fixed seed, enabling reproducible experiments.
The quantum source reads from the OS entropy pool (getrandom/SecRandomCopyBytes) and
is non-deterministic.

### 5.2 Basic Primitives

```scheme
(require random)

;; PRNG
(random-float)            ;; uniform [0, 1)
(random-int lo hi)        ;; uniform integer in [lo, hi] inclusive
(random-bool)             ;; #t or #f with p=0.5
(random-choice lst)       ;; random element from list (or #f if empty)

;; Quantum
(qrandom)                 ;; uniform [0, 1) from hardware entropy
(qrandom-int lo hi)       ;; quantum integer in [lo, hi]
(qrandom-bool)            ;; quantum boolean
(qrandom-choice lst)      ;; quantum element selection
```

### 5.3 Continuous Distributions

#### Uniform

    X ~ Uniform(lo, hi):   x = lo + U·(hi − lo)

```scheme
(uniform 1.0 6.0)    ;; simulates continuous die
(quniform lo hi)     ;; quantum variant
```

#### Normal (Gaussian)

    X ~ N(μ, σ²)

**Box-Muller transform:** Given U₁, U₂ ~ Uniform(0,1):

    Z₁ = √(−2 ln U₁) · cos(2πU₂)   ~  N(0, 1)
    Z₂ = √(−2 ln U₁) · sin(2πU₂)   ~  N(0, 1)

`normal-pair` returns both variates (avoiding a wasted sample). `normal` returns one
scaled variate: x = μ + σ·Z₁.

```scheme
(normal 0.0 1.0)       ;; standard normal
(normal 100.0 15.0)    ;; IQ score distribution
(normal-pair)          ;; => (z1 z2) both standard normal
```

#### Exponential

    X ~ Exp(λ):   x = −(1/λ) · ln U

Models inter-arrival times in a Poisson process.

```scheme
(exponential 2.0)    ;; mean = 0.5 (λ=2)
```

### 5.4 Discrete Distributions

#### Poisson

    X ~ Poisson(λ):   P(X=k) = e^{-λ}·λ^k / k!

**Algorithm (Knuth):** Generate U₁, U₂, ... until product falls below e^{-λ},
counting the number of draws minus one.

```scheme
(poisson 3.5)    ;; expected value 3.5
```

#### Bernoulli

    X ~ Bernoulli(p):  P(X=1) = p,  P(X=0) = 1-p

Returns integer 1 or 0.

#### Geometric

    X ~ Geometric(p):  x = ⌊ln U / ln(1-p)⌋

Number of failures before first success in Bernoulli trials.

#### Binomial

    X ~ Binomial(n, p):  P(X=k) = C(n,k)·p^k·(1-p)^(n-k)

**Algorithm:** Simple trial method — repeat n Bernoulli(p) draws, count successes.
Efficient for small n (≤ a few hundred); for large n use the normal approximation.

```scheme
(binomial 100 0.5)   ;; ~50 with std ~5
```

### 5.5 Tensor and Vector Generation

```scheme
;; Tensors (homogeneous f64, hardware-accelerated)
(random-tensor dims)          ;; uniform [0,1), dims is a list
(random-normal-tensor dims)   ;; N(0,1)

;; Example: 3x4 random matrix
(random-tensor (list 3 4))
(random-normal-tensor (list 100 100))   ;; 100x100 weight matrix

;; Vectors (heterogeneous tagged values)
(random-vector n)                       ;; n uniform [0,1) floats
(random-uniform-vector n lo hi)         ;; n uniform [lo,hi] floats
(random-normal-vector n)                ;; n standard normal floats
```

### 5.6 Combinatorics

#### `(shuffle lst)` — Fisher-Yates Shuffle

Generates a uniformly random permutation in O(n) time. The standard Knuth/Durstenfeld
algorithm: for i from n-1 down to 1, swap element i with a random j ∈ [0, i].

```scheme
(shuffle '(1 2 3 4 5))   ;; => e.g. (3 1 5 2 4)
```

#### `(sample lst k)` — Sample Without Replacement

Returns k elements chosen uniformly without replacement. Implemented by shuffling then
taking the first k elements.

```scheme
(sample '(a b c d e) 3)   ;; => e.g. (c a e)
```

#### `(weighted-choice items weights)` — Weighted Sampling

Samples one item with probability proportional to its weight. Scans the cumulative
weight distribution with a single random draw. Returns `#f` if `items` is empty.

```scheme
(weighted-choice '(red green blue) '(0.5 0.3 0.2))
;; red 50%, green 30%, blue 20%
```

### 5.7 Seed Control

```scheme
(set-random-seed! seed)   ;; set drand48 seed (integer)
(current-time-seed)       ;; seconds since epoch (for time-based seed)
(randomize!)              ;; shorthand: (set-random-seed! (current-time-seed))
```

For reproducible experiments, always call `(set-random-seed! n)` before any use of
PRNG functions. The quantum functions are unaffected by the seed.

---

## 6. Code Examples

### 6.1 Special Functions — Computing Probabilities

```scheme
(require math.special)
(require math.constants)

;; Chi-squared CDF via incomplete gamma: P(X ≤ x | k dof) = γ(k/2, x/2) / Γ(k/2)
(define (chisq-cdf x k)
  (/ (gammainc-lower (* 0.5 k) (* 0.5 x))
     (gamma (* 0.5 k))))

(chisq-cdf 9.488 4)   ;; => ~0.95  (critical value for 4 dof at 5%)

;; Normal tail probability
(- 1.0 (normcdf 1.96))   ;; => ~0.025  (upper 2.5% tail)

;; Stirling's approximation check
(define (stirling n)
  (* (sqrt (* 2.0 pi n)) (expt (/ n e) n)))
(/ (factorial 20) (stirling 20.0))   ;; => ~1.008 (0.8% error at n=20)
```

### 6.2 ODE Solving — Lorenz System

```scheme
(require math.ode)

;; Lorenz strange attractor
;; d/dt [x y z] = [σ(y-x), x(ρ-z)-y, xy-βz]
(define sigma 10.0)
(define rho   28.0)
(define beta  (/ 8.0 3.0))

(define (lorenz t state)
  (let ((x (vref state 0))
        (y (vref state 1))
        (z (vref state 2)))
    (vector (* sigma (- y x))
            (- (* x (- rho z)) y)
            (- (* x y) (* beta z)))))

;; Solve with adaptive RK45 for precision in chaotic regime
(define traj
  (rk45 lorenz 0.0 (vector 0.1 0.0 0.0) 50.0 0.01 1e-9))

;; Extract final state
(define final-state (cadr (car (reverse traj))))
(vref final-state 0)   ;; x-coordinate at t=50
```

### 6.3 Statistical Analysis — Descriptive Statistics

```scheme
(require math.statistics)

(define measurements '(23.1 22.8 24.5 23.9 22.3 24.1 23.7 22.9 24.3 23.5))

(define stats (describe measurements))
(cdr (assq 'mean   stats))   ;; => 23.51
(cdr (assq 'std    stats))   ;; => 0.663...
(cdr (assq 'median stats))   ;; => 23.6

(iqr measurements)           ;; => Q3 - Q1

;; Pearson correlation between two variables
(define xs '(1 2 3 4 5 6 7 8))
(define ys '(2.1 3.9 6.2 7.8 10.1 11.9 14.2 15.8))
(correlation xs ys)          ;; => ~0.9998 (near-perfect linear)
```

### 6.4 Monte Carlo — Estimating π

```scheme
(require random)
(require math.constants)

(define (estimate-pi n-samples)
  (set-random-seed! 42)                   ;; reproducible
  (let loop ((i 0) (inside 0))
    (if (= i n-samples)
        (* 4.0 (/ inside n-samples))
        (let ((x (uniform -1.0 1.0))
              (y (uniform -1.0 1.0)))
          (loop (+ i 1)
                (if (< (+ (* x x) (* y y)) 1.0)
                    (+ inside 1)
                    inside))))))

(estimate-pi 1000000)     ;; => ~3.1415 ± 0.002
(approx= (estimate-pi 1000000) pi 0.01)  ;; => #t  (requires math.constants)
```

### 6.5 Distribution Sampling — Generating Training Data

```scheme
(require random)

;; Generate a linearly separable dataset
(define (make-cluster mu-x mu-y n)
  (map (lambda (_)
         (list (normal mu-x 0.5)
               (normal mu-y 0.5)))
       (make-list n 0)))

(define class-a (make-cluster 2.0  2.0  100))
(define class-b (make-cluster -2.0 -2.0 100))

;; Random weight initialization for a neural network layer
(define W (random-normal-tensor (list 256 128)))   ;; Glorot-like init
(define b (make-tensor (list 256) 0.0))            ;; zero bias
```

---

## 7. Performance Considerations

**Special functions:** All evaluations are O(1) amortized (fixed-degree polynomial
or continued fraction, bounded iteration). The gamma function Lanczos approximation
uses 9 floating-point divisions and one `expt`/`exp`; on modern hardware this runs in
tens of nanoseconds.

**ODE solvers:** Function evaluation count per integration step:
- Euler: 1 f-call
- Heun, Midpoint: 2 f-calls
- RK4: 4 f-calls
- RK45: 6 f-calls (but adaptive — fewer total steps for smooth problems)

For vector-valued ODEs, all vector arithmetic is realized with alloca-based heap
vectors and element-wise loops. For large state spaces (d > 1000), consider
wrapping tensor operations inside f rather than using the vector helpers.

**Adaptive vs fixed step:** RK45 is typically faster than RK4 at equivalent accuracy
because the error-controlled step size can be much larger over smooth solution
intervals. Reserve fixed-step methods for when the step size is externally mandated
(e.g., control system sampling rates, co-simulation interfaces).

**Backward Euler:** Each step incurs a Newton solve — up to 10 additional f-calls
for scalar problems. Use only when the problem is genuinely stiff (explicit stability
limit h < 10⁻³ or smaller). Otherwise RK4 or RK45 is faster per unit accuracy.

**Statistics:** All functions call `flatten-to-list` which traverses the input once.
Sorting (median, percentile, histogram) costs O(n log n). Everything else is O(n).
For large tensors, prefer passing tensors directly (avoids Scheme list allocation).

**Random generation:** `random-tensor` and `random-normal-tensor` are implemented
via the built-in `rand`/`randn` tensor primitives and benefit from SIMD vectorization.
For generating large batches of samples (> 10⁴), strongly prefer these over element-
by-element calls to `normal` or `uniform`.

**Quantum RNG:** Each call to `qrandom` or `qrandom-int` makes a system call. Batch
needs by generating a vector and iterating, or use `random-tensor` (PRNG-backed) where
quantum properties are not required.

---

## 8. See Also

| Topic                             | Reference                                      |
|-----------------------------------|------------------------------------------------|
| Linear algebra / tensors          | `docs/breakdown/TENSOR_SYSTEM.md`             |
| Machine learning utilities        | `lib/ml/`                                      |
| Built-in numeric types            | `docs/breakdown/NUMERIC_TOWER.md`             |
| Compiler numeric codegen          | `lib/backend/llvm_codegen.cpp` (ArithmeticCodegen) |
| BLAS/AMX tensor acceleration      | `lib/backend/blas_backend.cpp`                |
| Language reference                | `ESHKOL_V1_LANGUAGE_REFERENCE.md`             |

**External references:**

- Abramowitz, M. & Stegun, I.A. (1964). *Handbook of Mathematical Functions.* NIST.
- Press, W.H. et al. (2007). *Numerical Recipes, 3rd ed.* Cambridge University Press.
- Hairer, E., Norsett, S.P., & Wanner, G. (1993). *Solving ODEs I.* Springer.
- Cash, J.R. & Karp, A.H. (1990). ACM TOMS 16(3): 201–222. (RK45 coefficients)
- Lanczos, C. (1964). SIAM J. Numer. Anal. 1(1): 86–96. (Gamma approximation)
- CODATA 2019. *The 2018 CODATA Recommended Values of the Fundamental Physical Constants.*
