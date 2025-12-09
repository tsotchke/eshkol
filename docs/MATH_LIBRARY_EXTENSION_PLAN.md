# Eshkol Mathematics Library Extension Plan

## Executive Summary

This document provides a comprehensive analysis of Eshkol's current mathematical capabilities and a strategic plan for extending them. Eshkol possesses an extraordinarily powerful foundation for mathematical computing:

- **Full symbolic differentiation** at compile-time (AST → AST transformation)
- **Forward-mode autodiff** via dual numbers with nested gradient support
- **Reverse-mode autodiff** via computational graphs with tape recording
- **N-dimensional tensors** with broadcasting and element-wise operations
- **HoTT type system** with dependent types for dimension checking
- **Linear types** for resource management
- **Homoiconicity** enabling symbolic manipulation

---

## Part 1: Current Capabilities (Deep Analysis)

### 1.1 Symbolic Differentiation (`diff`)

**Implementation**: `lib/backend/llvm_codegen.cpp:14270-14718`

The `diff` operator performs compile-time symbolic differentiation via AST transformation:

```scheme
(diff (* x x) x)  ; Returns: (* 2 x)
(diff (sin x) x)  ; Returns: (cos x)
(diff (/ f g) x)  ; Returns: (/ (- (* f' g) (* f g')) (* g g))
```

**Supported rules**:
- Constant rule: `d/dx(c) = 0`
- Variable rule: `d/dx(x) = 1`, `d/dx(y) = 0`
- Sum/difference rules
- **Product rule**: `d/dx(f·g) = f'·g + f·g'`
- **Quotient rule**: `d/dx(f/g) = (f'·g - f·g') / g²`
- **Power rule**: `d/dx(f^n) = n·f^(n-1)·f'`
- **Chain rule** for: `sin`, `cos`, `exp`, `log`, `sqrt`, `pow`

**Algebraic simplifications**: `0*x → 0`, `1*x → x`, `x+0 → x`

### 1.2 Automatic Differentiation

#### Forward-Mode (Dual Numbers)

**Implementation**: `lib/backend/autodiff_codegen.cpp`

```scheme
(derivative f x)     ; First derivative at point x
(derivative f)       ; Returns derivative function (higher-order)
```

Dual numbers: `(a + bε)` where `ε² = 0`

Propagates through all arithmetic operations with chain rule:
- `sin(a + bε) = sin(a) + b·cos(a)·ε`
- `exp(a + bε) = exp(a) + b·exp(a)·ε`

#### Reverse-Mode (Computation Graph)

**Implementation**: `lib/backend/autodiff_codegen.cpp:1200-1698`

```scheme
(gradient f point)              ; ∇f at point
(jacobian f point)              ; J[i,j] = ∂fᵢ/∂xⱼ
(hessian f point)               ; H[i,j] = ∂²f/∂xᵢ∂xⱼ
```

**Tape-based recording**:
1. Forward pass: Build computation graph (AD nodes)
2. Backward pass: Propagate gradients via chain rule
3. Nested gradients: Tape stack with push/pop context

**Supported AD node types**:
- `AD_NODE_CONSTANT` (type=0)
- `AD_NODE_VARIABLE` (type=1)
- `AD_NODE_ADD` (type=2)
- `AD_NODE_SUB` (type=3)
- `AD_NODE_MUL` (type=4)
- `AD_NODE_DIV` (type=5)
- `AD_NODE_SIN` (type=6)
- `AD_NODE_COS` (type=7)

#### Vector Calculus Operators

```scheme
(divergence F point)            ; ∇·F = Σ ∂Fᵢ/∂xᵢ
(curl F point)                  ; ∇×F (3D only)
(laplacian f point)             ; ∇²f = Σ ∂²f/∂xᵢ²
(directional-derivative f p d)  ; Dᵥf = ∇f · v
```

### 1.3 Tensor Operations

**Implementation**: `lib/backend/tensor_codegen.cpp`

#### Creation
```scheme
(zeros d1 d2 ...)       ; Zero-filled tensor
(ones d1 d2 ...)        ; One-filled tensor
(eye n)                 ; n×n identity matrix
(arange start stop step); Range tensor
(linspace start stop n) ; Evenly spaced
```

#### Element Access
```scheme
(vref T i)              ; 1D access (AD-aware!)
(tensor-get T i j ...)  ; N-D access
(tensor-set T val i j); Mutation
```

#### Arithmetic (Element-wise with broadcasting)
```scheme
(tensor-add A B)
(tensor-sub A B)
(tensor-mul A B)        ; Element-wise
(tensor-div A B)
(tensor-dot A B)        ; Dot product / matmul
(matmul A B)            ; Matrix multiplication
```

#### Reductions
```scheme
(tensor-sum T)          ; Sum all elements
(tensor-mean T)         ; Mean of all elements
(tensor-reduce-all T f init)
(tensor-reduce T f init dim)
```

#### Shape Operations
```scheme
(tensor-shape T)        ; Get dimensions
(transpose T)           ; Transpose
(reshape T d1 d2 ...)   ; Reshape
(flatten T)             ; 1D view
```

### 1.4 Math Library (Pure Eshkol)

**File**: `lib/math.esk` (442 lines)

| Function | Description |
|----------|-------------|
| `det` | Determinant via LU decomposition with pivoting |
| `inv` | Matrix inverse via Gauss-Jordan elimination |
| `solve` | Linear system solver (Ax = b) |
| `cross` | 3D cross product |
| `dot` | Vector dot product |
| `normalize` | Unit vector |
| `power-iteration` | Dominant eigenvalue estimation |
| `integrate` | Simpson's rule numerical integration |
| `newton` | Newton-Raphson root finding |
| `variance`, `std`, `covariance` | Statistics |

### 1.5 Higher-Order Functions

**Files**: `lib/core/list/higher_order.esk`, `lib/core/functional/*.esk`

```scheme
(map1 f lst)            ; Map over single list
(map2 f lst1 lst2)      ; Map over two lists
(fold proc init lst)    ; Left fold
(fold-right proc init lst)
(any pred lst)          ; Existential
(every pred lst)        ; Universal
(compose f g)           ; Function composition
(curry f)               ; Currying
(identity x)
(constantly x)
```

---

## Part 2: What Can Be Implemented in Pure Eshkol

### 2.1 Numerical Methods (100% Pure Eshkol)

These require only arithmetic, closures, and recursion:

| Category | Functions |
|----------|-----------|
| **Integration** | Gaussian quadrature, Romberg, adaptive Simpson, Monte Carlo |
| **Root Finding** | Bisection, secant method, Brent's method, fixed-point iteration |
| **Optimization** | Gradient descent, BFGS, conjugate gradient, Nelder-Mead |
| **Interpolation** | Lagrange, Newton divided differences, splines |
| **ODE Solvers** | Euler, RK4, RK45 (adaptive), multistep methods |

**Example** (already working in tests):
```scheme
;; Newton-Raphson with autodiff
(define (newton-solve f x0 iterations)
  (define (iterate x n)
    (if (= n 0) x
        (iterate (- x (/ (f x) (derivative f x))) (- n 1))))
  (iterate x0 iterations))

;; RK4 ODE solver
(define (rk4-step f x y h)
  (let* ((k1 (* h (f x y)))
         (k2 (* h (f (+ x (/ h 2)) (+ y (/ k1 2)))))
         (k3 (* h (f (+ x (/ h 2)) (+ y (/ k2 2)))))
         (k4 (* h (f (+ x h) (+ y k3)))))
    (+ y (/ (+ k1 (* 2 k2) (* 2 k3) k4) 6))))
```

### 2.2 Linear Algebra (95% Pure Eshkol)

Most linear algebra can be pure Eshkol using existing tensor operations:

| Function | Pure Eshkol? | Notes |
|----------|--------------|-------|
| QR decomposition | Yes | Gram-Schmidt or Householder |
| LU decomposition | Yes | Already implemented |
| Cholesky | Yes | For positive definite matrices |
| SVD | Mostly | May need eigenvalue iterations |
| Eigenvalues (power iteration) | Yes | Already implemented |
| Eigenvalues (QR algorithm) | Yes | Iterative |
| Matrix norms (Frobenius, L1, L∞) | Yes | |
| Condition number | Yes | Via SVD |
| Pseudoinverse | Yes | Via SVD |
| Least squares | Yes | Via QR or normal equations |

### 2.3 Probability & Statistics (100% Pure Eshkol)

| Category | Functions |
|----------|-----------|
| **Descriptive** | median, mode, percentiles, skewness, kurtosis, correlation |
| **Distributions** | Normal, Poisson, Binomial, Exponential (PDFs, CDFs) |
| **Sampling** | LCG, Mersenne Twister, Box-Muller transform |
| **Hypothesis Testing** | t-test, chi-square, ANOVA |
| **Regression** | Linear, polynomial, logistic |

**Example** (Box-Muller for normal distribution):
```scheme
(define (box-muller rng)
  (let* ((u1 (rng))
         (u2 (rng))
         (r (sqrt (* -2 (log u1))))
         (theta (* 2 pi u2)))
    (values (* r (cos theta))
            (* r (sin theta)))))
```

### 2.4 Special Functions (Mostly Pure Eshkol)

| Function | Pure Eshkol? | Implementation |
|----------|--------------|----------------|
| Gamma function | Yes | Lanczos approximation |
| Beta function | Yes | Via Gamma |
| Error function (erf) | Yes | Taylor series |
| Bessel functions | Yes | Series expansion |
| Legendre polynomials | Yes | Recurrence relation |
| Chebyshev polynomials | Yes | Recurrence relation |
| Elliptic integrals | Yes | Carlson's algorithm |

### 2.5 Abstract Algebra (100% Pure Eshkol)

Perfect for Eshkol's homoiconic nature:

| Structure | Implementation |
|-----------|----------------|
| Polynomials | List of coefficients, symbolic operations |
| Groups | Closure tables, generators |
| Rings/Fields | Quotient structures |
| GCD/LCM | Extended Euclidean algorithm |
| Modular arithmetic | Native with `modulo` |
| Permutations | List-based, cycle notation |

### 2.6 Calculus Extensions (Pure Eshkol)

| Feature | Implementation |
|---------|----------------|
| Taylor series | Via `diff` and factorial |
| Fourier series coefficients | Via integration |
| Partial derivatives | Via `diff` with variable substitution |
| Multiple integrals | Nested integration |
| Path integrals | Parameterized integration |

---

## Part 3: What Requires C/C++ Extensions

### 3.1 Performance-Critical Operations

| Operation | Why C? | Priority |
|-----------|--------|----------|
| FFT | O(n log n) vs O(n²), SIMD | High |
| BLAS operations | Hardware optimization, cache efficiency | High |
| Sparse matrix storage | Memory layout efficiency | Medium |
| Hardware RNG | True hardware entropy (RDRAND, /dev/urandom) | Medium |

### 3.1.1 Existing: Quantum-Inspired RNG (`lib/quantum/quantum_rng.c`)

Eshkol already has a sophisticated **quantum-inspired random number generator**:

```scheme
(quantum-random)              ; Double in [0, 1)
(quantum-random-int)          ; Full 64-bit unsigned integer
(quantum-random-range 1 6)    ; Integer in [min, max] (dice roll)
```

**Implementation features**:
- **8-qubit circuit simulation** with Hadamard gates, phase gates, entanglement
- **Multiple entropy sources**: `gettimeofday`, PID, stack addresses, `rdtsc` (CPU cycles)
- **Quantum mixing**: splitmix64, Pauli gates (X, Y, Z), physical constants
- **State entanglement** via `qrng_entangle_states()`
- **Quantum measurement collapse** via `qrng_measure_state()`

This provides high-quality randomness suitable for Monte Carlo simulations and most probabilistic algorithms. For cryptographic applications or scenarios requiring true hardware entropy, a dedicated hardware RNG backend (RDRAND, /dev/urandom) should still be added.

### 3.2 External Library Integration

| Library | Purpose | Integration |
|---------|---------|-------------|
| FFTW | Fast Fourier Transform | FFI binding |
| LAPACK | Dense linear algebra | Optional backend |
| OpenBLAS | Optimized BLAS | Optional backend |
| GMP | Arbitrary precision | For exact arithmetic |

### 3.3 Proposed C Extensions

```c
// In runtime/math_extensions.c

// Fast Fourier Transform (Cooley-Tukey)
void eshkol_fft(double* real, double* imag, size_t n);
void eshkol_ifft(double* real, double* imag, size_t n);

// Optimized matrix operations
void eshkol_matmul_blocked(double* A, double* B, double* C,
                           size_t M, size_t K, size_t N);

// Random number generation with better period
typedef struct { uint64_t state[4]; } xoshiro256_state;
double eshkol_random_double(xoshiro256_state* state);
```

---

## Part 4: Architectural Principles

### 4.1 Module Organization

```
lib/
├── math.esk              # Core numerics (existing)
├── math/
│   ├── linalg.esk       # Linear algebra
│   ├── calculus.esk     # Integration, ODE
│   ├── probability.esk  # Distributions, sampling
│   ├── statistics.esk   # Descriptive, inference
│   ├── special.esk      # Special functions
│   ├── optimize.esk     # Optimization algorithms
│   ├── polynomial.esk   # Polynomial algebra
│   ├── complex.esk      # Complex arithmetic
│   ├── number-theory.esk # GCD, primes, modular
│   └── geometry.esk     # Computational geometry
```

### 4.2 Design Principles

#### 4.2.1 Composability via Higher-Order Functions

All algorithms should accept function parameters:

```scheme
;; Good: Accepts any objective function
(define (gradient-descent f initial-point learning-rate iterations)
  ...)

;; Good: Accepts any derivative (symbolic or numeric)
(define (newton-raphson f df x0 tolerance)
  ...)
```

#### 4.2.2 Leverage Autodiff Everywhere

Replace manual derivative implementations:

```scheme
;; Bad: Manual derivative
(define (newton-raphson-bad f df x0 tol)
  (iterate x0 (/ (f x0) (df x0)) tol))

;; Good: Use autodiff
(define (newton-raphson f x0 tol)
  (iterate x0 (/ (f x0) (derivative f x0)) tol))
```

#### 4.2.3 Type Safety via Dependent Types

Use HoTT type system for dimension checking:

```scheme
;; Matrix multiplication with dimension checking
(: matmul (-> (Matrix m k) (Matrix k n) (Matrix m n)))
(define (matmul A B) ...)

;; Vector operations with length constraints
(: dot (-> (Vector n) (Vector n) Real))
(define (dot u v) ...)
```

#### 4.2.4 Lazy Evaluation for Large Data

Use closures for deferred computation:

```scheme
;; Lazy infinite sequence
(define (naturals)
  (define (from n)
    (cons n (lambda () (from (+ n 1)))))
  (from 0))

;; Lazy matrix operations
(define (lazy-matmul A B)
  (lambda (i j)
    (sum-row-col A B i j)))
```

#### 4.2.5 Numerical Stability

Always prefer numerically stable algorithms:

```scheme
;; Bad: Naive variance (catastrophic cancellation)
(define (variance-bad xs)
  (let ((mean (/ (sum xs) (length xs))))
    (/ (sum (map (lambda (x) (sqr (- x mean))) xs))
       (length xs))))

;; Good: Welford's online algorithm
(define (variance-welford xs)
  (let loop ((xs xs) (n 0) (mean 0.0) (M2 0.0))
    (if (null? xs)
        (/ M2 n)
        (let* ((x (car xs))
               (n1 (+ n 1))
               (delta (- x mean))
               (mean1 (+ mean (/ delta n1)))
               (delta2 (- x mean1))
               (M2-1 (+ M2 (* delta delta2))))
          (loop (cdr xs) n1 mean1 M2-1)))))
```

### 4.3 Error Handling

Use option types for operations that can fail:

```scheme
;; Return (some result) or none for singular matrix
(define (matrix-inverse M)
  (let ((det (determinant M)))
    (if (< (abs det) epsilon)
        'none
        (some (compute-inverse M det)))))
```

---

## Part 5: Initial Release Priorities

### 5.1 Tier 1: Core Mathematical Functions (Must Have)

| Module | Functions | Effort |
|--------|-----------|--------|
| `linalg.esk` | QR, Cholesky, SVD, eigenvalues, pseudoinverse | Medium |
| `calculus.esk` | Gaussian quadrature, adaptive integration, RK45 | Medium |
| `optimize.esk` | Gradient descent variants (SGD, Adam, momentum), BFGS | Medium |
| `special.esk` | Gamma, Beta, erf, Bessel J₀/J₁ | Medium |
| `probability.esk` | Normal, Poisson, Binomial, Exponential distributions | Low |
| `statistics.esk` | Correlation, regression, percentiles | Low |

### 5.2 Tier 2: Extended Capabilities (Should Have)

| Module | Functions | Effort |
|--------|-----------|--------|
| `complex.esk` | Full complex arithmetic, polar form | Low |
| `polynomial.esk` | Evaluation, roots, GCD, factorization | Medium |
| `number-theory.esk` | Prime testing, factorization, modular exp | Low |
| `geometry.esk` | Convex hull, Voronoi, triangulation | High |
| `fft.c` | FFT/IFFT (C extension) | Medium |

### 5.3 Tier 3: Advanced Features (Nice to Have)

| Module | Functions | Effort |
|--------|-----------|--------|
| `measure.esk` | Lebesgue integration, measure spaces | High |
| `category.esk` | Functors, monads, natural transformations | Medium |
| `topology.esk` | Simplicial complexes, homology | High |
| `diffgeom.esk` | Manifolds, Riemannian metrics | High |

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

1. **Reorganize math library** into modular structure
2. **Implement core linear algebra**: QR, Cholesky, SVD via pure Eshkol
3. **Add Gaussian quadrature** for improved integration
4. **Create test suite** with numerical accuracy benchmarks

### Phase 2: Numerical Methods (Weeks 3-4)

1. **Optimization suite**: SGD, Adam, BFGS, Nelder-Mead
2. **ODE solvers**: Adaptive RK45, multistep methods
3. **Root finding**: Brent's method, polynomial roots
4. **Interpolation**: Cubic splines, rational interpolation

### Phase 3: Probability & Statistics (Weeks 5-6)

1. **Distribution functions**: PDF, CDF, quantile, sampling
2. **Descriptive statistics**: Complete suite
3. **Statistical tests**: t-test, chi-square, correlation
4. **Regression**: Linear, polynomial, logistic

### Phase 4: Special Functions & Number Theory (Weeks 7-8)

1. **Special functions**: Gamma, Beta, Bessel, elliptic
2. **Complex arithmetic**: Full implementation
3. **Polynomial algebra**: Complete module
4. **Number theory**: Primes, factorization, modular arithmetic

### Phase 5: C Extensions & Performance (Weeks 9-10)

1. **FFT implementation** in C
2. **Optimized BLAS** bindings (optional)
3. **Sparse matrix** support
4. **Benchmarking** against NumPy/Julia

---

## Part 7: Integration with Existing Features

### 7.1 Autodiff Integration

Every numerical algorithm should work seamlessly with autodiff:

```scheme
;; Optimize with automatic gradients
(define (optimize-with-autodiff objective x0)
  (gradient-descent objective x0 0.01 1000))

;; Sensitivity analysis
(define (parameter-sensitivity model params)
  (jacobian model params))
```

### 7.2 Type System Integration

Use HoTT types for mathematical structures:

```scheme
;; Dependent type for matrices
(define-type (Matrix m n)
  (vector-of (vector-of Real n) m))

;; Proposition: matrix is positive definite
(define-type (PositiveDefinite M)
  (forall (v : (Vector n))
    (> (quadratic-form M v) 0)))
```

### 7.3 Neural Network Integration

The existing NN test shows integration with autodiff:

```scheme
(define (nn-forward input)
  (let* ((h1 (sigmoid (+ (* w1 x1) (* w2 x2) b1)))
         (h2 (sigmoid (+ (* w3 x1) (* w4 x2) b2)))
         (out (sigmoid (+ (* w5 h1) (* w6 h2) b3))))
    out))

;; Automatic backpropagation
(define gradients (gradient nn-forward params))
```

---

## Appendix A: Function Reference (Proposed)

### Linear Algebra (`math/linalg.esk`)

```scheme
;; Decompositions
(qr-decomposition M)          ; Returns (Q R)
(lu-decomposition M)          ; Returns (L U P)
(cholesky M)                  ; Returns L where M = L·Lᵀ
(svd M)                       ; Returns (U Σ Vᵀ)
(eigendecomposition M)        ; Returns (eigenvalues eigenvectors)

;; Solvers
(solve-triangular L b)        ; Forward/back substitution
(least-squares A b)           ; Minimize ||Ax - b||²
(ridge-regression A b lambda) ; Tikhonov regularization

;; Matrix properties
(rank M)                      ; Numerical rank
(condition-number M)          ; κ(M) = ||M|| · ||M⁻¹||
(nullspace M)                 ; Basis for null space
(range M)                     ; Basis for column space
```

### Calculus (`math/calculus.esk`)

```scheme
;; Integration
(integrate-gauss f a b n)     ; Gauss-Legendre quadrature
(integrate-adaptive f a b tol); Adaptive Simpson
(integrate-monte-carlo f bounds samples)
(integrate-2d f x-range y-range) ; Double integral

;; ODE Solvers
(ode-solve f t-span y0 method); Generic solver
(euler f t-span y0 dt)        ; Euler method
(rk4 f t-span y0 dt)          ; Runge-Kutta 4
(rk45 f t-span y0 tol)        ; Adaptive RK45
(bdf f t-span y0)             ; Backward differentiation

;; PDE (simple cases)
(poisson-solve f boundary grid)
(heat-equation u0 alpha t grid)
```

### Optimization (`math/optimize.esk`)

```scheme
;; Unconstrained
(gradient-descent f x0 options)
(sgd f data x0 options)       ; Stochastic GD
(adam f data x0 options)      ; Adam optimizer
(bfgs f x0 options)           ; Quasi-Newton
(lbfgs f x0 options)          ; Limited-memory BFGS
(nelder-mead f x0 options)    ; Simplex method
(conjugate-gradient f x0)     ; CG method

;; Constrained
(lagrange-multipliers f g x0) ; Equality constraints
(penalty-method f g x0)       ; Inequality constraints

;; Line search
(backtracking f x d alpha)
(wolfe-line-search f x d)
```

### Probability (`math/probability.esk`)

```scheme
;; Distributions (pdf, cdf, quantile, sample)
(normal-pdf x mu sigma)
(normal-cdf x mu sigma)
(normal-quantile p mu sigma)
(normal-sample rng mu sigma)

(poisson-pmf k lambda)
(binomial-pmf k n p)
(exponential-pdf x lambda)
(gamma-pdf x alpha beta)
(beta-pdf x alpha beta)
(chi-squared-pdf x df)
(t-pdf x df)
(f-pdf x df1 df2)

;; Random number generation
(make-rng seed)               ; Create RNG
(rng-uniform rng)             ; U(0,1)
(rng-normal rng)              ; N(0,1)
(rng-exponential rng lambda)
```

### Special Functions (`math/special.esk`)

```scheme
;; Gamma family
(gamma x)                     ; Γ(x)
(log-gamma x)                 ; ln(Γ(x))
(digamma x)                   ; ψ(x) = d/dx ln(Γ(x))
(beta a b)                    ; B(a,b) = Γ(a)Γ(b)/Γ(a+b)
(incomplete-gamma a x)        ; γ(a,x)
(incomplete-beta a b x)       ; I_x(a,b)

;; Error functions
(erf x)                       ; Error function
(erfc x)                      ; Complementary error function
(erfinv x)                    ; Inverse error function

;; Bessel functions
(bessel-j n x)                ; Bessel J_n(x)
(bessel-y n x)                ; Bessel Y_n(x)
(bessel-i n x)                ; Modified I_n(x)
(bessel-k n x)                ; Modified K_n(x)

;; Orthogonal polynomials
(legendre n x)                ; Legendre P_n(x)
(chebyshev-t n x)             ; Chebyshev T_n(x)
(chebyshev-u n x)             ; Chebyshev U_n(x)
(hermite n x)                 ; Hermite H_n(x)
(laguerre n x)                ; Laguerre L_n(x)

;; Elliptic integrals
(elliptic-k k)                ; Complete elliptic K(k)
(elliptic-e k)                ; Complete elliptic E(k)
(elliptic-f phi k)            ; Incomplete F(φ,k)
```

---

## Appendix B: Test Coverage Requirements

Every function must have:

1. **Unit tests** with known values
2. **Edge cases** (zero, infinity, singularities)
3. **Numerical accuracy** tests (compare to high-precision reference)
4. **Autodiff compatibility** tests
5. **Performance benchmarks**

Example test structure:

```scheme
;; tests/math/linalg_test.esk

;; Test QR decomposition
(define (test-qr)
  (let* ((A (matrix [[1 2] [3 4] [5 6]]))
         ((Q R) (qr-decomposition A)))
    ;; Q is orthogonal
    (check-approx "QᵀQ = I" (matmul (transpose Q) Q) (eye 2) 1e-10)
    ;; A = QR
    (check-approx "A = QR" (matmul Q R) A 1e-10)
    ;; R is upper triangular
    (check "R upper triangular" (upper-triangular? R))))
```

---

## Conclusion

Eshkol possesses an exceptional foundation for mathematical computing. The combination of:

1. **Symbolic differentiation** for algebraic manipulation
2. **Automatic differentiation** for gradient-based methods
3. **HoTT type system** for mathematical rigor
4. **Homoiconicity** for metaprogramming
5. **Pure functional core** for correctness

...makes it uniquely suited for implementing a comprehensive mathematics library that rivals systems like Mathematica, MATLAB, or Julia—while maintaining the elegance of Scheme and the power of dependent types.

The majority of mathematical functions can be implemented in **pure Eshkol**, with C extensions only needed for performance-critical operations like FFT. This ensures portability, correctness through type checking, and seamless integration with autodiff.
