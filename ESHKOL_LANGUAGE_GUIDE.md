# Eshkol: A Scientific Computing Language with First-Class Automatic Differentiation

<p align="center">
<em>Scheme-like syntax meets LLVM performance with built-in calculus, GPU acceleration, and a consciousness engine</em>
</p>

---

## Executive Summary

**Eshkol** is a modern functional programming language designed from the ground up for scientific computing and machine learning. It combines the elegance of Scheme with the performance of native code compilation via LLVM, featuring **automatic differentiation as a first-class language primitive**, **GPU-accelerated tensor operations**, and an **active inference consciousness engine**.

### Key Differentiators

| Feature | Eshkol | Python/NumPy | Julia | Scheme |
|---------|--------|--------------|-------|--------|
| Native Compilation | LLVM | Interpreted | JIT | Varies |
| Built-in Autodiff | **First-class** | Library (JAX) | Library | None |
| Vector Calculus | **Built-in operators** | Library | Library | None |
| Garbage Collection | **None (Arena)** | Yes | Yes | Yes |
| Homoiconicity | **Yes** | No | No | Yes |
| REPL + JIT | **Yes** | Yes | Yes | Yes |
| GPU Acceleration | **Auto-dispatch** | Library (CuPy) | Library | None |
| Exact Arithmetic | **Full numeric tower** | No | Partial | Yes |
| Consciousness Engine | **Built-in** | No | No | No |
| Parallel Primitives | **Built-in** | Library | Built-in | None |
| Signal Processing | **Built-in** | Library (SciPy) | Library | None |
| WebAssembly | **Native compilation** | Pyodide | No | No |

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Language Overview](#language-overview)
3. [Core Features](#core-features)
4. [Automatic Differentiation](#automatic-differentiation)
5. [Tensor & Matrix Operations](#tensor--matrix-operations)
6. [Exact Arithmetic](#exact-arithmetic)
7. [Complex Numbers](#complex-numbers)
8. [Continuations & Exception Handling](#continuations--exception-handling)
9. [Parallel Primitives](#parallel-primitives)
10. [GPU Acceleration](#gpu-acceleration)
11. [Consciousness Engine](#consciousness-engine)
12. [Signal Processing](#signal-processing)
13. [Web Platform](#web-platform)
14. [Higher-Order Functions](#higher-order-functions)
15. [Memory Model](#memory-model)
16. [C Interoperability](#c-interoperability)
17. [Interactive REPL](#interactive-repl)
18. [Examples](#examples)
19. [Technical Architecture](#technical-architecture)
20. [Why Eshkol?](#why-eshkol)

---

## Getting Started

### Installation

```bash
# Clone and build
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
mkdir build && cd build
cmake .. && make -j8

# Run a program
./eshkol-run examples/fibonacci.esk

# Compile to binary
./eshkol-run program.esk -o program
./program

# Start interactive REPL
./eshkol-repl
```

### Hello World

```scheme
;; hello.esk
(display "Hello, Eshkol!")
(newline)
```

### First Derivative

```scheme
;; Compute derivative of f(x) = x^2 at x = 3
(define f (lambda (x) (* x x)))
(display (derivative f 3.0))  ;; -> 6.0
```

---

## Language Overview

Eshkol uses **S-expression syntax** familiar to Lisp/Scheme programmers:

```scheme
;; Variables
(define x 42)
(define pi 3.14159)
(define name "Eshkol")

;; Functions
(define (square x) (* x x))
(define (add a b) (+ a b))

;; Lambda expressions
(define double (lambda (x) (* 2 x)))

;; Lists
(define nums (list 1 2 3 4 5))
(car nums)   ;; -> 1
(cdr nums)   ;; -> (2 3 4 5)

;; Conditionals
(if (> x 0) "positive" "non-positive")

(cond ((< x 0) "negative")
      ((= x 0) "zero")
      (else "positive"))

;; Let bindings
(let ((x 10) (y 20))
  (+ x y))  ;; -> 30

;; Iteration
(do ((i 0 (+ i 1)))
    ((= i 10) 'done)
  (display i))
```

---

## Core Features

### Data Types

| Type | Example | Description |
|------|---------|-------------|
| Integer | `42`, `-17` | 64-bit signed (exact) |
| Bignum | `99999999999999999999` | Arbitrary precision (exact) |
| Rational | `1/3`, `355/113` | Exact fractions |
| Float | `3.14`, `2.5e-10` | IEEE 754 double (inexact) |
| Complex | `3+4i` | Complex numbers |
| Boolean | `#t`, `#f` | True/False |
| Character | `#\a`, `#\newline` | Unicode |
| String | `"hello"` | UTF-8 |
| Symbol | `'foo` | Interned identifier |
| List | `(list 1 2 3)` | Linked cons cells |
| Vector | `(vector 1 2 3)` | Indexed array (heterogeneous) |
| Tensor | `#(1.0 2.0 3.0)` | N-dimensional array (homogeneous doubles) |
| Lambda | `(lambda (x) x)` | First-class function |
| Closure | Captured lambda | Function + environment |
| Port | `(open-input-file ...)` | I/O channel |
| Logic Variable | `?x` | Unification variable |
| Substitution | `(make-substitution)` | Logic binding map |
| Knowledge Base | `(make-kb)` | Fact store |
| Factor Graph | `(make-factor-graph n)` | Probabilistic graphical model |
| Workspace | `(make-workspace)` | Global workspace (consciousness) |

### 555+ Built-in Functions

Eshkol v1.1 ships with over 555 built-in functions spanning arithmetic, math, strings, lists, vectors, tensors, automatic differentiation, vector calculus, exact arithmetic, complex numbers, continuations, parallel primitives, GPU operations, signal processing, logic programming, active inference, and web platform APIs.

**Arithmetic:** `+`, `-`, `*`, `/`, `abs`, `floor`, `ceiling`, `round`, `truncate`, `modulo`, `remainder`, `quotient`, `gcd`, `lcm`, `min`, `max`, `expt`, `exact->inexact`, `inexact->exact`

**Math:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log10`, `sqrt`, `pow`

**Comparison:** `<`, `>`, `=`, `<=`, `>=`, `eq?`, `eqv?`, `equal?`

**Lists:** `cons`, `car`, `cdr`, `list`, `append`, `reverse`, `length`, `map`, `filter`, `fold`, `member`, `assoc`, + 30 compound accessors (`caar`, `cadr`, `caddr`, ...)

**Strings:** `string-append`, `substring`, `string-length`, `string->list`, `number->string`, `string->number`, `string-for-each`, `string-map`, `string-fill!`

**Vectors:** `vector`, `make-vector`, `vector-ref`, `vector-set!`, `vector-length`, `vector-for-each`, `vector-map`

**I/O:** `display`, `newline`, `printf`, `open-input-file`, `open-output-file`, `read-line`, `write-string`, `read-char`, `peek-char`, `close-port`

**Exact Arithmetic:** `exact?`, `inexact?`, `exact->inexact`, `inexact->exact`, `numerator`, `denominator`, `rationalize`

**Complex:** `make-rectangular`, `make-polar`, `real-part`, `imag-part`, `magnitude`, `angle`

**Continuations:** `call/cc`, `call-with-current-continuation`, `dynamic-wind`, `guard`, `raise`, `with-exception-handler`, `values`, `call-with-values`

**Parallel:** `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`, `future`, `force`, `parallel-execute`

**GPU:** `gpu-matmul`, `gpu-elementwise`, `gpu-reduce`, `gpu-softmax`, `gpu-transpose`

**Signal Processing:** `fft`, `ifft`, `convolve`, `fir-filter`, `iir-filter`, `butterworth-lowpass`, `window-hamming`, `window-hann`

**Consciousness Engine:** `unify`, `walk`, `make-substitution`, `make-fact`, `make-kb`, `kb-assert!`, `kb-query`, `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`, `free-energy`, `expected-free-energy`, `make-workspace`, `ws-register!`, `ws-step!`

**Web Platform:** `web-create-element`, `web-set-text`, `web-add-event-listener`, `web-canvas-*`

---

## Automatic Differentiation

### The Killer Feature

Eshkol provides **three modes of automatic differentiation** as built-in language primitives:

#### 1. Symbolic Differentiation (`diff`)

Compile-time expression transformation:

```scheme
(diff (* x x) x)           ;; -> (* 2 x)
(diff (sin (* x x)) x)     ;; -> (* (cos (* x x)) (* 2 x))
(diff (+ (* x x) (* 3 x)) x)  ;; -> (+ (* 2 x) 3)
```

#### 2. Forward-Mode AD (`derivative`)

Numerical derivatives using dual numbers:

```scheme
(define f (lambda (x) (* x x x)))  ;; f(x) = x^3
(derivative f 2.0)                  ;; -> 12.0 (f'(x) = 3x^2)

(define g (lambda (x) (sin (* x x))))
(derivative g 1.0)                  ;; -> 1.0806... (2x*cos(x^2))
```

#### 3. Reverse-Mode AD (`gradient`, `jacobian`, `hessian`)

Computation graph-based differentiation for multivariate functions:

```scheme
;; Gradient of f(x,y) = x^2 + y^2
(define f (lambda (v)
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1)))))

(gradient f (vector 3.0 4.0))  ;; -> #(6.0 8.0)

;; Jacobian matrix for vector-valued functions
(define polar->cartesian (lambda (v)
  (let ((r (vref v 0)) (theta (vref v 1)))
    (vector (* r (cos theta))
            (* r (sin theta))))))

(jacobian polar->cartesian (vector 1.0 0.0))
;; -> #((1.0 0.0) (0.0 1.0))

;; Hessian matrix (second derivatives)
(hessian f (vector 1.0 1.0))  ;; -> #((2.0 0.0) (0.0 2.0))
```

### Vector Calculus Operators

Built-in operators for physics and engineering:

```scheme
;; Divergence: div(F)
(define F (lambda (v) (vector (* 2 (vref v 0)) (* 3 (vref v 1)))))
(divergence F (vector 1.0 1.0))  ;; -> 5.0

;; Curl: curl(F) (3D only)
(define rotating (lambda (v)
  (vector (- 0 (vref v 1)) (vref v 0) 0.0)))
(curl rotating (vector 1.0 1.0 0.0))  ;; -> #(0.0 0.0 2.0)

;; Laplacian: laplacian(f)
(define harmonic (lambda (v)
  (- (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1)))))
(laplacian harmonic (vector 1.0 1.0))  ;; -> 0.0 (it's harmonic!)

;; Directional derivative: D_u f
(directional-derivative f (vector 3.0 4.0) (vector 1.0 0.0))  ;; -> 6.0
```

---

## Tensor & Matrix Operations

### Creation

```scheme
(vector 1.0 2.0 3.0)           ;; 1D vector
#(1.0 2.0 3.0)                 ;; tensor literal
(zeros 3 4)                     ;; 3x4 matrix of zeros
(ones 2 3)                      ;; 2x3 matrix of ones
(eye 3)                         ;; 3x3 identity matrix
(arange 10)                     ;; #(0 1 2 3 4 5 6 7 8 9)
(linspace 0 1 5)                ;; #(0.0 0.25 0.5 0.75 1.0)
(reshape (arange 6) 2 3)        ;; 2x3 matrix
(make-tensor (list 4096 4096) 0.0)  ;; 4096x4096 matrix filled with 0.0
(rand 100 100)                  ;; 100x100 random matrix
```

### Operations

```scheme
;; Element-wise arithmetic
(tensor-add v1 v2)
(tensor-sub v1 v2)
(tensor-mul v1 v2)
(tensor-div v1 v2)

;; Linear algebra
(tensor-dot v1 v2)              ;; Dot product
(matmul A B)                    ;; Matrix multiplication (auto-dispatches SIMD/BLAS/GPU)
(transpose M)                   ;; Transpose
(norm v)                        ;; L2 norm
(trace M)                       ;; Sum of diagonal
(outer u v)                     ;; Outer product

;; Reductions
(tensor-sum v)                  ;; Sum all elements
(tensor-mean v)                 ;; Mean value
(tensor-reduce v + 0)           ;; Custom reduction
(tensor-reduce-all M * 1)       ;; Reduce entire tensor

;; Shape manipulation
(flatten M)                     ;; Flatten to 1D
(reshape v 2 3)                 ;; Reshape to 2x3
(tensor-shape M)                ;; Get dimensions
```

### Example: Neural Network Layer

```scheme
(define input (vector 1.0 0.5 -0.5))
(define weights (vector 0.5 0.3 0.2))
(define bias 0.1)

;; Linear layer: y = Wx + b
(define output (+ (tensor-dot input weights) bias))
(display output)  ;; -> 0.65
```

---

## Exact Arithmetic

Eshkol v1.1 implements the full R7RS numeric tower with arbitrary precision integers (bignums) and exact rational numbers. Exact arithmetic preserves precision through all operations -- no floating-point rounding errors.

### Bignums (Arbitrary Precision Integers)

```scheme
;; Bignums are created automatically when values exceed 64-bit range
(define huge (expt 2 256))
;; -> 115792089237316195423570985008687907853269984665640564039457584007913129639936

(define factorial-100
  (fold * 1 (iota-from 100 1)))  ;; exact 100!

;; All arithmetic works seamlessly
(+ (expt 10 50) 1)
(* (expt 2 128) (expt 3 80))
(gcd (expt 2 100) (expt 6 50))

;; Comparisons are exact
(> (expt 2 256) (expt 10 77))  ;; -> #f
```

### Rational Numbers

```scheme
;; Rational literals with / syntax
(define half 1/2)
(define third 1/3)

;; Arithmetic preserves exactness
(+ 1/3 1/6)           ;; -> 1/2 (exact)
(* 2/3 3/4)           ;; -> 1/2 (exact)
(- 7/8 3/8)           ;; -> 1/2 (exact)

;; Rationals auto-reduce
(+ 1/4 1/4)           ;; -> 1/2 (not 2/4)

;; Access components
(numerator 3/7)        ;; -> 3
(denominator 3/7)      ;; -> 7

;; Mixed exact/inexact: result is inexact (R7RS rule)
(+ 1/3 0.5)           ;; -> 0.8333... (inexact)
```

### Exactness Predicates and Conversion

```scheme
;; Predicates
(exact? 42)            ;; -> #t
(exact? 1/3)           ;; -> #t
(exact? 3.14)          ;; -> #f
(inexact? 3.14)        ;; -> #t

;; Conversion
(exact->inexact 1/3)   ;; -> 0.3333333333333333
(inexact->exact 0.5)   ;; -> 1/2
(inexact->exact 0.1)   ;; -> 3602879701896397/36028797018963968

;; number->string / string->number with bignums
(number->string (expt 2 128))
(string->number "99999999999999999999999")  ;; -> bignum
```

---

## Complex Numbers

Eshkol supports complex numbers as a first-class numeric type, with heap-allocated `{real:f64, imag:f64}` representation and Smith's formula for numerically stable division.

### Creation

```scheme
;; Rectangular form
(make-rectangular 3.0 4.0)     ;; -> 3.0+4.0i

;; Polar form
(make-polar 5.0 0.9273)        ;; -> 3.0+4.0i (approximately)
```

### Accessors

```scheme
(define z (make-rectangular 3.0 4.0))

(real-part z)      ;; -> 3.0
(imag-part z)      ;; -> 4.0
(magnitude z)      ;; -> 5.0
(angle z)          ;; -> 0.9273... (radians)
```

### Arithmetic

```scheme
;; All standard arithmetic works on complex numbers
(+ (make-rectangular 1.0 2.0)
   (make-rectangular 3.0 4.0))     ;; -> 4.0+6.0i

(* (make-rectangular 1.0 2.0)
   (make-rectangular 3.0 4.0))     ;; -> -5.0+10.0i

;; Division uses Smith's formula (overflow-safe)
(/ (make-rectangular 1.0 0.0)
   (make-rectangular 0.0 1.0))     ;; -> 0.0-1.0i

;; Math functions extend to complex domain
(sqrt (make-rectangular -1.0 0.0)) ;; -> 0.0+1.0i
(exp (make-rectangular 0.0 3.14159)) ;; -> -1.0+0.0i (approximately)
```

---

## Continuations & Exception Handling

Eshkol implements first-class continuations (`call/cc`), dynamic wind guards, and structured exception handling following R7RS semantics.

### First-Class Continuations

```scheme
;; call/cc captures the current continuation
(call/cc (lambda (k)
  (+ 1 (k 42))))  ;; -> 42 (k aborts and returns 42)

;; Non-local exit from a loop
(define (find-first pred lst)
  (call/cc (lambda (return)
    (for-each (lambda (x)
      (when (pred x) (return x)))
      lst)
    #f)))

(find-first even? (list 1 3 5 4 7))  ;; -> 4
```

### Dynamic Wind

```scheme
;; dynamic-wind guarantees before/after thunks execute
;; even across continuation invocations
(dynamic-wind
  (lambda () (display "entering\n"))
  (lambda () (+ 1 2))
  (lambda () (display "leaving\n")))
;; prints: entering
;; prints: leaving
;; -> 3

;; Resource management
(define (with-file path thunk)
  (let ((port (open-input-file path)))
    (dynamic-wind
      (lambda () #t)
      (lambda () (thunk port))
      (lambda () (close-port port)))))
```

### Structured Exception Handling

```scheme
;; guard provides try/catch-like exception handling
(guard (exn
        ((string? exn) (string-append "caught: " exn))
        (#t "unknown error"))
  (raise "something went wrong"))
;; -> "caught: something went wrong"

;; with-exception-handler for low-level handling
(with-exception-handler
  (lambda (exn) (display "Error!\n"))
  (lambda () (raise "boom")))

;; Raise errors
(raise "division by zero")
(raise (list 'error 'file-not-found "/tmp/missing.txt"))
```

---

## Parallel Primitives

Eshkol v1.1 provides built-in parallel execution primitives that automatically distribute work across available CPU cores. No threading libraries or manual synchronization required.

### Parallel Map, Filter, Fold

```scheme
;; parallel-map: apply function to list elements in parallel
(parallel-map (lambda (x) (* x x)) (list 1 2 3 4 5 6 7 8))
;; -> (1 4 9 16 25 36 49 64)

;; parallel-filter: filter elements in parallel
(parallel-filter prime? (iota 1000000))

;; parallel-fold: parallel reduction
(parallel-fold + 0 (iota 1000000))  ;; -> 499999500000

;; parallel-for-each: side effects in parallel
(parallel-for-each
  (lambda (file) (process-data file))
  file-list)
```

### Futures

```scheme
;; future: spawn computation, force: wait for result
(define f1 (future (lambda () (heavy-computation-a))))
(define f2 (future (lambda () (heavy-computation-b))))

;; Both computations run concurrently
(define result (+ (force f1) (force f2)))
```

### Parallel Execute

```scheme
;; Run multiple thunks concurrently
(parallel-execute
  (lambda () (train-model-a data))
  (lambda () (train-model-b data))
  (lambda () (train-model-c data)))
```

### Performance

Parallel primitives use work-stealing thread pools and are optimized for scientific computing workloads. For small collections, they automatically fall back to sequential execution to avoid overhead.

---

## GPU Acceleration

Eshkol v1.1 features automatic hardware dispatch for tensor operations. The runtime measures actual hardware throughput and routes computations to the fastest available backend: SIMD, Apple Accelerate/BLAS, Metal (macOS), or CUDA (Linux/Windows).

### Automatic Dispatch

```scheme
;; matmul automatically selects the fastest backend
;; SIMD for small matrices, BLAS for medium, GPU for large
(define A (rand 4096 4096))
(define B (rand 4096 4096))
(define C (matmul A B))  ;; dispatches to BLAS (~1.2 TFLOPS on Apple Silicon AMX)
```

The cost model uses measured peak throughput:
- **SIMD**: Small matrices, low overhead
- **Apple Accelerate (AMX)**: ~1.1 TFLOPS measured, best for most matrix sizes
- **Metal (macOS)**: ~200 GFLOPS in simulated float64, used when GPU genuinely outperforms
- **CUDA (Linux)**: Native float64 on NVIDIA GPUs

### Explicit GPU Operations

```scheme
;; Force GPU execution for specific operations
(gpu-matmul A B)                    ;; GPU matrix multiply
(gpu-elementwise * A B)             ;; Element-wise on GPU
(gpu-reduce + M)                    ;; Reduction on GPU
(gpu-softmax v)                     ;; Softmax on GPU
(gpu-transpose M)                   ;; Transpose on GPU
```

### Metal Backend (macOS)

The Metal backend uses simulated float64 via double-single arithmetic with correctly rounded intermediate results. Key implementation details:

- Software float64 on GPU with 0x3FF rounding mask (IEEE 754 compliant)
- Fused multiply-add support for numerical stability
- Automatic memory management bridging arena allocator to Metal buffers

### CUDA Backend (Linux)

Native float64 support on NVIDIA GPUs with automatic kernel generation for element-wise, reduction, and matrix operations.

---

## Consciousness Engine

Eshkol v1.1 includes a built-in **active inference consciousness engine** implementing the Global Workspace Theory with probabilistic factor graphs and free energy minimization. This provides logic programming, probabilistic reasoning, and cognitive modeling as native language features.

### Logic Variables and Unification

```scheme
;; Logic variables use ?name syntax (R7RS compatible)
(define s (make-substitution))

;; Unify binds logic variables
(define s1 (unify ?x 42 s))
(walk ?x s1)               ;; -> 42

;; Structural unification
(define s2 (unify (list ?x ?y) (list 1 2) s))
(walk ?x s2)               ;; -> 1
(walk ?y s2)               ;; -> 2

;; Unification failure returns #f
(unify 1 2 s)              ;; -> #f
```

### Knowledge Bases

```scheme
;; Create a knowledge base and assert facts
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent (list 'alice 'bob)))
(kb-assert! kb (make-fact 'parent (list 'bob 'charlie)))

;; Query the knowledge base
(kb-query kb 'parent)      ;; -> list of matching facts

;; Type predicates
(logic-var? ?x)             ;; -> #t
(substitution? s)           ;; -> #t
(kb? kb)                    ;; -> #t
```

### Factor Graphs and Probabilistic Inference

```scheme
;; Create a factor graph with 3 variables
(define fg (make-factor-graph 3))

;; Add factors connecting variables with conditional probability tables
(fg-add-factor! fg (list 0 1) cpt-matrix)

;; Run belief propagation inference
(fg-infer! fg 10)           ;; 10 iterations of message passing

;; Update CPTs for learning
(fg-update-cpt! fg 0 new-cpt)

;; Compute free energy (surprise)
(free-energy fg observations)

;; Expected free energy for action selection
(expected-free-energy fg action)

;; Type predicates
(fact? f)                   ;; -> #t
(factor-graph? fg)          ;; -> #t
```

### Global Workspace

```scheme
;; Create a workspace (Global Workspace Theory)
(define ws (make-workspace))

;; Register cognitive modules (each is a closure returning a content tensor)
(ws-register! ws "perception" perception-module)
(ws-register! ws "memory" memory-module)
(ws-register! ws "planning" planning-module)

;; Step the workspace: modules compete via softmax, winner broadcasts
(ws-step! ws input-tensor)

;; Type predicate
(workspace? ws)             ;; -> #t
```

### Example: Bayesian Sensor Fusion

```scheme
;; Two noisy sensors observing the same binary state
(define fg (make-factor-graph 3))  ;; state + 2 sensors

;; Prior: uniform
(fg-add-factor! fg (list 0) #(0.5 0.5))

;; Sensor 1: 90% accurate
(fg-add-factor! fg (list 0 1) #(0.9 0.1 0.1 0.9))

;; Sensor 2: 80% accurate
(fg-add-factor! fg (list 0 2) #(0.8 0.2 0.2 0.8))

;; Observe both sensors reporting state 0
(fg-infer! fg 20)
;; Posterior belief in state 0 is much higher than either sensor alone
```

---

## Signal Processing

Eshkol v1.1 includes a built-in signal processing library for spectral analysis, filtering, and windowing -- compiled to native code with no interpreter overhead.

### FFT and Spectral Analysis

```scheme
;; Fast Fourier Transform
(define signal #(1.0 0.0 -1.0 0.0 1.0 0.0 -1.0 0.0))
(define spectrum (fft signal))
(define recovered (ifft spectrum))

;; Power spectrum
(define power (tensor-mul spectrum spectrum))
```

### Windowing Functions

```scheme
;; Apply window functions before FFT to reduce spectral leakage
(define n 1024)
(define hamming-win (window-hamming n))
(define hann-win (window-hann n))

(define windowed-signal (tensor-mul signal hamming-win))
(define spectrum (fft windowed-signal))
```

### Digital Filters

```scheme
;; FIR filter (finite impulse response)
(define coefficients #(0.1 0.2 0.4 0.2 0.1))
(define filtered (fir-filter coefficients signal))

;; IIR filter (infinite impulse response)
(define b-coeffs #(0.0675 0.1349 0.0675))   ;; numerator
(define a-coeffs #(1.0 -1.1430 0.4128))     ;; denominator
(define filtered (iir-filter b-coeffs a-coeffs signal))

;; Butterworth lowpass filter design
(define cutoff 0.1)    ;; normalized frequency
(define order 4)
(define lp-filter (butterworth-lowpass order cutoff))
```

### Convolution

```scheme
;; Linear convolution
(define kernel #(1.0 2.0 1.0))
(define result (convolve signal kernel))
```

---

## Web Platform

Eshkol can compile to WebAssembly for browser deployment, with a DOM API for building interactive web applications.

### WASM Compilation

```bash
# Compile to WebAssembly
eshkol-run program.esk --target wasm -o program.wasm

# Generates: program.wasm + program.js (loader)
```

### DOM API

```scheme
;; Create and manipulate DOM elements
(define heading (web-create-element "h1"))
(web-set-text heading "Hello from Eshkol!")

(define button (web-create-element "button"))
(web-set-text button "Click me")
(web-add-event-listener button "click"
  (lambda (event)
    (web-set-text heading "Clicked!")))
```

### Canvas API

```scheme
;; 2D canvas drawing
(define canvas (web-canvas-create 800 600))
(web-canvas-fill-rect canvas 10 10 100 50 "blue")
(web-canvas-stroke-circle canvas 400 300 100 "red")
(web-canvas-draw-text canvas "Eshkol Graphics" 300 50 "white")

;; Animation loop
(define (render timestamp)
  (web-canvas-clear canvas)
  (web-canvas-fill-rect canvas
    (* 200 (sin (* timestamp 0.001))) 300 50 50 "green")
  (web-request-animation-frame render))

(web-request-animation-frame render)
```

---

## Higher-Order Functions

### Map, Filter, Fold

```scheme
;; Map: apply function to each element
(map (lambda (x) (* x 2)) (list 1 2 3 4))  ;; -> (2 4 6 8)

;; Multi-list map
(map + (list 1 2 3) (list 10 20 30))  ;; -> (11 22 33)

;; Filter: select matching elements
(filter (lambda (x) (> x 2)) (list 1 2 3 4 5))  ;; -> (3 4 5)

;; Fold: reduce to single value
(fold + 0 (list 1 2 3 4 5))    ;; -> 15
(fold * 1 (list 1 2 3 4 5))    ;; -> 120
```

### Closures

```scheme
;; Factory pattern
(define (make-adder n)
  (lambda (x) (+ x n)))

(define add5 (make-adder 5))
(define add10 (make-adder 10))

(add5 3)   ;; -> 8
(add10 3)  ;; -> 13

;; Counter with mutable state
(define (make-counter)
  (let ((count 0))
    (lambda ()
      (set! count (+ count 1))
      count)))
```

### Function Composition

```scheme
(define (compose f g)
  (lambda (x) (f (g x))))

(define (square x) (* x x))
(define (double x) (* 2 x))

(define square-then-double (compose double square))
(define double-then-square (compose square double))

(square-then-double 3)  ;; -> 18 (3^2 * 2)
(double-then-square 3)  ;; -> 36 ((3 * 2)^2)
```

### Standard Library Combinators

```scheme
;; From stdlib (two equivalent forms)
(require stdlib)
(load "path/to/file.esk")  ;; R7RS-compatible file loading (alias for require)

(identity x)           ;; Return x unchanged
(constantly 5)         ;; Returns (lambda (x) 5)
(flip f)               ;; Swap arguments: (flip -) -> (lambda (a b) (- b a))
(negate pred)          ;; (negate even?) -> odd?

;; Currying
(define curried-add (curry2 +))
((curried-add 5) 10)   ;; -> 15

;; Partial application
(define add5 (partial2 + 5))
(add5 10)              ;; -> 15
```

---

## Memory Model

### Arena Allocation (No Garbage Collection)

Eshkol uses **arena-based memory management** for deterministic, low-latency performance:

```
+------------------------------------------+
|              Arena Allocator              |
+------------------------------------------+
|  Block 1 (8KB)  ->  Block 2  ->  Block 3 |
|  [cons cells]      [closures]  [tensors]  |
+------------------------------------------+
|  Scope Stack: [scope1] -> [scope2] -> ... |
|  (Push on function entry, pop on exit)    |
+------------------------------------------+
```

**Benefits:**
- **No GC pauses** - Critical for real-time applications
- **Cache-friendly** - Sequential allocation
- **Deterministic cleanup** - Memory freed when scope exits
- **Fast allocation** - O(1) bump-pointer allocation

### Tagged Value System

Every runtime value is a 16-byte tagged structure:

```c
struct eshkol_tagged_value {
    uint8_t type;        // INT64, DOUBLE, CONS_PTR, CLOSURE_PTR, LOGIC_VAR, etc.
    uint8_t flags;       // Exactness, port direction bits
    uint16_t reserved;
    uint32_t padding;
    union {
        int64_t int_val;
        double double_val;
        void* ptr_val;
    } data;              // Field index {4} in LLVM IR
};
```

This enables **polymorphic operations** with runtime type dispatch:

```scheme
(list 1 2.5 "hello" (lambda (x) x) 1/3 ?x)  ;; Mixed-type list supported!
```

### Stack Configuration

Deep recursion is supported via configurable stack sizes:
- Default: 512MB stack via linker flags
- Maximum recursion depth: 100,000 frames
- Override via `ESHKOL_STACK_SIZE` environment variable

---

## C Interoperability

### Foreign Function Interface

Call C functions directly:

```scheme
;; Declare external C functions
(extern void printf char* ...)
(extern void* malloc int)
(extern double sin double)

;; Use them in Eshkol
(printf "The sine of %f is %f\n" 1.0 (sin 1.0))
```

### Function Aliasing

Map Eshkol names to C names:

```scheme
;; Map friendly names to C functions
(extern void log-message :real printf char* ...)
(extern void* allocate-memory :real malloc int)

(log-message "Allocating %d bytes\n" 1024)
```

---

## Interactive REPL

### Features

- **Tab completion** for all 555+ builtins
- **Syntax highlighting** with ANSI colors
- **Command history** (persistent across sessions)
- **Multi-line input** with balanced parenthesis detection
- **JIT compilation** via LLVM ORC
- **Full stdlib access** (precompiled .o + .bc metadata, 237 functions, 305 globals)

### REPL Commands

```
:help      Show help
:quit      Exit REPL
:env       Show defined symbols
:load FILE Load and execute file
:reload    Reload last file
:time EXPR Time expression execution
:ast EXPR  Show AST for expression
:clear     Clear screen
:history   Show command history
```

### Example Session

```
$ ./eshkol-repl
Eshkol REPL v1.1-accelerate
Type :help for assistance, :quit to exit

eshkol> (define (square x) (* x x))
square

eshkol> (square 5)
25

eshkol> (map square (list 1 2 3 4 5))
(1 4 9 16 25)

eshkol> (expt 2 256)
115792089237316195423570985008687907853269984665640564039457584007913129639936

eshkol> (+ 1/3 1/6)
1/2

eshkol> (parallel-map square (iota 8))
(0 1 4 9 16 25 36 49)

eshkol> (derivative (lambda (x) (* x x x)) 2.0)
12.0

eshkol> :quit
Goodbye!
```

---

## Examples

### Fibonacci Sequence

```scheme
(define (fibonacci n)
  (if (< n 2)
      n
      (+ (fibonacci (- n 1))
         (fibonacci (- n 2)))))

;; Print first 10 Fibonacci numbers
(do ((i 0 (+ i 1)))
    ((= i 10) 'done)
  (display "fib(")
  (display i)
  (display ") = ")
  (display (fibonacci i))
  (newline))
```

### Gradient Descent Optimization

```scheme
;; Minimize f(x,y) = (x-3)^2 + (y-4)^2
(define (loss v)
  (+ (* (- (vref v 0) 3.0) (- (vref v 0) 3.0))
     (* (- (vref v 1) 4.0) (- (vref v 1) 4.0))))

(define (gradient-step point lr)
  (let ((grad (gradient loss point)))
    (vector (- (vref point 0) (* lr (vref grad 0)))
            (- (vref point 1) (* lr (vref grad 1))))))

;; Optimize
(define start (vector 0.0 0.0))
(define lr 0.1)

(display "Start: ") (display start) (newline)
(display "Loss: ") (display (loss start)) (newline)

(define step1 (gradient-step start lr))
(display "After step 1: ") (display step1) (newline)
(display "Loss: ") (display (loss step1)) (newline)
;; Converges toward (3, 4)
```

### Neural Network Training

```scheme
;; Model: y = w*x + b (linear regression)
(define (model w b x) (+ (* w x) b))

;; Loss: Mean Squared Error
(define (mse w b x target)
  (let ((pred (model w b x)))
    (* (- pred target) (- pred target))))

;; Training step using autodiff
(define (train-step w b lr x y)
  (let ((grad-w (derivative (lambda (w-val) (mse w-val b x y)) w))
        (grad-b (derivative (lambda (b-val) (mse w b-val x y)) b)))
    (list (- w (* lr grad-w))
          (- b (* lr grad-b)))))

;; Train to learn y = 2x
(define training-x (list 1.0 2.0 3.0 4.0 5.0))
(define training-y (list 2.0 4.0 6.0 8.0 10.0))

;; After training: w ~ 2.0, b ~ 0.0
```

### Parallel Monte Carlo Pi Estimation

```scheme
;; Estimate pi using parallel random sampling
(require stdlib)

(define (in-circle? _)
  (let ((x (random))
        (y (random)))
    (if (<= (+ (* x x) (* y y)) 1.0) 1 0)))

(define samples 1000000)
(define hits (parallel-fold + 0
  (parallel-map in-circle? (iota samples))))

(define pi-estimate (* 4.0 (/ hits samples)))
(display "Pi ~ ") (display pi-estimate) (newline)
```

### Active Inference Agent

```scheme
;; Simple agent using consciousness engine for decision making
(define fg (make-factor-graph 4))  ;; state + 3 actions

;; Prior beliefs about state
(fg-add-factor! fg (list 0) #(0.5 0.5))

;; Action-outcome models
(fg-add-factor! fg (list 0 1) #(0.8 0.2 0.3 0.7))
(fg-add-factor! fg (list 0 2) #(0.4 0.6 0.6 0.4))
(fg-add-factor! fg (list 0 3) #(0.5 0.5 0.5 0.5))

;; Observe current state
(fg-infer! fg 20)

;; Select action minimizing expected free energy
(define efe-1 (expected-free-energy fg 1))
(define efe-2 (expected-free-energy fg 2))
(define efe-3 (expected-free-energy fg 3))

(display "Action EFEs: ")
(display (list efe-1 efe-2 efe-3))
(newline)
```

### Merge Sort

```scheme
(define (sort lst less?)
  (define (merge l1 l2)
    (cond ((null? l1) l2)
          ((null? l2) l1)
          ((less? (car l1) (car l2))
           (cons (car l1) (merge (cdr l1) l2)))
          (else
           (cons (car l2) (merge l1 (cdr l2))))))

  (if (or (null? lst) (null? (cdr lst)))
      lst
      (let ((mid (quotient (length lst) 2)))
        (merge (sort (take mid lst) less?)
               (sort (drop mid lst) less?)))))

(sort (list 3 1 4 1 5 9 2 6) <)  ;; -> (1 1 2 3 4 5 6 9)
```

---

## Technical Architecture

### Codebase Statistics

| Component | Lines of Code | Purpose |
|-----------|--------------|---------|
| LLVM Codegen | ~29,000 | IR generation, compilation |
| Tensor Codegen | ~2,500 | Tensor/matrix ops, GPU dispatch |
| System Codegen | ~1,800 | Parallel primitives, system ops |
| Arithmetic Codegen | ~3,000 | Exact arithmetic, bignums, rationals |
| String/IO Codegen | ~2,200 | String ops, port I/O |
| Binding Codegen | ~1,500 | let/letrec, closures, TCO |
| Parser | ~4,000 | Tokenizer, AST construction |
| Logic Engine | ~1,200 | Unification, substitutions, KB |
| Active Inference | ~1,500 | Factor graphs, BP, free energy |
| Global Workspace | ~800 | Softmax competition, broadcast |
| BLAS Backend | ~600 | Accelerate/BLAS integration |
| GPU Backend | ~1,200 | Metal/CUDA code generation |
| XLA Runtime | ~500 | XLA compilation target |
| Arena Memory | ~934 | Memory management |
| REPL JIT | ~630 | Interactive execution, ORC |
| Standard Library | ~2,000 | Higher-order utilities, math, ML |
| Signal Processing | ~800 | FFT, filters, windows |
| LSP Server | ~500 | IDE integration |
| **Total** | **~55,000+** | Complete implementation |

### Compilation Pipeline

```
+----------+   +----------+   +----------+   +----------+
|  Source   |-->|  Parser  |-->|   LLVM   |-->|  Native  |
|  (.esk)  |   |   (AST)  |   |   (IR)   |   |  Binary  |
+----------+   +----------+   +----------+   +----------+
                                   |    \
                                   v     \
                              +--------+  +--------+
                              | ORC JIT|  |  WASM  |
                              | (REPL) |  | Target |
                              +--------+  +--------+
```

### Hardware Dispatch

```
+--------------------+
|   Tensor Operation |
+--------------------+
         |
    +----+----+
    | Cost    |
    | Model   |
    +----+----+
    /    |     \
   v     v      v
+------+------+------+
| SIMD | BLAS | GPU  |
|      | AMX  | Metal|
|      |      | CUDA |
+------+------+------+
```

### Type System

Eshkol uses a **polymorphic tagged value system** at runtime:

- **18+ value types**: NULL, INT64, DOUBLE, CONS_PTR, DUAL_NUMBER, AD_NODE_PTR, TENSOR_PTR, LAMBDA_SEXPR, STRING_PTR, CHAR, VECTOR_PTR, SYMBOL, CLOSURE_PTR, COMPLEX, HEAP_PTR (bignum/rational/port), LOGIC_VAR, and heap subtypes for SUBSTITUTION, FACT, KNOWLEDGE_BASE, FACTOR_GRAPH, WORKSPACE
- **Full numeric tower**: Integer -> Bignum -> Rational -> Real -> Complex
- **Exactness tracking**: R7RS-compatible exact/inexact number distinction
- **Mixed-type lists**: Heterogeneous data fully supported
- **Homoiconicity**: Lambda S-expressions preserved for metaprogramming

---

## Why Eshkol?

### For Machine Learning Researchers

- **First-class autodiff**: No library imports, no tape management
- **Vector calculus operators**: Built-in gradient, jacobian, hessian, divergence, curl, laplacian
- **Native performance**: LLVM compilation, no interpreter overhead
- **GPU acceleration**: Automatic dispatch to fastest available hardware
- **Active inference**: Built-in consciousness engine for cognitive modeling

### For Scientific Computing

- **Exact arithmetic**: Full numeric tower -- bignums, rationals, complex numbers
- **Matrix operations**: Comprehensive linear algebra with hardware acceleration
- **Signal processing**: Built-in FFT, filters, windowing
- **Math library**: Integration, root finding, eigenvalues, ODE solvers

### For Real-Time Systems

- **No GC pauses**: Arena allocation with deterministic cleanup
- **Predictable latency**: No stop-the-world garbage collection
- **Efficient memory**: 16-byte aligned tagged values
- **Parallel primitives**: Built-in concurrent execution

### For Functional Programming Enthusiasts

- **Scheme heritage**: Familiar S-expression syntax, R7RS compatible
- **First-class functions**: Closures with proper lexical scoping
- **First-class continuations**: call/cc, dynamic-wind, guard/raise
- **Homoiconicity**: Code as data, metaprogramming ready
- **Interactive development**: Full-featured REPL with JIT

### For Systems Programmers

- **C FFI**: Direct foreign function calls
- **LLVM backend**: Native code generation
- **Library mode**: Compile as shared libraries
- **Low-level control**: Arena memory, no hidden allocations
- **WebAssembly**: Compile to WASM for browser deployment

### For AI Researchers

- **Logic programming**: Unification, knowledge bases, query
- **Probabilistic reasoning**: Factor graphs, belief propagation
- **Active inference**: Free energy minimization, expected free energy for action selection
- **Global workspace**: Cognitive architecture with softmax competition

---

## Comparison with Alternatives

| Feature | Eshkol | PyTorch | JAX | Julia | TensorFlow |
|---------|--------|---------|-----|-------|------------|
| **Syntax** | Scheme | Python | Python | Julia | Python |
| **Compilation** | AOT/JIT/WASM | JIT | JIT | JIT | Graph |
| **Autodiff** | Built-in | Library | Library | Library | Library |
| **Vector Calculus** | Built-in | Manual | Manual | Manual | Manual |
| **GC** | None | Yes | Yes | Yes | Yes |
| **Homoiconicity** | Yes | No | No | No | No |
| **Closures** | Native | Limited | Limited | Native | No |
| **Exact Arithmetic** | Full tower | No | No | Partial | No |
| **GPU Auto-dispatch** | Yes | Manual | Manual | Manual | Manual |
| **Logic Programming** | Built-in | No | No | No | No |
| **Continuations** | First-class | No | No | No | No |
| **Signal Processing** | Built-in | torchaudio | No | Library | tf.signal |

---

## Roadmap

- [x] Core language implementation
- [x] LLVM code generation
- [x] Automatic differentiation (3 modes)
- [x] Vector calculus operators
- [x] Interactive REPL with JIT
- [x] Standard library
- [x] Math library
- [x] Exact arithmetic (bignums, rationals)
- [x] Complex numbers
- [x] First-class continuations (call/cc, dynamic-wind, guard/raise)
- [x] GPU acceleration (Metal/CUDA) -- automatic hardware dispatch
- [x] Logic programming and knowledge bases
- [x] Active inference consciousness engine
- [x] Signal processing (FFT, filters, windows)
- [x] Parallel primitives (parallel-map, future/force)
- [x] WebAssembly compilation target
- [x] LSP server for IDE integration
- [x] Package manager (eshkol-pkg)
- [ ] Distributed computing (multi-node)
- [ ] Debugger (step/break/inspect)

---

## Getting Involved

- **Repository**: https://github.com/tsotchke/eshkol
- **Issues**: Report bugs, request features
- **Pull Requests**: Contributions welcome

---

## License

MIT License - Copyright (C) tsotchke

---

<p align="center">
<strong>Eshkol v1.1-accelerate</strong>: Where functional programming meets scientific computing, GPU acceleration, and machine consciousness.
</p>
