# Eshkol: Scientific Computing with Deterministic Memory and Integrated Numerics

This document details Eshkol v1.0-architecture's capabilities for scientific computing, focusing on the **arena memory management system**, **tensor operations**, and **numerical algorithms** actually implemented in the compiler and standard library.

## The Scientific Computing Landscape

Scientific computing demands:
- **High performance** for large-scale simulations
- **Numerical stability** for accurate results
- **Deterministic behavior** for reproducible science
- **Memory efficiency** for massive datasets
- **Interactive development** for algorithm exploration

Traditional solutions force compromises:
- **Python/MATLAB**: Easy to use but slow, unpredictable GC pauses
- **Fortran/C++**: Fast but difficult, manual memory management
- **Julia**: JIT compilation causes startup delays, GC still present

Eshkol addresses these challenges through **compiler-integrated numerics**, **arena-based memory**, and **LLVM-native performance**.

## Arena Memory Management

### Ownership-Aware Lexical Regions (OALR)

Eshkol implements **OALR** - a deterministic memory system without garbage collection:

**Core Concepts:**
1. **Arena Allocation** - Bump-pointer allocation in large blocks
2. **Lexical Regions** - Memory tied to code structure
3. **Ownership Analysis** - Compile-time tracking prevents errors
4. **Escape Analysis** - Automatic allocation strategy selection

### Arena Structure

```c
struct arena {
    arena_block_t* current_block;   // Current allocation block
    arena_scope_t* current_scope;   // Lexical scope tracking
    size_t default_block_size;      // Block size (default 64KB)
    size_t total_allocated;         // Tracking for statistics
    size_t alignment;               // Memory alignment (8 bytes)
}
```

**Allocation Strategy:**
- Bump `current_block->used` pointer forward
- Allocate new block when current exhausted
- All blocks freed when arena destroyed
- No per-object metadata overhead

### Global Arena and Region Stack

**Global State:**
```c
arena_t* __global_arena;               // Default allocation target
eshkol_region_t* __region_stack[16];   // Nested regions
uint64_t __region_stack_depth;         // Current nesting
```

**Region Management:**
```c
region_create(name, size_hint)  // Create named region
region_push(region)             // Activate region
region_pop()                    // Deactivate and free
region_current()                // Get active region
```

### with-region Syntax

**Lexical Memory Scoping:**
```scheme
; Anonymous region
(with-region
  (define large-matrix (zeros 1000 1000))
  (define result (compute large-matrix)))
; Matrix freed here automatically

; Named region with size hint
(with-region ('temp 8192)  ; 8KB hint
  (define data (iota 1000))
  (process data))

; Nested regions
(with-region 'outer
  (define x (compute-x))
  (with-region 'inner
    (define y (compute-y x))
    (combine x y)))
; Inner freed, then outer freed
```

### Escape Analysis

**Compiler determines allocation strategy:**

```
Escape Analysis:
├─ NO_ESCAPE
│  └─ Stack allocation (fastest, no cleanup needed)
│     Example: Local temporary in function body
│
├─ RETURN_ESCAPE  
│  └─ Region allocation (lexical lifetime)
│     Example: Allocated, modified, returned from function
│
└─ CLOSURE_ESCAPE / GLOBAL_ESCAPE
   └─ Shared allocation (reference counted)
      Example: Captured in closure, stored globally
```

**Implementation in [`exe/eshkol-run.cpp`](../../exe/eshkol-run.cpp):**
- Scans AST to determine variable lifetimes
- Marks allocation strategy for each value
- Backend uses marks to select allocation
- Prevents heap allocation when unnecessary

### Ownership Tracking

**States:**
```c
enum OwnershipState {
    UNOWNED,     // Not yet owned
    OWNED,       // Exclusively owned
    MOVED,       // Ownership transferred
    BORROWED     // Temporarily accessed
}
```

**Compile-Time Checks:**
- Cannot use after move
- Cannot move while borrowed
- Borrows must not outlive owned value
- Linear types must be consumed exactly once

**Example:**
```scheme
; Ownership example (syntax exists, enforcement partial in v1.0)
(define x (owned (list 1 2 3)))
(define y (move x))
; x now invalid, y owns the list

; Borrow for read-only access
(borrow y
  (display (length y))  ; OK - read only
  (car y))              ; OK
; y still owned after borrow
```

## Tensor Operations

### Tensor Structure

```c
struct eshkol_tensor {
    uint64_t* dimensions;      // [rows, cols, ...] for N-D
    uint64_t num_dimensions;   // Rank (1=vector, 2=matrix, etc.)
    int64_t* elements;         // Data (doubles as int64 bits)
    uint64_t total_elements;   // Product of dimensions
}
```

**Memory Layout:**
- Header: 8 bytes (object header with HEAP_SUBTYPE_TENSOR)
- Tensor struct: 32 bytes
- Dimensions array: 8 × num_dimensions bytes
- Elements array: 8 × total_elements bytes (doubles stored as int64 bit patterns)

### Tensor Creation

```scheme
; 1D vector
(define v #(1.0 2.0 3.0 4.0 5.0))

; 2D matrix (rows × cols)
(define M (matrix 3 2  1.0 2.0
                       3.0 4.0
                       5.0 6.0))

; N-D tensor
(define T (tensor 2 3 4  ; 2×3×4 tensor
                  1.0 2.0 3.0 4.0
                  5.0 6.0 7.0 8.0
                  9.0 10.0 11.0 12.0
                  13.0 14.0 15.0 16.0
                  17.0 18.0 19.0 20.0
                  21.0 22.0 23.0 24.0))
```

### Tensor Access

```scheme
; 1D access
(vref v 0)              ; => 1.0
(vref v 4)              ; => 5.0

; N-D access (implemented for matrices)
(tensor-get M 0 0)      ; => 1.0
(tensor-get M 2 1)      ; => 6.0
```

### Tensor Operations

**Element-Wise Arithmetic:**
```scheme
(tensor-add v1 v2)      ; Component-wise addition
(tensor-sub v1 v2)      ; Component-wise subtraction
(tensor-mul v1 v2)      ; Component-wise multiplication
(tensor-div v1 v2)      ; Component-wise division
```

**Matrix Operations:**
```scheme
(tensor-dot v1 v2)      ; Dot product (scalar)
(matmul M1 M2)          ; Matrix multiplication
(transpose M)           ; Matrix transpose
```

**Reductions:**
```scheme
(tensor-sum v)          ; Sum all elements
(tensor-mean v)         ; Mean of elements
(tensor-reduce-all v + 0.0)  ; Custom reduction
```

**Transformations:**
```scheme
(reshape T new-dims)    ; Change shape (shares data)
(tensor-shape T)        ; Get dimension list
(tensor-apply T func)   ; Apply function element-wise
```

## Numerical Algorithms in math.esk

The [`lib/math.esk`](../../lib/math.esk) library implements classical numerical algorithms in pure Eshkol:

### Constants

```scheme
(import 'math.esk')

pi       ; => 3.141592653589793
e        ; => 2.718281828459045  
epsilon  ; => 1e-15 (machine epsilon)
```

### Linear Algebra

#### Matrix Determinant (LU Decomposition)

```scheme
(define M (matrix 3 3  1 2 3
                       4 5 6
                       7 8 10))

(det M 3)  ; => -3.0
; Uses LU decomposition with partial pivoting
; Implementation: Gaussian elimination in pure Eshkol
```

**Algorithm (from math.esk):**
- Find pivot (max element in column)
- Swap rows if needed (track sign)
- Eliminate column below pivot
- Multiply diagonal elements
- Apply sign corrections

#### Matrix Inverse (Gauss-Jordan)

```scheme
(define M (matrix 3 3  2 1 1
                       1 2 1
                       1 1 2))

(define M-inv (inv M 3))
; => Inverse matrix via Gauss-Jordan elimination
; Returns #f if matrix is singular
```

**Algorithm (from math.esk):**
- Augment with identity matrix
- Row reduction to reduced row echelon form
- Extract inverse from augmented portion

#### Linear System Solver

```scheme
(define A (matrix 3 3  2 1 1
                       1 2 1
                       1 1 2))
(define b #(6.0 5.0 4.0))

(define x (solve A b 3))
; => Solution vector x where Ax = b
; Uses LU decomposition with forward/backward substitution
```

### Vector Operations

```scheme
; Dot product
(dot #(1.0 2.0 3.0) #(4.0 5.0 6.0))
; => 32.0

; Cross product (3D only)
(cross #(1.0 0.0 0.0) #(0.0 1.0 0.0))
; => #(0.0 0.0 1.0)

; Normalize to unit vector
(normalize #(3.0 4.0))
; => #(0.6 0.8)
```

### Numerical Integration

**Simpson's Rule:**
```scheme
(define (integrate f a b n)
  ; Numerical integration over [a, b] with n intervals
  ; Uses Simpson's 1/3 rule for accuracy
  ...)

; Integrate sin(x) from 0 to π
(integrate sin 0.0 pi 100)
; => ~2.0 (analytical result is 2)
```

### Root Finding

**Newton-Raphson Method:**
```scheme
(define (newton f df x0 tolerance max-iters)
  ; Find root of f using Newton's method
  ; df is derivative of f
  ...)

; Find √2 as root of x² - 2
(newton (lambda (x) (- (* x x) 2.0))
        (lambda (x) (* 2.0 x))
        1.0
        1e-10
        100)
; => 1.4142135623730951 (√2)
```

### Eigenvalues

**Power Iteration Method:**
```scheme
(define (power-iteration A n max-iters tolerance)
  ; Estimate dominant eigenvalue
  ; A is n×n matrix
  ...)

(define M (matrix 2 2  4 1
                       2 3))
(power-iteration M 2 100 1e-10)
; => ~5.0 (dominant eigenvalue)
```

### Statistics

```scheme
; Variance
(variance #(1.0 2.0 3.0 4.0 5.0))
; => 2.0

; Standard deviation
(std #(1.0 2.0 3.0 4.0 5.0))
; => 1.4142135623730951 (√2)

; Covariance
(covariance #(1.0 2.0 3.0) #(2.0 4.0 6.0))
; => Covariance between two vectors
```

## Quantum-Inspired RNG for Stochastic Computing

### What It Actually Is

Eshkol includes a **quantum-inspired random number generator** - an 8-qubit quantum circuit simulation used for high-quality randomness, **not actual quantum computing**.

**Implementation in [`lib/quantum/quantum_rng.c`](../../lib/quantum/quantum_rng.c):**
- 8-qubit quantum state simulation
- Hadamard gates for superposition
- Phase gates for entanglement
- Measurement collapse for classical randomness
- 16-element entropy pool with runtime entropy injection

### QRNG Architecture

**Context Structure:**
```c
struct qrng_ctx {
    uint64_t phase[8];              // 8 qubit phases
    double quantum_state[8];        // Quantum state values
    uint64_t entangle[8];           // Entanglement correlations
    uint64_t last_measurement[8];   // Previous measurements
    double entropy_pool[16];        // Entropy accumulation
    uint64_t pool_mixer;            // Entropy mixing state
    // ... system entropy, timestamps, etc.
}
```

**Quantum Operations:**
```c
hadamard_gate(x)           // Create superposition
phase_gate(x, angle)       // Quantum phase rotation  
measure_state(ctx, state)  // Collapse to classical value
quantum_step(ctx)          // Full mixing round
```

### Usage for Stochastic Methods

```scheme
; High-quality random numbers
(define rand (qrng-double))        ; [0, 1)
(define rand-int (qrng-range 1 100))  ; [1, 100]

; Monte Carlo integration
(define (monte-carlo-pi samples)
  (let loop ((i 0) (inside 0))
    (if (>= i samples)
        (* 4.0 (/ inside samples))
        (let* ((x (qrng-double))
               (y (qrng-double))
               (dist (+ (* x x) (* y y))))
          (loop (+ i 1)
                (if (<= dist 1.0) (+ inside 1) inside))))))

(monte-carlo-pi 1000000)  ; => ~3.14159...
```

**Important:** This is **not quantum computing** (no qubits, gates, or quantum algorithms). It's classical simulation of quantum randomness for **stochastic computing applications**.

## Memory Efficiency for Scientific Workloads

### Arena Benefits for Numerical Computing

**1. Predictable Allocation:**
```scheme
; Traditional GC: unpredictable pauses during computation
; Eshkol Arena: allocation is O(1), deallocation is bulk

(with-region 'simulation
  (define state-vectors (make-vector 1000))
  (define jacobian (matrix 1000 1000))
  ; Allocate 1M+ elements rapidly
  (run-time-step state-vectors jacobian)
  ; No GC pause during critical computation
  )
; All memory freed in O(1) when region exits
```

**2. Cache-Friendly Layout:**
- Sequential allocation improves cache locality
- Related data allocated together
- Fewer cache misses than scattered heap allocation

**3. Zero Fragmentation:**
- No per-object headers (except for heap objects)
- No free-list management
- Compact memory layout

### Allocation Decision Examples

**Stack Allocation (NO_ESCAPE):**
```scheme
(define (compute-local x)
  (let ((temp (* x x)))  ; Stack allocated
    (+ temp 1)))
```

**Region Allocation (RETURN_ESCAPE):**
```scheme
(define (build-matrix n)
  (let ((M (matrix n n)))  ; Region allocated
    ; Fill matrix
    M))  ; Returns from region
```

**Shared Allocation (CLOSURE_ESCAPE):**
```scheme
(define (make-closure x)
  (lambda (y)
    (+ x y)))  ; x captured → shared allocation
```

## Numerical Algorithm Performance

### Matrix Operations

**Implemented in Pure Eshkol (math.esk):**

```scheme
; Solve 3×3 linear system
(define A (matrix 3 3  2 1 1
                       1 2 1  
                       1 1 2))
(define b #(6.0 5.0 4.0))
(define x (solve A b 3))
; => #(2.0 1.0 1.0) solution vector

; Verify solution
(tensor-dot (matrix-row A 0) x)  ; => 6.0 ✓
```

**Algorithm Complexity:**
- `det`: O(n³) - Gaussian elimination
- `inv`: O(n³) - Gauss-Jordan
- `solve`: O(n³) - LU decomposition
- `power-iteration`: O(kn²) - k iterations

### Numerical Stability

**Techniques Employed:**
- Partial pivoting in LU decomposition
- Compensated summation (Kahan) for reductions
- Machine epsilon for singularity detection
- Careful handling of numerical precision

```scheme
; Epsilon used for stability checks
(define epsilon 1e-15)

; Check for singular matrix
(if (< (abs pivot) epsilon)
    #f  ; Singular
    (compute-inverse))
```

## Tensor Implementation Details

### N-Dimensional Display

**Recursive tensor display** from [`lib/core/arena_memory.cpp`](../../lib/core/arena_memory.cpp):
```scheme
(define T (tensor 2 2 2  1 2 3 4 5 6 7 8))
(display T)
; => #(((1 2) (3 4)) ((5 6) (7 8)))
; Nested structure reflects dimensionality
```

**Display Algorithm:**
- Recursively descend dimensions
- Innermost: print element list
- Higher levels: print nested sublists
- Preserves tensor structure visually

### Mixed-Type Lists for Heterogeneous Data

**32-byte cons cells** store complete tagged values:
```scheme
; List with different types
(define data (list 42                    ; int64
                   3.14                  ; double  
                   "hello"               ; string
                   #(1 2 3)              ; vector
                   (lambda (x) (* x 2))  ; closure
                   ))

; Each element fully typed, no boxing overhead
(integer? (car data))        ; => #t
(real? (cadr data))          ; => #t
(string? (caddr data))       ; => #t
(vector? (cadddr data))      ; => #t
(procedure? (car (cddddr data)))  ; => #t
```

**Implementation:**
```c
struct arena_tagged_cons_cell {
    eshkol_tagged_value_t car;  // 16 bytes - complete type info
    eshkol_tagged_value_t cdr;  // 16 bytes - complete type info
}  // Total: 32 bytes (cache-aligned)
```

## Scientific Computing Workflows

### Physics Simulation

```scheme
(require stdlib)
(require math.esk)

; Particle system with forces
(define (simulate-particles positions velocities forces dt steps)
  (let loop ((pos positions)
             (vel velocities)
             (step 0))
    (if (>= step steps)
        (values pos vel)
        (let* (; Compute forces (could use gradient of potential)
               (new-forces (compute-forces pos))
               ; Velocity Verlet integration
               (new-vel (tensor-add vel 
                          (tensor-mul new-forces (* 0.5 dt))))
               (new-pos (tensor-add pos
                          (tensor-mul new-vel dt))))
          (loop new-pos new-vel (+ step 1))))))
```

### Statistical Analysis

```scheme
; Load and analyze data
(require core.data.csv)

(define data (csv-parse-file "experiment.csv"))
(define values (map (lambda (row) (string->number (cadr row))) data))

; Compute statistics
(define mean (tensor-mean (list->vector values)))
(define stddev (std (list->vector values)))
(define var (variance (list->vector values)))

(display "Mean: ") (display mean) (newline)
(display "StdDev: ") (display stddev) (newline)
```

### Optimization with Automatic Differentiation

```scheme
; Minimize a mathematical function
(define (minimize f initial-guess)
  (let loop ((x initial-guess)
             (iteration 0))
    (if (> iteration 100)
        x
        (let* ((fx (f x))
               (grad (gradient f x))
               (step-size 0.01)
               (new-x (tensor-sub x (tensor-mul grad step-size))))
          (if (< (tensor-sum (tensor-mul grad grad)) 1e-10)
              new-x  ; Converged
              (loop new-x (+ iteration 1)))))))
```

## Interactive Scientific Development

### REPL for Algorithm Exploration

```scheme
$ eshkol-repl
> (require stdlib)
> (require math.esk)

> ; Define function interactively
> (define (my-func x) (+ (* x x) (* 2 x) 1))

> ; Test with values
> (my-func 3)
16

> ; Compute derivative
> (derivative my-func 3.0)
8.0  ; 2x + 2 at x=3

> ; Explore numerically
> (map my-func (iota-from 10 -5))
(16 10 6 4 4 6 10 16 24 34)
```

### JIT Compilation in REPL

**LLVM ORC JIT:**
- Expressions compiled on-the-fly
- Native code execution
- Persistent state across evaluations
- Global arena shared between evaluations

**Shared Arena Benefits:**
```scheme
> (define data (iota 1000))  ; Allocated in __repl_shared_arena
> (define squared (map (lambda (x) (* x x)) data))
> ; Both data and squared persist
> (length squared)
1000
> ; Available in next evaluation
```

## Practical Example: Gradient Descent for Linear Regression

```scheme
(require stdlib)

; Generate noisy linear data
(define (generate-data n slope intercept noise)
  (let ((xs (arange 0.0 n)))
    (map (lambda (x)
           (vector x (+ (* slope x) intercept 
                        (* noise (- (qrng-double) 0.5)))))
         (vector->list xs))))

; Model: y = wx + b
(define (predict w b x)
  (+ (* w x) b))

; Mean squared error loss
(define (mse-loss w b data)
  (let ((errors 
         (map (lambda (point)
                (let ((x (vref point 0))
                      (y (vref point 1)))
                  (let ((pred (predict w b x)))
                    (* (- pred y) (- pred y)))))
              data)))
    (/ (fold + 0.0 errors) (length data))))

; Training loop using gradient
(define (train data w0 b0 lr epochs)
  (let loop ((w w0) (b b0) (e 0))
    (if (>= e epochs)
        (vector w b)
        (let* (; Create loss function for current parameters
               (loss-fn (lambda (params)
                          (mse-loss (vref params 0) (vref params 1) data)))
               ; Compute gradients
               (grad (gradient loss-fn (vector w b)))
               (grad-w (vref grad 0))
               (grad-b (vref grad 1))
               ; Update parameters
               (new-w (- w (* lr grad-w)))
               (new-b (- b (* lr grad-b))))
          (when (= (modulo e 100) 0)
            (display "Epoch ") (display e)
            (display ", Loss: ") (display (mse-loss w b data))
            (newline))
          (loop new-w new-b (+ e 1))))))

; Run training
(define data (generate-data 100 2.5 1.0 0.5))
(define params (train data 0.0 0.0 0.01 1000))
(display "Final params: ") (display params) (newline)
```

## What v1.0 Does NOT Include

**Not in v1.0:**
- ❌ GPU/CUDA tensor operations
- ❌ Automatic parallelization (pmap, pfold)
- ❌ Built-in plotting/visualization
- ❌ Units of measurement syntax
- ❌ Fully Symbolic understanding (apart from lambdas and basic differentiation)
- ❌ Distributed computing
- ❌ BLAS/LAPACK integration (math.esk is pure Eshkol)
- ❌ Sparse matrix support
- ❌ FFT operations

**v1.0 Provides:**
- ✅ Deterministic arena memory
- ✅ Tensor operations (vector, matrix, N-D)
- ✅ Automatic differentiation (8 operators)
- ✅ Numerical algorithms (det, inv, solve, integrate, newton, power-iteration)
- ✅ Statistical functions (variance, std, covariance)
- ✅ Quantum RNG for stochastic methods
- ✅ Interactive REPL for exploration
- ✅ CSV/JSON data I/O

## Comparison with Scientific Languages

### vs. Python + NumPy

**Memory:**
- NumPy: Garbage collection with unpredictable pauses
- Eshkol: Arena allocation, deterministic timing

**Performance:**
- NumPy: C extensions for speed (Python overhead at boundaries)
- Eshkol: LLVM-native throughout (no language boundaries)

**AD Integration:**
- NumPy+JAX: Separate framework, graph tracing
- Eshkol: Compiler-integrated, direct recording

### vs. Julia

**Compilation:**
- Julia: JIT compilation, startup delays
- Eshkol: AOT compilation, instant startup

**Memory:**
- Julia: Garbage collection
- Eshkol: Arena allocation

**Type System:**
- Julia: Multiple dispatch on types
- Eshkol: Tagged values with runtime dispatch + HoTT gradual typing

### vs. MATLAB

**Licensing:**
- MATLAB: Commercial, expensive
- Eshkol: Open source (MIT)

**Language:**
- MATLAB: Proprietary array-oriented syntax
- Eshkol: S-expressions with Scheme semantics

**Memory:**
- MATLAB: Automatic memory, hidden behavior
- Eshkol: Explicit arenas, clear lifetimes

## Future Scientific Computing Features

See [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for:
- GPU tensor acceleration
- BLAS/LAPACK integration
- Sparse matrix support
- FFT operations
- Parallel map/fold
- Built-in plotting
- Symbolic mathematics expansion

---

*Eshkol v1.0-architecture provides a solid foundation for scientific computing through deterministic memory management, compiler-integrated automatic differentiation, and numerical algorithms implemented in the language itself. The arena system eliminates garbage collection pauses while the AD system enables natural expression of gradient-based algorithms.*
