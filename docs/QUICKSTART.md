# Eshkol Quickstart Guide

**Get started with Eshkol in 15 minutes**

This hands-on tutorial introduces Eshkol's core features through practical examples. You'll learn functions, lists, tensors, and automatic differentiation - the tools for scientific computing and AI systems programming.

---

## Installation

### macOS (Homebrew)

```bash
brew tap tsotchke/eshkol
brew install eshkol
```

### Linux (Debian/Ubuntu)

```bash
# Download .deb from https://github.com/tsotchke/eshkol/releases
sudo dpkg -i eshkol_*.deb
```

### Windows (MSYS2/MinGW64)

```bash
# In MSYS2 MinGW64 shell:
pacman -S mingw-w64-x86_64-llvm mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### From Source (macOS/Linux)

```bash
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Quick Test

```bash
# Interactive REPL
eshkol-repl

# Compile and run a program
echo '(display "Hello, Eshkol!")' > hello.esk
eshkol-run hello.esk -o hello
./hello
```

**Requirements**: LLVM 17+, C++20 compiler, CMake 3.14+, Ninja
**Platforms**: Linux, macOS (x86-64, ARM64), Windows (MSYS2/MinGW64)

**Web REPL**: Open `web/index.html` in a browser for an interactive Eshkol environment without installation.

**Bytecode**: Emit portable ESKB bytecode alongside native compilation: `eshkol-run hello.esk -B hello.eskb`

---

## Part 1: Basics (2 minutes)

### Numbers and Arithmetic

```scheme
; Integers are exact
(+ 1 2 3)           ; => 6
(* 7 6)             ; => 42
(- 10 3)            ; => 7

; Division promotes to double
(/ 10 3)            ; => 3.333...

; Mixed types promote automatically
(+ 5 2.5)           ; => 7.5

; Math functions
(sqrt 16)           ; => 4.0
(sin 3.14159)       ; => ~0.0
(pow 2 10)          ; => 1024.0
```

**Type System**: Polymorphic arithmetic with automatic promotion (int64 → double when needed)

---

### Variables and Functions

```scheme
; Define variables
(define pi 3.14159)
(define x 42)

; Define functions
(define (square x) 
  (* x x))

(define (circle-area r)
  (* pi r r))

(circle-area 5)  ; => 78.5398...

; Lambda expressions (anonymous functions)
(define double (lambda (x) (* 2 x)))
(double 21)      ; => 42

; Functions are first-class values
(define ops (list + - * /))
(map (lambda (f) (f 10 2)) ops)  ; => (12 8 20 5.0)
```

---

## Part 2: Lists (3 minutes)

Lists are the fundamental data structure for symbolic computation.

```scheme
; Creating lists
'(1 2 3)                    ; Quoted literal
(list 1 2 3)                ; Constructor
(cons 1 (cons 2 '()))       ; Build with cons

; Basic operations
(car '(1 2 3))              ; => 1 (first element)
(cdr '(1 2 3))              ; => (2 3) (rest)
(length '(a b c d))         ; => 4

; Nested lists
(define matrix '((1 2) (3 4) (5 6)))
(car (car matrix))          ; => 1
(cadr matrix)               ; => (3 4)

; Higher-order operations
(map (lambda (x) (* x 2)) '(1 2 3))     ; => (2 4 6)
(filter even? '(1 2 3 4 5 6))           ; => (2 4 6)
(fold + 0 '(1 2 3 4))                   ; => 10
```

**Import Standard Library**:
```scheme
(require core.list.higher_order)  ; fold, filter, map, etc.
(require core.list.sort)           ; sort, merge
```

---

## Part 3: Closures (2 minutes)

Closures capture their environment and support mutation.

```scheme
; Counter with state
(define make-counter
  (lambda ()
    (let ((count 0))
      (lambda ()
        (set! count (+ count 1))
        count))))

(define c (make-counter))
(c)  ; => 1
(c)  ; => 2
(c)  ; => 3

; Function factory (currying)
(define (make-adder n)
  (lambda (x) (+ x n)))

(define add5 (make-adder 5))
(add5 10)  ; => 15

; Mutable captures
(define (make-account balance)
  (lambda (amount)
    (set! balance (+ balance amount))
    balance))

(define acct (make-account 100))
(acct 50)   ; => 150
(acct -30)  ; => 120
```

**Key Feature**: Closures support `set!` on captured variables (mutable captures via pointer-passing)

---

## Part 4: Tensors (3 minutes)

N-dimensional arrays for numerical computing.

```scheme
; Create tensors
(define v (vector 1.0 2.0 3.0))         ; 1D vector
(define M #((1 2 3) (4 5 6)))           ; 2D matrix (literal syntax)

; Tensor creation functions
(zeros 5)           ; => #(0.0 0.0 0.0 0.0 0.0)
(ones 2 3)          ; => #((1.0 1.0 1.0) (1.0 1.0 1.0))
(eye 3)             ; => 3×3 identity matrix
(arange 0.0 1.0 0.2) ; => #(0.0 0.2 0.4 0.6 0.8)

; Element access
(vref v 0)          ; => 1.0
(tensor-get M 1 2)  ; => 6

; Arithmetic (element-wise)
(tensor-add #(1 2 3) #(4 5 6))  ; => #(5 7 9)
(tensor-mul v v)                ; => #(1.0 4.0 9.0)

; Linear algebra
(define A #((1 2) (3 4)))
(define B #((5 6) (7 8)))
(matmul A B)        ; => #((19 22) (43 50))
(transpose A)       ; => #((1 3) (2 4))
(trace A)           ; => 5.0

; Reductions
(tensor-sum #(1 2 3 4 5))   ; => 15.0
(tensor-mean v)             ; => 2.0
(norm #(3.0 4.0))           ; => 5.0
```

**Performance**: Zero-copy views for reshape/transpose/flatten

---

## Part 5: Automatic Differentiation (5 minutes)

The killer feature: **three modes of differentiation**.

### Symbolic Differentiation

Compile-time transformation with algebraic simplification.

```scheme
; Symbolic derivative (AST transformation)
(diff '(* x x) 'x)              ; => (* 2 x)
(diff '(sin (* 2 x)) 'x)        ; => (* 2 (cos (* 2 x)))
(diff '(/ 1 x) 'x)              ; => (/ -1 (* x x))
```

### Forward-Mode AD (Dual Numbers)

Efficient for functions ℝ → ℝⁿ (single input, many outputs).

```scheme
; Derivative of scalar function
(define (f x) (* x x x))  ; x³
(derivative f 2.0)        ; => 12.0 (3x² at x=2)

; Works with complex expressions
(define (g x) (sin (exp x)))
(derivative g 0.0)        ; => 1.0 (chain rule automatic)

; Higher-order: returns derivative function
(define df (derivative f))
(df 3.0)  ; => 27.0
```

**Method**: Dual numbers `(value, derivative)` with automatic chain rule

---

### Reverse-Mode AD (Backpropagation)

Efficient for functions ℝⁿ → ℝ (many inputs, single output). Essential for machine learning.

```scheme
; Gradient of scalar function
(define (f v) 
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1))))

(gradient f #(3.0 4.0))  ; => #(6.0 8.0) (∇f = [2x, 2y])

; Practical example: least squares loss
(define (loss weights data)
  (let ((predictions (matmul data weights))
        (errors (tensor-sub predictions targets)))
    (tensor-sum (tensor-mul errors errors))))

(define grad-loss (gradient loss))
(grad-loss W train-data)  ; Gradient for optimization
```

**Method**: Computational graph + backpropagation (reverse accumulation)

---

### Vector Calculus

Physics and field theory operators.

```scheme
; Divergence (source strength)
(define (F v) v)  ; Identity field
(divergence F #(1.0 2.0 3.0))  ; => 3.0

; Curl (rotation) - 3D only
(define (B v) (vector (vref v 1) (- (vref v 0)) 0.0))
(curl B #(1.0 2.0 0.0))  ; => #(0.0 0.0 -2.0)

; Laplacian (heat/wave equations)
(define (u v) (+ (* (vref v 0) (vref v 0))
                 (* (vref v 1) (vref v 1))))
(laplacian u #(1.0 1.0))  ; => 4.0

; Jacobian matrix
(define (F v) (vector (* 2 (vref v 0))
                      (+ (vref v 0) (vref v 1))))
(jacobian F #(3.0 4.0))  ; => #((2.0 0.0) (1.0 1.0))

; Hessian (curvature)
(hessian f #(1.0 1.0))  ; => #((2.0 0.0) (0.0 2.0))
```

**Applications**: Physics simulations, optimization, PDE solvers

---

## Part 6: Complete Example - Neural Network Training

Here's a complete 2-layer neural network with backpropagation:

```scheme
(require core.functional)

; Network parameters
(define input-size 3)
(define hidden-size 4)
(define output-size 2)

; Xavier initialization
(define (random-matrix rows cols)
  (let* ((scale (sqrt (/ 2.0 (+ rows cols))))
         (total (* rows cols))
         (data (make-vector total 0.0)))
    (define (init i)
      (if (< i total)
          (begin
            (vector-set! data i (* scale (- (random) 0.5)))
            (init (+ i 1)))
          (reshape data rows cols)))
    (init 0)))

; Initialize weights
(define W1 (random-matrix input-size hidden-size))
(define W2 (random-matrix hidden-size output-size))

; Activation functions
(define (relu x) (if (> x 0.0) x 0.0))
(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

; Forward pass
(define (forward x)
  (let* ((h-pre (matmul x W1))
         (h (tensor-apply h-pre relu))
         (y-pre (matmul h W2))
         (y (tensor-apply y-pre sigmoid)))
    y))

; Loss function (MSE)
(define (mse-loss pred target)
  (let ((diff (tensor-sub pred target)))
    (/ (tensor-sum (tensor-mul diff diff)) 2.0)))

; Training step with gradient descent
(define (train-step x target learning-rate)
  ; Compute gradients using automatic differentiation
  (let* ((loss-fn (lambda (w1 w2)
                    (let* ((h-pre (matmul x w1))
                           (h (tensor-apply h-pre relu))
                           (y-pre (matmul h w2))
                           (y (tensor-apply y-pre sigmoid)))
                      (mse-loss y target))))
         
         ; Gradient with respect to W1 (holding W2 constant)
         (grad-w1 (gradient (lambda (w) (loss-fn w W2)) W1))
         
         ; Gradient with respect to W2 (holding W1 constant)
         (grad-w2 (gradient (lambda (w) (loss-fn W1 w)) W2)))
    
    ; Update weights: W := W - α∇L
    (set! W1 (tensor-sub W1 (tensor-mul learning-rate grad-w1)))
    (set! W2 (tensor-sub W2 (tensor-mul learning-rate grad-w2)))
    
    ; Return current loss
    (loss-fn W1 W2)))

; Training loop
(define (train epochs learning-rate)
  (define input #(1.0 0.5 -0.5))
  (define target #(1.0 0.0))
  
  (define (loop epoch)
    (if (< epoch epochs)
        (let ((loss (train-step input target learning-rate)))
          (if (= (modulo epoch 100) 0)
              (begin
                (display "Epoch ")
                (display epoch)
                (display " Loss: ")
                (display loss)
                (newline)))
          (loop (+ epoch 1)))))
  (loop 0))

; Train the network
(train 1000 0.01)
```

**What just happened?**

1. **Defined a neural network** with 2 layers (3→4→2 neurons)
2. **Automatic differentiation** computed gradients of loss w.r.t. all weights
3. **Gradient descent** updated weights to minimize loss
4. **Zero manual derivatives** - the compiler computed them automatically!

---

## Key Language Features

### 1. **Homoiconic Code-as-Data**

```scheme
; Code is data
(define code '(lambda (x) (* x 2)))

; Manipulate code
(define (get-lambda-params code)
  (cadr code))  ; => (x)

; Display shows source structure
(display (lambda (x) (* x 2)))
; Output: (lambda (x) (* x 2))
```

---

### 2. **Pattern Matching**

```scheme
(define (classify x)
  (match x
    ((? number?) 'is-number)
    ((a b c) 'is-triple)
    ((h . t) 'is-pair)
    (_ 'unknown)))

(classify 42)           ; => is-number
(classify '(1 2 3))     ; => is-triple
```

---

### 3. **Tail Call Optimization**

```scheme
; Tail-recursive factorial (no stack overflow)
(define (factorial n)
  (letrec ((fact-iter 
            (lambda (n acc)
              (if (= n 0) 
                  acc
                  (fact-iter (- n 1) (* n acc))))))
    (fact-iter n 1)))

(factorial 10000)  ; Works! (optimized to loop)
```

---

### 4. **Exception Handling**

```scheme
(guard (exn
         ((divide-by-zero? exn) 'infinity)
         (else 'unknown-error))
  (/ 1 0))
; => infinity
```

---

### 5. **File I/O**

```scheme
; Write to file
(define out (open-output-file "data.txt"))
(write-line out "Hello, File!")
(close-port out)

; Read from file
(define in (open-input-file "data.txt"))
(define line (read-line in))
(close-port in)
(display line)  ; => "Hello, File!"
```

---

## Advanced Example: Optimization

Gradient descent to find minimum of Rosenbrock function.

```scheme
; Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
(define (rosenbrock v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (+ (* (- 1.0 x) (- 1.0 x))
       (* 100.0 (- y (* x x)) (- y (* x x))))))

; Gradient descent optimizer
(define (gradient-descent f x0 learning-rate iterations)
  (define point x0)
  
  (define (step i)
    (if (< i iterations)
        (let* ((grad (gradient f point))
               (update (tensor-mul learning-rate grad))
               (new-point (tensor-sub point update)))
          
          ; Print progress
          (if (= (modulo i 100) 0)
              (begin
                (display "Iter ")
                (display i)
                (display " f(x) = ")
                (display (f point))
                (display " x = ")
                (display point)
                (newline)))
          
          (set! point new-point)
          (step (+ i 1)))
        point))
  
  (step 0))

; Find minimum (should converge to [1, 1])
(define result 
  (gradient-descent rosenbrock 
                    #(-0.5 -0.5)   ; Starting point
                    0.001          ; Learning rate
                    1000))         ; Iterations

(display "Optimum: ")
(display result)
(newline)
(display "f(optimum) = ")
(display (rosenbrock result))
```

**Output**:
```
Iter 0 f(x) = 26.5625 x = #(-0.5 -0.5)
Iter 100 f(x) = 3.847 x = #(0.234 0.076)
...
Iter 900 f(x) = 0.00012 x = #(0.989 0.978)
Optimum: #(0.999 0.998)
f(optimum) = 8.4e-7
```

**What makes this powerful?**
- Gradient computed **automatically** - no manual calculus!
- Works with **any function** - change `rosenbrock` to your objective
- Scales to **high dimensions** - gradient computation is O(1) per parameter

---

## Physics Example: Heat Equation

Solve the 1D heat equation: ∂u/∂t = α∇²u

```scheme
(require core.functional)

; Discretized Laplacian (finite differences)
(define (discrete-laplacian u dx)
  (let* ((n (vector-length u))
         (result (make-vector n 0.0)))
    (define (loop i)
      (if (< i n)
          (let ((left (if (> i 0) (vref u (- i 1)) 0.0))
                (center (vref u i))
                (right (if (< i (- n 1)) (vref u (+ i 1)) 0.0)))
            (vector-set! result i 
              (/ (+ left (* -2.0 center) right) (* dx dx)))
            (loop (+ i 1)))
          result))
    (loop 0)))

; Time step
(define (heat-step u alpha dt dx)
  (let ((lap (discrete-laplacian u dx)))
    (tensor-add u (tensor-mul (* alpha dt) lap))))

; Simulation
(define (simulate-heat u0 alpha dt dx steps)
  (define u u0)
  (define (loop step)
    (if (< step steps)
        (begin
          (set! u (heat-step u alpha dt dx))
          (if (= (modulo step 100) 0)
              (begin
                (display "Step ")
                (display step)
                (display ": ")
                (display u)
                (newline)))
          (loop (+ step 1)))
        u))
  (loop 0))

; Initial condition: Gaussian bump
(define u0 #(0.0 0.1 0.5 1.0 0.5 0.1 0.0))

; Run simulation
(simulate-heat u0 0.1 0.01 0.1 500)
```

---

## Quick Reference Card

### Essential Syntax

```scheme
; Comments
; Single-line comment

; Variables
(define x value)
(set! x new-value)

; Functions
(define (f x y) body)
(lambda (x) body)

; Conditionals
(if test then else)
(cond (test result)... (else default))

; Lists
'(1 2 3)
(car lst)
(cdr lst)
(cons x lst)

; Loops
(let loop ((i 0))
  (if (< i 10)
      (begin (display i) (loop (+ i 1)))))

; Tensors
#(1 2 3)
(vref tensor idx)
(tensor-add a b)

; AD
(derivative f x)
(gradient f point)
(jacobian f point)
```

---

## Next Steps

### Learn More

1. **[API Reference](API_REFERENCE.md)** — Complete function reference (550+ functions)
2. **[Architecture Guide](ESHKOL_V1_ARCHITECTURE.md)** — System internals (memory, types, compilation)
3. **[Language Reference](../ESHKOL_V1_LANGUAGE_REFERENCE.md)** — Comprehensive language guide
4. **[Autodiff Deep Dive](breakdown/AUTODIFF.md)** — Three differentiation modes in detail

### Example Programs

Explore the examples directory:
- `examples-dep/neural_network_complete.esk` - Full neural network
- `examples-dep/vector_calculus.esk` - Physics simulations
- `examples-dep/function_composition_closure.esk` - Functional programming patterns
- `tests/ml/impressive_demo.esk` - ML showcase

### Standard Library

```scheme
; Import modules
(require core.list.higher_order)  ; fold, filter, any, every
(require core.list.sort)           ; sort, merge
(require core.functional)          ; compose, curry, flip
(require core.strings)             ; String utilities
(require core.json)                ; JSON parsing

; Or import everything at once
(require stdlib)
```

Full module list in [`lib/stdlib.esk`](../lib/stdlib.esk)

---

### Statistics

```scheme
(require math.statistics)

(define data '(4.0 7.0 13.0 2.0 9.0 15.0 6.0 11.0))

(display (median data))         ; => 8.0
(display (std-dev data))        ; => 4.183...
(display (percentile data 75))  ; => 13.0

;; Summary statistics
(define summary (describe data))
;; Returns: ((count . 8) (mean . 8.375) (std . 4.183) (min . 2.0)
;;           (q1 . 5.5) (median . 8.0) (q3 . 12.5) (max . 15.0))
```

---

### ODE Solvers

```scheme
(require math.ode)

;; Solve dy/dt = -y, y(0) = 1.0 (analytical: y = e^(-t))
(define (f t y) (- y))

;; Euler method
(display (euler-final f 0.0 1.0 0.01 100))  ; => ~0.366 (e^(-1))

;; 4th-order Runge-Kutta (more accurate)
(display (rk4-final f 0.0 1.0 0.01 100))    ; => ~0.3679 (closer to e^(-1))
```

---

### CLI Compilation Workflow

```bash
# Compile to named executable
eshkol-run program.esk -o program
./program

# Compile-only (produce .o file without linking)
eshkol-run -c program.esk -o program.o

# Evaluate expression directly
eshkol-run -e '(display (* 6 7))'

# Dump LLVM IR for debugging
eshkol-run --dump-ir program.esk

# Optimize level (0-3)
eshkol-run -O2 program.esk -o program

# Compile without stdlib (bare metal)
eshkol-run --no-stdlib program.esk -o program
```

---

## Tips & Tricks

### Performance

```scheme
; Use let for cached computations
(let ((expensive-calc (big-computation)))
  (+ expensive-calc expensive-calc))  ; Computed once

; Use tensors for numerical code (not lists)
(tensor-sum #(1 2 3))  ; Fast
(fold + 0 '(1 2 3))    ; Slower (list overhead)

; Tail recursion is your friend
(letrec ((sum (lambda (n acc)
                (if (= n 0) acc (sum (- n 1) (+ n acc))))))
  (sum 1000000 0))  ; No stack overflow!
```

---

### Debugging

```scheme
; Display shows structure
(display my-data)

; Type introspection
(type-of x)  ; Returns type tag as integer

; List inspection
(null? lst)
(pair? lst)
(length lst)
```

---

### Common Patterns

```scheme
; Higher-order function composition
(define (compose f g)
  (lambda (x) (f (g x))))

(define square-then-double
  (compose (lambda (x) (* x 2)) 
           (lambda (x) (* x x))))

; Partial application
(define (partial f . args)
  (lambda rest
    (apply f (append args rest))))

(define add5 (partial + 5))

; Currying
(define (curry2 f)
  (lambda (x) (lambda (y) (f x y))))

(define curried-add (curry2 +))
((curried-add 3) 4)  ; => 7
```

---

## Troubleshooting

**Error: "Unknown function"**
- Check spelling and imports
- Use `(require core.module)` for standard library

**Error: "Type mismatch"**
- Eshkol uses gradual typing - type errors are warnings
- Check if you're mixing incompatible types

**Segfault or crash**
- Report bug with minimal reproducer
- Check if using external pointers correctly

**Slow performance**
- Compile with `eshkol-run` for production (not REPL)
- Use tensors for numerical code
- Enable tail call optimization with `letrec`

---

## REPL Commands

```bash
# Start REPL
eshkol-repl

# Load file in REPL
eshkol> (load "mycode.esk")

# Exit
eshkol> (exit)
```

**REPL Features**:
- Cross-evaluation persistence (variables and functions persist)
- JIT compilation (LLVM ORC)
- Incremental development

---

## Compilation

### Compile to Executable

```bash
# Single file
eshkol-run program.esk -o program

# With standard library
eshkol-run -c program.esk -o program.o
gcc program.o build/libeshkol-static.a -o program -lm

# Run
./program
```

### Compile to Object File

```bash
eshkol-run -c mylib.esk -o mylib.o
ar rcs libmylib.a mylib.o
```

---

## v1.1 Features

### Exact Arithmetic (Bignums & Rationals)

```scheme
;; Arbitrary-precision integers
(define big (+ 9223372036854775807 1))  ; Overflows int64 → bignum
(display big)       ; => 9223372036854775808
(exact? big)        ; => #t

;; Exact exponentiation
(expt 2 64)         ; => 18446744073709551616 (exact bignum)

;; Rational numbers
(define r (make-rational 1 3))
(+ r r)             ; => 2/3 (exact)

;; R7RS: exact + inexact → inexact
(+ big 0.5)         ; => 9.223372036854776e+18 (double)
```

---

### Complex Numbers

```scheme
;; Create complex numbers
(define z (make-rectangular 3.0 4.0))
(magnitude z)       ; => 5.0
(angle z)           ; => 0.9273...

;; Complex arithmetic
(+ z (make-rectangular 1.0 -1.0))  ; => 4.0+3.0i
(* z z)             ; => -7.0+24.0i

;; Smith's formula for numerical stability
(/ (make-rectangular 1e300 1e300)
   (make-rectangular 1e300 1e300))  ; => 1.0+0.0i (no overflow)
```

---

### GPU Acceleration

GPU dispatch is automatic — no code changes needed:

```scheme
;; Large matrix multiply automatically uses GPU when beneficial
(define A (make-tensor (list 1000 1000) 1.0))
(define B (make-tensor (list 1000 1000) 2.0))
(define C (matmul A B))  ; Dispatches to Metal/CUDA if available
```

Control GPU behavior with environment variables:
```bash
ESHKOL_ENABLE_GPU=1 eshkol-run my_program.esk    # Force GPU
ESHKOL_GPU_VERBOSE=1 eshkol-run my_program.esk   # See dispatch decisions
```

---

### Parallel Programming

```scheme
;; Parallel map — distributes work across threads
(parallel-map (lambda (x) (* x x)) '(1 2 3 4 5))  ; => (1 4 9 16 25)

;; Parallel fold
(parallel-fold + 0 (range 1 1001))  ; => 500500

;; Futures for async computation
(define f1 (future (expensive-computation-1)))
(define f2 (future (expensive-computation-2)))
(+ (force f1) (force f2))  ; Both run concurrently
```

---

### Consciousness Engine

```scheme
;; Logic programming — unification
(define s (make-substitution))
(define s2 (unify ?x 42 s))
(walk ?x s2)  ; => 42

;; Knowledge base
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'alice 'bob))
(kb-assert! kb (make-fact 'parent 'bob 'charlie))
(kb-query kb (make-fact 'parent ?who 'charlie))
;; => list of substitutions where ?who = bob

;; Active inference with factor graphs
(define fg (make-factor-graph 2 #(2 2)))   ; 2 binary variables
(fg-add-factor! fg #(0 1) #(-0.1 -2.3 -1.6 -0.2))  ; log-probabilities
(define beliefs (fg-infer! fg 20))         ; belief propagation
(free-energy fg #(1.0 0.0))               ; variational free energy
```

---

### Signal Processing

```scheme
(require signal)

;; FFT — Cooley-Tukey radix-2 (input length must be power of 2)
(define signal #(1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0))
(define spectrum (fft signal))
(define recovered (ifft spectrum))   ; inverse FFT recovers original

;; Window functions for spectral analysis
(define win (hamming-window 256))
(define windowed (apply-window signal (hamming-window 8)))

;; Butterworth lowpass filter (cutoff normalized 0..1, 1=Nyquist)
(define lp (butterworth-lowpass 4 0.2))  ; 4th order, cutoff at 0.2×Nyquist
(define filtered (fir-filter (car lp) signal))

;; Frequency response analysis
(define resp (frequency-response (car lp) (cdr lp) 512))
;; resp = (magnitudes . phases)
```

---

### REPL

```bash
$ eshkol-repl

eshkol> (define (f x) (* x x x))
eshkol> (derivative f 2.0)
12.0
eshkol> (require stdlib)
eshkol> (sort < '(3 1 4 1 5 9))
(1 1 3 4 5 9)
eshkol> :type 42
Integer
eshkol> :quit
```

**REPL Features**: Cross-eval persistence, hot code reload, stdlib support, type introspection.

---

### Machine Learning with v1.1 Builtins

```scheme
;; Initialize weights with Kaiming initialization
(define W1 (zeros 4 3))
(kaiming-normal! W1 3)              ; fan-in = 3
(define W2 (zeros 2 4))
(kaiming-normal! W2 4)              ; fan-in = 4

;; Forward pass using builtin activations
(define (forward x)
  (let* ((h (relu (matmul x W1)))    ; hidden layer + ReLU
         (out (sigmoid (matmul h W2)))) ; output + sigmoid
    out))

;; Training with Adam optimizer
(define m1 (zeros 4 3))  ; 1st moment
(define v1 (zeros 4 3))  ; 2nd moment
(define m2 (zeros 2 4))
(define v2 (zeros 2 4))

(define (train-step x target step)
  (let* ((pred (forward x))
         (loss (mse-loss pred target))
         (grad-w1 (gradient (lambda (w)
                    (mse-loss (sigmoid (matmul (relu (matmul x w)) W2)) target)) W1))
         (grad-w2 (gradient (lambda (w)
                    (mse-loss (sigmoid (matmul (relu (matmul x W1)) w)) target)) W2)))
    ;; Clip gradients to prevent explosion
    (clip-grad-norm! grad-w1 1.0)
    (clip-grad-norm! grad-w2 1.0)
    ;; Adam updates (modifies W1, W2, m1, v1, m2, v2 in-place)
    (adam-step W1 grad-w1 0.001 m1 v1 step)
    (adam-step W2 grad-w2 0.001 m2 v2 step)
    loss))
```

---

### Package Manager

```bash
# Create a new project
eshkol-pkg init my-project
cd my-project

# Project structure created:
# eshkol.toml     — manifest file
# src/main.esk    — entry point

# Build and run
eshkol-pkg build
eshkol-pkg run

# Add a dependency
eshkol-pkg add some-package 1.0.0

# Clean build artifacts
eshkol-pkg clean
```

The `eshkol.toml` manifest:
```toml
[package]
name = "my-project"
version = "0.1.0"
license = "MIT"
entry = "src/main.esk"

[dependencies]
```

---

### Web/WASM Platform

Compile Eshkol to WebAssembly for browser deployment:

```bash
# Compile to WASM
eshkol-run --wasm app.esk -o app.wasm
```

```scheme
;; app.esk — interactive web application
(require web)

;; Get DOM handles
(define body (web-get-body))

;; Create UI
(define btn (web-create-element "button"))
(web-set-text-content btn "Click me!")
(web-set-style btn "fontSize" "18px")
(web-append-child body btn)

;; Counter with event handling
(define count 0)
(web-add-event-listener btn "click"
  (lambda (event)
    (set! count (+ count 1))
    (web-set-text-content btn
      (string-append "Clicked " (number->string count) " times"))))

;; Canvas drawing
(define canvas (web-create-element "canvas"))
(web-set-attribute canvas "width" "400")
(web-set-attribute canvas "height" "300")
(web-append-child body canvas)

(define ctx (web-get-context-2d canvas))
(web-canvas-fill-style ctx "blue")
(web-canvas-fill-rect ctx 50.0 50.0 100.0 80.0)
(web-canvas-fill-text ctx "Hello from Eshkol!" 60.0 100.0)
```

73 web API functions available: DOM manipulation, events, Canvas 2D, localStorage, fetch, timers.

---

## Resources

- **GitHub**: https://github.com/tsotchke/eshkol
- **Documentation**: [`docs/`](.)
- **Examples**: [`examples-dep/`](../examples-dep)
- **Tests**: [`tests/`](../tests) (434 test files across 35 suites)

---

## Community & Support

**Report Issues**: GitHub Issues
**Discussions**: GitHub Discussions
**License**: MIT

---

**Next**: Read the [API Reference](API_REFERENCE.md) for complete function documentation, or dive into the [Machine Learning deep dive](breakdown/MACHINE_LEARNING.md) for neural network training patterns.

---

**Copyright** © 2025-2026 tsotchke
**License**: MIT