# Getting Started with Eshkol

## Table of Contents
- [Installation](#installation)
- [Your First Eshkol Program](#your-first-eshkol-program)
- [Basic Syntax](#basic-syntax)
- [Common Operations](#common-operations)
- [Development Workflow](#development-workflow)
- [Interactive REPL](#interactive-repl)

---

## Installation

### Prerequisites

Eshkol requires:
- **LLVM 14+** (core dependency)
- **CMake 3.16+** (build system)
- **C++17 compiler** (GCC 9+, Clang 10+, or MSVC 2019+)
- **Git** (for source checkout)

### Building from Source

```bash
# Clone repository
git clone https://github.com/tsotchke/eshkol.git
cd eshkol

# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build (parallel compilation)
cmake --build build -j$(nproc)

# Install system-wide (optional)
sudo cmake --install build

# Or add to PATH
export PATH="$PATH:$(pwd)/build"
```

### Platform-Specific Installation

#### Ubuntu/Debian

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    cmake \
    build-essential

# Build Eshkol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### macOS

```bash
# Install dependencies via Homebrew
brew install llvm@14 cmake

# Build Eshkol
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=$(brew --prefix llvm@14)/lib/cmake/llvm
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Verification

```bash
# Check compiler version
build/eshkol-run --version
# Expected: Eshkol 1.0.0-foundation

# Check REPL
build/eshkol-repl
# Expected: Interactive prompt
```

---

## Your First Eshkol Program

### Hello World

Create `hello.esk`:

```scheme
(display "Hello, Eshkol!")
(newline)
```

**Compile and run:**

```bash
# Compile (generates executable in current directory)
build/eshkol-run hello.esk

# Run
./hello
# Output: Hello, Eshkol!
```

### Factorial (Recursion)

Create `factorial.esk`:

```scheme
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

(display "5! = ")
(display (factorial 5))
(newline)
```

**Run:**

```bash
build/eshkol-run factorial.esk
# Output: 5! = 120
```

---

## Basic Syntax

Eshkol uses S-expressions (prefix notation):

### Variables and Functions

```scheme
;; Define variable
(define pi 3.14159)

;; Define function (short form)
(define (square x)
  (* x x))

;; Define function (lambda form)
(define square
  (lambda (x) (* x x)))

;; Call function
(square 5)  ; Returns 25
```

### Conditionals

```scheme
;; if expression
(if (> x 0)
    "positive"
    "non-positive")

;; cond expression (multi-way conditional)
(cond
  ((< x 0) "negative")
  ((= x 0) "zero")
  (else "positive"))

;; Boolean operators
(and (> x 0) (< x 10))  ; Short-circuit and
(or (= x 0) (= x 1))    ; Short-circuit or
```

### Let Bindings

```scheme
;; let - parallel bindings
(let ((x 1)
      (y 2))
  (+ x y))  ; Returns 3

;; let* - sequential bindings
(let* ((x 1)
       (y (* x 2)))  ; y can reference x
  (+ x y))  ; Returns 3

;; letrec - recursive bindings
(letrec ((even? (lambda (n)
                  (if (= n 0) #t (odd? (- n 1)))))
         (odd? (lambda (n)
                 (if (= n 0) #f (even? (- n 1))))))
  (even? 10))  ; Returns #t
```

---

## Common Operations

### Working with Lists

```scheme
;; Create list
(define numbers (list 1 2 3 4 5))
;; Or: (define numbers '(1 2 3 4 5))

;; Access elements
(car numbers)     ; First: 1
(cadr numbers)    ; Second: 2
(caddr numbers)   ; Third: 3
(cdr numbers)     ; Rest: (2 3 4 5)

;; List operations
(length numbers)              ; 5
(append (list 1 2) (list 3 4))  ; (1 2 3 4)
(reverse numbers)             ; (5 4 3 2 1)

;; Higher-order functions
(map (lambda (x) (* x x)) numbers)      ; (1 4 9 16 25)
(filter (lambda (x) (> x 2)) numbers)   ; (3 4 5)
(fold + 0 numbers)                      ; 15 (sum)
```

### Working with Strings

```scheme
;; String operations
(string-append "Hello, " "world!")  ; "Hello, world!"
(string-length "Eshkol")            ; 6
(substring "Eshkol" 0 3)            ; "Esh"

;; String predicates
(string? "hello")     ; #t
(string=? "a" "a")    ; #t
```

### Working with Tensors

```scheme
;; Create tensor
(define v (tensor 1.0 2.0 3.0))

;; Tensor operations
(tensor-length v)         ; 3
(vref v 0)                ; 1.0
(tensor-dot v v)          ; 14.0 (1² + 2² + 3²)

;; Linear algebra
(define A (reshape (tensor 1.0 2.0 3.0 4.0) (vector 2 2)))
(tensor-transpose A)      ; [[1,3],[2,4]]
```

### Automatic Differentiation

```scheme
;; Compute derivative
(derivative (lambda (x) (* x x)) 3.0)
;; Returns: 6.0 (derivative of x² at x=3 is 2x = 6)

;; Compute gradient
(gradient (lambda (v)
            (+ (* (vref v 0) (vref v 0))
               (* (vref v 1) (vref v 1))))
          (vector 3.0 4.0))
;; Returns: #(6.0 8.0)
```

---

## Development Workflow

### Project Structure

```
my-project/
├── src/
│   ├── main.esk          # Entry point
│   └── utils.esk         # Utilities
├── lib/
│   └── module.esk        # Reusable modules
├── tests/
│   └── test_utils.esk    # Tests
└── build/                # Build artifacts
```

### Build Script

Create `build.sh`:

```bash
#!/bin/bash
set -e

# Build main program
eshkol-run src/main.esk -o build/program

# Run tests
for test in tests/*.esk; do
    echo "Running $test..."
    eshkol-run "$test"
done

echo "Build complete!"
```

### Running Tests

Eshkol includes test scripts in `scripts/`:

```bash
# Run specific test suite
./scripts/run_autodiff_tests.sh
./scripts/run_list_tests.sh
./scripts/run_tensor_tests.sh

# Run all tests
for script in scripts/run_*_tests.sh; do
    $script || echo "FAILED: $script"
done
```

---

## Interactive REPL

The Eshkol REPL provides interactive development with JIT compilation:

```bash
# Start REPL
build/eshkol-repl
```

### REPL Session Example

```scheme
eshkol> (define (square x) (* x x))
eshkol> (square 5)
25

eshkol> (define numbers (list 1 2 3 4 5))
eshkol> (map square numbers)
(1 4 9 16 25)

eshkol> (fold + 0 numbers)
15

eshkol> (gradient (lambda (x) (* x x)) 3.0)
6.0
```

### REPL Commands

- **`(exit)`** or **Ctrl+D** - Exit REPL
- **`(load "file.esk")`** - Load file into REPL
- **`:help`** - Show help (if implemented)

---

## Common Patterns

### Factorial

```scheme
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))
```

### Map Implementation

```scheme
(define (my-map f lst)
  (if (null? lst)
      '()
      (cons (f (car lst))
            (my-map f (cdr lst)))))
```

### Filter Implementation

```scheme
(define (my-filter pred lst)
  (cond
    ((null? lst) '())
    ((pred (car lst))
     (cons (car lst) (my-filter pred (cdr lst))))
    (else
     (my-filter pred (cdr lst)))))
```

### Fold Implementation

```scheme
(define (my-fold f init lst)
  (if (null? lst)
      init
      (my-fold f (f (car lst) init) (cdr lst))))
```

---

## Next Steps

Now that you have Eshkol installed and running:

1. **Learn the type system:** [Type System](TYPE_SYSTEM.md)
2. **Explore tensors and AD:** [Vector Operations](VECTOR_OPERATIONS.md), [Automatic Differentiation](AUTODIFF.md)
3. **Understand memory management:** [Memory Management](MEMORY_MANAGEMENT.md)
4. **Study the compiler:** [Compiler Architecture](COMPILER_ARCHITECTURE.md)
5. **Check API reference:** [API Reference](../API_REFERENCE.md)
6. **Try the quickstart tutorial:** [Quickstart](../QUICKSTART.md)

---

## See Also

- [Quickstart Tutorial](../QUICKSTART.md) - 15-minute hands-on tutorial
- [API Reference](../API_REFERENCE.md) - Complete function reference
- [Scheme Compatibility](SCHEME_COMPATIBILITY.md) - R5RS/R7RS support
- [Language Overview](OVERVIEW.md) - Design philosophy and technical depth
