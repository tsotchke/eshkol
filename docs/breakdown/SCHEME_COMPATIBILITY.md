# Scheme Compatibility in Eshkol

## Table of Contents
- [Overview](#overview)
- [R5RS/R7RS Compatibility](#r5rsr7rs-compatibility)
- [Differences from Standard Scheme](#differences-from-standard-scheme)
- [Eshkol Extensions to Scheme](#eshkol-extensions-to-scheme)
- [Migration Considerations](#migration-considerations)

---

## Overview

Eshkol is a **Scheme-family language** providing extensive R5RS and R7RS compatibility while adding:
- Automatic differentiation (3 modes)
- Tensors and linear algebra
- Gradual typing (HoTT-inspired)
- OALR memory management (transparent to users)
- LLVM-based native compilation

Most Scheme code runs in Eshkol with no modifications. The syntax is identical, and core semantics are preserved.

---

## R5RS/R7RS Compatibility

### Core Language Features

**✅ Fully Supported:**

```scheme
;; Special forms
define, lambda, if, cond, case, and, or, let, let*, letrec, begin, quote, quasiquote

;; List operations
cons, car, cdr, caar, cadr, cdar, cddr, caaar, caadr, ..., null?, pair?, list?, list, append, reverse, length, list-ref

;; Higher-order functions
map, filter, fold (reduce), apply, for-each

;; Arithmetic
+, -, *, /, abs, quotient, remainder, modulo, gcd, lcm, floor, ceiling, round, truncate, sqrt, expt, exp, log, sin, cos, tan, asin, acos, atan

;; Comparison
=, <, >, <=, >=, eq?, eqv?, equal?

;; Type predicates
number?, integer?, real?, boolean?, char?, string?, symbol?, procedure?, list?, pair?, null?, vector?

;; String operations
string-append, string-length, substring, string=?, string<?, string>?, string-ref

;; Vector operations (Scheme vectors, not tensors)
vector, make-vector, vector-ref, vector-set!, vector-length

;; I/O
display, write, newline, read, open-input-file, open-output-file, close-input-port, close-output-port, read-char, write-char, read-line

;; Control flow
begin, when, unless

;; Assignment
set!

;; Macros
define-syntax, syntax-rules
```

### R7RS-small Additions

**✅ Supported:**

```scheme
;; Multiple return values
(values 1 2 3)
(call-with-values producer consumer)
(let-values (((a b) (values 1 2))) (+ a b))

;; Exception handling
(guard (var ((type-error? var) "type error")
             (else "other error"))
  (error "test"))

(raise (make-error "message"))

;; Module system
(require 'module-name)
(provide symbol1 symbol2)

;; Additional list operations
take, drop, split-at, last-pair, assoc, member
```

**⚠️ Partial Support:**

- `call/cc` and `call-with-current-continuation` - Defined in AST but full implementation planned
- `bytevectors` - Defined as subtype but operations not yet implemented
- `records` - Defined as subtype but full record system planned

**❌ Not Supported:**

- `dynamic-wind` - Not implemented
- `delay`/`force` (promises) - Planned
- Numeric tower (exact/inexact arithmetic) - Eshkol uses doubles primarily

---

## Differences from Standard Scheme

### 1. Memory Management (Transparent)

**Scheme:** Garbage collector manages memory automatically.

**Eshkol:** Global arena manages memory automatically (users don't see this).

```scheme
;; This code works identically in Scheme and Eshkol
(define (process-data data)
  (let ((result (map (lambda (x) (* x x)) data)))
    result))

;; Eshkol's arena system handles allocation/deallocation transparently
;; Users don't need to think about memory management for most code
```

**Note:** Advanced users can use OALR operators (`owned`, `move`, `borrow`, `shared`) for explicit resource management, but this is optional.

### 2. Type System

**Scheme:** Dynamically typed only.

**Eshkol:** Gradual typing with optional annotations.

```scheme
;; Works in both Scheme and Eshkol (no annotations)
(define (add a b) (+ a b))

;; Eshkol-specific: Optional type annotations
(define (add : (-> integer integer integer))
  (lambda (a b) (+ a b)))

;; Type errors produce warnings, not compilation failures
```

### 3. Compilation Model

**Scheme:** Most implementations are interpreted or JIT-compiled.

**Eshkol:** Compiles to native code via LLVM.

```scheme
;; Same source code
(define (factorial n)
  (if (<= n 1) 1 (* n (factorial (- n 1)))))

;; Scheme: Interpreted or JIT (slower startup, flexible)
;; Eshkol: Native binary (faster execution, static deployment)
```

### 4. Numerical Semantics

**Scheme:** Supports exact/inexact number distinction.

**Eshkol:** Primarily uses IEEE 754 doubles for performance.

```scheme
;; Scheme
(/ 1 3)  ; Returns exact rational 1/3
(exact? (/ 1 3))  ; #t

;; Eshkol
(/ 1 3)  ; Returns 0.3333... (double)
;; Rational arithmetic available via symbolic mode
```

---

## Eshkol Extensions to Scheme

Extensions that make Eshkol unique for scientific computing:

### 1. Automatic Differentiation

```scheme
;; Symbolic differentiation
(diff (lambda (x) (* x x)) 'x)
;; Returns: (lambda (x) (* 2 x))

;; Numeric differentiation
(derivative (lambda (x) (* x x)) 3.0)
;; Returns: 6.0

;; Vector calculus
(gradient (lambda (v) (+ (* (vref v 0) (vref v 0))
                         (* (vref v 1) (vref v 1))))
          (vector 3.0 4.0))
;; Returns: #(6.0 8.0)
```

### 2. Tensors (Distinct from Scheme Vectors)

```scheme
;; Scheme vector (heterogeneous)
(define v (vector 1 "hello" #t))

;; Tensor (homogeneous, numeric, AD-aware)
(define t (tensor 1.0 2.0 3.0))
(tensor-dot t t)  ; 14.0
(tensor-add t t)  ; #(2.0 4.0 6.0)
```

### 3. Module System

```scheme
;; Load module by symbolic name
(require 'data.json)
(require 'core.list.transform)

;; Export symbols
(provide square cube factorial)
```

### 4. Pattern Matching

```scheme
;; Pattern matching (if implemented)
(match value
  ((cons x xs) (process-pair x xs))
  ('() (handle-empty))
  (_ (handle-other)))
```

### 5. Hygienic Macros (syntax-rules)

```scheme
(define-syntax when
  (syntax-rules ()
    ((when test body ...)
     (if test (begin body ...) #f))))

;; Macro expansion is hygienic (no variable capture)
```

---

## Migration Considerations

### Migrating Scheme Code to Eshkol

Most Scheme code runs in Eshkol without modifications:

**✅ Works as-is:**
- List processing
- Higher-order functions
- Recursion
- Closures
- Basic I/O
- Most R5RS procedures

**⚠️ Requires adaptation:**
- Code relying on exact arithmetic (use doubles or symbolic mode)
- Code using `call/cc` extensively (rewrite with exceptions or CPS)
- Code expecting specific GC behavior (Eshkol uses deterministic arena)

**Example: Fibonacci (works identically):**

```scheme
(define (fib n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))
```

**Example: List utilities (work identically):**

```scheme
(define (sum lst)
  (fold + 0 lst))

(define (average lst)
  (/ (sum lst) (length lst)))
```

### Using Eshkol Extensions

When porting Scheme code, consider leveraging Eshkol's extensions:

**1. Replace manual iteration with tensor operations:**

```scheme
;; Scheme approach
(define (dot-product v1 v2)
  (let ((sum 0))
    (for-each (lambda (i)
                (set! sum (+ sum (* (vector-ref v1 i)
                                    (vector-ref v2 i)))))
              (range 0 (vector-length v1)))
    sum))

;; Eshkol approach (if v1, v2 are tensors)
(define (dot-product t1 t2)
  (tensor-dot t1 t2))
```

**2. Add type annotations for documentation:**

```scheme
;; Before
(define (distance p1 p2)
  (sqrt (+ (* (- (car p1) (car p2)) (- (car p1) (car p2)))
           (* (- (cdr p1) (cdr p2)) (- (cdr p1) (cdr p2))))))

;; After (with types for clarity)
(define (distance : (-> (pair real real) (pair real real) real))
  (lambda (p1 p2)
    (sqrt (+ (* (- (car p1) (car p2)) (- (car p1) (car p2)))
             (* (- (cdr p1) (cdr p2)) (- (cdr p1) (cdr p2)))))))
```

**3. Use automatic differentiation:**

```scheme
;; Instead of manually computing derivatives
(define (f x) (* x x x))
(define (df x) (* 3 (* x x)))  ; Manual derivative

;; Use built-in AD
(define f (lambda (x) (* x x x)))
(define df (gradient f))  ; Automatic derivative
```

---

## Compatibility Testing

Test Scheme code compatibility using Eshkol's test suite structure:

```bash
# Run Scheme compatibility tests
./scripts/run_list_tests.sh      # List operations
./scripts/run_stdlib_tests.sh    # Standard library functions
./scripts/run_features_tests.sh  # Language features
```

**Test example:**

```scheme
;;; tests/scheme_compat_test.esk

;; Test R5RS list operations
(define (test-lists)
  (let* ((lst (list 1 2 3))
         (doubled (map (lambda (x) (* x 2)) lst))
         (evens (filter (lambda (x) (= (remainder x 2) 0)) doubled)))
    (display "Doubled: ") (display doubled) (newline)
    (display "Evens: ") (display evens) (newline)
    (display "Sum: ") (display (fold + 0 evens)) (newline)))

(test-lists)
;; Expected output:
;; Doubled: (2 4 6)
;; Evens: (2 4 6)
;; Sum: 12
```

---

## Known Limitations

**Compared to full Scheme implementations:**

1. **No dynamic code evaluation** - `eval` not currently implemented (homoiconicity provides partial alternative) <- coming very soon!
2. **No continuations** - `call/cc` defined but full implementation pending <- coming very soon!
3. **Limited exact arithmetic** - Eshkol optimizes for floating-point performance <- coming very soon!
4. **No garbage collector** - Arena allocation is deterministic but has different trade-offs

**These limitations reflect Eshkol's focus on scientific computing performance over general-purpose Scheme compatibility.**

---

## See Also

- [Language Overview](OVERVIEW.md) - Eshkol's design philosophy
- [Getting Started](GETTING_STARTED.md) - Installation and first programs  
- [Function Composition](FUNCTION_COMPOSITION.md) - Closures and higher-order functions
- [API Reference](../API_REFERENCE.md) - Complete function reference
