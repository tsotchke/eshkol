# Scheme Compatibility in Eshkol

## Table of Contents

- [Overview](#overview)
- [R7RS-small Compliance Summary](#r7rs-small-compliance-summary)
- [Core Language Forms](#core-language-forms)
- [Standard Procedures](#standard-procedures)
- [Continuation Semantics](#continuation-semantics)
- [Numeric Tower](#numeric-tower)
- [Promises](#promises)
- [Eval and Environments](#eval-and-environments)
- [Module System](#module-system)
- [Macro System](#macro-system)
- [Records](#records)
- [Bytevectors](#bytevectors)
- [Tail Call Optimization](#tail-call-optimization)
- [Eshkol Extensions Beyond R7RS](#eshkol-extensions-beyond-r7rs)
- [Known Limitations](#known-limitations)
- [Migration Guide](#migration-guide)
- [Compatibility Testing](#compatibility-testing)

---

## Overview

Eshkol is a **compiled Scheme-family language** that implements approximately 95% of the R7RS-small standard while extending it with automatic differentiation, tensor computing, a consciousness engine, GPU acceleration, and other scientific computing capabilities. The compiler translates R7RS programs to native code via LLVM, producing standalone binaries with no runtime interpreter overhead.

The R7RS-small standard defines 244 standard procedures and ~30 special forms. Eshkol implements 232 of these procedures and all essential special forms. The missing procedures are concentrated in a small set of error introspection functions (`error-object?`, `error-object-message`, `error-object-irritants`), one arithmetic function (`exact-integer-sqrt`), and the R7RS library system syntax (`define-library`/`import`/`export`), which is replaced by the `require`/`provide` module system.

Most well-formed R7RS Scheme programs compile and run in Eshkol without modification.

**Implementation references:**
- Parser: [parser.cpp](lib/frontend/parser.cpp) (7,540 lines)
- Code generation: [llvm_codegen.cpp](lib/backend/llvm_codegen.cpp) (34,928 lines)
- Type checker: [type_checker.cpp](lib/types/type_checker.cpp) (1,999 lines)

---

## R7RS-small Compliance Summary

### Section-by-Section Compliance Matrix

| R7RS Section | Feature | Status | Notes |
|:---|:---|:---:|:---|
| **4.1** Primitive expressions | `quote`, `lambda`, `if`, `set!`, `include` | ✅ | Full support |
| **4.2.1** Conditionals | `cond`, `case`, `and`, `or`, `when`, `unless` | ✅ | Including `=>` clause in `cond` |
| **4.2.2** Binding | `let`, `let*`, `letrec`, `letrec*`, `let-values`, `let*-values` | ✅ | `letrec*` with correct R7RS semantics |
| **4.2.3** Sequencing | `begin` | ✅ | |
| **4.2.4** Iteration | `do`, named `let` | ✅ | `do` at [llvm_codegen.cpp:15408](lib/backend/llvm_codegen.cpp#L15408) |
| **4.2.5** Delayed evaluation | `delay`, `delay-force`, `force`, `make-promise`, `promise?` | ✅ | Full iterative forcing; see [Promises](#promises) |
| **4.2.6** Dynamic bindings | `make-parameter`, `parameterize` | ✅ | Macro-transformed at parse time |
| **4.2.7** Exception handling | `guard`, `raise`, `raise-continuable` | ⚠️ | `guard`/`raise` full; `raise-continuable` missing |
| **4.2.8** Quasiquotation | `quasiquote`, `unquote`, `unquote-splicing` | ✅ | |
| **4.2.9** Case-lambda | `case-lambda` | ✅ | Macro-transformed to variadic dispatch |
| **4.3** Macros | `define-syntax`, `syntax-rules`, `let-syntax`, `letrec-syntax` | ✅ | Hygienic; `syntax-case` not supported |
| **5.1** Programs | Top-level expressions | ✅ | |
| **5.2** Import | `import` | ⚠️ | Uses `require` instead; see [Module System](#module-system) |
| **5.3** Variable definitions | `define`, `define-values` | ✅ | Internal defines → `letrec*` |
| **5.4** Syntax definitions | `define-syntax` | ✅ | |
| **5.5** Record type definitions | `define-record-type` | ✅ | See [Records](#records) |
| **6.1** Equivalence | `eqv?`, `eq?`, `equal?` | ✅ | |
| **6.2** Numbers | Full numeric tower | ✅ | See [Numeric Tower](#numeric-tower) |
| **6.3** Booleans | `not`, `boolean?`, `boolean=?` | ✅ | |
| **6.4** Pairs and lists | `cons` through `assoc`/`member` | ✅ | 20+ procedures |
| **6.5** Symbols | `symbol?`, `symbol->string`, `string->symbol`, `symbol=?` | ✅ | |
| **6.6** Characters | 18 character procedures | ✅ | Including `char-upcase`, `char-downcase`, `char-foldcase` |
| **6.7** Strings | 18+ string procedures | ✅ | Including `string-upcase`, `string-downcase`, `string-for-each`, `string-map` |
| **6.8** Vectors | 10 vector procedures | ✅ | Including `vector-for-each`, `vector-map`, `vector-fill!` |
| **6.9** Bytevectors | 9 bytevector procedures | ✅ | See [Bytevectors](#bytevectors) |
| **6.10** Control | `procedure?`, `apply`, `map`, `for-each`, `call/cc`, `values`, `dynamic-wind` | ✅ | See [Continuation Semantics](#continuation-semantics) |
| **6.11** Exceptions | `with-exception-handler`, `raise`, `error` | ✅ | `raise-continuable` missing |
| **6.12** Environments and eval | `eval`, `interaction-environment`, `scheme-report-environment` | ✅ | See [Eval and Environments](#eval-and-environments) |
| **6.13** I/O | Ports, read, write, display | ✅ | 27+ I/O procedures |
| **6.14** System interface | `features`, `command-line`, `exit` | ✅ | |

### Compliance Statistics

- **Special forms and derived expressions**: 28/30 (93%) — missing `define-library`, `import`
- **Standard procedures**: 232/244 (95%) — missing 12 procedures
- **Overall R7RS-small compliance**: ~95%

---

## Core Language Forms

### Primitive Expression Types (R7RS 4.1)

All primitive expression types are fully supported:

```scheme
;; Variable reference
x

;; Literal (self-evaluating)
42  3.14  #t  #f  "hello"  #\a  #(1 2 3)

;; Procedure call
(+ 1 2)
(f x y)

;; Lambda expression
(lambda (x) (* x x))
(lambda (x y . rest) rest)          ; variadic

;; Conditional
(if (> x 0) "positive" "non-positive")

;; Assignment
(set! x 42)

;; Quotation
(quote (a b c))
'(a b c)
```

### Derived Expression Types (R7RS 4.2)

```scheme
;; Conditionals
(cond ((> x 0) "positive")
      ((= x 0) "zero")
      (else "negative"))

(cond ((assv x alist) => cdr))      ; => clause supported

(case (car pair)
  ((a e i o u) "vowel")
  ((w y) "semivowel")
  (else "consonant"))

(and x y z)                          ; short-circuit
(or x y z)
(when (> x 0) (display x))
(unless (= x 0) (display "nonzero"))

;; Binding constructs
(let ((x 1) (y 2)) (+ x y))
(let* ((x 1) (y (+ x 1))) y)
(letrec ((even? (lambda (n) (if (= n 0) #t (odd? (- n 1)))))
         (odd?  (lambda (n) (if (= n 0) #f (even? (- n 1))))))
  (even? 10))
(letrec* ((x 1) (y (+ x 1))) y)     ; sequential binding

;; Named let (iteration)
(let loop ((i 0) (acc 0))
  (if (= i 10) acc
      (loop (+ i 1) (+ acc i))))

;; Multiple return values
(let-values (((a b) (values 1 2)))
  (+ a b))

;; do loop
(do ((i 0 (+ i 1))
     (sum 0 (+ sum i)))
    ((= i 10) sum))

;; Sequencing
(begin (display "hello") (newline))

;; Case-lambda (multi-arity dispatch)
(define f
  (case-lambda
    ((x) (* x x))
    ((x y) (+ x y))
    ((x y z) (+ x y z))))

;; Dynamic parameters
(define p (make-parameter 10))
(parameterize ((p 20))
  (p))                               ; => 20

;; cond-expand (feature-based conditional compilation)
(cond-expand
  (eshkol (display "Running on Eshkol"))
  (else (display "Unknown implementation")))
```

### Definitions (R7RS 5)

```scheme
;; Variable definition
(define x 42)

;; Function definition (shorthand)
(define (square x) (* x x))

;; Variadic function
(define (f x y . rest) (apply + x y rest))

;; Internal definitions (transformed to letrec*)
(define (process data)
  (define (helper x) (* x 2))        ; internal define
  (define threshold 10)               ; internal define
  (map helper (filter (lambda (x) (> x threshold)) data)))
```

**Implementation note:** Internal definitions are transformed to `letrec*` by the parser ([parser.cpp:1171](lib/frontend/parser.cpp#L1171)). Only consecutive defines at the start of a body are collected; once a non-define expression appears, subsequent defines are treated as expressions. This matches R7RS semantics and prevents side-effect reordering.

---

## Standard Procedures

### Numeric Operations (R7RS 6.2)

The full R7RS numeric tower is implemented with 31+ procedures:

```scheme
;; Arithmetic
(+ 1 2)  (- 5 3)  (* 2 3)  (/ 10 3)
(abs -5)                             ; => 5
(quotient 10 3)                      ; => 3
(remainder 10 3)                     ; => 1
(modulo 10 3)                        ; => 1

;; Comparison
(= 1 1)  (< 1 2)  (> 2 1)  (<= 1 1)  (>= 2 1)

;; Min/max
(min 1 2 3)                          ; => 1
(max 1 2 3)                          ; => 3

;; Rounding
(floor 3.7)                          ; => 3.0
(ceiling 3.2)                        ; => 4.0
(round 3.5)                          ; => 4.0 (banker's rounding)
(truncate 3.7)                       ; => 3.0

;; Transcendental functions
(sqrt 16)                            ; => 4.0
(expt 2 10)                          ; => 1024
(exp 1)                              ; => 2.718281828...
(log 2.718281828)                    ; => ~1.0
(log 8 2)                            ; => 3.0 (log base 2)
(sin 0)  (cos 0)  (tan 0)
(asin 1)  (acos 0)  (atan 1)  (atan 1 1)

;; Type predicates
(number? 42)                         ; => #t
(integer? 42)                        ; => #t
(real? 3.14)                         ; => #t
(rational? 1/3)                      ; => #t
(complex? 3+4i)                      ; => #t
(exact? 42)                          ; => #t
(inexact? 3.14)                      ; => #t
(zero? 0)  (positive? 1)  (negative? -1)
(odd? 3)  (even? 4)

;; Exactness conversion
(exact->inexact 1)                   ; => 1.0
(inexact->exact 1.5)                 ; => 3/2 (rational)
(exact 1.0)                          ; => 1
(inexact 1)                          ; => 1.0

;; GCD/LCM
(gcd 12 8)                           ; => 4
(lcm 4 6)                            ; => 12

;; String conversion
(number->string 42)                  ; => "42"
(number->string 255 16)              ; => "ff"
(string->number "42")                ; => 42
(string->number "3.14")              ; => 3.14
(string->number "ff" 16)             ; => 255
```

### Pairs and Lists (R7RS 6.4)

```scheme
;; Construction
(cons 1 2)                           ; => (1 . 2)
(list 1 2 3)                        ; => (1 2 3)

;; Access
(car '(1 2 3))                       ; => 1
(cdr '(1 2 3))                       ; => (2 3)
(caar '((1 2) 3))                    ; => 1
(cadr '(1 2 3))                      ; => 2
;; ... all c[ad]{2,4}r combinations

;; Predicates
(pair? '(1 . 2))                     ; => #t
(null? '())                          ; => #t
(list? '(1 2 3))                     ; => #t

;; Operations
(length '(1 2 3))                    ; => 3
(append '(1 2) '(3 4))              ; => (1 2 3 4)
(reverse '(1 2 3))                   ; => (3 2 1)
(list-ref '(a b c) 1)               ; => b
(list-tail '(a b c d) 2)            ; => (c d)

;; Higher-order
(map (lambda (x) (* x x)) '(1 2 3))          ; => (1 4 9)
(for-each display '(1 2 3))                    ; prints 123
(filter (lambda (x) (> x 2)) '(1 2 3 4))     ; => (3 4)
(fold + 0 '(1 2 3))                            ; => 6
(apply + '(1 2 3))                             ; => 6

;; Search
(member 2 '(1 2 3))                 ; => (2 3)
(assoc 'b '((a 1) (b 2) (c 3)))    ; => (b 2)
```

### Characters (R7RS 6.6)

```scheme
(char? #\a)                          ; => #t
(char=? #\a #\a)                     ; => #t
(char<? #\a #\b)                     ; => #t
(char->integer #\A)                  ; => 65
(integer->char 65)                   ; => #\A
(char-alphabetic? #\a)               ; => #t
(char-numeric? #\0)                  ; => #t
(char-whitespace? #\space)           ; => #t
(char-upper-case? #\A)               ; => #t
(char-lower-case? #\a)               ; => #t
(char-upcase #\a)                    ; => #\A
(char-downcase #\A)                  ; => #\a
(char-foldcase #\A)                  ; => #\a
```

### Strings (R7RS 6.7)

```scheme
(string? "hello")                    ; => #t
(string-length "hello")              ; => 5
(string-ref "hello" 1)              ; => #\e
(string-append "hello" " " "world") ; => "hello world"
(substring "hello" 1 3)             ; => "el"
(string=? "abc" "abc")              ; => #t
(string<? "abc" "abd")              ; => #t

;; Mutation
(let ((s (string-copy "hello")))
  (string-set! s 0 #\H) s)          ; => "Hello"
(let ((s (make-string 5 #\x)))
  (string-fill! s #\o) s)           ; => "ooooo"

;; Conversion
(string->list "abc")                 ; => (#\a #\b #\c)
(list->string '(#\a #\b #\c))       ; => "abc"
(string->number "42")                ; => 42
(number->string 42)                  ; => "42"

;; Case operations
(string-upcase "hello")              ; => "HELLO"
(string-downcase "HELLO")            ; => "hello"

;; Iteration
(string-for-each (lambda (c) (display (char-upcase c))) "hello")
;; prints HELLO

(string-map char-upcase "hello")     ; => "HELLO"

;; Copy with range
(string-copy "hello" 1 3)           ; => "el"
(let ((s (string-copy "abcde")))
  (string-copy! s 1 "xyz" 0 2) s)   ; => "axyde"
```

### Vectors (R7RS 6.8)

```scheme
(vector 1 2 3)                       ; => #(1 2 3)
(make-vector 5 0)                    ; => #(0 0 0 0 0)
(vector-ref #(a b c) 1)             ; => b
(vector-length #(1 2 3))            ; => 3

;; Mutation
(let ((v (vector 1 2 3)))
  (vector-set! v 1 99) v)           ; => #(1 99 3)

(let ((v (make-vector 3 0)))
  (vector-fill! v 7) v)             ; => #(7 7 7)

;; Iteration
(vector-for-each (lambda (x) (display x)) #(1 2 3))  ; prints 123
(vector-map (lambda (x) (* x x)) #(1 2 3))           ; => #(1 4 9)

;; Conversion
(vector->list #(a b c))             ; => (a b c)
(list->vector '(a b c))             ; => #(a b c)
```

### I/O and Ports (R7RS 6.13)

```scheme
;; Output
(display "hello")                    ; prints hello
(write '(1 "two" 3))               ; prints (1 "two" 3)
(newline)                            ; prints newline

;; File I/O
(define port (open-input-file "data.txt"))
(read port)                          ; reads S-expression
(read-char port)                     ; reads character
(read-line port)                     ; reads line as string
(peek-char port)                     ; peeks without consuming
(close-input-port port)

(define out (open-output-file "out.txt"))
(write "hello" out)
(write-char #\newline out)
(close-output-port out)

;; String ports
(define sp (open-input-string "(+ 1 2)"))
(read sp)                            ; => (+ 1 2)

;; Port predicates
(port? port)                         ; => #t
(input-port? port)                   ; => #t
(output-port? out)                   ; => #t
(eof-object? (read-char port))       ; #t at end-of-file

;; call-with-port (auto-close)
(call-with-input-file "data.txt"
  (lambda (port) (read port)))
```

---

## Continuation Semantics

### call/cc (R7RS 6.10)

Eshkol implements `call-with-current-continuation` (aliased as `call/cc`) using an exception-based model built on `setjmp`/`longjmp`.

**Implementation:** [llvm_codegen.cpp:14243](lib/backend/llvm_codegen.cpp#L14243) (`codegenCallCC`)

```scheme
;; Basic continuation capture
(call/cc (lambda (k) (k 42)))        ; => 42

;; Escaping from nested computation
(define (find-first pred lst)
  (call/cc (lambda (return)
    (for-each (lambda (x)
      (when (pred x) (return x)))
      lst)
    #f)))

(find-first even? '(1 3 5 4 7))     ; => 4

;; Non-local exit from deep recursion
(call/cc (lambda (exit)
  (let loop ((n 1000000))
    (if (= n 0) "done"
        (if (= n 500000) (exit "early!")
            (loop (- n 1)))))))       ; => "early!"
```

**Semantics:**
- Continuations are **single-shot**: invoking a captured continuation more than once is not supported. This is a deliberate design choice — single-shot continuations via `setjmp`/`longjmp` have zero overhead on the normal (non-escape) path, whereas multi-shot continuations require copying the entire call stack.
- The continuation object is a CALLABLE heap value with `HEAP_SUBTYPE_CONTINUATION`.
- Continuation invocation triggers `longjmp` back to the capture point, unwinding the stack.

**Comparison with other Scheme implementations:**
- Racket/Chez Scheme: Full multi-shot continuations (can invoke the same continuation multiple times)
- Gambit/Chicken: Full multi-shot via stack copying
- Eshkol: Single-shot via setjmp/longjmp (matching the common use case of non-local exit)

For most practical Scheme patterns (early return, exception handling, coroutine-like constructs), single-shot continuations are sufficient. Programs requiring true coroutines or backtracking should use explicit state machines or the consciousness engine's logic programming facilities.

### dynamic-wind (R7RS 6.10)

`dynamic-wind` installs before/after thunks that execute on entry to and exit from a dynamic extent, including non-local exits via continuations.

**Implementation:** [llvm_codegen.cpp:14354](lib/backend/llvm_codegen.cpp#L14354) (`codegenDynamicWind`)

```scheme
;; Resource cleanup on non-local exit
(let ((port (open-input-file "data.txt")))
  (dynamic-wind
    (lambda () (display "entering\n"))
    (lambda ()
      (call/cc (lambda (k) (k "escaped!"))))
    (lambda ()
      (close-input-port port)
      (display "cleanup done\n"))))
;; prints: entering
;; prints: cleanup done
;; => "escaped!"
```

The wind stack tracks nested `dynamic-wind` scopes. When a continuation is invoked, the runtime unwinds the current stack (executing "after" thunks) and rewinds to the continuation's saved stack (executing "before" thunks).

### guard/raise (R7RS 4.2.7)

R7RS exception handling with `cond`-style clauses:

```scheme
;; Exception handling with guard
(guard (exn
        ((string? exn) (string-append "Error: " exn))
        ((number? exn) (string-append "Code: " (number->string exn)))
        (else "Unknown error"))
  (raise "file not found"))
;; => "Error: file not found"

;; with-exception-handler (low-level)
(with-exception-handler
  (lambda (exn) (display "caught: ") (display exn) (newline))
  (lambda () (raise "test")))

;; error procedure (R7RS 6.11)
(error "invalid argument" 'foo 42)
;; Raises an error with message and irritants
```

**Implementation:** `guard` is implemented as a parser transformation that wraps the body in a continuation capture. The guard clauses are evaluated in the dynamic environment of the guard expression, and if no clause matches, the exception is re-raised.

---

## Numeric Tower

Eshkol implements the full R7RS numeric tower with five numeric types, listed in order of increasing generality:

```
int64 → bignum → rational → double → complex
 (exact)  (exact)   (exact)  (inexact)  (inexact)
```

### Type Hierarchy

**int64** (64-bit signed integer): The default exact integer representation. Values in the range [-2^63, 2^63-1] are stored as immediate values in the tagged value's 64-bit data field.

**bignum** (arbitrary-precision integer): When an integer operation overflows int64, the result is automatically promoted to a bignum. Bignums are heap-allocated sign-magnitude representations with limb arrays. Operations dispatch through `eshkol_bignum_binary_tagged` and `eshkol_bignum_compare_tagged` runtime functions.

```scheme
(expt 2 100)                         ; => 1267650600228229401496703205376 (bignum)
(* 9999999999999999999 9999999999999999999)  ; automatic promotion
```

**rational** (exact fraction): Pairs of bignums representing numerator/denominator, always maintained in reduced form via GCD. Comparison uses cross-multiplication to avoid precision loss.

```scheme
(/ 1 3)                              ; => 1/3 (exact rational)
(+ 1/3 1/6)                          ; => 1/2
(exact? (/ 1 3))                     ; => #t
```

**double** (IEEE 754 64-bit float): The default inexact representation. Stored as int64 bit patterns in the tagged value data field.

```scheme
(exact->inexact 1/3)                 ; => 0.3333333333333333
(inexact? 3.14)                      ; => #t
```

**complex** (complex number): Heap-allocated struct containing two doubles (real and imaginary parts). Type tag `ESHKOL_VALUE_COMPLEX` (7). Division uses Smith's formula to avoid overflow.

```scheme
(make-rectangular 3 4)               ; => 3+4i
(magnitude 3+4i)                     ; => 5.0
(angle 0+1i)                         ; => 1.5707963267948966
(real-part 3+4i)                     ; => 3.0
(imag-part 3+4i)                     ; => 4.0
```

### Exactness Semantics

R7RS requires that exact operations on exact arguments produce exact results. Eshkol tracks exactness via the `ESHKOL_FLAG_EXACT` bit in the tagged value's flags field:

```scheme
;; Exact + exact = exact
(+ 1 2)                              ; => 3 (exact)
(* 1/3 3)                            ; => 1 (exact)

;; Exact + inexact = inexact (R7RS rule)
(+ 1 1.0)                            ; => 2.0 (inexact)
(+ 1/3 0.5)                          ; => 0.8333... (inexact)

;; Explicit conversion
(exact 1.5)                          ; => 3/2
(inexact 1/3)                        ; => 0.3333...
```

---

## Promises

Eshkol implements the full R7RS promise system (R7RS 4.2.5) with memoization and iterative forcing.

**Implementation:** [llvm_codegen.cpp:10360-10466](lib/backend/llvm_codegen.cpp#L10360)

Promises are heap-allocated objects (`HEAP_SUBTYPE_PROMISE = 18`) with a 40-byte structure:

```
[forced:i64 | thunk:tagged_value | cached:tagged_value]
```

```scheme
;; Basic lazy evaluation
(define p (delay (begin (display "computing...") 42)))
(force p)                            ; prints "computing...", => 42
(force p)                            ; => 42 (memoized, no recomputation)

;; make-promise (already-forced)
(define q (make-promise 99))
(promise? q)                         ; => #t
(force q)                            ; => 99

;; delay-force (iterative forcing, prevents stack overflow)
(define (stream-take n s)
  (if (= n 0) '()
      (cons (force (car s))
            (stream-take (- n 1) (force (cdr s))))))

(define (integers-from n)
  (cons (delay n) (delay-force (integers-from (+ n 1)))))
```

The parser desugars `delay` to `(%make-lazy-promise (lambda () expr))` and `delay-force` to `(%make-lazy-promise-force (lambda () expr))` ([parser.cpp:1894-1925](lib/frontend/parser.cpp#L1894)). The `delay-force` form enables iterative (non-recursive) forcing, preventing stack overflow on deeply chained promises.

---

## Eval and Environments

Eshkol implements `eval` as a compiler-level builtin that compiles and executes S-expressions at runtime.

**Implementation:** [llvm_codegen.cpp:16811](lib/backend/llvm_codegen.cpp#L16811) (`codegenEval`)

```scheme
;; Basic eval
(eval '(+ 1 2))                      ; => 3

;; Eval with environment
(eval '(+ 1 2) (interaction-environment))  ; => 3
(eval '(if #t "yes" "no") (scheme-report-environment 7))  ; => "yes"

;; Dynamic code construction and evaluation
(define expr (list '* 6 7))
(eval expr)                           ; => 42

;; Null environment (syntax only, no bindings)
(eval '(if #t 1 2) (null-environment 7))  ; => 1
```

### Environment Functions

| Function | Description |
|:---|:---|
| `(interaction-environment)` | Returns the current environment with all user-defined bindings |
| `(scheme-report-environment version)` | Returns an environment with all R7RS standard bindings (232 procedures) |
| `(null-environment version)` | Returns an environment with only syntax keywords (no procedure bindings) |

**Implementation details:**
- `interaction-environment` ([llvm_codegen.cpp:16967](lib/backend/llvm_codegen.cpp#L16967)) walks the symbol table to build an environment containing all currently-visible bindings.
- `scheme-report-environment` ([llvm_codegen.cpp:16856](lib/backend/llvm_codegen.cpp#L16856)) constructs an environment with the 232 R7RS standard procedures.
- `null-environment` ([llvm_codegen.cpp:16850](lib/backend/llvm_codegen.cpp#L16850)) creates an empty environment containing only syntax keywords.

The runtime functions `eshkol_eval` and `eshkol_eval_env` handle the actual compilation and execution of the evaluated expression. Because Eshkol is a compiled language, `eval` involves a full compilation step (parse → type-check → codegen → execute), which makes it slower than in interpreted Scheme implementations but produces optimized native code.

---

## Module System

Eshkol uses `require`/`provide` instead of R7RS `define-library`/`import`/`export`. This is a deliberate design choice: the `require`/`provide` system integrates with precompiled stdlib objects and LLVM's `LinkOnceODRLinkage` for efficient linking.

```scheme
;; Import a module
(require stdlib)                      ; entire standard library
(require core.list.transform)         ; specific sub-module
(require math)                        ; math library
(require signal)                      ; signal processing

;; Export symbols
(provide my-function my-variable)
```

### Precompiled Stdlib

The standard library is compiled to a single object file (`stdlib.o`) using the `--shared-lib` flag. All symbols use `LinkOnceODRLinkage`, which means:
- If a user defines a function with the same name, the user's definition wins (External linkage overrides LinkOnceODR).
- No duplicate symbol errors at link time.
- The linker keeps exactly one copy of each function.

Module discovery is automatic: `collect_all_submodules()` recursively discovers all modules in the stdlib by parsing source files, so new stdlib directories work without hardcoded prefix checks.

**Stdlib directories:** `core/`, `math/`, `signal/`, `ml/`, `random/`, `web/`, `tensor/`, `quantum/`

---

## Macro System

Eshkol implements R7RS hygienic macros via `syntax-rules` pattern matching ([macro_expander.cpp](lib/frontend/macro_expander.cpp), 861 lines).

```scheme
;; Pattern-based macros
(define-syntax my-if
  (syntax-rules ()
    ((my-if test then else)
     (cond (test then) (#t else)))))

;; Ellipsis patterns
(define-syntax my-and
  (syntax-rules ()
    ((my-and) #t)
    ((my-and x) x)
    ((my-and x rest ...)
     (if x (my-and rest ...) #f))))

;; Local macros
(let-syntax ((swap! (syntax-rules ()
  ((swap! a b)
   (let ((tmp a)) (set! a b) (set! b tmp))))))
  (let ((x 1) (y 2))
    (swap! x y)
    (list x y)))                      ; => (2 1)
```

**Hygiene:** The macro expander maintains a symbol table that renames identifiers introduced by macros to avoid capture. This means macros are safe to use in any context without variable name collisions.

**Supported forms:**
- `define-syntax` — top-level macro definition
- `syntax-rules` — pattern-based transformer with `...` (ellipsis) for repetition
- `let-syntax` — local macro bindings
- `letrec-syntax` — recursive local macro bindings

**Not supported:** `syntax-case` (low-level macro system). Only `syntax-rules` is available. This covers the vast majority of practical macro patterns. Programs using `syntax-case` will need to be rewritten using `syntax-rules` or Eshkol's other metaprogramming facilities (homoiconic S-expression manipulation).

---

## Records

R7RS `define-record-type` is fully supported. The parser transforms record definitions into vector operations at parse time.

```scheme
(define-record-type <point>
  (make-point x y)
  point?
  (x point-x set-point-x!)
  (y point-y set-point-y!))

(define p (make-point 3.0 4.0))
(point? p)                           ; => #t
(point-x p)                          ; => 3.0
(set-point-y! p 5.0)
(point-y p)                          ; => 5.0

;; Records work with higher-order functions
(define points (list (make-point 1 2) (make-point 3 4) (make-point 5 6)))
(map point-x points)                 ; => (1 3 5)
```

**Implementation:** The parser recognizes `define-record-type` and generates:
- A constructor function that creates a tagged vector with a type discriminator
- A predicate that checks the type discriminator
- Accessor functions mapped to `vector-ref` at the appropriate index
- Mutator functions mapped to `vector-set!` at the appropriate index

---

## Bytevectors

Full R7RS 6.9 bytevector support:

```scheme
(define bv (make-bytevector 4 0))
(bytevector-u8-set! bv 0 72)         ; 'H'
(bytevector-u8-set! bv 1 101)        ; 'e'
(bytevector-u8-set! bv 2 108)        ; 'l'
(bytevector-u8-set! bv 3 108)        ; 'l'
(bytevector-u8-ref bv 0)             ; => 72
(bytevector-length bv)               ; => 4
(bytevector? bv)                     ; => #t

;; Copy operations
(bytevector-copy bv)                  ; new copy
(bytevector-copy bv 1 3)             ; sub-range copy
(let ((dest (make-bytevector 4 0)))
  (bytevector-copy! dest 0 bv 0 4))  ; copy into dest

;; Append
(bytevector-append bv bv)            ; => 8-byte vector

;; String conversion
(utf8->string bv)                    ; => "Hell"
(string->utf8 "Hello")              ; => bytevector of UTF-8 bytes
```

---

## Tail Call Optimization

Eshkol implements proper tail calls as required by R7RS. Functions in tail position are compiled to loop-back branches rather than recursive calls, preventing stack overflow.

**Implementation:** The `TailCallContext` in [binding_codegen.h:283](inc/eshkol/backend/binding_codegen.h#L283) tracks whether a call is in tail position. When TCO is active, `codegenCall` emits a branch back to the function entry point instead of a call instruction.

```scheme
;; This runs in O(1) stack space
(define (loop n)
  (if (= n 0) "done"
      (loop (- n 1))))

(loop 10000000)                       ; => "done" (no stack overflow)

;; Mutual tail recursion
(define (even? n)
  (if (= n 0) #t (odd? (- n 1))))

(define (odd? n)
  (if (= n 0) #f (even? (- n 1))))

(even? 1000000)                       ; => #t (no stack overflow)

;; Named let with TCO
(let loop ((i 0))
  (if (= i 10000000) i
      (loop (+ i 1))))               ; => 10000000

;; letrec with TCO (nested letrec contexts save/restore correctly)
(letrec ((f (lambda (n acc)
              (if (= n 0) acc
                  (f (- n 1) (+ acc n))))))
  (f 1000000 0))                     ; => 500000500000
```

**Stack size:** For deeply recursive non-tail calls, Eshkol provides a 512MB stack via linker flags. The maximum recursion depth defaults to 100,000 and can be overridden via the `ESHKOL_STACK_SIZE` environment variable.

---

## Eshkol Extensions Beyond R7RS

Eshkol extends R7RS with capabilities designed for scientific computing, machine learning, and cognitive architectures. These extensions use standard Scheme syntax and do not conflict with R7RS programs.

### Automatic Differentiation

Three modes of automatic differentiation integrated into the compiler:

```scheme
;; Symbolic differentiation (compile-time AST transformation)
(diff (lambda (x) (* x x x)) 'x)

;; Forward-mode (dual numbers)
(derivative (lambda (x) (* x x)) 3.0)   ; => 6.0

;; Reverse-mode (computational graph)
(gradient (lambda (v) (+ (* (vref v 0) (vref v 0))
                         (* (vref v 1) (vref v 1))))
          #(3.0 4.0))                     ; => #(6.0 8.0)
```

### Tensors (Distinct from Scheme Vectors)

Homogeneous numeric arrays optimized for SIMD, GPU, and automatic differentiation:

```scheme
;; Scheme vector (heterogeneous, 16 bytes/element)
(define v (vector 1 "hello" #t))

;; Eshkol tensor (homogeneous doubles, 8 bytes/element, AD-aware)
(define t #(1.0 2.0 3.0))
(tensor-dot t t)                      ; => 14.0
(tensor-add t t)                      ; => #(2.0 4.0 6.0)
(reshape t #(3 1))                    ; => 3x1 matrix
```

### Machine Learning (75+ Builtins)

Compiler-level ML operations with SIMD acceleration and GPU dispatch:

```scheme
(require ml)
(relu #(-1.0 0.0 1.0))              ; => #(0.0 0.0 1.0)
(softmax #(1.0 2.0 3.0))            ; => #(0.09 0.24 0.67)
(adam-step params grads 0.001 m v t)  ; Adam optimizer step
(conv2d input kernel 1 1)            ; 2D convolution
```

### Consciousness Engine (22 Builtins)

Logic programming, active inference with factor graphs, and global workspace theory:

```scheme
;; Logic programming
(define s (make-substitution))
(define s2 (unify ?x 42 s))
(walk ?x s2)                          ; => 42

;; Active inference (factor graph belief propagation)
(define fg (make-factor-graph 2 #(2 2)))
(fg-add-factor! fg #(0) #(-0.356 -1.204))
(fg-infer! fg 20)                     ; belief propagation

;; Global workspace
(define ws (make-workspace 4 3))
(ws-register! ws "perception" (lambda (c) (cons 0.9 #(1.0 0.0 0.0 0.0))))
(ws-step! ws)                         ; softmax competition
```

### Signal Processing

FFT, window functions, and digital filters:

```scheme
(require signal)
(define spectrum (fft signal))
(define windowed (apply-window signal (hann-window (vector-length signal))))
(define filtered (fir-filter signal (butterworth-lowpass 4 0.2)))
```

### Parallel Computing

Work-stealing thread pool with parallel higher-order functions:

```scheme
(parallel-map (lambda (x) (* x x)) data)
(parallel-fold + 0 data)
(parallel-filter even? data)
```

### GPU Acceleration

Automatic dispatch to Metal GPU or cBLAS based on cost model:

```scheme
(tensor-dot A B)                      ; auto-dispatches: SIMD → cBLAS → GPU
```

### Web Platform (WASM)

Compile to WebAssembly with 73 DOM/Canvas/Event API bindings:

```scheme
(require web)
(define btn (web-create-element "button"))
(web-set-text-content btn "Click me")
(web-add-event-listener btn "click"
  (lambda (e) (web-alert "Clicked!")))
(web-append-child (web-get-body) btn)
```

### Pattern Matching

```scheme
(match value
  ((cons x xs) (process x xs))
  ('() "empty")
  (42 "the answer")
  (_ "other"))
```

### Eval (Full R7RS)

```scheme
(eval '(+ 1 2))                       ; => 3
(eval '(map car '((1 2) (3 4)))
      (scheme-report-environment 7))   ; => (1 3)
```

---

## Known Limitations

### Genuinely Missing R7RS Features

| Feature | R7RS Section | Status | Notes |
|:---|:---:|:---:|:---|
| `raise-continuable` | 6.11 | Missing | `raise` works; continuable variant not needed for most programs |
| `error-object?` | 6.11 | Missing | Error objects are strings; introspection API not yet implemented |
| `error-object-message` | 6.11 | Missing | Use string operations on the error value directly |
| `error-object-irritants` | 6.11 | Missing | |
| `exact-integer-sqrt` | 6.2.6 | Missing | Use `(inexact->exact (floor (sqrt n)))` as workaround |
| `define-library` | 5.6 | Missing | Use `require`/`provide` module system |
| `import` (R7RS syntax) | 5.2 | Missing | Use `require` |
| `export` (R7RS syntax) | 5.6 | Missing | Use `provide` |
| Multi-shot continuations | 6.10 | Limited | `call/cc` is single-shot (setjmp/longjmp) |
| `syntax-case` | — | Missing | Only `syntax-rules` supported (sufficient for most macros) |
| `char-ci=?` etc. | 6.6 | Missing | Case-insensitive character comparison |

### Semantic Differences

**Memory model:** Eshkol uses arena allocation instead of garbage collection. For most programs this is transparent, but programs that create cyclic data structures may accumulate memory until the arena is reset. Long-running programs with complex object graphs should use explicit arena management.

**Tail calls:** Properly implemented. Mutual tail recursion, named-let loops, and letrec tail calls all run in O(1) stack space.

**Number semantics:** The full numeric tower is implemented. However, by default, arithmetic operations may use double precision when mixing exact and inexact numbers (R7RS: exact + inexact = inexact). Use explicit `exact` conversions to maintain precision.

---

## Migration Guide

### From R5RS/R7RS Scheme to Eshkol

**Works without modification:**

- All list processing (cons, car, cdr, map, filter, fold, etc.)
- Higher-order functions and closures
- Recursion (with proper tail calls)
- String and character operations
- Vector operations
- Basic I/O (display, write, newline, read)
- Macros via define-syntax/syntax-rules
- Numeric computation (exact and inexact)
- Exception handling (guard/raise)
- Continuations (call/cc for non-local exit)
- Dynamic binding (parameterize)
- Promises (delay/force)
- eval with environments
- do loops
- Records (define-record-type)

**Requires minor adaptation:**

| Pattern | Scheme | Eshkol |
|:---|:---|:---|
| Module import | `(import (scheme base))` | `(require stdlib)` |
| Module export | `(export foo bar)` | `(provide foo bar)` |
| Library definition | `(define-library ...)` | File with `(provide ...)` |
| Multi-shot continuation | `(call/cc (lambda (k) (k 1) (k 2)))` | Use explicit state; second `(k 2)` will not execute |
| Error introspection | `(error-object-message exn)` | Treat `exn` as string directly |

### Leveraging Eshkol Extensions

When porting Scheme code to Eshkol, consider enhancing it with Eshkol's scientific computing capabilities:

```scheme
;; Replace manual numerical differentiation with AD
;; Before (manual):
(define (df f x) (/ (- (f (+ x 0.0001)) (f x)) 0.0001))

;; After (automatic, exact):
(define df (gradient f))

;; Replace manual iteration with tensor operations
;; Before:
(define (dot-product v1 v2)
  (fold + 0 (map * (vector->list v1) (vector->list v2))))

;; After:
(define (dot-product t1 t2) (tensor-dot t1 t2))

;; Replace manual parallelism with parallel primitives
;; Before:
(map expensive-computation data)

;; After:
(parallel-map expensive-computation data)
```

---

## Compatibility Testing

Eshkol includes comprehensive test suites verifying Scheme compatibility:

```bash
# Core language features
./scripts/run_features_tests.sh       # Special forms, control flow
./scripts/run_list_tests.sh           # List operations
./scripts/run_stdlib_tests.sh         # Standard library procedures
./scripts/run_string_tests.sh         # String operations
./scripts/run_closures_tests.sh       # Closures and higher-order functions
./scripts/run_tco_tests.sh            # Tail call optimization
./scripts/run_macros_tests.sh         # Macro expansion
./scripts/run_control_flow_tests.sh   # Continuations, exceptions
./scripts/run_bignum_tests.sh         # Exact arithmetic
./scripts/run_rational_tests.sh       # Rational numbers
./scripts/run_complex_tests.sh        # Complex numbers
./scripts/run_io_tests.sh             # I/O and ports

# Run all 35 test suites (438 tests)
./scripts/run_all_tests.sh
```

**Example compatibility test:**

```scheme
;;; tests/features/r7rs_compat_test.esk

;; Test R7RS numeric tower
(display (exact? (/ 1 3))) (newline)          ; #t
(display (inexact? 3.14)) (newline)           ; #t
(display (rational? 1/3)) (newline)           ; #t
(display (= (+ 1/3 1/6) 1/2)) (newline)      ; #t

;; Test continuations
(display (call/cc (lambda (k) (k 42)))) (newline)  ; 42

;; Test promises
(let ((p (delay (+ 1 2))))
  (display (force p)) (newline)                ; 3
  (display (force p)) (newline))               ; 3 (memoized)

;; Test eval
(display (eval '(+ 1 2))) (newline)            ; 3

;; Test guard/raise
(display (guard (exn (#t "caught"))
           (raise "error"))) (newline)         ; caught

;; Test case-lambda
(let ((f (case-lambda
           ((x) (* x x))
           ((x y) (+ x y)))))
  (display (f 5)) (newline)                    ; 25
  (display (f 3 4)) (newline))                 ; 7

;; Test define-record-type
(define-record-type <pair-record>
  (make-pair-record fst snd) pair-record?
  (fst pair-record-fst) (snd pair-record-snd))
(let ((p (make-pair-record 1 2)))
  (display (pair-record-fst p)) (newline)      ; 1
  (display (pair-record-snd p)) (newline))     ; 2
```

---

## See Also

- [Language Overview](OVERVIEW.md) — Design philosophy and architecture
- [Continuations](CONTINUATIONS.md) — Deep dive on call/cc, dynamic-wind, guard/raise
- [Exact Arithmetic](EXACT_ARITHMETIC.md) — Numeric tower implementation details
- [Module System](MODULE_SYSTEM.md) — require/provide, precompiled stdlib
- [Function Composition](FUNCTION_COMPOSITION.md) — Closures and higher-order functions
- [API Reference](../API_REFERENCE.md) — Complete function reference
- [Getting Started](GETTING_STARTED.md) — Installation and first programs
