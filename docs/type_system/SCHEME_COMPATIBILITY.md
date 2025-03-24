# Scheme Compatibility in Eshkol

## Table of Contents

1. [Introduction](#introduction)
2. [Gradual Typing Approach](#gradual-typing-approach)
3. [Syntactic Compatibility](#syntactic-compatibility)
4. [Semantic Compatibility](#semantic-compatibility)
5. [Implementation Strategies](#implementation-strategies)
6. [Handling Scheme's Dynamic Features](#handling-schemes-dynamic-features)
7. [Migration Path for Existing Code](#migration-path-for-existing-code)
8. [Educational Approach](#educational-approach)
9. [Compatibility Testing](#compatibility-testing)
10. [Future Compatibility Considerations](#future-compatibility-considerations)

## Introduction

Eshkol extends Scheme with a powerful type system while maintaining full compatibility with standard Scheme. This document explains how Eshkol achieves this compatibility, allowing existing Scheme code to run unchanged while providing the benefits of static typing for new or gradually migrated code.

Maintaining Scheme compatibility is a core design principle of Eshkol. Rather than creating a new language inspired by Scheme, Eshkol is designed to be a proper superset of Scheme, where every valid Scheme program is also a valid Eshkol program. This approach allows for:

- Leveraging existing Scheme libraries and code
- Gradual adoption of typing in existing codebases
- Educational continuity for those familiar with Scheme
- Preserving the elegance and simplicity of Scheme

## Gradual Typing Approach

Eshkol implements gradual typing, which allows mixing typed and untyped code seamlessly:

### Optional Type Annotations

All type annotations in Eshkol are optional:

```scheme
;; Fully untyped (standard Scheme)
(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

;; Same function with type annotations
(: factorial (-> integer integer))
(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))
```

This allows existing Scheme code to run without modification while enabling gradual addition of types.

### Type Inference as a Bridge

Eshkol's type inference system serves as a bridge between untyped and typed code:

```scheme
;; Untyped function
(define (square x)
  (* x x))

;; Typed function that uses the untyped function
(: compute-area (-> float float))
(define (compute-area radius)
  (* pi (square radius)))  ; Type of square is inferred
```

The type inference system automatically determines the types of untyped code based on how it's used, allowing for seamless integration.

### Dynamic Type Fallback

When static typing is too restrictive, code can fall back to dynamic typing:

```scheme
;; Using the Any type for dynamic behavior
(: process-value (-> any any))
(define (process-value x)
  (cond
    ((number? x) (+ x 1))
    ((string? x) (string-append x "!"))
    ((list? x) (map process-value x))
    (else x)))
```

The `any` type allows for traditional Scheme-style dynamic typing when needed, providing an escape hatch for truly dynamic code.

## Syntactic Compatibility

Eshkol's type syntax is designed to be minimally invasive and compatible with standard Scheme:

### S-Expression Based Type Language

Type annotations use standard S-expressions rather than introducing new syntactic forms:

```scheme
;; Types are just S-expressions
(: identity (-> 'a 'a))
(define (identity x) x)

;; Complex types are still S-expressions
(: fold (-> (-> 'a 'b 'b) 'b (list 'a) 'b))
(define (fold f init lst)
  (if (null? lst)
      init
      (fold f (f (car lst) init) (cdr lst))))
```

This maintains Scheme's homoiconicity and ensures that type annotations can be processed by standard Scheme tools.

### Backward Compatible Extensions

New type-related forms are designed as natural extensions of Scheme:

```scheme
;; Type declarations look like normal Scheme definitions
(: map (-> (-> 'a 'b) (list 'a) (list 'b)))

;; Type class definitions use familiar define syntax
(define-type-class (Eq 'a)
  (= : (-> 'a 'a boolean)))

;; Type class instances also use familiar syntax
(define-type-class-instance (Eq string)
  (define (= s1 s2)
    (string=? s1 s2)))
```

These extensions follow Scheme's syntactic conventions, making them feel like natural parts of the language.

### Comments for Standard Scheme

In standard Scheme implementations, Eshkol's type annotations can be treated as comments or ignored:

```scheme
;; In standard Scheme, this is equivalent to:
;; (define (identity x) x)
(: identity (-> 'a 'a))
(define (identity x) x)
```

This allows Eshkol code to be processed by standard Scheme implementations with minimal preprocessing.

## Semantic Compatibility

Eshkol preserves Scheme's evaluation model and semantic guarantees:

### Preserving Scheme's Evaluation Model

Eshkol maintains Scheme's applicative order evaluation:

```scheme
;; Evaluation order is the same as in standard Scheme
(define (test x y)
  (if (> x 0)
      y
      (begin
        (display "Negative x")
        0)))

;; This behaves identically in Eshkol and Scheme
(test -1 (begin (display "Computing y") 42))
;; Prints: Negative x
```

The addition of types does not change the runtime behavior of correct programs.

### Proper Tail Recursion

Eshkol preserves Scheme's guarantee of proper tail recursion:

```scheme
;; Tail-recursive function
(: sum-list (-> (list number) number number))
(define (sum-list lst acc)
  (if (null? lst)
      acc
      (sum-list (cdr lst) (+ acc (car lst)))))

;; This works for arbitrarily long lists without stack overflow
(sum-list (make-long-list 1000000) 0)
```

The type system is designed to work with tail-recursive code without introducing any overhead that would break tail-call optimization.

### First-Class Functions and Closures

Eshkol fully supports Scheme's first-class functions and lexical closures:

```scheme
;; Higher-order function with lexical closure
(: make-adder (-> number (-> number number)))
(define (make-adder n)
  (lambda (x) (+ x n)))

;; Usage
(define add-5 (make-adder 5))
(add-5 10)  ; Returns 15
```

The type system accurately models closures and their captured environments.

### Macros and Metaprogramming

Eshkol preserves Scheme's powerful macro system:

```scheme
;; Standard Scheme macro
(define-syntax when
  (syntax-rules ()
    ((_ condition body ...)
     (if condition (begin body ...) (void)))))

;; Usage with typed code
(: test-when (-> boolean void))
(define (test-when flag)
  (when flag
    (display "Flag is true")
    (newline)))
```

Macros work seamlessly with typed code, maintaining Scheme's metaprogramming capabilities.

## Implementation Strategies

Eshkol uses several implementation strategies to maintain compatibility with Scheme:

### Type Erasure

Types are erased after checking, generating standard Scheme code:

```scheme
;; Typed Eshkol code
(: add-vectors (-> (vector float64) (vector float64) (vector float64)))
(define (add-vectors v1 v2)
  (vector-map + v1 v2))

;; After type erasure, becomes:
;; (define (add-vectors v1 v2)
;;   (vector-map + v1 v2))
```

This ensures that the generated code is standard Scheme code that can run in any Scheme implementation.

### Separate Compilation Phases

Eshkol separates type checking from code generation:

1. Parse Eshkol code
2. Perform type checking and inference
3. Erase types
4. Generate standard Scheme code
5. Compile or interpret the generated code

This separation allows for skipping type checking for pure Scheme code, maintaining compatibility.

### Interoperability with Scheme Libraries

Eshkol provides type definitions for standard Scheme procedures:

```scheme
;; Type definitions for standard Scheme procedures
(: + (-> number number number))
(: cons (-> 'a (list 'a) (list 'a)))
(: map (-> (-> 'a 'b) (list 'a) (list 'b)))
```

These type definitions allow typed Eshkol code to seamlessly use standard Scheme libraries.

## Handling Scheme's Dynamic Features

Eshkol provides typed interfaces to Scheme's dynamic features:

### Typing Dynamic Operations

Eshkol provides type-safe wrappers for dynamic operations:

```scheme
;; Type-safe eval
(: safe-eval (-> (syntax 'a) environment 'a))
(define (safe-eval expr env)
  (eval expr env))

;; Type-safe dynamic-wind
(: safe-dynamic-wind (-> (-> 'a) (-> 'b) (-> 'c) 'b))
(define (safe-dynamic-wind before thunk after)
  (dynamic-wind before thunk after))
```

These wrappers provide type safety while maintaining access to Scheme's dynamic features.

### Safety Boundaries

Eshkol establishes safety boundaries between typed and untyped regions:

```scheme
;; Import untyped code with a type annotation
(: quicksort (-> (list number) (list number)))
(define quicksort (unsafe-require "untyped-quicksort.scm"))

;; The boundary ensures that quicksort behaves according to its type
```

At these boundaries, runtime checks ensure that values flowing between typed and untyped code conform to their expected types.

### First-Class Continuations

Eshkol's type system is designed to work with call/cc and continuations:

```scheme
;; Typed call/cc
(: call/cc (-> (-> (-> 'a 'b) 'a) 'a))
(define call/cc call-with-current-continuation)

;; Usage
(: find-first (-> (list 'a) (-> 'a boolean) (maybe 'a)))
(define (find-first lst pred)
  (call/cc
   (lambda (return)
     (for-each (lambda (x)
                 (when (pred x)
                   (return (just x))))
               lst)
     (nothing))))
```

The type system accurately models the types of continuations, ensuring type safety across control flow jumps.

## Migration Path for Existing Code

Eshkol provides a clear migration path for existing Scheme code:

### Incremental Typing

Existing code can be typed incrementally:

1. Start with unmodified Scheme code
2. Add type annotations to function signatures
3. Add type annotations to key data structures
4. Gradually add more detailed types as needed

This allows for a smooth transition from untyped to typed code.

### Type Inference for Legacy Code

Eshkol's type inference can help with typing legacy code:

```scheme
;; Legacy untyped function
(define (process-data data)
  (map (lambda (x) (* x 2)) data))

;; Inferred type: (-> (list number) (list number))
```

The type inference system can automatically determine the types of existing code, making it easier to integrate with typed code.

### Compatibility Wrappers

For libraries that are difficult to type directly, compatibility wrappers can be used:

```scheme
;; Untyped library function
;; (define (complex-algorithm data) ...)

;; Typed wrapper
(: safe-complex-algorithm (-> (list number) (list number)))
(define (safe-complex-algorithm data)
  (let ((result (complex-algorithm data)))
    (if (list? result)
        (map (lambda (x)
               (if (number? x) x 0))
             result)
        '())))
```

These wrappers provide type safety for untyped code without requiring modifications to the original code.

## Educational Approach

Eshkol is designed to be teachable as "just Scheme" initially:

### Gradual Introduction of Types

Eshkol can be taught in stages:

1. Start with standard Scheme concepts
2. Introduce optional type annotations
3. Demonstrate the benefits of type checking
4. Gradually introduce more advanced type features

This allows for a smooth learning curve for students familiar with Scheme.

### Compatibility with Existing Materials

Eshkol is compatible with existing Scheme educational materials:

```scheme
;; Standard SICP example
(define (square x) (* x x))
(define (sum-of-squares x y)
  (+ (square x) (square y)))
(define (f a)
  (sum-of-squares (+ a 1) (* a 2)))

;; Works unchanged in Eshkol
```

Existing textbooks and examples can be used without modification, with types added as an enhancement.

### Type System as an Educational Tool

The type system itself can be an educational tool:

```scheme
;; Type error provides educational feedback
(: square (-> number number))
(define (square x) (* x x))

;; This would produce a helpful type error:
;; (square "hello")
```

Type errors can help students understand the expected types and behaviors of functions.

## Compatibility Testing

Eshkol includes comprehensive compatibility testing:

### R7RS Test Suite

Eshkol passes the R7RS test suite, ensuring compatibility with the Scheme standard:

```scheme
;; R7RS compliance tests
(define (run-r7rs-tests)
  (display "Running R7RS compliance tests...")
  (let ((results (run-tests r7rs-test-suite)))
    (display-results results)))
```

These tests ensure that Eshkol remains compatible with standard Scheme.

### Regression Testing

Eshkol includes regression tests to ensure that compatibility is maintained:

```scheme
;; Regression tests for compatibility
(define (run-compatibility-tests)
  (display "Running compatibility regression tests...")
  (let ((results (run-tests compatibility-test-suite)))
    (display-results results)))
```

These tests catch any changes that might break compatibility with existing Scheme code.

### Performance Benchmarks

Eshkol includes performance benchmarks to ensure that typed code performs well:

```scheme
;; Performance benchmarks
(define (run-performance-benchmarks)
  (display "Running performance benchmarks...")
  (let ((results (run-benchmarks performance-benchmark-suite)))
    (display-results results)))
```

These benchmarks ensure that the addition of types does not significantly impact performance.

## Future Compatibility Considerations

Eshkol is committed to maintaining Scheme compatibility in the future:

### R7RS-large Compatibility

Eshkol plans to support R7RS-large as it evolves:

```scheme
;; R7RS-large libraries
(import (scheme base)
        (scheme char)
        (scheme file)
        (scheme inexact)
        (scheme process-context)
        (scheme write)
        (scheme complex)
        (scheme time))
```

As new R7RS-large libraries are standardized, Eshkol will provide typed interfaces for them.

### Scheme Evolution

Eshkol will evolve alongside Scheme:

```scheme
;; Future Scheme features
(import (scheme r8rs))  ; Hypothetical future Scheme standard
```

As Scheme evolves, Eshkol will adapt to maintain compatibility with new features.

### Backward Compatibility

Eshkol is committed to backward compatibility:

```scheme
;; Code written for earlier versions of Eshkol
(import (eshkol compatibility))
```

As Eshkol evolves, it will maintain compatibility with code written for earlier versions.

## Related Documentation

- [TYPE_SYSTEM.md](TYPE_SYSTEM.md): Overview of Eshkol's type system
- [INFLUENCES.md](INFLUENCES.md): Influences on Eshkol's type system
- [SCIENTIFIC_COMPUTING_AND_AI.md](SCIENTIFIC_COMPUTING_AND_AI.md): How the type system enables scientific computing and AI
- [AUTODIFF.md](AUTODIFF.md): The synergy between the type system and automatic differentiation
