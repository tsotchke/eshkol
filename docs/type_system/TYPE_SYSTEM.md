# Eshkol Type System

## Table of Contents

1. [Introduction](#introduction)
2. [Rationale](#rationale)
3. [Type System Design Principles](#type-system-design-principles)
4. [The Three Typing Approaches](#the-three-typing-approaches)
   - [Implicit Typing](#implicit-typing)
   - [Inline Explicit Typing](#inline-explicit-typing)
   - [Separate Type Declarations](#separate-type-declarations)
5. [Type System and Scheme Compatibility](#type-system-and-scheme-compatibility)
6. [Scientific Computing and AI Capabilities](#scientific-computing-and-ai-capabilities)
7. [Advanced Type Features](#advanced-type-features)
8. [Implementation Details](#implementation-details)
9. [Future Directions](#future-directions)

## Introduction

Eshkol extends Scheme with a powerful, flexible type system designed to enhance code safety, performance, and expressiveness while maintaining full compatibility with standard Scheme. This document provides a comprehensive overview of Eshkol's type system, its design principles, and its capabilities.

The type system is a cornerstone of Eshkol's design, enabling high-precision scientific computing and AI capabilities while preserving the elegance and simplicity of Scheme. It represents a significant departure from traditional Scheme's dynamic typing approach, but does so in a way that respects and extends Scheme's philosophy rather than replacing it.

## Rationale

### Why Add Types to Scheme?

Scheme is renowned for its elegant minimalism, powerful macro system, and clean functional programming model. However, in scientific computing and AI domains, the lack of static typing can lead to several challenges:

1. **Runtime Errors**: Type errors discovered only at runtime can be costly, especially in long-running scientific computations.
2. **Performance Limitations**: Dynamic typing imposes performance overhead and limits certain optimizations.
3. **Code Comprehension**: Large scientific codebases benefit from the documentation aspect of types.
4. **Specialized Domains**: Scientific computing often requires specialized types (vectors, matrices, tensors) with strict rules.

Eshkol's type system addresses these challenges while preserving Scheme's strengths:

- **Safety**: Catch errors at compile time rather than runtime
- **Performance**: Enable aggressive optimizations based on type information
- **Expressiveness**: Provide domain-specific types for scientific computing
- **Compatibility**: Maintain full compatibility with standard Scheme

### Historical Context

Eshkol's type system draws inspiration from several sources:

- **ML Family Languages**: Hindley-Milner type inference that allows for strong typing without explicit annotations
- **Haskell**: Type classes for ad-hoc polymorphism and higher-kinded types
- **Typed Racket/Scheme**: Gradual typing approach that allows mixing typed and untyped code
- **TypeScript**: Structural typing and declaration files for typing existing code
- **Julia**: Type parameterization and multiple dispatch for scientific computing
- **Scientific Computing Languages**: Matrix-oriented design from MATLAB/Octave, vectorized operations from NumPy/SciPy

By synthesizing these influences, Eshkol creates a type system uniquely suited to scientific computing within a Scheme framework.

## Type System Design Principles

Eshkol's type system is guided by several core principles:

### Gradual Typing

Types in Eshkol are optional. Code can be fully typed, partially typed, or completely untyped. This allows for:

- Incremental adoption in existing codebases
- Typing only performance-critical or error-prone sections
- Maintaining the rapid prototyping benefits of dynamic typing

### Type Inference

Eshkol employs sophisticated type inference to minimize the need for explicit annotations:

- Local type inference within function bodies
- Global type inference across module boundaries
- Contextual type inference based on usage

### Performance Without Compromise

The type system is designed to enable performance optimizations without sacrificing Scheme's semantics:

- Type-directed optimizations for numerical code
- Specialized implementations for typed data structures
- Elimination of runtime type checks where possible

### Safety Guarantees

Types provide strong safety guarantees:

- Prevention of common errors like type mismatches
- Dimensional analysis for scientific computations
- Bounds checking for array accesses

### Expressiveness

The type system is designed to be expressive enough for complex scientific domains:

- Higher-order function types
- Parametric polymorphism
- Type-level computation where necessary

## The Three Typing Approaches

Eshkol supports three complementary approaches to typing, giving programmers flexibility in how they express and enforce types.

### Implicit Typing

Implicit typing relies on Eshkol's type inference system to automatically determine types based on how values are used. This approach requires no type annotations, making it ideal for rapid prototyping or when migrating existing Scheme code.

```scheme
;; No type annotations, but fully type-checked
(define (add x y)
  (+ x y))

;; The compiler infers that x and y must be numbers
;; and that the function returns a number
```

Benefits of implicit typing:

- **Clean, concise code** that looks like standard Scheme
- **No annotation overhead** for straightforward code
- **Seamless integration** with existing Scheme codebases

Limitations:

- Type errors may be reported far from their source
- Complex polymorphic functions may require annotations
- Less self-documenting than explicit approaches

### Inline Explicit Typing

Inline explicit typing allows programmers to annotate parameters and return types directly in the function definition. This provides clear documentation and ensures type safety at the definition site.

```scheme
;; Explicitly typed parameters and return type
(define (add-integers x : number y : number) : number
  (+ x y))

;; Vector operations with explicit types
(define (compute-distance point1 : vector<float> point2 : vector<float>) : float
  (let ((x1 (vector-ref point1 0))
        (y1 (vector-ref point1 1))
        (x2 (vector-ref point2 0))
        (y2 (vector-ref point2 1)))
    (sqrt (+ (expt (- x2 x1) 2) (expt (- y2 y1) 2)))))
```

Benefits of inline explicit typing:

- **Self-documenting code** that clearly states expectations
- **Localized type errors** at the definition site
- **IDE support** for autocompletion and tooltips

### Separate Type Declarations

Separate type declarations allow programmers to specify types separately from the implementation. This is particularly useful for complex type signatures or when adding types to existing code without modifying it.

```scheme
;; Separate type declaration
(: gradient-descent (-> function vector<float> number vector<float>))

;; Implementation without type annotations
(define (gradient-descent f initial-point learning-rate)
  (let ((gradient (autodiff-gradient f initial-point)))
    (v- initial-point (v* gradient learning-rate))))
```

Benefits of separate type declarations:

- **Clean separation** of types from implementation
- **Complex type signatures** without cluttering function definitions
- **Adding types** to existing code without modification
- **Library interfaces** can be typed separately from implementation

## Type System and Scheme Compatibility

Eshkol maintains full compatibility with standard Scheme while adding type system capabilities. This section details how this compatibility is achieved.

### Syntactic Compatibility

Type annotations in Eshkol follow Scheme's parenthesized syntax and are designed to be ignored by standard Scheme implementations:

- Type annotations use standard S-expressions
- No mandatory type declarations that would break Scheme parsers
- Type syntax designed to be treated as comments or ignored by standard Scheme

### Semantic Compatibility

Eshkol preserves Scheme's evaluation model and semantic guarantees:

- Applicative order evaluation is maintained
- Proper tail recursion is preserved
- First-class functions and closures work as expected
- Macros and metaprogramming capabilities are retained

### Implementation Strategy

Eshkol uses type erasure to maintain runtime compatibility with Scheme:

- Types are checked at compile time and erased before execution
- No runtime type information unless explicitly requested
- Generated code is standard Scheme code

### Handling Scheme's Dynamic Features

Eshkol provides typed interfaces to Scheme's dynamic features:

- Typed wrappers for `eval` and other dynamic operations
- Safety boundaries between typed and untyped regions
- Type-aware macro system

## Scientific Computing and AI Capabilities

Eshkol's type system enables powerful scientific computing and AI capabilities. For a detailed discussion, see [SCIENTIFIC_COMPUTING_AND_AI.md](SCIENTIFIC_COMPUTING_AND_AI.md).

Key capabilities include:

- **Numerical Precision**: Static distinction between different floating-point precisions
- **Vector and Matrix Operations**: Statically typed linear algebra with dimension checking
- **Automatic Differentiation**: Type-directed automatic differentiation for machine learning
- **Performance Optimizations**: Type-directed compilation for high-performance computing

## Advanced Type Features

Eshkol's type system includes several advanced features for expressing complex type relationships:

### Parametric Polymorphism

Functions can be parameterized over types, allowing for generic code:

```scheme
;; A function that works with any type of vector
(: map-vector (-> (-> 'a 'b) vector<'a> vector<'b>))
(define (map-vector f vec)
  (let* ((len (vector-length vec))
         (result (make-vector len)))
    (do ((i 0 (+ i 1)))
        ((= i len) result)
      (vector-set! result i (f (vector-ref vec i))))))
```

### Type Classes/Traits

Eshkol supports type classes (similar to Haskell) or traits (similar to Rust) for ad-hoc polymorphism:

```scheme
;; Define a Numeric type class
(define-type-class (Numeric 'a)
  (+ : (-> 'a 'a 'a))
  (- : (-> 'a 'a 'a))
  (* : (-> 'a 'a 'a))
  (/ : (-> 'a 'a 'a)))

;; A function that works with any Numeric type
(: sum (-> (vector<'a>) 'a) (where (Numeric 'a)))
(define (sum vec)
  (let ((result (vector-ref vec 0)))
    (do ((i 1 (+ i 1)))
        ((= i (vector-length vec)) result)
      (set! result (+ result (vector-ref vec i))))))
```

### Dependent Types

For advanced scientific computing, Eshkol supports limited forms of dependent types:

```scheme
;; A vector type parameterized by its length
(: dot-product (-> (vector<float, 'n>) (vector<float, 'n>) float))
(define (dot-product v1 v2)
  (let ((sum 0.0))
    (do ((i 0 (+ i 1)))
        ((= i (vector-length v1)) sum)
      (set! sum (+ sum (* (vector-ref v1 i) (vector-ref v2 i)))))))
```

## Implementation Details

Eshkol's type system is implemented through several components:

### Type Inference

The type inference system uses a variant of the Hindley-Milner algorithm with extensions for:

- Row polymorphism for structural typing
- Effect typing for tracking side effects
- Subtyping for object-oriented features

### Type Checking

The type checker verifies that operations are type-safe:

- Function application checks
- Operator overloading resolution
- Subtyping checks
- Type class instance resolution

### Code Generation

The code generator uses type information to produce optimized code:

- Specialized numeric operations
- Unboxed primitive types
- Inlined function calls
- SIMD vectorization

## Future Directions

Eshkol's type system continues to evolve. Future directions include:

- **Refinement Types**: Types that carry logical predicates
- **Linear Types**: For resource management and optimization
- **Effect System**: For tracking and controlling side effects
- **Quantum Computing Types**: For quantum algorithm development

For more details on future plans, see [TYPE_SYSTEM_FUTURE.md](TYPE_SYSTEM_FUTURE.md).

## Related Documentation

- [INFLUENCES.md](INFLUENCES.md): Detailed discussion of influences on Eshkol's type system
- [SCIENTIFIC_COMPUTING_AND_AI.md](SCIENTIFIC_COMPUTING_AND_AI.md): How the type system enables scientific computing and AI
- [SCHEME_COMPATIBILITY.md](SCHEME_COMPATIBILITY.md): Detailed discussion of Scheme compatibility
- [AUTODIFF.md](AUTODIFF.md): The synergy between the type system and automatic differentiation
- [TYPE_SYSTEM_TUTORIAL.md](../tutorials/TYPE_SYSTEM_TUTORIAL.md): Practical tutorial on using the type system
- [TYPE_SYSTEM_REFERENCE.md](../reference/TYPE_SYSTEM_REFERENCE.md): Comprehensive reference for all type-related syntax
