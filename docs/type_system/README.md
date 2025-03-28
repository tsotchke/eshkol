# Type System Documentation

This directory contains documentation related to Eshkol's type system, its design principles, capabilities, and implementation details.

## Overview

Eshkol's type system is designed to provide the benefits of static typing while maintaining the flexibility and expressiveness of Scheme. It supports gradual typing, allowing developers to add type annotations incrementally as needed.

## Documents

### [Type System Overview](TYPE_SYSTEM.md)
Comprehensive overview of Eshkol's type system, including its design principles, features, and implementation details.

### [Influences](INFLUENCES.md)
Languages and systems that influenced Eshkol's type system, including Typed Racket, Haskell, OCaml, TypeScript, and others.

### [Scientific Computing and AI](SCIENTIFIC_COMPUTING_AND_AI.md)
How Eshkol's type system enables scientific computing and AI applications, including vector operations, automatic differentiation, and more.

### [Scheme Compatibility](SCHEME_COMPATIBILITY.md)
How Eshkol maintains compatibility with Scheme while adding a type system, including gradual typing, type inference, and more.

### [Automatic Differentiation](AUTODIFF.md)
The synergy between Eshkol's type system and automatic differentiation, including type-directed automatic differentiation and more.

### [MCP Tools for Autodiff](MCP_TOOLS_FOR_AUTODIFF.md)
Documentation for MCP tools available for analyzing autodiff functions, including tools for analyzing type inference, lambda captures, and binding access.

## Key Features

### Gradual Typing

Eshkol's type system supports gradual typing, allowing developers to add type annotations incrementally as needed. This provides the benefits of static typing while maintaining the flexibility and expressiveness of Scheme.

```scheme
;; Untyped function
(define (square x)
  (* x x))

;; Typed function
(: cube (-> number number))
(define (cube x)
  (* x x x))
```

### Type Inference

Eshkol's type system includes a powerful type inference system that can infer types for most expressions, reducing the need for explicit type annotations.

```scheme
;; Type inference
(define (add1 x) (+ x 1))
;; Inferred type: (-> number number)
```

### Polymorphic Types

Eshkol's type system supports polymorphic types, allowing functions to work with multiple types.

```scheme
;; Polymorphic function
(: identity (forall (a) (-> a a)))
(define (identity x) x)
```

### Type Classes

Eshkol's type system includes type classes, allowing for ad-hoc polymorphism and operator overloading.

```scheme
;; Type class for equality
(define-type-class (Eq a)
  (== : (-> a a boolean)))

;; Instance for numbers
(define-instance (Eq number)
  (define (== x y) (= x y)))
```

### Dependent Types

Eshkol's type system includes limited support for dependent types, allowing types to depend on values.

```scheme
;; Dependent type for vectors
(: vector-ref (forall (a n i) (-> (Vector a n) (Index i n) a)))
```

### Scientific Computing Types

Eshkol's type system includes specialized types for scientific computing, including vectors, matrices, tensors, and more.

```scheme
;; Vector type
(: v (Vector float64 3))
(define v (vector 1.0 2.0 3.0))

;; Matrix type
(: m (Matrix float64 2 3))
(define m (matrix [[1.0 2.0 3.0] [4.0 5.0 6.0]]))
```

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

### Automatic Differentiation Types

Eshkol's type system includes specialized types for automatic differentiation, including dual numbers, gradients, and more.

```scheme
;; Automatic differentiation
(: f (-> float64 float64))
(define (f x) (* x x))

(: df/dx (-> float64 float64))
(define df/dx (autodiff-gradient f))
```

```scheme
;; Separate type declaration
(: gradient-descent (-> function vector<float> number vector<float>))

;; Implementation without type annotations
(define (gradient-descent f initial-point learning-rate)
  (let ((gradient (autodiff-gradient f initial-point)))
    (v- initial-point (v* gradient learning-rate))))
```

## MCP Tools for Type System Analysis

Eshkol provides several MCP tools for analyzing the type system:

- **[analyze-types](MCP_TOOLS_FOR_AUTODIFF.md#analyze-types)**: Analyzes type inference and type checking for Eshkol files
- **[analyze-lambda-captures](MCP_TOOLS_FOR_AUTODIFF.md#analyze-lambda-captures)**: Analyzes closure environments and variable captures
- **[analyze-binding-lifetime](MCP_TOOLS_FOR_AUTODIFF.md#analyze-binding-lifetime)**: Tracks binding creation and destruction
- **[analyze-binding-access](MCP_TOOLS_FOR_AUTODIFF.md#analyze-binding-access)**: Examines how bindings are used
- **[visualize-binding-flow](MCP_TOOLS_FOR_AUTODIFF.md#visualize-binding-flow)**: Tracks binding values through transformation stages

These tools can help identify issues with type inference, lambda captures, and other aspects of the type system. See the [MCP Tools for Autodiff](MCP_TOOLS_FOR_AUTODIFF.md) document for more information.

## Related Documentation

- [Type System Tutorial](../tutorials/TYPE_SYSTEM_TUTORIAL.md): A practical guide to using Eshkol's type system
- [Type System Reference](../reference/TYPE_SYSTEM_REFERENCE.md): A comprehensive reference for all type-related syntax and semantics
