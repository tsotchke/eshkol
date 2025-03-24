# Eshkol Type System Documentation

This directory contains comprehensive documentation about Eshkol's type system, its design principles, capabilities, and implementation details.

## Overview

Eshkol extends Scheme with a powerful, flexible type system designed to enhance code safety, performance, and expressiveness while maintaining full compatibility with standard Scheme. The type system is a cornerstone of Eshkol's design, enabling high-precision scientific computing and AI capabilities while preserving the elegance and simplicity of Scheme.

## Documentation Files

- [TYPE_SYSTEM.md](TYPE_SYSTEM.md): Main documentation providing a comprehensive overview of Eshkol's type system, its design principles, and capabilities.
- [INFLUENCES.md](INFLUENCES.md): Detailed discussion of the languages and systems that influenced Eshkol's type system design.
- [SCIENTIFIC_COMPUTING_AND_AI.md](SCIENTIFIC_COMPUTING_AND_AI.md): How Eshkol's type system enables high-precision scientific computing and AI capabilities.
- [SCHEME_COMPATIBILITY.md](SCHEME_COMPATIBILITY.md): How Eshkol maintains full compatibility with standard Scheme while adding type system capabilities.
- [AUTODIFF.md](AUTODIFF.md): The synergy between Eshkol's type system and automatic differentiation capabilities.

## Key Features

1. **Gradual Typing**: Types in Eshkol are optional, allowing for incremental adoption and mixing of typed and untyped code.
2. **Type Inference**: Sophisticated type inference minimizes the need for explicit annotations.
3. **Three Typing Approaches**:
   - Implicit typing through type inference
   - Inline explicit typing with parameter annotations
   - Separate type declarations
4. **Scientific Computing Support**: Specialized types and features for scientific computing and AI.
5. **Scheme Compatibility**: Full compatibility with standard Scheme, preserving its semantics and features.
6. **Automatic Differentiation**: Deep integration with the type system for efficient and correct differentiation.

## Example Usage

### Implicit Typing

```scheme
;; No type annotations, but fully type-checked
(define (add x y)
  (+ x y))

;; The compiler infers that x and y must be numbers
;; and that the function returns a number
```

### Inline Explicit Typing

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

### Separate Type Declarations

```scheme
;; Separate type declaration
(: gradient-descent (-> function vector<float> number vector<float>))

;; Implementation without type annotations
(define (gradient-descent f initial-point learning-rate)
  (let ((gradient (autodiff-gradient f initial-point)))
    (v- initial-point (v* gradient learning-rate))))
```

## Related Documentation

- [../tutorials/TYPE_SYSTEM_TUTORIAL.md](../tutorials/TYPE_SYSTEM_TUTORIAL.md): Practical tutorial on using the type system (planned)
- [../reference/TYPE_SYSTEM_REFERENCE.md](../reference/TYPE_SYSTEM_REFERENCE.md): Comprehensive reference for all type-related syntax (planned)
