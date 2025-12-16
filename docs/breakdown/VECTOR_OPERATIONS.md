# Vectors and Tensors in Eshkol

**Status**: v1.0-architecture - Production-ready
**Last Updated**: December 2025

> This document describes the **actual implemented** vector and tensor systems based on source code verification.

---

## Table of Contents

- [Vectors and Tensors in Eshkol](#vectors-and-tensors-in-eshkol)
  - [Table of Contents](#table-of-contents)
  - [Overview: Two Types of Arrays](#overview-two-types-of-arrays)
  - [Scheme Vectors (Heterogeneous)](#scheme-vectors-heterogeneous)
    - [Creation](#creation)
    - [Access](#access)
    - [Storage Layout](#storage-layout)
    - [Operations](#operations)
  - [Tensors (Homogeneous Numeric)](#tensors-homogeneous-numeric)
    - [Creation](#creation-1)
    - [Access](#access-1)
    - [Storage Layout](#storage-layout-1)
    - [Linear Algebra Operations](#linear-algebra-operations)
    - [Automatic Differentiation with Tensors](#automatic-differentiation-with-tensors)
    - [Working Example: Neural Network Layer](#working-example-neural-network-layer)
  - [Performance Considerations](#performance-considerations)
    - [Memory Layout](#memory-layout)
    - [When to Use Each](#when-to-use-each)
  - [See Also](#see-also)

---

## Overview: Two Types of Arrays

Eshkol supports **two distinct array types**, each optimized for different use cases:

| Feature | **Scheme Vectors** | **Tensors** |
|---------|-------------------|-------------|
| **Element types** | Heterogeneous (any type) | Homogeneous (numeric only) |
| **Storage** | Array of 16-byte tagged values | Flat array of double bit patterns |
| **Creation** | `(vector ...)`, `make-vector` | `zeros`, `ones`, `eye`, `arange` |
| **Access** | `vector-ref`, `vector-set!` | `vref`, `tensor-get` |
| **Use case** | Mixed data, collections | Numeric computation, ML |
| **Autodiff** | No | **Yes** (AD-aware) |
| **Linear algebra** | No | **Yes** (`tensor-dot`, etc.) |

**Key Distinction**:
- **Scheme vectors** = Scheme's built-in heterogeneous arrays (like Python lists)
- **Tensors** = NumPy-style numeric arrays for scientific computing

---

## Scheme Vectors (Heterogeneous)

**Implementation**: [`lib/backend/collection_codegen.cpp`](../../lib/backend/collection_codegen.cpp:1195)

Scheme vectors can hold **any type** of element:

### Creation

```scheme
;; Vector literal
(define v (vector 1 "hello" 3.14 #t))

;; make-vector: create with fill value
(define v2 (make-vector 10 0))     ; 10 elements, all 0

;; vector-copy: duplicate
(define v3 (vector-copy v))
```

### Access

```scheme
;; vector-ref: get element at index
(vector-ref v 0)        ; → 1
(vector-ref v 1)        ; → "hello"
(vector-ref v 2)        ; → 3.14

;; vector-set!: mutate element
(vector-set! v 0 42)
(vector-ref v 0)        ; → 42

;; vector-length: get size
(vector-length v)       ; → 4
```

### Storage Layout

```
Scheme Vector Memory Layout:
┌─────────────┬──────────────┬──────────────┬──────────────┐
│ length (8)  │ elem0 (16)   │ elem1 (16)   │ elem2 (16)   │
└─────────────┴──────────────┴──────────────┴──────────────┘
              ↑                ↑                ↑
              Each element is a full eshkol_tagged_value_t
              Can hold ANY type (int, string, closure, etc.)
```

**Size**: 8 + (16 × num_elements) bytes

### Operations

**Scheme vectors are first-class Scheme values**:

```scheme
;; Can be elements in lists
(define mixed (list v 42 "test"))

;; Can be passed to functions
(define (process-vector v)
  (vector-ref v 0))

;; car/cdr work on vectors (returns first element)
(car v)  ; → same as (vector-ref v 0)
```

**NOT supported on Scheme vectors**:
- ❌ Linear algebra operations
- ❌ Automatic differentiation
- ❌ Element-wise arithmetic

For numeric computation, use **tensors** instead.

---

## Tensors (Homogeneous Numeric)

**Implementation**: [`lib/backend/tensor_codegen.cpp`](../../lib/backend/tensor_codegen.cpp:1) (3,041 lines)

Tensors are **N-dimensional numeric arrays** optimized for scientific computing:

### Creation

```scheme
;; zeros: create tensor filled with 0.0
(zeros 10)              ; 1D: shape [10]
(zeros (list 3 3))      ; 2D: shape [3, 3]
(zeros (list 2 3 4))    ; 3D: shape [2, 3, 4]

;; ones: create tensor filled with 1.0
(ones 5)
(ones (list 4 4))

;; eye: identity matrix
(eye 3)                 ; 3x3 identity

;; arange: range with step
(arange 0 10 0.5)       ; [0.0, 0.5, 1.0, ..., 9.5]

;; linspace: evenly spaced values
(linspace 0 1 100)      ; 100 points from 0 to 1

;; reshape: change dimensions (zero-copy)
(reshape (arange 9) 3 3)  ; 1D [9] → 2D [3, 3]
```

### Access

```scheme
;; vref: 1D access (AD-aware!)
(define v (vector 1.0 2.0 3.0))
(vref v 0)              ; → 1.0
(vref v 1)              ; → 2.0

;; tensor-get: N-D access
(define M (reshape (arange 9) 3 3))
(tensor-get M 0 0)      ; → 0.0 (element [0,0])
(tensor-get M 1 2)      ; → 5.0 (element [1,2])

;; Partial indexing returns slice (view, not copy)
(tensor-get M 1)        ; → 1D tensor [3.0, 4.0, 5.0] (row 1)
```

### Storage Layout

```
Tensor Memory Layout:
┌───────────────────────────────────────────────────────────┐
│ eshkol_tensor_t structure (32 bytes, cache-aligned)      │
├───────────────────────────────────────────────────────────┤
│ dimensions*    (8 bytes) → [dim0, dim1, ..., dimN]       │
│ num_dimensions (8 bytes)   Rank (1-4 typical)            │
│ elements*      (8 bytes) → flat array of int64 bit patterns │
│ total_elements (8 bytes)   Total element count           │
└───────────────────────────────────────────────────────────┘

Elements stored as int64 bit patterns of doubles (reinterpret_cast)
Row-major (C-style) ordering for cache efficiency
```

**Note:** Elements are stored as `int64_t` values that are bit patterns of `double` values. This enables storage in cons cells which use int64/double union.

### Linear Algebra Operations

```scheme
;; Element-wise operations
(tensor-add t1 t2)          ; t1 + t2 (element-wise)
(tensor-sub t1 t2)          ; t1 - t2 (element-wise)
(tensor-mul t1 t2)          ; t1 * t2 (element-wise, NOT matrix multiplication)
(tensor-div t1 t2)          ; t1 / t2 (element-wise)

;; Matrix operations
(tensor-dot A B)            ; Matrix multiplication / dot product
(tensor-transpose M)        ; Transpose matrix
(tensor-reshape t shape)    ; Change dimensions (zero-copy)

;; Reduction operations
(tensor-sum t)              ; Sum all elements
(tensor-mean t)             ; Average of all elements
(tensor-max t)              ; Maximum element
(tensor-min t)              ; Minimum element

;; Shape operations
(tensor-shape t)            ; Returns dimensions as vector
(tensor-rank t)             ; Returns number of dimensions
(tensor-size t)             ; Returns total number of elements
```

### Automatic Differentiation with Tensors

The `vref` operator is **AD-aware**: when used during gradient computation, it creates computational graph nodes:

```scheme
;; Define function using tensor access
(define (norm-squared v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (+ (* x x) (* y y))))

;; Compute gradient
(gradient norm-squared (vector 3.0 4.0))
;; Returns: #(6.0 8.0)
;; Because ∂/∂x(x²+y²) = 2x = 6.0
;; And     ∂/∂y(x²+y²) = 2y = 8.0

;; vref creates AD nodes when in gradient context
;; But is a simple memory access in normal execution
```

### Working Example: Neural Network Layer

```scheme
;; Forward pass through dense layer
(define (dense-layer inputs weights biases)
  (let ((z (tensor-dot inputs weights)))  ; Linear transformation
    (tensor-add z biases)))                 ; Add bias

;; Activation function (applied element-wise)
(define (relu x)
  (if (> x 0.0) x 0.0))

;; Compute loss gradient
(define (layer-gradient inputs targets weights biases)
  (gradient (lambda (w)
              (let* ((z (dense-layer inputs w biases))
                     (activated (tensor-map relu z))
                     (loss (mse-loss activated targets)))
                loss))
            weights))
```

---

## Performance Considerations

### Memory Layout

**Scheme vectors:**
- Cache-unfriendly for numeric computation (16-byte tagged values)
- Type-flexible (can mix types)
- Suitable for: data structures, mixed collections

**Tensors:**
- Cache-friendly (contiguous doubles)
- SIMD-vectorizable
- Suitable for: linear algebra, numerical algorithms, ML

### When to Use Each

Use **Scheme vectors** when:
- Elements have different types
- Implementing data structures (stacks, queues)
- Building collections of mixed values

Use **tensors** when:
- All elements are numeric
- Performing linear algebra
- Computing gradients
- Optimizing for cache/SIMD

---

## See Also

- [Automatic Differentiation](AUTODIFF.md) - How `vref` creates AD nodes
- [Type System](TYPE_SYSTEM.md) - HEAP_SUBTYPE_VECTOR vs. HEAP_SUBTYPE_TENSOR
- [Memory Management](MEMORY_MANAGEMENT.md) - Object headers on vectors/tensors
- [API Reference](../API_REFERENCE.md) - Complete tensor operation reference
