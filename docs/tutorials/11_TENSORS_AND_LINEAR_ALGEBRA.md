# Tutorial 11: Tensors and Linear Algebra

Eshkol's tensor system provides N-dimensional arrays of doubles with
automatic SIMD vectorisation and optional GPU dispatch.

---

## Creating Tensors

```scheme
;; Tensor literal (homogeneous doubles, 8 bytes per element)
(define v #(1.0 2.0 3.0 4.0 5.0))

;; From functions
(define z (zeros 3 3))         ;; 3x3 zero matrix
(define o (ones 4))            ;; 4-element ones vector
(define I (eye 3))             ;; 3x3 identity matrix
(define r (arange 10))         ;; #(0 1 2 3 4 5 6 7 8 9)
(define ls (linspace 0.0 1.0 5)) ;; #(0.0 0.25 0.5 0.75 1.0)
(define rnd (rand 3 3))        ;; 3x3 random uniform [0,1)
```

**Tensors vs Vectors:** `#(1 2 3)` creates a tensor (homogeneous doubles,
8 bytes each). `(vector 1 "two" #t)` creates a heterogeneous vector
(tagged values, 16 bytes each). Use tensors for numeric computation.

---

## Element Access and Mutation

```scheme
(define v #(10 20 30 40 50))

(display (vector-ref v 2))    ;; => 30
(vector-set! v 0 99)
(display v)                   ;; => #(99 20 30 40 50)
(display (vector-length v))   ;; => 5
```

---

## Reshaping

```scheme
(define flat #(1 2 3 4 5 6))
(define M (reshape flat 2 3))   ;; 2x3 matrix
(display (tensor-shape M))      ;; => (2 3)
(define col (flatten M))        ;; back to 1D
```

---

## Element-Wise Operations

```scheme
(define a #(1.0 2.0 3.0))
(define b #(4.0 5.0 6.0))

(display (tensor-add a b))     ;; => #(5.0 7.0 9.0)
(display (tensor-sub a b))     ;; => #(-3.0 -3.0 -3.0)
(display (tensor-mul a b))     ;; => #(4.0 10.0 18.0)
(display (tensor-div a b))     ;; => #(0.25 0.4 0.5)
(display (tensor-scale a 2.0)) ;; => #(2.0 4.0 6.0)
```

---

## Reductions

```scheme
(define v #(1.0 2.0 3.0 4.0 5.0))

(display (tensor-sum v))   ;; => 15.0
(display (tensor-mean v))  ;; => 3.0
(display (tensor-min v))   ;; => 1.0
(display (tensor-max v))   ;; => 5.0
```

---

## Linear Algebra

```scheme
;; Dot product
(display (tensor-dot #(1 2 3) #(4 5 6)))  ;; => 32

;; Matrix multiplication — auto-dispatches SIMD -> BLAS -> GPU
(define A (reshape #(1 2 3 4) 2 2))
(define B (reshape #(5 6 7 8) 2 2))
(define C (matmul A B))
(display C)  ;; => 2x2 result

;; Transpose
(display (transpose A))

;; Norm, trace
(display (norm #(3.0 4.0)))   ;; => 5.0
(display (trace (eye 4)))     ;; => 4.0

;; Outer product
(display (outer #(1 2) #(3 4)))  ;; => 2x2 matrix
```

---

## GPU Acceleration

When tensor sizes exceed a threshold, operations automatically dispatch
to GPU (Metal on macOS, CUDA on Linux/Windows):

```scheme
;; Explicit GPU dispatch
(define result (gpu-matmul A B))
(define soft (gpu-softmax large-vector))

;; The cost-model dispatcher chooses automatically:
;; Small tensors → SIMD (SSE/AVX/NEON)
;; Medium tensors → cBLAS
;; Large tensors → Metal/CUDA
(define big-result (matmul large-A large-B))  ;; auto-dispatched
```

---

## Integration with Autodiff

Tensor operations are differentiable:

```scheme
;; Gradient of a function that uses tensors
(define (quadratic-form x)
  (tensor-dot x (matmul (eye 3) x)))

(display (gradient quadratic-form #(1.0 2.0 3.0)))
;; => gradient vector
```

---

*All tensor operations compose with `parallel-map`, `fold-left`, and
the rest of the language.*
