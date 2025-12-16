# Automatic Differentiation in Eshkol

## Table of Contents
- [Overview](#overview)
- [Three AD Modes](#three-ad-modes)
- [Symbolic Differentiation](#symbolic-differentiation)
- [Forward-Mode AD (Dual Numbers)](#forward-mode-ad-dual-numbers)
- [Reverse-Mode AD (Computational Graph)](#reverse-mode-ad-computational-graph)
- [Vector Calculus Operations](#vector-calculus-operations)
- [Implementation Architecture](#implementation-architecture)
- [Usage Examples](#usage-examples)
- [Performance Characteristics](#performance-characteristics)

---

## Overview

Eshkol provides **three automatic differentiation modes** for computing exact derivatives:

1. **Symbolic Differentiation** - Compile-time AST transformation (12 rules)
2. **Forward-Mode AD** - Runtime dual numbers for efficient single-variable derivatives
3. **Reverse-Mode AD** - Computational graph with backpropagation for many-input functions

All three modes compute **exact derivatives** (not numerical approximations) by systematically applying the chain rule to elementary operations.

---

## Three AD Modes

### Mode Selection

```scheme
;; Symbolic differentiation (compile-time)
(diff (lambda (x) (* x x)) 'x)
;; Returns: (lambda (x) (* 2 x))

;; Forward-mode AD (runtime, dual numbers)
(derivative (lambda (x) (* x x)) 3.0)
;; Returns: 6.0 (derivative at x=3.0)

;; Reverse-mode AD (runtime, computational graph)
(gradient (lambda (x y) (+ (* x x) (* y y))) (vector 3.0 4.0))
;; Returns: #(6.0 8.0) (gradient at (3.0, 4.0))
```

---

## Symbolic Differentiation

**Implementation:** [`lib/backend/autodiff_codegen.cpp:1366-1596`](lib/backend/autodiff_codegen.cpp:1366)

Symbolic differentiation transforms ASTs at **compile-time** using 12 differentiation rules:

### Differentiation Rules

| Expression | Derivative | Rule |
|------------|-----------|------|
| Constant `c` | `0` | Constant rule |
| Variable `x` | `1` | Variable rule (dx/dx = 1) |
| Variable `y` (≠x) | `0` | Variable rule (dy/dx = 0) |
| `(+ u v)` | `(+ du dv)` | Sum rule |
| `(- u v)` | `(- du dv)` | Difference rule |
| `(* u v)` | `(+ (* u dv) (* v du))` | Product rule |
| `(/ u v)` | `(/ (- (* v du) (* u dv)) (* v v))` | Quotient rule |
| `(expt u n)` | `(* (* n (expt u (- n 1))) du)` | Power rule |
| `(sin u)` | `(* (cos u) du)` | Chain rule: sin |
| `(cos u)` | `(* (- (sin u)) du)` | Chain rule: cos |
| `(exp u)` | `(* (exp u) du)` | Chain rule: exp |
| `(log u)` | `(* (/ 1.0 u) du)` | Chain rule: log |

### Example: Symbolic Differentiation

```scheme
;; Input function
(define f (lambda (x) (* x x)))

;; Symbolic derivative
(diff f 'x)
;; Returns AST equivalent to: (lambda (x) (* 2 x))

;; More complex example
(define g (lambda (x) (+ (* x x x) (* 2 x))))
(diff g 'x)
;; Returns: (lambda (x) (+ (* 3 (* x x)) 2))
```

---

## Forward-Mode AD (Dual Numbers)

**Implementation:** [`lib/backend/autodiff_codegen.cpp:234-388`](lib/backend/autodiff_codegen.cpp:234)

Forward-mode AD uses **dual numbers** to compute derivatives in a single forward pass. Each dual number stores both the function value and its derivative:

### Dual Number Structure

```c
typedef struct eshkol_dual_number {
    double value;       // f(x) - the function value
    double derivative;  // f'(x) - the derivative value
} eshkol_dual_number_t;  // 16 bytes
```

**Tagged value type:** [`ESHKOL_VALUE_DUAL_NUMBER`](inc/eshkol/eshkol.h:62) (type = 6)

### Dual Number Arithmetic

Each operation propagates both value and derivative:

```c
// Addition: (f + g)' = f' + g'
dual_add(a, b) = {a.value + b.value, a.deriv + b.deriv}

// Multiplication: (f * g)' = f'*g + f*g'
dual_mul(a, b) = {a.value * b.value, a.deriv * b.value + a.value * b.deriv}

// Division: (f / g)' = (f'*g - f*g') / g²
dual_div(a, b) = {a.value / b.value, (a.deriv * b.value - a.value * b.deriv) / (b.value * b.value)}
```

### Example: Forward-Mode AD

```scheme
;; Compute derivative of f(x) = x² at x = 3.0
(derivative (lambda (x) (* x x)) 3.0)
;; Returns: 6.0

;; How it works internally:
;; 1. Create dual number: x = {value: 3.0, deriv: 1.0}
;; 2. Evaluate (* x x) with dual arithmetic:
;;    result = {value: 9.0, deriv: 6.0}
;; 3. Extract derivative: 6.0
```

**Best for:** Functions with **1 input, many outputs** (e.g., f: ℝ → ℝⁿ)

---

## Reverse-Mode AD (Computational Graph)

**Implementation:** [`lib/backend/autodiff_codegen.cpp:390-858`](lib/backend/autodiff_codegen.cpp:390)

Reverse-mode AD builds a **computational graph** during the forward pass, then propagates gradients backward from outputs to inputs. This is the foundation of neural network training.

### AD Node Structure

```c
typedef struct ad_node {
    ad_node_type_t type;     // Operation type (16 types total)
    double value;            // Computed value (forward pass)
    double gradient;         // Accumulated gradient (backward pass)
    struct ad_node* input1;  // First parent node (NULL for leaves)
    struct ad_node* input2;  // Second parent node (NULL for unary ops)
    size_t id;              // Unique node ID for topological sorting
} ad_node_t;
```

### AD Node Types (16 total)

```c
typedef enum {
    AD_NODE_CONSTANT,    // Constant value (no gradient)
    AD_NODE_VARIABLE,    // Input variable (accumulates gradient)
    AD_NODE_ADD,         // Addition: f + g
    AD_NODE_SUB,         // Subtraction: f - g
    AD_NODE_MUL,         // Multiplication: f * g
    AD_NODE_DIV,         // Division: f / g
    AD_NODE_SIN,         // Sine: sin(f)
    AD_NODE_COS,         // Cosine: cos(f)
    AD_NODE_EXP,         // Exponential: exp(f)
    AD_NODE_LOG,         // Natural logarithm: log(f)
    AD_NODE_POW,         // Power: f^g
    AD_NODE_NEG          // Negation: -f
} ad_node_type_t;
```

### Computational Tape

The tape records all operations for backpropagation:

```c
typedef struct ad_tape {
    ad_node_t** nodes;       // Array of nodes in evaluation order
    size_t num_nodes;        // Current number of nodes
    size_t capacity;         // Allocated capacity
    ad_node_t** variables;   // Input variable nodes
    size_t num_variables;    // Number of input variables
} ad_tape_t;
```

**Nested gradient support:** 32-level tape stack for computing higher-order derivatives (implementation: [`lib/backend/autodiff_codegen.cpp:82-120`](lib/backend/autodiff_codegen.cpp:82))

### Example: Reverse-Mode AD

```scheme
;; Compute gradient of f(x, y) = x² + y² at (3.0, 4.0)
(gradient (lambda (v)
            (let ((x (vref v 0))
                  (y (vref v 1)))
              (+ (* x x) (* y y))))
          (vector 3.0 4.0))
;; Returns: #(6.0 8.0)  ; [∂f/∂x, ∂f/∂y]

;; How it works internally:
;; Forward pass (build graph):
;;   x_node = variable(3.0)
;;   y_node = variable(4.0)
;;   x2_node = mul(x_node, x_node) → value = 9.0
;;   y2_node = mul(y_node, y_node) → value = 16.0
;;   result = add(x2_node, y2_node) → value = 25.0
;;
;; Backward pass (propagate gradients):
;;   result.gradient = 1.0 (seed)
;;   x2_node.gradient += 1.0
;;   y2_node.gradient += 1.0
;;   x_node.gradient += 2 * x_node.value * 1.0 = 6.0
;;   y_node.gradient += 2 * y_node.value * 1.0 = 8.0
```

**Best for:** Functions with **many inputs, 1 output** (e.g., f: ℝⁿ → ℝ)

---

## Vector Calculus Operations

Eshkol provides high-level vector calculus operators built on reverse-mode AD:

### Scalar Field Operators (ℝⁿ → ℝ)

| Operator | Mathematical Definition | Eshkol Syntax |
|----------|------------------------|---------------|
| **Gradient** | ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] | `(gradient f point)` |
| **Hessian** | H_ij = ∂²f/(∂xᵢ∂xⱼ) | `(hessian f point)` |
| **Laplacian** | ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² | `(laplacian f point)` |
| **Directional** | D_vf = ∇f · v | `(directional-deriv f point direction)` |

### Vector Field Operators (ℝⁿ → ℝᵐ)

| Operator | Mathematical Definition | Eshkol Syntax |
|----------|------------------------|---------------|
| **Jacobian** | J_ij = ∂fᵢ/∂xⱼ | `(jacobian f point)` |
| **Divergence** | ∇·F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z | `(divergence f point)` |
| **Curl** | ∇×F = [∂F₃/∂y - ∂F₂/∂z, ...] | `(curl f point)` |

### Example: Gradient Computation

```scheme
;; Scalar field: f(x, y) = x² + y²
(define (f v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (+ (* x x) (* y y))))

;; Gradient at (3, 4)
(gradient f (vector 3.0 4.0))
;; Returns: #(6.0 8.0)
```

### Example: Jacobian Computation

```scheme
;; Vector field: F(x, y) = [x², y²]
(define (F v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (vector (* x x) (* y y))))

;; Jacobian at (3, 4)
(jacobian F (vector 3.0 4.0))
;; Returns: #(#(6.0 0.0)    ; [∂F₁/∂x, ∂F₁/∂y]
;;           #(0.0 8.0))    ; [∂F₂/∂x, ∂F₂/∂y]
```

### Example: Higher-Order Derivatives

```scheme
;; Second derivative: f''(x) = d²/dx² (x³)
(gradient (lambda (v)
            (gradient (lambda (w) 
                        (let ((x (vref w 0)))
                          (* x (* x x))))
                      v))
          (vector 2.0))
;; Returns: #(12.0)  ; f''(2) = 6*2 = 12
```

---

## Implementation Architecture

### Source Files

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **AD Codegen** | [`lib/backend/autodiff_codegen.cpp`](lib/backend/autodiff_codegen.cpp:1) | 1,766 | All 3 AD modes, vector calculus |
| **AD Runtime** | [`inc/eshkol/eshkol.h:565-603`](inc/eshkol/eshkol.h:565) | 38 | AD node structures, tape definition |
| **Dual Numbers** | [`inc/eshkol/eshkol.h:135-144`](inc/eshkol/eshkol.h:135) | 10 | Forward-mode dual number struct |
| **Type System** | [`lib/backend/type_system.cpp`](lib/backend/type_system.cpp:1) | 287 | AD type generation (dual_t, ad_node_t) |

### AD Mode Detection

The compiler automatically selects the AD mode based on the operator used:

```cpp
// lib/backend/autodiff_codegen.cpp:1294-1366
case ESHKOL_DIFF_OP:       → Symbolic differentiation
case ESHKOL_DERIVATIVE_OP: → Forward or reverse (auto-selected)
case ESHKOL_GRADIENT_OP:   → Reverse-mode (always)
case ESHKOL_JACOBIAN_OP:   → Reverse-mode (always)
case ESHKOL_HESSIAN_OP:    → Nested reverse-mode
// ... other vector calculus operators
```

### Tape Management

**Tape Stack:** 32-level nested gradient support for computing derivatives of derivatives:

```cpp
// Global tape stack (lib/backend/autodiff_codegen.cpp:82-86)
static ad_tape_t* g_ad_tape_stack[32];
static int g_ad_tape_depth = 0;

// Push new tape for nested gradient
void push_ad_tape() { g_ad_tape_depth++; }

// Pop tape after gradient computation
void pop_ad_tape() { g_ad_tape_depth--; }
```

**Example: Nested gradients for Hessian**

```scheme
;; f(x) = x⁴
;; f'(x) = 4x³
;; f''(x) = 12x²

(hessian (lambda (v) 
           (let ((x (vref v 0)))
             (* x (* x (* x x)))))
         (vector 2.0))
;; Returns: #(#(48.0))  ; f''(2) = 12*4 = 48
```

---

## Usage Examples

### Example 1: Simple Derivative

```scheme
;; f(x) = sin(x²)
;; f'(x) = 2x·cos(x²)

(derivative (lambda (x) (sin (* x x))) 1.0)
;; Returns: 1.0806  ; 2·1·cos(1) ≈ 1.0806
```

### Example 2: Gradient Descent

```scheme
;; Minimize f(x, y) = (x - 2)² + (y - 3)²
(define (f v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (+ (* (- x 2.0) (- x 2.0))
       (* (- y 3.0) (- y 3.0)))))

;; Gradient descent step
(define (gd-step point learning-rate)
  (let ((grad (gradient f point)))
    (vector (- (vref point 0) (* learning-rate (vref grad 0)))
            (- (vref point 1) (* learning-rate (vref grad 1))))))

;; Run optimization
(define initial (vector 0.0 0.0))
(define step1 (gd-step initial 0.1))
;; step1 ≈ #(0.4 0.6)  ; Move toward minimum (2, 3)
```

### Example 3: Neural Network Backpropagation

```scheme
;; Simple 2-layer network
(define (forward inputs weights1 weights2)
  (let* ((hidden (tensor-dot inputs weights1))
         (activated (tensor-map relu hidden))
         (output (tensor-dot activated weights2)))
    output))

;; Compute weight gradients
(define (compute-gradients inputs targets weights1 weights2)
  (let* ((prediction (forward inputs weights1 weights2))
         (loss (mse-loss prediction targets))
         ;; Gradient with respect to weights
         (grad-w2 (gradient (lambda (w) 
                              (forward inputs weights1 w))
                           weights2))
         (grad-w1 (gradient (lambda (w)
                              (forward inputs w weights2))
                           weights1)))
    (values grad-w1 grad-w2)))
```

### Example 4: Vector Calculus

```scheme
;; Divergence of vector field F(x, y) = [x², y²]
(define (F v)
  (vector (* (vref v 0) (vref v 0))
          (* (vref v 1) (vref v 1))))

(divergence F (vector 3.0 4.0))
;; Returns: 14.0  ; ∂(x²)/∂x + ∂(y²)/∂y = 2x + 2y = 6 + 8 = 14

;; Curl of vector field (ℝ³ → ℝ³)
(define (F3 v)
  (vector (* (vref v 1) (vref v 2))   ; y*z
          (* (vref v 0) (vref v 2))   ; x*z
          (* (vref v 0) (vref v 1)))) ; x*y

(curl F3 (vector 1.0 2.0 3.0))
;; Returns: #(0.0 0.0 0.0)  ; ∇×F = 0 for this field
```

---

## Performance Characteristics

### Time Complexity

| Mode | Forward Pass | Backward Pass | Total | Best Use Case |
|------|-------------|---------------|-------|---------------|
| **Symbolic** | O(1) compile-time | — | O(1) | Simple closed-form derivatives |
| **Forward** | O(n) | — | O(n) | 1 input, many outputs |
| **Reverse** | O(n) | O(n) | O(2n) | Many inputs, 1 output |

Where n = number of operations in the function.

### Space Complexity

| Mode | Memory Usage | Notes |
|------|-------------|-------|
| **Symbolic** | O(AST size) | AST transformation only |
| **Forward** | O(1) | Just dual numbers (16 bytes each) |
| **Reverse** | O(n) | Tape stores all nodes (~48 bytes/node) |

### Choosing the Right Mode

```scheme
;; Use symbolic for compile-time known derivatives
(diff (lambda (x) (* x x)) 'x)

;; Use forward-mode for f: ℝ → ℝⁿ
(derivative (lambda (x) (vector (* x x) (* x x x))) 2.0)

;; Use reverse-mode for f: ℝⁿ → ℝ
(gradient (lambda (v) (+ (vref v 0) (vref v 1))) (vector 1.0 2.0))
```

---

## AD-Aware Execution

**Critical feature:** Eshkol automatically switches between **AD mode** and **normal mode** based on argument types:

```scheme
;; Same function works with BOTH regular values AND AD values
(define (f x) (* x x))

;; Regular execution (no AD overhead)
(f 3.0)
;; Returns: 9.0

;; AD execution (creates computational graph)
(gradient (lambda (v) (f (vref v 0))) (vector 3.0))
;; Returns: #(6.0)
```

**Implementation:** Closures check argument types at runtime and dispatch to the appropriate code path ([`lib/backend/function_codegen.cpp:456-523`](lib/backend/function_codegen.cpp:456)).

---

## Integration with Tensors

Autodiff works seamlessly with tensor operations. The [`vref`](lib/backend/tensor_codegen.cpp:1234) operation creates AD nodes when in AD mode:

```scheme
;; Gradient of tensor sum
(gradient (lambda (t)
            (+ (+ (vref t 0) (vref t 1))
               (vref t 2)))
          (vector 1.0 2.0 3.0))
;; Returns: #(1.0 1.0 1.0)  ; Gradient is [1, 1, 1]
```

---

## See Also

- [Vector Operations](VECTOR_OPERATIONS.md) - Tensor operations, `vref`, linear algebra
- [Type System](TYPE_SYSTEM.md) - Dual number type, AD node type
- [Memory Management](MEMORY_MANAGEMENT.md) - Tape allocation, node lifecycle
- [API Reference](../API_REFERENCE.md) - Complete AD function reference
