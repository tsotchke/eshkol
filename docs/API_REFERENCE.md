# Eshkol v1.0 API Reference

**Version**: 1.0  
**Last Updated**: 2025-12-11  
**Audience**: Scientific Computing & AI Systems Programming

This comprehensive reference documents all special forms, functions, and operations in the Eshkol language. All documentation is code-verified against the production compiler implementation (67,000+ lines of LLVM-based C++ code).

---

## Table of Contents

1. [Special Forms](#special-forms) (70+)
2. [Automatic Differentiation](#automatic-differentiation) (9 operations, 3 modes)
3. [Tensor Operations](#tensor-operations) (30+ operations)
4. [List Processing](#list-processing) (50+ operations)
5. [Arithmetic & Comparison](#arithmetic--comparison)
6. [Math Functions](#math-functions) (30+ functions)
7. [String Operations](#string-operations)
8. [File & I/O](#file--io)
9. [System Operations](#system-operations)
10. [Hash Tables](#hash-tables)
11. [Type System](#type-system)
12. [Standard Library](#standard-library) (25 modules)

---

## Special Forms

### Core Language Constructs

#### `define`
**Syntax**: `(define name value)` or `(define (name params...) body)`

Binds a value to a name in the current scope. Functions are first-class values.

**Examples**:
```scheme
(define x 42)
(define pi 3.14159)
(define (square x) (* x x))
(define (add a b) (+ a b))
```

**Type**: Top-level or nested definition  
**Returns**: Unspecified (implementation returns the defined value)  
**Implementation**: [`codegenDefine()`](../lib/backend/llvm_codegen.cpp:6110)

---

#### `lambda`
**Syntax**: `(lambda (params...) body)` or `(lambda (params... . rest) body)`

Creates an anonymous function (closure). Supports variable arity with rest parameters.

**Examples**:
```scheme
(lambda (x) (* x 2))
(lambda (x y) (+ x y))
(lambda (x . rest) (cons x rest))  ; Variadic
```

**Type**: Function expression  
**Returns**: Closure (32-byte structure with captures and S-expression for homoiconicity)  
**Implementation**: [`codegenLambda()`](../lib/backend/llvm_codegen.cpp:12962)  
**Closure Structure**: `{func_ptr(8), env(8), sexpr_ptr(8), return_type(1), arity(1), flags(1)}`

**Advanced Features**:
- **Mutable Captures**: Captured variables support `set!` with pointer-passing semantics
- **Nested Closures**: Supports arbitrary-depth closure nesting with correct scoping
- **TCO Support**: Tail-recursive lambdas optimize to loops (when used with `letrec`)
- **Homoiconic Display**: Lambdas preserve source code structure for metaprogramming

---

#### `let`, `let*`, `letrec`
**Syntax**: 
- `(let ((var val)...) body)`
- `(let* ((var val)...) body)` - Sequential bindings
- `(letrec ((var val)...) body)` - Recursive bindings
- `(let name ((var val)...) body)` - Named let (iteration)

Establishes local bindings. `let*` evaluates bindings sequentially (left-to-right). `letrec` supports mutually recursive definitions.

**Examples**:
```scheme
(let ((x 1) (y 2)) (+ x y))  ; => 3
(let* ((x 1) (y (+ x 1))) y)  ; => 2
(letrec ((even? (lambda (n) (if (= n 0) #t (odd? (- n 1)))))
         (odd? (lambda (n) (if (= n 0) #f (even? (- n 1))))))
  (even? 4))  ; => #t
(let loop ((n 10) (acc 1))  ; Named let for iteration
  (if (= n 0) acc (loop (- n 1) (* n acc))))  ; => 3628800
```

**Type**: Binding forms  
**Scope**: `let` - parallel, `let*` - sequential, `letrec` - recursive  
**Implementation**: [`codegenLet()`](../lib/backend/llvm_codegen.cpp:13923), [`BindingCodegen`](../lib/backend/binding_codegen.cpp)

---

#### `set!`
**Syntax**: `(set! var value)`

Mutates an existing variable. Works with parameters, let-bindings, and closure captures.

**Examples**:
```scheme
(define x 10)
(set! x 20)  ; x is now 20

(let ((counter 0))
  (lambda () (set! counter (+ counter 1)) counter))  ; Stateful closure
```

**Type**: Mutation operator  
**Returns**: Unspecified (implementation returns the new value)  
**Implementation**: [`codegenSet()`](../lib/backend/llvm_codegen.cpp:7604)

---

#### `if`
**Syntax**: `(if test then-expr else-expr)`

Conditional evaluation. `else-expr` is optional (defaults to unspecified).

**Examples**:
```scheme
(if (> x 0) "positive" "non-positive")
(if (null? lst) '() (car lst))
```

**Type**: Conditional expression  
**Returns**: Value of executed branch  
**Implementation**: [`codegenIfCall()`](../lib/backend/control_flow_codegen.cpp)

---

#### `cond`
**Syntax**: `(cond (test expr...)... (else expr...))`

Multi-way conditional. Evaluates tests sequentially, executes first matching clause.

**Examples**:
```scheme
(cond
  ((< x 0) "negative")
  ((= x 0) "zero")
  ((> x 0) "positive"))

(cond
  ((null? lst) 0)
  (else (+ 1 (length (cdr lst)))))
```

**Type**: Multi-conditional  
**Returns**: Value of first matching clause  
**Implementation**: [`codegenCond()`](../lib/backend/control_flow_codegen.cpp)

---

#### `case`
**Syntax**: `(case key ((datum...) expr...)... (else expr...))`

Pattern matching on values using `eqv?` comparison.

**Examples**:
```scheme
(case (car '(a b c))
  ((a) 'first)
  ((b) 'second)
  (else 'other))  ; => first
```

**Type**: Pattern matching  
**Returns**: Value of matching clause  
**Implementation**: [`codegenCase()`](../lib/backend/control_flow_codegen.cpp)

---

#### `begin`
**Syntax**: `(begin expr...)`

Evaluates expressions sequentially, returns last value. Used for side effects.

**Examples**:
```scheme
(begin
  (display "Computing...")
  (newline)
  (* 6 7))  ; => 42
```

**Type**: Sequence operator  
**Returns**: Value of last expression  
**Implementation**: [`codegenBegin()`](../lib/backend/control_flow_codegen.cpp)

---

#### `and`, `or`, `not`
**Syntax**: `(and expr...)`, `(or expr...)`, `(not expr)`

Short-circuit boolean operators. `and` returns first falsy value or last value. `or` returns first truthy value or last value.

**Examples**:
```scheme
(and (> x 0) (< x 10))
(or (null? lst) (pair? lst))
(not #f)  ; => #t
```

**Type**: Boolean operators  
**Implementation**: [`codegenAnd()`](../lib/backend/control_flow_codegen.cpp), [`codegenOr()`](../lib/backend/control_flow_codegen.cpp)

---

#### `when`, `unless`
**Syntax**: `(when test body...)`, `(unless test body...)`

One-armed conditionals for side effects.

**Examples**:
```scheme
(when (file-exists? "data.txt")
  (display "Processing...")
  (process-file "data.txt"))

(unless (null? errors)
  (display-errors errors))
```

**Type**: Conditional side effects  
**Returns**: Unspecified or last body value  
**Implementation**: [`codegenWhen()`](../lib/backend/control_flow_codegen.cpp)

---

#### `do`
**Syntax**: `(do ((var init step)...) (test result...) body...)`

Iteration construct with explicit state variables.

**Examples**:
```scheme
(do ((i 0 (+ i 1))
     (sum 0 (+ sum i)))
    ((>= i 10) sum))  ; => 45
```

**Type**: Iteration  
**Returns**: Value of result expressions when test succeeds  
**Implementation**: [`codegenDo()`](../lib/backend/llvm_codegen.cpp:11117)

---

#### `quote`, `quasiquote`, `unquote`
**Syntax**: `(quote expr)` or `'expr`, `` `expr``, `,expr`

Creates S-expressions as data structures. Quasiquote allows selective evaluation.

**Examples**:
```scheme
'(1 2 3)  ; => (1 2 3)
`(1 ,(+ 1 1) 3)  ; => (1 2 3)
'(lambda (x) (* x 2))  ; => (lambda (x) (* x 2))
```

**Type**: Data literals  
**Returns**: S-expression (cons-based data structure)  
**Implementation**: [`codegenQuotedAST()`](../lib/backend/llvm_codegen.cpp:16279), [`HomoiconicCodegen`](../lib/backend/homoiconic_codegen.cpp)

---

#### `apply`
**Syntax**: `(apply proc arg... args-list)`

Applies procedure to arguments from a list.

**Examples**:
```scheme
(apply + '(1 2 3))  ; => 6
(apply * 2 3 '(4 5))  ; => 120
```

**Type**: Higher-order application  
**Returns**: Result of procedure application  
**Implementation**: [`codegenApply()`](../lib/backend/call_apply_codegen.cpp)

---

### Exception Handling

#### `guard`
**Syntax**: `(guard (var (test handler)...) body...)`

Exception handling with pattern matching on exception type.

**Examples**:
```scheme
(guard (exn
         ((divide-by-zero? exn) 'infinity)
         (else 'error))
  (/ 1 0))
```

**Type**: Exception handler  
**Implementation**: [`codegenGuard()`](../lib/backend/llvm_codegen.cpp:10201)

---

#### `raise`
**Syntax**: `(raise obj)`

Raises an exception. Uses longjmp-based unwinding.

**Examples**:
```scheme
(raise "error message")
(if (< x 0) (raise "negative value") x)
```

**Type**: Exception trigger  
**Implementation**: [`codegenRaise()`](../lib/backend/llvm_codegen.cpp:10438)

---

### Multiple Return Values

#### `values`
**Syntax**: `(values expr...)`

Returns multiple values from a function.

**Examples**:
```scheme
(values 1 2 3)
(define (divmod a b) (values (quotient a b) (remainder a b)))
```

**Type**: Multi-value constructor  
**Implementation**: [`codegenValues()`](../lib/backend/llvm_codegen.cpp:10500)

---

#### `call-with-values`
**Syntax**: `(call-with-values producer consumer)`

Calls consumer with values produced by producer.

**Examples**:
```scheme
(call-with-values
  (lambda () (values 1 2))
  (lambda (a b) (+ a b)))  ; => 3
```

**Type**: Multi-value application  
**Implementation**: [`codegenCallWithValues()`](../lib/backend/llvm_codegen.cpp:10564)

---

### Pattern Matching

#### `match`
**Syntax**: `(match expr (pattern body)...)`

Advanced pattern matching with support for literals, variables, cons patterns, lists, wildcards, predicates, and or-patterns.

**Examples**:
```scheme
(match '(1 2 3)
  ((a b c) (+ a b c)))  ; => 6

(match x
  ((? number?) 'is-number)
  (_ 'not-number))
```

**Type**: Pattern matching  
**Patterns**: `_` (wildcard), `var`, `literal`, `(p1 . p2)`, `(p...)`, `(? pred)`, `(or p...)`  
**Implementation**: [`codegenMatch()`](../lib/backend/llvm_codegen.cpp:11117)

---

### Memory Management (OALR)

#### `with-region`
**Syntax**: `(with-region [name size-hint] body...)`

Creates a lexical memory region, automatically freed after body execution.

**Examples**:
```scheme
(with-region
  (let ((big-data (make-vector 1000000 0)))
    (process big-data)))  ; Memory freed after block
```

**Type**: Region-based memory management  
**Implementation**: [`codegenWithRegion()`](../lib/backend/llvm_codegen.cpp:22635)

---

#### `owned`, `move`, `borrow`, `shared`
**Syntax**: `(owned expr)`, `(move value)`, `(borrow val body...)`, `(shared expr)`

Ownership annotations for linear types and reference counting (compile-time tracking).

**Type**: Ownership markers  
**Implementation**: [`codegenOwned()`](../lib/backend/llvm_codegen.cpp:22705)

---

### External Interface

#### `extern`
**Syntax**: `(extern return-type name [real-name] param-types...)`

Declares external C function for FFI.

**Examples**:
```scheme
(extern "double" "my_sin" "sin" "double")
(extern "int" "puts" "string")
```

**Type**: FFI declaration  
**Implementation**: [`codegenExtern()`](../lib/backend/llvm_codegen.cpp:12201)

---

#### `extern-var`
**Syntax**: `(extern-var type name)`

Declares external C variable.

**Type**: FFI variable  
**Implementation**: [`codegenExternVar()`](../lib/backend/llvm_codegen.cpp:12157)

---

## Automatic Differentiation

Eshkol provides **three modes** of automatic differentiation:

1. **Symbolic**: Compile-time AST transformation (mathematical simplification)
2. **Forward-mode**: Dual numbers (efficient for functions ℝ→ℝⁿ)
3. **Reverse-mode**: Computational graphs with backpropagation (efficient for ℝⁿ→ℝ)

### Symbolic Differentiation

#### `diff`
**Syntax**: `(diff expr var)`

Symbolic differentiation at compile time. Returns S-expression of derivative formula.

**Examples**:
```scheme
(diff '(* x x) 'x)  ; => (* 2 x)
(diff '(sin (* 2 x)) 'x)  ; => (* 2 (cos (* 2 x)))
(diff '(+ (* x x) (* 3 x) 5) 'x)  ; => (+ (* 2 x) 3)
```

**Mode**: Compile-time symbolic transformation  
**Input**: Expression AST, variable symbol  
**Output**: S-expression (simplified derivative formula)  
**Rules**: Sum, product, quotient, chain rule for sin/cos/exp/log/pow/sqrt  
**Implementation**: [`buildSymbolicDerivative()`](../lib/backend/llvm_codegen.cpp:16041)

---

### Forward-Mode AD (Dual Numbers)

#### `derivative`
**Syntax**: `(derivative f x)` or `(derivative f)`

Computes derivative using dual numbers (forward-mode AD).

**Examples**:
```scheme
; Direct computation
(derivative (lambda (x) (* x x)) 3.0)  ; => 6.0

; Higher-order (returns derivative function)
(define df (derivative (lambda (x) (sin x))))
(df 0.0)  ; => 1.0 (derivative of sin at 0)
```

**Mode**: Forward-mode automatic differentiation  
**Input**: Function `f: ℝ → ℝ`, point `x: ℝ` (optional)  
**Output**: Derivative value or derivative function  
**Method**: Dual numbers `(a, a')` with automatic chain rule propagation  
**Complexity**: O(1) overhead per operation  
**Implementation**: [`codegenDerivative()`](../lib/backend/llvm_codegen.cpp:17451), [`AutodiffCodegen::derivative()`](../lib/backend/autodiff_codegen.cpp:902)

**Dual Number Arithmetic**:
- Addition: `(a, a') + (b, b') = (a+b, a'+b')`
- Multiplication: `(a, a') * (b, b') = (ab, a'b + ab')`
- Sin: `sin(a, a') = (sin(a), a' cos(a))`

---

### Reverse-Mode AD (Computational Graphs)

#### `gradient`
**Syntax**: `(gradient f point)` or `(gradient f)`

Computes gradient vector using reverse-mode AD (backpropagation).

**Examples**:
```scheme
; Scalar function ℝⁿ → ℝ
(define (f v) (+ (* (vref v 0) (vref v 0))
                 (* (vref v 1) (vref v 1))))
(gradient f #(2.0 3.0))  ; => #(4.0 6.0)

; Higher-order usage
(define grad-f (gradient f))
(grad-f #(1.0 1.0))  ; => #(2.0 2.0)
```

**Mode**: Reverse-mode automatic differentiation  
**Input**: Function `f: ℝⁿ → ℝ`, point `x: ℝⁿ`  
**Output**: Gradient vector `∇f(x): ℝⁿ` (tensor)  
**Method**: Computational graph with tape-based backpropagation  
**Complexity**: O(1) backward pass (optimal for scalar outputs)  
**Implementation**: [`codegenGradient()`](../lib/backend/llvm_codegen.cpp:18317)

**Graph Construction**:
- Forward pass builds computation graph (AD nodes on tape)
- Backward pass propagates gradients via chain rule
- Supports nested gradients (arbitrary depth) with tape stack

---

#### `jacobian`
**Syntax**: `(jacobian f point)`

Computes Jacobian matrix using reverse-mode AD.

**Examples**:
```scheme
; Vector function ℝⁿ → ℝᵐ
(define (f v) (vector (* 2 (vref v 0))
                      (+ (vref v 0) (vref v 1))))
(jacobian f #(3.0 4.0))  ; => #((2.0 0.0) (1.0 1.0))
```

**Mode**: Reverse-mode automatic differentiation  
**Input**: Function `f: ℝⁿ → ℝᵐ`, point `x: ℝⁿ`  
**Output**: Jacobian matrix `J[i,j] = ∂fᵢ/∂xⱼ` (m×n tensor)  
**Method**: m gradient computations (one per output component)  
**Complexity**: O(m) backward passes  
**Implementation**: [`codegenJacobian()`](../lib/backend/llvm_codegen.cpp:20327)

---

#### `hessian`
**Syntax**: `(hessian f point)`

Computes Hessian matrix (matrix of second derivatives).

**Examples**:
```scheme
(define (f v) (+ (* (vref v 0) (vref v 0))
                 (* (vref v 1) (vref v 1))))
(hessian f #(1.0 1.0))  ; => #((2.0 0.0) (0.0 2.0))
```

**Mode**: Numerical differentiation (finite differences on gradient)  
**Input**: Function `f: ℝⁿ → ℝ`, point `x: ℝⁿ`  
**Output**: Hessian matrix `H[i,j] = ∂²f/∂xᵢ∂xⱼ` (n×n tensor)  
**Method**: Perturb each dimension, compute gradient, finite difference  
**Complexity**: O(n²) gradient computations  
**Implementation**: [`codegenHessian()`](../lib/backend/llvm_codegen.cpp:21175)

---

### Vector Calculus Operators

#### `divergence`
**Syntax**: `(divergence f point)`

Computes divergence of vector field: `∇·F = ∂F₁/∂x₁ + ... + ∂Fₙ/∂xₙ`

**Examples**:
```scheme
(define (F v) v)  ; Identity field
(divergence F #(1.0 2.0 3.0))  ; => 3.0
```

**Mode**: Trace of Jacobian  
**Input**: Vector field `F: ℝⁿ → ℝⁿ`, point `x: ℝⁿ`  
**Output**: Scalar (sum of diagonal Jacobian elements)  
**Physics**: Measures "source strength" at a point  
**Implementation**: [`codegenDivergence()`](../lib/backend/llvm_codegen.cpp:21942)

---

#### `curl`
**Syntax**: `(curl f point)`

Computes curl of 3D vector field: `∇×F`

**Examples**:
```scheme
(define (F v) (vector (vref v 1) (- (vref v 0)) 0.0))
(curl F #(1.0 2.0 0.0))  ; => #(0.0 0.0 -2.0)
```

**Mode**: Antisymmetric part of Jacobian  
**Input**: Vector field `F: ℝ³ → ℝ³`, point `x: ℝ³`  
**Output**: Vector `(∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y)`  
**Physics**: Measures "rotation" of field  
**Generalization**: Works for n≥2 (differential 2-forms)  
**Implementation**: [`codegenCurl()`](../lib/backend/llvm_codegen.cpp:22059)

---

#### `laplacian`
**Syntax**: `(laplacian f point)`

Computes Laplacian: `∇²f = ∂²f/∂x₁² + ... + ∂²f/∂xₙ²`

**Examples**:
```scheme
(define (f v) (+ (* (vref v 0) (vref v 0))
                 (* (vref v 1) (vref v 1))))
(laplacian f #(1.0 2.0))  ; => 4.0
```

**Mode**: Trace of Hessian  
**Input**: Scalar field `f: ℝⁿ → ℝ`, point `x: ℝⁿ`  
**Output**: Scalar (sum of second partial derivatives)  
**Physics**: Fundamental in heat equation, wave equation, quantum mechanics  
**Implementation**: [`codegenLaplacian()`](../lib/backend/llvm_codegen.cpp:22300)

---

#### `directional-deriv`
**Syntax**: `(directional-deriv f point direction)`

Computes directional derivative: `D_v f = ∇f · v`

**Examples**:
```scheme
(define (f v) (* (vref v 0) (vref v 1)))
(directional-deriv f #(2.0 3.0) #(1.0 0.0))  ; => 3.0
```

**Mode**: Gradient dot product  
**Input**: Scalar field `f: ℝⁿ → ℝ`, point, direction vector  
**Output**: Scalar (rate of change in specified direction)  
**Implementation**: [`codegenDirectionalDerivative()`](../lib/backend/llvm_codegen.cpp:22422)

---

## Tensor Operations

Tensors are N-dimensional arrays with homogeneous double-precision elements stored as int64 bit patterns.

**Tensor Structure** (32 bytes):
```c
struct eshkol_tensor_t {
    uint64_t* dims;          // Dimension sizes [d0, d1, ..., d(n-1)]
    uint64_t num_dimensions; // Rank of tensor
    int64_t* elements;       // Flattened data (double bits as int64)
    uint64_t total_elements; // Product of all dimensions
};
```

### Tensor Creation

#### `zeros`, `ones`
**Syntax**: `(zeros dims...)` or `(zeros '(dims...))`, `(ones dims...)`

Creates tensors filled with 0.0 or 1.0.

**Examples**:
```scheme
(zeros 5)           ; => #(0.0 0.0 0.0 0.0 0.0)
(zeros 2 3)         ; => #((0.0 0.0 0.0) (0.0 0.0 0.0))
(zeros '(2 3))      ; Same as above
(ones 3 3)          ; => 3×3 identity-ready matrix
```

**Type**: Tensor constructors  
**Returns**: Tensor (HEAP_PTR with HEAP_SUBTYPE_TENSOR)  
**Implementation**: [`zeros()`](../lib/backend/tensor_codegen.cpp:2311), [`ones()`](../lib/backend/tensor_codegen.cpp:2518)

---

#### `eye`
**Syntax**: `(eye n)` or `(eye rows cols)`

Creates identity matrix.

**Examples**:
```scheme
(eye 3)      ; => #((1.0 0.0 0.0) (0.0 1.0 0.0) (0.0 0.0 1.0))
(eye 2 3)    ; => #((1.0 0.0 0.0) (0.0 1.0 0.0))
```

**Type**: Matrix constructor  
**Implementation**: [`eye()`](../lib/backend/tensor_codegen.cpp:2718)

---

#### `arange`
**Syntax**: `(arange n)` or `(arange start end [step])`

Creates range of values (like NumPy arange).

**Examples**:
```scheme
(arange 5)           ; => #(0.0 1.0 2.0 3.0 4.0)
(arange 2.0 5.0)     ; => #(2.0 3.0 4.0)
(arange 0.0 1.0 0.25) ; => #(0.0 0.25 0.5 0.75)
```

**Type**: Range constructor  
**Implementation**: [`arange()`](../lib/backend/tensor_codegen.cpp:2788)

---

#### `linspace`
**Syntax**: `(linspace start end num)`

Creates `num` evenly spaced values from `start` to `end` (inclusive).

**Examples**:
```scheme
(linspace 0.0 1.0 5)  ; => #(0.0 0.25 0.5 0.75 1.0)
```

**Type**: Linear spacing constructor  
**Implementation**: [`linspace()`](../lib/backend/tensor_codegen.cpp:2934)

---

### Tensor Access & Manipulation

#### `tensor-get`
**Syntax**: `(tensor-get tensor idx...)` or `(vref tensor idx)`

N-dimensional tensor indexing with slicing support.

**Examples**:
```scheme
(define M #((1 2 3) (4 5 6)))
(tensor-get M 0 1)   ; => 2 (scalar element)
(tensor-get M 1)     ; => #(4 5 6) (row slice)
(vref #(10 20 30) 1) ; => 20 (1D shorthand)
```

**Type**: Tensor indexing  
**Slicing**: Partial indexing returns view tensor (zero-copy)  
**AD-Aware**: Preserves AD node pointers during gradient computation  
**Implementation**: [`tensorGet()`](../lib/backend/tensor_codegen.cpp:130), [`codegenTensorVectorRef()`](../lib/backend/llvm_codegen.cpp:15054)

---

#### `tensor-set`
**Syntax**: `(tensor-set tensor value idx...)`

Sets tensor element (mutable).

**Examples**:
```scheme
(define M #((1 2) (3 4)))
(tensor-set M 99 0 1)  ; M[0,1] := 99
```

**Type**: Tensor mutation  
**Implementation**: [`tensorSet()`](../lib/backend/tensor_codegen.cpp:330)

---

#### `transpose`
**Syntax**: `(transpose matrix)`

Transposes 2D matrix (swaps rows and columns).

**Examples**:
```scheme
(transpose #((1 2 3) (4 5 6)))  ; => #((1 4) (2 5) (3 6))
```

**Type**: Matrix transformation  
**Complexity**: O(mn) time, O(mn) space  
**Implementation**: [`transpose()`](../lib/backend/tensor_codegen.cpp:1920)

---

#### `reshape`
**Syntax**: `(reshape tensor dim...)` or `(reshape tensor dims-list)`

Changes tensor shape (zero-copy view).

**Examples**:
```scheme
(reshape #(1 2 3 4 5 6) 2 3)  ; => #((1 2 3) (4 5 6))
(reshape #(1 2 3 4 5 6) '(3 2))  ; => #((1 2) (3 4) (5 6))
```

**Type**: Shape transformation  
**Constraint**: Total elements must match  
**Implementation**: [`reshape()`](../lib/backend/tensor_codegen.cpp:2041)

---

#### `flatten`
**Syntax**: `(flatten tensor)`

Flattens N-dimensional tensor to 1D (zero-copy view).

**Examples**:
```scheme
(flatten #((1 2) (3 4)))  ; => #(1 2 3 4)
```

**Type**: Dimensionality reduction  
**Implementation**: [`codegenFlatten()`](../lib/backend/llvm_codegen.cpp:15280)

---

#### `tensor-shape`
**Syntax**: `(tensor-shape tensor)`

Returns dimension sizes as Scheme list.

**Examples**:
```scheme
(tensor-shape #((1 2 3) (4 5 6)))  ; => (2 3)
```

**Type**: Metadata query  
**Returns**: Cons-based list of int64 dimensions  
**Implementation**: [`tensorShape()`](../lib/backend/tensor_codegen.cpp:1796)

---

### Tensor Arithmetic

#### `tensor-add`, `tensor-sub`, `tensor-mul`, `tensor-div`
**Syntax**: `(tensor-add A B)`, etc.

Element-wise arithmetic operations. Supports both tensors and Scheme vectors.

**Examples**:
```scheme
(tensor-add #(1 2 3) #(4 5 6))  ; => #(5 7 9)
(tensor-mul #((1 2) (3 4)) #((2 2) (2 2)))  ; => #((2 4) (6 8))
```

**Type**: Element-wise binary operations  
**Constraint**: Operands must have same shape  
**Implementation**: [`tensorArithmetic()`](../lib/backend/tensor_codegen.cpp:382)

---

#### `tensor-dot`
**Syntax**: `(tensor-dot A B)` or `(matmul A B)`

Matrix multiplication for 2D, dot product for 1D.

**Examples**:
```scheme
; 1D vectors: dot product
(tensor-dot #(1 2 3) #(4 5 6))  ; => 32

; 2D matrices: matmul
(matmul #((1 2) (3 4)) #((5 6) (7 8)))  ; => #((19 22) (43 50))
```

**Type**: Linear algebra operation  
**Complexity**: O(mnk) for m×k @ k×n matrices  
**Implementation**: [`tensorDot()`](../lib/backend/tensor_codegen.cpp:662), [`codegenMatmul()`](../lib/backend/llvm_codegen.cpp:15333)

---

#### `trace`
**Syntax**: `(trace matrix)`

Sum of diagonal elements.

**Examples**:
```scheme
(trace #((1 2) (3 4)))  ; => 5
```

**Type**: Matrix invariant  
**Implementation**: [`codegenTrace()`](../lib/backend/llvm_codegen.cpp:15495)

---

#### `norm`
**Syntax**: `(norm vector)`

Euclidean (L2) norm: `√(Σ xᵢ²)`

**Examples**:
```scheme
(norm #(3.0 4.0))  ; => 5.0
```

**Type**: Vector magnitude  
**Implementation**: [`codegenNorm()`](../lib/backend/llvm_codegen.cpp:15593)

---

#### `outer`
**Syntax**: `(outer u v)`

Outer product of vectors: `(u ⊗ v)[i,j] = uᵢ vⱼ`

**Examples**:
```scheme
(outer #(1 2) #(3 4 5))  ; => #((3 4 5) (6 8 10))
```

**Type**: Tensor product  
**Implementation**: [`codegenOuterProduct()`](../lib/backend/llvm_codegen.cpp:15712)

---

### Tensor Reductions

#### `tensor-sum`, `tensor-mean`
**Syntax**: `(tensor-sum tensor)`, `(tensor-mean tensor)`

Sum or mean of all elements.

**Examples**:
```scheme
(tensor-sum #(1 2 3 4 5))   ; => 15.0
(tensor-mean #(2 4 6 8))    ; => 5.0
```

**Type**: Statistical reductions  
**Implementation**: [`tensorSum()`](../lib/backend/tensor_codegen.cpp:1556), [`tensorMean()`](../lib/backend/tensor_codegen.cpp:1675)

---

#### `tensor-reduce`
**Syntax**: `(tensor-reduce tensor func init)` or `(tensor-reduce tensor func init dim)`

Reduces tensor with custom function.

**Examples**:
```scheme
(tensor-reduce #(1 2 3 4) + 0)   ; => 10
(tensor-reduce matrix max -inf 0) ; Row-wise maximum
```

**Type**: Generalized reduction  
**Implementation**: [`tensorReduceAll()`](../lib/backend/tensor_codegen.cpp:1152), [`tensorReduceWithDim()`](../lib/backend/tensor_codegen.cpp:1315)

---

#### `tensor-apply`
**Syntax**: `(tensor-apply tensor func)`

Maps function over tensor elements.

**Examples**:
```scheme
(tensor-apply #(1 2 3) (lambda (x) (* x x)))  ; => #(1 4 9)
```

**Type**: Element-wise map  
**Implementation**: [`tensorApply()`](../lib/backend/tensor_codegen.cpp:1022)

---

## List Processing

Eshkol lists are heterogeneous cons cells with tagged values (16-byte storage per element).

**Cons Cell Structure**:
```c
struct eshkol_tagged_cons_cell_t {
    eshkol_tagged_value_t car;  // 16 bytes
    eshkol_tagged_value_t cdr;  // 16 bytes
    // Total: 32 bytes per cell
};
```

### Basic List Operations

#### `cons`
**Syntax**: `(cons car cdr)`

Creates cons cell. Fundamental list constructor.

**Examples**:
```scheme
(cons 1 '())        ; => (1)
(cons 1 (cons 2 '())) ; => (1 2)
(cons 'a '(b c))    ; => (a b c)
```

**Type**: Pair constructor  
**Returns**: HEAP_PTR (consolidated pointer with HEAP_SUBTYPE_CONS)  
**Implementation**: [`cons()`](../lib/backend/collection_codegen.cpp:94)

---

#### `car`, `cdr`
**Syntax**: `(car pair)`, `(cdr pair)`

Extracts first element or rest of list. Also works with vectors/tensors.

**Examples**:
```scheme
(car '(1 2 3))   ; => 1
(cdr '(1 2 3))   ; => (2 3)
(car #(10 20 30)) ; => 10 (vector support)
```

**Type**: Pair accessors  
**Returns**: Tagged value (preserves type of stored element)  
**Implementation**: [`car()`](../lib/backend/collection_codegen.cpp:121), [`cdr()`](../lib/backend/collection_codegen.cpp:550)

---

#### `list`
**Syntax**: `(list expr...)`

Creates list from arguments.

**Examples**:
```scheme
(list 1 2 3)        ; => (1 2 3)
(list)              ; => '()
(list 'a (+ 1 1) 'c) ; => (a 2 c)
```

**Type**: List constructor  
**Evaluation**: Left-to-right (preserves side-effect order)  
**Implementation**: [`list()`](../lib/backend/collection_codegen.cpp:985)

---

#### `list*`
**Syntax**: `(list* e1 e2 ... en)`

Improper list constructor (last arg becomes tail, not element).

**Examples**:
```scheme
(list* 1 2 3 '(4 5))  ; => (1 2 3 4 5)
(list* 'a 'b 'c)      ; => (a b . c)
```

**Type**: Improper list constructor  
**Implementation**: [`codegenListStar()`](../lib/backend/llvm_codegen.cpp:25470)

---

### Compound Accessors

#### `cadr`, `caddr`, `cadddr`, etc.
**Syntax**: `(c[ad]+r list)` where `a` = car, `d` = cdr

Compound accessors (up to 4 levels deep).

**Examples**:
```scheme
(cadr '(1 2 3))    ; => 2 (car of cdr)
(caddr '(1 2 3))   ; => 3 (car of cdr of cdr)
(cddr '(1 2 3 4))  ; => (3 4)
```

**Supported**: cadr, caddr, cadddr, caar, cdar, cddr, caaar, caadr, cadar, cdaar, cdadr, cddar, cdddr  
**Optimization**: For vectors/tensors, uses direct indexing instead of traversal  
**Implementation**: [`codegenCompoundCarCdr()`](../lib/backend/llvm_codegen.cpp:23506)

---

### Predicates

#### `null?`, `pair?`, `list?`
**Syntax**: `(null? obj)`, `(pair? obj)`, `(list? obj)`

Type predicates for lists.

**Examples**:
```scheme
(null? '())        ; => #t
(pair? '(1 2))     ; => #t
(list? '(1 2 3))   ; => #t
(list? '(1 . 2))   ; => #f (improper list)
```

**Type**: Boolean predicates  
**Implementation**: [`isNull()`](../lib/backend/collection_codegen.cpp:1035), [`isPair()`](../lib/backend/collection_codegen.cpp:1078)

---

### Mutators

#### `set-car!`, `set-cdr!`
**Syntax**: `(set-car! pair value)`, `(set-cdr! pair value)`

Mutates cons cell fields (destructive update).

**Examples**:
```scheme
(define lst (list 1 2 3))
(set-car! lst 99)  ; lst => (99 2 3)
```

**Type**: Destructive update  
**Returns**: Unspecified  
**Implementation**: [`codegenSetCar()`](../lib/backend/llvm_codegen.cpp:24865)

---

### Higher-Order List Functions

#### `map`
**Syntax**: `(map proc list1 list2 ...)`

Applies procedure to corresponding elements of lists.

**Examples**:
```scheme
(map (lambda (x) (* x 2)) '(1 2 3))  ; => (2 4 6)
(map + '(1 2 3) '(10 20 30))         ; => (11 22 33)
```

**Type**: Higher-order transformation  
**Arity**: Supports multi-list mapping (proc must accept n arguments for n lists)  
**Implementation**: [`MapCodegen::map()`](../lib/backend/map_codegen.cpp)  
**Performance**: Iterative LLVM IR (not recursive) for efficiency

---

## Standard Library Reference

Eshkol's standard library provides 25 modules with 180+ functions. Access via:

```scheme
(import core.list.higher_order)
(import core.functional)
```

### Module Organization

**Core Modules** (25):
- `core.io` - File I/O, ports, display
- `core.strings` - String manipulation
- `core.json` - JSON parsing/generation
- `core.data.base64` - Base64 encoding
- `core.data.csv` - CSV processing
- `core.operators.arithmetic` - +, -, *, /, mod, quotient, gcd, lcm
- `core.operators.compare` - <, >, =, <=, >=
- `core.logic.boolean` - Boolean operations
- `core.logic.predicates` - Type predicates
- `core.logic.types` - Type conversions
- `core.functional.compose` - Function composition
- `core.functional.curry` - Currying
- `core.functional.flip` - Argument flipping
- `core.list.compound` - cadr, caddr, etc.
- `core.list.convert` - Type conversions
- `core.list.generate` - range, iota, make-list
- `core.list.higher_order` - fold, filter, any, every
- `core.list.query` - length, find, take, drop
- `core.list.search` - member, assoc, binary-search
- `core.list.sort` - Merge sort, quick sort
- `core.list.transform` - append, reverse, map
- `core.control.trampoline` - Tail call optimization

### Key Standard Library Functions

*Note: Complete list available in [`lib/stdlib.esk`](../lib/stdlib.esk)*

**List Processing** (from `core.list.*`):
- `length`, `append`, `reverse` - Basic utilities
- `filter`, `fold`, `fold-right` - Higher-order
- `member`, `assoc` - Search operations
- `sort`, `merge` - Sorting algorithms
- `partition`, `zip`, `unzip` - List transformations

**Functional** (from `core.functional`):
- `compose` - Function composition
- `curry`, `uncurry` - Currying transformations
- `flip` - Argument order reversal
- `partial` - Partial application

**Math Library** ([`lib/math.esk`](../lib/math.esk)):
- `det` - Determinant (LU decomposition)
- `inv` - Matrix inverse (Gauss-Jordan)
- `solve` - Linear system solver
- `cross` - Cross product (3D)
- `normalize` - Unit vector
- `variance`, `std`, `covariance` - Statistics
- `integrate` - Simpson's rule integration
- `newton` - Root finding

---

## Arithmetic & Comparison

### Arithmetic Operators

**Polymorphic**: Support int64, double, dual numbers (forward AD), and AD nodes (reverse AD)

#### `+`, `-`, `*`, `/`
**Syntax**: `(+ a b ...)`, `(- a b ...)`, `(* a b ...)`, `(/ a b ...)`

**Examples**:
```scheme
(+ 1 2 3)       ; => 6
(- 10 3)        ; => 7
(- 5)           ; => -5 (unary negation)
(* 2 3 4)       ; => 24
(/ 10 2)        ; => 5.0 (always returns double)
```

**Type Promotion**:
- Int64 + Int64 = Int64 (exact)
- Int64 + Double = Double (inexact)
- Double + Double = Double (inexact)

**AD Support**:
- Dual numbers: Automatic derivative propagation
- AD nodes: Computational graph construction

**Implementation**: [`polymorphicAdd()`](../lib/backend/arithmetic_codegen.cpp), etc.

---

#### `abs`, `quotient`, `remainder`, `modulo`
**Syntax**: `(abs x)`, `(quotient a b)`, `(remainder a b)`, `(modulo a b)`

**Examples**:
```scheme
(abs -5)           ; => 5
(quotient 17 5)    ; => 3
(remainder 17 5)   ; => 2
(modulo -7 3)      ; => 2 (sign matches divisor)
```

**Type**: Numeric operations  
**abs**: Preserves type (int→int, double→double)  
**Implementation**: [`codegenAbs()`](../lib/backend/llvm_codegen.cpp:9698)

---

#### `gcd`, `lcm`
**Syntax**: `(gcd a b)`, `(lcm a b)`

Greatest common divisor and least common multiple.

**Examples**:
```scheme
(gcd 48 18)  ; => 6
(lcm 12 18)  ; => 36
```

**Algorithm**: Euclidean algorithm  
**Implementation**: [`codegenGCD()`](../lib/backend/llvm_codegen.cpp:9920)

---

### Comparison Operators

#### `=`, `<`, `>`, `<=`, `>=`
**Syntax**: `(= a b)`, `(< a b)`, etc.

Numeric comparison with polymorphic type handling.

**Examples**:
```scheme
(< 1 2)      ; => #t
(>= 5.0 5)   ; => #t (mixed int/double)
(= 3 3)      ; => #t
```

**Type**: Boolean predicates  
**Optimization**: HoTT type system enables compile-time resolution when types are known  
**Implementation**: [`polymorphicCompare()`](../lib/backend/arithmetic_codegen.cpp)

---

#### `eq?`, `eqv?`, `equal?`
**Syntax**: `(eq? a b)`, `(eqv? a b)`, `(equal? a b)`

Equality predicates with different semantics.

- `eq?`: Pointer/identity equality
- `eqv?`: Operational equivalence (same as eq? for most types)
- `equal?`: Deep structural equality (recursive for lists)

**Examples**:
```scheme
(eq? 'a 'a)          ; => #t
(equal? '(1 2) '(1 2)) ; => #t
(eq? '(1 2) '(1 2))    ; => #f (different pointers)
```

**Implementation**: [`codegenEq()`](../lib/backend/llvm_codegen.cpp:11893), [`codegenEqual()`](../lib/backend/llvm_codegen.cpp:12036)

---

## Math Functions

All math functions support dual numbers (forward-mode AD) and AD nodes (reverse-mode AD) automatically.

### Trigonometric Functions

**Functions**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`

**Examples**:
```scheme
(sin pi)         ; => ~0.0
(cos 0.0)        ; => 1.0
(atan2 1.0 1.0)  ; => 0.785... (π/4)
```

**AD Support**: Chain rule automatically applied with dual numbers  
**Implementation**: [`codegenMathFunction()`](../lib/backend/llvm_codegen.cpp:9569)

---

### Hyperbolic Functions

**Functions**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Examples**:
```scheme
(sinh 1.0)  ; => 1.175...
(tanh 0.0)  ; => 0.0
```

**Implementation**: C math library bindings with dual number support

---

### Exponential & Logarithmic

**Functions**: `exp`, `exp2`, `log`, `log10`, `log2`, `pow`, `sqrt`, `cbrt`

**Examples**:
```scheme
(exp 1.0)       ; => 2.718... (e)
(log e)         ; => 1.0
(pow 2 10)      ; => 1024.0
(sqrt 16)       ; => 4.0
```

**AD Support**: All functions differentiable  
**Implementation**: [`dualExp()`](../lib/backend/autodiff_codegen.cpp:287), etc.

---

### Rounding Functions

**Functions**: `floor`, `ceiling`, `round`, `truncate`

**Examples**:
```scheme
(floor 3.7)    ; => 3.0
(ceiling 3.2)  ; => 4.0
(round 3.5)    ; => 4.0
```

---

### Numeric Predicates

**Functions**: `number?`, `integer?`, `real?`, `zero?`, `positive?`, `negative?`, `even?`, `odd?`, `nan?`, `infinite?`, `finite?`

**Examples**:
```scheme
(even? 4)       ; => #t
(zero? 0.0)     ; => #t
(nan? (/ 0.0 0.0))  ; => #t
(finite? 1e100) ; => #t
```

**Implementation**: [`codegenNumericPredicate()`](../lib/backend/llvm_codegen.cpp:11836)

---

## String Operations

Strings are heap-allocated with header (HEAP_SUBTYPE_STRING).

#### `string-length`, `string-ref`, `string-set!`
**Syntax**: `(string-length str)`, `(string-ref str idx)`, `(string-set! str idx char)`

**Examples**:
```scheme
(string-length "hello")    ; => 5
(string-ref "hello" 1)     ; => #\e
(string-set! s 0 #\H)      ; Mutates string
```

**Implementation**: [`StringIOCodegen`](../lib/backend/string_io_codegen.cpp)

---

#### `string-append`, `substring`
**Syntax**: `(string-append str...)`, `(substring str start end)`

**Examples**:
```scheme
(string-append "Hello" " " "World")  ; => "Hello World"
(substring "hello" 1 4)              ; => "ell"
```

---

#### `string=?`, `string<?`, etc.
**Syntax**: `(string=? s1 s2)`, `(string<? s1 s2)`, etc.

Lexicographic string comparison.

**Functions**: `string=?`, `string<?`, `string>?`, `string<=?`, `string>=?`

---

#### `number->string`, `string->number`
**Syntax**: `(number->string n)`, `(string->number str)`

Conversion between numbers and strings.

**Examples**:
```scheme
(number->string 42)     ; => "42"
(string->number "3.14") ; => 3.14
```

---

## File & I/O

### Port Operations

#### `open-input-file`, `open-output-file`, `close-port`
**Syntax**: `(open-input-file filename)`, `(open-output-file filename)`, `(close-port port)`

**Examples**:
```scheme
(define in (open-input-file "data.txt"))
(define line (read-line in))
(close-port in)
```

**Type**: File handle (opaque pointer)  
**Implementation**: [`openInputFile()`](../lib/backend/string_io_codegen.cpp)

---

#### `read-line`, `write-line`
**Syntax**: `(read-line port)`, `(write-line port str)`

**Examples**:
```scheme
(define in (open-input-file "file.txt"))
(define line (read-line in))  ; Returns string or EOF

(define out (open-output-file "out.txt"))
(write-line out "Hello, World!")
```

---

#### `display`, `newline`
**Syntax**: `(display obj [port])`, `(newline [port])`

Output operations. `display` uses homoiconic S-expression printer for lambdas.

**Examples**:
```scheme
(display "Hello")
(display (lambda (x) (* x 2)))  ; Shows: (lambda (x) (* x 2))
(newline)
```

**Implementation**: [`display()`](../lib/backend/string_io_codegen.cpp) with [`eshkol_display_value`](../lib/core/printer.cpp) runtime

---

## System Operations

### Environment & Process

#### `getenv`, `setenv`, `unsetenv`
**Syntax**: `(getenv name)`, `(setenv name value overwrite)`, `(unsetenv name)`

**Examples**:
```scheme
(getenv "PATH")
(setenv "MY_VAR" "value" 1)
```

---

#### `system`, `exit`
**Syntax**: `(system command)`, `(exit [code])`

**Examples**:
```scheme
(system "ls -la")
(exit 0)
```

---

#### `current-seconds`, `sleep`
**Syntax**: `(current-seconds)`, `(sleep milliseconds)`

**Examples**:
```scheme
(define start (current-seconds))
(sleep 1000)  ; Sleep 1 second
```

---

### File System

#### `file-exists?`, `directory-exists?`
#### `file-delete`, `file-rename`
#### `make-directory`, `delete-directory`
#### `directory-list`, `current-directory`

**Examples**:
```scheme
(file-exists? "data.txt")
(directory-list "/tmp")
(make-directory "output")
```

**Implementation**: [`SystemCodegen`](../lib/backend/system_codegen.cpp)

---

## Hash Tables

#### `make-hash-table`
**Syntax**: `(make-hash-table)`

Creates mutable hash table.

**Implementation**: Uses open addressing with linear probing  
**Type**: HEAP_PTR with HEAP_SUBTYPE_HASH

---

#### `hash-ref`, `hash-set!`, `hash-has-key?`
**Syntax**: `(hash-ref ht key [default])`, `(hash-set! ht key value)`, `(hash-has-key? ht key)`

**Examples**:
```scheme
(define ht (make-hash-table))
(hash-set! ht "name" "Alice")
(hash-ref ht "name")       ; => "Alice"
(hash-has-key? ht "age")   ; => #f
```

---

#### `hash-remove!`, `hash-clear!`
#### `hash-keys`, `hash-values`, `hash-count`

**Examples**:
```scheme
(hash-keys ht)    ; => List of keys
(hash-count ht)   ; => Number of entries
```

**Implementation**: [`HashCodegen`](../lib/backend/hash_codegen.cpp)

---

## Type System

Eshkol uses a **triple-layer type system**:

1. **Runtime Tagged Values** (16-byte discriminated unions)
2. **HoTT Compile-Time Types** (dependent types, proof erasure)
3. **LLVM Types** (code generation layer)

### Tagged Value Types

**Immediate Types** (0-7, value in data field):
- `NULL` (0) - Empty list, unspecified
- `INT64` (1) - 64-bit signed integer
- `DOUBLE` (2) - IEEE 754 double
- `CHAR` (3) - Unicode codepoint
- `BOOL` (4) - Boolean (#t/#f)
- `SYMBOL` (5) - Interned symbol

**Consolidated Pointer Types** (8-9, subtype in object header):
- `HEAP_PTR` (8) - Cons, string, vector, tensor, hash (header subtype distinguishes)
- `CALLABLE` (9) - Closure, lambda, AD node, continuation (header subtype)

**Dual Number** (10) - Forward-mode AD (primal, tangent)

### Type Predicates

**Functions**: `number?`, `integer?`, `real?`, `string?`, `char?`, `boolean?`, `symbol?`, `procedure?`, `vector?`

**Implementation**: Runtime type tag inspection

---

### HoTT Type Annotations

**Syntax**: `(: expr type)`

Compile-time type annotations for gradual typing.

**Examples**:
```scheme
(: 42 Integer)
(: (lambda (x : Real) : Real (* x 2.0)) (→ Real Real))
```

**Types**: Integer, Real, Number, Boolean, Char, String, List, Vector, Tensor, Function arrows  
**Erasure**: Proof types erased at runtime (dependent types compile to nothing)  
**Implementation**: [`TypeChecker`](../lib/types/type_checker.cpp)

---

## Performance & Implementation Notes

### Compilation Pipeline

1. **Parse** → AST (S-expressions)
2. **Macro Expansion** → Expanded AST
3. **HoTT Type Checking** → Annotated AST (gradual)
4. **LLVM IR Generation** → Optimized IR
5. **Native Code** → Platform-specific machine code

**Optimization Levels**:
- Type-directed optimizations (when HoTT types known)
- Tail call optimization (self-recursive functions)
- Arena-based memory pooling (no GC pauses)

---

### Memory Architecture

**OALR** (Ownership-Aware Lexical Regions):
- Arena allocation (bump pointer, O(1) allocation)
- Lexical region lifetimes (deterministic deallocation)
- Zero-copy tensor views (reshape/slice without allocation)

**Allocation Hierarchy**:
1. Stack allocas (function-local)
2. Arena regions (lexical scope)
3. Global arena (persistent data)

---

### Closure Implementation

**Structure** (32 bytes):
```c
struct eshkol_closure_t {
    uint64_t func_ptr;        // Function pointer
    void* env;                // Capture environment
    uint64_t sexpr_ptr;       // S-expression for display
    uint8_t return_type;      // Return type category
    uint8_t input_arity;      // Number of parameters
    uint8_t flags;            // Variadic, etc.
    uint8_t reserved[5];
};
```

**Capture Environment**:
```c
struct eshkol_closure_env_t {
    uint64_t packed_info;     // num_captures | fixed_params | variadic_flag
    eshkol_tagged_value_t captures[];  // Mutable via pointer-passing
};
```

**Features**:
- Mutable captures (set! inside closures)
- Nested closures (arbitrary depth)
- Variadic functions (rest parameters)
- S-expression preservation (homoiconicity)

---

### Tensor Memory Layout

**Tensor Structure** (32 bytes):
```c
struct eshkol_tensor_t {
    uint64_t* dims;          // [d0, d1, ..., d(n-1)]
    uint64_t num_dimensions; // Rank
    int64_t* elements;       // Doubles stored as int64 bits
    uint64_t total_elements; // Product of dims
};
```

**Element Storage**: Doubles stored as int64 bit patterns (enables uniform storage for doubles and AD node pointers)

**Row-Major Order**: `elem[i,j,k] = elements[i*stride0 + j*stride1 + k]`

---

## Code Examples

### Neural Network Example

```scheme
(import core.functional)
(import core.list.higher_order)

; Sigmoid activation
(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

; Forward pass (2-layer network)
(define (forward W1 W2 x)
  (let* ((h (tensor-apply (matmul W1 x) sigmoid))
         (y (matmul W2 h)))
    y))

; Training with gradient descent
(define (train W1 W2 x target learning-rate)
  (let* ((loss-fn (lambda (w1 w2)
                    (let ((pred (forward w1 w2 x)))
                      (* 0.5 (norm (tensor-sub pred target))))))
         (grad-w1 (gradient (lambda (w) (loss-fn w W2)) W1))
         (grad-w2 (gradient (lambda (w) (loss-fn W1 w)) W2)))
    (values (tensor-sub W1 (tensor-mul learning-rate grad-w1))
            (tensor-sub W2 (tensor-mul learning-rate grad-w2)))))
```

### Physics Simulation Example

```scheme
; Electromagnetic field analysis
(define (analyze-field E B point)
  (let ((div-E (divergence E point))
        (curl-B (curl B point)))
    (list 'charge-density div-E
          'current-density curl-B)))

; Heat equation solver
(define (heat-step u alpha dt)
  (let ((laplacian-u (laplacian u)))
    (tensor-add u (tensor-mul (* alpha dt) laplacian-u))))
```

---

## Implementation Statistics

**Codebase Size**: 67,079 lines of production C++  
**Main Backend**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) - 27,079 lines  
**Compiler Modules**: 20+ specialized code generators  
**Test Suite**: 300+ test files  
**Verified Operations**: 70+ special forms, 180+ standard library functions  

---

## See Also

- [**Architecture Guide**](ESHKOL_V1_ARCHITECTURE.md) - Complete system architecture
- [**Language Guide**](../ESHKOL_LANGUAGE_GUIDE.md) - Tutorial and examples  
- [**Quick Reference**](../ESHKOL_QUICK_REFERENCE.md) - One-page cheat sheet  
- [**Type System Extension**](HOTT_TYPE_SYSTEM_EXTENSION.md) - HoTT types and dependent types

---

**Copyright** © 2025 tsotchke  
**License**: MIT