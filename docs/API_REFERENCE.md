# Eshkol v1.3.0-evolve API Reference

**Version**: 1.3.0-evolve
**Last Updated**: 2026-05-20
**Audience**: Scientific Computing & AI Systems Programming

This comprehensive reference documents all special forms, functions, and operations in the Eshkol language. All documentation is code-verified against the production compiler implementation (~232,000 lines of LLVM-based C++ code, 555+ builtins).

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
12. [Standard Library](#standard-library) (25+ modules)
13. [Exact Arithmetic](#exact-arithmetic) (v1.1)
14. [Complex Numbers](#complex-numbers) (v1.1)
15. [Continuations & Control Flow](#continuations--control-flow) (v1.1)
16. [Parallel Primitives](#parallel-primitives) (v1.1)
17. [Consciousness Engine](#consciousness-engine) (v1.1)
18. [GPU Operations](#gpu-operations) (v1.1)
19. [Signal Processing](#signal-processing) (v1.1)
20. [Bytevectors](#bytevectors) (v1.1)
21. [Environment Variables](#environment-variables)
22. [Statistics](#statistics) (v1.1)
23. [Tensor Utilities](#tensor-utilities) (v1.1)
24. [Special Functions](#special-functions) (v1.1)
25. [Mathematical Constants](#mathematical-constants) (v1.1)
26. [ODE Solvers](#ode-solvers) (v1.1)
27. [Random Numbers](#random-numbers) (v1.1)
28. [Web Platform (WASM)](#web-platform-wasm) (v1.1)
29. [Thread Pool & Promise Builtins](#thread-pool--promise-builtins) (v1.1)
30. [Command-Line Tools](#command-line-tools)

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
**Implementation**: [`codegenDefine()`](../lib/backend/llvm_codegen.cpp)

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
**Returns**: Closure (40-byte structure with captures and S-expression for homoiconicity)  
**Implementation**: [`codegenLambda()`](../lib/backend/llvm_codegen.cpp)
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
**Implementation**: [`codegenLet()`](../lib/backend/llvm_codegen.cpp), [`BindingCodegen`](../lib/backend/binding_codegen.cpp)

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
**Implementation**: [`codegenSet()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenDo()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenQuotedAST()`](../lib/backend/llvm_codegen.cpp), [`HomoiconicCodegen`](../lib/backend/homoiconic_codegen.cpp)

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
**Implementation**: [`codegenGuard()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenRaise()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenValues()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenCallWithValues()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenMatch()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenWithRegion()`](../lib/backend/llvm_codegen.cpp)

---

#### `owned`, `move`, `borrow`, `shared`
**Syntax**: `(owned expr)`, `(move value)`, `(borrow val body...)`, `(shared expr)`

Ownership annotations for linear types and reference counting (compile-time tracking).

**Type**: Ownership markers  
**Implementation**: [`codegenOwned()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenExtern()`](../lib/backend/llvm_codegen.cpp)

---

#### `extern-var`
**Syntax**: `(extern-var type name)`

Declares external C variable.

**Type**: FFI variable  
**Implementation**: [`codegenExternVar()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`buildSymbolicDerivative()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenDerivative()`](../lib/backend/llvm_codegen.cpp), [`AutodiffCodegen::derivative()`](../lib/backend/autodiff_codegen.cpp)

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
**Implementation**: [`codegenGradient()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenJacobian()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenHessian()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenDivergence()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenCurl()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenLaplacian()`](../lib/backend/llvm_codegen.cpp)

---

#### `directional-derivative`
**Syntax**: `(directional-derivative f point direction)`

Computes directional derivative: `D_v f = ∇f · v`

**Examples**:
```scheme
(define (f v) (* (vref v 0) (vref v 1)))
(directional-derivative f #(2.0 3.0) #(1.0 0.0))  ; => 3.0
```

**Mode**: Gradient dot product  
**Input**: Scalar field `f: ℝⁿ → ℝ`, point, direction vector  
**Output**: Scalar (rate of change in specified direction)  
**Implementation**: [`codegenDirectionalDerivative()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`zeros()`](../lib/backend/tensor_codegen.cpp), [`ones()`](../lib/backend/tensor_codegen.cpp)

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
**Implementation**: [`eye()`](../lib/backend/tensor_codegen.cpp)

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
**Implementation**: [`arange()`](../lib/backend/tensor_codegen.cpp)

---

#### `linspace`
**Syntax**: `(linspace start end num)`

Creates `num` evenly spaced values from `start` to `end` (inclusive).

**Examples**:
```scheme
(linspace 0.0 1.0 5)  ; => #(0.0 0.25 0.5 0.75 1.0)
```

**Type**: Linear spacing constructor  
**Implementation**: [`linspace()`](../lib/backend/tensor_codegen.cpp)

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
**Implementation**: [`tensorGet()`](../lib/backend/tensor_codegen.cpp), [`codegenTensorVectorRef()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`tensorSet()`](../lib/backend/tensor_codegen.cpp)

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
**Implementation**: [`transpose()`](../lib/backend/tensor_codegen.cpp)

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
**Implementation**: [`reshape()`](../lib/backend/tensor_codegen.cpp)

---

#### `flatten`
**Syntax**: `(flatten tensor)`

Flattens N-dimensional tensor to 1D (zero-copy view).

**Examples**:
```scheme
(flatten #((1 2) (3 4)))  ; => #(1 2 3 4)
```

**Type**: Dimensionality reduction  
**Implementation**: [`codegenFlatten()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`tensorShape()`](../lib/backend/tensor_codegen.cpp)

---

#### `tensor-save`
**Syntax**: `(tensor-save path tensor)`

Writes a tensor checkpoint to disk.

**Examples**:
```scheme
(tensor-save "/tmp/tensor.em" #(1 2 3))  ; => #t
```

**Type**: Serialization  
**Returns**: `#t` on success, `#f` on failure

---

#### `tensor-load`
**Syntax**: `(tensor-load path)`

Loads a tensor checkpoint from disk.

**Examples**:
```scheme
(tensor-load "/tmp/tensor.em")  ; => #(1 2 3)
```

**Type**: Serialization  
**Returns**: Tensor on success, `()`/null-equivalent on failure

---

#### `model-save`
**Syntax**: `(model-save path entries)`

Writes a multi-tensor checkpoint to disk. `entries` is a list of `(name . tensor)` pairs.

**Examples**:
```scheme
(model-save "/tmp/model.em" (list (cons "w" #(1 2)) (cons "b" #(3))))  ; => #t
```

**Type**: Serialization  
**Returns**: `#t` on success, `#f` on failure

---

#### `model-load`
**Syntax**: `(model-load path)`

Loads a multi-tensor checkpoint from disk.

**Examples**:
```scheme
(model-load "/tmp/model.em")  ; => (("w" . #(1 2)) ("b" . #(3)))
```

**Type**: Serialization  
**Returns**: List of `(name . tensor)` pairs on success, `()`/null-equivalent on failure

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
**Implementation**: [`tensorArithmetic()`](../lib/backend/tensor_codegen.cpp)

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
**Implementation**: [`tensorDot()`](../lib/backend/tensor_codegen.cpp), [`codegenMatmul()`](../lib/backend/llvm_codegen.cpp)

---

#### `trace`
**Syntax**: `(trace matrix)`

Sum of diagonal elements.

**Examples**:
```scheme
(trace #((1 2) (3 4)))  ; => 5
```

**Type**: Matrix invariant  
**Implementation**: [`codegenTrace()`](../lib/backend/llvm_codegen.cpp)

---

#### `norm`
**Syntax**: `(norm vector)`

Euclidean (L2) norm: `√(Σ xᵢ²)`

**Examples**:
```scheme
(norm #(3.0 4.0))  ; => 5.0
```

**Type**: Vector magnitude  
**Implementation**: [`codegenNorm()`](../lib/backend/llvm_codegen.cpp)

---

#### `outer`
**Syntax**: `(outer u v)`

Outer product of vectors: `(u ⊗ v)[i,j] = uᵢ vⱼ`

**Examples**:
```scheme
(outer #(1 2) #(3 4 5))  ; => #((3 4 5) (6 8 10))
```

**Type**: Tensor product  
**Implementation**: [`codegenOuterProduct()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`tensorSum()`](../lib/backend/tensor_codegen.cpp), [`tensorMean()`](../lib/backend/tensor_codegen.cpp)

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
**Implementation**: [`tensorReduceAll()`](../lib/backend/tensor_codegen.cpp), [`tensorReduceWithDim()`](../lib/backend/tensor_codegen.cpp)

---

#### `tensor-apply`
**Syntax**: `(tensor-apply tensor func)`

Maps function over tensor elements.

**Examples**:
```scheme
(tensor-apply #(1 2 3) (lambda (x) (* x x)))  ; => #(1 4 9)
```

**Type**: Element-wise map
**Implementation**: [`tensorApply()`](../lib/backend/tensor_codegen.cpp)

---

### Broadcasting Rules

When operating on tensors of different shapes, Eshkol applies **NumPy-style broadcasting**:
dimensions are compared right-to-left, and are compatible when (a) they are equal, or (b) one
of them is 1. The output shape is the element-wise maximum of the input shapes.

```
Shape (3, 4, 1) + Shape (1, 4, 5) → Shape (3, 4, 5)
Shape (5,)      + Shape (3, 5)     → Shape (3, 5)
Shape (1,)      + Shape (4, 3)     → Shape (4, 3)
Shape (3, 1)    + Shape (3,)       → Error: trailing dims (1 vs empty, not 1 vs 1)
```

Broadcasting applies to: `tensor-add`, `tensor-sub`, `tensor-mul`, `tensor-div`, and all
element-wise operations. Internally implemented via `eshkol_broadcast_elementwise_f64()`
in [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

---

### Tensor Random Number Generation

Compiler-level builtins for generating random tensors. These are distinct from the stdlib
`random-tensor` / `random-normal-tensor` functions — the builtins accept variadic dimension
arguments directly rather than a list.

#### `rand`

**Syntax:** `(rand d₁ d₂ ...)` → Tensor

Uniform random tensor with values in [0, 1). Uses `drand48()` internally.

**Implementation:** `tensorRand` in [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

```scheme
(rand 3)       ; => #(0.482 0.917 0.123)
(rand 2 3)     ; => 2×3 tensor with uniform random values
```

#### `randn`

**Syntax:** `(randn d₁ d₂ ...)` → Tensor

Standard normal N(0, 1) random tensor. Uses the Box-Muller transform internally:
generates pairs of uniform samples u₁, u₂ and computes z = √(-2 ln u₁) · cos(2π u₂).

**Implementation:** `tensorRandn` in [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

```scheme
(randn 5)      ; => #(-0.31 1.24 0.08 -1.53 0.67) (varies)
(randn 3 3)    ; => 3×3 tensor with standard normal values
```

#### `randint`

**Syntax:** `(randint lo hi d₁ d₂ ...)` → Tensor

Random integer tensor with values in [lo, hi).

**Implementation:** `tensorRandint` in [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

```scheme
(randint 0 10 5)     ; => #(3.0 7.0 1.0 9.0 4.0) (integers as doubles)
(randint 1 7 2 3)    ; => 2×3 tensor, dice rolls
```

---

### Tensor Linear Algebra

Matrix decompositions factor matrices into structured products, enabling efficient solution
of linear systems, eigenvalue problems, least-squares optimization, and dimensionality
reduction. All decompositions operate on 2D tensors (matrices).

#### `tensor-lu`

**Syntax:** `(tensor-lu A)` → List: (L U P)

LU decomposition with partial pivoting: PA = LU, where L is lower-triangular with unit
diagonal, U is upper-triangular, and P is a permutation matrix.

**Algorithm:** Gaussian elimination with partial pivoting (row swaps for numerical stability).

**Complexity:** O(n³/3) for n×n matrices.

**Implementation:** `tensorLU` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(define A #((2.0 1.0) (1.0 3.0)))
(let ((result (tensor-lu A)))
  (let ((L (car result)) (U (cadr result)) (P (caddr result)))
    ;; P·A = L·U
    (display L) (display U)))
```

#### `tensor-det`

**Syntax:** `(tensor-det A)` → Number

Matrix determinant computed via LU decomposition: det(A) = ∏ diag(U) · sign(P).

**Complexity:** O(n³/3) — dominated by the LU factorization.

**Implementation:** `tensorDet` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(tensor-det #((2.0 1.0) (1.0 3.0)))   ; => 5.0
(tensor-det #((1.0 0.0) (0.0 1.0)))   ; => 1.0 (identity)
```

#### `tensor-inverse`

**Syntax:** `(tensor-inverse A)` → Tensor

Matrix inverse A⁻¹ via LU decomposition with forward and back substitution.

**Complexity:** O(n³). Raises an error for singular matrices (det ≈ 0).

**Implementation:** `tensorInverse` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(define A #((2.0 1.0) (1.0 3.0)))
(define Ainv (tensor-inverse A))
(tensor-matmul A Ainv)   ; => ~identity matrix
```

#### `tensor-solve`

**Syntax:** `(tensor-solve A b)` → Tensor

Solves the linear system Ax = b via LU decomposition with forward and back substitution.

**Complexity:** O(n³/3) for LU + O(n²) for substitution.

**Implementation:** `tensorSolve` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(define A #((2.0 1.0) (1.0 3.0)))
(define b #(3.0 5.0))
(tensor-solve A b)   ; => #(0.8 1.4) — solution x where Ax = b
```

#### `tensor-cholesky`

**Syntax:** `(tensor-cholesky A)` → Tensor

Cholesky decomposition: A = LL^T where L is lower-triangular. Requires A to be symmetric
positive-definite.

**Algorithm:** Cholesky-Banachiewicz (row-wise computation).

**Complexity:** O(n³/6) — roughly half the cost of LU decomposition.

**Use case:** Solving symmetric positive-definite systems (covariance matrices, kernel
matrices in Gaussian processes). Twice as efficient as general LU.

**Implementation:** `tensorCholesky` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(define A #((4.0 2.0) (2.0 3.0)))   ; symmetric positive-definite
(define L (tensor-cholesky A))
;; L·L^T = A
```

#### `tensor-qr`

**Syntax:** `(tensor-qr A)` → List: (Q R)

QR decomposition: A = QR where Q is orthogonal (Q^T Q = I) and R is upper-triangular.

**Algorithm:** Householder reflections — numerically stable and efficient.

**Complexity:** O(2n²m - 2n³/3) for an m×n matrix with m ≥ n.

**Use case:** Least-squares problems (∥Ax - b∥² minimization), eigenvalue algorithms (QR iteration).

**Implementation:** `tensorQR` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(define A #((1.0 1.0) (0.0 1.0) (1.0 0.0)))   ; 3×2
(let ((result (tensor-qr A)))
  (let ((Q (car result)) (R (cadr result)))
    ;; Q is 3×2 orthogonal, R is 2×2 upper-triangular
    ;; Q·R = A
    ))
```

#### `tensor-svd`

**Syntax:** `(tensor-svd A)` → List: (U S V^T)

Singular Value Decomposition: A = UΣV^T where U and V are orthogonal and Σ is diagonal
with non-negative entries (singular values) in descending order.

**Algorithm:** One-sided Jacobi SVD.

**Complexity:** O(n²m) for m×n matrices.

**Use case:** Principal Component Analysis (PCA), low-rank approximation (truncated SVD),
matrix pseudoinverse, condition number computation.

**Implementation:** `tensorSVD` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(define A #((1.0 2.0) (3.0 4.0) (5.0 6.0)))   ; 3×2
(let ((result (tensor-svd A)))
  (let ((U (car result)) (S (cadr result)) (Vt (caddr result)))
    ;; U is 3×2, S is #(σ₁ σ₂), Vt is 2×2
    ;; A ≈ U · diag(S) · Vt
    ))
```

#### `einsum`

**Syntax:** `(einsum spec t ...)` → Tensor

Einstein summation notation — a compact specification language for tensor contractions,
traces, transposes, and outer products. The `spec` string follows NumPy's einsum convention.

**Supported patterns:**
| Spec | Operation | Example |
|------|-----------|---------|
| `"ij,jk->ik"` | Matrix multiplication | `(einsum "ij,jk->ik" A B)` |
| `"ij->ji"` | Transpose | `(einsum "ij->ji" A)` |
| `"ii->"` | Trace | `(einsum "ii->" A)` → scalar |
| `"ij,ij->"` | Frobenius inner product | `(einsum "ij,ij->" A B)` → scalar |
| `"i,i->"` | Dot product | `(einsum "i,i->" u v)` → scalar |

**Implementation:** `tensorEinsum` in [lib/backend/tensor_linalg_codegen.cpp](../lib/backend/tensor_linalg_codegen.cpp).

```scheme
(define A #((1.0 2.0) (3.0 4.0)))
(define B #((5.0 6.0) (7.0 8.0)))
(einsum "ij,jk->ik" A B)   ; matrix multiply: #((19.0 22.0) (43.0 50.0))
(einsum "ii->" A)            ; trace: 5.0
(einsum "ij->ji" A)          ; transpose: #((1.0 3.0) (2.0 4.0))
```

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
**Implementation**: [`cons()`](../lib/backend/collection_codegen.cpp)

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
**Implementation**: [`car()`](../lib/backend/collection_codegen.cpp), [`cdr()`](../lib/backend/collection_codegen.cpp)

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
**Implementation**: [`list()`](../lib/backend/collection_codegen.cpp)

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
**Implementation**: [`codegenListStar()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenCompoundCarCdr()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`isNull()`](../lib/backend/collection_codegen.cpp), [`isPair()`](../lib/backend/collection_codegen.cpp)

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
**Implementation**: [`codegenSetCar()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenAbs()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenGCD()`](../lib/backend/llvm_codegen.cpp)

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

**Implementation**: [`codegenEq()`](../lib/backend/llvm_codegen.cpp), [`codegenEqual()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`codegenMathFunction()`](../lib/backend/llvm_codegen.cpp)

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
**Implementation**: [`dualExp()`](../lib/backend/autodiff_codegen.cpp), etc.

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

**Implementation**: [`codegenNumericPredicate()`](../lib/backend/llvm_codegen.cpp)

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

#### `string-byte-length`
**Syntax**: `(string-byte-length str)`

Returns the string's **byte** length (the heap object header's `size` field,
minus the trailing NUL) — distinct from `string-length`, which counts
**codepoints**. For ASCII-only strings the two agree; for strings containing
multibyte UTF-8 they diverge. This matters at the FFI boundary: C functions
like `fwrite` take a byte count, so `(fwrite s 1 (string-length s) f)`
silently truncates a string containing e.g. an em-dash (`—`, 3 bytes / 1
char). Use `string-byte-length` there instead:

```scheme
(string-length "café")       ; => 4  (4 codepoints)
(string-byte-length "café")  ; => 5  (the é is 2 bytes in UTF-8)

(fwrite s 1 (string-byte-length s) f)  ; correct — writes every byte
(fwrite s 1 (string-length s) f)       ; WRONG on multibyte input — truncates
```

**Implementation**: [`StringIOCodegen::stringByteLength`](../lib/backend/string_io_codegen.cpp), backed by the existing runtime helper `eshkol_string_byte_length` (`lib/core/runtime_string.cpp`).

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

## Exact Arithmetic

Added in v1.1. Full R7RS numeric tower with arbitrary-precision integers and exact rationals.

### Bignum Operations

Bignums are automatically promoted from int64 on overflow: `(+ 9223372036854775807 1)` produces a bignum.

#### Arithmetic
All standard arithmetic operators (`+`, `-`, `*`, `/`, `quotient`, `remainder`, `modulo`, `gcd`, `lcm`, `expt`, `abs`, `min`, `max`) work transparently with bignums. R7RS rule: exact + inexact → inexact.

#### Predicates
- `(exact? n)` — Returns `#t` if n is exact (int64, bignum, or rational)
- `(inexact? n)` — Returns `#t` if n is inexact (double)
- `(integer? n)` — Returns `#t` for int64 and bignum values

#### Conversion
- `(exact->inexact n)` — Converts to double (may lose precision for large bignums)
- `(inexact->exact n)` — Converts double to nearest integer
- `(number->string n)` — Works correctly with bignums (not pointer bits)
- `(string->number s)` — Parses large integers as bignums when they exceed int64 range

#### Bitwise Operations
- `(bitwise-and a b)`, `(bitwise-or a b)`, `(bitwise-xor a b)` — Bitwise ops (two's complement for bignums)
- `(bitwise-not a)` — Bitwise complement
- `(arithmetic-shift a n)` — Shift left (positive n) or right (negative n)
- `(bit-count n)` — Population count

### Rational Numbers

Exact fractions with automatic simplification.

- `(make-rational num den)` — Creates rational number, auto-simplifies
- `(numerator r)` — Returns numerator
- `(denominator r)` — Returns denominator
- `(rational? x)` — Returns `#t` for rational values

All arithmetic and comparison operators work with rationals.

---

## Complex Numbers

First-class complex number type with full AD support.

### Construction
- `(make-rectangular real imag)` — Creates complex from real and imaginary parts
- `(make-polar magnitude angle)` — Creates complex from polar coordinates

### Accessors
- `(real-part z)` — Returns real component
- `(imag-part z)` — Returns imaginary component
- `(magnitude z)` — Returns |z| = sqrt(real² + imag²)
- `(angle z)` — Returns argument (phase angle)

### Predicates
- `(complex? z)` — Returns `#t` for complex values

### Arithmetic
All standard operators (`+`, `-`, `*`, `/`) work with complex numbers. Division uses Smith's formula for numerical stability with large magnitudes.

```scheme
(define z1 (make-rectangular 3.0 4.0))
(magnitude z1)      ; => 5.0
(angle z1)          ; => 0.9273... (atan(4/3))
(+ z1 z1)           ; => 6.0+8.0i
(* z1 (make-rectangular 0.0 1.0))  ; => -4.0+3.0i
```

---

## Continuations & Control Flow

Eshkol implements full R7RS continuations with `call/cc`, `dynamic-wind`, and structured
exception handling via `guard`/`raise`. Continuations are first-class values that capture the
"rest of the computation" — invoking a continuation abandons the current computation and
resumes from the captured point.

**Performance note:** Continuation capture is O(stack-depth). Avoid capturing continuations
in hot inner loops; prefer explicit control flow (`let/ec`, early return patterns) when
performance is critical.

### `call/cc` (call-with-current-continuation)

**Syntax:** `(call/cc proc)` or `(call-with-current-continuation proc)`

Captures the current continuation as a first-class escape procedure and passes it to `proc`.
When the escape procedure is invoked with a value, execution resumes at the point where
`call/cc` was called, returning that value.

```scheme
(call/cc (lambda (k)
  (k 42)))  ; => 42

;; Non-local exit
(define (find-first pred lst)
  (call/cc (lambda (return)
    (for-each (lambda (x)
      (when (pred x) (return x)))
      lst)
    #f)))

(find-first even? '(1 3 4 5))  ; => 4
```

### `dynamic-wind`

**Syntax:** `(dynamic-wind before thunk after)`

Establishes entry and exit handlers that execute even during non-local exits via continuations.
`before` is called on entry (including re-entry via captured continuation), `after` is called
on exit (including exit via captured continuation).

**Guarantee:** The `after` thunk is always executed when control leaves the dynamic extent
of `thunk`, regardless of the exit mechanism (normal return, exception, or continuation invocation).

```scheme
(dynamic-wind
  (lambda () (display "entering\n"))
  (lambda () (+ 1 2))
  (lambda () (display "leaving\n")))
;; Prints: entering, leaving
;; Returns: 3
```

### `guard`

**Syntax:** `(guard (var clause ...) body ...)`

R7RS structured exception handler. Evaluates `body`; if an exception is raised (via `raise`),
binds the exception object to `var` and evaluates the `clause`s like `cond`.

Each clause has the form `(test expr ...)`. If `test` evaluates to true, the corresponding
`expr` is evaluated and its value returned. An `else` clause serves as the default handler.

```scheme
(guard (exn
  ((string? exn) (string-append "Error: " exn))
  ((number? exn) (string-append "Code: " (number->string exn)))
  (else "Unknown error"))
  (raise "file not found"))
;; => "Error: file not found"

(guard (exn
  (else (display "caught!\n")))
  (/ 1 0))
;; Prints: caught!
```

### `raise`

**Syntax:** `(raise obj)`

Raises an exception with `obj` as the exception value. Unwinds the call stack to the nearest
`guard` or `with-exception-handler`. If no handler is found, the program terminates with an
error message.

The exception object can be any Eshkol value — strings, numbers, records, lists, etc.

```scheme
(raise "something went wrong")
(raise 404)
(raise (list 'error 'division-by-zero))
```

### `with-exception-handler`

**Syntax:** `(with-exception-handler handler thunk)`

Installs `handler` (a one-argument procedure) as the current exception handler for the
dynamic extent of `thunk`. If an exception is raised during `thunk`, `handler` is called
with the exception object.

Unlike `guard`, the handler does not automatically unwind the stack — it runs in the
dynamic context of the `raise` call. If the handler returns normally, the exception is
re-raised to the next handler.

```scheme
(with-exception-handler
  (lambda (exn) (display exn) (newline))
  (lambda () (raise "test")))
```

### Interaction with `dynamic-wind`

When an exception causes a non-local exit, `dynamic-wind` cleanup handlers (`after` thunks)
execute in the correct order during stack unwinding. This ensures resource cleanup (file
handles, locks, etc.) even in the presence of exceptions.

```scheme
(guard (exn (else (display "handled\n")))
  (dynamic-wind
    (lambda () (display "enter\n"))
    (lambda () (raise "oops"))
    (lambda () (display "cleanup\n"))))
;; Prints: enter, cleanup, handled
```

---

## Parallel Primitives

Eshkol provides compiler-level parallel primitives backed by a **work-stealing thread pool**
with hardware-aware sizing. The thread pool defaults to the number of hardware cores and
uses Chase-Lev work-stealing deques for load balancing.

**Work-stealing algorithm:** Each thread maintains a local double-ended queue (deque) of tasks.
When a thread exhausts its local work, it steals from a random other thread's deque tail.
This provides O(1) amortized per-task overhead with automatic load balancing across
heterogeneous workloads.

**Thread pool sizing:** Defaults to `std::thread::hardware_concurrency()` (number of logical
cores). Can be tuned via environment variable `ESHKOL_NUM_THREADS`.

**Implementation:** [system_codegen.cpp](lib/backend/system_codegen.cpp).

### `parallel-map`

**Syntax:** `(parallel-map proc list)` → List

Like `map`, but distributes work items across the thread pool. Each element is processed
independently — `proc` must be thread-safe (no shared mutable state without synchronization).

Order is preserved: the output list has the same element ordering as the input.

```scheme
(parallel-map (lambda (x) (* x x)) '(1 2 3 4 5))  ; => (1 4 9 16 25)
```

### `parallel-fold`

**Syntax:** `(parallel-fold proc init list)` → Value

Parallel reduction. The list is partitioned across threads; each thread reduces its partition,
then partial results are combined.

**Correctness requirement:** `proc` must be **associative** — i.e., `(proc (proc a b) c)` =
`(proc a (proc b c))`. Commutativity is not required; the fold tree structure is deterministic
for a given thread count. Non-associative operations produce undefined results.

```scheme
(parallel-fold + 0 '(1 2 3 4 5))  ; => 15
(parallel-fold * 1 '(1 2 3 4 5))  ; => 120
```

### `parallel-filter`

**Syntax:** `(parallel-filter pred list)` → List

Applies the predicate `pred` in parallel, then collects elements for which it returned `#t`.
Order is preserved.

```scheme
(parallel-filter even? '(1 2 3 4 5 6))  ; => (2 4 6)
```

### `parallel-for-each`

**Syntax:** `(parallel-for-each proc list)` → void

Applies `proc` to each element in parallel for side effects. **No ordering guarantee** on
when side effects occur — if `proc` writes to shared state, external synchronization is required.

### `parallel-execute`

**Syntax:** `(parallel-execute thunk1 thunk2 ...)` → void

Executes multiple zero-argument thunks concurrently. Returns when all thunks have completed.
Useful for independent I/O operations or initialization tasks.

```scheme
(parallel-execute
  (lambda () (display "task A\n"))
  (lambda () (display "task B\n"))
  (lambda () (display "task C\n")))
;; All three run concurrently; output order varies
```

### `future` / `force`

**Syntax:** `(future expr)` → Future, `(force f)` → Value

Asynchronous computation primitives. `future` launches the expression in a background thread
and returns immediately with a future handle. `force` blocks the current thread until the
future's computation completes, then returns the result.

If the future has already completed, `force` returns immediately (cached result).

```scheme
(define f (future (heavy-computation x)))
;; ... other work continues ...
(define result (force f))   ; blocks until ready, returns result
```

### Nested Parallelism

Nested `parallel-map` / `parallel-fold` calls are supported but may serialize if the thread
pool is saturated. The work-stealing scheduler handles this gracefully — inner parallel calls
enqueue tasks that existing threads can steal, but no additional threads are created beyond
the pool size.

```scheme
(define f (future (expensive-computation)))
;; ... do other work ...
(force f)  ; => result of expensive-computation
```

---

## Consciousness Engine

The Consciousness Engine provides 22 compiler-level builtins implementing three foundational
pillars of computational intelligence: **Logic Programming** (Robinson, 1965), **Active Inference**
(Friston, 2010), and **Global Workspace Theory** (Baars, 1988). These are not library functions —
they are first-class compiled primitives with dedicated type tags, heap subtypes, and LLVM IR
codegen paths, enabling deep integration with Eshkol's automatic differentiation, tensor operations,
and closure systems.

The engine spans six distinct type representations:

| Type | Tag / Subtype | Description |
|------|--------------|-------------|
| Logic Variable | Type tag `10` (`ESHKOL_VALUE_LOGIC_VAR`) | Unbound variable for unification |
| Substitution | Heap subtype `12` (`HEAP_SUBTYPE_SUBSTITUTION`) | Triangular variable binding map |
| Fact | Heap subtype `13` (`HEAP_SUBTYPE_FACT`) | Structured predicate term |
| Knowledge Base | Heap subtype `15` (`HEAP_SUBTYPE_KNOWLEDGE_BASE`) | Indexed fact collection |
| Factor Graph | Heap subtype `16` (`HEAP_SUBTYPE_FACTOR_GRAPH`) | Bipartite probabilistic graphical model |
| Workspace | Heap subtype `17` (`HEAP_SUBTYPE_WORKSPACE`) | Competitive module broadcast arena |

**Implementation:** `LogicWorkspaceCodegen` in
[lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp)
holds the codegen methods (`codegenMakeFactorGraph`, `codegenFGAddFactor`,
`codegenFGInfer`, `codegenFreeEnergy`, `codegenEFE`, `codegenFGUpdateCPT`,
`codegenFGObserve`, `codegenMakeWorkspace`, `codegenWSRegister`,
`codegenWSStep`, plus the logic-engine entry points); the dispatch
table lives in `LLVMCodeGenerator::codegenSpecialOperation` in
[lib/backend/llvm_codegen.cpp](../lib/backend/llvm_codegen.cpp).
Runtime: [logic.h](inc/eshkol/backend/logic.h) / [logic.cpp](lib/backend/logic.cpp),
[inference.h](inc/eshkol/backend/inference.h) / [inference.cpp](lib/backend/inference.cpp),
[workspace.h](inc/eshkol/backend/workspace.h) / [workspace.cpp](lib/backend/workspace.cpp).

### Logic Programming

Logic programming in Eshkol is grounded in first-order unification theory. The central operation
is finding a substitution σ such that σ(t₁) = σ(t₂) for arbitrary terms t₁ and t₂. This enables
pattern matching, constraint solving, and symbolic reasoning as primitive operations within a
compiled, statically-typed language.

#### Logic Variables

**Syntax:** `?name`

Logic variables are denoted by the `?` prefix, which is a valid R7RS identifier start character.
They are first-class values with type tag 10 (`ESHKOL_VALUE_LOGIC_VAR`), meaning they can be
stored in data structures, passed to functions, and returned from functions like any other value.

Each logic variable has a unique identity — two occurrences of `?x` in the same lexical scope
refer to the same variable, enabling relational constraints.

```scheme
?x            ; a logic variable
?position     ; another logic variable
(logic-var? ?x)     ; => #t
(logic-var? 42)     ; => #f
```

#### `make-substitution`

**Syntax:** `(make-substitution)` → Substitution

Creates an empty substitution σ = ∅. A substitution is a finite map from logic variables to
terms, represented as a triangular substitution (Baader & Snyder, 2001) — a list of bindings
where each variable maps to a term that may reference other variables, but never itself
(guaranteed by the occurs check).

**Returns:** An empty `Substitution` (heap subtype 12).

**Implementation:** `LogicWorkspaceCodegen::codegenMakeSubst` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_make_substitution_tagged`.

#### `unify`

**Syntax:** `(unify t₁ t₂ σ)` → Substitution | #f

Attempts to extend substitution σ to make terms t₁ and t₂ identical. Returns the extended
substitution on success, or `#f` if unification fails.

**Algorithm** (Martelli-Montanari with occurs check):

1. **Walk** both t₁ and t₂ through σ (resolve existing bindings).
2. If both resolve to identical atoms, return σ unchanged.
3. If t₁ resolves to a logic variable, perform the **occurs check** (t₁ must not appear
   in the resolved t₂). If clear, extend σ with the binding t₁ → t₂.
4. If t₂ resolves to a logic variable, symmetric to case 3.
5. If both resolve to pairs (cons cells), recursively unify the `car` parts, then unify
   the `cdr` parts with the intermediate substitution.
6. Otherwise, unification fails — return `#f`.

**Occurs check:** Prevents circular substitutions. `(unify ?x (list ?x) σ)` fails because
binding ?x → (list ?x) would create an infinite term. This is essential for soundness.

**Complexity:** O(n · α(n)) amortized with path compression, where n is the number of
variables in the terms and α is the inverse Ackermann function. In practice, near-linear.

**Implementation:** `LogicWorkspaceCodegen::codegenUnify` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_unify_tagged`.

```scheme
(define s (make-substitution))
(define s1 (unify ?x 42 s))           ; bind ?x → 42
(define s2 (unify ?x ?y s))           ; bind ?x → ?y
(define s3 (unify ?x (list ?x) s))    ; => #f (occurs check)
(unify ?a ?b (unify ?b 7 s))          ; ?a → ?b → 7 (transitive)
```

#### `walk`

**Syntax:** `(walk term σ)` → Value

Dereferences a term through a substitution chain. If `term` is a logic variable bound in σ,
follows the chain of bindings until reaching either an unbound variable or a non-variable value.

**Invariant:** Substitutions are triangular by construction (the occurs check prevents cycles).
Therefore, `walk` always terminates.

**Implementation:** `LogicWorkspaceCodegen::codegenWalk` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_walk_tagged`.

```scheme
(define s (make-substitution))
(define s1 (unify ?x ?y s))
(define s2 (unify ?y 7 s1))
(walk ?x s2)    ; => 7 (chain: ?x → ?y → 7)
(walk ?z s2)    ; => ?z (unbound, returned as-is)
(walk 42 s2)    ; => 42 (non-variable, returned as-is)
```

### Knowledge Base

The knowledge base provides structured storage and retrieval of logical facts. Facts are
predicate-argument structures that can contain both ground terms and logic variables. The
KB supports assertion (adding facts) and pattern-based query (retrieving matching facts
via unification).

#### `make-fact`

**Syntax:** `(make-fact predicate arg ...)` → Fact

Creates a structured fact with a predicate symbol and zero or more arguments. Facts are
first-class values (heap subtype 13) that can be stored in knowledge bases or manipulated
directly.

**Implementation:** `LogicWorkspaceCodegen::codegenMakeFact` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_make_fact_tagged`.

```scheme
(make-fact 'parent 'alice 'bob)      ; parent(alice, bob)
(make-fact 'edge 1 2 5.0)            ; edge(1, 2, 5.0) — weighted graph edge
(make-fact 'type ?x 'integer)        ; type(?x, integer) — with logic variable
```

#### `make-kb`

**Syntax:** `(make-kb)` → KnowledgeBase

Creates an empty knowledge base. A KB is a mutable collection of facts that supports
assertion and unification-based query.

**Returns:** An empty `KnowledgeBase` (heap subtype 15).

**Implementation:** `LogicWorkspaceCodegen::codegenMakeKB` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_make_kb_tagged`.

#### `kb-assert!`

**Syntax:** `(kb-assert! kb fact)` → void

Adds a fact to the knowledge base. This is a mutating operation — the KB is modified
in-place. Facts are stored in insertion order and all are considered during queries.

**Implementation:** `LogicWorkspaceCodegen::codegenKBAssert` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_kb_assert_tagged`.

#### `kb-query`

**Syntax:** `(kb-query kb pattern)` → List of Substitutions

Queries the knowledge base by attempting to unify `pattern` against every fact in the KB.
Returns a list of substitutions — one for each fact that successfully unifies with the pattern.
An empty list indicates no matches.

**Algorithm:** For each fact f in kb, attempt `(unify pattern f (make-substitution))`.
Collect all successful substitutions into a list.

**Implementation:** `LogicWorkspaceCodegen::codegenKBQuery` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_kb_query_tagged`.

```scheme
;; Build a family knowledge base
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'alice 'bob))
(kb-assert! kb (make-fact 'parent 'bob 'charlie))
(kb-assert! kb (make-fact 'parent 'alice 'diana))

;; Query: who are bob's parents?
(define results (kb-query kb (make-fact 'parent ?who 'bob)))
;; results: list with one substitution where ?who = 'alice

;; Query: who are alice's children?
(define children (kb-query kb (make-fact 'parent 'alice ?child)))
;; children: list of substitutions where ?child = 'bob and ?child = 'diana

;; Query with no matches
(define empty (kb-query kb (make-fact 'sibling ?a ?b)))
;; empty: '()
```

#### Type Predicates

| Predicate | Tests For | Type |
|-----------|-----------|------|
| `(logic-var? x)` | Logic variable | Tag 10 |
| `(substitution? x)` | Substitution | Heap subtype 12 |
| `(fact? x)` | Fact | Heap subtype 13 |
| `(kb? x)` | Knowledge base | Heap subtype 15 |

All return `#t` or `#f`. Implementation lives in the predicate helpers
`LLVMCodeGenerator::codegenLogicVarPred`, `codegenSubstPred`,
`codegenKBPred`, `codegenFactPred`, `codegenFactorGraphPred`, and
`codegenWorkspacePred` in [lib/backend/llvm_codegen.cpp](../lib/backend/llvm_codegen.cpp).

### Active Inference

Active inference (Friston, 2010) models perception and action as variational inference on a
generative model. Eshkol implements this via **factor graphs** — bipartite probabilistic
graphical models where variable nodes represent random variables with discrete state spaces
and factor nodes encode conditional probability tables (CPTs) as log-probability tensors.

The framework provides belief propagation for inference, free energy computation for
model evaluation, and expected free energy for action selection.

#### Factor Graph Structure

A factor graph is a bipartite graph G = (V, F, E) where:

- **V** = {v₁, ..., v_n} is the set of **variable nodes**. Each v_i has a discrete state
  space with |S_i| possible values.
- **F** = {f₁, ..., f_m} is the set of **factor nodes**. Each factor f_j encodes a local
  function over its neighboring variables — specifically, a conditional probability table
  stored as a flat tensor of log-probabilities.
- **E** connects each factor to the variables in its scope (neighborhood).

#### `make-factor-graph`

**Syntax:** `(make-factor-graph n-vars dims-tensor)` → FactorGraph

Creates a factor graph with `n-vars` variable nodes. The second argument `dims-tensor` is a
tensor specifying the number of discrete states for each variable: `dims-tensor[i]` = |S_i|.

**Note:** The second argument is a **tensor of per-variable state counts**, not a scalar.
This allows heterogeneous state spaces (e.g., one binary variable and one ternary variable).

**Returns:** A `FactorGraph` (heap subtype 16) with no factors attached.

**Implementation:** `LogicWorkspaceCodegen::codegenMakeFactorGraph` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_make_factor_graph_tagged`.

```scheme
;; Two binary variables (each with 2 states)
(define fg (make-factor-graph 2 #(2 2)))

;; Three variables: binary, ternary, binary
(define fg2 (make-factor-graph 3 #(2 3 2)))
```

#### `fg-add-factor!`

**Syntax:** `(fg-add-factor! fg var-indices cpt)` → void

Adds a factor node to the graph. `var-indices` is a tensor of variable indices specifying
which variables are in this factor's scope. `cpt` is a flat tensor of **log-probabilities**
indexed by the joint state assignment of the scoped variables.

**CPT Layout:** For a factor over variables v_a and v_b with |S_a| and |S_b| states respectively,
the CPT has |S_a| × |S_b| entries. Entry at index `i * |S_b| + j` corresponds to
log P(v_a = i, v_b = j) (or the conditional log P(v_b = j | v_a = i), depending on the
model structure).

**Implementation:** `LogicWorkspaceCodegen::codegenFGAddFactor` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_fg_add_factor_tagged`.

```scheme
;; Unary factor: prior over variable 0 (binary)
;; P(sunny) = 0.7, P(rainy) = 0.3
(fg-add-factor! fg #(0) #(-0.3567 -1.2040))   ; log(0.7), log(0.3)

;; Binary factor: conditional P(umbrella | weather)
;; P(umbrella=yes | sunny) = 0.9,  P(umbrella=no | sunny) = 0.1
;; P(umbrella=yes | rainy) = 0.2,  P(umbrella=no | rainy) = 0.8
(fg-add-factor! fg #(0 1)
  #(-0.1054 -2.3026 -1.6094 -0.2231))
;; Layout: [log P(s=0,u=0), log P(s=0,u=1), log P(s=1,u=0), log P(s=1,u=1)]
```

#### `fg-infer!`

**Syntax:** `(fg-infer! fg max-iterations)` → Tensor (beliefs)

Runs **loopy belief propagation** (sum-product algorithm) for up to `max-iterations` rounds.
Returns a tensor of marginal beliefs for all variables.

**Algorithm** — Sum-product message passing in log-space:

**Factor→variable messages:**
```
μ_{f→v}(x_v) = Σ_{x_{ne(f)\v}} [ f(x_{ne(f)}) · ∏_{v'∈ne(f)\v} ν_{v'→f}(x_{v'}) ]
```

**Variable→factor messages:**
```
ν_{v→f}(x_v) = ∏_{f'∈ne(v)\f} μ_{f'→v}(x_v)
```

**Beliefs (marginals):**
```
b(x_v) = (1/Z) · ∏_{f∈ne(v)} μ_{f→v}(x_v)
```

where Z is the normalization constant ensuring Σ_x b(x) = 1.

**Convergence:** The algorithm terminates when either (a) message deltas fall below an
internal threshold ε, or (b) `max-iterations` is reached. For tree-structured factor graphs,
convergence is guaranteed in one pass. For loopy graphs, convergence is empirical but
generally reliable for well-conditioned problems.

**Returns:** A flat tensor of beliefs concatenated across variables: beliefs for variable 0
(|S_0| entries), then variable 1 (|S_1| entries), etc.

**Implementation:** `LogicWorkspaceCodegen::codegenFGInfer` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_fg_infer_tagged`.

```scheme
(define beliefs (fg-infer! fg 20))    ; run 20 iterations of BP
;; beliefs: #(0.7 0.3 0.65 0.35)
;; Variable 0: P(sunny)=0.7, P(rainy)=0.3
;; Variable 1: P(umbrella=yes)=0.65, P(umbrella=no)=0.35
```

#### `fg-update-cpt!`

**Syntax:** `(fg-update-cpt! fg factor-index new-cpt)` → void

Replaces the conditional probability table of the factor at `factor-index` with `new-cpt`
and **resets all messages** in the graph. Subsequent calls to `fg-infer!` will reconverge
beliefs under the updated parameters.

This is the key primitive enabling **learning**: by iteratively updating CPTs based on
observed data (e.g., via gradient descent on the free energy), the model adapts its
generative structure.

**Implementation:** `LogicWorkspaceCodegen::codegenFGUpdateCPT` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_fg_update_cpt_tagged`.

```scheme
;; Update the prior: now P(sunny) = 0.2, P(rainy) = 0.8
(fg-update-cpt! fg 0 #(-1.6094 -0.2231))   ; log(0.2), log(0.8)
(fg-infer! fg 20)   ; beliefs reconverge under new prior
```

#### `free-energy`

**Syntax:** `(free-energy fg observations)` → Number

Computes the **variational free energy** F, which measures the divergence between the
model's beliefs and the observed data:

```
F = E_q[log q(s)] - E_q[log p(o, s)]
  = -Σ_i Σ_s q(s_i) · log p(o | s_i) + Σ_i Σ_s q(s_i) · log q(s_i)
```

where q(s_i) = b(x_i) are the current beliefs (from the most recent `fg-infer!` call), and
p(o, s) is the joint probability of observations and states under the generative model.

**Observations format:** A tensor of `#(var_index observed_state)` pairs — NOT full state
vectors. Each pair specifies one observed variable and its observed state.

**Interpretation:** Lower free energy indicates better model fit. Minimizing F with respect
to model parameters (CPTs) is equivalent to maximizing the evidence lower bound (ELBO).

**Implementation:** `LogicWorkspaceCodegen::codegenFreeEnergy` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_free_energy_tagged`.

```scheme
;; Observe: variable 1 (umbrella) is in state 0 (yes)
(define F (free-energy fg #(1.0 0.0)))
;; F: scalar free energy value

;; Multiple observations
(define F2 (free-energy fg #(0.0 1.0 1.0 0.0)))
;; Observe var 0 in state 1, var 1 in state 0
```

#### `expected-free-energy`

**Syntax:** `(expected-free-energy fg action-var action-state)` → Number

Computes the **expected free energy** (EFE) G for a prospective action, which quantifies the
anticipated information gain and pragmatic value of taking that action:

```
G = E_q[log q(s') - log p(o', s')]
```

This decomposes into two components:
- **Epistemic value** (information gain): actions that reduce uncertainty about hidden states
- **Pragmatic value** (goal satisfaction): actions that bring observations closer to preferences

The policy minimizing G is preferred — this is the core decision-making principle in active
inference. The agent selects actions that simultaneously maximize information gain and
goal achievement.

**Parameters:**
- `fg` — the factor graph (with current beliefs from `fg-infer!`)
- `action-var` — the variable index representing the action
- `action-state` — the specific action state to evaluate

**Returns:** Scalar expected free energy for the given action.

**Implementation:** `LogicWorkspaceCodegen::codegenEFE` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_efe_tagged`.

```scheme
;; Evaluate expected free energy for each possible action
(define g0 (expected-free-energy fg 0 0))   ; EFE of action-var 0, state 0
(define g1 (expected-free-energy fg 0 1))   ; EFE of action-var 0, state 1
;; Select action minimizing G
(if (< g0 g1) 'action-0 'action-1)
```

#### `factor-graph?`

**Syntax:** `(factor-graph? x)` → Boolean

Returns `#t` if `x` is a factor graph (heap subtype 16), `#f` otherwise.

**Implementation:** `LLVMCodeGenerator::codegenFactorGraphPred` in [lib/backend/llvm_codegen.cpp](../lib/backend/llvm_codegen.cpp).

#### Complete Active Inference Example

```scheme
;; Weather→Umbrella Bayesian network as a factor graph
;;
;;   Weather (binary: sunny/rainy)
;;      |
;;   P(umbrella | weather)
;;      |
;;   Umbrella (binary: yes/no)

;; 1. Create the graph
(define fg (make-factor-graph 2 #(2 2)))

;; 2. Add prior: P(sunny) = 0.7
(fg-add-factor! fg #(0) #(-0.3567 -1.2040))

;; 3. Add conditional: P(umbrella | weather)
(fg-add-factor! fg #(0 1)
  #(-0.1054 -2.3026 -1.6094 -0.2231))

;; 4. Infer beliefs
(define beliefs (fg-infer! fg 20))

;; 5. Observe someone carrying an umbrella → compute free energy
(define F (free-energy fg #(1.0 0.0)))

;; 6. Evaluate actions
(define g-sunny (expected-free-energy fg 0 0))
(define g-rainy (expected-free-energy fg 0 1))

;; 7. Learn: update prior based on new evidence
(fg-update-cpt! fg 0 #(-1.6094 -0.2231))   ; P(sunny) = 0.2 now
(define new-beliefs (fg-infer! fg 20))       ; reconverge
```

### Global Workspace

The Global Workspace subsystem implements a computational model of Baars' Global Workspace
Theory (GWT, 1988). In GWT, consciousness is modeled as a "spotlight" mechanism where
specialized processing modules (perception, memory, planning, etc.) compete for access to a
shared broadcast medium. The winning module's content becomes globally available to all
other modules, enabling integrated information processing.

This architecture is particularly relevant for multi-modal AI systems where heterogeneous
processing streams must be integrated into coherent behavior.

#### Competition Mechanism

Each workspace step executes a competition cycle:

1. **Invoke:** Each registered module receives the current broadcast content tensor as input.
2. **Score:** Each module returns a pair: `(cons salience proposal-tensor)`, where the
   `salience` (a scalar) represents the module's activation strength and `proposal-tensor`
   is the content it wishes to broadcast.
3. **Select:** Softmax selection over activation strengths determines the winner:
   σ(z_i) = exp(z_i) / Σ_j exp(z_j)
4. **Broadcast:** The winning module's proposal tensor becomes the new global content,
   available to all modules on the next step.

This soft competition ensures that the most "salient" module dominates while maintaining
differentiability of the selection process.

#### `make-workspace`

**Syntax:** `(make-workspace dim max-modules)` → Workspace

Creates a global workspace with content tensors of dimensionality `dim` and capacity for
up to `max-modules` registered modules.

**Parameters:**
- `dim` — dimensionality of the content tensor (integer)
- `max-modules` — maximum number of module slots (integer)

**Returns:** A `Workspace` (heap subtype 17) initialized with a zero content tensor.

**Implementation:** `LogicWorkspaceCodegen::codegenMakeWorkspace` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_make_workspace_tagged`.

```scheme
(define ws (make-workspace 4 3))   ; 4-dimensional content, 3 module slots
```

#### `ws-register!`

**Syntax:** `(ws-register! ws name process-fn)` → void

Registers a closure as a named module in the workspace. The closure `process-fn` must
accept one argument (a content tensor of the workspace's dimensionality) and return a
pair `(cons salience proposal-tensor)` where:

- `salience` is a scalar number representing the module's activation strength
- `proposal-tensor` is the content the module proposes to broadcast

**Parameters:**
- `ws` — the workspace
- `name` — a string identifier for the module (for debugging/inspection)
- `process-fn` — a closure with signature `(tensor → (cons number tensor))`

**Implementation:** `LogicWorkspaceCodegen::codegenWSRegister` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp),
runtime `eshkol_ws_register_tagged`.

```scheme
(ws-register! ws "perception"
  (lambda (content)
    (cons 0.9 #(1.0 0.0 0.0 0.0))))    ; high salience, sensory input

(ws-register! ws "memory"
  (lambda (content)
    (cons 0.3 #(0.0 1.0 0.0 0.0))))    ; low salience, recall

(ws-register! ws "planning"
  (lambda (content)
    (cons 0.6 #(0.0 0.0 1.0 0.0))))    ; medium salience, goals
```

#### `ws-step!`

**Syntax:** `(ws-step! ws)` → Workspace

Executes one complete competition cycle:

1. Calls each registered module's closure with the current content tensor.
2. Extracts activation strengths from each module's returned pair.
3. Applies softmax normalization: σ(z_i) = exp(z_i) / Σ_j exp(z_j).
4. Selects the module with the highest softmax probability as the winner.
5. Broadcasts the winner's proposal tensor as the new workspace content.
6. Returns the mutated workspace.

**Note:** This function takes exactly **one argument** (the workspace). There is no separate
input parameter — modules receive the current broadcast content, which serves as the input.

**Implementation:** `LogicWorkspaceCodegen::codegenWSStep` in [lib/backend/logic_workspace_codegen.cpp](../lib/backend/logic_workspace_codegen.cpp).
The codegen generates a loop calling `codegenClosureCall` for each registered module.
C runtime helpers `eshkol_ws_make_content_tensor` and `eshkol_ws_step_finalize` handle
tensor wrapping and softmax broadcast respectively.

```scheme
(ws-step! ws)
;; softmax([0.9, 0.3, 0.6]) → [0.422, 0.232, 0.313]
;; "perception" wins with highest probability
;; Content tensor becomes #(1.0 0.0 0.0 0.0)

(ws-step! ws)
;; Next step: all modules now receive #(1.0 0.0 0.0 0.0) as input
;; Modules may change their salience based on the new content
```

#### `workspace?`

**Syntax:** `(workspace? x)` → Boolean

Returns `#t` if `x` is a workspace (heap subtype 17), `#f` otherwise.

**Implementation:** `LLVMCodeGenerator::codegenWorkspacePred` in [lib/backend/llvm_codegen.cpp](../lib/backend/llvm_codegen.cpp).

#### Complete Global Workspace Example

```scheme
;; Multi-modal integration: perception + memory + planning
(define ws (make-workspace 4 3))

;; Perception module: high salience when content is "neutral"
(ws-register! ws "perception"
  (lambda (content)
    (let ((novelty (- 1.0 (tensor-dot content content))))
      (cons (* 0.9 (max 0.1 novelty))      ; salience drops as content fills
            #(0.8 0.1 0.0 0.1)))))          ; sensory representation

;; Memory module: salience increases when perception matches stored pattern
(ws-register! ws "memory"
  (lambda (content)
    (let ((match (tensor-dot content #(0.8 0.1 0.0 0.1))))
      (cons (* 0.7 match)                   ; salience proportional to match
            #(0.7 0.2 0.1 0.0)))))          ; retrieved memory

;; Planning module: constant moderate salience
(ws-register! ws "planning"
  (lambda (content)
    (cons 0.5 #(0.0 0.0 0.8 0.2))))        ; action plan

;; Run 10 competition cycles
(do ((i 0 (+ i 1))) ((= i 10))
  (ws-step! ws))
;; Over iterations, modules dynamically compete based on content state
```

#### Builtins Summary

| Category | Builtins | Count |
|----------|----------|-------|
| Unification | `unify`, `walk`, `make-substitution` | 3 |
| Facts & KB | `make-fact`, `make-kb`, `kb-assert!`, `kb-query` | 4 |
| Active Inference | `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`, `free-energy`, `expected-free-energy` | 6 |
| Global Workspace | `make-workspace`, `ws-register!`, `ws-step!` | 3 |
| Type Predicates | `logic-var?`, `substitution?`, `fact?`, `kb?`, `factor-graph?`, `workspace?` | 6 |
| **Total** | | **22** |

---

## Machine Learning & Neural Networks

Eshkol provides a comprehensive suite of compiler-level machine learning primitives spanning
activation functions, loss functions, optimizers, weight initializers, learning rate schedulers,
convolutional operations, and transformer architectures. All operations are first-class builtins
dispatched through the LLVM codegen pipeline — not library wrappers — enabling SIMD acceleration
for large tensors, automatic GPU dispatch above cost model thresholds, and seamless integration
with Eshkol's automatic differentiation system via `gradient`.

**Architecture:** Operations are dispatched in `LLVMCodeGenerator::codegenCall` in [lib/backend/llvm_codegen.cpp](../lib/backend/llvm_codegen.cpp)
and implemented in [tensor_codegen.cpp](lib/backend/tensor_codegen.cpp). Backward passes are
available for AD integration (e.g., `tensorReluBackward`, `tensorSigmoidBackward`).

### Activation Functions

Activation functions introduce non-linearity into neural networks. Without them, any composition
of linear layers collapses to a single affine transformation y = Wx + b, regardless of depth.
Each activation function has distinct gradient properties that affect training dynamics — in
particular, susceptibility to vanishing or exploding gradients, and the presence or absence of
"dead" regions where gradients are zero.

All activation functions operate element-wise on tensors and return a new tensor of the same shape.

#### `relu`

**Syntax:** `(relu t)` → Tensor

Rectified Linear Unit: f(x) = max(0, x).

**Gradient:** f'(x) = 1 if x > 0, 0 otherwise. Sparse gradients promote efficient computation
but risk "dead neurons" — units that output zero for all inputs and receive zero gradient,
permanently ceasing to learn.

**Use case:** Default activation for hidden layers in most architectures.

**Implementation:** `tensorRelu` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).
Backward: `tensorReluBackward` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `sigmoid`

**Syntax:** `(sigmoid t)` → Tensor

Logistic sigmoid: σ(x) = 1 / (1 + e^(-x)).

**Gradient:** σ'(x) = σ(x)(1 - σ(x)). Output range (0, 1). Saturates at extremes (x ≪ 0 or
x ≫ 0), causing vanishing gradients in deep networks. Outputs are not zero-centered.

**Use case:** Binary classification output layers, gating mechanisms (LSTM, GRU).

**Implementation:** `tensorSigmoid` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).
Backward: [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

#### `softmax`

**Syntax:** `(softmax t)` → Tensor

Normalized exponential: softmax(x)_i = exp(x_i) / Σ_j exp(x_j).

**Gradient:** The Jacobian is J_ij = p_i(δ_ij - p_j) where p = softmax(x). Unlike element-wise
activations, softmax couples all elements — each output depends on all inputs.

Numerically stabilized: internally computes exp(x_i - max(x)) to prevent overflow.

**Use case:** Multi-class classification output (produces a probability distribution that sums to 1),
attention weight normalization in transformers.

**Implementation:** `tensorSoftmax` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).
Backward: [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

#### `gelu`

**Syntax:** `(gelu t)` → Tensor

Gaussian Error Linear Unit: GELU(x) = x · Φ(x) ≈ x · σ(1.702x), where Φ is the standard
Gaussian CDF.

**Gradient:** Smooth and non-monotonic. Unlike ReLU, provides non-zero gradients for negative
inputs, enabling information flow through negative activations.

**Use case:** Transformer architectures (BERT, GPT, ViT). Outperforms ReLU in language
modeling tasks due to smoother optimization landscape.

**Implementation:** `tensorGelu` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).
Backward: [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

#### `leaky-relu`

**Syntax:** `(leaky-relu t [α])` → Tensor

Leaky ReLU: f(x) = x if x > 0, αx otherwise. Default α = 0.01.

**Gradient:** f'(x) = 1 if x > 0, α otherwise. The small positive slope for negative inputs
eliminates dead neurons — every unit always receives non-zero gradient.

**Use case:** Drop-in replacement for ReLU when dead neurons are observed during training.

**Implementation:** `tensorLeakyRelu` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).
Backward: [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

#### `silu`

**Syntax:** `(silu t)` → Tensor

Sigmoid Linear Unit (Swish): SiLU(x) = x · σ(x).

**Gradient:** Self-gated — the sigmoid acts as a learnable gate on the linear signal. Smooth,
non-monotonic, and bounded below. Gradients flow more uniformly than ReLU.

**Use case:** EfficientNet, modern ConvNets. Often outperforms both ReLU and GELU.

**Implementation:** `tensorSilu` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).
Backward: [lib/backend/tensor_codegen.cpp](../lib/backend/tensor_codegen.cpp).

#### `elu`

**Syntax:** `(elu t [α])` → Tensor

Exponential Linear Unit: f(x) = x if x > 0, α(e^x - 1) otherwise. Default α = 1.0.

**Gradient:** f'(x) = 1 if x > 0, f(x) + α if x ≤ 0. Negative saturation pushes mean
activations closer to zero, reducing the bias shift effect and accelerating convergence.

**Use case:** When negative outputs are needed and mean activation near zero is desirable.

**Implementation:** `tensorElu` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `selu`

**Syntax:** `(selu t)` → Tensor

Scaled Exponential Linear Unit: SELU(x) = λ · ELU(x, α) where λ ≈ 1.0507 and α ≈ 1.6733.

These specific constants were derived analytically (Klambauer et al., 2017) to ensure the
**self-normalizing property**: if inputs have zero mean and unit variance, outputs maintain
zero mean and unit variance. This holds across arbitrarily deep networks, eliminating the
need for batch normalization.

**Prerequisite:** Works best with LeCun normal initialization (`lecun-normal!`).

**Use case:** Deep fully-connected networks without batch normalization.

**Implementation:** `tensorSelu` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `mish`

**Syntax:** `(mish t)` → Tensor

Mish: f(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + e^x)).

**Gradient:** Smooth, non-monotonic, and self-regularizing. The function is C∞ (infinitely
differentiable), unlike ReLU (non-differentiable at 0) or Leaky ReLU (non-smooth at 0).

**Use case:** Object detection (YOLOv4 adopted Mish as default activation).

**Implementation:** `tensorMish` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `hard-swish`

**Syntax:** `(hard-swish t)` → Tensor

Piecewise linear approximation of Swish: f(x) = x · min(max(x + 3, 0), 6) / 6.

**Gradient:** Piecewise linear, computationally cheaper than SiLU/Swish. Negligible
accuracy difference in practice.

**Use case:** Mobile and edge deployment (MobileNetV3). Optimized for inference speed
on resource-constrained devices.

**Implementation:** `tensorHardSwish` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `hard-sigmoid`

**Syntax:** `(hard-sigmoid t)` → Tensor

Piecewise linear approximation of sigmoid: f(x) = min(max(x + 3, 0), 6) / 6.

**Gradient:** Simple, efficient. Three constant regions (0, linear, 1).

**Use case:** Mobile architectures where sigmoid's exponential is expensive. Gating in
efficient architectures.

**Implementation:** `tensorHardSigmoid` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `softplus`

**Syntax:** `(softplus t)` → Tensor

Smooth approximation of ReLU: f(x) = ln(1 + e^x).

**Gradient:** f'(x) = σ(x) (the sigmoid function). Always positive, never zero. Numerically
stabilized for large x: softplus(x) ≈ x when x ≫ 0.

**Use case:** Parameterizing positive-only quantities (e.g., variance in variational inference,
concentration parameters in Bayesian models).

**Implementation:** `tensorSoftplus` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `celu`

**Syntax:** `(celu t [α])` → Tensor

Continuously Differentiable ELU: f(x) = max(0, x) + min(0, α(e^(x/α) - 1)). Default α = 1.0.

**Gradient:** Continuous and everywhere differentiable (unlike ELU which has a derivative
discontinuity at x = 0). This smoothness property improves optimization convergence.

**Use case:** Research architectures requiring smooth activation functions.

**Implementation:** `tensorCelu` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### `dropout`

**Syntax:** `(dropout t rate training)` → Tensor

Regularization technique: during training, randomly zeros elements with probability `rate` and
scales surviving elements by 1/(1 - rate) to maintain expected value. During inference
(training = `#f`), returns the input unchanged.

**Parameters:**
- `t` — input tensor
- `rate` — dropout probability ∈ [0, 1)
- `training` — boolean: `#t` for training mode, `#f` for inference

**Theory:** Dropout (Srivastava et al., 2014) can be interpreted as approximate Bayesian
inference over an ensemble of 2^n sub-networks, where n is the number of units. It prevents
co-adaptation of feature detectors.

**Implementation:** `tensorDropout` in [lib/backend/tensor_activation_codegen.cpp](../lib/backend/tensor_activation_codegen.cpp).

#### Activation Function Examples

```scheme
(define x #(-2.0 -1.0 0.0 1.0 2.0))
(relu x)               ; => #(0.0 0.0 0.0 1.0 2.0)
(sigmoid x)            ; => #(0.119 0.269 0.5 0.731 0.881)
(softmax x)            ; => #(0.012 0.032 0.087 0.237 0.644)
(gelu x)               ; => #(-0.045 -0.159 0.0 0.841 1.955)
(leaky-relu x 0.1)     ; => #(-0.2 -0.1 0.0 1.0 2.0)
(silu x)               ; => #(-0.238 -0.269 0.0 0.731 1.762)
(elu x)                ; => #(-0.865 -0.632 0.0 1.0 2.0)
(mish x)               ; => #(-0.252 -0.303 0.0 0.865 1.944)
(dropout x 0.5 #t)     ; => #(0.0 -2.0 0.0 2.0 0.0) (varies, scaled by 2)
(dropout x 0.5 #f)     ; => #(-2.0 -1.0 0.0 1.0 2.0) (unchanged in inference)
```

### Loss Functions

Loss functions quantify the discrepancy between model predictions and ground truth targets.
They provide the scalar objective L(θ) that gradient descent minimizes. The choice of loss
function encodes assumptions about the data distribution: MSE corresponds to Gaussian noise,
cross-entropy to categorical distributions, and MAE to Laplacian noise.

All loss functions accept tensor inputs and return a scalar (0-dimensional) value.

#### `mse-loss`

**Syntax:** `(mse-loss pred target)` → Number

Mean Squared Error: L = (1/n) Σ_i (ŷ_i - y_i)².

**Gradient:** ∂L/∂ŷ_i = (2/n)(ŷ_i - y_i). Linear gradient — penalizes large errors quadratically.

**Probabilistic interpretation:** Maximizing the log-likelihood under a Gaussian noise model
N(y | ŷ, σ²) with fixed σ.

**Use case:** Regression tasks with Gaussian-distributed residuals.

**Implementation:** `mseLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `mae-loss`

**Syntax:** `(mae-loss pred target)` → Number

Mean Absolute Error: L = (1/n) Σ_i |ŷ_i - y_i|.

**Gradient:** ∂L/∂ŷ_i = (1/n) · sign(ŷ_i - y_i). Constant magnitude — robust to outliers
compared to MSE, but non-differentiable at ŷ_i = y_i (subgradient used).

**Probabilistic interpretation:** Maximizing log-likelihood under Laplacian noise.

**Use case:** Regression with outliers; median regression.

**Implementation:** `maeLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `cross-entropy-loss`

**Syntax:** `(cross-entropy-loss logits targets)` → Number

Cross-entropy with integrated softmax: L = -Σ_i y_i · log(softmax(ŷ)_i).

Internally computes `log-softmax(logits)` using the numerically stable log-sum-exp trick,
then dots with the target distribution.

**Gradient:** ∂L/∂ŷ_i = softmax(ŷ)_i - y_i. Remarkably simple — the error signal is the
difference between predicted and target probabilities.

**Use case:** Multi-class classification. `targets` should be one-hot encoded.

**Implementation:** `crossEntropyLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `bce-loss`

**Syntax:** `(bce-loss pred target)` → Number

Binary Cross-Entropy: L = -(1/n) Σ_i [y_i · log(ŷ_i) + (1 - y_i) · log(1 - ŷ_i)].

Predictions must be in (0, 1) — typically the output of a sigmoid layer. Internally clamped
to [ε, 1-ε] for numerical stability.

**Use case:** Binary classification, multi-label classification.

**Implementation:** `bceLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `binary-cross-entropy-loss`

**Syntax:** `(binary-cross-entropy-loss pred target)` → Number

Alias for `bce-loss`. Accepts probability inputs (post-sigmoid).

**Implementation:** Dispatches to `bceLoss` at `LLVMCodeGenerator::codegenCall` in [lib/backend/llvm_codegen.cpp](../lib/backend/llvm_codegen.cpp).

#### `huber-loss`

**Syntax:** `(huber-loss pred target [δ])` → Number

Huber loss: L = 0.5a² if |a| ≤ δ, δ(|a| - 0.5δ) otherwise, where a = ŷ - y. Default δ = 1.0.

**Gradient:** Behaves like MSE for small errors (|a| ≤ δ) and like MAE for large errors
(|a| > δ). This interpolation provides robustness to outliers while maintaining smooth
gradients near the optimum.

**Use case:** Regression that needs robustness without the non-differentiability of MAE.

**Implementation:** `huberLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `kl-div-loss`

**Syntax:** `(kl-div-loss P Q)` → Number

Kullback-Leibler Divergence: D_KL(P ‖ Q) = Σ_i P_i · log(P_i / Q_i).

**Properties:** Non-negative (D_KL ≥ 0), zero iff P = Q. **Not symmetric:** D_KL(P‖Q) ≠ D_KL(Q‖P).

**Use case:** Distribution matching in variational autoencoders (VAEs), knowledge distillation.
In VAEs, minimizes D_KL(q(z|x) ‖ p(z)) where q is the encoder and p is the prior.

**Implementation:** `klDivLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `hinge-loss`

**Syntax:** `(hinge-loss pred target)` → Number

Hinge loss: L = (1/n) Σ_i max(0, 1 - y_i · f(x_i)).

Targets should be ±1 (not 0/1). The margin of 1 means correctly classified examples with
confidence ≥ 1 incur zero loss.

**Use case:** Support Vector Machine (SVM) classification.

**Implementation:** `hingeLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `smooth-l1-loss`

**Syntax:** `(smooth-l1-loss pred target [β])` → Number

Smooth L1 (Huber variant): parameterized by β (default 1.0). Equivalent to `huber-loss` with
δ = β, but follows the convention used in object detection literature.

**Use case:** Object detection bounding box regression (Faster R-CNN, SSD).

**Implementation:** `smoothL1Loss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `focal-loss`

**Syntax:** `(focal-loss pred target [γ])` → Number

Focal loss: L = -(1 - p_t)^γ · log(p_t), where p_t is the predicted probability for the true
class. Default γ = 2.0.

**Theory:** (Lin et al., 2017) The modulating factor (1 - p_t)^γ down-weights the contribution
of easy examples (high p_t), focusing the training signal on hard, misclassified examples. When
γ = 0, focal loss reduces to standard cross-entropy.

**Use case:** Severe class imbalance (RetinaNet for object detection, where background examples
vastly outnumber foreground objects).

**Implementation:** `focalLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `triplet-loss`

**Syntax:** `(triplet-loss anchor positive negative [margin])` → Number

Triplet loss: L = max(d(a, p) - d(a, n) + margin, 0), where d is Euclidean distance.
Default margin = 1.0.

**Theory:** Trains embeddings such that anchor-positive distance is smaller than anchor-negative
distance by at least `margin`. When satisfied, loss is zero (no gradient signal from easy triplets).

**Use case:** Metric learning, face recognition (FaceNet), retrieval systems.

**Implementation:** `tripletLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `contrastive-loss`

**Syntax:** `(contrastive-loss t1 t2 y [margin])` → Number

Contrastive loss: L = (1-y) · d² + y · max(margin - d, 0)², where d = ‖t1 - t2‖.
Default margin = 1.0. Label y = 0 for similar pairs, y = 1 for dissimilar pairs.

**Use case:** Siamese networks, pairwise similarity learning.

**Implementation:** `contrastiveLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `label-smoothing-loss`

**Syntax:** `(label-smoothing-loss logits targets n-classes [ε])` → Number

Cross-entropy with smoothed targets: y'_i = (1 - ε) · y_i + ε / n-classes. Default ε = 0.1.

**Theory:** Prevents the model from becoming overconfident by spreading a small probability
mass ε uniformly across all classes. Improves calibration and generalization (Szegedy et al., 2016).

**Use case:** Large classification tasks, knowledge distillation, model calibration.

**Implementation:** `labelSmoothingLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### `cosine-embedding-loss`

**Syntax:** `(cosine-embedding-loss t1 t2 y [margin])` → Number

Cosine embedding loss: L = 1 - cos(t1, t2) if y = 1, max(0, cos(t1, t2) - margin) if y = -1.
Default margin = 0.0.

**Use case:** Sentence similarity, image-text matching, embedding alignment tasks.

**Implementation:** `cosineEmbeddingLoss` in [lib/backend/tensor_loss_codegen.cpp](../lib/backend/tensor_loss_codegen.cpp).

#### Loss Function Examples

```scheme
(define pred #(2.0 1.0 0.1))
(define target #(1.0 0.0 0.0))            ; one-hot: class 0

(mse-loss pred target)                     ; => 0.67
(cross-entropy-loss pred target)           ; => 0.417 (after internal softmax)
(focal-loss pred target 2.0)               ; => 0.069 (downweights easy examples)
(huber-loss pred target 1.0)               ; => 0.335

;; Metric learning
(define anchor #(1.0 0.0 0.0))
(define positive #(0.9 0.1 0.0))
(define negative #(0.0 1.0 0.0))
(triplet-loss anchor positive negative 1.0) ; => max(d(a,p) - d(a,n) + 1, 0)
```

### Optimizers

Optimizers adjust model parameters θ to minimize a loss function L(θ) by following gradient
information. The simplest approach is stochastic gradient descent (SGD): θ ← θ - η∇L, where
η is the learning rate. Adaptive methods (Adam, RMSprop, AdaGrad) maintain per-parameter
learning rates based on gradient history, enabling faster convergence on ill-conditioned
loss landscapes.

All Eshkol optimizers modify parameter tensors **in-place** for memory efficiency.

#### `sgd-step`

**Syntax:** `(sgd-step params grads lr [momentum velocity])` → void

Stochastic Gradient Descent with optional Polyak momentum.

**Without momentum** (3 args): Vanilla SGD update θ ← θ - η · g.

**With momentum** (5 args):
```
v_t = μ · v_{t-1} + g_t
θ_t = θ_{t-1} - η · v_t
```
where μ is the momentum coefficient. The velocity tensor `velocity` is modified in-place
to accumulate gradient history.

**Parameters:**
- `params` — parameter tensor (modified in-place)
- `grads` — gradient tensor (same shape as params)
- `lr` — learning rate η (scalar)
- `momentum` — momentum coefficient μ (scalar, optional)
- `velocity` — velocity tensor (modified in-place, optional, same shape as params)

**Implementation:** `sgdStep` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `adam-step`

**Syntax:** `(adam-step params grads lr m v t [β₁ β₂ ε])` → void

Adam optimizer (Kingma & Ba, 2014). Maintains exponential moving averages of the gradient
(first moment m) and squared gradient (second moment v):

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t           (first moment estimate)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          (second moment estimate)
m̂_t = m_t / (1 - β₁^t)                         (bias correction)
v̂_t = v_t / (1 - β₂^t)                         (bias correction)
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

**Defaults:** β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

**Parameters:**
- `params` — parameter tensor (modified in-place)
- `grads` — gradient tensor
- `lr` — learning rate η
- `m` — first moment tensor (modified in-place, initialized to zeros)
- `v` — second moment tensor (modified in-place, initialized to zeros)
- `t` — step count (integer, for bias correction; starts at 1)

**Implementation:** `adamStep` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `adamw-step`

**Syntax:** `(adamw-step params grads lr m v t [β₁ β₂ ε wd])` → void

AdamW (Loshchilov & Hutter, 2019). Decouples weight decay from the adaptive learning rate:

```
θ_t = θ_{t-1} · (1 - η · λ) - η · m̂_t / (√v̂_t + ε)
```

The critical difference from L2 regularization in Adam: weight decay is applied directly to
the parameters, not through the gradient. This decoupling prevents the adaptive learning rate
from counteracting the regularization effect.

**Defaults:** β₁ = 0.9, β₂ = 0.999, ε = 1e-8, wd = 0.01.

**Implementation:** `adamwStep` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `rmsprop-step`

**Syntax:** `(rmsprop-step params grads lr v [α ε])` → void

RMSprop (Hinton, unpublished lecture notes). Divides the learning rate by a running average
of gradient magnitudes:

```
v_t = α · v_{t-1} + (1 - α) · g_t²
θ_t = θ_{t-1} - η · g_t / (√v_t + ε)
```

**Defaults:** α = 0.99, ε = 1e-8.

**Intuition:** Reduces the effective learning rate for parameters with large gradients (steep
directions) and increases it for parameters with small gradients (flat directions).

**Implementation:** `rmspropStep` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `adagrad-step`

**Syntax:** `(adagrad-step params grads lr accum [ε])` → void

Adagrad (Duchi et al., 2011). Accumulates squared gradients over all time steps:

```
G_t = G_{t-1} + g_t²
θ_t = θ_{t-1} - η · g_t / (√G_t + ε)
```

**Default:** ε = 1e-8.

**Properties:** The effective learning rate monotonically decreases over training. This makes
Adagrad well-suited for sparse features (common in NLP) but may cause premature convergence
for dense features. RMSprop was developed as a fix for this issue.

**Implementation:** `adagradStep` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `zero-grad!`

**Syntax:** `(zero-grad! grad-tensor)` → void

Zeros all elements in the gradient tensor in-place. Must be called before each forward pass
to prevent gradient accumulation across mini-batches (unless accumulation is intentional,
e.g., for gradient accumulation with large effective batch sizes).

**Implementation:** `zeroGrad` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `clip-grad-norm!`

**Syntax:** `(clip-grad-norm! grads max-norm)` → Number

If the L2 norm of the gradient tensor exceeds `max-norm`, scales the gradient by
`max-norm / ‖g‖` to constrain it. Returns the original (pre-clip) norm.

**Theory:** Gradient clipping (Pascanu et al., 2013) is essential for training recurrent
networks and transformers, where gradients can explode due to repeated matrix multiplication
in the computational graph. The returned norm can be monitored for training health diagnostics.

**Implementation:** `clipGradNorm` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `check-grad-health`

**Syntax:** `(check-grad-health tensor)` → Integer

Counts the number of NaN and Inf values in the tensor. A return value of 0 indicates a
healthy gradient tensor. Non-zero values indicate numerical instability — typically caused
by learning rates that are too high, division by zero in the loss function, or overflow in
deep computational graphs.

**Implementation:** `checkGradHealth` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### Optimizer Examples

```scheme
;; Simple SGD training step
(define params #(1.0 2.0 3.0))
(define grads #(0.1 -0.2 0.05))
(sgd-step params grads 0.01)           ; params modified in-place

;; Adam with bias correction
(define m (make-tensor (list 3) 0.0))   ; first moment (zeros)
(define v (make-tensor (list 3) 0.0))   ; second moment (zeros)
(adam-step params grads 0.001 m v 1)    ; step 1

;; Gradient safety
(zero-grad! grads)                      ; zero before next forward pass
(let ((norm (clip-grad-norm! grads 1.0)))
  (when (> norm 10.0)
    (display "Warning: gradient explosion detected\n")))
(when (> (check-grad-health grads) 0)
  (display "Error: NaN/Inf in gradients\n"))
```

### Weight Initializers

Weight initialization sets the scale of parameters to preserve signal variance through
forward and backward passes. Incorrect initialization causes vanishing gradients (outputs
shrink layer by layer) or exploding gradients (outputs grow unbounded). The key insight is
matching the initialization variance to the activation function's properties.

All initializers modify the tensor **in-place** by sampling from the specified distribution.
Parameters `fi` (fan-in) and `fo` (fan-out) refer to the number of input and output units
of the layer, respectively.

#### `xavier-uniform!`

**Syntax:** `(xavier-uniform! t fi fo)` → void

Glorot & Bengio (2010): U[-√(6/(fi+fo)), √(6/(fi+fo))].

**Variance:** 2/(fi+fo). Derived to preserve forward and backward signal variance
simultaneously under linear activations. Optimal for sigmoid and tanh networks.

**Implementation:** `xavierUniform` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `xavier-normal!`

**Syntax:** `(xavier-normal! t fi fo)` → void

Gaussian variant: N(0, √(2/(fi+fo))). Same variance as Xavier uniform.

**Implementation:** `xavierNormal` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `kaiming-uniform!`

**Syntax:** `(kaiming-uniform! t fi)` → void

He et al. (2015): U[-√(6/fi), √(6/fi)].

**Variance:** 2/fi. Derived for ReLU activations, which zero out half the signal (effectively
halving the variance). Compensates by doubling the initialization variance compared to Xavier.

**Implementation:** `kaimingUniform` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `kaiming-normal!`

**Syntax:** `(kaiming-normal! t fi)` → void

Gaussian variant: N(0, √(2/fi)). Same variance as Kaiming uniform.

**Implementation:** `kaimingNormal` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `lecun-normal!`

**Syntax:** `(lecun-normal! t fi)` → void

LeCun et al. (1998): N(0, √(1/fi)).

**Variance:** 1/fi. Designed for SELU activations (required for the self-normalizing property).
Also suitable for tanh-based networks.

**Implementation:** `lecunNormal` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### Initializer Selection Guide

| Activation | Recommended Initializer | Why |
|-----------|------------------------|-----|
| ReLU, Leaky ReLU, ELU | `kaiming-uniform!` or `kaiming-normal!` | Compensates for ReLU's half-zeroing |
| SELU | `lecun-normal!` | Required for self-normalizing property |
| Sigmoid, Tanh | `xavier-uniform!` or `xavier-normal!` | Preserves variance under symmetric activations |
| GELU, SiLU, Mish | `kaiming-normal!` | Empirically works well for ReLU-like activations |

### Learning Rate Schedulers

Learning rate scheduling decays η over training, enabling large initial steps (fast convergence
across the loss landscape) followed by small steps (fine-tuning near the optimum). All schedulers
return the adjusted learning rate as a scalar — they do not modify any state.

#### `cosine-annealing-lr`

**Syntax:** `(cosine-annealing-lr step total base-lr)` → Number

Cosine annealing: η_t = base-lr · 0.5 · (1 + cos(π · step / total)).

Smoothly decays from `base-lr` to approximately 0 over `total` steps following a cosine curve.
The gradual decay avoids the abrupt drops of step-based schedules.

**Use case:** Standard for transformer training (BERT, GPT).

**Implementation:** `cosineAnnealingLR` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `step-decay-lr`

**Syntax:** `(step-decay-lr step size γ base-lr)` → Number

Step decay: η_t = base-lr · γ^⌊step/size⌋.

Reduces the learning rate by factor γ every `size` steps. Common values: γ = 0.1, size = 30 epochs.

**Use case:** Classic ResNet training schedule.

**Implementation:** `stepDecayLR` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `linear-warmup-lr`

**Syntax:** `(linear-warmup-lr step warmup base-lr)` → Number

Linear warmup: η_t = base-lr · min(step / warmup, 1).

Ramps the learning rate linearly from 0 to `base-lr` over `warmup` steps. After warmup
completes, returns `base-lr` unchanged.

**Theory:** Large learning rates early in training (before the running statistics in Adam/batch-norm
stabilize) can cause divergence. Warmup allows the optimizer state to accumulate meaningful
statistics before taking large steps.

**Use case:** Combined with cosine annealing for transformer training.

**Implementation:** `linearWarmupLR` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### `exponential-decay-lr`

**Syntax:** `(exponential-decay-lr step γ base-lr)` → Number

Exponential decay: η_t = base-lr · γ^step.

Continuous exponential decay — more aggressive than step decay as it reduces every step.

**Implementation:** `exponentialDecayLR` in [lib/backend/tensor_training_codegen.cpp](../lib/backend/tensor_training_codegen.cpp).

#### Scheduler Composition Example

```scheme
;; Warmup for 1000 steps, then cosine decay for 9000 steps
(define (lr-schedule step)
  (if (< step 1000)
    (linear-warmup-lr step 1000 0.001)
    (cosine-annealing-lr (- step 1000) 9000 0.001)))

(lr-schedule 0)       ; => 0.0 (start of warmup)
(lr-schedule 500)     ; => 0.0005 (mid-warmup)
(lr-schedule 1000)    ; => 0.001 (full learning rate)
(lr-schedule 5500)    ; => 0.0005 (mid-cosine)
(lr-schedule 10000)   ; => ~0.0 (end of training)
```

### Convolutional Neural Networks

Convolution extracts local spatial features by sliding a learnable kernel over the input tensor.
The convolution theorem shows that this operation in the spatial domain corresponds to
pointwise multiplication in the frequency domain; however, for small kernels the direct
spatial computation (or the im2col transformation + GEMM) is more efficient.

#### `conv1d`

**Syntax:** `(conv1d input kernel stride padding)` → Tensor

1D convolution for sequential data. Slides a kernel along the length dimension.

**Shapes:**
- Input: (batch, length, in_channels)
- Kernel: (out_channels, in_channels, kernel_size)
- Output: (batch, out_length, out_channels) where out_length = ⌊(length + 2·padding - kernel_size) / stride⌋ + 1

**Use case:** Time-series processing, audio analysis, 1D signal convolution.

**Implementation:** `conv1d` in [lib/backend/tensor_conv_codegen.cpp](../lib/backend/tensor_conv_codegen.cpp).

#### `conv2d`

**Syntax:** `(conv2d input kernel stride padding)` → Tensor

2D convolution via im2col + GEMM. The im2col transformation rearranges input patches into
columns of a matrix, converting the convolution to a single matrix multiplication — enabling
use of highly optimized BLAS routines (cBLAS/Apple Accelerate).

**Shapes:**
- Input: (batch, H, W, in_channels)
- Kernel: (out_channels, in_channels, kH, kW)
- Output: (batch, out_H, out_W, out_channels)
  where out_H = ⌊(H + 2·padding - kH) / stride⌋ + 1 (analogous for W)

**Implementation:** `conv2d` in [lib/backend/tensor_conv_codegen.cpp](../lib/backend/tensor_conv_codegen.cpp).

#### `conv3d`

**Syntax:** `(conv3d input kernel stride padding)` → Tensor

3D convolution for volumetric data (video, medical imaging, 3D point clouds).

**Shapes:**
- Input: (batch, D, H, W, in_channels)
- Kernel: (out_channels, in_channels, kD, kH, kW)
- Output: (batch, out_D, out_H, out_W, out_channels)

**Implementation:** `conv3d` in [lib/backend/tensor_extras_codegen.cpp](../lib/backend/tensor_extras_codegen.cpp).

#### `max-pool2d`

**Syntax:** `(max-pool2d input kernel-size stride padding)` → Tensor

2D max pooling: takes the maximum value in each kernel-sized window. Reduces spatial
dimensions while retaining the most activated features. Provides partial translation
invariance.

**Output:** (batch, ⌊(H + 2·padding - kernel_size) / stride⌋ + 1, analogous for W, channels)

**Implementation:** `maxPool2d` in [lib/backend/tensor_conv_codegen.cpp](../lib/backend/tensor_conv_codegen.cpp).

#### `avg-pool2d`

**Syntax:** `(avg-pool2d input kernel-size stride padding)` → Tensor

2D average pooling: computes the mean value in each kernel-sized window. Smoother than
max pooling — preserves more information about the spatial distribution of activations.

**Implementation:** `avgPool2d` in [lib/backend/tensor_conv_codegen.cpp](../lib/backend/tensor_conv_codegen.cpp).

#### `batch-norm`

**Syntax:** `(batch-norm input γ β ε)` → Tensor

Batch Normalization (Ioffe & Szegedy, 2015): normalizes each feature across the batch dimension:

```
ŷ = γ · (x - μ_batch) / √(σ²_batch + ε) + β
```

where μ_batch and σ²_batch are the mean and variance computed per feature over the batch.
γ (scale) and β (shift) are learnable parameters that allow the network to restore
representational power after normalization.

**Properties:** Reduces internal covariate shift, enables higher learning rates, acts as
regularization. Batch-dependent — behavior changes between training and inference.

**Implementation:** `batchNorm` in [lib/backend/tensor_conv_codegen.cpp](../lib/backend/tensor_conv_codegen.cpp).

#### `layer-norm`

**Syntax:** `(layer-norm input γ β ε)` → Tensor

Layer Normalization (Ba et al., 2016): normalizes across features within each sample:

```
ŷ = γ · (x - μ_sample) / √(σ²_sample + ε) + β
```

Unlike batch norm, statistics are computed per-sample, making layer norm independent of
batch size. This is critical for transformer architectures where batch sizes vary and
sequence positions must be processed independently.

**Implementation:** `layerNorm` in [lib/backend/tensor_conv_codegen.cpp](../lib/backend/tensor_conv_codegen.cpp).

#### CNN Example

```scheme
;; Simple convolutional feature extractor
(define (cnn-forward x conv-k1 conv-k2 gamma beta)
  (let* ((h1 (relu (conv2d x conv-k1 1 1)))       ; conv + ReLU
         (h2 (max-pool2d h1 2 2 0))                ; downsample 2×
         (h3 (relu (conv2d h2 conv-k2 1 1)))       ; second conv
         (h4 (batch-norm h3 gamma beta 1e-5)))     ; normalize
    h4))
```

### Transformer Operations

The transformer architecture (Vaswani et al., 2017) replaces sequential recurrence with
self-attention, enabling parallel processing of entire sequences. The core mechanism is
scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

where Q (queries), K (keys), and V (values) are linear projections of the input, and d_k
is the key dimensionality. The scaling factor √d_k prevents softmax saturation for large d_k.

Multi-head attention projects Q, K, V into h parallel subspaces, computes attention
independently in each, concatenates results, and applies an output projection — enabling
the model to attend to information from different representation subspaces simultaneously.

#### `scaled-dot-attention`

**Syntax:** `(scaled-dot-attention Q K V [mask])` → Tensor

Core attention mechanism: softmax(QK^T / √d_k + mask) · V.

**Shapes:**
- Q: (..., seq_q, d_k) — queries
- K: (..., seq_k, d_k) — keys
- V: (..., seq_k, d_v) — values
- mask (optional): (..., seq_q, seq_k) — 0 = attend, -∞ = block
- Output: (..., seq_q, d_v)

**Complexity:** O(seq_q · seq_k · d_k) for the QK^T product.

**Implementation:** `scaledDotProductAttention` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### `multi-head-attention`

**Syntax:** `(multi-head-attention Q K V n-heads W_Q W_K W_V W_O [mask])` → Tensor

Full multi-head attention:

1. **Project:** Q' = QW_Q, K' = KW_K, V' = VW_V
2. **Split:** Reshape each into (batch, n-heads, seq, d_k/n-heads)
3. **Attend:** Apply scaled dot-product attention per head
4. **Concat:** Concatenate head outputs
5. **Project:** Output = concat · W_O

**Shapes:**
- W_Q, W_K, W_V: (d_model, d_model) — projection matrices
- W_O: (d_model, d_model) — output projection
- n-heads must evenly divide d_model

**Implementation:** `multiHeadAttention` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### `positional-encoding`

**Syntax:** `(positional-encoding max-len d-model)` → Tensor

Sinusoidal positional encoding (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Output shape:** (max-len, d-model). Added to token embeddings to inject position information.

The sinusoidal design enables the model to generalize to sequence lengths not seen during
training, and allows it to learn relative position attention via linear combinations.

**Implementation:** `positionalEncoding` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### `rotary-embedding`

**Syntax:** `(rotary-embedding x seq-positions dim)` → Tensor

Rotary Position Embedding (RoPE) (Su et al., 2021): encodes relative positions via rotation
of query/key vectors in pairs of dimensions.

**Theory:** Each pair of dimensions (2i, 2i+1) is rotated by θ_i · position, where
θ_i = 1/10000^(2i/d). The dot product Q·K then depends only on the relative position
m - n, not absolute positions — enabling length generalization.

**Advantages over sinusoidal PE:** Encodes relative positions naturally, decays with distance,
and is more efficient (no additive embedding table needed).

**Implementation:** `rotaryEmbedding` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### `causal-mask`

**Syntax:** `(causal-mask seq-len)` → Tensor

Creates an autoregressive (causal) attention mask: lower-triangular matrix of zeros with
upper triangle filled with -∞. Shape: (seq-len, seq-len).

Prevents position i from attending to positions j > i, enforcing the left-to-right
generation constraint in decoder models (GPT, language models).

**Implementation:** `causalMask` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### `padding-mask`

**Syntax:** `(padding-mask lengths max-len)` → Tensor

Creates a padding mask for variable-length sequences. Marks valid positions as 0 and
padding positions as -∞. Shape: (batch, max-len).

`lengths` is a tensor of actual sequence lengths for each batch element.

**Implementation:** `paddingMask` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### `feed-forward`

**Syntax:** `(feed-forward x W₁ b₁ W₂ b₂)` → Tensor

Position-wise feed-forward network: FFN(x) = W₂ · ReLU(W₁x + b₁) + b₂.

This is the standard two-layer MLP applied independently to each position in the sequence.
Typically d_ff = 4 · d_model (the inner dimension is 4× the model dimension).

**Implementation:** `feedForward` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### `embedding`

**Syntax:** `(embedding indices weights)` → Tensor

Token embedding lookup. Maps integer token indices to dense vectors by indexing into
the weight matrix.

**Shapes:**
- indices: (batch, seq) — integer token IDs
- weights: (vocab_size, d_model) — embedding matrix
- Output: (batch, seq, d_model)

**Implementation:** `embedding` in [lib/backend/tensor_transformer_codegen.cpp](../lib/backend/tensor_transformer_codegen.cpp).

#### Transformer Example

```scheme
;; Transformer decoder block
(define (decoder-block x n-heads W_Q W_K W_V W_O
                       W1 b1 W2 b2
                       gamma1 beta1 gamma2 beta2)
  (let* ((mask (causal-mask (tensor-length x)))
         (attn (multi-head-attention x x x n-heads
                 W_Q W_K W_V W_O mask))
         (x1 (layer-norm (tensor-add x attn)             ; residual + norm
                gamma1 beta1 1e-5))
         (ff (feed-forward x1 W1 b1 W2 b2))
         (x2 (layer-norm (tensor-add x1 ff)              ; residual + norm
                gamma2 beta2 1e-5)))
    x2))

;; Full forward pass with embedding + positional encoding
(define (transformer-forward tokens weights pos-enc blocks)
  (let* ((embed (embedding tokens weights))
         (x (tensor-add embed pos-enc)))
    (fold (lambda (block x) (block x))
          x blocks)))
```

### Complete Training Loop Example

This end-to-end example demonstrates how Eshkol's AD, tensor, and ML systems compose into
a complete neural network training pipeline.

```scheme
;; 1. Create and initialize weights
(define W1 (make-tensor (list 784 128) 0.0))
(define b1 (make-tensor (list 128) 0.0))
(define W2 (make-tensor (list 128 10) 0.0))
(define b2 (make-tensor (list 10) 0.0))

(kaiming-normal! W1 784)     ; He initialization for ReLU
(xavier-normal! W2 128 10)   ; Xavier for output layer

;; 2. Allocate optimizer state (Adam)
(define m-W1 (make-tensor (list 784 128) 0.0))
(define v-W1 (make-tensor (list 784 128) 0.0))
(define m-W2 (make-tensor (list 128 10) 0.0))
(define v-W2 (make-tensor (list 128 10) 0.0))

;; 3. Define forward pass
(define (forward x)
  (let* ((h (relu (tensor-add (tensor-matmul x W1) b1)))
         (h (dropout h 0.2 #t))         ; 20% dropout during training
         (logits (tensor-add (tensor-matmul h W2) b2)))
    logits))

;; 4. Training loop
(do ((step 1 (+ step 1))) ((> step 10000))
  (let* ((batch (dataloader-next loader))
         (x (car batch)) (y (cdr batch))

         ;; Forward + loss
         (logits (forward x))
         (loss (cross-entropy-loss logits y))

         ;; Backward (AD computes all gradients)
         (grads (gradient (lambda (params)
           (cross-entropy-loss (forward x) y)) params))

         ;; Learning rate schedule
         (lr (if (< step 1000)
               (linear-warmup-lr step 1000 0.001)
               (cosine-annealing-lr (- step 1000) 9000 0.001))))

    ;; Gradient clipping
    (clip-grad-norm! (car grads) 1.0)

    ;; Optimizer step
    (adam-step W1 (car grads) lr m-W1 v-W1 step)
    (adam-step W2 (cadr grads) lr m-W2 v-W2 step)

    ;; Zero gradients for next iteration
    (zero-grad! (car grads))
    (zero-grad! (cadr grads))))
```

---

## GPU Operations

GPU acceleration via Metal (Apple Silicon) and CUDA (NVIDIA). Dispatch is automatic based
on a cost model comparing CPU (cBLAS/SIMD) vs GPU throughput, accounting for data transfer
latency.

**Implementation:** [gpu_memory.mm](lib/backend/gpu/gpu_memory.mm) (Metal),
[system_codegen.cpp](lib/backend/system_codegen.cpp) (dispatch logic).

### Automatic Dispatch

The compiler automatically dispatches tensor operations to GPU when beneficial:

- Element-wise operations (add, sub, mul, div) on large tensors
- Matrix multiplication above cost model threshold
- Reduction operations (sum, max, min)
- Softmax normalization
- Matrix transpose

No user code changes needed — the same tensor operations (`matmul`, `tensor-add`, etc.)
automatically use GPU when available and the cost model predicts a speedup. The dispatch
chain is: SIMD → cBLAS (Apple Accelerate / AMX) → GPU.

### Cost Model

Dispatch decisions are based on measured peak performance:

| Backend | Peak FLOPS | Notes |
|---------|-----------|-------|
| cBLAS (Apple Accelerate AMX) | ~1100 GFLOPS | Measured; up to 1.2 TFLOPS for large matrices |
| GPU (Metal SF64) | ~200 GFLOPS | Software float64 emulation |
| GPU (CUDA native f64) | Hardware-dependent | Native double precision |

GPU is selected only when: `data_size × compute_intensity > transfer_latency_cost`,
meaning the computation must be intensive enough to amortize the host ↔ device data transfer.
For most operations below ~4096 elements, CPU (cBLAS) is faster.

### SF64 — Software Float64 on Metal

Apple's Metal Shading Language lacks native float64 support. Eshkol implements **SF64**
(Software Float64) — a double-precision emulation layer running on the GPU. SF64 represents
each double as a pair of floats using Dekker's algorithm for error-free arithmetic, achieving
full 64-bit precision in software.

**When to use SF64:** For large matrices where the data transfer cost is amortized by the
massively parallel GPU execution. For small to medium tensors, cBLAS on CPU is significantly
faster due to zero transfer overhead and native double precision.

**Enable:** `ESHKOL_GPU_PRECISION=sf64`

### Memory Management

GPU memory is managed through a pooled allocation strategy:

- **Automatic transfer:** Host ↔ device copies happen transparently during dispatch
- **Pooled buffers:** Frequently-used buffer sizes are recycled to avoid allocation overhead
- **Lazy synchronization:** Results are only copied back to the host when accessed by CPU code

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_ENABLE_GPU` | auto-detect | Enable/disable GPU dispatch (`0` or `1`) |
| `ESHKOL_GPU_THRESHOLD` | cost-model | Minimum tensor size for GPU dispatch |
| `ESHKOL_GPU_PRECISION` | auto | Precision mode: `auto`, `sf64`, `f32` |
| `ESHKOL_GPU_VERBOSE` | 0 | Enable verbose GPU logging |
| `ESHKOL_BLAS_PEAK_GFLOPS` | 1100 | Measured CPU BLAS peak (for cost model) |
| `ESHKOL_GPU_PEAK_GFLOPS` | 200 | Measured GPU peak (for cost model) |

---

## XLA Backend

XLA (Accelerated Linear Algebra) provides an alternative compilation path for tensor
operations via MLIR and StableHLO. Instead of generating direct LLVM IR for each tensor
operation, XLA captures sequences of operations and compiles them as fused kernels —
eliminating intermediate tensor allocations and enabling whole-graph optimizations.

**Architecture:** Eshkol tensor IR → StableHLO dialect → MLIR optimization passes →
target-specific code (CPU with SIMD / GPU via Metal or CUDA).

**Implementation:** [xla_runtime.cpp](lib/backend/xla/xla_runtime.cpp).

### Operation Fusion

XLA's primary advantage is **operation fusion**: chains of element-wise operations and
reductions are compiled into single kernels. For example:

```scheme
;; Without XLA: 3 intermediate tensors allocated
(tensor-add (tensor-mul A B) (tensor-mul C D))

;; With XLA: fused into 1 kernel, 0 intermediates
;; Equivalent computation, but compiled as: result[i] = A[i]*B[i] + C[i]*D[i]
```

This is particularly impactful for training loops where the same operation graph is
executed thousands of times — the compilation cost is amortized over many iterations.

### When XLA Outperforms Direct Codegen

- **Long operation chains:** 3+ chained element-wise operations benefit from fusion
- **Repeated execution:** The JIT compilation cost is amortized over multiple runs
- **Large tensors:** Memory bandwidth savings from fusion are proportional to tensor size

### When Direct Codegen is Preferred

- **Single operations:** No fusion opportunity; direct LLVM IR avoids JIT overhead
- **Small tensors:** JIT compilation latency dominates execution time
- **First execution:** Cold-start compilation penalty before any speedup

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_ENABLE_XLA` | 0 | Enable XLA compilation path (`1` to enable) |
| `ESHKOL_XLA_VERBOSE` | 0 | Log XLA compilation and fusion decisions |

---

## Signal Processing

### FFT / IFFT

Built-in Cooley-Tukey radix-2 FFT:

- `(fft tensor)` — Forward FFT, returns complex tensor
- `(ifft tensor)` — Inverse FFT

### Signal Processing Library (`signal.filters`)

```scheme
(require signal.filters)
```

#### Window Functions
- `(hamming-window n)` — Hamming window of length n
- `(hann-window n)` — Hann window
- `(blackman-window n)` — Blackman window
- `(kaiser-window n beta)` — Kaiser window with parameter beta

#### Convolution
- `(convolve signal kernel)` — Direct convolution O(N*M)
- `(fft-convolve signal kernel)` — FFT-based convolution O(N log N)

#### Filters
- `(fir-filter coefficients signal)` — FIR filter application
- `(iir-filter b-coeffs a-coeffs signal)` — IIR Direct Form I
- `(butterworth-lowpass order cutoff sample-rate)` — Butterworth lowpass design
- `(butterworth-highpass order cutoff sample-rate)` — Butterworth highpass
- `(butterworth-bandpass order low high sample-rate)` — Butterworth bandpass

#### Analysis
- `(frequency-response b-coeffs a-coeffs n-points)` — Magnitude + phase response

---

## Bytevectors

R7RS bytevector operations for binary data.

- `(make-bytevector k [fill])` — Create bytevector of length k
- `(bytevector b1 b2 ...)` — Create from byte values
- `(bytevector-length bv)` — Length
- `(bytevector-u8-ref bv k)` — Read byte at index k
- `(bytevector-u8-set! bv k byte)` — Write byte at index k
- `(bytevector-copy bv [start [end]])` — Copy (sub)range
- `(bytevector-copy! to at from [start [end]])` — Copy into existing
- `(bytevector-append bv1 bv2 ...)` — Concatenate
- `(bytevector? x)` — Type predicate

---

## Environment Variables

Eshkol behavior can be configured via environment variables. None are required — all have sensible defaults.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_PATH` | (none) | Library search path |
| `ESHKOL_STACK_SIZE` | (system) | Runtime stack size |
| `ESHKOL_MAX_HEAP` | (default) | Maximum heap allocation |
| `ESHKOL_TIMEOUT_MS` | (none) | Execution timeout in milliseconds; hosted runtime starts a watchdog when set |
| `ESHKOL_MAX_STACK` | (default) | Maximum stack depth |
| `ESHKOL_MAX_TENSOR_ELEMS` | (default) | Maximum tensor element count |
| `ESHKOL_MAX_STRING_LEN` | (default) | Maximum string length |
| `ESHKOL_ENFORCE_LIMITS` | on for limit API | Request runtime interrupts on hard heap/timeout limits |
| `ESHKOL_LIMIT_WARNINGS` | on | Emit soft-limit warnings |

Malformed runtime-limit values are ignored and preserve the default limit. Size
variables accept optional `K`, `M`, or `G` suffixes, with an optional trailing
`B`.

### Logging & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_LOG_LEVEL` | (default) | Log verbosity level |
| `ESHKOL_LOG_FILE` | (none) | Log file path |
| `ESHKOL_DEBUG` | 0 | Enable debug mode |
| `ESHKOL_DUMP_REPL_IR` | off | Dump REPL LLVM IR |

### Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_OPT_LEVEL` | (default) | Optimization level |
| `ESHKOL_ENABLE_SIMD` | on | Enable SIMD acceleration |
| `ESHKOL_ENABLE_XLA` | off | Enable XLA backend |
| `ESHKOL_ENABLE_GPU` | auto | Enable GPU acceleration |

### GPU Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_GPU_PRECISION` | auto | GPU precision mode |
| `ESHKOL_GPU_THRESHOLD` | auto | GPU dispatch threshold |
| `ESHKOL_GPU_VERBOSE` | off | GPU verbose logging |
| `ESHKOL_BLAS_PEAK_GFLOPS` | 1100 | CPU BLAS peak GFLOPS |
| `ESHKOL_GPU_PEAK_GFLOPS` | 200 | GPU peak GFLOPS |

### Threading

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_DISABLE_WORK_STEALING` | off | Disable work stealing in thread pool |

### Package Manager

| Variable | Default | Description |
|----------|---------|-------------|
| `ESHKOL_REGISTRY` | (default) | Package registry URL |
| `ESHKOL_COMPILER` | (default) | Compiler path for builds |

---

## Statistics

Statistical analysis functions for tensors and lists. All functions accept tensors, vectors, or lists as input and internally flatten to a uniform representation before computation.

**Module**: `lib/math/statistics.esk`
**Import**: `(require statistics)`

### Central Tendency and Spread

#### `median`
**Syntax**: `(median data)`

Computes the median (50th percentile) of a dataset. For datasets of even cardinality, returns the arithmetic mean of the two central values. The input is sorted internally.

**Examples**:
```scheme
(median #(3 1 4 1 5))       ; => 3
(median #(1 2 3 4))          ; => 2.5
(median '(10 20 30 40 50))   ; => 30
```

**Returns**: Number (exact midpoint value or interpolated mean)

---

#### `percentile`
**Syntax**: `(percentile data p)`

Computes the *p*-th percentile of a dataset using linear interpolation between adjacent ranks. The parameter *p* ranges from 0 to 100. For a sorted dataset of size *n*, the index is computed as `i = p/100 * (n-1)`, and the result is linearly interpolated between `data[floor(i)]` and `data[ceil(i)]`.

**Examples**:
```scheme
(percentile #(1 2 3 4 5 6 7 8 9 10) 25)   ; => 3.25
(percentile #(1 2 3 4 5 6 7 8 9 10) 50)   ; => 5.5
(percentile #(1 2 3 4 5 6 7 8 9 10) 90)   ; => 9.1
```

**Returns**: Number (interpolated percentile value)

---

#### `quartiles`
**Syntax**: `(quartiles data)`

Returns the three quartile boundaries Q1, Q2, Q3 (25th, 50th, and 75th percentiles) as a three-element list.

**Examples**:
```scheme
(quartiles #(1 2 3 4 5 6 7 8 9 10))  ; => (3.25 5.5 7.75)
```

**Returns**: List of three numbers `(Q1 Q2 Q3)`

---

#### `iqr`
**Syntax**: `(iqr data)`

Computes the interquartile range, defined as IQR = Q3 - Q1. This is a robust measure of statistical dispersion that is resistant to outliers.

**Examples**:
```scheme
(iqr #(1 2 3 4 5 6 7 8 9 10))  ; => 4.5
```

**Returns**: Number (Q3 - Q1)

---

#### `variance`
**Syntax**: `(variance data)`

Computes the population variance of a dataset, defined as:

    Var(X) = (1/n) * Sum_i (x_i - mu)^2

where mu is the arithmetic mean and n is the number of observations. This is the biased estimator; for the unbiased sample variance, multiply by `n/(n-1)`.

**Examples**:
```scheme
(variance #(2 4 4 4 5 5 7 9))  ; => 4.0
(variance '(1 2 3 4 5))        ; => 2.0
```

**Returns**: Number (non-negative population variance)

---

#### `std-dev`
**Syntax**: `(std-dev data)`

Computes the population standard deviation, defined as the square root of the population variance:

    sigma = sqrt(Var(X))

**Examples**:
```scheme
(std-dev #(2 4 4 4 5 5 7 9))  ; => 2.0
(std-dev '(1 2 3 4 5))        ; => 1.4142135623730951
```

**Returns**: Number (non-negative standard deviation)

---

### Correlation and Covariance

#### `covariance`
**Syntax**: `(covariance xs ys)`

Computes the population covariance between two datasets:

    Cov(X,Y) = (1/n) * Sum_i (x_i - mu_x)(y_i - mu_y)

If the datasets differ in length, the shorter length is used.

**Examples**:
```scheme
(covariance #(1 2 3 4 5) #(2 4 6 8 10))  ; => 4.0
(covariance #(1 2 3) #(3 2 1))            ; => -0.6666666666666666
```

**Returns**: Number (covariance, may be negative)

---

#### `correlation`
**Syntax**: `(correlation xs ys)`

Computes the Pearson product-moment correlation coefficient:

    r = Cov(X,Y) / (sigma_x * sigma_y)

The result lies in [-1, 1]. Returns 0.0 if either standard deviation is zero.

**Examples**:
```scheme
(correlation #(1 2 3 4 5) #(2 4 6 8 10))  ; => 1.0
(correlation #(1 2 3 4 5) #(5 4 3 2 1))   ; => -1.0
```

**Returns**: Number in [-1, 1]

---

### Histogram and Binning

#### `histogram`
**Syntax**: `(histogram data num-bins)`

Constructs a histogram by partitioning the data range into `num-bins` equal-width bins. Returns an association list of `(bin-center . count)` pairs.

**Examples**:
```scheme
(histogram #(1 2 2 3 3 3 4 4 5) 3)
;; => ((2.333... . 3) (3.666... . 4) (5.0 . 2))
```

**Returns**: List of `(center . count)` pairs

---

#### `bin-data`
**Syntax**: `(bin-data data num-bins)`

Assigns each data value to a zero-indexed bin. The data range is partitioned into `num-bins` equal-width intervals, and each value is mapped to its corresponding bin index.

**Examples**:
```scheme
(bin-data #(1 3 5 7 9) 3)  ; => (0 0 1 2 2)
```

**Returns**: List of integers (bin indices)

---

### Normalization and Scoring

#### `zscore`
**Syntax**: `(zscore data)`

Computes the z-score (standard score) for each observation:

    z_i = (x_i - mu) / sigma

If the standard deviation is zero, all z-scores are returned as 0.0.

**Examples**:
```scheme
(zscore #(2 4 4 4 5 5 7 9))  ; => (-1.5 -0.5 -0.5 -0.5 0.0 0.0 1.0 2.0)
```

**Returns**: List of numbers (z-scores)

---

#### `describe`
**Syntax**: `(describe data)`

Produces a summary statistics table as an association list containing: count, mean, standard deviation, minimum, Q1, median, Q3, and maximum.

**Examples**:
```scheme
(describe #(1 2 3 4 5))
;; => ((count . 5) (mean . 3.0) (std . 1.414...) (min . 1)
;;     (q1 . 2.0) (median . 3.0) (q3 . 4.0) (max . 5))
```

**Returns**: Association list of `(symbol . value)` pairs

---

## Tensor Utilities

High-level shape manipulation and query utilities for tensors. These functions provide convenient wrappers over the core tensor operations documented in [Tensor Operations](#tensor-operations).

**Module**: `lib/tensor/utils.esk`
**Import**: `(require tensor-utils)`

### Shape Manipulation

#### `hstack`
**Syntax**: `(hstack tensor1 tensor2 ...)`

Horizontally stacks tensors by concatenating along axis 1. For 2D tensors, this joins columns side by side. All tensors must have the same number of rows.

**Examples**:
```scheme
(hstack #((1 2) (3 4)) #((5) (6)))  ; => #((1 2 5) (3 4 6))
```

**Returns**: Tensor (concatenated along axis 1)

---

#### `vstack`
**Syntax**: `(vstack tensor1 tensor2 ...)`

Vertically stacks tensors by concatenating along axis 0. For 2D tensors, this joins rows top to bottom. All tensors must have the same number of columns.

**Examples**:
```scheme
(vstack #((1 2) (3 4)) #((5 6)))  ; => #((1 2) (3 4) (5 6))
```

**Returns**: Tensor (concatenated along axis 0)

---

#### `reshape-as`
**Syntax**: `(reshape-as tensor template)`

Reshapes `tensor` to match the shape of `template`. The total number of elements must be equal.

**Examples**:
```scheme
(define A #(1 2 3 4 5 6))
(define B #((0 0 0) (0 0 0)))
(reshape-as A B)  ; => #((1 2 3) (4 5 6))
```

**Returns**: Tensor (reshaped to template's shape)

---

#### `squeeze-all`
**Syntax**: `(squeeze-all tensor)`

Removes all dimensions of size 1 from the tensor shape. Delegates to the builtin `squeeze`.

**Examples**:
```scheme
(tensor-shape (squeeze-all (reshape #(1 2 3) 1 3 1)))  ; => (3)
```

**Returns**: Tensor (with all size-1 dimensions removed)

---

#### `unsqueeze-first`
**Syntax**: `(unsqueeze-first tensor)`

Inserts a new dimension of size 1 at position 0 (the leading axis). Useful for converting a vector to a row matrix.

**Examples**:
```scheme
(tensor-shape (unsqueeze-first #(1 2 3)))  ; => (1 3)
```

**Returns**: Tensor (with new leading dimension)

---

#### `unsqueeze-last`
**Syntax**: `(unsqueeze-last tensor)`

Inserts a new dimension of size 1 at the trailing position. Useful for converting a vector to a column matrix.

**Examples**:
```scheme
(tensor-shape (unsqueeze-last #(1 2 3)))  ; => (3 1)
```

**Returns**: Tensor (with new trailing dimension)

---

#### `expand-dims`
**Syntax**: `(expand-dims tensor axis)`

Alias for `unsqueeze`. Inserts a new dimension of size 1 at the specified axis position.

**Examples**:
```scheme
(tensor-shape (expand-dims #(1 2 3) 0))  ; => (1 3)
(tensor-shape (expand-dims #(1 2 3) 1))  ; => (3 1)
```

**Returns**: Tensor (with new dimension at specified axis)

---

#### `broadcast-to`
**Syntax**: `(broadcast-to tensor target-shape)`

Broadcasts a tensor to a target shape. The tensor's total element count must match the product of the target dimensions. Full NumPy-style broadcasting with dimension expansion is performed automatically during arithmetic operations; this function handles explicit reshape-based broadcasting.

**Examples**:
```scheme
(broadcast-to #(1 2 3 4 5 6) '(2 3))  ; => #((1 2 3) (4 5 6))
```

**Returns**: Tensor (reshaped to target dimensions)

---

### Tensor Query

#### `tensor-ndim`
**Syntax**: `(tensor-ndim tensor)`

Returns the number of dimensions (rank) of the tensor.

**Examples**:
```scheme
(tensor-ndim #(1 2 3))           ; => 1
(tensor-ndim #((1 2) (3 4)))     ; => 2
(tensor-ndim (zeros 2 3 4))      ; => 3
```

**Returns**: Integer (number of dimensions)

---

#### `tensor-size`
**Syntax**: `(tensor-size tensor)`

Returns the total number of elements in the tensor (product of all dimension sizes).

**Examples**:
```scheme
(tensor-size #(1 2 3 4 5))       ; => 5
(tensor-size #((1 2) (3 4)))     ; => 4
(tensor-size (zeros 2 3 4))      ; => 24
```

**Returns**: Integer (total element count)

---

#### `is-scalar?`
**Syntax**: `(is-scalar? tensor)`

Returns `#t` if the tensor contains exactly one element (regardless of shape).

**Examples**:
```scheme
(is-scalar? #(42))              ; => #t
(is-scalar? #(1 2))             ; => #f
```

**Returns**: Boolean

---

#### `is-vector?`
**Syntax**: `(is-vector? tensor)`

Returns `#t` if the tensor is one-dimensional.

**Examples**:
```scheme
(is-vector? #(1 2 3))           ; => #t
(is-vector? #((1 2) (3 4)))     ; => #f
```

**Returns**: Boolean

---

#### `is-matrix?`
**Syntax**: `(is-matrix? tensor)`

Returns `#t` if the tensor is two-dimensional.

**Examples**:
```scheme
(is-matrix? #((1 2) (3 4)))     ; => #t
(is-matrix? #(1 2 3))           ; => #f
```

**Returns**: Boolean

---

### Slicing Utilities

#### `tensor-take`
**Syntax**: `(tensor-take tensor n)`

Returns the first `n` elements from a 1D tensor.

**Examples**:
```scheme
(tensor-take #(10 20 30 40 50) 3)  ; => #(10 20 30)
```

**Returns**: Tensor (first n elements)

---

#### `tensor-drop`
**Syntax**: `(tensor-drop tensor n)`

Returns all elements except the first `n` from a 1D tensor.

**Examples**:
```scheme
(tensor-drop #(10 20 30 40 50) 2)  ; => #(30 40 50)
```

**Returns**: Tensor (elements after position n)

---

#### `tensor-head`
**Syntax**: `(tensor-head tensor)`

Returns the first element as a scalar tensor (slice of length 1).

**Examples**:
```scheme
(tensor-head #(10 20 30))  ; => #(10)
```

**Returns**: Tensor (single-element slice)

---

#### `tensor-tail`
**Syntax**: `(tensor-tail tensor)`

Returns all elements except the first.

**Examples**:
```scheme
(tensor-tail #(10 20 30 40))  ; => #(20 30 40)
```

**Returns**: Tensor (all but first element)

---

## Special Functions

Special mathematical functions for scientific and statistical computing. Implementations use established numerical approximation algorithms (Lanczos, Abramowitz & Stegun, Miller recurrence) with precision suitable for double-precision arithmetic.

**Module**: `lib/math/special.esk`
**Import**: `(require "math/special")`

### Gamma Function and Related

#### `gamma`
**Syntax**: `(gamma z)`

Computes the gamma function using the Lanczos approximation with g=7:

    Gamma(z) = integral from 0 to infinity of t^(z-1) * e^(-t) dt

For z < 0.5, the reflection formula is applied: Gamma(1-z) * Gamma(z) = pi / sin(pi*z). Extends the factorial to the complex plane: Gamma(n+1) = n! for non-negative integers.

**Examples**:
```scheme
(gamma 5.0)    ; => 24.0  (4!)
(gamma 0.5)    ; => 1.7724538509055159  (sqrt(pi))
(gamma 1.0)    ; => 1.0
```

**Returns**: Number (Gamma(z), undefined for non-positive integers)

---

#### `lgamma`
**Syntax**: `(lgamma z)`

Computes the natural logarithm of the gamma function: ln(Gamma(z)). This is numerically more stable than `(log (gamma z))` for large arguments, avoiding overflow.

**Examples**:
```scheme
(lgamma 100.0)  ; => 359.1342053695754
(lgamma 0.5)    ; => 0.5723649429247001  (ln(sqrt(pi)))
```

**Returns**: Number (ln(Gamma(z)))

---

#### `factorial`
**Syntax**: `(factorial n)`

Computes n! via the gamma function: n! = Gamma(n+1). Returns `#f` for negative integers. For non-negative integers less than 2, returns 1.0.

**Examples**:
```scheme
(factorial 0)   ; => 1.0
(factorial 5)   ; => 120.0
(factorial 10)  ; => 3628800.0
(factorial -1)  ; => #f
```

**Returns**: Number (n!) or `#f` for invalid input

---

#### `beta`
**Syntax**: `(beta a b)`

Computes the beta function via the log-gamma representation for numerical stability:

    B(a,b) = Gamma(a) * Gamma(b) / Gamma(a+b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))

**Examples**:
```scheme
(beta 2.0 3.0)  ; => 0.08333333333333333  (1/12)
(beta 0.5 0.5)  ; => 3.141592653589793    (pi)
```

**Returns**: Number (B(a,b))

---

#### `digamma`
**Syntax**: `(digamma x)`

Computes the digamma (psi) function, the logarithmic derivative of the gamma function:

    psi(x) = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x)

Uses the recurrence relation psi(x+1) = psi(x) + 1/x for small x, and an asymptotic expansion for x >= 6.

**Examples**:
```scheme
(digamma 1.0)  ; => -0.5772156649015329  (-euler-gamma)
(digamma 2.0)  ; => 0.4227843350984671
```

**Returns**: Number (psi(x))

---

### Error Function and Related

#### `erf`
**Syntax**: `(erf x)`

Computes the error function using the Abramowitz & Stegun approximation (7.1.26) with maximum error 1.5e-7:

    erf(x) = (2 / sqrt(pi)) * integral from 0 to x of e^(-t^2) dt

**Examples**:
```scheme
(erf 0.0)    ; => 0.0
(erf 1.0)    ; => 0.8427007929497149
(erf -1.0)   ; => -0.8427007929497149
```

**Returns**: Number in [-1, 1]

---

#### `erfc`
**Syntax**: `(erfc x)`

Computes the complementary error function:

    erfc(x) = 1 - erf(x)

**Examples**:
```scheme
(erfc 0.0)   ; => 1.0
(erfc 3.0)   ; => 0.0000220904969985854
```

**Returns**: Number in [0, 2]

---

#### `erfinv`
**Syntax**: `(erfinv y)`

Computes the inverse error function: returns x such that erf(x) = y. Uses a rational function approximation. Returns `+inf.0` for y=1, `-inf.0` for y=-1, and `+nan.0` for |y| > 1.

**Examples**:
```scheme
(erfinv 0.0)   ; => 0.0
(erfinv 0.5)   ; => 0.4769362762044699
```

**Returns**: Number (x such that erf(x) = y)

---

#### `normcdf`
**Syntax**: `(normcdf x)`

Computes the cumulative distribution function of the standard normal distribution:

    Phi(x) = (1/2) * (1 + erf(x / sqrt(2)))

**Examples**:
```scheme
(normcdf 0.0)    ; => 0.5
(normcdf 1.96)   ; => 0.9750021048517796
(normcdf -1.96)  ; => 0.024997895148220428
```

**Returns**: Number in [0, 1]

---

#### `normpdf`
**Syntax**: `(normpdf x)`

Computes the probability density function of the standard normal distribution:

    phi(x) = (1 / sqrt(2*pi)) * e^(-x^2 / 2)

**Examples**:
```scheme
(normpdf 0.0)   ; => 0.3989422804014327
(normpdf 1.0)   ; => 0.24197072451914337
```

**Returns**: Number (non-negative density)

---

### Bessel Functions

#### `besselj0`
**Syntax**: `(besselj0 x)`

Computes the Bessel function of the first kind of order 0. Uses rational function approximation for |x| < 8 and an asymptotic form for |x| >= 8.

**Examples**:
```scheme
(besselj0 0.0)   ; => 1.0
(besselj0 2.0)   ; => 0.22389077914123567
```

**Returns**: Number (J_0(x))

---

#### `besselj1`
**Syntax**: `(besselj1 x)`

Computes the Bessel function of the first kind of order 1.

**Examples**:
```scheme
(besselj1 0.0)   ; => 0.0
(besselj1 2.0)   ; => 0.5767248077568734
```

**Returns**: Number (J_1(x))

---

#### `besseljn`
**Syntax**: `(besseljn n x)`

Computes the Bessel function of the first kind of integer order *n*, using Miller's downward recurrence algorithm for numerical stability. Negative orders use the identity J_{-n}(x) = (-1)^n * J_n(x).

**Examples**:
```scheme
(besseljn 0 2.0)   ; => 0.22389077914123567  (same as besselj0)
(besseljn 2 3.0)   ; => 0.48609126058589107
(besseljn 5 1.0)   ; => 0.00024975773021123443
```

**Returns**: Number (J_n(x))

---

### Incomplete Gamma Functions

#### `gammainc-lower`
**Syntax**: `(gammainc-lower a x)`

Computes the lower incomplete gamma function using a series expansion:

    gamma(a,x) = integral from 0 to x of t^(a-1) * e^(-t) dt

Returns `#f` if a <= 0 or x < 0. Returns 0.0 if x = 0.

**Examples**:
```scheme
(gammainc-lower 1.0 1.0)  ; => 0.6321205588285577  (1 - e^(-1))
(gammainc-lower 2.0 3.0)  ; => 0.8008517265285442
```

**Returns**: Number (lower incomplete gamma)

---

#### `gammainc-upper`
**Syntax**: `(gammainc-upper a x)`

Computes the upper incomplete gamma function:

    Gamma(a,x) = Gamma(a) - gamma(a,x) = integral from x to infinity of t^(a-1) * e^(-t) dt

**Examples**:
```scheme
(gammainc-upper 1.0 1.0)  ; => 0.36787944117144233  (e^(-1))
```

**Returns**: Number (upper incomplete gamma)

---

### Number-Theoretic and Other Functions

#### `zeta`
**Syntax**: `(zeta s)`

Computes the Riemann zeta function for s > 1 using the Euler-Maclaurin summation formula:

    zeta(s) = Sum_{k=1}^{infinity} 1/k^s

Returns `+inf.0` at the pole s = 1, and `#f` for s < 1 (not implemented for this domain).

**Examples**:
```scheme
(zeta 2.0)   ; => 1.6449340668482264  (pi^2 / 6)
(zeta 4.0)   ; => 1.0823232337111381  (pi^4 / 90)
```

**Returns**: Number (zeta(s) for s > 1)

---

#### `expint-e1`
**Syntax**: `(expint-e1 x)`

Computes the exponential integral:

    E_1(x) = integral from x to infinity of e^(-t)/t dt

Uses a power series for x < 1 and a continued fraction expansion for x >= 1. Returns `+inf.0` for x <= 0.

**Examples**:
```scheme
(expint-e1 1.0)   ; => 0.21938393439552029
(expint-e1 0.1)   ; => 1.8229239584193906
```

**Returns**: Number (E_1(x))

---

## Mathematical Constants

High-precision mathematical and physical constants for scientific computing. All values are defined to the maximum precision of IEEE 754 double-precision floating-point (approximately 15-17 significant digits).

**Module**: `lib/math/constants.esk`
**Import**: `(require "math/constants")`

### Fundamental Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `pi` | 3.14159265358979323846... | Ratio of circumference to diameter |
| `e` | 2.71828182845904523536... | Base of natural logarithm |
| `phi` | 1.61803398874989484820... | Golden ratio: (1 + sqrt(5)) / 2 |
| `euler-gamma` | 0.57721566490153286060... | Euler-Mascheroni constant |

### Derived Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `tau` | 6.28318530717958647692... | 2*pi (full circle in radians) |
| `pi/2` | 1.57079632679489661923... | Half pi |
| `pi/4` | 0.78539816339744830961... | Quarter pi |
| `sqrt-pi` | 1.77245385090551602729... | Square root of pi |
| `sqrt-2pi` | 2.50662827463100050241... | Square root of 2*pi (Gaussian normalization) |

### Logarithmic Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `ln2` | 0.69314718055994530941... | Natural log of 2 |
| `ln10` | 2.30258509299404568401... | Natural log of 10 |
| `log2e` | 1.44269504088896340735... | Log base 2 of e |
| `log10e` | 0.43429448190325182765... | Log base 10 of e |

### Square Root Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `sqrt2` | 1.41421356237309504880... | Square root of 2 |
| `sqrt3` | 1.73205080756887729352... | Square root of 3 |
| `sqrt5` | 2.23606797749978969640... | Square root of 5 |
| `inv-sqrt2` | 0.70710678118654752440... | 1/sqrt(2) |

### Machine Precision

| Constant | Value | Description |
|----------|-------|-------------|
| `machine-epsilon` | 2.220446049250313e-16 | Smallest x where 1.0 + x != 1.0 |
| `double-min` | 2.2250738585072014e-308 | Smallest positive normalized double |
| `double-max` | 1.7976931348623157e+308 | Largest finite double |
| `epsilon` | 1e-15 | Small value for numerical comparisons |

### Physical Constants (2019 CODATA SI)

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| `c` | *c* | 299792458.0 | m/s |
| `h` | *h* | 6.62607015e-34 | J*s |
| `hbar` | *hbar* | 1.054571817e-34 | J*s |
| `elementary-charge` | *e* | 1.602176634e-19 | C |
| `k-boltzmann` | *k_B* | 1.380649e-23 | J/K |
| `avogadro` | *N_A* | 6.02214076e23 | 1/mol |
| `G` | *G* | 6.67430e-11 | m^3/(kg*s^2) |
| `m-electron` | *m_e* | 9.1093837015e-31 | kg |
| `m-proton` | *m_p* | 1.67262192369e-27 | kg |
| `m-neutron` | *m_n* | 1.67492749804e-27 | kg |
| `alpha` | *alpha* | 7.2973525693e-3 | (dimensionless) |
| `a0` | *a_0* | 5.29177210903e-11 | m |
| `eV` | *eV* | 1.602176634e-19 | J |

### Angle Conversion

| Constant / Function | Description |
|---------------------|-------------|
| `deg->rad` | Multiplier: degrees to radians (pi/180) |
| `rad->deg` | Multiplier: radians to degrees (180/pi) |
| `(degrees->radians deg)` | Converts degrees to radians |
| `(radians->degrees rad)` | Converts radians to degrees |

### Utility Functions

#### `approx=`
**Syntax**: `(approx= a b tolerance)`

Returns `#t` if |a - b| < tolerance.

**Examples**:
```scheme
(approx= 3.14159 pi 0.001)   ; => #t
(approx= 1.0 2.0 0.5)        ; => #f
```

**Returns**: Boolean

---

#### `approx-zero?`
**Syntax**: `(approx-zero? x tolerance)`

Returns `#t` if |x| < tolerance.

**Examples**:
```scheme
(approx-zero? 1e-10 1e-8)  ; => #t
(approx-zero? 0.1 1e-8)    ; => #f
```

**Returns**: Boolean

---

## ODE Solvers

Numerical methods for solving ordinary differential equations of the form dy/dt = f(t, y). All solvers support both scalar states (y is a number) and vector states (y is a Scheme vector). Fixed-step methods return a trajectory as a list of `(t y)` pairs; adaptive methods adjust step size to meet a specified tolerance.

**Module**: `lib/math/ode.esk`
**Import**: `(require "math/ode")`

### Euler Method -- O(h)

#### `euler`
**Syntax**: `(euler f t0 y0 tf h)`

Solves an ODE using Euler's method (first-order explicit). The update rule is:

    y_{n+1} = y_n + h * f(t_n, y_n)

This is the simplest ODE integrator. It is first-order accurate (global error O(h)) and suitable for pedagogical purposes or when function evaluations are expensive and low accuracy suffices.

**Examples**:
```scheme
;; Solve dy/dt = -y, y(0) = 1 from t=0 to t=2 with h=0.1
(define trajectory (euler (lambda (t y) (- y)) 0.0 1.0 2.0 0.1))
(car (reverse trajectory))  ; => (2.0 0.1215...)  (exact: e^(-2) = 0.1353)
```

**Returns**: List of `(t y)` pairs (full trajectory)

---

#### `euler-step`
**Syntax**: `(euler-step f t y h)`

Performs a single Euler step. Useful for custom integration loops.

**Returns**: Number or vector (y at next step)

---

#### `euler-final`
**Syntax**: `(euler-final f t0 y0 tf h)`

Solves using Euler's method and returns only the final value.

**Returns**: Number or vector (y at time tf)

---

### Heun's Method -- O(h^2)

#### `heun`
**Syntax**: `(heun f t0 y0 tf h)`

Solves an ODE using Heun's method (second-order, predictor-corrector). The update rule is:

    y_predict = y_n + h * f(t_n, y_n)
    y_{n+1} = y_n + (h/2) * (f(t_n, y_n) + f(t_{n+1}, y_predict))

Second-order accurate (global error O(h^2)). Requires two function evaluations per step.

**Examples**:
```scheme
(define trajectory (heun (lambda (t y) (- y)) 0.0 1.0 2.0 0.1))
```

**Returns**: List of `(t y)` pairs

---

#### `heun-step`
**Syntax**: `(heun-step f t y h)`

Performs a single Heun step.

**Returns**: Number or vector (y at next step)

---

### Midpoint Method -- O(h^2)

#### `midpoint`
**Syntax**: `(midpoint f t0 y0 tf h)`

Solves an ODE using the explicit midpoint method (second-order). The update rule is:

    k1 = f(t, y)
    y_{n+1} = y_n + h * f(t + h/2, y + h/2 * k1)

**Examples**:
```scheme
(define trajectory (midpoint (lambda (t y) (- y)) 0.0 1.0 2.0 0.1))
```

**Returns**: List of `(t y)` pairs

---

#### `midpoint-step`
**Syntax**: `(midpoint-step f t y h)`

Performs a single midpoint step.

**Returns**: Number or vector (y at next step)

---

### Classical Runge-Kutta -- O(h^4)

#### `rk4`
**Syntax**: `(rk4 f t0 y0 tf h)`

Solves an ODE using the classical fourth-order Runge-Kutta method. The update rule is:

    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

Fourth-order accurate (global error O(h^4)). The workhorse of fixed-step ODE integration, requiring four function evaluations per step.

**Examples**:
```scheme
;; Solve dy/dt = -y, y(0) = 1 from t=0 to t=2 with h=0.1
(define trajectory (rk4 (lambda (t y) (- y)) 0.0 1.0 2.0 0.1))
(car (reverse trajectory))  ; => (2.0 0.13533...)  (exact: 0.13534)

;; Solve a system: d[x,v]/dt = [v, -x] (harmonic oscillator)
(define traj (rk4 (lambda (t state)
                    (vector (vref state 1) (- (vref state 0))))
                  0.0 #(1.0 0.0) 6.28 0.01))
```

**Returns**: List of `(t y)` pairs

---

#### `rk4-step`
**Syntax**: `(rk4-step f t y h)`

Performs a single RK4 step.

**Returns**: Number or vector (y at next step)

---

#### `rk4-final`
**Syntax**: `(rk4-final f t0 y0 tf h)`

Solves using RK4 and returns only the final value.

**Returns**: Number or vector (y at time tf)

---

### Runge-Kutta-Fehlberg -- Adaptive O(h^4)/O(h^5)

#### `rk45`
**Syntax**: `(rk45 f t0 y0 tf h0 tol)`

Solves an ODE using the Runge-Kutta-Fehlberg method with adaptive step size control (Cash-Karp coefficients). Computes both a fourth-order and fifth-order solution at each step; the difference provides an error estimate. The step size is automatically adjusted to keep the local error below `tol`.

**Examples**:
```scheme
;; Adaptive solve with tolerance 1e-8
(define trajectory (rk45 (lambda (t y) (- y)) 0.0 1.0 2.0 0.1 1e-8))
```

**Returns**: List of `(t y)` pairs (variable spacing due to adaptive steps)

---

#### `rk45-step`
**Syntax**: `(rk45-step f t y h tol)`

Performs a single adaptive RK45 step. Returns a list `(y-new h-new accepted?)` where `accepted?` is `#t` if the step met the tolerance.

**Returns**: List `(y-new h-new accepted?)`

---

#### `rk45-final`
**Syntax**: `(rk45-final f t0 y0 tf h0 tol)`

Solves with RK45 and returns only the final value.

**Returns**: Number or vector (y at time tf)

---

### Backward Euler -- Implicit O(h)

#### `backward-euler`
**Syntax**: `(backward-euler f t0 y0 tf h)`

Solves an ODE using the implicit (backward) Euler method with Newton iteration. The implicit equation solved at each step is:

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

Newton iteration (scalar case) or fixed-point iteration (vector case) is used to solve for y_{n+1}. The implicit method is unconditionally A-stable, making it suitable for stiff systems where explicit methods require impractically small step sizes. Uses up to 10 Newton iterations with tolerance 1e-10.

**Examples**:
```scheme
;; Stiff problem: dy/dt = -1000*y
(define trajectory (backward-euler (lambda (t y) (* -1000 y)) 0.0 1.0 0.01 0.001))
```

**Returns**: List of `(t y)` pairs

---

#### `backward-euler-step`
**Syntax**: `(backward-euler-step f t y h max-iters tol)`

Performs a single implicit Euler step with configurable Newton iteration parameters.

**Returns**: Number or vector (y at next step)

---

## Random Numbers

Pseudorandom and hardware-entropy random number generation with support for common probability distributions.

**Module**: `lib/random/random.esk`
**Import**: `(require "random")`

### Basic Pseudorandom Functions

#### `random-float`
**Syntax**: `(random-float)`

Returns a pseudorandom floating-point number uniformly distributed in [0, 1). Delegates to the builtin `random` (drand48-based PRNG).

**Examples**:
```scheme
(random-float)  ; => 0.417022004702574 (varies)
```

**Returns**: Number in [0, 1)

---

#### `random-int`
**Syntax**: `(random-int low high)`

Returns a pseudorandom integer uniformly distributed in [low, high] (inclusive).

**Examples**:
```scheme
(random-int 1 6)    ; => 4 (simulates a die roll)
(random-int 0 100)  ; => 73 (varies)
```

**Returns**: Integer in [low, high]

---

#### `random-bool`
**Syntax**: `(random-bool)`

Returns `#t` or `#f` with equal probability.

**Examples**:
```scheme
(random-bool)  ; => #t or #f
```

**Returns**: Boolean

---

#### `random-choice`
**Syntax**: `(random-choice lst)`

Selects a uniformly random element from a list. Returns `#f` if the list is empty.

**Examples**:
```scheme
(random-choice '(red green blue))  ; => green (varies)
```

**Returns**: Element from list or `#f`

---

### Quantum Random Functions

Quantum random functions derive their entropy from hardware sources — specifically, the OS
entropy pool (`/dev/urandom` on Unix, `CryptGenRandom` on Windows), which collects entropy
from hardware noise sources including thermal noise, interrupt timing jitter, and (where
available) dedicated hardware random number generators using quantum tunneling effects in
semiconductor junctions.

**Key distinction from pseudorandom:** Unlike the `random-*` family (which uses `drand48`,
a deterministic PRNG with a 48-bit linear congruential generator), the `qrandom` family
provides **true non-determinism**. Each call is statistically independent of all previous
and future calls, with no seed and no reproducibility.

**When to use quantum random:**
- Cryptographic nonce generation and key material
- Monte Carlo simulations requiring true statistical independence
- Security-sensitive applications where PRNG state compromise is a concern
- Randomized algorithms where seed-dependent bias could affect results

**When to use pseudorandom (`random-*`):**
- Reproducible experiments (via `set-random-seed!`)
- Performance-critical code (PRNG is ~10× faster than hardware entropy)
- Debugging (deterministic behavior aids reproduction of issues)

#### `qrandom`

**Syntax:** `(qrandom)` → Number

Returns a hardware-entropy random float uniformly distributed in [0, 1). The raw entropy
bytes are passed through a whitening function to ensure uniform distribution.

**Contrast:** `(random-float)` uses `drand48` (deterministic PRNG).

```scheme
(qrandom)   ; => 0.847291... (truly random, never reproducible)
```

---

#### `qrandom-int`

**Syntax:** `(qrandom-int low high)` → Integer

Returns a hardware-entropy random integer uniformly distributed in [low, high] (inclusive).
Uses rejection sampling to avoid modular bias.

**Contrast:** `(random-int low high)` uses `drand48`.

```scheme
(qrandom-int 1 6)     ; => 3 (truly random die roll)
(qrandom-int 0 255)   ; => 187 (random byte)
```

---

#### `qrandom-bool`

**Syntax:** `(qrandom-bool)` → Boolean

Returns `#t` or `#f` with equal probability using hardware entropy.

**Contrast:** `(random-bool)` uses `drand48`.

```scheme
(qrandom-bool)   ; => #t or #f (truly random)
```

---

#### `qrandom-choice`

**Syntax:** `(qrandom-choice lst)` → Value

Selects a uniformly random element from `lst` using hardware entropy. Returns `#f` if the
list is empty.

**Contrast:** `(random-choice lst)` uses `drand48`.

```scheme
(qrandom-choice '(red green blue))   ; => blue (truly random)
```

---

### Distribution Sampling

#### `uniform`

**Syntax:** `(uniform low high)` → Number

Samples from a continuous uniform distribution on [low, high].

```scheme
(uniform 5.0 10.0)  ; => 7.234... (varies)
```

---

#### `quniform`

**Syntax:** `(quniform low high)` → Number

Samples from uniform [low, high] using hardware entropy. Same distribution as `uniform`,
but with non-deterministic entropy source.

**Contrast:** `(uniform low high)` uses `drand48`.

```scheme
(quniform 0.0 1.0)   ; => 0.573... (truly random)
```

---

#### `normal-pair`
**Syntax**: `(normal-pair)`

Generates a pair of independent standard normal random variates using the Box-Muller transform. Returns a two-element list.

**Examples**:
```scheme
(normal-pair)  ; => (0.342... -1.215...)  (varies)
```

**Returns**: List of two standard normal samples

---

#### `normal`
**Syntax**: `(normal mu sigma)`

Samples from a normal (Gaussian) distribution with mean mu and standard deviation sigma, using the Box-Muller transform.

**Examples**:
```scheme
(normal 0.0 1.0)    ; => -0.527... (standard normal)
(normal 100.0 15.0)  ; => 112.3...  (IQ-scale normal)
```

**Returns**: Number (normally distributed)

---

#### `exponential`
**Syntax**: `(exponential lambda)`

Samples from an exponential distribution with rate parameter lambda using inverse transform sampling:

    X = -(1/lambda) * ln(U),  where U ~ Uniform(0,1)

**Examples**:
```scheme
(exponential 1.0)   ; => 0.693... (varies, mean = 1.0)
(exponential 0.5)   ; => 2.415... (varies, mean = 2.0)
```

**Returns**: Non-negative number

---

#### `poisson`
**Syntax**: `(poisson lambda)`

Samples from a Poisson distribution with mean lambda using Knuth's algorithm. Suitable for small to moderate lambda values.

**Examples**:
```scheme
(poisson 5.0)   ; => 4 (varies, mean = 5)
(poisson 0.1)   ; => 0 (varies)
```

**Returns**: Non-negative integer

---

#### `bernoulli`
**Syntax**: `(bernoulli p)`

Samples from a Bernoulli distribution: returns 1 with probability p, 0 otherwise.

**Examples**:
```scheme
(bernoulli 0.7)  ; => 1 or 0 (1 with prob 0.7)
```

**Returns**: 0 or 1

---

#### `geometric`
**Syntax**: `(geometric p)`

Samples from a geometric distribution: returns the number of failures before the first success, where each trial succeeds with probability p.

**Examples**:
```scheme
(geometric 0.5)  ; => 0, 1, 2, ... (varies, mean = 1)
```

**Returns**: Non-negative integer

---

#### `binomial`
**Syntax**: `(binomial n p)`

Samples from a binomial distribution B(n, p): the number of successes in n independent Bernoulli trials, each with success probability p.

**Examples**:
```scheme
(binomial 10 0.5)  ; => 5 (varies, mean = 5)
(binomial 100 0.3) ; => 28 (varies, mean = 30)
```

**Returns**: Integer in [0, n]

---

### Tensor and Vector Random Generation

#### `random-tensor`
**Syntax**: `(random-tensor dims)`

Creates a tensor with elements drawn uniformly from [0, 1). The argument `dims` is a list of dimension sizes.

**Examples**:
```scheme
(random-tensor '(3))      ; => #(0.41 0.72 0.00)  (varies, 1D)
(random-tensor '(2 3))    ; => 2x3 tensor of uniform values
```

**Returns**: Tensor

---

#### `random-normal-tensor`
**Syntax**: `(random-normal-tensor dims)`

Creates a tensor with elements drawn from the standard normal distribution N(0,1).

**Examples**:
```scheme
(random-normal-tensor '(2 2))  ; => 2x2 tensor of normal values
```

**Returns**: Tensor

---

#### `random-vector`
**Syntax**: `(random-vector n)`

Creates a Scheme vector of n random floats in [0, 1).

**Returns**: Vector

---

#### `random-uniform-vector`
**Syntax**: `(random-uniform-vector n low high)`

Creates a Scheme vector of n random floats uniformly distributed in [low, high].

**Returns**: Vector

---

#### `random-normal-vector`
**Syntax**: `(random-normal-vector n)`

Creates a Scheme vector of n standard normal samples.

**Returns**: Vector

---

### Combinatorial Utilities

#### `shuffle`
**Syntax**: `(shuffle lst)`

Returns a random permutation of the list using the Fisher-Yates algorithm.

**Examples**:
```scheme
(shuffle '(1 2 3 4 5))  ; => (3 1 5 2 4)  (varies)
```

**Returns**: List (randomly permuted)

---

#### `sample`
**Syntax**: `(sample lst k)`

Returns a random sample of k elements from the list without replacement.

**Examples**:
```scheme
(sample '(a b c d e) 3)  ; => (c a e)  (varies)
```

**Returns**: List of k elements

---

#### `weighted-choice`
**Syntax**: `(weighted-choice items weights)`

Selects an element from `items` with probability proportional to the corresponding entry in `weights`. Weights must be non-negative.

**Examples**:
```scheme
(weighted-choice '(a b c) '(1 2 7))  ; => c (most likely)
```

**Returns**: Element from items

---

### Seed Control

#### `set-random-seed!`
**Syntax**: `(set-random-seed! seed)`

Sets the pseudorandom number generator seed (drand48). Use for reproducible experiments.

**Examples**:
```scheme
(set-random-seed! 42)
(random-float)  ; => deterministic value
```

**Returns**: Unspecified

---

#### `randomize!`
**Syntax**: `(randomize!)`

Initializes the PRNG with a time-based seed, ensuring different sequences on each run.

**Returns**: Unspecified

---

## Web Platform (WASM)

DOM manipulation and browser APIs for building web applications. All functions operate on integer handles that reference JavaScript objects. Requires compilation with `eshkol-run --wasm`.

**Module**: `lib/web/web.esk`
**Import**: `(require web)`

### Special Handles

#### `web-get-document`
**Syntax**: `(web-get-document)`

Returns the document handle (always handle 1).

**Returns**: Integer (document handle)

---

#### `web-get-window`
**Syntax**: `(web-get-window)`

Returns the window handle (always handle 2).

**Returns**: Integer (window handle)

---

#### `web-get-body`
**Syntax**: `(web-get-body)`

Returns the body element handle (always handle 3).

**Returns**: Integer (body element handle)

---

### Document Methods

#### `web-create-element`
**Syntax**: `(web-create-element tag-name)`

Creates a new DOM element with the given HTML tag name.

**Examples**:
```scheme
(define btn (web-create-element "button"))
(define div (web-create-element "div"))
```

**Returns**: Integer (element handle)

---

#### `web-create-text-node`
**Syntax**: `(web-create-text-node text)`

Creates a DOM text node with the given content.

**Returns**: Integer (node handle)

---

#### `web-get-element-by-id`
**Syntax**: `(web-get-element-by-id id)`

Finds a DOM element by its ID attribute. Returns 0 if not found.

**Examples**:
```scheme
(define el (web-get-element-by-id "my-button"))
```

**Returns**: Integer (element handle, or 0)

---

#### `web-query-selector`
**Syntax**: `(web-query-selector selector)`

Finds the first element matching a CSS selector. Returns 0 if not found.

**Examples**:
```scheme
(define el (web-query-selector ".my-class"))
```

**Returns**: Integer (element handle, or 0)

---

#### `web-query-selector-all`
**Syntax**: `(web-query-selector-all selector)`

Finds all elements matching a CSS selector.

**Returns**: Integer (nodelist handle)

---

### Node Tree Manipulation

#### `web-append-child`
**Syntax**: `(web-append-child parent child)`

Appends a child node to a parent element.

**Examples**:
```scheme
(define div (web-create-element "div"))
(define txt (web-create-text-node "Hello"))
(web-append-child div txt)
(web-append-child (web-get-body) div)
```

**Returns**: Integer (1 on success, 0 on failure)

---

#### `web-remove-child`
**Syntax**: `(web-remove-child parent child)`

Removes a child node from its parent.

**Returns**: Integer (1 on success, 0 on failure)

---

#### `web-insert-before`
**Syntax**: `(web-insert-before parent new-node ref-node)`

Inserts a node before a reference node.

**Returns**: Integer (1 on success, 0 on failure)

---

#### `web-replace-child`
**Syntax**: `(web-replace-child parent new-node old-node)`

Replaces a child node with a new node.

**Returns**: Integer (1 on success, 0 on failure)

---

#### `web-clone-node`
**Syntax**: `(web-clone-node node deep)`

Clones a DOM node. Pass deep=1 for deep clone (includes children).

**Returns**: Integer (cloned node handle)

---

Additional node traversal functions: `web-get-parent`, `web-get-first-child`, `web-get-last-child`, `web-get-next-sibling`, `web-get-prev-sibling`, `web-get-children-count`, `web-get-child-at`.

---

### Content and Attributes

#### `web-set-attribute`
**Syntax**: `(web-set-attribute element name value)`

Sets an HTML attribute on an element.

**Examples**:
```scheme
(web-set-attribute link "href" "https://example.com")
(web-set-attribute img "src" "photo.png")
```

**Returns**: Integer (1 on success)

---

#### `web-get-attribute`
**Syntax**: `(web-get-attribute element name buffer size)`

Reads an attribute value into a buffer. Returns the string length.

**Returns**: Integer (string length)

---

#### `web-set-text-content`
**Syntax**: `(web-set-text-content element text)`

Sets the text content of an element (safe alternative to innerHTML).

**Examples**:
```scheme
(web-set-text-content btn "Click me!")
```

**Returns**: Integer (1 on success)

---

#### `web-set-inner-html`
**Syntax**: `(web-set-inner-html element html)`

Sets the innerHTML of an element. Use with caution (XSS risk with untrusted input).

**Returns**: Integer (1 on success)

---

### CSS Classes and Styles

#### `web-add-class`
**Syntax**: `(web-add-class element class-name)`

Adds a CSS class to an element.

**Returns**: Integer (1 on success)

---

#### `web-remove-class`
**Syntax**: `(web-remove-class element class-name)`

Removes a CSS class from an element.

**Returns**: Integer (1 on success)

---

#### `web-toggle-class`
**Syntax**: `(web-toggle-class element class-name)`

Toggles a CSS class on an element.

**Returns**: Integer (1 if class was added, 0 if removed)

---

#### `web-set-style`
**Syntax**: `(web-set-style element property value)`

Sets an inline CSS style property.

**Examples**:
```scheme
(web-set-style div "backgroundColor" "red")
(web-set-style div "fontSize" "24px")
```

**Returns**: Integer (1 on success)

---

### Events

#### `web-add-event-listener`
**Syntax**: `(web-add-event-listener element event-type callback)`

Registers an event listener. The callback receives an event handle.

**Examples**:
```scheme
(web-add-event-listener btn "click" on-click)
```

**Returns**: Integer (callback ID for later removal)

---

#### `web-remove-event-listener`
**Syntax**: `(web-remove-event-listener callback-id)`

Removes a previously registered event listener by its callback ID.

**Returns**: Integer (1 on success)

---

Event inspection functions: `web-event-prevent-default`, `web-event-stop-propagation`, `web-event-get-target`, `web-event-get-key`, `web-event-get-key-code`, `web-event-get-mouse-x`, `web-event-get-mouse-y`.

---

### Timers and Animation

#### `web-set-timeout`
**Syntax**: `(web-set-timeout callback ms)`

Schedules a callback to execute after `ms` milliseconds.

**Returns**: Integer (timer ID)

---

#### `web-set-interval`
**Syntax**: `(web-set-interval callback ms)`

Schedules a callback to execute repeatedly every `ms` milliseconds.

**Returns**: Integer (timer ID)

---

#### `web-clear-timeout`
**Syntax**: `(web-clear-timeout timer-id)`

Cancels a timeout.

---

#### `web-clear-interval`
**Syntax**: `(web-clear-interval timer-id)`

Cancels an interval.

---

#### `web-request-animation-frame`
**Syntax**: `(web-request-animation-frame callback)`

Requests the browser to call `callback` before the next repaint (typically 60Hz).

**Returns**: Integer (request ID)

---

### Canvas 2D

#### `web-get-context-2d`
**Syntax**: `(web-get-context-2d canvas-element)`

Obtains a 2D rendering context from a canvas element.

**Returns**: Integer (context handle)

---

#### `web-canvas-fill-rect`
**Syntax**: `(web-canvas-fill-rect ctx x y width height)`

Fills a rectangle on the canvas.

**Examples**:
```scheme
(define canvas (web-create-element "canvas"))
(define ctx (web-get-context-2d canvas))
(web-canvas-fill-style ctx "#FF0000")
(web-canvas-fill-rect ctx 10.0 10.0 100.0 50.0)
```

---

Drawing functions: `web-canvas-stroke-rect`, `web-canvas-clear-rect`, `web-canvas-fill-style`, `web-canvas-stroke-style`, `web-canvas-line-width`, `web-canvas-begin-path`, `web-canvas-close-path`, `web-canvas-move-to`, `web-canvas-line-to`, `web-canvas-arc`, `web-canvas-fill`, `web-canvas-stroke`, `web-canvas-fill-text`, `web-canvas-font`.

Transform functions: `web-canvas-save`, `web-canvas-restore`, `web-canvas-translate`, `web-canvas-rotate`, `web-canvas-scale`.

---

### Storage

#### `web-storage-set`
**Syntax**: `(web-storage-set key value)`

Stores a key-value pair in localStorage.

**Returns**: Integer (1 on success)

---

#### `web-storage-get`
**Syntax**: `(web-storage-get key buffer size)`

Reads a value from localStorage into a buffer.

**Returns**: Integer (string length)

---

#### `web-storage-remove`
**Syntax**: `(web-storage-remove key)`

Removes a key from localStorage.

**Returns**: Integer (1 on success)

---

#### `web-storage-clear`
**Syntax**: `(web-storage-clear)`

Clears all localStorage entries.

**Returns**: Integer (1 on success)

---

### Fetch

#### `web-fetch`
**Syntax**: `(web-fetch url method body)`

Performs an HTTP fetch request. Returns a promise handle for async resolution.

**Examples**:
```scheme
(web-fetch "https://api.example.com/data" "GET" "")
(web-fetch "https://api.example.com/post" "POST" "{\"key\":\"value\"}")
```

**Returns**: Integer (promise handle)

---

### Console

#### `web-console-log`
**Syntax**: `(web-console-log message)`

Logs a message to the browser console.

---

#### `web-console-warn`
**Syntax**: `(web-console-warn message)`

Logs a warning to the browser console.

---

#### `web-console-error`
**Syntax**: `(web-console-error message)`

Logs an error to the browser console.

---

### Window

Functions for browser window interaction: `web-alert`, `web-confirm`, `web-prompt`, `web-get-window-width`, `web-get-window-height`, `web-get-scroll-x`, `web-get-scroll-y`, `web-scroll-to`.

Location management: `web-get-href`, `web-set-href`, `web-get-hash`, `web-set-hash`.

---

### Handle Management

#### `web-release-handle`
**Syntax**: `(web-release-handle handle)`

Releases a JavaScript object handle when it is no longer needed. Important for preventing memory leaks in long-running applications.

---

## Thread Pool & Promise Builtins

Low-level primitives for inspecting the work-stealing thread pool and managing promises/futures. The thread pool is initialized automatically with hardware-aware sizing.

### Thread Pool Inspection

#### `thread-pool-info`
**Syntax**: `(thread-pool-info)`

Returns the number of worker threads in the thread pool.

**Examples**:
```scheme
(thread-pool-info)  ; => 8 (on an 8-core machine)
```

**Returns**: Integer (thread count)

---

#### `thread-pool-size`
**Syntax**: `(thread-pool-size)`

Alias for `thread-pool-info`. Returns the number of worker threads.

**Returns**: Integer (thread count)

---

#### `thread-pool-stats`
**Syntax**: `(thread-pool-stats)`

Prints diagnostic information about the thread pool state (active tasks, queue lengths, work-stealing statistics) to standard output.

**Returns**: Unspecified (side effect: prints to stdout)

---

### Promises and Futures

#### `make-promise`
**Syntax**: `(make-promise value)`

Creates an already-forced promise containing `value`. This is useful for wrapping a computed value in the promise protocol so that `force` can be applied uniformly.

**Examples**:
```scheme
(define p (make-promise 42))
(force p)  ; => 42
```

**Returns**: Promise

---

#### `promise?`
**Syntax**: `(promise? obj)`

Returns `#t` if `obj` is a promise (created by `delay` or `make-promise`).

**Examples**:
```scheme
(promise? (delay (+ 1 2)))   ; => #t
(promise? 42)                ; => #f
```

**Returns**: Boolean

---

#### `future-ready?`
**Syntax**: `(future-ready? f)`

Returns `#t` if the future `f` has completed its computation and the result is available without blocking.

**Examples**:
```scheme
(define f (future (begin (sleep 100) 42)))
(future-ready? f)  ; => #f (computation still running)
;; ... later ...
(future-ready? f)  ; => #t
(force f)           ; => 42
```

**Returns**: Boolean

---

## Command-Line Tools

### `eshkol-run` -- Compiler and JIT Runner

The primary compiler driver. Compiles `.esk` source files to native binaries, object files, or WebAssembly, and supports JIT evaluation.

**Usage**:
```
eshkol-run [options] <input.esk|input.o> [input.esk|input.o ...]
eshkol-run -e '<expression>'     (JIT evaluate expression)
eshkol-run -r <file.esk>         (JIT run file)
```

**Flags**:

| Flag | Short | Description |
|------|-------|-------------|
| `--help` | `-h` | Print help message |
| `--debug` | `-d` | Emit debugging information inside the program |
| `--dump-ast` | `-a` | Dump the AST to a `.ast` file |
| `--dump-ir` | `-i` | Dump LLVM IR to a `.ll` file |
| `--output FILE` | `-o` | Output path for compiled binary |
| `--compile-only` | `-c` | Compile to intermediate object file (`.o`) only |
| `--shared-lib` | `-s` | Compile as shared library (LinkOnceODR linkage) |
| `--wasm` | `-w` | Compile to WebAssembly (`.wasm`) format |
| `--lib LIB` | `-l` | Link an external shared library |
| `--lib-path DIR` | `-L` | Add directory to library search path |
| `--no-stdlib` | `-n` | Do not auto-load the standard library |
| `--eval EXPR` | `-e` | JIT evaluate an expression and print the result |
| `--run` | `-r` | JIT run a file (interpret without compiling to disk) |
| `--debug-info` | `-g` | Emit DWARF debug info (enables lldb/gdb) |
| `--optimize N` | `-O` | Set LLVM optimization level (0=none, 1=basic, 2=full, 3=aggressive) |
| `--strict-types` | | Type errors are fatal (default: gradual/warnings) |
| `--unsafe` | | Skip all runtime type checks |

**Examples**:
```scheme
;; Compile to binary
;; $ eshkol-run main.esk -o myprogram
;; $ ./myprogram

;; JIT evaluate an expression
;; $ eshkol-run -e '(+ 1 2 3)'
;; 6

;; Compile to WebAssembly
;; $ eshkol-run --wasm app.esk -o app.wasm

;; Dump LLVM IR for debugging
;; $ eshkol-run --dump-ir program.esk -o program
```

---

### `eshkol-repl` -- Interactive REPL

A visual live coding environment with JIT compilation, tab completion, and crash recovery.

**Features**:
- JIT compilation via LLVM ORC (expressions compiled and executed immediately)
- Tab completion for all builtins (555+) and user-defined symbols
- Readline integration with persistent history (`~/.eshkol_history`)
- Crash recovery: segfaults during JIT execution are caught and reported without terminating the session
- Multi-line input with automatic bracket balancing

**REPL Commands**:

| Command | Short | Description |
|---------|-------|-------------|
| `:help` | `:h` | Show help message |
| `:quit` | `:q` | Exit the REPL |
| `:cancel` | `:c` | Cancel multi-line input |
| `:clear` | | Clear the screen |
| `:env` | `:e` | Show defined symbols in environment |
| `:type EXPR` | `:t` | Show type of an expression |
| `:doc NAME` | `:d` | Show documentation for a function |
| `:ast EXPR` | | Show AST for an expression |
| `:time EXPR` | | Time execution of an expression |
| `:load FILE` | `:l` | Load and execute a file |
| `:reload` | `:r` | Reload the last loaded file |
| `:stdlib` | | Load the standard library |
| `:reset` | | Reset the REPL state |
| `:history [N]` | | Show command history |
| `:version` | `:v` | Show version information |
| `:examples` | | Show example expressions |

---

### `eshkol-pkg` -- Package Manager

Manages Eshkol project dependencies and builds using `eshkol.toml` manifests and a git-based package registry.

**Usage**: `eshkol-pkg <command> [args]`

**Commands**:

| Command | Description |
|---------|-------------|
| `init` | Create a new `eshkol.toml` project in the current directory |
| `build` | Compile the current project (reads entry point from manifest) |
| `run` | Build and run the project |
| `install` | Install all dependencies from `eshkol.toml` |
| `add <pkg> [version]` | Add a dependency to the manifest |
| `remove <pkg>` | Remove a dependency from the manifest |
| `search <query>` | Search the package registry (local cache) |
| `publish` | Display instructions for publishing to the registry |
| `clean` | Remove the `build/` directory |

**Manifest Format** (`eshkol.toml`):
```toml
[package]
name = "my-project"
version = "0.1.0"
description = "A scientific computing project"
author = "Author Name"
license = "MIT"
entry = "src/main.esk"
sources = ["src/*.esk"]

[dependencies]
math-utils = "1.0.0"
```

**Environment Variables**:
- `ESHKOL_COMPILER` -- Path to the `eshkol-run` binary (default: `eshkol-run`)
- `ESHKOL_REGISTRY` -- Package registry URL

---

### `eshkol-lsp` -- Language Server

Provides IDE integration via the Language Server Protocol (LSP) over stdin/stdout using JSON-RPC 2.0 transport.

**Capabilities**:

| Feature | LSP Method | Description |
|---------|-----------|-------------|
| Diagnostics | `textDocument/publishDiagnostics` | Parse error reporting with line/column |
| Completion | `textDocument/completion` | Keyword and builtin symbol completion |
| Hover | `textDocument/hover` | Type information and documentation on hover |
| Go to Definition | `textDocument/definition` | Navigate to symbol definitions |
| Document Sync | `didOpen`, `didChange`, `didClose` | Full document synchronization |

**Editor Integration**: Configure your editor to launch `eshkol-lsp` as the language server for `.esk` files. Example for VS Code `settings.json`:
```json
{
  "eshkol.lsp.path": "/usr/local/bin/eshkol-lsp"
}
```

---

## Data Loading

The dataloader abstracts batch iteration over tensor datasets, handling index management,
batch slicing, and optional shuffling. This decouples data access logic from the training
loop, enabling clean separation of concerns.

### `make-dataloader`

**Syntax:** `(make-dataloader data batch-size [shuffle])` → Dataloader

Creates a batch iterator over the tensor `data`. The dataloader maintains an internal
position and returns consecutive slices of `batch-size` elements on each call to
`dataloader-next`.

**Parameters:**
- `data` — the dataset tensor
- `batch-size` — number of samples per batch (integer)
- `shuffle` — if `#t`, randomly permutes the data indices. Default: `#f`

**Implementation:** `makeDataloader` in [lib/backend/tensor_dataloader_codegen.cpp](../lib/backend/tensor_dataloader_codegen.cpp).

### `dataloader-next`

**Syntax:** `(dataloader-next loader)` → Tensor | null

Returns the next batch tensor, or null when all batches have been consumed. Advances the
internal position by `batch-size`. The last batch may be smaller than `batch-size` if the
dataset size is not evenly divisible.

**Implementation:** `dataloaderNext` in [lib/backend/tensor_dataloader_codegen.cpp](../lib/backend/tensor_dataloader_codegen.cpp).

### `dataloader-reset!`

**Syntax:** `(dataloader-reset! loader)` → Dataloader

Resets the internal position to 0. If the dataloader was created with `shuffle = #t`,
re-permutes the indices. Returns the loader for chaining.

**Implementation:** `dataloaderReset` in [lib/backend/tensor_dataloader_codegen.cpp](../lib/backend/tensor_dataloader_codegen.cpp).

### `dataloader-length`

**Syntax:** `(dataloader-length loader)` → Integer

Returns the number of batches: ⌈n / batch-size⌉.

**Implementation:** `dataloaderLength` in [lib/backend/tensor_dataloader_codegen.cpp](../lib/backend/tensor_dataloader_codegen.cpp).

### `dataloader-has-next?`

**Syntax:** `(dataloader-has-next? loader)` → Boolean

Returns `#t` if more batches remain, `#f` otherwise.

**Implementation:** `dataloaderHasNext` in [lib/backend/tensor_dataloader_codegen.cpp](../lib/backend/tensor_dataloader_codegen.cpp).

### `train-test-split`

**Syntax:** `(train-test-split data labels ratio [shuffle])` → Vector of 4 Tensors

Splits data and labels into training and test sets. `ratio` ∈ [0, 1] specifies the training
fraction (e.g., 0.8 = 80% train, 20% test). If `shuffle` is `#t`, indices are permuted
before splitting.

**Returns:** A vector of four tensors: `(train-data train-labels test-data test-labels)`.

**Implementation:** `trainTestSplit` in [lib/backend/tensor_dataloader_codegen.cpp](../lib/backend/tensor_dataloader_codegen.cpp).

### Dataloader Example

```scheme
;; Create shuffled dataloader with batch size 32
(define loader (make-dataloader training-data 32 #t))

(do ((epoch 0 (+ epoch 1))) ((= epoch 100))
  (dataloader-reset! loader)             ; re-shuffle each epoch
  (do () ((not (dataloader-has-next? loader)))
    (let ((batch (dataloader-next loader)))
      ;; forward pass, loss, backward, optimizer step on batch
      )))

;; Train/test split
(define split (train-test-split features labels 0.8 #t))
(define train-x (vector-ref split 0))
(define train-y (vector-ref split 1))
(define test-x (vector-ref split 2))
(define test-y (vector-ref split 3))
```

---

## HoTT Sum Types

Sum types (coproducts) implement the type-theoretic disjoint union A + B. A value of type
A + B is either a value of type A (injected left) or a value of type B (injected right),
together with a tag distinguishing the two cases. In Homotopy Type Theory (HoTT), coproducts
satisfy the universal property: given f: A → C and g: B → C, there exists a unique
h: A + B → C implementing case analysis.

Sum types are represented as tagged pairs `(tag . value)` where tag is 0 (left) or 1 (right).

**Implementation:** dispatched from `LLVMCodeGenerator::codegenCall` in [lib/backend/llvm_codegen.cpp](../lib/backend/llvm_codegen.cpp);
`inject-left` / `inject-right` lower through `codegenSumInject`.

### `inject-left`

**Syntax:** `(inject-left value)` → Sum

Wraps `value` as the left variant of a sum type. Returns a tagged pair `(0 . value)`.

```scheme
(inject-left 42)         ; => (0 . 42) — left-injected integer
(inject-left "hello")    ; => (0 . "hello") — left-injected string
```

### `inject-right`

**Syntax:** `(inject-right value)` → Sum

Wraps `value` as the right variant. Returns a tagged pair `(1 . value)`.

```scheme
(inject-right #t)        ; => (1 . #t) — right-injected boolean
```

### `sum-tag`

**Syntax:** `(sum-tag sum)` → Integer

Extracts the tag: 0 for left, 1 for right.

```scheme
(sum-tag (inject-left 42))    ; => 0
(sum-tag (inject-right #t))   ; => 1
```

### `sum-value`

**Syntax:** `(sum-value sum)` → Value

Extracts the inner value from either variant.

```scheme
(sum-value (inject-left 42))    ; => 42
(sum-value (inject-right #t))   ; => #t
```

### `left?`, `right?`

**Syntax:** `(left? sum)` → Boolean, `(right? sum)` → Boolean

Type predicates for sum variants.

```scheme
(left? (inject-left 42))     ; => #t
(right? (inject-left 42))    ; => #f
(right? (inject-right #t))   ; => #t
```

### Case Analysis Pattern

```scheme
;; Type-safe case analysis (the universal property of coproducts)
(define (case-sum s on-left on-right)
  (if (left? s)
    (on-left (sum-value s))
    (on-right (sum-value s))))

;; Example: Result type (Success | Error)
(define (parse-int str)
  (let ((n (string->number str)))
    (if n (inject-left n) (inject-right "parse error"))))

(case-sum (parse-int "42")
  (lambda (n) (display n))          ; success path
  (lambda (err) (display err)))     ; error path
```

---

## Record Types (R7RS)

Eshkol implements R7RS `define-record-type` as a syntactic abstraction over vectors. Record
types provide named constructors, field accessors, optional mutators, and type predicates —
enabling structured data with a clear, type-safe interface.

### Syntax

```scheme
(define-record-type <name>
  (<constructor> <field-name> ...)
  <predicate>
  (<field-name> <accessor> [<mutator>]) ...)
```

### Semantics

The `define-record-type` form defines:

1. **Constructor** — a procedure that creates a new record instance from the specified fields.
2. **Predicate** — a procedure that tests whether a value is an instance of this record type.
3. **Accessors** — procedures that extract field values from a record instance.
4. **Mutators** (optional) — procedures that modify field values in-place.

**Implementation:** Parser transforms `define-record-type` into vector operations
(`ESHKOL_DEFINE_RECORD_TYPE_OP` branch in
[lib/frontend/parser.cpp](../lib/frontend/parser.cpp)). The constructor creates a tagged
vector with a unique type identifier, the predicate checks this identifier, and
accessors/mutators map to `vector-ref` / `vector-set!`.

### Examples

```scheme
;; Define a 2D point record
(define-record-type point
  (make-point x y)
  point?
  (x point-x set-point-x!)
  (y point-y set-point-y!))

(define p (make-point 3.0 4.0))
(point? p)              ; => #t
(point-x p)             ; => 3.0
(point-y p)             ; => 4.0
(set-point-y! p 5.0)
(point-y p)             ; => 5.0

;; Immutable record (no mutators)
(define-record-type color
  (make-color r g b)
  color?
  (r color-r)
  (g color-g)
  (b color-b))

(define red (make-color 255 0 0))
(color-r red)           ; => 255

;; Records as function arguments
(define (distance p1 p2)
  (sqrt (+ (expt (- (point-x p1) (point-x p2)) 2)
           (expt (- (point-y p1) (point-y p2)) 2))))

(distance (make-point 0 0) (make-point 3 4))   ; => 5.0
```

---

## ML Standard Library

The ML standard library provides high-level optimization algorithms and utility functions
implemented in Eshkol. These are stdlib functions (compiled to `stdlib.o`), distinct from
the compiler-level ML builtins documented in [Machine Learning & Neural Networks](#machine-learning--neural-networks).

### Optimization Algorithms

**Module:** `lib/ml/optimization.esk`
**Import:** `(require ml.optimization)` or `(require stdlib)`

These functions use Eshkol's automatic differentiation system (`gradient`) internally to
compute derivatives, making them self-contained optimization routines.

#### `gradient-descent`

**Syntax:** `(gradient-descent f x0 [lr max-iter tol])` → Tensor

Vanilla gradient descent: x_{t+1} = x_t - η · ∇f(x_t).

**Parameters:**
- `f` — objective function: Tensor → Number (scalar-valued)
- `x0` — initial parameter tensor
- `lr` — learning rate (default: 0.01)
- `max-iter` — maximum iterations (default: 1000)
- `tol` — convergence tolerance on |∇f| (default: 1e-8)

**Returns:** Optimized parameter tensor.

#### `adam`

**Syntax:** `(adam f x0 [lr max-iter tol β₁ β₂ ε])` → Tensor

High-level Adam optimizer wrapper. Maintains first and second moment estimates internally.

**Defaults:** lr = 0.001, max-iter = 1000, tol = 1e-8, β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

#### `l-bfgs`

**Syntax:** `(l-bfgs f x0 [max-iter tol m])` → Tensor

Limited-memory BFGS quasi-Newton method. Approximates the inverse Hessian using the most
recent `m` gradient pairs (default m = 10), enabling super-linear convergence without
storing the full n×n Hessian.

**Algorithm:** Two-loop recursion (Nocedal, 1980) with backtracking line search satisfying
the Wolfe conditions.

**Use case:** Large-scale unconstrained optimization where second-order information
significantly accelerates convergence.

#### `conjugate-gradient`

**Syntax:** `(conjugate-gradient f x0 [max-iter tol])` → Tensor

Fletcher-Reeves conjugate gradient with automatic restarts every n iterations (where n is
the parameter dimensionality). Converges in at most n iterations for quadratic objectives.

#### `line-search`

**Syntax:** `(line-search f x d grad [alpha c rho max-iter])` → Number

Backtracking line search satisfying the Armijo sufficient decrease condition:

```
f(x + α·d) ≤ f(x) + c · α · ∇f(x)^T · d
```

**Parameters:**
- `f` — objective function
- `x` — current position
- `d` — search direction
- `grad` — gradient at x
- `alpha` — initial step size (default: 1.0)
- `c` — Armijo parameter (default: 1e-4)
- `rho` — backtracking factor (default: 0.9)
- `max-iter` — maximum backtracking steps (default: 100)

**Returns:** Step size α satisfying the Armijo condition.

#### Optimization Example

```scheme
(require ml.optimization)

;; Rosenbrock function — classic optimization test
(define (rosenbrock v)
  (let ((x (tensor-ref v 0)) (y (tensor-ref v 1)))
    (+ (expt (- 1.0 x) 2) (* 100.0 (expt (- y (* x x)) 2)))))

;; Optimize from (0, 0) — minimum is at (1, 1)
(adam rosenbrock #(0.0 0.0))              ; => ~#(1.0 1.0)
(l-bfgs rosenbrock #(0.0 0.0))           ; => ~#(1.0 1.0) (faster)
(gradient-descent rosenbrock #(0.0 0.0) 0.0001 10000)  ; slower but simplest
```

### Activation & Normalization Utilities

**Module:** `lib/ml/activations.esk`
**Import:** `(require ml)` or `(require ml.activations)`

Scalar and tensor utility functions for activation and normalization. These are stdlib
wrappers — the compiler also provides builtin versions (`relu`, `sigmoid`, etc.) as
SIMD-accelerated operations. Use the builtins for performance; use these stdlib versions
for composability and custom pipelines.

| Function | Signature | Description |
|----------|-----------|-------------|
| `(relu-scalar x)` | Number → Number | max(0, x) — scalar ReLU |
| `(sigmoid-scalar x)` | Number → Number | 1/(1+exp(-x)) — scalar sigmoid |
| `(tanh-scalar x)` | Number → Number | Hyperbolic tangent (alias) |
| `(softplus-scalar x)` | Number → Number | log(1+exp(x)), numerically stable |
| `(silu tensor)` | Tensor → Tensor | SiLU: x · sigmoid(x) |
| `(swish tensor beta)` | Tensor × Number → Tensor | Swish with configurable β |
| `(mish tensor)` | Tensor → Tensor | x · tanh(softplus(x)) |
| `(normalize-minmax tensor)` | Tensor → Tensor | Scale to [0, 1]: (x - min) / (max - min) |
| `(normalize-zscore tensor)` | Tensor → Tensor | Z-score: (x - μ) / σ |
| `(clip tensor min max)` | Tensor × Number × Number → Tensor | Clamp values to [min, max] |

```scheme
(require ml.activations)
(normalize-zscore #(10.0 20.0 30.0 40.0 50.0))
;; => #(-1.414 -0.707 0.0 0.707 1.414)
(clip #(1.0 5.0 10.0) 2.0 8.0)
;; => #(2.0 5.0 8.0)
```

---

## Implementation Statistics

**Codebase Size**: ~232,000 lines of production C++
**Main Backend**: [llvm_codegen.cpp](lib/backend/llvm_codegen.cpp) — 34,928 lines
**Tensor Codegen**: [tensor_codegen.cpp](lib/backend/tensor_codegen.cpp) — 20,000+ lines
**Compiler Modules**: 21 specialized code generators
**Test Suite**: 37 suites, 528 self-reported tests
**Verified Operations**: 555+ builtins, 300+ standard library functions

---

## See Also

- [**Architecture Guide**](ESHKOL_V1_ARCHITECTURE.md) — Complete system architecture
- [**Known Issues**](KNOWN_ISSUES.md) — Current limitations and planned features

---

**Copyright** © 2025-2026 tsotchke
**License**: MIT
