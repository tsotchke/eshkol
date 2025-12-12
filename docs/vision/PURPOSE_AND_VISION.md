# Eshkol: Purpose and Vision

## Mission Statement

Eshkol v1.0-architecture delivers a **production-ready programming language** for gradient-based optimization, neural network development, and scientific computing. Built on a sophisticated LLVM compiler infrastructure, Eshkol combines Scheme's homoiconicity with compiler-integrated automatic differentiation and deterministic arena memory management.

## What Eshkol Actually Is (v1.0)

Eshkol is a **compiled Scheme dialect** that excels at:

1. **Gradient-Based Optimization**
   - Forward-mode automatic differentiation (dual numbers)
   - Reverse-mode automatic differentiation (computational graphs)
   - Nested gradients up to 32 levels deep
   - Vector calculus operators: gradient, jacobian, hessian, divergence, curl, laplacian

2. **Neural Network Development**
   - Automatic differentiation of arbitrary functions
   - Tensor operations (vectors, matrices, N-dimensional arrays)
   - Backpropagation through computational graphs
   - Mixed-type data structures for heterogeneous models

3. **Scientific Computing**
   - Linear algebra algorithms (LU decomposition, Gauss-Jordan, matrix solvers)
   - Numerical integration (Simpson's rule)
   - Root finding (Newton-Raphson)
   - Statistical computations (variance, covariance)
   - N-dimensional tensor operations

4. **Deterministic Memory Management**
   - Arena allocation with scope-based cleanup
   - Ownership tracking (owned, moved, borrowed states)
   - Escape analysis (stack/region/shared allocation decisions)
   - No garbage collection - predictable performance

5. **Interactive Development**
   - LLVM ORC JIT compilation in REPL
   - Persistent state across evaluations
   - Homoiconic lambda display (code as data)

## Core Principles

### 1. Homoiconicity with LLVM Performance

Eshkol preserves the homoiconicity of Scheme's code-as-data property while compiling to native code via LLVM. Every lambda stores its source S-expression in a 32-byte closure structure, enabling runtime introspection without sacrificing execution speed.

**Implementation:**
```c
struct eshkol_closure {
    uint64_t func_ptr;              // Compiled LLVM function
    eshkol_closure_env_t* env;      // Captured variables
    uint64_t sexpr_ptr;             // S-expression for display
    // ... type metadata
}
```

**Example:**
```scheme
(define square (lambda (x) (* x x)))
(display square)  ; => (lambda (x) (* x x))
; S-expression retrieved from closure, not decompiled
```

### 2. Gradual Typing for Scientific Computing

HoTT-inspired bidirectional type checker with optional annotations:
- **Dynamic typing** for prototyping and exploration
- **Static annotations** for production code and optimization
- **Type inference** reduces annotation burden
- **Warnings, not errors** - gradual refinement path

**Example:**
```scheme
; No annotations - dynamic typing
(define (f x) (* x x))

; Partial annotations - type-guided optimization
(define (g (x : real)) (* x x))

; Full annotations - maximum optimization
(define (h (x : real)) : real (* x x))
```

### 3. Compiler-Integrated Automatic Differentiation

Unlike library-based AD (JAX, PyTorch), Eshkol integrates differentiation at the **compiler level**, operating simultaneously on AST, runtime values, and LLVM IR.

**Three AD Modes:**
1. **Symbolic** - AST transformation during compilation
2. **Forward** - Dual numbers at runtime
3. **Reverse** - Computational graph with tape stack

**Nested Gradients:**
```scheme
(define (f x)
  (gradient 
    (lambda (y) (* x y y))
    (vector 1.0)))

(gradient f (vector 2.0))
; Computes ∂/∂x[∂/∂y(xy²)] = y² at inner point
; Tape stack tracks nesting depth
```

### 4. Deterministic Memory Management

OALR (Ownership-Aware Lexical Regions) provides:
- **Arena allocation** - bump-pointer in lexical regions
- **Ownership tracking** - compile-time analysis prevents use-after-move
- **Escape analysis** - automatic stack/region/shared decisions
- **No GC pauses** - deterministic performance for real-time systems

**Memory Allocation Decision Tree:**
```
Escape Analysis Result:
├─ NO_ESCAPE → Stack allocation (fastest)
├─ RETURN_ESCAPE → Region allocation (local lifetime)
└─ CLOSURE_ESCAPE/GLOBAL_ESCAPE → Shared allocation (reference counted)
```

### 5. Production Compiler Infrastructure

**LLVM Backend Architecture:**
- Modular design with 15 specialized codegen components
- TypedValue carries LLVM value + runtime type + HoTT type
- Global arena shared across all functions
- REPL mode vs standalone executable modes
- Module system with dependency resolution

**Parser Features:**
- S-expression with type annotations (`:`, `->`)
- HoTT type expressions (arrow, forall, container types)
- Pattern matching (literal, variable, wildcard, cons, list, predicate, or)
- Macro system (define-syntax with syntax-rules)
- Internal defines transformed to letrec

## Target Domains (v1.0-architecture)

### Currently Supported

**1. Machine Learning & Neural Networks**
```scheme
; Actual working example from test suite
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

(define (mse-loss pred target)
  (let ((diff (- pred target)))
    (* 0.5 (* diff diff))))

; Gradient for backpropagation
(define grad-loss (gradient mse-loss))
```

**2. Scientific Algorithm Development**
```scheme
(require math.esk)

; Matrix inverse using Gauss-Jordan (implemented in Eshkol)
(define M (matrix 3 3  2 1 1
                       1 2 1
                       1 1 2))
(define M-inv (inv M 3))

; Linear system solver
(define b #(6.0 5.0 4.0))
(define x (solve M b 3))
```

**3. Gradient-Based Optimization**
```scheme
; Rosenbrock function minimization
(define (rosenbrock v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (+ (* 100.0 (* (- y (* x x)) (- y (* x x))))
       (* (- 1.0 x) (- 1.0 x)))))

(define point #(0.0 0.0))
(define grad (gradient rosenbrock point))
; Use grad for optimization step
```

**4. Interactive Scientific Computing**
```scheme
$ eshkol-repl
> (require stdlib)
> (define data (iota 100))
> (define squared (map (lambda (x) (* x x)) data))
> (fold + 0 squared)
328350
```

### Post-v1.0 Plans

See [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for:
- GPU acceleration (CUDA, Metal, Vulkan)
- Multi-threading and parallelism
- Distributed training
- Quantum computing extensions (actual qubits, not just RNG)
- Embedded ML deployment

## Design Philosophy

### Performance Without Compromise

**LLVM-Native Compilation:**
- Direct LLVM IR generation (not C transpilation)
- LLVM optimization passes applied
- Native code generation for target architecture
- Comparable performance to hand-optimized C++

**Memory Efficiency:**
- Arena allocation eliminates GC overhead
- 16-byte tagged values for compact representation
- Cons cells store complete tagged values (32 bytes) - no indirection
- Object headers add 8 bytes - clear ownership model

### Safety Through Analysis

**Compile-Time Checks:**
- Ownership analysis prevents use-after-move
- Escape analysis optimizes allocation strategy
- Type checking with inference reduces annotations
- Module dependency cycle detection

**Runtime Safety:**
- Tagged values prevent type confusion
- Bounds checking on tensor operations
- Exception handling with guard/raise
- Deep equality comparison for structural data

### Extensibility Through Homoiconicity

**Lambda Registry:**
- Maps function pointers to S-expressions
- Enables runtime code introspection
- Supports metaprogramming patterns
- Foundation for self-modifying systems

**Macro System:**
- Hygienic macros (define-syntax with syntax-rules)
- Pattern matching with ellipsis repetition
- Compile-time code transformation
- Type-aware macro expansion

## Historical Context

Eshkol builds on:
- **Scheme R7RS** - Lexical scoping, first-class functions, homoiconicity
- **LLVM Project** - Modern compiler infrastructure
- **HoTT** - Homotopy Type Theory for gradual typing
- **Region-based memory** - Academic research on deterministic allocation

The name "Eshkol" (אֶשְׁכּוֹל) means "cluster" in Hebrew, reflecting the language's integration of multiple computational paradigms.

## Implementation Principles

### 1. Modular Backend Architecture

15 specialized codegen modules instead of monolithic compiler:
- `TaggedValueCodegen` - Pack/unpack 16-byte values
- `AutodiffCodegen` - Dual numbers and AD graphs
- `FunctionCodegen` - Closures with environment
- `ArithmeticCodegen` - Polymorphic dispatch (int64/double/dual/tensor)
- `ControlFlowCodegen` - Short-circuit logic, pattern matching
- `CollectionCodegen` - Cons cells, vectors
- `TensorCodegen` - N-dimensional arrays
- `HashCodegen` - FNV-1a hash tables
- `StringIOCodegen` - Strings, ports, file I/O
- `TailCallCodegen` - Loop transformation
- Plus 5 more

### 2. Tagged Value Polymorphism

Runtime values carry complete type information:
```c
struct eshkol_tagged_value {
    uint8_t type;        // 0-255 type encoding
    uint8_t flags;       // Exactness, special markers
    uint16_t reserved;
    union {
        int64_t int_val;     // Integers, bools, chars
        double double_val;   // Floating-point
        uint64_t ptr_val;    // Heap pointers
    } data;
}
```

**Arithmetic dispatch** checks type at runtime, invokes specialized operations:
- `int64 + int64` → native add
- `double + double` → native fadd
- `dual + dual` → dual arithmetic
- `tensor + tensor` → element-wise operation

### 3. Closure Compilation Strategy

**Static capture analysis:**
1. Scan lambda body for free variables
2. Determine captures from enclosing scope
3. Generate function with captures as extra parameters
4. Allocate closure with environment storage

**Calling convention:**
```c
tagged_value func(param1, param2, ..., capture1, capture2, ...)
```

**Environment structure:**
```c
struct eshkol_closure_env {
    size_t num_captures;  // Packed: count | (fixed_params << 16) | (is_variadic << 63)
    eshkol_tagged_value_t captures[];  // Flexible array
}
```

## The Path Forward

Eshkol v1.0-architecture establishes a **solid foundation** for advanced computational programming:

**Current State (v1.0):**
- ✅ Production compiler with comprehensive test suite
- ✅ Automatic differentiation for gradient-based optimization
- ✅ Neural network training capability
- ✅ Linear algebra and numerical algorithms
- ✅ Interactive REPL with JIT compilation
- ✅ Module system with visibility control
- ✅ Exception handling
- ✅ Hash tables and data structures

**Immediate Next Steps (Post-v1.0):**
- GPU acceleration for tensor operations
- Quantum computing integration
- Multi-threading primitives
- Distributed computing support
- Performance profiling tools
- Expanded standard library

**Medium-Term Vision (v1.5+):**
- Advanced type system features (dependent types, linear types)
- Hardware-specific optimizations
- Domain-specific embedded languages

See [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for detailed development plans.

## Why Eshkol Matters

**For ML Researchers:**
- Write differentiable algorithms directly in Scheme syntax
- Compose gradient computations naturally
- No framework boundary - AD is the language

**For Scientific Programmers:**
- Express mathematical algorithms clearly
- LLVM-native performance without C++ complexity
- Deterministic memory for real-time constraints

**For Language Researchers:**
- Study compiler-integrated AD implementation
- Explore homoiconic closures in native compilation
- Investigate gradual typing with HoTT foundations

---

*Eshkol v1.0-architecture represents a genuine achievement in programming language implementation: a complete, working compiler that successfully integrates automatic differentiation, deterministic memory management, and homoiconic closures while maintaining compatibility with Scheme's elegant semantics.*
