# Eshkol v1.1-accelerate Feature Matrix

**Status Key**: ✅ Production | 🚧 In Progress | 📋 Planned | ❌ Not Planned

This matrix documents all implemented and planned features for the Eshkol language ecosystem. All **Production** features are code-verified with extensive test coverage (35 test suites, 438 test files).

---

## Core Language Features

| Feature | Status | Notes | Test Coverage |
|---------|--------|-------|---------------|
| **Special Forms** |
| `define` (variables) | ✅ | Global and local bindings | 50+ tests |
| `define` (functions) | ✅ | Top-level and nested | 50+ tests |
| `lambda` | ✅ | Closures with captures | 100+ tests |
| `let`, `let*`, `letrec` | ✅ | All binding forms | 80+ tests |
| Named let (iteration) | ✅ | Tail-recursive loops | 10+ tests |
| `set!` | ✅ | Mutable variables and captures | 30+ tests |
| `if`, `cond`, `case` | ✅ | All conditionals | 40+ tests |
| `begin` | ✅ | Sequence evaluation | 20+ tests |
| `and`, `or`, `not` | ✅ | Short-circuit boolean logic | 15+ tests |
| `when`, `unless` | ✅ | One-armed conditionals | 10+ tests |
| `do` loops | ✅ | Iteration with state | 5+ tests |
| `quote`, `quasiquote` | ✅ | S-expression literals | 20+ tests |
| `apply` | ✅ | Dynamic function application | 15+ tests |
| **Pattern Matching** |
| `match` | ✅ | Full pattern matching | 10+ tests |
| Variable patterns | ✅ | Binding in patterns | Verified |
| Literal patterns | ✅ | Constant matching | Verified |
| Cons patterns | ✅ | `(p1 . p2)` destructuring | Verified |
| List patterns | ✅ | `(p1 p2 ...)` matching | Verified |
| Predicate patterns | ✅ | `(? pred)` guards | Verified |
| Or-patterns | ✅ | `(or p1 p2 ...)` alternatives | Verified |
| **Closures** |
| Basic closures | ✅ | Capture environment | 50+ tests |
| Mutable captures | ✅ | `set!` on captured vars | 20+ tests |
| Nested closures | ✅ | Arbitrary depth | 15+ tests |
| Variadic closures | ✅ | Rest parameters | 10+ tests |
| Closure homoiconicity | ✅ | Display shows source code | Verified |
| **Tail Call Optimization** |
| Self-recursive TCO | ✅ | Functions calling themselves | 15+ tests |
| Mutual recursion TCO | ✅ | Functions calling each other | Trampoline-based |
| Trampoline runtime | ✅ | Non-self tail calls | 5+ tests |

---

## Type System

| Feature | Status | Notes | Implementation |
|---------|--------|-------|----------------|
| **Runtime Tagged Values** |
| Int64 | ✅ | Exact integers | 16-byte struct |
| Double | ✅ | IEEE 754 floats | 16-byte struct |
| Boolean | ✅ | #t/#f | Type tag + bit |
| Char | ✅ | Unicode codepoints | Type tag + int64 |
| String | ✅ | Heap-allocated | HEAP_PTR + header |
| Symbol | ✅ | Interned strings | HEAP_PTR + header |
| Cons/List | ✅ | Heterogeneous pairs | HEAP_PTR + header |
| Vector | ✅ | Scheme vectors | HEAP_PTR + header |
| Tensor | ✅ | N-dimensional arrays | HEAP_PTR + header |
| Closure | ✅ | Function + environment | CALLABLE + header |
| Hash Table | ✅ | Mutable maps | HEAP_PTR + header |
| **HoTT Compile-Time Types** |
| Integer, Real, Number | ✅ | Numeric hierarchy | Gradual typing |
| Boolean, Char, String | ✅ | Primitive types | Gradual typing |
| List<T>, Vector<T> | ✅ | Parameterized collections | Element type tracking |
| Tensor<T> | ✅ | Typed tensors | Element type tracking |
| Function arrows (→) | ✅ | `(→ A B)` types | Type inference |
| Dependent types | ✅ | Path types, universes | Proof erasure |
| Gradual typing | ✅ | Optional annotations | Warning-only errors |
| **Exact Arithmetic (v1.1)** |
| Bignum (arbitrary-precision int) | ✅ | Automatic overflow promotion | int64 → bignum |
| Rational numbers | ✅ | Exact fractions (num/den) | HEAP_PTR + header |
| Complex numbers | ✅ | `make-rectangular`, `make-polar` | Type tag 7 |
| `exact?`, `inexact?` | ✅ | Exactness predicates | Runtime tags |
| `exact->inexact`, `inexact->exact` | ✅ | Exactness conversion | Type conversion |
| **Type Predicates** |
| `number?`, `integer?`, `real?` | ✅ | Numeric predicates | Runtime tags |
| `string?`, `char?`, `boolean?` | ✅ | Primitive predicates | Runtime tags |
| `null?`, `pair?`, `list?` | ✅ | List predicates | Runtime tags |
| `vector?`, `procedure?` | ✅ | Compound predicates | Header subtype |
| `complex?`, `rational?` | ✅ | Extended numeric predicates | Runtime tags |

---

## Automatic Differentiation

| Feature | Status | Mode | Performance |
|---------|--------|------|-------------|
| **Symbolic Differentiation** |
| Compile-time AST transform | ✅ | Symbolic | O(1) - compile time |
| Sum rule | ✅ | Symbolic | Verified |
| Product rule | ✅ | Symbolic | Verified |
| Quotient rule | ✅ | Symbolic | Verified |
| Chain rule | ✅ | Symbolic | sin, cos, exp, log, pow, sqrt |
| Algebraic simplification | ✅ | Symbolic | 0 + x → x, etc. |
| `diff` operator | ✅ | Symbolic | 20+ tests |
| **Forward-Mode AD** |
| Dual numbers | ✅ | Forward | O(1) overhead/op |
| Scalar derivatives | ✅ | Forward | `derivative` |
| Higher-order derivatives | ✅ | Forward | Nested differentials |
| Math function support | ✅ | Forward | sin, cos, exp, log, sqrt, tan, sinh, cosh, tanh, abs, pow |
| Dual arithmetic | ✅ | Forward | +, -, *, / |
| `derivative` operator | ✅ | Forward | 30+ tests |
| **Reverse-Mode AD** |
| Computational graphs | ✅ | Reverse | Tape-based |
| Gradient computation | ✅ | Reverse | `gradient` |
| Backpropagation | ✅ | Reverse | Full backward pass |
| Nested gradients | ✅ | Reverse | Arbitrary depth |
| Double backward | ✅ | Reverse | Second derivatives |
| Jacobian matrices | ✅ | Reverse | `jacobian` |
| Hessian matrices | ✅ | Reverse | `hessian` |
| Tape stack (nesting) | ✅ | Reverse | 32-level depth |
| AD-aware tensor ops | ✅ | Reverse | vref, matmul work with AD nodes |
| `gradient` operator | ✅ | Reverse | 40+ tests |
| `jacobian` operator | ✅ | Reverse | 15+ tests |
| `hessian` operator | ✅ | Reverse | 10+ tests |
| **Vector Calculus** |
| Divergence | ✅ | Reverse | ∇·F (trace of Jacobian) |
| Curl | ✅ | Reverse | ∇×F (3D + generalized 2-forms) |
| Laplacian | ✅ | Reverse | ∇²f (trace of Hessian) |
| Directional derivative | ✅ | Reverse | D_v f = ∇f·v |
| `divergence` operator | ✅ | Reverse | 5+ tests |
| `curl` operator | ✅ | Reverse | 5+ tests |
| `laplacian` operator | ✅ | Reverse | 5+ tests |
| `directional-deriv` operator | ✅ | Reverse | 5+ tests |

---

## Tensor & Linear Algebra

| Feature | Status | Dimensions | Notes |
|---------|--------|------------|-------|
| **Tensor Creation** |
| Literals `#(...)` | ✅ | 1D-4D | Uniform syntax |
| `zeros` | ✅ | N-D | Efficient memset |
| `ones` | ✅ | N-D | Fill with 1.0 |
| `eye` | ✅ | 2D | Identity matrix |
| `arange` | ✅ | 1D | Range with step |
| `linspace` | ✅ | 1D | Evenly spaced |
| **Tensor Access** |
| `tensor-get` | ✅ | N-D | Multi-index access |
| `vref` (1D shorthand) | ✅ | 1D | AD-aware |
| Slicing | ✅ | N-D | Zero-copy views |
| `tensor-set` | ✅ | N-D | Mutable update |
| **Tensor Reshaping** |
| `reshape` | ✅ | N-D | Zero-copy |
| `transpose` | ✅ | 2D | Matrix transpose |
| `flatten` | ✅ | N-D → 1D | Zero-copy |
| `tensor-shape` | ✅ | N-D | Dimension query |
| **Element-wise Ops** |
| `tensor-add`, `tensor-sub` | ✅ | N-D | Broadcasting: ✅ |
| `tensor-mul`, `tensor-div` | ✅ | N-D | Element-wise |
| `tensor-apply` | ✅ | N-D | Map function |
| **Linear Algebra** |
| `tensor-dot` / `matmul` | ✅ | 1D, 2D | Dot product, matrix multiply |
| `trace` | ✅ | 2D | Diagonal sum |
| `norm` | ✅ | 1D | L2 norm (Euclidean) |
| `outer` | ✅ | 1D×1D→2D | Outer product |
| Determinant | ✅ | 2D | Via lib/math.esk (LU decomposition) |
| Matrix inverse | ✅ | 2D | Via lib/math.esk (Gauss-Jordan) |
| Linear solve | ✅ | 2D | Via lib/math.esk |
| Eigenvalues | ✅ | 2D | Via lib/math.esk (power iteration) |
| SVD | ✅ | 2D | Native (tensor_codegen.cpp) |
| QR decomposition | ✅ | 2D | Native (tensor_codegen.cpp) |
| **Reductions** |
| `tensor-sum` | ✅ | N-D | Sum all elements |
| `tensor-mean` | ✅ | N-D | Average |
| `tensor-reduce` | ✅ | N-D | Custom reduction |
| Axis-specific reduce | ✅ | N-D | Reduce along dimension |
| **Data Types** |
| Float64 elements | ✅ | N-D | IEEE 754 double |
| Int64 elements | 📋 | N-D | Planned integer tensors |
| Complex elements | 📋 | N-D | Planned |
| Sparse tensors | 📋 | N-D | Planned |

---

## List Processing

| Feature | Status | Performance | Notes |
|---------|--------|-------------|-------|
| **Basic Operations** |
| `cons`, `car`, `cdr` | ✅ | O(1) | Tagged cons cells |
| `list` | ✅ | O(n) | Left-to-right eval |
| `list*` | ✅ | O(n) | Improper lists |
| `length` | ✅ | O(n) | Stdlib |
| `append` | ✅ | O(n+m) | Stdlib |
| `reverse` | ✅ | O(n) | Stdlib |
| `list-ref` | ✅ | O(n) | Stdlib |
| **Higher-Order** |
| `map` | ✅ | O(n) | Builtin (iterative IR) |
| `filter` | ✅ | O(n) | Stdlib |
| `fold`, `fold-right` | ✅ | O(n) | Stdlib |
| `for-each` | ✅ | O(n) | Stdlib |
| `any`, `every` | ✅ | O(n) | Stdlib |
| **Search & Query** |
| `member`, `memq`, `memv` | ✅ | O(n) | Stdlib |
| `assoc`, `assq`, `assv` | ✅ | O(n) | Stdlib |
| `find` | ✅ | O(n) | Stdlib |
| Binary search | ✅ | O(log n) | Stdlib |
| **Transformations** |
| `take`, `drop` | ✅ | O(n) | Stdlib |
| `split-at` | ✅ | O(n) | Builtin |
| `partition` | ✅ | O(n) | Builtin |
| `zip`, `unzip` | ✅ | O(n) | Stdlib |
| **Sorting** |
| Merge sort | ✅ | O(n log n) | Stdlib |
| Quick sort | ✅ | O(n log n) avg | Stdlib |
| Custom comparator | ✅ | - | Passed as function |
| **Generators** |
| `range` | ✅ | O(n) | Stdlib |
| `iota` | ✅ | O(n) | Stdlib |
| `make-list` | ✅ | O(n) | Stdlib |

---

## Memory Management

| Feature | Status | Type | Notes |
|---------|--------|------|-------|
| **OALR System** |
| Arena allocation | ✅ | Manual | Bump-pointer, O(1) alloc |
| Lexical regions | ✅ | Manual | `with-region` |
| Global arena | ✅ | Manual | Shared across functions |
| Region nesting | ✅ | Manual | Stack-based |
| Zero-copy views | ✅ | Automatic | reshape, slice, transpose |
| **Ownership** |
| Linear types | ✅ | Compile-time | `owned`, `move` markers |
| Borrow checking | ✅ | Compile-time | `borrow` construct |
| Escape analysis | ✅ | Compile-time | Region-based with conservative heap fallback |
| Reference counting | 📋 | Runtime | Planned (`shared`, `weak-ref`) |
| **Garbage Collection** |
| Mark-sweep GC | ❌ | - | By design (arena-based instead) |
| Generational GC | ❌ | - | By design |

---

## Compilation & Runtime

| Feature | Status | Backend | Performance |
|---------|--------|---------|-------------|
| **Compiler** |
| S-expression parser | ✅ | Recursive descent | Fast |
| Macro system | ✅ | Hygenic macros | `define-syntax` |
| HoTT type checker | ✅ | Bidirectional | Gradual typing |
| LLVM IR generation | ✅ | LLVM 21 | 34,928 lines |
| Native code emission | ✅ | x86-64, ARM64 | Object files |
| Executable linking | ✅ | System linker | Standalone binaries |
| **Optimizations** |
| Constant folding | ✅ | LLVM | Automatic |
| Dead code elimination | ✅ | LLVM | Automatic |
| Inlining | ✅ | LLVM | Automatic |
| Tail call optimization | ✅ | Custom | Self-recursion |
| Type-directed optimization | ✅ | HoTT | When types known |
| SIMD vectorization | ✅ | LLVM | Loop metadata + micro-kernels |
| **REPL** |
| Interactive evaluation | ✅ | JIT | LLVM ORC |
| Cross-eval persistence | ✅ | JIT | Symbols/functions persist |
| Incremental compilation | ✅ | JIT | Per-expression |
| Hot code reload | ✅ | JIT | LLVM ORC remove() |
| **Debugging** |
| Source location tracking | ✅ | DWARF | Via `-g` flag |
| Stack traces | 📋 | - | Planned |
| Breakpoint support | 📋 | - | Planned |
| REPL introspection | ✅ | - | `type-of`, `display` |

---

## Standard Library

| Module | Status | Functions | Test Coverage |
|--------|--------|-----------|---------------|
| `core.operators.arithmetic` | ✅ | +, -, *, /, mod, quotient, gcd, lcm, min, max | 20+ tests |
| `core.operators.compare` | ✅ | <, >, =, <=, >= | 15+ tests |
| `core.logic.boolean` | ✅ | and, or, not, xor | 10+ tests |
| `core.logic.predicates` | ✅ | even?, odd?, zero?, positive?, negative? | 15+ tests |
| `core.logic.types` | ✅ | Type conversions | 10+ tests |
| `core.list.compound` | ✅ | cadr, caddr, etc. (16 functions) | 20+ tests |
| `core.list.higher_order` | ✅ | fold, filter, any, every | 25+ tests |
| `core.list.query` | ✅ | length, find, take, drop | 20+ tests |
| `core.list.search` | ✅ | member, assoc, binary-search | 15+ tests |
| `core.list.sort` | ✅ | sort, merge, insertion-sort | 10+ tests |
| `core.list.transform` | ✅ | append, reverse, map, filter | 30+ tests |
| `core.list.generate` | ✅ | range, iota, make-list, zip | 15+ tests |
| `core.functional.compose` | ✅ | compose, pipe | 10+ tests |
| `core.functional.curry` | ✅ | curry, uncurry | 5+ tests |
| `core.functional.flip` | ✅ | flip arguments | 5+ tests |
| `core.strings` | ✅ | String utilities | 20+ tests |
| `core.json` | ✅ | JSON parse/generate | 10+ tests |
| `core.io` | ✅ | File I/O, ports | 15+ tests |
| `core.data.base64` | ✅ | Base64 encode/decode | 5+ tests |
| `core.data.csv` | ✅ | CSV parsing | 5+ tests |
| `core.control.trampoline` | ✅ | TCO helpers | 5+ tests |
| Math library | ✅ | det, inv, solve, integrate, newton | 10+ tests |
| `math.statistics` | ✅ | mean, variance, normal, poisson, binomial | 10+ tests |
| `math.ode` | ✅ | rk4, euler, midpoint ODE solvers | 5+ tests |
| `signal.filters` | ✅ | Window functions, FIR/IIR, Butterworth, convolution | 12+ tests |
| `ml.optimization` | ✅ | Gradient descent, Adam, L-BFGS, conjugate gradient | 10+ tests |
| `ml.activations` | ✅ | relu, sigmoid, tanh, gelu, leaky-relu, silu | 5+ tests |

---

## I/O & System Integration

| Feature | Status | API | Notes |
|---------|--------|-----|-------|
| **File I/O** |
| Text file reading | ✅ | `open-input-file`, `read-line` | Buffered |
| Text file writing | ✅ | `open-output-file`, `write-line` | Buffered |
| Binary I/O | ✅ | R7RS bytevectors | Full R7RS binary I/O |
| Port operations | ✅ | `close-port`, `eof-object?` | Complete |
| **Console I/O** |
| `display` | ✅ | - | Homoiconic (shows lambdas) |
| `newline` | ✅ | - | Standard |
| `error` | ✅ | - | Exception-based |
| **System** |
| Environment vars | ✅ | `getenv`, `setenv`, `unsetenv` | POSIX |
| Command execution | ✅ | `system` | Shell commands |
| Process control | ✅ | `exit` | Exit codes |
| Time | ✅ | `current-seconds` | Unix timestamp |
| Sleep | ✅ | `sleep` | Milliseconds |
| Command-line args | ✅ | `command-line` | argc/argv |
| **File System** |
| File queries | ✅ | `file-exists?`, `file-size`, etc. | POSIX stat |
| Directory ops | ✅ | `make-directory`, `directory-list` | POSIX |
| Current directory | ✅ | `current-directory`, `set-current-directory!` | chdir |
| File operations | ✅ | `file-delete`, `file-rename` | POSIX |
| **Random Numbers** |
| Pseudo-random | ✅ | `random` | drand48 |
| Quantum random | ✅ | `quantum-random` | ANU QRNG API |
| Integer ranges | ✅ | `quantum-random-range` | Uniform distribution |

---

## Advanced Features

| Feature | Status | Maturity | Notes |
|---------|--------|----------|-------|
| **Metaprogramming** |
| Homoiconic code | ✅ | Stable | Code-as-data |
| S-expression manipulation | ✅ | Stable | quote, quasiquote |
| Lambda S-expression display | ✅ | Stable | Shows source code |
| Macro system | ✅ | Stable | `define-syntax` |
| **Exception Handling** |
| `guard` / `raise` | ✅ | Stable | setjmp/longjmp |
| Exception types | ✅ | Stable | User-defined |
| Stack unwinding | ✅ | Stable | Handler stack |
| **Multiple Return Values** |
| `values` | ✅ | Stable | Multi-value objects |
| `call-with-values` | ✅ | Stable | Consumer pattern |
| `let-values` | ✅ | Stable | Destructuring |
| **Control Flow (v1.1)** |
| `call/cc` | ✅ | Stable | First-class continuations |
| `dynamic-wind` | ✅ | Stable | Cleanup handlers |
| `guard` / `raise` | ✅ | Stable | Exception handling |
| **FFI (Foreign Function Interface)** |
| C function calls | ✅ | Stable | `extern` declarations |
| C variable access | ✅ | Stable | `extern-var` |
| Variadic C functions | ✅ | Stable | printf, etc. |
| Callback registration | 📋 | - | Planned |
| **Concurrency (v1.1)** |
| `parallel-map` | ✅ | Stable | Work-stealing thread pool |
| `parallel-fold` | ✅ | Stable | Parallel reduction |
| `parallel-filter` | ✅ | Stable | Parallel predicate filter |
| `parallel-for-each` | ✅ | Stable | Parallel side effects |
| `parallel-execute` | ✅ | Stable | Concurrent execution |
| `future` / `force` | ✅ | Stable | Asynchronous computation |
| Thread pool scheduler | ✅ | Stable | Hardware-aware sizing |
| **Module System** |
| `import` / `require` | ✅ | Stable | DFS dependency resolution |
| `load` (R7RS file loading) | ✅ | Stable | Alias for require with file path conversion |
| `provide` / `export` | ✅ | Stable | Symbol export |
| Module prefixing | ✅ | Stable | Namespace isolation |
| Circular dependency detection | ✅ | Stable | Compile-time error |
| Separate compilation | ✅ | Stable | .o file linking |

---

## Performance Characteristics

| Operation | Big-O | Notes |
|-----------|-------|-------|
| **Memory** |
| Arena allocation | O(1) | Bump pointer |
| Cons cell creation | O(1) | 32 bytes + header |
| Tensor creation (n elements) | O(n) | Contiguous allocation |
| Region cleanup | O(1) | Mark used pointer |
| **Arithmetic** |
| Int64 operations | O(1) | Direct CPU instructions |
| Double operations | O(1) | FPU instructions |
| Polymorphic dispatch | O(1) | Runtime type check |
| **AD Operations** |
| Forward-mode derivative | O(1) | Per operation overhead |
| Reverse-mode gradient (n→1) | O(1) | One backward pass |
| Jacobian (n→m) | O(m) | m gradient computations |
| Hessian (n→1) | O(n²) | Numerical finite differences |
| **List Operations** |
| cons, car, cdr | O(1) | Pointer operations |
| length | O(n) | Traversal |
| append | O(n+m) | Copy first list |
| reverse | O(n) | Iterative |
| map | O(n) | Single pass |
| sort (merge) | O(n log n) | Divide-and-conquer |
| **Tensor Operations** |
| Element access | O(1) | Computed index |
| Reshape | O(1) | Zero-copy view |
| Transpose (2D) | O(mn) | Element reordering |
| Matrix multiply (m×k, k×n) | O(mnk) | Triple loop |
| Element-wise ops | O(n) | Single pass |

---

## Platform Support

| Platform | Status | Architecture | Notes |
|----------|--------|--------------|-------|
| **Operating Systems** |
| Linux | ✅ | x86-64, ARM64 | Primary platform |
| macOS | ✅ | x86-64, ARM64 | Full support |
| Windows | ✅ | x86-64 | Native Visual Studio 2022 + ClangCL/LLVM 21 |
| FreeBSD | 📋 | x86-64 | Planned |
| **Architectures** |
| x86-64 | ✅ | SSE2+ | AVX/AVX2/AVX-512 supported |
| ARM64 | ✅ | Neon | Full support |
| RISC-V | 📋 | - | Planned |
| WebAssembly | ✅ | wasm32 | Via `--wasm` flag (LLVM 21 backend) |
| Web REPL | ✅ | Browser | `web/index.html` — interactive Eshkol in-browser |
| **Build Systems** |
| CMake | ✅ | 3.14+ | Primary (Ninja recommended) |
| Makefile | 📋 | - | Planned |
| Nix | 📋 | - | Planned |
| **Package Managers** |
| Homebrew | ✅ | macOS/Linux | Formula complete |
| APT (Debian/Ubuntu) | ✅ | Linux | .deb pipeline complete |
| RPM (Fedora/RHEL) | 📋 | Linux | Planned |

---

## Tooling & Ecosystem

| Tool | Status | Purpose | Notes |
|------|--------|---------|-------|
| **Compiler Tools** |
| `eshkol-compile` | ✅ | Ahead-of-time compiler | Produces executables |
| `eshkol-run` | ✅ | Script runner | Compile + execute |
| `eshkol-repl` | ✅ | Interactive shell | JIT-based with stdlib |
| `eshkol-pkg` | ✅ | Package manager | Registry support |
| `eshkol-lsp` | ✅ | Language server | IDE integration |
| **Development Tools** |
| Syntax highlighter | ✅ | Editor support | VS Code extension |
| LSP server | ✅ | IDE integration | Diagnostics, completion |
| Debugger | 📋 | Interactive debugging | Planned |
| Profiler | 📋 | Performance analysis | Planned |
| **Documentation** |
| API Reference | ✅ | Complete | 700+ builtins |
| Quickstart Guide | ✅ | Tutorial | 15-minute intro |
| Architecture Guide | ✅ | Internals | System design |
| Type System Guide | ✅ | HoTT types | Dependent types |
| Examples | ✅ | Demo programs | Neural networks, physics, ML |
| **Testing** |
| Unit tests | ✅ | Component tests | 426 files |
| Integration tests | ✅ | End-to-end | Full programs |
| AD verification | ✅ | Numerical validation | Gradient checking |
| Benchmark suite | ✅ | Performance tracking | GPU + CPU benchmarks |

---

## ML & AI Capabilities

| Feature | Status | Level | Applications |
|---------|--------|-------|--------------|
| **Neural Networks** |
| Forward pass | ✅ | Production | Any architecture |
| Backpropagation | ✅ | Production | Via `gradient` |
| Activation functions | ✅ | Production | 14 builtins: relu, sigmoid, softmax, gelu, silu, mish, etc. |
| Loss functions | ✅ | Production | 14 builtins: MSE, cross-entropy, focal, triplet, etc. |
| Optimizers | ✅ | Production | SGD, Adam, AdamW, RMSprop, Adagrad (builtins) + stdlib |
| Weight initialization | ✅ | Production | xavier, kaiming, lecun (5 builtin initializers) |
| LR schedulers | ✅ | Production | cosine-annealing, step-decay, warmup, exponential |
| **Supported Architectures** |
| Feedforward | ✅ | Production | Fully connected |
| CNN | ✅ | Production | conv1d/2d/3d, max-pool2d, avg-pool2d, batch/layer norm |
| RNN | 🚧 | Prototype | Sequential processing |
| Transformer | ✅ | Production | scaled-dot-attention, multi-head, RoPE, positional-encoding |
| **Training Features** |
| Batch training | ✅ | Production | Via user code |
| Mini-batch SGD | ✅ | Production | Via user code |
| Learning rate scheduling | ✅ | Production | Via user code |
| Regularization | ✅ | Production | L1/L2 in loss |
| Early stopping | ✅ | Production | Via user code |
| **Model Operations** |
| Save/load weights | 🚧 | - | Via file I/O |
| Model serialization | 📋 | - | Planned |
| ONNX export | 📋 | - | Planned |
| **Datasets** |
| In-memory datasets | ✅ | Production | Lists/tensors |
| Lazy loading | 📋 | - | Planned |
| Data augmentation | 📋 | - | Planned |

---

## Scientific Computing

| Domain | Status | Features | Examples |
|--------|--------|----------|----------|
| **Numerical Analysis** |
| Root finding | ✅ | Newton-Raphson | lib/math.esk |
| Integration | ✅ | Simpson's rule | lib/math.esk |
| Interpolation | 📋 | - | Planned |
| ODE solvers | ✅ | RK4, Euler, Midpoint | math.ode |
| PDE solvers | 🚧 | Finite differences | Via user code |
| **Linear Algebra** |
| Matrix operations | ✅ | Full suite | matmul, transpose, trace |
| LU decomposition | ✅ | Pure Eshkol | lib/math.esk |
| Matrix inverse | ✅ | Gauss-Jordan | lib/math.esk |
| Linear systems | ✅ | Gaussian elim | lib/math.esk |
| Eigenvalues | ✅ | Power iteration | lib/math.esk |
| **Statistics** |
| Descriptive stats | ✅ | mean, variance, std | lib/math.esk |
| Covariance | ✅ | Vector covariance | lib/math.esk |
| Distributions | ✅ | Normal, Poisson, Binomial, etc. | math.statistics |
| Hypothesis testing | 📋 | - | Planned |
| **Optimization** |
| Gradient descent | ✅ | Via `gradient` | ml.optimization |
| Adam optimizer | ✅ | Adaptive moments | ml.optimization |
| L-BFGS | ✅ | Two-loop recursion | ml.optimization |
| Conjugate gradient | ✅ | Fletcher-Reeves | ml.optimization |
| Newton's method | ✅ | Via `hessian` | Second-order |
| Constrained optimization | 📋 | - | Planned |
| **Physics Simulation** |
| Vector calculus | ✅ | ∇, ∇·, ∇×, ∇² | Full support |
| Field theory | ✅ | Differential forms | curl, divergence |
| Heat equation | ✅ | Via Laplacian | Verified |
| Wave propagation | 🚧 | - | Via user code |
| Fluid dynamics | 📋 | - | Planned |

---

## Signal Processing (v1.1)

| Feature | Status | Module | Notes |
|---------|--------|--------|-------|
| **Window Functions** |
| Hamming window | ✅ | `signal.filters` | w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1)) |
| Hann window | ✅ | `signal.filters` | w[n] = 0.5*(1 - cos(2*pi*n/(N-1))) |
| Blackman window | ✅ | `signal.filters` | 3-term Blackman |
| Kaiser window | ✅ | `signal.filters` | Parametric beta, inline Bessel I0 |
| **Convolution** |
| Direct convolution | ✅ | `signal.filters` | O(N*M) time-domain |
| FFT convolution | ✅ | `signal.filters` | O(N log N) via fft/ifft |
| **Filters** |
| FIR filter | ✅ | `signal.filters` | Arbitrary coefficient application |
| IIR filter | ✅ | `signal.filters` | Direct Form I |
| Butterworth lowpass | ✅ | `signal.filters` | Bilinear transform |
| Butterworth highpass | ✅ | `signal.filters` | Frequency inversion |
| Butterworth bandpass | ✅ | `signal.filters` | Two-stage cascade |
| **Analysis** |
| Frequency response | ✅ | `signal.filters` | Magnitude + phase at N points |
| FFT | ✅ | Builtin | Cooley-Tukey radix-2 |
| IFFT | ✅ | Builtin | Inverse FFT |

---

## Consciousness Engine (v1.1)

| Feature | Status | Module | Notes |
|---------|--------|--------|-------|
| **Logic Programming** |
| Unification | ✅ | Builtin | `unify`, `walk` |
| Substitutions | ✅ | Builtin | `make-substitution` |
| Knowledge base | ✅ | Builtin | `make-kb`, `kb-assert!`, `kb-query` |
| Logic variables | ✅ | Builtin | `?x` syntax |
| **Active Inference** |
| Factor graphs | ✅ | Builtin | `make-factor-graph`, `fg-add-factor!` |
| Belief propagation | ✅ | Builtin | `fg-infer!` |
| CPT mutation | ✅ | Builtin | `fg-update-cpt!` |
| Free energy | ✅ | Builtin | `free-energy`, `expected-free-energy` |
| **Global Workspace** |
| Workspace creation | ✅ | Builtin | `make-workspace` |
| Module registration | ✅ | Builtin | `ws-register!` |
| Softmax competition | ✅ | Builtin | `ws-step!` |

---

## GPU Acceleration (v1.1)

| Feature | Status | Backend | Notes |
|---------|--------|---------|-------|
| **Metal (Apple Silicon)** |
| Elementwise operations | ✅ | Metal | SF64 software float64 |
| Matrix multiplication | ✅ | Metal | Ozaki-II adaptive N |
| Reduce operations | ✅ | Metal | Sum, max, min |
| Softmax | ✅ | Metal | Numerically stable |
| Transpose | ✅ | Metal | 2D matrix transpose |
| **CUDA (NVIDIA)** |
| Elementwise operations | ✅ | CUDA | cuBLAS integration |
| Matrix multiplication | ✅ | CUDA | cuBLAS GEMM |
| Reduce operations | ✅ | CUDA | Custom kernels |
| Softmax | ✅ | CUDA | Numerically stable |
| Transpose | ✅ | CUDA | cuBLAS transpose |
| **Dispatch** |
| Automatic CPU/GPU selection | ✅ | Runtime | Cost model based |
| Threshold-based dispatch | ✅ | Runtime | XLA → cBLAS → SIMD → scalar |

---

## XLA Backend (v1.1)

| Feature | Status | Mode | Notes |
|---------|--------|------|-------|
| StableHLO/MLIR path | ✅ | When MLIR available | Hardware-optimized |
| LLVM-direct path | ✅ | Default | Hand-tuned IR |
| Matmul fusion | ✅ | Both | Fused multiply-add |
| Elementwise fusion | ✅ | Both | Operation chains |
| Reduce operations | ✅ | Both | Sum, max, min |
| Transpose | ✅ | Both | Shape operations |

---

## Interoperability

| Interface | Status | Direction | Notes |
|-----------|--------|-----------|-------|
| **C Integration** |
| Call C functions | ✅ | Eshkol → C | extern declarations |
| Access C globals | ✅ | Eshkol → C | extern-var |
| C calls Eshkol | 📋 | C → Eshkol | Planned callback API |
| **Python Integration** |
| Call Python from Eshkol | 📋 | - | Planned (ctypes/cffi) |
| Call Eshkol from Python | 📋 | - | Planned (wrapper lib) |
| NumPy interop | 📋 | - | Planned (array protocol) |
| **Data Formats** |
| JSON | ✅ | - | Parse and generate |
| CSV | ✅ | - | Read and write |
| Base64 | ✅ | - | Encode and decode |
| MessagePack | 📋 | - | Planned |
| Protocol Buffers | 📋 | - | Planned |
| **Databases** |
| SQLite | 📋 | - | Planned |
| PostgreSQL | 📋 | - | Planned |

---

## Comparison with Other Languages

| Feature | Eshkol | Python | Julia | Haskell | Scheme |
|---------|--------|--------|-------|---------|--------|
| **Language Type** |
| Paradigm | Functional-first | Multi-paradigm | Multi-paradigm | Purely functional | Functional |
| Type System | Gradual + Dependent | Dynamic | Dynamic | Static | Dynamic |
| Memory Model | OALR (regions) | GC | GC | GC | GC |
| **Automatic Differentiation** |
| Built-in AD | ✅ 3 modes | ❌ (libraries) | ✅ (libraries) | ❌ (libraries) | ❌ |
| Forward-mode | ✅ Dual numbers | JAX, PyTorch | ForwardDiff.jl | ad | ❌ |
| Reverse-mode | ✅ Tape-based | JAX, PyTorch | Zygote.jl | - | ❌ |
| Symbolic | ✅ Compile-time | SymPy | Symbolics.jl | - | ❌ |
| **Performance** |
| Native compilation | ✅ LLVM | ❌ (CPython) | ✅ LLVM | ✅ GHC | ❌ (most) |
| JIT available | ✅ REPL | ❌ (CPython) | ✅ | ❌ | ❌ (most) |
| Zero-copy views | ✅ | ✅ (NumPy) | ✅ | ❌ | ❌ |
| Tail call optimization | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Ease of Use** |
| Interactive REPL | ✅ | ✅ | ✅ | ✅ | ✅ |
| Package manager | ✅ eshkol-pkg | ✅ pip | ✅ Pkg | ✅ cabal | Varies |
| IDE support | ✅ LSP | ✅ | ✅ | ✅ | ✅ |
| Learning curve | Medium | Low | Medium | High | Medium |

---

## Test Coverage Summary

| Category | Test Files | Status | Notes |
|----------|-----------|--------|-------|
| **Core Language** | 80+ | ✅ | All special forms verified |
| **List Processing** | 60+ | ✅ | Comprehensive coverage |
| **Automatic Differentiation** | 50+ | ✅ | All 3 modes validated |
| **Tensors** | 30+ | ✅ | N-D operations verified |
| **Neural Networks** | 10+ | ✅ | Training loops work |
| **Standard Library** | 40+ | ✅ | All modules tested |
| **Type System** | 15+ | ✅ | HoTT types validated |
| **Memory Management** | 20+ | ✅ | Arena correctness |
| **System Integration** | 15+ | ✅ | File I/O, system calls |
| **REPL/JIT** | 10+ | ✅ | Cross-eval persistence |
| **Total** | **426** | **✅** | **High confidence** |

---

## Roadmap

### v1.1-accelerate (Q1 2026) — COMPLETED

- ✅ **GPU Support**: Metal (Apple Silicon) + CUDA (NVIDIA)
- ✅ **XLA Backend**: StableHLO/MLIR + LLVM-direct
- ✅ **Parallel Primitives**: parallel-map, parallel-fold, future/force
- ✅ **Exact Arithmetic**: Bignums, rationals, full numeric tower
- ✅ **Consciousness Engine**: Logic, inference, workspace (22 builtins)
- ✅ **Signal Processing**: FFT, filters, window functions
- ✅ **Optimizers**: Adam, L-BFGS, conjugate gradient in stdlib
- ✅ **R7RS Extensions**: call/cc, dynamic-wind, bytevectors, let-syntax

### v1.2-scale (Q2 2026)

- **Data I/O**: Image/audio I/O, typed buffers, streams, DataFrame, plotting
- **Vulkan Compute**: Cross-platform GPU backend, multi-GPU
- **Model Deployment**: Serialization, ONNX export, quantization
- **Python Bindings**: Call Eshkol from Python and vice versa
- **Distributed Training**: AllReduce, MPI, gRPC

### v1.3-evolve (Q3 2026)

- **Language Extensions**: Full R7RS library system, string interpolation, keyword arguments
- **Advanced Types**: Refinement types, effect types, higher-rank types, row polymorphism
- **Compiler Optimization**: PGO, whole-program optimization, polyhedral loop optimization

### v1.4-connection (Q4 2026)

- **Platform Abstraction**: Cross-platform windows, event system, event loop
- **Real-Time Audio**: Device management, synthesis, MIDI I/O
- **Networking**: TCP/UDP sockets with linear resource management
- **Embedded & Robotics**: GPIO, I2C/SPI/UART, PWM, ADC/DAC, mobile targets

### v1.5-intelligence (Q1 2027)

- **Neuro-Symbolic Bridge**: Soft unification, symbol embeddings, attention over KB
- **Program Synthesis**: Type-directed holes, neural-guided search
- **Advanced Neural**: LSTM/GRU cells, Graph Neural Networks

### v2.0-starlight (2027+)

- **Quantum Computing**: Qubit types with linear tracking, gates, VQE/QAOA
- **Formal Verification**: Proof assistant integration, certified compilation
- **Next-Gen Types**: Session types, algebraic effects, quantitative type theory

---

## Production Readiness

### ✅ Production-Ready (v1.1)

- Core language (70+ special forms, 700+ builtins)
- Automatic differentiation (3 modes)
- Tensor operations (30+ functions)
- List processing (50+ operations)
- Standard library (25+ modules, 300+ functions)
- LLVM-based native compilation
- Arena-based memory management
- REPL with JIT compilation and stdlib
- Module system with package manager
- GPU acceleration (Metal + CUDA)
- Parallel primitives (thread pool, futures)
- Exact arithmetic (bignums, rationals)
- Complex numbers with AD
- Signal processing (FFT, filters)
- Consciousness engine (22 builtins)
- call/cc and dynamic-wind
- Bytevectors
- LSP server

### 🚧 Beta Quality

- FFI (works but callback registration planned)
- Quantum RNG (external dependency)
- XLA StableHLO path (requires MLIR, LLVM-direct is default)

### 📋 Not Yet Production

- Distributed computing
- Model serialization/ONNX export
- Vulkan Compute

---

## Dual Backend Architecture (v1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| **Bytecode VM** |
| 63-opcode ISA | ✅ | Register+stack architecture, computed-goto dispatch |
| 250+ native call IDs | ✅ | Math, string, IO, complex, rational, bignum, dual, AD, tensor, logic, inference, workspace, hash, bytevector, parameter |
| ESKB binary format | ✅ | Section-based layout, LEB128 encoding, CRC32 checksums |
| `-B` flag (bytecode emission) | ✅ | `eshkol-run input.esk -B output.eskb` |
| VM compiler integration | ✅ | eshkol_vm.c linked into compiler build |
| Closures & upvalues | ✅ | Closure creation, open/close upvalues, mutable captures |
| call/cc & dynamic-wind | ✅ | Continuation capture, wind stack |
| guard/raise exceptions | ✅ | Handler stack with continuation restore |
| Variadic functions | ✅ | OP_PACK_REST for rest parameters |
| **Weight Matrix Transformer** |
| Transformer interpreter | ✅ | d_model=36, 5 layers, FFN_DIM=512, 307K params |
| 3-way verification | ✅ | Reference = simulated = matrix-based (55/55 tests) |
| QLMW binary export | ✅ | For qLLM weight loading |
| 25 core opcodes in weights | ✅ | Remaining 38 via native dispatch |
| **qLLM Bridge** |
| Eshkol↔qLLM tensors | ✅ | Type conversion (double↔float32) with AD integration |
| Web Platform | ✅ Complete | WebAssembly compilation, 59 DOM bindings, browser REPL, eshkol.ai |
| VM Dual Number AD | ✅ Complete | Forward-mode AD via dual numbers in bytecode VM |
| VM Production | ✅ Complete | 176/176 tests, zero stubs, zero stdout contamination |
| KB Pattern Matching | ✅ Complete | Knowledge base queries with ?-wildcard pattern matching |

## Tensor Linear Algebra (v1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| `tensor-cholesky` | ✅ | Cholesky decomposition |
| `tensor-lu` | ✅ | LU decomposition |
| `tensor-qr` | ✅ | QR decomposition |
| `tensor-svd` | ✅ | Singular value decomposition |
| `tensor-solve` | ✅ | Linear system solver |
| `tensor-det` | ✅ | Determinant |
| `tensor-inverse` | ✅ | Matrix inverse |
| `tensor-cov` | ✅ | Covariance matrix |
| `tensor-corrcoef` | ✅ | Correlation coefficient matrix |

## Data Loading (v1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| `make-dataloader` | ✅ | Create batched data iterator |
| `dataloader-next` | ✅ | Get next batch |
| `dataloader-reset` | ✅ | Reset to beginning |
| `dataloader-length` | ✅ | Total number of batches |
| `dataloader-has-next` | ✅ | Check if more batches available |
| `train-test-split` | ✅ | Split dataset into train/test |

---

## Known Limitations

1. **Single GPU dispatch** - One GPU at a time (multi-GPU planned v1.2)
3. **Small ecosystem** - Growing standard library, but not as extensive as Python/Julia
4. **Learning curve** - Functional programming + AD concepts require study
5. **Platform support** - Linux, macOS, and native Windows x64

---

## Strengths

1. **Best-in-class AD** - Three modes (symbolic, forward, reverse) in one language
2. **Zero manual derivatives** - Compute gradients of **any** Eshkol function automatically
3. **Production compiler** - LLVM backend produces optimized native code
4. **Scientific focus** - Designed for numerical computing and physics simulation
5. **Homoiconic** - Code is data, metaprogramming is natural
6. **Memory safety** - OALR prevents leaks without GC pauses
7. **Scheme heritage** - Clean, powerful functional programming model

---

## Version History

### v1.1 (March 2026) - Accelerate Release

**Highlights**:
- XLA backend with dual-mode tensor acceleration
- GPU acceleration: Metal (Apple Silicon) + CUDA (NVIDIA)
- Parallel primitives with work-stealing thread pool
- Arbitrary-precision arithmetic (bignums + rationals)
- Consciousness engine (logic, inference, workspace)
- Signal processing library (FFT, filters, window functions)
- R7RS extensions (call/cc, dynamic-wind, bytevectors)
- 700+ builtins, 35 test suites, 438 test files

**Codebase**: ~232,000 lines of production C/C++

### v1.0 (December 2025) - Foundation Release

**Highlights**:
- Complete automatic differentiation system (3 modes)
- N-dimensional tensor operations
- 70+ special forms
- 180+ standard library functions
- HoTT dependent type system
- LLVM native compilation
- Arena-based memory management

**Codebase**: 67,079 lines of production C++

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

**Key Areas**:
- GPU acceleration (CUDA/Metal backends)
- Advanced ML ops (convolution, attention)
- IDE tooling (LSP, debugger)
- Python/Julia interop
- Package ecosystem

---

## License & Credits

**License**: MIT  
**Copyright**: © 2025 tsotchke  
**LLVM**: Apache 2.0 with LLVM Exception  

**Acknowledgments**:
- LLVM Project (compiler infrastructure)
- Scheme community (language design inspiration)
- JAX/PyTorch (AD implementation insights)
- Julia (technical computing design patterns)

---

**Last Updated**: 2026-04-09
**Document Version**: 1.1.13

For detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md)
