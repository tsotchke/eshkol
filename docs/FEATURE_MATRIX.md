# Eshkol v1.3.4 Feature Matrix

**Status Key** (table cells): `Yes` = Production | `WIP` = In Progress | `Planned` = Planned | `No` = Not Planned | `Partial` = Partially supported

This matrix documents all implemented and planned features for the Eshkol language ecosystem. All **Production** features are code-verified with extensive test coverage (37 suites, 528 self-reported tests).

---

## Core Language Features

| Feature | Status | Notes | Test Coverage |
|---------|--------|-------|---------------|
| **Special Forms** |
| `define` (variables) | Yes | Global and local bindings | 50+ tests |
| `define` (functions) | Yes | Top-level and nested | 50+ tests |
| `lambda` | Yes | Closures with captures | 100+ tests |
| `let`, `let*`, `letrec` | Yes | All binding forms | 80+ tests |
| Named let (iteration) | Yes | Tail-recursive loops | 10+ tests |
| `set!` | Yes | Mutable variables and captures | 30+ tests |
| `if`, `cond`, `case` | Yes | All conditionals | 40+ tests |
| `begin` | Yes | Sequence evaluation | 20+ tests |
| `and`, `or`, `not` | Yes | Short-circuit boolean logic | 15+ tests |
| `when`, `unless` | Yes | One-armed conditionals | 10+ tests |
| `do` loops | Yes | Iteration with state | 5+ tests |
| `quote`, `quasiquote` | Yes | S-expression literals | 20+ tests |
| `apply` | Yes | Dynamic function application | 15+ tests |
| **Pattern Matching** |
| `match` | Yes | Full pattern matching | 10+ tests |
| Variable patterns | Yes | Binding in patterns | Verified |
| Literal patterns | Yes | Constant matching | Verified |
| Cons patterns | Yes | `(p1 . p2)` destructuring | Verified |
| List patterns | Yes | `(p1 p2 ...)` matching | Verified |
| Predicate patterns | Yes | `(? pred)` guards | Verified |
| Or-patterns | Yes | `(or p1 p2 ...)` alternatives | Verified |
| **Closures** |
| Basic closures | Yes | Capture environment | 50+ tests |
| Mutable captures | Yes | `set!` on captured vars | 20+ tests |
| Nested closures | Yes | Arbitrary depth | 15+ tests |
| Variadic closures | Yes | Rest parameters | 10+ tests |
| Closure homoiconicity | Yes | Display shows source code | Verified |
| **Tail Call Optimization** |
| Self-recursive TCO | Yes | Functions calling themselves | 15+ tests |
| Mutual recursion TCO | Yes | Functions calling each other | Trampoline-based |
| Trampoline runtime | Yes | Non-self tail calls | 5+ tests |

---

## Type System

| Feature | Status | Notes | Implementation |
|---------|--------|-------|----------------|
| **Runtime Tagged Values** |
| Int64 | Yes | Exact integers | 16-byte struct |
| Double | Yes | IEEE 754 floats | 16-byte struct |
| Boolean | Yes | #t/#f | Type tag + bit |
| Char | Yes | Unicode codepoints | Type tag + int64 |
| String | Yes | Heap-allocated | HEAP_PTR + header |
| Symbol | Yes | Interned strings | HEAP_PTR + header |
| Cons/List | Yes | Heterogeneous pairs | HEAP_PTR + header |
| Vector | Yes | Scheme vectors | HEAP_PTR + header |
| Tensor | Yes | N-dimensional arrays | HEAP_PTR + header |
| Closure | Yes | Function + environment | CALLABLE + header |
| Hash Table | Yes | Mutable maps | HEAP_PTR + header |
| **HoTT Compile-Time Types** |
| Integer, Real, Number | Yes | Numeric hierarchy | Gradual typing |
| Boolean, Char, String | Yes | Primitive types | Gradual typing |
| List<T>, Vector<T> | Yes | Parameterized collections | Element type tracking |
| Tensor<T> | Yes | Typed tensors | Element type tracking |
| Function arrows (→) | Yes | `(→ A B)` types | Type inference |
| Dependent types | Yes | Path types, universes | Proof erasure |
| Gradual typing | Yes | Optional annotations | Warning-only errors |
| Checked ascription `(the <type> expr)` | Yes | v1.3.4: trusted assertion to the checker; runtime no-op (byte-identical IR) | Type checker |
| Predicate-guarded narrowing | Yes | v1.3.4: 8 predicates, honored across `if`/`and`, cancelled at `set!` | Type checker |
| Linear `Qubit` type | Yes | v1.3.4: use-exactly-once enforcement on `define`d linear params | HoTT linear types |
| Sum-type annotations on named-let params | Yes | v1.3.4: honored across iterations | Type checker |
| Numeric-tower join for accumulators | Yes | v1.3.4: recursive accumulator gets least-upper-bound numeric type | Type checker |
| **Exact Arithmetic (v1.1)** |
| Bignum (arbitrary-precision int) | Yes | Automatic overflow promotion | int64 → bignum |
| Rational numbers | Yes | Exact fractions (num/den) | HEAP_PTR + header |
| 128-bit integer (`i128`) | Yes | v1.3.4: fixed-width wrapping signed int off the numeric tower; native + VM | HEAP_SUBTYPE_I128 (25) |
| Complex numbers | Yes | `make-rectangular`, `make-polar` | Type tag 7 |
| `exact?`, `inexact?` | Yes | Exactness predicates | Runtime tags |
| `exact->inexact`, `inexact->exact` | Yes | Exactness conversion | Type conversion |
| **Type Predicates** |
| `number?`, `integer?`, `real?` | Yes | Numeric predicates | Runtime tags |
| `string?`, `char?`, `boolean?` | Yes | Primitive predicates | Runtime tags |
| `null?`, `pair?`, `list?` | Yes | List predicates | Runtime tags |
| `vector?`, `procedure?` | Yes | Compound predicates | Header subtype |
| `complex?`, `rational?` | Yes | Extended numeric predicates | Runtime tags |

---

## Automatic Differentiation

| Feature | Status | Mode | Performance |
|---------|--------|------|-------------|
| **Symbolic Differentiation** |
| Compile-time AST transform | Yes | Symbolic | O(1) - compile time |
| Sum rule | Yes | Symbolic | Verified |
| Product rule | Yes | Symbolic | Verified |
| Quotient rule | Yes | Symbolic | Verified |
| Chain rule | Yes | Symbolic | sin, cos, exp, log, pow, sqrt |
| Algebraic simplification | Yes | Symbolic | 0 + x → x, etc. |
| `diff` operator | Yes | Symbolic | 20+ tests |
| **Forward-Mode AD** |
| Dual numbers | Yes | Forward | O(1) overhead/op |
| Scalar derivatives | Yes | Forward | `derivative` |
| Higher-order derivatives | Yes | Forward | Nested differentials |
| Math function support | Yes | Forward | sin, cos, exp, log, sqrt, tan, sinh, cosh, tanh, abs, pow |
| Dual arithmetic | Yes | Forward | +, -, *, / |
| `derivative` operator | Yes | Forward | 30+ tests |
| **Reverse-Mode AD** |
| Computational graphs | Yes | Reverse | Tape-based |
| Gradient computation | Yes | Reverse | `gradient` |
| Backpropagation | Yes | Reverse | Full backward pass |
| Nested gradients | Yes / Partial | Reverse | Exact via nested scalar `derivative`; vector gradient-of-gradient returns zeros (ESH-0096) |
| Double backward | Yes / Partial | Reverse | Second derivatives via nested scalar `derivative`; `hessian` works on vector points (crashes on tensor points, ESH-0095) |
| Jacobian matrices | Yes | Reverse | `jacobian` |
| Hessian matrices | Yes | Reverse | `hessian` |
| Tape stack (nesting) | Yes | Reverse | 32-level depth |
| AD-aware tensor ops | Yes | Reverse | vref, matmul work with AD nodes |
| `gradient` operator | Yes | Reverse | 40+ tests |
| `jacobian` operator | Yes | Reverse | 15+ tests |
| `hessian` operator | Yes | Reverse | 10+ tests |
| **Vector Calculus** |
| Divergence | Yes | Reverse | ∇·F (trace of Jacobian) |
| Curl | Yes | Reverse | ∇×F (3D + generalized 2-forms) |
| Laplacian | Yes | Reverse | ∇²f (trace of Hessian) |
| Directional derivative | Yes | Reverse | D_v f = ∇f·v |
| `divergence` operator | Yes | Reverse | 5+ tests |
| `curl` operator | Yes | Reverse | 5+ tests |
| `laplacian` operator | Yes | Reverse | 5+ tests |
| `directional-derivative` operator | Yes | Reverse | 5+ tests |

---

## Tensor & Linear Algebra

| Feature | Status | Dimensions | Notes |
|---------|--------|------------|-------|
| **Tensor Creation** |
| Literals `#(...)` | Yes | 1D-4D | Uniform syntax |
| `zeros` | Yes | N-D | Efficient memset |
| `ones` | Yes | N-D | Fill with 1.0 |
| `eye` | Yes | 2D | Identity matrix |
| `arange` | Yes | 1D | Range with step |
| `linspace` | Yes | 1D | Evenly spaced |
| **Tensor Access** |
| `tensor-get` | Yes | N-D | Multi-index access |
| `vref` (1D shorthand) | Yes | 1D | AD-aware |
| Slicing | Yes | N-D | Zero-copy views |
| `tensor-set` | Yes | N-D | Mutable update |
| **Tensor Reshaping** |
| `reshape` | Yes | N-D | Zero-copy |
| `transpose` | Yes | 2D | Matrix transpose |
| `flatten` | Yes | N-D → 1D | Zero-copy |
| `tensor-shape` | Yes | N-D | Dimension query |
| **Element-wise Ops** |
| `tensor-add`, `tensor-sub` | Yes | N-D | Broadcasting: Yes |
| `tensor-mul`, `tensor-div` | Yes | N-D | Element-wise |
| `tensor-apply` | Yes | N-D | Map function |
| **Linear Algebra** |
| `tensor-dot` / `matmul` | Yes | 1D, 2D | Dot product, matrix multiply |
| `trace` | Yes | 2D | Diagonal sum |
| `norm` | Yes | 1D | L2 norm (Euclidean) |
| `outer` | Yes | 1D×1D→2D | Outer product |
| Determinant | Yes | 2D | Via lib/math.esk (LU decomposition) |
| Matrix inverse | Yes | 2D | Via lib/math.esk (Gauss-Jordan) |
| Linear solve | Yes | 2D | Via lib/math.esk |
| Eigenvalues | Yes | 2D | Via lib/math.esk (power iteration) |
| SVD | Yes | 2D | Native (tensor_codegen.cpp) |
| QR decomposition | Yes | 2D | Native (tensor_codegen.cpp) |
| **Reductions** |
| `tensor-sum` | Yes | N-D | Sum all elements |
| `tensor-mean` | Yes | N-D | Average |
| `tensor-reduce` | Yes | N-D | Custom reduction |
| Axis-specific reduce | Yes | N-D | Reduce along dimension |
| **Data Types** |
| Float64 elements | Yes | N-D | IEEE 754 double |
| Int64 elements | Planned | N-D | Planned integer tensors |
| Complex elements | Planned | N-D | Planned |
| Sparse tensors | Planned | N-D | Planned |

---

## List Processing

| Feature | Status | Performance | Notes |
|---------|--------|-------------|-------|
| **Basic Operations** |
| `cons`, `car`, `cdr` | Yes | O(1) | Tagged cons cells |
| `list` | Yes | O(n) | Left-to-right eval |
| `list*` | Yes | O(n) | Improper lists |
| `length` | Yes | O(n) | Stdlib |
| `append` | Yes | O(n+m) | Stdlib |
| `reverse` | Yes | O(n) | Stdlib |
| `list-ref` | Yes | O(n) | Stdlib |
| **Higher-Order** |
| `map` | Yes | O(n) | Builtin (iterative IR) |
| `filter` | Yes | O(n) | Stdlib |
| `fold`, `fold-right` | Yes | O(n) | Stdlib |
| `for-each` | Yes | O(n) | Stdlib |
| `any`, `every` | Yes | O(n) | Stdlib |
| **Search & Query** |
| `member`, `memq`, `memv` | Yes | O(n) | Stdlib |
| `assoc`, `assq`, `assv` | Yes | O(n) | Stdlib |
| `find` | Yes | O(n) | Stdlib |
| Binary search | Yes | O(log n) | Stdlib |
| **Transformations** |
| `take`, `drop` | Yes | O(n) | Stdlib |
| `split-at` | Yes | O(n) | Builtin |
| `partition` | Yes | O(n) | Builtin |
| `zip`, `unzip` | Yes | O(n) | Stdlib |
| **Sorting** |
| Merge sort | Yes | O(n log n) | Stdlib |
| Quick sort | Yes | O(n log n) avg | Stdlib |
| Custom comparator | Yes | - | Passed as function |
| **Generators** |
| `range` | Yes | O(n) | Stdlib |
| `iota` | Yes | O(n) | Stdlib |
| `make-list` | Yes | O(n) | Stdlib |

---

## Memory Management

| Feature | Status | Type | Notes |
|---------|--------|------|-------|
| **OALR System** |
| Arena allocation | Yes | Manual | Bump-pointer, O(1) alloc |
| Lexical regions | Yes | Manual | `with-region` |
| Global arena | Yes | Manual | Shared across functions |
| Region nesting | Yes | Manual | Stack-based |
| Zero-copy views | Yes | Automatic | reshape, slice, transpose |
| **Ownership** |
| Linear types | Yes | Compile-time | `owned`, `move` markers |
| Borrow checking | Yes | Compile-time | `borrow` construct |
| Escape analysis | Yes | Compile-time | Region-based with conservative heap fallback |
| Reference counting | Planned | Runtime | Planned (`shared`, `weak-ref`) |
| **Garbage Collection** |
| Mark-sweep GC | No | - | By design (arena-based instead) |
| Generational GC | No | - | By design |

---

## Compilation & Runtime

| Feature | Status | Backend | Performance |
|---------|--------|---------|-------------|
| **Compiler** |
| S-expression parser | Yes | Recursive descent | Fast |
| Macro system | Yes | Hygenic macros | `define-syntax` |
| HoTT type checker | Yes | Bidirectional | Gradual typing |
| LLVM IR generation | Yes | LLVM 21 | 34,928 lines |
| Native code emission | Yes | x86-64, ARM64 | Object files |
| Executable linking | Yes | System linker | Standalone binaries |
| **Optimizations** |
| Constant folding | Yes | LLVM | Automatic |
| Dead code elimination | Yes | LLVM | Automatic |
| Inlining | Yes | LLVM | Automatic |
| Tail call optimization | Yes | Custom | Self-recursion |
| Type-directed optimization | Yes | HoTT | When types known |
| SIMD vectorization | Yes | LLVM | Loop metadata + micro-kernels |
| **REPL** |
| Interactive evaluation | Yes | JIT | LLVM ORC |
| Cross-eval persistence | Yes | JIT | Symbols/functions persist |
| Incremental compilation | Yes | JIT | Per-expression |
| Hot code reload | Yes | JIT | LLVM ORC remove() |
| **Debugging** |
| Source location tracking | Yes | DWARF | Via `-g` flag |
| Stack traces | Planned | - | Planned |
| Breakpoint support | Planned | - | Planned |
| REPL introspection | Yes | - | `type-of`, `display` |

---

## Standard Library

| Module | Status | Functions | Test Coverage |
|--------|--------|-----------|---------------|
| `core.operators.arithmetic` | Yes | +, -, *, /, mod, quotient, gcd, lcm, min, max | 20+ tests |
| `core.operators.compare` | Yes | <, >, =, <=, >= | 15+ tests |
| `core.logic.boolean` | Yes | and, or, not, xor | 10+ tests |
| `core.logic.predicates` | Yes | even?, odd?, zero?, positive?, negative? | 15+ tests |
| `core.logic.types` | Yes | Type conversions | 10+ tests |
| `core.list.compound` | Yes | cadr, caddr, etc. (16 functions) | 20+ tests |
| `core.list.higher_order` | Yes | fold, filter, any, every | 25+ tests |
| `core.list.query` | Yes | length, find, take, drop | 20+ tests |
| `core.list.search` | Yes | member, assoc, binary-search | 15+ tests |
| `core.list.sort` | Yes | sort, merge, insertion-sort | 10+ tests |
| `core.list.transform` | Yes | append, reverse, map, filter | 30+ tests |
| `core.list.generate` | Yes | range, iota, make-list, zip | 15+ tests |
| `core.functional.compose` | Yes | compose, pipe | 10+ tests |
| `core.functional.curry` | Yes | curry, uncurry | 5+ tests |
| `core.functional.flip` | Yes | flip arguments | 5+ tests |
| `core.strings` | Yes | String utilities | 20+ tests |
| `core.json` | Yes | JSON parse/generate | 10+ tests |
| `core.io` | Yes | File I/O, ports | 15+ tests |
| `core.data.base64` | Yes | Base64 encode/decode | 5+ tests |
| `core.data.csv` | Yes | CSV parsing | 5+ tests |
| `core.control.trampoline` | Yes | TCO helpers | 5+ tests |
| Math library | Yes | det, inv, solve, integrate, newton | 10+ tests |
| `math.statistics` | Yes | mean, variance, normal, poisson, binomial | 10+ tests |
| `math.ode` | Yes | rk4, euler, midpoint ODE solvers | 5+ tests |
| `signal.filters` | Yes | Window functions, FIR/IIR, Butterworth, convolution | 12+ tests |
| `ml.optimization` | Yes | Gradient descent, Adam, L-BFGS, conjugate gradient | 10+ tests |
| `ml.activations` | Yes | relu, sigmoid, tanh, gelu, leaky-relu, silu | 5+ tests |

---

## I/O & System Integration

| Feature | Status | API | Notes |
|---------|--------|-----|-------|
| **File I/O** |
| Text file reading | Yes | `open-input-file`, `read-line` | Buffered |
| Text file writing | Yes | `open-output-file`, `write-line` | Buffered |
| Binary I/O | Yes | R7RS bytevectors | Full R7RS binary I/O |
| Port operations | Yes | `close-port`, `eof-object?` | Complete |
| **Console I/O** |
| `display` | Yes | - | Homoiconic (shows lambdas) |
| `newline` | Yes | - | Standard |
| `error` | Yes | - | Exception-based |
| **System** |
| Environment vars | Yes | `getenv`, `setenv`, `unsetenv` | POSIX |
| Command execution | Yes | `system` | Shell commands |
| Process control | Yes | `exit` | Exit codes |
| Time | Yes | `current-seconds` | Unix timestamp |
| Sleep | Yes | `sleep` | Milliseconds |
| Command-line args | Yes | `command-line` | argc/argv |
| **File System** |
| File queries | Yes | `file-exists?`, `file-size`, etc. | POSIX stat |
| Directory ops | Yes | `make-directory`, `directory-list` | POSIX |
| Current directory | Yes | `current-directory`, `set-current-directory!` | chdir |
| File operations | Yes | `file-delete`, `file-rename` | POSIX |
| **Random Numbers** |
| Pseudo-random | Yes | `random` | drand48 |
| Quantum random | Yes | `quantum-random` | Classical software PRNG fallback (NOT the ANU QRNG API, NOT real quantum hardware, NOT Bell-verified). Real quantum entropy (Moonlab, Bell-verified) is opt-in via `-DESHKOL_QUANTUM_ENABLED=ON`; see `docs/design/MOONLAB_INTEGRATION.md`. Check the active source at runtime via `eshkol_qrng_source_label()`. |
| Integer ranges | Yes | `quantum-random-range` | Uniform distribution (same classical-fallback-by-default caveat as above) |

---

## Advanced Features

| Feature | Status | Maturity | Notes |
|---------|--------|----------|-------|
| **Metaprogramming** |
| Homoiconic code | Yes | Stable | Code-as-data |
| S-expression manipulation | Yes | Stable | quote, quasiquote |
| Lambda S-expression display | Yes | Stable | Shows source code |
| Macro system | Yes | Stable | `define-syntax` |
| String interpolation | Yes | Experimental | `~{expr}` inside strings; `~~{` escapes the opener |
| **Exception Handling** |
| `guard` / `raise` | Yes | Stable | setjmp/longjmp |
| Exception types | Yes | Stable | User-defined |
| Stack unwinding | Yes | Stable | Handler stack |
| **Multiple Return Values** |
| `values` | Yes | Stable | Multi-value objects |
| `call-with-values` | Yes | Stable | Consumer pattern |
| `let-values` | Yes | Stable | Destructuring |
| **Control Flow (v1.1)** |
| `call/cc` | Yes | Stable | First-class continuations |
| `dynamic-wind` | Yes | Stable | Cleanup handlers |
| `guard` / `raise` | Yes | Stable | Exception handling |
| **FFI (Foreign Function Interface)** |
| C function calls | Yes | Stable | `extern` declarations |
| C variable access | Yes | Stable | `extern-var` |
| Variadic C functions | Yes | Stable | printf, etc. |
| Callback registration | Planned | - | Planned |
| **Concurrency (v1.1)** |
| `parallel-map` | Yes | Stable | Work-stealing thread pool |
| `parallel-fold` | Yes | Stable | Parallel reduction |
| `parallel-filter` | Yes | Stable | Parallel predicate filter |
| `parallel-for-each` | Yes | Stable | Parallel side effects |
| `parallel-execute` | Yes | Stable | Concurrent execution |
| `future` / `force` | Yes | Stable | Asynchronous computation |
| Thread pool scheduler | Yes | Stable | Hardware-aware sizing |
| **Module System** |
| `import` / `require` | Yes | Stable | DFS dependency resolution |
| `load` (R7RS file loading) | Yes | Stable | Alias for require with file path conversion |
| `provide` / `export` | Yes | Stable | Symbol export |
| Module prefixing | Yes | Stable | Namespace isolation |
| Circular dependency detection | Yes | Stable | Compile-time error |
| Separate compilation | Yes | Stable | .o file linking |

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
| Linux | Yes | x86-64, ARM64 | Primary platform |
| macOS | Yes | x86-64, ARM64 | Full support |
| Windows | Yes | x86-64 | Native Visual Studio 2022 + ClangCL/LLVM 21 |
| FreeBSD | Planned | x86-64 | Planned |
| **Architectures** |
| x86-64 | Yes | SSE2+ | AVX/AVX2/AVX-512 supported |
| ARM64 | Yes | Neon | Full support |
| RISC-V | Planned | - | Planned |
| WebAssembly | Yes | wasm32 | Via `--wasm` flag (LLVM 21 backend) |
| Web REPL | Yes | Browser | `web/index.html` — interactive Eshkol in-browser |
| **Build Systems** |
| CMake | Yes | 3.14+ | Primary (Ninja recommended) |
| Makefile | Planned | - | Planned |
| Nix | Planned | - | Planned |
| **Package Managers** |
| Homebrew | Yes | macOS/Linux | Formula complete |
| APT (Debian/Ubuntu) | Yes | Linux | .deb pipeline complete |
| RPM (Fedora/RHEL) | Planned | Linux | Planned |

---

## Tooling & Ecosystem

| Tool | Status | Purpose | Notes |
|------|--------|---------|-------|
| **Compiler Tools** |
| `eshkol-compile` | Yes | Ahead-of-time compiler | Produces executables |
| `eshkol-run` | Yes | Script runner | Compile + execute |
| `eshkol-repl` | Yes | Interactive shell | JIT-based with stdlib |
| `eshkol-pkg` | Yes | Package manager | Registry support |
| `eshkol-lsp` | Yes | Language server | IDE integration |
| **Development Tools** |
| Syntax highlighter | Yes | Editor support | VS Code extension |
| LSP server | Yes | IDE integration | Diagnostics, completion |
| Debugger | Planned | Interactive debugging | Planned |
| Profiler | Planned | Performance analysis | Planned |
| **Documentation** |
| API Reference | Yes | Complete | 555+ builtins |
| Quickstart Guide | Yes | Tutorial | 15-minute intro |
| Architecture Guide | Yes | Internals | System design |
| Type System Guide | Yes | HoTT types | Dependent types |
| Examples | Yes | Demo programs | Neural networks, physics, ML |
| **Testing** |
| Unit tests | Yes | Component tests | 426 files |
| Integration tests | Yes | End-to-end | Full programs |
| AD verification | Yes | Numerical validation | Gradient checking |
| Benchmark suite | Yes | Performance tracking | GPU + CPU benchmarks |

---

## ML & AI Capabilities

| Feature | Status | Level | Applications |
|---------|--------|-------|--------------|
| **Neural Networks** |
| Forward pass | Yes | Production | Any architecture |
| Backpropagation | Yes | Production | Via `gradient` |
| Activation functions | Yes | Production | 14 builtins: relu, sigmoid, softmax, gelu, silu, mish, etc. |
| Loss functions | Yes | Production | 14 builtins: MSE, cross-entropy, focal, triplet, etc. |
| Optimizers | Yes | Production | SGD, Adam, AdamW, RMSprop, Adagrad (builtins) + stdlib |
| Weight initialization | Yes | Production | xavier, kaiming, lecun (5 builtin initializers) |
| LR schedulers | Yes | Production | cosine-annealing, step-decay, warmup, exponential |
| **Supported Architectures** |
| Feedforward | Yes | Production | Fully connected |
| CNN | Yes | Production | conv1d/2d/3d, max-pool2d, avg-pool2d, batch/layer norm |
| RNN | WIP | Prototype | Sequential processing |
| Transformer | Yes | Production | scaled-dot-attention, multi-head, RoPE, positional-encoding |
| **Training Features** |
| Batch training | Yes | Production | Via user code |
| Mini-batch SGD | Yes | Production | Via user code |
| Learning rate scheduling | Yes | Production | Via user code |
| Regularization | Yes | Production | L1/L2 in loss |
| Early stopping | Yes | Production | Via user code |
| **Model Operations** |
| Save/load weights | WIP | - | Via file I/O |
| Model serialization | Planned | - | Planned |
| ONNX export | Planned | - | Planned |
| **Datasets** |
| In-memory datasets | Yes | Production | Lists/tensors |
| Lazy loading | Planned | - | Planned |
| Data augmentation | Planned | - | Planned |

---

## Scientific Computing

| Domain | Status | Features | Examples |
|--------|--------|----------|----------|
| **Numerical Analysis** |
| Root finding | Yes | Newton-Raphson | lib/math.esk |
| Integration | Yes | Simpson's rule | lib/math.esk |
| Interpolation | Planned | - | Planned |
| ODE solvers | Yes | RK4, Euler, Midpoint | math.ode |
| PDE solvers | WIP | Finite differences | Via user code |
| **Linear Algebra** |
| Matrix operations | Yes | Full suite | matmul, transpose, trace |
| LU decomposition | Yes | Pure Eshkol | lib/math.esk |
| Matrix inverse | Yes | Gauss-Jordan | lib/math.esk |
| Linear systems | Yes | Gaussian elim | lib/math.esk |
| Eigenvalues | Yes | Power iteration | lib/math.esk |
| **Statistics** |
| Descriptive stats | Yes | mean, variance, std | lib/math.esk |
| Covariance | Yes | Vector covariance | lib/math.esk |
| Distributions | Yes | Normal, Poisson, Binomial, etc. | math.statistics |
| Hypothesis testing | Planned | - | Planned |
| **Optimization** |
| Gradient descent | Yes | Via `gradient` | ml.optimization |
| Adam optimizer | Yes | Adaptive moments | ml.optimization |
| L-BFGS | Yes | Two-loop recursion | ml.optimization |
| Conjugate gradient | Yes | Fletcher-Reeves | ml.optimization |
| Newton's method | Yes | Via `hessian` | Second-order |
| Constrained optimization | Planned | - | Planned |
| **Physics Simulation** |
| Vector calculus | Yes | ∇, ∇·, ∇×, ∇² | Full support |
| Field theory | Yes | Differential forms | curl, divergence |
| Heat equation | Yes | Via Laplacian | Verified |
| Wave propagation | WIP | - | Via user code |
| Fluid dynamics | Planned | - | Planned |

---

## Signal Processing (v1.1)

| Feature | Status | Module | Notes |
|---------|--------|--------|-------|
| **Window Functions** |
| Hamming window | Yes | `signal.filters` | w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1)) |
| Hann window | Yes | `signal.filters` | w[n] = 0.5*(1 - cos(2*pi*n/(N-1))) |
| Blackman window | Yes | `signal.filters` | 3-term Blackman |
| Kaiser window | Yes | `signal.filters` | Parametric beta, inline Bessel I0 |
| **Convolution** |
| Direct convolution | Yes | `signal.filters` | O(N*M) time-domain |
| FFT convolution | Yes | `signal.filters` | O(N log N) via fft/ifft |
| **Filters** |
| FIR filter | Yes | `signal.filters` | Arbitrary coefficient application |
| IIR filter | Yes | `signal.filters` | Direct Form I |
| Butterworth lowpass | Yes | `signal.filters` | Bilinear transform |
| Butterworth highpass | Yes | `signal.filters` | Frequency inversion |
| Butterworth bandpass | Yes | `signal.filters` | Two-stage cascade |
| **Analysis** |
| Frequency response | Yes | `signal.filters` | Magnitude + phase at N points |
| FFT | Yes | Builtin | Cooley-Tukey radix-2 |
| IFFT | Yes | Builtin | Inverse FFT |

---

## Consciousness Engine (v1.1)

| Feature | Status | Module | Notes |
|---------|--------|--------|-------|
| **Logic Programming** |
| Unification | Yes | Builtin | `unify`, `walk` |
| Substitutions | Yes | Builtin | `make-substitution` |
| Knowledge base | Yes | Builtin | `make-kb`, `kb-assert!`, `kb-query` |
| Logic variables | Yes | Builtin | `?x` syntax |
| **Active Inference** |
| Factor graphs | Yes | Builtin | `make-factor-graph`, `fg-add-factor!` |
| Belief propagation | Yes | Builtin | `fg-infer!` |
| CPT mutation | Yes | Builtin | `fg-update-cpt!` |
| Free energy | Yes | Builtin | `free-energy`, `expected-free-energy` |
| **Global Workspace** |
| Workspace creation | Yes | Builtin | `make-workspace` |
| Module registration | Yes | Builtin | `ws-register!` |
| Softmax competition | Yes | Builtin | `ws-step!` |

---

## GPU Acceleration (v1.1)

| Feature | Status | Backend | Notes |
|---------|--------|---------|-------|
| **Metal (Apple Silicon)** |
| Elementwise operations | Yes | Metal | SF64 software float64 |
| Matrix multiplication | Yes | Metal | Ozaki-II adaptive N |
| Reduce operations | Yes | Metal | Sum, max, min |
| Softmax | Yes | Metal | Numerically stable |
| Transpose | Yes | Metal | 2D matrix transpose |
| **CUDA (NVIDIA)** |
| Elementwise operations | Yes | CUDA | cuBLAS integration |
| Matrix multiplication | Yes | CUDA | cuBLAS GEMM |
| Reduce operations | Yes | CUDA | Custom kernels |
| Softmax | Yes | CUDA | Numerically stable |
| Transpose | Yes | CUDA | cuBLAS transpose |
| **Dispatch** |
| Automatic CPU/GPU selection | Yes | Runtime | Cost model based |
| Threshold-based dispatch | Yes | Runtime | XLA → cBLAS → SIMD → scalar |

---

## XLA Backend (v1.1)

| Feature | Status | Mode | Notes |
|---------|--------|------|-------|
| StableHLO/MLIR path | Yes | When MLIR available | Hardware-optimized |
| LLVM-direct path | Yes | Default | Hand-tuned IR |
| Matmul fusion | Yes | Both | Fused multiply-add |
| Elementwise fusion | Yes | Both | Operation chains |
| Reduce operations | Yes | Both | Sum, max, min |
| Transpose | Yes | Both | Shape operations |

---

## Interoperability

| Interface | Status | Direction | Notes |
|-----------|--------|-----------|-------|
| **C Integration** |
| Call C functions | Yes | Eshkol → C | extern declarations |
| Access C globals | Yes | Eshkol → C | extern-var |
| C calls Eshkol | Planned | C → Eshkol | Planned callback API |
| **Python Integration** |
| Call Python from Eshkol | Planned | - | Planned (ctypes/cffi) |
| Call Eshkol from Python | Planned | - | Planned (wrapper lib) |
| NumPy interop | Planned | - | Planned (array protocol) |
| **Data Formats** |
| JSON | Yes | - | Parse and generate |
| CSV | Yes | - | Read and write |
| Base64 | Yes | - | Encode and decode |
| MessagePack | Planned | - | Planned |
| Protocol Buffers | Planned | - | Planned |
| **Databases** |
| SQLite | Planned | - | Planned |
| PostgreSQL | Planned | - | Planned |

---

## Comparison with Other Languages

| Feature | Eshkol | Python | Julia | Haskell | Scheme |
|---------|--------|--------|-------|---------|--------|
| **Language Type** |
| Paradigm | Functional-first | Multi-paradigm | Multi-paradigm | Purely functional | Functional |
| Type System | Gradual + Dependent | Dynamic | Dynamic | Static | Dynamic |
| Memory Model | OALR (regions) | GC | GC | GC | GC |
| **Automatic Differentiation** |
| Built-in AD | Yes 3 modes | No (libraries) | Yes (libraries) | No (libraries) | No |
| Forward-mode | Yes Dual numbers | JAX, PyTorch | ForwardDiff.jl | ad | No |
| Reverse-mode | Yes Tape-based | JAX, PyTorch | Zygote.jl | - | No |
| Symbolic | Yes Compile-time | SymPy | Symbolics.jl | - | No |
| **Performance** |
| Native compilation | Yes (LLVM) | No (CPython) | Yes (LLVM) | Yes GHC | No (most) |
| JIT available | Yes REPL | No (CPython) | Yes | No | No (most) |
| Zero-copy views | Yes | Yes (NumPy) | Yes | No | No |
| Tail call optimization | Yes | No | Yes | Yes | Yes |
| **Ease of Use** |
| Interactive REPL | Yes | Yes | Yes | Yes | Yes |
| Package manager | Yes eshkol-pkg | Yes pip | Yes Pkg | Yes cabal | Varies |
| IDE support | Yes LSP | Yes | Yes | Yes | Yes |
| Learning curve | Medium | Low | Medium | High | Medium |

---

## Test Coverage Summary

| Category | Test Files | Status | Notes |
|----------|-----------|--------|-------|
| **Core Language** | 80+ | Yes | All special forms verified |
| **List Processing** | 60+ | Yes | Comprehensive coverage |
| **Automatic Differentiation** | 50+ | Yes | All 3 modes validated |
| **Tensors** | 30+ | Yes | N-D operations verified |
| **Neural Networks** | 10+ | Yes | Training loops work |
| **Standard Library** | 40+ | Yes | All modules tested |
| **Type System** | 15+ | Yes | HoTT types validated |
| **Memory Management** | 20+ | Yes | Arena correctness |
| **System Integration** | 15+ | Yes | File I/O, system calls |
| **REPL/JIT** | 10+ | Yes | Cross-eval persistence |
| **Total** | **426** | **Yes** | **High confidence** |

---

## Roadmap

> This section is a historical snapshot and may lag; see the canonical,
> continuously-updated [ROADMAP.md](../ROADMAP.md) for current status. As of
> v1.3.1, v1.1-accelerate, v1.2-scale, and v1.3.0-evolve have all shipped.

### v1.1-accelerate (Q1 2026) — COMPLETED

- **GPU Support**: Metal (Apple Silicon) + CUDA (NVIDIA)
- **XLA Backend**: StableHLO/MLIR + LLVM-direct
- **Parallel Primitives**: parallel-map, parallel-fold, future/force
- **Exact Arithmetic**: Bignums, rationals, full numeric tower
- **Consciousness Engine**: Logic, inference, workspace (22 builtins)
- **Signal Processing**: FFT, filters, window functions
- **Optimizers**: Adam, L-BFGS, conjugate gradient in stdlib
- **R7RS Extensions**: call/cc, dynamic-wind, bytevectors, let-syntax

### v1.2-scale (Q2 2026) — SHIPPED

- **Data I/O**: Image/audio I/O, typed buffers, streams, DataFrame, plotting
- **Vulkan Compute**: Cross-platform GPU backend, multi-GPU
- **Model Deployment**: Serialization, ONNX export, quantization
- **Python Bindings**: Call Eshkol from Python and vice versa
- **Distributed Training**: AllReduce, MPI, gRPC

### v1.3-evolve (Q3 2026) — SHIPPED as v1.3.0-evolve

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

### Production-Ready (v1.1)

- Core language (39 special forms, 555+ builtins)
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

### Beta Quality

- FFI (works but callback registration planned)
- Quantum RNG (external dependency)
- XLA StableHLO path (requires MLIR, LLVM-direct is default)

### Not Yet Production

- Distributed computing
- Model serialization/ONNX export
- Vulkan Compute

---

## Dual Backend Architecture (v1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| **Bytecode VM** |
| 64-opcode core ISA | Yes | Register+stack architecture, computed-goto dispatch |
| 550+ native call IDs | Yes | Math, string, IO, complex, rational, bignum, dual, AD, tensor, logic, inference, workspace, hash, bytevector, parameter |
| ESKB binary format | Yes | Section-based layout, LEB128 encoding, CRC32 checksums |
| `-B` flag (bytecode emission) | Yes | `eshkol-run input.esk -B output.eskb` |
| VM compiler integration | Yes | eshkol_vm.c linked into compiler build |
| Closures & upvalues | Yes | Closure creation, open/close upvalues, mutable captures |
| call/cc & dynamic-wind | Yes | Continuation capture, wind stack |
| guard/raise exceptions | Yes | Handler stack with continuation restore |
| Variadic functions | Yes | OP_PACK_REST for rest parameters |
| Tensor matmul parity | Yes | v1.3.4: `arange` (1/2/3-arg), nested-literal tensor operands, and multi-dimensional `tensor-ref`/`tensor-set!` compute the same answers as native codegen (parity corpus `31_tensor_matmul`) |
| Shortest-round-trip number printing | Yes | v1.3.4: `display`/`write`/`number->string` share one portable-C routine with native, byte-identical output (R7RS 6.2.6) |
| Reverse-mode `gradient` (`op:GRADIENT`) | WIP | Implementation in progress; VM AD surface is currently scalar `derivative` only, full parity targeted for this release line |
| Checked ascription `(the <type> expr)` | No | native-only-justified: compile-time type-checker construct, runtime no-op — a VM program that omits it computes the identical result |
| **Weight Matrix Transformer** |
| Transformer interpreter | Yes | d_model=256, 6 layers, FFN_DIM=2304, 12.22M params |
| 3-way verification | Yes | Reference = simulated = matrix-based (126/126 inline, 123/123 traced) |
| QLMW binary export | Yes | For qLLM weight loading |
| 82 canonical opcodes in weights | Yes | `OP_NATIVE_CALL` remains the external dispatch boundary |
| **qLLM Bridge** |
| Eshkol↔qLLM tensors | Yes | Type conversion (double↔float32) with AD integration |
| Web Platform | Complete | WebAssembly compilation, 59 DOM bindings, browser REPL, eshkol.ai |
| VM Dual Number AD | Complete | Forward-mode AD via dual numbers in bytecode VM |
| VM Production | Complete | 176/176 tests, zero stubs, zero stdout contamination |
| KB Pattern Matching | Complete | Knowledge base queries with ?-wildcard pattern matching |

## Tensor Linear Algebra (v1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| `tensor-cholesky` | Yes | Cholesky decomposition |
| `tensor-lu` | Yes | LU decomposition |
| `tensor-qr` | Yes | QR decomposition |
| `tensor-svd` | Yes | Singular value decomposition |
| `tensor-solve` | Yes | Linear system solver |
| `tensor-det` | Yes | Determinant |
| `tensor-inverse` | Yes | Matrix inverse |
| `tensor-cov` | Yes | Covariance matrix |
| `tensor-corrcoef` | Yes | Correlation coefficient matrix |

## Data Loading (v1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| `make-dataloader` | Yes | Create batched data iterator |
| `dataloader-next` | Yes | Get next batch |
| `dataloader-reset` | Yes | Reset to beginning |
| `dataloader-length` | Yes | Total number of batches |
| `dataloader-has-next` | Yes | Check if more batches available |
| `train-test-split` | Yes | Split dataset into train/test |

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
- 555+ builtins, 37 test suites, 528 self-reported tests (87/87 v1.2 edge cases)

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

**Last Updated**: 2026-07-08
**Document Version**: 1.3.3

For detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md)
