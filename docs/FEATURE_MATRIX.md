# Eshkol v1.3.5-evolve Feature Matrix

**Status Key** (table cells): `Yes` = Production | `WIP` = In Progress | `Planned` = Planned | `No` = Not Planned | `Partial` = Partially supported

This matrix lists every implemented and planned feature in the Eshkol ecosystem. Every **Production** feature is code-verified, with extensive test coverage (45 suites, 770 individual tests).

**Language surface count (canonical, reconciled 2026-08-26 against commit
`afbaaf5b` — doc-truth audit finding N4):** the declared language surface is
**1,108** constructs. Older cited figures — 1,091, 1,106 and 1,107 here and in
`docs/COMPILER_ROADMAP.md`, 1,078 in `.icc/architecture-model.yaml`,
"550+ built-in functions"/"39 special forms" in `README.md` — were each
correct on the day they were written but drifted as the surface grew; the
gate has ratcheted upward since. The "1,058 in ADR-0011 §2.1" citation from
the previous reconciliation pass was itself wrong: ADR-0011 is the
guest-collector adapter design and contains no surface-count section — there
is no §2.1 that states 1,058. The only correct citation for this number is
the coverage manifest itself: `tests/coverage/language_surface.json` (1,042
builtins + 116 special forms + 113 AST ops + 16 prelude), deduplicated by
name and with internal-only helpers excluded exactly the way
`scripts/language_coverage.py` already deduplicates it to compute the
number the coverage gate enforces: `tests/coverage/coverage_policy.json`
`baseline_surface_total` = `tests/coverage/execution_deficit.json`
`surface_total` = **1,108**, confirmed by a fresh run of
`scripts/run_language_coverage.sh` at `afbaaf5b` on 2026-08-26. This is the
figure this doc, README.md, and `.icc/architecture-model.yaml` now use
uniformly; `scripts/check_surface_counts.py` fails CI if any of them drift
from the manifest again.

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
| Mutual recursion TCO | Yes | Functions calling each other, in every tail spelling (`if`/`cond`/`case`/`when`/`unless`/`and`/`or`), at any pair of arities | LLVM `musttail`; 1e8 hops flat, JIT + AOT |
| Tail-transfer dispatcher | Yes | Mutual tail calls `musttail` cannot express: differing arities, and every non-AArch64 target | 1e8 hops flat; `mutual_tail_arity` + the harness's `aot-xfer` lane |

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
| Checked ascription `(the <type> expr)` | Yes | v1.3.4: trusted assertion to the checker; a provably disjoint ascription is reported (warning under gradual typing, fatal under `--strict-types`); still a runtime no-op (byte-identical IR) | Type checker |
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
| Higher-order derivatives | Yes | Forward | Nested differentials; every spelling agrees — `(derivative-n f x k)`, the nested-lambda form, and the curried form `(define df (derivative f))` … `(derivative df)` / `(derivative (derivative f))` (v1.3.4, ESH-0369: the returned closure is dual-transparent, so orders 1-3 are exact) |
| Math function support | Yes | Forward | sin, cos, exp, log, sqrt, tan, sinh, cosh, tanh, abs, pow |
| Dual arithmetic | Yes | Forward | +, -, *, / |
| `derivative` operator | Yes | Forward | 30+ tests |
| **Reverse-Mode AD** |
| Computational graphs | Yes | Reverse | Tape-based |
| Gradient computation | Yes | Reverse | `gradient` |
| Backpropagation | Yes | Reverse | Full backward pass |
| Nested gradients | Yes | Reverse | Exact for nested scalar `derivative` and for the direct vector-point gradient-of-gradient (the ESH-0096 shape); the *curried* `(define g (gradient f))` route raises `unsupported nested differentiation` instead of silently answering zeros (SW-05) — use `hessian` for exact second order |
| Double backward | Yes | Reverse | Second derivatives via `hessian` (tensor-literal, vector, and variable-bound points, #343) or nested scalar `derivative` |
| Jacobian matrices | Yes | Reverse | `jacobian` |
| Hessian matrices | Yes | Reverse | `hessian` |
| Tape stack (nesting) | Yes | Reverse | 32-level depth |
| AD-aware tensor ops | Yes | Reverse | vref, matmul work with AD nodes |
| `gradient` operator | Yes | Reverse | 40+ tests |
| `jacobian` operator | Yes | Reverse | 15+ tests |
| `hessian` operator | Yes | Reverse | 10+ tests |
| **Vector Calculus** |
| Divergence | Yes | Reverse (LLVM) / forward dual (VM) | ∇·F (trace of Jacobian); exact on both engines — `autodiff_codegen.cpp` on LLVM, and one forward dual seeded per input variable at `vm_native.c` case 753 on the bytecode VM (v1.3.5-evolve, #487). No step size anywhere |
| Curl | Yes | Reverse (LLVM) / forward dual (VM) | ∇×F (3D + generalized 2-forms); exact on both engines. The VM builds its 3×3 Jacobian from three closure calls where the former central difference needed six, and a gradient field's curl now returns exactly `#(0 0 0)` where the difference quotient returned `#(1.1e-09 -3.3e-09 0)` (v1.3.5-evolve, #487) |
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
| Lexical regions | Yes | Manual | `with-region`; unwinds correctly through `raise`/`call/cc` as of v1.3.4-evolve |
| Region handles | Yes | Manual | v1.3.4-evolve (#341): `region-open` / `region-close` / `region-open?`, a non-lexical surface over the same machinery. Generation-counted handles, so every stale token is detectably stale; out-of-order close is a defined cascade. `region-open?` is `vm-supported`; HANDLE reclamation is native-only (Stage-2), while `with-region` reclaims on both engines |
| Automatic per-iteration nursery | Yes | Automatic | ESH-0214e: a resident loop that mutates persistent state reclaims per iteration, matching `with-region`. Loop bodies containing a `gradient` op, a `set!` or a `tensor-set!` are excluded by design — scope those with a region handle or `with-region` |
| Global arena | Yes | Manual | Shared across functions |
| Region nesting | Yes | Manual | Stack-based |
| Zero-copy views | Yes | Automatic | reshape, slice, transpose |
| **Ownership** |
| Linear types | Yes | Advisory (not enforced) | `owned`, `move` markers accepted and type-checked as identity `Value`; no production enforcement path — corrected 2026-08-25, conformity audit item d6 |
| Borrow checking | Yes | Advisory annotation + one runtime check | `borrow` construct; the complete `BorrowChecker` flow checker (`lib/types/type_checker.cpp:407-729`) has zero production callers — only `emitBorrowMutationCheck`, a runtime header-flag test, actually runs. Wiring `BorrowChecker` into the type-checker walk is a BUILD ITEM (ADR-0001 §7, docs/design/adr/0001-oalr-concurrent-resident.md) — corrected 2026-08-25, conformity audit item d6 |
| Escape analysis | Yes | Compile-time | Region-based with conservative heap fallback |
| Reference counting | Planned | Runtime | Planned (`shared`, `weak-ref`) |
| **Garbage Collection** |
| Mark-sweep GC for Eshkol values | No | - | By design, permanently (arena-based instead) |
| Generational GC for Eshkol values | No | - | By design, permanently |
| Hosted guest collector (Python / CL heap in a region) | Planned | Runtime | Guest traces its own heap inside its own region on a declared budget; Eshkol traces nothing ([ADR-0011](design/adr/0011-guest-collector-adapter.md)) |

---

## Compilation & Runtime

| Feature | Status | Backend | Performance |
|---------|--------|---------|-------------|
| **Compiler** |
| S-expression parser | Yes | Recursive descent | Fast |
| Macro system | Yes | Hygenic macros | `define-syntax` |
| HoTT type checker | Yes | Bidirectional | Gradual typing |
| LLVM IR generation | Yes | LLVM 21 | 43,959 lines (`lib/backend/llvm_codegen.cpp`) |
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
| Stack traces | Yes | Source spans | Source-span diagnostics; `-g` emits DWARF for GDB/LLDB |
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
| **Event Loop** |
| Portable readiness loop | Yes | `make-event-loop`, `event-loop-poll`, `event-loop-close` | v1.3.4-evolve (ESH-0011): kqueue (BSD/macOS), epoll (Linux), IOCP + WSAPoll/PeekNamedPipe (Windows); fail-closed stub on WASM. `vm-supported` |
| Descriptor registration | Yes | `event-loop-add-fd!`, `event-loop-remove-fd!` | Readiness-style interest bits |
| Backend introspection | Yes | `event-loop-backend` | Reports the live backend |
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
| Hessian (n→1) | O(n²) | Exact: routed through the Taylor tower, identity with `(derivative-n f x 2)` in value and exactness |
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
| Windows | Yes | x86-64, ARM64 | Native Visual Studio 2022 + ClangCL/LLVM 21. x86-64 covers Lite, XLA and CUDA. ARM64 builds and passes the hosted `windows-arm64-lite` and `windows-arm64-xla` CI lanes (`windows-11-arm` runners) and ships those two packages; there is no mesh/self-verified ARM64 hardware coverage, and ARM64 CUDA is not advertised because NVIDIA does not ship the required toolkit |
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
| `eshkol-run` (AOT mode, `eshkol-run in.esk -o out`) | Yes | Ahead-of-time compiler | Produces executables |
| `eshkol-run` (JIT mode, `eshkol-run -r`) | Yes | Script runner | Compile + execute |
| `eshkol-repl` | Yes | Interactive shell | JIT-based with stdlib |
| `eshkol-pkg` | Yes | Package manager | Registry support |
| `eshkol-lsp` | Yes | Language server | IDE integration |
| **Development Tools** |
| Syntax highlighter | Yes | Editor support | VS Code extension |
| LSP server | Yes | IDE integration | Diagnostics, completion |
| Debugger | Planned | Interactive debugging | Planned |
| Profiler | Planned | Performance analysis | Planned |
| **Documentation** |
| API Reference | Yes | Complete | 1,042 builtins across a 1,108-construct declared surface (canonical count, see below) |
| Quickstart Guide | Yes | Tutorial | 15-minute intro |
| Architecture Guide | Yes | Internals | System design |
| Type System Guide | Yes | HoTT types | Dependent types |
| Examples | Yes | Demo programs | Neural networks, physics, ML |
| **Testing** |
| Unit tests | Yes | Component tests | 1,812 `.esk` files |
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
| Save/load weights | Yes | v1.2 | Via file I/O; `model_io_test` PASS (`ctest -R model_io_test`) — corrected 2026-08-25 from `WIP`, conformity audit item d4 |
| Model serialization | Yes | v1.2 | Native `.eshkol-model` serialiser (`lib/core/model_io.cpp`, 795 lines) |
| ONNX export | Yes | v1.2 | `lib/core/onnx_export.c` (239 lines); `tests/v1_2_edge_cases/onnx_export_test.esk` |
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
| **Python Integration** — see [reference/bindings/python.md](reference/bindings/python.md) |
| Call Python from Eshkol | Yes | v1.2 | Stable C FFI |
| Call Eshkol from Python | Yes | v1.2 | pybind11 bindings (`bindings/python/`) |
| NumPy interop | Yes | v1.2; lifetime-hardened v1.3.5 (#458) | Zero-copy array interop; an exported array now holds a strong reference to its owning `Context` via a NumPy capsule, so the array stays valid past the `Context` object's own lifetime (closes audit H1, SW-44) |
| **Data Formats** |
| JSON | Yes | - | Parse and generate |
| CSV | Yes | - | Read and write |
| Base64 | Yes | - | Encode and decode |
| MessagePack | Yes | Stable | `lib/core/msgpack.esk`; see `docs/reference/stdlib/msgpack.md` |
| Protocol Buffers | Planned | - | Planned |
| **Databases** |
| SQLite | Yes | v1.2 | `agent.sqlite` FFI module; see `docs/reference/agent/sqlite.md` |
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
| **Total** | **1,812** | **Yes** | **High confidence** |

---

## Roadmap

> This section is a historical snapshot and may lag; see the canonical,
> continuously-updated [ROADMAP.md](../ROADMAP.md) for current status. As of
> v1.3.4-evolve (shipped 2026-08-19), the entire v1.1-accelerate through
> v1.3.4-evolve line has shipped. The dates below were re-anchored to
> measured velocity in the v1.3.5 documentation wave — see ROADMAP.md's
> "Development workstreams" section for the six workstreams (W1-W6,
> including distributed computing as a first-class W6 track) every release
> from v1.3.5 now draws from.

### v1.1-accelerate (Q1 2026) — COMPLETED

- **GPU Support**: Metal (Apple Silicon) + CUDA (NVIDIA)
- **XLA Backend**: StableHLO/MLIR + LLVM-direct
- **Parallel Primitives**: parallel-map, parallel-fold, future/force
- **Exact Arithmetic**: Bignums, rationals, full numeric tower
- **Consciousness Engine**: Logic, inference, workspace (22 builtins)
- **Signal Processing**: FFT, filters, window functions
- **Optimizers**: Adam, L-BFGS, conjugate gradient in stdlib
- **R7RS Extensions**: call/cc, dynamic-wind, bytevectors, let-syntax

### v1.2-scale (May 2026) — SHIPPED

- **Data I/O**: Image/audio I/O, typed buffers, streams, DataFrame, plotting
- **Model Deployment**: `.eshkol-model` serialization (ESKB-extended binary format)
- **Python Bindings**: Stable C FFI + pybind11, NumPy zero-copy interop

Not part of the actual v1.2 ship (still ahead, on the W6/W3 ladder — see
ROADMAP.md): Vulkan Compute, model quantization, and distributed training
(AllReduce/MPI/gRPC). An earlier revision of this table listed those as
shipped in v1.2; they were not. **ONNX export is an exception, re-verified
2026-08-25 by direct execution** (see the "Not Yet Production" note below):
`onnx-export-tensor` is a real wired builtin producing valid ONNX files
that pass `onnx.checker.check_model` with bit-exact round-trip, 9/9 on the
Python validator — it did ship, and stays `Yes | v1.2` below. Only the
StableHLO export path (a different, MLIR-dependent capability) remains
gated on the locked interop-first sequence.

### v1.3-evolve (Jul 2026) — SHIPPED as v1.3.0-evolve through v1.3.4-evolve

- **Language Extensions**: Full R7RS library system, string interpolation, keyword arguments
- **Arbitrary-order AD**: all 13 phases (P0-P12) shipped in v1.3.0-evolve
- **Resident/quantum/high-precision wave**: opt-in Moonlab quantum stack,
  `core.dbsp`, linear `Qubit`, Ozaki-II exact GEMM, VM region-reclaim gap
  narrowed to the evacuator alone — shipped across v1.3.1 through
  v1.3.4-evolve (tagged 2026-08-19)

> **Corrected 2026-08-25 (conformity audit item d2, folded into the #464
> v1.3.5 wave's own restructure of this section):** this row previously
> also listed "Advanced Types: refinement types, effect types, higher-rank
> types, row polymorphism" and "Compiler Optimization: PGO, whole-program
> optimization, polyhedral loop optimization" as SHIPPED. None of the six
> exists in the tree (`grep -i 'Refinement\|EffectRow\|SMT\|polyhedral'` is
> empty; ROADMAP.md itself marks WPO `[ ]` and PGO `[~]`). These are real,
> intended capabilities, not deleted — they are BUILD ITEMs below (v1.9.0),
> targeted per ADR-0004 (types) and ADR-0007 (PGO/WPO/polyhedral), not
> claimed as shipped. See also the paragraph immediately below, which #464
> independently added making the same point.

Advanced type-theory items (refinement types, effect types, higher-rank
types, row polymorphism) did not ship in the v1.3 line; they remain staged
at v1.9.0-types. PGO remains a build-time scaffold (workload selection is
the open gap); whole-program optimization and polyhedral loop optimization
are not yet scheduled to a specific release.

### v1.3.5 — the consolidation release (target: late Sep 2026)

- **VM region evacuator**: `with-region` reclaims on the bytecode VM
  (flagship item)
- **Assurance wave 1**: harness CI lanes, oracle/ledger schema checks,
  self-verdict scanner
- **Performance wave 1**: benchmarks on Eshkol's own axes, published and
  reproducible

### v1.4.0-connection (target: Nov 2026)

- **Networking**: TCP/UDP/TLS sockets, Unix domain sockets, HTTP client/
  server, WebSocket, with linear resource management
- **Distributed computing (W6 spike)**: PJRT client spike, XLA multi-device
  single-host, native collectives over sockets
- **Interop wave 2**: exactness across the Python/NumPy boundary, silent-
  demotion CI gate, definition-of-done rule goes live

### v1.5.0-intelligence (target: Q1 2027)

- **Resident/DBSP spine**: `core.dbsp` GA, native PGO in the release
  workflow (BUILD ITEM, moved from v1.3-evolve per ADR-0007, conformity
  audit item d2: workload-selection + canonical merge step to close out
  the existing scaffold, `CMakeLists.txt:340-388`)
- **Neuro-Symbolic Bridge**: Soft unification, symbol embeddings, attention over KB
- **Advanced Neural**: LSTM/GRU cells
- **Distributed computing (W6)**: Tier-1 data-parallel + Tier-2 mesh
  bit-identity gate

### v1.7.0-synthesis (target: Q3-Q4 2027)

- **Program Synthesis**: Type-directed holes, neural-guided search
- **Advanced Neural**: Graph Neural Networks

### v1.9.0 (per ADR-0004/ADR-0007, ADR-0000 Stage 12)

- **Advanced Types (BUILD ITEM, moved from v1.3-evolve, ADR-0004)**: Refinement types, effect types, higher-rank types, row polymorphism — `NominalTypeId`/`TypeRef`/`EffectRowRef` do not exist yet; tracked as ADR-0000 Stage 12
- **Whole-Program Optimization + polyhedral loop optimization (BUILD ITEM, moved from v1.3-evolve, ADR-0007)**: no closed-world root analysis, no LTO/ThinLTO exists yet

### v2.0-starlight (target: Q4 2028)

- **Unified differentiation**: `differentiate` primitive over `numeric` and
  `incremental` interpretations
- **Quantum Computing**: QAOA and circuit-level optimisation (Qubit types with linear tracking, gates and VQE already SHIPPED in v1.3.3/v1.3.4-evolve — but note the "linear tracking" qualifier: see the Qubit row above, tracking is advisory not enforced, conformity audit item a7)
- **Formal Verification**: Lean kernel export re-checks the compiler on the
  normative corpus; "HoTT-inspired" retired as a claim; proof assistant
  integration and certified compilation, per ADR-0000 Stage 14b
- **Next-Gen Types**: Session types, algebraic effects, quantitative type theory
- **Distributed computing (W6) gates**: Tier 1 >=85% scaling efficiency at 8
  devices; Tier 2 bit-identical gradients at any node count

---

## Production Readiness

### Production-Ready (v1.1)

- Core language (116 special forms, 1,042 builtins — 1,108-construct canonical surface, see "Language surface count" below)
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

- Distributed computing (W6 spike targeted at v1.4.0; see ROADMAP.md)
- StableHLO export path specifically (requires MLIR; see "Beta Quality"
  above — separate from raw ONNX tensor export, which is production, see
  below)
- Vulkan Compute

(**Re-verified 2026-08-25 during rebase onto #464 (docs/conformity-audit-resolution
vs docs/v135-refresh):** the two PRs disagreed here — #464's merged text put
"ONNX/StableHLO export" in this list as "ships post-training-win, no fixed
date"; this branch's own conformity-audit fix (item d4) had instead removed
ONNX specifically, keeping only StableHLO. Re-checked by direct execution
rather than trusting either prior claim: `eshkol-run -r
tests/v1_2_edge_cases/onnx_export_test.esk` → 7/7 PASS, and the deeper
Python validator `tests/v1_2_edge_cases/onnx_export_test.py` (which calls
the real `onnx` package's `onnx.checker.check_model` and verifies bit-exact
value round-trip for 1D/2D/3D tensors) → **9/9 PASS**, both against a
from-source build at this branch's tip. `onnx-export-tensor` is a real,
wired native builtin (`lib/core/system_builtins.c:5028`), not a stub.
Conclusion: raw ONNX tensor export is production (kept as "Yes | v1.2" in
Model Operations below, `model_io_test` also PASS); only the *StableHLO*
export path — a different, MLIR-dependent capability — remains genuinely
not-yet-production, and is listed above accordingly.)

---

## Dual Backend Architecture (v1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| **Bytecode VM** |
| 67-opcode core ISA | Yes | Register+stack architecture, computed-goto dispatch; `OP_COUNT = 67` in `lib/backend/vm_core.c`, the enum `vm_run.c`'s dispatch table indexes — corrected 2026-08-28 from "66" after #491 appended `OP_GLOBAL_MARK = 66` without renumbering; corrected 2026-08-25 from "64" (conformity audit item d7; three other `OpCode` definitions elsewhere in `lib/backend/` disagree at 63, a separate ODR-cleanup code issue tracked independently of this doc) |
| 724 VM-reachable native call IDs | Yes | Math, string, IO, complex, rational, bignum, dual, AD, tensor, logic, inference, workspace, hash, bytevector, parameter; `tests/coverage/language_surface.json` `counts.builtins_in_vm_table` — corrected 2026-08-28 from "722"; corrected 2026-08-25 from "694" (conformity audit item d7) |
| ESKB binary format | Yes | Section-based layout, LEB128 encoding, CRC32 checksums |
| `-B` flag (bytecode emission) | Yes | `eshkol-run input.esk -B output.eskb` |
| VM compiler integration | Yes | eshkol_vm.c linked into compiler build |
| Closures & upvalues | Yes | Closure creation, open/close upvalues, mutable captures |
| call/cc & dynamic-wind | Yes | Continuation capture, wind stack |
| guard/raise exceptions | Yes | Handler stack with continuation restore |
| Variadic functions | Yes | OP_PACK_REST for rest parameters |
| Tensor matmul parity | Yes | v1.3.4: `arange` (1/2/3-arg), nested-literal tensor operands, and multi-dimensional `tensor-ref`/`tensor-set!` compute the same answers as native codegen (parity corpus `31_tensor_matmul`) |
| Shortest-round-trip number printing | Yes | v1.3.4: `display`/`write`/`number->string` share one portable-C routine with native, byte-identical output (R7RS 6.2.6) |
| Reverse-mode `gradient` (`op:GRADIENT`) | Yes | v1.3.4 (#337): forward/reverse-mode, arity-resolved (scalar / N-arg / arity-1 whole-vector) incl. the curried form, byte-identical to native across `vm-src`/`vm-eskb`; higher-order nesting (gradient-of-derivative / Taylor tower) stays native-only |
| Forward-mode `derivative` (`op:DERIVATIVE`) | Yes / first order | v1.3.4 (ESH-0369): direct `(derivative f x)` and curried `((derivative f) x)` both lower to the same native call with the same `(f, x)`, so they agree exactly with native. First order only — the VM's carrier is a flat dual `{value, tangent}` with a single perturbation, so nested/higher-order differentiation now **raises a catchable error** instead of returning the `0.0` it used to fabricate. Higher-order AD needs the native jet's `e1`/`e2`/`ep` slots or a VM Taylor tower; native-only until a VM jet carrier is built |
| R7RS module forms (`op:IMPORT` / `op:PROVIDE` / `op:REQUIRE`) | Yes | v1.3.4-evolve (#402): the VM lane previously knew none of `define-library` / `import` / `export`. All three move to `vm-supported` with no new waivers; three latent VM defects were fixed with them (`provide` emitting nothing via a slot shift, a module-loader POP desync, and a fail-open forward reference now refused) |
| Portable event loop | Yes | v1.3.4-evolve (ESH-0011): `make-event-loop`, `event-loop-poll`, `event-loop-add-fd!`, `event-loop-remove-fd!`, `event-loop-close`, `event-loop-backend` all `vm-supported` |
| `with-region` | Yes, including reclamation | `op:WITH_REGION` stays `native-only-justified` for ONE undocumented spelling (`(with-region (quote name))` with no other body), not for reclamation: the Stage-1 VM region evacuator (`lib/backend/vm_region_evac.c`, SW-14) brackets the body with native 2213/2214 and reclaims measurably — flat peak RSS at 26 MB across 1 000/4 000/16 000 iterations against 796 MB with the evacuator disabled. Every documented spelling is value-identical on both substrates |
| Region handles (`region-open` / `region-close`) | Surface yes, reclamation Stage-2 | `region-open?` is `vm-supported`; `region-open` and `region-close` are `native-only-justified`. The name resolves on both substrates and the handle protocol, its validation and every error message are byte-identical (one shared C implementation), so output parity holds — what is native-only is the reclamation. A handle can be closed out of order, from another dynamic extent, or never, whereas `with-region`'s lexical extent tells the teardown where the region ends; that is why the lexical form landed first. The VM announces the boundary at the point of use |
| Checked ascription `(the <type> expr)` | No | native-only-justified: compile-time type-checker construct, runtime no-op — a VM program that omits it computes the identical result. The contradiction diagnostic added in v1.3.4 is likewise compile-time, so runtime parity is unchanged |
| **Weight Matrix Transformer** |
| Transformer interpreter | Yes | d_model=256, 6 layers, FFN_DIM=2304, 12.22M params |
| 3-way verification | Yes | Reference = simulated = matrix-based (127/127 inline, 124/124 traced — corrected 2026-08-25 from "126/126, 123/123" by direct execution: `build/tools/weight_matrices` → `=== Results: 127 passed, 0 failed ===`; matches `docs/SDNC.md`, conformity audit item d8) |
| QLMW binary export | Yes | For qLLM weight loading |
| 82 canonical opcodes in weights | Yes | `OP_NATIVE_CALL` remains the external dispatch boundary |
| **qLLM Bridge** |
| Eshkol↔qLLM tensors | Yes | Type conversion (double↔float32) with AD integration |
| Web Platform | Complete | WebAssembly compilation, 59 DOM bindings, browser REPL, eshkol.ai |
| VM Dual Number AD | Complete | Forward-mode AD via dual numbers in bytecode VM |
| VM Production | Partial (documented subset) | Zero stubs, zero stdout contamination on the surface it implements, gated by the VM source suite, the 81/81 C-API suite, and the 188/188 differential gate (`scripts/run_vm_parity.sh`, remeasured 2026-08-25). But `tests/vm_parity/PARITY.tsv` carries 331 `gap` rows out of 956, plus 323 further names in `tests/vm_parity/SURFACE_BASELINE.tsv` outside that ledger entirely (see [VM_PARITY.md](VM_PARITY.md)) — corrected from "Complete" 2026-08-25, conformity audit item d9 |
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

This is a short summary list; the full, actively-maintained catalogue of
known issues and limitations (including engine-parity gaps, AD scope
caveats, and the items filed by the 2026-08-25 conformity audit) lives in
[docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md) — corrected 2026-08-25, conformity
audit item d12; numbering below also corrected (previously skipped item 2).

1. **Single GPU dispatch** - One GPU at a time (multi-GPU not yet scheduled)
2. **Small ecosystem** - Growing standard library, but not as extensive as Python/Julia
3. **Learning curve** - Functional programming + AD concepts require study
4. **Platform support** - Linux, macOS, and native Windows x64

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
- 555+ builtins, 37 test suites, 528 self-reported tests (87/87 v1.2 edge cases) [as of v1.1]

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

**Last Updated**: 2026-08-28
**Document Version**: 1.3.5-evolve

For detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md)
