# Eshkol v1.1-accelerate Test Coverage

**Version**: v1.1.12-accelerate
**Last Updated**: 2026-04-07
**Status**: 35 suites, 438 tests, 100% pass rate

**Additional verification**: Bytecode VM passes 331/332 tests (99.7%), weight matrix transformer passes 55/55 tests with 3-way verification.

---

## Test Execution

All 35 suites are orchestrated by `scripts/run_all_tests.sh`, which invokes each
suite script in sequence, aggregates pass/fail counts, and reports individual
failing tests at the bottom of its output. Each suite can also be run
independently.

```bash
# Run all 35 suites
./scripts/run_all_tests.sh

# Run a single suite
./scripts/run_autodiff_tests.sh
```

Programs are compiled with `eshkol-run`, linked against `build/stdlib.o`, and the
resulting binary is executed. A test passes if it compiles, runs to completion
without segfault, and produces no `FAIL:` assertion lines in its output.

---

## Bytecode VM Tests

The bytecode VM (`lib/backend/eshkol_vm.c` unity build) has its own comprehensive test suite:

| Suite | Tests | Description |
|-------|-------|-------------|
| Built-in | 50 | 10 bytecode-level + 40 source-level with verified output capture |
| Stress | 64 | Arithmetic, strings, lists, closures, control flow, AD, rationals |
| Final Verification | 62 | Adversarial edge cases across all features |
| **Total** | **176** | **All passing** |

Test command:
```bash
gcc -O2 -std=c11 -w lib/backend/eshkol_vm.c -o test_vm -lm -lpthread
ESHKOL_VM_NO_DISASM=1 ./test_vm
```

Coverage includes: arithmetic (int/float/rational/complex/bignum), strings (append/ref/substring/upcase/split/join), lists (map/filter/fold/sort/assoc/member), closures (capture/mutation/composition), control flow (call/cc/guard/raise/dynamic-wind/values), automatic differentiation (derivative/gradient via dual numbers), consciousness engine (KB/factor-graph/workspace), and R7RS forms (let/letrec/named-let/do/cond/case-lambda/quasiquote).

---

## Test Suite Overview

| # | Suite | Script | Tests | Coverage |
|---|-------|--------|------:|----------|
| 1 | Features | `run_features_tests.sh` | 24 | R7RS wave 2/3, bytevectors, bitwise ops, char/type predicates, trigonometric/hyperbolic functions, pattern matching, stress tests |
| 2 | Stdlib | `run_stdlib_tests.sh` | 6 | Standard library functions: CSV parsing, string operations with closures, list queries, refactoring regression |
| 3 | Lists | `run_list_tests.sh` | 129 | Cons cells, car/cdr, map, filter, fold, for-each, apply, assoc, variadic operations, homoiconicity, arena cons, tagged cons (Phase 3B) |
| 4 | Memory | `run_memory_tests.sh` | 6 | OALR linear types, double-move detection, use-after-move detection, borrow checker, arena regions |
| 5 | Modules | `run_modules_tests.sh` | 5 | `require`/`provide`, symbol visibility, circular dependency detection, stdlib module loading |
| 6 | Types | `run_types_tests.sh` | 13 | HoTT type system, type annotations, type introspection, predicate matrix, symbol ops, list element types, hash table types, mixed-type stress |
| 7 | Type System | `run_typesystem_tests.sh` | 8 | Static type checker: strict/gradual mismatch detection, unsafe suppression, arity checking, negative indexing, backward inference, nested expression checking, false-positive avoidance |
| 8 | Autodiff | `run_autodiff_tests.sh` | 52 | Forward-mode AD, reverse-mode AD, gradient composition, Jacobian, Hessian, vector calculus, dual numbers, computational graph construction, backward pass validation, tensor-AD integration |
| 9 | ML | `run_ml_tests.sh` | 37 | Tensor operations, matmul, transpose, convolution, pooling, activations (SiLU, SIMD), statistics, random tensors, covariance/correlation, bitwise tensor ops, optimizers, loss functions, weight initialization, LR schedulers, dataloaders, gradient utilities, transformers, linear algebra |
| 10 | Neural | `run_neural_tests.sh` | 6 | Neural network layers, forward pass, backpropagation training, complete network construction |
| 11 | JSON | `run_json_tests.sh` | 3 | JSON parsing, serialization, file I/O, large-document stress tests |
| 12 | System | `run_system_tests.sh` | 13 | Hash tables (create, ref, set, int keys), display system, file I/O (inline, port-based, stress, trace) |
| 13 | Complex | `run_complex_tests.sh` | 2 | Complex number arithmetic (Smith's formula division), FFT |
| 14 | C++ Types | `run_cpp_type_tests.sh` | 2 | C++ unit tests for HoTT type checker and type system internals (compiled and run as native C++) |
| 15 | Parser | `run_parser_tests.sh` | 12 | Special forms, numeric literals, comments, s-expression roundtrip, string escapes, function shadowing, edge cases, `define`-in-`begin`, `letrec*` regression, nested `define`-in-`if` |
| 16 | Control Flow | `run_control_flow_tests.sh` | 10 | `if`/`cond`, loops, recursion, `call/cc`, `dynamic-wind`, `call/cc`+`dynamic-wind` interaction, continuation edge cases, TCO validation, deep recursion (512 MB stack) |
| 17 | Logic | `run_logic_tests.sh` | 6 | Unification, logic variables, knowledge base (`kb-assert!`/`kb-query`), factor graphs (`fg-add-factor!`/`fg-infer!`), global workspace (`ws-register!`/`ws-step!`), continuous learning |
| 18 | Bignum | `run_bignum_tests.sh` | 2 | Arbitrary-precision integer arithmetic, edge cases (overflow, sign, exponentiation, mixed exact/inexact) |
| 19 | Rational | `run_rational_tests.sh` | 3 | Exact fraction arithmetic, `rationalize`, comprehensive rational operations, comparison, conversion |
| 20 | Parallel | `run_parallel_tests.sh` | 6 | `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`, `parallel-execute`, futures |
| 21 | Signal | `run_signal_tests.sh` | 2 | DSP filters, comprehensive FFT (windowing, spectral analysis) |
| 22 | Optimization | `run_optimization_tests.sh` | 1 | ML optimizer convergence: gradient descent, Adam, conjugate gradient |
| 23 | Examples | `run_examples_tests.sh` | 0 | End-to-end example programs (compile + run validation); directory currently empty |
| 24 | XLA | `run_xla_tests.sh` | 12 | XLA/StableHLO tensor operations: matmul (basic, large, accuracy, special), transpose, shape ops, elementwise, reduce, dispatch threshold |
| 25 | GPU | `run_gpu_tests.sh` | 12 | Metal/CUDA dispatch: elementwise, reduce, matmul, transpose, softmax/normalize, scale correctness, sf64 primitives (uniform/non-uniform), large matmul, diagnostics |
| 26 | Error Handling | `run_error_handling_tests.sh` | 7 | `guard`/`raise`, advanced guard patterns, nested exceptions, bounds checking, division by zero, edge cases, stack overflow detection |
| 27 | Macros | `run_macros_tests.sh` | 5 | `define-syntax`, `syntax-rules`, hygiene, `let-syntax`, nested patterns |
| 28 | REPL | `run_repl_tests.sh` | 20 | Interactive evaluation: arithmetic, comparisons, booleans, conditionals, variables, functions, lambdas, `let`, lists, closures, stdlib, math, autodiff, vectors, types, complex numbers, REPL commands, hot reload, type predicates, stdlib combined |
| 29 | Web | `run_web_tests.sh` | 2 | HTTP client (`web-extern`), canvas rendering |
| 30 | TCO | `run_tco_tests.sh` | 1 | Tail call optimization: 7 nested TCO patterns (mutual recursion, letrec, continuation-passing, accumulator, trampoline, A-normal form, CPS transform) |
| 31 | I/O | `run_io_tests.sh` | 5 | String ports, `write`/`read` roundtrip, binary I/O, port edge cases |
| 32 | Benchmark | `run_benchmark_tests.sh` | 3 | Performance regression: timing harness, tensor SIMD benchmark, BLAS matmul benchmark |
| 33 | Migration | `run_migration_tests.sh` | 1 | Backward compatibility: pointer consolidation comprehensive test |
| 34 | Codegen | `run_codegen_tests.sh` | 2 | LLVM IR generation correctness: integer ops, floating-point ops |
| 35 | Numeric | `run_numeric_tests.sh` | 7 | Critical numeric regressions: bignum, rational, rounding, expt, min/max |

**Total**: 438 tests across 35 suites.

---

## Suite Descriptions

### 1. Features (`tests/features/`)

Language-level feature tests spanning R7RS compliance waves and stress testing. Wave 2
covers bytevectors, bitwise operations, character predicates, and type predicates.
Wave 3 adds trigonometric/hyperbolic functions, data encoding, port predicates,
environment system access, and quantum random. Stress tests exercise closure-autodiff
interaction, extreme mathematical operations, and mixed-feature scenarios. Pattern
matching and exception handling have baseline coverage here (with deeper coverage in
their dedicated suites).

Key files: `r7rs_wave2_test.esk`, `r7rs_wave3_comprehensive_test.esk`,
`bytevector_test.esk`, `bitwise_ops_test.esk`, `char_predicates_comprehensive_test.esk`,
`type_predicates_test.esk`, `trig_hyperbolic_test.esk`, `pattern_matching_test.esk`,
`extreme_stress_test.esk`

### 2. Stdlib (`tests/stdlib/`)

Validates the precompiled standard library (`build/stdlib.o`). Tests cover CSV
parsing, string operations with closure arguments, list query functions, and
refactoring regression (ensuring symbol renaming across modules preserves
semantics). All stdlib functions are compiled from `.esk` source in `lib/core/`
and linked via `LinkOnceODRLinkage`.

Key files: `csv_comprehensive_test.esk`, `strings_closure_test.esk`,
`query_only_test.esk`, `refactor_test.esk`

### 3. Lists (`tests/lists/`)

The largest suite (129 tests). Covers the complete cons cell implementation:
`car`, `cdr`, `cons`, `list`, `list*`, `append`, `reverse`, `map`, `filter`,
`fold-left`, `fold-right`, `for-each`, `assoc`, `member`, `apply`. Includes
tagged cons (Phase 3B), multi-list map, variadic operations, higher-order
function composition, homoiconicity (`lambda-sexpr`), arena-allocated cons cells,
binary search, and polymorphic list functions. Debug and phase-specific regression
tests ensure no regressions across the tagged-value cons cell evolution.

Key files: `comprehensive_list_test.esk`, `map_comprehensive_test.esk`,
`phase3_polymorphic_completion_test.esk`, `lambda_sexpr_homoiconic_test.esk`,
`for_each_test.esk`, `assoc_test.esk`, `higher_order_test.esk`

### 4. Memory (`tests/memory/`)

Tests the Ownership-Aware Linear Resource (OALR) memory system. Validates
compile-time detection of double-move errors, use-after-move violations, and
mutation during active borrows. Positive tests confirm that correct linear type
usage compiles and runs without error. Arena region tests verify scoped
allocation and deallocation.

Key files: `double_move.esk`, `use_after_move.esk`, `move_while_borrowed.esk`,
`valid_ownership.esk`, `region_test.esk`

### 5. Modules (`tests/modules/`)

Exercises the `require`/`provide` module system. Tests basic module loading,
symbol visibility enforcement (ensuring unexported symbols are inaccessible),
circular dependency detection (compile-time error), and standard library module
import. The precompiled module discovery system (`collect_all_submodules`) is
implicitly tested through `stdlib_test.esk`.

Key files: `module_test.esk`, `visibility_test.esk`, `visibility_fail_test.esk`,
`cycle_test.esk`, `stdlib_test.esk`

### 6. Types (`tests/types/`)

Validates the HoTT (Homotopy Type Theory) type system at runtime. Tests type
annotations, type introspection (`type-of`), the complete type predicate matrix
(`number?`, `string?`, `pair?`, `vector?`, `symbol?`, etc.), symbol operations,
list element type propagation, and hash table type handling. Includes 2 C++ unit
tests (`hott_types_test.cpp`, `type_checker_test.cpp`) that test the type
checker internals directly.

Key files: `hott_comprehensive_test.esk`, `type_predicate_matrix_test.esk`,
`type_annotations_test.esk`, `type_introspection_test.esk`, `symbol_ops_test.esk`

### 7. Type System (`tests/typesystem/`)

Static type checker diagnostic tests. Each test is designed to trigger (or
not trigger) specific compile-time type errors. Covers strict mode type mismatch,
gradual mode relaxation, `unsafe` suppression of type errors, arity mismatch
detection, negative index rejection, backward type inference, nested expression
type checking, and false-positive avoidance (ensuring valid programs are not
rejected).

Key files: `type_mismatch_strict_test.esk`, `type_mismatch_gradual_test.esk`,
`unsafe_suppresses_test.esk`, `arity_mismatch_test.esk`, `backward_inference_test.esk`,
`no_false_positive_test.esk`

### 8. Autodiff (`tests/autodiff/`)

Comprehensive automatic differentiation coverage across forward-mode (dual
numbers), reverse-mode (computational graph + backward pass), and vector calculus
(gradient, Jacobian, Hessian). Tests span the full AD pipeline: type detection,
graph construction, backward pass correctness, tensor integration, gradient
composition, closure capture under AD, let-binding interaction, and nested lambda
differentiation. Validation tests (`validation_01` through `validation_04`)
serve as a structured regression suite.

Key files: `validation_01_type_detection.esk`, `validation_02_graph_construction.esk`,
`validation_03_backward_pass.esk`, `validation_04_tensor_integration.esk`,
`phase4_vector_calculus_test.esk`, `test_gradient_composition.esk`,
`autodiff_edge_test.esk`

### 9. ML (`tests/ml/`)

Machine learning primitives and training infrastructure. Covers tensor
operations (matmul, transpose, dot product, reshape, arange), activations
(ReLU, sigmoid, tanh, SiLU with SIMD), convolution, pooling, statistics
(mean, variance, covariance, correlation), random tensor generation,
optimizers (SGD, Adam), loss functions (MSE, cross-entropy), weight
initialization (Xavier, He), learning rate schedulers, data loaders,
gradient utilities, and transformer building blocks (attention, layer norm).

Key files: `impressive_demo.esk`, `matmul_test.esk`, `transformer_test.esk`,
`optimizer_test.esk`, `loss_functions_test.esk`, `activations_comprehensive_test.esk`,
`linalg_test.esk`, `dataloader_test.esk`, `lr_scheduler_test.esk`

### 10. Neural (`tests/neural/`)

End-to-end neural network tests. Validates complete network construction,
forward pass computation, backpropagation training loops, and minimal
network configurations. These tests exercise the full stack from tensor
allocation through closure-based layer composition to gradient-driven
weight updates.

Key files: `nn_working.esk`, `nn_training.esk`, `nn_computation.esk`,
`nn_complete.esk`, `nn_minimal.esk`, `nn_simple.esk`

### 11. JSON (`tests/json/`)

JSON parsing, serialization, and file I/O. Tests cover parsing JSON strings
into Eshkol values, generating JSON from native data structures, reading/writing
JSON files, and stress testing with large documents.

Key files: `json_test.esk`, `json_file_io_test.esk`, `json_stress_test.esk`

### 12. System (`tests/system/`)

System-level operations: hash tables (creation, reference, mutation, integer
keys, simple and complex operations), display formatting, and file I/O through
multiple interfaces (inline, port-based, stress, trace). The file I/O tests
exercise both the R7RS port system and direct file operations.

Key files: `hash_test.esk`, `hash_ref_test.esk`, `hash_set_test.esk`,
`file_io_port_test.esk`, `file_io_stress_test.esk`, `system_test.esk`

### 13. Complex (`tests/complex/`)

Complex number arithmetic using the heap-allocated `{real:f64, imag:f64}`
representation (type tag 7). Tests cover addition, subtraction, multiplication,
division (Smith's formula for overflow-safe division), magnitude, and FFT
(Fast Fourier Transform) as a practical application.

Key files: `complex_arithmetic_test.esk`, `fft_test.esk`

### 14. C++ Types (`tests/types/*.cpp`)

Native C++ unit tests compiled and executed outside the Eshkol compiler pipeline.
These test the HoTT type checker implementation (`hott_types.cpp`) and type
checker internals (`type_checker.cpp`) directly, catching regressions in the C++
type system infrastructure that would not surface through `.esk` tests alone.

Key files: `hott_types_test.cpp`, `type_checker_test.cpp`

### 15. Parser (`tests/parser/`)

Parser edge cases and regression tests. Covers special forms (`let`, `letrec`,
`letrec*`, `begin`, `if`, `cond`, `define`), numeric literal formats, comments
(line and block), s-expression roundtrip fidelity, string escape sequences,
function shadowing, `define`-in-`begin` hoisting, `letrec*` body reordering
regression (the define-hoisting bug), and nested `define`-in-`if` rejection.

Key files: `special_forms_test.esk`, `letrec_star_regression_test.esk`,
`define_in_begin_test.esk`, `nested_define_in_if_test.esk`, `edge_cases_test.esk`,
`string_escapes_test.esk`

### 16. Control Flow (`tests/control_flow/`)

Control flow primitives from basic conditionals through first-class
continuations. Tests `if`/`cond` branching, loop constructs, general and
tail-recursive recursion, `call/cc` (first-class continuations), `dynamic-wind`
(entry/exit thunks), the interaction between `call/cc` and `dynamic-wind`,
continuation edge cases (multi-shot, escape), TCO validation, and deep
recursion under the 512 MB stack configuration.

Key files: `callcc_test.esk`, `dynamic_wind_test.esk`,
`callcc_dynamic_wind_test.esk`, `continuation_edge_test.esk`,
`tco_validation_test.esk`, `deep_recursion_test.esk`

### 17. Logic (`tests/logic/`)

Consciousness engine primitives. Tests logic variable creation and unification,
substitution walking, knowledge base construction and querying (`kb-assert!`,
`kb-query`), factor graph belief propagation (`fg-add-factor!`, `fg-infer!`,
`fg-update-cpt!`), global workspace theory (`ws-register!`, `ws-step!` with
softmax competition), free energy / expected free energy computation, and
continuous learning (CPT mutation with belief reconvergence).

Key files: `logic_var_test.esk`, `unification_test.esk`, `kb_test.esk`,
`inference_test.esk`, `workspace_test.esk`, `continuous_learning.esk`

### 18. Bignum (`tests/bignum/`)

Arbitrary-precision integer arithmetic via the C runtime dispatch layer
(`eshkol_bignum_binary_tagged`, `eshkol_bignum_compare_tagged`). Tests cover
basic bignum operations, edge cases (overflow from int64 to bignum, sign
handling, exponentiation via repeated squaring, mixed bignum+double
promotion per R7RS exact/inexact rules, `number->string`/`string->number`
roundtrip for large integers).

Key files: `bignum_test.esk`, `bignum_edge_cases_test.esk`

### 19. Rational (`tests/rational/`)

Exact fraction arithmetic (`rational.h`/`rational.cpp`). Tests rational
construction, arithmetic operations (add, subtract, multiply, divide),
comparison (via `eshkol_rational_compare_tagged_ptr`), conversion to/from
other numeric types, `rationalize` (R7RS), and comprehensive edge cases
(zero denominator, negative fractions, GCD reduction, mixed rational/integer
operations).

Key files: `rational_arithmetic_test.esk`, `rational_comprehensive_test.esk`,
`rationalize_test.esk`

### 20. Parallel (`tests/parallel/`)

Parallel execution primitives. Tests `parallel-map`, `parallel-fold`,
`parallel-filter`, `parallel-for-each`, `parallel-execute` (fork-join), and
futures (async/await). Worker functions use `LinkOnceODRLinkage` to avoid
duplicate symbol errors across compilation units.

Key files: `parallel_map_test.esk`, `parallel_fold_test.esk`,
`parallel_filter_test.esk`, `parallel_for_each_test.esk`,
`parallel_execute_test.esk`, `futures_test.esk`

### 21. Signal (`tests/signal/`)

Digital signal processing library tests. Covers DSP filters (FIR/IIR
construction and application) and comprehensive FFT testing (windowing
functions, spectral analysis, inverse FFT, frequency-domain operations).

Key files: `filters_test.esk`, `fft_comprehensive_test.esk`

### 22. Optimization (`tests/ml/`)

ML optimizer convergence tests. Validates that gradient descent, Adam,
and conjugate gradient optimizers converge on known objective functions
within expected iteration bounds.

Key files: `optimization_test.esk`

### 23. Examples (`examples/`)

End-to-end example program validation. The test script compiles and runs
every `.esk` file in the `examples/` directory, verifying that example
programs remain functional as the compiler evolves. The directory is
currently empty; examples will be populated as the v1.1 API stabilizes.

### 24. XLA (`tests/xla/`)

XLA (Accelerated Linear Algebra) backend tests via the StableHLO emitter.
Tests matmul at multiple scales (basic, large, accuracy, special cases),
transpose, shape operations, elementwise operations, reduce, and dispatch
threshold behavior (ensuring XLA is only invoked when tensor dimensions
exceed the cost-model threshold).

Key files: `matmul_test.esk`, `matmul_basic.esk`, `matmul_large.esk`,
`matmul_accuracy.esk`, `elementwise_test.esk`, `reduce_test.esk`,
`transpose_test.esk`, `shape_ops_test.esk`, `dispatch_threshold_test.esk`

### 25. GPU (`tests/gpu/`)

Metal (macOS) and CUDA (Linux) GPU compute dispatch tests. Validates
correctness of GPU-offloaded operations: elementwise arithmetic, reduce
(sum/product/min/max), matrix multiplication, transpose, softmax,
normalize, and scale. Includes sf64 (simulated float64 on GPU) primitive
tests for both uniform and non-uniform workloads, large matmul dispatch,
and diagnostic output. The cost model ensures GPU dispatch only when
`gpu_peak_gflops` (200) exceeds the CPU path.

Key files: `gpu_test.esk`, `elementwise_correctness_test.esk`,
`reduce_correctness_test.esk`, `matmul_correctness_test.esk`,
`transpose_correctness_test.esk`, `softmax_normalize_test.esk`,
`sf64_primitives_test.esk`, `gpu_diagnostic_test.esk`

### 26. Error Handling (`tests/error_handling/`)

Exception handling via R7RS `guard`/`raise`. Tests basic exception throwing
and catching, advanced guard clause patterns (multiple conditions,
re-raising), nested exception handlers, bounds checking (vector and string
index validation), division by zero detection, edge cases in guard
evaluation order, and stack overflow detection with graceful recovery.

Key files: `exception_test.esk`, `guard_advanced_test.esk`,
`exception_nesting_test.esk`, `bounds_check_test.esk`,
`division_by_zero_test.esk`, `guard_edge_test.esk`, `stack_overflow_test.esk`

### 27. Macros (`tests/macros/`)

Hygienic macro system tests. Covers `define-syntax`/`syntax-rules` basic
expansion, minimal macro usage, macro-free baseline, `let-syntax` (locally
scoped macros), and nested pattern matching within macro templates.

Key files: `basic_macro_test.esk`, `let_syntax_test.esk`,
`nested_pattern_test.esk`, `minimal_macro_test.esk`, `no_macro_test.esk`

### 28. REPL (`tests/repl/`)

Interactive REPL (Read-Eval-Print Loop) evaluation tests, run via the
LLJIT-based `eshkol-repl`. Tests are numbered sequentially and cover
arithmetic, comparisons, booleans, conditionals, variable binding,
function definition, lambdas, `let` forms, lists, closures, stdlib
access, math functions, autodiff, vectors, types, complex numbers,
REPL-specific commands, hot reload, type predicates, and stdlib
combined operations. The REPL JIT uses `CodeGenOptLevel::None` to
match precompiled stdlib ABI.

Key files: `01_arithmetic.esk` through `19_stdlib_combined.esk`

### 29. Web (`tests/web/`)

Web/HTTP client tests. Validates the extern-based HTTP client interface
and canvas rendering primitives. These tests exercise the FFI boundary
for web-oriented functionality.

Key files: `web_extern_test.esk`, `web_canvas_test.esk`

### 30. TCO (`tests/tco/`)

Tail call optimization validation. A single comprehensive test file
exercises 7 distinct TCO patterns: mutual recursion, `letrec`-bound
tail calls, continuation-passing style, accumulator pattern, trampoline
pattern, A-normal form, and CPS transform. Validates that
`tco_context_` save/restore in `binding_codegen.cpp` prevents nested
letrec TCO context corruption.

Key files: `nested_tco_test.esk`

### 31. I/O (`tests/io/`)

Input/output system tests. Covers string ports (in-memory I/O), `write`/`read`
roundtrip fidelity (ensuring serialized values can be parsed back), binary I/O
(byte-level read/write), and port edge cases (EOF handling, port predicate
dispatch with flag bits).

Key files: `string_port_test.esk`, `write_test.esk`, `read_test.esk`,
`binary_io_test.esk`, `port_edge_test.esk`

### 32. Benchmark (`tests/benchmark/`, `tests/benchmarks/`)

Performance regression tests. The timing harness validates compilation and
execution of benchmark infrastructure. SIMD and BLAS benchmarks verify that
tensor operations dispatch to optimized paths (Apple Accelerate AMX for
matmul, SIMD for elementwise) and produce correct results under performance
workloads.

Key files: `timing_test.esk`, `tensor_simd_benchmark.esk`,
`blas_matmul_benchmark.esk`

### 33. Migration (`tests/migration/`)

Backward compatibility tests ensuring that code written against earlier
versions of the tagged value ABI continues to function after internal
refactoring. The pointer consolidation test validates that the transition
from multiple pointer representations to a unified scheme preserves
runtime semantics.

Key files: `pointer_consolidation_comprehensive_test.esk`

### 34. Codegen (`tests/codegen/`)

LLVM IR generation correctness tests. Validates that the arithmetic
codegen layer (`ArithmeticCodegen`) produces correct LLVM IR for
integer operations (add, sub, mul, div, mod, comparisons) and
floating-point operations (add, sub, mul, div, transcendental
functions). Tests are organized under `codegen/arithmetic/`.

Key files: `arithmetic/integer_ops_test.esk`, `arithmetic/float_ops_test.esk`

---

## Supplementary Test Directories

The following directories contain tests that are exercised by existing suite
scripts or serve as regression/debug artifacts. They do not have dedicated
runner scripts but contribute to overall coverage:

| Directory | Files | Notes |
|-----------|------:|-------|
| `tests/autodiff_debug/` | 2 | AD debug/regression (run by autodiff suite) |
| `tests/closures/` | 1 | Closure edge cases (run by features suite) |
| `tests/collections/` | 2 | Empty collection and tensor edge cases |
| `tests/integration/` | 1 | Cross-subsystem integration (autodiff + tensor) |
| `tests/numeric/` | 7 | Numeric edge cases: infinity/NaN, integer boundaries, rational edges, mixed exact/inexact, complex edges, rounding, critical regressions |
| `tests/string/` | 3 | String operations: empty string edge cases, boundary conditions, comprehensive ops |

---

## Coverage by Subsystem

| Subsystem | Suites | Total Tests |
|-----------|-------:|------------:|
| Core Language (lists, closures, macros, control flow) | 6 | 176 |
| Type System (HoTT, predicates, static checker) | 3 | 23 |
| Numeric Tower (bignum, rational, complex) | 3 | 7 |
| Automatic Differentiation | 1 | 52 |
| Machine Learning (tensors, neural, optimizers) | 3 | 44 |
| I/O and Serialization (file I/O, JSON, string ports) | 3 | 11 |
| Module System and Stdlib | 2 | 11 |
| Memory Management (OALR, arenas) | 1 | 6 |
| Hardware Acceleration (GPU, XLA, parallel) | 3 | 30 |
| Consciousness Engine (logic, inference, workspace) | 1 | 6 |
| REPL and Interactive | 1 | 20 |
| Compiler Internals (parser, codegen, migration) | 3 | 15 |
| R7RS Features and Stress | 1 | 24 |
| Error Handling | 1 | 7 |
| Benchmarks | 1 | 3 |
| Web | 1 | 2 |

---

## See Also

- [V1.1 Scope](V1.1_SCOPE.md) -- Release scope and feature inventory
- [Feature Matrix](FEATURE_MATRIX.md) -- Implementation status per feature
- [Architecture](ESHKOL_V1_ARCHITECTURE.md) -- System architecture reference
