# Changelog

All notable changes to Eshkol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Started v1.3-evolve with parser-level string interpolation: string literals
  can embed `~{expr}` forms, each expression is formatted with display
  semantics, and `~~{` preserves a literal interpolation opener.

- Added v1.3 keyword formal support for function and lambda parameters such as
  `#:scale scale`, plus `let-match` destructuring bindings for day-to-day
  pattern binding.

- Added the first v1.3 R7RS library surface: simple `(import (scheme base))`
  and `(define-library ... (export ...) (import ...) (begin ...))` forms now
  lower onto the existing module loader.

## [1.2.3-scale] - 2026-05-25

Packaging closeout for the v1.2 line. This patch supersedes the unpublished
`v1.2.2-scale` tag attempt, which failed before GitHub release publication in
the hosted Windows x64 artifact job.

### Fixed

- Fixed hosted Windows x64 release packaging links by keeping generated
  parallel worker initializer symbols module-local on native Windows.

## [1.2.2-scale] - 2026-05-25

Packaging closeout for the v1.2 line. This patch release keeps the v1.2.1
language/runtime surface and republishes through the guarded 16-asset platform
release workflow.

### Added

- Added `release_workflow_surface_test`, which checks that every `v*` tag
  publishes the complete Linux/macOS/Windows lite/XLA/CUDA asset set,
  generates `SHA256SUMS.txt`, and refuses to append to an existing release.

### Changed

- Updated release-facing version metadata and the Homebrew formula template to
  target `v1.2.2-scale`.

## [1.2.1-scale] - 2026-05-20

The v1.2-scale closeout point release. This release keeps the v1.2.0-scale
feature surface and closes the remaining downstream substrate blockers found
by Noesis aggregate validation.

### Added

- Added `examples/milli_mag_bohrification.esk`, a public executable sketch
  that keeps the milli-magnetic Bohrification model inside today's Eshkol
  surface and passes the examples suite.

### Fixed

- Closed the remaining Noesis-filed Eshkol issues W, Z, BB, GG, JJ, KK, and
  LL, with the Noesis tracker reconciled to zero open filed substrate bugs.
- Fixed the intermittent Noesis dual-neural crash by serializing runtime
  hash-table access.
- Fixed work-stealing external task submission so main-thread `parallel-map`
  producers cannot push into worker-owned Chase-Lev deques.
- Added the object-build CLI contract required by Noesis build integration:
  `--emit-object`, exact `-o path.o` handling, `--shared-lib`, `-fPIC`,
  `-I`, and `-D` compatibility.
- Added stdlib/filesystem closeout items including atomic output-file writes
  and JSON read/write aliases.
- Hardened the public release test harnesses: aggregate counting now includes
  `Results: N passed, M failed` suite summaries, I/O tests no longer depend on
  Perl timeout behavior, system tests default `BUILD_DIR=build`, and the HTTP
  server smoke has bounded timeout/client cleanup.
- Updated the Homebrew formula template to target the public `v1.2.1-scale`
  archive; the tap formula carries the computed release checksum after tagging.

### Verified

- `scripts/run_all_tests.sh` passes 37/37 suites and 528/528 self-reported
  individual tests.
- `tests/v1_2_edge_cases` passes 87/87.
- `build/test_vm_c_api` passes 81/81.
- `ctest --test-dir build --output-on-failure --timeout 180` passes 15/15.
- `scripts/run_stress_tests.sh` passes 3/3.
- Noesis `tests/smoke/all.esk` exits with `NOESIS_ALL_RC=0`.

## [1.2.0-scale] - 2026-05-01

The production-readiness release.  Closes 14 audit blockers,
finalises the v1.2 stdlib (json_schema, reflection, time API,
regex capture groups, memoization, PRNG seeding, lazy streams),
and lands the deep-architecture fixes that surfaced when the
edge-case suite was widened: parser line markers, stdlib LinkOnceODR
linkage, macOS stack-flag wiring, --wasm path separation, AD
scalar-derivative through runtime closures, value-typed-capture
LLVM verification, variadic-info hygiene on user redefines.
Master suite exits 0 across 37 sub-suites.

The detailed changelog runs from "Fixed — SDNC paper artifact
(weight_matrices.c)" below through the original 2026-04-24 release
notes a few hundred lines further down.  This date represents the
final v1.2.0-scale public tag; the 2026-04-24 entry is the
mid-cycle internal preview.

### Fixed — late-cycle quality (parser, codegen, AD)

- **Variadic→fixed redefine hygiene** (`bbfb357`).
  `createFunctionDeclaration` only ADDED to `variadic_function_info`
  on the variadic branch — the inverse case (redefining a
  previously-variadic name as fixed-arity) left the stale entry
  behind and call-site dispatch lowered with the wrong calling
  convention.  Symptom:
  `tests/features/ultimate_math_stress.esk`'s user
  `(define (gradient-descent f start lr iters))` (4 fixed) on top
  of stdlib's `(gradient-descent f x0 . opts)` compiled with a
  "no-capture call to gradient-descent.4 expected 4 got 3"
  warning and crashed at runtime.  Fix: erase the stale entry on
  the non-variadic branch.  Regression test:
  `tests/v1_2_edge_cases/redefine_variadic_to_fixed_test.esk`.

- **AD value-typed captures** (`ecb567d`).
  `derivativeHigherOrder` line ~2009 unconditionally
  `CreatePtrToInt`'d the resolved capture storage even when
  `storage` was a function-parameter Argument with `tagged_value`
  struct type — LLVM IR verification rejected this with
  "PtrToInt source must be pointer".  The previously-disabled
  new-style derivative body had the right case-split (preserved
  under `#if 0` for v1.3 re-extraction).  Fix: pack the pointer
  when storage is one, otherwise pass the value-typed
  tagged_value through a fresh alloca temp slot.  Reproducer:
  `tests/neural/{nn_working,nn_training}.esk`'s
  `compute-loss-gradient` capturing `input`/`target`/`b`.

- **Scalar derivative on runtime closures** (`1321a3f`, closes
  v1.3 task #215 "AD-1: scalar-derivative tape-state hygiene").
  `AutodiffCodegen::derivative()` was extracted from
  `codegenDerivativeMonolith` but missed the runtime-function-
  parameter handling: when `f` was a lambda passed as a function
  parameter (the common pattern, e.g.
  `(newton-solve (lambda (x) (- (pow x 2) 2)) 1.5 10)`), the
  callback returned nullptr and the new method bailed out
  without dispatching through the closure ABI.  The dispatcher
  propagated the null and the surrounding arithmetic produced
  -inf or wrong values.  Fix: have `derivative()` delegate to
  `codegenDerivativeMonolith` (which has the full path).  v1.3
  re-extraction will produce one shared implementation.
  Reproducer: Newton-Raphson sqrt(2) ≈ 1.25872 instead of
  1.41421.

- **`--wasm` no longer falls through to native link** (`151a026`).
  The WASM emit branch produces a self-contained .wasm via LLVM
  in-memory codegen, but the unconditional `compiled_files` link
  block then ran clang++ on the same .o files and failed with
  `Undefined symbols for architecture arm64: _main referenced
  from <initial-undefines>`.  Fix: gate the link block and its
  sibling "unused object files" warning on `!wasm_output`.
  Reproducer: `tests/web/{web_canvas_test,web_extern_test}.esk`
  succeeded in `eshkol_compile_llvm_ir_to_wasm_file()` but were
  marked compile-fail by the redundant native link below.  Web
  suite returns to 100%.

- **macOS `-Wl,-stack_size` on the compiled-files link path**
  (`38f0ca2`).  `exe/eshkol-run.cpp`'s compile-and-link path had
  Win32 + Linux stack-size guards but was missing the macOS
  `-Wl,-stack_size,0x20000000`.  Every binary built via the
  common `eshkol-run file.esk -o exe` flow shipped with
  `LC_MAIN.stacksize = 0` (i.e. linker default 8 MB on macOS).
  `lib/backend/llvm_codegen.cpp` already had the macOS branch on
  the parallel single-step link path; this commit mirrors it.
  Reproducer: `tests/tco/nested_tco_test.esk` Test 4 (3-level
  nesting, depth 10000 of non-tail-recursive `outer`) segfaulted
  in `eshkol_check_recursion_depth + 4` itself once the user
  stack was exhausted.  The same commit caps Test 4 at depth
  4000 — at -O0 each frame for that pattern is ~95 KB, and ARM64
  macOS hard-caps the stack at 512 MB.  Smaller per-frame size
  is v1.3 work.

- **Stdlib LinkOnceODR linkage** (`ce4ec65`).
  `createLibraryInitFunction` had a hardcoded
  `pair.second->setLinkage(GlobalValue::ExternalLinkage)` on
  macOS/Linux that overrode the LinkOnceODR linkage that
  `createFunctionDeclaration` had just set.  Result: every
  stdlib function shipped as a strong external symbol, so a
  user `(define (foo …))` with the same name as a stdlib
  function failed with `duplicate symbol _foo`.  Fix: both
  branches call `publicDefinitionLinkage(true)`
  (`LinkOnceODRLinkage` on macOS/Linux, `WeakAnyLinkage` on
  Windows).  After: `nm -m build/stdlib.o | grep vec-scale`
  shows `weak external`, user override works cleanly.

- **Parser line markers** (`5992fdb`, `e41957c`).
  `eshkol_parse_next_ast_from_stream` stripped comment lines
  *including their trailing newline* (`std::getline`) and started
  a fresh `SchemeTokenizer` at line 1 for every form — so
  `(undefined-fn …)` on file line 6 was reported as line 1:2.
  Two-part fix: (1) reader consumes comment body up to but not
  including `\n`, leaving the newline in `input` so the
  tokenizer's line counter stays accurate within a form; (2)
  thread-local `g_stream_line` / `g_stream_column` track
  cumulative file position across successive
  `eshkol_parse_next_ast_from_stream` calls and are passed to
  `SchemeTokenizer`'s constructor.  `eshkol_reset_parse_line_counter()`
  is called at every fresh parse session
  (load_file_asts, parse_string in REPL, parseAllAstsFromString
  in repl_jit, compile_to_wasm in eshkol-server).  Regression
  suite at `tests/v1_2_edge_cases/error_line_marker_test.sh` (5
  cases: top-of-file, post-comment, multi-line, nested-body,
  stdlib-loaded).

- **`core.json_schema` validator** (`7ef7753`, closes M1 task
  #172).  Draft 7 subset: type, properties, required,
  additionalProperties, items, min/max length/items,
  minimum/maximum (with exclusive variants), enum, const,
  pattern (substring containment), oneOf / anyOf / allOf / not.
  Auto-loaded via stdlib.  API: `(json-schema-valid? schema
  value)` returns boolean; `(json-schema-validate schema value)`
  returns a list of error strings carrying JSON-pointer-style
  paths.

- **cpp_type tests link cleanly** (`cda7b9d`).  The HoTT
  type-checker C++ tests now link against
  `build/libeshkol-static.a` + macOS frameworks (Accelerate,
  Metal, MetalPerformanceShaders, Foundation, Security,
  CoreFoundation, libobjc, libncurses, libpcre2-8, libsqlite3)
  instead of pulling raw .cpp sources by hand and missing
  `arena_strdup` / `arena_allocate_zeroed` / `get_global_arena`.
  Suite goes from "SOME C++ TYPE TESTS FAILED" (0/2) to PASSED
  (2/2, 61/61 internal asserts).

- **`visibility_fail_test` aligned with Bug Z** (`65cac3e`).  Bug
  Z (`1235e0a`) made `(provide …)` informational; this test had
  asserted the opposite (calling a non-`provide`d helper should
  error) and was failing.  Updated to document the new
  semantics; true module privacy is filed as v1.3 architectural
  work.  Modules suite back to 100%.

### Fixed — SDNC paper artifact (weight_matrices.c)

Three commits (`df2fabd`, `7b1b765`, `7301dc4`) restore the
reproducibility package for the SDNC paper ("The Self-Differentiating
Neural Computer: Computable Transformers via Analytical Weight
Construction", tsotchke 2026) and bring it from "matches outputs to
0.01 tolerance" to **bit-identical agreement at every step of every
program** between the reference C interpreter and the matrix forward
pass.

- **Restored** `lib/backend/weight_matrices.c` (3998 lines), the
  archive predecessors under `lib/backend/archive/`, four standalone
  binaries (`eshkol_benchmark`, `qllm_distributed`, `qllm_interpreter`,
  `stackvm_codegen`), and `inc/eshkol/bridge/qllm_bridge.h` — all
  required to regenerate `weights.qlmw` from the pinned commit.
  CMake now exposes a `weight_matrices` target gated on file
  existence and a CTest case `sdnc_paper_74_tests` asserting the
  three-way "74 passed, 0 failed" line.
- **Wired** the dump-trace + comparison pipeline end-to-end
  (`scripts/paper/{dump_vm_trace.sh, dump_transformer_trace.sh,
  compare_traces.py, gen_paper_tables.py, run_paper_suite.sh}`) so
  `run_paper_suite.sh` produces a real `comparison-report.json`,
  `opcode-coverage.json`, and four LaTeX tables instead of the
  previous TODO stubs.
- **Achieved bit-identical agreement (71/71 full per-step state)**
  by fixing five real bugs in the matrix encoding:
  - softmax temperature too low (`SCALE=100`→`300`) — attention
    residue of `~4.6e-16` was leaking into accumulators;
  - layer-4 forward tape-write missed the `AD_IS_FORWARD` gate —
    the comment promised it, the code never wired it;
  - dual-input AND gates required `10·SCALE` weight on the binary
    condition so the integer condition (max 7) couldn't dominate;
  - backward-pass cursor termination off-by-one (`indicator(c, -1)`
    fires one cycle late; fixed to `indicator(c, 0)`);
  - reference VM `ad_backward_step` uses direct `grad·saved` where
    the matrix architecture is forced to use polarisation
    `½·(a+b)² − ½·a² − ½·b²` (SQUARE-FFN limitation). Reference now
    uses the same polarisation arithmetic so float-order matches —
    the two are mathematically equal but differ by 1–13 ULPs in
    float32.
  Also: `pe[]` zero-init for out-of-bounds attention determinism,
  and a one-character fix to the `set-car!` test (`n=8`→`n=9` —
  the program array had 9 instructions but `n` was off by one).

### Added — bisection infrastructure

- New `--trace-vm`, `--trace-transformer`, and `--trace-simulated`
  CLI flags on the `weight_matrices` binary emit per-step JSONL
  traces with the schema consumed by `compare_traces.py`. The
  three-way trace was essential for finding the bugs above.

### Refactored — codegen modularisation (v1.2 mechanical split)

The 32K-line `lib/backend/llvm_codegen.cpp` and the 20K-line
`lib/backend/tensor_codegen.cpp` are now split into focused per-domain
files.  IR-identical to the prior monolith — verified at every step
against per-PR baselines (58/58 match, 0 diffs) — so this is purely a
modularity / build-time / readability win, not a behaviour change.

- **Extracted from `llvm_codegen.cpp`:**
  - `lib/backend/logic_workspace_codegen.cpp` (`c066c8b`) — 23
    consciousness-engine handlers (logic vars, KB, factor graphs,
    workspace, tensor/model serialization).
  - In-place sub-method split of the early `codegenCall` dispatch
    arms into `codegenCallInlineLambda`, `codegenCallResultAsFunc`,
    `codegenCallOperationResultAsFunc` (`769480b`) — first concrete
    payload of the audited prerequisite split before further
    extractions.
- **Extracted from `tensor_codegen.cpp`** (now ~1,280 lines, down from
  19,940 — a 94% reduction):
  - `tensor_dataloader_codegen.cpp` (`00e4bd4`) — 6 dataloader methods.
  - `tensor_transformer_codegen.cpp` (`342bcb5`) — Track 8 attention
    stack (9 methods, ~2,550 lines).
  - `tensor_loss_codegen.cpp` (`40669c6`) — 14 loss functions
    (~1,650 lines).
  - `tensor_linalg_codegen.cpp` (`9c1efc1`) — 8 linear-algebra ops
    (LU, det, inv, solve, Cholesky, QR, SVD, einsum; ~1,260 lines).
  - `tensor_training_codegen.cpp` (`c1dc0fe`) — 17 optimiser/weight-
    init/LR-scheduler methods (~1,500 lines).  Required promoting
    `taggedNumericToDouble` to a private static method on
    `TensorCodegen` so every split file can reach it.
  - `tensor_conv_codegen.cpp` (`052b5cf`) — 7 conv/pool methods plus
    the shared `extractAsDouble` helper (~1,595 lines).
  - `tensor_activation_codegen.cpp` (`16e33bc`) — 36 activation forward
    + backward methods (~2,587 lines).
  - `tensor_reduce_codegen.cpp` (`da2c330`) — matmul, dot, reduce,
    sum, mean, apply (9 methods, ~1,730 lines).
  - `tensor_arith_codegen.cpp` (`7542131`) — internal + SIMD
    elementwise arithmetic (~565 lines).
  - `tensor_shape_codegen.cpp` (`4c6cc9e`) — 11 shape methods
    (reshape, transpose, squeeze, etc.; ~1,690 lines).
  - `tensor_creation_codegen.cpp` (`9773bfe`) — `createTensorWithDims`
    plus zeros/ones/eye/arange/linspace/full factories (~1,236
    lines).
  - `tensor_extras_codegen.cpp` (`dadec34`) — Phase 4/5/7 supplements
    (tile, pad, statistics, conv3d) plus tensor unary/binary/scale/
    batch-matmul (~1,660 lines).

### Fixed — runtime / codegen / packaging

- **Bug X — `codegenNamedLet` leaked the loop name** (`590495c`).
  After the body and outer call were emitted, the function used to
  leave its `function_table[loop_name]`, `symbol_table[loop_name +
  "_func"]`, and `global_symbol_table[loop_name + "_func"]` entries
  pointing at its loop_func.  When stdlib's 1-binding `(let loop ((i
  0)) …)` in `time-it` was compiled alongside Noesis source files
  containing 2/3/4-binding `let loop` forms — and an earlier
  let-binding had populated `function_table["loop"]` with a lambda
  via `binding_codegen.cpp:registerLambdaBinding` — the next named-
  let's body resolved `(loop x)` against the wrong function and
  produced a misleading `Arity mismatch: loop expects 2 arguments
  but got 1` ahead of the genuine forward-ref errors.  Fix: save and
  restore the prior bindings under `loop_name` around the body+call
  emission.  IR-identical for code that previously compiled cleanly.
- **Bug X minimal-repro — silent AOT no-output** (`d7c97db`).  When
  `eshkol-run foo.esk` (no `-o`, no `-r`) AOT-compiled a single file
  through the LLVM-direct path (no separate object inputs to link),
  the `[eshkol-run] compiled to 'a.out'. Run it (./a.out) or use
  \`eshkol-run -r foo.esk\`…` notice was skipped because it lived
  only on the link-objects branch.  Users with the Lisp-shebang
  expectation saw nothing on stdout despite a top-level `(display
  …)` and could not tell whether a binary had been produced.  Fix:
  emit the same notice on the LLVM-direct path.
- **Deterministic IR via counter-based name uniquifier** (`d5b3ebf`).
  Pattern-match alloca slot names previously used the heap pointer
  address (`reinterpret_cast<uintptr_t>(val_slot)`) as a suffix,
  making repeated builds emit different `__pat_pred_arg_*` LLVM
  names.  Replaced with a per-compilation counter reset in
  `generateIR()`, so IR baselines reproduce.
- **Exception-aware error paths** (`33466e7`).  Replaced 14
  `std::abort()` and 2 `assert(0)` calls in `lib/core/runtime.cpp`,
  `lib/backend/vm_inference.c`, and `lib/backend/vm_tensor_ops.c`
  with `eshkol_raise(eshkol_make_exception(...))` that prints a
  diagnostic and exits 1 — so a `(guard ...)` handler can catch
  them and `assert(0)` no longer disappears under `-DNDEBUG`.
- **Archive cleanup** (`39f145f`).  Removed
  `lib/backend/archive/eshkol_compiler_standalone.{c,h}` and
  `lib/backend/archive/qllm_distributed.c` (4 files, ~7,800 lines).
  These were near-duplicate copies of the active dispatchers (the
  ICC-driven audit confirmed this with complexity-score-identical
  fingerprints).

### Fixed — diagnostics + cross-mode parity (Noesis residual audit Y/Z)

- **Bug Y — AOT couldn't find stdlib symbols** (`4ca7637`).
  `eshkol-run foo.esk` (no `-r`, no `-o`) rejected calls to plain
  stdlib functions — `length`, `reverse`, `append`, `assoc`,
  `filter`, `for-each`, … — with `Unknown function: NAME`,
  whereas `eshkol-run -r foo.esk` (JIT) ran the same source.  Root
  cause: a 2026-04 deprecation comment in `exe/eshkol-run.cpp`
  removed the AOT auto-load and gated stdlib-linking on the source
  containing an explicit `(require stdlib)`; JIT was unaffected
  because eshkol-repl-lib auto-discovers stdlib symbols.  Fix:
  synthesise a top-of-module `(require stdlib)` in the AST when
  `--no-stdlib` is not passed.  `--no-stdlib` remains the
  documented opt-out and is now the only way to skip stdlib.
- **Bug Z — `(provide ...)` enforced under AOT but informational
  under JIT** (`1235e0a`).  A function defined in `lib.esk` but
  absent from its `(provide …)` list was unreachable from a file
  that `(load …)`d `lib.esk` under AOT, while JIT (and the
  documented + Eshkol stdlib's own use of `provide`) treats the
  list as informational.  Root cause: `process_requires` called
  `rename_private_symbols`, which mangled every non-exported
  define.  Fix: skip the rename so cross-file calls resolve the
  same way they do under JIT.  The `rename_private_symbols`
  function and the `ESHKOL_PROVIDE_OP` machinery stay in place so a
  future per-file pragma can opt in to strict export enforcement
  without breaking existing code.
- **`list?` no longer mis-classifies non-cons heap pointers**
  (`bec1978`).  `codegenListPredicate` previously assumed any
  `HEAP_PTR` tagged value was a cons cell.  `(list? "abcdefgh")`
  could therefore return `#t` when the string's heap layout
  happened to look pair-like, causing later `cdr`-recursion to
  crash with `cdr: argument is not a pair`.  `pair?` already does
  the proper `HEAP_SUBTYPE_CONS` check; `list?` now does too.  This
  was the root cause of the v1.2 edge-case `json_schema_test`
  crashing partway through — `validate`'s array-errs branch was
  treating strings as list candidates and recursing into
  `length`-on-string.
- **Stdlib functions now use weak (LinkOnceODR) linkage in library
  mode** so user code can override a stdlib symbol without a
  `duplicate symbol` link error.  `createLibraryInitFunction` was
  hardcoding `GlobalValue::ExternalLinkage` on macOS/Linux for every
  non-lambda function in `function_table` after Step 1's
  `createFunctionDeclaration` had set the right linkage; the
  Windows path correctly used `publicDefinitionLinkage(true)`
  (`WeakAnyLinkage`).  Now both branches use
  `publicDefinitionLinkage(true)` (`LinkOnceODRLinkage` on
  macOS/Linux, `WeakAnyLinkage` on Windows).  Pre-fix
  reproducer: `tests/features/ultimate_math_stress.esk` defining
  its own `vec-scale` collided with `lib/math/ode.esk`'s helper
  `vec-scale` (both at strong external) and failed at link time.
  Bug Z (commit `1235e0a`) made `(provide ...)` informational,
  exposing this latent issue.  Note: this is "weak override"
  semantics — if user defines `f` and stdlib internally calls `f`,
  the user's `f` wins everywhere.  True module-private internals
  remain v1.3 architectural work.
- **Compile-error line markers now point at the actual source
  line** (carry-forward closed).  The reader,
  `eshkol_parse_next_ast_from_stream`, used to strip comment lines
  *including their trailing newline* (`std::getline` consumes the
  `\n` it found) and started a fresh `SchemeTokenizer` at line 1
  for every form — so `(undefined-fn …)` on file line 6 was
  reported as line 1:2 (or wherever the most recent stdlib AST
  happened to sit).  Two-part fix: (1) the reader now consumes
  the comment body up to but not including the `\n`, leaving the
  newline in `input` so the tokenizer's line counter stays
  accurate within a form; (2) a new thread-local
  `g_stream_line` / `g_stream_column` pair tracks cumulative file
  position across successive `eshkol_parse_next_ast_from_stream`
  calls and is passed to `SchemeTokenizer`'s constructor.
  `load_file_asts` (and the REPL/server stringstream parsers) now
  call the new `eshkol_reset_parse_line_counter()` API at the
  start of each fresh parse session.  Regression suite at
  `tests/v1_2_edge_cases/error_line_marker_test.sh`.

### Build + test infrastructure

- **CMake `stdlib.o` now tracks transitive sources** (`bec1978`).
  The `DEPENDS` list previously named only `lib/stdlib.esk`, so
  edits to any `(require)`d submodule (`lib/core/json_schema.esk`,
  `lib/core/streams.esk`, `lib/core/url.esk`, …) didn't trigger
  a stdlib rebuild.  `file(GLOB_RECURSE … CONFIGURE_DEPENDS)` now
  watches `lib/{core,math,signal,random,web,tensor,quantum,ml}/*.esk`
  and `lib/math.esk` so newly-added modules pick up automatically.
- **v1.2 edge-case runner honours `;; mode: jit` markers**
  (`bec1978`).  Eight of the 58 v1.2 tests are JIT-only (they
  exercise `eval`, dynamic loads, or REPL-side symbol resolution
  that AOT compilation can't model).  The runner now forwards them
  through `eshkol-run -r` so JIT-only passes don't show up as AOT
  failures.

### Tooling — release-process gaps closed

- **v1_2_edge_cases suite now invoked by `scripts/run_all_tests.sh`**
  via the new `scripts/run_v1_2_edge_cases_tests.sh` runner.  Per
  v1.2 audit blocker #1.
- **CI sanitizer lane** added: `linux-x64-asan-ubsan` runs the v1.2
  edge-case suite under `-DESHKOL_ENABLE_ASAN=ON
  -DESHKOL_ENABLE_UBSAN=ON`.  TSan and MSan are still deferred —
  they need TSan/MSan-built libstdc++ which apt.llvm.org doesn't
  ship.
- **Homebrew formula bumped** from `v1.1.13-accelerate` to
  `v1.2.0-scale`; `sha256` is reset and will be filled in by
  `scripts/update-homebrew-formula.sh` after the release tarball
  is published.

## [1.2.0-scale-pre1] - 2026-04-24 (mid-cycle internal preview)

The production-readiness release.  Mid-cycle internal preview tag —
the final v1.2.0-scale public release is the 2026-05-01 entry above.
Model serialization, a stable C ABI with Python bindings, per-thread
arenas, image/CSV I/O, a plotting stdlib, actionable error messages,
Windows ARM64 support, and a long tail of Noesis- / Moonlab-driven
hardening, perf, and correctness fixes.

### Fixed — late-cycle correctness (Bugs J–W, Quirks 1/3/4/6/7/10/11/14/15)

- **Quirk 14 — named-let capture broke for pointer-typed Instructions
  + missed sync-back.** Two bugs in codegenNamedLet's free-variable
  capture machinery. (1) When the captured outer storage was an
  IntToPtrInst (the typical shape inside a closure-env-capturing
  helper), the capture global was seeded with the POINTER bits
  instead of the value through it — the loop body then read garbage
  (effectively 0). (2) After the loop returned, the capture global
  held the latest value but the outer storage was never updated.
  Both fixed: load through pointer-typed Instructions on entry, and
  add a post-call sync-back that stores the global's final value
  back to any writable outer slot. Closes Noesis Quirk 14
  (dg-extract-symbols silently dropped chars from string tokens).
- **Quirk 15 — UTF-8 char literals + (string …) round-trip.** Two
  bugs combined to corrupt non-ASCII characters: (1) the reader's
  `#\<char>` fallback consumed exactly ONE byte, so multi-byte
  codepoints (`#\█` = U+2588 = E2 96 88) leaked their continuation
  bytes as garbage tokens; (2) `(string ch …)` codegen truncated each
  codepoint to int8, producing invalid UTF-8 byte sequences. Fix:
  reader uses UTF-8 lead-byte high bits to consume the right number
  of bytes; parse_atom decodes the bytes into an int64 codepoint;
  `(string …)` codegen calls a new runtime helper
  `eshkol_string_from_codepoints` that emits proper 1..4-byte UTF-8.
  Round-trips verified for ASCII / 2-byte / 3-byte / 4-byte
  codepoints + `string-length` correctly counts codepoints, not bytes.
- **Loader use-after-free in update_ast_references (EXTERN_OP).**
  The require-time symbol-rename walker read `call_op.num_vars` /
  `call_op.variables` for `ESHKOL_EXTERN_OP`, but EXTERN_OP populates
  `extern_op` (name / real_name / return_type / parameters /
  num_params) — different union slot. The walker dereferenced
  `extern_op.return_type` (a `char*`) as a uint64_t length and walked
  off into uninitialised memory, SIGSEGV'ing every precompiled module
  that had BOTH a `(provide …)` list and a private `(define …)`
  referencing an `(extern …)` declaration. Trigger surfaced in
  `core.testing` (used by `collections_test` and `cache_test`); both
  tests now compile and pass (49/49 + 33/33). Edge-case suite jumped
  from 35/35-with-9-skipped to 42/44.
- **R7RS current-output-port is now a real parameter object.**
  Before: `(current-output-port)` returned the literal stdout FILE*
  via hardcoded codegen; the setter form was a silent no-op. So
  `parameterize ((current-output-port p)) (display x)` always wrote
  to stdout. Fix: runtime-side cells (`g_current_{input,output,error}_fp`)
  back the parameter; codegen reads the cell on the getter form and
  writes it on the setter form (which `parameterize` generates for
  save/restore). `display` / `write` / `newline` (no port arg) now
  consult the cell via `eshkol_runtime_current_output_fp()` —
  redirect-into-string-port now Just Works for all output paths.
- **Bug W — forward-ref errors now name the function.** Before,
  calling a forward-referenced function whose define-site was never
  loaded raised "called a forward-referenced function that was
  never defined" with no indication WHICH function. Codegen now
  emits a per-call-site guard `eshkol_check_forward_ref(slot,
  stub_sentinel, name_literal)` that compares the loaded slot
  pointer to the published stub address; if equal, raises
  "called undefined function 'NAME' (forward-referenced but never
  defined; check that the file containing its `define` is `(load …)`ed
  or `(require …)`d before the call site)" and exit 1. The legacy
  nameless stub remains for paths where the slot pointer escapes
  through a captured value.
- **`(map display lst)` no longer crashes the compiler.** Before,
  the legacy first-class `display` wrapper returned `i64 0`; map's
  cons-builder fed that i64 back into `unpackDouble`, hitting a
  nullptr deref in LLVM `Value::setName`. Wrapper now returns
  tagged null with the `tagged_value(tagged_value)` ABI matching
  the closure dispatcher and the Quirk 11 path.
- **Quirk 11 — `display`/`write`/`newline` are now first-class.**
  Before: bare references (`(for-each display xs)`,
  `(define printer display)`) raised "Unbound variable: display"
  because the codegen wrapper only existed in call position.
  codegenVariable now wraps each as a unary closure (see
  `createBuiltinIOFunction`); the type checker agrees they're
  callable. With the port-plumbing fix above, these now correctly
  honor `current-output-port` under `parameterize` — output
  capture into a string port works for all forms.
- **Quirk 10 — `append` silently dropped args 3+.** The stdlib
  `append` was defined fixed-arity 2; `(append a b c d)` quietly
  truncated to the first two. Rewritten as properly variadic per
  R7RS §6.4: `(append)` returns `()`, `(append a)` returns `a`
  as-is, N-ary produces the concatenation of all lists. Improper
  tails permitted in the last position. (Noesis originally filed
  this against a 4-arg repro in `self_model_sync.esk` and later
  retracted the specific trigger, but the underlying arity-2
  stdlib definition was still wrong per R7RS §6.4.)
- **Bug T (reader) — R7RS dotted-pair literals.** `'(a . b)` was
  mis-parsed: the dot became a literal symbol, producing the
  3-element list `(a |.| b)` instead of a cons pair `(a . b)`.
  `parse_quoted_list_internal` / `parse_quasiquoted_list_internal`
  now detect a bare `.` token, read one tail datum, and build a
  right-nested cons chain. `codegenQuotedList` special-cases
  `CALL_OP(cons, car, cdr)` to emit a real cons cell; `codegenQuasiquote`
  gained matching handling so `` `(,key . ,val) `` works.
- **Bug T (strict-typing safety).** `car` / `cdr` of any non-pair
  heap object (symbol, string, hash, record, bignum, etc.) now
  raises "argument is not a pair" instead of silently dereferencing
  the wrong memory. The `subtype_probe` block in both codegen paths
  gates `list_block` on `HEAP_SUBTYPE_CONS`; every other subtype is
  routed to a dedicated raise block.
- **Bug U — REPL entry picker.** The substring match was greedy:
  `budget-remaining`, `remain`, `remainder-user` all collided with
  `main` because the picker matched anywhere in the symbol rather
  than at position 0. Renaming a user-define to be the batch entry
  is now explicitly refused; the picker uses whole-token equality.
- **Bug S — REPL-mangled variadic apply.** `apply` on a user
  variadic whose name had been mangled by the REPL (e.g. during
  file-level `(define (f . args) …)`) lost `variadic_info` and
  silently dropped the rest list. The apply path now resolves the
  pre-mangle name before looking up variadic_info.
- **Bug R — empty-map zombie HEAP_PTR.** `map` over an empty list
  produced a HEAP_PTR with no valid header, so a follow-up `ptr-8`
  read (pair? / vector-ref) SIGSEGV'd. Empty-map now returns a
  properly-tagged null.
- **Bug Q — append-mode ports.** New `open-output-file-append` for
  write-ahead logs (dKB persistence, Mneme episode store, Hiereia
  cycle-log).
- **Bug P — apply on cross-file user functions in REPL mode.**
  Apply resolution now searches all loaded modules, not just the
  currently-compiling one; Noesis can call `apply` on functions
  `require`d from another module.
- **Bug O — case with symbol-literal keys.** `(case x ((sigma) …))`
  was evaluating the key list as a call; case now treats keys as
  quoted data uniformly.
- **Bug M — shadowable-OP misses letrec bindings.** The shadowable
  check saw `let`/`define` bindings but not `letrec` / `letrec*`,
  so a user `unify` inside a letrec silently resolved to the
  builtin. Fixed in `transformInternalDefinesToLetrec`.
- **Bug J — named-let non-tail self-call.** A non-tail recursive
  call from inside a named-let produced LLVM IR where the phi
  predecessor list referenced a block already replaced by a later
  optimization pass. Captured the exit block explicitly before
  branching.
- **T1 — arity warnings ignore rest-args.** The type checker's
  arity warning counted rest-arg functions as fixed-arity,
  producing spurious warnings on every `(apply f …)` call.
- **Quirk 1 — HoTT cons type.** `cons(A, B)` synthesize-application
  now narrows to `List` when the cdr is already `List` or `Null`
  (per R7RS "a list is `()` or `(cons X list)`"). Eliminates the
  false "expected List, got Pair<List, List>" warnings that
  peppered every Noesis smoke.
- **Quirk 3 — cross-file eq? on interned symbols.** Not
  reproducible under current HEAD; fixed by earlier M/P/S/T/R7RS-1
  changes. Regression test added covering all reported shapes
  (bare literal, memq, assq, hash-table storage, vector-as-record,
  filter across file boundaries, string->symbol roundtrip).
- **Quirk 4 — s-expression printing.** Stdlib now ships
  `sexp->canonical-string` and `sexp->string` helpers that
  correctly handle proper lists, dotted pairs, improper lists,
  alists, and mixed structure. The naive user walk crashed the
  moment it hit a dotted pair; the stdlib helper doesn't.
- **Quirk 6 — REPL exit propagation.** The REPL swallowed codegen
  failures; `eshkol-run -r` now propagates a non-zero exit when
  the script fails to compile.
- **Quirk 7 — clearer `if` multi-else diagnostic.** Generic
  "expected closing parenthesis after if expression" replaced with
  a concrete message suggesting `begin` or `cond`.
- **SEQUENCE_OP flattening.** `define-record-type` used in a user
  function ("Unknown function: make-point") failed because the
  three top-level pre-declaration passes only walked flat
  `DEFINE_OP` nodes, missing the sub-defines wrapped in a single
  `SEQUENCE_OP`. Added an architectural "top-level AST list is
  flat" invariant: a single `SEQUENCE_OP` flattening pre-pass in
  `generateIR()` feeds every downstream pass.
- `set-cdr!` / `set-car!` now preserve the HEAP_PTR tag when the
  replacement is a tagged value (list, cons, variable reference).
  Previously `detectValueType` flattened tagged_value structs to
  INT64, so `(set-cdr! p (list 4 5))` stored the list's heap
  address with an INT64 tag and later cdr walks saw an integer.
  Noesis Bug E — blocked dKB, Mneme ring, Workspace queue,
  proof-tree child lists, Hiereia cycle log.
- `(read port)` now interns symbols through the process-global
  pool (`eshkol_intern_symbol_lookup`). Previously each `(read)`
  produced a fresh arena allocation, so `(eq? (read port) 'foo)`
  always returned #f — violating R7RS §6.5. Noesis Bug F —
  blocked dKB persistence, Mneme load, proof-tree replay,
  Workspace state restore.
- ONNX export: `double_data` stored in TensorProto field 10 (was
  field 5, which is int32_data). Required `GraphProto.name` field
  emitted so `onnx.checker.check_model` accepts the output.

### Added — late-cycle
- **R7RS §7.1.1 radix literals** — `#b` (binary), `#o` (octal),
  `#d` (decimal), `#x` (hex), with optional sign and exactness
  prefix (`#e` / `#i`) chained in either order. The tokenizer
  converts to a decimal `TOKEN_NUMBER` so downstream code paths
  are unchanged. Before, `#xFF` was tokenized as a symbol and
  failed as an undefined variable; `0xFF` (C syntax) split into
  two tokens.
- `eshkol_ffi_tensor_shape()` FFI accessor so pybind11 can return
  N-D numpy arrays (previously everything flattened to 1-D).
- Subprocess stdin-null fast path: `process-spawn-nostdin` wires
  the child's stdin to `/dev/null` instead of creating a pipe we
  won't use. Saves a `pipe()` + 2 `close()` per call —
  `run-command-capture` / `run-argv-capture` (the hot paths) drop
  from 2.33 ms to 2.21 ms at N=5000 on macOS.
- `POSIX_SPAWN_CLOEXEC_DEFAULT` on Darwin: drops 6 `addclose`
  entries per spawn by marking all fds close-on-exec in the child
  by default.
- VM hyper-dual laplacian: exact second derivatives via hyper-duals
  (replaces central-difference finite-difference).

### Added — roadmap items

- **Model serialization** (`.eshkol-model`). `model-save` /
  `model-load` in `lib/core/model_io.cpp`. Compact binary format
  (magic + version + per-tensor metadata + contiguous float data),
  inspired by safetensors / GGUF. Save/load named tensor checkpoints
  with round-trip correctness and CRC validation.
- **Stable C FFI header** (`inc/eshkol/eshkol_ffi.h`). Clean C ABI for
  init/shutdown, parse/compile/call, tensor create/read/write, arena
  lifecycle. Behind `extern "C"`, C-compatible includes, suitable for
  embedding in any language. Header compiles as plain C (no `<cstddef>`
  / `<cstdint>`).
- **Python bindings via pybind11** (`bindings/python/eshkol_module.cpp`).
  NumPy interop with zero-copy tensor views. `ESHKOL_PYTHON_BINDINGS=ON`
  CMake option.
- **Per-thread arenas**. `arena_create_thread_local()` /
  `arena_merge_to_parent()`. Parallel workers allocate in their own
  arenas without contention; results flushed into the parent arena on
  join.
- **Image I/O** (stb_image vendored in `deps/stb/`). `image-read`,
  `image-write`, `image-to-grayscale`, `image-resize` load/save PNG,
  JPEG, BMP as `(height, width, channels)` tensors.
- **CSV/DataFrame** (`lib/core/data/csv.esk`). Column-typed CSV loader
  with type inference; select, filter, group-by, join operations.
- **Terminal plotting** (`lib/core/plot.esk`). `sparkline`,
  `bar-chart`, `histogram` — Unicode block-character visualization
  with no external dependencies.
- **Source-location error messages** throughout the frontend and
  codegen: `file.esk:line:col: error:` + caret + underline for the
  offending span.
- **GPU API — `eshkol_gpu_has_fp64()`**. Reports 1 when any fp64 path
  is available (CUDA native OR Metal SF64 emulation); the older
  `eshkol_gpu_supports_f64()` is now documented as "native hardware
  fp64 only".

### Added — perf and parallelism

- Per-call subprocess latency reduced 4× (77 ms → 19 ms) via
  pthread pipe drainers + single blocking waitpid, the canonical POSIX
  pattern used by CPython / Go os/exec / libuv. No more pipe-full
  deadlocks, no polling roundoff.
- GPU matmul dispatch: AMX peak measured at 1.1 TFLOPS, driven by the
  updated blas/gpu cost model (blas_peak=1100, gpu_peak=200 GFLOPS).
  GPU selected only when it's actually faster.
- Metal SF64 tier-1 `[GPU] df64 completed: …` spam now gated on
  `ESHKOL_VERBOSE=1` (default silent).

### Added — R7RS and language

- Symbol interning across modules (`symbol_intern.cpp`) — `eq?` / `eqv?`
  now correct for symbols generated in different stdlib modules.
- Codegen builtins as first-class values (sret wrapper registry for
  AD ops + `call-with-values` consumers). Lambda forms that used
  `reverse`, `append`, `list`, `map`, etc. as rvalues now work.
- Internal `define` hoisting follows Racket-compatible letrec* order
  (all `define`s hoisted, not only leading-consecutive ones).
- `string-length` honours the header byte count. `substring` validates
  start / end bounds before memcpy.
- `call-with-values` routes stdlib-named consumers correctly.
- Binary ports + bytevector I/O (`read-bytevector` with k=0 returns
  empty bytevector per R7RS §6.13.2).
- `string*` / `acons` / `partition` / `split-at` return HEAP_PTR
  tagged values so `(define x (list* …))` / `(car x)` work end-to-end.
- Bignum arithmetic: full 35-gap audit closed, including rational
  comparison, `abs`, `min`/`max` precision, `expt` with exact integer
  exponents, `number->string` / `string->number` bignum round-trip,
  and `bignum + double` → double per R7RS exact+inexact semantics.

### Added — tooling and CI

- Sanitizer build infrastructure: ASan / UBSan / TSan / MSan / LSan
  wired via CMake + `scripts/build-sanitizer.sh`.
- 16-lane CI matrix (linux/macos/windows × x64/arm64 × lite/xla/cuda).
- 512 MB stack by default on macOS/Linux for deep-recursion workloads;
  `ESHKOL_STACK_SIZE` env override.

### Added — Windows ARM64 native support (carried forward from 1.1.13)

- VS 2022 + ClangCL + LLVM 21 aarch64 SDK build path.
- Runtime symbol renames (eshkol_fopen, eshkol_access, …) resolve
  MSVC POSIX-shim warnings.
- Dynamic `jmp_buf` sizing; architecture-appropriate LLVM target
  libraries (AArch64 on ARM64, X86 on x64).

### Fixed — Noesis integration

Four waves of Noesis residual audits (v2 → v5) closed:

- Quasiquote `,x` / `,@xs` interpolation codegen.
- `hash-table` runtime wiring (make, ref, set!, delete, keys, values).
- `define-record-type` constructor/predicate/accessor/mutator codegen.
- `match` with `(? pred)` patterns — predicate lookup across clauses.
- `#:keyword` syntax (Racket-style self-quoting keywords).
- Colon-keyword tokenizer disambiguation (`:foo` glued vs `:` spaced).
- Extern declarations accept `:real` both tokenized and spelled.
- `transformInternalDefinesToLetrec` hoists all internal defines.
- Named-let inside mutually-recursive fns TCO bug (empty loop returned
  0); save/restore TCO context at inner-letrec boundaries.
- `call-with-values` named-consumer resolution.
- `tensor-ref` with cons-cell-wrapped index (`(tensor-ref t (list i))`)
  now dispatches to the new `eshkol_unwrap_list_index` runtime helper.
- `list*` / `acons` / `split-at` / `partition` return HEAP_PTR.
- Symbolic AD arena lifetime (`free()` on arena-allocated AST nodes
  was aborting with "pointer being freed was not allocated"; removed
  the erroneous free calls since the arena owns the lifetime).
- Subprocess `run-command-capture` — two intertwined bugs fixed:
  - Return-code contract: `process-wait` now returns `0=exited,
    1=timeout, -1=error` per the .esk docstring (previously returned
    the child's exit code, so every non-zero exit collided with the
    timeout sentinel).
  - Pipe drainer: pthread per stream + blocking waitpid avoids
    pipe-fill deadlocks on chatty children and keeps fast-exit cost
    ~sub-ms over the fork+exec baseline.
- `string-append` header-size off-by-one: the allocator already adds
  the NUL byte; callers now pass the bare byte count so
  `(string-length (string-append "a" "b"))` is `2`, not `3`.
- stdlib.o JIT trio (REPL path): `__eshkol_lib_init__` is invoked
  after `addObjectFile` so module-level defines populate;
  `eshkol-variadic` LLVM attribute preserves Scheme-level variadic-
  ness across the stdlib.bc boundary; both together let
  `(make-list 3 'x)` and `(base64-encode-string "Hello")` work in
  REPL mode.

### Fixed — Moonlab integration (GPU backend)

- Header `<cstddef>` / `<cstdint>` → `<stddef.h>` / `<stdint.h>` so C
  consumers (Moonlab, lilirrep, QGTL, SbNN) can include without
  wrapping as C++.
- `eshkol_gpu_init()` return convention documented clearly in the
  header (1 = success, 0 = no GPU) with explicit warning about the
  `!= 0` false-negative idiom.
- `eshkol_gpu_supports_f64()` docstring updated to say "native hardware
  fp64 only"; `eshkol_gpu_has_fp64()` added for "any fp64 path".

### Fixed — consciousness / AD

- `ws-step!` fully wired: LLVM codegen loop calls closures via
  `codegenClosureCall`; C runtime helpers handle tensor wrapping and
  softmax broadcast.
- `fg-update-cpt!` enables real learning: CPT mutation + message
  reset → beliefs reconverge.
- `fg-update-cpt!` bench 14: vector-typed CPTs no longer silently
  ignored.
- `kb-load` format: no more dangling raw HEAP_PTR across save/load.
- `kb-query` now works in JIT mode (was working compiled-only).
- AD gradient wrong when `set!` on outer-scope var from inside AD
  body (Bug C).
- `ad-value` undefined symbol in JIT (Bug B).
- Reverse-mode AD tape: 6 missing tensor-backward ops
  (TRANSPOSE, SUM, BROADCAST_ADD/MUL, EMBEDDING, ATTENTION) —
  silent gradient corruption removed.
- `findFreeVariablesImpl` recurses into all ~30 op types
  (DYNAMIC_WIND_OP, CALL_CC_OP, GUARD_OP, RAISE_OP, VALUES_OP,
  MATCH_OP, calculus ops, …) — fixes "Cannot capture k from outer
  function" on call/cc inside dynamic-wind.

### Fixed — parallel / concurrency

- `parallel-map` actually parallelizes (B5/B6/B7 — previously ran
  serial).
- `parallel-map` at scale (N=100K) no longer hangs.
- `parallel-map` in JIT mode no longer hangs.
- `parallel-map` "workers not registered" inside `define`d function —
  llvm.global_ctors now emits worker registration for stdlib too.
- TCO context corruption in nested `letrec` — save/restore at entry/
  exit.
- JIT thread-pool state hang: map + parallel-map sequence deadlocks
  cleared.

### Fixed — hardening (epics #189–#195 landed)

- `#189` — SECURITY.md + HARDENING.md + threat model.
- `#190` — subprocess shell-string injection: `run-argv` / `process-
  spawn-argv` (execvp, no shell).
- `#191` — Python FFI `derivative` method AST injection: input
  validated against lambda-source whitelist.
- `#192` — memory-safety integer overflows in arena allocator, KB
  persistence, image I/O.
- `#193` — path traversal + TOCTOU + Windows-subprocess buffer
  overflow (4 items).
- `#194` — 36 silent-swallow error-propagation sites surfaced
  through logs + marked explicit.
- `#195` — ReDoS protection (PCRE2 match_limit + depth_limit) +
  SQL-injection guards + URL CRLF injection.

### Fixed — runtime correctness

- `string->number` returns `#f` for non-numeric input per R7RS.
- `string-fill!`, `string-set!` bounds-check properly.
- Port type check (input/output port flag bits, not HEAP_PTR
  equality).
- Parser `#(...)` vector literals parse inside function call arg
  positions AND inside `if` expressions.
- `let-rec*` letrec* define hoisting preserves R7RS semantics.
- `apply min`/`max` on numeric lists return the actual min/max (was
  returning `()`).
- `floor`/`ceil`/`round`/`truncate` no longer spam "not supported in
  reverse-mode AD" warnings for non-AD contexts; the runtime abort
  path remains for actual AD misuse.

### Changed — behaviour

- Precompiled `core.*` module discovery now auto-finds sub-modules
  in all pre-compiled libraries (no hardcoded prefix check); new
  stdlib directories "just work".
- Stdlib `--shared-lib` mode uses LinkOnceODRLinkage throughout so
  user code can override stdlib functions without duplicate-symbol
  errors.
- REPL JIT: uses `-force_load` (macOS) / `--whole-archive` (Linux)
  + `-export_dynamic` so new runtime functions auto-resolve
  without manual `ADD_SYMBOL` entries.

### Contributor credits

Many of the Noesis and Moonlab audit fixes were driven by detailed
bug reports from those downstream projects. See
`docs/audits/eshkol-residual-bugs-*.md` for the full trail.

## [1.1.13-accelerate] - 2026-04-09

### Windows ARM64 + Release Workflow Overhaul + VM Closure Bug Fixes

#### Windows ARM64 Native Support
- Full build path for Windows ARM64 via VS 2022 + ClangCL + LLVM 21 aarch64 SDK
- New CMake auto-detection of `clang_rt.builtins-{x86_64|aarch64}.lib` based on `CMAKE_VS_PLATFORM_NAME`
- Multi-arch DIA SDK lookup (both `Program Files` and `Program Files (x86)` for both `amd64` and `arm64`)
- REPL JIT now links the architecture-appropriate LLVM target libraries (`LLVMAArch64*` on ARM64, `LLVMX86*` on x64)

#### setjmp/longjmp Cross-Platform Hardening
- Windows ARM64: uses `Intrinsic::sponentry` as the hidden `_setjmpex` context (matches Clang lowering)
- Windows x64: switched from `Intrinsic::localaddress` to `Intrinsic::frameaddress(0)` for the hidden `_setjmpex` context
- Removed compile-time `#ifdef _WIN32` branches in favor of runtime `Triple::isOSWindows()` checks — proper cross-compilation
- Dynamic `jmp_buf` sizing via `eshkol_jmp_buf_size()` runtime helper (no more hard-coded 256-byte buffers)

#### Runtime Symbol Renames (Windows POSIX shim disambiguation)
- `fopen` → `eshkol_fopen`, `access` → `eshkol_access`, `remove` → `eshkol_remove`, `rename` → `eshkol_rename`, `mkdir` → `eshkol_mkdir`, `rmdir` → `eshkol_rmdir`, `chdir` → `eshkol_chdir`, `stat` → `eshkol_stat`, `opendir` → `eshkol_opendir`
- Avoids MSVC's deprecated POSIX shim warnings on Windows
- Generated programs now call `eshkol_runtime_init()` at start of `main` (non-REPL mode)

#### Codegen Error Handling
- New `fatal_codegen_error_` flag — codegen now **fails hard** on undefined-function/undefined-variable/private-symbol errors instead of silently emitting `printf`/`exit` runtime stubs
- New `declared_functions_by_ast` map keyed by AST node identity — fixes function resolution when multiple defines share a name within the same module

#### VM Closure Bug Fixes (browser REPL + bytecode VM)
- **Named-let nested closure PC offset**: When a lambda is created inside a `let loop` body, the loop's bytecode is inlined into the parent function with PC adjustments — but the inner lambda's `OP_CLOSURE` constant (its `func_pc`) was *not* offset by the loop's start position, causing the inner closure to jump to a stale location with the wrong upvalue count. Symptom: "UPVALUE INDEX OUT OF BOUNDS" + gradient always equal to 1 in named-let gradient descent
- **Native 252 upvalue relay**: When a lambda inside a function captures a variable via the parent's upvalue (`is_local=false`), native 252 was reading `vm->stack[vm->fp + slot]` — treating the upvalue index as a stack-frame offset. Fix: read from `vm->stack[vm->fp - 1]` (the parent closure per the calling convention), then index into `parent_cl->closure.upvalues[slot]`. Together with the named-let fix, this restores correct gradients for all autodiff demos involving captured upvalues
- Both fixes verified end-to-end: gradient descent converges, train demo returns ~0.891, named-let gradient descent converges to y/x

#### CI / Release Workflow
- Release workflow rewritten as two matrices (`unix-release-matrix` × 10 + `windows-release-matrix` × 6) plus a `publish-release` job that downloads all artifacts, generates `SHA256SUMS.txt`, and publishes the GitHub release
- New release lanes: `windows-arm64-{lite,xla,cuda}`, `windows-x64-{lite,xla,cuda}`, `linux-{x64,arm64}-{lite,xla,cuda}`, `macos-{x64,arm64}-{lite,xla}` — 16 total per release
- Per-architecture LLVM SDK caching on Windows runners (cache key includes `${arch}` and SDK version)
- CI workflow updated: `windows-2022` → `windows-latest`, `max-parallel: 2` Windows throttling
- Removed Docker-based XLA/CUDA build paths in favor of native CMake builds

#### Website Mobile Responsiveness
- Hamburger nav menu collapses 7 nav links on screens ≤720px; opens as full-width dropdown; auto-closes when a link is clicked
- `html, body { overflow-x: hidden }` plus `min-width: 0` on flex/grid children — no more horizontal page scroll on any viewport
- Code blocks (`runnable-code` wrappers) now scroll horizontally *inside* the block instead of pushing the page wider
- `.docs-layout` switched from `1fr` to `minmax(0, 1fr)` — fixes the docs page being 972px wide on a 375px viewport
- `.comparison-table` becomes scrollable on ≤720px so the comparison table on `/downloads` doesn't push the page

#### Browser REPL Error Display
- REPL now captures stderr (compile warnings, parse errors) into `_vmStderr` and displays them as `error: undefined variable 'foo'` instead of silently re-prompting
- Suppresses the trailing `()` NIL fallback when a compile error fired
- Shows `error: could not parse expression` when nothing parses
- Same fix applied to runnable code blocks (Run ▶ buttons across the site)

#### Test Results
- 35/35 test suites, 100% pass rate (macOS ARM64, Linux x64, Windows x64, Windows ARM64)
- 32/32 runnable site examples verified in headless Chromium across mobile/tablet/desktop viewports

### Bytecode VM — Production Complete

The bytecode VM is now a fully production-grade execution engine with 555+ built-in functions, forward-mode automatic differentiation, R7RS control flow, exact arithmetic, and the consciousness engine.

- **Automatic differentiation**: Forward-mode AD via dual number propagation. Arithmetic and transcendental functions automatically track derivatives. `(derivative (lambda (x) (* x x)) 3.0)` → `6`
- **R7RS control flow**: `call/cc` with full continuation capture/restore and dynamic-wind unwinding, `guard`/`raise` exception handling, `values`/`call-with-values`
- **Exact arithmetic**: Rational literals (`1/3`), arbitrary-precision integers, complex numbers, R7RS special floats (`+nan.0`, `+inf.0`, `-inf.0`)
- **Consciousness engine**: Knowledge base queries with `?`-wildcard pattern matching, factor graphs with belief propagation, global workspace
- **555+ built-in functions**: Character operations, bitwise operations, type predicates, string processing (`split`, `join`, `trim`, `reverse`, `repeat`), list operations (`take`, `drop`, `any`, `every`, `find`), math extensions (`cosh`, `sinh`, `tanh`), complex numbers, port I/O
- **Mutual recursion**: Top-level function defines can reference each other without forward declarations
- **System integration**: `directory-entries` (POSIX readdir), `command-line` (argc/argv), thread pool
- **176/176 tests passing**

### Web Platform

- **eshkol.ai**: Complete website written in Eshkol (1,400 lines), compiled to WebAssembly
- **Browser REPL**: 63-opcode bytecode interpreter with 555+ builtins, running in WebAssembly via Emscripten
- **AD in the browser**: Automatic differentiation works through the REPL — gradient descent converges in the browser
- **Interactive learning**: 8-chapter textbook and 10-example gallery where every code example has a Run button
- **Live documentation**: Docs page loads markdown directly from GitHub with syntax highlighting
- **Downloads**: Platform-aware downloads page with GitHub Releases API integration
- **GitHub Pages deployment**: Automated via `.github/workflows/pages.yml`

---

## [1.1.12-accelerate] - 2026-04-07

### Toolchain Unification + Platform Hardening Release

#### LLVM 21 Toolchain Unification
- Standardized entire build on LLVM 21 across Linux, macOS, and Windows (previously mixed LLVM 17/18)
- New `cmake/LLVMToolchain.cmake`: authoritative LLVM version discovery and enforcement at configure time
- New `scripts/lib/llvm21-env.sh`: platform-aware LLVM 21 activation for all shell scripts
- All platform scripts now hand off LLVM policy to CMake instead of embedding independent logic
- Hard version check: configure fails with a clear error if LLVM major version is not exactly 21
- Removed misleading `LLVM 18+` compatibility branches from backend codegen

#### Native Windows Support
- Full build via Visual Studio 2022 + ClangCL + LLVM 21 SDK
- Configures with `Visual Studio 17 2022` generator and `-T ClangCL`
- `region_escape_tagged_value_into` ABI fix: now passes `eshkol_tagged_value_t` by pointer (`const eshkol_tagged_value_t*`) to satisfy Windows x64 calling convention for 16-byte aggregates

#### ARM64 ABI Fix
- Fixed `call_thunk_closure` in `arena_memory.cpp`: ARM64 returns 16-byte `eshkol_tagged_value_t` in registers (not via hidden return buffer as on x86/Windows)
- Added `#if defined(__aarch64__)` dispatch — direct return ABI on ARM64, hidden-buffer ABI on x86/Windows
- Resolves dynamic-wind + call/cc thunk invocation on Apple Silicon and Linux ARM64

#### Mutual TCO Fix
- `llvm_codegen.cpp`: version-gated tail call kind — `TCK_MustTail` on LLVM < 18, `TCK_Tail` on LLVM ≥ 18
- Fixes "LLVM ERROR: cannot use musttail" on Linux (LLVM 21 rejects musttail for aggregate-return functions)

#### Website
- Clean URL routing: navigation now uses `/downloads`, `/learn`, `/docs` etc. instead of `/#/downloads`
- GitHub Pages 404-redirect SPA routing for direct URL access
- Updated LLVM requirement strings: LLVM 17+ → LLVM 21+
- Updated WASM size stats to reflect current build sizes

#### CI/CD Expansion
- New GitLab CI matrix: Linux x64/arm64 × lite/XLA/CUDA + macOS × lite/XLA + Windows
- GitHub CI updated to LLVM 21 baseline across all runners
- Docker parity images (`docker/debian/`, `docker/ubuntu/`) updated to LLVM 21

#### Test Results
- 35/35 test suites, 438/438 tests, 100% pass rate (local, macOS ARM64)

---

## [1.1.11-accelerate] - 2026-03-27

### Performance Acceleration Release

Eshkol v1.1-accelerate delivers comprehensive performance acceleration through XLA integration, SIMD vectorization, parallelism primitives, and expanded math/ML libraries.

#### XLA Backend Integration
- Dual-mode architecture: StableHLO/MLIR path (when MLIR available) + LLVM-direct path (default)
- 6 core tensor operations wired through XLA: matmul, elementwise, reduce, transpose, broadcast, slice
- Threshold-based dispatch: XLA (>=100K elements) -> cBLAS (>=64) -> SIMD (>=64) -> scalar
- JIT compilation for dynamic shapes via LLVM ORC
- CPU/GPU code generation from single source with unified dispatch hierarchy

#### SIMD Vectorization
- CPU feature detection: SSE2, SSE4.1, AVX, AVX2, AVX-512, NEON (ARM64)
- Hand-written SIMD micro-kernels for tensor arithmetic (add, sub, mul, div)
- SIMD-accelerated activation functions: ReLU, sigmoid, GELU, LeakyReLU, SiLU
- SIMD dot product with horizontal sum reduction
- LLVM loop vectorization metadata on all tensor loop back-edges
- 64-byte AVX-512 aligned tensor memory allocation
- Platform-specific tuning via cache-blocked matrix kernels

#### Parallelism Primitives
- `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`
- `future`, `force`, `future-ready?` for asynchronous computation
- Work-stealing thread pool scheduler with hardware-aware sizing
- Thread-safe arena memory management

#### Extended Math Library
- **Complex numbers**: Full R7RS complex arithmetic with autodiff integration
- **FFT/IFFT**: Cooley-Tukey radix-2 implementation
- **Signal processing filters** (13 functions): Hamming/Hann/Blackman/Kaiser windows, direct and FFT-based convolution, FIR/IIR filter application, Butterworth filter design (lowpass/highpass/bandpass), frequency response analysis
- **Statistical distributions**: Normal, Poisson, Binomial, Exponential, Uniform, Geometric, Bernoulli (in stdlib)
- **Optimization algorithms** (7 functions): Gradient descent, Adam (adaptive moment estimation), L-BFGS (limited-memory BFGS with two-loop recursion), conjugate gradient (Fletcher-Reeves), backtracking Armijo line search

#### Arbitrary-Precision Arithmetic
- Bignum (arbitrary-precision integers) with full R7RS compliance
- Rational numbers (exact fractions) with all arithmetic operations
- Automatic int64 -> bignum overflow promotion and bignum -> int64 demotion
- Bitwise operations on bignums (two's complement semantics)
- 35 codegen gaps audited and fixed across arithmetic, comparison, conversion, and I/O

#### Consciousness Engine
- Logic programming primitives: unification, substitutions, knowledge base
- Active inference engine: factor graphs, belief propagation, free energy minimization
- Global workspace theory: modules, softmax competition, content broadcasting
- 22 builtin operations for logic, inference, and workspace manipulation
- CPT mutation with belief reconvergence for real-time learning

#### R7RS Compliance Extensions
- `call/cc` and `dynamic-wind` with proper continuation semantics
- `guard`/`raise` exception handling
- Bytevectors with full R7RS operations
- `let-syntax` / `syntax-rules` hygienic macros
- Tail call optimization validation
- Symbol operations (`symbol->string`, `string->symbol`)

#### GPU Backends
- Metal backend for Apple Silicon with SF64 software float64 emulation
- CUDA backend with cuBLAS integration and real compute kernels
- 5 GPU operations: elementwise, matmul, reduce, softmax, transpose

#### Production Hardening
- All 47/47 roadmap items completed (including GPU 5/5, Signal Processing 4/4, Web Platform 3/3)
- Tensor bounds checking with runtime validation
- Metal buffer leak fix (@autoreleasepool)
- REPL complex type handling
- Module visibility enforcement
- 35 test suites passing (438 test files)

#### Dual Backend Architecture (NEW)
- **Bytecode VM**: 63-opcode register+stack interpreter (eshkol_vm.c, 8457 lines) with 250+ native call IDs covering the full language
  - 15 runtime libraries: complex, rational, bignum, dual, autodiff, tensor, logic, inference, workspace, string, IO, hashtable, bytevector, multivalue, parameter
  - ESKB binary format with LEB128 encoding, CRC32 checksums, section-based layout
  - Bytecode emission via `-B` flag: `eshkol-run input.esk -B output.eskb`
  - VM linked into compiler build (ESHKOL_VM_LIBRARY_MODE)
- **Weight Matrix Transformer**: Programs as neural network weights (weight_matrices.c, ~6,800 lines)
  - d_model=256, 6 layers, FFN_DIM=2304, 12.22M parameters
  - 3-way verification: reference interpreter = simulated transformer = matrix-based forward pass
  - 126/126 inline programs and 123/123 traced programs passing, exports QLMW binary format for qLLM loading
- **qLLM Bridge**: Eshkol-qLLM tensor conversion with AD integration (qllm_bridge.h)

#### Windows Platform Support (NEW)
- Native Windows build via MSYS2/MinGW64 (contributed by mattneel, PR #9)
- UTF-8-safe REPL console output
- Runtime DLL bundling in CI artifacts
- MSYS-style file path normalization
- Platform runtime abstraction layer (platform_runtime.cpp/h)

#### Production Hardening (continued)
- ARM64 parallel ABI fix: struct return -> output pointer for eshkol_parallel_execute/map/fold/filter
- REPL CodeGenOptLevel::None fix for ARM64 3+ arg stdlib struct passing
- Cons cell header fix: arena_allocate_cons_with_header for proper HEAP_PTR display
- LinkOnceODRLinkage for stdlib symbol override prevention (no more duplicate symbols)
- Precompiled module discovery: collect_all_submodules() for automatic stdlib sub-module detection
- Weight matrix stack overflow fix: double-buffer State cur/nxt replaces 1.15MB trace[8192]
- `(load "path/to/file.esk")`: R7RS-compatible file loading (alias for require with path conversion)
- Port type check fix: flag bit detection instead of exact HEAP_PTR equality
- Substring bounds overflow protection
- Tensor reshape OOM null check

---

## [1.0.0-foundation] - 2025-12-12

### Production Release

Eshkol v1.0-foundation represents a complete, production-ready compiler with unprecedented integration of automatic differentiation, deterministic memory management, and homoiconic native code execution.

#### Core Compiler Implementation
- Modular LLVM backend with 21 specialized codegen modules
- Recursive descent parser with HoTT type expression support
- Bidirectional type checker with gradual typing
- Ownership and escape analysis for memory optimization
- Module system with dependency resolution and cycle detection
- Hygienic macro system (define-syntax with syntax-rules)
- Exception handling (guard/raise with R7RS semantics)

#### Automatic Differentiation System
- **Forward-mode AD**: Dual number arithmetic for efficient first derivatives
- **Reverse-mode AD**: Computational graph with tape stack for gradient computation
- **Symbolic AD**: Compile-time AST transformation
- **Nested gradients**: Up to 32 levels deep via global tape stack
- **Vector calculus operators** (8 total):
  - `derivative` - First derivative (forward-mode)
  - `gradient` - Gradient vector (reverse-mode)
  - `jacobian` - Jacobian matrix for vector functions
  - `hessian` - Hessian matrix (second derivatives)
  - `divergence` - Vector field divergence (∇·F)
  - `curl` - Vector field curl (∇×F, 3D only)
  - `laplacian` - Laplacian operator (∇²f)
  - `directional-derivative` - Derivative in specified direction
- Polymorphic arithmetic supporting int64/double/dual/tensor/AD-node

#### Memory Management (OALR)
- **Arena allocation**: O(1) bump-pointer with deterministic cleanup
- **Ownership tracking**: Compile-time analysis (owned, moved, borrowed states)
- **Escape analysis**: Automatic stack/region/shared allocation decisions
- **with-region syntax**: Lexical memory scopes
- **Zero garbage collection**: Fully deterministic performance
- **Global arena**: 64KB default block size, expandable
- **Region stack**: 16-level nesting depth

#### Tagged Value System
- 16-byte runtime representation with 8-bit type tags
- Immediate types (0-7): NULL, INT64, DOUBLE, BOOL, CHAR, SYMBOL, DUAL_NUMBER
- Consolidated types (8-9): HEAP_PTR, CALLABLE with object header subtypes
- 8-byte object headers for heap objects (subtype, flags, ref_count, size)
- 32-byte cons cells with complete tagged values (car and cdr)
- Mixed-type lists with zero type erasure

#### Closure System
- Static capture analysis during parsing
- Environment encoding with packed info (captures | fixed_params | is_variadic)
- Homoiconic display via embedded S-expressions
- Lambda registry for function pointer → S-expression mapping
- Variadic function support (fixed + rest parameters, or all-args-as-list)

#### Data Structures
- N-dimensional tensors with autodiff integration
- Hash tables (FNV-1a hashing, open addressing, 0.75 load factor)
- Heterogeneous vectors (Scheme-compatible)
- Strings with UTF-8 support
- Proper and improper lists
- Exception objects with source locations

#### Language Features (300+ Total)
- **39 special forms**: define, lambda, let/let*/letrec, if/cond/case/match, quote/quasiquote, etc.
- **300+ built-in functions**: Complete Scheme R7RS subset
- **60+ list operations**: map, filter, fold, compound accessors (caar through cddddr), etc.
- **30+ string utilities**: join, split, trim, case conversion, search, replace
- **25+ tensor operations**: element-wise arithmetic, linear algebra, reductions, transformations
- **10 hash table operations**: ref, set!, has-key?, remove!, keys, values, count, clear!
- **8 autodiff operators**: Complete vector calculus support
- Scheme-compatible syntax (R7RS subset)
- Module system with `require`/`provide`
- Pattern matching with 7 pattern types
- First-class functions and closures
- Tail call optimization (self-recursion → loops)
- Hygienic macros (syntax-rules)

#### Standard Library (Modular)
- `stdlib.esk` - Re-exports core modules
- `math.esk` - Linear algebra (det, inv, solve), numerical integration, root finding, statistics
- `core.functional.*` - compose, curry, flip
- `core.list.*` - higher-order, transforms, queries, search, sort, convert, generate, compound accessors
- `core.strings.*` - Extended string manipulation
- `core.json.*` - JSON parsing and serialization
- `core.data.*` - CSV processing, Base64 encoding
- `core.control.*` - Trampoline for deep recursion

#### Development Tools
- **eshkol-run**: Standalone compiler with multiple output modes
- **eshkol-repl**: Interactive REPL with LLVM ORC JIT compilation
- **CMake build system**: Cross-platform with Docker support
- **Comprehensive test suite**: 170+ test files covering all features
- **stdlib.o**: Pre-compiled standard library

#### Platform Support
- macOS (Intel x86_64, Apple Silicon ARM64)
- Linux (x86_64, ARM64)
- Docker containers (Debian, Ubuntu)

#### Build Requirements
- LLVM 17
- CMake 3.14+
- C17 runtime, C++20 compiler
- readline (optional, for REPL features)
