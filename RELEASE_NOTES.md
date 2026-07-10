# Eshkol v1.3.3-evolve — Release Notes

**Release Date**: July 10, 2026

Eshkol v1.3.3-evolve is an evolve point release over v1.3.2-evolve. It
corrects an overstated automatic-differentiation claim from the v1.3.2
CHANGELOG, closes out the region-escape evacuator series for the last
remaining heap subtype, and documents a subprocess race fix that had already
shipped. Full technical detail lives in [CHANGELOG.md](CHANGELOG.md); this
page is the user-facing summary.

**Release gates**: builds on the v1.3.2-evolve gates; the
`region_evac_subtype_coverage` gate now also exercises `PROMISE`
escape-then-force, and the `input2` gradient gate (24/24, JIT+AOT) now
verifies exact gradients rather than passing on unchanged zero-gradient
behavior.

## Highlights

### Automatic differentiation — correction

- **Exact tensor AD gradients for first-class losses and vector/learnable
  gamma.** v1.3.2's CHANGELOG entry for #212 claimed the `input2` gradient
  path was complete for `conv2d`/`batchnorm`/`layernorm`/`attention`; an
  adversarial audit found that change was a no-op (test and roadmap updates
  only, no gradient code). This release ships the real fix: first-class
  losses (no compile-time `Function*`) no longer silently lose the tangent
  through the forward-mode-dual closure path; batch-norm/layer-norm gamma/beta
  are now differentiated per-feature rather than as one scalar; and any
  remaining unsupported tensor-op backward path raises an explicit error
  instead of returning a silent zero gradient. (#229)

### Resident-memory correctness

- **Region escape evacuator now covers `PROMISE`.** A `delay`/`make-promise`
  created inside `with-region` that escaped the region no longer dangles
  after the region is popped. This completes the ESH-0214 region-evacuator
  series. (#230, ESH-0214e)

### Documentation

- **Subprocess `process-wait` kqueue race — documented.** The fix for a
  lost-wakeup race in `process-wait` on macOS (a child exiting before its
  `kevent` exit filter was registered could make `process-wait` block for the
  full timeout and misreport a dead process as still running) shipped in
  v1.3.2-evolve but was omitted from that release's notes. It is recorded
  here for completeness; no v1.3.3 code change was required.

---

# Eshkol v1.3.2-evolve — Release Notes

**Release Date**: July 9, 2026

Eshkol v1.3.2-evolve is an evolve point release over v1.3.1-evolve. It closes
the last resident-memory correctness gap for long-running loops, makes region
scoping safe under parallelism, completes the automatic-differentiation
`input2` gradient path, and adds developer tooling and Binary Lambda Calculus
depth. Full technical detail lives in [CHANGELOG.md](CHANGELOG.md); this page is
the user-facing summary.

**Release gates**: builds on the v1.3.1-evolve gates with a new poison-hardened
region-evacuator coverage gate
(`tests/memory/region_evac_subtype_coverage_test.sh`) that promotes and reads
back logic/workspace state over 1,000,000 region-wrapped mutations under
`ESHKOL_ARENA_POISON=1` at flat RSS.

## Highlights

### Resident-memory correctness

- **Forever-flat loops that mutate persistent logic/workspace state.** The
  region escape evacuator now deep-walks the `SUBSTITUTION`, `FACT`,
  `KNOWLEDGE_BASE`, `FACTOR_GRAPH`, and `WORKSPACE` subtypes (ESH-0214d). A
  resident tick loop can wrap its body in `(with-region ...)` to reclaim
  per-iteration transient garbage while its escaping knowledge-base/workspace
  state is promoted intact — previously those subtypes were shallow-copied and
  dangled into the freed arena. (#226)
- **Region scoping is thread-safe.** `parallel-map` combined with `with-region`
  no longer races on the shared current-arena slot. (#217)

### Automatic differentiation

- **`input2` gradients complete for `conv2d`/`batchnorm`/`layernorm`/
  `attention`** — the second operand (kernel / gamma / K / V) now receives
  gradients. (#212)

### Tooling and language

- **`eshkol-doc`** generates an API reference from Doxygen comments. (#213)
- **Binary Lambda Calculus universal machine**: `(blc-U)` decodes and runs
  Tromp's 232-bit self-interpreter; BLC8 byte I/O and ASCII lambda diagrams
  round out `core.blc`. (#218)

### Robustness

- Three deferred latent bugs triaged: ESH-0223, ESH-0227, ESH-0228. (#215)

---

# Eshkol v1.3.1-evolve — Release Notes

**Release Date**: July 9, 2026

Eshkol v1.3.1-evolve is a resident-robustness point release over
v1.3.0-evolve: two fixes aimed squarely at long-running/daemon processes and
large-persisted-state workloads, plus a comprehensive documentation pass.
Full technical detail lives in [CHANGELOG.md](CHANGELOG.md); this page is the
user-facing summary.

**Release gates**: builds on the v1.3.0-evolve release gates (see below) with
a new AOT flat-RSS regression gate
(`tests/memory/define_loop_flat_rss_aot_test.sh`) that compiles the
guard-wrapped self-tail-recursive `define`-loop shape ahead-of-time and fails
if peak RSS exceeds a generous flat threshold, so the ESH-0214b fix cannot
silently regress.

## Highlights

### Resident-robustness fixes

- **Long persisted-state files now read safely.** The reader's list parser
  (`read_list`) was rewritten from per-element native recursion to an
  iterative loop, so reading a long flat list — e.g. a 46K-entry persisted
  state file — no longer overflows the native stack. Verified: the pre-fix
  reader crashed (SIGBUS) at 20M elements; post-fix, the same input reads
  cleanly. (#191)
- **`define`-loop daemons now hold flat memory.** Automatic per-iteration
  arena-scope reclamation — previously limited to named-let loops — now also
  covers self-tail-recursive top-level `define` loops, and the escape
  analysis that gates it accepts a catch-all `guard` clause instead of
  rejecting any guarded body outright. This is exactly the shape of a
  production daemon/resident loop: a top-level `define` loop wrapped in an
  error boundary. Verified in AOT mode: a 1,000,000-iteration allocating
  guard-wrapped `define` loop holds peak RSS at 27MB with the fix on, versus
  2608MB with the fix off. (#192)

### Comprehensive documentation pass

- Doxygen doc-comments added across all 64 public headers (`inc/eshkol/**`)
  and most implementation files (`lib/**`).
- A new navigable documentation index (`docs/README.md`); orphaned
  (unindexed) docs reduced from 73 to 3.
- Press materials and website content updated to reflect the shipped v1.3
  state; roadmap views aligned with what has actually shipped.

### Known issues

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) and the CHANGELOG's Known
Issues section for the current, itemized list (none block ordinary use).

---

## Previous Releases

### Eshkol v1.3.0-evolve — Arbitrary-Order Automatic Differentiation

**Release Date**: July 7, 2026

Eshkol v1.3.0-evolve is the "evolve" release: arbitrary-order automatic
differentiation, full R7RS conformance on the portable differential
corpus, and a hardening pass across closures, tail calls, and long-running
processes.

**Release gates** (all green on the release SHA): ICC readiness oracle
`v1.3-evolve` ready (100/100, trace-verified); CI 14/14 lanes including
windows-arm64 lite/CUDA/XLA; SICP full-book gate 88/88 probes across all 5
chapters under both `-r` and AOT; reference-Scheme differential oracle 34/34
AGREE vs. chibi-scheme.

#### Highlights

##### Automatic differentiation, best-in-class and beyond

Eshkol's AD system already did exact forward-mode, reverse-mode, and
symbolic differentiation. v1.3.0-evolve adds a second axis on top: **order**.
A new Taylor-tower engine (13 phases, P0 through P12 — see the
[Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md)
and the [CHANGELOG](CHANGELOG.md#added--automatic-differentiation-taylor-tower-campaign-p0-p12)
for the full phase-by-phase breakdown) computes *every* derivative up to an
arbitrary order `k` in a single pass:

- **Arbitrary order** — `(taylor f x k)` and `(derivative-n f x k)` return
  the full coefficient series or the `k`-th derivative for any `k`, not just
  first/second order.
- **Exact, not approximate** — when the input is an exact number and the
  function only uses exact-preserving arithmetic, the coefficients come back
  as exact arbitrary-precision (bignum/rational) values instead of `double`
  approximations. Most autodiff systems, JAX included, only ever produce
  floating-point derivatives; Eshkol can hand you the exact rational
  derivative when the math supports it.
- **Validated** — Taylor models (`taylor-model`, `tm-range`, `tm-eval`) pair
  the polynomial with a rigorous interval-remainder bound, giving a
  *provable* enclosure of a function's range or value, not just a point
  estimate. This is a step beyond what mainstream AD/ML frameworks expose to
  user code at all.
- **Multivariate and sparse** — `mixed-partial`/`gradient-n` recover
  arbitrary-order mixed partials via a Griewank-Utke-Walther (GUW)
  propagation layer; `sparse-hessian`/`sparse-mixed-partials` exploit sparsity
  with graph-coloring so the cost scales with variable-interaction bandwidth,
  not dimension.
- **Composes with everything else** — tensor-valued towers differentiate
  through `matmul`/`conv2d`/`sigmoid`/`tanh`; reverse-over-Taylor lets
  `gradient` differentiate through a `derivative-n` call; checkpointed
  reverse-mode keeps the memory cost of high-order reverse AD sub-linear;
  towers work correctly through `if`/`cond`/named-let/recursion; and
  tower-based numerics (`taylor-ode-solve`, `taylor-root`,
  `taylor-inverse-series`) put all of this to work solving ODEs, root-finding,
  and series inversion.

See the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md)
for worked examples and the full API reference.

##### Full R7RS conformance on the portable corpus

A new reference-Scheme differential oracle diffs Eshkol's behavior against
chibi-scheme 0.12.0 across a 34-program portable R7RS corpus (numeric, list,
vector, string, char, binding, control-flow, equality, and I/O). It started
this release cycle at 27/34 (79.4%) and every divergence is now fixed:
**34/34 (100%) AGREE** with chibi-scheme on that corpus. Fixed along the way:
`apply` with leading arguments, multi-vector `vector-map`/`vector-for-each`,
quasiquoted vector literals, `cond`/`case` `=>` arrow clauses, an allocating
`vector-copy` (including on `#(...)` tensor-backed literals), the
`error-object?`/`error-object-message`/`error-object-irritants` condition
family, R7RS string-escaping in `write`, nested ellipsis (`x ... ...`) in
`syntax-rules`, and the 2-argument form of `substring`. See the
[CHANGELOG](CHANGELOG.md#fixed--r7rs-conformance) for the itemized list.

##### Robustness: closures, tail calls, and long-running processes

A cluster of fixes targets programs that run for a long time or recurse
deeply — the kind of bug that only shows up in production, not in a quick
test:

- Mutual tail calls (`even?`/`odd?`-style cross-function recursion) are now
  proper O(1)-stack R7RS tail calls on AArch64.
- Named-let loops are tail-call-optimized in every legal tail position, not
  just the immediate loop body — including tail calls made through a `guard`
  error-boundary wrapper.
- Curried closures can now capture up to 64 variables (up from a
  silently-corrupting ceiling of 16).
- A production-triggered class of unbounded RSS growth in long-running loops
  is fixed, and loops that are provably safe now get automatic,
  zero-annotation per-iteration memory reclamation.
- A graceful-shutdown race that could SIGSEGV after `SIGTERM` is fixed, deep
  recursion overflow now fails with a diagnostic instead of an unexplained
  `SIGILL`, and `eshkol-run -r`/AOT caching now correctly invalidates when an
  indirectly loaded/required dependency changes.

See [CHANGELOG.md](CHANGELOG.md#fixed--compiler--runtime-robustness) for the
full list with root causes.

##### Also in this release

- A new build integration surface: `--emit-depfile` plus a canonical
  `cmake/EshkolCompile.cmake` for consumers embedding the Eshkol compiler in
  their own CMake build.
- Browser REPL / WASM fixes so every example on [eshkol.ai](https://eshkol.ai)
  runs, including the tensor computing examples.
- A permanent, ICC-wired adversarial-testing infrastructure — differential,
  edge-matrix, AD-oracle, stress, VM-parity, depth-parametric, and external
  (reference-Scheme / sanitizer-fuzz / metamorphic) test pillars — so these
  classes of bug keep getting caught going forward. See
  [docs/TESTING.md](docs/TESTING.md).

##### Known issues

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) and the CHANGELOG's Known
Issues section for the current, itemized list (none block ordinary use).

### Eshkol v1.2.3-scale — Platform Artifact Closeout

**Base Release Date**: May 1, 2026
**Closeout Date**: May 20, 2026
**Platform Artifact Date**: May 25, 2026

Eshkol v1.2.3-scale is the platform-artifact closeout point release for
v1.2.0-scale, the *production-readiness* release. The v1.1
line proved the math (autodiff, tensors, the consciousness engine);
v1.2 makes it shippable: trained models save and load, error messages
point at the actual line, the Python FFI is stable and zero-copy,
deep recursion doesn't blow the stack on Darwin, and a long tail of
correctness/security bugs that surfaced under real workloads is now
fixed.

The headline addition isn't a feature — it's the edge-case regression
suite that catches every fix in this release going forward. The
v1.2.0 release shipped with 62 tests; the current v1.2.x Noesis M0
closeout build carries **87 passing edge/security tests**, a clean
37-suite aggregate gate, and a full Noesis aggregate smoke pass.

## v1.2.3 Platform Artifact Addendum (May 25, 2026)

`v1.2.3-scale` is a packaging and release-integrity patch over the v1.2.1
language/runtime surface. It supersedes the unpublished `v1.2.2-scale` tag
attempt by adding the hosted Windows x64 COFF linker fix needed for the full
artifact matrix:

- the release workflow now treats the 16-package platform set as a checked
  contract before publishing:
  - Linux x64/ARM64 lite/XLA/CUDA tarballs
  - macOS arm64/x64 lite/XLA tarballs
  - Windows x64/ARM64 lite/XLA/CUDA zips
- `SHA256SUMS.txt` is generated from the final merged `dist/` directory.
- the publish job refuses to append to or overwrite an existing GitHub release.
- `release_workflow_surface_test` pins this behavior in CTest so future
  release-workflow edits cannot silently drop platform artifacts.
- generated parallel worker initializer symbols are module-local on native
  Windows so hosted x64 release packages link cleanly against `stdlib.o`.
- the Homebrew formula template now targets the public `v1.2.3-scale` archive;
  the tap formula still needs its computed SHA256 after the release tarball is
  published.

## v1.2.1 Noesis M0 Closeout Addendum (May 20, 2026)

`v1.2.1-scale` closed the Noesis M0 audit path:

- `tests/v1_2_edge_cases` passes **87/87**, including shared
  hash-table mutation under `parallel-map`, late variadic REPL forward
  refs, binary I/O, match predicate binding, tensor pixel fill,
  first-class builtins, channels, threads, object-build CLI contract
  coverage, bounded HTTP server smoke coverage, and shell-level CLI/linker
  probes.
- `scripts/run_all_tests.sh` passes **37/37 suites** and **528/528
  self-reported individual tests**. The aggregate now counts suites that
  report `Results: N passed, M failed`, so the logic and v1.2 edge/security
  runners are included in the release total.
- Noesis `tests/smoke/all.esk` passes with `NOESIS_ALL_RC=0`.
- VM C API checks pass **81/81**, CTest passes **15/15**, and stress tests
  pass **3/3** on the final release-gate build.
- The previously intermittent dual-neural crash is fixed by
  serializing runtime hash-table access; the focused Noesis
  `dual_neural` smoke passed 8/8 stress repeats on the fixed build.
- Bug LL's underlying CLI behavior is fixed: `--emit-object` accepts
  compatibility flags, writes the requested `-o path`, and no longer
  creates the stale `.o.o` output.
- The Homebrew formula template points at the public `v1.2.1-scale` release
  archive; the public tap formula carries the computed SHA256 after tagging.
- The release workflow's platform asset matrix is now guarded by
  `release_workflow_surface_test`: every `v*` tag must publish the 16 expected
  Linux/macOS/Windows lite/XLA/CUDA archives, generate `SHA256SUMS.txt`, and
  refuse to append assets to an existing release.

## What's New in v1.2.0-scale

### Production Deployment

- **Model serialization** — `.eshkol-model` is an ESKB-extended
  binary format that round-trips trained networks (architecture +
  weights + metadata) so you can save a model on the training box
  and load it on the inference box.
- **Stable C ABI + Python bindings** — `inc/eshkol/c_abi.h` is the
  versioned public header; `pip install eshkol` gives you pybind11
  bindings with NumPy zero-copy interop.  Gradient computation,
  structured returns, and error recovery all crossed the FFI cleanly
  after the v1.2 hardening pass.
- **Per-thread arenas** — concurrent code paths
  (parallel-map workers, thread-pool tasks) now use thread-local
  arena slots so allocation in one worker never stomps another's.
- **Image I/O** — PNG/JPEG/WebP/BMP read/write/resize for vision
  pipelines, backed by native platform/system codec APIs
  (ImageIO/CoreGraphics on macOS, system libpng/libjpeg/libwebp on
  Linux, GDI+ on Windows).
- **Plotting stdlib** — inline matplotlib-style charts via PNG output
  for notebook-style workflows.

### Compiler Diagnostics

- **Actionable error messages** — compile errors now report the
  exact source line and column with a caret underline:
  ```
  /path/to/file.esk:6:4: error: Unknown function: undefined-fn
      6 |   (undefined-fn 1 2 3))
        |    ^
  ```
  Previously every error pointed at line 1 because the parser's
  comment-stripping reader consumed newlines.  The fix preserves
  newlines from comments and threads a cumulative file-line counter
  across `eshkol_parse_next_ast_from_stream` calls.

### Stdlib + Language

- **JSON Schema validation** (Draft 7 subset) — `json-schema-valid?`
  and `json-schema-validate` for experiment-manifest /
  preregistration enforcement.  Supports type, properties, required,
  additionalProperties, items, min/max length/items, minimum/maximum
  (with exclusive variants), enum, const, pattern (substring),
  oneOf / anyOf / allOf / not.  Auto-loaded via stdlib.
- **R7RS-compliant scoping for stdlib redefines** — user `(define
  (foo …))` after `(require stdlib)` cleanly shadows stdlib's `foo`
  at link time (LinkOnceODR linkage on stdlib functions) and at
  call-site lowering (variadic-info hygiene clears stale entries
  on redefine).  Previously a user redefine of a variadic stdlib
  function with a fixed-arity signature compiled with an
  arity-mismatch warning and crashed at runtime.
- **AD scalar derivative on inline lambdas** — `(derivative
  (lambda (x) …) point)` inside a wrapper function now correctly
  flows through the runtime closure dispatch.  Previously it
  returned -inf / wrong values because the new-style derivative
  codegen path bailed out without calling the closure for
  function-parameter operands.
- **Reflection** — `procedure-arity`, `record-fields`, `describe`
  for runtime inspection of user-defined procedures and records.
- **Memoization / LRU cache stdlib** — `(memoize fn :lru 256)`.
- **PRNG seeding + deterministic replay** — `(seed-prng! …)` and
  per-stream isolation for reproducible experiments.
- **Lazy sequences / streams** (SRFI 41).
- **Time API** — ISO-8601 parse/format + duration types.
- **Regex capture groups** — `regex-group` / `match-groups`.
- **CLI argument parser** — `(parse-args)` for noesis CLI entry
  points.
- **call-with-values + URL/base64url encoding** finalised.

### Build, Link, and Platform

- **macOS deep recursion** — every binary now ships with
  `LC_MAIN.stacksize = 512 MB` (`-Wl,-stack_size,0x20000000`) on
  Darwin.  The flag had only been wired into one of the two link
  paths in `eshkol-run`; the common compile-and-link path silently
  inherited the 8 MB default and any non-tail-recursive Scheme code
  hit `eshkol_check_recursion_depth + 4` with a SIGSEGV on its own
  frame push.
- **`--wasm` is self-contained** — the WASM emit path no longer
  falls through to native clang++ link.  `eshkol-run file.esk
  --wasm -o foo.wasm` produces the .wasm via LLVM in-memory codegen
  and exits cleanly; no spurious "_main referenced from
  initial-undefines" link errors.
- **Stdlib functions are weak-linked** — user code can override a
  stdlib symbol with their own definition without a "duplicate
  symbol" link error.  The fix mirrors the Windows
  `WeakAnyLinkage` path onto macOS/Linux LinkOnceODR.
- **AD value-typed captures** — derivatives that close over
  function-parameter `tagged_value` Arguments (e.g. `loss-fn`
  capturing `input`/`target`/`b` in `compute-loss-gradient`) no
  longer fail LLVM IR verification with "PtrToInt source must be
  pointer".
- **CI**: new `linux-x64-asan-ubsan` lane runs the v1.2 edge-case
  suite under `-DESHKOL_ENABLE_ASAN=ON -DESHKOL_ENABLE_UBSAN=ON`.
  `ESHKOL_ENABLE_TSAN=ON` and `ESHKOL_ENABLE_MSAN=ON` are scaffolded;
  TSan/MSan-built libstdc++ on apt.llvm.org is a v1.3 prerequisite.
- **Tagged release assets**: the release workflow treats the complete
  16-package platform set as a checked contract before publishing:
  Linux x64/ARM64 lite/XLA/CUDA tarballs, macOS arm64/x64 lite/XLA tarballs,
  Windows x64/ARM64 lite/XLA/CUDA zips, plus `SHA256SUMS.txt`.
- **`stdlib.o` rebuild correctness** — `file(GLOB_RECURSE …
  CONFIGURE_DEPENDS)` now watches every `lib/{core,math,signal,
  random,web,tensor,quantum,ml}/*.esk` so editing a transitive
  required module triggers a stdlib rebuild.  Previously only edits
  to `lib/stdlib.esk` itself did.

### Hardening

- **CRITICAL**: shell-string injection in `agent_subprocess.c` —
  fixed by switching to `posix_spawn` with `argv` arrays; the
  `popen("sh -c …")` path is gone.
- **CRITICAL**: Python FFI derivative-method AST injection in
  `eshkol_module.cpp` — fixed by canonicalising via the parser
  rather than string-substituting into source.
- **HIGH** (3 items): integer-overflow guards on arena, KB-load,
  and image-IO size computations (`__builtin_mul_overflow`).
- **HIGH** (4 items): path-traversal defence with percent-decode +
  component-check, TOCTOU race fixes on `stat → open`, and a
  Windows-subprocess buffer-size off-by-one.
- **HIGH**: 36 silent-swallow sites across the runtime now either
  surface the error or are documented as intentional.
- **MEDIUM**: ReDoS-resistant regex engine (counted-quantifier
  backtracker with bounded-state ceiling), SQL-injection guards on
  the persistence path, URL validator that rejects scheme
  smuggling.

### Testing

- **87-test v1.2 edge/security suite** at `tests/v1_2_edge_cases/`
  covering symbol consistency under gensym, AD tape state across
  worker threads, parser line tracking, stdlib symbol resolution,
  the JSON Schema validator, HTTP server smoke behavior, every real bug fix in
  this release.
  Runs under `bash scripts/run_v1_2_edge_cases_tests.sh` (also
  invoked by `run_all_tests.sh`). Includes shell-style tests for compile-time
  diagnostics, CLI/linker probes, REPL protocol checks, and server smoke paths
  that don't fit the `.esk → run → check exit` shape.
- **Master suite EXIT=0** end-to-end across 37 sub-suites and 528
  self-reported tests:
  features, stdlib, list, memory, modules, types, typesystem,
  autodiff, ml, neural, json, system, complex, cpp_type, vm, parser,
  control_flow, logic, bignum, rational, parallel, signal,
  optimization, examples, xla, gpu, error_handling, macros, repl,
  web, tco, io, benchmark, migration, codegen, numeric,
  v1_2_edge_cases.

## Carry-forward to v1.3

- **Native media stack** — use ImageIO/CoreGraphics, system
  libpng/libjpeg/libwebp, and GDI+ on the three host platforms so the
  active image backend does not rely on vendored third-party media
  code.
- **AD `input2` plumbing for non-matmul tensor ops** — the
  backward kernels for conv2d / batchnorm / layernorm / attention
  / multi-head-attention exist; the forward implementations need
  to be rewritten to multi-channel / per-feature shape so the
  Wengert tape can consume them.  Matmul (the only AD-supported
  tensor op exercised by the suite today) is correctly wired.
- **TSan / MSan CI lanes** — pending TSan/MSan-built libstdc++.
- **Spec-doc generator (`eshkol-doc`)** — extract type signatures
  + docstrings from the indexed module graph.
- **True module-private internals** — `(provide …)` is currently
  informational under both AOT and JIT (Bug Z); v1.3 reintroduces
  a proper rename pass while keeping cross-file calls to provided
  symbols working.
- **AD-1 follow-up** — re-extract `codegenDerivativeMonolith` into
  the new code path; the v1.2 fix delegates from `derivative()`
  to the monolith as a stop-gap.

---

# Eshkol v1.1.13-accelerate — Windows ARM64 + Release Workflow + VM Closure Fixes

**Release Date**: April 9, 2026

Eshkol v1.1.13-accelerate adds native Windows ARM64 support, rewrites the release workflow into a 16-lane build matrix that produces lite/XLA/CUDA variants for every supported platform, fixes two critical bytecode-VM closure bugs that affected the browser REPL and gradient descent demos, hardens setjmp/longjmp on Windows for both x64 and ARM64, and overhauls the website for full mobile responsiveness.

## What's New in v1.1.13-accelerate

### Windows ARM64 Native Support

- Full build path for Windows ARM64 via VS 2022 + ClangCL + LLVM 21 aarch64 SDK
- CMake auto-detects `clang_rt.builtins-{x86_64|aarch64}.lib` based on `CMAKE_VS_PLATFORM_NAME`
- Multi-arch DIA SDK lookup: scans both `Program Files` and `Program Files (x86)` for both `amd64` and `arm64`
- REPL JIT links the architecture-appropriate LLVM target libraries (`LLVMAArch64*` on ARM64, `LLVMX86*` on x64)
- 16 release artifacts per tag: 6 Windows (x64/arm64 × lite/xla/cuda), 6 Linux, 4 macOS

### setjmp/longjmp Cross-Platform Hardening

- Windows ARM64: uses `Intrinsic::sponentry` as the hidden `_setjmpex` context (matches Clang lowering)
- Windows x64: switched from `Intrinsic::localaddress` to `Intrinsic::frameaddress(0)` for the hidden context — produces stable, correctly-aligned frames
- Removed all compile-time `#ifdef _WIN32` branches in favor of runtime `Triple::isOSWindows()` — proper cross-compilation
- Dynamic `jmp_buf` sizing via `eshkol_jmp_buf_size()` runtime helper (no more hard-coded 256-byte buffers)

### Runtime Symbol Renames

POSIX shim functions are now renamed with an `eshkol_` prefix to disambiguate from MSVC's deprecated POSIX shims:

`fopen → eshkol_fopen`, `access → eshkol_access`, `remove → eshkol_remove`, `rename → eshkol_rename`, `mkdir → eshkol_mkdir`, `rmdir → eshkol_rmdir`, `chdir → eshkol_chdir`, `stat → eshkol_stat`, `opendir → eshkol_opendir`

Generated programs call `eshkol_runtime_init()` at start of `main()` (non-REPL mode).

### Codegen Error Handling

- New `fatal_codegen_error_` flag — codegen now fails hard on undefined-function/undefined-variable/private-symbol errors instead of silently emitting `printf`/`exit` runtime stubs
- New `declared_functions_by_ast` map keyed by AST node identity — fixes function resolution when multiple `define`s share a name within the same module

### VM Closure Bug Fixes (browser REPL + bytecode VM)

Two critical closure-handling bugs in the bytecode VM that broke autodiff demos involving captured upvalues:

- **Named-let nested closure PC offset**: When a lambda is created inside a `(let loop ...)` body, the loop's bytecode is inlined into the parent function with PC adjustments — but the inner lambda's `OP_CLOSURE` constant (its `func_pc`) was *not* offset by the loop's start position. The inner closure ended up jumping to a stale location with the wrong upvalue count, manifesting as "UPVALUE INDEX OUT OF BOUNDS" plus gradient always equal to 1 in named-let gradient descent.
- **Native 252 upvalue relay**: When a lambda inside a function captures a variable via the parent's upvalue (`is_local=false`), native 252 was reading `vm->stack[vm->fp + slot]` — treating the upvalue index as a stack-frame offset, reading whichever local happened to be at that slot. Fix: read from `vm->stack[vm->fp - 1]` (the parent closure per the calling convention), then index into `parent_cl->closure.upvalues[slot]`.

Together these restore correct gradients for **every** autodiff demo on the website. The "Train a Neural Network" front-page card now converges to ~0.891 over 3 data points, and the named-let gradient descent in `/learn` chapter 5 converges to `y/x`.

### CI / Release Workflow Overhaul

- Release workflow rewritten as two matrices: `unix-release-matrix` (10 jobs) + `windows-release-matrix` (6 jobs)
- New `publish-release` job downloads all artifacts, generates `SHA256SUMS.txt`, and publishes the GitHub release
- Per-architecture LLVM SDK caching on Windows runners (cache key includes `${arch}` and SDK version)
- CI workflow updated: `windows-2022` → `windows-latest`, `max-parallel: 2` Windows throttling
- Removed Docker-based XLA/CUDA build paths in favor of native CMake builds

### Website — Mobile Responsiveness

- Hamburger nav menu collapses the 7 top-level nav links on screens ≤720px; opens as a full-width dropdown; auto-closes when a link is clicked
- `html, body { overflow-x: hidden }` plus `min-width: 0` on flex/grid children — no more horizontal page scroll on any viewport (verified across 5 viewport sizes × 6 routes)
- Code blocks (`runnable-code` wrappers) now scroll horizontally *inside* the block instead of pushing the page wider
- `.docs-layout` switched from `1fr` to `minmax(0, 1fr)` — fixes the docs page reporting 972px wide on a 375px viewport
- `.comparison-table` becomes scrollable on ≤720px so the comparison table on `/downloads` doesn't push the page

### Browser REPL Error Display

- REPL now captures stderr (compile warnings, parse errors) into `_vmStderr` and displays them as `error: undefined variable 'foo'` instead of silently re-prompting
- Suppresses the trailing `()` NIL fallback when a compile error fired
- Shows `error: could not parse expression` when nothing parses
- Same fix applied to runnable code blocks (Run ▶ buttons across the site)

### Test Results

- 35/35 test suites, 100% pass rate on macOS ARM64, Linux x64, Linux ARM64, Windows x64, Windows ARM64
- 32/32 runnable site examples verified end-to-end in headless Chromium across mobile/tablet/desktop viewports
- All 16 release-build lanes green on the v1.1.13-accelerate tag

---

# Eshkol v1.1.12-accelerate — Toolchain Unification + Platform Hardening

**Release Date**: April 7, 2026

Eshkol v1.1.12-accelerate unifies the toolchain on LLVM 21 across all platforms, adds a native Windows build path via Visual Studio 2022 + ClangCL, fixes ARM64 and Windows x64 ABI issues in the runtime, adds clean URL routing to the website, and expands CI/CD coverage.

## What's New in v1.1.12-accelerate

### LLVM 21 Toolchain Unification

- Standardized entire build on LLVM 21 across Linux, macOS, and Windows
- New `cmake/LLVMToolchain.cmake`: authoritative LLVM version discovery and enforcement at configure time
- New `scripts/lib/llvm21-env.sh`: platform-aware LLVM 21 activation for all shell scripts
- Hard version check: configure fails with a clear error if LLVM major version is not exactly 21
- Removed misleading `LLVM 18+` compatibility branches from backend codegen

### Native Windows Build (Visual Studio 2022)

- Full native build via Visual Studio 2022 + ClangCL + LLVM 21 SDK
- Configures with `Visual Studio 17 2022` generator and `-T ClangCL`
- `region_escape_tagged_value_into` ABI fix: passes `eshkol_tagged_value_t` by pointer to satisfy Windows x64 calling convention for 16-byte aggregates

### ARM64 ABI Fix

- Fixed `call_thunk_closure` in `arena_memory.cpp:3908`: ARM64 returns 16-byte structs in register pairs (x0:x1), not via hidden return buffer
- Resolves dynamic-wind + call/cc thunk invocation on Apple Silicon and Linux ARM64

### Mutual TCO Fix

- `llvm_codegen.cpp`: version-gated tail call kind — `TCK_MustTail` on LLVM < 18, `TCK_Tail` on LLVM ≥ 18
- Fixes "LLVM ERROR: cannot use musttail" on Linux with LLVM 21

### Website — Clean URL Routing

- Navigation now uses `/downloads`, `/learn`, `/docs` etc. instead of `/#/downloads`
- GitHub Pages 404-redirect SPA routing for direct URL access
- History API (`pushState`/`popstate`) replaces `hashchange`

### CI/CD Expansion

- New GitLab CI matrix: Linux x64/arm64 × lite/XLA/CUDA + macOS × lite/XLA + Windows
- GitHub CI updated to LLVM 21 baseline across all runners
- Docker parity images (`docker/debian/`, `docker/ubuntu/`) updated to LLVM 21

### Test Results

- 35/35 test suites, 438/438 tests, 100% pass rate (macOS ARM64, Linux x64)

---

# Eshkol v1.1.11-accelerate - Performance Acceleration Release

**Release Date**: March 27, 2026

Eshkol v1.1-accelerate builds on the v1.0-foundation with comprehensive performance acceleration. Every v1.1 roadmap item is now complete: XLA backend (5/5), SIMD vectorization (4/4), concurrency (5/5), extended math (5/5), bignum/rational (6/6), consciousness engine (4/4), R7RS extensions (6/6), dual backend (7/7), and Windows platform (5/5) -- totaling 47/47 items.

## What's New in v1.1-accelerate

### Web Platform

Eshkol compiles to WebAssembly and runs in the browser. The project website ([eshkol.ai](https://eshkol.ai)) is itself written in Eshkol — 1,500+ lines compiled to a 502KB WASM binary.

- **Browser REPL**: A 63-opcode bytecode interpreter compiled via Emscripten runs in the browser with 555+ built-in functions. Users can evaluate Eshkol expressions without installing anything.
- **Automatic Differentiation in Browser**: Forward-mode AD via dual numbers works through the bytecode VM. Arithmetic opcodes detect dual number operands and dispatch to dual arithmetic (product rule, quotient rule, chain rule). `(derivative (lambda (x) (* x x)) 3.0)` returns `6` in the browser.
- **Interactive Examples**: Every code example on the website has a Run button with inline output. Examples span AD, neural network training, ODE solving, knowledge base queries, and exact arithmetic.
- **59 DOM Bindings**: Create elements, manipulate styles, handle events, draw on canvas, manage routing, access local storage — all from Eshkol compiled to WASM.
- **8-Chapter Interactive Textbook**: Progressive tutorial from basics through AD, tensors, scientific computing, and the consciousness engine — every example runnable.

### Bytecode VM — Production Complete

The bytecode VM is a fully production-grade execution engine:

- **555+ built-in functions** including character operations, bitwise logic, type predicates, string processing, list utilities, math extensions, complex numbers, and port I/O
- **Automatic differentiation in the VM**: Forward-mode AD via dual number propagation through all arithmetic and transcendental operations
- **R7RS control flow**: `call/cc` with continuation capture/restore, `guard`/`raise`, `dynamic-wind`, `values`/`call-with-values`
- **Exact arithmetic**: Rational literals (`1/3`), bignums, complex numbers, `+nan.0`/`+inf.0`/`-inf.0`
- **Consciousness engine**: Knowledge base queries with pattern matching, factor graphs, global workspace
- **Mutual recursion**: Top-level function defines can reference each other
- **System integration**: `directory-entries`, `command-line`, thread pool
- **176/176 tests passing**

### XLA Backend (Dual-Mode Architecture)

Tensor operations now dispatch through a multi-tier acceleration hierarchy:
- **StableHLO/MLIR path**: When MLIR is available, emits StableHLO ops for HW-optimized execution
- **LLVM-direct path**: Default mode with hand-tuned LLVM IR generation
- **Threshold dispatch**: XLA (>=100K elements) -> cBLAS (>=64) -> SIMD (>=64) -> scalar
- 6 core operations fully wired: matmul, elementwise, reduce, transpose, broadcast, slice

### SIMD Vectorization

Tensor loops are now explicitly vectorized with LLVM loop metadata and 64-byte aligned allocation:
- CPU feature detection for SSE2, SSE4.1, AVX, AVX2, AVX-512, and NEON
- SIMD micro-kernels for all tensor arithmetic and activation functions
- Loop vectorization metadata attached to all tensor operation back-edges
- Platform-specific tuning via cache-blocked matrix multiplication

### Signal Processing Library

New `signal.filters` module with 13 DSP functions:
- **Window functions**: Hamming, Hann, Blackman, Kaiser (with inline Bessel I0)
- **Convolution**: Direct O(N*M) and FFT-based O(N log N)
- **Filters**: FIR filter application, IIR Direct Form I
- **Butterworth design**: Lowpass, highpass, bandpass via bilinear transform
- **Analysis**: Frequency response (magnitude + phase)

### Optimization Algorithms

New `ml.optimization` module with 4 gradient-based optimizers:
- **Gradient descent** with configurable learning rate and convergence tolerance
- **Adam** (Adaptive Moment Estimation) with bias correction
- **L-BFGS** with two-loop recursion and backtracking Armijo line search
- **Conjugate gradient** (Fletcher-Reeves) with automatic restarts

All optimizers use the builtin `gradient` function (forward-mode AD with dual numbers).

### Parallelism & Concurrency

- `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`
- `future`/`force` for asynchronous computation
- Work-stealing thread pool with hardware-aware sizing
- Thread-safe arena memory management

### Arbitrary-Precision Arithmetic

- Bignum integers with full R7RS compliance (35 codegen gaps fixed)
- Rational numbers (exact fractions)
- Automatic overflow promotion (int64 -> bignum) and demotion
- All arithmetic, comparison, and I/O operations for both types

### Consciousness Engine

Novel AI primitives integrated at the compiler level:
- Logic programming (unification, substitutions, knowledge bases)
- Active inference (factor graphs, belief propagation, free energy minimization)
- Global workspace theory (modules, softmax competition, content broadcasting)
- 22 builtin operations spanning logic, inference, and workspace

### Dual Backend Architecture

Eshkol now ships with a complete bytecode VM alongside the LLVM native compiler:
- **Bytecode VM**: 64 opcodes, 250+ native calls, ESKB binary format, invoked via `-B` flag
- **Weight Matrix Transformer**: 126/126 inline programs and 123/123 traced programs passing, 3-way verified, 12.22M analytical parameters
- **qLLM Bridge**: Eshkol-to-qLLM tensor conversion for semiclassical inference

### Windows Platform Support

Native Windows builds are now supported:
- **MSYS2/MinGW64 native build** (PR #9 by mattneel)
- UTF-8-safe REPL with proper console code page handling
- Runtime DLL bundling for standalone distribution
- Path normalization for Windows-style backslash paths

### R7RS Compliance

- `call/cc` and `dynamic-wind`
- `guard`/`raise` exception handling
- Bytevectors, `let-syntax`/`syntax-rules`, symbol operations
- Tail call optimization validation
- `(load "path")` R7RS file loading support

### GPU Backends

- Metal backend for Apple Silicon (SF64 software float64 emulation)
- CUDA backend with cuBLAS integration
- 5 GPU operations: elementwise, matmul, reduce, softmax, transpose

## Test Results

35 test suites passing with 438 test files covering all subsystems.

---

# Eshkol v1.0.0-foundation - Production Release

**Release Date**: December 12, 2025

v1.0-foundation marks the first production release of Eshkol — a programming language that integrates compiler-level automatic differentiation, deterministic arena memory management, and homoiconic native-code execution as first-class language features rather than library overlays.

## What is Eshkol?

Eshkol is a production-grade Scheme dialect built on LLVM infrastructure, designed for gradient-based optimization, neural network development, and scientific computing. It combines functional programming elegance with native performance while eliminating garbage collection entirely.

## v1.0-foundation Achievements

### Complete Production Compiler

Eshkol v1.0-foundation delivers a fully functional compiler with:

- **Modular LLVM backend** with 21 specialized code generation modules
- **HoTT-inspired gradual type system** with bidirectional type checking
- **Comprehensive parser** supporting S-expressions, type annotations, pattern matching, and macros
- **Ownership and escape analysis** for automatic allocation strategy optimization
- **Module system** with dependency resolution and circular dependency detection
- **Interactive REPL** with LLVM ORC JIT compilation
- **170+ test files** providing comprehensive verification

### Compiler-Integrated Automatic Differentiation

First-class AD system operating at compiler, runtime, and LLVM IR levels:

- **Forward-mode AD** using dual number arithmetic
- **Reverse-mode AD** with computational graph and tape stack
- **Symbolic AD** through AST transformation
- **Nested gradients** up to 32 levels deep
- **8 vector calculus operators**: derivative, gradient, jacobian, hessian, divergence, curl, laplacian, directional-derivative
- **Polymorphic implementation** supporting int64, double, dual numbers, AD nodes, and tensors

### Deterministic Memory Management (OALR)

Zero garbage collection with ownership-aware lexical regions:

- **Arena allocation** with O(1) bump-pointer allocation
- **Escape analysis** automatically determining stack/region/shared allocation
- **with-region syntax** for lexical memory scopes  
- **Ownership tracking** preventing use-after-move at compile time
- **Fully deterministic** - zero GC pauses for real-time applications

### Comprehensive Language Features

**300+ language elements including:**
- 39 special forms (define, lambda, let/let*/letrec, if/cond/case/match, etc.)
- 60+ list operations with full Scheme compatibility
- 30+ string utilities
- 25+ tensor operations
- 10 hash table operations
- Complete I/O system with ports and exception handling
- Hygienic macros (syntax-rules)
- Pattern matching with 7 pattern types
- Multiple return values (values, call-with-values, let-values)

### Rich Standard Library

Modular library organization with pure Eshkol implementations:

- **stdlib.esk** - Central module re-exporting core functionality
- **math.esk** - Linear algebra (det, inv, solve), numerical integration, root finding, statistics
- **core.functional** - compose, curry, flip combinators
- **core.list** - higher-order functions, transformations, queries, sorting
- **core.strings** - extended string manipulation
- **core.json** - JSON parsing and serialization
- **core.data** - CSV processing, Base64 encoding

### Production-Ready Infrastructure

- **Cross-platform**: macOS (Intel/Apple Silicon), Linux (x86_64/ARM64), Windows (MSYS2/MinGW64)
- **Docker containers**: Debian and Ubuntu images
- **CMake build system**: Modern, maintainable build infrastructure
- **Comprehensive documentation**: Language specification, user reference, API docs
- **Package generation**: Homebrew formula, Debian packages

## Installation

### Quick Start

```bash
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run a program
build/eshkol-run tests/neural/nn_working.esk

# Start interactive REPL
build/eshkol-repl
```

### System Requirements

- **LLVM** 10.0+ (14+ recommended)
- **CMake** 3.14+
- **C17/C++20 compiler** (GCC 8+, Clang 6+)
- **readline** (optional, for REPL enhancements)

## Example: Neural Network Training

```scheme
(require stdlib)

;; Sigmoid activation
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

;; Mean squared error loss
(define (mse-loss pred target)
  (let ((diff (- pred target)))
    (* 0.5 (* diff diff))))

;; Forward pass
(define (forward weights bias input)
  (sigmoid (+ (tensor-dot weights input) bias)))

;; Compute loss gradient for backpropagation
(define (loss-gradient weights bias input target)
  (gradient 
    (lambda (params)
      (mse-loss 
        (forward (vref params 0) (vref params 1) input)
        target))
    (vector weights bias)))

;; Training works - automatic differentiation handles the calculus
```

## What Makes v1.0-foundation Special

### 1. Compiler-Integrated AD - Not a Library

Unlike JAX, PyTorch, or TensorFlow, Eshkol's automatic differentiation is built into the **compiler itself**, operating on AST, runtime values, and LLVM IR simultaneously. This enables differentiation of **any** Eshkol function without framework constraints or graph tracing overhead.

### 2. Homoiconic Native Code

Lambdas compile to LLVM-native code but retain their source S-expressions in closure structures, enabling both **runtime introspection** and **native performance** - a combination no other compiled language achieves.

### 3. Zero Garbage Collection

Arena-based memory management provides **fully deterministic** performance without GC pauses, making Eshkol suitable for real-time systems, trading algorithms, and control systems where predictable timing is critical.

### 4. Production-Quality Implementation

This isn't a research prototype - it's a complete compiler with comprehensive testing, thorough documentation, and a clear architectural foundation for future expansion.

## Documentation

- **[Language Specification](docs/COMPLETE_LANGUAGE_SPECIFICATION.md)** - Complete technical specification
- **[Language Reference](docs/reference/language/INDEX.md)** - User-focused reference with examples
- **[Vision Documents](docs/vision/)** - Purpose, competitive analysis, roadmap
- **[Architecture Guide](docs/ESHKOL_V1_ARCHITECTURE.md)** - Technical architecture overview
- **[API Reference](docs/API_REFERENCE.md)** - Comprehensive function documentation
- **[Quickstart](docs/QUICKSTART.md)** - Hands-on tutorial

## Known Limitations

v1.1-accelerate builds on v1.0-foundation. Remaining planned features:

- **Distributed computing** - Planned v1.2 (Q2 2026)

See [ROADMAP.md](ROADMAP.md) and [docs/vision/FUTURE_ROADMAP.md](docs/vision/FUTURE_ROADMAP.md) for detailed development plans.

## Next Steps

### For Users

1. **Explore the REPL**: `build/eshkol-repl`
2. **Try the examples**: `build/eshkol-run tests/autodiff/*.esk`
3. **Read the docs**: Start with [docs/ESHKOL_LANGUAGE_GUIDE.md](docs/ESHKOL_LANGUAGE_GUIDE.md)
4. **Experiment with AD**: The automatic differentiation system is production-ready

### For Contributors

1. **Review architecture**: [docs/ESHKOL_V1_ARCHITECTURE.md](docs/ESHKOL_V1_ARCHITECTURE.md)
2. **Check the roadmap**: [ROADMAP.md](ROADMAP.md) for v1.1/v1.2 plans
3. **See contribution guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Join development**: See open issues on GitHub for contribution areas

### For Researchers

1. **Study the AD implementation**: [docs/vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md](docs/vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md)
2. **Examine memory architecture**: [docs/breakdown/MEMORY_MANAGEMENT.md](docs/breakdown/MEMORY_MANAGEMENT.md)
3. **Analyze type system**: [docs/breakdown/TYPE_SYSTEM.md](docs/breakdown/TYPE_SYSTEM.md)
4. **Explore homoiconic closures**: [docs/vision/AI_FOCUS.md](docs/vision/AI_FOCUS.md)

## Acknowledgments

Eshkol v1.0-foundation represents years of research and implementation, synthesizing ideas from:
- **Scheme** for elegant functional programming
- **LLVM** for world-class code generation
- **Homotopy Type Theory** for rigorous type foundations
- **Region-based memory** research for deterministic allocation

We thank early testers and contributors who provided valuable feedback during development.

## License

Eshkol is released under the **MIT License** - see [LICENSE](LICENSE) for details.

## Contact

- **GitHub Repository**: https://github.com/tsotchke/eshkol
- **Issues**: Bug reports and feature requests
- **Discussions**: Technical questions and community engagement

---

**Eshkol v1.0-foundation** establishes a new standard for programming languages combining automatic differentiation, deterministic memory, and homoiconic native code. This is not a preview - this is only the beginning. Eshkol has a production-grade compiler ready for gradient-based computing, neural network development, and scientific applications where mathematical correctness and performance are non-negotiable.

*Where mathematical elegance meets uncompromising performance.*
