# Changelog

All notable changes to Eshkol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **ADR-0000 Stage 1, phase A: the frontend node-identity substrate.** Every
  AST node the parser produces now carries a stable `NodeId`, and a side table
  maps that id to a `SourceSpan` — the first column of the
  `NodeId -> {SourceSpan, BindingId, TypedExprInfo}` substrate that ADR-0000 §7
  risk 1 calls "the single most important co-design constraint in the whole
  program", and that ADRs 0004, 0006 and 0008 all have to be built on if the
  compiler, LSP, docs, REPL and VM are to give one answer rather than five.

  New: `inc/eshkol/frontend/node_identity.h` (the API),
  `lib/frontend/node_identity.cpp` (a chunked, lock-free-read span table),
  `eshkol_ast_t::node_id` (the 4-byte key; the payload stays out of line, per
  ADR-0008 §4.2), and `tests/frontend/node_identity_test.cpp`.

  The parser's 32 location-stamping sites now write the location and the
  identity in one statement, so the two cannot drift apart, and the stream
  reader closes each top-level form's span with a measured *extent* — the first
  place in the frontend that records where a construct ends rather than only
  where it begins.

  `NodeId`s are tagged, not bare indices, because `eshkol_ast_t` is declared
  uninitialised in a dozen places (`macro_expander.cpp`, `sexp_to_ast.cpp`,
  `introspection.cpp`) and an unset field holds garbage. A garbage word is
  rejected by its tag and again by its bound, so it reads as "unknown" — the
  same discipline `source_file_id` already follows, for the same reason: a
  diagnostic may fail to name a location, but it must never name a wrong one
  confidently.

  Strictly additive. `line`, `column` and `source_file_id` keep their exact
  previous values and every consumer that has not moved onto the substrate
  reads what it always read.

- **The LLVM codegen dispatcher resolves diagnostics through the substrate.**
  `codegenAST` asks `NodeId -> SourceSpan` for the file, line and column it
  reports, falling back to the node's own fields when the substrate does not
  know a node (one synthesized after parsing — an import lowered to a `define`
  alias, a macro expansion). This is the first real consumer, and it is the
  diagnostics path deliberately: it is where ESH-0364 (a location naming the
  wrong file) and ESH-0365 (a location on the closing paren) both landed.

  Emitted locations are unchanged — the span was recorded from the same token
  that set `line`/`column`. What changes is that a node's *file* no longer
  depends on the traversal that reached it: `source_file_id` is stamped only on
  top-level forms and every inner node borrowed the ambient context, whereas in
  the span table each stamped node carries its own file. `ScopedAstProvenance`
  gained the integer fast path its own comment always claimed, now that it runs
  per node instead of per top-level form.

- **`scripts/run_node_identity_gate.py` measures substrate coverage.** With
  `ESHKOL_NODE_IDENTITY_STATS=1` every frontend process prints
  `eshkol-node-identity: allocated=N queried=N resolved=N located=N extent=N`
  as it exits. The gate compiles a mixed corpus, aggregates, and emits
  `node_identity_substrate_present` and `node_identity_span_coverage` as ICC
  `runtime_event` traces against a monotonic floor in
  `tests/coverage/NODE_IDENTITY_BASELINE.json`. Graded by the new
  `adr0000-s1-identity` completion oracle.

  The gated number is measured at a *consumer*, not at the parser: a
  parser-side count says only how many ids were minted, never whether the
  answer arrived where it was needed. "Has an identity", "has a location" and
  "has an extent" stay three separate numbers so none can be read as another.
  This is what makes the stage falsifiable, which is the whole point of
  ADR-0000's gates.

  Phase A only, and the oracle says so: `BindingId` (ADR-0006 slices 1-2),
  `TypedExprInfo` (ADR-0004 spine part 1), the one semantic tooling core
  (ADR-0008 M0/M1 — the LSP still counts parentheses by hand), byte-offset
  canonical spans and the expansion-origin table are all still outstanding, and
  none of them is half-built here.

- **The bytecode VM reclaims memory: a Stage-1 OALR region evacuator ports
  `with-region` reclamation to the VM heap (SW-14, the v1.3.5 flagship item).**
  `(with-region ...)` used to lower to `begin` on the VM. The body ran, its
  value was returned, and not one byte came back: measured on
  `tests/memory/vm_region_growth_watchdog_test.esk`, peak RSS was 1.503 GB
  *with* the wrapper and 1.504 GB *without* it — the same to within 0.06%. The
  form was inert, not merely weak, and the VM had no heap reclamation of any
  kind.

  It now reclaims, and the claim is measured rather than asserted. The same
  fixture, swept by iteration count: **26 MB at 1 000 iterations, 26 MB at
  4 000, 26 MB at 16 000** — sixteen times the work for two megabytes — against
  **796 MB** for the identical program on the identical binary with
  `ESHKOL_VM_REGION_EVAC=0`. Gated by
  `tests/memory/vm_region_flat_rss_test.sh`, which requires the flat curve, the
  on/off separation, *and* the printed answer to be identical either way.

  The port matches the native engine's semantics, not its implementation.
  Native copies the escaping subgraph (Cheney-style, forwarding map, mutation
  write barrier at every store into a longer-lived slot). The VM marks from its
  root set and sweeps at arena-block granularity instead, because a VM value
  addresses the heap by a small integer index rather than by pointer: a copying
  evacuator there would have to rewrite indices, and a missed rewrite aliases a
  live object and returns a wrong *value* rather than crashing — the failure
  mode the SW-14 ruling called strictly worse than the leak. Marking moves
  nothing, so `eq?` identity, shared structure and cycles survive without
  special handling. Live objects' fixed-size headers are copied one level out,
  which needs no layout knowledge because the object table is the only holder of
  that address.

  Coverage is total by construction. `vm_evac_subtype_table[]` classifies the
  **full 33-wide heap tag space** — the 28 `HeapType` members plus the three
  tags `vm_geometric.c` defines outside the enum as bare macros and the two
  unassigned slots — with a compile-time span check, a fatal startup check that
  every row is filled in, and a `default:` arm that pins the region rather than
  guessing. That width is the ESH-0214d lesson applied up front: a switch over
  the enum alone would have fallen through for the manifold tags exactly as
  `evac_kind_for` once fell through for KB/FACT/SUBSTITUTION/WORKSPACE.
  `tests/memory/vm_region_evac_subtype_coverage_test.esk` builds one live
  instance of every subtype a program can construct inside a region, escapes it,
  drops 600 regions' worth of sibling garbage and reads each one back to compare
  its contents — under `ESHKOL_ARENA_POISON=1` (dead blocks stamped `0xCB` and
  kept mapped, retired indices never recycled), under an audit that
  independently scans the object table for a surviving reference to a retired
  index, and with reclamation disabled.

  A raise crossing a region and a continuation transfer out of one go through
  the same teardown call as normal exit, so the structured and unstructured
  surfaces cannot drift apart — the discipline native keeps around
  `eshkol_region_unwind_to()`.

  Every case the evacuator is not certain about degrades toward the leak and
  never toward a dangling index: an unclassified subtype, a continuation
  captured inside a region, or a failed bookkeeping allocation all pin the
  region, which is then promoted whole.

### Changed

- **Outside a region the bytecode VM still does not reclaim**, and the heap
  growth watchdog stays for exactly that case. What changed is what it must not
  say: the interim note claiming `with-region` reclaims nothing is gone, the
  budget diagnostic now names the mechanism that *does* reclaim, and
  `tests/memory/vm_region_growth_watchdog_test.sh` pins that the same allocation
  volume which trips the budget unwrapped does not trip it wrapped.
- The user-reachable region **handle** surface (`region-open` / `region-close`)
  remains bookkeeping-only on the VM and announces that at the point of use.
  A handle can be closed out of order, from another dynamic extent, or never,
  whereas `with-region`'s lexical extent tells the teardown where the region
  ends — which is why the lexical form landed first. Stage-2.
- New environment variables, all documented in
  `docs/reference/runtime/environment-variables.md`: `ESHKOL_VM_REGION_EVAC`,
  `ESHKOL_VM_REGION_VERIFY`, `ESHKOL_VM_REGION_VERIFY_FATAL`,
  `ESHKOL_VM_REGION_COMPACT`, `ESHKOL_VM_REGION_RECYCLE`. `ESHKOL_ARENA_POISON`
  — the same variable the native arena reads — now also arms the VM's.

### Known limits of Stage-1

- An escaping object with an **out-of-line payload** (a vector's element array,
  a bignum's limbs) keeps the arena block that payload occupies; escaping
  cons/closure structure is copied out exactly. A cons-only loop is therefore
  perfectly flat and a payload-heavy one is merely much smaller.
- A **continuation captured inside a region** pins that region.
- Objects promoted out of a region live in the enclosing arena for its lifetime,
  which is OALR's semantics and is equally true natively.

### Fixed

- **Python bindings: NumPy capsule keeps the tensor buffer alive (#458,
  audit H1).** A NumPy array exported from a tensor `eval()` held no
  reference back to its owning `Context`; deleting (or losing the last
  reference to) the `Context` object called `eshkol_ffi_shutdown()`
  unconditionally, with nothing to stop that while an exported array — or
  a view/slice/reshape of one — was still alive and depending on that
  memory staying valid. `EshkolContext` now holds the context as a
  `std::shared_ptr`, and every exported array's `base` capsule carries its
  own copy; real shutdown is deferred until every holder, including every
  live array's capsule, has released its reference. Closes
  `.icc/silent-wrong-ledger.yaml` SW-44 (interop/lifetime, SILENT-WRONG
  bucket — the pre-fix behavior read silently corrupted memory at exit 0,
  no diagnostic). New regression: `tests/bindings/python_capsule_lifetime_test.py`,
  wired into `ctest` as `python_bindings_capsule_lifetime`. See
  [docs/reference/bindings/python.md](docs/reference/bindings/python.md).

### Added

- **Assurance wave 1: ledger-integrity and oracle-schema gates with
  self-tests (#454).** `scripts/check_ledger_integrity.py` fails
  `.icc/silent-wrong-ledger.yaml` on a parse error, a duplicate `id`
  across the file, or an entry missing a required field or closure
  evidence (`SW-33`/`SW-35`/`SW-42` were each independently double-
  allocated across branches, invisible to a textual merge).
  `scripts/check_oracle_schema.py` fails `.icc/completion-oracles.yaml` on
  a parse error or a structurally invalid criterion, and always reports
  declared-vs-graded criteria counts per oracle. `scripts/gate_no_silent_wrong.py`
  gains `--self-test`. All three wired into the `eshkol-compiler-readiness`
  oracle, added as `ctest` entries, and run in a new `assurance-gates` CI
  job. See [docs/TESTING.md](docs/TESTING.md#assurance-gates-v135-wave-1-454).

### Changed

- **CI: docs-only PRs now get every required context reported (#455).**
  `paths-ignore` on the `pull_request` trigger previously meant a
  docs-only PR (like this one) never started the main workflow at all, so
  8 of 9 required branch-protection contexts never reported a status and
  permanently blocked the PR. The docs-only decision moved into a
  job-level `changes` gate instead, so a docs-only PR now gets every
  required context reported as skipped. See
  [docs/TESTING.md](docs/TESTING.md#ci-docs-only-prs-now-actually-report-required-contexts-455).

### Added

- **R7RS 7.1.1 vertical-line symbol syntax: read and write (#462).**
  `<identifier> -> <vertical line> <symbol element>* <vertical line>` is
  one of R7RS's three `<identifier>` productions; Eshkol previously
  implemented only the other two, so `'|weird sym|` lexed as two separate
  tokens. All four readers (native tokenizer, VM tokenizer, native
  runtime `read`, VM runtime `read`) now accept the full `<symbol
  element>` alphabet, including the mnemonic escapes, `\|`, and
  `\x<hex>;`. The bars request a verbatim spelling (`#!fold-case` does not
  apply inside them), and `|.|` is an ordinary symbol distinct from the
  bare `.` dotted-pair delimiter. `write` emits bars only when a name
  cannot be spelled bare under the R7RS grammar; `display` never bars.
  Shared predicate/escaper in `inc/eshkol/core/symbol_syntax.h` keeps the
  native and VM writers byte-identical. New regression:
  `tests/features/pipe_symbol_test.esk` (51 checks), run on native
  JIT/AOT/VM as a three-way parity check. See
  [Complete Language Specification §2.1.6](docs/COMPLETE_LANGUAGE_SPECIFICATION.md#216-symbol-interned-symbol).

### Documentation

- **v1.3.5 documentation wave.** `ROADMAP.md` re-dated (maintainer ruling R1,
  executed): the previously published v1.4-v2.0 dates were not achievable at
  measured velocity (the v1.3.1-v1.3.4 line averaged ~5 weeks/point release);
  v2.0 moves from the previously published "Q1 2027" to ~Q4 2028. Added the
  six standing workstreams every release now draws from (W1 resident/DBSP
  spine, W2 assurance, W3 performance, W4 codebase health, W5 interop &
  adoption, W6 two-tier distributed computing — PJRT/XLA scale-tier plus a
  native exact-allreduce mesh tier), replaced the stale per-version AD
  staging bullets with pointers to the already-shipped P0-P12 truth, and
  added the v1.3.5/v1.4.1/v1.5.1/v1.6.1/v1.8.1/v1.9.1/v1.9.2 point-release
  rows. `docs/KNOWN_ISSUES.md`'s future-releases table re-pinned off the new
  ladder (distributed/multi-GPU rows point at the W6 ladder instead of a
  fixed version; PGO -> v1.5.0-intelligence; ONNX -> post-training-win, no
  fixed date; Python bindings row tracks the v1.4.0-connection interop wave).
  `docs/FEATURE_MATRIX.md`'s historical-snapshot roadmap section corrected
  (it had listed Vulkan Compute, ONNX export, quantization, and distributed
  training as SHIPPED in v1.2-scale; none of those shipped) and re-dated to
  match. Press sheets (`press/ESHKOL_DESCRIPTION_COPY.md`,
  `press/ESHKOL_PRESS_INFORMATION_SHEET.md`) refreshed with numbers
  re-measured this cycle against a from-source build of commit `694c3179`:
  exact rational derivative (`(derivative-n g 1/3 1)` => `16/3`, exact),
  the H2 vibrational example (5003.2 cm⁻¹), the Ozaki-II Metal exact-GEMM
  certification gate (25/25, 0 mismatches), a fresh CHSH run (S = 2.835,
  gate `2.4 < S <= 2.95`), gradient parity across native JIT / native AOT /
  bytecode VM (byte-identical `#(24 57)`), and the ESH-0214b flat-RSS AOT
  gate (8 MB vs. 2,620 MB with the fix compiled out). Added a new
  [Python bindings reference page](docs/reference/bindings/python.md) (no
  such page existed previously) documenting the `Context.eval`/
  `derivative`/`gradient` API and the #458 capsule-lifetime guarantee, with
  a working example re-run against a from-source build including the
  merged fix; flagged in passing that the module's own docstring example
  (`ctx.derivative('sin', 0.5)`) does not work against the current
  `func_source` validation and needs a full `(lambda ...)` form instead.
  Documented the #454 assurance gates and #455 CI fix in
  [docs/TESTING.md](docs/TESTING.md), re-running all three gates'
  `--self-test` modes plus both non-self-test invocations against the
  real repo files. Added a "Bytecode VM Region Reclamation" section to
  [docs/breakdown/RUNTIME_CONFIGURATION.md](docs/breakdown/RUNTIME_CONFIGURATION.md)
  (the one doc file #461 didn't already normalize) with the full
  `ESHKOL_VM_REGION_*` variable table and a fresh flat-RSS measurement
  (25/26/27 MB at 1,000/4,000/16,000 iterations vs. 793 MB with the
  evacuator disabled); `docs/VM_PARITY.md` checked and needs no changes
  (its row counts still match `tests/vm_parity/PARITY.tsv` exactly).
  `ROADMAP.md`'s v1.3.5 section and Release Timeline row updated to mark
  the evacuator, H1 fix, and assurance-wave items shipped rather than
  planned. Documented the #462 vertical-line symbol syntax in
  [Complete Language Specification §2.1.6](docs/COMPLETE_LANGUAGE_SPECIFICATION.md#216-symbol-interned-symbol),
  [Language Guide](docs/ESHKOL_LANGUAGE_GUIDE.md)'s Data Types table, and
  [FAQ.md](docs/FAQ.md)'s R7RS conformance answer, re-running
  `tests/features/pipe_symbol_test.esk` myself on native JIT, native AOT,
  and the bytecode VM (51/51 checks, 0 errors, all three paths agree).
  Checked #406 (Moonlab pin bump to the real v1.2.0 tag SHA,
  `e441957b`→`4bf83a6c`) against every doc referencing the Moonlab
  version: the published label was already "v1.2.0" everywhere and stays
  "v1.2.0" — the bump corrects an internal SHA/tag mismatch, not the
  version Eshkol advertises — so no doc text needed to change.

## [1.3.4-evolve] - 2026-07-31

A resident-correctness release over v1.3.3-evolve. Every defect surfaced by
long-duration resident workloads is fixed at the architectural root: automatic
per-iteration memory reclamation now matches explicit `with-region` even for
loops that mutate persistent state, `parallel-map` is race-free for
collection-valued closures, gradients are exact through every callable form
(indirect and curried, with no finite-difference fallback in the gradient path)
and through every differentiation point classified by its runtime value, printed
floats round-trip, and the strict type checker accepts idiomatic
dynamic-but-validated code. `gradient` now runs at full parity on the bytecode
VM as well as native codegen, so Eshkol is self-differentiating on every
substrate. The release also lands the high-precision numerics wave (Ozaki-II
exact and reduced-precision GEMM tiers on Metal and CUDA, a mixed-precision
linear solver, and a native 128-bit integer type), a Moonlab v1.2.0 quantum pin
with quantum-natural-gradient support, full hosted-VM tensor-matmul parity, and
a round-tripping numeric printer. It also hardens the toolchain (transitive
FFI-link discovery with fatal link failures, Homebrew-compatible
system-dependency builds) and the assurance surface (dynamic edge coverage
across every new-feature family).

The second half of the cycle is a consumer-hardening correctness wave, and its
organising principle is that a wrong answer must not be able to look like a
right one. An emitted compile-time error now prevents artifact emission and
execution, which turned a family of silent wrong answers into build failures
and exposed the rest: exactness is now decided by an operand's runtime tag
rather than by a result's value shape, on the native flonum integer-division
family and on the bytecode VM's whole numeric surface; automatic
differentiation answers exactly at exact points, survives per-iteration nursery
reclamation, and no longer produces zeros above gradient arity 16;
`define-library` and `import` resolve same-unit libraries on all three back
ends; and `--shared-lib` links a real, C-ABI-correct shared library instead of
exiting zero with no artifact. Alongside those corrections the release adds a
portable event loop (kqueue / epoll / IOCP), a fixed-point and `i128`
exact-accumulation engine, the qLLM bridge implementation that its documented
backward rules were waiting for, embedding and Fréchet-mean backward passes,
and a release gate that finally reads CTest results as oracle evidence.

### Added

- **User-reachable region handles: `region-open` / `region-close` /
  `region-open?` (#341).** A non-lexical surface over the region machinery
  `with-region` already uses, for loop shapes where a lexical block is awkward.
  The motivating case is an autodiff training step: the automatic per-iteration
  nursery (ESH-0214e) disqualifies any loop body containing a `gradient` op, a
  `set!` or a `tensor-set!` — a training step trips all three, by design — so a
  161-parameter MLP doing a full-batch `gradient` per step grew ~123 MB/step
  unbounded. `(region-open ['name] [size])` returns an opaque exact-integer
  handle; `(region-close handle v ...)` deep-promotes the named values out
  through the validated escape evacuator (interior-pointer walk included) and
  reclaims everything else. Measured on that loop, peak RSS is **flat** at
  131-132 MB across 5/10/20/40 steps against 632/1258/2510/5013 MB unscoped,
  with bit-identical trained parameters. `with-region` remains the recommended
  default and is unchanged: it cannot be left un-closed.

  Safety is the whole design. The handle is a slot index plus a **generation**
  counter rather than a pointer, so every stale token is detectably stale:
  double close, use after close, a token from another thread and a fabricated
  integer all fail validation and raise a clean catchable error instead of
  touching freed memory. Closing an outer handle while an inner one is live is a
  **defined cascade** (inner regions closed innermost-first, keeps promoted at
  every level, inner tokens invalidated) rather than an error, because that is
  the identical operation an unwind performs. Never closing is bounded: the 65th
  simultaneous handle raises. A loop using handles is excluded from the
  automatic nursery, so the two mechanisms never nest unexpectedly.

- **Non-local exits now unwind regions (#341).** A `raise`/`guard` or a `call/cc`
  escape crossing an open region closes it, after **deep-promoting the in-flight
  value** (the raised value, or the value delivered to the continuation) out of
  every region being torn down, and restoring the allocation-routing slot before
  any arena is freed. The region depth is recorded as a mark beside the existing
  `wind_mark` / `promise_mark` on the exception-handler record and the captured
  continuation state. This also **fixes `with-region`**, which previously leaked
  its region on a `raise` out of the body *and* left the shared allocation slot
  pointing at an arena that was never freed. All teardown — explicit close,
  out-of-order cascade, `with-region` exit, raise, continuation escape — now
  funnels through one `eshkol_region_unwind_to()` primitive, so the structured
  and unstructured surfaces cannot drift apart.

- **INT8 tensor-core Ozaki f64 GEMM (CUDA, opt-in).** A new f64 GPU matmul path
  recovers FP64-accurate `C = A*B` from the INT8 (IMMA) tensor cores, which run
  ~500x faster than the deliberately crippled native FP64 pipeline on
  consumer/prosumer NVIDIA GPUs (GeForce Ampere f64 = 1/64 FP32). Following the
  Ootomo-Ozaki-Yokota scheme, each f64 operand is scaled per-row (A) / per-col
  (B) into `[-1,1]`, sliced into `T+1` signed 7-bit integer slices, multiplied
  as INT8->INT32 GEMMs via `cublasGemmEx` on the fast **TN/IMMA** layout
  (transposed B-slices — mandatory on sm_86, a 3.7x cliff otherwise; Blackwell
  is layout-indifferent so TN is safe everywhere), and reconstructed to f64 with
  a **diagonal-fused** int32 reconstruction (same-weight slice-pairs on a
  diagonal `d=p+q` accumulate via `beta=1` into int32-safe grouped buffers, then
  a single fused kernel), provably int32-exact at any N. The int32 accumulation
  is exact; the only error source is dropping slice-pairs with `p+q>T`. Measured
  on an RTX 3090 (sm_86): **4.74 TFLOP/s-eq at full f64** (normwise error
  2.7e-15) = **8.8x native cublasDgemm**, up to 16.6x at ~1e-11; on an RTX PRO
  6000 Blackwell: **~30 TFLOP/s** (20x native f64). Opt-in and default OFF —
  select via `ESHKOL_CUDA_F64_KERNEL=ozaki-int8` (the f64 GPU matmul otherwise
  stays `cublasDgemm`). The accuracy/throughput knob is `ESHKOL_OZAKI_CUDA_T`
  (default 6 = full f64 ~1e-15; T=4 ~1e-11 and ~2x faster), mirroring the Metal
  Ozaki-II env conventions. `K` is guarded below 133,000 for int32 exactness;
  out of range it falls back loudly to `cublasDgemm`. Wide-dynamic-range inputs
  stay accurate via the per-row/col scaling (7.5e-15 on 1e-3..1e3 data). The
  path is auto-gated to engage only when a measured crossover beats native
  `cublasDgemm`, so small or skinny GEMMs keep native DGEMM, and a cost-model
  accuracy-budget selector chooses between the native DGEMM and INT8-Ozaki tiers
  per problem shape and requested accuracy (#346, #347). Adds
  `tests/gpu/cuda_ozaki_correctness_gate.sh` / `cuda_ozaki_correctness_test.esk`
  (INT8-Ozaki vs an independent CPU f64 reference across
  integer/fractional/pi-e/wide-magnitude regimes at K up to 4096, a T=4/6 sweep,
  and an out-of-range-knob loud-clamp check) and the
  `cuda-ozaki-int8-correctness` ICC oracle.
  (`lib/backend/gpu/gpu_memory_cuda.cpp`, `lib/backend/gpu/gpu_cuda_kernels.cu`)
- **Native 128-bit integer type (`i128`).** A first-class, fixed-width,
  two's-complement signed integer (range −2¹²⁷…2¹²⁷−1) that lives **off** the
  numeric tower: unlike bignum (which grows), i128 arithmetic **wraps** modulo
  2¹²⁸, and it never auto-promotes — every crossing to/from the tower is an
  explicit conversion. Heap-boxed under a new subtype `HEAP_SUBTYPE_I128` (25)
  with a 16-byte little-endian `{lo, hi}` payload whose layout matches the
  planned two-u64 FFI ABI. Full surface on **both** the native codegen path and
  the bytecode VM: constructors/predicate `i128` / `int->i128` / `string->i128`
  (full range incl. −2¹²⁷) / `i128?`; wrapping arithmetic `i128-add` / `-sub` /
  `-mul` / `-neg`; shifts `i128-shl` / `-ashr` / `-lshr` (count 0…127, out of
  range raises); comparisons `i128=?` / `<?` / `>?` / `<=?` / `>=?`; truncated
  `i128-quotient` / `i128-remainder` (C sign semantics; divide-by-zero raises);
  conversions `i128->string` / `i128->int` (raises out of fixnum range); and
  decimal `display`/`write`. The pure arithmetic core (`inc/eshkol/core/i128.h`)
  is shared verbatim by the native runtime (`lib/core/i128_runtime.cpp`,
  arena-boxed, `eshkol_raise`) and the VM (`lib/backend/vm_native.c`, region-heap
  boxed) so both compute bit-identical results. Docs:
  `docs/reference/language/i128.md`; tests: `tests/types/i128_test.esk`
  (native + VM parity) and `tests/types/i128_error_test.esk`. The reduced-precision
  Metal Ozaki tier is certified exact against this i128 path across the same
  correctness regimes (`tests/gpu`), so the GPU fast tier's accuracy claim is
  checked against a bit-exact reference rather than another float path.
- **Linear `Qubit` type and linear-parameter enforcement in `define`.** A
  first-class linear `Qubit` type whose values must be used exactly once; a
  `define`d function may declare linear parameters and the checker enforces the
  use-exactly-once discipline (double-use and drop are both rejected), giving
  quantum-register operations a no-cloning guarantee at the type level. This
  extends the HoTT type surface (`lib/types/hott_types.cpp`,
  `inc/eshkol/types/hott_types.h`).
- **`linear-solve` — full-f64 dense linear solver with a mixed-precision fast
  path.** `(linear-solve A b)` solves `A x = b` for a square `N×N` tensor `A`
  and length-`N` `b` with a full-f64 accuracy guarantee. On Apple/Accelerate it
  factorizes `A` in fp32 (the `O(n³)` cost, ~2–3× faster than f64 on the AMX
  units) and polishes the result back to full f64 by iterative refinement
  (Langou/Baboulin/Dongarra): the residual is recomputed in f64 and a fp32
  back-solve corrects `x` until the relative backward residual is certified at
  ~`1e-13`. When refinement cannot certify a full-f64 result (ill-conditioned
  systems) it silently falls back to a plain-f64 LAPACK `dgesv`, so the caller
  always gets a correct f64 answer — the speedup is opportunistic, the
  correctness is not. Non-Apple builds use a direct f64 LU. Singular,
  non-square, and dimension-mismatch inputs raise catchable conditions.
  Measured on an M2 Ultra: ~1.15–1.4× faster than the forced-`dgesv` path at
  `N = 2048–4096` at `~1e-15` residual. Implemented in `lib/core/linear_solve.cpp`
  with native-codegen and bytecode-VM surfaces (native call id 472) and an
  `linear-solve-full-f64` ICC oracle; see
  `tests/features/linear_solve_test.esk` and
  [docs/reference/tensors/operations.md](docs/reference/tensors/operations.md#linear-solve--full-f64-solve-with-a-mixed-precision-fast-path).
- **Ozaki-II reduced-precision fast tier (Metal, opt-in) — beats AMX f64 at
  large N.** A fully-GPU reduced-precision DGEMM tier alongside the bit-exact CRT
  path, selected with `ESHKOL_SF64_KERNEL=ozaki-fast` (or `ESHKOL_OZAKI_FAST=1`).
  A linear CRT over **near-peak MPS f32 GEMMs**: the moduli are cap-limited
  pairwise-coprime prime powers chosen so that a single MPS f32 GEMM of centered
  residues is integer-exact (`K·(p/2)² < 2²⁴`), running each modulus at the GPU's
  ~20 TF f32 ceiling. The residue split (`ozaki_fast_split`, straight from the f64
  bit-pattern) and the df32 fractional-CRT reconstruction (`ozaki_fast_accum` +
  `ozaki_fast_finalize`) run entirely on the GPU; the host only uploads A,B once,
  does one O(N^2) exponent pass, and downloads C. All moduli run in one command
  buffer with a single CPU-GPU sync. The accuracy knob is the moduli count
  (`ESHKOL_OZAKI_FAST_MODULI`, default 11, clamped loudly to `[2,16]`); the
  default targets ~1e-8 (worst-case rel err ~2.4e-8 over the four correctness
  regimes), and df32 caps this tier at ~1e-11. **Measured on an M2 Ultra
  (best-of-5, internal pipeline): N=8192 = ~1384 GF at 11 moduli = 1.26x clean
  AMX `cblas_dgemm` (1099 GF), up to ~1448 GF (1.32x) at 10 moduli; ~7.7–9.7x
  faster than the exact 16-modulus tier.** N=4096 ties AMX (overhead-bound at
  smaller N). Requires fast-math OFF (already enforced) and `ldexp` not `exp2` —
  both annihilate/inject ~1e-7 errors otherwise. The **default DGEMM path is
  unchanged**; the tier is strictly opt-in and the existing exact
  `ozaki-ii-correctness` gate is untouched. Adds `tests/gpu/ozaki_fast_gate.sh` /
  `ozaki_fast_test.esk` (rel err <= 1e-7 across integer/fractional/pi-e/wide
  regimes at K up to 4096, asserting the fast path engages with no silent
  fallback) and the `ozaki-ii-fast` ICC oracle. `ESHKOL_OZAKI_PROFILE=1` reports
  per-matmul internal pipeline GFLOP/s.
  (`lib/backend/gpu/gpu_memory.mm`, `lib/backend/gpu/metal_softfloat.h`)
- **Reverse/forward-mode `gradient` on the bytecode VM — full AD parity for the
  gradient surface, self-differentiating on every substrate.** `gradient` now
  works on the VM exactly as on the native `-r`/AOT path: direct
  `(gradient f point)`, through a callable parameter `(define (w f p) (gradient
  f p))`, and curried `((gradient f) point)` / `((gradient f) x y)`. It resolves
  the callable's true arity and expands the point accordingly — a scalar point
  yields a scalar derivative, an N-argument scalar loss spreads into N seeded
  coordinates, and an arity-1 whole-vector loss (read via `vref`) is seeded as a
  single vector — matching the native codegen's arity handling. The callable's
  fixed arity is packed into the high bits of its func-PC constant at compile
  time and unpacked by `OP_CLOSURE` into the closure, so it survives ESKB
  serialization (whose per-function offsets do not). Results are byte-identical
  to native across native / vm-src / vm-eskb for the whole gradient corpus
  (`tests/vm_parity/corpus/32_gradient_reverse.esk`), including non-polynomial
  losses (`sin`/`cos`/`exp`/`log`/`/`) and repeated in-loop calls (no leak).
  `tests/autodiff/gradient_callable_arity_test.esk` is 25/25 on the VM.
  Transcendental unary ops (`sin`/`cos`/`exp`/`log`/`sqrt`) are now recorded on
  the one reverse-mode AD tape too. `op:GRADIENT` and `op:DERIVATIVE` flip to
  `vm-supported` in `tests/vm_parity/PARITY.tsv`; higher-order nesting (gradient
  of a derivative / Taylor tower) remains native-only. New ICC gate
  `vm_gradient_parity`.
- **Public low-level reverse-mode AD tape surface on the LLVM path (JIT + AOT).**
  `ad-pow`, `ad-gradient-of`, `ad-value-of` and `ad-tape-length` — previously
  `Unknown function` under JIT/AOT despite being VM builtins and documented —
  are now first-class codegen builtins wired through the same sret-runtime
  machinery as the rest of the `ad-*` tape ops (new `eshkol_ad_pow_sret` /
  `eshkol_ad_tape_length_sret` wrappers over the shared `vm_autodiff.c` tape).
  `(ad-pow tape base exponent)` gives ordinary `pow` forward semantics with
  correct reverse derivatives (e.g. `d/dx x^0.5` at 4 → `0.25`).
  `tests/vm/ad_tape_lowlevel_regression.esk` now passes on JIT, AOT and the VM
  (`build/eshkol-run --strict-types -r …` exits 0), and the low-level ad-pow
  case is in the parity corpus (`tests/vm_parity/corpus/33_ad_pow_lowlevel.esk`,
  byte-identical native/vm-src/vm-eskb).
- **Checked `(the <type> expr)` ascription and predicate-guarded narrowing
  (strict mode).** The type checker gained several idiom-accepting features so
  that idiomatic dynamic-but-validated code type-checks without escape hatches:
  a new `(the <type> expr)` form asserts `expr` has the given type as a *trusted*
  assertion to the checker (it narrows the checker's view but is a pure runtime
  no-op — the emitted IR is byte-identical to `expr` alone); predicate-guarded
  narrowing teaches the checker that a value tested by one of eight type
  predicates (`number?`, `integer?`, `string?`, `symbol?`, `pair?`, `null?`,
  `vector?`, `procedure?`) is that type inside the guarded branch, honored across
  `if` and `and`, and cancelled at a `set!` of the narrowed variable; sum-type
  annotations are honored on named-let parameters; and a numeric-tower join gives
  recursive accumulators their least-upper-bound numeric type instead of
  rejecting a widening. Nine new type-system tests including negative soundness
  cases. (`lib/types/`, `lib/frontend/parser.cpp`)
- **SDNC weight-matrix backward pass wired into the build with a gradient
  check (#335).** `lib/backend/qllm_backward.c` — the reverse-mode
  (training-mode) companion to the analytical forward constructor in
  `weight_matrices.c` — was previously source-only: no build target, every
  function `static`, zero test coverage. It is now compiled by the normal build
  (static library `eshkol-qllm-backward`, header
  `inc/eshkol/backend/qllm_backward.h`), with the two FFN backward passes
  (SQUARE-activation and gated-sigmoid) exposed as a public surface. The backward
  math is precision-generic (`qllm_real`, default `float` so the
  QLMW/`InterpreterWeights` layout is byte-identical). A gradient-check test
  (`tests/backend/qllm_backward_gradcheck_test.c`) validates the analytical
  gradients against a central finite-difference reference — recompiled in double
  so the finite-difference floor drops below the documented **1e-6**
  relative-error bar (achieved SQUARE `3.7e-9`, gated `2.3e-9`) — and is wired as
  the `qllm_backward_gradcheck` ICC smoke probe and a completion-oracle criterion.
- **`eshkol-qllm-run` tool (#335).** `lib/backend/qllm_interpreter.c` (previously
  unbuilt) is now a standalone executable that loads a QLMW v3 weight file and
  executes Eshkol bytecode through the six-layer transformer forward pass — the
  weights *are* the interpreter. The default build uses the portable C reference
  matmul; `-DUSE_QLLM` links the qLLM NEON/Metal backend.
- **Dynamic edge coverage for the v1.3.4 surface (#336).** A seeded, bounded,
  depth-parametric generator (`scripts/gen_edge_v134.py`) and runner
  (`scripts/run_edge_coverage_v134.sh`) reconcile the generative/adversarial
  machinery (P2 edge-matrix + P6 depth-parametric + differential oracle) with
  every new-surface family the v1.3.4 wave added: nursery iter-scope mutating
  loops (all six barrier channels, escape-set size, nested-loop depth),
  capturing `parallel-map`/`parallel-execute` returning collections (n at the
  pool threshold, closure shapes, nesting depth), exact gradient through a
  callable parameter + curried form (arity 1..5, list/vector points, composition
  depth), native `i128` boundaries and wraparound (differential native-vs-VM,
  arithmetic-chain depth), native tensor/`matmul` (arange arities, reshape/arange
  product, multi-dim ref/set), the shortest-round-trip `number->string` /
  `string->number` family, and low-level `ad-tape`/`ad-pow` on the VM
  (fractional/negative/zero exponents, tape reuse, 1024-node growth). Every probe
  is self-checking against a generator-computed ground truth and runs across
  JIT / AOT-O0 / AOT-O2 (and the VM where the surface exists). Gated in ICC by
  the `v1.3.4-edge-coverage` oracle (one criterion per family) plus a
  `v1.3-evolve` roll-up; new depth axes registered in
  `scripts/depth_coverage_registry.json`.
- **WASM execute-and-diff differential lane (#353).** A new assurance lane
  executes Eshkol-compiled WebAssembly and diffs its output against native,
  closing the gap where the web tests only checked that a produced `.wasm` was a
  valid binary. It builds the VM WASM module (the bytecode VM compiled via
  Emscripten — the same module family that powers the browser REPL) fresh from
  current source, runs each program of the VM-supported corpus under Node through
  a new append-only batch `run_program` export, and byte-diffs the captured
  stdout against native `eshkol-run -r` (reusing the VM-parity newline
  normalization, comparing float text raw). Divergences are documented per file
  in `tests/wasm_diff/EXCLUSIONS.tsv` — `EXCLUDED` for a surface that cannot run
  in the sandbox, `XFAIL` for a real reported WASM bug (the comparison still runs
  and an unexpected match fails the gate to force the row's removal) — with no
  silent skips, and a `kind:"wasm_parity"` JSON-L trace feeds ICC.
  (`scripts/run_wasm_differential.sh`, `scripts/lib/wasm_diff_runner.js`,
  `lib/backend/vm_wasm_repl.c`)
- **Execution-backed language-coverage gate (#352).** The language-surface
  coverage gate now certifies executed, verified behaviour — a construct counts
  only when it dispatched or executed in a passing run (or, for the bounded
  compile-time-form allowlist, was parsed and code-generated) — with lexical
  name-presence demoted to a diagnostic that earns no release credit. A permanent
  invariant tripwire keeps the credited set a subset of the runtime/compile-time
  evidence, and a monotonic deficit ledger refuses to record a larger deficit
  without an explicit override. The proven surface is re-baselined from 1,056 to
  1,078 (the `i128` tower, `linear-solve`, and string/pointer conversions landed
  as new core builtins) and the policy floor and ledger are ratcheted to 1,078
  (1,078/1,078 execution-backed).
- **Shortest-round-trip numeric printing (R7RS 6.2.6).** `display`, `write`, and
  `number->string` now emit the shortest decimal string that reads back as the
  identical `double`, replacing the previous fixed-precision output that could
  lose or add digits. `(sqrt 2.0)` prints `1.4142135623730951`; integral doubles
  keep their no-`.0` form and non-finite flonums keep the R7RS
  `+inf.0` / `-inf.0` / `+nan.0` external representations. Native codegen and the
  bytecode VM share one portable-C routine (`eshkol_dtoa_shortest`), so their
  output is byte-identical.
  (`lib/core/runtime_display_hosted.cpp`, `lib/backend/vm_native.c`,
  `lib/backend/vm_core.c`, `lib/backend/vm_string.c`)
- **Hosted-VM tensor matmul parity.** The bytecode VM now matches native codegen
  on the full tensor-matmul surface: `arange` in 1-, 2-, and 3-argument forms,
  nested-literal tensor operands, and multi-dimensional `tensor-ref` /
  `tensor-set!`. The parity corpus gains `31_tensor_matmul`, closing the last VM
  divergences on this surface. (`lib/backend/vm_native.c`, `lib/backend/vm_compiler.c`)
- **Moonlab quantum backend pinned to v1.2.0.** The `agent.quantum` integration
  now targets Moonlab v1.2.0, which adds `vqe_compute_qgt` (quantum geometric
  tensor / quantum natural gradient support) and a smooth first-principles
  H2/LiH potential-energy surface. The H2 equilibrium oracle at bond length
  0.735 Å is updated to `-1.142200155381` Ha (a `-2.95e-5` Ha shift from the
  earlier PES). Adds differentiable quantum-chemistry examples (five programs)
  and an arbitrary-order-AD H2 vibrational-frequency example.
  (`docs/design/MOONLAB_INTEGRATION.md`, `examples/`)

- **Portable event loop (ESH-0011).** A new event-loop primitive —
  `make-event-loop`, `event-loop-add-fd!`, `event-loop-remove-fd!`,
  `event-loop-poll`, `event-loop-close`, `event-loop-backend` — wrapping kqueue
  on macOS/BSD, epoll on Linux/Android, and IOCP (plus WSAPoll /
  `PeekNamedPipe` readiness) on Windows, with a fail-closed stub on
  WebAssembly. Exactly one platform backend is compiled in, selected in
  `CMakeLists.txt` the same way the GPU layer selects its `gpu_memory`
  implementation. The portable half — handle registry, argument validation,
  generation-tagged handles, result coalescing — lives in
  `lib/core/event_loop.c` and is shared by every backend *and* by the bytecode
  VM, so there is no native/VM parity surface to maintain separately.
  Verified: a pipe read/write round-trip completing inside its timeout, an idle
  poll that waits its budget and returns rather than hanging, and 1,000
  sequential open/close cycles that never exhaust the descriptor table (the
  macOS default limit is 256, so a leak would fail long before 1,000).
  (`lib/core/event_loop.c`, `lib/core/event_loop_*.c`)

- **Fixed-point / `i128` exact-accumulation engine.** A fully additive module:
  `esk_i128`, a parametric `fixed<W,F>` with explicit per-operation rounding,
  and a block-scaled `dot_exact` reduction over an i128 accumulator. Its
  purpose is order-independent bit-exact reduction — the same reduction over
  the same elements produces byte-identical results regardless of summation
  order, which an f64 accumulator cannot guarantee. Measured: 200 shuffles ×
  4,096 elements and 50 matmul contraction orders are byte-identical under the
  exact path, while an f64 control drifts about 1.5e-06 between orderings on
  the same inputs; and the exact path is *faster* than an f64 double-double
  baseline (13.6 GB/s vs 4.3 GB/s), so exactness is not a throughput trade-off
  here. Standalone suite 84/84. `ESHKOL_FIXED_POINT_ENGINE` defaults ON and is
  forced OFF on MSVC/ClangCL, which lack the GCC/Clang `__int128` ABI the
  engine relies on. The `eshkol-fixedpoint` shared library is a self-contained
  C11 module depending only on libc and libm — no LLVM, no `eshkol-static` — so
  it is consumable independently of the rest of the toolchain.
  (`lib/math/fixed_point/`, `tests/fixed_point/`)

- **The qLLM bridge is implemented, not just declared.**
  `inc/eshkol/bridge/qllm_bridge.h` declared 17 functions and
  `docs/api/bridge/qllm_bridge.md` reported all 20 of its symbols documented,
  but **none of the 17 was defined anywhere in the tree**. The consequence ran
  deeper than a missing entry point: the bridge's *backward* half already
  shipped — `lib/bridge/tensor_backward.cpp` implements exact gradient rules
  for 11 of 13 tensor AD node types and is compiled into the main library —
  and was unreachable, because nothing ever created a node of those types. All
  17 functions are now implemented in `lib/bridge/qllm_bridge.cpp`: each
  computes the forward value and, when a tape is supplied, records a node of
  the canonical `ad_node_type_t` with the exact input wiring the matching
  backward rule reads, so those gradient rules now run. Where a backward rule
  cannot differentiate a shape exactly, the forward **refuses** rather than
  recording a node whose gradient would be wrong (matmul is 2-D only; softmax
  takes only the axis its rule normalises over) — exact AD or an explicit
  error, never a silent zero. Lifecycle is a real `dlopen` resolving a known
  qLLM entry point, so `ready()` reports a fact. `ESHKOL_QLLM_ENABLED` is a new
  CMake option, OFF by default per `docs/SDNC.md` §13; ON without a
  discoverable library is a configure-time `FATAL_ERROR` naming
  `ESHKOL_QLLM_ROOT`, because an explicit opt-in that silently does nothing is
  worse than no option at all. A new `lib/backend/sdnc_isa.h` makes the SDNC
  ISA, state-vector layout and type tags a single source shared by producer and
  consumer, closing a live divergence: the producer emitted `OP_SWAP = 83`
  while the consumer's private copy stopped at `OP_COUNT = 83` and rejected
  that opcode outright — a drift across a file format, where a renumbering
  changes behaviour silently instead of failing to link. New ctest:
  `qllm_bridge_gradcheck`.

- **Embedding backward pass and Fréchet-mean derivative.** Two backward rules
  that previously refused outright. The embedding forward `y[i,:] =
  W[idx[i],:]` is a gather, so its adjoint is the scatter-add `dW[idx[i],:] +=
  dy[i,:]`; `tensor_embedding_backward` could not compute it because the lookup
  indices were not on the AD node at all. The node contract now carries them
  (`input1` = weights, `input2` = the index node, `params = [num_indices,
  d_model, vocab_size]`), and the two properties that carry correctness are
  asserted directly: duplicate indices **accumulate** (assigning instead is the
  classic scatter-add bug, and it under-counts every repeated token), and
  unselected rows are **bitwise** zero (the adjoint of a gather is genuinely
  sparse). The integer-valued index operand gets no gradient — left untouched
  rather than seeded with a zero that would read as "differentiated, came out
  zero" — and a missing, fractional, or out-of-range index raises instead of
  scattering into the wrong row. The weighted Fréchet (Karcher) mean on the
  Poincaré ball gets its derivative by implicit differentiation of the
  first-order optimality condition rather than by unrolling the solver. Fixed
  on the way: `eshkol_tensor_backward_dispatch` bracketed each node's backward
  in a scope on the **global** arena, and `arena_pop_scope` rewinds over
  everything allocated since the push — including the destination gradient
  buffers allocated lazily from that same arena on first accumulation, which
  the upstream node then read back as reclaimed memory. Temporaries now get
  their own arena: two lifetimes, two arenas, and no allocation site needs to
  know whether a backward scope is open above it.

- **CTest results are completion-oracle evidence.** No oracle criterion
  anywhere consumed a CTest result: the only `ctest` mentions in
  `.icc/completion-oracles.yaml` were `action:` strings, and the nine
  `test_evidence` criteria were index-level ("the tests exist and are
  runnable"), not execution-backed — so a red CTest run could not turn the
  release gate red. `scripts/run_ctest_gate.sh` now runs CTest, parses the
  per-test verdicts, and emits `kind: "ctest"` and `kind: "test_result"` trace
  events, with a roll-up per named group and one for the whole suite. **A group
  whose regex matches no configured test is reported ABSENT and fails the
  gate**, so a pillar cannot quietly stop being covered because its tests were
  renamed or configured out. Eight criteria are wired under
  `eshkol-compiler-readiness`, including the fixed-point engine, the
  exact-input AD identity tier, the runtime-closure arity spread, same-unit
  `define-library`, the self-checking VM surface suite, VM parity, and the
  event loop — several of which had shipped into this cut with a working CTest
  gate that the release target never looked at.

### Changed

- **Opt-in TensorCore adapter.** A TensorCore acceleration adapter is now
  available behind an opt-in switch, pinned to a tested `tensorcore` version
  range (`ESHKOL_TENSORCORE_MIN_VERSION` / `_MAX_TESTED_VERSION`); the default
  numeric path is unchanged.
- **Homebrew-compatible system-dependency builds (#344).** Every bundled
  agent-FFI dependency is now resolvable without a live `FetchContent` download
  so `brew install` works: an `ESHKOL_HOMEBREW_BUILD` umbrella option enables the
  packaging contract (system PCRE2 imported under the bundled target name, the
  remaining deps resolved via `FETCHCONTENT_SOURCE_DIR_<NAME>` source-dir
  overrides that skip the dependency provider entirely), while the default
  developer/release build stays byte-for-byte unchanged. The same rewrite fixes
  pre-existing packaging bugs that left the keg non-functional (installs
  `libeshkol-runtime.a`, the agent-FFI archives, and module sources under
  `share/eshkol/lib`), and the release auto-bump substitutions are anchored to
  the top-level indent so the new `resource` sha256 pins survive a release bump.
- **Dead-code sweep.** Removed 56 ICC-confirmed orphan symbols superseded by the
  BindingCodegen and AD-matrix refactors; no behavioral change.
- **Build and platform.** A universal Linux build script provisions LLVM 21
  per distro family; CMake `< 3.24` no longer errors on
  `DOWNLOAD_EXTRACT_TIMESTAMP`; the WASM `scheme_main` re-entry export is
  JS-safe with a null-guarded data layout; documentation-site section anchor
  links resolve (heading ids + fragment scroll). The language-surface coverage
  manifest was regenerated for the new VM tensor special forms (builtin count
  unchanged). A CI identity guard rejects new commits that carry a forbidden
  private author email.
- **`Rational` in the compile-time numeric tower.** Eshkol has had R7RS exact
  rationals at runtime (`HEAP_SUBTYPE_RATIONAL`, `rational?`) with no
  corresponding type, so `rational` was not a spellable type name. It is now a
  registered exact type under `Number`, a sibling of `Integer`/`Real` — matching
  how `Complex` already sits beside `Real` in this graph rather than above it.

### Fixed

- **Every documented resource-limit environment variable was accepted and then
  enforced by nothing.** `ESHKOL_MAX_HEAP`, `ESHKOL_TIMEOUT_MS`,
  `ESHKOL_MAX_STACK`, `ESHKOL_MAX_TENSOR_ELEMS`, `ESHKOL_MAX_STRING_LEN`,
  `ESHKOL_ENFORCE_LIMITS` and `ESHKOL_LIMIT_WARNINGS` were parsed into the
  active configuration by `eshkol_init_limits_from_env()`, and the functions
  that check them — `eshkol_track_allocation`, `eshkol_check_string_length`,
  `eshkol_check_tensor_size`, `eshkol_is_timed_out`, `eshkol_stack_push` — were
  written, compiled and never called from anywhere outside their own unit
  tests. `ESHKOL_MAX_HEAP=1` ran a 20-million-iteration allocating loop to
  completion and exited 0; `ESHKOL_TIMEOUT_MS=500` printed `ERROR: Execution
  timeout: 500ms limit exceeded` and then *also* ran to completion and exited
  0, because the watchdog thread could only request an interrupt and nothing
  polled for one. A consumer who set a heap ceiling to contain untrusted code
  got no containment and no warning.

  Each ceiling is now applied at the one place every path to it converges:
  `ESHKOL_MAX_HEAP` in `create_arena_block()`, the arena's sole OS-request
  site, so the check is amortized over a whole block and the bump-pointer fast
  path is untouched; `ESHKOL_MAX_STRING_LEN` in
  `arena_allocate_string_with_header()`, which also closes a silent truncation
  of the `uint32_t` header size past 4 GiB; `ESHKOL_MAX_TENSOR_ELEMS` in
  `arena_allocate_tensor_full()` and, because native codegen assembles tensors
  from three separate allocations rather than calling it, at the point codegen
  computes the element count; `ESHKOL_TIMEOUT_MS` as a cooperative poll emitted
  on every tail-call loop back-edge of hosted native codegen (the profiles that
  link no hosted watchdog — freestanding objects and `--wasm` modules — have
  nothing that could request an interrupt, so the poll is not emitted there)
  and run at every guarded function entry.
  `ESHKOL_MAX_STACK` now drives the recursion-depth guard that codegen already
  emitted, which had been comparing against a hard-coded 100000 with no
  connection to the configurable limit the variable fed — two mechanisms with
  the same default and no wire between them, now one.

  Each ceiling is **opt-in**: it binds a run only when that run sets its
  variable (or sets the matching `ESHKOL_LIMIT_ACTIVE_*` bit before
  `eshkol_set_limits()`). The documented defaults are the values a limit takes
  when you turn it on, not ceilings every program is silently held to —
  `tests/features/blc_test.esk` allocates past the 1 GiB heap default, and the
  VM's computed-goto dispatch never had an instruction guard at all, so
  applying every default to every run would impose new ceilings rather than
  enforce documented ones. Whether the defaults should also bind an
  unconfigured run is a release decision about defaults, recorded in
  `docs/reference/runtime/environment-variables.md`.

  A violation is loud and terminal by default: pending output is flushed, one
  `eshkol: fatal: …` line names the limit, the ceiling and the variable that
  set it, and the process exits with a status specific to that limit — 120
  heap, 121 stack, 122 tensor elements, 123 string length, 124 execution
  timeout (matching GNU coreutils `timeout(1)` and this project's existing
  `run-command` convention), 125 VM instructions. `ESHKOL_ENFORCE_LIMITS=false`
  makes them advisory: the breach is recorded, warned about, and the program
  runs on. Staying under a ceiling costs nothing measurable and changes no
  computed value — no check reads or writes program data.

- **The bytecode VM's documented runaway-instruction guard did not exist on the
  path that actually runs.** `ESHKOL_VM_MAX_INSN` is documented with a default
  of 10,000,000, but `vm_run()`'s computed-goto dispatch — the path every
  GCC/Clang build takes — had no instruction counter at all, while the MSVC
  `switch` fallback capped at a hard-coded 10,000,000 with no environment
  override. One interpreter, two dispatch implementations, two different
  answers to "has this program run away". Both now share one counter and one
  configurable ceiling, checked once per 4096 instructions so the per-opcode
  cost is a single decrement and branch. The variable is parsed alongside its
  six siblings in `eshkol_init_limits_from_env()` and reaches the VM through
  `eshkol_get_limits()`, because the VM's sources are freestanding-safe and may
  not read the environment themselves.

- **`(the <type> expr)` never rejected an ascription, so a false one was
  invisible.** `(display (the string (+ 1 2)))` printed `3` and exited 0. The
  checker synthesized the wrapped expression, discarded the type it derived,
  and adopted the ascribed type unconditionally — so an ascription no value can
  satisfy was indistinguishable from one every value satisfies, and "checked
  ascription" checked nothing. The two types are now compared, and a *provable*
  contradiction is reported through the same enforcement point as every other
  type issue: a warning under gradual typing, fatal under `--strict-types`.

  The form's contract is otherwise unchanged, and deliberately so. It remains a
  trusted narrowing boundary — narrowing from a dynamic value is the reason it
  exists and is never questioned — and widening, unresolved types, and
  ascriptions that move around the numeric tower (whose members are siblings
  under `number` rather than a chain, so `(the real 1)` has no subtyping
  relation in either direction) are all still accepted. The diagnostic is
  compile-time only: the emitted IR is still byte-identical to the wrapped
  expression, there is still no runtime tag check and no cost, and a VM program
  that omits the ascription still computes the identical result, so
  `COMPLETE_LANGUAGE_SPECIFICATION.md` § 3.6.6's guarantee and the
  `native-only-justified` VM-parity row both continue to hold.

- **A relative `(load "sib.esk")` resolved to a different file depending on
  which execution engine ran the program.** `eshkol-run -r prog.esk` uses the
  persistent JIT run cache for a single input with no `-d`/dump flags and no
  `$ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR`, and the in-process LLVM JIT otherwise.
  Each engine carried its own copy of the module search order, and the copies
  disagreed about the first tier: the AOT lane (and therefore the run cache)
  looked beside the **source file**, the in-process JIT looked in the process's
  **working directory**. The same command on the same file therefore printed
  one answer with the cache warm and `Module not found:` — or, with a
  same-named file in the working directory, a *silently different* answer —
  with it bypassed. A third copy in the JIT's path-form `(import "…")` handler
  searched cwd-then-`lib/` only, and `process_imports` in the AOT lane searched
  the source directory only. Resolution now happens once, in
  `eshkol::platform::resolve_module_source_path()`, which every lane calls and
  none may shadow; the requiring file is established by the same scope that
  attributes source text, so nested loads root at their own file rather than at
  the outermost one. The order is the documented one — requiring file's
  directory, working directory, `$ESHKOL_PATH`/`-I`, install, build-tree
  fallback (`docs/reference/language/modules.md`) — so both spellings the test
  corpus uses (sibling-relative and project-root-relative) keep resolving.
  Path literals written without `.esk` are now probed with and without the
  extension in every tier, not only against the working directory. New CTest
  gate `load_path_engine_parity_test` runs one program through all four lanes
  and demands byte-identical output, with a same-named decoy planted in the
  working directory so a cwd-rooted regression fails on a different answer
  rather than on a missing file. Found because `run_language_coverage.sh` sets
  the coverage trace dir, which bypasses the cache: the entire qllm oracle
  suite — whose exporters `(load "qllm_oracle_lib.esk")` from a sibling —
  aborted under it while passing under every other harness.

- **`(tensor (list (list …) (list …)))` built a rank-1 tensor of zeros, and the
  rank-2 read that followed segfaulted.** `(tensor X)` on a single collection
  argument walked exactly ONE level of `X`, coercing every element with
  "heap pointer -> 0.0", so a nest of lists silently lost its shape and
  displayed as `#(0 0)`; `(tensor #(#(1 2) #(3 4)))` looked correct only because
  the *parser* flattens nested `#(...)` literals at compile time. A rectangular
  nest of lists and/or vectors — in any combination, to any rank up to 8, with
  nested tensors as elements — now builds the N-dimensional tensor its shape
  describes, matching the "classify by runtime value, not construction form"
  principle. A ragged or otherwise non-rectangular nest raises a clean catchable
  error instead of fabricating a wrong shape.
- **Every rank and bounds guard in `tensor-get`/`tensor-ref` and `tensor-set!`
  had been silently deleted by the optimizer.** Those guards emitted their
  diagnostic only `if (printf && exit)` resolved through the codegen function
  table, which never registers either symbol — so each failure block compiled to
  a bare `unreachable`, from which LLVM infers the guarded condition is
  impossible and removes the branch. A multi-index read of a lower-rank tensor
  therefore ran off the end of the dimensions array and underflowed the slice
  rank (SIGSEGV on both JIT and AOT), and an out-of-range `tensor-set!` wrote
  outside the element buffer. All four guards now raise a catchable error
  through the runtime, and the list-index idiom `(tensor-ref t (list i j …))`
  gained the runtime rank check its multi-argument sibling always had.
- **`(number->string -0.0)` produced `"-0"`, which reads back as the exact
  integer `0`.** `"-0"` is not a flonum external representation, so the reader
  took it as an exact zero — which has no sign — and both the inexactness and
  the sign bit were lost: `(/ 1.0 -0.0)` is `-inf.0` but
  `(/ 1.0 (string->number (number->string -0.0)))` came back `+inf.0`. That is a
  loss of the VALUE, not merely of its exactness, so negative zero now renders
  `"-0.0"`, which reads back as an inexact negative zero (R7RS 6.2.6
  round-trip). Positive zero and the other integral-valued doubles keep the
  established no-`.0` form (`0`, `3`, `1234567`), whose read-back recovers the
  same numeric value. The shared formatter backs native and the bytecode VM, so
  all three substrates emit the same bytes.
- **`--strict-types` did not make type errors fatal, contradicting `--help`
  ("Type errors are fatal") and `docs/reference/runtime/eshkol-run.md`.** The
  flag only changed the *wording* of the diagnostic — `[ERROR] Type error:`
  instead of `[WARN] Type warning:` — after which code generation ran to
  completion, the compile exited **0**, and the AOT path wrote a finished
  binary for the program the type checker had just rejected. Every reject
  fixture in `tests/typesystem/` demonstrated it: the error was printed and the
  build succeeded anyway, so any build step trusting `$?` (or the existence of
  the output file) certified ill-typed code. Under `--strict-types`, accumulated
  type errors now abort compilation at the end of the type-checking phase: the
  compile exits nonzero and **no** binary is produced. Gradual mode (the
  default) is byte-for-byte unchanged — it warns and continues — and `--unsafe`
  still reports nothing.
- **A checked cast silently swallowed errors inside the expression it wrapped.**
  `(the <type> <expr>)` synthesized `<expr>` and then *discarded* the result, so
  a failing inner synthesis (an unbound variable, say) produced no diagnostic at
  all and was not counted — the ascription hid exactly the nested error its
  documentation promises not to hide, and `--strict-types` had nothing to be
  fatal about. Nested failures now go through the unified enforcement point
  while the ascribed type still flows onward: `the` trusts the *type*, not the
  expression.
- **`(the <type> expr)` rejected the most natural bare type names.** `(the
  number 3)` was a parse error — "Unknown function: the" followed by "Undefined
  variable: number" — because the form was recognised via a hand-maintained
  allow-list inside the parser that omitted `number`, `pair`, `vector`,
  `procedure`, `list`, `tensor`, `complex` and `rational`, so the ascription
  degraded into a call to an undefined procedure named `the`. `number` is one of
  the eight documented narrowing predicates, and
  `docs/COMPLETE_LANGUAGE_SPECIFICATION.md` used `(the number (car
  mixed-list))` as its own example. The parser's private list is gone: bare type
  names now come from one canonical registry in the type system
  (`eshkol::hott::builtinTypeSpellings()`), which also populates the type
  environment's name table — so a spelling the parser accepts is always a
  spelling the checker can resolve, and the two cannot drift apart again. Bare
  container/constructor names (`pair`, `vector`, `list`, `tensor`, `complex`,
  `procedure`, `closure`, `hash-table`, `qubit`, the sized integer and
  autodiff spellings) resolve to their real types instead of silently widening
  to `any`, `number` resolves through the previously-unwired
  `HOTT_TYPE_NUMBER` kind, and calling through an ascribed callable
  (`((the procedure f) x)`) no longer hits codegen's "Call expression requires
  variable or inline lambda" bail-out.
- **ESH-0362: an arity error is now FATAL, on every execution path — no more
  poisoned handles.** Calling a fixed-arity function with the wrong number of
  arguments printed a named diagnostic and then *kept going*. Three distinct
  fail-open cells, all closed at the root:
  - **Too few arguments** (the reported case). The closure-call arity check
    emitted `Arity mismatch: f expects 2 arguments but got 1` and returned
    `nullptr` without marking the compilation fatal. A `nullptr` from codegen is
    indistinguishable from "this form produced no value", so the enclosing
    `(define h (f a))` bound `h` to **null** and compilation continued. Under
    `-r` the program ran with the poisoned binding — `(process-pid h)` answered
    `0` and the next consumer dereferenced NULL (`SIGSEGV at 0x0`, far from the
    real mistake); under `-o` the driver wrote a complete binary and **exited
    0**, shipping the poisoned program.
  - **Too many arguments.** The argument loop simply never pushed a surplus
    argument, so the call was emitted at the callee's parameter count and the
    extra arguments *vanished*: `(add2 1 2 99)` ran as `(add2 1 2)` and printed
    `3`. The only trace was a gradual-typing `Type warning: function 'add2'
    expects 2 arguments, got 3`, which by design never fails a build.
  - **`-r` / REPL slot calls.** The two `__repl_fwd_<name>` indirect-call paths
    synthesise the callee's signature from the *call's* argument count, so a
    wrong count was not even a mismatch — it was a silent ABI disagreement, and
    the callee read its missing parameter out of whatever the register happened
    to hold. This is the path a `(require …)`d module's functions are called
    through, so the same file that named the error under `-o` reported nothing
    under `-r`. Both paths now consult the registered arity, and abstain only
    when it cannot be established (a genuine forward reference, a variadic
    callee, or a closure whose signature carries capture slots).

  All three now fail the compilation: `-r` exits nonzero without running, `-o`
  writes no binary, and the named diagnostic text is preserved verbatim — it is
  a consumer-facing contract. A related silent rebind is now surfaced too: when
  an `extern` reuses a C symbol already declared in the module with a *different
  parameter count* (e.g. `(extern void h :real strlen)` against the runtime's own
  1-parameter `strlen`), the declaration site warns instead of letting the call
  site paper the difference over with a null argument.

- **ESH-0363: FFI pointer arguments are type-checked at the boundary — an
  integer is no longer dereferenced as an address.** An `extern` parameter
  declared `ptr` / `string` / `char*` was passed to C by unconditionally
  reinterpreting the tagged value's 64-bit payload as a pointer. A number in a
  pointer position therefore became a pointer *equal to that number*:
  `(run-argv-capture argv 5000)` — the timeout supplied where the positional
  `cwd` belongs — reached the `execvp` shim as `const char* 0x1388` and died with
  `SIGSEGV at address 0x1388`, with no diagnostic and no exit code. Codegen now
  emits a check ahead of that conversion for every pointer-declared `extern`
  parameter and raises a **catchable** `ESHKOL_EXCEPTION_TYPE_ERROR` naming the
  extern, the argument position, the declared type and the offending value:
  `FFI type error in process-spawn-argv-flags-raw (C symbol
  qllm_process_spawn_argv_flags): argument 2 is declared 'ptr' and requires a
  string or pointer handle, but got the integer 5000`. Applied across the whole
  `extern` surface — **216 pointer-typed parameters over 323 declarations**, of
  which 115 across 80 declarations are the `agent.*` FFI surface — not just the
  reported call. A statically-decidable case (a numeric *literal* in a pointer
  position) is reported at compile time instead of as an LLVM verifier message.
  The predicate is a denylist of the immediate tags that cannot denote memory
  (numbers, characters, symbols, `#t`, dual/complex numbers, logic variables),
  never an allowlist of pointer tags, so a legitimate handle, port, bytevector,
  callable, `'()` or `#f` (Eshkol's spelling of a NULL pointer argument) can
  never be rejected. `--freestanding` and `wasm32` targets are excluded: neither
  has the hosted error runtime to raise into.
  `lib/agent/subprocess.esk` additionally validates `cwd` and `timeout-ms` by
  name, so the reported error identifies the parameter the caller actually got
  wrong rather than the internal `-raw` extern, and the spawn-family docstrings
  plus `docs/reference/agent/ffi.md` now state exact arities and positional
  meanings (`cwd` is the REQUIRED SECOND positional of every `process-spawn*`,
  and the THIRD positional of the `run-*` wrappers is the timeout).

- **ESH-0364: a diagnostic for code from a `(require …)`d module named the wrong
  file.** The AOT driver inlines every required module's forms into ONE flat AST
  array and compiles them as a single unit under a single ambient source context
  — the entry file. Any diagnostic for a form that came from a module therefore
  printed the ENTRY file's NAME beside the MODULE's LINE number. For a 3-line
  entry file requiring a module, the reported location was `entry.esk:6:13` — a
  line that file does not have; where the entry file was longer, it pointed at
  real but unrelated source, which is worse than reporting no location at all.
  (The JIT path was already correct: `executeBatch` takes explicit per-module
  provenance, which is why the same mistake was named accurately under `-r` and
  misattributed under `-o`.) Root cause: `eshkol_ast_t` carried `line`/`column`
  but no FILE, so a location was only meaningful relative to whatever context
  happened to be ambient. AST nodes now carry `source_file_id`, an id into a
  process-lifetime interned table, stamped on every top-level form at the single
  choke point every form passes through (a form cannot span two files, so inner
  nodes inherit their enclosing form's file). Codegen adopts that file for the
  duration of the form's codegen, so all of `generateLLVMIR`'s separate top-level
  walks — externs, function defines, global defines, `createMainWrapper`,
  `createLibraryInitFunction` — are covered by one scope, along with any future
  one. The module's source TEXT is resolved with its name, so the caret block
  renders the offending line from the right file; a file that cannot be read
  degrades to `file:line:col:` without the excerpt rather than to a wrong
  excerpt. Deliberately an id rather than a `const char*`: AST nodes are built in
  many places with no central zero-init, so an unset field holds garbage — a
  garbage id falls outside the table and reads as "unknown", where a garbage
  pointer would be dereferenced by the diagnostic printer.
- **ESH-0365: `(import (lib name))` reported the position of its CLOSING PAREN,
  and the language-coverage gate was certifying `import` on an accident.** The
  R7RS import form is lowered to a `require`, and the lowered node took its
  source position from the token the spec loop had just consumed — which is the
  form's closing `)`, not the `(import` that begins it. So a diagnostic about a
  malformed import pointed its caret at the closing paren. It also meant `import`
  had no execution-backed coverage evidence of its own: the tracker credits a
  compile-time form only when a parser-dispatch event and an accept/codegen event
  share an exact source position, and the dispatch event is recorded at the
  operator token while the accept event took its position from the closing paren.
  `import` was nonetheless certified covered at 1078/1078 — by a **cross-file
  collision**. Before ESH-0364, a required module's codegen events were attributed
  to the REQUIRING file, and in `tests/modules/r7rs_import_modifiers_test.esk` the
  imported module's `define`s at lines 5-7 column 2 landed on exactly the same
  positions as that file's own `(import …)` forms at lines 5-7 column 2. The
  position-only credit rule ignores the operation kind, so an unrelated `define`
  in another file was granting `import` its coverage. Fixing the attribution
  removed the collision and exposed the wrong position underneath it. The lowered
  node now carries the `(import` position, which both puts the caret on the
  construct and earns `import` genuine same-file evidence in all six import tests
  — coverage stays 1078/1078 on real evidence rather than a coincidence.
- **`iota` silently ignored its `start` and `step` arguments.** The stdlib
  defined `iota` as a strictly 1-argument function while callers (including
  `tests/features/new_functions_test.esk`, comments and all) were already writing
  `(iota 5 1)` and `(iota 5 0 2)`. Because codegen discarded arguments past the
  callee's parameter count (the ESH-0362 fail-open above), both returned
  `(0 1 2 3 4)` — the wrong list, with no error and no build-failing warning, and
  a test that "passed" while asserting nothing about start or step. `iota` now
  takes the optional positional `start` and `step` of SRFI-1 / R7RS-large
  (`(iota 5 1)` → `(1 2 3 4 5)`, `(iota 5 0 2)` → `(0 2 4 6 8)`), delegating to
  the existing `iota-from` / `iota-step`. Found by making arity errors fatal,
  which turned the silent wrong answer into a compile error.
- **The type-system negative suite aborted at its first fixture.**
  `scripts/run_typesystem_tests.sh` runs under `set -e` but invoked the compiler
  unguarded — and it is a *negative* suite, where a nonzero compile exit is the
  expected outcome. This never fired only because the faults its fixtures inject
  used to compile successfully anyway (`arity_mismatch_test.esk` reported its
  error and exited 0). With arity errors now fatal, the first fixture killed the
  harness mid-file, with no PASS/FAIL line and no summary — a suite that reports
  nothing looks nothing like a suite that fails. The compile invocation no longer
  trips `set -e`; verdicts come from the EXPECT-STDERR patterns, never the exit
  status. All 20 fixtures now run (20/20).

- **Higher-order derivatives through a variable-bound derivative closure were
  silently wrong.** `(define df (derivative f))` followed by `(derivative df)`
  returned `0.0` — for `f = x⁴` at `x=2` the second derivative is `48` — and the
  unnamed spellings `(derivative (derivative f))` / `(derivative (car fs))`
  printed `Failed to resolve function for higher-order derivative` at compile
  time while still emitting a binary. The second-derivative *mathematics* was
  never at fault: `(derivative-n f 2.0 2)` and
  `(derivative (lambda (x) (derivative f x)) 2.0)` both already returned the
  exact `48`. Two independent causes, both instances of classifying by the wrong
  thing. **(1) Resolution.** `derivativeHigherOrder`'s runtime path first
  required the differentiand to be a bare `ESHKOL_VAR`, then matched the bound
  `llvm::Value` against a whitelist of subclasses (`Argument` / `AllocaInst` /
  `LoadInst` / `GlobalVariable`). The `ESHKOL_VAR` gate rejects every *unnamed*
  differentiand outright, so no whitelist entry could ever have fixed
  `(derivative (derivative f))` — there is no binding to inspect — and the
  whitelist itself encodes where the compiler happened to put a binding, which is
  a storage decision, not a property of the language. Resolution now goes through
  the ordinary expression codegen (the language's single authority on what an
  expression denotes) and coerces the result to a tagged value, so named and
  unnamed, local and global, parameter and computed differentiands all resolve
  through one path. **(2) Nesting.** The emitted derivative closure unpacked its
  argument to a raw double and seeded a fixed single-level dual `{x,1,0,0}`; that
  unpack discards any perturbation the incoming point already carries, so the
  moment the closure was itself differentiated the outer tangent was destroyed.
  Both the static and the runtime-closure wrapper now seed *this* perturbation
  level (`seedForwardAndPush`) and extract *this* level's coefficient
  (`popAndExtractForward`) — the same shared runtime-level machinery
  `(derivative f x)` uses — which makes the returned closure **dual-transparent**:
  it differentiates like any other function. For `f = x⁴`, `derivative-n`, the
  nested-lambda form, the curried named form and the curried unnamed form now all
  return `32/48/48` at `x=2` and `108/108/72` at `x=3` on the JIT and the AOT
  lane, with no compile-time diagnostic. `gradient`'s identical resolution block
  is routed through the same helper. Regression:
  `tests/ad/curried_higher_order_derivative_test.esk` (46 cells, both lanes).
- **Curried `(derivative f)` did not exist on the bytecode VM, and nested VM
  derivatives silently returned 0.** `derivative` reaches the VM as a native call
  that pops exactly `(f, x)`, so the curried form popped whatever was below `f`
  on the operand stack and bound a non-callable — applying it failed with
  `calling non-function`. The curry is now lowered the same way `gradient`
  already lowers its own, to `(lambda (__dx__) (derivative f __dx__))`, so both
  spellings reach the same native call with the same `(f, x)` and agree exactly
  with native (`32` at `x=2`, `108` at `x=3` for `x⁴`). Separately, the VM's
  forward-mode carrier is a flat dual `{value, tangent}` with a single
  perturbation, so a point that was *already* a dual got flattened, the inner
  pass returned a plain float, and the outer pass read "non-dual result =
  constant function" and pushed `0.0` — a silently wrong second derivative. The
  VM now **raises a catchable error** naming the limitation instead. Higher-order
  AD on the VM needs the native jet's `e1`/`e2`/`ep` slots or a VM Taylor tower
  and stays native-only, recorded on the `op:DERIVATIVE` row of
  `tests/vm_parity/PARITY.tsv`.
- **`with-region` was mis-lowered on the bytecode VM, two independent ways, and
  returned an untagged value on native.** The VM compiled `(with-region spec
  body ...)` as a bare expression sequence. It emitted no `OP_POP` for non-final
  body expressions, so every multi-expression body stranded one value per
  non-final expression on the operand stack — and because top-level bindings are
  stack slots handed out by counting, the strands shifted every later `define`
  onto an occupied slot: after a three-expression region body, `(+ 111 222)` read
  back **0** instead of 333, and after `(with-region 'scratch …)` a later
  `(display x)` printed the region NAME instead of `x`. It also compiled the
  region SPECIFIER as an expression, so the documented `(with-region ('name
  size) body ...)` spelling became a call of `name` with argument `size` and died
  with `ERROR: calling non-function` — a documented spelling that could not run
  on the VM at all. Both are fixed by recognising the specifier (all three
  documented spellings) and lowering the body exactly like `begin`. On native,
  `codegenWithRegion` stored the body result into the tagged-value slot it hands
  to `eshkol_region_unwind_to()` WITHOUT packing it, so a primitive-literal
  result carried an uninitialised type tag: `(with-region 41)` displayed
  `#<unknown>` and `(with-region 4.5)` displayed `()`. That tag is what the
  unwind path dispatches promotion on, so it was a latent memory-safety hazard
  as well as a wrong value. All three axes (native, `vm-src`, `vm-eskb`) now
  agree and are gated permanently by
  `tests/vm_parity/corpus/with_region_lowering.esk`. One undocumented spelling
  does not agree and is filed rather than fixed: `(with-region (quote name))`
  with no other body, where the VM reader's collapse of `'name` and
  `(quote name)` into one node makes the sole argument look like a specifier
  (`tests/vm_parity/found/with_region_explicit_quote_body_vm.esk`). VM-side
  *reclamation* remains absent and is documented as such: the VM heap still has
  no escape evacuator, the same boundary a VM `region-close` declares.

- **`scripts/check_wasm_imports.py` could report present WASM stubs as
  missing.** The `env: { … }` key scanner matched braces and quotes with a
  character scan that did not know where comments were, so an apostrophe inside
  a `//` comment (`WASM can't longjmp out of host frames`) opened a phantom
  string literal; the next real quote closed it, and any brace in between was
  read in the wrong context. One unmatched brace ends the block early, after
  which every remaining stub is reported MISSING though present — a gate that
  names a symbol sitting right there in the file. Regex literals containing
  backticks or braces, and strings containing `//`, desynchronised it the same
  way, and quoted keys were never recognised. The scanner now tokenizes the glue
  (comments, all three string flavours, `${…}` substitutions, regex literals)
  and brace-matches over the token stream. A 13-fixture self-test pins every one
  of those constructs and runs before the tool reports any verdict, so a broken
  scanner fails loudly instead of returning a phantom red or a silent green;
  both JS glue files are now mandatory rather than skipped-with-a-warning, since
  a stub present in only one still breaks the other.

- **A named-let loop procedure used as a first-class value SIGSEGVed.** R7RS
  4.2.4 defines `(let loop ((v init) …) body)` as a letrec binding, so `loop` is
  an ordinary value that may be stored, returned or passed on, and it keeps the
  bindings its body closed over. But a named-let loop function's signature
  carries one capture *pointer* per captured free variable (the per-call capture
  design that makes concurrent invocations race-free) and is deliberately not the
  closure ABI, so a first-class reference used to fall through to the generic
  function-table path and yield the bare function pointer. Calling the leaked
  procedure then entered the loop function with its capture parameters filled
  from whatever the closure dispatcher happened to pass, and the first read of a
  captured variable dereferenced that as a capture cell — a wild-pointer crash on
  both JIT and AOT, not a wrong answer. A first-class reference now materialises a
  real closure whose environment holds the address of each shared capture cell,
  with a lazily emitted per-loop trampoline forwarding those cells as the pointer
  arguments the loop function expects. The cells are the arena storage a
  set!-mutated capture already gets, so a leaked procedure keeps reading and
  writing the same storage the loop wrote through and it outlives the enclosing
  frame. Regression tests: `tests/features/namedlet_escaped_closure_test.esk`
  (15 checks over six escape shapes, green under `-r` and AOT) and
  `tests/vm_parity/corpus/47_namedlet_escaped_closure.esk`, which gates the two
  shapes the VM can express across native, `vm-src` and `vm-eskb`.
- **A nested collection meant different things depending on how it was
  written.** A rectangular nested vector *literal* is flattened into a
  higher-rank tensor literal by the parser, so it never reached the shared
  tensor-operand check as a nest; the identical value built at run time did, and
  was rejected. `(tensor-shape #(#(1.0 2.0) #(3.0 4.0)))` was `(2 2)` while
  `(tensor-shape (vector (vector 1.0 2.0) (vector 3.0 4.0)))` was a type error —
  one operation, two answers, decided by construction form rather than by value.
  The operand check now classifies a nest by value and routes it to the same
  rank-N walker `(tensor X)` uses, which makes every tensor operation accept a
  runtime-built nest of lists and/or vectors at any rank and extent. Nothing
  about what either value *is* changed: `(vector (vector 1 2) (vector 3 4))`
  remains an R7RS vector of two vectors and `#(#(1 2) #(3 4))` remains Eshkol's
  rank-2 tensor literal; only the tensor-coercion question now has one answer.
- **A ragged nested vector literal could not be written at all, though the
  identical runtime-built value works.** The parser reported a *parse*
  diagnostic for a nest whose sub-shapes disagree and returned an invalid node.
  Once compile diagnostics became fatal, that diagnostic refuses the whole
  translation unit — `(define v #(#(1.0 2.0) #(3.0)))` is a hard compile failure
  even where the program never asks for a tensor — while
  `(vector (vector 1.0 2.0) (vector 3.0))` compiles and runs. (Before
  diagnostics were fatal the same invalid node was worse rather than better:
  `(tensor-shape #(#(1.0 2.0) #(3.0)))` silently answered `()`,
  `(tensor-shape #(#(1.0 2.0) 3.0))` answered `(2)`, and binding the literal and
  then reading it dereferenced the hole the invalid node left behind.) A
  non-rectangular nest cannot be a tensor, but it is a perfectly ordinary nested
  vector, so the parser now lowers one to `(vector <sub-literal> …)` and lets the
  value-based walker rule on it: a catchable error naming the mismatch, raised at
  the operation that demanded a tensor, and the literal remains a nameable nested
  vector in the meantime. That leaves exactly one raggedness check in the
  language rather than one per spelling. Gated by
  `tests/vm_parity/corpus/46_tensor_literal_spellings.esk` and
  `tests/features/tensor_nested_literal_spellings_test.esk`.
- **A compiler could link a stale *system* runtime archive in preference to its
  own, and `ESHKOL_LIB_DIR` could not override it.** `find_runtime_library()`
  searched name-major: every location for `libeshkol-runtime.a` — including
  `/usr/local/lib`, `/usr/lib` and `/opt/homebrew/lib` — was exhausted before
  `libeshkol-static.a` was tried *anywhere*, including in the directory holding
  the running `eshkol-run`. An install that ships only the legacy aggregate
  archive name therefore linked whatever `libeshkol-runtime.a` an older Eshkol
  had left in a system prefix, and `ESHKOL_LIB_DIR` did not rescue it because
  the env directory was consulted once per *name* rather than ahead of every
  location. When the stale archive's symbol set still matches, the link
  succeeds and the program silently mixes runtime versions — a wrong-runtime
  class defect, not a build failure. All install-artifact resolution (the
  runtime archive, the agent-FFI archives beside it, `stdlib.o`, `stdlib.bc`,
  and the `lib/**.esk` module tree) now runs through one shared,
  **location-major** root list in `lib/core/platform_runtime.cpp`:
  `$ESHKOL_LIB_DIR` first and absolutely, then `-L`/`-I` directories, then the
  install the compiler belongs to — resolved from the executable's **real**
  path, so a `bin/eshkol-run` symlink into a Homebrew Cellar keg resolves
  inside the keg — then the working directory's build trees, then the system
  prefixes; and within one directory the split archive is preferred over the
  legacy aggregate, so a co-located archive can never lose to a system one. The
  driver and `llvm_codegen.cpp`'s AOT link path shared none of this logic
  before and could disagree about which archive a program links; they now use
  the same resolver. An artifact taken from a system location is reported on
  stderr with its path instead of being resolved silently, and archives carry
  the Eshkol version they were built from (a build stamp in every
  `libeshkol-runtime.a` / `libeshkol-static.a`), so a version disagreement is
  reported as a warning. `$ESHKOL_PATH`/`-I` now also precede the installed
  `lib/` tree for `(require …)`, so a module search path the user named is not
  silently outranked by a module that ships with the compiler.
  `$ESHKOL_SYSTEM_PREFIXES` overrides the built-in system prefix list for
  unusual installs and for packaging tests. Regression test:
  `tests/toolchain/runtime_archive_resolution_test.sh` (ctest
  `runtime_archive_resolution_test`) stages a stale system archive and pins all
  four precedence rules plus both stderr diagnostics.
- **`(require stdlib)` (and any no-op `require`) silently shifted every
  subsequent top-level binding down one slot.** The bytecode compiler lowers a
  `require` of the always-available prelude to nothing, but the top-level (and
  function-body) compilers emit an `OP_POP` after any expression that grows no
  local — so the POP discarded a *live* stack value, misaligning the operand
  stack from the compiler's slot model for the rest of the program. A
  two-argument `define` written after `(require stdlib)` therefore bound to a
  stale slot and called as `NIL` ("calling non-function"). Every no-op `require`
  path now leaves an explicit `OP_NIL` placeholder for that POP to balance.
- **Type-annotated function parameters and return types were ignored by the VM
  compiler.** `(define (f (x : real) (y : real)) : real …)` bound its parameters
  to empty names (so `f` computed on unbound zeros) and compiled the `: rettype`
  annotation as a body expression. The compiler now resolves a `(name : type)`
  formal to its name and skips a `: rettype` return annotation between the
  signature and the body.
- **ESH-0214e: iter-scope partial reclamation — a resident tick loop that
  mutates persistent state every tick no longer leaks.** This closes the
  ESH-0214 memory-management series. ESH-0214b's automatic per-iteration
  arena reclamation was *all-or-nothing*: the static escape analysis rejected a
  loop body outright the moment it contained any persistent mutation, because a
  value the iteration allocates and then stores into outer/persistent state
  would dangle when the per-iteration arena scope was rewound. So a daemon/tick
  loop that mutates a knowledge base / workspace / growing list on **every**
  iteration got **no** reclamation and leaked one iteration's transient garbage
  forever — measured at ~3,366 bytes/tick (linear, unbounded; ~355 MB at
  100,000 ticks). ESH-0214e lowers a mutating-but-escape-safe loop with a
  **per-loop nursery region** instead of rejecting it, reusing — verbatim — the
  `with-region` deep-transitive escape-promotion path validated over a 48-hour
  resident run (ESH-0214c/d), not a second evacuator: every iteration allocation
  lands in the nursery arena, each of the six mutation channels' *existing*
  write barriers promotes any persistent-mutation escapee out of the nursery at
  the store, each TCO back edge promotes the loop-carried out-values out and then
  resets the nursery, and the loop exit escapes the result and tears the nursery
  down. The reclamation is sound by the same invariant as `with-region`'s
  `region_pop` (after promotion no surviving object points into the reset span) —
  textbook deterministic generational minor collection (nursery = young
  generation, write barrier = remembered set, back-edge recycle = minor
  collection), with no tracing pause. Admitted mutators are the five *structural*
  channels barriered unconditionally on the mutated structure pointer
  (`vector-set!`, `vector-fill!`, `hash-table-set!`, `set-car!`, `set-cdr!`);
  `set!` is deliberately excluded (its barrier fires only for globals, and
  proving a `set!` target is global rather than a shadowing enclosing-scope local
  needs lexical resolution this downward-only analysis lacks). Non-mutating loops
  keep the exact ESH-0214b arena-scope path (zero change to the
  `define_loop_flat_rss_aot` gate). After the fix the same tick loop is flat at
  34 MB — **identical to its explicit `with-region` twin** — with every stored
  value reading back correct, JIT and AOT, and clean under
  `ESHKOL_ARENA_POISON=1`. Adds `tests/memory/iter_scope_partial_reclaim_test.esk`
  / `.sh` (+ the `with-region` baseline twin) and the
  `iter_scope_partial_reclaim` ICC oracle.
  (`lib/core/runtime_regions.cpp`, `lib/backend/llvm_codegen.cpp`,
  `inc/eshkol/backend/binding_codegen.h`)
- **Driver: agent-FFI link requirements now propagate through the full
  transitive source closure, and a generated-program link failure under `-r`
  is fatal instead of silently masked (#334).** Two coupled defects in the
  compile/link driver. *(A) Dropped transitive dependency.* Native-link discovery
  walked a narrower graph than compilation: the compiler splices `(load "…")`,
  `(import "…")`, and `(require module)` transitively, but the agent-FFI
  requirement scan only followed top-level `(require …)`, so a helper reached
  only through `(load …)`/`(import …)` had its requirement lost and the produced
  binary failed the native link with unresolved `qllm_process_*` /
  `eshkol_sqlite_*` symbols. Requirement discovery now walks the **same**
  canonical transitive source closure as compilation (one `collectTransitiveSources`
  traversal produces both the source list and the requirements set), with no
  over-linking for programs that use no agent-FFI. *(B) Masked link failure under
  `-r`.* When the persistent-cache child's AOT build failed, `eshkol-run -r` fell
  back to a reduced in-process run and exited **0**, certifying a build that never
  linked. The driver now distinguishes the failure stages via a distinct exit
  sentinel (with a Windows `_putenv_s` portability shim for the sentinel env
  var): a native **link** failure is now **fatal** under `-r` (linker diagnostic
  surfaced, nonzero exit, the reduced program never runs), while a
  **codegen/compile** failure still falls back to the in-process JIT, preserving
  the named "called undefined function 'x'" diagnostic (the Bug-W contract) and
  its nonzero exit. A missing/unopenable input file under `-r` is likewise no
  longer swallowed with a zero exit. Regression:
  `tests/toolchain/transitive_ffi_link_test.sh`.
- **`gradient` recovers callable arity through a function parameter or
  wrapper (#330).** `(gradient f point)` misbehaved when `f` was reached indirectly —
  through a function parameter, a curried `((gradient f) point)` form, or any
  wrapper — instead of being named directly at the call site. A
  first-class-tensor-loss path added later (reverse-mode element seeding)
  unconditionally captured every vector/list/tensor point and invoked the
  closure with a single tensor argument, ignoring the callable's real arity, so
  a multi-parameter scalar loss such as `(loss x y)` was invoked as
  `loss(<tensor>)` and its scalar body misdispatched. The operator now recovers
  the callable's arity from its closure metadata and unpacks an N-element point
  into N scalar arguments, exactly as the direct-call path already did. Indirect
  and curried gradients are now byte-identical to the direct call for
  scalar multi-argument, vector, and non-polynomial losses, on both the JIT and
  AOT paths, with no closure-ABI change. There is no finite-difference fallback
  anywhere in the gradient path — every form is exact reverse-mode AD. A 25-check
  suite pins the direct/indirect/curried equivalence.
  (`lib/backend/autodiff_codegen.cpp`)
- **Arity-1 whole-point tensor-loss gradients no longer silently zero or crash
  (#338).** An arity-1 loss receives the whole point as one vector/tensor
  argument; when its body applies scalar arithmetic directly to that whole
  argument (elementwise tensor semantics, e.g. `(define (loss x) (* x x))`), the
  loss value is a *tensor* of AD nodes, not a single scalar AD node — a case the
  gradient paths assumed away. The forward-mode dual scheme-vector path (reached
  by a `(vector …)`/`(list …)` point) read only each element's primal and dropped
  the tangent (a silent all-zero gradient); the reverse-mode tape path (reached
  by a `#(…)`/`(tensor …)` point or any wrapped/curried form) saw a non-scalar
  output, skipped backprop, and read unwritten gradient slots or dereferenced a
  plain double as an AD-node pointer (zeros, or a SIGSEGV). `gradient` is ℝⁿ→ℝ:
  a scalar or 1-element-tensor output now backpropagates from the sole element's
  validated AD node (exact gradient), and a genuinely multi-element output raises
  a clean diagnostic naming `jacobian` — never zeros, never a crash. Exact and
  byte-identical across direct / wrapped / curried forms and
  `#()`/`(vector)`/`(list)` points on JIT and AOT; adds
  `tests/autodiff/gradient_tensor_loss_test.esk` (19/19).
  (`lib/backend/autodiff_codegen.cpp`)
- **Differentiation points are classified by runtime value, not AST node (#343).**
  The differentiation *point* of `gradient`/`hessian` was classified from the AST
  node kind, so a point that is a variable bound to a vector (or a general
  expression, or a `(the …)` wrapper) was misrouted the moment its concrete value
  diverged from what the node kind implied — two externally reported bugs share
  this root cause. `hessian` hard-classified a variable as scalar, so a variable
  bound to a vector took the scalar `f''(x)` path, read the vector pointer as a
  double, and SIGSEGV'd (#339) — while the identical point written as a `#(…)`
  literal worked because that is a provable collection at the AST level. And
  `gradient` returned a silently wrong value when the loss routed tracked values
  through a cons cell and the point was a `(vector …)`/`(list …)`, because the
  forward-mode dual path stored a dual-number into a cons slot and read it back
  with the int64 accessor, dropping the tangent (#340). Now only provable cases
  are classified at compile time (numeric literal → scalar; `#(…)` / `(tensor …)`
  / `(vector …)` / `(list …)` → collection) and the ambiguous ones (variable,
  general call/op, `(the …)`) defer to a runtime check on the evaluated value's
  tag; a cons-routing arity-1 vector point is routed through the reverse tape,
  and a `(list …)` point is normalized to a Scheme vector so it no longer falls
  through to the tensor path. Adds `tests/autodiff/varbound_point_matrix_test.esk`
  (the `{#(…) literal, inline (vector …), variable vector, inline/variable list,
  (the …) wrapper} × {plain, cons-routed, vector-ref, arity-N} × {gradient,
  hessian}` matrix, 33 cells, green JIT + AOT). (`lib/backend/autodiff_codegen.cpp`)
- **Vector-field AD operators at a `(list …)` point, and residual exit-0 masking
  in the driver (#354).** Two classes surfaced by the P8 escape-closure pillar.
  `jacobian` / `curl` / `divergence` SIGSEGV'd at a `(list …)` point: the
  cons-to-scheme-vector normalization added for the scalar-output operators
  (`gradient` / `hessian` / `laplacian`, #343) was never extended to the
  vector-field operators, so a cons-cell point fell through to the tensor path
  and was misread — while the identical point written as
  `#(…)`/`(vector …)`/`(tensor …)` or bound to a variable worked. The
  normalization is now applied across every AD entry point that reads a point
  subtype. Separately, the last exit-0 masking paths in the driver are closed:
  an AOT missing/unreadable entry file (which previously compiled an empty
  stdlib-only binary and exited 0), a `-r` syntax error (which previously ran
  the parsed prefix), and an unresolved `(require …)` (which previously ran
  without the module) are each now fatal with a nonzero exit before any binary
  is written. (`lib/backend/autodiff_codegen.cpp`, `exe/eshkol-run.cpp`)
- **Custom-VJP gradient silently zeroed on transitive captures.** A custom
  vector-Jacobian-product whose backward closure reached a captured value
  transitively (through an intermediate closure) had its contribution silently
  dropped, returning a zero gradient instead of raising. The capture walk now
  follows transitive closure references, so a custom-VJP node contributes its
  full sensitivity.
  (`lib/backend/autodiff_codegen.cpp`, `lib/core/runtime_autodiff.cpp`)
- **Reverse-mode AD tape pointer-array grows from the tape's owning arena
  (#345).** The reverse-mode tape's node-pointer array grew (on doubling) from
  the pinned shared arena, which `eshkol_region_enter` never region-swaps. Inside
  a `(with-region …)` the tape header and every forward-pass AD node live in — and
  are reclaimed at `region_pop` from — the region arena, but the grown pointer
  array kept accreting in the pinned arena and was never reclaimed: residual
  memory per step in a large region-scoped reverse-mode training loop. The tape
  now records the arena its header and initial node array were allocated from
  (`owner_arena`) and grows from it, so a tape created inside a region has its
  grown array reclaimed with the region, while a tape created outside a region
  grows into the same surviving arena as its header (never dangling behind a live
  header when growth happens inside an inner region — the use-after-free the
  naive "grow from the current arena" would cause). A tape with no recorded owner
  falls back to the shared arena, preserving prior behavior. Adds
  `tests/core/ad_tape_region_growth_test.cpp` (run under `ESHKOL_ARENA_POISON=1`):
  a tape grown inside a region reads back intact after `region_pop`, and a loop
  of region-scoped tape grows leaves the global arena flat (deterministic byte
  measurement).
- **`hessian`/`laplacian` residual at a variable-bound point (#349).** A remaining
  `hessian`/`laplacian` residual at a variable-bound differentiation point — left
  after the value-based point classification (#343) and the AD tape owner-arena
  routing (#345) — is closed, so repeated evaluation at a variable-bound point
  matches the inline-literal form. Externally contributed.
- **`parallel-map` corrupted results from closures using scope-based
  reclamation (memory-safety).** A closure mapped in parallel whose body used
  per-iteration scope reclamation — an internal named-let loop, or a builtin
  such as `memv` that brackets scratch allocation in a scope push/pop — could
  return dangling/overlapping structure once the input crossed the parallel
  threshold, surfacing nondeterministically as `car`/`cdr: argument is not a
  pair`, `SIGSEGV`/`SIGBUS`, or a hang; serial `map` over the same closure was
  always correct. Root cause: work-stealing pool workers are all pinned to the
  single shared thread-safe process arena, but a bump-arena's scope stack
  (`arena_push_scope`/`arena_pop_scope`/`eshkol_arena_iter_scope_end`) is
  intrinsically single-threaded — a pop rewinds the arena's shared bump pointer
  and frees everything allocated since the matching push. Concurrent workers
  therefore raced that one scope stack, and one worker's pop freed memory
  another was still using. Fix: on a pool worker operating on a thread-safe
  (shared) arena, scope operations degrade to **commit-only** (allocations are
  retained; the shared scope stack is never rewound), which is exactly the
  established "commit over reclaim = correctness over throughput" fallback.
  Per-iteration reclamation is deferred for the duration of parallel execution
  only; single-threaded and per-worker/region arenas keep full reclamation, so
  the flat-RSS loop behavior is unchanged. Verified with a repeated
  crash→green AOT/JIT fixture under `ESHKOL_ARENA_POISON=1`, ThreadSanitizer
  (73 arena data races → 0), and the existing parallel + region-race +
  per-thread-arena suites. New ICC gate: `parallel_map_scope_reclaim_race`.
- **Benign data race on the arena-poison diagnostic flag.** The
  `eshkol_arena_poison_enabled()` env-var cache was a plain function-local
  `int`, first-touched concurrently by pool workers (identical value, but a
  real ThreadSanitizer-visible race). Now a relaxed `std::atomic<int>`.
- **matmul tensor reads inside a defined function (#309) — verified resolved
  and now guarded.** #309 reported that a `matmul` result read back via
  `tensor-ref`/`tensor-data` from inside a `define`d function returned zeros,
  while only top-level reads were correct. The symptom was a mis-attribution of
  the Ozaki-II CRT-overflow/precision defect fixed in #307 (below): once #307
  landed, in-function reads return the identical correct data as top-level
  reads. This was reconfirmed by rebuilding at the #307 merge commit and at
  current master — the tensor read + arena + matmul codegen/runtime path is
  byte-identical between them — and exercised on JIT, AOT and the forced-GPU
  Ozaki path across captured-global / argument / in-function-matmul /
  nested-define / closure-capture / `with-region`-escape / large
  (GPU/BLAS-dispatched) forms; every in-function read matched the top-level
  read. Adds `tests/tensor/matmul_read_in_define_test.esk`
  (`matmul_read_in_define_jit_smoke` + `matmul_read_in_define_aot_smoke` CTests)
  and the `matmul_tensor_read_scope_oracle` ICC gate so the scope contract can
  never silently regress, and retires the defensive "read only at top level"
  work-around comments in the Ozaki correctness/certification fixtures.
- **Ozaki-II exact DGEMM correctness (Metal).** The opt-in CRT matmul
  (`ESHKOL_SF64_KERNEL=ozaki`) silently produced 5-30% numerical errors on any
  non-integer input, and returned NaN/garbage when asked for more than 16
  moduli. Three defects, all in `lib/backend/gpu/gpu_memory.mm`: (A) the
  modulus product `P` was formed in `__int128`, which overflows for N>=17 —
  poisoning every CRT constant while the count cap allowed up to 49; N is now
  hard-capped at 16 (the exact-f64 limit for K~=4096) and any larger request is
  clamped loudly; (B) adaptive-N minimised the moduli count subject only to a
  non-negative scaling exponent, driving the exponent toward 0 and truncating
  fractional inputs to near-integers — it now targets full f64 precision
  (E=52), and fixed N=16 is the shipped default; (C) the exponent used to scale
  inputs dropped the `frac_A/frac_B` terms the CRT uniqueness bound requires,
  so the scale overshot the bound on fractional data — it now matches the
  validated bound. Adds `tests/gpu/ozaki_correctness_gate.sh` /
  `ozaki_correctness_test.esk` (Ozaki vs an independent CPU f64 reference across
  integer/fractional/pi-e/wide-magnitude regimes at K up to 4096, plus a
  moduli-sweep) and the `ozaki-ii-correctness` ICC oracle.
- **Restored the `linear-solve` core boundary.** A build-boundary regression in
  the mixed-precision linear solver was corrected so the native and VM surfaces
  resolve the solver entry point consistently. (`lib/core/linear_solve.cpp`)
- **Linear-type checker scope leak and conditional over-restriction (#348).**
  Leaving a scope now clears its linear bindings (the checker's `popScope`
  previously leaked them), and a linear value used in exactly one branch of an
  `if`/`cond` is no longer rejected as a double-use by the branch sum-counting —
  so valid conditional linear gates type-check. Externally contributed.
  (`lib/types/type_checker.cpp`)

#### Automatic differentiation

- **`gradient` returned a silently wrong gradient whenever the function filled
  a vector with `vector-set!` inside a loop.** The derivative was attributed to
  the *last* element written, for every element read. Primal values stayed
  exact, the compiler exited 0 and stderr was empty, so a Jacobian assembled
  row by row — the standard idiom — came out uniform garbage while looking
  entirely plausible:

  ```scheme
  (define (vscale v s)
    (let* ((n (vector-length v)) (o (make-vector n 0.0)))
      (let loop ((i 0))
        (if (< i n) (begin (vector-set! o i (* (vector-ref v i) s)) (loop (+ i 1)))))
      o))
  (gradient (lambda (v) (vector-ref (vscale v 3.0) 0)) (vector 0.6 -0.8))
  ;; before: #(0 3)        after: #(3 0)
  ```

  Root cause, found by bisecting 94 commits: the iter-scope partial-reclamation
  work (ESH-0214e, above) gives such a loop a nursery `arena_reset()` per
  tail-call back edge, and the write barrier that must promote escapees tested
  `ESHKOL_IS_ANY_PTR_TYPE`, which **excludes** `DUAL_NUMBER` — a dual number is
  tagged in the immediate block but its data field addresses a headerless
  16-byte `{value, derivative}` pair in the nursery. The reset recycled it, the
  same address was reallocated on the next iteration, the primal survived and
  the tangent was silently corrupted. Fixed at the root with one shared
  "carries an arena pointer" predicate consulted by the write barrier, the
  `with-region` escape path and the nursery recycle alike (headerless payloads
  — dual and complex — get flat forwarded copies); tape-retained AD nodes now
  allocate from the tape's owning arena; and `evac_kind_for` reports
  live-interior-graph AD nodes in **every** build, not only instrumented ones.

- **`derivative`, `gradient` and `hessian` returned garbage at exact
  (rational or bignum) points, and lost exactness where the tower keeps it.**
  An exact rational or bignum is an ordinary number that happens to be
  HEAP-tagged, so its tagged data field holds a *pointer*. Every AD entry point
  turned its point into a double by *reinterpreting* that field, choosing the
  reinterpretation from the shapes it expected rather than from the tag the
  value actually carries — so the compiler differentiated at the rational
  object's **address**. `SIToFP` over a pointer yields a value of heap
  magnitude, a bitcast yields a denormal near 5e-314, and failing the scalar
  test routed the point down the *collection* path, where the rational object
  was dereferenced as `[dims][rank][elems]` (SIGSEGV). One authority per
  question now dispatches on the runtime tag —
  `eshkol_ad_seed_to_double` / `eshkol_ad_point_to_double` convert int64,
  double, bignum, rational, a jet's primal and a tower's `c[0]`, and **refuse**
  a non-numeric point with a catchable type error instead of inventing a number
  for it; `eshkol_ad_point_is_scalar` decides scalar-versus-collection once for
  every operator. Fixing that produced a correct *double*, which exposed the
  second half: the Taylor carrier `derivative-n` uses can hold exact
  coefficients and nothing routed the other operators to it, so
  `(derivative f 1/3)` answered `0.6666666666666666` where
  `(derivative-n f 1/3 1)` answered `2/3`. At an exact point the operators now
  run the same Taylor-tower pass, so the contract is an identity in value *and*
  in exactness — `(derivative f x)` = `(derivative-n f x 1)`,
  `(hessian f x)` = `(derivative-n f x 2)` — keeping `+ - * /` and
  non-negative-integer `expt` exact and demoting to f64 at the first
  transcendental, per R7RS exactness contagion.

- **A gradient of a runtime closure with declared arity 17-32 crashed, silently
  returned all zeros, or raised a type error.** Reaching a closure as a value —
  through a wrapper, a parameter or a variable — means its argument count is
  known only at run time, and the spread that supplies those arguments was
  clamped by a `MAX_CALL_ARGS_LIMIT` of 16, so the declared ceiling above 16
  was fictional. The spread is now emitted **once per module, out of line**
  (`__eshkol_ad_gradient_spread_call`, internal linkage), with the arity
  dispatch performed by the closure dispatcher's own runtime argument-count
  switch rather than a second per-arity one layered on top, and the variadic
  rest list built by a single runtime loop instead of an unrolled cons chain
  per arm. The ceiling of 32 is now real, and every existing call site without
  the new descriptor emits byte-identical IR. The out-of-line form is also
  substantially *smaller* than what preceded the arity widening — the shipped
  `stdlib.bc` is 6,004,048 bytes against 6,654,064 before — which restores
  Windows compile times, where COFF cannot discard the weak-linkage expansion
  that the inline spread multiplied at every gradient site.

#### Numeric tower and exactness

- **The bytecode VM decided a numeric result's exactness from its *value*
  rather than from its operands' tags, so inexact arithmetic silently became
  exact.**

  ```
  (/ (- 2.0 1.0) (+ 2.0 1.0))                        VM 1/3  native 0.3333333333333333
  (* 0.5493061443340549 (/ (- 2.0 1.0) (+ 2.0 1.0)))  VM 0    native 0.1831020481113516
  ```

  `number_val()` collapsed any integral-valued double to the exact `VAL_INT`,
  and the compile-time constant folder repeated the same value-shape test on
  folded literals, so `(- 2.0 1.0)` produced the exact integer 1 and
  `(+ 2.0 1.0)` the exact 3 — after which `OP_DIV` correctly divided two exact
  integers and answered `1/3`. R7RS 6.2.2 requires inexactness to be
  contagious: an inexact computation must not change numeric domain because an
  intermediate landed on an integer. Fixed at the constructor —
  `number_val_contagious()` decides exactness from the operand tags and is used
  by every arithmetic opcode and numeric builtin — and the folder now folds by
  the parser's literal exactness flags, declining to fold an exact literal it
  cannot represent (an integer wider than `int64`) rather than silently making
  it inexact. Independently, the rational-domain natives built both operands as
  `{(int64_t)as_number(v), 1}`, which can only carry an exact fixnum: a flonum
  was **truncated into the numerator**, so `(* 0.5493061443340549 1/3)`
  evaluated `(* 0 1/3)` and answered the exact 0, and `(+ 0.5 1/3)` answered
  `1/3`. Every mixed flonum/ratnum `+ - * /` was wrong, in both operand orders.
  Divergent native-versus-VM numeric combinations drop from 79 to 20; the
  remainder are bignum-over-bignum rationals not representable in the VM's
  rational type, now answered as a correctly-rounded inexact value and recorded
  as a justified parity row rather than left silent.

- **Native flonum `modulo`, `remainder`, `quotient` and the floor-division
  family were wrong across the board.** These are the native side of R7RS
  6.2.6, and a VM-parity differential sweep proved native was the wrong side:

  | expression | before | after |
  |---|---|---|
  | `(modulo 5.5 2.0)` | `6192449487634432` | `1.5` |
  | `(modulo 5.0 3)` | `0` | `2.0` |
  | `(remainder 5.5 2.0)` | `-0.5` | `1.5` |
  | `(remainder 7.0 2.0)` | `-1.0` | `1.0` |
  | `(exact? (quotient 7.0 2.0))` | `#t` | `#f` |
  | `(quotient 1e20 3.0)` | `9223372036854774784` | `3.3333333333333332e19` |
  | `(quotient 5.0 0.0)` | `9223372036854774784` | raises |
  | `(remainder 5.5 0.0)` | `+nan.0` | raises |
  | `(/ (expt 2 100) 0.0)` | `0` | `+inf.0` |
  | `(floor-remainder 7.0 2.0)` | `7881299347898368` | `1.0` |
  | `(floor-quotient (expt 2 100) 3)` | `1745178066` | exact |

  `modulo` and `remainder` had no flonum path at all, so both operands went
  through the int64 unpack and the result was the `SRem` of two IEEE-754 bit
  patterns; `remainder` additionally called the C library's `remainder()`,
  which is IEEE round-to-**nearest**, a different function from Scheme's
  truncated one. `quotient`'s double path `FPToSI`'d the truncated quotient
  into an int64 and packed it **exact**, breaking contagion by construction and
  saturating to one constant past 2^63 — a plausible integer is the worst shape
  a wrong answer can take, and `(quotient 5.0 0.0)` returned that same
  constant. `modulo` and `remainder` now have a flonum path built on `frem`
  (which is exactly C's `fmod`), `quotient` keeps its value as a double, the
  floor-division family is representation-polymorphic so `floor-remainder`
  genuinely **is** `modulo` as R7RS defines it, and a zero divisor raises
  uniformly across all of them including the mixed-bignum route. Fixed
  alongside: assignment conversion's `set!` scan had no case for `do` and never
  descended into cons-cell subtrees, so `(set! n …)` in a `do` body reported
  `variable 'n' is not mutable` and lost the assignment.

#### Language, modules and diagnostics

- **A compile-time error did not stop the build.** The compiler printed
  `ERROR: …` and then emitted, linked and *ran* a binary anyway, so a diagnosed
  program produced a wrong answer instead of a failed build. This is the
  mechanism that made this release's other defects silent: every one of them
  was reported at compile time, and every report was ignored. Reporting an
  error is one call at any of 805 sites across `lib/` and `exe/`, while
  propagating one is a return path through every enclosing frame — and the
  codegen frames recover by substituting a placeholder value and carrying on.
  All 805 sites funnel through four primitives in `lib/core/logger.cpp`, so an
  authoritative error tally now lives in one place and every path increments it
  for free; an emitted error-severity diagnostic prevents artifact emission and
  execution.

- **`define-library` validated its library name and threw it away, so an
  `import` could not see a library defined in the same file.**

  ```scheme
  (define-library (smoke v1_3) (export greet)
    (begin (define (greet who) (string-append "hi " who))))
  (import (smoke v1_3))
  (greet "world")
  ```

  reported `Module 'smoke.v1_3' not found` about a library written one line
  above the import, because `import` had nothing to consult but a filesystem
  search that can only ever find a library living in some *other* file.
  R7RS-small 5.6.1 defines a library by its `define-library` form and lets the
  forms that follow import it, so the unit's own libraries now come first in
  the resolution order — libraries established earlier in this compilation
  unit, then precompiled stdlib modules, then the search path — on all three
  back ends. Because a library becomes resolvable by being *processed* rather
  than by living in a particular file, the ordering rule falls out rather than
  being special-cased: an import above its `define-library` still fails, now
  with a diagnostic naming the line the library is defined on. The VM lane was
  the real gap — it knew none of `define-library`, `import` or `export`, and
  compiled the program above into bytecode that warned about five undefined
  variables and then died at run time calling a non-function, while the same
  file ran on JIT and AOT. Three latent VM defects were fixed with it:
  `(provide …)` emitted nothing at all, so the `OP_POP` after a form that bound
  nothing discarded a **live** value and shifted every later binding down a
  slot; the module loader compiled a required file's top-level forms without
  that POP discipline, desynchronising `n_locals` from the real stack depth;
  and a compile-time defect flag now stops both VM drivers, so
  `compile_and_run()` refuses to execute and the ESKB emitter refuses to write
  bytecode rather than leaving the process looking like it produced a good
  artifact.

#### Driver, linking and browser targets

- **`--shared-lib` never linked a shared library, and exited 0 anyway.** The
  documented capability raised `--compile-only`, wrote `<name>.o` and
  `<name>.bc`, and exited zero with no library anywhere. Making it link exposed
  why it could never have worked as documented: LLVM's calling convention for a
  first-class-struct return is not the platform C calling convention for the
  same struct. `eshkol_tagged_value_t` is 16 bytes passed internally as an LLVM
  struct of five fields, which the backend flattens into one return register
  *per field* — so a C caller compiled against the public header followed AAPCS,
  read the first two registers, and got the flags byte masquerading as the
  payload; with two tagged parameters the flattened fields overflow the
  argument registers outright. Library-mode codegen now emits, for each
  exported top-level function, a thunk that owns the exported name and speaks
  the platform C ABI, forwarding to the unwrapped body: `[2 x i64]` in and out
  on AArch64, x86-64 SysV, riscv64, ppc64le and loongarch64, `sret` plus
  by-pointer on Windows x64, and a diagnostic refusal on 32-bit targets, which
  neither shape models. Only the *linked-library* flavour gets thunks; the
  relocatable `--shared-lib -c` object is linked into other Eshkol modules that
  call with the internal convention and stays unwrapped. A second defect
  surfaced with it: the runtime archives were not position-independent, so a
  `thread_local` in `runtime_autodiff.cpp` got the local-exec TLS model and the
  library could not be produced at all on ELF (`relocation R_X86_64_TPOFF32 …
  can not be used when making a shared object`). `POSITION_INDEPENDENT_CODE`
  now applies to the object libraries feeding `libeshkol-runtime.a` and
  `libeshkol-agent-ffi.a` — on the object libraries, not the archives, where it
  would have been a silent no-op that looked like a fix.

- **The browser WASM glue was missing import stubs, so programs reaching them
  failed to instantiate.** `eshkol_write_value`, `eshkol_write_value_to_port`
  (the explicit-port form of `write`) and `eshkol_builtin_arena_used` had no
  `env` import stubs in either `web/eshkol-repl.js` or
  `site/static/eshkol-runtime.js`. Verified the authoritative way — by
  compiling a program that reaches those symbols with `--wasm` and diffing the
  module's real `env` imports against the glue, rather than trusting the import
  scanner. The environment-independent half of the execution-backed coverage
  gate also moves into the fast `surface-manifest` job, making it a **required**
  check so a deliberately broken policy floor or deficit ratchet can no longer
  hide behind a `continue-on-error` lane.

- **The checked-in browser artifacts were stale against the release tree.**
  `site/static/eshkol-vm.wasm` was last rebuilt on 25 July and the WASM
  differential caught `42_iota_srfi1` and `52_numeric_tag_dispatch` failing
  against it once those corpus files landed. Both checked-in artifacts are
  regenerated from the current source through their canonical recipes —
  `eshkol-site.wasm` 226,764 bytes, `eshkol-vm.wasm` 708,844 bytes (was
  696,657) — with the pandoc-rendered documentation fragments under
  `site/static/content/` refreshed to match current docs.

#### Test harness and release gates

- **Every green Linux test run was reported as an invalid run.** The
  toolchain-fingerprint guard in `scripts/lib/test_isolation.sh` tried
  `stat -f '%z %m'` before `stat -c '%s %Y'`. On BSD/macOS `-f` is the format
  flag and this is correct; on GNU coreutils `-f` is `--file-system` and takes
  no argument, so the "fingerprint" became free-block and inode counters, which
  change between samples — declaring `INVALID RUN: the compiler binary changed
  during this run` (exit 3) on fully passing runs. This had made every Linux
  lane's job conclusion non-informative since the isolation guard landed, on
  any branch, and plausibly accounts for much of the project's recorded
  "unreliable Linux runner" history. GNU is tried first now, with the BSD form
  as the fallback.

- **A test suite could die silently when its temp root was already clean.**
  `eshkol_test_isolation_prune_stale` globbed `"$root"/eshkol-test.*` straight
  into `du`; with zero matching directories the glob stays a literal unmatched
  pattern, `du` exits non-zero, and under a caller's `set -euo pipefail` that
  status is fatal for the assignment — with `2>/dev/null` swallowing the
  diagnostic, so there was no visible error at all. This is what killed
  `run_language_coverage.sh` (exit 1, zero output), making the
  `language_surface_coverage_floor` probe read as a false red when the coverage
  floor was in fact green. Fixed by returning early when there is nothing to
  prune, in both the count-based sweep and the size-based loop, which can drain
  to zero *inside* the loop and re-glob an empty root on the next pass.

- **The five-way surface baseline is re-anchored for the region-handle
  rename.** The P8 axis-6 baseline is a shrink-only ratchet, and current master
  produces the same disagreement count (640) with a different *set*: four
  `ad-*` entries resolved and four `region-open`/`region-close` entries
  appeared, a 4-for-4 swap a shrink-only ratchet cannot absorb, so the gate
  read FAIL with nothing regressed. The four new entries record a genuine,
  pre-existing cross-backend naming asymmetry — the VM dispatch table registers
  `_region-open` / `_region-close-list` while the native backend registers
  `region-open` / `region-close` — which stays open as a tracked build item
  rather than being resolved by editing the baseline.

- **A stale test reference became a build failure once diagnostics were
  fatal.** `tests/lists/stdlib_display_test.esk` displayed a symbol named
  `div`, which is defined nowhere: the division wrapper in
  `core.operators.arithmetic` was renamed `div` to `divide` because a function
  named `div` is emitted as a weak external that collides with libc's
  `div(int,int)` and corrupts the 16-byte tagged-value ABI. The test predated
  the rename and had been silently doing nothing; it now uses the real
  spelling.

### Documentation

- Added `agent.quantum`, `agent.pqc`, and `core.dbsp` reference pages, and
  closed the remaining ICC-flagged symbol-documentation gaps.
- **`docs/SDNC.md` refreshed to verified reality (#335).** Added the
  two-execution-layer framing — the SDNC weight-matrix layer (83-opcode ISA
  including 19 AD opcodes, 127/127 three-way verified) and the production
  bytecode VM (66-opcode enum + 720 native-call IDs) as two real, verified layers
  of one system with a precise opcode-for-capability correspondence; updated
  counts upward where reality exceeds the doc; corrected the
  `execute_step`/`run_reference`/`forward_with_weights`/`export_weights_binary`
  line references; documented the native-call ID surface (AD, tensors,
  consciousness, i128); and re-pinned the §7.3 SHA-256 checksums at the current
  verification SHA.

## [1.3.3-evolve] - 2026-07-16

An evolve release over v1.3.2-evolve that completes the Moonlab quantum
trajectory, closes native/VM semantic gaps, makes every declared language
surface executable under deterministic coverage, and incorporates the
correctness and memory-safety defects exposed while driving every release
gate green.

### Added

- **Moonlab quantum trajectory, S1-S5.** A gated `agent.quantum` integration
  now provides circuit construction, gates and measurement, Bell-verified
  quantum randomness, H2 Hamiltonians and VQE, exact/variational energy and
  gradients, differentiability through quantum circuits using the new
  `AD_NODE_CUSTOM` custom-VJP node, and FIPS 203 ML-KEM 512/768/1024 key
  encapsulation seeded from Moonlab's QRNG. The capstone adds a hosted macOS
  quantum lane, a Bell-CHSH gate, adversarial finite-difference checking,
  coverage evidence, and an ICC architecture invariant. Bell correlation is
  200/200, H2 VQE agrees with the exact energy to `4.4e-16`, and the CHSH gate
  measures `S ~= 2.86`. (#261, #268-#270, #272-#273)
- **Executable language-surface completion.** Deterministic native and VM
  probes now execute every one of the 1,057 declared language-surface rows.
  The coverage policy is ratcheted to **1057/1057 (100%)**, with no token-only,
  unreachable, or dead-code credit and zero uncovered high-risk rows. (#258,
  #274 and this release)
- **Production architecture evidence.** The ICC architecture model now checks
  eight static/runtime invariants, including honest VM dispatch, quantum QRNG
  provenance, WebAssembly import glue, executable coverage, and the corrected
  Poincare tangent metric.

### Changed

- **`make-parameter` and `parameterize` are fully wired, not emulated.** Native,
  VM, and WebAssembly-hosted paths use real dynamic parameter objects with
  converter-once semantics, unwind-safe push/pop behavior, and region write
  barriers. (#267, #271)
- **Hosted VM parity is explicit and executable.** The native-vs-VM corpus is
  now 68/68 across source and ESKB execution; the extended VM surface is 53/53.
  Multiple values, empty vectors, closure mutation, parameters, system calls,
  image operations, datum read/write serialization, polling,
  environment-aware process spawning, and other formerly dormant dispatch
  paths now execute with native-compatible results.
- **Large-list sort is stable and memory-bounded.** The old arena-retained
  list merge sort consumed roughly 32 GB for two million values. A stable
  bottom-up vector merge sort reduces peak RSS to about 362 MiB while
  preserving order for equal keys. (#266)
- **Persisted artifacts default to O2**, while JIT execution remains O0 unless
  requested; opt-level behavior is pinned by seven contract checks.
- **The cross-platform GPU correctness gate now executes on Windows** instead
  of silently treating Git Bash/MSYS hosts as unsupported. The Windows path
  uses the production compiler contract (official LLVM SDK ClangCL with Ninja,
  and MSVC as nvcc's host compiler), resolves multi-config `.exe` layouts, and
  accepts external build roots. A real RTX 3060 run dispatched through CUDA
  cuBLAS and matched the CPU reference across 10 probes with maximum relative
  difference `0`.
- **Release shell entry points reject unsafe paths before cleanup.** Shared
  guards enforce isolated build roots, reject symlink escapes and repository
  roots, and keep Bash, Git Bash, and constrained ARM64 behavior aligned.
  (#278)
- **The tag workflow can execute as a non-publishing dry run.** Manual runs
  build, test, package, validate, and checksum the complete 15-asset matrix,
  but cannot publish a GitHub release or update Homebrew. Packaged archives
  include the curated release notes, and the published release body is taken
  from the current release section rather than generated commit summaries.

### Verification

- Aggregate suite: **44/44 suites, 716/716 tests**.
- CTest: **76/76**; SICP full-book gate: **88/88** JIT+AOT probes.
- Chibi Scheme reference differential: **34/34 AGREE**; generative five-oracle
  differential: **127 programs, zero divergences**.
- VM parity: **68/68**; VM extended surface: **53/53**.
- Executable language coverage: **1057/1057 (100%)**; WebAssembly import glue:
  **101/101 imports provided**.
- Taylor monomorphization equivalence: **441/441 JIT + 441/441 AOT**, bit-exact
  through order eight.
- ICC architecture model: **8/8 invariants**; ICC release readiness:
  **100/100, oracle complete** (recorded in the readiness report).

### Fixed

- **CUDA-labeled release assets contain the real CUDA backend.** Linux x64 and
  ARM64-SBSA lanes install a pinned NVIDIA CUDA 12.4 toolkit; Windows x64 uses the
  matching NVIDIA network installer. CMake now fails closed when a required GPU
  backend is absent, and a build-graph gate requires `nvcc`, the CUDA runtime,
  cuBLAS, and both real CUDA sources while rejecting `gpu_memory_stub.cpp`.
  NVIDIA does not ship a native Windows ARM64 CUDA toolkit, so that unsupported
  archive is no longer advertised. Portable `sm_72/75/80/86/89/90` code keeps
  Xavier and current RTX/datacenter GPUs in the 15 honest artifacts. CUDA 12
  builds on newer GNU hosts also fail early unless the whole build uses a
  supported compiler, preventing nvcc-only host overrides from mixing
  libstdc++ ABI and search paths at final link time. Unix configure steps pass
  the toolkit root as a scalar CMake definition, retaining compatibility with
  the Bash 3.2 `set -u` environment on hosted macOS runners.
- **Generated CUDA links resolve the consumer toolkit.** Release packages no
  longer serialize hosted-runner `CUDA::cudart`/cuBLAS absolute paths into AOT
  and persistent-cache commands. Logical CUDA library names are resolved from
  explicit roots, `nvcc`, and standard Linux multiarch/toolkit layouts on the
  consumer, with the configured ABI major required in each selected directory
  and exact-major ELF link names preventing silent CUDA 12/13 substitution.
  Windows driver arguments retain native path separators, avoiding the newer
  MSVC STL `__std_replace_copy_2` helper that current build headers introduce
  for generic-path conversion but older compatible consumer import libraries
  do not provide.
- **Windows CUDA lanes use the CUDA 12.4 installer vocabulary.** The pinned
  network installer now receives only documented 12.4 subpackages. Compiler
  internals remain supplied by `nvcc`; nonexistent standalone `crt`/`nvvm`
  names from newer toolkit layouts can no longer abort setup before configure.
  CUDA builds use Ninja Multi-Config rather than the incompatible
  Visual-Studio-generator `ClangCL` CUDA integration: Eshkol C/C++ stays on the
  LLVM 21 SDK while `nvcc` receives the installed CUDA-supported v142 `cl.exe`
  host. This avoids CUDA MSBuild's empty-metadata `MSB4023` failure and retains
  the existing `Release/` test and package layout. The fail-closed backend
  verifier follows Ninja Multi-Config's nested implementation graphs, so it
  still proves the real CUDA sources are present. The selected v142 host path
  is normalized to CMake's forward-slash form before it becomes nvcc's
  `-ccbin` argument, preventing native backslashes from being consumed as
  escapes during CUDA compiler identification.
- **Generic release stdlibs no longer inherit the builder's AVX width.**
  `ESHKOL_TARGET_CPU=generic` now caps tensor codegen at the common 128-bit
  x86-64/AArch64 baseline while normal compiler and JIT runs remain host-
  specialized. The bitcode portability gate rejects fixed double vectors wider
  than two lanes in addition to scalable vectors and optional ISA attributes.
- **Relocated Windows package JITs publish their complete AD data ABI.** The
  Taylor-tower state globals are now explicitly registered with ORC, exported
  through the bounded PE runtime table, and required by the package validator.
  This prevents cache-disabled x64/ARM64 package runs from failing module
  materialization and cascading into duplicate initializer diagnostics.
- **Windows ARM64 package JIT unwind metadata and complete data reach are
  correct.** Live LLJIT and the persistent stdlib object cache now share one
  target-machine contract: the SEH-correct Small code model, per-function/data
  COFF sections, absolute RuntimeDyld call stubs, and import-address cells for
  host data. LLVM 21's AArch64-COFF Large model emitted invalid unwind metadata
  for probed frames. Small-model `PAGEBASE_REL21` references could then truncate
  because RuntimeDyld allocated JIT-owned code, read-only data, and writable
  data in unrelated address ranges. External declarations are lowered through
  `__imp_` cells, while a per-object RuntimeDyld memory manager now reserves all
  internal sections in one bounded arena. Explicit 120 MiB code and 2 GiB total
  span guards fail safely before either Branch26 or ADRP reach is exceeded,
  preserving stack probing, exceptions, cacheability, and full host-data reach.

- **Windows release-package links are relocatable.** AOT and persistent-cache
  links no longer replay the build runner's absolute compiler-rt or LLVM archive
  paths. Split-runtime programs link only their actual runtime dependency
  closure, while legacy `eshkol-static` consumers retain LLVM linkage. The
  native compiler driver is resolved relocatably at runtime and can be selected
  explicitly with `ESHKOL_CXX_COMPILER`; generated ClangCL/MSVC links resolve
  that consumer toolchain's architecture-matched LLVM 21 compiler-rt builtins
  archive, rather than assuming the driver will inject it or restoring a
  builder-only path. ClangCL/MSVC hosts now also retain and publish the bounded
  runtime symbol closure required by cache-disabled ORC JIT execution, matching
  the existing MinGW, Linux, and macOS behavior.
- **The poisoned region-evacuation RSS gates are self-contained.** The ESH-0214c
  and ESH-0214d/e million-iteration AOT harnesses now pass their source library
  path explicitly, so they exercise the persistent-mutation evacuation proof
  from clean shells instead of depending on an ambient `ESHKOL_PATH`.
- **Installed source modules resolve in cache-disabled JIT mode.** The REPL/JIT
  module search now selects the executable-relative source tree containing
  `stdlib.esk`, rather than mistaking the package's native-archive `lib/`
  directory for a module root. Missing explicit `require` forms fail the run
  instead of printing a diagnostic and continuing, and the release-package
  verifier now runs core and agent smokes with `ESHKOL_JIT_CACHE=0` and rejects
  module-loading diagnostics even if a lower layer returns zero.
- **Windows hosts avoid overlapping LLVM target retention.** ClangCL release binaries retain
  and publish the bounded cache-disabled-JIT ABI through their generated PE
  export table. They no longer force-load static X86/AArch64 LLVM target
  archives alongside `LLVM-C.dll`, which defined the `LLVMInitialize*` entry
  points twice and broke every native Windows link.

- **Correct Poincare-ball exponential-map convention.** Tangent vectors now
  use the Riemannian norm induced by `g_x = lambda_x^2 I`; off-origin
  exp/log round trips and geodesic lengths are locked against analytic
  identities instead of an inconsistent Euclidean tangent norm.
- **Complete R7RS multiple-value semantics.** `values`, `call-with-values`,
  `let-values`, `let*-values`, zero-value producers, multi-value arity, and
  nested producers now agree across native JIT/AOT and VM execution instead
  of silently collapsing to zero or one value.
- **Hosted port and system contracts.** Rebinding an input/output file port no
  longer invalidates the live stream; string ports preserve cursor/lifecycle
  semantics; `directory-walk` returns a proper Scheme list in deterministic
  breadth-first order; and `current-jiffy` preserves its exact 64-bit value
  instead of round-tripping through `double`.
- **Image buffer ownership in the VM.** Image read/grayscale/resize results are
  global-arena-owned and are no longer passed to `free`, eliminating a latent
  invalid-free/use-after-free path when these hosted operations execute.
- **Green-CI root fixes.** Tail-call-terminated library functions now complete
  symbol registration before return emission, preventing O2 dead stripping;
  Windows lite compile timeouts are reported honestly and use an appropriate
  budget; rational/bignum region evacuation preserves interior pointers; and
  `quantum-random-int` honors its bound on the LLVM path. (#262, #265)
- **Exact numeric and AD hardening.** Exact bignum `gcd`/division, rational and
  complex equality, bignum-aware VM arithmetic, forward-over-reverse
  Jacobian/Hessian composition, tensor-vector dual propagation, reshape and
  2-D matmul Hessians, and first-class/vector-gamma tensor gradients are now
  covered by differential and finite-difference oracles. (#229, #241,
  #246-#249, #252, #257)
- **Release-oracle portability and isolation.** Aggregate C++/SICP gates honor
  their requested build directory, the Chibi reference supervisor forces the
  portable `C` locale on macOS, and the generative oracle honors `BUILD_DIR`.
  LLVM target intrinsics remain allowed in freestanding objects while all
  undeclared hosted ABI dependencies are still rejected.
- **Windows hosted-runtime portability.** The region-runtime fallback no longer
  declares ELF weak functions on PE/COFF. Windows uses the hosted runtime
  directly, while non-Windows builds retain the weak fallback contract.
- **Generated ELF AOT binaries retain their dependency search paths.** Linux
  AOT linking now derives RUNPATH entries from linked `-L` directories,
  absolute shared-library inputs, and the selected host C++ compiler. Generated
  programs therefore find LLVM, the C++ runtime, curl, SQLite, ncurses,
  OpenSSL, and Nix-store dependencies without a custom `LD_LIBRARY_PATH`.
  (#279)
- **Linux release archives carry their image-codec runtime closure.** The
  packaged compiler, REPL, and generated run-cache/AOT executables resolve
  hashed and licensed libpng/libjpeg/libwebp/zlib shared objects under
  `lib/eshkol/runtime-deps`, rather than requiring target-host development
  packages or retaining release-builder paths. Release dependency installation
  also uses bounded retries so transient package-mirror failures cannot strand
  otherwise-valid ARM64 matrix jobs.
- **Precompiled standard-library artifacts are ISA-portable.** Release builds
  retain O2 optimization while targeting LLVM's generic architecture baseline
  for `stdlib.o` and `stdlib.bc`; ordinary compiler/JIT work remains
  host-specialized. A disassembly gate validates both the CMake target contract
  and the emitted IR, rejecting scalable-vector and optional wide-vector
  features inherited from a release builder. This prevents SVE-optimized ARM64
  stdlib artifacts from crashing on baseline Cortex-A72/ARMv8 consumers.
- **Exact tensor AD gradients for first-class losses and vector/learnable
  gamma; silent-zero backward paths now error instead of returning zero.**
  This corrects the v1.3.2-evolve CHANGELOG entry for #212, which claimed
  `input2` gradient plumbing was "complete" for `conv2d`/`batchnorm`/
  `layernorm`/`attention`. An adversarial audit found #212 was in fact a
  no-op — its test and roadmap updates landed, but no gradient code changed.
  The real fix is #229: (1) a loss with no compile-time `Function*` fell to
  the forward-mode-dual closure path, which loses the tangent for tensor ops
  and silently returns a zero gradient — added a reverse-mode tensor path in
  the closure branch of `AutodiffCodegen::gradient`; (2) batch-norm/layer-norm
  now wire per-feature gamma/beta as individual AD nodes instead of a single
  scalar, so vector/learnable gamma differentiates correctly; (3) remaining
  silent-zero backward paths for unsupported tensor ops now raise explicit
  unsupported-op errors rather than returning zero, honoring
  exact-AD-or-error. Finite-difference-verified exact in both literal and
  first-class forms across matmul/conv2d/attention-K-V/vector-gamma; autodiff
  suite 54/54, new input2 gate 24/24 under both JIT and AOT. (#229)
- **Region escape evacuator now covers the `PROMISE` heap subtype
  (ESH-0214e).** Adversarial-audit follow-up to ESH-0214d: `PROMISE` was left
  `EVAC_LEAF` despite carrying interior pointers (thunk at `+8`, cached value
  at `+24`). A `delay`/`make-promise` created inside `with-region` that
  escaped outward dangled after `region_pop`, observed as a segfault or
  `car: not a pair` under `ESHKOL_ARENA_POISON=1` when the promise was later
  forced. Adds an `EVAC_PROMISE` case that evacuates both slots; extends
  `region_evac_subtype_coverage` to exercise escape-then-force for both
  `delay` and `make-promise`. Flat ~116MB under poison; memory suite 100%.
  This completes the ESH-0214 region-evacuator series (ESH-0214a-e). (#230)
- **Subprocess `process-wait` kqueue lost-wakeup race** (documentation-only
  entry — the fix itself shipped in v1.3.2-evolve as commit `8443ddae` but was
  never recorded here). On macOS, `qllm_process_wait` registered
  `EVFILT_PROC`/`NOTE_EXIT` and then blocked in `kevent()`. If the child had
  already exited before the filter was registered — routine right after
  `process-kill`, and common under load for any short-lived child — the
  exit notification was never delivered, so `kevent()` blocked for the full
  timeout and reported "timed out" for a process that was already dead. This
  was the source of intermittent failures in
  `subprocess_shell_argv_test`'s "process-wait after process-kill exits"
  check on the macos-arm64-lite CI lane. Fix: after registering the filter,
  probe once with `waitpid(WNOHANG)`; if the child is already a zombie,
  drain, reap, and report exited — any exit strictly after the probe is still
  caught by the already-registered filter, closing the gap. The same
  `WNOHANG` recheck was added on the timeout branch as defense in depth.
  Verified on macOS arm64 (M2): 0/200 failures under 20-way parallel load,
  versus a reliable reproduction (roughly 1/48) beforehand.

## [1.3.2-evolve] - 2026-07-09

An evolve point release over v1.3.1-evolve: a resident-memory correctness fix
that unblocks forever-flat long-running loops that mutate persistent logic and
workspace state, thread-safe region scoping under parallelism, completion of
the automatic-differentiation input2 gradient path, an API-reference generator,
the Binary Lambda Calculus universal machine, and triage of three deferred
latent bugs. All release gates from v1.3.1-evolve remain green, plus a new
poison-hardened region-evacuator coverage gate.

### Added

- **`eshkol-doc` — API reference generator**: harvests Doxygen `/** @brief */`
  comments from `inc/` and `lib/` and generates `docs/api/` (Markdown pages
  plus an HTML index). First deliverable of the developer-experience tooling
  track. (#213)
- **Automatic-differentiation `input2` gradient plumbing**: `conv2d`,
  `batchnorm`, `layernorm`, and `attention` now propagate gradients to their
  second operand (kernel / gamma / K / V), completing the AD coverage matrix
  for these operators and hardening the finite-difference differential oracle.
  (#212)
- **`core.blc` — Binary Lambda Calculus**: a pure-Eshkol module implementing
  John Tromp's Binary Lambda Calculus, showcasing the language's
  lambda-calculus foundations. De Bruijn-indexed terms are represented
  homoiconically as s-expressions (`(var i)`, `(lam B)`, `(app M N)`);
  `blc-encode`/`blc-decode` convert to and from Tromp's self-delimiting bit
  encoding, and `blc-eval` reduces to beta normal form using **normal-order
  (leftmost-outermost)** reduction with correct De Bruijn shift/substitution
  and a divergence step-cap. Loaded on demand via `(require core.blc)`. The
  reference encodings are reproduced exactly (`I` = `0010`, `K` = `0000110`,
  pairing `λλλ.132` = `0000000101101110110`). See
  `docs/guide/BINARY_LAMBDA_CALCULUS.md`.
- **`core.blc` — universal machine U, BLC8 byte I/O, and lambda diagrams**:
  three deepenings of the BLC module. `(blc-U)` decodes Tromp's 232-bit
  (29-byte) self-interpreter `U`; applied via `(blc-encode-input (blc-encode M)
  input)` it runs the encoded program `M` on the input bit stream (Scott-list
  of `True`/`False` bits built with the `blc-pair` combinator), demonstrated on
  identity and constant-output programs. `blc-bytes->term`/`blc-term->bytes`
  (plus `blc-string->term`/`blc-term->string`) implement the BLC8 convention —
  a byte is a delimited big-endian list of 8 bits — round-tripping byte
  strings through lambda terms. `(blc-diagram term)` renders a term as a
  Tromp-style ASCII lambda diagram (abstractions as horizontal bars, variables
  as vertical lines, applications as horizontal links). Ground-truth `U` bits
  cross-checked against Tromp's De Bruijn term.

### Fixed

- **Region escape evacuator now covers logic and workspace subtypes
  (ESH-0214d).** The deep transitive escape evacuator (ESH-0214c) only
  deep-walked `CONS`/`VECTOR`/`HASH`/`TENSOR`/`EXCEPTION`/`CLOSURE`; the logic
  and workspace subtypes it mutates into persistent state — `SUBSTITUTION`,
  `FACT`, `KNOWLEDGE_BASE`, `FACTOR_GRAPH`, `WORKSPACE` — fell through to a
  shallow leaf copy that left their interior pointers dangling into the popped
  region arena (observed as `car`/`cdr` corruption in a resident tick loop).
  The evacuator now deep-walks these subtypes; records gain an explicit
  `RECORD -> VECTOR` mapping; and `arena_destroy` is poisoned under
  `ESHKOL_ARENA_POISON` so region use-after-free crashes loudly instead of
  passing by luck. New gate `region_evac_subtype_coverage_test` runs flat at
  ~110MB over 1,000,000 iterations under poison. (#226)
- **Thread-safe region scope stack (parallel-map + `with-region`).**
  `parallel-map`/future callbacks that opened a `with-region` raced on the
  shared current-arena slot and could crash under concurrency; the region
  hijack moved into the runtime with a parallel-scope guard, and new
  `eshkol_region_enter`/`eshkol_region_leave` runtime functions carry matching
  WebAssembly stubs so the lite build's import surface stays complete. (#217)
- **Deferred latent bugs triaged**: ESH-0223 (named-let stack overflow at high
  iteration counts), ESH-0227 (apply-loop SIGBUS), and ESH-0228 (`sleep-ms`
  argument type check). (#215)

### Changed

- **CI skips the build matrix for documentation-only changes.** A
  `paths-ignore` filter on the `push` and `pull_request` triggers means changes
  touching only `docs/`, Markdown, `notes/`, `press/`, or `LICENSE` no longer
  spin up the full compile/test/WebAssembly/sanitizer matrix. Website rebuilds
  are unaffected — they run through the separate Pages deploy on site source,
  compiler code, or the site-rendered documents.

## [1.3.1-evolve] - 2026-07-09

A resident-robustness point release over v1.3.0-evolve: two fixes that
matter specifically for long-running/daemon and large-persisted-state
workloads, plus a comprehensive documentation pass.

### Fixed

- **Iterative `read_list`** (#191): the reader's list-parsing path was
  rewritten from per-element native recursion to an iterative loop, so
  reading long flat lists — e.g. a 46K-entry persisted-state file — no
  longer overflows the native stack. Verified: the pre-fix reader SIGBUS'd
  at 20M elements; post-fix, the same input reads cleanly.
- **ESH-0214b per-iteration arena scope for `define` loops + catch-all
  guard** (#192): automatic per-iteration arena-scope reclamation, previously
  named-let-only, now also applies to self-tail-recursive top-level `define`
  loops, and the escape analysis that gates it no longer rejects a guard body
  outright — it accepts a catch-all guard clause (`#t`/`else`) whose body is
  itself escape-free. This enables flat-memory resident/daemon workloads
  built on the `define`-loop-plus-guard idiom. Verified in AOT mode: a
  1,000,000-iteration allocating guard-wrapped `define` loop holds peak RSS
  at 27MB with the fix on, versus 2608MB with the fix off.

### Documentation

- Added Doxygen doc-comments across all 64 public headers (`inc/eshkol/**`)
  and most implementation files (`lib/**`).
- Added a navigable documentation index (`docs/README.md`); reduced orphaned
  (unindexed) docs from 73 to 3.
- Updated press materials and website content to reflect the shipped v1.3
  state.
- Aligned roadmap views with what has actually shipped.

## [1.3.0-evolve] - 2026-07-07

The "evolve" release: an arbitrary-order automatic-differentiation system
(Taylor towers, phases P0-P12), full R7RS conformance on the portable
differential corpus, closure/TCO/memory robustness hardening, and a
permanent multi-pillar adversarial-testing infrastructure.

Release gates (green on the release SHA): SICP full-book gate 88/88 probes
under both `-r` and AOT (`scripts/run_sicp_smoke.sh`); CI 14/14 lanes
including windows-arm64; reference-Scheme differential oracle 34/34 AGREE vs
chibi-scheme 0.12.0 on the P7a portable corpus; ICC readiness oracle
`v1.3-evolve` ready.

### Added — Automatic Differentiation (Taylor-tower campaign, P0-P12)

Eshkol's AD system gains a second, orthogonal axis: **order**. Where the
existing forward-dual / reverse-tape engine differentiates once (or, with
perturbation tagging, is nested by hand), the Taylor-tower engine computes
*all* derivatives up to an arbitrary compile- or run-time order `k` in one
pass, exactly where the arithmetic allows it. Full design writeup:
`docs/design/AD_TAYLOR_TOWER.md`; campaign-to-release map: `docs/AD_CAMPAIGN.md`.
See the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md)
for a user-facing walkthrough and worked examples.

- **P0 — design + proof of concept** (#147, ESH-0185): Taylor recurrence
  design doc plus a standalone C proof-of-concept validating the recurrences
  to order 8 (63/63 checks) before any compiler work started.
- **P1 — runtime Taylor tower** (#148, ESH-0186): new heap subtype
  (`HEAP_SUBTYPE_TAYLOR`) and the core builtins:
  - `(taylor f x k)` → list of `k+1` coefficients `c[0..k]` where
    `c[n] = f⁽ⁿ⁾(x)/n!`, e.g. `(taylor (lambda (x) (exp x)) 0.5 4)`.
  - `(derivative-n f x k)` → the scalar `k`-th derivative `f⁽ᵏ⁾(x)`, e.g.
    `(derivative-n (lambda (y) (* y y y)) 3.0 1)` → `27`.
  - Epoch-tagged perturbations keep nested towers safe against
    perturbation confusion.
- **P2 — no-heap monomorphization** (#158, ESH-0187): when `k` is a
  compile-time literal and `f`'s body stays inside a whitelisted set of
  primitive ops, the whole tower is unrolled into branch-free SSA IR with
  zero heap/arena allocation (measured 0 B/iteration vs. P1's 288 B/iteration,
  ~1.2x faster at `-O2`), and is bit-exact with the P1 heap path.
- **P3 — JET8 subsumption analysis** (#160, ESH-0188): investigated folding
  the existing 8-jet forward/reverse-composition dual representation into the
  tower; found this is not fully achievable until P4/P5 land, so JET8 is kept
  as-is and the finding is documented rather than forcing a premature merge.
- **P4 — GUW multivariate (mixed partials)** (#162, ESH-0189): a
  Griewank-Utke-Walther directional-propagation layer, `core.ad.guw`,
  recovering arbitrary-order mixed partials of `f : ℝᵐ → ℝ` by propagating
  univariate towers along principal-lattice direction vectors and solving
  the resulting linear system:
  - `(taylor-propagate f xs v k)` → coefficients of `g(t) = f(xs + t·v)`.
  - `(mixed-partial f xs idxs)` → scalar `Dᵝf(xs)` for a multi-index list of
    variable indices with repetition, e.g. `(mixed-partial f xs '(0 1 1))`
    is `∂³f/∂x₀∂x₁²`.
  - `(gradient-n f xs order)` → the full symmetric order-`≥3` tensor as
    `(β . value)` pairs. `gradient`/`hessian` (order ≤ 2) are unchanged and
    still use the existing jet path.
- **P5 — reverse-over-Taylor** (#167, ESH-0190): fixes `gradient` composed
  with an inner `derivative-n`/`taylor` call, which previously returned 0
  because the tower was disconnected from the reverse tape. Tower
  coefficients now carry a parallel seed-tangent series so a `gradient` over
  a function containing a Taylor-tower call differentiates through it
  correctly.
- **P6 — exact-coefficient towers** (#163, ESH-0191): an
  `ESH_TAYLOR_COEFF_RATIONAL` tower mode stores coefficients as Eshkol's
  existing tagged numeric values (int64 / arbitrary-precision bignum /
  rational) instead of `double`, so `taylor`/`derivative-n` return **exact**
  arbitrary-order derivatives when `x` is exact and `f` uses only
  exact-preserving ops (`+ - * /`, non-negative-integer `expt`); the tower
  automatically demotes to `double` on overflow or on first transcendental
  call (verified with 68 exact-coefficient checks).
- **P7 — tensor-valued Taylor towers** (#169, ESH-0192): `core.ad.tensor_tower`
  generalizes a tower to "a tower of tensors" (one Cauchy-convolution series
  per tensor element, sharing a single shape), so high-order AD now composes
  with `matmul`/`conv2d`/`sigmoid`/`tanh` and other tensor ops unchanged.
- **P8 — Taylor models (validated AD)** (#173, ESH-0193): `core.ad.taylor_models`
  pairs a Taylor polynomial with a rigorous interval remainder bound, giving
  provable range/point enclosures — `(taylor-model f x0 r k)`,
  `(tm-range tm)`, `(tm-eval tm x)`, plus `tm-add`/`tm-mul` (Makino-Berz
  arithmetic) and accessors (`tm-order`, `tm-coeffs`, `tm-center`,
  `tm-radius`, `tm-remainder`, `tm-domain`).
- **P9 — differentiable control flow** (#178, ESH-0194): `if`/`cond`/`case`,
  named-let, recursion, and `map`/`fold` over Taylor-tower values now branch
  correctly — the one real gap was `compare()` in the arithmetic codegen not
  recognizing the Taylor heap subtype as numeric for `< > = <= >=`.
- **P10 — checkpointed high-order reverse** (#177, ESH-0195): a
  Griewank/binomial √N checkpoint schedule for reverse-mode differentiation
  through long chains, demonstrated in `core.ad.checkpoint`
  (`checkpointed-gradient` et al.), holding at most one block's tape live at
  a time instead of the whole chain (measured peak-node ratio ≈1.8 at N=200
  vs. ≈4.0 for the dense/non-checkpointed reverse sweep at the same depth).
- **P11 — tower-based user numerics** (#168, ESH-0196): `core.ad.taylor_numerics`
  builds numerical methods directly on top of the tower: `(taylor-ode-solve
  f y0 t0 t1 k n)` (fixed-step order-`k` scalar IVP solver),
  `(taylor-root f x0 k)` (Householder-family root refinement; `k=1` Newton,
  `k=2` Halley), and `(taylor-inverse-series f x0 k)` (Lagrange-inversion
  series reversion). All arguments are positional — the design doc's
  `#:order`/`#:steps` keyword-arg sketch had to be dropped because
  keyword-arg formals don't compile in any file with a dotted `require`
  (tracked as ESH-0220).
- **P12 — sparse high-order tensors** (#174, ESH-0197): `core.ad.sparse_guw`
  adds `(sparse-hessian f xs)` / `(sparse-hessian-pat f xs pattern)` (greedy
  star-coloring graph recovery of a sparse Hessian via one
  reverse-over-Taylor Hessian-vector product per color, plus accessors
  `sparse-hessian-{ref,nonzeros,row-ptr,col-idx,values,colors,directions,dense?}`)
  and `(sparse-mixed-partials f xs order pattern)` (order-≥3 block-decomposed
  sparse recovery). `sparse-mixed-partials` is implemented and unit-verified
  but not yet exercised by the release gate, since it triggers a pre-existing
  multivariate tower-codegen fragility at vector length ≥4 — treat it as
  available but not yet gate-hardened.

Known, documented AD limitations after this campaign (see
[Known Issues](docs/KNOWN_ISSUES.md) for the full, current list): plain
(order-≤2) vector gradient-of-gradient via `gradient`/`hessian` composition
is unaffected by P0-P12 and still needs the ESH-0096/ESH-0097 workarounds
below; `sparse-mixed-partials` is order-≥3-only and not gate-verified.

### Added — Build

- `--emit-depfile PATH` (#164, ESH-0215, `exe/eshkol-run.cpp`): walks the
  entry file's full `(load ...)`/`(import ...)`/`(require ...)` graph and
  writes a Makefile-format depfile, so incremental builds correctly
  recompile when an indirectly-loaded dependency changes (previously only
  the entry file itself was tracked, so editing a `(load ...)`ed helper left
  a stale object "up to date").
- `cmake/EshkolCompile.cmake` (#164, ESH-0215): the canonical
  `eshkol_compile_library` / `eshkol_compile_executable` CMake functions
  (previously only vendored ad hoc by downstream consumers), wiring
  `--emit-depfile` into `DEPFILE` on Ninja/Makefiles generators. See
  `docs/BUILD_INTEGRATION.md`.

### Fixed — R7RS Conformance

The reference-Scheme differential oracle (P7a, #140) diffs Eshkol against
chibi-scheme 0.12.0 "magnesium" over a 34-program portable corpus (numeric,
list, vector, string, char, binding, control-flow, equality, and I/O
probes). It started the campaign at 27/34 AGREE (79.4%) and the fixes below
bring it to **34/34 AGREE (100%) on that corpus**:

- `apply` with leading arguments before the final list argument, e.g.
  `(apply + 1 2 '(3 4 5))`, previously SIGSEGV'd (#142, ESH-0150).
- `vector-map`/`vector-for-each` over multiple vectors (R7RS §6.9), e.g.
  `(vector-map + #(1 2 3) #(10 20 30) #(100 200 300))`, previously ignored
  every vector past the first (#142, ESH-0151).
- Quasiquoted vector literals, `` `#(1 ,@(list 2 3) 4) ``, previously
  produced no output (#142, ESH-0154).
- `(substring s start)` (2-argument form) previously silently returned an
  empty result instead of defaulting `end` to `(string-length s)` (#155,
  ESH-0180).
- `cond`/`case` `=>` arrow clauses (`(cond (test => proc) ...)`), an
  allocating `vector-copy` (`(vector-copy v)`/`(vector-copy v start)`/
  `(vector-copy v start end)` — previously only the in-place `vector-copy!`
  existed), the `error-object?`/`error-object-message`/`error-object-irritants`
  condition-object family, and R7RS-conformant `write` string escaping
  (`\"`, `\\`, `\a`, `\b`, `\t`, `\n`, `\r`, and `\xNN;` for other control
  bytes) all landed together (#156, ESH-0152/0153/0155/0156).
- Nested ellipsis in `syntax-rules` templates, e.g. `(x ... ...)`/
  `((row ...) ...)` with `row ... ...` in the template, previously
  mis-expanded silently (exit 0, wrong value); pattern matching now tracks
  ellipsis depth via a `MatchTree` (#159, ESH-0128).
- `vector-copy` on a tensor-backed vector literal `#(...)` (as opposed to a
  `(vector ...)`-allocated vector) previously rejected the argument as "not
  a vector"; it now dispatches on heap subtype like `vector-ref`/`vector-map`
  already did (#175, ESH-0225) — this was the fix that closed the last gap
  in the reference-differential corpus.

The stale pre-fix reference-differential snapshot has been superseded by
this changelog entry; see `docs/reports/REFERENCE_DIFFERENTIAL_REPORT.md`
for the underlying probe list.

### Fixed — Compiler / Runtime Robustness

- **Mutual tail-call TCO** (#143, ESH-0102): a tail call from one function
  to a *different* function now emits a real LLVM `musttail` (guarded by
  matching signature/arity and no pointer-into-frame arguments) instead of
  the `TCK_Tail` hint the backend ignored, so mutually tail-recursive state
  machines (`even?`/`odd?`, ping/pong cycles) run in O(1) stack instead of
  overflowing at ~200-300k hops — verified to 5,000,000+ hops on AArch64.
  x86_64/i386/arm32/riscv64 keep the hint (their backends reject `musttail`
  for the tagged-value aggregate return); a real fix there needs an i128
  tagged-return ABI, tracked separately as ESH-0171.
- **Named-let TCO in every tail position** (#157, ESH-0211): the self-tail-call
  walk now recognizes named-let self-calls inside `cond`/`when`/`and`/`or`
  and nested-body tails, not just the loop's immediate body.
- **Closure-capture ceiling raised 16 → 64** (#154, ESH-0210): deeply curried
  lambda chains beyond 16 captures aliased into the wrong dispatch case and
  SIGSEGV'd (misreported as "stack overflow"); the call-site dispatch now
  over-provisions all capture-pointer slots instead of switching on capture
  count, so the callee reads only its own N ≤ 64 captures by address, not by
  dispatch axis.
- **TCO-loop capture-by-address bug in AD codegen** (#170, ESH-0221/ESH-0220):
  a distinct bug from the above — a TCO loop-carried alloca's *address*
  (rather than its current value) was forwarded as a derivative/gradient
  capture, producing garbage doubles; fixed with the same `isTcoLoopAlloca()`
  guard already used elsewhere in the map codegen. The same PR fixes
  keyword-args-with-dotted-`require` failing to compile.
- **`make-tensor` arbitrary rank** (#153, ESH-0205): `(make-tensor '(2 3 4 5)
  v)` previously silently truncated to a 3D shape; the dimension walk now
  follows the full cons chain (up to 16 dims) for any rank.
- **Iterative `length`** (#153, ESH-0206): stdlib `length` was non-tail-recursive
  and SIGBUS'd near 10⁶ elements; rewritten as a tail-recursive accumulator
  loop, now handling 10⁷+ elements in O(1) stack.
- **AOT/JIT cache invalidation on transitive dependencies** (#146, ESH-0183):
  the run cache key hashed only the entry file's own bytes, so editing a
  `(load ...)`/`(require ...)`/`(import ...)`ed dependency left `eshkol-run
  -r` silently running a stale cached binary; the cache key now hashes every
  file reachable from the entry file's full module graph.
- **Shutdown teardown-ordering race** (#165, ESH-0216): runtime shutdown now
  joins the parallel worker pool before running shutdown hooks and restoring
  signal handlers (previously a hook could race a still-running worker and
  use-after-free); AOT `main()` now always pairs `eshkol_runtime_init()` with
  `eshkol_runtime_shutdown()` at every return site. Verified clean across 50
  adversarial shutdown cycles and 50 external `SIGTERM` restart cycles.
- **Bounded-RSS long-running loops** (#166, ESH-0214/ESH-0214b/ESH-0214c): a
  production `read-line`-in-a-loop daemon ballooned to 9-24 GB RSS from
  three compounding bugs (an `if`-guarded named-let losing TCO, in-loop
  scratch allocas that leaked native stack per iteration, and a
  `with-region` control struct allocated from the arena instead of
  malloc/free). Fixed, and a new *automatic, zero-annotation* per-iteration
  arena-scoping optimization was added on top: a named-let TCO loop whose
  body is proven escape-safe by a conservative static analysis
  (`namedLetIterScopeSafe`) now reclaims its arena scope on every iteration
  back-edge, with a whole-program reachability pre-pass to hard-disable the
  optimization for any loop invoked from inside a `parallel-map` worker.
- **SIGILL on stack overflow / no altstack** (#135, ESH-0119): deep recursion
  overflow often surfaced as an uncaught `SIGILL` with zero diagnostic
  because the fatal-signal handler wasn't registered for `SIGILL`/`SIGFPE`
  and wasn't installed on an alternate signal stack. The handler now runs on
  a dedicated `sigaltstack`, stays async-signal-safe (`write()`-only), and is
  skipped under sanitizer builds so it doesn't fight ASan/TSan/MSan's own
  handlers. This was the shared root cause behind several previously-silent
  crash reports.
- **Tail calls through `guard`** (#172, ESH-0222): a self-recursive
  `define`/named-let tail call made through a `guard` error-boundary wrapper
  wasn't recognized as a tail call at all, so a per-tick `guard`-wrapped loop
  stack-overflowed after tens of minutes; the tail-position/recursive-call
  analyses now recurse into the guard node, and `guard`'s setjmp-based
  handler stack and dynamic allocas are kept sound across TCO back-edges
  (stacksave/stackrestore per iteration). A remaining named-let-only variant
  is tracked as ESH-0223.
- **Bare `()` in call/macro-argument position** (#171, ESH-0217): now lowers
  to the same zero-arg `CALL_OP` shape as `'()`/`(list)`, so macro pattern
  matching (which structurally matches that shape) recognizes it.
- **Forward-over-forward-over-reverse nested AD** (#138, ESH-0117): `gradient`
  over a 2-level nested `derivative` returned 0 from jet exhaustion (the
  4-jet dual had no free perturbation slot); extended to an 8-jet
  representation with a third nilpotent perturbation, plus a fix for a
  transitively-captured variable being double-indirected. Verified 56/56
  against a new `gofdofd` oracle generator.
- **Quasiquote inside macro templates** (#144, ESH-0126/ESH-0127): a
  `syntax-rules` template containing a quasiquote with an unquoted pattern
  variable stopped substituting past depth 1, because template substitution
  never recursed into the quote family of AST nodes; max correct depth went
  from 1 to 48.
- **FFI ergonomics** (#161): crypto FFI symbols moved from the optional
  agent-FFI archive into the always-linked core runtime archive (closing an
  AOT link race); added `string-byte-length` (byte count vs. codepoint
  count, needed anywhere a byte-sized `fwrite` was using `string-length` and
  truncating multibyte UTF-8 output).

### Fixed — VM / Web

- **Browser-REPL builtin reconciliation** (#179, ESH-0226): `tensor-matmul`
  had no VM `BUILTINS` table entry despite the native call existing; fixing
  the alias surfaced and fixed three deeper VM gaps in the same pass — every
  tensor-op case now type-checks its operand before reinterpreting it as a
  `VmTensor*` (previously segfaulted on a bare vector literal), variadic
  `(reshape tensor d1 d2 ...)` is a real VM compile-time special form, and
  `vm_tensor_matmul` gained the same 1D-operand promotion/contraction as the
  LLVM path. The precompiled `vm_prelude_cache.h` was regenerated (surfacing
  8 more pre-existing gaps: `gpu-*` ops and `tensor-cast`/`tensor-data`/
  `tensor-dtype`, tracked in `tests/vm_parity/PARITY.tsv`).
- **WASM glue completeness for eshkol.ai** (#176, ESH-0224, plus the
  precursor stub batches that led up to it): the hand-written WASM JS glue
  (`web/eshkol-repl.js`, `site/static/eshkol-runtime.js`) is now checked by
  `scripts/check_wasm_imports.py` in CI and stubs every `eshkol_*` symbol the
  LLVM wasm backend can emit as an `env` import — arena multi-value returns,
  tensor libm mapping, exception irritants, Taylor-tower reverse-over-Taylor
  helpers, and the new R7RS error-object accessors, among others. Previously
  a program reaching an unstubbed path LinkError'd at
  `WebAssembly.instantiate()`, surfacing on the live site as an endless
  "Loading Eshkol...".

### Added — Adversarial Testing Infrastructure

A permanent, ICC-wired set of test pillars beyond example-based regression
tests, aimed at classes of bug that fixed-shallow-depth, single-path testing
structurally cannot find. See `docs/TESTING.md` and `docs/VM_PARITY.md`.

- **P1 — differential harness + fuzzer** (#114): checks identical behavior
  (exit code, normalized stdout) across `jit`, `jit-nocache`, `aot-o0`, and
  `aot-o2` execution paths for a corpus plus a seeded fuzzer.
- **P2 — feature-pair edge matrix** (#112): ~30 language-feature axes
  composed pairwise, classified PASS/ASSERT-FAIL/CRASH/COMPILE-ERR/HANG.
- **P3 — AD finite-difference oracle** (#111): every generated AD probe
  self-checks against an in-language central finite-difference
  approximation, under both `-r` and AOT.
- **P4 — stress harness** (#115): wall-time and max-RSS budgets per
  workload (`tests/stress/budgets.tsv`), gated on exit 0 plus required
  stdout.
- **P5 — VM parity ratchet** (#118): `tests/vm_parity/PARITY.tsv` tracks
  every language surface as `vm-supported`/`native-only-justified`/`gap`;
  seeded at release time with 520 vm-supported / 41 native-only-justified /
  351 gap rows (27 of the gap rows are confirmed bytecode-VM behavioral
  divergences, not just missing coverage).
- **P6 — depth-parametric sweeps**: the meta-lesson from ESH-0117 (fixed
  shallow test depths miss depth-*dependent* bugs) turned into six permanent
  sweep families plus a coverage auditor: AD depth (#133), recursion/control
  depth (#132), syntax/data nesting depth (#152), numeric-tower depth/scale
  (#136), metaprogramming/module depth (#134), tensor/collection/string
  depth (#151), and a whole-language depth-coverage completeness gate (#131,
  `scripts/check_depth_coverage.py`). See
  `.swarm/DEPTH_PARAMETRIC_TESTING.md`.
- **P7 — external oracles**: reference-Scheme differential against
  chibi-scheme (#140, the R7RS-conformance oracle described above), a
  sanitizer fuzz harness with a bounded disk budget (#139), and a
  metamorphic-law oracle checking algebraic invariants (list/vector,
  numeric, roundtrip/control, sorting, string/char laws) (#137).

### Known Issues

Deep-edge findings surfaced by the adversarial harnesses; each has a minimal
repro and a ledger entry under `.swarm/tasks/`. None block ordinary use. The
canonical, continuously-maintained list is
[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — the summary below reflects
this release:

- Vector gradient-of-gradient (order-≤2 `gradient`/`hessian` composition,
  unaffected by the P0-P12 Taylor-tower work above) silently returns zeros;
  use nested scalar `derivative`/`derivative-n`, or the new order-≥3
  `mixed-partial`/`gradient-n` builtins, for exact higher-order results
  (ESH-0096).
- `hessian`/`laplacian` SIGSEGV when the evaluation point is a tensor
  literal `#(...)`/`(tensor ...)`; a `(vector ...)` point works (ESH-0095).
- Vector-param AD op combined with a captured local parameter fails LLVM
  verification (`PtrToInt source must be pointer`) (ESH-0072, ESH-0097).
- A closure created inside a named-let loop that `set!`s a global loses the
  mutation (ESH-0094).
- Deep non-tail recursion (~270k frames) is now a diagnosed error rather
  than a silent SIGILL (ESH-0119 fixed the missing diagnostic), but stdlib
  `sort`/`filter` are still non-tail-recursive and fail on very large inputs
  (ESH-0098, ESH-0101, ESH-0108). Mutual tail calls are proper R7RS tail
  calls on AArch64 (ESH-0102 resolved there); x86_64/arm32/riscv64 remain a
  bounded call pending an i128 tagged-return ABI (ESH-0171).
- Exact rational arithmetic degrades to double once a bignum is involved in
  ordinary (non-AD) arithmetic (ESH-0105) — this is orthogonal to the P6
  Taylor-tower exact-coefficient mode, which has its own dedicated
  bignum/rational representation.
- `sparse-mixed-partials` (P12) is implemented and unit-verified but not yet
  exercised by the release gate at vector length ≥4 (pre-existing
  multivariate tower-codegen fragility).
- 27 confirmed bytecode-VM behavioral divergences and 351 VM parity gaps are
  documented and tracked in the VM parity manifest
  (`tests/vm_parity/PARITY.tsv`, see `docs/VM_PARITY.md`).

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
- **Image I/O** (native platform/system codec backend: ImageIO/CoreGraphics,
  system libpng/libjpeg/libwebp, or GDI+). `image-read`, `image-write`,
  `image-to-grayscale`, `image-resize` load/save images as
  `(height, width, channels)` tensors.
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

- `#189` — SECURITY.md + docs/HARDENING.md + threat model.
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
