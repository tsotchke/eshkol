# Eshkol compiler roadmap — v1.2-scale → v2.0-starlight

**Date**: 2026-05-20
**Position in the doc set**: this is the **engineering-detail companion** to
the canonical [`ROADMAP.md`](../ROADMAP.md) at the repo root. When the two
disagree, `ROADMAP.md` is correct and this document needs updating.
`docs/NOESIS_TRAJECTORY.md` is the Noesis-readiness view of the same plan;
both stay consistent with `ROADMAP.md`.

Two axes are tracked side-by-side:

1. **Release version** (v1.2, v1.3, …, v2.0) — when a feature ships to users.
   Monthly cadence since v1.1-accelerate (2026-04-10); v1.2.0-scale shipped 2026-05-01 on schedule, and the v1.2.1-scale closeout point release shipped 2026-05-20.
2. **Noesis milestone** (M0, M1, M2, M3, M4) — what Noesis capability each
   feature unblocks. Taken from the 2026-04-17 Noesis gap audit.

The two axes are orthogonal. A single feature has both coordinates. For
example: `define-record-type` codegen is **v1.2-finalisation (release)** and
**M0 (Noesis)** — it ships in the next v1.2.x point release and unblocks
research-grade Noesis.

---

## Current status (verified 2026-05-20)

**Branch**: `master`
**Last shipped release**: v1.2.1-scale (2026-05-20)
**Base release**: v1.2.0-scale (2026-05-01).
**Status**: v1.2.1-scale closeout complete — all M0 audit blockers cleared.

Delivered in the v1.2.x closeout:
- All v1.2 audit-fix items (#107–#133) — image I/O arena safety, symbol
  consistency, FFI error paths, worker AD-tape init, etc.
- Noesis audit critical items: #136 quasiquote, #137 hash tables
  (SRFI-125 aliases), #139 match `(? pred)` scoping.
- Noesis regression suites live under `tests/v1_2_edge_cases/`.
- Noesis M0 late-close items #138, #140–#144, #166–#169, #134,
  and #141.
- Runtime and CLI fixes found during Noesis aggregate validation:
  atomic file writes, module/stdlib test-artifact isolation, runtime
  hash-table serialization, work-stealing external-submission routing,
  JSON read/write aliases, string search predicate clarification, and
  object-build CLI compatibility.

Verification snapshot:
- `scripts/run_all_tests.sh`: 37/37 suites, 528/528 self-reported tests.
- `ctest --test-dir build --output-on-failure --timeout 180`: 15/15.
- `scripts/run_v1_2_edge_cases_tests.sh`: 87/87.
- `scripts/run_stdlib_tests.sh`: 11/11.
- `scripts/run_modules_tests.sh`: 5/5.
- `scripts/run_parallel_tests.sh`: 7/7.
- `build/test_vm_c_api`: 81/81.
- `scripts/run_stress_tests.sh`: 3/3.
- Noesis focused neural smokes and `tests/smoke/all.esk`: passing,
  with aggregate `NOESIS_ALL_RC=0`.

---

## Principles for this roadmap

1. **Noesis-blocking items take precedence within their version's scope.**
   If M0 needs a feature that was originally scheduled for v1.3, it moves to
   v1.2-finalisation.
2. **Stdlib-only items ship continuously** — we don't bundle them into a
   release just because they share a module. Tests gate promotion.
3. **Cross-cutting infrastructure (networking, concurrency) ships in the
   version that introduces its primitive**; stdlib on top fills in later.
4. **Version-bump rule**: a new minor number (v1.3 → v1.4) requires a
   significant new capability surface, not just bug fixes. Point releases
   (v1.2.1, v1.2.2, …) carry audit fixes and stdlib additions.

---

## Version timeline

| Version | Codename | Target date | Theme |
|---|---|---|---|
| v1.2.x | scale | May 2026 | Model I/O + Noesis M0 closeout |
| v1.3 | evolve | Jun 2026 | R7RS polish + dev-experience + stdlib surface |
| v1.4 | connection | Jul 2026 | Networking + concurrency + linear types |
| v1.5 | intelligence | Aug 2026 | Neuro-symbolic bridge |
| v1.6 | reasoning | Sep 2026 | Production logic engine |
| v1.7 | synthesis | Oct 2026 | Self-writing programs |
| v1.8 | platform | Nov 2026 | Windowing + audio + embedded |
| v1.9 | types | Dec 2026 | Dependent types + effect types |
| v2.0 | starlight | Q1 2027 | Quantum + Lean integration |

---

## v1.2.x — "scale" (closeout)

Already-shipped items as of current branch:
- #136 quasiquote codegen ✅
- #137 hash-table SRFI-125 aliases ✅
- #139 match `(? pred)` scoping ✅
- Per-thread AD tape + arena init (#130) ✅
- Image I/O arena safety (#107, #132) ✅
- Symbol tagged-value consistency (#129) ✅
- Python FFI structured returns + error recovery (#110) ✅
- Car/cdr type guards on non-pair input (#135) ✅
- v1.2 regression suite plus current 87-test edge/security suite ✅

**Noesis M0 closeout status (verified 2026-05-19)**

| # | Item | Effort | Noesis tier |
|---|---|---|---|
| #138 | `define-record-type` codegen | ✅ done | M0 |
| #140 | keyword-symbol parsing used by Noesis `':keyword` paths | ✅ done | M0 |
| #142 | Testing framework (`define-test`, `check-equal?`) | ✅ done | M0 |
| #143 | timing helpers / `(time …)` surface | ✅ done | M0 |
| #144 | Binary ports + bytevector I/O | ✅ done | M0 |
| #166 | `call-with-values` consumer-lambda stdlib | ✅ done | M0 |
| #167 | Regex capture groups | ✅ done | M0 |
| #168 | Time API (ISO8601 parse/format, duration) | ✅ done | M0 |
| #169 | CLI argument parser stdlib | ✅ done | M0 |
| #134 | Compile-to-binary `eval` linker | ✅ done | — |
| #141 | match apostrophe-quote subject hang | ✅ done | — |

**Exit criterion for v1.2**: Noesis M0 is fully unblocked. All 11 closeout
items shipped. The current v1.2 edge/security suite passes 87/87,
`scripts/run_all_tests.sh` passes 37/37 suites and 528/528 self-reported tests,
CTest passes 15/15, stress tests pass 3/3, and Noesis `tests/smoke/all.esk`
exits with `NOESIS_ALL_RC=0` using the v1.2-scale build.

**Release naming**: tag this closeout as `v1.2.1-scale`.

### Noesis tracker reconciliation — 2026-05-19

Noesis' checked-in tracker now records W, Z, BB, GG, JJ, KK, and LL as closed
and lists zero open filed Eshkol bugs. The table below is the Eshkol-side
evidence used for that reconciliation.

| ID | Current Eshkol result | Disposition |
|---|---|---|
| W | `bug-W-forward-ref-error-unhelpful.esk` now exits at the first unresolved call, names `meta-meta-cycle`, and suggests `(require core.meta_cycle.meta_meta)`. The local shell regression `forward_ref_named_test.sh` passes 3/3. | Close as fixed. Keep a v1.3 diagnostics item for richer source stacks, not as an M0 blocker. |
| Z | `tests/v1_2_edge_cases/provide_aot_jit_parity_test.sh` passes: `(provide ...)` is informational in both JIT and AOT. | Close as fixed. True module-private internals move to v1.3 as an explicit module-system design, not as Bug Z. |
| BB | The stale Noesis repro no longer reaches the original second-call SIGSEGV; it now stops earlier with the fixed W-style missing-load diagnostic. Eshkol's minimized double-call / cross-file indirector regressions pass, and the full Noesis aggregate smoke passes. | Close as fixed/stale-repro. If Noesis wants the historical PCC repro retained, update its loads and re-file only if a current SIGSEGV returns. |
| GG | `NOESIS_TEST_VAR=42 ... bug-GG-getenv-string-predicate-mismatch.esk` prints `string?: #t` and `display: 42`. | Close as fixed. |
| JJ | `bug-JJ-loaded-helper-variadic-rest-raw.esk` now prints `rest: (#f)`, `pair?: #t`, `car rest: #f`, exit 0. | Close as fixed. |
| KK | `eshkol-run --version` exits 0 and prints `Eshkol Compiler v1.2.1-scale`. | Close as fixed. |
| LL | The Noesis repro script and Eshkol's `tests/v1_2_edge_cases/object_build_cli_contract_test.sh` both verify the positive contract: `--emit-object` is accepted, `-o requested.o` creates exactly that path, no `.o.o` artifact is produced, and `--shared-lib`, `-fPIC`, `-I`, and `-D` are accepted. | Closed. Keep the Eshkol regression and docs as the build-system contract guard. |

Result: there are no currently verified Noesis M0 substrate blockers left in
Eshkol. The remaining work is v1.3+ productization.

---

## v1.3 — "evolve" (June 2026)

R7RS polish, language ergonomics, stdlib expansion, developer experience.

v1.3 is not another bug-fix train. Its job is to make the v1.2 substrate
pleasant to build with, explicit enough for downstream build systems, and
stable enough for Noesis M1 single-agent production work before networking
lands in v1.4.

Several items originally planned for v1.3 landed during v1.2.x Noesis closeout:
LRU/memoization (#171), JSON Schema validation (#172), deterministic PRNG
seeding (#173), SRFI-41 streams (#174), collections (#155), and reflection's
`procedure-arity` / `type-name` / `describe` surface (#170). Structured logging
has a first implementation in `core.logging`; resource-limit primitives and a
minimal blocking HTTP surface also exist, but their production integration stays
in v1.3/v1.4 as listed below.

### Phase 0 — reconcile the v1.2 closeout

Deliverables:
- Publish `v1.2.1-scale`.
- Keep the Noesis tracker's closed status for W, Z, BB, GG, JJ, KK, and LL in
  sync with the 2026-05-19 repro evidence above.
- Keep the positive LL contract regression on the Eshkol side so object-build
  compatibility does not depend on the stale Noesis failing script.
- Keep the supported object-build command shape documented:
  `eshkol-run --emit-object -o path.o [--shared-lib] [-fPIC] [-I dir ...]
  [-D name[=value] ...] file.esk`.
- Document the current `(provide ...)` semantics: informational export list
  until v1.3's stricter module-privacy design lands.

Acceptance:
- Eshkol tree clean after release-tag prep.
- Noesis aggregate smoke remains `NOESIS_ALL_RC=0`.
- Noesis `noesis-bin` build path consumes the documented object CLI without
  a local compatibility shim.

### R7RS / language

Phase 1 delivers the promised syntax surface:
- R7RS library system: `define-library`, `import`, `export`, and import
  modifiers `rename`, `prefix`, `only`, and `except`.
- String interpolation: `~{expr}` inside string literals, preserving source
  locations for errors inside embedded expressions.
- Named keyword arguments: `(f #:key value)` with defaulting, required-key
  diagnostics, duplicate-key diagnostics, and ordinary positional/rest-arg
  interop.
- Destructuring let: `(let-match ((pattern value) ...) body ...)`, desugared
  through the existing `match` infrastructure.
- R7RS error object introspection: `error-object?`,
  `error-object-message`, and `error-object-irritants`.

Acceptance:
- Parser tests for all new forms, including malformed forms.
- JIT and AOT parity tests for each new form.
- Noesis keyword-argument call sites can delete any shim code.
- `docs/breakdown/SCHEME_COMPATIBILITY.md` updated from "missing" to
  "supported" or "supported subset" with examples.

### Module system and package tooling

Deliverables:
- Keep `(require ...)` / `(provide ...)` as the fast path for existing code.
- Lower R7RS `define-library` to the existing module graph so the stdlib object
  and LLVM weak-linking behavior remain intact.
- Add an explicit module-private mechanism instead of retroactively making
  `(provide ...)` hard-private. Candidate spelling: `(module-private ...)`
  or `(export ...)` strict mode, to be finalized before implementation.
- Add package registry metadata: package name, version, root module, exported
  libraries, native dependencies, and supported host profiles.
- Build `eshkol-doc`: parse docstrings/signatures from the module graph and
  emit Markdown/HTML API pages.

Acceptance:
- Existing v1.2 Noesis code keeps running unchanged.
- A new R7RS-style sample library imports with `only`, `except`, `prefix`, and
  `rename` in both JIT and AOT.
- `eshkol-doc` can generate stdlib docs without evaluating user code.

### Noesis M1 (Hiereia production stack)
| # | Item | Effort |
|---|---|---|
| #154 | Extra AD ops (atan2, asin, acos, softmax, gelu, silu, sinh, cosh) | ✅ finite-difference coverage landed in v1.2.x |
| #155 | Priority queues + sets + deques stdlib | ✅ landed in v1.2.x |
| #147 | Structured logging (JSON-L + trace IDs) | core contracts tested; needs Noesis integration |
| #149 | Capability / permission hooks | hosted allow-list hooks landed for agent surfaces and core file I/O |
| #150 | Resource limits (CPU / memory / wall-time) | hosted env initialization, accounting, and watchdog primitives tested |
| #170 | Reflection primitives (`procedure-arity`, `record-fields`, `describe`) | ✅ landed in v1.2.x |
| #171 | Memoization / LRU cache stdlib | ✅ landed in v1.2.x |
| #172 | JSON Schema validation | ✅ landed in v1.2.x |
| #173 | PRNG seeding + deterministic replay | ✅ landed in v1.2.x |

Phase 2 productionizes this surface:
- Keep focused tests for `core.logging`, including JSON escaping, trace-id
  scoping, level filtering, port redirection, and path-string sinks.
- Keep `core.capabilities` coverage on subprocess, shell, network, agent FFI
  wrappers, env access, generated file ports, and core file I/O. Future HTTP
  handlers should use the same deny-by-default active policy while preserving
  no-policy compatibility.
- Keep hosted `resource_limits` coverage for env initialization, malformed-env
  fallback, heap/stack accounting, tensor/string validators, and wall-time
  watchdog interrupts.
- Keep reflection docs and tests for user procedures, deferred record-field
  status, builtins, variadics, and imported functions.
- Keep #154 AD-op finite-difference checks in the stdlib/autodiff gates.
- Keep collections as stdlib, not syntax: priority queues, sets, and deques
  should remain ordinary data structures.

Acceptance:
- Noesis Hiereia can produce structured JSON-L logs with trace IDs.
- Noesis can install a capability policy and prove a denied subprocess/FFI/file
  operation fails deterministically.
- Hosted binaries can initialize resource limits from the environment and prove
  a deliberate wall-time limit requests a runtime interrupt.
- Extra AD ops pass numerical-gradient checks in JIT and AOT where applicable.

### Dev experience (original v1.3 roadmap)

Phase 3 improves the day-to-day compiler experience:
- Debugger with REPL step-through, breakpoints, variable inspection, and
  source-span-aware stack display.
- Sampling profiler with per-function inclusive/exclusive time, allocation
  counts where available, and JIT/AOT symbol demangling.
- Stack traces in runtime errors, including exceptions raised from malformed
  data produced by generated code.
- Profile-guided optimisation (PGO) and LTO flags in CMake and `eshkol-run`.
- ICC/smoke trace generation in CI so completion oracles have evidence rather
  than "blocked: no trace" statuses.
- User testing framework — **superseded by #142 landing in v1.2**.

Acceptance:
- A failing Noesis smoke reports a useful source stack without LLDB.
- Profiler can identify hot Eshkol functions in a Noesis neural smoke.
- PGO/LTO are opt-in and do not perturb default debug builds.
- CI uploads trace artifacts for oracle inspection.

### Stdlib — Noesis M2 items that are pure stdlib (no new substrate)
| # | Item | Effort |
|---|---|---|
| #174 | SRFI-41 lazy streams | ✅ landed in v1.2.x |
| #176 | Unicode `string-normalize`, TOML, YAML, URL parse/encode | 5-7 days |
| #177 | SQLite migrations stdlib | 2 days |

Phase 4 stdlib additions:
- Unicode normalization: NFC, NFD, NFKC, NFKD, case-folding helpers, and
  grapheme-aware iteration where practical.
- TOML parser/writer for configuration.
- YAML safe-subset parser/writer; no arbitrary object construction.
- URL parser in addition to the current percent-encode/decode helpers.
- SQLite migrations on top of the existing SQLite agent wrapper: migration
  table, ordered application, rollback-on-failure, checksum drift detection.
- Native glob, ANSI escape stripping, format-string helper, and cross-platform
  keychain support if time permits.

Acceptance:
- Stdlib tests cover malformed input and round trips.
- Noesis config files can be parsed without Python helpers.
- SQLite migration tests run in an isolated temp database and leave no artifacts.

**Exit criterion for v1.3**: Noesis M1 single-agent production surface usable
(minus production HTTP server and async networking — those come in v1.4).
All R7RS polish items shipped or explicitly demoted with rationale. Debugger,
profiler, object-build contract docs, structured logging tests, and capability
policy hooks are available for bench work.

---

## v1.4 — "connection" (July 2026)

Networking + concurrency. This is the **biggest release since v1.1** because
it establishes the substrate both M1 production HTTP and M3 concurrent
faculty evaluation depend on.

v1.4 turns the v1.3 local production surface into a hosted runtime. The
headline is not "a server demo"; it is a resource-safe, observable, concurrent
host where Hiereia can accept requests, stream logs, expose metrics, and stop
misbehaving work without taking down the process.

### Networking (M1 + M4 substrate)
| # | Item | Effort | Noesis tier |
|---|---|---|---|
| #145 | HTTP server (build on `inc/eshkol/http_request_utils.h`) | 1 week | M1 |
| #146 | WebSocket server | 3 days | M1 |
| #148 | Prometheus metrics primitive + `/metrics` endpoint | counters, gauges, histograms, and standard `/metrics` helper tested | M1 |
| #150 | Resource limits (CPU / memory / wall-time) | 2 days | M1 |
| #161 | TCP / UDP sockets | 1 week | M4 |
| — | TLS server (OpenSSL / mbedTLS wrap) | 3 days | M4 |
| — | Incremental compilation | 1 week | — |

Phase 1 networking deliverables:
- TCP sockets: listen, accept, connect, read, write, close, local/remote addr,
  non-blocking mode, and error objects.
- UDP sockets: bind, sendto, recvfrom, multicast-safe option surface later.
- TLS wrapper: server and client contexts, certificate/key loading, peer
  verification controls, and explicit failure diagnostics.
- HTTP server: keep the initial standard route helpers and custom route
  dispatcher for GET-only `/health`, `/ready`, `/metrics`, and Noesis-style
  tool endpoints, plus the shared 10 MiB body-size preflight. Next add richer
  request/response records, chunked-body handling, and graceful close.
- HTTP client can remain libcurl-backed initially, but the public Eshkol API
  should not expose libcurl-specific handles.
- WebSocket server: upgrade handshake, text/binary frames, close/ping/pong,
  and backpressure behavior.

Acceptance:
- Loopback HTTP and WebSocket conformance tests.
- `/health`, `/ready`, `/metrics`, and one Noesis tool endpoint run from
  Eshkol, not a Python wrapper.
- TLS smoke with a local self-signed cert.
- Socket handles are closed on normal return and on exception.

### Concurrency (M3)
| # | Item | Effort |
|---|---|---|
| #156 | Threads + mutex + condvars | 1 week |
| #157 | Channels (CSP, bounded/unbounded) | 3 days |
| #158 | Async I/O event loop (epoll / kqueue / IOCP) | 1 week |
| #159 | Fibers / coroutines | 1 week |
| #160 | Promises / futures | 2 days |
| — | Atomic ops (CAS, fetch-add) | 1-2 days |
| — | Semaphores / barriers | 1 day |

Phase 2 concurrency deliverables:
- Public thread API with join/detach and exception propagation.
- Mutex, recursive mutex if needed, condition variable, semaphore, barrier.
- Channels: bounded, unbounded, close semantics, select-style wait if feasible.
- Futures/promises aligned with existing runtime futures so `parallel-map` and
  user futures do not diverge.
- Fibers/coroutines for lightweight cooperative tasks.
- Async I/O event loop over kqueue/epoll/IOCP, with timers and socket readiness.
- Atomic operations over boxed mutable cells or a dedicated atomic type.

Acceptance:
- No data races under the thread/parallel regression suite.
- Blocking primitives cannot deadlock the work-stealing pool in nested
  `parallel-map` or future waits.
- Async loop handles timers plus socket readiness in one process.
- Noesis can evaluate independent faculty tasks concurrently without global
  hash-table corruption.

### Wire formats
| # | Item | Effort | Noesis tier |
|---|---|---|---|
| #162 | MessagePack | 3 days | M4 |
| #163 | Protocol Buffers (proto3) | 3 days | M4 |
| #175 | Content-addressable storage + Merkle trees | 1-2 days | M2 |

Phase 3 observability and limits:
- Keep the existing metrics primitive available through the standard `/metrics`
  response helper, then promote it into the full HTTP server endpoint loop.
- Keep counters, gauges, and histograms with stable names, label validation,
  Prometheus escaping, cumulative bucket rendering, and reset semantics covered
  in `core.metrics`.
- Route structured logs through the WebSocket log stream with trace-id filters.
- Enforce resource limits per request: CPU/wall-time, memory, tensor elements,
  string/body sizes, subprocess count, and file/network capability policy.
- Add request cancellation and shutdown draining.

Phase 4 wire-format deliverables:
- MessagePack encode/decode for numbers, strings, booleans, null, arrays,
  maps, bytevectors, and extension hooks.
- Proto3 parser/compiler subset sufficient for generated encoders/decoders.
- Content-addressable blobs and Merkle trees for Mneme and Ecumene handoff:
  hash, store, fetch, verify inclusion proof.

### Linear types (original v1.4 item)
- Linear resource types (`ESHKOL_VALUE_SOCKET`) — compile-time single-ownership
  enforcement on handles. Complements networking — once sockets exist, they
  should be linear so they can't leak.

Implementation rule:
- Every network/file/thread resource gets an owning value and a borrowed access
  form. The compiler should reject double-close and obvious use-after-close
  where the type checker can see it; runtime guards handle dynamic paths.

**Exit criterion for v1.4**: Hiereia deployable as production HTTP agent with
async I/O under 10K concurrent connections, /metrics scraped, log stream
accessible via WebSocket, resource-bounded requests. Ecumene wire formats
and socket primitives available (the Ecumene faculty layer itself is
separate — the substrate is ready).

**Big release** — tag this `v1.4.0`, not a point release.

---

## v1.5 — "intelligence" (August 2026)

Neuro-symbolic bridge. Unblocks Noesis M2 (Mneme at scale).

v1.5 is where Eshkol stops being only a fast symbolic/runtime substrate and
becomes the memory-and-learning substrate Noesis needs. The release should be
split so Mneme gets a usable vector/token/sparse-tensor stack early, while the
more speculative differentiable-logic pieces land behind tests rather than
marketing claims.

### Symbolic/neural fusion (original v1.5 theme)
- Symbol embeddings — each KB symbol gets a learnable vector
- Soft unification — differentiable cosine similarity
- LSTM / GRU cells
- Differentiable logic programs
- Attention over KB
- Gradient estimators (Gumbel-Softmax, straight-through)

### Noesis M2 (Mneme at scale)
| # | Item | Effort |
|---|---|---|
| #151 | HNSW vector index | 3 days (FFI) or 1-2 weeks (native) |
| #152 | BPE / SentencePiece tokenizer | 3 days (FFI) or 1 week (native) |
| #153 | Sparse tensors (CSR + COO) | 1 week |
| — | Integer tensors (int64 element type) | 3 days |
| — | Complex-element tensors | 3 days |
| — | Interpolation methods (cubic spline, RBF) | 1 week |
| — | Hypothesis-testing framework (stats) | 3 days |

Phase 1 tensor substrate:
- Sparse tensors: CSR and COO storage, construction, conversion to/from dense,
  sparse-dense matmul, sparse elementwise map where meaningful, and serialization.
- Integer tensor element type with clear promotion rules and no accidental
  float round trips.
- Complex tensor element type wired through basic arithmetic, FFT-facing paths,
  and serialization.
- AD compatibility matrix: explicitly document which sparse/int/complex ops
  are differentiable and which are value-only.
- Interpolation: cubic spline and radial-basis functions with numerical tests.
- Hypothesis testing: t-test, chi-square, KS test, bootstrap confidence
  intervals, and multiple-comparison helpers.

Phase 2 tokenizer and text substrate:
- BPE tokenizer with training from corpus, encode/decode, vocabulary save/load,
  byte fallback, and deterministic merges.
- SentencePiece-compatible importer/exporter if native SentencePiece is not
  implemented directly.
- Unicode normalization from v1.3 becomes mandatory input hygiene here.
- Tokenizer must be usable by Noesis without qLLM or Python.

Phase 3 vector index:
- HNSW index API: create, add, remove/tombstone, search-k, save/load, stats.
- Distance metrics: cosine, L2, inner product.
- Deterministic build mode for reproducible tests.
- FFI-backed implementation is acceptable for v1.5.0 if the API preserves a
  future native implementation path.

Phase 4 neuro-symbolic bridge:
- Symbol embedding table integrated with the existing KB symbol space.
- Soft unification primitive with differentiable similarity scores.
- Differentiable logic program representation: rules as weighted clauses or
  matrices, explicit loss functions, and gradient tests.
- Attention over KB facts and proof-tree nodes.
- LSTM/GRU cells with forward/backward numerical-gradient tests.
- Gradient estimators: Gumbel-Softmax and straight-through estimator, both
  documented with their bias/variance tradeoffs.

Acceptance:
- Mneme stores and searches 10M+ embeddings in a stress profile, with a smaller
  deterministic CI profile.
- Tokenizer trains and round-trips on a Noesis corpus without qLLM.
- TransE and DistMult examples run end-to-end using Eshkol tensors and AD.
- Sparse tensor ops have finite-difference or algebraic correctness tests.
- Noesis M2 smoke covers tokenize -> embed -> HNSW insert -> retrieve -> use in
  an Aletheia/Sigma decision.

**Exit criterion for v1.5**: Mneme can hold 10M+ embeddings, tokenize without
qLLM, run TransE/DistMult KG embeddings. Noesis M2 fully unblocked.

---

## v1.6 — "reasoning" (September 2026)

Production logic engine.

- Backward chaining (Prolog-style SLD resolution + cut)
- Forward chaining (Rete for production rules)
- Constraint solving (finite domain + optional SAT)
- Knowledge graphs (RDF-style SPO/POS/OSP indexing)
- KG embeddings (TransE, DistMult — wired to v1.5 gradient substrate)
- Constrained optimisation (SLSQP, interior-point)

Noesis impact: none blocking — Aletheia and Sigma already use the v1.1 logic
engine. v1.6 adds production-quality depth for scaled workloads.

---

## v1.7 — "synthesis" (October 2026)

Self-writing programs.

- Neural-guided search with beam-width scoring
- Synthesis holes (`??`) — type-directed enumeration
- Graph Neural Networks (message passing)
- I/O example synthesis
- Neural theorem provers (v1.5 embeddings + v1.6 backward chaining)
- Symbolic autodiff (compile-time gradient code generation)

Noesis impact: unblocks Aletheia meta-tactic synthesis, Sigma grammar
self-extension.

---

## v1.8 — "platform" (November 2026)

Windowing, audio, embedded targets.

- Cross-platform windowing (X11/Wayland/Cocoa/Win32 via SDL2)
- Event system (keyboard, mouse, touch, resize)
- Real-time audio (CoreAudio/ALSA/WASAPI, lock-free ring buffer)
- MIDI (PortMIDI)
- Vulkan compute (SPIR-V from Eshkol)
- Embedded cross-compile (ARM bare-metal, RISC-V, freestanding)
- Multi-GPU (device selection, P2P)
- FreeBSD support, Nix/RPM packaging
- ROCm / HIP (AMD GPU)
- OneAPI / SYCL (Intel GPU)

Noesis impact: indirect — only matters if Noesis ships a desktop UI or
embedded-agent variant.

### Noesis M4 follow-ups that land here (distributed, multi-node)
| # | Item | Effort |
|---|---|---|
| #164 | CRDT library | 2 weeks |
| — | Vector clocks / Lamport timestamps stdlib | 1-2 days |
| — | Distributed multi-agent gradient sync | 1 week |
| — | Distributed training primitives (all-reduce, DDP) | 1 week |

Initial stdlib substrate landed early: `core.distributed` provides Lamport
clocks, vector clocks, state-based G-Counter / PN-Counter CRDTs, OR-Set, LWW
register/map CRDTs, and an RGA-style ordered sequence. The first wire-format
substrate also landed early: `core.msgpack` provides bytevector MessagePack
encode/decode for null, booleans, exact 32-bit-range integers, UTF-8 strings,
binary bytevectors, list arrays, and explicit maps. The remaining v1.8 work is
networked replication over the v1.4 socket layer plus broader MessagePack /
Proto3 coverage.

---

## v1.9 — "types" (December 2026)

Types-as-proofs.

- Dependent types (full enforcement on existing HoTT scaffolding)
- Refinement types with SMT solver (Z3 or miniZ3)
- Effect types (Pure / IO / State / Exception)
- Algebraic effects
- Row polymorphism
- Session types (communication protocols)
- Higher-rank types (rank-2 polymorphism)
- Weak references with finalizer callbacks

Noesis impact: optional — dynamic types suffice for M0-M4. v1.9 enables
certified-code tracks in Aletheia and provides static guarantees for
production deploy.

---

## v2.0 — "starlight" (Q1 2027)

Quantum + formal verification.

- Qubit type (linear, no-cloning enforced via v1.9 dependent linear)
- Gate primitives (H, CNOT, Rz, T, S, SWAP, Toffoli)
- Measurement (collapses qubit to classical bit)
- Variational algorithms (VQE, QAOA, parameter-shift rule through AD)
- Quantum simulator (state-vector, GPU-accelerated, ~25 qubits)
- Lean integration (export type judgments for formal verification)
- #165 Byzantine fault-tolerant consensus (PBFT / Tendermint / HotStuff)

Noesis impact: Ecumene adversarial multi-tenant deploy uses BFT consensus;
everything else is research-frontier extension.

---

## Post-v2.0 (candidate — not yet scheduled)

From the Noesis audit §5.6 nice-to-have list, plus cross-cutting items
that don't fit cleanly into a single version:

- Jupyter kernel for Eshkol (research UX)
- AEAD encryption (libsodium FFI)
- Interactive debugger (`break` → REPL drop-in, step)
- Code coverage reporting
- AOT bundle packager (produces self-contained distributable)
- GPU memory residency API (`tensor-to-gpu`, `tensor-to-cpu`)
- XLA operation fusion (activate MLIR/StableHLO path)
- Polyhedral optimisation for nested tensor loops
- PostgreSQL, HDF5, Parquet
- NumPy array-protocol interop
- Python → Eshkol + Eshkol → Python calls (bidirectional)
- Fluid dynamics solvers

---

## Cross-version views

### Noesis tier × Version matrix

| Noesis tier | v1.2.x | v1.3 | v1.4 | v1.5 | v1.6+ |
|---|---|---|---|---|---|
| M0 research-grade | 12 items + 2 untiered closeout fixes | — | — | — | — |
| M1 Hiereia prod | 5 early stdlib/runtime items | 3 remaining local-production items | 4 hosted-production items | — | — |
| M2 Mneme scale | 1 early stdlib item | 2 pure-stdlib items | 1 storage item | 3 at-scale items | — |
| M3 concurrency | — | — | 7 items | — | — |
| M4 Ecumene | — | — | 3 items | — | 3+ items |

### Critical-path dependencies

```
v1.2.x (Noesis M0)
  │
  ├─► v1.3 stdlib (Noesis M1 partial + M2 partial)
  │     │
  │     └─► v1.4 (Noesis M1 HTTP + M3 concurrency + M4 substrate)
  │           │
  │           ├─► v1.5 (Noesis M2 at-scale)
  │           ├─► v1.6 reasoning engine
  │           ├─► v1.7 synthesis
  │           └─► v1.8 platform + M4 distributed
  │                 │
  │                 ├─► v1.9 types
  │                 └─► v2.0 quantum + BFT
```

Noesis M0 ships with v1.2.x. M1 = early v1.2.x stdlib/runtime wins + v1.3
local-production hardening + v1.4 networking. M2 = v1.2.x/v1.3 stdlib +
v1.4 CAS/Merkle + v1.5 at-scale memory. M3 = v1.4 concurrency. M4 =
v1.4 wire formats + v1.8 distributed + v2.0 BFT.

---

## Tracking table — all Noesis-audit items by ID

Complete list, filed in the task tracker, linked to Noesis audit tier and
target release version. Use this as the handoff cheatsheet.

| # | Item | Noesis tier | Release |
|---|---|---|---|
| #136 | Quasiquote interpolation | M0 | v1.2.x ✅ |
| #137 | Hash tables | M0 | v1.2.x ✅ |
| #138 | `define-record-type` | M0 | v1.2.x ✅ |
| #139 | match `(? pred)` | M0 | v1.2.x ✅ |
| #140 | keyword-symbol parsing used by Noesis `':keyword` paths | M0 | v1.2.x ✅ |
| #141 | match apostrophe-quote subject | — | v1.2.x ✅ |
| #142 | Testing framework | M0 | v1.2.x ✅ |
| #143 | `(time …)` macro | M0 | v1.2.x ✅ |
| #144 | Binary ports + bytevector I/O | M0 | v1.2.x ✅ |
| #145 | HTTP server | M1 | v1.4 |
| #146 | WebSocket server | M1 | v1.4 |
| #147 | Structured logging | M1 | v1.3 |
| #148 | Prometheus metrics | M1 | v1.4 |
| #149 | Capability hooks | M1 | v1.3 |
| #150 | Resource limits | M1 | v1.4 |
| #151 | HNSW vector index | M2 | v1.5 |
| #152 | Tokenizer | M2 | v1.5 |
| #153 | Sparse tensors | M2 | v1.5 |
| #154 | Extra AD ops | M1 | v1.3 |
| #155 | Priority queues / sets / deques | M1 | v1.2.x ✅ |
| #156 | Threads + mutex + condvars | M3 | v1.4 |
| #157 | Channels | M3 | v1.4 |
| #158 | Async I/O event loop | M3 | v1.4 |
| #159 | Fibers / coroutines | M3 | v1.4 |
| #160 | Promises / futures | M3 | v1.4 |
| #161 | TCP / UDP sockets | M4 | v1.4 |
| #162 | MessagePack | M4 | v1.4 |
| #163 | Protocol Buffers | M4 | v1.4 |
| #164 | CRDT library | M4 | v1.8 |
| #165 | Byzantine consensus | M4 | v2.0 |
| #166 | call-with-values consumer | M0 | v1.2.x ✅ |
| #167 | Regex capture groups | M0 | v1.2.x ✅ |
| #168 | Time API (ISO8601 + duration) | M0 | v1.2.x ✅ |
| #169 | CLI argparse | M0 | v1.2.x ✅ |
| #170 | Reflection (`procedure-arity`, `type-name`, `describe`) | M1 | v1.2.x ✅ |
| #171 | LRU cache | M1 | v1.2.x ✅ |
| #172 | JSON Schema validation | M1 | v1.2.x ✅ |
| #173 | PRNG seeding + deterministic replay | M1 | v1.2.x ✅ |
| #174 | SRFI-41 streams | M2 stdlib | v1.2.x ✅ |
| #175 | CAS + Merkle trees | M2 | v1.4 |
| #176 | Unicode NFC/NFD + TOML + YAML + URL | M2 stdlib | v1.3 |
| #177 | SQLite migrations | M2 | v1.3 |
| #134 | Compile-to-binary eval linker | — | v1.2.x ✅ |

---

## Disagreements with the audit (and why)

Two items where my scheduling differs from the audit's ordering:

1. **Audit puts AD ops (#154) and pqueues (#155) in M1 / v1.3; my original
   trajectory put them in M2 / v1.5.** Audit wins — pqueues unblock workspace
   scheduling (immediate faculty need) and AD ops unblock Sigma/Aletheia
   paper benchmarks. Pqueues/sets/deques landed early in v1.2.x; AD-op
   verification remains v1.3.
2. **Audit lists CRDTs (#164) under M4 / v1.4 effort.** I'm parking CRDT work
   in v1.8 because the distributed-multi-agent use case isn't pressing until
   after the single-agent HTTP layer is proven in v1.4-v1.5. M4 wire-format
   substrate (TCP/UDP + msgpack + protobuf) still ships in v1.4 as planned —
   the gap is only in the higher-level CRDT library on top.

Neither is a classification disagreement on whether the feature is needed —
both ship. The difference is ~3 versions of lead time on CRDTs.

---

## How to run against this roadmap

1. **Sprint planning**: pick all open tasks filtered by `release = v1.x`. Use
   the task-tracker `blockedBy` field for ordering within a sprint.
2. **Ship rule**: a release cannot cut until every task targeted at it is
   either completed or explicitly demoted to the next release (with
   rationale captured in the task comments).
3. **Noesis faculty work** proceeds in parallel — faculty engineers consume
   whichever M-tier's prerequisites are met at the time.
4. **Audit cycle**: re-run the Noesis audit at each release tag. New items
   filed become next-release scope.
