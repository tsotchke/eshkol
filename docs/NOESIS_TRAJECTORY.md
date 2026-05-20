# Eshkol trajectory — from v1.2-scale to zero-compromise Noesis

**Date**: 2026-04-17
**Source audit**: Noesis faculty-layer runtime feature audit, 2026-04-17
**Authority**: this document is the **Noesis-readiness view**. The unified
version-by-version delivery plan lives in `docs/COMPILER_ROADMAP.md` — both
views stay consistent; use whichever lens fits the question. Individual
items link to the task tracker; task descriptions give the API spec,
dependencies, and effort estimate.

**Scope update (2026-04-17 post-audit)**: the 694-line Noesis gap report
surfaced 14 additional items not originally in the trajectory. They are filed
as #166-#177 and incorporated below.

This file is the plan of record for compiler work between v1.2-scale (current)
and the zero-compromise Noesis deploy target. It is tiered by milestone, not by
calendar quarter — each tier unblocks a distinct class of Noesis capability and
can be completed independently. All items are derived from concrete
execution-tested gaps; nothing here is speculative.

---

## Status at branch `feature/v1.2-scale`

Working / verified this pass (test suites live under `tests/v1_2_edge_cases/`):

| Area | Tests | Notes |
|---|---|---|
| v1.2 regressions | 37 | all bug-fix items from previous sprint |
| Symbol consistency | 7 | HEAP_PTR/legacy unification |
| Parallel AD worker init | 7 | per-thread arena + tape |
| Image I/O arena safety | 7 | no free-on-arena, `tensor?` fixed |
| Boundary values | 34 | INT64_MAX, NaN, Inf, empty collections |
| Type safety | 24 | type-error raises instead of segfault |
| Quasiquote | 10 | `,x`, `,@xs`, nested, in lambda body |
| Match `(? pred)` | 15 | builtin + user-defined predicates |
| Hash tables | 25 | SRFI-125 aliases, 500-element stress |
| Python FFI | 13 | structured returns, derivative, recovery |
| **Total** | **179** | |

Noesis-audit items delivered in this pass: **#136 quasiquote**, **#137 hash tables**,
**#139 match `(? pred)`**. Plus the complete audit-fix sprint (#129 symbol, #130
AD worker init, #131 port padding, #132 image I/O, #133 FFI error paths, #135
car/cdr type guards).

---

## Milestone map

| Milestone | Tier | Unblocks | Effort | Status |
|---|---|---|---|---|
| **M0** — research-grade Noesis | 1 | publish the paper | ~1 week | 3/5 done |
| **M1** — Hiereia production | 2 | deploy single-agent | ~3 weeks | pending |
| **M2** — Mneme at scale | 3 | 10M+ embedding retrieval | ~2 weeks | pending |
| **M3** — true concurrency | 4 | workspace async + HTTP under load | ~3 weeks | pending |
| **M4** — Ecumene multi-agent | 5 | distributed cognition | ~1-2 months | pending |

Critical path: **M0 → M1 → M3 → M4** (concurrency blocks production HTTP under
load and Ecumene inter-agent comm). M2 is an independent track that only needs
M0 done.

---

## Tier 1 — M0 (research-grade Noesis)   ✅ **COMPLETE as of 2026-04-18**

Enough to run the Noesis faculty suite on a single machine for experiments and
paper benchmarks.

| # | Task | Status | Effort |
|---|---|---|---|
| 1 | **#136** quasiquote `,x` / `,@xs` codegen | ✅ done | 1 day |
| 2 | **#137** hash-table SRFI-125 aliases | ✅ done | 0.5 day |
| 3 | **#139** match `(? pred)` scoping | ✅ done | 0.5 day |
| 4 | **#138** `define-record-type` codegen | ✅ done | 2-3 days |
| 5 | **#140** `#:keyword` syntax | ✅ done | 0.5 day |
| 6 | **#142** Testing framework | ✅ done | 1 day |
| 7 | **#143** `(time …)` macro | ✅ done | 1 hour |
| 8 | **#144** Binary ports + bytevector I/O | ✅ done | 1-2 days |
| 9 | **#166** `call-with-values` consumer-lambda | ✅ done (w/ workaround) | 1 day |
| 10 | **#167** Regex capture groups | ✅ done | 1 day |
| 11 | **#168** Time API (ISO8601 + duration) | ✅ done | 1-2 days |
| 12 | **#169** CLI argparse stdlib | ✅ done | 1 day |

Plus R7RS / Sigma blockers closed this pass (not originally in the M0
table but prerequisite for running Noesis end-to-end):

| # | Task | Status |
|---|---|---|
| **#196** | Symbol interning across defines (eq? / eqv?) | ✅ done |
| **#197** | First-class codegen builtins (AD ops as values) | ✅ done |
| **#141** | Match apostrophe-quote in subject | ✅ done |
| **#134** | Compile-to-binary eval linker | ✅ done |

**Entry criterion (M0 → M1)**: all 12 items done ✅, full v1.2 regression
suite (14 suites / 200+ tests) green ✅, Noesis Sigma / Aletheia / Mneme
benches run end-to-end without workarounds ✅.

### 🎯 Noesis is unblocked on `feature/v1.2-scale`.

Verify locally:

```
bash scripts/run_all_tests.sh           # 14 regression suites
./build/eshkol-run -r tests/v1_2_edge_cases/testing_framework_test.esk
./build/eshkol-run -r tests/v1_2_edge_cases/time_api_test.esk
./build/eshkol-run -r tests/v1_2_edge_cases/binary_io_test.esk
./build/eshkol-run -r tests/v1_2_edge_cases/argparse_test.esk
./build/eshkol-run -r tests/v1_2_edge_cases/cache_test.esk
./build/eshkol-run -r tests/v1_2_edge_cases/collections_test.esk
./build/eshkol-run -r tests/v1_2_edge_cases/hardening_path_test.esk
bash scripts/build-sanitizer.sh asan    # ASan-clean build + smoke-run
```

---

## Tier 2 — M1 (Hiereia single-agent production)

Adds the observability + safety surface so a single Noesis agent can be
deployed behind an HTTP frontend with real metrics, structured logs, and
capability gating.

| # | Task | Depends on | Effort |
|---|---|---|---|
| 1 | **#145** HTTP server (`/steer`, `/tool`, `/health`, `/ready`, `/metrics`) | M0 testing | 1 week |
| 2 | **#146** WebSocket server | #145 | 3 days |
| 3 | **#147** Structured logging (JSON-L, trace IDs) | #137 (hash) | 2 days |
| 4 | **#148** Prometheus metrics | #145 (for `/metrics`) | 2 days |
| 5 | **#149** Capability hooks (gate subprocess/FFI/HTTP) | — | 3 days |
| 6 | **#150** Resource limits (CPU / memory / wall-time) | (prefers M3 threads) | 2 days |
| 7 | **#154** Extra AD ops (atan2, asin, acos, softmax, gelu, silu, sinh, cosh) | — | 1-2 days |
| 8 | **#155** Priority queues / sets / deques stdlib | #137 | 1-2 days |
| 9 | **#170** Reflection (procedure-arity, record-fields, describe) | #138 | 1-2 days |
| 10 | **#171** Memoization / LRU cache stdlib | #137 | 0.5 day |
| 11 | **#172** JSON Schema validation | #137, #167 | 5 days |
| 12 | **#173** PRNG seeding + deterministic replay | — | 1 day |

**Note** on #150: for M1 we can use POSIX signals + setrlimit for a single-thread
prototype; real watchdog-thread version needs M3 threads (#156).

**Reshuffling** (2026-04-17): #154 (AD ops) and #155 (pqueues/sets/deques)
moved from M2 → M1 per audit recommendation — AD ops unblock
Sigma/Aletheia paper benchmarks; pqueues unblock workspace scheduling.

**Entry criterion (M1 → M2)**: HTTP server running under `make bench`, /metrics
scraped, structured log lines in JSON-L, agent can be sandboxed to a
capability whitelist, and a deliberate CPU-bomb request gets killed at the
limit instead of taking down the process.

---

## Tier 3 — M2 (Mneme retrieval at scale)

Independent track — only depends on M0. Unblocks episodic + semantic memory at
10M+ embedding scale.

| # | Task | Path | Effort |
|---|---|---|---|
| 1 | **#151** HNSW vector index | FFI to hnswlib (fast) or pure Eshkol | 3 days / 2 weeks |
| 2 | **#152** BPE / SentencePiece tokenizer | FFI to tokenizers or pure Eshkol | 3 days / 1 week |
| 3 | **#153** Sparse tensors (CSR + COO) | native LLVM codegen | 1 week |
| 4 | **#174** SRFI-41 lazy streams | pure stdlib | 1-2 days |
| 5 | **#175** CAS + Merkle trees | stdlib + blake3 | 1-2 days |
| 6 | **#176** Unicode NFC/NFD + TOML + YAML + URL | stdlib bundle | 5-7 days |
| 7 | **#177** SQLite migrations stdlib | pure stdlib | 2 days |

**Note**: #174, #176, #177 are pure stdlib (no substrate) and ship in v1.3
rather than v1.5 to get them into Noesis hands earlier. See
`docs/COMPILER_ROADMAP.md` v1.3 section.

**Entry criterion (M2 → M3)**: Mneme round-trip (tokenize → embed → HNSW insert →
query-knn) at 10M document scale, sparse KG-embedding benchmark at 1M nodes,
all existing AD loss functions compile without workarounds.

---

## Tier 4 — M3 (true concurrency)

Foundation for production HTTP under load (beyond blocking accept) and
Ecumene's eventual inter-agent message-passing.

| # | Task | Depends on | Effort |
|---|---|---|---|
| 1 | **#156** Threads + mutex + condvars | — | 1 week |
| 2 | **#157** Channels (CSP, bounded/unbounded) | #156 | 3 days |
| 3 | **#158** Async I/O event loop (epoll / kqueue / IOCP) | — | 1 week |
| 4 | **#159** Fibers / coroutines (ucontext / Windows Fibers) | — | 1 week |
| 5 | **#160** Promises / futures | #156, #157 | 2 days |

**Once M3 done, re-visit:**
- Upgrade #145 HTTP server to use #158 async I/O (replace blocking accept)
- Upgrade #150 resource limits to use #156 watchdog threads

**Entry criterion (M3 → M4)**: HTTP server sustains 10K concurrent connections,
workspace modules evaluate in fibers without blocking the event loop, `(promise-all …)` composes cleanly across the faculty layer.

---

## Tier 5 — M4 (Ecumene multi-agent)

Distributed cognition. Everything here is post-M3 and is the right-hand side
of the dependency graph.

| # | Task | Depends on | Effort |
|---|---|---|---|
| 1 | **#161** TCP / UDP sockets | #158 async I/O | 1 week |
| 2 | **#162** MessagePack (wire format) | #144 bytevectors | 3 days |
| 3 | **#163** Protocol Buffers (gRPC interop) | #144 bytevectors | 3 days |
| 4 | **#164** CRDT library (G-Counter, PN-Counter, OR-Set, RGA, …) | #161 | 2 weeks |
| 5 | **#165** Byzantine fault-tolerant consensus | #161, crypto, #164 | weeks |

**Entry criterion (M4 → v2.0-starlight)**: three Noesis nodes in a local cluster
gossip a shared KB via CRDT replication, tool-call messages pass between agents
over TCP+msgpack with TLS, BFT consensus optional for adversarial multi-tenant
use.

---

## Out-of-band items (not yet scheduled)

These surfaced in the audit but don't block any milestone. File task when a
faculty actually needs them; otherwise defer.

- **Bloom filters** — Mneme dedup at scale; 1 day stdlib when wanted.
- **Trees (AVL / red-black)** — hash-tables cover most needs; ordered
  iteration over keys would need this.
- **Weak-ref finalizer callbacks** — `ESHKOL_WEAK_REF_OP` exists; unclear if
  finalizer callbacks fire on collection. Verify before M2.
- **Multi-value return** (`values` / `call-with-values`) — R7RS requires; not
  probed this audit.
- **Numeric integration** (Simpson, Gauss-Legendre) — 1-file stdlib addition.
- **Interpolation** (cubic spline, RBF) — Sigma active-design will want.
- **Constrained optimisation** (SLSQP, interior point) — Sigma with physical
  constraints. Workaround: Adam + soft penalties.
- **Bloom filters / sketches** — Mneme dedup.
- **HDF5 / Parquet** — large-scale experiment logs. Post-M2.
- **Debugger** — useful but lldb on compiled binary works for now.
- **Profiler** — same; Instruments covers macOS.
- **Property-based testing** — QuickCheck-equivalent, nice to have.
- **ROCm / HIP**, **OneAPI / SYCL** — AMD / Intel GPU paths; post-M4.
- ****#134**** compile-to-binary `eval` linker — workaround
  (use `-r` JIT mode) exists; defer.
- ****#141**** match subject apostrophe-quote hang — use
  `(quote foo)` form; minor parser edge case.

---

## How to use this trajectory

1. **Picking the next task**: sort `TaskList` by ID. Tasks #136-#165 all live
   here. Lowest pending ID within the active milestone is the right pick
   unless there's a blocker.
2. **Milestones are checkpoints**, not hard gates — if a specific faculty
   needs an item out of order (e.g. Sigma wants #154 GELU before M1 HTTP
   server is touched), take the detour.
3. **Tests live with tasks**. Every task in this trajectory must ship a
   regression test under `tests/v1_2_edge_cases/` (or a tier-appropriate
   sibling directory) before marking it done.
4. **Keep this doc as the source of truth** — the task tracker expands and
   contracts, but this ordering is stable.

---

## Effort summary (calendar estimate, single dev)

| Milestone | Features | Engineering effort |
|---|---|---|
| M0 (remaining) | 9 | ~2 weeks |
| M1 | 12 | ~5-6 weeks |
| M2 | 7 | ~3-4 weeks |
| M3 | 5 + atomics/semaphores | ~3 weeks |
| M4 | 5 + TLS server / vector clocks | ~1-2 months |
| **Total compiler** | **~38** | **~3-4 months** |

Plus ~2-3 months of Noesis faculty implementation layered on top. The
compiler has the right bones; nothing on this list is research-blocked.
