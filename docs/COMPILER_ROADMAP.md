# Eshkol compiler roadmap — v1.2-scale → v2.0-starlight

**Date**: 2026-04-17
**Authority**: this document supersedes the earlier single-axis trajectory.
`docs/NOESIS_TRAJECTORY.md` remains valid as the Noesis-readiness view;
this document is the authoritative version-by-version delivery plan.

Two axes are tracked side-by-side:

1. **Release version** (v1.2, v1.3, …, v2.0) — when a feature ships to users.
   Monthly cadence since v1.1-accelerate (2026-04-10); v1.2.0-scale shipped 2026-05-01 on schedule.
2. **Noesis milestone** (M0, M1, M2, M3, M4) — what Noesis capability each
   feature unblocks. Taken from the 2026-04-17 gap audit
   (`/Users/tyr/Desktop/noesis/docs/audits/eshkol-feature-gap-report-2026-04-17.md`).

The two axes are orthogonal. A single feature has both coordinates. For
example: `define-record-type` codegen is **v1.2-finalisation (release)** and
**M0 (Noesis)** — it ships in the next v1.2.x point release and unblocks
research-grade Noesis.

---

## Current status (2026-04-17)

**Branch**: `feature/v1.2-scale`
**Last shipped release**: v1.2.0-scale (2026-05-01)
**Tests passing**: 179 across 9 regression suites.

Delivered in this sprint:
- All v1.2 audit-fix items (#107–#133) — image I/O arena safety, symbol
  consistency, FFI error paths, worker AD-tape init, etc.
- Noesis audit critical items: #136 quasiquote, #137 hash tables
  (SRFI-125 aliases), #139 match `(? pred)` scoping.
- Noesis regression suites live under `tests/v1_2_edge_cases/`.

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
- 179-test regression suite ✅

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
items shipped. The current v1.2 edge/security suite passes 85/85, CTest passes
14/14, and Noesis `tests/smoke/all.esk` exits with `NOESIS_ALL_RC=0` using the
v1.2-scale build.

**Rename**: the next v1.2.x release should be tagged `v1.2.1-noesis-m0` so
external consumers know the Noesis audit gaps are closed.

---

## v1.3 — "evolve" (June 2026)

R7RS polish, language ergonomics, stdlib expansion, developer experience.

Several items originally planned for v1.3 landed during v1.2.x Noesis closeout:
LRU/memoization (#171), JSON Schema validation (#172), deterministic PRNG
seeding (#173), and SRFI-41 streams (#174). Reflection (#170) has useful
`procedure-arity` coverage but still needs the broader `record-fields` /
`describe` completion pass before it should be treated as fully closed.

### R7RS / language
- R7RS library system (`define-library`, `import` with renaming / prefixing / `only` / `except`)
- String interpolation (`~{expr}` syntax)
- Named keyword arguments (`(f #:key value)`)
- Destructuring let (`(let-match ((cons a b) lst) …)`)
- Package registry + documentation generator (`eshkol-doc`)

### Noesis M1 (Hiereia production stack)
| # | Item | Effort |
|---|---|---|
| #154 | Extra AD ops (atan2, asin, acos, softmax, gelu, silu, sinh, cosh) | 1-2 days |
| #155 | Priority queues + sets + deques stdlib | 1-2 days |
| #147 | Structured logging (JSON-L + trace IDs) | 2 days |
| #149 | Capability / permission hooks | 3 days |
| #170 | Reflection primitives (`procedure-arity`, `record-fields`, `describe`) | 1-2 days |
| #171 | Memoization / LRU cache stdlib | 0.5 day |
| #172 | JSON Schema validation | 5 days |
| #173 | PRNG seeding + deterministic replay | 1 day |

### Dev experience (original v1.3 roadmap)
- Debugger with REPL step-through, breakpoints, variable inspection
- Sampling profiler
- User testing framework — **superseded by #142 landing in v1.2**
- Documentation generator
- Profile-guided optimisation (PGO) + LTO

### Stdlib — Noesis M2 items that are pure stdlib (no new substrate)
| # | Item | Effort |
|---|---|---|
| #174 | SRFI-41 lazy streams | 1-2 days |
| #176 | Unicode `string-normalize`, TOML, YAML, URL parse/encode | 5-7 days |
| #177 | SQLite migrations stdlib | 2 days |

**Exit criterion for v1.3**: Noesis M1 single-agent production surface usable
(minus HTTP server and networking — those come in v1.4). All R7RS polish
items shipped. Debugger + profiler available for bench work.

---

## v1.4 — "connection" (July 2026)

Networking + concurrency. This is the **biggest release since v1.1** because
it establishes the substrate both M1 production HTTP and M3 concurrent
faculty evaluation depend on.

### Networking (M1 + M4 substrate)
| # | Item | Effort | Noesis tier |
|---|---|---|---|
| #145 | HTTP server (build on `inc/eshkol/http_request_utils.h`) | 1 week | M1 |
| #146 | WebSocket server | 3 days | M1 |
| #148 | Prometheus metrics primitive + `/metrics` endpoint | 2 days | M1 |
| #150 | Resource limits (CPU / memory / wall-time) | 2 days | M1 |
| #161 | TCP / UDP sockets | 1 week | M4 |
| — | TLS server (OpenSSL / mbedTLS wrap) | 3 days | M4 |
| — | Incremental compilation | 1 week | — |

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

### Wire formats
| # | Item | Effort | Noesis tier |
|---|---|---|---|
| #162 | MessagePack | 3 days | M4 |
| #163 | Protocol Buffers (proto3) | 3 days | M4 |
| #175 | Content-addressable storage + Merkle trees | 1-2 days | M2 |

### Linear types (original v1.4 item)
- Linear resource types (`ESHKOL_VALUE_SOCKET`) — compile-time single-ownership
  enforcement on handles. Complements networking — once sockets exist, they
  should be linear so they can't leak.

**Exit criterion for v1.4**: Hiereia deployable as production HTTP agent with
async I/O under 10K concurrent connections, /metrics scraped, log stream
accessible via WebSocket, resource-bounded requests. Ecumene wire formats
and socket primitives available (the Ecumene faculty layer itself is
separate — the substrate is ready).

**Big release** — tag this `v1.4.0`, not a point release.

---

## v1.5 — "intelligence" (August 2026)

Neuro-symbolic bridge. Unblocks Noesis M2 (Mneme at scale).

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
| M0 research-grade | 11 items | — | — | — | — |
| M1 Hiereia prod | — | 8 items | 4 items | — | — |
| M2 Mneme scale | — | 3 items | 1 item | 3 items | — |
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

Noesis M0 ships with v1.2.x. M1 = v1.3 stdlib + v1.4 networking. M2 = v1.5.
M3 = v1.4 concurrency. M4 = v1.4 wire formats + v1.8 distributed + v2.0 BFT.

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
| #155 | Priority queues / sets / deques | M1 | v1.3 |
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
| #170 | Reflection (procedure-arity, etc.) | M1 | v1.3 |
| #171 | LRU cache | M1 | v1.2.x ✅ |
| #172 | JSON Schema validation | M1 | v1.2.x ✅ |
| #173 | PRNG seeding + deterministic replay | M1 | v1.2.x ✅ |
| #174 | SRFI-41 streams | M2 stdlib | v1.2.x ✅ |
| #175 | CAS + Merkle trees | M2 | v1.4 |
| #176 | Unicode NFC/NFD + TOML + YAML + URL | M2 stdlib | v1.3 |
| #177 | SQLite migrations | M2 | v1.3 |
| #134 | Compile-to-binary eval linker | — | v1.3 |

---

## Disagreements with the audit (and why)

Two items where my scheduling differs from the audit's ordering:

1. **Audit puts AD ops (#154) and pqueues (#155) in M1 / v1.3; my original
   trajectory put them in M2 / v1.5.** Audit wins — pqueues unblock workspace
   scheduling (immediate faculty need) and AD ops unblock Sigma/Aletheia
   paper benchmarks. Rescheduled to v1.3.
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
