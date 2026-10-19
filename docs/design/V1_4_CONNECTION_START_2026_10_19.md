---
kind: project
status: current
owner-area: runtime
since: v1.4.0-connection
sources:
  - docs/design/adr/0000-unified-trajectory.md
  - docs/design/adr/0004-type-system-trajectory.md
  - inc/eshkol/core/handle.h
  - lib/core/runtime_handle.cpp
  - tests/core/handle_lifecycle_test.cpp
---

# v1.4-connection development start

Starting point: released `v1.3.6-evolve`, source
`341ac812ed600fd9d27c28e9509d66ab71363947`. This plan records the first
development slice; it does not declare the release ready or set a release date.

The existing scope comes from [the compiler roadmap](../COMPILER_ROADMAP.md)
and [ADR 0000, Stage 3](adr/0000-unified-trajectory.md#stage-3--v140-resource-sound-systems-profile).
Connection combines the hosted networking surface with checked resource
ownership. ADR 0000 deliberately puts the memory ABI change and portable tail
transfer in **v1.4.1**, after the **v1.4.0** resource-sound systems profile.
Their separate gates remain in force.

## Starting inventory

| Area | Released baseline | Remaining acceptance |
| --- | --- | --- |
| Portable event loop | `inc/eshkol/core/event_loop.h`, portable registry and kqueue/epoll/Windows backends; native and VM fixtures exist | Integrate resource ownership and prove concurrent hosted workloads; existence of the API does not meet the 10K-connection gate |
| Semantic identity | Shared `NodeId`, `BindingId`, interned type terms and typed-expression side table | Ownership-aware flow analysis, region/loan rules, deterministic cleanup and effect checking |
| External-resource handles | `ESHKOL_VALUE_HANDLE` and reserved file/socket subtype identifiers | Shared runtime lifecycle authority, then consumer integration and static enforcement |
| Modules and proper tail calls | Stage 2 resolver substrate is recorded as attained | Stage 3 strict visibility, shared tail annotation/verifier, and reproducible locked resolution |
| Networking and observability | HTTP, socket, metrics and resource-limit implementations already exist | Audit each roadmap contract with executable native/VM/package evidence before crediting completion |
| Release readiness | `.icc/completion-oracles.yaml` defines `v1.4-connection` (alias `v1.4-release`), backed by `scripts/run_v14_connection_gate.sh` | Establish fresh source-bound behavioral evidence, audit executable versus stub criteria, and distinguish v1.4.1-stage criteria from v1.4.0 gates |

The old local swarm tickets are working notes, not current master contracts.
The workplan's claim that the connection oracle is still a placeholder is
also stale: the tracked target already has separate deliverable criteria.
In particular, ESH-0011's event-loop implementation already exists, and
ESH-0010's proposal to defer static resource safety to v1.9 conflicts with the
accepted v1.4.0 gate. A warning-only lifecycle check cannot satisfy that gate.

## First implementation: managed-resource lifecycle substrate

Introduce a C-compatible hosted registry in `inc/eshkol/core/handle.h` and
`lib/core/runtime_handle.cpp`. Its ownership records live outside evacuating
arenas. Generation-tagged IDs reject a prior owner's stale identity when a
slot is reused. File, socket and event-loop resource kinds share one lifecycle:
open, exclusively borrowed, closing and closed. Illegal transitions return
errors before accessing or closing the resource.

Native close callbacks run outside the registry lock. An attempted close is
terminal even when its callback reports failure, because retrying an uncertain
OS close can target a reused descriptor. Destroy refuses live borrows or
in-progress closes. Consumers must attach their source spans to returned errors.

Acceptance for this slice:

- A real native resource is borrowed, released and closed exactly once.
- Closing or destroying while borrowed fails without closing the resource.
- Double close, use after close, invalid IDs and stale IDs cannot reach a native
  resource; a failed close callback is never retried.
- Reentrant callbacks and concurrent lifecycle operations do not deadlock or
  double close. The C header compiles as C, and focused sanitizer checks pass.

This is an additive runtime substrate. It does not yet change Scheme ports,
socket/event-loop public APIs or tagged-value layout, implement a typechecker,
or prove statically checked cleanup. CTest covers it independently of LLVM.

Focused validation on macOS arm64: `handle_lifecycle_test` and
`handle_c_api_test` pass through CTest; the lifecycle fixture also passes with
`-DNDEBUG`, ASan/UBSan and TSan. ASan leak detection is disabled because the
host does not support it. The hosted-runtime object compiles with LLVM 21, and
the existing runtime source-boundary check passes. Full compiler, Linux,
Windows, VM and packaged-release qualification remain later checks; these
focused tests do not substitute for those gates.

## Next dependency order

1. Integrate one real file/socket consumer with this lifecycle authority and
   define source-span diagnostics and exception/continuation cleanup contracts.
2. Add ownership facts to the shared typed HIR and flow analysis; reject known
   double consumption, invalid loans and escaping region-local resources.
3. Connect deterministic cleanup edges to normal returns, exceptions and
   cancellation; test native, VM and REPL behavior together.
4. Complete the hosted HTTP/TLS/concurrency contracts and their executable
   resource-safe 10K-connection gate, visibility and module-resolution gates.
5. Qualify the resulting compiler and packaged SDKs against real v1.4 oracles.

The existing oracle also records distribution work pending maintainer ruling
R7. This first slice does not settle that scope question or treat its spikes as
completed. The old calendar estimates are historical planning, not a current
release ETA.

## Carryover from v1.3.6

The branch includes local cleanup fix
`b773d1721cdce4ffba00c902093ff87a8722c87d`: the AOT driver exclusively reserves
and removes its own scratch objects, preserving caller and package objects.
Its native cleanup and shared-library ABI regressions passed during the
post-release investigation. It is **not in the shipped v1.3.6 packages**.

Refusing an incompatible fallback stdlib remains separate work. The current
stdlib object/bitcode has no authenticated release identity; an adjacent runtime
version is insufficient evidence of that artifact's identity. Implement and
test an explicit compatibility policy before claiming this part fixed.

Richard's 46 open PRs remain an integration queue. They need dependency-aware
review against the selected release scope; none is implicitly accepted or
merged by this kickoff. The published v1.3.6 tag and proof archives stay intact.
