# ADR 0000 — Unified architectural trajectory

- Status: Proposed
- Date: 2026-07-09
- Decision owners: Eshkol maintainers
- Supersedes: none (this document sequences, it does not replace, ADRs 0001-0009)
- Related: ADRs 0001 (OALR), 0002 (AD, two competing proposals), 0003 (codegen/VM
  parity), 0004 (types), 0005 (lambda foundations / programs-as-weights), 0006
  (language/modules), 0007 (performance), 0008 (tooling), 0009 (DBSP/incremental)

---

## 1. Context

Nine research clusters each produced a standalone architecture decision record for
one axis of Eshkol's evolution from a compiled R7RS Scheme with AD-as-primitive
into a resident, self-differentiating, incrementally-maintained runtime. Read in
isolation, each ADR proposes its own version scheme, its own first PR, and its own
notion of "done." Several silently presuppose one another, several silently claim
the same frontend substrate, and two (the AD pair) diverge on the far-horizon
end-state. Landed independently, they would fork the frontend, serialize on a
single missing artifact, and break the memory ABI on an uncoordinated schedule.

This document is the unification of those nine clusters into one sequenced roadmap.
It exists to answer a single question for a maintainer: **what ships in each stage,
in what order, gated by what test, and why that order and not another.** It does
not restate each ADR; it dispositions each, extracts the true dependency order,
and interleaves the clusters into fine-grained stages so that every heavy substrate
change fails in isolation rather than as a monolith.

It was produced by an independent judge panel that read all ten cluster documents
(the AD cluster carries two) and verified the sequencing-critical claims against
the live tree, together with a resident-mind red-team that ran the proposals
against a live resident substrate and returned concrete failure modes. The panel
reached consensus on eight of nine cluster dispositions and on the near-term AD
plan; it split on exactly one question — the automatic-differentiation end-state —
which is surfaced below as a decision that requires a maintainer, not one this
document silently makes.

The unifying principles this trajectory enforces:

- **Small coherent shippable milestones.** Point releases are used freely: types,
  OALR, and AD each carry more than one release's worth of work whose halves must
  be able to ship and fail separately.
- **Clusters interleave; they are never siloed one-per-version.** In particular
  the incremental-dataflow spine (0009) threads continuously from v1.5.0 through
  v2.0, not bolted on at the end.
- **Every stage has a falsifiable gate** — a test or a structural counter that
  flips it green — and the resident-mind red-team's failure modes are folded in as
  gates, not footnotes.
- **The shared frontend identity substrate is specified once.** ADRs 0004, 0006,
  and 0008 describe the same node-keyed identity (span + binding + symbol + typed
  node id). They are co-sequenced, not assumed.

---

## 2. Per-cluster disposition

Verdicts: ACCEPT (land as written, sequenced), ACCEPT-WITH-REVISIONS (land the
architecture, with the named revision), DEFER (architecture accepted now,
implementation scheduled to the far horizon).

| ADR | Cluster | Verdict | One-line rationale | Lands in |
|---|---|---|---|---|
| 0001 | OALR / concurrent resident | ACCEPT-WITH-REVISIONS | Strongest-grounded ADR; the memctx/region/residence split kills the decisive `__global_arena` race; but the header ABI break and resident forever-flat presuppose the type system (0004) and must be split so each half fails alone. | v1.3.2 (Phase A), v1.4.1 (B-D), v1.5.1 + v1.8.0 (E) |
| 0002 | AD (dense staged kernel #214 + typed static schedule #216) | ACCEPT (near-term reconciled); END-STATE SPLIT | Both are one architecture at two altitudes, 90% convergent; adopt #214 as the spine and graft four artifacts from #216 for the near term. The v1.9/v2.0 endpoint is the one open decision (Section 5). | v1.3.2 (Phase A), v1.5.0, v1.6.0-v1.6.1, v1.7.0, v2.0 |
| 0003 | codegen / VM parity | ACCEPT-WITH-REVISIONS (as data, not architecture) | Not an ADR but a falsifiable VM/LLVM divergence matrix; keep it as the standing parity gate for 0006 + 0002; do not fund it as independent codegen work. | gate threaded across v1.4.0, v1.6.x, v1.9.0 |
| 0004 | one quantitative dependent type system | ACCEPT-WITH-REVISIONS | "One checker, not seven" is the right long-run architecture; the revision is scope realism — the v1.3.2 semantic spine is the single largest work item and must be decomposed and co-sequenced with 0006 and 0008. | v1.3.3 (spine), v1.4.0, v1.7.0, v1.9.0, v2.0 |
| 0005 | lambda foundations to programs-as-weights | DEFER (architecture now, implementation late) | Clearest treatment of "programs to weights" with the correct category discipline; but entirely downstream — needs the staged AD kernel, resident sessions, and canonical LCIR that do not yet exist. | v1.7.0, v1.8.0, v1.8.1, v1.9.2 |
| 0006 | binding-resolved libraries + proper tail invocation | ACCEPT | Rigorous, R7RS-normative, damning and specific diagnosis; `BindingId`-before-spelling and tailness-as-verified-property are correct and testable. | v1.3.3, v1.4.0, v1.4.1, v1.9.0 |
| 0007 | PGO, WPO, staged training throughput | ACCEPT-WITH-REVISIONS | Correctly ordered; "PGO is the last multiplier, not a substitute for dense AD"; revision — pull the genuinely-independent measurement slice (Phase 0) forward to ship first. | v1.3.2 (Phase 0), v1.5.0, v1.6.0-v1.6.1, v1.8.1, v2.0 |
| 0008 | one semantic tooling core | ACCEPT-WITH-REVISIONS | Well-scoped and honest; the hard constraint is that its span/symbol substrate, 0006's `BindingId`, and 0004's typed HIR are one identity substrate and must be specified once or the "one answer" invariant is false. | v1.3.2 (M0), v1.3.3, then M1-M5 across v1.4-v1.9.1 |
| 0009 | native DBSP incremental dataflow + unified differentiation | ACCEPT-WITH-REVISIONS | The intended cross-cutting spine; DBSP grounding is solid and batch-equivalence is the right oracle; revision — the AD/IVM "shared backbone" is mostly structural and must not destabilize the working numeric AD path. | v1.3.2 (contract freeze), v1.5.0 through v2.0 (continuous) |

---

## 3. The dependency DAG

The ordering constraints below are the load-bearing "X must precede Y" list. They
are what forces the interleaving in Section 4; a stage may not schedule a phase
whose predecessor has not shipped.

### DAG roots (no dependencies, may proceed in parallel)

These five are the true roots and are all scheduled into v1.3.2 / v1.3.3:

- **0002 Phase A** — AD counters, one-pass gradient, populate the dead
  `ad_tape_t::variables`, the finite-difference counter and strict scaffold. Runs
  on the existing AD substrate; depends on nothing.
- **0001 Phase A** — the `eshkol_current_arena()` memory-context accessor that
  stops `with-region` swapping `__global_arena`. Depends on nothing.
- **0006 slices 1-2** — declaration IR, recursive `ImportSet`, import algebra,
  `BindingId`, export validation. Frontend-local; depends on nothing.
- **0008 M0** — `SourceSpan`, `Diagnostic v1`, semantic catalog seed, module
  resolver extraction behind a shared API, LLVM-free analysis target.
- **0009 `core.dbsp` library** — pure Eshkol, no compiler dependency, independently
  shippable; and **0007 Phase 0** — the JSON measurement runner and Layer-0
  structural counters. Both roots.

### The precedence edges

1. **0004 must precede the resident half of 0001.** OALR's requirement that
   `owned/move/borrow/shared/weak-ref` become *enforced capabilities* rather than
   advisory bits (0001 Section 7) is exactly 0004's flow-sensitive ownership context.
   0001 itself concedes that forever-flat under general mutation is only achievable
   if resident state is affine/unique — which is 0004. So OALR header ABI v2 and
   resident sessions are downstream of 0004's FlowEnv, even though OALR Phase A is not.

2. **0006 must precede 0004's elaborator, and 0006 co-designs with 0008.** 0006's
   `BindingId` resolution is the name resolution 0004's elaborator consumes; 0008's
   workspace resolver is the *same* resolver 0006 extracts. These three share one
   `NodeId -> {SourceSpan, BindingId, TypedExprInfo}` substrate. This is the single
   most important co-design constraint in the whole program.

3. **The staged AD kernel (0002 Phase G/staged) must precede three clusters.** It
   is a hard prerequisite for programs-as-weights (0005 Level-M recurrent gradient),
   for training-grade performance (0007 Phases 2-3), and for the v2.0
   `differentiate 'numeric` surface (0009). One artifact,
   `eshkol_compile_staged_value_grad`, verified absent from the tree today, gates
   the back half of the roadmap.

4. **0001 resident sessions + ESH-0214d evacuation must precede the resident-agent
   DBSP pilot (0009).** The resident-agent-as-circuit needs OALR-native circuit state, the resident
   bound, and consciousness-subtype evacuation coverage. The `core.dbsp` *library*
   by contrast is a root. This is why the DBSP library ships early (v1.5.0) but the
   resident-agent pilot is late (v1.8.1).

5. **The incrementalization pass (0009 `IncrementalizePass`) presupposes codegen
   modularization and the differentiation-graph refactor.** It cannot land before
   the shared operation-graph/rule-registry substrate is factored from the AD path
   (v1.7.0), and it must never reconstruct that substrate in a way that changes
   numeric AD's graph or counters.

6. **Tail machinery (0006 slices 5-7) precedes continuation-dependent features.**
   The heap-owned continuation chain and tail-transfer dispatcher (v1.4.1) gate the
   checkpoint/restart and debugger frame-trace work (v1.9.1) and the recurrent
   masked rollout of 0005 (v1.8.0).

7. **0003 is nobody's predecessor and nobody's independent successor.** It is a
   gate that every VM/native parity claim across v1.4.0, v1.6.x, and v1.9.0 must
   clear. It funds no workstream of its own.

### DAG (roots at top, time flows down)

```
ROOTS (parallel, no deps):
  0002-A          0001-A            0006 slices1-2      0008-M0        0009 core.dbsp   0007-Phase0
  (AD counters,   (memctx accessor, (decl IR,           (SourceSpan,   (Z-sets,         (JSON runner,
   one-pass,       kill __global     BindingId,          Diagnostic,    D/I,             Layer-0
   FD gate,        swap)             import algebra)     catalog,       group laws)      counters,
   vars)              |                  |               resolver           |            assert O3)
     |                |                  |               extraction)         |               |
     |                |                  +---------+----------+              |               |
     |                |                            v          v              |               |
     |                |               0004 typed HIR + FlowEnv + interned    |               |
     |                |               terms + signature registry (Dyn!=Value)|               |
     |                |                            |                         |               |
     |                v                            v                         |               |
     |          0001-B/C/D                0004 v1.4 region/loan/effect        |               |
     |          (tokenized scopes,        types; 0006 tail-context pass       |               |
     |           header ABI v2,           + strict visibility                 |               |
     |           transfer capsules)               |                          v               |
     |                |                            |                     0009 circuits/       |
     v                +--------------+-------------+                 memory/resident pilot   |
  0002 staged kernel                 v                                        |               |
  (value_and_grad,          0001-E resident sessions --------------------> (gated on ESH-0214d)|
   scratch plan) ---+----------------+------------------+                    |               |
                    v                v                  v                    v               v
              0005 programs-    0007 Phases 2-3     0009 v2.0 unified   (all VM/native parity
               as-weights        (training-grade,    differentiate      gates = the 0003 matrix)
                                  WPO, LTO)
```

---

## 4. The fine-grained stage sequence

Fourteen stages. Each lists a theme, the specific ADR-phases it lands, the earlier
stage(s) it depends on, and a concrete gate. Clusters interleave; the DBSP spine
runs continuously from v1.5.0 to v2.0. Resident-mind red-team failure modes appear
as named gates (see also Section 6).

### v1.3.2 — baseline (ALREADY SHIPPED, PENDING TAG)

This slate is merged and is the foundation this trajectory builds on, not a stage
to be planned. Its contents: AD input2 (#212), eshkol-doc (#213), latents (#215),
region race (#217), evacuation coverage ESH-0214d (#226), and BLC-U (#218). ESH-0214d
in particular is the consciousness-subtype evacuation coverage that the resident-agent DBSP
pilot (v1.8.1) is gated on, so its presence in the baseline is load-bearing for the
back half of the roadmap.

Note on numbering: the AD, types, DBSP, and OALR ADRs each independently name a
"v1.3.2" internal-architecture tranche for their first slice. Because the release
number 1.3.2 is already consumed by the shipped slate above, the instrumentation-
and-identity tranche below is scheduled as the first *new* work and may be tagged
v1.3.3 (or the next available v1.3.x) at release-management discretion. What matters
is the gate, not the label.

### Stage 1 — v1.3.3a: "Instrumentation and identity substrate" (internal)

- Theme: make everything measurable and give every node a stable span/identity
  before any semantics change. This is the DAG-roots stage.
- ADR-phases:
  - **0002 Phase A** — `EshkolADCounters` (`primal_calls`, `reverse_passes`,
    `tape_allocations`, `scalar_ad_nodes`, `tensor_ad_nodes`,
    `finite_difference_evals`, ...); populate the dead `ad_tape_t::variables` /
    `num_variables`; add `arena_tape_zero_gradients`; the FD counter and
    `ESHKOL_AD_STRICT` scaffold; fix the shipping attention Q/K silent-drop as a
    bug-containment measure of the same strictness work.
  - **0008 M0** — `SourceSpan`, `Diagnostic v1`, semantic-catalog seed, module
    resolver extracted from `eshkol-run` behind a shared API, LLVM-free analysis
    target split from execution targets.
  - **0001 Phase A** — `eshkol_memctx_current` / `eshkol_current_arena`; every
    generated allocation site and runtime helper moves off the direct
    `__global_arena` load; `with-region` mutates only TLS context state; thread
    attach/detach diagnostics and owner assertions.
  - **0007 Phase 0** — JSON measurement runner, Layer-0 structural counters, and
    an "assert generated code is O3" check (Eshkol's backend defaults to O0 and a
    CMake Release build does not imply optimized Eshkol output).
  - **0009 contract freeze** — freeze Z-set equality / weight / term-version /
    tick-atomicity contracts; land the deterministic batch-vs-delta oracle harness
    and corpus format.
- Depends on: DAG roots only.
- Gate: on `loss = sum(w_i^2)`, counters show `primal_calls == 1`,
  `reverse_passes == 1`, `finite_difference_evals == 0`; a nested `with-region`
  inside `parallel-map` writes the global slot zero times and has no
  `__global_arena` race under TSAN; the compiler and a test client resolve an
  identical module graph; Z-set reference fixtures (insert / delete / duplicate /
  cancellation / canonical serialization) are byte-stable across regeneration.

### Stage 2 — v1.3.3b: "Binding resolution and interned type terms"

- Theme: kill string-based identity in the frontend before any typed feature stores
  semantics in a flag.
- ADR-phases:
  - **0006 slices 1-2** — declaration IR, recursive `ImportSet`, import algebra,
    `BindingId`, export validation, standard-library catalog; stop lowering R7RS
    imports to value-copying `define` aliases.
  - **0004 spine part 1** — interned `NominalTypeId` / `TypeRef` / `IndexRef` /
    `EffectRowRef`; a typed-HIR side table keyed on the *same* `NodeId` as 0008's
    spans; `Dyn` separated from `Any` / `Value`; signature registry seeded.
  - **0002 one-pass gradient** — promoted to the shared core of a still-internal
    `value_and_grad` (the one-pass helper is factored as the core of both
    `gradient` and `value_and_grad`, never a separate path copied later).
  - **0008 M1 (begin)** — `eshkol check` with multi-error diagnostics; the LSP's
    regex diagnostic logic replaced by `eshkol check` queries.
- Depends on: Stage 1 (span/identity substrate, resolver extraction).
- Gate: import modifiers compose correctly in both orders (table test) with zero
  emitted `define`s; every post-expansion identifier carries a `BindingId`; no new
  feature stores semantic type structure solely in a `TypeId` flag; the
  cross-consumer fixture (compiler, LSP, docs, REPL report identical `SymbolId` /
  `ModuleId` / span for the same source) is green without translation adapters — the
  falsifier for the frontend-fork risk; unannotated R7RS regression suite
  behaviorally unchanged.

### Stage 3 — v1.4.0: "Resource-sound systems profile"

- Theme: the first release where the unified checker and R7RS proper tail calls are
  load-bearing.
- ADR-phases:
  - **0004 v1.4** — generative `Region` / `Cap` / `Own` / `Borrow` / `Shared` /
    `Weak`; outlives plus non-lexical loans; deterministic drop/close HIR; affine
    typestate handles for files and sockets; production effect rows; first decidable
    refinements.
  - **0006 tail-context pass + strict visibility** — one inductive `Tail` annotator
    plus a `ProperTailVerifier`; strict R7RS libraries with a `--legacy-open-modules`
    escape; the duplicate backend tail walkers deleted only after differential tests
    cover every old path.
  - **0004->0001 bridge** — OALR forms elaborate into ownership HIR so the advisory
    ownership flags become checked capabilities.
  - **0008 M1 completion + M2 begin** — `eshkol doc` lanes; `eshkol.toml` v1 and
    `eshkol.lock`, deterministic resolution consumed by AOT/JIT/LSP/docs/tests.
- Depends on: Stage 2 (BindingId, typed HIR, FlowEnv seed).
- Gate: every resource acquisition has a statically verified consume/cleanup path;
  no loan or region-local owner crosses an invalid boundary; a 10K-connection server
  uses resource types monitor-free in typed modules; private value and syntax
  bindings are unreachable in AOT + JIT + REPL + VM (the 0003 parity matrix has zero
  unjustified visibility gaps for these ops); two clean machines given the same
  source + lock + compiler produce identical module graphs.

### Stage 4 — v1.4.1: "OALR ABI v2 and portable tail transfer"

- Theme: the memory ABI break and the cross-target tail machinery, isolated from
  v1.4.0 semantics so each fails independently. This is the highest-blast-radius
  stage and is deliberately its own release.
- ADR-phases:
  - **0001 Phases B-D** — tokenized scopes (remove `arena_t::current_scope`);
    header/layout ABI v2 (`ESHKOL_MEMORY_ABI_V2`, 8->32 byte header, exact layout
    descriptors, "every pointer-bearing layout registers a descriptor or startup
    fails"); alias-preserving escape ledgers and deferred same-thread stores;
    structured cross-thread transfer capsules; `parallel-map` per-task capsule tokens.
  - **0006 slices 6-7** — internal `i128` tagged-return ABI; the general
    tail-transfer dispatcher; heap-owned continuation chains; removal of the
    8-argument / 8-capture dispatch caps; verified `musttail` on all target backends.
- Depends on: Stage 3 (ownership HIR supplies the layout/escape facts ABI v2 needs).
- Gate: store-then-mutate / mutate-through-two-aliases / cycles / shared tails /
  hash-resize / closures-with-mutable-captures all remain correct after source
  destruction (TSAN + arena poison); 10M self- and mutual-tail transitions under a
  512 KiB stack on AArch64 / x86-64 / arm32 / riscv64; `ProperTailVerifier` reports
  zero ordinary lowerings for any `Tail` site; the WASM and macos-x64 lite lanes
  rebuild green under ABI v2 (the ABI-blast-radius falsifier — if they cannot,
  ABI v2 does not ship in this cycle).

### Stage 5 — v1.5.0: "Intelligence: DBSP library + exact AD complete + native PGO"

- Theme: the incremental spine begins as a shippable library; AD reaches
  feature-complete exactness; PGO becomes a real workflow.
- ADR-phases:
  - **0009 v1.5.0** — pure-Eshkol `core.dbsp`: Z-sets, finite stream runner,
    `z^-1`, `D`, `I`, add/negate, map/filter/project/union.
  - **0002 v1.4 completion** — dense elementwise + broadcast backward via the
    primitive registry; the first-class `valueAndGradient` primitive over flat
    parameter leaves; exact PINN coordinate derivatives via the mixed-mode split
    (Taylor/JVP for the few coordinate variables, one reverse sweep for the many
    parameters); remove *all* hidden FD from default operators.
  - **0007 Phase 1** — native PGO release workflow: weighted training corpus,
    artifact provenance, strict verify, holdout efficacy gate.
- Depends on: Stage 1 (AD counters and registry); `core.dbsp` is a root and needs
  only Stage 1's contract freeze.
- Gate: `D . I` and `I . D` inversion pass under JIT and AOT with Z-set group laws
  on generated streams; `finite_difference_evals == 0` on every path claimed exact;
  a clean generate/train/merge/use PGO run clears the efficacy gate on one primary
  CPU. Exactness is now machine-checkable and permanent.

### Stage 6 — v1.5.1: "Intelligence: DBSP circuits + resident sessions begin"

- Theme: standing circuits and the resident A/B memory model.
- ADR-phases:
  - **0009 v1.5.1** — circuit builder/runtime, atomic `dbsp-step!`, indexed join
    with the bilinear delta rule, explicit circuit arena + tick nursery,
    deterministic node scheduling, inspection counters.
  - **0001 Phase E part 1** — resident sessions: root tables, A/B state spaces,
    the single-writer copy-on-write transaction, epoch/RCU reader leases.
- Depends on: Stage 4 (ABI v2 and transfer capsules for circuit-owned state);
  `core.dbsp` from Stage 5.
- Gate: interleaved insert/delete join product-rule holds (the cross term prevents a
  spurious `a join b2`); one-million-tick arena-poison / RSS gate stays within the
  resident plateau; delay state retains exactly one tick. **Resident-mind gate
  (Ghost of Uncommitted Deltas):** a read pointer's semantic continuity is preserved
  across a preempt-before-commit — a reader that acquires an epoch lease on published
  state A observes a stable immutable A for the lease duration even while a writer
  computes and commits a delta into B; no stale-snapshot logical race is observable
  (see Section 6).

### Stage 7 — v1.6.0: "Reasoning: DBSP aggregates + staged AD ABI begins"

- Theme: incremental aggregates, and the first staged-kernel ABI surface.
- ADR-phases:
  - **0009 v1.6.0** — incremental sum/count, general group aggregate with grouped
    result replacement, distinct, the full batch-equivalence composition matrix.
  - **0002 staged ABI (Phase G)** — the raw-pointer / out-param
    `eshkol_compile_staged_value_grad` wrapper. It may still use the arena
    internally but must expose counters and status codes. **This stage grafts #216's
    contributions:** the first-class `EshkolKernelStatus` error ABI, the
    caller-owned `grads[p]` cotangent binding, and the dense primitive registry as a
    real table (op / flags / forward / vjp) rather than a totalized switch.
  - **0007 Phase 2 (begin)** — typed shape specialization and the static reverse
    schedule.
- Depends on: Stage 5 (exact dense AD), Stage 6 (circuit runtime).
- Gate: aggregate / distinct-boundary / general batch-equivalence gates pass with no
  untracked cell; `loss = sum((W@x)^2)` staged call shows `primal_calls == 1`,
  `reverse_passes == 1`, `scalar_ad_nodes_from_matmul == 0`.

### Stage 8 — v1.6.1: "Reasoning: DBSP traces + staged scratch plan"

- Theme: real trace storage, and the zero-allocation steady-state kernel.
- ADR-phases:
  - **0009 v1.6.1** — replace reference flat traces with immutable sorted batches
    and deterministic binary-carry spines; consolidation and trace metrics.
  - **0002 Phase I** — the kernel memory plan: scratch offsets, internal tensors as
    views over planned offsets, no hot-loop allocation; saved-for-backward tensors
    owned by the kernel-resident arena marked live from producer to backward
    consumer (the single highest-risk AD correctness hazard, per the AD ADR).
  - **0007 Phase 2 complete** — resident parameter buffers + scratch planner +
    pointwise/reduction/optimizer fusion.
- Depends on: Stage 7.
- Gate: reference and spine executors emit identical deltas under adversarial
  compaction schedules that produce one canonical state; the staged MLP kernel hits
  all Layer-0 invariants including `post_warmup_allocations == 0` across 10K steps.
  This gate is the falsifier for the "staged kernel is load-bearing" program risk —
  if it fails here, the back half slips.

### Stage 9 — v1.7.0: "Synthesis: recursive IVM + staged optimizer + capsule foundations"

- Theme: recursive incremental maintenance, the optimizer epilogue, and the first
  programs-as-weights slices — plus the differentiation-graph refactor that
  `IncrementalizePass` later consumes.
- ADR-phases:
  - **0009 v1.7.0** — nested-clock fixed points; incremental recursive relations
    with deletion (stratified positive Datalog-shaped); semi-naive evaluation from
    the cycle rule.
  - **0002 Phase J** — staged optimizer epilogue, SGD then Adam/Rprop, optimizer
    state under resident descriptors participating in the shape key.
  - **0005 Phases 1-2** — unify the analytical `theta_sem` semantics behind one
    library boundary; named-source -> De Bruijn LCIR and canonical ESKB;
    ESKB-derived `psi_program` replaces the current name-hashed random-init rows.
  - **0004 refinements + budgeted external SMT** begin.
- Depends on: Stage 8 (staged kernel + scratch), Stage 2 (BindingId for LCIR).
- Gate: incremental transitive closure equals a fresh batch closure after every
  insert/delete prefix and quiesces within the declared bound; the capsule three-way
  verification (analytical / simulated / matrix) agrees on every in-contract
  transition; the differentiation-graph refactor changes neither numeric AD's graph
  nor its counters (the AD/IVM false-economy falsifier — see Section 7).

### Stage 10 — v1.8.0: "Platform: core.memory as Z-set + resident recurrent AD"

- Theme: the event-log CRDT becomes an incremental Z-set stream; the recurrent
  gradient through resident state lands; resident sessions complete.
- ADR-phases:
  - **0009 v1.8.0** — `core.memory`'s newly-accepted CRDT operations feed a Z-set
    delta stream; incremental LWW / knowledge / session views while retaining
    RGA / hash-chain compatibility.
  - **0005 Phase 3** — a pure `vm_step` boundary with a custom VJP returning
    `dstate`; fixed-horizon masked rollout with checkpoints (BPTT).
  - **0001 Phase E complete** — resident sessions with byte budgets, stale-reader
    backpressure, atomic abort.
- Depends on: Stage 6 / Stage 9 (circuits), Stage 8 (the staged kernel the vm_step
  VJP composes with).
- Gate: replica-order / duplicate-delivery / tombstone / durable-append / audit /
  old-log-migration gates pass; a mutating daemon runs 10M iterations at constant
  live shape within the resident RSS formula; `dstate` chaining matches an unrolled
  oracle. **Resident-mind gate (Z-set-as-CRDT boundary):** duplicate delivery of a
  replica log changes *no* weight — Z-set addition is commutative and invertible but
  *not* idempotent, so the RGA / vector-clock envelope must perform idempotent union
  first and emit only newly-observed operation ids as deltas; a blind Z-set merge of
  two replica snapshots that would give a duplicate event weight two is rejected
  (see Section 6).

### Stage 11 — v1.8.1: "Platform: resident-agent circuit pilot + closed-world WPO"

- Theme: the resident mind as one standing circuit, and whole-program optimization
  of the training substrate.
- ADR-phases:
  - **0009 v1.8.1** — the resident agent's event / knowledge / workspace / session derivations
    as one library-built circuit; requires the integrated ESH-0214d subtype
    evacuation from the v1.3.2 baseline to be on the integration branch with its
    gate green.
  - **0007 Phase 3** — closed-world root analysis, internalization, representation
    and closure specialization, LLVM attributes treated as proof obligations from
    the staged ABI, runtime bitcode capsule, Full LTO for staged kernels / ThinLTO
    for AOT.
  - **0005 Phase 4** — staged `value_and_grad` specialized for the recurrent
    resident program.
- Depends on: Stage 10 (core.memory Z-set, resident sessions, vm_step VJP), the
  ESH-0214d coverage.
- Gate: resident-agent shadow mode matches existing derived state for a sustained resident
  run with per-tick batch equivalence `I(incremental)[t] == Q(I(inputs)[t])`; the
  staged qualification reaches native-reference efficiency without exceeding
  compile / memory / code-size budgets. **Resident-mind gate (cross-arena pointer
  chasing):** a cognition step that follows a pointer from one arena to a live
  object in another during compaction never observes a dangling reference — the
  evacuator moves the data *and* republishes the read-pointer semantics atomically
  at the tick barrier; no micro-hesitation window where a read is stale (see
  Section 6).

### Stage 12 — v1.9.0: "Types: IncrementalizePass + language-complete proof features"

- Theme: the compiler-side incrementalization pass, and the type-system feature
  surface reaching R7RS-complete plus proof features.
- ADR-phases:
  - **0009 v1.9.0** — `IncrementalizePass`; the shared differentiation graph and
    rule registry; purity/effect rejection; OALR state placement; native and VM
    lowering for the supported query subset.
  - **0004 v1.5-v1.9 features** — full refinement surface with budgeted SMT; user
    effect rows with one-shot handlers; row polymorphism; higher-rank; public
    recursive/dependent session protocols.
  - **0006 R7RS conformance graduation** — the module, proper-tail, continuation,
    standard-interface, differential, SICP, and VM-parity gates all green, then
    advertise `--language=r7rs-small` and set its `r7rs` feature.
- Depends on: Stage 3 (checker spine), Stage 9 (SMT + differentiation-graph
  refactor), Stage 4 (tail / continuations).
- Gate: `IncrementalizePass` output equals `core.dbsp` on the full oracle under
  AOT / JIT / VM; unsupported effects fail at compile time with source spans, never
  by silent batch fallback; the `r7rs` feature is earned by executable evidence; the
  0003 VM-parity matrix has zero unjustified gaps for shipped ops (the falsifier for
  the "0003 becomes a parallel VM-AD surface" risk — any VM AD op made to pass by a
  VM-only path that does not route through the shared node model is rejected here).

### Stage 13 — v1.9.1: "Types: checkpoint/restart + session protocol + debugger"

- Theme: resident continuity and the developer-facing session and debug surfaces.
- ADR-phases:
  - **0009 v1.9.1** — tick-barrier checkpoint/restart with circuit schema hashes and
    `core.memory` frontiers.
  - **0008 M4-M5** — the framed session protocol; transactional definition/reload;
    DAP over LLVM / DWARF / ORC; Eshkol-level frame traces.
- Depends on: Stage 11 (resident-agent circuit + resident state), Stage 4 (frame traces on
  tail / continuation boundaries).
- Gate: crash-and-restore at every generated cut point is identical to
  uninterrupted execution; a 46K-entry history restores derived state from a
  checkpoint plus suffix (not a full replay); a failed redefinition leaves the prior
  binding usable; arbitrary user output cannot spoof protocol completion.

### Stage 14a — v1.9.2: "Types: spill tier + reflective self-modification"

- Theme: the storage overflow tier, and the plastic self-modification surface — the
  furthest-horizon programs-as-weights slice.
- ADR-phases:
  - **0009 v1.9.2** — deterministic file-backed trace batches and a bounded-arena
    spill policy on hosted targets.
  - **0005 Phase 5** — plastic adapters with zero initial residual, shadow
    checkpoints, atomic promotion/rollback, state migration, experience provenance,
    and typed differentiable sketches only after the exact path is certified.
- Depends on: Stage 11 (WPO / LTO + capsule), Stage 13 (checkpoint discipline).
- Gate: forced-spill equivalence + checksum-corruption rejection + resident-memory
  ceiling; repeated promotion/rollback preserves capsule and resident-state
  provenance; no relaxed (differentiable-sketch) capsule ever receives an exactness
  certificate. The `theta_sem` / `psi_program` hashes are verified before and after
  every learning step, confining plasticity to declared interfaces.

### Stage 14b — v2.0: "Starlight: unified differentiate + quantum + training-grade"

- Theme: the two differentiation interpretations under one primitive; quantum
  resource types; and the AD end-state (Section 5).
- ADR-phases:
  - **0009 v2.0** — `differentiate` with explicit `numeric` / `incremental`
    interpretations; lower existing AD forms and `incrementalize` through the shared
    backbone.
  - **0004 v2.0** — `QRegion` / `Qubit` / `QReg`, gate transitions, Lean kernel
    export, mechanized core metatheory.
  - **0007 Phase 4** — application / kernel IR PGO; training-grade qualification.
  - **0002 END-STATE** — *see Section 5: this is the one open decision.* The near
    term (dense per-node tape, strict-or-error, staged value_and_grad, no FD) is
    settled; the v2.0 endpoint (resident dense mutable tape vs typed static reverse
    schedule) requires a maintainer decision before this stage is planned in detail.
- Depends on: all prior; specifically Stage 12 (IncrementalizePass +
  language-complete types), Stage 8 (staged kernel), Stage 11 (WPO).
- Gate: the numeric-AD and DBSP-composition oracles both pass across
  AOT / JIT / VM / WASM with no interpretation-specific representation leaking into
  the other (the unified-differentiation regression gate); cloning / dropping /
  serializing a qubit is rejected on every path; the Lean export re-checks the
  normative core; staged kernels meet the native reference-efficiency targets.

---

## 5. THE ONE OPEN DECISION — AD END-STATE (requires a maintainer decision)

**The judge panel split on exactly one question: the automatic-differentiation
end-state at v1.9/v2.0. This document does not resolve it. It requires a maintainer
decision.**

The two AD cluster proposals are not two architectures; they are two descriptions
of one architecture at different altitudes, and they agree on the entire near term.
Both were verified against the live tree; both are excellent. The disagreement is
only at the far horizon.

### What both positions agree on (the settled near term — Stages 1, 5, 7, 8)

- One dense AD node per tensor op, never `M*N*K` scalar nodes (the PyTorch `grad_fn`
  shape); the bypassed `recordADNodeTensor` and the `after_matmul_compute` guard are
  the shared diagnosis.
- Counters first (`primal_calls`, `reverse_passes`, `tensor_ad_nodes`,
  `finite_difference_evals`) as the definition of "done" — PR-1 for both.
- Strictness before rollout: unsupported tensor backward is a hard error, never a
  silent zero. (The alternative names it Phase C0 and moves it explicitly ahead of
  dense-node widening.)
- One-pass scalar-loss gradient (one primal, one reverse) as the first semantic
  shape change, factored as the shared core of `gradient` and `value_and_grad`.
- Exact AD or an explicit unsupported-op error on the default path. Finite
  differences survive only behind explicitly named numeric APIs. **No hidden FD.**
  This constraint is non-negotiable and is honored identically by both.
- Taylor towers remain the exact higher-order coordinate engine; the mixed-mode
  split (few coordinate vars via forward/Taylor jets, many parameters via one
  reverse sweep) for PINNs; no dense-tensor Hessian in horizon.
- A raw-pointer / out-param staged ABI with resident param/grad buffers, fixed
  scratch, a guarded shape-specialized cache key, and a status-code error surface.

### Where they genuinely differ (the v1.9/v2.0 endpoint)

| Axis | Position A (dense resident tape) | Position B (typed static reverse schedule) |
|---|---|---|
| End-state node model | Extend `ad_node_t` with dense op IDs; keep per-node backward dispatch as the VJP executor; a resident mutable tape reused across steps (`arena_tape_zero_gradients`, allocate once) | A versioned typed payload compiled into a static reverse *schedule* per `value_and_grad` call; the mutable tape is only the transitional form |
| Paradigm | PyTorch per-node tape (explicit, matches the existing substrate); linearize/transpose deferred as the v2.0 north star | JAX-like typed primitive IR (`jaxpr` / linearize / VJP transpose) as the *stated target*, per-node as the near-term step |
| What "compiled" means | The tape structure is identical every call, so the win is reuse-without-rebuild of a resident mutable tape | Each `value_and_grad` compiles to a typed, shape-specialized, straight-line reverse schedule with no reusable runtime tape |

### Recommendation, with trade-offs

The judge panel's majority recommendation is to **adopt Position A (the dense
resident tape, direction of #214) as the spine and graft four artifacts from
Position B (#216):** (1) the dense primitive registry as a real table rather than a
totalized switch; (2) the cotangent-layout and error ABI as first-class and up
front, not deferred; (3) strict-mode as a compile/run kernel flag with exact-only
default for SciML, env var as diagnostic override only; (4) the versioned dense
payload as the migration endpoint so the field mapping is not frozen around
overloaded `int64_t` params forever. These four grafts are already scheduled into
Stages 7-8 above and do not depend on resolving the endpoint.

- **Trade-off for Position A:** shortest path to a training win, because it turns
  on a substrate that already exists (explicit tape, dense VJP kernels, working
  tensor fast-path dispatch, a numerically-correct scalar oracle) rather than
  building JAX's linearize/transpose partial-evaluation machinery first. Its risk is
  that a resident mutable tape is not the fully-optimized static reverse pass; for
  static-shape straight-line kernels an emitted straight-line reverse (Enzyme-style)
  is strictly faster and tape-free, so Position A leaves performance on the table at
  the very end.

- **Trade-off for Position B:** the typed static reverse schedule is the cleaner
  and more optimizable end-state and unifies naturally with the forward-jet/Taylor
  machinery via linearize/transpose. Its risk is that it front-loads the largest,
  highest-risk compiler surface (partial evaluation, transpose rules, a typed shape
  IR) before the first training benchmark, and the panel's own verification found
  the near-term substrate already carries most of the win.

Both positions reach the *same* v2.0 stretch goal — an Enzyme-style emitted
straight-line reverse for static-shape kernels — from different starting shapes.
The decision is therefore not "which is right" but "which transitional form does
the codebase carry from Stage 8 to v2.0": a resident mutable tape that is later
replaced, or a typed schedule that is compiled from the start.

**This decision must be made by a maintainer before v2.0 (Stage 14b) is planned in
detail. It does not affect Stages 1 through 12 and must not block them.** The
recommendation above is the panel's majority view, recorded as a recommendation and
not as a resolution.

---

## 6. Resident-mind risks (acceptance criteria, not footnotes)

The resident-mind red-team ran these proposals against a live resident substrate and
returned concrete failure modes. Its central verdict on OALR: **"flatness is
conditional, not absolute."** The per-thread memory-context and region/arena split
works for stateless inference and isolated task execution, but forever-flat
cognition fails in two specific patterns. These are folded into the stage gates
above and restated here as standing acceptance criteria.

### R1 — The Ghost of Uncommitted Deltas (gate at v1.5.1, Stage 6)

When a thread holds a mutable reference to shared resident state and is preempted
before commit, a naive "flat" view sees the old state. If a second thread reads that
stale snapshot while the first is still computing a delta, the result is a *logical*
race — "memory drift" or contradictory self-description — even though no memory
unsafety occurs. The evacuator (ESH-0214d) handles the data move; it does not by
itself preserve the *semantic continuity of the read pointer*.

- Acceptance criterion: the resident session (0001 Phase E) publishes state through
  the single-writer COW transaction and epoch/RCU reader leases such that a reader
  that acquires a lease on published generation A observes a stable, immutable A for
  the whole lease, and the writer's delta is only visible after the atomic root-table
  swap to B. No transactional read, mutation, or `eq?` resolves to a half-committed
  delta. The one-million-tick RSS gate runs with readers interleaved against a writer,
  not only in isolation.

### R2 — Cross-arena pointer chasing during compaction (gate at v1.8.1, Stage 11)

A cognition pattern that follows a pointer from arena A (for example `memory/notes`)
to a live object in arena B (for example `state/mind.json`) sees a dangling
reference if B is being compacted, until the next reclamation cycle. In the
red-team's counter-example — a three-step reasoning chain that reads A, modifies B,
then reads A again — an overlap of B's compaction with the third step yields a stale
read. The system recovers, but the cognition was not flat; the observable symptom is
a 50-100ms micro-hesitation or "thought stutter" in the output stream.

- Acceptance criterion: cross-residence edges obey OALR invariant 3 (a pointer stored
  in residence D must refer to an object whose residence outlives D, or the store is
  remembered and repaired before the source can die) and the tick-barrier publication
  discipline of the DBSP circuit (0009 Section 7). Compaction of a circuit-owned
  region republishes the read-pointer semantics atomically at the tick barrier;
  superseded batch regions are destroyed only after the replacement root is published
  and no node can reference them. The resident-agent shadow-mode gate must show no stale
  cross-arena read across a sustained resident run, not merely bounded RSS.

### R3 — Z-set-as-CRDT holds only with the causal envelope (gate at v1.8.0, Stage 10)

The DBSP mapping of `core.memory` to `Z[Event]` is sound *only* because Z-set
addition, while commutative and invertible, is **not idempotent**. Blindly adding two
replica snapshots' Z-sets would give a duplicate event weight two. The CRDT holds
for knowledge-base and workspace state precisely when the RGA / vector-clock envelope
performs idempotent union first and emits only newly-observed operation ids as
deltas — the Z-set algebra is the data plane, the CRDT is the delivery envelope, and
the two are complementary layers of one transition, never interchangeable merge
algorithms.

- Acceptance criterion: the CRDT-delivery gate (merge the same replica log repeatedly
  and in all replica orders) shows each event id enters the Z-set stream exactly once
  and duplicate delivery changes no weight; tombstone/retraction views converge; the
  existing RGA merge remains commutative, associative, and idempotent. A test that
  attempts a blind Z-set snapshot merge and would double a weight is rejected in
  strict mode with the offending row and tick.

---

## 7. Top program risks and what falsifies the plan

1. **Frontend-rewrite fork (highest).** 0004 (typed HIR + FlowEnv), 0006 (BindingId
   resolution), and 0008 (SourceSpan + SymbolId + workspace resolver) are three
   overlapping frontend rewrites over one node-keyed identity substrate. Separate
   side tables or resolvers silently fork the program into subtly different
   frontends and the "one answer" invariant becomes false by construction.
   *Falsifier:* after Stage 2 (v1.3.3b), the cross-consumer fixture (compiler, LSP,
   docs, REPL report identical `SymbolId` / `ModuleId` / span for the same source)
   cannot be made green without adapters that translate between distinct identity
   schemes.

2. **The staged AD kernel is load-bearing for three clusters.** 0005, 0007 (Phases
   2-3), and 0009 (v2.0) all sit downstream of `eshkol_compile_staged_value_grad`,
   verified absent today. If the staged kernel cannot reach zero-allocation steady
   state, the back half of the roadmap slips together.
   *Falsifier:* the Layer-0 counters (`post_warmup_allocations == 0`,
   `primal_calls == 1`, `scalar_ad_nodes_from_dense_ops == 0`) fail on the first MLP
   kernel at Stage 8 (v1.6.1).

3. **OALR ABI v2 blast radius.** The 8->32 byte header plus "every layout registers
   a descriptor or startup fails" touches AOT / runtime / stdlib bitcode / JIT /
   WASM at once, and resident forever-flat under general mutation is only achievable
   if 0004's affine/unique typing has landed.
   *Falsifier:* `ESHKOL_MEMORY_ABI_V2` cannot be adopted without breaking the
   (chronically flaky) WASM / macos-x64 lite lane or forcing a full stdlib rebuild
   within Stage 4 (v1.4.1); or the 10M-tick constant-RSS gate fails for
   genuinely-aliased mutable state — proving, as 0001 itself warns, that no-GC +
   forever-flat + arbitrary aliasing cannot coexist and that resident state must be
   affine/unique.

4. **The AD/IVM unification may be a false economy.** 0009's shared
   `DifferentiationGraph` refactor risks destabilizing a working numeric AD path for
   a mostly-structural surface unification whose rule sets and runtimes are disjoint
   (the ADR itself concedes "they are not mathematically identical").
   *Falsifier:* the unified-differentiation regression gate (neither interpretation
   changes the other's graph / counters / results) cannot be met at Stage 9
   (v1.7.0) without effectively maintaining two pipelines behind one name — in which
   case do not refactor; keep two clusters with a thin common surface.

5. **Scope compression at v1.4.** The union of these ADRs is a multi-year program;
   the failure mode is a single v1.4 branch that tries to ship types + OALR ABI +
   exact AD + strict modules + tail machinery together.
   *Falsifier:* the v1.4.0 and v1.4.1 exit gates cannot both be green in one release
   cycle — which is precisely why this trajectory splits them. If even the split
   cannot land, the type-system spine (0004) must be decomposed further before v1.4.

6. **0003 becomes a parallel VM-AD surface.** Treating the parity matrix as fundable
   codegen work rather than as the gate for 0006 + 0002 risks re-implementing,
   rather than sharing, the LLVM AD substrate in the VM.
   *Falsifier:* any VM AD op is made to pass by a VM-only implementation that does
   not route through the shared node model — a divergence the Stage 12 (v1.9.0)
   VM-parity gate must reject.

7. **Programs-as-weights is pulled forward prematurely.** 0005 is the
   furthest-horizon cluster; its recurrent meta-gradient needs the staged kernel,
   a custom `vm_step` VJP, and BPTT checkpointing that do not exist, and its resident
   capsule needs 0001's resident sessions.
   *Falsifier:* any 0005 slice scheduled before its predecessor stage (Stage 8 for
   the kernel, Stage 10 for resident sessions) cannot meet the capsule three-way
   verification or the `dstate`-vs-unrolled-oracle gate.

### Overall assessment

The nine-cluster set is unusually coherent and evidence-grounded. Two clusters are
near-unconditional accepts (0006, and the reconciled near-term 0002); one is data
masquerading as a decision (0003, kept as a gate); the rest are strong-but-large,
and their principal shared risk is not the correctness of any vision but the
serialization of one very large, tightly-coupled frontend and memory substrate. This
trajectory is built to make that coupling explicit and to force each heavy substrate
change — the frontend identity substrate, the memory ABI, the staged AD kernel — to
fail in isolation rather than as a monolith. The one question it deliberately leaves
open is the AD end-state (Section 5), which is a maintainer's to decide.
