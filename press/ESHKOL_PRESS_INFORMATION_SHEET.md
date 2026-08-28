# Eshkol v1.3.5-evolve — Press Information Sheet

For trade-press and academic readers preparing coverage of the v1.3 release
line — arbitrary-order automatic differentiation, full R7RS conformance,
resident/daemon-workload robustness, opt-in differentiable quantum computing
with post-quantum cryptography, and, as of v1.3.5-evolve, multi-shot re-entrant
continuations on every engine, region reclamation on the bytecode VM, constant
stack space for mutual tail recursion in every tail-position spelling, and
differentiation dispatch the compiler proves exhaustive — and the SDNC paper
artefact it carries.

---

## Identity

| | |
|:---|:---|
| Project | Eshkol |
| Version | v1.3.5-evolve |
| Builds on | v1.3.4-evolve (31 July 2026), v1.3.3-evolve (16 July 2026), v1.3.2-evolve (9 July 2026), v1.3.1-evolve, v1.3.0-evolve (7 July 2026) |
| Release date | 28 August 2026 |
| Licence | MIT |
| Source | https://github.com/tsotchke/eshkol |
| Website | https://eshkol.ai |
| Maintainer | tsotchke / Tsotchke Corporation |
| Contact | team@tsotchke.org |

---

## Headline

In v1.3.5-evolve you can call a procedure that has already returned. That is
not a metaphor. `call/cc` is multi-shot and re-entrant on all three engines
Eshkol ships — native JIT, native AOT, and the bytecode VM — because a capture
that may outlive its frame takes a durable copy of the live C stack and restores
it to the same addresses before resuming, so every frame pointer, every spilled
register, and every address of a local that a closure is holding stays valid
with no relocation. Six programs, three engines, eighteen byte-exact
transcripts.
<!-- source: tests/continuations/ (6 fixtures), scripts/run_continuation_tests.sh (each fixture on -r, AOT and VM against tests/continuations/expected) -->

A six-layer transformer with 12.22 million analytically-constructed parameters
executes a bounded 83-opcode bytecode VM bit-identically. The result is a
constructive — not statistical — proof that a fixed-weight transformer can
*be* an interpreter when its weights are derived from the instruction-set
specification rather than fit by gradient descent. The reproducibility
artefact, including the weight tensor, traces, and a three-way agreement
report, ships in the same repository as the compiler that hosts it, and
regenerates in one command on a developer laptop.

The host language has moved through this release line: v1.3.0-evolve gave
Eshkol's automatic differentiation a second axis, arbitrary order, with exact
bignum/rational derivatives and full R7RS conformance on a portable differential
corpus; v1.3.1-evolve and v1.3.2-evolve made an Eshkol program safe to run
unattended as a daemon, with flat memory in long-running loops, an iterative
reader for large persisted state, region-escape evacuation across the
logic/workspace heap subtypes, and thread-safe region scoping under
`parallel-map`. v1.3.3-evolve added the opt-in quantum computing stack — Moonlab
state-vector simulation, a variational quantum eigensolver whose gradients flow
through Eshkol's own automatic differentiation via custom-VJP tape nodes, a CHSH
Bell-inequality gate, Bell-verified quantum randomness, and ML-KEM (FIPS 203)
post-quantum cryptography — alongside real `make-parameter`/`parameterize`
dynamic parameters, the `core.dbsp` incremental-dataflow module, bignum-capable
exact rationals, and one-pass reverse-mode gradients. v1.3.4-evolve made
automatic per-iteration reclamation match explicit regions, made `parallel-map`
race-free for collection-valued closures, gave `gradient` full parity on the
bytecode VM, and landed the high-precision numerics wave: Ozaki-II exact and
reduced-precision GEMM tiers, a mixed-precision `linear-solve`, and a native
128-bit integer type.

v1.3.5-evolve is the re-entrant-continuations release. `call/cc` becomes
multi-shot on every engine; a continuation captured inside `with-region` stays
safe to resume after that region has closed; the bytecode VM gains its own
region evacuator, so `with-region` reclaims memory there the way it already did
natively; mutual tail recursion runs in constant stack space through every
tail-position spelling rather than only `if`; cloning a linear `Qubit` is a
rejected compile in the default build; and differentiation dispatch is closed by
construction, with the AD node registry generated from a single declaration file
that has no `default:` arm and exhaustiveness enforced by the compiler itself.
Alongside those, the release starts the object-ABI migration, ships a canonical
`find_package(Eshkol)` link contract, publishes a one-command reproducible
benchmark suite on Eshkol's own axes, and lays the first column of the
frontend node-identity substrate that the compiler, the LSP, the docs, the REPL,
and the VM will share.

---

## Why this is news

The dominant pattern in machine learning today is *parametric*: a generic
architecture is trained to approximate a target behaviour over a data
distribution. SDNC is *constructive*: an architecture is given a closed-form
weight assignment that realises a target behaviour exactly. The result is
narrow — the target is a 256-dimensional, 83-opcode VM — but it is exact,
auditable, and reproducible. For a transformer with weights tied to a
specification rather than to a checkpoint, every numerical claim is decidable
by inspecting the weights.

Eshkol is the host language for that result. The language is a compiled R7RS
Scheme dialect with automatic differentiation and a small neuro-symbolic stack
as compiler primitives. The SDNC artefact uses Eshkol's differentiation
infrastructure (the same that powers `gradient`, `jacobian`, `hessian`) as
the substrate in which the transformer's own forward and backward passes are
expressed. The relationship is reflexive: a language with first-class AD
hosts a transformer that has AD as part of its instruction set.

The paper is *The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction* (tsotchke, 2026). The
artefact directory is
`artifacts/paper/`; reference documentation lives at *docs/SDNC.md*,
*docs/breakdown/COMPUTABLE_TRANSFORMER.md*, and
*docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md*.

---

## What is new in v1.3.5-evolve

A re-entrant-continuations release, and the release in which the bytecode VM
starts giving memory back. Every capability below is pinned by an executable
gate.

**Continuations**

- **Multi-shot, re-entrant `call/cc` on native JIT, native AOT, and the
  bytecode VM.** A captured continuation can be invoked any number of times,
  from any dynamic extent, including after the procedure that captured it has
  already returned — the shape generators, coroutines, and `amb`-style
  backtracking search all need. Native gives a capture that may outlive its
  frame a durable copy of the live C stack, restored to the same addresses
  before resuming, so every interior pointer stays valid with no relocation; an
  escape-only capture keeps the original zero-overhead `setjmp`/`longjmp` path.
  The VM snapshots its operand stack and call-frame array while excluding
  top-level bindings — the *store* — from the *control* snapshot R7RS asks
  `call/cc` to capture, so `set!` and `define` effects at top level survive
  re-entry. `dynamic-wind` reroots on both engines per R7RS 6.10.
- **A continuation captured inside `with-region` is safe to resume.** Both
  engines pin every open region at capture time and promote a pinned region into
  its parent on pop rather than freeing it, so the failure direction is a
  bounded leak and never a dangling read. Pinning triggers only on an actual
  capture: a program that never calls `call/cc` shows zero behaviour change,
  confirmed by the unchanged flat-RSS gates.

**Memory**

- **`with-region` reclaims on the bytecode VM.** A Stage-1 OALR region evacuator
  ports region reclamation to the VM heap. The port matches the native engine's
  semantics rather than its implementation: native copies the escaping subgraph
  Cheney-style, while the VM marks from its root set and sweeps at arena-block
  granularity, because a VM value addresses the heap by a small integer index
  rather than by pointer — marking moves nothing, so `eq?` identity, shared
  structure, and cycles survive with no special handling. Coverage is total by
  construction: a compile-time-checked 33-wide table classifies the full heap
  tag space (the 28 `HeapType` members, the three manifold tags defined outside
  the enum, and the two unassigned slots), a fatal startup check requires every
  row to be filled in, and an unclassified subtype pins its region rather than
  guessing. A `raise` crossing a region and a continuation transfer out of one
  go through the same teardown call as normal exit, so the structured and
  unstructured surfaces cannot drift apart.
- **The scope of Stage 1, stated exactly.** Outside a region the VM's heap
  growth watchdog remains the bound, and the same allocation volume that trips
  the budget unwrapped does not trip it wrapped. The user-reachable region
  *handle* surface (`region-open`/`region-close`) stays bookkeeping-only on the
  VM and announces that at the point of use — a handle can be closed out of
  order, from another dynamic extent, or never, whereas `with-region`'s lexical
  extent tells the teardown where the region ends, which is why the lexical form
  landed first; the handle surface is Stage 2. An escaping object with an
  out-of-line payload (a vector's element array, a bignum's limbs) keeps the
  arena block that payload occupies, while escaping cons and closure structure
  is copied out exactly, so a cons-only loop is perfectly flat and a
  payload-heavy one is much smaller. Objects promoted out of a region live in
  the enclosing arena for its lifetime, which is OALR's semantics and equally
  true natively.
- **Five new runtime variables** — `ESHKOL_VM_REGION_EVAC`,
  `ESHKOL_VM_REGION_VERIFY`, `ESHKOL_VM_REGION_VERIFY_FATAL`,
  `ESHKOL_VM_REGION_COMPACT`, `ESHKOL_VM_REGION_RECYCLE` — all documented in
  [docs/reference/runtime/environment-variables.md](../docs/reference/runtime/environment-variables.md).
  `ESHKOL_ARENA_POISON`, the variable the native arena already reads, now arms
  the VM's arena too.
- **An exception guard entered once per tick costs nothing in steady state.**
  Handler frames come from a thread-local LIFO free list, malloc-backed rather
  than arena-backed, since an arena address can be retracted by a region or
  iter-scope rewind and handing it back would alias a fresh object. Total frame
  memory is bounded by peak nesting depth rather than by entry count. A resident
  long-run gate measures at 200,000 and at 1,600,000 ticks — eight times apart —
  and gates on the slope rather than on a ceiling: transient garbage and all four
  persistent-mutation channels come back at exactly 0.000 bytes per tick, with
  identical byte totals at both horizons, so what a resident loop retains is what
  it publishes and nothing else. `ESHKOL_ARENA_REPORT=1` prints the global
  arena's byte-exact allocation total at exit, because peak RSS is a high-water
  mark of instantaneous residency that reads low under memory pressure.
  [docs/reference/runtime/memory-model.md](../docs/reference/runtime/memory-model.md)
  carries a measured matrix of which workload classes are exactly flat and which
  are not.
  <!-- source: docs/reference/runtime/memory-model.md:308-313; tests/memory/resident_longrun_flat_gate.sh:77-79 -->

**Tail calls**

- **Every tail-position spelling gets the constant-stack guarantee `if` always
  had.** A mutual tail call written with `cond`, `case`, `when`, `unless`, or as
  the last operand of `and`/`or` now runs flat, and the walker that offers
  candidate sites is as wide as the oracle that confirms them. The depth ladder
  runs each mutual-recursion shape at 500,000, 5,000,000 and 100,000,000 hops,
  including a four-cycle that routes through `cond`, `when`, `or` and `case`,
  one form per hop.
  <!-- source: scripts/gen_recursion_depth.py:125-155 (mutual_tail_cond, mutual_tail_forms) -->
- **A tail-transfer dispatcher removes the remaining bounds.** A transferring
  procedure does not call its target: it copies its evaluated arguments into a
  per-thread transfer record, records the callee's uniform entry, and returns,
  and a driver loop compiled into the public entry point runs the transfer in
  its place — one native frame live per hop, reused regardless of arity or
  target. Differing-signature mutual tail calls run 100,000,000 hops at 9.1 MB
  peak resident memory, and non-AArch64 targets get the same bound without an
  aggregate-return `musttail`. Tail calls through `guard` stay bounded
  deliberately: R7RS does not make that a tail context, so optimizing it would
  be wrong rather than merely unfinished. The same work lands the OALR ABI v2
  Phase A groundwork, tracked separately against ADR-0006.

**Automatic differentiation**

- **Dispatch is exhaustive by construction.** `ad_node_type_t`,
  `callable_subtype_t`, and `EvacKind` are generated from a single declaration
  registry (*inc/eshkol/ad_node_registry.def*) with no `default:` arm, with
  `-Werror=switch` and `-Werror=switch-enum` making an unhandled member a
  compile error rather than a plausible answer, and an ICC invariant re-derives
  each enum's members from its own definition so the guarantee holds on
  toolchains where the compiler flag alone cannot enforce it. A registry row
  naming a backward function that does not exist is itself a compile error,
  which is what makes a row's claim to be registered mean registered. Open sets
  stay loud on purpose: a subtype byte read out of an object header is untrusted
  input, so it is split off with a value-naming fallback, and the VM's opcode and
  value-type switches, which dispatch on bytecode, are left as loud backstops.
- **Exact backwards for the four geometric bridge operators.** Hyperbolic
  distance, the Poincaré exponential and logarithmic maps, and geodesic
  attention carry exact closed-form rules, each declared as a bridged row in the
  registry. The exp and log rules reuse the Möbius-addition and log-map
  Jacobians the Fréchet rule already differentiates rather than re-deriving
  them, since the log map *is* the function that rule differentiates and a
  second derivation could only introduce a disagreement. Two points are made
  explicit rather than hidden: the distance is not differentiable at coincident
  points, and geodesic attention is therefore not differentiable when a query
  row equals a key row exactly — the ordinary case when `Q` and `K` are the same
  tensor — and both refuse loudly, naming the offending index, rather than
  picking a plausible subgradient. The gradcheck is pinned by count at 13
  checks, and an exhaustive-dispatch test carries 11: the registry holds 18
  bridged rows against 4 that no forward produces yet.
  <!-- source: tests/bridge/qllm_bridge_geometric_gradcheck_test.cpp (13 checks); tests/backend/exhaustive_dispatch_test.cpp (11 checks); inc/eshkol/ad_node_registry.def (18 BRIDGE rows, 4 UNREGISTERED) -->
- **Forward producers for the tensor-embedding and Fréchet-mean AD nodes.**
  `ad_tensor_embedding` and `ad_frechet_mean` record real nodes through real
  dispatch, so the backward rules are exercised by the producer that fills their
  contract rather than by hand-assembled fixtures written from the same contract
  the rule reads. Fractional, negative, and out-of-range embedding indices are
  refused at record time rather than rounded or clamped into a wrong row, and
  the Fréchet forward shares its Karcher iteration with the VM's own opcode so
  forward and backward cannot disagree about what "converged" means. Gradchecked
  through the real producers and the real dispatch against exact analytic
  references: exact scatter-add for the embedding with 0 mismatches, the exact
  Euclidean closed form for the Fréchet mean at 0.0, and a hyperbolic finite
  difference of 8.3e-10 over 48 partials.
  <!-- source: CHANGELOG.md v1.3.5-evolve (#497); tests/bridge/qllm_bridge_producer_gradcheck_test.cpp -->
- **The no-finite-differences guarantee is enforced, and exactness gains a
  structural gate.** The counter behind `(ad-finite-difference-evals)` has a real
  writer on the one central-difference backward the tape defines, reported
  through the zero-arity builtin `(ad-note-finite-difference!)` on native and on
  the VM alike, and the exactness gate runs a positive case beside a negative
  control — a difference quotient deliberately planted in the gradient path — on
  JIT, AOT, and the VM. Separately, `.icc/ad-carrier-manifest.yaml` declares, per
  operator and per engine, which differentiation carrier answers it and whether
  it is exact, and a gate re-derives each declaration by extracting and
  classifying the actual `case` body in the emitted sources, so a declaration
  cannot be laundered through a helper. An output differential can only compare
  what two carriers compute, never which carrier computed it; this closes that
  gap.

**Language surface**

- **R7RS 7.1.1 vertical-line symbol syntax, read and written.**
  `<identifier> -> <vertical line> <symbol element>* <vertical line>` is one of
  R7RS's three `<identifier>` productions. All four readers — the native
  tokenizer, the VM tokenizer, and both runtime `read` implementations — accept
  the full `<symbol element>` alphabet, including the mnemonic escapes, `\|`,
  and `\x<hex>;`. The bars request a verbatim spelling, so `#!fold-case` does
  not apply inside them, and `|.|` is an ordinary symbol distinct from the bare
  `.` dotted-pair delimiter. `write` emits bars only when a name cannot be
  spelled bare under the R7RS grammar; `display` never bars. A shared
  predicate and escaper in *inc/eshkol/core/symbol_syntax.h* keeps the native
  and VM writers byte-identical.
- **`gensym` is reachable on every engine**, identically on native JIT, native
  AOT, and the bytecode VM.
- **The full quoted-datum grammar on both engines.** Quoted vectors,
  quasiquote, and unquote are read as data by all four readers, and the reader's
  `is_char`/`is_inexact` flags survive into the quoted form, so `'(#\a)` is a
  list of a character.
- **Cloning a linear `Qubit` is a rejected compile.** In the default
  compilation mode a linearity violation stops before code generation, exits
  nonzero, and writes no artifact — the same discipline `--strict-types` already
  had. Measured against 24 shapes.

**Frontend, ABI, and packaging**

- **The frontend node-identity substrate (ADR-0000 Stage 1, phase A).** Every
  AST node the parser produces carries a stable `NodeId`, and a chunked,
  lock-free-read side table maps that id to a `SourceSpan` — the first column of
  the `NodeId -> {SourceSpan, BindingId, TypedExprInfo}` substrate that the
  compiler, the LSP, the docs, the REPL, and the VM all have to be built on if
  they are to give one answer rather than five. The parser's 32
  location-stamping sites write the location and the identity in one statement
  so the two cannot drift apart, and the stream reader closes each top-level
  form's span with a measured *extent* — the first place in the frontend that
  records where a construct ends rather than only where it begins. `NodeId`s are
  tagged rather than bare indices, and a garbage word is rejected by its tag and
  again by its bound, so it reads as "unknown": a diagnostic may fail to name a
  location, but it must never name a wrong one confidently. The change is
  strictly additive — `line`, `column`, and `source_file_id` keep their exact
  previous values. The LLVM codegen dispatcher is the first consumer, resolving
  the file, line, and column it reports through the substrate and falling back
  to the node's own fields for a node synthesized after parsing, so a node's
  *file* does not depend on the traversal that reached it. Coverage is measured
  at the consumer rather than at the parser, because a parser-side count says
  only how many ids were minted and never whether the answer arrived where it
  was needed; "has an identity", "has a location", and "has an extent" stay three
  separate numbers, graded by their own completion oracle against a monotonic
  span-coverage floor of 99.48% written truncated rather than rounded so it
  cannot drift upward by accident.
  <!-- source: tests/coverage/NODE_IDENTITY_BASELINE.json (span_coverage_floor 0.9948); scripts/run_node_identity_gate.py:197-243 --> Phase A only, and the oracle says so.
- **Object ABI migration, stage 0 (ADR-0012).** Three layers of detection —
  lexical token matching, libclang semantic resolution, and emitted-LLVM-IR
  ground truth — enumerate 1,273 sites that depend on the current object-header
  layout, ratcheted against a committed baseline so a new site fails the build.
  <!-- source: docs/design/adr/0012-object-abi-staged-migration.md:100-102 (816 lexical sites across 98 files; 1,273 once the semantic layer is added); ratchet baseline .icc/abi-header-baseline.json --> A link-time guard whose symbol name is derived from
  the four numbers that determine object-exchange compatibility means a stale
  object file, JIT cache entry, installed runtime, or `--shared-lib` artifact
  fails to *link*, with an undefined-symbol error naming the layout it wanted. A
  layout-pin test pins the header's size and every field's offset both through
  the accessor and as raw bytes at the negative offsets generated code actually
  uses. ADR-0012 sequences the seven remaining migration stages, each with a
  named falsifier.
- **A canonical `find_package(Eshkol)` link contract.** `cmake/FindEshkol.cmake`
  ships as the one discovery module: it resolves the compiler, the runtime
  archive a compiled program actually needs, and the stdlib object and module
  directory, producing an `Eshkol::eshkol` imported target whose link interface
  unconditionally includes `stdlib.o` plus, on Apple, the system frameworks the
  runtime needs — with no hand-written library search in the consumer at all.
  The homebrew formula and both release-asset steps install the module and
  `EshkolCompile.cmake`, and *tests/integration/system_package/* is a
  from-scratch consumer CMake project run against a staged package by the
  package manifest, so what is checked is that the discovery contract works and
  not merely that its files landed. Scoped to macOS and Linux; the Windows and
  MSVC link recipe is a later item.

**Benchmarks and assurance**

- **A public, one-command, reproducible benchmark suite**
  (`bench/run_public_benchmarks.sh`). From a clean checkout it measures the four
  axes where Eshkol claims something distinctive — exact-AD cost curves, Ozaki-II
  CRT exact f64 GEMM, flat RSS under resident load, and differentiable quantum
  kernels — and emits machine-readable JSON alongside a human-readable table,
  <!-- source: bench/run_public_benchmarks.sh:107-110 --> with the noise-control methodology and an explicit not-benchmarked list
  documented in *bench/README.md*. It is not a competition entry against XLA,
  PyTorch, or JAX.
- **A gated compile-time benchmark on a large single file.**
  `bench/generate_large_single_file.py` generates a deterministic, self-contained
  fixture of top-level defines referencing earlier-numbered functions, calibrated
  against a measured growth curve, and `bench/large_single_file_compile_bench.sh`
  compiles a 1,600-define fixture against a 900-second ceiling nightly, killed
  with `SIGKILL` rather than `SIGTERM` because the compiler does not exit
  promptly on `SIGTERM` mid-codegen, and capturing an `ESHKOL_PHASE_TIME=1`
  breakdown that attributes about 98% of the wall clock to LLVM's own backend
  rather than to Eshkol's frontend.
  <!-- source: bench/large_single_file_compile_bench.sh:44-45; CHANGELOG.md v1.3.5-evolve (#495) -->
- **The PGO training corpus has a consumer.** `scripts/run_pgo_corpus_smoke.sh`
  runs every corpus program under the JIT and AOT and asserts both exit 0 with
  byte-identical, non-empty stdout, which makes it a real differential check
  rather than a liveness smoke test; wired as a nightly job. This is Stage 1
  only — the CMake orchestration that drives profile-guided compilation over the
  corpus is tracked against ADR-0007.
- **A GPU correctness gate that can fail.** Every `tests/gpu/*.esk` file
  aggregates a failure counter and exits on a final `PASS:`/`FAIL:` verdict line;
  the test isolation layer gains an opt-in verdict-grammar check that fails a
  test which exits 0 without printing a recognized marker; and a permanent,
  deliberately-failing canary (*tests/gpu/gate_canary_must_fail.esk*) runs on
  every invocation and is required to fail, forcing the whole run red if it is
  ever not red. On the strength of a measured Metal-versus-CPU divergence of
  exactly 0 across ten probes, the gate tolerance tightens from `1e-4` to
  `1e-9`. The Windows judge honours the same contract: it applies the same
  canary inversion, and every marker check now evaluates its regexes in
  multiline mode at the one place those checks are made, so a marker printed
  after a banner line is seen.
- **Two assurance waves that check the assurance.** Ledger-integrity and
  oracle-schema gates fail on a duplicate identifier, a missing required field,
  or a structurally invalid criterion, and report declared-versus-graded
  criteria counts per oracle. A self-verdict scanner fails a PASS-graded
  artifact whose own text still contains a failure marker. Build-fingerprint
  checks record and check the compiler binary's size, mtime, and SHA-256
  alongside the checkout's git SHA, failing when a built binary predates the
  most recent build-relevant source change. A PowerShell-encoding gate reads the
  files' own bytes and fails any tracked `*.ps1`/`*.psm1` file carrying a byte at
  or above `0x80` without a UTF-8 byte-order mark, reporting the exact
  `file:line:col` and codepoint — a check CI could never make by execution,
  since running the same bytes under a different default encoding is precisely
  what hides the problem. A false-green audit fails an oracle target whose
  evidence can go missing without the target going red, and an adversarial
  scenario suite exercises the gates themselves under a dirty worktree, a stale
  binary, a model-server outage, disk pressure, and an actually failing gate.
  Every gate ships a `--self-test` mode, and every one of those modes is wired
  into `ctest` or CI.
- **`icc readiness` is machine-reachable.** Every finished trace is mirrored
  unconditionally into the directory the readiness oracle reads. A `pillars-fast`
  job runs the cheap gates on every pull request — depth coverage, the ADR-0009
  DBSP acceptance gate, the monotone-equivalence Taylor gate, the AD validated
  bounds gate, and VM parity — and a nightly workflow runs the expensive sweeps:
  depth-parametric, differential, edge-matrix, metamorphic, sanitizer-fuzz, the
  full smoke battery, and SICP. Every trace-emitting harness now shares a
  PASS/FAIL/INFRA/SKIP vocabulary with a real fork-based timeout and a
  retry-once helper, so an infrastructure timeout cannot publish itself as a
  code defect. The `v1.4-connection` completion oracle carries one criterion per
  named deliverable, each bound to a real harness, and v1.3.5-evolve has a
  completion-oracle target of its own so that "all of v1.3.5" is a
  machine-checkable claim.
- **CI reports what it ran.** Every required context reports on every pull
  request, including a documentation-only one, because the docs-only decision is
  a job-level gate rather than a trigger-level path filter — and that gate fails
  safe, running everything when it cannot compute a diff. The push trigger is
  narrowed to the long-lived branches, so a branch under review runs the matrix
  once rather than twice. Self-hosted lanes are opt-in behind a repository
  variable and restricted to non-fork pull requests, with no required hosted
  lane removed, weakened, or made conditional; the mesh gate is an explicitly
  advisory job that emits an honest warning when its telemetry is unavailable
  rather than fabricating a verdict; and the release-readiness gate, which needs
  ICC, runs on the maintainer's own runner with a preflight step that resolves
  its toolchain and fails loud and specific if anything is missing.

### Measured this cycle

Every number below is produced by a gate or harness in the repository. The
v1.3.4-evolve rows were re-measured against a from-source build during the
v1.3.5 documentation wave and carry into this release unchanged.

| Claim | Where it is measured | Result |
|---|---|---|
| Re-entrant continuations across engines | `scripts/run_continuation_tests.sh` over `tests/continuations/` | 6 fixtures on native JIT, native AOT and the bytecode VM; transcripts byte-identical to the committed expected files |
| VM region reclamation, one fixture swept by iteration count | `tests/memory/vm_region_flat_rss_test.sh`; [RUNTIME_CONFIGURATION.md](../docs/breakdown/RUNTIME_CONFIGURATION.md#bytecode-vm-region-reclamation) | flat 25-27 MB at 1,000 / 4,000 / 16,000 iterations, against 793 MB with `ESHKOL_VM_REGION_EVAC=0` and 704 MB for an unwrapped control |
| VM heap tag coverage | `vm_evac_subtype_table[]`, compile-time span check | 33 rows: 28 `HeapType` members, 3 manifold tags, 2 unassigned slots |
| Mutual tail recursion in every spelling | `scripts/gen_recursion_depth.py`, `mutual_tail_cond` and `mutual_tail_forms` | ladder of 500,000 / 5,000,000 / 100,000,000 hops, all expected to pass |
| Differing-signature mutual tail calls | tail-transfer dispatcher gate, `.icc/completion-oracles.yaml` | 100,000,000 hops at 9.1 MB peak RSS |
| Resident daemon loop, two horizons 8× apart | `tests/memory/resident_longrun_flat_gate.sh` at 200,000 and 1,600,000 ticks | transient garbage and all four persistent-mutation channels at exactly 0.000 bytes/tick, identical byte totals at both horizons |
| Frontend span coverage at the consumer | `scripts/run_node_identity_gate.py` against `tests/coverage/NODE_IDENTITY_BASELINE.json` | monotonic floor of 99.48% |
| qLLM oracle | `tests/qllm_oracle/` | gate 10/10 across five exporters on the JIT and AOT lanes, over 77 in-language checks |
| Geometric bridge backwards vs. independent golden Jacobians | `tests/bridge/` | agreement to 3.7e-16 and 1.1e-14 |
| Geometric bridge backwards vs. derivation-independent identities | conformal gradient-norm and inverse-Jacobian identities, `.icc/silent-wrong-ledger.yaml` SW-65 evidence | max relative deviation 5.0e-16 and 6.7e-16 |
| Embedding and Fréchet-mean producers, gradchecked through real dispatch | `tests/bridge/qllm_bridge_producer_gradcheck_test.cpp` | embedding exact scatter-add, 0 mismatches; Fréchet exact Euclidean closed form, 0.0; hyperbolic finite difference 8.3e-10 over 48 partials |
| Object-header layout dependence | `scripts/abi_header_inventory.py`, ratcheted against `.icc/abi-header-baseline.json` | 1,273 enumerated sites over three detection layers |
| GPU correctness gate tolerance | `scripts/run_gpu_tests.sh`, `GPU_GATE_TOL` | `1e-9`, on a measured Metal-versus-CPU divergence of exactly 0 across ten probes |
| Vertical-line symbol syntax | `tests/features/pipe_symbol_test.esk` | 51 checks, run on native JIT, native AOT and the VM as a three-way parity check |
| `gensym` on every engine | `tests/control_flow/gensym_test.esk` | 9/9 on all three engines |
| Large single-file AOT compile | `bench/large_single_file_compile_bench.sh` | 1,600 defines inside a 900-second ceiling; about 98% of wall clock in LLVM's backend |
| Exact rational derivative | `(derivative-n g 1/3 1)` for `g(x) = 8x²` | `16/3`, `exact?` `#t` |
| Exact bignum derivative | `(derivative-n f 7 12)` for `f(x) = x^30` | `67465815595294257109436307840000`, `exact?` `#t` |
| H2 vibrational frequency, exact second derivative | `eshkol-run -r examples/h2_vibrational.esk` | 5003.2038 cm⁻¹ (R* = 1.38869 bohr, E(R*) = -1.13731 Ha) |
| Ozaki-II CRT exact GEMM, Metal on Apple M2 Ultra | `tests/gpu/ozaki_certification_test.esk` | 25/25 samples, 0 mismatches, max 58 correct dot bits, PASS |
| CHSH Bell-inequality gate | `tests/quantum/bell_chsh_test.esk` | S = 2.835 over 16,000 shots, gate `2.4 < S <= 2.95`, PASS |
| Gradient parity across engines | `(gradient f 3.0 4.0)` for `f(x,y) = x²y + y³` | `#(24 57)` on native JIT, native AOT and the VM, byte-identical |
| Flat-RSS resident loop, AOT gate | `tests/memory/define_loop_flat_rss_aot_test.sh`, 1,000,000 iterations | 8 MB peak RSS, against 2,620 MB with reclamation compiled out |
| Linear `Qubit` no-cloning | `eshkol-run --strict-types -r tests/typesystem/qubit_no_cloning_test.esk` | compile-time type error: "linear variable 'q' was consumed more than once" |

<!-- sources for the carried-over v1.3.4-evolve rows: CHANGELOG.md v1.3.5-evolve, "v1.3.5 documentation wave" (all re-measured against a from-source build of commit 694c3179) -->

Not measured in this environment: the CUDA Ozaki-II path (no CUDA hardware
here; the Metal path above stands in) and WebAssembly gradient parity (no
Emscripten toolchain here; the native JIT / native AOT / VM three-way agreement
above stands in for cross-engine parity).

---

## The language

### Identity

Eshkol is an R7RS-compatible Scheme dialect. The implementation passes
roughly 232 of 244 R7RS-small procedures (~95%), includes hygienic
`syntax-rules` macros, first-class multi-shot re-entrant continuations
(`call/cc`, `dynamic-wind`, `guard`/`raise`, `delay`/`force`) as of
v1.3.5-evolve, bytevectors, records via `define-record-type`, and `eval` with
all three R7RS environment constructors
(*docs/breakdown/OVERVIEW.md §Eshkol vs. Scheme*). As of v1.3.5-evolve the
reader also accepts R7RS 7.1.1 vertical-line symbol syntax across all four
readers, and `gensym` is reachable identically on native JIT, native AOT, and
the bytecode VM. A separate,
newer measure — a reference-Scheme differential oracle that diffs Eshkol
against chibi-scheme 0.12.0 on a 34-program portable corpus — reports
34 of 34 AGREE (100%), up from 27/34 at the start of the v1.3.0-evolve cycle.

The parser handles ninety-four operation types over an S-expression syntax
with line/column tracking and an R7RS-compliant internal-defines transform
to `letrec*`. The macro expander supports ellipsis patterns, nested
patterns, and hygienic renaming. The S-expression reader (`read_list`) is iterative rather than per-element
recursive, so reading back a very large persisted list costs no native stack
frame per element.
See *lib/frontend/parser.cpp*, *lib/frontend/macro_expander.cpp*, and
*lib/core/runtime_reader_hosted.cpp*.

The continuation surface states its own scope. `call/cc` captures the control
state and not the store, so `set!` and `define` effects at top level survive
re-entry on both engines. Two shapes are documented as outstanding rather than
inferred: a binding established after capture on the VM's operand-stack store is
refused with a diagnostic rather than answered, and a `set!`-assigned local that
is neither a top-level binding nor closure-captured rolls back on re-entry,
pending assignment conversion. Both are named in
[docs/reference/language/continuations.md](../docs/reference/language/continuations.md)
and tracked in the project's silent-wrong ledger.

### Implementation

| Component | File / directory |
|:---|:---|
| Main LLVM codegen | *lib/backend/llvm_codegen.cpp* |
| LLVM backend modules | *lib/backend/* |
| Autodiff codegen (order ≤ 2, incl. custom-VJP) | *lib/backend/autodiff_codegen.cpp* |
| AD node registry (single declaration, no `default:`) | *inc/eshkol/ad_node_registry.def* |
| Taylor-tower runtime (arbitrary order) | *lib/core/runtime_taylor.c*, *lib/core/taylor_recurrences.def* |
| Taylor-tower stdlib modules | *lib/core/ad/{guw,tensor_tower,taylor_models,checkpoint,taylor_numerics,sparse_guw,interval}.esk* |
| Backward-mode kernels | *lib/backend/tensor_backward.cpp* |
| qLLM geometric bridge and its exact backwards | *lib/bridge/qllm_bridge.cpp*, *lib/bridge/tensor_backward.cpp* |
| String / I/O / JSON / CSV | *lib/backend/string_io_codegen.cpp* |
| Work-stealing parallel codegen | *lib/backend/parallel_llvm_codegen.cpp* |
| Arithmetic (incl. bignum/rational/complex) | *lib/backend/arithmetic_codegen.cpp* |
| Collection ops | *lib/backend/collection_codegen.cpp* |
| Parser | *lib/frontend/parser.cpp* |
| Frontend node identity (v1.3.5-evolve) | *inc/eshkol/frontend/node_identity.h*, *lib/frontend/node_identity.cpp* |
| Macro expander | *lib/frontend/macro_expander.cpp* |
| Type checker | *lib/types/type_checker.cpp* |
| Arena memory | *lib/core/arena_memory.h*, *lib/core/runtime_arena_\*.cpp* |
| Region runtime and escape evacuation | *lib/core/runtime_regions.cpp* |
| VM region evacuator (v1.3.5-evolve) | *lib/backend/vm_region_evac.c* |
| Object ABI fingerprint (v1.3.5-evolve) | *inc/eshkol/abi_fingerprint.h*, *lib/core/abi_fingerprint.c* |
| S-expression reader (iterative) | *lib/core/runtime_reader_hosted.cpp* |
| Logic engine | *lib/core/logic.cpp* |
| Active-inference engine | *lib/core/inference.cpp* |
| Global workspace | *lib/core/workspace.cpp* |
| Quantum FFI (opt-in) | *lib/agent/quantum.esk*, *lib/agent/c/agent_quantum.c* |
| ML-KEM post-quantum KEM (opt-in) | *lib/agent/pqc.esk*, *lib/agent/c/agent_pqc.c* |
| Incremental dataflow | *lib/core/dbsp.esk* |
| Weight-matrix transformer | *lib/backend/weight_matrices.c* |
| Bytecode VM + runtime libs | *lib/backend/eshkol_vm.c* and its *vm_\*.c* modules |

The public C-API headers under `inc/eshkol/` and the implementation files under
`lib/` carry Doxygen-format documentation, harvested automatically into a
generated `docs/api/` reference by `eshkol-doc`.

### Target backend

LLVM 21 is the version-enforced target on every platform; the build aborts
with a clear error message if `llvm-config` reports any other major version
(*cmake/LLVMToolchain.cmake §`eshkol_validate_llvm_major`*).
Targets currently supported:

- macOS, Intel and Apple Silicon
- Linux, x86-64 and ARM64
- Windows, x86-64 and ARM64, via Visual Studio 2022 + ClangCL + the LLVM 21 SDK
- WebAssembly, via `eshkol-run --wasm` (self-contained module, does not fall
  through to a native link step)

CI is green across all 14 lanes: Linux and macOS (x64/ARM64) each cover
lite/XLA/CUDA-capable variants, Windows x64 ships lite, XLA, and CUDA, and
Windows ARM64 ships lite and XLA. Windows ARM64 CUDA is not offered because
NVIDIA does not provide the required toolkit for that target
(*README.md §Platform*).

### Positioning

The language occupies a region not covered by the established alternatives.

- **Compared with other Scheme implementations** (Racket, Chez, Chicken): Eshkol
  compiles to native code through LLVM rather than to bytecode or via a
  source-to-C transform; it integrates automatic differentiation, GPU
  dispatch, and a neuro-symbolic stack at the compiler level rather than as
  optional libraries. As of v1.3.5-evolve its continuations are multi-shot and
  re-entrant on all three engines, so a captured continuation can be invoked any
  number of times and from any dynamic extent, including after the capturing
  procedure has returned — the property Racket and Chez have and a compiled
  setjmp/longjmp implementation has to earn, here by giving a capture that may
  outlive its frame a durable copy of the live C stack restored to the same
  addresses.
- **Compared with AD-first systems** (Julia + Zygote, Python + JAX or PyTorch):
  Eshkol's AD is integrated into the compiler at the IR level rather than
  obtained through tracing, source-to-source rewriting, or operator
  overloading, and — since v1.3.0-evolve — computes exact derivatives at any
  order via Taylor towers, which JAX's `jax.experimental.jet` approaches only
  numerically; as of v1.3.3-evolve those exact gradients also flow correctly
  through the `input2` path (kernel/gamma/K/V) for first-class losses, not
  just literal-loss compile-time-known functions. The host runtime has no
  garbage collector; allocation is bounded by the arena reset boundary, and
  as of v1.3.1-evolve self-tail-recursive loops reclaim their arena scope
  automatically per iteration — a region-escape evacuator, completed in
  v1.3.3-evolve, promotes any heap object that escapes that scope (including
  logic/workspace state and, as of this release, a `delay`d promise) rather
  than leaving it dangling — which makes Eshkol viable in contexts (real-time
  control loops, embedded inference, long-running daemons) where Python or
  Julia's GC pauses, or unbounded process RSS growth, are disqualifying.

Each comparison is grounded in *docs/breakdown/OVERVIEW.md §Comparison with
other languages*; the document explicitly enumerates where Eshkol gives less
than the alternative (mature library ecosystem, GUI toolkits).

---

## Memory model: OALR

Ownership-Aware Lexical Regions replace garbage collection. The model
consists of:

- A single global arena allocator with an 8 KB minimum block size,
  bump-pointer allocation, and batch deallocation via arena reset.
- An 8-byte object header prepended to every heap object
  `{subtype:u8, flags:u8, ref_count:u16, size:u32}`. The header is at
  offset −8 from the data pointer returned by allocators.
- Twenty-four heap-subtype slots assigned through v1.3.1-evolve (slot 14
  remains reserved for a future `RULE` backward-chaining type; slot 23 is
  `HEAP_SUBTYPE_TAYLOR`, added in v1.3.0-evolve for the arbitrary-order
  Taylor-tower AD engine) and five callable subtypes, consolidating eight
  historical pointer types into two supertypes (`HEAP_PTR`, `CALLABLE`).
  Every subtype that carries interior pointers is deep-walked by the
  region-escape evacuator rather than shallow-copied on escape from a
  `with-region` scope, on native codegen and, as of v1.3.5-evolve, on the
  bytecode VM. See *inc/eshkol/eshkol.h §heap subtypes*.
- 16-byte tagged values laid out `{type:u8, flags:u8, reserved:u16,
  padding:u32, data:u64}`. When the compiler can prove the type at
  compile time it emits untagged LLVM IR, eliminating the tagging
  overhead entirely.
- A 512 MB main-thread stack via linker flags (`-Wl,-stack_size,0x20000000`
  on Darwin; `-Wl,-z,stacksize=0x20000000` on Linux), runtime
  configurable via `ESHKOL_STACK_SIZE`. The default maximum
  recursion depth is 100,000 frames.
- Per-thread arenas (1 MB, lazily allocated through `thread_local`) for
  parallel workers; the global arena is used only to construct result
  lists after parallel tasks have completed (*docs/breakdown/PARALLEL_COMPUTING.md §2.1*).
- **Per-iteration arena-scope reclamation for self-tail-recursive loops**
  (extended in v1.3.1-evolve). A conservative static escape analysis
  (`namedLetIterScopeSafe`) proves a loop body's arena allocations don't
  escape across the tail-call back-edge; when it does, the loop's arena
  scope is reclaimed every iteration with zero source annotation. v1.3.0-evolve
  covered named-let loops; v1.3.1-evolve extended the same analysis to
  self-tail-recursive `define` loops and accepts a catch-all guard clause in
  the loop body. Verified on a 1,000,000-iteration loop: RSS goes from
  1,369 MB (unbounded growth) to 224 MB (flat). See
  *lib/backend/llvm_codegen.cpp*.
- **Region-escape evacuator.** When a value allocated inside a `with-region`
  block escapes that block (is returned, stored into an outer binding, or
  captured by a closure), a companion evacuator deep-walks it and promotes its
  interior pointers into the surviving arena before the region is popped. Every
  `HEAP_SUBTYPE_*` member carries an explicit deep-walk or leaf tag with its
  reasoning, including the exact `COEFF_RATIONAL` Taylor tower, whose
  coefficient array is walked because an overflowing coefficient is a pointer to
  an independently arena-allocated bignum; the architecture-model invariant that
  checks this is derived from the source rather than from a hand-typed list,
  resolved through a libclang semantic index of the real case arms.
  `ESHKOL_ARENA_POISON=1` poisons `arena_destroy`'d memory so any gap in that
  coverage crashes loudly instead of corrupting silently. See
  *lib/core/runtime_regions.cpp* and
  *tests/memory/region_evac_taylor_exact_test.esk*.
- **Region reclamation on the bytecode VM** (Stage 1, v1.3.5-evolve).
  `with-region` reclaims on the VM as well as on native codegen. The VM marks
  from its root set and sweeps at arena-block granularity rather than copying,
  because a VM value addresses the heap by a small integer index rather than by
  pointer — marking moves nothing, so `eq?` identity, shared structure, and
  cycles survive with no special handling, and live objects' fixed-size headers
  are copied one level out, which needs no layout knowledge because the object
  table is the only holder of that address. A compile-time-checked 33-row table
  classifies the full heap tag space, a fatal startup check requires every row
  to be filled in, and an unclassified subtype, a continuation captured inside a
  region, or a failed bookkeeping allocation all pin the region, so every
  uncertainty degrades toward a bounded leak and never toward a dangling index.
  Measured on one fixture swept by iteration count: 25, 26 and 27 MB at 1,000,
  4,000 and 16,000 iterations, against 793 MB with `ESHKOL_VM_REGION_EVAC=0`.
  Five runtime variables — `ESHKOL_VM_REGION_EVAC`, `_VERIFY`, `_VERIFY_FATAL`,
  `_COMPACT`, `_RECYCLE` — are documented in
  [environment-variables.md](../docs/reference/runtime/environment-variables.md),
  and `ESHKOL_ARENA_POISON` arms the VM's arena as well as native's. The
  user-reachable region handle surface (`region-open`/`region-close`) stays
  bookkeeping-only on the VM and announces that at the point of use; the lexical
  form landed first because its extent tells the teardown where the region ends.
  See *lib/backend/vm_region_evac.c* and
  [RUNTIME_CONFIGURATION.md](../docs/breakdown/RUNTIME_CONFIGURATION.md#bytecode-vm-region-reclamation).
  <!-- source: docs/breakdown/RUNTIME_CONFIGURATION.md:92-96; lib/backend/vm_region_evac.c:163 (VM_EVAC_TYPE_COUNT 33) -->
- **A continuation captured inside a region pins it, on both engines**
  (v1.3.5-evolve). Capturing a continuation while any region is open pins every
  currently-open region, and a pinned region's arena is promoted into its parent
  on pop rather than freed, so the failure direction is a bounded leak and never
  a dangling read. Pinning triggers only on an actual capture, so a program that
  never calls `call/cc` shows no behaviour change — confirmed by the unchanged
  flat-RSS gates.
- **Exception handler frames cost nothing in steady state** (v1.3.5-evolve).
  Handler frames come from a thread-local LIFO free list, malloc-backed rather
  than arena-backed, since an arena address can be retracted by a region or
  iter-scope rewind and handing it back would alias a fresh object. Total frame
  memory is bounded by peak nesting depth rather than by entry count, so a
  `guard` entered once per tick in a resident loop is free after the first.
  A long-run gate measures at 200,000 and at 1,600,000 ticks and gates on the
  slope rather than on a ceiling: transient garbage and all four
  persistent-mutation channels come back at exactly 0.000 bytes per tick, with
  identical byte totals at both horizons. `ESHKOL_ARENA_REPORT=1` prints the
  global arena's byte-exact allocation total at exit, because peak RSS is a
  high-water mark of instantaneous residency that reads low under memory
  pressure. See
  [memory-model.md](../docs/reference/runtime/memory-model.md).
  <!-- source: docs/reference/runtime/memory-model.md:308-313; tests/memory/resident_longrun_flat_gate.sh:77-79 -->
- Optional linear types via `(owned ...)`, `(borrow value body)`, and
  `(shared ...)`; the third activates reference counting against the
  header's 16-bit `ref_count` field (*README.md §Memory architecture*).

The model is what makes the system bit-reproducible. Two back-to-back release
builds produce byte-identical `build/stdlib.bc` and `build/eshkol-run`;
hash-map iteration order is keyed on stable strings from the AST and does
not reach the emitted IR (*docs/HARDENING.md §`#184` deterministic execution*).

---

## Numeric tower

The R7RS numeric tower is complete, with exactness tracked through a flags
byte on each tagged value.

- **int64**: 64-bit signed integers stored inline in the tagged value (immediate).
- **bignum**: Arbitrary-precision integers as `HEAP_PTR` with subtype 11.
  Automatic promotion on int64 overflow and demotion when the result fits
  in 64 bits again.
- **rational**: Exact fractions (subtype 19), always reduced via GCD. As of
  v1.3.3-evolve the representation is a canonical discriminated union: a
  zero-allocation int64 numerator/denominator fast path, with a bignum
  numerator/denominator pair taken only on overflow — so exact rationals no
  longer degrade to double at bignum magnitudes, and bignum-magnitude
  rational literals parse. Verified byte-identical against Python
  `Fraction`.
- **double**: IEEE 754 64-bit floats (inexact).
- **complex**: Heap-allocated `{real:f64, imag:f64}` with Smith's-formula
  division.

R7RS semantics hold for mixed arithmetic: exact + exact = exact, exact + inexact
= inexact. As of v1.3.0-evolve, exactness propagates through arbitrary-order
differentiation as well: `derivative-n`/`taylor` return exact bignum/rational
coefficients when the seed point is exact and the function only uses
exact-preserving operators, demoting to `double` on overflow or at the first
transcendental call. See *docs/DESIGN.md §Exact arithmetic* and
*lib/backend/arithmetic_codegen.cpp*.

---

## Automatic differentiation

Two orthogonal axes: **mode** (symbolic, forward, reverse) and, since
v1.3.0-evolve, **order** (arbitrary, via Taylor towers).

**Symbolic mode.** AST rewriting at compile time using twelve differentiation
rules. Zero runtime overhead when the function is syntactically known.

**Forward mode.** 16-byte dual numbers `{value:f64, derivative:f64}`
propagated through arithmetic, transcendentals, and activations. Suitable
for functions R → Rⁿ. The dual-number type has its own tag
(`ESHKOL_VALUE_DUAL_NUMBER = 6`) and is dispatched at the LLVM IR level.

**Reverse mode.** A computational graph with more than twenty AD node types
(elementary arithmetic, transcendentals, scalar utilities, neural-network
activations) recorded onto a Wengert tape during the forward pass. The tape
is topologically sorted and walked in reverse during gradient construction.
A 32-level tape stack enables nested gradients for Hessians, natural
gradient, and meta-learning constructions. See
*lib/backend/autodiff_codegen.cpp* and *lib/backend/tensor_backward.cpp*.

**Custom-VJP tape nodes (new in v1.3.3-evolve).** A new reverse-mode tape
node (`AD_NODE_CUSTOM`) carries an externally supplied vector-Jacobian
product, so a foreign/FFI computation with a known adjoint participates
exactly in reverse-mode AD rather than through a finite-difference
approximation. First user: Moonlab's VQE gradient is bridged into the tape,
making `(vqe-energy ...)` differentiable through ordinary `gradient` code —
the release gate requires the bridged adjoint to match Moonlab's native
adjoint to within `1e-8` and a central finite difference to within `1e-4`,
and a VQE custom-VJP probe now sits permanently in the AD adversarial
oracle.

**One-pass reverse gradients and AD introspection (staged-kernel Phase A,
new in v1.3.3-evolve).** The per-component gradient replay is collapsed into
one primal plus one reverse pass reading every input gradient from the tape
(N primal calls become one, verified at N=4 and N=64), with a runtime
mixed-record guard keeping mixed-mode reverse-over-forward on the proven
per-component path. New `(ad-counters)`/`(ad-primal-calls)` builtins expose
primal-call/reverse-pass/tape counters to Scheme.

**Arbitrary-order mode (Taylor towers, new in v1.3.0-evolve).** A closed-recurrence
engine (`lib/core/taylor_recurrences.def`, `lib/core/runtime_taylor.c`)
computes every derivative up to an arbitrary order `k` in one pass: `k+1`
coefficients and O(k²) work, versus the 2^k blow-up of nested dual numbers.
Delivered across thirteen gated phases, P0 through P12 (see
*docs/design/AD_TAYLOR_TOWER.md* and *docs/AD_CAMPAIGN.md*):

- `(taylor f x k)` / `(derivative-n f x k)` — the coefficient series or the
  scalar `k`-th derivative, for any `k`.
- Exact bignum/rational coefficients when the seed point is exact and the
  function uses only exact-preserving arithmetic (verified with 68
  exact-coefficient checks); automatic demotion to `double` on overflow or
  the first transcendental call.
- `taylor-model` / `tm-range` / `tm-eval` — a Taylor polynomial paired with
  a rigorous interval-remainder bound, for a provable range/value enclosure.
- `mixed-partial` / `gradient-n` — arbitrary-order mixed partials via a
  Griewank-Utke-Walther (GUW) directional-propagation layer.
- `sparse-hessian` / `sparse-mixed-partials` — sparse high-order recovery
  via greedy star-coloring graph recovery.
- `checkpointed-gradient` — a Griewank/binomial √N checkpoint schedule for
  high-order reverse-mode AD, holding at most one block's tape live at a
  time (measured peak-node ratio ≈1.8 at N=200 vs. ≈4.0 dense).
- `taylor-ode-solve`, `taylor-root`, `taylor-inverse-series` — numerical
  methods built directly on the tower (fixed-step order-`k` IVP solving,
  Householder-family root refinement, Lagrange-inversion series reversion).
- Towers are tensor-valued (`core.ad.tensor_tower`) and compose with
  `matmul`/`conv2d`/activations; they work correctly through
  `if`/`cond`/named-let/recursion and `map`/`fold`.
- Perturbation confusion is handled structurally: every differentiation
  context carries its own epoch tag in the tower's header.
- Zero heap allocation on the common path: when the order `k` is a
  compile-time literal, the whole tower unrolls into stack-allocated,
  branch-free SSA IR.

Eight vector-calculus operators (order ≤ 2) are language primitives:

```
derivative              (lambda (x) ...) -> R -> R
gradient                (lambda (v) ...) -> R^n -> R^n
jacobian                (lambda (v) ...) -> R^n -> R^{m x n}
hessian                 (lambda (v) ...) -> R^n -> R^{n x n}
divergence              vector field   F: R^n -> R^n      => R^n -> R
curl                    vector field   F: R^3 -> R^3      => R^3 -> R^3
laplacian               scalar field   f: R^n -> R        => R^n -> R
directional-derivative  f, point, direction               => R
```

The `vref` operator is AD-aware: during gradient computation it creates AD
nodes; outside that context it is a simple pointer dereference. This
context-sensitivity is achieved through runtime type inspection on closure
arguments (*docs/breakdown/OVERVIEW.md §Automatic differentiation*).

A note on costs as currently measured: forward mode incurs roughly a 2–3×
slowdown, reverse mode 3–5× with O(n) memory, symbolic mode zero runtime
overhead because the rewrite is at compile time (*README.md §Autodiff
overhead*). The Taylor-tower engine is O(k²) in the requested order `k` and
zero-heap when `k` is a compile-time literal.

**Reverse-mode tensor-op gradients (`input2`).** `gradient` on `conv2d`,
`batchnorm`, `layernorm`, and `attention` propagates an exact gradient to the
second operand (kernel / gamma-beta / K-V) in both the literal-loss form (a
compile-time-known `Function*`) and the first-class form (a loss value selected
or constructed at runtime). Batch-norm and layer-norm gamma/beta are
differentiated per-feature rather than as one scalar, and any unsupported
tensor-op backward path is an explicit error rather than a silent zero.
Finite-difference-verified across matmul, conv2d, attention K-V, and vector
gamma in both forms.

**Dispatch closed by construction (new in v1.3.5-evolve).** `ad_node_type_t`,
`callable_subtype_t`, and `EvacKind` are generated from a single declaration
registry (*inc/eshkol/ad_node_registry.def*) with no `default:` arm, and
`-Werror=switch` with `-Werror=switch-enum` makes an unhandled member a compile
error rather than a plausible answer; an ICC invariant re-derives each enum's
members from its own definition so the guarantee survives on toolchains where
the compiler flag alone cannot enforce it. A registry row naming a backward
function that does not exist is itself a compile error, which is what makes a
row's claim to be registered mean registered. Open sets stay loud on purpose: a
subtype byte read out of an object header is untrusted input, so it is split off
with a value-naming fallback, and the VM's opcode and value-type switches, which
dispatch on bytecode rather than on the enum, are left as loud backstops. The
registry currently holds 18 bridged rows against 4 node types no forward
produces yet.
<!-- source: inc/eshkol/ad_node_registry.def (18 BRIDGE, 4 UNREGISTERED rows); tests/backend/exhaustive_dispatch_test.cpp (11 checks) -->

**Exact backwards for the geometric bridge (new in v1.3.5-evolve).** Hyperbolic
distance, the Poincaré exponential and logarithmic maps, and geodesic attention
carry exact closed-form backward rules, each declared as a bridged row in that
registry. The exp and log rules reuse the Möbius-addition and log-map Jacobians
the Fréchet rule already differentiates rather than re-deriving them, since the
log map *is* the function that rule differentiates and a second derivation could
only introduce a disagreement. Validated against golden Jacobians from an
independently written Eshkol transcription of the same formulas (agreement to
3.7e-16 and 1.1e-14) and against two derivation-independent identities — the
conformal gradient-norm identity and the inverse-Jacobian identity, at maximum
relative deviations of 5.0e-16 and 6.7e-16 — before finite differences are
consulted at all. The distance is not differentiable at coincident points, and
geodesic attention is therefore not differentiable when a query row equals a key
row exactly; both refuse loudly, naming the offending index, rather than picking
a plausible subgradient. Gradcheck pinned by count at 13 checks.
<!-- source: .icc/silent-wrong-ledger.yaml SW-65 evidence block; tests/bridge/qllm_bridge_geometric_gradcheck_test.cpp -->

**Forward producers for the tensor-embedding and Fréchet-mean nodes (new in
v1.3.5-evolve).** `ad_tensor_embedding` and `ad_frechet_mean` record real AD
nodes through the real dispatch path, so those backward rules are exercised by
the producer that fills their contract rather than by hand-assembled fixtures
written from the same contract the rule reads. Fractional, negative, and
out-of-range embedding indices are refused at record time rather than rounded or
clamped into a wrong row, and the Fréchet forward shares its Karcher iteration
with the VM's own opcode (extracted into
*inc/eshkol/backend/frechet_mean_core.h*) so forward and backward cannot
disagree about what "converged" means. Gradchecked against exact analytic
references: exact scatter-add for the embedding with 0 mismatches, the exact
Euclidean closed form for the Fréchet mean at 0.0, and a hyperbolic finite
difference of 8.3e-10 over 48 partials.
<!-- source: CHANGELOG.md v1.3.5-evolve (#497); tests/bridge/qllm_bridge_producer_gradcheck_test.cpp -->

**A no-finite-differences guarantee that can fail, and a structural exactness
gate (new in v1.3.5-evolve).** The counter behind `(ad-finite-difference-evals)`
has a real writer on the one central-difference backward the tape defines,
reported through the zero-arity builtin `(ad-note-finite-difference!)` on native
and on the VM alike, and the exactness gate runs a positive case beside a
negative control — a difference quotient deliberately planted in the gradient
path — on JIT, AOT, and the VM. Separately,
`.icc/ad-carrier-manifest.yaml` declares, per operator and per engine, which
differentiation carrier answers it and whether it is exact, and
`scripts/gate_ad_shared_node_model.py` re-derives each declaration by extracting
and classifying the actual `case` body in the emitted sources, so a declaration
cannot be laundered through a helper. Seven checks, including the requirement
that a `vm-supported` row in the VM parity manifest declares its carrier and
that the declared carrier equals the one the source is observed to use. An
output differential can only compare what two carriers compute, never which
carrier computed it; this is the gate that closes that gap.

See the [Automatic Differentiation guide](../docs/guide/AUTOMATIC_DIFFERENTIATION.md)
for a worked, example-verified walkthrough of all thirteen phases, and
[`docs/reference/ad/INDEX.md`](../docs/reference/ad/INDEX.md) for the API
reference.

---

## Neuro-symbolic stack

Twenty-two compiler builtins implement three theoretical frameworks as
first-class language operations rather than library calls.

**Logic programming (Robinson's resolution, 1965)**

```
unify  walk  make-substitution  make-fact  make-kb
kb-assert!  kb-query
logic-var?  substitution?  kb?  fact?
```

Logic variables use the `?x` syntax, which the parser transforms into
`ESHKOL_LOGIC_VAR_OP` AST nodes. The leading `?` is a valid R7RS identifier
start character, so the syntax requires no grammar change. Implementation:
*lib/core/logic.cpp*.

**Active inference (Friston's free-energy principle, 2010)**

```
make-factor-graph  fg-add-factor!  fg-infer!  fg-update-cpt!
free-energy  expected-free-energy
factor-graph?
```

The runtime supports belief propagation, CPT updates (which enable real
learning by mutating the CPT and resetting messages so beliefs reconverge),
and both variational free energy and expected free energy. Implementation:
*lib/core/inference.cpp*.

**Global workspace theory (Baars 1988; Bengio 2017 computational formulation)**

```
make-workspace  ws-register!  ws-step!  workspace?
```

`ws-step!` is fully implemented end-to-end: the LLVM codegen loop calls
registered closures via the closure dispatcher, and C runtime helpers
(`eshkol_ws_make_content_tensor`, `eshkol_ws_step_finalize`) handle
content-tensor wrapping and softmax broadcast.
Implementation: *lib/core/workspace.cpp*.

Heap subtypes assigned to these objects:
`HEAP_SUBTYPE_SUBSTITUTION = 12`, `HEAP_SUBTYPE_FACT = 13`,
`HEAP_SUBTYPE_KNOWLEDGE_BASE = 15`, `HEAP_SUBTYPE_FACTOR_GRAPH = 16`,
`HEAP_SUBTYPE_WORKSPACE = 17`. Type tag `ESHKOL_VALUE_LOGIC_VAR = 10`.
See *inc/eshkol/eshkol.h §heap subtypes*.

---

## The SDNC artefact

The repository ships a three-way verification harness for the SDNC paper:

1. **Reference C interpreter** — a direct switch over the eighty-three opcodes;
   the ground truth.
2. **Simulated transformer** — C functions that mirror the six layers
   (Gaussian-attention instruction fetch, polarisation-identity product,
   address-resolution preprocessing, gated opcode dispatch, tape write +
   parent load, backward gradient dispatch + write-back).
3. **Matrix-based forward pass** — explicit weight matrices generated by
   `generate_weights`, applied via the gated FFN formula at each layer:
   y = W_down · (σ(W_g · x + b_g) ⊙ (W_u · x + b_u)) + b_d.

Agreement across all three modes constitutes the verification chain.
Coverage of the 256-dimensional state vector is field-wise (PC, SP, TOS, SOS,
registers, arena cells, tape, flags); the current artefact reports
123 of 123 traced programs agreeing on the final output and 123 of 123
agreeing at every intermediate step. The strict weight artefact covers
82 of 83 canonical opcodes; the one remaining opcode is `OP_NATIVE_CALL`,
the deliberate external boundary for host-runtime services.

Reproduction is one command:

```bash
scripts/paper/run_paper_suite.sh
```

Expected wall time on a 2023 M2 Max is under five minutes. Outputs land in
`artifacts/paper/outputs/` and the harness prints stable SHA-256 hashes:

```
SHA-256  weights.qlmw              381599e7a5607b4047ede0d6c8e6d270cb81dbdebfdb0bf0c0eba38758aa3f0c
SHA-256  vm-traces.jsonl           4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  transformer-traces.jsonl  4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  comparison-report.json    80aa6fed4db40bca521217ae8777677173fe7eeb239baa69847111e7ac674105
SHA-256  opcode-coverage.json      152a4bacc483d8985abeb08bc0d44112144f536ed663274bc7b1eeccbdd2dfe4
```

The transformer trace and the VM trace share a hash because they agree
bitwise. Platform divergence is treated as a bug; issue reports should
include the CPU, libc, and floating-point environment.

Implementation note: the paper proves the gated indicator function is exact
in float32 for any scale S > 33.2; the artefact ships with S = 300 (rather
than the working constant S = 100) because at S = 100 the softmax score gap
between the peak position and its neighbours is ≈ 35.4, leaving a residue
e⁻³⁵·⁴ ≈ 4.6 × 10⁻¹⁶ that accumulates as `tos = 4.4e-16` at step 1206 of
`tail sum(100)` versus exactly zero in the reference. Raising S to 300
pushes the gap above 87 so e⁻ᵍᵃᵖ underflows to literal float32 zero. See
*lib/backend/weight_matrices.c:59-84* and *docs/SDNC.md §float32 saturation margin*.

---

## Parallelism

The scheduler is a per-worker Chase-Lev work-stealing deque
(*Dynamic Circular Work-Stealing Deque*, Chase and Lev, 2005) with
epoch-based reclamation, three-stage idle backoff (spin / yield / sleep),
and hardware-aware sizing (`std::thread::hardware_concurrency()`,
override via `ESHKOL_NUM_THREADS`). See
*inc/eshkol/backend/work_stealing_deque.h* (documented in v1.3.1-evolve) and
*lib/backend/thread_pool.cpp*.

Primitives: `parallel-map`, `parallel-fold`, `parallel-filter`,
`parallel-for-each`, `future`, `force`, `future-ready?`.

A void-pointer ABI boundary keeps tagged values from being passed by value
across the C/LLVM boundary, so the representation does not depend on the
optimisation level or on the target's aggregate-passing rules. All tagged-value
construction and destruction occurs within LLVM IR; only `void*` crosses into
C.

Measured 4–12× speed-up of `parallel-map` on 24 cores
(*docs/breakdown/ROADMAP.md §1.1-accelerate completed*; the underlying
root-cause fix is recorded in project memory as the parallel-map
flags-byte fix: worker tagged-value flags were hardcoded to zero, which
mis-dispatched into the bignum path; packing `{type, flags}` into
`item_type:i64` with the default flipped on restored real parallelism
across both AOT and JIT paths). `eshkol_runtime_shutdown()` stops and joins the global parallel thread pool
before running shutdown hooks, so a graceful `SIGTERM` tears the pool down in a
defined order.

---

## GPU acceleration

Adaptive dispatch through *lib/backend/blas_backend.cpp*. Calibration
constants (measured on Apple Silicon):

| Backend | Peak | Overhead | Dispatch range |
|:---|---:|---:|:---|
| SIMD (vectorised) | 25 GFLOPS | ~0 | ≤ 16 elements |
| cBLAS (Apple Accelerate / AMX) | 1,100 GFLOPS | 5 µs | 17 to ~10⁹ elements |
| Metal GPU (SF64 software float64) | 200 GFLOPS | 200 µs | > 10⁹ elements |

SF64 (Software Float64) emulates double precision using double-double
arithmetic — two 32-bit mantissas combined for an effective precision of
roughly 100 bits — because Metal GPUs lack native float64. Implementation:
*lib/backend/gpu/metal_softfloat.h* and *lib/backend/gpu/gpu_memory.mm*.
CUDA dispatches through cuBLAS on NVIDIA.

The cost model selects the backend per operation. The defaults are
empirically calibrated and configurable through `ESHKOL_GPU_PRECISION`,
`ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_GPU_PEAK_GFLOPS`.

As of v1.3.5-evolve the GPU correctness gate is a gate. Every
`tests/gpu/*.esk` file aggregates a failure counter and exits on an explicit
`PASS:`/`FAIL:` verdict line; the test isolation layer fails a test that exits 0
without printing a recognized marker; and a permanent, deliberately-failing
canary (*tests/gpu/gate_canary_must_fail.esk*) runs on every invocation and is
required to fail, forcing the whole run red if it is ever not red. On the
strength of a measured Metal-versus-CPU divergence of exactly 0 across ten
probes, `GPU_GATE_TOL` tightens from `1e-4` to `1e-9`.
<!-- source: tests/gpu/gpu_correctness_gate.sh:71 (GPU_GATE_TOL default 1e-9); tests/gpu/gate_canary_must_fail.esk; scripts/run_gpu_tests.sh:186-223 -->

---

## Tensor and ML framework

Compiler-level tensor operations span more than a dozen domain-specific
codegen modules plus the dispatcher in *lib/backend/tensor_codegen.cpp*.
Coverage:

- 16 activations (relu, relu6, sigmoid, tanh, gelu, swish, mish,
  softmax, log-softmax, softplus, softsign, leaky-relu, prelu, elu,
  selu, celu)
- 14 loss functions (mse-loss, mae-loss, cross-entropy-loss, bce-loss,
  huber-loss, kl-div-loss, hinge-loss, smooth-l1-loss, focal-loss,
  triplet-loss, contrastive-loss, label-smoothing-loss,
  cosine-embedding-loss)
- 5 optimisers + 3 gradient utilities (sgd-step, adam-step, adamw-step,
  rmsprop-step, adagrad-step; zero-grad!, clip-grad-norm!, check-grad-health)
- 5 weight initialisers (xavier-uniform!, xavier-normal!, kaiming-uniform!,
  kaiming-normal!, lecun-normal!)
- 4 learning-rate schedulers (linear-warmup-lr, step-decay-lr,
  exponential-decay-lr, cosine-annealing-lr)
- 7 CNN layers (conv1d, conv2d, conv3d, max-pool2d, avg-pool2d,
  batch-norm, layer-norm)
- 8 transformer ops (scaled-dot-attention, multi-head-attention,
  positional-encoding, rotary-embedding, causal-mask, padding-mask,
  feed-forward, embedding)
- 6 data-loading ops (make-dataloader, dataloader-next, dataloader-reset!,
  dataloader-length, dataloader-has-next?, train-test-split)
- `agent.quantum` builtins (opt-in, built with `-DESHKOL_QUANTUM_ENABLED=ON`):
  state creation/teardown, Hadamard/Pauli/CNOT/rotation gates,
  `measure`/`expectation-z`, H2/LiH/H2O molecular Hamiltonians, VQE
  energy/gradient/optimise, the `bell-chsh` CHSH experiment, and
  `with-quantum-state`/`with-hamiltonian` helpers — plus `agent.pqc`'s
  ML-KEM `mlkem-keygen`/`mlkem-encaps`/`mlkem-decaps` builtins

All ML builtins integrate with reverse-mode AD: calling `gradient` on any
composition produces an exact gradient, and — since v1.3.0-evolve —
tensor-valued Taylor towers (`core.ad.tensor_tower`) extend arbitrary-order
differentiation through `matmul`/`conv2d`/activations as well. As of
v1.3.3-evolve that includes the quantum VQE energy: Moonlab's exact adjoint
gradient enters the reverse tape through a custom-VJP node, so a variational
quantum circuit differentiates like any other composition. The conv2d
backward pass uses stride-based scatter/gather indexing that doesn't map
cleanly to GEMM, and LayerNorm/BatchNorm backward are inherently sequential
reductions; see *docs/KNOWN_ISSUES.md* for the current, itemized state of
these ML-kernel performance characteristics.

---

## Agent FFI

A native FFI surface for systems programming. Each backend is implemented in
C and exposed to Eshkol through tagged-value calling conventions.

- **HTTP client** — libcurl-backed, per-thread easy interface, TLS feature
  check. Implementation: *lib/agent/c/agent_http_client.c*.
- **SQLite** — `sqlite3_open_v2`, prepared-statement bindings, dynamic-size
  column-text retrieval. Implementation: *lib/agent/c/agent_sqlite.c*.
- **Subprocess** — `posix_spawn` with argv arrays. The `popen("sh -c …")`
  path was removed in v1.2 (security advisory `#190`); the `-argv`
  variants (`process-spawn-argv`, `run-argv`, `run-argv-capture`) are
  the recommended interface for any command built from external input.
  Implementation: *lib/agent/c/agent_subprocess.c*.
- **Filesystem watch** — kqueue on macOS, inotify on Linux. Implementation:
  *lib/agent/c/agent_watch.c*.
- **Crypto, regex, terminal, compression, etc.** — additional native
  backends in *lib/agent/c/*, each with a corresponding `.esk` wrapper
  in *lib/agent/*.

AOT linking is automatic: `ESHKOL_HOST_AGENT_FFI_LINK_ARGS` in the build
config is consulted, and the AST is scanned pre-process for require
declarations so AOT binaries link the HTTP, SQLite, and subprocess
backends without the user having to specify library flags. v1.3.1-evolve
added Doxygen documentation across every agent-FFI implementation file in
*lib/agent/c/*.

---

## Tooling

- **eshkol-run** — production AOT compiler with executable, object file
  (`-c -o`), shared library (`--shared-lib`), and WebAssembly (`--wasm`)
  output modes; supports JIT execution (`-r`).
- **eshkol-repl** — interactive REPL via LLVM OrcJIT, documented as part of
  the v1.3.1-evolve implementation doc-comment pass. Preloads stdlib functions
  and globals from precompiled `.o` and `.bc` metadata. The `--machine` mode
  emits `EREPL READY` / `DONE` / `FAIL` framing on stderr for warm-worker
  IPC.
- **eshkol-doc** — API reference generator, added in v1.3.2-evolve. Harvests
  Doxygen `/** @brief */` comments from `inc/` and `lib/` and generates
  `docs/api/` (Markdown pages plus an HTML index).
- **eshkol-pkg** — package manager. TOML manifests, git-based
  registry, recursive submodule discovery. Commands: `init`, `build`,
  `run`, `add`, `clean`.
- **eshkol-lsp** — Language Server Protocol. Completions,
  hover, go-to-definition, diagnostics, formatting.
- **VS Code extension** — syntax highlighting, LSP integration, build tasks.
  Source: *tools/vscode-eshkol/*.
- **CMake package discovery** — `cmake/FindEshkol.cmake` ships as the one
  canonical module, added in v1.3.5-evolve. `find_package(Eshkol)` resolves the
  compiler, the runtime archive a compiled program needs, and the stdlib object
  and module directory, producing an `Eshkol::eshkol` imported target whose link
  interface unconditionally includes `stdlib.o` plus, on Apple, the system
  frameworks the runtime needs. A from-scratch consumer project under
  *tests/integration/system_package/* is run against a staged package by the
  package manifest, so what is checked is that the contract works and not merely
  that its files landed. Scoped to macOS and Linux.
- **Public benchmarks** — `bench/run_public_benchmarks.sh`, added in
  v1.3.5-evolve. One command from a clean checkout measures exact-AD cost
  curves, Ozaki-II CRT exact f64 GEMM, flat RSS under resident load, and
  differentiable quantum kernels, emitting machine-readable JSON alongside a
  human-readable table. Methodology and an explicit not-benchmarked list are in
  *bench/README.md*; a per-axis disk cap keeps a long run bounded.
  <!-- source: bench/run_public_benchmarks.sh:107-110; bench/README.md -->
- **Inter-Component Communication (ICC)** — agent-FFI-ready oracle and
  pytest-format smoke harness under `.icc/`; native Eshkol-aware (accepts
  Eshkol VM step / halt records as `eshkol_vm_step` / `eshkol_vm_halt`
  events; recognises `runtime_event` compact dict form and explicit
  `kind` JSON records). Region-evacuator poison coverage, the `input2` gradient
  gate, the multi-oracle differential harness, and the AD-versus-finite-difference
  adversarial oracle are permanent release gates. As of v1.3.5-evolve every
  finished trace is mirrored unconditionally into the directory `icc readiness`
  reads, so the oracle is reachable from an ordinary pull request; the
  `v1.4-connection` target carries one criterion per named deliverable, each
  bound to a real harness; and v1.3.5-evolve has a completion-oracle target of
  its own, so "all of v1.3.5" is a machine-checkable claim rather than a
  narrative one.
  <!-- source: .icc/completion-oracles.yaml (target v1.3.5-evolve, 24 criteria) -->

---

## Documentation

Documentation is a first-class artefact of the project rather than a
by-product:

- **Public C-API headers.** Doxygen-format documentation across 50 of the
  64 public headers under `inc/eshkol/` — backend codegen, runtime core,
  the type system, the XLA backend, subprocess/macro-expander/qLLM-bridge
  surfaces, the thread pool and work-stealing deque, the logger, model I/O,
  platform runtime, and runtime exports.
- **Implementation doc-comments** across the implementation files under
  `lib/` — agent FFI, the type checker, the parser, the REPL, core
  non-runtime modules, the quantum RNG, and the FFI bridges.
- **Navigable reference index.** A per-subsystem documentation index at
  [`docs/reference/language/`](../docs/reference/language/INDEX.md),
  [`ad/`](../docs/reference/ad/INDEX.md),
  [`runtime/`](../docs/reference/runtime/INDEX.md),
  [`tensors/`](../docs/reference/tensors/INDEX.md),
  [`stdlib/`](../docs/reference/stdlib/INDEX.md), and
  [`agent/`](../docs/reference/agent/INDEX.md), each an example-verified
  index into the corresponding function and syntax reference, linked from
  *README.md §Documentation*. v1.3.5-evolve adds
  [`bindings/python.md`](../docs/reference/bindings/python.md), documenting
  the `Context.eval`/`derivative`/`gradient` API and the lifetime guarantee
  a NumPy array exported from a tensor `eval()` carries, and
  [`language/continuations.md`](../docs/reference/language/continuations.md),
  documenting the multi-shot surface and its stated scope.
- **Generated API reference (`eshkol-doc`).** Those comments are harvested
  automatically into `docs/api/` rather than requiring a hand-maintained
  index.
- **A documentation-truth ratchet.** `scripts/check_surface_counts.py` reads
  the canonical surface and builtin totals from the coverage policy files and
  fails on a mismatch against every registered doc, red-proofed by planting a
  stale claim and confirming the gate catches it; it runs in CI's assurance
  job. The canonical totals it enforces are a 1,108-construct language surface
  and 1,042 builtins.
  <!-- source: tests/coverage/coverage_policy.json (baseline_surface_total 1108); tests/coverage/language_surface.json (counts.builtins_total 1042); scripts/check_surface_counts.py -->

---

## Hardening and robustness posture

The organising principle across this release line is that a wrong answer must
not be able to look like a right one, and that every assurance claim must be
falsifiable by something the repository can run.

**v1.3.5-evolve (current release).** The assurance surface this cycle is aimed
at the gates themselves.

- **A gate that cannot go red is treated as no gate.** A permanent,
  deliberately-failing canary runs on every GPU-suite invocation and is required
  to fail; if it ever comes back green the whole run goes red. The test
  isolation layer fails a test that exits 0 without printing a recognized
  verdict marker, every `tests/gpu/*.esk` file aggregates a failure counter and
  exits on an explicit `PASS:`/`FAIL:` verdict line, and the Windows judge
  applies the same canary inversion and evaluates every marker regex in
  multiline mode at the one place those checks are made.
- **Assurance wave 1.** Ledger-integrity and oracle-schema gates fail on a parse
  error, a duplicate identifier, a missing required field, or a structurally
  invalid criterion, and always report declared-versus-graded criteria counts per
  oracle. All three ship a `--self-test` mode, are wired into `ctest`, and run in
  a dedicated CI job.
- **Assurance wave 2.** A self-verdict scanner fails a PASS-graded artifact
  whose own text still reports a failure — including in manifest mode over every
  VM-parity corpus output graded PASS by native/VM agreement, because comparing
  two engines cannot see them print the identical self-reported failure line and
  call that equal. Build-fingerprint checks record and check the compiler
  binary's size, mtime, and SHA-256 alongside the checkout's git SHA, and fail
  when a built binary predates its most recent build-relevant source change. A
  results-schema check gives ICC real execution evidence for the doc-example
  harness. An adversarial scenario suite exercises the gates under a dirty
  worktree, a stale or rebuilt binary, a model-server outage, disk pressure, an
  actually failing gate, and the requirement that every gate's `--self-test`
  contract is wired somewhere real.
- **Encoding checked at the byte level.** A gate reads every tracked
  `*.ps1`/`*.psm1` file's own bytes and fails any that carry a byte at or above
  `0x80` without a UTF-8 byte-order mark, reporting the exact `file:line:col`
  and codepoint. Running the same bytes under a different default encoding is
  precisely what hides this class of problem, so execution could not be the
  check.
- **Oracle severities cannot let absent evidence read as ready.** Every target
  asserting a correctness or capability claim carries a high-severity criterion,
  four targets keep an explicit and commented advisory exception, and an audit
  gate fails if another target regresses into the shape where missing evidence
  grades as a warning. Evidence staleness is a repo-side gate with a real
  no-data verdict distinct from pass.
- **Pillar harnesses are armed and machine-reachable.** Every finished trace is
  mirrored unconditionally into the directory `icc readiness` reads. A
  `pillars-fast` job runs the cheap gates on every pull request — depth
  coverage, the ADR-0009 DBSP acceptance gate, the monotone-equivalence Taylor
  gate, the AD validated-bounds gate, and VM parity — and a nightly workflow
  runs the expensive sweeps: depth-parametric, differential, edge-matrix,
  metamorphic, sanitizer-fuzz, the full smoke battery, and SICP. Every
  trace-emitting harness shares a PASS/FAIL/INFRA/SKIP vocabulary with a real
  fork-based timeout and a retry-once helper, so an infrastructure timeout
  cannot publish itself as a code defect.
- **The leak-detection lane reports what it finds.** LeakSanitizer's verdict
  reaches the process exit status on every workload the project ships, including
  the REPL, whose fast-exit path performs an explicit whole-process leak check
  before exiting.
- **Machine-checked structure.** `icc architecture-verify` resolves switch
  dispatch through a libclang C/C++ semantic index rather than by token
  matching, so an invariant is checked against the case arms the compiler sees;
  the key-space-equality invariants are derived from source patterns rather than
  hand-copied enumerations.

**v1.3.4-evolve.** A consumer-hardening correctness wave with the same
organising principle: an emitted compile-time error prevents artifact emission
and execution; exactness is decided from an operand's runtime tag rather than a
result's value shape on both engines; differentiation is exact at exact points
and survives per-iteration nursery reclamation at any gradient arity;
`define-library` and `import` resolve same-unit libraries on all three back
ends; and `--shared-lib` links a real, C-ABI-correct shared library.

**v1.3.3-evolve.** Two generative exposure engines joined the release oracle
permanently:

- **Generative multi-oracle differential harness**: deterministically grown
  R7RS-subset programs cross-checked against chibi-scheme, the JIT, AOT at
  O0/O2, and the bytecode VM, plus metamorphic invariants.
- **Generative AD-versus-finite-difference adversarial oracle**: 147 probes and
  436 component checks across 21 generated files under JIT and AOT, where a zero
  AD gradient at a point where finite differences are nonzero is a hard failure.
- **Self-tail recursion in `cond`/`case`/`when`/`unless`/`and`/`or`** compiles
  to real loop back-edges, verified to 2,000,000 iterations under JIT and AOT.
  <!-- source: tests/tco/cond_case_tail_test.esk (N = 2000000) -->
- **Scalable, stable `sort`/`filter`**: accumulator tail loops and a stable
  bottom-up vector merge sort, 2M elements at about 362 MiB peak.
- **Region-escape evacuation covering `PROMISE`**, closing the ESH-0214 series,
  verified flat at about 116 MB under `ESHKOL_ARENA_POISON=1` over
  escape-then-force for both `delay` and `make-promise`.
- **ICC oracle hardening**: with quantum enabled, Bell-pair (200/200), CHSH
  (gate `2.4 < S <= 2.95`), VQE-versus-exact-energy, and ML-KEM NIST-KAT gates
  join the matrix.

**v1.3.2-evolve and v1.3.1-evolve.** Region-escape evacuation extended to the
logic and workspace subtypes, a thread-safe region scope stack under
`parallel-map` and future callbacks, per-iteration arena-scope reclamation
extended from named-let loops to self-tail-recursive `define` loops including a
catch-all guard body (1,000,000-iteration loop measured flat), and an iterative
S-expression reader that reads a 20-million-element list without touching the
native stack per element.

**v1.3.0-evolve release gates** (green on the release SHA, and the base this
release line builds on): ICC readiness oracle 100/100, trace-verified; CI green
across every lane including windows-arm64 lite/CUDA/XLA; SICP full-book gate
88/88 probes across all five chapters under both `-r` and AOT
(`scripts/run_sicp_smoke.sh`); reference-Scheme differential oracle 34/34 AGREE
against chibi-scheme 0.12.0 on the portable corpus
(`scripts/run_reference_differential.sh`).

**Permanent adversarial-testing infrastructure**, shipped in v1.3.0-evolve and
wired into the ICC release oracle rather than run once and discarded: a
multi-path differential harness with a seeded fuzzer, a feature-pair edge
matrix, an AD finite-difference oracle, a stress harness with RSS and time
budgets, a VM-parity ratchet, depth-parametric sweeps, and the external
reference-Scheme differential oracle. See *docs/TESTING.md*.

**Security posture.** The full audit findings table — every item, its severity,
and its resolution — is published in *docs/HARDENING.md*, and the disclosure
process and supported-version table are in *SECURITY.md*. The subprocess surface
is `posix_spawn` with argv arrays; the recommended interface for any command
built from external input is the `-argv` family.

---
## Web platform

Eshkol compiles to WebAssembly via `eshkol-run --wasm`, producing a
self-contained module that does not fall through to a native link step.
`--wasm` output dead-strips unused stdlib: a small program that genuinely uses
the stdlib emits a 60 KB module carrying 21 functions, with first-class
functions and native homoiconic display preserved, and a CI size gate holds the
bound.

The project website at https://eshkol.ai is itself an Eshkol program,
compiled to WebAssembly and served by GitHub Pages. The site embeds a
browser REPL where forward-mode automatic differentiation
(`(derivative (lambda (x) (* x x)) 3.0)` returning 6.0) runs through the
bytecode-VM dual-number propagation path without native code. The
interactive textbook has every example runnable in-browser.

The browser REPL uses the bytecode VM rather than LLVM JIT: an
opcode-dispatch register-plus-stack interpreter with 250+ native call IDs,
ESKB binary format with LEB128 encoding and CRC32 checksums
(*docs/DESIGN.md §Dual backend architecture*).

---

## Dual backend architecture

Two production execution backends share the same language semantics with
independent value representations:

- **LLVM native** (primary). 16-byte tagged values, roughly thirty codegen
  modules, the default for `eshkol-run`.
- **Bytecode VM** (*lib/backend/eshkol_vm.c* plus its *vm_\*.c* modules). A
  register-plus-stack interpreter with 250+ native call IDs, ESKB binary file
  format (section-based, LEB128, CRC32). Invoked via `eshkol-run input.esk -B
  output.eskb`. Coverage: arithmetic, closures, multi-shot continuations,
  exception handling, tensors, complex / rational / bignum, logic / inference /
  workspace, hash tables, bytevectors, parameters, I/O.
  `make-parameter`/`parameterize` are genuine runtime dynamic parameter objects
  — converters, a dynamic binding stack, correct unwinding — on both the native
  and VM execution paths. As of v1.3.5-evolve the VM reclaims region memory, and
  `with-region`'s teardown is reached identically by lexical exit, by a `raise`
  crossing the region, and by a continuation transfer out of it, so the
  structured and unstructured surfaces cannot drift apart.

The weight-matrix transformer (*lib/backend/weight_matrices.c*) is a third
execution surface that exists for the SDNC paper and
the qLLM/transformer weight-loading pipeline. The strict-artefact contract
is 123 of 123 traced programs verified three ways (reference interpreter =
simulated transformer = matrix-based forward pass). Exports use the QLMW
binary format for qLLM consumption.

---

## Standard library

Auto-loaded stdlib modules compiled to `build/stdlib.o` via `--shared-lib`.
Namespaces:

- `core.functional.*` — composition, currying, combinators
- `core.list.*` — higher-order list functions
- `core.data.*` — JSON, CSV, Base64
- `core.strings.*` — thirty-plus string utilities
- `core.ad.*` — Taylor-tower AD stdlib layer added in v1.3.0-evolve: `guw`
  (multivariate mixed partials), `tensor_tower` (tensor-valued towers),
  `taylor_models` (validated enclosures), `checkpoint` (checkpointed
  reverse-over-Taylor), `taylor_numerics` (ODE/root/series-inversion
  solvers), `sparse_guw` (sparse Hessian recovery), `interval` (interval
  arithmetic support)
- `math.*` — special functions (Bessel, Gamma, Beta), ODE solvers
  (Euler, RK4), root finding, statistics
- `signal.*` — Cooley-Tukey radix-2 DIT FFT, IFFT, Hamming / Hann /
  Blackman / Kaiser windows, FIR and IIR direct-form, Butterworth
  low-pass / high-pass / band-pass via the bilinear transform
- `ml.*` — Adam, AdamW, L-BFGS, conjugate gradient, learning-rate
  schedulers
- `random.*` — PRNG with explicit seeding (`seed-prng!`), per-stream
  isolation; as of v1.3.3-evolve `quantum-random` and friends draw from
  Moonlab's Bell-verified QRNG when the opt-in quantum build is enabled,
  and are honestly labeled as a classical fallback when it is not
  (`eshkol_qrng_source_label()` reports which source is active)
- `web.*` — WASM / DOM API, HTTP fetch
- `tensor.*` — shape manipulation, stacking, broadcasting helpers
- `core.blc` — Binary Lambda Calculus, added in v1.3.2-evolve. A pure-Eshkol
  implementation of John Tromp's BLC: De Bruijn-indexed terms as
  homoiconic s-expressions, `blc-encode`/`blc-decode` for Tromp's
  self-delimiting bit encoding, normal-order `blc-eval`, a decoded
  232-bit universal machine (`blc-U`), BLC8 byte I/O, and ASCII lambda
  diagrams. Loaded on demand via `(require core.blc)`. See
  *docs/guide/BINARY_LAMBDA_CALCULUS.md*.
- `core.dbsp` — incremental dataflow, added in v1.3.3-evolve: Z-sets
  (weighted multisets) as a commutative group, the `z^-1`/`D`/`I` stream
  operators, incremental relational operators (including the three-term
  incremental join and multiplicity-correct `distinct`), and the generic
  incrementalizer `Q^Δ = D ∘ lift(Q) ∘ I`, in pure Eshkol. Acceptance gate
  27/27 under JIT and AOT. See *lib/core/dbsp.esk*.
- `agent.quantum` — Moonlab quantum state-vector simulation, added in
  v1.3.3-evolve: state creation, gates, measurement, expectation values,
  VQE with molecular Hamiltonians, and the `bell-chsh` CHSH experiment.
  Opt-in: built when the compiler is configured with
  `-DESHKOL_QUANTUM_ENABLED=ON`. See *lib/agent/quantum.esk*.
- `agent.pqc` — ML-KEM (FIPS 203) post-quantum key encapsulation, added in
  v1.3.3-evolve: `mlkem-keygen`/`mlkem-encaps`/`mlkem-decaps` at the
  512/768/1024 security levels over R7RS bytevectors, NIST-KAT-verified.
  Opt-in via `ESHKOL_QUANTUM_ENABLED`, with honest not-enabled stubs
  otherwise. See *lib/agent/pqc.esk*.

The stdlib uses `LinkOnceODR` linkage so user redefinitions cleanly
shadow stdlib functions without the historical "duplicate symbol"
link errors.

---

## Availability

- **Repository**: https://github.com/tsotchke/eshkol
- **Website**: https://eshkol.ai
- **Browser REPL**: https://eshkol.ai/learn
- **Licence**: MIT
- **Build prerequisites**: CMake 3.14+, LLVM 21, a C17 + C++20 compiler
  (GCC 11+ or Clang 14+ — the toolchain the CI matrix builds with; AppleClang
  on macOS, LLVM 21 ClangCL on Windows), Ninja recommended.
- **Build**: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
- **Homebrew tap**: `brew tap tsotchke/eshkol && brew install eshkol`; the
  tap formula carries the computed release SHA-256 after tagging, and installs
  `FindEshkol.cmake` and `EshkolCompile.cmake` alongside the compiler.
- **Embedding from CMake**: `find_package(Eshkol)` then link
  `Eshkol::eshkol` — the imported target carries the runtime archive, the
  stdlib object, and the platform frameworks, with no hand-written library
  search in the consumer. macOS and Linux.

---

## Citation

```bibtex
@software{eshkol2026,
  title    = {Eshkol: A Programming Language for Mathematical Computing},
  author   = {tsotchke},
  version  = {1.3.5-evolve},
  year     = {2026},
  url      = {https://github.com/tsotchke/eshkol}
}
```

The SDNC paper is *The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction* (tsotchke, 2026); the
companion repository is `noesis` and the artefact is `artifacts/paper/`.

---

## Contact

Press, programme committees, and academic correspondence:
**team@tsotchke.org**.
Security disclosures: **security@eshkol.ai** (see *SECURITY.md* for the
disclosure process and the supported-version table; initial response within
3 business days, fix-or-mitigation plan within 14 days for HIGH and CRITICAL).
