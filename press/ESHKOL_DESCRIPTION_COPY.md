# Eshkol — Description Copy

## A compiled Scheme with a constructive proof that a transformer is an interpreter

In Eshkol v1.3.5-evolve you can call a procedure that has already returned.
That is not a metaphor. `call/cc` is multi-shot and re-entrant on all three
engines the project ships — native JIT, native AOT, and the bytecode VM. A
capture that may outlive its frame takes a durable copy of the live C stack and
restores it to the same addresses before resuming, so every frame pointer, every
spilled register, and every address of a local that a closure is holding stays
valid with no relocation; the VM snapshots its own operand stack and call-frame
array, and `dynamic-wind` reroots on both engines per R7RS 6.10. Six programs,
three engines, eighteen byte-exact transcripts.
<!-- source: tests/continuations/ (6 fixtures), scripts/run_continuation_tests.sh (runs each on -r, AOT, VM against tests/continuations/expected) -->

Eshkol is a compiled programming language for mathematical and cognitive
computing. The repository ships v1.3.5-evolve (August 2026) of the compiler.
Alongside re-entrant continuations, the bytecode VM gains its own region
evacuator, so `with-region` reclaims memory there the way it already does on
native codegen: the same fixture holds flat at 25-27 MB across 1,000, 4,000 and
16,000 iterations, against 793 MB on the identical binary with reclamation
switched off.
<!-- source: ROADMAP.md §v1.3.5 flagship; docs/breakdown/RUNTIME_CONFIGURATION.md#bytecode-vm-region-reclamation -->
Mutual tail recursion runs in constant stack space through every tail-position
spelling — `cond`, `case`, `when`, `unless`, and the last operand of `and`/`or`,
not only `if` — and a tail-transfer dispatcher extends that bound to calls whose
signatures differ and to targets that are not AArch64: 100,000,000 hops at
9.1 MB peak resident memory.
<!-- source: CHANGELOG.md §1.3.5-evolve, tail-transfer dispatcher (#483) -->
Differentiation dispatch is closed by construction: the AD node registry, the
callable subtypes, and the region evacuation kinds are generated from a single
declaration file with no `default:` arm, with `-Werror=switch-enum` making an
unhandled member a compile error rather than a plausible answer, and the four
geometric bridge operators — hyperbolic distance, the Poincaré exponential and
logarithmic maps, and geodesic attention — carry exact closed-form Jacobians
agreeing with independently derived golden Jacobians to 3.7e-16 and with two
derivation-independent identities to maximum relative deviations of 5.0e-16 and
6.7e-16, before a finite difference is consulted at all.
<!-- source: inc/eshkol/ad_node_registry.def; lib/bridge/tensor_backward.cpp; CHANGELOG.md §1.3.5-evolve (#500, #498) -->

The repository carries all of this together with the reproducibility artefact
for *The Self-Differentiating Neural Computer: Computable Transformers via
Analytical Weight Construction* (tsotchke, 2026), in which a six-layer
transformer with 12.22 million analytically-constructed parameters executes a
bounded 83-opcode bytecode VM bit-identically. The result is a constructive, not
statistical, demonstration that a fixed-weight transformer can be an interpreter
when its weights are derived from an instruction-set specification rather than
fit by gradient descent.

---

## Launch post

```
Eshkol v1.3.5-evolve is out now.

∂ is now exhaustive by construction: the compiler will not build a program
whose differentiation dispatch has a hole.

https://github.com/tsotchke/eshkol
```

## Long-form article title

`Eshkol v1.3.5-evolve: Multi-Shot Continuations in a Compiled Scheme`

---

## Lede

Eshkol is an R7RS-compatible Scheme dialect that compiles through LLVM 21 to
native binaries on macOS, Linux, and Windows, and to WebAssembly for browser
execution. The language treats automatic differentiation, arena memory, and a
neuro-symbolic computation layer as compiler primitives rather than library
add-ons. Differentiation is available in symbolic, forward, and reverse modes
alongside eight vector-calculus operators — and, as of v1.3.0-evolve, at
arbitrary order via a Taylor-tower engine that returns exact bignum/rational
derivatives when the math supports it. Memory is allocated through
Ownership-Aware Lexical Regions with deterministic, per-scope deallocation and
no garbage collector; as of v1.3.5-evolve that reclamation runs on the bytecode
VM as well as on native codegen. Continuations are multi-shot and re-entrant on
every engine. The consciousness engine exposes twenty-two builtins covering
unification, factor-graph belief propagation, free-energy minimisation, and
global-workspace softmax competition.

The flagship demonstration is the SDNC paper artefact: a single shell invocation
regenerates the 12.22M-parameter weight tensor and verifies that a reference C
interpreter, a simulated transformer, and a matrix-based forward pass agree on
123 of 123 traced programs at every step on every dimension of the
256-dimensional state vector. The artefact lives in the same repository as the
compiler that hosts it.

---

## Differentiating capabilities

Each item below cites the file or measurement that grounds the claim.

- **Multi-shot, re-entrant continuations on a compiled backend
  (v1.3.5-evolve).** A captured continuation can be invoked any number of
  times, from any dynamic extent, including after the procedure that captured
  it has already returned — the shape generators, coroutines, and `amb`-style
  backtracking search all need. Native gives a capture that may outlive its
  frame a durable copy of the live C stack, restored to the same addresses
  before resuming, so interior pointers stay valid with no relocation; an
  escape-only capture (early return, exception-style unwinding) keeps the
  original zero-overhead `setjmp`/`longjmp` path. The bytecode VM snapshots its
  operand stack and call-frame array while deliberately excluding top-level
  bindings — the *store* — from the *control* snapshot R7RS asks `call/cc` to
  capture, so `set!` and `define` effects at top level survive re-entry. A
  continuation captured inside `with-region` pins every open region on both
  engines, so the failure direction is a bounded leak and never a dangling read.
  Gated on all three engines against six fixtures whose transcripts are compared
  byte-for-byte.
  See *tests/continuations/*, *scripts/run_continuation_tests.sh*, and the
  [continuations reference](../docs/reference/language/continuations.md).

- **Region reclamation on both engines (v1.3.5-evolve).** `with-region` now
  reclaims on the bytecode VM as well as on native codegen. The port matches the
  native engine's semantics rather than its implementation: native copies the
  escaping subgraph Cheney-style, while the VM marks from its root set and
  sweeps at arena-block granularity, because a VM value addresses the heap by a
  small integer index rather than by pointer — marking moves nothing, so `eq?`
  identity, shared structure, and cycles survive with no special handling.
  Coverage is total by construction: a compile-time-checked 33-wide table
  classifies the full heap tag space (the 28 `HeapType` members, the three
  manifold tags defined outside the enum, and the two unassigned slots), a fatal
  startup check requires every row to be filled in, and an unclassified subtype
  pins its region rather than guessing. Measured on the same fixture: flat at
  25-27 MB across 1,000, 4,000 and 16,000 iterations, against 793 MB with the
  evacuator disabled and 704 MB for an unwrapped control.
  See *ROADMAP.md §v1.3.5* and
  [docs/breakdown/RUNTIME_CONFIGURATION.md](../docs/breakdown/RUNTIME_CONFIGURATION.md#bytecode-vm-region-reclamation).

- **Tail calls that do not care how they were spelled (v1.3.5-evolve).** A
  mutual tail call written with `cond`, `case`, `when`, `unless`, or as the last
  operand of `and`/`or` gets the same constant-stack guarantee `if` always had.
  A transferring procedure does not call its target: it copies its evaluated
  arguments into a per-thread transfer record, records the callee's uniform
  entry, and returns, and a driver loop compiled into the public entry point
  runs the transfer in its place — one native frame live per hop, reused
  regardless of arity or target. Differing-signature mutual tail calls run
  100,000,000 hops at 9.1 MB peak resident memory, and non-AArch64 targets get
  the same bound without an aggregate-return `musttail`; the depth ladder runs
  each mutual-recursion shape at 500,000, 5,000,000 and 100,000,000 hops,
  including a four-cycle that routes through `cond`, `when`, `or` and `case`, one
  form per hop. Tail calls through `guard` stay bounded deliberately: R7RS does
  not make that a tail context.
  <!-- source: .icc/completion-oracles.yaml (100,000,000 hops at 9.1 MB peak RSS); scripts/gen_recursion_depth.py:125-155 -->
  See ADR-0006 §3.

- **Exhaustive differentiation dispatch, enforced by the compiler
  (v1.3.5-evolve).** A `default:` arm may not stand in for an invariant.
  `ad_node_type_t`, `callable_subtype_t`, and `EvacKind` are generated from a
  single declaration registry with no `default:` arm, and `-Werror=switch`
  plus `-Werror=switch-enum` make an unhandled member a compile error instead
  of a plausible answer; an ICC invariant re-derives each enum's members from
  its own definition so the guarantee holds on toolchains where the flag alone
  cannot enforce it. A registry row naming a backward function that does not
  exist is itself a compile error, which is what makes a row's claim to be
  registered mean registered. Open sets stay loud on purpose: a subtype byte
  read out of an object header is untrusted input, so it is split off with a
  value-naming fallback rather than folded into the closed enum.
  See *inc/eshkol/ad_node_registry.def* and *.icc/architecture-model.yaml*.

- **Exact gradients through hyperbolic geometry (v1.3.5-evolve).** Hyperbolic
  distance, the Poincaré exponential and logarithmic maps, and geodesic
  attention carry exact closed-form backward rules, each declared as a bridged
  row in the dispatch registry. The exp and log rules reuse the Möbius-addition
  and log-map Jacobians the Fréchet rule already differentiates rather than
  re-deriving them, since a second derivation could only introduce a
  disagreement. Validated against golden Jacobians from an independently written
  Eshkol transcription of the same formulas (agreement to 3.7e-16 and 1.1e-14)
  and against two derivation-independent identities — the conformal
  gradient-norm identity and the inverse-Jacobian identity, at maximum relative
  deviations of 5.0e-16 and 6.7e-16 — before finite differences are consulted at
  all. Two points are made
  explicit rather than hidden: the distance is not differentiable at coincident
  points, and geodesic attention is therefore not differentiable when a query
  row equals a key row exactly, and both refuse loudly, naming the offending
  index, rather than picking a plausible subgradient.
  See *lib/bridge/tensor_backward.cpp* and
  *tests/bridge/qllm_bridge_geometric_gradcheck_test.cpp* (13 checks, pinned by
  count).
  <!-- source: .icc/silent-wrong-ledger.yaml SW-65 evidence block -->

- **Forward producers for the tensor-embedding and Fréchet-mean AD nodes
  (v1.3.5-evolve).** `ad_tensor_embedding` and `ad_frechet_mean` record real AD
  nodes through the real dispatch path, so the backward rules are exercised by
  the producer that fills their contract rather than by hand-assembled fixtures
  written from the same contract the rule reads. Fractional, negative, and
  out-of-range embedding indices are refused at record time rather than rounded
  or clamped into a wrong row, and the Fréchet forward shares its Karcher
  iteration with the VM's own opcode so forward and backward cannot disagree
  about what "converged" means. Gradchecked against exact analytic references:
  exact scatter-add for the embedding with 0 mismatches, the exact Euclidean
  closed form for the Fréchet mean at 0.0, and a hyperbolic finite difference of
  8.3e-10 over 48 partials.
  <!-- source: CHANGELOG.md v1.3.5-evolve (#497); tests/bridge/qllm_bridge_producer_gradcheck_test.cpp -->
  See *lib/bridge/qllm_bridge.cpp*, *inc/eshkol/backend/frechet_mean_core.h*,
  and *tests/bridge/qllm_bridge_producer_gradcheck_test.cpp*.

- **Constructive proof of a transformer as an interpreter.** Six layers,
  d_model = 256, feed-forward width 2304, 16 attention heads, 12.22M parameters.
  The artefact covers 82 of the 83 canonical opcodes; the one remaining external
  boundary is `OP_NATIVE_CALL`, the deliberate dispatch point for host-runtime
  services. See *docs/SDNC.md* and *lib/backend/weight_matrices.c*. The
  reproduction harness is *scripts/paper/run_paper_suite.sh*; expected wall time
  is under five minutes on a 2023 M2 Max.

- **Arbitrary-order automatic differentiation.** A Taylor-tower engine (thirteen
  gated phases, P0-P12) computes every derivative up to an arbitrary order `k`
  in one pass — `k+1` coefficients and O(k²) work, not the 2^k blow-up of nested
  dual numbers. When the seed point is exact and the function only uses
  exact-preserving operators, `derivative-n` and `taylor` return exact
  arbitrary-precision (bignum/rational) results rather than floating-point
  approximations; `taylor-model`/`tm-range`/`tm-eval` pair the polynomial with a
  rigorous interval-remainder bound for a provable range enclosure. Towers are
  tensor-valued, compose through reverse-mode (checkpointed reverse-over-Taylor),
  recover sparse Hessian structure via graph coloring, and work through
  `if`/`cond`/named-let/recursion. See *lib/core/taylor_recurrences.def*,
  *lib/core/runtime_taylor.c*, and the
  [Automatic Differentiation guide](../docs/guide/AUTOMATIC_DIFFERENTIATION.md).

- **A no-finite-differences guarantee that can fail.** The counter behind
  `(ad-finite-difference-evals)` has a real writer on the one central-difference
  backward the tape defines, reported through the zero-arity builtin
  `(ad-note-finite-difference!)` on native and on the VM alike, and the
  exactness gate runs a positive case beside a negative control — a difference
  quotient deliberately planted in the gradient path — on JIT, AOT, and the VM.
  Exactness also gains a structural gate that an output differential cannot
  provide: `.icc/ad-carrier-manifest.yaml` declares, per operator and per
  engine, which differentiation carrier answers it and whether it is exact, and
  a gate re-derives each declaration by extracting and classifying the actual
  `case` body in the emitted sources, so a declaration cannot be laundered
  through a helper. See *scripts/run_ad_exactness_gate.sh* and
  *scripts/gate_ad_shared_node_model.py*.

- **Compiler-integrated automatic differentiation (order ≤ 2).** Three modes:
  symbolic AST rewriting at compile time using twelve differentiation rules;
  forward mode through 16-byte dual numbers `{value, derivative}`; reverse mode
  through a computational graph spanning more than twenty AD node types with a
  32-level tape stack for nested gradients. Eight vector-calculus operators —
  `derivative`, `gradient`, `jacobian`, `hessian`, `divergence`, `curl`,
  `laplacian`, `directional-derivative` — are language primitives. Custom-VJP
  tape nodes (`AD_NODE_CUSTOM`) carry an externally supplied vector-Jacobian
  product, so a foreign computation with a known adjoint participates exactly in
  reverse-mode AD (first user: Moonlab's VQE gradient), and per-component
  gradient replay is collapsed into one primal plus one reverse pass reading
  every input gradient from the tape. See *lib/backend/autodiff_codegen.cpp* and
  *docs/DESIGN.md §Automatic Differentiation*.

- **Quantum computing, opt-in and differentiable
  (`-DESHKOL_QUANTUM_ENABLED=ON`).** The `agent.quantum` module binds the
  Moonlab state-vector simulator: state creation/teardown,
  Hadamard/Pauli/CNOT/rotation gates, `measure`, `expectation-z`, and a
  `with-quantum-state` auto-destroy helper, plus VQE builtins with H2/LiH/H2O
  molecular Hamiltonians whose energies differentiate through Eshkol's own AD —
  custom-VJP tape nodes bridge Moonlab's exact adjoint gradient into the reverse
  tape, so `(vqe-energy ...)` composes with ordinary `gradient`/optimizer code
  (the release gate requires the bridged adjoint to match Moonlab's native
  adjoint to within `1e-8` and a central finite difference to within `1e-4`). A
  permanent 16K-shot CHSH Bell-inequality gate (`bell-chsh`,
  *tests/quantum/bell_chsh_test.esk*) requires `2.4 < S <= 2.95`, beyond the
  classical bound of 2 — a run this cycle measured S = 2.835; the exact value is
  a random-shot measurement that varies run to run within those gate bounds,
  which is what proves genuine quantum correlations rather than a classical
  imitation. Cloning a linear `Qubit` is a rejected compile in the default build:
  the violation stops before code generation, exits nonzero, and writes no
  artifact. `quantum-random` draws from Moonlab's Bell-verified QRNG when quantum
  is enabled, with an honestly-labeled classical fallback otherwise. The
  companion `agent.pqc` module provides ML-KEM (FIPS 203) post-quantum key
  encapsulation at the 512/768/1024 security levels over R7RS bytevectors,
  QRNG-seeded, verified against NIST KAT fingerprints. Differentiable quantum
  chemistry examples ship in *examples/* — *vqe_h2.esk*, *qng_vqe.esk*, and
  *h2_vibrational_quantum.esk*. See *lib/agent/quantum.esk*,
  *lib/agent/pqc.esk*, and *lib/agent/c/agent_quantum.c*.

- **Incremental dataflow (`core.dbsp`).** Z-sets (weighted multisets) as a
  commutative group, the `z^-1`/`D`/`I` stream operators (D and I mutual
  inverses), incremental relational operators — linear map/filter/project/union,
  join via the discrete three-term product rule, multiplicity-correct `distinct`
  — and the generic incrementalizer `Q^Δ = D ∘ lift(Q) ∘ I`, in pure Eshkol with
  zero compiler changes; the first shipped slice of the incremental-dataflow
  spine (ADR 0009). Acceptance gate 27/27 under JIT and AOT, wired into the
  per-pull-request gate set as of v1.3.5-evolve. See *lib/core/dbsp.esk*.

- **100% R7RS conformance on the portable differential corpus.** A
  reference-Scheme oracle runs the same 34-program portable R7RS-small corpus on
  Eshkol and on chibi-scheme 0.12.0 and diffs the output: 34 of 34 AGREE (100%).
  Separately, Eshkol implements roughly 95% of the broader R7RS-small procedure
  surface (232 of 244 procedures) — full numeric tower, continuations,
  exceptions, promises, `eval`, records, bytevectors, hygienic macros. As of
  v1.3.5-evolve the reader accepts the third R7RS `<identifier>` production,
  vertical-line symbol syntax (`'|weird sym|`), across all four readers — the
  native tokenizer, the VM tokenizer, and both runtime `read` implementations —
  including the mnemonic escapes, `\|`, and `\x<hex>;`, with `write` emitting
  bars only when a name cannot be spelled bare and `display` never barring; and
  `gensym` is reachable identically on native JIT, native AOT, and the VM.
  See *scripts/run_reference_differential.sh*, *tests/reference-diff/corpus/*,
  *tests/features/pipe_symbol_test.esk*, and
  *tests/control_flow/gensym_test.esk*.

- **Full R7RS numeric tower.** int64, arbitrary-precision bignum (with automatic
  overflow promotion and demotion), exact rational with GCD reduction, IEEE 754
  double, and complex numbers with Smith's-formula division. Exactness tracked
  via a flags byte on each 16-byte tagged value. Exact rationals are
  bignum-capable: a canonical discriminated union with a zero-allocation int64
  fast path and a bignum numerator/denominator path taken only on overflow, so
  exact fractions hold their exactness at bignum magnitudes — verified
  byte-identical against Python `Fraction`. See
  *lib/backend/arithmetic_codegen.cpp* and *inc/eshkol/eshkol.h §Heap subtypes*.

- **Flat memory for resident and daemon workloads.** Self-tail-recursive loops —
  both named-let and plain `define`, including a catch-all guard body — get
  automatic, zero-annotation per-iteration arena-scope reclamation. The AOT gate
  for the plain-`define` case
  (*tests/memory/define_loop_flat_rss_aot_test.sh*) measures 8 MB peak RSS with
  the reclamation compiled in, against 2,620 MB with it compiled out, on the
  same 1,000,000-iteration program. As of v1.3.5-evolve an exception guard
  entered once per tick costs nothing in steady state: handler frames come from
  a thread-local LIFO free list, so total frame memory is bounded by peak nesting
  depth rather than by entry count. A resident-longrun gate measures at 200,000
  and at 1,600,000 ticks — eight times apart — and gates on the slope rather than
  on a ceiling: transient garbage and all four persistent-mutation channels come
  back at exactly 0.000 bytes per tick, with identical byte totals at both
  horizons, so what a resident loop retains is what it publishes and nothing
  else.
  <!-- source: docs/reference/runtime/memory-model.md:308-313; tests/memory/resident_longrun_flat_gate.sh:77-79 -->
  `ESHKOL_ARENA_REPORT=1` prints the global arena's byte-exact allocation total
  at exit, since peak RSS is a high-water mark that reads low under memory
  pressure. See *lib/backend/llvm_codegen.cpp*,
  *tests/memory/resident_longrun_flat_gate.sh*, and
  [docs/reference/runtime/memory-model.md](../docs/reference/runtime/memory-model.md).

- **Region-escape evacuation across every heap subtype.** A value allocated
  inside `with-region` that escapes the region — is returned, stored outward, or
  captured by a closure — is deep-walked and promoted into the surviving arena
  instead of being left dangling after the region pops. Every `HEAP_SUBTYPE_*`
  member carries an explicit deep-walk or leaf tag with its reasoning, including
  the exact `COEFF_RATIONAL` Taylor tower, whose coefficient array is walked
  because an overflowing coefficient is a pointer to an independently
  arena-allocated bignum; the architecture-model invariant that checks this is
  derived from the source rather than from a hand-typed list, resolved through a
  libclang semantic index of the real case arms. `ESHKOL_ARENA_POISON=1` poisons
  freed arena memory so any remaining gap crashes loudly instead of corrupting
  silently. See *lib/core/runtime_regions.cpp* and
  *tests/memory/region_evac_taylor_exact_test.esk*.

- **Node identity in the frontend (ADR-0000 Stage 1, phase A).** Every AST node
  the parser produces carries a stable `NodeId`, and a side table maps that id to
  a `SourceSpan` — the first column of the
  `NodeId -> {SourceSpan, BindingId, TypedExprInfo}` substrate the compiler, the
  LSP, the docs, the REPL, and the VM all have to share if they are to give one
  answer rather than five. The parser's 32 location-stamping sites write the
  location and the identity in one statement so the two cannot drift, and the
  stream reader closes each top-level form's span with a measured *extent* — the
  first place in the frontend that records where a construct ends rather than
  only where it begins. `NodeId`s are tagged rather than bare indices, so a
  garbage word reads as "unknown": a diagnostic may fail to name a location, but
  it must never name a wrong one confidently. The LLVM codegen dispatcher is the
  first consumer, resolving the file, line, and column it reports through the
  substrate, so a node's file does not depend on the traversal that reached it.
  Coverage is measured at the consumer rather than at the parser, with "has an
  identity", "has a location", and "has an extent" kept as three separate
  numbers, graded against a monotonic span-coverage floor of 99.48% written
  truncated rather than rounded so it cannot drift upward by accident.
  <!-- source: tests/coverage/NODE_IDENTITY_BASELINE.json (span_coverage_floor 0.9948) --> See *inc/eshkol/frontend/node_identity.h*,
  *scripts/run_node_identity_gate.py*, and
  *tests/coverage/NODE_IDENTITY_BASELINE.json*.

- **An object ABI that cannot be mixed by accident (ADR-0012, stage 0).** A
  three-layer inventory — lexical token matching, libclang semantic resolution,
  and emitted-LLVM-IR ground truth — enumerates 1,273 sites that depend on the
  current object-header layout, ratcheted against a committed baseline so a new
  site fails the build.
  <!-- source: docs/design/adr/0012-object-abi-staged-migration.md:100-102; ratchet baseline .icc/abi-header-baseline.json --> A
  link-time guard whose symbol name is derived from the four numbers that
  determine object-exchange compatibility means a stale object file, JIT cache
  entry, installed runtime, or `--shared-lib` artifact fails to *link*, with an
  undefined-symbol error naming the layout it wanted. A layout-pin test pins the
  header's size and every field's offset both through the accessor and as raw
  bytes at the negative offsets generated code actually uses.
  See *scripts/abi_header_inventory.py*, *inc/eshkol/abi_fingerprint.h*, and
  *tests/core/abi_layout_pin_test.cpp*.

- **A packaged link contract a downstream project can rely on.**
  `cmake/FindEshkol.cmake` ships as the one canonical discovery module:
  `find_package(Eshkol)` resolves the compiler, the runtime archive a compiled
  program actually needs, and the stdlib object and module directory, producing
  an `Eshkol::eshkol` imported target whose link interface unconditionally
  includes `stdlib.o` plus, on Apple, the system frameworks the runtime needs —
  with no hand-written library search in the consumer at all. The homebrew
  formula and both release-asset steps install the module and
  `EshkolCompile.cmake`, and a from-scratch consumer CMake project under
  *tests/integration/system_package/* is run against a staged package by the
  package manifest, so what is checked is that the discovery contract works and
  not merely that its files landed. Scoped to macOS and Linux.
  See *cmake/FindEshkol.cmake* and
  *scripts/run_system_package_integration_test.sh*.

- **Neuro-symbolic stack as compiler primitives.** Twenty-two builtins:
  `unify`, `walk`, `make-substitution`, `make-fact`, `make-kb`, `kb-assert!`,
  `kb-query`, `logic-var?`, `substitution?`, `kb?`, `fact?`,
  `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`,
  `free-energy`, `expected-free-energy`, `factor-graph?`, `make-workspace`,
  `ws-register!`, `ws-step!`, `workspace?`. Runtime implementations:
  *lib/core/logic.cpp*, *lib/core/inference.cpp*, *lib/core/workspace.cpp*;
  lineage Robinson 1965 / Friston 2010 / Baars 1988.

- **Deterministic arena memory (OALR).** Single global arena with 8 KB minimum
  blocks, O(1) bump-pointer allocation, batch reset, 8-byte headers prepended to
  every heap object. Per-thread arenas (1 MB, lazily allocated) isolate parallel
  workers. See *lib/core/arena_memory.h*, the *lib/core/runtime_arena_\*.cpp*
  modules, and *docs/breakdown/PARALLEL_COMPUTING.md §2.1*.

- **Work-stealing parallelism.** Chase-Lev deques per worker (Chase and Lev,
  2005) with epoch-based reclamation. Measured 4–12× speed-up of `parallel-map`
  on 24 cores per *docs/breakdown/ROADMAP.md §1.1-accelerate completed*.
  Primitives: `parallel-map`, `parallel-fold`, `parallel-filter`,
  `parallel-for-each`, `future` / `force`.

- **GPU acceleration with cost-model dispatch, and a correctness gate that can
  fail.** SIMD micro-kernels for small tensors, Apple Accelerate cBLAS at the AMX
  peak (≈1,100 GFLOPS measured), Metal with double-double SF64 emulation for
  native float64 absence, and a CUDA path through cuBLAS. Backend chosen per
  operation by *lib/backend/blas_backend.cpp*, configurable via
  `ESHKOL_GPU_PRECISION`, `ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_GPU_PEAK_GFLOPS`.
  As of v1.3.5-evolve every `tests/gpu/*.esk` file aggregates a failure counter
  and exits nonzero on a `FAIL:` verdict, the test isolation layer fails a test
  that exits 0 without printing a recognized verdict marker, and a permanent,
  deliberately-failing canary (*tests/gpu/gate_canary_must_fail.esk*) runs on
  every invocation and is required to fail — if the canary ever goes green the
  whole run goes red. On the strength of a measured Metal-versus-CPU divergence
  of exactly 0 across ten probes, the gate tolerance tightens from `1e-4` to
  `1e-9`. The Ozaki-II CRT exact-GEMM certification gate
  (*tests/gpu/ozaki_certification_test.esk*) was measured on Metal (Apple M2
  Ultra) at 25/25 samples, 0 mismatches, max 58 correct dot-product bits,
  verdict PASS — floating-point GEMM certified bit-exact against the integer
  reference, not merely close.

- **A molecular Hessian from a compiled Scheme's own AD, no finite
  differences.** *examples/h2_vibrational.esk* writes the STO-3G H2
  Born-Oppenheimer energy curve as ordinary Eshkol code and differentiates it to
  exact second order with `derivative-n`: equilibrium R* = 1.3887 bohr,
  E(R*) = -1.1373 Ha, force constant d²E/dR² = 0.4771 Ha/bohr², vibrational
  frequency **5003.2 cm⁻¹** (experimental H2 ≈ 4401 cm⁻¹; the gap is the STO-3G
  basis, not the AD — the second derivative itself is exact).

- **Native agent FFI.** libcurl-backed HTTP client
  (*lib/agent/c/agent_http_client.c*), sqlite3 (*lib/agent/c/agent_sqlite.c*),
  `posix_spawn` subprocess execution with argv arrays
  (*lib/agent/c/agent_subprocess.c*), kqueue/inotify filesystem watching
  (*lib/agent/c/agent_watch.c*).

- **Comprehensively documented public API and implementation.** Doxygen-format
  documentation across 50 of the 64 public headers under `inc/eshkol/` and 56
  implementation files under `lib/`, harvested automatically into a generated
  `docs/api/` reference by `eshkol-doc`. A navigable per-subsystem reference
  index (*docs/reference/{language,ad,runtime,tensors,stdlib,agent}/INDEX.md*)
  organizes the language surface for lookup. v1.3.5-evolve adds a
  [Python bindings reference](../docs/reference/bindings/python.md) documenting
  the `Context.eval`/`derivative`/`gradient` API.

- **Hardened, permanent adversarial-testing program.** A multi-pillar
  adversarial harness — differential, feature-pair edge matrix, AD
  finite-difference oracle, stress (RSS/time budgets), VM-parity ratchet,
  depth-parametric sweeps, and the external reference-Scheme differential oracle
  — is wired permanently into the ICC release oracle rather than run once and
  discarded. As of v1.3.5-evolve the cheap pillar gates run on every pull
  request and the expensive sweeps run nightly; every finished trace is mirrored
  unconditionally into the directory the readiness oracle reads, so
  `icc readiness` is machine-reachable rather than only runnable by hand. Two
  assurance waves this cycle add gates that check the assurance itself: ledger
  integrity and oracle-schema checks, a self-verdict scanner that fails a
  PASS-graded artifact whose own text reports a failure, build fingerprints that
  fail a binary predating its most recent build-relevant source change, a
  PowerShell-encoding gate that reads the files' own bytes because CI running
  those bytes under a different default encoding could never catch the problem
  by execution, a false-green audit that fails an oracle target whose evidence
  can go missing without the target going red, and an adversarial scenario suite
  that exercises the gates themselves under a dirty worktree, a stale binary, a
  model-server outage, disk pressure, and an actually failing gate. Every
  trace-emitting harness now has a shared PASS/FAIL/INFRA/SKIP vocabulary, so an
  infrastructure timeout cannot publish itself as a code defect. Release gates,
  remeasured at commit `afbaaf5b` on 2026-08-26: the aggregate suite 45/45
  suites and 770 individual tests; CTest 198/198; executable language coverage
  1,108/1,108 (100.0%, floor PASS); SICP full-book gate 88/88 probes across all
  five chapters under both `-r` and AOT; reference-Scheme differential oracle
  34/34 AGREE against chibi-scheme 0.12.0; VM parity differential 188/188; qLLM
  oracle gate 10/10; ICC readiness 100, verdict `ready`.
  <!-- source: README.md §Testing (remeasured 2026-08-26 at commit afbaaf5b) -->
  See *docs/TESTING.md*.

- **Binary Lambda Calculus (`core.blc`).** A pure-Eshkol implementation of John
  Tromp's BLC: De Bruijn-indexed terms as homoiconic s-expressions,
  self-delimiting bit encode/decode, normal-order evaluation, a decoded 232-bit
  universal machine, BLC8 byte I/O, and ASCII lambda diagrams. Loaded on demand
  via `(require core.blc)`. See *docs/guide/BINARY_LAMBDA_CALCULUS.md*.

---

## Example

The training loop below is verbatim from *README.md §Why Eshkol*. It uses the
language's `derivative` primitive to fit `y = 2x` from five points. Nothing here
is a library import or a framework call — `derivative` is in the compiler.

```scheme
(define training-data '((1.0 2.0) (2.0 4.0) (3.0 6.0) (4.0 8.0) (5.0 10.0)))

(define (predict w x) (* w x))

(define (loss w)
  (fold-left (lambda (total pair)
    (let ((error (- (predict w (car pair)) (cadr pair))))
      (+ total (* error error))))
    0.0 training-data))

(define (train w lr steps)
  (if (= steps 0) w
    (train (- w (* lr (derivative loss w))) lr (- steps 1))))

(display (train 0.0 0.01 200))  ;; => 2.0
```

Arbitrary-order AD, run for real (`eshkol-run -r`):

```scheme
(define (f x) (expt x 30))
(display (derivative-n f 7 12))   ;; => 67465815595294257109436307840000 (exact bignum)
(display (exact? (derivative-n f 7 12)))  ;; => #t
```

Exact rational derivatives, not just exact integer ones — a rational seed
propagates through `derivative-n` as an exact fraction with no floating-point
rounding at any step:

```scheme
(define (g x) (* 8 (* x x)))
(display (derivative-n g 1/3 1))          ;; => 16/3
(display (exact? (derivative-n g 1/3 1))) ;; => #t
```

A continuation captured at top level, saved, and re-invoked until a counter runs
out — verbatim from *tests/continuations/doc_example_multishot.esk*, whose
transcript is compared byte-for-byte on native JIT, native AOT, and the bytecode
VM. `call/cc` captures the control state and not the store, so the `set!`
survives re-entry and the loop guard advances:

```scheme
(define k #f)
(define n 0)
(display (+ 1 (call/cc (lambda (c) (set! k c) 0))))
(newline)
(set! n (+ n 1))
(if (< n 3) (k n))
(display "done") (newline)
;; => 1 / 2 / 3 / done
```
<!-- source: tests/continuations/doc_example_multishot.esk, tests/continuations/expected/doc_example_multishot.txt -->


---

## Dual backend

`(gradient f 3.0 4.0)` on `f(x,y) = x²y + y³` returns the byte-identical
`#(24 57)` under the native JIT, the native AOT path, and the bytecode VM —
three independent executions of one source file.

Eshkol ships two production execution backends with the same language semantics
and independent value representations. The LLVM backend compiles to native ARM64
or x86-64 (or WebAssembly) and is the default for `eshkol-run`. The bytecode VM
(*lib/backend/eshkol_vm.c* plus its 32 *vm_\*.c* modules) is a register-plus-stack
interpreter with more than 250 native call IDs, an ESKB binary file format with
LEB128 encoding and CRC32 checksums, and full coverage of the language including
multi-shot continuations, exception handling, tensors, complex / rational /
bignum, the consciousness engine, and I/O. As of v1.3.5-evolve the VM reclaims
region memory, and `with-region`'s teardown is reached identically by lexical
exit, by a `raise` crossing the region, and by a continuation transfer out of it,
so the structured and unstructured surfaces cannot drift apart. The browser REPL
runs the bytecode VM compiled to WebAssembly via Emscripten; forward-mode AD via
dual numbers works through the same arithmetic opcodes.

The weight-matrix transformer artefact (*lib/backend/weight_matrices.c*) is the
third execution surface — the one that proves the SDNC theorem by being a
transformer that runs the same VM through its forward and backward passes.

---

## Lineage and references

R7RS Scheme (the language definition); Homotopy Type Theory (the type-system
foundation, gradual rather than strict); LLVM 21 (the code generation target,
hard version-enforced in *cmake/LLVMToolchain.cmake*); Robinson's resolution
principle, 1965; Friston's free-energy principle, 2010; Baars' global workspace
theory, 1988; Chase and Lev, *Dynamic Circular Work-Stealing Deque*, 2005.

The SDNC paper provides the constructive proof that ties the language's gradient
infrastructure to the transformer artefact — *docs/SDNC.md* and
*docs/breakdown/COMPUTABLE_TRANSFORMER.md*.

---

## Reproducibility

The SDNC artefact reproduces in one command:

```bash
scripts/paper/run_paper_suite.sh
```

Outputs land under `artifacts/paper/outputs/` with stable SHA-256 hashes printed
by the harness. A current successful run produces, among others,
`weights.qlmw = 381599e7…3f0c`, `vm-traces.jsonl = 4239cbb9…4801` (the
transformer trace agrees bitwise: same SHA), and
`comparison-report.json = 80aa6fed…4105`. Platform divergence is treated as a
bug.

As of v1.3.5-evolve the performance claims reproduce in one command too:

```bash
bench/run_public_benchmarks.sh
```

From a clean checkout it measures the four axes on which Eshkol claims something
distinctive — exact-AD cost curves, Ozaki-II CRT exact f64 GEMM, flat RSS under
resident load, and differentiable quantum kernels — and emits machine-readable
JSON alongside a human-readable table,
<!-- source: bench/run_public_benchmarks.sh:107-110 --> with the noise-control
methodology and an explicit not-benchmarked list documented in *bench/README.md*.
It is not a competition entry against XLA, PyTorch, or JAX. A companion
compile-time benchmark generates a deterministic large single-file fixture (1,600
top-level defines) and runs it against a 900-second ceiling nightly, capturing a
phase-time breakdown that attributes about 98% of the wall clock to LLVM's own
backend.

The compiler itself is bit-reproducible at link time: two back-to-back release
builds produce byte-identical `build/stdlib.bc` and `build/eshkol-run`
(*docs/HARDENING.md §`#184`*).

---

## Repository and version

| | |
|:---|:---|
| Project | Eshkol |
| Version | v1.3.5-evolve |
| Release date | 28 August 2026 (builds on v1.3.4-evolve, 31 July 2026; v1.3.3-evolve, 16 July 2026; v1.3.2-evolve, 9 July 2026; v1.3.1-evolve and v1.3.0-evolve, 7 July 2026) |
| Implementation | C17 runtime, C++20 compiler |
| Backend | LLVM 21 (version-enforced) |
| Platforms | macOS Intel and Apple Silicon, Linux x86-64 and ARM64, Windows x86-64 and ARM64 via Visual Studio 2022 + ClangCL |
| WebAssembly target | yes (`eshkol-run --wasm`) |
| Licence | MIT |
| Source | https://github.com/tsotchke/eshkol |
| Website | https://eshkol.ai |
| Paper companion | *docs/SDNC.md*, artefact `artifacts/paper/` |
