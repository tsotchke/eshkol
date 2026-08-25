# ADR 0010 — Closed-loop assurance architecture

- Status: Accepted
- Date: 2026-07-24
- Decision owners: Eshkol compiler/runtime maintainers; assurance/adversarial-testing maintainers
- Supersedes: none (this document ratifies and sequences the adversarial-testing
  system described piecemeal across ADR 0003 (codegen/VM parity), ADR 0006
  (language/conformance), ADR 0008 (tooling), and the P1-P8 pillar campaign)
- Related: the AR1 assurance-architecture audit (the "coverage-of-coverage" map and
  the 13 ranked gaps A1-A13), `.icc/completion-oracles.yaml`,
  `.icc/architecture-model.yaml`, `scripts/language_coverage.py`,
  `tests/coverage/language_surface.json`

---

## 1. Context

Eshkol ships a large adversarial-testing system: differential execution (P1), the
feature-pair edge matrix (P2), the AD composition oracle (P3), stress/RSS budgets
(P4), VM parity (P5), the depth-parametric sweeps (P6a-f), the external-oracle
family (P7a reference-Scheme differential, P7b sanitizer-over-fuzzer, P7c
generative multi-oracle + metamorphic), the SICP and SDNC completeness suites, and
the ICC architecture-model invariants. Each instrument is real and each emits
evidence. Yet the AR1 audit found the system as a whole is **diagnostic, not
enforcing**, and that its headline coverage number certifies the wrong thing:

1. **Coverage measures spelling, not behavior.** `language_coverage.py` marks a
   construct "covered" when its *name appears in the source text* of a generator or
   corpus. A construct that is present-but-wrong, or exercised on one axis only, or
   named in a commented-out test, still counts covered on every axis. "76.7%
   covered" and "zero uncovered high-risk" both certify name-presence (AR1 gap A1).

2. **Nothing mechanically enforces any of it.** master carried **no branch
   protection and zero required status checks**; the lite CI lanes run none of
   P1-P7; the planned nightly adversarial workflow was never written;
   `scripts/icc_traces/` is git-ignored so **CI emits no oracle evidence at all**.
   `icc readiness = 100` was a *workstation* verdict a human chose to honor, never a
   repository-enforced gate. A silent-wrong-gradient regression could merge green
   (AR1 gap A5).

3. **Whole failure classes have no instrument.** The coverage-of-coverage matrix
   has fully EMPTY cells: data-race on every surface, WASM correctness (the web
   lane asserts only WebAssembly magic bytes — no wasm value is executed and diffed
   against native), packaging/install, the compiled-bytecode (`--emit-eskb`) route
   beyond P5, and the exit-code contract as a class.

4. **The one external oracle is blinded to its own claim.** The reference-Scheme
   differential normalizes floats to `%.6g`, collapses `1.0` to `1`, and strips
   string quoting before comparing — so float round-trip precision, exact/inexact
   print distinction, and write-vs-display rendering are unobservable in the only
   place that could check them (AR1 gaps A2, A3).

The instruments exist. What was missing is the *architecture that closes the loop*:
a single source of pressure, evidence that survives CI, oracles that cannot share a
blind spot, a feedback path from every escaped defect back into the generators, and
mechanical enforcement at merge and release. This ADR ratifies that architecture and
lands its first enforcement mechanics.

## 2. Decision — the closed-loop model

Assurance is organized as six coupled commitments. Each is a standing invariant of
the project, not a one-time task.

### 2.1 Single source of truth — the manifest and grammar drive all pressure

The language-surface manifest (`tests/coverage/language_surface.json`, the 1058
constructs of builtins + special forms) and the grammar are the **sole authority**
for what must be exercised. Generative pressure is *derived* from them, not
hand-curated in parallel. Hand-authored corpora are permitted only as regression
pins on top of manifest-derived generation, never as the primary coverage source. A
construct enters the surface exactly once, in the manifest; every generator,
every arity sweep, and every coverage claim reads from there. This kills the failure
mode where the manifest lists 986 builtins but the edge-matrix/ad-oracle/reference
corpora enumerate a disjoint hand-written feature list (AR1 gap A6), and the mode
where a new surface has no probe because nobody remembered to add one.

### 2.2 Evidence — every gate emits a trace; coverage is execution-backed only

Every gate emits machine-readable evidence in the established shapes:
`runtime_event` (long-form `kind: runtime_event` records, or the compact
`kind: eshkol_smoke` probe dict) and `test_result`. Evidence lands in
`scripts/icc_traces/*` and the `.icc/runtime-traces*` oracle-view directories, and
the ICC completion oracles consume it. Two hard rules:

- **Coverage claims are execution-backed only.** A construct is "covered on axis X"
  iff there is a trace stamped with that construct *executed on axis X and its output
  oracle-checked*. Name-presence is retired as a coverage signal (AR1 gap A1). This
  is the exposure-engine v2 contract: the floor ratchets on executed-and-verified
  records, the way `%make-lazy-promise` is already special-cased, generalized to
  every construct and every axis.
- **Evidence must survive CI.** A gate that runs in CI but uploads no trace is
  indistinguishable from a gate that did not run. Every lane that emits traces
  uploads them as build artifacts (Part B below).

"Green because the oracle was absent" is the specific anti-pattern this commitment
forbids. Absence of evidence is never evidence of readiness.

### 2.3 Oracles — three independent families that cannot share a blind spot

Correctness is judged by three oracle families chosen so that a single shared defect
cannot fool all three:

1. **Differential oracles (cross-implementation, RAW output).** Compare independent
   implementations of the same semantics: native JIT vs AOT-O0 vs AOT-O2 vs VM
   (`--emit-eskb`) vs an external reference (chibi-scheme). The differential must be
   taken on **un-normalized bytes** so that display/formatting defects are visible.
   The intra-Eshkol axes share the printer, the reader, and the arena
   (`runtime_display_hosted.cpp` is called by `vm_core.c` and `vm_native.c` too), so
   a differential over shared components is blind to shared defects (AR1 gap A2). The
   raw-output leg plus the external reference cover what the normalized leg erases
   (AR1 gap A3). A separate normalized lane is retained for semantic-value diffs; the
   raw lane flags float / exact-inexact / string-escape divergence as findings to
   triage.

2. **Reference-free property oracles (immune to shared-defect blindness).** Assert
   properties an implementation must satisfy against *itself*, with no second
   implementation and no golden output: printer round-trip (`read` of `write` output
   re-parses to an equal value), algebraic identities (`(+ a b) = (+ b a)`,
   `(reverse (reverse x)) = x`, associativity, AD vs central finite-difference within
   tolerance), and byte-determinism across repeated runs. Because these check the
   output against a law rather than against another engine, they catch a defect that
   *all* implementations share — the exact case differential oracles miss.

3. **Parity-contract oracles (native / VM / wasm byte-agreement).** Assert that the
   same program produces byte-identical observable behavior across the execution
   surfaces that are contractually required to agree: native codegen vs VM (P5,
   `PARITY.tsv`, both the `vm-src` and `--emit-eskb` routes), and native vs wasm
   (the wasm-execute-diff lane, closing the WASM-correctness EMPTY cell — the web
   lane must *execute* wasm and diff its output against native, not merely validate
   the binary's magic bytes). Divergence is a defect or an explicitly justified
   `native-only` row, never silent.

No single instrument is trusted alone. A regression must survive all three families
to escape, which is exponentially less likely than surviving one.

### 2.4 The escape-analysis feedback loop (the aviation model)

Every externally-reported defect is treated the way aviation treats an incident: not
as a point fix, but as evidence that the detection system has a *hole*. The doctrine:

> Every escaped defect gets a written escape analysis — *why did our own adversarial
> framework not find this first?* — whose deliverable is a **new generator axis or a
> new gate**, not merely a regression test for the one input.

The escape analysis identifies the missing dimension (a construction form, an axis, a
failure class, an arity, a normalization that hid the signal), adds it to the
manifest-derived generators, and requires **retro-catch evidence**: the extended
generator must find the original defect at the pre-fix SHA. The point regression test
is kept, but it is the floor, not the ceiling. The ledger of these analyses lives at
`.swarm/P8_ESCAPE_ANALYSIS.md`. The historical bug clusters — closure × `set!`,
named-let capture, input2 literal-vs-first-class, `do`-composition, VM silent-zero —
are exactly the recurring shapes this loop is built to convert into standing axes
rather than one-off patches. This is the P8 pillar's founding directive.

### 2.5 Enforcement — protection, required lanes, release-blocking readiness

Diagnosis becomes enforcement through three mechanical gates:

- **Branch protection with a required-check set.** master is branch-protected; a
  defined set of CI lanes is *required* (the cross-platform build/test lanes plus the
  identity guard); specialized and networked lanes (XLA, CUDA, quantum-macos) stay
  advisory because they depend on optional backends, special hardware, or a networked
  service and cannot gate the default merge. The protection itself is applied in
  repository settings; the required set is documented in `CONTRIBUTING.md`.

- **A required nightly adversarial lane.** The pillars that are too heavy for
  per-PR CI (P1/P3/P6/P7 and the P8 axes) run on a scheduled `nightly-adversarial`
  workflow that uploads all traces. A regression that a PR lane is too small to see is
  caught within a day, on the record.

- **Release-blocking readiness at the cut SHA.** The release workflow regenerates
  traces and runs `icc architecture-verify` + `icc readiness --target v1.3-evolve`
  against the tag it is about to publish, and **refuses to publish on anything but
  ready/100**. If ICC is unavailable on the runner the gate emits a loud error and
  blocks — it never fail-opens to a green publish. `readiness = 100` is thereby a
  property of the released artifact, not of a workstation.

### 2.6 The coverage-of-coverage matrix as a maintained artifact

The (surface-family × execution-axis × failure-class) matrix produced by AR1 is not
a one-time audit output; it is a **maintained artifact with EMPTY cells that are
burned down by release**. Every EMPTY cell is either (a) assigned a closing pillar and
an owning release in the map below, or (b) explicitly and durably justified as
not-applicable (for example, `quantum` for a non-quantum surface). A release may not
introduce a new EMPTY cell without a tracked closure. The matrix is the scoreboard the
five commitments above are measured against.

## 3. Gap → closure → owning-release map

Each of AR1's 13 ranked gaps, its closure mechanism, the pillar or branch that
delivers it, and the release that owns it. Class is ARCHITECTURAL (the system could
not *express* the pressure) or PROCESS/INCIDENTAL. Status is as of this ADR.

| Gap | Class | Closure mechanism | Pillar / branch | Owning release | Status |
|---|---|---|---|---|---|
| **A5** — no enforcement; master unprotected; traces git-ignored | PROCESS | Branch protection + required lanes; CI evidence upload; required nightly adversarial; release-blocking `icc readiness` at the cut SHA | **AR4 (this ADR + `feat/ar4-closed-loop-assurance`)**; nightly on P8 | v1.3.5 | In-flight (this PR lands CI evidence upload + release gate + docs; branch protection applied in settings) |
| **A1** — coverage is lexical name-presence, not executed+verified | ARCHITECTURAL | Bind every construct to a per-axis runtime-exercise record; promote the floor from name-presence to executed + oracle-checked (exposure-engine v2) | **`a1` (`test/a1-execution-backed-coverage`)** | v1.3.5 | In-flight |
| **A2** — intra-Eshkol differential blind to shared components (printer shared by all axes) | ARCHITECTURAL | Reference-free property oracles on RAW output + external round-trip printer oracle (§2.3) | **P8** (raw-output differential + reference-free property oracles) | v1.3.5 | In-flight |
| **A3** — normalization erases float / exact-inexact / string-render signal | ARCHITECTURAL | Second un-normalized reference-diff lane that flags divergence as findings; keep the normalized lane for semantic diffs | **P8** (raw-output leg) | v1.3.5 | In-flight |
| **A4** — construction / binding-form axis absent from every generator | ARCHITECTURAL | Binding-form sweep (literal / let / letrec / internal-define / lambda-arg / vector-ref / closure-capture / `set!`-mutated) + HOF/indirection wrappers, cross-product with the existing feature/depth axes | **P8** (binding-form sweep + indirection wrappers) | v1.3.5 | In-flight |
| **A6** — corpora hand-curated, not manifest-derived; 175/295 ffi + arity unmodeled | ARCHITECTURAL | Manifest-driven generation: one exercised+checked probe per construct per arity; coverage floor consumes those traces (ties to A1) | **P8** (manifest-driven arity sweep) → full manifest-driven | v1.3.5 (start) → v1.5 (full) | In-flight |
| **A7** — no fault-injection / exit-code pillar | ARCHITECTURAL | Toolchain fault-injection matrix asserting (exit-code, stderr-has-diagnostic, no-silent-0) as a gated class | **P8** (toolchain fault-injection matrix) | v1.3.5 | In-flight |
| **A8** — race / concurrency has zero detection instrument | ARCHITECTURAL | Concurrency fuzz + nightly TSan (advisory first); then TSan lane required; then full race matrix over every surface | **P8** (concurrency fuzz + nightly TSan) | v1.4 (TSan lane required) → v1.5 (full race matrix) | Planned (blocked on LLVM-21-with-tsan-runtime build) |
| **A10** — toolchain-driver / type-checker / packaging outside the exposure manifest | ARCHITECTURAL | New manifest categories `toolchain_flag` / `type_diagnostic` / `package_surface` from `cli-flag-audit` + the type-checker error enum + the install manifest; each gets a floor; packaging lane installs and runs the artifact | **P8** (packaging lane) + manifest extension | v1.4 | In-flight (`feat/assurance-a10a12`: `package_surface` category shipped as `.icc/package-manifest.yaml` + `scripts/check_package_manifest.py`, wired into every `release.yml` packaging step and `.icc/architecture-model.yaml`/`completion-oracles.yaml`; `toolchain_flag`/`type_diagnostic` categories and a dedicated install-and-run packaging lane beyond the existing `verify_release_package.py` smoke test remain open) |
| **A11** — diagnostic-quality is a single boolean | ARCHITECTURAL | Diagnostic golden-corpus (input → expected diagnostic code + span) gated per surface family | manifest extension (`type_diagnostic`) | v1.4 | In-flight (`feat/assurance-a10a12`: `tests/diagnostics/` corpus + `scripts/check_diagnostic_corpus.py`, message+code+span per case seeded from ESH-0364/ESH-0365/arity/ascription/linearity/PRs #451+#452, wired into CI; ESH-0365 is pinned via the language-coverage dispatch/accept position mechanism rather than a stderr message, since a well-formed import has no user-visible diagnostic) |
| **A12** — leak/RSS native-only; ASan leak detection off | ARCHITECTURAL | LSan lane with an arena/JIT-cache suppression file (not blanket `detect_leaks=0`); extend `region_evac` POISON to the VM route | CI lane + P8 workload-shaped RSS | v1.4 | In-flight (`feat/assurance-a10a12`: `linux-x64-asan-ubsan` now runs `detect_leaks=1` against `.icc/lsan-suppressions.txt`, proven live by `scripts/check_leak_detection_selftest.sh`; `ESHKOL_ARENA_POISON` extended to `lib/backend/vm_arena.h` — the VM arena primitives every non-evacuated VM path allocates through, underneath the region-evacuator's existing coverage) |
| **A13** — GPU/quantum correctness produce no hosted evidence; fail *open* to SKIP | INCIDENTAL→ARCHITECTURAL | Scheduled self-hosted (Jetson/Metal) lane so the GPU/quantum oracles have recurring hosted evidence; "no evidence in N days" becomes WARN, not silent pass | scheduled hosted lane + oracle staleness rule | v1.4 | Planned |
| **A9** — ICC liveness false-orphans on Eshkol's own dispatch tables | ARCHITECTURAL (ICC-side) | Registry-aware liveness: teach ICC to read the native `BUILTINS[]` closure table, the VM native-call table, the func_name/AST-op dispatch switch, and the AD tape/op registries as synthetic call edges. Until then, backend `find-dead-code` is advisory-only | **AR3** (ICC tooling track) | AR3 (tooling; not a release gate) | Planned |

**EMPTY-cell burn-down** (matrix cells with no instrument at all, and their closers):

| EMPTY cell | Closer | Owning release |
|---|---|---|
| race — every surface | P8 concurrency fuzz + nightly TSan (A8) | v1.4 → v1.5 |
| WASM correctness (wrong-value / crash / hang) | **wasm-execute-diff branch** (`test/wasm-execute-diff`) — execute wasm and diff vs native | v1.3.5 → v1.4 |
| packaging / install | P8 packaging lane (A10) — install the `.deb`/homebrew artifact and run it | v1.3.5 → v1.4 |
| `--emit-eskb` route beyond P5 | route the depth/AD/stress/reference pillars through the compiled-bytecode axis | v1.4 → v1.5 |
| exit-code contract as a class | P8 toolchain fault-injection matrix (A7) | v1.3.5 |

The v1.3.5 residue after P8 + a1 + AR4 land — i.e. what remains architectural after
the near-term wave — is A9/A10/A11 (ICC and the unmodeled infra surfaces), which is
why they are sequenced into AR3 and v1.4/v1.5 rather than v1.3.5.

## 4. What this PR lands (Part B enforcement mechanics for A5)

This ADR is accepted together with the first concrete enforcement wiring:

1. **CI evidence survives (ci.yml).** Every lane that emits traces uploads
   `scripts/icc_traces/*` and the `.icc/runtime-traces*` oracle-view directories as a
   build artifact (`if: always()`, `if-no-files-found: ignore`, per-lane artifact
   name). CI now produces the oracle evidence AR1 found it was silently discarding.

2. **Release-blocking readiness (release.yml).** A `release-readiness-gate` job
   regenerates traces at the tagged SHA (build → `run_icc_smoke.sh` →
   `run_vm_parity.sh`), runs `icc architecture-verify --emit-trace` and the canonical
   `icc readiness --target v1.3-evolve`, and **fails the release on anything but
   ready/100**. `publish-release` gains a `needs:` dependency on it, so no asset is
   published unless readiness certifies. The job uses the pinned ICC-path convention
   (`ICC_BIN`, default `scripts/run_v1_3_readiness.sh`), and if ICC is unavailable it
   emits a loud error and blocks — it never fail-opens.

3. **Documented required-check set (CONTRIBUTING.md).** The branch-protection
   required-check baseline is documented so the repository-settings configuration is
   reviewable. The protection itself is applied in settings; this file is the
   authoritative reference for *what* the required set is.

The nightly adversarial workflow and the per-gap generators (A1-A4, A6, A7, A8) are
delivered on the sibling `a1`, `p8`, and `wasm-execute-diff` branches; this ADR is
their shared architectural contract.

## 5. Consequences

- **Positive.** A silent-wrong regression must now defeat three independent oracle
  families, survive the nightly pillars, and pass a release-blocking readiness gate
  before it can ship. Coverage stops certifying spelling and starts certifying
  behavior per axis. Every escaped defect widens the net rather than adding one test.
  The matrix gives a single, honest scoreboard whose EMPTY cells have owners.

- **Cost.** The release workflow now depends on ICC being provisioned on its runner
  (`ICC_BIN`); a release cannot self-certify without the oracle, by design. The
  nightly lane and the manifest-driven arity sweep add compute. Execution-backed
  coverage is more expensive to compute than name-presence and will, transiently,
  *lower* the reported coverage number as one-axis and present-but-wrong constructs
  are reclassified — this is the metric becoming honest, not a regression.

- **Non-goals.** This ADR does not itself apply branch protection (a settings action),
  does not build the TSan toolchain (the A8 blocker), and does not fix ICC's
  dispatch-table liveness (AR3). It defines the architecture those close into.

- **Doctrine.** Consistent with the project's standing rules: fix at the architectural
  root, never dial back a test or feature to make a gate pass, and never let absence of
  evidence read as success. The coverage floor ratchets up and is never edited down.
