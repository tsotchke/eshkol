# Testing & Adversarial Harnesses

Eshkol's correctness is defended by two layers of automated tests:

1. **Functional gates** — shell-driven suites in `scripts/` that build the
   compiler and run corpora of `.esk` programs under both execution paths
   (`-r` JIT and AOT). The flagship is the **SICP full-book gate**.
2. **Adversarial harnesses** — the permanent pillars introduced in
   v1.3.0-evolve and extended since, whose job is to *find* bugs, not just
   confirm known-good behavior. Each emits [ICC](https://github.com/tsotchke) trace events that a
   readiness oracle consumes, so a green release requires them to pass.

Every root-cause fix ships with a dedicated regression gate wired into the
readiness oracle. The v1.3.4-evolve cycle added, among others:
`iter_scope_partial_reclaim` (resident-loop flat RSS with persistent mutation),
`resident_longrun_flat` (SW-57: the same claim measured at two tick horizons and
gated on the *slope*, on the arena's exact byte counter rather than peak RSS —
see `tests/memory/resident_longrun_flat_gate.sh`),
`parallel_map_scope_reclaim_race` (deterministic + ThreadSanitizer-clean
`parallel-map`), the 25-check gradient-through-callable suite, the Ozaki-II
exact/fast GEMM correctness gates, `i128` native+VM parity, and the
`31_tensor_matmul` VM-parity corpus.

The v1.3.5-evolve cycle adds the assurance-gate family (every
`scripts/check_*.py` and `scripts/gate_*.py`, each with its own `--self-test`),
the VM region-evacuator memory gates
(`tests/memory/vm_region_flat_rss_test.sh`,
`tests/memory/vm_region_evac_subtype_coverage_test.sh`,
`tests/memory/vm_region_growth_watchdog_test.sh`), the leak-detector self-test
and leak audit (`scripts/check_leak_detection_selftest.sh`,
`tests/memory/leak_audit_gate.sh`), the closed-enum exhaustive-dispatch gate
(`scripts/gate_exhaustive_dispatch.py`), the AD exactness and one-pass gradient
gates (`scripts/run_ad_exactness_gate.sh`,
`scripts/run_one_pass_gradient_gate.sh`), the AD carrier gate
(`scripts/gate_ad_shared_node_model.py`), the node-identity and object-ABI
baselines (`scripts/run_node_identity_gate.py`,
`scripts/abi_header_inventory.py`), the linear-`Qubit` engine-parity gate
(`tests/typesystem/qubit_linearity_engine_parity_gate.sh`), the continuations
suite (`scripts/run_continuation_tests.sh`), and the GPU correctness gate with
its must-fail canary (`tests/gpu/gpu_correctness_gate.sh`,
`tests/gpu/gate_canary_must_fail.esk`). Each is described below.

All harnesses honor the `BUILD_DIR` environment variable (default `build/`), so
you can point them at any built tree:

```bash
# Build the compiler + stdlib once
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target eshkol-run stdlib -j
```

---

## The CTest gate

`scripts/run_ctest_gate.sh` runs the CTest suite and turns its verdicts into
oracle evidence. Until it existed, no completion-oracle criterion anywhere
consumed a CTest *result* — the only `ctest` mentions in
`.icc/completion-oracles.yaml` were `action:` strings, and the `test_evidence`
criteria are index-level ("the tests exist and are runnable"). A pillar could
therefore ship with a perfectly good CTest gate and still be unmeasured by the
target that judges the release cut.

```bash
BUILD_DIR=build scripts/run_ctest_gate.sh
```

It writes `scripts/icc_traces/ctest_gate.jsonl`: one `kind:"ctest"` event per
test, one per named **group**, the `ctest_suite_green` roll-up, and the
canonical `kind:"test_result"` events the ICC architecture invariants read.

A *group* is a regex over test names with its own roll-up event, so one
criterion gates a whole pillar and stays correct as tests are added to it. The
groups are declared in `CTEST_GATE_GROUPS` at the top of the script. **A group
whose regex matches no configured test is reported `ABSENT` and fails the
gate** — a pillar cannot quietly stop being covered because its tests were
renamed or configured out.

To gate a new pillar: give it CTest entries with a shared name prefix, add one
`CTEST_GATE_GROUPS` line, and add one `runtime_event` criterion with
`event_kinds: [ctest]` under `eshkol-compiler-readiness`.

Nine criteria are wired as of v1.3.5-evolve. Six read CTest directly — the
`ctest_suite_green` whole-suite roll-up plus the
`fixed_point_exact_accumulation_gate`, `exact_input_ad_identity_gate`,
`runtime_closure_arity_spread_gate`, `define_library_same_unit_gate` and
`module_load_path_engine_parity_gate` groups — and three read the sibling
harnesses that judge the same cut: `vm_surface_regression_suite` (kind
`vm_surface`), `vm_parity_gate` (kind `vm_parity`) and `event_loop_works`
(kind `eshkol_smoke`). Remeasured at commit `afbaaf5b` on 2026-08-26, CTest is
**198/198** (superseding the prior 183/183, which was correct on an earlier
commit); the value-position and compound-accessor regression is green.

---

## SICP full-book gate

`scripts/run_sicp_smoke.sh` runs the corpus in `tests/sicp/` — 88 probes across
SICP chapters 1-5, including the metacircular, analyzing, lazy, and `amb`
nondeterministic evaluators, the query system, and the register-machine
simulators — under both `-r` and AOT.

```bash
BUILD_DIR=build scripts/run_sicp_smoke.sh
# => SICP smoke summary: 88/88 gate probes PASS; 0 xfail, 0 XPASS; 88 total.
```

The gate fails on any real failure, any stale XFAIL that now XPASSes, or any
missing full-book system probe. Trace: `scripts/icc_traces/sicp_smoke.jsonl`.
To add a probe, drop a self-checking `.esk` file in `tests/sicp/` and register
it in the script's coverage manifest.

---

## The five adversarial harnesses

The ICC oracle wiring for every pillar lives in
`.icc/completion-oracles.yaml`; the per-harness CI census (what runs per-PR,
what is nightly, what needs special hardware, and what is deliberately
unwired) is `docs/design/PILLAR_CI_INVENTORY.md`.

### P1 — Multi-path differential harness + fuzzer

**What:** Eshkol has several execution paths that must agree on every
deterministic program: identical program + identical input must produce an
identical (exit code, normalized stdout) on every axis — `jit`, `jit-nocache`
(`ESHKOL_JIT_CACHE=0`), `aot-o0`, and `aot-o2`. Any divergence is a bug by
definition, so no external oracle is needed.

```bash
# Run the curated corpus across all native axes
BUILD_DIR=build scripts/run_differential.sh

# Seeded random fuzzing; divergent programs auto-shrink to a minimal repro
scripts/run_differential_fuzz.sh --seed 1 --count 200
```

Corpus: `tests/differential/corpus/`. Shrunk repros land in
`tests/differential/found/`. Generator: `scripts/gen_differential.py`.
Trace: `scripts/icc_traces/differential_fuzz.jsonl` (`kind=differential_smoke`).

### P2 — Feature-pair edge matrix

**What:** generates programs that *compose* pairs of language features (AD ×
closures, `set!` × TCO, quasiquote × match, …) and classifies each probe under
both `-r` and AOT as `PASS`, `ASSERT-FAIL` (wrong value — a compiler bug),
`CRASH`, `COMPILE-ERR`, or `HANG`.

```bash
BUILD_DIR=build scripts/run_edge_matrix.sh
```

Feature list: `tests/edge_matrix/FEATURES.md`. Generator:
`tests/edge_matrix/gen_matrix.py` → `tests/edge_matrix/generated/`. Known,
triaged failures are allowlisted in `tests/edge_matrix/KNOWN_FAILURES.txt`.
Trace kind: `edge_matrix`.

### P3 — AD finite-difference oracle

**What:** every generated AD probe self-checks its analytic AD result against an
in-language central finite difference, under both `-r` and AOT. This is the
oracle that catches silent wrong-gradient regressions.

```bash
BUILD_DIR=build scripts/run_ad_oracle.sh
```

Generator: `tests/ad_oracle/gen_ad_oracle.py` → `tests/ad_oracle/generated/`.
Emits pytest-style `PASSED …::<mode>` lines plus ICC events (`kind=ad_oracle`,
oracle `ad-oracle` in `.icc/completion-oracles.yaml`).

### P4 — Stress harness (RSS/time budgets)

**What:** runs the programs in `tests/stress/budgets.tsv` under `-r` and/or AOT
with explicit budgets asserted by the runner (not the program): a wall-time
ceiling, a max-RSS ceiling, exit code 0, and a required stdout substring.

```bash
BUILD_DIR=build scripts/run_stress.sh
```

Budgets/data: `tests/stress/`. Trace kind: `stress_smoke`
(`stress_suite_green` is the whole-sweep verdict). To add a case, add a data
`.esk` file and a budget row in `tests/stress/budgets.tsv`.

### P5 — VM parity ratchet

**What:** makes the bytecode-VM's supported subset explicit and makes drift
impossible to miss: `scripts/vm_parity_audit.py` extracts the native-codegen
surface and the VM surface and fails if any codegen symbol is neither
VM-supported nor consciously waived in `tests/vm_parity/PARITY.tsv`. A
VM-vs-native differential over `tests/vm_parity/corpus/` then keeps shared
symbols honest. Full write-up in [VM_PARITY.md](VM_PARITY.md).

Last measured at commit `afbaaf5b` on 2026-08-26: the differential is
**188/188** (superseding the earlier 184/184, which counted the corpus
differential rather than the full manifest gate) and the manifest is **956
rows — 581 `vm-supported`, 44 `native-only-justified`, 331 `gap`**; verified
behavioral divergences remain explicit `gap` rows with reproducible programs
under `tests/vm_parity/found/`. The corpus has grown since that commit, so the
differential figure must be regenerated on the v1.3.5-evolve release cut.

```bash
BUILD_DIR=build scripts/run_vm_parity.sh
```

**Self-checking half.** `scripts/run_vm_surface_tests.sh` compiles every
`tests/vm/*_surface_regression.esk` probe to `.eskb` and executes it on the
standalone VM. Unlike the differential above, each probe asserts against R7RS
(or a closed form) *inside one run*, so a defect **shared** by native and the
VM cannot pass by agreement. Trace kind: `vm_surface`
(`vm_surface_regression_suite` is the whole-suite verdict).

```bash
BUILD_DIR=build scripts/run_vm_surface_tests.sh
```

**WASM substrate.** `scripts/run_wasm_differential.sh` extends the parity
guarantee to WebAssembly: it builds the VM WASM module from current source,
executes the VM-supported corpus under Node, and byte-diffs its stdout against
native `eshkol-run -r`, so a VM regression that only manifests in WASM is caught.
Per-file divergences are tracked in `tests/wasm_diff/EXCLUSIONS.tsv`
(`EXCLUDED` / `XFAIL`), an unexpected match failing the gate.


## P8 — escape-closure pillar

**Why:** the earlier pillars find *new* defects well, but a class of bug still
occasionally reached a downstream consumer before our own tests flagged it. The
escape-closure pillar closes that gap: for every bug **class** observed in a
release cycle it adds a generator or gate designed so that the *same class*
would have been caught here first. Each axis names the escape it closes.

**What (eight axes):**

1. **Binding-form** — every automatic-differentiation operator
   (`gradient`/`jacobian`/`hessian`/`laplacian`/`curl`/`divergence`/`derivative`)
   is exercised with its differentiation *point* built in every construction
   form — numeric literal, `#(…)` literal, `(vector …)`, `(list …)`,
   `(tensor …)`, a variable, a `let`-binding, a function return, and
   `(the (vector any) …)` — across arity 1–3, each checked against the
   closed-form ground truth *and* against every other form. Closes a class where
   the point was classified from its AST node kind rather than its runtime value
   (crash or silent-zero the moment the value diverged from the node kind).
2. **Indirection** — the same operators reached `direct` / through a function
   parameter / `curried` / stored in a `let` / threaded through two wrapper
   frames must be byte-identical. Closes a gradient-through-callable arity class.
3. **Arity sweep** — driven from `tests/coverage/language_surface.json`: every
   builtin registered on *both* the native and VM backends is called at its
   documented arity (type-correct args) plus one wrong-arity and one wrong-type
   call, native vs VM, asserting value parity or the same clean error. A
   shrink-only baseline (`tests/escape_matrix/arity_parity_baseline.json`)
   grandfathers known parity gaps; a *new* divergence fails. Closes a
   VM-vs-native special-form class.
4. **Property oracles** — reference-free invariants that run on every substrate
   *independently* (so a shared normalizer can never hide a shared defect):
   `number->string`∘`string->number` identity over diverse doubles,
   `read`∘`write` round-trip for data, and exact algebraic identities. Closes a
   float-printing class a lossy-normalized cross-implementation differential had
   missed.
5. **Concurrency fuzz** — seeded `parallel-map` worker corpora (scope-op-heavy,
   allocating, string, mixed-`#f` closures) at input sizes straddling the pool
   threshold `{4,15,16,17,64}`, repeated 20×, each compared to the serial-`map`
   oracle in the same program. Closes a shared-arena scope-stack race; the
   nightly runs the same corpus under ThreadSanitizer.
6. **Five-way surface agreement** — a static ratchet cross-checking every
   builtin across doc mention ↔ manifest entry ↔ native registration ↔ VM
   dispatch ↔ module provide list, against a shrink-only baseline
   (`tests/escape_matrix/five_way_baseline.json`). Closes a
   documented-but-not-registered backend-asymmetry class.
7. **Fault injection** — a matrix that injects missing / unopenable / malformed
   source, a bad `(require …)`, a broken `--lib`, a bad output path, an
   undefined symbol, and a hang into the `-r` and AOT drivers, asserting a
   nonzero exit and a naming diagnostic. Closes an exit-0-masking class.
8. **Memory profiles** — workload shapes (AD-training step loop, resident KB
   tick, proof-search churn, `parallel-map` batch) × `{auto-scope, with-region}`,
   asserting the machine-independent invariant that peak RSS is *flat* in the
   iteration count (a leak scales RSS with work). Extends P4.

```bash
# CI subset (all eight axes, bounded, < ~2 min):
BUILD_DIR=build scripts/run_p8_escape.sh --quick
# Full sweep (JIT+AOT+VM lanes, full arity sweep) — also the nightly lane:
BUILD_DIR=build scripts/run_p8_escape.sh --full
```

Generators and gates live in `scripts/p8/`; corpora are generated into a
per-run temp dir (seeded, disk-capped, removed on exit). Trace kind:
`escape_matrix` (roll-up `p8_escape_matrix_green`). The full JIT+AOT+VM sweep,
the ThreadSanitizer concurrency lane, and the packaging lanes (Homebrew
build-from-source, Linux install smoke) run in
`.github/workflows/adversarial-nightly.yml`. Bugs a generator surfaces but that
are not yet fixed are quarantined (tracked-open) and recorded under
`tests/escape_matrix/found/`, flipping to a hard gate automatically when fixed.

---

## ICC readiness oracle

Each harness writes JSON-L trace events under `scripts/icc_traces/`. The oracle
definitions in `.icc/completion-oracles.yaml` map required event kinds/names to
release gates (e.g. `stress-budget`, `ad-oracle`). A release is "ready" only
when the required oracles report their green verdicts, which is how the
adversarial layer is enforced rather than merely available.

On the v1.3.4-evolve cut the oracle reported a score of **100** with verdict
**`ready`**. The gate figures behind it: aggregate suite **46/46** suites;
CTest **198/198**; executable language coverage **1,108/1,108** (100.0%, floor
PASS); SICP full-book **88/88** probes under both `-r` and AOT;
reference-Scheme differential **34/34 AGREE** against chibi-scheme 0.12.0; VM
parity differential **188/188**; qLLM oracle gate **10/10**. CTest, coverage
and VM parity were remeasured at commit `afbaaf5b` on 2026-08-26 (doc-truth
audit findings B6/N4) and supersede the prior 183/183, 1,091/1,091 and 184/184
figures. The v1.3.5-evolve roll-up is regenerated by the release battery; the
individual-test total must come from a full suite run on that cut rather than
being carried forward.

---

## Assurance gates (v1.3.5-evolve)

Three release doctrines that used to live only as prose are now enforced,
self-testing CI gates:

- **`scripts/check_ledger_integrity.py`** fails
  `.icc/silent-wrong-ledger.yaml` on a YAML parse error, any duplicate `id`
  across the whole file, or an entry missing `id`/`bucket`/`status`/`title`
  or closure evidence when its status isn't `open`. Motivating incident:
  `SW-33` was independently allocated on three separate branches, `SW-35`
  twice and `SW-42` twice — a textual merge doesn't notice two branches
  picking the same next-free id, so nothing short of an explicit
  uniqueness check catches it.
- **`scripts/check_oracle_schema.py`** fails `.icc/completion-oracles.yaml`
  on a parse error, or on any criterion that is structurally invalid for
  its kind (e.g. a `runtime_event` criterion with no `event_kinds`, or a
  duplicate criterion id), and always prints a declared-vs-graded criteria
  count per oracle so a silently-under-grading oracle is visible on every
  run rather than only when someone thinks to ask. Motivating incident:
  PR #429 once broke this file's parse by dropping a list-item opener
  while adding a criterion.
- **`scripts/gate_no_silent_wrong.py --self-test`** — the existing
  no-open-silent-wrong gate gained a self-test mode; it graded the ledger
  already but had never actually been wired into any CI workflow before
  this wave.

Each gate's self-test feeds it deliberately-broken fixtures (malformed
YAML, a duplicate id, a missing required field) plus one well-formed one,
generated into a repo-local temp directory and cleaned up immediately, and
asserts the gate goes red on every broken fixture and green on the good
one. Re-run directly for this documentation wave against commit `6d8e5c4e`
(all three `--self-test` PASS; both non-self-test invocations against the
real repo files also PASS — the ledger currently carries 87 entries across
7 buckets, and every oracle's declared criteria count matches its graded
count):

```bash
python3 scripts/check_ledger_integrity.py --self-test   # PASS
python3 scripts/check_oracle_schema.py --self-test       # PASS
python3 scripts/gate_no_silent_wrong.py --self-test      # PASS
python3 scripts/check_ledger_integrity.py                # PASS (87 entries, 0 errors)
python3 scripts/check_oracle_schema.py                   # PASS (every oracle's criteria fully graded)
```

Both non-self-test gates are wired into the `eshkol-compiler-readiness`
oracle alongside the pre-existing `no_open_silent_wrong` criterion, added
as `ctest` entries next to the repo's other Python-based validators, and
run in a fast `assurance-gates` CI job (pure Python over checked-out
files, no build step) modeled on the existing surface-manifest job.

### Wave 2 (#465) and the release-machinery gates (#493, #485, #500, #503)

The `assurance-gates` job has since grown well past those three. It builds
nothing — it is pure Python over the checked-out files — which is why it is
the one required context that runs on every PR shape, docs-only included.
Every gate runs its own `--self-test` first (deliberately broken fixtures it
must go red on, plus a good fixture it must go green on), so the job proves
each gate CAN fail before it reports that nothing failed.

- **`scripts/check_self_verdicts.py`** scans harness output for a
  self-reported failure hiding behind an overall PASS verdict. In this job
  nothing has been built, so it grades vacuously clean; the real scan runs
  inside `scripts/run_ctest_gate.sh`, `scripts/run_icc_smoke.sh` and
  `scripts/run_vm_parity.sh`.
- **`scripts/check_build_fingerprint.py`** records the binary a harness
  actually measured, so a trace cannot be credited to a build it did not come
  from.
- **`scripts/check_evidence_staleness.py --require-trace-dir`** runs last,
  after the gates above have deposited fresh evidence in
  `scripts/icc_traces/`. `--require-trace-dir` turns an empty or absent trace
  directory into `NO_DATA` (exit 2) rather than a silent PASS.
- **`scripts/audit_oracle_false_green.py`** enforces that no oracle target may
  read `ready` on zero evidence. This is the gate that closed the
  `gpu-execution` false green (#472): absent evidence now reads as unmeasured,
  never as satisfied.
- **`scripts/check_required_context_consistency.py --offline`** (#485)
  certifies that every intended required status context in
  `.icc/required-status-contexts.json` is reportable on every PR shape,
  including a docs-only PR where the matrix jobs never instantiate. It grades
  the committed target file, not only the live set, so a branch-protection
  list that is temporarily narrower than intended cannot read as "nothing to
  fix", and it reports `NO_DATA` rather than PASS when neither source is
  readable.
- **`scripts/check_doc_claims_residual.py`** (#493) requires every ICC
  `doc-typed-claims` "wrong" finding to be either allowlisted in
  `.icc/doc-claims-allowlist.yaml` or an open, maintainer-tracked DOC-DEBT
  ledger entry — zero unexplained remainder. It is a completion-oracle
  criterion and runs with ICC rather than in the fast job.
- **`scripts/check_surface_counts.py`** (#492) is the drift checker for the
  language-surface counts quoted across README, FEATURE_MATRIX, the
  architecture model and every `docs/reference/*/INDEX.md`. It reads the
  canonical totals from `tests/coverage/coverage_policy.json` and
  `tests/coverage/language_surface.json` and fails on any mismatch.
- **`scripts/check_package_manifest.py`** and
  **`scripts/check_diagnostic_corpus.py`** check packaging-manifest and
  diagnostic-corpus integrity.
- **`scripts/check_ps1_encoding.py`** (#503) fails any tracked `*.ps1`/`*.psm1`
  carrying a non-ASCII byte without a UTF-8 BOM. PowerShell 5.1 decodes a
  BOM-less script in the system ANSI code page while pwsh 7 assumes UTF-8, so
  a file that parses cleanly under pwsh 7 can throw a cascade of parse errors
  on a real 5.1 host. Execution could not have caught this; reading the files'
  own bytes can.
- **`scripts/gate_exhaustive_dispatch.py --no-trace`** (#500) enforces that a
  `switch` over a closed enum may not carry a `default:`. Primary enforcement
  is the compiler (`-Werror=switch -Werror=switch-enum`, or the
  `ESHKOL_EXHAUSTIVE_SWITCH_BEGIN` macros in
  `inc/eshkol/exhaustive_dispatch.h`), which can only produce an absence, not
  evidence. This gate is the half that reports: it re-derives each closed
  enum's members from its own definition and fails on any registered dispatch
  site that carries a `default:`, omits a member, or has had its arming
  removed.
- **`scripts/gate_ad_shared_node_model.py`** (#487) checks that every AD
  operator routes through a declared, exact, source-verified carrier, by
  re-deriving each declaration from the emitted `case` body and comparing it
  against `.icc/ad-carrier-manifest.yaml`. It is structural, not comparative:
  an output differential can only compare what two carriers compute, never
  which carrier computed it.

### Pillar harnesses armed in CI (#470, ADR-0010 section 2.5)

Several real, fast oracle pillars ran nowhere in CI, so their readiness
criteria could only be graded from a trace a human generated by hand. The
`pillars-fast` job builds `eshkol-run`, the standalone VM and the stdlib, then
runs `scripts/check_depth_coverage.py`, `scripts/run_dbsp_gate.sh`,
`scripts/run_mono_equiv_ad_taylor_gate.sh`,
`scripts/run_ad_validated_bounds_gate.sh` and `scripts/run_vm_parity.sh`
inside a disk-budget guard. `pillars-readiness` then asks ICC for a readiness
verdict over the evidence that run produced, and uploads the trace bundle
either way. The expensive sweeps stay nightly in `pillars-nightly.yml`.

### Infrastructure failure is not code failure (#475)

`scripts/lib/harness_outcome.sh` gives every harness a shared outcome
taxonomy, so a trace records whether a probe failed because the code is wrong
or because the environment could not run it. A missing toolchain, an
unreachable network dependency or an absent GPU now emits a distinct outcome
rather than an indistinguishable FAIL. Pinned by
`tests/harness/harness_outcome_taxonomy_test.sh`.

### Memory and leak gates (#461, #486)

- **`scripts/check_leak_detection_selftest.sh`** proves LeakSanitizer is armed
  under the exact `ASAN_OPTIONS`/`LSAN_OPTIONS` the `linux-x64-asan-ubsan`
  lane uses, by compiling one probe that must be reported and one that must
  not. Before it, the suppression file could quietly grow broad enough to
  swallow a real leak and the lane would still read clean.
- **`tests/memory/leak_audit_gate.sh`** is the product half: an AOT compile,
  the compiled program, the VM and the REPL all run under `detect_leaks=1`,
  failing on any leak `.icc/lsan-suppressions.txt` does not already name and
  justify, plus a slope check on the one retention the suppressions do hide.
- **`tests/memory/vm_region_flat_rss_test.sh`**,
  **`tests/memory/vm_region_evac_subtype_coverage_test.sh`** and
  **`tests/memory/vm_region_growth_watchdog_test.sh`** gate the Stage-1 VM
  region evacuator: a flat peak-RSS curve across a swept iteration count,
  complete subtype coverage on the evacuation walk, and the heap-growth
  watchdog. The coverage gate re-runs its fixture with reclamation on and off
  and requires identical printed results, so a reclamation knob can never
  change an answer.

### Identity, ABI, type-system and AD gates

- **`scripts/run_node_identity_gate.py`** with
  `tests/coverage/NODE_IDENTITY_BASELINE.json` feeds the
  `adr0000-s1-identity` oracle target (#476). It measures coverage at a
  *consumer*, not at the parser, and keeps "has an identity", "has a location"
  and "has an extent" as three separate numbers.
- **`scripts/abi_header_inventory.py`** with `.icc/abi-header-baseline.json`
  ratchets the layout-dependent site inventory (ADR-0012, #488).
- **`tests/typesystem/qubit_linearity_engine_parity_gate.sh`** pins that
  cloning a linear `Qubit` is a rejected compile on both engines, not a
  warning (#471).
- **`scripts/run_ad_exactness_gate.sh`** and
  **`scripts/run_one_pass_gradient_gate.sh`** (#474) run the
  no-finite-differences assertion together with a negative control — a
  difference quotient deliberately planted in the gradient path — on JIT, AOT
  and the VM, so the assertion can still go red.

### Benchmarks as a gate, not a number (#469, #490)

`bench/run_public_benchmarks.sh` is a public, reproducible benchmark suite on
the exactness axes. The `bench-smoke` job runs it with `--smoke` and asserts
only that `results.json` is well-formed, carries the expected schema, is
marked `smoke_mode: true`, and that every axis produced data. The job is
`continue-on-error` and named so that a hosted-runner timing is never mistaken
for a published figure. `bench/pgo_corpus/` is wired as a Stage-1 smoke
consumer (ADR 0007) in the nightly `pgo-corpus-smoke` job.

### GPU correctness gate (#501)

See [GPU_ACCELERATION.md](breakdown/GPU_ACCELERATION.md). The gate was vacuous
before this wave; it now carries a must-fail canary
(`tests/gpu/gate_canary_must_fail.esk`, which asserts something permanently
false and turns the whole run red if it exits 0, fails to compile, is missing,
or exits non-zero without a `FAIL:` marker) and a trace-contract self-test
proving that a SKIP writes no trace record at all, so an absent GPU can never
be credited as a pass.

### Release readiness on a self-hosted runner (#502)

The `release-readiness-gate` job runs on a self-hosted Linux runner carrying
the `eshkol` label, because ICC is not installed on hosted runners and a
hosted runner can only report the oracle unavailable, never certify it. On a
real tag push, absence of the oracle blocks the release; it never fail-opens
to a green publish.

## CI: docs-only PRs report every required context (#455, #477, #485)

`paths-ignore` on the `pull_request` trigger meant GitHub never started
the main CI workflow for a docs-only PR (this documentation wave included)
— so 8 of the repo's 9 required branch-protection contexts never reported
a status for that PR's head SHA, and a required context with no status
blocks a PR forever. The docs-only decision moved into a fast `changes`
job (git diff against the PR's merge-base over the same path list the old
`paths-ignore` used: `**/*.md`, `docs/**`, `notes/**`, `press/**`,
`.swarm/**`, `LICENSE`), with every heavy job gated on its `docs_only`
output. A docs-only PR now gets every required context reported as
skipped (satisfying branch protection); a PR touching any non-doc file
still runs the full matrix exactly as before. `paths-ignore` remains on
the `push` trigger, where it only reduces CI load rather than blocking a
merge.
