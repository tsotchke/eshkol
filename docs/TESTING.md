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
readiness oracle. The v1.3.4-evolve cycle adds, among others:
`iter_scope_partial_reclaim` (resident-loop flat RSS with persistent mutation),
`resident_longrun_flat` (SW-57: the same claim measured at two tick horizons and
gated on the *slope*, on the arena's exact byte counter rather than peak RSS —
see `tests/memory/resident_longrun_flat_gate.sh`),
`parallel_map_scope_reclaim_race` (deterministic + ThreadSanitizer-clean
`parallel-map`), the 25-check gradient-through-callable suite, the Ozaki-II
exact/fast GEMM correctness gates, `i128` native+VM parity, and the
`31_tensor_matmul` VM-parity corpus.

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

Eight criteria are wired as of v1.3.4-evolve. Five read CTest directly — the
`ctest_suite_green` whole-suite roll-up plus the
`fixed_point_exact_accumulation_gate`, `exact_input_ad_identity_gate`,
`runtime_closure_arity_spread_gate` and `define_library_same_unit_gate` groups —
and three read the sibling harnesses that judge the same cut:
`vm_surface_regression_suite` (kind `vm_surface`), `vm_parity_gate` (kind
`vm_parity`) and `event_loop_works` (kind `eshkol_smoke`). Measured on the
current v1.3.4 tree, CTest is **183/183**; the value-position and compound-accessor regression is green.

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

The campaign design lives in `.swarm/ADVERSARIAL_TESTING_CAMPAIGN.md`; the ICC
oracle wiring lives in `.icc/completion-oracles.yaml`.

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

Measured on the current v1.3.4 tree: the differential is **184/184** and the
manifest is **956 rows — 581 `vm-supported`, 44 `native-only-justified`, 331
`gap`**; verified behavioral divergences remain explicit `gap` rows with
reproducible programs under `tests/vm_parity/found/`.

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

On the v1.3.4-evolve cut the oracle reports a score of **100** with verdict
**`ready`**. The gate figures behind it, all measured on that cut: aggregate
suite **45/45** suites and **770** individual tests; CTest **183/183**;
executable language coverage **1,091/1,091** (100.0%, floor PASS); SICP
full-book **88/88** probes under both `-r` and AOT; reference-Scheme
differential **34/34 AGREE** against chibi-scheme 0.12.0; VM parity
differential **184/184**; qLLM oracle gate **10/10**.

---

## Assurance gates (v1.3.5 wave 1, #454)

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

## CI: assertion-enabled Debug lane

`linux-x64-debug` is the assertion-enabled Linux x64 compilation axis. It
configures `-DCMAKE_BUILD_TYPE=Debug`, explicitly removes `NDEBUG`, and treats
both `-Werror=switch` and `-Werror=switch-enum` as hard errors for the
registered closed-enum dispatch translation units. The lane builds the full
test-enabled tree, then runs the focused runtime CTest subset and the
closed-enum dispatch test separately:

```bash
ctest --test-dir build-debug --output-on-failure \
  -R '^runtime_(arena_core|arena_cpp|object_alloc|regions|tagged_cons)_test$'
ctest --test-dir build-debug --output-on-failure -R '^exhaustive_dispatch$'
```

The exact matrix name is also present in `docs-only-required-context-stubs`,
so it is a branch-protection candidate that remains reportable for a docs-only
pull request.

## CI: docs-only PRs now actually report required contexts (#455)

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
