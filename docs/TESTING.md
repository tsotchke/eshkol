# Testing & Adversarial Harnesses

Eshkol's correctness is defended by two layers of automated tests:

1. **Functional gates** — shell-driven suites in `scripts/` that build the
   compiler and run corpora of `.esk` programs under both execution paths
   (`-r` JIT and AOT). The flagship is the **SICP full-book gate**.
2. **Adversarial harnesses** — five permanent harnesses introduced in
   v1.3.0-evolve whose job is to *find* bugs, not just confirm known-good
   behavior. Each emits [ICC](https://github.com/tsotchke) trace events that a
   readiness oracle consumes, so a green release requires them to pass.

All harnesses honor the `BUILD_DIR` environment variable (default `build/`), so
you can point them at any built tree:

```bash
# Build the compiler + stdlib once
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target eshkol-run stdlib -j
```

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

```bash
BUILD_DIR=build scripts/run_vm_parity.sh
```

> The VM parity harness lands with the v1.3.0-evolve release (PR #118); the
> other four harnesses are already on `master`.

---

## ICC readiness oracle

Each harness writes JSON-L trace events under `scripts/icc_traces/`. The oracle
definitions in `.icc/completion-oracles.yaml` map required event kinds/names to
release gates (e.g. `stress-budget`, `ad-oracle`). A release is "ready" only
when the required oracles report their green verdicts, which is how the
adversarial layer is enforced rather than merely available.
