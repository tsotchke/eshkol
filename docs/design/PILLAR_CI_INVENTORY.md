# Pillar-harness CI inventory (ADR-0010 §2.5)

**Why this file exists.** The 2026-08-25 architecture audit
(`docs/reports/` history / private audit note) found that `icc readiness`
targets report `blocked/0` almost entirely because their runtime traces do
not exist in CI, not because the underlying code is broken — after the
audit regenerated `vm-parity` and `depth-coverage` traces by hand, both
oracles went to `ready/97`. Only ~14 of the harnesses under `scripts/`
were reachable from any GitHub Actions workflow; the rest existed, worked,
and were provably dark.

This document is the honest census that PR "ci: arm the pillar harnesses
so readiness is machine-reachable (ADR-0010 §2.5)" is built from: every
script that emits an ICC trace event (`scripts/icc_traces/*.jsonl`), every
`gate_*.py`/`check_*.py`, and the `p8/`/`doc_audit/` families, classified
into what's already wired, what this PR newly wires and where, what is
deliberately deferred, and why. Runtimes below are **measured on this
machine** (Apple Silicon, RelWithDebInfo/Release build, warm ccache-free
build) unless marked "estimate" — hosted GitHub runners will typically be
somewhat slower.

## The crux bug this PR fixes: trace-location mismatch under `ESHKOL_DURABLE_WORK_ROOT`

`scripts/lib/durable_work_root.sh` lets a caller opt out of ephemeral
scratch space by setting `ESHKOL_DURABLE_WORK_ROOT`. Seven gates redirect
their *entire* trace file under that root when it is set, and never touch
`scripts/icc_traces/` at all:

| Script | Trace file |
|---|---|
| `run_icc_smoke.sh` | `eshkol_smoke.jsonl` |
| `run_ad_adversarial.sh` | `ad_adversarial.jsonl` |
| `run_mono_equiv_ad_taylor_gate.sh` | `mono_equiv.jsonl` |
| `run_language_coverage.sh` | `language_surface_coverage.jsonl` |
| `run_vm_parity.sh` | `vm_parity.jsonl` |
| `run_wasm_differential.sh` | `wasm_parity.jsonl` |
| `run_v1_3_readiness.sh` | (delegates to the above) |

`icc readiness` / `icc completion-oracle` default to reading
`scripts/icc_traces/`. A human (or an audit) who runs the documented
`ESHKOL_DURABLE_WORK_ROOT` workflow gets real, fresh traces that
`readiness` cannot see without a hand copy — exactly what happened during
the 2026-08-25 audit and, per its own account, at least twice before that.

**Fix** (`scripts/lib/durable_work_root.sh`, new function
`eshkol_durable_mirror_trace`): each of the six gates above (the seventh,
`run_v1_3_readiness.sh`, only delegates) now mirrors its finished trace
file into `scripts/icc_traces/<name>.jsonl` as a final, best-effort step
when the durable root was used. CI never sets
`ESHKOL_DURABLE_WORK_ROOT`, so this is a no-op there (those gates already
wrote `scripts/icc_traces/` directly); a human running the durable-root
workflow no longer has to hand-copy anything before asking for a
readiness verdict.

## Classification

**A = already in CI** (cite job / file:line). **B = wired by this PR into
the new `pillars-fast` job in `ci.yml`** (runs on every PR). **C = wired
by this PR into the new `pillars-nightly.yml`** (cron + `workflow_dispatch`,
not on the PR critical path). **D = requires special hardware/network,
not wired**. **E = deliberately deferred** (real pillar harness, not
wired in this pass, with a reason). Ordinary per-suite unit-test runners
(`run_bignum_tests.sh`, `run_list_tests.sh`, ~40 similar files) and
build/packaging scripts are out of scope — they don't emit ICC trace
events and are not oracle pillars.

### A — already in CI

| Script | Where | Notes |
|---|---|---|
| `gate_no_silent_wrong.py` | `ci.yml` `assurance-gates`, ~L286 | required, every PR |
| `check_ledger_integrity.py` | `ci.yml` `assurance-gates`, ~L290 | required, every PR |
| `check_oracle_schema.py` | `ci.yml` `assurance-gates`, ~L294 | required, every PR |
| `test_language_coverage_gate.py` | `ci.yml` `surface-manifest`, ~L231 | required, every PR |
| `run_ad_oracle.sh` | `ci.yml` `unix-matrix` → `linux-x64-asan-ubsan` test_command, ~L607 | required lane, every PR, under ASan+UBSan |
| `run_ad_adversarial.sh --quick` | `ci.yml` `quantum-macos`, ~L404 | advisory (`continue-on-error`) |
| `run_language_coverage.sh` | `ci.yml` `quantum-macos`, ~L441; `release.yml` ~L1148 | advisory in CI; blocking at release |
| `run_p8_escape.sh --full` | `adversarial-nightly.yml` `escape-matrix` | nightly only, both macOS/Linux |
| `run_v1_3_readiness.sh`, `run_vm_parity.sh` | `release.yml` `release-readiness-gate`, ~L1149/1189 | release-time only, gated on `ICC_AVAILABLE` (self-hosted) |
| `run_wasm_differential.sh --quick` | `ci.yml` `wasm-execute-diff`, ~L1331 | required, every PR |
| `run_xla_tests.sh`, `run_gpu_tests.sh` | `ci.yml` `unix-matrix` XLA/GPU rows | hosted runners, build/link-only (no real GPU) |

`run_icc_smoke.sh` — the primary emitter behind most `eshkol_smoke`
criteria (`v1.3-evolve`, `eshkol-compiler-readiness`, `agent-ffi-ready`,
…) — was **not referenced by any workflow** before this PR (only by
`Makefile`'s human/agent-invoked `eshkol-swarm-cycle` target). It is now
wired (see B below).

### B — wired into `pillars-fast` (`ci.yml`, every PR)

One job, one targeted build (`eshkol-run eshkol-vm-standalone-test
stdlib`, Release), then these gates in sequence. Measured wall time
below is the gate alone, after the build; the whole job (toolchain
install + configure + build + gates) is budgeted at 20 minutes and runs
in parallel with `unix-matrix`/`wasm-execute-diff`, which are already the
slower required lanes, so it does not lengthen the PR critical path.

| Script | Oracle target(s) fed | Measured time | Build needed |
|---|---|---:|---|
| `check_depth_coverage.py` | `depth-coverage` | 0.2s | none (pure Python over committed JSON) |
| `run_dbsp_gate.sh` | *(no oracle event — see note)* | 70s | `eshkol-run` |
| `run_mono_equiv_ad_taylor_gate.sh` | `v1.3-evolve` (P2/P3), `ad-taylor-campaign` (P2/P3) | 7s | `eshkol-run` |
| `run_ad_validated_bounds_gate.sh` | `ad-taylor-campaign` (P8) | 32s | `eshkol-run` |
| `run_vm_parity.sh` | `vm-parity`, `v1.3-evolve` (dispatch-table invariant) | see below | `eshkol-run`, `eshkol-vm-standalone-test` |

`run_dbsp_gate.sh` note: this is ADR-0009's own acceptance gate
(`tests/stdlib/dbsp_test.esk`, JIT+AOT dual mode) and F7 of the audit
names it explicitly as **zero callers anywhere** — not `run_all_tests.sh`,
not a CMake target, not any workflow. Wiring it closes that specific
dead-gate finding and gives real regression protection for `core.dbsp`.
It does not itself move an `icc readiness` score because it was never
written to emit a `runtime_event`/`test_result` — no completion-oracle
criterion reads a `dbsp` trace kind today. Teaching it to emit one is a
reasonable follow-up but is a schema change to
`.icc/completion-oracles.yaml`, out of scope for a wiring-only PR.

`run_vm_parity.sh` is the audit's own control case for this whole
finding (`vm-parity` went `blocked/0` → `ready/97` purely by regenerating
this one trace) and is high-value enough to include even though it is the
heaviest gate in the fast lane; see the measured runtime in the PR body.

### C — wired into `pillars-nightly.yml` (cron `0 8 * * *` + `workflow_dispatch`)

Grouped into three jobs plus a collection job, each with its own
`timeout-minutes` and a disk-budget guard (`df` before/after + a hard
cap, and cleanup of any generated corpus with `if: always()`), per the
project rule that every fuzz/generative harness needs a time budget and a
disk cap.

**`pillars-depth-sweep`** (P6, the depth-parametric family — none of
these six scripts were referenced by any workflow before this PR):

| Script | Oracle target |
|---|---|
| `run_ad_depth.sh` | `ad-depth`, `v1.3-evolve`, `ad-taylor-campaign` (P1) |
| `run_recursion_depth.sh` | `recursion-depth`, `v1.3-evolve` |
| `gen_numeric_depth.py` + `run_numeric_depth.sh` | `numeric-depth`, `v1.3-evolve` |
| `run_metaprog_depth.sh` | `metaprog-depth`, `v1.3-evolve` |
| `run_nesting_depth.sh` | `nesting-depth`, `v1.3-evolve` |
| `run_tensor_collection_depth.sh` | `tensor-collection-depth` |

**`pillars-adversarial-sweep`**:

| Script | Oracle target |
|---|---|
| `run_differential.sh` | `differential-clean` |
| `run_differential_fuzz.sh --seed 42 --count 200` | `differential-clean` |
| `tests/edge_matrix/gen_matrix.py` + `run_edge_matrix.sh` | `edge-matrix` |
| `run_metamorphic.sh` | `metamorphic-laws`, `v1.3-evolve` |
| `run_sanitizer_fuzz.sh` | `sanitizer-fuzz-clean`, `v1.3-evolve` |

`run_sanitizer_fuzz.sh` already caps its own artifact disk usage at
300MB by default (`ESHKOL_FUZZ_MAX_GB`, bounded-gate sweep unless
`--full`); this PR runs it in default (bounded) mode and adds a
job-level `df` guard on top as a second, independent safety net — the
prior incident this project tracks (a fuzz harness filling 58GB on a
shared node) was a harness with no cap at all, not this one, but the
guard costs nothing and catches any future regression in the cap itself.

**`pillars-smoke-sweep`**:

| Script | Oracle target |
|---|---|
| `run_icc_smoke.sh` | most `eshkol_smoke`-based criteria across `v1.3-evolve`, `eshkol-compiler-readiness`, `agent-ffi-ready`, `no-regression`, `stdlib-ready` |
| `run_sicp_smoke.sh` | `sicp-completeness` (partial — see note) |

`run_sicp_smoke.sh` note: `tests/sicp/` currently holds 44 `.esk` files;
`sicp-completeness` in `.icc/completion-oracles.yaml` enumerates roughly
double that many criteria, and a number of them read `implement
tests/sicp/chX_....esk and run ./scripts/run_sicp_smoke.sh` as their
action — i.e. the probe file itself does not exist yet. Wiring the
script runs everything that *can* run today; the remaining gap is a
content gap (missing test files), not a CI-wiring gap, and is out of
scope for this PR.

**`pillars-readiness-nightly`** (needs the three jobs above): downloads
their trace artifacts, merges them into `scripts/icc_traces/` alongside
whatever this same push already produced, and — following the exact
`ICC_AVAILABLE` pattern `release.yml`'s `release-readiness-gate` already
uses — runs `icc readiness --target v1.3-evolve --trace-dir
scripts/icc_traces` if ICC is provisioned on the runner, else uploads the
merged trace bundle with an explanatory message. See PR body for the
score this produced when run locally with ICC available.

### D — requires special hardware or network, not wired

| Item | Requirement |
|---|---|
| `gpu-execution-gate.yml` (`verify_gpu_backend.py` under it) | `runs-on: [self-hosted, gpu]` — explicitly cannot run on a GitHub-hosted runner (the workflow's own header comment says so) |
| `*-cuda` / `*-xla` rows in `unix-matrix` | build/link-only on hosted runners; real CUDA execution needs GPU hardware not present there (already advisory per `CONTRIBUTING.md`) |
| `run_all_tests.ps1` | Windows-only PowerShell; the *only* runner exercising `use_after_move.esk`/`double_move.esk`/`valid_ownership.esk` (the negative ownership-rejection tests) — no `.sh` equivalent exists, so these cannot run on any Unix CI lane today (this is ADR-0001/F6 territory, not a task this PR's scope covers) |
| Moonlab/quantum networked corpus | opt-in inside `quantum-macos`, already advisory; unchanged |

### E — deliberately deferred (real pillar harnesses, not wired here)

| Script | Reason |
|---|---|
| `run_reference_differential.sh` | Needs `chibi-scheme` provisioned as a reference R7RS implementation. It is a one-line install on both apt (`chibi-scheme`, Ubuntu 22.04) and Homebrew, so this is not a hardware gate — but it is a *new* toolchain dependency in no current workflow's install list, and pulling it in deserves its own review of what the differential actually proves rather than being folded into an already-large wiring PR. Tracked as a follow-up. |
| `run_sdnc_oracle.sh` | 71-program halt-evidence + trace-agreement suite for the SDNC (programs-as-weights) subsystem, which ADR-0000 records as deliberately DEFERRED architecture; low priority to gate in CI while the subsystem itself is not on the active roadmap. |
| `run_qllm_oracle_tests.sh` | Regenerates committed golden JSON as part of its own gate (diff-on-change is the real assertion); needs a decision about what "the gate fails" should mean for a golden-file regeneration step before it belongs in an unattended nightly run. |
| `run_generative_differential.py` / `.sh` | Property-oracle generation (`gen_property_oracles.py`) is already exercised nightly via the P8 family in `adversarial-nightly.yml`; folding this in too risks duplicate coverage without a clear marginal gate. Left for a follow-up once the P8/this split is reconciled. |
| `run_edge_coverage_v134.sh` | Superseded in spirit by `run_edge_matrix.sh` (wired in C) and the depth-parametric family; the v1.3.4-specific corpus is a point-in-time snapshot rather than a maintained pillar. |
| `run_vm_surface_tests.sh`, `run_tensorcore_icc_smoke.sh` | Narrower smoke checks whose oracle criteria substantially overlap `run_icc_smoke.sh`/`run_vm_parity.sh` (already wired); left out to avoid redundant CI time until their marginal coverage is measured. |
| `scripts/p8/p8_arity_sweep.py`, `p8_fault_injection.sh`, `p8_mem_profiles.sh`, `gen_ad_escape.py`, `gen_property_oracles.py`, `five_way_surface.py` | Axes 1-6 of the P8 escape-closure campaign; only `run_p8_escape.sh --full` (which already exercises several of these) runs today, nightly-only in `adversarial-nightly.yml`. Extending that workflow further is adjacent scope to this PR (which targets the trace-availability gap, not a P8 campaign expansion) — flagged for a follow-up PR against `adversarial-nightly.yml` specifically. |
| `scripts/doc_audit/*.py` (8 files) | Extract/execute every fenced example across 358 markdown docs. Maintenance-pipeline shaped (regenerates doc examples), not a per-PR or nightly correctness gate; zero workflow references before and after this PR, unchanged. |
| `run_stress.sh` | P4 stress pillar: 81-row wall-time/RSS budget sweep. The audit itself notes RSS/flatness measurements need "a quiet machine" — running it on a shared, noisy GitHub-hosted runner risks false regressions from runner variance rather than real ones. Needs its own calibration pass (baseline budgets measured *on* a hosted runner, not transplanted from a workstation) before it can gate anything; tracked as a follow-up, not wired here. |

## The payoff: what score this produces

See the PR body for the exact `icc readiness --target v1.3-evolve
--trace-dir scripts/icc_traces` output this branch produces when run
locally with the traces this PR's `pillars-fast` set (plus, where time
allowed, a subset of the nightly set) actually generates, and the list of
criteria that remain unmet and why.
