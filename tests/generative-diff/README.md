# Generative multi-oracle differential harness (adversarial testing pillar P7c)

> "If our system does not constantly expose every single hidden bug then it has
> no coverage." — the maintainer.

The hand-written reference-differential corpus (`scripts/gen_reference_corpus.py`,
34 programs) is now 34/34 AGREE against chibi on master — it no longer exposes
anything. This pillar makes differential testing **generative and multi-oracle**:
it generates a large, deterministic family of closed R7RS-small programs and runs
each one through every execution oracle installed, flagging any pairwise
disagreement.

## Pieces

| File | Role |
|---|---|
| `scripts/gen_generative_corpus.py` | Deterministic program generator (a pure function of `seed`, `count`). Two families: **diff** (typed value-printing probes) and **meta** (self-checking metamorphic properties). Every program is closed, printable, TOTAL (no div-by-zero / out-of-range / car-of-'()) and deterministic. |
| `scripts/run_generative_differential.py` | The harness: generates programs, runs every oracle, normalises, cross-checks, writes divergence artifacts, emits the ICC trace. |
| `scripts/run_generative_differential.sh` | Thin wrapper used as the ICC `action:` and by the smoke probe. |
| `tests/generative-diff/baseline.txt` | Known-divergence signatures (`program::kind`). The smoke probe fails only on a divergence **not** in this baseline — i.e. a NEW miscompile. |

## Oracles (auto-discovered)

* `chibi`  — `chibi-scheme` (external R7RS ground truth; optional).
* `jit`    — `build/eshkol-run -r prog.esk` (LLVM JIT).
* `aot-O0` — `build/eshkol-run -O0 prog.esk -o BIN && BIN`.
* `aot-O2` — `build/eshkol-run -O2 prog.esk -o BIN && BIN`.
* `vm`     — `eshkol-run --profile hosted-vm --emit-eskb X.eskb prog.esk`
             then `eshkol-vm-standalone-test X.eskb` (bytecode VM).

## Divergence kinds

| Kind | Meaning |
|---|---|
| `JIT_VS_CHIBI_MISMATCH` / `AOT_O0_VS_CHIBI_MISMATCH` / `AOT_O2_VS_CHIBI_MISMATCH` | An Eshkol native path disagrees with chibi ground truth (R7RS conformance bug). |
| `*_VS_CHIBI_ERROR` | chibi ran clean (exit 0) but an Eshkol path errored/crashed/timed out. |
| `AOT_O0_VS_O2_MISMATCH` | AOT output changes with optimisation level — a miscompile. Reference-free; guards the O2-default change. |
| `JIT_VS_AOT_MISMATCH` | JIT and AOT disagree on the same source. Reference-free. |
| `VM_SILENT_WRONG` | The VM exited **without an error marker** but its value differs from the reference — a silent VM miscompile (the treasure; the VM exits 0 even on fatal errors, so a wrong value with no diagnostic is the dangerous case). |
| `META_PROPERTY_FALSE` | A `meta`-family program printed `#f` for a property that must hold — a reference-free bug on whichever oracle printed it. |

The VM is only flagged when it runs *clean*: if the VM prints an
`ERROR`/`OVERFLOW`/unhandled-native marker the feature is simply outside the VM
subset (tracked by the P5 `vm-parity` ratchet) and is **not** counted here.

## Normalisation

`chibi`/`jit`/`aot` outputs pass through `scripts/lib/normalize_scheme_output.py`
(documented cosmetic canonicalisation — boolean spelling, nested char/string
rendering, float precision to 6 sig-figs; can only hide an implementation-defined
rendering difference, never manufacture a false agreement). Any comparison
involving the `vm` additionally strips the VM banner/loader lines and then all
newlines (the VM appends a newline per `display`, a filed quirk) — the strongest
comparison that quirk permits; value divergences still surface.

## Running

```sh
# full discovery run (RED while divergences remain — the philosophy)
scripts/run_generative_differential.py --seed 1234 --count 60

# regression mode: fail only on a divergence NOT already in the baseline
scripts/run_generative_differential.py --smoke --baseline tests/generative-diff/baseline.txt

# regenerate the on-disk corpus for inspection / manual minimisation
python3 scripts/gen_generative_corpus.py --out tests/generative-diff/corpus
```

Determinism: the corpus is a pure function of `(seed, count)`, so a divergence
found in CI reproduces locally byte-for-byte. On any divergence the exact program
and every oracle's raw stdout/stderr are written under
`artifacts/generative-diff/divergences/<program>/` (gitignored).

## ICC wiring

The harness writes `kind:"generative_differential"` events to
`scripts/icc_traces/generative_differential.jsonl`. The gate event
`generative_differential_oracle` is consumed by
`.icc/completion-oracles.yaml::generative-differential` and by the
`generative_differential_oracle` probe in `scripts/run_icc_smoke.sh`.
