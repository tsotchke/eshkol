# Eshkol `.icc/` — Infinite Context Coder configs

These files configure ICC's release-readiness machinery for the Eshkol
compiler. ICC (`infinite_context_coder`, invoked as `icc`) is the assistant
tool whose Eshkol-aware indexer treats `.esk` files as a first-class language and
the configs here let `completion-oracle`, `production-audit`, and
`assistant-status` give answers tailored to this repo instead of generic
"is the build green" boilerplate.

## Files

| File | Purpose |
|---|---|
| `completion-oracles.yaml` | Defines what "ready" means for various Eshkol targets. Each oracle is a list of criteria (test evidence, runtime checks, contract gaps, stub classification) and a verdict severity. |
| `production-audit.yaml` | Audit profiles that compose oracle output with artifact freshness, source drift, and guard-rail diffs into one go/no-go verdict. Different gate strictness per profile (per-PR vs release tag). |
| `assistant-goals.yaml` | Prioritized work goals the assistant suggests when no specific bug is queued. Tracks dependency order; pop items as their oracle goes green. |
| `eshkol_doc_intel.md` | Latest ICC `doc-intelligence` snapshot for the documentation tree: per-document grounded / unsupported / unresolved-ref counts and the highest-risk individual claims. Read-only reference; regenerate via `icc doc-intelligence --repo <repo> --format markdown`. |
| `eshkol_arch_summary.md` | Latest ICC architecture snapshot: indexed-file totals, public module roots, integration surfaces, dependency hubs, test roots, and per-module size. Read-only reference; regenerate via `icc architecture-summary --repo <repo>` after `icc reindex --repo <repo> --full` (the committed copy replaces the absolute checkout path with "repository checkout root"). |
| `eshkol_feature_inventory.md` | Latest ICC `feature-inventory` snapshot: symbol clusters with their documentation coverage (`none` / `mentioned` / `shallow` / `adequate` / `deep`) and documentation home. Read-only reference; regenerate via `icc feature-inventory --repo <repo> --format markdown`. |
| `invariants.yaml`, `architecture-model.yaml` | Structural invariants and the architecture model checked by `icc architecture-verify` and the pre-commit checks. |
| `ledger/`, `silent-wrong-ledger.yaml` | The per-entry ledger and its generated aggregate (below). |

`<repo>` is the name a checkout was registered under with `icc register`.

## Silent-wrong ledger

`.icc/silent-wrong-ledger.yaml` is a generated compatibility aggregate. Add or
change one entry per file under `.icc/ledger/entries/`, using the entry ID as
the filename, and put top-level ledger metadata in `.icc/ledger/meta.yaml`.
`.icc/ledger/order.txt` preserves the historical evidence order so generating
the aggregate is byte-stable. The generator rejects duplicate IDs and the
assurance-gates job rejects a stale aggregate:

```sh
python3 scripts/gen_silent_wrong_ledger.py
python3 scripts/gen_silent_wrong_ledger.py --check
```

Existing consumers continue to read the aggregate. New entries therefore add
only a new file, while duplicate IDs fail at generation time.

## Targets

| Target | When to run |
|---|---|
| `eshkol-compiler-readiness` | Daily smoke. Is the compiler healthy enough to land changes? |
| `agent-ffi-ready` | Before publishing any FFI change. Confirms HTTP/SQLite/subprocess contracts hold. |
| `stdlib-ready` | After touching `lib/core/`. Catches missing tests / known-builtins entries. |
| `v1.2-release` | Pre-tag. Strict gate; even mediums fail. |
| `v1.3-evolve` | Pre-v1.3 gate. Run through `scripts/run_v1_3_readiness.sh` so smoke traces are refreshed and passed to ICC. |
| `v1.3.5-evolve` | The v1.3.5 strict assurance gates, which remain standing for later releases. |
| `v1.3.6-evolve` | The v1.3.6-evolve release identity oracle. |
| `no-regression` | Per-PR sanity. Highs only fail. |

The file also defines one target per adversarial pillar and campaign
(`ad-oracle`, `ad-depth`, `differential-clean`, `metamorphic-laws`,
`edge-matrix`, `vm-parity`, `generative-differential`, `stress-budget`,
`total-language-coverage`, the depth targets, the Ozaki targets, and others);
`grep -n '^  - name:' .icc/completion-oracles.yaml` lists them all. Pass
`--trace-dir scripts/icc_traces` to `icc readiness` so the runtime-event
criteria see the traces the harnesses write.

## Usage

```bash
# What's blocking the daily compiler health check?
icc completion-oracle --repo <repo> \
    --target eshkol-compiler-readiness --format markdown

# Pre-release audit composing oracle + artifacts + drift + guards
icc production-audit --repo <repo> \
    --target v1.2-release --format markdown

# What should I work on next?
icc assistant-status --repo <repo> --format markdown

# Refresh v1.3 smoke evidence and check trace-aware readiness.
scripts/run_v1_3_readiness.sh
```

## Editing tips

- `requires:` is the criteria list (legacy alias for `criteria:`). Each item
  picks a kind: `test_evidence`, `runtime_check`, `runtime_event`,
  `runtime_or_contract`, `contract_kind`, `no_contract_gap`, `no_stubbed_paths`.
- For YAML strings that contain `:`, `#`, or backticks-with-colon, **quote them**.
  We hit this once already (`/bin/sh: fork: …` confused the parser).
- Severity `high` blocks; `medium` warns; `low` is advisory. Adjust per profile.
- `aliases:` let `--target eshkol` resolve to `eshkol-compiler-readiness`.

## Why these specific oracles

The oracle set was curated 2026-05-06 to match the agent-blocker matrix of
that date (a planning note kept outside this repository). The five blockers (HTTP, subprocess
contract, durable session persistence, AOT link wiring, no-return verifier)
each have either a runtime check that ICC can probe or a runtime trace
event the assistant can emit. Going green on `agent-ffi-ready` means the
agent's three top blockers (HTTP/SSE, subprocess, sessions) are unblocked
on the eshkol side.
