# tests/escape_matrix — P8 "escape-closure" pillar

Data and baselines for the P8 escape-closure adversarial pillar. The generators
and gates live in `scripts/p8/`; the orchestrator is `scripts/run_p8_escape.sh`.
See `docs/TESTING.md` → "P8 — escape-closure pillar" for the axis descriptions.

The pillar exists so that every externally-observed bug **class** of a release
cycle would have been caught by our own framework first. Each axis targets one
escape; the internal ledger `.swarm/P8_ESCAPE_ANALYSIS.md` maps each escape to
its originating fix and its retro-catch evidence.

## Layout

| Path | What |
|------|------|
| `arity_parity_baseline.json` | axis-3 shrink-only ratchet — known native-vs-VM parity gaps; a NEW divergence fails the gate. Regenerate with `python3 scripts/p8/p8_arity_sweep.py --native build/eshkol-run --vm build/eshkol-vm-standalone-test --full --update-baseline`. |
| `five_way_baseline.json` | axis-6 shrink-only ratchet — known doc/manifest/native/VM/provide disagreements; a NEW one fails the gate. Regenerate with `python3 scripts/p8/five_way_surface.py --update-baseline`. |
| `found/` | minimal repros for REAL bugs the pillar surfaced that are not yet fixed (tracked-open). Do not delete while the bug is open — each flips its generator/suite cell from XKNOWN to a hard gate when the bug is fixed, and is retired only once a permanent regression test replaces it (see "Closed" below). |

## Corpora are NOT committed

Axes 1/2/4/5 generate self-checking `.esk` programs deterministically from a
fixed seed into a per-run temp directory that is removed on exit (bounded,
seeded, disk-capped — the runner enforces a 512 MB corpus cap). Regenerate any
corpus for inspection, e.g.:

```bash
python3 scripts/p8/gen_ad_escape.py        --out /tmp/ad    # axes 1 & 2
python3 scripts/p8/gen_property_oracles.py --out /tmp/prop   # axis 4
python3 scripts/p8/gen_concurrency_fuzz.py --out /tmp/conc   # axis 5
```

Every generated file is in the shared `scripts/p8/harness.py` format: a
`;; CHECKS: N` header, one `PASS:`/`FAIL:` line per assertion, and a trailing
`SUMMARY` line. A `;; P8-XCRASH <task>` file is a tracked-open cell the runner
tolerates (XKNOWN) until it is fixed, at which point it reports XPASS so the cell
is promoted to a hard gate.

## Currently tracked-open (found by this pillar, unfixed)

None. `found/` is empty and every axis cell is a hard gate.

## Closed (found by this pillar, fixed, promoted to hard gates)

| ID | Bug | Fixed in | Permanent regression test |
|---|---|---|---|
| **ESH-0360** | `jacobian`/`divergence`/`curl` SIGSEGV at a `(list …)` point (axis 1) — the #343 cons→svec point normalization reached the scalar-output operators only | #354 | `tests/autodiff/field_ops_list_point_test.esk` (ctest `field_ops_list_point_runtime_smoke` / `_aot_smoke`); the `(list …)` point forms are hard gates in the axis-1 generator |
| **ESH-0361** | exit-0 masking in five `-r`/AOT fault paths (axis 7) — missing AOT input, malformed `-r`, unreadable `-r`, bad `require` under `-r` and AOT | #354 | `tests/toolchain/driver_fault_exit_code_test.sh` (ctest `driver_fault_exit_code_test`); all five cells are hard gates with diagnostic tokens in `scripts/p8/p8_fault_injection.sh` |

When a cell is fixed, the ratchet reports XPASS and the gate goes red on
purpose: promote the cell (drop its `xmask` / `KNOWN_CRASH` entry), delete its
`found/` repro once a permanent regression test exists, and record the closure
here and in `.swarm/P8_ESCAPE_ANALYSIS.md`.
