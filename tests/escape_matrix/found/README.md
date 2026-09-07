# tests/escape_matrix/found — P8 quarantine

This directory holds minimal repros for REAL bugs the P8 escape-closure
pillar (`scripts/run_p8_escape.sh`, axes documented in `docs/TESTING.md` →
"P8 — escape-closure pillar") surfaced that are **not yet fixed**. See
`../README.md` for the pillar's full layout and axis descriptions.

## Quarantine role

A bug a generator finds is not silently swallowed and not allowed to
red-flag every run either: it is quarantined here as **tracked-open**.

1. When axis 1/2/5's harness (`scripts/p8/harness.py`) or axis 7's fault
   matrix (`scripts/p8/p8_fault_injection.sh`) hits a real, reproducible
   crash or wrong-answer it did not expect, the minimal repro is committed
   here (one file per bug, named after its tracking id, e.g.
   `ESH-0NNN_repro.esk`) alongside a one-line note of the exact failing
   command.
2. The generator or gate that found it is given an explicit `xmask` /
   `KNOWN_CRASH` entry (or, for the shrink-only ratchets, an entry in
   `../arity_parity_baseline.json` / `../five_way_baseline.json`) naming
   that bug — the cell reports `XKNOWN` (expected-crash / expected-fail)
   instead of failing the build, and instead of silently passing.
3. The bug is tracked to closure the same way every other flaw in this
   repo is: a `SILENT-WRONG`/`LOUD-ERROR` (etc.) entry in
   `.icc/silent-wrong-ledger.yaml`, `status: open`, pointing back at the
   repro file here.

## The flip: quarantine to hard gate

When the underlying bug is fixed, the generator or gate that used to report
`XKNOWN` for that cell now reports `XPASS` — an expected failure that
started passing — and **the ratchet goes red on purpose**. That is the
signal to:

- drop the cell's `xmask` / `KNOWN_CRASH` entry (or baseline row) so the
  cell is a normal hard gate from then on;
- add a **permanent regression test** under the relevant `tests/<area>/`
  directory covering the exact repro, wired into `ctest`;
- delete the repro file here, once that permanent regression test exists
  (this directory holds live quarantine, not a bug-history archive — the
  history lives in `../README.md`'s "Closed" table and in the ledger); and
- record the closure in `../README.md`'s "Closed" table and in
  `.swarm/P8_ESCAPE_ANALYSIS.md`.

Do **not** delete a repro here while its bug is still open — that would
turn a tracked-open quarantine back into a silent, undetected regression
risk the next time the same input shape is hit.

## Currently tracked-open

None. This directory is empty (this `README.md` is the only file in it) and
every P8 axis cell is a hard gate — see `../README.md`'s "Currently
tracked-open" section, which is the single source of truth for this list.
