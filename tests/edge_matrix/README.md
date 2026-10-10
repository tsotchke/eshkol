# Edge matrix — feature-pair composition testing (adversarial pillar P2)

Every compiler divergence found in the v1.3 campaign lived at a FEATURE
COMPOSITION point
(set! × multi-closure capture, letrec × multiple instances, quote ×
let-body-tail-inside-define, first-class value × builtin predicate, ...).
This harness systematically generates probe programs that compose every
pair of language features and self-check the result against expectations
computed by the generator.

## Layout

- `gen_matrix.py`      deterministic generator (axes + forms defined inline)
- `FEATURES.md`        auto-generated axis/form documentation
- `generated/`         probe corpus (`pairNNN_<A>__<B>.esk` + MANIFEST.tsv);
                       288 files / 2,372 checks at v1.3.6-evolve
- `found/`             minimal repros of real divergences discovered by the
                       matrix (`EM1`..`EM8`)
- `KNOWN_FAILURES.txt` triaged failures (`<basename> <mode>` per line) that
                       are tracked as open entries; they do not fail the sweep
- `../../scripts/run_edge_matrix.sh`  runner/classifier (JIT `-r` + AOT)

## Running

    cmake --build build --target eshkol-run stdlib -j
    python3 tests/edge_matrix/gen_matrix.py --max-pairs 300   # regenerate corpus
    scripts/run_edge_matrix.sh                       # full sweep, jit+aot

The committed corpus is the 300-pair sweep: `--max-pairs 300` reproduces
`generated/` byte for byte, while the generator's default of 150 pairs writes
a smaller corpus.

Env knobs: `MODES="jit"|"aot"|"jit aot"`, `FILTER='pair012*'`, `JOBS=N`,
`JIT_TIMEOUT`/`AOT_COMPILE_TIMEOUT`/`AOT_RUN_TIMEOUT` (seconds).

Classification per file×mode:

- `PASS`         every self-check passed, ran to completion
- `ASSERT-FAIL`  unexpected VALUE from valid code — a compiler divergence candidate
- `CRASH`        killed by a signal
- `COMPILE-ERR`  nonzero exit without failing checks (compile/runtime error)
- `HANG`         per-file timeout

ICC traces land in `scripts/icc_traces/edge_matrix.jsonl` (kind
`edge_matrix`); the sweep-level event `edge_matrix_sweep_clean` is PASS iff
every non-allowlisted file×mode passed. The `edge-matrix` completion-oracle
target in `.icc/completion-oracles.yaml` consumes these events.

## Extending the matrix

1. Add/extend an axis in `gen_matrix.py` (`axis(...)` block):
   - a Producer needs an expression AND an expected-value expression,
     preferably built via a different syntactic route (so one broken
     feature cannot make both sides wrong identically);
   - a Context must evaluate its `{X}` hole exactly once (bind with `let`
     if you need the value twice) — effectful counter producers rely on
     this to detect double evaluation;
   - use `{ID}` in every top-level identifier a form defines.
2. `python3 tests/edge_matrix/gen_matrix.py --emit-features` to refresh
   FEATURES.md, then regenerate the corpus and commit both.
3. Widen the sweep over time: `--max-pairs 0` emits ALL ordered pairs
   (~1000); the default 300 is the committed corpus and covers every pair
   touching a high-risk axis.

## Triaging a non-PASS

1. Reproduce: `build/eshkol-run -r tests/edge_matrix/generated/<file>.esk`.
2. Decide: generator mistake (unexpected expectation/invalid form) → fix
   the generator. Real divergence → shrink to a minimal repro, save it in
   `found/`, file a tracking entry, and add `<basename> <mode>` to
   KNOWN_FAILURES.txt with a comment referencing the repro.

## Gotchas learned while building this

- `equal?` is numeric-tower-tolerant: `(equal? 6 6.0)` → `#t`, so the
  matrix cannot see exactness divergences; cover those with dedicated
  producers.
- The check harness wraps every check in a defined function on purpose:
  bare top-level constructor arguments are re-evaluated (see found/), and
  the harness must not sit on top of the very failure class it hunts. Top-level
  behavior is probed explicitly by the `toplevel` axis.
