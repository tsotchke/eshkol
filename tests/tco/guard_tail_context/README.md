# Which handler answers when a `guard` body carries a loop?

Nine fixtures that answer that question by measurement rather than by reading,
and one place to re-measure it. They exist because the tail-transfer dispatcher
(ESH-0102c) *could* optimize a call in a `guard` body and it turned out that
doing so naively changes the answer.

## The question

R7RS 7.3 derives `guard` from `with-exception-handler` wrapping the body, so the
handler is installed for the body's whole dynamic extent. A call in a guard body
is therefore **not** in a tail context by R7RS 7.1.3's grammar: the guard has to
still be able to catch what the callee raises, and — crucially — every
activation of a self-recursive procedure installs its own handler, so at depth N
there are N LIVE guards, each holding that activation's variables.

If a compiler collapses those activations into one loop iteration and drops the
handlers with them, the guard still catches the ordinary case (the innermost
activation's own handler fires either way). What it can no longer do is find the
*enclosing* guard when a handler re-raises, or when a clause body raises. That
is not a stack question — it costs nothing and shows up only as a different
value, which is why it needs a value-level gate.

## The oracle

`chibi-scheme` (measured on 0.12), an R7RS reference implementation. Each fixture
runs unchanged there once prefixed with

```scheme
(import (scheme base) (scheme write))
```

which is the only edit — Eshkol takes these forms without an import prologue.

## The fixtures and their reference answers

| fixture | reference (chibi 0.12) | what it isolates |
|---|---|---|
| `01_outer_guard_catches_callee_raise.esk` | `(caught-by-a 0)` | the callee does not re-establish a guard, so the caller's must still be standing |
| `02_innermost_guard_wins.esk` | `(gb 0)` | both partners guard; the callee's own guard is innermost and answers |
| `03_nested_guards_inner_answers.esk` | `inner` | two levels of guard around the same recursion; the inner one answers |
| `04_reraise_reaches_enclosing_guard.esk` | `(inner 1)` | the innermost handler RE-RAISES, so the answer names the next guard out |
| `05_reraise_chain_walks_out_one_activation_at_a_time.esk` | `(answered-at 0)` | 200 live guards, met one at a time, each with its own `n` |
| `06_clause_raise_reaches_previous_activation.esk` | `(bottom (1 2 3 . boom))` | a raise from inside a CLAUSE lands on the previous activation's guard |
| `07_nested_guards_reraise_through_both.esk` | `(outer-answered 1 (i o i o i . boom))` | two guards per activation: the chain alternates inner, outer, inner, … |
| `08_guard_dynamic_wind_order.esk` | `((before 0) (before 1) (before 2) (after 2) (handler 2) (after 1) (handler 1) (after 0) (caught 0))` | after-thunks run innermost-first, interleaved with the handlers that let the condition past |
| `09_clause_reads_a_binding_the_loop_rebinds.esk` | `(inner 0 0)` | the clause reads a `let` binding the loop re-establishes each iteration |

## What Eshkol answers

All nine, on the native JIT, the native AOT path and the bytecode VM alike.
Gated by `scripts/run_guard_tail_context.sh`, which is run by
`scripts/run_tco_tests.sh`.

Rows 01-03 passed even before SW-58 was fixed, because the innermost
activation's own guard is the one that fires either way. Rows 04-07 are the ones
that can tell the difference, and on the native engines they answered wrongly
until SW-58: ESH-0222's self-tail transform through `guard` collapsed the
activations into one loop iteration and drained the handler chain with them, so
there was no enclosing guard left for a re-raise to find. The bytecode VM never
treated a `guard` body as a tail position, so it answered correctly throughout —
which is exactly why every row is measured on all three engines.

The fix keeps the collapse (it is what makes a resident tick loop run forever in
one native frame) and stops it destroying the chain: a back edge taken from
inside a guard body leaves the handler frames standing and attaches the
departing activation's loop parameters to each, and the guard's landing pad puts
those values back before the clauses read them. Row 09 is the residue the
snapshot cannot cover — its clauses read `m`, which is not a loop parameter —
so that call stays a real call, which is what R7RS costs for a non-tail `guard`
body anyway. See `docs/reference/language/tail-calls.md`, "Tail calls in a
`guard` body".

Rows 01 and 03 are also the regression fence for the mutual side. Lowering those
sites as tail transfers made 01 report an unhandled exception and 03 answer
`outer`, which is why `collectMutualTailCallSites()` still does not descend into
`GUARD_OP`.

## Running

```sh
BUILD_DIR=build scripts/run_guard_tail_context.sh

# by hand, one engine at a time:
for f in tests/tco/guard_tail_context/*.esk; do
    printf '%-58s eshkol=%s\n' "$(basename "$f")" "$(./build/eshkol-run -r "$f" 2>/dev/null | tail -1)"
done

# and against the reference:
for f in tests/tco/guard_tail_context/*.esk; do
    { echo '(import (scheme base) (scheme write))'; cat "$f"; } > "${TMPDIR:-.}/ref.scm"
    printf '%-58s chibi=%s\n' "$(basename "$f")" "$(chibi-scheme "${TMPDIR:-.}/ref.scm" 2>&1 | tail -1)"
done
```

The deep, self-checking counterpart — one million collapsed iterations, a
100 000-deep re-raise chain — is `tests/tco/guard_tail_position_test.esk`, which
`scripts/run_tco_tests.sh` picks up with the rest of `tests/tco/*.esk`.
