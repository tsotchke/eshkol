# Is a `guard` body a tail context?

Four fixtures that answer that question by measurement rather than by reading,
and one place to re-measure it. They exist because the tail-transfer dispatcher
(ESH-0102c) *could* optimize a mutual tail call in a `guard` body and it turned
out that it must not.

## The question

R7RS 7.3 derives `guard` from `with-exception-handler` wrapping the body, so the
handler is installed for the body's whole dynamic extent. A call in a guard body
is therefore **not** in a tail context: the guard has to still be able to catch
what the callee raises. R7RS 7.1.3's tail-context grammar does not list `guard`,
and it cannot.

If a compiler treats that position as tail anyway, the guard is left before the
callee runs and a different handler answers. That is not a stack question — it
costs nothing and shows up only as a different value.

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

## What Eshkol answers today

Measured on `feat/tail-transfer-complete`, JIT and AOT alike:

| fixture | Eshkol | agrees? |
|---|---|---|
| 01 | `(caught-by-a 0)` | yes |
| 02 | `(gb 0)` | yes |
| 03 | `inner` | yes |
| 04 | `outer` | **no — SW-53** |

Rows 01-03 pass because the innermost activation's own guard is the one that
fires either way. Row 04 is the one that can tell the difference, because a
re-raise has to find the *enclosing* guard of the same recursive procedure — and
ESH-0222's self-tail transform through `guard` collapses those activations into
one loop iteration, so there is no enclosing guard left. See `SW-53` in
`.icc/silent-wrong-ledger.yaml`; the fix it needs is a heap-owned handler
continuation (ADR-0006 §4), not a revert of ESH-0222.

Rows 01 and 03 are also the regression fence for the mutual side. Lowering those
sites as tail transfers made 01 report an unhandled exception and 03 answer
`outer`, which is why `collectMutualTailCallSites()` still does not descend into
`GUARD_OP`.

## Running

```sh
for f in tests/tco/guard_tail_context/*.esk; do
    printf '%-46s eshkol=%s\n' "$(basename "$f")" "$(./build/eshkol-run -r "$f" 2>/dev/null | tail -1)"
done

# and against the reference:
for f in tests/tco/guard_tail_context/*.esk; do
    { echo '(import (scheme base) (scheme write))'; cat "$f"; } > /dev/shm/ref.scm 2>/dev/null || \
    { echo '(import (scheme base) (scheme write))'; cat "$f"; } > "${TMPDIR:-.}/ref.scm"
    printf '%-46s chibi=%s\n' "$(basename "$f")" "$(chibi-scheme "${TMPDIR:-.}/ref.scm" 2>&1 | tail -1)"
done
```

These are deliberately *not* wired into `scripts/run_tco_tests.sh`: that runner
globs `tests/tco/*.esk` and gates on a clean exit, which row 04 would pass while
still answering wrongly. A gate that cannot see the defect is worse than no gate,
so the divergence is carried in the ledger, where it has a waiver and an owner.
