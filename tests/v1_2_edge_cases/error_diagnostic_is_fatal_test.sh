#!/usr/bin/env bash
# error_diagnostic_is_fatal_test.sh — an emitted error diagnostic must prevent
# artifact emission and execution, and must exit non-zero.
#
# The defect this guards against: the compiler printed "ERROR: ..." and then
# emitted, linked and ran a binary anyway, so a diagnosed program produced a
# wrong answer instead of a failed build. Every silent miscompile of that era
# was reported at compile time and the report was ignored.
#
# The test is written as an invariant over several candidate programs rather
# than as an assertion about one diagnostic, because the individual diagnostics
# belong to other subsystems and are expected to come and go as those are
# fixed. For every candidate that still draws an error diagnostic, the gate is
# checked; and at least one candidate must still draw one, so the suite cannot
# quietly stop exercising the gate. The clean program at the end is the
# anti-vacuity control: a gate that rejects everything would satisfy the
# invariant while making the compiler useless.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_error_fatal.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

FAILURES=0
DIAGNOSED=0

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

# Candidate programs that should draw a compile-time error diagnostic. Each is
# a *call-shape* error — an operator applied in a way codegen refuses — which is
# the class that used to be reported and then compiled anyway.
cat > "$WORK/arity.esk" <<'ESK'
(display (cosine-annealing-lr 0 100 0.01))
(newline)
ESK

cat > "$WORK/higher_order.esk" <<'ESK'
(define (f x) (* x x x x))
(display ((derivative (derivative f)) 2.0))
(newline)
ESK

# ---------------------------------------------------------------------------
# AOT: a reported error means no artifact and a non-zero exit.
# ---------------------------------------------------------------------------
for src in "$WORK/arity.esk" "$WORK/higher_order.esk"; do
    name=$(basename "$src" .esk)
    out="$WORK/${name}_aot"
    rm -f "$out" "$out.o"

    "$RUN" "$src" -o "$out" >"$WORK/$name.stdout" 2>"$WORK/$name.stderr"
    status=$?

    if ! grep -qE '(^|[^A-Za-z])(ERROR|error):' "$WORK/$name.stderr"; then
        # No diagnostic for this candidate any more; nothing to assert.
        continue
    fi
    DIAGNOSED=$((DIAGNOSED + 1))

    if [ "$status" -eq 0 ]; then
        fail "$name: an error diagnostic was reported but the compiler exited 0"
    fi
    if [ -e "$out" ]; then
        fail "$name: an error diagnostic was reported but an artifact was emitted"
    fi
    if [ -s "$WORK/$name.stdout" ]; then
        fail "$name: diagnostics must go to stderr; stdout was not empty"
    fi
done

# ---------------------------------------------------------------------------
# -r: a reported error means the program never begins executing.
# ---------------------------------------------------------------------------
for src in "$WORK/arity.esk" "$WORK/higher_order.esk"; do
    name=$(basename "$src" .esk)

    "$RUN" -r "$src" >"$WORK/${name}_r.stdout" 2>"$WORK/${name}_r.stderr"
    status=$?

    if ! grep -qE '(^|[^A-Za-z])(ERROR|error):' "$WORK/${name}_r.stderr"; then
        continue
    fi

    if [ "$status" -eq 0 ]; then
        fail "$name (-r): an error diagnostic was reported but the run exited 0"
    fi
    # The candidates all end in (display ...); reaching it means the rejected
    # program executed anyway, which is the defect itself.
    if grep -qE '[0-9]' "$WORK/${name}_r.stdout"; then
        fail "$name (-r): the rejected program produced output, so it executed"
    fi
done

if [ "$DIAGNOSED" -eq 0 ]; then
    fail "no candidate program drew an error diagnostic; the gate is untested. \
Add a program that still draws one rather than deleting this check."
fi

# ---------------------------------------------------------------------------
# Anti-vacuity control: a clean program still compiles, links, runs, exits 0.
# ---------------------------------------------------------------------------
cat > "$WORK/clean.esk" <<'ESK'
(define (square x) (* x x))
(display (+ (square 3) (square 4)))
(newline)
ESK

CLEAN_BIN="$WORK/clean_bin"
if ! "$RUN" "$WORK/clean.esk" -o "$CLEAN_BIN" >"$WORK/clean.compile" 2>&1; then
    fail "a clean program failed to compile"
    sed -n '1,40p' "$WORK/clean.compile" | sed 's/^/  /'
elif [ ! -x "$CLEAN_BIN" ]; then
    fail "a clean program compiled but emitted no executable"
else
    clean_out=$("$CLEAN_BIN")
    clean_status=$?
    if [ "$clean_status" -ne 0 ]; then
        fail "a clean program's binary exited $clean_status"
    fi
    if [ "$clean_out" != "25" ]; then
        fail "a clean program printed '$clean_out', expected '25'"
    fi
fi

if ! "$RUN" -r "$WORK/clean.esk" >"$WORK/clean_r.out" 2>"$WORK/clean_r.err"; then
    fail "a clean program failed under -r"
    sed -n '1,40p' "$WORK/clean_r.err" | sed 's/^/  /'
elif ! grep -qx "25" "$WORK/clean_r.out"; then
    fail "a clean program under -r printed $(cat "$WORK/clean_r.out"), expected 25"
fi

if [ "$FAILURES" -ne 0 ]; then
    echo "error_diagnostic_is_fatal: $FAILURES assertion(s) failed"
    exit 1
fi

echo "PASS: error diagnostics are fatal ($DIAGNOSED diagnosed candidate(s) gated)"
exit 0
