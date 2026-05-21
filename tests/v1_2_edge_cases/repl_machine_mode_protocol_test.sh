#!/bin/bash
# repl_machine_mode_protocol_test.sh — machine-mode warm worker keeps
# framing on stderr and user output on stdout.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPL="$ROOT/${BUILD_DIR:-build}/eshkol-repl"
READY_TIMEOUT="${ESHKOL_REPL_MACHINE_READY_TIMEOUT:-120}"
DONE_TIMEOUT="${ESHKOL_REPL_MACHINE_DONE_TIMEOUT:-30}"

if [ ! -x "$REPL" ]; then
    echo "SKIP: $REPL not built"
    exit 0
fi

TMPDIR_=$(mktemp -d)
trap 'rm -rf "$TMPDIR_"' EXIT

IN_FIFO="$TMPDIR_/in"
OUT_TMP="$TMPDIR_/stdout"
ERR_TMP="$TMPDIR_/stderr"
mkfifo "$IN_FIFO"
: > "$OUT_TMP"
: > "$ERR_TMP"

"$REPL" --machine < "$IN_FIFO" > "$OUT_TMP" 2> "$ERR_TMP" &
REPL_PID=$!

exec 3>"$IN_FIFO"

wait_for_ready() {
    local deadline now
    deadline=$((SECONDS + READY_TIMEOUT))
    while kill -0 "$REPL_PID" 2>/dev/null; do
        if grep -qx "EREPL READY" "$ERR_TMP"; then
            return 0
        fi
        now=$SECONDS
        if [ "$now" -ge "$deadline" ]; then
            return 1
        fi
        sleep 0.05
    done
    return 1
}

wait_for_done() {
    local deadline now
    deadline=$((SECONDS + DONE_TIMEOUT))
    while kill -0 "$REPL_PID" 2>/dev/null; do
        if grep -qx "EREPL DONE" "$ERR_TMP"; then
            return 0
        fi
        if grep -qx "EREPL FAIL" "$ERR_TMP"; then
            return 1
        fi
        now=$SECONDS
        if [ "$now" -ge "$deadline" ]; then
            return 1
        fi
        sleep 0.05
    done
    return 1
}

fail() {
    echo "FAIL: $1"
    echo "stdout:"
    sed 's/^/  /' "$OUT_TMP"
    echo "stderr:"
    sed 's/^/  /' "$ERR_TMP"
    kill "$REPL_PID" 2>/dev/null || true
    exit 1
}

wait_for_ready || fail "READY marker missing"

if [ -s "$OUT_TMP" ]; then
    fail "machine mode wrote stdout before user form"
fi

printf '%s\n' '(+ 1 2)' >&3
wait_for_done || fail "DONE marker missing after form"

exec 3>&-
wait "$REPL_PID" || true

OUT="$(cat "$OUT_TMP")"
if [ "$OUT" != "3" ]; then
    fail "stdout was not exactly user expression output"
fi

if grep -q "Goodbye" "$OUT_TMP"; then
    fail "machine mode leaked interactive goodbye text"
fi

echo "PASS: eshkol-repl --machine keeps stdout clean"
exit 0
