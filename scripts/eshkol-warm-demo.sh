#!/usr/bin/env bash
# eshkol-warm-demo.sh — show the eshkol-repl --machine warm-worker
# protocol from the client side. Useful as a copy-paste template
# for sister projects (Noesis, eshkol-agent, etc.) that want to
# reuse a single JIT-warm REPL across many code submissions.
#
# Protocol:
#   1. Spawn `eshkol-repl --machine`
#   2. Read STDERR until you see a line matching "EREPL READY"
#      → JIT + stdlib are warm, you can start sending forms.
#   3. Send a form on STDIN (newline-terminated, parens balanced).
#   4. Read STDOUT for that form's output, AND read STDERR for the
#      next "EREPL DONE" or "EREPL FAIL" line, which tells you the
#      submission has finished. Per-form output never crosses
#      stderr; framing markers never cross stdout.
#   5. Repeat (3)-(4) for each new form.  No more cold starts.
#
# Why stderr for framing: keeps stdout 100% pure user-program output,
# so binary writes (P6 PPM, etc.) and JSON-L emission round-trip
# without contamination.
#
# This shell driver is illustrative; in practice you'd use a
# language-native subprocess library (Python's subprocess,
# Node's child_process, Rust's std::process) for non-blocking
# read-line semantics.

set -euo pipefail

ESHKOL_REPL="${ESHKOL_REPL:-./build/eshkol-repl}"

if [ ! -x "$ESHKOL_REPL" ]; then
    echo "Build eshkol-repl first: cmake --build build --target eshkol-repl" >&2
    exit 1
fi

# Use a named pipe for stdin and regular files for stdout/stderr.  A
# FIFO stdout with no reader can block the REPL before it reaches
# READY, so the demo keeps output in files and prints it after exit.
TMPDIR_=$(mktemp -d)
trap 'rm -rf "$TMPDIR_"' EXIT
IN_FIFO="$TMPDIR_/in"
OUT_LOG="$TMPDIR_/stdout"
ERR_LOG="$TMPDIR_/stderr"
mkfifo "$IN_FIFO"
: > "$OUT_LOG"
: > "$ERR_LOG"

# Spawn the REPL with machine framing on stderr and pure program output
# on stdout.
"$ESHKOL_REPL" --machine < "$IN_FIFO" > "$OUT_LOG" 2> "$ERR_LOG" &
REPL_PID=$!

# Open writer end of the input pipe so it doesn't EOF between writes.
exec 3>"$IN_FIFO"

wait_for_ready() {
    local deadline now
    deadline=$((SECONDS + 120))
    while kill -0 "$REPL_PID" 2>/dev/null; do
        if grep -qx "EREPL READY" "$ERR_LOG"; then
            return 0
        fi
        if grep -qx "EREPL FAIL" "$ERR_LOG"; then
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

wait_for_done_after() {
    local before deadline now count marker
    before="$1"
    deadline=$((SECONDS + 30))
    while kill -0 "$REPL_PID" 2>/dev/null; do
        count=$(grep -Ec '^EREPL (DONE|FAIL)$' "$ERR_LOG" || true)
        if [ "$count" -gt "$before" ]; then
            marker=$(grep -E '^EREPL (DONE|FAIL)$' "$ERR_LOG" | tail -n 1)
            [ "$marker" = "EREPL DONE" ]
            return $?
        fi
        now=$SECONDS
        if [ "$now" -ge "$deadline" ]; then
            return 1
        fi
        sleep 0.05
    done
    return 1
}

send_form() {
    local form before
    form="$1"
    before=$(grep -Ec '^EREPL (DONE|FAIL)$' "$ERR_LOG" || true)
    printf '%s\n' "$form" >&3
    wait_for_done_after "$before"
}

echo "[demo] waiting for JIT to warm up..." >&2
if ! wait_for_ready; then
    echo "[demo] startup failed or timed out" >&2
    sed 's/^/[repl-stderr] /' "$ERR_LOG" >&2
    kill "$REPL_PID" 2>/dev/null || true
    exit 1
fi
echo "[demo] warm" >&2

# Send forms; each ends with a newline.  In machine mode the
# REPL reads until parens balance, so multi-line forms work too.
send_form '(display "hello from warm worker")'
send_form '(newline)'
send_form '(define x 42)'
send_form '(display (* x x))'
send_form '(newline)'

# Close writer side → REPL sees EOF, exits cleanly.
exec 3>&-

wait "$REPL_PID" || true
cat "$OUT_LOG"
echo "[demo] done" >&2
