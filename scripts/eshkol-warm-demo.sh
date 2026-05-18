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

# Use named pipes so we can drive stdin/stdout/stderr independently.
TMPDIR_=$(mktemp -d)
trap 'rm -rf "$TMPDIR_"' EXIT
IN_TMP="$TMPDIR_/in"
OUT_TMP="$TMPDIR_/out"
ERR_TMP="$TMPDIR_/err"
mkfifo "$IN_TMP" "$OUT_TMP" "$ERR_TMP"

# Spawn the REPL with all three streams piped to the named pipes.
"$ESHKOL_REPL" --machine < "$IN_TMP" > "$OUT_TMP" 2> "$ERR_TMP" &
REPL_PID=$!

# Open writer end of the input pipe so it doesn't EOF between writes.
exec 3>"$IN_TMP"

# Wait for READY on stderr.
echo "[demo] waiting for JIT to warm up..." >&2
while IFS= read -r line < "$ERR_TMP"; do
    case "$line" in
        "EREPL READY") echo "[demo] warm" >&2 ; break ;;
        EREPL\ FAIL)   echo "[demo] startup FAIL: $line" >&2 ; exit 1 ;;
    esac
done &
READER_PID=$!

# Send three forms; each ends with a newline.  In machine mode the
# REPL reads until parens balance, so multi-line forms work too.
echo '(display "hello from warm worker")' >&3
echo '(newline)' >&3
echo '(define x 42)' >&3
echo '(display (* x x))(newline)' >&3

# Close writer side → REPL sees EOF, exits cleanly.
exec 3>&-

wait "$REPL_PID" || true
wait "$READER_PID" 2>/dev/null || true
echo "[demo] done. Output captured in $OUT_TMP (pipe; consumed by reader)" >&2
