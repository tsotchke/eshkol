#!/bin/bash
# object_emit_phase_trace_test.sh — ESH-0088 object-emission diagnostics.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_object_emit_trace.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

SRC="$WORK/object_emit_phase_trace.esk"
OUT="$WORK/object_emit_phase_trace"
LOG="$WORK/object_emit_phase_trace.log"

cat > "$SRC" <<'ESK'
(define (square x) (* x x))
(square 7)
ESK

if ! env LC_ALL=C LANG=C \
    ESHKOL_AOT_PHASE_TRACE=1 \
    ESHKOL_OBJECT_EMIT_TIMEOUT_SECONDS=30 \
    "$RUN" --no-stdlib --compile-only --output "$OUT" "$SRC" >"$LOG" 2>&1; then
    echo "FAIL: compile-only trace probe exited nonzero"
    sed -n '1,160p' "$LOG" | sed 's/^/  /'
    exit 1
fi

if ! grep -qF "phase=pass_manager.run" "$LOG"; then
    echo "FAIL: pass_manager.run trace missing"
    sed -n '1,160p' "$LOG" | sed 's/^/  /'
    exit 1
fi

if ! grep -qF "phase=publish_object" "$LOG"; then
    echo "FAIL: publish_object trace missing"
    sed -n '1,160p' "$LOG" | sed 's/^/  /'
    exit 1
fi

if grep -qF "watchdog timed out" "$LOG"; then
    echo "FAIL: watchdog fired on small object emission"
    sed -n '1,160p' "$LOG" | sed 's/^/  /'
    exit 1
fi

if [ ! -s "$OUT.o" ]; then
    echo "FAIL: expected non-empty object at $OUT.o"
    sed -n '1,160p' "$LOG" | sed 's/^/  /'
    exit 1
fi

echo "PASS: object emission phase trace"
