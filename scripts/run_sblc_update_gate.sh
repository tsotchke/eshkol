#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${ESHKOL_BIN:-${BUILD_DIR:-build}/eshkol-run}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
source "$ROOT/scripts/lib/test_isolation.sh"
eshkol_test_isolation_init sblc-update-gate
WORK="$ESHKOL_TEST_TMPDIR"
export ESHKOL_LIB_DIR="$ROOT/lib" ESHKOL_PATH="$ROOT/lib" ESHKOL_JIT_CACHE=0 ESHKOL_JIT_COMPILE_THREADS=1
ARGS=(--no-stdlib -I "$ROOT/lib" -L "$ROOT/lib" -O0)
checked() {
 local log="$1"; shift
 if ! eshkol_test_timeout 240 "$@" > "$log.out" 2> "$log.err"; then cat "$log.out" "$log.err" >&2; return 1; fi
 if eshkol_test_output_has_failure "$log.out"; then cat "$log.out" "$log.err" >&2; return 1; fi
}
checked "$WORK/jit" "$RUN" "${ARGS[@]}" -r "$ROOT/tests/stdlib/sblc_update_test.esk"
checked "$WORK/compile" "$RUN" "${ARGS[@]}" -o "$WORK/unit" "$ROOT/tests/stdlib/sblc_update_test.esk"
checked "$WORK/aot" "$WORK/unit"
grep -Fxq 'SBLC UPDATE RESULT: PASS' "$WORK/jit.out"
grep -Fxq 'SBLC UPDATE checks: 61' "$WORK/jit.out"
cmp "$WORK/jit.out" "$WORK/aot.out"
cat > "$WORK/producer.esk" <<'EOF'
(require stdlib) (require core.sblc)
(define I (blc-lam (blc-var 1)))
(write (sblc-search-checkpoint (sblc-search-start 7 4 0 (list (cons I I))))) (newline)
EOF
cat > "$WORK/consumer.esk" <<'EOF'
(require stdlib) (require core.sblc) (require core.dbsp)
(define I (blc-lam (blc-var 1)))
(define input (read)) (if (not (eof-object? (read))) (exit 1))
(define old (sblc-search-restore input 7 4 (list (cons I I))))
(define update (sblc-search-update old (zset-singleton (list I I) 1) 14 3))
(if (not (string=? (sblc-receipt->string (sblc-search-receipt (sblc-update-search update)))
                  (sblc-receipt->string (sblc-condition 7 14 3 (list (cons I I) (cons I I)))))) (exit 1))
(display "SBLC COLD UPDATE: PASS") (newline)
EOF
checked "$WORK/producer" "$RUN" "${ARGS[@]}" -r "$WORK/producer.esk"
checked "$WORK/cold-jit" "$RUN" "${ARGS[@]}" -r "$WORK/consumer.esk" < "$WORK/producer.out"
checked "$WORK/cold-build" "$RUN" "${ARGS[@]}" -o "$WORK/cold" "$WORK/consumer.esk"
checked "$WORK/cold-aot" "$WORK/cold" < "$WORK/producer.out"
grep -Fxq 'SBLC COLD UPDATE: PASS' "$WORK/cold-jit.out"
cmp "$WORK/cold-jit.out" "$WORK/cold-aot.out"
cat "$WORK/jit.out" "$WORK/cold-aot.out"
echo 'SBLC UPDATE GATE: PASS (DBSP observer, JIT/AOT and cold update)'
