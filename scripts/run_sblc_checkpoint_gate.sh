#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${ESHKOL_BIN:-${BUILD_DIR:-build}/eshkol-run}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
source "$ROOT/scripts/lib/test_isolation.sh"
eshkol_test_isolation_init sblc-checkpoint-gate
WORK="$ESHKOL_TEST_TMPDIR"
export ESHKOL_LIB_DIR="$ROOT/lib" ESHKOL_PATH="$ROOT/lib" ESHKOL_JIT_CACHE=0 ESHKOL_JIT_COMPILE_THREADS=1
ARGS=(--no-stdlib -I "$ROOT/lib" -L "$ROOT/lib" -O0)
checked() {
  local log="$1"; shift
  if ! eshkol_test_timeout 240 "$@" > "$log.out" 2> "$log.err"; then cat "$log.out" "$log.err" >&2; return 1; fi
  if eshkol_test_output_has_failure "$log.out"; then cat "$log.out" "$log.err" >&2; return 1; fi
}
checked "$WORK/jit" "$RUN" "${ARGS[@]}" -r "$ROOT/tests/stdlib/sblc_checkpoint_test.esk"
checked "$WORK/compile" "$RUN" "${ARGS[@]}" -o "$WORK/unit" "$ROOT/tests/stdlib/sblc_checkpoint_test.esk"
checked "$WORK/aot" "$WORK/unit"
grep -Fxq 'SBLC CHECKPOINT RESULT: PASS' "$WORK/jit.out"
grep -Fxq 'SBLC CHECKPOINT checks: 273' "$WORK/jit.out"
cmp "$WORK/jit.out" "$WORK/aot.out"
cat > "$WORK/producer.esk" <<'EOF'
(require stdlib) (require core.sblc)
(define I (blc-lam (blc-var 1)))
(define W (blc-lam (blc-app (blc-var 1) (blc-var 1))))
(define examples (list (cons (sblc-choice I (blc-app W W)) I)))
(write (sblc-search-checkpoint (sblc-search-start 7 11 2 examples))) (newline)
EOF
cat > "$WORK/consumer.esk" <<'EOF'
(require stdlib) (require core.sblc)
(define I (blc-lam (blc-var 1)))
(define W (blc-lam (blc-app (blc-var 1) (blc-var 1))))
(define examples (list (cons (sblc-choice I (blc-app W W)) I)))
(define wire (read))
(if (not (eof-object? (read))) (exit 1))
(define result (sblc-search-resume (sblc-search-restore wire 7 11 examples) 6))
(if (not (string=? (sblc-receipt->string (sblc-search-receipt result))
                  (sblc-receipt->string (sblc-condition 7 11 6 examples)))) (exit 1))
(display "SBLC COLD RESTORE: PASS") (newline)
EOF
checked "$WORK/producer-jit" "$RUN" "${ARGS[@]}" -r "$WORK/producer.esk"
checked "$WORK/producer-build" "$RUN" "${ARGS[@]}" -o "$WORK/producer" "$WORK/producer.esk"
checked "$WORK/producer-aot" "$WORK/producer"
cmp "$WORK/producer-jit.out" "$WORK/producer-aot.out"
checked "$WORK/consumer-jit" "$RUN" "${ARGS[@]}" -r "$WORK/consumer.esk" < "$WORK/producer-aot.out"
checked "$WORK/consumer-build" "$RUN" "${ARGS[@]}" -o "$WORK/consumer" "$WORK/consumer.esk"
checked "$WORK/consumer-aot" "$WORK/consumer" < "$WORK/producer-jit.out"
grep -Fxq 'SBLC COLD RESTORE: PASS' "$WORK/consumer-jit.out"
cmp "$WORK/consumer-jit.out" "$WORK/consumer-aot.out"
cat "$WORK/jit.out" "$WORK/consumer-aot.out"
echo 'SBLC CHECKPOINT GATE: PASS (JIT/AOT and cold-process restoration)'
