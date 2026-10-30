#!/usr/bin/env bash
# Copyright (C) 2026 Richard Hoekstra. SPDX-License-Identifier: MIT
# Qualify the library AND separately compiled emitted residuals on JIT and AOT.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${ESHKOL_BIN:-${BUILD_DIR:-build}/eshkol-run}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
[[ -x "$RUN" ]] || { echo 'ERROR: set ESHKOL_BIN or BUILD_DIR' >&2; exit 2; }
source "$ROOT/scripts/lib/test_isolation.sh"
eshkol_test_isolation_init mix-gate
WORK="$ESHKOL_TEST_TMPDIR"
export ESHKOL_LIB_DIR="$ROOT/lib" ESHKOL_PATH="$ROOT/lib:$ROOT/tests${ESHKOL_PATH:+:$ESHKOL_PATH}"
export ESHKOL_JIT_CACHE=0 ESHKOL_JIT_COMPILE_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# External address-space admission is separate from the interpreter counters.
if [[ $(uname -s) = Linux ]]; then ulimit -v "${MIX_MEMORY_KIB:-4194304}"; fi
ARGS=(--no-stdlib -I "$ROOT/lib" -I "$ROOT/tests" -L "$ROOT/lib" -O0)
run_checked() {
    local log="$1"; shift
    if ! eshkol_test_timeout 600 "$@" > "$log.out" 2> "$log.err"; then
        cat "$log.out" "$log.err" >&2; return 1
    fi
    if eshkol_test_output_has_failure "$log.out"; then cat "$log.out" "$log.err" >&2; return 1; fi
}
run_checked "$WORK/unit-jit" "$RUN" "${ARGS[@]}" -r "$ROOT/tests/stdlib/mix_test.esk"
run_checked "$WORK/unit-build" "$RUN" "${ARGS[@]}" -o "$WORK/unit" "$ROOT/tests/stdlib/mix_test.esk"
run_checked "$WORK/unit-aot" "$WORK/unit"
grep -Fxq 'MIX RESULT: PASS' "$WORK/unit-jit.out"
grep -Fxq 'MIX RESULT: PASS' "$WORK/unit-aot.out"
cmp "$WORK/unit-jit.out" "$WORK/unit-aot.out"
# These forms are actual F3 outputs plus independent fixture-answer checks.
sed -n 's/^MIX-SOURCE //p' "$WORK/unit-jit.out" > "$WORK/residuals.esk"
complete_residual_panel() {
    [[ $(grep -c '^(provide mix-output-' "$1") = 14 &&
       $(grep -c '^(if (not (= ' "$1") = 571 ]]
}
complete_residual_panel "$WORK/residuals.esk"
# A damaged observation must not inherit the complete panel's printed verdict.
awk 'BEGIN { removed=0 } /^\(if \(not \(= / && !removed { removed=1; next } { print }' \
    "$WORK/residuals.esk" > "$WORK/missing-assertion.esk"
if complete_residual_panel "$WORK/missing-assertion.esk"; then
    echo 'FAIL: residual panel accepted a missing assertion' >&2; exit 1
fi
printf '%s\n' '(display "MIX RESIDUALS: PASS (571 comparisons)") (newline)' >> "$WORK/residuals.esk"
run_checked "$WORK/residual-jit" "$RUN" "${ARGS[@]}" -r "$WORK/residuals.esk"
run_checked "$WORK/residual-build" "$RUN" "${ARGS[@]}" -o "$WORK/residuals" "$WORK/residuals.esk"
run_checked "$WORK/residual-aot" "$WORK/residuals"
grep -Fxq 'MIX RESIDUALS: PASS (571 comparisons)' "$WORK/residual-jit.out"
cmp "$WORK/residual-jit.out" "$WORK/residual-aot.out"
# Execute self-hosted normalization, then compile its emitted standalone module.
run_checked "$WORK/generator-jit" "$RUN" "${ARGS[@]}" -r "$ROOT/examples/mix/generate.esk"
run_checked "$WORK/generator-build" "$RUN" "${ARGS[@]}" -o "$WORK/generator" "$ROOT/examples/mix/generate.esk"
run_checked "$WORK/generator-aot" "$WORK/generator"
cmp "$WORK/generator-jit.out" "$WORK/generator-aot.out"
cp "$WORK/generator-jit.out" "$WORK/self.esk"
cat >> "$WORK/self.esk" <<'EOF'
(let loop ((x -20))
  (if (< x 20)
      (begin (if (not (= (mix-generated x) (+ (* x x) 4))) (exit 1)) (loop (+ x 1)))))
(if (not (= (mix-generated 4294967296) 18446744073709551620)) (exit 1))
(display "MIX SELF-HOSTED: PASS (41 comparisons)") (newline)
EOF
run_checked "$WORK/self-jit" "$RUN" "${ARGS[@]}" -r "$WORK/self.esk"
run_checked "$WORK/self-build" "$RUN" "${ARGS[@]}" -o "$WORK/self" "$WORK/self.esk"
run_checked "$WORK/self-aot" "$WORK/self"
grep -Fxq 'MIX SELF-HOSTED: PASS (41 comparisons)' "$WORK/self-jit.out"
cmp "$WORK/self-jit.out" "$WORK/self-aot.out"
# Entry validation survives even when arithmetic could otherwise accept a real.
cp "$WORK/generator-jit.out" "$WORK/bad.esk"
printf '%s\n' '(mix-erased 1.5)' '(display "UNEXPECTED ACCEPTANCE") (newline)' >> "$WORK/bad.esk"
eshkol_test_timeout 60 "$RUN" "${ARGS[@]}" -r "$WORK/bad.esk" > "$WORK/bad-jit.out" 2> "$WORK/bad-jit.err" || true
! grep -Fq 'UNEXPECTED ACCEPTANCE' "$WORK/bad-jit.out"
grep -Fq 'mix residual requires exact integer input' "$WORK/bad-jit.err"
run_checked "$WORK/bad-build" "$RUN" "${ARGS[@]}" -o "$WORK/bad" "$WORK/bad.esk"
eshkol_test_timeout 60 "$WORK/bad" > "$WORK/bad-aot.out" 2> "$WORK/bad-aot.err" || true
! grep -Fq 'UNEXPECTED ACCEPTANCE' "$WORK/bad-aot.out"
grep -Fq 'mix residual requires exact integer input' "$WORK/bad-aot.err"
sed -n '/^MIX comparisons:/p;/^MIX checks:/p;/^MIX RESULT:/p' "$WORK/unit-jit.out"
cat "$WORK/residual-jit.out" "$WORK/self-jit.out"
echo 'MIX GATE: PASS (library, emitted code, JIT/AOT parity, integer-entry refusal)'
