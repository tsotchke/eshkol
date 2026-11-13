#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${ESHKOL_BIN:-${BUILD_DIR:-build}/eshkol-run}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
[[ -x "$RUN" ]] || { echo 'ERROR: set ESHKOL_BIN or BUILD_DIR' >&2; exit 2; }
source "$ROOT/scripts/lib/test_isolation.sh"
eshkol_test_isolation_init sblc-resume-gate
export ESHKOL_LIB_DIR="$ROOT/lib" ESHKOL_PATH="$ROOT/lib"
export ESHKOL_JIT_CACHE=0 ESHKOL_JIT_COMPILE_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ $(uname -s) = Linux ]]; then ulimit -v "${SBLC_MEMORY_KIB:-4194304}"; fi
ARGS=(--no-stdlib -I "$ROOT/lib" -L "$ROOT/lib" -O0)
checked() {
    local log="$1"; shift
    if ! eshkol_test_timeout 180 "$@" > "$log.out" 2> "$log.err"; then
        cat "$log.out" "$log.err" >&2; return 1
    fi
    if eshkol_test_output_has_failure "$log.out"; then cat "$log.out" "$log.err" >&2; return 1; fi
}
checked "$ESHKOL_TEST_TMPDIR/jit" "$RUN" "${ARGS[@]}" -r "$ROOT/tests/stdlib/sblc_resume_test.esk"
checked "$ESHKOL_TEST_TMPDIR/compile" "$RUN" "${ARGS[@]}" -o "$ESHKOL_TEST_TMPDIR/test" "$ROOT/tests/stdlib/sblc_resume_test.esk"
checked "$ESHKOL_TEST_TMPDIR/aot" "$ESHKOL_TEST_TMPDIR/test"
grep -Fxq 'SBLC RESUME RESULT: PASS' "$ESHKOL_TEST_TMPDIR/jit.out"
grep -Fxq 'SBLC RESUME RESULT: PASS' "$ESHKOL_TEST_TMPDIR/aot.out"
grep -Fxq 'SBLC RESUME checks: 530' "$ESHKOL_TEST_TMPDIR/jit.out"
cmp "$ESHKOL_TEST_TMPDIR/jit.out" "$ESHKOL_TEST_TMPDIR/aot.out"
cat "$ESHKOL_TEST_TMPDIR/jit.out"
echo 'SBLC RESUME GATE: PASS (residual continuation, JIT/AOT parity)'
