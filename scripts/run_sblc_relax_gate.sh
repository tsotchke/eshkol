#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${ESHKOL_BIN:-${BUILD_DIR:-build}/eshkol-run}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
source "$ROOT/scripts/lib/test_isolation.sh"
eshkol_test_isolation_init sblc-relax-gate
WORK="$ESHKOL_TEST_TMPDIR"
export ESHKOL_LIB_DIR="$ROOT/lib" ESHKOL_PATH="$ROOT/lib"
export ESHKOL_JIT_CACHE=0 ESHKOL_JIT_COMPILE_THREADS=1
ARGS=(--no-stdlib -I "$ROOT/lib" -L "$ROOT/lib" -O0)
TEST="$ROOT/tests/stdlib/sblc_relax_test.esk"
checked() {
    local log="$1"
    shift
    if ! eshkol_test_timeout 240 "$@" > "$log.out" 2> "$log.err" ||
       eshkol_test_output_has_failure "$log.out"; then
        cat "$log.out" "$log.err" >&2
        return 1
    fi
}
cd "$WORK"
checked "$WORK/jit" "$RUN" "${ARGS[@]}" -r "$TEST"
checked "$WORK/compile" "$RUN" "${ARGS[@]}" -o "$WORK/unit" "$TEST"
checked "$WORK/aot" "$WORK/unit"
grep -Fxq 'SBLC RELAX RESULT: PASS' "$WORK/jit.out"
grep -Fxq 'SBLC RELAX checks: 126' "$WORK/jit.out"
cmp "$WORK/jit.out" "$WORK/aot.out"
cat "$WORK/jit.out"
echo 'SBLC RELAX GATE: PASS'
