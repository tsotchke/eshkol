#!/usr/bin/env bash
# Bounded classical BLC byte observer: JIT/AOT semantic and receipt parity.
# Pass --corpus for the slower seven-program, 31-fixture transfer panel.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN="${ESHKOL_BIN:-${BUILD_DIR:-build}/eshkol-run}"
[[ "$RUN" = /* ]] || RUN="$PROJECT_ROOT/$RUN"
[[ -x "$RUN" ]] || { echo "ERROR: set ESHKOL_BIN or BUILD_DIR" >&2; exit 2; }
[[ $# -eq 0 || ( $# -eq 1 && "$1" = --corpus ) ]] || { echo "usage: $0 [--corpus]" >&2; exit 2; }
source "$SCRIPT_DIR/lib/test_isolation.sh"
eshkol_test_isolation_init blc-bytes-gate
export ESHKOL_PATH="$PROJECT_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"
export ESHKOL_LIB_DIR="$PROJECT_ROOT/lib"
export ESHKOL_JIT_CACHE=0 ESHKOL_JIT_COMPILE_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ARGS=(--no-stdlib -I "$PROJECT_ROOT/lib" -L "$PROJECT_ROOT/lib" -O0)
TEST="$PROJECT_ROOT/tests/stdlib/blc_bytes_test.esk"
WORK="$ESHKOL_TEST_TMPDIR"
if ! eshkol_test_timeout 180 "$RUN" "${ARGS[@]}" -r "$TEST" > "$WORK/jit.out" 2> "$WORK/jit.err"; then
    cat "$WORK/jit.out" "$WORK/jit.err" >&2; exit 1
fi
if ! eshkol_test_timeout 180 "$RUN" "${ARGS[@]}" -o "$WORK/unit-aot" "$TEST" > "$WORK/compile.out" 2> "$WORK/compile.err"; then
    cat "$WORK/compile.out" "$WORK/compile.err" >&2; exit 1
fi
if ! eshkol_test_timeout 180 "$WORK/unit-aot" > "$WORK/aot.out" 2> "$WORK/aot.err"; then
    cat "$WORK/aot.out" "$WORK/aot.err" >&2; exit 1
fi
for mode in jit aot; do
    grep -Fxq 'BLC-BYTES RESULT: PASS' "$WORK/$mode.out"
    if eshkol_test_output_has_failure "$WORK/$mode.out"; then cat "$WORK/$mode.out" >&2; exit 1; fi
done
cmp "$WORK/jit.out" "$WORK/aot.out"
cat "$WORK/jit.out"
if [[ $# -eq 1 ]]; then
    CORPUS="$PROJECT_ROOT/examples/blc-corpus"
    for mode in jit aot; do
        python3 "$CORPUS/run.py" run --backend "eshkol-$mode" --binary "$RUN" \
            --timeout 1800 --memory-mib 2048 --report "$WORK/$mode.toml" --work-dir "$WORK/$mode"
    done
    python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
p = Path(sys.argv[1])
a, b = [tomllib.loads((p / (mode + '.toml')).read_text()) for mode in ['jit', 'aot']]
assert a['cases'] == b['cases'] == a['passed'] == b['passed'] == 31
for x, y in zip(a['case'], b['case']):
    assert x['id'] == y['id'] and x['status'] == y['status'] == 'pass'
    assert (p / 'jit' / (x['id'] + '.stdout')).read_bytes() == (p / 'aot' / (y['id'] + '.stdout')).read_bytes()
print('BLC-BYTES CORPUS: PASS (31 exact JIT/AOT receipts)')
PY
fi
echo 'BLC-BYTES GATE: PASS'
