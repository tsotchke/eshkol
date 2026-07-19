#!/usr/bin/env bash
# GitHub #297: pure-AD plus opt-in Moonlab quantum-chemistry examples.
set -u

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) : ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"

: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-gh297-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_quantum_chemistry_examples_gate.sh: missing $ESHKOL_RUN" >&2
    exit 2
fi

RUN_TIMEOUT="${RUN_TIMEOUT:-600}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-600}"

run_guarded() {
    perl -e 'my $seconds=shift; alarm $seconds; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

extract_value() {
    local label="$1"
    awk -v label="$label" 'index($0, label) == 1 { print $(NF-1); exit }'
}

in_range() {
    perl -e 'my ($value,$low,$high)=@ARGV; exit !(defined($value) && $value >= $low && $value <= $high)' -- \
        "$1" "$2" "$3"
}

overall=PASS
echo "== GitHub #297 differentiable quantum chemistry gate =="

pure_source="$REPO_ROOT/examples/h2_vibrational.esk"
pure_output="$(run_guarded "$RUN_TIMEOUT" "$ESHKOL_RUN" -r "$pure_source" -L"$BUILD_DIR" 2>&1)"
pure_rc=$?
pure_frequency="$(printf '%s\n' "$pure_output" | extract_value 'vibrational frequency =')"
if [ "$pure_rc" -eq 0 ] && in_range "$pure_frequency" 4990 5020; then
    echo "  pure AD JIT PASS (omega=$pure_frequency cm^-1)"
else
    echo "  pure AD JIT FAIL (rc=$pure_rc omega=${pure_frequency:-missing})" >&2
    printf '%s\n' "$pure_output" >&2
    overall=FAIL
fi

aot_binary="$(mktemp "${TMPDIR:-/tmp}/eshkol-h2-vibration-aot.XXXXXX")"
trap 'rm -f "$aot_binary"' EXIT
compile_output="$(run_guarded "$COMPILE_TIMEOUT" "$ESHKOL_RUN" "$pure_source" -o "$aot_binary" -L"$BUILD_DIR" 2>&1)"
compile_rc=$?
if [ "$compile_rc" -eq 0 ] && [ -x "$aot_binary" ]; then
    aot_output="$(run_guarded "$RUN_TIMEOUT" "$aot_binary" 2>&1)"
    aot_rc=$?
    aot_frequency="$(printf '%s\n' "$aot_output" | extract_value 'vibrational frequency =')"
    if [ "$aot_rc" -eq 0 ] && in_range "$aot_frequency" 4990 5020; then
        echo "  pure AD AOT PASS (omega=$aot_frequency cm^-1)"
    else
        echo "  pure AD AOT FAIL (rc=$aot_rc omega=${aot_frequency:-missing})" >&2
        printf '%s\n' "$aot_output" >&2
        overall=FAIL
    fi
else
    echo "  pure AD AOT COMPILE FAIL (rc=$compile_rc)" >&2
    printf '%s\n' "$compile_output" >&2
    overall=FAIL
fi

quantum_enabled=0
if [ -f "$BUILD_DIR/CMakeCache.txt" ] &&
   grep -q '^ESHKOL_QUANTUM_ENABLED:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt"; then
    quantum_enabled=1
fi

if [ "$quantum_enabled" -eq 0 ]; then
    if [ "${REQUIRE_QUANTUM:-0}" = 1 ]; then
        echo "  Moonlab examples FAIL: REQUIRE_QUANTUM=1 but build is not quantum-enabled" >&2
        overall=FAIL
    else
        echo "  Moonlab examples SKIP (build is not configured with ESHKOL_QUANTUM_ENABLED=ON)"
    fi
else
    api_output="$(run_guarded "$RUN_TIMEOUT" "$ESHKOL_RUN" -r \
        "$REPO_ROOT/tests/quantum/quantum_chemistry_api_test.esk" -L"$BUILD_DIR" 2>&1)"
    api_rc=$?
    if [ "$api_rc" -eq 0 ] && printf '%s\n' "$api_output" | grep -q '^QUANTUM_CHEMISTRY_API_PASS$'; then
        echo "  generic Pauli/QGT API PASS"
    else
        echo "  generic Pauli/QGT API FAIL (rc=$api_rc)" >&2
        printf '%s\n' "$api_output" >&2
        overall=FAIL
    fi

    for example in vqe_h2 h2_vibrational_quantum h2_vibrational_full qng_vqe; do
        example_output="$(run_guarded "$RUN_TIMEOUT" "$ESHKOL_RUN" -r \
            "$REPO_ROOT/examples/$example.esk" -L"$BUILD_DIR" 2>&1)"
        example_rc=$?
        case "$example" in
            vqe_h2)
                numeric_label='Final VQE energy:'
                numeric_low=-1.143
                numeric_high=-1.141
                ;;
            h2_vibrational_quantum)
                numeric_label='  omega'
                numeric_low=4900
                numeric_high=5000
                ;;
            h2_vibrational_full)
                numeric_label='vibrational frequency ='
                numeric_low=4990
                numeric_high=5020
                ;;
            qng_vqe)
                numeric_label='final QNG:'
                numeric_low=-1.143
                numeric_high=-1.142
                ;;
        esac
        numeric_value="$(printf '%s\n' "$example_output" | extract_value "$numeric_label")"
        if [ "$example_rc" -eq 0 ] &&
           in_range "$numeric_value" "$numeric_low" "$numeric_high"; then
            echo "  $example PASS (value=$numeric_value)"
        else
            echo "  $example FAIL (rc=$example_rc value=${numeric_value:-missing})" >&2
            printf '%s\n' "$example_output" >&2
            overall=FAIL
        fi
    done
fi

echo "GitHub #297 differentiable quantum chemistry gate: $overall"
[ "$overall" = PASS ]
