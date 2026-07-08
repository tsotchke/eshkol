#!/usr/bin/env bash
# Edge harness for scripts/run_sicp_smoke.sh's full-book coverage gate.
#
# This intentionally uses a fake repository and stub build/eshkol-run so the
# checks cover script behavior without requiring a real Eshkol build.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE="$ROOT/scripts/run_sicp_smoke.sh"

fail() {
    echo "FAIL: $*" >&2
    if [ -n "${LAST_OUT:-}" ] && [ -f "$LAST_OUT" ]; then
        echo "--- last output ---" >&2
        sed -n '1,160p' "$LAST_OUT" >&2
    fi
    exit 1
}

[ -f "$SMOKE" ] || fail "missing smoke script: $SMOKE"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-sicp-smoke-gate.XXXXXX")" || exit 1
cleanup() {
    [ -n "${WORK:-}" ] && [ -d "$WORK" ] && rm -rf -- "$WORK"
}
trap cleanup EXIT

FAKE="$WORK/repo"
TRACE="$FAKE/scripts/icc_traces/sicp_smoke.jsonl"
LAST_OUT=""
LAST_RC=0

setup_fake_repo() {
    mkdir -p "$FAKE/scripts" "$FAKE/build" "$FAKE/tests/sicp"
    cp "$SMOKE" "$FAKE/scripts/run_sicp_smoke.sh" || fail "copy smoke script"
    chmod +x "$FAKE/scripts/run_sicp_smoke.sh"

    cat > "$FAKE/tests/sicp/ch1_newton.esk" <<'ESK'
(display "fake sicp probe") (newline)
ESK

    write_fake_runner "$FAKE/build/eshkol-run" pass
}

write_fake_runner() {
    local path="$1"
    local mode="$2"
    mkdir -p "$(dirname "$path")"
    cat > "$path" <<'SH_HEAD'
#!/usr/bin/env bash
set -u

SH_HEAD
    printf 'mode="%s"\n' "$mode" >> "$path"
    cat >> "$path" <<'SH_TAIL'
out=""
is_jit=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        -r)
            is_jit=1
            ;;
        -o)
            shift
            [ "$#" -gt 0 ] || exit 2
            out="$1"
            ;;
    esac
    shift || true
done

if [ "$mode" = "jit_fail" ] && [ "$is_jit" -eq 1 ]; then
    echo "FAIL: forced jit failure"
    exit 0
fi

if [ -n "$out" ]; then
    if [ "$mode" = "aot_no_binary" ]; then
        echo "fake compile success without output"
        exit 0
    fi
    cat > "$out" <<'BIN'
#!/usr/bin/env bash
echo "fake sicp aot pass"
exit 0
BIN
    chmod +x "$out"
    exit 0
fi

echo "fake sicp jit pass"
exit 0
SH_TAIL
    chmod +x "$path"
}

run_smoke() {
    local label="$1"
    shift
    LAST_OUT="$WORK/${label}.out"
    : "${LAST_OUT:?LAST_OUT must be set}"
    # The smoke script's timeout shim uses perl; force a portable locale so
    # macOS hosts without C.UTF-8 do not fail before the fake runner executes.
    if [ -n "${RUN_BUILD_DIR:-}" ]; then
        BUILD_DIR="$RUN_BUILD_DIR" \
            ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache-$label" \
            LC_ALL=C LC_CTYPE=C LANG=C \
            "$FAKE/scripts/run_sicp_smoke.sh" "$@" >"$LAST_OUT" 2>&1
    else
        ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache-$label" \
            LC_ALL=C LC_CTYPE=C LANG=C \
            "$FAKE/scripts/run_sicp_smoke.sh" "$@" >"$LAST_OUT" 2>&1
    fi
    LAST_RC=$?
}

assert_rc() {
    local expected="$1"
    [ "$LAST_RC" -eq "$expected" ] ||
        fail "expected exit $expected, got $LAST_RC"
}

assert_out_has() {
    local pattern="$1"
    grep -Eq "$pattern" "$LAST_OUT" ||
        fail "missing output pattern: $pattern"
}

assert_trace_has() {
    local pattern="$1"
    [ -f "$TRACE" ] || fail "trace was not written: $TRACE"
    grep -Eq "$pattern" "$TRACE" ||
        fail "missing trace pattern: $pattern"
}

trace_count() {
    local pattern="$1"
    if [ ! -f "$TRACE" ]; then
        echo 0
        return
    fi
    grep -Ec "$pattern" "$TRACE" || true
}

assert_trace_count() {
    local pattern="$1"
    local expected="$2"
    local actual
    actual="$(trace_count "$pattern")"
    [ "$actual" -eq "$expected" ] ||
        fail "expected $expected trace matches for $pattern, got $actual"
}

setup_fake_repo

write_fake_runner "$FAKE/custom-build/eshkol-run" pass
RUN_BUILD_DIR=custom-build run_smoke custom_build --allow-incomplete
unset RUN_BUILD_DIR
assert_rc 0
assert_out_has 'PASS  tests/sicp/ch1_newton\.esk::r'
assert_out_has 'PASS  tests/sicp/ch1_newton\.esk::aot'

run_smoke default
assert_rc 1
assert_out_has 'Full-book SICP gate: 22 required system probes missing\.'
assert_out_has 'MISSING tests/sicp/ch2_picture_painters\.esk::r'
assert_out_has 'MISSING tests/sicp/ch3_digital_circuit\.esk::r'
assert_out_has 'MISSING tests/sicp/ch4_lazy_evaluator\.esk::r'
assert_out_has 'MISSING tests/sicp/ch4_lazy_evaluator\.esk::aot'
assert_trace_count '"value":"FAIL"' 44
assert_trace_has '"name":"sicp_ch2_picture_painters_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch3_digital_circuit_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch4_lazy_evaluator_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch4_lazy_evaluator_aot","value":"FAIL"'

run_smoke allow_incomplete --allow-incomplete
assert_rc 0
assert_out_has 'Incomplete coverage accepted only because --allow-incomplete was provided\.'
assert_trace_count '"value":"FAIL"' 44
assert_trace_has '"name":"sicp_ch2_polynomial_arithmetic_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch3_concurrency_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch5_compiler_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch5_compiler_aot","value":"FAIL"'

write_fake_runner "$FAKE/build/eshkol-run" jit_fail
run_smoke allow_incomplete_real_failure --allow-incomplete
assert_rc 1
assert_out_has 'FAILED tests/sicp/ch1_newton\.esk::r'

write_fake_runner "$FAKE/build/eshkol-run" aot_no_binary
run_smoke missing_aot_binary --allow-incomplete
assert_rc 1
assert_trace_has '"name":"sicp_ch1_newton_aot","value":"FAIL"'
if grep -Eq '"name":"sicp_ch1_newton_aot","value":"PASS"' "$TRACE"; then
    fail "missing AOT binary was reported as PASS"
fi

write_fake_runner "$FAKE/build/eshkol-run" pass
run_smoke no_aot --no-aot
assert_rc 1
assert_trace_count '"value":"FAIL"' 22
assert_trace_has '"name":"sicp_ch3_constraints_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch4_amb_parser_r","value":"FAIL"'
assert_trace_has '"name":"sicp_ch4_query_system_r","value":"FAIL"'
if grep -q '_aot' "$TRACE"; then
    fail "--no-aot emitted AOT trace events"
fi

run_smoke unknown_arg --bogus
assert_rc 2
assert_out_has 'run_sicp_smoke\.sh: unknown argument: --bogus'

echo "PASS: run_sicp_smoke gate edge cases"
