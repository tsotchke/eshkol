#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
TEST_ESK="$REPO_ROOT/tests/gpu/ozaki_certification_test.esk"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/ozaki_certification.jsonl"

emit_trace() { # emit_trace <value> <reason>
    local value="$1"
    local reason="$2"
    mkdir -p "$TRACE_DIR"
    local snippet
    if command -v python3 >/dev/null 2>&1; then
        snippet="$(printf '%s' "$reason" | python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.stdin.read()))')"
    else
        snippet="$(printf '%s' "$reason" | sed -e 's/\\/\\\\/g' -e 's/"/\\\"/g')"
        snippet="\"$snippet\""
    fi
    printf '{"kind":"ozaki_certification","name":"ozaki_certification_gate","value":"%s","confidence":0.95,"snippet":%s}\n' \
        "$value" "$snippet" >> "$TRACE_FILE"
}

log() { printf '%s\n' "$*"; }
skip() { log "SKIP: $*"; exit 0; }
fail() { log "FAIL: $*"; emit_trace FAIL "$*"; exit 1; }
pass() { log "PASS: $*"; emit_trace PASS "$*"; exit 0; }

UNAME_S="$(uname -s)"
[ "$UNAME_S" = "Darwin" ] || skip "Ozaki certification requires macOS/Metal"

BIN="${ESHKOL_RUN:-}"
if [ -z "$BIN" ]; then
    for d in build-ozaki build build-gpu-gate build-*; do
        if [ -x "$REPO_ROOT/$d/eshkol-run" ]; then
            BIN="$REPO_ROOT/$d/eshkol-run"
            break
        fi
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || skip "no eshkol-run binary found (set ESHKOL_RUN)"
log "using eshkol-run: $BIN"

if ! command -v otool >/dev/null 2>&1; then
    fail "otool unavailable in PATH"
fi

otool -L "$BIN" | grep -q '/Accelerate\.framework/' || fail "eshkol-run is not linked against Accelerate"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-ozaki-certification.XXXXXX")"
cleanup_tmp() {
    if [ -n "${TMP_DIR:-}" ] && [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
    fi
}
trap cleanup_tmp EXIT
JIT_CACHE_BLAS_DIR="$TMP_DIR/jit-cache-blas"
JIT_CACHE_EXACT_DIR="$TMP_DIR/jit-cache-exact"
mkdir -p "$JIT_CACHE_BLAS_DIR" "$JIT_CACHE_EXACT_DIR"

AOT_BIN="$TMP_DIR/ozaki_certification_aot"
"$BIN" "$TEST_ESK" -o "$AOT_BIN"
aot_compile_rc=$?
[ "$aot_compile_rc" -eq 0 ] || fail "AOT compilation exited with status $aot_compile_rc"
[ -x "$AOT_BIN" ] || fail "AOT binary missing or not executable: $AOT_BIN"
otool -L "$AOT_BIN" | grep -q '/Accelerate\.framework/' || fail "AOT binary is not linked against Accelerate"

# -r exercises the default cached-object path; use separate JIT cache dirs per mode so
# BLAS and exact lanes compile/execute independently but remain cached-path true.
BLAS_ENV=(
  ESHKOL_OZAKI_CERT_MODE=blas
  ESHKOL_GPU_MATMUL_THRESHOLD=999999999999
  ESHKOL_GPU_THRESHOLD=999999999999
  ESHKOL_GPU_VERBOSE=1
  ESHKOL_JIT_CACHE_DIR="$JIT_CACHE_BLAS_DIR"
)

EXACT_ENV=(
  ESHKOL_OZAKI_CERT_MODE=exact
  ESHKOL_GPU_MATMUL_THRESHOLD=0
  ESHKOL_GPU_THRESHOLD=1
  ESHKOL_GPU_VERBOSE=1
  ESHKOL_BLAS_PEAK_GFLOPS=0.001
  ESHKOL_GPU_PEAK_GFLOPS=1000000
  ESHKOL_GPU_WAIT_TIMEOUT=300
  ESHKOL_SF64_KERNEL=ozaki
  ESHKOL_OZAKI_NUM_MODULI=16
  ESHKOL_OZAKI_ADAPTIVE=0
  ESHKOL_JIT_CACHE_DIR="$JIT_CACHE_EXACT_DIR"
)

BLAS_SUMMARY_RE='^SUMMARY mode=blas total_samples=25 mismatches=[1-9][0-9]* max_dot_bits=58 verdict=PASS$'
EXACT_SUMMARY_RE='^SUMMARY mode=exact total_samples=25 mismatches=0 max_dot_bits=58 verdict=PASS$'
INIT_LINE_RE='^\[GPU\] Ozaki-II: N=16 moduli, log2\(P\)='
DISPATCH_LINE_RE='^\[GPU\] Ozaki-II: N=16, log2\(P\)='
REJECT_DISPATCH_RE='^\[GPU\] Ozaki-II: N=[0-9]+( \(adaptive\))?, log2\(P\)='
REJECT_FALLBACK='matmul dispatch failed, falling back to CPU BLAS'

RUN_COUNTER=0
RUN_OUT=""
RUN_ERR=""
RUN_RC=0
run_case() {  # run_case <label> <env-array-name> <cmd...>
    local label="$1"
    local env_name="$2"
    shift 2
    local env_cmd=()
    case "$env_name" in
        BLAS_ENV) env_cmd=("${BLAS_ENV[@]}") ;;
        EXACT_ENV) env_cmd=("${EXACT_ENV[@]}") ;;
        *) fail "unknown env array: $env_name" ;;
    esac

    RUN_COUNTER=$((RUN_COUNTER + 1))
    RUN_OUT="$TMP_DIR/ozaki-certification-run-${RUN_COUNTER}.out"
    RUN_ERR="$TMP_DIR/ozaki-certification-run-${RUN_COUNTER}.err"

    env "${env_cmd[@]}" "$@" >"$RUN_OUT" 2>"$RUN_ERR"
    RUN_RC=$?

    log "--- $label stdout ---"
    cat "$RUN_OUT"
    log "--- $label stderr ---"
    cat "$RUN_ERR"
}

extract_mismatches() { # extract_mismatches <summary-line>
    printf '%s\n' "$1" | sed -nE 's/^SUMMARY mode=[a-z]+ total_samples=25 mismatches=([0-9]+) .*/\1/p'
}

run_case "JIT BLAS" BLAS_ENV "$BIN" -r "$TEST_ESK"
BLAS_JIT_RC="$RUN_RC"
BLAS_JIT_SUMMARY="$(cat "$RUN_OUT")"
BLAS_JIT_ERR="$RUN_ERR"
[ "$BLAS_JIT_RC" -eq 0 ] || fail "JIT BLAS execution failed with status $BLAS_JIT_RC"
printf '%s\n' "$BLAS_JIT_SUMMARY" | grep -qE "$BLAS_SUMMARY_RE" || fail "JIT BLAS summary did not match expected Accelerate mismatch contract"
BLAS_JIT_MISMATCHES="$(extract_mismatches "$BLAS_JIT_SUMMARY")"
[ -n "$BLAS_JIT_MISMATCHES" ] || fail "JIT BLAS summary missing mismatch count"
[ "$BLAS_JIT_MISMATCHES" -gt 0 ] || fail "JIT BLAS mismatches must be > 0"
grep -Eq "$REJECT_DISPATCH_RE" "$BLAS_JIT_ERR" && fail "JIT BLAS execution printed forbidden Ozaki-II real dispatch line"
grep -Fq "$REJECT_FALLBACK" "$BLAS_JIT_ERR" && fail "JIT BLAS execution fell back to CPU BLAS"

run_case "AOT BLAS" BLAS_ENV "$AOT_BIN"
BLAS_AOT_RC="$RUN_RC"
BLAS_AOT_SUMMARY="$(cat "$RUN_OUT")"
BLAS_AOT_ERR="$RUN_ERR"
[ "$BLAS_AOT_RC" -eq 0 ] || fail "AOT BLAS execution failed with status $BLAS_AOT_RC"
printf '%s\n' "$BLAS_AOT_SUMMARY" | grep -qE "$BLAS_SUMMARY_RE" || fail "AOT BLAS summary did not match expected Accelerate mismatch contract"
BLAS_AOT_MISMATCHES="$(extract_mismatches "$BLAS_AOT_SUMMARY")"
[ -n "$BLAS_AOT_MISMATCHES" ] || fail "AOT BLAS summary missing mismatch count"
[ "$BLAS_AOT_MISMATCHES" -gt 0 ] || fail "AOT BLAS mismatches must be > 0"
grep -Eq "$REJECT_DISPATCH_RE" "$BLAS_AOT_ERR" && fail "AOT BLAS execution printed forbidden Ozaki-II real dispatch line"
grep -Fq "$REJECT_FALLBACK" "$BLAS_AOT_ERR" && fail "AOT BLAS execution fell back to CPU BLAS"

run_case "JIT EXACT" EXACT_ENV "$BIN" -r "$TEST_ESK"
EXACT_JIT_RC="$RUN_RC"
EXACT_JIT_SUMMARY="$(cat "$RUN_OUT")"
EXACT_JIT_ERR="$RUN_ERR"
[ "$EXACT_JIT_RC" -eq 0 ] || fail "JIT exact execution failed with status $EXACT_JIT_RC"
printf '%s\n' "$EXACT_JIT_SUMMARY" | grep -qE "$EXACT_SUMMARY_RE" || fail "JIT exact summary was incorrect"
[ "$(grep -cE "$INIT_LINE_RE" "$EXACT_JIT_ERR")" -eq 1 ] || fail "JIT exact stderr must contain exactly one init line"
[ "$(grep -cE "$DISPATCH_LINE_RE" "$EXACT_JIT_ERR")" -eq 1 ] || fail "JIT exact stderr must contain exactly one dispatch line"
grep -E '^\[GPU\] Ozaki-II:' "$EXACT_JIT_ERR" | grep -Ev "$INIT_LINE_RE|$DISPATCH_LINE_RE" >/dev/null && fail "JIT exact stderr contains unexpected Ozaki-II lines"
grep -Fq "$REJECT_FALLBACK" "$EXACT_JIT_ERR" && fail "JIT exact execution fell back to CPU BLAS"

run_case "AOT EXACT" EXACT_ENV "$AOT_BIN"
EXACT_AOT_RC="$RUN_RC"
EXACT_AOT_SUMMARY="$(cat "$RUN_OUT")"
EXACT_AOT_ERR="$RUN_ERR"
[ "$EXACT_AOT_RC" -eq 0 ] || fail "AOT exact execution failed with status $EXACT_AOT_RC"
printf '%s\n' "$EXACT_AOT_SUMMARY" | grep -qE "$EXACT_SUMMARY_RE" || fail "AOT exact summary was incorrect"
[ "$(grep -cE "$INIT_LINE_RE" "$EXACT_AOT_ERR")" -eq 1 ] || fail "AOT exact stderr must contain exactly one init line"
[ "$(grep -cE "$DISPATCH_LINE_RE" "$EXACT_AOT_ERR")" -eq 1 ] || fail "AOT exact stderr must contain exactly one dispatch line"
grep -E '^\[GPU\] Ozaki-II:' "$EXACT_AOT_ERR" | grep -Ev "$INIT_LINE_RE|$DISPATCH_LINE_RE" >/dev/null && fail "AOT exact stderr contains unexpected Ozaki-II lines"
grep -Fq "$REJECT_FALLBACK" "$EXACT_AOT_ERR" && fail "AOT exact execution fell back to CPU BLAS"

pass "JIT and AOT both prove Accelerate mismatch>0 in BLAS mode (JIT:$BLAS_JIT_MISMATCHES AOT:$BLAS_AOT_MISMATCHES) and fixed-N16 Metal exact mismatch=0"
