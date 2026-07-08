#!/usr/bin/env bash
# run_ad_validated_bounds_gate.sh — P8 Taylor-model validated-AD gate (ESH-0193).
#
# Runs tests/ad/taylor_models_test.esk under BOTH the JIT (-r) and AOT and
# checks that all in-test SOUNDNESS + TIGHTENING + HARD-CASE assertions pass
# (final "PASS: taylor_models_test" line and a "N passed, 0 failed" summary).
# The test proves that (taylor-model f x0 r k):
#   * SOUNDNESS  — (tm-range tm) contains a dense grid of f over the whole
#                  domain box, and (tm-eval tm x) contains f(x), for exp/sin/
#                  1/(1-x)/x^5 (outward-rounded interval remainder enclosure);
#   * TIGHTENING — the enclosure width is non-increasing as order k grows and
#                  shrinks as the domain radius shrinks;
#   * HARD CASE  — for exp(x)-1-x the enclosure soundly contains the stable
#                  reference even where naive floating evaluation cancels.
#
# On success it writes an ICC runtime_event to
#   scripts/icc_traces/ad_validated_bounds.jsonl
# consumed by .icc/completion-oracles.yaml
#   (event_kinds: [ad_validated_bounds],
#    event_names: ["ad_taylor_p8_taylor_models"], event_values: ["PASS"]).
#
# Usage: scripts/run_ad_validated_bounds_gate.sh [--no-aot]
set -u

# The timeout/json helpers are byte-oriented Perl snippets. Force a portable
# locale instead of relying on C.UTF-8 being installed on every host.
export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/ad_validated_bounds.jsonl"
TEST_FILE="$REPO_ROOT/tests/ad/taylor_models_test.esk"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"        # clear stale passes each run

: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-ad-vbounds-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_ad_validated_bounds_gate.sh: $BUILD_DIR/eshkol-run not found — run" \
         "\`cmake --build $BUILD_DIR --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

DO_AOT=1
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        *) echo "run_ad_validated_bounds_gate.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

JIT_TIMEOUT="${JIT_TIMEOUT:-180}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-300}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-60}"

# macOS has no `timeout(1)`; emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

# returns 0 (ok) if output has the final PASS line, a "N passed, 0 failed"
# summary, and no FAIL:/crash markers.
check_output() {
    local out="$1"
    printf '%s' "$out" | grep -q '^PASS: taylor_models_test' || return 1
    printf '%s' "$out" | grep -qE 'taylor-models tests: [0-9]+ passed, 0 failed' || return 1
    printf '%s' "$out" | grep -qE '^FAIL:|fatal signal|LLVM module verification failed' && return 1
    return 0
}

overall=PASS
detail=""

echo "== P8 Taylor-model validated-AD gate =="

# ---- JIT (-r) ----
jout="$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$TEST_FILE" -L"$REPO_ROOT/$BUILD_DIR" 2>&1)"
if check_output "$jout"; then
    jpass="$(printf '%s' "$jout" | grep -oE 'tests: [0-9]+ passed' | grep -oE '[0-9]+')"
    echo "  JIT  PASS ($jpass checks)"
    detail="jit=PASS($jpass)"
else
    echo "  JIT  FAIL"
    printf '%s\n' "$jout" | grep -E '^FAIL:|fatal signal|failed$' | head
    overall=FAIL
    detail="jit=FAIL"
fi

# ---- AOT ----
if [ "$DO_AOT" -eq 1 ]; then
    bin="$(mktemp "${TMPDIR:-/tmp}/tm_gate_bin.XXXXXX")"
    cout="$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$TEST_FILE" -o "$bin" -L"$REPO_ROOT/$BUILD_DIR" 2>&1)"; crc=$?
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
        echo "  AOT  COMPILE-FAIL rc=$crc"
        overall=FAIL; detail="$detail aot=COMPILE-FAIL"
    else
        aout="$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1)"
        if check_output "$aout"; then
            apass="$(printf '%s' "$aout" | grep -oE 'tests: [0-9]+ passed' | grep -oE '[0-9]+')"
            echo "  AOT  PASS ($apass checks)"
            detail="$detail aot=PASS($apass)"
        else
            echo "  AOT  FAIL"
            printf '%s\n' "$aout" | grep -E '^FAIL:|fatal signal|failed$' | head
            overall=FAIL; detail="$detail aot=FAIL"
        fi
    fi
    rm -f "$bin"
fi

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
snip="$(json_escape "P8 Taylor-model soundness+tightening+hard-case gate; $detail; $ts")"
printf '{"kind":"ad_validated_bounds","name":"ad_taylor_p8_taylor_models","value":"%s","snippet":"%s","confidence":0.95}\n' \
    "$overall" "$snip" >> "$TRACE_FILE"

echo "scripts/icc_traces/ad_validated_bounds.jsonl written (value=$overall)."
[ "$overall" = "PASS" ] && exit 0 || exit 1
