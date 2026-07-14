#!/usr/bin/env bash
# run_ad_oracle.sh — AD composition oracle (adversarial testing campaign P3).
#
# Runs every generated probe file in tests/ad_oracle/generated/ under BOTH
# the JIT (-r) and AOT. Each file self-checks every AD value against an
# in-language central finite difference and prints PASS:/FAIL:/XKNOWN:
# lines plus a Passed:/Failed:/Xknown: summary. Mirrors
# scripts/run_sicp_smoke.sh: it emits
#   * pytest-style lines : "PASSED tests/ad_oracle/generated/<f>::<mode>"
#   * ICC JSON-L events  : kind=ad_oracle, consumed by
#                          .icc/completion-oracles.yaml::ad-oracle
#
# Verdicts per file+mode:
#   PASS    ran to completion, no FAIL:, no nonzero Failed: summary
#   XKNOWN  known-open compiler bug: either the probe printed XKNOWN: lines
#           (in-language, tolerance check failed on a tracked cell) or the
#           file is an expected-crash cell (ad_oracle_xc_<ESH-task>_*.esk)
#           that crashed / failed to compile
#   FAIL    a finite-difference check failed on an untracked cell
#   CRASH   fatal signal, nonzero exit, or codegen/IR failure
#   HANG    exceeded the per-run timeout
#
# The gate is green when there are no FAIL/CRASH/HANG verdicts (XKNOWN is
# tolerated — each XKNOWN references an ESH task in .swarm/tasks/).
#
# Usage: scripts/run_ad_oracle.sh [--quick] [--no-aot] [--regen]
#   --quick   run only the *_01.esk file of each section plus the first
#             expected-crash file of each tracked task (CI smoke subset)
#   --no-aot  skip the AOT lane
#   --regen   re-run the (deterministic) generator before running
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
GEN_DIR="$REPO_ROOT/tests/ad_oracle/generated"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/ad_oracle.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"
: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-ad-oracle-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_ad_oracle.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

DO_AOT=1
QUICK=0
REGEN=0
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        --quick) QUICK=1 ;;
        --regen) REGEN=1 ;;
        *)
            echo "run_ad_oracle.sh: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

if [ "$REGEN" -eq 1 ] || ! ls "$GEN_DIR"/ad_oracle_*.esk >/dev/null 2>&1; then
    python3 "$REPO_ROOT/tests/ad_oracle/gen_ad_oracle.py" || exit 2
fi

JIT_TIMEOUT="${JIT_TIMEOUT:-180}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-300}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-60}"

# macOS has no `timeout(1)`; emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $seconds = shift; alarm $seconds; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() {
    local name="$1" value="$2" snippet="$3" esc_name esc_value esc_snippet
    esc_name=$(json_escape "$name")
    esc_value=$(json_escape "$value")
    esc_snippet=$(json_escape "$snippet")
    printf '{"kind":"ad_oracle","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$esc_name" "$esc_value" "$esc_snippet" >> "${TRACE_FILE:?}"
}

# classify a completed (non-timeout) run of one probe file
# args: rc out xc(0/1) -> echoes PASS|FAIL|XKNOWN|CRASH|HANG
verdict() {
    local rc="$1" out="$2" xc="$3"
    if [ "$rc" -eq 142 ]; then
        [ "$xc" -eq 1 ] && { echo XKNOWN; return; }
        echo HANG; return
    fi
    if [ "$rc" -ge 128 ] || printf '%s' "$out" | grep -q "fatal signal"; then
        [ "$xc" -eq 1 ] && { echo XKNOWN; return; }
        echo CRASH; return
    fi
    if [ "$rc" -ne 0 ] || printf '%s' "$out" | grep -qE \
        "Failed to generate LLVM IR|JIT batch execution failed|LLVM module verification failed"; then
        [ "$xc" -eq 1 ] && { echo XKNOWN; return; }
        echo CRASH; return
    fi
    if printf '%s' "$out" | grep -qE '^FAIL:|Failed:[[:space:]]+[1-9]'; then
        [ "$xc" -eq 1 ] && { echo XKNOWN; return; }
        echo FAIL; return
    fi
    if ! printf '%s' "$out" | grep -q '^Passed:'; then
        # never reached the summary — treat as crash (silent early death)
        [ "$xc" -eq 1 ] && { echo XKNOWN; return; }
        echo CRASH; return
    fi
    if printf '%s' "$out" | grep -q '^XKNOWN:'; then
        echo XKNOWN; return
    fi
    echo PASS
}

declare -i total=0 passed=0 failed=0 xknown=0 crashed=0 hung=0
BAD=""

count_verdict() { # verdict file mode
    local v="$1" f="$2" mode="$3" pyv
    total+=1
    # filenames already carry the ad_oracle_ prefix
    emit_event "${f%.esk}_${mode}" "$v" "$f $mode -> $v"
    case "$v" in
        PASS)   passed+=1;  pyv="PASSED" ;;
        XKNOWN) xknown+=1;  pyv="XFAIL" ;;
        FAIL)   failed+=1;  pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
        CRASH)  crashed+=1; pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
        HANG)   hung+=1;    pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
    esac
    printf '  %-6s tests/ad_oracle/generated/%s::%s\n' "$v" "$f" "$mode"
    echo "$pyv tests/ad_oracle/generated/$f::$mode"
}

echo "AD composition oracle -> $TRACE_FILE"
echo

for path in "$GEN_DIR"/ad_oracle_*.esk; do
    f=$(basename "$path")
    base="${f%.esk}"
    xc=0
    case "$f" in ad_oracle_xc_*) xc=1 ;; esac
    if [ "$QUICK" -eq 1 ]; then
        case "$f" in
            *_01.esk) : ;;
            *) continue ;;
        esac
    fi

    # ----- JIT (-r) -----
    rout=$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$path" 2>&1); rrc=$?
    rv=$(verdict "$rrc" "$rout" "$xc")
    count_verdict "$rv" "$f" "r"
    if [ "$rv" = "FAIL" ]; then
        printf '%s\n' "$rout" | grep -E '^FAIL:' | head -5 | sed 's/^/         /'
    fi

    # ----- AOT -----
    if [ "$DO_AOT" -eq 1 ]; then
        bin="${TMPDIR:-/tmp}/ad_oracle_${base}.bin"; rm -f "$bin"
        cout=$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$path" -o "$bin" 2>&1); crc=$?
        if [ "$crc" -ne 0 ] || [ ! -x "$bin" ] || printf '%s' "$cout" | grep -qE \
            "Failed to generate LLVM IR|LLVM module verification failed"; then
            if [ "$xc" -eq 1 ]; then av=XKNOWN; elif [ "$crc" -eq 142 ]; then av=HANG; else av=CRASH; fi
        else
            aout=$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1); arc=$?
            av=$(verdict "$arc" "$aout" "$xc")
            if [ "$av" = "FAIL" ]; then
                printf '%s\n' "$aout" | grep -E '^FAIL:' | head -5 | sed 's/^/         /'
            fi
        fi
        rm -f "$bin"
        count_verdict "$av" "$f" "aot"
    fi
done

echo
echo "ad_oracle summary: total=$total passed=$passed xknown=$xknown failed=$failed crashed=$crashed hung=$hung"
[ -n "$BAD" ] && echo "ad_oracle offenders:$BAD"

gate=PASS
if [ "$failed" -ne 0 ] || [ "$crashed" -ne 0 ] || [ "$hung" -ne 0 ] || [ "$total" -eq 0 ]; then
    gate=FAIL
fi
emit_event "ad_oracle_gate" "$gate" \
    "total=$total passed=$passed xknown=$xknown failed=$failed crashed=$crashed hung=$hung quick=$QUICK aot=$DO_AOT"

echo "ad_oracle gate: $gate"
if [ "$gate" = "PASS" ]; then
    echo "PASSED tests/ad_oracle::gate"
    exit 0
else
    echo "FAILED tests/ad_oracle::gate"
    exit 1
fi
