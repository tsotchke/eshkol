#!/usr/bin/env bash
# run_metamorphic.sh - deterministic metamorphic/property oracle (P7c).
#
# Runs generated Eshkol programs under JIT twice and requires byte-identical
# combined output for determinism. Optionally runs the same corpus through AOT.
# Probe files self-check algebraic laws and print PASS:/FAIL:/XKNOWN: lines.
#
# Usage: scripts/run_metamorphic.sh [--quick] [--regen] [--no-aot] [--trials N]
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/metamorphic.jsonl"
: "${TRACE_FILE:?TRACE_FILE not set}"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_metamorphic.sh: $BUILD_DIR/eshkol-run not found - run \`cmake --build build --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

GEN_DIR="$REPO_ROOT/tests/metamorphic/generated"
GENERATOR="$REPO_ROOT/tests/metamorphic/gen_metamorphic.py"
DO_AOT=1
QUICK=0
REGEN=0
TRIALS=32
CUSTOM_TRIALS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --no-aot) DO_AOT=0; shift ;;
        --quick) QUICK=1; shift ;;
        --regen) REGEN=1; shift ;;
        --trials)
            if [ $# -lt 2 ]; then
                echo "run_metamorphic.sh: --trials requires a value" >&2
                exit 2
            fi
            TRIALS="$2"; CUSTOM_TRIALS=1; shift 2 ;;
        *)
            echo "run_metamorphic.sh: unknown argument: $1" >&2
            exit 2 ;;
    esac
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-metamorphic.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [ "$QUICK" -eq 1 ]; then
    [ "$CUSTOM_TRIALS" -eq 0 ] && TRIALS=8
    RUN_DIR="$WORK/generated"
    python3 "$GENERATOR" --outdir "$RUN_DIR" --trials "$TRIALS" || exit 2
elif [ "$REGEN" -eq 1 ] || ! ls "$GEN_DIR"/metamorphic_*.esk >/dev/null 2>&1; then
    python3 "$GENERATOR" --outdir "$GEN_DIR" --trials "$TRIALS" || exit 2
    RUN_DIR="$GEN_DIR"
else
    RUN_DIR="$GEN_DIR"
fi

JIT_TIMEOUT="${METAMORPHIC_JIT_TIMEOUT:-180}"
AOT_COMPILE_TIMEOUT="${METAMORPHIC_AOT_COMPILE_TIMEOUT:-300}"
AOT_RUN_TIMEOUT="${METAMORPHIC_AOT_RUN_TIMEOUT:-90}"

# macOS has no timeout(1); emulate with perl alarm (exit 142 on expiry).
run_guarded() {
    LC_ALL=C LANG=C LC_CTYPE=C perl -e 'my $seconds = shift; $SIG{ALRM}=sub{ exit 142 }; alarm $seconds; exec @ARGV; exit 127' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | LC_ALL=C LANG=C LC_CTYPE=C perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() {
    local name="$1" value="$2" snippet="$3"
    printf '{"kind":"metamorphic","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$(json_escape "$name")" "$(json_escape "$value")" "$(json_escape "$snippet")" >> "$TRACE_FILE"
}

verdict() {
    local rc="$1" out="$2"
    if [ "$rc" -eq 142 ]; then
        echo HANG; return
    fi
    if [ "$rc" -ge 128 ] || printf '%s' "$out" | grep -q "fatal signal"; then
        echo CRASH; return
    fi
    if [ "$rc" -ne 0 ] || printf '%s' "$out" | grep -qE \
        "Failed to generate LLVM IR|JIT batch execution failed|LLVM module verification failed"; then
        echo CRASH; return
    fi
    if printf '%s' "$out" | grep -qE '^FAIL:'; then
        echo FAIL; return
    fi
    if ! printf '%s' "$out" | grep -q '^METAMORPHIC-SUMMARY '; then
        echo CRASH; return
    fi
    if printf '%s' "$out" | grep -q '^XKNOWN:'; then
        echo XKNOWN; return
    fi
    echo PASS
}

declare -i total=0 passed=0 xknown=0 failed=0 crashed=0 hung=0 deterministic_failed=0
BAD=""

count_verdict() {
    local v="$1" file="$2" mode="$3" pyv
    total+=1
    emit_event "${file%.esk}_${mode}" "$v" "$file $mode -> $v"
    case "$v" in
        PASS) passed+=1; pyv="PASSED" ;;
        XKNOWN) xknown+=1; pyv="XFAIL" ;;
        FAIL) failed+=1; pyv="FAILED"; BAD="$BAD $file:$mode=$v" ;;
        CRASH) crashed+=1; pyv="FAILED"; BAD="$BAD $file:$mode=$v" ;;
        HANG) hung+=1; pyv="FAILED"; BAD="$BAD $file:$mode=$v" ;;
    esac
    printf '  %-6s tests/metamorphic/generated/%s::%s\n' "$v" "$file" "$mode"
    echo "$pyv tests/metamorphic/generated/$file::$mode"
}

export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

echo "Metamorphic oracle -> $TRACE_FILE"
echo "Corpus: $RUN_DIR"
echo "Trials: $TRIALS$([ "$QUICK" -eq 1 ] && printf ' (quick)')"
echo "AOT: $DO_AOT"
echo

shopt -s nullglob
files=("$RUN_DIR"/metamorphic_*.esk)
if [ "${#files[@]}" -eq 0 ]; then
    echo "run_metamorphic.sh: no generated metamorphic_*.esk files in $RUN_DIR" >&2
    emit_event "metamorphic_gate" "FAIL" "no generated files"
    exit 2
fi

for path in "${files[@]}"; do
    f=$(basename "$path")

    # JIT lane, run twice with cache disabled and compare combined output bytes.
    out1=$(ESHKOL_JIT_CACHE=0 run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$path" 2>&1); rc1=$?
    out2=$(ESHKOL_JIT_CACHE=0 run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$path" 2>&1); rc2=$?
    v=$(verdict "$rc1" "$out1")
    count_verdict "$v" "$f" "jit"

    if [ "$rc1" -ne "$rc2" ] || [ "$out1" != "$out2" ]; then
        deterministic_failed+=1
        BAD="$BAD $f:jit-determinism=FAIL"
        printf '  FAIL   tests/metamorphic/generated/%s::jit-determinism\n' "$f"
        echo "FAILED tests/metamorphic/generated/$f::jit-determinism"
        emit_event "${f%.esk}_jit_determinism" "FAIL" "run twice output/rc mismatch: rc1=$rc1 rc2=$rc2"
    else
        printf '  PASS   tests/metamorphic/generated/%s::jit-determinism\n' "$f"
        echo "PASSED tests/metamorphic/generated/$f::jit-determinism"
        emit_event "${f%.esk}_jit_determinism" "PASS" "byte-identical repeated jit output rc=$rc1"
    fi

    if [ "$v" = "FAIL" ]; then
        printf '%s\n' "$out1" | grep -E '^FAIL:' | head -8 | sed 's/^/         /'
    elif [ "$v" = "XKNOWN" ]; then
        printf '%s\n' "$out1" | grep -E '^XKNOWN:' | head -8 | sed 's/^/         /'
    elif [ "$v" = "CRASH" ] || [ "$v" = "HANG" ]; then
        printf '%s\n' "$out1" | head -10 | sed 's/^/         /'
    fi

    # AOT lane.
    if [ "$DO_AOT" -eq 1 ]; then
        base="${f%.esk}"
        bin="$WORK/${base}.bin"
        cout=$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$path" -o "$bin" 2>&1); crc=$?
        if [ "$crc" -eq 142 ]; then
            av=HANG
        elif [ "$crc" -ne 0 ] || [ ! -x "$bin" ] || printf '%s' "$cout" | grep -qE \
            "Failed to generate LLVM IR|LLVM module verification failed"; then
            av=CRASH
            aout="$cout"
        else
            aout=$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1); arc=$?
            av=$(verdict "$arc" "$aout")
        fi
        rm -f "$bin"
        count_verdict "$av" "$f" "aot"
        if [ "$av" = "FAIL" ]; then
            printf '%s\n' "$aout" | grep -E '^FAIL:' | head -8 | sed 's/^/         /'
        elif [ "$av" = "XKNOWN" ]; then
            printf '%s\n' "$aout" | grep -E '^XKNOWN:' | head -8 | sed 's/^/         /'
        elif [ "$av" = "CRASH" ] || [ "$av" = "HANG" ]; then
            printf '%s\n' "$aout" | head -10 | sed 's/^/         /'
        fi
    fi
done

echo
echo "metamorphic summary: total=$total passed=$passed xknown=$xknown failed=$failed crashed=$crashed hung=$hung determinism_failed=$deterministic_failed"
[ -n "$BAD" ] && echo "metamorphic offenders:$BAD"

gate=PASS
if [ "$failed" -ne 0 ] || [ "$crashed" -ne 0 ] || [ "$hung" -ne 0 ] || \
   [ "$deterministic_failed" -ne 0 ] || [ "$total" -eq 0 ]; then
    gate=FAIL
fi
emit_event "metamorphic_gate" "$gate" \
    "total=$total passed=$passed xknown=$xknown failed=$failed crashed=$crashed hung=$hung determinism_failed=$deterministic_failed trials=$TRIALS quick=$QUICK aot=$DO_AOT"

echo "metamorphic gate: $gate"
if [ "$gate" = "PASS" ]; then
    echo "PASSED tests/metamorphic::gate"
    exit 0
else
    echo "FAILED tests/metamorphic::gate"
    exit 1
fi
