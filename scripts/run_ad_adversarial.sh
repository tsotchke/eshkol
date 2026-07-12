#!/usr/bin/env bash
# run_ad_adversarial.sh — generative adversarial AD-vs-finite-difference oracle.
#
# Runs every generated probe file in tests/ad_adversarial/generated/ under the
# JIT (-r) and (unless --no-aot) AOT. Each file self-checks every AD value
# against an in-language central finite difference and prints PASS:/FAIL:/
# XKNOWN: lines plus a Passed:/Failed:/Xknown: summary. Mirrors
# scripts/run_ad_oracle.sh; it emits
#   * pytest-style lines : "PASSED tests/ad_adversarial/generated/<f>::<mode>"
#   * ICC JSON-L events  : kind=ad_adversarial, into
#                          scripts/icc_traces/ad_adversarial.jsonl
# and a gate event ad_adversarial_gate consumed alongside the eshkol_smoke
# probe ad_adversarial_fd_oracle in .icc/completion-oracles.yaml.
#
# Verdicts per file+mode:
#   PASS    ran to completion, no FAIL:, no nonzero Failed: summary
#   XKNOWN  a tracked open bug printed XKNOWN: lines, OR an expected-crash cell
#           (ad_adv_xc_<ESH-task>_*.esk) crashed / failed to compile
#   FAIL    a finite-difference check failed on an untracked cell (a NEW,
#           actionable AD defect — the whole point of this harness)
#   CRASH   fatal signal / nonzero exit / codegen/IR failure on an untracked
#           cell
#   HANG    exceeded the per-run timeout
#
# The gate is green iff there are no FAIL/CRASH/HANG (XKNOWN is tolerated while
# its ESH task is open). This harness is meant to be RUN CONSTANTLY.
#
# Usage: scripts/run_ad_adversarial.sh [--quick] [--no-aot] [--regen]
#   --quick   one file per family (fast CI subset); JIT lane only
#   --no-aot  skip the AOT lane
#   --regen   re-run the (deterministic) generator before running
#
# When ESHKOL_QUANTUM_ENABLED=ON, the harness also runs the fixed
# tests/quantum/vqe_ad_adversarial.esk probe. It compares Eshkol's custom-VJP
# gradient of vqe-energy with central finite differences through vqe-energy;
# ordinary default builds remain Moonlab-free and skip this opt-in probe.
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
GEN_DIR="$REPO_ROOT/tests/ad_adversarial/generated"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/ad_adversarial.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"

# Fresh per-run JIT cache (mirrors run_icc_smoke.sh): a stale entry from a
# previous file must never mask a regression, and a fresh cache avoids the
# intermittent cross-file JIT crash noted below.
if [ -z "${ESHKOL_JIT_CACHE_DIR:-}" ]; then
    ADV_JIT_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-ad-adversarial-cache.XXXXXX")
    export ESHKOL_JIT_CACHE_DIR="$ADV_JIT_CACHE_DIR"
    trap 'rm -rf "$ADV_JIT_CACHE_DIR"' EXIT
else
    mkdir -p "$ESHKOL_JIT_CACHE_DIR"
fi

# Single-threaded JIT compile by default. The generative sweep exposed an
# INTERMITTENT SIGSEGV (fault address 0xfffffffffffffff8, i.e. -8 off a
# null/uninitialised pointer — not true stack exhaustion; the 512MB default
# stack is untouched) in the MULTI-THREADED JIT compile path when it compiles
# AD/tensor-heavy modules (the tensor and htensor families). It is a race, not
# a wrong-gradient bug: with ESHKOL_JIT_COMPILE_THREADS=1 the JIT lane is
# deterministically clean, and AOT is unaffected. Pinning to 1 keeps THIS gate
# reliably testing gradient CORRECTNESS; the concurrency crash is tracked
# separately (see tests/ad_adversarial/README.md). Override by exporting
# ESHKOL_JIT_COMPILE_THREADS to reproduce the race.
: "${ESHKOL_JIT_COMPILE_THREADS:=1}"
export ESHKOL_JIT_COMPILE_THREADS

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_ad_adversarial.sh: $BUILD_DIR/eshkol-run not found — run" \
         "\`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi
LIBFLAG="-L$REPO_ROOT/$BUILD_DIR"
case "$BUILD_DIR" in /*) LIBFLAG="-L$BUILD_DIR" ;; esac

DO_AOT=1
QUICK=0
REGEN=0
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        --quick)  QUICK=1; DO_AOT=0 ;;
        --regen)  REGEN=1 ;;
        *) echo "run_ad_adversarial.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$REGEN" -eq 1 ] || ! ls "$GEN_DIR"/ad_adv_*.esk >/dev/null 2>&1; then
    python3 "$REPO_ROOT/tests/ad_adversarial/gen_ad_adversarial.py" || exit 2
fi

JIT_TIMEOUT="${JIT_TIMEOUT:-240}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-300}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-120}"

# macOS has no timeout(1); emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() {
    local name="$1" value="$2" snippet="$3" en ev es
    en=$(json_escape "$name"); ev=$(json_escape "$value")
    es=$(json_escape "$snippet")
    printf '{"kind":"ad_adversarial","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$en" "$ev" "$es" >> "$TRACE_FILE"
}

# classify a completed run: args rc out xc -> PASS|FAIL|XKNOWN|CRASH|HANG
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
    emit_event "${f%.esk}_${mode}" "$v" "$f $mode -> $v"
    case "$v" in
        PASS)   passed+=1;  pyv="PASSED" ;;
        XKNOWN) xknown+=1;  pyv="XFAIL" ;;
        FAIL)   failed+=1;  pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
        CRASH)  crashed+=1; pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
        HANG)   hung+=1;    pyv="FAILED"; BAD="$BAD $f:$mode=$v" ;;
    esac
    printf '  %-6s %s::%s\n' "$v" "$f" "$mode"
    echo "$pyv $f::$mode"
}

echo "Generative adversarial AD/FD oracle -> $TRACE_FILE"
echo

for path in "$GEN_DIR"/ad_adv_*.esk; do
    f=$(basename "$path")
    base="${f%.esk}"
    xc=0
    case "$f" in ad_adv_xc_*) xc=1 ;; esac
    if [ "$QUICK" -eq 1 ]; then
        case "$f" in
            *_01.esk) : ;;
            ad_adv_xc_*) : ;;
            *) continue ;;
        esac
    fi

    rout=$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$path" "$LIBFLAG" 2>&1); rrc=$?
    rv=$(verdict "$rrc" "$rout" "$xc")
    count_verdict "$rv" "tests/ad_adversarial/generated/$f" "r"
    if [ "$rv" = "FAIL" ]; then
        printf '%s\n' "$rout" | grep -E '^FAIL:' | head -6 | sed 's/^/         /'
    fi

    if [ "$DO_AOT" -eq 1 ]; then
        bin="${TMPDIR:-/tmp}/ad_adv_${base}.bin"; rm -f "$bin"
        cout=$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$path" "$LIBFLAG" -o "$bin" 2>&1); crc=$?
        if [ "$crc" -ne 0 ] || [ ! -x "$bin" ] || printf '%s' "$cout" | grep -qE \
            "Failed to generate LLVM IR|LLVM module verification failed"; then
            if [ "$xc" -eq 1 ]; then av=XKNOWN; elif [ "$crc" -eq 142 ]; then av=HANG; else av=CRASH; fi
        else
            aout=$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1); arc=$?
            av=$(verdict "$arc" "$aout" "$xc")
            if [ "$av" = "FAIL" ]; then
                printf '%s\n' "$aout" | grep -E '^FAIL:' | head -6 | sed 's/^/         /'
            fi
        fi
        rm -f "$bin"
        count_verdict "$av" "tests/ad_adversarial/generated/$f" "aot"
    fi
done

# The Moonlab VQE bridge is opt-in. Keep the default adversarial sweep exactly
# dependency-free, but make a quantum-enabled build expose the same custom-VJP
# path to the oracle's JIT and AOT verdict machinery.
if [ "${ESHKOL_QUANTUM_ENABLED:-OFF}" = "ON" ]; then
    quantum_path="$REPO_ROOT/tests/quantum/vqe_ad_adversarial.esk"
    quantum_label="tests/quantum/$(basename "$quantum_path")"
    quantum_base="${quantum_label##*/}"
    quantum_base="${quantum_base%.esk}"
    echo "Running opt-in quantum VQE AD/FD probe"

    qout=$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$quantum_path" "$LIBFLAG" 2>&1); qrc=$?
    qv=$(verdict "$qrc" "$qout" 0)
    count_verdict "$qv" "$quantum_label" "r"
    if [ "$qv" = "FAIL" ]; then
        printf '%s\n' "$qout" | grep -E '^FAIL:' | head -6 | sed 's/^/         /'
    fi

    if [ "$DO_AOT" -eq 1 ]; then
        qbin="${TMPDIR:-/tmp}/ad_adv_${quantum_base}.bin"; rm -f "$qbin"
        qcout=$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$quantum_path" "$LIBFLAG" -o "$qbin" 2>&1); qcrc=$?
        if [ "$qcrc" -ne 0 ] || [ ! -x "$qbin" ] || printf '%s' "$qcout" | grep -qE \
            "Failed to generate LLVM IR|LLVM module verification failed"; then
            if [ "$qcrc" -eq 142 ]; then qav=HANG; else qav=CRASH; fi
        else
            qaout=$(run_guarded "$AOT_RUN_TIMEOUT" "$qbin" 2>&1); qarc=$?
            qav=$(verdict "$qarc" "$qaout" 0)
            if [ "$qav" = "FAIL" ]; then
                printf '%s\n' "$qaout" | grep -E '^FAIL:' | head -6 | sed 's/^/         /'
            fi
        fi
        rm -f "$qbin"
        count_verdict "$qav" "$quantum_label" "aot"
    fi
else
    echo "Skipping quantum VQE AD/FD probe (ESHKOL_QUANTUM_ENABLED is not ON)"
fi

echo
echo "ad_adversarial summary: total=$total passed=$passed xknown=$xknown failed=$failed crashed=$crashed hung=$hung"
[ -n "$BAD" ] && echo "ad_adversarial offenders:$BAD"

gate=PASS
if [ "$failed" -ne 0 ] || [ "$crashed" -ne 0 ] || [ "$hung" -ne 0 ] || [ "$total" -eq 0 ]; then
    gate=FAIL
fi
emit_event "ad_adversarial_gate" "$gate" \
    "total=$total passed=$passed xknown=$xknown failed=$failed crashed=$crashed hung=$hung quick=$QUICK aot=$DO_AOT"

echo "ad_adversarial gate: $gate"
if [ "$gate" = "PASS" ]; then
    echo "PASSED tests/ad_adversarial::gate"
    exit 0
else
    echo "FAILED tests/ad_adversarial::gate"
    exit 1
fi
