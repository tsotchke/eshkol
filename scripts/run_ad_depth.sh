#!/usr/bin/env bash
# run_ad_depth.sh — depth-parametric AD oracle (adversarial pillar P6a).
#
# Sweeps every composable AD construct at nesting depth d = 1..8 under BOTH
# the JIT (-r) and AOT, checking each depth against a closed-form ANALYTIC
# ground truth (n-th derivative of the base shape) plus an in-language n-th
# central-difference stencil at shallow depth. It records the MAX-CORRECT-
# DEPTH of each construct and classifies every (cell,depth) as
#   PASS   correct
#   FAIL   silent WRONG value at that depth (the treasure)
#   LIMIT  clean crash / timeout boundary
#
# Products:
#   docs/reports/AD_DEPTH_REPORT.md       per-cell depth tables + MCD
#   scripts/icc_traces/ad_depth.jsonl      ICC events (kind ad_depth),
#                                          consumed by .icc/completion-oracles.yaml::ad-depth
#
# Gate (ad_depth_gate) is PASS when no cell REGRESSES below its tracked
# baseline max-depth (BASELINE in scripts/ad_depth_report.py). Known-open
# depth boundaries (ESH-0117..0121) are the baseline, so they stay green
# while tracked and flip to "improvement" automatically once fixed.
#
# Usage: scripts/run_ad_depth.sh [--no-aot] [--regen] [--quick] [--max-depth N]
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# macOS runner images do not consistently provide C.UTF-8. The Perl alarm
# wrapper below must not fail before Eshkol is even invoked.
export LC_ALL=C
export LANG=C
export LC_CTYPE=C

GEN_DIR="$REPO_ROOT/tests/ad_depth/generated"
RAW_LOG="$(mktemp "${TMPDIR:-/tmp}/ad_depth_raw.XXXXXX")"
trap 'rm -f "$RAW_LOG"' EXIT

: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-ad-depth-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_ad_depth.sh: $BUILD_DIR/eshkol-run not found — run" \
         "\`cmake --build $BUILD_DIR --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

DO_AOT=1; REGEN=0; QUICK=0; MAXD=8
while [ $# -gt 0 ]; do
    case "$1" in
        --no-aot) DO_AOT=0 ;;
        --regen)  REGEN=1 ;;
        --quick)  QUICK=1 ;;
        --max-depth) shift; MAXD="$1" ;;
        *) echo "run_ad_depth.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$REGEN" -eq 1 ] || ! ls "$GEN_DIR"/ad_depth_*.esk >/dev/null 2>&1; then
    python3 "$REPO_ROOT/scripts/gen_ad_depth.py" --max-depth "$MAXD" || exit 2
fi

JIT_TIMEOUT="${JIT_TIMEOUT:-240}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-360}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-90}"

# macOS has no timeout(1); emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

# record RESULT/FDCHK lines from a run into the raw log, tagged by mode+file.
record() {
    local mode="$1" file="$2" rc="$3" crashed="$4" out="$5"
    printf 'RUN\t%s\t%s\t%s\t%s\n' "$mode" "$file" "$rc" "$crashed" >> "$RAW_LOG"
    printf '%s\n' "$out" | grep -E '^(RESULT|FDCHK) ' | while IFS= read -r ln; do
        printf 'OUT\t%s\t%s\t%s\n' "$mode" "$file" "$ln" >> "$RAW_LOG"
    done
}

is_crash() {  # rc, out
    local rc="$1" out="$2"
    [ "$rc" -ge 128 ] && return 0
    printf '%s' "$out" | grep -q "fatal signal" && return 0
    return 1
}

files=$(ls "$GEN_DIR"/ad_depth_*.esk | sort)
if [ "$QUICK" -eq 1 ]; then
    files=$(printf '%s\n' $files | grep -E '_(deriv|gofd|compose|hessod_xc)_01\.esk$|hessod_xc_0[12]\.esk$')
fi

echo "== depth-parametric AD oracle (JIT$([ "$DO_AOT" -eq 1 ] && echo '+AOT')) =="
for f in $files; do
    base="$(basename "$f")"
    # ---- JIT (-r) ----
    out="$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$f" 2>&1)"; rc=$?
    cr=0; is_crash "$rc" "$out" && cr=1
    record "jit" "$base" "$rc" "$cr" "$out"
    jsum="$(printf '%s' "$out" | grep -c '^RESULT ')"
    printf '  %-34s jit  rc=%-3s crash=%s results=%s\n' "$base" "$rc" "$cr" "$jsum"

    [ "$DO_AOT" -eq 0 ] && continue
    # ---- AOT ----
    bin="$(mktemp "${TMPDIR:-/tmp}/ad_depth_bin.XXXXXX")"
    cout="$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$f" -o "$bin" 2>&1)"; crc=$?
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
        # compile failed => whole file is a LIMIT under AOT
        record "aot" "$base" "$crc" "1" "$cout"
        printf '  %-34s aot  COMPILE-FAIL rc=%s\n' "$base" "$crc"
        rm -f "$bin"; continue
    fi
    aout="$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1)"; arc=$?
    acr=0; is_crash "$arc" "$aout" && acr=1
    record "aot" "$base" "$arc" "$acr" "$aout"
    asum="$(printf '%s' "$aout" | grep -c '^RESULT ')"
    printf '  %-34s aot  rc=%-3s crash=%s results=%s\n' "$base" "$arc" "$acr" "$asum"
    rm -f "$bin"
done

modes="jit"; [ "$DO_AOT" -eq 1 ] && modes="jit aot"
python3 "$REPO_ROOT/scripts/ad_depth_report.py" "$RAW_LOG" $modes
gate_rc=$?
echo "docs/reports/AD_DEPTH_REPORT.md + scripts/icc_traces/ad_depth.jsonl written."
exit $gate_rc
