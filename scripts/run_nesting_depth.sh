#!/usr/bin/env bash
# run_nesting_depth.sh — depth-parametric syntax/data NESTING sweep (pillar P6c).
#
# Runs every generated probe in tests/nesting_depth/generated/ under THREE
# execution axes — the JIT (-r), AOT at -O0, and AOT at -O2 — sweeping nesting
# depth per construct to find and gate the maximum depth at which each construct
# still yields the CORRECT value on every axis. Mirrors the other depth pillars:
#   * pytest-style lines : "PASSED tests/nesting_depth/generated/<f>::<cell>"
#   * ICC JSON-L events  : kind=nesting_depth, consumed by
#                          .icc/completion-oracles.yaml::nesting-depth
#
# Per-axis raw outcome (from exit code + emitted output):
#   VAL <n>   clean exit, printed a numeric result
#   LIMIT     nonzero/fatal exit WITH a diagnostic (fatal-signal handler,
#             compile error, recursion-depth guard) — a documented boundary
#   HANG      exceeded the per-run timeout — a resource boundary (grouped
#             with LIMIT for classification, but surfaced in the detail)
#   SILENT    clean exit with NO numeric output, or a fatal signal with NO
#             diagnostic — a bug
#
# Cell classification (combines the three axes against the closed-form VALUE).
# BOUNDARY := LIMIT or HANG (a construct hit a documented depth boundary).
#   PASS             every axis == the closed form
#   WRONG            all axes agree with each other but the value is wrong
#                    (a silent miscompile only the closed form catches)
#   AXIS-DIVERGENCE  axes disagree, or one axis returns a value while another
#                    hits a boundary / crashes (-r vs AOT-O0 vs AOT-O2)
#   LIMIT            a consistent boundary (LIMIT/HANG) on every axis, no value
#   SILENT-CRASH     an undiagnosed failure (clean exit no output, or fatal
#                    signal with no diagnostic) on some axis
#
# Gate PASS when no cell is WRONG / AXIS-DIVERGENCE / SILENT-CRASH. PASS and
# clean LIMIT cells are fine (a clean depth limit is a documented capability
# boundary, not a bug).
#
# Usage: scripts/run_nesting_depth.sh [--quick] [--regen] [--depths d,d,..]
#   --quick     drop the two deepest ladder cells (faster CI pass)
#   --regen     re-run the (deterministic) generator before running
#   --depths L  regenerate with an explicit comma-separated depth ladder
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Keep the Perl timeout wrapper independent of host locale availability.
export LC_ALL=C
export LANG=C
export LC_CTYPE=C

GEN_DIR="$REPO_ROOT/tests/nesting_depth/generated"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/nesting_depth.jsonl"
ART_DIR="$REPO_ROOT/artifacts/nesting-depth"
mkdir -p "$TRACE_DIR" "$ART_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"

# ── disk budget (mandatory) ──────────────────────────────────────────────
# One reused temp binary, deleted after every run; failure logs only, capped.
ART_CAP_MB=1024
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nesting-depth.XXXXXX")"
BIN="$WORK/probe.bin"
: "${BIN:?BIN not set}"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT INT TERM
du_mb() { du -sm "$1" 2>/dev/null | awk '{print $1}'; }
check_disk() {
    local mb; mb=$(du_mb "$ART_DIR"); mb=${mb:-0}
    if [ "$mb" -gt "$ART_CAP_MB" ]; then
        echo "run_nesting_depth.sh: artifacts/ exceeded ${ART_CAP_MB}MB (${mb}MB) — aborting." >&2
        exit 3
    fi
}

: "${ESHKOL_JIT_CACHE_DIR:=$WORK/jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_nesting_depth.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib -j8\` first." >&2
    exit 2
fi

QUICK=0
REGEN=0
DEPTHS=""
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --regen) REGEN=1 ;;
        --depths) echo "run_nesting_depth.sh: --depths needs =VALUE form (e.g. --depths=1,2,4)" >&2; exit 2 ;;
        --depths=*) DEPTHS="${arg#--depths=}"; REGEN=1 ;;
        *) echo "run_nesting_depth.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$REGEN" -eq 1 ] || ! ls "$GEN_DIR"/nest_*.esk >/dev/null 2>&1; then
    if [ -n "$DEPTHS" ]; then
        python3 "$REPO_ROOT/scripts/gen_nesting_depth.py" --depths "$DEPTHS" || exit 2
    else
        python3 "$REPO_ROOT/scripts/gen_nesting_depth.py" || exit 2
    fi
fi

JIT_TIMEOUT="${JIT_TIMEOUT:-90}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-150}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-90}"

# macOS has no timeout(1); emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $s = shift; alarm $s; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() {
    local name="$1" value="$2" snippet="$3" en ev es
    en=$(json_escape "$name"); ev=$(json_escape "$value"); es=$(json_escape "$snippet")
    printf '{"kind":"nesting_depth","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$en" "$ev" "$es" >> "$TRACE_FILE"
}

DIAG_RE="fatal signal|terminating|recursion depth|stack overflow|[Ee]rror:|Unhandled exception|exceeded|SIGSEGV|SIGILL|SIGBUS|SIGABRT|Type error|out of memory|assert"

# axis_outcome rc out  ->  "VAL <n>" | "LIMIT" | "SILENT" | "HANG"
axis_outcome() {
    local rc="$1" out="$2" numline
    if [ "$rc" -eq 142 ]; then echo HANG; return; fi
    numline=$(printf '%s' "$out" | grep -E '^-?[0-9]+$' | tail -1)
    if [ "$rc" -eq 0 ]; then
        if [ -n "$numline" ]; then echo "VAL $numline"; return; fi
        echo SILENT; return   # clean exit, produced no numeric value
    fi
    if printf '%s' "$out" | grep -qiE "$DIAG_RE"; then echo LIMIT; return; fi
    echo SILENT
}

# run one axis: mode file -> sets globals RC/OUT ; echoes axis_outcome
run_axis() {
    local mode="$1" path="$2" out rc
    case "$mode" in
        r)
            out=$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$path" 2>&1); rc=$? ;;
        o0|o2)
            local lvl=0; [ "$mode" = "o2" ] && lvl=2
            rm -f "$BIN"
            out=$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$path" -O "$lvl" -o "$BIN" 2>&1); rc=$?
            if [ "$rc" -eq 0 ] && [ -x "$BIN" ]; then
                out=$(run_guarded "$AOT_RUN_TIMEOUT" "$BIN" 2>&1); rc=$?
            fi
            rm -f "$BIN" ;;
    esac
    LAST_OUT="$out"
    axis_outcome "$rc" "$out"
}

read_dir() { grep -m1 "^; $2:" "$1" | sed "s/^; $2:[[:space:]]*//"; }

declare -i total=0 npass=0 nlimit=0 nwrong=0 ndiv=0 ncrash=0
BAD=""
SAFE_TMP="$WORK/safe.tsv"; : > "$SAFE_TMP"

echo "nesting-depth sweep (P6c) -> $TRACE_FILE"
echo "  axes: -r (JIT) | AOT -O0 | AOT -O2    build: $ESHKOL_RUN"
echo

for path in "$GEN_DIR"/nest_*.esk; do
    f=$(basename "$path"); base="${f%.esk}"
    construct=$(read_dir "$path" CONSTRUCT)
    depth=$(read_dir "$path" DEPTH)
    expect=$(read_dir "$path" VALUE)

    if [ "$QUICK" -eq 1 ]; then
        case "$depth" in 128|256) continue ;; esac
    fi

    check_disk

    o_r=$(run_axis r  "$path")
    o_0=$(run_axis o0 "$path")
    o_2=$(run_axis o2 "$path")

    # collect per-axis values
    v_r="${o_r#VAL }"; [ "${o_r%% *}" != "VAL" ] && v_r=""
    v_0="${o_0#VAL }"; [ "${o_0%% *}" != "VAL" ] && v_0=""
    v_2="${o_2#VAL }"; [ "${o_2%% *}" != "VAL" ] && v_2=""

    nval=0
    for v in "$v_r" "$v_0" "$v_2"; do [ -n "$v" ] && nval=$((nval+1)); done
    # distinct value count
    distinct=$(printf '%s\n%s\n%s\n' "$v_r" "$v_0" "$v_2" | grep -E '^-?[0-9]+$' | sort -u | wc -l | tr -d ' ')
    all_match_expect=1
    for v in "$v_r" "$v_0" "$v_2"; do
        [ -n "$v" ] && [ "$v" != "$expect" ] && all_match_expect=0
    done
    # BOUNDARY = LIMIT or HANG (documented depth/resource boundary, diagnosed
    # or timed out). SILENT = an undiagnosed failure (bug).
    all_boundary=1 any_silent=0
    for o in "$o_r" "$o_0" "$o_2"; do
        case "$o" in
            LIMIT|HANG) ;;
            SILENT) all_boundary=0; any_silent=1 ;;
            *) all_boundary=0 ;;
        esac
    done

    if [ "$nval" -eq 3 ] && [ "$distinct" -eq 1 ] && [ "$all_match_expect" -eq 1 ]; then
        status=PASS
    elif [ "$nval" -eq 3 ] && [ "$distinct" -eq 1 ]; then
        status=WRONG
    elif [ "$nval" -ge 1 ]; then
        status=AXIS-DIVERGENCE      # some axes give values that disagree / others fail
    elif [ "$all_boundary" -eq 1 ]; then
        status=LIMIT               # every axis hit a diagnosed limit or timed out
    elif [ "$any_silent" -eq 1 ]; then
        status=SILENT-CRASH        # an undiagnosed failure on some axis (bug)
    else
        status=SILENT-CRASH
    fi

    total+=1
    detail="r=$o_r|o0=$o_0|o2=$o_2 expect=$expect"
    emit_event "${base}" "$status" "$construct d=$depth $detail"

    pyv="PASSED"
    case "$status" in
        PASS)            npass+=1;  record="$construct $depth"; printf '%s\n' "$record" >> "$SAFE_TMP" ;;
        LIMIT)           nlimit+=1 ;;
        WRONG)           nwrong+=1; pyv="FAILED"; BAD="$BAD $construct:d$depth=WRONG(got=$v_r want=$expect)" ;;
        AXIS-DIVERGENCE) ndiv+=1;   pyv="FAILED"; BAD="$BAD $construct:d$depth=DIVERGE($detail)" ;;
        SILENT-CRASH)    ncrash+=1; pyv="FAILED"; BAD="$BAD $construct:d$depth=SILENT($detail)" ;;
    esac
    printf '  %-16s %-18s d=%-4s %s\n' "$status" "$construct" "$depth" "$detail"
    echo "$pyv tests/nesting_depth/generated/$f::${construct}_d${depth}"
done

echo
echo "── max correct nesting depth per construct (largest all-axis PASS) ──"
if [ -s "$SAFE_TMP" ]; then
    sort -k1,1 -k2,2n "$SAFE_TMP" | awk '{m[$1]=$2} END{for(k in m) print k, m[k]}' | sort | while read -r k d; do
        printf '  %-20s %s\n' "$k" "$d"
        emit_event "max_correct_${k}" "$d" "$k max correct nesting depth $d"
    done
fi
# constructs that never reached a single PASS
for k in $(awk -F'\t' 'NR>1{print $2}' "$GEN_DIR/MANIFEST.txt" | sort -u); do
    if ! grep -q "^$k " "$SAFE_TMP" 2>/dev/null; then
        printf '  %-20s %s\n' "$k" "0 (no correct depth)"
        emit_event "max_correct_${k}" "0" "$k never produced a correct value"
    fi
done

echo
echo "nesting_depth summary: total=$total pass=$npass limit=$nlimit wrong=$nwrong diverge=$ndiv silent_crash=$ncrash"
[ -n "$BAD" ] && echo "nesting_depth offenders:$BAD"

gate=PASS
if [ "$nwrong" -ne 0 ] || [ "$ndiv" -ne 0 ] || [ "$ncrash" -ne 0 ] || [ "$total" -eq 0 ]; then gate=FAIL; fi
emit_event "nesting_depth_gate" "$gate" \
    "total=$total pass=$npass limit=$nlimit wrong=$nwrong diverge=$ndiv silent_crash=$ncrash quick=$QUICK"

echo "nesting_depth gate: $gate"
if [ "$gate" = "PASS" ]; then
    echo "PASSED tests/nesting_depth::gate"
    exit 0
else
    echo "FAILED tests/nesting_depth::gate"
    exit 1
fi
