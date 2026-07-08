#!/usr/bin/env bash
# run_recursion_depth.sh — depth-parametric recursion/control sweep (pillar P6b).
#
# Runs every generated probe in tests/recursion_depth/generated/ under BOTH the
# JIT (-r) and AOT, sweeping depth per KIND to find and gate the maximum safe
# depth of each recursion/control construct. Mirrors scripts/run_ad_oracle.sh:
#   * pytest-style lines : "PASSED tests/recursion_depth/generated/<f>::<mode>"
#   * ICC JSON-L events  : kind=recursion_depth, consumed by
#                          .icc/completion-oracles.yaml::recursion-depth
#
# Raw verdict per probe run (from exit signal + emitted diagnostic):
#   PASS          value correct, clean exit (probe printed "PASS:")
#   WRONG-VALUE   probe printed "WRONG:" — a silent wrong answer (BUG)
#   CLEAN-LIMIT   nonzero/fatal exit WITH a diagnostic (recursion-depth guard,
#                 "[Eshkol] fatal signal" handler) — a documented boundary
#   SILENT-CRASH  fatal signal (e.g. SIGILL rc132) with NO diagnostic (BUG)
#   HANG          exceeded the per-run timeout
#
# Each probe declares "; EXPECT: pass|limit|xknown <ESH-task>". The final
# status combines raw verdict with the expectation:
#   EXPECT pass  : must be PASS               (else -> gate FAIL, a regression)
#   EXPECT limit : PASS or CLEAN-LIMIT ok; SILENT-CRASH/WRONG-VALUE -> FAIL
#   EXPECT xknown: any failure is tolerated (XKNOWN, references an ESH task);
#                  if it now PASSes the run notes XPASS (task may be closable)
#
# Gate is PASS when no probe is a FAIL (SILENT-CRASH / WRONG-VALUE / HANG on a
# pass|limit cell, or a pass cell that hit a clean limit). Clean limits and
# tracked XKNOWN silent crashes are tolerated.
#
# CALIBRATION — which constructs MUST be unbounded vs. which have a clean ceiling:
#   * PROPER TAIL CALLS must run in constant stack per R7RS (section 3.5) — this
#     covers BOTH self recursion (self_tail) AND mutual recursion (mutual_tail2 /
#     mutual_tail3, e.g. even?/odd? or a state machine expressed as mutually
#     tail-calling functions). Their deep cells are therefore `pass`: a CLEAN-LIMIT
#     there is a real BUG (the tail call was not optimized) and fails the gate.
#     Mutual tail calls are emitted as LLVM `musttail` (see llvm_codegen.cpp), so
#     the 5,000,000-hop cells prove O(1) stack.
#   * NON-TAIL recursion (non_tail) keeps one native frame per level, so it has a
#     finite, environment-dependent stack ceiling and is NOT required to be
#     unbounded. Its deep cells are `limit`: a CLEAN-LIMIT (a caught SIGBUS/SIGSEGV/
#     SIGILL with a diagnostic + nonzero exit) is CORRECT graceful degradation and
#     is accepted; only a SILENT crash or WRONG value there fails the gate. This is
#     the deliberate distinction from the proper-tail-call kinds above.
#
# Usage: scripts/run_recursion_depth.sh [--quick] [--no-aot] [--regen]
#   --quick   skip the two deepest / slowest cells of self_tail & stdlib_length
#   --no-aot  skip the AOT lane
#   --regen   re-run the (deterministic) generator before running
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Keep the Perl timeout wrapper independent of host locale availability.
export LC_ALL=C
export LANG=C
export LC_CTYPE=C

GEN_DIR="$REPO_ROOT/tests/recursion_depth/generated"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/recursion_depth.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"
: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-recursion-depth-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_recursion_depth.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
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
        *) echo "run_recursion_depth.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$REGEN" -eq 1 ] || ! ls "$GEN_DIR"/rec_*.esk >/dev/null 2>&1; then
    python3 "$REPO_ROOT/scripts/gen_recursion_depth.py" || exit 2
fi

JIT_TIMEOUT="${JIT_TIMEOUT:-120}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-180}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-120}"

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
    printf '{"kind":"recursion_depth","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$esc_name" "$esc_value" "$esc_snippet" >> "$TRACE_FILE"
}

# raw_verdict rc out -> PASS|WRONG-VALUE|CLEAN-LIMIT|SILENT-CRASH|HANG
raw_verdict() {
    local rc="$1" out="$2"
    if [ "$rc" -eq 142 ]; then echo HANG; return; fi
    if printf '%s' "$out" | grep -q '^WRONG:'; then echo WRONG-VALUE; return; fi
    if [ "$rc" -eq 0 ]; then
        if printf '%s' "$out" | grep -q '^PASS:'; then echo PASS; return; fi
        echo SILENT-CRASH; return   # clean exit but never self-checked
    fi
    # nonzero exit (signal or error): a diagnostic makes it a clean limit
    if printf '%s' "$out" | grep -qiE \
        "maximum recursion depth|fatal signal|Unhandled exception|not-applicable|recursion depth|stack overflow|exceeded|[Ee]rror:"; then
        echo CLEAN-LIMIT; return
    fi
    echo SILENT-CRASH
}

# expectation directive from a probe file:
#   "pass" | "boundary" | "limit" | "xknown ESH-XXXX"
read_expect() { grep -m1 '^; EXPECT:' "$1" | sed 's/^; EXPECT:[[:space:]]*//'; }
read_kind()   { grep -m1 '^; KIND:'   "$1" | sed 's/^; KIND:[[:space:]]*//'; }
read_depth()  { grep -m1 '^; DEPTH:'  "$1" | sed 's/^; DEPTH:[[:space:]]*//'; }

declare -i total=0 npass=0 nlimit=0 nxknown=0 nfail=0
BAD=""
# per-kind max safe depth (largest PASS depth), tracked via temp file
SAFE_TMP="$(mktemp)"; : "${SAFE_TMP:?SAFE_TMP must be set}"; : > "$SAFE_TMP"

record_safe() { # kind depth
    printf '%s\t%s\n' "$1" "$2" >> "$SAFE_TMP"
}

# combine raw verdict with expectation -> final status + accounting
# args: raw expect kind depth file mode
classify() {
    local raw="$1" expect="$2" kind="$3" depth="$4" f="$5" mode="$6"
    local exp_kind exp_task status pyv
    exp_kind="${expect%% *}"
    exp_task="${expect#* }"; [ "$exp_task" = "$expect" ] && exp_task=""

    case "$exp_kind" in
        pass)
            if [ "$raw" = "PASS" ]; then status=PASS; record_safe "$kind" "$depth"
            else status="FAIL($raw)"; fi ;;
        boundary)
            if [ "$raw" = "PASS" ]; then status="PASS(boundary-not-hit)"; record_safe "$kind" "$depth"
            elif [ "$raw" = "CLEAN-LIMIT" ]; then status=CLEAN-BOUNDARY
            else status="FAIL($raw)"; fi ;;
        limit)
            if [ "$raw" = "PASS" ]; then status="PASS(limit-not-hit)"; record_safe "$kind" "$depth"
            elif [ "$raw" = "CLEAN-LIMIT" ]; then status=CLEAN-LIMIT
            else status="FAIL($raw)"; fi ;;
        xknown)
            if [ "$raw" = "PASS" ]; then status="XPASS($exp_task)"; record_safe "$kind" "$depth"
            else status="XKNOWN($exp_task:$raw)"; fi ;;
        *)  status="FAIL(bad-expect:$expect)" ;;
    esac

    total+=1
    emit_event "${f%.esk}_${mode}" "$status" "$kind d=$depth $mode raw=$raw expect=$expect"
    case "$status" in
        PASS|PASS\(*)   npass+=1;  pyv="PASSED" ;;
        CLEAN-LIMIT|CLEAN-BOUNDARY) nlimit+=1; pyv="PASSED" ;;
        XKNOWN*)        nxknown+=1; pyv="XFAIL" ;;
        XPASS*)         nxknown+=1; pyv="XPASS" ;;
        FAIL*)          nfail+=1;  pyv="FAILED"; BAD="$BAD $kind:d$depth:$mode=$status" ;;
    esac
    printf '  %-24s %-14s d=%-9s %s\n' "$status" "$kind" "$depth" "$mode"
    echo "$pyv tests/recursion_depth/generated/$f::$mode"
}

echo "recursion-depth sweep (P6b) -> $TRACE_FILE"
echo

for path in "$GEN_DIR"/rec_*.esk; do
    f=$(basename "$path")
    base="${f%.esk}"
    kind=$(read_kind "$path")
    depth=$(read_depth "$path")
    expect=$(read_expect "$path")

    if [ "$QUICK" -eq 1 ]; then
        case "$f" in
            rec_self_tail_d100000000.esk|rec_self_tail_d10000000.esk) continue ;;
            rec_stdlib_length_d1000000.esk) continue ;;
        esac
    fi

    # ----- JIT (-r) -----
    rout=$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$path" 2>&1); rrc=$?
    rraw=$(raw_verdict "$rrc" "$rout")
    classify "$rraw" "$expect" "$kind" "$depth" "$f" "r"

    # ----- AOT -----
    if [ "$DO_AOT" -eq 1 ]; then
        bin="${TMPDIR:-/tmp}/rec_depth_${base}.bin"; rm -f "$bin"
        cout=$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$path" -o "$bin" 2>&1); crc=$?
        if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
            if printf '%s' "$cout" | grep -qiE "maximum recursion depth|[Ee]rror:|fatal signal"; then
                araw=CLEAN-LIMIT; else araw=SILENT-CRASH; fi
            [ "$crc" -eq 142 ] && araw=HANG
        else
            aout=$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1); arc=$?
            araw=$(raw_verdict "$arc" "$aout")
        fi
        rm -f "$bin"
        classify "$araw" "$expect" "$kind" "$depth" "$f" "aot"
    fi
done

echo
echo "── max safe depth per kind (largest PASS depth observed) ──"
if [ -s "$SAFE_TMP" ]; then
    sort -t$'\t' -k1,1 -k2,2n "$SAFE_TMP" | awk -F'\t' '{m[$1]=$2} END{for(k in m) print k, m[k]}' | sort | while read k d; do
        printf '  %-16s %s\n' "$k" "$d"
        emit_event "max_safe_${k}" "$d" "$k max safe depth $d"
    done
fi
rm -f "$SAFE_TMP"

echo
echo "recursion_depth summary: total=$total pass=$npass clean_limit=$nlimit xknown=$nxknown fail=$nfail"
[ -n "$BAD" ] && echo "recursion_depth offenders:$BAD"

gate=PASS
if [ "$nfail" -ne 0 ] || [ "$total" -eq 0 ]; then gate=FAIL; fi
emit_event "recursion_depth_gate" "$gate" \
    "total=$total pass=$npass clean_limit=$nlimit xknown=$nxknown fail=$nfail quick=$QUICK aot=$DO_AOT"

echo "recursion_depth gate: $gate"
if [ "$gate" = "PASS" ]; then
    echo "PASSED tests/recursion_depth::gate"
    exit 0
else
    echo "FAILED tests/recursion_depth::gate"
    exit 1
fi
