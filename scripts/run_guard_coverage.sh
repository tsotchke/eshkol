#!/usr/bin/env bash
# run_guard_coverage.sh — ESH-0101 guard coverage gate.
#
# Eshkol's R7RS `guard` had exactly one kind of coverage before this gate:
# programs that raise, catch with a `#t`/predicate/`else` clause, and are run
# on ONE engine (tests/error_handling/*.esk are compiled AOT by
# scripts/run_error_handling_tests.sh and never run under -r or the VM). Whole
# clause forms had no test anywhere in the repo — the R7RS `=>` cond-clause and
# the test-only `(test)` clause above all — and the differential harnesses that
# do run several engines compare the engines against EACH OTHER, so a form both
# engines get wrong the same way is invisible to them (the "shared-defect
# blindness" recorded as SW-06).
#
# This gate closes both holes at once:
#
#   stage 1  GOLDEN x ENGINE.  Every program in tests/error_handling/
#            guard_coverage/*.esk is run on FIVE engines and its normalized
#            stdout is compared against a hand-authored, engine-independent
#            golden (<case>.expected) derived from R7RS 4.2.7 — not against
#            another engine. An answer that is wrong everywhere still fails.
#              jit      $BUILD/eshkol-run -r f.esk
#              aot-o0   $BUILD/eshkol-run -O0 f.esk -o bin && ./bin
#              aot-o2   $BUILD/eshkol-run -O2 f.esk -o bin && ./bin
#              vm-src   $BUILD/eshkol-vm-standalone-test f.esk
#              vm-eskb  eshkol-run --profile hosted-vm --emit-eskb | vm
#            Two AOT optimization levels are separate axes on purpose: the
#            ESH-0102 lineage was an optimization-level-dependent crash that
#            -O0 alone could never have caught.
#
#   stage 2  FAIL-CLOSED.  tests/error_handling/guard_coverage/fatal/*.esk are
#            programs whose raise reaches no handler. Each must exit NONZERO,
#            name the failure on stderr, and print nothing past the raising
#            form (every probe ends with a MUST-NOT-PRINT sentinel). A guard
#            that swallowed an unmatched condition and returned a value would
#            print the sentinel and be caught here.
#
#   stage 3  MANIFEST.  ENGINES.tsv declares every (case, axis) pair that is
#            NOT required, with a mandatory justification. The default is that
#            every case is required on every axis, so a form that stops working
#            on an engine cannot be quietly dropped — it has to be written
#            down. The stage fails on a manifest row for a case or axis that
#            does not exist (stale waiver) and on any row without a
#            justification.
#
# Deliberately NOT covered here: `guard` in TAIL position. That is SW-58 and
# belongs to fix/tco-guard-tail-position; duplicating it here would give two
# lanes two different pins on the same behaviour.
#
# Emits (mirroring scripts/run_vm_parity.sh):
#   * pytest-style lines : "PASSED tests/error_handling/guard_coverage/<f>::<axis>"
#   * ICC JSON-L events  : kind=guard_coverage into
#                          scripts/icc_traces/guard_coverage.jsonl, consumed by
#                          .icc/completion-oracles.yaml::guard_coverage_gate
#
# Usage: scripts/run_guard_coverage.sh [--no-vm] [--no-eskb] [--case NAME]
#   BUILD_DIR selects the build directory (default: build).
set -u

# Byte-oriented gate; some macOS hosts inherit a UTF-8 locale name Perl
# cannot materialize (same reason run_vm_parity.sh pins it).
export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"
if eshkol_durable_enabled; then
    GC_WORK="$(eshkol_durable_prepare_dir guard-coverage)" || exit $?
    TRACE_DIR="${TRACE_DIR:-$GC_WORK/traces}"
else
    TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
fi
TRACE_FILE="$TRACE_DIR/guard_coverage.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?TRACE_FILE must be set}"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) : ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
VM_BIN="$BUILD_DIR/eshkol-vm-standalone-test"
[ -x "$VM_BIN" ] || VM_BIN="$BUILD_DIR/eshkol-vm-standalone"

CORPUS="$REPO_ROOT/tests/error_handling/guard_coverage"
FATAL_DIR="$CORPUS/fatal"
MANIFEST="$CORPUS/ENGINES.tsv"

DO_VM=1
DO_ESKB=1
ONLY_CASE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-vm) DO_VM=0; DO_ESKB=0 ;;
        --no-eskb) DO_ESKB=0 ;;
        --case) shift; ONLY_CASE="${1:-}" ;;
        *) echo "run_guard_coverage.sh: unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

TIMEOUT_RUN="${GUARD_COVERAGE_TIMEOUT:-120}"
WARMUP_TIMEOUT="${GUARD_COVERAGE_WARMUP_TIMEOUT:-300}"

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() { # name value snippet
    printf '{"kind":"guard_coverage","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$(json_escape "$1")" "$(json_escape "$2")" "$(json_escape "$3")" >> "$TRACE_FILE"
}

emit_test_result() { # name PASS|FAIL snippet
    local passed=false
    [ "$2" = "PASS" ] && passed=true
    printf '{"kind":"test_result","name":"%s","value":{"passed":%s,"summary":"%s"},"timestamp":%s}\n' \
        "$(json_escape "$1")" "$passed" "$(json_escape "$3")" "$(date +%s)" >> "${TRACE_FILE:?}"
}

pass=0; fail=0; infra=0; waived=0
report() { # PASS|FAIL|INFRA nodeid event_name snippet
    case "$1" in
        PASS)  pass=$((pass+1));  echo "PASSED $2" ;;
        INFRA) infra=$((infra+1)); echo "INFRA  $2 — $4 (no verdict obtained; not counted as a coverage defect)" ;;
        *)     fail=$((fail+1));  echo "FAILED $2 — $4" ;;
    esac
    emit_event "$3" "$1" "$4"
}

if [ ! -d "$CORPUS" ]; then
    echo "run_guard_coverage.sh: corpus $CORPUS not found" >&2; exit 2
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_guard_coverage.sh: need $ESHKOL_RUN — build with:" >&2
    echo "  cmake --build $BUILD_DIR --target eshkol-run stdlib eshkol-vm-standalone-test" >&2
    exit 2
fi
if [ $DO_VM -eq 1 ] && [ ! -x "$VM_BIN" ]; then
    echo "run_guard_coverage.sh: $VM_BIN missing; VM axes disabled (--no-vm to silence)" >&2
    DO_VM=0; DO_ESKB=0
fi

. "$REPO_ROOT/scripts/lib/build_fingerprint.sh"
eshkol_emit_build_fingerprint_event "$TRACE_DIR" "run_guard_coverage" "$BUILD_DIR" eshkol-run
if [ $DO_VM -eq 1 ]; then
    eshkol_emit_build_fingerprint_event "$TRACE_DIR" "run_guard_coverage" "$BUILD_DIR" "$(basename "$VM_BIN")"
fi

if eshkol_durable_enabled; then
    WORK="$GC_WORK/work"; mkdir -p "$WORK"
else
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-guard-coverage.XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT
fi
: "${WORK:?WORK must be set}"
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

# Pay the cold stdlib compile once, outside every timed window (identical
# reasoning to run_vm_parity.sh's warm-up; see F13 in that script).
WARM_FILE="$WORK/_warmup.esk"
printf '(display 1)\n(newline)\n' > "$WARM_FILE"
echo "== warm-up: priming \$ESHKOL_JIT_CACHE_DIR (budget ${WARMUP_TIMEOUT}s) =="
if eshkol_outcome_guarded "$WARMUP_TIMEOUT" "$ESHKOL_RUN" -r "$WARM_FILE" \
        >"$WORK/_warmup.out" 2>"$WORK/_warmup.err"; then
    echo "warm-up: cache primed"
else
    echo "run_guard_coverage.sh: warm-up did not finish within ${WARMUP_TIMEOUT}s;" \
         "per-case INFRA retry still applies (see $WORK/_warmup.err)" >&2
fi

# Normalization: drop engine banners/loader noise, remove ALL newlines (the
# VM's display appends one per call — filed as
# tests/vm_parity/found/display_newline_per_call.esk), then collapse runs of
# blanks and trim, so a golden can be written as one readable line.
normalize() { # infile outfile
    perl -ne 'next if
        /^WARN/ or /^INFO:/ or /^DEBUG/ or
        /^\[ESKB\]/ or /^\[GPU\]/ or /^\s*\[compiled:/ or
        /^=== Eshkol VM/ or /^=== Execution complete ===/ or
        /^remark:/ or /^warning: <unknown>/;
        print' "$1" \
      | tr -d '\n' \
      | perl -pe 's/[ \t]+/ /g; s/^ //; s/ $//' > "$2"
}

vm_stderr_clean() { # errfile
    ! grep -qE "ERROR|OVERFLOW|unhandled native call|Assertion|Segmentation|abort" "$1"
}

# ── stage 3 (read first): the engine manifest ───────────────────────────
AXES="jit aot-o0 aot-o2"
[ $DO_VM -eq 1 ] && AXES="$AXES vm-src"
[ $DO_ESKB -eq 1 ] && AXES="$AXES vm-eskb"

shopt -s nullglob
corpus_files=("$CORPUS"/*.esk)
if [ "${#corpus_files[@]}" -eq 0 ]; then
    echo "run_guard_coverage.sh: no corpus files in $CORPUS" >&2; exit 2
fi

manifest_ok=1
manifest_problems=""
declare -a WAIVER_KEYS=()
declare -a WAIVER_WHY=()
if [ -f "$MANIFEST" ]; then
    while IFS=$'\t' read -r m_case m_axis m_status m_why; do
        case "${m_case:-}" in ''|'#'*) continue ;; esac
        if [ ! -f "$CORPUS/$m_case.esk" ] && [ ! -f "$FATAL_DIR/$m_case.esk" ]; then
            manifest_ok=0
            manifest_problems="$manifest_problems stale-case:$m_case"
            continue
        fi
        case " jit aot-o0 aot-o2 vm-src vm-eskb " in
            *" $m_axis "*) : ;;
            *) manifest_ok=0; manifest_problems="$manifest_problems unknown-axis:$m_case/$m_axis"; continue ;;
        esac
        if [ "${m_status:-}" != "not-required" ]; then
            manifest_ok=0
            manifest_problems="$manifest_problems bad-status:$m_case/$m_axis/${m_status:-<empty>}"
            continue
        fi
        if [ -z "${m_why:-}" ]; then
            manifest_ok=0
            manifest_problems="$manifest_problems no-justification:$m_case/$m_axis"
            continue
        fi
        WAIVER_KEYS+=("$m_case/$m_axis")
        WAIVER_WHY+=("$m_why")
    done < "$MANIFEST"
fi

is_waived() { # case axis -> 0 if waived (sets WAIVER_REASON)
    local key="$1/$2" i=0
    WAIVER_REASON=""
    while [ $i -lt ${#WAIVER_KEYS[@]} ]; do
        if [ "${WAIVER_KEYS[$i]}" = "$key" ]; then
            WAIVER_REASON="${WAIVER_WHY[$i]}"
            return 0
        fi
        i=$((i+1))
    done
    return 1
}

echo
echo "== stage 3: engine manifest ($MANIFEST) =="
if [ $manifest_ok -eq 1 ]; then
    report PASS "tests/error_handling/guard_coverage/ENGINES.tsv::manifest" \
        "guard_coverage_manifest" \
        "${#WAIVER_KEYS[@]} declared (case,axis) waivers, each justified"
else
    report FAIL "tests/error_handling/guard_coverage/ENGINES.tsv::manifest" \
        "guard_coverage_manifest" "malformed rows:$manifest_problems"
fi

# ── stage 1: golden x engine ────────────────────────────────────────────
echo
echo "== stage 1: golden-vs-engine ($(echo $AXES | tr ' ' ',')) =="

run_axis() { # axis srcfile outdir -> writes $outdir/$axis.out, returns rc
    local axis="$1" f="$2" d="$3" rc=0
    case "$axis" in
        jit)
            eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/$axis.raw" "$d/$axis.err" \
                "$ESHKOL_RUN" -r "$f"; rc=$? ;;
        aot-o0|aot-o2)
            local opt=-O0; [ "$axis" = aot-o2 ] && opt=-O2
            if ! eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/$axis.compile.out" "$d/$axis.err" \
                    "$ESHKOL_RUN" "$opt" "$f" -L"$BUILD_DIR" -o "$d/$axis.bin"; then
                rc=$?
                : > "$d/$axis.raw"
                return $rc
            fi
            eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/$axis.raw" "$d/$axis.err" \
                "$d/$axis.bin"; rc=$? ;;
        vm-src)
            ESHKOL_VM_NO_DISASM=1 eshkol_outcome_retry_guarded "$TIMEOUT_RUN" \
                "$d/$axis.raw" "$d/$axis.err" "$VM_BIN" "$f"; rc=$? ;;
        vm-eskb)
            if ! eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/$axis.compile.out" "$d/$axis.err" \
                    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$d/prog.eskb" "$f"; then
                rc=$?
                : > "$d/$axis.raw"
                return $rc
            fi
            ESHKOL_VM_NO_DISASM=1 eshkol_outcome_retry_guarded "$TIMEOUT_RUN" \
                "$d/$axis.raw" "$d/$axis.err" "$VM_BIN" "$d/prog.eskb"; rc=$? ;;
    esac
    return $rc
}

for f in "${corpus_files[@]}"; do
    base=$(basename "$f" .esk)
    if [ -n "$ONLY_CASE" ] && [ "$base" != "$ONLY_CASE" ]; then continue; fi
    golden="$CORPUS/$base.expected"
    nodeid="tests/error_handling/guard_coverage/$base.esk"
    if [ ! -f "$golden" ]; then
        report FAIL "$nodeid::golden" "guard_${base}_golden" \
            "no $base.expected — every case must carry a hand-authored golden"
        continue
    fi
    d="$WORK/$base"; mkdir -p "$d"
    normalize "$golden" "$d/golden.out"

    for axis in $AXES; do
        if is_waived "$base" "$axis"; then
            waived=$((waived+1))
            echo "WAIVED $nodeid::$axis — $WAIVER_REASON"
            emit_event "guard_${base}_${axis}" "WAIVED" "$WAIVER_REASON"
            continue
        fi
        run_axis "$axis" "$f" "$d"; rc=$?
        cls=$(eshkol_outcome_classify_exit "$rc")
        normalize "$d/$axis.raw" "$d/$axis.out"
        if [ "$cls" = INFRA ]; then
            report INFRA "$nodeid::$axis" "guard_${base}_${axis}" \
                "$axis timed out/infra after retry (rc=$rc)"
        elif [ $rc -ne 0 ]; then
            report FAIL "$nodeid::$axis" "guard_${base}_${axis}" \
                "$axis exited $rc: $(head -c 200 "$d/$axis.err" 2>/dev/null)"
        elif [ "${axis#vm-}" != "$axis" ] && ! vm_stderr_clean "$d/$axis.err"; then
            report FAIL "$nodeid::$axis" "guard_${base}_${axis}" \
                "$axis errored on stderr: $(head -c 200 "$d/$axis.err")"
        elif ! cmp -s "$d/golden.out" "$d/$axis.out"; then
            report FAIL "$nodeid::$axis" "guard_${base}_${axis}" \
                "golden mismatch: want=<$(head -c 240 "$d/golden.out")> got=<$(head -c 240 "$d/$axis.out")>"
        else
            report PASS "$nodeid::$axis" "guard_${base}_${axis}" \
                "matches hand-authored golden ($(wc -c < "$d/golden.out" | tr -d ' ') bytes)"
        fi
    done
done

# ── stage 2: fail-closed probes ─────────────────────────────────────────
echo
echo "== stage 2: fail-closed probes (unhandled conditions must be fatal) =="
fatal_files=("$FATAL_DIR"/*.esk)
if [ "${#fatal_files[@]}" -eq 0 ]; then
    report FAIL "tests/error_handling/guard_coverage/fatal::present" \
        "guard_fatal_present" "no fail-closed probes found in $FATAL_DIR"
fi
for f in "${fatal_files[@]}"; do
    base=$(basename "$f" .esk)
    if [ -n "$ONLY_CASE" ] && [ "$base" != "$ONLY_CASE" ]; then continue; fi
    nodeid="tests/error_handling/guard_coverage/fatal/$base.esk"
    d="$WORK/fatal-$base"; mkdir -p "$d"
    for axis in $AXES; do
        if is_waived "$base" "$axis"; then
            waived=$((waived+1))
            echo "WAIVED $nodeid::$axis — $WAIVER_REASON"
            emit_event "guard_fatal_${base}_${axis}" "WAIVED" "$WAIVER_REASON"
            continue
        fi
        run_axis "$axis" "$f" "$d"; rc=$?
        cls=$(eshkol_outcome_classify_exit "$rc")
        normalize "$d/$axis.raw" "$d/$axis.out"
        got="$(cat "$d/$axis.out" 2>/dev/null)"
        if [ "$cls" = INFRA ]; then
            report INFRA "$nodeid::$axis" "guard_fatal_${base}_${axis}" \
                "$axis timed out/infra after retry (rc=$rc)"
        elif grep -q "MUST-NOT-PRINT" "$d/$axis.out" 2>/dev/null; then
            report FAIL "$nodeid::$axis" "guard_fatal_${base}_${axis}" \
                "FAIL-OPEN: execution continued past the unhandled condition (stdout=<$got>)"
        elif [ $rc -eq 0 ]; then
            report FAIL "$nodeid::$axis" "guard_fatal_${base}_${axis}" \
                "unhandled condition exited 0 — a fatal condition must never look like success"
        elif [ ! -s "$d/$axis.err" ]; then
            report FAIL "$nodeid::$axis" "guard_fatal_${base}_${axis}" \
                "died with rc=$rc but printed NO diagnostic on stderr (silent death)"
        else
            report PASS "$nodeid::$axis" "guard_fatal_${base}_${axis}" \
                "rc=$rc, diagnostic present, sentinel not printed"
        fi
    done
done

gate_status="$([ $fail -eq 0 ] && echo PASS || echo FAIL)"
gate_summary="$pass passed, $fail failed, $infra infra, $waived waived across: $(echo $AXES | tr ' ' ',')"
emit_event "guard_coverage_gate" "$gate_status" "$gate_summary"
emit_test_result "guard_coverage_gate" "$gate_status" "$gate_summary"
echo
echo "guard-coverage: $gate_summary"
[ $fail -eq 0 ] || exit 1
exit 0
