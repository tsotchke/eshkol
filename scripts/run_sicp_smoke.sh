#!/usr/bin/env bash
# run_sicp_smoke.sh — full-book SICP gate (ESH-0005).
#
# Runs every implemented program in tests/sicp/ under BOTH the JIT (-r) and
# AOT, then emits hard FAIL events for required full-book SICP systems that do
# not yet have runnable probes. Mirrors scripts/run_icc_smoke.sh: it emits
#   * pytest-style lines  : "PASSED tests/sicp/<file>::<mode>" / "FAILED ..."
#   * ICC JSON-L events   : kind=sicp_smoke, consumed by
#                           .icc/completion-oracles.yaml::sicp-completeness
#
# A program PASSES a mode when the run exits 0, prints no "^FAIL:" line, and
# prints no nonzero failure summary.
#
# Usage: scripts/run_sicp_smoke.sh [--no-aot] [--allow-incomplete]
set -u

# The timeout/json helpers are byte-oriented Perl snippets. Force a portable
# locale here, not only in the test wrapper, so real release runs do not depend
# on host-specific C.UTF-8 availability.
export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/sicp_smoke.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"
: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-sicp-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_sicp_smoke.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR\` first." >&2
    exit 2
fi

DO_AOT=1
ALLOW_INCOMPLETE=0
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        --allow-incomplete) ALLOW_INCOMPLETE=1 ;;
        *)
            echo "run_sicp_smoke.sh: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# macOS has no `timeout(1)`; emulate with perl alarm.
run_guarded() {
    perl -e 'my $seconds = shift; alarm $seconds; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

# Programs that are EXPECTED to fail (documented gaps, not gate regressions).
is_xfail() {
    case "$1" in
        *) return 1 ;;
    esac
}

emit_event() {
    local name="$1" value="$2" snippet="$3" esc_name esc_value esc_snippet
    esc_name=$(json_escape "$name")
    esc_value=$(json_escape "$value")
    esc_snippet=$(json_escape "$snippet")
    printf '{"kind":"sicp_smoke","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$esc_name" "$esc_value" "$esc_snippet" >> "$TRACE_FILE"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

# Decide PASS/FAIL from a captured run.
verdict() { # rc out -> echoes PASS|FAIL
    local rc="$1" out="$2"
    if [ "$rc" -ne 0 ]; then echo FAIL; return; fi
    if printf '%s' "$out" | grep -qE '^FAIL:|HAS-FAIL|Failed:[[:space:]]+[1-9]|FAILS'; then echo FAIL; return; fi
    echo PASS
}

full_book_required_bases="
ch2_picture_painters
ch2_generic_tower_coercion
ch2_polynomial_arithmetic
ch3_mutable_pairs_cycles
ch3_digital_circuit
ch3_constraints
ch3_concurrency
ch3_stream_acceleration
ch3_stream_power_series
ch3_stream_signal_systems
ch3_stream_random_monte_carlo
ch4_analyzing_evaluator
ch4_metacircular_derived_forms
ch4_lazy_evaluator
ch4_amb_evaluator
ch4_amb_parser
ch4_query_system
ch5_register_machine_stack
ch5_register_machine_recursive
ch5_storage_gc
ch5_explicit_control_evaluator
ch5_compiler
"

required_gap_label() {
    case "$1" in
        ch2_picture_painters)
            echo "SICP 2.2.4 picture language painters, transforms, combinators, and square-limit" ;;
        ch2_generic_tower_coercion)
            echo "SICP 2.5 generic arithmetic tower and coercion" ;;
        ch2_polynomial_arithmetic)
            echo "SICP 2.5.3 polynomial arithmetic and symbolic algebra" ;;
        ch3_mutable_pairs_cycles)
            echo "SICP 3.3.1 mutable pair sharing, append!, cycles, and count-pairs examples" ;;
        ch3_digital_circuit)
            echo "SICP 3.3.4 digital-circuit simulator" ;;
        ch3_constraints)
            echo "SICP 3.3.5 constraint-propagation system" ;;
        ch3_concurrency)
            echo "SICP 3.4 concurrency, serializers, and account exchange" ;;
        ch3_stream_acceleration)
            echo "SICP 3.5 accelerated streams, Euler transform, and tableaux" ;;
        ch3_stream_power_series)
            echo "SICP 3.5 power series streams" ;;
        ch3_stream_signal_systems)
            echo "SICP 3.5 stream integrators, signal systems, and zero crossings" ;;
        ch3_stream_random_monte_carlo)
            echo "SICP 3.5 random streams and Monte Carlo streams" ;;
        ch4_analyzing_evaluator)
            echo "SICP 4.1.7 analyzing evaluator" ;;
        ch4_metacircular_derived_forms)
            echo "SICP 4.1 derived forms in the metacircular evaluator" ;;
        ch4_lazy_evaluator)
            echo "SICP 4.2 lazy evaluator / normal-order evaluator" ;;
        ch4_amb_evaluator)
            echo "SICP 4.3 ambeval nondeterministic evaluator and driver loop" ;;
        ch4_amb_parser)
            echo "SICP 4.3 nondeterministic natural-language parser" ;;
        ch4_query_system)
            echo "SICP 4.4 logic query evaluator" ;;
        ch5_register_machine_stack)
            echo "SICP 5.1-5.2 register-machine stack operations and stack statistics" ;;
        ch5_register_machine_recursive)
            echo "SICP 5.1-5.2 recursive register machines" ;;
        ch5_storage_gc)
            echo "SICP 5.3 storage allocation and garbage-collector model" ;;
        ch5_explicit_control_evaluator)
            echo "SICP 5.4 explicit-control evaluator" ;;
        ch5_compiler)
            echo "SICP 5.5 SICP compiler targeting the register-machine simulator" ;;
        *)
            echo "full-book SICP required system" ;;
    esac
}

total=0; passed=0; xfailed=0; xpassed=0; gate_fail=0; coverage_missing=0; coverage_fail=0
echo "SICP smoke → $TRACE_FILE"
echo

for f in "$REPO_ROOT"/tests/sicp/*.esk; do
    base=$(basename "$f" .esk)
    # ----- JIT (-r) -----
    rout=$(run_guarded 180 "$ESHKOL_RUN" -r "$f" 2>&1); rrc=$?
    rv=$(verdict "$rrc" "$rout")
    # ----- AOT -----
    av="SKIP"
    if [ "$DO_AOT" -eq 1 ]; then
        bin="/tmp/sicp_${base}.bin"; rm -f "$bin"
        cout=$(run_guarded 180 "$ESHKOL_RUN" "$f" -o "$bin" 2>&1); crc=$?
        if [ "$crc" -ne 0 ]; then
            av=FAIL
        else
            aout=$(run_guarded 60 "$bin" 2>&1); arc=$?
            av=$(verdict "$arc" "$aout")
            rm -f "$bin"
        fi
    fi

    for mode in r aot; do
        [ "$mode" = "aot" ] && [ "$DO_AOT" -eq 0 ] && continue
        v=$rv; [ "$mode" = "aot" ] && v=$av
        total=$((total+1))
        emit_event "sicp_${base}_${mode}" "$v" "$base $mode -> $v"
        if is_xfail "$base"; then
            if [ "$v" = "PASS" ]; then
                xpassed=$((xpassed+1))
                printf '  XPASS tests/sicp/%s::%s  (expected fail but passed!)\n' "$base.esk" "$mode"
                echo "XPASS tests/sicp/$base.esk::$mode"
            else
                xfailed=$((xfailed+1))
                printf '  xfail tests/sicp/%s::%s\n' "$base.esk" "$mode"
                echo "XFAIL tests/sicp/$base.esk::$mode"
            fi
        else
            if [ "$v" = "PASS" ]; then
                passed=$((passed+1))
                printf '  PASS  tests/sicp/%s::%s\n' "$base.esk" "$mode"
                echo "PASSED tests/sicp/$base.esk::$mode"
            else
                gate_fail=$((gate_fail+1))
                printf '  FAIL  tests/sicp/%s::%s\n' "$base.esk" "$mode"
                echo "FAILED tests/sicp/$base.esk::$mode"
            fi
        fi
    done
done

echo
echo "Full-book required coverage:"
for base in $full_book_required_bases; do
    [ -e "$REPO_ROOT/tests/sicp/${base}.esk" ] && continue
    label=$(required_gap_label "$base")
    coverage_missing=$((coverage_missing+1))
    for mode in r aot; do
        [ "$mode" = "aot" ] && [ "$DO_AOT" -eq 0 ] && continue
        total=$((total+1))
        gate_fail=$((gate_fail+1))
        coverage_fail=$((coverage_fail+1))
        emit_event "sicp_${base}_${mode}" "FAIL" "$label missing: tests/sicp/${base}.esk"
        printf '  MISSING tests/sicp/%s::%s  (%s)\n' "$base.esk" "$mode" "$label"
        echo "FAILED tests/sicp/$base.esk::$mode"
    done
done

echo
echo "SICP smoke summary: $passed/$((passed+gate_fail)) gate probes PASS; ${xfailed} xfail, ${xpassed} XPASS; $total total."
if [ "$coverage_missing" -ne 0 ]; then
    echo "Full-book SICP gate: $coverage_missing required system probes missing."
    if [ "$ALLOW_INCOMPLETE" -eq 1 ]; then
        echo "Incomplete coverage accepted only because --allow-incomplete was provided."
    fi
fi
echo "Trace written: $TRACE_FILE"
# Gate fails on any real failure, unexpected XPASS, or missing full-book system.
# --allow-incomplete is for local subset runs only: it may forgive missing
# full-book probes, but never ordinary test failures or stale XFAILs.
non_coverage_fail=$((gate_fail - coverage_fail))
if [ "$non_coverage_fail" -ne 0 ] || [ "$xpassed" -ne 0 ]; then
    exit 1
fi
if [ "$coverage_fail" -ne 0 ] && [ "$ALLOW_INCOMPLETE" -ne 1 ]; then
    exit 1
fi
exit 0
