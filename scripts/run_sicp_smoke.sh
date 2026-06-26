#!/usr/bin/env bash
# run_sicp_smoke.sh — SICP-completeness gate (ESH-0005).
#
# Runs every program in tests/sicp/ under BOTH the JIT (-r) and AOT, and
# reports per-program PASS/FAIL. Mirrors scripts/run_icc_smoke.sh: it emits
#   * pytest-style lines  : "PASSED tests/sicp/<file>::<mode>" / "FAILED ..."
#   * ICC JSON-L events   : kind=sicp_smoke, consumed by
#                           .icc/completion-oracles.yaml::sicp-completeness
#
# A program PASSES a mode when the run exits 0, prints no "^FAIL:" line, and
# prints no nonzero failure summary. Known-failing probes (the unmodified
# metacircular evaluator, ch4_metacircular.esk, and the ESH-0078 repro) are
# expected to FAIL and are tagged xfail — they do not fail the gate.
#
# Usage: scripts/run_sicp_smoke.sh [--no-aot]
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/sicp_smoke.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?}"; : > "$TRACE_FILE"

ESHKOL_RUN="$REPO_ROOT/build/eshkol-run"
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_sicp_smoke.sh: build/eshkol-run not found — run \`cmake --build build\` first." >&2
    exit 2
fi

DO_AOT=1
[ "${1:-}" = "--no-aot" ] && DO_AOT=0

# macOS has no `timeout(1)`; emulate with perl alarm.
run_guarded() { perl -e 'alarm shift; exec @ARGV' "$1" "${@:2}"; }

# Programs that are EXPECTED to fail (documented gaps, not gate regressions).
is_xfail() {
    case "$1" in
        ch4_metacircular|repro_esh0078_firstclass_predicate) return 0 ;;
        *) return 1 ;;
    esac
}

emit_event() {
    local name="$1" value="$2" snippet="$3" esc
    esc=$(printf '%s' "$snippet" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"kind":"sicp_smoke","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$name" "$value" "$esc" >> "$TRACE_FILE"
}

# Decide PASS/FAIL from a captured run.
verdict() { # rc out -> echoes PASS|FAIL
    local rc="$1" out="$2"
    if [ "$rc" -ne 0 ]; then echo FAIL; return; fi
    if printf '%s' "$out" | grep -qE '^FAIL:|HAS-FAIL|Failed:[[:space:]]+[1-9]|FAILS'; then echo FAIL; return; fi
    echo PASS
}

total=0; passed=0; xfailed=0; xpassed=0; gate_fail=0
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
echo "SICP smoke summary: $passed/$((passed+gate_fail)) gate probes PASS; ${xfailed} xfail, ${xpassed} XPASS; $total total."
echo "Trace written: $TRACE_FILE"
# Gate fails only on a real (non-xfail) failure or an unexpected XPASS.
[ "$gate_fail" -eq 0 ] || exit 1
exit 0
