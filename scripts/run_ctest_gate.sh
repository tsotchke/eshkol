#!/usr/bin/env bash
# run_ctest_gate.sh — turn the CTest suite into ICC evidence.
#
# WHY THIS EXISTS
#   Until this script, no completion-oracle criterion anywhere consumed a CTest
#   RESULT.  The only mentions of ctest in .icc/completion-oracles.yaml were
#   `action:` strings — advice on what to run — and the nine `test_evidence`
#   criteria are index-level ("the tests exist and are runnable"), not
#   execution-backed.  The whole CTest suite therefore had no oracle teeth: a
#   pillar could ship with a perfectly good CTest gate and still be ungated by
#   the release readiness target, which is what happened to the fixed-point
#   engine and the exact-input AD tier.
#
#   This is the structural fix.  It runs CTest, parses its per-test verdicts,
#   and writes them into scripts/icc_traces/ in the trace shapes the oracle and
#   the architecture invariants already read:
#
#     kind "ctest"        one event per test, plus one per named GROUP below,
#                         plus the `ctest_suite_green` roll-up.  This is what
#                         the completion-oracle runtime_event criteria match.
#     kind "test_result"  the canonical execution proof ICC's architecture
#                         invariants consume, one per test.
#
#   It also prints `PASSED <suite>::<test>` / `FAILED <suite>::<test>` lines, the
#   pytest-style node ids ICC's runtime-evidence reader recognizes, so the run
#   is legible without opening the trace.
#
# GROUPS
#   A group is a regex over test names with its own roll-up event, so a pillar
#   can be gated by ONE criterion that stays correct as tests are added to it.
#   Adding a pillar means adding a line to CTEST_GATE_GROUPS and a criterion in
#   .icc/completion-oracles.yaml — not another bespoke smoke probe that shells
#   out to ctest and re-runs the same binaries.
#
#   A group whose regex matches NOTHING is reported as ABSENT and fails the
#   gate: a criterion that silently stops being covered because its tests were
#   renamed or configured out is exactly the hole this script closes.
#
# USAGE
#   scripts/run_ctest_gate.sh [--build-dir DIR] [-- <extra ctest args>]
#   BUILD_DIR=build-quantum scripts/run_ctest_gate.sh
#
# EXIT
#   0 when every executed test passed and every group is present and green.
#   1 on any failure (the trace still records what happened).
#   2 when the build directory has no test configuration at all.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/ctest_gate.jsonl"
mkdir -p "$TRACE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="${2:-}"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "run_ctest_gate.sh: unknown flag: $1" >&2; exit 2 ;;
    esac
done
case "$BUILD_DIR" in
    /*) : ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac

if [ ! -r "$BUILD_DIR/CTestTestfile.cmake" ]; then
    echo "run_ctest_gate.sh: no CTest configuration in $BUILD_DIR" >&2
    echo "  configure with -DESHKOL_BUILD_TESTS=ON and build first." >&2
    exit 2
fi

# ── the gated pillars ────────────────────────────────────────────────────
# event_name<TAB>test-name regex<TAB>what the group certifies
#
# Each line becomes one roll-up event of kind "ctest", consumed by one
# criterion under `eshkol-compiler-readiness`.
CTEST_GATE_GROUPS=$(cat <<'GROUPS'
fixed_point_exact_accumulation_gate	^fixedpoint_	Fixed-point / i128 exact-accumulation engine
exact_input_ad_identity_gate	^(exact_point_ad|exact_taylor)_(runtime|aot)_smoke$	Exact-input AD identity tier
runtime_closure_arity_spread_gate	^runtime_closure_arity_spread_	Runtime-closure gradient arity spread
define_library_same_unit_gate	^define_library_same_unit_	R7RS same-unit define-library resolution
GROUPS
)

# ── trace emission ───────────────────────────────────────────────────────
: "${TRACE_FILE:?}"
: > "$TRACE_FILE"

emit_event() { # name PASS|FAIL snippet
    python3 -c '
import json, sys
print(json.dumps({"kind": "ctest", "name": sys.argv[1], "value": sys.argv[2],
                  "snippet": sys.argv[3], "confidence": 0.95},
                 ensure_ascii=False))
' "$1" "$2" "$3" >> "${TRACE_FILE:?}"
}

emit_test_result() { # name PASS|FAIL summary
    python3 -c '
import json, sys, time
print(json.dumps({"kind": "test_result", "name": sys.argv[1],
                  "value": {"passed": sys.argv[2] == "PASS",
                            "summary": sys.argv[3]},
                  "timestamp": int(time.time())}, ensure_ascii=False))
' "$1" "$2" "$3" >> "${TRACE_FILE:?}"
}

# ── run ──────────────────────────────────────────────────────────────────
RUN_LOG="$(mktemp "${TMPDIR:-/tmp}/eshkol-ctest-gate.XXXXXX")"
JUNIT="$(mktemp "${TMPDIR:-/tmp}/eshkol-ctest-junit.XXXXXX").xml"
cleanup() { rm -f "$RUN_LOG" "$JUNIT"; }
trap cleanup EXIT

echo "== ctest gate =="
echo "build dir: $BUILD_DIR"
echo

# --output-junit needs CMake >= 3.21; fall back to parsing the console
# summary when the host's ctest is older, so the gate never silently
# reports "no tests" on an older toolchain.
if ctest --test-dir "$BUILD_DIR" --output-junit "$JUNIT" -N >/dev/null 2>&1; then
    HAVE_JUNIT=1
else
    HAVE_JUNIT=0
    rm -f "$JUNIT"
fi

if [ "$HAVE_JUNIT" -eq 1 ]; then
    ctest --test-dir "$BUILD_DIR" --output-on-failure --output-junit "$JUNIT" \
        ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"} >"$RUN_LOG" 2>&1
else
    ctest --test-dir "$BUILD_DIR" --output-on-failure \
        ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"} >"$RUN_LOG" 2>&1
fi
CTEST_RC=$?

tail -40 "$RUN_LOG"
echo

# ── parse per-test verdicts ──────────────────────────────────────────────
# One "<name>\t<PASS|FAIL>\t<detail>" line per test on stdout.
RESULTS="$(mktemp "${TMPDIR:-/tmp}/eshkol-ctest-results.XXXXXX")"
if [ "$HAVE_JUNIT" -eq 1 ] && [ -s "$JUNIT" ]; then
    python3 - "$JUNIT" > "$RESULTS" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
for case in root.iter("testcase"):
    name = case.get("name") or ""
    if not name:
        continue
    status = case.get("status") or ""
    failed = (case.find("failure") is not None
              or case.find("error") is not None
              or status in ("fail", "failed", "notrun", "error"))
    skipped = case.find("skipped") is not None or status == "skipped"
    if skipped:
        continue
    detail = "%s in %ss" % (status or ("failed" if failed else "passed"),
                            case.get("time") or "?")
    print("%s\t%s\t%s" % (name, "FAIL" if failed else "PASS", detail))
PY
else
    # `      1/126 Test   #1: name .......   Passed    0.42 sec`
    perl -ne '
        if (m{^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s*(\**)\s*(\w[\w ]*?)\s+([\d.]+)\s+sec}) {
            my ($n, $verdict, $secs) = ($1, $3, $4);
            $verdict =~ s/\s+$//;
            my $ok = ($verdict eq "Passed") ? "PASS" : "FAIL";
            print "$n\t$ok\t$verdict in ${secs}s\n";
        }
    ' "$RUN_LOG" > "$RESULTS"
fi

TOTAL=0; PASSED=0; FAILED=0
while IFS="$(printf '\t')" read -r name verdict detail; do
    [ -n "$name" ] || continue
    TOTAL=$((TOTAL + 1))
    if [ "$verdict" = "PASS" ]; then
        PASSED=$((PASSED + 1))
        echo "PASSED ctest::$name"
    else
        FAILED=$((FAILED + 1))
        echo "FAILED ctest::$name — $detail"
    fi
    emit_event "ctest_$name" "$verdict" "$detail"
    emit_test_result "ctest::$name" "$verdict" "$detail"
done < "$RESULTS"

if [ "$TOTAL" -eq 0 ]; then
    emit_event "ctest_suite_green" FAIL "ctest produced no parseable test verdicts"
    emit_test_result "ctest::suite" FAIL "no parseable test verdicts"
    echo "ctest gate: FAIL — no test verdicts parsed from the run" >&2
    rm -f "$RESULTS"
    exit 1
fi

# ── group roll-ups ───────────────────────────────────────────────────────
GROUP_FAILURES=0
while IFS="$(printf '\t')" read -r event regex label; do
    [ -n "${event:-}" ] || continue
    matched=0; g_pass=0; g_fail=0; first_fail=""
    while IFS="$(printf '\t')" read -r name verdict detail; do
        [ -n "$name" ] || continue
        printf '%s' "$name" | grep -Eq "$regex" || continue
        matched=$((matched + 1))
        if [ "$verdict" = "PASS" ]; then
            g_pass=$((g_pass + 1))
        else
            g_fail=$((g_fail + 1))
            [ -n "$first_fail" ] || first_fail="$name: $detail"
        fi
    done < "$RESULTS"

    if [ "$matched" -eq 0 ]; then
        GROUP_FAILURES=$((GROUP_FAILURES + 1))
        emit_event "$event" FAIL "ABSENT: no configured test matches /$regex/ — $label is not covered by this build"
        emit_test_result "ctest-group::$event" FAIL "no test matches /$regex/"
        echo "FAILED ctest-group::$event — ABSENT (no test matches /$regex/)"
        continue
    fi
    if [ "$g_fail" -eq 0 ]; then
        emit_event "$event" PASS "$label: $g_pass/$matched ctest gates green"
        emit_test_result "ctest-group::$event" PASS "$g_pass/$matched green"
        echo "PASSED ctest-group::$event ($g_pass/$matched)"
    else
        GROUP_FAILURES=$((GROUP_FAILURES + 1))
        emit_event "$event" FAIL "$label: $g_fail/$matched failed — $first_fail"
        emit_test_result "ctest-group::$event" FAIL "$g_fail/$matched failed"
        echo "FAILED ctest-group::$event ($g_fail/$matched) — $first_fail"
    fi
done <<GROUPS_EOF
$CTEST_GATE_GROUPS
GROUPS_EOF

# ── suite roll-up ────────────────────────────────────────────────────────
SUMMARY="$PASSED/$TOTAL ctest tests passed"
if [ "$FAILED" -eq 0 ] && [ "$GROUP_FAILURES" -eq 0 ] && [ "$CTEST_RC" -eq 0 ]; then
    emit_event "ctest_suite_green" PASS "$SUMMARY"
    emit_test_result "ctest::suite" PASS "$SUMMARY"
    echo
    echo "Trace written: $TRACE_FILE"
    echo "ctest gate: PASS ($SUMMARY)"
    rm -f "$RESULTS"
    exit 0
fi

DETAIL="$SUMMARY; $FAILED failed; $GROUP_FAILURES group(s) failed or absent; ctest exit $CTEST_RC"
emit_event "ctest_suite_green" FAIL "$DETAIL"
emit_test_result "ctest::suite" FAIL "$DETAIL"
echo
echo "Trace written: $TRACE_FILE"
echo "ctest gate: FAIL ($DETAIL)" >&2
rm -f "$RESULTS"
exit 1
