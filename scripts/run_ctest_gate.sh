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
#   A group may also declare a MEMBER FLOOR — the third column, or `-` for
#   none. "Matched at least one" is the right rule for a pillar whose test set
#   is configuration-dependent (the fixed-point engine's shared-ABI test only
#   exists when the shared library is built), but it is the WRONG rule for a
#   pillar whose whole claim is that several suites hold TOGETHER. The
#   exact-coefficient Taylor tier (P6) and the reverse-over-Taylor seed tangent
#   (P5) rewrote the SAME tower-extraction point from two directions, so
#   "68/68 and 18/18 in one run" is the acceptance; with no floor, deleting the
#   reverse-over-Taylor registration would leave the group matching, green, and
#   no longer making that claim. The floor is a MINIMUM, not an equality, so
#   adding tests to a pillar never fails the gate — a shrink-only ratchet, the
#   same shape as the P8 baselines.
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
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"
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

# Record which exact binary this ctest run is about (D-11 BUILD FRESHNESS).
# shellcheck source=lib/build_fingerprint.sh
. "$REPO_ROOT/scripts/lib/build_fingerprint.sh"
[ -x "$BUILD_DIR/eshkol-run" ] && eshkol_emit_build_fingerprint_event "$TRACE_DIR" "run_ctest_gate" "$BUILD_DIR" eshkol-run

# ── the gated pillars ────────────────────────────────────────────────────
# event_name<TAB>test-name regex<TAB>what the group certifies
#
# Each line becomes one roll-up event of kind "ctest", consumed by one
# criterion under `eshkol-compiler-readiness`.
CTEST_GATE_GROUPS=$(cat <<'GROUPS'
fixed_point_exact_accumulation_gate	^fixedpoint_	-	Fixed-point / i128 exact-accumulation engine
exact_input_ad_identity_gate	^(exact_point_ad|exact_taylor)_(runtime|aot)_smoke$	-	Exact-input AD identity tier
taylor_tower_exactness_gate	^(taylor_tower|taylor_tower_mono|exact_taylor|reverse_over_taylor|taylor_numerics|region_evac_taylor_exact)_(runtime|aot)_smoke$	12	Taylor-tower exactness stack (P1/P2/P5/P6/P11) in one run
runtime_closure_arity_spread_gate	^runtime_closure_arity_spread_	-	Runtime-closure gradient arity spread
define_library_same_unit_gate	^define_library_same_unit_	-	R7RS same-unit define-library resolution
module_load_path_engine_parity_gate	^load_path_engine_parity_test$	-	Relative (load …) resolves identically on every execution engine
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
import re, sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
# CTest reports a per-test TIMEOUT (its own TIMEOUT test property firing,
# distinct from a test that ran to completion and returned nonzero) as a
# failed testcase whose <failure>/<error> text names it — CTest itself has
# no separate JUnit "status" value for it. A timeout is exactly the shape
# scripts/lib/harness_outcome.sh calls INFRA: the harness could not obtain
# a verdict in the time budget, which says nothing about whether the code
# is right. Folding it into FAIL here is the same defect class the
# 2026-08-25 architecture audit measured in run_vm_parity.sh and
# run_language_coverage.sh (section 3) — a clock fact reported as a
# code-correctness verdict.
TIMEOUT_RE = re.compile(r"\btimeout\b", re.IGNORECASE)
for case in root.iter("testcase"):
    name = case.get("name") or ""
    if not name:
        continue
    status = case.get("status") or ""
    failure_el = case.find("failure")
    error_el = case.find("error")
    failed = (failure_el is not None or error_el is not None
              or status in ("fail", "failed", "notrun", "error"))
    skipped = case.find("skipped") is not None or status == "skipped"
    if skipped:
        continue
    detail = "%s in %ss" % (status or ("failed" if failed else "passed"),
                            case.get("time") or "?")
    if failed:
        failure_text = " ".join(filter(None, [
            (failure_el.get("message") if failure_el is not None else None),
            (failure_el.text if failure_el is not None else None),
            (error_el.get("message") if error_el is not None else None),
            (error_el.text if error_el is not None else None),
        ]))
        if TIMEOUT_RE.search(failure_text) or TIMEOUT_RE.search(status):
            print("%s\tINFRA\t%s" % (name, detail))
            continue
    print("%s\t%s\t%s" % (name, "FAIL" if failed else "PASS", detail))
PY
else
    # `      1/126 Test   #1: name .......   Passed    0.42 sec`
    # CTest's console verdict word for a per-test TIMEOUT firing is
    # literally "Timeout" — recognize it explicitly rather than folding it
    # into the FAIL bucket with every other non-"Passed" word (see the
    # JUnit branch above for why: a timeout is INFRA, not a code verdict).
    perl -ne '
        if (m{^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s*(\**)\s*(\w[\w ]*?)\s+([\d.]+)\s+sec}) {
            my ($n, $verdict, $secs) = ($1, $3, $4);
            $verdict =~ s/\s+$//;
            my $ok = ($verdict eq "Passed") ? "PASS"
                   : ($verdict =~ /timeout/i) ? "INFRA"
                   : "FAIL";
            print "$n\t$ok\t$verdict in ${secs}s\n";
        }
    ' "$RUN_LOG" > "$RESULTS"
fi

TOTAL=0; PASSED=0; FAILED=0; INFRA=0
while IFS="$(printf '\t')" read -r name verdict detail; do
    [ -n "$name" ] || continue
    TOTAL=$((TOTAL + 1))
    case "$verdict" in
        PASS)
            PASSED=$((PASSED + 1))
            echo "PASSED ctest::$name"
            emit_test_result "ctest::$name" "$verdict" "$detail"
            ;;
        INFRA)
            INFRA=$((INFRA + 1))
            echo "INFRA  ctest::$name — $detail (no verdict obtained; not counted as a failure)"
            # Deliberately no emit_test_result call: INFRA must never
            # publish a passed:false test_result record. See
            # scripts/lib/harness_outcome.sh, "WHY INFRA EMITS NO
            # test_result" — writing nothing leaves any prior PASS/FAIL on
            # file (within its freshness window) standing instead of
            # overwriting it with an environment-caused false red.
            ;;
        *)
            FAILED=$((FAILED + 1))
            echo "FAILED ctest::$name — $detail"
            emit_test_result "ctest::$name" "$verdict" "$detail"
            ;;
    esac
    emit_event "ctest_$name" "$verdict" "$detail"
done < "$RESULTS"

if [ "$TOTAL" -eq 0 ]; then
    emit_event "ctest_suite_green" FAIL "ctest produced no parseable test verdicts"
    emit_test_result "ctest::suite" FAIL "no parseable test verdicts"
    echo "ctest gate: FAIL — no test verdicts parsed from the run" >&2
    rm -f "$RESULTS"
    exit 1
fi

# ── self-verdict gate ────────────────────────────────────────────────────
# CTest's own PASS/FAIL comes from exit status (or a FAIL_REGULAR_EXPRESSION
# a handful of tests set) — it never looks at what a test actually PRINTED.
# scripts/check_self_verdicts.py reads the same JUnit file for the captured
# <system-out>/<system-err> of every test CTest counted as passing, and fails
# this gate if any of them self-reports a failure marker anyway (D-05: SW-24
# printed `FAIL: Expected 12` on a green baseline for months).
SELF_VERDICT_FAILURES=0
if [ "$HAVE_JUNIT" -eq 1 ] && [ -s "$JUNIT" ]; then
    if self_verdict_out=$(python3 "$REPO_ROOT/scripts/check_self_verdicts.py" \
            --junit "$JUNIT" --no-trace 2>&1); then
        emit_event "ctest_self_verdict_scan" PASS "no PASS-graded ctest test self-reports a failure"
        emit_test_result "ctest::self-verdict-scan" PASS "clean"
    else
        SELF_VERDICT_FAILURES=1
        echo "$self_verdict_out"
        emit_event "ctest_self_verdict_scan" FAIL "a PASS-graded ctest test's own output contains a self-reported failure marker"
        emit_test_result "ctest::self-verdict-scan" FAIL "contradiction found"
        echo "FAILED ctest::self-verdict-scan"
    fi
else
    echo "run_ctest_gate.sh: no JUnit output available (older CMake?) — self-verdict scan skipped" >&2
fi

# ── group roll-ups ───────────────────────────────────────────────────────
GROUP_FAILURES=0
GROUP_INFRA=0
while IFS="$(printf '\t')" read -r event regex floor label; do
    [ -n "${event:-}" ] || continue
    matched=0; g_pass=0; g_fail=0; g_infra=0; first_fail=""
    while IFS="$(printf '\t')" read -r name verdict detail; do
        [ -n "$name" ] || continue
        printf '%s' "$name" | grep -Eq "$regex" || continue
        matched=$((matched + 1))
        case "$verdict" in
            PASS)  g_pass=$((g_pass + 1)) ;;
            INFRA) g_infra=$((g_infra + 1)) ;;
            *)
                g_fail=$((g_fail + 1))
                [ -n "$first_fail" ] || first_fail="$name: $detail"
                ;;
        esac
    done < "$RESULTS"

    if [ "$matched" -eq 0 ]; then
        GROUP_FAILURES=$((GROUP_FAILURES + 1))
        emit_event "$event" FAIL "ABSENT: no configured test matches /$regex/ — $label is not covered by this build"
        emit_test_result "ctest-group::$event" FAIL "no test matches /$regex/"
        echo "FAILED ctest-group::$event — ABSENT (no test matches /$regex/)"
        continue
    fi
    # Member floor (see GROUPS above). A pillar that claims several suites hold
    # TOGETHER stops making that claim the moment one of them is unregistered,
    # and it does so while still matching, still green and still reported.
    if [ "$floor" != "-" ] && [ "$matched" -lt "$floor" ]; then
        GROUP_FAILURES=$((GROUP_FAILURES + 1))
        emit_event "$event" FAIL "SHRUNK: $matched configured test(s) match /$regex/, floor is $floor — $label no longer covers what it asserts"
        emit_test_result "ctest-group::$event" FAIL "$matched < floor $floor"
        echo "FAILED ctest-group::$event — SHRUNK ($matched < floor $floor)"
        continue
    fi
    if [ "$g_fail" -gt 0 ]; then
        GROUP_FAILURES=$((GROUP_FAILURES + 1))
        emit_event "$event" FAIL "$label: $g_fail/$matched failed — $first_fail"
        emit_test_result "ctest-group::$event" FAIL "$g_fail/$matched failed"
        echo "FAILED ctest-group::$event ($g_fail/$matched) — $first_fail"
    elif [ "$g_pass" -eq 0 ]; then
        # Every matched test in this group hit INFRA (e.g. its own per-test
        # ctest TIMEOUT) — no defect was observed, but no PASS was either.
        # Never counted toward GROUP_FAILURES (an unresolved clock fact must
        # not fail the gate), and never a false PASS either — reported as
        # its own state, loudly.
        GROUP_INFRA=$((GROUP_INFRA + 1))
        emit_event "$event" INFRA "$label: $g_infra/$matched infra (no verdict obtained)"
        echo "INFRA  ctest-group::$event ($g_infra/$matched) — no verdict obtained"
    else
        emit_event "$event" PASS "$label: $g_pass/$matched ctest gates green$([ "$g_infra" -gt 0 ] && printf '; %d infra' "$g_infra")"
        emit_test_result "ctest-group::$event" PASS "$g_pass/$matched green"
        echo "PASSED ctest-group::$event ($g_pass/$matched)$([ "$g_infra" -gt 0 ] && printf ' — %d infra (no verdict)' "$g_infra")"
    fi
done <<GROUPS_EOF
$CTEST_GATE_GROUPS
GROUPS_EOF

# ── suite roll-up ────────────────────────────────────────────────────────
#
# ctest itself returns nonzero on ANY non-Passed test, timeouts included, so
# CTEST_RC == 0 used to be required for a green gate — which means a single
# per-test ctest TIMEOUT (an environment fact: this host was too slow/loaded
# to finish inside the test's TIMEOUT property) failed the whole suite
# exactly like a real wrong-answer test would, indistinguishably. That is
# the same class of defect the 2026-08-25 architecture audit measured in
# run_vm_parity.sh and run_language_coverage.sh (section 3).
#
# The fix: PASS no longer requires CTEST_RC == 0 outright. It requires no
# real FAILED test and no failed/absent group — and if CTEST_RC is still
# nonzero, that nonzero must be FULLY explained by counted INFRA tests
# (INFRA -gt 0), never silently assumed. If CTEST_RC is nonzero for any
# OTHER reason — every parsed testcase says PASS/INFRA yet ctest itself
# disagrees — that is a harness contradiction (the same shape
# scripts/run_all_tests.sh already guards: individual verdicts and the
# aggregate disagreeing), and is treated as a failure rather than trusted.
SUMMARY="$PASSED/$TOTAL ctest tests passed"
if [ "$FAILED" -eq 0 ] && [ "$GROUP_FAILURES" -eq 0 ] && [ "$SELF_VERDICT_FAILURES" -eq 0 ]; then
    if [ "$CTEST_RC" -eq 0 ]; then
        emit_event "ctest_suite_green" PASS "$SUMMARY"
        emit_test_result "ctest::suite" PASS "$SUMMARY"
        echo
        echo "Trace written: $TRACE_FILE"
        echo "ctest gate: PASS ($SUMMARY)"
        rm -f "$RESULTS"
        exit 0
    elif [ "$INFRA" -gt 0 ]; then
        SUMMARY="$SUMMARY; $INFRA infra (no verdict, not counted as failure)"
        emit_event "ctest_suite_green" PASS "$SUMMARY"
        emit_test_result "ctest::suite" PASS "$SUMMARY"
        echo
        echo "Trace written: $TRACE_FILE"
        echo "ctest gate: PASS ($SUMMARY)"
        echo "WARNING: $INFRA ctest test(s) could not obtain a verdict (per-test TIMEOUT) — re-run under less contention if this persists." >&2
        rm -f "$RESULTS"
        exit 0
    else
        DETAIL="$SUMMARY; every parsed testcase is PASS/INFRA but ctest itself exited $CTEST_RC with 0 INFRA to explain it — harness contradiction, not trusted"
        emit_event "ctest_suite_green" FAIL "$DETAIL"
        emit_test_result "ctest::suite" FAIL "$DETAIL"
        echo
        echo "Trace written: $TRACE_FILE"
        echo "ctest gate: FAIL ($DETAIL)" >&2
        rm -f "$RESULTS"
        exit 1
    fi
fi

DETAIL="$SUMMARY; $FAILED failed; $INFRA infra; $GROUP_FAILURES group(s) failed or absent; $SELF_VERDICT_FAILURES self-verdict contradiction(s); ctest exit $CTEST_RC"
emit_event "ctest_suite_green" FAIL "$DETAIL"
emit_test_result "ctest::suite" FAIL "$DETAIL"
echo
echo "Trace written: $TRACE_FILE"
echo "ctest gate: FAIL ($DETAIL)" >&2
rm -f "$RESULTS"
exit 1
