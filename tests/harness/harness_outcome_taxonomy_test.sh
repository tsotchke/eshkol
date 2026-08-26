#!/usr/bin/env bash
# harness_outcome_taxonomy_test.sh — regression test for
# scripts/lib/harness_outcome.sh: proves the PASS/FAIL/INFRA distinction is
# real, not aspirational documentation.
#
# WHY THIS EXISTS
#
# The 2026-08-25 architecture audit (ESHKOL-ARCHITECTURE-AUDIT-2026-08-25.md,
# section 3) found two harnesses that could not tell "the code failed" from
# "the harness could not run" — a 140s cold-start JIT compile killed by a
# 60s alarm read as a VM-parity defect, and an unrelated flaky test aborting
# a full-suite prerequisite read as a language-coverage regression. Both
# incidents have the same shape: a real environmental non-completion
# (timeout, missing binary) getting reported through the same channel as a
# genuine wrong answer, with nothing downstream able to tell them apart.
#
# This test forces both shapes on purpose and asserts they land in
# DIFFERENT buckets:
#   1. An INFRA condition (timeout; missing binary) must classify as INFRA
#      and — critically — must NOT publish a test_result record, per
#      harness_outcome.sh's own "WHY INFRA EMITS NO test_result" contract.
#   2. A genuine wrong answer (the code ran to completion and returned
#      nonzero, or crashed on a signal it raised itself) must still
#      classify as FAIL and DOES publish a passed:false test_result.
#   3. The retry-once policy never gives a real FAIL a second chance, but
#      does retry a transient INFRA condition exactly once.
#
# If a future edit collapses this distinction (e.g. by widening the INFRA
# exit-code set to swallow a real FAIL, or by making FAIL retry), this test
# fails.
set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-harness-outcome-taxonomy-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
check() { # description ok(0/1)
    if [ "$2" -eq 0 ]; then
        pass=$((pass + 1))
        echo "PASS: $1"
    else
        fail=$((fail + 1))
        echo "FAIL: $1"
    fi
}

# ── 1. timeout classifies as INFRA, not FAIL, and is actually enforced ──
start=$(date +%s)
eshkol_outcome_guarded 1 sleep 30
rc=$?
elapsed=$(( $(date +%s) - start ))
check "eshkol_outcome_guarded kills a real hang instead of waiting for it (elapsed=${elapsed}s < 10s)" \
    "$([ "$elapsed" -lt 10 ] && echo 0 || echo 1)"
check "a timeout exit code (rc=$rc) is exactly 124" "$([ "$rc" -eq 124 ] && echo 0 || echo 1)"
check "eshkol_outcome_classify_exit(124) = INFRA" \
    "$([ "$(eshkol_outcome_classify_exit "$rc")" = INFRA ] && echo 0 || echo 1)"

# ── 2. a missing binary classifies as INFRA (exec-not-found = 127) ──
eshkol_outcome_guarded 5 "$WORK/this-binary-does-not-exist-$$"
rc=$?
check "exec of a missing binary reports 127 (rc=$rc)" "$([ "$rc" -eq 127 ] && echo 0 || echo 1)"
check "eshkol_outcome_classify_exit(127) = INFRA (missing dependency, per the taxonomy)" \
    "$([ "$(eshkol_outcome_classify_exit "$rc")" = INFRA ] && echo 0 || echo 1)"

# ── 3. a genuine wrong answer (nonzero, ran to completion) is FAIL ──
eshkol_outcome_guarded 5 sh -c 'exit 1'
rc=$?
check "a completed nonzero exit (rc=$rc) is exactly 1, not folded into a timeout code" \
    "$([ "$rc" -eq 1 ] && echo 0 || echo 1)"
check "eshkol_outcome_classify_exit(1) = FAIL" \
    "$([ "$(eshkol_outcome_classify_exit "$rc")" = FAIL ] && echo 0 || echo 1)"

# ── 4. a self-inflicted crash is still FAIL, never mistaken for our alarm ──
# This is the exact defect class the audit's own bridging fix documents:
# the OLD exec-then-perl-alarm pattern could not tell "our alarm fired" from
# "the child raised a real signal on its own", because exec replaces the
# process image the alarm handler lived in. eshkol_outcome_guarded forks
# instead, so a real SIGSEGV the child raises on its own is preserved as
# 128+11=139, never masked as our 124.
eshkol_outcome_guarded 5 sh -c 'kill -SEGV $$'
rc=$?
check "a self-raised SIGSEGV reports 128+11=139 (rc=$rc), not 124" \
    "$([ "$rc" -eq 139 ] && echo 0 || echo 1)"
check "eshkol_outcome_classify_exit(139) = FAIL (a real crash is a defect, not infra)" \
    "$([ "$(eshkol_outcome_classify_exit "$rc")" = FAIL ] && echo 0 || echo 1)"

# ── 5. INFRA never publishes a test_result; PASS/FAIL always do ──
trace="$WORK/trace.jsonl"
: > "$trace"
eshkol_outcome_emit_test_result "$trace" probe_infra INFRA "should not appear"
eshkol_outcome_emit_test_result "$trace" probe_fail FAIL "a real wrong answer"
eshkol_outcome_emit_test_result "$trace" probe_pass PASS "a real correct answer"
infra_lines=$(grep -c '"name": "probe_infra"' "$trace" 2>/dev/null || true)
[ -z "$infra_lines" ] && infra_lines=0
fail_lines=$(grep -c '"name": "probe_fail"' "$trace" 2>/dev/null || true)
[ -z "$fail_lines" ] && fail_lines=0
pass_lines=$(grep -c '"name": "probe_pass"' "$trace" 2>/dev/null || true)
[ -z "$pass_lines" ] && pass_lines=0
check "INFRA writes NO test_result record (found $infra_lines)" \
    "$([ "$infra_lines" -eq 0 ] && echo 0 || echo 1)"
check "a genuine FAIL DOES write a test_result record (found $fail_lines)" \
    "$([ "$fail_lines" -eq 1 ] && echo 0 || echo 1)"
check "a genuine PASS DOES write a test_result record (found $pass_lines)" \
    "$([ "$pass_lines" -eq 1 ] && echo 0 || echo 1)"
if [ "$fail_lines" -eq 1 ]; then
    check "the FAIL record's passed field is false" \
        "$(grep '"name": "probe_fail"' "$trace" | grep -q '"passed": false' && echo 0 || echo 1)"
fi
if [ "$pass_lines" -eq 1 ]; then
    check "the PASS record's passed field is true" \
        "$(grep '"name": "probe_pass"' "$trace" | grep -q '"passed": true' && echo 0 || echo 1)"
fi

# ── 6. retry-once: INFRA gets exactly one retry; a real FAIL never does ──
marker="$WORK/warm-marker"
rm -f "$marker" "$WORK/attempts-infra"
cat > "$WORK/cold-then-warm.sh" <<'EOF'
#!/usr/bin/env bash
marker="$1"
if [ ! -e "$marker" ]; then
    touch "$marker"
    sleep 30   # first attempt: looks like a hang
else
    echo "warm"
    exit 0     # second attempt: fast and correct
fi
EOF
chmod +x "$WORK/cold-then-warm.sh"
eshkol_outcome_retry_guarded 1 "$WORK/out.txt" "$WORK/err.txt" \
    "$WORK/cold-then-warm.sh" "$marker"
rc=$?
check "retry_guarded recovers a transient INFRA condition on its one retry (rc=$rc)" \
    "$([ "$rc" -eq 0 ] && grep -q warm "$WORK/out.txt" && echo 0 || echo 1)"

rm -f "$WORK/attempts-fail"
eshkol_outcome_retry_guarded 5 "$WORK/out2.txt" "$WORK/err2.txt" \
    sh -c 'echo x >> "'"$WORK"'/attempts-fail"; exit 1'
rc=$?
attempts=$(wc -l < "$WORK/attempts-fail" 2>/dev/null | tr -d ' ')
check "retry_guarded returns the real FAIL's own exit code (rc=$rc)" \
    "$([ "$rc" -eq 1 ] && echo 0 || echo 1)"
check "retry_guarded NEVER retries a real FAIL (ran exactly once, ran $attempts times)" \
    "$([ "$attempts" = "1" ] && echo 0 || echo 1)"

echo
echo "harness_outcome_taxonomy_test.sh: $pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
    echo "harness_outcome_taxonomy_test.sh: PASS"
    exit 0
fi
echo "harness_outcome_taxonomy_test.sh: FAIL"
exit 1
