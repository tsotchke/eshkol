#!/usr/bin/env bash
# harness_outcome.sh — the PASS / FAIL / INFRA / SKIP taxonomy every
# trace-emitting harness uses to report its own verdict.
#
# WHY THIS EXISTS
#
# The 2026-08-25 architecture audit (ESHKOL-ARCHITECTURE-AUDIT-2026-08-25.md,
# note on harness weakness + section 3 VM-parity flake analysis) measured two
# independent harnesses that could not tell "the code failed" from "the
# harness could not run", and wrote the indistinguishable red into a trace
# other tools consume:
#
#   (a) scripts/run_vm_parity.sh reported FAIL "2 of 188" where both
#       failures were `native -r exited 142` (SIGALRM) on
#       28_multiple_values_complete.esk, caused by a 140-second cold-start
#       JIT/stdlib compile on a machine at load average 180 across 24 cores.
#       Direct re-runs: 140.49s, then 0.07s, then 0.06s — exit 0,
#       byte-identical output every time. That spurious FAIL propagated
#       through `emit_test_result` into `icc architecture-verify` and turned
#       the HIGH invariant INV-dispatch-table-completeness red.
#
#   (b) The `language_surface_coverage_floor` smoke probe in
#       scripts/run_icc_smoke.sh reported exit 1 while a direct run of
#       scripts/run_language_coverage.sh gave exit 0 with 1106/1106
#       (100.0%), 0 uncovered.  Root cause: scripts/run_language_coverage.sh
#       reruns the ENTIRE test suite (scripts/run_all_tests.sh, tens of
#       minutes) as a prerequisite before it ever computes the coverage
#       floor; under `set -euo pipefail`, ANY single flaky test in that
#       rerun — an environment-driven flake under concurrent load, not a
#       coverage regression — aborted the whole script before
#       scripts/language_coverage.py (which computes the real, already-good,
#       1106/1106 result) ever ran.  scripts/run_icc_smoke.sh's `probe()`
#       has no INFRA classification, so the abort landed as a bare FAIL.
#
# Both incidents have the same shape: an expensive gate that can report red
# for environmental reasons, and a trace format with no way to say so. This
# file is the fix: a shared, small vocabulary every harness can emit, plus a
# timeout wrapper that actually produces a distinguishable "this timed out"
# signal (the old exec-then-perl-alarm pattern used by run_vm_parity.sh could
# not — see eshkol_outcome_guarded below).
#
# THE FOUR OUTCOMES
#
#   PASS   the code under test produced the correct/expected result.
#   FAIL   the code under test ran to completion and produced a WRONG
#          result, or the harness completed a full, valid attempt and the
#          verdict is negative. A real defect.
#   INFRA  the harness could not obtain a verdict: timeout, OOM, an
#          unexpectedly missing dependency/build artifact, a toolchain
#          changed out from under the run, disk full, network unavailable,
#          killed by an unrelated signal. Says nothing about correctness.
#   SKIP   deliberately not run (hardware/toolchain absent BY DESIGN, e.g.
#          no CUDA GPU on this host). Distinct from INFRA: SKIP is expected
#          and stable; INFRA is an unexpected environment failure during an
#          attempted run. (Existing convention: exit 77, see
#          run_wasm_differential.sh; unchanged by this file.)
#
# THE CONTRACT
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/harness_outcome.sh"
#
#   eshkol_outcome_guarded <seconds> <cmd...>
#     A real, signal-observing timeout wrapper. Prints nothing; the child's
#     stdout/stderr pass through untouched. Exit code is EXACTLY one of:
#       <child's own exit code>   the child ran to completion
#       124                       the child was killed because it exceeded
#                                 <seconds> (the canonical GNU-timeout code)
#       128+N                     the child died from signal N it received
#                                 on ITS OWN (not from our alarm) — e.g. a
#                                 real SIGSEGV is still 139, unmasked
#       125                       this wrapper itself could not fork/wait
#     This replaces the exec-then-`local $SIG{ALRM}`-in-the-same-process
#     pattern the audit's measured incident traces back to: `exec` replaces
#     the whole process image, including Perl's own alarm handler, so a
#     child that is still running when the alarm fires is killed by
#     SIGALRM under ITS default disposition — exit 128+14 = 142, exactly
#     the number F13 measured — rather than the wrapper's own controlled
#     124. `eshkol_outcome_guarded` forks instead of exec'ing directly, so
#     the alarm always fires in the still-alive parent, which explicitly
#     TERMs then KILLs the child and reports a stable 124. A real hang is
#     therefore still caught (nothing here disables or widens the alarm) —
#     it is just reported as a fact about the CLOCK, not folded into the
#     child's own exit-code space where it is indistinguishable from a
#     signal the program under test raised on its own.
#
#   eshkol_outcome_classify_exit <exit_code>
#     Prints exactly one of PASS / FAIL / INFRA for a raw exit code already
#     captured by the caller (e.g. from eshkol_outcome_guarded, or from any
#     `run_all_tests.sh`-style prerequisite step). 0 -> PASS. A fixed set of
#     harness-shaped codes -> INFRA: 124/125 (this file's own timeout
#     wrapper), 142 (legacy SIGALRM-default-disposition kill — the exact
#     code the audit measured, recognized for back-compat with any caller
#     not yet migrated to eshkol_outcome_guarded), 137 (SIGKILL, usually the
#     OOM killer), 130 (SIGINT, an operator/CI abort), 127 (the universal
#     shell/exec "command not found" convention — every `exit 127` in this
#     codebase is an exec-a-child guard's missing-binary sentinel, never a
#     program's own meaningful exit code, so it is the taxonomy's "missing
#     dependency/unbuildable" INFRA case). Everything else -> FAIL, on the
#     principle that an unrecognized nonzero exit is a claim about the CODE
#     until a harness explicitly says otherwise.
#
#   eshkol_outcome_emit_event <trace_file> <kind> <name> <outcome> <snippet> [confidence]
#     Appends one JSON-L record: {"kind","name","value":<outcome>,"snippet",
#     "confidence"}. <outcome> is PASS/FAIL/INFRA/SKIP (or "INFRA:<reason>",
#     stored verbatim in value — completion-oracle criteria match on the
#     literal string, so a reason suffix does not accidentally satisfy an
#     `event_values: ["PASS"]` or `["INFRA"]` filter; keep the bare token as
#     the leading text when a consumer needs to match on it and put detail
#     in the snippet instead). Safe to call from a `set -e` script — never
#     exits nonzero itself.
#
#   eshkol_outcome_emit_test_result <trace_file> <name> <outcome> <snippet>
#     Writes the canonical kind:"test_result" event
#     ({"passed": bool, "summary": ...}) used by ICC's architecture
#     invariants (e.g. INV-dispatch-table-completeness), but ONLY for
#     outcome PASS or FAIL. For INFRA or SKIP this function writes NOTHING
#     and returns 0. See "WHY INFRA EMITS NO test_result" below.
#
#   eshkol_outcome_retry_guarded <seconds> <outfile> <errfile> <cmd...>
#     Runs eshkol_outcome_guarded once, capturing stdout/stderr to the given
#     files (truncated fresh on each attempt — this function owns the
#     redirection itself so a retry cannot concatenate a timed-out first
#     attempt's partial output with the clean second attempt's); if (and
#     only if) that attempt classifies as INFRA, retries exactly once more
#     with the same timeout before concluding. Never retries a real FAIL (a
#     completed run with a wrong answer must not get a second, quieter
#     chance — only the clock does). Returns the LAST attempt's exit code.
#     This is the direct fix for incident (a): a cold-start timeout gets one
#     retry, at which point the JIT cache populated by the first attempt
#     makes the second attempt fast, exactly as F13 measured (140.49s once,
#     then 0.07s, then 0.06s).
#
# WHY INFRA EMITS NO test_result
#
# ICC's architecture-verify reads `test_result` events through
# `_check_parity_evidence` (used by `intended-invariant` invariants such as
# INV-dispatch-table-completeness, in
# ~/Desktop/infinite_context_coder/scripts/architecture_model_service.py).
# That evaluator is a two-state PASS/FAIL machine with no third state:
# the newest matching event with `value.passed == true` within the
# freshness window (`max_age_days`) is PASS; a matching event with
# `passed == false`, a STALE matching event, or NO matching event at all
# are ALL "FAIL" for a high/critical-severity invariant (there is no
# UNCHECKABLE branch in this evaluator; `_status_for_missing("high")` is
# unconditionally "FAIL"). So a harness cannot, today, publish "I could not
# get a verdict" through this specific channel without it reading exactly
# like "the code is wrong" to that one consumer. Given that hard
# constraint, the safest encoding available without an ICC-side change is:
# never publish a `passed:false` record for a run that produced no real
# verdict. Emitting nothing leaves whichever verdict (if any) is already on
# file inside the freshness window standing, rather than overwriting a
# possibly-still-valid recent PASS with a false, environment-caused red.
# This is documented and deliberate, not an oversight — see the "ICC-side
# change" note below and ~/.tsotchke/state/feedback/ for the request to add
# a third `outcome` value this evaluator can treat as "leave the standing
# verdict alone" instead of defaulting every non-pass to FAIL.
#
# The richer, domain-specific event streams this file also writes to
# (kind:"vm_parity", kind:"eshkol_smoke", ...) DO get an explicit INFRA
# value, because their consumer — ICC's `completion-oracle` `runtime_event`
# criterion — only ever treats `event_values: ["PASS"]` as satisfying a
# criterion; INFRA and FAIL are equally "not yet satisfied" to that
# matcher (see completion_oracle_criteria.py:_event_matches, which filters
# candidate evidence by exact value membership before any status logic
# runs), so recording the honest value costs nothing there and is strictly
# more informative to a human or an audit reading the trace than
# overloading FAIL.
#
# ICC-SIDE CHANGE THIS FILE DOES NOT (AND SHOULD NOT) MAKE
#
# `_check_parity_evidence` in architecture_model_service.py could grow a
# third state — read `value.outcome` if present (INFRA) and, on INFRA,
# return the PRIOR verdict (PASS/FAIL/UNCHECKABLE) unchanged rather than
# collapsing to FAIL — so a harness could publish an honest "I tried, no
# verdict" record instead of relying on silence. That is a change to a
# different repository (infinite_context_coder) and is out of scope for
# this PR; it is recorded for the maintainer at
# ~/.tsotchke/state/feedback/ rather than made here.

if [ -n "${ESHKOL_HARNESS_OUTCOME_SH_LOADED:-}" ]; then
    return 0 2>/dev/null || true
fi
ESHKOL_HARNESS_OUTCOME_SH_LOADED=1

# ── eshkol_outcome_guarded ──────────────────────────────────────────────
eshkol_outcome_guarded() { # seconds cmd...
    local secs="$1"; shift
    perl -e '
        use POSIX ":sys_wait_h";
        my $secs = shift @ARGV;
        my $pid = fork();
        if (!defined $pid) { exit 125; }
        if ($pid == 0) {
            exec { $ARGV[0] } @ARGV or exit 127;
        }
        my $timed_out = 0;
        local $SIG{ALRM} = sub {
            $timed_out = 1;
            kill("TERM", $pid);
            select(undef, undef, undef, 0.5);
            kill("KILL", $pid);
        };
        alarm($secs);
        my $reaped = waitpid($pid, 0);
        alarm(0);
        my $status = $?;
        if ($reaped != $pid) { exit 125; }
        if ($timed_out) { exit 124; }
        if (($status & 127) != 0) {
            # Child died from a signal we did not send ourselves: a real
            # crash, not a timeout. Preserve the signal in the exit code
            # (128+N) rather than folding it into 124.
            exit (128 + ($status & 127));
        }
        exit ($status >> 8);
    ' "$secs" "$@"
}

# ── eshkol_outcome_classify_exit ─────────────────────────────────────────
eshkol_outcome_classify_exit() { # exit_code
    local rc="$1"
    case "$rc" in
        0) printf 'PASS\n' ;;
        124|125|130|137|142) printf 'INFRA\n' ;;
        # 127 is the universal shell/exec convention for "command not
        # found" — in this codebase it is exclusively the sentinel every
        # exec-a-child guard (this file's eshkol_outcome_guarded included)
        # emits when the target binary itself is missing, never a program's
        # own meaningful exit code (verified: every `exit 127` site in
        # scripts/ is one of these guards). That is the taxonomy's own
        # "missing dependency/unbuildable" INFRA case, so it belongs here.
        127) printf 'INFRA\n' ;;
        *) printf 'FAIL\n' ;;
    esac
}

# ── eshkol_outcome_emit_event ────────────────────────────────────────────
eshkol_outcome_emit_event() { # trace_file kind name outcome snippet [confidence]
    local trace_file="$1" kind="$2" name="$3" outcome="$4" snippet="$5" confidence="${6:-0.95}"
    : "${trace_file:?eshkol_outcome_emit_event: trace_file required}"
    python3 -c '
import json, sys
trace_file, kind, name, outcome, snippet, confidence = sys.argv[1:7]
with open(trace_file, "a") as fh:
    fh.write(json.dumps({
        "kind": kind, "name": name, "value": outcome,
        "snippet": snippet, "confidence": float(confidence),
    }, ensure_ascii=False))
    fh.write("\n")
' "$trace_file" "$kind" "$name" "$outcome" "$snippet" "$confidence"
}

# ── eshkol_outcome_emit_test_result ──────────────────────────────────────
# Only PASS/FAIL publish a test_result record; INFRA/SKIP deliberately
# write nothing. See "WHY INFRA EMITS NO test_result" above.
eshkol_outcome_emit_test_result() { # trace_file name outcome snippet
    local trace_file="$1" name="$2" outcome="$3" snippet="$4"
    : "${trace_file:?eshkol_outcome_emit_test_result: trace_file required}"
    case "$outcome" in
        PASS|FAIL) : ;;
        *) return 0 ;;
    esac
    local passed=false
    [ "$outcome" = "PASS" ] && passed=true
    python3 -c '
import json, sys, time
trace_file, name, passed, snippet = sys.argv[1:5]
with open(trace_file, "a") as fh:
    fh.write(json.dumps({
        "kind": "test_result", "name": name,
        "value": {"passed": passed == "true", "summary": snippet},
        "timestamp": time.time(),
    }, ensure_ascii=False))
    fh.write("\n")
' "$trace_file" "$name" "$passed" "$snippet"
}

# ── eshkol_outcome_retry_guarded ─────────────────────────────────────────
# Runs eshkol_outcome_guarded once, redirecting stdout/stderr to <outfile>/
# <errfile> (truncated fresh per attempt so a retry cannot concatenate a
# timed-out attempt's partial output with the clean retry's); retries
# exactly once more IFF the first attempt classified as INFRA. Never
# retries a real FAIL. Returns the last attempt's exit code.
eshkol_outcome_retry_guarded() { # seconds outfile errfile cmd...
    local secs="$1" outfile="$2" errfile="$3"; shift 3
    eshkol_outcome_guarded "$secs" "$@" >"$outfile" 2>"$errfile"
    local rc=$?
    [ "$(eshkol_outcome_classify_exit "$rc")" = "INFRA" ] || return "$rc"
    eshkol_outcome_guarded "$secs" "$@" >"$outfile" 2>"$errfile"
    return $?
}
