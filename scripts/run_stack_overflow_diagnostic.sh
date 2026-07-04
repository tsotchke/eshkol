#!/usr/bin/env bash
# ESH-0119: verify that a deep-recursion stack overflow is a CLEAN, diagnosable
# failure (nonzero exit + "[Eshkol] fatal signal" stack-overflow diagnostic on
# stderr) on BOTH the JIT (-r) and AOT lanes, not a silent rc132.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/build/eshkol-run"
TEST="$ROOT/tests/runtime/stack_overflow_diagnostic_test.esk"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# macOS has no timeout(1); use a perl alarm wrapper.
run_capped() {  # run_capped <seconds> <cmd...>
    local secs="$1"; shift
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' "$secs" "$@"
}

pass=0; fail=0

check() {  # check <lane-name> <stderr-file> <rc>
    local lane="$1" errf="$2" rc="$3"
    local ok=1
    if [ "$rc" -eq 0 ]; then
        echo "FAIL: $lane exited 0 (expected nonzero — the overflow should fail)"; ok=0
    fi
    if grep -qE "fatal signal|stack overflow|recursion too deep" "$errf"; then
        :
    else
        echo "FAIL: $lane produced NO stack-overflow diagnostic (silent crash)"; ok=0
    fi
    if [ "$ok" -eq 1 ]; then
        echo "PASS: $lane rc=$rc with diagnostic: $(grep -m1 -E 'fatal signal|stack overflow' "$errf")"
        pass=$((pass+1))
    else
        echo "---- $lane stderr ----"; cat "$errf"; echo "----------------------"
        fail=$((fail+1))
    fi
}

# --- JIT (-r) lane ---
run_capped 60 "$RUN" -r "$TEST" >"$TMP/jit.out" 2>"$TMP/jit.err"
check "JIT (-r)" "$TMP/jit.err" "$?"

# --- AOT lane ---
if run_capped 120 "$RUN" "$TEST" -o "$TMP/sotest" >"$TMP/aot_build.log" 2>&1; then
    run_capped 60 "$TMP/sotest" >"$TMP/aot.out" 2>"$TMP/aot.err"
    check "AOT" "$TMP/aot.err" "$?"
else
    echo "FAIL: AOT compile failed"; cat "$TMP/aot_build.log"; fail=$((fail+1))
fi

echo
echo "stack_overflow_diagnostic: pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
