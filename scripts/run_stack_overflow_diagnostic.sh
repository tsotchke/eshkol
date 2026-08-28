#!/usr/bin/env bash
# ESH-0101: verify the native stack contract on JIT, AOT, and parallel workers:
# the default 512 MiB ESHKOL_STACK_SIZE reports a clean failure, while a 1 GiB
# stack completes the same 2M-frame program. Worker stacks use the separate
# ESHKOL_WORKER_STACK_BYTES setting and must receive the same signal backstop.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/build/eshkol-run"
TEST="$ROOT/tests/stress/found/deep_recursion_270k_no_diagnostic.esk"
WORKER_TEST="$ROOT/tests/runtime/parallel_stack_overflow_diagnostic_test.esk"
SCRATCH="$ROOT/.scratch/stack-overflow-diagnostic.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

# The initial thread's stack extent is fixed at exec time on Linux. The
# ESHKOL_STACK_SIZE=1G completion leg therefore requires a generous inherited
# soft limit; this is the documented invocation for this gate.
if ! ulimit -s 1048576 2>/dev/null; then
    echo "FAIL: could not raise the shell stack limit to 1 GiB" >&2
    exit 1
fi

# macOS has no timeout(1); use a perl alarm wrapper.
run_capped() {  # run_capped <seconds> <cmd...>
    local secs="$1"; shift
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' "$secs" "$@"
}

pass=0; fail=0

check_diag() {  # check_diag <lane-name> <stderr-file> <rc>
    local lane="$1" errf="$2" rc="$3"
    local ok=1
    if [ "$rc" -eq 0 ]; then
        echo "FAIL: $lane exited 0 (expected the default stack guard to fail)"; ok=0
    fi
    if grep -qE '^eshkol: stack overflow: recursion depth exceeded the [0-9]+ MiB stack \((ESHKOL_STACK_SIZE|ESHKOL_WORKER_STACK_BYTES)\)' "$errf"; then
        :
    else
        echo "FAIL: $lane produced no exact stack-overflow diagnostic"; ok=0
    fi
    if [ "$ok" -eq 1 ]; then
        echo "PASS: $lane rc=$rc with diagnostic: $(grep -m1 -E '^eshkol: stack overflow:' "$errf")"
        pass=$((pass+1))
    else
        echo "---- $lane stderr ----"; cat "$errf"; echo "----------------------"
        fail=$((fail+1))
    fi
}

check_complete() {  # check_complete <lane-name> <stdout-file> <stderr-file> <rc> <marker>
    local lane="$1" outf="$2" errf="$3" rc="$4" marker="$5"
    if [ "$rc" -eq 0 ] && grep -qF "$marker" "$outf"; then
        echo "PASS: $lane completed with $marker"
        pass=$((pass+1))
    else
        echo "FAIL: $lane did not complete with $marker (rc=$rc)"
        cat "$outf" "$errf"
        fail=$((fail+1))
    fi
}

unset ESHKOL_STACK_SIZE ESHKOL_WORKER_STACK_BYTES ESHKOL_PARALLEL_NO_WARMUP

# --- Main-thread JIT (-r): default stack must fail loudly. ---
run_capped 120 "$RUN" -r "$TEST" >"$SCRATCH/main-default-jit.out" 2>"$SCRATCH/main-default-jit.err"
check_diag "main JIT default" "$SCRATCH/main-default-jit.err" "$?"

# --- Main-thread JIT (-r): 1 GiB must complete the same source. ---
ESHKOL_STACK_SIZE=1G run_capped 180 "$RUN" -r "$TEST" >"$SCRATCH/main-large-jit.out" 2>"$SCRATCH/main-large-jit.err"
check_complete "main JIT ESHKOL_STACK_SIZE=1G" "$SCRATCH/main-large-jit.out" "$SCRATCH/main-large-jit.err" "$?" "OK 2000000"

# --- Main-thread AOT: compile once, run at both stack settings. ---
if run_capped 180 "$RUN" "$TEST" -o "$SCRATCH/main-aot" >"$SCRATCH/main-aot-build.log" 2>&1; then
    unset ESHKOL_STACK_SIZE
    run_capped 120 "$SCRATCH/main-aot" >"$SCRATCH/main-default-aot.out" 2>"$SCRATCH/main-default-aot.err"
    check_diag "main AOT default" "$SCRATCH/main-default-aot.err" "$?"

    ESHKOL_STACK_SIZE=1G run_capped 180 "$SCRATCH/main-aot" >"$SCRATCH/main-large-aot.out" 2>"$SCRATCH/main-large-aot.err"
    check_complete "main AOT ESHKOL_STACK_SIZE=1G" "$SCRATCH/main-large-aot.out" "$SCRATCH/main-large-aot.err" "$?" "OK 2000000"
else
    echo "FAIL: main AOT compile failed"; cat "$SCRATCH/main-aot-build.log"; fail=$((fail+1))
fi

# --- Worker JIT/AOT: the per-thread altstack must make the default worker
# stack failure diagnosable, and a 1 GiB worker stack must complete. ---
ESHKOL_WORKER_STACK_BYTES=16M ESHKOL_PARALLEL_NO_WARMUP=1 \
    run_capped 120 "$RUN" -r "$WORKER_TEST" >"$SCRATCH/worker-default-jit.out" 2>"$SCRATCH/worker-default-jit.err"
check_diag "parallel worker JIT default" "$SCRATCH/worker-default-jit.err" "$?"

ESHKOL_WORKER_STACK_BYTES=1G ESHKOL_PARALLEL_NO_WARMUP=1 \
    run_capped 180 "$RUN" -r "$WORKER_TEST" >"$SCRATCH/worker-large-jit.out" 2>"$SCRATCH/worker-large-jit.err"
check_complete "parallel worker JIT ESHKOL_WORKER_STACK_BYTES=1G" "$SCRATCH/worker-large-jit.out" "$SCRATCH/worker-large-jit.err" "$?" "OK 4"

if run_capped 180 "$RUN" "$WORKER_TEST" -o "$SCRATCH/worker-aot" >"$SCRATCH/worker-aot-build.log" 2>&1; then
    ESHKOL_WORKER_STACK_BYTES=16M ESHKOL_PARALLEL_NO_WARMUP=1 \
        run_capped 120 "$SCRATCH/worker-aot" >"$SCRATCH/worker-default-aot.out" 2>"$SCRATCH/worker-default-aot.err"
    check_diag "parallel worker AOT default" "$SCRATCH/worker-default-aot.err" "$?"

    ESHKOL_WORKER_STACK_BYTES=1G ESHKOL_PARALLEL_NO_WARMUP=1 \
        run_capped 180 "$SCRATCH/worker-aot" >"$SCRATCH/worker-large-aot.out" 2>"$SCRATCH/worker-large-aot.err"
    check_complete "parallel worker AOT ESHKOL_WORKER_STACK_BYTES=1G" "$SCRATCH/worker-large-aot.out" "$SCRATCH/worker-large-aot.err" "$?" "OK 4"
else
    echo "FAIL: parallel worker AOT compile failed"; cat "$SCRATCH/worker-aot-build.log"; fail=$((fail+1))
fi

echo
echo "stack_overflow_diagnostic: pass=$pass fail=$fail"

TRACE_DIR="$ROOT/scripts/icc_traces"
mkdir -p "$TRACE_DIR"
if [ "$fail" -eq 0 ]; then
    verdict=PASS
else
    verdict=FAIL
fi
printf '{"kind":"runtime_evidence","name":"stack_overflow_diagnostic","value":"%s","confidence":1.0}\n' "$verdict" > "$TRACE_DIR/stack_overflow_diagnostic.jsonl"
[ "$fail" -eq 0 ]
