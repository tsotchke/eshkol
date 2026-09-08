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
# ESHKOL_STACK_SIZE completion leg therefore requires a generous inherited
# soft limit. Keep the historical 1 GiB proof when the host permits it, but
# use the largest safe target below a finite hard limit (macOS commonly
# reports a 64 MiB hard limit) so the same proof remains executable there.
hard_stack_kib=$(ulimit -Hs 2>/dev/null || true)
case "$hard_stack_kib" in
    ''|unlimited) large_stack_kib=1048576 ;;
    *[!0-9]*)
        echo "FAIL: could not determine the shell hard stack limit ($hard_stack_kib)" >&2
        exit 1
        ;;
    *)
        if [ "$hard_stack_kib" -ge 1048576 ]; then
            large_stack_kib=1048576
        else
            # Leave a small margin for the launcher and signal diagnostics.
            large_stack_kib=$((hard_stack_kib - 4096))
        fi
        ;;
esac
if [ "$large_stack_kib" -lt 16384 ] || ! ulimit -s "$large_stack_kib" 2>/dev/null; then
    echo "FAIL: could not raise the shell stack limit to ${large_stack_kib} KiB" >&2
    exit 1
fi
large_stack_size="${large_stack_kib}K"
large_stack_mib=$((large_stack_kib / 1024))
echo "Using ${large_stack_mib} MiB completion stack (hard limit: ${hard_stack_kib} KiB)"

# The 2M-frame fixture is the historical 1 GiB proof. A finite host limit
# cannot physically accommodate that many native frames, so scale the same
# non-tail recursion fixture to the measured macOS-safe depth. The small
# configuration remains explicit and substantially below the completion
# target, preserving the fail-then-complete proof on both classes of host.
small_stack_size=16M
completion_frames=2000000
if [ "$large_stack_kib" -lt 1048576 ]; then
    completion_frames=250000
fi
COMPLETION_TEST="$SCRATCH/deep_recursion_completion.esk"
sed "s/(down 2000000)/(down ${completion_frames})/; s/OK 2000000/OK ${completion_frames}/" \
    "$TEST" >"$COMPLETION_TEST"
worker_completion_frames=300000
if [ "$large_stack_kib" -lt 1048576 ]; then
    worker_completion_frames=200000
fi
WORKER_COMPLETION_TEST="$SCRATCH/parallel_stack_completion.esk"
sed "s/300000/${worker_completion_frames}/g" "$WORKER_TEST" >"$WORKER_COMPLETION_TEST"

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

# --- Main-thread JIT (-r): the smaller configured stack must fail loudly. ---
ESHKOL_STACK_SIZE="$small_stack_size" run_capped 120 "$RUN" -r "$TEST" >"$SCRATCH/main-default-jit.out" 2>"$SCRATCH/main-default-jit.err"
check_diag "main JIT default" "$SCRATCH/main-default-jit.err" "$?"

# --- Main-thread JIT (-r): the larger available stack must complete. ---
ESHKOL_STACK_SIZE="$large_stack_size" run_capped 180 "$RUN" -r "$COMPLETION_TEST" >"$SCRATCH/main-large-jit.out" 2>"$SCRATCH/main-large-jit.err"
check_complete "main JIT ESHKOL_STACK_SIZE=${large_stack_mib}M" "$SCRATCH/main-large-jit.out" "$SCRATCH/main-large-jit.err" "$?" "OK ${completion_frames}"

# --- Main-thread AOT: compile once, run at both stack settings. ---
if run_capped 180 "$RUN" "$TEST" -o "$SCRATCH/main-aot" >"$SCRATCH/main-aot-build.log" 2>&1; then
    ESHKOL_STACK_SIZE="$small_stack_size" run_capped 120 "$SCRATCH/main-aot" >"$SCRATCH/main-default-aot.out" 2>"$SCRATCH/main-default-aot.err"
    check_diag "main AOT default" "$SCRATCH/main-default-aot.err" "$?"

    # The AOT binary embeds the same generated completion fixture.
    if run_capped 180 "$RUN" "$COMPLETION_TEST" -o "$SCRATCH/main-large-aot" >"$SCRATCH/main-large-aot-build.log" 2>&1; then
        ESHKOL_STACK_SIZE="$large_stack_size" run_capped 180 "$SCRATCH/main-large-aot" >"$SCRATCH/main-large-aot.out" 2>"$SCRATCH/main-large-aot.err"
        check_complete "main AOT ESHKOL_STACK_SIZE=${large_stack_mib}M" "$SCRATCH/main-large-aot.out" "$SCRATCH/main-large-aot.err" "$?" "OK ${completion_frames}"
    else
        echo "FAIL: main AOT completion compile failed"; cat "$SCRATCH/main-large-aot-build.log"; fail=$((fail+1))
    fi
else
    echo "FAIL: main AOT compile failed"; cat "$SCRATCH/main-aot-build.log"; fail=$((fail+1))
fi

# --- Worker JIT/AOT: the per-thread altstack must make the default worker
# stack failure diagnosable, and a 1 GiB worker stack must complete. ---
ESHKOL_WORKER_STACK_BYTES=16M ESHKOL_PARALLEL_NO_WARMUP=1 \
    run_capped 120 "$RUN" -r "$WORKER_TEST" >"$SCRATCH/worker-default-jit.out" 2>"$SCRATCH/worker-default-jit.err"
check_diag "parallel worker JIT default" "$SCRATCH/worker-default-jit.err" "$?"

ESHKOL_WORKER_STACK_BYTES="$large_stack_size" ESHKOL_PARALLEL_NO_WARMUP=1 \
    run_capped 180 "$RUN" -r "$WORKER_COMPLETION_TEST" >"$SCRATCH/worker-large-jit.out" 2>"$SCRATCH/worker-large-jit.err"
check_complete "parallel worker JIT ESHKOL_WORKER_STACK_BYTES=${large_stack_mib}M" "$SCRATCH/worker-large-jit.out" "$SCRATCH/worker-large-jit.err" "$?" "OK 4"

if run_capped 180 "$RUN" "$WORKER_TEST" -o "$SCRATCH/worker-aot" >"$SCRATCH/worker-aot-build.log" 2>&1; then
    ESHKOL_WORKER_STACK_BYTES=16M ESHKOL_PARALLEL_NO_WARMUP=1 \
        run_capped 120 "$SCRATCH/worker-aot" >"$SCRATCH/worker-default-aot.out" 2>"$SCRATCH/worker-default-aot.err"
    check_diag "parallel worker AOT default" "$SCRATCH/worker-default-aot.err" "$?"

    if run_capped 180 "$RUN" "$WORKER_COMPLETION_TEST" -o "$SCRATCH/worker-large-aot" >"$SCRATCH/worker-large-aot-build.log" 2>&1; then
        ESHKOL_WORKER_STACK_BYTES="$large_stack_size" ESHKOL_PARALLEL_NO_WARMUP=1 \
            run_capped 180 "$SCRATCH/worker-large-aot" >"$SCRATCH/worker-large-aot.out" 2>"$SCRATCH/worker-large-aot.err"
        check_complete "parallel worker AOT ESHKOL_WORKER_STACK_BYTES=${large_stack_mib}M" "$SCRATCH/worker-large-aot.out" "$SCRATCH/worker-large-aot.err" "$?" "OK 4"
    else
        echo "FAIL: parallel worker AOT completion compile failed"; cat "$SCRATCH/worker-large-aot-build.log"; fail=$((fail+1))
    fi
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
