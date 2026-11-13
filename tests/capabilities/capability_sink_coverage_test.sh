#!/usr/bin/env bash
# The capability policy at the non-file sinks, on every engine that carries
# them.
#
# For each sink class the same probe runs three ways:
#   none    no policy installed — the sink runs, exactly as it did before
#           capabilities existed
#   deny    a policy that grants file-read and nothing else — the sink is
#           refused with a catchable error naming the capability
#   grant   a policy that names the capability — the sink runs
#
# and on each engine that carries the sink: the LLVM JIT, an ahead-of-time
# executable, and the bytecode VM. `(system ...)` is a code-generated call with
# no VM binding, and `(dlopen ...)` is a VM operation with no code-generated
# form, so each of those runs on the engines that have it. On the native
# engines the ffi row is a program's own `(extern ...)` naming a libc symbol
# directly, which is the other way out of the language.
#
# The last case is the one the policy exists for: under a policy granting only
# file-read, a shell command that would write a file must leave no file behind.

set -u

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <eshkol-run> <eshkol-vm-standalone-test> <build-dir>" >&2
    exit 2
fi

ESH="$1"
VM="$2"
BUILD_DIR="$3"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_DIR="$ROOT_DIR/tests/capabilities"
WORK_DIR="$(mktemp -d "$BUILD_DIR/capability-sinks.XXXXXX")"

failures=0
checks=0

report_work_dir() {
    echo "NOTE: lane logs are under $WORK_DIR" >&2
}
trap report_work_dir EXIT

# run_engine <engine> <probe.esk> <tag> -> writes "$WORK_DIR/$tag.out"
run_engine() {
    engine="$1"
    source="$2"
    tag="$3"
    case "$engine" in
        jit)
            ESHKOL_JIT_CACHE=0 "$ESH" -r "$source" -L"$BUILD_DIR" \
                >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
            ;;
        aot)
            if ! "$ESH" "$source" -o "$WORK_DIR/$tag.bin" -L"$BUILD_DIR" \
                    >"$WORK_DIR/$tag.build" 2>&1; then
                echo "FAIL: $tag ahead-of-time compile did not succeed" >&2
                tail -40 "$WORK_DIR/$tag.build" >&2
                failures=$((failures + 1))
                return 1
            fi
            "$WORK_DIR/$tag.bin" >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
            ;;
        vm)
            ESHKOL_VM_NO_DISASM=1 "$VM" "$source" \
                >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
            ;;
    esac
    return 0
}

# expect <engine> <probe> <mode> <capability> <pattern>
#   pattern is a grep -E pattern the probe's stdout must match
expect() {
    engine="$1"; probe="$2"; mode="$3"; capability="$4"; pattern="$5"
    tag="${capability}-${engine}-${mode}"
    checks=$((checks + 1))
    if ! ESHKOL_CAP_MODE="$mode" run_engine "$engine" "$TEST_DIR/$probe" "$tag"; then
        return
    fi
    if grep -Eq "$pattern" "$WORK_DIR/$tag.out"; then
        echo "  ok  $capability/$engine/$mode"
    else
        echo "FAIL: $capability on $engine in mode $mode did not match '$pattern'" >&2
        echo "  stdout: $(cat "$WORK_DIR/$tag.out")" >&2
        tail -20 "$WORK_DIR/$tag.err" >&2
        failures=$((failures + 1))
    fi
}

# A sink class over the engines that carry it: runs with no policy, is refused
# under a policy that withholds the capability, and runs under one that grants
# it.
sink_matrix() {
    probe="$1"; capability="$2"; ran="$3"; shift 3
    for engine in "$@"; do
        expect "$engine" "$probe" none  "$capability" "$ran"
        expect "$engine" "$probe" deny  "$capability" "capability denied: $capability"
        expect "$engine" "$probe" grant "$capability" "$ran"
    done
}

echo "=== shell: (system cmd) ==="
sink_matrix capability_sink_shell_probe.esk shell 'ran:0' jit aot

echo "=== subprocess: (process-spawn cmd env) ==="
sink_matrix capability_sink_subprocess_probe.esk subprocess \
    'ran:(spawned|unavailable)' jit aot vm

echo "=== network: (http-server-create port) ==="
sink_matrix capability_sink_network_probe.esk network \
    'ran:(listening|unavailable)' jit aot vm

echo "=== ffi: (dlopen path) ==="
sink_matrix capability_sink_ffi_probe.esk ffi 'ran:(loaded|unavailable)' vm

# On the native engines a program reaches outside the language through its own
# (extern ...) declaration, so that is what the ffi row exercises there. The
# denied case also asserts that nothing happened: the command the foreign
# symbol would have run writes a marker, and the marker must not appear.
echo "=== ffi: a program's own (extern ...) on the native engines ==="
for engine in jit aot; do
    for mode in none deny grant; do
        marker="$WORK_DIR/ffi-extern-$engine-$mode.txt"
        tag="ffi-extern-$engine-$mode"
        checks=$((checks + 1))
        ESHKOL_CAP_MODE="$mode" ESHKOL_CAP_FFI_MARKER="$marker" \
            run_engine "$engine" "$TEST_DIR/capability_sink_ffi_native_probe.esk" "$tag" || continue
        if [ "$mode" = "deny" ]; then
            if grep -q "capability denied: ffi" "$WORK_DIR/$tag.out" && [ ! -e "$marker" ]; then
                echo "  ok  ffi-extern/$engine/$mode"
            else
                echo "FAIL: ffi-extern/$engine/$mode was not refused, or it had an effect" >&2
                echo "  stdout: $(cat "$WORK_DIR/$tag.out")" >&2
                [ -e "$marker" ] && echo "  the foreign call ran: marker exists" >&2
                failures=$((failures + 1))
            fi
        else
            if grep -Eq "ran:[0-9]+" "$WORK_DIR/$tag.out"; then
                echo "  ok  ffi-extern/$engine/$mode"
            else
                echo "FAIL: ffi-extern/$engine/$mode did not reach the foreign symbol" >&2
                echo "  stdout: $(cat "$WORK_DIR/$tag.out")" >&2
                tail -20 "$WORK_DIR/$tag.err" >&2
                failures=$((failures + 1))
            fi
        fi
    done
done

# The policy's own entry points are not declarable by the program it governs.
echo "=== a program cannot declare the policy's own entry points ==="
for engine in jit aot; do
    checks=$((checks + 1))
    tag="extern-policy-$engine"
    if [ "$engine" = jit ]; then
        ESHKOL_JIT_CACHE=0 "$ESH" -r "$TEST_DIR/capability_extern_policy_negative.esk" \
            -L"$BUILD_DIR" >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
    else
        "$ESH" "$TEST_DIR/capability_extern_policy_negative.esk" \
            -o "$WORK_DIR/$tag.bin" -L"$BUILD_DIR" \
            >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
    fi
    rc=$?
    if [ "$rc" -ne 0 ] && \
       grep -q "is the capability policy itself" "$WORK_DIR/$tag.err" "$WORK_DIR/$tag.out"; then
        echo "  ok  extern-policy/$engine refused at compile time"
    else
        echo "FAIL: extern-policy/$engine was accepted (exit $rc)" >&2
        tail -10 "$WORK_DIR/$tag.err" >&2
        failures=$((failures + 1))
    fi
done

echo "=== a policy granting only file-read also holds for the shell ==="
for engine in jit aot; do
    target="$WORK_DIR/escape-$engine.txt"
    checks=$((checks + 1))
    ESHKOL_CAP_ESCAPE_TARGET="$target" \
        run_engine "$engine" "$TEST_DIR/capability_shell_escape_probe.esk" \
        "escape-$engine" || continue
    out="$WORK_DIR/escape-$engine.out"
    ok=1
    grep -q "policy-active:#t" "$out" || { echo "FAIL: escape/$engine policy not active" >&2; ok=0; }
    grep -q "shell:.*capability denied: shell" "$out" || {
        echo "FAIL: escape/$engine shell command was not refused" >&2; ok=0; }
    grep -Eq "file-write:(.*capability denied: file-write|refused:write-file)" "$out" || {
        echo "FAIL: escape/$engine file write was not refused" >&2; ok=0; }
    if [ -e "$target" ]; then
        echo "FAIL: escape/$engine the refused shell command left a file behind" >&2
        ok=0
    fi
    if [ "$ok" -eq 1 ]; then
        echo "  ok  escape/$engine"
    else
        echo "  stdout: $(cat "$out")" >&2
        failures=$((failures + 1))
    fi
done

echo "checks: $checks, failures: $failures"
if [ "$failures" -ne 0 ]; then
    echo "FAIL: capability_sink_coverage_test" >&2
    exit 1
fi
echo "PASS: capability_sink_coverage_test"
