#!/usr/bin/env bash
# What a program does at exit, with no capability policy anywhere in sight.
#
# A program that ends while a child it started is still running leaves that
# child running — that is what a program that daemonizes, or starts a server and
# returns, depends on. And the exit itself does not wait for a child: a child
# that ignores SIGTERM and sleeps for half a minute does not hold the program
# open.
#
# Both halves run on every engine, because the runtime's child bookkeeping is
# the native engines' and the bytecode VM has its own; they must agree.
#
# A pseudo-terminal child is a different case and is not asserted to survive: a
# terminal hangs up its session when the master descriptor goes away, and that
# happens when the program ends whether or not the runtime closes it, so the
# child receives SIGHUP from the terminal rather than anything the runtime
# sends. What is asserted for it is the half that is ours — that the exit does
# not wait.

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
WORK_DIR="$(mktemp -d "$BUILD_DIR/process-exit-child.XXXXXX")"

failures=0

# The children. A script spawned by path with no arguments is the one spelling
# that means the same thing to the native engines, which hand a command string
# to /bin/sh, and to the bytecode VM, which execs it directly.
make_survivor_script() {   # <path> <marker>
    cat > "$1" <<SCRIPT
#!/bin/sh
sleep 3
printf child-survived > '$2'
SCRIPT
    chmod +x "$1"
}

make_stubborn_script() {   # <path>
    cat > "$1" <<'SCRIPT'
#!/bin/sh
trap '' TERM
sleep 25
SCRIPT
    chmod +x "$1"
}

report_work_dir() {
    echo "NOTE: lane logs are under $WORK_DIR" >&2
}
trap report_work_dir EXIT

# run <engine> <probe> <tag> <seconds> ; exports must already be set
run_engine() {
    engine="$1"; probe="$2"; tag="$3"; limit="$4"
    case "$engine" in
        jit)
            ESHKOL_JIT_CACHE=0 timeout "$limit" "$ESH" -r "$TEST_DIR/$probe" -L"$BUILD_DIR" \
                >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
            ;;
        aot)
            if [ ! -x "$WORK_DIR/$tag.bin" ]; then
                "$ESH" "$TEST_DIR/$probe" -o "$WORK_DIR/$tag.bin" -L"$BUILD_DIR" \
                    >"$WORK_DIR/$tag.build" 2>&1 || return 2
            fi
            timeout "$limit" "$WORK_DIR/$tag.bin" \
                >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
            ;;
        vm)
            ESHKOL_VM_NO_DISASM=1 timeout "$limit" "$VM" "$TEST_DIR/$probe" \
                >"$WORK_DIR/$tag.out" 2>"$WORK_DIR/$tag.err"
            ;;
    esac
}

# A child started before the program ends is still there afterwards.
check_survives() {
    engine="$1"; probe="$2"; label="$3"
    marker="$WORK_DIR/marker-$label-$engine.txt"
    tag="survives-$label-$engine"
    script="$WORK_DIR/survivor-$label-$engine.sh"
    make_survivor_script "$script" "$marker"
    ESHKOL_EXIT_CHILD_SCRIPT="$script" \
        run_engine "$engine" "$probe" "$tag" 30
    rc=$?
    if [ "$rc" -eq 2 ]; then
        echo "FAIL: $tag ahead-of-time compile did not succeed" >&2
        tail -30 "$WORK_DIR/$tag.build" >&2
        failures=$((failures + 1))
        return
    fi
    if ! grep -q "spawned:yes" "$WORK_DIR/$tag.out" 2>/dev/null; then
        echo "FAIL: $tag did not spawn a child" >&2
        echo "  stdout: $(cat "$WORK_DIR/$tag.out" 2>/dev/null)" >&2
        tail -20 "$WORK_DIR/$tag.err" >&2
        failures=$((failures + 1))
        return
    fi
    # The program has returned; the child writes its marker after that.
    waited=0
    while [ "$waited" -lt 20 ] && [ ! -s "$marker" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if [ -s "$marker" ]; then
        echo "  ok  $label/$engine child outlived the program (after ${waited}s)"
    else
        echo "FAIL: $label/$engine the child did not outlive the program" >&2
        failures=$((failures + 1))
    fi
}

# A child that refuses to stop does not hold the program open.
check_no_hang() {
    engine="$1"
    tag="stubborn-$engine"
    script="$WORK_DIR/stubborn-$engine.sh"
    make_stubborn_script "$script"
    started=$(date +%s)
    ESHKOL_EXIT_CHILD_SCRIPT="$script" \
        run_engine "$engine" process_exit_child_probe.esk "$tag" 20
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if [ "$rc" -eq 2 ]; then
        echo "FAIL: $tag ahead-of-time compile did not succeed" >&2
        tail -30 "$WORK_DIR/$tag.build" >&2
        failures=$((failures + 1))
        return
    fi
    if [ "$rc" -eq 124 ]; then
        echo "FAIL: stubborn/$engine the program did not exit within 20s" >&2
        failures=$((failures + 1))
        return
    fi
    if [ "$elapsed" -gt 12 ]; then
        echo "FAIL: stubborn/$engine exit took ${elapsed}s; it waited for the child" >&2
        failures=$((failures + 1))
        return
    fi
    echo "  ok  stubborn/$engine exited in ${elapsed}s without waiting for the child"
}

echo "=== a child started before the program ends keeps running ==="
for engine in jit aot vm; do
    check_survives "$engine" process_exit_child_probe.esk plain
done
echo "=== a child that ignores SIGTERM does not hold the program open ==="
for engine in jit aot vm; do
    check_no_hang "$engine"
done

# The same for a pseudo-terminal child: its descriptor is the program's, and
# handing it back is not a reason for the exit to wait on the child.
echo "=== a pseudo-terminal child does not hold the program open either ==="
for engine in jit aot vm; do
    tag="pty-exit-$engine"
    checks_started=$(date +%s)
    script="$WORK_DIR/pty-child-$engine.sh"
    make_survivor_script "$script" "$WORK_DIR/pty-marker-$engine.txt"
    ESHKOL_EXIT_CHILD_SCRIPT="$script" \
        run_engine "$engine" process_exit_child_probe_pty.esk "$tag" 30
    rc=$?
    elapsed=$(( $(date +%s) - checks_started ))
    if [ "$rc" -eq 2 ]; then
        echo "FAIL: $tag ahead-of-time compile did not succeed" >&2
        failures=$((failures + 1))
    elif [ "$rc" -eq 124 ]; then
        echo "FAIL: pty-exit/$engine did not exit within 30s" >&2
        failures=$((failures + 1))
    elif ! grep -q "spawned:yes" "$WORK_DIR/$tag.out" 2>/dev/null; then
        echo "FAIL: pty-exit/$engine did not spawn a pseudo-terminal child" >&2
        echo "  stdout: $(cat "$WORK_DIR/$tag.out" 2>/dev/null)" >&2
        failures=$((failures + 1))
    elif [ "$elapsed" -gt 12 ]; then
        echo "FAIL: pty-exit/$engine exit took ${elapsed}s; it waited for the child" >&2
        failures=$((failures + 1))
    else
        echo "  ok  pty-exit/$engine exited in ${elapsed}s"
    fi
done

if [ "$failures" -ne 0 ]; then
    echo "FAIL: process_exit_child_test" >&2
    exit 1
fi
echo "PASS: process_exit_child_test"
