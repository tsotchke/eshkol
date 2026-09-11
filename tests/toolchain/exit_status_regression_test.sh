#!/usr/bin/env bash
# LE-21 — (exit <computed integer>) failed LLVM module verification, on both
# the native JIT (-r) and AOT, while (exit 3) with a literal worked.
#
# Root cause: SystemCodegen::exitProgram() read its argument through the
# typed-AST fast path, which only produces a raw, unboxed LLVM value (double /
# i64 / i1) for an operand it can type at compile time — a literal. Anything
# computed at runtime (e.g. `(vector-length w)`) falls through that fast
# path's generic call arm still carrying the full boxed
# `eshkol_tagged_value_t` struct, and the struct was passed straight into
# libc `exit`'s i32 parameter, producing:
#
#   LLVM module verification failed: Call parameter type does not match
#   function signature!
#     %27 = insertvalue %eshkol_tagged_value { ... }, i64 %length, 4
#    i32  call void @exit(%eshkol_tagged_value %27)
#
# on BOTH engines that go through LLVM (JIT and AOT); the bytecode VM was
# never affected (its `exit` builtin, case 2097 in vm_native.c, dispatches on
# the runtime Value tag dynamically, no static IR to verify).
#
# The fix (SystemCodegen::unpackExitCode) dispatches on the boxed value's
# runtime type tag exactly like every other polymorphic numeric builtin
# (ArithmeticCodegen::abs is the model): a double is clamped to [0, 255] and
# truncated, an int64 is truncated, a boolean follows R7RS 6.11 (#t => 0,
# #f => 1, see docs/COMPLETE_LANGUAGE_SPECIFICATION.md section 4.14.2), and
# any other runtime type raises a catchable runtime error instead of feeding
# an arbitrary bit pattern to the process exit status. vm_native.c case 2097
# was brought in line with the same three rules (it previously answered exit
# code 0 for EITHER boolean, and silently answered 0 for any other type via
# as_number_vm()'s fallback).
#
# This script asserts the exact process exit status on the native JIT, the
# native AOT binary, and the standalone VM, for every documented shape:
# literal integer, computed integer, computed flonum, #t, #f, and a
# non-numeric argument (which must now raise rather than silently pick 0).

set -uo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
ESHKOL_VM="${2:-${ESHKOL_VM:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    else
        echo "FAIL: exit_status_regression_test could not locate eshkol-run" >&2
        exit 1
    fi
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: exit_status_regression_test eshkol-run is not executable: $ESHKOL_RUN" >&2
    exit 1
fi
if [ -z "$ESHKOL_VM" ] && [ -x "./build/eshkol-vm-standalone-test" ]; then
    ESHKOL_VM="./build/eshkol-vm-standalone-test"
fi

tmp="$(mktemp -d)"
trap 'chmod -R u+rwx "$tmp" 2>/dev/null; rm -rf "$tmp"' EXIT
# Isolate the persistent -r JIT cache so a prior run cannot serve a stale
# (pre-fix) compiled module for one of these sources.
export ESHKOL_JIT_CACHE_DIR="$tmp/jit"; mkdir -p "$ESHKOL_JIT_CACHE_DIR"
export ESHKOL_VM_NO_DISASM=1

fail() { echo "FAIL: exit_status_regression_test — $1" >&2; exit 1; }

# assert_exit_native <label> <source.esk> <expected-status> [stderr-substring]
# Compiles and runs <source.esk> under -r (JIT) and as an AOT binary, and
# requires BOTH to terminate with <expected-status>. A LOUD LLVM verification
# failure (LE-21's actual symptom) shows up here as a JIT/AOT exit status of
# 1 with "module verification failed" on stderr, not as the expected code.
assert_exit_native() {
    local label="$1" src="$2" expected="$3" want_stderr="${4:-}"

    "$ESHKOL_RUN" -r "$src" >"$tmp/jit.out" 2>"$tmp/jit.err"
    local jit_ec=$?
    if [ "$jit_ec" -ne "$expected" ]; then
        echo "--- JIT stderr ($label) ---" >&2; cat "$tmp/jit.err" >&2
        fail "$label: JIT (-r) exited $jit_ec, expected $expected"
    fi
    if [ -n "$want_stderr" ] && ! grep -qF "$want_stderr" "$tmp/jit.err"; then
        echo "--- JIT stderr ($label) ---" >&2; cat "$tmp/jit.err" >&2
        fail "$label: JIT stderr missing expected text: $want_stderr"
    fi
    echo "  ok: $label JIT (-r) exited $jit_ec"

    local bin="$tmp/aot_$label"
    rm -f "$bin"
    "$ESHKOL_RUN" "$src" -o "$bin" >"$tmp/aot_compile.out" 2>"$tmp/aot_compile.err"
    local compile_ec=$?
    [ "$compile_ec" -eq 0 ] || { cat "$tmp/aot_compile.err" >&2; fail "$label: AOT compile exited $compile_ec"; }
    [ -x "$bin" ] || fail "$label: AOT compile wrote no executable"

    "$bin" >"$tmp/aot.out" 2>"$tmp/aot.err"
    local aot_ec=$?
    if [ "$aot_ec" -ne "$expected" ]; then
        echo "--- AOT stderr ($label) ---" >&2; cat "$tmp/aot.err" >&2
        fail "$label: AOT exited $aot_ec, expected $expected"
    fi
    if [ -n "$want_stderr" ] && ! grep -qF "$want_stderr" "$tmp/aot.err"; then
        echo "--- AOT stderr ($label) ---" >&2; cat "$tmp/aot.err" >&2
        fail "$label: AOT stderr missing expected text: $want_stderr"
    fi
    echo "  ok: $label AOT exited $aot_ec"
}

# assert_exit_vm <label> <source.esk> <expected-status> [stderr-substring]
assert_exit_vm() {
    local label="$1" src="$2" expected="$3" want_stderr="${4:-}"
    [ -n "$ESHKOL_VM" ] || { echo "  skip: $label VM (eshkol-vm-standalone-test not found)"; return; }

    "$ESHKOL_VM" "$src" >"$tmp/vm.out" 2>"$tmp/vm.err"
    local vm_ec=$?
    if [ "$vm_ec" -ne "$expected" ]; then
        echo "--- VM stderr ($label) ---" >&2; cat "$tmp/vm.err" >&2
        fail "$label: VM exited $vm_ec, expected $expected"
    fi
    if [ -n "$want_stderr" ] && ! grep -qF "$want_stderr" "$tmp/vm.err"; then
        echo "--- VM stderr ($label) ---" >&2; cat "$tmp/vm.err" >&2
        fail "$label: VM stderr missing expected text: $want_stderr"
    fi
    echo "  ok: $label VM exited $vm_ec"
}

# ── fixtures ──────────────────────────────────────────────────────────────
literal_int="$tmp/literal_int.esk"
printf '(exit 3)\n' > "$literal_int"

# The LE-21 reproducer: exit with a value the compiler cannot fold to a
# constant, forcing it through the boxed (tagged) representation.
computed_int="$tmp/computed_int.esk"
printf '(define w (make-vector 4 0))\n(exit (vector-length w))\n' > "$computed_int"

# A computed flonum exit code: the boxed-double arm of unpackExitCode().
computed_double="$tmp/computed_double.esk"
printf '(define lst (list 1.0 2.0 3.0 4.0 5.0))\n(exit (/ (apply + lst) 2.5))\n' > "$computed_double"

# Booleans: R7RS 6.11. Computed (not literal) so this also exercises the
# boxed-bool arm, not just the typed-AST i1 fast path.
bool_true="$tmp/bool_true.esk"
printf '(define w (make-vector 4 0))\n(exit (= (vector-length w) 4))\n' > "$bool_true"

bool_false="$tmp/bool_false.esk"
printf '(define w (make-vector 4 0))\n(exit (= (vector-length w) 99))\n' > "$bool_false"

# Non-integer, non-boolean, non-flonum: must raise a catchable runtime error
# rather than silently picking a status (native's old fallback fed the raw
# struct/pointer into exit(); the VM's old fallback answered 0 via
# as_number_vm()'s default case).
bad_type="$tmp/bad_type.esk"
printf '(exit "not-a-status")\n' > "$bad_type"

# ── native (JIT + AOT) ───────────────────────────────────────────────────
assert_exit_native "literal_int"    "$literal_int"    3
assert_exit_native "computed_int"   "$computed_int"    4
assert_exit_native "computed_double" "$computed_double" 6
assert_exit_native "bool_true"      "$bool_true"       0
assert_exit_native "bool_false"     "$bool_false"      1
assert_exit_native "bad_type"       "$bad_type"        1 "exit: exit code must be an integer, flonum, or boolean"

# ── VM (bytecode interpreter) ───────────────────────────────────────────
assert_exit_vm "literal_int"     "$literal_int"     3
assert_exit_vm "computed_int"    "$computed_int"     4
assert_exit_vm "computed_double" "$computed_double"  6
assert_exit_vm "bool_true"       "$bool_true"        0
assert_exit_vm "bool_false"      "$bool_false"       1
assert_exit_vm "bad_type"        "$bad_type"         1 "exit: exit code must be an integer, flonum, or boolean"

echo "PASS: exit_status_regression_test"
exit 0
