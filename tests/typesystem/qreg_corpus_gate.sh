#!/usr/bin/env bash
# Quantum register types (ADR-0048): the compile-time corpus on both engines.
#
# Every tests/typesystem/qreg_*.esk fixture declares
#   ;; EXPECT-COMPILE: fail | ok
#   ;; EXPECT-STDERR: <text>      (each must appear in the diagnostics)
#   ;; EXPECT-NO-STDERR: <text>   (none may appear)
# and is checked
#   native:  eshkol-run <fixture> -o <bin>   -- a `fail` fixture must exit
#            nonzero, write no binary and print every EXPECT-STDERR text; an
#            `ok` fixture must compile, and its binary must run and exit 0;
#   bytecode: eshkol-run <fixture> --profile hosted-vm --emit-eskb <eskb> and
#            the standalone VM on the source -- a `fail` fixture must be
#            refused by both with the same diagnostics and no bytecode; an
#            `ok` fixture must produce bytecode.
#
# usage: qreg_corpus_gate.sh <eshkol-run> <vm-standalone|none> <workdir>
set -u

ESHKOL_RUN="${1:?usage: $0 <eshkol-run> <vm-standalone|none> <workdir>}"
VM_STANDALONE="${2:?usage: $0 <eshkol-run> <vm-standalone|none> <workdir>}"
WORKDIR="${3:?usage: $0 <eshkol-run> <vm-standalone|none> <workdir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINK_DIR="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)"
mkdir -p "$WORKDIR" || { echo "FAIL: cannot create $WORKDIR"; exit 1; }

export ESHKOL_VM_NO_DISASM=1
FAILED=0
COUNT=0
fail() { echo "FAIL: $1"; FAILED=1; }

expect_lines() { # fixture key
    sed -n "s/^;; $2: //p" "$1"
}

for fixture in "$HERE"/qreg_*.esk; do
    COUNT=$((COUNT + 1))
    name="$(basename "$fixture" .esk)"
    expect="$(expect_lines "$fixture" EXPECT-COMPILE | head -1)"
    if [ "$expect" != "fail" ] && [ "$expect" != "ok" ]; then
        fail "$name: no EXPECT-COMPILE line"
        continue
    fi
    bin="$WORKDIR/$name.bin"
    rm -f "$bin" "$bin.tmp.o"
    log="$WORKDIR/$name.native.log"
    "$ESHKOL_RUN" "$fixture" -o "$bin" -L"$LINK_DIR" > "$log" 2>&1
    rc=$?
    if [ "$expect" = "fail" ]; then
        [ "$rc" -ne 0 ] || fail "$name: native compile accepted a program the corpus rejects"
        [ ! -e "$bin" ] || fail "$name: native compile wrote a binary for a rejected program"
    else
        if [ "$rc" -ne 0 ] || [ ! -x "$bin" ]; then
            fail "$name: native compile refused a correct program (exit $rc): $(tail -5 "$log")"
        else
            run_log="$WORKDIR/$name.run.log"
            "$bin" > "$run_log" 2>&1 || fail "$name: the compiled program exited nonzero: $(tail -5 "$run_log")"
            if grep -q "Unhandled exception" "$run_log"; then fail "$name: the compiled program raised: $(tail -3 "$run_log")"; fi
            cat "$run_log"
        fi
    fi
    while IFS= read -r pat; do
        [ -n "$pat" ] || continue
        grep -qF -- "$pat" "$log" || fail "$name: native diagnostics lack: $pat"
    done < <(expect_lines "$fixture" EXPECT-STDERR)
    while IFS= read -r pat; do
        [ -n "$pat" ] || continue
        if grep -qF -- "$pat" "$log"; then fail "$name: native diagnostics contain: $pat"; fi
    done < <(expect_lines "$fixture" EXPECT-NO-STDERR)

    if [ "$VM_STANDALONE" = "none" ] || [ ! -x "$VM_STANDALONE" ]; then
        continue
    fi
    eskb="$WORKDIR/$name.eskb"
    rm -f "$eskb"
    vlog="$WORKDIR/$name.vm.log"
    "$ESHKOL_RUN" "$fixture" --profile hosted-vm --emit-eskb "$eskb" > "$vlog" 2>&1
    rc=$?
    if [ "$expect" = "fail" ]; then
        [ "$rc" -ne 0 ] || fail "$name: --emit-eskb accepted a rejected program"
        [ ! -e "$eskb" ] || fail "$name: --emit-eskb wrote bytecode for a rejected program"
        slog="$WORKDIR/$name.vmsrc.log"
        "$VM_STANDALONE" "$fixture" > "$slog" 2>&1
        src_rc=$?
        [ "$src_rc" -ne 0 ] || fail "$name: the bytecode VM ran a rejected program to completion"
        first="$(expect_lines "$fixture" EXPECT-STDERR | head -1)"
        if [ -n "$first" ]; then
            grep -qF -- "$first" "$vlog" || fail "$name: --emit-eskb diagnostics lack: $first"
            grep -qF -- "$first" "$slog" || fail "$name: VM diagnostics lack: $first"
        fi
    else
        if [ "$rc" -ne 0 ] || [ ! -s "$eskb" ]; then
            fail "$name: --emit-eskb refused a correct program (exit $rc): $(tail -5 "$vlog")"
        fi
    fi
done

[ "$COUNT" -ge 10 ] || fail "only $COUNT qreg fixtures found"
if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
echo "PASS: quantum types corpus ($COUNT fixtures, native and bytecode engines)"
exit 0
