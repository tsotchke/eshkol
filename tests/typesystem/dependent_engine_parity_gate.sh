#!/usr/bin/env bash
#
# Dependent-type engine parity (ADR-0042). The native engine refuses a
# program whose dependent types do not hold; the bytecode VM must refuse the
# same programs and run the same well-typed ones, because it reaches the
# same front end through eshkol_vm_install_dependent_check().
#
#   1. every tests/typesystem/dep_*_reject_test.esk is refused by the VM
#      (nonzero exit, the obligation diagnostic) and by --emit-eskb (nonzero
#      exit, no .eskb written);
#   2. every tests/typesystem/dep_*_ok_test.esk runs on the VM with exit 0
#      and prints exactly what the native engine (-r) prints.
#
# Usage: dependent_engine_parity_gate.sh <eshkol-run> <vm-standalone|none> <workdir>
set -u
ESHKOL_RUN="${1:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"
VM="${2:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"
WORK="${3:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"
cd "$(dirname "$0")/../.."
if [ "$VM" = "none" ] || [ ! -x "$VM" ]; then
    echo "SKIP: bytecode VM not built in this configuration"
    echo "PASS: dependent type engine parity (vacuous: no VM in this build)"
    exit 0
fi
mkdir -p "$WORK"
export ESHKOL_VM_NO_DISASM=1
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
fail=0
checked=0
for f in tests/typesystem/dep_*_reject_test.esk; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .esk)
    out=$("$VM" "$f" 2>&1); rc=$?
    if [ "$rc" -eq 0 ] || ! printf '%s' "$out" | grep -qF "[ERROR] Type error:"; then
        echo "FAIL: the VM did not refuse $name (exit $rc)"
        fail=1
    fi
    eskb="$WORK/$name.eskb"
    rm -f "$eskb"
    "$ESHKOL_RUN" "$f" --profile hosted-vm --emit-eskb "$eskb" > /dev/null 2>&1; rc=$?
    if [ "$rc" -eq 0 ] || [ -e "$eskb" ]; then
        echo "FAIL: --emit-eskb did not refuse $name (exit $rc)"
        fail=1
    fi
    checked=$((checked + 1))
done
for f in tests/typesystem/dep_*_ok_test.esk; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .esk)
    native=$("$ESHKOL_RUN" "$f" -r 2>/dev/null); nrc=$?
    vm=$("$VM" "$f" 2>/dev/null | grep -v '^=== '); vrc=${PIPESTATUS[0]}
    if [ "$nrc" -ne 0 ] || [ "$vrc" -ne 0 ] || [ "$native" != "$vm" ]; then
        echo "FAIL: $name native (exit $nrc) and VM (exit $vrc) differ"
        fail=1
    fi
    checked=$((checked + 1))
done
if [ "$checked" -lt 10 ]; then
    echo "FAIL: only $checked fixtures found"
    fail=1
fi
[ "$fail" -eq 0 ] && echo "PASS: dependent type engine parity ($checked fixtures)"
exit "$fail"
