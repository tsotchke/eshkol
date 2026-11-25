#!/usr/bin/env bash
# Compare a release regression on all three engines. The calling harness owns
# the fresh output directory and its retention policy.
set -u
if [ "$#" -ne 4 ]; then
    echo "usage: $0 SOURCE RUNNER VM OUTPUT_DIR" >&2
    exit 2
fi
source=$1
run=$2
vm=$3
work=$4
if [ ! -x "$vm" ]; then
    echo "Missing VM executable: $vm" >&2
    exit 127
fi
ESHKOL_JIT_CACHE=0 "$run" -r "$source" >"$work/jit.out" || exit $?
"$run" "$source" -o "$work/aot.bin" || exit $?
"$work/aot.bin" >"$work/aot.out" || exit $?
ESHKOL_VM_NO_DISASM=1 "$vm" "$source" >"$work/vm.out" || exit $?
for engine in jit aot vm; do
    if ! grep -Fxq "ALL PRINTER ROUND-TRIP CHECKS PASSED" "$work/$engine.out" ||
       grep -Fq "error:" "$work/$engine.out"; then
        cat "$work/$engine.out" >&2
        echo "$engine printer regression did not pass" >&2
        exit 1
    fi
done
cmp "$work/jit.out" "$work/aot.out" && cmp "$work/jit.out" "$work/vm.out"
