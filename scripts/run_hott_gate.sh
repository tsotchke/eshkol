#!/usr/bin/env bash
# HoTT gate (ADR-0043): every program in tests/hott/programs must print its
# .expected output through the kernel driver, the JIT, an AOT binary and the
# bytecode VM; every file in tests/hott/reject must be rejected by the kernel
# driver and by the compiler (JIT and VM routes), with the diagnostic named
# on its `;; expect-error:` line. Prints "RESULT: PASS" when all hold.
#
# usage: run_hott_gate.sh BUILD_DIR [SOURCE_DIR]
set -u
B=${1:?build dir}
S=${2:-$(cd "$(dirname "$0")/.." && pwd)}
RUN="$B/eshkol-run"
VM="$B/eshkol-vm-standalone-test"
K="$B/hott_kernel_driver"
W="$B/hott-gate"
mkdir -p "$W"
pass=0
fail=0
ok() { pass=$((pass + 1)); echo "ok   $1"; }
bad() { fail=$((fail + 1)); echo "FAIL: $1"; }

for f in "$S"/tests/hott/programs/*.esk; do
    n=$(basename "$f" .esk)
    exp="${f%.esk}.expected"
    "$K" "$f" > "$W/$n.kernel" 2> "$W/$n.kernel.err" && cmp -s "$W/$n.kernel" "$exp" \
        && ok "$n kernel" || bad "$n kernel: $(head -c 400 "$W/$n.kernel.err")"
    "$RUN" -r "$f" -L"$B" > "$W/$n.jit" 2> "$W/$n.jit.err" && cmp -s "$W/$n.jit" "$exp" \
        && ok "$n jit" || bad "$n jit: $(diff "$W/$n.jit" "$exp" | head -4) $(head -c 400 "$W/$n.jit.err")"
    if "$RUN" -O 1 -o "$W/$n.aot" "$f" -L"$B" > "$W/$n.build" 2>&1 && "$W/$n.aot" > "$W/$n.aot.out" 2>&1 \
        && cmp -s "$W/$n.aot.out" "$exp"; then ok "$n aot"; else bad "$n aot: $(head -c 400 "$W/$n.build")"; fi
    if [ -x "$VM" ]; then
        if "$RUN" --profile hosted-vm --emit-eskb "$W/$n.eskb" "$f" -L"$B" > "$W/$n.vmb" 2>&1 \
            && "$VM" "$W/$n.eskb" > "$W/$n.vm.out" 2>&1 && cmp -s "$W/$n.vm.out" "$exp"; then
            ok "$n vm"
        else
            bad "$n vm: $(head -c 400 "$W/$n.vmb") $(diff "$W/$n.vm.out" "$exp" 2>/dev/null | head -4)"
        fi
    fi
done

for f in "$S"/tests/hott/reject/*.esk; do
    n=$(basename "$f" .esk)
    want=$(sed -n 's/^;; expect-error: //p' "$f" | head -1)
    "$K" --reject "$f" > /dev/null 2> "$W/$n.kr" && ok "$n kernel rejects" || bad "$n kernel: $(head -c 300 "$W/$n.kr")"
    if "$RUN" -r "$f" -L"$B" > "$W/$n.jo" 2>&1; then
        bad "$n jit accepted a rejected program"
    elif grep -qF "$want" "$W/$n.jo"; then ok "$n jit rejects"
    else bad "$n jit: diagnostic lacks '$want': $(head -c 300 "$W/$n.jo")"; fi
    if [ -x "$VM" ]; then
        if "$RUN" --profile hosted-vm --emit-eskb "$W/$n.eskb" "$f" -L"$B" > "$W/$n.vo" 2>&1; then
            bad "$n vm emitted bytecode for a rejected program"
        elif grep -qF "$want" "$W/$n.vo"; then ok "$n vm rejects"
        else bad "$n vm: diagnostic lacks '$want': $(head -c 300 "$W/$n.vo")"; fi
    fi
done

echo "hott gate: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
[ "$fail" -eq 0 ]
