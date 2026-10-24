#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${ESHKOL_BIN:-${BUILD_DIR:-build}/eshkol-run}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
VM="${VM_BIN:-$(dirname "$RUN")/eshkol-vm-standalone-test}"
LIB="${ESHKOL_NATIVE_LIB_DIR:-$(dirname "$RUN")}"
[[ -x "$RUN" && -x "$VM" && -f "$LIB/stdlib.bc" ]] || { echo 'ERROR: build runtime, VM and stdlib first' >&2; exit 2; }
source "$ROOT/scripts/lib/test_isolation.sh"
eshkol_test_isolation_init sblc-rng-parity
WORK="$ESHKOL_TEST_TMPDIR"
export ESHKOL_LIB_DIR="$LIB" ESHKOL_PATH="$ROOT/lib" ESHKOL_JIT_CACHE=0 ESHKOL_JIT_COMPILE_THREADS=1
export ESHKOL_VM_NO_DISASM=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ARGS=(-I "$ROOT/lib" -L "$LIB" -O0)
checked() {
 local log="$1"; shift
 if ! eshkol_test_timeout 240 "$@" > "$log.out" 2> "$log.err"; then tail -30 "$log.out"; tail -15 "$log.err" >&2; return 1; fi
 if grep -qE 'FAIL:|ERROR|OVERFLOW|unhandled native call' "$log.out" "$log.err"; then tail -20 "$log.out"; tail -15 "$log.err" >&2; return 1; fi
}
panel() {
 local name="$1" source="$2" verdict="$3"
 checked "$WORK/$name-jit" "$RUN" "${ARGS[@]}" -r "$source"
 checked "$WORK/$name-build" "$RUN" "${ARGS[@]}" -o "$WORK/$name" "$source"
 checked "$WORK/$name-aot" "$WORK/$name"
 checked "$WORK/$name-vm" "$VM" "$source"
 checked "$WORK/$name-emit" "$RUN" --profile hosted-vm --emit-eskb "$WORK/$name.eskb" "$source"
 checked "$WORK/$name-eskb" "$VM" "$WORK/$name.eskb"
 grep -Fxq "$verdict" "$WORK/$name-jit.out"
 for mode in aot vm eskb; do cmp "$WORK/$name-jit.out" "$WORK/$name-$mode.out"; done
}
panel prng "$ROOT/tests/vm_parity/corpus/59_isolated_prng.esk" 'ISOLATED PRNG RESULT: PASS'
grep -Fxq 'ISOLATED PRNG checks: 121' "$WORK/prng-jit.out"
panel sblc "$ROOT/tests/stdlib/sblc_rng_parity_test.esk" 'SBLC RNG RESULT: PASS'
grep -Fxq 'SBLC RNG cases: 120' "$WORK/sblc-jit.out"
grep -Fxq 'SBLC RNG API panels: 4' "$WORK/sblc-jit.out"
# VM-only invalid-domain cases must diagnose rather than dereference a scalar.
for spec in seed state bound integer-state; do
 case "$spec" in
 seed) form='(make-prng "bad")'; reason='expected an int64 seed';;
 state) form='(prng-random 7)'; reason='expected an isolated PRNG';;
 bound) form='(prng-random-integer (make-prng 7) "bad")'; reason='expected an int64 bound';;
 integer-state) form='(prng-random-integer 7 2)'; reason='expected an isolated PRNG';;
 esac
 printf '%s\n%s\n' "$form" '(display "UNREACHABLE-PRNG-SENTINEL")' > "$WORK/$spec.esk"
 checked "$WORK/$spec-emit" "$RUN" --profile hosted-vm --emit-eskb "$WORK/$spec.eskb" "$WORK/$spec.esk"
 for suffix in esk eskb; do
   if eshkol_test_timeout 30 "$VM" "$WORK/$spec.$suffix" > "$WORK/refusal.out" 2> "$WORK/refusal.err"; then
      echo "FAIL: $spec $suffix accepted" >&2; exit 1
   fi
   grep -Fq "$reason" "$WORK/refusal.err"
   if grep -Fq 'UNREACHABLE-PRNG-SENTINEL' "$WORK/refusal.out"; then exit 1; fi
 done
done
cat "$WORK/prng-jit.out"
tail -3 "$WORK/sblc-jit.out"
echo 'SBLC RNG GATE: PASS (JIT, AOT, VM source, ESKB and eight VM refusals)'
