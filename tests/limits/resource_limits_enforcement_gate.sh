#!/usr/bin/env bash
# SW-10 gate: the documented resource-limit environment variables are ENFORCED.
#
# Every variable in docs/reference/runtime/environment-variables.md's
# "Resource limits" table used to be parsed into the active configuration and
# then consulted by nothing: `ESHKOL_MAX_HEAP=1` ran a 20M-iteration allocating
# loop to completion and exited 0, and `ESHKOL_TIMEOUT_MS=500` printed
# "Execution timeout: 500ms limit exceeded" and then also ran to completion and
# exited 0. This gate pins the three properties that fix has to keep:
#
#   1. a violation terminates with the documented per-limit exit status
#      (ESHKOL_EXIT_LIMIT_* in inc/eshkol/core/resource_limits.h) and says so
#      on stderr, naming the variable that set the ceiling;
#   2. the same program under a limit it does not reach is completely
#      unaffected — same output, exit 0 — so enforcement cannot be mistaken for
#      a program that merely got slower or noisier;
#   3. ESHKOL_ENFORCE_LIMITS=false downgrades a breach to a warning and lets the
#      program finish, which is what "hard limit violations terminate the
#      process; when false, errors are returned" means at the process boundary.
#
# Usage: resource_limits_enforcement_gate.sh <eshkol-run> <vm-standalone> <workdir>

set -u

ESHKOL_RUN="${1:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"
VM_RUN="${2:?missing vm standalone binary}"
WORK="${3:?missing work dir}"

mkdir -p "$WORK" || exit 1

PASS=0
FAIL=0

# Keep the environment clean between cases: a leaked ceiling from one case
# would silently change the meaning of the next.
unset_all_limits() {
    unset ESHKOL_MAX_HEAP ESHKOL_TIMEOUT_MS ESHKOL_MAX_STACK \
          ESHKOL_MAX_TENSOR_ELEMS ESHKOL_MAX_STRING_LEN \
          ESHKOL_ENFORCE_LIMITS ESHKOL_LIMIT_WARNINGS ESHKOL_VM_MAX_INSN
}

# check <label> <expected-exit> <expected-stderr-substring|-> -- <command...>
check() {
    local label="$1" expect_exit="$2" expect_err="$3"
    shift 4  # drop the literal --
    local out err rc
    out="$WORK/out.$$"
    err="$WORK/err.$$"
    "$@" >"$out" 2>"$err"
    rc=$?

    local ok=true
    if [ "$rc" -ne "$expect_exit" ]; then
        ok=false
        echo "FAIL: $label — expected exit $expect_exit, got $rc"
    fi
    if [ "$expect_err" != "-" ] && ! grep -qF "$expect_err" "$err"; then
        ok=false
        echo "FAIL: $label — stderr missing: $expect_err"
        sed -n '1,10p' "$err"
    fi
    if $ok; then
        PASS=$((PASS + 1))
        echo "PASS: $label"
    else
        FAIL=$((FAIL + 1))
    fi
    rm -f "$out" "$err"
}

# --- fixtures ---------------------------------------------------------------

cat > "$WORK/alloc_loop.esk" <<'EOF'
(define (sum-to n)
  (let loop ((i 0) (acc 0))
    (if (>= i n) acc (loop (+ i 1) (+ acc i)))))
(display (sum-to 20000000))
(newline)
EOF

cat > "$WORK/big_string.esk" <<'EOF'
(display (string-length (make-string 5000 #\x)))
(newline)
EOF

cat > "$WORK/big_tensor.esk" <<'EOF'
(display (tensor-shape (make-tensor (list 100 100) 0.0)))
(newline)
EOF

cat > "$WORK/deep_rec.esk" <<'EOF'
(define (down n) (if (= n 0) 0 (+ 1 (down (- n 1)))))
(display (down 5000))
(newline)
EOF

cat > "$WORK/vm_loop.esk" <<'EOF'
(define (count n acc) (if (= n 0) acc (count (- n 1) (+ acc 1))))
(display (count 3000000 0))
EOF

# --- ceilings are OPT-IN ----------------------------------------------------
#
# The defaults in the docs are the values a limit takes WHEN YOU TURN IT ON,
# not a ceiling every program is silently held to. This case pins that, and it
# is not hypothetical: the first cut of this fix enforced the documented 1 GiB
# heap default unconditionally and killed tests/features/blc_test.esk, a
# program that had run for years and had never been under any ceiling. An
# unconfigured run must behave exactly as it did before limits were enforced.

if [ -f "tests/features/blc_test.esk" ]; then
    unset_all_limits
    check "a run that sets nothing gets no ceiling, even past the 1 GiB default" 0 "-" -- \
        "$ESHKOL_RUN" -r tests/features/blc_test.esk
fi

# --- ESHKOL_MAX_HEAP --------------------------------------------------------

unset_all_limits
ESHKOL_MAX_HEAP=1 \
check "ESHKOL_MAX_HEAP=1 terminates with 120" 120 "ESHKOL_MAX_HEAP" -- \
    "$ESHKOL_RUN" -r "$WORK/alloc_loop.esk"

unset_all_limits
ESHKOL_MAX_HEAP=4G \
check "ESHKOL_MAX_HEAP=4G leaves the run untouched" 0 "-" -- \
    "$ESHKOL_RUN" -r "$WORK/alloc_loop.esk"

unset_all_limits
ESHKOL_MAX_HEAP=1 ESHKOL_ENFORCE_LIMITS=false \
check "ESHKOL_ENFORCE_LIMITS=false makes the heap ceiling advisory" 0 "-" -- \
    "$ESHKOL_RUN" -r "$WORK/alloc_loop.esk"

# --- ESHKOL_TIMEOUT_MS ------------------------------------------------------

unset_all_limits
ESHKOL_TIMEOUT_MS=500 \
check "ESHKOL_TIMEOUT_MS=500 terminates with 124" 124 "ESHKOL_TIMEOUT_MS" -- \
    "$ESHKOL_RUN" -r "$WORK/alloc_loop.esk"

unset_all_limits
ESHKOL_TIMEOUT_MS=600000 \
check "a timeout the program never reaches changes nothing" 0 "-" -- \
    "$ESHKOL_RUN" -r "$WORK/alloc_loop.esk"

# --- ESHKOL_MAX_STRING_LEN --------------------------------------------------

unset_all_limits
ESHKOL_MAX_STRING_LEN=100 \
check "ESHKOL_MAX_STRING_LEN=100 terminates with 123" 123 "ESHKOL_MAX_STRING_LEN" -- \
    "$ESHKOL_RUN" -r "$WORK/big_string.esk"

unset_all_limits
ESHKOL_MAX_STRING_LEN=1M \
check "a string ceiling above the string is inert" 0 "-" -- \
    "$ESHKOL_RUN" -r "$WORK/big_string.esk"

# --- ESHKOL_MAX_TENSOR_ELEMS ------------------------------------------------

unset_all_limits
ESHKOL_MAX_TENSOR_ELEMS=100 \
check "ESHKOL_MAX_TENSOR_ELEMS=100 terminates with 122" 122 "ESHKOL_MAX_TENSOR_ELEMS" -- \
    "$ESHKOL_RUN" -r "$WORK/big_tensor.esk"

unset_all_limits
ESHKOL_MAX_TENSOR_ELEMS=1000000 \
check "a tensor ceiling above the tensor is inert" 0 "-" -- \
    "$ESHKOL_RUN" -r "$WORK/big_tensor.esk"

# --- ESHKOL_MAX_STACK -------------------------------------------------------
#
# The depth guard is emitted at lambda entry, so this uses the eval path where
# the recursive function is compiled as one. Codegen does not yet emit the
# guard for every top-level `define`d function — that gap is ESH-0101 (see
# tests/stress/found/deep_recursion_270k_no_diagnostic.esk) and is deliberately
# NOT closed here: tests/stress/rec_deep_nontco_250k.esk pins plain non-tail
# recursion working to 250000 frames, which the documented 100000 default would
# break. What this case pins is that the VARIABLE is live where the guard runs.

unset_all_limits
ESHKOL_MAX_STACK=100 \
check "ESHKOL_MAX_STACK=100 terminates with 121" 121 "ESHKOL_MAX_STACK" -- \
    "$ESHKOL_RUN" -e '(begin (define (down n) (if (= n 0) 0 (+ 1 (down (- n 1))))) (display (down 5000)))'

unset_all_limits
ESHKOL_MAX_STACK=100000 \
check "a stack ceiling above the recursion is inert" 0 "-" -- \
    "$ESHKOL_RUN" -e '(begin (define (down n) (if (= n 0) 0 (+ 1 (down (- n 1))))) (display (down 5000)))'

# --- ESHKOL_VM_MAX_INSN (bytecode VM) ---------------------------------------

if [ -x "$VM_RUN" ]; then
    unset_all_limits
    ESHKOL_VM_MAX_INSN=100000 \
    check "ESHKOL_VM_MAX_INSN=100000 terminates the VM with 125" 125 "ESHKOL_VM_MAX_INSN" -- \
        "$VM_RUN" "$WORK/vm_loop.esk"

    unset_all_limits
    ESHKOL_VM_MAX_INSN=0 \
    check "ESHKOL_VM_MAX_INSN=0 means unlimited" 0 "-" -- \
        "$VM_RUN" "$WORK/vm_loop.esk"
else
    echo "SKIP: bytecode VM standalone binary not built ($VM_RUN)"
fi

# --- summary ----------------------------------------------------------------

echo
echo "resource-limit enforcement: $PASS passed, $FAIL failed"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
echo "PASS: resource limits enforced"
exit 0
