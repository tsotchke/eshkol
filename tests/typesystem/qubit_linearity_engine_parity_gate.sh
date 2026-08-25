#!/usr/bin/env bash
# BI-5 gate: the linear (no-cloning) rule holds on BOTH engines, or neither.
#
# The language advertises a linear `Qubit` as a compile-time guarantee: cloning
# one is a type error, the compile stops, and no artifact is written. #471 made
# that true of the LLVM engine and measured that the bytecode VM enforced
# NOTHING — `eshkol-vm-standalone-test` ran a clone to completion with exit 0
# and no diagnostic, and `--emit-eskb` wrote the bytecode for it. A type-system
# guarantee that holds on one engine and not the other is an engine-parity
# defect, and this gate is what stops it coming back.
#
# It pins all four corners, because pinning only the rejection would let a fix
# that simply refuses every qubit program pass:
#
#   1. VM source execution REJECTS a clone   — nonzero exit, a real diagnostic.
#   2. VM source execution ACCEPTS a correct linear program — exit 0, its output.
#   3. --emit-eskb REJECTS a clone           — nonzero exit AND no .eskb written.
#   4. --emit-eskb ACCEPTS a correct program — exit 0 AND a .eskb written.
#
# Case 3's "no .eskb written" is the load-bearing half: an emitted bytecode file
# is indistinguishable from a good one once it leaves the process, so anything
# downstream that trusts the file's existence would certify a clone.
#
# Usage: qubit_linearity_engine_parity_gate.sh <eshkol-run> <vm-standalone> <workdir>

set -u

ESHKOL_RUN="${1:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"
VM_STANDALONE="${2:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"
WORKDIR="${3:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"

if [ "$VM_STANDALONE" = "none" ] || [ ! -x "$VM_STANDALONE" ]; then
    # A build without the bytecode VM has no parity to check. Say so rather
    # than reporting a PASS that measured nothing.
    echo "SKIP: bytecode VM not built in this configuration"
    echo "PASS: qubit linearity engine parity (vacuous — no VM in this build)"
    exit 0
fi

mkdir -p "$WORKDIR" || { echo "FAIL: cannot create $WORKDIR"; exit 1; }
rm -f "$WORKDIR"/*.esk "$WORKDIR"/*.eskb 2>/dev/null

CLONE="$WORKDIR/clone.esk"
LEGAL="$WORKDIR/legal.esk"

cat > "$CLONE" <<'EOF'
(define (h (q : Qubit)) : Qubit q)
(define (bad-clone (q : Qubit))
  (cons q q))
(display "THIS MUST NEVER RUN")
(newline)
EOF

cat > "$LEGAL" <<'EOF'
(define (h (q : Qubit)) : Qubit q)
(define (z (q : Qubit)) : Qubit q)
(define (pick (b : Bool) (q : Qubit)) : Qubit
  (cond (b (h q))
        (else (z q))))
(display "LEGAL-LINEAR-PROGRAM-RAN")
(newline)
EOF

FAILED=0
fail() { echo "FAIL: $1"; FAILED=1; }

export ESHKOL_VM_NO_DISASM=1

# ---- 1. VM source execution must REJECT the clone -------------------------
out=$("$VM_STANDALONE" "$CLONE" 2>&1); rc=$?
if [ "$rc" -eq 0 ]; then
    fail "the VM ran a qubit clone to completion (exit 0) — engine parity broken"
fi
case "$out" in
    *"linear"*) : ;;
    *) fail "the VM rejected the clone with no linearity diagnostic" ;;
esac
case "$out" in
    *"THIS MUST NEVER RUN"*)
        fail "the VM executed the body of a program it was supposed to refuse" ;;
esac

# ---- 2. VM source execution must ACCEPT the correct program ---------------
out=$("$VM_STANDALONE" "$LEGAL" 2>&1); rc=$?
if [ "$rc" -ne 0 ]; then
    fail "the VM refused a CORRECT linear program (exit $rc): $out"
fi
case "$out" in
    *"LEGAL-LINEAR-PROGRAM-RAN"*) : ;;
    *) fail "the VM accepted the correct program but never ran it" ;;
esac

# ---- 3. --emit-eskb must REJECT the clone and write nothing ---------------
CLONE_ESKB="$WORKDIR/clone.eskb"
rm -f "$CLONE_ESKB"
out=$("$ESHKOL_RUN" "$CLONE" --profile hosted-vm --emit-eskb "$CLONE_ESKB" 2>&1); rc=$?
if [ "$rc" -eq 0 ]; then
    fail "--emit-eskb accepted a qubit clone (exit 0)"
fi
if [ -e "$CLONE_ESKB" ]; then
    fail "--emit-eskb wrote bytecode for a program that violates linear typing"
fi

# ---- 4. --emit-eskb must ACCEPT the correct program -----------------------
LEGAL_ESKB="$WORKDIR/legal.eskb"
rm -f "$LEGAL_ESKB"
out=$("$ESHKOL_RUN" "$LEGAL" --profile hosted-vm --emit-eskb "$LEGAL_ESKB" 2>&1); rc=$?
if [ "$rc" -ne 0 ]; then
    fail "--emit-eskb refused a CORRECT linear program (exit $rc): $out"
fi
if [ ! -s "$LEGAL_ESKB" ]; then
    fail "--emit-eskb accepted the correct program but wrote no bytecode"
fi

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
echo "PASS: qubit linearity engine parity (VM enforces the same rule as LLVM)"
exit 0
