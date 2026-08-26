#!/usr/bin/env bash
# SW-46 gate: the bytecode VM's divergence and curl are EXACT.
#
# Both operators used to reach their answer through a central difference at
# h = 1e-7 (lib/backend/vm_native.c, cases 753 and 754). The error that
# introduces is small, plausible and completely silent: `curl` of a gradient
# field — whose curl vanishes identically at every point — returned
#
#     #(1.1102230246251565e-09 -3.3306690738754696e-09 0)
#
# instead of #(0 0 0). No diagnostic, exit 0. Nothing could see it: the VM
# parity differential compares program OUTPUT between engines and op:CURL is a
# `gap` row, so the comparison is never attempted; and native curl SIGSEGVs on
# the same field (ledger LE-12), so there was no other side to compare against.
#
# This gate therefore compares against MATHEMATICS rather than against the
# other engine. Every field below has a curl and a divergence that is exactly
# representable in binary floating point, so the assertions are equality, not
# tolerance — a tolerance is what let the defect live.
#
# It also asserts (ad-finite-difference-evals) reads 0. The counter is the
# structural half of the claim: an exact answer produced by six extra function
# evaluations is not exact AD, it is a difference quotient that happened to
# land. Before the fix this program printed FD=6.
#
# Usage: ad_vm_exactness_gate.sh <eshkol-run> <eshkol-vm|none> <workdir>
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT

set -u

ESHKOL_RUN="${1:?usage: ad_vm_exactness_gate.sh <eshkol-run> <eshkol-vm|none> <workdir>}"
ESHKOL_VM="${2:-none}"
WORKDIR="${3:?workdir required}"

# Fails closed. An absent VM binary is not evidence that the VM's AD is exact,
# and a gate that passes when its subject is missing is the shape this whole
# wave exists to remove (ledger SW-51).
if [ "$ESHKOL_VM" = "none" ] || [ ! -x "$ESHKOL_VM" ]; then
    echo "FAIL: bytecode VM binary not available ('$ESHKOL_VM') — cannot certify VM AD exactness"
    exit 1
fi

mkdir -p "$WORKDIR" || { echo "FAIL: cannot create $WORKDIR"; exit 1; }
PROG="$WORKDIR/ad_vm_exactness.esk"

cat > "$PROG" <<'ESK'
(ad-reset-counters!)

;; F1 = grad(xyz): a gradient field, so curl F1 = (0 0 0) identically,
;; and div F1 = 0 at every point.
(define (F1 x y z) (list (* y z) (* x z) (* x y)))

;; F2 = rigid rotation about z: curl = (0 0 2) everywhere, div = 0.
(define (F2 x y z) (list (- y) x (* 0.0 z)))

;; F3: div = 2x + 2y + 2z = 12 at (1 2 3); curl = (0 0 0).
(define (F3 x y z) (list (* x x) (* y y) (* z z)))

;; F4 = (y^2 z, x z^2, x^2 y): at (1 2 3)
;;   curl = (x^2 - 2xz, y^2 - 2xy, z^2 - 2yz) = (-5 0 -3); div = 0.
(define (F4 x y z) (list (* y (* y z)) (* x (* z z)) (* x (* x y))))

(define P (list 1.0 2.0 3.0))

(display "C1=") (display (curl F1 P))
(display "|C2=") (display (curl F2 P))
(display "|C3=") (display (curl F3 P))
(display "|C4=") (display (curl F4 P))
(display "|D1=") (display (divergence F1 P))
(display "|D2=") (display (divergence F2 P))
(display "|D3=") (display (divergence F3 P))
(display "|D4=") (display (divergence F4 P))
(display "|FD=") (display (ad-finite-difference-evals))
(display "|END")
ESK

OUT=$("$ESHKOL_VM" "$PROG" 2>&1)
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAIL: VM exited $RC"
    printf '%s\n' "$OUT" | tail -20
    exit 1
fi

# The VM appends a newline per `display` call (a filed, normalized quirk:
# tests/vm_parity/found/display_newline_per_call.esk), so flatten before
# matching.
FLAT=$(printf '%s' "$OUT" | tr -d '\n')

fail=0
expect() {
    local key="$1" want="$2" got
    got=$(printf '%s' "$FLAT" | sed -n "s/.*|\{0,1\}${key}=\(.*\)/\1/p" | sed 's/|.*//')
    if [ "$got" != "$want" ]; then
        echo "FAIL: ${key} = '${got}', expected exactly '${want}'"
        fail=1
    else
        echo "  ok  ${key} = ${want}"
    fi
}

case "$FLAT" in
    *"|END"*) ;;
    *) echo "FAIL: program did not run to completion"; printf '%s\n' "$OUT" | tail -20; exit 1 ;;
esac

# Exact equality, deliberately. Each of these is representable.
expect C1 "#(0 0 0)"
expect C3 "#(0 0 0)"
expect C4 "#(-5 0 -3)"
expect D1 "0"
expect D2 "0"
expect D3 "12"
expect D4 "0"
expect FD "0"

# curl F2 is (0 0 2); the x and y components may carry a signed zero, which is
# numerically equal to 0 and is not a derivative error. The z component is the
# load-bearing one and must be exactly 2 — the difference quotient returned
# 1.9999999995023998.
case "$FLAT" in
    *"C2=#(0 0 2)"*|*"C2=#(0 -0.0 2)"*|*"C2=#(-0.0 0 2)"*|*"C2=#(-0.0 -0.0 2)"*)
        echo "  ok  C2 z-component = 2 exactly" ;;
    *)
        got=$(printf '%s' "$FLAT" | sed -n 's/.*C2=\(.*\)/\1/p' | sed 's/|.*//')
        echo "FAIL: C2 = '${got}', expected the z-component to be exactly 2"
        fail=1 ;;
esac

if [ "$fail" -ne 0 ]; then
    echo "FAIL: VM divergence/curl are not exact"
    # The VM echoes its whole bytecode listing before running, so report only
    # the result segment rather than 100 KB of disassembly.
    printf 'observed: %s\n' "$(printf '%s' "$FLAT" | sed -n 's/.*\(C1=.*|END\).*/\1/p')"
    exit 1
fi

echo "PASS: VM divergence and curl are exact (0 finite-difference evaluations)"
exit 0
