#!/usr/bin/env python3
r"""harness.py — shared self-checking program builder for the P8 escape-closure
pillar generators.

Every P8 generator emits SELF-CHECKING closed Eshkol programs in the exact
format the v1.3.4 edge runner already understands: a `;; CHECKS: N` header, a
`chk` helper that prints `PASS: <name>` / `FAIL: <name>` per assertion, and a
trailing `SUMMARY <pass>/<fail>` line. A file is green iff it prints N PASS
lines, zero FAIL lines, and exits 0 on every execution axis the runner drives.

Design rules (identical to tests/edge_matrix/gen_matrix.py and
scripts/gen_edge_v134.py, so a CI divergence reproduces byte-for-byte locally):
  * Deterministic: a pure function of (seed, counts, depth). No runtime RNG.
  * Ground truth computed HERE (in Python) and embedded as a literal, so a
    single broken builtin cannot make both sides wrong the same way. Where an
    external analytic value is unavailable, a metamorphic property
    (identity / involution / agreement-across-forms) is asserted instead.
  * Bounded: every family caps its case count.

This module is import-only; it defines no corpus of its own.
"""

import struct

# The embedded prologue. `chk` records PASS/FAIL; `apx`/`rel` are tolerance
# compares; `close?` is a relative-or-absolute compare robust near zero.
HARNESS = (
    "(define __pass 0)\n"
    "(define __fail 0)\n"
    "(define (chk name ok)\n"
    "  (if ok (begin (set! __pass (+ __pass 1)) (display \"PASS: \") (display name) (newline))\n"
    "         (begin (set! __fail (+ __fail 1)) (display \"FAIL: \") (display name) (newline))))\n"
    "(define (apx a b tol) (< (abs (- a b)) tol))\n"
    "(define (close? a b rtol atol)\n"
    "  (< (abs (- a b)) (+ atol (* rtol (abs b)))))\n"
)


def fmt_double(x):
    """Emit a double literal that reads back bit-exact in Eshkol (shortest
    round-trip repr, which Python guarantees round-trips to the same float)."""
    if x != x:
        return "+nan.0"
    if x == float("inf"):
        return "+inf.0"
    if x == float("-inf"):
        return "-inf.0"
    s = repr(float(x))
    if "." not in s and "e" not in s and "E" not in s and "n" not in s:
        s += ".0"
    return s


def double_bits(x):
    """IEEE-754 bit pattern of a double (for exact round-trip assertions)."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


class Program:
    """Accumulates check lines and renders a complete self-checking file."""

    def __init__(self, doc):
        self.doc = doc
        self.top = []          # top-level definitions / expressions (in order)
        self.checks = []       # (name, eshkol-bool-expr)
        self.vm_skip = False   # runner skips this file on the VM lane if set
        self.tags = []         # freeform ;; TAG lines (runner-readable)

    def define(self, text):
        self.top.append(text)
        return self

    def tag(self, text):
        self.tags.append(text)
        return self

    def check(self, name, expr):
        # name must be a bare token (no spaces) so a runner grep stays simple.
        self.checks.append((name.replace(" ", "-"), expr))
        return self

    def render(self):
        out = [";; %s" % self.doc, ";; CHECKS: %d" % len(self.checks)]
        if self.vm_skip:
            out.append(";; VM-SKIP")
        for t in self.tags:
            out.append(";; %s" % t)
        out.append(HARNESS)
        out.extend(self.top)
        for name, expr in self.checks:
            out.append("(chk \"%s\" %s)" % (name, expr))
        out.append(
            "(display \"SUMMARY \") (display __pass) (display \"/\") "
            "(display __fail) (newline)")
        return "\n".join(out) + "\n"
