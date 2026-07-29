#!/usr/bin/env python3
r"""gen_property_oracles.py — P8 escape-closure axis 4(b): reference-FREE
property oracles.

Originating escape (see .swarm/P8_ESCAPE_ANALYSIS.md): the cross-implementation
(chibi) differential MISSED a 6-significant-figure float-printing defect because
its own output normalizer reformatted every float token to '%.6g' on BOTH
sides before comparing — collapsing the very precision distinction that was
broken. Two engines agreeing after a lossy normalization is "shared-defect
blindness": a differential can only ever be as strong as the property it
compares, and a lossy-normalized differential silently pinned nothing about
float text.

The durable fix is a family of oracles that assert a MATHEMATICAL property of
each substrate INDEPENDENTLY — no second engine, no shared normalizer, so no
shared-defect blindness is possible. Each check is a closed self-verifying
program that runs identically on native-JIT / AOT-O0 / AOT-O2 / VM.

Families:
  numrt   number->string . string->number == identity over a diverse set of
          doubles (subnormals, +-0.0, powers of two, irrationals, 1e+-308,
          seeded randoms). This is exactly what R7RS pins and what the printer
          bug violated: (sqrt 2) printed "1.41421" does NOT read back to (sqrt 2).
  datart  (read . write) round-trips a generated datum: nested lists/vectors,
          strings with escapes, symbols, chars, exact and inexact numbers.
  alg     algebraic identities that hold EXACTLY: (a+b)-b == a, (a*b)/b == a,
          a+0 == a, commutativity/associativity over exact integers and
          rationals (a wrong result is a hard FAIL, no tolerance).

Deterministic: pure function of --seed. Output: self-checking programs in the
shared scripts/p8/harness.py format.

Usage: python3 scripts/p8/gen_property_oracles.py --out DIR [--seed N]
                 [--family numrt|datart|alg] [--list]
"""

import argparse
import math
import os
import random
import struct
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Program, fmt_double            # noqa: E402


def _interesting_doubles(rng):
    xs = [
        0.1, 0.2, 0.3, 1.0 / 3.0, 2.0 / 3.0, math.sqrt(2.0), math.pi, math.e,
        math.sqrt(3.0), 1.0 / 7.0, 123456.789, 9.999999999999999,
        1.0000000000000002,                 # 1 + 1 ulp
        2.0 ** -52, 2.0 ** 52, 2.0 ** -1022,  # smallest normal
        2.0 ** 1023, 2.0 ** -30, 2.0 ** 30,
        5e-324, 1e-310,                      # subnormals
        1e308, 1e-308, 1.7976931348623157e308,  # near max
        -0.0, 0.0, -1.5, -math.pi, -1e-7,
        1e-5, 6.022e23, 1.602176634e-19,
    ]
    # Seeded randoms across many magnitudes.
    for _ in range(40):
        sign = rng.choice((1.0, -1.0))
        mant = rng.random()
        exp = rng.randint(-300, 300)
        xs.append(sign * mant * (10.0 ** exp))
    # Random bit patterns that are finite.
    for _ in range(20):
        bits = rng.getrandbits(64)
        v = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if v == v and v not in (float("inf"), float("-inf")):
            xs.append(v)
    return xs


def gen_numrt(rng, files):
    xs = _interesting_doubles(rng)
    # Chunk into a few files so a single crash isolates a smaller batch.
    per = 25
    chunks = [xs[i:i + per] for i in range(0, len(xs), per)]
    for ci, chunk in enumerate(chunks):
        p = Program("property: number->string . string->number == identity (chunk %d)" % ci)
        p.tag("P8-AXIS property-oracle")
        p.tag("P8-FAMILY numrt")
        p.define(
            "(define (numrt x) (= (string->number (number->string x)) x))")
        for k, x in enumerate(chunk):
            # The literal reads to some finite y; the R7RS round-trip property
            # (= (string->number (number->string y)) y) must hold for that y.
            p.check("numrt-%d-%d" % (ci, k), "(numrt %s)" % fmt_double(x))
        files["prop_numrt_%02d" % ci] = p.render()


_DATA_ATOMS = [
    "0", "1", "-1", "42", "-99999", "3/4", "-7/8", "1/1000000",
    "3.5", "-2.25", "0.1", "1.5e10",
    "#\\a", "#\\Z", "#\\space", "#\\newline",
    "\"plain\"", "\"with space\"", "\"tab\\ttab\"", "\"nl\\nnl\"",
    "\"quote\\\"inside\"", "\"back\\\\slash\"", "\"\"",
    # Symbols must be QUOTED. These atoms are emitted into evaluated
    # constructor calls -- `(vector 0 sym)` -- so a bare symbol is a variable
    # reference, not symbol data. Unquoted, the generated program referenced
    # undefined variables; the compiler printed a diagnostic for each and
    # emitted a binary anyway, so this oracle passed while pinning nothing.
    #
    # `|weird sym|` is deliberately absent: the reader does not yet implement
    # R7RS 7.1.1 vertical-line symbol syntax, so `'|weird sym|` lexes as the
    # two tokens `'|weird` and `sym|` and the latter becomes an undefined
    # variable. That gap is pinned by
    # tests/vm_parity/found/r7rs_pipe_delimited_symbol.esk -- restore this atom
    # once the reader supports it.
    "'sym", "'another-symbol", "'with->arrow",
    "#t", "#f",
]


def _rand_datum(rng, depth):
    if depth <= 0 or rng.random() < 0.4:
        return rng.choice(_DATA_ATOMS)
    n = rng.randint(0, 4)
    kids = [_rand_datum(rng, depth - 1) for _ in range(n)]
    if rng.random() < 0.5:
        return "(list %s)" % " ".join(kids) if kids else "(list)"
    return "(vector %s)" % " ".join(kids) if kids else "(vector)"


def gen_datart(rng, files):
    # Each check builds a datum, writes it to a string, reads it back, and
    # asserts equal?. Using constructors (list/vector) so the datum is a fresh
    # heap value the reader must reconstruct.
    p = Program("property: (read . write) round-trips a datum (equal?)")
    p.tag("P8-AXIS property-oracle")
    p.tag("P8-FAMILY datart")
    p.define("(define (w2s d) (let ((o (open-output-string))) (write d o) (get-output-string o)))")
    p.define("(define (r4s s) (read (open-input-string s)))")
    p.define("(define (datart d) (equal? (r4s (w2s d)) d))")
    for k in range(60):
        datum = _rand_datum(rng, rng.randint(1, 3))
        p.check("datart-%d" % k, "(datart %s)" % datum)
    files["prop_datart"] = p.render()


def gen_alg(rng, files):
    p = Program("property: exact algebraic identities (no tolerance)")
    p.tag("P8-AXIS property-oracle")
    p.tag("P8-FAMILY alg")
    # Integers.
    for k in range(30):
        a = rng.randint(-10 ** 6, 10 ** 6)
        b = rng.randint(-10 ** 6, 10 ** 6)
        p.check("int-addsub-%d" % k, "(= (- (+ %d %d) %d) %d)" % (a, b, b, a))
        p.check("int-comm-%d" % k, "(= (+ %d %d) (+ %d %d))" % (a, b, b, a))
        if b != 0:
            p.check("int-muldiv-%d" % k,
                    "(= (quotient (* %d %d) %d) %d)" % (a, b, b, a))
    # Rationals (exact division).
    for k in range(20):
        an, ad = rng.randint(-999, 999), rng.randint(1, 999)
        bn, bd = rng.randint(-999, 999), rng.randint(1, 999)
        a = Fraction(an, ad)
        b = Fraction(bn, bd)
        al = "%d/%d" % (a.numerator, a.denominator) if a.denominator != 1 else "%d" % a.numerator
        bl = "%d/%d" % (b.numerator, b.denominator) if b.denominator != 1 else "%d" % b.numerator
        p.check("rat-addsub-%d" % k, "(= (- (+ %s %s) %s) %s)" % (al, bl, bl, al))
        if b != 0:
            p.check("rat-muldiv-%d" % k, "(= (/ (* %s %s) %s) %s)" % (al, bl, bl, al))
        # associativity of + over three exact rationals
        cn, cd = rng.randint(-999, 999), rng.randint(1, 999)
        c = Fraction(cn, cd)
        cl = "%d/%d" % (c.numerator, c.denominator) if c.denominator != 1 else "%d" % c.numerator
        p.check("rat-assoc-%d" % k,
                "(= (+ (+ %s %s) %s) (+ %s (+ %s %s)))" % (al, bl, cl, al, bl, cl))
    files["prop_alg"] = p.render()


FAMILIES = {"numrt": gen_numrt, "datart": gen_datart, "alg": gen_alg}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=8804)
    ap.add_argument("--family", choices=sorted(FAMILIES) + ["all"], default="all")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = {}
    fams = sorted(FAMILIES) if args.family == "all" else [args.family]
    for fam in fams:
        FAMILIES[fam](rng, files)

    if args.list:
        for k in sorted(files):
            print(k)
        return 0
    if not args.out:
        sys.exit("--out DIR required (or --list)")
    os.makedirs(args.out, exist_ok=True)
    for name, text in sorted(files.items()):
        with open(os.path.join(args.out, name + ".esk"), "w") as fh:
            fh.write(text)
    print("wrote %d files to %s" % (len(files), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
