#!/usr/bin/env python3
"""Certify golden/squared_distance.json independently of Eshkol.

Every finite gradient in the golden file is compared with the gradient of the
same closed form (arcosh for the ball, arccos for the sphere, the flat norm,
their weighted sum for the product) differentiated SYMBOLICALLY by sympy at the
exact binary64 inputs and evaluated to 50 digits. No Eshkol code path takes
part, so a changed implementation cannot certify its own output.

    python3 tests/qllm_oracle/certify_squared_distance.py [golden.json ...]

For each file it prints, per case, the error of each component in units in
the last place of the correctly rounded reference, and the exact values of
the two identities the exporter measures in binary64 (the ball and sphere
Gauss lemma |grad|_g = 2 d, and the sphere's radial component <g, x>).
"""
import json
import math
import sys
from pathlib import Path

import sympy as sp


def exact(v):
    return sp.Rational(*float(v).as_integer_ratio())


def ball_d2(X, Y):
    dv = [a - b for a, b in zip(X, Y)]
    a = 1 - sum(t * t for t in X)
    b = 1 - sum(t * t for t in Y)
    return sp.acosh(1 + 2 * sum(t * t for t in dv) / (a * b)) ** 2


def sphere_d2(X, Y):
    nx = sp.sqrt(sum(t * t for t in X))
    ny = sp.sqrt(sum(t * t for t in Y))
    return sp.acos(sum(a * b for a, b in zip(X, Y)) / (nx * ny)) ** 2


def euclid_d2(X, Y):
    return sum((a - b) ** 2 for a, b in zip(X, Y))


def product_d2(X, Y):   # H2(w=1) x S2(ambient 3, w=2) x R2(w=1/2)
    return (ball_d2(X[0:2], Y[0:2]) + 2 * sphere_d2(X[2:5], Y[2:5])
            + sp.Rational(1, 2) * euclid_d2(X[5:7], Y[5:7]))


FORMS = {"euclidean": euclid_d2, "hyperbolic": ball_d2,
         "spherical": sphere_d2, "product": product_d2}


def certify(path):
    doc = json.loads(Path(path).read_text())
    total = 0.0
    print(path)
    for case in doc["cases"]:
        if not case["gradients"]["all_finite"]:
            continue
        x = [exact(v) for v in case["inputs"]["x"]]
        y = [exact(v) for v in case["inputs"]["y"]]
        syms = sp.symbols("x0:%d" % len(x))
        f = FORMS[case["form"]](list(syms), y)
        at = dict(zip(syms, x))
        ref = [sp.N(sp.diff(f, s).subs(at), 50) for s in syms]
        got = case["gradients"]["d_d2_d_x"]
        errs = [float(abs(exact(g) - r)) / math.ulp(float(r)) if float(r) else float(abs(exact(g)))
                for g, r in zip(got, ref)]
        cr = sum(1 for g, r in zip(got, ref) if float(g) == float(r))
        total += sum(errs)
        print("  %-34s max %.3f ulp  sum %.3f ulp  correctly rounded %d/%d"
              % (case["id"], max(errs), sum(errs), cr, len(errs)))
        norm = sp.sqrt(sum(exact(g) ** 2 for g in got))
        if case["form"] == "hyperbolic":
            d = sp.sqrt(ball_d2(x, y))
            lam = 2 / (1 - sum(t * t for t in x))
            print("      exact |grad|/lambda vs 2d: %.3e" % float(abs(norm / lam - 2 * d) / (1 + 2 * d)))
        if case["form"] == "spherical":
            d = sp.sqrt(sphere_d2(x, y))
            print("      exact |grad| vs 2d: %.3e   exact radial <g,x>: %.3e"
                  % (float(abs(norm - 2 * d) / (1 + 2 * d)),
                     float(abs(sum(exact(g) * xi for g, xi in zip(got, x))))))
    print("  total error over all finite components: %.3f ulp" % total)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for p in sys.argv[1:] or [str(here / "golden" / "squared_distance.json")]:
        certify(p)
