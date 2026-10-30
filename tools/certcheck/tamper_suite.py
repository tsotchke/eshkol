#!/usr/bin/env python3
"""tamper_suite -- every altered certificate must be rejected.

Not part of the checker's trusted base: this is the test that the checker
rejects false statements.  For each claim of each document it builds

  * an UNSEALED edit (the output changed, identifiers left as they were),
    which the checker must reject on the identifiers alone;
  * a RESEALED false output (the output changed to a false statement and the
    sha256 id and fnv1a hash recomputed), which the checker must reject on
    the mathematics, because a forger can always recompute a hash;
  * a RESEALED broken witness (one witness entry changed so the witness
    identity no longer holds), likewise rejected on the mathematics.

Usage: tamper_suite.py FILE.json [...] | --dir DIR
Prints one line per mutation; "RESULT: ALL REJECTED" iff every mutation was
rejected and every original claim was accepted.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eshkol_certcheck as cc  # noqa: E402


def z(n):
    return {"t": "z", "v": str(n)}


def enc(x):
    x = Fraction(x)
    return z(x.numerator) if x.denominator == 1 else {"t": "q", "n": str(x.numerator), "d": str(x.denominator)}


def bump(v, by=1):
    """Add `by` to an encoded exact number (z, q or dyadic)."""
    return enc(cc.dec(v) + by)


def reseal(claim):
    claim["id"] = cc.claim_id(claim)
    claim["hash"] = cc.claim_hash(claim)
    return claim


# --- false outputs ---------------------------------------------------------

def false_output(c):
    k, out = c["kind"], c["output"]
    if k in ("rank", "determinant", "integer_gcd"):
        c["output"] = bump(out)
    elif k == "smith_normal_form":
        if out["data"]:
            out["data"][0] = bump(out["data"][0])
        else:
            out["data"].append(z(1))
            out["n"] = 1
    elif k == "linear_solution":
        out["data"][0] = bump(out["data"][0])
    elif k == "linear_solve_enclosure":
        if cc.dec(out["status"]) != "proved":
            return None
        hi = cc.dec(out["boxes"]["data"][0]["data"][1])
        out["boxes"]["data"][0]["data"] = [enc(hi + 1), enc(hi + 1)]
    elif k in ("psd", "polynomial_identity", "primality", "cocycle"):
        out["v"] = not out["v"]
    elif k == "poly_gcd":
        out["coeffs"][0] = bump(out["coeffs"][0])
    elif k in ("polynomial_range", "taylor_model_range"):
        hi = cc.dec(out["data"][1])
        out["data"] = [enc(hi + 1), enc(hi + 1)]
    elif k == "cohomology_class":
        out["v"] = "coboundary" if out["v"] == "nontrivial" else "nontrivial"
    else:
        return None
    return c


# --- broken witnesses --------------------------------------------------------

def mat_bump(m, i, j, by=1):
    m["data"][i][j] = bump(m["data"][i][j], by)


def broken_witness(c):
    k, w, inp = c["kind"], c["witness"], c["inputs"]
    if k == "rank":
        cols = cc.dec(w["cols"])
        if not cols or w["kernel"]["rows"] == 0:
            return None
        mat_bump(w["kernel"], 0, cols[0])           # A e_pivot != 0
    elif k == "determinant":
        mat_bump(w["U"], 0, 0)                      # (L U)[0][0] moves
    elif k == "smith_normal_form":
        if w["U_inv"]["rows"] == 0:
            return None
        mat_bump(w["U_inv"], 0, 0)                  # U U_inv != I
    elif k == "linear_solution":
        mat_bump(w["inverse"], 0, 0)                # A A^-1 != I
    elif k == "linear_solve_enclosure":
        if cc.dec(c["output"]["status"]) != "proved":
            return None
        mat_bump(inp["inverse"], 0, 0, 1000)        # theta >= 1
    elif k == "psd":
        if "D" in w:
            w["D"]["data"][0] = bump(w["D"]["data"][0])
        else:
            w["v"]["data"] = [z(0)] * len(w["v"]["data"])
    elif k == "integer_gcd":
        if cc.dec(inp["a"]) == 0:
            return None
        w["s"] = bump(w["s"])                       # changes s a + t b by a
    elif k == "poly_gcd":
        if not cc.ptrim(cc.dec(inp["a"])):
            return None
        if w["s"]["coeffs"]:
            w["s"]["coeffs"][0] = bump(w["s"]["coeffs"][0])
        else:
            w["s"]["coeffs"] = [z(1)]
    elif k == "polynomial_identity":
        if w["difference"]:
            w["difference"] = []
        else:
            w["difference"] = [{"e": {"t": "list", "n": len(w["vars"]),
                                      "data": [z(0)] * len(w["vars"])}, "c": z(1)}]
    elif k == "primality":
        if "pratt" in w:
            node = w["pratt"]
            if "factors" not in node:
                node["a"] = z(1)
            else:
                node["factors"][0]["e"] = bump(node["factors"][0]["e"])
        elif "miller_rabin_base" in w:
            del w["miller_rabin_base"]
            w["factor"] = z(4)
        else:
            return None
    elif k == "polynomial_range":
        if len(w["shifted"]["coeffs"]) < 2:
            return None
        w["shifted"]["coeffs"][1] = bump(w["shifted"]["coeffs"][1])
    elif k == "taylor_model_range":
        lo, hi = inp["remainder"]["data"]
        inp["remainder"]["data"] = [lo, enc(cc.dec(hi) + 1)]
    elif k == "cocycle":
        if not w["product"]["data"]:
            return None
        w["product"]["data"][0] = bump(w["product"]["data"][0])
    elif k == "cohomology_class":
        if "w" in w:
            w["w"]["data"][0] = bump(w["w"]["data"][0])
        else:
            m = inp["coboundary_map"]["rows"]
            y = cc.matvec(cc.mat(w["U"]), cc.dec(inp["cocycle"]))
            ds = cc.dec(w["diagonal"])
            ok = [i for i in range(m)
                  if (i < len(ds) and y[i] % ds[i] == 0) or (i >= len(ds) and y[i] == 0)]
            if not ok:
                return None
            w["index"] = z(ok[0])
    else:
        return None
    return c


def run(docs):
    total, failures = 0, 0
    for name, doc in docs:
        for claim in doc["claims"]:
            label = f"{name}: {claim['kind']} {claim['label']!r}"
            verdict, detail = cc.check_claim(claim)
            if verdict == "REJECTED":
                print(f"FAIL original rejected: {label}: {detail}")
                failures += 1
                continue
            unsealed = false_output(copy.deepcopy(claim))
            cases = []
            if unsealed is not None:
                cases.append(("unsealed edit", unsealed))
                cases.append(("resealed false output", reseal(false_output(copy.deepcopy(claim)))))
            broken = broken_witness(copy.deepcopy(claim))
            if broken is not None:
                cases.append(("resealed broken witness", reseal(broken)))
            for what, mutated in cases:
                total += 1
                v, d = cc.check_claim(mutated)
                ok = v == "REJECTED"
                if not ok:
                    failures += 1
                print(f"{'ok  ' if ok else 'FAIL'} {what}: {label}: {v}: {d}")
    print(f"mutations: {total}, accepted by mistake or originals rejected: {failures}")
    return failures == 0 and total > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tamper suite for eshkol_certcheck.")
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    args = ap.parse_args(argv)
    files = list(args.files) + (sorted(args.dir.glob("*.json")) if args.dir else [])
    if not files:
        ap.error("no input")
    ok = run([(f.name, cc.load(f)) for f in files])
    print("RESULT: ALL REJECTED" if ok else "RESULT: TAMPER ACCEPTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
