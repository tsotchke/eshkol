#!/usr/bin/env python3
"""Pure-Python replay checker for core.transcript JSON documents.

For every claim in every given transcript file this script:

  1. Verifies the claim's "hash" against an independent recomputation of
     FNV-1a 64 over the canonical JSON text of {kind,label,inputs,output}
     in that field order (TAMPER CHECK -- catches a transcript edited
     after the fact; it does not by itself prove the claim is correct).
  2. For every claim kind this script knows how to recompute, replays
     the mathematics from "inputs" with exact_math.py (a second,
     independently written implementation) and compares against
     "output" (CORRECTNESS CHECK).

It needs only the Python 3 standard library -- no PARI/GP, no SymPy, no
third-party package -- so it is the piece CTest can run in ordinary CI.
scripts/math_crosscheck/crosscheck_pari_sympy.py is the atlas-only
sibling that additionally checks against PARI/GP and SymPy.

Usage:
    python3 replay_check.py FILE.json [FILE2.json ...]
    python3 replay_check.py --dir fixtures/

Exit code 0 iff every hash check and every known-kind correctness check
agreed; nonzero otherwise.  Unknown claim kinds are reported as SKIPPED,
not AGREE, and do not by themselves fail the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List

import exact_math as em

KNOWN_KINDS = (
    "determinant",
    "smith_normal_form",
    "rank_mod_p",
    "gram_matrix_rank_mod_p",
    "gcd",
    "character_count",
)


class ReplayError(ValueError):
    pass


def decode(v: Any):
    """Decode one canonical tv-* value into a plain Python object."""
    if not isinstance(v, dict) or "t" not in v:
        raise ReplayError(f"not a canonical transcript value: {v!r}")
    t = v["t"]
    if t == "z":
        return int(v["v"])
    if t == "q":
        return Fraction(int(v["n"]), int(v["d"]))
    if t == "bool":
        return bool(v["v"])
    if t == "str":
        return v["v"]
    if t == "mat":
        return [[decode(e) for e in row] for row in v["data"]]
    if t == "vec" or t == "list":
        return [decode(e) for e in v["data"]]
    if t == "poly":
        return [decode(e) for e in v["coeffs"]]
    if t == "opaque":
        return v.get("repr")
    raise ReplayError(f"unknown canonical value tag: {t!r}")


def recompute_hash(claim: Dict[str, Any]) -> str:
    without_hash = {
        "kind": claim["kind"],
        "label": claim["label"],
        "inputs": claim["inputs"],
        "output": claim["output"],
    }
    text = json.dumps(without_hash, separators=(",", ":"))
    return em.hash_hex16(em.fnv1a_64(text))


def _first_matrix_input(inputs: Dict[str, Any]) -> List[List[int]]:
    for key in ("matrix", "boundary", "gram_matrix", "coboundary_image", "delta_n_rows", "m"):
        if key in inputs:
            return decode(inputs[key])
    # fall back to the first input that decodes to a matrix
    for v in inputs.values():
        if isinstance(v, dict) and v.get("t") == "mat":
            return decode(v)
    raise ReplayError("no matrix-shaped input found")


def _prime_input(inputs: Dict[str, Any]) -> int:
    for key in ("p", "prime"):
        if key in inputs:
            return decode(inputs[key])
    raise ReplayError("no prime input found (expected key 'p' or 'prime')")


def replay_claim(claim: Dict[str, Any]):
    """Returns (status, detail) where status is one of
    'AGREE', 'DISAGREE', 'SKIPPED'."""
    kind = claim["kind"]
    inputs = claim["inputs"]
    output = claim["output"]

    if kind == "determinant":
        mat = _first_matrix_input(inputs)
        got = em.determinant(mat)
        want = decode(output)
        ok = Fraction(got) == Fraction(want)
        return ("AGREE" if ok else "DISAGREE", f"det={got} committed={want}")

    if kind == "smith_normal_form":
        mat = _first_matrix_input(inputs)
        got = sorted(em.smith_normal_form_diagonal(mat))
        want = sorted(decode(output))
        ok = got == want
        return ("AGREE" if ok else "DISAGREE", f"snf={got} committed={want}")

    if kind in ("rank_mod_p", "gram_matrix_rank_mod_p"):
        mat = _first_matrix_input(inputs)
        p = _prime_input(inputs)
        got = em.rank_mod_p(mat, p)
        want = decode(output)
        ok = got == want
        return ("AGREE" if ok else "DISAGREE", f"rank_mod_{p}={got} committed={want}")

    if kind == "gcd":
        a = decode(inputs["a"])
        b = decode(inputs["b"])
        got = em.poly_gcd_z(a, b)
        want = decode(output)
        ok = got == want
        return ("AGREE" if ok else "DISAGREE", f"gcd={got} committed={want}")

    if kind == "character_count":
        out = output
        griffiths = decode(out["griffiths"]) if "griffiths" in out else None
        shioda = decode(out["shioda"]) if "shioda" in out else None
        if griffiths is None or shioda is None:
            return ("SKIPPED", "character_count claim missing griffiths/shioda fields")
        ok = griffiths == shioda
        return ("AGREE" if ok else "DISAGREE",
                f"griffiths={griffiths} shioda={shioda} (checked: the two independently "
                "derived counts are equal; this does not re-derive them from m,n)")

    return ("SKIPPED", f"no replay implementation for claim kind {kind!r}")


def check_file(path: Path) -> bool:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "eshkol.transcript.v1":
        print(f"{path}: FAIL schema != eshkol.transcript.v1 (got {doc.get('schema')!r})")
        return False
    ok_all = True
    claims = doc.get("claims", [])
    print(f"{path}: program={doc.get('program')!r} claims={len(claims)}")
    for i, claim in enumerate(claims):
        label = claim.get("label", f"#{i}")
        kind = claim.get("kind", "?")
        recomputed = recompute_hash(claim)
        hash_ok = recomputed == claim.get("hash")
        if not hash_ok:
            print(f"  HASH MISMATCH kind={kind} label={label!r} "
                  f"committed={claim.get('hash')} recomputed={recomputed}")
            ok_all = False
            continue
        try:
            status, detail = replay_claim(claim)
        except ReplayError as e:
            print(f"  ERROR kind={kind} label={label!r}: {e}")
            ok_all = False
            continue
        print(f"  {status} kind={kind} label={label!r} {detail}")
        if status == "DISAGREE":
            ok_all = False
    return ok_all


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=None,
                         help="check every *.json file in this directory")
    args = parser.parse_args(argv)

    files = list(args.files)
    if args.dir:
        files.extend(sorted(args.dir.glob("*.json")))
    if not files:
        parser.error("no input files (pass FILE.json... or --dir DIR)")

    ok = True
    for f in files:
        if not check_file(f):
            ok = False

    if ok:
        print("RESULT: ALL PASS")
        return 0
    print("RESULT: FAILURES DETECTED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
