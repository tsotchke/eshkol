#!/usr/bin/env python3
"""Atlas-only cross-check of core.transcript claims against PARI/GP and SymPy.

scripts/math_crosscheck/replay_check.py already recomputes every known claim
kind with a pure-Python-stdlib implementation (exact_math.py) and needs
nothing beyond Python 3, so CTest runs it in ordinary CI. This script is the
*additional*, heavier cross-check against two real, independently maintained
computer algebra systems -- PARI/GP (determinant, Smith normal form, rank mod
p) and SymPy (determinant, Smith normal form, polynomial gcd) -- that are not
installed on every CI host, so it is not wired into CTest; it is meant to be
run by hand (or by a lane script) wherever both tools are available, which at
the time of writing means atlas only (`/opt/homebrew/bin/gp`, SymPy via the
system python3).

Usage:
    python3 crosscheck_pari_sympy.py FILE.json [FILE2.json ...]
    python3 crosscheck_pari_sympy.py --dir fixtures/ --receipt receipt.json

Exit code 0 iff every hash check and every claim this script knows how to
recompute AGREED with both PARI/GP and SymPy (where each tool applies to that
claim kind); nonzero otherwise. A claim kind this script does not implement
is SKIPPED, not AGREE, exactly as in replay_check.py. With --receipt, writes a
JSON receipt recording the tool versions used and a per-claim verdict, so the
result is itself evidence (see docs/reports/).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# exact_math.py and replay_check.py's decode()/recompute_hash() are reused
# directly rather than copied, so the canonical-value decoding and the hash
# check are defined in exactly one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_math as em  # noqa: E402
import replay_check as rc  # noqa: E402

GP_BIN = "gp"
GP_TIMEOUT_S = 60


class ToolError(RuntimeError):
    pass


def _run_gp(script: str) -> str:
    try:
        result = subprocess.run(
            [GP_BIN, "-q"],
            input=script,
            capture_output=True,
            text=True,
            timeout=GP_TIMEOUT_S,
        )
    except FileNotFoundError as e:
        raise ToolError(f"PARI/GP ('{GP_BIN}') not found") from e
    if result.returncode != 0 or result.stderr.strip():
        raise ToolError(f"gp error: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def _gp_matrix_literal(rows: List[List]) -> str:
    def cell(x) -> str:
        from fractions import Fraction
        f = Fraction(x)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    return "[" + ";".join(",".join(cell(x) for x in row) for row in rows) + "]"


def gp_version() -> str:
    out = _run_gp('print(version())')
    return out.strip()


def gp_determinant(rows: List[List]) -> "em.Fraction":
    from fractions import Fraction
    out = _run_gp(f"print(matdet({_gp_matrix_literal(rows)}))")
    return Fraction(out) if "/" in out else Fraction(int(out))


def gp_smith_normal_form(rows: List[List[int]]) -> List[int]:
    out = _run_gp(f"print(matsnf({_gp_matrix_literal(rows)}))")
    vals = [int(v.strip()) for v in out.strip("[]").split(",") if v.strip()]
    return sorted(abs(v) for v in vals if v != 0)


def gp_rank_mod_p(rows: List[List[int]], p: int) -> int:
    mat = "[" + ";".join(",".join(f"Mod({x},{p})" for x in row) for row in rows) + "]"
    out = _run_gp(f"print(matrank({mat}))")
    return int(out)


def gp_poly_gcd(a: List[int], b: List[int]) -> List[int]:
    """a, b: coefficient lists, LOWEST degree first (core.transcript's
    convention). Vecrev on the way in and out keeps PARI's own
    highest-degree-first convention entirely internal to this function."""
    av = "[" + ",".join(str(int(x)) for x in reversed(a)) + "]" if a else "[0]"
    bv = "[" + ",".join(str(int(x)) for x in reversed(b)) + "]" if b else "[0]"
    out = _run_gp(f"print(Vecrev(gcd(Pol({av}),Pol({bv}))))")
    vals = [int(v.strip()) for v in out.strip("[]").split(",") if v.strip() != ""]
    while vals and vals[-1] == 0:
        vals.pop()
    return vals


_sympy = None


def _lazy_sympy():
    global _sympy
    if _sympy is None:
        import sympy
        _sympy = sympy
    return _sympy


def sympy_version() -> str:
    return _lazy_sympy().__version__


def sympy_determinant(rows: List[List]):
    sp = _lazy_sympy()
    return sp.Rational(sp.Matrix(rows).det())


def sympy_smith_normal_form(rows: List[List[int]]) -> List[int]:
    sp = _lazy_sympy()
    from sympy.matrices.normalforms import smith_normal_form
    M = smith_normal_form(sp.Matrix(rows), domain=sp.ZZ)
    n = min(M.rows, M.cols)
    diag = [abs(int(M[i, i])) for i in range(n)]
    return sorted(d for d in diag if d != 0)


def sympy_poly_gcd(a: List[int], b: List[int]) -> List[int]:
    sp = _lazy_sympy()
    x = sp.symbols("x")
    pa = sp.Poly(list(reversed(a)) or [0], x)
    pb = sp.Poly(list(reversed(b)) or [0], x)
    g = pa.gcd(pb)
    coeffs = g.all_coeffs()  # highest degree first
    if coeffs == [0]:
        return []
    return list(reversed([int(c) for c in coeffs]))


def crosscheck_claim(claim: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns a list of {"tool":..., "status":..., "detail":...} verdicts,
    one per (tool, aspect) this script knows how to check for this claim's
    kind. Empty list means SKIPPED (unknown kind, same convention as
    replay_check.py)."""
    kind = claim["kind"]
    inputs = claim["inputs"]
    output = claim["output"]
    verdicts: List[Dict[str, Any]] = []

    def record(tool: str, ok: Optional[bool], detail: str):
        verdicts.append({
            "tool": tool,
            "status": "AGREE" if ok is True else ("DISAGREE" if ok is False else "ERROR"),
            "detail": detail,
        })

    try:
        if kind == "determinant":
            mat = rc._first_matrix_input(inputs)
            want = rc.decode(output)
            from fractions import Fraction
            got_gp = gp_determinant(mat)
            record("pari/gp", Fraction(got_gp) == Fraction(want), f"matdet={got_gp} committed={want}")
            got_sp = sympy_determinant(mat)
            record("sympy", Fraction(str(got_sp)) == Fraction(want), f"Matrix.det={got_sp} committed={want}")

        elif kind == "smith_normal_form":
            mat = rc._first_matrix_input(inputs)
            want = sorted(rc.decode(output))
            got_gp = gp_smith_normal_form(mat)
            record("pari/gp", got_gp == want, f"matsnf={got_gp} committed={want}")
            got_sp = sympy_smith_normal_form(mat)
            record("sympy", got_sp == want, f"smith_normal_form={got_sp} committed={want}")

        elif kind in ("rank_mod_p", "gram_matrix_rank_mod_p"):
            mat = rc._first_matrix_input(inputs)
            p = rc._prime_input(inputs)
            want = rc.decode(output)
            got_gp = gp_rank_mod_p(mat, p)
            record("pari/gp", got_gp == want, f"matrank_mod_{p}={got_gp} committed={want}")

        elif kind == "gcd":
            a = rc.decode(inputs["a"])
            b = rc.decode(inputs["b"])
            want = rc.decode(output)
            got_gp = gp_poly_gcd(a, b)
            record("pari/gp", got_gp == want, f"gcd={got_gp} committed={want}")
            got_sp = sympy_poly_gcd(a, b)
            record("sympy", got_sp == want, f"Poly.gcd={got_sp} committed={want}")

        elif kind == "character_count":
            # No external tool applies: griffiths/shioda are both already
            # Eshkol-side counts over a finite set of characters, not a
            # linear-algebra primitive PARI/SymPy expose directly.
            return []

    except ToolError as e:
        # Every function that can raise ToolError calls out to PARI/GP; a
        # SymPy failure is an uncaught exception from sympy itself, not this.
        record("pari/gp", None, str(e))

    return verdicts


def check_file(path: Path) -> (bool, List[Dict[str, Any]]):
    doc = json.loads(path.read_text())
    ok_all = True
    receipts: List[Dict[str, Any]] = []
    if doc.get("schema") != "eshkol.transcript.v1":
        print(f"{path}: FAIL schema != eshkol.transcript.v1 (got {doc.get('schema')!r})")
        return False, receipts
    claims = doc.get("claims", [])
    print(f"{path}: program={doc.get('program')!r} claims={len(claims)}")
    for claim in claims:
        label = claim.get("label", "?")
        kind = claim.get("kind", "?")
        recomputed = rc.recompute_hash(claim)
        hash_ok = recomputed == claim.get("hash")
        if not hash_ok:
            print(f"  HASH MISMATCH kind={kind} label={label!r}")
            ok_all = False
            receipts.append({"file": str(path), "label": label, "kind": kind,
                              "hash_ok": False, "verdicts": []})
            continue
        verdicts = crosscheck_claim(claim)
        if not verdicts:
            print(f"  SKIPPED kind={kind} label={label!r} (no PARI/SymPy check for this kind)")
        for v in verdicts:
            print(f"  {v['status']} kind={kind} label={label!r} tool={v['tool']} {v['detail']}")
            if v["status"] != "AGREE":
                ok_all = False
        receipts.append({"file": str(path), "label": label, "kind": kind,
                          "hash_ok": True, "verdicts": verdicts})
    return ok_all, receipts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None,
                         help="write a JSON receipt (tool versions + per-claim verdicts) here")
    args = parser.parse_args(argv)

    files = list(args.files)
    if args.dir:
        files.extend(sorted(args.dir.glob("*.json")))
    if not files:
        parser.error("no input files (pass FILE.json... or --dir DIR)")

    tool_versions = {}
    try:
        tool_versions["pari_gp"] = gp_version()
    except ToolError as e:
        tool_versions["pari_gp"] = f"UNAVAILABLE: {e}"
    try:
        tool_versions["sympy"] = sympy_version()
    except Exception as e:  # pragma: no cover - import failure path
        tool_versions["sympy"] = f"UNAVAILABLE: {e}"

    ok = True
    all_receipts: List[Dict[str, Any]] = []
    for f in files:
        file_ok, receipts = check_file(f)
        ok = ok and file_ok
        all_receipts.extend(receipts)

    if args.receipt:
        args.receipt.write_text(json.dumps({
            "schema": "eshkol.math_crosscheck.pari_sympy_receipt.v1",
            "tool_versions": tool_versions,
            "files": [str(f) for f in files],
            "claims": all_receipts,
            "verdict": "PASS" if ok else "FAIL",
        }, indent=2))
        print(f"receipt written to {args.receipt}")

    print(f"tool versions: PARI/GP={tool_versions.get('pari_gp')!r} SymPy={tool_versions.get('sympy')!r}")
    if ok:
        print("RESULT: ALL PASS")
        return 0
    print("RESULT: FAILURES DETECTED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
