#!/usr/bin/env python3
"""lean_export -- turn the decidable finite facts of Eshkol certificates
into Lean 4 theorems that the Lean kernel re-checks.

Design record: docs/design/adr/0047-certificates-and-independent-checker.md

For every claim of a supported kind whose data is integral, the exporter
writes one theorem whose statement is the witness identity itself (a matrix
product equal to a stated matrix, s * a + t * b = g, a list of Pratt
conditions) and whose proof is `by decide`: the kernel evaluates both sides.
The file depends on Lean core only (no Mathlib, no Lake package).

Supported kinds: determinant (n <= 6), rank (integral matrix, rank and
nullity <= 6), smith_normal_form, integer_gcd, primality (prime claims),
cocycle, cohomology_class.  Other claims are listed in the file header as
not exported; eshkol_certcheck.py checks every kind.

Usage:
    lean_export.py CERT.json -o OUT.lean
    lean_export.py --dir DIR --out-dir OUTDIR      (one .lean per .json)
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from math import lcm
from pathlib import Path

PRELUDE = r"""namespace EshkolCert

/-! Exact integer matrices as lists of rows, and the operations the
    certificate statements use.  Every definition is structurally recursive,
    so `decide` evaluates it in the kernel. -/

def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

def col (m : List (List Int)) (j : Nat) : List Int := m.map (fun r => r.getD j 0)

/-- `mmul a b n`: the product of `a` and `b`, where `b` has `n` columns. -/
def mmul (a b : List (List Int)) (n : Nat) : List (List Int) :=
  a.map (fun r => (List.range n).map (fun j => dot r (col b j)))

def mvmul (a : List (List Int)) (v : List Int) : List Int := a.map (fun r => dot r v)

def eye (n : Nat) : List (List Int) :=
  (List.range n).map (fun i => (List.range n).map (fun j => if i = j then 1 else 0))

/-- Determinant by cofactor expansion along the first row (Laplace). -/
def det : Nat → List (List Int) → Int
  | 0, _ => 1
  | n + 1, m =>
    match m with
    | [] => 1
    | r :: rs =>
      ((List.range (n + 1)).map (fun j =>
        (if j % 2 = 0 then (1 : Int) else -1) * r.getD j 0 *
          det n (rs.map (fun row => row.eraseIdx j)))).foldl (· + ·) 0

def divChain : List Int → Bool
  | a :: b :: rest => b % a == 0 && divChain (b :: rest)
  | _ => true

def powModAux : Nat → Nat → Nat → Nat → Nat → Nat
  | 0, _, _, _, acc => acc
  | fuel + 1, b, e, m, acc =>
    if e = 0 then acc
    else powModAux fuel (b * b % m) (e / 2) m (if e % 2 = 1 then acc * b % m else acc)

/-- `b ^ e % m` by repeated squaring (exponents below 2^4096). -/
def powMod (b e m : Nat) : Nat := powModAux 4096 (b % m) e m (1 % m)

def prodPow : List (Nat × Nat) → Nat
  | [] => 1
  | (q, e) :: rest => q ^ e * prodPow rest

/-- One Pratt step for `n` with base `a` and the factorization `fs` of
    `n - 1`: every prime factor other than 2 must already be certified. -/
def prattStep (known : List Nat) (n a : Nat) (fs : List (Nat × Nat)) : Bool :=
  decide (n > 2) && prodPow fs == n - 1 && powMod a (n - 1) n == 1 &&
    fs.all (fun qe => powMod a ((n - 1) / qe.1) n != 1 && (qe.1 == 2 || known.contains qe.1))

/-- A Pratt certificate as a list of steps, smaller primes first.  By
    Lehmer's theorem (if a^(n-1) = 1 mod n and a^((n-1)/q) != 1 mod n for
    every prime q dividing n - 1, then n is prime; Mathlib: `lucas_primality`),
    every `n` in a list accepted here is prime.  That theorem is not
    re-proved in this file, which depends on Lean core only. -/
def prattOk : List Nat → List (Nat × Nat × List (Nat × Nat)) → Bool
  | _, [] => true
  | known, (n, a, fs) :: rest => prattStep known n a fs && prattOk (n :: known) rest

/-- Bezout data make `g` a greatest common divisor of `a` and `b`. -/
theorem bezout_gcd {a b g s t qa qb : Int}
    (h : s * a + t * b = g) (ha : a = g * qa) (hb : b = g * qb) :
    g ∣ a ∧ g ∣ b ∧ ∀ d : Int, d ∣ a → d ∣ b → d ∣ g := by
  refine ⟨⟨qa, ha⟩, ⟨qb, hb⟩, ?_⟩
  intro d hda hdb
  rw [← h]
  exact Int.dvd_add (Int.dvd_trans hda (Int.dvd_mul_left s a))
    (Int.dvd_trans hdb (Int.dvd_mul_left t b))

end EshkolCert

open EshkolCert
"""

MAX_DET = 6


class Skip(Exception):
    pass


def dec(v):
    t = v["t"]
    if t == "z":
        return int(v["v"])
    if t == "q":
        return Fraction(int(v["n"]), int(v["d"]))
    if t == "dyadic":
        return Fraction(int(v["m"])) * Fraction(2) ** int(v["e"])
    if t in ("bool", "str"):
        return v["v"]
    if t in ("list", "vec"):
        return [dec(x) for x in v["data"]]
    if t == "mat":
        return (v["rows"], v["cols"], [[dec(x) for x in r] for r in v["data"]])
    if t == "poly":
        return [dec(x) for x in v["coeffs"]]
    raise Skip(f"value tag {t!r}")


def as_int(x):
    x = Fraction(x)
    if x.denominator != 1:
        raise Skip("rational entry")
    return int(x)


def lint(x):
    x = as_int(x)
    return f"({x})" if x < 0 else str(x)


def lvec(v):
    return "[" + ", ".join(lint(x) for x in v) + "]"


def lmat(m):
    return "[" + ", ".join(lvec(r) for r in m[2]) + "]"


def scale_integral(vec):
    d = lcm(*[Fraction(x).denominator for x in vec]) if vec else 1
    return [as_int(Fraction(x) * d) for x in vec]


def name(kind, claim):
    return f"{kind}_{claim['id'].split(':', 1)[1][:12]}"


def stmt_determinant(c):
    A, d = dec(c["inputs"]["matrix"]), dec(c["output"])
    n = A[0]
    if n > MAX_DET:
        raise Skip(f"{n} x {n} exceeds the cofactor bound {MAX_DET}")
    return [f"theorem {name('determinant', c)} :\n    det {n} {lmat(A)} = {lint(d)} := by decide"]


def stmt_rank(c):
    A, r = dec(c["inputs"]["matrix"]), dec(c["output"])
    m, n, a = A
    for row in a:
        for x in row:
            as_int(x)
    w = c["witness"]
    rows, cols, free = dec(w["rows"]), dec(w["cols"]), dec(w["kernel_cols"])
    K = [scale_integral(v) for v in dec(w["kernel"])[2]]
    if r > MAX_DET or n - r > MAX_DET:
        raise Skip("minor exceeds the cofactor bound")
    minor = (r, r, [[a[i][j] for j in cols] for i in rows])
    kminor = (n - r, n - r, [[v[j] for j in free] for v in K])
    Kt = (n, n - r, [[K[k][j] for k in range(n - r)] for j in range(n)])
    zero = (m, n - r, [[0] * (n - r) for _ in range(m)])
    return [f"/-- rank {r}: a nonzero {r} x {r} minor and {n - r} independent kernel vectors\n"
            f"    (rank-nullity, not re-proved here). -/\n"
            f"theorem {name('rank', c)} :\n"
            f"    mmul {lmat(A)} {lmat(Kt)} {n - r} = {lmat(zero)} ∧\n"
            f"    det {r} {lmat(minor)} ≠ 0 ∧\n"
            f"    det {n - r} {lmat(kminor)} ≠ 0 := by decide"]


def stmt_snf(c):
    A = dec(c["inputs"]["matrix"])
    m, n, _ = A
    ds = [as_int(x) for x in dec(c["output"])]
    w = c["witness"]
    U, V, Ui, Vi = (dec(w[k]) for k in ("U", "V", "U_inv", "V_inv"))
    D = (m, n, [[ds[i] if i == j and i < len(ds) else 0 for j in range(n)] for i in range(m)])
    return [f"/-- Smith normal form: U A V = D with U, V invertible over Z. -/\n"
            f"theorem {name('smith_normal_form', c)} :\n"
            f"    mmul (mmul {lmat(U)} {lmat(A)} {n}) {lmat(V)} {n} = {lmat(D)} ∧\n"
            f"    mmul {lmat(U)} {lmat(Ui)} {m} = eye {m} ∧\n"
            f"    mmul {lmat(V)} {lmat(Vi)} {n} = eye {n} ∧\n"
            f"    divChain {lvec(ds)} = true := by decide"]


def stmt_gcd(c):
    a, b, g = (as_int(dec(x)) for x in (c["inputs"]["a"], c["inputs"]["b"], c["output"]))
    w = c["witness"]
    s, t, qa, qb = (as_int(dec(w[k])) for k in ("s", "t", "a_over_g", "b_over_g"))
    nm = name("integer_gcd", c)
    return [f"theorem {nm}_bezout :\n"
            f"    ({lint(s)} : Int) * {lint(a)} + {lint(t)} * {lint(b)} = {lint(g)} ∧\n"
            f"    ({lint(a)} : Int) = {lint(g)} * {lint(qa)} ∧ ({lint(b)} : Int) = {lint(g)} * {lint(qb)} := by decide",
            f"/-- {g} is a greatest common divisor of {a} and {b}. -/\n"
            f"theorem {nm} :\n"
            f"    ({lint(g)} : Int) ∣ {lint(a)} ∧ ({lint(g)} : Int) ∣ {lint(b)} ∧\n"
            f"    ∀ d : Int, d ∣ {lint(a)} → d ∣ {lint(b)} → d ∣ {lint(g)} :=\n"
            f"  bezout_gcd {nm}_bezout.1 {nm}_bezout.2.1 {nm}_bezout.2.2"]


def pratt_steps(node, out, seen):
    n = int(node["n"]["v"])
    if n == 2 or n in seen:
        return n
    fs = []
    for f in node["factors"]:
        q = pratt_steps(f["cert"], out, seen)
        fs.append((q, int(f["e"]["v"])))
    seen.add(n)
    out.append((n, int(node["a"]["v"]), fs))
    return n


def stmt_primality(c):
    if dec(c["output"]) is not True:
        raise Skip("compositeness claims are not exported")
    n = int(c["inputs"]["n"]["v"])
    steps = []
    pratt_steps(c["witness"]["pratt"], steps, set())
    if n == 2:
        return [f"theorem {name('primality', c)} : prattOk [] [] = true := by decide  -- 2 is the base case"]
    body = ", ".join(f"({sn}, {a}, [{', '.join(f'({q}, {e})' for q, e in fs)}])" for sn, a, fs in steps)
    return [f"/-- Pratt certificate for {n} (prime by Lehmer's theorem, see `prattOk`). -/\n"
            f"theorem {name('primality', c)} :\n"
            f"    prattOk [] [{body}] = true ∧\n"
            f"    ([{', '.join(str(s[0]) for s in steps)}] : List Nat).getLast? = some {n} := by decide"]


def stmt_cocycle(c):
    D, z = dec(c["inputs"]["coboundary_map"]), dec(c["inputs"]["cochain"])
    holds = dec(c["output"])
    rel = "=" if holds else "≠"
    return [f"theorem {name('cocycle', c)} :\n"
            f"    mvmul {lmat(D)} {lvec(z)} {rel} {lvec([0] * D[0])} := by decide"]


def stmt_class(c):
    D, z = dec(c["inputs"]["coboundary_map"]), dec(c["inputs"]["cocycle"])
    m, n, _ = D
    w = c["witness"]
    if dec(c["output"]) == "coboundary":
        return [f"theorem {name('cohomology_class', c)} :\n"
                f"    mvmul {lmat(D)} {lvec(dec(w['w']))} = {lvec(z)} := by decide"]
    U, V, Ui, Vi = (dec(w[k]) for k in ("U", "V", "U_inv", "V_inv"))
    ds = [as_int(x) for x in dec(w["diagonal"])]
    i = as_int(dec(w["index"]))
    S = (m, n, [[ds[a] if a == b and a < len(ds) else 0 for b in range(n)] for a in range(m)])
    last = (f"(mvmul {lmat(U)} {lvec(z)}).getD {i} 0 % {ds[i]} ≠ 0" if i < len(ds)
            else f"(mvmul {lmat(U)} {lvec(z)}).getD {i} 0 ≠ 0")
    return [f"/-- z is not in D Z^n: U D V is diagonal with U, V invertible over Z and\n"
            f"    (U z)_{i} is not in d_{i} Z. -/\n"
            f"theorem {name('cohomology_class', c)} :\n"
            f"    mmul (mmul {lmat(U)} {lmat(D)} {n}) {lmat(V)} {n} = {lmat(S)} ∧\n"
            f"    mmul {lmat(U)} {lmat(Ui)} {m} = eye {m} ∧\n"
            f"    mmul {lmat(V)} {lmat(Vi)} {n} = eye {n} ∧\n"
            f"    {last} := by decide"]


EXPORTERS = {
    "determinant": stmt_determinant,
    "rank": stmt_rank,
    "smith_normal_form": stmt_snf,
    "integer_gcd": stmt_gcd,
    "primality": stmt_primality,
    "cocycle": stmt_cocycle,
    "cohomology_class": stmt_class,
}


def export(doc, source_name):
    theorems, skipped = [], []
    for claim in doc["claims"]:
        fn = EXPORTERS.get(claim["kind"])
        if fn is None:
            skipped.append((claim["kind"], claim["label"], "kind is checked by eshkol_certcheck.py only"))
            continue
        try:
            theorems.append((claim, fn(claim)))
        except Skip as e:
            skipped.append((claim["kind"], claim["label"], str(e)))
    lines = [f"/- Generated by tools/certcheck/lean_export.py from {source_name}",
             f"   (program {doc.get('program')!r}, schema {doc.get('schema')}).",
             f"   Check with: lean <this file>   (Lean 4 core; no Mathlib, no Lake package)",
             f"   Exported claims: {len(theorems)}; not exported: {len(skipped)}."]
    for kind, label, why in skipped:
        lines.append(f"   - not exported: {kind} {label!r}: {why}")
    lines.append("-/")
    out = "\n".join(lines) + "\n\n" + PRELUDE
    for claim, stmts in theorems:
        out += f"\n-- {claim['kind']} {claim['label']!r} {claim['id']}\n"
        out += "\n\n".join("set_option maxRecDepth 100000 in\n" + st for st in stmts) + "\n"
    return out, len(theorems), skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", nargs="?", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="compare with the existing .lean files instead of writing them")
    args = ap.parse_args(argv)
    jobs = []
    if args.file:
        if not args.out:
            ap.error("-o OUT.lean is required with a single file")
        jobs.append((args.file, args.out))
    if args.dir:
        if not args.out_dir:
            ap.error("--out-dir is required with --dir")
        for f in sorted(args.dir.glob("*.json")):
            stem = "".join(p.capitalize() for p in f.stem.replace("-", "_").split("_"))
            jobs.append((f, args.out_dir / f"Cert{stem}.lean"))
    if not jobs:
        ap.error("no input")
    outdated = 0
    for src, dst in jobs:
        doc = json.loads(src.read_text(encoding="utf-8"))
        text, count, skipped = export(doc, src.name)
        if args.check:
            same = dst.exists() and dst.read_text(encoding="utf-8") == text
            outdated += 0 if same else 1
            print(f"{dst}: {'current' if same else 'OUT OF DATE'} ({count} claims exported)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        print(f"{dst}: {count} claims exported, {len(skipped)} not exported")
    if args.check:
        print("RESULT: LEAN EXPORT CURRENT" if outdated == 0 else "RESULT: LEAN EXPORT OUT OF DATE")
    return 1 if outdated else 0


if __name__ == "__main__":
    sys.exit(main())
