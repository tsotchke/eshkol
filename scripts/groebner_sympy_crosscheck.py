#!/usr/bin/env python3
"""SymPy cross-check of core.groebner's reduced Groebner bases.

SymPy is an independent oracle here, never a dependency of the library. The
check runs in two steps that may happen on different machines:

  corpus     write tests/sympy_oracle/groebner_corpus.json: named systems
             (Cox-Little-O'Shea examples, cyclic-n, Katsura-n) and seeded
             random systems over Q and F_p under lex, grlex and grevlex.
  expected   compute every reduced basis with SymPy's `groebner` and write
             tests/sympy_oracle/groebner_expected.json (needs SymPy).
  compare    compute every reduced basis with core.groebner (eshkol-run, JIT
             and AOT), compare with the expected bases term by term and write
             a receipt (tests/sympy_oracle/groebner_crosscheck_receipt.json
             by default). Needs eshkol-run, not SymPy.

Bases are compared exactly: both sides are reduced and monic, so they are
equal as sets of polynomials exactly when the ideals and orders agree.
Coefficients over F_p are compared as residues in [0, p).

Exit codes: 0 every case agrees, 1 a disagreement, 2 an infrastructure error.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "tests/sympy_oracle/groebner_corpus.json"
EXPECTED = REPO / "tests/sympy_oracle/groebner_expected.json"
RECEIPT = REPO / "tests/sympy_oracle/groebner_crosscheck_receipt.json"
SEED = 20261029


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

def term(c, *es):
    return [str(Fraction(c)), *es]


def cyclic(n):
    gens = []
    for d in range(n - 1):
        terms = {}
        for i in range(n):
            e = [0] * n
            for j in range(d + 1):
                e[(i + j) % n] += 1
            terms[tuple(e)] = terms.get(tuple(e), 0) + 1
        gens.append([term(c, *e) for e, c in terms.items()])
    gens.append([term(1, *([1] * n)), term(-1, *([0] * n))])
    return [f"x{i}" for i in range(n)], gens


def katsura(n):
    m = n + 1

    def u(l):
        l = abs(l)
        return l if l <= n else None

    gens = []
    first = {}
    for l in range(-n, n + 1):
        k = u(l)
        if k is not None:
            e = [0] * m
            e[k] = 1
            first[tuple(e)] = first.get(tuple(e), 0) + 1
    first[tuple([0] * m)] = first.get(tuple([0] * m), 0) - 1
    gens.append(first)
    for mm in range(n):
        t = {}
        for l in range(-n, n + 1):
            a, b = u(l), u(mm - l)
            if a is None or b is None:
                continue
            e = [0] * m
            e[a] += 1
            e[b] += 1
            t[tuple(e)] = t.get(tuple(e), 0) + 1
        e = [0] * m
        e[mm] = 1
        t[tuple(e)] = t.get(tuple(e), 0) - 1
        gens.append(t)
    return [f"u{i}" for i in range(m)], [[term(c, *e) for e, c in g.items() if c != 0] for g in gens]


def named_cases():
    cases = []
    clo27 = (["x", "y"], [[term(1, 3, 0), term(-2, 1, 1)], [term(1, 2, 1), term(-2, 0, 2), term(1, 1, 0)]])
    clo28 = (["x", "y", "z"], [[term(1, 2, 0, 0), term(1, 0, 1, 0), term(1, 0, 0, 1), term(-1, 0, 0, 0)],
                               [term(1, 1, 0, 0), term(1, 0, 2, 0), term(1, 0, 0, 1), term(-1, 0, 0, 0)],
                               [term(1, 1, 0, 0), term(1, 0, 1, 0), term(1, 0, 0, 2), term(-1, 0, 0, 0)]])
    lagrange = (["l", "x", "y", "z"], [[term(3, 0, 2, 0, 0), term(2, 0, 0, 1, 1), term(-2, 1, 1, 0, 0)],
                                       [term(2, 0, 1, 0, 1), term(-2, 1, 0, 1, 0)],
                                       [term(2, 0, 1, 1, 0), term(-2, 0, 0, 0, 1), term(-2, 1, 0, 0, 1)],
                                       [term(1, 0, 2, 0, 0), term(1, 0, 0, 2, 0), term(1, 0, 0, 0, 2), term(-1, 0, 0, 0, 0)]])
    cubic = (["t", "x", "y", "z"], [[term(1, 1, 0, 0, 0), term(-1, 0, 1, 0, 0)],
                                    [term(1, 2, 0, 0, 0), term(-1, 0, 0, 1, 0)],
                                    [term(1, 3, 0, 0, 0), term(-1, 0, 0, 0, 1)]])
    systems = {"clo-2.7": clo27, "clo-2.8": clo28, "clo-lagrange": lagrange, "twisted-cubic": cubic,
               "cyclic-4": cyclic(4), "cyclic-5": cyclic(5), "katsura-3": katsura(3), "katsura-4": katsura(4)}
    plan = [("clo-2.7", ["grlex", "lex", "grevlex"], ["QQ", 32003]),
            ("clo-2.8", ["lex", "grevlex"], ["QQ", 101]),
            ("clo-lagrange", ["lex", "grevlex"], ["QQ", 10007]),
            ("twisted-cubic", ["lex", "grevlex"], ["QQ", 7]),
            ("cyclic-4", ["grevlex", "lex"], ["QQ", 32003]),
            ("cyclic-5", ["grevlex"], ["QQ", 32003]),
            ("katsura-3", ["grevlex", "lex"], ["QQ", 32003]),
            ("katsura-4", ["grevlex"], ["QQ", 32003])]
    for name, orders, fields in plan:
        vars_, gens = systems[name]
        for order in orders:
            for field in fields:
                cases.append({"id": f"{name}/{order}/{field}", "vars": vars_, "order": order,
                              "field": field, "gens": gens})
    return cases


def random_cases(count):
    rng = random.Random(SEED)
    cases = []
    fields = ["QQ", "QQ", 7, 101, 32003]
    orders = ["lex", "grlex", "grevlex"]
    for k in range(count):
        n = rng.choice([2, 2, 3, 3, 3])
        vars_ = ["x", "y", "z"][:n]
        field = fields[k % len(fields)]
        order = orders[(k // len(fields)) % len(orders)]
        gens = []
        for _ in range(rng.choice([2, 2, 3])):
            terms = {}
            for _ in range(rng.choice([2, 3, 3, 4])):
                e = [0] * n
                for _ in range(rng.randint(0, 3)):
                    e[rng.randrange(n)] += 1
                c = Fraction(rng.randint(-6, 6), rng.choice([1, 1, 1, 2, 3]))
                if c != 0:
                    terms[tuple(e)] = terms.get(tuple(e), 0) + c
            g = [term(c, *e) for e, c in sorted(terms.items()) if c != 0]
            if g:
                gens.append(g)
        if gens:
            cases.append({"id": f"random-{k:03d}/{order}/{field}", "vars": vars_, "order": order,
                          "field": field, "gens": gens})
    return cases


def write_corpus(path, count):
    corpus = {"schema": "eshkol.groebner_corpus.v1", "seed": SEED,
              "cases": named_cases() + random_cases(count)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact_json(corpus, "cases"))
    print(f"corpus: {len(corpus['cases'])} cases -> {path.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# SymPy's bases
# ---------------------------------------------------------------------------

def residue(c, field):
    c = Fraction(c)
    if field == "QQ":
        return str(c)
    p = int(field)
    return str(c.numerator * pow(c.denominator, -1, p) % p)


def sympy_bases(corpus_path, out_path, limit_seconds):
    import sympy as sp
    corpus = json.loads(corpus_path.read_text())
    out = {"schema": "eshkol.groebner_expected.v1", "sympy_version": sp.__version__,
           "corpus_sha256": sha256(corpus_path), "cases": {}}
    for case in corpus["cases"]:
        syms = sp.symbols(case["vars"])
        polys = []
        for g in case["gens"]:
            expr = 0
            for t in g:
                mono = 1
                for s, e in zip(syms, t[1:]):
                    mono *= s ** e
                expr += sp.Rational(residue(t[0], case["field"])) * mono
            polys.append(expr)
        kw = {"order": case["order"]}
        if case["field"] == "QQ":
            kw["domain"] = sp.QQ
        else:
            kw["modulus"] = int(case["field"])
        start = time.time()
        G = sp.groebner(polys, *syms, **kw)
        basis = []
        for g in G.exprs:
            P = sp.Poly(g, *syms, **({"domain": sp.QQ} if case["field"] == "QQ" else {"modulus": int(case["field"])}))
            terms = P.terms(order=case["order"])
            lc = sp.Rational(terms[0][1]) if case["field"] == "QQ" else int(terms[0][1])
            row = []
            for mono, c in terms:
                if case["field"] == "QQ":
                    row.append([str(Fraction(str(sp.Rational(c) / lc))), *mono])
                else:
                    p = int(case["field"])
                    row.append([str(int(c) * pow(lc % p, -1, p) % p), *mono])
            basis.append(row)
        out["cases"][case["id"]] = canonical(basis, case["field"])
        elapsed = time.time() - start
        if elapsed > limit_seconds:
            print(f"note: {case['id']} took {elapsed:.1f} s in SymPy")
    out_path.write_text(compact_json(out, "cases"))
    print(f"expected: {len(out['cases'])} bases from SymPy {sp.__version__} -> {out_path.relative_to(REPO)}")


def canonical(basis, field):
    """A basis as a sorted list of polynomials, each a sorted list of terms."""
    polys = []
    for g in basis:
        polys.append(sorted([[residue(t[0], field), *[int(e) for e in t[1:]]] for t in g],
                            key=lambda t: t[1:]))
    return sorted(polys, key=lambda g: json.dumps(g))


# ---------------------------------------------------------------------------
# core.groebner's bases
# ---------------------------------------------------------------------------

def esk_field(field):
    return "rationals" if field == "QQ" else f"(make-zmod {int(field)})"


def driver_source(corpus):
    out = ["(require core.ring)", "(require core.groebner)",
           "(define (emit id G)",
           "  (display \"CASE \") (display id) (newline)",
           "  (for-each (lambda (g) (display \"POLY\")",
           "              (for-each (lambda (t) (display \" \") (display (car t))",
           "                          (for-each (lambda (e) (display \",\") (display e)) (cdr t)))",
           "                        (ring-poly-terms g)) (newline)) G)",
           "  (display \"END\") (newline))"]
    for case in corpus["cases"]:
        vars_ = " ".join(case["vars"])
        gens = " ".join("(ring-poly P '(" + " ".join("(" + " ".join([residue(t[0], case["field"])] + [str(e) for e in t[1:]]) + ")"
                                                     for t in g) + "))"
                        for g in case["gens"])
        out.append(f"(let ((P (make-polys {esk_field(case['field'])} '({vars_}) '{case['order']})))")
        out.append(f"  (emit \"{case['id']}\" (groebner-basis (list {gens}))))")
    out.append("(display \"RESULT: DONE\") (newline)")
    return "\n".join(out) + "\n"


def parse_output(text, corpus):
    fields = {c["id"]: c["field"] for c in corpus["cases"]}
    result, current, basis = {}, None, []
    for line in text.splitlines():
        if line.startswith("CASE "):
            current, basis = line[5:].strip(), []
        elif line.startswith("POLY") and current:
            terms = []
            for tok in line[4:].split():
                parts = tok.split(",")
                terms.append([parts[0], *[int(e) for e in parts[1:]]])
            basis.append(terms)
        elif line.startswith("END") and current:
            result[current] = canonical(basis, fields[current])
            current = None
    return result


def run_engine(eshkol_run, source, work, engine, env):
    if engine == "jit":
        proc = subprocess.run([eshkol_run, "-r", str(source)], capture_output=True, text=True, env=env, timeout=7200)
        return proc.returncode, proc.stdout + proc.stderr
    binary = work / "groebner_crosscheck_aot"
    link = Path(eshkol_run).resolve().parent
    comp = subprocess.run([eshkol_run, "-O", "1", "-o", str(binary), str(source), f"-L{link}"],
                          capture_output=True, text=True, env=env, timeout=7200)
    if comp.returncode != 0:
        return comp.returncode, comp.stdout + comp.stderr
    proc = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=7200)
    return proc.returncode, proc.stdout + proc.stderr


def compare(eshkol_run, engines, work, receipt_path):
    corpus = json.loads(CORPUS.read_text())
    expected = json.loads(EXPECTED.read_text())
    if expected.get("corpus_sha256") != sha256(CORPUS):
        print("compare: the expected bases were computed from a different corpus", file=sys.stderr)
        return 2
    work.mkdir(parents=True, exist_ok=True)
    source = work / "groebner_crosscheck_driver.esk"
    source.write_text(driver_source(corpus))
    env = dict(os.environ)
    env["ESHKOL_PATH"] = str(REPO / "lib") + (os.pathsep + env["ESHKOL_PATH"] if env.get("ESHKOL_PATH") else "")
    env.setdefault("ESHKOL_JIT_CACHE", "0")
    receipt = {"schema": "eshkol.groebner_crosscheck_receipt.v1",
               "oracle": f"SymPy {expected['sympy_version']} groebner (reduced, monic)",
               "corpus_sha256": expected["corpus_sha256"], "expected_sha256": sha256(EXPECTED),
               "library_sha256": {p: sha256(REPO / p) for p in ("lib/core/ring.esk", "lib/core/groebner.esk")},
               "cases": len(corpus["cases"]), "engines": {}}
    status = 0
    for engine in engines:
        start = time.time()
        rc, text = run_engine(eshkol_run, source, work, engine, env)
        got = parse_output(text, corpus)
        agree = [cid for cid in expected["cases"] if got.get(cid) == expected["cases"][cid]]
        differ = [cid for cid in expected["cases"] if cid in got and got[cid] != expected["cases"][cid]]
        missing = [cid for cid in expected["cases"] if cid not in got]
        receipt["engines"][engine] = {"exit": rc, "agreed": len(agree), "differed": differ, "missing": missing,
                                      "seconds": round(time.time() - start, 1)}
        print(f"{engine}: {len(agree)}/{len(expected['cases'])} bases agree with SymPy"
              f" ({len(differ)} differ, {len(missing)} missing, exit {rc})")
        if rc != 0 or differ or missing:
            status = 1
            tail = "\n".join(text.splitlines()[-20:])
            print(tail, file=sys.stderr)
    receipt["verdict"] = "PASS" if status == 0 else "FAIL"
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"receipt: {receipt['verdict']} -> {receipt_path}")
    return status


def compact_json(obj, key):
    """JSON with one entry of obj[key] per line, so a change is one line."""
    head = {k: v for k, v in obj.items() if k != key}
    lines = [json.dumps(head, sort_keys=True)[:-1] + f', "{key}": ' + ("[" if isinstance(obj[key], list) else "{")]
    items = obj[key] if isinstance(obj[key], list) else sorted(obj[key].items())
    rows = [json.dumps(v, separators=(",", ":")) if isinstance(obj[key], list)
            else json.dumps(v[0]) + ": " + json.dumps(v[1], separators=(",", ":")) for v in items]
    lines.append(",\n".join(rows))
    lines.append(("]" if isinstance(obj[key], list) else "}") + "}")
    return "\n".join(lines) + "\n"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["corpus", "expected", "compare"])
    ap.add_argument("--random", type=int, default=150, help="number of random systems in the corpus")
    ap.add_argument("--eshkol-run", default=str(REPO / "build/eshkol-run"))
    ap.add_argument("--engines", default="jit,aot")
    ap.add_argument("--work-dir", default=str(REPO / "build/test-work/groebner-crosscheck"))
    ap.add_argument("--receipt", default=str(RECEIPT))
    args = ap.parse_args()
    if args.step == "corpus":
        write_corpus(CORPUS, args.random)
        return 0
    if args.step == "expected":
        sympy_bases(CORPUS, EXPECTED, 30)
        return 0
    if not Path(args.eshkol_run).exists():
        print(f"compare: {args.eshkol_run} not found", file=sys.stderr)
        return 2
    return compare(args.eshkol_run, args.engines.split(","), Path(args.work_dir), Path(args.receipt))


if __name__ == "__main__":
    sys.exit(main())
