#!/usr/bin/env python3
"""eshkol_certcheck -- an independent checker for Eshkol certificates.

Design record: docs/design/adr/0047-certificates-and-independent-checker.md
Producer:      lib/core/certificate.esk (core.certificate)

A certificate states a mathematical fact about exact inputs together with a
witness that makes the fact cheap to confirm.  This program confirms each
fact by recomputation from the inputs and the witness alone: it multiplies
matrices, compares integers, evaluates modular powers.  It never runs an
elimination, a factorization or a search, it imports nothing from the
Eshkol tree, and its verdict never depends on a hash.

Usage:
    eshkol_certcheck.py FILE.json [FILE.json ...]
    eshkol_certcheck.py --dir DIR
    eshkol_certcheck.py --strict ...      conditional claims count as failures
    eshkol_certcheck.py --trusted-base    print the trusted-base statement

Exit status 0 iff every claim in every file is ACCEPTED (or CONDITIONAL
without --strict); 1 if any claim is REJECTED; 2 on a usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

SCHEMA = "eshkol.certificate.v1"

TRUSTED_BASE = """\
Trusted base of eshkol_certcheck (eshkol.certificate.v1)

Trusted:
  * a Python 3 interpreter and four standard-library modules: json (parsing),
    fractions (exact rationals over Python's unbounded int), hashlib (sha256,
    used only to recompute content identifiers) and argparse;
  * this one file, which contains no elimination, factorization or search:
    every check is a matrix product, a comparison, a modular power or a
    polynomial expansion;
  * the mathematical theorems named per claim kind below.

Not trusted: the Eshkol compiler, its runtime, LLVM, the machine that ran the
producing program, the producing library code, and the identifiers "id" and
"hash" (a mismatch rejects a claim, a match proves nothing).

Per kind (theorem relied on):
  rank                    rank >= r from a nonzero r x r minor; rank <= r from
                          n - r independent kernel vectors (rank-nullity).
  determinant             det(P) det(A) = det(L) det(U) for P A = L U, with L
                          unit lower and U upper triangular.
  smith_normal_form       U A V = D with U, V invertible over Z (integer
                          inverses supplied), d_i > 0 and d_i | d_(i+1): the
                          d_i are the invariant factors (uniqueness of the
                          Smith form).
  linear_solution         A A^-1 = I proves A nonsingular; A x = b.
  linear_solve_enclosure  the a-posteriori bound of core.certified_solve:
                          with R an approximate inverse, theta = max_i
                          (|I - R A| + |R| dA) w / w_i < 1 gives every
                          solution of (A + E) y = b + e, |E| <= dA,
                          |e| <= db, inside x +- (f + G w eta / (1 - theta))
                          (Banach fixed point for y -> y + R(b - A y)).
  psd                     M = B^T diag(D) B with D > 0 is positive
                          semidefinite; v^T M v < 0 refutes it.
  integer_gcd, poly_gcd   s a + t b = g and g | a, g | b make g a greatest
                          common divisor (any common divisor divides g).
  polynomial_identity     equality of expanded normal forms in Q[x1..xk].
  primality               Pratt (1975) / Lehmer: if a^(n-1) = 1 mod n and
                          a^((n-1)/q) != 1 mod n for every prime q | n - 1
                          then n is prime; a Miller-Rabin base or a proper
                          factor proves n composite.
  polynomial_range        p(x) = sum q_i (x - c)^i and |x - c| <= r give
                          |p(x) - q_0| <= sum_(i>=1) |q_i| r^i.
  taylor_model_range      the same bound for P(t), |t| <= r, plus the
                          remainder interval R, which is a PREMISE (verdict
                          CONDITIONAL): it rests on a sampled derivative
                          bound, not on a proof.
  cocycle                 D z = 0 by direct product.
  cohomology_class        z = D w (coboundary); or U D V = diag(d) with U, V
                          invertible over Z and (U z)_i not divisible by d_i
                          (d_i = 0 past the rank), impossible for z in D Z^n.
"""


class Reject(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Reject(msg)


# ---------------------------------------------------------------------------
# Canonical value decoding (core.transcript encoding)
# ---------------------------------------------------------------------------

def is_tagged(v, t):
    return isinstance(v, dict) and v.get("t") == t


def dec_int_text(s):
    need(isinstance(s, str), f"integer must be a decimal string, got {s!r}")
    body = s[1:] if s.startswith("-") else s
    need(body.isdigit() and body.isascii(), f"not a decimal integer: {s!r}")
    return int(s)


def dec(v):
    need(isinstance(v, dict) and "t" in v, f"not a canonical value: {v!r}")
    t = v["t"]
    if t == "z":
        return dec_int_text(v["v"])
    if t == "q":
        n, d = dec_int_text(v["n"]), dec_int_text(v["d"])
        need(d > 0, "rational with nonpositive denominator")
        return Fraction(n, d)
    if t == "dyadic":
        m, e = dec_int_text(v["m"]), dec_int_text(v["e"])
        return Fraction(m) * (Fraction(2) ** e)
    if t == "bool":
        need(isinstance(v["v"], bool), "bool value is not a JSON boolean")
        return v["v"]
    if t == "str":
        need(isinstance(v["v"], str), "str value is not a JSON string")
        return v["v"]
    if t in ("list", "vec"):
        data = v["data"]
        need(isinstance(data, list) and v.get("n") == len(data), "list length mismatch")
        return [dec(x) for x in data]
    if t == "mat":
        rows, cols, data = v["rows"], v["cols"], v["data"]
        need(isinstance(data, list) and len(data) == rows, "matrix row count mismatch")
        out = []
        for row in data:
            need(isinstance(row, list) and len(row) == cols, "matrix column count mismatch")
            out.append([dec(x) for x in row])
        return (rows, cols, out)
    if t == "poly":
        coeffs = v["coeffs"]
        need(isinstance(coeffs, list), "polynomial coefficients must be a list")
        need(v.get("deg") == max(0, len(coeffs) - 1), "polynomial degree mismatch")
        return [dec(x) for x in coeffs]
    raise Reject(f"unknown value tag {t!r}")


def mat(v):
    need(is_tagged(v, "mat"), "expected a matrix")
    return dec(v)


def integral(x):
    return isinstance(x, int) or (isinstance(x, Fraction) and x.denominator == 1)


def all_integral(m):
    return all(integral(x) for row in m[2] for x in row)


# ---------------------------------------------------------------------------
# Exact arithmetic (products and comparisons only)
# ---------------------------------------------------------------------------

def mmul(a, b):
    (ra, ca, da), (rb, cb, db) = a, b
    need(ca == rb, f"shape mismatch in product: {ra}x{ca} times {rb}x{cb}")
    return (ra, cb, [[sum((da[i][k] * db[k][j] for k in range(ca)), 0) for j in range(cb)]
                     for i in range(ra)])


def eye(n):
    return (n, n, [[1 if i == j else 0 for j in range(n)] for i in range(n)])


def same(a, b):
    return a[0] == b[0] and a[1] == b[1] and all(
        a[2][i][j] == b[2][i][j] for i in range(a[0]) for j in range(a[1]))


def matvec(a, v):
    need(a[1] == len(v), "shape mismatch in matrix-vector product")
    return [sum((a[2][i][k] * v[k] for k in range(a[1])), 0) for i in range(a[0])]


def transpose(a):
    return (a[1], a[0], [[a[2][i][j] for i in range(a[0])] for j in range(a[1])])


def det(rows):
    """Determinant of a small square matrix by cofactor-free fraction-free
    reduction (Bareiss); used only on the minors a witness names."""
    n = len(rows)
    if n == 0:
        return Fraction(1)
    m = [[Fraction(x) for x in r] for r in rows]
    sign, prev = 1, Fraction(1)
    for k in range(n - 1):
        if m[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if m[i][k] != 0), None)
            if swap is None:
                return Fraction(0)
            m[k], m[swap] = m[swap], m[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) / prev
        prev = m[k][k]
    return sign * m[n - 1][n - 1]


def distinct_indices(xs, bound, what):
    need(all(isinstance(x, int) and 0 <= x < bound for x in xs), f"{what}: index out of range")
    need(len(set(xs)) == len(xs), f"{what}: repeated index")


# Polynomials in one variable: coefficient lists, lowest degree first.
def ptrim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def padd(a, b):
    n = max(len(a), len(b))
    return ptrim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])


def pmul(a, b):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return ptrim(out)


def peq(a, b):
    return ptrim(a) == ptrim(b)


# ---------------------------------------------------------------------------
# Claim kinds
# ---------------------------------------------------------------------------

def k_rank(inp, out, wit):
    A = mat(inp["matrix"])
    m, n, _ = A
    r = dec(out)
    need(isinstance(r, int) and 0 <= r <= min(m, n), "rank out of range")
    rows, cols, free = dec(wit["rows"]), dec(wit["cols"]), dec(wit["kernel_cols"])
    need(len(rows) == r and len(cols) == r, "minor size differs from the rank")
    distinct_indices(rows, m, "minor rows")
    distinct_indices(cols, n, "minor columns")
    need(det([[A[2][i][j] for j in cols] for i in rows]) != 0, "the named minor is singular")
    K = mat(wit["kernel"])
    need(K[0] == n - r and K[1] == n, "kernel must hold n - rank vectors of length n")
    for vec in K[2]:
        need(all(x == 0 for x in matvec(A, vec)), "a kernel vector is not in the kernel")
    distinct_indices(free, n, "kernel columns")
    need(len(free) == n - r, "kernel column count")
    need(det([[vec[j] for j in free] for vec in K[2]]) != 0, "kernel vectors are dependent")
    return f"rank={r}"


def k_determinant(inp, out, wit):
    A = mat(inp["matrix"])
    n = A[0]
    need(A[1] == n, "determinant of a non-square matrix")
    perm, L, U = dec(wit["perm"]), mat(wit["L"]), mat(wit["U"])
    need(sorted(perm) == list(range(n)), "perm is not a permutation")
    need(L[0] == L[1] == U[0] == U[1] == n, "factor shapes")
    for i in range(n):
        need(L[2][i][i] == 1, "L is not unit diagonal")
        for j in range(i + 1, n):
            need(L[2][i][j] == 0, "L is not lower triangular")
        for j in range(i):
            need(U[2][i][j] == 0, "U is not upper triangular")
    PA = (n, n, [A[2][perm[i]] for i in range(n)])
    need(same(mmul(L, U), PA), "P A != L U")
    inversions = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
    d = Fraction(-1 if inversions % 2 else 1)
    for i in range(n):
        d *= U[2][i][i]
    need(d == dec(out), f"determinant {d} != claimed {dec(out)}")
    return f"det={d}"


def k_smith(inp, out, wit):
    A = mat(inp["matrix"])
    m, n, _ = A
    ds = dec(out)
    U, V, Ui, Vi = mat(wit["U"]), mat(wit["V"]), mat(wit["U_inv"]), mat(wit["V_inv"])
    for M_, what in ((A, "A"), (U, "U"), (V, "V"), (Ui, "U_inv"), (Vi, "V_inv")):
        need(all_integral(M_), f"{what} is not an integer matrix")
    need(U[0] == U[1] == m and V[0] == V[1] == n, "transform shapes")
    need(same(mmul(U, Ui), eye(m)), "U U_inv != I")
    need(same(mmul(V, Vi), eye(n)), "V V_inv != I")
    D = mmul(mmul(U, A), V)
    need(len(ds) <= min(m, n), "too many invariant factors")
    for i in range(m):
        for j in range(n):
            want = ds[i] if (i == j and i < len(ds)) else 0
            need(D[2][i][j] == want, f"(U A V)[{i}][{j}] = {D[2][i][j]}, expected {want}")
    for i, d in enumerate(ds):
        need(integral(d) and d > 0, "invariant factor must be a positive integer")
        if i + 1 < len(ds):
            need(ds[i + 1] % d == 0, "invariant factors do not divide in order")
    return f"invariant factors {ds}"


def k_linear_solution(inp, out, wit):
    A, b, x, inv = mat(inp["matrix"]), dec(inp["rhs"]), dec(out), mat(wit["inverse"])
    n = A[0]
    need(A[1] == n and inv[0] == inv[1] == n, "shapes")
    need(same(mmul(A, inv), eye(n)), "A A^-1 != I")
    need(len(x) == n and matvec(A, x) == list(b), "A x != b")
    return "unique solution"


def k_solve_enclosure(inp, out, wit):
    A, b, x, R = mat(inp["matrix"]), dec(inp["rhs"]), dec(inp["received"]), mat(inp["inverse"])
    w, dA, db = dec(inp["weights"]), mat(inp["matrix_radii"]), dec(inp["rhs_radii"])
    n = A[0]
    need(A[1] == n and R[0] == R[1] == n and dA[0] == dA[1] == n, "shapes")
    need(len(b) == len(x) == len(w) == len(db) == n, "vector lengths")
    need(all(wi > 0 for wi in w), "weights must be positive")
    need(all(r >= 0 for row in dA[2] for r in row) and all(r >= 0 for r in db), "radii must be nonnegative")
    need(dec(wit["norm"]) == "weighted-max", "unknown norm")
    a, r, da = A[2], R[2], dA[2]
    res = [b[i] - sum(a[i][j] * x[j] for j in range(n)) for i in range(n)]
    defect = [[(1 if i == j else 0) - sum(r[i][k] * a[k][j] for k in range(n)) for j in range(n)]
              for i in range(n)]
    forcing = [abs(sum(r[i][j] * res[j] for j in range(n)))
               + sum(abs(r[i][j]) * (db[j] + sum(da[j][k] * abs(x[k]) for k in range(n)))
                     for j in range(n)) for i in range(n)]
    major = [[abs(defect[i][j]) + sum(abs(r[i][k]) * da[k][j] for k in range(n)) for j in range(n)]
             for i in range(n)]
    weighted = [sum(major[i][j] * w[j] for j in range(n)) for i in range(n)]
    theta = max(weighted[i] / w[i] for i in range(n))
    eta = max(forcing[i] / w[i] for i in range(n))
    status = dec(out["status"])
    boxes = dec(out["boxes"])
    if status == "unknown":
        need(boxes == [], "an unknown result claims no boxes")
        return f"no enclosure claimed (theta={theta})"
    need(status == "proved", f"unknown status {status!r}")
    need(theta < 1, f"theta = {theta} is not below 1")
    need(len(boxes) == n, "box count")
    for i in range(n):
        rad = forcing[i] + weighted[i] * eta / (1 - theta)
        lo, hi = boxes[i]
        need(lo <= x[i] - rad and x[i] + rad <= hi, f"box {i} does not contain the proved enclosure")
    return f"enclosure proved (theta={float(theta):.3g})"


def k_psd(inp, out, wit):
    M = mat(inp["matrix"])
    n = M[0]
    need(M[1] == n, "PSD of a non-square matrix")
    need(all(M[2][i][j] == M[2][j][i] for i in range(n) for j in range(n)), "matrix is not symmetric")
    if dec(out) is True:
        B, D = mat(wit["B"]), dec(wit["D"])
        need(B[1] == n and B[0] == len(D), "factor shapes")
        need(all(d > 0 for d in D), "weights must be positive")
        BtD = (n, B[0], [[B[2][k][i] * D[k] for k in range(B[0])] for i in range(n)])
        need(same(mmul(BtD, B), M), "M != B^T diag(D) B")
        return "positive semidefinite"
    need(dec(out) is False, "output must be a boolean")
    v = dec(wit["v"])
    need(len(v) == n, "direction length")
    q = sum(v[i] * M[2][i][j] * v[j] for i in range(n) for j in range(n))
    need(q < 0, "v^T M v is not negative")
    return f"not PSD (v^T M v = {q})"


def k_integer_gcd(inp, out, wit):
    a, b, g = dec(inp["a"]), dec(inp["b"]), dec(out)
    s, t, qa, qb = (dec(wit[k]) for k in ("s", "t", "a_over_g", "b_over_g"))
    need(all(isinstance(x, int) for x in (a, b, g, s, t, qa, qb)), "integers required")
    need(g >= 0, "gcd must be nonnegative")
    need(s * a + t * b == g, "s a + t b != g")
    need(g * qa == a and g * qb == b, "g does not divide both")
    return f"gcd={g}"


def k_poly_gcd(inp, out, wit):
    a, b, g = ptrim(dec(inp["a"])), ptrim(dec(inp["b"])), ptrim(dec(out))
    s, t, qa, qb = (ptrim(dec(wit[k])) for k in ("s", "t", "a_over_g", "b_over_g"))
    need(g == [] or g[-1] == 1, "gcd must be monic")
    need(peq(padd(pmul(s, a), pmul(t, b)), g), "s a + t b != g")
    need(peq(pmul(g, qa), a) and peq(pmul(g, qb), b), "g does not divide both")
    return f"gcd degree {max(0, len(g) - 1)}"


def expand(e, vars_):
    """Sparse expansion: dict exponent-tuple -> nonzero Fraction."""
    k = len(vars_)
    if "const" in e:
        c = Fraction(dec(e["const"]))
        return {(0,) * k: c} if c != 0 else {}
    if "var" in e:
        need(e["var"] in vars_, f"variable {e['var']!r} not listed")
        ex = [0] * k
        ex[vars_.index(e["var"])] = 1
        return {tuple(ex): Fraction(1)}
    op, args = e["op"], e["args"]

    def add(p, q, sgn=1):
        r = dict(p)
        for m_, c in q.items():
            r[m_] = r.get(m_, 0) + sgn * c
            if r[m_] == 0:
                del r[m_]
        return r

    def mul(p, q):
        r = {}
        for m1, c1 in p.items():
            for m2, c2 in q.items():
                m_ = tuple(x + y for x, y in zip(m1, m2))
                r[m_] = r.get(m_, 0) + c1 * c2
                if r[m_] == 0:
                    del r[m_]
        return r

    parts = [expand(x, vars_) for x in args]
    if op == "+":
        acc = {}
        for p in parts:
            acc = add(acc, p)
        return acc
    if op == "*":
        acc = {(0,) * k: Fraction(1)}
        for p in parts:
            acc = mul(acc, p)
        return acc
    if op == "-":
        need(parts, "empty difference")
        if len(parts) == 1:
            return add({}, parts[0], -1)
        acc = parts[0]
        for p in parts[1:]:
            acc = add(acc, p, -1)
        return acc
    if op == "expt":
        n = dec(e["k"])
        need(isinstance(n, int) and 0 <= n <= 4096 and len(parts) == 1, "bad exponent")
        acc = {(0,) * k: Fraction(1)}
        for _ in range(n):
            acc = mul(acc, parts[0])
        return acc
    raise Reject(f"unknown operator {op!r}")


def collect_vars(e, acc):
    if "var" in e:
        acc.add(e["var"])
    for x in e.get("args", []):
        collect_vars(x, acc)
    return acc


def k_polynomial_identity(inp, out, wit):
    vars_ = list(wit["vars"])
    need(all(isinstance(v, str) for v in vars_) and len(set(vars_)) == len(vars_), "variable list")
    used = collect_vars(inp["lhs"], set()) | collect_vars(inp["rhs"], set())
    need(used <= set(vars_), "an expression uses an unlisted variable")
    l, r = expand(inp["lhs"], vars_), expand(inp["rhs"], vars_)
    diff = dict(l)
    for m_, c in r.items():
        diff[m_] = diff.get(m_, 0) - c
        if diff[m_] == 0:
            del diff[m_]
    claimed = {}
    for term in wit["difference"]:
        ex = tuple(dec(term["e"]))
        need(len(ex) == len(vars_) and ex not in claimed, "difference term exponents")
        c = Fraction(dec(term["c"]))
        need(c != 0, "zero coefficient in the stated difference")
        claimed[ex] = c
    need(claimed == diff, "the stated normal form of lhs - rhs is not the expansion")
    need(dec(out) == (len(diff) == 0), "output disagrees with the expansion")
    return "identity holds" if not diff else f"identity fails ({len(diff)} terms differ)"


def pratt(node, depth=0):
    need(depth < 64, "Pratt tree too deep")
    n = dec(node["n"])
    if n == 2:
        need(set(node) == {"n"}, "the base node 2 carries no data")
        return n
    a, factors = dec(node["a"]), node["factors"]
    need(n > 2 and 1 < a < n, "Pratt node range")
    prod = 1
    for f in factors:
        q, e = dec(f["q"]), dec(f["e"])
        need(isinstance(e, int) and e >= 1, "exponent")
        need(pratt(f["cert"], depth + 1) == q, "sub-certificate is for another number")
        prod *= q ** e
        need(pow(a, (n - 1) // q, n) != 1, f"a^((n-1)/{q}) = 1 mod n")
    need(prod == n - 1, "the factorization does not multiply to n - 1")
    need(pow(a, n - 1, n) == 1, "a^(n-1) != 1 mod n")
    return n


def k_primality(inp, out, wit):
    n, prime = dec(inp["n"]), dec(out)
    need(isinstance(n, int) and isinstance(prime, bool), "types")
    if prime:
        need(pratt(wit["pratt"]) == n, "the Pratt certificate is for another number")
        return f"{n} is prime (Pratt)"
    if "reason" in wit:
        need(dec(wit["reason"]) == "less-than-two" and n < 2, "n < 2 reason")
        return f"{n} < 2"
    if "factor" in wit:
        f = dec(wit["factor"])
        need(1 < f < n and n % f == 0, "not a proper factor")
        return f"{n} = {f} * {n // f}"
    a = dec(wit["miller_rabin_base"])
    need(n > 3 and n % 2 == 1 and 2 <= a <= n - 2, "Miller-Rabin base range")
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    x = pow(a, d, n)
    passes = x in (1, n - 1)
    for _ in range(s - 1):
        if passes:
            break
        x = x * x % n
        passes = x == n - 1
    need(not passes, "the base does not witness compositeness")
    return f"{n} is composite (Miller-Rabin base {a})"


def centered_bound(q, r):
    return sum((abs(c) * r ** i for i, c in enumerate(q) if i >= 1), Fraction(0))


def k_polynomial_range(inp, out, wit):
    p = dec(inp["poly"])
    lo, hi = dec(inp["box"])
    need(lo <= hi, "empty box")
    c, q = dec(wit["center"]), dec(wit["shifted"])
    need(lo <= c <= hi, "centre outside the box")
    # p(x) = sum q_i (x - c)^i, checked by expanding the right side.
    acc, power = [], [Fraction(1)]
    for qi in q:
        acc = padd(acc, [qi * x for x in power])
        power = pmul(power, [-c, Fraction(1)])
    need(peq(acc, p), "the re-expansion about the centre is not the polynomial")
    r = max(hi - c, c - lo)
    s = centered_bound(q, r)
    q0 = q[0] if q else 0
    L, H = dec(out)
    need(L <= q0 - s and q0 + s <= H, "the stated range does not contain the centred-form enclosure")
    return f"range within [{L}, {H}]"


def k_taylor_model_range(inp, out, wit):
    cs, x0, r = dec(inp["coeffs"]), dec(inp["center"]), dec(inp["radius"])
    Rlo, Rhi = dec(inp["remainder"])
    need(r >= 0 and Rlo <= Rhi, "radius or remainder malformed")
    need(dec(wit["rounding"]) == "outward", "unknown rounding")
    need(dec(wit["order"]) == len(cs) - 1, "order does not match the coefficients")
    s = centered_bound(cs, r)
    c0 = cs[0] if cs else 0
    L, H = dec(out)
    need(L <= c0 - s + Rlo and c0 + s + Rhi <= H,
         "the stated range does not contain P([-r, r]) + R")
    return f"range within [{float(L):.17g}, {float(H):.17g}] given the remainder premise"


def k_cocycle(inp, out, wit):
    D, z = mat(inp["coboundary_map"]), dec(inp["cochain"])
    product = matvec(D, z)
    need(dec(wit["product"]) == product, "the stated product is not D z")
    zero = all(x == 0 for x in product)
    need(dec(out) == zero, "output disagrees with D z")
    return "cocycle" if zero else "not a cocycle"


def k_cohomology_class(inp, out, wit):
    D, z = mat(inp["coboundary_map"]), dec(inp["cocycle"])
    m, n, _ = D
    need(all_integral(D) and all(integral(x) for x in z) and len(z) == m, "integer data required")
    verdict = dec(out)
    if verdict == "coboundary":
        w = dec(wit["w"])
        need(all(integral(x) for x in w) and len(w) == n, "w must be an integer vector")
        need(matvec(D, w) == list(z), "D w != z")
        return "coboundary"
    need(verdict == "nontrivial", f"unknown verdict {verdict!r}")
    U, V, Ui, Vi = mat(wit["U"]), mat(wit["V"]), mat(wit["U_inv"]), mat(wit["V_inv"])
    for M_ in (U, V, Ui, Vi):
        need(all_integral(M_), "transforms must be integer matrices")
    need(same(mmul(U, Ui), eye(m)) and same(mmul(V, Vi), eye(n)), "transforms are not invertible over Z")
    ds = dec(wit["diagonal"])
    need(all(integral(d) and d > 0 for d in ds), "diagonal entries must be positive integers")
    S = mmul(mmul(U, D), V)
    for i in range(m):
        for j in range(n):
            need(S[2][i][j] == (ds[i] if i == j and i < len(ds) else 0), "U D V is not the stated diagonal")
    i = dec(wit["index"])
    need(isinstance(i, int) and 0 <= i < m, "index out of range")
    y = matvec(U, z)
    if i < len(ds):
        need(y[i] % ds[i] != 0, "(U z)_i is a multiple of d_i")
    else:
        need(y[i] != 0, "(U z)_i is zero past the rank")
    return "not a coboundary over Z"


KINDS = {
    "rank": k_rank,
    "determinant": k_determinant,
    "smith_normal_form": k_smith,
    "linear_solution": k_linear_solution,
    "linear_solve_enclosure": k_solve_enclosure,
    "psd": k_psd,
    "integer_gcd": k_integer_gcd,
    "poly_gcd": k_poly_gcd,
    "polynomial_identity": k_polynomial_identity,
    "primality": k_primality,
    "polynomial_range": k_polynomial_range,
    "taylor_model_range": k_taylor_model_range,
    "cocycle": k_cocycle,
    "cohomology_class": k_cohomology_class,
}

# The premises a kind may carry: fixed here, never taken from the claim.
PREMISES = {"taylor_model_range": ["remainder_bound"]}

FIELDS = ["kind", "label", "inputs", "output", "witness", "premises", "id", "hash"]


# ---------------------------------------------------------------------------
# Identifiers and documents
# ---------------------------------------------------------------------------

def canonical(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def fnv1a_64(text):
    h = 0xcbf29ce484222325
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")


def claim_id(claim):
    body = {k: claim[k] for k in FIELDS[:6]}
    return "sha256:" + hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def claim_hash(claim):
    return fnv1a_64(canonical({k: claim[k] for k in FIELDS[:7]}))


def check_claim(claim):
    """Returns (verdict, detail); verdict is ACCEPTED, CONDITIONAL or REJECTED."""
    try:
        need(isinstance(claim, dict) and list(claim.keys()) == FIELDS, "claim fields or their order")
        need(claim["id"] == claim_id(claim), "id does not match the content")
        need(claim["hash"] == claim_hash(claim), "hash does not match the content")
        kind = claim["kind"]
        need(kind in KINDS, f"unknown kind {kind!r}")
        names = [p.get("name") for p in claim["premises"]] if isinstance(claim["premises"], list) else None
        need(names == PREMISES.get(kind, []), "premises differ from the kind's fixed premises")
        need(isinstance(claim["inputs"], dict) and isinstance(claim["witness"], dict), "inputs/witness")
        detail = KINDS[kind](claim["inputs"], claim["output"], claim["witness"])
        return ("CONDITIONAL" if names else "ACCEPTED", detail)
    except Reject as e:
        return ("REJECTED", str(e))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError, AttributeError) as e:
        return ("REJECTED", f"malformed claim: {type(e).__name__}: {e}")


def no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key in JSON object")
    return dict(pairs)


def reject_constant(name):
    raise ValueError(f"JSON constant {name} is not allowed")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)


def check_document(doc, name, strict=False, out=sys.stdout):
    if not (isinstance(doc, dict) and doc.get("schema") == SCHEMA and isinstance(doc.get("claims"), list)):
        print(f"{name}: REJECTED document: schema must be {SCHEMA} with a claims list", file=out)
        return False
    claims = doc["claims"]
    print(f"{name}: program={doc.get('program')!r} claims={len(claims)}", file=out)
    ok = bool(claims)
    if not claims:
        print("  REJECTED: a certificate document with no claims", file=out)
    for i, claim in enumerate(claims):
        verdict, detail = check_claim(claim)
        kind = claim.get("kind", "?") if isinstance(claim, dict) else "?"
        label = claim.get("label", f"#{i}") if isinstance(claim, dict) else f"#{i}"
        print(f"  {verdict} kind={kind} label={label!r}: {detail}", file=out)
        if verdict == "REJECTED" or (strict and verdict == "CONDITIONAL"):
            ok = False
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independent checker for Eshkol certificates (ADR-0047).")
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--strict", action="store_true", help="count CONDITIONAL claims as failures")
    ap.add_argument("--trusted-base", action="store_true", help="print the trusted-base statement")
    args = ap.parse_args(argv)
    if args.trusted_base:
        print(TRUSTED_BASE, end="")
        return 0
    files = list(args.files) + (sorted(args.dir.glob("*.json")) if args.dir else [])
    if not files:
        ap.error("no input (FILE.json ... or --dir DIR)")
    ok = True
    for f in files:
        try:
            doc = load(f)
        except (OSError, ValueError) as e:
            print(f"{f}: REJECTED unreadable document: {e}")
            ok = False
            continue
        ok = check_document(doc, str(f), args.strict) and ok
    print("RESULT: ALL ACCEPTED" if ok else "RESULT: REJECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
