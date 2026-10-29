#!/usr/bin/env python3
"""Pure-Python exact linear algebra used to independently recompute the
mathematical claims recorded by core.transcript (lib/core/transcript.esk).

No third-party dependencies (no numpy, no sympy): this module backs
replay_check.py, which CTest runs in CI with only the Python standard
library.  crosscheck_pari_sympy.py (atlas-only) additionally compares
against PARI/GP and SymPy as the independent-tool cross-check; this module
is a THIRD, independently-written implementation of the same algorithms,
used as the always-available CI-side replay.

Every function takes and returns plain Python int / fractions.Fraction so
precision is exact and unbounded, matching Eshkol's exact-number semantics.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import List, Sequence


Matrix = List[List[int]]


def fnv1a_64(text: str) -> int:
    """FNV-1a 64-bit over the UTF-8 bytes of text.  Matches
    lib/core/merkle.c's eshkol_fnv1a_64 (offset 0xcbf29ce484222325,
    prime 0x100000001b3, uint64 wraparound)."""
    h = 0xcbf29ce484222325
    prime = 0x100000001b3
    mask = (1 << 64) - 1
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * prime) & mask
    return h


def hash_hex16(h: int) -> str:
    """16-char lowercase hex of the low 64 bits of h, matching
    lib/core/merkle.c's eshkol_u64_to_hex16 / hash->hex."""
    return format(h & ((1 << 64) - 1), "016x")


# ---------------------------------------------------------------------------
# Determinant: fraction-free (Bareiss) elimination, exact for int or
# Fraction entries, bignum-safe (Python ints are already bignums).
# ---------------------------------------------------------------------------

def determinant(rows: Sequence[Sequence]) -> Fraction:
    n = len(rows)
    if n == 0:
        return Fraction(1)
    M = [[Fraction(x) for x in row] for row in rows]
    sign = 1
    prev_pivot = Fraction(1)
    for k in range(n - 1):
        if M[k][k] == 0:
            swap = None
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    swap = i
                    break
            if swap is None:
                return Fraction(0)
            M[k], M[swap] = M[swap], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) / prev_pivot
            M[i][k] = Fraction(0)
        prev_pivot = M[k][k]
    det = sign * M[n - 1][n - 1]
    return det


def determinant_exact_int(rows: Sequence[Sequence[int]]) -> int:
    d = determinant(rows)
    assert d.denominator == 1, "determinant of an integer matrix must be an integer"
    return int(d)


# ---------------------------------------------------------------------------
# Rank mod p: Gaussian elimination over GF(p), p prime.
# ---------------------------------------------------------------------------

def rank_mod_p(rows: Sequence[Sequence[int]], p: int) -> int:
    if not rows:
        return 0
    R = [[x % p for x in row] for row in rows]
    m, n = len(R), len(R[0])
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if R[i][c] % p != 0:
                piv = i
                break
        if piv is None:
            continue
        R[r], R[piv] = R[piv], R[r]
        inv = pow(R[r][c], -1, p)
        for i in range(m):
            if i != r and R[i][c] % p != 0:
                f = (R[i][c] * inv) % p
                for j in range(c, n):
                    R[i][j] = (R[i][j] - f * R[r][j]) % p
        r += 1
        if r == m:
            break
    return r


# ---------------------------------------------------------------------------
# Smith normal form over Z: returns the nonzero diagonal invariant
# factors, canonically sorted ascending (d_1 | d_2 | ... | d_k).  Sort
# order is the correct canonical form regardless of which pivoting path
# an implementation takes, so this is safe to compare against any other
# correct SNF implementation's nonzero-diagonal output.
# ---------------------------------------------------------------------------

def _min_abs_nonzero(M: List[List[int]], t: int, m: int, n: int):
    """Position of a nonzero entry of smallest absolute value in the
    submatrix M[t:, t:], or None if the submatrix is all zero.  Picking
    the SMALLEST magnitude (not just any nonzero entry) each time a
    pivot stops being a divisor of its row/column is what guarantees
    termination: the pivot's absolute value strictly decreases whenever
    it changes, and positive integers cannot decrease forever."""
    best = None
    for i in range(t, m):
        for j in range(t, n):
            v = M[i][j]
            if v != 0 and (best is None or abs(v) < abs(M[best[0]][best[1]])):
                best = (i, j)
    return best


def smith_normal_form_diagonal(rows: Sequence[Sequence[int]]) -> List[int]:
    if not rows or not rows[0]:
        return []
    M = [list(row) for row in rows]
    m, n = len(M), len(M[0])
    diag: List[int] = []
    t = 0
    while t < m and t < n:
        piv = _min_abs_nonzero(M, t, m, n)
        if piv is None:
            break
        pi, pj = piv
        M[t], M[pi] = M[pi], M[t]
        for row in M:
            row[t], row[pj] = row[pj], row[t]
        while True:
            pivot = M[t][t]
            clean = True
            for i in range(t + 1, m):
                a = M[i][t]
                if a != 0:
                    q = a // pivot
                    for j in range(t, n):
                        M[i][j] -= q * M[t][j]
                    if M[i][t] != 0:
                        clean = False
            for j in range(t + 1, n):
                a = M[t][j]
                if a != 0:
                    q = a // pivot
                    for i in range(t, m):
                        M[i][j] -= q * M[i][t]
                    if M[t][j] != 0:
                        clean = False
            if clean:
                bad = None
                for i in range(t + 1, m):
                    for j in range(t + 1, n):
                        if M[i][j] % pivot != 0:
                            bad = (i, j)
                            break
                    if bad:
                        break
                if bad is None:
                    break
                bi, _ = bad
                for j in range(n):
                    M[t][j] += M[bi][j]
            else:
                # The pivot no longer divides every entry in its row or
                # column: re-pick the smallest-magnitude nonzero entry in
                # the submatrix (strictly smaller than the current pivot,
                # since that entry's reduced remainder is what broke
                # cleanliness) and swap it into the (t, t) corner.
                ni, nj = _min_abs_nonzero(M, t, m, n)
                M[t], M[ni] = M[ni], M[t]
                for row in M:
                    row[t], row[nj] = row[nj], row[t]
        if M[t][t] != 0:
            diag.append(abs(M[t][t]))
        t += 1
    return sorted(diag)


# ---------------------------------------------------------------------------
# Polynomial gcd over Z[x]: Euclid over Q[x] (coefficients as Fraction),
# then clear denominators and divide out the content gcd, matching the
# "primitive, positive leading coefficient" convention used by
# core.multimodular's tests.  Coefficients are LOWEST degree first.
# ---------------------------------------------------------------------------

def _poly_trim(p: List[Fraction]) -> List[Fraction]:
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def _poly_divmod(a: List[Fraction], b: List[Fraction]):
    a = _poly_trim(a)
    b = _poly_trim(b)
    if not b:
        raise ZeroDivisionError("poly division by zero")
    q = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    r = list(a)
    while len(_poly_trim(r)) >= len(b) and _poly_trim(r):
        r = _poly_trim(r)
        shift = len(r) - len(b)
        coeff = r[-1] / b[-1]
        q[shift] = coeff
        for i, bc in enumerate(b):
            r[shift + i] -= coeff * bc
        r = _poly_trim(r)
    return _poly_trim(q), r


def _content(xs: Sequence[int]) -> int:
    c = 0
    for x in xs:
        c = gcd(c, abs(int(x)))
    return c


def poly_gcd_z(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """a, b: integer coefficient lists, lowest degree first.  Returns the
    Z[x] gcd's coefficient list, lowest degree first, with positive
    leading coefficient (the zero polynomial is []): the content gcd of
    a and b times the primitive part of their Q[x] Euclidean gcd -- the
    same convention the multimodular suite's own Euclid-over-Q[x]
    reference uses (content gcd * primitive part), which also covers the
    one-side-zero case correctly (gcd(0, b) = b, sign-normalized, since
    content(0) = 0 and gcd(0, content(b)) = content(b))."""
    cg = gcd(_content(a), _content(b))
    A = _poly_trim([Fraction(x) for x in a])
    B = _poly_trim([Fraction(x) for x in b])
    while B:
        _, r = _poly_divmod(A, B)
        A, B = B, r
    prim = _primitive_positive(A)
    if not prim:
        return []
    return [c * cg for c in prim]


def _primitive_positive(p: List[Fraction]) -> List[int]:
    p = _poly_trim(p)
    if not p:
        return []
    denom = 1
    for c in p:
        denom = denom * c.denominator // gcd(denom, c.denominator)
    ints = [int(c * denom) for c in p]
    content = 0
    for c in ints:
        content = gcd(content, abs(c))
    if content == 0:
        content = 1
    ints = [c // content for c in ints]
    if ints[-1] < 0:
        ints = [-c for c in ints]
    return ints
