#!/usr/bin/env python3
"""Regenerate the certificate block of SmithNormalForm.v.

For each integer matrix A it computes unimodular U, V together with their
inverses U', V' such that U A V = D is in Smith normal form, using exact
integer row and column operations, and checks the four identities in Python.
Rocq re-checks every identity by computation; this script only proposes the
certificates.

Usage: gen_snf_certificates.py [--check]   (--check: exit 1 if the file differs)
"""
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "SmithNormalForm.v"
BEGIN = "(* BEGIN GENERATED CERTIFICATES *)"
END = "(* END GENERATED CERTIFICATES *)"

# the six matrices of [Smith normal form#1/#2] in tests/nlab/nlab_ring_test.esk
SNF_CASES = [
    [[2, 4, 4], [-6, 6, 12], [10, -4, -16]],
    [[2, 4], [6, 8]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[6, 0, 0], [0, 10, 0], [0, 0, 15]],
    [[0, 3, 0, 6], [9, 0, 12, 0], [0, 0, 0, 0]],
    [[12345678901, 2], [3, 4]],
]

# the 6-vertex real projective plane of tests/nlab/nlab_space_test.esk
RP2_FACETS = [(1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 6), (1, 6, 2),
              (2, 3, 5), (3, 4, 6), (4, 5, 2), (5, 6, 3), (6, 2, 4)]


def identity(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def mul(A, B):
    k = len(B)
    cols = len(B[0]) if B else 0
    return [[sum(A[i][l] * B[l][j] for l in range(k)) for j in range(cols)] for i in range(len(A))]


def snf(A):
    m, n = len(A), len(A[0])
    D = [row[:] for row in A]
    U, Ui, V, Vi = identity(m), identity(m), identity(n), identity(n)

    def row_add(i, j, c):  # row i += c * row j
        for M in (D, U):
            M[i] = [a + c * b for a, b in zip(M[i], M[j])]
        for r in Ui:  # column j of U' -= c * column i
            r[j] -= c * r[i]

    def row_swap(i, j):
        for M in (D, U):
            M[i], M[j] = M[j], M[i]
        for r in Ui:
            r[i], r[j] = r[j], r[i]

    def row_neg(i):
        for M in (D, U):
            M[i] = [-a for a in M[i]]
        for r in Ui:
            r[i] = -r[i]

    def col_add(i, j, c):  # column i += c * column j
        for M in (D, V):
            for r in M:
                r[i] += c * r[j]
        Vi[j] = [a - c * b for a, b in zip(Vi[j], Vi[i])]

    def col_swap(i, j):
        for M in (D, V):
            for r in M:
                r[i], r[j] = r[j], r[i]
        Vi[i], Vi[j] = Vi[j], Vi[i]

    for t in range(min(m, n)):
        while True:
            nz = [(abs(D[i][j]), i, j) for i in range(t, m) for j in range(t, n) if D[i][j]]
            if not nz:
                break
            _, i, j = min(nz)
            row_swap(t, i)
            col_swap(t, j)
            p = D[t][t]
            done = True
            for i in range(t + 1, m):
                q = D[i][t] // p
                if q:
                    row_add(i, t, -q)
                if D[i][t]:
                    done = False
            for j in range(t + 1, n):
                q = D[t][j] // p
                if q:
                    col_add(j, t, -q)
                if D[t][j]:
                    done = False
            if not done:
                continue
            bad = [i for i in range(t + 1, m) for j in range(t + 1, n) if D[i][j] % p]
            if bad:
                row_add(t, bad[0], 1)
                continue
            break
        if D[t][t] < 0:
            row_neg(t)
    assert mul(mul(U, A), V) == D
    assert mul(Ui, U) == identity(m) and mul(U, Ui) == identity(m)
    assert mul(V, Vi) == identity(n) and mul(Vi, V) == identity(n)
    diag = [D[i][i] for i in range(min(m, n))]
    assert all(D[i][j] == 0 for i in range(m) for j in range(n) if i != j)
    nz = [d for d in diag if d]
    assert all(b % a == 0 for a, b in zip(nz, nz[1:]))
    return U, Ui, V, Vi, D


def boundary_matrices(facets):
    tris = sorted(set(tuple(sorted(f)) for f in facets))
    edges = sorted(set(e for t in tris for e in combinations(t, 2)))
    verts = sorted(set(v for t in tris for v in t))
    d1 = [[0] * len(edges) for _ in verts]
    for c, (a, b) in enumerate(edges):
        d1[verts.index(b)][c] += 1
        d1[verts.index(a)][c] -= 1
    d2 = [[0] * len(tris) for _ in edges]
    for c, (a, b, x) in enumerate(tris):
        for sign, e in ((1, (b, x)), (-1, (a, x)), (1, (a, b))):
            d2[edges.index(e)][c] += sign
    assert mul(d1, d2) == [[0] * len(tris) for _ in verts]
    return verts, edges, tris, d1, d2


def det(M):
    """Bareiss fraction-free determinant."""
    M = [r[:] for r in M]
    n, sign, prev = len(M), 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if M[i][k]), None)
            if sw is None:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def adjugate(B):
    n = len(B)
    minor = lambda i, j: [r[:j] + r[j + 1:] for k, r in enumerate(B) if k != i]
    return [[(-1) ** (i + j) * det(minor(j, i)) for j in range(n)] for i in range(n)]


def tail_certificate(d1, Ui, rank):
    """L, c with (L . d1 . U')[r][c] = c0 if c == rank + r else 0, for the
    columns of d1 . U' beyond the rank of d2."""
    M = mul(d1, Ui)
    m, k = len(M), len(M[0]) - rank
    assert all(M[i][c] == 0 for i in range(m) for c in range(rank))
    for rows in combinations(range(m), k):
        B = [[M[i][rank + c] for c in range(k)] for i in rows]
        c0 = det(B)
        if c0:
            adj = adjugate(B)
            L = [[0] * m for _ in range(k)]
            for r in range(k):
                for t, i in enumerate(rows):
                    L[r][i] = adj[r][t]
            E = mul(L, M)
            assert E == [[c0 if c == rank + r else 0 for c in range(len(M[0]))] for r in range(k)]
            return L, c0
    raise AssertionError("no nonsingular minor")


def rocq_list(xs):
    return "[" + "; ".join(str(x) for x in xs) + "]"


def rocq_matrix(M):
    rows = ["[" + "; ".join(str(x) for x in r) + "]" for r in M]
    return "[" + ";\n    ".join(rows) + "]"


def block():
    out = [BEGIN]
    for k, A in enumerate(SNF_CASES, 1):
        U, Ui, V, Vi, D = snf(A)
        out.append(f"(* case {k}: diagonal {[D[i][i] for i in range(min(len(A), len(A[0])))]} *)")
        m, n = len(A), len(A[0])
        diag = [D[i][i] for i in range(min(m, n))]
        nz = [x for x in diag if x]
        for name, M in (("A", A), ("U", U), ("Ui", Ui), ("V", V), ("Vi", Vi), ("D", D)):
            out.append(f"Definition snf{k}_{name} : list (list Z) :=\n  {rocq_matrix(M)}.")
        args = " ".join(f"(ofL snf{k}_{x})" for x in ("A", "U", "Ui", "V", "Vi", "D"))
        out.append(f"""Lemma snf{k}_check : check_cert {m} {n} {args} = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf{k}_cert : snf_cert {m} {n} {args} /\\ diagonal {m} {n} (ofL snf{k}_D).
Proof. apply check_cert_sound, snf{k}_check. Qed.
Theorem snf{k}_diagonal : diag_list {m} {n} (ofL snf{k}_D) = {rocq_list(diag)}.
Proof. vm_compute. reflexivity. Qed.
Theorem snf{k}_chain : chainb {rocq_list(nz)} = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf{k}_first_invariant_factor :
  forall d, divides_all {m} {n} d (ofL snf{k}_A) <-> (d | ofL snf{k}_D 0%nat 0%nat).
Proof.
  destruct snf{k}_cert as [C Hd].
  exact (first_invariant_factor _ _ _ _ _ _ _ _ C Hd ltac:(lia) ltac:(lia)
           (first_divides_sound {m} {n} (ofL snf{k}_D) ltac:(vm_compute; reflexivity))).
Qed.""")
    verts, edges, tris, d1, d2 = boundary_matrices(RP2_FACETS)
    out.append(f"(* RP^2: {len(verts)} vertices, {len(edges)} edges, {len(tris)} triangles *)")
    for tag, A in (("d1", d1), ("d2", d2)):
        U, Ui, V, Vi, D = snf(A)
        out.append(f"(* {tag}: invariant factors {[D[i][i] for i in range(min(len(A), len(A[0]))) if D[i][i]]} *)")
        for name, M in (("", A), ("U", U), ("Ui", Ui), ("V", V), ("Vi", Vi), ("D", D)):
            out.append(f"Definition rp2_{tag}{name} : list (list Z) :=\n  {rocq_matrix(M)}.")
    U2, Ui2, V2, Vi2, D2 = snf(d2)
    rank2 = sum(1 for i in range(min(len(d2), len(d2[0]))) if D2[i][i])
    L, c0 = tail_certificate(d1, Ui2, rank2)
    out.append(f"(* columns {rank2}.. of d1 . d2U' are independent: L . d1 . d2U' = {c0} on them *)")
    out.append(f"Definition rp2_L : list (list Z) :=\n  {rocq_matrix(L)}.")
    out.append(f"Definition rp2_c0 : Z := {c0}.")
    out.append(END)
    return "\n".join(out)


def main():
    text = TARGET.read_text()
    a, b = text.index(BEGIN), text.index(END) + len(END)
    new = text[:a] + block() + text[b:]
    if "--check" in sys.argv[1:]:
        if new != text:
            print("SmithNormalForm.v certificates are not current; run gen_snf_certificates.py")
            return 1
        print("SmithNormalForm.v certificates current")
        return 0
    TARGET.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
