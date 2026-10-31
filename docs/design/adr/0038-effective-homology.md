---
kind: explanation
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/effective.esk
  - lib/core/effective/chain.esk
  - lib/core/effective/perturbation.esk
  - lib/core/effective/simplicial.esk
  - lib/core/effective/homotopy.esk
  - lib/core/effective/hypercover.esk
  - tests/stdlib/effective_test.esk
  - tests/stdlib/effective_homotopy_test.esk
  - tests/stdlib/effective_hypercover_test.esk
  - tests/stdlib/effective_pi6_s3_test.esk
---
# ADR-0038: `core.effective` — effective homology, homotopy groups and hypercover descent

**Status:** Proposed
**Date:** 2026-10-30
**Scope:** reductions and the basic perturbation lemma; reductions from
discrete vector fields; objects with effective homology (products, twisted
cartesian products, classifying spaces `W-bar G`, Eilenberg-MacLane spaces
`K(Z, n)` and `K(Z/m, n)`); cohomology classes as simplicial maps into
`K(pi, n)`; the Whitehead tower and homotopy groups of simply connected
simplicial sets of finite type; hypercovers of finite spaces and descent for
mapping spaces into 2-types and 3-types (stage S6.3 of the S6 definition of
done). Built on `core.simplicial`, `core.homological` (ADR-0037), the finite
spaces of `core.sheaf` (ADR-0033) and `core.integer_linalg`.

## Context

`core.simplicial` and `core.homological` compute with simplicial sets of
finite type. The spaces of homotopy theory are not of finite type: `K(Z,1)`
has `Z^n` as its n-simplices, `K(Z,2) = W-bar K(Z,1)` and the stages of a
Whitehead tower are twisted products with such fibres. Their homology is
computable only through effective homology (Sergeraert, Rubio; the Kenzo
program): a reduction from the chains of the space to a complex of finite
type, obtained by perturbation from reductions that are known. ADR-0037
lists this as the next layer. S6.3 asks, on top of the same substrate, for a
hypercover that restores descent where the Cech nerve of program 19 fails,
and for a 3-type with a nontrivial second k-invariant computed on both sides
of a cover of a 4-dimensional finite model.

An ICC survey (`search-chunks` and `duplicate-implementations` over `lib/`
and `examples/` for perturbation, reductions, Eilenberg-Zilber, shuffles,
twisted products, Whitehead and Postnikov towers, hypercovers) found no
existing implementation: the examples use finite simplicial sets only, and
program 19 stops at the Cech nerve. The Alexander-Whitney front and back
faces, simplicial operators, products and cup products already exist in
`core.simplicial` / `core.homological` and are used as they are; the one
helper the twisted products need from `core.simplicial.sset` (the normal
form of a pair of simplices, `simplicial-product-simplex`) is exported there
rather than written a second time.

## Decision

1. **One reduction type, checked by its identities.** A reduction
   `(f, g, h) : C => D` between complexes whose cells are arbitrary values
   (`#(effective-complex name basis d)`, `basis` returning `#f` in degrees
   that are not enumerated) carries memoized cell-level maps;
   `reduction-violations` checks `f g = 1`, `g f + d h + h d = 1`,
   `f h = h g = h h = 0` and the chain-map conditions exactly on given
   cells. Every construction is a composition of reductions, so an object
   with effective homology is a reduction from its normalized chains, never
   a pair of equivalences.

2. **The basic perturbation lemma with its hypothesis checked.**
   `perturb-reduction` sums `Sigma = sum (-1)^i (delta h)^i delta` cell by
   cell and returns the perturbed reduction; a caller that knows which
   filtration the perturbation lowers passes it, and every term of
   `delta(c)` is checked to lower it (word length for bar constructions, the
   base degree by two for twisted products over 1-reduced bases): the
   nilpotency hypothesis is verified on every cell the computation uses,
   and a series that does not terminate is refused by name.

3. **Discrete vector fields produce the fundamental reductions.** One
   facility, `vector-field-reduction` (Romero-Sergeraert Theorem 19 with
   formulas (18) and (20)), turns an admissible discrete vector field into
   a reduction onto its critical cells; incidences are checked to be units
   and a V-path that returns to a cell is refused. The Eilenberg-Zilber
   reduction is the field of Definitions 41-43 (a diagonal step read
   backwards makes a source, a vertical step followed by a horizontal one a
   target, the last simplex is critical); the same field on a twisted
   product gives the twisted Eilenberg-Zilber reduction (it never uses the
   0-face); `K(Z,1) => C(S^1)` is Proposition 26; the classifying space
   `W-bar G` carries the field that reads `b` as `(d_0 b, x_1)` in
   `W-bar G x_tau G`, applies the Eilenberg-Zilber status and recurses on
   the base when the pair is critical (Kenzo's classifying-space field), so
   its critical complex is the bar construction of `C(G)`. Finite complexes
   are reduced by the gradient of a sequence of collapses. The explicit AW,
   EML and SHI formulas are implemented too and agree with the vector field
   map by map on every prism checked.

4. **Tensor products and bar constructions as words.** Cells of tensor
   products and bar constructions are words of letters `(deg . cell)`;
   the tensor product of reductions uses `h = sum gf ... gf h 1 ... 1` with
   Koszul signs. The bar construction lists the innermost component of a
   `W-bar G` simplex first and uses the sign convention of the critical
   complex (`(-1)^k` times the unshifted Koszul differential, the same
   factor on the homotopy), so the length-preserving part of the computed
   critical differential is exactly the tensor differential and the
   product part is a perturbation lowering the length.

5. **Simplicial groups and classifying spaces in Kenzo's representation.**
   `K(Z/m,1)` (m = 0 for Z) are words of nonzero residues; `W-bar G`
   follows May 21.7 with components listed from the top; simplices are
   `core.simplicial` normal forms `#(simplex eta x)` throughout, so the
   simplicial operators of ADR-0037 act on every space built here.
   `K(pi, n)` is `W-bar^(n-1) K(pi, 1)`, its effective homology the
   composite `C(W-bar G) => Bar(C G) => Bar(E_G)`.

6. **Cohomology classes enter through the minimal models.** A normalized
   n-cocycle `z` is the map `sigma -> z(sigma^*)` into the minimal
   `K(pi, n)`; the twisting of the minimal models `t y(J) = y(0, J+1) -
   [0 not in J] y(1, J+1)` (the cone on vertex 0 corrected by the cone of
   `d_0 y`, so that `t y` is again a cocycle) carries it into
   `W-bar^(n-1) K(pi, 1)`, and its first component is the twisting function
   of the principal fibration classified by `z`.

7. **Whitehead tower.** `pi_k = H_k(X_k)` by Hurewicz; the cocycle dual to
   a cyclic `H_k` is read from the Smith normal form of the effective
   complex and pulled back through the reduction's `f`; `X_(k+1) = X_k
   x_tau K(pi_k, k-1)`. Non-cyclic `pi_k` (a product of fibres) is planned
   for v1.4.1.

8. **Hypercovers by cones, built through coskeleta.** Level n of the
   hypercover of a finite T_0 space has one piece for every compatible
   boundary that is not the boundary of a degenerate simplex and every
   maximal point of the intersection of its faces' opens (the minimal open
   of a point is a cone). The same recursion with connected components
   instead of maximal points is the Cech nerve as a simplicial set. Pieces
   are numbered, with their faces in a table, so equality is cheap at every
   level. Mapping-space cardinalities come from integral homology of the
   effective complex by universal coefficients; for 3-types
   `K(B,3) x_k K(A,2)` the fibration formula, with the class of `k(x)`
   decided by evaluating the cup product (Alexander-Whitney on the simplices
   of the space, the cocycle pulled back by `f`) on the 4-cycles `g(z)`.

## Consequences

- Homotopy groups on Linux x64 (old-donkey, one core, the release build of
  the shared base): pi_3(S^2) = Z, pi_4(S^2) = Z/2 (0.36 s), pi_5(S^2) = Z/2
  (5.8 s), pi_4(S^3) = Z/2 (0.09 s), pi_5(S^3) = Z/2 (2.0 s), pi_6(S^3) = Z/12
  (146 s, 32.5 GB peak resident memory: every memoized cell is kept), after
  about 40 s of JIT compilation of the libraries.
- S6.3: the hypercover of the pseudo-2-sphere refining its maximal-open
  cover has (2, 4, 28, 196) nondegenerate pieces through level 3, satisfies
  the hypercover conditions, and gives `Map(-, BG)` = 1/2, 4, 2 for BZ/2,
  K(Z/2,2) and the split 2-type, as the space does, where the Cech nerve
  gives 1/2, 2, 1; the state sum on the hypercover gives 4 for K(Z/2,2).
- Memory is the bound: memo tables keep every cell a computation touched.
  Releasing tables between tower stages is planned for v1.4.1.
- The libraries are ordinary modules outside the `stdlib` closure; the VM
  prelude and binding tables are unchanged.

## Alternatives considered

- **Kenzo's perturbation-only proofs of the classifying-space reduction.**
  Rejected for the vector-field route: one facility proves EZ, twisted EZ,
  `K(Z,1)` and `W-bar G`, and its admissibility and incidence conditions
  are checked on use.
- **Small fibres by hand (Cartan's constructions).** Rejected: the twisting
  cochains of the tower must come from the actual simplicial sets.
- **Minimal (cocycle) models of K(pi, n) as the fibres.** Kept for the
  classifying maps only; the bar-construction reduction needs `W-bar`.
- **Semi-simplicial (ordered) hypercovers.** Rejected: the simplicial
  version excludes degenerate boundaries and is smaller.

## Verification

- `tests/stdlib/effective_test.esk` (CTest `effective_jit`, `effective_aot`).
- `tests/stdlib/effective_homotopy_test.esk` (CTest `effective_homotopy_jit`,
  `effective_homotopy_aot`).
- `tests/stdlib/effective_hypercover_test.esk` (CTest
  `effective_hypercover_jit`, `effective_hypercover_aot`).
- `tests/stdlib/effective_pi6_s3_test.esk` (CTest `effective_pi6_s3_jit`).
- `scripts/run_effective_gate.sh` emits the criterion `v14_effective_homology`.

## References

F. Sergeraert, J. Rubio, "Constructive algebraic topology", Bull. Sci. Math.
126 (2002); J. Rubio, F. Sergeraert, Genova lecture notes (2006); X. Dousson,
J. Rubio, F. Sergeraert, Y. Siret, the Kenzo program; A. Romero, F.
Sergeraert, "Discrete vector fields and fundamental algebraic topology",
arXiv:1005.5685 (2010); J. A. Delgado, A. Romero, J. Rubio, F. Sergeraert,
"On the role of discrete vector fields on classifying spaces"; W. Shih,
Publ. IHES 13 (1962); R. Brown, "The twisted Eilenberg-Zilber theorem"
(1967); S. Eilenberg, S. Mac Lane, Ann. of Math. 58 (1953), 60 (1954);
P. Real, HHA 2 (2000); R. Forman, Adv. Math. 134 (1998); J. P. May,
*Simplicial Objects in Algebraic Topology* (1967); G. W. Whitehead, PNAS 38
(1952); J.-P. Serre, Ann. of Math. 54 (1951); J. Lurie, *Higher Topos Theory*
(2009) 6.5.3-6.5.4; D. Dugger, S. Hollander, D. Isaksen, Math. Proc. Camb.
Phil. Soc. 136 (2004); D. Dugger, D. Isaksen, Math. Z. 246 (2004);
M. C. McCord, Duke Math. J. 33 (1966); J. Faria Martins, T. Porter, TAC 18
(2007); nLab, "effective homology", "homological perturbation theory",
"Whitehead tower", "hypercover", "Eilenberg-MacLane space".
