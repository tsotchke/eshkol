---
kind: explanation
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/simplicial.esk
  - lib/core/simplicial/sset.esk
  - lib/core/simplicial/group.esk
  - lib/core/simplicial/nerve.esk
  - lib/core/simplicial/kan.esk
  - lib/core/homological.esk
  - lib/core/homological/complex.esk
  - lib/core/homological/abelian.esk
  - lib/core/homological/spectral.esk
  - lib/core/homological/simplicial.esk
  - lib/core/integer_linalg.esk
  - tests/stdlib/simplicial_test.esk
  - tests/stdlib/homological_test.esk
---
# ADR-0037: `core.simplicial` and `core.homological` — the algebraic-topology substrate

**Status:** Proposed
**Date:** 2026-10-29
**Scope:** the representation of simplicial sets, nerves, crossed modules,
horns and simplicial maps; chain and cochain complexes over ring carriers
and everything computed from them (homology with torsion, chain maps and
cones, double complexes, Ext and Tor, spectral sequences, cup and cup-i
products, Steenrod squares, group cohomology); the boundary with
`core.sheaf` (ADR-0033), `core.ring` (ADR-0034) and `core.integer_linalg`.

## Context

The mathematics examples compute classifying spaces of finite groups,
nerves of crossed modules and posets, horn fillers, Kan's homotopy groups,
cohomology rings by cup products, Bocksteins, Postnikov k-invariants,
Dijkgraaf–Witten and Yetter state sums, descent double complexes and
hypercohomology. A survey with ICC (`search-chunks`, `find-symbol`,
`duplicate-implementations` over `examples/`) found the same machinery
rebuilt in each program on top of `(require stdlib)`: a multiplication-table
group type with cyclic, product, dihedral, quaternion and symmetric groups
(six programs), the face map of the bar construction and its normalized and
unnormalized boundary matrices (five), a Smith normal form with or without
transforms (seventeen; the sheaf library already owned one), Gaussian
elimination mod p (seven), crossed modules with their nerve enumeration,
face and degeneracy (eight), horn tables and Kan's homotopy relation (one),
order complexes and the cosimplicial replacement (seven), 3-manifold
Delta-complexes built from tetrahedra (eight) and the flat-colouring search
behind every state sum (eight). The copies drift: one bar construction indexes tuples with the first
coordinate fastest and another lexicographically; the crossed-module nerve
is enumerated over all simplices in one program and only over horns cut out
of existing simplices in another, so "every horn has a filler" was checked
only on horns that already had one.

S6.3 of the S6 definition of done (hypercovers, `Map(X, BG)` for 2-types and
3-types, descent along a cover, the n-th k-invariant detecting the failure of
Cech descent) needs these as one substrate that the effective-homology lane
builds on.

## Decision

1. **Two libraries in layers, outside the `stdlib` closure.**
   `core.simplicial` = `.sset` (simplices, simplicial sets, constructions),
   `.group` (finite groups, crossed modules), `.nerve` (nerves and finite
   models), `.kan` (horns, Kan conditions, homotopy groups, simplicial maps).
   `core.homological` = `.complex` (coefficients, complexes, maps, cones,
   double complexes), `.abelian` (finitely presented groups, Ext, Tor),
   `.spectral` (spectral sequences), `.simplicial` (cochains of simplicial
   sets, products, Steenrod operations, group cohomology). Ordinary modules
   loaded by `require`; the VM prelude and binding tables are unchanged.

2. **A simplex is a degeneracy operator on a nondegenerate simplex.** By the
   Eilenberg–Zilber lemma every n-simplex is uniquely `eta^* x` with `x`
   nondegenerate and `eta` a monotone surjection, stored as
   `#(simplex eta x)` (the Kenzo representation). A simplicial set is given
   by its nondegenerate simplices per degree and the faces of those; every
   other face, every degeneracy and every simplicial operator
   (`simplicial-set-operator`) is computed by the simplicial identities, so
   front and back faces, cup-i faces, products (pairs with no common
   degeneracy direction) and joins are one mechanism. Programs whose data
   is "all simplices with faces and degeneracies" (crossed-module nerves) use
   `simplicial-set-from-all`, which finds the normal form by peeling
   `v = s_j d_j v`.

3. **Coefficients are `core.ring` carriers.** A complex carries `integers`,
   `rationals` or a field carrier. Over Z every rank and invariant factor is
   the Smith normal form of `core.integer_linalg` (extended here to compute
   ranks and invariant factors without building the transforms, which an
   m x n elimination does not need); Q uses the same engine after clearing
   denominators; over Z/p the residues are reduced with the carrier's
   integer arithmetic mod p; any other field uses the carrier's closures.
   Composite Z/m, jets and polynomial rings are refused by name.

4. **One complex type with explicit truncation.** `#(complex kind R dims
   maps)` for chain and cochain complexes; a complex known through degree N
   may carry the one differential needed for exact (co)homology in degree N,
   which is how truncations of infinite complexes (bar constructions,
   classifying spaces) are represented without guessing. Homology tables are
   `((k betti torsion) ...)`, the format the examples and `core.sheaf` use.
   The cochain mapping cone is the cocone `A^n (+) B^{n-1}`, so complexes stay
   in degrees >= 0 and the cone of a restriction computes relative cohomology.

5. **Spectral sequences are computed, not assumed.** A filtered complex is a
   complex with a filtration degree per basis vector; `E_r = Z_r / B_r`
   (McCleary 2.2) is computed exactly (lattices over Z, ranks over a field),
   `E_{r+1} = H(E_r, d_r)` is checked page by page, and convergence compares
   `E_infinity` with `F^p H / F^{p+1} H` computed directly — over Z as
   groups, so the extension problem is visible rather than hidden.

6. **Ext and Tor through lattices.** A finitely presented abelian group is a
   quotient of `Z^n` by a relation lattice; kernels, cokernels and
   subquotients are lattice computations through the Smith normal form; Tor
   and Ext come from the length-one free resolution. The gcd formulas are
   tests, not the method.

7. **Cup-i products by the Medina-Mardones formula, Steenrod squares as
   `u cup_{n-k} u`.** Every face involved is a simplicial operator of item 2,
   so the formula works on any simplicial set (BG, nerves of 2-groups, lens
   spaces), not only on simplicial complexes.

8. **Mapping spaces by enumeration with early rejection.**
   `simplicial-maps-fold` enumerates `Map(X, Y)` for finite X, choosing edges
   so that triangles close early and indexing the simplices of Y by their
   boundary; Dijkgraaf–Witten and Yetter sums, flat connections and S6.3's
   `Map(hypercover, BG)` are folds over it.

## Consequences

- Eight examples are written against the libraries, each byte-identical to
  the previous program in JIT and AOT: Kan complexes and horns, 2-group
  Postnikov invariants, cohomology rings by cup products, the cohomology
  rings of D_4 and Q_8, group cohomology of order eight, the group-homology
  sweep (identical receipts on nine group specifications), the second
  k-invariant on four-manifolds and descent/hypercohomology; together they
  shrink from 3008 to 1529 lines. The homotopy-colimit example was moved by
  the category lane (ADR-0036).
- Programs that load the whole libraries pay JIT compilation time; a program
  can require only the layers it uses.
- The libraries also run on the bytecode VM (four of the examples are VM
  tests): each module defines a procedure before any procedure that calls it,
  because the VM compiles a module's top level in order, and the
  computations that build large temporaries (boundary matrices, cochain
  products, eliminations, homology) run inside `with-region`, which both
  engines reclaim; simplicial-set bases are computed before a region is
  entered, so only results leave it.
- `simplicial-set-simplices` enumerates every simplex of a degree, so horn and
  homotopy computations are for low degrees; chains use nondegenerate
  simplices only. Dense matrices bound complexes to thousands of cells.
- Planned for v1.4.1 on the same representation: Steenrod reduced powers at
  odd primes, simplicial groups and the Kan loop group, effective homology
  of infinite complexes (the next lane), Serre spectral sequences of
  fibrations, and 3-types (2-crossed modules) with their second k-invariant.

## Alternatives considered

- **Simplices as explicit vertex lists only (simplicial complexes).**
  Classifying spaces, crossed-module nerves and one-vertex triangulations
  are not simplicial complexes; the degeneracy-operator form covers them all.
- **Symbols `'Z`, `'Q` for coefficients, as in `core.sheaf.module`.** Fields
  F_p and number fields would each need a new symbol and a new branch; a
  carrier already says whether it is a field and how to compute in it.
- **Floating-point ranks or probabilistic rank tests mod a random prime.**
  Rejected: torsion and ranks are exact invariants; every result is exact.
- **A spectral-sequence engine that only reports page dimensions over a
  field.** Rejected: torsion in `E_r` over Z is the information the descent
  computations need.

## Verification

- `tests/stdlib/simplicial_test.esk` (CTest `simplicial_jit`,
  `simplicial_aot`): closed-form simplex counts, the five simplicial
  identities on every simplex, product and join counts, Euler
  characteristics, group axioms and class equations, crossed-module axioms
  with negative controls, nerve counts against cocycle counts, Kan and
  quasi-category classifications (BG, 2-group nerves, the pseudocircle,
  Delta^2, a non-groupoid category), Kan's homotopy groups against coker and
  ker, and mapping-space counts against homomorphism counts.
- `tests/stdlib/homological_test.esk` (CTest `homological_jit`,
  `homological_aot`): homology of spheres, RP^n, six lens spaces, products,
  suspensions and cones; BG for cyclic, dihedral, quaternion and symmetric
  groups; the normalization theorem; twisted coefficients; universal
  coefficients over Z, F_2, F_3; Kunneth; Ext/Tor against gcd formulas;
  exact long sequences and induced maps; chain homotopies with negative
  controls; cohomology rings of K(Z/2,1) and K(Z/3,1); Leibniz and
  associativity at the cochain level; Steenrod squares against the binomial
  formula, Cartan and the Adem relations; Q/Z coboundaries; spectral
  sequences page by page and in the limit, over F_2, F_3 and Z.
- The rewritten examples diffed byte for byte against the previous programs
  in JIT and AOT.

## References

J. P. May, *Simplicial Objects in Algebraic Topology* (1967); P. G. Goerss,
J. F. Jardine, *Simplicial Homotopy Theory* (1999); C. A. Weibel, *An
Introduction to Homological Algebra* (1994); J. McCleary, *A User's Guide to
Spectral Sequences* (2001); A. Hatcher, *Algebraic Topology* (2002); K. S.
Brown, *Cohomology of Groups* (1982); D. M. Kan, Ann. of Math. 67 (1958);
A. Joyal, JPAA 175 (2002); J. Duskin, TAC 9 (2002); J. Baez, A. Lauda, TAC 12
(2004); N. E. Steenrod, Ann. of Math. 48 (1947); A. M. Medina-Mardones,
Comput. Geom. 109 (2023); F. Sergeraert, J. Rubio, Bull. Sci. Math. 126
(2002); M. C. McCord, Duke Math. J. 33 (1966); nLab, "simplicial set",
"Kan complex", "spectral sequence", "Steenrod square".
