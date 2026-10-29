---
kind: explanation
status: current
owner-area: numeric
since: v1.4.0
sources:
  - lib/core/ring.esk
  - lib/core/groebner.esk
  - lib/core/multimodular.esk
  - lib/core/symbolic.esk
  - tests/stdlib/groebner_test.esk
  - scripts/groebner_sympy_crosscheck.py
---
# ADR-0040: Groebner bases on ring carriers

- **Status:** Proposed
- **Date:** 2026-10-29
- **Decision owners:** `core.ring`, `core.symbolic`, `core.multimodular` and exact algebra maintainers
- **Depends on:** ADR-0034 (ring carriers, the functor-of-points evaluation model)
- **Scope:** polynomial rings in several variables, monomial orders, Groebner bases over Q, F_p and other field carriers, and the ideal operations built on them, at library level (no compiler, VM or runtime change)
- **Evidence:** `tests/stdlib/groebner_test.esk` (JIT and AOT), the SymPy cross-check of `scripts/groebner_sympy_crosscheck.py` (180 reduced bases)

## Context

The mathematics program lacked a symbolic layer: no Groebner bases, no
polynomial rings in several variables over arbitrary coefficients, no
elimination. Work that needs them (Hodge classes beyond Fermat and diagonal
cases, implicitization, solving polynomial systems exactly) was restricted to
special cases or to hand-written eliminations.

What existed, found with ICC (`find-symbol`, `search-chunks`,
`duplicate-implementations` on `lib/` and `examples/`) and confirmed by
reading:

| Facility | What it is | Limits |
|---|---|---|
| `core.ring` `(make-polys K x)` | K[x], dense coefficient lists, any carrier K, euclidean over a field | one variable |
| `core.symbolic` `poly` | sparse Q[x1..xn] (#852 adds GF(p) through a field switch) | its own term order (degree, then lex), coefficients from the numeric tower, no division by an ideal |
| `core.polynomial_exact` | dense Q[x] with real-root isolation | one variable |
| `core.multimodular` | primes below 2^31, CRT, rational reconstruction, bounds | no polynomial systems |
| `examples/mathematics_aoki_*.esk` | monomial arithmetic in a Jacobian ring over Q(zeta) | specific to the Fermat case |

No Groebner basis, S-polynomial, monomial order, elimination or Hilbert
polynomial code existed anywhere in the tree.

## Decision

### 1. K[x1..xn] is a core.ring carrier

`(make-polys K (list x1 ... xn) [order])` extends `make-polys`: given a list
of variables it builds the carrier K[x1..xn] (family id 10, `mpolys`) over any
carrier K. It is not a second polynomial type: it has the ring-carrier
descriptor, the generic operations `r+ r* r-expt r= ...`, the law check, ring
maps and the coercion lattice of ADR-0034.

- **Representation.** An element is located, `(ring-element-v1 P terms)`,
  the terms `(exponent-vector . coefficient)` with nonzero coefficients in
  strictly decreasing monomial order. The leading term is the first; the
  constant term, the least monomial under every monomial order, is the last.
- **The order is part of the carrier.** `lex`, `grlex`, `grevlex` (default),
  `(block (k1 o1) ...)`, `(elimination k)`, `(weight w [o])` and
  `(homogenized o)`, each canonicalized and compiled once into a comparison.
  Keeping the order in the carrier makes the leading term a constant-time
  read and makes every term list canonical; the same variables under two
  orders are two carriers.
- **Coercion.** ADR-0034's lattice gains one generic rule: a carrier may
  declare what enters it (its extra `enters`). K[x1..xn] admits polynomials
  over K in a subset of its variables, under any order, and univariate K[x]
  with x among them. So Q[x] and Q[x,y] join at Q[x,y], and changing the
  order is `ring-admit` into the other carrier.
- **Units** are decided as for K[x]: a unit constant term and nilpotent other
  coefficients, inverted by Newton's iteration.
- The `ring-poly` accessors take the multivariate case: `ring-poly` builds
  from terms `(c e1 ... en)`, `ring-poly-variable` takes a name,
  `ring-poly-terms`, `ring-poly-leading-monomial`, `ring-poly-variables`,
  `ring-poly-variable-names`, `ring-poly-monomial-order` and
  `monomial-compare` are new, and `ring-poly-degree`, `ring-poly-leading`,
  `ring-poly-eval` and `ring-poly-evaluation` read the multivariate case.
  The univariate behaviour and every univariate call site are unchanged.

`core.symbolic` keeps its representation (its polynomials are expressions
over the numeric tower, used by the Navier-Stokes mechanization and its AD
interoperation); `core.groebner` converts in both directions
(`symbolic->ring-poly`, `ring-poly->symbolic`). Making `core.symbolic`'s
coefficient fields core.ring carriers is the ADR-0034 library rung item it
already lists.

### 2. Buchberger's algorithm with the standard criteria

`core.groebner` works on the carrier's term lists through the descriptor
(`compare`, `terms`, `from-terms` and K's operations), so it is written once
for every field carrier.

- **Division** by an ordered list is Cox-Little-O'Shea's algorithm, returning
  quotients and remainder. Reduction is one merge pass per step
  (p - c x^m g), with loops that keep their state in accumulators.
- **Buchberger** (1965) with the Gebauer-Moeller installation of the product
  and chain criteria (1988, as in Becker-Weispfenning's UPDATE), the sugar
  strategy (Giovini, Mora, Niesi, Robbiano, Traverso 1991) for pair
  selection, top reduction in the loop and a final interreduction. The output
  is the reduced basis: monic, minimal, tail-reduced, sorted by decreasing
  leading monomial; unique for the ideal and the order.
- **Orders that are not graded** run on the homogenization: the reduced
  basis of <f^h> under `(homogenized o)` dehomogenizes to a Groebner basis of
  <f> under o (if g is in <f>, some h^k g^h is in <f^h>, and the leading
  monomial of h^k g^h under the homogenized order is h^j times the leading
  monomial of g under o). This keeps lex computations degree by degree;
  Katsura-3 under lex over Q fell from minutes to milliseconds.

### 3. The multimodular method over Q

The method is Arnold's (2003), with the verification that makes it exact
independent of the primes:

1. Clear denominators to primitive integer generators and homogenize; J is
   homogeneous.
2. Compute the reduced basis of J_p in fibres F_p (core.multimodular's primes
   below 2^31). Group primes by their sets of leading monomials; keep a group
   whose leading ideal is not strictly contained in another group's, the
   largest such. For integer homogeneous generators dim (J_p)_d <= dim J_d in
   every degree, so an unlucky prime never has a strictly larger leading
   ideal than the true one.
3. CRT and Wang rational reconstruction per coefficient; when two consecutive
   reconstructions agree, check the candidate G over Q: homogeneous, every
   generator reduces to 0, every Gebauer-Moeller pair reduces to 0. Then G is
   a Groebner basis of <G> containing J with LM(G) = LM(G_p), and the Hilbert
   function inequality dim (J_p)_d = dim <G>_d >= dim J_d >= dim (J_p)_d
   forces <G> = J.
4. Dehomogenize and interreduce.

The verification is what makes the result exact; the lucky-prime rule only
saves time. A failed check moves on to further primes.

Over Q the default is the multimodular method; over other fields it is
Buchberger. Measured on Linux x64 (JIT), Buchberger over Q is faster on small
systems (Katsura-5 under grevlex in 4.7 s against 42 s, Katsura-4 under lex in
5.3 s against 34 s), where the exact check over Q dominates the multimodular
run, but its rational coefficients can grow without bound: cyclic-6 under
grevlex did not finish in Buchberger over Q within the hour it was given, and
takes 151 s by the multimodular method (25 s over F_32003). The default
bounds the worst case; `'method 'buchberger` remains available.

### 4. Ideals and invariants

Membership, containment, equality (of reduced bases), sum, product,
elimination (block order, Elimination Theorem), intersection
(<t I, (1 - t) J>), quotient (through intersection with principal ideals),
saturation and radical membership (Rabinowitsch) are each one composition of
`groebner-basis`. Krull dimension and Hilbert series and polynomials (affine
and projective) come from the leading ideal by the Bayer-Stillman recursion;
the Hilbert polynomial is an element of `(make-polys rationals 's)`.
Zero-dimensional ideals give their standard monomials, their number of
solutions with multiplicity, the triangular form of the lex basis, and their
solutions with coordinates in the coefficient field (rational roots over Q,
enumeration over small F_p), each checked on the generators.

## Consequences

- One polynomial-ring facility in several variables, generic over field
  carriers; Groebner computations over Q(alpha) or F_p are the same code.
- The Hodge lane and the other mathematics lanes can eliminate, implicitize
  and solve exactly instead of special-casing.
- Every result is exact. The SymPy cross-check (an independent oracle, used
  only by the script) agrees on 180 of 180 reduced bases on JIT and AOT.
- Performance at this rung: reduced bases of cyclic-6 (45 elements) and
  Katsura-5 (22 elements) over Q and F_p are reached; larger systems need the
  items below.

## Staged plan

| Rung | Item |
|---|---|
| v1.4.0-connection | K[x1..xn] carrier, Buchberger with Gebauer-Moeller, multimodular over Q, ideal operations, Hilbert polynomials, zero-dimensional solving |
| v1.4.1 | geobucket reduction and F4-style linear-algebra reduction; FGLM order change; triangular decomposition of positive-dimensional ideals; `core.symbolic` coefficient fields as carriers |
| v1.4.2 | factorization over Q and F_p for exact solving beyond rational roots; primary decomposition of small ideals |

## References

- B. Buchberger, *Ein Algorithmus zum Auffinden der Basiselemente des
  Restklassenringes nach einem nulldimensionalen Polynomideal*, 1965.
- D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*,
  4th ed., 2015.
- R. Gebauer, H. M. Moeller, *On an installation of Buchberger's algorithm*,
  J. Symbolic Computation 6, 1988.
- A. Giovini, T. Mora, G. Niesi, L. Robbiano, C. Traverso, *"One sugar cube,
  please" or selection strategies in the Buchberger algorithm*, ISSAC 1991.
- E. A. Arnold, *Modular algorithms for computing Groebner bases*,
  J. Symbolic Computation 35, 2003.
- T. Becker, V. Weispfenning, *Groebner Bases*, 1993.
- D. Bayer, M. Stillman, *Computation of Hilbert functions*, J. Symbolic
  Computation 14, 1992.
