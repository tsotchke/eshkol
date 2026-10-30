# nLab conformance

How much of the [nLab](https://ncatlab.org/nlab/show/HomePage) Eshkol computes,
measured claim by claim. Each computed claim is a sentence quoted from an nLab
page (read 2026-10-29), turned into a finite statement and decided exactly by an
Eshkol program on the mathematics libraries: [`core.sheaf`](../reference/stdlib/sheaf.md),
[`core.topos`](../reference/stdlib/topos.md), [`core.ring`](../reference/stdlib/ring.md),
[`core.multimodular`](../reference/stdlib/multimodular.md),
[`core.integer_linalg`](../reference/stdlib/integer_linalg.md), `core.category`
(ADR-0036), and `core.simplicial` with `core.homological` (ADR-0037). Every
check can fail: it compares the library's answer with an independent
computation (a definition re-evaluated by enumeration, a second algorithm, a
closed formula) or with the value nLab states.
The general theorems behind the central claims are machine-checked in Agda and
Rocq; see [Formal cross-checks](../reference/math/formal-crosschecks.md).

<!-- nlab-coverage: pages=51 computed-pages=50 computed=132 planned=9 -->

| | Pages | Claims |
|---|---:|---:|
| Computed, passing in native JIT and AOT | 50 | 132 |
| Listed, not yet computed (facility named below) | 1 further page, and 8 of the pages above | 9 |
| Total | 51 | 141 |

## How it runs

| File | Contents |
|---|---|
| [`tests/nlab/nlab_topos_test.esk`](../../tests/nlab/nlab_topos_test.esk) | subobject classifiers, sieves, Heyting algebras, Grothendieck and Lawvere-Tierney topologies, Boolean and De Morgan toposes, Kripke-Joyal semantics, Yoneda, exponentials, power objects |
| [`tests/nlab/nlab_space_test.esk`](../../tests/nlab/nlab_space_test.esk) | sheaves, sheafification, descent, Cech cohomology, finite spaces and the homotopy types they model, Euler characteristics, nerves |
| [`tests/nlab/nlab_ring_test.esk`](../../tests/nlab/nlab_ring_test.esk) | dual numbers, local rings, nilpotents, units mod n and the Chinese remainder theorem, quadratic reciprocity, group algebras, functor of points, jets, Smith normal form, prime spectra |
| [`tests/nlab/nlab_category_test.esk`](../../tests/nlab/nlab_category_test.esk) | Kan extensions, adjunctions, ends, comma categories |
| [`tests/nlab/nlab_simplicial_test.esk`](../../tests/nlab/nlab_simplicial_test.esk) | simplicial sets, Kan complexes, nerves, the bar construction, group cohomology, cup products, Steenrod squares, spectral sequences, crossed modules, Postnikov sections |
| [`tests/nlab/claims.tsv`](../../tests/nlab/claims.tsv) | the manifest: one row per claim, computed or planned |
| [`scripts/run_nlab_conformance.sh`](../../scripts/run_nlab_conformance.sh) | runs the ten CTest entries and checks every claim line against the manifest and the coverage line above |

Each claim prints one line, `PASS: [page#k] <what was computed> => <value>`, so
the output is the evidence. The header comment of each claim quotes the nLab
sentence it checks. CTest: `nlab_{topos,space,ring,category,simplicial}_{jit,aot}`;
ICC criterion `v14_nlab_conformance`.

```text
ctest --test-dir build --output-on-failure -R '^nlab_(topos|space|ring|category|simplicial)_(jit|aot)$'
BUILD_DIR=build ./scripts/run_nlab_conformance.sh
```

The finite models: presheaf toposes on nine small categories (the point, the
interval 0 -> 1, a cospan, a span, the pseudocircle poset, two discrete
objects, BZ/2, BZ/3, the walking isomorphism) and sheaf toposes on finite
spaces; homotopy types enter as finite T_0 spaces (McCord): the minimal
models of spheres and face posets of triangulations (boundaries of simplices,
the 6-vertex RP^2, 3 x 3 grids for the torus and the Klein bottle), whose
cohomology with constant coefficients is singular cohomology, and as
simplicial sets (standard simplices and horns, classifying spaces BG, nerves
of categories and of crossed modules); rings are Z/n, F_p[e]/e^k, Q[e]/e^k,
polynomial rings, Z[C_n] and Z[C2 x C2]; groups are Z/n, Z/2 x Z/2, S_3, Q_8.

## Computed claims, page by page

| nLab page | Suite | Claims | What is computed |
|---|---|---:|---|
| subobject classifier | `nlab_topos_test` | 5 | #1 Omega(U) = sieves(U) in PSh(C): \|Omega(U)\| per object; #2 \|Sub(Y(U))\| (sub-presheaves enumerated directly) = \|sieves(U)\|; #3 sieves on the object of BZ/2 and BZ/3; #4 every element of BZ/2, BZ/3 acts as the identity on Omega; #5 Set = PSh(point): \|Omega\|, \|Hom(X, Omega)\| for \|X\| = 4, chi round trip on all 16 subsets |
| sieve | `nlab_topos_test` | 3 | #1 g^* S equals {f : g f in S} and is a sieve, for every arrow g and sieve S (pairs checked); #2 generated sieve = arrows factoring through a generator = smallest sieve containing them (families checked); #3 the union and the intersection of every family of sieves on c is a sieve (families checked) |
| Heyting algebra | `nlab_topos_test` | 7 | #1 (x meet a) <= b iff x <= (a => b) for all x, a, b in Omega(c); stages checked; #2 not x = the largest y with y meet x = 0 (found by enumeration); #3 modus ponens, b <= a => b, and a => (a => b) = a => b; #4 b <= (b => a) => a; #5 the three Boolean criteria agree at every stage; number of Boolean stages; #6 not (x join y) = not x meet not y, and not x join not y <= not (x meet y); #7 (U => V) = int(U^c union V) on the opens of three finite spaces; opens per space |
| Grothendieck topology | `nlab_topos_test` | 3 | #1 axioms hold for six topologies; removing the covers of {a, b} violates; #2 trivial topology covers only by Ob(C/c), the discrete one by all sieves, both topologies; #3 open-cover topology meet dense topology on the opens of the pseudocircle: covering sieves per object |
| Lawvere-Tierney topology | `nlab_topos_test` | 3 | #1 j true = true, j j = j, j preserves meets, for the j of six sites; #2 walking arrow: (endomaps of Omega, Lawvere-Tierney topologies, Grothendieck topologies); the translations are mutually inverse; #3 (\|a(F x G)\|, \|aF\| \|aG\|) per open for three pairs; a(1) |
| dense topology | `nlab_topos_test` | 2 | #1 the covering sieves of dense-site are exactly the sieves of the definition, on nine categories; #2 (category, Ore?, J_d = J_at?): every dense sieve is nonempty and J_d = J_at exactly for Ore categories |
| double negation topology | `nlab_topos_test` | 2 | #1 not not on Omega of PSh(C) equals the closure of the dense topology, nine categories; #2 Omega_{not not}(c) is Boolean at every object, nine categories |
| Boolean topos | `nlab_topos_test` | 4 | #1 (category, groupoid?, excluded middle forced?); #2 (category, Omega = 1 + 1?, Boolean?); #3 excluded middle is forced in Sh_{not not}(C), nine categories; #4 Sh(2^n) Boolean for n = 1, 2, 3 |
| De Morgan topos | `nlab_topos_test` | 4 | #1 (category, Ore?, weak excluded middle forced?, De Morgan law forced?); #2 Boolean presheaf toposes are De Morgan; Set^-> is Ore and De Morgan but not Boolean; #3 (finite space, extremally disconnected?, Sh(X) De Morgan?); #4 (category, Ore?, Omega_{not not} decidable?) |
| Kripke-Joyal semantics | `nlab_topos_test` | 5 | #1 forcing = factoring through [[phi]] on every instance (formula, topos, stage); #2 monotone and local on every formula of the corpus; #3 the negation and implication clauses, recomputed over every g : V -> U (instances); #4 excluded middle on the walking arrow: first refuting (stage env); #5 sections on {a, b} forcing the disjunction but neither disjunct |
| Yoneda lemma | `nlab_topos_test` | 2 | #1 \|Nat(y(c), X)\| = \|X(c)\| and eta \|-> eta_c(id_c) is a bijection; counts; #2 \|Nat(y(c), y(d))\| = \|C(c, d)\| for all objects of nine categories |
| exponential object | `nlab_topos_test` | 2 | #1 \|Hom(Z, X^Y)\| = \|Hom(Z x Y, X)\| for 27 triples on three categories; counts; #2 (\|X^S\|, \|X\|^\|S\|) in Set |
| power object | `nlab_topos_test` | 2 | #1 \|P(A)\| for \|A\| = 0..4 in Set, and \|Omega^1\| = \|Omega\| in PSh(span); #2 \|Sub(c x d)\| = \|Hom(d, P c)\| on three categories; counts |
| Sierpinski topos | `nlab_topos_test` | 1 | #1 \|Omega\| on the opens (), (0), (0 1) of Sierpinski space vs on 0, 1 of the interval |
| sheaf | `nlab_space_test` | 3 | #1 every matching family on every cover of the pseudocircle has exactly one amalgamation in a(const {p, q}) and not in const {p, q}; families checked; #2 (const separated?, const with F(empty) = 1 separated?, sheaf?) and its failing open; #3 (a(const) local for every covering sieve?, const local?) |
| sheafification | `nlab_space_test` | 4 | #1 sections of a(const {p, q}) and a(const {0, 1, 2}) on the seven opens; both sheaves; #2 (\|Hom(F, G)\|, \|Hom(aF, G)\|) for F = const 2, const 3 and three sheaves G; #3 (unit const -> const^+ natural?, unit on a sheaf bijective?); #4 (locally surjective?, locally injective?, not an isomorphism?) |
| descent | `nlab_space_test` | 2 | #1 ranks of Gamma(U) for Z and the twisted system; both satisfy descent on every covering sieve; #2 glued value from agreeing pieces, and the obstruction from disagreeing ones |
| Cech cohomology | `nlab_space_test` | 2 | #1 (Leray: Cech H^1 of the twisted system; non-Leray on S^2: Cech, derived); #2 (order, cocycle) of the Cech H^1 generator |
| finite topological space | `nlab_space_test` | 2 | #1 (topologies on 3 points, preorders on 3 points, distinct opens families); #2 H^* of the face posets of the boundary of the 2- and 3-simplex |
| specialization topology | `nlab_space_test` | 2 | #1 opens = up-sets of the opposite order, closed under arbitrary intersection; opens per space; #2 pseudocircle -> 3-chain: (maps, monotone maps); monotone iff continuous |
| pseudocircle | `nlab_space_test` | 2 | #1 the opens of the 4-point space with l, r < t, b; #2 H^*(pseudocircle; Z) |
| sphere | `nlab_space_test` | 2 | #1 Betti numbers of the (2n+2)-point model, n = 0..3; #2 top cohomology of the boundary of the 2-, 3-, 4-simplex |
| torus | `nlab_space_test` | 2 | #1 H^* of pseudocircle x pseudocircle and of the 3 x 3 triangulated torus; #2 reduced Betti numbers of T^2, S^1, S^2 |
| real projective space | `nlab_space_test` | 4 | #1 H^*(RP^2; Z); #2 H_*(RP^2; Z); #3 top cohomology of RP^1 and RP^2; #4 H_k(BZ/2; Z), k = 0..5 |
| Klein bottle | `nlab_space_test` | 1 | #1 H^*(K; Z) of the 3 x 3 triangulated Klein bottle |
| Euler characteristic | `nlab_space_test` | 5 | #1 chi of the minimal models of S^0 .. S^4; #2 (cell count chi, Betti chi) for RP^2, T^2, K; #3 (V - E + F, chi of the face poset) for RP^2, T^2, K, S^2; #4 (weighting chi, chain chi) for S^2 and RP^2 face posets; weighting chi of BZ/3; #5 (chi(X union Y), chi(X) + chi(Y) - chi(X meet Y)) for open covers of S^1, S^2 and RP^2 models |
| nerve | `nlab_space_test` | 4 | #1 \|N([1])_n\| for n = 0..5; #2 (non-degenerate 1- and 2-simplices of N(P), strict chains) for the 6-point S^2 model; #3 (category, groupoid?, nerve Kan through degree 3?); #4 (inner horns of four nerves uniquely filled?, inner-horn report of the boundary of Delta[2]) |
| dual number | `nlab_ring_test` | 3 | #1 e * e in Q[e]/e^2, and Q[x] -> Q[e]/e^2 a surjective ring map with kernel (x^2); #2 (field?, e unit?, units of F_5[e]/e^2 of 25); #3 \|a + e b\| for (3 4) (-2 1/2) (0 7) (5/3 -1) (-1 0) |
| local ring | `nlab_ring_test` | 6 | #1 local Z/n for n = 2..60; #2 Z/n whose non-units form an ideal, n = 2..60; #3 (n, k, Z/n[x]/x^k local?); #4 F_p is a field and local, p <= 47; #5 Z/n with exactly one maximal ideal, n = 2..40; #6 F_5[e]-points of y = x^2 and of y^2 = x^3, and the cusp's points over the origin |
| nilpotent element | `nlab_ring_test` | 2 | #1 (nilpotents of Z/72, count of nilpotents in F_7[e]/e^3); #2 (1 - x)(1 + x + ... + x^(n-1)) = 1 for nilpotents of Z/72, Z/4[x], Q[e]/e^4 |
| multiplicative group of integers modulo n | `nlab_ring_test` | 3 | #1 units = residues coprime to n (n <= 60); phi(mn) = phi(m) phi(n) on coprime pairs; #2 ((m n) bijective? inverse by chinese-remainder and multiplicative?); #3 phi(n) from counted units = product formula, n = 2..200; phi(200) |
| quadratic reciprocity law | `nlab_ring_test` | 4 | #1 Euler's criterion = squares (a/p) for all a mod p, odd p < 100; #2 reciprocity on all pairs of odd primes below 100; #3 supplementary laws for odd p < 400 and p = 2^31 - 1: ((-1/p), (2/p)) at 2^31 - 1; #4 the mod-4 form on all ordered pairs of odd primes below 100 |
| group algebra | `nlab_ring_test` | 2 | #1 r* in Z[C5] and Z[C2 x C2] equals the convolution formula on sample pairs; #2 (n, p, characters C_n -> F_p^x, gcd(n, p - 1)); the character t -> 3 of Z[C5] -> F_11 is a ring map |
| functor of points | `nlab_ring_test` | 2 | #1 (\|circle(F_p)\| for odd p <= 31; \|Hom(Z[t]/(t^2+1), Z/n)\| for n = 2, 3, 5, 13, 25, 65); #2 (\|circle(Z/mn)\|, \|circle(Z/m)\| \|circle(Z/n)\|) for (5 13) (3 7) (4 5) |
| jet | `nlab_ring_test` | 1 | #1 5-jet of 1/(1 + x) at 2; polynomial jets = Taylor coefficients |
| Smith normal form | `nlab_ring_test` | 2 | #1 (diagonal of P A Q, diagonal?, \|det P\|, \|det Q\|, ring-det = multimodular-det?) for six integer matrices; #2 (nonzero invariant factors, rows, det or #f) for six matrices |
| prime spectrum | `nlab_ring_test` | 1 | #1 \|Spec Z/n\| for n = 12, 30, 49, 60 (primes dividing n); preimages of primes are prime |
| Kan extension | `nlab_category_test` | 3 | #1 Lan = coend formula and Ran = end formula, canonically isomorphic; functors checked; #2 (\|Lan_! G\|, \|colim G\|, \|Ran_! G\|, \|lim G\|) for two span diagrams; #3 (unit bijective along a full inclusion?, along discrete 2 -> arrow?) |
| adjoint functor | `nlab_category_test` | 2 | #1 R d for d = 0, 1; hom-bijection violations; right adjoint of 0 \|-> 1, 1 \|-> 2; #2 (triangle-identity violations, functors D -> C, those right adjoint to L) |
| end | `nlab_category_test` | 1 | #1 (\|end of Y(c)^X(c)\|, \|Nat(X, Y)\|) for six pairs |
| comma category | `nlab_category_test` | 2 | #1 (\|ob C/c\|, \|arrows into c\|, \|ar C/c\|, morphisms counted) per object of five categories; #2 (\|ob (id/id)\|, \|ar C\|, \|ar (id/id)\|, commutative squares) for four categories |
| simplicial set | `nlab_simplicial_test` | 2 | #1 \|Delta[n]_m\| for n, m = 0..3; #2 simplicial-identity violations through degree 3 for Delta[3], its boundary, a horn, BZ/3, the nerve of a span |
| Kan complex | `nlab_simplicial_test` | 1 | #1 Kan through degree 3: (BZ/3, Delta[1], boundary of Delta[2]) |
| bar construction | `nlab_simplicial_test` | 1 | #1 (nondegenerate n-simplices of BZ/3 and BS_3, (\|G\| - 1)^n); H_*(BZ/3; Z) |
| group cohomology | `nlab_simplicial_test` | 3 | #1 (G, p, \|Hom(G, Z/p)\|, p^dim H^1(G; F_p)); #2 D c = 0 iff the 2-cocycle formula, all 16 resp. 512 normalized 2-cochains of Z/3, Z/2 x Z/2 over F_2; #3 dim H^k(Z/2; F_2), dim H^k(Z/2 x Z/2; F_2), k = 0..3 |
| cup product | `nlab_simplicial_test` | 2 | #1 Leibniz rule on basis cochain pairs of the torus; #2 (a^2 nonzero in H^2(RP^2; F_2)?, xy = +- generator of H^2(T^2; Z)?, x^2 = 0 in H^2(T^2)?) |
| Steenrod square | `nlab_simplicial_test` | 4 | #1 Sq^i(x^n) on RP^infinity for n = 1..4, i = 0..n+1; #2 (Sq^1 x^n, Bockstein x^n) for n = 1..5; Sq^1 a - beta a a coboundary on RP^2; #3 Sq^1 Sq^j x^n = (j even ? Sq^(j+1) : 0) x^n, n = 1..5, j = 0..n; cases; #4 Cartan formula on x^a cup x^b, a = 1, 2, b = 1..3, every n |
| spectral sequence | `nlab_simplicial_test` | 2 | #1 (filtration violations, E_(r+1) /= H(E_r, d_r) violations); E_1 page; #2 E_infinity vs the associated graded of H^*(S^2; F_2): violations; limit page |
| crossed module | `nlab_simplicial_test` | 2 | #1 (violations of AUT(Z/5), \|pi_1\|, \|pi_2\|; a non-example has violations); #2 (violations, \|ker d\|, ker d central in Q_8?) |
| Postnikov tower | `nlab_simplicial_test` | 1 | #1 (\|pi_1\|, \|pi_2\|, \|pi_3\|) of the nerve of Z/2 -0-> Z/2 |

Values worth reading in the output: the walking arrow has 108 endomaps of
Omega, of which exactly 4 are Lawvere-Tierney topologies, matching its 4
Grothendieck topologies; presheaves on the nine categories are Boolean exactly
for the five groupoids and De Morgan exactly for the seven Ore categories;
there are 29 topologies and 29 preorders on a 3-point set; the 6-vertex RP^2
gives H^2 = Z/2 and H_1 = Z/2; the weighting of BZ/3 gives Euler
characteristic 1/3; |Spec Z/60| = 3; F_5[e]-points of the cusp y^2 = x^3 over
the origin form a 2-dimensional tangent space (25 of 45 points); of the 10
functors from the chain 0 < 1 to the chain 0 < 1 < 2 < 3 exactly one is right
adjoint to floor(c/2); Sq^i(x^n) on RP^infinity is binomial(n, i) mod 2; the
nerve of the crossed module Z/2 -0-> Z/2 has pi_1 = pi_2 = Z/2 and pi_3 = 0.

## Not yet computed

Each row is a claim the suite does not compute yet, with what it needs.

| nLab page | Claim (quoted) | Needs | Worked program today |
|---|---|---|---|
| Sierpinski topos #2 | "(Π ⊣ Disc ⊣ Γ ⊣ coDisc) : H^I ← → ← → H" | functors between toposes and their adjunctions (`core.category` with `core.topos`): Planned for v1.4.1 | - |
| structure sheaf #1 | "For a ringed topos (𝒳, 𝒪) the ring object 𝒪 ∈ 𝒳 is called the structure sheaf." (with 𝒪 internally local) | the Zariski site of a finite ring and its structure sheaf as a library constructor: Planned for v1.4.1 | [topos_local_ring_test](../../tests/stdlib/topos_local_ring_test.esk) builds the site by hand |
| local ring #7 | "whenever a + b = 1, a or b is invertible", read in the Zariski topos | the same constructor: Planned for v1.4.1 | same |
| prime spectrum #2 | "given by the basis of open sets D_f = Spec R[f^{-1}] ⊂ Spec R" | the localization carrier R[1/f] in `core.ring`: Planned for v1.4.1 | - |
| functor of points #3 | "The functor which sends R to the R-points of the projective space ℙⁿ corresponds to a non-affine scheme." | schemes glued from affine charts: Planned for v1.4.2 | - |
| Cech cohomology #3 | "If X is a paracompact space the canonical morphism Ȟ(X,A) → H(X,A) from Čech cohomology to abelian sheaf cohomology is an isomorphism" | a statement about infinite spaces; its finite counterpart (Leray covers) is Cech cohomology #1 | - |
| torus #3 | "One has TC(Tⁿ) = n + 1 for the topological complexity" | topological complexity is a homotopy invariant, not a finite enumeration | - |
| sphere #3 | "TC(S^n) = 2 when n is odd; TC(S^n) = 3 when n is even." | as above | - |
| Klein bottle #2 | "The topological complexity of the Klein bottle is 4 (with convention TC(*)=0)." | as above | - |

## Notes on the method

- The quotes were read through the page text on 2026-10-29. "double negation
  topology", "Alexandroff space" and "group ring" open the nLab pages
  "double negation", "specialization topology" and "group algebra"; the claims
  are filed under the page that holds them. nLab has no page "Chinese remainder
  theorem"; the theorem is checked as stated on "multiplicative group of
  integers modulo n".
- Adding a claim: write it in the suite with its quote, add its row to
  `tests/nlab/claims.tsv`, and update the coverage line at the top of this
  page; `scripts/run_nlab_conformance.sh` checks all three agree.
