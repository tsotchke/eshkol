---
kind: reference
status: current
owner-area: testing
since: v1.4.0-connection
sources:
  - tests/formal/agda/KanExtensionTerminal.agda
  - tests/formal/agda/KanExtensionPointwise.agda
  - tests/formal/agda/SievesKripkeJoyal.agda
  - tests/formal/agda/NerveSimplicial.agda
  - tests/formal/coq/SmithNormalForm.v
  - tests/formal/coq/gen_snf_certificates.py
  - scripts/run_formal_agda_gate.sh
  - tests/nlab/claims.tsv
---

# Formal cross-checks of the nLab conformance claims

The [nLab conformance suite](../../reports/NLAB_CONFORMANCE.md) decides 132
claims quoted from nLab pages by running Eshkol on finite instances. This page
covers the step after that. For the most important of those claims, the general
theorem is stated and machine-checked in a proof assistant, and it is paired
with the concrete instance that Eshkol computes. The Eshkol test shows that the
library computes the instance correctly. The proof shows that the instance is
an example of a theorem that holds for every category, every functor and every
matrix.

The proofs are in [`tests/formal/agda/`](../../../tests/formal/agda) (Agda 2.8,
every file `--safe`, with `--cubical` where it uses set quotients) and
[`tests/formal/coq/`](../../../tests/formal/coq) (Rocq 9, standard library
only). `scripts/run_formal_agda_gate.sh` checks all of them.

## What is checked

| Proof file | nLab claim | Eshkol test that computes the instance | General theorem (checked) | Instance (checked) |
|---|---|---|---|---|
| `KanExtensionTerminal.agda` | Kan extension#2 | `tests/nlab/nlab_category_test.esk` | For every small category C and every functor F : C → Set, the quotient colim F represents cocones and the set of compatible families lim F represents cones, naturally in the test set. These are the universal properties of Lan and Ran along C → \*. Corollary: for a group acting on a set, Lan = orbits and Ran = fixed points. | The span {a b c} ← {x y} → {p q} of the suite: colim ≃ Fin 3 and lim ≃ Fin 2, with explicit inverse maps. Eshkol prints (3 3 2 2). |
| `KanExtensionPointwise.agda` | Kan extension#1, #3 | `tests/nlab/nlab_category_test.esk` | For every functor p : C → D and every F : C → Set, the coend (Σ c. D(p c, d) × F c)/∼ is a functor with Nat(Lan_p F, G) ≃ Nat(F, G ∘ p). The end of compatible families is a functor with Nat(G, Ran_p F) ≃ Nat(G ∘ p, F). When p is fully faithful, the unit F c → (Lan_p F)(p c) is a bijection. | General statement only. Eshkol checks the comma-category and coend/end constructions against each other on every functor span → arrow and arrow → BZ/2, and checks the unit along a full and a non-full functor. |
| `SievesKripkeJoyal.agda` | sieve#1, #2, #3; subobject classifier#1, #3; Kripke-Joyal semantics#1, #2, #3, #4 | `tests/nlab/nlab_topos_test.esk` | For every category: sieves pull back; the generated sieve is the smallest one containing the generators; arbitrary unions and intersections are sieves; Ω(c) is a Heyting algebra with R ∧ S ≤ T ⇔ R ≤ (S ⇒ T); Ω is a presheaf under pullback. Kripke–Joyal forcing for ⊤ ⊥ ∧ ∨ ⇒ ¬ and ∀/∃ over Ω is monotone along every arrow, is invariant under equivalent environments, and its truth value at U is a sieve that contains id_U exactly when U forces the formula. For a group G, an inhabited sieve on BG is maximal, so BG has two truth values. | Walking arrow: the sieve {0 → 1} has no complement, so Ω is not Boolean, and stage 1 refutes p ∨ ¬p and ∀p:Ω. p ∨ ¬p. Sieve counts by enumeration: (2, 3) on the walking arrow, 2 on BZ/2, 2 on BZ/3. |
| `NerveSimplicial.agda` | nerve#1, #2; simplicial set#1, #2 | `tests/nlab/nlab_space_test.esk`, `tests/nlab/nlab_simplicial_test.esk` | For every category, the nerve built from composable strings satisfies all five families of simplicial identities. Inner faces compose, outer faces drop an end, and s_k inserts an identity at vertex k. The identities use exactly the unit and associativity laws. | \|N([1])_n\| = n + 2 for n = 0..5. \|Δ[n]_m\| = \|N([n])_m\| = binomial(n+m+1, m+1) for n, m = 0..3. |
| `SmithNormalForm.v` (Rocq) | Smith normal form#1, #2; real projective space#2 | `tests/nlab/nlab_ring_test.esk`, `tests/nlab/nlab_space_test.esk` | For any integer matrix with U A V = D and integer inverses of U and V: z is in the image of A ⇔ U z is in the image of D. A diagonal image is cut out by divisibility. Nonzero diagonal entries in every column give a trivial kernel. The common divisors of the entries are the same for A and D. The first invariant factor is the gcd of the entries of A, so A alone determines it. | The six matrices of the suite: U A V = D with integer inverses, the diagonals Eshkol prints, the divisibility chain, and the first invariant factor. The 6-vertex RP²: H_0 = Z, H_1 = Z/2 (a 1-cycle bounds exactly when one coordinate of U z is even, and some cycle makes it odd), H_2 = 0. |

## Scope

These are the limits of what the proofs cover. Each limit marks a later step.

- Kan extension#1 and #3 are proved in general. No finite instance is checked
  in Agda. The comma-category (colimit over p ↓ d) form of Lan is not
  formalised. Agda checks the coend form, and Eshkol checks the comma and coend
  forms against each other. Both forms are planned for v1.4.1.
- The Kripke–Joyal language in Agda has propositional variables valued in Ω,
  with quantifiers over Ω. Eshkol's corpus also has typed variables over other
  objects of the topos, and it checks locality over covering sieves, which is
  trivial in a presheaf topos. Both are planned for v1.4.1.
- The sieve counts are exhaustive enumerations over the finite categories, with
  each enumerated set tested for sieve closure. The general Heyting-algebra
  proofs apply to every sieve. The proof that the enumeration finds every sieve
  is planned for v1.4.1.
- simplicial set#2 is checked in Eshkol on Δ[3], its boundary, a horn, BZ/3 and
  the nerve of a span. The Agda theorem covers every nerve, which includes Δ[3],
  BZ/3 and the span. The boundary and the horn are not nerves. The count of
  non-degenerate simplices in nerve#2 is planned for v1.4.1.
- Smith normal form#2 also compares the product of the diagonal with ±det A.
  Determinants are planned for v1.4.1. Eshkol computes the homology of RP²
  from the order complex of its face poset, which is the barycentric
  subdivision. The Rocq proof uses the 6-vertex triangulation itself. The two
  values agree, but the proof that homology is invariant under subdivision is
  not part of this check.
- Sheafification as a reflector (S6.2 of the mathematics programme) is planned
  for v1.4.1.

## How the instances are certified

- **Agda**: an instance is an equivalence `≃ Fin k` built from explicit maps
  in both directions, or an equation that holds by `refl`. In the second case
  Agda decides it by evaluating both sides.
- **Rocq**: [`gen_snf_certificates.py`](../../../tests/formal/coq/gen_snf_certificates.py)
  proposes U, U', V, V' and D for each matrix, using exact integer row and
  column operations. It also proposes a left-inverse certificate for the
  columns of d1 · U' beyond the rank of d2 in RP². Rocq re-checks every
  identity by computation (`vm_compute`). It then derives each statement from
  the general theorems, which make no assumption about where the certificates
  came from. The generator's `--check` mode confirms that the certificates in
  the file are the ones it produces.

## Running the gate

```
scripts/run_formal_agda_gate.sh             # PASS / FAIL / SKIP per file
scripts/run_formal_agda_gate.sh --require   # a SKIP fails the gate
```

The gate type-checks every `tests/formal/agda/*.agda` and
`tests/formal/coq/*.v` file. It refuses a file that contains a postulate, a
hole, an `Admitted`, an `Axiom`, or a pragma that turns off termination or
positivity checking. It refuses a Rocq file whose `Print Assumptions` report
lists an axiom. It also refuses a certificate block that is out of date. When Agda, its
`cubical` and `standard-library` libraries, Rocq or python3 is not installed,
the gate reports SKIP for that part. It never reports a silent PASS. If no
prover is available, it exits with status 2. Agda finds its libraries through
the `example-libraries` file shipped next to its data directory, or through
`AGDA_LIBRARY_FILE`. Each run also writes an ICC `runtime_event` trace to
`scripts/icc_traces/formal_agda_gate.jsonl`.

Measured on macOS arm64 with Agda 2.8.0 and Rocq 9.3.0, with the library
interfaces already built: four Agda files, one Rocq file and the certificate
check, 6 of 6 PASS in about 10 seconds.
