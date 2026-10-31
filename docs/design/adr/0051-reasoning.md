# ADR-0051: `core.reasoning` — chaining, constraints and SAT, knowledge graphs

**Status:** Proposed
**Date:** 2026-10-30
**Scope:** the v1.6.0-reasoning items of the roadmap: backward chaining,
forward chaining, constraint solving with SAT integration, knowledge graphs
with SPO/POS/OSP indexing, knowledge-graph embeddings, and incremental
knowledge-base evaluation for search over ansatz families; the boundary with
the consciousness engine, `core.dbsp` (ADR-0009), `core.ml.neurosymbolic`,
the refinement-type decision procedure (ADR-0044) and Noesis.

## Context

The consciousness engine shipped in v1.1 gives Eshkol unification over cons
terms, substitutions, facts and a fact base (`unify`, `walk`, `make-kb`,
`kb-query`). It answers one-step pattern queries; it has no proof search, no
rules, no fixed points, no constraint propagation, no Boolean solver and no
graph store. The roadmap's v1.6.0 rung asks for those, and the Navier-Stokes
mechanization plan (rows 21, 46, 75 and 291 of
`docs/design/NAVIER_STOKES_RESIDUAL_MECHANIZATION.md`) asks for an ansatz
family to be a query whose answer updates incrementally as constraints are
added, on the DBSP spine.

Noesis, the reference consumer, ships with v1.5 and v1.6. A map of its
implemented components against the v1.6 items:

| v1.6 item | Noesis | Decision here |
|---|---|---|
| Backward chaining (SLD, cut, negation, tabling) | absent; Aletheia proves through tactic search and an external prover | build |
| Forward chaining | designed, not implemented: a bounded rule forward-chain for grounding and saturation-based lemma discovery in Aletheia | build as the substrate those designs need |
| Finite-domain constraints | absent | build |
| SAT solving | absent | build |
| Triple store with SPO/POS/OSP | absent (the dKB is a fact base with confidences) | build |
| KG embeddings | absent (term embeddings and soft unification exist) | build on the shared embedding tables |
| Incremental KB evaluation | absent | build on `core.dbsp` |

Nothing below copies Noesis code; the map only decides what Eshkol must
provide.

## Decision

1. **One library, `core.reasoning`, in parts.** `core.reasoning.prolog`,
   `.datalog`, `.fd`, `.sat`, `.kg`, `.kge`, `.ansatz`; `(require
   core.reasoning)` loads all of them. They are ordinary Eshkol modules
   outside the `(require stdlib)` closure, so the VM prelude and its
   generated binding tables are unchanged, and they run unchanged in JIT and
   AOT.

2. **The consciousness engine's term language is the interface.** Every part
   takes terms with `?x` logic variables as `logic-var?` decides them, and
   answers are engine substitutions built with `unify`, read with `walk`.
   One-way matching (Datalog atoms against rows, triple patterns against
   triples) is the engine's `unify`. Knowledge bases made with `make-kb` are
   first-class inputs: attached to a Prolog database they answer goals on
   predicates without clauses through `kb-query`; `datalog-saturate-kb!` and
   `ansatz->kb!` write facts back with `kb-assert!`.

3. **Backward chaining is a trail machine, not the engine's substitution.**
   The engine's substitutions copy on extend and its variable registry is a
   bounded name table; proof search renames clause variables at every step.
   SLD resolution therefore runs on mutable variable cells with a binding
   trail and explicit choice points (one loop, so deep searches use no host
   stack for backtracking), clause templates matched head-first against the
   goal, first-argument indexing, and no choice point for the last matching
   clause. Cut is a choice-point barrier; `if`, `once` and `not` are
   sub-searches that share the trail. Tabling is linear tabling: a tabled
   call iterates its clauses to a fixed point over its strongly connected
   component, the component leader marks the tables complete, and negation
   that reads an incomplete table is reported as non-stratified. Answers
   leave the machine as engine substitutions.

4. **Forward chaining is stratified Datalog compiled to DBSP circuits.** There
   is one incremental engine (ADR-0009): each stratum is a nested-clock
   recursive circuit of `core.dbsp.recursive`. Negation is expressed with a
   new linear circuit operator `difference` (left minus right), added to
   `core.dbsp.circuit` beside `union`; a negated atom becomes
   `difference(binding, semijoin(binding, distinct(negated)))`, with the
   negated relation from a lower stratum, so each recursive body stays
   monotone in its own approximant. Evaluation from empty is semi-naive;
   updates are signed Z-set changes and deletions are exact, with no
   delete-rederive pass. A Rete network was considered and rejected: its
   alpha and beta memories are the circuit's filters and join arrangements,
   and a second incremental engine would duplicate the spine.

5. **Constraints and SAT are separate solvers with one bridge.**
   `core.reasoning.fd` propagates to a fixed point with a constraint queue:
   bounds consistency for linear constraints, arc consistency for relations
   and tables, Regin's matching-based domain consistency for all-different;
   depth-first search with first-fail or dom/deg ordering and branch and
   bound. Every constraint also carries a checker, which gives the direct
   encoding to CNF (`fd->cnf`) for the CDCL solver of `core.reasoning.sat`.
   The SAT solver records every learned clause and deletion, so UNSAT answers
   carry DRAT proofs, and `drat-check` is an independent checker that shares
   no code with the solver.

6. **Knowledge graphs: three indexes over one set.** A triple store keeps
   nested hash indexes SPO, POS and OSP; every pattern shape is answered from
   one index, and the same index sizes give the cardinality estimates of the
   greedy join order. The `select` form covers basic graph patterns, filters,
   optional and union groups, distinct, ordering, slicing and counting.

7. **Embeddings share the neuro-symbolic tables.** Entity and relation vectors
   of TransE, DistMult, ComplEx and RotatE live in `core.ml.neurosymbolic`
   embedding tables, so soft unification and KB attention read the trained
   vectors. Losses are plain functions of one packed parameter vector and
   their gradients come from reverse-mode `gradient`.

8. **Ansatz search is a standing query on one circuit.** Survivors =
   distinct(candidates) - distinct(rejected rows). Constraint verdicts are
   recorded; adding a constraint evaluates survivors only, adding candidates
   evaluates new rows only, retracting a constraint re-evaluates only the rows
   that lost their last recorded rejection.

## Upstreaming proposal (Noesis)

For the maintainers to decide through the Noesis boundary process;
nothing moves without that process.

- The grounding design's rule forward-chain and Aletheia's saturation-based
  lemma discovery can be written against `core.reasoning.datalog` (exact,
  incremental) instead of a Noesis-local chainer; confidences stay in the dKB.
- dKB facts can be mirrored into a `core.reasoning.kg` store for indexed
  graph queries while the dKB keeps ownership of confidence and retraction.
- Aletheia's search can use `core.reasoning.prolog` for goal-directed
  premise and tactic retrieval and `core.reasoning.sat` for propositional
  side conditions, with DRAT proofs as checkable certificates.
- The term-embedding and soft-unification code in Noesis and
  `core.ml.neurosymbolic` should converge on one embedding table; the KGE
  models here already use the Eshkol one.

## Shared SAT core (proposal for ADR-0044)

The refinement-type checker (ADR-0044, `lib/types/smt/dpllt.cpp`) runs a
chronological DPLL over Tseitin-encoded theory atoms inside the compiler.
This record does not touch it. Proposal: when that search needs clause
learning, port the CDCL core of `core.reasoning.sat` (watched literals,
first-UIP learning, VSIDS, restarts, DRAT emission) to one C++ module that
both the type checker and a native `sat-solve` call, and keep `drat-check` as
the independent checker for both. Until then the two solvers have disjoint
users and no shared code.

## Consequences

- Programs get Prolog-style proof search, Datalog, constraint solving, SAT
  with proofs, a graph store and embeddings from one require, all over the
  consciousness engine's terms.
- `core.dbsp.circuit` gains `difference`; the existing circuit and recursive
  suites pass unchanged and a test covers the new operator's linearity.
- The library is interpreted Eshkol: proof search and CDCL are fast enough
  for the benchmark set and formulas with hundreds of variables, not for
  industrial instances. A native SAT core (above) and compiled clause
  indexing are the scaling path, Planned for v1.6.1.
- Tabling covers definite programs and stratified negation; well-founded
  negation through tables (SLG delay and simplification) is Planned for
  v1.6.1.

## Alternatives considered

- **Proof search on engine substitutions.** Rejected: copy-on-extend
  substitutions and a bounded variable registry make renaming quadratic and
  exhaustible.
- **A Rete network for forward chaining.** Rejected for the reason in
  decision 4.
- **SAT through an external solver.** Rejected as the only path: the solver
  must run where Eshkol runs, and its proofs must be checkable in-process.
- **Embedding tables private to the KGE module.** Rejected: one symbol, one
  vector across the neuro-symbolic layer.

## Verification

- `tests/stdlib/reasoning_prolog_test.esk`: control constructs, tabling over
  cyclic and mutually recursive programs, the KB bridge, naive reverse (496
  inferences), n-queens counts 2, 10, 4, 40, 92 and the zebra puzzle.
- `reasoning_datalog_test.esk`: 30 signed updates each compared with
  from-scratch evaluation and Warshall closure; stratified negation under
  insertions and retractions; refusal of unsafe and unstratifiable programs.
- `reasoning_sat_test.esk`: pigeonhole n = 1..5 with checked DRAT proofs,
  proofs with deletions, queens n = 1..8, 40 random 3-SAT instances against
  exhaustive search; on a development machine, 30 random 70-variable
  instances at the threshold agree with z3.
- `reasoning_fd_test.esk`: domain consistency, SEND+MORE=MONEY, queens,
  Sudoku, magic squares, random CSPs against enumeration and the CNF
  encoding, branch and bound against brute force.
- `reasoning_kg_test.esk`, `reasoning_kge_test.esk`,
  `reasoning_ansatz_test.esk`: index consistency, queries against a
  nested-loop scan; gradients against closed forms and central differences;
  survivors against from-scratch filtering with fewer evaluations.
- `reasoning_soundness_test.esk`: tabled SLD, Datalog and Warshall agree on
  random graphs; every SLD answer re-proves; negation agrees across engines;
  CDCL against exhaustive search with proof checks and a mutated proof;
  FD against enumeration and CNF; graph queries against Prolog rules.
- Gate: `scripts/run_reasoning_gate.sh` (sixteen CTest entries in JIT and
  AOT plus the reference-page examples) emits `v16_reasoning`.

## References

R. Kowalski (1974); J. W. Lloyd, *Foundations of Logic Programming* (1987);
N.-F. Zhou, T. Sato, Y.-D. Shen, TPLP 8(1) (2008); W. Chen, D. S. Warren,
JACM 43(1) (1996); K. R. Apt, H. A. Blair, A. Walker (1988); S. Abiteboul,
R. Hull, V. Vianu, *Foundations of Databases* (1995); M. Budiu et al., VLDB
2023; A. K. Mackworth (1977); J.-C. Regin, AAAI 1994; J. P. Marques-Silva,
K. A. Sakallah (1999); N. Een, N. Sorensson, SAT 2003; G. Audemard, L. Simon,
IJCAI 2009; N. Wetzler, M. Heule, W. Hunt, SAT 2014; W3C SPARQL 1.1 (2013);
C. Weiss, P. Karras, A. Bernstein, VLDB 2008; A. Bordes et al., NIPS 2013;
B. Yang et al., ICLR 2015; T. Trouillon et al., ICML 2016; Z. Sun et al.,
ICLR 2019.
