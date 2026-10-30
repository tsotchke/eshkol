---
kind: reference
status: current
owner-area: stdlib
since: v1.6.0-reasoning
sources:
  - lib/core/reasoning.esk
  - lib/core/reasoning/prolog.esk
  - lib/core/reasoning/datalog.esk
  - lib/core/reasoning/fd.esk
  - lib/core/reasoning/sat.esk
  - lib/core/reasoning/kg.esk
  - lib/core/reasoning/kge.esk
  - lib/core/reasoning/ansatz.esk
  - tests/stdlib/reasoning_prolog_test.esk
  - tests/stdlib/reasoning_datalog_test.esk
  - tests/stdlib/reasoning_fd_test.esk
  - tests/stdlib/reasoning_sat_test.esk
  - tests/stdlib/reasoning_kg_test.esk
  - tests/stdlib/reasoning_kge_test.esk
  - tests/stdlib/reasoning_ansatz_test.esk
  - tests/stdlib/reasoning_soundness_test.esk
---

# `core.reasoning` — chaining, constraints, SAT and knowledge graphs

**Source:** [reasoning.esk](../../../lib/core/reasoning.esk) and
[`lib/core/reasoning/`](../../../lib/core/reasoning/).
**Require:** `(require core.reasoning)` for everything, or one part, for
example `(require core.reasoning.prolog)`. The library is ordinary Eshkol on
top of the consciousness engine (`unify`, `walk`, `make-kb`, `kb-query`),
[`core.dbsp`](dbsp.md) and [`core.ml.neurosymbolic`](../../../lib/core/ml/neurosymbolic.esk);
it is not part of `(require stdlib)`.
**Design record:** [ADR-0051](../../design/adr/0051-reasoning.md).

| Part | What it does |
|---|---|
| `core.reasoning.prolog` | Goal-directed proof search: SLD resolution, cut, negation as failure, tabling, attached knowledge bases |
| `core.reasoning.datalog` | Forward chaining: stratified Datalog compiled to incremental DBSP circuits |
| `core.reasoning.fd` | Finite-domain constraints: propagation, all-different, search, branch and bound, CNF encoding |
| `core.reasoning.sat` | CDCL SAT solving with DRAT proofs and an independent proof checker |
| `core.reasoning.kg` | Triple store with SPO/POS/OSP indexes and a SPARQL-like `select` |
| `core.reasoning.kge` | TransE, DistMult, ComplEx and RotatE embeddings trained with reverse-mode AD |
| `core.reasoning.ansatz` | Incremental search over ansatz families on a DBSP circuit |

Terms are the consciousness engine's: symbols, numbers, strings, lists, and
logic variables written `?x` (`?_` is anonymous). A compound term is a list
whose head is the functor, so `(?h . ?t)` matches `(1 2 3)`.

## Backward chaining

```scheme
(require stdlib)
(require core.reasoning.prolog)

(define db (make-prolog))
(prolog-consult! db
  '((parent tom bob) (parent bob ann) (parent ann joe)
    (<- (ancestor ?x ?y) (parent ?x ?y))
    (<- (ancestor ?x ?y) (parent ?x ?z) (ancestor ?z ?y))))
(map (lambda (s) (walk '?w s)) (prolog-query db '(ancestor tom ?w)))  ;; => (bob ann joe)
(prolog-count db '(append ?x ?y (1 2 3)))                            ;; => 4
(prolog-query-terms db '(is ?x (+ 1/3 1/6)))                          ;; => ((is 1/2 (+ 1/3 1/6)))
```

A clause is a fact `(p a b)` or a rule `(<- head goal ...)`. Resolution is
depth-first SLD with chronological backtracking over a binding trail,
first-argument indexing, and no choice point for the last matching clause.

| Procedure | Meaning |
|---|---|
| `(make-prolog)` | A database with the built-ins and the list library |
| `(prolog-add! db clause)`, `(prolog-consult! db clauses)` | Add clauses in order |
| `(prolog-query db goal [limit])` | One engine substitution per solution; `(walk '?x s)` reads it |
| `(prolog-query-terms db goal [limit])` | The instantiated goal per solution |
| `(prolog-solve db goal proc)` | Calls `(proc instance)` per solution while it returns true |
| `(prolog-prove? db goal)`, `(prolog-first db goal)`, `(prolog-count db goal)` | One answer, a boolean, or the number of solutions |
| `(prolog-table! db name arity)` | Evaluate the predicate by tabling |
| `(prolog-table-answers db goal)` | The completed answer table of a tabled call |
| `(prolog-attach-kb! db kb)` | Answer goals on predicates without clauses with `kb-query` on a `make-kb` base |
| `(prolog-dynamic! db name arity)` | A predicate filled by `assertz` fails until it has clauses |
| `(prolog-define-builtin! db name arity proc)` | A deterministic test written in Eshkol |
| `(prolog-define-function! db name arity proc)` | Unifies the last argument with `(proc args ...)` |
| `(prolog-set-unknown! db 'error|'fail)` | Calling an undefined predicate reports it (default) or fails |
| `(prolog-inferences db)`, `(prolog-reset-inferences! db)` | Logical inferences (clause tries) |

Built-in goals: `true`, `fail`, `!`, `(and g ...)`, `(or g ...)`,
`(if c t [e])`, `(not g)`, `(once g)`, `(call g a ...)`, `(findall t g l)`,
`(forall c a)`, `=`, `/=`, `==`, `/==`, `unify-with-occurs-check`,
`(is x e)`, `num=`, `num/=`, `<`, `>`, `=<`, `<=`, `>=`, `succ`, `between`,
`var`, `nonvar`, `atom`, `number`, `integer`, `string`, `atomic`, `compound`,
`is-list`, `ground`, `copy-term`, `functor`, `arg`, `=..`, `length`, `write`,
`nl`, `assertz`, `asserta`, `retract`. The library adds `append`, `member`,
`memberchk`, `reverse`, `select`, `permutation`, `last`, `nth0`, `nth1` and
`sum-list`; a program that defines one of these names replaces it.
Arithmetic is Eshkol's exact numeric tower.

**Tabling.** A tabled call is evaluated by linear tabling: the call is
re-evaluated over its clauses, consuming its own and its strongly connected
companions' current answers, until no table gains an answer; the leader of
the component then marks every table complete. Left-recursive and mutually
recursive definitions over cyclic data terminate with their complete answer
sets. Negation over a table that is still being computed is reported as a
program that is not stratified.

```scheme
(require stdlib)
(require core.reasoning.prolog)

(define g (make-prolog))
(prolog-consult! g '((edge 1 2) (edge 2 3) (edge 3 1)
                     (<- (path ?x ?y) (path ?x ?z) (edge ?z ?y))
                     (<- (path ?x ?y) (edge ?x ?y))))
(prolog-table! g 'path 2)
(prolog-count g '(path 1 ?y))   ;; => 3
```

The test suite runs the classic benchmark set with checked answers: naive
reverse of a 30-element list in 496 inferences, n-queens solution counts
2, 10, 4, 40, 92 for n = 4..8, and the zebra puzzle's unique solution.

## Forward chaining

```scheme
(require stdlib)
(require core.reasoning.datalog)

(define db (make-datalog
  '((<- (reach ?x ?y) (edge ?x ?y))
    (<- (reach ?x ?y) (reach ?x ?z) (edge ?z ?y))
    (<- (node ?x) (edge ?x ?_))
    (<- (node ?y) (edge ?_ ?y))
    (<- (sink ?x) (node ?x) (not (edge ?x ?_))))))
(datalog-assert! db '((edge a b) (edge b c)))
(map cdr (datalog-relation db 'reach))   ;; => ((a b) (a c) (b c))
(map cdr (datalog-relation db 'sink))    ;; => ((c))
(datalog-retract! db '((edge b c)))
(map cdr (datalog-relation db 'reach))   ;; => ((a b))
```

Rules are `(<- head literal ...)` with positive atoms, negated atoms
`(not (p ...))` and comparisons `<`, `<=`, `>`, `>=`, `=`, `/=`. Every
variable of the head, of a comparison and of a negated atom (other than
`?_`, which is existential inside the negation) must occur in a positive
atom. The model is the stratified one: strata are the strongly connected
components of the predicate dependency graph, and negation inside a
component is refused.

Each stratum is one nested-clock recursive DBSP circuit
([`core.dbsp.recursive`](dbsp_recursive.md)). Atoms are filters over the base
relation or the stratum's own approximant (matched with `unify`), joins are
on shared variables, a negated atom is a difference against a semijoin with
the distinct negated relation, and the heads are unioned and made distinct.
Evaluating from empty is semi-naive iteration; `datalog-assert!`,
`datalog-retract!` and `datalog-update!` push a signed change through the
strata and return the change of the model. Deletions are exact.

| Procedure | Meaning |
|---|---|
| `(make-datalog rules [round-budget])` | Parse, check safety, stratify, compile |
| `(datalog-update! db inserts retracts)` | Signed change of the base facts; returns the model change |
| `(datalog-query db pattern)` | Substitutions for the matching model facts |
| `(datalog-facts db)`, `(datalog-relation db pred)` | The model, sorted |
| `(datalog-strata db)` | Predicates per stratum, in evaluation order |
| `(datalog-work db)` | Accumulated circuit work counters |
| `(datalog-eval rules facts)` | One-shot model |
| `(datalog-saturate-kb! kb rules signatures)` | Run rules over a `make-kb` base and assert the derived facts |

## Finite-domain constraints

```scheme
(require stdlib)
(require core.reasoning.fd)

(define m (make-fd-model))
(define x (fd-var m 'x (fd-range 0 9)))
(define y (fd-var m 'y (fd-range 0 9)))
(fd-linear! m (list (cons 1 x) (cons 1 y)) '= 10)
(fd-ne! m x y 0)
(fd-count m)                  ;; => 8
(fd-solve m)                  ;; => ((x . 1) (y . 9))
```

| Procedure | Meaning |
|---|---|
| `(make-fd-model)`, `(fd-var m name domain)`, `(fd-range lo hi)` | Variables over exact integers |
| `(fd-linear! m ((coef . var) ...) op c)`, `(fd-sum! m vars op c)` | Linear `=`, `<=`, `>=` (bounds consistent) and `/=` |
| `(fd-ne! m x y [k])` | `x /= y + k` |
| `(fd-all-different! m vars)` | Domain consistent by Regin's matching algorithm |
| `(fd-relation! m x y pred)`, `(fd-table! m vars tuples)` | Arc consistent binary relation; extensional constraint |
| `(fd-predicate! m vars pred)` | Any predicate, checked by forward checking |
| `(fd-propagate! m)`, `(fd-domain v)` | Propagate to a fixed point at the root |
| `(fd-solve m [options])`, `(fd-solutions m [limit])`, `(fd-count m)` | Depth-first search |
| `(fd-minimize m var [options])` | Branch and bound: `(value . solution)` |
| `(fd->cnf m)`, `(fd-solve-sat m)` | Direct encoding to CNF, solved by `core.reasoning.sat` |

Options: `(var-order . first-fail|dom-deg|input)` and
`(value-order . min|max)`.

## SAT

```scheme
(require stdlib)
(require core.reasoning.sat)

(define php (cnf-pigeonhole 3))            ; 4 pigeons, 3 holes
(define r (sat-solve php))
(sat-status r)                             ;; => unsat
(drat-check php (sat-proof r))             ;; => #t
(sat-status (sat-solve '((1 2) (-1 2))))   ;; => sat
```

Clauses are lists of nonzero integers (DIMACS literals). The solver is
conflict-driven clause learning: two watched literals, first-UIP learning with
clause minimization, VSIDS activities with phase saving, Luby restarts and
learned-clause reduction by literal block distance. An UNSAT result carries a
DRAT proof (every learned clause and every deletion); `drat-check` checks it
forward by reverse unit propagation with the RAT condition as fallback and
shares no code with the solver.

| Procedure | Meaning |
|---|---|
| `(sat-solve clauses [options])` | Result with `sat-status` (`sat`, `unsat`, `unknown`), `sat-model`, `sat-proof`, `sat-stats` |
| `(sat-satisfies? clauses model)`, `(sat-brute-force clauses)` | Check a model; exhaustive search for small formulas |
| `(drat-check clauses proof)` | Independent proof check |
| `(dimacs-parse text)`, `(dimacs-format clauses)`, `(drat-format proof)` | Text formats |
| `(cnf-pigeonhole n)`, `(cnf-random-ksat n m k seed)`, `(cnf-queens n)` | Standard families |

Options: `(max-conflicts . n)`, `(num-vars . n)`, `(learned-limit . n)`.

### SAT interchange admission

`dimacs-parse` requires one `p cnf` header, nonnegative exact decimal
variable/clause declarations, in-range integer literals, the declared number
of clauses and every zero terminator. Repeated/complementary literals in a
DIMACS clause refuse; `dimacs-format` canonicalizes logical input clauses
before serialization. Unused declared variables and explicitly empty clauses
remain valid.

`drat-check` returns false for malformed, improper or cyclic CNF/proof data,
unknown step tags and invalid proof literals. Logical source clauses are
deduplicated and tautologies ignored, as in the solver. Proof literal order
preserves the RAT pivot. Deletion removes one matching normalized clause,
including a unit; it never removes every duplicate clause at once.
`drat-format` rejects malformed proof data and emits deduplicated clauses.
See [the bounded admission/replay gate](../../../repro/sat-proof-admission/README.md)
for independent oracles, producer-free checking and qualification limits.

## Knowledge graphs

```scheme
(require stdlib)
(require core.reasoning.kg)

(define g (make-triple-store))
(kg-load! g '((alice knows bob) (bob knows carol) (bob age 34) (carol age 27)))
(kg-select g '(select (?x ?a) (where (?x knows ?y) (?y age ?a) (filter (> ?a 30)))))  ;; => ((alice 34))
(kg-count g '?s 'knows '?o)                                                          ;; => 2
```

The store keeps SPO, POS and OSP indexes over one triple set, so every
combination of bound positions is answered from one index. `kg-query` solves
a basic graph pattern into engine substitutions, joining patterns in a greedy
order of estimated matches. `kg-select` adds `filter`, `optional`, `union`,
`distinct`, `order-by` (with `(desc ?v)`), `limit`, `offset`, `(count)` and
`*`.

## Knowledge-graph embeddings

`(make-kge model dim)` with model `transe`, `distmult`, `complex` or
`rotate`; `(kge-train! m triples epochs lr [options])` returns the mean loss
per epoch; `(kge-score m h r t)`, `(kge-rank m h r t known)` (filtered rank)
and `(kge-evaluate m test known)` (MRR and hits at 1, 3, 10). Vectors are
stored in `core.ml.neurosymbolic` embedding tables (`kge-tables`), the same
tables `soft-unify` and `kb-attention` read. Gradients come from reverse-mode
`gradient`; `kge-loss-gradient` exposes them and the tests compare them with
closed forms and central differences.

## Incremental ansatz search

```scheme
(require stdlib)
(require core.reasoning.ansatz)

(define s (make-ansatz-search))
(ansatz-add-candidates! s (ansatz-grid '((-3 -2 -1 0 1 2 3))))
(ansatz-add-constraint! s 'euler (lambda (row) (let ((a (car row))) (= (- (* a (- a 1)) 2) 0))))
(ansatz-survivors s)   ;; => ((-1) (2))
```

The survivors of an ansatz family are maintained by one DBSP circuit:
survivors = distinct(candidates) - distinct(rejected rows). A new constraint
is evaluated on the current survivors only; new candidates are evaluated
against the active constraints only; retracting a constraint re-evaluates
only the rows that lost their last recorded rejection, reusing every
recorded verdict. `ansatz-work` reports constraint evaluations, and
`ansatz->kb!` publishes the survivors as knowledge-base facts.
