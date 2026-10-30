---
kind: explanation
status: current
owner-area: types
since: v1.9.0
sources:
  - inc/eshkol/types/smt.h
  - inc/eshkol/types/refinement.h
  - lib/types/smt/exact.cpp
  - lib/types/smt/terms.cpp
  - lib/types/smt/linear.cpp
  - lib/types/smt/dpllt.cpp
  - lib/types/smt/z3_pipe.cpp
  - lib/types/refinement.cpp
  - lib/types/linear_check_bridge.cpp
  - tests/types/refinement_smt_test.cpp
  - tests/types/refinement/refinement_gate.sh
---
# ADR-0044: Refinement types with a built-in decision procedure

**Status:** Accepted (implementation for root review; lands in rung order)
**Rung:** v1.9.0-types
**Builds on:** ADR-0004 (one quantitative dependent type system), ADR-0013 (the gradual type relation)

## Context

ADR-0004 gives refinements a place in the single typing judgment: a refinement
`{x : A | phi}` is an erased dependent pair whose runtime representation is
`A`, and refinement subtyping is the entailment `Phi |= phi`, decided as the
unsatisfiability of `Phi /\ ~phi`. It orders obligation discharge (normalise,
local theories, then a solver for the decidable fragment, never success on a
timeout) and asks for deterministic solver budgets. The roadmap's v1.9.0 entry
names the surface, `(Refine Integer (> x 0))`, "with SMT solver integration".

Two constraints shape the design. Eshkol builds on Linux, macOS, Windows and
WebAssembly, and the type checker runs inside the compiler on all of them, so
the procedure that decides the obligations cannot be an external dependency.
And an external solver is part of the trusted computing base, so its answers
should be checked against an independent procedure rather than trusted alone.

## Decision

### Surface

- `(Refine T phi)` and `(Refine T x phi)` are type expressions (parser:
  `HOTT_TYPE_REFINE`, payload base / binder / predicate). They may appear
  wherever a type may: parameter and result annotations, `(: f (-> ...))`
  signatures, `let` bindings including named-let loop variables, `(the T e)`,
  and `define-type` aliases.
- Without an explicit binder, the binder is the annotated name when the
  predicate mentions it, else the single free variable of the predicate that
  names nothing in scope; anything else is an error that asks for the
  three-argument form.
- Predicates are pure and total: integer, rational and real arithmetic,
  comparisons, `not`/`and`/`or`/`if`, `abs`/`min`/`max`, `quotient`/
  `remainder`/`modulo`, `floor`/`ceiling`/`round`/`truncate`, `even?`/`odd?`,
  `zero?`/`positive?`/`negative?`, `vector-length`/`string-length`/`length`/
  `tensor-length`, `null?`/`pair?`, and equality. A predicate using anything
  else is rejected; arbitrary Scheme predicates are not refinements.
- Type resolution erases a refinement to its base (TypeChecker::resolveType,
  the codegen resolver). Code generation is unchanged.

### Checking

One pass, after ordinary type checking (`checkProgramRefinements`), is the
entailment component of ADR-0004's judgment. It walks each top-level form
symbolically and keeps two kinds of facts: path facts (assumed refinements of
parameters and call results, branch conditions; scoped, and carried out of a
branch as `cond -> facts`) and definitions (constraints that only introduce a
fresh variable: the quotient and remainder of a division by a constant, the
rounding interval of an inexact operation, a length's non-negativity).

Obligations: arguments to refined parameters (with earlier parameters
substituted, so refinements may depend on them), refined results, annotated
`let`/loop/`set!` values, `(the ...)` ascriptions, and, inside functions that
carry refinement annotations, vector/string/list/tensor index bounds, sizes
given to `make-vector`/`make-string`, and exact division by zero.

Soundness choices: a variable assigned by `set!` is re-read as a fresh value
(satisfying its declared refinement, if any); floating-point results are fresh
reals within the round-to-nearest interval `|r - s| <= 2^-53 |s| + 2^-1000` of
the exact result `s`; a value of unknown numeric type is treated as inexact;
forms the walk does not model are still walked for obligations, in a scope that
hides outer bindings and facts; recursion is checked against the function's own
contract (partial correctness).

Verdicts and severity:

| Verdict | Meaning | Severity |
|---|---|---|
| proved | `Phi /\ ~phi` unsatisfiable | none |
| contradicted | also `Phi /\ phi` unsatisfiable | error, every mode but `--unsafe` |
| not proved | a counterexample exists | error inside annotated functions; a gradual-boundary type issue (warning, error under `--strict-types`) in unannotated code and for `(the ...)` |
| undecided | the solver answered Unknown | type issue; never success |

Errors stop code generation (as linearity violations do). Counterexamples are
printed from the solver's model in source names. The bytecode VM's source and
`--emit-eskb` paths run the same pass through the front-end bridge
(`eshkol_refinement_check_source`), so both engines accept and refuse the same
programs.

### The decision procedure

Quantifier-free formulas over linear integer and rational arithmetic and
uninterpreted functions, decided by a DPLL(T) search:

- **Rational arithmetic:** the general simplex of Dutertre and de Moura over
  exact rationals, strict bounds as delta-rationals, Bland's rule.
- **Integer arithmetic (Presburger):** Fourier-Motzkin projection of the
  rational variables (exact over Q for any values of the rest), then the Omega
  test: exact equality elimination (the mod-hat step for non-unit
  coefficients), exact projection, dark shadow, real shadow and splinters.
  Complete for conjunctions; each elimination records what it removed, so a
  model is rebuilt in reverse.
- **Uninterpreted functions:** congruence closure over purified definitions
  `r = f(v...)`.
- **Combination:** delayed theory combination. Every pair of numeric variables
  shared by the two theories gets an interface equality that the search
  decides, so both theories check the same arrangement (complete for these
  stably infinite theories). A false arithmetic equality reaches the
  arithmetic engine as a strict inequality chosen by the search, so the
  engines see conjunctions only.
- **Search:** chronological DPLL with unit propagation; theory checks after
  each propagation fixpoint; conflicts shrunk to a minimal core by deletion
  and learned.
- **Numbers:** `Integer` is Eshkol's bignum (lib/core/bignum.cpp) with an
  int64 fast path; `Rational` is a normalised pair. Nothing is rounded.
- **Answers:** Unsat is a proof relative to the procedure. Sat carries a model
  re-evaluated against the original formula with exact semantics; nonlinear
  products and Scheme division by a non-constant are searched as uninterpreted
  symbols and evaluated exactly at validation, so a spurious model becomes
  Unknown, never Sat. The step budget is a deterministic count
  (`ESHKOL_REFINEMENT_BUDGET`, default 2,000,000); exhaustion is Unknown.

### The optional z3 backend

`-DESHKOL_REFINEMENT_Z3=ON` (POSIX) compiles a backend that sends each query
as SMT-LIB2 over a pipe to `z3 -in -smt2` with a deterministic `rlimit`. The
SMT-LIB2 rendering defines Scheme's `quotient`/`remainder`/`modulo` exactly,
so z3 reads the semantics model validation uses. `ESHKOL_REFINEMENT_SOLVER`
selects `internal` (default), `z3`, or `cross` (both; a disagreement on a
decided query is an error).

## Verification

- `refinement_smt_test` (CTest): 16,416 known-answer queries. Every single
  linear atom over two integer variables in a box, every second pair from a
  grid of 150 atoms, and 3,000 seeded random Boolean combinations over three
  variables (a third with divisibility conjuncts that exercise the mod-hat
  step), each compared with brute force over every point of the box; 2,000
  rational conjunctions decided by both the simplex and projection, which must
  agree, with every model checked; 6,000 unbounded integer, UF+LIA and mixed
  integer/rational formulas whose models are checked; named cases (parity,
  Pugh's dark-shadow example, congruence, non-convex combination,
  over-approximated products, a tiny budget giving Unknown).
- The same corpus compared with z3 5.1.0 query by query: 16,413 of 16,413
  decided queries give the same sat/unsat answer (the remaining three are
  checked for Unknown/Unsat directly).
- `refinement_types_gate` (CTest): one accepted program with fourteen
  obligations kinds and exact expected output, ten rejected programs each
  pinning its diagnostic, a gradual-boundary program (runs with a warning,
  refused under `--strict-types`), on the native engine and the bytecode VM.

## Consequences

- Programs without refinement types are not walked; their compilation is
  unchanged.
- The procedure is part of the trusted computing base for refinement
  verdicts. The z3 cross-check and the brute-force corpus bound that trust
  until proof certificates (ADR-0047) can be checked independently.
- Refinement checking is a separate pass over the program, not yet threaded
  through the bidirectional core of ADR-0042; when that core lands, the same
  facts and obligations move into its judgment unchanged.

## Planned

- Runtime checks for reifiable unproved refinements at gradual boundaries
  (ADR-0004 discharge step 5), and loop-invariant inference for unannotated
  named-let variables: planned for v1.9.1.
- Refinements over tensor shapes beyond total length, and user-defined pure
  functions in predicates (after the totality checker): planned for v1.9.2.
