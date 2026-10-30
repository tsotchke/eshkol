# ADR 0042: Dependent type enforcement — bidirectional checking over one core term language

- Status: Accepted — implemented for the v1.9.0-types rung (this record's stages 1-4); stages 5-7 planned
- Date: 2026-10-30
- Decision owners: language, types, frontend, bytecode VM
- Scope: the v1.9.0-types rung of [ADR 0004](0004-type-system-trajectory.md): the checking
  direction of bidirectional type checking, Pi and Sigma types, universe levels, dependent type
  families, definitional equality, and a compile mode in which type errors are compile-time errors
- Builds on: [ADR 0004](0004-type-system-trajectory.md) (one quantitative dependent type system),
  [ADR 0013](0013-gradual-type-relation.md) (the gradual type relation)
- Extended by: ADR 0043 (identity and path types), ADR 0044 (refinement types), ADR 0045 (effect
  types), ADR 0046 (row polymorphism, higher-rank and session types)

## Context

ADR 0004 assigns the general proof features to v1.9.0-types and requires that they "elaborate
through the same core judgment". Before this record the checker had the judgment's names but not
its substance:

- `TypeChecker::check` existed and had no caller: every expression was synthesized. The checking
  rule for lambdas synthesized and ignored the expected type.
- `TypeEnvironment::areEquivalent` was identity on numeric ids. Nothing normalized a type.
- `PiType` carried a "dependent" flag that nothing set. There was no Sigma type, no universe
  above U2, no type-family application (the parser read `(Matrix real 2 3)` as `Matrix` and dropped
  the arguments), and no index term language.
- `--strict-types` made type errors fatal, but the errors it could report were nominal mismatches
  found by synthesis alone. A join of two branches is `Value`, which is consistent with every type,
  so `(define (f b) : integer (if b 1 "no"))` was accepted in every mode.

## Decision

### 1. One core term language

`inc/eshkol/types/type_term.h` defines the terms every dependent judgment uses, interned in a
`TermStore` owned by `TypeEnvironment`:

| Kind | Meaning |
|---|---|
| `Universe n` | `(universe n)`; it lives in `(universe n+1)` (predicative) |
| `Const` | a nominal type (`Int64`, `Vector`, a sum, ...) |
| `Bound i` | a variable bound by an enclosing Pi or Sigma (de Bruijn index) |
| `Free` | a stable program variable, identified by a binding stamp |
| `Meta` | a metavariable a call site solves |
| `IntLit`, `Op` | index literals, `+ - *`, and the stable projections `car`, `cdr`, `vector-length` |
| `Pi` | an n-ary telescope; flags mark a rest argument and implicit binders |
| `Sigma` | a dependent pair |
| `App` | a type-family application, e.g. the length-indexed `(vector A n)` |

Terms are hash-consed with binder names excluded from identity, so alpha-equivalent terms are one
`TermRef`. **Definitional equality is normalization followed by reference comparison**:

- *delta*: a stable variable bound to an index term unfolds to it, so in `(let ((n 3)) ...)` `n`
  and `3` are the same index;
- *beta* of type families happens when an annotation is resolved: a `define-type` family is
  expanded with its arguments, so a term never contains an unexpanded user family;
- index arithmetic is put in polynomial normal form over the integers (monomials with integer
  coefficients, in a canonical order), which decides equality of polynomial index expressions
  exactly: `(+ n 1)` and `(+ 1 n)` are one term, `(* 2 (+ n 1))` is `(+ (* 2 n) 2)`.

Unification extends equality with metavariables: it solves them first-order and solves a linear
equation in one metavariable (`?n + 1 = 4`). The universe of a type is computed by the formation
rules over the levels the nominal types are registered at (`U_max(i,j)` for Pi and Sigma, the
element universe for an indexed family, `n+1` for `(universe n)`), with cumulativity.

### 2. Types are TypeIds; dependent precision is checker-only

A type the nominal ids express keeps its nominal id: a non-dependent Pi is the function signature
`makeFunctionType` builds, a non-dependent Sigma a tracked pair. Any other type is a term-backed id
in the range 40000-59999. `typeFromTerm` and `termOf` are inverse on normal forms, so definitional
equality coincides with `TypeId` equality and `areEquivalent` is the normalizing comparison.

Every judgment over types stays in `TypeRelation` (ADR 0013): two dependent types are compared on
their terms (indices by definitional equality, nominal leaves by the relation itself); a dependent
type against a nominal one is compared through its erasure. `(vector real 3) <: Vector`; a
`Vector` whose length is not known is consistent with `(vector real 3)` but not a static subtype of
it; `(vector real 2)` does not flow to `(vector real 3)`.

What code generation reads is the erasure: `codegenReprOf` maps a dependent type to the runtime
type its values have (a Pi to its signature, a Sigma to a pair, `(vector A n)` to `Vector`, a
universe to `Value`). Dependent types never change the generated code.

### 3. The surface language

```scheme
(Pi ((n : integer) (x : real)) (vector real n))      ; dependent function type
(Sigma ((n : integer)) (vector real n))              ; dependent pair type
(universe 0)                                         ; the universe of ground types
(vector real n)                                      ; length-indexed vector family
(define-type (Matrix (a : (universe 0)) (m : integer) (n : integer))
  (vector (vector a n) m))                           ; a family with declared parameter sorts
(: v (Matrix real 2 3))                              ; a declaration the definition is checked against
(define (dot (u : (vector real n)) (v : (vector real n))) : real ...)  ; n is implicit
(define (replicate (n : integer) (x : real)) : (vector real n) (make-vector n x))
```

A function's parameter annotations form a telescope: each annotation may name earlier parameters,
and the return annotation may name all of them. An index variable that no parameter binds is an
implicit binder, solved at each call site by unification. Index positions accept integer literals,
`+ - *`, stable variables, and `(car p)`, `(cdr p)`, `(vector-length v)`. A family parameter
declared `(a : (universe k))` accepts a type whose universe is at most `k`; one declared with a
value type accepts an index term.

The synthesized type of `(vector e ...)` is `(vector Value k)` and of `(make-vector n x)` is
`(vector Value n)` when `n` has an index term: the length of a vector never changes, its elements
may, so only the length is part of the synthesized type.

### 4. The checking judgment

`Γ ⊢ e ⇐ A` is synthesis followed by the **tail rule**: synthesis records the precise type of
every node in a side table, and each expression whose value becomes the value of `e` — each branch
of `if`, `cond`, `case`, `when`, `unless`, the last expression of a body — is judged against `A` at
its own precise type. The mismatch is reported at that tail, with the expected and the found type
and the source span. The judgment is applied at every place an expected type exists:

| Site | Expected type | Diagnostic label |
|---|---|---|
| a call argument | the parameter type (numeric tower converts) | `argument i of 'f'` |
| a lambda argument | the parameter's signature: the lambda's parameters take its domains, its body is checked against its codomain | `result of the lambda in argument i of 'f'` |
| a function body | the return annotation | `result of 'f'` |
| an annotated `let` binding | the annotation | `binding 'x'` |
| a definition after `(: name type)` | the declaration | `definition of 'name'` |
| `(the T e)` | `T`, judged by castability | `ascription (the T ...)` |
| a vector literal against `(vector A n)` | length `n`, each element `A` | `element i of ...` |
| `(cons a b)` against a Sigma | the first component, then the second at `a`'s value | `first/second component of ...` |

A dependent call instantiates the callee's telescope: implicit binders become metavariables the
arguments solve, each argument's index term is substituted into the later domains and the result.
`(replicate 3 1.5)` has type `(vector real 3)`. An argument that is not a stable index term leaves
a metavariable; a result that still mentions one is erased.

**The AST is unchanged by checking.** A checking pass that binds variables at types synthesis did
not see (a lambda checked against an expected signature, an annotated `let` initializer that was
not synthesized before) runs with AST writes suppressed and adds only diagnostics the synthesis did
not already make. Checking adds judgments; it never changes what code generation reads.

**Stability.** A `let`-bound variable and a parameter have a `Free` term; a `let` variable whose
initializer is an index term is defined as it while nothing assigns it. `set!` of a variable whose
term an annotation used is an obligation failure (only stable terms occur in types, ADR 0004
invariant 3); after any `set!` the variable no longer denotes an index. A `let` result type that
still mentions a variable of that `let` is erased when it leaves scope.

### 5. Enforcement and its staging

| Finding | `--types=gradual` (default) | `--types=strict` (`--strict-types`) | `--unsafe` |
|---|---|---|---|
| A failed obligation of the dependent fragment: an index not definitionally equal to the one required, a universe inconsistency, an ill-formed Pi/Sigma/family type, a family or dependent call with the wrong number of arguments, assigning an index variable | **error** | **error** | not reported |
| A nominal mismatch, including those the checking direction finds, and a vector index known statically to be outside a length known statically | warning | **error** | not reported |
| A linearity violation (unchanged) | error | error | not reported |
| An index not known statically at a typed boundary (a `Vector` of unknown length passed where `(vector real 3)` is expected) | accepted | accepted | accepted |

A dependent obligation is fatal in the default mode because only a program that writes a dependent
type can incur one: the rule cannot change the status of a program that does not use the dependent
language. Staging along the release ladder:

| Rung | Stage |
|---|---|
| v1.9.0-types | Stages 1-4 of this record and the table above. `--types=gradual|strict` on `eshkol-run`; the VM refuses the same dependent programs as the native engine. |
| v1.9.1 | Planned for v1.9.1: a runtime length check at a gradual boundary where the index is not known (ADR 0004's checked coercion), and explicit type binders erased from runtime calls (rank-1 type polymorphism through implicit `(universe k)` binders). |
| v1.9.2 | Planned for v1.9.2: shape-indexed tensor families `(tensor A (m n))` and the matrix kernels' dependent signatures in the builtin signature registry. |
| v2.0 | Planned for v2.0: `--types=strict` the default for modules that carry annotations, once the corpus measured under strict mode has no remaining status changes; the typed-HIR evidence of ADR 0004 for every accepted non-identity judgment. |

### 6. Engine parity

The bytecode VM has its own reader and compiler. It reaches the same judgment through a hook,
exactly as the linearity rule does (`inc/eshkol/backend/vm.h`): a source that writes a type in the
dependent language (a Pi, Sigma or universe; a define-type family with declared sorts; a
length-indexed vector type in an annotation, an ascription, a declaration or a define-type) is
checked by `eshkol_dependent_check_source()` before the VM compiles it, and refused when an
obligation fails. The VM compiles `(: name type)` to the unspecified value and `(the T e)` to `e`,
as the native engine does, and a top-level `define-type` leaves the unspecified value for its
context like every other expression form.

### 7. Extension points

The other v1.9.0 type records extend this core rather than adding a checker of their own: a new
type former is a `TermKind` with its cases in `normalize`, `unify`, `universeOf` and `print`, a
new surface form a branch of `parseTypeExpression` and a case of `TypeChecker::resolveTerm`, and a
new obligation is reported through `reportObligation`. Identity types (ADR 0043) use the index
terms and definitional equality for `refl`; refinement predicates (ADR 0044) are index terms over
the refined binder; effect rows (ADR 0045) annotate the Pi codomain.

## Alternatives rejected

- **A separate dependent checker.** ADR 0004 rejects one checker per feature; the dependent fragment
  is the same synthesis, the same relation and the same diagnostics path.
- **Checking mode as a second implementation of every rule.** Each rule would exist twice (synthesis
  and checking) and drift. The tail rule reuses the synthesis rules and their precise types.
- **Writing the checking pass's types into the AST.** Code generation reads the AST; types derived
  from an expected signature are not guaranteed by gradual code, so they stay in the checker.
- **Making every new finding fatal by default.** A program that does not write dependent types would
  change status silently. The measured numbers below show what `--types=strict` changes.
- **Structural equality of indices without normalization.** `(+ n 1)` and `(+ 1 n)` would differ,
  and the family `Matrix` would not be equal to its expansion.

## Evidence

- `tests/types/type_term_test.cpp` (CTest `type_term_test`): normalization, definitional equality,
  delta, binders, alpha-equivalence, unification with metavariables, universes, the TypeId bridge
  and the gradual relation over dependent types.
- `tests/typesystem/dep_*_test.esk` and `tests/typesystem/bidir_*_test.esk` (CTest
  `typesystem_dependent_corpus`): every positive fixture compiles under `--types=strict` with no
  diagnostic and runs; every negative fixture is refused with its diagnostic.
- `tests/typesystem/dependent_engine_parity_gate.sh` (CTest `dependent_type_engine_parity_gate`):
  the VM and `--emit-eskb` refuse every negative fixture; every positive fixture prints the same on
  the VM as on the native engine.
- Corpus impact: see "Measured corpus impact" below.

## Measured corpus impact

CORPUS_IMPACT_PLACEHOLDER
