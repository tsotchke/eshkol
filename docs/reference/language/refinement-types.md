# Refinement types

A refinement type is a type together with a proposition its values satisfy:
`(Refine Integer (> x 0))` is the positive integers. The compiler proves every
place a value must satisfy a refinement, with its own decision procedure, before
it generates code. Design: [ADR-0044](../../design/adr/0044-refinement-types.md).

## Syntax

```scheme
(Refine base predicate)          ; binder inferred
(Refine base binder predicate)   ; binder explicit
```

`base` is any type. The runtime representation is `base`'s; a refinement adds
no runtime work. The binder names the value being refined:

- on a parameter, the parameter's own name when the predicate mentions it;
- otherwise the single free variable of the predicate that names nothing in
  scope (`(define-type Pos (Refine Integer (> x 0)))` binds `x`);
- in any other case write the binder explicitly.

Refinements may appear wherever a type annotation may: parameters, results,
`(: f (-> ...))` signatures, `let` bindings (including named-`let` loop
variables), `(the T e)`, and `define-type` aliases. A parameter's refinement
may mention earlier parameters:

```scheme
(define-type Pos (Refine Integer (> x 0)))

(define (safe-div (n : Integer) (d : (Refine Integer (not (= d 0))))) : Integer
  (quotient n d))

(define (clamp (lo : Integer) (hi : (Refine Integer (>= hi lo))) (x : Integer))
  : (Refine Integer v (and (<= lo v) (<= v hi)))
  (cond ((< x lo) lo) ((> x hi) hi) (else x)))

(define (get (v : (vector integer))
             (i : (Refine Integer (and (<= 0 i) (< i (vector-length v))))))
  (vector-ref v i))
```

## The predicate language

Predicates are pure and total: `+ - * /`, `< <= > >= =`, `not and or if`,
`abs min max`, `quotient remainder modulo`, `floor ceiling round truncate`,
`even? odd? zero? positive? negative?`, `vector-length string-length length
tensor-length`, `null? pair?`, `eq? eqv? equal?`, and integer, rational and
real literals. Anything else in a predicate is an error.

## What is proved

| Obligation | Where |
|---|---|
| an argument satisfies its parameter's refinement | every call |
| a result satisfies the function's refined result type | every function with one |
| a bound value satisfies its annotation | annotated `let`, named-`let` loop variables (initial value and every iteration), `set!` |
| the ascribed refinement holds | `(the (Refine ...) e)` |
| `0 <= i < (length)` | `vector-ref`/`vector-set!`, `string-ref`/`string-set!`, `list-ref`/`list-tail`, `vref`/`tensor-ref` inside functions with refinement annotations |
| size `>= 0`, divisor `/= 0` | `make-vector`/`make-string`; exact `/`, `quotient`, `remainder`, `modulo` inside functions with refinement annotations |

Facts come from parameter and result refinements, branch conditions (`if`,
`cond`, `when`, `unless`, `and`, `or`), `let` bindings, arithmetic, and the
lengths of vectors, strings and lists built by `vector`, `make-vector`, `list`,
`cons`, `append`, `map` and friends. Integers are exact (Eshkol integers do not
overflow); floating-point results are bounded by their rounding error, so
`(- x 1.0)` is provably positive when `x > 2.0` but not when `x > 1.0`.
A variable assigned with `set!` is not assumed to keep its earlier value.

## Diagnostics

```text
[ERROR] Refinement error: refinement not proved: result of 'next' must satisfy (> r x) in 'next'; counterexample: x = 101 (line 2:1)
[ERROR] Refinement error: refinement violated: argument 2 (d) of 'safe-div' must satisfy (not (= d 0)), and no execution reaching this point satisfies it (for example any value) (line 4:28)
[WARN] Type warning: refinement not proved at a gradual boundary: argument 2 (d) of 'safe-div' must satisfy (not (= d 0)) in 'unknown-divisor'; counterexample: k = 0 (line 5:42)
```

- **violated**: every execution reaching the point breaks the refinement. An
  error in every mode but `--unsafe`; no code is generated.
- **not proved** inside a function that carries refinement annotations: an
  error, with a counterexample.
- **not proved** in unannotated code calling an annotated function, or for a
  `(the ...)` ascription: a gradual-boundary type issue, a warning that becomes
  an error under `--strict-types`. The refinement is assumed afterwards.
- **undecided** (the solver's step budget ran out, or nonlinear arithmetic):
  a type issue; never treated as proved.

The bytecode VM's source and `--emit-eskb` paths run the same check, so both
engines accept and refuse the same programs.

## Solvers

The built-in procedure decides linear integer arithmetic (the Omega test),
linear rational arithmetic (exact simplex), and uninterpreted functions
(congruence closure), combined by DPLL(T); it needs nothing outside the
compiler. A build configured with `-DESHKOL_REFINEMENT_Z3=ON` can also use z3:

| Variable | Effect | Default |
|---|---|---|
| `ESHKOL_REFINEMENT_SOLVER` | `internal`, `z3`, or `cross` (both; a disagreement is an error) | `internal` |
| `ESHKOL_REFINEMENT_BUDGET` | deterministic step budget per query | `2000000` |
| `ESHKOL_Z3` | z3 executable for the z3 backend | the one found at configure time |
