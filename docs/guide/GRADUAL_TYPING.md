---
kind: guide
status: current
owner-area: types
since: v1.3.5
sources:
  - lib/types/type_relation.cpp
  - lib/types/type_checker.cpp
  - tests/typesystem
---
# Gradual Typing in Eshkol

*A user guide to the type checker as it stands in v1.3.5-evolve.*

Eshkol is a Scheme: every program runs without a single type annotation, and
every value carries its type at run time. On top of that sits an optional
static checker. You annotate what you want checked, the checker reports what it
can prove is wrong, and everything it cannot see stays dynamic. This guide is
about using that checker: where to put annotations, what is checked and where,
how the checker decides that two types fit, and how to read what it prints.

The theory, the type representation and the runtime tags are in
[the type system breakdown](../breakdown/TYPE_SYSTEM.md). The design record for
the rules on this page is
[ADR 0013 — One gradual type relation](../design/adr/0013-gradual-type-relation.md).

Every example on this page is executed by the documentation example gate on the
JIT and as a compiled binary, and the output shown is what the build prints.

## Contents

1. [Annotations](#1-annotations)
2. [Warnings and errors](#2-warnings-and-errors)
3. [What is checked, and where](#3-what-is-checked-and-where)
4. [Fitting one type to another](#4-fitting-one-type-to-another)
5. [Function types](#5-function-types)
6. [Branches join](#6-branches-join)
7. [Loops are typed by what they carry](#7-loops-are-typed-by-what-they-carry)
8. [Telling the checker what you know](#8-telling-the-checker-what-you-know)
9. [Reading a diagnostic](#9-reading-a-diagnostic)
10. [Quick reference](#10-quick-reference)

## 1. Annotations

A parameter is annotated by writing it as `(name : type)`. A return type follows
the parameter list after a `:`. Both are optional and independent.

```scheme
(define (area (w : number) (h : number)) : number
  (* w h))

(display (area 3 4))
(newline)
(display (area 2.5 4))
(newline)
(display (area 1/2 4))
(newline)
```

```text
12
10
2
```

`number` accepts the whole numeric tower, so the exact integer, the flonum and
the exact rational are all legal arguments, and the arithmetic keeps its usual
exactness. Annotations never change what a program computes. They tell the
checker what you intend, and they let the compiler pick a direct instruction
where it would otherwise dispatch on the runtime tag.

`lambda` takes the same syntax: `(lambda ((x : number)) : number (* x x))`.
The type names are those of the type registry: the numeric tower (`integer`,
`real`, `float64`, `number`, ...), `string`, `symbol`, `boolean`, `char`,
`list`, `vector`, `procedure`, function types written `(-> argument ... result)`,
and the types you introduce with `define-type`. The full grammar is in
[the language specification, section 3.6](../COMPLETE_LANGUAGE_SPECIFICATION.md).

## 2. Warnings and errors

By default a type mismatch is a **warning**. The program is compiled and run
anyway, because the checker may simply know less than you do. This program
warns and still runs:

```scheme
(define (area (w : number) (h : number)) : number (* w h))

(define (report flag)
  (when flag
    (area "wide" 3)))

(report #f)
(display "still runs")
(newline)
```

```text
still runs
```

The compiler writes this on standard error:

```console
[WARN] Type warning: argument 1 of 'area': expected Number, got String (line 5:11)
   WARNING: HoTT: 1 type warnings detected (gradual typing continues)
```

Two flags change the policy:

| Flag | Effect |
|---|---|
| `--strict-types` | Every type diagnostic is an error. No code is generated and the exit status is nonzero. |
| `--unsafe` | Skip all type checks. |

Under `--strict-types` the same program stops at compile time:

```console
[ERROR] Type error: argument 1 of 'area': expected Number, got String (line 5:11)
     ERROR: HoTT: 1 type errors detected (strict mode)
     ERROR: Refusing to generate code for a program with type errors (--strict-types)
```

One family of diagnostics is an error in every mode: **linearity**. Using a
linear value such as a `Qubit` twice, or dropping it, is a compile-time error
even without `--strict-types`, because no run-time check could recover from it.
See [the language guide](../ESHKOL_LANGUAGE_GUIDE.md) for the linear types.

## 3. What is checked, and where

A call is checked against the callee's annotations **wherever the call is
written**. The checker descends into the body of every control form, so a call
inside a `cond` clause, a `do` step or a `guard` handler is examined exactly as
it would be at top level. The warning above came from inside a `when`.

```scheme
(define (area (w : number) (h : number)) (* w h))

(define (describe flag)
  (cond (flag (area "wide" 3))
        (else 0)))

(define (tally n)
  (do ((i 0 (+ i 1))
       (total 0 (+ total (area i "tall"))))
      ((= i n) total)))

(define (checked x)
  (guard (e (#t (area "handler" 1)))
    (when (> x 0) (area x "body"))))

(display (describe #f))
(newline)
```

```text
0
```

Each of the four wrong calls is reported, with its own position:

```console
[WARN] Type warning: argument 1 of 'area': expected Number, got String (line 4:21)
[WARN] Type warning: argument 2 of 'area': expected Number, got String (line 9:34)
[WARN] Type warning: argument 2 of 'area': expected Number, got String (line 14:27)
[WARN] Type warning: argument 1 of 'area': expected Number, got String (line 13:23)
```

The positions the checker examines:

| Form | Examined |
|---|---|
| `begin`, function and `let` bodies | every expression, not only the last |
| `if`, `cond`, `case`, `when`, `unless` | tests, keys and every branch body |
| `and`, `or` | every operand |
| `match` | the scrutinee and every clause body |
| `do` | initialisers, steps, the test, the result and the body |
| `guard`, `raise` | the body, the handler clauses, the raised operand |
| `set!` | the assigned value |
| quasiquote | every `unquote` and `unquote-splicing` escape, at the right nesting level; quoted data is not evaluated and not examined |
| `call/cc`, `dynamic-wind`, `parameterize` | the procedure and all three thunks |
| `values`, `call-with-values`, `let-values`, `let*-values`, `receive`, `define-values` | operands, producer, consumer and body |
| `with-region` and the ownership forms (`borrow`, `owned`, `move`, `shared`, `weak-ref`) | the body or operand |
| the calculus operators (`derivative`, `gradient`, `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`, `directional-derivative`, `taylor`, `derivative-n`) | the function, the point, the direction and the order |
| a call through a computed callee, such as `((pick-operation flag) 1 2)` | the callee expression and the arguments |
| tensor literals and arithmetic with more than two operands | every element and operand |

A branch body is checked in its own scope. What the checker learns inside one
`cond` clause does not leak into the next clause or into the enclosing function.
A test does refine what follows it, as section 8 shows.

## 4. Fitting one type to another

The checker asks one question at every boundary (an argument, a return, an
assignment to an annotated binding): does the type it derived fit the type that
was asked for? Two ideas answer it.

**Subtyping** is the static part: every `integer` is a `number`, every
`Pair<Float64, Float64>` is a `Pair`, every type is a `Value`. A subtype always
fits where its supertype is expected.

**Consistency** is the gradual part. When the checker does not know a type, it
calls it `Value`, the dynamic type. `Value` is *consistent* with every type: it
fits anywhere, and anything fits it. That is not a claim that the program is
right. It says the checker has no evidence that it is wrong, so the run-time tag
is what decides.

The rule the checker applies is the combination, *consistent subtyping*: the
derived type fits if it is a subtype of the expected type, treating every
unknown part as acceptable. So:

- a known mismatch is reported: `String` does not fit `Number`;
- an unknown is accepted: `(car xs)` has type `Value`, and fits `number`;
- a partly known type is checked where it is known: pair components are
  covariant, so a pair fits wherever each of its components fits, and a
  function type is compared parameter by parameter (section 5).

The numeric tower is flat at a call. A parameter declared `integer` accepts an
argument the checker knows only as a `number`, and the other way round, because
every member of the tower is a number and the arithmetic dispatches on the
run-time tag. Inside a function type the components are compared exactly, as
section 5 shows.

```scheme
(define (first-number (xs : list)) : number
  (car xs))

(display (first-number (list 7 8)))
(newline)
```

```text
7
```

`(car xs)` is `Value`, so the return annotation accepts it and nothing is
printed on standard error. Return annotations, arguments, `if` and `cond`
results, and `the` ascriptions all use this one relation, so a type that fits in
one position fits in the others.

## 5. Function types

A procedure's signature is a type, written `(-> argument ... result)`, and it
appears in diagnostics in that form. A parameter can require one:

```scheme
(define (twice (f : (-> number number)) (x : number))
  (f (f x)))

(define (halve (x : number)) : number (/ x 2))
(define (int-only (x : integer)) : number (+ x 1))
(define (shout (s : string)) : string s)

(display (twice halve 10))
(newline)

(define (never-called)
  (twice int-only 1)
  (twice shout 1))
```

```text
5/2
```

```console
[WARN] Type warning: argument 1 of 'twice': expected (-> Number Number), got (-> Int64 Number) (line 12:10)
[WARN] Type warning: argument 1 of 'twice': expected (-> Number Number), got (-> String String) (line 13:10)
```

`halve` fits. `int-only` does not: `twice` promises to call `f` with any
number, and `int-only` only accepts integers. Function types are
**contravariant in their parameters** (a function that accepts *more* fits where
one that accepts less is expected) and **covariant in their result** (a function
that returns *less* fits where one that returns more is expected). A function
with a different number of parameters never fits. A parameter or result that is
unknown stays acceptable, so an unannotated `(lambda (x) x)` fits
`(-> number number)`.

`procedure` is the top of the function types: any procedure fits it, and it
says nothing about the signature.

## 6. Branches join

The type of an expression with several branches is the **join** of the branch
types: the most specific type that every branch fits. Two strings join to
`String`; an exact integer and a flonum join to `Number`; a number and a string
have nothing more specific in common than `Value`. The rule is the same for
`if`, `cond`, `case`, `match`, `when`, `unless`, `and` and `or`, so rewriting an
`if` as a `cond` cannot change a program's types.

```scheme
(define (pick flag) (if flag 1 2.5))
(define (label (s : string)) (string-length s))

(define (never-called flag)
  (label (pick flag)))

(display (pick #t))
(newline)
```

```text
1
```

```console
[WARN] Type warning: argument 1 of 'label': expected String, got Number (line 5:11)
```

A form that may run no branch contributes the value it then yields: `cond`,
`case`, `when` and `unless` without a matching branch yield `#f`, so
`(when flag 1)` is the join of `Int64` and `Boolean`. `match` raises and
`guard` re-raises when nothing matches, so they add nothing. A branch the
checker cannot type contributes `Value` and does not fail the form.

The result of a recursive procedure is found the same way: the join of its base
cases, with the recursive calls taking the type being inferred.

## 7. Loops are typed by what they carry

A named `let` binds a loop procedure whose parameters have no annotations. The
checker gives each parameter the join of its initial value and of **every
argument the loop passes back to it**, so an accumulator may start as one thing
and grow into a wider one.

```scheme
(define (mean-pair xs)
  (let loop ((rest xs) (acc (cons 0.0 0)))
    (if (null? rest)
        (/ (car acc) (cdr acc))
        (loop (cdr rest)
              (cons (+ (car acc) (car rest)) (+ (cdr acc) 1))))))

(display (mean-pair (list 1.0 2.0 3.0 6.0)))
(newline)
```

```text
3
```

`acc` starts as the literal pair `(0.0 . 0)` and is fed the result of two
additions on every turn. The loop carries a pair of numbers, that is what `acc`
is typed as, and no diagnostic is printed.

The join has to be informative. If the only thing the seed and an argument have
in common is `Value`, the parameter keeps its type and the argument is reported:

```scheme
(define (count-to n)
  (let loop ((i 0))
    (if (< i n) (loop (+ i 1)) "done")))

(display (count-to 3))
(newline)

(define (never-called n)
  (let loop ((i 0))
    (if (< i n) (loop "three") i)))
```

```text
done
```

```console
[WARN] Type warning: argument 1 of 'loop': expected Int64, got String (line 10:23)
```

Three refinements:

- An annotated loop parameter is a contract. It is never widened, and a sum-type
  annotation on a named-`let` parameter is honoured across iterations.
- A `#f` seed means "nothing yet". A parameter seeded with `#f` and later given
  a value is typed `Value` and accepted.
- A `do` loop variable is the join of its initialiser and its step, and adopts
  the join even when it reaches `Value`, because a `do` step is an assignment,
  not an argument being checked.

## 8. Telling the checker what you know

**Predicates narrow.** Inside a branch guarded by `number?`, `integer?`,
`string?`, `symbol?`, `pair?`, `null?`, `vector?` or `procedure?`, the tested
variable has that type. Narrowing works in `if`, `cond`, `when` and across the
later operands of `and`, and a `match` pattern `(? pred name)` narrows `name`
the same way. A `set!` of the variable cancels it.

**`the` ascribes.** `(the type expr)` tells the checker that `expr` has `type`.
It costs nothing at run time: the generated code is that of `expr` alone. It is
accepted whenever the two types can overlap, and reported only when no value
could have both.

```scheme
(define (describe x)
  (cond ((number? x) (+ x 1))
        ((string? x) (string-length x))
        (else 0)))

(display (describe 41))
(newline)
(display (describe "four"))
(newline)

(define (label (s : string)) : integer (string-length s))
(define mixed (list 1 "two" 3.0))

(display (label (the string (cadr mixed))))
(newline)
```

```text
42
4
3
```

## 9. Reading a diagnostic

```console
[WARN] Type warning: argument 1 of 'twice': expected (-> Number Number), got (-> Int64 Number) (line 12:10)
```

| Part | Meaning |
|---|---|
| `[WARN] Type warning` | Severity. `[ERROR] Type error` under `--strict-types`, and for linearity in every mode. |
| `argument 1 of 'twice'` | The boundary that was checked: a numbered argument of a named procedure, a `body type ... doesn't match return annotation ...`, or an `ascription (the T ...) contradicts the inferred type ...`. |
| `expected (-> Number Number)` | The type the annotation asked for, in the checker's spelling: `number` prints as `Number`, `integer` as `Int64`, `real` as `Float64`. |
| `got (-> Int64 Number)` | The type the checker derived for the expression. |
| `(line 12:10)` | Line and column of the offending expression in the file being compiled. |

Types print structurally: `Pair<Float64, Float64>` is a pair whose components
are known, `Pair` one whose components are not; `(-> String ... Value)` is a
variadic procedure; `Function` is a procedure whose signature is unknown;
`Value` is the dynamic type. A `got Value` never appears in a warning, because
`Value` fits everything.

The summary line `HoTT: N type warnings detected (gradual typing continues)`
follows the individual diagnostics. A program with no diagnostics prints
nothing.

When a warning is wrong for your program, in order of preference: narrow with a
predicate, so the run-time test and the checker agree; ascribe with `the` where
you know something the checker cannot; widen the annotation if the procedure
really does accept more than it declared.

## 10. Quick reference

| You write | The checker does |
|---|---|
| `(define (f (x : T)) ...)` | checks every call's argument against `T`, wherever the call is written |
| `(define (f ...) : T ...)` | checks the body's type against `T` |
| no annotation | treats the binding as `Value`: nothing is checked, nothing is reported |
| `(if c a b)`, `(cond ...)`, `(case ...)`, `(match ...)` | result is the join of the branches |
| `(let loop ((acc seed)) ...)` | `acc` is the join of `seed` and every argument passed back |
| `(do ((v init step)) ...)` | `v` is the join of `init` and `step` |
| `(number? x)` in a test | `x` is `Number` in the guarded branch |
| `(the T expr)` | takes `expr` as `T`; free at run time; reported only if impossible |
| `--strict-types` | diagnostics are errors; no code is generated |
| `--unsafe` | no type checks |

## See also

- [Type system breakdown](../breakdown/TYPE_SYSTEM.md): the type
  representation, the universe hierarchy, runtime tags and the implementation
  files.
- [ADR 0013 — One gradual type relation](../design/adr/0013-gradual-type-relation.md).
- [Language specification, section 3.6](../COMPLETE_LANGUAGE_SPECIFICATION.md):
  annotation grammar, `the`, linear types.
- [`eshkol-run` reference](../reference/runtime/eshkol-run.md): `--strict-types`
  and `--unsafe`.
- [Upgrading to v1.3.5](../UPGRADING.md): what the wider checking means for a
  program written against v1.3.4.
