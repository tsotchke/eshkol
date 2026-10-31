# core.synthesis — programs from examples

`(require core.synthesis)` loads the whole facility; each part can also be
required alone: `core.synthesis.dsl`, `core.synthesis.enumerate`,
`core.synthesis.deduce`, `core.synthesis.benchmarks`. Design: ADR-0052
([0052-program-synthesis.md](../../design/adr/0052-program-synthesis.md)).
Completion of `??` holes in source programs is a compiler feature:
[synthesis-holes.md](../language/synthesis-holes.md).

```scheme
(require core.synthesis)
(define spec (syn-spec '(xs) '(list) '((((1 2 3)) . 14) (((5 -1)) . 26)) 'int))
(define r (syn-synthesize spec))
(syn-result-program r)          ; => (syn-sum (syn-map (lambda (e) (* e e)) xs))
((syn-program->procedure (syn-result-program r) '(xs)) '(2 2 2 2))   ; => 16
```

## Specifications and results

| Procedure | Meaning |
|---|---|
| `(syn-spec vars types examples goal)` | Variables, their types, examples `((input ...) . output)`, goal type. Types: `int`, `bool`, `list` (of integers), `str`, `int->int`, `int->bool`, `int*int->int`. |
| `(syn-synthesize spec [lib] [options])` | Quick enumeration, then the deductions for the goal type, then a wider enumeration. The result is re-checked on every example. |
| `(syn-enumerate spec lib [options])` | Bottom-up enumeration by cost with observational-equivalence pruning. |
| `(syn-flashfill spec [options])` | String deduction (FlashFill). |
| `(syn-deduce-map spec lib)`, `(syn-deduce-filter spec lib)`, `(syn-deduce-filter-map spec lib)` | List deductions; element functions are synthesised from per-element examples. |
| `(syn-result-found? r)`, `(syn-result-program r)`, `(syn-result-cost r)`, `(syn-result-evaluated r)`, `(syn-result-kept r)`, `(syn-result-method r)` | Result fields. `evaluated` counts candidate programs run on the examples. |
| `(syn-consistent? program spec)` | Does the program reproduce every example? |

Options are an alist: `max-cost` (8), `max-evaluated` (200000),
`observational-equivalence` (`#t`), `quick-cost` (5), `quick-evaluated`
(20000), `body-max-cost` (6), `body-max-evaluated` (60000), `max-states`
(4000), `deduction` (`#t`).

## Components and costs

A component is `(syn-component name arg-types ret-type impl cost [term])`;
`impl` must be total, returning `syn-bottom` outside its domain.
`(syn-default-library)` is the union of `syn-int-components`,
`syn-bool-components`, `syn-lambda-components`, `syn-list-components` and
`syn-string-components`. A program's cost is the sum of its component costs
(variables cost 1) and enumeration proceeds in cost order, so the cost of a
component is how a scorer guides search:

```scheme
(define guided (syn-library-costs (map (lambda (c) (syn-with-cost c 2)) (syn-default-library))
                                  '((syn-sum . 1) (syn-map . 1) ((lambda (e) (* e e)) . 1))))
(syn-result-evaluated (syn-enumerate spec guided))   ; 45 candidates, against 571 with uniform costs
```

## The DSL

Total list primitives, exported so synthesised programs run as Eshkol code:
`syn-head syn-last syn-take syn-drop syn-access syn-minimum syn-maximum syn-sum
syn-count syn-sort syn-reverse syn-cons syn-append syn-map syn-filter
syn-count-if syn-zipwith syn-scanl1`. Strings: `syn-substr`, `syn-pos`,
`syn-tokens`, `syn-upcase`, `syn-downcase`, `syn-length`.

Positions in the FlashFill language: an integer `k` (absolute), `(end m)`
(length minus `m`), or `(tok class k start|end)`, the start or end of the
`k`-th token of a class (negative `k` counts from the right). Classes: `alpha`,
`upper`, `lower`, `digit`, `alnum`, `space`, `(char c)`.

```scheme
(syn-substr "ada@example.com" '(tok (char #\@) 1 end) '(end 0))   ; => "example.com"
```

`(syn-eval program env)` evaluates program text with `env` an alist;
`(syn-program->procedure program vars)`, `(syn-program-size p)`,
`(syn-program-components p)`, `(syn-show p)`.

## Benchmarks

`syn-benchmarks` has 42 tasks: 8 integer, 16 DeepCoder-style list, 8 list
tasks for map/filter deduction, 10 FlashFill-style string tasks. Each has
training examples and held-out examples; `(syn-run-benchmarks syn-benchmarks)`
counts a task as solved only when the program reproduces both.

| Configuration | Solved |
|---|---|
| `syn-synthesize` (enumeration and deduction) | 42 / 42 |
| enumeration alone (`'((deduction . #f))`) | 29 / 42 |

On the sum-of-squares task observational-equivalence pruning evaluates 571
candidates, against 1191 without it. Measured by
`tests/stdlib/synthesis_test.esk`.
