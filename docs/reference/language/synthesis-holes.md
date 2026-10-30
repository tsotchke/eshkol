# Synthesis holes: `??`

An unbound `??` in an expression position is a *synthesis hole*. The compiler
fills it with a well-typed expression built from what is in scope, prints what
it chose and the other candidates, and compiles the completed program.
Design: ADR-0052 ([0052-program-synthesis.md](../../design/adr/0052-program-synthesis.md)).

```scheme
(define (add2 (x : integer) (y : integer)) : integer ??)
(display (add2 3 4))   ;; => 7
```

The compiler reports, on standard error:

```text
warning: synthesis hole ?? in 'add2' : Integer filled with (+ x y); other candidates: (* x y), (- x y), ...
```

## Expected type

The hole takes the type its position requires:

| Position | Expected type |
|---|---|
| the body of a function with a return annotation | the annotated return type |
| an argument of a function whose signature is known | that parameter's type |
| an operand of `+`, `-`, `*`, `/` | a number |
| the test of an `if` | boolean |
| an expression checked against a type | that type |
| anywhere else | unknown: any type |

## Search and ranking

Terms are built bottom-up by size, up to five nodes, from the local bindings
and the globals of known type in scope at the hole, the literals `0 1 0.0 1.0
"" #t #f`, the functions in scope with fully known signatures (never the
definition that contains the hole), and a typed table of builtin operators
(arithmetic, comparisons, `abs`, `quotient`, `remainder`, `even?`, `zero?`,
`not`, `and`, `or`, string and list operations, and `if` at the expected
type). An operator is applied only to arguments of its parameter types, and
every candidate is accepted by the type checker at the expected type before it
is offered.

Candidates are ranked by the number of distinct local bindings they use (most
first), then the number of literals (fewest first), then size, then the order
in which they were built. The first candidate fills the hole. Type information
alone often admits several completions of equal rank; the diagnostic lists
them so that the author can write the intended one.

## No completion

A hole with no well-typed completion within the size bound stops compilation
with an error naming the hole and its expected type; no code is generated and
no binary is written:

```text
error: synthesis hole ?? in 'f': no well-typed completion of type Char up to size 5
```

## Scope of the feature

- `??` is a hole only when it is unbound: a program that defines `??` uses it
  as an ordinary variable.
- Holes are completed by the native compiler, under `eshkol-run -r` (JIT) and
  when building a binary (AOT). Planned for a later rung: completion before the
  bytecode VM compiler and in the REPL, and holes directed by examples using
  [core.synthesis](../stdlib/synthesis.md).
