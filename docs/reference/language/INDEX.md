# Eshkol Language Core Reference

Complete reference for the Eshkol language core, targeting **v1.3.5-evolve**.

Every example in these documents was executed against a fresh build of the compiler
(`cmake --build build --target eshkol-run stdlib`) and the shown output is the real
program output. Examples are run with the JIT runner:

```sh
eshkol-run -r file.esk        # JIT-run a file
eshkol-run -e '(display (+ 2 3))'   # JIT-evaluate one expression
eshkol-run file.esk -o binary # ahead-of-time (AOT) compile to a native binary
```

Unless noted otherwise, behaviour described here is that of the **native code path**
(the LLVM JIT used by `-r`, and AOT). Where the separate bytecode VM backend differs,
it is called out explicitly.

## Contents

| # | Area | File |
|---|------|------|
| 1 | Definitions, `lambda`, `let`-family, `letrec`, named `let`, `begin` | [special-forms.md](special-forms.md) |
| 2 | `set!`, closure capture, lexical scope, shadowing | [binding-mutation-and-scope.md](binding-mutation-and-scope.md) |
| 3 | `quote`, `quasiquote`, `unquote`, `unquote-splicing` | [quote-and-quasiquote.md](quote-and-quasiquote.md) |
| 4 | Booleans and type predicates | [booleans-and-predicates.md](booleans-and-predicates.md) |
| 5 | `if`, `cond`, `case`, `when`, `unless`, `do`, `and`, `or` | [control-flow.md](control-flow.md) |
| 6 | Tail-call guarantees | [tail-calls.md](tail-calls.md) |
| 7 | `raise`, `guard`, `error`, `with-exception-handler` | [error-handling.md](error-handling.md) |
| 8 | `call/cc`, `dynamic-wind` | [continuations.md](continuations.md) |
| 9 | `match` | [pattern-matching.md](pattern-matching.md) |
| 10 | `values`, `call-with-values`, `let-values` | [multiple-values.md](multiple-values.md) |
| 11 | Function parameters: variadic, keyword args, `apply` | [functions-and-parameters.md](functions-and-parameters.md) |
| 12 | Modules: `require`/`provide`, `load`, `define-library`/`import` | [modules.md](modules.md) |
| 13 | Characters, strings, symbols, string interpolation `~{}` | [strings-chars-symbols.md](strings-chars-symbols.md) |
| 14 | Numeric tower: exact/inexact/rational/bignum/complex | [numeric-tower.md](numeric-tower.md) |
| 15 | Capability policy (`core.capabilities`) | [capabilities.md](capabilities.md) |
| 16 | Native 128-bit integers (`i128`): distinct wrapping fixed-width type | [i128.md](i128.md) |

## Known-issue conventions

Open defects are tracked in the project ledger (`.swarm/tasks/ESH-*.json`). Where a
form has a documented defect, the reference links it inline as, e.g., **ESH-0090**.
These are documented honestly as *Known Issues* — they are real, reproducible, and not
worked around in the examples.

Consolidated list of language-core known issues referenced here:

| Ledger | Summary |
|--------|---------|
| ESH-0090 | A user `(define (raise …) …)` cannot shadow the builtin `raise`. |
| ESH-0101 / ESH-0102 | Differential findings around `guard` value corruption / optimization-level-dependent crashes. |
| ESH-0109 (part) | Curried `define` sugar `(define ((f x) y) …)` is a parse error, and `raise-continuable` is an unknown function on the native path (it exists in the bytecode VM). The `cond`/`case` `=>` and `define-values` parts of that ledger entry are done. |
| — | Mutual tail recursion IS optimized, in every tail spelling (`if`, `cond`, `case`, `when`, `unless`, `and`/`or`), at any pair of arities, and on every target, to 100,000,000 hops in constant stack (ESH-0102, ESH-0102b, ESH-0102c). Two lowerings carry it: LLVM `musttail` where the target can express it, and the tail-transfer dispatcher everywhere else. Bounded exceptions — indirect tail calls through a procedure value, mutual tail calls between `letrec`-bound lambdas or from inside a named `let` loop, sites forwarding a pointer into the caller's frame, and tail calls in the body of `guard` (which R7RS does not make a tail context) — are listed in [tail-calls.md](tail-calls.md). |

Closed since v1.3.3 — kept here because earlier releases documented them, and
each is now covered by an example in the page that used to carry the warning:

| Ledger | Was | Now |
|--------|-----|-----|
| ESH-0092 / ESH-0103 | A top-level global named after a libc symbol (`free`, `log`, …) corrupted it: SIGBUS at teardown, or a `set!` silently lost. | Ordinary bindings on JIT and AOT — see [binding-mutation-and-scope.md](binding-mutation-and-scope.md). |
| ESH-0104 | Long forms `(quasiquote …)`/`(unquote …)`/`(unquote-splicing …)` were inert data. | Identical to the reader sugar — see [quote-and-quasiquote.md](quote-and-quasiquote.md). |
| ESH-0105 | Exact rational arithmetic degraded to `double` (or `0`) once a bignum operand appeared. | Stays exact: `(+ 1/3 (expt 2 70))` → `3541774862152233910273/3`, `exact?` → `#t`. |
| ESH-0106 | `'sym` inside a `guard` form compiled as a variable reference. | Both quote spellings agree everywhere — see [error-handling.md](error-handling.md). |
| ESH-0107 | Nested `quasiquote` (level ≥ 2) collapsed to `()`. | Follows the R7RS level rule — see [quote-and-quasiquote.md](quote-and-quasiquote.md). |
| ESH-0108 | stdlib `length`/`filter` crashed (SIGILL) on very large lists. | `(length (iota 1000000))` → `1000000`; `(filter even? (iota 1000000))` → 500,000 elements. |
| ESH-0109 (`=>` and `define-values`) | `=>` was parsed as a variable reference; `define-values` was unsupported. | R7RS `=>` clauses work in `cond` and `case` — see [control-flow.md](control-flow.md); `(define-values (a b) (values 1 2))` binds both. |
