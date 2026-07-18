# Eshkol Language Core Reference

Complete reference for the Eshkol language core, targeting **v1.3.0-evolve**.

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
form has a documented defect, the reference links it inline as, e.g., **ESH-0104**.
These are documented honestly as *Known Issues* — they are real, reproducible, and not
worked around in the examples.

Consolidated list of language-core known issues referenced here:

| Ledger | Summary |
|--------|---------|
| ESH-0090 | A user `(define (raise …) …)` cannot shadow the builtin `raise`. |
| ESH-0092 | Top-level globals are emitted as raw C symbols; a name colliding with libc (e.g. `free`) crashes at teardown (SIGBUS). |
| ESH-0104 | Long forms `(quasiquote …)`/`(unquote …)`/`(unquote-splicing …)` are not wired; only the reader sugar ``` ` ``` `,` `,@` evaluate. |
| ESH-0105 | Exact rational arithmetic silently degrades to `double` (or `0`) once a bignum operand is involved. |
| ESH-0106 | `'sym` reader sugar inside a `guard` form (clause body or `raise` argument) is compiled as a variable reference. |
| ESH-0107 | Nested `quasiquote` (level ≥ 2) collapses to `()`. |
| ESH-0108 | stdlib `length`/`filter` are non-tail-recursive; they crash (SIGILL) on very large lists. |
| ESH-0101 / ESH-0102 | Differential findings around `guard` value corruption / optimization-level-dependent crashes. |
| — | Mutual tail recursion is **not** optimized; ping-pong recursion crashes around 500k depth (self tail recursion is fully optimized). |
