---
kind: reference
status: current
owner-area: types
since: v1.4.0-connection
sources:
  - lib/frontend/ownership.cpp
  - inc/eshkol/frontend/ownership.h
  - lib/types/linear_check_bridge.cpp
  - tests/memory/ownership_default_modes.py
  - tests/memory/ownership_lexical_scope_test.py
---
# Ownership: `owned`, `move`, `borrow`

An owned value has exactly one owner at a time. `(owned e)` makes the binding
it initializes the owner of `e`'s value, `(move x)` hands that value to a new
owner and leaves `x` moved, and `(borrow x body ...)` lends `x` to its body
and gives it back.

The ownership checker runs before any code is generated, with the same verdict
function on every engine: the JIT (`eshkol-run -r`), AOT (`eshkol-run -o`),
the REPL (each input is checked against the session's earlier inputs) and the
hosted bytecode compiler. A refused program prints `Ownership analysis failed:`
followed by one line per rule it breaks, exits with status 1 and writes no
artifact. `--unsafe` turns the static check off for legacy programs.

## What is guaranteed

For every program the checker accepts, no owned value is read or moved through
any name after it has been moved. That covers the owner itself and every other
way to reach the value:

- another binding built from it, at any depth: `(define y (cons x '()))`;
- a procedure or closure whose body names it, including a closure that
  outlives the `let` it was made in;
- a lambda parameter that receives it, from an immediate application or from a
  procedure such as `map` or `for-each` that calls the lambda;
- a named-`let` variable that a loop call passes it to;
- a global defined later in the same unit, or in an earlier REPL input.

A move is a transfer, so a second move of the same owner is refused as well. In
an engine whose owner is the only holder of a value, these two rules are what
rule out a use after free and a double free of an owned value.

## The rules

**Use after move.** Reading a moved owner is refused (`Use of moved value 'x'`),
and so is reading any binding that may refer to it (`Use of 'y', which refers
to moved value 'x'`). A value moved on one branch only is *maybe moved*: every
later read is refused (`which may refer to moved value`).

**Moves.** A second move is refused (`Double move of 'x'`). Moving an owner
while it is borrowed is refused (`Cannot move 'x' while it is borrowed`). A
procedure or loop body may run more than once, so a `move` inside one may only
take an owner bound inside it: a procedure that moves a global owner is refused
(`Cannot move 'x' inside a procedure or loop body that may run more than
once`). An immediately applied `lambda` runs once and may move an owner from
its enclosing scope.

**Copies.** `(define y x)` and `(set! y x)` of an owner or a borrowed value are
refused; use `move`.

**Borrows.** The value of a `borrow` may not be the borrowed owner itself, nor a
closure that names it (`Borrowed value 'x' escapes its borrow scope`). A
borrowed owner may not be assigned.

**References the checker cannot follow.** A value that refers to an owner may
not reach a place the checker does not see:

| Place | Diagnostic |
|---|---|
| `set!` of any variable | `Assignment to 'g' would keep a reference to owned value 'x'` |
| An argument that a user procedure stores past the call (a `set!` of a variable it does not bind, a call that keeps it, a `raise`) | `'f' may keep a reference to owned value 'x' after the call` |
| A runtime builtin that keeps an argument: a mutator such as `vector-set!` (every argument after the object it mutates), and thread, channel, promise, parameter and registration builtins (every argument) | `'vector-set!' may keep a reference to owned value 'x' after the call` |
| A procedure the checker cannot resolve: a computed operator, a parameter, a variable rebound with `set!` | `Owned value 'x' is passed to a procedure the ownership checker cannot resolve` |
| A raised value (under `--strict-resources`) | `Raised value refers to owned value 'x'` |
| `eval` | `eval may name owned value ...` |

A user procedure's effect on its arguments is summarized from its body: which
parameters it may keep past the call, and which procedure parameters it calls
with which other arguments. The summary is flow-insensitive and conservative.
Mutually recursive procedures are summarized together to a fixed point, and a
procedure the checker cannot summarize is treated as keeping every argument.
Procedures from a precompiled library are classified by name, as runtime
builtins are.

Passing an owned value to a procedure that does not keep it is accepted:

```scheme
(define x (owned (list 1 2)))
(define (show v) (display v) (newline))
(borrow x (show (cadr x)))
(define y (move x))
(show (car y))
```

**Forms the checker does not model.** `case`, `match`, `do`, `guard`,
`raise`, `call/cc`, `dynamic-wind`, `values`, `call-with-values`,
`let-values`, `case-lambda`, `parameterize`, quasiquote and the operator forms
of automatic differentiation and tensors are accepted as long as they neither
contain `owned`, `move` or `borrow` nor name a binding that refers to an owner.
Otherwise they are refused (`The ownership checker does not follow case that
refers to owned value 'x'`); bind the value with `let` and use `if` or `cond`.
Under `--strict-resources`, `do`, `guard`, `raise`, `call/cc` and
`dynamic-wind` are followed path by path, as described in
[linear handles](linear-handles.md).

## Examples

An alias is valid until its owner moves:

```scheme
(define x (owned (list 1 2)))
(define y (cons x '()))
(display (car (car y)))
(newline)
(define z (move x))
(display (car z))
(newline)
```

Reading the alias after the move is refused on every engine:

<!-- doc-example: skip fragment: refused by the ownership checker; the diagnostic follows -->
```scheme
(define x (owned (list 1 2)))
(define y (cons x '()))
(define z (move x))
(display (car y))
```

```console
error: Ownership analysis failed:
error: Use of 'y', which refers to moved value 'x'
```

So is calling a procedure that reads the owner, and storing the owner where the
checker cannot follow it:

<!-- doc-example: skip fragment: refused by the ownership checker -->
```scheme
(define x (owned (list 1 2)))
(define (peek) (car x))
(define z (move x))
(display (peek))                 ; Use of 'peek', which refers to moved value 'x'

(define cache '())
(define (remember! v) (set! cache v))
(define w (owned (list 3 4)))
(borrow w (remember! w))         ; 'remember!' may keep a reference to owned value 'w'
```

## Scope of the guarantee

- The unit is the program and every file it brings in with `(load "file")`,
  on every engine. A module brought in with `require` is part of the native
  unit; the hosted bytecode compiler checks a required module as its own unit,
  and its procedures are classified by name at the call site.
- The checker does not follow an owned value into a foreign function or through
  `eval`; passing one to either is refused or classified as above.
- Values that are never marked `owned` are not ownership-checked. Their
  memory is managed by regions, which reclaim at region exit and promote values
  that escape (see the [memory model](../runtime/memory-model.md)).
- `(the T e)` is checked at run time by its tag; see the
  [language specification](../../COMPLETE_LANGUAGE_SPECIFICATION.md).
