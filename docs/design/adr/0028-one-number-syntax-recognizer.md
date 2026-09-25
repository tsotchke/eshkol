---
kind: explanation
status: current
owner-area: language
since: v1.3.5
sources:
  - inc/eshkol/core/number_syntax.h
  - inc/eshkol/core/symbol_syntax.h
  - lib/frontend/parser.cpp
  - lib/core/bignum.cpp
  - lib/core/runtime_reader_hosted.cpp
  - lib/backend/vm_parser.c
  - lib/backend/vm_native.c
  - lib/backend/llvm_codegen.cpp
  - .icc/ledger/entries/SW-223.yaml
---
# ADR-0028: One number-syntax recognizer for every reader

**Status:** Accepted
**Ledger:** SW-223
**Scope:** The native source parser, the native runtime reader (`read`) and
`string->number`, and the bytecode VM's parser, reader and `string->number`
(native and WebAssembly builds of the VM).

## Context

Six places turned text into numbers, and each carried its own partial idea of
the R7RS 7.1.1 `<number>` grammar. They disagreed with the standard and with
each other: the native parser split `1+1i` into `1` and `+1i`; the VM parser
read it as a call; `string->number` answered `#f` for `1+1i` while `read`
returned the symbol `1+1i`; `#i42` was the exact integer 42; `#e1.5` was not a
number; `read` of `99999999999999999999` clamped to `INT64_MAX`; and the VM's
`string->number` had three implementations, two of which answered `#f` for
`+nan.0`.

## Decision

`inc/eshkol/core/number_syntax.h` is the grammar, and every reader asks it.
It is header-only C, like `symbol_syntax.h` and `dtoa_shortest.h`, so the VM
unity build (native and WebAssembly), the C++ runtime and the parser each
compile a private copy with no link dependency.

The recognizer never builds a value, because each engine has its own
representation. It returns the form (real, rectangular, polar) and each part
in a canonical spelling that every engine's existing constructors read:

- **INTEGER**: exact decimal digits, converted from radix 2, 8 or 16 here,
  bignum-sized values included.
- **RATIONAL**: exact `n/d`, reduced by the consumer's rational constructor.
- **DECIMAL**: inexact, a decimal spelling whose correctly rounded double is
  the value.
- **INFNAN**: one of `+inf.0`, `-inf.0`, `+nan.0`, `-nan.0`.

Exactness is resolved in the recognizer, so no consumer converts between exact
and inexact itself. `#e` turns a decimal into the integer or rational it spells.
`#i` turns an exact part into the DECIMAL spelling of its correctly rounded
double: a rational is expanded to 800 significant digits with a trailing
sticky digit when the division is inexact, which no double rounding boundary
can separate from the true quotient.

Two representation decisions are made once, here:

1. **An exact zero imaginary part, or an exact zero angle, is the real
   number.** `1+0i` is the exact integer 1, as R7RS defines it, on every
   engine. It is not a complex number whose printing depends on the engine.
2. **Complex parts are inexact.** Eshkol stores a complex number as two
   doubles (numeric-tower.md), so the parts of a non-real result are always
   DECIMAL or INFNAN, and `#e` on a non-real complex number has no value. The
   same applies to `#e` on an infinity or NaN, and to a zero denominator.

A token the recognizer rejects is an identifier. Number syntax with no value
is an error: a compile error for a program literal, a read error for `read`,
and `#f` for `string->number`. The symbol writer consults the same predicate,
so a symbol named `+i` is written `|+i|` and reads back as that symbol.

The parser represents a complex literal as `(make-rectangular <double>
<double>)`, the same desugar pattern the rational literal uses with
`make-rational`. Quote, and a `#(...)` literal's tensor-safety check,
recognise the desugar's exact shape (two DOUBLE literal operands), so a quoted
complex literal is the number, not list data.

## Consequences

- `string->number` accepts the radices 2, 8, 10 and 16, which are the R7RS
  set. The earlier native acceptance of radices up to 36 is gone. The VM's
  `string->number` still takes no radix argument, because a first-class VM
  builtin with an optional argument cannot yet be called with fewer
  arguments; a radix prefix works on both engines.
- An exact `#e` exponent is bounded at 4096 digits, so a short token cannot
  demand an unbounded allocation.
- A new number spelling is added to the grammar in one file, and every reader
  gains it together.
