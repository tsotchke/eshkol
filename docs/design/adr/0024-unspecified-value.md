---
kind: explanation
status: current
owner-area: runtime
since: v1.3.5
sources:
  - inc/eshkol/eshkol.h
  - lib/backend/tagged_value_codegen.cpp
  - lib/core/runtime_display_hosted.cpp
  - lib/backend/vm_compiler.c
  - exe/eshkol-repl.cpp
  - tests/core/unspecified_value_test.esk
  - tests/vm_parity/corpus/86_binder_in_operand_and_void.esk
---
# ADR-0024: One unspecified value

**Status:** Accepted
**Ledger:** SW-189, SW-190 (the VM defects found on the way), the REPL echo build item
**Scope:** The native value model (JIT, AOT, REPL JIT), the bytecode VM, the REPL and its machine-mode protocol.

## Context

R7RS leaves the value of many forms unspecified: `set!`, `display`, `newline`,
`for-each`, `vector-set!`, a one-armed `if` whose test is false, `when` and
`unless` when their body does not run. The bytecode VM has always had a value
for this, `VAL_VOID`, which its printer skips. The native value model had none.
Each form returned whatever was convenient: `display` and `newline` returned
the empty list, `when` and `unless` returned `#f`, `set!` and `set-car!`
returned the stored value, `vector-set!` returned the vector,
`hash-table-set!` returned the table, `for-each` returned `#f`.

Two consequences were visible. The REPL wraps every shown form in
`(begin (display expr) (newline))`, and since `display` returned the empty
list, `(when (> x 1) (display "hi"))` echoed `hi()`; the wrapper could not tell
"nothing to show" from `'()`. And the engines disagreed: the VM printed nothing
where the native engines printed `()` or `#f`.

## Decision

There is one unspecified value in the native value model,
`ESHKOL_VALUE_UNSPECIFIED`: an immediate with no payload, distinct from the
empty list. `null?` is `#f` for it and `eq?` against `'()` is `#f`; it is
`eq?` and `equal?` to itself; `type-of` names it `unspecified`; `display` and
`write` print nothing for it, so a list holding one shows a gap. It is the
native counterpart of the VM's `VAL_VOID`: the two engines print the same text
for the same program.

Every form R7RS leaves unspecified evaluates to it, on both engines: a
one-armed `if` with a false test (the parser spells the missing arm as a call
to `void`), `when` and `unless` when the body does not run, `set!`, `display`,
`newline`, `write`, `for-each` and `vector-for-each`, `vector-set!`,
`vector-fill!`, `vector-copy!`, `set-car!`, `set-cdr!`, `hash-table-set!`,
and `(void)`. The VM's `set!`, `vector-set!`, `set-car!` and `set-cdr!` now
push `VAL_VOID` where they pushed nil, and its prelude `for-each` returns
`(if #f #f)` instead of `0`.

The REPL no longer wraps a form in `display`. It evaluates the form through
`executeTagged()` -- the path machine mode already used -- and prints the value
itself, nothing for the unspecified value and the value plus a newline
otherwise (`repl_run_form` in `exe/eshkol-repl.cpp`). Machine mode reports such
a value with an empty `value` and `value_type: "unspecified"`, where it said
`null` before.

## Alternatives considered

**Print nothing for the empty list in the REPL.** Rejected: `'()` is a value a
reader asks to see.

**Keep `#f` / the stored value as the result.** Rejected: the engines then
disagree, and a program that tests `(null? (display x))` or prints the result
of `for-each` gets a different answer on the VM.

## Consequences

- A program that relied on `set!` returning the stored value, or on
  `vector-set!` returning the vector, now gets the unspecified value. R7RS never
  promised either; the reference documented `set!` as returning an unspecified
  value.
- `(display (list (when #f 1)))` prints `()` on every engine: the gap in a
  one-element list is invisible. That is the VM's established rendering.
- The machine-mode `value_type` for a no-value result changes from `null` to
  `unspecified`; `null` now means the empty list only.

## Verification

`tests/core/unspecified_value_test.esk` on the cached run path, the in-process
JIT and AOT; `tests/vm_parity/corpus/86_binder_in_operand_and_void.esk` on
native against the VM; the piped REPL and machine-mode tests.
