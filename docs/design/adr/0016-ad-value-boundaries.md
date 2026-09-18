# ADR-0016: One boundary for each place a value crosses a representation

**Status:** Accepted
**Ledger:** SW-182, SW-183
**Scope:** Native LLVM code generation, JIT and AOT. The bytecode VM already
kept whole values at both boundaries and is unchanged.

## Context

Automatic differentiation in Eshkol is carried by values: a forward-mode dual
number, a Taylor jet, a reverse-mode AD node. A carrier is correct only if every
place a value moves through keeps it intact. Two such places each assumed a
closed set of representations and produced 0 for anything outside the set,
with exit status 0.

**Points.** `(gradient f a b ...)` is shorthand for the vector point
`(gradient f #(a b ...))`. The parser rewrote it by assembling its own node, of
a kind nothing else produces, whose lowering only understood numeric literals
and stored 0 for any other element. A point given as variables was
differentiated at the origin (SW-182). ICC `trace-callees` on
`AutodiffCodegen::gradient` shows the operator evaluates its point once, as a
single expression (`codegen_ast_callback_` on `gradient_op.point`), before it
branches to `gradientJetPath` or `tryExactTowerRoute`; the other operators
(`jacobian`, `hessian`, `laplacian`, `divergence`, `curl`,
`directional-derivative`, `derivative`, `derivative-n`) do the same. The defect
was not in any operator's evaluation but in the one rewrite that fed the
gradient a node whose lowering could not evaluate an expression.

**Cells.** A cons slot is a tagged value, but emitted code never moved it as
one. The cell builder branched at run time over storage classes and called a
typed runtime setter; seven readers each branched over their own list of
representations and rebuilt the value through a typed getter. A dual number
was in no list: the setter rejected it and the getter answered 0 (SW-183).
The lists had been extended one type at a time as types were found missing,
and had drifted apart. The cdr reader never received the character arm, and
the two copies inside the compound accessor had 13 and 11 arms. ICC
`duplicate-implementations --path-prefix lib/backend` reported no divergent
cross-file pair at its 12-line threshold, because the copies differ by whole
arms. `trace-callers` on `arena_tagged_cons_set_int64` found the typed setters
reached only through these dispatches.

## Decision

A form defined as sugar is parsed as what it stands for. The separate-scalar
gradient point seeds `parse_vector_body`, so it becomes the ordinary vector
literal node and every argument is evaluated by the element evaluator every
vector literal uses. The bare literal lowering refuses an element it cannot
store at compile time; a lowering never invents a value.

A cons slot is read and written only through `TaggedValueCodegen::loadConsSlot`
and `storeConsSlot`, which move the tagged value whole: type, flags and payload.
The builder and all seven readers delegate to them. The set of value types a
list can hold is therefore open, and a new carrier (a jet, a complex number, a
future value type) needs no case. The typed runtime accessors remain for slots
whose type is known statically, such as the cdr link followed while walking a
proper list.

## Consequences

- A differentiated value survives `map`, `for-each`, `fold-left`, `fold-right`,
  `reduce`, `filter`, `apply`, `cons`, `car`, `cdr` and the compound accessors,
  in forward and reverse mode and at second order.
- The exactness flag of an element survives a list, which the typed setters
  never wrote.
- About 900 lines of emitted dispatch are removed; each slot access is one load
  or one store.
- `tests/ad/gradient_scalar_point_arguments_test.esk` and
  `tests/ad/differentiated_values_in_list_cells_test.esk` pin every case, and
  corpus programs 85 and 86 hold the native engine to the VM.
