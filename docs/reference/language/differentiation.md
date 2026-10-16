# Compiler differentiation graph

The native frontend supports an explicit numeric interpretation of a pure scalar
arithmetic computation:

```scheme
(require stdlib)
(differentiate 'numeric
  (lambda (x y) (+ (* x x) (* 3 x y)))
  (vector 2.0 4.0)) ; gradient: #(16.0 6.0)
```

The point is evaluated once. The result is a gradient vector in lambda parameter
order, produced by the existing `gradient` engine. This form does not introduce a
second automatic differentiation engine. Native JIT and AOT use the same lowering;
the separate bytecode frontend explicitly refuses this syntax.

## Admission and representation

The computation must be an inline lambda with 1–64 distinct, fixed scalar
parameters and one expression. Its expression may contain finite numeric
literals, parameter references, `+`, `-`, `*`, `/`, and ordinary `let`/`let*`.
Exact integer, bignum and rational literals retain their spelling/value; floating
literals retain a round-trip representation. Runtime point slots must be finite
real numbers. Numeric differentiation retains the existing engine's numerical
and exactness behavior; this form does not certify floating error.

Parallel `let` initializers read the outer environment. `let*` initializers read
preceding bindings. Variadic arithmetic is normalized in source order, including
unary negation/reciprocal and the zero-operand identities of `+` and `*`.
Identical ordered expressions share graph nodes; operands are not reassociated.
Every admitted binding is validated, including unused initializers.

Free captures must become explicit lambda inputs. Effects, opaque calls, recursion,
control flow, typed lambdas/bindings and nested callable forms are refused at
compile time. Extraction follows macro expansion and binding resolution, and
arithmetic operators and generated helper references must resolve to their
primitive bindings. Shadowing a required helper is a diagnostic, even if that
helper is needed only by the generated wrapper.

The compiler-owned `DifferentiationGraph` has topologically ordered input,
constant and binary arithmetic nodes, ordered edges, owned literal storage and
source identities/locations. It has no LLVM or runtime-value dependency. The
frontend caps the graph at 1,024 nodes, traversal at 8,192 expressions and syntax
depth at 128. These are admission limits, not silent truncation.

`differentiate 'incremental` is reserved for the incremental compiler pass and
currently emits an explicit unavailable-pass diagnostic. The graph's rule table
records the finite arithmetic interpretation alongside the numeric one so both
passes can consume the same admitted program.

## Existing symbolic syntax

`(differentiate expression variable)` remains an alias of `diff` and returns the
existing symbolic derivative representation. Quoted symbolic expressions retain
their existing catchable unsupported-input error. The new mode is selected only
by the explicit quoted `numeric`/`incremental` first operand.

This is a scalar frontend slice of [ADR 0009](../../design/adr/0009-incremental-dataflow-dbsp.md).
General array/tensor lowering, arbitrary closure extraction, staged dense-kernel
ABIs and a bytecode graph frontend remain separate work.

## Verification

Build `eshkol-run`, `stdlib` and `differentiation_graph_test`, then run:

```sh
ctest --test-dir build --output-on-failure \
  -R '^(differentiation_graph_test|compiler_differentiation_(jit|aot|diagnostics))$'
```

The standalone graph test checks topology, sharing, lexical scope, literal
ownership, source coordinates and refusal budgets without LLVM. The native test
compares independent analytic partials and the existing gradient engine, and
checks point evaluation, generated-name isolation and legacy symbolic behavior.
Diagnostics tests require compile failure with no executable for inadmissible
programs and explicitly check bytecode refusal.
