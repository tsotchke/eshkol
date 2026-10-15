# Exact algebraic reverse-mode differentiation

`(require core.ad.algebraic_reverse)` adapts the existing `core.ad.tape`
custom-op recorder to an explicit `core.algebraic_domain`. The numeric tape's
historical two-slot representation and default arithmetic remain available.
An exact tape owns coefficient copies and uses exact zero/one/add operations;
it never treats a field element's representation vector as an f64 tensor.

`(algebraic-value-and-pullback owner function point)` calls `function` once
with `(tape input-node-vector)` and returns `#(value pullback summary)`.
`(pullback seed)` returns an owned vector of all input cotangents. Repeated
sequential calls reset adjoints, retain the same forward invocation, and never
record additional nodes. Summary is `(input-count node-count op-count 1)`.
The point is a vector of at most 1024 coefficients. Recording is explicitly
sealed after the function returns. Exact tape snapshots own their values;
restoring a sealed exact recorder refuses. Raw tape/node storage is opaque; mutation
through generic low-level recorder observers is outside this contract.

The algebraic operations take the tape explicitly:
`algebraic-reverse-constant`, `algebraic-reverse-add`,
`algebraic-reverse-sub`, `algebraic-reverse-mul`,
`algebraic-reverse-inverse`, and `algebraic-reverse-pow` (integer powers 0–32).
`algebraic-reverse-value` returns an owned coefficient copy.
Inverse requires an invertible enclosing primal. Sibling tapes, foreign fields,
floats and nonunit reciprocals refuse. No transcendental closure is implied.

`(algebraic-hvp owner function point direction)` runs the same function once
over a fresh order-one Taylor coefficient domain, then performs one reverse
sweep. It returns `(H*v gradient value J*v)`. Both directions and enclosing
derivatives are retained exactly. It connects directly to the whole-vector HVP
callback of `algebraic-sparse-hessian`.

The shared recorder now also exposes `(tape-pullback tape output targets seed)`
and `(tape-seal! tape)`. An optional `make-tape` scalar protocol is a vector
`(copy zero-thunk one-thunk add zero? [metadata])`. Exact protocol nodes retain
tape identity, validate contribution arity and copy admitted seeds/results.
Custom backward callbacks must return correct derivatives and avoid mutating
recorded state; these semantics are a caller obligation.

`tests/stdlib/algebraic_reverse_test.esk` checks closed-form irrational and
rational polynomial gradients/HVPs, reciprocal sensitivities, nested derivatives,
sparse recovery, one-forward reuse, failure recovery and ownership refusals.
Existing numeric custom-op tape and explicit finite-difference tests exercise
the default recorder path. This is a stdlib reference surface, with no compiler
transformation, native staged-kernel ABI or general speed guarantee.
