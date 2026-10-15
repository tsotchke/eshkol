# Exact algebraic matrix calculus

`core.algebraic_domain` supplies `(algebraic-domain owner)` and
`(algebraic-domain-ref domain key)`. An owner is a certified field or an explicit
Taylor/hyperdual context from `core.ad.algebraic`. Operation keys are `copy`,
`add`, `sub`, `mul`, `inverse`, `equal?`, `zero?`, `unit?`, `coefficients` and
`element`; metadata keys are `owner`, `kind`, `dimension`, `parent` and `depth`.
`zero` and `one` return owned values. Operations enforce the declared domain;
only its enclosing chain can be embedded. Domain descriptors and owners are
immutable by contract. Coordinate observers and constructors copy storage.

`core.algebraic_linalg` exports:

- `(algebraic-factor owner matrix)` for square row-vector matrices of size 1–32.
- `(algebraic-factor-solve factor rhs)` for an owned solution vector.
- `(algebraic-solve owner matrix rhs)` as the factor-and-solve convenience.
- `(algebraic-matrix-mul owner left right)` for explicit rectangular products.
- `(algebraic-factor-summary factor)` returning `#(size AD-depth bottom-LU-count)`.

At the bottom field, deterministic row-pivoted LU verifies `P*A=L*U` exactly.
For an enclosing AD carrier, coefficient matrices are retained and the primal
matrix is factored recursively. Taylor coefficients solve
`A0*xk = bk - sum_(i>0) Ai*x(k-i)`; the hyperdual cross coefficient retains both
mixed products and the mixed input coefficient. Every level uses the same
bottom field LU, including repeated RHS solves. No pivoting on nilpotents occurs.
Every returned solution satisfies the entire retained equation `A*x=b` exactly.

Inputs, retained matrices and returned values own their coefficient storage;
historical factors can be reused after caller arrays are changed. Singular
bottom primals, nonunit pivots, foreign fields/sibling contexts, inexact inputs,
ragged shapes and unsupported sizes refuse. Existing scalar exactness limits
apply: arithmetic demotion raises through the certified field operations.
There is no sparse factorization, rectangular least-squares solver or compiler
AD transformation in this module.

Run `ctest --test-dir build --output-on-failure -R algebraic_linalg`.
The suite compares irrational inverse columns, a Hilbert solve and Taylor,
hyperdual and three-level derivatives against independently derived formulas.

`algebraic-factor-solve-transpose` reuses the retained LU for `A^T*x=b`,
including nested coefficient recurrences and a full transposed residual.

`algebraic-factor-positive-definite?` checks full-domain symmetry and selected-real
positive bottom-field LU pivots with identity permutation. It reuses the
verified factorization; exact sign refinement can refuse on budget exhaustion.
