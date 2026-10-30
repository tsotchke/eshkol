# Certified small convex quadratic programs

`(require core.ad.algebraic_qp)` solves
`min 1/2*x^T*H*x + g^T*x`, subject to `A*x=b` and `C*x<=d`, over an explicit
exact field or nested coefficient owner.

`(algebraic-qp-solve owner H g A b C d [budget])` admits 1–8 variables, at most
eight equality and eight inequality rows, and a symmetric positive-definite
primal Hessian. Empty constraint collections are empty vectors. A bounded,
deterministic enumeration checks up to 256 inequality active sets. Each
candidate solves the existing exact KKT system and checks full-domain
stationarity/equalities/complementarity and selected-real primal feasibility
and nonnegative multipliers. Singular KKT candidates are not admitted.

`algebraic-qp-solution` returns owned
`#(primal equality-multipliers inequality-multipliers objective)`.
`(algebraic-qp-verify owner H g A b C d result)` independently checks the caller's
problem and the certificate without repeating active-set search. Strict
convexity and the KKT conditions certify the unique primal minimizer. Search
failure/budget exhaustion refuses; it is not an infeasibility certificate.
Dependent equalities and degenerate KKT systems can refuse even when a primal
optimum exists.

`(algebraic-qp-pullback result seed)` reuses the retained transposed KKT solve.
It returns six owned cotangents in parameter order `#(H-bar g-bar A-bar b-bar
C-bar d-bar)`. With KKT adjoint `(p,q,r)`, they are `-p*x^T`, `-p`,
`-(lambda*p^T+q*x^T)`, `q`, `-(mu*p^T+r*x^T)`, and `r`. Inactive inequality
cotangents are zero. Hessian cotangents are unconstrained; a symmetric
parameterization combines both off-diagonal entries.

A pullback requires strictly negative inactive slacks, strictly positive active
multipliers and the admitted nonsingular KKT factor. The solver can return a
certificate at a transition while the pullback refuses ambiguous derivatives.
Nested AD solves follow the admitted moving active branch and preserve enclosing
coefficients. `algebraic-qp-info` returns active-set attempts, copied active
indices, strict-admission status and total Hessian/KKT factorization attempts.

Run `ctest --test-dir build -R algebraic_qp --output-on-failure`. Closed-form
equality, clipping and corner oracles, independent certificate checks, forged
certificates, moving branches and all six cotangent pairings are exercised.
No finite-difference derivative is used. This is a bounded exact reference
solver, not a large sparse/interior-point solver or a global differentiability
claim at changes of active set.
