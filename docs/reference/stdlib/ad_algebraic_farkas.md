# Exact Farkas infeasibility witnesses

`(require core.ad.algebraic_farkas)` exports:

```scheme
(algebraic-farkas-search owner n A b C limits [support-budget])
(algebraic-farkas-verify owner n A b C limits result)
```

These operations concern the original constraints `A*x=b`, `C*x<=limits` over
one selected-real algebraic field. They can establish that a quadratic program's
feasible set is empty, independently of its objective. The existing
`algebraic-qp-solve` refusal behavior is unchanged: failing its active-set search
is not an infeasibility certificate.

A witness has unrestricted equality multipliers `u` and nonnegative inequality
multipliers `v`, satisfying exactly

```
A^T*u + C^T*v = 0
b^T*u + limits^T*v < 0
```

For any feasible `x`, the left dot product would be zero and at most the strictly
negative right side, a contradiction. The checker evaluates the original
constraints and selected-real signs, not floating residuals or a numerical
status. It also compares the owned source constraint copies using the caller's
exact domain. The owner must be a selected-real field; a Taylor ring has no
ordered-field theorem here and refuses. Dimension is 1..8, with at most eight
rows of each constraint kind. All coefficients go through the existing owner
copy/admission protocol; inexact/foreign values refuse.

Search splits `u=uplus-uminus`, normalizes the separating margin to `-1`, and
enumerates nonnegative supports in the augmented columns
`[A^T,-A^T,C^T; b^T,-b^T,limits^T]`. It attempts support sizes 1..n+1 in a
deterministic order. Exact rectangular elimination solves full-column-rank
supports; singular or nonunit supports are skipped. Every proposed ray passes
the independent source constraint checker before publication. There is no
floating reconstruction or new QP optimization engine.

The support budget defaults to 4096 and must be 1..65536. The result is
`#(algebraic-farkas-v1 infeasible|unknown source u v attempts reason)`.
An infeasible result contains a checkable ray. Unknown contains `#f` ray fields
and reason `budget-exhausted` or `enumeration-complete`. Neither unknown result
certifies feasibility, infeasibility, optimality or unboundedness. The checker
returns false for malformed/edited certificates or unknown status. Exhaustive
combinatorics can still be large; exact coefficient sizes and arithmetic costs
are not bounded by the support counter.

Tests cover inconsistent dependent equalities, opposing inequalities, mixed
constraints, zero-row contradictions, positive scaling, both selected embeddings
of a quadratic field, unknown/exhaustion controls, tampered signs/annihilation/
margin and unchanged QP refusal semantics. No tensor, nilpotent-ring or native
floating decision is involved.
