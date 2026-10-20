# `core.ad.root_branch`

`(require core.ad.root_branch)` follows a selected simple real polynomial root
through certified whole parameter steps. `family[i]` is the existing exact-poly
coefficient vector in t for `F(x,t)=sum_i family[i](t)*x^i`.

```scheme
(define family (vector (vector -1 0 -1) (vector 0) (vector 1)))
(define start (make-root-branch family 0 (cons 3/4 5/4)))
(define step (root-branch-step start 1/4 (cons 3/4 5/4)))
(root-branch-enclosure (vector-ref step 0))
(root-branch-sensitivity (vector-ref step 0))
(verify-root-branch-step start (vector-ref step 0) (vector-ref step 1))
```

The example follows the positive eigenvalue of symmetric `((t,1),(1,-t))`.
Data, times and intervals are exact rationals, with x degree 1–8 and parameter
degree at most eight. Construction copies the family and admits one simple
root using the existing exact root counter. Construction/steps optionally
accept 0–64 bisections, default 12. Operations preserve their input snapshots; enclosure observers own their pairs.
Treat snapshot vectors and their nested family data as read-only; the language
does not prevent callers from mutating the representation.

A positive-width tube X must contain the old enclosure. On the whole parameter
interval T, rigorous interval arithmetic on exact rational endpoints bounds the
original polynomial and x derivative. For center m and exact midpoint inverse R,
strict `q=sup abs(1-R*F_x(X,T))<1` plus
`sup abs(R*F(m,T))+q*radius<=radius` proves a unique root in X for every t in T.
Polynomial smoothness and uniformly nonzero derivative connect it continuously
to the old selected root. Endpoint isolation alone is insufficient. Exact
polynomial signs/root counting then refine the endpoint enclosure.

Step returns `#(successor certificate)`, where certificate is
`#(root-step-v1 time-interval tube R jacobian-bound residual-bound q forcing margin)`.
The verifier checks both endpoint admissions, family equality and connection,
and recomputes that whole-step receipt. Sensitivity encloses `-F_t/F_x` over the
snapshot box at its exact time. Arithmetic never trusts a sampled residual.

Failed admission is unresolved, not proof of collision or branch absence. Work
budgets do not promise arbitrary accuracy. This slice does not automatically
choose/subdivide tubes, certify all/complex roots, handle singular branches or
identify eigenvalues across collisions. Tests include algebraic endpoint signs,
rational branches, reverse traversal, tampering and simple endpoints separated
by an interior double root.
