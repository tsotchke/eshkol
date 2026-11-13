# Differentiate source-bound Bernoulli SBLC work laws

Status: review-ready. Selected MIT contribution by Richard Hoekstra.

Dependencies: sblc, resource. Prerequisite implementations are outside this patch. `SOURCE.toml` records exact identities.

```sh
python3 repro/sblc-cost-sensitivities/run_cost_gate.py --eshkol-bin /opt/eshkol/bin/eshkol-run --results /tmp/sblc-cost-sensitivities.json
```


`cost.esk` gives a fixed, source-admitted recursive SBLC program an explicit
Bernoulli(theta) coin law: left has probability theta, right 1-theta. It does
not differentiate the discrete leaf-count parameter or change core SBLC's
fair-coin semantics. At theta=1/2 the admitted outputs agree with the prior
fair-coin resource law.

The actual ordered four leaves determine the quadratic terminating-leaf
probability p(theta). A recursive invocation costs three steps at a leaf and
four plus two child invocations at a branch, so its defective work PGF obeys
H=p*z^3+(1-p)*z^4*H^2. Existing jets compute termination probability, restricted
expected work, unconditional work when finite, conditional work and the first
25 PGF coefficients. An exact quadratic range check over the entire requested
parameter interval refuses any interval touching p=1/2 before division. Zero
conditioning mass, unadmitted syntax and arithmetic loss also refuse.

Eleven parameter/source contexts match independently expanded rational
functions and Catalan coefficients through degree 24. A separate operational
control uses the existing `sblc-step` reducer with weighted branches; it does
not introduce another substitution engine. Tests distinguish source layout
from leaf count, critical-point and interval-crossing refusals, and infinite
unconditional work from a finite conditional expectation.


The gate imports its source-datum reader from the SBLC-resource prerequisite. Its rational-function/Catalan oracle has no Hawkes or observable-closure dependency.
