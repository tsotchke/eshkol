# Differentiate certified reduced Hawkes forecasts

Status: review-ready. Selected MIT contribution by Richard Hoekstra.

Dependencies: observable. Prerequisite implementations are outside this patch. `SOURCE.toml` records exact identities.

```sh
python3 repro/hawkes-forecast/run_forecast_gate.py --eshkol-bin /opt/eshkol/bin/eshkol-run --compile-linalg-source --results /tmp/hawkes-forecast.json
```


`forecast.esk` admits an owned source through `core.observable_closure` and
independently replays its certificate. The family proof is valid for all
parameter values, not just the numerical center. It projects the 15 ambient
monomials to 4 coordinates for E[N1], or 11 for E[N1], E[N2], E[N1*N2].
The admitted reduced operator runs over `core.ring`'s Q[e]/e^3 jet carrier.

The gate checks one RK4 step with h=1/16 at 15 physical contexts. Python builds
the full 15-coordinate polynomial RK4 map from the original drift and jumps,
then differentiates its coefficients independently. All three Taylor
coefficients must match; coefficient 2 is half the second derivative. Every
mean context has a nonzero second coefficient. Covariance uses the complete
product jet, and its query order is explicitly checked.

These are exact sensitivities of the finite RK4 map, with initial state and
step size fixed. They are not an exact continuous-time solution or an error
bound for RK4. Exactness is checked after arithmetic; unsupported input,
inexact parameters and arithmetic demotion refuse rather than publishing a
supposedly exact answer. Bounds: four state variables, the closure compiler's
parameter/basis limits, 1..8 RK4 steps and exact 0<h<=1/8. The gate covers the
one-step profile only; a larger exact run can refuse an arithmetic limit.
