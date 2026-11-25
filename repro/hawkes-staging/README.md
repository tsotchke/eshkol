# Specialize certified Hawkes operators into checked native kernels

Status: review-ready. Selected MIT contribution by Richard Hoekstra.

Dependencies: PR 1; observable, mix. Prerequisite implementations are outside this patch. `SOURCE.toml` records exact identities.

```sh
python3 repro/hawkes-staging/run_staging_gate.py --eshkol-bin /opt/eshkol/bin/eshkol-run --results /tmp/hawkes-staging.json
```


`staging.esk` specializes all four rows of the admitted reduced mean generator
at the deployment representation w=wn/4. It clears rational denominators and
retains an explicit denominator per kernel. The dynamic integer ABI includes
all retained coordinates, including erased reads. This adapts the model to
Mix's existing integer fragment; it does not add a floating-point Mix dialect.

The actual quoted Mix specializer runs. Its existing independent polynomial
checker verifies every residual; the existing emitter produces ordinary
Eshkol source. The gate independently replays the complete source certificate
and compares five-variable integer polynomials, then replays residuals using
only the checker and runtime, with producer history discarded. Corrupted
residuals refuse. All 128 emitted values match independent arithmetic under
JIT and AOT-O2. These are model RHS kernels, not a staged entire RK4 integrator.

Receipts charge compiler startup, model admission, specialization, independent
checking, AOT compilation and native execution. The qualified sample's full
deployment cost is recorded in the gate receipt; its specialization and residual-check
parts are also recorded separately. These machine-dependent measurements do
not establish amortized speedup or a production deployment crossover.
