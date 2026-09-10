# Mathematics acceptance tests

Each program here states, as an executable check, what a mathematics example
needs from the compiler and cannot get today. The CTest entries are wired
with `WILL_FAIL TRUE` while the corresponding ledger entry is open, so the
suite stays green; when the fix lands the entry starts passing, CTest reports
the inversion as a failure, and the `WILL_FAIL` line is removed in the same
change that closes the ledger entry. That makes each ledger closure a
measured event, which is the ledger's own rule.

| test | ledger | what must hold |
|---|---|---|
| `exact_rational_memory_test.esk` | SW-164, SW-165 | the exact harmonic sum H_5000 completes with the right denominator size and prints no heap-limit diagnostic |
| `nested_ad_exactness_test.esk` | SW-154, SW-159..162 | nested `derivative`/`derivative-n`/`taylor` at exact seeds, and `gradient`/`hessian` at exact vector points, return exact values |
| `exact_tensor_and_sqrt_test.esk` | SW-166, SW-167 | rational tensor entries survive a round trip (or the constructor refuses loudly); `sqrt` and half-integer `expt` of perfect squares are exact |
