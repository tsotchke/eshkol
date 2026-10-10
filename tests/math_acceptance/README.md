# Mathematics acceptance tests

Each program here states, as an executable check, what a mathematics example
needs from the compiler. The CTest entries are registered in `CMakeLists.txt`
(the `ESHKOL_MATH_ACCEPTANCE` list) as `math_acceptance_<name>`, each running
`eshkol-run -r <program>` with a 120-second timeout. An entry passes when the
program prints `RESULT: ALL PASS` and none of `FAIL:`,
`RESULT: FAILURES DETECTED`, `Heap limit exceeded` or `fatal signal`.

An entry whose capability is still being built is wired with
`WILL_FAIL TRUE`, so the suite stays green; when the capability lands the
entry starts passing, CTest reports the inversion as a failure, and the
`WILL_FAIL` line is removed in the same change that closes the entry. That
makes each closure a measured event. The inversion is per entry, not per
suite: the CMake block names each closed entry individually.

Three of the six entries are closed and run without inversion. The other three
are still inverted. The `status` column records the measured state of each at
v1.3.6-evolve, taken from a run of every program under `-r`:

```sh
cmake --build build --target eshkol-run stdlib
ctest --test-dir build -R math_acceptance
```

The two memory entries are the reason the CTest `FAIL_REGULAR_EXPRESSION`
exists: before they closed, both programs reached `RESULT: ALL PASS` while
emitting the heap-limit diagnostic their entries are about. Reading only the
verdict line would have called them closed; the regular expression kept them
open until the diagnostic was gone.

| test | topic | what must hold | status |
|---|---|---|---|
| `exact_rational_memory_test.esk` | exact-rational temporaries are reclaimed like integers; the heap limit is a fail-closed contract | the exact harmonic sum H_5000 completes with the right denominator size (2165 digits) and prints no heap-limit diagnostic | closed: prints `RESULT: ALL PASS` with no heap-limit diagnostic; not inverted |
| `nested_ad_exactness_test.esk` | exact seeds through nested AD | nested `derivative`/`derivative-n`/`taylor` at exact seeds, and `gradient`/`hessian` at exact vector points, return exact values | open: the four nested scalar checks pass; `gradient` and `hessian` at an exact vector point return inexact values (4 passed / 2 failed); inverted |
| `exact_tensor_and_sqrt_test.esk` | exact tensor entries; exact roots | rational tensor entries survive a round trip (or the constructor refuses loudly); `sqrt` and half-integer `expt` of perfect squares are exact | open: every `sqrt`/`expt` exactness check passes; the rational tensor round trip does not yet hold (6 passed / 1 failed); inverted |
| `nursery_cond_test_shape_test.esk` | nursery reclamation under an allocating `cond` test | a loop whose tail call sits under a `cond` test that allocates reclaims per iteration, printing no heap-limit diagnostic | closed: prints `RESULT: ALL PASS` with no heap-limit diagnostic; not inverted |
| `le17_chains_in_closure_test.esk` | LE-17: recursive enumeration inside a mutating loop | a recursive chain enumerator called from a `for-each` lambda inside a loop that writes an outer matrix completes on the native engine | completes (exit 0, ends with `(done 94 30)`) but prints no `RESULT` line yet, so the inversion is what keeps it green; still inverted |
| `shadowed_named_let_scope_test.esk` | named-let parameter scope under a nested named let that rebinds the name | the program compiles and prints 176 under both JIT and AOT | closed: prints 176 and `RESULT: ALL PASS`; not inverted |
