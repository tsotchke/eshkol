# Mathematics acceptance tests

Each program here states, as an executable check, what a mathematics example
needs from the compiler and cannot get today. The CTest entries are wired
with `WILL_FAIL TRUE` while the corresponding ledger entry is open, so the
suite stays green; when the fix lands the entry starts passing, CTest reports
the inversion as a failure, and the `WILL_FAIL` line is removed in the same
change that closes the ledger entry. That makes each ledger closure a
measured event, which is the ledger's own rule.

As of this suite's merge onto the v1.3.5 candidate, one of the six entries
below has closed and its `WILL_FAIL` inversion is gone (the CMake block names
it individually rather than inverting the whole suite). The other five still
fail and are still inverted. The `status` column records the measured state
of each, taken from a run of every program under `-r` against both this tree
and the candidate's own build.

Two of those five now pass every check they print and still fail the entry:
the exact-rational memory program and the nursery `cond` program both reach
`RESULT: ALL PASS` while emitting the heap-limit diagnostic their entries are
about (139,224 and 15,382 lines of it respectively on this tree, within five
lines of the candidate's own counts). Reading only the verdict line would
call those two closed; the CTest `FAIL_REGULAR_EXPRESSION` is what catches
them, and the entry stays open.

| test | ledger | what must hold | status |
|---|---|---|---|
| `exact_rational_memory_test.esk` | SW-164, SW-165 | the exact harmonic sum H_5000 completes with the right denominator size and prints no heap-limit diagnostic | open: the arithmetic is right and RESULT: ALL PASS is printed, but 139,224 heap-limit diagnostic lines come with it; inverted |
| `nested_ad_exactness_test.esk` | SW-154, SW-159..162 | nested `derivative`/`derivative-n`/`taylor` at exact seeds, and `gradient`/`hessian` at exact vector points, return exact values | open: two exactness checks still fail and a captured-carrier nesting still raises; inverted |
| `exact_tensor_and_sqrt_test.esk` | SW-166, SW-167 | rational tensor entries survive a round trip (or the constructor refuses loudly); `sqrt` and half-integer `expt` of perfect squares are exact | open: 1 passed / 6 failed; inverted |
| `nursery_cond_test_shape_test.esk` | (filed 2026-09-10: nursery defeated by a `cond` clause with an allocating test) | a loop whose tail call sits under a `cond` test that allocates reclaims per iteration, printing no heap-limit diagnostic | open: prints RESULT: ALL PASS with 15,382 heap-limit diagnostic lines; inverted |
| `le17_chains_in_closure_test.esk` | LE-17 | a recursive chain enumerator called from a for-each lambda inside a loop that writes an outer matrix completes instead of crashing the native engine | completes (exit 0) but prints no RESULT line, so the inversion is what keeps it green; still inverted |
| `shadowed_named_let_scope_test.esk` | (filed 2026-09-10: named-let parameter reported undefined under a nested named let when an inner loop rebinds the name) | the program compiles and prints 176; today both JIT and AOT stop with `Undefined variable: acc` | closed: prints 176 and RESULT: ALL PASS, not inverted |
