# SICP-Completeness Report (ESH-0005)

Release gate for **v1.3-evolve**: support **100% of the entire SICP book**, and
prove it with executable Eshkol programs. The corpus lives in `tests/sicp/`;
each implemented program self-checks (`PASS:`/`FAIL:` lines, ending in
`ALL PASS` or a fail summary). The harness `scripts/run_sicp_smoke.sh` runs
implemented programs under **both** the JIT (`-r`) and **AOT**
(`eshkol-run f.esk -o bin && bin`), then emits hard failing ICC events for
required full-book systems that do not yet have runnable probes.

Platform: macOS arm64, LLVM 21. Runs guarded with `perl -e 'alarm N; exec @ARGV'`
(macOS has no `timeout`). Verified on master after the recent correctness fixes
(merge through PR #86).

## Summary

The manifest-required full-book probes are now present on this branch. The
release gate is no longer allowed to pass a representative subset: every row in
the full-book manifest below has a runnable `tests/sicp/*.esk` file, and the
newly added probes have focused JIT and AOT verification. The full
`scripts/run_sicp_smoke.sh` corpus remains the final gate/CI check.

| Chapter | Programs | -r | AOT | Notes |
|---------|----------|----|----|-------|
| ch1 (building abstractions w/ procedures) | 3 | 3/3 | 3/3 | full |
| ch2 (building abstractions w/ data)       | 11 | focused PASS | focused PASS | includes painters, tower/coercion, and polynomials |
| ch3 (modularity, objects, state)          | 13 | focused PASS | focused PASS | includes mutation/cycles, circuits, constraints, concurrency, and richer stream systems |
| ch4 (metalinguistic abstraction)          | 10 | focused PASS | focused PASS | includes analyzer, derived forms, lazy, ambeval/parser, and query |
| ch5 (computing with register machines)    | 6 | focused PASS | focused PASS | includes stack/recursive machines, storage/GC, ECE, and compiler |
| repro (codegen-gap probe)                 | 1 | PASS | PASS | first-class predicate regression probe |

**Manifest-required corpus probes: present and focused-verified under both -r
and AOT.**
**Full-book SICP gate: must pass the full smoke harness before merge/release.**

## Required Full-Book Probes

These are not optional, not XFAIL, and not "out of scope" for the release gate.
`scripts/run_sicp_smoke.sh` emits failing ICC events for any missing or failing
probe. On this branch, all required executable files exist.

| Required probe | SICP section | Required executable file |
|----------------|--------------|--------------------------|
| Picture-language painters/combinators/square-limit | 2.2.4 | `tests/sicp/ch2_picture_painters.esk` |
| Generic arithmetic tower and coercion | 2.5 | `tests/sicp/ch2_generic_tower_coercion.esk` |
| Polynomial arithmetic / symbolic algebra | 2.5.3 | `tests/sicp/ch2_polynomial_arithmetic.esk` |
| Mutable pairs, sharing, append!, cycles, count-pairs examples | 3.3.1 | `tests/sicp/ch3_mutable_pairs_cycles.esk` |
| Digital-circuit simulator | 3.3.4 | `tests/sicp/ch3_digital_circuit.esk` |
| Constraint-propagation system | 3.3.5 | `tests/sicp/ch3_constraints.esk` |
| Concurrency, serializers, and account exchange | 3.4 | `tests/sicp/ch3_concurrency.esk` |
| Accelerated streams, Euler transform, tableaux | 3.5 | `tests/sicp/ch3_stream_acceleration.esk` |
| Power-series streams | 3.5 | `tests/sicp/ch3_stream_power_series.esk` |
| Stream integrators, signal systems, zero crossings | 3.5 | `tests/sicp/ch3_stream_signal_systems.esk` |
| Random streams / Monte Carlo streams | 3.5 | `tests/sicp/ch3_stream_random_monte_carlo.esk` |
| Analyzing evaluator | 4.1.7 | `tests/sicp/ch4_analyzing_evaluator.esk` |
| Metacircular evaluator derived forms | 4.1 | `tests/sicp/ch4_metacircular_derived_forms.esk` |
| Lazy evaluator / normal-order evaluator | 4.2 | `tests/sicp/ch4_lazy_evaluator.esk` |
| `ambeval` nondeterministic evaluator and driver | 4.3 | `tests/sicp/ch4_amb_evaluator.esk` |
| Nondeterministic natural-language parser | 4.3 | `tests/sicp/ch4_amb_parser.esk` |
| Logic query evaluator | 4.4 | `tests/sicp/ch4_query_system.esk` |
| Register-machine stack operations/statistics | 5.1-5.2 | `tests/sicp/ch5_register_machine_stack.esk` |
| Recursive register-machine programs | 5.1-5.2 | `tests/sicp/ch5_register_machine_recursive.esk` |
| Storage allocation and garbage-collector model | 5.3 | `tests/sicp/ch5_storage_gc.esk` |
| Explicit-control evaluator | 5.4 | `tests/sicp/ch5_explicit_control_evaluator.esk` |
| SICP compiler targeting the register-machine simulator | 5.5 | `tests/sicp/ch5_compiler.esk` |

## Full-Book Coverage Manifest

This manifest is the release checklist. A row marked PASS must be backed by a
runnable `tests/sicp/*.esk` probe under both JIT and AOT. No book-required row
is currently marked blocked on this branch.

| Book section / example family | Evidence | Status |
|-------------------------------|----------|--------|
| 1.1 Elements of programming | `ch1_higher_order.esk`, `ch1_newton.esk` | PASS |
| 1.2 Procedures and processes | `ch1_primes.esk`, `ch1_newton.esk` | PASS |
| 1.3 Higher-order procedures | `ch1_higher_order.esk`, `ch1_newton.esk` | PASS |
| 2.1 Data abstraction | `ch2_rational.esk`, `ch2_interval.esk` | PASS |
| 2.2 Hierarchical data and closure properties | `ch2_sequences.esk`, `ch2_picture.esk` for list/tree/vector/frame mechanics | PASS |
| 2.2.4 Picture-language painters/combinators/square-limit | `tests/sicp/ch2_picture_painters.esk` | PASS |
| 2.3 Symbolic data | `ch2_symbolic_deriv.esk`, `ch2_sets.esk`, `ch2_huffman.esk` | PASS |
| 2.4 Multiple representations for abstract data | `ch2_generic.esk` | PASS |
| 2.5 Systems with generic operations, same-type dispatch | `ch2_generic.esk` | PASS |
| 2.5 Generic tower/coercion | `tests/sicp/ch2_generic_tower_coercion.esk` | PASS |
| 2.5.3 Polynomial arithmetic / symbolic algebra | `tests/sicp/ch2_polynomial_arithmetic.esk` | PASS |
| 3.1 Assignment and local state | `ch3_accounts.esk`, `ch3_monte_carlo.esk` | PASS |
| 3.2 Environment model of evaluation | `ch4_metacircular.esk`, `ch4_metacircular_full.esk` exercise environment frames through evaluator behavior | PASS |
| 3.3.1 Mutable list structure used by queues/tables | `ch3_queue.esk`, `ch3_tables.esk` | PASS |
| 3.3.1 Mutable pair sharing, append!, cycles, count-pairs examples | `tests/sicp/ch3_mutable_pairs_cycles.esk` | PASS |
| 3.3.2 Queues | `ch3_queue.esk` | PASS |
| 3.3.3 Tables | `ch3_tables.esk` | PASS |
| 3.3.4 Digital-circuit simulator | `tests/sicp/ch3_digital_circuit.esk` | PASS |
| 3.3.5 Constraint propagation | `tests/sicp/ch3_constraints.esk` | PASS |
| 3.4 Concurrency, serializers, and account exchange | `tests/sicp/ch3_concurrency.esk` | PASS |
| 3.5 Basic streams, infinite streams, sieve, partial sums | `ch3_streams.esk` | PASS |
| 3.5 Accelerated streams / tableaux | `tests/sicp/ch3_stream_acceleration.esk` | PASS |
| 3.5 Power-series streams | `tests/sicp/ch3_stream_power_series.esk` | PASS |
| 3.5 Stream integrators, signal systems, zero crossings | `tests/sicp/ch3_stream_signal_systems.esk` | PASS |
| 3.5 Random streams / Monte Carlo streams | `tests/sicp/ch3_stream_random_monte_carlo.esk` | PASS |
| 4.1 Metacircular evaluator core | `ch4_metacircular.esk`, `ch4_metacircular_full.esk` | PASS |
| 4.1 Derived forms in the metacircular evaluator | `tests/sicp/ch4_metacircular_derived_forms.esk` | PASS |
| 4.1.7 Analyzing evaluator | `tests/sicp/ch4_analyzing_evaluator.esk` | PASS |
| 4.2 Lazy evaluator / normal order | `tests/sicp/ch4_lazy_evaluator.esk` | PASS |
| 4.3 Nondeterministic primitives / direct CPS amb capability | `ch4_amb.esk`, `ch4_amb_deep_cps_test.esk` | PASS |
| 4.3 `ambeval` evaluator and driver loop | `tests/sicp/ch4_amb_evaluator.esk` | PASS |
| 4.3 Nondeterministic natural-language parser | `tests/sicp/ch4_amb_parser.esk` | PASS |
| 4.4 Logic query system | `tests/sicp/ch4_query_system.esk` | PASS |
| 5.1 Designing register machines, basic controller execution | `ch5_register_machine.esk` | PASS |
| 5.1-5.2 Stack operations/statistics | `tests/sicp/ch5_register_machine_stack.esk` | PASS |
| 5.1-5.2 Recursive register-machine programs | `tests/sicp/ch5_register_machine_recursive.esk` | PASS |
| 5.2 Register-machine simulator basic assign/test/branch/goto | `ch5_register_machine.esk` | PASS |
| 5.3 Storage allocation and garbage collection | `tests/sicp/ch5_storage_gc.esk` | PASS |
| 5.4 Explicit-control evaluator | `tests/sicp/ch5_explicit_control_evaluator.esk` | PASS |
| 5.5 Compiler | `tests/sicp/ch5_compiler.esk` | PASS |
| Regression: first-class/apply predicate semantics needed by SICP evaluators | `repro_esh0078_firstclass_predicate.esk` | PASS |
| Beyond-book stress: deeper nondeterministic continuation/backtracking path | `ch4_amb_deep_cps_test.esk` | PASS |

## Beyond-Book Bar

Passing the book is the floor. The Eshkol gate also requires:

- **JIT and AOT parity** for every SICP system.
- **Stress coverage beyond the book examples**, including the deep `amb`
  continuation/backtracking probe.
- **No stale XFAILs**: once a blocker is fixed, the probe becomes part of the
  normal passing corpus.
- **Sourceful diagnostics and bounded failure modes** for any compiler/runtime
  limit discovered while implementing the book systems.

## Per-program detail

### Chapter 1 — procedures
| Program | -r | AOT | checks | Notes |
|---------|----|----|--------|-------|
| ch1_higher_order.esk | PASS | PASS | 17 | sum/product abstractions, lambda, let, iterative+recursive |
| ch1_primes.esk | PASS | PASS | 14 | Fermat test, smallest-divisor, primality |
| ch1_newton.esk | PASS | PASS | 7 | fixed-point, average-damp, Newton's method, sqrt/cube-root, continued fractions, repeated |

### Chapter 2 — data
| Program | -r | AOT | checks | Notes |
|---------|----|----|--------|-------|
| ch2_rational.esk | PASS | PASS | 10 | rational arithmetic, gcd normalization |
| ch2_interval.esk | PASS | PASS | 11 | interval arithmetic (2.1.4) |
| ch2_sequences.esk | PASS | PASS | 11 | map/filter/accumulate, sequences as conventional interfaces |
| ch2_sets.esk | PASS | PASS | n | sets as lists / ordered / binary trees (2.3.3) |
| ch2_huffman.esk | PASS | PASS | 6 | Huffman encoding/decoding (2.3.4) |
| ch2_picture.esk | PASS | PASS | 8 | picture-language vectors/frames/transforms (2.2.4) |
| ch2_symbolic_deriv.esk | PASS | PASS | 8 | symbolic differentiation (2.3.2) |
| ch2_generic.esk | PASS | PASS | 12 | **tagged data, data-directed dispatch (op/type table), generic arithmetic, message passing (2.4-2.5)** |

**Chapter 2 full-book systems are now represented in the corpus.** The gate
includes full picture-language painters and combinators (2.2.4), generic
arithmetic tower/coercion (2.5), and polynomial arithmetic/symbolic algebra
(2.5.3), in addition to the original data-abstraction probes.

### Chapter 3 — state, streams
| Program | -r | AOT | checks | Notes |
|---------|----|----|--------|-------|
| ch3_accounts.esk | PASS | PASS | 13 | mutable state, make-account, message-passing objects (3.1) |
| ch3_monte_carlo.esk | PASS | PASS | 3 | Monte Carlo / Cesaro pi estimate (3.1.2) |
| ch3_streams.esk | PASS | PASS | 10 | cons-stream/delay/force, infinite integers/fibs, **sieve of Eratosthenes**, partial sums, **pi (Leibniz) signal stream** (3.5) |
| ch3_tables.esk | PASS | PASS | 11 | 1-D and 2-D mutable tables (3.3.3) |
| ch3_queue.esk | PASS | PASS | 9 | mutable queue with front/rear pointers, set-car!/set-cdr! (3.3.2) |

**Chapter 3 full-book systems are now represented in the corpus.** The gate
includes mutable pair sharing/cycle examples (3.3.1), the digital-circuit
simulator (3.3.4), the constraint-propagation system (3.3.5),
concurrency/serializers/account exchange (3.4), and richer stream systems from
3.5: acceleration/tableaux, power series, signal systems, and random Monte
Carlo streams.

### Chapter 4 — metalinguistic abstraction
| Program | -r | AOT | checks | Notes |
|---------|----|----|--------|-------|
| ch4_metacircular_full.esk | PASS | PASS | 8 | **full metacircular eval/apply running non-trivial programs**: recursive factorial, 2^n accumulator, tree-recursive Fibonacci, list length/sum, define+set! state. |
| ch4_amb.esk | PASS | PASS | 4 | **nondeterministic `amb`** via CPS success/failure continuations: Pythagorean triples + a logic puzzle + require-filtering with real backtracking (4.3). |
| ch4_amb_deep_cps_test.esk | PASS | PASS | 2 | Beyond-book stress: raises the Pythagorean search bound to exercise deeper success/failure continuation chains. |
| ch4_metacircular.esk | PASS | PASS | 6/6 | The "textbook" metacircular evaluator that binds raw builtins (`=`,`<`) as first-class env values. `recursive-fact` now returns 120 after ESH-0079 / PR #86. |

**Chapter 4 full-book systems are now represented in the corpus.** The gate
includes the analyzing evaluator (4.1.7), derived forms in the metacircular
evaluator (4.1), the lazy evaluator (4.2), the SICP-faithful `ambeval`
evaluator/driver and nondeterministic parser (4.3), and the query system (4.4).
The query probe implements pattern matching, unification, rules, frames,
stream-of-frames evaluation, and canonical company-database queries.

### Chapter 5 — register machines
| Program | -r | AOT | checks | Notes |
|---------|----|----|--------|-------|
| ch5_register_machine.esk | PASS | PASS | 5 | **register-machine simulator**: registers (mutable a-list), op table, label table, flat instruction vector + pc, the assign/test/branch/goto instruction set. Runs the iterative-factorial machine and Euclid's GCD machine (5.1-5.2). |

**Chapter 5 full-book systems are now represented in the corpus.** The
register-machine simulator core (5.2) is exercised for iterative factorial and
GCD, and the gate also includes stack operations/statistics, recursive
register-machine programs, storage allocation/GC (5.3), the explicit-control
evaluator (5.4), and the compiler (5.5).

## Root-cause: ch4 metacircular recursion gap — **fixed by ESH-0079 / PR #86**

**Symptom.** In the textbook metacircular evaluator (`ch4_metacircular.esk`),
`(meval '(fact 5) e)` returns **1** instead of 120. For
`g = (lambda (n) (if (= n 0) 1 (* 2 (g (- n 1)))))`, `(g 3)` returns **1** — the
wrapping `(* 2 _)` / `(* n _)` accumulation is lost at **every** level; the
result is always the base case. Tail-recursive `f` "passes" only by accident
(its base value equals the expected value, so reaching the base immediately
looks correct).

**Bisection (native Eshkol repros).**
- Reentrant `map` (mapping function recurses, itself calling `map`): **WORKS**
  (`/tmp/r1.esk` → 120). Replacing `map` with hand-written `eval-list` recursion
  reproduces the failure identically → **`map` is NOT the cause.**
- `apply` of a primitive (`*`,`+`) on a reentrant-mapped list: **WORKS** (120).
- Instrumenting `mapply` shows that for `(g 2)` only **one** primitive call ever
  happens — `(= 2 0)` — then the evaluator returns the base case. So
  `(if (not (eq? (= 2 0) #f)) base ...)` took the **then** branch, i.e.
  `(= 2 0)` evaluated to a value that is **not** `eq?` to `#f`.

**Root cause.** A builtin **comparison / equality / type predicate**, when
invoked as a **first-class value** (extracted from a list/env and called
indirectly, OR applied via `apply`), returned the **raw integer `0`/`1`** (and in
some paths `'()`) instead of a boolean `#f`/`#t`. **Direct named calls are
fine.** Minimal repro (`tests/sicp/repro_esh0078_firstclass_predicate.esk`):

```
(define cmp (car (list =)))
(= 2 0)            ; => #f      (direct call: correct)
(cmp 2 0)          ; used to return 0; now => #f
(apply = (list 2 0)) ; used to return 0/(); now => #f
(eq? (cmp 2 0) #f) ; now => #t
```

Because R7RS treats **only `#f`** as false, the interpreter's `(if (not (eq?
<pred-result> #f)) ...)` always sees `0` as truthy, takes the base/then branch,
and never builds the pending `(* n _)` — so all non-tail recursion collapses to
the base case. Arithmetic primitives (`+`,`*`,`-`) round-trip correctly through
both indirect call and `apply`; the defect is specific to results that should be
**booleans** (and likely any builtin whose result is an i1 that the direct-call
codegen packs as a boolean tagged value but the generic/indirect call trampoline
leaves unpacked). `equal?`/`null?` additionally report "apply: Unknown function"
via the `apply` path.

**Fixed by ESH-0079 / PR #86.** First-class comparison/equality predicate calls
and `apply` now pack predicate results as booleans. `ch4_metacircular_full` and
`ch5_register_machine` still use boolean-normalizing wrapper lambdas, but the raw
textbook evaluator and minimal repro are now passing regression probes.

## Secondary gap: deep CPS continuation depth (amb)

`ch4_amb.esk`'s backtracking sweep crashes with **SIGILL** when the search is
deep: `(amb-all (pyth-triples n))` works for n<=14 and crashes for n>=~16
(thousands of nested success/fail closures). The corpus keeps n=13 (still finds
three triples). Bucket: **deep-CPS-continuation-depth** — a runtime depth/stack
limit on long non-tail continuation chains (candidate follow-up ticket; distinct
from ESH-0078). The metacircular evaluator's own recursion depth (fact 8, fib 10)
is well within limits.

## Honest coverage assessment

- **ch1: complete against the current manifest.** Procedures, processes, and
  higher-order procedures pass on -r and AOT.
- **ch2: manifest-complete.** Data abstraction, symbolic data, generic
  operations, message passing, picture-language painters, tower/coercion, and
  polynomial arithmetic all have runnable probes.
- **ch3: manifest-complete.** Assignment/mutation, message-passing objects,
  Monte Carlo state, tables/queues, mutable sharing/cycles, circuits,
  constraints, concurrency/serializers, and the richer stream systems all have
  runnable probes.
- **ch4: manifest-complete.** The metacircular evaluator, derived forms,
  analyzing evaluator, lazy evaluator, `amb`/ambeval/parser coverage, query
  evaluator, and deep-CPS stress all have runnable probes.
- **ch5: manifest-complete.** The register-machine simulator, stack and
  recursive-machine coverage, storage allocation/GC, explicit-control
  evaluator, and compiler all have runnable probes.

## How to run

```
cmake --build build --target eshkol-run -j
./scripts/run_sicp_smoke.sh            # -r + AOT, writes scripts/icc_traces/sicp_smoke.jsonl
./scripts/run_sicp_smoke.sh --no-aot   # JIT only (faster)
```
