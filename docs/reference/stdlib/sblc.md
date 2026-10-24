---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/sblc.esk
  - tests/stdlib/sblc_test.esk
  - scripts/run_sblc_gate.sh
  - docs/design/adr/0017-stochastic-binary-lambda-calculus.md
---

# `core.sblc` — Stochastic Binary Lambda Calculus

**Source**: [`lib/core/sblc.esk`](../../../lib/core/sblc.esk)
**Require**: `(require core.sblc)`. It must be required individually; `(require stdlib)` does not load it. Terms are built with `core.blc`'s constructors, so programs that build terms also `(require core.blc)`.
**Design record**: [ADR-0017](../../design/adr/0017-stochastic-binary-lambda-calculus.md) (this module is its first slice).

Stochastic Binary Lambda Calculus (SBLC) is the untyped lambda calculus with a
fair binary choice `M ⊕ N`, written in a self-delimiting binary prefix code.
SBLC is Eshkol's synthesis of two established calculi: stochastic lambda
calculus (Dal Lago and Zorzi's probabilistic operational semantics) and
Tromp's binary lambda calculus, which ships as [`core.blc`](blc.md). The module
keeps three layers apart:

1. **Language**: the `sblc-v1` code, closedness, choice-aware substitution and
   strong leftmost-outermost reduction.
2. **Probability semantics**: a sampled semantics, where every coin comes from
   an explicitly seeded, isolated PRNG and is recorded in a trace, and an
   exact distribution semantics, where the result under a step bound is an
   exact rational sub-distribution over normal forms plus an exact timeout
   mass, and the two always sum to 1.
3. **Prior over programs**: the weight `2^-length` of the prefix code, exact
   per-length counts and enumeration, and exact conditioning on input/output
   examples.

No floating-point value takes part in the exact path: every mass, weight,
likelihood and posterior is an exact Eshkol rational (bignum numerator and
denominator as needed).

All samples on this page were run under both the JIT (`eshkol-run -r`) and AOT
(`eshkol-run -o`) engines; their outputs are byte-identical.

## Terms and the `sblc-v1` code

Terms reuse `core.blc`'s homoiconic forms and add one:

| Form | Meaning |
|---|---|
| `(var i)` | variable, `i` a 1-based De Bruijn index |
| `(lam B)` | abstraction |
| `(app M N)` | application |
| `(choice M N)` | fair binary choice `M ⊕ N` |

The code is a separate, versioned format. Classic BLC encodings come only from
`core.blc` and are never re-encoded here, so no decoder has to guess which
grammar it is reading.

```
sblc(var i)      = 0 1^(i-1) 0
sblc(lam M)      = 10  sblc(M)
sblc(app M N)    = 110 sblc(M) sblc(N)
sblc(choice M N) = 111 sblc(M) sblc(N)
```

Every bit string means something, so the code is complete: reading fair bits
until a term is finished yields a term with probability exactly 1. The root
constructor is a variable with probability 1/2, an abstraction with 1/4, an
application with 1/8 and a choice with 1/8. The generating function
`S(z) = z^2/(1-z) + z^2 S(z) + 2 z^3 S(z)^2` gives, at `z = 1/2`,
`S = 1/2 + S/4 + S^2/4`, whose roots are 1 and 2; the prior is the least
root, 1.

### Codec and identity

| Function | Contract |
|---|---|
| `(sblc-encode t)` | the `sblc-v1` bit string of `t` (a string of `#\0`/`#\1`) |
| `(sblc-decode bits)` | the term, consuming every bit; signals an error on trailing bits, a truncated term or a non-bit character |
| `(sblc-try-decode bits)` | the term when `bits` is exactly one term, otherwise `#f`; never signals |
| `(sblc-prefix-status bits)` | `'complete`, `'trailing` (a term followed by extra bits), `'incomplete` (a proper prefix of some term) or `'invalid` |
| `(sblc-program-id t)` | versioned identity `"sblc-v1:<bits>"` |
| `(sblc-length t)` | code length in bits, computed structurally |
| `(sblc-weight t)` | exact prior weight `1/2^length` |
| `sblc-format-version` | the symbol `sblc-v1` |

```scheme
(require core.blc)
(require core.sblc)
(define I (blc-lam (blc-var 1)))
(define K (blc-lam (blc-lam (blc-var 2))))
(display (sblc-encode I)) (newline)
(display (sblc-encode (sblc-choice I K))) (newline)
(display (sblc-decode "1000")) (newline)
(display (sblc-try-decode "10000")) (newline)
(display (list (sblc-prefix-status "1000") (sblc-prefix-status "10000")
               (sblc-prefix-status "110") (sblc-prefix-status "10x")))
(newline)
(display (sblc-program-id K)) (newline)
(display (list (sblc-length K) (sblc-weight K))) (newline)
(display sblc-format-version) (newline)
```
```
1000
11110001010010
(lam (var 1))
#f
(complete trailing incomplete invalid)
sblc-v1:1010010
(7 1/128)
sblc-v1
```

### Choice, closedness and rendering

| Function | Contract |
|---|---|
| `(sblc-choice m n)` | the term `(choice m n)` |
| `(sblc-choice? t)` | `#t` iff `t` is a choice |
| `(sblc-choice-left t)`, `(sblc-choice-right t)` | the branches |
| `(sblc-term? t)` | `#t` iff `t` is a well-formed four-form term |
| `(sblc-closed? t)` | `#t` iff every variable is bound inside `t` |
| `(sblc-closed-at? t d)` | closedness under `d` enclosing binders |
| `(sblc->string t)` | De Bruijn rendering, `λ` for abstraction and `⊕` for choice |

```scheme
(require core.blc)
(require core.sblc)
(define coin (blc-lam (sblc-choice (blc-var 1) (blc-lam (blc-var 2)))))
(display (sblc->string coin)) (newline)
(display (sblc-choice? (blc-lam-body coin))) (newline)
(display (sblc-choice-left (blc-lam-body coin))) (newline)
(display (sblc-choice-right (blc-lam-body coin))) (newline)
(display (list (sblc-term? coin) (sblc-closed? coin)
               (sblc-closed? (sblc-choice (blc-var 1) (blc-var 1)))
               (sblc-closed-at? (sblc-choice (blc-var 1) (blc-var 1)) 1)))
(newline)
```
```
λ(1 ⊕ λ2)
#t
(var 1)
(lam (var 2))
(#t #t #f #t)
```

## Reduction

Reduction is strong leftmost-outermost. A redex is a beta redex
`(app (lam B) N)` or a choice. A choice inside an argument that is discarded
is never reached: `K I (Ω ⊕ M)` contracts both beta redexes, consumes zero
coins and normalizes to `I`.

| Function | Contract |
|---|---|
| `(sblc-step t)` | `(normal)`, `(beta t1)` with the contractum, or `(choice t0 t1)` with the leftmost-outermost choice replaced by its left / right branch |
| `(sblc-normal? t)` | `#t` iff `t` has no redex |
| `(sblc-shift d cutoff t)` | add `d` to every free index `>= cutoff` |
| `(sblc-subst t j s)` | replace `(var j)` by `s`, shifting under binders |

```scheme
(require core.blc)
(require core.sblc)
(define I (blc-lam (blc-var 1)))
(define K (blc-lam (blc-lam (blc-var 2))))
(display (sblc-step (blc-app I K))) (newline)
(display (sblc-step (blc-app (sblc-choice I K) (blc-var 1)))) (newline)
(display (sblc-step K)) (newline)
(display (sblc-normal? K)) (newline)
(display (sblc-shift 1 1 (blc-lam (blc-app (blc-var 1) (blc-var 2))))) (newline)
(display (sblc-subst (sblc-choice (blc-var 1) (blc-var 2)) 1 I)) (newline)
```
```
(beta (lam (lam (var 2))))
(choice (app (lam (var 1)) (var 1)) (app (lam (lam (var 2))) (var 1)))
(normal)
#t
(lam (app (var 1) (var 3)))
(choice (lam (var 1)) (var 2))
```

## Exact distribution semantics

`(sblc-distribution t S)` returns the exact distribution of `t` under the step
bound `S`, where `S` limits the number of contractions (beta steps plus coins)
along every execution path. Reduction is explored layer by layer; identical
terms in a layer are merged (De Bruijn terms are canonical, so
alpha-equivalent terms merge). A path still reducible after `S` contractions
contributes its mass to the timeout. A timeout is a statement about the bound,
never a claim of divergence.

| Function | Contract |
|---|---|
| `(sblc-distribution t S)` | distribution record |
| `(sblc-dist-outcomes d)` | list of `(normal-form . mass)`, sorted by code length, then code bits |
| `(sblc-dist-timeout d)` | exact timeout mass |
| `(sblc-dist-normal-mass d)` | sum of outcome masses; `normal + timeout = 1` |
| `(sblc-dist-mass-of d t)` | mass of the normal form `t` (0 when absent) |

```scheme
(require core.blc)
(require core.sblc)
(define I (blc-lam (blc-var 1)))
(define K (blc-lam (blc-lam (blc-var 2))))
(define omega (blc-app (blc-lam (blc-app (blc-var 1) (blc-var 1)))
                       (blc-lam (blc-app (blc-var 1) (blc-var 1)))))
(define d (sblc-distribution (blc-app (sblc-choice I K) (sblc-choice K I)) 10))
(display (sblc-dist-outcomes d)) (newline)
(display (list (sblc-dist-normal-mass d) (sblc-dist-timeout d))) (newline)
(define e (sblc-distribution (sblc-choice omega I) 25))
(display (list (sblc-dist-mass-of e I) (sblc-dist-timeout e))) (newline)
(define lazy (sblc-distribution (blc-app (blc-app K I) (sblc-choice omega I)) 10))
(display (sblc-dist-outcomes lazy)) (newline)
(display (sblc-dist-outcomes (sblc-distribution (sblc-choice K K) 1))) (newline)
```
```
(((lam (var 1)) . 1/4) ((lam (lam (var 1))) . 1/4) ((lam (lam (var 2))) . 1/4) ((lam (lam (lam (var 2)))) . 1/4))
(1 0)
(1/2 1/2)
(((lam (var 1)) . 1))
(((lam (lam (var 2))) . 1))
```

## Sampled semantics

A run is driven by a coin stream: a thunk that returns `(bit . next-stream)`,
or `#f` when the stream is exhausted. Coin 0 selects the left branch, coin 1
the right. The seeded sampler draws each coin with
`(prng-random-integer prng 2)` from `(make-prng seed)`, the RNG policy named
by `sblc-rng-policy` (`drand48-msb-v1`). The tuple
`(policy, seed, bound, program, input)` reproduces the same coins, result,
counters and trace on JIT and AOT.

| Function | Contract |
|---|---|
| `(sblc-run t S coins)` | run under bound `S`, drawing from the coin stream `coins` |
| `(sblc-run-coins t bits S)` | replay a run from an explicit coin string |
| `(sblc-sample t seed S)` | seeded run with coins from `(make-prng seed)` |
| `(sblc-sample-prng prng t S)` | run drawing from an existing PRNG |
| `(sblc-coins-from-bits bits)`, `(sblc-coins-from-prng prng)` | coin streams |
| `(sblc-run-status r)` | `'normal`, `'timeout` (bound reached) or `'needs-coin` (stream exhausted) |
| `(sblc-run-term r)` | final term |
| `(sblc-run-beta r)`, `(sblc-run-coins-used r)`, `(sblc-run-steps r)` | counters; steps = beta + coins, never above the bound |
| `(sblc-run-trace r)` | the coins drawn, as a bit string |
| `sblc-rng-policy` | the symbol `drand48-msb-v1` |

```scheme
(require core.blc)
(require core.sblc)
(define I (blc-lam (blc-var 1)))
(define K (blc-lam (blc-lam (blc-var 2))))
(define t (blc-app (sblc-choice I K) (sblc-choice K I)))
(define r (sblc-run-coins t "10" 20))
(display (list (sblc-run-status r) (sblc-run-term r) (sblc-run-beta r)
               (sblc-run-coins-used r) (sblc-run-steps r) (sblc-run-trace r)))
(newline)
(display (sblc-run-status (sblc-run-coins t "1" 20))) (newline)
(display (sblc-run-status (sblc-run-coins t "10" 2))) (newline)
(define s (sblc-sample t 2026 20))
(display (list (sblc-run-status s) (sblc-encode (sblc-run-term s)) (sblc-run-trace s)))
(newline)
(display (equal? s (sblc-sample t 2026 20))) (newline)
(define r2 (sblc-run t 20 (sblc-coins-from-bits "01")))
(display (sblc-run-trace r2)) (newline)
(define p (make-prng 7))
(display (sblc-run-trace (sblc-sample-prng p t 20))) (newline)
(display (sblc-run-trace (sblc-run t 20 (sblc-coins-from-prng p)))) (newline)
(display sblc-rng-policy) (newline)
```
```
(normal (lam (lam (lam (var 2)))) 1 2 3 10)
needs-coin
timeout
(normal 1010010 00)
#t
01
01
00
drand48-msb-v1
```

## Prior, counts and enumeration

`count(n, d)`, the number of terms of code length `n` closed under `d`
binders, satisfies `count(n, d) = [2 <= n <= d+1] + count(n-2, d+1) +
2 Σ_{a+b=n-3} count(a, d) count(b, d)`; a term of length `n` mentions only
indices below `n`, so `d >= n` gives the unrestricted count.

| Function | Contract |
|---|---|
| `(sblc-count n)`, `(sblc-count-closed n)` | exact number of terms / closed programs of code length `n` |
| `(sblc-enumerate n)`, `(sblc-enumerate-closed n)` | every such term, sorted by code bits |
| `(sblc-prior-mass L)` | exact `Σ_{n<=L} count(n) / 2^n` (below 1, tending to 1) |
| `(sblc-closed-prior-mass L)` | the same over closed programs |
| `(sblc-sample-program prng max-bits)` | raw sampler: fair bits until a term is complete; `#f` if unfinished after `max-bits` |
| `(sblc-sample-closed-program prng L)` | exact draw from the prior restricted to closed programs of length `<= L` (rejection); `L >= 4` |

```scheme
(require core.blc)
(require core.sblc)
(display (map sblc-count '(2 3 4 5 6 7 8))) (newline)
(display (map sblc-count-closed '(4 5 6 7 8 9))) (newline)
(display (map sblc-encode (sblc-enumerate 5))) (newline)
(display (map sblc->string (sblc-enumerate-closed 9))) (newline)
(display (sblc-prior-mass 10)) (newline)
(display (sblc-closed-prior-mass 10)) (newline)
(define p (make-prng 11))
(display (sblc-encode (sblc-sample-program p 40))) (newline)
(display (sblc->string (sblc-sample-closed-program p 12))) (newline)
```
```
(1 1 2 2 3 5 8)
(1 0 1 1 1 3)
(01110 10010)
(λλλ2 λ(1 1) λ(1 ⊕ 1))
753/1024
25/256
11010000110
λ1
```

## Observations and exact conditioning

An example pairs an input term with the expected output term, both in
`core.blc`'s Scott-list encodings: bit strings through `blc-list-of-bits`,
BLC8 byte lists through `blc-bytes->term`. A program `P` explains an example
with the exact mass of normal forms of `(app P input)` equal to the expected
output.

`(sblc-condition seed L S examples)` evaluates the **complete bounded program
set** (every closed program of length `<= L`, sorted by length, then bits)
against every example and returns a receipt:

- per program: code bits, length, prior `2^-length`, per-example likelihoods
  and timeout masses, likelihood (their product), weight (`prior ×
  likelihood`) and posterior (`weight / evidence`, or a zero sentinel when
  evidence is zero; that sentinel is not a normalized posterior);
- the evidence (sum of weights), the **shortest** witness (the first program
  with positive weight) and the **MDL** witness (the largest weight, ties to
  the earlier program);
- status `'consistent` when bounded evidence is positive; `'unresolved` when
  evidence is zero but at least one candidate has positive
  `likelihood + timeout mass` on **every** example; or `'no-consistent-program`
  when every candidate is definitively ruled out by some example (including
  an empty program set);
- seeded runs of the MDL witness on each example input, drawing from
  `(make-prng seed)`.

Unfinished trials preserve the possibility of matching. For independent trials,
a program's joint likelihood is bounded above by the product of its per-example
`likelihood + timeout mass` values. This upper bound is positive only if all
factors are positive: a timed-out trial does not rescue a candidate contradicted
by a different completed trial. The status is relative to the declared finite
program set, not a claim about all programs. Positive-evidence behavior and the
receipt layout are unchanged; both zero-evidence statuses have no selected
witness and retain the zero-posterior sentinel.

For example, identity needs one beta step to return its identity input:

```scheme
(define status-identity (sblc-decode "1000"))
(define status-example (list (cons status-identity status-identity)))
(sblc-receipt-status (sblc-condition 7 4 0 status-example)) ; unresolved
(sblc-receipt-status (sblc-condition 7 4 1 status-example)) ; consistent
```

`(sblc-receipt->string r)` is the canonical text of a receipt; identical
tuples `(seed, L, S, examples)` give byte-identical text on JIT and AOT.

### Resuming bounded search in memory

```scheme
(define pending (sblc-search-start 7 4 0 status-example))
(sblc-receipt-status (sblc-search-receipt pending)) ; unresolved
(define refined (sblc-search-resume pending 1))
(sblc-receipt-status (sblc-search-receipt refined)) ; consistent
(sblc-search-work refined) ; (1 1): newly selected, cumulative transitions
```

`sblc-search-start` fixes `(seed, L, examples)` and captures a detached snapshot
of the complete bounded catalogue's independent program/example trials.
`sblc-search-resume` accepts the handle and a nondecreasing exact step bound;
it advances unfinished probability frontiers with their exact masses and
retains completed outcomes. Same-bound continuation selects no transitions.
The returned handle is a separate continuation: the original remains usable
for independent forks. There is no worker or implicit filesystem write.

`sblc-search-receipt` returns the existing conditioning receipt format, using
the same likelihood, posterior, shortest/MDL and seeded witness construction
as `sblc-condition`. Increasing a bound produces the same canonical receipt as
fresh evaluation at that bound, including witness traces. Returned receipt
pairs and strings are detached from the handle. Changing the catalogue,
examples, their order/multiplicity or the seed requires a new search. Repeated
examples remain independent trials; the continuation never couples their coins.

`sblc-search-work` returns `(newly-selected cumulative-selected)` transition
counts for the exact distribution exploration. One beta contraction or choice
expansion of a merged frontier term counts once. These counts exclude normality
inspection, term traversal, receipt construction and seeded witness replay;
they are not elapsed time, total kernel invocations or allocation measurements.

Handles are opaque in-memory procedures produced by this module. Pass trusted
module-created handles to the accessors; arbitrary procedures are not admitted
as authenticated execution history. They are not serialized checkpoints or
portable certificates. Durable restoration and deterministic shard composition
remain separate work. Bounds must be exact nonnegative integers; the seed must
be an exact integer. Example admission allows at most 64 finite, structurally
valid term pairs, 300000 datum occurrences and 256 car-nesting levels, rejecting
cycles before copying. Search expansion retains the existing bounded evaluator's
costs and can still require substantial memory for a large catalogue/frontier.

Run `ESHKOL_BIN=/path/to/eshkol-run bash scripts/run_sblc_resume_gate.sh`.
The gate checks fresh/resumed equality across 168 context/bound combinations,
partially completed probability, independent repeated trials, same-bound zero
work, detached mutation/forks and invalid continuation bounds in JIT and AOT.

| Function | Contract |
|---|---|
| `(sblc-bits-example in out)` | example from two bit strings |
| `(sblc-bytes-example in out)` | example from two lists of byte values (BLC8) |
| `(sblc-likelihood program example S)` | `(likelihood timeout-mass)` of one example |
| `(sblc-condition seed L S examples)` | receipt |
| `(sblc-receipt-status r)`, `(sblc-receipt-evidence r)`, `(sblc-receipt-programs r)`, `(sblc-receipt-shortest r)`, `(sblc-receipt-mdl r)`, `(sblc-receipt-witness-runs r)` | receipt fields; the witnesses are entries or `#f` |
| `(sblc-entry-bits e)`, `(sblc-entry-length e)`, `(sblc-entry-prior e)`, `(sblc-entry-likelihoods e)`, `(sblc-entry-timeouts e)`, `(sblc-entry-likelihood e)`, `(sblc-entry-weight e)`, `(sblc-entry-posterior e)` | entry fields |
| `(sblc-receipt->string r)` | canonical receipt text |

```scheme
(require core.blc)
(require core.sblc)
(define examples (list (sblc-bits-example "0" "0") (sblc-bits-example "10" "10")))
(define r (sblc-condition 2026 9 40 examples))
(display (list (sblc-receipt-status r) (sblc-receipt-evidence r))) (newline)
(display (sblc-entry-bits (sblc-receipt-shortest r))) (newline)
(define mdl (sblc-receipt-mdl r))
(display (list (sblc-entry-bits mdl) (sblc-entry-length mdl) (sblc-entry-prior mdl)
               (sblc-entry-likelihoods mdl) (sblc-entry-timeouts mdl)
               (sblc-entry-likelihood mdl) (sblc-entry-weight mdl) (sblc-entry-posterior mdl)))
(newline)
(display (length (sblc-receipt-programs r))) (newline)
(display (length (sblc-receipt-witness-runs r))) (newline)
(display (sblc-likelihood (blc-lam (sblc-choice (blc-var 1) (blc-lam (blc-lam (blc-var 1)))))
                          (sblc-bits-example "" "") 40))
(newline)
(define none (sblc-condition 7 8 30 (list (sblc-bytes-example '(65) '(65 66)))))
(display (list (sblc-receipt-status none) (sblc-receipt-shortest none))) (newline)
(display (sblc-receipt->string (sblc-condition 1 6 10 (list (sblc-bytes-example '(1) '(1))))))
(newline)
```
```
(consistent 33/512)
1000
(1000 4 1/16 (1 1) (0 0) 1 1/16 32/33)
7
2
(1 0)
(no-consistent-program #f)
sblc-receipt sblc-v1 rng drand48-msb-v1 seed 1 L 6 S 10 examples 1
status consistent
programs 2
evidence 1/16
shortest 1000
mdl 1000
program 1000 length 4 prior 1/16 likelihoods [1] timeouts [0] weight 1/16 posterior 1
program 101000 length 6 prior 1/64 likelihoods [0] timeouts [0] weight 0 posterior 0
run normal result 1011011000101101100010100101011011000101001010110110001010010101101100010100101011011000101001010110110001010010101101100010100101011011000101000101000101000 beta 1 coins 0 trace -
```

The identity `λ1` (`1000`) and the stochastic program `λ(1 ⊕ 1)`
(`101110000`, likelihood 1) both explain the two bit-list examples; their
posterior masses are `32/33` and `1/33`.

## Verification

`scripts/run_sblc_gate.sh` (CTest: `sblc_gate`) runs
[`tests/stdlib/sblc_test.esk`](../../../tests/stdlib/sblc_test.esk) on JIT and
AOT. Each oracle is independent of the code path it checks:

| Oracle | Check |
|---|---|
| codec | for every length up to 12, brute-force strict decoding of all bit strings equals structural enumeration and the count recurrence; round trips consume every bit; no trailing-bit aliases; closedness is depth-aware |
| mass conservation | for every `L` up to 14, prior mass plus the mass of incomplete prefixes of length `L` is exactly 1; the root constructor prefixes carry exactly 1/2, 1/4, 1/8, 1/8; the fixed point selects the least root; `normal + timeout = 1` for every distribution checked |
| coin traces | every exact distribution equals the sum, over exhaustively replayed coin traces, of `2^-|trace|` per outcome |
| semantics | lazy choice, half mass per branch, `M ⊕ M` and alpha-identical merging, step-bound edges, impossible evidence |
| sampler statistics | constructor and term frequencies inside the predeclared bound `|observed - Np| <= 5 sqrt(Np(1-p))`, seeded reproducibility |
| conditioning receipts | bit-list and BLC8 receipts, exact posteriors summing to 1, reproducible receipt text |
| JIT/AOT parity | the two engines' complete output (receipts and a seeded sample trace) is byte-identical |

The gate writes ICC `runtime_event` traces (kind `sblc`) to
`scripts/icc_traces/sblc.jsonl`; the `v1.4-connection` readiness target
reads them.

## Engines

JIT and AOT run the whole module. On the bytecode VM, the exact path
(codec, enumeration, counts, prior masses and `sblc-distribution`) runs; the
sampled functions (`sblc-sample`, `sblc-sample-prng`, `sblc-coins-from-prng`,
`sblc-sample-program`, `sblc-sample-closed-program`) and `sblc-condition`,
whose receipts carry seeded witness runs, draw from `make-prng`, which the VM
does not bind: the VM refuses those calls instead of substituting another
generator. VM support for them is Planned for v1.4.1 (recorded in
[`tests/vm_parity/PARITY.tsv`](../../../tests/vm_parity/PARITY.tsv)).

## Scope of this slice

This page covers the first slice of ADR-0017. The universal interpreter
`U⊕`, resumable mesh-sharded search, DBSP/N3 receipt integration and
differentiable proposals are later ADR-0017 stages.
