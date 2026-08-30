# ADR 0011: Stochastic Binary Lambda Calculus (Λ⊕ on a self-delimiting binary program grammar)

- **Status:** Proposed
- **Date:** 2026-08-30
- **Decision owners:** `core.blc`, exact arithmetic, PRNG/VM parity, search, and Noesis maintainers
- **Depends on:** ADR-0005 (lambda foundations), `core.blc`, exact bignum rationals, the isolated PRNG, ADR-0009 / `core.dbsp` (v1.5 integration)
- **Scope:** SBLC syntax and codec, fair binary choice, sampled and distribution semantics, the exact `2^-length` prior, conditioning, search receipts, a universal interpreter, and experimental proposal learning

## Decision

Eshkol defines **Stochastic Binary Lambda Calculus (SBLC)** as the composition of two existing calculi:

- **Stochastic lambda calculus** Λ⊕ — the untyped lambda calculus extended with a fair binary
  choice `M ⊕ N`, with a sampling operational semantics (a coin is drawn when a choice is
  reduced) and a distribution semantics (the result is a sub-probability distribution over
  normal forms), in the sense of Dal Lago and Zorzi's probabilistic operational semantics and
  Scott's stochastic lambda-calculi.
- **Binary lambda calculus** — Tromp's self-delimiting prefix code for closed De Bruijn terms,
  already shipped as `core.blc` with the 232-bit universal interpreter and BLC8 I/O.

The literal term does not appear in the literature; SBLC is Eshkol's synthesis, and it is
documented as such. Three layers are kept distinct and never conflated: (1) the language
(syntax, codec, reduction), (2) the probability semantics (sampling with explicit seeds, and
exact rational distributions under a step bound), and (3) the prior over programs
(`2^-length` from the prefix code), which makes every SBLC program space simultaneously a
probabilistic programming language and a Solomonoff-style universal prior.

### Grammar (`sblc-v1`)

A separate four-form prefix code. Classic BLC terms are **not** re-encoded; `sblc-v1` is a
distinct, versioned format so that no decoder ever has to guess which grammar it is reading.

```
sblc(var i)      = 0 1^(i-1) 0
sblc(lam M)      = 10  sblc(M)
sblc(app M N)    = 110 sblc(M) sblc(N)
sblc(choice M N) = 111 sblc(M) sblc(N)
```

The root alternatives are prefix-free. The generating function
`S(z) = z^2/(1-z) + z^2 S(z) + 2 z^3 S(z)^2` has least nonnegative root `S(1/2) = 1`, so
`Σ_M 2^-|sblc(M)| = 1` exactly: the fair-bit sampler over the grammar is a probability
distribution, with root probabilities variable `1/2`, lambda `1/4`, application `1/8`,
choice `1/8`.

### Semantics

- **Reduction:** strong leftmost-outermost reduction with choice-aware shift and
  substitution; `choice` in a discarded argument is never evaluated (`K I (Ω ⊕ M)` consumes
  zero coins and normalizes to `I`).
- **Sampling:** every coin comes from an explicitly seeded, isolated PRNG
  (`algorithm-version, seed, bounds, program, input`) and is recorded in a trace; the same
  tuple reproduces identical coins, result, counters and trace digest on every engine.
- **Distribution:** under a structured step bound `S`, the result is an exact rational
  sub-distribution over normal forms plus an exact timeout mass, with
  `normal_mass + timeout_mass = 1`; a timeout is never a claim of divergence.
- **Prior and posterior:** program weight `2^-|program|`, joint program-and-trace weight
  `2^-|program| · 2^-|trace|`; conditioning on I/O examples (bit lists, or BLC8 bytes) yields
  exact likelihoods and posterior masses over the complete bounded program set; description
  length and output mass are the search objectives handed to Noesis (N3) and the bounded
  search kernel (E6).

## Non-goals

No claim that SBLC is established literature; no decoder that guesses classic versus
stochastic format; no claim that a timeout proves divergence; no computable exact Solomonoff
induction or exact Kolmogorov complexity; no change to fair choice or learned semantics without
a new language version; no relaxed or neural result labelled exact before projection and
re-verification; no AIXI implementation in this milestone.

## Delivery

### v1.4 (with the bridge work of ADR-0002)

1. **Syntax and codec** — `core.sblc` (new, `lib/core/sblc.esk`): `choice` constructor and
   predicates, strict four-form encode/decode that consumes every bit, closedness, versioned
   identity, classic BLC non-regression (`core.blc` unchanged; `blc_test` expected bits
   unchanged).
2. **Operational kernel** — extended shift/substitution, strong leftmost-outermost
   `sblc-step`, structured step bounds, the seeded sampled evaluator, the exact rational
   distribution evaluator.
3. **Prior and search kernel** — exact per-length counts and enumeration, raw and
   bounded-closed samplers, `2^-length` weights, BLC/BLC8 observation adapters, exact
   conditioning, description-length and output-mass receipts.
4. **Verification** — `scripts/run_sblc_gate.sh`: brute-force codec oracle, exact mass
   conservation, exhaustive coin-trace oracle, sampler statistics under predeclared bounds,
   fixed-seed JIT/AOT parity, ICC trace; `docs/guide/STOCHASTIC_BINARY_LAMBDA_CALCULUS.md`;
   `core.blc`/`core.sblc` added to `.icc/architecture-model.yaml`.

**v1.4 exit:** given `(seed, L, S, examples)`, Eshkol emits a deterministic receipt with the
complete bounded program set, exact per-program likelihood and posterior mass, exact output
and timeout distributions, and the shortest and MDL witnesses; byte-identical on JIT and AOT.
The VM reports `unsupported` rather than falling back until its PRNG parity gaps close.

### v1.5

1. **Universal `U⊕`** — a self-interpreter with a coin (`lib/core/sblc/interpreter.esk`);
   direct and interpreted exact distributions and seeded traces must agree.
2. **Resumable E6 search** — dovetailed residual distributions, content-hash checkpoints,
   mesh sharding with deterministic merge (`lib/core/sblc/search.esk`).
3. **DBSP / N3 integration** — incremental example deltas and structured
   accepted / rejected / timeout / failure receipts.
4. **VM RNG parity** — close the four PRNG gaps (`make-prng`, `prng?`, `prng-random`,
   `prng-random-integer`) and require JIT/AOT/VM trace equality.
5. **Differentiable proposals** (experimental, `lib/core/sblc/relax.esk`) — exact finite-logit
   oracle on enumerated domains, REINFORCE with Rao–Blackwellized inner coins, Gumbel/Concrete
   grammar relaxation only under approximate receipts, every relaxed candidate projected to a
   strict closed program and re-verified; learned proposal order can never alter
   completeness, prior weights, fair-choice probabilities, or verifier outcomes.
6. **ADR-0005 hand-off** — export exact program bits, semantic version, RNG policy and the
   verification certificate to the program-capsule boundary.

## Verification gates (normative)

- Codec: `decode(encode(M)) = M` consuming every bit; classic encodings unchanged.
- Exact weights: every term weight is exactly `1/2^length` as an Eshkol rational; per-length
  counts equal the independent recurrence; cumulative mass never exceeds one; the grammar fixed
  point selects the least root one.
- Enumeration: structural `(length, bits)` enumeration equals brute-force strict decoding at
  every length in the gate envelope; no duplicates, no trailing-bit aliases; closedness is
  depth-aware.
- Sampler statistics: constructor frequencies match `(1/2, 1/4, 1/8, 1/8)` and term frequencies
  match `2^-length` under predeclared simultaneous confidence bounds; exact mass gates remain
  primary.
- Semantics: lazy choice (`K I (Ω ⊕ M)`), exact half mass per branch, `M ⊕ M` merging to
  mass one, step-bound edges reconciling beta and coin counters, iteration-order invariance,
  alpha-identical merging, impossible evidence returning `no-consistent-program`.
- Reproducibility: identical tuples reproduce identical receipts; v1.4 JIT/AOT, v1.5 JIT/AOT/VM.

## Consequences

`core.blc` remains the deterministic truth oracle and is untouched in the first slice.
SBLC gives the mathematical-research plan (ADR-0002) its program prior and bounded search
kernel, and gives ADR-0005 a canonical, verifiable program space with an explicit RNG policy.
The first experiment is a small falsifiable campaign: enumerate every closed `sblc-v1`
program through a modest `L`, compare exact distributions with exhaustive coin traces and
fixed-seed samples, condition on two bit-list examples and one BLC8 case, and emit N3 receipts
with deterministic mesh sharding.

## References

Tromp, *Binary Lambda Calculus and Combinatory Logic* (doi:10.1142/9789812770837_0014);
Dal Lago & Zorzi, *Probabilistic Operational Semantics for the Lambda Calculus*
(arXiv:1104.0195); Scott, *Stochastic λ-calculi: an extended abstract*; Ehrhard, Pagani &
Tasson, *Full abstraction for probabilistic PCF* (arXiv:1511.01272); Solomonoff (1964), Levin
(1973); Goodman et al., *Church*; Wood et al., *Anglican*; Ellis et al., *DreamCoder*;
Williams, *REINFORCE* (1992); Jang et al. / Maddison et al., Gumbel-Softmax / Concrete.
