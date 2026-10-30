# `core.sblc.resource`

Exact source-derived termination and work certificates for one explicitly
admitted SBLC grammar. Require this module separately. It uses the existing
SBLC syntax, codec and reduction semantics without changing `core.sblc`.

```scheme
(require core.sblc.resource)
(define source (sblc-latent-source 'shared 1 3 2))
(define result (sblc-resource-analyze source 24))
(sblc-resource-status result) ; certified
(define proof (sblc-resource-certificate result))
(sblc-resource-verify source proof) ; #t
(sblc-resource-summary proof) ; (5/9 10/3 5 infinite 15)
```

## Source grammar and work units

For `I = lambda u.u`, let
`D = lambda x. choose2(I, ((x x)(x x)))` and `X = D D`.
`choose2` is a complete depth-two fair-choice tree with exactly k identity
leaves and 4-k recursive leaves. Every invocation draws exactly two coins,
including when the chosen leaves are identical. A terminating tree with n
internal nodes costs `3n+1` beta contractions and `4n+2` coins: **7n+3 total
steps**, exactly Eshkol's bound convention.

Consequently its total-work PGF satisfies
`H(z) = p*z^3 + (1-p)*z^4*H(z)^2`, where `p=k/4`.
Finite trees exhaust termination; its probability is the least root of
`q=p+(1-p)*q^2`. At p=1/2 termination is almost sure with infinite expected
work. Below p=1/2, nontermination has positive probability. These are results
of the admitted recursive source rule, never conclusions from a timeout.

`sblc-branching-source k` constructs X for k in 0..4.
`sblc-latent-source mode k0 k1 m` constructs m left-associated invocations:
`shared` chooses the parameter once outside the whole sequence, while `fresh`
chooses it independently for each invocation. Modes are symbols, m is 1..8,
and selectors are fair. Recursive descendants keep their invocation's parameter.
At m=1 shared and fresh constructors have the same source and normalize to
the shared grammar. The analyzer reads actual syntax, not constructor metadata.

Shared composition has PGF `z^m*(H0^m+H1^m)/2`; fresh composition has
`z^(2m-1)*((H0+H1)/2)^m`. Identity sequencing costs m-1 betas. Selection
costs one coin for shared sources and m coins for fresh sources.

At k0=1, k1=3 and m=2, termination probabilities are **5/9 versus 4/9**.
Conditional total work is **15 versus 16**, rather than equal after discarding
coin costs. Neither program has finite unconditional expected total work.

## API and checking

`sblc-resource-analyze source degree` returns a versioned result with status
`certified` and a certificate, or `unknown` without one. Degree must be 0..48.
The structural profile is 4096 AST occurrences and depth below 64; unsupported,
malformed or open sources cannot certify. Cost coefficients occupy degree+1
exact rational entries; omitted higher costs are not assigned zero probability.
Lost arithmetic exactness refuses. Source/certificate arguments must remain
immutable during a call; published certificates must remain immutable.

Read a result with `sblc-resource-status` and `sblc-resource-certificate`.
Read fresh certificate lists with `sblc-resource-coefficients` and
`sblc-resource-summary`. The summary is
`(termination-mass restricted-beta-moment restricted-coin-moment
unconditional-total-mean conditional-total-mean)`.
Restricted moments integrate over finite executions only; `infinite` and
`undefined` are explicit symbols. Coefficient mass through a finite bound is
completed work; termination mass minus completed mass is still-terminating
tail mass. A bounded evaluator's timeout also contains any nonterminating mass.

Certificates have shape
`#(sblc-resource-v1 program-id descriptor degree branch-records coefficients summary)`.
The checker re-admits source syntax and its full codec identity, verifies the
least-root condition, both differentiated work equations, and every coefficient
of the positive-cost fixed point. It composes output coefficients by independent
diagonal convolutions and moments by a separate mass/moment semiring. It calls
no source constructor, coefficient producer or evaluator. Source admission,
the head-to-identity recursion/sequence lemmas, exact arithmetic and compiler
remain trusted. This is no proof-assistant formalization or general stochastic
higher-order analysis. Formal-series payload grows with requested degree.

Run `python3 scripts/run_sblc_resource_gate.py --eshkol-bin /path/to/eshkol-run`.
The gate checks eleven profiles, sixteen operational observations against a
separate de Bruijn reducer, exact moment/coefficient oracles, and source-bound
replay with eleven producers disabled, on JIT and AOT. It also rejects forged
coefficients, latent summaries, source identities and incomplete native output.
