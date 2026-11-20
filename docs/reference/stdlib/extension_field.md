# `core.extension_field`

Finite fields `GF(p^m)` as ordinary `core.ring` carriers. Require this module
explicitly. Arithmetic, polynomial division and gcd reuse `core.ring`.

```scheme
(require core.extension_field)
(define F (make-extension-field 2 '(1 1 1))) ; F_2[x]/(x^2+x+1)
(define a (extension-element F '(0 1)))
(extension-coefficients F (r* F a a)) ; (1 1)
(extension-minimal-polynomial F a)    ; (1 1 1)
(r-zero? F (r-int F 2))             ; #t
(extension-element->index F (extension-index->element F 2)) ; 2
```

`make-extension-field p modulus` requires a supplied monic polynomial,
coefficients lowest degree first. It checks exact primality and Rabin's
irreducibility criterion before returning a carrier. The finite profile is
`2 <= p <= 65521` and degree `1..32`. Composite bases, reducible/nonmonic
moduli, inexact coefficients and inputs outside this profile raise errors.
There is no implicit modulus search, probabilistic primality admission or
silent quotient by a reducible polynomial.

| Operation | Contract |
|---|---|
| `extension-field?` | Recognize this module's carriers. |
| `extension-prime`, `extension-degree`, `extension-order`, `extension-modulus` | Read p, m, p^m, or a fresh modulus coefficient list. |
| `extension-element F coefficients` | At most m exact integer coefficients, reduced modulo p; owns its result. |
| `extension-coefficients F a` | Fresh canonical coefficient list; trailing zero coefficients omitted. |
| `extension-index->element F n`, `extension-element->index F a` | Bijection with indices `0..p^m-1` via base-p digits. This is distinct from the ring map `r-int`. |
| `extension-frobenius F a [k]` | `a^(p^k)` for nonnegative exact k, reduced modulo m; k defaults to 1. |
| `extension-conjugates F a` | Distinct Frobenius orbit. |
| `extension-minimal-polynomial F a` | Monic minimal polynomial over F_p, as a fresh coefficient list. |

Use `r+`, `r-`, `r*`, `r/`, `r-inv`, `r-expt`, `r=`, `ring-det`,
`ring-solve`, and `make-jets` with the returned carrier. The base prime field
embeds as constants. Different constructor calls produce distinct carriers;
no implicit isomorphism is selected between representations. Elements and
carriers must remain immutable after publication, as with existing ring
descriptors. Constructor/readout lists are copied. Zero inversion refuses.

The source-derived irreducibility rule, the ring implementation and compiler
are trusted code. Finite gates independently compare every sum/product in
four fields and all 54 minimal polynomials using coefficient-array arithmetic;
they also cover the AES polynomial, exact characteristic, jets, determinants,
ownership and refusals. This is no cryptographic timing or security claim.

Run `python3 scripts/run_extension_field_gate.py --eshkol-bin /path/to/eshkol-run`.
JIT and AOT-O2 must match the independent oracle byte for byte. The separate
`scripts/extension_field_gate_self_test.py` rejects an equal but false pair of
native receipts.

The native gate explicitly sets `ESHKOL_JIT_CACHE=0` for every child, so its
`-r` leg runs in-process JIT rather than the persistent executable-cache route.
The readiness self-test intercepts the real first child boundary with the
variable unset and set to `1` in the caller. Both must pass `0` to the child
without changing the caller environment; this control launches no native work.
The gate separately requires complete native receipts and independent oracles.
