# `core.observable_closure`

Compile the smallest rational row space containing requested observations and
closed under every declared generator action. An independent checker proves
query inclusion, invariance, independence and reachable provenance. Polynomial
drift/jump sources lower through `core.symbolic`; exact affine solves reuse
`core.exact_linalg`.

```scheme
(require core.observable_closure)
(define queries '((1 1)))
(define actions '((g (2 0) (0 2))))
(define result (compile-observable-closure queries actions))
(observable-status result)    ; closed
(observable-dimension result) ; 1
(define proof (observable-certificate result))
(verify-observable-closure queries actions proof) ; #t
(project-observations queries actions proof '((3 3))) ; (available ((3)))
(car (project-observations queries actions proof '((1 0)))) ; reconstruction-required
```

Rows are lists of exact rationals. An action is `(name row ...)`, with a
unique symbol or exact integer name and a square matrix; rows act on the
right. The profile admits 1..32 ambient coordinates, 1..32 queries and 0..16
actions. The optional third argument to `compile-observable-closure` is
`(max-retained-dimension max-actions)`, default `(32 4096)`, capped at
`(32 65536)`. Exhaustion returns `unknown` with no certificate. A zero query
may produce a zero-dimensional space. Inputs are snapshotted; keep them
immutable during a call, and keep a published certificate immutable afterward.

The certificate is
`#(observable-closure-v1 n B R reduced-actions origins D)`.
Replay checks `Q=RB`, `BG_a=C_aB`, `BD=I`, and that each row of B is an actual
query followed by the recorded action word. These four facts prove minimality
in the declared representation. Replay calls no search, rank or affine solver.
Malformed, inexact, changed-source and forged certificates return false.
Numerical overflow that loses rational exactness cannot certify a result.

`project-observations queries actions proof requested` returns
`(available coordinates)` or `(reconstruction-required query-index residual)`.
It cannot invent a distinction absent from the retained space.

## Polynomial generators

`compile-generator-closure spec [limits]` accepts exactly six association-list
fields: `variables`, `parameters`, `basis`, `drift`, `jumps`, `observations`.
The polynomial generator is
`A p = sum b_i*d_i(p) + sum rate_j*(p(updates_j)-p)`.
Jump updates are simultaneous. A basis entry is one state-exponent list.

```scheme
(define spec
  '((variables x n) (parameters w)
    (basis (0 0) (1 0) (0 1) (2 0) (1 1) (0 2))
    (drift (* -2 x) 0)
    (jumps ((+ 1/2 x) ((+ x w) (+ n 1))))
    (observations n)))
(define compiled (compile-generator-closure spec))
(verify-generator-closure spec (observable-certificate compiled)) ; #t
```

Source expressions admit exact rational literals, declared symbols, `+`, `-`,
`*`, and `expt` with exponent 0..8. There are at most four state variables, two
parameters, eight jumps, 32 basis monomials and 32 queries. Expressions have
depth at most 24 and arity at most 16; each polynomial has total degree at most
16 and 1024 terms, with preflight at 4096 multiplication pairs. These are
finite admission bounds, not limits on every host allocation or arithmetic bit.

Parameter monomials become separate actions, so closure holds for the entire
declared rational parameter family, rather than one fitted parameter value.
Ambient projection is only a proposal: the source checker independently uses
monomial derivative rules and simultaneous substitutions to check full,
unprojected generator identities on every retained observable. Unneeded
ambient monomials may leave the envelope. A required nonzero residual returns
`outside-basis` without a certificate; expansion/admission exhaustion is
`unknown`, and malformed source data raises. Generator certificates have shape
`#(generator-closure-v1 parameter-powers queries actions linear-certificate)`.

The two-neuron Hawkes example in the gate retains 4/15 coordinates for one
mean and 11/15 for covariance seeds. It does not establish finite moments,
nonnegative rates, non-explosion, continuous-time integration accuracy or a
runtime speedup. The polynomial arithmetic, source rules, rational arithmetic
and compiler remain trusted code; this is no proof-assistant formalization.

Run `python3 scripts/run_observable_closure_gate.py --eshkol-bin /path/to/eshkol-run`.
The native gate compares exact span dimensions and source matrices against a
separate Fraction-based oracle. It also replays a source-bound certificate
with seven producer routines replaced by errors, in both JIT and AOT.
For a published compiler whose precompiled stdlib predates `exact-affine-solve`,
add `--compile-linalg-source`: AOT explicitly compiles this checkout's existing
`core.exact_linalg` source. A matching current build needs no compatibility flag.

The native gate explicitly sets `ESHKOL_JIT_CACHE=0` for every child, so its
`-r` leg runs in-process JIT rather than the persistent executable-cache route.
The readiness self-test intercepts the real first child boundary with the
variable unset and set to `1` in the caller. Both must pass `0` to the child
without changing the caller environment; this control launches no native work.
The gate separately requires complete native receipts and independent oracles.
