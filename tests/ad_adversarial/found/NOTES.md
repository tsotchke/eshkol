# Findings from the generative adversarial AD sweep

Shrunk repros for divergences this harness exposes live here (one `.esk` per
finding), mirroring `tests/ad_oracle/found/`.

## Master run, 2026-07-10

### Zero gradient at a `(vector …)` evaluation point

`esh0235_tensor_grad_vector_ctor_point_zero.esk`. When filed, reverse-mode AD
through a tensor op (`tensor-dot`, `tensor-sum` of `tensor-mul`,
`tensor-matmul`, …) returned an all-zero gradient when the differentiation
point was built with the `(vector …)` constructor, while the same point as a
`#(…)` reader literal or a `(tensor …)` value gave the analytic gradient. It
reproduced identically under `-r` and AOT: the tensor-op reverse path did not
recognise a `(vector …)`-constructed value as a differentiable tensor seed and
taped nothing. Scalar/vector-FIELD AD at a `(vector …)` point was unaffected,
and the existing tensor-AD unit tests could not see it because they always
seeded the gradient with a `#(…)` literal or a `(tensor …)` point.

At v1.3.6-evolve all three spellings agree:

```
(gradient (lambda (z) (tensor-dot z #(2.0 -1.0 0.5))) (vector 1.0 2.0 4.0))
  => #(2 -1 0.5)       ; (vector …) constructor
(gradient (lambda (z) (tensor-dot z #(2.0 -1.0 0.5))) #(1.0 2.0 4.0))
  => #(2 -1 0.5)       ; reader literal
(gradient (lambda (z) (tensor-dot z #(2.0 -1.0 0.5))) (tensor 3 1.0 2.0 4.0))
  => #(2 -1 0.5)       ; tensor constructor
```

The `vecpoint` family keeps this shape under the gate as an ordinary family.

### AD values elsewhere

In the 2026-07-10 run all 427 non-`vecpoint` generated component checks
matched central finite differences under BOTH the JIT and AOT — no divergent or
silent-zero gradient across the scalar, field, gradient-of-gradient,
tensor/ML-op and higher-order-tensor families. That run also confirmed the two
shapes this family was built to watch:

- vector-parameter gradient-of-gradient — the `gofg` family returns the
  non-zero second-order gradient (e.g. `d/dv 3v^2 = 6v = 12` at `v=2`), not
  `#(0)`;
- Hessian / Laplacian at a tensor-literal point — the `htensor` and `field`
  families evaluate at `(tensor …)` points and match FD.

## JIT compile-path robustness (not a gradient value)

Running the sweep surfaced an **intermittent fatal signal in the MULTI-THREADED
JIT compile path** on the AD/tensor-heavy modules (the `tensor` and `htensor`
families). The fault address was `0xfffffffffffffff8` (`-8` off a
null/uninitialised pointer), and the 512MB default stack was untouched, so this
is a compile-time race, not stack exhaustion or a derivative value. It moved
between files run-to-run and disappeared with `ESHKOL_JIT_COMPILE_THREADS=1`;
AOT was unaffected. `scripts/run_ad_adversarial.sh` therefore pins the JIT lane
to a single compile thread and a fresh per-run cache so the gate reliably tests
gradient values; reproduce the race by exporting `ESHKOL_JIT_COMPILE_THREADS`
to a value > 1. This belongs to the JIT infrastructure, not the AD engine, and
is tracked separately.
