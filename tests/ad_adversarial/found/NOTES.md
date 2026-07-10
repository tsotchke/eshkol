# Findings from the generative adversarial AD sweep

Shrunk repros for real defects this harness exposes live here (one `.esk` per
defect, referenced by an ESH task), mirroring `tests/ad_oracle/found/`.

## Master run, 2026-07-10

### ESH-0235 — silent-zero gradient at a `(vector …)` evaluation point (NEW)

`esh0235_tensor_grad_vector_ctor_point_zero.esk`. Reverse-mode AD through a
tensor op (`tensor-dot`, `tensor-sum` of `tensor-mul`, `tensor-matmul`, …)
returns a silently-WRONG all-zero gradient when the differentiation point is
built with the `(vector …)` constructor:

```
(gradient (lambda (z) (tensor-dot z #(2.0 -1.0 0.5))) (vector 1.0 2.0 4.0))
  => #(0 0 0)          ; WRONG
(gradient (lambda (z) (tensor-dot z #(2.0 -1.0 0.5))) #(1.0 2.0 4.0))
  => #(2 -1 0.5)       ; correct (reader literal)
(gradient (lambda (z) (tensor-dot z #(2.0 -1.0 0.5))) (tensor 3 1.0 2.0 4.0))
  => #(2 -1 0.5)       ; correct (tensor ctor)
```

Reproduces identically under `-r` and AOT, so it is a codegen/AD-tape defect.
Scalar/vector-FIELD AD at a `(vector …)` point is unaffected — the bug is
specific to the tensor-op reverse path, which does not recognise a `(vector …)`
-constructed value as a differentiable tensor seed and tapes nothing. Invisible
to the existing tensor-AD unit tests because they always seed the gradient with
a `#(…)` literal or a `(tensor …)` point. Tracked as the `vecpoint` XKNOWN
family. Filed: `.swarm/tasks/ESH-0235.json`.

### AD correctness elsewhere is clean

All 427 non-`vecpoint` generated component checks match central finite
differences under BOTH the JIT and AOT — no wrong or silent-zero gradient
across the scalar, field, gradient-of-gradient, tensor/ML-op and
higher-order-tensor families. The two flagship silent-wrong shapes this family
was built to catch are confirmed FIXED on master (refuting the stale
`ad_oracle/README.md` "known-open" table):

- **ESH-0096** vector-parameter gradient-of-gradient — `gofg` family returns
  the correct non-zero second-order gradient (e.g. `d/dv 3v^2 = 6v = 12` at
  `v=2`), not `#(0)`.
- **ESH-0095** Hessian / Laplacian at a tensor-literal point — `htensor` and
  `field` families evaluate at `(tensor …)` points without SIGSEGV and match FD.

## Exposed robustness defect (not a wrong-gradient bug)

Running the sweep surfaced an **intermittent SIGSEGV in the MULTI-THREADED JIT
compile path** on the AD/tensor-heavy modules (the `tensor` and `htensor`
families). Fault address is `0xfffffffffffffff8` (i.e. `-8` off a
null/uninitialised pointer), and the 512MB default stack is untouched, so this
is a compile-time race, not stack exhaustion or a bad derivative. It moves
between files run-to-run and disappears with `ESHKOL_JIT_COMPILE_THREADS=1`;
AOT is unaffected. `scripts/run_ad_adversarial.sh` therefore pins the JIT lane
to a single compile thread and a fresh per-run cache so the gate reliably tests
gradient CORRECTNESS; reproduce the race by exporting
`ESHKOL_JIT_COMPILE_THREADS` to a value > 1. This belongs to the JIT
infrastructure, not the AD engine, and is filed as a separate concern.
