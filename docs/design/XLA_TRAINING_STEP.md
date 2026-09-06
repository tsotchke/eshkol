# The mixed-curvature training step, and what the device program mirrors

Status: implemented (Stage 4 of the XLA-to-TPU program, criterion
`xla_training_step_parity` in `.icc/completion-oracles.yaml`).

This document is written BEFORE the device lowering, and it exists because a
parity test is only worth something when both sides are pinned to a written
definition first. Everything below is transcribed from host code that this
repository contains; where a formula had to be chosen between two host
conventions that disagree, the disagreement is named and the choice is
justified rather than quietly settled.

## 0. The thing this stage found first: there was no host training step

The stage brief assumed a host training step existed to be mirrored. It did
not. Searching `lib/`, `inc/` and `tests/` for a function that performs
embedding forward, scoring, loss, reverse pass and a Riemannian optimizer
update as one unit returns nothing:

- `lib/ml/` holds three `.esk` files (`activations.esk`, `optimization.esk`,
  `nested_test_module.esk`). `optimization.esk`'s `adam` is Euclidean and has
  no manifold anywhere in it.
- `lib/bridge/qllm_bridge.cpp` has forward AD nodes for the HYPERBOLIC
  primitives only (`ad_poincare_exp_map`, `ad_poincare_log_map`,
  `ad_hyperbolic_distance`, `ad_frechet_mean`) plus `ad_geodesic_attention`
  and `ad_tensor_cross_entropy`. There is no spherical or Euclidean-manifold
  AD node, and no optimizer.
- Riemannian Adam exists ONLY inside the VM, as `static` functions in
  `lib/backend/vm_geometric.c` reachable through opcodes 839-842 and 860-861.
  It is not callable from C or C++ at all, its retraction is `float32`, and in
  a build without `ESHKOL_GEOMETRIC_ENABLED` it is a plain Euclidean add with
  the curvature popped and discarded.
- The only place all three curvature families appear as one set is the S4
  device lowering, `inc/eshkol/backend/xla/geometric_lowering.h`.

So "call the host's own training step, never re-implement it in the harness"
could not be satisfied by any existing entry point. The root-cause fix is to
give the host the training step it was missing, as a real module with a public
header — `lib/ml/mixed_curvature_step.cpp` and
`inc/eshkol/ml/mixed_curvature_step.h` — and then to have the harness call
that, and the device program mirror it. The harness re-implements nothing.

This is stated plainly because it changes what the parity claim means. The
device and the host are not an old implementation and a new one; they are two
independent implementations of one written definition:

| | host | device |
|---|---|---|
| forward | scalar C++ loops over f64, `lib/ml/mixed_curvature_step.cpp` | StableHLO tensor ops, `lib/backend/xla/training_step_lowering.cpp` |
| backward | **hand-derived analytic** VJPs, written out per stage | **automatic**, `StableHLOEmitter::emitVJP` over the forward graph |
| optimizer | scalar C++ | StableHLO ops in the SAME `func.func @main` |
| execution | host CPU | PJRT device |

The backward passes come from genuinely different derivations — one written by
hand from the chain rule, one produced by a reverse-mode walk of an SSA graph —
so a mismatch between them is real evidence about one of the two. That is what
makes the K-step comparison a test rather than a tautology.

## 1. Where every formula comes from

The forward primitives are transcribed from the S4 device compositions in
`lib/backend/xla/geometric_lowering.cpp`, which are themselves line-by-line
transcriptions of the oracle exporters in `tests/qllm_oracle/` that generated
`tests/qllm_oracle/golden/*.json`. That keeps the whole chain anchored to the
one convention the golden Jacobians grade.

| step in the training step | host function (`lib/ml/mixed_curvature_step.cpp`) | mirrors |
|---|---|---|
| projection `U = X W` | `forwardProjection` | `stablehlo.dot_general`, the S3 matmul at `HIGHEST` precision |
| tangent-to-ball `Z = exp_0^c(alpha U)` | `forwardExpMapOrigin` | `GeometricPrimitive::PoincareExpMapOrigin`, `geometric_lowering.cpp:~230` |
| hyperbolic score `Dh = d_c(Z, Ph)^2` | `forwardHyperbolicSq` | `GeometricPrimitive::HyperbolicDistance`, `geometric_lowering.cpp:~296`, squared |
| sphere normalise `Y = U/\|U\|` | `forwardSphereNormalise` | the `clamp_min` scaling in `GeometricPrimitive::SphereRetract` |
| spherical score `Ds = theta(Y, Ps)^2` | `forwardSphericalSq` | `GeometricPrimitive::SphericalDistance` (`acos` as `atan2`), squared |
| Euclidean score `De = \|U - Pe\|^2` | `forwardEuclideanSq` | `GeometricPrimitive::EuclideanDistance`, squared |
| logits and loss | `forwardLogitsAndLoss` | `ad_tensor_cross_entropy`, `lib/bridge/qllm_bridge.cpp:476`, made a MEAN (see 3.4) |
| Riemannian gradient, hyperbolic | `riemannianGradHyp` | `GeometricPrimitive::PoincareProject` |
| Riemannian gradient, spherical | `riemannianGradSph` | `GeometricPrimitive::SphereProject` |
| Adam moments and delta | `adamDelta` | `vm_riemannian_adam_delta`, `lib/backend/vm_geometric.c:267` |
| hyperbolic retraction | `retractHyperbolic` | `GeometricPrimitive::PoincareExpMap` then `PoincareRetract` |
| spherical retraction | `retractSpherical` | `GeometricPrimitive::SphereExpMap` then `SphereRetract` |
| Euclidean retraction | `retractEuclidean` | `GeometricPrimitive::EuclideanExpMap` (`x + v`) |

## 2. The model

Shapes. `N = batch * seq` rows, `D` model dimension, `C` prototypes (the
classes). The batch is a dense `X [N,D]` of reals and a soft target `T [N,C]`
whose rows sum to one. There is no integer embedding lookup in the step: a
gather would add an index dtype and a scatter-add VJP to the first program
that ever trains on the device, and neither is on trial here.

Parameters, one per manifold so that all three retraction paths are exercised
and the tolerance classes split cleanly:

| parameter | shape | manifold | tolerance class in the harness |
|---|---|---|---|
| `W` | `[D,D]` | Euclidean | arithmetic |
| `P_hyp` | `[C,D]` | Poincare ball, curvature `c` | transcendental |
| `P_sph` | `[C,D]` | unit sphere | transcendental |
| `P_euc` | `[C,D]` | Euclidean | arithmetic |

The two manifold parameters take the transcendental class because their
retractions pass through `tanh`, `atanh`, `sin`, `cos` and `atan2`; the two
Euclidean ones are reached only by exact arithmetic and keep the tighter
bound. This is the classification rule already stated in
`tests/xla/parity_compare.h`: by what the operation IS, not by what it
measured today.

Optimizer state: `m` and `v` shaped like each parameter, plus one shared
integer step count. Eight moment tensors in, eight out.

## 3. The step, exactly

Hyperparameters: curvature `c > 0`, tangent scale `alpha`, `lr`, `beta1`,
`beta2`, Adam `eps`, guard `eps_g`, and the three score weights
`w_hyp, w_sph, w_euc`.

### 3.1 Forward

```
U[n]      = sum_j X[n,j] W[j,:]                                     [N,D]
v[n]      = alpha * U[n]
t[n]      = sqrt(c) * |v[n]|
Z[n]      = |v[n]| < 1e-10 ? v[n] : v[n] * tanh(t[n]) / t[n]        [N,D]

q[n,k]    = |Z[n] - P_hyp[k]|^2
ax[n]     = 1 - c |Z[n]|^2
bx[k]     = 1 - c |P_hyp[k]|^2
A[n,k]    = max_floor(1 + 2 c q / (ax bx), 1 + 1e-12)
Dh[n,k]   = ( log(A + sqrt(A^2 - 1)) / sqrt(c) )^2                  [N,C]

Y[n]      = U[n] / clamp_min(|U[n]|, eps_g)                         [N,D]
s[n,k]    = clamp_unit(<Y[n], P_sph[k]>)          (clamp at +-(1 - 1e-7))
th[n,k]   = atan2( sqrt(1 - s^2), s )
Ds[n,k]   = th^2                                                    [N,C]

De[n,k]   = |U[n] - P_euc[k]|^2                                     [N,C]

L[n,k]    = -( w_hyp Dh + w_sph Ds + w_euc De )                     [N,C]
loss      = (1/N) sum_n ( logsumexp_k L[n,k] - sum_k T[n,k] L[n,k] )
```

Three deliberate departures from the S4 primitives, each because the SQUARE of
a distance is what is being differentiated rather than the distance:

1. `A` is floored at `1 + 1e-12`, where the S4 `HyperbolicDistance` floors at
   exactly `1`. `d^2`'s derivative carries `acosh(A)/sqrt(A^2-1)`, whose limit
   at `A = 1` is finite (it tends to 1) but whose numerator and denominator
   both vanish there, so evaluating it at `A = 1` is `0/0` and yields NaN on
   both sides. The floor bounds `sqrt(A^2-1)` below by `1.4e-6` and is applied
   identically in the host and in the device module, so it is part of the
   operator, not a device workaround.
2. The cosine clamp at `1 - 1e-7` is the oracle's own (`clampUnit` in
   `geometric_lowering.cpp`) and is likewise inside the operator. Above it
   `theta` is constant and its exact derivative is zero, on both sides.
3. The sphere normaliser is a `clamp_min` divide with no `select`, unlike
   `SphereRetract`, which returns `x` unchanged below the guard. There is no
   "unchanged" answer here — the input is an activation, not a point already
   on the manifold — so the guarded divide is the whole operator.

### 3.2 Backward

`dL/dL[n,k] = (softmax(L)[n,:] - T[n,:]) / N`, which is the gradient of the
mean soft-target cross entropy and matches `tensor_cross_entropy_backward`
(`lib/bridge/tensor_backward.cpp:408`) up to the `1/N`.

From there the host propagates analytically, stage by stage, in
`backwardEuclideanSq`, `backwardSphericalSq`, `backwardSphereNormalise`,
`backwardHyperbolicSq`, `backwardExpMapOrigin`, `backwardProjection`. Each is
derived in a comment above its definition. The device gets the same gradients
from `emitVJP` over the forward graph, with no rule written for this step at
all. `X` receives no gradient (it is data); `W`, `P_hyp`, `P_sph`, `P_euc` do.

### 3.3 Riemannian Adam and the retraction

Per parameter `x` with Euclidean gradient `g`:

```
rg = g                                             (Euclidean)
rg = 0.25 * clamp_min(1 - c|x|^2, eps_g)^2 * g     (hyperbolic, per row)
rg = g - <g, x> x                                  (spherical, per row)

m  = beta1 m + (1 - beta1) rg
v  = beta2 v + (1 - beta2) rg*rg
step += 1
delta = -lr * (m / (1 - beta1^step)) / ( sqrt(v / (1 - beta2^step)) + eps )
```

exactly as `vm_riemannian_adam_delta` writes it, including `eps` OUTSIDE the
square root, the bias corrections as `1 - beta^step`, and the minus sign
carried in `delta` rather than in the retraction. The step count enters the
device program as a rank-0 operand and the corrections are computed there with
`stablehlo.power`, so one executable serves every step index.

Retraction, which is where this departs from the VM and says so:

```
x' = x + delta                                                  (Euclidean)
x' = radial_clip( exp^c_x(delta) )                              (hyperbolic)
x' = normalise( exp_x(delta) )                                  (spherical)
```

`exp^c_x` is the Ganea formula of `GeometricPrimitive::PoincareExpMap` and
`radial_clip` is `GeometricPrimitive::PoincareRetract` with a zero step;
`exp_x` and `normalise` are `SphereExpMap` and `SphereRetract` with a zero
step. The VM's optimizer instead applies the HYPERBOLIC exp map to every
parameter regardless of manifold, in float32, and in a portable build applies
no map at all. That is not a definition a mixed-curvature model can train
under — a spherical parameter moved by a Poincare exp map leaves the sphere on
the first step — so this step uses the per-manifold retraction, which is also
what the oracle's own `poincare_retract` / `sphere_retract` exist to do. The
VM's behaviour is left alone; nothing here changes it.

### 3.4 The loss is a mean, not a sum

`ad_tensor_cross_entropy` sums over the batch without dividing. A sum makes
the gradient scale with `N`, so the two batch shapes in the harness would need
two learning rates to take comparable steps and the loss trajectories would not
be comparable at all. The step divides by `N`. This is a difference from that
host node and is recorded here rather than absorbed.

## 4. The device program

ONE `func.func @main` in one StableHLO module: forward, backward and optimizer
update, no host round trip inside the step.

- Inputs, in order: `W`, `P_hyp`, `P_sph`, `P_euc`, `mW, vW, mH, vH, mS, vS,
  mE, vE`, `X [N,D]`, `T [N,C]`, then the rank-0 scalars `c, alpha, lr, beta1,
  beta2, adam_eps, eps_g, step, w_hyp, w_sph, w_euc`.
- Outputs, in order: the four updated parameters, the eight updated moments,
  and the rank-0 `loss`. Thirteen results.
- Curvature, the learning rate and the step count are OPERANDS, for the reason
  `geometric_lowering.h` already gives: a mixed-curvature model learns its
  curvature, and a constant in a shape-keyed executable cache is a wrong number
  with nothing to see.
- Cache key: `train|N|D|C|dtype`. Two curvatures at one shape therefore share
  one executable, and every step after the first at a given shape is a cache
  hit. The harness counts them.

## 5. What the harness grades

`tests/xla/training_step_parity_test.cpp`, per shape and per curvature, from
identical initial parameters and an identical batch, K = 5 steps, in TWO row
families that answer two different questions:

**`step` rows.** The host is stepped from the DEVICE's own current state, so
both sides consume identical inputs and one application of the step operator is
what is compared. These keep the per-op tolerance classes of
`docs/design/ESHKOL_S_FRAGMENT.md` unchanged, and they are graded at five
different points along a real trajectory with real accumulating moments.

**`traj` rows.** A second host model runs free and is never re-seeded, so the
two trajectories evolve independently for K steps and an error at step 1
compounds through the moments.

The `traj` rows cannot be graded at the per-op bound, and the reason is a
property of Adam rather than of the lowering. Adam's delta is
`-lr m_hat/(sqrt(v_hat) + eps)`: a NORMALISED step whose magnitude is about
`lr` whatever the gradient's magnitude, so a relative error in the gradient
survives into the delta essentially undamped, and at most doubled (numerator
and denominator each carry it). After k steps the two parameter sets can
therefore differ by

    2 * k * lr * (relative accuracy of the gradient)

and, since every gradient in this model flows through `acosh`, `atan2` and
`tanh`, that accuracy is the transcendental class whichever parameter the
gradient lands on. That expression, with nothing fitted to a measurement, is
the `traj` bound; the harness prints it with every row. Grading the compounding
rows at the per-op bound instead would not be stricter, it would be wrong: it
would demand that five f32 optimizer steps land where five f64 ones did.

Both families grade:

- the loss after every step;
- all four parameter tensors after every step (in the `step` family:
  arithmetic for `W` and `P_euc`, transcendental for `P_hyp` and `P_sph`);
- all eight moment tensors after every step (same classes as their parameter);

and, once per configuration:

- the loss trajectory's direction over K steps on a fixed batch, required to
  agree step for step between device and host, to contain the same number of
  downhill steps, and to be net downhill on both. A decrease at EVERY step is
  deliberately not required: a normalised Adam step has magnitude about `lr`
  regardless of the gradient, so a coordinate near its minimum is stepped past
  by a fixed distance and one step can raise the loss. That is a property of
  the optimizer being mirrored; the per-step counts are printed so that the two
  sides overstepping in the same places stays visible;
- the manifold constraints on the DEVICE outputs alone: `c |P_hyp[k]|^2 <= 1 -
  eps_g` with margin, and `| |P_sph[k]| - 1 | <= 1e-6`;
- a negative control that perturbs one expected parameter after step 1 and
  must be reported FAIL.

Initial parameters come from `eshkol_mixed_curvature_init`, a deterministic
generator in the host module, so that "identical initial parameters" is a
property of one function both sides call rather than of two copied loops.
