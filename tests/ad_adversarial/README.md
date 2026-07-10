# Generative adversarial AD oracle (pillar P3-gen)

A CONTINUOUS, GENERATIVE exposure engine for Eshkol's automatic
differentiation. Where `tests/ad_oracle/` enumerates a FIXED composition
matrix, this harness *grows* random-but-seeded differentiable programs out of
the AD-supported primitives and checks every gradient/Jacobian/Hessian against
an in-language CENTRAL finite difference. Its job is to keep finding NEW
silent-wrong gradients, not to regression-guard known repros.

> "If our system does not constantly expose every single hidden bug then it has
> no coverage." AD is Eshkol's crown jewel and silent-wrong gradients are its
> most dangerous defect class. A zero AD gradient where the finite difference is
> non-zero is a hard FAIL here.

## What it generates

The generator (`gen_ad_adversarial.py`) is deterministic: every random choice
comes from `random.Random(seed)` over a FIXED seed list, so regenerating
reproduces the corpus byte-for-byte and CI is stable even though the
compiler/runtime has no RNG. Each generated `.esk` file self-checks and prints
`PASS:`/`FAIL:`/`XKNOWN:` lines plus a `Passed:/Failed:/Xknown:` summary.

Primitives composed: `+ - * / exp log sin cos tanh sqrt pow` and the tensor/ML
operators `tensor-sum tensor-mean tensor-dot tensor-add tensor-mul tensor-scale
tensor-matmul conv2d batch-norm layer-norm scaled-dot-attention softmax`.
Domain-sensitive nodes (`log`/`sqrt`/`pow`/division) are wrapped so their
argument stays strictly positive and away from zero near the evaluation point,
which keeps the central difference an accurate ground truth.

| family    | what it exposes |
|-----------|-----------------|
| `scalar`  | random scalar expression trees -> 1st derivative AND 2nd derivative (derivative-of-derivative) vs FD |
| `field`   | random `R^n -> R` fields -> gradient (per component), laplacian, at BOTH vector and **tensor-literal** (`(tensor …)`) evaluation points |
| `gofg`    | vector-parameter **gradient-of-gradient** (the ESH-0096 silent-zero shape) at 1- and multi-component points |
| `tensor`  | random ML-op loss compositions -> gradient of a flattened operand vs FD, across **literal / first-class / higher-order-wrapper** loss forms, **first AND second** operand, and **scalar AND per-feature (vector) gamma** for batch/layer-norm |
| `htensor` | **higher-order over tensor ops**: full Hessian of a tensor loss at vector and tensor-literal points (the ESH-0095 SIGSEGV shape) |

The `tensor` family folds every op result against a random weight tensor before
reducing to a scalar, so the true gradient is generically non-zero and a silent
`#(0 0 …)` is caught. Every `tensor` loss is exercised as (1) a literal lambda
at the call site, (2) a first-class value bound to a variable, and (3) a value
threaded through a higher-order wrapper — the three forms whose divergence was
the ESH-0212 silent-zero regression.

Current corpus: **21 files, 147 probes, 436 component checks**, run under BOTH
the JIT (`-r`) and AOT.

The `vecpoint` family is a dedicated tracked repro of **ESH-0235** (found by
this harness): tensor-op losses differentiated at a `(vector …)`-constructed
point, which the tensor reverse path silently zeroes. It is marked XKNOWN so
the gate stays green-able and flips to PASS automatically once fixed.

## Running

```
scripts/run_ad_adversarial.sh            # full sweep, JIT + AOT
scripts/run_ad_adversarial.sh --quick    # one file per family, JIT only (CI)
scripts/run_ad_adversarial.sh --no-aot   # JIT lane only
scripts/run_ad_adversarial.sh --regen    # re-run the generator first
```

Verdicts per file+mode: `PASS` / `FAIL` (an untracked cell diverged from finite
differences — a NEW, actionable defect) / `XKNOWN` (a tracked open bug) /
`CRASH` / `HANG`. The gate is green iff there are no FAIL/CRASH/HANG. The runner
emits pytest-style `PASSED/FAILED/XFAIL` lines and ICC JSON-L events
(`kind:"ad_adversarial"`) into `scripts/icc_traces/ad_adversarial.jsonl`; a
`--quick` run is also driven by the `ad_adversarial_fd_oracle` probe in
`scripts/run_icc_smoke.sh`, whose PASS/FAIL is consumed by
`.icc/completion-oracles.yaml`. Readiness therefore CONTINUOUSLY asserts "AD
matches FD across a generated family," not a narrow smoke.

## Triaging a FAIL

A FAIL prints the exact probe id, the full AD gradient vector, and the finite
difference it diverged from, so the failing expression is reproducible from the
named `.esk` file. Decide generator error vs real AD bug. A real bug gets:

1. a shrunk repro in `found/`,
2. an ESH task in `.swarm/tasks/`,
3. an entry in the generator's `XKNOWN` (non-crash) or `KNOWN_CRASHERS`
   (SIGSEGV / IR-verify) table, keyed by an id-prefix and referencing the task.

`XKNOWN` cells print `XKNOWN:` and are tolerated by the gate; crash cells are
emitted one-per-file as `ad_adv_xc_<task>_<NN>.esk` so a crasher can stay
tracked without turning the whole harness red. Both flip to PASS automatically
once the compiler is fixed — then delete the table entry so a future regression
FAILs loudly instead of hiding as XKNOWN.

## Status on master (2026-07-10)

Green-able: every check either matches finite differences or is a tracked
XKNOWN. The scalar, field, gofg, tensor and htensor families (all 427
non-vecpoint checks) match FD under JIT and AOT.

- **NEW defect found: ESH-0235** — reverse-mode AD through a tensor op returns a
  silently-WRONG all-zero gradient when the differentiation point is built with
  the `(vector …)` constructor. The mathematically identical point as a `#(…)`
  reader literal or a `(tensor …)` value gives the correct gradient. Reproduces
  under BOTH `-r` and AOT. Captured by the `vecpoint` family (XKNOWN) and
  `found/esh0235_tensor_grad_vector_ctor_point_zero.esk`. This is the worst AD
  class (silent-wrong training gradients) and was invisible to the existing
  tensor-AD unit tests, which only ever seed with a `#(…)`/`(tensor …)` point.
- The two flagship silent-wrong shapes this family also targets — **ESH-0096**
  (vector gradient-of-gradient) and **ESH-0095** (Hessian/Laplacian at a
  tensor-literal point) — are confirmed FIXED on master (the `ad_oracle`
  README's "known-open" table for them is stale).
- The sweep additionally surfaced an intermittent SIGSEGV in the multi-threaded
  JIT compile path on AD/tensor-heavy modules (a race, not a wrong gradient);
  see `found/NOTES.md`. The runner pins the JIT lane to a single compile thread
  so the gate reliably tests gradient correctness.
