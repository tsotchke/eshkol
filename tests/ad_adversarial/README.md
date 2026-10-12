# Generative adversarial AD oracle (pillar P3-gen)

A CONTINUOUS, GENERATIVE exposure engine for Eshkol's automatic
differentiation. Where `tests/ad_oracle/` enumerates a FIXED composition
matrix, this harness *grows* random-but-seeded differentiable programs out of
the AD-supported primitives and checks every gradient/Jacobian/Hessian against
an in-language CENTRAL finite difference. Its job is to keep exposing NEW
gradient divergences, not to regression-guard known repros.

> "If our system does not constantly expose every hidden divergence then it has
> no coverage." AD is Eshkol's crown jewel, and a gradient that silently
> disagrees with the function is the most consequential failure class it can
> have. A zero AD gradient where the finite difference is non-zero is a hard
> FAIL here.

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
| `gofg`    | vector-parameter **gradient-of-gradient** (a shape that once returned a silent zero) at 1- and multi-component points |
| `tensor`  | random ML-op loss compositions -> gradient of a flattened operand vs FD, across **literal / first-class / higher-order-wrapper** loss forms, **first AND second** operand, and **scalar AND per-feature (vector) gamma** for batch/layer-norm |
| `vecpoint` | tensor-op losses differentiated at a `(vector …)`-constructed point, next to the same point as a `#(…)` literal or a `(tensor …)` value |
| `exactpoint` | AD evaluated at an **exact** point (rational, bignum, small integer) and checked against FD at `(exact->inexact <same point>)`, for `derivative`, `gradient` and the variable-bound forms |
| `htensor` | **higher-order over tensor ops**: full Hessian of a tensor loss at vector and tensor-literal points, including `tensor-matmul` over `reshape` |

The `tensor` family folds every op result against a random weight tensor before
reducing to a scalar, so the true gradient is generically non-zero and a silent
`#(0 0 …)` is caught. Every `tensor` loss is exercised as (1) a literal lambda
at the call site, (2) a first-class value bound to a variable, and (3) a value
threaded through a higher-order wrapper — three forms that once diverged from
one another with a silent zero.

Current corpus (v1.3.6-evolve, `generated/MANIFEST.txt`): **32 files, 189
probes, 674 component checks**, run under BOTH the JIT (`-r`) and AOT, so a full
sweep grades 64 file×mode cells.

The `vecpoint` family was added when this harness found that a tensor-op loss
differentiated at a `(vector …)`-constructed point returned an all-zero
gradient. That path now matches the `#(…)` and `(tensor …)` points, and the
family is an ordinary gated family. The `exactpoint` family pins the AD entry
points' coercion of exact points: an exact rational or bignum is heap-tagged,
so the seed must be read as the number it denotes, never as its address.

## Running

```
scripts/run_ad_adversarial.sh            # full sweep, JIT + AOT
scripts/run_ad_adversarial.sh --quick    # one file per family, JIT only (CI)
scripts/run_ad_adversarial.sh --no-aot   # JIT lane only
scripts/run_ad_adversarial.sh --regen    # re-run the generator first
```

Verdicts per file+mode: `PASS` / `FAIL` (an untracked cell diverged from finite
differences — a NEW, actionable divergence) / `XKNOWN` (a tracked open
divergence) / `CRASH` / `HANG`. The gate is green iff there are no FAIL/CRASH/HANG. The runner
emits pytest-style `PASSED/FAILED/XFAIL` lines and ICC JSON-L events
(`kind:"ad_adversarial"`) into `scripts/icc_traces/ad_adversarial.jsonl`; a
`--quick` run is also driven by the `ad_adversarial_fd_oracle` probe in
`scripts/run_icc_smoke.sh`, whose PASS/FAIL is consumed by
`.icc/completion-oracles.yaml`. Readiness therefore CONTINUOUSLY asserts "AD
matches FD across a generated family," not a narrow smoke.

## Triaging a FAIL

A FAIL prints the exact probe id, the full AD gradient vector, and the finite
difference it diverged from, so the failing expression is reproducible from the
named `.esk` file. Decide generator error vs real AD divergence. A real
divergence gets:

1. a shrunk repro in `found/`,
2. a tracking entry,
3. an entry in the generator's `XKNOWN` (value divergence) or `KNOWN_CRASHERS`
   (fatal signal / IR-verify) table, keyed by an id-prefix and referencing the
   tracking entry. At v1.3.6-evolve both tables are empty.

`XKNOWN` cells print `XKNOWN:` and are tolerated by the gate; fatal-signal
cells are emitted one-per-file as `ad_adv_xc_<task>_<NN>.esk` so such a cell
can stay tracked without turning the whole harness red. Both flip to PASS automatically
once the compiler is fixed — then delete the table entry so a future regression
FAILs loudly instead of hiding as XKNOWN.

## Status at v1.3.6-evolve

The `XKNOWN` and `KNOWN_CRASHERS` tables in `gen_ad_adversarial.py` are both
empty: no cell is tolerated, so every check in every family must match finite
differences under JIT and AOT for the gate to be green.

- The `(vector …)`-point case this harness found on 2026-07-10 now returns the
  same gradient as the `#(…)` and `(tensor …)` points under `-r` and AOT;
  `found/esh0235_tensor_grad_vector_ctor_point_zero.esk` prints `#(2 -1 0.5)`
  for all three point spellings.
- The two shapes the `gofg` and `htensor` families were built around —
  vector-parameter gradient-of-gradient, and Hessian/Laplacian at a
  tensor-literal point — match finite differences.
- The sweep surfaced an intermittent fault in the multi-threaded JIT compile
  path on AD/tensor-heavy modules (a compile-time race, not a gradient value);
  see `found/NOTES.md`. The runner pins the JIT lane to a single compile thread
  (`ESHKOL_JIT_COMPILE_THREADS=1`) so the gate tests gradient values reliably.

Set `ESHKOL_DURABLE_WORK_ROOT` to keep the per-run JIT cache and traces under a
durable directory instead of a temporary one.
