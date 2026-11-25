# Add CSR Krylov solves with exact implicit sensitivity enclosures

CG and full-orthogonalization symmetric minimum residual over existing CSR, plus explicit implicit JVP/VJP and exact diagonal-dominance enclosures of original-model solution and cotangent errors.

Base: `41c8aa2d4e8f966f7cebdca86779ac62eb291943` (`feat/v15-neurosymbolic`). This is one independently reviewable cut.

## Run

```sh
python3 repro/sparse-solves/self_test.py
python3 repro/sparse-solves/gate.py --eshkol-bin /path/to/eshkol-run  --results repro/sparse-solves/results.json
```

The gate forces `ESHKOL_JIT_CACHE=0` in every native child, even when its caller
sets it to 1. It requires complete expected output and uncached in-process JIT /
AOT-O0 equality, freezes the tested sources for the duration of the run, and
executes producer-free replay with disabled producers exercised as negative
controls. Boundary self-tests launch no native compiler or workload.

The native three-state consumer passes nonzero-radius source/sensitivity and refusal controls. Independent rational dense solves check 33 derivative cases. Larger consumers use a 32x32 screened-Poisson stencil and a symmetric indefinite signed-diagonal variant, each with 1,024 variables and 4,992 stored entries.

## Check received witnesses

Run `python3 repro/sparse-solves/oracle.py repro/sparse-solves/results.json` to replay saved solution and sensitivity enclosures without a native solver. The independent checker uses Python Fractions; small cases also compare to exact dense Gaussian elimination.

## Limits

CG requires SPD; a negative-curvature guard is not a complete SPD test. MINRES uses a full orthogonal basis with O(n*k+k^2) storage. Exact certificates require strict row diagonal dominance and exact rational model inputs; unadmitted arithmetic, nonconvergence or missing bounds produce unknown with no gradient artifact. Explicit JVP/VJP APIs are supplied; no new automatic tape primitive or performance speedup is claimed.

Qualification uses the published v1.3.6-evolve Linux x64 lite compiler/runtime in
Ubuntu 22.04/gcc-11. These are native library-level qualifications against pinned
feature-branch source interfaces. No new compiler build, full upstream CI/CTest,
VM/WASM, Windows, GPU or hardware qualification is implied. The small standalone
CTest project checks this cut's registration and boundary readiness separately.
