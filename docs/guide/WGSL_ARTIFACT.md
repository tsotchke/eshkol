---
kind: guide
status: current
owner-area: gpu
since: v1.3.5
sources:
  - examples/wgsl_artifact/generate.esk
  - examples/wgsl_artifact/build_artifact.py
  - examples/wgsl_artifact/demo/run.mjs
---

# WGSL artifact from Eshkol AD

The [bounded strain energy example](../../examples/wgsl_artifact/README.md)
turns a real-variable Eshkol Taylor model into standalone WGSL. Build Eshkol,
run `python3 examples/wgsl_artifact/build_artifact.py`, then run
`node examples/wgsl_artifact/demo/run.mjs` with Chrome and Playwright installed.
The demo uses `playwright`'s Chrome channel and `--enable-unsafe-webgpu`, serves
the WGSL on localhost, and requires a real WebGPU adapter.

`generate.esk` uses `taylor` at order 10 and checks `derivative` and
`derivative-n` order 2 on a 41 point grid. The builder requires native and VM
output to match exactly. The versioned manifest records the model, physical
units, reference/normalization, coefficient order, compiler and source hashes,
binding layouts, dispatch limit, status schema, and error bounds. Regenerate
the entire WGSL/manifest pair when changing coefficients. Do not edit the
embedded numeric literals in place.

The browser chain is producer → evaluator → consumer. All three pipelines use
one caller-owned `GPUDevice`. The producer writes resident f32 samples, the
evaluator reads those samples and writes 32-byte results, and the consumer
compacts each result to 16 bytes. One command encoder and one queue submission
perform the demonstrated chain; only its final compact buffer is read back.
The demo separately benchmarks warm submissions after pipeline compilation.

Error categories are distinct. Native AD grid disagreement estimates the
coefficient derivation path. The exponential Taylor remainder is bounded
analytically on `|e|≤0.2`; this is the approximation error. Each coefficient
is rounded to f32 when WGSL is compiled, with a separate weighted coefficient
conversion bound. The Chrome demo reports the observed difference between GPU
outputs and separate-operation f32 Horner, plus total deployed error against
the analytic law. The former is an empirical rounding-path comparison, not a
worst-case bound; it may vary by adapter and contraction behavior.
The deployed absolute tolerances are 2e-6 Pa for value, 1e-5 Pa/strain for
first derivative, and 5e-5 Pa/strain² for second derivative. They apply only
inside the declared envelope with finite input and the fixed parameters.
