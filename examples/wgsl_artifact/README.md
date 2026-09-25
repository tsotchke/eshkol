# Bounded strain energy artifact

`generate.esk` is an ordinary Eshkol program. It derives an order 10 Taylor
model of `W(e)=(a/b)(exp(b e)-1-b e)` with `a=2 Pa`, `b=3`, and a single
independent dimensionless strain `e`. Both parameters stay fixed. The
reference is `e=0`; normalized `t=(e-0)/1` lies in `[-0.2,0.2]`. Coefficients
are in ascending powers of `t`. The first and second derivative coefficient
arrays are shifted from the **same** Taylor array. Value, first derivative,
and second derivative have units Pa, Pa/strain, and Pa/strain².

Build Eshkol, then run `python3 examples/wgsl_artifact/build_artifact.py`.
The script runs the source on native and VM, checks byte equality and a 41
point AD comparison, then writes `evaluator.wgsl` and `manifest.json`.
Run `node examples/wgsl_artifact/demo/run.mjs` to validate WGSL and execute
the three-kernel chain in Chrome. The browser writes `demo/evidence.json`.

`manifest.json` defines the byte layout and all status values. The caller
provides one device and owns the buffers and command stream. The bound is
4096 samples; the caller must enforce that limit before dispatch. The WGSL
also bounds each invocation against `count` and both runtime array lengths.
An outside-envelope or nonfinite input produces status 1 and zero numeric
outputs. An evaluation overflow produces status 2. Successful values have
status 0 and envelope flag 1. The generation field lets a consumer reject a
stale set. This artifact fixes coefficients in the WGSL source; an online
refit produces a new WGSL/manifest pair with a new generation and swaps the
whole prepared pipeline set at a caller-controlled boundary.

The manifest separates the 41 point AD observation, analytic Taylor remainder
bound, and coefficient conversion bound. `demo/evidence.json` reports measured
GPU errors against the analytic law and warm evaluator and three-kernel
submission costs. These include queue and synchronization overhead; they are
not kernel-only timestamps. The measurements apply to the recorded adapter
and f32 path.
