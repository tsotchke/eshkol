# Eshkol Examples

Fourteen runnable programs spanning the language's load-bearing capabilities — automatic differentiation, parallel work-stealing, symbolic computation, the consciousness engine, real-time streaming, and physical simulation. Every example in this directory compiles cleanly, runs in well under a minute, and produces output you can actually inspect.

```bash
# Compile and run any example in one command:
./build/eshkol-run examples/<name>.esk -o /tmp/eshkol-<name>
/tmp/eshkol-<name>
```

Or use the AOT compiler with no separate run step:

```bash
./build/eshkol-run examples/<name>.esk
```

---

## Start here

| Example | What it shows | LOC |
|---------|--------------|----:|
| **[hello.esk](hello.esk)** | The simplest possible Eshkol program | 4 |
| **[autodiff.esk](autodiff.esk)** | Forward- and reverse-mode AD in 16 lines | 20 |
| **[tensors.esk](tensors.esk)** | Matrix creation, matmul, GPU dispatch (Metal / CUDA / SIMD) | 30 |
| **[consciousness.esk](consciousness.esk)** | The 22-builtin neuro-symbolic surface: KB + factor graph + workspace | 24 |

## Automatic differentiation

The compiler differentiates through arbitrary Eshkol code — no framework, no graph object, no Python.

| Example | What it shows | LOC |
|---------|--------------|----:|
| **[gradient_descent_demo.esk](gradient_descent_demo.esk)** | Train a 3-parameter quadratic on noisy data; loss falls from 63 → 0.06 in 400 steps | 93 |
| **[newton_method.esk](newton_method.esk)** | Newton's root finder in 15 lines using `derivative`. Cube root, square root, the Dottie number, and a quintic root — each converges in 4–5 iterations | 70 |
| **[symbolic_diff.esk](symbolic_diff.esk)** | Three modes of AD (`diff` / `derivative` / `gradient`) agree to machine precision on `sin(x²) + 3x`. The symbolic mode prints its rewritten AST | 87 |
| **[differentiable_physics.esk](differentiable_physics.esk)** | Optimise a projectile launch angle by differentiating *through* a recursive Euler integrator with linear drag. Converges to the high-arc 68° solution in 200 steps with final error 1.3e-8 | 95 |
| **[neural_xor.esk](neural_xor.esk)** | Two-layer MLP (2 → 4 hidden tanh → 1 sigmoid) learns XOR by full-batch gradient descent in 1,500 epochs. Loss 1.10 → <0.001 | 123 |
| **[h2_vibrational.esk](h2_vibrational.esk)** | Molecular vibrational frequency from *arbitrary-order* AD: build an STO-3G H₂ potential-energy surface in pure Eshkol, take the force constant k = d²E/dR² with `derivative-n`, and recover ω = 5003 cm⁻¹ — matching the reference value | 121 |

## Parallelism and performance

Work-stealing thread pool, Chase-Lev deques, per-worker arenas, no GC pauses.

| Example | What it shows | LOC |
|---------|--------------|----:|
| **[parallel.esk](parallel.esk)** | `parallel-execute` with three concurrent thunks across 24 workers | 10 |
| **[monte_carlo_pi.esk](monte_carlo_pi.esk)** | Estimate π by parallel Monte Carlo: 1.6M samples across 8 independent PRNG streams in ~45 ms (≈35M samples/sec on M2 Ultra) | 63 |
| **[streaming_stats.esk](streaming_stats.esk)** | Welford's online algorithm for running mean, variance, min, max. 200,000 samples processed without ever storing the stream | 90 |

## Cognitive computing

The consciousness engine: logic programming, active inference, global workspace.

| Example | What it shows | LOC |
|---------|--------------|----:|
| **[bayesian_diagnosis.esk](bayesian_diagnosis.esk)** | Medical triage agent that combines a symbolic KB with a 3-variable factor graph; tracks free energy across three observation regimes | 110 |

## Scientific computing

Exact arithmetic, the numeric tower, category-theoretic models.

| Example | What it shows | LOC |
|---------|--------------|----:|
| **[milli_mag_bohrification.esk](milli_mag_bohrification.esk)** | CODATA physical-constants demonstration: ten PASS assertions covering Bohrification of the milli-magnetic model, K-homology pairing, projection round-trip | 60 |

---

## What makes these examples interesting

**No black-box framework.** Every gradient, every parallel dispatch, every probabilistic-graph belief is a compiler primitive. `gradient`, `parallel-map`, `fg-infer!` are not library function calls into a runtime VM — they lower directly to LLVM IR.

**Each program tells one story.** A neural network learns XOR. A projectile finds its angle. A medical agent updates its beliefs. The examples are sized to fit on one screen and they all run in under a minute.

**Reproducible.** Every PRNG in this directory uses a fixed seed. Run the same example on two different machines and you get bit-identical output. (`monte_carlo_pi.esk` and `neural_xor.esk` both have this property; see their PRNG-seed comments.)

**Reflects v1.2.1-scale reality.** Every API call here matches the actual production codegen signatures. The `(gradient fn vector)` arity, the `(cons salience proposal)` workspace closure contract, the `(make-factor-graph num-vars dims-tensor)` factor-graph signature — all verified against `lib/backend/` source.

---

## Beyond this directory

| Where | What |
|---|---|
| **[`docs/tutorials/`](../docs/tutorials/)** | 29 step-by-step tutorials — start with `00_FIRST_5_MINUTES.md`, then `01_AUTODIFF_AND_ML.md`, then pick by interest |
| **[`docs/breakdown/`](../docs/breakdown/)** | 36 per-subsystem deep dives — read `AUTODIFF.md` for the AD architecture, `CONSCIOUSNESS_ENGINE.md` for the neuro-symbolic stack |
| **[`docs/API_REFERENCE.md`](../docs/API_REFERENCE.md)** | 336 documented procedures with signatures and examples |
| **[`docs/SDNC.md`](../docs/SDNC.md)** | The Self-Differentiating Neural Computer paper artefact — a constructive proof that a six-layer transformer can be an interpreter |
| **[eshkol.ai](https://eshkol.ai)** | Browser REPL with a 64-opcode VM and 555+ built-in functions — no installation required |

## Try it in the browser

Visit **[eshkol.ai](https://eshkol.ai)** to run Eshkol without installing anything. The website itself is a 1,500-line Eshkol program compiled to a 220,306-byte (about 215 KiB) WebAssembly binary.
