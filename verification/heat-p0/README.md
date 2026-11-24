# P0 — Verified heat-stencil execution

P0 lowers one step of the one-dimensional heat equation on the unit rod to a straight-line Eshkol CPU kernel. The endpoints stay at zero, diffusivity is one, the grid spacing is `h = 1/N`, and the time step is `k = h²/4`. The certified profile starts from samples of `sin(pi*x)` and ends at `T = 1/50`. The public replay accepts grid sizes divisible by five so the final time is exactly reachable. The impulse control uses four intervals and one step.

## Replay

Use Python 3, Lean 4.35.0, and an installed native Eshkol compiler. The witness was exercised with Eshkol v1.3.6-evolve. Set `LEAN` and `ESHKOL_RUN` if the executables are outside the search path. From this directory run

```sh
python3 replay.py --intervals 20 40 --output profile.json
```

The JSON contains the exact rational temperatures at all `N+1` points, the per-step defect, initial sampling error, final maximum-norm bound, and digests of the source and binary actually used. The double-precision observed error is a display statistic. The replay checks the certificate using rational arithmetic. Compilation and native execution take place in a temporary directory.

## Schedule and evidence

[The generator](solver.py) emits a fresh zero-filled vector and one store per interior point. [The replay](replay.py) decodes the emitted kernel's actual allocation, addresses, weights, evaluation order and returned vector. The decoder admits only the displayed straight-line fragment. The presence of an unrecognized statement fails admission.

[The Lean checker](Heat.lean) tracks the origin of each physical cell separately from its numeric value. It also symbolically executes the stores and compares each returned expression with an independently specified Dirichlet FTCS target. Its `admitted_refines` theorem says that equality of the symbolic outputs implies equality at every grid point for **all rational interior input fields** with zero endpoints. Each positive case instantiates this theorem by kernel reduction against the decoded schedule. The native solver runs on Eshkol's exact arithmetic. A separate simultaneous rational implementation checks all returned cells at the requested time. The native-output comparison covers those runs. The Lean theorem covers the decoded model, with the small source decoder and the interpretation of the admitted Eshkol forms as trust assumptions. General LLVM refinement and floating-point kernels are outside this fragment.

Aliasing the fresh vector to the input changes one allocation expression. Cell 1 writes a new value to the old storage. Cell 2 then reads cell 1 at its new time level. The gate reports `REJECT cell 2: expected old[1], observed new[1]` before assigning a profile certificate. An isolated native replay of both programs on the exact impulse returns interior values `(1/4, 1/2, 1/4)` and `(1/4, 9/16, 9/64)`. Both remain in the initial temperature range. Two other controls change the middle coefficient and return the wrong buffer.

## From grid values to the model solution

For `u(x,t) = exp(-pi²t) sin(pi*x)`, both the second time derivative and the fourth space derivative have absolute value at most `pi^4`. Taylor's theorem bounds the one-step defect by

```text
d = (22/7)^4 * (k^2/2 + k*h^2/12).
```

At every interior cell the weights are `(1/4, 1/2, 1/4)`. Their convex combination is nonexpansive in the maximum norm, with exact zero endpoints. After `s` steps the accumulated error is therefore at most `initial_error + s*d`. The native profile must equal the exact rational reference field before its backend error can be reported as zero.

The initial sampler derives a rational enclosure of pi from Machin's identity and alternating arctangent sums. It reflects about the rod midpoint, bounds the sine Taylor remainder, rounds each sample to a dyadic rational and carries the resulting sampling error into the certificate. An independent final-time calculation uses a rational pi bracket, evaluates sine across the full interval and encloses the exponential decay. It checks the claimed bound point by point. These Taylor and maximum-principle arguments are explicit mathematical premises of the numerical certificate. The Lean proof concerns discrete execution.
