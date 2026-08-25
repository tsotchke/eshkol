# Eshkol public benchmark results

Mode: FULL measurement run  
Started: 2026-08-25T15:41:40.060976+00:00  
Finished: 2026-08-25T15:47:08.267631+00:00

## Environment

- OS: Darwin 15.1 (kernel 24.1.0)
- CPU: Apple M2 Ultra (24 physical / 24 logical cores)
- GPU: Apple M2 Ultra
- Memory: 192 GiB
- Compiler: Homebrew clang version 21.1.7
- LLVM: 21.1.7
- BLAS: Apple Accelerate (vecLib/AMX)
- git: 4bf871a08774a39c97f01743c17e23eff59fbb55 (feat/benchmarks-wave1), dirty=True
- eshkol version: 1.3.4-evolve
- build: Release, quantum_enabled=ON, gpu_enabled=ON
- load average at capture: 166.93 153.50 162.74

### Axis 1: exact-AD cost curves

Order-sweep fit exponent (ns/call ~ k^p): float p=0.27, exact p=1.22 (p=2 is the claimed O(k^2))


**Order sweep — `(derivative-n f x k)`, f(x)=1/(1-x)**

| k | float ns/call | exact ns/call | exact/float ratio |
|---:|---:|---:|---:|
| 1 | 700 ns | 2.65 us | 3.79x |
| 2 | 1.04 us | 5.08 us | 4.87x |
| 3 | 938 ns | 5.23 us | 5.57x |
| 4 | 1.02 us | 8.54 us | 8.38x |
| 5 | 1.27 us | 13.23 us | 10.39x |
| 6 | 1.02 us | 17.29 us | 16.95x |
| 8 | 1.06 us | 23.90 us | 22.64x |
| 10 | 1.40 us | 30.52 us | 21.86x |
| 12 | 1.53 us | 13.93 us | 9.11x |
| 16 | 1.85 us | 63.23 us | 34.16x |
| 20 | 2.26 us | 88.34 us | 39.06x |
| 24 | 1.23 us | 172.44 us | 139.80x |

Crossover: exact-rational path first costs >=2x the float path at k=1 (3.79x). Below that order, exactness is close to free; above it, bignum growth dominates and paying for exactness is a real cost — reported honestly, not hidden.


**Dimension sweep — `(derivative-n f_d x 4)`, f_d(x)=x^(d+1) via a d-step chain**

| d | float ns/call | exact ns/call | exact/float ratio |
|---:|---:|---:|---:|
| 1 | 1.24 us | 10.15 us | 8.20x |
| 2 | 2.45 us | 26.67 us | 10.90x |
| 4 | 4.21 us | 39.37 us | 9.35x |
| 8 | 4.62 us | 69.98 us | 15.15x |
| 16 | 10.88 us | 134.77 us | 12.39x |
| 32 | 15.00 us | 127.98 us | 8.53x |
| 64 | 31.54 us | 1.226 ms | 38.87x |

Dimension-sweep fit exponent (ns/call ~ d^p): float p=0.74, exact p=0.97 (p=1 is linear in workload size)


### Axis 2: Ozaki-II CRT exact f64 GEMM vs vendor BLAS

**Throughput (GF/s, median of repeated samples)**

| N | AMX (vendor BLAS) | Ozaki-II exact | Ozaki-II fast |
|---:|---:|---:|---:|
| 256 | 223.7 | 11.6 | 12.3 |
| 512 | 56.0 | 19.9 | 64.1 |
| 1024 | 109.8 | 43.3 | 183.4 |
| 2048 | 170.6 | 88.5 | 320.5 |
| 4096 | 178.0 | 106.2 | 507.0 |
| 8192 | 59.8 | 80.7 | 614.8 |

**Accuracy vs an exact-rational reference (max/mean relative error over sampled entries)**

| Kernel | max relerr | mean relerr | samples |
|---|---:|---:|---:|
| amx | 2.936e-16 | 1.109e-16 | 64 |
| ozaki | 2.517e-13 | 1.226e-13 | 64 |
| ozaki-fast | 2.517e-13 | 1.226e-13 | 64 |

The correctness gate this axis's methodology follows (tests/gpu/ozaki_correctness_gate.sh) verifies Ozaki-II exact against a TOL=1e-9 threshold, not literal bit-exactness — its own reference is a naive f64 accumulation, accurate to ~K*epsilon. At the modest K this axis samples (N=64), vendor BLAS's own few accumulation steps can measure MORE accurate against a true exact-rational reference than Ozaki-II's fixed ~1e-13 CRT-reconstruction floor — that is not a regression, it is what a fixed reconstruction-precision floor vs a K-dependent accumulation error look like at small K. Ozaki-II's accuracy advantage over vendor BLAS is expected to widen as K grows and/or input dynamic range widens, since BLAS's per-step rounding error grows with K while Ozaki-II's stays governed by its fixed moduli budget. Report the numbers as measured; do not round this nuance away.


### Axis 3: flat-RSS under resident load

**Native (AOT) — ESH-0214e-shaped resident daemon loop**

| ticks | peak RSS (MB) | ok |
|---:|---:|:---:|
| 10000 | 11 | yes |
| 25000 | 15 | yes |
| 50000 | 23 | yes |
| 100000 | 36 | yes |
| 200000 | 64 | yes |
| 400000 | 119 | yes |

10000 -> 400000 ticks (40.0x more work): 11MB -> 119MB peak RSS (10.82x). NOT FLAT (allowance 1.5x).


**VM (bytecode, eshkol-vm-standalone-test) — with-region sweep**

| iterations | peak RSS (MB) | ok |
|---:|---:|:---:|
| 1000 | 26 | yes |
| 4000 | 26 | yes |
| 16000 | 27 | yes |
| 64000 | 28 | yes |

1000 -> 64000 iterations (64.0x more work): 26MB -> 28MB peak RSS (1.08x). FLAT (allowance 1.5x).


Evacuator on vs off at n=64000: 28MB (on) vs 3093MB (off).


### Axis 4: differentiable quantum kernels

**H2 vibrational frequency (exact 2nd-order AD Hessian, no quantum build required)**

- equilibrium R* = 1.3886947151919542 bohr
- force constant d2E/dR2 = 0.47709682545822807 Ha/bohr^2
- vibrational frequency = 5003.203812553002 cm^-1
- wall clock (median of 5 runs): 1.148 s

**VQE H2 energy + native adjoint gradient (requires -DESHKOL_QUANTUM_ENABLED=ON)**

- H2 exact ground energy = -1.142200155381327 Ha
- H2 VQE optimized energy = -1.1422001553813284 Ha
- |VQE - exact| = 1.3322676295501878e-15 Ha
- gradient entries = 4
- wall clock (median of 5 runs): 167.1 ms

