# Quantum-Inspired Random Number Generation

**Status:** Production (v1.1.11)
**Applies to:** Eshkol compiler v1.1-accelerate and later

---

## Overview

Eshkol provides a dual-layer random number generation system: a standard pseudorandom number generator (PRNG) based on `drand48`, and a quantum-inspired RNG (QRNG) that simulates quantum circuit behavior on classical hardware. Both layers are exposed through a unified Eshkol API in `lib/random/random.esk`.

The quantum-inspired RNG differs from standard PRNGs in several important ways:

- **Quantum circuit simulation.** The QRNG maintains an 8-qubit state vector and applies Hadamard gates, phase gates, and entanglement operations to evolve the state. Each random number is produced by "measuring" the quantum state.
- **Hardware entropy seeding.** The QRNG collects entropy from system sources (timing jitter, process ID, system entropy pool) to initialize its state, rather than relying solely on a numeric seed.
- **Multiple mixing stages.** Output passes through Hadamard mixing, Pauli gate rotations, and splitmix64 avalanche functions. Physical constants (fine structure, Planck, Rydberg) serve as mixing constants.
- **Non-deterministic output.** Because the QRNG incorporates runtime entropy (timing, floating-point uncertainty), identical seeds do not guarantee identical output sequences.

For most applications, the standard PRNG functions are sufficient. Use the quantum functions when you need higher-quality randomness or want non-reproducible sequences.

---

## Getting Started

```scheme
(require random)

;; Standard PRNG
(display (random-float))    ; => 0.7234... (uniform [0,1))
(display (random-int 1 6))  ; => 4 (dice roll)

;; Quantum-inspired RNG
(display (qrandom))         ; => 0.5128... (quantum uniform [0,1))
(display (qrandom-int 1 6)) ; => 2 (quantum dice roll)
```

---

## Standard PRNG Functions

These functions use `drand48`-based pseudorandom number generation. They are deterministic given the same seed.

### random-float

```scheme
(random-float) -> float
```

Returns a random floating-point number in the range [0, 1).

### random-int

```scheme
(random-int low high) -> integer
```

Returns a random integer in the range [low, high] (inclusive on both ends).

### random-bool

```scheme
(random-bool) -> boolean
```

Returns `#t` or `#f` with equal probability.

### random-choice

```scheme
(random-choice lst) -> element
```

Selects a random element from a list. Returns `#f` if the list is empty.

---

## Quantum-Inspired RNG Functions

These functions use the quantum circuit simulator for higher-quality randomness. They call through to the C-level QRNG library (`lib/quantum/quantum_rng.c`).

### qrandom

```scheme
(qrandom) -> float
```

Returns a quantum-inspired random float in [0, 1). Delegates to `quantum-random`, a compiler builtin that calls `eshkol_qrng_double()`.

### qrandom-int

```scheme
(qrandom-int low high) -> integer
```

Returns a quantum-inspired random integer in [low, high] (inclusive). Delegates to `quantum-random-range`.

### qrandom-bool

```scheme
(qrandom-bool) -> boolean
```

Returns `#t` or `#f` using quantum randomness.

### qrandom-choice

```scheme
(qrandom-choice lst) -> element
```

Selects a random element from a list using quantum randomness.

---

## Distribution Sampling

All distribution functions use the standard PRNG. For quantum-seeded distributions, call `quniform` or compose quantum base functions manually.

### uniform

```scheme
(uniform low high) -> float
```

Returns a random float uniformly distributed in [low, high].

### quniform

```scheme
(quniform low high) -> float
```

Same as `uniform` but uses the quantum RNG as the source.

### normal

```scheme
(normal mu sigma) -> float
```

Samples from a normal (Gaussian) distribution with mean `mu` and standard deviation `sigma`. Uses the Box-Muller transform internally.

### normal-pair

```scheme
(normal-pair) -> (list z1 z2)
```

Returns a pair of independent standard normal samples (mean 0, std 1) using the Box-Muller transform. More efficient than calling `normal` twice since Box-Muller naturally produces pairs.

### exponential

```scheme
(exponential lambda) -> float
```

Samples from an exponential distribution with rate parameter `lambda`. Uses inverse transform sampling: `-(1/lambda) * ln(U)`.

### poisson

```scheme
(poisson lambda) -> integer
```

Samples from a Poisson distribution with mean `lambda`. Uses the Knuth algorithm, which is efficient for small values of lambda.

### bernoulli

```scheme
(bernoulli p) -> integer
```

Samples from a Bernoulli distribution. Returns 1 with probability `p`, 0 otherwise.

### geometric

```scheme
(geometric p) -> integer
```

Samples from a geometric distribution with success probability `p`. Returns the number of failures before the first success.

### binomial

```scheme
(binomial n p) -> integer
```

Samples from a binomial distribution B(n, p) using the direct trial method. Returns the number of successes in `n` independent Bernoulli trials, each with success probability `p`.

---

## Tensor and Vector Random Functions

### random-tensor

```scheme
(random-tensor dims) -> tensor
```

Creates a tensor with random uniform values in [0, 1). The `dims` argument is a list of dimension sizes.

```scheme
(random-tensor (list 3 4))   ; 3x4 matrix of random values
(random-tensor (list 10))    ; 10-element random vector
```

Delegates to the `rand` builtin with `apply`.

### random-normal-tensor

```scheme
(random-normal-tensor dims) -> tensor
```

Creates a tensor with standard normal (mean 0, std 1) values. The `dims` argument is a list of dimension sizes.

```scheme
(random-normal-tensor (list 100 100))  ; 100x100 matrix of normal samples
```

Delegates to the `randn` builtin with `apply`.

### random-vector

```scheme
(random-vector n) -> vector
```

Creates a Scheme vector of `n` random floats in [0, 1).

### random-uniform-vector

```scheme
(random-uniform-vector n low high) -> vector
```

Creates a Scheme vector of `n` random floats uniformly distributed in [low, high].

### random-normal-vector

```scheme
(random-normal-vector n) -> vector
```

Creates a Scheme vector of `n` standard normal samples.

---

## Combinatorial Functions

### shuffle

```scheme
(shuffle lst) -> list
```

Returns a random permutation of a list using the Fisher-Yates algorithm. The original list is not modified.

```scheme
(shuffle '(1 2 3 4 5))  ; => (3 1 5 2 4) or any permutation
```

### sample

```scheme
(sample lst k) -> list
```

Returns a random sample of `k` elements from a list without replacement. Implemented by shuffling the list and taking the first `k` elements.

```scheme
(sample '(a b c d e) 3)  ; => (c a e) or any 3-element subset
```

### weighted-choice

```scheme
(weighted-choice items weights) -> element
```

Selects an element from `items` with probability proportional to the corresponding entry in `weights`. Weights should be non-negative numbers.

```scheme
(weighted-choice '(rare common) '(1 9))  ; "common" ~90% of the time
```

---

## Seed Control

### set-random-seed!

```scheme
(set-random-seed! seed) -> void
```

Sets the PRNG seed. This affects all standard random functions (`random-float`, `random-int`, etc.) but not the quantum functions.

### randomize!

```scheme
(randomize!) -> void
```

Initializes the PRNG with a time-based seed. Call this at program start for non-reproducible standard random sequences.

### current-time-seed

```scheme
(current-time-seed) -> integer
```

Returns the current time in seconds since the epoch, suitable for use as a seed.

---

## C-Level API

The quantum RNG is implemented in C with the following architecture:

| File | Purpose |
|------|---------|
| `lib/quantum/quantum_rng.h` | Core QRNG API (context, bytes, double, range) |
| `lib/quantum/quantum_rng.c` | Quantum circuit simulation implementation |
| `lib/quantum/quantum_rng_wrapper.h` | Global context wrapper for Eshkol runtime |
| `lib/quantum/quantum_rng_wrapper.c` | Wrapper implementation (auto-init on first call) |

### Core C Functions

```c
qrng_error qrng_init(qrng_ctx **ctx, const uint8_t *seed, size_t seed_len);
void       qrng_free(qrng_ctx *ctx);
qrng_error qrng_reseed(qrng_ctx *ctx, const uint8_t *seed, size_t seed_len);
qrng_error qrng_bytes(qrng_ctx *ctx, uint8_t *out, size_t len);
uint64_t   qrng_uint64(qrng_ctx *ctx);
double     qrng_double(qrng_ctx *ctx);
int32_t    qrng_range32(qrng_ctx *ctx, int32_t min, int32_t max);
uint64_t   qrng_range64(qrng_ctx *ctx, uint64_t min, uint64_t max);
double     qrng_get_entropy_estimate(qrng_ctx *ctx);
```

### Eshkol Runtime Wrapper

The wrapper provides a global context that is initialized automatically on first use:

```c
int      eshkol_qrng_init(void);
double   eshkol_qrng_double(void);
uint64_t eshkol_qrng_uint64(void);
int64_t  eshkol_qrng_range(int64_t min, int64_t max);
int      eshkol_qrng_bytes(uint8_t* buffer, size_t len);
```

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `QRNG_SUCCESS` | Operation completed successfully |
| -1 | `QRNG_ERROR_NULL_CONTEXT` | NULL context provided |
| -2 | `QRNG_ERROR_NULL_BUFFER` | NULL buffer provided |
| -3 | `QRNG_ERROR_INVALID_LENGTH` | Invalid length parameter |
| -4 | `QRNG_ERROR_INSUFFICIENT_ENTROPY` | Not enough entropy available |
| -5 | `QRNG_ERROR_INVALID_RANGE` | Invalid range parameters |

---

## Implementation Notes

The quantum circuit simulator uses 8 qubits with the following internal state:

- `phase[8]` -- Phase angles for each qubit
- `entangle[8]` -- Entanglement state between qubit pairs
- `quantum_state[8]` -- Continuous quantum amplitudes (doubles)
- `last_measurement[8]` -- Previous measurement results
- `entropy_pool[16]` -- Accumulated entropy from system sources

Each step of the quantum circuit applies:

1. Hadamard gates (superposition creation)
2. Phase gates (parameterized by physical constants)
3. Entanglement operations (CNOT-like mixing between qubit pairs)
4. Measurement (state collapse to definite values)
5. Output mixing (splitmix64 avalanche + Pauli gate rotations)

The mixing constants are derived from physical constants: the fine structure constant, Planck's constant, the Rydberg constant, and the electron g-factor, encoded as 64-bit integer representations.

---

## See Also

- [Math Standard Library](MATH_STDLIB.md) -- Statistics, special functions, ODE solvers
- [Vector Operations](VECTOR_OPERATIONS.md) -- Tensor creation and operations
- [Machine Learning](MACHINE_LEARNING.md) -- Training pipelines that use random initialization
- [API Reference](../API_REFERENCE.md) -- Complete function reference
