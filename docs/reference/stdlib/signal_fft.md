# `signal.fft` — Fast Fourier Transform and inverse

**Source**: [`lib/signal/fft.esk`](../../../lib/signal/fft.esk)
**Require**: `(require signal.fft)` — also **auto-loaded** by `(require stdlib)`.

Discrete Fourier transform (`fft`) and its inverse (`ifft`), radix-2 Cooley–Tukey decimation-in-time. Input is a vector of real (or complex) numbers; output is a vector of complex numbers.

> **Note — these names resolve to a codegen builtin.** The `.esk` source in this module defines Scheme implementations, but `fft` and `ifft` are also compiler intrinsics (`codegenFFT` in `lib/backend/llvm_codegen.cpp`), and the builtin takes precedence. The observable behavior documented here (power-of-2 requirement, complex-vector output, tensor **and** vector inputs accepted) is that of the native builtin. `(require signal.fft)` is still the supported way to make the names available in portable source.

## Functions

### `(fft x)`
Forward DFT of vector `x`. Length must be a power of 2. Real inputs are treated as complex with zero imaginary part; output is always a complex vector.

```scheme
(require signal.fft)
(display (fft #(1.0 2.0 3.0 4.0))) (newline)
```
```
#(10 -2+2i -2 -2-2i)
```

Edge cases: a non-power-of-2 length aborts the program with a printed error (from the native builtin) rather than raising a catchable condition:

```scheme
(display (fft #(1.0 2.0 3.0))) (newline)
```
```
Error: FFT requires input length to be a power of 2
```

### `(ifft x)`
Inverse DFT. Length must be a power of 2. Returns a complex vector; `ifft ∘ fft` recovers the input (up to ~1e-17 floating-point round-off in the imaginary parts).

```scheme
(display (ifft (fft #(1.0 2.0 3.0 4.0)))) (newline)
```
```
#(1 2+5.72119e-18i 3 4-5.72119e-18i)
```

Edge cases: same power-of-2 requirement as `fft`.

## Known issues

### Chaining `fft` → `ifft` inside one compiled function returns garbage

At the REPL top level, `fft` and `ifft` each work and round-trip correctly. But when the result of `fft` is fed into `ifft` **within the body of a user- or library-defined function**, the round-trip is corrupted — the first element becomes a huge number (~5e9) and the rest collapse to a constant. `fft` alone inside a function is fine; it is the `fft`→`ifft` chain that breaks.

```scheme
;; repro.esk
(require signal.fft)
(define (roundtrip a) (ifft (fft a)))
(display (roundtrip #(1.0 2.0 3.0 4.0))) (newline)
```
```
#(4.96616e+09 -8-8i -8 -8+8i)      ;; expected ~ #(1 2 3 4)
```
This is the direct cause of the `fast-convolve` corruption documented in [`signal.filters`](signal_filters.md#known-issues). Workaround: keep `fft` and `ifft` in separate top-level steps, or use direct-time-domain `convolve`.
