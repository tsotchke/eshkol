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
#(10 -2+2i -2 -1.9999999999999998-2i)
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
#(1 2+5.721188726109833e-18i 3 4-5.721188726109833e-18i)
```

Edge cases: same power-of-2 requirement as `fft`.

## Complex input and the round trip

`fft` and `ifft` accept real or complex elements, and `(ifft (fft x))` returns
`x` up to rounding on every engine — the JIT with or without the run cache,
AOT, the bytecode VM, and the precompiled stdlib (which is how
[`fast-convolve`](signal_filters.md) reaches them):

```scheme
(require signal.fft)
(define (roundtrip a) (ifft (fft a)))
(display (roundtrip #(1.0 2.0 3.0 4.0))) (newline)
;; => #(1 2+5.72e-18i 3 4-5.72e-18i)   ; ~ #(1 2 3 4)
(display (fft (fft #(1.0 2.0 3.0 4.0)))) (newline)
;; => #(4 16+4.67e-16i 12 8-4.67e-16i) ; ~ N * x[-n] = #(4 16 12 8)
```

The recursion's length-1 base case wraps a *real* element as a complex number
and passes a complex element through unchanged (`tests/signal/fft_complex_input_test.esk`).
