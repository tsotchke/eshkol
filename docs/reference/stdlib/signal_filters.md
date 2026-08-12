# `signal.filters` — windows, convolution, FIR/IIR filters, Butterworth design

**Source**: [`lib/signal/filters.esk`](../../../lib/signal/filters.esk)
**Require**: `(require signal.filters)` — also **auto-loaded** by `(require stdlib)`. Internally requires `signal.fft`.

Digital-signal-processing primitives: window functions (Hamming, Hann, Blackman, Kaiser), windowing, direct and FFT-based convolution, FIR and IIR filtering, Butterworth low/high/band-pass design, and frequency response. All signals are real-valued vectors written `#(...)`; filter coefficient sets are returned as `(b-coeffs . a-coeffs)` pairs.

## Window functions

### `(hamming-window N)`
Length-`N` Hamming window, `w[n] = 0.54 − 0.46·cos(2πn/(N−1))`.

```scheme
(require signal.filters)
(display (hamming-window 5)) (newline)
```
```
#(0.08000000000000002 0.54 1 0.5400000000000001 0.08000000000000002)
```

### `(hann-window N)`
Length-`N` Hann (Hanning) window, `w[n] = 0.5·(1 − cos(2πn/(N−1)))`.

```scheme
(display (hann-window 5)) (newline)
```
```
#(0 0.49999999999999994 1 0.5000000000000001 0)
```

### `(blackman-window N)`
Length-`N` Blackman window, `w[n] = 0.42 − 0.5·cos(2πn/(N−1)) + 0.08·cos(4πn/(N−1))`.

```scheme
(display (blackman-window 5)) (newline)
```
```
#(-1.3877787807814457e-17 0.3399999999999999 0.9999999999999999 0.3400000000000001 -1.3877787807814457e-17)
```

### `(kaiser-window N beta)`
Length-`N` Kaiser window with shape parameter `beta`; uses an internal 25-term series for the modified Bessel function `I₀`.

```scheme
(display (kaiser-window 5 4.0)) (newline)
```
```
#(0.08848052607644988 0.6334317797559347 1 0.6334317797559347 0.08848052607644988)
```

### `(apply-window signal window)`
Element-wise product of `signal` and `window` (they should be the same length; the loop runs to `(vector-length signal)`).

```scheme
(display (apply-window #(1.0 1.0 1.0 1.0 1.0) (hamming-window 5))) (newline)
```
```
#(0.08000000000000002 0.54 1 0.5400000000000001 0.08000000000000002)
```

## Convolution

### `(convolve a b)`
Direct time-domain convolution. Output length `len(a) + len(b) − 1`.

```scheme
(display (convolve #(1.0 2.0 3.0) #(1.0 1.0))) (newline)
```
```
#(1 3 5 3)
```

### `(fast-convolve a b)`
FFT-based convolution, intended to be O(N log N) (zero-pads both signals to the next power of 2, multiplies spectra, inverse-transforms, takes real parts). **Currently broken — returns garbage.** See [Known issues](#known-issues); use `convolve` instead.

```scheme
(display (fast-convolve #(1.0 2.0 3.0) #(1.0 1.0))) (newline)
```
```
#(4966129304 -8 -8 -7.999999999999999)   ;; expected #(1 3 5 3); the leading
                                         ;; value is uninitialised memory and
                                         ;; differs on every run
```

## FIR / IIR filters

### `(fir-filter coeffs signal)`
FIR filter: `y[n] = Σ_k coeffs[k]·signal[n−k]`. Output length equals `signal`.

```scheme
(display (fir-filter #(0.5 0.5) #(1.0 2.0 3.0 4.0))) (newline)
```
```
#(0.5 1.5 2.5 3.5)
```

### `(iir-filter b-coeffs a-coeffs signal)`
IIR filter, Direct Form I: `y[n] = (1/a[0])·(Σ_k b[k]·x[n−k] − Σ_{k≥1} a[k]·y[n−k])`.

```scheme
;; one-pole lowpass y[n] = x[n] + 0.5 y[n-1]
(display (iir-filter #(1.0) #(1.0 -0.5) #(1.0 0.0 0.0 0.0))) (newline)
```
```
#(1 0.5 0.25 0.125)
```

## Butterworth design

Each design routine returns coefficients as a `(b-coeffs . a-coeffs)` cons pair (bandpass returns a two-element list of such pairs to be cascaded). Cutoffs are normalized: `1.0` = Nyquist.

### `(butterworth-lowpass order cutoff)`
Design an `order`-th Butterworth lowpass (bilinear transform of the analog prototype, DC-gain normalized). Returns `(b . a)`.

```scheme
(display (butterworth-lowpass 2 0.5)) (newline)
```
```
(#(0.2928932188134524 0.5857864376269049 0.2928932188134524) . #(1 -1.3877787807814457e-16 0.17157287525380988))
```

### `(butterworth-highpass order cutoff)`
Highpass via the lowpass-to-highpass spectral inversion (`z → −z` on the numerator). Returns `(b . a)`.

```scheme
(display (butterworth-highpass 2 0.5)) (newline)
```
```
(#(0.2928932188134524 -0.5857864376269049 0.2928932188134524) . #(1 -1.3877787807814457e-16 0.17157287525380988))
```

### `(butterworth-bandpass order low-cutoff high-cutoff)`
Bandpass as a lowpass+highpass cascade. Returns a **list of two** `(b . a)` pairs `(list lp hp)`; apply them in sequence to a signal.

```scheme
(display (butterworth-bandpass 2 0.2 0.6)) (newline)
```
```
((#(0.39133577250176854 0.7826715450035371 0.39133577250176854) . #(1 0.36952737735124097 0.19581571265583303)) (#(0.6389455251590224 -1.2778910503180447 0.6389455251590224) . #(1 1.142980502539901 0.4128015980961886)))
```

## Frequency response

### `(frequency-response b-coeffs a-coeffs n-points)`
Evaluate `H(e^{jω}) = B/A` at `n-points` frequencies from 0 to π. Returns `(magnitudes . phases)` as a cons of two vectors.

```scheme
(define lp (butterworth-lowpass 2 0.5))
(display (frequency-response (car lp) (cdr lp) 4)) (newline)
```
```
(#(1 0.9486832980505139 0.316227766016838 0) . #(0 -0.8860771237926136 -2.2555155297971794 0))
```

## Internal helpers (not in `provide`)

`window-ratio`, `bessel-i0`, `next-power-of-2`, `zero-pad-vector`, `complex-vector-real`, `butterworth-poles`, `bilinear-transform`, `poly-from-roots`, `real-part-vector` are defined in the module for use by the exported functions but are not part of the `provide` list.

## Known issues

### `fast-convolve` returns garbage

`fast-convolve` produces completely wrong output — the first element is a huge number in the billions that **differs on every run** (it is uninitialised memory) and the remaining elements collapse to a negative constant — whereas the direct `convolve` is correct.

```scheme
;; repro.esk
(require signal.filters)
(display (convolve      #(1.0 2.0 3.0 4.0 5.0) #(1.0 1.0 1.0))) (newline)
(display (fast-convolve #(1.0 2.0 3.0 4.0 5.0) #(1.0 1.0 1.0))) (newline)
```
```
#(1 3 6 9 12 9 5)                     ;; convolve — correct
#(4430393720 -8 -8 -8 -8 -7.999999999999999 -7.999999999999999)
                                      ;; fast-convolve — garbage; the leading
                                      ;; value differs on every run
```
Root cause: `fast-convolve` computes `(ifft (fft …))` inside its body, and
because `fast-convolve` lives in the **precompiled stdlib shared library**, it
hits the precompiled `fft`→`ifft` chaining corruption documented in
[`signal.fft`](signal_fft.md#known-issues) (an identical `fast-convolve` compiled
in ordinary user code — even with `(require stdlib)` — works). This is a compiler
bug (tracked as **ESH-0115**), not a library bug, so it is reported rather than
worked around in Scheme. Workaround: use the direct-time-domain `convolve`.
