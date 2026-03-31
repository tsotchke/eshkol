# Signal Processing in Eshkol

**Status:** Production (v1.1-accelerate)
**Module path:** `lib/signal/`
**Source files:** `lib/signal/fft.esk`, `lib/signal/filters.esk`

---

## Overview

Eshkol's signal processing library is implemented entirely in the Eshkol language itself, as
first-class stdlib code compiled alongside user programs. There is no separate C++ codegen for
these routines — they use the same tagged-value arithmetic, complex number primitives, and
vector operations that are available to any Eshkol program.

The library is split into two modules with a clean dependency:

```
signal.filters  →  signal.fft  (filters requires fft for fast-convolve)
```

**Import options:**

```scheme
(require signal)           ; imports both fft and filters
(require signal.fft)       ; FFT/IFFT only
(require signal.filters)   ; filters, windows, convolution (pulls in fft automatically)
```

**Provided symbols from `signal.fft`:**
`fft`, `ifft`

**Provided symbols from `signal.filters`:**
`hamming-window`, `hann-window`, `blackman-window`, `kaiser-window`, `apply-window`,
`convolve`, `fast-convolve`, `fir-filter`, `iir-filter`,
`butterworth-lowpass`, `butterworth-highpass`, `butterworth-bandpass`, `frequency-response`

---

## 1. FFT / IFFT

### 1.1 Theoretical Background

The Discrete Fourier Transform (DFT) maps a length-N sequence x[n] into the frequency domain:

```
         N-1
X[k] =  sum  x[n] * e^(-2*pi*i*k*n/N)    for k = 0, 1, ..., N-1
         n=0
```

Direct evaluation of this sum costs O(N²) operations per output bin, giving O(N²) overall.
The Fast Fourier Transform (FFT) achieves O(N log N) via the Cooley-Tukey divide-and-conquer
factorization.

### 1.2 Cooley-Tukey Radix-2 DIT Algorithm

The Cooley-Tukey algorithm (1965) exploits the symmetry and periodicity of the complex
exponential twiddle factors W_N^k = e^(-2*pi*i*k/N). For even N, the DFT can be split:

```
X[k] = E[k] + W_N^k * O[k]           for k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - W_N^k * O[k]     for k = 0, ..., N/2 - 1
```

where:
- E[k] = DFT of the even-indexed subsequence x[0], x[2], x[4], ..., x[N-2]
- O[k] = DFT of the odd-indexed subsequence x[1], x[3], x[5], ..., x[N-1]
- W_N^k = e^(-2*pi*i*k/N) is the twiddle factor

This is the **butterfly operation**: each output pair (X[k], X[k+N/2]) is computed from one
element of E, one element of O, and one twiddle factor. Applying this recursively to
sub-problems of size N/2, N/4, ..., 1 yields log₂N stages, each with N/2 butterflies, for
a total of (N/2) log₂N complex multiplications.

The decimation-in-time (DIT) variant recursively splits on input indices. The input length must
be a power of 2.

**Base case:** A length-1 input has DFT equal to itself (wrapped as a complex number).

### 1.3 Complexity

| Method | Time | Space |
|--------|------|-------|
| Direct DFT | O(N²) | O(N) |
| Cooley-Tukey FFT | O(N log N) | O(N log N) recursive |

The recursive implementation allocates O(N) working vectors per level, for O(N log N) total
allocation. An iterative in-place version (bit-reversal permutation + iterative butterfly) would
reduce allocation to O(N), but the recursive form is preferred here for clarity.

### 1.4 API

#### `(fft x)` → vector of complex

Compute the forward DFT of vector `x`.

- **Input:** vector of real numbers or complex numbers, length must be a power of 2. If real
  inputs are provided, each element is promoted to complex with zero imaginary part at the leaf.
- **Output:** vector of complex numbers, same length as input.
- **Indexing:** X[0] = DC component, X[1]..X[N/2-1] = positive frequencies,
  X[N/2]..X[N-1] = negative frequencies (conjugate mirror for real input).

#### `(ifft x)` → vector of complex

Compute the inverse DFT of vector `x`.

- **Input:** vector of complex numbers, length must be a power of 2.
- **Output:** vector of complex numbers. For real-valued original signals the imaginary parts
  will be near zero (floating-point rounding noise).

**Implementation — conjugate trick:**

The IFFT is computed without a separate inverse butterfly pass:

```
IFFT(x) = conj(FFT(conj(x))) / N
```

This reuses the forward FFT kernel by:
1. Conjugating all inputs (negate imaginary parts).
2. Calling the forward `fft`.
3. Conjugating and scaling all outputs by 1/N.

This approach avoids duplicating the butterfly logic and is exact for both real and complex
input sequences.

### 1.5 Code Examples

**Spectrum of a real sinusoid:**

```scheme
(require signal.fft)

; Generate 8 samples of a 1/4-Nyquist sine (period = 8 samples)
(define signal (vector 0.0 1.0 0.0 -1.0 0.0 1.0 0.0 -1.0))
(define spectrum (fft signal))

; Extract magnitudes: (magnitude (vref spectrum k)) for each bin
; Bin 2 and bin 6 (conjugate) should have magnitude 4.0
```

**Round-trip identity test:**

```scheme
(require signal.fft)

(define x (vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0))
(define y (fft x))
(define z (ifft y))
; (real-part (vref z 0)) ≈ 1.0, etc. (within floating-point tolerance)
```

**Frequency bin to Hz conversion:**

```scheme
; For N-point FFT at sample rate fs, bin k corresponds to frequency:
;   f_k = k * fs / N     for k = 0, 1, ..., N/2
; Bin 0 = DC (0 Hz), bin N/2 = Nyquist (fs/2)
```

---

## 2. Window Functions

### 2.1 Spectral Leakage and Windowing

When computing the DFT of a finite-length signal, the implicit assumption is that the signal
repeats periodically with period N. If the signal is not periodic within the observation window
(the common case), the discontinuity at the boundary introduces **spectral leakage**: energy
from a single-frequency sinusoid spreads across neighboring bins in the frequency domain.

A window function w[n] tapers the signal to zero at both ends before computing the DFT. This
eliminates the discontinuity at the cost of widening the main lobe of each frequency peak
(reduced frequency resolution). The trade-off between main lobe width and side lobe level
distinguishes the available window types.

The windowed DFT is:
```
X[k] = DFT(x[n] * w[n])
```

### 2.2 Hamming Window

The Hamming window is an optimally weighted sum of two cosine terms designed to minimize the
peak side lobe level:

```
w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))     n = 0, 1, ..., N-1
```

The coefficients 0.54 and 0.46 (summing to 1.0) are chosen to place a zero in the frequency
domain exactly at the first side lobe, reducing it to approximately -43 dB.

**Source:** `lib/signal/filters.esk`, lines 30-40.

#### `(hamming-window N)` → vector of N real values

### 2.3 Hann Window

The Hann (often misnamed "Hanning") window uses a pure cosine taper:

```
w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))     n = 0, 1, ..., N-1
```

Equivalently: w[n] = sin²(pi*n/(N-1)). This is the standard window for spectral analysis
where adjacent bins are expected to overlap (overlap-add processing). Side lobe level
approximately -31 dB, main lobe width 4 bins.

**Source:** `lib/signal/filters.esk`, lines 44-54.

#### `(hann-window N)` → vector of N real values

### 2.4 Blackman Window

The Blackman window uses three cosine terms:

```
w[n] = 0.42 - 0.5 * cos(2*pi*n/(N-1)) + 0.08 * cos(4*pi*n/(N-1))
```

The additional cos(4*pi*n/(N-1)) term suppresses the second and third side lobes, achieving
approximately -58 dB side lobe attenuation. The main lobe is wider (6 bins) than Hann or
Hamming, making Blackman appropriate when spurious frequency content must be suppressed at the
expense of frequency resolution.

**Source:** `lib/signal/filters.esk`, lines 57-70.

#### `(blackman-window N)` → vector of N real values

### 2.5 Kaiser Window

The Kaiser window is parameterized by a shape parameter β that continuously trades main lobe
width for side lobe attenuation:

```
w[n] = I0(β * sqrt(1 - ((2n/(N-1)) - 1)²)) / I0(β)     n = 0, 1, ..., N-1
```

where I0 is the modified Bessel function of the first kind, order zero:

```
I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)²
```

The implementation evaluates the series to k=25 terms, which provides full double-precision
accuracy for β ≤ 20.

```scheme
; bessel-i0 series (lib/signal/filters.esk, lines 72-81):
; loop: k from 1 to 25
;   new-term = term * (half-x)^2 / k^2
;   sum += new-term
; starting from sum=1.0 (k=0 term)
```

**Typical β values and their characteristics:**

| β    | Approximate side lobe level | Main lobe width (bins) |
|------|-----------------------------|------------------------|
| 0    | -13 dB (rectangular)        | 2                      |
| 2.12 | -30 dB                      | ~4                     |
| 5.65 | -60 dB                      | ~8                     |
| 8.96 | -90 dB                      | ~12                    |

**Source:** `lib/signal/filters.esk`, lines 83-96.

#### `(kaiser-window N beta)` → vector of N real values

- `N` — window length (integer)
- `beta` — shape parameter β ≥ 0 (real). β = 0 gives a rectangular window.

### 2.6 Applying a Window

#### `(apply-window signal window)` → vector of N real values

Element-wise multiplication of `signal` and `window` vectors. Both must have the same length.
Returns a new vector; neither input is modified.

```scheme
(define w (hamming-window 512))
(define windowed (apply-window raw-signal w))
(define spectrum (fft windowed))
```

### 2.7 Window Comparison Table

| Window    | Side lobe level | Main lobe width | Scalloping loss | Recommended use |
|-----------|-----------------|-----------------|-----------------|-----------------|
| Rectangular | -13 dB        | 2 bins          | 3.9 dB          | Transient analysis |
| Hann      | -31 dB          | 4 bins          | 1.4 dB          | General spectrum analysis |
| Hamming   | -43 dB          | 4 bins          | 1.8 dB          | Narrowband detection |
| Blackman  | -58 dB          | 6 bins          | 1.1 dB          | High dynamic range |
| Kaiser β=6 | ~-70 dB        | ~10 bins        | configurable    | Filter design, arbitrary sidelobe |

---

## 3. Convolution

### 3.1 Theory

Discrete linear convolution of sequences a[n] (length N_a) and b[n] (length N_b) is:

```
          N_b - 1
(a*b)[n] = sum  a[n-k] * b[k]     for n = 0, ..., N_a + N_b - 2
           k=0
```

The output has length N_a + N_b - 1.

Convolution in the time domain corresponds to pointwise multiplication in the frequency
domain (convolution theorem):

```
DFT(a * b) = DFT(a) · DFT(b)
```

This enables an O(N log N) convolution via FFT when the signals are long.

### 3.2 Direct Convolution

#### `(convolve a b)` → vector of length N_a + N_b - 1

Computes the linear convolution using the direct O(N_a * N_b) double loop.

```scheme
(convolve (vector 1.0 2.0 3.0)
          (vector 1.0 1.0))
; → #(1.0 3.0 5.0 3.0)
```

Use when either signal is short (M ≤ ~32), or when minimizing allocation is a priority.

**Source:** `lib/signal/filters.esk`, lines 116-135.

### 3.3 Fast (FFT-Based) Convolution

#### `(fast-convolve a b)` → vector of length N_a + N_b - 1

Computes convolution via the overlap-direct method:

1. Compute output length: N_c = N_a + N_b - 1.
2. Find the smallest power of 2 ≥ N_c: `fft-len = next-power-of-2(N_c)`.
3. Zero-pad both `a` and `b` to `fft-len`.
4. Compute FFT of both zero-padded signals.
5. Element-wise complex multiply the two spectra.
6. IFFT the product spectrum.
7. Take real parts and truncate to N_c samples.

```scheme
(fast-convolve (vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
               (vector 1.0 -1.0))
; Computes the first-difference filter applied to an 8-sample signal
```

**Complexity:** O(N log N) where N = next-power-of-2(N_a + N_b - 1).

**Source:** `lib/signal/filters.esk`, lines 167-199.

**Helper functions (internal):**
- `next-power-of-2`: returns smallest power of 2 >= n (lines 138-141).
- `zero-pad-vector`: pads a vector to a given length with zeros (lines 144-153).
- `complex-vector-real`: extracts real parts from a complex-valued vector (lines 155-165).

### 3.4 When to Use Each

| Condition | Recommended |
|-----------|-------------|
| N_a ≤ 32 or N_b ≤ 32 | `convolve` (lower overhead) |
| Both N_a, N_b ≥ 64 | `fast-convolve` |
| FIR filtering long signal | `fast-convolve` |
| Exact integer arithmetic needed | `convolve` (avoids FFT rounding) |

---

## 4. Digital Filters

### 4.1 FIR Filters

A **Finite Impulse Response (FIR)** filter has output:

```
          M
y[n] =   sum  b[k] * x[n-k]
          k=0
```

where b[k] are the M+1 filter coefficients (the impulse response). Properties:
- Always BIBO stable (no feedback, bounded output for bounded input).
- Can be designed with exactly linear phase (symmetric coefficients).
- Requires more coefficients (higher order) than IIR for the same selectivity.

#### `(fir-filter coeffs signal)` → vector of length N

Apply an FIR filter to a signal.

- `coeffs` — vector of filter coefficients b[0], b[1], ..., b[M].
- `signal` — vector of input samples.

For each output sample y[n], only past and present input samples are used. If n-k < 0 (before
the start of the signal), the contribution is zero (zero initial conditions).

```scheme
(require signal.filters)

; 3-point moving average: b = [1/3, 1/3, 1/3]
(define ma3 (vector (/ 1.0 3.0) (/ 1.0 3.0) (/ 1.0 3.0)))
(define signal (vector 1.0 2.0 3.0 4.0 5.0 6.0))
(define smoothed (fir-filter ma3 signal))
; → #(0.333 1.0 2.0 3.0 4.0 5.0)
```

**Source:** `lib/signal/filters.esk`, lines 207-225.

### 4.2 IIR Filters

An **Infinite Impulse Response (IIR)** filter incorporates feedback:

```
           1      ( M               N          )
y[n] = ------- * ( sum b[k]*x[n-k] - sum a[k]*y[n-k] )
         a[0]    ( k=0              k=1         )
```

This is the Direct Form I difference equation. Properties:
- Can achieve high selectivity with fewer coefficients than FIR.
- Potentially unstable if poles lie outside the unit circle.
- Generally nonlinear phase (except special symmetric designs).

#### `(iir-filter b-coeffs a-coeffs signal)` → vector of length N

Apply an IIR filter in Direct Form I.

- `b-coeffs` — feedforward coefficients b[0], b[1], ..., b[M].
- `a-coeffs` — feedback coefficients a[0], a[1], ..., a[N]. a[0] is the normalization factor
  (typically 1.0 for normalized designs).
- `signal` — vector of input samples.

```scheme
(require signal.filters)

; Simple first-order lowpass IIR: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
; alpha = 0.1 → slow-responding smoother
(define b (vector 0.1))
(define a (vector 1.0 -0.9))   ; a[0]=1, a[1]=-0.9 (feedback of previous output)
(define signal (vector 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0))
(define filtered (iir-filter b a signal))
; Output rises exponentially towards 1.0 (step response of first-order system)
```

**Source:** `lib/signal/filters.esk`, lines 229-258.

**Note on stability:** The feedback loop in IIR filters means numerical errors accumulate.
For Butterworth designs generated by `butterworth-lowpass` / `butterworth-highpass`, the poles
are by construction inside the unit circle. For user-supplied coefficients, stability is the
caller's responsibility.

---

## 5. Butterworth Filter Design

### 5.1 Analog Prototype

The Butterworth filter is the maximally flat magnitude (no ripple in passband or stopband)
IIR filter. The N-th order analog Butterworth lowpass prototype has magnitude response:

```
|H(jΩ)|² = 1 / (1 + (Ω/Ωc)^(2N))
```

The N poles of this prototype lie on the unit circle in the left half of the complex s-plane
at angles:

```
θ_k = (2k + N + 1) * pi / (2N)     for k = 0, 1, ..., N-1
s_k = cos(θ_k) + i * sin(θ_k)
```

Because all poles are in the left half-plane (real part < 0), the analog prototype is stable.

**Source:** `lib/signal/filters.esk`, `butterworth-poles`, lines 266-274.

### 5.2 Bilinear Transform

The bilinear transform maps the analog s-plane to the digital z-plane with no aliasing:

```
s = (2/T) * (z - 1) / (z + 1)
```

or equivalently, solving for z:

```
z = (1 + s * T/2) / (1 - s * T/2)
```

The parameter T is chosen via **frequency prewarping** to place the digital cutoff frequency
at the desired normalized angular frequency ωc:

```
T = 2 * tan(ωc / 2)
```

where ωc = π * cutoff (cutoff normalized to [0, 1] where 1 = Nyquist = fs/2).

The bilinear transform compresses the entire infinite analog frequency axis [−∞, +∞] onto the
unit circle [−π, π] in the z-domain. This compression is nonlinear (the "frequency warping"
effect), which is why prewarping is needed to hit the correct digital cutoff frequency.

**Source:** `lib/signal/filters.esk`, `bilinear-transform`, lines 278-281.

### 5.3 Polynomial Construction

After applying the bilinear transform to each analog pole, the denominator polynomial
a(z) = (z - p_0)(z - p_1)···(z - p_{N-1}) is expanded by successive polynomial multiplication.
The internal helper `poly-from-roots` (lines 284-307) performs this expansion iteratively.

For the Butterworth lowpass, all N digital zeros are placed at z = -1 (DC gain of 1, zero at
Nyquist), giving the numerator polynomial b(z) = (z + 1)^N. The gain is then normalized so
that H(z=1) = 1 (unity DC response).

### 5.4 Lowpass Design

#### `(butterworth-lowpass order cutoff)` → pair (b-coeffs . a-coeffs)

Design an N-th order Butterworth lowpass filter.

- `order` — filter order N (integer ≥ 1). Higher order = steeper rolloff. Each order adds
  6 dB/octave rolloff beyond the cutoff. A 4th-order filter rolls off at 24 dB/octave.
- `cutoff` — normalized cutoff frequency in (0, 1), where 1.0 = Nyquist (fs/2). For a
  1000 Hz sample rate and 100 Hz cutoff, use cutoff = 100/500 = 0.2.

Returns a cons pair. Access with `(car result)` for b-coefficients, `(cdr result)` for
a-coefficients.

```scheme
(require signal.filters)

(define lp (butterworth-lowpass 4 0.2))   ; 4th-order, fc = 0.2 * Nyquist
(define b (car lp))
(define a (cdr lp))
(define filtered (iir-filter b a signal))
```

**Source:** `lib/signal/filters.esk`, lines 324-362.

### 5.5 Highpass Design

#### `(butterworth-highpass order cutoff)` → pair (b-coeffs . a-coeffs)

Design an N-th order Butterworth highpass filter.

Uses the lowpass-to-highpass spectral transformation z → -z applied to a lowpass prototype
designed at the complementary frequency (1 - cutoff). This transformation negates the
odd-indexed b-coefficients:

```
b_hp[k] = (-1)^k * b_lp[k]
```

while the a-coefficients are shared with the lowpass design at the mirror frequency. The result
passes frequencies above `cutoff` and attenuates frequencies below it.

```scheme
(define hp (butterworth-highpass 2 0.5))  ; 2nd-order highpass above half-Nyquist
(define filtered (iir-filter (car hp) (cdr hp) signal))
```

**Source:** `lib/signal/filters.esk`, lines 366-383.

### 5.6 Bandpass Design

#### `(butterworth-bandpass order low-cutoff high-cutoff)` → list of two pairs

Design a bandpass filter by cascading a lowpass and a highpass:

- Lowpass at `high-cutoff` passes frequencies below the upper band edge.
- Highpass at `low-cutoff` passes frequencies above the lower band edge.

Returns a list of two filter pairs: `(list lp-pair hp-pair)`.

Apply both filters in sequence:

```scheme
(require signal.filters)

(define bp (butterworth-bandpass 3 0.1 0.4))   ; pass 0.1–0.4 Nyquist band
(define lp-pair (car bp))
(define hp-pair (cadr bp))
(define after-lp (iir-filter (car lp-pair) (cdr lp-pair) signal))
(define bandpassed (iir-filter (car hp-pair) (cdr hp-pair) after-lp))
```

**Note:** Cascading two N-th order filters yields a 2N-th order response, which may be
overly selective near the band edges. Reduce `order` accordingly.

**Source:** `lib/signal/filters.esk`, lines 388-392.

### 5.7 Frequency Response

#### `(frequency-response b-coeffs a-coeffs n-points)` → pair (magnitudes . phases)

Evaluate the filter's frequency response H(e^{jω}) at `n-points` equally spaced frequencies
from ω = 0 (DC) to ω = π (Nyquist):

```
          B(e^{jω})     sum_k b[k] * e^{-jωk}
H(e^jω) = --------- = -------------------------
          A(e^{jω})     sum_k a[k] * e^{-jωk}
```

The magnitude |H(e^jω)| and phase angle(H(e^jω)) are stored in separate vectors.

- Returns `(cons magnitudes phases)` where each is a vector of length `n-points`.
- `(car result)` — magnitudes (linear scale, not dB).
- `(cdr result)` — phases in radians, range [-π, π].

```scheme
(require signal.filters)

(define lp (butterworth-lowpass 4 0.25))
(define resp (frequency-response (car lp) (cdr lp) 512))
(define mags (car resp))
(define phases (cdr resp))

; Magnitude at DC (bin 0) should be ≈ 1.0
; Magnitude at Nyquist (bin 511) should be ≈ 0.0
; (vref mags 0)   → ~1.0
; (vref mags 511) → ~0.0
```

**Source:** `lib/signal/filters.esk`, lines 402-439.

---

## 6. Complete Pipeline Example

The following example demonstrates a full analysis and reconstruction pipeline: synthesizing a
two-tone signal, applying a window, computing the spectrum, designing and applying a bandpass
filter, and verifying the result via frequency response.

```scheme
(require signal)

;;; --- Signal Generation ---
;;; 256-sample signal at normalized frequencies 0.1 and 0.4
(define N 256)
(define signal (make-vector N 0.0))

(letrec ((gen (lambda (i)
          (if (< i N)
              (begin
                (vector-set! signal i
                  (+ (* 1.0 (sin (* 2.0 3.14159 0.1 i)))    ; 0.1 * Nyquist tone
                     (* 0.5 (sin (* 2.0 3.14159 0.4 i)))))  ; 0.4 * Nyquist tone (quieter)
                (gen (+ i 1)))))))
  (gen 0))

;;; --- Windowing ---
;;; Apply Hann window to reduce spectral leakage before FFT
(define window (hann-window N))
(define windowed-signal (apply-window signal window))

;;; --- Spectral Analysis ---
;;; Compute FFT. Bins 0..N/2-1 cover DC to Nyquist.
;;; Bin k corresponds to frequency k/N * fs_normalized.
(define spectrum (fft windowed-signal))
;;; Bin 26 (≈ 0.1*256) should show the first tone.
;;; Bin 102 (≈ 0.4*256) should show the second tone.

;;; --- Bandpass Filtering ---
;;; Isolate the band 0.05–0.25 Nyquist to keep only the first tone.
(define bp-filters (butterworth-bandpass 4 0.05 0.25))
(define lp-pair (car bp-filters))
(define hp-pair (cadr bp-filters))

(define after-lp (iir-filter (car lp-pair) (cdr lp-pair) signal))
(define filtered  (iir-filter (car hp-pair) (cdr hp-pair) after-lp))

;;; --- Frequency Response Check ---
;;; Verify the lowpass stage is well-behaved.
(define resp (frequency-response (car lp-pair) (cdr lp-pair) 128))
(define mags (car resp))
;;; (vref mags 0)  should be ≈ 1.0 (DC passes)
;;; (vref mags 64) should be ≈ 0.0 (Nyquist is attenuated)

;;; --- Spectral Reconstruction (Synthesis) ---
;;; Inverse FFT to reconstruct time-domain signal from modified spectrum.
(define reconstructed (ifft spectrum))
;;; Take real parts for a pure real output:
(define reconstructed-real
  (let ((result (make-vector N 0.0)))
    (letrec ((loop (lambda (i)
              (if (< i N)
                  (begin
                    (vector-set! result i (real-part (vref reconstructed i)))
                    (loop (+ i 1)))))))
      (loop 0))
    result))
```

---

## 7. Performance Considerations

### 7.1 FFT Input Size

The FFT implementation requires inputs of length exactly 2^k. If your signal is not a power of
2 in length, zero-pad it to the next power of 2 using `zero-pad-vector` (internal) or manually:

```scheme
(define (pad-to-pow2 v)
  (let* ((n (vector-length v))
         (m (let loop ((p 1)) (if (>= p n) p (loop (* p 2)))))
         (result (make-vector m 0.0)))
    (letrec ((copy (lambda (i)
              (if (< i n)
                  (begin (vector-set! result i (vref v i)) (copy (+ i 1)))))))
      (copy 0))
    result))
```

### 7.2 Direct vs FFT Convolution Crossover

Based on the asymptotic analysis, `fast-convolve` outperforms `convolve` when:

```
N_a * N_b > (N_a + N_b) * log2(N_a + N_b)
```

In practice:
- For filter kernels up to length ~32 applied to long signals, direct convolution is often
  faster due to FFT overhead (allocation, twiddle computation).
- For both sequences of length ≥ 64, `fast-convolve` is recommended.
- For real-time block processing of long audio, an overlap-add or overlap-save algorithm built
  on top of `fast-convolve` would give the best throughput, though that is not currently
  implemented in the stdlib.

### 7.3 IIR vs FIR Filtering

When using Butterworth designs via `butterworth-lowpass` / `butterworth-highpass`:

- **IIR (`iir-filter`):** O(N * (nb + na)) where nb, na are coefficient counts. For a 4th-order
  Butterworth, nb = na = 5, so cost is ~10 multiplies per sample.
- **FIR via `fir-filter`:** A linear-phase FIR with equivalent selectivity requires many more
  coefficients (typically 10-100x more), so IIR is much cheaper per sample for Butterworth
  designs.
- **FIR via `fast-convolve`:** For very long FIR filters (M ≥ 64), `fast-convolve` with the
  impulse response is more efficient than `fir-filter`.

### 7.4 Recursive FFT Allocation

The recursive Cooley-Tukey implementation allocates new vectors at each recursion level,
totaling O(N log N) allocation for an N-point FFT. For N = 1024 this is approximately
10 × 1024 = 10240 vector elements. Eshkol's arena allocator handles this efficiently;
allocation is not expected to be a bottleneck at moderate block sizes (N ≤ 65536).

For N > 65536 or repeated FFT calls in a real-time loop, consider pre-allocating working
buffers using tensors and reusing them across calls (not currently exposed in the stdlib API).

### 7.5 Kaiser Window Bessel Computation

The `bessel-i0` series runs for exactly 25 iterations per window sample. For a 1024-point
Kaiser window this is 25 × 1024 = 25600 multiply-add operations. This is a one-time setup
cost — the window vector is created once and reused via `apply-window`.

---

## 8. See Also

- `lib/signal/fft.esk` — FFT/IFFT source
- `lib/signal/filters.esk` — Window functions, convolution, FIR/IIR, Butterworth source
- `docs/breakdown/README.md` — Module breakdown index
- `ESHKOL_V1_LANGUAGE_REFERENCE.md` — Complex number primitives (`make-rectangular`,
  `real-part`, `imag-part`, `magnitude`, `angle`)
- `lib/math/constants.esk` — Mathematical constants used across stdlib
- `benchmarks/matmul_bench.esk` — Example of tensor/vector operations at scale

**External references:**
- Cooley, J.W.; Tukey, J.W. (1965). "An algorithm for the machine calculation of complex
  Fourier series." *Mathematics of Computation* 19(90): 297–301.
- Kaiser, J.F. (1974). "Nonrecursive digital filter design using I0-sinh window function."
  *Proc. IEEE Int. Symp. Circuits and Systems.* 20–23.
- Parks, T.W.; Burrus, C.S. (1987). *Digital Filter Design.* Wiley-Interscience.
- Oppenheim, A.V.; Schafer, R.W. (2009). *Discrete-Time Signal Processing, 3rd ed.*
  Prentice Hall. Chapters 7–8 (filter design), Chapter 9 (DFT computation).
