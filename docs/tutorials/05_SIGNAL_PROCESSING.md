# Tutorial 5: Signal Processing

Eshkol includes a complete signal processing library — FFT, window
functions, and digital filter design — all integrated with the autodiff
system so you can differentiate through signal processing chains.

---

## Part 1: FFT and Inverse FFT

```scheme
;; Create a signal: 3 Hz + 7 Hz sine waves, 32 samples
(define (make-signal n sample-rate)
  (let ((dt (/ 1.0 sample-rate)))
    (let loop ((i 0) (result '()))
      (if (= i n) (reverse result)
          (let ((t (* i dt)))
            (loop (+ i 1)
                  (cons (+ (sin (* 2.0 3.14159 3.0 t))
                           (* 0.5 (sin (* 2.0 3.14159 7.0 t))))
                        result)))))))

;; fft, ifft and the filters below work on vectors
(define signal (list->vector (make-signal 32 32.0)))

;; Compute FFT
(define spectrum (fft signal))
(display "Spectrum computed")
(newline)

;; Inverse FFT recovers the original signal
(define recovered (ifft spectrum))
```

---

## Part 2: Window Functions

Window functions reduce spectral leakage when analysing finite-length
signals. Four are built in:

```scheme
;; Generate a 64-sample window
(define ham (hamming-window 64))
(define han (hann-window 64))
(define blk (blackman-window 64))
(define kai (kaiser-window 64 8.0))  ;; beta parameter

;; Apply a window to a signal before FFT: apply-window multiplies the two
;; vectors element by element
(define windowed (apply-window signal (hamming-window 32)))
(define windowed-spectrum (fft windowed))
```

| Function | Sidelobe level | Main lobe width |
|---|---|---|
| `hamming-window` | -43 dB | Moderate |
| `hann-window` | -31 dB | Moderate |
| `blackman-window` | -58 dB | Wide |
| `kaiser-window` | Adjustable (beta) | Adjustable |

---

## Part 3: FIR Filters

Design and apply finite impulse response filters:

```scheme
;; Create a simple moving-average FIR filter (5-tap)
(define coeffs #(0.2 0.2 0.2 0.2 0.2))

;; Apply FIR filter to a signal
(define filtered (fir-filter coeffs signal))
(display filtered)
(newline)
```

### Convolution

```scheme
;; General convolution of two sequences
(define result (convolve signal coeffs))
```

---

## Part 4: IIR / Butterworth Filters

Design classic Butterworth filters for lowpass, highpass, or bandpass:

```scheme
;; Design a 4th-order Butterworth lowpass filter.
;; Cutoffs are normalised so that 1.0 is the Nyquist frequency: a 100 Hz
;; cutoff at a 1000 Hz sample rate (Nyquist 500 Hz) is 100/500 = 0.2.
(define lp (butterworth-lowpass 4 0.2))

;; The design is a pair (b-coeffs . a-coeffs); apply it with iir-filter
(define smoothed (iir-filter (car lp) (cdr lp) signal))
(display (vector-length smoothed))   ;; => 32
```

The filter design uses the bilinear transform from the analog prototype,
computing poles on the Butterworth circle and mapping to the z-plane.

---

## Part 5: Integration with Autodiff

All signal processing functions compose with Eshkol's autodiff:

```scheme
;; How does the output power change with respect to filter cutoff?
;; This is differentiable because the filter design and the filter are
;; smooth operations on floats.
(define (signal-power cutoff)
  (let* ((lp (butterworth-lowpass 2 cutoff))
         (filtered (iir-filter (car lp) (cdr lp) signal)))
    ;; power of the filtered signal
    (let loop ((i 0) (power 0.0))
      (if (= i (vector-length filtered))
          power
          (loop (+ i 1)
                (+ power (* (vector-ref filtered i) (vector-ref filtered i))))))))

;; Gradient of power with respect to the (normalised) cutoff frequency
(display (derivative signal-power 0.2))
;; => 73.81...
```

---

## Builtin Reference

| Function | Description |
|---|---|
| `fft` | Fast Fourier Transform (Cooley-Tukey radix-2) |
| `ifft` | Inverse FFT |
| `hamming-window` | Hamming window function |
| `hann-window` | Hann (raised cosine) window |
| `blackman-window` | Blackman window |
| `kaiser-window` | Kaiser window (adjustable beta) |
| `fir-filter` | Apply FIR filter coefficients to signal |
| `convolve` | General convolution of two sequences |
| `butterworth-lowpass` | Butterworth lowpass filter design |
| `butterworth-highpass` | Butterworth highpass filter design |
| `butterworth-bandpass` | Butterworth bandpass filter design |

---

*Next: Tutorial 6 — Exact Arithmetic*
