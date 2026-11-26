"""Rational sine-rod reference enclosures from the P0 checker."""
import math
from fractions import Fraction as F


def require(ok, message):
    if not ok:
        raise ValueError(message)

def atan_bounds(q):
    # Alternating series at 1/q. Two consecutive partial sums enclose atan(1/q).
    s = sum(F((-1)**j, (2*j+1)*q**(2*j+1)) for j in range(12))
    return s, s + F(1, 25*q**25)


def sine_samples(n):
    a, b = atan_bounds(5)
    c, d = atan_bounds(239)
    lo, hi = 16*a-4*d, 16*b-4*c  # Machin's identity
    require(hi < F(22, 7), "pi bound")
    values, errors = [], []
    for i in range(n+1):
        x = min(i, n-i) * (lo+hi) / (2*n)
        approx = sum((-1)**j*x**(2*j+1)/math.factorial(2*j+1) for j in range(8))
        # Round the initial sample to an exact dyadic. |sin'| <= 1.
        sample = F(round(approx * 2**32), 2**32)
        error = abs(sample-approx) + x**17/math.factorial(17) + (hi-lo)/4
        values.append(sample)
        errors.append(error)
    # Return a compact rational upper bound (upward rounding).
    return values, F(math.ceil(max(errors)*2**48), 2**48)


def max_error(field):
    """Independently enclose exp(-pi²/50) sin(pi*i/n) at every returned cell."""
    pi = F(3141592653589793, 10**15)
    eps = F(1, 10**15)
    a, b = atan_bounds(5)
    c, d = atan_bounds(239)
    require(pi < 16*a-4*d <= 16*b-4*c < pi+eps, "analytic pi bracket")
    rate = pi**2/50
    decay = sum((-rate)**k/math.factorial(k) for k in range(15))
    decay_err = rate**15/math.factorial(15) + ((pi+eps)**2-pi**2)/50
    n = len(field)-1
    worst = (F(0), 0)
    for i, value in enumerate(field):
        x = pi*i/n
        wave = sum((-1)**k*x**(2*k+1)/math.factorial(2*k+1) for k in range(12))
        wave_err = x**25/math.factorial(25) + eps
        # |true sine| <= 1; decompose the product using the rational decay.
        err = abs(value-decay*wave) + decay_err + abs(decay)*wave_err
        worst = max(worst, (err, i))
    return worst
