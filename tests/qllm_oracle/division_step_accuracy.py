#!/usr/bin/env python3
"""Accuracy of the forward-jet quotient rule, before and after SW-222.

Run: python3 tests/qllm_oracle/division_step_accuracy.py (deterministic seed).
"""
# The first-order jet quotient q1 = d(a/b): the removed reciprocal chain
# (a * (1/b), g' = -1/b0^2) versus the division recurrence q1 = (a1 - b1 q0)/b0,
# both in binary64, measured against the exact rational value.
import random, math
from fractions import Fraction as Q
random.seed(20260922)
def ulperr(x, exact):
    return abs(Q(x) - exact) / Q(math.ulp(float(exact)))
old=[];new=[]
for _ in range(200000):
    a0,a1,b0,b1=[random.uniform(-4,4) for _ in range(4)]
    if abs(b0)<1e-3: continue
    exact=(Q(a1)*Q(b0)-Q(a0)*Q(b1))/(Q(b0)*Q(b0))
    if exact==0: continue
    inv=1.0/b0; inv2=inv*inv; fpa=-inv2; o1=fpa*b1
    q_old=a1*inv + a0*o1                       # a1*r0 + a0*r1, r = 1/b chain
    q0=a0/b0; q_new=(a1 - b1*q0)/b0            # the division recurrence
    old.append(float(ulperr(q_old,exact))); new.append(float(ulperr(q_new,exact)))
def stats(v):
    v=sorted(v); n=len(v); return (sum(v)/n, v[n//2], v[int(0.99*n)], v[-1])
print('n=%d' % len(old))
print('reciprocal chain   mean %.3f  median %.3f  p99 %.3f  max %.1f ulp' % stats(old))
print('division recurrence mean %.3f  median %.3f  p99 %.3f  max %.1f ulp' % stats(new))
print('new better %d, old better %d, equal %d' % (sum(n<o for n,o in zip(new,old)), sum(n>o for n,o in zip(new,old)), sum(n==o for n,o in zip(new,old))))
