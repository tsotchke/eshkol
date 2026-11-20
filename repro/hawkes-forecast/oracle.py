"""Independent full-source polynomial forecast; never consumes reduced matrices."""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import importlib.util


def source_tools(root):
    path = Path(root)/'scripts/run_observable_closure_gate.py'
    spec = importlib.util.spec_from_file_location('closure_source_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add(a, b):
    out = list(a)+[F(0)]*max(0, len(b)-len(a))
    for i, x in enumerate(b):
        out[i] += x
    while len(out)>1 and not out[-1]:
        out.pop()
    return out


def scale(a, c):
    return [x*c for x in a]


def mul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def taylor(polynomial, center):
    return [sum((c*comb(i, order)*center**(i-order) for i, c in enumerate(polynomial) if i>=order), F(0))
            for order in range(3)]


def full_forecast(root, source, point, center, h=F(1, 16), steps=1):
    tools = source_tools(root)
    family, matrices, queries = tools.source_oracle(source)
    specification = {row[0]: row[1:] for row in source}
    state = [[F(1)] for _ in specification['basis']]
    for i, powers in enumerate(specification['basis']):
        state[i] = [F(1)]
        for value, power in zip(point, powers):
            state[i][0] *= value**power

    def rhs(values):
        out = [[F(0)] for _ in values]
        for powers, matrix in zip(family, matrices):
            assert len(powers)==1
            for i, row in enumerate(matrix):
                total = [F(0)]
                for coefficient, value in zip(row, values):
                    total = add(total, scale(value, coefficient))
                out[i] = add(out[i], [F(0)]*powers[0]+total)
        return out

    def shift(values, derivatives, amount):
        return [add(v, scale(d, amount)) for v, d in zip(values, derivatives)]

    for _ in range(steps):
        k1 = rhs(state)
        k2 = rhs(shift(state, k1, h/2))
        k3 = rhs(shift(state, k2, h/2))
        k4 = rhs(shift(state, k3, h))
        state = [add(v, scale(add(add(a, d), scale(add(b, c), 2)), h/6))
                 for v, a, b, c, d in zip(state, k1, k2, k3, k4)]
    polynomials = []
    for row in queries:
        p = [F(0)]
        for coefficient, value in zip(row, state):
            p = add(p, scale(value, coefficient))
        polynomials.append(p)
    covariance = add(polynomials[2], scale(mul(polynomials[0], polynomials[1]), -1)) if len(polynomials)==3 else [F(0)]
    return [taylor(p, center) for p in polynomials], taylor(covariance, center), polynomials
