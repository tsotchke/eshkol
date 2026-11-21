"""Steane fault pairing and dense replay from the Q2 consumer."""
import itertools
from . import pauli

def decoder(n, checks, cert):
    bits, inner = pauli.bits, pauli.inner
    table = {tuple(s): bits(p, n) for s, p in cert[3]}
    cb = [bits(c, n) for c in checks]
    zero = (0, 0, 0)
    def syndrome(e):
        return tuple(inner(c, e, n) for c in cb)

    def complete(s):
        # Independent CSS corrections from the X-check and Z-check syndromes.
        return [a ^ b for a, b in zip(table[s[:3] + zero], table[zero + s[3:]])]

    for s in itertools.product(range(2), repeat=6):
        if syndrome(complete(s)) != s or (s in table and complete(s) != table[s]):
            raise ValueError('CSS extension does not agree with the frozen decoder')
    lx, lz = bits('X' * n, n), bits('Z' * n, n)
    def decode(word):
        error = bits(word, n)
        s = syndrome(error)
        residual = [a ^ b for a, b in zip(error, complete(s))]
        if any(syndrome(residual)):
            raise ValueError('nonzero residual syndrome')
        logical = 'IZXY'[2 * inner(residual, lz, n) + inner(residual, lx, n)]
        return dict(syndrome=''.join(map(str, s)), frozen=logical if s in table else 'uncovered',
                    logical=logical, propagated=word)

    return decode, complete


def dense_gate(state, gate):
    # Integer amplitudes with a common denominator. Each H contributes sqrt(2).
    control = 1 << (6 - gate[1])
    if gate[0] == 'H':
        return [state[i & ~control] + (-1 if i & control else 1) * state[i | control]
                for i in range(128)]
    target = 1 << (6 - gate[2])
    return [state[i ^ target] if i & control else state[i] for i in range(128)]


def dense_pauli(state, bits):
    # XZ acts up to a constant phase for Y. All measured observables are
    # pure X or pure Z, so their eigenvalue signs are preserved exactly.
    x = sum(bit << (6 - q) for q, bit in enumerate(bits[:7]))
    z = sum(bit << (6 - q) for q, bit in enumerate(bits[7:]))
    return [(-1 if ((i ^ x) & z).bit_count() % 2 else 1) * state[i ^ x] for i in range(128)]


def eigenbit(state, observable):
    image = dense_pauli(state, observable)
    if image == state:
        return 0
    if image == [-x for x in state]:
        return 1
    raise ValueError('fault output outside a definite Pauli eigenspace')


def outcomes(gates, order, decode, complete, checks):
    rows = {}
    observables = [pauli.bits(c, 7) for c in checks]
    logical_x = pauli.bits('XXXXXXX', 7)
    prefix = [1] + [0] * 127
    for pos, gate_id in enumerate(order):
        gate = gates[gate_id]
        prefix = dense_gate(prefix, gate)
        wires = gate[1:]
        for letters in itertools.product('IXYZ', repeat=len(wires)):
            if all(p == 'I' for p in letters):
                continue
            word = list('IIIIIII')
            for q, p in zip(wires, letters):
                word[q] = p
            image = (1, ''.join(word))
            state = dense_pauli(prefix, pauli.bits(image[1], 7))
            for successor in order[pos + 1:]:
                image = pauli.conjugate(image, gates[successor], 7)
                state = dense_gate(state, gates[successor])
            prediction = decode(image[1])
            syndrome = tuple(eigenbit(state, c) for c in observables)
            recovered = dense_pauli(state, complete(syndrome))
            if (prediction['syndrome'] != ''.join(map(str, syndrome)) or
                    eigenbit(recovered, logical_x) != (prediction['logical'] in ('Y', 'Z'))):
                raise ValueError(f'dense replay disagrees at gate {gate_id}, fault {letters}')
            rows[gate_id, ''.join(letters)] = prediction
    return rows
