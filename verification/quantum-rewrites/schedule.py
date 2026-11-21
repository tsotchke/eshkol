"""Independent schedule validation for the H/CNOT Steane preparation."""
from collections import Counter


def commute_on_basis(first, second):
    # Normalize Hadamards after a whole pair. Both orders have the same
    # denominator, so signed integer numerators decide exact commutation.
    wires = sorted(set(first[1:]) | set(second[1:]))

    compact = [[op, *(wires.index(q) for q in qs)] for op, *qs in (first, second)]
    return all(exact_h_cnot_column(len(wires), compact, i) ==
               exact_h_cnot_column(len(wires), compact[::-1], i)
               for i in range(1 << len(wires)))


def exact_h_cnot_column(n, tape, index):
    if not 0 <= index < (1 << n):
        raise ValueError("basis index exceeds circuit width")
    state = {index: 1}
    for op, *qs in tape:
        masks = [1 << q for q in qs]
        output = Counter()
        for basis, weight in state.items():
            if op == 'CNOT':
                output[basis ^ masks[1] if basis & masks[0] else basis] += weight
            elif op == 'H':
                low = basis & ~masks[0]
                output[low] += weight
                output[low | masks[0]] += -weight if basis & masks[0] else weight
            else:
                raise ValueError(f'unsupported Steane gate {op}')
        state = output
    return state


def schedule_cost(n, original, order):
    if sorted(order) != list(range(len(original))):
        raise ValueError('schedule must contain each source instruction exactly once')
    ready = [0] * n
    layers = {}
    occupied = set()
    counts = Counter()
    for i in order:
        gate, *wires = original[i]
        if gate not in ('H', 'CNOT') or len(wires) != (1 if gate == 'H' else 2):
            raise ValueError('unrecognized instruction or arity')
        if len(set(wires)) != len(wires) or any(type(q) is not int or not 0 <= q < n for q in wires):
            raise ValueError('invalid wire binding')
        finish = 1 + max(ready[q] for q in wires)
        for q in wires:
            if (finish, q) in occupied:
                raise ValueError('wire conflict within schedule layer')
            occupied.add((finish, q))
            ready[q] = finish
        layers[i] = finish
        counts[gate] += 1
    for i, first in enumerate(original):
        for j in range(i + 1, len(original)):
            if layers[i] >= layers[j] and set(first[1:]).intersection(original[j][1:]):
                if not commute_on_basis(first, original[j]):
                    raise ValueError(f'violated noncommuting dependency {i} -> {j}')
    return {'CNOT': counts['CNOT'], 'H': counts['H'], 'T': counts['T'], 'depth': max(ready),
            'data_wires': n, 'additional_ancillas': 0}
