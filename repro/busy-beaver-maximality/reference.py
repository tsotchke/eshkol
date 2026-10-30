"""Independent sparse-tape replay of EVERY two-state binary source table."""
from collections import Counter


def control_closure(table):
    pending, seen = [0], set()
    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        for read in (0, 1):
            next_state = table[2*state+read]//4
            if next_state == 2:
                return False
            pending.append(next_state)
    return True


def classify(machine_id):
    table = [(machine_id//12**i) % 12 for i in range(4)]
    if control_closure(table):
        return 'nonhalt', 0, 0
    state, head, ones = 0, 0, set()
    trace = [(state, head, frozenset(ones))]
    for steps in range(1, 17):
        instruction = table[2*state+int(head in ones)]
        if instruction % 2:
            ones.add(head)
        else:
            ones.discard(head)
        head += 1 if (instruction//2) % 2 else -1
        state = instruction//4
        if state == 2:
            return 'halt', steps, len(ones)
        for start, (before_state, before_head, before_ones) in enumerate(trace):
            if state != before_state:
                continue
            delta = head-before_head
            if not delta and ones == before_ones:
                return 'nonhalt', 0, 0
            if delta > 0:
                cut = min(item[1] for item in trace[start:])
                if {x for x in before_ones if x >= cut} == {x-delta for x in ones if x-delta >= cut}:
                    return 'nonhalt', 0, 0
            if delta < 0:
                cut = max(item[1] for item in trace[start:])
                if {x for x in before_ones if x <= cut} == {x-delta for x in ones if x-delta <= cut}:
                    return 'nonhalt', 0, 0
        trace.append((state, head, frozenset(ones)))
    raise AssertionError(f'Undecided two-state source {machine_id}; a timeout is not nonhalting')


def whole_class():
    counts, max_time, max_marks = Counter(), 0, 0
    for machine_id in range(12**4):
        status, time, marks = classify(machine_id)
        counts[status] += 1
        max_time, max_marks = max(max_time, time), max(max_marks, marks)
    assert dict(counts) == {'nonhalt': 10952, 'halt': 9784}
    assert (max_time, max_marks) == (6, 4)
    return {'machines': 20736, 'halts': counts['halt'], 'nonhalts': counts['nonhalt'],
            'time': max_time, 'marks': max_marks}


if __name__ == '__main__':
    print(whole_class())
