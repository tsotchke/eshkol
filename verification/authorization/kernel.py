"""Sequential decision and release primitives translated by extract_kernel.py."""


def decide(needs, check):
    if not needs:
        return "inconclusive"
    for need in needs:
        verdict = check(need)
        if verdict != "checked":
            return verdict
    return "authorized"


def release(issued, current, payload, expected):
    return payload if issued == current and payload == expected else None
