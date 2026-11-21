"""Finite, fail-closed translation table for the two reviewed gate functions."""

import ast
from pathlib import Path

# Review this grammar independently of the implementation being translated.
# The two operator choices at each site allow the proof's negative controls.
GRAMMAR = """
def decide(needs, check):
    if not needs:
        return "inconclusive"
    for need in needs:
        verdict = check(need)
        if verdict {comparison} "checked":
            return verdict
    return "authorized"

def release(issued, current, payload, expected):
    return payload if issued == current {join} payload == expected else None
"""


def translate(source):
    module = ast.parse(source, type_comments=True)
    # The module docstring and source locations have no bearing on either function.
    if not ast.get_docstring(module, clean=False):
        raise ValueError("gate module must carry its docstring")
    module.body.pop(0)
    actual = ast.dump(module, include_attributes=False)
    for cmp, relation in (("!=", "≠"), ("==", "=")):
        for join, connective in (("and", "∧"), ("or", "∨")):
            admitted = ast.parse(GRAMMAR.format(comparison=cmp, join=join), type_comments=True)
            if actual == ast.dump(admitted, include_attributes=False):
                return f"""import Authority
namespace Verificatum.Authority
def pyDecide (needs : List N) (check : N → Verdict) : Verdict :=
  if needs.isEmpty then Verdict.inconclusive
  else (needs.findSome? fun need =>
    let verdict := check need
    if (verdict {relation} Verdict.checked) then some verdict else none).getD Verdict.authorized
def pyRelease (issued current payload expected : List UInt8) : Option (List UInt8) :=
  if (issued = current) {connective} (payload = expected) then some payload else none
end Verificatum.Authority
"""
    raise ValueError("outside the reviewed gate grammar")


if __name__ == "__main__":
    print(translate(Path(__file__).with_name("kernel.py").read_text()), end="")
