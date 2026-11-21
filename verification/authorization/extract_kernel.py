"""Translate the two side-effect-free control structures in kernel.py to Lean.

The accepted Python fragment uses immutable built-ins, a finite tuple of needs
and a synchronous checker callback. Reject any unrecognized syntax.
"""

import ast
from pathlib import Path


def expect(ok):
    if not ok:
        raise ValueError("outside the admitted Python gate fragment")


def name(node, allowed):
    expect(isinstance(node, ast.Name) and node.id in allowed)
    return node.id


def status(node):
    if isinstance(node, ast.Name):
        return name(node, {"verdict"})
    expect(
        isinstance(node, ast.Constant)
        and node.value in ("checked", "authorized", "denied", "inconclusive")
    )
    return "Verdict." + node.value


def returned(body):
    expect(len(body) == 1 and isinstance(body[0], ast.Return))
    return body[0].value


def comparison(node, operand):
    expect(isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1)
    operators = {ast.Eq: "=", ast.NotEq: "≠"}
    expect(type(node.ops[0]) in operators)
    return f"({operand(node.left)} {operators[type(node.ops[0])]} {operand(node.comparators[0])})"


def decision(body):
    guard, loop, end = body
    expect(
        isinstance(guard, ast.If)
        and not guard.orelse
        and isinstance(guard.test, ast.UnaryOp)
        and isinstance(guard.test.op, ast.Not)
        and name(guard.test.operand, {"needs"})
    )
    expect(
        isinstance(loop, ast.For)
        and not loop.orelse
        and name(loop.iter, {"needs"})
        and name(loop.target, {"need"})
    )
    assign, branch = loop.body
    expect(
        isinstance(assign, ast.Assign)
        and len(assign.targets) == 1
        and name(assign.targets[0], {"verdict"})
        and isinstance(assign.value, ast.Call)
    )
    call = assign.value
    expect(
        name(call.func, {"check"})
        and len(call.args) == 1
        and not call.keywords
        and name(call.args[0], {"need"})
    )
    expect(isinstance(branch, ast.If) and not branch.orelse)
    test = comparison(branch.test, status)
    # findSome? runs the callback from left to right and stops at the first return.
    return f"""def pyDecide (needs : List N) (check : N → Verdict) : Verdict :=
  if needs.isEmpty then {status(returned(guard.body))}
  else (needs.findSome? fun need =>
    let verdict := check need
    if {test} then some {status(returned(branch.body))} else none).getD {status(returned([end]))}
"""


def release(body):
    expr = returned(body)
    expect(isinstance(expr, ast.IfExp) and isinstance(expr.test, ast.BoolOp))
    joins = {ast.And: " ∧ ", ast.Or: " ∨ "}
    expect(type(expr.test.op) in joins)

    def variable(node):
        return name(node, {"issued", "current", "payload", "expected"})

    def result(node):
        if isinstance(node, ast.Constant) and node.value is None:
            return "none"
        return "some " + variable(node)

    test = joins[type(expr.test.op)].join(comparison(test, variable) for test in expr.test.values)
    return f"""def pyRelease (issued current payload expected : List UInt8) : Option (List UInt8) :=
  if {test} then {result(expr.body)} else {result(expr.orelse)}
"""


def translate(source):
    doc, decide, handoff = ast.parse(source).body
    expect(
        isinstance(doc, ast.Expr)
        and isinstance(doc.value, ast.Constant)
        and isinstance(doc.value.value, str)
    )
    for fn, expected in (
        (decide, "def decide(needs, check): pass"),
        (handoff, "def release(issued, current, payload, expected): pass"),
    ):
        header = ast.parse(expected).body[0]
        expect(
            isinstance(fn, ast.FunctionDef)
            and fn.name == header.name
            and not fn.decorator_list
            and fn.returns is None
            and not getattr(fn, "type_params", [])
            and ast.dump(fn.args) == ast.dump(header.args)
        )
    return (
        "import Authority\nnamespace Verificatum.Authority\n"
        + decision(decide.body)
        + release(handoff.body)
        + "end Verificatum.Authority\n"
    )


if __name__ == "__main__":
    print(translate(Path(__file__).with_name("kernel.py").read_text()), end="")
