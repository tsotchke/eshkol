"""Enumerate reachable source-level pair operations for well-formed blc-fast calls."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys

from publication import tokens

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[1] / "lib/core"


def forms(text):
    stream = iter(tokens(text))

    def read(tok):
        if tok == "'":
            return ["quote", read(next(stream))]
        if tok == ")":
            raise ValueError("unexpected closing parenthesis")
        if tok != "(":
            return tok
        result = []
        for t in stream:
            if t == ")":
                return result
            result.append(read(t))
        raise ValueError("unclosed source form")

    return [read(tok) for tok in stream]


def render(expr):
    return "(" + " ".join(map(render, expr)) + ")" if isinstance(expr, list) else expr


# These are the operations for this module. Unknown primitives and indirect
# calls need an explicit effect classification before this gate will pass.
KINDS = {op: kind for kind, ops in {
    "allocate": ("cons", "list"),
    "read": ("car", "cdr", "cadr", "caddr", "cddr"),
    "inspect": ("pair?", "null?", "eq?"),
    "write": ("set-car!", "set-cdr!"),
    "register": ("set!",),
    "error": ("error",),
}.items() for op in ops}
PURE = {"+", "-", "=", ">=", "integer?", "not", "and", "if"}

# Match the operands, guards, counter increments and saved continuations of
# the source arms transcribed in Dispatch.lean. This is a small syntax reader.
APPLICATION = """((blc-app? item)
 (let ((cell (cons 'pending (cons (blc-app-arg item) env))))
   (set! suspensions (+ suspensions 1))
   (loop 'eval (blc-app-fun item) env (cons (cons 'argument cell) frames) (+ steps 1))))"""
FORCE = """((eq? mode 'force)
 (cond ((eq? (car item) 'pending) (set! forces (+ forces 1))
        (loop 'eval (cadr item) (cddr item) (cons (cons 'update item) frames) (+ steps 1)))
       (else (if (eq? (car item) 'memo) (set! hits (+ hits 1)))
             (loop 'value (cdr item) '() frames (+ steps 1)))))"""
ARGUMENT = """((eq? action 'argument)
 (if (and (pair? item) (eq? (car item) 'closure))
     (loop 'eval (cadr item) (cons data (cddr item)) rest (+ steps 1))
     (loop 'value (list 'spine item data) '() rest (+ steps 1))))"""


def inventory(fast, core):
    [require, provide, entry] = forms(fast)
    if (require != ["require", "core.blc"] or provide != ["provide", "blc-fast-run"]
            or entry[:2] != ["define", ["blc-fast-run", "term", "fuel"]]):
        raise ValueError("unexpected evaluator module boundary")
    definitions = {}
    for form in forms(core):
        if isinstance(form, list) and form[:1] == ["define"] and isinstance(form[1], list):
            name = form[1][0]
            if name in definitions:
                raise ValueError(f"ambiguous imported procedure {name}")
            definitions[name] = form
    rows = []
    visited = set()
    bound_arms = []
    named_loops = []

    def walk(expr, scope, variables):
        if not isinstance(expr, list):
            if (expr not in variables and not expr.startswith('"') and expr not in ("#t", "#f")
                    and not expr.lstrip("-").isdigit()):
                raise ValueError(f"unbound or procedure-valued operand {expr} in {scope}")
            return
        op, *args = expr
        if not isinstance(op, str) or (op in variables and op != "loop"):
            raise ValueError(f"indirect call in {scope}: {render(expr)}")
        if op == "quote":
            if isinstance(args[0], list) and args[0]:
                rows.append((scope, "literal-pair", render(expr)))
            return
        if op in ("let", "let*"):
            if args[0] == "loop":
                if scope != "blc-fast-run":
                    raise ValueError("unexpected named loop")
                named_loops.append("loop")
                args = args[1:]
            bindings, *body = args
            locals_ = set(variables)
            for name, init in bindings:
                if name == "loop":
                    raise ValueError("shadowed evaluator loop")
                walk(init, scope, locals_ if op == "let*" else variables)
                locals_.add(name)
            if op == "let" and expr[1] == "loop":
                locals_.add("loop")
            for e in body:
                walk(e, scope, locals_)
            return
        if op == "cond":
            for guard, *body in args:
                branch = scope
                if guard != "else":
                    walk(guard, scope, variables)
                    if isinstance(guard, list) and guard[:2] == ["eq?", "mode"]:
                        branch = "dispatch/" + guard[2][1]
                    elif scope == "blc-fast-run" and guard == ["=", "steps", "fuel"]:
                        branch = "exit/exhausted"
                    elif scope == "blc-fast-run" and guard[0] == "and":
                        branch = "exit/normal"
                elif scope == "blc-fast-run":
                    branch = "dispatch/emit"
                for name, parent, pattern in patterns:
                    if scope == parent and guard == pattern[0]:
                        if [guard, *body] != pattern:
                            raise ValueError(f"changed {name} dispatch arm")
                        bound_arms.append(name)
                for e in body:
                    walk(e, branch, variables)
            return
        if op == "loop":
            if op not in variables or scope.startswith("core.blc/"):
                raise ValueError("unbound evaluator loop")
        elif op in definitions:
            rows.append((scope, "import", render(expr)))
            if op not in visited:
                visited.add(op)
                _, (_, *params), *body = definitions[op]
                for e in body:
                    walk(e, "core.blc/" + op, set(params))
        elif op in KINDS:
            rows.append((scope, KINDS[op], render(expr)))
        elif op not in PURE:
            raise ValueError(f"unclassified operation {op} in {scope}")
        for e in args:
            walk(e, scope, variables)

    patterns = [(name, scope, forms(src)[0]) for name, scope, src in (
        ("application", "dispatch/eval", APPLICATION), ("force", "blc-fast-run", FORCE),
        ("argument", "dispatch/value", ARGUMENT))]
    for expr in entry[2:]:
        walk(expr, "blc-fast-run", {"term", "fuel"})
    if named_loops != ["loop"]:
        raise ValueError("missing or duplicated evaluator loop")
    if sorted(bound_arms) != ["application", "argument", "force"]:
        raise ValueError("missing or duplicated dispatch arm")
    lines = ["scope\teffect\tform\tsites"]
    for (scope, kind, form), n in sorted(Counter(rows).items()):
        lines.append(f"{scope}\t{kind}\t{form}\t{n}")
    return "\n".join(lines) + "\n"


# The flat-stack model transcribes this named let, including the order of
# terminal/fuel guards, the combined update and the quote/emit continuations.
# Lock the reviewed S-expression, ignoring comments and formatting.
def check_loop(source):
    loop = forms(source)[2][-1][-1]
    if sha256(render(loop).encode()).hexdigest() != "2803080bc6935e80c78a61dfdb9bcd519ca904477dd27e5069b2fda9b8da160c":
        raise ValueError("changed flat-stack dispatch contract")


def check(lib, lock):
    fast = (lib / "blc-fast.esk").read_text()
    core = (lib / "blc.esk").read_text()
    check_loop(fast)
    # Exchanging the two operands produces a different readback of a neutral
    # application. The source binding must reject it even before native replay.
    swapped = fast.replace("(blc-app (cdr frame) item)", "(blc-app item (cdr frame))")
    try:
        check_loop(swapped)
    except ValueError:
        pass
    else:
        raise ValueError("operand-swapped readback escaped the source contract")
    expected = lock.read_text()
    if inventory(fast, core) != expected:
        raise ValueError("BLC effect inventory changed")
    # A hidden write inside a transitive import must change the inventory.
    target = "(and (pair? t) (eq? (car t) tag))"
    if core.count(target) != 1:
        raise ValueError("changed imported effect mutation site")
    extra = core.replace(target, "(if (pair? t) (set-car! t tag) #f) " + target)
    if inventory(fast, extra) == expected:
        raise ValueError("unreported imported pair mutation")
    return sum(int(row.rsplit("\t", 1)[1]) for row in expected.splitlines()[1:])


if __name__ == "__main__":
    sys.stdout.write(inventory((HERE / "fixtures/core/blc-fast.esk").read_text(), (LIB / "blc.esk").read_text()))
