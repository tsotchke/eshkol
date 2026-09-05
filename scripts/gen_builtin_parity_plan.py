#!/usr/bin/env python3
"""Turn lib/backend/xla/device_lowering_table.yaml into a parity plan.

WHAT THIS DOES, AND WHY IT IS A SEPARATE PROGRAM.

For each builtin in the table it

  1. emits one Eshkol program that applies the REAL builtin to the table's
     inputs and prints the results, and runs it through eshkol-run with the
     device switch OFF, and
  2. writes a plan file pairing those printed values with the builtin's
     StableHLO lowering, for tests/xla/builtin_parity_test to execute on the
     device and compare against.

The reference therefore comes out of the language's own compiled
implementation, and the device value out of a StableHLO graph. Those are two
independent code paths. That separation is the whole point: a reference
computed by evaluating the `lowering` in double precision here would agree
with the device however wrong both were, and the row would be worthless.

It is a separate program from the harness because the harness has to link
MLIR, PJRT and the whole XLA backend, and none of that has any business
parsing YAML or shelling out to a compiler. It also means the plan can be
inspected, diffed and re-run by hand when a row disagrees.

Output (default .scratch/xla_gate/builtin_parity_plan.txt) is line-oriented on
purpose, so the C++ side needs no parser beyond `>>`:

    BUILTIN sqrt
    CLASS transcendental
    N 12
    IN 0 <v0> <v1> ...
    REF <r0> <r1> ...
    OP unary Sqrt r a0
    RESULT r
    END

A builtin whose reference program fails to compile or run is written with
NOREF and a reason instead of REF, and the harness reports it uncovered
rather than passing it.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(REPO, "lib", "backend", "xla", "device_lowering_table.yaml")


def load_table(path):
    """Parse the table.

    yaml is not in this repo's test dependencies and this file's shape is
    fixed and small, so it is read directly rather than adding an import that
    would make the gate fail on a host without it.
    """
    try:
        import yaml  # noqa
        with open(path) as f:
            return yaml.safe_load(f)["builtins"]
    except ImportError:
        pass

    # Minimal reader for exactly the shape this file has.
    builtins, cur, cur_name, section = {}, None, None, None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            body = line.strip()
            if indent == 0:
                continue
            if indent == 2 and body.endswith(":"):
                cur_name = body[:-1].strip().strip('"')
                cur = {"inputs": [], "lowering": []}
                builtins[cur_name] = cur
                section = None
                continue
            if cur is None:
                continue
            if indent == 4 and body.startswith("- "):
                item = parse_inline(body[2:])
                if section == "inputs":
                    cur["inputs"].append(item)
                elif section == "lowering":
                    cur["lowering"].append(item)
                continue
            if indent == 4:
                key, _, val = body.partition(":")
                key, val = key.strip(), val.strip()
                if key in ("inputs", "lowering"):
                    section = key
                    if val:
                        cur[key] = [parse_inline(x) for x in split_flow(val)]
                        section = None
                else:
                    section = None
                    cur[key] = val.strip('"')
    return builtins


def split_flow(val):
    """Split `[{...}, {...}]` into its `{...}` pieces."""
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        val = val[1:-1]
    out, depth, cur = [], 0, ""
    for ch in val:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_inline(text):
    """Parse a `{k: v, k: v}` flow mapping, dropping any trailing comment."""
    text = text.strip()
    if "#" in text and "}" in text:
        text = text[: text.rindex("}") + 1]
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    item = {}
    for part in text.split(","):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        item[k.strip()] = v.strip().strip('"')
    return item


def intlist(v, default=""):
    """Normalise an int list that may arrive as a YAML list or as raw text.

    load_table uses PyYAML when it is importable and a minimal reader when it
    is not, and the two hand back different Python types for `axes: [0]` — a
    list from one, the string "[0]" from the other. Formatting either directly
    produced "[0]" in the plan, which the harness then read as no axes at all.
    Both shapes normalise here, once, so the plan does not depend on which
    reader ran.
    """
    if v is None:
        v = default
    if isinstance(v, (list, tuple)):
        return " ".join(str(int(x)) for x in v)
    text = str(v).replace("[", " ").replace("]", " ").replace(",", " ")
    return " ".join(t for t in text.split() if t)


def values(spec):
    n = int(spec["count"])
    base = float(spec["base"])
    step = float(spec["step"])
    return [base + i * step for i in range(n)]


def fmt(x):
    return repr(float(x)) if x == x else "nan"


def esk_int_literal(x):
    """An Eshkol integer literal.

    An integer builtin must not be handed `12.0`: the numeric tower would
    dispatch it as a flonum and the reference would answer a different
    question from the one the integer lowering asks.
    """
    return str(int(round(float(x))))


def esk_literal(x):
    """An Eshkol float literal. `1.0` must not become `1`, or the builtin may
    take an integer overload and answer a different question."""
    s = repr(float(x))
    if "e" in s or "E" in s:
        return s
    if "." not in s:
        s += ".0"
    return s


def build_program(builtins):
    """One Eshkol program covering every builtin, tagged per line.

    One program rather than one per builtin because each eshkol-run invocation
    pays a full compile; forty of them turns a ten-second step into minutes on
    every gate run.
    """
    lines = []
    for name, spec in builtins.items():
        ins = [values(s) for s in spec["inputs"]]
        n = len(ins[0])
        call = name
        if spec.get("kind") == "tensor":
            sym = re.sub(r"\W", "_", name)
            operands = []
            for k, vs in enumerate(ins):
                args = " ".join(esk_literal(v) for v in vs)
                lines.append('(define T%d_%s (reshape (vector %s) 1 %d))'
                             % (k, sym, args, n))
                operands.append("T%d_%s" % (k, sym))
            extra = spec.get("extra_args", "")
            if isinstance(extra, str):
                extra = [x for x in extra.split() if x]
            application = "(%s %s)" % (call, " ".join(operands + list(extra)))
            # A scalar-valued builtin prints one number; a tensor-valued one
            # prints its elements. Asking for tensor-data of a scalar would
            # fail, and printing a tensor without it would print a handle.
            if spec.get("result") is not None and spec.get("result_shape") == "scalar":
                lines.append('(display "@%s ") (display %s) (newline)' % (name, application))
            else:
                lines.append('(display "@%s ") (display (tensor-data %s)) (newline)'
                             % (name, application))
        else:
            lit = esk_int_literal if spec.get("etype") == "s64" else esk_literal
            predicate = spec.get("reference_wrap") == "predicate"
            parts = ['(display "@%s ")' % name]
            for i in range(n):
                argv = " ".join(lit(ins[k][i]) for k in range(len(ins)))
                app = "(%s %s)" % (call, argv)
                # A predicate answers #t/#f, which is not a number. Both sides
                # map the boolean to a number: the host with an `if` here, the
                # device with a convert from i1. Two different routes to the
                # same encoding, which is what keeps the comparison honest.
                if predicate:
                    app = "(if %s 1 0)" % app
                parts.append('(display %s) (display " ")' % app)
            parts.append("(newline)")
            lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


NUM = re.compile(r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf)")


def parse_refs(stdout):
    """Collect `@name v v v` lines, tolerating the `#(...)` vector form."""
    refs = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("@"):
            continue
        name, _, rest = line[1:].partition(" ")
        rest = rest.replace("#(", " ").replace(")", " ")
        refs[name] = [float(m.group(0)) for m in NUM.finditer(rest)]
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eshkol-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True,
                    help="durable scratch directory for the generated program")
    args = ap.parse_args()

    builtins = load_table(TABLE)
    if not builtins:
        print("gen_builtin_parity_plan: no builtins parsed from %s" % TABLE,
              file=sys.stderr)
        return 2

    os.makedirs(args.work, exist_ok=True)
    prog = os.path.join(args.work, "builtin_reference.esk")
    with open(prog, "w") as f:
        f.write(build_program(builtins))

    # Device execution OFF: this run is the HOST reference, and if the switch
    # leaked in the two sides would be the same computation.
    env = dict(os.environ)
    env.pop("ESHKOL_XLA_PJRT", None)
    env["ESHKOL_XLA_DEVICE"] = "0"
    proc = subprocess.run([args.eshkol_run, "-r", prog],
                          capture_output=True, text=True, env=env, timeout=1800)
    with open(os.path.join(args.work, "builtin_reference.log"), "w") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    refs = parse_refs(proc.stdout)

    with open(args.out, "w") as f:
        for name, spec in builtins.items():
            ins = [values(s) for s in spec["inputs"]]
            n = len(ins[0])
            f.write("BUILTIN %s\n" % name)
            f.write("CLASS %s\n" % spec.get("tolerance_class", "arithmetic"))
            f.write("N %d\n" % n)
            f.write("RSHAPE %s\n" % ("scalar" if spec.get("result_shape") == "scalar"
                                      else "vector"))
            f.write("ETYPE %s\n" % spec.get("etype", "f32"))
            for i, vs in enumerate(ins):
                f.write("IN %d %s\n" % (i, " ".join(fmt(v) for v in vs)))
            ref = refs.get(name)
            if ref is None:
                f.write("NOREF the reference program printed no @%s line "
                        "(the builtin did not compile or did not run)\n" % name)
            else:
                want = 1 if spec.get("result_shape") == "scalar" else n
                if len(ref) != want:
                    f.write("NOREF the reference printed %d values, expected %d\n"
                            % (len(ref), want))
                else:
                    f.write("REF %s\n" % " ".join(fmt(v) for v in ref))
            for step in spec["lowering"]:
                if step["op"] == "unary":
                    f.write("OP unary %s %s %s\n" % (step["kind"], step["out"], step["in"]))
                elif step["op"] == "binary":
                    f.write("OP binary %s %s %s %s\n"
                            % (step["kind"], step["out"], step["lhs"], step["rhs"]))
                elif step["op"] == "const":
                    f.write("OP const %s %s %s\n"
                            % (step["out"], fmt(step["value"]), step["like"]))
                elif step["op"] == "compare":
                    f.write("OP compare %s %s %s %s\n"
                            % (step["dir"], step["out"], step["lhs"], step["rhs"]))
                elif step["op"] == "convert":
                    f.write("OP convert %s %s %s\n"
                            % (step["to"], step["out"], step["in"]))
                elif step["op"] == "broadcast":
                    f.write("OP broadcast %s %s %s | %s\n"
                            % (step["out"], step["in"],
                               intlist(step.get("shape"), "0"),
                               intlist(step.get("dims"))))
                elif step["op"] == "reduce":
                    f.write("OP reduce %s %s %s %s\n"
                            % (step["kind"], step["out"], step["in"],
                               intlist(step.get("axes"), "0")))
                else:
                    f.write("NOREF unknown lowering step '%s'\n" % step["op"])
            f.write("RESULT %s\n" % spec["result"])
            f.write("END\n")

    missing = [n for n in builtins if n not in refs]
    print("plan: %d builtins, %d with a reference, %d without%s"
          % (len(builtins), len(builtins) - len(missing), len(missing),
             (": " + " ".join(sorted(missing))) if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
