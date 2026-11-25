"""Enumerate potential memory effects reachable by direct calls in dumped LLVM IR."""
import argparse
from hashlib import sha256
import re
from collections import Counter
from pathlib import Path
import sys


PREDICATES = {"blc-var?", "blc-lam?", "blc-app?"}
TAG = "__core_blc__blc-tagged?"
UNIFORM = TAG + "__eshkol_tail_uniform"

# Fail closed on new CFGs, SSA operands, call sites, or stores. Only the
# predicate body's own name and the symbol-literal label differ in the three
# reviewed protocols. Neither is a target in the thread-local transfer slot.
DRIVER = "e1b2d6f6ccc55080b378fff063607764a651087425787c859181f54b6120d8f6"
PRODUCER = "745146713540f72ff8854611a3546069f563a5b57e3949a5f11f3cba78a18df9"
RETURN_PATH = {
    UNIFORM: "9bbb81f80e0feb1ed049fb263a90d4eed447b2973b030c2b95dd0750d1effd0e",
    TAG + "__eshkol_tail_body": "45832247dda27051abc976d08e9d0680d41d8a8f20c68dbf5f867d6194091c23",
    TAG: "7a0e5592bba3fa6296c3a72588af998352112ae02d241b783ee70c75cf51cc8a",
}


def fingerprint(fn, pred=None):
    code = "\n".join(b.name + "\n" + "\n".join(str(i).strip() for i in b.instructions)
                     for b in fn.blocks)
    if pred:
        code = code.replace('@"' + pred + '__eshkol_tail_body"', '@PRED_BODY')
        code = re.sub(r'@\.sym_name(?:\.\d+)?\b', '@TAG_NAME', code)
    return sha256(code.encode()).hexdigest()


def transfer_targets(functions):
    # The producer publishes argc=2 and the uniform entry at offsets 4 and 8,
    # then pending=1 at offset 0. Driver clears pending before calling the
    # loaded target. The inspected return path never requests another transfer.
    # External callees must preserve the TLS record and return normally.
    if not all(k in functions and fingerprint(functions[k]) == v for k, v in RETURN_PATH.items()):
        return {}
    return {n: UNIFORM for n in PREDICATES if n in functions and n+"__eshkol_tail_body" in functions
            and fingerprint(functions[n], n) == DRIVER
            and fingerprint(functions[n+"__eshkol_tail_body"], n) == PRODUCER}


def sites(module, llvm):
    functions = {function.name: function for function in module.functions}
    targets = transfer_targets(functions)
    todo = ["blc-fast-run"]
    visited = set()
    rows = []
    while todo:
        name = todo.pop()
        if name in visited:
            continue
        visited.add(name)
        function = functions[name]
        if function.is_declaration:
            raise ValueError(f"no definition for {name}")
        for block_no, block in enumerate(function.blocks):
            for index, insn in enumerate(block.instructions):
                op = insn.opcode
                if op in {"call", "invoke", "callbr"}:
                    callee = list(insn.operands)[-1]
                    if callee.value_kind != llvm.ValueKind.function:
                        if name in targets and callee.name == "tt_target":
                            kind, detail = "conditional-transfer", targets[name]
                            todo.append(detail)
                        else:
                            kind, detail = "unresolved-call", callee.name or "<indirect>"
                    else:
                        target = functions[callee.name]
                        detail = target.name
                        if target.is_declaration:
                            kind = "external-call"
                        else:
                            kind = "defined-call"
                            todo.append(detail)
                elif op in {"alloca", "load", "store", "atomicrmw", "cmpxchg", "fence", "va_arg"}:
                    kind, detail = op, ""
                else:
                    continue
                rows.append((name, block_no, index, kind, detail, " ".join(str(insn).split())))
    return sorted(visited), sorted(rows)


def inspect(path, output, require_transfers=False):
    # Optional evidence path. The standard source and Lean gate has no LLVM
    # Python dependency; this path requires bindings that accept the runner's IR.
    from llvmlite import binding as llvm
    module = llvm.parse_assembly(path.read_text(), context=llvm.create_context())
    module.verify()
    functions, rows = sites(module, llvm)
    counts = Counter(row[3] for row in rows)
    if require_transfers:
        resolved = {row[0] for row in rows if row[3] == "conditional-transfer" and row[4] == UNIFORM}
        if resolved != PREDICATES or counts["unresolved-call"] or counts["conditional-transfer"] != 3:
            raise ValueError("predicate transfer contract is outside the reviewed fragment")
        # Change the producer while keeping a well-typed module and driver.
        # Its indirect call must immediately revert to unresolved.
        body = str(module.get_function("blc-var?__eshkol_tail_body"))
        store = f'store ptr @"{UNIFORM}"'
        if body.count(store) != 1:
            raise ValueError("missing predicate publication")
        changed = llvm.parse_assembly(str(module).replace(body, body.replace(store, "store ptr null")), context=llvm.create_context())
        changed.verify()
        remaining = transfer_targets({f.name: f for f in changed.functions})
        if set(remaining) != PREDICATES - {"blc-var?"}:
            raise ValueError("retargeted predicate producer escaped identification")
    if output is not None:
        print("function\tblock\tinstruction\tkind\tcallee\tinstruction text", file=output)
        for row in rows:
            print(*row, sep="\t", file=output)
    print(f"IR from blc-fast-run: {len(functions)} defined functions, "
          + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())), file=sys.stderr)
    external = Counter(row[4] for row in rows if row[3] == "external-call")
    print(f"{len(external)} distinct external callees; "
          f"{external['arena_tagged_cons_set_tagged_value']} tagged-slot setter call sites.",
          file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path)
    parser.add_argument("--summary", action="store_true", help="omit individual effect sites")
    args = parser.parse_args()
    inspect(args.ir, None if args.summary else sys.stdout)
