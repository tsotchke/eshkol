#!/usr/bin/env python3
"""Check emitted checked-store CFG and reject two branch/load mutants.

This checks explicit compiler IR before LLVM's optimization pipeline; executing
the separately optimized AOT fixture is a distinct requirement. It is not a
general LLVM verifier or a proof about arbitrary indirect memory aliases.
"""
import re
import sys

NAME = r"[-a-zA-Z$._0-9]+"


def verify(text):
    checked = 0
    for function in re.findall(r"^define .*?^}", text, re.M | re.S):
        blocks = {}
        block = "entry"
        for line in function.splitlines()[1:-1]:
            label = re.match(r"^(" + NAME + r"):", line)
            if label:
                block = label[1]
            blocks.setdefault(block, []).append(line)
        predecessors = {name: set() for name in blocks}
        for name, lines in blocks.items():
            for target in re.findall(r"label %(" + NAME + r")", "\n".join(lines)):
                if target in blocks:
                    predecessors[target].add(name)
        entry = next(iter(blocks))
        dominators = {name: ({entry} if name == entry else set(blocks)) for name in blocks}
        changed = True
        while changed:
            changed = False
            for name in blocks:
                if name == entry:
                    continue
                parents = predecessors[name]
                current = {name} | (set.intersection(*(dominators[p] for p in parents))
                                    if parents else set())
                if current != dominators[name]:
                    dominators[name] = current
                    changed = True
        for name, lines in blocks.items():
            body = "\n".join(lines)
            for call in re.finditer(r"(%" + NAME + r") = call i32 "
                                    r"@eshkol_region_write_barrier_checked_v1"
                                    r"\(ptr (%" + NAME + r"),", body):
                checked += 1
                status, output = call.groups()
                compare = re.search(r"(%" + NAME + r") = icmp eq i32 " +
                                    re.escape(status) + r", 0", body[call.end():])
                assert compare, "missing exact success status check"
                branch = re.search(r"br i1 " + re.escape(compare[1]) +
                                   r", label %(" + NAME + r"), label %(" + NAME + r")", body)
                assert branch, "missing checked success/failure branch"
                success, failure = branch.groups()
                failure_body = "\n".join(blocks[failure])
                assert re.search(r"call void @eshkol_runtime_emergency_raise_v1\(i32 " +
                                 re.escape(status) + r"\)", failure_body), "wrong emergency transfer"
                assert re.search(r"^\s*unreachable\s*$", failure_body, re.M), "failure can continue"
                loads = 0
                for load_block, load_lines in blocks.items():
                    for line in load_lines:
                        if re.search(r"\bload .*?, ptr " + re.escape(output) + r"(?:,|\s*$)", line):
                            loads += 1
                            assert success in dominators[load_block], "staging load lacks success dominance"
                assert loads, "checked staging output has no verified load"
    assert checked, "no checked barrier calls found"
    return checked


def main():
    text = open(sys.argv[1]).read()
    count = verify(text)
    # Mutant 1 removes the first checked branch, bypassing its status test.
    branch = re.search(r"br i1 %" + NAME + r", label %(wb_success" + NAME +
                       r"|wb_success), label %(wb_failure" + NAME + r"|wb_failure)", text)
    assert branch, "expected emitted wb_success/wb_failure labels"
    mutant = text[:branch.start()] + "br label %" + branch[1] + text[branch.end():]
    try:
        verify(mutant)
    except AssertionError:
        pass
    else:
        raise AssertionError("branch-removal mutant was accepted")
    # Mutant 2 injects an output load into the failure successor.
    call = re.search(r"@eshkol_region_write_barrier_checked_v1\(ptr (%" + NAME + r"),", text)
    label = re.search(r"^" + re.escape(branch[2]) + r":.*$", text, re.M)
    mutant = (text[:label.end()] + "\n  %mutant_early = load { i8, i8, i16, i64 }, ptr " +
              call[1] + text[label.end():])
    try:
        verify(mutant)
    except AssertionError:
        pass
    else:
        raise AssertionError("failure-output-load mutant was accepted")
    print(f"PASS {count} checked CFG sites; branch-removal and early-load mutants rejected")


if __name__ == "__main__":
    main()
