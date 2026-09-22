#!/usr/bin/env python3
"""Admitted constructor null-dominance checks, with rejection mutants.

Only vector allocations inside the four named arithmetic fallback helpers are
excluded: general numeric/tensor allocation is outside this constructor scope.
Cons calls and every other matching allocator site still require a null guard.
"""
import re
import sys

NAME = r"[-a-zA-Z$._0-9]+"
ALLOCATORS = r"arena_allocate_(?:vector_with_header|cons_with_header|cons_cell)"


def verify(text):
    count = 0
    excluded = {}
    for function in re.findall(r"^define .*?^}", text, re.M | re.S):
        arithmetic = re.search(r"@(__eshkol_arith_(?:add|sub|mul|div))\(", function.splitlines()[0])
        blocks = {}
        name = "entry"
        for line in function.splitlines()[1:-1]:
            label = re.match(r"^(" + NAME + r"):", line)
            if label:
                name = label[1]
            blocks.setdefault(name, []).append(line)
        predecessors = {name: set() for name in blocks}
        for name, lines in blocks.items():
            for target in re.findall(r"label %(" + NAME + r")", "\n".join(lines)):
                if target in blocks:
                    predecessors[target].add(name)
        entry = next(iter(blocks))
        dom = {name: ({entry} if name == entry else set(blocks)) for name in blocks}
        changed = True
        while changed:
            changed = False
            for name in blocks:
                if name == entry:
                    continue
                parents = predecessors[name]
                new = {name} | (set.intersection(*(dom[p] for p in parents)) if parents else set())
                if new != dom[name]:
                    dom[name] = new
                    changed = True
        for name, lines in blocks.items():
            body = "\n".join(lines)
            for call in re.finditer(r"(%" + NAME + r") = call ptr @" + ALLOCATORS + r"\([^\n]*", body):
                if arithmetic and "@arena_allocate_vector_with_header(" in call[0]:
                    excluded[arithmetic[1]] = excluded.get(arithmetic[1], 0) + 1
                    continue
                count += 1
                pointer = call[1]
                comparison = re.search(r"(%" + NAME + r") = icmp ne ptr " +
                                       re.escape(pointer) + r", null", body[call.end():])
                assert comparison, f"missing constructor null check in {name}: {call[0]}"
                branch = re.search(r"br i1 " + re.escape(comparison[1]) +
                                   r", label %(" + NAME + r"), label %(" + NAME + r")", body)
                assert branch, "missing constructor success/failure branch"
                success, failure = branch.groups()
                failure_body = "\n".join(blocks[failure])
                assert "call void @eshkol_runtime_emergency_raise_v1(i32 5)" in failure_body
                assert re.search(r"^\s*unreachable\s*$", failure_body, re.M)
                token = re.compile(re.escape(pointer) + r"(?![-a-zA-Z$._0-9])")
                for user_block, user_lines in blocks.items():
                    for line in user_lines:
                        if not token.search(line) or line.strip() == call[0].strip():
                            continue
                        if re.search(r"icmp ne ptr " + re.escape(pointer) + r", null", line):
                            continue
                        assert success in dom[user_block], "constructor pointer used without success dominance"
    assert count, "no constructor calls found"
    return count, excluded


def main():
    text = open(sys.argv[1]).read()
    count, excluded = verify(text)
    branch = re.search(r"br i1 %" + NAME + r", label %(constructor_allocated(?:" + NAME +
                       r")?), label %(constructor_failed(?:" + NAME + r")?)", text)
    assert branch
    mutated = text[:branch.start()] + "br label %" + branch[1] + text[branch.end():]
    try:
        verify(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError("removed constructor branch accepted")
    call = re.search(r"(%" + NAME + r") = call ptr @" + ALLOCATORS, text)
    failure = re.search(r"^" + re.escape(branch[2]) + r":.*$", text, re.M)
    mutated = text[:failure.end()] + "\n  store i64 1, ptr " + call[1] + text[failure.end():]
    try:
        verify(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError("constructor failure-store mutant accepted")
    for helper, sites in sorted(excluded.items()):
        print(f"EXCLUDED {sites} vector allocator site(s) in {helper}: numeric allocation outside admitted constructor scope")
    print(f"PASS {count} admitted constructor CFG sites; removed-branch and failure-store mutants rejected")


if __name__ == "__main__":
    main()
