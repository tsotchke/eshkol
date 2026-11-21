"""Observe one native Eshkol FTCS step on Linux/x86-64 with Valgrind Lackey."""
import argparse
from fractions import Fraction
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
P0 = HERE.parent / "heat-p0"
sys.path.insert(0, str(P0))
from replay import check_kernel, reference, require, sine_samples
from solver import lower, program


def run(args, cwd=None):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=240, cwd=cwd).stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def disassemble(binary, symbol):
    dump = run(["objdump", "--no-show-raw-insn", "--disassemble="+symbol, binary])
    instructions = re.findall(r"^\s*([0-9a-f]+):\s+([^\n]+)", dump, re.M)
    require(instructions, f"missing disassembly for {symbol}")
    return [(int(pc, 16), op.strip()) for pc, op in instructions]


def unique(items, label):
    require(len(items) == 1, f"expected one {label}, found {len(items)}")
    return items[0]


ALLOC = "arena_allocate_vector_with_header"


def alloc_calls(code):
    return [pc for pc, op in code if re.fullmatch(r"call\s+[0-9a-f]+ <"+ALLOC+">", op)]


def observe(binary, log, n):
    """Locate grid storage from allocation events before interpreting any stencil access."""
    main = disassemble(binary, "main")
    kernel = disassemble(binary, "heat-step")
    alloc = disassemble(binary, ALLOC)
    input_alloc = unique(alloc_calls(main), "driver grid allocation site")
    output_alloc = alloc_calls(kernel)
    require(len(output_alloc) <= 1, "unsupported kernel allocation profile")
    # On this ABI the vector header write is followed by an 8-byte length and
    # N+1 tagged slots of 16 bytes. Fail closed on a different allocator.
    header = unique([pc for pc, op in alloc if re.fullmatch(r"movl\s+\$0x2,\(%rax\)", op)],
                    "vector type-header store")
    ret = unique([pc for pc, op in kernel if op == "ret"], "kernel return")
    with log.open("rb") as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as trace:
        section = b"------ name = "+os.fsencode(binary.resolve())+b"\n"
        begin = trace.find(section)
        require(begin >= 0, "missing binary mapping")
        mapping = re.search(rb"PT_LOAD\[\d+\]: +acquired as rx, bias (0x[0-9a-f]+)",
                            trace[begin:begin+8192].split(b"------ name = ", 2)[1])
        require(mapping, "missing executable text mapping")
        bias = int(mapping[1], 16)

        def marker(pc):
            return f"\nI  {bias+pc:08x},".encode()

        def unique_hit(pc):
            pattern = marker(pc)
            hit = trace.find(pattern)
            require(hit >= 0 and trace.find(pattern, hit+1) < 0, f"ambiguous PC {pc:x}")
            return hit

        start, entry, end = map(unique_hit, [input_alloc, kernel[0][0], ret])
        require(start < entry < end, "unexpected native call order")
        def slots(region):
            hits = re.findall(marker(header) + rb"\d+\n S ([0-9a-f]+),4\n", region)
            return int(unique(hits, "vector header event"), 16)+16

        banks = {"old": slots(trace[start:entry])}
        region = trace[entry:end]
        if output_alloc:
            loc = region.find(marker(output_alloc[0]))
            require(loc >= 0, "missing destination allocation")
            banks["fresh"] = slots(region[loc:])
            require(abs(banks["fresh"]-banks["old"]) >= 16*(n+1), "overlapping allocated vectors")
        else:
            require(marker(header) not in region, "unexpected vector allocation")
        events = []
        pc = None
        for line in region.splitlines():
            if line.startswith(b"I "):
                pc = int(line[1:].split(b",")[0], 16)-bias
            elif line[:2] in (b" L", b" S", b" M"):
                addr, size = line[3:].split(b",")
                a, b = int(addr, 16), int(addr, 16)+int(size)
                for bank, base in banks.items():
                    lo, hi = max(a, base), min(b, base+16*(n+1))
                    if lo < hi:
                        require(pc is not None, "orphan memory access")
                        events.append((chr(line[1]), bank, lo-base, hi-base, pc))
    return events


def audit(events, n):
    """Project byte accesses to complete tagged-cell accesses, retaining their order."""
    groups = []
    initial_bytes = set()
    active = False
    for op, bank, lo, hi, pc in events:
        if not active and bank == "fresh" and op == "S":
            initial_bytes.update(range(lo, hi))
            continue
        active = True
        require(op in ("L", "S") and lo//16 == (hi-1)//16, "unsupported grid access")
        key = (op, bank, lo//16)
        if not groups or groups[-1][0] != key:
            groups.append((key, set(), pc))
        touched = set(range(lo%16, (hi-1)%16+1))
        require(groups[-1][1].isdisjoint(touched), "repeated bytes within one cell access")
        groups[-1][1].update(touched)
    require(all(len(group[1]) == 16 for group in groups), "incomplete tagged-cell access")
    changed = set()
    first_store = None
    observations = []
    for row in range(1, n):
        require(len(groups) >= 4*row, f"missing native accesses at cell {row}")
        for offset, expected in enumerate((row-1, row, row+1)):
            (op, bank, i), _, pc = groups[4*(row-1)+offset]
            observations.append([op, bank, i, hex(pc)])
            require(op == "L" and i == expected, f"wrong native operand at cell {row}")
            if bank != "old" or (bank, i) in changed:
                return dict(verdict="REJECT", cell=row, expected=f"old[{expected}]",
                            observed=f"new[{i}]" if (bank,i) in changed else f"fresh[{i}]",
                            load_pc=hex(pc), first_wrong_store=first_store, accesses=observations)
        (op, dest, i), _, pc = groups[4*row-1]
        require(op == "S" and i == row, f"wrong native store at cell {row}")
        observations.append([op, dest, i, hex(pc)])
        changed.add((dest, i))
        if dest == "old" and first_store is None:
            first_store = dict(cell=i, bank=dest, pc=hex(pc))
    require(len(groups) == 4*(n-1), "extra native grid operations")
    require(first_store is None and len(initial_bytes) == 16*(n+1),
            "incorrect destination or incomplete initialization")
    return dict(verdict="ACCEPT", cells=n+1, grid_loads=3*(n-1), grid_stores=n-1,
                accesses=observations)


def one_case(label, n, samples, work, lean, compiler):
    admitted = lower(n)
    require(check_kernel(admitted, n, work, lean) == "ACCEPT", "P0 admission failed")
    candidate = admitted
    if label == "aliased-zero":
        allocation = f"(make-vector {n+1} 0)"
        require(candidate.count(allocation) == 1, "mutation site changed")
        candidate = candidate.replace(allocation, "old")
    source = program(candidate, samples, 1)
    (work / "heat.esk").write_text(source)
    binary = work / "heat"
    run([compiler, "heat.esk", "-o", binary], work)
    raw = run([binary], work)
    expected = reference(samples, 1)
    require([Fraction(s) for s in raw.splitlines()] == expected, "ordinary native output differs")
    log = work / "trace"
    instrumented = run(["valgrind", "--tool=lackey", "--basic-counts=no", "--trace-symtab=yes",
                        "--trace-mem=yes", "--log-file="+str(log), binary], work)
    require(instrumented == raw, "instrumented output differs")
    result = audit(observe(binary, log, n), n)
    if label == "aliased-zero":
        require((result["verdict"], result["cell"], result["expected"], result["observed"]) ==
                ("REJECT", 2, "old[1]", "new[1]"), "native alias control failed")
        require(result["first_wrong_store"]["cell"] == 1, "alias store origin")
    else:
        require(result["verdict"] == "ACCEPT", "native access mismatch")
    return dict(case=label, intervals=n, admitted_kernel_sha256=digest(admitted.encode()),
                compiled_kernel_sha256=digest(candidate.encode()), source_sha256=digest(source.encode()),
                elf_sha256=digest(binary.read_bytes()), field=raw.splitlines(), **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lean = os.environ.get("LEAN", "lean")
    compiler = os.environ.get("ESHKOL_RUN", "eshkol-run")
    cases = [("sine", 20, sine_samples(20)[0]), ("zero", 4, [Fraction(0)]*5),
             ("aliased-zero", 4, [Fraction(0)]*5)]
    results = []
    for label, n, samples in cases:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shutil.copyfile(P0 / "Heat.lean", work / "Heat.lean")
            run([lean, "-DwarningAsError=true", "-o", "Heat.olean", "Heat.lean"], work)
            result = one_case(label, n, samples, work, lean, compiler)
            results.append(result)
            print(label, result["verdict"], flush=True)
    if args.output:
        files = [Path(__file__), P0/"solver.py", P0/"replay.py", P0/"Heat.lean"]
        receipt = dict(fragment="P1/x86-64/tagged-vector/v1", steps=1,
                       lean=run([lean, "--version"]).strip(), eshkol=run([compiler, "--version"]).strip(),
                       observer=run(["valgrind", "--version"]).strip(),
                       verifier_sha256={f.name:digest(f.read_bytes()) for f in files}, cases=results)
        args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
