"""Translate valid-pointer paths in the linked SysV x86-64 pair primitives."""
import re
import subprocess

# Only decode instructions used by the successful tagged getter and setter.
# The valid-path walker checks branches using their signed byte displacement.
DATA = {
    "f30f6f02": ".readInput 0", "f30f6f0a": ".readInput 1",
    "0f114710": ".storeXmm 0 2", "0f110f": ".storeXmm 1 0",
    "488b5718": ".readRdx 3", "488b5708": ".readRdx 1",
    "488b4710": ".readRax 2", "488b07": ".readRax 0",
}


def instructions(exe, symbol):
    listing = subprocess.check_output(
        ["objdump", "-d", "--insn-width=16", "--disassemble=" + symbol, exe], text=True)
    if "file format elf64-x86-64" not in listing:
        raise ValueError("linked primitive gate requires x86-64 ELF")
    entry = re.search(r"^([0-9a-f]+) <" + re.escape(symbol) + r">:$", listing, re.M)
    if not entry:
        raise ValueError("missing linked symbol " + symbol)
    rows = re.findall(r"^\s*([0-9a-f]+):\s*\t((?:[0-9a-f]{2} )+)", listing, re.M)
    return int(entry[1], 16), {int(a, 16): bytes.fromhex(code) for a, code in rows}


def path(entry, code, cdr, setter):
    pc = entry
    zero = None
    seen = set()
    data = []
    while pc in code and pc not in seen:
        seen.add(pc)
        insn = code[pc]
        pc += len(insn)
        word = insn.hex()
        if word == "f30f1efa":  # endbr64
            continue
        if word == "4885ff" or (setter and word == "4885d2"):  # test rdi/rdx
            zero = False
        elif word == "4084f6":  # test sil, sil
            zero = not cdr
        elif len(insn) == 2 and insn[0] == 0x74 and zero is not None:
            if zero:
                pc += int.from_bytes(insn[1:], "little", signed=True)
        elif word in DATA:
            data.append(DATA[word])
        elif word == "c3":
            return data
        else:
            raise ValueError(f"unmodeled linked primitive instruction {word}")
    raise ValueError("linked primitive path did not return")


def check_linked(work, exe, lean, env, execute):
    proof = ["import LinkedPair", "open Verificatum.LinkedPair"]
    for setter, symbol in [(True, "set"), (False, "get")]:
        entry, code = instructions(exe, f"arena_tagged_cons_{symbol}_tagged_value")
        for cdr in (False, True):
            ops = ", ".join(path(entry, code, cdr, setter))
            flag = str(cdr).lower()
            ty = "Setter" if setter else "Getter"
            lemma = f"setter_ok {flag} {0 if cdr else 1}" if setter else f"getter_ok {flag}"
            proof.append(f"example : {ty} [{ops}] {flag} := {lemma}")
    file = work / "LinkedArtifact.lean"
    file.write_text("\n".join(proof) + "\n")
    execute([lean, file.name], work, env)
    # A supported vector store to the peer is still a well-formed operation.
    # The generated getter/setter contract must fail on this wrong destination.
    altered = file.read_text().replace(".storeXmm 0 2", ".storeXmm 0 0")
    if altered == file.read_text():
        raise ValueError("missing cdr store in linked contract")
    bad = work / "WrongField.lean"
    bad.write_text(altered)
    mutation = subprocess.run([lean, bad.name], cwd=work, env=env, capture_output=True, text=True)
    if mutation.returncode == 0 or "Type mismatch" not in mutation.stdout + mutation.stderr:
        raise ValueError("wrong linked field escaped the contract")
    print("Linked pair getter and setter satisfy all four valid-pointer path contracts.")

def check_direct_store(work, exe, lean, env, execute, control, heap_rows):
    # The helper contains exactly byte store [RDI] <- SIL and RET on this ABI.
    # Do not accept a profile for a different function body or a stack spill.
    pc, code = instructions(exe, "blc_probe_direct_tag")
    if code[pc].hex() == "f30f1efa":
        pc += 4
    if code[pc].hex() != "408837" or code[pc+3].hex() != "c3":
        raise ValueError("unexpected direct-tag data-memory path")
    healthy = execute([exe], work, env)
    n = sum(line.startswith("RAW-WRITE ") for line in control.splitlines())
    if n == 0:
        raise ValueError("direct-store experiment needs a payload store")
    for direct in (False, True):
        profile = work / f"tag-{int(direct)}.cg"
        cmd = ["valgrind", "--tool=callgrind", "--cache-sim=yes", "--collect-atstart=no",
               "--toggle-collect=blc_probe_direct_tag", "--compress-strings=no",
               "--callgrind-out-file=" + str(profile), exe]
        run = execute(cmd, work, {**env, **({"BLC_PROBE_DIRECT": "1"} if direct else {})})
        if run != healthy:
            raise ValueError("direct store changed a public result or any audit channel")
        rows = profile.read_text().splitlines()
        events = next(line.split()[1:] for line in rows if line.startswith("events:"))
        totals = next(line.split()[1:] for line in rows if line.startswith("totals:"))
        # Callgrind elides trailing zero columns.
        counts = dict(zip(events, map(int, totals)))
        writes = counts.get("Dw", 0)
        if "Dw" not in events or writes != (n if direct else 0):
            raise ValueError(f"unexpected direct store count {writes}")
        (work / "observed.log").write_text(run)
        (work / "heap.events").write_text("\n".join(heap_rows(run)) + "\n")
        for gate in ("Audit.lean", "stores.lean", "runtime.lean"):
            execute([lean, gate], work, env)
    print(f"Direct tag writes 0 -> {n}; identical heap, Scheme and linked setter records.")
