"""Measure a gate-ID-bound fault in both Steane preparations on the native simulator."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from replay import HERE, ROOT, eshkol_proposal
from schedule import exact_h_cnot_column, schedule_cost

WIDTH = 7
SOURCE_GATE = 12
TAPE_SHA = "dc9975687f9f42099c37d3f7bdf0c12b7f4fb6e8d512c857912ad614b2a35be3"
DECODER_SHA = "2cef52cab0bf45c5440a79700eaf8415039b1d9e491122a81539a196a4891eed"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(binary, tape, seed):
    gates = " ".join("(" + " ".join(map(str, g)) + ")" for g in tape)
    result = subprocess.run(
        [binary],
        input=f"{seed}\n({gates})\n",
        text=True,
        capture_output=True,
        timeout=30,
        cwd=binary.parent,
    )
    if result.returncode:
        raise RuntimeError(f"native measurement failed: {result.stderr[-1500:]}")
    bits = re.fullmatch(r"\(([01](?: [01]){6})\)", result.stdout.strip())
    require(bits is not None, f"unexpected readout: {result.stdout!r}")
    return list(map(int, bits[1].split()))


def parity(bits, mask):
    return sum(b for b, axis in zip(bits, mask, strict=True) if axis != "I") % 2


def observation(binary, tape, checks, table, seed):
    # One X-basis and one Z-basis preparation determine the commuting CSS signs.
    rotate = [["H", i] for i in range(WIDTH)]
    x = measure(binary, tape + rotate, seed)
    z = measure(binary, tape, seed)
    syndrome = "".join(str(parity(x if "X" in check else z, check)) for check in checks)
    require(syndrome in table, f"syndrome {syndrome} outside published recovery table")
    correction = table[syndrome]
    recovery = [[op, i] for i, op in enumerate(correction) if op != "I"]
    corrected = measure(binary, tape + recovery + rotate, seed)
    return dict(
        syndrome=syndrome,
        recovery=correction,
        before_X=(-1) ** (sum(x) % 2),
        recovered_X=(-1) ** (sum(corrected) % 2),
    )


def replay(compiler, work):
    tape_path = HERE / "steane-tape.json"
    table_path = HERE / "steane-recovery.json"
    require(
        sha(tape_path) == TAPE_SHA and sha(table_path) == DECODER_SHA,
        "unexpected Steane inputs",
    )
    gates = json.loads(tape_path.read_bytes())
    decoder = json.loads(table_path.read_bytes())
    env = dict(os.environ, ESHKOL_PATH=str(ROOT / "lib"))
    _, moved = eshkol_proposal(compiler, work, [], WIDTH, gates, env)
    original = list(range(len(gates)))
    expected = original.copy()
    expected.insert(20, expected.pop(SOURCE_GATE))
    require(
        moved == expected and gates[SOURCE_GATE] == ["CNOT", 0, 6],
        "unexpected schedule or gate identity",
    )
    moved_gates = [gates[i] for i in moved]
    for i in range(1 << WIDTH):
        require(
            exact_h_cnot_column(WIDTH, gates, i)
            == exact_h_cnot_column(WIDTH, moved_gates, i),
            f"ideal columns differ at {i}",
        )

    probe = HERE / "fault_readout.esk"
    shutil.copyfile(probe, work / probe.name)
    binary = work / "readout"
    command = [compiler, "-I", ROOT / "lib", "-O", "1", "-o", binary, probe.name]
    build = subprocess.run(
        command, cwd=work, env=env, capture_output=True, text=True, timeout=360
    )
    if build.returncode:
        raise RuntimeError(
            f"native build failed: {build.stderr[-3000:]} {build.stdout[-1000:]}"
        )
    seeds = (1, 73)
    cases = []
    for name, order, depth in (("original", original, 9), ("reordered", moved, 8)):
        require(
            schedule_cost(WIDTH, gates, order)["depth"] == depth,
            "unexpected schedule depth",
        )
        for fault in (False, True):
            tape = []
            for i in order:
                tape.append(gates[i])
                if fault and i == SOURCE_GATE:
                    tape.extend(["Y", wire] for wire in gates[i][1:])
            readings = [
                observation(binary, tape, decoder["checks"], decoder["recovery"], s)
                for s in seeds
            ]
            changed = fault and name == "reordered"
            target = dict(
                syndrome="110110" if changed else "000000",
                recovery="IIIIIYI" if changed else "IIIIIII",
                before_X=1,
                recovered_X=-1 if changed else 1,
            )
            require(
                all(r == target for r in readings), f"{name}, fault={fault}: {readings}"
            )
            cases.append(
                dict(
                    schedule=name,
                    depth=depth,
                    source_gate_position=order.index(SOURCE_GATE),
                    fault="YY" if fault else None,
                    **readings[0],
                )
            )
    return dict(
        compiler_version=subprocess.check_output(
            [compiler, "--version"], text=True
        ).strip(),
        compiler_sha256=sha(compiler),
        probe_sha256=sha(probe),
        native_elf_sha256=sha(binary),
        tape_sha256=TAPE_SHA,
        recovery_sha256=DECODER_SHA,
        seeds=seeds,
        ideal_columns=1 << WIDTH,
        cases=cases,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eshkol", default=os.environ.get("ESHKOL_RUN", "eshkol-run"))
    args = parser.parse_args()
    binary = shutil.which(args.eshkol)
    if not binary:
        parser.error("set --eshkol to the quantum feature compiler")
    with tempfile.TemporaryDirectory() as tmp:
        print(json.dumps(replay(Path(binary).resolve(), Path(tmp)), indent=2))
