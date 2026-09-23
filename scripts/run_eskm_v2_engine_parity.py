#!/usr/bin/env python3
"""Experimental ESKM v2 four-engine interoperability and fail-closed controls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zlib

from run_eskm_tensor_engine_parity import ENGINES, Failure, Infrastructure, ROOT, run, verify_bytes

SOURCE = ROOT / "tests/core/eskm_v2_engine_parity.esk"
FIXTURES = ROOT / "tests/core/fixtures"
TENSORS = ("scalar", "empty", "ordinary", "rank8")


def wire(records, version=2, extensions=b"", flags=0):
    body = struct.pack("<4sIII", b"ESKM", version, len(records), flags)
    if version == 2:
        body += struct.pack("<Q", len(extensions)) + extensions
    for record in records:
        name = bytes.fromhex(record["name_hex"])
        dims = record["dimensions"]
        body += struct.pack("<I", len(name)) + name + struct.pack("<I", len(dims))
        body += b"".join(struct.pack("<Q", d) for d in dims) + b"\0"
        body += b"".join(struct.pack("<Q", int(v, 16)) for v in record["elements_hex"])
    return body + struct.pack("<I", zlib.crc32(body))


def cases():
    manifest = json.loads((FIXTURES / "eskm-v1/manifest.json").read_text())
    historical = {f["file"]: f["expected"]["records"] for f in manifest["fixtures"]
                  if f["expected"].get("accept")}
    names = dict(scalar="scalar", empty="empty-0x3", ordinary="ordinary-2x3", rank8="rank8")
    records = {name: historical[stem + ".eskm"] for name, stem in names.items()}
    # Verify the independent record encoder against immutable historical bytes.
    for name, stem in names.items():
        verify_bytes(FIXTURES / f"eskm-v1/{stem}.eskm", wire(records[name], 1))
    records["multi"] = [dict(records[name][0], name_hex=b"duplicate".hex())
                        for name in ("scalar", "empty")]
    records["empty-model"] = []
    return records


def matrix(args, work):
    compiler, vm = args.compiler.resolve(), args.vm.resolve()
    for binary in (compiler, vm):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise Infrastructure(f"missing executable: {binary}")
    env = dict(os.environ, ESHKOL_VM_NO_DISASM="1", ESHKOL_JIT_CACHE="0")
    env.pop("ESHKOL_EXPERIMENTAL_ESKM_V2", None)
    env["ESHKOL_PATH"] = str(ROOT / "lib") + os.pathsep + env.get("ESHKOL_PATH", "")
    aot, bytecode = work / "fixture-aot", work / "fixture.eskb"
    run([str(compiler), "--no-stdlib", str(SOURCE), "-o", str(aot), f"-L{compiler.parent}"],
        work, "compile-aot", env, args.timeout)
    run([str(compiler), "--profile", "hosted-vm", "--emit-eskb", str(bytecode), str(SOURCE)],
        work, "compile-bytecode", env, args.timeout)
    commands = {"jit": [str(compiler), "--no-stdlib", "-r", str(SOURCE), f"-L{compiler.parent}"],
                "aot": [str(aot)], "vm-source": [str(vm), str(SOURCE)],
                "vm-bytecode": [str(vm), str(bytecode)]}
    records = cases()
    v1 = {name: wire(value, 1) for name, value in records.items()}
    v2 = {name: wire(value) for name, value in records.items()}
    wrong_oracles = 0

    def execute(engine, label, inputs, mode="roundtrip", opt="write", poison=False):
        directory = work / label
        directory.mkdir()
        for name, data in inputs.items():
            (directory / f"{name}.eskm").write_bytes(data)
        runtime_env = dict(env, ESKM_V2_PARITY_MODE=mode)
        if opt is not None:
            runtime_env["ESHKOL_EXPERIMENTAL_ESKM_V2"] = opt
        if poison:
            runtime_env.update(ESHKOL_ARENA_POISON="1", ESHKOL_VM_ARENA_POISON="1")
        run(commands[engine], directory, "run", runtime_env, args.timeout)
        lines = (directory / "run.stdout").read_text().splitlines()
        if lines.count("ESKM-V2-PARITY:PASS") != 1 or any("FAIL" in line for line in lines):
            raise Failure(f"{label}: semantic marker missing or failed")
        for name, data in inputs.items():
            verify_bytes(directory / f"{name}.eskm", data)
        diagnostics = (directory / "run.stderr").read_text()
        load_failures = 2 * len(inputs) if mode == "reject" else 2 if mode in ("off", "unknown") else 0
        if diagnostics.count("invalid or unreadable ESKM checkpoint") != load_failures:
            raise Failure(f"{label}: expected {load_failures} checkpoint failure diagnostics")
        if mode == "unknown" and diagnostics.count("experimental ESKM v2 requires") != 3:
            raise Failure(f"{label}: expected 3 invalid opt-in diagnostics")
        return directory

    def outputs(directory, expected, tensors=TENSORS):
        nonlocal wrong_oracles
        for name, data in expected.items():
            for prefix in (("model-", "tensor-") if name in tensors else ("model-",)):
                path = directory / f"{prefix}{name}.eskm"
                verify_bytes(path, data)
                if args.self_test:
                    try:
                        verify_bytes(path, data[:-1] + bytes([data[-1] ^ 1]))
                    except Failure:
                        wrong_oracles += 1
                    else:
                        raise Failure("wrong byte oracle accepted")

    producers = {}
    for engine in ENGINES:
        directory = execute(engine, f"produce-{engine}", v1)
        outputs(directory, v2)
        producers[engine] = {name: (directory / f"model-{name}.eskm").read_bytes() for name in records}
        # Both public writers are compared to the same independent oracle.
        print(f"PASS {engine}: 10 exact v2 producer files", flush=True)
    for consumer in ENGINES:
        for producer in ENGINES:
            directory = execute(consumer, f"{producer}-to-{consumer}", producers[producer])
            outputs(directory, v2)
            print(f"PASS {producer} -> {consumer}: 10 exact rewrites", flush=True)

    annotated = (FIXTURES / "eskm-v2/annotations-scalar.eskm").read_bytes()
    annotation_manifest = json.loads((FIXTURES / "eskm-v2/manifest.json").read_text())
    annotation_records = next(f["expected"]["records"] for f in annotation_manifest["fixtures"]
                              if f["file"] == "annotations-scalar.eskm")
    invalid = {"mandatory": wire(records["scalar"], flags=1),
               "rank9": wire([dict(records["scalar"][0], dimensions=[1] * 9)]),
               "signed-empty": wire([dict(records["empty"][0], dimensions=[0, 1 << 63])]),
               "nul-name": (FIXTURES / "eskm-v2/raw-bits.eskm").read_bytes(),
               "version": wire(records["scalar"], version=99),
               "tlv": wire(records["scalar"], extensions=struct.pack("<II", 0, 0)),
               "crc": v2["scalar"][:-1] + bytes([v2["scalar"][-1] ^ 1]),
               "trailing": v2["scalar"] + b"\0"}
    refusal_count = 2 * len(invalid) + 5
    for engine in ENGINES:
        outputs(execute(engine, f"default-{engine}", v1, opt=None), v1)
        outputs(execute(engine, f"read-{engine}", v2, opt="read"), v1)
        outputs(execute(engine, f"annotations-{engine}", {"annotated": annotated}, mode="annotated"),
                {"annotated": wire(annotation_records)}, tensors=("annotated",))
        small = {name: v2[name] for name in ("scalar", "empty")}
        outputs(execute(engine, f"region-{engine}", small, mode="region", poison=True), small)
        execute(engine, f"reject-{engine}", invalid, mode="reject")
        execute(engine, f"off-{engine}", {"scalar": v2["scalar"]}, mode="off", opt=None)
        denied = execute(engine, f"unknown-{engine}", {"scalar": v1["scalar"]},
                         mode="unknown", opt="invalid")
        if (denied / "refused.eskm").exists():
            raise Failure("unknown opt-in mode created output")
        print(f"PASS {engine}: v1 defaults/read mode, annotations, poisoned region escape, {refusal_count} refusals", flush=True)
    if args.self_test:
        if wrong_oracles != 304:
            raise Failure(f"wrong-oracle count {wrong_oracles}, expected 304")
        print(f"PASS {wrong_oracles} wrong-byte oracles refused", flush=True)
    print("PASS ESKM v2: 16/16 pairs, 40 producer files, 160 cross-engine rewrites; "
          f"80 v1 compatibility outputs, 8 metadata-discard rewrites, 16 region-escape rewrites, "
          f"{refusal_count * len(ENGINES)} refusals with diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path, nargs="?", default=ROOT / "build/eshkol-run")
    parser.add_argument("vm", type=Path, nargs="?", default=ROOT / "build/eshkol-vm-standalone-test")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    work = Path(tempfile.mkdtemp(prefix="eshkol-eskm-v2-parity-",
                                dir=os.environ.get("ESHKOL_TEST_TMPDIR"))).resolve()
    keep = args.keep or bool(os.environ.get("ESHKOL_TEST_KEEP_TMPDIR"))
    try:
        matrix(args, work)
        return 0
    except Failure as exc:
        keep = True
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except (Infrastructure, OSError) as exc:
        keep = True
        print(f"INFRA: {exc}", file=sys.stderr)
        return 125
    finally:
        if keep:
            print(f"ESKM v2 artifacts: {work}", file=sys.stderr)
        else:
            shutil.rmtree(work)


if __name__ == "__main__":
    sys.exit(main())
