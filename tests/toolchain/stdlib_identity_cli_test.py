#!/usr/bin/env python3
"""Exercise the production identity helpers with real LLVM object/bitcode files.

Without --compiler, compile a standalone metadata-helper driver with an explicit
fixture identity. Outputs are retained. This does not qualify an Eshkol runtime.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_identity(text):
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        need(separator and key and value and key not in fields,
             "invalid or duplicate identity field: " + line)
        fields[key] = value
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--cxx", type=Path)
    parser.add_argument("--llvm-config", type=Path)
    parser.add_argument("--llvm-as", type=Path, required=True)
    parser.add_argument("--llc", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    for name in ("release", "abi", "source", "symbol"):
        parser.add_argument("--expected-" + name)
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="identity-cli-", dir=args.work_root)).resolve()
    (work / "tmp").mkdir()
    env = dict(os.environ, TMPDIR=str(work / "tmp"), PYTHONDONTWRITEBYTECODE="1",
               ESHKOL_LIB_DIR=str(work / "intentionally-absent-sdk"),
               ESHKOL_PATH=str(work / "intentionally-absent-modules"))
    commands, cases = [], []
    result = {"status": "RUNNING", "production_cli_invoked": bool(args.compiler),
              "product_runtime_qualification": False, "cases": cases}

    def record():
        (work / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")

    def run(label, argv, expected=0):
        start = time.monotonic()
        child = subprocess.run(list(map(str, argv)), capture_output=True, text=True,
                               env=env, timeout=60)
        (work / (label + ".stdout")).write_text(child.stdout)
        (work / (label + ".stderr")).write_text(child.stderr)
        commands.append({"label": label, "argv": list(map(str, argv)),
                         "exit": child.returncode,
                         "seconds": round(time.monotonic() - start, 4)})
        (work / "COMMANDS.json").write_text(json.dumps(commands, indent=2) + "\n")
        need(child.returncode == expected,
             f"{label}: expected exit {expected}, got {child.returncode}: {child.stderr[-1200:]}")
        return child

    record()
    try:
        if args.compiler:
            executable = args.compiler.resolve()
            expected = {name: getattr(args, "expected_" + name)
                        for name in ("release", "abi", "source", "symbol")}
            need(all(expected.values()), "production mode requires all four expected fields")
            result["kind"] = "PRODUCTION_CLI_METADATA_ONLY"
        else:
            need(args.cxx and args.llvm_config, "fixture mode requires --cxx and --llvm-config")
            expected = {"release": "1.4.0-fixture", "abi": "object-abi-v2", "source": "a" * 64}
            expected["symbol"] = ("eshkol_stdlib_identity_release_1_4_0_fixture_abi_object_abi_v2_source_"
                                  + expected["source"])
            include = work / "include/eshkol"
            include.mkdir(parents=True)
            config = "#pragma once\n" + "".join(
                "#define ESHKOL_STDLIB_IDENTITY_" + name.upper() + " " + json.dumps(value) + "\n"
                for name, value in expected.items())
            (include / "stdlib_identity_config.h").write_text(config)
            flags = run("llvm-config", [args.llvm_config, "--cxxflags", "--ldflags",
                        "--libs", "core", "object", "bitreader", "--system-libs"]).stdout
            executable = work / "identity-helper"
            run("compile-helper", [args.cxx, ROOT / "tests/toolchain/stdlib_identity_cli_fixture.cpp",
                "-o", executable, *shlex.split(flags), "-std=c++20",
                "-I" + str(include.parent), "-I" + str(ROOT / "inc")])
            result["kind"] = "METADATA_HELPER_FIXTURE_NOT_PRODUCT"
        query = run("query", [executable, "--stdlib-identity"])
        identity = parse_identity(query.stdout)
        need(identity == dict(schema="eshkol.stdlib-identity/v1", **expected),
             "query does not match the externally supplied compiled identity")
        need(not query.stderr, "query unexpectedly emitted diagnostics")
        need(re.fullmatch(r"[0-9a-f]{64}", expected["source"]), "invalid source fingerprint")
        expected_name = ("eshkol_stdlib_identity_release_" +
                         re.sub(r"[^A-Za-z0-9_]", "_", expected["release"]) + "_abi_" +
                         re.sub(r"[^A-Za-z0-9_]", "_", expected["abi"]) + "_source_" +
                         expected["source"])
        need(expected_name == expected["symbol"], "identity fields and symbol disagree")
        symbol = expected["symbol"]
        variants = {
            "matching": f"@{symbol} = global i8 0\n",
            "wrong-release": f"@{symbol.replace('release_', 'release_foreign_', 1)} = global i8 0\n",
            "wrong-source": f"@{symbol.rsplit('_source_', 1)[0]}_source_{('0' if expected['source'] != '0' * 64 else '1') * 64} = global i8 0\n",
            "wrong-abi": f"@{symbol.replace('_abi_', '_abi_foreign_', 1)} = global i8 0\n",
            "unstamped": "@ordinary = global i8 0\n",
            "declaration": f"@{symbol} = external global i8\n@keep = global ptr @{symbol}\n",
            "mixed": f"@{symbol} = global i8 0\n@eshkol_stdlib_identity_foreign = global i8 0\n",
            # Outside Mach-O a leading underscore is part of the symbol name;
            # it must not be normalized into the expected marker.
            "forged-leading-underscore": f"@_{symbol} = global i8 0\n",
        }

        def check(label, artifact, allowed):
            child = run(label, [executable, "--check-stdlib-identity", artifact], 0 if allowed else 1)
            if allowed:
                need(parse_identity(child.stdout) == dict(identity, status="PASS"),
                     label + ": wrong successful identity report")
                need(not child.stderr, label + ": successful check emitted diagnostics")
            else:
                need(child.stdout == "", label + ": rejection emitted successful metadata")
                need("stdlib identity check failed" in child.stderr, label + ": missing rejection reason")
            cases.append({"name": label, "expected_admission": allowed, "pass": True})
            record()

        for name, ir in variants.items():
            ll = work / (name + ".ll")
            bc = work / (name + ".bc")
            obj = work / (name + ".o")
            ll.write_text(ir)
            run(name + "-assemble", [args.llvm_as, ll, "-o", bc])
            run(name + "-object", [args.llc, "-filetype=obj", bc, "-o", obj])
            for kind, path in (("bitcode", bc), ("object", obj)):
                check(name + "-" + kind, path, name == "matching")
        # Detection follows bytes rather than filenames.
        for kind in ("bc", "o"):
            data = (work / ("matching." + kind)).read_bytes()
            renamed = work / ("matching-" + kind + ".data")
            renamed.write_bytes(data)
            check("renamed-" + kind, renamed, True)
        for name, data in (("garbage", b"not an object or bitcode"),
                           ("truncated-bitcode", b"BC\xc0\xde\x00")):
            path = work / name
            path.write_bytes(data)
            check(name, path, False)
        check("missing-file", work / "absent", False)
        check("directory", work / "tmp", False)
        if args.compiler:
            # Metadata actions finish option parsing before they run, so unknown
            # options, operands, empty paths and repeated or combined actions are
            # refused without printing identity metadata.
            matching = work / "matching.o"
            refusals = {
                "leading-unknown": (["--unknown-identity-option", "--stdlib-identity"], None),
                "trailing-unknown": (["--stdlib-identity", "--unknown-identity-option"], None),
                "trailing-invalid-types": (["--stdlib-identity", "--types", "bogus"],
                                           "--types takes gradual or strict"),
                "metadata-positional": (["--stdlib-identity", work / "main.esk"],
                                        "do not accept positional inputs"),
                "empty-check-path": (["--check-stdlib-identity="], "requires a non-empty path"),
                "repeated-print": (["--stdlib-identity", "--stdlib-identity"], "may be specified only once"),
                "repeated-check": (["--check-stdlib-identity", matching, "--check-stdlib-identity", matching],
                                   "may be specified only once"),
                "combined-actions": (["--stdlib-identity", "--check-stdlib-identity", matching],
                                     "may be specified only once"),
                "combined-actions-reverse": (["--check-stdlib-identity", matching, "--stdlib-identity"],
                                             "may be specified only once"),
                "check-trailing-unknown": (["--check-stdlib-identity", matching, "--unknown-identity-option"], None),
                "check-positional": (["--check-stdlib-identity", matching, "extra-input"],
                                     "do not accept positional inputs"),
            }
            metadata = re.compile(r"^(schema|release|abi|source|symbol|status)=", re.M)
            for label, (argv, reason) in refusals.items():
                child = subprocess.run([str(executable), *map(str, argv)], capture_output=True,
                                       text=True, env=env, timeout=60)
                (work / ("cli-" + label + ".stdout")).write_text(child.stdout)
                (work / ("cli-" + label + ".stderr")).write_text(child.stderr)
                need(child.returncode != 0, label + ": metadata command unexpectedly succeeded")
                need(not metadata.search(child.stdout) and not metadata.search(child.stderr),
                     label + ": refusal emitted identity metadata")
                if reason:
                    need(reason in child.stderr, label + ": missing diagnostic: " + reason)
                cases.append({"name": "cli-" + label, "expected_admission": False, "pass": True})
                record()
        need(len(cases) == 22 + (11 if args.compiler else 0), "case matrix incomplete")
        result.update(status="PASS_STDLIB_IDENTITY_METADATA_CONTROLS",
                      cases_passed=len(cases), expected_identity=identity,
                      executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
        record()
        print(json.dumps({"status": result["status"], "kind": result["kind"],
                          "cases_passed": len(cases), "output": str(work),
                          "product_runtime_qualification": False}))
        return 0
    except Exception as exc:
        result.update(status="FAIL", error=type(exc).__name__ + ": " + str(exc))
        record()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
