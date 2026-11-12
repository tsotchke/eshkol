#!/usr/bin/env python3
"""Retained, build-free refusal controls for consumer cohort admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts/check_consumer_toolchain.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(work, name, no_git=False):
    case = work / name
    source, artifacts = case / "source", case / "artifacts"
    source.mkdir(parents=True)
    artifacts.mkdir()
    files = {"CMakeLists.txt": "# fixture source\n", "exe/eshkol-run.cpp": "// fixture\n",
             "inc/eshkol/eshkol.h": '#define ESHKOL_VERSION_STRING "1.4.0-fixture"\n',
             "lib/stdlib.esk": "; fixture\n", "lib/math.esk": "; fixture\n",
             "cmake/StdlibIdentityInputs.cmake": (ROOT / "cmake/StdlibIdentityInputs.cmake").read_text()}
    for relative, value in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (source / ".gitignore").write_text("/inc/eshkol/.ignored\n")
    head = "1" * 40
    if not no_git:
        commands = [["git", "init", "-q", str(source)], ["git", "-C", str(source), "add", "."],
                    ["git", "-C", str(source), "-c", "user.name=Admission Fixture",
                     "-c", "user.email=fixture@invalid", "-c", "commit.gpgsign=false",
                     "commit", "-qm", "retained synthetic cohort"]]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, timeout=20)
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    material = "".join(p + ":" + sha(source / p) + "\n" for p in sorted(files))
    material += "toolchain:c=fixture;cxx=fixture;llvm=fixture\n"
    (artifacts / "source-material.txt").write_text(material)
    fingerprint = sha(artifacts / "source-material.txt")
    identity = dict(release="1.4.0-fixture", abi="object-abi-v2", source=fingerprint)
    identity["symbol"] = "eshkol_stdlib_identity_release_1_4_0_fixture_abi_object_abi_v2_source_" + fingerprint
    compiler = artifacts / "eshkol-run"
    compiler.write_text("#!" + sys.executable + "\nimport sys\nfrom pathlib import Path\n" +
        "identity = " + repr(dict(schema="eshkol.stdlib-identity/v1", **identity)) + "\n" +
        "if sys.argv[1] == '--check-stdlib-identity':\n" +
        "    if Path(sys.argv[2]).read_bytes() != b'matching fixture stdlib':\n" +
        "        print('stdlib identity check failed: fixture', file=sys.stderr); sys.exit(1)\n" +
        "    identity['status'] = 'PASS'\n" +
        "print('\\n'.join(k+'='+v for k,v in identity.items()))\n")
    compiler.chmod(0o755)
    (artifacts / "stdlib.o").write_bytes(b"matching fixture stdlib")
    (artifacts / "runtime.a").write_bytes(b"fixture runtime bytes")
    names = dict(compiler="eshkol-run", stdlib="stdlib.o", runtime="runtime.a", source_material="source-material.txt")
    manifest = dict(schema="eshkol.consumer-toolchain/v1", source_revision=head, identity=identity,
                    artifacts={kind: dict(path=name, sha256=sha(artifacts / name)) for kind, name in names.items()})
    return case, source, artifacts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="consumer-admission-", dir=args.work_root)).resolve()
    started, rows = time.monotonic(), []
    cases = [
        ("matching", None), ("wrong-revision", "source revision mismatch"),
        ("source-archive-no-git", "not a git repository"),
        ("missing-version-header", "configured source namespace or bytes differ"),
        ("missing-stdlib", "No such file"), ("missing-runtime", "No such file"),
        ("foreign-compiler", "artifact digest mismatch: compiler"),
        ("foreign-stdlib", "artifact digest mismatch: stdlib"),
        ("foreign-runtime", "artifact digest mismatch: runtime"),
        ("edited-source", "configured source namespace or bytes differ"),
        ("ignored-source", "configured source namespace or bytes differ"),
        ("git-unavailable", "No such file"),
        ("wrong-compiled-release", "compiled identity mismatch"),
        ("wrong-compiled-source", "compiled identity mismatch"),
        ("wrong-compiled-abi", "compiled identity mismatch"),
        ("unstamped-stdlib", "stdlib identity check failed"),
        ("wrong-manifest-pin", "manifest differs from independently supplied"),
        ("unknown-field", "invalid fields in manifest"),
        ("duplicate-key", "duplicate manifest key"),
        ("unsafe-path", "unsafe artifact path"),
        ("artifact-symlink", "symlink in source input"),
        ("artifact-fifo", "not a regular file"),
        ("wrong-material", "material digest differs"),
        ("runtime-drift", "artifact changed during admission: runtime"),
        ("source-drift", "configured source namespace or bytes differ"),
        ("duplicate-identity", "invalid or duplicate compiler identity field"),
        ("missing-pin", "--expected-manifest-sha256"),
    ]
    for name, reason in cases:
        case, source, artifacts, data = fixture(work, name, name == "source-archive-no-git")
        compiler = artifacts / "eshkol-run"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GIT_CEILING_DIRECTORIES=str(case))
        if name == "wrong-revision":
            data["source_revision"] = "0" * 40
        if name == "missing-version-header":
            (source / "inc/eshkol/eshkol.h").rename(case / "retained-version-header.h")
        if name in ("missing-stdlib", "missing-runtime"):
            kind = name.removeprefix("missing-")
            (artifacts / data["artifacts"][kind]["path"]).rename(case / ("retained-" + kind))
        if name.startswith("foreign-"):
            kind = name.removeprefix("foreign-")
            (artifacts / data["artifacts"][kind]["path"]).write_bytes(b"foreign bytes")
        if name == "edited-source":
            (source / "exe/eshkol-run.cpp").write_text("// modified after commit\n")
        if name == "ignored-source":
            (source / "inc/eshkol/.ignored").write_text("new configured source input\n")
            assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"])
        if name == "git-unavailable":
            env["PATH"] = str(case / "absent-bin")
        if name.startswith("wrong-compiled-"):
            field = name.removeprefix("wrong-compiled-")
            old = data["identity"][field]
            compiler.write_text(compiler.read_text().replace(repr(old), repr("foreign"), 1))
            data["artifacts"]["compiler"]["sha256"] = sha(compiler)
        if name == "unstamped-stdlib":
            (artifacts / "stdlib.o").write_bytes(b"unstamped fixture stdlib")
            data["artifacts"]["stdlib"]["sha256"] = sha(artifacts / "stdlib.o")
        if name == "unknown-field":
            data["unrecognized"] = True
        if name == "unsafe-path":
            data["artifacts"]["runtime"]["path"] = "../runtime.a"
        if name in ("artifact-symlink", "artifact-fifo"):
            original = artifacts / "runtime.a"
            original.rename(case / "retained-runtime.a")
            if name == "artifact-symlink":
                original.symlink_to(case / "retained-runtime.a")
            else:
                os.mkfifo(original)
        if name == "wrong-material":
            path = artifacts / "source-material.txt"
            path.write_text(path.read_text() + "foreign material\n")
            data["artifacts"]["source_material"]["sha256"] = sha(path)
        if name in ("runtime-drift", "source-drift", "duplicate-identity"):
            if name == "runtime-drift":
                change = "Path(" + repr(str(artifacts / "runtime.a")) + ").write_bytes(b'changed during query')\n"
            elif name == "source-drift":
                change = "Path(" + repr(str(source / "exe/eshkol-run.cpp")) + ").write_text('// changed during query\\n')\n"
            else:
                change = "print('schema=duplicate')\n"
            compiler.write_text(compiler.read_text() + change)
            data["artifacts"]["compiler"]["sha256"] = sha(compiler)
        manifest = case / "manifest.json"
        blob = json.dumps(data)
        if name == "duplicate-key":
            blob = blob[:-1] + ', "schema": "eshkol.consumer-toolchain/v1"}'
        manifest.write_text(blob)
        pin = "0" * 64 if name == "wrong-manifest-pin" else sha(manifest)
        command = [sys.executable, CHECK, "--manifest", manifest,
                   "--source-root", source, "--artifact-root", artifacts]
        if name != "missing-pin":
            command += ["--expected-manifest-sha256", pin]
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, env=env, timeout=20)
        (case / "stdout.log").write_text(result.stdout)
        (case / "stderr.log").write_text(result.stderr)
        expected = 0 if reason is None else (2 if name == "missing-pin" else 1)
        observed = result.stdout + result.stderr
        passed = result.returncode == expected and (reason is None or reason in observed)
        rows.append(dict(case=name, passed=passed, returncode=result.returncode, expected_exit=expected, expected_reason=reason))
        (work / "CASES.json").write_text(json.dumps(rows, indent=2) + "\n")
        assert passed, (name, result.returncode, observed)

    for name, omit_pin, expected in [("cmake-matching", False, 0), ("cmake-missing-pin", True, 1)]:
        case, source, artifacts, data = fixture(work, name)
        manifest = case / "manifest.json"
        manifest.write_text(json.dumps(data))
        project = case / "project"
        project.mkdir()
        body = ('cmake_minimum_required(VERSION 3.14)\nproject(ConsumerAdmission NONE)\ninclude("' +
                str(ROOT / "cmake/ConsumerToolchainAdmission.cmake") + '")\neshkol_require_consumer_toolchain(\n' +
                'MANIFEST "' + str(manifest) + '"\nSOURCE_ROOT "' + str(source) + '"\nARTIFACT_ROOT "' + str(artifacts) + '"\n')
        if not omit_pin:
            body += 'MANIFEST_SHA256 "' + sha(manifest) + '"\n'
        body += ')\nfile(WRITE "${CMAKE_BINARY_DIR}/ADMISSION.json" "${ESHKOL_CONSUMER_ADMISSION_JSON}")\n'
        (project / "CMakeLists.txt").write_text(body)
        result = subprocess.run([shutil.which("cmake"), "-S", str(project), "-B", str(case / "configure")], capture_output=True, text=True, timeout=30)
        (case / "stdout.log").write_text(result.stdout)
        (case / "stderr.log").write_text(result.stderr)
        assert result.returncode == expected, (name, result.stdout, result.stderr)
        if not omit_pin:
            assert json.loads((case / "configure/ADMISSION.json").read_text())["status"] == "PASS_CONSUMER_TOOLCHAIN_ADMISSION"
        else:
            assert "requires explicit MANIFEST_SHA256" in result.stderr
        rows.append(dict(case=name, passed=True, returncode=result.returncode, expected_exit=expected))
    report = dict(status="PASS_CONSUMER_ADMISSION_FIXTURE_CONTROLS", controls=len(rows),
                  accepted=2, refused=len(rows)-2, seconds=round(time.monotonic()-started, 3),
                  product_executions=0, product_builds=0, release_tuple_selected=False,
                  scope="Python metadata compiler fixtures and real Git/CMake admission; no Eshkol runtime qualification", cases=rows)
    (work / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "cases"}, sort_keys=True))
    print("retained=" + str(work))


if __name__ == "__main__":
    main()
