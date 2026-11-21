"""Exercise R1 on the released CLI with a kernel-observed, sealed execution."""

import argparse
import ast
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from unittest.mock import patch

import sink
from authority import encode, fingerprint
import extract_kernel
import heat
from replay import audit_kernel  # authorization/replay.py on the path installed by sink


def require(ok, message):
    if not ok:
        raise ValueError(message)


def translator_controls():
    source = (sink.HERE.parent / "authorization/kernel.py").read_text()
    probes = {
        "module directive": source + "\n# type: ignore\n",
        "extra statement": source + "\nx = 1\n",
        "decorator": source.replace("def decide", "@staticmethod\ndef decide"),
        "parameter annotation": source.replace("needs, check", "needs: tuple, check"),
        "default": source.replace("needs, check", "needs, check=None"),
        "variadic": source.replace("needs, check", "needs, *check"),
        "keyword argument": source.replace("check(need)", "check(arg=need)"),
        "loop else": source.replace(
            '    return "authorized"', '    else:\n        pass\n    return "authorized"'
        ),
        "side effect before release": source.replace(
            "    return payload if", "    print(payload)\n    return payload if"
        ),
        "identity comparison": source.replace("payload == expected", "payload is expected"),
        "unchecked operand": source.replace("payload == expected", "expected == expected"),
        "additional conjunct": source.replace("else None", "and expected else None"),
        "different return value": source.replace("return payload if", "return current if"),
        "type comment": source.replace(
            "verdict = check(need)", "verdict = check(need)  # type: str"
        ),
    }
    for label, text in probes.items():
        ast.parse(text, type_comments=True)  # Every control is valid Python syntax.
        try:
            extract_kernel.translate(text)
        except ValueError:
            continue
        raise ValueError("translator accepted " + label)
    # Trivia does not change the accepted AST or its translation.
    require(
        extract_kernel.translate("# formatting\n" + source) == extract_kernel.translate(source),
        "comment changed translation",
    )
    return list(probes)


def require_refusal(operation):
    try:
        operation()
    except PermissionError as exc:
        return str(exc)
    raise ValueError("unexpected permission")


def observed_launch(argv, *, witness, supplied, replacement, **kwargs):
    # After release and sealing, before exec: replace the input pathname with a
    # different, independently reviewed executable that prints identical output.
    replacement.replace(supplied)
    (fd,) = kwargs["pass_fds"]
    require(fcntl.fcntl(fd, fcntl.F_GET_SEALS) & sink.SEALS == sink.SEALS, "missing seal")
    try:
        os.pwrite(fd, b"X", 0)
    except OSError as exc:
        require(exc.errno == errno.EPERM, "unexpected write error")
    else:
        raise ValueError("sealed executable was writable")

    # A full stdout pipe keeps this short-lived command observable until the
    # parent has inspected its executed image. Drain only the known padding.
    read_fd, write_fd = os.pipe()
    proc = None
    try:
        capacity = fcntl.fcntl(write_fd, fcntl.F_GETPIPE_SZ)
        require(os.write(write_fd, bytes(capacity)) == capacity, "pipe setup")
        kwargs["stdout"] = write_fd
        proc = subprocess.Popen(argv, **kwargs)
        base = Path("/proc") / str(proc.pid)
        deadline = time.monotonic() + 10
        while (base / "wchan").read_text().strip() not in ("pipe_write", "anon_pipe_write"):
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("could not observe CLI blocked on stdout")
            time.sleep(0.002)
        exe = base / "exe"
        require(os.path.samestat(os.fstat(fd), exe.stat()), "child executed another inode")
        args = (base / "cmdline").read_bytes().split(b"\0")[:-1]
        require(args == [s.encode() for s in sink.ARGV], "actual arguments changed")
        env = dict(
            row.split(b"=", 1) for row in (base / "environ").read_bytes().split(b"\0") if row
        )
        require(
            env == {k.encode(): v.encode() for k, v in kwargs["env"].items()},
            "actual environment changed",
        )
        require((base / "cwd").resolve() == Path("/"), "actual cwd changed")
        libraries = {}
        for row in (base / "maps").read_text().splitlines():
            parts = row.split(maxsplit=5)
            if "x" not in parts[1]:
                continue
            if len(parts) == 6 and parts[5] in ("[vdso]", "[vsyscall]"):
                continue
            device = tuple(int(x, 16) for x in parts[3].split(":"))
            mapped_ino = int(parts[4])

            def matches(stat):
                return (
                    os.major(stat.st_dev),
                    os.minor(stat.st_dev),
                ) == device and stat.st_ino == mapped_ino

            if matches(os.fstat(fd)):
                continue
            require(len(parts) == 6 and parts[5].startswith("/"), "unknown executable mapping")
            path = Path(parts[5])
            with path.open("rb") as mapped:
                require(matches(os.fstat(mapped.fileno())), "mapped file identity changed")
                libraries[str(path)] = fingerprint(mapped.read())
        witness.update(
            kernel_exe_sha256=fingerprint(exe.read_bytes()),
            same_sealed_inode=True,
            argv=[s.decode() for s in args],
            cwd="/",
            operation_sha256=fingerprint(
                encode(
                    [
                        [s.decode() for s in args],
                        str((base / "cwd").resolve()),
                        {k.decode(): v.decode() for k, v in env.items()},
                    ]
                )
            ),
            mapped_library_digests=sorted(libraries.values()),
            sealed_write_errno="EPERM",
            replacement_sha256=fingerprint(supplied.read_bytes()),
        )
        left = capacity
        while left:
            padding = os.read(read_fd, left)
            require(padding and not any(padding), "stdout observation lost framing")
            left -= len(padding)
        proc.stdout = os.fdopen(read_fd, "rb")
        read_fd = -1
        return proc
    except BaseException:
        if proc is not None:
            proc.kill()
            proc.wait()
        raise
    finally:
        os.close(write_fd)
        if read_fd != -1:
            os.close(read_fd)


def replay(reference, dirs, lean, work):
    raw = reference.read_bytes()
    folded, r2 = sink.reviewed_images(raw)
    expected = ("\n".join(r2["fingerprint"]) + "\n").encode()
    abi = sink.FingerprintSink(raw, dirs, work)
    supplied, replacement = work / "candidate", work / "reference"
    supplied.write_bytes(folded)
    replacement.write_bytes(raw)
    # Caller-side proof and execution outputs can agree on either ELF.
    baseline = subprocess.run(
        [reference, "--abi-fingerprint"],
        env={"LC_ALL": "C", "LD_LIBRARY_PATH": ":".join(map(str, dirs))},
        cwd="/",
        capture_output=True,
        check=True,
        timeout=30,
    )
    require(baseline.stdout == expected and not baseline.stderr, "reference control")
    ticket, decision = abi.authorize(supplied.read_bytes())
    require(ticket is not None, "reviewed fold denied")
    witness = {}

    def observe(argv, **kwargs):
        return observed_launch(
            argv, witness=witness, supplied=supplied, replacement=replacement, **kwargs
        )

    with patch.object(sink, "Popen", observe):
        result = abi.fingerprint(ticket)
    require(
        witness["kernel_exe_sha256"] == r2["selected_sha256"]
        and witness["replacement_sha256"] == r2["reference_sha256"]
        and witness["operation_sha256"] == decision["scope"]["dependencies"]["operation"]
        and witness["kernel_exe_sha256"] == decision["scope"]["artifact"]
        and result == baseline.stdout,
        "replacement changed the consumed image",
    )
    with sink.sealed_image(folded) as fd:
        manifest = sink.loader_manifest(fd, abi._env)
    require(
        decision["scope"]["dependencies"]["loader"] == fingerprint(encode(manifest))
        and sorted(set(manifest.values())) == sorted(set(witness["mapped_library_digests"])),
        "observed loader closure differs from authorization",
    )
    witness["loader_manifest"] = manifest
    witness["stdout"] = result.decode().splitlines()

    refused = {}

    def fresh():
        t, row = abi.authorize(folded)
        require(t is not None and row["status"] == "authorized", "fresh authorization failed")
        return t

    with patch.object(sink, "Popen", side_effect=AssertionError("forbidden dispatch")) as launches:
        refused["spent"] = require_refusal(lambda: abi.fingerprint(ticket))
        checkpoint = work / "abi-policy.json"
        before = checkpoint.read_bytes()
        expired = fresh()
        abi.renew_policy()
        current = checkpoint.read_bytes()
        before_rollback = fresh()
        checkpoint.write_bytes(before)
        refused["restored checkpoint with old ticket"] = require_refusal(
            lambda: abi.fingerprint(expired)
        )
        refused["restored checkpoint with current ticket"] = require_refusal(
            lambda: abi.fingerprint(before_rollback)
        )
        no_ticket, rejected = abi.authorize(folded)
        require(
            no_ticket is None and rejected.get("observed") == "stale policy",
            "rollback allowed issuance",
        )
        checkpoint.write_bytes(current)
        live = fresh()
        # The existing P5 execution sink independently looks up the ticket.
        _, grid_sink = heat.gates(dict)
        refused["ABI ticket presented to heat execution"] = require_refusal(
            lambda: heat.solve(grid_sink.consume(live, folded))
        )
        other = work / "second"
        other.mkdir()
        second = sink.FingerprintSink(raw, dirs, other)
        refused["same operation on different issuer"] = require_refusal(
            lambda: second.fingerprint(live)
        )
        launches.assert_not_called()
    require(abi.fingerprint(live) == expected, "wrong-sink attempt spent the original ticket")
    checks = translator_controls()
    axioms, model = audit_kernel(lean)
    return {
        "schema": 1,
        "r2_sha256": sink.R2_ID,
        "authorization": decision,
        "execution": witness,
        "rollback": {
            "previous": json.loads(before),
            "current": json.loads(current),
            "fresh_decision": rejected,
        },
        "refused": refused,
        "forbidden_native_launches": 0,
        "restored_current_policy_ran": True,
        "translator_rejections": checks,
        "kernel_refinement": model,
        "axioms": axioms.splitlines(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=shutil.which("eshkol-run"))
    parser.add_argument("--llvm-dir", type=Path, default=Path("/usr/lib/llvm-21/lib"))
    parser.add_argument("--lean", default=shutil.which("lean"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.reference or not args.lean:
        parser.error("supply the Eshkol release and Lean 4.35.0")
    reference = args.reference.resolve(strict=True)
    if "version 4.35.0," not in subprocess.check_output([args.lean, "--version"], text=True):
        parser.error("expected Lean 4.35.0")
    dirs = [reference.parent.parent / "lib/eshkol/runtime-deps", args.llvm_dir.resolve(strict=True)]
    with tempfile.TemporaryDirectory() as tmp:
        receipt = replay(reference, dirs, args.lean, Path(tmp))
    text = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
