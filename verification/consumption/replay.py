"""Exercise R1 on the released CLI with a kernel-observed, sealed execution."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import ast
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
from unittest.mock import patch

import sink
from binding import LaunchFailure, Phase, SEALS
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


def observed_launch(
    argv, *, witness, supplied, replacement, sentinel=None, at_output_wait=None, **kwargs
):
    # After release and sealing, before exec: replace the input pathname with a
    # different, independently reviewed executable that prints identical output.
    replacement.replace(supplied)
    (fd,) = kwargs["pass_fds"]
    require(fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS == SEALS, "missing seal")
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
        child_fds = {int(path.name): path.stat() for path in (base / "fd").iterdir()}
        require(os.path.samestat(child_fds[0], Path("/dev/null").stat()), "stdin identity")
        require(os.path.samestat(child_fds[1], os.fstat(write_fd)), "stdout identity")
        require(
            stat.S_ISFIFO(child_fds[2].st_mode)
            and not os.path.samestat(child_fds[1], child_fds[2]),
            "stderr identity",
        )
        require(os.path.samestat(child_fds[fd], exe.stat()), "inherited image descriptor")
        witness["observed_extra_fds"] = sorted(set(child_fds) - {0, 1, 2, fd})
        if sentinel is not None:
            require(
                not any(os.path.samestat(sentinel, value) for value in child_fds.values()),
                "parent sentinel leaked into child",
            )
            witness["inheritable_parent_file_excluded"] = True
        if at_output_wait is not None:
            at_output_wait(libraries)
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


def exercise_failures(abi, folded, expected):
    rows = {}
    for case, expected_phase, real_run in (
        ("preparation failed", Phase.RELEASED, False),
        ("launch failed before child creation", Phase.REQUESTED, False),
        ("confirmation lost after child exit", Phase.REQUESTED, True),
    ):
        ticket, row = abi.authorize(folded)
        require(row["status"] == "authorized", case)
        observed = []
        requests = 0

        def launch(*args, **kwargs):
            nonlocal requests
            requests += 1
            if real_run:
                with subprocess.Popen(*args, **kwargs) as child:
                    out, err = child.communicate(timeout=30)
                    require(child.returncode == 0 and not err and out == expected, case)
                    observed.append(fingerprint(out))
            raise OSError("launch confirmation unavailable")

        with patch.object(sink, "Popen", launch):
            # _context also seals a descriptor for the loader probe. Fail only
            # the subsequent snapshot, after the gate has returned the payload.
            original = sink.sealed_image
            snapshots = 0

            def prepare(data):
                nonlocal snapshots
                snapshots += 1
                if case == "preparation failed" and snapshots == 2:
                    raise OSError("snapshot unavailable")
                return original(data)

            with patch.object(sink, "sealed_image", prepare):
                try:
                    abi.fingerprint(ticket)
                except LaunchFailure as exc:
                    require(exc.phase == expected_phase, case)
                else:
                    raise ValueError(case)
            refused = require_refusal(lambda: abi.fingerprint(ticket))
        require(
            requests == int(expected_phase is Phase.REQUESTED) and len(observed) == int(real_run),
            "incorrect launch cardinality: " + case,
        )
        rows[case] = {
            "launch_calls_including_retry": requests,
            "last_confirmed": expected_phase.value,
            "children_observed": len(observed),
            "completed_output_digests": observed,
            "retry": refused,
        }
    ticket, decision = abi.authorize(folded)
    require(decision["status"] == "authorized", "race admission")
    start = Barrier(2)

    def compete():
        start.wait(timeout=10)
        try:
            require(abi.fingerprint(ticket) == expected, "race output")
        except PermissionError:
            return "spent"
        return "published"

    with patch.object(sink, "Popen", wraps=subprocess.Popen) as launch:
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(compete) for _ in range(2)]
            outcomes = sorted(f.result(timeout=60) for f in futures)
        require(outcomes == ["published", "spent"] and launch.call_count == 1, "duplicate dispatch")
    rows["two simultaneous callers"] = {"outcomes": outcomes, "launch_calls": launch.call_count}
    return rows


def post_start_dependency(reference, dirs, work, folded, expected):
    # Only this replay's child uses this private copy of the release libraries.
    directory = work / "private-runtime"
    shutil.copytree(dirs[0], directory)
    selected = directory / "libpng16.so.16"
    policy_dir = work / "dependency-policy"
    policy_dir.mkdir()
    owner = sink.FingerprintSink(reference.read_bytes(), [directory, *dirs[1:]], policy_dir)
    ticket, decision = owner.authorize(folded)
    require(decision["status"] == "authorized", "dependency control admission")
    supplied, replacement = work / "dep-candidate", work / "dep-reference"
    supplied.write_bytes(folded)
    replacement.write_bytes(reference.read_bytes())
    witness = {}

    def change(libraries):
        require(str(selected.resolve()) in libraries, "private dependency was not mapped")
        original = libraries[str(selected.resolve())]
        with selected.open("rb") as held:
            marker = directory / "withdrawn"
            marker.write_bytes(b"dependency withdrawn during replay\n")
            marker.replace(selected)
            require(fingerprint(held.read()) == original, "held dependency changed")
        witness["mapped_dependency"] = original
        witness["replacement"] = fingerprint(selected.read_bytes())
        require(original != witness["replacement"], "dependency control did not change bytes")

    def observe(argv, **kwargs):
        proc = observed_launch(
            argv,
            witness=witness,
            supplied=supplied,
            replacement=replacement,
            at_output_wait=change,
            **kwargs,
        )
        communicate = proc.communicate

        def capture(**options):
            out, err = communicate(**options)
            require(
                proc.returncode == 0 and out == expected and not err,
                "child result during withdrawal",
            )
            witness["child_output_sha256"] = fingerprint(out)
            return out, err

        proc.communicate = capture
        return proc

    with patch.object(sink, "Popen", observe):
        try:
            owner.fingerprint(ticket)
        except LaunchFailure as exc:
            require(
                exc.phase == Phase.EXITED
                and str(exc.__cause__) == "startup dependency changed before publication",
                "wrong publication failure",
            )
            witness["failure"] = str(exc)
        else:
            raise ValueError("published under withdrawn dependency")
    witness["retry"] = require_refusal(lambda: owner.fingerprint(ticket))
    # The withdrawal is now visible to admission as well. Read the changed
    # closure as ordinary data; no further loader probe is needed.
    missing, rejected = owner.authorize(folded)
    require(
        missing is None and rejected.get("observed") == "changed loader closure",
        "readmission after withdrawal",
    )
    witness["fresh_status"] = rejected["status"]
    selected.rename(directory / "absent-from-closure")
    absent, incomplete = owner.authorize(folded)
    require(
        absent is None
        and incomplete["status"] == "inconclusive"
        and incomplete.get("missing") == ["loader"],
        "missing dependency admission",
    )

    return {
        "dependency": "libpng16.so.16",
        "mapped_sha256": witness["mapped_dependency"],
        "replacement_sha256": witness["replacement"],
        "child_output_sha256": witness["child_output_sha256"],
        "kernel_exe_sha256": witness["kernel_exe_sha256"],
        "failure": witness["failure"],
        "retry": witness["retry"],
        "fresh_status": witness["fresh_status"],
        "missing_file_status": incomplete["status"],
    }


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
            argv,
            witness=witness,
            supplied=supplied,
            replacement=replacement,
            sentinel=sentinel,
            **kwargs,
        )

    with tempfile.TemporaryFile(dir=work) as parent_file:
        os.set_inheritable(parent_file.fileno(), True)
        sentinel = os.fstat(parent_file.fileno())
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
        manifest = sink.loader_files(fd, dict(abi.binding.environment)).manifest()
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
    profile = json.loads(abi.binding.dependency_contract)
    require(
        (json.dumps(profile, indent=2) + "\n").encode() == abi.binding.dependency_contract,
        "profile fixture serialization",
    )
    profile["fixed"]["later_dynamic_loading"] = profile["outside"].pop("later_dynamic_loading")
    try:
        sink.reviewed_context((json.dumps(profile, indent=2) + "\n").encode())
    except ValueError as exc:
        require(str(exc) == "unsupported ABI execution contract", "wrong profile refusal")
    else:
        raise ValueError("accepted a lifetime code-loading requirement")
    require(
        decision["scope"]["dependencies"]["dependency contract"] == sink.ABI_CONTEXT_ID,
        "execution policy absent from ticket",
    )
    failures = exercise_failures(abi, folded, expected)
    withdrawal = post_start_dependency(reference, dirs, work, folded, expected)
    checks = translator_controls()
    axioms, model = audit_kernel(lean)
    return {
        "schema": 2,
        "dependency_contract": json.loads(abi.binding.dependency_contract),
        "launch_attempts": failures,
        "lifetime_loading_profile": "unsupported",
        "post_start_dependency": withdrawal,
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
