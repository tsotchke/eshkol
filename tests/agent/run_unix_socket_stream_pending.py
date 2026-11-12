#!/usr/bin/env python3
"""Run one bounded socket acceptance case; retain commands, logs and fixtures."""
import argparse
import json
import os
import re
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Some host/interpreter profiles deny signal0 after the leader exits.
        # Do not equate denial with absence: require a successful read-only
        # process-group inventory, or fail closed.
        observed = subprocess.run(["ps", "-axo", "pgid="], capture_output=True,
                                  text=True, timeout=2)
        tokens = observed.stdout.split()
        if observed.returncode or any(not x.isdigit() for x in tokens):
            raise RuntimeError("cannot verify owned process-group membership")
        return pid in {int(x) for x in tokens}


def stop_group(process):
    for sig, seconds in [(signal.SIGTERM, 2), (signal.SIGKILL, 2)]:
        if not group_alive(process.pid):
            process.wait()
            return
        os.killpg(process.pid, sig)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_alive(process.pid):
                process.wait()
                return
            time.sleep(0.01)
    raise RuntimeError("owned test process group did not stop")


def run_owned(argv, log, env, timeout=45):
    with log.open("xb") as stream:
        process = subprocess.Popen([str(x) for x in argv], stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True, restore_signals=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_group(process)
            raise RuntimeError("owned test deadline exceeded: " + str(log))
        if group_alive(process.pid):
            stop_group(process)
            raise RuntimeError("test returned with owned descendants: " + str(log))
    if log.stat().st_size > 8 * 1024 * 1024:
        raise RuntimeError("test output exceeds retained display budget")
    return code, log.read_text(errors="replace")


def socket_scratch():
    # Honour TMPDIR when it fits sun_path. A short /tmp fallback is necessary
    # on hosts whose normal temporary directory exceeds the Unix path limit.
    preferred = Path(tempfile.gettempdir())
    parent = preferred if len(os.fsencode(preferred)) <= 55 else Path("/tmp")
    return Path(tempfile.mkdtemp(prefix="eshkol-unix-", dir=parent))


def transfer_budget(compile_seconds):
    # Compilation under contention measures available throughput for this run.
    # Keep both the caller's transfer loop and the outer process bound finite.
    return min(180, max(45, 30 + 4 * compile_seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--mode", choices=["jit", "aot", "vm"], required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--peer", type=Path)
    parser.add_argument("--fixture-library", type=Path)
    parser.add_argument("--control-library", type=Path)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=args.scenario + "-" + args.mode + "-", dir=args.work_root)).resolve()
    sockets = socket_scratch()
    sockets.chmod(0o700)
    result = {"status": "FAIL", "mode": args.mode, "scenario": args.scenario,
              "socket_directory": str(sockets), "work": str(work), "commands": [],
              "group_verification": "signal0; successful ps group inventory on permission denial"}
    env = os.environ.copy()
    for name in ["LD_PRELOAD", "DYLD_INSERT_LIBRARIES"]:
        env.pop(name, None)
    env.update(ESHKOL_JIT_CACHE="0", ESHKOL_UNIX_SOCKET_PEER_BUDGET_MS="300000",
               ESHKOL_UNIX_STREAM_SOCKET_SCENARIO=args.scenario,
               ESHKOL_UNIX_STREAM_SOCKET_SCRATCH=str(sockets),
               ESHKOL_PATH=str(args.source.resolve().parents[2] / "lib"),
               ESHKOL_LIB_DIR=str(args.runner.resolve().parent))
    peer = None
    peer_log = None
    try:
        if args.mode == "vm":
            output = work / "unsupported.eskb"
            argv = [args.runner, "--profile", "hosted-vm", "--emit-eskb", output, args.source]
            result["commands"].append([str(x) for x in argv])
            code, text = run_owned(argv, work / "vm.log", env)
            if code != 1 or "unsupported or malformed VM extern declaration" not in text or output.exists():
                raise RuntimeError("expected explicit pre-emission VM unsupported result")
        else:
            if not args.fixture_library or not args.peer:
                parser.error("public cases require --fixture-library and --peer")
            library = args.fixture_library.resolve(strict=True)
            name = library.name
            if not name.startswith("lib") or not name.endswith((".so", ".dylib")):
                raise RuntimeError("unexpected fixture library name")
            link_name = name[3:].rsplit(".", 1)[0]
            build = args.build_dir.resolve() if args.build_dir else library.parent
            env["ESHKOL_LIB_DIR"] = str(build)
            env["LD_LIBRARY_PATH"] = str(library.parent)
            env["DYLD_LIBRARY_PATH"] = str(library.parent)
            # -l supplies AOT link flags; the JIT resolves process-visible symbols.
            # Load only this test observation/control DSO, never a socket core.
            env["DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"] = str(library)
            result["fixture_loading"] = "observation DSO has no connect interposer; separate pending control is labelled when loaded"
            if sys.platform == "darwin" and args.scenario in ["deadline_connect", "connect_cancellation", "zero_budget"]:
                if not args.control_library:
                    raise RuntimeError("Darwin pending cases require the separate control DSO")
                env["ESHKOL_UNIX_SOCKET_TEST_PENDING_PATH"] = str(sockets / "stream.sock")
                env["DYLD_INSERT_LIBRARIES"] = str(library) + ":" + str(args.control_library.resolve(strict=True))
                result["pending_control"] = "deterministic EAGAIN injection; not OS-pending evidence"
            else:
                env.pop("ESHKOL_UNIX_SOCKET_TEST_PENDING_PATH", None)
            if args.scenario in ["deadline_connect", "connect_cancellation", "zero_budget", "destination_errors"]:
                peer_log = (work / "peer.log").open("xb")
                mode = "--leftover-peer" if args.scenario == "destination_errors" else "--backlog-peer"
                argv = [str(args.peer.resolve(strict=True)), mode, str(sockets)]
                result["commands"].append(argv)
                peer = subprocess.Popen(argv, stdout=peer_log, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True, restore_signals=True)
                deadline = time.monotonic() + 5
                while not (sockets / "ready").exists():
                    if peer.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("controlled peer did not become ready")
                    time.sleep(0.01)
                ready = (sockets / "ready").read_text().strip()
                expected = ("leftover-ready" if mode == "--leftover-peer" else
                            "darwin-controlled-pending" if sys.platform == "darwin" else
                            "linux-backlog-saturated")
                if ready != expected:
                    raise RuntimeError("peer readiness does not confirm its controlled state")
                result["peer_ready_state"] = ready
            base = [args.runner, "-L", build, "-L", library.parent, "-l", link_name]
            budget = 45
            if args.scenario == "nul_and_large_payload" and args.mode == "jit":
                # A compiler-only pass measures this exact fixture without
                # running it. JIT execution still compiles and runs normally.
                probe = [*base, "-O0", "-c", "-o", work / "compile-probe", args.source]
                result["commands"].append([str(x) for x in probe])
                started = time.monotonic()
                code, _ = run_owned(probe, work / "compile-probe.log", env, timeout=90)
                if code:
                    raise RuntimeError("fixture compile-time measurement did not complete")
                result["measured_compile_seconds"] = time.monotonic() - started
                budget = transfer_budget(result["measured_compile_seconds"])
            if args.mode == "jit":
                argv = [*base, "-r", args.source]
            else:
                executable = work / "socket-acceptance"
                argv = [*base, "-O0", "-o", executable, args.source]
                result["commands"].append([str(x) for x in argv])
                started = time.monotonic()
                code, text = run_owned(argv, work / "compile.log", env, timeout=90)
                if code:
                    raise RuntimeError("AOT compilation did not complete")
                result["measured_compile_seconds"] = time.monotonic() - started
                if args.scenario == "nul_and_large_payload":
                    budget = transfer_budget(result["measured_compile_seconds"])
                argv = [executable]
            if args.scenario == "nul_and_large_payload":
                result["transfer_budget_seconds"] = budget
                env["ESHKOL_UNIX_SOCKET_TRANSFER_BUDGET_MS"] = str(int(budget * 1000))
            result["commands"].append([str(x) for x in argv])
            code, text = run_owned(argv, work / "run.log", env, timeout=budget)
            marker = "PASS: unix-socket " + args.scenario
            if code != 0 or text.splitlines().count(marker) != 1 or re.search(r"^FAIL:|JIT session error|LLVM ERROR:", text, re.M):
                raise RuntimeError("public socket contract not met; see retained run.log")
        result["status"] = "PASS"
    except (OSError, RuntimeError) as error:
        result["reason"] = str(error)
    finally:
        if peer is not None:
            (sockets / "stop").write_text("stop\n")
            try:
                code = peer.wait(timeout=2)
                if code or group_alive(peer.pid):
                    result.update(status="FAIL", reason="controlled peer did not finish cleanly")
                    stop_group(peer)
            except subprocess.TimeoutExpired:
                stop_group(peer)
                result.update(status="FAIL", reason="controlled peer required termination")
        if peer_log:
            peer_log.close()
        if result["status"] == "PASS":
            try:
                shutil.rmtree(sockets)
                result["socket_directory_removed"] = True
            except OSError as error:
                result.update(status="FAIL", reason="socket scratch cleanup: " + str(error))
        (work / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
