#!/usr/bin/env python3
"""Reject false success in the public acceptance driver, using synthetic executables."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="socket-runner-contract-", dir=args.work_root)).resolve()
    source = args.source_dir.resolve()
    driver = source / "tests/agent/run_unix_socket_stream_pending.py"
    rows = []
    fixture = work / "libobservation.dylib"
    fixture.write_text("not loaded by synthetic executable\n")
    fake = work / "synthetic-runner"
    fake.write_text("#!" + sys.executable + r'''
import os,sys,pathlib
mode=os.environ['SOCKET_RUNNER_CONTROL']
marker='PASS: unix-socket capability_denial'
if '--emit-eskb' in sys.argv:
 if mode=='vm-artifact':pathlib.Path(sys.argv[sys.argv.index('--emit-eskb')+1]).write_bytes(b'partial')
 print('unrelated parser error' if mode=='vm-unrelated' else 'unsupported or malformed VM extern declaration')
 raise SystemExit(1)
if '-o' in sys.argv:
 path=pathlib.Path(sys.argv[sys.argv.index('-o')+1])
 path.write_text('#!'+sys.executable+'\nprint('+repr(marker)+')\n')
 path.chmod(0o700)
 raise SystemExit(0)
if mode!='missing':print(marker)
if mode=='duplicate':print(marker)
if mode=='failure-marker':print('FAIL: unrelated failure')
if mode=='jit-error':print('JIT session error: controlled missing symbol')
if mode=='llvm-error':print('LLVM ERROR: controlled compiler failure')
raise SystemExit(3 if mode=='nonzero' else 0)
''')
    fake.chmod(0o700)
    for control, mode, good in [
        ("success", "jit", True), ("missing", "jit", False),
        ("duplicate", "jit", False), ("failure-marker", "jit", False),
        ("nonzero", "jit", False), ("jit-error", "jit", False),
        ("llvm-error", "jit", False), ("success", "aot", True),
        ("vm-specific", "vm", True), ("vm-unrelated", "vm", False),
        ("vm-artifact", "vm", False),
    ]:
        argv = [sys.executable, str(driver), "--runner", str(fake), "--mode", mode,
                "--scenario", "capability_denial", "--work-root", str(work / "retained"),
                "--source", str(source / "tests/agent/unix_socket_stream_pending.esk"),
                "--peer", str(fake), "--fixture-library", str(fixture)]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=10,
                           env={**os.environ, "SOCKET_RUNNER_CONTROL": control})
        (work / (mode + "-" + control + ".stdout")).write_text(r.stdout)
        (work / (mode + "-" + control + ".stderr")).write_text(r.stderr)
        assert r.returncode == (0 if good else 1), (mode, control, r.stdout, r.stderr)
        result = json.loads(r.stdout)
        assert (result["status"] == "PASS") == good
        if good:
            assert result["socket_directory_removed"]
            assert not Path(result["socket_directory"]).exists()
        rows.append({"case": mode + ":" + control, "expected_pass": good, "status": result["status"]})
    # Exercise timeout cleanup directly with one owned, isolated child.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("socket_runner", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.transfer_budget(2) == 45
    assert module.transfer_budget(20) == 110
    assert module.transfer_budget(100) == 180
    rows.append({"case": "measured-compile-budget-bounded", "status": "PASS"})
    preferred = work / "temp"
    preferred.mkdir()
    original_tempdir = module.tempfile.tempdir
    try:
        module.tempfile.tempdir = str(preferred)
        scratch = module.socket_scratch()
        expected = preferred if len(os.fsencode(preferred)) <= 40 else Path("/tmp")
        assert scratch.parent == expected
        scratch.rmdir()
        module.tempfile.tempdir = "/tmp"
        scratch = module.socket_scratch()
        assert scratch.parent == Path("/tmp")
        assert len(os.fsencode(scratch / ("x" * 32))) < 104
        scratch.rmdir()
        private_parent = Path(tempfile.mkdtemp(prefix="sktmp-", dir="/tmp"))
        module.tempfile.tempdir = str(private_parent)
        scratch = module.socket_scratch()
        assert scratch.parent == private_parent
        assert len(os.fsencode(scratch / ("x" * 32))) < 104
        scratch.rmdir()
        private_parent.rmdir()
    finally:
        module.tempfile.tempdir = original_tempdir
    rows.append({"case": "scratch-tempdir-path-budget", "status": "PASS"})
    original_killpg, original_run = module.os.killpg, module.subprocess.run
    def denied_probe(pid, sig):
        raise PermissionError("controlled visibility denial")
    try:
        module.os.killpg = denied_probe
        module.subprocess.run = lambda *a, **k: SimpleNamespace(returncode=0, stdout="12345\n")
        assert module.group_alive(12345)
        rows.append({"case": "denied-probe-live-group", "status": "LIVE_AS_EXPECTED"})
        module.subprocess.run = lambda *a, **k: SimpleNamespace(returncode=0, stdout="12346\n")
        assert not module.group_alive(12345)
        rows.append({"case": "denied-probe-absent-group", "status": "ABSENT_AS_OBSERVED"})
        module.subprocess.run = lambda *a, **k: SimpleNamespace(returncode=1, stdout="")
        try:
            module.group_alive(12345)
            raise AssertionError("unreadable group state accepted")
        except RuntimeError:
            rows.append({"case": "unreadable-group-state", "status": "REFUSED_AS_EXPECTED"})
    finally:
        module.os.killpg, module.subprocess.run = original_killpg, original_run
    pid_file = work / "timed-child.pid"
    code = "import os,pathlib,time;pathlib.Path(" + repr(str(pid_file)) + ").write_text(str(os.getpid()));time.sleep(10)"
    start = time.monotonic()
    try:
        module.run_owned([sys.executable, "-c", code], work / "timeout.log", os.environ.copy(), timeout=0.5)
        raise AssertionError("timeout accepted")
    except RuntimeError as e:
        assert "deadline exceeded" in str(e)
    assert time.monotonic() - start < 5
    assert not module.group_alive(int(pid_file.read_text()))
    rows.append({"case": "timeout-cleanup", "expected_pass": False, "status": "FAIL_AS_EXPECTED"})
    result = {"status": "PASS_RUNNER_ORACLE_CONTROLS", "controls": len(rows), "rows": rows,
              "product_executions": 0, "work": str(work)}
    (work / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
