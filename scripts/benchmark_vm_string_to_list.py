#!/usr/bin/env python3
"""Portable paired benchmark for the validated Eshkol VM string-to-list workload."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

EXPECTED = b"262144\n"
PASSES = 32
MIN_START = 21 * 2**30
MIN_STOP = int(20.25 * 2**30)
GUARD_SECONDS = 60


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_arm(exe, bytecode):
    free = shutil.disk_usage(pathlib.Path.cwd()).free
    if free < MIN_START:
        raise RuntimeError(f"disk reserve below 21 GiB at start ({free} bytes)")
    state = {"reserve_stop": False, "minimum_free_bytes": free}
    stop = threading.Event()
    holder = {}

    def stop_child(proc):
        if os.name == "nt":
            if proc.poll() is not None:
                return
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=3, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        # A leader can exit on TERM while a descendant ignores it. Clear the
        # residual owned POSIX group even when wait() reaped the leader.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def monitor():
        while not stop.wait(0.1):
            avail = shutil.disk_usage(pathlib.Path.cwd()).free
            state["minimum_free_bytes"] = min(state["minimum_free_bytes"], avail)
            if avail < MIN_STOP:
                state["reserve_stop"] = True
                proc = holder.get("proc")
                if proc is not None:
                    stop_child(proc)
                return

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    env = os.environ.copy()
    env["ESHKOL_VM_NO_DISASM"] = "1"
    timed_out = False
    start = time.perf_counter_ns()
    popen_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([str(exe), str(bytecode)], cwd=os.getcwd(), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_options)
    holder["proc"] = proc
    previous_handlers = {}
    def interrupt(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"received signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, interrupt)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=GUARD_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_child(proc)
            stdout, stderr = proc.communicate(timeout=2)
        finished = time.perf_counter_ns()
    except BaseException:
        stop_child(proc)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        stop.set()
        thread.join()
        if proc.poll() is None:
            stop_child(proc)
    elapsed = finished - start
    valid = (proc.returncode == 0 and stdout == EXPECTED and not timed_out
             and not state["reserve_stop"])
    return {"elapsed_ns": elapsed, "seconds_per_conversion": elapsed / 1e9 / PASSES,
            "exit_code": proc.returncode, "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "correct_output": stdout == EXPECTED, "timed_out": timed_out,
            "pass": valid, **state}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="baseline VM executable")
    p.add_argument("--candidate", required=True, help="candidate VM executable")
    p.add_argument("--bytecode", required=True, help="precompiled .eskb module")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", help="write JSON here (default: stdout)")
    a = p.parse_args()
    if a.repeats < 1 or a.warmup < 0:
        p.error("--repeats must be positive and --warmup nonnegative")
    arms = {"baseline": os.path.abspath(a.baseline), "candidate": os.path.abspath(a.candidate)}
    bytecode = os.path.abspath(a.bytecode)
    for name, path in [*arms.items(), ("bytecode", bytecode)]:
        if not os.path.isfile(path):
            p.error(f"{name} file does not exist: {path}")
    if shutil.disk_usage(os.getcwd()).free < MIN_START:
        p.error("disk reserve below 21 GiB before benchmark")
    hashes = {name: digest(path) for name, path in arms.items()}
    bytecode_hash = digest(bytecode)
    all_runs = []
    for arm in arms:
        for i in range(a.warmup):
            result = run_arm(arms[arm], bytecode)
            if not result["pass"]:
                raise RuntimeError(f"{arm} warmup {i + 1} failed validation")
    # Pair order alternates; with odd repeats, baseline leads one extra pair.
    for pair in range(a.repeats):
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        for arm in order:
            result = run_arm(arms[arm], bytecode)
            if not result["pass"]:
                raise RuntimeError(f"{arm} pair {pair + 1} failed validation")
            all_runs.append({"pair": pair + 1, "arm": arm, **result})
    for name, path in arms.items():
        if digest(path) != hashes[name]:
            raise RuntimeError(f"{name} executable changed during benchmark")
    if digest(bytecode) != bytecode_hash:
        raise RuntimeError("bytecode changed during benchmark")
    values = {name: [r["seconds_per_conversion"] for r in all_runs if r["arm"] == name]
              for name in arms}
    output = {
        "schema": "eshkol.vm_string_to_list_benchmark.v1",
        "workload_scope": "One precompiled VM process for 32 conversions of an 8192-codepoint lambda string; whole process startup, allocation, validation and output are included, elapsed time divided by 32.",
        "validation": {"stdout_exact_utf8": EXPECTED.decode("utf-8").rstrip("\n"), "stdout_bytes": len(EXPECTED), "trailing_newline_validated": EXPECTED.endswith(b"\n"), "all_runs_validated": True},
        "configuration": {"repeats_per_arm": a.repeats, "warmups_per_arm": a.warmup,
                           "pair_interleaved": True, "guard_seconds_per_child": GUARD_SECONDS,
                           "disk_free_start_bytes": MIN_START, "disk_free_stop_bytes": MIN_STOP},
        "artifacts": {"baseline_sha256": hashes["baseline"], "candidate_sha256": hashes["candidate"],
                      "bytecode_sha256": bytecode_hash},
        "raw_samples_seconds_per_conversion": values,
        "means_seconds_per_conversion": {k: statistics.mean(v) for k, v in values.items()},
        "runs": all_runs,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if a.output:
        pathlib.Path(a.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
