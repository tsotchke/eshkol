#!/usr/bin/env python3
"""Fibers and coroutines gate: behaviour on JIT and AOT, and flat memory.

Runs tests/fibers/{coroutine_primitives,fibers,fibers_stress}_test.esk through
the JIT (-r) and an O1 AOT binary, requiring each program's own PASS marker,
exit status 0 and no FAIL line. Then it runs the AOT stress program with one
wave and with four waves of 10,000 fibers and compares the peak resident set
of the two processes. Each wave's fibers leave their Scheme records in the
arena (closures and fiber records; the arena is reclaimed by regions, not per
object), which the program reports as arena-bytes; what remains of the RSS
growth after subtracting the arena growth is what the coroutine runtime kept,
and it must stay under a few MiB. A stack that were not given back would cost
at least two pages per fiber: 80 MiB or more per wave of 10,000.
"""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

TESTS = (
    ("coroutine_primitives_test.esk", "PASS: coroutine_primitives_test"),
    ("fibers_test.esk", "PASS: fibers_test"),
    ("fibers_stress_test.esk", "PASS: fibers_stress_test"),
)

# Allowed growth per extra wave of 10,000 fibers of peak RSS not explained by
# the arena (malloc arenas and page-granular arena blocks round up).
GROWTH_LIMIT_BYTES_PER_WAVE = 8 * 1024 * 1024


def run_measured(command, env, cwd, timeout):
    """Run a command; return (exit status, stdout, stderr, peak RSS bytes)."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err)
        deadline = time.monotonic() + timeout
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() > deadline:
                proc.kill()
                pid, status, usage = os.wait4(proc.pid, 0)
                status = -1
                break
            time.sleep(0.02)
        proc.returncode = os.waitstatus_to_exitcode(status) if status >= 0 else -1
        out.seek(0)
        err.seek(0)
        rss = usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        return (proc.returncode, out.read().decode(errors="replace"),
                err.read().decode(errors="replace"), rss)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path,
                        default=root / os.environ.get("BUILD_DIR", "build") / "eshkol-run")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    runner = args.runner.resolve()
    env = dict(os.environ, ESHKOL_PATH=str(root / "lib"),
               ESHKOL_LIB_DIR=str(runner.parent), ESHKOL_JIT_CACHE="0")

    outputs = {}

    def stage(label, command, marker=None, extra_env=None):
        e = dict(env, **(extra_env or {}))
        status, out, err, rss = run_measured(command, e, root, args.timeout)
        outputs[label] = out
        print(out, end="")
        ok = (status == 0 and "FAIL:" not in out + err
              and (marker is None or marker in out.splitlines()))
        if not ok:
            print(err, end="", file=sys.stderr)
            print(f"FAIL: {label} (exit {status})")
            return None
        print(f"PASS: {label} (peak RSS {rss / (1024 * 1024):.1f} MiB)")
        return rss

    with tempfile.TemporaryDirectory(prefix="eshkol-fibers-") as work:
        binaries = {}
        for test, marker in TESTS:
            source = root / "tests/fibers" / test
            if stage(f"{test} JIT", [str(runner), "-r", str(source),
                                    "-L" + str(runner.parent)], marker) is None:
                return 1
            binary = Path(work) / (source.stem + (".exe" if os.name == "nt" else ".bin"))
            if stage(f"{test} O1 AOT compile",
                     [str(runner), "-O", "1", "-o", str(binary), str(source),
                      "-L" + str(runner.parent)]) is None:
                return 1
            if stage(f"{test} O1 AOT execute", [str(binary)], marker) is None:
                return 1
            binaries[test] = binary

        stress = binaries["fibers_stress_test.esk"]
        one = stage("stress, 1 wave of 10000 fibers", [str(stress)],
                    "PASS: fibers_stress_test", {"FIBER_STRESS_WAVES": "1"})
        four = stage("stress, 4 waves of 10000 fibers", [str(stress)],
                     "PASS: fibers_stress_test", {"FIBER_STRESS_WAVES": "4"})
        if one is None or four is None:
            return 1

        def arena_bytes(label):
            for line in outputs[label].splitlines():
                if line.startswith("arena-bytes: "):
                    return int(line.split()[1])
            return None

        arena_one = arena_bytes("stress, 1 wave of 10000 fibers")
        arena_four = arena_bytes("stress, 4 waves of 10000 fibers")
        if arena_one is None or arena_four is None:
            print("FAIL: the stress program did not report arena-bytes")
            return 1
        arena_growth = max(0, arena_four - arena_one) / 3
        growth = max(0, (four - one) - (arena_four - arena_one)) / 3
        print(f"peak RSS: 1 wave {one / 2**20:.1f} MiB, 4 waves {four / 2**20:.1f} MiB; "
              f"arena growth per extra wave {arena_growth / 2**20:.2f} MiB; "
              f"growth beyond the arena per extra wave {growth / 2**20:.2f} MiB "
              f"(limit {GROWTH_LIMIT_BYTES_PER_WAVE / 2**20:.0f} MiB)")
        if growth > GROWTH_LIMIT_BYTES_PER_WAVE:
            print("FAIL: fiber memory grows with the number of finished fibers' stacks")
            return 1
        print("PASS: fiber memory is flat across waves")
    print("PASS: fibers gate (JIT and AOT, 10000-fiber stress, flat memory)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
