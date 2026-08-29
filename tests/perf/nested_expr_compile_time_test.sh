#!/usr/bin/env bash
# ESH-0103: deeply nested native expressions must scale approximately linearly.
#
# ArithmeticCodegen used to inline the complete numeric tower at every node of
# one nested expression. LLVM's SROA pass then had to promote O(depth) allocas
# across O(depth) basic blocks in one function, producing a quadratic
# compile-time and RSS curve. The dispatch is now emitted once per operator as
# a module-local noinline helper; this gate keeps that architectural fix
# measurable on both native execution paths.
#
# The ladder is deliberately wide: 1,000 -> 4,000 -> 16,000 nested additions.
# A 4x depth step may take at most ESH0103_RATIO_MAX times as long or consume
# that multiple of peak RSS. A quadratic curve is approximately 16x per step
# and therefore fails even with the generous default threshold of 8x.
#
# All generated input and measurement logs live below the worktree's
# .scratch/ directory. The gate never uses the host temporary directory. The
# measurement child reserves up to 512 MiB of virtual stack because the parser
# checks actual stack headroom before descending; only resident pages count in
# the RSS measurement.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ESHKOL_RUN="${1:-${ESHKOL_RUN:-$REPO_ROOT/build/eshkol-run}}"

if [ "${1:-}" = "--self-test" ]; then
    ESHKOL_RUN=""
    SELF_TEST=1
else
    SELF_TEST=0
fi

if [ "$SELF_TEST" -eq 0 ] && [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: nested_expr_compile_time_test could not execute $ESHKOL_RUN" >&2
    exit 1
fi

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

python3 - "$REPO_ROOT" "$ESHKOL_RUN" "$SELF_TEST" <<'PY'
import json
import atexit
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

repo_root = Path(sys.argv[1])
compiler = sys.argv[2]
self_test = sys.argv[3] == "1"


def positive_int(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"FAIL: {name} must be a positive integer, got {raw!r}")
    if value <= 0:
        raise SystemExit(f"FAIL: {name} must be positive, got {value}")
    return value


def positive_float(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"FAIL: {name} must be a positive number, got {raw!r}")
    if value <= 0:
        raise SystemExit(f"FAIL: {name} must be positive, got {value}")
    return value


D1 = positive_int("ESH0103_D1", 1000)
D2 = positive_int("ESH0103_D2", 4000)
D3 = positive_int("ESH0103_D3", 16000)
if not D1 < D2 < D3:
    raise SystemExit(f"FAIL: depths must be strictly increasing, got {D1}, {D2}, {D3}")
RATIO_MAX = positive_float("ESH0103_RATIO_MAX", 8.0)
ABS_CEIL_SECONDS = positive_float("ESH0103_ABS_CEIL_SECONDS", 120.0)
RSS_CEIL_MB = positive_float("ESH0103_RSS_CEIL_MB", 2048.0)


def grade(samples):
    """Return (passed, diagnostics) for the measured mode/depth ladder."""
    diagnostics = []
    passed = True
    for mode in ("jit", "aot"):
        mode_samples = sorted(
            (sample for sample in samples if sample["mode"] == mode),
            key=lambda sample: sample["depth"],
        )
        if len(mode_samples) != 3:
            diagnostics.append(f"{mode}: expected three samples, got {len(mode_samples)}")
            passed = False
            continue
        for sample in mode_samples:
            if sample["returncode"] != 0:
                diagnostics.append(
                    f"{mode} depth {sample['depth']}: compiler exited {sample['returncode']}"
                )
                passed = False
            if sample["elapsed_s"] <= 0 or sample["rss_mb"] <= 0:
                diagnostics.append(
                    f"{mode} depth {sample['depth']}: invalid measurement "
                    f"time={sample['elapsed_s']} rss={sample['rss_mb']}"
                )
                passed = False
            if sample["elapsed_s"] > ABS_CEIL_SECONDS:
                diagnostics.append(
                    f"{mode} depth {sample['depth']}: time {sample['elapsed_s']:.3f}s "
                    f"> {ABS_CEIL_SECONDS:.3f}s ceiling"
                )
                passed = False
            if sample["rss_mb"] > RSS_CEIL_MB:
                diagnostics.append(
                    f"{mode} depth {sample['depth']}: RSS {sample['rss_mb']:.1f}MB "
                    f"> {RSS_CEIL_MB:.1f}MB ceiling"
                )
                passed = False
        for before, after in zip(mode_samples, mode_samples[1:]):
            time_ratio = after["elapsed_s"] / before["elapsed_s"]
            rss_ratio = after["rss_mb"] / before["rss_mb"]
            diagnostics.append(
                f"{mode} {before['depth']}->{after['depth']}: "
                f"time_ratio={time_ratio:.3f} rss_ratio={rss_ratio:.3f}"
            )
            if time_ratio > RATIO_MAX:
                diagnostics.append(
                    f"{mode} {before['depth']}->{after['depth']}: "
                    f"time ratio {time_ratio:.3f} > {RATIO_MAX:.3f}"
                )
                passed = False
            if rss_ratio > RATIO_MAX:
                diagnostics.append(
                    f"{mode} {before['depth']}->{after['depth']}: "
                    f"RSS ratio {rss_ratio:.3f} > {RATIO_MAX:.3f}"
                )
                passed = False
    return passed, diagnostics


if self_test:
    linear = [
        {"mode": mode, "depth": depth, "elapsed_s": float(depth),
         "rss_mb": float(depth), "returncode": 0}
        for mode in ("jit", "aot")
        for depth in (1, 4, 16)
    ]
    quadratic = [
        {"mode": mode, "depth": depth, "elapsed_s": float(depth * depth),
         "rss_mb": float(depth * depth), "returncode": 0}
        for mode in ("jit", "aot")
        for depth in (1, 4, 16)
    ]
    old_limits = (RATIO_MAX, ABS_CEIL_SECONDS, RSS_CEIL_MB)
    RATIO_MAX = 8.0
    ABS_CEIL_SECONDS = 1000000.0
    RSS_CEIL_MB = 1000000.0
    linear_ok, _ = grade(linear)
    quadratic_ok, _ = grade(quadratic)
    RATIO_MAX, ABS_CEIL_SECONDS, RSS_CEIL_MB = old_limits
    if not linear_ok or quadratic_ok:
        print("FAIL: nested-expression compile-time gate self-test")
        print(f"      linear accepted={linear_ok}, quadratic rejected={not quadratic_ok}")
        raise SystemExit(1)
    print("PASS: nested-expression compile-time gate self-test")
    raise SystemExit(0)


scratch = repo_root / ".scratch"
scratch.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="esh0103.", dir=scratch))
atexit.register(shutil.rmtree, work, ignore_errors=True)


def generate(depth, path):
    with path.open("w", encoding="ascii") as handle:
        handle.write("(+ 1 " * depth)
        handle.write("0")
        handle.write(")" * depth)
        handle.write("\n")


worker = r'''
import json
import os
import resource
import subprocess
import sys
import time

compiler, mode, source, output, log, cache = sys.argv[1:]
# The parser deliberately checks actual remaining stack instead of imposing a
# depth limit. Give this bounded measurement child enough stack for the 16k
# probe; the extra virtual address space is not resident unless used and is
# included honestly in the measured RSS if the parser needs it.
try:
    stack_soft, stack_hard = resource.getrlimit(resource.RLIMIT_STACK)
    stack_target = 512 * 1024 * 1024
    if stack_hard == resource.RLIM_INFINITY or stack_hard >= stack_target:
        resource.setrlimit(resource.RLIMIT_STACK, (stack_target, stack_hard))
except (OSError, ValueError):
    pass
argv = [compiler, "-n"]
if mode == "jit":
    argv += ["-r", source]
else:
    argv += ["-O0", "-c", source, "-o", output]
env = os.environ.copy()
env["ESHKOL_JIT_CACHE_DIR"] = cache
start = time.monotonic()
with open(log, "w", encoding="utf-8") as stream:
    completed = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
elapsed = time.monotonic() - start
raw_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
# Linux reports KiB; macOS reports bytes. This gate runs on Unix native lanes.
rss_mb = raw_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else raw_rss / 1024.0
print(json.dumps({"returncode": completed.returncode, "elapsed_s": elapsed, "rss_mb": rss_mb}))
'''


def measure(mode, depth):
    source = work / f"{mode}-{depth}.esk"
    output = work / f"{mode}-{depth}.o"
    log = work / f"{mode}-{depth}.log"
    cache = work / f"jit-cache-{mode}-{depth}"
    cache.mkdir()
    generate(depth, source)
    process = subprocess.Popen(
        [sys.executable, "-c", worker, compiler, mode, str(source), str(output), str(log), str(cache)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=ABS_CEIL_SECONDS + 5.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return {"mode": mode, "depth": depth, "returncode": 124,
                "elapsed_s": ABS_CEIL_SECONDS + 5.0, "rss_mb": RSS_CEIL_MB + 1.0}
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "measurement worker failed"
        if log.is_file():
            detail += " :: " + log.read_text(encoding="utf-8", errors="replace")[-1200:].replace("\n", " ")
        print(f"FAIL: {mode} depth {depth} measurement worker: {detail}", file=sys.stderr)
        return {"mode": mode, "depth": depth, "returncode": process.returncode,
                "elapsed_s": 0.0, "rss_mb": 0.0}
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        print(f"FAIL: {mode} depth {depth} returned invalid measurement: {stdout!r}", file=sys.stderr)
        result = {"returncode": 1, "elapsed_s": 0.0, "rss_mb": 0.0}
    result.update({"mode": mode, "depth": depth})
    if result["returncode"] != 0 and log.is_file():
        detail = log.read_text(encoding="utf-8", errors="replace")[-1200:].replace("\n", " ")
        print(f"FAIL: {mode} depth {depth} compiler output: {detail}", file=sys.stderr)
    if mode == "aot" and result["returncode"] == 0 and not output.is_file():
        result["returncode"] = 1
        print(f"FAIL: AOT depth {depth} produced no object at {output}", file=sys.stderr)
    return result


print("ESH-0103 nested-expression compile-time/RSS gate")
print(f"  compiler : {compiler}")
print(f"  depths   : {D1} -> {D2} -> {D3}")
print(f"  limits   : ratio <= {RATIO_MAX}, time <= {ABS_CEIL_SECONDS}s, RSS <= {RSS_CEIL_MB}MB")
samples = []
for mode in ("jit", "aot"):
    for depth in (D1, D2, D3):
        sample = measure(mode, depth)
        samples.append(sample)
        print(
            f"  {mode:3s} depth {depth:5d}: "
            f"time={sample['elapsed_s']:.3f}s rss={sample['rss_mb']:.1f}MB "
            f"rc={sample['returncode']}"
        )

passed, diagnostics = grade(samples)
for diagnostic in diagnostics:
    print(f"  scaling : {diagnostic}")

trace_dir = Path(os.environ.get("ESH0103_TRACE_DIR", str(repo_root / "scripts" / "icc_traces")))
trace_dir.mkdir(parents=True, exist_ok=True)
trace = {
    "kind": "performance_budget",
    "name": "nested_expr_compile_time_gate",
    "value": "PASS" if passed else "FAIL",
    "snippet": "; ".join(diagnostics[-6:])[:1000],
    "confidence": 0.95,
    "depths": [D1, D2, D3],
    "samples": samples,
}
with (trace_dir / "nested_expr_compile_time_gate.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(trace, sort_keys=True) + "\n")

if not passed:
    print("FAIL: nested-expression compile time/RSS scales super-linearly (ESH-0103)", file=sys.stderr)
    print("      The 4x ladder exceeded the configured near-linear ratio or absolute ceiling.", file=sys.stderr)
    raise SystemExit(1)
print("PASS: nested-expression compile-time/RSS scaling is approximately linear.")
PY
