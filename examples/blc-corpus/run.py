#!/usr/bin/env python3
"""Check the standalone corpus or execute it on an explicitly supplied backend.

Python 3.11+ is dependency free; Python 3.10 may use tomli. Runtime execution
uses POSIX process groups and explicit time/address-space limits.
"""
import argparse
import hashlib
import functools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

try:
    import tomllib
except ImportError:
    import tomli as tomllib

BASE = Path(__file__).resolve().parent
LIBRARY = BASE / "lib" if (BASE / "lib").is_dir() else BASE.parents[1] / "lib"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def version_space(frame):
    if not frame or frame[0] > 81 or any(x > 7 for x in frame[1:]):
        return [254]
    spaces = []
    for code in range(81):
        digits = [(code // 3**p) % 3 for p in range(4)]
        spaces.append(frozenset(t for t in range(16) if all(
            d == 0 or (t >> p) & 1 == d - 1 for p, d in enumerate(digits))))
    spaces.append(frozenset())
    remaining = spaces[frame[0]]
    for action in frame[1:]:
        point, label = divmod(action, 2)
        remaining = frozenset(t for t in remaining if (t >> point) & 1 == label)
    return [spaces.index(remaining)]


def polynomial(frame):
    invalid = [255, 0, 0, 0, 0, 0]
    if (len(frame) < 6 or frame[0] > 16 or frame[1] > 8
            or len(frame) != 6 + 3 * frame[1]
            or any(x > 8 for x in frame[2:6]) or any(x > 1 for x in frame[6:])):
        return invalid
    state = [x - 4 for x in frame[2:6]]
    accepted = 0
    for _ in range(frame[0]):
        for offset in range(6, len(frame), 3):
            x, y, target = frame[offset:offset + 3]
            features = [x, y, x * y, 1]
            prediction = int(sum(a * b for a, b in zip(state, features)) > 0)
            next_state = [a + (target - prediction) * b for a, b in zip(state, features)]
            if any(abs(x) > 4 for x in next_state):
                return [1] + [x + 4 for x in state] + [accepted]
            state = next_state
            accepted += 1
    return [0] + [x + 4 for x in state] + [accepted]


ORACLES = {"version-space": version_space, "polynomial": polynomial}
PROGRAMS = set(ORACLES) | {"grad-depth", "selfapply", "seed", "active-query", "transformer"}


def active_query(frame):
    if (len(frame) < 2 or frame[0] > 8 or frame[1] > 16 or len(frame) != frame[1] + 2
            or frame[2:] != sorted(set(frame[2:])) or any(x > 15 for x in frame[2:])):
        return [254, 4, 0]
    masks = tuple(frame[2:])
    if frame[0] < 8:
        point, label = divmod(frame[0], 2)
        masks = tuple(m for m in masks if (m >> point) & 1 == label)

    @functools.lru_cache(None)
    def cost(values):
        if len(values) < 2:
            return 0
        return min(len(values) + cost(left) + cost(right) for p in range(4)
                   if (left := tuple(m for m in values if not (m >> p) & 1))
                   and (right := tuple(m for m in values if (m >> p) & 1)))

    if len(masks) < 2:
        query = 4
    else:
        query = min((len(masks) + cost(left) + cost(right), p) for p in range(4)
                    if (left := tuple(m for m in masks if not (m >> p) & 1))
                    and (right := tuple(m for m in masks if (m >> p) & 1)))[1]
    return [0 if masks else 1, query, len(masks)] + list(masks)


ORACLES["active-query"] = active_query


MAX_BITS = 1000000
MAX_NODES = 500000
MAX_DEPTH = 4096
MAX_FILE_BYTES = 4 * 1024 * 1024


def corpus_bytes(relative, limit=MAX_FILE_BYTES):
    """Bounded regular-file read from an immutable checkout; reject symlinks."""
    import stat
    relative = Path(relative)
    if (relative.is_absolute() or not relative.parts
            or any(p in ("..", ".") for p in relative.parts)):
        raise ValueError("unsafe corpus path")
    path = BASE
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in corpus path")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("corpus file type or byte budget")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("corpus file byte budget")
    return data


def closed_classic(bits):
    """Bounded iterative prefix/scope admission; return a readable node table.

    This never evaluates a term. Nodes are postorder (tag, child...) tuples;
    every child precedes its parent. No Python reader/eval or recursion.
    """
    if not bits or len(bits) > MAX_BITS or set(bits) - {"0", "1"}:
        raise ValueError("expected bounded classic binary code")
    cursor, entered = 0, 0
    tasks, values, nodes = [("term", 0, 0)], [], []
    while tasks:
        task, scope, nesting = tasks.pop()
        if task == "lambda":
            nodes.append(("lambda", values.pop()))
            values.append(len(nodes) - 1)
        elif task == "apply":
            right, left = values.pop(), values.pop()
            nodes.append(("apply", left, right))
            values.append(len(nodes) - 1)
        else:
            entered += 1
            if entered > MAX_NODES or nesting > MAX_DEPTH or cursor >= len(bits):
                raise ValueError("truncated term or syntax budget")
            if bits[cursor] == "1":
                start = cursor
                while cursor < len(bits) and bits[cursor] == "1":
                    cursor += 1
                    if cursor - start > scope:
                        raise ValueError("open variable")
                if cursor == len(bits):
                    raise ValueError("truncated variable")
                nodes.append(("bound", cursor - start))
                values.append(len(nodes) - 1)
                cursor += 1
            else:
                tag = bits[cursor:cursor + 2]
                cursor += 2
                if tag == "00":
                    tasks.extend([("lambda", 0, 0), ("term", scope + 1, nesting + 1)])
                elif tag == "01":
                    tasks.extend([("apply", 0, 0), ("term", scope, nesting + 1), ("term", scope, nesting + 1)])
                else:
                    raise ValueError("truncated constructor")
    if cursor != len(bits) or len(values) != 1:
        raise ValueError("trailing classic BLC data")
    return nodes


def canonical_bits(program):
    raw = corpus_bytes(program["bits_path"], MAX_BITS + 1)
    bits = raw[:-1].decode("ascii") if raw.endswith(b"\n") else ""
    if not bits or raw != bits.encode("ascii") + b"\n":
        raise ValueError("expected ASCII bits and exactly one final newline")
    closed_classic(bits)
    return bits


def check():
    manifest = tomllib.loads(corpus_bytes("manifest.toml", 512 * 1024).decode("utf-8"))
    if manifest["schema_version"] != 1 or manifest["license"] != "MIT":
        raise ValueError("unknown corpus contract")
    if not 1 <= len(manifest["file"]) <= 128:
        raise ValueError("manifest file budget")
    paths = set()
    for item in manifest["file"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative in paths:
            raise ValueError("unsafe or duplicated manifest path")
        paths.add(relative)
        if digest(corpus_bytes(relative)) != item["sha256"]:
            raise ValueError("file hash differs: " + item["path"])
    programs = set()
    for program in manifest["program"]:
        if (program["id"] not in PROGRAMS or program["id"] in programs
                or Path(program["bits_path"]) not in paths
                or Path(program["packed_path"]) not in paths):
            raise ValueError("program identity or sealed path differs")
        programs.add(program["id"])
        bits = canonical_bits(program)
        packed = corpus_bytes(program["packed_path"])
        closed_classic(bits)
        padded = bits + "0" * (-len(bits) % 8)
        if (len(bits) != program["logical_bits"] or len(packed) != program["packed_bytes"]
                or -len(bits) % 8 != program["padding_bits"]
                or packed != bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))):
            raise ValueError("packed/text identity differs")
    if programs != PROGRAMS:
        raise ValueError("missing program")
    fixtures = tomllib.loads(corpus_bytes("fixtures.toml").decode("utf-8"))["fixture"]
    if len(fixtures) != manifest["fixture_count"] or not 1 <= len(fixtures) <= 1024:
        raise ValueError("fixture count differs")
    names = set()
    for case in fixtures:
        if (not re.fullmatch(r"[a-z0-9-]+", case["id"]) or case["id"] in names
                or len(case["input"]) > 65536 or len(case["expected"]) > 65536
                or any(type(x) is not int or not 0 <= x <= 255 for x in case["input"] + case["expected"])):
            raise ValueError("invalid fixture")
        names.add(case["id"])
        if case["program"] in ORACLES and ORACLES[case["program"]](case["input"]) != case["expected"]:
            raise ValueError("independent fixture oracle differs: " + case["id"])
        if case["program"] in {"seed", "selfapply"}:
            if not case["expected"] or case["expected"][0] != 0:
                raise ValueError("missing emitted-code success status")
            closed_classic("".join(str(x) for x in case["expected"][1:]))
    return manifest, fixtures


def command(argv, payload, timeout, memory_mib):
    import resource

    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (memory_mib * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(timeout) + 1,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2,) * 2)

    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True,
                               preexec_fn=limits)
    # Bound pipes as well as elapsed time: communicate() alone can exhaust
    # the parent process on a hostile or broken backend's stdout/stderr.
    import selectors
    selector = selectors.DefaultSelector()
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    for stream in buffers:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    timed_out, output_limited = False, False
    deadline = started + timeout
    try:
        if payload:
            # Corpus inputs are bounded by admission, and fit in the pipe on
            # current fixtures. A separate writer prevents a blocked stdin
            # from escaping the same wall deadline on future inputs.
            import threading
            def send():
                try:
                    process.stdin.write(payload)
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            writer = threading.Thread(target=send, daemon=True)
            writer.start()
        else:
            process.stdin.close()
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(min(remaining, 0.1)):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                else:
                    buffers[key.fileobj].extend(data)
                    if sum(map(len, buffers.values())) > 8 * 1024**2:
                        output_limited = True
                        break
            if output_limited:
                break
        if timed_out or output_limited:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return {"exit_code": process.returncode, "wall_seconds": time.monotonic() - started,
            "timed_out": timed_out, "output_limited": output_limited,
            "stdout": bytes(buffers[process.stdout]), "stderr": bytes(buffers[process.stderr])}


def execute(args, manifest, fixtures, work):
    programs = {p["id"]: p for p in manifest["program"]}
    rows = []
    for case in fixtures:
        program = programs[case["program"]]
        binary = str(Path(args.binary).resolve(strict=True))
        phase = "execute"
        if args.backend == "tromp":
            payload = corpus_bytes(program["packed_path"]) + bytes(case["input"])
            result = command([binary], payload, args.timeout, args.memory_mib)
        else:
            bits = canonical_bits(program)
            probe = work / (case["id"] + ".esk")
            data = work / (case["id"] + ".input")
            data.write_text("(" + " ".join(map(str, case["input"])) + ")\n")
            work_limit = args.max_work if args.max_work is not None else case.get("work_limit", 100000000)
            capacity = args.heap_capacity if args.heap_capacity is not None else case.get("heap_capacity", 250000)
            output_limit = case.get("output_limit", 65536)
            probe.write_text((BASE / "probe.esk").read_text() + "\n(learner-probe "
                             + json.dumps(case["id"]) + " " + json.dumps(bits)
                             + " (learner-read " + json.dumps(str(data)) + ") "
                             + f"{work_limit} {capacity} {output_limit})\n")
            prefix = ["env", "ESHKOL_JIT_CACHE=0", "ESHKOL_JIT_COMPILE_THREADS=1",
                      "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", binary,
                      "--no-stdlib", "-I", str(LIBRARY), "-L", str(LIBRARY), "-O", "0"]
            if args.backend == "eshkol-jit":
                result = command(prefix + ["-r", str(probe)], b"", args.timeout, args.memory_mib)
            else:
                artifact = work / (case["id"] + ".aot")
                phase = "compile"
                result = command(prefix + ["-o", str(artifact), str(probe)], b"", args.timeout, args.memory_mib)
                (work / (case["id"] + ".compile.stderr")).write_bytes(result["stderr"])
                if result["exit_code"] == 0 and not result["timed_out"]:
                    phase = "execute"
                    result = command([str(artifact)], b"", args.timeout, args.memory_mib)
        (work / (case["id"] + ".stdout")).write_bytes(result["stdout"])
        (work / (case["id"] + ".stderr")).write_bytes(result["stderr"])
        actual = None
        status = "process-failure"
        counters = [-1] * 6
        if result["output_limited"]:
            status = "output-limit"
        elif result["timed_out"]:
            status = "wall-timeout"
        elif result["exit_code"] != 0 and any(message in result["stderr"].lower()
                for message in (b"out of memory", b"failed to allocate")):
            status = "allocation-failure"
        elif result["exit_code"] == 0 and phase == "execute":
            if args.backend == "tromp":
                actual = list(result["stdout"])
            else:
                text = result["stdout"].decode("utf-8", "replace")
                match = re.fullmatch(r"START " + re.escape(case["id"]) + r"\nOBS "
                                     + re.escape(case["id"])
                                     + r" \(([a-z-]+) \(([\d ]*)\)((?: \d+){6})\)\n", text)
                if match:
                    actual = [int(x) for x in match[2].split()]
                    counters = [int(x) for x in match[3].split()]
                    if match[1] != "normal":
                        status = match[1]
                    elif counters[0] > work_limit or counters[5] >= capacity:
                        status = "invalid-counters"
                    else:
                        status = "pass" if actual == case["expected"] else "mismatch"
            if args.backend == "tromp" and actual is not None:
                status = "pass" if actual == case["expected"] else "mismatch"
        row = {"id": case["id"], "program": case["program"], "status": status,
               "phase": phase, "input": case["input"], "expected": case["expected"],
               "actual": actual if actual is not None else [], "machine_counters": counters,
               "exit_code": result["exit_code"], "wall_seconds": result["wall_seconds"],
               "stdout_sha256": digest(result["stdout"]), "stderr_sha256": digest(result["stderr"])}
        rows.append(row)
        print(case["id"], status, flush=True)
    return rows


def report(path, args, rows):
    values = {"schema_version": 1, "backend": args.backend,
              "binary_sha256": digest(Path(args.binary).read_bytes()),
              "manifest_sha256": args.manifest_sha256,
              "runner_sha256": args.runner_sha256,
              "module_sha256": args.module_sha256,
              "timeout_seconds": args.timeout, "memory_mib": args.memory_mib,
              "cases": len(rows),
              "passed": sum(r["status"] == "pass" for r in rows)}
    def fields(value):
        return "".join(f"{k} = {json.dumps(v)}\n" for k, v in value.items())
    Path(path).write_text(fields(values) + "".join("\n[[case]]\n" + fields(row) for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "run"])
    parser.add_argument("--backend", choices=["tromp", "eshkol-jit", "eshkol-aot"])
    parser.add_argument("--binary")
    parser.add_argument("--fixture", action="append")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--max-work", type=int)
    parser.add_argument("--heap-capacity", type=int)
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--work-dir")
    parser.add_argument("--report")
    args = parser.parse_args()
    args.manifest_sha256 = digest(corpus_bytes("manifest.toml", 512 * 1024))
    args.runner_sha256 = digest(Path(__file__).read_bytes())
    args.module_sha256 = digest((LIBRARY / "core/blc_bytes.esk").read_bytes()) if (LIBRARY / "core/blc_bytes.esk").exists() else ""
    manifest, fixtures = check()
    if args.mode == "check":
        print(f'CHECK-PASS: {len(manifest["program"])} closed programs, {len(fixtures)} sealed fixtures')
        return 0
    if (not args.backend or not args.binary or not args.report or os.name != "posix"
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or (args.max_work is not None and args.max_work < 0)
            or (args.heap_capacity is not None and args.heap_capacity < 16)
            or args.memory_mib < 64):
        parser.error("run requires backend, binary, report, POSIX and positive finite resource bounds")
    if args.fixture:
        if set(args.fixture) - {c["id"] for c in fixtures}:
            parser.error("unknown fixture")
        fixtures = [c for c in fixtures if c["id"] in args.fixture]
    with tempfile.TemporaryDirectory(prefix="blc-learners-") as temporary:
        work = Path(args.work_dir).resolve() if args.work_dir else Path(temporary)
        work.mkdir(parents=True, exist_ok=True)
        rows = execute(args, manifest, fixtures, work)
    report(args.report, args, rows)
    return 0 if all(r["status"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
