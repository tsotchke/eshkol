#!/usr/bin/env python3
"""Deterministic, resource-bounded ESKM mutation campaign."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
ARTIFACT_BUDGET = 8 * 1024 * 1024
DEFAULT_SEED = 0x5EED5EED
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_FILE = REPO_ROOT / "scripts" / "icc_traces" / "eskm_model_fuzz.jsonl"
ORACLE_FILE = REPO_ROOT / ".icc" / "completion-oracles.yaml"
RELEASE_EVENT_KIND = "eshkol_smoke"
RELEASE_EVENT_NAME = "eskm_model_fuzz_smoke"
CATEGORIES = (
    "valid-single",
    "valid-multi",
    "valid-scalar",
    "valid-empty",
    "valid-rank32",
    "valid-long-name",
    "valid-many",
    "bad-magic",
    "bad-version",
    "bad-flags",
    "count-overclaim",
    "name-overclaim",
    "rank-overclaim",
    "second-rank-overclaim",
    "shape-overflow",
    "bad-dtype",
    "truncated",
    "trailing",
    "bad-crc",
    "valid-payload-mutation",
)


@dataclasses.dataclass(frozen=True)
class Record:
    name: bytes
    dims: tuple[int, ...]
    bits: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Case:
    ordinal: int
    category: str
    data: bytes
    records: tuple[Record, ...] | None

    @property
    def case_id(self) -> str:
        return f"{self.ordinal:06d}-{self.category}"


@dataclasses.dataclass
class RunResult:
    kind: str
    detail: str
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None


def u32(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFFFFFF)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF)


def encode(records: Sequence[Record], *, version: int = 1, flags: int = 0) -> bytes:
    payload = bytearray(b"ESKM" + u32(version) + u32(len(records)) + u32(flags))
    for record in records:
        payload += u32(len(record.name)) + record.name + u32(len(record.dims))
        for dim in record.dims:
            payload += u64(dim)
        payload.append(0)
        for bits in record.bits:
            payload += u64(bits)
    return bytes(payload) + u32(zlib.crc32(payload))


def with_crc(payload: bytes | bytearray) -> bytes:
    raw = bytes(payload)
    return raw + u32(zlib.crc32(raw))


def put_u32(payload: bytearray, offset: int, value: int) -> None:
    payload[offset : offset + 4] = u32(value)


def put_u64(payload: bytearray, offset: int, value: int) -> None:
    payload[offset : offset + 8] = u64(value)


def fnv_bytes(value: int, data: bytes) -> int:
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def tensor_digest(record: Record) -> str:
    value = fnv_bytes(FNV_OFFSET, u32(len(record.dims)))
    for dim in record.dims:
        value = fnv_bytes(value, u64(dim))
    value = fnv_bytes(value, u64(len(record.bits)))
    for bits in record.bits:
        value = fnv_bytes(value, u64(bits))
    return f"{value:016x}"


def model_digest(records: Sequence[Record]) -> str:
    value = FNV_OFFSET
    for record in records:
        value = fnv_bytes(value, u32(len(record.name)))
        value = fnv_bytes(value, record.name)
        value = fnv_bytes(value, u32(len(record.dims)))
        for dim in record.dims:
            value = fnv_bytes(value, u64(dim))
        value = fnv_bytes(value, u64(len(record.bits)))
        for bits in record.bits:
            value = fnv_bytes(value, u64(bits))
    value = fnv_bytes(value, u32(len(records)))
    return f"{value:016x}"


def expected_line(case: Case) -> str:
    if case.records is None:
        return "model=reject tensor=reject"
    model = f"model=accept:{model_digest(case.records)}"
    if len(case.records) == 1:
        tensor = f"tensor=accept:{tensor_digest(case.records[0])}"
    else:
        tensor = "tensor=reject"
    return f"{model} {tensor}"


def random_record(rng: random.Random, ordinal: int, suffix: str = "") -> Record:
    dims = (rng.randint(1, 3), rng.randint(1, 3))
    name = f"tensor-{ordinal}-{suffix or 'a'}".encode("ascii")
    return record_with_dims(rng, name, dims)


def record_with_dims(rng: random.Random, name: bytes, dims: tuple[int, ...]) -> Record:
    count = 1
    for dim in dims:
        count *= dim
    bits = tuple(rng.getrandbits(64) for _ in range(count))
    return Record(name, dims, bits)


def encoded_record_size(record: Record) -> int:
    return 4 + len(record.name) + 4 + 8 * len(record.dims) + 1 + 8 * len(record.bits)


def mutate_case(seed: int, ordinal: int) -> Case:
    category = CATEGORIES[ordinal % len(CATEGORIES)]
    rng = random.Random(seed ^ (ordinal * 0x9E3779B97F4A7C15))
    first = random_record(rng, ordinal)
    records: tuple[Record, ...] = (first,)

    if category == "valid-multi":
        records = (first, random_record(rng, ordinal, "b"))
        return Case(ordinal, category, encode(records), records)
    if category == "valid-single":
        return Case(ordinal, category, encode(records), records)
    if category == "valid-scalar":
        records = (record_with_dims(rng, b"scalar", ()),)
        return Case(ordinal, category, encode(records), records)
    if category == "valid-empty":
        records = (record_with_dims(rng, b"empty", (0, 3)),)
        return Case(ordinal, category, encode(records), records)
    if category == "valid-rank32":
        records = (record_with_dims(rng, b"rank32", (1,) * 32),)
        return Case(ordinal, category, encode(records), records)
    if category == "valid-long-name":
        records = (record_with_dims(rng, b"n" * (64 * 1024), (1,)),)
        return Case(ordinal, category, encode(records), records)
    if category == "valid-many":
        records = tuple(record_with_dims(rng, f"item-{i}".encode(), (1,)) for i in range(128))
        return Case(ordinal, category, encode(records), records)
    if category == "second-rank-overclaim":
        records = (first, random_record(rng, ordinal, "b"))

    data = encode(records)
    payload = bytearray(data[:-4])
    name_length = len(first.name)
    rank_offset = 20 + name_length
    dims_offset = rank_offset + 4
    dtype_offset = dims_offset + 8 * len(first.dims)
    element_offset = dtype_offset + 1

    if category == "bad-magic":
        payload[0] ^= 0x80
    elif category == "bad-version":
        put_u32(payload, 4, 2 + rng.randrange(0xFFFF))
    elif category == "bad-flags":
        put_u32(payload, 12, 1 << rng.randrange(32))
    elif category == "count-overclaim":
        put_u32(payload, 8, 0xFFFFFFFF)
    elif category == "name-overclaim":
        put_u32(payload, 16, 0xFFFFFFFF)
    elif category == "rank-overclaim":
        put_u32(payload, rank_offset, 0xFFFFFFFF)
    elif category == "second-rank-overclaim":
        second_rank = 16 + encoded_record_size(first) + 4 + len(records[1].name)
        put_u32(payload, second_rank, 0xFFFFFFFF)
    elif category == "shape-overflow":
        put_u64(payload, dims_offset, 1 << 63)
        put_u64(payload, dims_offset + 8, 3)
    elif category == "bad-dtype":
        payload[dtype_offset] = rng.randint(1, 255)
    elif category == "truncated":
        cut = rng.choice((1, 4, 8, max(1, len(payload) - element_offset)))
        del payload[-min(cut, len(payload) - 1) :]
    elif category == "trailing":
        payload += bytes((rng.randrange(256),))
    elif category == "bad-crc":
        corrupted = bytearray(data)
        corrupted[-1] ^= 0x80
        return Case(ordinal, category, bytes(corrupted), None)
    elif category == "valid-payload-mutation":
        mask = 1 << rng.randrange(64)
        changed = first.bits[0] ^ mask
        put_u64(payload, element_offset, changed)
        updated = Record(first.name, first.dims, (changed,) + first.bits[1:])
        return Case(ordinal, category, with_crc(payload), (updated,))
    else:
        raise AssertionError(category)
    return Case(ordinal, category, with_crc(payload), None)


def child_limits(memory_mb: int) -> Callable[[], None] | None:
    if os.name != "posix":
        return None

    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if memory_mb > 0:
            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


def decoded_output(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def run_command(command: Sequence[str], timeout: float, memory_mb: int = 0) -> RunResult:
    child_env = dict(os.environ)
    ubsan = [item for item in child_env.get("UBSAN_OPTIONS", "").split(":")
             if item and not item.startswith("halt_on_error=")]
    child_env["UBSAN_OPTIONS"] = ":".join(("halt_on_error=1", *ubsan))
    try:
        completed = subprocess.run(
            command,
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            preexec_fn=child_limits(memory_mb),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult("hang", f"timeout after {timeout:g}s",
                         decoded_output(exc.stdout), decoded_output(exc.stderr))
    except OSError as exc:
        return RunResult("infra", str(exc))

    if completed.returncode < 0:
        sig = -completed.returncode
        return RunResult("crash", f"signal {sig} ({signal.strsignal(sig)})",
                         completed.stdout, completed.stderr, completed.returncode)
    if completed.returncode != 0:
        return RunResult("exit", f"exit {completed.returncode}", completed.stdout,
                         completed.stderr, completed.returncode)
    return RunResult("ok", "exit 0", completed.stdout, completed.stderr, completed.returncode)


def run_case(probe: Sequence[str], case: Case, scratch: Path, timeout: float,
             memory_mb: int) -> RunResult:
    path = scratch / f"{case.case_id}.eskm"
    path.write_bytes(case.data)
    try:
        result = run_command((*probe, str(path)), timeout, memory_mb)
        if result.kind != "ok":
            return result
        actual = result.stdout.strip()
        expected = expected_line(case)
        if actual != expected:
            return RunResult("oracle", f"expected {expected!r}, got {actual!r}",
                             result.stdout, result.stderr, result.returncode)
        return result
    finally:
        path.unlink(missing_ok=True)


def retain_failure(artifact_dir: Path, case: Case, result: RunResult, seed: int,
                   used_bytes: int) -> int:
    metadata = {
        "case": case.case_id,
        "category": case.category,
        "seed": seed,
        "ordinal": case.ordinal,
        "result": result.kind,
        "detail": result.detail,
        "expected": expected_line(case),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    meta = json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"
    needed = len(case.data) + len(meta)
    if used_bytes + needed > ARTIFACT_BUDGET:
        return used_bytes
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f"{case.case_id}.eskm").write_bytes(case.data)
    (artifact_dir / f"{case.case_id}.json").write_bytes(meta)
    return used_bytes + needed


def release_event(status: str, snippet: str, probe: Path | None = None) -> dict[str, object]:
    event: dict[str, object] = {
        "kind": RELEASE_EVENT_KIND,
        "name": RELEASE_EVENT_NAME,
        "value": status,
        "snippet": snippet[:2000],
        "confidence": 1.0,
        "timestamp": int(time.time()),
    }
    if probe and probe.is_file():
        event["probe_sha256"] = hashlib.sha256(probe.read_bytes()).hexdigest()
    try:
        head = subprocess.run(("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
                              text=True, capture_output=True, check=False)
        if head.returncode == 0:
            event["git_sha"] = head.stdout.strip()
    except OSError:
        pass
    return event


def write_release_event(path: Path, status: str, snippet: str,
                        probe: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(release_event(status, snippet, probe), ensure_ascii=False) + "\n",
                    encoding="utf-8")


def release_criterion(path: Path = ORACLE_FILE) -> tuple[set[str], set[str], set[str]]:
    text = path.read_text(encoding="utf-8")
    start = re.search(r"(?m)^  - name: v1\.3\.5-evolve\s*$", text)
    if not start:
        raise ValueError("v1.3.5-evolve oracle is missing")
    end = re.search(r"(?m)^  - name: ", text[start.end():])
    target = text[start.end():start.end() + end.start()] if end else text[start.end():]

    def values(block: str, key: str) -> set[str]:
        match = re.search(rf"(?m)^\s+{key}:\s*\[([^]]*)\]\s*$", block)
        if not match:
            return set()
        return {item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()}

    for block in re.split(r"(?m)(?=^      - runtime_event:\s*$)", target):
        names = values(block, "event_names")
        if RELEASE_EVENT_NAME in names:
            return values(block, "event_kinds"), names, values(block, "event_values")
    raise ValueError(f"{RELEASE_EVENT_NAME} criterion is missing from v1.3.5-evolve")


def trace_satisfies_release_criterion(
        path: Path, criterion: tuple[set[str], set[str], set[str]]) -> bool:
    if not path.is_file():
        return False
    kinds, names, values = criterion
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (event.get("kind") in kinds and event.get("name") in names and
                event.get("value") in values):
            return True
    return False


def release_run_eligible(*, self_test: bool, replay: int | None, count: int,
                         timeout: float, memory_mb: int,
                         supports_limits: bool) -> bool:
    return (supports_limits and self_test and replay is None and count >= 70 and
            timeout <= 2.0 and memory_mb == 256)


def internal_self_test(timeout: float) -> bool:
    with tempfile.TemporaryDirectory(prefix="eshkol-eskm-fuzz-selftest-") as raw:
        root = Path(raw)
        fake = root / "fake_probe.py"
        fake.write_text(
            "import os, sys, time\n"
            "mode = sys.argv[1]\n"
            "if mode == 'crash': os.abort()\n"
            "if mode == 'hang': print('partial', flush=True); time.sleep(60)\n"
            "if mode == 'env': print(os.environ.get('UBSAN_OPTIONS', '')); sys.exit()\n"
            "print('model=accept:wrong tensor=reject')\n",
            encoding="utf-8",
        )
        crash = run_command((sys.executable, str(fake), "crash"), timeout)
        hang = run_command((sys.executable, str(fake), "hang"), 0.05)
        sanitizer = run_command((sys.executable, str(fake), "env"), timeout)
        wrong_case = mutate_case(DEFAULT_SEED, 0)
        wrong = run_case((sys.executable, str(fake)), wrong_case, root, timeout, 0)
        deterministic = all(mutate_case(DEFAULT_SEED, i) == mutate_case(DEFAULT_SEED, i)
                            for i in range(len(CATEGORIES) * 2))
        abnormal = crash.kind == ("crash" if os.name == "posix" else "exit")
        timeout_output = hang.kind == "hang" and hang.stdout.strip() == "partial"
        sanitizer_halts = "halt_on_error=1" in sanitizer.stdout.strip().split(":")
        oracle_detected = wrong.kind == "oracle"
        finding_dir = root / "findings"
        retained = retain_failure(finding_dir, wrong_case, hang, DEFAULT_SEED, 0)
        finding = json.loads((finding_dir / f"{wrong_case.case_id}.json").read_text())
        artifact_ok = retained > 0 and finding["stdout"].strip() == "partial"
        cleaned = not (root / f"{wrong_case.case_id}.eskm").exists()
        try:
            criterion = release_criterion()
            criterion_wired = criterion == (
                {RELEASE_EVENT_KIND}, {RELEASE_EVENT_NAME}, {"PASS"})
        except (OSError, ValueError):
            criterion = (set(), set(), set())
            criterion_wired = False
        trace = root / "trace.jsonl"
        missing_event_rejected = not trace_satisfies_release_criterion(trace, criterion)
        trace.write_text("\n".join(json.dumps(event) for event in (
            {"kind": "wrong", "name": RELEASE_EVENT_NAME, "value": "PASS"},
            {"kind": RELEASE_EVENT_KIND, "name": "wrong", "value": "PASS"},
            {"kind": RELEASE_EVENT_KIND, "name": RELEASE_EVENT_NAME, "value": "FAIL"},
        )) + "\n", encoding="utf-8")
        wrong_events_rejected = not trace_satisfies_release_criterion(trace, criterion)
        write_release_event(trace, "PASS", "self-test")
        exact_event_accepted = trace_satisfies_release_criterion(trace, criterion)
        partial_runs_rejected = all(not release_run_eligible(**shape) for shape in (
            {"self_test": False, "replay": None, "count": 70,
             "timeout": 2.0, "memory_mb": 256, "supports_limits": True},
            {"self_test": True, "replay": None, "count": 69,
             "timeout": 2.0, "memory_mb": 256, "supports_limits": True},
            {"self_test": True, "replay": 37, "count": 70,
             "timeout": 2.0, "memory_mb": 256, "supports_limits": True},
            {"self_test": True, "replay": None, "count": 70,
             "timeout": 3.0, "memory_mb": 256, "supports_limits": True},
            {"self_test": True, "replay": None, "count": 70,
             "timeout": 2.0, "memory_mb": 0, "supports_limits": True},
            {"self_test": True, "replay": None, "count": 70,
             "timeout": 2.0, "memory_mb": 256, "supports_limits": False},
        ))
        release_shape_accepted = release_run_eligible(
            self_test=True, replay=None, count=70, timeout=2.0, memory_mb=256,
            supports_limits=True)
        ok = all((abnormal, timeout_output, sanitizer_halts, oracle_detected,
                  deterministic, artifact_ok, cleaned, missing_event_rejected,
                  wrong_events_rejected, exact_event_accepted,
                  partial_runs_rejected, release_shape_accepted, criterion_wired))
        print(f"self-test abnormal-exit: {'PASS' if abnormal else 'FAIL'}")
        print(f"self-test timeout-output: {'PASS' if timeout_output else 'FAIL'}")
        print(f"self-test ubsan-halt: {'PASS' if sanitizer_halts else 'FAIL'}")
        print(f"self-test broken-oracle: {'PASS' if oracle_detected else 'FAIL'}")
        print(f"self-test deterministic-replay: {'PASS' if deterministic else 'FAIL'}")
        print(f"self-test artifact-retention: {'PASS' if artifact_ok else 'FAIL'}")
        print(f"self-test scratch-input-cleanup: {'PASS' if cleaned else 'FAIL'}")
        print(f"self-test release-event-missing: {'PASS' if missing_event_rejected else 'FAIL'}")
        print(f"self-test release-event-wrong: {'PASS' if wrong_events_rejected else 'FAIL'}")
        print(f"self-test release-event-exact: {'PASS' if exact_event_accepted else 'FAIL'}")
        print(f"self-test release-event-partial-run: {'PASS' if partial_runs_rejected else 'FAIL'}")
        print(f"self-test release-event-canonical-run: {'PASS' if release_shape_accepted else 'FAIL'}")
        print(f"self-test release-oracle-wiring: {'PASS' if criterion_wired else 'FAIL'}")
        return ok


def parse_int(text: str) -> int:
    return int(text, 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run the 70-case CI-sized campaign")
    mode.add_argument("--full", action="store_true", help="run the 700-case local campaign")
    parser.add_argument("--probe", type=Path, help="path to eskm_model_fuzz_probe")
    parser.add_argument("--seed", type=parse_int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, help="override the mode's generated-case count")
    parser.add_argument("--replay", type=int, metavar="ORDINAL", help="run exactly one ordinal")
    parser.add_argument("--timeout", type=float, default=2.0, help="seconds per process")
    parser.add_argument("--memory-mb", type=int, default=256,
                        help="POSIX address-space cap per probe; use 0 for sanitizer builds")
    parser.add_argument("--artifact-dir", type=Path, help="retain only failing cases here")
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE,
                        help="ICC JSON-L release-evidence path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.timeout <= 0 or args.memory_mb < 0:
        parser.error("--timeout must be positive and --memory-mb non-negative")
    count = args.count if args.count is not None else (700 if args.full else 70)
    if count <= 0:
        parser.error("--count must be positive")
    if args.replay is not None and args.replay < 0:
        parser.error("--replay must be non-negative")
    release_run = bool(args.probe) and release_run_eligible(
        self_test=args.self_test,
        replay=args.replay,
        count=count,
        timeout=args.timeout,
        memory_mb=args.memory_mb,
        supports_limits=os.name == "posix",
    )
    if release_run:
        args.trace_file.parent.mkdir(parents=True, exist_ok=True)
        args.trace_file.write_text("", encoding="utf-8")
    if args.self_test and not internal_self_test(args.timeout):
        if release_run:
            write_release_event(args.trace_file, "FAIL", "fuzz harness negative control failed")
        return 1
    if args.self_test and not args.probe:
        print("PASS: ESKM fuzz harness negative controls")
        return 0
    if not args.probe:
        parser.error("--probe is required unless only --self-test is requested")

    ok = True
    if args.probe:
        probe = args.probe.resolve()
        if not probe.is_file() or not os.access(probe, os.X_OK):
            parser.error(f"probe is not executable: {probe}")
        ordinals = (args.replay,) if args.replay is not None else range(count)
        categories: Counter[str] = Counter()
        failures = 0
        artifact_bytes = 0
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="eshkol-eskm-fuzz-") as raw:
            scratch = Path(raw)
            artifact_dir = args.artifact_dir or (scratch / "findings")
            for ordinal in ordinals:
                assert ordinal is not None
                case = mutate_case(args.seed, ordinal)
                categories[case.category] += 1
                result = run_case((str(probe),), case, scratch, args.timeout, args.memory_mb)
                if result.kind != "ok":
                    failures += 1
                    artifact_bytes = retain_failure(artifact_dir, case, result, args.seed, artifact_bytes)
                    print(f"FAIL: {case.case_id}: {result.kind}: {result.detail}", file=sys.stderr)
        elapsed = time.monotonic() - started
        total = sum(categories.values())
        print(f"ESKM fuzz seed={args.seed} cases={total} failures={failures} elapsed={elapsed:.2f}s")
        print("categories: " + ", ".join(f"{name}={categories[name]}" for name in CATEGORIES))
        print(f"artifact-bytes={artifact_bytes} budget={ARTIFACT_BUDGET}")
        ok = ok and failures == 0
        if release_run:
            write_release_event(
                args.trace_file,
                "PASS" if ok else "FAIL",
                f"seed={args.seed} cases={total} failures={failures} artifact-bytes={artifact_bytes}",
                probe,
            )

    print(f"RESULT: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
