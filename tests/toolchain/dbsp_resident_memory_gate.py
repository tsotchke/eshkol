"""Native fixed-live-state memory family, with an unreclaimed negative control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def sample(binary, ticks, threshold, poison):
    import resource  # Linux gate; no RSS claim on unsupported platforms.
    env = dict(os.environ, ESHKOL_RESIDENT_TICKS=str(ticks),
               ESHKOL_RESIDENT_THRESHOLD=str(threshold), ESHKOL_ARENA_POISON=str(poison))
    start = time.monotonic()
    run = subprocess.run([str(binary)], env=env, text=True, capture_output=True, timeout=300)
    fields = {k: int(v) for k, v in re.findall(r"^(TICKS|LIVE|PREUSED|USED|COMPACTIONS|COPIED): ([0-9]+)$",
                                               run.stdout, re.M)}
    return dict(ticks=ticks, threshold=threshold, poison=poison, returncode=run.returncode,
                seconds=time.monotonic() - start,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                fields=fields, stdout=run.stdout, stderr=run.stderr,
                passed=run.returncode == 0 and "RESULT: PASS\n" in run.stdout
                and not re.search(r"FAIL:|RESULT: FAIL|fatal signal", run.stdout + run.stderr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--lib-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--sample-binary", type=Path)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--threshold", type=int, default=1048576)
    parser.add_argument("--poison", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if sys.platform != "linux":
        print("SKIP: this RSS gate requires Linux")
        return 77
    if args.sample_binary:
        print(json.dumps(sample(args.sample_binary, args.ticks, args.threshold, args.poison)))
        return 0
    for name in ("runner", "source", "lib_dir", "work_dir"):
        if getattr(args, name) is None:
            parser.error("missing --" + name.replace("_", "-"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    binary = args.work_dir / "dbsp_resident_benchmark"
    command = [str(args.runner), "-O", "1", "-o", str(binary), str(args.source), "-L" + str(args.lib_dir)]
    compile_run = subprocess.run(command, text=True, capture_output=True, timeout=720)
    (args.work_dir / "compile.log").write_text(compile_run.stdout + compile_run.stderr)
    if compile_run.returncode:
        print("FAIL: benchmark compilation")
        return 1
    rows = []
    # Alternating sizes, separate fresh child and resource ledger for each run.
    negatives = []
    for poison in (0, 1):
        for repeat in range(3):
            for ticks in ((1024, 8192, 65536) if repeat % 2 == 0 else (65536, 8192, 1024)):
                process = subprocess.run([sys.executable, __file__, "--sample-binary", str(binary),
                                          "--ticks", str(ticks), "--poison", str(poison)],
                                         text=True, capture_output=True, timeout=330)
                if process.returncode:
                    raise RuntimeError(process.stderr)
                row = json.loads(process.stdout)
                rows.append(row)
                if not row["passed"]:
                    raise RuntimeError("memory sample failed: " + row["stderr"])
        control = subprocess.run([sys.executable, __file__, "--sample-binary", str(binary),
                                  "--ticks", "4096", "--threshold", "1073741824", "--poison", str(poison)],
                                 text=True, capture_output=True, timeout=330)
        negatives.append(json.loads(control.stdout))
    medians = {poison: {n: statistics.median(r["peak_rss_kib"] for r in rows
                                            if r["ticks"] == n and r["poison"] == poison)
                       for n in (1024, 8192, 65536)} for poison in (0, 1)}
    passed = all(r["passed"] and r["fields"]["TICKS"] == r["ticks"]
                 and r["fields"]["LIVE"] == 1 and r["fields"]["USED"] < 524288
                 and r["fields"]["PREUSED"] <= 1048576 + 524288
                 and r["fields"]["COMPACTIONS"] > 1 for r in rows)
    for poison in (0, 1):
        negative = negatives[poison]
        passed = passed and medians[poison][65536] <= medians[poison][1024] + 8192
        passed = passed and negative["passed"] and negative["fields"]["COMPACTIONS"] == 1
        passed = passed and negative["fields"]["PREUSED"] > 8388608
        passed = passed and negative["peak_rss_kib"] > max(medians[poison].values()) + 8192
    evidence = dict(samples=rows, negative_controls=negatives, median_peak_rss_kib=medians,
                    compiler_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), passed=passed)
    (args.work_dir / "memory-results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"median_peak_rss_kib": medians, "control_peak_rss_kib": [n["peak_rss_kib"] for n in negatives],
                      "passed": passed}))
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
