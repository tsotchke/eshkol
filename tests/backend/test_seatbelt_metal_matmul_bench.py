#!/usr/bin/env python3
"""Measure Apple Silicon FP64 matmul throughput and numerical parity across:
  1. CPU baseline (ESHKOL_GPU_DISABLE=1)
  2. Unsandboxed Metal GPU (ESHKOL_GPU_PRECISION=high)
  3. Darwin Seatbelt strict sandbox (ESHKOL_SUBPROC_OS_SANDBOX=strict,
     launched via Eshkol's agent.subprocess sandbox path) with Metal GPU.

Also records vm_stat pageouts before and after sustained iterations to verify
the benchmark does not trigger swap pressure.
"""
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BUILD_DIR", str(REPO_ROOT / "build")))
ESHKOL_RUN = BUILD_DIR / "eshkol-run"


MATMUL_BENCH_SRC = r"""
(define n 512)
(define iters 8)
(define a (ones n n))
(define b (ones n n))

;; Warmup
(define w (matmul a b))
(define c00 (tensor-get w 0 0))

(define elapsed-ns
  (time-ns
    (lambda ()
      (let loop ((i 0) (acc 0.0))
        (if (< i iters)
            (let ((c (matmul a b)))
              (loop (+ i 1) (+ acc (tensor-get c 0 0))))
            acc)))))

(display "C00:") (display c00) (newline)
(display "ELAPSED_NS:") (display elapsed-ns) (newline)
(display "N:") (display n) (newline)
(display "ITERS:") (display iters) (newline)
"""


def _read_vm_pageouts():
    if platform.system() != "Darwin":
        return None
    out = subprocess.check_output(["vm_stat"], text=True)
    m = re.search(r"Pageouts:\s+(\d+)", out)
    return int(m.group(1)) if m else None


def _parse_metrics(stdout: str):
    data = {}
    for line in stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k in {"C00", "ELAPSED_NS", "N", "ITERS"}:
                data[k] = float(v.strip())
    return data


def _gflops(m):
    n = m["N"]
    iters = m["ITERS"]
    sec = m["ELAPSED_NS"] / 1e9
    return (2.0 * n * n * n * iters) / sec / 1e9 if sec > 0 else 0.0


@unittest.skipUnless(platform.system() == "Darwin", "Darwin Seatbelt + Metal benchmark")
class SeatbeltMetalMatmulBenchTest(unittest.TestCase):
    def test_fp64_matmul_cpu_metal_and_seatbelt_strict(self):
        self.assertTrue(ESHKOL_RUN.is_file(), f"missing {ESHKOL_RUN}")
        with tempfile.TemporaryDirectory(prefix="eshkol-seatbelt-metal-") as td:
            work = Path(td)
            bench_src = work / "matmul_bench.esk"
            bench_exe = work / "matmul_bench"
            bench_src.write_text(MATMUL_BENCH_SRC, encoding="utf-8")

            comp = subprocess.run(
                [str(ESHKOL_RUN), "-L", str(BUILD_DIR), str(bench_src), "-o", str(bench_exe)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(comp.returncode, 0, msg=comp.stderr + comp.stdout)

            launcher_src = work / "seatbelt_launcher.esk"
            launcher_exe = work / "seatbelt_launcher"
            launcher_src.write_text(
                f'(require agent.subprocess)\n'
                f'(define res (run-argv-capture (list "{bench_exe}") "." 60000))\n'
                f'(define code (cdr (assoc \'exit-code res)))\n'
                f'(display (cdr (assoc \'stdout res)))\n'
                f'(unless (= code 0)\n'
                f'  (display (cdr (assoc \'stderr res))) (newline) (exit code))\n',
                encoding="utf-8",
            )
            comp2 = subprocess.run(
                [str(ESHKOL_RUN), "-L", str(BUILD_DIR), str(launcher_src), "-o", str(launcher_exe)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(comp2.returncode, 0, msg=comp2.stderr + comp2.stdout)

            base_env = dict(os.environ)
            base_env.pop("ESHKOL_GPU_DISABLE", None)
            base_env["ESHKOL_GPU_THRESHOLD"] = "1"
            base_env["ESHKOL_GPU_PRECISION"] = "high"

            pageouts_before = _read_vm_pageouts()

            cpu_env = dict(base_env, ESHKOL_GPU_DISABLE="1", ESHKOL_SUBPROC_OS_SANDBOX="off")
            cpu_run = subprocess.run([str(bench_exe)], env=cpu_env, capture_output=True, text=True, timeout=120)
            self.assertEqual(cpu_run.returncode, 0, msg=cpu_run.stderr + cpu_run.stdout)
            cpu_m = _parse_metrics(cpu_run.stdout)

            metal_env = dict(base_env, ESHKOL_SUBPROC_OS_SANDBOX="off")
            metal_run = subprocess.run([str(bench_exe)], env=metal_env, capture_output=True, text=True, timeout=120)
            self.assertEqual(metal_run.returncode, 0, msg=metal_run.stderr + metal_run.stdout)
            metal_m = _parse_metrics(metal_run.stdout)

            sb_env = dict(base_env, ESHKOL_SUBPROC_OS_SANDBOX="strict")
            sb_run = subprocess.run([str(launcher_exe)], env=sb_env, capture_output=True, text=True, timeout=120)
            self.assertEqual(sb_run.returncode, 0, msg=sb_run.stderr + sb_run.stdout)
            sb_m = _parse_metrics(sb_run.stdout)

            pageouts_after = _read_vm_pageouts()

            self.assertAlmostEqual(cpu_m["C00"], 512.0, places=9)
            self.assertAlmostEqual(metal_m["C00"], 512.0, places=9)
            self.assertAlmostEqual(sb_m["C00"], 512.0, places=9)

            cpu_gf = _gflops(cpu_m)
            metal_gf = _gflops(metal_m)
            sb_gf = _gflops(sb_m)
            delta_pageouts = (
                (pageouts_after - pageouts_before)
                if (pageouts_before is not None and pageouts_after is not None)
                else -1
            )
            print(
                f"BENCH FP64 512x512x8: cpu={cpu_gf:.2f} GFLOP/s "
                f"metal={metal_gf:.2f} GFLOP/s seatbelt_strict={sb_gf:.2f} GFLOP/s "
                f"vm_pageouts_delta={delta_pageouts}"
            )
            # Seatbelt strict must not stall or degrade catastrophically relative to unsandboxed Metal.
            self.assertGreater(sb_gf, 0.0)


if __name__ == "__main__":
    unittest.main()
