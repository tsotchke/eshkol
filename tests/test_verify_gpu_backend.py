#!/usr/bin/env python3

from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import verify_gpu_backend


class VerifyGpuBackendTests(unittest.TestCase):
    def make_build(self, cache: str, graph: str) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        build = Path(temp.name)
        (build / "CMakeCache.txt").write_text(cache, encoding="utf-8")
        (build / "build.ninja").write_text(graph, encoding="utf-8")
        return build

    def test_accepts_real_required_cuda_graph(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            "build gpu.o: CUDA_COMPILER gpu_memory_cuda.cpp gpu_cuda_kernels.cu\n",
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_accepts_ninja_multi_config_graph(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=C:/CUDA/bin/nvcc.exe\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            "include CMakeFiles/impl-Release.ninja\n",
        )
        graph_dir = build / "CMakeFiles"
        graph_dir.mkdir()
        (graph_dir / "impl-Release.ninja").write_text(
            "build gpu.obj: CUDA_COMPILER gpu_memory_cuda.cpp gpu_cuda_kernels.cu\n",
            encoding="utf-8",
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_rejects_cpu_stub_mislabeled_as_cuda(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=OFF\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=NONE\n",
            "build gpu.o: CXX_COMPILER gpu_memory_stub.cpp\n",
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("ESHKOL_GPU_BACKEND" in item for item in failures))
        self.assertTrue(any("CPU GPU stub" in item for item in failures))

    def test_rejects_cuda_graph_without_cuda_compiler(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n",
            "build gpu.o: CXX_COMPILER gpu_memory_cuda.cpp gpu_cuda_kernels.cu\n",
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertIn("CMAKE_CUDA_COMPILER is absent or unresolved", failures)


if __name__ == "__main__":
    unittest.main()
