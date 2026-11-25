#!/usr/bin/env python3

from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import verify_gpu_backend


VALID_CUDA_EDGE = (
    "build CMakeFiles/eshkol-gpu-runtime-obj.dir/gpu.o: CUDA_COMPILER "
    "gpu_memory_cuda.cpp gpu_cuda_kernels.cu\n"
)


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
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            VALID_CUDA_EDGE,
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_accepts_ninja_multi_config_graph(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=C:/CUDA/bin/nvcc.exe\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            "include CMakeFiles/impl-Release.ninja\n",
        )
        graph_dir = build / "CMakeFiles"
        graph_dir.mkdir()
        (graph_dir / "impl-Release.ninja").write_text(
            VALID_CUDA_EDGE,
            encoding="utf-8",
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_accepts_cuda_13_portable_graph_without_sm72(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=13\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=75;80;86;89;90\n",
            VALID_CUDA_EDGE,
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_rejects_sm72_for_cuda_13(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=13\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            VALID_CUDA_EDGE,
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("does not support portable architecture 72" in item
                            for item in failures))

    # #606: the verifier checks against the supported set the configure
    # recorded from nvcc (or from the toolkit's documented range).
    def test_accepts_architectures_from_recorded_nvcc_set(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=13\n"
            "ESHKOL_CUDA_SUPPORTED_ARCHITECTURES:INTERNAL=75;80;86;89;90;100;120;121\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=75;80;86;89;90\n",
            VALID_CUDA_EDGE,
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_rejects_architecture_outside_recorded_nvcc_set(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=13\n"
            "ESHKOL_CUDA_SUPPORTED_ARCHITECTURES:INTERNAL=75;80;86;89;90\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            VALID_CUDA_EDGE,
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("does not support portable architecture 72" in item
                            for item in failures))

    def test_recorded_cuda_12_set_requires_sm72(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "ESHKOL_CUDA_SUPPORTED_ARCHITECTURES:INTERNAL=RANGE;50;999\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=75;80;86;89;90\n",
            VALID_CUDA_EDGE,
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("portable CUDA architecture 72 is missing" in item
                            for item in failures))

    def test_rejects_cpu_stub_mislabeled_as_cuda(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=OFF\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=NONE\n",
            "build CMakeFiles/eshkol-gpu-runtime-obj.dir/gpu.o: CXX_COMPILER gpu_memory_stub.cpp\n",
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("ESHKOL_GPU_BACKEND" in item for item in failures))
        self.assertTrue(any("CPU GPU stub" in item for item in failures))

    def test_rejects_cuda_graph_without_cuda_compiler(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n",
            VALID_CUDA_EDGE,
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertIn("CMAKE_CUDA_COMPILER is absent or unresolved", failures)

    def test_ignores_stub_in_unrelated_ninja_edge(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            VALID_CUDA_EDGE
            + "build CMakeFiles/optional-test.dir/gpu.o: CXX_COMPILER gpu_memory_stub.cpp\n",
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_ignores_stub_in_cmake_regeneration_dependency(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            VALID_CUDA_EDGE
            + "build build.ninja: RERUN_CMAKE | /repo/lib/backend/gpu/gpu_memory_stub.cpp\n",
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_rejects_stub_on_runtime_edge_when_unrelated_edge_has_cuda_sources(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            "build CMakeFiles/eshkol-gpu-runtime-obj.dir/gpu.o: CXX_COMPILER gpu_memory_stub.cpp\n"
            "build CMakeFiles/optional-test.dir/gpu.o: CUDA_COMPILER gpu_memory_cuda.cpp gpu_cuda_kernels.cu\n",
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("real CUDA source missing" in item for item in failures))
        self.assertTrue(any("CPU GPU stub" in item for item in failures))

    def test_scopes_visual_studio_sources_to_gpu_runtime_project(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=C:/CUDA/bin/nvcc.exe\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            "",
        )
        (build / "build.ninja").unlink()
        (build / "eshkol-gpu-runtime-obj.vcxproj").write_text(
            '<Project><ItemGroup><ClCompile Include="gpu_memory_cuda.cpp" />'
            '<CudaCompile Include="gpu_cuda_kernels.cu" /></ItemGroup></Project>',
            encoding="utf-8",
        )
        (build / "unrelated.vcxproj").write_text(
            '<Project><ItemGroup><ClCompile Include="gpu_memory_stub.cpp" />'
            '</ItemGroup></Project>', encoding="utf-8",
        )
        self.assertEqual(verify_gpu_backend.verify(build, "CUDA"), [])

    def test_unknown_graph_without_runtime_edges_fails_closed(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            "build CMakeFiles/unrelated.dir/gpu.o: CXX_COMPILER gpu_memory_stub.cpp\n",
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("source edges not found" in item for item in failures))

    def test_cuda_source_names_must_be_exact_graph_inputs(self) -> None:
        build = self.make_build(
            "ESHKOL_GPU_ENABLED:BOOL=ON\n"
            "ESHKOL_REQUIRE_GPU_BACKEND:BOOL=ON\n"
            "ESHKOL_GPU_BACKEND:INTERNAL=CUDA\n"
            "ESHKOL_HOST_CUDA_MAJOR:INTERNAL=12\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
            "CMAKE_CUDA_ARCHITECTURES:STRING=72;75;80;86;89;90\n",
            "build CMakeFiles/eshkol-gpu-runtime-obj.dir/gpu.o: CXX_COMPILER "
            "gpu_memory_cuda.cpp.old gpu_cuda_kernels.cu.tmp\n",
        )
        failures = verify_gpu_backend.verify(build, "CUDA")
        self.assertTrue(any("gpu_memory_cuda.cpp" in item for item in failures))
        self.assertTrue(any("gpu_cuda_kernels.cu" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
