#!/usr/bin/env python3
"""Exercise the CUDA contract gate against actual CTest registrations."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / 'scripts/run_cuda_runtime_contracts.py'
NAMES = ('cublas_lazy_loader_test', 'cuda_runtime_link_args_test',
         'cuda_cpu_link_closure_test')
SPEC = importlib.util.spec_from_file_location('cuda_imports',
    ROOT / 'tests/toolchain/cuda_cpu_link_closure_test.py')
IMPORTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTS)


class CudaContractGateTests(unittest.TestCase):
    def test_late_cuda_resolution_enables_ctest_with_tests_off(self):
        cmake_source = (ROOT / 'CMakeLists.txt').read_text()
        selection = cmake_source.index('set(ESHKOL_GPU_BACKEND "CUDA" CACHE INTERNAL')
        enable_start = cmake_source.index(
            'if(ESHKOL_GPU_BACKEND STREQUAL "CUDA")\n'
            '    # The backend is resolved below project setup.', selection)
        enable_end = cmake_source.index('\nmacro(eshkol_append_host_runtime_link_arg ',
                                       enable_start)
        enable_block = cmake_source[enable_start:enable_end]
        self.assertLess(selection, enable_start)
        self.assertIn('enable_testing()', enable_block)

        vm_condition = ('if((ESHKOL_BUILD_TESTS OR ESHKOL_GPU_BACKEND STREQUAL '
                        '"CUDA") AND NOT WIN32)')
        vm_start = cmake_source.index(vm_condition)
        vm_end = cmake_source.index('\nendif()', vm_start)
        vm_target_block = cmake_source[vm_start:vm_end]
        self.assertIn('add_executable(eshkol-vm-standalone-test', vm_target_block)
        self.assertIn('target_link_libraries(eshkol-vm-standalone-test PRIVATE',
                      vm_target_block)
        self.assertEqual(cmake_source.count(
            'add_executable(eshkol-vm-standalone-test'), 1)
        self.assertIn('add_dependencies(cuda_runtime_contracts eshkol-vm-standalone-test)',
                      cmake_source)

        with tempfile.TemporaryDirectory(prefix='cuda-ctest-enable-') as directory:
            root = Path(directory)
            (root / 'CMakeLists.txt').write_text('\n'.join([
                'cmake_minimum_required(VERSION 3.20)',
                'project(cuda_ctest_enable_fixture NONE)',
                'set(ESHKOL_BUILD_TESTS OFF)',
                'set(ESHKOL_GPU_BACKEND NONE CACHE INTERNAL "" FORCE)',
                'if(ESHKOL_BUILD_TESTS)',
                '  enable_testing()',
                'endif()',
                'set(ESHKOL_GPU_BACKEND CUDA)',
                enable_block,
                f'add_test(NAME {NAMES[0]} COMMAND "{sys.executable}" -c "pass")',
                ''])
            )
            build_dir = root / 'build'
            configured = subprocess.run(
                ['cmake', '-S', str(root), '-B', str(build_dir)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(configured.returncode, 0, configured.stderr)
            listed = subprocess.run(
                ['ctest', '--test-dir', str(build_dir), '--show-only=json-v1'],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(listed.returncode, 0, listed.stderr)
            names = [test['name'] for test in json.loads(listed.stdout)['tests']]
            self.assertIn(NAMES[0], names)

    def test_elf_parser_ignores_checkout_name_and_rejects_vendor_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'image'
            binary.write_bytes(b'\x7fELF')
            listing = 'File: /work/cublas_repair/image\nNeededLibraries [\n libcudart.so.12\n libc.so.6\n]\n'
            with patch.object(IMPORTS, 'run', return_value=listing):
                self.assertEqual(IMPORTS.eager_cublas(IMPORTS.imports(binary, 'reader')), [])
            with patch.object(IMPORTS, 'run', return_value=listing.replace('libc.so.6', 'libcublas.so.12')):
                self.assertEqual(IMPORTS.eager_cublas(IMPORTS.imports(binary, 'reader')), ['libcublas.so.12'])

    def test_pe_parser_rejects_cublaslt_dll_and_missing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'image.exe'
            binary.write_bytes(b'MZ')
            with patch.object(IMPORTS, 'run', return_value='Import {\n Name: cublasLt64_12.dll\n}\n'):
                self.assertEqual(IMPORTS.eager_cublas(IMPORTS.imports(binary, 'reader')), ['cublasLt64_12.dll'])
            with patch.object(IMPORTS, 'run', return_value=''):
                with self.assertRaisesRegex(RuntimeError, 'Missing PE import records'):
                    IMPORTS.imports(binary, 'reader')

    def exercise(self, names=NAMES, failing=None, multi_config=False):
        with tempfile.TemporaryDirectory(prefix='cuda-contract-gate-') as directory:
            root = Path(directory)
            source = ['cmake_minimum_required(VERSION 3.20)',
                      'project(cuda_gate_fixture NONE)', 'enable_testing()']
            for name in names:
                exit_code = 7 if name == failing else 0
                command = (f"sys.exit({exit_code} if '$<CONFIG>' == 'Release' else 9)"
                           if multi_config else f'sys.exit({exit_code})')
                source.append(f'add_test(NAME {name} COMMAND "{sys.executable}" '
                              f'-c "import sys;{command}")')
            source.append('add_custom_target(cuda_runtime_contracts)')
            (root / 'CMakeLists.txt').write_text('\n'.join(source) + '\n')
            build_dir = root / 'build'
            configure = ['cmake', '-S', str(root), '-B', str(build_dir)]
            config = None
            if multi_config:
                configure += ['-G', 'Ninja Multi-Config']
                config = 'Release'
            configured = subprocess.run(configure,
                                        capture_output=True, text=True, timeout=60)
            self.assertEqual(configured.returncode, 0, configured.stderr)
            command = [sys.executable, str(GATE), '--build-dir', str(build_dir)]
            if config:
                command += ['--config', config]
            return subprocess.run(command,
                                  capture_output=True, text=True, timeout=60)

    def test_complete_contract_set_passes(self):
        result = self.exercise()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('runtime contracts (3/3)', result.stdout)

    def test_deleted_loader_registration_fails_before_tests(self):
        result = self.exercise(NAMES[1:])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing or ambiguous', result.stderr)
        self.assertIn('cublas_lazy_loader_test', result.stderr)

    def test_failed_loader_cannot_be_hidden_by_other_passes(self):
        result = self.exercise(failing=NAMES[0])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CUDA runtime contracts failed', result.stderr)

    def test_multi_config_release_registrations_pass(self):
        result = self.exercise(multi_config=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('runtime contracts (3/3)', result.stdout)

    def test_multi_config_missing_registration_fails_closed(self):
        result = self.exercise(NAMES[1:], multi_config=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing or ambiguous', result.stderr)
        self.assertIn('cublas_lazy_loader_test', result.stderr)


if __name__ == '__main__':
    unittest.main()
