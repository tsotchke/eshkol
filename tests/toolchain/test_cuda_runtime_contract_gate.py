#!/usr/bin/env python3
"""Exercise the CUDA contract gate against actual CTest registrations."""
from pathlib import Path
import importlib.util
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

    def exercise(self, names=NAMES, failing=None):
        with tempfile.TemporaryDirectory(prefix='cuda-contract-gate-') as directory:
            root = Path(directory)
            source = ['cmake_minimum_required(VERSION 3.20)',
                      'project(cuda_gate_fixture NONE)', 'enable_testing()']
            for name in names:
                exit_code = 7 if name == failing else 0
                source.append(f'add_test(NAME {name} COMMAND "{sys.executable}" '
                              f'-c "import sys;sys.exit({exit_code})")')
            (root / 'CMakeLists.txt').write_text('\n'.join(source) + '\n')
            configured = subprocess.run(['cmake', '-S', str(root), '-B', str(root/'build')],
                                        capture_output=True, text=True, timeout=60)
            self.assertEqual(configured.returncode, 0, configured.stderr)
            return subprocess.run([sys.executable, str(GATE), '--build-dir', str(root/'build')],
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


if __name__ == '__main__':
    unittest.main()
