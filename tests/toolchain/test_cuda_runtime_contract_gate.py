#!/usr/bin/env python3
"""Exercise the CUDA contract gate against actual CTest registrations."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / 'scripts/run_cuda_runtime_contracts.py'
NAMES = ('cublas_lazy_loader_test', 'cuda_runtime_link_args_test',
         'cuda_cpu_link_closure_test')


class CudaContractGateTests(unittest.TestCase):
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
