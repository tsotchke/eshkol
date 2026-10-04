#!/usr/bin/env python3
"""Check CUDA fake vendor targets with a Windows Ninja Multi-Config graph."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def fake_target_block():
    cmake = (ROOT / 'CMakeLists.txt').read_text()
    start = cmake.index('    foreach(_fake_variant complete incomplete wrong_major')
    end = cmake.index('    endforeach()', start) + len('    endforeach()')
    return cmake[start:end]


class CublasFakeWindowsOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for tool in ('cmake', 'ninja', 'clang++'):
            if not shutil.which(tool):
                raise unittest.SkipTest(f'{tool} is required for Windows graph generation')

    def configure_fixture(self, root, *, legacy=False):
        block = fake_target_block()
        if legacy:
            # Recreate the former runtime-only isolation to prove the fixture
            # catches the duplicate import-library rule from PR 740.
            block = re.sub(
                r'\n                ARCHIVE_OUTPUT_DIRECTORY\n.*?\n'
                r'                COMPILE_PDB_OUTPUT_DIRECTORY\n.*?compile-pdb"',
                '', block, flags=re.DOTALL)

        fake_source = root / 'tests' / 'toolchain' / 'cublas_lazy_loader_fake.cpp'
        fake_source.parent.mkdir(parents=True)
        shutil.copy(ROOT / 'tests/toolchain/cublas_lazy_loader_fake.cpp', fake_source)
        (root / 'CMakeLists.txt').write_text('\n'.join([
            'cmake_minimum_required(VERSION 3.20)',
            'project(cublas_fake_windows_fixture LANGUAGES CXX)',
            'set(CUDAToolkit_VERSION_MAJOR 12)',
            'set(CUDAToolkit_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")',
            block,
            'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/fake-output-dirs-$<CONFIG>.txt"',
            '  CONTENT "$<TARGET_PROPERTY:cublas_fake_complete,RUNTIME_OUTPUT_DIRECTORY>\\n$<TARGET_PROPERTY:cublas_fake_complete,ARCHIVE_OUTPUT_DIRECTORY>\\n$<TARGET_PROPERTY:cublas_fake_complete,PDB_OUTPUT_DIRECTORY>\\n$<TARGET_PROPERTY:cublas_fake_complete,COMPILE_PDB_OUTPUT_DIRECTORY>\\n")',
            '']))
        build = root / 'build'
        result = subprocess.run([
            'cmake', '-S', str(root), '-B', str(build),
            '-G', 'Ninja Multi-Config', '-DCMAKE_SYSTEM_NAME=Windows',
            f'-DCMAKE_CXX_COMPILER={shutil.which("clang++")}',
            '-DCMAKE_CXX_COMPILER_WORKS=TRUE',
            '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY',
        ], capture_output=True, text=True, timeout=90)
        return build, result

    def test_windows_ninja_multiconfig_graph_has_variant_config_outputs(self):
        with tempfile.TemporaryDirectory(prefix='cublas-fake-windows-') as directory:
            root = Path(directory)
            build, configured = self.configure_fixture(root)
            self.assertEqual(configured.returncode, 0,
                             configured.stdout + configured.stderr)

            targets = subprocess.run([
                'ninja', '-C', str(build), '-f', 'build-Release.ninja',
                '-t', 'targets', 'all',
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(targets.returncode, 0, targets.stdout + targets.stderr)
            graph = '\n'.join(path.read_text(errors='replace')
                               for path in build.rglob('*.ninja'))
            output_dirs = (build / 'fake-output-dirs-Release.txt').read_text()
            self.assertEqual(output_dirs.splitlines(), [
                str(build / 'cublas-fake/complete/bin'),
                str(build / 'cublas-fake/complete/lib'),
                str(build / 'cublas-fake/complete/pdb'),
                str(build / 'cublas-fake/complete/compile-pdb'),
            ])
            for variant in ('complete', 'incomplete', 'wrong_major',
                            'property_failure', 'create_failure', 'stream_failure'):
                output = f'cublas-fake/{variant}'
                self.assertIn(output + '/bin/Release/cublas64_12.dll', graph)
                self.assertIn(output + '/lib/Release/cublas64_12.lib', graph)
                self.assertIn(output + '/pdb/Release/cublas64_12.pdb', graph)
                self.assertIn(output + '/compile-pdb/Release/', graph)

    def test_runtime_only_isolation_reproduces_import_library_collision(self):
        with tempfile.TemporaryDirectory(prefix='cublas-fake-windows-legacy-') as directory:
            build, configured = self.configure_fixture(Path(directory), legacy=True)
            self.assertEqual(configured.returncode, 0,
                             configured.stdout + configured.stderr)
            parsed = subprocess.run([
                'ninja', '-C', str(build), '-f', 'build-Release.ninja',
                '-t', 'targets', 'all',
            ], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(parsed.returncode, 0,
                                parsed.stdout + parsed.stderr)
            self.assertIn('multiple rules generate Release/cublas64_12.lib',
                          parsed.stdout + parsed.stderr)


if __name__ == '__main__':
    unittest.main()
