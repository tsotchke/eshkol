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

    def configure_fixture(self, root, *, legacy=False, compiler_target):
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
        metadata = [
            'set(_fake_target_metadata "")',
            'foreach(_variant complete incomplete wrong_major property_failure create_failure stream_failure)',
            '  string(APPEND _fake_target_metadata',
            '    "$<TARGET_FILE:cublas_fake_${_variant}>|$<TARGET_LINKER_FILE:cublas_fake_${_variant}>|"',
            '    "$<TARGET_PROPERTY:cublas_fake_${_variant},PDB_OUTPUT_DIRECTORY>|"',
            '    "$<TARGET_PROPERTY:cublas_fake_${_variant},COMPILE_PDB_OUTPUT_DIRECTORY>\\n")',
            'endforeach()',
            'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/fake-targets-$<CONFIG>.txt"',
            '  CONTENT "${_fake_target_metadata}")',
        ]
        (root / 'CMakeLists.txt').write_text('\n'.join([
            'cmake_minimum_required(VERSION 3.20)',
            'project(cublas_fake_windows_fixture LANGUAGES CXX)',
            'set(CUDAToolkit_VERSION_MAJOR 12)',
            'set(CUDAToolkit_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")',
            block,
            *metadata,
            'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/fake-output-dirs-$<CONFIG>.txt"',
            '  CONTENT "$<TARGET_PROPERTY:cublas_fake_complete,RUNTIME_OUTPUT_DIRECTORY>\\n$<TARGET_PROPERTY:cublas_fake_complete,ARCHIVE_OUTPUT_DIRECTORY>\\n$<TARGET_PROPERTY:cublas_fake_complete,PDB_OUTPUT_DIRECTORY>\\n$<TARGET_PROPERTY:cublas_fake_complete,COMPILE_PDB_OUTPUT_DIRECTORY>\\n")',
            '']))
        build = root / 'build'
        result = subprocess.run([
            'cmake', '-S', str(root), '-B', str(build),
            '-G', 'Ninja Multi-Config', '-DCMAKE_SYSTEM_NAME=Windows',
            f'-DCMAKE_CXX_COMPILER={shutil.which("clang++")}',
            f'-DCMAKE_CXX_COMPILER_TARGET={compiler_target}',
            '-DCMAKE_CXX_COMPILER_WORKS=TRUE',
            '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY',
        ], capture_output=True, text=True, timeout=90)
        return build, result

    def test_windows_ninja_multiconfig_graph_has_variant_config_outputs(self):
        for compiler_target in ('x86_64-pc-windows-msvc',
                                'x86_64-w64-windows-gnu'):
            with self.subTest(compiler_target=compiler_target), \
                    tempfile.TemporaryDirectory(prefix='cublas-fake-windows-') as directory:
                root = Path(directory)
                build, configured = self.configure_fixture(
                    root, compiler_target=compiler_target)
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

                variants = ('complete', 'incomplete', 'wrong_major',
                            'property_failure', 'create_failure', 'stream_failure')
                configs = ('Debug', 'Release', 'RelWithDebInfo')
                runtime_files = set()
                linker_files = set()
                for config in configs:
                    rows = (build / f'fake-targets-{config}.txt').read_text().splitlines()
                    self.assertEqual(len(rows), len(variants), rows)
                    for variant, row in zip(variants, rows):
                        runtime, linker, pdb_dir, compile_pdb_dir = row.split('|')
                        expected_base = build / 'cublas-fake' / variant
                        runtime_path = Path(runtime)
                        linker_path = Path(linker)
                        self.assertEqual(runtime_path.parent,
                                         expected_base / 'bin' / config)
                        self.assertEqual(runtime_path.name, 'cublas64_12.dll')
                        self.assertEqual(linker_path.parent,
                                         expected_base / 'lib' / config)
                        self.assertIn(linker_path.suffix, ('.lib', '.a'), linker)
                        self.assertEqual(Path(pdb_dir), expected_base / 'pdb')
                        self.assertEqual(Path(compile_pdb_dir),
                                         expected_base / 'compile-pdb')
                        runtime_files.add(runtime_path)
                        linker_files.add(linker_path)
                self.assertEqual(len(runtime_files), len(variants) * len(configs))
                self.assertEqual(len(linker_files), len(variants) * len(configs))
                for variant in variants:
                    self.assertIn(f'cublas-fake/{variant}/pdb/Release/', graph)
                    self.assertIn(f'cublas-fake/{variant}/compile-pdb/Release/', graph)

    def test_runtime_only_isolation_reproduces_import_library_collision(self):
        for compiler_target in ('x86_64-pc-windows-msvc',
                                'x86_64-w64-windows-gnu'):
            with self.subTest(compiler_target=compiler_target), \
                    tempfile.TemporaryDirectory(prefix='cublas-fake-windows-legacy-') as directory:
                build, configured = self.configure_fixture(
                    Path(directory), legacy=True, compiler_target=compiler_target)
                self.assertEqual(configured.returncode, 0,
                                 configured.stdout + configured.stderr)
                linker = (build / 'fake-targets-Release.txt').read_text().splitlines()[0].split('|')[1]
                generated_name = Path(linker).relative_to(build).as_posix()
                parsed = subprocess.run([
                    'ninja', '-C', str(build), '-f', 'build-Release.ninja',
                    '-t', 'targets', 'all',
                ], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(parsed.returncode, 0,
                                    parsed.stdout + parsed.stderr)
                self.assertIn(f'multiple rules generate {generated_name}',
                              parsed.stdout + parsed.stderr)


if __name__ == '__main__':
    unittest.main()
