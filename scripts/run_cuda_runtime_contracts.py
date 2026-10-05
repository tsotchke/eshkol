#!/usr/bin/env python3
"""Run device-independent CUDA contracts; absent registrations fail closed."""
import argparse
import json
import subprocess

REQUIRED = {'cublas_lazy_loader_test', 'cuda_runtime_link_args_test',
            'cuda_cpu_link_closure_test'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--config')
    args = parser.parse_args()
    common = ['ctest', '--test-dir', args.build_dir]
    if args.config:
        common += ['-C', args.config]

    build = ['cmake', '--build', args.build_dir, '--target',
             'cuda_runtime_contracts']
    if args.config:
        build += ['--config', args.config]
    built = subprocess.run(build, capture_output=True, text=True, timeout=1200)
    if built.returncode:
        raise RuntimeError(
            'Failed to build CUDA runtime contract targets: '
            f'exit {built.returncode}\n{built.stdout}{built.stderr}')

    manifest = subprocess.run(common + ['--show-only=json-v1'],
                              capture_output=True, text=True, timeout=60)
    if manifest.returncode:
        raise RuntimeError(manifest.stderr)
    tests = [test['name'] for test in json.loads(manifest.stdout)['tests']]
    invalid = sorted(name for name in REQUIRED if tests.count(name) != 1)
    if invalid:
        raise RuntimeError(
            f'Missing or ambiguous CUDA contract registrations: {invalid}')
    regex = '^(' + '|'.join(sorted(REQUIRED)) + ')$'
    result = subprocess.run(common + ['-R', regex, '--output-on-failure'], timeout=1200)
    if result.returncode:
        raise RuntimeError(f'CUDA runtime contracts failed: exit {result.returncode}')
    print('PASS: CUDA device-independent runtime contracts (3/3)')


if __name__ == '__main__':
    main()
