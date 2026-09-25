#!/usr/bin/env python3
"""Materialize the ordinary Eshkol generator's WGSL and sealed manifest."""
import hashlib
import json
import math
import pathlib
import re
import struct
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "generate.esk"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(argv, env=None):
    p = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    return p.stdout


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def main():
    import os
    native = run([str(ROOT / 'build/eshkol-run'), '-r', str(SOURCE), '-L', str(ROOT / 'build')],
                 {**os.environ, 'ESHKOL_LIB_DIR': str(ROOT / 'build')})
    vm = run([str(ROOT / 'build/eshkol-vm-standalone-test'), str(SOURCE)],
             {**os.environ, 'ESHKOL_VM_NO_DISASM': '1'})
    if native != vm or 'AD_MODEL_PASS\n' not in native:
        raise SystemExit('native/VM generator or AD model check differs')
    def field(name):
        match = re.search(r'^// ' + name + r': ([^\n]+)$', native, re.M)
        if not match:
            raise SystemExit(f'missing {name}')
        return [float(x) for x in match.group(1).split(',')]
    cs = field('coefficients-ascending')
    d = field('first-ascending')
    s = field('second-ascending')
    ad = field('ad-grid-max-errors')
    if len(cs) != 11 or len(d) != 10 or len(s) != 9:
        raise SystemExit('unexpected Taylor derivative orders')
    expected = [0.0, 0.0] + [2.0 * 3.0 ** (k-1) / math.factorial(k) for k in range(2, 11)]
    coefficient_error = max(abs(x-y) for x, y in zip(cs, expected))
    if coefficient_error > 1e-12 or any(abs(d[k]-(k+1)*cs[k+1]) > 1e-12 for k in range(10)) or any(abs(s[k]-(k+1)*d[k+1]) > 1e-12 for k in range(9)):
        raise SystemExit('Taylor coefficients or derivative shifts disagree with analytic law')
    wgsl = native.split('// wgsl-begin\n', 1)[1].split('// wgsl-end\n', 1)[0]
    (HERE / 'evaluator.wgsl').write_text(wgsl)
    r = 0.2
    # Analytic Lagrange bounds for W=(a/b)(exp(b*t)-1-b*t), derivatives 0..2.
    truncation = [2.0 * 3.0 ** 10 * math.exp(3 * r) * r ** (11-j)
                  / math.factorial(11-j) for j in range(3)]
    storage = [sum(abs(f32(c) - c) * r ** k for k, c in enumerate(seq))
               for seq in (cs, d, s)]
    compiler = ROOT / 'build/eshkol-run'
    model = {'potential': '(a/b)*(exp(b*e)-1-b*e)', 'a_Pa': 2.0, 'b_dimensionless': 3.0,
             'independent': 'e', 'e_units': 'dimensionless strain', 'held_fixed': ['a', 'b'],
             'reference_e': 0.0, 'normalization': 't=(e-reference_e)/1',
             'radius_t': r, 'order': 10, 'coefficient_order': 'ascending powers of t',
             'value_units': 'Pa', 'first_units': 'Pa/strain',
             'second_units': 'Pa/strain^2', 'coefficients': cs,
             'first_coefficients': d, 'second_coefficients': s}
    manifest = {
        'schema_version': 1, 'artifact_set': 'fung-strain-energy-v1', 'generation': 7,
        'sealed': True, 'replacement': 'Replace the entire WGSL/manifest pair as one generation; never mutate coefficients in place.',
        'model': model, 'model_sha256': sha(json.dumps(model, sort_keys=True, separators=(',', ':')).encode()),
        'source_sha256': sha(SOURCE.read_bytes()), 'wgsl_sha256': sha(wgsl.encode()),
        'compiler': {'git_commit': run(['git', 'rev-parse', 'HEAD']).strip(),
                     'native_binary_sha256': sha(compiler.read_bytes()),
                     'vm_generator_byte_identical': True},
        'abi': {'precision': 'IEEE 754 f32 storage and WGSL arithmetic', 'endianness': 'little',
                'workgroup_size': 64, 'max_count': 4096,
                'dispatch': 'ceil(count/64), at most 64 workgroups; bounds checked against count and both runtime arrays',
                'bindings': [
                    {'group': 0, 'binding': 0, 'name': 'params', 'access': 'read', 'bytes': 16,
                     'fields': {'count': 'u32@0', 'generation': 'u32@4', 'pad0': 'u32@8', 'pad1': 'u32@12'}},
                    {'group': 0, 'binding': 1, 'name': 'samples', 'access': 'read', 'stride_bytes': 4,
                     'fields': {'strain': 'f32@0'}},
                    {'group': 0, 'binding': 2, 'name': 'results', 'access': 'read_write', 'stride_bytes': 32,
                     'fields': {'value': 'f32@0', 'first': 'f32@4', 'second': 'f32@8',
                                'in_envelope': 'u32@12', 'status': 'u32@16',
                                'generation': 'u32@20', 'pad0': 'u32@24', 'pad1': 'u32@28'}}],
                'status': {'0': 'finite result inside envelope', '1': 'nonfinite input or outside envelope',
                           '2': 'nonfinite evaluation'},
                'outside_envelope': 'zero numeric outputs; status 1; in_envelope 0'},
        'errors': {'analytic_coefficient_max_absolute': coefficient_error,
                   'ad_grid_41_max_absolute': dict(zip(('value', 'first', 'second'), ad)),
                   'analytic_taylor_remainder_upper_bound': dict(zip(('value', 'first', 'second'), truncation)),
                   'f32_coefficient_conversion_upper_bound': dict(zip(('value', 'first', 'second'), storage)),
                   'gpu_rounding': 'Demo reports GPU difference from separate-operation f32 Horner; implementation may contract multiply-add.',
                   'acceptance_absolute_tolerance': {'value': 2e-6, 'first': 1e-5, 'second': 5e-5},
                   'domain': '|e|<=0.2, finite e; a and b held fixed'},
        'browser_evidence': 'demo/evidence.json'
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('artifact PASS: native=VM; AD grid max', ad)


if __name__ == '__main__':
    main()
