#!/usr/bin/env python3
"""Portable AOT contract gate, with an opt-in independent OpenQASM/Qiskit oracle.

Default mode needs Python's standard library and a working native eshkol-run
toolchain. --qiskit-oracle additionally requires numpy, openqasm3[parser],
qiskit, and qiskit-qasm3-import; missing dependencies are an error, not a skip.
The oracle runs classical simulation only and never contacts a quantum service.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import os
from pathlib import Path
import re
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
# An independent list of the promised API, not extracted from the implementation.
GATES = {
    **{g: (0, 1) for g in ('id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx')},
    **{g: (1, 1) for g in ('p', 'rx', 'ry', 'rz')},
    **{g: (0, 2) for g in ('cx', 'cy', 'cz', 'ch', 'swap')},
    **{g: (1, 2) for g in ('cp', 'crx', 'cry', 'crz')},
    **{g: (0, 3) for g in ('ccx', 'cswap')},
}


def cases():
    result = []
    for gate, (parameters, wires) in GATES.items():
        # Asymmetric, nonconsecutive operands expose control/target reversal.
        operands = [2, 0, 1][:wires]
        args = (['-0.3712345678901234'] if parameters else []) + operands
        result.append((f'gate_{gate}', 3, 0, [(gate, *args)], None))
    result.extend([
        ('bell', 2, 0, [('h', 0), ('cx', 0, 1)], None),
        ('ghz', 3, 0, [('h', 0), ('cx', 0, 1), ('cx', 1, 2)], None),
        ('rotations', 3, 0, [('h', 0), ('rx', '1/3', 2), ('ry', '1e-7', 0),
                            ('rz', '-1.2345678901234567', 1), ('cp', '0.0', 2, 0)], None),
        ('bell_measure', 2, 2, [('h', 0), ('cx', 0, 1), ('barrier', 1, 0),
                                ('measure', 0, 0), ('measure', 1, 1)], {0: .5, 3: .5}),
        ('ghz_measure', 3, 3, [('h', 0), ('cx', 0, 1), ('cx', 1, 2),
                               ('measure', 0, 2), ('measure', 1, 0), ('measure', 2, 1)], {0: .5, 7: .5}),
        ('bit_order', 3, 3, [('x', 2), ('measure', 2, 0), ('measure', 0, 2),
                             ('measure', 1, 1)], {1: 1.0}),
        ('entangled_reset', 2, 2, [('h', 0), ('cx', 0, 1), ('reset', 0),
                                   ('measure', 0, 0), ('measure', 1, 1)], {0: .5, 2: .5}),
        ('overwrite_measurement', 1, 1, [('x', 0), ('measure', 0, 0), ('reset', 0),
                                         ('measure', 0, 0)], {0: 1.0}),
        ('empty', 1, 0, [], None),
    ])
    # Fixed seed makes failures reproducible. Every gate appears as the first
    # operation in four circuits; random compositions exercise its context.
    rng = random.Random(20261025)
    names = list(GATES)
    for index in range(100):
        nq = 3 + index % 2
        operations = []
        for step in range(8 + index % 17):
            gate = names[index % len(names)] if step == 0 else rng.choice(names)
            parameters, wires = GATES[gate]
            args = ([format(rng.uniform(-6.0, 6.0), '.17g')] if parameters else [])
            args += rng.sample(range(nq), wires)
            operations.append((gate, *args))
        result.append((f'generated_{index:03d}', nq, 0, operations, None))
    return result


def run(command, root, timeout):
    completed = subprocess.run([str(x) for x in command], cwd=root,
                               capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f'command exited {completed.returncode}: {command}\n'
                           f'{completed.stdout[-6000:]}\n{completed.stderr[-6000:]}')
    if re.search(r'\b(?:ERROR:|FAIL:)|LLVM module verification failed',
                 completed.stdout + completed.stderr):
        raise RuntimeError(f'error diagnostic despite exit 0: {completed.stdout}\n{completed.stderr}')
    return completed.stdout


def compile_run(compiler, source, output, root, timeout):
    run([compiler, '--no-stdlib', '-L', root / 'lib', source, '-o', output], root, timeout)
    if not output.is_file():
        raise RuntimeError(f'compiler did not create {output}')
    return run([output], root, timeout)


def verify_contract_output(output):
    match = re.fullmatch(r'circuit_export_test: (\d+) checks, 0 failed\s*', output)
    if not match or int(match.group(1)) < 50:
        raise AssertionError(f'missing or insufficient contract verdict: {output!r}')


def self_test():
    """Prove that a broken child or missing verdict cannot become a pass."""
    verify_contract_output('circuit_export_test: 50 checks, 0 failed\n')
    for output in ('', 'PASS\n', 'circuit_export_test: 50 checks, 1 failed\n',
                   'circuit_export_test: 49 checks, 0 failed\n'):
        try:
            verify_contract_output(output)
        except AssertionError:
            continue
        raise AssertionError(f'accepted invalid verdict: {output!r}')
    for program in ('raise SystemExit(7)', 'print("ERROR: injected")',
                    'import sys; print("FAIL: injected", file=sys.stderr)',
                    'print("LLVM module verification failed")'):
        try:
            run([sys.executable, '-c', program], ROOT, 10)
        except RuntimeError:
            continue
        raise AssertionError(f'accepted broken child: {program}')
    print('PASS: driver self-test: valid verdict and 8 deliberate failure cases')


def write_export_fixture(path, fixtures):
    code = ['(require core.quantum.circuit)']
    for name, nq, nb, operations, _ in fixtures:
        ops = ' '.join('(' + ' '.join(str(x) for x in op) + ')' for op in operations)
        code += [f'(display "@@CASE {name}@@\\n")',
                 f"(display (quantum-circuit->openqasm3 (make-quantum-circuit {nq} {nb} '({ops}))))",
                 '(display "@@END@@\\n")']
    path.write_text('\n'.join(code) + '\n', encoding='utf-8')


def parse_exports(output, fixtures):
    exported = {}
    pos = 0
    pattern = re.compile(r'@@CASE ([a-z0-9_]+)@@\n(.*?)@@END@@\n', re.S)
    for match in pattern.finditer(output):
        if output[pos:match.start()].strip():
            raise AssertionError('unexpected text outside exported QASM fixture')
        name, text = match.groups()
        if name in exported:
            raise AssertionError(f'duplicate export: {name}')
        exported[name] = text
        pos = match.end()
    if output[pos:].strip() or set(exported) != {x[0] for x in fixtures}:
        raise AssertionError('missing, extra or malformed exported fixtures')
    return exported


def load_oracle():
    try:
        import numpy as np
        import openqasm3
        from openqasm3 import parser
        import qiskit
        import qiskit_qasm3_import  # noqa: F401 - qasm3.loads optional dependency
        from qiskit import QuantumCircuit, qasm3
        from qiskit.quantum_info import DensityMatrix, Operator, Statevector
    except ImportError as exc:
        raise RuntimeError('--qiskit-oracle requires numpy, openqasm3[parser], '
                           'qiskit and qiskit-qasm3-import; dependencies missing: '
                           f'{exc}. No oracle check was performed.') from exc
    print(f'oracle versions: qiskit={qiskit.__version__}, openqasm3={openqasm3.__version__}, numpy={np.__version__}')
    return np, parser, QuantumCircuit, qasm3, DensityMatrix, Operator, Statevector


def reference_circuit(QuantumCircuit, nq, nb, operations):
    reference = QuantumCircuit(nq, nb)
    for name, *args in operations:
        if name in GATES and GATES[name][0]:
            args[0] = float(Fraction(args[0]))
        getattr(reference, name)(*args)
    return reference


def without_initial_resets(circuit, nq):
    initial = circuit.data[:nq]
    if len(initial) != nq or any(x.operation.name != 'reset' for x in initial):
        raise AssertionError('export must explicitly initialize every qubit with reset')
    if [circuit.find_bit(x.qubits[0]).index for x in initial] != list(range(nq)):
        raise AssertionError('initialization reset order/coverage mismatch')
    body = circuit.copy_empty_like()
    for instruction in circuit.data[nq:]:
        body.append(instruction.operation,
                    [circuit.find_bit(q).index for q in instruction.qubits],
                    [circuit.find_bit(c).index for c in instruction.clbits])
    return body


def exact_classical_probabilities(circuit, np, DensityMatrix):
    """Branch projective measurements exactly; no shot noise or Aer dependency.

    Density matrices remain unnormalised so traces carry branch probabilities.
    Reset is the sum of |0><0| and |0><1| Kraus actions, including entangled inputs.
    """
    projectors = [np.diag([1., 0.]), np.diag([0., 1.])]
    reset_kraus = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 1.], [0., 0.]])]
    branches = [(DensityMatrix.from_label('0' * circuit.num_qubits), 0)]
    for instruction in circuit.data:
        name = instruction.operation.name
        wires = [circuit.find_bit(q).index for q in instruction.qubits]
        if name == 'barrier':
            continue
        next_branches = []
        for state, bits in branches:
            if name == 'measure':
                bit = circuit.find_bit(instruction.clbits[0]).index
                for outcome, projector in enumerate(projectors):
                    projected = state.evolve(projector, qargs=wires)
                    if np.trace(projected.data).real > 1e-15:
                        next_branches.append((projected, (bits & ~(1 << bit)) | (outcome << bit)))
            elif name == 'reset':
                data = sum(state.evolve(k, qargs=wires).data for k in reset_kraus)
                next_branches.append((DensityMatrix(data), bits))
            else:
                next_branches.append((state.evolve(instruction.operation, qargs=wires), bits))
        branches = next_branches
    probabilities = {}
    for state, bits in branches:
        probabilities[bits] = probabilities.get(bits, 0.0) + float(np.trace(state.data).real)
    return probabilities


def verify_oracle(exported, fixtures, oracle):
    np, parser, QuantumCircuit, qasm3, DensityMatrix, Operator, Statevector = oracle
    for name, nq, nb, operations, expected_probs in fixtures:
        text = exported[name]
        ast = parser.parse(text)
        if ast.version != '3.0':
            raise AssertionError(f'{name}: wrong OpenQASM version')
        imported = qasm3.loads(text)
        if (imported.num_qubits, imported.num_clbits) != (nq, nb):
            raise AssertionError(f'{name}: register-size mismatch')
        body = without_initial_resets(imported, nq)
        expected = reference_circuit(QuantumCircuit, nq, nb, operations)
        # Inspect measurements/resets/barriers, including operand order. Gate
        # matrices below independently check gate identity, angles and wire order.
        def side_effects(circuit):
            return [(i.operation.name,
                     tuple(circuit.find_bit(q).index for q in i.qubits),
                     tuple(circuit.find_bit(c).index for c in i.clbits))
                    for i in circuit.data if i.operation.name in ('measure', 'reset', 'barrier')]
        if side_effects(body) != side_effects(expected):
            raise AssertionError(f'{name}: measurement/reset/barrier operand mismatch')
        if expected_probs is None:
            # Exact matrix comparison intentionally includes global phase;
            # state-only comparison would miss control errors on |000>.
            np.testing.assert_allclose(Operator(body).data, Operator(expected).data,
                                       atol=1e-12, rtol=1e-12, err_msg=name)
            actual_state = Statevector.from_instruction(body).data
            np.testing.assert_allclose(actual_state, Statevector.from_instruction(expected).data,
                                       atol=1e-12, rtol=1e-12, err_msg=name)
            np.testing.assert_allclose(np.abs(actual_state)**2,
                                       Statevector.from_instruction(expected).probabilities(),
                                       atol=1e-10, rtol=1e-10, err_msg=name)
            np.testing.assert_allclose(np.vdot(actual_state, actual_state), 1.0,
                                       atol=1e-12, rtol=1e-12, err_msg=name)
            if name in ('bell', 'ghz'):
                analytic = np.zeros(2**nq, dtype=complex)
                analytic[0] = analytic[-1] = 1 / np.sqrt(2)
                np.testing.assert_allclose(actual_state, analytic, atol=1e-12, rtol=1e-12)
        else:
            probabilities = exact_classical_probabilities(imported, np, DensityMatrix)
            if set(probabilities) != set(expected_probs):
                raise AssertionError(f'{name}: outcome mismatch: {probabilities}')
            for bits, probability in expected_probs.items():
                np.testing.assert_allclose(probabilities[bits], probability, atol=1e-12, rtol=1e-12)
        print(f'  oracle PASS: {name}')
    print(f'PASS: OpenQASM parser and Qiskit oracle: {len(GATES)} gate kinds, {len(fixtures)} circuits')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eshkol-run', type=Path)
    ap.add_argument('--repo-root', type=Path, default=ROOT)
    ap.add_argument('--work-dir', type=Path, help='retain generated fixtures/executables in a unique subdirectory')
    ap.add_argument('--timeout', type=int, default=120, help='per compiler/program timeout in seconds')
    ap.add_argument('--qiskit-oracle', action='store_true')
    ap.add_argument('--self-test', action='store_true', help='check failure handling without a compiler')
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.eshkol_run is None:
            raise ValueError('--eshkol-run is required unless --self-test is selected')
        if args.timeout < 1:
            raise ValueError('--timeout must be positive')
        compiler = args.eshkol_run.resolve(strict=True)
        root = args.repo_root.resolve(strict=True)
        oracle = load_oracle() if args.qiskit_oracle else None
        if args.work_dir:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix='circuit-export-', dir=args.work_dir))
            temporary = None
        else:
            temporary = tempfile.TemporaryDirectory(prefix='eshkol-circuit-export-')
            work = Path(temporary.name)
        try:
            suffix = '.exe' if os.name == 'nt' else ''
            output = compile_run(compiler, root / 'tests/quantum/circuit_export_test.esk',
                                 work / ('contract' + suffix), root, args.timeout)
            verify_contract_output(output)
            print(output.strip())
            example = compile_run(compiler, root / 'examples/quantum/bell_openqasm3.esk',
                                  work / ('bell_example' + suffix), root, args.timeout)
            expected_example = ('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
                                'bit[2] c;\nreset q;\nh q[0];\ncx q[0], q[1];\n'
                                'c[0] = measure q[0];\nc[1] = measure q[1];\n')
            if example != expected_example:
                raise AssertionError(f'Bell example export mismatch: {example!r}')
            print('PASS: Bell example compiled and exported the expected program')
            if oracle:
                fixtures = cases()
                source = work / 'oracle_exports.esk'
                write_export_fixture(source, fixtures)
                exported = parse_exports(compile_run(compiler, source, work / ('exports' + suffix),
                                                     root, args.timeout), fixtures)
                for name, text in exported.items():
                    (work / (name + '.qasm')).write_text(text, encoding='utf-8')
                verify_oracle(exported, fixtures, oracle)
            else:
                print('PASS: portable AOT contract gate (independent Qiskit oracle not requested)')
            if args.work_dir:
                print(f'artifacts: {work}')
        finally:
            if temporary:
                temporary.cleanup()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'FAIL: quantum circuit export: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
