"""Replay the context-bound rewrites against Lean and the feature branch’s exact circuit library."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from schedule import schedule_cost, exact_h_cnot_column
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASES = {'t', 's', 'z'}


def run(argv, cwd, *, stdin=None, env=None):
    result = subprocess.run(argv, cwd=cwd, input=stdin, text=True, capture_output=True,
                            timeout=120, env=env)
    if result.returncode:
        raise RuntimeError(f'{Path(argv[0]).name} failed\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}')
    return result.stdout


@dataclass(frozen=True)
class DirectedPhase:
    """Ordered statement U(original) = phase * U(replacement)."""
    width: int
    original: tuple
    replacement: tuple
    context: str
    coefficients: str


def checked_relation(n, lhs, rhs, context, env, executable, work):
    def form(tape):
        gates = ['(' + op + ' ' + ' '.join(map(str, wires)) + ')' for op, *wires in tape]
        return "'(" + ' '.join(gates) + ")"

    # The branch's scalar library computes old = ratio * new.
    source = work / 'exact.esk'
    source.write_text(f"""(require core.quantum.circuit)
(require core.monoidal.finvect)
(require core.monoidal.scalars)
(define old (make-qcircuit {n} {form(lhs)}))
(define new (make-qcircuit {n} {form(rhs)}))
(define field (qcircuit-exact-scalars old new))
(define ratio (linear-map-proportional (qcircuit-exact-unitary old field)
                                      (qcircuit-exact-unitary new field)))
(display "PHASE ")
(write (and ratio (let loop ((i 0))
         (cond ((= i 8) #f)
               ((scalars-equal? field ratio (scalars-phase field (/ i 4))) i)
               (else (loop (+ i 1)))))))
(newline)
""")
    out = run([executable, '-I', ROOT / 'lib', '-r', source], work,
              env={**env, 'ESHKOL_JIT_CACHE': '0'})
    phase, = re.findall(r'^PHASE (#f|[0-7])$', out, re.M)
    if phase == '#f' or (context == 'coherent' and phase != '0'):
        return None
    exponent = int(phase)
    coefficients = [0] * 4
    coefficients[exponent % 4] = 1 if exponent < 4 else -1
    return DirectedPhase(n, tuple(map(tuple, lhs)), tuple(map(tuple, rhs)), context,
                         ' '.join(map(str, coefficients)))


def phase_certificate(n, lhs, rhs, context, env, executable, work):
    result = checked_relation(n, lhs, rhs, context, env, executable, work)
    if result is None:
        raise ValueError(f'expected {context} equality')
    return result


def eshkol_proposal(executable, work, phases, n, tape, env):
    indexed = ' '.join(f'({i} {op.lower()} {" ".join(map(str, qs))})'
                       for i, (op, *qs) in enumerate(tape))
    calls = [f'(begin (display "PHASE ") (write (control-correction \'({p}))) (newline))'
             for p in phases]
    calls.append(f'(begin (display "ORDER ") (write (map car (q0-schedule {n} \'({indexed})))) (newline))')
    source = work / 'proposal.esk'
    source.write_text((HERE / 'optimizer.esk').read_text() + '\n'.join(calls) + '\n')
    output = run([executable, '-r', source], work, env={**env, 'ESHKOL_JIT_CACHE': '0'})
    corrections = [s.split() for s in re.findall(r'^PHASE \(([^()\n]*)\)$', output, re.M)]
    if len(corrections) != len(phases) or any(set(gs) - PHASES for gs in corrections):
        raise ValueError(f'malformed correction in {output!r}')
    order, = re.findall(r'^ORDER \(([\d ]+)\)$', output, re.M)
    indices = [int(i) for i in order.split()]
    return corrections, indices


def check_lean(executable, work, corrections, roots, env):
    # The consumer specifies the expected tapes/layout. Only emitted gates enter
    # the submitted claim. Proofs reduce in the kernel; no producer flag is read.
    cases = [
        ('compensated', 'coherent'),
        ('{ coherent with original := [.y,.x], replacement := [.z], correction := [.s] }',
         '{ coherent with original := [.y,.x], replacement := [.z] }'),
    ]
    (work / 'Context.lean').write_bytes((HERE / 'Context.lean').read_bytes())
    run([executable, '-DwarningAsError=true', '-o', 'Context.olean', 'Context.lean'], work, env=env)
    lean_env = {**env, 'LEAN_PATH': str(work)}
    header = 'import Context\nnamespace Q0\nnoncomputable section\n'
    source = header
    audited = ['contextual_sound', 'control_phase', 'phase_unit', 'hyh_phase', 'xy_phase', 'xy_reverse_phase', 'channelProof']
    for i, ((target, base), gates) in enumerate(zip(cases, corrections, strict=True)):
        seq = '[' + ','.join('.' + g for g in gates) + ']'
        name = f'emitted{i}'
        source += f'def claim{i} : Site := {{ ({base}) with correction := {seq} }}\n'
        source += f'theorem {name} : ValidRewrite claim{i} := by decide +kernel\n'
        source += f'example : bind ({target}) (witness claim{i} {name}) = .accept := by decide +kernel\n'
        audited.append(name)
    for exponent, gates in enumerate(roots):
        seq = '[' + ','.join('.' + g for g in gates) + ']'
        source += f'example : ({seq} : List PhaseGate).foldl (fun a g => mul (phase g) a) one = ' \
                  f'(List.replicate {exponent} (phase .t)).foldl mul one := by decide +kernel\n'
    source += ''.join(f'#print axioms {name}\n' for name in audited)
    verifier = work / 'Q0Replay.lean'
    verifier.write_text(source + 'end\nend Q0\n')
    audit = run([executable, '-DwarningAsError=true', verifier], work, env=lean_env)
    allowed = {'propext', 'Classical.choice', 'Quot.sound'}
    for name in audited:
        match = re.search(rf"^'Q0\.{name}' (?:depends on axioms: \[(.*?)\]|"
                          r"does not depend on any axioms)$", audit, re.M)
        if not match or (match[1] and set(match[1].split(', ')) - allowed):
            raise ValueError(f'unexpected axiom audit for {name}: {audit}')
    # Omit the Eshkol-emitted Z from the replacement. Require a type-correct
    # target whose validity actually reduces to false in the kernel.
    verifier.write_text(header + 'example : ValidRewrite coherent := by decide +kernel\nend\nend Q0\n')
    forged = subprocess.run([executable, verifier], cwd=work, env=lean_env,
                            capture_output=True, text=True, timeout=120)
    if forged.returncode != 1 or 'false' not in forged.stdout or 'decide' not in forged.stdout:
        raise ValueError(f'uncompensated proof control failed: {forged.stdout}')



def check_emitter(executable, work, env, evidence, reverse):
    # The consumer owns these sites. The producer supplies only an instruction
    # stream; it cannot select the observation or reorder source wire roles.
    yx = (('Y', 0), ('X', 0))
    z = (('Z', 0),)
    if (evidence, reverse) != (DirectedPhase(1, yx, z, 'channel', '0 0 1 0'),
                               DirectedPhase(1, z, yx, 'channel', '0 0 -1 0')):
        raise ValueError('directed source/replacement phase binding failed')
    producer = (HERE / 'optimizer.esk').read_text()
    emitter = '(lambda (gate) (list gate control))'
    mutation = "(lambda (gate) (list (if (eq? gate 's) 'sdg gate) control))"
    if producer.count(emitter) != 1:
        raise ValueError('emission mutation site changed')
    consumer = (HERE / 'emission-check.esk').read_text()
    # CY = S(target) · CX · S†(target) in operator order. The native
    # checker and circuit lowerings interpret the same emitted wire operands.
    def lower(tape):
        gates = []
        for op, *wires in tape:
            if op == 'cy':
                c, t = wires
                gates.extend([('Z', t), ('S', t), ('CNOT', c, t), ('S', t)])
            elif op == 'sdg':
                gates.extend([('Z', *wires), ('S', *wires)])
            else:
                gates.append(({'cx':'CNOT', 'cz':'CZ', 's':'S', 'z':'Z'}[op], *wires))
        return gates

    cy_cx = [('cy', 0, 1), ('cx', 0, 1)]
    cz = [('cz', 0, 1)]
    cases = [
        (producer, evidence, '(y x)', '(z)', cy_cx, cz + [('s', 0)], 'accept'),
        (producer, reverse, '(z)', '(y x)', cz, cy_cx + [('z', 0), ('s', 0)], 'accept'),
        (producer.replace(emitter, mutation), evidence, '(y x)', '(z)', cy_cx,
         cz + [('sdg', 0)], 'reject-operator'),
    ]
    for implementation, cert, old, new, source_tape, expected_tape, expected_verdict in cases:
        source = work / 'emission.esk'
        source.write_text(implementation + consumer + f"""
(define emitted (q0-emit '{new} '({cert.coefficients}) 0 1))
(display "TAPE ") (write emitted) (newline)
(display "VERDICT ") (write (q0-admit-emission '{old} '{new} 0 1 emitted)) (newline)
""")
        output = run([executable, '-r', source], work, env={**env, 'ESHKOL_JIT_CACHE':'0'})
        # Only admitted flat opcodes and literal wire numbers cross into the exact circuit checker.
        row, = re.findall(r'^TAPE (.+)$', output, re.M)
        gate_pattern = r'\((cx|cy|cz|z|s|sdg)((?: [01]){1,2})\)'
        if not re.fullmatch(r'\(' + gate_pattern + '(?: ' + gate_pattern + r')*\)', row):
            raise ValueError(f'malformed Eshkol emission {row}')
        actual = [(op, *map(int, qs.split())) for op, qs in re.findall(gate_pattern, row)]
        if actual != expected_tape or re.findall(r'^VERDICT (.+)$', output, re.M) != [expected_verdict]:
            raise ValueError(f'unexpected native emission or verdict: {output}')
        exact = checked_relation(2, lower(source_tape), lower(actual), 'coherent', env, executable, work)
        if expected_verdict == 'accept':
            if exact is None or exact.coefficients != '1 0 0 0':
                raise ValueError('emitted operator failed independent exact equivalence')
        elif exact is not None:
            raise ValueError('independent exact checker accepted the S-dagger mutation')
        print(f'{cert.coefficients} emits {row}: {expected_verdict} (feature-branch exact circuit check agrees)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eshkol', default='eshkol-run')
    parser.add_argument('--lean', default='lean')
    args = parser.parse_args()
    original = json.loads((HERE / 'steane-tape.json').read_text())
    n = 7
    if any(g[0] not in ('H', 'CNOT') for g in original):
        raise ValueError('unexpected instruction in the Steane fragment')

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        env = os.environ.copy()
        env.setdefault('XDG_CACHE_HOME', str(work / 'cache'))
        negative = phase_certificate(1, [('H',0),('Y',0),('H',0)], [('Y',0)], 'channel', env, args.eshkol, work)
        # Temporal order Y then X gives X·Y = iZ; detects an inverted phase.
        imaginary = phase_certificate(1, [('Y',0),('X',0)], [('Z',0)], 'channel', env, args.eshkol, work)
        reverse = phase_certificate(1, [('Z',0)], [('Y',0),('X',0)], 'channel', env, args.eshkol, work)
        roots = [' '.join(str(sign if j == i else 0) for j in range(4))
                 for sign in (1, -1) for i in range(4)]
        sequences, order = eshkol_proposal(args.eshkol, work,
                                            [negative.coefficients, imaginary.coefficients, *roots],
                                            n, original, env)
        baseline = schedule_cost(n, original, list(range(len(original))))
        optimized = schedule_cost(n, original, order)
        if (n, baseline['depth'], optimized['depth']) != (7, 9, 8):
            raise ValueError('Steane width or schedule depth changed')
        # Reverse an actual noncommuting H/CNOT pair. Gate counts and the
        # instruction permutation still pass, so rejection must be for ordering.
        bad = list(range(len(original)))
        h, cnot = next((i, j) for i, a in enumerate(original) for j, b in enumerate(original)
                       if i < j and a[0] == 'H' and b[0] == 'CNOT' and a[1] in b[1:])
        bad[h], bad[cnot] = bad[cnot], bad[h]
        try:
            schedule_cost(n, original, bad)
        except ValueError as error:
            if 'noncommuting dependency' not in str(error):
                raise
        else:
            raise ValueError('Steane dependency mutation passed')
        proposed = [original[i] for i in order]
        check_lean(args.lean, work, sequences[:2], sequences[2:], env)
        # Both tapes have the same Hadamard count and hence a common
        # denominator. Compare all signed integer numerators, in wire order.
        if any(exact_h_cnot_column(n, original, i) != exact_h_cnot_column(n, proposed, i)
               for i in range(1 << n)):
            raise ValueError('Steane whole-operator equality failed')
        print('Six Q0 verdicts verified. Corrections:', negative.coefficients, sequences[0],
              imaginary.coefficients, sequences[1])
        print('Steane costs:', baseline, '->', optimized)
        print('Steane exact operator equality holds on all 128 basis columns.')
        check_emitter(args.eshkol, work, env, imaginary, reverse)


if __name__ == '__main__':
    main()
