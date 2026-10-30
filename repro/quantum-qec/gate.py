#!/usr/bin/env python3
"""Qualify Clifford/QEC consumers and independently kernel-check decoder tables. MIT."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
from gate_support import Engine,datum,read,disable,exact_output,sha
from oracle import CODES,prepare_valid,decoder_valid,lean_source,conjugate


def strings(xs):
    return '('+' '.join(json.dumps(x) for x in xs)+')'


def table_data(rows):
    return '('+' '.join('('+datum(key)+' '+json.dumps(s)+')' for key,s in rows)+')'


def large_generators(n):
    gs=[(1,'I'*q+'Z'+'I'*(n-q-1)) for q in range(n)]
    rng=random.Random(20261030)
    for i in range(4*n):
        q=rng.randrange(n)
        if i%3==0:
            t=rng.randrange(n-1);t+=t>=q;g=['CNOT',q,t]
        else:g=[('H' if i%3==1 else 'S'),q]
        gs=[conjugate(p,g,n) for p in gs]
    return [('-' if s<0 else '')+w for s,w in gs]


def lean_run(binary,source,accept):
    p=subprocess.run([str(binary),str(source)],capture_output=True,text=True,timeout=180)
    if accept and (p.returncode!=0 or 'sorry' in p.stdout+p.stderr or 'error:' in p.stdout+p.stderr):
        raise AssertionError('Lean rejected an admitted decoder:\n'+p.stdout+p.stderr)
    if not accept and (p.returncode==0 or 'error:' not in p.stdout+p.stderr):
        raise AssertionError('Lean negative control did not reject the false theorem')
    return hashlib.sha256((p.stdout+p.stderr).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--eshkol-bin',required=True)
    p.add_argument('--lean',type=Path,required=True);p.add_argument('--results',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    lean=a.lean.resolve(strict=True)
    version=subprocess.run([str(lean),'--version'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    e=Engine(a.eshkol_bin,root)
    e.capture(['lib/core/quantum/qec.esk','tests/stdlib/quantum_qec_test.esk']+
              ['repro/quantum-qec/'+n for n in ['gate.py','oracle.py','gate_support.py']])
    out=e.both(root/'tests/stdlib/quantum_qec_test.esk','unit')
    lines=out.splitlines()
    if len(lines)!=4 or lines[-1]!='PASS quantum_qec 46':raise AssertionError('Incomplete QEC receipt: '+out[-2000:])
    records=[];lean_receipts=[]
    for i,(line,(n,gs)) in enumerate(zip(lines[:3],CODES)):
        if not line.startswith('CODE '):raise AssertionError('Bad code receipt')
        nn,ss,gates,cert=read(line[5:])
        if (nn,ss)!=(n,gs) or not prepare_valid(n,gs,gates) or not decoder_valid(cert,n,gs):raise AssertionError('Independent QEC replay')
        false=copy.deepcopy(cert)
        false[3]=[[key,'I'*n] for key,_ in cert[3]]
        if decoder_valid(false,n,gs):raise AssertionError('No-op decoder control')
        missing=copy.deepcopy(cert);missing[3]=missing[3][1:]
        if decoder_valid(missing,n,gs):raise AssertionError('Omitted identity control')
        duplicate=copy.deepcopy(cert);duplicate[3].append(duplicate[3][0])
        if decoder_valid(duplicate,n,gs):raise AssertionError('Duplicate syndrome control')
        original=e.work/f'code{i}.lean';original.write_text(lean_source(cert,n,gs))
        negative=e.work/f'code{i}-false.lean';negative.write_text(lean_source(false,n,gs))
        good_hash=lean_run(lean,original,True);bad_hash=lean_run(lean,negative,False)
        lean_receipts.append({'code':i,'source_sha256':sha(original),'accepted_stdout_sha256':good_hash,
                              'false_decoder_rejected':True,'negative_stdout_sha256':bad_hash})
        records.append([n,gs,gates,cert])
    n=128;gs=large_generators(n)
    driver='(require core.quantum.qec)\n(define gs \'%s)\n(define c (stabilizer-prepare %d gs))\n'% (strings(gs),n)
    driver+='(if (not (verify-stabilizer-preparation 128 gs c)) (error "Large preparation failed"))\n'
    driver+='(display "LARGE ") (write (qcircuit-gates c)) (newline)\n'
    large=e.both(e.source('large',driver),'large')
    if not large.startswith('LARGE ') or len(large.splitlines())!=1:raise AssertionError('Incomplete large receipt')
    gates=read(large[6:])
    if not prepare_valid(n,gs,gates):raise AssertionError('Independent 128-qubit preparation failed')
    blocked=disable((root/'lib/core/quantum/qec.esk').read_text(),{'stabilizer-prepare','make-qec-code','qec-certificate'})
    replay=blocked+'\n(define (refuses? f) (guard (e (else #t)) (f) #f))\n'
    for call in ['(stabilizer-prepare 1 \'("Z"))','(make-qec-code 1 \'("Z"))',"(qec-certificate '#(unused))"]:
        replay+='(if (not (refuses? (lambda () '+call+'))) (error "Producer live"))\n'
    for nn,ss,g,c in records:
        replay+='(if (not (verify-stabilizer-preparation %d \'%s (make-qcircuit %d \'%s))) (error "Preparation replay"))\n'%(nn,strings(ss),nn,datum(g))
        cert="(vector 'qec-decoder-v1 %d '%s '%s)"%(nn,strings(ss),table_data(c[3]))
        replay+='(if (not (verify-qec-certificate %s %d \'%s)) (error "Decoder replay"))\n'%(cert,nn,strings(ss))
    replay+='(display "PASS qec_producer_free 9") (newline)\n'
    exact_output(e.both(e.source('replay',replay),'replay'),'PASS qec_producer_free 9\n')
    names=['lib/core/quantum/qec.esk','tests/stdlib/quantum_qec_test.esk',
           'repro/quantum-qec/gate.py','repro/quantum-qec/oracle.py','repro/quantum-qec/gate_support.py']
    result=e.receipt(names)
    result.update(unit_checks=46,quantum_circuit_consumers=3,largest_preparation_qubits=128,
                  large_preparation_gates=len(gates),independent_recovery_cases=78,
                  lean_version=version,lean_binary_sha256=sha(lean),lean=lean_receipts,
                  producer_free_checks=9,witnesses=records)
    if a.results:
        a.results.write_text(json.dumps(result,indent=2)+'\n')
        dest=a.results.parent/'lean';dest.mkdir(exist_ok=True)
        for i,(nn,ss,_,c) in enumerate(records):(dest/f'code{i}.lean').write_text(lean_source(c,nn,ss))
    print('PASS QEC bridge: native circuits, independent signed replay, 128 qubits, Lean positive/negative, producer-free JIT/AOT')


if __name__=='__main__':main()
