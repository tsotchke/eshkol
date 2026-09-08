#!/usr/bin/env python3
"""Execute compiler gate treatments and record ICC-consumable, raw evidence.

Dose is the number of closed-enum cases removed from real compiler source in
an isolated source projection. Each observation runs the unmodified production
gate in a fresh process. Baselines are independent executions, not copied rows.
The optional runtime lane separately executes real compiler capability probes.
No mutation of the user's checkout or toolchain is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def execute(argv, cwd, env=None):
    start = time.time()
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=120)
        return {'argv': argv, 'exit_code':proc.returncode, 'stdout':proc.stdout,
                'stderr':proc.stderr, 'started_at':start, 'elapsed_s':time.time()-start}
    except subprocess.TimeoutExpired as exc:
        return {'argv':argv, 'exit_code':124, 'stdout':str(exc.stdout or ''),
                'stderr':str(exc.stderr or ''), 'started_at':start, 'elapsed_s':time.time()-start}


def validate(rows):
    errors = []
    by_dose = {dose:[r for r in rows if r['dose']==dose] for dose in (0,1,2)}
    for dose, cohort in by_dose.items():
        if len(cohort)<4:
            errors.append(f'dose {dose}: fewer than four actual executions')
        if len({r['execution_id'] for r in cohort}) != len(cohort):
            errors.append(f'dose {dose}: duplicate execution identity')
        if any(not r.get('execution') or r.get('parse_error') for r in cohort):
            errors.append(f'dose {dose}: missing or invalid raw gate execution')
    if errors: return {'status':'FAIL','errors':errors}
    scores = {d:[r['score'] for r in cohort] for d,cohort in by_dose.items()}
    noise = max(max(v)-min(v) for v in scores.values())
    margin = min(min(scores[1]),min(scores[2])) - max(scores[0])
    if any(r['execution']['exit_code'] != (0 if r['dose']==0 else 1) for r in rows):
        errors.append('baseline failed or injected fault survived')
    if max(scores[0]) != 0: errors.append('baseline has findings')
    if not (statistics.mean(scores[0]) < statistics.mean(scores[1]) < statistics.mean(scores[2])):
        errors.append('finding score did not respond monotonically to dose')
    if margin <= noise: errors.append('treatment response does not exceed measured noise')
    return {'status':'FAIL' if errors else 'PASS','errors':errors,
            'scores':scores,'measured_noise_range':noise,'treatment_margin':margin,
            'instrument_resolution':1,'decision_threshold':0.5,
            'noise_method':'maximum within-dose range of four fresh process observations; findings are integer counts'}


def corpus(root, out):
    import gate_exhaustive_dispatch as gate
    rows=[]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    with tempfile.TemporaryDirectory(prefix='eshkol-assurance-') as temp:
        projection=Path(temp)
        files={s['file'] for s in gate.SITES}|{s['enum_file'] for s in gate.SITES}|{
            gate.AD_REGISTRY,gate.BRIDGE_IMPL,gate.CMAKELISTS,'scripts/gate_exhaustive_dispatch.py'}
        for rel in files:
            target=projection/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/rel,target)
        target=projection/'lib/backend/tensor_backward.cpp'
        original=target.read_text()
        # Mutate actual cases in the actual first registered dispatcher.
        body=gate.function_body(original,'eshkol_tensor_backward_dispatch')
        import re
        cases=re.findall(r'case\s+(AD_NODE_\w+)\s*:',body)
        if len(cases)<2: raise RuntimeError('cannot find two mutation anchors')
        for repeat in range(4):
            for dose in (0,1,2):
                altered=body
                for member in cases[:dose]:
                    altered=re.sub(r'case\s+'+member+r'\s*:', '', altered, count=1)
                target.write_text(original.replace(body,altered,1))
                raw=execute([sys.executable,str(projection/'scripts/gate_exhaustive_dispatch.py'),
                             '--format','json','--no-trace'],projection)
                row={'schema':'eshkol.compiler_assurance.observation.v1','execution_id':str(uuid.uuid4()),
                     'git_sha':head,'gate_sha256':digest(root/'scripts/gate_exhaustive_dispatch.py'),
                     'source_sha256':digest(target),'dose':dose,'execution':raw,
                     'mutation':'remove-closed-enum-case','score':None}
                try:
                    report=json.loads(raw['stdout'])
                    if report.get('error'): raise ValueError(report['error'])
                    row['score']=sum(len(x['findings']) for x in report['results'])
                except (ValueError,KeyError,TypeError) as exc: row['parse_error']=str(exc)
                rows.append(row)
                # The full raw execution lives separately so constant process
                # metadata is not misrepresented as treatment-response metrics.
                write(out/'raw'/f'{repeat}-{dose}.json',row)
                write(out/'closed-enum'/f'{repeat}-{dose}.json',{
                    'schema':row['schema'],'execution_id':row['execution_id'],'dose':dose,
                    'gate_score':row['score'],'raw_receipt':str(out/'raw'/f'{repeat}-{dose}.json'),
                    'source_sha256':row['source_sha256'],'git_sha':head})
    report=validate(rows)
    write(out/'sensitivity.json',report)
    return report,rows


PROBES = {
    'captured-closure': ('(let ((x 40)) (display ((lambda (y) (+ x y)) 2)))','42'),
    'case-lambda': ('(display ((case-lambda (() 0) ((x) (+ x 1))) 41))','42'),
    'parameter-call': ('(define p (make-parameter 42)) (display (p))','42'),
    'apply-capture': ('(let ((x 40)) (display (apply (lambda (y) (+ x y)) (list 2))))','42'),
}


def runtime(root, binary, out):
    rows=[]
    if not binary.is_file(): return {'status':'FAIL','errors':[f'missing compiler {binary}']}
    before=digest(binary)
    with tempfile.TemporaryDirectory(prefix='eshkol-capability-') as temp:
        env=dict(os.environ,ESHKOL_JIT_CACHE_DIR=str(Path(temp)/'cache'))
        for name,(program,expected) in PROBES.items():
            raw=execute([str(binary),'-e',program],root,env)
            row={'name':name,'program':program,'expected':expected,'execution':raw,
                 'binary_sha256':before,'status':'PASS' if raw['exit_code']==0 and raw['stdout'].strip()==expected else 'FAIL'}
            rows.append(row)
    report={'status':'PASS' if all(r['status']=='PASS' for r in rows) and before==digest(binary) else 'FAIL',
            'binary':str(binary),'binary_sha256':before,'probes':rows}
    write(out/'runtime.json',report)
    return report


def self_test():
    rows=[{'dose':d,'score':d,'execution_id':str(uuid.uuid4()),'execution':{'exit_code':0 if d==0 else 1}}
          for d in (0,1,2) for _ in range(4)]
    if validate(rows)['status']!='PASS': raise AssertionError('valid controls rejected')
    import copy
    for defect in ('empty','constant','noise','duplicates','failed-baseline','parse-error'):
        bad=copy.deepcopy(rows)
        if defect=='empty': bad=[]
        elif defect=='constant':
            for row in bad: row['score']=0
        elif defect=='noise': bad[4]['score']=20
        elif defect=='duplicates': bad[1]['execution_id']=bad[0]['execution_id']
        elif defect=='failed-baseline': bad[0]['execution']['exit_code']=1
        else: bad[0]['parse_error']='malformed output'
        if validate(bad)['status']!='FAIL': raise AssertionError(f'{defect} survived')
    return {'status':'PASS','controls':6}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'.icc/evidence/compiler-assurance')
    p.add_argument('--binary',type=Path)
    p.add_argument('--runtime-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--verify',action='store_true',help='fail closed on missing, incomplete, or vacuous stored executions')
    a=p.parse_args()
    if a.self_test:
        print(json.dumps(self_test())); return 0
    if a.verify:
        rows=[json.loads(path.read_text()) for path in sorted((a.output/'raw').glob('*.json'))]
        report=validate(rows)
        print(json.dumps(report,indent=2))
        return report['status']!='PASS'
    # Replace only this runner's own evidence tree; never union stale passes.
    if a.output.exists():
        if not (a.output/'assurance-owner').is_file():
            p.error('output exists without assurance-owner; choose a fresh directory')
        shutil.rmtree(a.output)
    a.output.mkdir(parents=True)
    (a.output/'assurance-owner').write_text('run_compiler_assurance.py\n')
    report,rows=({'status':'PASS'},[]) if a.runtime_only else corpus(a.root,a.output)
    if a.runtime_only and not a.binary: p.error('--runtime-only requires --binary')
    if a.binary:
        report['runtime']=runtime(a.root,a.binary.resolve(),a.output)
        if report['runtime']['status']!='PASS': report['status']='FAIL'
    trace=a.root/('.icc/runtime-traces/compiler-capabilities.jsonl' if a.runtime_only else '.icc/runtime-traces/compiler-assurance.jsonl')
    trace.parent.mkdir(parents=True,exist_ok=True)
    events=[{'kind':'compiler_assurance','name':'compiler_gate_sensitivity','value':report['status'],
             'evidence':str(a.output),'timestamp':time.time()}] if not a.runtime_only else []
    events += [{'kind':'compiler_assurance_execution','name':'closed_enum_dispatch',
                'value':'PASS' if r['execution']['exit_code']==0 else 'FAIL',
                'execution_id':r['execution_id'],'dose':r['dose']} for r in rows]
    if a.binary: events.append({'kind':'compiler_assurance','name':'compiler_runtime_capabilities','value':report['runtime']['status']})
    text=''.join(json.dumps(e)+'\n' for e in events)
    trace.write_text(text)
    oracle_trace=a.root/'scripts/icc_traces'/trace.name
    oracle_trace.parent.mkdir(parents=True,exist_ok=True)
    oracle_trace.write_text(text)
    print(json.dumps(report,indent=2))
    return report['status']!='PASS'

if __name__=='__main__': sys.exit(main())
