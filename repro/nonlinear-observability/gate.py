#!/usr/bin/env python3
"""Exact nonlinear-observability bridge gate. MIT."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from fractions import Fraction
from gate_support import Engine, datum, read, exact_output, disable, sha
from oracle import check, fiber, power, expression, symmetry


def fixtures():
    vs = ['x0','x1','x2','x3']
    f = [['expt', x, 2] for x in vs]
    h = [['+',vs[i],vs[(i+1)%4]] for i in range(4)] + [['+',vs[0],['*',2,vs[1]]]]
    task = ['-', ['+',vs[0],vs[2]], ['+',vs[1],vs[3]]]
    qs = [5,2,-1,0,-4]+[0]*5
    yield vs,f,h,1,task,1,qs
    yield ['x'],[0],[['expt','x',3],['expt','x',2]],0,'x',3,[4,['*',-3,['+','x','observation$x']]]
    for n in (4,40,128):
        vs = ['x'+str(i) for i in range(n)]
        f = [['expt',x,2] for x in vs]
        h = [['+',vs[i],vs[(i+1)%n]] for i in range(n)]+[['+',vs[0],['*',2,vs[1]]]]
        for j in (0,n//2,n-1):
            # x_j-y_j = (-1)^j(2*g_0-g_active)+sum alternating edge equations.
            qs = [0]*(n+1)
            qs[0] = (-1)**j*2
            qs[n] = -(-1)**j
            for k in range(j):
                qs[k] += (-1)**(j-1-k)
            yield vs,f,h,0,vs[j],1,qs


def wire_expr(e):
    if isinstance(e, (int,Fraction)):
        q = Fraction(e)
        return {'const':{'t':'q','n':str(q.numerator),'d':str(q.denominator)}}
    if isinstance(e,str):
        return {'var':e}
    if e[0] == 'expt':
        return {'op':'expt','args':[wire_expr(e[1])],'k':{'t':'z','v':str(e[2])}}
    return {'op':e[0],'args':[wire_expr(x) for x in e[1:]]}


def standard_claim(record, checker):
    vs,f,h,d,t,k,qs = record
    av, gs, target = fiber(vs,f,h,d,t)
    lhs = ['expt',expression(target,av),k]
    rhs = ['+']+[['*',q,expression(g,av)] for q,g in zip(qs,gs)]
    c = {'kind':'polynomial_identity','label':'observation_task',
         'inputs':{'lhs':wire_expr(lhs),'rhs':wire_expr(rhs)},
         'output':{'t':'bool','v':True},'witness':{'vars':av,'difference':[]},'premises':[]}
    c['id'] = checker.claim_id(c)
    c['hash'] = checker.claim_hash(c)
    return c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--eshkol-bin',required=True)
    p.add_argument('--certcheck',type=Path,required=True)
    p.add_argument('--results',type=Path)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    e=Engine(a.eshkol_bin,root)
    e.capture(['lib/core/nonlinear_observability.esk','tests/stdlib/nonlinear_observability_test.esk']+
              ['repro/nonlinear-observability/'+n for n in ['gate.py','oracle.py','gate_support.py']])
    unit=e.both(root/'tests/stdlib/nonlinear_observability_test.esk','unit')
    exact_output(unit,'PASS nonlinear_observability 15\n')
    fs=list(fixtures())
    vs,f,h,d,t,_,_=fs[0]
    symmetry_models=[(vs,f,h[:4],t,['a','b'],['a','b','a','b'],[['expt','a',2],['expt','b',2]],['b','a'],True),
                     (vs,f,h,t,['a','b'],['a','b','a','b'],[['expt','a',2],['expt','b',2]],['b','a'],False),
                     (vs,[f[0],f[1],0,f[3]],h[:4],t,['a','b'],['a','b','a','b'],[['expt','a',2],['expt','b',2]],['b','a'],False)]
    if any(symmetry(*m[:-1])!=m[-1] for m in symmetry_models):raise AssertionError('Independent all-order symmetry control')
    driver='(require core.nonlinear_observability)\n'
    for i, row in enumerate(fs):
        vs,f,h,d,t,k,qs=row
        call='(certify-observation-task '+ ' '.join("'"+datum(x) for x in (vs,f,h,d,t,k,qs))+')'
        driver+=f'(define cert{i} {call})\n(if (not cert{i}) (error "Fixture refused"))\n'
        driver+=f'(display "PROOF ") (write (list {i} cert{i})) (newline)\n'
    out=e.both(e.source('cases',driver),'cases')
    lines=out.splitlines()
    if len(lines)!=len(fs): raise AssertionError('Incomplete proof corpus')
    spec=importlib.util.spec_from_file_location('upstream_certcheck',a.certcheck.resolve(strict=True))
    cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)
    claims=[]
    for i,(line,row) in enumerate(zip(lines,fs)):
        if not line.startswith('PROOF '):raise AssertionError('Bad proof prefix')
        ix,c=read(line[6:])
        if ix!=i or c!=['observation-task-v1',row[5],row[6]]: raise AssertionError('Wrong proof body')
        if not check(*row): raise AssertionError('Independent source replay failed')
        claim=standard_claim(row,cc)
        if cc.check_claim(claim)[0]!='ACCEPTED':raise AssertionError(cc.check_claim(claim))
        forged=copy.deepcopy(claim)
        forged['inputs']['rhs']=wire_expr(1)
        forged['id']=cc.claim_id(forged);forged['hash']=cc.claim_hash(forged)
        if cc.check_claim(forged)[0]!='REJECTED':raise AssertionError('Resealed false identity accepted')
        claims.append(claim)
    blocked=disable((root/'lib/core/nonlinear_observability.esk').read_text(),
                    {'certify-observation-task','observation-task-hint'})
    replay=blocked+'\n(define (refuses? f) (guard (e (else #t)) (f) #f))\n'
    for name in ('certify-observation-task','observation-task-hint'):
        call=f'({name} \'(x) \'(0) \'(x) 0 \'x'+(" 1 '(1)" if name.startswith('certify') else '')+')'
        replay+='(if (not (refuses? (lambda () '+call+'))) (error "Producer live"))\n'
    for row in (fs[0],fs[-1]):
        vs,f,h,d,t,k,qs=row
        replay+='(if (not (verify-observation-task (vector \'observation-task-v1 '+datum(k)+" '"+datum(qs)+') '+ ' '.join("'"+datum(x) for x in (vs,f,h,d,t))+')) (error "Producer-free replay failed"))\n'
    replay+='(display "PASS observation_producer_free 4") (newline)\n'
    exact_output(e.both(e.source('replay',replay),'replay'),'PASS observation_producer_free 4\n')
    names=[
           'lib/core/nonlinear_observability.esk','tests/stdlib/nonlinear_observability_test.esk',
           'repro/nonlinear-observability/gate.py','repro/nonlinear-observability/oracle.py',
           'repro/nonlinear-observability/gate_support.py']
    result=e.receipt(names)
    result.update(unit_checks=15,independent_task_proofs=len(fs),largest_state_dimension=128,
                  producer_free_checks=4,resealed_false_identity_controls=len(fs),
                  certcheck_sha256=sha(a.certcheck),claims=claims,source_models=fs,symmetry_models=symmetry_models)
    if a.results:
        a.results.write_text(json.dumps(result,indent=2)+'\n')
        (a.results.parent/'certificates.json').write_text(json.dumps(
            {'schema':'eshkol.certificate.v1','program':'nonlinear-observability','claims':claims},indent=2)+'\n')
    print('PASS nonlinear bridge: source replay, certificate checker, 128-state consumer, producer-free JIT/AOT')


if __name__=='__main__':main()
