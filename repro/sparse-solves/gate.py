#!/usr/bin/env python3
"""Native sparse solve qualification with independent exact derivative replay. MIT."""
import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import random
from gate_support import Engine,datum,read,exact_output,disable
from oracle import validate,enclosure,matrix,mv


def vector(xs):return '(vector '+' '.join(datum(x) for x in xs)+')'


def fixtures():
    rng=random.Random(20261030)
    for case in range(32):
        n=2+case%8;a={}
        for i in range(n):
            for j in range(i):
                if rng.randrange(3)==0:a[i,j]=a[j,i]=rng.choice([-1,1])
        for i in range(n):a[i,i]=(sum(abs(a.get((i,j),0)) for j in range(n) if i!=j)+2)*(-1 if case>=16 and i%2 else 1)
        ts=[[i,j,v] for (i,j),v in sorted(a.items())]
        b=[rng.randrange(-4,5) for _ in range(n)]
        seed=[rng.randrange(-3,4) for _ in range(n)]
        db=[Fraction(rng.randrange(-2,3),8) for _ in range(n)]
        dv=[Fraction(rng.randrange(-2,3),16) for _ in ts]
        yield case,n,ts,b,seed,dv,db,('cg' if case<16 else 'minres')


def main():
    p=argparse.ArgumentParser();p.add_argument('--eshkol-bin',required=True);p.add_argument('--results',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];e=Engine(a.eshkol_bin,root)
    e.capture(['lib/core/ml/sparse_solve.esk','tests/stdlib/sparse_solve_test.esk']+
              ['repro/sparse-solves/'+n for n in ['gate.py','oracle.py','gate_support.py']])
    unit=e.both(root/'tests/stdlib/sparse_solve_test.esk','unit');lines=unit.splitlines()
    if len(lines)!=2 or not lines[0].startswith('SMALL ') or lines[1]!='PASS sparse_solve 25':raise AssertionError('Incomplete sparse unit receipt')
    small=[-1,3]+read(lines[0][6:]);validate(small)
    fs=list(fixtures());src='(require core.ml.sparse_solve)\n'
    src+='''(define (emit id n ts b seed dv db method)
 (let* ((a (sparse-from-triplets n n ts))
        (j (sparse-solve-jvp method a b dv db 1e-10 64))
        (v (sparse-solve-vjp method a b seed 1e-10 64)))
  (if (not (and (verify-sparse-jvp j a b dv db) (verify-sparse-vjp v a b seed))) (error "Sensitivity refused"))
  (display "CASE ") (write (list id n ts (vector->list b) (vector->list seed) (vector->list dv) (vector->list db) j v)) (newline)))
'''
    for i,n,ts,b,seed,dv,db,method in fs:
        src+='(emit %d %d \'%s %s %s %s %s \'%s)\n'%(i,n,datum(ts),vector(b),vector(seed),vector(dv),vector(db),method)
    out=e.both(e.source('cases',src),'cases');rows=out.splitlines()
    if len(rows)!=len(fs):raise AssertionError('Incomplete derivative corpus')
    records=[small]
    for line,fixture in zip(rows,fs):
        if not line.startswith('CASE '):raise AssertionError('Case prefix')
        r=read(line[5:])
        if r[:7]!=list(fixture[:7]):raise AssertionError('Wrong source model')
        validate(r);records.append(r)
    large='''(require core.ml.sparse_solve)
(define (tab n f) (let ((v (make-vector n 0))) (let loop ((i 0)) (if (< i n) (begin (vector-set! v i (f i)) (loop (+ i 1))))) v))
(define (run n indefinite?)
 (define (diag i) (if (and indefinite? (= (modulo i 2) 1)) -5 5))
 (define (x i) (let ((r (quotient i 32)) (c (modulo i 32))) (/ (* (+ r 1) (- 32 r) (+ c 1) (- 32 c)) 65536)))
 (define (l i) (/ (- (modulo i 7) 3) 4))
 (define (neighbors i)
   (append (if (>= i 32) (list (- i 32)) '()) (if (> (modulo i 32) 0) (list (- i 1)) '())
           (if (< (modulo i 32) 31) (list (+ i 1)) '()) (if (< (+ i 32) n) (list (+ i 32)) '())))
 (define (rhs f i) (- (* (diag i) (f i)) (fold-left + 0 (map f (neighbors i)))))
 (let* ((ts (let rows ((i (- n 1)) (out '())) (if (< i 0) out
     (rows (- i 1) (append (map (lambda (j) (list i j -1)) (filter (lambda (j) (< j i)) (neighbors i)))
          (list (list i i (diag i))) (map (lambda (j) (list i j -1)) (filter (lambda (j) (> j i)) (neighbors i))) out)))))
        (a (sparse-from-triplets n n ts)) (b (tab n (lambda (i) (rhs x i)))) (seed (tab n (lambda (i) (rhs l i))))
        (dv (list->vector (map (lambda (t) (if (= (car t) (cadr t)) 1/8 1/16)) ts)))
        (db (tab n (lambda (i) (/ (- (modulo i 5) 2) 16)))) (method (if indefinite? 'minres 'cg))
        (j (sparse-solve-jvp method a b dv db 1e-10 128)) (v (sparse-solve-vjp method a b seed 1e-10 128)))
   (if (not (and (verify-sparse-jvp j a b dv db) (verify-sparse-vjp v a b seed))) (error "Large sensitivity refused"))
   (display "LARGE ") (write (list (if indefinite? 1001 1000) n ts (vector->list b) (vector->list seed) (vector->list dv) (vector->list db) j v)) (newline)))
(run 1024 #f)
(run 1024 #t)
'''
    output=e.both(e.source('large',large),'large');ls=output.splitlines()
    if len(ls)!=2:raise AssertionError('Incomplete large sparse receipt')
    large_records=[]
    for line,ident in zip(ls,[1000,1001]):
        if not line.startswith('LARGE '):raise AssertionError('Large prefix')
        r=read(line[6:])
        if r[:2]!=[ident,1024] or len(r[2])!=4992:raise AssertionError('Large source identity')
        expected={}
        for i in range(1024):
            expected[i,i]=(-5 if ident==1001 and i%2 else 5)
            row,col=divmod(i,32)
            for j in ([i-32] if row else [])+([i-1] if col else [])+([i+1] if col<31 else [])+([i+32] if row<31 else []):expected[i,j]=-1
        if r[2]!=[[i,j,v] for (i,j),v in sorted(expected.items())]:raise AssertionError('Wrong screened-Poisson stencil')
        validate(r,dense=False)
        want=[Fraction((i//32+1)*(32-i//32)*(i%32+1)*(32-i%32),65536) for i in range(1024)]
        if r[3]!=mv(1024,expected,want) or r[4]!=mv(1024,expected,[Fraction((i%7)-3,4) for i in range(1024)]):raise AssertionError('Wrong manufactured forcing/seed')
        if r[5]!=[Fraction(1,8) if i==j else Fraction(1,16) for i,j in sorted(expected)] or r[6]!=[Fraction((i%5)-2,16) for i in range(1024)]:raise AssertionError('Wrong PDE parameter direction')
        if r[8][4][1]!=want:raise AssertionError('Large primal oracle')
        records.append(r);large_records.append(r)
    # False narrow enclosures and false cotangents fail exact independent replay.
    false=copy.deepcopy(small);false[7][7]=0
    try:validate(false)
    except AssertionError:pass
    else:raise AssertionError('Forged JVP radius accepted')
    false=copy.deepcopy(small);false[8][6][0][0]+=1
    try:validate(false)
    except AssertionError:pass
    else:raise AssertionError('Forged VJP accepted')
    blocked=disable((root/'lib/core/ml/sparse_solve.esk').read_text(),
                    {'sparse-cg','sparse-minres','sparse-enclose','sparse-solve-jvp','sparse-solve-vjp','ss-solve'})
    replay=blocked+'\n(define (refuses? f) (guard (e (else #t)) (f) #f))\n'
    ts,b,seed,dv,db,j,v=small[2:]
    replay+='(define a (sparse-from-triplets 3 3 \'%s))\n(define b %s)\n'%(datum(ts),vector(b))
    for call in ['(sparse-cg a b 1e-10 32)','(sparse-minres a b 1e-10 32)',
                 '(sparse-enclose a b b)',"(sparse-solve-jvp 'cg a b b b 1e-10 32)",
                 "(sparse-solve-vjp 'cg a b b 1e-10 32)","(ss-solve 'cg a b 1e-10 32)"]:
        replay+='(if (not (refuses? (lambda () '+call+'))) (error "Producer live"))\n'
    # Convert received nested vectors to explicit (vector ...) constructors.
    def obj(x):
        if isinstance(x,list):return '(vector '+' '.join(obj(y) for y in x)+')'
        if isinstance(x,str):return "'"+x
        return datum(x)
    replay+='(if (not (verify-sparse-jvp %s a b %s %s)) (error "JVP replay"))\n'%(obj(j),vector(dv),vector(db))
    replay+='(if (not (verify-sparse-vjp %s a b %s)) (error "VJP replay"))\n'%(obj(v),vector(seed))
    replay+='(display "PASS sparse_producer_free 8") (newline)\n'
    exact_output(e.both(e.source('replay',replay),'replay'),'PASS sparse_producer_free 8\n')
    names=['lib/core/ml/sparse_solve.esk','tests/stdlib/sparse_solve_test.esk',
           'repro/sparse-solves/gate.py','repro/sparse-solves/oracle.py','repro/sparse-solves/gate_support.py']
    result=e.receipt(names)
    result.update(unit_checks=25,independent_dense_derivative_cases=33,large_sparse_dimension=1024,
                  large_nnz=4992,large_consumers=['32x32 screened Poisson/cg','32x32 signed diagonal/minres'],producer_free_checks=8,
                  negative_derivative_controls=2,small_witness=small,large_witnesses=large_records,
                  large_metrics=[{'method':r[8][2][1],'iterations':r[8][2][5], 'residual_2_norm':r[8][2][4],
                                  'primal_radius':r[8][4][3],'jvp_radius':r[7][7]} for r in large_records],
                  large_receipt_sha256=__import__('hashlib').sha256(output.encode()).hexdigest())
    if a.results:a.results.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('PASS sparse bridge: CG/MINRES, exact source and sensitivity enclosures, 1024-state consumers, producer-free JIT/AOT')


if __name__=='__main__':main()
