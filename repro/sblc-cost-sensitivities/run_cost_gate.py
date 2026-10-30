#!/usr/bin/env python3
"""Check derivatives from an independent rational-function Taylor expansion."""
import argparse, hashlib, json, os, subprocess, tempfile
from pathlib import Path
from fractions import Fraction as F
from math import comb
import importlib.util
BASE=Path(__file__).resolve().parent

def source_tools(root):
    spec=importlib.util.spec_from_file_location('sblc_resource_oracle',Path(root)/'scripts/run_sblc_resource_gate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def add(a, b):
    out = list(a)+[F(0)]*max(0, len(b)-len(a))
    for i, x in enumerate(b):
        out[i] += x
    while len(out)>1 and not out[-1]:
        out.pop()
    return out


def scale(a, c):
    return [x*c for x in a]


def mul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def taylor(polynomial, center):
    return [sum((c*comb(i, order)*center**(i-order) for i, c in enumerate(polynomial) if i>=order), F(0))
            for order in range(3)]


def div(a,b):
    out=[]
    for n in range(3):
        out.append((a[n]-sum(b[i]*out[n-i] for i in range(1,n+1)))/b[0])
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--eshkol-bin',required=True);p.add_argument('--source-root',type=Path,default=BASE.parents[1]);p.add_argument('--results',type=Path)
    a=p.parse_args(); root=a.source_root.resolve(); env=dict(os.environ,ESHKOL_PATH=str(root/'lib'),ESHKOL_JIT_CACHE='0',ESHKOL_JIT_COMPILE_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(cmd):
        r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=240)
        assert r.returncode==0,r.stderr[-1000:]
        return r.stdout
    jit=run([a.eshkol_bin,'-r',str(BASE/'cost_test.esk')]); lines=jit.splitlines()
    assert len(lines)==12 and lines[-1]=='PASS cost 20',jit[-2000:]
    tools=source_tools(root)
    for line,(k,theta) in zip(lines[:-1],[(k,t) for k in range(1,5) for t in [F(1,4),F(1,2),F(3,4)] if (k,t)!=(2,F(1,2))]):
        assert line.startswith('COST ')
        got_k,got_t,r=tools.read(line[5:]);assert (got_k,got_t)==(k,theta)
        polynomial={1:[0,0,1],2:[0,1],3:[0,2,-1],4:[1]}[k]
        pj=taylor(polynomial,theta); one=[F(1),F(0),F(0)]
        p0=pj[0]; sub=p0>F(1,2)
        q=one if sub else div(pj,[x-y for x,y in zip(one,pj)])
        num=[x-y for x,y in zip([4,0,0],pj)] if sub else [x+y for x,y in zip([3,0,0],pj)]
        den=[2*p0-1,2*pj[1],2*pj[2]] if sub else [1-2*p0,-2*pj[1],-2*pj[2]]
        conditional=div(num,den)
        restricted=mul(q,conditional)[:3]
        assert r[5]==('almost-sure' if sub else 'defective') and r[6]==q and r[7]==restricted
        assert r[8]==(conditional if sub else 'infinite') and r[9]==conditional
        expected=[[F(0)]*3 for _ in range(25)]
        other=add([1],scale(polynomial,-1))
        for n in range(4):
            power=[F(1)]
            for _ in range(n+1):power=mul(power,polynomial)
            for _ in range(n):power=mul(power,other)
            catalan=comb(2*n,n)//(n+1)
            expected[7*n+3]=taylor(scale(power,catalan),theta)
        assert r[10]==expected
    with tempfile.TemporaryDirectory(prefix='cost-native-') as d:
        binary=str(Path(d)/'cost');run([a.eshkol_bin,str(BASE/'cost_test.esk'),'-O','0','-o',binary]);assert run([binary])==jit
    result={'schema_version':1,'engines':['jit','aot-O0'],'checks_per_engine':20,'contexts':11,
      'cost_degree':24,'operational_oracle':'existing sblc-step with exact Bernoulli branch weights',
      'derivative_oracle':'independent rational-function Taylor and Catalan coefficient expansion',
      'refusals':['critical point','interval crossing critical boundary','zero conditioning mass','unadmitted source'],
      'compiler_sha256':hashlib.sha256(Path(a.eshkol_bin).read_bytes()).hexdigest(),
      'files_sha256':{str((BASE/n).resolve().relative_to(root)):hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ['cost.esk','cost_test.esk','run_cost_gate.py']}}
    if a.results:a.results.write_text(json.dumps(result,indent=2)+'\n')
    print('COST PASS: 11 source/coin contexts, value/first/second coefficients, critical refusals, JIT/AOT')
if __name__=='__main__':main()
