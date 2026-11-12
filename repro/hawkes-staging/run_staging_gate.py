#!/usr/bin/env python3
"""Admit the source, replay residuals without the producer, compile emitted kernels."""
import argparse,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
from fractions import Fraction as F
import importlib.util
BASE=Path(__file__).resolve().parent
ZERO=(0,)*5

def source_tools(root):
    spec=importlib.util.spec_from_file_location('closure_source_oracle',Path(root)/'scripts/run_observable_closure_gate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def plus(a,b):
    c=dict(a)
    for k,v in b.items():c[k]=c.get(k,F(0))+v
    return {k:v for k,v in c.items() if v}
def times(a,b):
    c={}
    for k,v in a.items():
        for l,w in b.items():
            key=tuple(x+y for x,y in zip(k,l));c[key]=c.get(key,F(0))+v*w
    return {k:v for k,v in c.items() if v}
def polynomial(e,env):
    if type(e) is int:return {} if e==0 else {ZERO:F(e)}
    tag=e[0]
    if tag=='quote':assert type(e[1]) is int;return polynomial(e[1],env)
    if tag=='var':return env[e[1]]
    if tag=='let':
        env=dict(env)
        for name,value in e[1]:env[name]=polynomial(value,env)
        return polynomial(e[2],env)
    assert tag=='prim' and len(e)==4 and e[1] in ['+','-','*']
    a,b=polynomial(e[2],env),polynomial(e[3],env)
    return times(a,b) if e[1]=='*' else plus(a,{k:(-v if e[1]=='-' else v) for k,v in b.items()})
def datum(x):
    if isinstance(x,list):
        vector=x and x[0] in ['generator-closure-v1','observable-closure-v1','mix-result-v1'] if x and isinstance(x[0],str) else False
        return ('#(' if vector else '(')+' '.join(datum(y) for y in x)+')'
    return str(x)
def evaluate(p,point):return sum(v*__import__('math').prod(x**n for x,n in zip(point,k)) for k,v in p.items())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--eshkol-bin',required=True);ap.add_argument('--source-root',type=Path,default=BASE.parents[1]);ap.add_argument('--results',type=Path);a=ap.parse_args();root=a.source_root.resolve()
    env=dict(os.environ,ESHKOL_PATH=str(root/'lib'),ESHKOL_JIT_CACHE='0',ESHKOL_JIT_COMPILE_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    timings={}
    def run(cmd,label):
        t=time.monotonic();r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=240);timings[label]=time.monotonic()-t
        assert r.returncode==0,r.stdout[-1200:]+r.stderr[-1200:]
        return r.stdout
    receipt=run([a.eshkol_bin,'-r',str(BASE/'staging_test.esk')],'construction_wall_seconds')
    lines=receipt.splitlines();assert len(lines)==6 and lines[-1]=='PASS staging 8'
    tools=source_tools(root);plan=tools.read(lines[4][6:]);source,proof=plan[1:];tools.check_certificate(source,proof)
    artifacts=[tools.read(line[6:]) for line in lines[:4]]
    envpoly={}
    for i,name in enumerate(['wn','u0','u1','u2','u3']):
        power=list(ZERO);power[i]=1;envpoly[name]={tuple(power):F(1)}
    polynomials=[];definitions=[];replay=['(require core.observable_closure)','(require core.compiler.mix.verify)',
        '(if (not (verify-generator-closure \''+datum(source)+" '"+datum(proof)+')) (error "Model replay"))']
    for i,artifact in enumerate(artifacts):
        src,result,emission=artifact[1:4];program,statics,den,dynamic,row,wd=src[1:]
        assert row==i and wd==4 and den>0 and emission[1]=='normal'
        envstatic=dict(envpoly)
        for name,dot,value in statics:assert dot=='.' and type(value) is int;envstatic[name]=polynomial(value,{})
        expected={}
        for power,action in zip(proof[1],proof[4][4]):
            for j,c in enumerate(action[1:][i]):
                key=[0]*5;key[0]=power[0];key[j+1]=1
                if c:expected[tuple(key)]=F(c)*den/F(4)**power[0]
        expected={k:v for k,v in expected.items() if v}
        assert polynomial(program[2],envstatic)==expected
        assert polynomial(result[2][2],envpoly)==expected
        polynomials.append((expected,den));definitions+=emission[2][1:]
        result[6]='detached-producer-history'
        replay+=['(if (not (eq? (mix-verification-status (mix-verify-specialization \''+datum(program)+" '"+datum(statics)+" '"+datum(result)+
            " (mix-verification-limits 1000000 128 512 300000))) 'equivalent)) (error \"Residual replay\"))"]
        bad=list(result);bad[2]=['program',[],987654321]
        replay+=['(if (not (eq? (mix-verification-status (mix-verify-specialization \''+datum(program)+" '"+datum(statics)+" '"+datum(bad)+
            " (mix-verification-limits 1000000 128 512 300000))) 'different)) (error \"Changed residual accepted\"))"]
    replay+=['(display "PASS producer-free staging 9") (newline)']
    points=[[n%7-2,(n*3)%9-4,(n*5)%11-5,(n*7)%13-6,(n*11)%17-8] for n in range(32)]
    wanted=[];test=[]
    for i,(p,den) in enumerate(polynomials):
        abi=artifacts[i][3][3]
        for point in points:
            bindings=dict(zip(['wn','u0','u1','u2','u3'],point));args=' '.join(str(bindings[k]) for k in abi)
            test+=['(write (hawkes_row_'+str(i)+' '+args+')) (newline)'];wanted.append(evaluate(p,point))
    with tempfile.TemporaryDirectory(prefix='staged-model-') as d:
        d=Path(d);rp=d/'replay.esk';rp.write_text('\n'.join(replay)+'\n')
        assert run([a.eshkol_bin,'-r',str(rp)],'independent_replay_wall_seconds')=='PASS producer-free staging 9\n'
        entry=d/'kernels.esk';entry.write_text('\n'.join(datum(x) for x in definitions)+'\n'+'\n'.join(test)+'\n')
        jit=run([a.eshkol_bin,'-r',str(entry)],'emitted_jit_wall_seconds');assert [F(x) for x in jit.splitlines()]==wanted
        binary=str(d/'kernels');run([a.eshkol_bin,str(entry),'-O','2','-o',binary],'aot_O2_compile_wall_seconds')
        assert run([binary],'aot_execution_wall_seconds')==jit
    specialization=sum(x[4]/x[6] for x in artifacts);checking=sum(x[5]/x[6] for x in artifacts)
    result={'schema_version':1,'model':'coupled Hawkes reduced mean generator, w=wn/4',
        'residual_rows':4,'native_test_values':128,'producer_free_checks':9,'engines':['jit','aot-O2'],
        'independent_oracle':'full-source invariant identities and exact five-variable polynomial equality',
        'specialization_seconds':specialization,'residual_check_seconds':checking,**timings,
        'charged_deployment_seconds':sum(timings[k] for k in ['construction_wall_seconds','independent_replay_wall_seconds','aot_O2_compile_wall_seconds','aot_execution_wall_seconds']),
        'performance_claim':'deployment costs measured; no amortized speedup established',
        'compiler_sha256':hashlib.sha256(Path(a.eshkol_bin).read_bytes()).hexdigest(),
        'files_sha256':{str((BASE/n).resolve().relative_to(root)):hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ['staging.esk','staging_test.esk','run_staging_gate.py']}}
    if a.results:a.results.write_text(json.dumps(result,indent=2)+'\n')
    print('STAGING PASS: four certified model rows, producer-free replay, 128 independently checked JIT/AOT-O2 outputs')
if __name__=='__main__':main()
