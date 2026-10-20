#!/usr/bin/env python3
"""Produce source-bound ESH-0011 evidence; never runs broad ICC smoke."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, re, shlex, subprocess, sys, time
from pathlib import Path
from verify_event_loop_gate import verify_log, GateError

HERE=Path(__file__).resolve().parent.parent
FIXTURE=HERE/'tests/v1_3_edge_cases/event_loop_test.esk'
BACKENDS={'kqueue':'lib/core/event_loop_kqueue.c','epoll':'lib/core/event_loop_epoll.c','iocp':'lib/core/event_loop_iocp.c'}

class GateFailure(Exception): pass

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def git(*args:str)->str:
    p=subprocess.run(['git','-C',str(HERE),*args],capture_output=True,text=True)
    if p.returncode: raise GateFailure(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()

def cmake_source(cache:Path)->str:
    m=re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$',cache.read_text(),re.M)
    if not m: raise GateFailure('CMakeCache lacks CMAKE_HOME_DIRECTORY')
    return str(Path(m.group(1)).resolve())

def require_source_owner(actual:str, expected:str)->None:
    if str(Path(actual).resolve()) != str(Path(expected).resolve()):
        raise GateFailure(f'wrong CMake source: {actual} (expected {expected})')

def require_freshness_status(status:int, output:str)->None:
    if status != 0: raise GateFailure('check_build_fingerprint rejected stale build: '+output[-1200:])

def reset_trace(trace:Path)->None:
    trace.parent.mkdir(parents=True,exist_ok=True)
    trace.write_text('',encoding='utf-8')

def clean_source()->tuple[str,str]:
    rev=git('rev-parse','HEAD'); dirty=git('status','--porcelain','--untracked-files=all')
    if dirty: raise GateFailure('source tree dirty: '+dirty[:600])
    return rev,dirty

def identify_backend(build:Path)->str:
    # Ninja and Make both expose the compiled object rule. Do not infer from
    # uname or from a comment; require exactly one selected source rule.
    rule_files=list(build.rglob('build.ninja'))+list(build.rglob('build.make'))
    if not rule_files: raise GateFailure('no generated Ninja or Make rules found')
    texts=[]
    for p in rule_files:
        try: texts.append(p.read_text(errors='ignore'))
        except OSError: continue
    return select_backend_from_rules(texts,str(HERE))

def select_backend_from_rules(texts:list[str], source_root:str)->str:
    rx=re.compile(r'^\s*(?:build\s+)?[^\n]*event_loop_(kqueue|epoll|iocp|stub)\.c\.(?:o|obj)\s*:',re.M)
    rules=[]
    for text in texts:
        for line in text.splitlines():
            if rx.search(line): rules.append((rx.search(line).group(1),line))
    actual=sorted({name for name,_ in rules})
    if len(actual)!=1 or actual[0]=='stub': raise GateFailure(f'generated rules must select exactly one supported backend object, found {actual}')
    name=actual[0]; expected=str(Path(source_root)/BACKENDS[name])
    selected=[line for n,line in rules if n==name]
    absolute=re.compile(r'(?:^|\s)(/[\w .@+\-/]+/lib/core/event_loop_'+name+r'\.c)(?=$|\s)')
    abs_sources=[m.group(1).strip() for line in selected for m in absolute.finditer(line)]
    if abs_sources and any(str(Path(p).resolve())!=str(Path(expected).resolve()) for p in abs_sources):
        raise GateFailure(f'generated {name} rule points outside selected source root: {abs_sources}')
    if not any(expected in line or BACKENDS[name] in line for line in selected):
        raise GateFailure(f'generated {name} object rule does not name expected source {expected}')
    return name

def append_event(trace:Path, value:str, detail:str, payload:dict)->None:
    trace.parent.mkdir(parents=True,exist_ok=True)
    ev={'kind':'eshkol_smoke','name':'event_loop_works','value':value,'snippet':detail,'confidence':0.95,**payload}
    with trace.open('a',encoding='utf-8') as f: f.write(json.dumps(ev,sort_keys=True,separators=(',',':'))+'\n')

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--build-dir',default=os.environ.get('BUILD_DIR',str(HERE/'build'))); ap.add_argument('--evidence-dir',default=os.environ.get('EVENT_LOOP_EVIDENCE_DIR',str(HERE.parent/'evidence'))); ap.add_argument('--trace-file',default=os.environ.get('TRACE_FILE'))
    a=ap.parse_args(); build=Path(a.build_dir).resolve(); evidence=Path(a.evidence_dir).resolve(); evidence.mkdir(parents=True,exist_ok=True)
    trace=Path(a.trace_file).resolve() if a.trace_file else evidence/'eshkol_smoke.jsonl'; trace.parent.mkdir(parents=True,exist_ok=True)
    # Remove prior verdict before any preflight so a failure cannot inherit green.
    reset_trace(trace)
    work=evidence/f'run-{os.getpid()}-{int(time.time())}'; work.mkdir(parents=True,exist_ok=False)
    receipt_path=evidence/'event-loop-receipt.json'
    state={'source_root':str(HERE),'build_dir':str(build),'trace_file':str(trace),'work_dir':str(work),'commands':{},'statuses':{},'env':{},'backend':None,'source_revision':None}
    payload={'receipt_path':str(receipt_path),'receipt_sha256':None,'source_revision':None,'backend':None,'commands':{},'env':{}}
    def finish(status:str,why:str)->int:
        state['result']=status; state['detail']=why; state['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        receipt_path.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        payload.update({'receipt_sha256':sha(receipt_path),'source_revision':state.get('source_revision'),'backend':state.get('backend'),'commands':state['commands'],'env':state['env']})
        append_event(trace,status,why,payload)
        print(f"event-loop gate: {status}: {why}",file=sys.stderr if status!='PASS' else sys.stdout)
        return 0 if status=='PASS' else 1
    try:
        rev,_=clean_source(); state['source_revision']=rev; payload['source_revision']=rev
        tree_oid=git('rev-parse','HEAD^{tree}'); state['source_tree_oid']=tree_oid; payload['source_tree_oid']=tree_oid
        run=build/'eshkol-run'; lib=build/'libeshkol-runtime.a'; cache=build/'CMakeCache.txt'; stdobj=build/'stdlib.o'; stdbc=build/'stdlib.bc'
        for p in (run,lib,cache,stdobj,stdbc,FIXTURE):
            if not p.is_file(): raise GateFailure(f'missing required artifact/input: {p}')
        if not os.access(run,os.X_OK): raise GateFailure(f'compiler is not executable: {run}')
        owner=cmake_source(cache)
        require_source_owner(owner,str(HERE))
        backend=identify_backend(build); backend_src=HERE/BACKENDS[backend]; state['backend']=backend; payload['backend']=backend
        # Existing whole-input freshness check covers committed and dirty C/C++ source paths.
        freshness=[sys.executable,str(HERE/'scripts/check_build_fingerprint.py'),'--repo-root',str(HERE),'--build-dir',str(build),'--no-trace','--format','json']
        state['commands']['freshness_before']=freshness
        fp=subprocess.run(freshness,cwd=HERE,capture_output=True,text=True); state['statuses']['freshness_before']=fp.returncode; state['freshness_before_stdout']=fp.stdout; state['freshness_before_stderr']=fp.stderr
        require_freshness_status(fp.returncode,fp.stdout+fp.stderr)
        inputs=[cache,HERE/'CMakeLists.txt',HERE/'lib/core/event_loop.c',backend_src,FIXTURE]
        pre={str(p):sha(p) for p in inputs}; state['source_hashes_before']=pre
        tracked=[run,lib,stdobj,stdbc,FIXTURE]
        state['artifact_hashes_before']={str(p):sha(p) for p in tracked}
        cache_dir=work/'empty-orc-cache'; cache_dir.mkdir()
        jitcmd=[str(run),'-r',str(FIXTURE),'-L'+str(build)]
        jit_env={'ESHKOL_JIT_CACHE':'0','ESHKOL_JIT_CACHE_DIR':str(cache_dir)}
        state['commands']['orc_cache0']=jitcmd; state['env']['orc_cache0']=jit_env
        def execute(key,cmd,env=None):
            merged=os.environ.copy(); merged.update(env or {})
            try: p=subprocess.run(cmd,cwd=HERE,env=merged,capture_output=True,text=True,timeout=180)
            except (OSError,subprocess.TimeoutExpired) as e: state['statuses'][key]=124; state[key+'_log']=str(e); return 124
            state['statuses'][key]=p.returncode; state[key+'_stdout']=p.stdout; state[key+'_stderr']=p.stderr
            (work/(key+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8'); return p.returncode
        rc=execute('orc_cache0',jitcmd,jit_env); jitlog=work/'orc_cache0.log'
        aot=work/'event_loop_test_aot'; compilecmd=[str(run),str(FIXTURE),'-o',str(aot),'-L'+str(build)]; state['commands']['aot_compile']=compilecmd; state['env']['aot_compile']={}
        crc=execute('aot_compile',compilecmd); clog=work/'aot_compile.log';
        if crc==0 and aot.is_file() and os.access(aot,os.X_OK):
            state['aot_binary_hash_before_run']=sha(aot)
            arun=[str(aot)]; state['commands']['aot_run']=arun; state['env']['aot_run']={}
            arc=execute('aot_run',arun)
            if aot.exists(): state['aot_binary_hash_after_run']=sha(aot)
        else:
            state['commands']['aot_run']=[]; state['env']['aot_run']={}; state['statuses']['aot_run']=125; arc=125
        if rc or crc or arc: raise GateFailure(f'command statuses ORC={rc}, AOT compile={crc}, AOT={arc}')
        # Include compiler output in diagnostics and exact assertion verification.
        for log in (jitlog,work/'aot_run.log'):
            verify_log(log,0,backend)
        from verify_event_loop_gate import verify_compile_log
        verify_compile_log(clog,crc)
        fresh_after=subprocess.run(freshness,cwd=HERE,capture_output=True,text=True); state['commands']['freshness_after']=freshness; state['statuses']['freshness_after']=fresh_after.returncode; state['freshness_after_stdout']=fresh_after.stdout; state['freshness_after_stderr']=fresh_after.stderr
        try: require_freshness_status(fresh_after.returncode,fresh_after.stdout+fresh_after.stderr)
        except GateFailure as e: raise GateFailure('post-run whole-input freshness check failed: '+str(e))
        rev_after,_=clean_source()
        if rev_after!=rev: raise GateFailure(f'source revision changed during gate: {rev}->{rev_after}')
        tree_after=git('rev-parse','HEAD^{tree}')
        if tree_after!=tree_oid: raise GateFailure(f'source tree changed during gate: {tree_oid}->{tree_after}')
        post={str(p):sha(p) for p in inputs}; state['source_hashes_after']=post
        if post!=pre: raise GateFailure('CMake/event-loop/backend/fixture content changed during gate')
        state['artifact_hashes_after']={str(p):sha(p) for p in tracked}
        if state['artifact_hashes_after']!=state['artifact_hashes_before']: raise GateFailure('compiler/runtime/stdlib/fixture changed during gate')
        if state.get('aot_binary_hash_before_run')!=state.get('aot_binary_hash_after_run'): raise GateFailure('fresh AOT binary changed during execution')
        return finish('PASS',f'ESH-0011 exact 27 assertions passed on ORC cache=0 and AOT; host backend {backend}')
    except Exception as e:
        return finish('FAIL',str(e))
if __name__=='__main__': sys.exit(main())
