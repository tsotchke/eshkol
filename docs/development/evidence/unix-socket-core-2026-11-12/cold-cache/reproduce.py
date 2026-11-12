import concurrent.futures, hashlib, json, os, pathlib, shutil, subprocess, tempfile, datetime
src=pathlib.Path('/home/gabe/Documents/ChatGPT/eshkol-sockets')
build=pathlib.Path('/home/gabe/.codex/worktrees/unix-socket-core-20261112/build')
out=pathlib.Path('/home/gabe/.codex/worktrees/unix-socket-core-20261112/checks/cold-cache');out.mkdir(exist_ok=True)
cache=pathlib.Path(tempfile.mkdtemp(prefix='eshkol-jit-cold-cache-'))
shutil.copy2(build/'stdlib.bc',cache/'stdlib.bc')
probe=cache/'probe.esk';probe.write_text('(require stdlib)\n(display "PASS: concurrent cold stdlib cache")\n(newline)\n')
env=os.environ.copy();env.update(ESHKOL_LIB_DIR=str(cache),ESHKOL_PATH=str(src/'lib'),ESHKOL_JIT_CACHE='0')
cmd=[str(build/'eshkol-run'),'-r',str(probe)]
def run(i):
    p=subprocess.run(cmd,cwd=src,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    (out/f'writer-{i}.log').write_text(p.stdout)
    good=p.returncode==0 and p.stdout.splitlines().count('PASS: concurrent cold stdlib cache')==1 and not any(x in p.stdout for x in ['JIT session error','LLVM ERROR:','Warning: stdlib'])
    return {'writer':i,'exit':p.returncode,'passed':good,'log':f'writer-{i}.log'}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(run,range(3)))
objects=list(cache.glob('stdlib-jit-*.o'));temps=list(cache.glob('*.tmp'))
objects_ok=len(objects)==1
if objects:
    inspected=subprocess.run(['readelf','-h',str(objects[0])],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/'object-header.log').write_text(inspected.stdout);objects_ok &= inspected.returncode==0
passed=all(r['passed'] for r in results) and objects_ok and not temps
record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'environment':{k:env[k] for k in ['ESHKOL_LIB_DIR','ESHKOL_PATH','ESHKOL_JIT_CACHE']},'writers':results,'object_count':len(objects),'leftover_temporary_files':[str(x) for x in temps],'cache_object_sha256':hashlib.sha256(objects[0].read_bytes()).hexdigest() if objects else None,'passed':passed}
(out/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));raise SystemExit(0 if passed else 1)
