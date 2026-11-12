import pathlib, subprocess, os, json, hashlib, datetime
root=pathlib.Path('/home/gabe/Documents/ChatGPT/eshkol-sockets')
work=pathlib.Path('/tmp/eshkol-unix-socket-sanitized')
sources=['tests/agent/subprocess_stream_test.c','tests/agent/subprocess_stream_clock_test.c','tests/agent/subprocess_stream_watch_failure_test.c','tests/agent/run_subprocess_stream_test.sh','lib/agent/c/agent_subprocess.c','inc/eshkol/core/subprocess_stream.h','lib/core/resource_lease.c','lib/core/runtime_handle.cpp','lib/core/event_loop.c','lib/core/event_loop_epoll.c','inc/eshkol/core/event_loop.h','inc/eshkol/core/resource_lease.h','inc/eshkol/core/handle.h','lib/core/event_loop_internal.h']
hashes={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources}
records=[]
for name,extra in [('process-native',{}),('process-sanitized',{'CC':'clang','CXX':'clang++','ESHKOL_STREAM_SANITIZE':'1','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1:abort_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})]:
    env=os.environ.copy();env.update(extra)
    cmd=['tests/agent/run_subprocess_stream_test.sh']
    r=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    log=work/(name+'.log');log.write_bytes(r.stdout)
    records.append(dict(name=name,command=cmd,environment=extra,exit_code=r.returncode,log=log.name,log_sha256=hashlib.sha256(r.stdout).hexdigest()))
    print(name,r.returncode,flush=True)
    if r.returncode:print(r.stdout.decode(errors='replace'));break
assert all(hashlib.sha256((root/s).read_bytes()).hexdigest()==h for s,h in hashes.items()),'source changed'
x=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_hashes=hashes,records=records,scope='Native actual PROCESS/shared socketpair integration, clock failure, missing capability checker, and wrapped watch-removal failure controls. Normal and ASan/UBSan/leak runs.')
(work/'process-results.json').write_text(json.dumps(x,indent=2)+'\n')
assert len(records)==2 and all(r['exit_code']==0 for r in records)
