import datetime, hashlib, json, os, pathlib, subprocess, time

root=pathlib.Path('/home/gabe/Documents/ChatGPT/eshkol-sockets')
work=pathlib.Path('/tmp/eshkol-unix-socket-sanitized')
work.mkdir(exist_ok=True)
records=[]
env=os.environ.copy()
env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1:abort_on_error=1'
env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
def run(name,cmd,timeout=60):
    started=time.monotonic()
    try:
        r=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        output=r.stdout; status=r.returncode
    except subprocess.TimeoutExpired as e:
        output=e.stdout or b'';status='TIMEOUT'
    (work/(name+'.log')).write_bytes(output)
    records.append(dict(name=name,command=cmd,exit_code=status,seconds=round(time.monotonic()-started,3),log=name+'.log'))
    (work/'results.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
    print(name,status,flush=True)
    if status!=0:
        print(output.decode(errors='replace')[-12000:],flush=True)
        raise SystemExit(1)

san=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
cflags=['-std=c11','-D_GNU_SOURCE','-D_XOPEN_SOURCE=700','-O1','-g','-Wall','-Wextra','-pthread','-I'+str(root/'inc')]+san
runtime=['lib/agent/c/agent_unix_socket_stream.c','lib/core/resource_lease.c','lib/core/event_loop.c','lib/core/event_loop_epoll.c']
sources=runtime+['lib/core/runtime_handle.cpp','tests/agent/unix_socket_stream_test.c','tests/agent/unix_socket_cleanup_test.c','tests/agent/unix_socket_stream_fixture_support.c','inc/eshkol/core/unix_socket_stream.h','inc/eshkol/core/resource_lease.h','inc/eshkol/core/event_loop.h','inc/eshkol/core/handle.h','lib/core/event_loop_internal.h']
source_hashes={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources}
run('fixture-build',['clang']+cflags+['-fPIC','-shared',str(root/'tests/agent/unix_socket_stream_fixture_support.c'),'-ldl','-o',str(work/'libsocket-fixture.so')])
objects=[]
for i,source in enumerate(runtime):
    obj=work/(str(i)+'.o');objects.append(str(obj))
    run('compile-'+str(i),['clang']+cflags+['-c',str(root/source),'-o',str(obj)])
obj=work/'handle.o';objects.append(str(obj))
run('compile-handle',['clang++','-std=c++17','-O1','-g','-Wall','-Wextra','-pthread','-I'+str(root/'inc')]+san+['-c',str(root/'lib/core/runtime_handle.cpp'),'-o',str(obj)])
for case,source in [('native','tests/agent/unix_socket_stream_test.c'),('cleanup','tests/agent/unix_socket_cleanup_test.c')]:
    obj=work/(case+'.o')
    run('compile-'+case,['clang']+cflags+['-c',str(root/source),'-o',str(obj)])
    link=['clang++','-pthread']+san+[str(obj)]+objects
    if case=='native':link+=['-L'+str(work),'-lsocket-fixture','-Wl,-rpath,'+str(work),'-ldl']
    run('link-'+case,link+['-o',str(work/case)])
cases='connect_accept partial_slow_peer peer_close_epipe deadline_accept deadline_connect cancellation nul_and_large_payload descriptor_count_check capability_denial path_lifecycle destination_errors connect_cancellation capability_revocation loop_lifetime borrowed_close zero_budget'.split()
for case in cases:run('native-'+case,[str(work/'native'),case,str(work/'scratch')])
run('cleanup',[str(work/'cleanup')])
assert all(hashlib.sha256((root/s).read_bytes()).hexdigest()==h for s,h in source_hashes.items()),'sources changed during qualification'
result=dict(record_created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),platform='Linux x86_64 epoll',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_hashes=source_hashes,compiler=subprocess.check_output(['clang','--version'],text=True).splitlines()[0],sanitizers=env['ASAN_OPTIONS'],ubsan_options=env['UBSAN_OPTIONS'],scope='standalone fully instrumented Unix socket core, native acceptance and cleanup; fixture DSO linked; no compiler/JIT/AOT claim',native_scenarios=len(cases),records=records,artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [work/'native',work/'cleanup',work/'libsocket-fixture.so']},log_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(work.glob('*.log'))})
(work/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: all 16 native cases plus cleanup under ASan/UBSan/LSan',flush=True)
