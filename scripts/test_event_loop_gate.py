#!/usr/bin/env python3
"""Offline negative controls for the ESH-0011 ICC producer."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path
import run_event_loop_gate as gate

ROOT=Path(__file__).resolve().parent.parent

def main()->int:
    checks=[]
    def record(name,fn):
        try: fn(); ok=True
        except Exception as e: ok=False; print(f"  detail {name}: {e}")
        checks.append((name,ok)); print(f"{'PASS' if ok else 'FAIL'} {name}")
    p=subprocess.run([sys.executable,str(ROOT/'scripts/verify_event_loop_gate.py'),'--self-test'],capture_output=True,text=True)
    record('log controls (echo/status0, compile error, missing, skip, renamed, count, backend)',lambda: (_ for _ in ()).throw(AssertionError(p.stdout+p.stderr)) if p.returncode else None)
    def rejected(fn):
        try: fn()
        except gate.GateFailure: return
        raise AssertionError('invalid input was accepted')
    record('wrong source rejected',lambda: rejected(lambda: gate.require_source_owner('/tmp/other-source',str(ROOT))))
    record('stale binary rejected',lambda: rejected(lambda: gate.require_freshness_status(1,'eshkol-run predates source')))
    def compile_error():
        from verify_event_loop_gate import verify_compile_log, GateError
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'compile.log'; p.write_text(' [JIT] LLVM ERROR: synthetic link failure\n')
            try: verify_compile_log(p,0)
            except GateError: return
            raise AssertionError('status-zero compile error was accepted')
    record('AOT compile diagnostic rejected with status zero',compile_error)
    def generated_rule_controls():
        source='/tmp/event-loop-gate-source'
        one_ninja=f'build CMakeFiles/runtime.dir/lib/core/event_loop_kqueue.c.o: CC {source}/lib/core/event_loop_kqueue.c'
        one_make=f'CMakeFiles/runtime.dir/lib/core/event_loop_epoll.c.o: {source}/lib/core/event_loop_epoll.c'
        gate.select_backend_from_rules([one_ninja],source)
        gate.select_backend_from_rules([one_make],source)
        for bad in (
            f'build CMakeFiles/runtime.dir/lib/core/event_loop_stub.c.o: CC {source}/lib/core/event_loop_stub.c',
            one_ninja+f'\nbuild CMakeFiles/runtime.dir/lib/core/event_loop_epoll.c.o: CC {source}/lib/core/event_loop_epoll.c',
            f'build CMakeFiles/runtime.dir/lib/core/other.c.o: CC {source}/lib/core/other.c',
        ):
            rejected(lambda: gate.select_backend_from_rules([bad],source))
        rejected(lambda: gate.select_backend_from_rules(['build CMakeFiles/runtime.dir/lib/core/event_loop_kqueue.c.o: CC /other/source/lib/core/event_loop_kqueue.c'],source))
    record('Ninja/Make backend rule selection and stub/multiple/missing/wrong-source controls',generated_rule_controls)
    def stale_pass_cleared():
        with tempfile.TemporaryDirectory() as d:
            trace=Path(d)/'trace.jsonl'; trace.write_text(json.dumps({'kind':'eshkol_smoke','name':'event_loop_works','value':'PASS'})+'\n')
            env=os.environ.copy(); env['TRACE_FILE']=str(trace)
            proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_event_loop_gate.py'),'--build-dir',str(Path(d)/'missing-build'),'--evidence-dir',d],env=env,capture_output=True,text=True)
            rows=[json.loads(x) for x in trace.read_text().splitlines()]
            assert proc.returncode!=0 and len(rows)==1 and rows[0]['value']=='FAIL'
            assert Path(rows[0]['receipt_path']).is_file() and rows[0]['receipt_sha256']
    record('stale PASS cleared before failure verdict',stale_pass_cleared)
    total=len(checks); passed=sum(ok for _,ok in checks)
    print(f'event-loop gate failure controls: {passed}/{total} passed')
    return 0 if passed==total else 1
if __name__=='__main__': sys.exit(main())
