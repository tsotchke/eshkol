#!/usr/bin/env python3
"""Actual source/ESKB binding loads, not spelling-based read credit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('coverage',ROOT/'scripts/language_coverage.py')
cov=importlib.util.module_from_spec(spec);spec.loader.exec_module(cov)
PROGRAMS={
 'empty':('(display 99) (newline)','99'),
 'value':('(display (null? stream-null)) (newline)','#t'),
 'procedure':('(define held stream-car) (display (procedure? held)) (newline)','#t'),
 'builtin':('(define held abs) (display (procedure? held)) (newline)','#t'),
 'shadow':('(let ((stream-null 77) (stream-car 88)) (display stream-null) (display stream-car)) (newline)','7788'),
 'capture':('(define (reader) (lambda () stream-null)) (display (null? ((reader)))) (newline)','#t'),
 'relay':('(define (reader) (lambda () (lambda () stream-null))) (display (null? (((reader))))) (newline)','#t'),
 'shadow-capture':('(define (reader stream-null) (lambda () stream-null)) (display ((reader 77))) (newline)','77'),
 'assignment':('(let ((stream-null 77)) (set! stream-null 88)) (display 99) (newline)','99'),
 'untaken':('(if #f stream-null 99) (display 99) (newline)','99'),
 'alias':('(import (rename (core streams) (stream-null imported-null))) (display (null? imported-null)) (newline)','#t'),
}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--compiler',required=True);parser.add_argument('--vm',required=True);parser.add_argument('--switch',required=True);parser.add_argument('--evidence-dir');args=parser.parse_args()
 with tempfile.TemporaryDirectory(prefix='binding-reads-') as temporary:
  base=Path(args.evidence_dir or temporary);base.mkdir(exist_ok=True,parents=True);results=[]
  env=dict(os.environ,ESHKOL_PATH=str(ROOT/'lib'),ESHKOL_VM_NO_DISASM='1')
  for name,(program,expected) in PROGRAMS.items():
   source=base/(name+'.esk');source.write_text(program+'\n');bytecode=base/(name+'.eskb');compile_trace=base/(name+'-compile');compile_trace.mkdir(exist_ok=True)
   compiled=subprocess.run([args.compiler,'--profile','hosted-vm','--emit-eskb',str(bytecode),str(source)],env=dict(env,ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR=str(compile_trace)),text=True,capture_output=True,timeout=120)
   (base/(name+'-compile.log')).write_text(compiled.stdout+compiled.stderr)
   assert compiled.returncode==0 and bytecode.is_file(),name+': ESKB compilation failed'
   bytecode_hash=hashlib.sha256(bytecode.read_bytes()).hexdigest()
   for engine in [args.vm,args.switch]:
    for suffix,fixture in [('source',source),('eskb',bytecode)]:
     label=name+'-'+Path(engine).name+'-'+suffix;trace=base/(label+'-trace');trace.mkdir(exist_ok=True)
     assert hashlib.sha256(bytecode.read_bytes()).hexdigest()==bytecode_hash,'generated ESKB identity changed'
     ran=subprocess.run([engine,str(fixture)],env=dict(env,ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR=str(trace)),text=True,capture_output=True,timeout=120)
     (base/(label+'.log')).write_text(ran.stdout+ran.stderr)
     assert ran.returncode==0 and ran.stdout.strip()==expected,(label,ran.returncode,ran.stdout[-200:],ran.stderr[-200:])
     assert hashlib.sha256(bytecode.read_bytes()).hexdigest()==bytecode_hash,'executed ESKB identity changed'
     evidence=cov.load_runtime_evidence([str(trace)]);reads=evidence['binding_read_names'];calls=evidence['vm_dispatch_names']
     if name in ['value','capture','relay','alias']:assert 'stream-null' in evidence['value_read_names'],label
     if name in ['empty','shadow','shadow-capture','assignment','untaken']:assert 'stream-null' not in reads,label
     if name=='procedure':assert 'stream-car' in reads and 'stream-car' not in calls and 'stream-car' not in evidence['covered_names'],label
     if name=='builtin':assert 'abs' in reads,label
     results.append({'case':name,'engine':Path(engine).name,'input':suffix,'bytecode_sha256':bytecode_hash,'pass':True,'stream_null_read':'stream-null' in reads,'stream_car_invoked':'stream-car' in calls})
  (base/'RESULT.json').write_text(json.dumps({'pass':True,'checks':results,'count':len(results)},indent=2)+'\n')
  print('PASS: %d binding-read execution controls'%len(results))
if __name__=='__main__':main()
