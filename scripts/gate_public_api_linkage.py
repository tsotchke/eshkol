#!/usr/bin/env python3
"""Generate live linker relocations for every eshkol_* prototype in eshkol.h.

CMake compiles this against the real public header and links eshkol-static with
its declared dependencies. No nm-only attestation or hand-maintained symbol
allowlist can turn a missing definition green. This gate covers the umbrella
public C API; backend-private C++ methods and optional headers are outside its
scope. Runtime capability probes are enforced separately by the same CI lane.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from gate_compiler_architecture import code

ROOT=Path(__file__).resolve().parents[1]


def prototypes(header):
    clean=code(header)
    found=[]
    for m in re.finditer(r'(?:^|(?<=[;{}]))([^;{}]*?)\b(eshkol_\w+)\s*\(([^;{}]*)\)\s*;',clean,re.S):
        if not re.search(r'\b(?:static|typedef)\b',m[1]): found.append(m[2])
    if not found: raise ValueError('no public eshkol_* prototypes found; refusing empty linkage gate')
    return sorted(set(found))


def generate(header, include='eshkol/eshkol.h'):
    names=prototypes(header)
    lines=[f'#include <{include}>', '// Each volatile global forces an actual linker relocation.']
    lines += [f'auto volatile assurance_symbol_{i} = &{name};' for i,name in enumerate(names)]
    lines += ['int main() {', '  unsigned missing = 0;']
    lines += [f'  missing += assurance_symbol_{i} == nullptr;' for i in range(len(names))]
    lines += ['  return missing ? 1 : 0;', '}']
    return '\n'.join(lines)+'\n',names


def self_test():
    compiler=shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
    if not compiler: raise RuntimeError('C++ compiler required to test real linkage failure')
    with tempfile.TemporaryDirectory(prefix='api-link-selftest-') as temp:
        root=Path(temp)
        header='extern "C" { int eshkol_present(void); int eshkol_missing(void); }\n'
        source,names=generate(header,'api.h')
        if len(names)!=2: raise AssertionError('prototype coverage')
        (root/'api.h').write_text(header)
        (root/'probe.cpp').write_text(source)
        (root/'definitions.cpp').write_text('extern "C" int eshkol_present() {return 1;}\n')
        argv=[compiler,'-std=c++17','-I',str(root),str(root/'probe.cpp'),str(root/'definitions.cpp'),'-o',str(root/'probe')]
        broken=subprocess.run(argv,text=True,capture_output=True)
        if broken.returncode==0 or 'eshkol_missing' not in broken.stderr:
            raise AssertionError('missing public definition failed to produce a named linker error')
        (root/'definitions.cpp').write_text('extern "C" int eshkol_present() {return 1;}\nextern "C" int eshkol_missing() {return 2;}\n')
        subprocess.run(argv,check=True,capture_output=True)
        subprocess.run([str(root/'probe')],check=True)
        try: prototypes('// void eshkol_fake();')
        except ValueError: pass
        else: raise AssertionError('empty public surface accepted')
    return {'status':'PASS','controls':['missing-definition-link-fails','complete-link-executes','empty-surface-fails']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--header',type=Path,default=ROOT/'inc/eshkol/eshkol.h')
    p.add_argument('--generate',type=Path)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--run',type=Path,help='execute the linked public API probe and emit oracle evidence')
    a=p.parse_args()
    if a.self_test: print(json.dumps(self_test())); return 0
    if a.run:
        import time, hashlib
        try:
            result=subprocess.run([str(a.run.resolve())],capture_output=True,text=True,timeout=60)
            report={'kind':'compiler_assurance','name':'compiler_public_api_linkage',
                'value':'PASS' if result.returncode==0 else 'FAIL','exit_code':result.returncode,
                'stdout':result.stdout,'stderr':result.stderr,'binary':str(a.run.resolve()),
                'binary_sha256':hashlib.sha256(a.run.read_bytes()).hexdigest(),'timestamp':time.time()}
        except (OSError,subprocess.TimeoutExpired) as exc:
            report={'kind':'compiler_assurance','name':'compiler_public_api_linkage',
                    'value':'FAIL','error':str(exc),'timestamp':time.time()}
        trace=ROOT/'scripts/icc_traces/compiler-public-api.jsonl'
        trace.parent.mkdir(parents=True,exist_ok=True)
        trace.write_text(json.dumps(report)+'\n')
        print(json.dumps(report))
        return report['value']!='PASS'
    if not a.generate: p.error('--generate, --run, or --self-test required')
    source,names=generate(a.header.read_text())
    a.generate.parent.mkdir(parents=True,exist_ok=True)
    a.generate.write_text(source)
    print(json.dumps({'generated':str(a.generate),'symbols':names,'count':len(names)}))
    return 0

if __name__=='__main__': sys.exit(main())
