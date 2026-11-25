#!/usr/bin/env python3
"""Standing signed forecasts must equal fresh full-model ensemble evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from live_oracle import group_forecasts, source_tools
from fractions import Fraction as F

BASE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--eshkol-bin',required=True)
    p.add_argument('--source-root',type=Path,default=BASE.parents[1])
    p.add_argument('--compile-linalg-source',action='store_true')
    p.add_argument('--results',type=Path)
    p.add_argument('--jit',action='store_true',help='Compatibility alias; JIT and AOT are always required')
    args=p.parse_args()
    root=args.source_root.resolve()
    env=dict(os.environ,ESHKOL_PATH=str(root/'lib'),ESHKOL_JIT_CACHE='0',ESHKOL_JIT_COMPILE_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='live-forecast-') as directory:
        work=Path(directory)
        def run(command):
            r=subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                             capture_output=True,text=True,timeout=240)
            if r.returncode:
                raise RuntimeError(r.stdout[-1800:]+r.stderr[-1800:])
            return r.stdout
        source=BASE/'live_test.esk'
        extra=[str(root/'lib/core/exact_linalg.esk')] if args.compile_linalg_source else []
        run([args.eshkol_bin,str(source),*extra,'-O','0','-o',str(work/'live')])
        aot=run([str(work/'live')])
        jit=run([args.eshkol_bin,'-r',str(source)])
        if jit != aot:
            raise AssertionError('In-process JIT and AOT receipts differ')
        lines=jit.splitlines()
        if len(lines)!=19 or lines[-1]!='PASS live 29' or not lines[-2].startswith('SOURCE '):
            raise AssertionError('Incomplete live forecast receipt: '+jit[-500:])
        tools=source_tools(root)
        spec=tools.read(lines[-2][7:])
        labels=list(range(1,13))+['parameter-revision','query-upgrade']
        for label,line in zip(labels,lines[:14]):
            assert line.startswith('LIVE ')
            actual,w,rows,output=tools.read(line[5:])
            assert actual==label
            assert output==group_forecasts(root,spec,rows,w)
        activation=tools.read(lines[-3][18:])
        for line,w in zip(lines[14:16],[0,F(1,2)]):
            assert line.startswith('ACTIVATED ')
            actual,rows,output=tools.read(line[10:]);assert actual==w
            assert output==group_forecasts(root,activation,rows,w)
        result={'schema_version' :1,'engines':['jit','aot-O0'],
                'jit_qualification':'passed','jit_executable_cache':'disabled',
                'eshkol_bin_sha256':hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
                'checks_per_engine':29,
                'full_model_ensemble_prefixes':16, 'activation_dimensions':[3,4],'signed_ticks':12,
                'covariance':'aggregate moments before nonlinear observation',
                'global_revisions':'explicit reconstruction from retained raw observations',
                'stdout_sha256':hashlib.sha256(jit.encode()).hexdigest(),
                'files_sha256':{str((BASE/name).resolve().relative_to(root)):hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                  for name in ['live.esk','live_test.esk','../hawkes-forecast/source.esk','../hawkes-forecast/forecast.esk','live_oracle.py','../hawkes-forecast/oracle.py','run_live_gate.py','../../lib/core/observable_closure.esk','../../scripts/run_observable_closure_gate.py']}}
        if args.results:args.results.write_text(json.dumps(result,indent=2)+'\n')
    print('LIVE FORECAST PASS: 16 full-model prefixes, signed updates, exact jets, admission and persistence, uncached JIT/AOT parity')


if __name__=='__main__':main()
