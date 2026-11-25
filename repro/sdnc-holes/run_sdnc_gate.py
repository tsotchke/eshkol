#!/usr/bin/env python3
"""Bounded native consumer of Eshkol's actual analytic matrix interpreter."""
import argparse,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=BASE.parents[1]);p.add_argument('--cc',default='cc');p.add_argument('--results',type=Path);a=p.parse_args();root=a.source_root.resolve()
    with tempfile.TemporaryDirectory(prefix='constructed-sdnc-') as d:
        binary=Path(d)/'holes';command=[a.cc,'-std=gnu11','-O2','-ffunction-sections','-fdata-sections',
          '-I',str(root/'lib/backend'),'-I',str(root/'inc'),str(BASE/'sdnc_holes.c'),
          '-Wl,--gc-sections','-lm','-lpthread','-o',str(binary)]
        t=time.monotonic();compiled=subprocess.run(command,capture_output=True,text=True,timeout=120);build=time.monotonic()-t
        assert compiled.returncode==0,compiled.stderr[-3000:]
        t=time.monotonic();run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30);elapsed=time.monotonic()-t
        assert run.returncode==0,run.stdout[-1000:]+run.stderr[-1000:]
        lines=run.stdout.splitlines();assert len([x for x in lines if x.startswith('TRAIN ')])==4
        assert [int(x.split()[2]) for x in lines if x.startswith('TRAIN ')]==[0,1,2,3]
        heldout=[x.split()[1:] for x in lines if x.startswith('HELDOUT ')]
        assert len(heldout)==3
        for x,y,pred,target in heldout:
            wanted=3*(18*int(x)+9)+24*int(y)+16-144*int(x)
            assert abs(float(pred)-wanted)<.03 and float(target)==wanted
        data=json.loads(lines[-1].removeprefix('SDNC '))
        assert data['heldout_mse_before']==31833 and data['heldout_mse_after']<.001
        assert data['learned_k']==3 and data['weight_bytes_unchanged'] and data['retention_programs']==4
        assert data['native_delegated_steps']==0 and data['matrix_runs']==22 and data['matrix_steps']==186
        assert data['weight_bytes']==48881688 and data['checks']==147
        # Hash the actual transitive local include closure, not just the adapter.
        deps=subprocess.check_output(command[:1]+['-std=gnu11','-MM','-I',str(root/'lib/backend'),'-I',str(root/'inc'),str(BASE/'sdnc_holes.c')],text=True)
        paths=[Path(s) for s in deps.replace('\\\n',' ').split(':',1)[1].split()]
        hashes={str(path.resolve().relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        result={'schema_version':1,'engine':'actual constructed SDNC matrix forward and object-program AD',
          'task':'calibrate designated integer hole in the denominator-cleared Hawkes drift; w=k/4',
          'training_contexts':1,'heldout_contexts':3,'updates':4,'plastic_role':'one program operand; fixed context operands supplied separately',
          'frozen_roles':['all analytic interpreter weights','program opcodes and non-input operands'],
          'optimizer':'host projected clipped gradient descent on integer hole, driven by matrix-executed object AD',
          'retention_scope':'four fixed arithmetic/local/AD programs; not a full ISA proof',
          'build_seconds':build,'wall_seconds':elapsed,**data,'source_files_sha256':hashes,
          'gate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if a.results:a.results.write_text(json.dumps(result,indent=2)+'\n')
    print('SDNC PASS: held-out MSE 31833 -> 0; 48,881,688 weight bytes frozen; four retained programs; zero native delegation')
if __name__=='__main__':main()
