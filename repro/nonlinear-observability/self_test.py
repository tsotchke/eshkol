#!/usr/bin/env python3
"""Contained gate-boundary controls; no native compiler/workload is launched. MIT."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from gate_support import exact_output,read,disable


def main():
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('submitted_gate',here/'gate.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    root=here.parents[1]
    checks=0
    for inherited in (None,'1'):
        with patch.dict(os.environ,{},clear=False):
            if inherited is None:os.environ.pop('ESHKOL_JIT_CACHE',None)
            else:os.environ['ESHKOL_JIT_CACHE']=inherited
            e=gate.Engine(sys.executable,root)
            calls=[]
            def boundary(command,**kwargs):
                calls.append(kwargs['env']['ESHKOL_JIT_CACHE'])
                return SimpleNamespace(returncode=0,stdout='boundary\n',stderr='')
            with patch('gate_support.subprocess.run',side_effect=boundary):
                assert e.run(['contained-boundary'])=='boundary\n'
            assert calls==['0'];checks+=1
    e=gate.Engine(sys.executable,root)
    with patch('gate_support.subprocess.run',return_value=SimpleNamespace(returncode=7,stdout='PASS\n',stderr='failed')):
        try:e.run(['contained-failure'])
        except RuntimeError:checks+=1
        else:raise AssertionError('Failed child admitted')
    try:exact_output('PASS','PASS\n')
    except AssertionError:checks+=1
    else:raise AssertionError('Truncated output admitted')
    for text in ('(1','(1) trailing'):
        try:read(text)
        except ValueError:checks+=1
        else:raise AssertionError('Incomplete proof datum admitted')
    try:gate.Engine(str(here/'no-such-compiler'),root)
    except FileNotFoundError:checks+=1
    else:raise AssertionError('Missing compiler admitted')
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'source';p.write_text('before')
        e=gate.Engine(sys.executable,d);e.capture(['source']);p.write_text('after')
        try:e.receipt(['source'])
        except AssertionError:checks+=1
        else:raise AssertionError('Source mutation admitted')
    module={'nonlinear-observability':('lib/core/nonlinear_observability.esk',{'certify-observation-task','observation-task-hint'}),
            'quantum-qec':('lib/core/quantum/qec.esk',{'stabilizer-prepare','make-qec-code','qec-certificate'}),
            'sparse-solves':('lib/core/ml/sparse_solve.esk',{'sparse-cg','sparse-minres','sparse-enclose','sparse-solve-jvp','sparse-solve-vjp','ss-solve'})}[here.name]
    blocked=disable((root/module[0]).read_text(),module[1])
    assert blocked.count('(error "Producer disabled")')==len(module[1]);checks+=1
    assert checks==9
    print('PASS bridge_readiness 9')


if __name__=='__main__':main()
