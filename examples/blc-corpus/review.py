#!/usr/bin/env python3
"""Review exact corpus bytes without evaluation; sandbox explicit backend runs.

Copyright (c) 2026 Richard Hoekstra. SPDX-License-Identifier: MIT
"""
import argparse, hashlib, json, os
from pathlib import Path
import re, shutil, subprocess, tempfile, stat
import run as corpus
BASE=Path(__file__).resolve().parent


def view(program):
    bits=corpus.canonical_bits(program);nodes=corpus.closed_classic(bits)
    # Independent structural re-encoding from the admitted node table.
    pending=[len(nodes)-1];parts=[]
    while pending:
        node=nodes[pending.pop()]
        if node[0]=='bound':parts.append('1'*node[1]+'0')
        elif node[0]=='lambda':parts.append('00');pending.append(node[1])
        else:parts.append('01');pending.extend([node[2],node[1]])
    if ''.join(parts)!=bits:raise ValueError('node-table re-encoding differs')
    lines=['; Classic BLC: postorder De Bruijn syntax, not evaluated.',
           '; A bound index is 1-based; each child precedes its parent.',
           '; logical_bits='+str(len(bits))+' nodes='+str(len(nodes)),
           '; packed_sha256='+corpus.digest(corpus.corpus_bytes(program['packed_path']))]
    for i,node in enumerate(nodes):
        operands=' '.join(str(x) if node[0]=='bound' else 'n'+str(x) for x in node[1:])
        lines.append('n'+str(i)+' = '+node[0]+' '+operands)
    lines.append('root = n'+str(len(nodes)-1))
    return '\n'.join(lines)+'\n'


def check():
    manifest,fixtures=corpus.check()
    sealed={item['path'] for item in manifest['file']}
    for program in manifest['program']:
        path='review/'+program['id']+'.txt'
        if path not in sealed or corpus.corpus_bytes(path).decode('ascii')!=view(program):
            raise ValueError('missing/stale readable review: '+path)
    print('REVIEW-PASS: 7 exact readable programs; bounded syntax, full consumption, closed scope and packed padding')
    return manifest


def generate():
    manifest=corpus.tomllib.loads(corpus.corpus_bytes('manifest.toml').decode())
    (BASE/'review').mkdir(exist_ok=True)
    if (BASE/'review').is_symlink():raise ValueError('review directory is a symlink')
    extra=['review.py','review_self_test.py','REVIEW.md']
    for program in manifest['program']:
        if program['id'] not in corpus.PROGRAMS:raise ValueError('unknown program')
        relative='review/'+program['id']+'.txt';path=BASE/relative
        if path.is_symlink():raise ValueError('review output is a symlink')
        path.write_text(view(program),encoding='ascii');extra.append(relative)
    text=(BASE/'manifest.toml').read_text()
    present={item['path'] for item in manifest['file']}
    for relative in extra:
        if relative not in present:text+='\n[[file]]\npath = '+json.dumps(relative)+'\nsha256 = "'+'0'*64+'"\n'
    def seal(match):
        relative=json.loads(match[1]);return 'path = '+match[1]+'\nsha256 = "'+corpus.digest(corpus.corpus_bytes(relative))+'"'
    text=re.sub(r'path = ("[^"\n]+")\nsha256 = "[0-9a-f]{64}"',seal,text)
    (BASE/'manifest.toml').write_text(text)
    check()


def export_regular(source,destination,limit):
    # The sandbox is gone before export. Never follow a sandbox-created link.
    info=source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>limit:
        raise ValueError('invalid sandbox export')
    fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('sandbox export byte budget')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data)


def sandbox(args):
    check()
    bubble=shutil.which('bwrap')
    if not bubble:raise RuntimeError('bubblewrap unavailable; refusing unsandboxed execution')
    binary=Path(args.binary).resolve(strict=True)
    roots=[Path(p).resolve(strict=True) for p in args.runtime_root]
    for root in roots:
        if not root.is_dir() or root in [Path('/'),Path('/tmp'),Path('/home'),Path.home()]:
            raise ValueError('supply specific runtime directories, not workspace/home/filesystem roots')
    if not any(binary.is_relative_to(root) for root in roots):
        raise ValueError('binary must belong to an explicit runtime root')
    source_lib=corpus.LIBRARY.resolve(strict=True)
    prefix=[bubble,'--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
        '--clearenv','--setenv','PATH','/usr/bin:/bin','--setenv','HOME','/nonexistent',
        '--setenv','ESHKOL_PATH',str(source_lib),'--setenv','ESHKOL_LIB_DIR',str(source_lib),
        '--setenv','ESHKOL_JIT_CACHE','0','--setenv','ESHKOL_JIT_COMPILE_THREADS','1',
        '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
        '--tmpfs','/tmp','--proc','/proc','--dev','/dev','--ro-bind','/usr','/usr',
        '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
        '--ro-bind',str(BASE),str(BASE),'--ro-bind',str(source_lib),str(source_lib)]
    libraries=[]
    for root in roots:
        if root!=Path('/usr'):prefix+=['--ro-bind',str(root),str(root)]
        for relative in ['lib','usr/lib/x86_64-linux-gnu','lib/x86_64-linux-gnu']:
            if (root/relative).is_dir():libraries.append(str(root/relative))
    if libraries:prefix+=['--setenv','LD_LIBRARY_PATH',':'.join(libraries)]
    if Path('/etc/alternatives').is_dir():prefix+=['--ro-bind','/etc/alternatives','/etc/alternatives']
    if Path('/etc/ld.so.cache').is_file():prefix+=['--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache']
    # Python 3.10's optional TOML reader may live outside system paths.
    # Expose that package alone, never the containing home/site-packages tree.
    if corpus.tomllib.__name__ == 'tomli':
        package=Path(corpus.tomllib.__file__).resolve().parent
        prefix+=['--ro-bind',str(package),'/opt/blc-review-python/tomli',
                 '--setenv','PYTHONPATH','/opt/blc-review-python']
    # Preflight before any backend execution; no silent fallback.
    probe_code=('import json,os,errno\nreadonly=False\ntry:\n f=open('+repr(str(BASE/'run.py'))+',"r+b");f.close()\n'
                'except OSError as e: readonly=e.errno in (errno.EROFS,errno.EACCES)\n'
                'print(json.dumps({"net":os.readlink("/proc/self/ns/net"),"pid":os.readlink("/proc/self/ns/pid"),'
                '"home_visible":os.path.exists('+repr(str(Path.home()))+'),"readonly":readonly}))')
    probe=subprocess.run(prefix+['/usr/bin/python3','-c',probe_code],capture_output=True,text=True,timeout=10)
    if probe.returncode:raise RuntimeError('sandbox unavailable; backend was not started: '+probe.stderr.strip())
    isolation=json.loads(probe.stdout)
    if (isolation['net']==os.readlink('/proc/self/ns/net') or isolation['pid']==os.readlink('/proc/self/ns/pid') or isolation['home_visible'] or not isolation['readonly']):
        raise RuntimeError('namespace/home isolation did not hold; backend was not started')
    with tempfile.TemporaryDirectory(prefix='blc-sandbox-') as directory:
        forwarded=args.runner_args[1:] if args.runner_args[:1]==['--'] else args.runner_args
        command=prefix+['--bind',directory,'/work','--chdir','/work','/usr/bin/python3',str(BASE/'run.py'),'run',
            *forwarded,'--binary',str(binary),'--work-dir','/work','--report','/work/report.toml']
        result=subprocess.run(command,stdin=subprocess.DEVNULL,timeout=7200)
        report=Path(directory)/'report.toml'
        if report.exists() or report.is_symlink():export_regular(report,Path(args.report),1024*1024)
        if args.work_dir:
            target=Path(args.work_dir)
            if target.is_symlink():raise ValueError('export directory is a symlink')
            target.mkdir(parents=True,exist_ok=True)
            for output in Path(directory).glob('*.stdout'):
                export_regular(output,target/output.name,8*1024**2+65536)
        if result.returncode:
            errors='\n'.join(p.read_bytes()[-2000:].decode('utf-8','replace') for p in Path(directory).glob('*.stderr'))
            raise RuntimeError('sandboxed corpus runner failed: '+errors)
        if not report.exists():raise RuntimeError('sandbox produced no receipt')
    print('SANDBOX-PASS: isolated network/PID namespaces; read-only code/runtime; bounded backend processes')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('check');sub.add_parser('generate')
    run=sub.add_parser('run');run.add_argument('--binary',required=True);run.add_argument('--runtime-root',action='append',required=True)
    run.add_argument('--report',required=True);run.add_argument('--work-dir');run.add_argument('runner_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.mode=='check':check()
    elif args.mode=='generate':generate()
    else:sandbox(args)
if __name__=='__main__':main()
