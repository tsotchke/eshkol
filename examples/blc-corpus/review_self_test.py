#!/usr/bin/env python3
"""Negative controls for corpus admission and the bounded process owner."""
import os
from pathlib import Path
import sys,tempfile,shutil,re
from types import SimpleNamespace
from unittest.mock import patch
import run as corpus
import review
checks=0

def check(condition):
    global checks
    if not condition:raise AssertionError('binary review control')
    checks+=1

def refuses(thunk):
    try:thunk()
    except (ValueError,UnicodeError,OSError):return True
    return False

for bits in ['', '001', '00100', '00100110', '10', '0011', '00x10', '0010\n', '1'*10000]:
    check(refuses(lambda:corpus.closed_classic(bits)))
check(corpus.closed_classic('0010')==[('bound',1),('lambda',0)])
check(refuses(lambda:corpus.closed_classic('00'*(corpus.MAX_DEPTH+1)+'10')))
check(len(corpus.closed_classic('00'*corpus.MAX_DEPTH+'10'))==corpus.MAX_DEPTH+1)
check(refuses(lambda:corpus.closed_classic('0'*(corpus.MAX_BITS+1))))
old=corpus.BASE
with tempfile.TemporaryDirectory() as directory:
    corpus.BASE=Path(directory)
    (corpus.BASE/'good').write_bytes(b'0010\n')
    check(corpus.corpus_bytes('good')==b'0010\n')
    check(refuses(lambda:corpus.corpus_bytes('../escape')))
    (corpus.BASE/'link').symlink_to('/etc/passwd')
    check(refuses(lambda:corpus.corpus_bytes('link')))
    (corpus.BASE/'bad.bits').write_bytes(b'0010\n\n')
    check(refuses(lambda:corpus.canonical_bits({'bits_path':'bad.bits'})))
    check(refuses(lambda:corpus.corpus_bytes('good',2)))
corpus.BASE=old
normal=corpus.command([sys.executable,'-c','import sys;sys.stdout.write("ok")'],b'',2,256)
check(normal['stdout']==b'ok' and normal['exit_code']==0 and not normal['output_limited'])
flood=corpus.command([sys.executable,'-c','import os\nwhile True: os.write(1,b"x"*65536)'],b'',2,256)
check(flood['output_limited'] and len(flood['stdout'])<=8*1024**2+65536)
blocked=corpus.command([sys.executable,'-c','import time;time.sleep(2)'],b'x'*65536,.1,256)
check(blocked['timed_out'] and blocked['wall_seconds']<1)
# Resealing a modified payload must not bypass structural/identity checks.
with tempfile.TemporaryDirectory() as directory:
    copy=Path(directory)/'corpus';shutil.copytree(old,copy)
    corpus.BASE=copy
    def reseal(relative):
        manifest=copy/'manifest.toml';text=manifest.read_text()
        pattern=r'(path = "'+re.escape(relative)+r'"\nsha256 = ")[0-9a-f]{64}("|$)'
        manifest.write_text(re.sub(pattern,lambda m:m[1]+corpus.digest((copy/relative).read_bytes())+m[2],text))
    relative='version-space.blc8';original=(copy/relative).read_bytes()
    (copy/relative).write_bytes(original[:-1]+bytes([original[-1]^1]));reseal(relative)
    check(refuses(corpus.check))
    (copy/relative).write_bytes(original);reseal(relative)
    relative='version-space.blc';original=(copy/relative).read_bytes()
    (copy/relative).write_bytes(original[:-1]+b'0010\n');reseal(relative)
    check(refuses(corpus.check))
    (copy/relative).write_bytes(b'\n'+original);reseal(relative)
    check(refuses(corpus.check))
    (copy/relative).write_bytes(original);reseal(relative)
    relative='review/version-space.txt';(copy/relative).write_text('forged view\n');reseal(relative)
    check(refuses(review.check))
    (copy/'link-output').symlink_to('/etc/passwd')
    check(refuses(lambda:review.export_regular(copy/'link-output',copy/'export',1024)))
    (copy/'regular-output').write_bytes(b'ok')
    review.export_regular(copy/'regular-output',copy/'export',8)
    check((copy/'export').read_bytes()==b'ok')
    check(refuses(lambda:review.export_regular(copy/'regular-output',copy/'export',1)))
corpus.BASE=old
with patch.object(review.shutil,'which',return_value=None), patch.object(review.subprocess,'run') as child:
    try:review.sandbox(SimpleNamespace())
    except RuntimeError:check(not child.called)
    else:raise AssertionError('missing sandbox accepted')
review.check()
print('BINARY REVIEW CONTROLS: PASS',checks)
