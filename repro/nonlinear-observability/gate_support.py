"""Bounded, uncached native qualification helpers. MIT."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from fractions import Fraction


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def datum(x):
    if isinstance(x, (list, tuple)):
        return '(' + ' '.join(map(datum, x)) + ')'
    if type(x) is bool:
        return '#t' if x else '#f'
    return str(x)


def read(text):
    ts = re.findall(r'"(?:\\.|[^"\\])*"|#\(|[()]|[^\s()]+', text)
    i = 0
    def one():
        nonlocal i
        if i == len(ts):
            raise ValueError('Truncated datum')
        t = ts[i]
        i += 1
        if t.startswith('"'):
            return json.loads(t)
        if t in ('(', '#('):
            out = []
            while i < len(ts) and ts[i] != ')':
                out.append(one())
            if i == len(ts):
                raise ValueError('Unclosed datum')
            i += 1
            return out
        if t == ')':
            raise ValueError('Unexpected close')
        if t in ('#t', '#f'):
            return t == '#t'
        if re.fullmatch(r'[+-]?\d+', t):
            return int(t)
        if re.fullmatch(r'[+-]?\d+/\d+', t):
            return Fraction(t)
        if re.fullmatch(r'[+-]?(?:(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)', t):
            return float(t)
        return t
    out = one()
    if i != len(ts):
        raise ValueError('Trailing datum')
    return out


def exact_output(actual, expected):
    if actual != expected:
        raise AssertionError('Incomplete or unexpected native receipt: ' + actual[-1600:])


def disable(source, names):
    # Read complete top-level forms; comments and strings cannot fake a boundary.
    forms, start, depth, string, comment = [], 0, 0, False, False
    for i, ch in enumerate(source):
        if comment:
            comment = ch != '\n'
            continue
        if ch == ';' and not string:
            comment = True
            continue
        if ch == '"' and (i == 0 or source[i-1] != '\\'):
            string = not string
        if string:
            continue
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                forms.append(source[start:i+1])
    if depth != 0 or string:
        raise ValueError('Unbalanced module')
    found = set()
    for i, form in enumerate(forms):
        m = re.match(r'\(define\s+\(([^\s()]+)', form)
        if m and m[1] in names:
            forms[i] = form[:form.index(')')+1] + ' (error "Producer disabled"))'
            found.add(m[1])
    if found != set(names):
        raise AssertionError(('Missing producer definitions', found, names))
    return '\n'.join(forms)


class Engine:
    def __init__(self, binary, root):
        self.binary = str(Path(binary).resolve(strict=True))
        self.root = Path(root).resolve(strict=True)
        self.env = dict(os.environ, ESHKOL_PATH=str(self.root/'lib'), ESHKOL_JIT_CACHE='0',
                        ESHKOL_JIT_COMPILE_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        self.temp = tempfile.TemporaryDirectory(prefix='eshkol-bridge-')
        self.work = Path(self.temp.name)
        self.elapsed = {}
        self.contract = {}

    def capture(self, files):
        self.contract = {n:sha(self.root/n) for n in files}

    def run(self, command):
        p = subprocess.run(command, cwd=self.root, env=self.env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=360)
        if p.returncode:
            raise RuntimeError(f'Native command failed ({p.returncode}):\n{p.stdout[-2400:]}\n{p.stderr[-2400:]}')
        return p.stdout

    def both(self, source, name):
        start = time.monotonic()
        jit = self.run([self.binary, '-r', str(source)])
        exe = self.work/name
        self.run([self.binary, str(source), '-O', '0', '-o', str(exe)])
        aot = self.run([str(exe)])
        exact_output(aot, jit)
        self.elapsed[name] = round(time.monotonic()-start, 3)
        return jit

    def source(self, name, text):
        p = self.work/(name+'.esk')
        p.write_text(text)
        return p

    def receipt(self, files):
        if any(sha(self.root/n) != digest for n,digest in self.contract.items()):
            raise AssertionError('Qualification sources changed during the run')
        return {'schema_version': 1, 'engines': ['jit', 'aot-O0'], 'jit_executable_cache': 'disabled',
                'compiler_sha256': sha(self.binary), 'elapsed_seconds': self.elapsed,
                'files_sha256': {n:sha(self.root/n) for n in files}}
