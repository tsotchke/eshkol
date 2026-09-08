#!/usr/bin/env python3
"""Fail-closed, compiler-wide inventories of AST routing and callable consumers.

This is a structural policy check, not a C++ type checker. It discovers switches
from the enum's actual case labels, ignores nested switches when checking their
parent, and rejects defaults/omissions. Intentionally partial analyses must spell
out their remaining cases; there is no grandfathered baseline or allowlist.
Callable invocation may unpack a closure only in codegenClosureCall. Consumers
are found across all backend implementation files, not a curated site list.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def code(text):
    # Preserve offsets and newlines; braces and labels in comments/strings do
    # not count as source. Raw strings are handled before ordinary strings.
    return re.sub(r'R"([^ ()\\\t\r\n]{0,16})\(.*?\)\1"|/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: re.sub(r'[^\n]', ' ', m.group()), text, flags=re.S)


def endbrace(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return i + 1
    raise ValueError('unbalanced source braces')


def ast_routes(header, sources):
    match = re.search(r'typedef\s+enum\s*\{([^{}]*)\}\s*eshkol_op_t\s*;', code(header), re.S)
    if not match:
        return [{'rule': 'ast_enum_missing'}], []
    members = set(re.findall(r'\bESHKOL_\w+_OP\b', match[1]))
    if not members:
        return [{'rule': 'ast_enum_empty'}], []
    findings, sites = [], []
    for path, source in sorted(sources.items()):
        clean = code(source)
        for sw in re.finditer(r'\bswitch\s*\([^;{}]*\)\s*\{', clean):
            start = clean.index('{', sw.start())
            stop = endbrace(clean, start)
            body = clean[start:stop]
            # Remove nested switch bodies: their labels cannot cover a parent.
            for child in reversed(list(re.finditer(r'\bswitch\s*\([^;{}]*\)\s*\{', body))):
                child_start = body.index('{', child.start())
                child_end = endbrace(body, child_start)
                body = body[:child.start()] + ' ' * (child_end-child.start()) + body[child_end:]
            cases = set(re.findall(r'\bcase\s+(ESHKOL_\w+_OP)\s*:', body))
            if not cases and not re.search(r'(?:operation\s*\.\s*op|\bop\s*->\s*op)', sw.group()):
                continue
            site = {'path': path, 'line': clean.count('\n', 0, sw.start())+1,
                    'cases': sorted(cases), 'missing': sorted(members-cases)}
            sites.append(site)
            if members-cases:
                findings.append({**site, 'rule': 'ast_operation_omitted'})
            if re.search(r'\bdefault\s*:', body):
                findings.append({'rule': 'ast_default_masks_new_operation',
                                 'path': path, 'line': site['line']})
    if not sites:
        findings.append({'rule': 'ast_no_dispatch_sites'})
    return findings, sites


def callable_consumers(sources):
    findings, sites = [], []
    # Function definitions, including inline members. A parent enclosing a
    # lambda remains the consumer. No path can exempt arbitrary new functions.
    definition = re.compile(r'\b([A-Za-z_]\w*(?:::\w+)*)\s*\([^;{}]*\)\s*(?:const\s*)?\{')
    for path, source in sorted(sources.items()):
        clean = code(source)
        for m in definition.finditer(clean):
            name = m[1].split('::')[-1]
            if name in {'if', 'for', 'while', 'switch', 'catch'}:
                continue
            start = clean.index('{', m.start())
            body = clean[start:endbrace(clean, start)]
            # Direct LLVM invocation coupled to runtime closure representation
            # is the dangerous ABI consumer; mere closure construction is not.
            if not (re.search(r'\bCreateCall\s*\(\s*\w+\s*,\s*\w+\s*,', body) and
                    re.search(r'\b(?:closure_ptr\w*|num_captures|capture_count)\b', body) and
                    re.search(r'\bCreate(?:Load|IntToPtr)\s*\(', body)):
                continue
            site = {'path': path, 'line': clean.count('\n', 0, m.start())+1, 'function': name}
            sites.append(site)
            if (path, name) != ('lib/backend/llvm_codegen.cpp', 'codegenClosureCall'):
                findings.append({**site, 'rule': 'callable_bypasses_canonical_dispatch'})
    if not any((s['path'], s['function']) == ('lib/backend/llvm_codegen.cpp', 'codegenClosureCall') for s in sites):
        findings.append({'rule': 'canonical_callable_dispatch_missing'})
    return findings, sites


def scan(root):
    sources = {str(p.relative_to(root)): p.read_text() for folder in ('lib/frontend', 'lib/backend')
               for p in (root/folder).rglob('*') if p.suffix in {'.cpp', '.c', '.h'}}
    ast, switches = ast_routes((root/'inc/eshkol/eshkol.h').read_text(), sources)
    calls, consumers = callable_consumers({p:s for p,s in sources.items() if p.startswith('lib/backend/')})
    return {'status': 'FAIL' if ast or calls else 'PASS', 'findings': ast+calls,
            'ast_sites': switches, 'callable_sites': consumers,
            'scope': 'All lib/frontend and lib/backend C/C++ sources; structural lexical policy'}


def self_test():
    h = 'typedef enum { ESHKOL_A_OP, ESHKOL_B_OP } eshkol_op_t;'
    good = 'void walk() { switch (op) { case ESHKOL_A_OP: break; case ESHKOL_B_OP: break; } }'
    if ast_routes(h, {'x':good})[0]: raise AssertionError('clean exhaustive switch rejected')
    for bad in [good.replace('case ESHKOL_B_OP:', ''), good.replace('case ESHKOL_B_OP:', 'default:'),
                good.replace('case ESHKOL_B_OP:', 'switch (x) { case ESHKOL_B_OP: break; }')]:
        if not ast_routes(h, {'x':bad})[0]: raise AssertionError('AST fault survived')
    if not ast_routes(h.replace('ESHKOL_B_OP', 'ESHKOL_B_OP, ESHKOL_C_OP'), {'x':good})[0]:
        raise AssertionError('new enum member survived')
    canonical = 'void codegenClosureCall() { auto closure_ptr = b.CreateLoad(x); b.CreateCall(ft, fp, args); }'
    if callable_consumers({'lib/backend/llvm_codegen.cpp':canonical})[0]: raise AssertionError('canonical rejected')
    if not callable_consumers({'lib/backend/llvm_codegen.cpp':canonical, 'new.cpp':canonical.replace('codegenClosureCall', 'rogue')})[0]:
        raise AssertionError('new callable consumer survived')
    if not callable_consumers({'new.cpp':canonical})[0]: raise AssertionError('duplicate dispatcher name exempted')
    if not callable_consumers({})[0]: raise AssertionError('missing dispatcher passed')
    return {'status':'PASS', 'controls':['omitted-case','default','nested-case','new-enum','new-consumer','missing-dispatcher']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--trace-dir', type=Path, default=ROOT/'scripts/icc_traces')
    args = parser.parse_args()
    report = self_test() if args.self_test else scan(args.root)
    if not args.self_test:
        import time
        args.trace_dir.mkdir(parents=True,exist_ok=True)
        (args.trace_dir/'compiler-architecture.jsonl').write_text(json.dumps({
            'kind':'compiler_assurance','name':'compiler_architecture',
            'value':report['status'],'timestamp':time.time(),
            'findings':report['findings']})+'\n')
    print(json.dumps(report, indent=2))
    return report['status'] != 'PASS'

if __name__ == '__main__':
    sys.exit(main())
