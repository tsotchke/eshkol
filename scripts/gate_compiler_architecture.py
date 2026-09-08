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


def switches(clean):
    """Balanced conditions include routing groups' template/initializer syntax."""
    for match in re.finditer(r'\bswitch\s*\(', clean):
        paren = clean.index('(', match.start())
        depth, end = 1, paren + 1
        while depth and end < len(clean):
            depth += (clean[end] == '(') - (clean[end] == ')')
            end += 1
        brace = end
        while brace < len(clean) and clean[brace].isspace():
            brace += 1
        if brace < len(clean) and clean[brace] == '{':
            yield match.start(), clean[paren+1:end-1], brace, endbrace(clean, brace)


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
        for begin, condition, start, stop in switches(clean):
            body = clean[start+1:stop-1]
            # Child labels cannot satisfy their enclosing switch.
            for cb, _, cs, ce in reversed(list(switches(body))):
                body = body[:cb] + ' ' * (ce-cb) + body[ce:]
            cases = re.findall(r'\bcase\s+(ESHKOL_\w+_OP)\s*:', body)
            routed = 'routeAstOperation' in condition
            if routed:
                groups = re.findall(r'AstRouteGroup\s*<\s*(\w+::\w+)\s*,([^<>]*)>\s*\{\s*\}', condition)
                cases = re.findall(r'\bESHKOL_\w+_OP\b', ''.join(g[1] for g in groups))
                routes = [g[0] for g in groups]
                labels = re.findall(r'\bcase\s+(\w+::\w+)\s*:', body)
                if not groups or set(routes) != set(labels) or len(routes) != len(set(routes)):
                    findings.append({'rule': 'ast_route_arm_mismatch', 'path': path,
                                     'line': clean.count('\n', 0, begin)+1})
            if not cases and not routed and not re.search(r'(?:operation\s*\.\s*op|\bop\s*->\s*op)', condition):
                continue
            site = {'path': path, 'line': clean.count('\n', 0, begin)+1,
                    'kind': 'policy' if routed else 'dispatch',
                    'cases': sorted(set(cases)), 'missing': sorted(members-set(cases))}
            sites.append(site)
            if members-set(cases):
                findings.append({**site, 'rule': 'ast_operation_omitted'})
            if set(cases)-members or len(cases) != len(set(cases)):
                findings.append({**site, 'rule': 'ast_operation_duplicate_or_unknown'})
            if re.search(r'\bdefault\s*:', body):
                findings.append({'rule': 'ast_default_masks_new_operation',
                                 'path': path, 'line': site['line']})
            if not routed and path != 'inc/eshkol/core/ast_routing.h':
                findings.append({'rule': 'ast_bypasses_canonical_routing',
                                 'path': path, 'line': site['line']})
    canonical = [s for s in sites if s['kind'] == 'dispatch' and s['path'] == 'inc/eshkol/core/ast_routing.h']
    if len(canonical) != 1:
        findings.append({'rule': 'ast_canonical_dispatch_missing_or_duplicate'})
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
    sources = {str(p.relative_to(root)): p.read_text() for folder in ('lib/frontend', 'lib/backend', 'inc/eshkol/core', 'inc/eshkol/backend', 'inc/eshkol/frontend')
               for p in (root/folder).rglob('*') if p.suffix in {'.cpp', '.c', '.h'}}
    ast, switches = ast_routes((root/'inc/eshkol/eshkol.h').read_text(), sources)
    calls, consumers = callable_consumers({p:s for p,s in sources.items() if p.startswith('lib/backend/')})
    return {'status': 'FAIL' if ast or calls else 'PASS', 'findings': ast+calls,
            'ast_sites': switches, 'callable_sites': consumers,
            'scope': 'Frontend/backend C/C++ sources and headers, core routing headers; complete operation and policy domains'}


def self_test():
    h = 'typedef enum { ESHKOL_A_OP, ESHKOL_B_OP } eshkol_op_t;'
    good = 'void walk() { switch (op) { case ESHKOL_A_OP: break; case ESHKOL_B_OP: break; } }'
    if ast_routes(h, {'inc/eshkol/core/ast_routing.h':good})[0]: raise AssertionError('clean exhaustive switch rejected')
    for bad in [good.replace('case ESHKOL_B_OP:', ''), good.replace('case ESHKOL_B_OP:', 'default:'),
                good.replace('case ESHKOL_B_OP:', 'switch (x) { case ESHKOL_B_OP: break; }')]:
        if not ast_routes(h, {'inc/eshkol/core/ast_routing.h':bad})[0]: raise AssertionError('AST fault survived')
    if not ast_routes(h.replace('ESHKOL_B_OP', 'ESHKOL_B_OP, ESHKOL_C_OP'), {'inc/eshkol/core/ast_routing.h':good})[0]:
        raise AssertionError('new enum member survived')
    routed = """void walk() {
      enum class AstRoute { A, B };
      switch (eshkol::routeAstOperation(op,
        eshkol::AstRouteGroup<AstRoute::A, ESHKOL_A_OP>{},
        eshkol::AstRouteGroup<AstRoute::B, ESHKOL_B_OP>{})) {
        case AstRoute::A: break;
        case AstRoute::B: break;
      }
    }"""
    domain = {'inc/eshkol/core/ast_routing.h':good, 'consumer.cpp':routed}
    if ast_routes(h, domain)[0]: raise AssertionError('complete routing policy rejected')
    mutations = {
        'missing-policy-member': routed.replace('ESHKOL_B_OP', ''),
        'duplicate-policy-member': routed.replace('ESHKOL_B_OP', 'ESHKOL_A_OP'),
        'unknown-policy-member': routed.replace('ESHKOL_B_OP', 'ESHKOL_C_OP'),
        'missing-policy-arm': routed.replace('case AstRoute::B:', ''),
        'default-policy-arm': routed.replace('case AstRoute::B:', 'default:'),
        'nested-policy-arm': routed.replace('case AstRoute::B:', 'switch (x) { case AstRoute::B: break; }'),
        'duplicate-policy-route': routed.replace('AstRouteGroup<AstRoute::B', 'AstRouteGroup<AstRoute::A'),
    }
    for name, mutated in mutations.items():
        if not ast_routes(h, {**domain, 'consumer.cpp':mutated})[0]:
            raise AssertionError(name + ' survived')
    if not ast_routes(h, {**domain, 'rogue.cpp':good})[0]:
        raise AssertionError('raw operation routing bypass survived')
    canonical = 'void codegenClosureCall() { auto closure_ptr = b.CreateLoad(x); b.CreateCall(ft, fp, args); }'
    if callable_consumers({'lib/backend/llvm_codegen.cpp':canonical})[0]: raise AssertionError('canonical rejected')
    if not callable_consumers({'lib/backend/llvm_codegen.cpp':canonical, 'new.cpp':canonical.replace('codegenClosureCall', 'rogue')})[0]:
        raise AssertionError('new callable consumer survived')
    if not callable_consumers({'new.cpp':canonical})[0]: raise AssertionError('duplicate dispatcher name exempted')
    if not callable_consumers({})[0]: raise AssertionError('missing dispatcher passed')
    return {'status':'PASS', 'controls':['omitted-case','default','nested-case','new-enum','new-consumer','missing-dispatcher','raw-routing-bypass', *mutations]}


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
