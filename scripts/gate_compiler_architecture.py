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


def token_signature(text):
    """Whitespace-insensitive C++ token signature for a declaration."""
    return tuple(re.findall(r'::|[A-Za-z_]\w*|\d+|[^\s]', text))


def top_level_char(text, target):
    """Find target outside templates, parentheses, brackets, and braces."""
    angle = paren = bracket = brace = 0
    for index, ch in enumerate(text):
        if ch == '<': angle += 1
        elif ch == '>' and angle: angle -= 1
        elif ch == '(': paren += 1
        elif ch == ')' and paren: paren -= 1
        elif ch == '[': bracket += 1
        elif ch == ']' and bracket: bracket -= 1
        elif ch == '{': brace += 1
        elif ch == '}' and brace: brace -= 1
        elif ch == target and not (angle or paren or bracket or brace): return index
    return -1


def split_top_level(text, target):
    parts, start = [], 0
    while True:
        index = top_level_char(text[start:], target)
        if index < 0:
            parts.append(text[start:].strip())
            return parts
        index += start
        parts.append(text[start:index].strip())
        start = index + 1


def class_definition(clean, name):
    """Return the body of one class definition, rejecting absent/ambiguous hits."""
    matches = list(re.finditer(r'\bclass\s+' + re.escape(name) + r'\b[^;{}]*\{', clean))
    if len(matches) != 1:
        raise ValueError(f'expected one definition of class {name}, found {len(matches)}')
    start = clean.index('{', matches[0].start())
    stop = endbrace(clean, start)
    return clean[start+1:stop-1]


def _member_declaration(statement):
    """Classify one semicolon-terminated top-level declaration.

    Returns a data-member signature or None for functions/type aliases/access
    labels. Unknown declaration shapes fail closed instead of silently
    disappearing from the inventory.
    """
    statement = re.sub(r'^\s*(?:public|private|protected)\s*:\s*', '', statement).strip()
    if not statement:
        return None
    if re.match(r'^(?:friend|using|typedef|static_assert)\b', statement):
        return None
    if re.match(r'^(?:struct|class|union|enum)\b', statement):
        return None
    # Static data are intentionally outside the non-static object layout.
    if re.match(r'^static\b', statement):
        return None
    # A top-level parameter list marks a member function/operator. Parentheses
    # in a default initializer are ignored because the initializer is removed.
    equals = top_level_char(statement, '=')
    declaration = statement[:equals].strip() if equals >= 0 else statement
    if re.search(r'\bstatic\b', declaration):
        return None
    # Parentheses in template arguments are nested and do not turn a field
    # into a function declaration.
    angle = 0
    has_top_paren = False
    for ch in declaration:
        if ch == '<': angle += 1
        elif ch == '>' and angle: angle -= 1
        elif ch in '()' and not angle:
            has_top_paren = True
            break
    if has_top_paren:
        if re.search(r'\(\s*[*&]\s*[A-Za-z_]\w*\s*\)\s*\(', declaration):
            raise ValueError(f'cannot inventory function-pointer data member: {statement[:100]!r}')
        if re.search(r'\bdecltype\s*\(', declaration) and not re.search(
                r'\b[A-Za-z_]\w*\s*\(.*\)\s*(?:const|noexcept|override|final|&|&&|->|$)', declaration):
            raise ValueError(f'cannot inventory decltype data member: {statement[:100]!r}')
        return None
    # Keep bit-field widths: equal field types with different widths can alter
    # layout while leaving ordinary type/name inventories unchanged.
    bitfield = re.search(r'(?<!:):(?!:)', declaration)
    bit_width = declaration[bitfield.end():].strip() if bitfield else None
    if bitfield:
        declaration = declaration[:bitfield.start()].strip()
    declarators = split_top_level(declaration, ',')
    if bit_width is not None and len(declarators) != 1:
        raise ValueError(f'cannot inventory multi-declarator bit-field: {statement[:100]!r}')
    first = declarators[0]
    match = re.search(r'([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$', first)
    if not match:
        raise ValueError(f'cannot inventory class declaration: {statement[:100]!r}')
    name = match.group(1)
    suffix = match.group(2).strip()
    prefix = first[:match.start(1)].strip()
    if not prefix:
        raise ValueError(f'cannot establish type for data member {name}')
    signature = token_signature(prefix + suffix)
    if bit_width is not None:
        if not bit_width or re.search(r'(?<!:):(?!:)', bit_width):
            raise ValueError(f'cannot inventory bit-field width: {statement[:100]!r}')
        signature += (':bit-width',) + token_signature(bit_width)
    result = [(name, signature)]
    for declarator in declarators[1:]:
        later = re.fullmatch(r'([*&\s]*)([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)', declarator)
        if not later:
            raise ValueError(f'cannot inventory multi-declarator member: {statement[:100]!r}')
        later_prefix, later_name, later_suffix = later.groups()
        result.append((later_name, token_signature(prefix + later_prefix.strip() + later_suffix.strip())))
    return tuple(result)


def _has_top_level_parenthesis(statement):
    angle = 0
    for ch in statement:
        if ch == '<': angle += 1
        elif ch == '>' and angle: angle -= 1
        elif ch == '(' and not angle: return True
    return False


def class_data_layout(clean, name):
    """Inventory ordered non-static members, recursively including nested types.

    This deliberately uses the gate's balanced-brace scanner. Method bodies
    and member initializers are skipped as balanced regions, so tokens that
    look like fields inside them cannot enter the inventory.
    """
    body = class_definition(clean, name)

    def method_only_conditional_end(class_body, start, owner):
        """Permit class-scope conditionals only when they guard methods alone."""
        directive_re = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b[^\n]*', re.M)
        directives = list(directive_re.finditer(class_body, start))
        if not directives or class_body[directives[0].start():start].strip():
            raise ValueError(f'layout-level preprocessor directive in {owner}')
        depth = 0
        end = None
        for directive in directives:
            kind = directive.group(1)
            if kind in {'if', 'ifdef', 'ifndef'}:
                depth += 1
            elif kind == 'endif':
                depth -= 1
                if depth == 0:
                    end = directive.end()
                    break
                if depth < 0:
                    break
        if end is None or directives[0].group(1) not in {'if', 'ifdef', 'ifndef'}:
            raise ValueError(f'unmatched layout-level preprocessor directive in {owner}')
        block = class_body[start:end]
        block = re.sub(r'^\s*#\s*[^\n]*(?:\n|$)', '', block, flags=re.M)
        block = code(block)
        segment_start = 0
        i = 0
        while i < len(block):
            if block[i] == ';':
                statement = block[segment_start:i].strip()
                if statement:
                    item = _member_declaration(statement)
                    if item is not None or not _has_top_level_parenthesis(statement):
                        raise ValueError(f'conditional may change member layout in {owner}: {statement[:80]!r}')
                segment_start = i + 1
            elif block[i] == '{':
                prefix = block[segment_start:i].strip()
                if '(' in prefix and '=' not in prefix:
                    close_brace = endbrace(block, i)
                    segment_start = close_brace
                    i = close_brace
                    continue
                if '=' in prefix:
                    i = endbrace(block, i)
                    continue
                raise ValueError(f'conditional contains non-method brace in {owner}: {prefix[:80]!r}')
            i += 1
        trailing = block[segment_start:].strip()
        if trailing:
            raise ValueError(f'conditional contains unterminated declaration in {owner}: {trailing[:80]!r}')
        return end

    def inventory(class_body, owner):
        members, nested = [], []
        segment_start = 0
        i = 0
        while i < len(class_body):
            ch = class_body[i]
            line_start = class_body.rfind('\n', 0, i) + 1
            if ch == '#' and not class_body[line_start:i].strip():
                directive = re.match(r'#[ \t]*([A-Za-z_]\w*)', class_body[i:])
                if directive:
                    if directive.group(1) in {'if', 'ifdef', 'ifndef'}:
                        close = method_only_conditional_end(class_body, i, owner)
                        segment_start = close
                        i = close
                        continue
                    raise ValueError(f'layout-level preprocessor directive in {owner}: #{directive.group(1)}')
            if ch == ';':
                statement = class_body[segment_start:i].strip()
                # A nested definition is recorded at its opening brace and
                # its trailing semicolon is only a terminator.
                if statement:
                    enum_forward = re.fullmatch(
                        r'enum\s+(?:(?:class|struct)\s+)?([A-Za-z_]\w*)\s*(?::\s*([^;]+))?', statement)
                    if enum_forward:
                        enum_name, underlying = enum_forward.groups()
                        nested.append((enum_name, ((), (('__enum__', token_signature(
                            'enum ' + enum_name + (':' + underlying if underlying else '') + ';'))),)))
                        segment_start = i + 1
                        i += 1
                        continue
                    if re.match(r'^enum\b', statement):
                        raise ValueError(f'cannot inventory enum declaration in {owner}: {statement[:100]!r}')
                    item = _member_declaration(statement)
                    if item is not None:
                        members.extend(item)
                segment_start = i + 1
            elif ch == '{':
                prefix = class_body[segment_start:i].strip()
                # Enum bodies define value names rather than data members;
                # record their complete declaration shape for member types.
                enum_match = re.search(r'\benum\s+(?:(?:class|struct)\s+)?([A-Za-z_]\w*)\b[^{}]*$', prefix)
                if enum_match:
                    close = endbrace(class_body, i)
                    enum_body = class_body[i+1:close-1]
                    if re.search(r'^\s*#\s*[A-Za-z_]\w*', enum_body, flags=re.M):
                        raise ValueError(f'preprocessor directive in nested enum {owner}::{enum_match.group(1)}')
                    enum_name = enum_match.group(1)
                    nested.append((enum_name, ((), (('__enum__', token_signature(prefix + '{' + enum_body + '}')),))))
                    segment_start = close
                    i = close
                    continue
                nested_match = re.search(r'\b(struct|class|union)\s+([A-Za-z_]\w*)\b[^{}]*$', prefix)
                if nested_match:
                    nested_name = nested_match.group(2)
                    close = endbrace(class_body, i)
                    nested.append((nested_name, inventory(class_body[i+1:close-1], owner + '::' + nested_name)))
                    # Preserve the nested declaration until its semicolon so
                    # the ordinary terminator path can safely discard it.
                    segment_start = close
                    i = close
                    continue
                # A brace after '=' belongs to a default initializer (often a
                # lambda); a parameter list marks a method body. Both are
                # balanced and cannot contribute field-looking text.
                if '=' in prefix or '(' in prefix or re.search(r'\btry\s*$', prefix):
                    close = endbrace(class_body, i)
                    if '(' in prefix and '=' not in prefix:
                        segment_start = close
                    i = close
                    continue
                raise ValueError(f'cannot classify brace in {owner} declaration: {prefix[:100]!r}')
            i += 1
        trailing = class_body[segment_start:].strip()
        # Access labels at end of the body are valid and carry no member.
        trailing = re.sub(r'^(?:public|private|protected)\s*:\s*$', '', trailing).strip()
        if trailing:
            raise ValueError(f'unterminated declaration in {owner}: {trailing[:100]!r}')
        return tuple(members), tuple(nested)

    layout = inventory(body, name)

    def referenced_nested(nested_types, member_rows):
        by_name = dict(nested_types)
        referenced = set()
        pending = []
        for _, type_tokens in member_rows:
            pending.extend(token for token in type_tokens if token in by_name)
        while pending:
            nested_name = pending.pop()
            if nested_name in referenced:
                continue
            referenced.add(nested_name)
            nested_members, nested_children = by_name[nested_name]
            for _, type_tokens in nested_members:
                pending.extend(token for token in type_tokens if token in dict(nested_children))
        return tuple((nested_name, by_name[nested_name]) for nested_name in sorted(referenced))

    return layout[0], referenced_nested(layout[1], layout[0])


def duplicate_class_layout_findings(header, implementation):
    """Compare the duplicated codegen class's complete member inventories."""
    try:
        header_layout = class_data_layout(code(header), 'EshkolLLVMCodeGen')
        implementation_layout = class_data_layout(code(implementation), 'EshkolLLVMCodeGen')
    except ValueError as error:
        return [{'rule': 'codegen_class_layout_unparseable', 'detail': str(error)}]
    if header_layout != implementation_layout:
        first_member_difference = next((index for index, pair in enumerate(zip(header_layout[0], implementation_layout[0]))
                                        if pair[0] != pair[1]), min(len(header_layout[0]), len(implementation_layout[0])))
        header_member = header_layout[0][first_member_difference] if first_member_difference < len(header_layout[0]) else None
        implementation_member = implementation_layout[0][first_member_difference] if first_member_difference < len(implementation_layout[0]) else None
        header_nested = dict(header_layout[1])
        implementation_nested = dict(implementation_layout[1])
        nested_differences = [nested_name for nested_name in sorted(set(header_nested) | set(implementation_nested))
                              if header_nested.get(nested_name) != implementation_nested.get(nested_name)]
        return [{'rule': 'codegen_class_layout_mismatch',
                 'header_members': len(header_layout[0]),
                 'implementation_members': len(implementation_layout[0]),
                 'header_nested_types': [name for name, _ in header_layout[1]],
                 'implementation_nested_types': [name for name, _ in implementation_layout[1]],
                 'first_member_difference': {'index': first_member_difference,
                                             'header': header_member,
                                             'implementation': implementation_member},
                 'nested_layout_differences': nested_differences}]
    return []


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
    class_layout = duplicate_class_layout_findings(
        (root/'inc/eshkol/backend/llvm_codegen.h').read_text(),
        (root/'lib/backend/llvm_codegen.cpp').read_text())
    findings = ast + calls + class_layout
    return {'status': 'FAIL' if findings else 'PASS', 'findings': findings,
            'ast_sites': switches, 'callable_sites': consumers,
            'class_layout': {'class': 'EshkolLLVMCodeGen',
                             'header': 'inc/eshkol/backend/llvm_codegen.h',
                             'implementation': 'lib/backend/llvm_codegen.cpp',
                             'findings': class_layout},
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
    class_header = '''class EshkolLLVMCodeGen {
      struct Metadata { int first; long long second; void method() { int fake_in_method; } };
      enum class Mode : unsigned char { First = 1, Second = 2 };
      Metadata metadata;
      Mode mode;
      int first;
      long long second;
      unsigned flags : 1;
      int initialized = [] { int fake_in_initializer; return 1; }();
      void method() { int fake_in_method; }
      /* int fake_in_comment; */
    };'''
    class_impl = class_header.replace('class EshkolLLVMCodeGen', 'class EshkolLLVMCodeGen')
    if duplicate_class_layout_findings(class_header, class_impl):
        raise AssertionError('matching duplicated class layout rejected')
    class_mutations = {
        'missing-member': class_impl.replace('      long long second;\n', ''),
        # Same member count and representative ABI size; source order still matters.
        'reordered-member': class_impl.replace('      int first;\n      long long second;', '      long long second;\n      int first;'),
        'changed-member-type': class_impl.replace('      long long second;', '      double second;'),
        'nested-layout-drift': class_impl.replace('struct Metadata { int first; long long second;', 'struct Metadata { int first; double second;'),
        'bitfield-width-drift': class_impl.replace('unsigned flags : 1;', 'unsigned flags : 2;'),
        'nested-enum-underlying-drift': class_impl.replace('enum class Mode : unsigned char', 'enum class Mode : unsigned int'),
        'nested-enum-values-drift': class_impl.replace('Second = 2', 'Second = 3'),
    }
    for name, mutated in class_mutations.items():
        if not duplicate_class_layout_findings(class_header, mutated):
            raise AssertionError(name + ' survived class layout guard')
    ignored = class_impl.replace('int fake_in_method;', 'int fake_in_method; int fake_extra;').replace(
        'int fake_in_comment;', 'int fake_in_comment; int another_fake;')
    ignored = ignored.replace('void method() { int fake_in_method; }',
                              'void method() {\n#ifdef METHOD_ONLY\n int fake_in_method;\n#endif\n }')
    if duplicate_class_layout_findings(class_header, ignored):
        raise AssertionError('method/comment field-like text affected class inventory')
    conditional = class_impl.replace('      int first;', '#ifdef FIELD_VARIANT\n      int first;\n#endif')
    if not duplicate_class_layout_findings(class_header, conditional):
        raise AssertionError('layout-level preprocessor conditional passed')
    unparseable = class_impl.replace('int first;', 'mystery { int first; };', 1)
    if not duplicate_class_layout_findings(class_header, unparseable):
        raise AssertionError('unparseable class inventory passed')
    return {'status':'PASS', 'controls':['omitted-case','default','nested-case','new-enum','new-consumer','missing-dispatcher','raw-routing-bypass', *mutations, *class_mutations, 'fake-method-comment-fields-ignored', 'method-body-directives-ignored', 'layout-conditional-fails-closed', 'unparseable-layout-fails-closed']}


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
