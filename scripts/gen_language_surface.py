#!/usr/bin/env python3
"""gen_language_surface.py — extract the COMPLETE Eshkol language surface.

Ground truth for TOTAL-LANGUAGE exposure-engine coverage. Parses production
VM, LLVM dispatch, parser, and library sources directly and emits a single
machine-readable manifest describing every construct a program can invoke:

  * BUILTINS      — the VM name->(id,arity) registration table is the sole
                    core ID/arity authority. LLVM dispatch names remain separate
                    source evidence and make no ABI claim. Each entry records
                    VM registration and LLVM name evidence, and — where the source row is
                    annotated `/* mirrors: <public name> */` — the public
                    construct the row is one engine's private spelling of
                    (`mirrors`).
  * SPECIAL FORMS — the surface keywords the parser recognises as syntax rather
                    than a call (lib/frontend/parser.cpp get_operator_type plus
                    the directly-dispatched forms begin/define-library/delay/...),
                    keyed to their eshkol_op_t (inc/eshkol/eshkol.h).
  * PRELUDE       — VM bootstrap, root stdlib definitions and direct stdlib
                    imports, with recursive dependency exports labeled only
                    as closure-presence evidence unless re-exported.

Every entry is categorised by a risk/kind bucket that drives exposure-engine
prioritisation (numeric, list_pair, vector, string_char, control_flow,
higher_order, io_port, tensor_ad, macro_syntax, ffi, consciousness, vm_system,
predicate, hash, memory_region, misc).

Usage:
  python3 scripts/gen_language_surface.py [--out tests/coverage/language_surface.json]
"""

import argparse
import difflib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VM_C = os.path.join(REPO, "lib", "backend", "eshkol_vm.c")
LLVM_CPP = os.path.join(REPO, "lib", "backend", "llvm_codegen.cpp")
PARSER = os.path.join(REPO, "lib", "frontend", "parser.cpp")
ESHKOL_H = os.path.join(REPO, "inc", "eshkol", "eshkol.h")
AGENT_QUANTUM = os.path.join(REPO, "lib", "agent", "quantum.esk")
AGENT_PQC = os.path.join(REPO, "lib", "agent", "pqc.esk")

# Agent modules are part of the user-callable language surface even though they
# are not registered in the core VM/native builtin tables. Keep this list small
# and intentional: it is the Moonlab-backed contract exercised only on the
# ESHKOL_QUANTUM_ENABLED=ON lane. The source check in
# extract_agent_module_provides() below prevents a manifest entry from drifting
# away from its exported Scheme API.
QUANTUM_AGENT_BUILTINS = {
    "make-quantum-state": (AGENT_QUANTUM, 1, "ffi_system"),
    "apply-hadamard": (AGENT_QUANTUM, 2, "ffi_system"),
    "apply-pauli-x": (AGENT_QUANTUM, 2, "ffi_system"),
    "apply-pauli-y": (AGENT_QUANTUM, 2, "ffi_system"),
    "apply-pauli-z": (AGENT_QUANTUM, 2, "ffi_system"),
    "apply-cnot": (AGENT_QUANTUM, 3, "ffi_system"),
    "apply-rx": (AGENT_QUANTUM, 3, "ffi_system"),
    "apply-ry": (AGENT_QUANTUM, 3, "ffi_system"),
    "apply-rz": (AGENT_QUANTUM, 3, "ffi_system"),
    "measure": (AGENT_QUANTUM, 2, "ffi_system"),
    "expectation-z": (AGENT_QUANTUM, 2, "ffi_system"),
    "bell-chsh": (AGENT_QUANTUM, 1, "ffi_system"),
    "make-h2-hamiltonian": (AGENT_QUANTUM, 1, "ffi_system"),
    "make-lih-hamiltonian": (AGENT_QUANTUM, 1, "ffi_system"),
    "make-h2o-hamiltonian": (AGENT_QUANTUM, 0, "ffi_system"),
    "hamiltonian-exact-ground-energy": (AGENT_QUANTUM, 1, "tensor_ad"),
    "vqe-optimize": (AGENT_QUANTUM, 2, "tensor_ad"),
    "vqe-gradient": (AGENT_QUANTUM, 2, "tensor_ad"),
    "vqe-energy": (AGENT_QUANTUM, 2, "tensor_ad"),
    "mlkem-keygen": (AGENT_PQC, 1, "ffi_system"),
    "mlkem-encaps": (AGENT_PQC, 1, "ffi_system"),
    "mlkem-decaps": (AGENT_PQC, 2, "ffi_system"),
}

# A BuiltinDef row is `{"name", id, arity}` or, for the few ops whose real
# caller minimum differs from the opcode's operand count, the four-field
# `{"name", id, arity, min_arity}` (see BuiltinDef in eshkol_vm.c). The fourth
# field has to be OPTIONAL here rather than a second pattern: while this one
# required exactly three, adding `min_arity` to the `hash-ref` row stopped it
# matching at all, so hash-ref silently vanished from the VM table and the
# generated surface reported it as having no VM backend (vm count 727 -> 726).
# A row this parser cannot read is dropped in SILENCE, which is why the shape
# it accepts must track BuiltinDef itself. The optional fifth field controls
# variadic closure dispatch and does not change the opcode operand count.
BUILTIN_ROW = re.compile(
    r'\{\s*"([^"]+)"\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?(?:,\s*-?\d+\s*)?\}')

# A BUILTINS[] row may carry a trailing `/* mirrors: <public-name> */`
# annotation declaring that it is one engine's private spelling -- an arity
# split, or a lower-level handle form -- of a public construct the other engine
# reaches by a different vehicle (a codegen intrinsic, or a core module
# compiled into every program). See the `mirrors:` contract above BUILTINS[] in
# lib/backend/eshkol_vm.c. It is recorded on the manifest entry so the
# cross-surface gates can relate the two spellings explicitly instead of by
# name identity, which is what broke when the geometric fallback constructors
# were renamed to the `-handle` family.
MIRROR_ROW = re.compile(
    r'\{\s*"([^"]+)"\s*,\s*-?\d+\s*,\s*-?\d+\s*(?:,\s*-?\d+\s*){0,2}\}\s*,?'
    r'[ \t]*/\*\s*mirrors:\s*([^\s*]+)\s*\*/')


def _strip_c_comments(text):
    """Remove C/C++ comments while preserving quoted literals byte-for-byte.

    Dispatch extraction must not promote commented-out experiments into the
    public language manifest (for example the disabled `det` alias).  A small
    state machine avoids the usual regex bug of treating `//` inside a string
    literal as a comment.
    """
    out = []
    i = 0
    state = "code"
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if c == '"':
                state = "string"; out.append(c); i += 1
            elif c == "'":
                state = "char"; out.append(c); i += 1
            elif c == "/" and nxt == "/":
                state = "line"; out.extend("  "); i += 2
            elif c == "/" and nxt == "*":
                state = "block"; out.extend("  "); i += 2
            else:
                out.append(c); i += 1
        elif state in ("string", "char"):
            out.append(c); i += 1
            if c == "\\" and i < len(text):
                out.append(text[i]); i += 1
            elif (state == "string" and c == '"') or (state == "char" and c == "'"):
                state = "code"
        elif state == "line":
            if c == "\n": out.append(c); state = "code"
            else: out.append(" ")
            i += 1
        else:
            if c == "*" and nxt == "/": out.extend("  "); i += 2; state = "code"
            else: out.append("\n" if c == "\n" else " "); i += 1
    return "".join(out)


def _slice_table(path, start_marker, end_marker_after):
    """Return the source text of the BuiltinDef table body."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    i = text.index(start_marker)
    j = text.index(end_marker_after, i)
    return text[i:j]


def resolve_min_arity(arity, column):
    """The CALLER minimum a BuiltinDef row states, from its min_arity column.

    Mirrors vm_builtin_row_min_arity() in lib/backend/eshkol_vm.c — 0 means
    "the minimum IS arity", -1 means the native lowering is variadic and the
    row makes no claim, -2 means the documented minimum is genuinely zero.
    Keeping the manifest's number the CALLER's obligation rather than the
    opcode's operand count is the whole point: `arity` says `substring` takes
    three operands, and `(substring s 1)` is still a legal call.
    """
    if column == -1:
        return None
    if column == -2:
        return 0
    return column if column else arity


def extract_builtin_table(path):
    """Extract {name: {id, arity, min_arity}} from `static const BuiltinDef BUILTINS[]`."""
    body = _slice_table(path, "BuiltinDef BUILTINS[]", "\n};")
    out = {}
    for m in BUILTIN_ROW.finditer(body):
        name, nid, arity = m.group(1), int(m.group(2)), int(m.group(3))
        column = int(m.group(4)) if m.group(4) else 0
        if name == "NULL":
            continue
        # A name may appear with several ids (overloads / legacy+new). Keep all.
        rec = out.setdefault(name, {"ids": [], "arity": arity,
                                    "min_arity": resolve_min_arity(arity, column)})
        if nid not in rec["ids"]:
            rec["ids"].append(nid)
    return out


def extract_mirror_annotations(path):
    """Extract {row-name: public-name} from the `/* mirrors: … */` row comments.

    Read from the same table slice as the rows themselves, before any comment
    stripping, so an annotation cannot drift away from the row it annotates.
    A row that mirrors itself, or that is absent from the table, is a typo the
    manifest must not carry forward silently.
    """
    body = _slice_table(path, "BuiltinDef BUILTINS[]", "\n};")
    rows = {m.group(1) for m in BUILTIN_ROW.finditer(body) if m.group(1) != "NULL"}
    out = {}
    for m in MIRROR_ROW.finditer(body):
        name, target = m.group(1), m.group(2)
        if name not in rows:
            raise ValueError("mirrors: annotation on unknown row %r in %s"
                             % (name, os.path.relpath(path, REPO)))
        if target == name:
            raise ValueError("row %r in %s mirrors itself"
                             % (name, os.path.relpath(path, REPO)))
        out[name] = target
    return out


# Alternate spellings that dispatch to the exact same codegen path as another
# name in this same table (`if (func_name == "X" || func_name == "Y") ...`).
# These are one construct with two spellings, not two constructs: recorded as
# an `aliases` field on the canonical entry in build_manifest() rather than as
# a second `builtins` record, so builtins_total counts the construct once.
# Keyed alias -> canonical (BI-7: `dataloader-has-next` is the pre-existing
# spelling kept for compatibility; `dataloader-has-next?` is canonical --
# docs, PARITY.tsv and the implementation's own diagnostics all use the `?`
# form).
ALIASES = {
    "dataloader-has-next": "dataloader-has-next?",
}


def extract_llvm_dispatch():
    """Extract call-head builtin names dispatched by the LLVM AOT backend
    (`if (func_name == "name") ...`). This is the AOT intrinsic surface: it
    overlaps the native/VM id-tables and adds AOT-only intrinsics (the R7RS
    IO/mutation forms, the NN/optimizer/linalg surface, atomics, etc.)."""
    with open(LLVM_CPP, encoding="utf-8", errors="replace") as fh:
        text = _strip_c_comments(fh.read())
    names = set(re.findall(r'func_name\s*==\s*"([^"]+)"', text))
    # a few dispatch strings are internal markers, not user-callable builtins
    NOISE = {"target-intrinsic", "sum-tag", "sum-value", "features",
             "__arena-used"}
    return {n for n in names if n not in NOISE}


def extract_ast_ops():
    """Extract the eshkol_op_t enum member names."""
    with open(ESHKOL_H, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = re.search(r"typedef enum\s*\{(.*?)\}\s*eshkol_op_t;", text, re.S)
    body = m.group(1)
    return [x for x in re.findall(r"\bESHKOL_[A-Z0-9_]+_OP\b", body)]


def extract_special_forms():
    """Extract surface-keyword -> eshkol_op_t from the parser dispatch, plus the
    directly-dispatched syntactic forms not routed through get_operator_type."""
    with open(PARSER, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    forms = {}
    # get_operator_type body: `if (op == "kw") return ESHKOL_X_OP;`
    gm = re.search(r"get_operator_type\(const std::string& op\)\s*\{(.*?)\n\}",
                   text, re.S)
    for kw, op in re.findall(r'op == "([^"]+)"\)\s*return\s+(ESHKOL_[A-Z0-9_]+_OP)',
                             gm.group(1)):
        forms[kw] = op
    # Directly-dispatched syntactic forms (handled before/around get_operator_type).
    extra = {
        "begin": "ESHKOL_SEQUENCE_OP",
        "define-library": "ESHKOL_DEFINE_TYPE_OP",   # library form (parser-transformed)
        "delay": "ESHKOL_CALL_OP",                    # promise form (codegen intrinsic)
        "delay-force": "ESHKOL_CALL_OP",
        "make-promise": "ESHKOL_CALL_OP",
        "force": "ESHKOL_CALL_OP",
        "let-match": "ESHKOL_MATCH_OP",
        "define-values": "ESHKOL_DEFINE_OP",
        "named-let": "ESHKOL_LET_OP",                 # (let name (bindings) body)
    }
    for k, v in extra.items():
        forms.setdefault(k, v)
    return forms


# --- canonical stdlib module closure ---------------------------------------
STDLIB = os.path.join(REPO, "lib", "stdlib.esk")


def _scheme_tokens(text):
    """Tokenize the source forms needed by the stdlib surface reader.

    Strings are opaque tokens, and both line and nested block comments are
    skipped. This deliberately avoids regex scans that promote quoted text,
    comments, or a local define into a public top-level binding.
    """
    tokens = []
    i = 0
    line = 1
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if c.isspace():
            if c == "\n": line += 1
            i += 1
        elif c == ";":
            while i < len(text) and text[i] != "\n": i += 1
        elif c == "#" and nxt == ";":
            tokens.append(("datum-comment", "#;", line)); i += 2
        elif c == "#" and nxt == "|":
            i += 2
            depth = 1
            while i < len(text) and depth:
                if text.startswith("#|", i): depth += 1; i += 2
                elif text.startswith("|#", i): depth -= 1; i += 2
                else:
                    if text[i] == "\n": line += 1
                    i += 1
            if depth:
                raise ValueError("unterminated Scheme block comment")
        elif c == '"':
            start_line = line
            i += 1
            value = []
            while i < len(text) and text[i] != '"':
                if text[i] == "\\" and i + 1 < len(text):
                    value.extend((text[i], text[i + 1]))
                    if text[i + 1] == "\n": line += 1
                    i += 2
                else:
                    if text[i] == "\n": line += 1
                    value.append(text[i]); i += 1
            if i >= len(text):
                raise ValueError("unterminated Scheme string")
            i += 1
            tokens.append(("string", "".join(value), start_line))
        elif text.startswith("#\\", i):
            start_line = line
            start = i
            i += 2
            if i < len(text) and text[i] in "()'`,;\"":
                i += 1
            else:
                while i < len(text) and not text[i].isspace() and text[i] not in "()'`,;\"":
                    i += 1
            tokens.append(("symbol", text[start:i], start_line))
        elif c in "()":
            tokens.append((c, c, line)); i += 1
        elif c in "'`,":
            start_line = line
            if c == "," and nxt == "@":
                tokens.append(("prefix", ",@", line)); i += 2
            else:
                tokens.append(("prefix", c, line)); i += 1
        elif c == "|":
            start_line = line
            i += 1
            value = ["|"]
            closed = False
            while i < len(text):
                value.append(text[i])
                if text[i] == "\\" and i + 1 < len(text):
                    i += 1; value.append(text[i])
                elif text[i] == "|":
                    i += 1; closed = True; break
                i += 1
            if not closed:
                raise ValueError("unterminated escaped Scheme symbol at line %d" % start_line)
            tokens.append(("symbol", "".join(value), start_line))
        else:
            start = i
            start_line = line
            while i < len(text) and not text[i].isspace() and text[i] not in "()'`,;\"":
                if text.startswith("#|", i): break
                i += 1
            if i == start:
                tokens.append(("symbol", text[i], line)); i += 1
            else:
                tokens.append(("symbol", text[start:i], start_line))
    return tokens


def parse_scheme_forms(text):
    """Parse balanced Scheme list structure, retaining atom kinds and lines."""
    tokens = _scheme_tokens(text)
    cursor = 0

    def skip_datum_comments():
        nonlocal cursor
        while cursor < len(tokens) and tokens[cursor][0] == "datum-comment":
            _, _, line = tokens[cursor]
            cursor += 1
            # A nested #; is itself interlexeme trivia: it comments out its
            # datum, then the outer comment still consumes the next datum.
            skip_datum_comments()
            if cursor >= len(tokens) or tokens[cursor][0] == ")":
                raise ValueError("datum comment at line %d has no datum" % line)
            parse_raw()

    def parse_next():
        nonlocal cursor
        skip_datum_comments()
        if cursor >= len(tokens):
            return None
        return parse_raw()

    def parse_raw():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("unexpected end of Scheme source")
        kind, value, line = tokens[cursor]
        cursor += 1
        if kind == "(":
            result = []
            while True:
                skip_datum_comments()
                if cursor >= len(tokens):
                    raise ValueError("unclosed Scheme list at line %d" % line)
                if tokens[cursor][0] == ")":
                    break
                result.append(parse_raw())
            if cursor >= len(tokens):
                raise ValueError("unclosed Scheme list at line %d" % line)
            cursor += 1
            return ("list", result, line)
        if kind == ")":
            raise ValueError("unexpected ) at line %d" % line)
        if kind == "prefix":
            prefix = {"'": "quote", "`": "quasiquote", ",": "unquote",
                      ",@": "unquote-splicing"}[value]
            datum = parse_next()
            if datum is None:
                raise ValueError("prefix %s at line %d has no datum" % (value, line))
            return ("list", [("symbol", prefix, line), datum], line)
        return (kind, value, line)

    forms = []
    while cursor < len(tokens):
        form = parse_next()
        if form is not None:
            forms.append(form)
    return forms


def _scheme_list(form):
    return form[1] if isinstance(form, tuple) and form[0] == "list" else None


def _scheme_symbol(form):
    return form[1] if isinstance(form, tuple) and form[0] == "symbol" else None


def _top_level_definition(form):
    items = _scheme_list(form)
    if not items or _scheme_symbol(items[0]) not in {
            "define", "define-syntax", "define-macro", "define-values"}:
        return []
    if len(items) < 2:
        return []
    if _scheme_symbol(items[0]) == "define-values":
        target = _scheme_list(items[1])
        return [(name, items[1][2]) for name in
                (_scheme_symbol(x) for x in target or []) if name]
    target = _scheme_list(items[1])
    if target:
        name = _scheme_symbol(target[0])
        return [(name, form[2])] if name else []
    name = _scheme_symbol(items[1])
    return [(name, items[1][2])] if name else []


_NUMBER_LITERAL = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)


def _quoted_constant(form):
    """Whether a parsed datum contains only inert quoted literal data."""
    if not isinstance(form, tuple):
        return False
    kind, value = form[0], form[1]
    if kind == "string":
        return True
    if kind == "symbol":
        return (value in {"#t", "#f"} or value.startswith("#\\") or
                bool(_NUMBER_LITERAL.fullmatch(value)))
    if kind != "list":
        return False
    items = value
    # The parser represents quote as a two-item list. Nested quote is inert too.
    if len(items) == 2 and _scheme_symbol(items[0]) == "quote":
        return _quoted_constant(items[1])
    return all(_quoted_constant(item) for item in items)


def _binding_expression_category(rhs):
    """Classify only source forms whose binding kind is directly provable."""
    items = _scheme_list(rhs)
    if items and _scheme_symbol(items[0]) == "lambda" and len(items) >= 3:
        return "procedure"
    if (items and _scheme_symbol(items[0]) == "quote" and len(items) == 2
            and _quoted_constant(items[1])):
        return "value"
    if isinstance(rhs, tuple) and rhs[0] == "string":
        return "value"
    symbol = _scheme_symbol(rhs)
    if symbol is not None and (symbol in {"#t", "#f"} or
                               symbol.startswith("#\\") or
                               _NUMBER_LITERAL.fullmatch(symbol)):
        return "value"
    return "unknown"


def _top_level_binding(form):
    """Return (name, line, category, alias-target) for a simple top-level binding."""
    items = _scheme_list(form)
    if not items or len(items) < 2:
        return []
    kind = _scheme_symbol(items[0])
    target = _scheme_list(items[1])
    if kind in {"define-syntax", "define-macro"}:
        name = _scheme_symbol(items[1]) or (_scheme_symbol(target[0]) if target else None)
        return [(name, form[2], "syntax", None)] if name else []
    if kind == "define-values":
        return [(name, items[1][2], "unknown", None)
                for name in (_scheme_symbol(x) for x in target or []) if name]
    if kind != "define":
        return []
    if target:
        name = _scheme_symbol(target[0])
        return [(name, form[2], "procedure", None)] if name else []
    name = _scheme_symbol(items[1])
    if not name or len(items) < 3:
        return []
    rhs = items[2]
    alias = _scheme_symbol(rhs)
    return [(name, items[1][2], _binding_expression_category(rhs), alias)]


def _assigned_names(forms):
    """Find set! targets anywhere in parsed code, including nested closures."""
    assigned = set()

    def visit(node):
        items = _scheme_list(node)
        if items is None:
            return
        if items and _scheme_symbol(items[0]) == "quote":
            return
        if (len(items) >= 2 and _scheme_symbol(items[0]) == "set!"
                and _scheme_symbol(items[1])):
            assigned.add(_scheme_symbol(items[1]))
        for item in items:
            visit(item)

    for form in forms:
        visit(form)
    return assigned


def _top_level_exports(form):
    items = _scheme_list(form)
    if not items or _scheme_symbol(items[0]) != "provide":
        return []
    names = []
    for item in items[1:]:
        parts = _scheme_list(item)
        if parts:
            raise ValueError("nested provide specifier at line %d is unsupported "
                             "by the production stdlib surface reader" % item[2])
        name = _scheme_symbol(item)
        if not name:
            raise ValueError("non-symbol provide entry at line %d is unsupported" % item[2])
        names.append((name, item[2]))
    return names


def _require_module_names(form):
    items = _scheme_list(form)
    if not items or _scheme_symbol(items[0]) != "require":
        return []
    names = []

    for item in items[1:]:
        symbol = _scheme_symbol(item)
        if not symbol:
            raise ValueError("non-symbol require specifier at line %d is unsupported "
                             "by the production stdlib surface reader" % item[2])
        names.append((symbol, item[2]))
    return names


def extract_stdlib_surface(repo=REPO):
    """Return public names and exact top-level provenance for stdlib closure.

    The native stdlib surface is `lib/stdlib.esk`'s top-level definitions plus
    exports from modules it directly requires. Recursively required module
    exports are retained as closure-presence evidence, but are not promoted to
    root stdlib availability unless a directly required module re-exports them.
    Module-local definitions remain provenance, not public names. The current
    production closure uses simple symbol `require` and flat symbol `provide`
    forms; import modifiers and export renaming fail closed until their
    visibility mapping is modeled explicitly.
    """
    lib_root = os.path.join(repo, "lib")
    root_path = os.path.join(lib_root, "stdlib.esk")
    modules = {}
    requiring = {}

    def resolve(module):
        if module == "stdlib": return root_path
        relative = module.replace(".", os.sep) + ".esk"
        return os.path.join(lib_root, relative)

    pending = [("stdlib", root_path, "<entry>")]
    while pending:
        module, path, parent = pending.pop()
        if module in modules:
            requiring[module].add(parent)
            continue
        if not os.path.isfile(path):
            raise ValueError("stdlib require %r from %s has no source at %s" %
                             (module, parent, os.path.relpath(path, repo)))
        with open(path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        try:
            forms = parse_scheme_forms(source)
        except ValueError as exc:
            raise ValueError("cannot read production module %s: %s" %
                             (os.path.relpath(path, repo), exc)) from exc
        defs = {}
        binding_defs = {}
        exports = {}
        requires = []
        try:
            for form in forms:
                for name, line in _top_level_definition(form):
                    defs.setdefault(name, line)
                for name, line, category, alias in _top_level_binding(form):
                    binding_defs.setdefault(name, []).append({
                        "line": line, "category": category, "alias": alias})
                for name, line in _top_level_exports(form):
                    exports.setdefault(name, line)
                requires.extend(_require_module_names(form))
        except ValueError as exc:
            raise ValueError("unsupported stdlib surface in %s: %s" %
                             (os.path.relpath(path, repo), exc)) from exc
        modules[module] = {"path": os.path.relpath(path, repo),
                           "definitions": defs, "binding_definitions": binding_defs,
                           "assigned_names": _assigned_names(forms),
                           "exports": exports,
                           "requires": []}
        requiring.setdefault(module, set()).add(parent)
        for required, line in requires:
            child_path = resolve(required)
            modules[module]["requires"].append({"module": required, "line": line})
            requiring.setdefault(required, set()).add(module)
            if required not in modules:
                pending.append((required, child_path, module))

    # Import mutation/provenance is not modeled. Treat a same-spelled write in
    # any reachable module as potentially affecting every exported binding.
    assigned_names = set().union(*(record["assigned_names"]
                                   for record in modules.values()))

    def resolve_binding(module, name, seen=None):
        """Resolve a public name to a unique definition and conservative kind."""
        seen = set() if seen is None else set(seen)
        key = (module, name)
        if key in seen:
            return None
        seen.add(key)
        record = modules[module]
        definitions = record["binding_definitions"].get(name, [])
        if definitions:
            # Multiple same-spelled definitions may be a deliberate redefinition;
            # without expansion/evaluation order guarantees it is unsafe to label.
            if len(definitions) != 1:
                return None
            definition = definitions[0]
            if name in assigned_names:
                return {
                    "module": module, "path": record["path"], "name": name,
                    "line": definition["line"], "chain": [module],
                    "category": "unknown",
                }
            alias = definition["alias"]
            if alias and alias != name:
                candidates = []
                if alias in record["binding_definitions"]:
                    candidates.append((module, alias))
                else:
                    candidates = [(required["module"], alias)
                                  for required in record["requires"]
                                  if required["module"] in modules and
                                  alias in modules[required["module"]]["exports"]]
                if len(candidates) == 1:
                    target_module, target_name = candidates[0]
                    target = resolve_binding(target_module, target_name, seen)
                    if target:
                        return {**target,
                                "chain": [module] + [m for m in target["chain"]
                                                     if m != module]}
                    return None
                if len(candidates) > 1:
                    return None
            return {
                "module": module, "path": record["path"], "name": name,
                "line": definition["line"], "chain": [module],
                "category": definition["category"],
            }
        if name in assigned_names:
            return None
        candidates = [required["module"] for required in record["requires"]
                      if required["module"] in modules and
                      name in modules[required["module"]]["exports"]]
        if len(candidates) != 1:
            return None
        child = candidates[0]
        target = resolve_binding(child, name, seen)
        if not target:
            return None
        return {**target, "chain": [module] + [m for m in target["chain"]
                                              if m != module]}

    def definition_origin(module, name, seen=None):
        """Keep the original source-provenance behavior for compatibility."""
        seen = set() if seen is None else set(seen)
        if module in seen:
            return None
        seen.add(module)
        record = modules[module]
        if name in record["definitions"]:
            return {"module": module, "path": record["path"],
                    "line": record["definitions"][name], "chain": [module]}
        candidates = [required["module"] for required in record["requires"]
                      if required["module"] in modules and
                      name in modules[required["module"]]["exports"]]
        if len(candidates) > 1:
            raise ValueError("ambiguous re-export %r in %s from %s" %
                             (name, module, ", ".join(candidates)))
        if candidates:
            child = candidates[0]
            origin = definition_origin(child, name, seen)
            if origin:
                return {**origin, "chain": [module] + origin["chain"]}
        return None

    root_requires = {item["module"] for item in modules["stdlib"]["requires"]}

    def binding_metadata(public_name, origin):
        if not origin:
            return {"public_name": public_name, "binding_category": "unknown",
                    "resolved_binding_origin": None}
        identity = "%s:%s#%s" % (origin["module"], origin["path"], origin["name"])
        return {
            "public_name": public_name,
            "binding_category": origin["category"],
            "binding_identity": identity,
            "resolved_binding_origin": {key: origin[key] for key in
                                         ("module", "path", "name", "line", "chain")},
        }

    names = {}
    # Direct definitions in the root stdlib source are prelude definitions.
    for name, line in modules["stdlib"]["definitions"].items():
        evidence = {"kind": "stdlib_root_definition", "module": "stdlib",
                    "path": modules["stdlib"]["path"], "line": line,
                    "definition_origin": definition_origin("stdlib", name)}
        evidence.update(binding_metadata(name, resolve_binding("stdlib", name)))
        names.setdefault(name, []).append(evidence)
    closure_names = {}
    for module, record in modules.items():
        if module == "stdlib": continue
        for name, line in record["exports"].items():
            origin = definition_origin(module, name)
            evidence = {
                "kind": "module_export", "module": module,
                "path": record["path"], "line": line,
                "defined_at_top_level": name in record["definitions"],
                "definition_line": record["definitions"].get(name),
                "definition_origin": origin,
                "visibility": ("stdlib_root_required_export" if module in root_requires
                               else "closure_export_presence"),
            }
            evidence.update(binding_metadata(name, resolve_binding(module, name)))
            closure_names.setdefault(name, []).append(evidence)
            if module in root_requires:
                names.setdefault(name, []).append(evidence)

    closure = []
    for module, record in sorted(modules.items()):
        closure.append({
            "module": module,
            "path": record["path"],
            "required_by": sorted(requiring[module]),
            "stdlib_visibility": ("stdlib_root_required_module" if module in root_requires
                                 else "closure_presence_only" if module != "stdlib"
                                 else "stdlib_root"),
            "requires": record["requires"],
            "exports": [{"name": name, "line": line,
                         "definition_line": record["definitions"].get(name),
                         "definition_origin": definition_origin(module, name)}
                        for name, line in sorted(record["exports"].items())],
            "top_level_definitions": [{"name": name, "line": line}
                                      for name, line in sorted(record["definitions"].items())],
        })
    return {"names": names, "closure_names": closure_names,
            "root_requires": sorted(root_requires), "modules": closure}


# --- the bytecode VM's own prelude (lib/backend/vm_prelude_source.h) --------
VM_PRELUDE_H = os.path.join(REPO, "lib", "backend", "vm_prelude_source.h")


def extract_vm_prelude_defs():
    """Top-level names the VM prelude defines -- the VM's second vehicle.

    The VM reaches some public names only through Scheme definitions compiled
    into every VM program (`write`, `newline`, `swish`, the standard port
    parameters, ...). A name defined there IS on the VM, so it counts as a VM
    backend for a construct already on the surface; the prelude never adds a
    construct of its own.
    """
    names = set()
    with open(VM_PRELUDE_H, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    for literal in re.findall(r'"((?:[^"\\]|\\.)*)"', text):
        for name in re.findall(r'^\(define \(?([^\s()]+)', literal):
            if not name.startswith("_"):
                names.add(name)
    return names


def extract_agent_module_provides(path):
    """Extract the flat public API from an agent module's `(provide ...)` form.

    Agent modules are compiled into programs that require them, rather than
    registered in the core BUILTINS[] tables. This is deliberately a narrow
    parser for the flat provide lists used by lib/agent/*.esk, not a general
    Scheme parser.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    match = re.search(r"\(provide\s+([^)]*)\)", text, re.S)
    if not match:
        raise ValueError("no flat (provide ...) form in %s" % path)
    return set(re.findall(r"[A-Za-z][A-Za-z0-9!?*/<>=._+-]*", match.group(1)))


# ---------------------------------------------------------------------------
# categoriser — risk/kind bucket per construct (drives prioritisation)
# ---------------------------------------------------------------------------
GEOMETRY_PREFIXES = ("manifold", "riemannian", "geodesic", "spherical",
                     "hyperbolic", "mobius", "se3-", "so3-", "poincare",
                     "curvature", "christoffel", "ricci", "riemann",
                     "quaternion")
GEOMETRY_NAMES = {"hodge-star", "wedge-product", "frechet-mean",
                  "great-circle-distance", "exterior-derivative",
                  "interior-product", "metric-tensor", "pullback",
                  "pullback", "sectional-curvature", "transition-geometry!",
                  "parallel-transport", "geodesic-distance", "slerp",
                  "adaptive-curvature-step", "get-curvature", "set-curvature!",
                  "retraction", "make-euclidean-manifold",
                  "make-product-manifold", "make-riemannian-adam-state"}


STRAGGLERS = {
    "digit-value": "string_char", "matrix-to-string": "string_char",
    "split": "string_char", "procedure-arity": "misc_core",
    "inject-left": "misc_core", "inject-right": "misc_core",
    "%make-lazy-promise": "control_flow",
    "%make-lazy-promise-force": "control_flow",
}


def categorize(name):
    if name in STRAGGLERS:
        return STRAGGLERS[name]
    n = name.lstrip("_")
    # AD-tape / autodiff primitives
    if n.startswith("ad-") or n in {"reverse-gradient"}:
        return "tensor_ad"
    # differential geometry (silent-wrong-prone numeric — own bucket)
    if (n.startswith(GEOMETRY_PREFIXES) or n in GEOMETRY_NAMES
            or n.endswith("-manifold")):
        return "geometry"
    if n in {"batch-matmul", "matmul"}:
        return "tensor_ad"
    if n in {"vref"}:
        return "vector"
    if n in {"current-error-port"}:
        return "io_port"
    if n == "make-hash-table" or n.startswith("hash"):
        return "hash"
    if n.startswith("unix-socket"):
        return "ffi_system"
    # ESH-0011 event loop — same family as make-pipe / fd-write / fd-close
    # below: descriptor plumbing over a kernel multiplexer.
    if n == "make-event-loop" or n.startswith("event-loop-"):
        return "ffi_system"
    # gpu compute
    if n.startswith("gpu-"):
        return "tensor_ad"
    # parallel higher-order
    if n.startswith("parallel-"):
        return "higher_order"
    # neural-net / optimizer / tensor-ml surface
    if (n.endswith("-loss") or n.endswith("-step") or n.endswith("-lr")
            or n.endswith("-pool2d") or n.startswith("dataloader")
            or n.startswith("conv") or n.startswith("dequant")
            or n in {"gelu", "elu", "celu", "selu", "silu", "mish", "softplus",
                     "leaky-relu", "hard-sigmoid", "hard-swish", "relu", "sigmoid",
                     "embedding", "feed-forward", "multi-head-attention",
                     "positional-encoding", "rotary-embedding", "causal-mask",
                     "padding-mask", "dropout", "check-grad-health",
                     "clip-grad-norm!", "zero-grad!", "make-dataloader",
                     "train-test-split", "onnx-export-tensor", "concatenate",
                     "stack", "tile", "squeeze", "unsqueeze", "slice", "pad",
                     "outer", "einsum", "norm", "batch-matmul", "avg-pool2d",
                     "max-pool2d", "fft", "ifft", "det"}
            or n.startswith("xavier") or n.startswith("kaiming")
            or n.startswith("lecun")):
        return "tensor_ad"
    # atomics / low-level memory / FFI pointers
    if (n.startswith("atomic-") or n.startswith("volatile-")
            or n.startswith("ptr") or n.endswith("->ptr")
            or n in {"addr-of", "null-ptr", "memory-fence", "compiler-fence",
                     "usize->ptr", "ptr-add"}):
        return "ffi_system"
    # rng (nondeterministic surface)
    if (n.startswith("prng") or n.startswith("make-prng")
            or n in {"rand", "randint", "randn", "random", "set-random-seed!",
                     "quantum-random", "quantum-random-int", "quantum-random-range"}):
        return "ffi_system"
    # eval / environments (meta)
    if n in {"eval", "interaction-environment", "current-environment",
             "null-environment", "scheme-report-environment"}:
        return "misc_core"
    # extended numeric tower (AOT)
    if n in {"acosh", "asinh", "atanh", "atan2", "cbrt", "ceil", "exp2", "log2",
             "log10", "mod", "floor/", "floor-quotient", "floor-remainder",
             "truncate/", "truncate-quotient", "truncate-remainder", "trunc",
             "square", "/rational", "%", "make-rational", "fabs", "abs"}:
        return "numeric"
    # macro / syntax
    if n in {"quote", "quasiquote", "unquote", "unquote-splicing", "define-syntax",
             "let-syntax", "letrec-syntax", "syntax-error", "syntax-rules"}:
        return "macro_syntax"
    # control flow / special-form control
    if n in {"if", "cond", "case", "when", "unless", "and", "or", "do", "begin",
             "guard", "raise", "call/cc", "call-with-current-continuation",
             "dynamic-wind", "values", "call-with-values", "let-values",
             "let*-values", "delay", "delay-force", "force", "make-promise",
             "match", "let-match", "when", "not", "with-exception-handler",
             "raise-continuable"}:
        return "control_flow"
    # binding forms
    if n in {"define", "lambda", "let", "let*", "letrec", "letrec*", "set!",
             "named-let", "define-values", "define-record-type", "case-lambda",
             "parameterize", "make-parameter", "define-type"}:
        return "binding_form"
    # module / library
    if n in {"import", "require", "provide", "load", "include", "include-ci",
             "define-library", "cond-expand"}:
        return "module"
    # memory / region (OALR + the #341 user-reachable region handles)
    if n in {"with-region", "owned", "move", "borrow", "shared", "weak-ref",
             "region-open", "region-close", "region-open?",
             "_region-open", "_region-close-list"}:
        return "memory_region"
    # higher-order
    if n in {"map", "filter", "fold-left", "fold-right", "for-each", "reduce",
             "compose", "apply", "any", "every", "find", "sort", "vector-map",
             "vector-for-each", "count", "merge"}:
        return "higher_order"
    # tensor / AD
    if (n.startswith("tensor") or n.startswith("dual") or n.startswith("dnc")
            or n.startswith("sdnc") or n in {
            "make-tensor", "reshape", "transpose", "flatten", "matmul", "conv2d",
            "softmax", "relu", "sigmoid", "dropout", "zeros", "ones", "arange",
            "linspace", "eye", "matrix", "gradient", "jacobian", "hessian",
            "derivative", "derivative-n", "taylor", "divergence", "curl",
            "laplacian", "directional-derivative", "diff", "differentiate", "D",
            "make-dual", "dual?", "mse-loss", "cross-entropy-loss", "batch-norm",
            "layer-norm", "scaled-dot-attention", "tensor-matmul", "tensor-scale",
            "tensor-add", "tensor-mul", "model-save", "model-load"}):
        return "tensor_ad"
    # consciousness engine (logic/inference/workspace)
    if (n.startswith("kb") or n.startswith("fg-") or n.startswith("ws-")
            or n.startswith("make-factor") or n.startswith("make-workspace")
            or n in {"unify", "walk", "make-substitution", "substitution?",
                     "make-fact", "fact?", "make-kb", "kb?", "logic-var?",
                     "factor-graph?", "workspace?", "free-energy",
                     "expected-free-energy", "forall"}):
        return "consciousness"
    # hash tables
    if n.startswith("hash"):
        return "hash"
    # io / port
    if (n.startswith("open-") or n.startswith("read") or n.startswith("write")
            or n.startswith("close") or n.startswith("peek")
            or n.startswith("call-with-input") or n.startswith("call-with-output")
            or n.startswith("call-with-port") or n.startswith("with-input-from")
            or n.startswith("with-output-to") or n.startswith("flush-output")
            or n in {"display", "newline", "eof-object?", "eof-object",
                     "get-output-string", "file-exists?", "flush-output",
                     "current-output-port", "current-input-port", "port?",
                     "input-port?", "output-port?", "with-output-to-string",
                     "call-with-output-string", "display-error", "format"}):
        return "io_port"
    # ffi / system / network / process / regex / time / compression
    if (n.startswith("http") or n.startswith("websocket") or n.startswith("ts-")
            or n.startswith("process") or n.startswith("socket")
            or n.startswith("signal") or n.startswith("dl") or n.startswith("regex")
            or n.startswith("url-") or n.startswith("base64") or n.startswith("semver")
            or n.startswith("term-") or n.startswith("fs-") or n.startswith("db-")
            or n.startswith("yoga") or n.startswith("lru-") or n.startswith("channel")
            or n.startswith("mutex") or n.startswith("condition") or n.startswith("condvar")
            or n.startswith("timer") or n.startswith("interval") or n.startswith("fd-")
            or n.startswith("line-reader") or n.startswith("make-line")
            or n.startswith("json") or n.startswith("compression")
            or n.startswith("file-") or n.startswith("directory")
            or n.startswith("path-") or n.startswith("glob") or n.startswith("image-")
            or n.startswith("symlink") or n.startswith("shell-") or n.startswith("os-")
            or n.startswith("mkdir") or n.startswith("mkstemp") or n.startswith("mkdtemp")
            or n.startswith("thread-pool") or n.startswith("future")
            or n.startswith("quantum-random")
            or n.startswith("current-second") or n.startswith("current-jiffy")
            or n.startswith("current-time") or n.startswith("dequant")
            or n in {"exit", "emergency-exit", "system", "sleep", "time", "trace",
                     "current-seconds", "jiffies-per-second", "poll-fd",
                     "delete-directory", "make-directory", "append-file"}
            or n in {"getpid", "hostname", "username", "sleep-ms", "cpu-count",
                     "current-directory", "set-current-directory!", "home-directory",
                     "command-line", "current-time-ms", "realpath", "delete-file",
                     "force-future", "with-mutex", "make-condition-variable",
                     "make-condvar", "make-timer", "make-interval",
                     "directory-delete-recursive", "directory-entries",
                     "directory-walk", "mkdir-recursive", "utf8->string"}
            or n in {"extern", "extern-var", "fork", "execv", "setenv", "unsetenv",
                     "getenv", "get-environment-variable", "uuid-v4", "sha256-file",
                     "monotonic-time-ms", "executable-path", "temp-directory",
                     "prevent-sleep", "allow-sleep", "make-temp-file", "make-temp-dir",
                     "deflate", "inflate", "gzip", "gunzip", "at-exit", "make-pipe",
                     "make-channel", "make-mutex", "make-event-emitter", "on!", "once!",
                     "off!", "poll", "io-poll", "constant-time-equal?", "make-lru-cache",
                     "current-timestamp", "current-time-ns", "format-iso8601",
                     "parse-iso8601", "format-relative", "local-timezone-offset",
                     "diff-lines", "fuzzy-match", "ansi-strip", "string-display-width"}):
        return "ffi_system"
    # complex / rational (numeric tower non-real)
    if n in {"make-rectangular", "make-polar", "real-part", "imag-part",
             "magnitude", "angle", "conjugate", "complex?", "numerator",
             "denominator", "rationalize", "rational?"}:
        return "numeric"
    # numeric (arithmetic / math / bit)
    if (n in {"+", "-", "*", "/", "add2", "sub2", "mul2", "div2", "expt", "pow",
              "min", "max", "modulo", "remainder", "quotient", "abs", "gcd", "lcm",
              "sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "sqrt",
              "floor", "ceiling", "round", "truncate", "sinh", "cosh", "tanh",
              "exact", "inexact", "exact->inexact", "inexact->exact",
              "number->string", "string->number", "sign", "arithmetic-shift",
              # Directional shift spellings and population count (documented in
              # docs/tutorials/20_BITWISE_AND_SYSTEM.md and
              # docs/API_REFERENCE.md) belong with their bitwise siblings.
              "bit-shift-left", "bit-shift-right", "popcount", "bit-count"}
            or n.startswith("bitwise") or n in {"<", ">", "<=", ">=", "="}):
        return "numeric"
    # predicates
    if n.endswith("?"):
        return "predicate"
    # string / char
    if (n.startswith("string") or n.startswith("char") or n.startswith("substring")
            or n.startswith("symbol->") or n.startswith("->string")
            or n in {"make-string", "list->string", "string->list", "list->symbol",
                     "symbol->string", "string->symbol", "char->integer",
                     "integer->char", "number->string", "string->number",
                     "list->string", "ansi-strip"}):
        return "string_char"
    # vectors / bytevectors
    if n.startswith("vector") or n.startswith("bytevector") or n in {
            "make-vector", "list->vector", "vector->list", "make-bytevector"}:
        return "vector"
    # list / pair
    if (n in {"cons", "car", "cdr", "list", "length", "append", "reverse",
              "member", "memq", "memv", "assoc", "assq", "assv", "list-ref",
              "list-tail", "last-pair", "iota", "list-copy", "map1", "take", "drop",
              "last", "list*", "remove", "remq", "remv", "acons", "split-at",
              "set-car!", "set-cdr!"}
            or re.fullmatch(r"c[ad]{2,4}r", n)):
        return "list_pair"
    # equality / misc core
    if n in {"eq?", "eqv?", "equal?", "not", "void", "error", "type-of",
             "boolean=?", "error-object?", "error-object-message",
             "error-object-irritants"}:
        return "misc_core"
    return "misc"


def build_manifest():
    vm = extract_builtin_table(VM_C)
    mirrors = extract_mirror_annotations(VM_C)
    aot = extract_llvm_dispatch()
    ops = extract_ast_ops()
    forms = extract_special_forms()
    stdlib_surface = extract_stdlib_surface()
    stdlib = stdlib_surface["names"]
    stdlib_closure = stdlib_surface["closure_names"]
    vm_prelude = extract_vm_prelude_defs()

    # AOT-dispatched names that are also declared special forms are syntax, not
    # builtins — keep them out of the builtin set (they appear under forms).
    form_names = set(forms)
    aot_builtins = {n for n in aot if n not in form_names}

    raw_names = sorted(set(vm) | aot_builtins)

    # Fold ALIASES into their canonical spelling before assigning one
    # language-surface record per construct: an alternate spelling for the
    # same dispatch is not a second builtin (see ALIASES above).
    aliases_by_canonical = {}
    for alias, canonical in ALIASES.items():
        if alias not in raw_names:
            continue
        if canonical not in raw_names:
            raise ValueError("alias %r names canonical %r, which is not on "
                             "the dispatch surface" % (alias, canonical))
        aliases_by_canonical.setdefault(canonical, []).append(alias)

    names = [n for n in raw_names if n not in ALIASES]
    builtins = []
    for name in names:
        member_names = [name] + aliases_by_canonical.get(name, [])
        in_v = any(m in vm or m in vm_prelude for m in member_names)
        in_a = any(m in aot_builtins for m in member_names)
        ids = sorted(set().union(*(set(vm.get(m, {}).get("ids", []))
                                   for m in member_names)))
        arity = None
        minimum = None
        for m in member_names:
            source = vm.get(m, {})
            arity = source.get("arity")
            if arity is not None:
                minimum = vm.get(m, {}).get("min_arity", arity)
                break
        backends = []
        if in_v:
            backends.append("vm")
        if in_a:
            backends.append("native_llvm")
        entry = {
            "name": name,
            "ids": ids,
            "arity": arity,
            # The CALLER's minimum, which is what a wrong-arity check must
            # read; `arity` above is the opcode's operand count and over-states
            # it for every builtin with a documented optional parameter.
            "min_arity": minimum,
            "backends": backends,
            "category": categorize(name),
        }
        if name in aliases_by_canonical:
            entry["aliases"] = sorted(aliases_by_canonical[name])
        mirror = next((mirrors[m] for m in member_names if m in mirrors), None)
        if mirror is not None:
            # Keep the source relationship even when the spelling is VM-backed;
            # LLVM name evidence does not make the mirror redundant.
            entry["mirrors"] = mirror
        builtins.append(entry)

    # The Moonlab agent APIs are opt-in, but are still user-callable Scheme
    # names and therefore need to participate in total-language coverage. They
    # have no core builtin id by design; `agent_ffi` honestly records their
    # dispatch vehicle without pretending the VM implements them.
    agent_provides = {}
    for _, (path, _, _) in QUANTUM_AGENT_BUILTINS.items():
        agent_provides.setdefault(path, extract_agent_module_provides(path))
    for name, (path, arity, category) in sorted(QUANTUM_AGENT_BUILTINS.items()):
        if name not in agent_provides[path]:
            raise ValueError("tracked quantum API %s is not provided by %s"
                             % (name, os.path.relpath(path, REPO)))
        if name in names:
            raise ValueError("quantum agent API %s unexpectedly collides with a core builtin"
                             % name)
        builtins.append({
            "name": name,
            "ids": [],
            "arity": arity,
            "min_arity": arity,
            "backends": ["agent_ffi"],
            "category": category,
            "source": os.path.relpath(path, REPO),
        })

    special = []
    for kw in sorted(forms):
        special.append({
            "name": kw,
            "op": forms[kw],
            "category": categorize(kw),
        })

    prelude_entries = []
    builtin_names = set(names)
    identity_origins = {}
    for name in sorted(set(vm_prelude) | set(stdlib) | set(stdlib_closure)):
        sources = []
        if name in vm_prelude:
            sources.append("vm_bootstrap")
        if name in stdlib:
            sources.append("stdlib")
        elif name in stdlib_closure:
            sources.append("stdlib_closure_presence")
        provenance = stdlib.get(name, []) + stdlib_closure.get(name, [])
        identities = set()
        categories = set()
        for evidence in provenance:
            category = evidence.get("binding_category", "unknown")
            categories.add(category)
            identity = evidence.get("binding_identity")
            if identity:
                origin = evidence["resolved_binding_origin"]
                origin_key = (origin["module"], origin["path"], origin["name"])
                previous = identity_origins.setdefault(identity, origin_key)
                if previous != origin_key:
                    raise ValueError("binding identity collision for %r: %r vs %r" %
                                     (identity, previous, origin_key))
                identities.add(identity)
        entry = {
            "name": name,
            "category": categorize(name),
            "also_builtin": name in builtin_names,
            "sources": sources,
            "provenance": provenance,
            "binding_category": (next(iter(categories)) if len(categories) == 1
                                 else "unknown"),
        }
        if len(identities) == 1:
            entry["binding_identity"] = next(iter(identities))
        prelude_entries.append(entry)

    manifest = {
        "_meta": {
            "schema_version": 3,
            "evidence_provenance": ("generated from live production sources; "
                                    "legacy compiler references are frozen separately"),
            "description": "Complete Eshkol language surface (ground truth for "
                           "exposure-engine coverage).",
            "sources": {
                "vm_builtins": "lib/backend/eshkol_vm.c BUILTINS[] (sole core ids/arity authority)",
                "native_llvm": "lib/backend/llvm_codegen.cpp dispatch names only; no ids/arity/ABI claim",
                "ast_ops": "inc/eshkol/eshkol.h eshkol_op_t",
                "special_forms": "lib/frontend/parser.cpp get_operator_type + "
                                 "direct dispatch",
                "vm_bootstrap": "lib/backend/vm_prelude_source.h",
                "stdlib": "top-level definitions in lib/stdlib.esk plus exports from modules it directly requires",
                "stdlib_closure_presence": "exports in recursive dependencies, retained as source evidence but not promoted to root stdlib availability without a direct re-export",
                "stdlib_direct_requires": stdlib_surface["root_requires"],
                "stdlib_module_closure": stdlib_surface["modules"],
                "mirrors": "BUILTINS[] rows annotated `/* mirrors: <public "
                           "name> */` — this row is one engine's private "
                           "spelling (arity split, or lower-level handle "
                           "form) of that public construct",
                "quantum_agent_ffi": [
                    "lib/agent/quantum.esk provide",
                    "lib/agent/pqc.esk provide",
                ],
            },
        },
        "counts": {
            "builtins_total": len(builtins),
            "builtins_in_vm_table": sum(1 for b in builtins
                                        if "vm" in b["backends"]),
            "builtins_in_llvm_aot_dispatch": sum(1 for b in builtins
                                                 if "native_llvm" in b["backends"]),
            "builtins_aot_only": sum(1 for b in builtins
                                     if b["backends"] == ["native_llvm"]),
            "quantum_agent_ffi_builtins": len(QUANTUM_AGENT_BUILTINS),
            "special_forms": len(special),
            "ast_ops": len(ops),
            "prelude": len(prelude_entries),
        },
        "builtins_by_category": _count_by_cat(builtins),
        "special_forms_by_category": _count_by_cat(special),
        "ast_ops": ops,
        "builtins": builtins,
        "special_forms": special,
        "prelude": prelude_entries,
    }
    return manifest


def _count_by_cat(entries):
    out = {}
    for e in entries:
        out[e["category"]] = out.get(e["category"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "tests", "coverage",
                                                  "language_surface.json"))
    ap.add_argument(
        "--check",
        action="store_true",
        help="fail when --out differs from the surface extracted from source",
    )
    args = ap.parse_args()
    manifest = build_manifest()
    rendered = json.dumps(manifest, indent=2, sort_keys=False) + "\n"

    if args.check:
        try:
            with open(args.out, encoding="utf-8") as fh:
                committed = fh.read()
        except FileNotFoundError:
            print("FAIL: language-surface manifest is missing: %s" % args.out,
                  file=sys.stderr)
            return 1
        if committed != rendered:
            print("FAIL: language-surface manifest is stale: %s" % args.out,
                  file=sys.stderr)
            diff = difflib.unified_diff(
                committed.splitlines(),
                rendered.splitlines(),
                fromfile=args.out,
                tofile="source-extracted language surface",
                lineterm="",
            )
            for line in list(diff)[:200]:
                print(line, file=sys.stderr)
            print("Regenerate with: python3 scripts/gen_language_surface.py",
                  file=sys.stderr)
            return 1
        print("OK: language-surface manifest matches compiler source (%s)" % args.out)
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(rendered)
    c = manifest["counts"]
    print("Wrote %s" % args.out)
    print("  builtins: %d (vm %d, llvm-aot %d, aot-only %d)"
          % (c["builtins_total"], c["builtins_in_vm_table"], c["builtins_in_llvm_aot_dispatch"],
             c["builtins_aot_only"]))
    print("  special forms: %d   ast ops: %d   prelude: %d"
          % (c["special_forms"], c["ast_ops"], c["prelude"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
