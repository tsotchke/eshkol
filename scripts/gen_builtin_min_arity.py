#!/usr/bin/env python3
r"""gen_builtin_min_arity.py — derive the CALLER MINIMUM of every builtin from
the contracts the compiler and the documentation already state, and write it
into the `min_arity` column of BUILTINS[] in lib/backend/eshkol_vm.c.

WHY THIS EXISTS. `arity` in BUILTINS[] is the number of operands
emit_builtin_preamble() loads into the opcode — the SHAPE of the call, not the
number of arguments a caller must supply. Every builtin with an OPTIONAL
parameter therefore has an `arity` that over-states its minimum:
`(substring s 1)`, `(make-string 3)`, `(read-line)` and `(append)` are legal
calls that the bytecode VM refused, because its under-arity check falls back to
`arity` whenever a row leaves `min_arity` at 0 — which was 739 rows out of 742.
`min_arity` exists for exactly this and had been filled in by hand, three times,
wherever somebody noticed.

Hand-filling 742 rows would reproduce the same defect one transcription later,
so the numbers are DERIVED, from the two places the contract is already
written down:

  1. THE NATIVE LOWERING'S OWN ARITY GUARD. `lib/backend/string_io_codegen.cpp`
     refuses a short `substring` with

         substring requires 2 or 3 arguments: (substring string start [end])

     That sentence is the contract native enforces, it names the optional
     parameter, and it is machine-readable. ~150 builtins carry one.

  2. A DECLARATIVE DOCUMENTED SIGNATURE — `(string-pad-left s width [char]) —
     left-pad a string...` in docs/api/**, the public headers, and the
     reference pages. Bracketed parameters are optional, `...` is variadic.

     EXAMPLE CALLS ARE NOT SIGNATURES. `(make-vector 3)` appears in a hundred
     documents as something a program does, and reading it as a signature would
     claim make-vector takes one argument. Only lines of the declarative
     `(name params) — description` / `- `(name params)` - description` shape are
     read, and only when the parameters look like parameter NAMES rather than
     literals.

Where a row has neither, the minimum stays `arity`: that is the status quo, so
a builtin nobody has documented cannot silently become more permissive.

Both engines consult the result through eshkol_builtin_min_arity()
(inc/eshkol/core/arity_contract.h), and scripts/check_builtin_min_arity.py
re-derives it from these same sources and fails the build when the table has
drifted away from them.

Usage:
  gen_builtin_min_arity.py                 # report what would change
  gen_builtin_min_arity.py --apply         # rewrite BUILTINS[] in place
  gen_builtin_min_arity.py --json OUT      # dump the derivation with evidence
"""

import argparse
import json
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
VM_C = REPO / "lib" / "backend" / "eshkol_vm.c"

# The min_arity column's encoding, mirroring BuiltinDef in eshkol_vm.c. Zero
# is "unset — the minimum IS arity", which is what a three-field initialiser
# produces and what 700 rows rely on, so a builtin whose documented minimum is
# genuinely ZERO (`(read-line)`, `(read-char)`) cannot say so with a 0 and gets
# its own marker instead of being silently indistinguishable from the default.
MIN_IS_ARITY = 0
MIN_VARIADIC = -1
MIN_ZERO = -2

# A BuiltinDef row: {"name", id, arity} with optional min_arity and variadic
# fields. Mirrors the pattern in check_builtin_min_arity.py and
# gen_language_surface.py — the field layout is load-bearing in all three, and
# a pattern that cannot see a field drops the builtin from that gate's view.
ROW = re.compile(
    r'\{\s*"((?:[^"\\]|\\.)*)"\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*'
    r'(?:,\s*(-?\d+)\s*)?(?:,\s*(-?\d+)\s*)?\}')

WORD_NUMBERS = {"zero": 0, "no": 0, "one": 1, "two": 2, "three": 3,
                "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def _count(token):
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return WORD_NUMBERS.get(token)


# ── source 1: the native lowering's own arity guard ────────────────────────
#
# Ordered: a more specific shape must match before a more general one, or
# "requires at least 2 arguments" is read by the bare "requires N arguments"
# rule as an exact 2.
GUARD_PATTERNS = [
    (re.compile(r'^(?P<name>\S+?):?\s+(?:requires|expects|takes|needs)\s+at\s+least\s+(?P<lo>\w+)\s+(?:\w+\s+)?arg'), "min"),
    (re.compile(r'^(?P<name>\S+?):?\s+(?:requires|expects|takes|needs)\s+at\s+most\s+(?P<lo>\w+)\s+arg'), "max"),
    (re.compile(r'^(?P<name>\S+?):?\s+(?:requires|expects|takes|needs)\s+(?P<lo>\d+)\s*(?:-|–|\s+to\s+|\s+or\s+)\s*(?P<hi>\d+)\s+arg'), "range"),
    (re.compile(r'^(?P<name>\S+?):?\s+(?:requires|expects|takes|needs)\s+exactly\s+(?P<lo>\w+)\s+(?:\w+\s+)?arg'), "exact"),
    (re.compile(r'^(?P<name>\S+?):?\s+(?:requires|expects|takes|needs)\s+(?P<lo>\w+)\s+(?:\w+\s+)?arg'), "exact"),
]

GUARD_SOURCES = ("lib/**/*.cpp", "lib/**/*.c", "lib/**/*.mm", "lib/**/*.h")


def guard_minima(repo):
    """name -> (minimum, evidence) mined from native lowering arity guards.

    A builtin can be guarded in more than one lowering (a fast path and a
    general one). The SMALLEST stated minimum wins: it is the shortest call
    native will actually compile, and claiming more would refuse a call the
    other lowering accepts.
    """
    out = {}
    for pattern in GUARD_SOURCES:
        for path in sorted(pathlib.Path(repo).glob(pattern)):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for literal in re.findall(r'"((?:[^"\\]|\\.)*)"', text):
                if "arg" not in literal:
                    continue
                for rx, kind in GUARD_PATTERNS:
                    m = rx.match(literal)
                    if not m:
                        continue
                    if kind == "max":
                        break                      # states a ceiling, not a floor
                    low = _count(m.group("lo"))
                    if low is None:
                        break
                    name = m.group("name")
                    evidence = "%s: %s" % (
                        os.path.relpath(path, repo), literal[:110])
                    prev = out.get(name)
                    if prev is None or low < prev[0]:
                        out[name] = (low, evidence)
                    break
    return out


# ── source 0: the native dispatch's own fixed-arity declaration ────────────
#
# lib/backend/system_codegen.cpp declares two hundred builtins through
# `<N>_ARG_BUILTIN(method, "c_func")` macros, each of which expands to
# `if (op->call_op.num_vars != N) return tagged_.packNull();`. That is an
# EXACT arity, stated by the code that runs: `(string-pad-left "a" 3)` does not
# pad on native, it returns null, because stringPadLeft is a THREE_ARG_BUILTIN.
#
# This outranks a documented signature, and where the two disagree the
# DOCUMENT is the thing that is wrong: `docs/api/backend/system_codegen.md`
# renders `(string-pad-left s width [char])` from a header comment describing
# an optional parameter no lowering implements. Believing the bracket would
# make the VM accept — and silently mis-answer — a call native refuses.
ARG_COUNT_WORDS = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
                   "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8}

SYSTEM_CODEGEN = "lib/backend/system_codegen.cpp"
LLVM_CODEGEN = "lib/backend/llvm_codegen.cpp"

# Two macro families say the same thing: `<N>_ARG_BUILTIN` for the system
# builtins and `CE_<N>_ARG` for the consciousness-engine ones. Both expand to
# `if (num_vars != N) return packNull();`. Reading only the first would have
# left `(fg-infer! g)` and `(make-workspace)` looking optional on the strength
# of a doc line, when native answers null for both.
MACRO_INSTANCE = re.compile(
    r'^(?:CE_)?(ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT)_ARG(?:_BUILTIN)?\(\s*(\w+)\s*,',
    re.M)
DISPATCH = re.compile(
    r'func_name == "((?:[^"\\]|\\.)*)"\)\s*co_return\s+(?:\w+_?->)?(\w+)\(op\)')


def fixed_native_arity(repo):
    """name -> (exact arity, evidence) for every builtin the native dispatch
    lowers through a fixed-arity macro."""
    repo = pathlib.Path(repo)
    system = (repo / SYSTEM_CODEGEN)
    llvm = (repo / LLVM_CODEGEN)
    if not system.exists() or not llvm.exists():
        return {}
    by_method = {}
    for words, method in MACRO_INSTANCE.findall(system.read_text(errors="replace")):
        by_method[method] = ARG_COUNT_WORDS[words]
    out = {}
    for name, method in DISPATCH.findall(llvm.read_text(errors="replace")):
        if method in by_method:
            out[name] = (by_method[method],
                         "%s: %s is a %s_ARG_BUILTIN"
                         % (SYSTEM_CODEGEN, method,
                            [w for w, n in ARG_COUNT_WORDS.items()
                             if n == by_method[method]][0]))
    return out


# ── source 1a: a native lowering with an explicit ZERO-ARGUMENT case ───────
#
# A lowering that opens with the identity for an empty call is variadic with a
# documented minimum of zero, and it says so in code:
#
#     if (func_name == "bytevector-append") {
#         uint64_t n = op->call_op.num_vars;
#         if (n == 0) { ... an empty bytevector ... co_return ...; }
#
# `(bytevector-append)` therefore answers the empty bytevector on native, and
# the VM refused it. This is the same test check_builtin_min_arity.py already
# applies to a row that DECLARES itself variadic — a `num_vars == 0` case in
# the handler — applied to every name the dispatch lowers inline rather than
# only to the two names a human listed.
#
# A zero-argument case that REFUSES is not an identity, so a branch that
# errors, warns, or answers null is not read as one.
ZERO_CASE_REFUSES = ("eshkol_error", "eshkol_arity_error", "eshkol_warn",
                     "markFatalCodegenError", "packNull", "nullptr")


def _brace_block(text, open_index):
    depth, i = 0, open_index
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
        i += 1
    return ""


def zero_argument_lowerings(repo):
    """name -> (0, evidence) for builtins whose inline lowering answers a
    zero-argument call with a value."""
    path = pathlib.Path(repo) / LLVM_CODEGEN
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    out = {}
    for m in re.finditer(r'func_name == "((?:[^"\\]|\\.)*)"\)\s*\{', text):
        body = _brace_block(text, m.end() - 1)
        if not body:
            continue
        # The count may be read into a local first (`uint64_t n = num_vars;`).
        aliases = {"num_vars"}
        for alias in re.finditer(r'\b(\w+)\s*=\s*[^;]*call_op\.num_vars', body):
            aliases.add(alias.group(1))
        for alias in sorted(aliases):
            zero = re.search(r'\b%s\s*==\s*0\s*\)\s*\{' % re.escape(alias), body)
            if not zero:
                continue
            branch = _brace_block(body, zero.end() - 1)
            if not branch or not re.search(r'\bco_return\b|\breturn\b', branch):
                continue
            if any(bad in branch for bad in ZERO_CASE_REFUSES):
                continue
            out[m.group(1)] = (0, "%s: `%s` lowering answers %s == 0 with a value"
                               % (LLVM_CODEGEN, m.group(1), alias))
            break
    return out


# ── source 1b: the Scheme definition the native engine compiles ────────────
#
# Several public procedures are not lowered by the backend at all — native
# reaches them through the core modules compiled into every program. `append`
# is one: lib/core/list/transform.esk defines
#
#     (define (append . lists)
#       (cond ((null? lists) '())
#             ...
#
# a rest-argument lambda with ZERO required parameters whose first clause IS
# the R7RS identity for `(append)`. That is not a description of the contract,
# it is the code that answers the call, so it outranks any prose. A guard or a
# fixed-arity macro still outranks IT, because those belong to a lowering that
# shadows the module definition.
SCHEME_SOURCES = ("lib/**/*.esk",)

SCHEME_DEFINE = re.compile(
    r'\(define\s+\(\s*([a-zA-Z0-9_!?*<>=/+%^~.$&:@\-]+)((?:[^()\n]|\([^()]*\))*)\)')


def scheme_minima(repo):
    """name -> (minimum, evidence) for rest-argument procedures defined in the
    core Scheme modules. A definition WITHOUT a rest argument states an exact
    arity and is not a claim about optionality, so only `(name . rest)` and
    `(name a b . rest)` are read here."""
    out = {}
    for pattern in SCHEME_SOURCES:
        for path in sorted(pathlib.Path(repo).glob(pattern)):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for m in SCHEME_DEFINE.finditer(text):
                params = m.group(2)
                if "." not in params:
                    continue                     # fixed arity: no claim to make
                fixed = params.split(".")[0].split()
                if any(not PARAM.match(tok) for tok in fixed):
                    continue
                name = m.group(1)
                evidence = "%s: (define (%s%s) …)" % (
                    os.path.relpath(path, repo), name, params.rstrip())
                prev = out.get(name)
                if prev is None or len(fixed) < prev[0]:
                    out[name] = (len(fixed), evidence)
    return out


# ── source 2: a declarative documented signature ───────────────────────────
#
# `(name a b [c]) — description`  (docs/api/**, public headers)
# `- `(name a b [c])` - description`  (reference pages, the language spec)
DOC_SOURCES = ("docs/**/*.md", "inc/**/*.h", "inc/**/*.hpp")

_SIG_BODY = r'([a-zA-Z0-9_!?*<>=/+%^~.$&:@\-]+)((?:[^()]|\([^()]*\))*)'
DOC_SIGNATURE_FORMS = [
    re.compile(r'^\(' + _SIG_BODY + r'\)\s*[—–]\s*\S'),          # (f a b) — desc
    re.compile(r'^-\s+`\(' + _SIG_BODY + r'\)`\s*[-—–:]\s*\S'),  # - `(f a b)` - desc
    re.compile(r'^`\(' + _SIG_BODY + r'\)`\s*[—–]\s*\S'),        # `(f a b)` — desc
    re.compile(r'^\*\*`?\(' + _SIG_BODY + r'\)`?\*\*\s*[-—–:]\s*\S'),
]

# A parameter NAME. Literals (`3`, `"s"`, `#t`, `'sym`) mean the line is an
# example call, not a signature, and the whole line is discarded.
PARAM = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_?!*<>=/+.\-]*$')
ELLIPSIS = {"...", "....", "…", ".", "·"}


def parse_signature(params_text):
    """(minimum, variadic) for a signature body, or None if it is not one.

    Bracketed parameters are optional and everything after the first optional
    is optional too; `...` and a trailing `x...` are variadic. A token that is
    not a plain identifier (a number, a string, a quote) means the line was an
    example call and the caller must discard it.
    """
    tokens, current, depth = [], "", 0
    for ch in params_text:
        if ch == "[":
            depth += 1
            current += ch
        elif ch == "]":
            depth -= 1
            current += ch
        elif ch.isspace() and depth == 0:
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)

    minimum, variadic, seen_optional, named = 0, False, False, 0
    for token in tokens:
        if token.startswith("["):
            seen_optional = True
            continue
        if token in ELLIPSIS or token.endswith("..."):
            # A repeated parameter (`lst...`, `dim...`) states a MAXIMUM, not a
            # minimum: these pages write `(ones dim...)` for a procedure whose
            # native lowering refuses `(ones)` outright ("ones requires at
            # least 1 dimension argument"). So the repetition is recorded as
            # variadic and the parameter still counts as required; only an
            # explicit `[bracket]` — or a native guard, which outranks the doc
            # — lowers a minimum.
            variadic = True
            if token not in ELLIPSIS:
                named += 1
                if not seen_optional:
                    minimum += 1
            continue
        if not PARAM.match(token):
            return None                      # an example call, not a signature
        named += 1
        if seen_optional:
            continue                         # a required param after an optional
        minimum += 1                         # one is a documentation bug, not ours
    if tokens and named == 0:
        # `(http-request ...)` and `(make-temp-file ...)` are PROSE — a page
        # saying "and its arguments" — not a claim that the procedure takes
        # none. Read as a signature they would set the minimum to zero and the
        # VM would stop refusing `(http-request)`, which native refuses. A body
        # that is EMPTY is a real nullary signature and stays one; a body that
        # is nothing but an ellipsis is discarded.
        return None
    return minimum, variadic


def doc_minima(repo):
    """name -> (minimum, evidence) from declarative documented signatures.

    The SMALLEST documented minimum wins for the same reason it does among
    guards: two pages describing the same procedure at different arities are
    describing one procedure whose extra parameters are optional.
    """
    out = {}
    for pattern in DOC_SOURCES:
        for path in sorted(pathlib.Path(repo).glob(pattern)):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for raw in text.splitlines():
                line = raw.strip().lstrip("*").strip()
                for rx in DOC_SIGNATURE_FORMS:
                    m = rx.match(line)
                    if not m:
                        continue
                    parsed = parse_signature(m.group(2))
                    if parsed is None:
                        break
                    name = m.group(1)
                    evidence = "%s: %s" % (
                        os.path.relpath(path, repo), line[:110])
                    prev = out.get(name)
                    if prev is None or parsed[0] < prev[0]:
                        out[name] = (parsed[0], evidence)
                    break
    return out


def table_rows(text):
    m = re.search(r"static const BuiltinDef BUILTINS\[\] = \{(.*?)\n\};", text, re.S)
    if not m:
        raise SystemExit("gen_builtin_min_arity: BUILTINS[] not found in eshkol_vm.c")
    rows = []
    for name, native_id, arity, min_arity, variadic in ROW.findall(m.group(1)):
        if not name:
            continue
        rows.append({
            "name": name,
            "native_id": int(native_id),
            "arity": int(arity),
            "min_arity": int(min_arity) if min_arity else 0,
            "variadic": int(variadic) if variadic else 0,
        })
    return rows


def derive(repo):
    """Return the derivation for every BUILTINS[] row.

    `minimum` is what the row's min_arity column should say: 0 when the
    minimum IS the opcode's operand count (the common case, and the cheapest
    thing to write), a positive number when a documented optional parameter
    makes the real minimum smaller, and the row's existing negative value when
    it declares itself variadic in the native lowering.
    """
    repo = pathlib.Path(repo)
    rows = table_rows(VM_C.read_text() if repo == REPO
                      else (repo / "lib/backend/eshkol_vm.c").read_text())
    guards, docs = guard_minima(repo), doc_minima(repo)
    fixed = fixed_native_arity(repo)
    zero_case = zero_argument_lowerings(repo)
    scheme = scheme_minima(repo)
    out = []
    for row in rows:
        name, arity = row["name"], row["arity"]
        record = dict(row, source="table", evidence="", derived=arity,
                      documented=None)
        if row["min_arity"] == MIN_VARIADIC:
            # A variadic declaration is a claim about the native lowering that
            # check_builtin_min_arity.py verifies against the handler itself;
            # nothing here can second-guess it. Note only -1 is carried over:
            # every other column value is RE-DERIVED from the sources below, so
            # running this generator twice produces the same table as running it
            # once — a generator that read its own last answer as evidence would
            # freeze whatever it wrote first, mistakes included.
            record.update(source="variadic", derived=row["min_arity"])
            out.append(record)
            continue
        for source, table in (("native-fixed", fixed), ("zero-case", zero_case),
                              ("guard", guards), ("scheme", scheme),
                              ("doc", docs)):
            if name in table:
                minimum, evidence = table[name]
                record.update(source=source, evidence=evidence,
                              documented=minimum)
                break
        # A documented optional parameter that the native lowering does not
        # implement is a DOCUMENTATION DEFECT, and the table must not act on
        # it. Record it so the report can name the page to fix.
        if record["source"] == "native-fixed" and name in docs:
            doc_minimum, doc_evidence = docs[name]
            if doc_minimum < record["documented"]:
                record["doc_conflict"] = {
                    "documented_minimum": doc_minimum,
                    "native_exact": record["documented"],
                    "evidence": doc_evidence,
                }
        documented = record["documented"]
        if documented is None or documented >= arity:
            # No evidence, or evidence that agrees with the opcode's operand
            # count. Either way the row makes no separate claim.
            record["derived"] = arity
            record["min_arity_column"] = MIN_IS_ARITY
        else:
            record["derived"] = documented
            record["min_arity_column"] = MIN_ZERO if documented == 0 else documented
        out.append(record)
    return out


def render_row(record):
    """The BuiltinDef initialiser for one row, in the table's own shape."""
    fields = ['"%s"' % record["name"], str(record["native_id"]), str(record["arity"])]
    column = record.get("min_arity_column", record["min_arity"])
    if record["source"] == "variadic":
        column = record["min_arity"]
    if column or record["variadic"]:
        fields.append(str(column))
    if record["variadic"]:
        fields.append(str(record["variadic"]))
    return "{" + ", ".join(fields) + "}"


def apply(records, path):
    """Rewrite each BUILTINS[] initialiser in place, preserving the table's
    layout: only the initialiser text of a row whose min_arity column changes
    is touched, so the comments and the grouping survive."""
    text = path.read_text()
    m = re.search(r"(static const BuiltinDef BUILTINS\[\] = \{)(.*?)(\n\};)", text, re.S)
    body = m.group(2)
    by_position = list(ROW.finditer(body))
    lookup = {r["name"]: r for r in records}
    pieces, last, changed = [], 0, 0
    for match in by_position:
        name = match.group(1)
        record = lookup.get(name)
        if record is None or not name:
            continue
        replacement = render_row(record)
        if replacement == match.group(0):
            continue
        pieces.append(body[last:match.start()])
        pieces.append(replacement)
        last = match.end()
        changed += 1
    pieces.append(body[last:])
    path.write_text(text[:m.start(2)] + "".join(pieces) + text[m.end(2):])
    return changed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--apply", action="store_true",
                    help="rewrite BUILTINS[] in lib/backend/eshkol_vm.c")
    ap.add_argument("--json", help="write the full derivation, with evidence")
    args = ap.parse_args()

    records = derive(args.repo)
    changes = [r for r in records
               if r.get("min_arity_column", r["min_arity"]) != r["min_arity"]]
    optional = [r for r in records if r.get("min_arity_column")]
    print("BUILTINS[] rows                : %d" % len(records))
    print("rows with a documented optional: %d" % len(optional))
    print("rows whose column changes      : %d" % len(changes))
    by_source = {}
    for r in optional:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for source in sorted(by_source):
        print("  from %-6s: %d" % (source, by_source[source]))
    conflicts = [r for r in records if r.get("doc_conflict")]
    if conflicts:
        print("documented optionals the native lowering does not implement: %d"
              % len(conflicts))
        for r in sorted(conflicts, key=lambda r: r["name"]):
            c = r["doc_conflict"]
            print("  %-28s doc minimum %d vs native exact %d — %s"
                  % (r["name"], c["documented_minimum"], c["native_exact"],
                     c["evidence"][:90]))
    for r in sorted(changes, key=lambda r: r["name"]):
        print("  %-28s arity %d -> minimum %d   [%s] %s"
              % (r["name"], r["arity"], r["derived"], r["source"],
                 r["evidence"][:90]))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(records, indent=1))
        print("wrote %s" % args.json)
    if args.apply:
        n = apply(records, pathlib.Path(args.repo) / "lib/backend/eshkol_vm.c")
        print("rewrote %d BUILTINS[] rows" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
