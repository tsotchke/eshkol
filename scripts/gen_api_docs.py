#!/usr/bin/env python3
"""Generate docs/api/ — a browsable API reference for the Eshkol public
headers (inc/eshkol/**/*.h) — by harvesting Doxygen /** ... */ comment
blocks that precede each declaration.

This is a lightweight, best-effort C-header scanner, not a real C parser.
It handles the constructs actually used across inc/eshkol/**/*.h:

    - function prototypes                 ret name(args);
    - static inline function definitions  static inline ret name(args) { ... }
    - typedef struct/union/enum { ... } name_t;
    - simple typedefs                     typedef unsigned long name_t;
    - function-pointer typedefs           typedef ret (*name_t)(args);
    - object-like / function-like macros  #define NAME ...  /  #define NAME(x) ...
    - extern variable declarations        extern int name;

For each declaration it looks for an immediately preceding /** ... */ block
(only whitespace, blank lines, other comments, and skippable preprocessor
directives may separate the two) and extracts @brief / @param / @return /
@note / @warning / @deprecated content from it. Declarations with no such
block are still listed (as "Undocumented" entries) so coverage is visible
and the reference stays complete.

Usage:
    scripts/gen_api_docs.py [--repo-root PATH] [--check]

    --check   Do not write files; exit 1 if the generated output would
              differ from what's on disk (drift check for CI/local use).

Output layout (mirrors inc/eshkol/):
    docs/api/README.md              Index grouped by subsystem + coverage
    docs/api/INDEX.md                Alphabetical symbol -> file table
    docs/api/<subsystem>/<header>.md One page per header, e.g.
                                      docs/api/core/bignum.md,
                                      docs/api/backend/gpu/gpu_memory.md,
                                      docs/api/eshkol.md (root headers)

Deterministic: symbols are emitted in source order within a file, files are
walked in sorted order, and the index/summary tables are sorted. Re-running
against an unchanged tree produces byte-identical output.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INC_ROOT = REPO_ROOT / "inc" / "eshkol"
OUT_ROOT = REPO_ROOT / "docs" / "api"

DOXY_TAGS = {"brief", "param", "return", "returns", "note", "warning", "deprecated", "see"}

SKIP_DIRECTIVES = re.compile(
    r"^\s*#\s*(include|ifdef|ifndef|endif|else|elif|pragma|undef|if)\b"
)
DEFINE_DIRECTIVE = re.compile(r"^\s*#\s*define\b")
GUARD_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*$")


# --------------------------------------------------------------------------- #
#  Comment / string scanning
# --------------------------------------------------------------------------- #


@dataclass
class Comment:
    start: int
    end: int  # exclusive, points just past the closing */
    is_doc: bool
    text: str  # raw text between /** and */ (or /* and */)


def scan_comments_and_strings(src: str) -> tuple[str, list[Comment]]:
    """Return (code_only, comments).

    code_only is `src` with every comment's characters (except newlines)
    replaced by spaces, so declaration scanning never sees comment content
    but line numbers / offsets are preserved. String and char literals are
    left untouched (so `extern "C"` remains intact).
    """
    n = len(src)
    out = list(src)
    comments: list[Comment] = []
    i = 0
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                j += 1
            i = j
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            is_doc = src.startswith("/**", i) and not src.startswith("/**/", i) and not src.startswith("/***", i)
            comments.append(Comment(i, j, is_doc, src[i + 3 if is_doc else i + 2 : j - 2]))
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out), comments


def blank_span(buf: list[str], start: int, end: int) -> None:
    for k in range(start, end):
        if buf[k] != "\n":
            buf[k] = " "


WRAPPER_BRACE_RE = re.compile(
    r'(extern\s*"C"\s*\{|\bnamespace\b(?:\s+[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)?\s*\{)'
)


def strip_wrapper_braces(code: list[str]) -> None:
    """Blank the `{` of every `extern "C" { ... }` / `namespace NAME { ... }`
    wrapper and its matching `}` in place, so the block-depth tracker below
    treats their contents as top-level declarations instead of a struct body.
    Re-scans until no more matches are found (wrappers can nest, e.g.
    `namespace eshkol { extern "C" { ... } }`).
    """
    while True:
        text = "".join(code)
        matches = list(WRAPPER_BRACE_RE.finditer(text))
        if not matches:
            return
        for m in matches:
            start, end = m.span()
            if "".join(code[start:end]).strip() == "":
                continue  # already stripped in a previous pass
            blank_span(code, start, end)
            depth = 1
            j = end
            while j < len(code) and depth > 0:
                if code[j] == "{":
                    depth += 1
                elif code[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                code[j - 1] = " "


# --------------------------------------------------------------------------- #
#  Declaration chunking
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    start: int
    end: int
    is_directive: bool


def line_spans(src: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for m in re.finditer(r"\n", src):
        spans.append((start, m.start() + 1))
        start = m.start() + 1
    if start < len(src):
        spans.append((start, len(src)))
    return spans


def find_directive_chunks(code_scan: str) -> list[Chunk]:
    """Find preprocessor directive lines (joining backslash-continuations),
    return their spans, independent of whether they'll become a symbol.
    """
    chunks: list[Chunk] = []
    lines = line_spans(code_scan)
    idx = 0
    while idx < len(lines):
        ls, le = lines[idx]
        line = code_scan[ls:le]
        if re.match(r"^\s*#", line):
            end_idx = idx
            while code_scan[lines[end_idx][0] : lines[end_idx][1]].rstrip("\n").rstrip().endswith("\\"):
                if end_idx + 1 >= len(lines):
                    break
                end_idx += 1
            chunks.append(Chunk(ls, lines[end_idx][1], True))
            idx = end_idx + 1
        else:
            idx += 1
    return chunks


TRAILING_NAME_RE = re.compile(r"\s*[A-Za-z_][\w\s,*\[\]]*;")
CONTAINER_KEYWORD_RE = re.compile(r"^(typedef|struct|union|enum|class)\b")


def find_code_chunks(code_scan: str) -> list[Chunk]:
    """Brace/paren-depth-aware scan for function/typedef/variable
    declarations. Assumes directive lines and extern "C" braces have
    already been blanked out of `code_scan`.
    """
    n = len(code_scan)
    brace_depth = 0
    paren_depth = 0
    chunk_start: int | None = None
    chunks: list[Chunk] = []
    i = 0
    while i < n:
        c = code_scan[i]
        if not c.isspace() and chunk_start is None:
            chunk_start = i
        if c == "{":
            brace_depth += 1
        elif c == "}":
            brace_depth = max(0, brace_depth - 1)
            if brace_depth == 0 and chunk_start is not None:
                # A `}` closing a struct/union/enum/class body may be
                # followed by a trailing type-alias name-list before the
                # terminating `;` (`typedef struct { ... } name_t, *pname_t;`).
                # Only a run of identifier/pointer/array/comma characters
                # ending in `;` counts as a continuation of this chunk, and
                # only when the chunk actually looks like a container body —
                # otherwise a function body immediately followed by an
                # unrelated `TYPE name;` declaration would be wrongly merged
                # (and the trailing declaration silently swallowed).
                is_container = bool(CONTAINER_KEYWORD_RE.match(code_scan[chunk_start : i + 1].lstrip()))
                m = TRAILING_NAME_RE.match(code_scan, i + 1) if is_container else None
                if m:
                    chunks.append(Chunk(chunk_start, m.end(), False))
                    i = m.end() - 1
                    chunk_start = None
                else:
                    chunks.append(Chunk(chunk_start, i + 1, False))
                    chunk_start = None
        elif c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth = max(0, paren_depth - 1)
        elif c == ";" and brace_depth == 0 and paren_depth == 0:
            if chunk_start is not None:
                chunks.append(Chunk(chunk_start, i + 1, False))
                chunk_start = None
        i += 1
    return chunks


# --------------------------------------------------------------------------- #
#  Symbol classification
# --------------------------------------------------------------------------- #


@dataclass
class Symbol:
    kind: str  # function | struct | union | enum | typedef | macro | variable
    name: str
    signature: str
    line: int
    brief: str = ""
    params: list[tuple[str, str]] = field(default_factory=list)
    returns: str = ""
    notes: list[str] = field(default_factory=list)
    documented: bool = False


IDENT_RE = re.compile(r"[A-Za-z_]\w*")
FUNC_PTR_NAME_RE = re.compile(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)")


def extract_trailing_name(body: str) -> str | None:
    body = body.strip()
    if body.endswith(";"):
        body = body[:-1]
    matches = list(FUNC_PTR_NAME_RE.finditer(body))
    if matches:
        return matches[-1].group(1)
    if "}" in body:
        tail = body.rsplit("}", 1)[1]
        names = [t.strip() for t in tail.split(",") if t.strip()]
        if names:
            return names[0]
    idents = IDENT_RE.findall(body)
    return idents[-1] if idents else None


def find_matching_brace(text: str, open_idx: int) -> int | None:
    """Given text[open_idx] == '{', return the index of its matching '}'."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


CLASS_LIKE_RE = re.compile(r"^(class|struct|union)\b(\s+[A-Za-z_]\w*)?")
ATTRIBUTE_IDENTS = {"__attribute__", "__declspec"}


def classify_chunk(text: str) -> tuple[Symbol, tuple[int, int] | None] | None:
    """Classify one declaration chunk. Returns (Symbol, inner_span) where
    inner_span is the (open_brace_idx, close_brace_idx) offsets within
    `text` for a C++ class/struct body worth recursing into, or None.
    """
    text = text.strip()
    if not text:
        return None

    if DEFINE_DIRECTIVE.match(text):
        if GUARD_DEFINE.match(text):
            return None  # include guard, not API
        m = re.match(r"^\s*#\s*define\s+([A-Za-z_]\w*)", text)
        if not m:
            return None
        name = m.group(1)
        sig = re.sub(r"\s+", " ", text).strip()
        return Symbol(kind="macro", name=name, signature=sig, line=0), None

    if SKIP_DIRECTIVES.match(text):
        return None
    if text.startswith("#"):
        return None
    if text in ("{", "}"):
        return None

    # C++11 alias declaration: `using Name = OtherType;`
    m = re.match(r"^using\s+([A-Za-z_]\w*)\s*=", text)
    if m:
        sig = re.sub(r"\s+", " ", text).strip()
        return Symbol(kind="typedef", name=m.group(1), signature=sig, line=0), None

    if text.startswith("typedef"):
        after = text[len("typedef") :].lstrip()
        first_word = re.match(r"[A-Za-z_]\w*", after)
        fw = first_word.group(0) if first_word else ""
        name = extract_trailing_name(text)
        if not name:
            return None
        if fw in ("struct", "union", "enum"):
            kind = fw
        else:
            kind = "typedef"
        sig = re.sub(r"[ \t]+", " ", text).strip()
        return Symbol(kind=kind, name=name, signature=sig, line=0), None

    # C++-style class/struct/union (no typedef prefix): `class Name : Base {
    # ... };` or `struct Name { ... };` or a bare forward declaration
    # `class Name;`. Recurse into class/struct bodies for member methods.
    m = CLASS_LIKE_RE.match(text)
    if m:
        keyword = m.group(1)
        tag = (m.group(2) or "").strip() or None
        brace_idx = text.find("{")
        if brace_idx == -1:
            if text.endswith(";") and tag:
                sig = re.sub(r"\s+", " ", text).strip()
                return Symbol(kind=keyword, name=tag, signature=sig, line=0), None
        elif tag:
            close_idx = find_matching_brace(text, brace_idx)
            header = re.sub(r"\s+", " ", text[:brace_idx]).strip()
            sym = Symbol(kind=keyword, name=tag, signature=header, line=0)
            inner = (brace_idx, close_idx) if (close_idx and keyword in ("class", "struct")) else None
            return sym, inner

    # Function decl/def: an identifier immediately followed by '(' at
    # top (chunk-relative) nesting depth 0, appearing before any '{'.
    # Attribute annotations (`__attribute__((...))`, `__declspec(...)`)
    # sometimes precede the real declarator; skip over those parenthesized
    # groups rather than mistaking the annotation for the function name.
    brace_idx = text.find("{")
    head = text if brace_idx == -1 else text[:brace_idx]
    depth = 0
    func_name = None
    paren_pos = None
    idx = 0
    n_head = len(head)
    while idx < n_head:
        ch = head[idx]
        if ch == "(":
            if depth == 0:
                m2 = re.search(r"(~?[A-Za-z_]\w*)\s*$", head[:idx])
                ident = m2.group(1) if m2 else None
                if ident and ident not in ATTRIBUTE_IDENTS:
                    func_name = ident
                    paren_pos = idx
                    break
                skip_depth = 1
                idx += 1
                while idx < n_head and skip_depth > 0:
                    if head[idx] == "(":
                        skip_depth += 1
                    elif head[idx] == ")":
                        skip_depth -= 1
                    idx += 1
                continue
            depth += 1
        elif ch == ")":
            depth -= 1
        idx += 1
    if func_name and paren_pos is not None:
        if brace_idx == -1:
            sig = text
        else:
            sig = head.rstrip() + " { ... }"
        sig = re.sub(r"[ \t]+", " ", sig).strip()
        return Symbol(kind="function", name=func_name, signature=sig, line=0), None

    # Fall back: plain variable-ish declaration `TYPE name;` or `TYPE name = init;`
    if text.endswith(";") and "(" not in text:
        name = extract_trailing_name(text)
        if name:
            sig = re.sub(r"[ \t]+", " ", text).strip()
            return Symbol(kind="variable", name=name, signature=sig, line=0), None

    return None


# --------------------------------------------------------------------------- #
#  Doxygen comment body parsing
# --------------------------------------------------------------------------- #


INLINE_REF_RE = re.compile(r"@(p|c|ref)\s+(\S+)")


def clean_inline(text: str) -> str:
    text = INLINE_REF_RE.sub(lambda m: f"`{m.group(2)}`", text)
    return text


def parse_doc_comment(raw: str) -> tuple[str, list[tuple[str, str]], str, list[str]]:
    lines = raw.split("\n")
    cleaned = []
    for l in lines:
        l = l.strip()
        if l.startswith("*"):
            l = l[1:]
        cleaned.append(l.strip())

    brief_lines: list[str] = []
    params: list[list[str]] = []
    return_lines: list[str] = []
    notes: list[str] = []

    current = ("brief", None)
    tag_re = re.compile(r"^@(\w+)\s*(.*)$")

    for l in cleaned:
        m = tag_re.match(l)
        if m and m.group(1) in DOXY_TAGS:
            tag, rest = m.group(1), m.group(2)
            if tag == "brief":
                brief_lines.append(rest)
                current = ("brief", None)
            elif tag == "param":
                pm = re.match(r"(?:\[[a-zA-Z, ]+\]\s*)?(\S+)\s*(.*)", rest)
                if pm:
                    params.append([pm.group(1), pm.group(2)])
                else:
                    params.append([rest, ""])
                current = ("param", len(params) - 1)
            elif tag in ("return", "returns"):
                return_lines.append(rest)
                current = ("return", None)
            elif tag in ("note", "warning", "deprecated", "see"):
                label = tag.capitalize()
                notes.append(f"{label}: {rest}".strip())
                current = ("note", len(notes) - 1)
        elif m:
            # Recognized-but-unhandled Doxygen tag (@file, @author, @ingroup,
            # @date, ...) — don't let its trailing text or continuation
            # lines bleed into the brief.
            current = ("skip", None)
        else:
            if not l and current[0] != "param":
                continue
            kind, idx = current
            if kind == "skip":
                continue
            if kind == "brief":
                brief_lines.append(l)
            elif kind == "param" and idx is not None:
                if l:
                    params[idx][1] = (params[idx][1] + " " + l).strip()
            elif kind == "return":
                return_lines.append(l)
            elif kind == "note" and idx is not None:
                if l:
                    notes[idx] = (notes[idx] + " " + l).strip()

    brief = clean_inline(" ".join(x for x in brief_lines if x).strip())
    brief = re.sub(r"\s+", " ", brief)
    return_desc = clean_inline(" ".join(x for x in return_lines if x).strip())
    return_desc = re.sub(r"\s+", " ", return_desc)
    clean_params = [(n, re.sub(r"\s+", " ", clean_inline(d)).strip()) for n, d in params]
    clean_notes = [re.sub(r"\s+", " ", clean_inline(n)).strip() for n in notes]
    return brief, clean_params, return_desc, clean_notes


# --------------------------------------------------------------------------- #
#  Per-file parse
# --------------------------------------------------------------------------- #


MAX_NEST_DEPTH = 4


FILE_TAG_RE = re.compile(r"^\s*\*?\s*@file\b", re.M)


def parse_header(path: Path) -> tuple[list[Symbol], str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    code_only, comments = scan_comments_and_strings(src)

    code_scan = list(code_only)
    strip_wrapper_braces(code_scan)
    code_scan_str = "".join(code_scan)

    directive_chunks = find_directive_chunks(code_scan_str)

    code_no_directives = list(code_scan_str)
    for ch in directive_chunks:
        blank_span(code_no_directives, ch.start, ch.end)
    code_no_directives_str = "".join(code_no_directives)

    # Gap-check buffer: comments blanked + extern "C"/namespace wrappers
    # blanked + only the *non-symbol* directive spans blanked (symbol-
    # producing chunks stay as "code" so association correctly stops there).
    gap_buf = list(code_scan_str)

    # `@file` blocks document the whole header, not the declaration that
    # happens to follow them — pull the first one out as a file overview so
    # it isn't misattributed to the first symbol in the file.
    file_brief = ""
    file_comment_id = None
    for c in comments:
        if c.is_doc and FILE_TAG_RE.search(c.text):
            file_brief, _, _, _ = parse_doc_comment(c.text)
            file_comment_id = id(c)
            break

    symbols: list[Symbol] = []
    doc_comments = [c for c in comments if c.is_doc and id(c) != file_comment_id]

    line_starts = [m.start() for m in re.finditer(r"\n", src)]

    def line_of(offset: int) -> int:
        lo, hi = 0, len(line_starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if line_starts[mid] < offset:
                lo = mid + 1
            else:
                hi = mid
        return lo + 1

    def attach_doc(sym: Symbol, chunk_start: int) -> None:
        best: Comment | None = None
        for c in doc_comments:
            if c.end <= chunk_start:
                best = c
            else:
                break
        if best is None:
            return
        gap = "".join(gap_buf[best.end : chunk_start])
        if gap.strip() != "":
            return
        brief, params, returns, notes = parse_doc_comment(best.text)
        sym.brief = brief
        sym.params = params
        sym.returns = returns
        sym.notes = notes
        sym.documented = bool(brief or params or returns or notes)

    def process(chunks: list[Chunk], prefix: str, depth: int) -> None:
        for chunk in sorted(chunks, key=lambda c: c.start):
            text = code_only[chunk.start : chunk.end]
            result = classify_chunk(text)
            if result is None:
                if chunk.is_directive:
                    blank_span(gap_buf, chunk.start, chunk.end)
                continue
            sym, inner = result
            base_name = sym.name
            sym.name = prefix + sym.name
            sym.line = line_of(chunk.start)
            attach_doc(sym, chunk.start)
            symbols.append(sym)
            if inner is not None and depth < MAX_NEST_DEPTH:
                open_idx, close_idx = inner
                inner_start = chunk.start + open_idx + 1
                inner_end = chunk.start + close_idx
                if inner_end > inner_start:
                    nested = find_code_chunks(code_no_directives_str[inner_start:inner_end])
                    shifted = [Chunk(c.start + inner_start, c.end + inner_start, False) for c in nested]
                    process(shifted, f"{prefix}{base_name}::", depth + 1)

    top_level = directive_chunks + find_code_chunks(code_no_directives_str)
    process(top_level, "", 0)

    symbols.sort(key=lambda s: s.line)
    return symbols, file_brief


# --------------------------------------------------------------------------- #
#  Markdown rendering
# --------------------------------------------------------------------------- #


def md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def rel_header_path(header: Path) -> Path:
    return header.relative_to(INC_ROOT)


def out_md_path(header: Path) -> Path:
    rel = rel_header_path(header)
    return OUT_ROOT / rel.with_suffix(".md")


KIND_LABEL = {
    "function": "Function",
    "struct": "Struct",
    "union": "Union",
    "enum": "Enum",
    "typedef": "Typedef",
    "macro": "Macro",
    "variable": "Variable",
}


def render_header_page(header: Path, symbols: list[Symbol], file_brief: str = "") -> str:
    rel = rel_header_path(header).as_posix()
    documented = [s for s in symbols if s.documented]
    undocumented = [s for s in symbols if not s.documented]

    out = []
    out.append(f"# `eshkol/{rel}`\n")
    if file_brief:
        out.append(f"{file_brief}\n")
    out.append(
        f"{len(symbols)} public symbol(s) — {len(documented)} documented, "
        f"{len(undocumented)} undocumented.\n"
    )
    out.append("Generated by `scripts/gen_api_docs.py`. Do not edit by hand.\n")

    if documented:
        out.append("## Symbols\n")
        for sym in documented:
            out.append(f"### `{sym.name}`\n")
            out.append(f"*{KIND_LABEL.get(sym.kind, sym.kind)}* — line {sym.line}\n")
            # Comment stripping preserves source columns with spaces; never
            # carry those padding bytes into generated Markdown where they
            # become trailing-whitespace churn and fail the release diff gate.
            clean_signature = "\n".join(
                line.rstrip() for line in sym.signature.splitlines()
            )
            out.append(f"```c\n{clean_signature}\n```\n")
            if sym.brief:
                out.append(f"{sym.brief}\n")
            if sym.params:
                out.append("**Parameters**\n")
                for pname, pdesc in sym.params:
                    if pdesc:
                        out.append(f"- `{pname}` — {pdesc}")
                    else:
                        out.append(f"- `{pname}`")
                out.append("")
            if sym.returns:
                out.append("**Returns**\n")
                out.append(f"{sym.returns}\n")
            if sym.notes:
                for note in sym.notes:
                    out.append(f"> {note}\n")

    if undocumented:
        out.append("## Undocumented\n")
        out.append("| Symbol | Kind | Line |")
        out.append("|---|---|---:|")
        for sym in undocumented:
            out.append(f"| `{md_escape(sym.name)}` | {KIND_LABEL.get(sym.kind, sym.kind)} | {sym.line} |")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def render_readme(per_file: list[tuple[Path, list[Symbol]]]) -> str:
    total = sum(len(s) for _, s in per_file)
    total_doc = sum(len([x for x in s if x.documented]) for _, s in per_file)
    total_undoc = total - total_doc
    pct = (100.0 * total_doc / total) if total else 0.0

    by_subsystem: dict[str, list[tuple[Path, list[Symbol]]]] = {}
    for header, symbols in per_file:
        rel = rel_header_path(header)
        subsystem = rel.parent.as_posix() if rel.parent != Path(".") else "(root)"
        by_subsystem.setdefault(subsystem, []).append((header, symbols))

    out = []
    out.append("# Eshkol API Reference\n")
    out.append(
        "Generated from the Doxygen `/** ... */` comment blocks in the public "
        "headers under `inc/eshkol/**/*.h`. Do not edit files under `docs/api/` "
        "by hand — regenerate with:\n"
    )
    out.append("```sh\npython3 scripts/gen_api_docs.py\n```\n")
    out.append(
        f"**Coverage:** {total_doc}/{total} public symbols documented "
        f"({pct:.1f}%), {total_undoc} undocumented.\n"
    )
    out.append("See also [INDEX.md](INDEX.md) for an alphabetical symbol table.\n")
    out.append("## Subsystems\n")

    for subsystem in sorted(by_subsystem.keys()):
        entries = sorted(by_subsystem[subsystem], key=lambda e: e[0].name)
        sub_total = sum(len(s) for _, s in entries)
        sub_doc = sum(len([x for x in s if x.documented]) for _, s in entries)
        out.append(f"### `{subsystem}/`\n" if subsystem != "(root)" else "### (root headers)\n")
        out.append(f"{sub_doc}/{sub_total} symbols documented.\n")
        out.append("| Header | Symbols | Documented |")
        out.append("|---|---:|---:|")
        for header, symbols in entries:
            link = out_md_path(header).relative_to(OUT_ROOT).as_posix()
            doc_count = len([x for x in symbols if x.documented])
            out.append(
                f"| [`{rel_header_path(header).as_posix()}`]({link}) | {len(symbols)} | {doc_count} |"
            )
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def render_index(per_file: list[tuple[Path, list[Symbol]]]) -> str:
    rows = []
    for header, symbols in per_file:
        link = out_md_path(header).relative_to(OUT_ROOT).as_posix()
        for sym in symbols:
            rows.append((sym.name, KIND_LABEL.get(sym.kind, sym.kind), rel_header_path(header).as_posix(), link, sym.documented))
    rows.sort(key=lambda r: (r[0].lower(), r[0], r[2]))

    out = []
    out.append("# Symbol Index\n")
    out.append(
        "Alphabetical index of every public symbol found in `inc/eshkol/**/*.h`. "
        "Generated by `scripts/gen_api_docs.py`.\n"
    )
    out.append(f"{len(rows)} total symbols.\n")
    out.append("| Symbol | Kind | Header | Documented |")
    out.append("|---|---|---|:---:|")
    for name, kind, header_rel, link, documented in rows:
        mark = "yes" if documented else "no"
        out.append(f"| [`{md_escape(name)}`]({link}) | {kind} | `{header_rel}` | {mark} |")
    out.append("")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--check", action="store_true", help="Check output is up to date; don't write.")
    args = ap.parse_args()

    inc_root = args.repo_root / "inc" / "eshkol"
    out_root = args.repo_root / "docs" / "api"

    global INC_ROOT, OUT_ROOT
    INC_ROOT = inc_root
    OUT_ROOT = out_root

    headers = sorted(inc_root.rglob("*.h"))
    if not headers:
        print(f"error: no headers found under {inc_root}", file=sys.stderr)
        return 2

    per_file: list[tuple[Path, list[Symbol]]] = []
    briefs: dict[Path, str] = {}
    for header in headers:
        symbols, file_brief = parse_header(header)
        per_file.append((header, symbols))
        briefs[header] = file_brief

    planned: dict[Path, str] = {}
    for header, symbols in per_file:
        planned[out_md_path(header)] = render_header_page(header, symbols, briefs[header])
    planned[out_root / "README.md"] = render_readme(per_file)
    planned[out_root / "INDEX.md"] = render_index(per_file)

    if args.check:
        drift = []
        for path, content in planned.items():
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                drift.append(path)
        existing_md = set(out_root.rglob("*.md")) if out_root.exists() else set()
        stale = existing_md - set(planned.keys())
        if drift or stale:
            for p in sorted(drift):
                print(f"stale/missing: {p.relative_to(args.repo_root)}", file=sys.stderr)
            for p in sorted(stale):
                print(f"orphaned: {p.relative_to(args.repo_root)}", file=sys.stderr)
            return 1
        print("docs/api/ is up to date.")
        return 0

    # Remove stale generated files from a previous run, then write fresh output.
    if out_root.exists():
        for p in sorted(out_root.rglob("*.md")):
            if p not in planned:
                p.unlink()
    for path, content in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    total = sum(len(s) for _, s in per_file)
    total_doc = sum(len([x for x in s if x.documented]) for _, s in per_file)
    print(f"Wrote {len(planned)} files to {out_root.relative_to(args.repo_root)}")
    print(f"Coverage: {total_doc}/{total} public symbols documented ({100.0*total_doc/total:.1f}%)" if total else "No symbols found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
