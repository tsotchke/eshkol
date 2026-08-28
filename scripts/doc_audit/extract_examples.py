#!/usr/bin/env python3
"""Extract fenced Scheme/Eshkol code blocks from the public docs with provenance.

Emits a JSON array of records:
  {file, start_line, end_line, lang, code, klass, expects}

klass is a first-pass classification:
  runnable      - a whole program: at least one top-level form, no obvious
                  placeholder syntax, not a REPL transcript
  needs-context - REPL transcript, partial form, or references identifiers the
                  block itself does not define
  illustrative  - explicitly marked non-runnable, or contains placeholder/ellipsis
                  syntax that cannot parse

expects is the list of `;; =>`, `;; returns`, and `;; prints` expected-value
annotations found inline.
"""

import json
import os
import re
import sys

SCOPE = [
    "README.md",
    "ANNOUNCEMENT.md",
    "docs/QUICKSTART.md",
    "docs/ESHKOL_LANGUAGE_GUIDE.md",
    "docs/ESHKOL_QUICK_REFERENCE.md",
    "docs/COMPLETE_LANGUAGE_SPECIFICATION.md",
    "docs/FAQ.md",
    "docs/STDLIB_V1_2_API.md",
    "docs/guide/AUTOMATIC_DIFFERENTIATION.md",
    "docs/reference",
]

LANGS = {"scheme", "eshkol", "lisp", "racket"}

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")

PLACEHOLDER = re.compile(r"(\.\.\.|<[a-z][a-z0-9 _-]*>|\bTODO\b|\bXXX\b)")
REPL_PROMPT = re.compile(r"^\s*(eshkol>|>>>|\$ )")
EXPECT_RE = re.compile(r";+\s*(?:=>|⇒|returns?:?|Returns?:?|prints?:?|Prints?:?)\s*(.+)$")


def closing_fence(lines, opening_index, opening_match):
    """Return the closing fence index for one opening fence, or ``None``.

    Markdown permits a longer fence to close a shorter one, but the fence
    character must match and a closing fence has no info string. Keeping this
    rule in one helper prevents the example extractor and output checker from
    disagreeing about where a block ends.
    """
    fence = opening_match.group(2)
    for index in range(opening_index + 1, len(lines)):
        candidate = FENCE_RE.match(lines[index])
        if (candidate and candidate.group(2)[0] == fence[0]
                and len(candidate.group(2)) >= len(fence)
                and not candidate.group(3)):
            return index
    return None


def iter_files(root):
    for item in SCOPE:
        p = os.path.join(root, item)
        if os.path.isdir(p):
            for dirpath, _dirnames, filenames in os.walk(p):
                for fn in sorted(filenames):
                    if fn.endswith(".md"):
                        yield os.path.relpath(os.path.join(dirpath, fn), root)
        elif os.path.isfile(p):
            yield item


def balanced(code):
    depth = 0
    in_str = False
    in_comment = False
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
            i += 1
            continue
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == ";":
            in_comment = True
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "#" and i + 1 < n and code[i + 1] == "\\":
            i += 3
            continue
        if ch == "(" or ch == "[":
            depth += 1
        elif ch == ")" or ch == "]":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def classify(code, preceding):
    stripped = "\n".join(
        ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith(";")
    )
    if not stripped.strip():
        return "illustrative"
    if any(REPL_PROMPT.match(ln) for ln in code.splitlines()):
        return "needs-context"
    low = preceding.lower()
    for marker in (
        "not implemented",
        "planned",
        "future",
        "conceptual",
        "pseudo",
        "sketch",
        "illustrat",
        "roadmap",
        "would look",
        "hypothetical",
        "does not compile",
        "for reference only",
    ):
        if marker in low:
            return "illustrative"
    if PLACEHOLDER.search(stripped):
        return "illustrative"
    if not balanced(code):
        return "needs-context"
    if not stripped.lstrip().startswith("("):
        return "needs-context"
    return "runnable"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out = []
    for rel in iter_files(root):
        path = os.path.join(root, rel)
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        i = 0
        n = len(lines)
        while i < n:
            m = FENCE_RE.match(lines[i])
            if not m:
                i += 1
                continue
            indent, fence, lang = m.group(1), m.group(2), m.group(3).lower()
            close = closing_fence(lines, i, m)
            if close is None:
                i += 1
                continue
            if lang in LANGS:
                body = lines[i + 1 : close]
                code = "\n".join(ln[len(indent):] if ln.startswith(indent) else ln for ln in body)
                preceding = "\n".join(lines[max(0, i - 6) : i])
                expects = []
                for k, ln in enumerate(body):
                    em = EXPECT_RE.search(ln)
                    if em:
                        expects.append({"line": i + 2 + k, "text": em.group(1).strip()})
                out.append(
                    {
                        "file": rel,
                        "start_line": i + 1,
                        "end_line": close + 1,
                        "lang": lang,
                        "code": code,
                        "klass": classify(code, preceding),
                        "expects": expects,
                    }
                )
            i = close + 1
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
