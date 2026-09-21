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

SCOPES

`SCOPE` is the audit sweep the pass scripts in this directory read. The
documentation example gate (`check_doc_examples.py`) reads `GATED_SCOPES`
instead: a named set of paths in which EVERY fenced example is executed and
its expectation compared on every build. Adding `docs/guide`, `docs/reference`
or the README samples to the gate is one more entry in `GATED_SCOPES` plus a
baseline refresh (`check_doc_examples.py --update-baseline`); nothing else in
the harness names a documentation path.

MARKERS

An example that cannot be executed says so in the markdown, in an HTML comment
on the line above its opening fence (blank lines between the two are allowed),
so the rendered page is unchanged and the reason sits next to the example:

    <!-- doc-example: skip <reason>: <why, in a sentence> -->
        not executed. <reason> is one of SKIP_REASONS.
    <!-- doc-example: run-only <reason>: <why> -->
        executed and must exit 0, but its printed values are not compared
        (the output depends on the machine or the moment).
    <!-- doc-example: known-defect <LEDGER-ID>: <what the page promises> -->
        the page states the designed behaviour and the implementation does
        not deliver it yet. <LEDGER-ID> names an open entry under
        .icc/ledger/entries/. The example is still executed: the day it
        passes, the gate fails until the marker is removed.

    <!-- doc-example: file <name>: <what it is> -->
        the block is also the content of `<name>`, which later examples on
        the page `require`, `load` or open. It is executed like any other
        example and written next to every later example on the page. Not an
        exclusion, so not ratcheted. On a fence of any language (```json,
        ```text) it only provides the file.

    <!-- doc-example: output stdout: <what it is> -->
        on a bare, ```text or ```output fence: the block is what the nearest
        scheme example above it prints, even with a heading or a rule between
        the two. (A bare fence directly under an example needs no marker.)
        Compared line by line under the annotation rules, so `0.0079...`
        and `5.0` for a printed `5` are both fine.

A comment that starts with `doc-example:` and does not parse is recorded with
kind `invalid`, which the gate reports as a failure: a typo in a marker must
never turn into an unmarked example or a silent skip.
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

# Named path sets whose examples are all executed by check_doc_examples.py.
# Extend this table to put more documentation under the gate.
GATED_SCOPES = {
    "tutorials": ["docs/tutorials"],
    # Guides join one page at a time, as each page's examples are brought
    # under the gate. A path may be a directory or a single file.
    "guides": ["docs/guide/GRADUAL_TYPING.md"],
    "upgrading": ["docs/UPGRADING.md"],
}

SKIP_REASONS = (
    "pseudo-code",        # a shape or a signature, not a program
    "fragment",           # part of a larger program shown elsewhere on the page
    "platform-specific",  # needs hardware, an OS or a toolchain the gate cannot assume
    "nondeterministic",   # output differs from run to run
    "interactive",        # reads from a terminal or waits for a user
    "external-resource",  # needs a network service, a credential or a file the page does not create
)
MARKER_KINDS = ("skip", "run-only", "known-defect", "file", "output")
MARKER_PREFIX_RE = re.compile(r"^\s*<!--\s*doc-example:")
MARKER_RE = re.compile(
    r"^\s*<!--\s*doc-example:\s*(skip|run-only|known-defect|file|output)\s+([A-Za-z0-9_.-]+)\s*:\s*(\S.*?)\s*-->\s*$"
)

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


def iter_files(root, scope=None):
    for item in (SCOPE if scope is None else scope):
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


def marker_for(lines, fence_index):
    """Return the `doc-example:` marker attached to the fence at `fence_index`.

    The marker is the nearest non-blank line above the fence. `None` means the
    example is unmarked. A `doc-example:` comment that does not parse, or that
    names an unknown skip reason, comes back with kind `invalid`.
    """
    k = fence_index - 1
    while k >= 0 and not lines[k].strip():
        k -= 1
    if k < 0 or not MARKER_PREFIX_RE.match(lines[k]):
        return None
    m = MARKER_RE.match(lines[k])
    if not m:
        return {"kind": "invalid", "line": k + 1, "token": "", "text": lines[k].strip(),
                "error": "expected `<!-- doc-example: skip|run-only|known-defect|file|output <token>: <text> -->`"}
    kind, token, text = m.group(1), m.group(2), m.group(3)
    if kind in ("skip", "run-only") and token not in SKIP_REASONS:
        return {"kind": "invalid", "line": k + 1, "token": token, "text": text,
                "error": "unknown reason %r; use one of %s" % (token, ", ".join(SKIP_REASONS))}
    if kind == "output" and token != "stdout":
        return {"kind": "invalid", "line": k + 1, "token": token, "text": text,
                "error": "an output marker reads `output stdout: ...`, not %r" % token}
    if kind == "file" and (token.startswith(".") or not re.match(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$", token)):
        return {"kind": "invalid", "line": k + 1, "token": token, "text": text,
                "error": "a file marker names a plain file name, not a path: %r" % token}
    return {"kind": kind, "line": k + 1, "token": token, "text": text}


def extract(root, scope=None):
    """Return one record per fenced example under `scope` (default: SCOPE)."""
    out = []
    for rel in iter_files(root, scope):
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
            marker = marker_for(lines, i)
            if lang not in LANGS and marker and marker["kind"] == "file":
                body = lines[i + 1 : close]
                out.append({
                    "file": rel, "start_line": i + 1, "end_line": close + 1, "lang": lang,
                    "code": "\n".join(ln[len(indent):] if ln.startswith(indent) else ln for ln in body),
                    "klass": "data-file", "expects": [], "marker": marker,
                })
            if lang in LANGS and marker and marker["kind"] == "output":
                marker = {"kind": "invalid", "line": marker["line"], "token": "stdout", "text": marker["text"],
                          "error": "an output marker belongs on the pasted output fence, not on an example"}
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
                        "marker": marker,
                    }
                )
            i = close + 1
    return out


def main():
    args = [a for a in sys.argv[1:]]
    scope = None
    if "--scope" in args:
        k = args.index("--scope")
        name = args[k + 1]
        if name not in GATED_SCOPES:
            raise SystemExit("unknown scope %r; known: %s" % (name, ", ".join(sorted(GATED_SCOPES))))
        scope = GATED_SCOPES[name]
        del args[k : k + 2]
    root = args[0] if args else "."
    json.dump(extract(root, scope), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
