#!/usr/bin/env python3
"""Regression tests for the documentation example parser and runners."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SCRATCH = REPO_ROOT / ".scratch"
sys.path.insert(0, str(HERE))

import check_expected  # noqa: E402
import check_output_blocks  # noqa: E402
import extract_examples  # noqa: E402


def run() -> bool:
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="doc-audit-test-", dir=SCRATCH) as temp:
        root = Path(temp)
        (root / "README.md").write_text(
            "~~~scheme\n"
            "(display 1) ; => one\n"
            "(display 2) ; returns: two\n"
            "(display 3) ; prints three\n"
            "~~~\n"
            "\n"
            "~~~text\n"
            "1\n"
            "2\n"
            "3\n"
            "~~~\n",
            encoding="utf-8",
        )
        original_scope = extract_examples.SCOPE
        extract_examples.SCOPE = ["README.md"]
        try:
            lines = (root / "README.md").read_text(encoding="utf-8").splitlines()
            opening = extract_examples.FENCE_RE.match(lines[0])
            if opening is None:
                return False
            close = extract_examples.closing_fence(lines, 0, opening)
            expectation_lines = lines[1:close]
            expectations = [
                extract_examples.EXPECT_RE.search(line).group(1).strip()
                for line in expectation_lines
                if extract_examples.EXPECT_RE.search(line)
            ]
            pairs = check_output_blocks.collect(str(root))
        finally:
            extract_examples.SCOPE = original_scope

    return (
        close == 4
        and expectations == ["one", "two", "three"]
        and len(pairs) == 1
        and pairs[0]["expected"] == "1\n2\n3"
    )


def check_annotations() -> list[str]:
    """Attachment and comparison rules the example gate depends on."""
    failures = []
    code = "\n".join([
        "(define xs (list 1 2))      ;; => (1 2)",
        "(define (f x) (* x 2))      ;; => doubles",
        "(f 2)                       ;; => 4",
        "(display (f 3))",
        "(newline)",
        ";; => 6",
        "(let ((y 1))",
        "  (+ y 1))   ;; => 2",
        "(f 5)",
        ";; an ordinary comment",
        ";; => 10",
    ])
    program, expectations = check_expected.instrument_block(code)
    modes = [(e["text"], e["mode"]) for e in expectations]
    wanted = [("(1 2)", "value"), ("doubles", "unattached"), ("4", "value"), ("6", "prints"),
              ("2", "value"), ("10", "unattached")]
    if modes != wanted:
        failures.append("attachment: %r" % (modes,))
    if not extract_examples.balanced(program):
        failures.append("instrumented program is not balanced")
    if program.count(check_expected.BEGIN_FMT % 3) != 1 or "(display (f 3))" not in program:
        failures.append("a printing form must be left as written")

    transcript = "> (define (greet) (display \"hi\"))\n> (greet)\nhi\n> (+ 1\n     2)\n3"
    as_program = check_expected.as_program(transcript)
    if as_program.split("\n") != ["(define (greet) (display \"hi\"))", "(greet)", ";; => hi",
                                  "(+ 1", "     2)", ";; => 3"]:
        failures.append("transcript: %r" % as_program)
    if check_expected.as_program("(> 2 1)\n3") != "(> 2 1)\n3":
        failures.append("a block that merely starts with a parenthesis is not a transcript")

    table = [
        ("12.0", "12", True), ("12.0", "13", False), ("0.5", "0.5000001", False),
        ("0.7616...", "0.7615941559557649", True), ("0.7616...", "0.7715941559557649", False),
        ("9332621544...", "93326215443944152681", True),
        ("\"HELLO\"", "HELLO", True), ("#\\e", "e", True), ("\"hello world\"", "hello world", True),
        ("8 ((3+1)*2)", "8", True), ("#t  (bare ?x, no quote)", "#t", True),
        ("five is greater", "five is greater", True), ("gradient vector", "#(1 2)", False),
        ("(1 2 3) (fresh copy)", "(1 2 3)", True), ("(2 3)", "(2 3 4)", False),
        ("(\"apple\" \"banana\")", "(apple banana)", True), ("#(4.0 5.0)", "#(4 5)", True),
        ("~2.0 (converges)", "1.9999", True), ("~2.0", "2.5", False),
        ("~0+2i", "1.2246467991473532e-16+2i", True),
        ("/usr/local/bin:...", "/usr/local/bin:/usr/bin", True),
        ("#(0.6931...)", "#(0.6931471805599453)", True),
        ("Minimum at: (0.994..., 0.988...)", "Minimum at: (0.9944009477, 0.9888107640)", True),
        ("Minimum at: (0.994..., 0.988...)", "Minimum at: (19.99, 0)", False),
    ]
    for expected, got, want in table:
        if check_expected.compare(expected, got) is not want:
            failures.append("compare(%r, %r) should be %s" % (expected, got, want))

    mark = check_expected.VALUE_MARK
    for expected, capture, want in [("Hello", "Hello" + mark + "()", True), ("4", mark + "4", True),
                                    ("4", mark + "5", False), ("x5", "x" + mark + "5", True)]:
        if check_expected.satisfied(expected, capture)[0] is not want:
            failures.append("satisfied(%r, %r) should be %s" % (expected, capture, want))
    if not check_expected.output_matches("a = 5.0\nb", "a = 5\nb") or check_expected.output_matches("a = 5.0", "a = 6"):
        failures.append("output_matches compares numbers by value and nothing else loosely")
    return failures


def check_markers() -> list[str]:
    """Marker parsing, the `file` and `output` markers, and the gated scopes."""
    failures = []
    page = "\n".join([
        "<!-- doc-example: skip pseudo-code: a shape -->", "```scheme", "(f ...)", "```", "",
        "<!-- doc-example: skip because: no such reason -->", "```scheme", "(g)", "```", "",
        "<!-- doc-example: skipped -->", "```scheme", "(h)", "```", "",
        "<!-- doc-example: known-defect SW-1: what the page promises -->", "", "```scheme", "(i)", "```", "",
        "<!-- doc-example: file data.json: input -->", "```json", "{}", "```", "",
        "<!-- doc-example: file ../up.esk: escapes -->", "```scheme", "(j)", "```", "",
        "```scheme", "(display 1)", "```", "", "## Expected output", "",
        "<!-- doc-example: output stdout: what it prints -->", "```", "1", "```", "",
    ])
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="doc-audit-markers-", dir=SCRATCH) as temp:
        root = Path(temp)
        (root / "PAGE.md").write_text(page, encoding="utf-8")
        records = extract_examples.extract(str(root), ["PAGE.md"])
        pairs = check_output_blocks.collect(str(root), ["PAGE.md"])
    kinds = [(r["klass"] == "data-file", (r["marker"] or {}).get("kind"), (r["marker"] or {}).get("token"))
             for r in records]
    wanted = [(False, "skip", "pseudo-code"), (False, "invalid", "because"), (False, "invalid", ""),
              (False, "known-defect", "SW-1"), (True, "file", "data.json"), (False, "invalid", ""),
              (False, None, None)]
    if kinds != wanted:
        failures.append("markers: %r" % (kinds,))
    if [(p["code"], p["expected"]) for p in pairs] != [("(display 1)", "1")]:
        failures.append("marked output block is paired with the example above it: %r" % (pairs,))
    for name, paths in extract_examples.GATED_SCOPES.items():
        for rel in paths:
            if not (REPO_ROOT / rel).exists():
                failures.append("gated scope %s names %s, which does not exist" % (name, rel))
    return failures


def main() -> int:
    ok = run()
    print("doc_audit_harness: %s" % ("PASS" if ok else "FAIL"))
    for label, failures in (("annotations", check_annotations()), ("markers", check_markers())):
        print("doc_audit_%s: %s" % (label, "PASS" if not failures else "FAIL"))
        for failure in failures:
            print("  " + failure)
        ok = ok and not failures
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
