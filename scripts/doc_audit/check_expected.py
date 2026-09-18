#!/usr/bin/env python3
"""Third pass: compare `;; => value` annotations against what the build prints.

Every annotation in a block is attached to the form it describes, the block is
rewritten so that form's value (or its printed output) lands between two
sentinels on stdout, and the captured text is compared with the annotation.

ATTACHMENT

    (take '(a b c) 2)        ;; => (a b)      same line: the form(s) ending here
    (sort '(3 1 2) <)
    ;; => (1 2 3)                             own line: the form ending above it
    (display (f 1)) (newline) ;; => 3         a form that prints is left as it
                                              is and its stdout is captured
    (define xs (list 1 2))   ;; => (1 2)      a variable definition: the value
                                              it binds
    (display (f 1))
    (newline)
    ;; => 3                                   a bare `(newline)` above the
                                              annotation belongs to the form
                                              printed before it
    ;; prints: text                           same attachment, same comparison
    > (+ 1 2)
    3                                         a REPL transcript: see as_program

A form that prints nothing is wrapped in `display`, which is what the REPL
shows for a value. Consecutive annotation lines under one form are one
multi-line expectation. An annotation that follows a definition, or that has
no form above it, is UNATTACHED: reported, never dropped.

COMPARISON (`compare`)

The annotation names a VALUE, so the comparison is between data, not strings:

  * the whole annotation equal to the captured text is a match;
  * otherwise the first datum of the annotation is the value and the rest of
    the line is commentary: `;; => 8 ((3+1)*2)`, `;; => #t (no quote needed)`;
  * numbers compare by value. `12.0` matches a printed `12`: an integral
    double prints without a decimal point (docs/reference/language/
    numeric-tower.md, "Display convention"), and the page is right to show
    the reader that the value is inexact;
  * a string or character literal matches its `display` form: `"abc"` matches
    `abc`, `#\a` matches `a`;
  * `...` after a number means the page shortened it: `0.7616...` matches any
    printed number that truncates or rounds to those digits. Anywhere else
    `...` matches any run of characters (`/usr/local/bin:...`);
  * a leading `~` means approximately: every number within 1% or 1e-6.

Verdicts per annotation:
  MATCH       captured text agrees with the annotation
  MISMATCH    it does not                          <- the finding class
  NORUN       the block did not run far enough to print this value
  UNATTACHED  no expression to attach the annotation to
  PROSE       (audit pass only) the annotation is not a value literal

The audit entry point below (`main`) keeps the PROSE verdict for the public-doc
sweep. The example gate (`check_doc_examples.py`) does not: inside a gated
scope every `;; =>` is a checked claim.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_examples import balanced  # noqa: E402

TIMEOUT = 60
ANNOTATION_RE = re.compile(r"^;+\s*(?:=>|\u21d2|prints:)\s*(.*?)\s*$")
BEGIN_FMT = "@@DOCAUDIT:B%d@@"
END_FMT = "@@DOCAUDIT:E%d@@"
VALUE_MARK = "@@DOCAUDIT:VALUE@@"
CAPTURE_RE = re.compile(r"@@DOCAUDIT:B(\d+)@@(.*?)@@DOCAUDIT:E\1@@", re.S)
TEMP_NAME = "docaudit-captured-value"
TRANSCRIPT_PROMPT_RE = re.compile(r"^\s*(?:eshkol)?>\s+(\S.*)$")

# A form whose text calls one of these already writes to stdout; it is run as
# written and its output captured, instead of being wrapped in `display`.
OUTPUT_CALL_RE = re.compile(
    r"\((?:display|write|write-string|write-char|write-line|newline|print|println|printf|displayln)(?=[\s()])"
)
# Heads that make a top-level form a definition or a directive: there is no
# value to show for it, so an annotation under one cannot be attached.
NON_EXPRESSION_HEADS = frozenset({
    "define", "define-values", "define-syntax", "define-record-type", "define-type",
    "define-macro", "require", "import", "provide", "load", "include",
})


VALUE_RE = re.compile(
    r"^(#t|#f|#true|#false|'?\(.*\)|#\(.*\)|\"[^\"]*\"|#\\.|[-+]?[0-9][0-9a-zA-Z.eE+/_-]*|[-+]?\.[0-9]+)$"
)


def is_value(text):
    t = text.strip()
    # strip a trailing explanatory clause after two spaces or a comma
    return bool(VALUE_RE.match(t))


def normalise(text):
    t = text.strip()
    if t.startswith("'"):
        t = t[1:]
    return t


# ─────────────────────────── source scanning ───────────────────────────

def scan(code):
    """Scan one block of Scheme text.

    Returns (forms, comments, atoms):
      forms     [(start, end, depth)] for every parenthesised form; `start`
                includes a quote, quasiquote, unquote or `#` prefix, `end` is
                one past the closing bracket, depth 0 is top level
      comments  {line_index: offset of the `;` that starts the line comment}
      atoms     [(start, end)] for every top-level token outside any form
    Strings, character literals, `#|...|#` block comments and line comments
    never open or close a form.
    """
    forms, comments, atoms = [], {}, []
    stack = []
    i, n, line = 0, len(code), 0
    while i < n:
        ch = code[i]
        if ch == "\n":
            line += 1
            i += 1
        elif ch == ";":
            comments.setdefault(line, i)
            while i < n and code[i] != "\n":
                i += 1
        elif ch == "#" and code.startswith("#|", i):
            depth = 1
            i += 2
            while i < n and depth:
                if code.startswith("|#", i):
                    depth -= 1
                    i += 2
                elif code.startswith("#|", i):
                    depth += 1
                    i += 2
                else:
                    if code[i] == "\n":
                        line += 1
                    i += 1
        elif ch == '"':
            start = i
            i += 1
            while i < n and code[i] != '"':
                if code[i] == "\\":
                    i += 1
                if i < n and code[i] == "\n":
                    line += 1
                i += 1
            i += 1
            if not stack:
                atoms.append((start, min(i, n)))
        elif ch == "#" and code.startswith("#\\", i):
            start = i
            i += 3
            while i < n and not code[i].isspace() and code[i] not in "()[]\";":
                i += 1
            if not stack:
                atoms.append((start, i))
        elif ch in "([":
            start = i
            while start > 0 and code[start - 1] in "'`,@#":
                start -= 1
            stack.append(start)
            i += 1
        elif ch in ")]":
            if stack:
                forms.append((stack.pop(), i + 1, len(stack)))
            i += 1
        elif ch.isspace():
            i += 1
        else:
            start = i
            while i < n and not code[i].isspace() and code[i] not in "()[]\";":
                i += 1
            if not stack and code[start:i].strip("'`,@"):
                atoms.append((start, i))
    return forms, comments, atoms


_DEFINE_VARIABLE_RE = re.compile(r"^\(\s*define\s+([^\s()\[\]\";]+)\s")


def _head(form_text):
    m = re.match(r"^['`,@#]*[(\[]\s*([^\s()\[\]\";]+)", form_text)
    return m.group(1) if m else ""


def as_program(code):
    """Return `code` as a program the compiler can read.

    A block whose first line starts with a REPL prompt (`> ` or `eshkol> `) is
    a transcript: prompt lines are input, a line that continues an unbalanced
    input is more input, and every other line is what the REPL printed. The
    printed lines become `;; =>` annotations under the input above them, line
    for line, so a transcript is checked exactly like an annotated program.
    Left as it was, a transcript "runs" -- `>` is a procedure, `3` is a
    literal -- and proves nothing. Any other block is returned unchanged.
    """
    lines = code.split("\n")
    first = next((ln for ln in lines if ln.strip()), "")
    if not TRANSCRIPT_PROMPT_RE.match(first):
        return code
    out, pending = [], ""
    for ln in lines:
        m = TRANSCRIPT_PROMPT_RE.match(ln)
        if m:
            pending = m.group(1)
            out.append(m.group(1))
        elif pending and not balanced(pending) and ln.strip():
            text = re.sub(r"^\s*\.\.\.?\s?", "", ln) if ln.lstrip().startswith("..") else ln
            pending += "\n" + text
            out.append(text)
        elif ln.strip():
            pending = ""
            out.append(";; => " + ln.strip())
        else:
            out.append(ln)
    return "\n".join(out)


def instrument_block(code):
    """Return (new_code, expectations) for one block.

    Each expectation is a dict: `index`, `line` (0-based in the block), `text`
    (the annotation), `mode` (`value`, `prints` or `unattached`) and, for an
    unattached annotation, `why`. Attached expectations print their captured
    text between BEGIN_FMT % index and END_FMT % index.
    """
    code = as_program(code)
    lines = code.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    forms, comments, atoms = scan(code)

    def line_of(offset):
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def code_part(i):
        end = comments[i] - offsets[i] if i in comments else len(lines[i])
        return lines[i][:end]

    ends_on = {}
    for start, end, depth in forms:
        ends_on.setdefault(line_of(end - 1), []).append((start, end, depth))
    atoms_on = {}
    for start, end in atoms:
        atoms_on.setdefault(line_of(start), []).append((start, end))

    # Gather annotations; consecutive comment-only annotation lines are one
    # multi-line expectation attached to the same form.
    raw = []
    i = 0
    while i < len(lines):
        if i not in comments:
            i += 1
            continue
        m = ANNOTATION_RE.match(code[comments[i]:offsets[i] + len(lines[i])])
        if not m:
            i += 1
            continue
        if code_part(i).strip():
            raw.append({"line": i, "target": i, "text": m.group(1)})
            i += 1
            continue
        texts, first = [m.group(1)], i
        while i + 1 < len(lines) and (i + 1) in comments and not code_part(i + 1).strip():
            nxt = ANNOTATION_RE.match(code[comments[i + 1]:offsets[i + 1] + len(lines[i + 1])])
            if not nxt:
                break
            texts.append(nxt.group(1))
            i += 1
        target = first - 1
        while target >= 0 and not code_part(target).strip():
            if lines[target].strip():      # a plain comment between form and annotation
                break
            target -= 1
        if target >= 0 and not code_part(target).strip():
            target = -1
        raw.append({"line": first, "target": target, "text": "\n".join(texts)})
        i += 1

    inserts = []      # (offset, tie_break, text)
    expectations = []
    claimed = set()
    for index, ann in enumerate(raw):
        exp = {"index": index, "line": ann["line"], "text": ann["text"]}
        expectations.append(exp)
        target = ann["target"]
        if target in claimed:
            exp["mode"] = "unattached"
            exp["why"] = "the form above already carries an annotation"
            continue
        claimed.add(target)
        closing = ends_on.get(target, []) if target >= 0 else []
        begin = '(display "%s")' % (BEGIN_FMT % index)
        finish = '(display "%s")(newline)' % (END_FMT % index)
        if closing:
            depth = min(f[2] for f in closing)
            group = sorted(f for f in closing if f[2] == depth)
            # `(display x)` / `(newline)` / `;; => v`: the line above the
            # annotation only ends the output, so the form that produced it
            # is the sibling on the code line before.
            while all(re.sub(r"\s+", "", code[f[0]:f[1]]) == "(newline)" for f in group):
                above = line_of(group[0][0]) - 1
                while above >= 0 and not code_part(above).strip():
                    above -= 1
                earlier = sorted(f for f in ends_on.get(above, []) if f[2] == depth) if above >= 0 else []
                if not earlier or above in claimed:
                    break
                claimed.add(above)
                group = earlier + group
            text = code[group[0][0]:group[-1][1]]
            if OUTPUT_CALL_RE.search(text):
                exp["mode"] = "prints"
                inserts.append((group[0][0], depth, begin + " "))
                inserts.append((group[-1][1], -depth, " " + finish))
                continue
            start, end, _ = group[-1]
            bound = _DEFINE_VARIABLE_RE.match(code[start:end])
            if depth == 0 and bound:
                # `(define xs (list 1 2 3))  ;; => (1 2 3)` shows the value bound.
                exp["mode"] = "value"
                inserts.append((end, 0, " %s(display %s)%s" % (begin, bound.group(1), finish)))
                continue
            if depth == 0 and _head(code[start:end]) in NON_EXPRESSION_HEADS:
                exp["mode"] = "unattached"
                exp["why"] = "the annotation follows a `%s`, which has no value to show" % _head(code[start:end])
                continue
            # Evaluate first, then show the value: whatever the form itself
            # prints lands before VALUE_MARK, its value after. The `let` also
            # hands the value on, so an annotated form inside a body still
            # returns what it returned before.
            exp["mode"] = "value"
            inserts.append((start, depth, "%s(let ((%s " % (begin, TEMP_NAME) if depth == 0
                            else "(let ((%s " % TEMP_NAME))
            inserts.append((end, -depth,
                            ')) (display "%s")(display %s)%s %s)' % (VALUE_MARK, TEMP_NAME, finish, TEMP_NAME)
                            if depth == 0 else
                            ')) %s(display "%s")(display %s)%s %s)' % (begin, VALUE_MARK, TEMP_NAME, finish, TEMP_NAME)))
            continue
        line_atoms = atoms_on.get(target, []) if target >= 0 else []
        if line_atoms:
            start, end = line_atoms[-1]
            exp["mode"] = "value"
            inserts.append((start, 0, begin + "(display "))
            inserts.append((end, 0, ")" + finish))
            continue
        exp["mode"] = "unattached"
        exp["why"] = "no form ends on the line the annotation describes"

    out = code
    for offset, _tie, text in sorted(inserts, key=lambda t: (t[0], t[1]), reverse=True):
        out = out[:offset] + text + out[offset:]
    return out, expectations


def captured(stdout):
    """Return {expectation index: captured text} from an instrumented run."""
    return {int(m.group(1)): m.group(2).strip() for m in CAPTURE_RE.finditer(stdout)}


def instrument(code):
    """Return (new_code, [expected...]) for the attached annotations."""
    new_code, expectations = instrument_block(code)
    return new_code, [e["text"] for e in expectations if e["mode"] != "unattached"]


# ───────────────────────────── comparison ──────────────────────────────

_TOKEN_RE = re.compile(r'''\s*(#\(|[()\[\]]|"(?:\\.|[^"\\])*"|#\\[^\s()\[\]]+|[^\s()\[\]"]+)''')
_NUMBER_RE = re.compile(r"^[-+]?(?:\d+/\d+|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)$")
_ELLIPSIS_NUMBER_RE = re.compile(r"^([-+]?\d*\.?\d+)(?:\.\.\.|\u2026)$")
_CHAR_NAMES = {"space": " ", "newline": "\n", "tab": "\t", "nul": "\0", "null": "\0"}


def tokens(text):
    return _TOKEN_RE.findall(text)


def first_datum(text):
    """Return the text of the first complete datum in `text` (or all of it)."""
    toks, depth, used = tokens(text), 0, []
    for tok in toks:
        used.append(tok)
        if tok in ("(", "[", "#("):
            depth += 1
        elif tok in (")", "]"):
            depth -= 1
        if depth <= 0:
            break
    return " ".join(used)


def _number(tok):
    if not _NUMBER_RE.match(tok):
        return None
    if "/" in tok:
        num, den = tok.split("/")
        return Fraction(int(num), int(den)) if int(den) else None
    if re.match(r"^[-+]?\d+$", tok):
        return Fraction(int(tok))
    value = float(tok)
    return Fraction(value) if math.isfinite(value) else None


def _display_form(tok):
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), tok[1:-1])
    if tok.startswith("#\\") and len(tok) > 2:
        name = tok[2:]
        return _CHAR_NAMES.get(name, name)
    return {"#true": "#t", "#false": "#f"}.get(tok, tok)


def _shortened_number_matches(prefix, got):
    """`prefix...`: `got` truncates or rounds to the digits the page shows."""
    if _number(got) is None and not re.match(r"^[-+]?\d+$", got):
        return False
    if got.startswith(prefix):
        return True
    if "." in prefix and "e" not in got.lower():
        places = len(prefix.split(".")[1])
        try:
            return ("%.*f" % (places, float(got))) == ("%.*f" % (places, float(prefix)))
        except ValueError:
            return False
    return False


def _token_equal(exp, got, approx):
    short = _ELLIPSIS_NUMBER_RE.match(exp)
    if short:
        return _shortened_number_matches(short.group(1), got)
    a, b = _number(exp), _number(got)
    if a is not None and b is not None:
        if a == b:
            return True
        if approx:
            fa, fb = float(a), float(b)
            return abs(fa - fb) <= max(1e-6, 0.01 * max(abs(fa), abs(fb)))
        return False
    if exp == got:
        return True
    return _display_form(exp) == _display_form(got)


def _data_equal(exp, got, approx):
    et, gt = tokens(exp), tokens(got)
    if len(et) == len(gt) and all(_token_equal(a, b, approx) for a, b in zip(et, gt)):
        return True
    if len(et) == 1 and _display_form(et[0]) == got.strip():
        return True      # a string literal with spaces, shown by `display`
    return False


def _complex_parts(text):
    m = re.match(r"^([-+]?[\d.eE]+(?:[eE][-+]?\d+)?)?([-+][\d.]*(?:[eE][-+]?\d+)?)i$", text.replace(" ", ""))
    if not m:
        return None
    try:
        re_part = float(m.group(1)) if m.group(1) else 0.0
        im_text = m.group(2)
        im_part = float(im_text + "1") if im_text in ("+", "-") else float(im_text)
    except ValueError:
        return None
    return re_part, im_part


def satisfied(expected, capture):
    """Return (ok, shown) for one captured region of an instrumented run.

    A value-mode capture is `<printed>VALUE_MARK<value>`. A form that printed
    nothing is judged by its value. A form that printed (a call to a procedure
    that calls `display`, a `for-each`) is judged by what it printed, by that
    plus its value, or by the value alone: `(greet)` printing `Hello` and
    returning nothing satisfies `Hello`.
    """
    if VALUE_MARK not in capture:
        return compare(expected, capture), capture
    printed, value = (part.strip() for part in capture.split(VALUE_MARK, 1))
    if not printed:
        return compare(expected, value), value
    for candidate in (printed, printed + value, value):
        if compare(expected, candidate):
            return True, candidate
    return False, printed if value in ("", "()") else printed + value


def output_matches(expected, got):
    """A pasted output block against a program's stdout: equal text, or the
    same number of lines, each satisfying `compare` or equal token for token
    with numbers compared by value (`... = 5.0` against a printed `... = 5`)."""
    exp_lines = [ln.rstrip() for ln in expected.rstrip("\n").split("\n")]
    got_lines = [ln.rstrip() for ln in got.rstrip("\n").split("\n")]
    if exp_lines == got_lines:
        return True
    return len(exp_lines) == len(got_lines) and all(
        (not e.strip() and not g.strip()) or compare(e, g) or _data_equal(e, g, False)
        for e, g in zip(exp_lines, got_lines))


def compare(expected, got):
    """Return True when captured text `got` satisfies annotation `expected`."""
    exp, out = normalise(expected), normalise(got)
    if exp == out:
        return True
    approx = exp.startswith("~")
    if approx:
        exp = exp[1:].strip()
    value = first_datum(exp)
    if not value:
        return False
    if _data_equal(value, out, approx):
        return True
    if approx:
        a, b = _complex_parts(value), _complex_parts(out)
        if a and b and all(abs(x - y) <= max(1e-6, 0.01 * max(abs(x), abs(y))) for x, y in zip(a, b)):
            return True
    for candidate in (exp, value):
        if ("..." in candidate or "\u2026" in candidate) and not _ELLIPSIS_NUMBER_RE.match(candidate):
            pattern = ".*".join(re.escape(part) for part in re.split(r"\.\.\.|\u2026", candidate))
            if re.match("^" + pattern + "$", out, re.S):
                return True
    return False


def main():
    examples = json.load(open(sys.argv[1]))
    eshkol_run = sys.argv[2]
    out_path = sys.argv[3]
    scratch = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".scratch")
    os.makedirs(scratch, exist_ok=True)
    work = tempfile.mkdtemp(prefix="docaudit-exp-", dir=scratch)
    findings = []
    stats = {"blocks_with_annotations": 0, "pairs": 0, "MATCH": 0, "MISMATCH": 0,
             "PROSE": 0, "NORUN": 0, "UNATTACHED": 0}
    for n, e in enumerate(examples):
        code, expectations = instrument_block(e["code"])
        if not expectations:
            continue
        stats["blocks_with_annotations"] += 1
        d = os.path.join(work, "%d" % n)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "ex.esk")
        with open(src, "w") as fh:
            fh.write(code + "\n")
        env = dict(os.environ)
        env["ESHKOL_JIT_CACHE"] = "0"
        try:
            p = subprocess.run([eshkol_run, "-r", src], cwd=d, env=env,
                               capture_output=True, text=True, timeout=TIMEOUT)
            rc, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            rc, so, se = -9, "", "TIMEOUT"
        printed = captured(so)
        for exp in expectations:
            stats["pairs"] += 1
            got = printed.get(exp["index"])
            if exp["mode"] == "unattached":
                verdict = "UNATTACHED"
            elif not is_value(first_datum(normalise(exp["text"]).lstrip("~"))):
                verdict = "PROSE"
            elif got is None:
                verdict = "NORUN"
            elif satisfied(exp["text"], got)[0]:
                verdict = "MATCH"
            else:
                verdict = "MISMATCH"
            stats[verdict] += 1
            if verdict in ("MISMATCH", "NORUN"):
                findings.append({
                    "file": e["file"], "start_line": e["start_line"],
                    "expected": exp["text"], "got": got, "verdict": verdict,
                    "exit": rc, "stderr": se[-600:],
                })
        if (n + 1) % 100 == 0:
            print("... %d/%d" % (n + 1, len(examples)), file=sys.stderr)
    with open(out_path, "w") as fh:
        json.dump({"stats": stats, "findings": findings}, fh, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
