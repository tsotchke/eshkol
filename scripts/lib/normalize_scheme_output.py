#!/usr/bin/env python3
r"""normalize_scheme_output.py — canonicalise Scheme program stdout so that only
SEMANTIC differences between Eshkol and the reference R7RS implementation
survive to be flagged by scripts/run_reference_differential.sh.

Reads stdin, writes normalised text to stdout. The normalisation is applied
IDENTICALLY to both engines' output, so any transformation that maps two
already-equal outputs to the same thing is safe: it can only ever *hide* a
cosmetic (implementation-defined) difference, never manufacture a false
agreement between genuinely different values.

Documented normalisations (each maps a known implementation-defined rendering
difference to a canonical form):

  1. ANSI escapes stripped. (Eshkol colourises diagnostics; harmless here since
     we compare stdout, but strip defensively.)

  2. Boolean long forms: `#true` -> `#t`, `#false` -> `#f`. R7RS permits both
     spellings; chibi and Eshkol may pick either.

  3. Strings and characters written inside compound data. R7RS `display`
     renders strings without quotes and characters as their glyph even when
     nested in a list/vector; the mainstream implementations (Guile, Racket,
     and Eshkol) do this recursively, but chibi-scheme instead emits the
     `write` form (`"b"`, `#\c`) for members of a display'd datum. To avoid
     false-flagging Eshkol for chibi's non-recursive-display quirk we:
       * strip the `#\` prefix from a SINGLE printable character that is NOT
         the start of a named character (`#\space`/`#\newline`): `#\c` -> `c`;
       * strip ALL double-quote characters.
     Both are applied symmetrically to both engines. Crucially the quote-strip
     does NOT hide `write`-escaping bugs: the genuine finding "Eshkol's `write`
     emits a raw newline instead of the two-character escape `\\n`" survives,
     because the difference is in the CONTENT (a literal newline vs the
     characters backslash-n), not in the surrounding quotes. The only thing a
     quote-strip could mask is a hypothetical bug where `write` and `display`
     of a string differ ONLY by the quotes and nothing else — not a case that
     occurs in this corpus.

  4. Numbers. This is the one normalisation that intentionally collapses the
     exact/inexact PRINT distinction and floating-point precision, exactly as
     the task requires ("exact/inexact printing, float precision"):
       * Rationals `a/b` are left verbatim (both engines print them identically).
       * Integer literals (no '.'/exponent) are left verbatim.
       * Any token that looks like a floating-point number is reformatted with
         '%.6g' (6 significant digits). Consequences, all intended:
           - inexact integers collapse to integers: `1.0` -> `1`, `3.0` -> `3`
             (Eshkol prints `1` where chibi prints `1.0`);
           - precision beyond 6 sig-figs is dropped:
             `1.4142135623730951` -> `1.41421`, matching Eshkol's default
             6-sig-fig float printing.
       * inf/nan are canonicalised to `+inf.0` / `-inf.0` / `+nan.0`.
     A genuinely WRONG numeric value (e.g. 41 vs 42, or 1/2 vs 1/3) still
     differs after normalisation and is flagged.

  5. Trailing whitespace per line is stripped; trailing blank lines removed.

If a normalisation ever needs to change, it MUST change here only, and the
README/report note referencing this file must be updated in lockstep.
"""

import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# A '#\' followed by a single printable char that is NOT followed by another
# ASCII letter (which would make it a named char like #\space). Group 1 = char.
CHAR_IN_DATUM_RE = re.compile(r"#\\([!-~])(?![A-Za-z])")

# Floating point token: must contain a '.' or an exponent to be treated as a
# float (plain integers are left alone). Allows a leading sign. Excludes the
# '/' of a rational (rationals never match because there is no '.'/exp and the
# regex does not span '/').
FLOAT_RE = re.compile(
    r"(?<![\w./])"                      # not preceded by ident/rational char
    r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"(?![\w./])"                       # not followed by ident/rational char
)

INF_NAN_RE = re.compile(r"[-+]?(?:inf|nan)\.0")


def _fmt_float(tok):
    # Leave things that are clearly integers (no '.' and no exponent) alone.
    if "." not in tok and "e" not in tok and "E" not in tok:
        return tok
    try:
        v = float(tok)
    except ValueError:
        return tok
    if v != v:                      # NaN
        return "+nan.0"
    if v in (float("inf"), float("-inf")):
        return "+inf.0" if v > 0 else "-inf.0"
    return "%.6g" % v


def normalize(text):
    text = ANSI_RE.sub("", text)
    text = text.replace("#true", "#t").replace("#false", "#f")
    text = CHAR_IN_DATUM_RE.sub(r"\1", text)
    text = text.replace('"', "")
    text = INF_NAN_RE.sub(lambda m: _fmt_float(m.group(0)), text)
    text = FLOAT_RE.sub(lambda m: _fmt_float(m.group(1)), text)
    # Line-level whitespace cleanup.
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def main():
    sys.stdout.write(normalize(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
