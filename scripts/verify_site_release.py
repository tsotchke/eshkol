#!/usr/bin/env python3
"""Verify that the public website agrees with the release's sources of truth.

The website (site/src/main.esk, compiled to the checked-in
site/static/eshkol-site.wasm) states release facts: the tag, the "what's new"
heading, download URLs and asset names, and the figures on the homepage.  This
gate fails when any of them disagrees with the file that owns it:

  release tag, status  tests/coverage/release_record.json
  asset names          the release workflow's validated asset set
                       (.github/workflows/release.yml, "Validate Release Asset Set")
  homepage figures     HOMEPAGE_CLAIMS below: each figure is bound to the file
                       that measures or documents it, and a digit in homepage
                       prose that no claim covers fails as unsourced
  published docs       site/pages.json via scripts/build_site_content.py

main.esk names the release in one block (BEGIN/END RELEASE FACTS).  The gate
evaluates that block, requires its tag to equal the record, requires every
download URL it derives to name a published asset, and fails if a release tag
or download URL is written anywhere else in the file.

Usage:
    python3 scripts/verify_site_release.py               # the repository
    python3 scripts/verify_site_release.py --root DIR    # another checkout
    python3 scripts/verify_site_release.py --self-test   # red/green fixtures
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_RECORD = "tests/coverage/release_record.json"
SITE_SOURCE = "site/src/main.esk"
SITE_WASM = "site/static/eshkol-site.wasm"
FACTS_BEGIN = ";; BEGIN RELEASE FACTS"
FACTS_END = ";; END RELEASE FACTS"
DOWNLOAD_PREFIX = "https://github.com/tsotchke/eshkol/releases/download/"
CHECKSUMS_ASSET = "SHA256SUMS.txt"
TAG_RE = re.compile(r"\bv\d+\.\d+\.\d+(?:-[a-z]+)?\b")
DOWNLOAD_URL_RE = re.compile(re.escape(DOWNLOAD_PREFIX) + r"(?P<tag>[^/\s\"'<]+)/(?P<asset>[^\s\"'<\\]+)")
ASSET_NAME_RE = re.compile(r"\beshkol-v\d[^\s\"'<\\/]*?\.(?:tar\.gz|zip)\b")
# Facts a page must render; each must be referenced outside the facts block.
RENDERED_FACTS = (
    "release-whats-new-heading",
    "release-publication-heading",
    "release-downloads-summary",
    "release-linux-install-html",
    "release-vm-banner-html",
)


class SiteCheckError(Exception):
    """A structural problem that stops the check (unparseable input)."""


def fail(message: str) -> None:
    raise SiteCheckError(message)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label} is missing {needle!r}")


# ---------------------------------------------------------------------------
# Release record and asset set
# ---------------------------------------------------------------------------

def load_release_tag(root: Path) -> str:
    try:
        record = json.loads((root / RELEASE_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"{RELEASE_RECORD} is unreadable: {exc}")
    tag = record.get("tag")
    if not isinstance(tag, str) or not TAG_RE.fullmatch(tag):
        fail(f"{RELEASE_RECORD} has no usable 'tag'")
    return tag


def release_assets(workflow: str, tag: str) -> list[str]:
    match = re.search(
        r"Validate Release Asset Set.*?expected=\(\n(?P<body>.*?)\n\s*\)",
        workflow,
        re.DOTALL,
    )
    if not match:
        fail("could not find the release workflow's expected asset array")
    assets = re.findall(r'"(eshkol-\$\{RELEASE_TAG\}-[^\"]+)"', match.group("body"))
    if not assets:
        fail("release workflow expected asset array is empty")
    return [asset.replace("${RELEASE_TAG}", tag) for asset in assets]


def matrix_name(asset: str, tag: str) -> str:
    prefix = f"eshkol-{tag}-"
    if not asset.startswith(prefix):
        fail(f"unexpected release asset name: {asset}")
    name = asset[len(prefix):]
    for suffix in (".tar.gz", ".zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    fail(f"unexpected release asset extension: {asset}")
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# The RELEASE FACTS block of main.esk
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'\s+|;[^\n]*|(?P<open>\()|(?P<close>\))|"(?P<str>(?:[^"\\]|\\.)*)"|(?P<sym>[^\s()";]+)')
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def _unescape(body: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), body)


def _read_forms(text: str) -> list:
    stack: list[list] = [[]]
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            fail(f"release facts block: cannot read {text[pos:pos + 30]!r}")
        pos = m.end()
        if m.group("open"):
            stack.append([])
        elif m.group("close"):
            if len(stack) == 1:
                fail("release facts block: unbalanced ')'")
            done = stack.pop()
            stack[-1].append(done)
        elif m.group("str") is not None:
            stack[-1].append(("str", _unescape(m.group("str"))))
        elif m.group("sym"):
            stack[-1].append(("sym", m.group("sym")))
    if len(stack) != 1:
        fail("release facts block: unbalanced '('")
    return stack[0]


def split_release_facts(source: str) -> tuple[str, str, str]:
    """(before, block, after) around the RELEASE FACTS markers."""
    if source.count(FACTS_BEGIN) != 1 or source.count(FACTS_END) != 1:
        fail(f"{SITE_SOURCE} must contain exactly one '{FACTS_BEGIN}' ... '{FACTS_END}' block")
    a = source.index(FACTS_BEGIN)
    b = source.index(FACTS_END)
    if b < a:
        fail("release facts block markers are out of order")
    return source[:a], source[a + len(FACTS_BEGIN):b], source[b + len(FACTS_END):]


def evaluate_release_facts(block: str) -> dict[str, str]:
    """Evaluate (define name "lit") / (define name (string-append ...))."""
    facts: dict[str, str] = {}

    def value(expr) -> str:
        if isinstance(expr, tuple) and expr[0] == "str":
            return expr[1]
        if isinstance(expr, tuple) and expr[0] == "sym":
            if expr[1] not in facts:
                fail(f"release facts block: {expr[1]} is used before it is defined")
            return facts[expr[1]]
        if isinstance(expr, list) and expr and expr[0] == ("sym", "string-append"):
            return "".join(value(part) for part in expr[1:])
        fail(f"release facts block: unsupported expression {expr!r}")
        raise AssertionError("unreachable")

    for form in _read_forms(block):
        if not (isinstance(form, list) and len(form) == 3 and form[0] == ("sym", "define")
                and isinstance(form[1], tuple) and form[1][0] == "sym"):
            fail(f"release facts block: only (define name expr) is allowed, found {form!r}")
        name = form[1][1]
        if name in facts:
            fail(f"release facts block: {name} is defined twice")
        facts[name] = value(form[2])
    if "release-tag" not in facts:
        fail("release facts block does not define release-tag")
    return facts


def release_fact_problems(source: str, tag: str, assets: list[str]) -> list[str]:
    problems: list[str] = []
    before, block, after = split_release_facts(source)
    facts = evaluate_release_facts(block)
    if facts["release-tag"] != tag:
        problems.append(
            f"site release-tag is {facts['release-tag']!r} but {RELEASE_RECORD} says {tag!r}"
        )
    heading = facts.get("release-whats-new-heading")
    if heading != f"What's new in {tag}":
        problems.append(f"site \"what's new\" heading is {heading!r}; expected 'What's new in {tag}'")
    for name, text in facts.items():
        for other in set(TAG_RE.findall(text)) - {tag}:
            problems.append(f"release fact {name} names {other}, not the recorded {tag}")
    published = set(assets) | {CHECKSUMS_ASSET}
    urls = [m for text in facts.values() for m in DOWNLOAD_URL_RE.finditer(text)]
    if not urls:
        problems.append("release facts derive no download URL; the install snippet must use one")
    for m in urls:
        if m.group("tag") != tag:
            problems.append(f"download URL {m.group(0)} is for {m.group('tag')}, not {tag}")
        if m.group("asset") not in published:
            problems.append(f"download URL {m.group(0)} names an asset the release does not publish")
    for text in facts.values():
        for asset in ASSET_NAME_RE.findall(text):
            if asset not in published:
                problems.append(f"release facts name {asset}, which the release does not publish")
    # Nothing outside the block may restate the release.
    outside = before + after
    for m in TAG_RE.finditer(outside):
        problems.append(f"{SITE_SOURCE} writes the release tag {m.group(0)} outside the RELEASE FACTS block")
    if DOWNLOAD_PREFIX in outside:
        problems.append(f"{SITE_SOURCE} writes a release download URL outside the RELEASE FACTS block")
    for asset in ASSET_NAME_RE.findall(outside):
        problems.append(f"{SITE_SOURCE} writes the asset name {asset} outside the RELEASE FACTS block")
    if "What's new in" in outside:
        problems.append(f"{SITE_SOURCE} writes a \"What's new in\" heading outside the RELEASE FACTS block")
    for name in RENDERED_FACTS:
        if name not in facts:
            problems.append(f"release facts block does not define {name}")
        elif not re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", after):
            problems.append(f"release fact {name} is defined but no page renders it")
    return problems


# ---------------------------------------------------------------------------
# Homepage figures
# ---------------------------------------------------------------------------

HOMEPAGE_FUNCTIONS = (
    "render-hero", "render-pillars", "render-whats-new", "render-foundation",
    "render-sdnc", "render-what-you-can-build", "render-comparison-table", "render-stats",
)
# Literals that are code or its printed output, not claims about the project.
CODE_LITERAL_RE = re.compile(r"\(define |\(display |;; =>|<pre|^(?:h[1-6]|div|p|span|a|code)$|The 12\.0 is computed")
# Named standards and product versions: identifiers, not measured figures.
NAME_RE = re.compile(
    r"\bLLVM \d+\b|\bCUDA \d+(?:\.\d+)?\b|\bIEEE 754\b|\bint64\b|\bx64\b|\bx86_64\b|\bARM64\b"
    r"|\bO\(1\)|\bfloat32\b|\bR7RS\b|\bbinary128\b|\bdegree-10\b|\bM[1-4]\b|\b[A-Z]{2,}\d+\b"
)
_STYLE_CALL_RE = re.compile(r'\((?:style!|web-set-attribute|web-add-class)\s+\S+\s+"[^"]*"(?:\s+"[^"]*")?\)')


def _function_body(source: str, name: str) -> str:
    start = source.find(f"(define ({name} ")
    if start < 0:
        fail(f"{SITE_SOURCE} has no {name}")
    nxt = re.search(r"\n\(define |\n;; ═", source[start + 5:])
    return source[start: start + 5 + (nxt.start() if nxt else len(source))]


def homepage_prose(source: str) -> str:
    """The homepage's prose literals, tags stripped, joined with ' | '."""
    literals: list[str] = []
    for name in HOMEPAGE_FUNCTIONS:
        body = _STYLE_CALL_RE.sub("", _function_body(source, name))
        for raw in re.findall(r'"((?:[^"\\]|\\.)*)"', body):
            text = _unescape(raw)
            if CODE_LITERAL_RE.search(text):
                continue
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text.replace("·", " ")).strip()
            if text:
                literals.append(text)
    return " | ".join(literals)


def _num(text: str) -> str:
    return text.replace(",", "")


def _doc_values(root: Path, rel: str, pattern: str) -> tuple[str, ...]:
    try:
        text = (root / rel).read_text(encoding="utf-8")
    except OSError:
        fail(f"homepage claim source {rel} is missing")
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        fail(f"homepage claim source {rel} no longer states {pattern!r}")
    return tuple(_num(g) for g in m.groups())


def _surface_total(root: Path) -> str:
    policy = json.loads((root / "tests/coverage/coverage_policy.json").read_text(encoding="utf-8"))
    return str(policy["baseline_surface_total"])


def _parity_counts(root: Path) -> tuple[str, str, str, str]:
    counts = {"vm-supported": 0, "native-only-justified": 0, "gap": 0}
    rows = 0
    for line in (root / "tests/vm_parity/PARITY.tsv").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if line.startswith("#") or len(cols) < 2 or cols[1] == "status":
            continue
        if cols[1] not in counts:
            fail(f"tests/vm_parity/PARITY.tsv has unknown status {cols[1]!r}")
        counts[cols[1]] += 1
        rows += 1
    return (str(rows), str(counts["vm-supported"]), str(counts["native-only-justified"]), str(counts["gap"]))


def _record_parity(root: Path) -> str:
    record = json.loads((root / RELEASE_RECORD).read_text(encoding="utf-8"))
    total = record.get("vm_parity_total")
    if not isinstance(total, int):
        fail(f"{RELEASE_RECORD} has no vm_parity_total")
    return str(total)


def _asset_counts(root: Path) -> tuple[str, str]:
    workflow = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    n = len(release_assets(workflow, load_release_tag(root)))
    return str(n), str(n + 1)


def _web_bindings(root: Path) -> str:
    glue = (root / "web/eshkol-repl.js").read_text(encoding="utf-8")
    return str(len(set(re.findall(r"\b(web_[a-z0-9_]+)\s*:", glue))))


def _ad_operator_count(source: str, root: Path) -> str:
    """The pillar card lists the operators; each must be a declared construct."""
    m = re.search(r"(\d+) differential operators lower straight to LLVM IR: ([a-z\-, ]+)\.", source)
    if not m:
        fail("the AD pillar card no longer lists its differential operators")
    names = [n.strip() for n in m.group(2).split(",")]
    surface = json.loads((root / "tests/coverage/language_surface.json").read_text(encoding="utf-8"))
    declared = set()
    for key in ("builtins", "special_forms"):
        block = surface.get(key)
        if isinstance(block, dict):
            declared |= set(block)
        elif isinstance(block, list):
            declared |= {e.get("name") if isinstance(e, dict) else e for e in block}
    missing = [n for n in names if n not in declared]
    if missing:
        fail(f"AD pillar card lists undeclared operators {missing}")
    return str(len(names))


WORDS = {"one": "1", "two": "2", "three": "3", "twelve": "12", "fifteen": "15", "sixteen": "16"}

NOTES = "RELEASE_NOTES.md"
ANNOUNCE = "ANNOUNCEMENT.md"
SDNC = "docs/SDNC.md"
# (id, homepage regex, source description, expected-values function)
HOMEPAGE_CLAIMS = [
    ("surface-hero", r"([\d,]+) declared language constructs", "coverage_policy baseline_surface_total",
     lambda r, s: (_surface_total(r),)),
    ("surface-declared", r"manifest declares ([\d,]+) constructs", "coverage_policy baseline_surface_total",
     lambda r, s: (_surface_total(r),)),
    ("surface-coverage", r"coverage is ([\d,]+)/([\d,]+) declared constructs", "coverage_policy baseline_surface_total",
     lambda r, s: (_surface_total(r), _surface_total(r))),
    ("ad-operators", r"([\d,]+) differential operators", "the operators the pillar card lists, each a declared construct",
     lambda r, s: (_ad_operator_count(s, r),)),
    ("ad-operators-table", r"Compiler-native \((\d+) ops\)", "the operators the pillar card lists",
     lambda r, s: (_ad_operator_count(s, r),)),
    ("parity-rows-hero", r"([\d,]+) parity rows(?!:)", "tests/vm_parity/PARITY.tsv",
     lambda r, s: (_parity_counts(r)[0],)),
    ("parity-rows", r"([\d,]+) parity rows: ([\d,]+) vm-supported, ([\d,]+) native-only-justified, ([\d,]+) gap",
     "tests/vm_parity/PARITY.tsv", lambda r, s: _parity_counts(r)),
    ("parity-differential", r"VM parity differential runs ([\d,]+)/([\d,]+)", "release record vm_parity_total",
     lambda r, s: (_record_parity(r), _record_parity(r))),
    ("packages", r"(\w+) platform packages plus SHA256SUMS\.txt", "release workflow asset set",
     lambda r, s: (_asset_counts(r)[0],)),
    ("payload", r"(\d+)-file release payload", "release workflow asset set + SHA256SUMS.txt",
     lambda r, s: (_asset_counts(r)[1],)),
    ("payload-stat", r"(\d+) \| release files, checksummed", "release workflow asset set + SHA256SUMS.txt",
     lambda r, s: (_asset_counts(r)[1],)),
    ("sdnc-params", r"([\d,]+) (?:\| )?(?:analytically )?constructed parameters", SDNC,
     lambda r, s: _doc_values(r, SDNC, r"\*\*([\d,]+)\*\* analytically-constructed")),
    ("sdnc-isa", r"(\d+)-instruction ISA \| (\d+) opcodes are implemented", SDNC,
     lambda r, s: tuple(reversed(_doc_values(r, SDNC, r"(\d+) of the paper's (\d+)")))),
    ("sdnc-programs", r"(\d+)/(\d+),? (?:\| verified|checked) three ways", SDNC,
     lambda r, s: _doc_values(r, SDNC, r"\*\*(\d+)/(\d+) inline programs")),
    ("consciousness", r"(\d+) (?:\| consciousness primitives|built-in primitives)", "docs/FEATURE_MATRIX.md",
     lambda r, s: _doc_values(r, "docs/FEATURE_MATRIX.md", r"Consciousness Engine\*\*: [^\n]*?\((\d+) builtins\)")),
    ("web-bindings", r"provides (\d+) browser-host bindings", "web/eshkol-repl.js web_* host functions",
     lambda r, s: (_web_bindings(r),)),
    ("parser-depth", r"(\d[\d,]*) levels\.", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"runs\s+([\d,]+)\s+levels of nesting")),
    ("parser-stack", r"under an (\d+) MiB stack limit", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"on an actual (\d+) MiB pthread stack")),
    ("vm-rss", r"Peak RSS is (\d+) MB at ([\d,]+) iterations and (\d+) MB at ([\d,]+), against (\d+) MB", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"\*\*(\d+) MB at ([\d,]+) iterations, \d+ MB at [\d,]+, and (\d+) MB at\s+([\d,]+)\*\*.*?against \*\*(\d+) MB\*\*")),
    ("dense-matmul", r"A (\d+)x\d+ matmul costs the same per operation as a (\d+)x\d+", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"(\d+)×\d+ matmul equals\s+that of a (\d+)×\d+")),
    ("tensor-apply", r"report all (\d+) ordered assertions", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"all \*\*(\d+)\*\* ordered assertions")),
    ("dynamic-wind", r"reroots per R7RS (\d+\.\d+)", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"reroots on both engines per R7RS (\d+\.\d+)")),
    ("exact-sqrt", r"\(sqrt (\d+/\d+)\) answers (\d+/\d+)", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"`\(sqrt (\d+/\d+)\)` answers\s+`(\d+/\d+)`")),
    ("exact-expt", r"\(expt (\d+/\d+) -(\d+)\) is (\d+/\d+)", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"`\(expt (\d+/\d+) -(\d+)\)` is `(\d+/\d+)`")),
    ("exact-derivative", r"\(derivative \(lambda \(x\) \(\* x x\)\) (\d+/\d+)\) returns (\d+/\d+)", ANNOUNCE,
     lambda r, s: _doc_values(r, ANNOUNCE, r"`\(derivative \(lambda \(x\) \(\* x x\)\) (\d+/\d+)\)`.*?It returns `(\d+/\d+)`")),
    ("curvature-sweep", r"across a ([\d,]+)-binade sweep", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"binary128 across a ([\d,]+)-binade")),
    ("antipode-sweep", r"over a ([\d.]+)-million-case", ANNOUNCE,
     lambda r, s: _doc_values(r, ANNOUNCE, r"([\d.]+)-million-case")),
    ("assurance-switches", r"finds (\d+) exhaustive AST operation switches", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"discovers \*\*(\d+)\*\* AST operation")),
    ("flat-ticks", r"for ([\d,]+) ticks and its memory stays flat", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"over ([\d,]+) ticks\)")),
    ("flat-rss", r"Over ([\d,]+) ticks the loop sits flat at (\d+) MB", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"over ([\d,]+) ticks\).*?now flat at (\d+) MB")),
    ("float-printer", r"double \(R7RS (\d+\.\d+\.\d+)\).*?\(sqrt 2\.0\) prints (\d\.\d+)", NOTES,
     lambda r, s: _doc_values(r, NOTES, r"Shortest-round-trip numeric printing \(R7RS (\d+\.\d+\.\d+)\).*?`(1\.\d+)`")),
]


def homepage_claim_problems(source: str, root: Path) -> list[str]:
    problems: list[str] = []
    prose = homepage_prose(source)
    covered = [False] * len(prose)
    for claim_id, pattern, origin, expected_fn in HOMEPAGE_CLAIMS:
        matches = list(re.finditer(pattern, prose))
        if not matches:
            continue
        try:
            expected = expected_fn(root, source)
        except SiteCheckError as exc:
            problems.append(f"homepage claim {claim_id}: {exc}")
            continue
        for m in matches:
            stated = tuple(_num(WORDS.get(g.lower(), g)) for g in m.groups())
            if stated != tuple(expected):
                problems.append(
                    f"homepage claim {claim_id} states {', '.join(m.groups())} but {origin} gives "
                    f"{', '.join(expected)}"
                )
            for i in range(m.start(), m.end()):
                covered[i] = True
    for m in NAME_RE.finditer(prose):
        for i in range(m.start(), m.end()):
            covered[i] = True
    for m in re.finditer(r"\d+(?:[.,]\d+)*", prose):
        if not all(covered[i] for i in range(m.start(), m.end())):
            context = prose[max(0, m.start() - 50): m.end() + 30]
            problems.append(
                f"homepage figure {m.group(0)!r} has no registered source "
                f"(add a HOMEPAGE_CLAIMS entry): ...{context}..."
            )
    return problems


# ---------------------------------------------------------------------------
# index.html, published documentation, compiled WASM
# ---------------------------------------------------------------------------

def index_problems(index: str, tag: str) -> list[str]:
    problems: list[str] = []
    m = re.search(r'"softwareVersion":\s*"([^"]+)"', index)
    if not m or m.group(1) != tag.lstrip("v"):
        problems.append(f"site/static/index.html softwareVersion is {m.group(1) if m else None!r}, "
                        f"expected {tag.lstrip('v')!r}")
    m = re.search(r"const expectedTag = '([^']+)'", index)
    if not m or m.group(1) != tag:
        problems.append(f"site/static/index.html expectedTag is {m.group(1) if m else None!r}, expected {tag!r}")
    for other in sorted(set(TAG_RE.findall(index)) - {tag}):
        problems.append(f"site/static/index.html names {other}, not the recorded {tag}")
    return problems


def _load_content_builder(root: Path):
    path = ROOT / "scripts" / "build_site_content.py"
    spec = importlib.util.spec_from_file_location("build_site_content_for_verify", path)
    if spec is None or spec.loader is None:
        fail(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def content_problems(source: str, root: Path) -> list[str]:
    builder = _load_content_builder(root)
    try:
        doc = builder.load_page_list(root / "site" / "pages.json", root)
    except builder.PageListError as exc:
        return [str(exc)]
    problems = builder.check(doc, root / "site" / "static" / "content")
    for name, view in doc["views"].items():
        if f'(string=? route-buf "{view["route"]}")' not in source:
            problems.append(f"{SITE_SOURCE} does not route {view['route']} (view {name})")
        if f'(web-load-content "/content/nav-{name}.html"' not in source:
            problems.append(f"{SITE_SOURCE} does not load the generated {name} navigation")
        if f'(web-load-content "/content/{view["default"]}.html"' not in source:
            problems.append(f"{SITE_SOURCE} does not load the {name} default page {view['default']}")
    if "raw.githubusercontent.com" in source:
        problems.append(f"{SITE_SOURCE} fetches documentation from the repository; publish it via site/pages.json")
    return problems


def wasm_fact_problems(wasm: bytes, source: str, tag: str) -> list[str]:
    """The committed WASM must be built from the current source."""
    problems: list[str] = []
    for other in sorted({t.decode() for t in re.findall(rb"v\d+\.\d+\.\d+-[a-z]+", wasm)} - {tag}):
        problems.append(f"committed site WASM names {other}; rebuild it with scripts/build-site.sh")
    _, block, _ = split_release_facts(source)
    literals = [_unescape(raw) for raw in re.findall(r'"((?:[^"\\]|\\.)*)"', block)]
    literals += [_unescape(raw) for raw in re.findall(r'"((?:[^"\\]|\\.)*)"', _function_body(source, "render-whats-new"))]
    for text in literals:
        if len(text) >= 6 and text.encode() not in wasm:
            problems.append(f"committed site WASM is missing {text[:60]!r}; rebuild it with scripts/build-site.sh")
    return problems


def read_varuint(data: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, i


def wasm_export_param_counts(wasm: bytes) -> dict[str, int]:
    """Map exported function names to their parameter counts."""
    i = 8
    func_types: list[int] = []
    type_params: list[int] = []
    imported_funcs = 0
    exports: dict[str, int] = {}
    while i < len(wasm):
        section_id = wasm[i]
        i += 1
        size, i = read_varuint(wasm, i)
        end = i + size
        j = i
        if section_id == 1:  # type section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                if wasm[j] != 0x60:
                    fail("unexpected wasm type-section entry")
                j += 1
                nparams, j = read_varuint(wasm, j)
                j += nparams
                nresults, j = read_varuint(wasm, j)
                j += nresults
                type_params.append(nparams)
        elif section_id == 2:  # import section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                mlen, j = read_varuint(wasm, j)
                j += mlen
                nlen, j = read_varuint(wasm, j)
                j += nlen
                kind = wasm[j]
                j += 1
                if kind == 0:
                    _, j = read_varuint(wasm, j)
                    imported_funcs += 1
                elif kind == 1:
                    j += 1
                    flags, j = read_varuint(wasm, j)
                    _, j = read_varuint(wasm, j)
                    if flags & 1:
                        _, j = read_varuint(wasm, j)
                elif kind == 2:
                    flags, j = read_varuint(wasm, j)
                    _, j = read_varuint(wasm, j)
                    if flags & 1:
                        _, j = read_varuint(wasm, j)
                elif kind == 3:
                    j += 2
        elif section_id == 3:  # function section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                type_idx, j = read_varuint(wasm, j)
                func_types.append(type_idx)
        elif section_id == 7:  # export section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                nlen, j = read_varuint(wasm, j)
                name = wasm[j : j + nlen].decode()
                j += nlen
                kind = wasm[j]
                j += 1
                idx, j = read_varuint(wasm, j)
                if kind == 0:
                    local_idx = idx - imported_funcs
                    if 0 <= local_idx < len(func_types):
                        exports[name] = type_params[func_types[local_idx]]
        i = end
    return exports



def legacy_problems(root: Path, tag: str, source: str, wasm: bytes, assets: list[str]) -> list[str]:
    """Release-matrix, loader, ABI and artifact-size checks."""
    problems: list[str] = []

    def need(text: str, needle: str, label: str) -> None:
        if needle not in text:
            problems.append(f"{label} is missing {needle!r}")

    index = (root / "site/static/index.html").read_text(encoding="utf-8")
    if len(assets) != 15:
        problems.append(f"expected 15 platform packages, release workflow declares {len(assets)}")
    if len(set(assets)) != len(assets):
        problems.append("release workflow contains duplicate package names")
    if any("windows-arm64-cuda" in asset for asset in assets):
        problems.append("release workflow advertises unsupported Windows ARM64 CUDA")
    for asset in assets:
        need(source, matrix_name(asset, tag), "site release matrix")
    need(source, "15 platform packages plus SHA256SUMS.txt", "site download summary")
    need(source, "16-file release payload", "site download summary")
    need(source, '"releases-container"', "site release loader target")
    need(source, '"release-status"', "site publication status target")
    need(source, "Unsupported", "Windows ARM64 CUDA matrix cell")
    need(source, "(define route-buf (make-string 256 #\\nul))", "mutable pathname router buffer")
    if "16 pre-built binaries" in source:
        problems.append("site still claims that all 16 release files are binaries")
    if "(let ((make-badge (lambda" in source or "(let ((make-link-card (lambda" in source:
        problems.append("site route rendering still uses unsupported local WASM helper closures")
    for needle in ("softwareVersion", "expectedTag", "escapeHtml"):
        need(index, needle, "site GitHub release loader")

    announcement = (root / "ANNOUNCEMENT.md").read_text(encoding="utf-8")
    announcement_html = (root / "site/static/content/announcement.html").read_text(encoding="utf-8")
    announcement_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", announcement_html))
    need(announcement, "15 platform packages plus `SHA256SUMS.txt`", "release announcement")
    need(announcement, "not a claim of complete backend parity", "VM parity disclosure")
    need(announcement_text, "15 platform packages plus SHA256SUMS.txt", "generated announcement HTML")

    # The exported scheme_main must take no parameters. The underlying Scheme
    # main returns a tagged value that the wasm ABI demotes to an sret
    # out-pointer; exporting that demoted signature let JS glue calling
    # scheme_main(0) write the return value through address 0 and corrupt the
    # first data globals (the SPA router broke on the second navigation).
    exports = wasm_export_param_counts(wasm)
    if "scheme_main" not in exports:
        problems.append("committed site WASM does not export scheme_main")
    elif exports["scheme_main"] != 0:
        problems.append(f"exported scheme_main takes {exports['scheme_main']} parameter(s); it must be a "
                        "zero-argument shim so JS re-entry cannot alias linear memory")

    # The playground statistics must match the committed artifacts.
    stat_claims = {slot: int(value) for value, slot in
                   re.findall(r'\(create-text "div" "(\d+)KB" (s\d)\)', source)}
    for label, slot, artifact in (("Site WASM", "s1", SITE_WASM), ("VM WASM", "s2", "site/static/eshkol-vm.wasm")):
        actual_kb = round((root / artifact).stat().st_size / 1000)
        claimed = stat_claims.get(slot)
        if claimed is None:
            problems.append(f"could not find the {label} statistic in {SITE_SOURCE}")
        elif abs(claimed - actual_kb) > 1:
            problems.append(f"{label} statistic says {claimed}KB but {artifact} is {actual_kb}KB; "
                            "update the playground stats")

    for needle in (tag.encode(), b"Fifteen platform packages plus SHA256SUMS.txt", b"16-file release payload",
                   b"Verified VM subset", b"Install Eshkol on macOS, Linux, Windows",
                   b"windows-arm64-lite", b"windows-arm64-xla"):
        if needle not in wasm:
            problems.append(f"committed site WASM is missing {needle.decode()!r}; rebuild the site")
    if b"16 pre-built binaries" in wasm:
        problems.append("committed site WASM still contains the stale 16-binary claim")
    return problems


def verify(root: Path) -> list[str]:
    """Every disagreement between the site and its sources, as messages."""
    problems: list[str] = []

    def section(fn, *args) -> None:
        try:
            problems.extend(fn(*args))
        except SiteCheckError as exc:
            problems.append(str(exc))
        except OSError as exc:
            problems.append(f"cannot read {exc.filename}: {exc.strerror}")

    try:
        tag = load_release_tag(root)
        assets = release_assets((root / ".github/workflows/release.yml").read_text(encoding="utf-8"), tag)
        source = (root / SITE_SOURCE).read_text(encoding="utf-8")
        wasm = (root / SITE_WASM).read_bytes()
        index = (root / "site/static/index.html").read_text(encoding="utf-8")
    except (SiteCheckError, OSError) as exc:
        return [str(exc)]
    section(release_fact_problems, source, tag, assets)
    section(index_problems, index, tag)
    section(homepage_claim_problems, source, root)
    section(content_problems, source, root)
    section(wasm_fact_problems, wasm, source, tag)
    section(legacy_problems, root, tag, source, wasm, assets)
    return problems


# ---------------------------------------------------------------------------
# Self-test: every check must reject a red fixture and accept its green twin
# ---------------------------------------------------------------------------

def _fixture_source(tag: str, *, whats_new: str | None = None, url_tag: str | None = None,
                    asset: str | None = None, outside: str = "", render_banner: bool = True) -> str:
    url_tag = url_tag or tag
    asset = asset or f"eshkol-{url_tag}-linux-x64-lite.tar.gz"
    heading = whats_new or '(string-append "What\'s new in " release-tag)'
    return f''';; BEGIN RELEASE FACTS
(define release-tag "{tag}")
(define release-download-base "{DOWNLOAD_PREFIX}{url_tag}/")
(define release-linux-asset "{asset}")
(define release-whats-new-heading {heading})
(define release-publication-heading (string-append release-tag " Publication"))
(define release-downloads-summary (string-append "Install " release-tag))
(define release-linux-install-html (string-append "curl -LO " release-download-base release-linux-asset))
(define release-vm-banner-html (string-append "VM " release-tag))
;; END RELEASE FACTS
(define (render-whats-new p) (section-heading p release-whats-new-heading))
(define (render-downloads p) (list release-publication-heading release-downloads-summary release-linux-install-html))
{"(define (render-playground p) release-vm-banner-html)" if render_banner else ""}
{outside}
'''


def _homepage_fixture(body: str) -> str:
    stubs = "".join(f'(define ({name} parent) "")\n' for name in HOMEPAGE_FUNCTIONS if name != "render-foundation")
    return stubs + f'(define (render-foundation parent)\n  (body-html parent "{body}"))\n'


def self_test() -> bool:
    tag, old = "v9.8.7-evolve", "v9.8.6-evolve"
    assets = [f"eshkol-{tag}-linux-x64-lite.tar.gz"]
    surface = _surface_total(ROOT)
    parity = _parity_counts(ROOT)
    cases: list[tuple[str, bool, list[str]]] = []

    def run(name: str, expect_ok: bool, fn, *args) -> None:
        try:
            found = fn(*args)
        except SiteCheckError as exc:
            found = [str(exc)]
        cases.append((name, expect_ok == (not found), found))

    run("facts_green", True, release_fact_problems, _fixture_source(tag), tag, assets)
    run("facts_tag_differs_from_record", False, release_fact_problems, _fixture_source(old), tag, assets)
    run("facts_stale_whats_new_heading", False, release_fact_problems,
        _fixture_source(tag, whats_new=f'"What\'s new in {old}"'), tag, assets)
    run("facts_install_url_for_old_tag", False, release_fact_problems, _fixture_source(tag, url_tag=old), tag, assets)
    run("facts_unpublished_asset", False, release_fact_problems,
        _fixture_source(tag, asset=f"eshkol-{tag}-linux-riscv-lite.tar.gz"), tag, assets)
    run("facts_tag_written_outside_block", False, release_fact_problems,
        _fixture_source(tag, outside=f'(define (x) "Valid after the {tag} tag")'), tag, assets)
    run("facts_download_url_outside_block", False, release_fact_problems,
        _fixture_source(tag, outside=f'(define (x) "{DOWNLOAD_PREFIX}")'), tag, assets)
    run("facts_fact_not_rendered", False, release_fact_problems, _fixture_source(tag, render_banner=False), tag, assets)
    run("facts_block_missing", False, release_fact_problems, '(define release-tag "v9.8.7-evolve")', tag, assets)
    run("today_v135_cut_site_source", False, release_fact_problems,
        _fixture_source(tag).split(";; BEGIN")[0] + f'(section-heading s "What\'s new in {old}")\n'
        f'"# Valid after the {old} tag is published"', tag, assets)

    green = f"The manifest declares {surface} constructs. {parity[0]} parity rows: {parity[1]} vm-supported, {parity[2]} native-only-justified, {parity[3]} gap."
    run("claims_green", True, homepage_claim_problems, _homepage_fixture(green), ROOT)
    run("claims_stale_figure", False, homepage_claim_problems,
        _homepage_fixture(f"The manifest declares {int(surface) + 1} constructs."), ROOT)
    run("claims_unsourced_figure", False, homepage_claim_problems,
        _homepage_fixture("It ships 42 widgets."), ROOT)
    run("claims_named_versions_are_not_figures", True, homepage_claim_problems,
        _homepage_fixture("Built on LLVM 21 and CUDA 12.4 for x64 and ARM64."), ROOT)

    good_index = f'"softwareVersion": "{tag[1:]}" const expectedTag = \'{tag}\';'
    run("index_green", True, index_problems, good_index, tag)
    run("index_stale_tag", False, index_problems, good_index.replace(tag, old), tag)
    run("index_mentions_old_release", False, index_problems, good_index + f" {old}", tag)

    source = _fixture_source(tag)
    literals = [_unescape(raw).encode() for raw in re.findall(r'"((?:[^"\\]|\\.)*)"', source)]
    run("wasm_green", True, wasm_fact_problems, b"\0".join(literals), source, tag)
    run("wasm_built_from_old_source", False, wasm_fact_problems,
        b"\0".join(literals).replace(tag.encode(), old.encode()), source, tag)

    ok = all(passed for _, passed, _ in cases)
    print("verify_site_release.py self-test:")
    for name, passed, found in cases:
        print(f"  [{'OK' if passed else 'GATE IS BROKEN'}] {name}" + ("" if passed else f": {found[:2]}"))
    print(f"self-test {'PASS' if ok else 'FAIL'}: {sum(p for _, p, _ in cases)}/{len(cases)} fixtures")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout to verify (default: this repository)")
    parser.add_argument("--self-test", action="store_true", help="run the red/green fixture suite")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    problems = verify(args.root.resolve())
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        print(f"verify_site_release: {len(problems)} disagreement(s) between the website and its sources",
              file=sys.stderr)
        return 1
    tag = load_release_tag(args.root.resolve())
    print(f"PASS: website names {tag} only through its release facts, which match {RELEASE_RECORD}; "
          "install URLs name published assets; every homepage figure matches its source; "
          "published documentation matches site/pages.json; the committed WASM is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
