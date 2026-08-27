#!/usr/bin/env python3
"""Machine inventory and ratchet gate for object-header layout dependence.

Every heap-allocated Eshkol object is prefixed with an ``eshkol_object_header_t``
and reached by a pointer to the byte *after* that header.  ``subtype`` — the only
field that says what kind of object this is — lives inside the header.  A change
to the header layout therefore has no discriminator: an object built under the
old layout and an object built under the new layout are indistinguishable without
already knowing which one you hold.  A half-migrated toolchain links and produces
garbage.

The migration is only safe if the set of sites that depend on the current layout
is established by machine and re-established on every commit.  A hand inventory
of five hundred sites is the kind of artifact that is 96% right, and 96% is the
failure mode.

This tool does three jobs.

``scan``       Run every detector and emit the classified inventory as JSON.
``snapshot``   Write the inventory to ``docs/design/abi/header-site-inventory.json``
               and a human-readable companion, both regenerable.
``check``      Ratchet.  Fail if a class/file pair carries more sites than the
               baseline records, or if a site appears in a file the baseline does
               not list.  Existing sites are baselined; new ones are forbidden.
``baseline``   Rewrite ``.icc/abi-header-baseline.json`` from the current scan.
``selftest``   Prove the ratchet can go red: inject a synthetic new site into a
               scratch tree, show ``check`` fails, remove it, show ``check``
               passes.  A gate never shown red does not count.

Detectors are layered by how much machinery they need, and every finding records
the detector that produced it so the finding method is itself auditable:

  lexical   Comment- and string-stripped token matching over ``git ls-files``.
            No build required, no third-party dependency.  This layer is what
            the CI ratchet enforces, because it must be able to run anywhere.
  semantic  libclang over ``compile_commands.json``.  Resolves accesses the
            lexical layer cannot see — a member access through a variable whose
            type is only known after name lookup, a ``sizeof *hdr``, a cast
            written through a typedef.  Optional; when it runs, its findings are
            merged into the inventory and must be baselined too.
  emitted   Ground truth from the compiler's own output.  Compiles a corpus with
            ``--dump-ir`` and counts header GEPs in the emitted LLVM IR.  The
            lexical and semantic layers find sites that *exist*; this layer finds
            offsets that actually *fire*, and so catches a source construct that
            produces a header offset by a route no source-level detector models.

Usage:
    scripts/abi_header_inventory.py scan [--clang] [--emitted]
    scripts/abi_header_inventory.py snapshot [--clang] [--emitted]
    scripts/abi_header_inventory.py check [--clang]
    scripts/abi_header_inventory.py baseline [--clang]
    scripts/abi_header_inventory.py selftest

Copyright (C) Tsotchke Corporation. MIT License.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

# ── Layout facts this tool is written against ────────────────────────────────
# These are asserted against the source by the `layout-pin` detector, so that a
# change to the real layout makes the tool complain instead of silently scanning
# for the wrong constant.
CANONICAL_HEADER_TYPE = "eshkol_object_header_t"
CANONICAL_HEADER_SIZE = 8
CANONICAL_FIELDS = ("subtype", "flags", "ref_count", "size")

REPO_ROOT = Path(__file__).resolve().parent.parent

BASELINE_PATH = REPO_ROOT / ".icc" / "abi-header-baseline.json"
SNAPSHOT_JSON = REPO_ROOT / "docs" / "design" / "abi" / "header-site-inventory.json"
SNAPSHOT_TEXT = REPO_ROOT / "docs" / "design" / "abi" / "header-site-inventory.txt"

C_FAMILY = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".m", ".mm", ".cu", ".inc"}
JS_FAMILY = {".js", ".mjs", ".ts"}
PROSE_FAMILY = {".md"}

# Directories whose contents are generated, vendored, or otherwise not ours.
EXCLUDED_PREFIXES = (
    "build/",
    "references/",
    "tests/recursion_depth/generated/",
)
# site/static/ is generated *except* for eshkol-runtime.js, which hand-writes an
# object header into WASM linear memory and is therefore a first-class ABI site.
EXCLUDED_EXCEPTIONS = ("site/static/eshkol-runtime.js",)
GENERATED_SITE_PREFIX = "site/static/"


# ── Detector registry ────────────────────────────────────────────────────────
# Each entry documents, in the artifact itself, exactly how that class of site is
# found.  The description is emitted into the snapshot so a reader can rerun the
# method by hand and get the same answer.
DETECTORS = {
    "A_macro_api": {
        "layer": "lexical",
        "what": "Uses of the ESHKOL_GET_HEADER macro family.",
        "how": (
            "Comment- and string-literal-stripped token match on the macro names "
            "declared in inc/eshkol/eshkol.h's HEADER ACCESS MACROS block. The "
            "macro names are unique identifiers, so the match is exact, not "
            "heuristic."
        ),
        "risk": "None. This class is complete by construction.",
    },
    "B_header_type": {
        "layer": "lexical",
        "what": "Textual references to the header type itself (declarations, casts, parameters).",
        "how": (
            "Stripped token match on `eshkol_object_header_t` / "
            "`struct eshkol_object_header`, excluding lines already claimed by "
            "class A or C."
        ),
        "risk": "None for the canonical spelling; aliases are covered by class F.",
    },
    "C_sizeof_header": {
        "layer": "lexical",
        "what": "Header size used as an arithmetic quantity — an offset, an overhead, an allocation adjustment.",
        "how": (
            "Stripped regex for `sizeof` applied to any known header type name. "
            "These are the sites where the size 8 is load-bearing rather than "
            "incidental."
        ),
        "risk": "Misses `sizeof *hdr`; that spelling is recovered by the semantic layer (class S2).",
    },
    "D_ir_const_offset": {
        "layer": "lexical",
        "what": "Header offsets baked into emitted LLVM IR as compile-time constants.",
        "how": (
            "Stripped regex over lib/backend and lib/repl for an LLVM integer "
            "constant equal to the negated header size — `ConstantInt::get(T, -8)`, "
            "`getInt64(-8)`, `CreateConstGEP1_64(..., -8)`. This is the class no "
            "grep for the header type finds, because the type name never appears: "
            "the compiler emits a raw number. Each such site bakes in TWO facts, "
            "the header size and the field offset within it, so a site reading "
            "`subtype` emits -8 today and would need -(newsize) + newsubtypeoffset."
        ),
        "risk": (
            "A -8 that means something other than a header offset would be a false "
            "positive; the snippet is recorded for every finding so each can be "
            "audited. Cross-checked against the `emitted` layer, which counts the "
            "offsets that actually reach the IR."
        ),
    },
    "D2_ir_field_offset": {
        "layer": "lexical",
        "what": "Offsets of individual fields WITHIN the header, emitted as LLVM constants.",
        "how": (
            "Stripped regex over lib/backend and lib/repl for an LLVM integer "
            "constant in -1..-7. `subtype` is at -8, but `flags` is at -7, "
            "`ref_count` at -6 and `size` at -4, so every published search for "
            "`-8` finds none of these. A migration that changes the header size "
            "and diligently fixes every -8 site still leaves this class reading "
            "the wrong field of the right header."
        ),
        "risk": (
            "A small negative constant that is not a header field offset is a "
            "false positive. The snippet is recorded for each so the class can be "
            "audited by eye; over-reporting here is the safe direction."
        ),
    },
    "K_raw_byte_offset": {
        "layer": "lexical",
        "what": "Pointer arithmetic that rebuilds the header from raw bytes, naming neither the type nor the macro.",
        "how": (
            "Stripped regex for a byte-pointer cast followed by `- 8`, for "
            "`*(p - 8)`, and for `ptr - 8`. These sites read as ordinary byte "
            "arithmetic; they contain no token that a search for the header type "
            "or the accessor macro would match, and they are the sites most "
            "likely to survive a migration untouched."
        ),
        "risk": "Any `- 8` in byte-pointer arithmetic matches. Recorded with snippets for audit.",
    },
    "L_layout_in_prose": {
        "layer": "lexical",
        "what": "The byte layout restated in comments and documentation.",
        "how": (
            "Regex over RAW text — comments deliberately included — for the "
            "layout's published spellings: `[header (8 bytes)]`, `8-byte header`, "
            "`header is at offset -8`, `[subtype(1)][flags(1)]`. Prose is not "
            "compiled, so nothing keeps it true; it goes stale silently and is "
            "then read as authoritative. Includes .md, and the published "
            "specification pages, because a wrong number in the spec is a wrong "
            "number an integrator will implement against."
        ),
        "risk": "None to correctness; this class is a documentation work-list, not a code hazard.",
    },
    "M_cache_key_without_abi": {
        "layer": "lexical",
        "what": "Persistent artifact caches whose key does not include the object ABI.",
        "how": (
            "A detector for an ABSENCE: a file that computes a persistent cache "
            "key (hashUpdate / xxh3_64bits / a hand-versioned artifact name / "
            "addObjectFile) and does NOT mention the ABI fingerprint anywhere. "
            "Such a cache will hand back an artifact built against the other "
            "layout and the program will run. Absence is the hardest property to "
            "watch by hand, which is exactly why it is machined; this class goes "
            "empty as the fingerprint is threaded into each key."
        ),
        "risk": (
            "Coarse — one finding per file, not per key. It is a work-list of "
            "caches to audit, and it is designed to be driven to zero."
        ),
    },
    "E_alloc_with_header": {
        "layer": "lexical",
        "what": "Calls through the `*_with_header` allocator family — the sites that WRITE the layout.",
        "how": (
            "Stripped token match on identifiers matching `\\w*_with_header\\b`. "
            "These are the constructor half of the ABI; classes A-D are the reader "
            "half. A migration that fixes readers and misses writers produces "
            "exactly the silent garbage this inventory exists to prevent."
        ),
        "risk": "None; the naming convention is enforced by class G.",
    },
    "F_parallel_header_decl": {
        "layer": "lexical",
        "what": "Struct declarations that structurally duplicate the canonical header under a different name.",
        "how": (
            "Parse every `typedef struct { ... } Name;` and `struct Name { ... };` "
            "body in the tree and compare its field sequence — types and names in "
            "order — against the canonical header's. A structural match under a "
            "different name is a second definition of the ABI that no search for "
            "the canonical type name will ever find, and that will not be updated "
            "when the canonical one is."
        ),
        "risk": (
            "A struct that coincidentally has the same four fields would be a false "
            "positive. That is the correct bias: a coincidence here is worth "
            "auditing."
        ),
    },
    "G_header_field_write": {
        "layer": "lexical",
        "what": "Direct writes to header fields through a pointer whose name suggests a header.",
        "how": (
            "Stripped regex for `<ident>->{subtype,flags,ref_count,size}` where "
            "<ident> matches a header-ish name (hdr, header, h, obj_header, ...). "
            "Heuristic by design; the semantic layer supersedes it when libclang "
            "is available."
        ),
        "risk": "Both directions. Superseded by class S1 under --clang.",
    },
    "H_wasm_glue": {
        "layer": "lexical",
        "what": "JavaScript/TypeScript that names a header-aware runtime import or a header field.",
        "how": (
            "Stripped token match over .js/.mjs/.ts for `*_with_header` and for the "
            "header field names appearing as object keys or parameters. The WASM "
            "lane reimplements the runtime import surface in JS; an import whose "
            "signature mentions a subtype is a layout commitment on the JS side."
        ),
        "risk": "Narrow by construction — it cannot see byte offsets computed arithmetically in JS.",
    },
    "J_secondary_prefix_abi": {
        "layer": "lexical",
        "what": "The OTHER prefix-header ABI — 24-byte eshkol_shared_header_t on shared/ref-counted allocations.",
        "how": (
            "Stripped token match on `eshkol_shared_header_t`, plus header-ish "
            "field accesses in files that name it but not the object header. "
            "Kept as a separate class deliberately: it uses the identical "
            "`data - sizeof(header)` idiom and the identical field names "
            "(`ref_count`, `flags`), so a migration that pattern-matches on the "
            "idiom rather than the type will convert these by accident."
        ),
        "risk": "None; the type name is unique.",
    },
    "I_public_abi": {
        "layer": "lexical",
        "what": "Layout dependence inside a PUBLIC installed header.",
        "how": (
            "Any class A-G finding whose path is an installed header (inc/eshkol/**) "
            "is additionally recorded here. These are not merely internal sites: "
            "they are commitments to embedders who compiled against them, so they "
            "cannot be changed by recompiling this repository alone."
        ),
        "risk": "None; derived from the other classes.",
    },
    "S1_clang_header_member": {
        "layer": "semantic",
        "what": "Member accesses whose base resolves to the header type, however spelled.",
        "how": (
            "libclang AST walk over the translation units in compile_commands.json; "
            "every MEMBER_REF_EXPR whose base expression's canonical type is "
            "(pointer to) the header type. Catches accesses through typedefs, "
            "`auto`, template parameters, and macro expansions the lexical layer "
            "cannot resolve."
        ),
        "risk": "Requires a configured build; skipped without one, and the skip is recorded in the snapshot.",
    },
    "S2_clang_sizeof": {
        "layer": "semantic",
        "what": "sizeof/alignof over the header type, including `sizeof *hdr` and `sizeof(decltype(...))`.",
        "how": "libclang: UnaryExprOrTypeTraitExpr whose operand type canonicalizes to the header type.",
        "risk": "Same build dependence as S1.",
    },
    "S3_clang_header_cast": {
        "layer": "semantic",
        "what": "Casts to a header pointer type — the spelling that reconstructs a header from raw bytes.",
        "how": "libclang: any cast expression whose destination type canonicalizes to a pointer to the header type.",
        "risk": "Same build dependence as S1.",
    },
    "R_emitted_ir": {
        "layer": "emitted",
        "what": "Header offsets that actually reach emitted LLVM IR.",
        "how": (
            "Compile a corpus with `eshkol-run --dump-ir` and count "
            "`getelementptr i8, ptr %x, i64 -8` in the .ll. This is ground truth "
            "for the compiler side: it does not depend on any model of the source, "
            "so it catches an offset produced by a route no source detector knows "
            "about. Reported as a count, not a site list — its purpose is to "
            "falsify the claim that the static inventory is complete."
        ),
        "risk": "Only covers constructs the corpus exercises. Coverage is the corpus's job, not this detector's.",
    },
}


@dataclass(frozen=True)
class Site:
    cls: str
    path: str
    line: int
    snippet: str
    detector: str


# ── Source text handling ─────────────────────────────────────────────────────
def strip_c_comments_and_strings(text: str) -> str:
    """Blank out comments and string/char literals, preserving byte offsets.

    Replacing rather than deleting keeps line and column numbers exact, so every
    finding can be pointed at with file:line and read in an editor.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif c in ("'", '"'):
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    out[i] = " "
                    i += 1
                    if i < n and text[i] != "\n":
                        out[i] = " "
                        i += 1
                    continue
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def tracked_sources(root: Path) -> list[Path]:
    """Enumerate candidate files from git, so the inventory is reproducible.

    git ls-files rather than a filesystem walk: the set of files is then exactly
    what is committed, independent of build artifacts, editor droppings, or a
    stale working tree.
    """
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True, text=True, check=True,
    )
    files = []
    for rel in proc.stdout.split("\0"):
        if not rel:
            continue
        if rel not in EXCLUDED_EXCEPTIONS:
            if any(rel.startswith(p) for p in EXCLUDED_PREFIXES):
                continue
            if rel.startswith(GENERATED_SITE_PREFIX):
                continue
        suffix = Path(rel).suffix
        if suffix in C_FAMILY or suffix in JS_FAMILY or suffix in PROSE_FAMILY:
            files.append(root / rel)
    return sorted(files)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def snippet_at(raw: str, line: int) -> str:
    lines = raw.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()[:200]
    return ""


# ── Lexical detectors ────────────────────────────────────────────────────────
MACRO_FAMILY = (
    "ESHKOL_GET_HEADER", "ESHKOL_GET_DATA_PTR", "ESHKOL_GET_SUBTYPE",
    "ESHKOL_GET_FLAGS", "ESHKOL_SET_SUBTYPE", "ESHKOL_SET_FLAGS",
    "ESHKOL_HAS_FLAG", "ESHKOL_ADD_FLAG", "ESHKOL_CLEAR_FLAG",
    "ESHKOL_GET_SIZE", "ESHKOL_SET_SIZE",
)
RE_MACRO = re.compile(r"\b(" + "|".join(MACRO_FAMILY) + r")\b")

HEADER_TYPE_NAMES = (
    "eshkol_object_header_t", "eshkol_object_header",
    "VmObjectHeader", "eshkol_object_header_v2_t", "eshkol_object_header_active_t",
)
# A second, independent prefix-header ABI in the same runtime: shared/ref-counted
# allocations carry a 24-byte eshkol_shared_header_t before the payload and are
# reached by the same `data - sizeof(header)` idiom.  It is not the object header,
# and conflating the two is precisely the mistake a migration makes at 3am, so it
# gets its own class rather than being folded in or discarded as noise.
SECONDARY_HEADER_TYPE_NAMES = ("eshkol_shared_header_t", "eshkol_shared_header")
RE_SECONDARY_HEADER = re.compile(r"\b(" + "|".join(SECONDARY_HEADER_TYPE_NAMES) + r")\b")
RE_HEADER_TYPE = re.compile(r"\b(" + "|".join(HEADER_TYPE_NAMES) + r")\b")
RE_SIZEOF_HEADER = re.compile(
    r"\b(?:sizeof|alignof|_Alignof|__alignof__)\s*\(\s*(?:struct\s+)?("
    + "|".join(HEADER_TYPE_NAMES) + r")\s*\)"
)

# LLVM IR constants that encode the negated header size.  Written against the
# canonical size so that changing CANONICAL_HEADER_SIZE re-points the detector.
_NEG = r"-\s*" + str(CANONICAL_HEADER_SIZE)
# `[^;]{0,160}?` rather than `[^;()]*?` because the type argument is itself a
# call — `ConstantInt::get(ctx_.int64Type(), -8)` — so a paren-free window would
# silently miss two thirds of the class.  Bounded and non-greedy so it cannot run
# past the statement.
RE_IR_CONST = re.compile(
    r"(?:ConstantInt::get\s*\([^;]{0,160}?,\s*" + _NEG + r"\s*\)"
    r"|\bgetInt(?:8|16|32|64)\s*\(\s*" + _NEG + r"\s*\)"
    r"|CreateConstGEP1_(?:32|64)\s*\([^;]{0,160}?,\s*" + _NEG + r"\s*\)"
    r"|CreateConstInBoundsGEP1_(?:32|64)\s*\([^;]{0,160}?,\s*" + _NEG + r"\s*\))"
)
IR_DIRS = ("lib/backend/", "lib/repl/", "lib/core/")

RE_WITH_HEADER = re.compile(r"\b(\w*_with_header)\b")

RE_FIELD_WRITE = re.compile(
    r"\b([A-Za-z_]\w*(?:hdr|header|Header|_h)|h|hdr|header)\s*->\s*("
    + "|".join(CANONICAL_FIELDS) + r")\b"
)

RE_JS_GLUE = re.compile(
    r"(?:\b\w*_with_header\b"
    r"|\bsubtype\b|\bref_count\b|\bheaderSize\b"
    r"|setUint(?:8|16|32)\s*\(\s*[0-7]\s*,"          # writing header fields by offset
    r"|_bump\s*\([^;)]{0,60}\)\s*\+\s*8)"            # payload = block + headerSize
)

# Intra-header field offsets emitted as LLVM constants.  `subtype` sits at -8,
# but `flags` is at -7, `ref_count` at -6 and `size` at -4, and a search for -8
# finds none of them.  These are the sites that encode a field's position
# *inside* the header, so a migration that changes only the header size and
# fixes every -8 still leaves them reading the wrong field.
#
# The set searched for is derived, not guessed: it is exactly the negative
# offset each header field sits at, given the current layout.  For the 8-byte
# header that is -7 (flags), -6 (ref_count) and -4 (size); -8 (subtype) is
# class D.  Deriving the set rather than sweeping -1..-7 keeps unrelated small
# sentinels such as `ConstantInt::get(i64, -1)` out of the class.
_FIELD_OFFSETS = sorted({
    CANONICAL_HEADER_SIZE - off
    for off in (1, 2, 4)  # flags, ref_count, size — subtype at 0 is class D
})
_FIELD_ALT = "|".join(str(o) for o in _FIELD_OFFSETS)
RE_IR_FIELD_OFFSET = re.compile(
    r"(?:ConstantInt::get\s*\([^;]{0,160}?,\s*-\s*(?:" + _FIELD_ALT + r")\s*\)"
    r"|\bgetInt(?:8|16|32|64)\s*\(\s*-\s*(?:" + _FIELD_ALT + r")\s*\)"
    r"|CreateConst(?:InBounds)?GEP1_(?:32|64)\s*\([^;]{0,160}?,\s*-\s*(?:" + _FIELD_ALT + r")\s*\))"
)

# C/C++ pointer arithmetic that reconstructs the header from raw bytes without
# naming the header type or the macro.  This is the class the type-name grep and
# the macro grep both miss, and the one most likely to be forgotten: the site
# reads like ordinary byte arithmetic.
RE_RAW_BYTE_OFFSET = re.compile(
    r"(?:\(\s*(?:const\s+)?(?:unsigned\s+char|uint8_t|char|int8_t)\s*\*\s*\)"
    r"[^;]{0,100}?-\s*" + str(CANONICAL_HEADER_SIZE) + r"\b"
    r"|\*\s*\(\s*\w+\s*-\s*" + str(CANONICAL_HEADER_SIZE) + r"\s*\)"
    r"|\bptr\s*-\s*" + str(CANONICAL_HEADER_SIZE) + r"\b)"
)

# The layout restated in prose.  Comments and documentation that spell out the
# byte layout are not compiled, so nothing makes them true; they go stale
# silently and are then read as authoritative by the next person to touch the
# code.  Scanned against RAW text, deliberately including comments.
RE_PROSE_LAYOUT = re.compile(
    r"(?:\[\s*header\s*\(\s*8\s*bytes?\s*\)"
    r"|\b8-byte\s+(?:object\s+)?header\b"
    r"|header\s+is\s+at\s+(?:offset\s+)?-8\b"
    r"|\bat\s+ptr\s*-\s*8\b"
    r"|\[subtype\(1\)\]\s*\[flags\(1\)\]"
    r"|offset\s+0:\s*uint8_t\s+subtype)",
    re.IGNORECASE,
)

# Cache-key construction.  This detector looks for an ABSENCE: a file that
# computes a persistent artifact cache key and does NOT mix the object ABI into
# it will silently reuse an artifact built against a different layout.  Absence
# is the hardest thing to keep an eye on by hand, which is why it is machined.
RE_CACHE_KEY = re.compile(
    r"(?:\bhashUpdate\s*\(|\bxxh3_64bits\s*\(|stdlib-jit-v\d|\baddObjectFile\s*\()"
)
RE_ABI_IN_KEY = re.compile(
    r"(?:ESHKOL_ABI_FINGERPRINT_NAME|eshkol_abi_fingerprint_name|object-abi|ESHKOL_OBJECT_ABI_)"
)

# Struct-body parser for the parallel-declaration detector.
RE_TYPEDEF_STRUCT = re.compile(
    r"typedef\s+struct\s*(?:\w+\s*)?\{(?P<body>[^{}]*)\}\s*(?P<name>\w+)\s*;", re.S
)
RE_NAMED_STRUCT = re.compile(
    r"\bstruct\s+(?P<name>\w+)\s*\{(?P<body>[^{}]*)\}\s*;", re.S
)
RE_FIELD = re.compile(r"\b(?P<type>u?int(?:8|16|32|64)_t|unsigned\s+\w+|\w+)\s+(?P<name>\w+)\s*;")


def _canonical_field_signature() -> list[tuple[str, str]]:
    return [
        ("uint8_t", "subtype"),
        ("uint8_t", "flags"),
        ("uint16_t", "ref_count"),
        ("uint32_t", "size"),
    ]


def scan_lexical(root: Path) -> list[Site]:
    sites: list[Site] = []
    canonical_sig = _canonical_field_signature()

    for path in tracked_sources(root):
        rel = str(path.relative_to(root))
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        suffix = path.suffix

        # L — the layout restated in prose. Runs against raw text on every file
        # type, because a stale comment in a .md is as misleading as one in a .c.
        for m in RE_PROSE_LAYOUT.finditer(raw):
            ln = line_of(raw, m.start())
            sites.append(Site("L_layout_in_prose", rel, ln, snippet_at(raw, ln), "L_layout_in_prose"))
        if suffix in PROSE_FAMILY:
            continue

        stripped = strip_c_comments_and_strings(raw)

        if suffix in JS_FAMILY:
            for m in RE_JS_GLUE.finditer(stripped):
                ln = line_of(stripped, m.start())
                sites.append(Site("H_wasm_glue", rel, ln, snippet_at(raw, ln), "H_wasm_glue"))
            continue

        claimed: set[int] = set()

        # A — macro family
        for m in RE_MACRO.finditer(stripped):
            ln = line_of(stripped, m.start())
            claimed.add(ln)
            sites.append(Site("A_macro_api", rel, ln, snippet_at(raw, ln), "A_macro_api"))

        # C — header size as arithmetic
        for m in RE_SIZEOF_HEADER.finditer(stripped):
            ln = line_of(stripped, m.start())
            claimed.add(ln)
            sites.append(Site("C_sizeof_header", rel, ln, snippet_at(raw, ln), "C_sizeof_header"))

        # B — bare references to the header type
        for m in RE_HEADER_TYPE.finditer(stripped):
            ln = line_of(stripped, m.start())
            if ln in claimed:
                continue
            sites.append(Site("B_header_type", rel, ln, snippet_at(raw, ln), "B_header_type"))

        # D / D2 — LLVM IR constant offsets: the header base, and the field
        # positions inside it
        if any(rel.startswith(d) for d in IR_DIRS):
            for m in RE_IR_CONST.finditer(stripped):
                ln = line_of(stripped, m.start())
                sites.append(Site("D_ir_const_offset", rel, ln, snippet_at(raw, ln), "D_ir_const_offset"))
            for m in RE_IR_FIELD_OFFSET.finditer(stripped):
                ln = line_of(stripped, m.start())
                sites.append(Site("D2_ir_field_offset", rel, ln, snippet_at(raw, ln),
                                  "D2_ir_field_offset"))

        # K — raw byte arithmetic that rebuilds the header without saying so
        for m in RE_RAW_BYTE_OFFSET.finditer(stripped):
            ln = line_of(stripped, m.start())
            sites.append(Site("K_raw_byte_offset", rel, ln, snippet_at(raw, ln), "K_raw_byte_offset"))

        # M — persistent cache keys that do not mix in the object ABI
        if RE_CACHE_KEY.search(stripped) and not RE_ABI_IN_KEY.search(stripped):
            m = RE_CACHE_KEY.search(stripped)
            ln = line_of(stripped, m.start())
            sites.append(Site("M_cache_key_without_abi", rel, ln, snippet_at(raw, ln),
                              "M_cache_key_without_abi"))

        # E — the writer half
        for m in RE_WITH_HEADER.finditer(stripped):
            ln = line_of(stripped, m.start())
            sites.append(Site("E_alloc_with_header", rel, ln, snippet_at(raw, ln), "E_alloc_with_header"))

        # F — structural duplicates of the header
        for rx in (RE_TYPEDEF_STRUCT, RE_NAMED_STRUCT):
            for m in rx.finditer(stripped):
                fields = [(re.sub(r"\s+", " ", f.group("type")).strip(), f.group("name"))
                          for f in RE_FIELD.finditer(m.group("body"))]
                if fields == canonical_sig:
                    ln = line_of(stripped, m.start())
                    sites.append(Site("F_parallel_header_decl", rel, ln,
                                      f"{m.group('name')}: " + snippet_at(raw, ln),
                                      "F_parallel_header_decl"))

        # G / J — field access through a header-ish pointer.  The file-level
        # precondition (the file must name a header type at all) is what keeps
        # this from claiming every `->size` in the tree; the choice between G and
        # J is by which header ABI the file actually names.
        names_object_header = bool(RE_HEADER_TYPE.search(stripped))
        names_shared_header = bool(RE_SECONDARY_HEADER.search(stripped))
        if names_object_header or names_shared_header:
            target = "G_header_field_write" if names_object_header else "J_secondary_prefix_abi"
            for m in RE_FIELD_WRITE.finditer(stripped):
                ln = line_of(stripped, m.start())
                if ln in claimed:
                    continue
                sites.append(Site(target, rel, ln, snippet_at(raw, ln), target))

        # J — the second prefix-header ABI, by name
        for m in RE_SECONDARY_HEADER.finditer(stripped):
            ln = line_of(stripped, m.start())
            sites.append(Site("J_secondary_prefix_abi", rel, ln, snippet_at(raw, ln),
                              "J_secondary_prefix_abi"))

    # I — the subset that is a public ABI commitment.  A site inside an
    # installed header is not merely internal: embedders compiled against it, so
    # it cannot be changed by rebuilding this repository alone.
    public = [s for s in sites if s.path.startswith("inc/eshkol/")]
    for s in public:
        sites.append(Site("I_public_abi", s.path, s.line, s.snippet, s.detector))

    # One finding per class per line.  Two detectors firing on the same line is
    # one site to fix, and counting it twice would make the ratchet churn on
    # unrelated edits.
    deduped: dict[tuple[str, str, int], Site] = {}
    for s in sites:
        deduped.setdefault((s.cls, s.path, s.line), s)
    return list(deduped.values())


# ── Semantic detectors (libclang) ────────────────────────────────────────────
def _toolchain_flags() -> list[str]:
    """Flags libclang needs that the compile database does not carry.

    compile_commands.json records the flags the *build* compiler was given. A
    standalone libclang has neither that compiler's implicit sysroot nor its
    builtin-header resource directory, so without these every translation unit
    fails at `#include <stdint.h>` and the semantic layer silently reports zero
    findings — the exact false-green this inventory exists to prevent. Missing
    flags are therefore fatal to the layer, not skipped.
    """
    flags: list[str] = []
    if sys.platform == "darwin":
        sdk = subprocess.run(["xcrun", "--show-sdk-path"],
                             capture_output=True, text=True)
        if sdk.returncode == 0 and sdk.stdout.strip():
            flags += ["-isysroot", sdk.stdout.strip()]
    for clang_bin in ("/opt/homebrew/opt/llvm@21/bin/clang", "clang"):
        rd = subprocess.run([clang_bin, "-print-resource-dir"],
                            capture_output=True, text=True)
        if rd.returncode == 0 and rd.stdout.strip():
            flags += ["-resource-dir", rd.stdout.strip()]
            break
    return flags


def scan_semantic(root: Path, compdb: Path) -> tuple[list[Site], str | None]:
    try:
        import clang.cindex as ci  # type: ignore
    except ImportError:
        return [], "libclang python bindings not importable (pip install libclang)"

    for candidate in (
        os.environ.get("ESHKOL_LIBCLANG"),
        "/opt/homebrew/opt/llvm@21/lib/libclang.dylib",
        "/opt/homebrew/opt/llvm/lib/libclang.dylib",
        "/usr/lib/llvm-21/lib/libclang.so.1",
    ):
        if candidate and Path(candidate).exists():
            try:
                ci.Config.set_library_file(candidate)
            except Exception:
                pass
            break

    if not compdb.exists():
        return [], f"no compile_commands.json at {compdb}"

    try:
        index = ci.Index.create()
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], f"libclang unusable: {exc}"

    extra_flags = _toolchain_flags()

    header_types = set(HEADER_TYPE_NAMES)
    sites: list[Site] = []
    seen: set[tuple[str, str, int]] = set()

    entries = json.loads(compdb.read_text())
    # Only translation units that can plausibly touch the header, so the walk is
    # bounded.  A TU is a candidate if its text mentions any header type name,
    # any macro in the family, or a negated header offset.
    candidates = []
    for e in entries:
        f = Path(e["file"])
        if not f.exists():
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if RE_HEADER_TYPE.search(txt) or RE_MACRO.search(txt) or RE_IR_CONST.search(txt):
            candidates.append(e)

    def is_header_type(t) -> bool:
        """True if t is, or points to, one of the header types.

        Compares against the canonical spelling as well as the written one, so a
        typedef, a `const` qualifier or a `struct` elaboration cannot hide the
        type — which is the whole reason this layer exists alongside the lexical
        one.
        """
        cur = t
        for _ in range(4):
            for spelling in (cur.spelling, cur.get_canonical().spelling):
                for name in header_types:
                    if re.search(r"\b" + re.escape(name) + r"\b", spelling):
                        return True
            if cur.kind == ci.TypeKind.POINTER:
                cur = cur.get_pointee()
            else:
                break
        return False

    def record(cls: str, cursor) -> None:
        loc = cursor.location
        if loc.file is None:
            return
        try:
            rel = str(Path(loc.file.name).resolve().relative_to(root))
        except ValueError:
            return
        if any(rel.startswith(p) for p in EXCLUDED_PREFIXES):
            return
        key = (cls, rel, loc.line)
        if key in seen:
            return
        seen.add(key)
        sites.append(Site(cls, rel, loc.line, "", cls))

    import shlex

    # The cursor kind for sizeof/alignof is spelled UNARY_EXPR in some libclang
    # releases and CXX_UNARY_EXPR in others. Resolve it once rather than letting
    # the walk raise on a version the tool was not written against.
    _UNARY_EXPR_KIND = getattr(ci.CursorKind, "UNARY_EXPR", None) or \
        getattr(ci.CursorKind, "CXX_UNARY_EXPR")

    cwd = Path.cwd()
    parsed = 0
    failed: list[str] = []
    for e in candidates:
        toks = shlex.split(e["command"])
        args, skip = [], False
        for tok in toks[1:]:
            if skip:
                skip = False
                continue
            if tok == "-o":
                skip = True
                continue
            if tok == "-c" or tok == e["file"] or tok.endswith(".o"):
                continue
            args.append(tok)
        args += extra_flags
        try:
            os.chdir(e["directory"])
            tu = index.parse(e["file"], args=args)
        except Exception:
            os.chdir(cwd)
            failed.append(e["file"])
            continue
        os.chdir(cwd)
        if tu is None:
            failed.append(e["file"])
            continue
        fatal = [d for d in tu.diagnostics if d.severity >= 3]
        if fatal:
            # A TU that did not parse contributes zero findings and would read as
            # "clean". Record it instead, so an unusable toolchain surfaces as an
            # incomplete scan rather than a reassuring one.
            failed.append(f"{e['file']}: {fatal[0].spelling}")
            continue
        parsed += 1
        for cursor in tu.cursor.walk_preorder():
            try:
                k = cursor.kind
            except ValueError:
                # The python bindings and the loaded libclang can be different
                # releases; a cursor kind the bindings do not know raises rather
                # than compares false. Skip the node instead of aborting the
                # whole scan, which would silently reduce the semantic layer to
                # nothing.
                continue
            if k == ci.CursorKind.MEMBER_REF_EXPR:
                for child in cursor.get_children():
                    if is_header_type(child.type):
                        record("S1_clang_header_member", cursor)
                        break
            elif k == _UNARY_EXPR_KIND:
                for child in cursor.get_children():
                    if is_header_type(child.type):
                        record("S2_clang_sizeof", cursor)
                        break
            elif k in (ci.CursorKind.CSTYLE_CAST_EXPR, ci.CursorKind.CXX_STATIC_CAST_EXPR,
                       ci.CursorKind.CXX_REINTERPRET_CAST_EXPR):
                if is_header_type(cursor.type):
                    record("S3_clang_header_cast", cursor)

    note = None
    if failed:
        note = (f"{len(failed)}/{len(candidates)} translation units did not parse "
                f"(first: {failed[0]}); semantic layer is INCOMPLETE")
    return sites, note


# ── Emitted-IR detector ──────────────────────────────────────────────────────
IR_CORPUS = [
    ('(define (f x) (car x))\n(display (f (list 1 2 3)))\n(newline)\n', "cons"),
    ('(display (string-length "hello"))\n(newline)\n', "string"),
    ('(define v (vector 1 2 3))\n(display (vector-ref v 1))\n(newline)\n', "vector"),
    ('(define h (make-hash-table))\n(hash-table-set! h (quote a) 1)\n'
     '(display (hash-table-ref h (quote a)))\n(newline)\n', "hash"),
    ('(define (g x) (* x x))\n(display ((lambda (y) (g y)) 4))\n(newline)\n', "closure"),
]
RE_EMITTED_GEP = re.compile(
    r"getelementptr\s+(?:inbounds\s+)?i8,\s*ptr\s+\S+,\s*i64\s+-" + str(CANONICAL_HEADER_SIZE) + r"\b"
)


def _repo_relative(path: Path, root: Path) -> str:
    """Render `path` relative to `root` for anything that gets committed.

    The absolute form embeds the local checkout's directory structure (and,
    on most machines, the user's account name) into a snapshot every
    contributor commits — a machine detail the artifact has no business
    carrying. Every path recorded in the inventory is repo-relative for the
    same reason `git ls-files` already is; this is the one path the emitted
    layer computes itself rather than reading from `git ls-files`.
    """
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def scan_emitted(root: Path, compiler: Path) -> dict:
    """Compile a corpus and count header offsets that actually reach the IR."""
    if not compiler.exists():
        return {"status": "skipped", "reason": f"compiler not built at {_repo_relative(compiler, root)}"}

    scratch = root / ".scratch" / "abi-ir"
    scratch.mkdir(parents=True, exist_ok=True)
    result = {"status": "ran", "compiler": _repo_relative(compiler, root), "programs": {}}
    total = 0
    for src, name in IR_CORPUS:
        d = scratch / name
        d.mkdir(exist_ok=True)
        esk = d / f"{name}.esk"
        esk.write_text(src)
        proc = subprocess.run(
            [str(compiler), "--dump-ir", "-c", esk.name, "-o", f"{name}.o"],
            cwd=d, capture_output=True, text=True, timeout=300,
        )
        ll = d / f"{name}.o.ll"
        if not ll.exists():
            result["programs"][name] = {"status": "no-ir", "stderr": proc.stderr[-400:]}
            continue
        hits = len(RE_EMITTED_GEP.findall(ll.read_text(errors="replace")))
        total += hits
        result["programs"][name] = {"status": "ok", "header_geps": hits}
    result["total_header_geps"] = total
    return result


# ── Report assembly ──────────────────────────────────────────────────────────
def verify_layout_pin(root: Path) -> dict:
    """Confirm the tool is scanning for the layout the source actually declares.

    If someone changes the header without updating this tool, the constants the
    detectors search for go stale and the inventory silently under-reports.  This
    check makes that a hard failure instead.
    """
    hdr = (root / "inc" / "eshkol" / "eshkol.h").read_text(errors="replace")
    m = re.search(
        r"typedef\s+struct\s+eshkol_object_header\s*\{(?P<body>.*?)\}\s*eshkol_object_header_t\s*;",
        hdr, re.S,
    )
    if not m:
        return {"ok": False, "reason": "canonical header struct not found in inc/eshkol/eshkol.h"}
    fields = [(re.sub(r"\s+", " ", f.group("type")).strip(), f.group("name"))
              for f in RE_FIELD.finditer(strip_c_comments_and_strings(m.group("body")))]
    expected = _canonical_field_signature()
    if fields != expected:
        return {"ok": False, "reason": f"header fields changed: {fields} != {expected}",
                "hint": "update CANONICAL_* in scripts/abi_header_inventory.py, then re-baseline"}
    if not re.search(r"sizeof\(eshkol_object_header_t\)\s*==\s*" + str(CANONICAL_HEADER_SIZE), hdr):
        return {"ok": False, "reason": f"static assert for size {CANONICAL_HEADER_SIZE} not found"}
    return {"ok": True, "header_size": CANONICAL_HEADER_SIZE, "fields": expected}


def build_report(root: Path, use_clang: bool, use_emitted: bool, compdb: Path,
                 compiler: Path) -> dict:
    pin = verify_layout_pin(root)
    sites = scan_lexical(root)
    semantic_note = None
    if use_clang:
        sem, semantic_note = scan_semantic(root, compdb)
        sites.extend(sem)

    by_class: dict[str, dict[str, int]] = {}
    for s in sites:
        by_class.setdefault(s.cls, {}).setdefault(s.path, 0)
        by_class[s.cls][s.path] += 1

    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    report = {
        "schema": 1,
        "tool": "scripts/abi_header_inventory.py",
        "commit": head,
        "layout_pin": pin,
        "detectors": DETECTORS,
        "semantic_layer": {
            "requested": use_clang,
            "note": semantic_note,
            "complete": bool(use_clang and semantic_note is None),
        },
        "totals": {
            "sites": len([s for s in sites if s.cls != "I_public_abi"]),
            "sites_including_public_view": len(sites),
            "classes": {c: sum(v.values()) for c, v in sorted(by_class.items())},
            "files": len({s.path for s in sites}),
        },
        "counts": {c: dict(sorted(v.items())) for c, v in sorted(by_class.items())},
        "sites": [asdict(s) for s in sorted(sites, key=lambda s: (s.cls, s.path, s.line))],
    }
    if use_emitted:
        report["emitted_ir"] = scan_emitted(root, compiler)
    return report


def render_text(report: dict) -> str:
    lines = []
    lines.append("Object-header layout dependence — machine inventory")
    lines.append("=" * 66)
    lines.append("")
    lines.append(f"commit        {report['commit']}")
    pin = report["layout_pin"]
    lines.append(f"layout pin    {'OK' if pin.get('ok') else 'FAILED: ' + pin.get('reason', '')}")
    lines.append(f"total sites   {report['totals']['sites']}")
    lines.append(f"files         {report['totals']['files']}")
    sem = report["semantic_layer"]
    if not sem["requested"]:
        lines.append("semantic      not requested (--clang)")
    elif sem["complete"]:
        lines.append("semantic      complete")
    else:
        lines.append(f"semantic      INCOMPLETE — {sem['note']}")
    if "emitted_ir" in report:
        e = report["emitted_ir"]
        lines.append(f"emitted IR    {e.get('status')} — {e.get('total_header_geps', 'n/a')} header GEPs across the corpus")
    lines.append("")
    lines.append("Class breakdown")
    lines.append("-" * 66)
    for cls, count in report["totals"]["classes"].items():
        d = DETECTORS.get(cls, {})
        files = len(report["counts"].get(cls, {}))
        lines.append(f"  {cls:<26} {count:>5} sites  {files:>3} files  [{d.get('layer', '?')}]")
        lines.append(f"      what: {d.get('what', '')}")
        lines.append(f"      how:  {' '.join(d.get('how', '').split())}")
        lines.append("")
    lines.append("Per-file counts")
    lines.append("-" * 66)
    for cls, files in report["counts"].items():
        lines.append(f"  {cls}")
        for path, n in sorted(files.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"      {n:>4}  {path}")
        lines.append("")
    lines.append("Regenerate with: scripts/abi_header_inventory.py snapshot --clang --emitted")
    return "\n".join(lines) + "\n"


# ── Ratchet ──────────────────────────────────────────────────────────────────
def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text())


def do_check(report: dict, baseline: dict) -> int:
    """Fail on any class/file pair above baseline, or in a file not baselined."""
    if not baseline:
        print("FAIL: no baseline at .icc/abi-header-baseline.json; run `baseline` first", file=sys.stderr)
        return 2

    pin = report["layout_pin"]
    if not pin.get("ok"):
        print(f"FAIL: layout pin — {pin.get('reason')}", file=sys.stderr)
        print(f"      {pin.get('hint', '')}", file=sys.stderr)
        return 2

    base_counts = baseline.get("counts", {})
    violations = []
    slack = []
    for cls, files in report["counts"].items():
        if cls == "I_public_abi":
            # Derived view; enforced through its source classes.
            continue
        base_files = base_counts.get(cls, {})
        for path, n in files.items():
            b = base_files.get(path)
            if b is None:
                violations.append((cls, path, 0, n, "file not in baseline"))
            elif n > b:
                violations.append((cls, path, b, n, "count above baseline"))
            elif n < b:
                slack.append((cls, path, b, n))
        for path, b in base_files.items():
            if path not in files:
                slack.append((cls, path, b, 0))

    if slack:
        print("Sites removed since baseline (re-run `baseline` to tighten the ratchet):")
        for cls, path, b, n in sorted(slack):
            print(f"  {cls}: {path}  {b} -> {n}")
        print()

    if violations:
        sys.stdout.flush()
        print("FAIL: new object-header layout dependence introduced.", file=sys.stderr)
        print(file=sys.stderr)
        for cls, path, b, n, why in sorted(violations):
            print(f"  {cls}: {path}  baseline {b}, now {n}  ({why})", file=sys.stderr)
            # The ratchet counts per (class, file), so it knows one more site
            # exists but not which. Every site of that class in that file is
            # listed; the new one is among them.
            for s in report["sites"]:
                if s["cls"] == cls and s["path"] == path:
                    print(f"        {path}:{s['line']}  {s['snippet']}", file=sys.stderr)
        print(file=sys.stderr)
        print("The header layout is scheduled to change. Every site listed above will", file=sys.stderr)
        print("have to move with it, and the header carries no discriminator, so a site", file=sys.stderr)
        print("missed by the migration produces wrong data rather than an error.", file=sys.stderr)
        print("Route new header access through the single accessor in", file=sys.stderr)
        print("inc/eshkol/abi_fingerprint.h / the ESHKOL_GET_HEADER family instead of", file=sys.stderr)
        print("computing the offset again, or re-baseline deliberately with a reason.", file=sys.stderr)
        return 1

    print(f"OK: {report['totals']['sites']} layout-dependent sites, all baselined.")
    return 0


def emit_trace(root: Path, report: dict, rc: int) -> None:
    """Write ICC runtime evidence for INV-object-abi-site-ratchet.

    A gate whose result is not recorded is a gate the readiness oracle cannot
    see. The event carries the site total so a silent collapse in detector
    coverage — a scan that suddenly finds nothing and therefore passes — reads
    as suspicious in the trace rather than as success.
    """
    trace_dir = root / "scripts" / "icc_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "kind": "eshkol_smoke",
        "name": "abi_object_header_ratchet",
        "value": "PASS" if rc == 0 else "FAIL",
        "snippet": (f"{report['totals']['sites']} layout-dependent sites across "
                    f"{report['totals']['files']} files; "
                    f"layout pin {'ok' if report['layout_pin'].get('ok') else 'FAILED'}; "
                    f"classes {report['totals']['classes']}"),
        "confidence": 0.99,
    }
    with (trace_dir / "abi_object_ratchet.jsonl").open("a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def do_baseline(report: dict) -> None:
    counts = {c: v for c, v in report["counts"].items() if c != "I_public_abi"}
    payload = {
        "schema": 1,
        "generated_by": "scripts/abi_header_inventory.py baseline",
        "commit": report["commit"],
        "layout": report["layout_pin"],
        "note": (
            "Ratchet baseline for object-header layout dependence. Counts are per "
            "(class, file). A commit may not raise any count or add a file. "
            "Lowering counts is always allowed; re-run `baseline` afterwards to "
            "tighten the ratchet. Raising one requires an explicit, reasoned "
            "re-baseline in the same commit."
        ),
        "totals": {c: sum(v.values()) for c, v in counts.items()},
        "counts": counts,
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {BASELINE_PATH.relative_to(REPO_ROOT)}: "
          f"{sum(payload['totals'].values())} sites across "
          f"{len({p for v in counts.values() for p in v})} files")


# ── Self-test: prove the gate can go red ─────────────────────────────────────
SELFTEST_INJECTION = """
/* abi_header_inventory selftest injection — not real code. */
static inline unsigned char selftest_raw_subtype(void *p) {
    return *((unsigned char *)p - sizeof(eshkol_object_header_t));
}
"""


def do_selftest(root: Path) -> int:
    """Feed the gate a deliberately-new raw header site and require it to fail.

    A gate that has never been observed red is an assumption, not a gate.
    """
    print("selftest: proving the ratchet goes red on a newly introduced site")
    print()

    baseline = load_baseline()
    if not baseline:
        print("FAIL: no baseline; run `baseline` first", file=sys.stderr)
        return 2

    report = build_report(root, False, False, root / "build" / "compile_commands.json", Path("/nonexistent"))
    rc_clean = do_check(report, baseline)
    print(f"  [1/3] clean tree -> exit {rc_clean} (expected 0)")
    if rc_clean != 0:
        print("FAIL: baseline does not match the clean tree", file=sys.stderr)
        return 1

    victim = root / "lib" / "core" / "runtime_object_alloc.cpp"
    original = victim.read_text()
    try:
        victim.write_text(original + SELFTEST_INJECTION)
        report_dirty = build_report(root, False, False,
                                    root / "build" / "compile_commands.json", Path("/nonexistent"))
        print("  [2/3] injected one new sizeof(eshkol_object_header_t) site into "
              f"{victim.relative_to(root)}")
        rc_dirty = do_check(report_dirty, baseline)
        print(f"        -> exit {rc_dirty} (expected 1)")
    finally:
        victim.write_text(original)

    report_restored = build_report(root, False, False,
                                   root / "build" / "compile_commands.json", Path("/nonexistent"))
    rc_restored = do_check(report_restored, baseline)
    print(f"  [3/3] injection removed -> exit {rc_restored} (expected 0)")
    print()

    if rc_dirty == 1 and rc_restored == 0:
        print("selftest PASS: the ratchet is red on a new site and green without it.")
        return 0
    print("selftest FAIL", file=sys.stderr)
    return 1


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scan", "snapshot", "check", "baseline", "selftest"])
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--clang", action="store_true",
                    help="also run the libclang semantic layer (needs compile_commands.json)")
    ap.add_argument("--emitted", action="store_true",
                    help="also compile a corpus and count header offsets in emitted IR")
    ap.add_argument("--compile-commands", type=Path, default=None)
    ap.add_argument("--compiler", type=Path, default=None,
                    help="path to eshkol-run for the emitted-IR layer")
    ap.add_argument("--trace", action="store_true",
                    help="on `check`, append ICC runtime evidence to scripts/icc_traces/")
    args = ap.parse_args()

    root = args.repo.resolve()
    compdb = args.compile_commands or (root / "build" / "compile_commands.json")
    compiler = args.compiler or (root / "build" / "eshkol-run")

    if args.command == "selftest":
        return do_selftest(root)

    report = build_report(root, args.clang, args.emitted, compdb, compiler)

    if args.command == "scan":
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "snapshot":
        SNAPSHOT_JSON.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        SNAPSHOT_TEXT.write_text(render_text(report))
        print(render_text(report))
        print(f"wrote {SNAPSHOT_JSON.relative_to(REPO_ROOT)}")
        print(f"wrote {SNAPSHOT_TEXT.relative_to(REPO_ROOT)}")
        return 0

    if args.command == "baseline":
        do_baseline(report)
        return 0

    rc = do_check(report, load_baseline())
    if args.trace:
        emit_trace(root, report, rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
