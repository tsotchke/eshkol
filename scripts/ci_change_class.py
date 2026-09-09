#!/usr/bin/env python3
"""Classify a PR's changed-file set by real build/CI impact.

Motivating incident: PR #624 touched only `lib/backend/eshkol_compiler.c`,
a file `CMakeLists.txt` explicitly excludes from the source glob it
compiles (`list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/[^/]*\\.c$")`,
kept as a reference-only standalone program) -- yet the PR ran the full
25-job cross-platform matrix. #618 and #619 each touched one test shell
script. None of the three could possibly change what the build or test
suite consumes, but the existing `docs_only` predicate in
`.github/workflows/ci.yml`'s `changes` job only recognises documentation
paths (`*.md`, `docs/**`, `notes/**`, `press/**`, `.swarm/**`, `LICENSE`)
-- everything else, including a change CMake itself declares irrelevant,
runs the expensive matrix.

This script computes a coarser-grained but still-safe classification by
asking the real build and CI inputs what they actually consume, instead of
hand-maintaining a second path list that would drift from `CMakeLists.txt`
the same way `docs-only-required-context-stubs` once drifted from branch
protection (see `scripts/check_required_context_consistency.py`).

Ground truth sources (never hardcoded file names -- always re-derived from
the current tree):

  * `CMakeLists.txt` and every `cmake/*.cmake` file it `include()`s --
    parsed (not executed) for `file(GLOB[_RECURSE] ...)` source globs,
    `list(APPEND|REMOVE_ITEM|FILTER ... EXCLUDE REGEX ...)` mutations of
    those source-list variables, explicit `set(<VAR> path...)` source
    lists, `add_executable`/`add_library`/`target_sources` source
    arguments, `target_include_directories` (an entire directory such as
    `inc/` is a build input the moment anything in it can be #included),
    `configure_file` inputs, and `add_custom_command` COMMAND/DEPENDS
    arguments (this is how generated-header producers like
    `cmake/gen_runtime_def.cmake`, `cmake/embed_metal_shader.cmake`, and
    the precompiled-stdlib step -- `lib/stdlib.esk` plus every `.esk`
    module it transitively `(require)`s, globbed with `CONFIGURE_DEPENDS`
    at CMakeLists.txt:2716 -- are picked up without a single filename
    typed into this script).
  * `.github/workflows/*.yml` -- every matrix `test_command`/`test_mode`
    field and every step's `run:` block is scanned for `scripts/`,
    `tests/`, and `examples/` references actually invoked by CI, plus the
    workflow files themselves (a workflow change is always `full`).
  * `add_test(...)` / `add_custom_target(...)` in CMakeLists.txt -- their
    COMMAND/DEPENDS arguments are CI/test inputs (this is how
    `scripts/regenerate_vm_prelude_cache.sh`, exercised by the
    `vm_prelude_cache_is_current` ctest, is discovered).
  * `tests/**` and `.icc/**` (read throughout CI's assurance gates --
    over a hundred references across `.github/workflows/*.yml` and
    `scripts/*.py`) unconditionally; `examples/**` only if some test
    runner is ever found to execute it (empirically, none does today, so
    an `examples/*.esk` edit alone currently classifies as `non-build` --
    this is a live fact re-derived every run, not a policy decision baked
    into this script).

Validation: when `--build-dir <dir>/compile_commands.json` exists (a real
configured build), every compiled translation unit's `file` entry MUST
appear in the derived build-input set -- a self-consistency check that
fails loud (nonzero exit, `"class": null`) rather than silently trusting
the CMake-parsing heuristics. No build is configured in the fast `changes`
job that calls this script, so that check is normally a no-op there; it
exists for local/offline use and is exercised directly by `--self-test`
against a synthetic fixture tree.

Classes (in priority order; the first that matches every changed file
wins; a heterogeneous set that satisfies none of the narrower classes
falls through to `full`):

  docs        Every file matches the pre-existing docs-only predicate
              (`*.md`, `docs/*`, `notes/*`, `press/*`, `.swarm/*`,
              `LICENSE` -- the exact case pattern `changes` used before
              this script existed; semantics intentionally unchanged).
  non-build   Every file is consumed by neither the build nor CI/tests,
              and is not a workflow file.
  tests-only  Every file is under `tests/`, or is a script CI actually
              runs as a test (an `add_test`/`add_custom_target` COMMAND
              argument, or a workflow matrix `test_command`/`run:` step
              reference), and none is a build input or workflow file.
              Informational only for now: still runs the full matrix.
  full        Anything else. A `.github/workflows/**` change is always
              `full`, independent of every other rule.

Usage
    git diff --name-only origin/master...HEAD | python3 scripts/ci_change_class.py
    python3 scripts/ci_change_class.py lib/backend/eshkol_compiler.c
    python3 scripts/ci_change_class.py --pr 624
    python3 scripts/ci_change_class.py --self-test
    python3 scripts/ci_change_class.py --build-dir build --self-test

Output: a JSON object on stdout (`class`, `summary`, `files[]` each with
its own `reasons[]`, and a `derivation` block naming what was parsed) plus
a one-line human summary on stderr. Exit status: 0 on a successful
classification (any class) or a passing `--self-test`; 1 on a usage error,
an unparseable required input, a failing self-consistency check, or a
failing `--self-test`.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The exact predicate `changes` used before this script existed (and still
# uses, via the `impact` output this script now feeds it) -- kept
# byte-for-byte so the `docs` class's semantics never drift from it.
# A bash `case` pattern's `*` matches `/` too, so `*.md` here already means
# `**/*.md` and `docs/*` means `docs/**` -- fnmatch's `*` has the same
# "matches anything, including `/`" behaviour, so fnmatchcase reproduces it
# exactly.
DOCS_ONLY_PATTERNS = ("*.md", "docs/*", "notes/*", "press/*", ".swarm/*", "LICENSE")

CLASS_DOCS = "docs"
CLASS_NON_BUILD = "non-build"
CLASS_TESTS_ONLY = "tests-only"
CLASS_FULL = "full"

CMAKE_PATH_VARS = (
    "CMAKE_CURRENT_SOURCE_DIR",
    "CMAKE_SOURCE_DIR",
    "CMAKE_CURRENT_LIST_DIR",
    "PROJECT_SOURCE_DIR",
)

# CMake keywords that can appear as bare arguments to add_executable /
# add_library / target_sources / target_include_directories and are never
# themselves a source or include path.
_TARGET_DECL_KEYWORDS = {
    "STATIC", "SHARED", "MODULE", "OBJECT", "INTERFACE", "IMPORTED",
    "GLOBAL", "ALIAS", "EXCLUDE_FROM_ALL", "WIN32", "MACOSX_BUNDLE",
}
_VISIBILITY_KEYWORDS = {"PRIVATE", "PUBLIC", "INTERFACE"}
_INCLUDE_DIR_KEYWORDS = {"PRIVATE", "PUBLIC", "INTERFACE", "SYSTEM", "BEFORE", "AFTER"}

# `target_include_directories` marks a whole directory as reachable via
# `#include`, but not every file under it is a compiler input the way a
# glob-matched/target-listed *source* is -- lib/backend, for instance, is
# on the include path for a couple of standalone generator executables
# (so a real header living there is fair game for #include) while also
# containing lib/backend/eshkol_compiler.c, a .c file CMakeLists.txt
# explicitly excludes from every source list (list(FILTER LIB_SRC EXCLUDE
# REGEX "lib/backend/[^/]*\\.c$")). Treating "under an include dir" as
# "is a build input" without this filter would make that exact file --
# this script's own motivating PR #624 case -- register as `full` again.
# Restricting the include-directory rule to header-shaped extensions
# keeps `inc/**/*.h` correctly `full` while leaving a source file's real
# compiled-or-excluded status to the glob/target derivation above.
HEADER_EXTENSIONS = {".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp"}

_ADD_CUSTOM_COMMAND_KEYWORDS = {
    "OUTPUT", "COMMAND", "MAIN_DEPENDENCY", "DEPENDS", "BYPRODUCTS",
    "IMPLICIT_DEPENDS", "WORKING_DIRECTORY", "COMMENT", "DEPFILE",
    "JOB_POOL", "VERBATIM", "APPEND", "USES_TERMINAL",
    "COMMAND_EXPAND_LISTS", "TARGET", "PRE_BUILD", "PRE_LINK", "POST_BUILD",
}

VAR_REF_FULL_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
VAR_REF_EMBEDDED_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

SCRIPT_REF_RE = re.compile(
    r"(?:\./|\.\\)?((?:scripts|tests|examples)[/\\][A-Za-z0-9_./\\-]+\.(?:sh|py|ps1|js))"
)


class CMakeParseError(Exception):
    """A required CMake input could not be read at all (fails closed)."""


# ───────────────────────── CMake tokenizing ─────────────────────────

def _strip_cmake_comments(text: str) -> str:
    """Remove `#`-to-end-of-line comments, respecting quoted strings."""
    out = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _iter_cmake_commands(text: str):
    """Yield (lowercased_command_name, raw_args_string) for every top-level
    command invocation in a CMake file's text.

    Deliberately flat: `if()`/`foreach()`/`endif()`/`endforeach()` are not
    specially handled, so a command inside a conditional or loop body is
    still found (its own `(...)`  closes right after its own arguments --
    CMake bodies are not textually nested inside the control command's
    parens). This means every branch of every `if()` is treated as if it
    always executes, which is the deliberately conservative choice: a
    source that only exists on one platform's branch is never missed, at
    the cost of very occasionally over-including a file that is dead on
    the classifier's own platform.
    """
    n = len(text)
    i = 0
    ident_start = re.compile(r"[A-Za-z_]")
    while i < n:
        c = text[i]
        if ident_start.match(c):
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            name = text[i:j]
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k < n and text[k] == "(":
                depth = 1
                p = k + 1
                start_args = p
                in_string = False
                while p < n and depth > 0:
                    ch = text[p]
                    if in_string:
                        if ch == "\\" and p + 1 < n:
                            p += 2
                            continue
                        if ch == '"':
                            in_string = False
                        p += 1
                        continue
                    if ch == '"':
                        in_string = True
                        p += 1
                        continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    p += 1
                yield name.lower(), text[start_args : p - 1]
                i = p
                continue
            i = j
            continue
        i += 1


def _tokenize_args(args_str: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(args_str)
    while i < n:
        c = args_str[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n and args_str[j] != '"':
                if args_str[j] == "\\" and j + 1 < n:
                    buf.append(args_str[j + 1])
                    j += 2
                    continue
                buf.append(args_str[j])
                j += 1
            tokens.append("".join(buf))
            i = j + 1
            continue
        j = i
        while j < n and args_str[j] not in " \t\r\n":
            j += 1
        tokens.append(args_str[i:j])
        i = j
    return tokens


# ───────────────────────── variable resolution ─────────────────────────

def _resolve_token(token: str, variables: dict[str, list[str]]) -> list[str]:
    """Expand `${VAR}` references. A token that is EXACTLY `${VAR}` expands
    to the variable's whole list (CMake list semantics); a `${VAR}`
    embedded in a larger string is substituted with the variable's first
    value (adequate here -- every such use in this file is a directory
    prefix variable with exactly one value). A token left containing an
    unresolved `${...}` after substitution is dropped rather than guessed
    at.
    """
    stripped = token.strip()
    m = VAR_REF_FULL_RE.match(stripped)
    if m:
        return list(variables.get(m.group(1), []))
    if "${" not in token:
        return [token]

    def repl(mm: re.Match) -> str:
        vals = variables.get(mm.group(1))
        if vals is None:
            return mm.group(0)
        return vals[0] if vals else ""

    new = VAR_REF_EMBEDDED_RE.sub(repl, token)
    if "${" in new:
        return []
    return [new]


def _to_repo_relative(value: str, repo_root: Path) -> str | None:
    """Best-effort conversion of a resolved CMake string to a repo-relative
    POSIX path. Returns None for anything that is clearly not a plain
    in-repo path (a leftover generator expression, an absolute path
    outside the repo, an empty string).
    """
    if not value:
        return None
    if "$<" in value or "${" in value:
        return None
    v = value.strip()
    if not v:
        return None
    root_str = str(repo_root)
    if v == root_str:
        return None
    if v.startswith(root_str + "/"):
        rel = v[len(root_str) + 1 :]
    elif v.startswith(root_str + os.sep) and os.sep != "/":
        rel = v[len(root_str) + 1 :]
    elif v.startswith("/"):
        return None
    else:
        rel = v
    rel = rel.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.rstrip("/")
    if not rel or rel.startswith(".."):
        return None
    return rel


def _evaluate_glob(pattern_value: str, recursive: bool, repo_root: Path) -> list[str]:
    rel = _to_repo_relative(pattern_value, repo_root)
    if rel is None:
        return []
    parts = rel.split("/")
    base_parts: list[str] = []
    idx = 0
    while idx < len(parts) and not any(ch in parts[idx] for ch in "*?["):
        base_parts.append(parts[idx])
        idx += 1
    base_dir = repo_root.joinpath(*base_parts) if base_parts else repo_root
    if idx >= len(parts):
        # No wildcard anywhere -- a literal single-file "glob".
        p = repo_root.joinpath(*parts)
        return [rel] if p.is_file() else []
    suffix = "/".join(parts[idx:])
    if not base_dir.is_dir():
        return []
    try:
        found = base_dir.rglob(suffix) if recursive else base_dir.glob(suffix)
    except (OSError, ValueError):
        return []
    out = []
    root_resolved = repo_root.resolve()
    for f in found:
        if not f.is_file():
            continue
        try:
            r = f.resolve().relative_to(root_resolved)
        except ValueError:
            continue
        out.append(r.as_posix())
    return sorted(out)


# ───────────────────────── build-input derivation ─────────────────────────

class BuildDerivation:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.variables: dict[str, list[str]] = {}
        for v in CMAKE_PATH_VARS:
            self.variables[v] = [str(repo_root)]
        self.build_inputs: set[str] = set()
        self.include_dirs: set[str] = set()
        self.ci_test_inputs: set[str] = set()
        self.test_runner_scripts: set[str] = set()
        self.reasons: dict[str, list[str]] = {}
        self.cmake_files_parsed: list[str] = []
        self._included: set[str] = set()

    def _note_build(self, relp: str, reason: str) -> None:
        self.build_inputs.add(relp)
        self.reasons.setdefault(relp, [])
        if reason not in self.reasons[relp]:
            self.reasons[relp].append(reason)

    def _note_ci_test(self, relp: str, reason: str) -> None:
        self.ci_test_inputs.add(relp)
        self.reasons.setdefault(relp, [])
        if reason not in self.reasons[relp]:
            self.reasons[relp].append(reason)

    def _resolve_and_check_files(self, token: str) -> list[tuple[str, str]]:
        """Resolve one token to every (relpath, absvalue) it names that is a
        real repo file. A token can be an exact `${LIST_VAR}` reference
        expanding to hundreds of entries (e.g. `${LIB_SRC}` in
        `add_library(eshkol-static STATIC ${LIB_SRC})`) -- every one of
        them is a real source, not just the first, so this returns the
        full list rather than short-circuiting on the first hit.
        """
        out: list[tuple[str, str]] = []
        for v in _resolve_token(token, self.variables):
            relp = _to_repo_relative(v, self.repo_root)
            if relp and (self.repo_root / relp).is_file():
                out.append((relp, v))
        return out

    def parse_file(self, path: Path) -> None:
        rel = path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        if rel in self._included:
            return
        self._included.add(rel)
        self.cmake_files_parsed.append(rel)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CMakeParseError(f"could not read {path}: {exc}") from exc
        text = _strip_cmake_comments(text)
        pending_includes: list[str] = []
        for cmd, raw_args in _iter_cmake_commands(text):
            if cmd == "set":
                self._handle_set(raw_args)
            elif cmd == "foreach":
                self._handle_foreach(raw_args)
            elif cmd == "list":
                self._handle_list(raw_args)
            elif cmd == "file":
                self._handle_file(raw_args)
            elif cmd in ("add_executable", "add_library"):
                self._handle_target_decl(cmd, raw_args)
            elif cmd == "target_sources":
                self._handle_target_sources(raw_args)
            elif cmd == "target_include_directories":
                self._handle_target_include_directories(raw_args)
            elif cmd == "configure_file":
                self._handle_configure_file(raw_args)
            elif cmd == "add_custom_command":
                self._handle_add_custom_command(raw_args)
            elif cmd in ("add_test", "add_custom_target"):
                self._handle_ci_reference(cmd, raw_args)
            elif cmd == "include":
                inc = self._resolve_include_target(raw_args)
                if inc is not None:
                    pending_includes.append(inc)
        for inc_rel in pending_includes:
            inc_path = self.repo_root / inc_rel
            if inc_path.is_file():
                self.parse_file(inc_path)

    # ---- command handlers ----

    def _handle_set(self, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        if not tokens:
            return
        name = tokens[0]
        rest = tokens[1:]
        for stop in ("CACHE", "PARENT_SCOPE"):
            if stop in rest:
                rest = rest[: rest.index(stop)]
        resolved: list[str] = []
        for t in rest:
            resolved.extend(_resolve_token(t, self.variables))
        self.variables[name] = resolved

    def _handle_foreach(self, raw_args: str) -> None:
        """`foreach(var ...)` / `foreach(var IN LISTS ...)` / `foreach(var
        IN ITEMS ...)`. This project's real loops that matter for source
        derivation are exactly `foreach (exe ${EXE_SRC}) ... add_executable
        (${exename} ${exe}) ... endforeach()` (each `exe/*.cpp` becomes its
        own executable) and a couple of small literal-item loops that
        `list(APPEND LIB_SRC "${loopvar}")` inside their body. Rather than
        actually iterating (this scanner has no body/endforeach scoping),
        the loop variable is bound to the UNION of every value the loop
        would ever take across all iterations. That is imprecise
        per-iteration (a single body execution sees every item at once
        instead of one item per pass) but exactly right for this script's
        only question -- set membership, not per-target precision -- and
        it is what lets `${exe}` inside the body still resolve to real
        paths instead of silently vanishing.
        """
        tokens = _tokenize_args(raw_args)
        if not tokens:
            return
        var = tokens[0]
        rest = tokens[1:]
        if len(rest) >= 2 and rest[0].upper() == "IN" and rest[1].upper() in ("LISTS", "ITEMS"):
            mode = rest[1].upper()
            resolved: list[str] = []
            for t in rest[2:]:
                if mode == "LISTS":
                    resolved.extend(self.variables.get(t, []))
                else:
                    resolved.extend(_resolve_token(t, self.variables))
            self.variables[var] = resolved
            return
        if rest and rest[0].upper() == "RANGE":
            return  # a numeric range never carries paths
        resolved = []
        for t in rest:
            resolved.extend(_resolve_token(t, self.variables))
        self.variables[var] = resolved

    def _handle_list(self, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        if len(tokens) < 2:
            return
        subcmd = tokens[0].upper()
        varname = tokens[1]
        cur = list(self.variables.get(varname, []))
        if subcmd == "APPEND":
            for t in tokens[2:]:
                cur.extend(_resolve_token(t, self.variables))
            self.variables[varname] = cur
        elif subcmd == "REMOVE_ITEM":
            remove_items: list[str] = []
            for t in tokens[2:]:
                remove_items.extend(_resolve_token(t, self.variables))
            remove_set = set(remove_items)
            self.variables[varname] = [x for x in cur if x not in remove_set]
        elif subcmd == "FILTER" and len(tokens) >= 5 and tokens[3].upper() == "REGEX":
            mode = tokens[2].upper()
            pattern = tokens[4]
            try:
                rx = re.compile(pattern)
            except re.error:
                return
            if mode == "EXCLUDE":
                self.variables[varname] = [x for x in cur if not rx.search(x)]
            elif mode == "INCLUDE":
                self.variables[varname] = [x for x in cur if rx.search(x)]
        # LENGTH / FIND / GET / SORT / REMOVE_DUPLICATES: no effect on
        # membership, intentionally ignored.

    def _handle_file(self, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        if not tokens:
            return
        kind = tokens[0].upper()
        if kind not in ("GLOB", "GLOB_RECURSE"):
            return
        if len(tokens) < 2:
            return
        varname = tokens[1]
        rest = tokens[2:]
        patterns: list[str] = []
        i = 0
        while i < len(rest):
            tok = rest[i]
            up = tok.upper()
            if up == "RELATIVE" and i + 1 < len(rest):
                i += 2
                continue
            if up in ("CONFIGURE_DEPENDS", "FOLLOW_SYMLINKS"):
                i += 1
                continue
            if up == "LIST_DIRECTORIES" and i + 1 < len(rest):
                i += 2
                continue
            patterns.append(tok)
            i += 1
        matched: list[str] = []
        for p in patterns:
            for v in _resolve_token(p, self.variables):
                matched.extend(_evaluate_glob(v, kind == "GLOB_RECURSE", self.repo_root))
        # Deliberately NOT marked as a build input here: a glob only
        # populates the variable's contents. Whether a matched file is
        # actually compiled depends on whatever later list(FILTER
        # EXCLUDE ...) / list(REMOVE_ITEM ...) mutations run against this
        # variable before it is finally consumed by add_executable /
        # add_library / target_sources / add_custom_command DEPENDS --
        # exactly the eshkol_compiler.c case this script exists to get
        # right. Attribution happens at the consuming command instead, in
        # _sources_from_tokens / _handle_add_custom_command, which reads
        # the variable's value as of that later point in the file.
        self.variables[varname] = matched

    def _sources_from_tokens(self, tokens: list[str], skip: set[str]) -> list[tuple[str, str]]:
        out = []
        for t in tokens:
            if t.upper() in skip:
                continue
            out.extend(self._resolve_and_check_files(t))
        return out

    def _handle_target_decl(self, cmd: str, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        if not tokens:
            return
        target = tokens[0]
        rest = tokens[1:]
        for relp, _ in self._sources_from_tokens(rest, _TARGET_DECL_KEYWORDS):
            self._note_build(relp, f"source of target '{target}' ({cmd})")

    def _handle_target_sources(self, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        if not tokens:
            return
        target = tokens[0]
        rest = tokens[1:]
        for relp, _ in self._sources_from_tokens(rest, _VISIBILITY_KEYWORDS):
            self._note_build(relp, f"source of target '{target}' (target_sources)")

    def _handle_target_include_directories(self, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        if len(tokens) < 2:
            return
        for t in tokens[1:]:
            if t.upper() in _INCLUDE_DIR_KEYWORDS:
                continue
            for v in _resolve_token(t, self.variables):
                relp = _to_repo_relative(v, self.repo_root)
                if relp and (self.repo_root / relp).is_dir():
                    self.include_dirs.add(relp)

    def _handle_configure_file(self, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        if not tokens:
            return
        for relp, _ in self._resolve_and_check_files(tokens[0]):
            self._note_build(relp, "configure_file input")

    def _handle_add_custom_command(self, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        segments: dict[str, list[str]] = {}
        cur_key: str | None = None
        for t in tokens:
            if t.upper() in _ADD_CUSTOM_COMMAND_KEYWORDS:
                cur_key = t.upper()
                segments.setdefault(cur_key, [])
                continue
            if cur_key:
                segments[cur_key].append(t)
        for key in ("DEPENDS", "COMMAND", "MAIN_DEPENDENCY"):
            for t in segments.get(key, []):
                for relp, _ in self._resolve_and_check_files(t):
                    self._note_build(relp, f"add_custom_command {key} argument")

    def _handle_ci_reference(self, cmd: str, raw_args: str) -> None:
        tokens = _tokenize_args(raw_args)
        for t in tokens:
            for relp, _ in self._resolve_and_check_files(t):
                self._note_ci_test(relp, f"referenced by {cmd}(...) in CMakeLists.txt")
                if relp.startswith("scripts/"):
                    self.test_runner_scripts.add(relp)

    def _resolve_include_target(self, raw_args: str) -> str | None:
        tokens = _tokenize_args(raw_args)
        if not tokens:
            return None
        for v in _resolve_token(tokens[0], self.variables):
            relp = _to_repo_relative(v, self.repo_root)
            if relp and (self.repo_root / relp).is_file():
                return relp
        # Either an unresolved reference, or a plain module name (e.g.
        # `include(Packing)`, `include(CTest)`) that _resolve_token echoes
        # back literally since it contains no `${...}`. CMake resolves a
        # bare module name against CMAKE_MODULE_PATH, which this project
        # points at cmake/ (`set(CMAKE_MODULE_PATH ".../cmake")`) -- try the
        # obvious cmake/<Name>.cmake guess; a CMake-builtin module (CTest,
        # GNUInstallDirs, FindPkgConfig, FetchContent) has no such file in
        # this repo and correctly resolves to nothing.
        bare = tokens[0]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bare):
            guess = f"cmake/{bare}.cmake"
            if (self.repo_root / guess).is_file():
                return guess
        return None


def derive_build_and_ci_inputs(repo_root: Path):
    """Parse CMakeLists.txt (and its includes) plus every workflow file,
    returning (BuildDerivation, workflow_paths, workflow_test_refs)."""
    entry = repo_root / "CMakeLists.txt"
    deriv = BuildDerivation(repo_root)
    if not entry.is_file():
        raise CMakeParseError(f"{entry} not found")
    deriv.parse_file(entry)

    workflows_dir = repo_root / ".github" / "workflows"
    workflow_paths: list[str] = []
    if workflows_dir.is_dir():
        for p in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
            workflow_paths.append(p.relative_to(repo_root).as_posix())

    workflow_test_refs: set[str] = set()
    workflow_all_refs: set[str] = set()
    for wf_rel in workflow_paths:
        wf_path = repo_root / wf_rel
        try:
            text = wf_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in SCRIPT_REF_RE.finditer(text):
            ref = m.group(1).replace("\\", "/")
            if (repo_root / ref).is_file():
                workflow_all_refs.add(ref)
                if ref.startswith(("scripts/", "tests/")):
                    workflow_test_refs.add(ref)

    for ref in workflow_all_refs:
        deriv._note_ci_test(ref, "referenced in a .github/workflows/*.yml step")
        if ref.startswith("scripts/"):
            deriv.test_runner_scripts.add(ref)

    return deriv, workflow_paths, workflow_test_refs


# ───────────────────────── compile_commands.json self-consistency ─────────────────────────

def check_compile_commands_consistency(deriv: BuildDerivation, compile_commands_path: Path) -> list[str]:
    """Return a list of translation-unit paths present in a real configured
    build's compile_commands.json but absent from the derived build-input
    set. An empty list means the derivation is self-consistent with a real
    build."""
    try:
        data = json.loads(compile_commands_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CMakeParseError(f"could not read {compile_commands_path}: {exc}") from exc
    root = deriv.repo_root.resolve()
    missing: list[str] = []
    for entry in data:
        file_field = entry.get("file")
        if not file_field:
            continue
        directory = entry.get("directory", str(root))
        fp = Path(file_field)
        if not fp.is_absolute():
            fp = Path(directory) / fp
        try:
            relp = fp.resolve().relative_to(root).as_posix()
        except ValueError:
            continue  # a TU outside the repo (a fetched dependency) is not ours to classify
        if relp in deriv.build_inputs:
            continue
        if any(relp == d or relp.startswith(d + "/") for d in deriv.include_dirs):
            continue
        missing.append(relp)
    return sorted(set(missing))


# ───────────────────────── classification ─────────────────────────

def _is_docs_only_path(path: str) -> bool:
    import fnmatch

    return any(fnmatch.fnmatchcase(path, pat) for pat in DOCS_ONLY_PATTERNS)


def _is_workflow_path(path: str) -> bool:
    return path.startswith(".github/workflows/")


def classify(paths: list[str], repo_root: Path, compile_commands: Path | None = None):
    deriv, workflow_paths, _wf_test_refs = derive_build_and_ci_inputs(repo_root)

    # CMakeLists.txt and every cmake/*.cmake file define the build itself;
    # touching them is always build-relevant, independent of whatever the
    # parse above did or did not manage to attribute to a specific rule.
    always_build = {"CMakeLists.txt"}
    cmake_dir = repo_root / "cmake"
    if cmake_dir.is_dir():
        for p in cmake_dir.rglob("*"):
            if p.is_file():
                always_build.add(p.relative_to(repo_root).as_posix())

    consistency_result = None
    if compile_commands is not None:
        if compile_commands.is_file():
            missing = check_compile_commands_consistency(deriv, compile_commands)
            consistency_result = {"checked": True, "path": str(compile_commands), "missing_tus": missing}
        else:
            consistency_result = {"checked": False, "reason": f"{compile_commands} not found"}

    file_reports = []
    any_workflow = False
    all_docs = True
    all_non_build = True
    all_tests_only = True

    for path in paths:
        p = path.strip()
        if not p:
            continue
        reasons: list[str] = []

        is_workflow = _is_workflow_path(p)
        if is_workflow:
            any_workflow = True
            reasons.append("is a .github/workflows/*.yml file (workflow changes are always full)")

        is_build = False
        if p in always_build:
            is_build = True
            reasons.append("CMakeLists.txt or a cmake/*.cmake build-definition file")
        if p in deriv.build_inputs:
            is_build = True
            reasons.extend(deriv.reasons.get(p, []))
        if not is_build and Path(p).suffix.lower() in HEADER_EXTENSIONS:
            for d in deriv.include_dirs:
                if p == d or p.startswith(d + "/"):
                    is_build = True
                    reasons.append(f"header under include directory '{d}' (target_include_directories)")
                    break

        is_ci_test = p in deriv.ci_test_inputs or p.startswith("tests/") or p.startswith(".icc/")
        if p.startswith("tests/") and "under tests/" not in reasons:
            reasons.append("under tests/")
        if p.startswith(".icc/") and "under .icc/ (read by assurance gates)" not in reasons:
            reasons.append("under .icc/ (read by assurance gates)")
        if p in deriv.ci_test_inputs:
            reasons.extend(r for r in deriv.reasons.get(p, []) if r not in reasons)

        is_docs = _is_docs_only_path(p)
        if is_docs:
            reasons.append("matches the docs-only predicate")

        is_test_runner_narrow = (
            p.startswith("tests/")
            or (p.startswith("scripts/") and p in deriv.test_runner_scripts)
        )

        if not is_docs:
            all_docs = False
        if is_build or is_ci_test or is_workflow:
            all_non_build = False
        if is_build or is_workflow or not is_test_runner_narrow:
            all_tests_only = False

        if not reasons:
            reasons.append("not matched by any build glob, target, workflow, or test reference")

        file_reports.append(
            {
                "path": p,
                "docs_only_predicate": is_docs,
                "build_input": is_build,
                "ci_test_input": is_ci_test,
                "workflow_file": is_workflow,
                "test_runner_eligible": is_test_runner_narrow,
                "reasons": reasons,
            }
        )

    if not file_reports:
        result_class = CLASS_NON_BUILD
        summary = "non-build: no changed files"
    elif any_workflow:
        result_class = CLASS_FULL
        summary = f"full: {len(file_reports)} file(s), includes a workflow file change"
    elif all_docs:
        result_class = CLASS_DOCS
        summary = f"docs: {len(file_reports)} file(s) match the docs-only predicate"
    elif all_non_build:
        result_class = CLASS_NON_BUILD
        summary = f"non-build: {len(file_reports)} file(s), none consumed by the build or CI/tests"
    elif all_tests_only:
        result_class = CLASS_TESTS_ONLY
        summary = f"tests-only: {len(file_reports)} file(s) under tests/ or a CI test-runner script"
    else:
        result_class = CLASS_FULL
        summary = f"full: {len(file_reports)} file(s), at least one is a build/workflow input (or a mixed set)"

    return {
        "class": result_class,
        "summary": summary,
        "files": file_reports,
        "derivation": {
            "cmake_files_parsed": deriv.cmake_files_parsed,
            "build_input_count": len(deriv.build_inputs) + len(always_build),
            "include_dir_count": len(deriv.include_dirs),
            "ci_test_input_count": len(deriv.ci_test_inputs),
            "test_runner_script_count": len(deriv.test_runner_scripts),
            "workflow_files": workflow_paths,
            "compile_commands_check": consistency_result,
        },
    }


# ───────────────────────── input gathering ─────────────────────────

def _paths_from_pr(pr_number: int, repo_root: Path) -> list[str]:
    gh = shutil.which("gh")
    if not gh:
        raise CMakeParseError("`gh` CLI not found on PATH; cannot use --pr")
    result = subprocess.run(
        [gh, "pr", "view", str(pr_number), "--json", "files", "--jq", ".files[].path"],
        capture_output=True, text=True, cwd=str(repo_root), timeout=30,
    )
    if result.returncode != 0:
        raise CMakeParseError(f"`gh pr view {pr_number}` failed: {result.stderr.strip()[:500]}")
    return [line for line in result.stdout.splitlines() if line.strip()]


# ───────────────────────── self-test ─────────────────────────

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def self_test() -> bool:
    print("ci_change_class.py self-test:")
    ok = True

    scratch_root = REPO_ROOT / ".scratch" / "ci_change_class_selftest"
    if scratch_root.exists():
        import shutil as _shutil

        _shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)

    try:
        fixture = scratch_root / "fixture_repo"
        fixture.mkdir(parents=True, exist_ok=True)

        _write(
            fixture / "CMakeLists.txt",
            """
            cmake_minimum_required(VERSION 3.20)
            project(fixture C CXX)

            include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Extra.cmake)

            file(GLOB_RECURSE LIB_SRC RELATIVE ${CMAKE_SOURCE_DIR} "lib/*.c*")
            list(FILTER LIB_SRC EXCLUDE REGEX "lib/excluded/.*")
            list(FILTER LIB_SRC EXCLUDE REGEX "lib/[^/]*\\\\.c$")

            set(EXTRA_SRC
              lib/extra/explicit_source.cpp
            )
            list(APPEND LIB_SRC ${EXTRA_SRC})

            add_library(fixture-lib STATIC ${LIB_SRC})
            target_include_directories(fixture-lib PRIVATE inc/)

            add_executable(standalone_tool lib/standalone_tool.c)

            configure_file(cmake/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

            add_custom_command(
              OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated.h
              COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate.cmake
              DEPENDS cmake/generate.cmake lib/gen_input.esk
            )
            add_custom_target(generated_header ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated.h)

            # Mirrors the real stdlib.esk pattern: a CONFIGURE_DEPENDS glob
            # whose result is consumed ONLY through add_custom_command
            # DEPENDS, never through add_executable/add_library/
            # target_sources directly.
            file(GLOB_RECURSE STDLIB_LIKE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/stdlib_modules/*.esk")
            add_custom_command(
              OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stdlib_like.o
              COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/stdlib_like.o
              DEPENDS ${STDLIB_LIKE_SOURCES}
            )

            # Mirrors the real EXE_SRC pattern: a glob consumed inside a
            # foreach loop, one add_executable per matched file.
            file(GLOB_RECURSE EXE_LIKE_SRC RELATIVE ${CMAKE_SOURCE_DIR} "exelike/*.cpp")
            foreach (one_exe ${EXE_LIKE_SRC})
              get_filename_component(one_exe_name ${one_exe} NAME_WE)
              add_executable(${one_exe_name} ${one_exe})
            endforeach ()

            enable_testing()
            add_test(NAME smoke COMMAND scripts/run_smoke_tests.sh)
            """,
        )
        _write(fixture / "cmake" / "Extra.cmake", "# nothing extra needed for the fixture\n")
        _write(fixture / "cmake" / "config.h.in", "#define FIXTURE_VERSION \"@VERSION@\"\n")
        _write(fixture / "cmake" / "generate.cmake", "# pretend generator\n")
        _write(fixture / "lib" / "core.cpp", "// compiled\n")
        _write(fixture / "lib" / "excluded" / "not_built.cpp", "// excluded by regex\n")
        _write(fixture / "lib" / "top_level_c_excluded.c", "// excluded: lib/[^/]*.c$\n")
        _write(fixture / "stdlib_modules" / "a.esk", "; transitively required\n")
        _write(fixture / "exelike" / "tool_main.cpp", "// int main() {}\n")
        _write(fixture / "lib" / "extra" / "explicit_source.cpp", "// compiled via explicit set()\n")
        _write(fixture / "lib" / "standalone_tool.c", "// compiled via add_executable\n")
        _write(fixture / "lib" / "gen_input.esk", "; a DEPENDS input, not glob-matched\n")
        _write(fixture / "inc" / "fixture" / "public.h", "// under an include dir, never globbed directly\n")
        _write(fixture / "scripts" / "run_smoke_tests.sh", "#!/bin/sh\necho smoke\n")
        _write(fixture / "docs" / "GUIDE.md", "# guide\n")
        _write(fixture / "README.md", "# fixture\n")
        _write(fixture / "notes" / "scratch.md", "note\n")
        _write(fixture / "tests" / "unit" / "case.esk", "; a test fixture\n")
        _write(fixture / "unrelated" / "artwork.svg", "<svg/>\n")
        _write(fixture / ".github" / "workflows" / "ci.yml", "name: ci\non: {push: {}}\njobs: {}\n")

        deriv, _wf, _wf_refs = derive_build_and_ci_inputs(fixture)

        checks: list[tuple[str, bool]] = []

        checks.append(("glob picks up lib/core.cpp", "lib/core.cpp" in deriv.build_inputs))
        checks.append(
            (
                "FILTER EXCLUDE removes lib/excluded/not_built.cpp (negative control: matched a glob, still excluded)",
                "lib/excluded/not_built.cpp" not in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "FILTER EXCLUDE removes lib/top_level_c_excluded.c (mirrors the real eshkol_compiler.c case)",
                "lib/top_level_c_excluded.c" not in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "explicit set()+APPEND source is a build input",
                "lib/extra/explicit_source.cpp" in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "add_executable source is a build input",
                "lib/standalone_tool.c" in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "configure_file input is a build input",
                "cmake/config.h.in" in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "add_custom_command DEPENDS picks up a non-glob-matched .esk input",
                "lib/gen_input.esk" in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "add_custom_command DEPENDS picks up an included cmake/ script",
                "cmake/generate.cmake" in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "add_test COMMAND is a CI/test input, not a build input",
                "scripts/run_smoke_tests.sh" in deriv.ci_test_inputs
                and "scripts/run_smoke_tests.sh" not in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "included cmake/Extra.cmake was actually parsed",
                "cmake/Extra.cmake" in deriv.cmake_files_parsed,
            )
        )
        checks.append(
            (
                "a CONFIGURE_DEPENDS glob consumed only via add_custom_command DEPENDS "
                "(the real stdlib.esk pattern) is still a build input",
                "stdlib_modules/a.esk" in deriv.build_inputs,
            )
        )
        checks.append(
            (
                "a glob consumed inside foreach(...)/add_executable (the real EXE_SRC "
                "pattern) is still a build input",
                "exelike/tool_main.cpp" in deriv.build_inputs,
            )
        )

        for desc, passed in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
            ok = ok and passed

        # ---- end-to-end classification fixtures ----
        def cls(paths: list[str]) -> dict:
            return classify(paths, fixture)

        e2e: list[tuple[str, list[str], str]] = [
            ("docs: README + docs/ + notes/", ["README.md", "docs/GUIDE.md", "notes/scratch.md"], CLASS_DOCS),
            (
                "non-build: excluded source (mirrors PR #624)",
                ["lib/top_level_c_excluded.c"],
                CLASS_NON_BUILD,
            ),
            ("non-build: unrelated non-source file", ["unrelated/artwork.svg"], CLASS_NON_BUILD),
            ("full: a header under inc/ (never individually globbed)", ["inc/fixture/public.h"], CLASS_FULL),
            ("tests-only: a test fixture file", ["tests/unit/case.esk"], CLASS_TESTS_ONLY),
            ("tests-only: a scripts/ test-runner script", ["scripts/run_smoke_tests.sh"], CLASS_TESTS_ONLY),
            ("full: a workflow file", [".github/workflows/ci.yml"], CLASS_FULL),
            (
                "non-build: mixed docs + non-build set (neither is a build/CI input)",
                ["README.md", "unrelated/artwork.svg"],
                CLASS_NON_BUILD,
            ),
            (
                "full: mixed docs + a real build input",
                ["README.md", "lib/core.cpp"],
                CLASS_FULL,
            ),
            (
                "full: a real compiled source",
                ["lib/core.cpp"],
                CLASS_FULL,
            ),
            (
                "full: CMakeLists.txt itself",
                ["CMakeLists.txt"],
                CLASS_FULL,
            ),
        ]
        for desc, paths, expected in e2e:
            got = cls(paths)["class"]
            passed = got == expected
            print(f"  [{'PASS' if passed else 'FAIL'}] {desc}: expected {expected}, got {got}")
            ok = ok and passed

        # ---- compile_commands.json self-consistency check ----
        build_dir = fixture / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        good_cc = [
            {"directory": str(build_dir), "file": str(fixture / "lib" / "core.cpp"), "command": "c++ -c lib/core.cpp"},
            {"directory": str(build_dir), "file": str(fixture / "lib" / "standalone_tool.c"), "command": "cc -c lib/standalone_tool.c"},
        ]
        (build_dir / "compile_commands.json").write_text(json.dumps(good_cc), encoding="utf-8")
        deriv2, _, _ = derive_build_and_ci_inputs(fixture)
        missing_good = check_compile_commands_consistency(deriv2, build_dir / "compile_commands.json")
        passed = missing_good == []
        print(f"  [{'PASS' if passed else 'FAIL'}] compile_commands.json consistency: a real build's TUs are all derived build inputs")
        ok = ok and passed

        bad_cc = good_cc + [
            {"directory": str(build_dir), "file": str(fixture / "lib" / "excluded" / "not_built.cpp"), "command": "c++ -c lib/excluded/not_built.cpp"}
        ]
        (build_dir / "compile_commands.json").write_text(json.dumps(bad_cc), encoding="utf-8")
        missing_bad = check_compile_commands_consistency(deriv2, build_dir / "compile_commands.json")
        passed = missing_bad == ["lib/excluded/not_built.cpp"]
        print(
            f"  [{'PASS' if passed else 'FAIL'}] compile_commands.json consistency: a TU CMake would never actually "
            f"compile (excluded by regex) is caught as a derivation gap"
        )
        ok = ok and passed

    finally:
        import shutil as _shutil

        _shutil.rmtree(scratch_root, ignore_errors=True)

    if ok:
        print("self-test: PASS")
    else:
        print("self-test: FAIL", file=sys.stderr)
    return ok


# ───────────────────────── CLI ─────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="changed file paths (repo-relative)")
    parser.add_argument("--pr", type=int, help="fetch changed files from `gh pr view <N>`")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="repository root (default: this script's repo)")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="a configured CMake build directory; if it contains compile_commands.json, "
        "every compiled translation unit is checked against the derived build-input set",
    )
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--self-test", action="store_true", help="run built-in derivation and classification fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    repo_root = args.repo_root.resolve()

    if args.pr is not None:
        try:
            paths = _paths_from_pr(args.pr, repo_root)
        except CMakeParseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    elif args.paths:
        paths = args.paths
    elif not sys.stdin.isatty():
        paths = [line for line in sys.stdin.read().splitlines() if line.strip()]
    else:
        print("error: no changed paths given (positional args, stdin, or --pr N)", file=sys.stderr)
        return 1

    compile_commands = None
    if args.build_dir is not None:
        compile_commands = args.build_dir / "compile_commands.json"

    try:
        result = classify(paths, repo_root, compile_commands)
    except CMakeParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result["summary"], file=sys.stderr)
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["summary"])
        for f in result["files"]:
            print(f"  {f['path']}: {'; '.join(f['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
