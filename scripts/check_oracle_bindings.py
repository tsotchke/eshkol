#!/usr/bin/env python3
"""Check completion-oracle actions against the repository's test surface.

The oracle file is configuration, not evidence.  In particular, CTest exits
successfully when ``ctest -R`` or ``ctest -L`` selects no tests, so a renamed
or conditionally removed test can make a criterion vacuous.  This gate reads
the registered CTest names and labels from the repository CMake files and
checks every selector in every oracle action.

It also checks script paths mentioned by actions.  Directly invoked scripts
must be executable; scripts passed to an interpreter only need to exist.

Usage:
    python3 scripts/check_oracle_bindings.py
    python3 scripts/check_oracle_bindings.py --oracles path/to/oracles.yaml
    python3 scripts/check_oracle_bindings.py --self-test
    python3 scripts/check_oracle_bindings.py --no-trace

Exit status is 0 on PASS and 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import sys
import tempfile
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORACLES = REPO_ROOT / ".icc" / "completion-oracles.yaml"
DEFAULT_TRACE_DIR = REPO_ROOT / "scripts" / "icc_traces"
TRACE_BASENAME = "oracle_bindings_gate.jsonl"
PROBE_ID = "oracle_bindings_clean"

SCRIPT_REF_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:\./)?(?:scripts|tests)/"
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:py|sh|ps1|psm1)"
)
CTEST_OPTIONS = {"-R": "name", "--tests-regex": "name",
                 "-L": "label", "--label-regex": "label"}
SHELL_SEPARATORS = {"&&", "||", ";", "|"}
CALL_RE = re.compile(r"\b(add_test|eshkol_add_test|gtest_discover_tests)\s*\(", re.I)


class BindingError(Exception):
    """The oracle file or CMake surface could not be inspected."""


def _load_yaml(path: Path) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BindingError("PyYAML is required (pip install pyyaml)") from exc
    if not path.is_file():
        raise BindingError(f"oracle file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise BindingError(f"oracle file is not parseable: {exc}") from exc


def _cmake_files(repo_root: Path) -> list[Path]:
    files = [repo_root / "CMakeLists.txt"]
    files.extend(sorted((repo_root / "tests").rglob("CMakeLists.txt")))
    return [path for path in files if path.is_file()]


def _strip_cmake_comments(text: str) -> str:
    # CMake comments end at the newline.  None of the registrations this gate
    # consumes embeds a # in a quoted argument, and preserving newlines keeps
    # diagnostics useful without attempting to implement the CMake language.
    return re.sub(r"^[ \t]*#.*$", "", text, flags=re.M)


def _call_blocks(text: str) -> Iterable[tuple[str, str]]:
    """Yield (command, parenthesized body) for the three registration APIs."""
    text = _strip_cmake_comments(text)
    for match in CALL_RE.finditer(text):
        depth = 1
        pos = match.end()
        quote: str | None = None
        while pos < len(text) and depth:
            char = text[pos]
            if quote:
                if char == quote and text[pos - 1] != "\\":
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            pos += 1
        if depth == 0:
            yield match.group(1).lower(), text[match.end() : pos - 1]


def registered_tests(repo_root: Path) -> tuple[set[str], dict[str, set[str]]]:
    names: set[str] = set()
    labels: dict[str, set[str]] = {}
    for path in _cmake_files(repo_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for command, body in _call_blocks(text):
            if command == "gtest_discover_tests":
                token = shlex.split(body, comments=False)[0:1]
                if token:
                    names.add(token[0])
                continue
            name_match = re.search(r"\bNAME\s+([^\s)]+)", body, re.I)
            if name_match:
                names.add(name_match.group(1))

        for match in re.finditer(
            r"\bset_tests_properties\s*\((.*?)\)", text, re.S | re.I
        ):
            body = _strip_cmake_comments(match.group(1))
            prop = re.search(r"\bLABELS\s+(?:\"([^\"]*)\"|'([^']*)'|(\S+))", body, re.I)
            if not prop:
                continue
            label = next(value for value in prop.groups() if value is not None)
            before_properties = re.split(r"\bPROPERTIES\b", body, maxsplit=1, flags=re.I)[0]
            for test_name in shlex.split(before_properties, comments=False):
                labels.setdefault(test_name, set()).add(label)

        for match in re.finditer(
            r"\bset_property\s*\(\s*TEST\s+(.*?)\)", text, re.S | re.I
        ):
            body = _strip_cmake_comments(match.group(1))
            prop = re.search(r"\bPROPERTY\s+LABELS\s+(?:\"([^\"]*)\"|'([^']*)'|(\S+))", body, re.I)
            if not prop:
                continue
            label = next(value for value in prop.groups() if value is not None)
            before_property = re.split(r"\bPROPERTY\b", body, maxsplit=1, flags=re.I)[0]
            for test_name in shlex.split(before_property, comments=False):
                labels.setdefault(test_name, set()).add(label)

    return names, labels


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def action_selectors(action: str) -> list[tuple[str, str]]:
    """Extract every selector, including multiple options on one ctest call."""
    selectors: list[tuple[str, str]] = []
    try:
        tokens = shlex.split(action, comments=False)
    except ValueError:
        return selectors
    for index, token in enumerate(tokens):
        if token != "ctest":
            continue
        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor] not in SHELL_SEPARATORS:
            option = tokens[cursor]
            if option in CTEST_OPTIONS and cursor + 1 < len(tokens):
                selectors.append((CTEST_OPTIONS[option], tokens[cursor + 1]))
                cursor += 2
                continue
            matched = re.match(r"^(--tests-regex|--label-regex)=([^=].*)$", option)
            if matched:
                selectors.append((CTEST_OPTIONS[matched.group(1)], matched.group(2)))
            # --test-dir and other ctest options do not alter binding lookup;
            # skipping their values is not necessary because only recognized
            # selector options contribute.
            cursor += 1
    return selectors


def _direct_script_refs(action: str, refs: list[str]) -> set[str]:
    """Return referenced paths that are command operands, not interpreter args."""
    try:
        tokens = shlex.split(action, comments=False)
    except ValueError:
        return set()
    direct: set[str] = set()
    interpreter = {"bash", "sh", "zsh", "python", "python3", "perl", "pwsh", "powershell"}
    for index, token in enumerate(tokens):
        normalized = token[2:] if token.startswith("./") else token
        if normalized not in refs:
            continue
        previous = tokens[index - 1] if index else ""
        if previous not in interpreter and not previous.endswith(("/bash", "/sh", "/python3", "/python")):
            direct.add(normalized)
    return direct


def _iter_criteria(data: object) -> Iterable[tuple[str, int, dict]]:
    if not isinstance(data, dict) or not isinstance(data.get("oracles"), list):
        return
    for oracle in data["oracles"]:
        if not isinstance(oracle, dict):
            continue
        oracle_name = str(oracle.get("name") or oracle.get("target") or "<unnamed>")
        criteria = oracle.get("requires", oracle.get("criteria", []))
        if not isinstance(criteria, list):
            continue
        for position, criterion in enumerate(criteria, 1):
            if isinstance(criterion, dict):
                yield oracle_name, position, criterion


def check(data: object, repo_root: Path = REPO_ROOT) -> dict:
    errors: list[str] = []
    inventory: list[dict] = []
    if not isinstance(data, dict):
        return {"passed": False, "errors": ["oracle document is not a mapping"], "inventory": []}
    if not isinstance(data.get("oracles"), list):
        return {"passed": False, "errors": ["oracle document has no top-level `oracles` list"], "inventory": []}
    try:
        names, labels = registered_tests(repo_root)
    except (OSError, UnicodeError, ValueError) as exc:
        return {"passed": False, "errors": [f"could not inspect CMake registrations: {exc}"], "inventory": []}

    for oracle_name, position, criterion in _iter_criteria(data):
        action = criterion.get("action", criterion.get("recommended_action"))
        if not isinstance(action, str):
            continue
        location = f"oracle {oracle_name!r} criterion #{position}"
        for kind, pattern in action_selectors(action):
            try:
                if kind == "name":
                    matched = sorted(name for name in names if re.search(pattern, name))
                else:
                    matched = sorted(
                        name for name, test_labels in labels.items()
                        if any(re.search(pattern, label) for label in test_labels)
                    )
            except re.error as exc:
                errors.append(f"{location}: invalid ctest {kind} regex {pattern!r}: {exc}")
                matched = []
            inventory.append({"oracle": oracle_name, "criterion": position,
                              "selector": f"ctest -{'R' if kind == 'name' else 'L'} {pattern}",
                              "matched": matched})
            if not matched:
                errors.append(
                    f"{location}: ctest {('-R' if kind == 'name' else '-L')} {pattern!r} "
                    "matches no registered test"
                )

        refs = sorted(set(SCRIPT_REF_RE.findall(action)))
        direct_refs = _direct_script_refs(action, refs)
        for relative in refs:
            path = repo_root / relative
            inventory.append({"oracle": oracle_name, "criterion": position,
                              "script": relative, "exists": path.is_file(),
                              "executable": os.access(path, os.X_OK)})
            if not path.is_file():
                errors.append(f"{location}: referenced script {relative!r} does not exist")
            elif relative in direct_refs and not os.access(path, os.X_OK):
                errors.append(f"{location}: directly invoked script {relative!r} is not executable")

    return {"passed": not errors, "errors": errors, "inventory": inventory,
            "registered_tests": sorted(names),
            "registered_labels": {key: sorted(value) for key, value in sorted(labels.items())}}


def emit_trace(trace_dir: Path, status: str, snippet: str) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / TRACE_BASENAME
    event = {"kind": "eshkol_smoke", "name": PROBE_ID, "value": status,
             "snippet": snippet[:2000], "confidence": 1.0}
    path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _fixture_oracles(action: str) -> str:
    return f"""
oracles:
  - name: selftest-oracle
    requires:
      - test_evidence: true
        severity: high
        label: fixture
        action: {action}
"""


def self_test() -> bool:
    """Exercise passing, zero-match, zero-label-match, and missing-script cases."""
    import yaml  # type: ignore

    scratch = REPO_ROOT / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    all_ok = True
    with tempfile.TemporaryDirectory(dir=scratch, prefix="oracle-bindings-") as temp:
        root = Path(temp)
        (root / ".icc").mkdir()
        (root / "scripts").mkdir()
        (root / "scripts" / "ok.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        os.chmod(root / "scripts" / "ok.sh", stat.S_IRWXU)
        (root / "CMakeLists.txt").write_text(
            "add_test(NAME real_test COMMAND true)\n"
            "set_tests_properties(real_test PROPERTIES LABELS smoke)\n",
            encoding="utf-8",
        )
        cases = [
            ("good", _fixture_oracles("ctest -R real_test -L smoke && ./scripts/ok.sh"), True),
            ("missing-name", _fixture_oracles("ctest -R absent"), False),
            ("missing-label", _fixture_oracles("ctest -L absent"), False),
            ("missing-script", _fixture_oracles("./scripts/missing.sh"), False),
        ]
        print("check_oracle_bindings.py self-test:")
        for name, text, expected in cases:
            result = check(yaml.safe_load(text), root)
            got = bool(result["passed"])
            ok = got == expected
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            print(f"  [{verdict}] {name}: expected passed={expected}, got passed={got}")
            for error in result["errors"]:
                print(f"           {error}")
    if all_ok:
        print("self-test: PASS — zero-match selectors and missing scripts fail")
    else:
        print("self-test: FAIL — binding failures were not detected", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oracles", default=os.environ.get("ESHKOL_ORACLE_FILE", str(DEFAULT_ORACLES)))
    parser.add_argument("--repo", default=str(REPO_ROOT), help="repository whose CMake files are inspected")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1

    try:
        data = _load_yaml(Path(args.oracles))
        result = check(data, Path(args.repo).resolve())
    except BindingError as exc:
        result = {"passed": False, "errors": [str(exc)], "inventory": []}

    status = "PASS" if result["passed"] else "FAIL"
    summary = (f"{len(result.get('registered_tests', []))} registered tests; "
               f"{len(result.get('inventory', []))} bindings inspected")
    if result["errors"]:
        summary = f"{len(result['errors'])} error(s): " + "; ".join(result["errors"][:4])
    if not args.no_trace:
        emit_trace(Path(args.trace_dir), status, summary)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  {summary}")
        for item in result.get("inventory", []):
            if "selector" in item:
                print(f"  {item['selector']}: {', '.join(item['matched']) or '(none)'}")
            else:
                state = "exists" if item["exists"] else "MISSING"
                print(f"  script {item['script']}: {state}")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
