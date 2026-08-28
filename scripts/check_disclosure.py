#!/usr/bin/env python3
"""Release gate: no PR text or diff discloses private infrastructure identifiers.

Motivating incident (2026-08-27/28, the second in one week): a squash-merged
commit landed on `master` carrying two internal hostnames in its commit body.
GitHub assembles a squash-merge commit's body out of the individual branch
commit messages, so a hostname mentioned in passing on a feature branch --
"verified on old-donkey", "the mesh's Windows host (desktop-jack-blupc)" --
survives into permanent, public history the moment the PR merges, with
nothing in code review that treats commit *prose* as a reviewable surface the
same way a diff hunk is. Nobody was reviewing for it because nothing failed
when it happened.

Scope narrowed by maintainer ruling (2026-08-28): internal machine/hostnames
by themselves are FINE in this public repo and are no longer flagged --
the two hostnames from the motivating incident above would not trip this
gate today. What still matters, and what the gate below actually catches, is
private IPv4 literals, ssh key-file references, ProxyCommand/`tailscale nc`
recipes, MAC addresses, and (Layer 2, optional) a maintainer's own denylist
of specific known-sensitive tokens.

This gate treats three text surfaces as disclosure-reviewable, matching
everywhere PR authorship text lands in this repo's public history:
    1. every commit message in `<base>..<head>` (`git log --format=%B`) --
       this is what a squash-merge body is assembled FROM;
    2. the PR title and body themselves (GitHub also lets these leak straight
       into a merge commit body, independent of any individual commit);
    3. the ADDED lines of `git diff <base>..<head>` over tracked text files --
       code, docs, comments, anything a PR actually ships.

Two independent detection layers:

  Layer 1 -- GENERIC patterns (hardcoded below; safe to publish, because they
  describe a STRUCTURAL shape -- "looks like a private IPv4 literal", "looks
  like an ssh key-file reference" -- never a specific real identifier).
  Internal machine/hostnames on their own are NOT in scope here (maintainer
  ruling 2026-08-28: this repo treats hostnames as public-safe); what this
  layer catches is the small set of shapes that are load-bearing regardless
  of whose name is attached -- a route INTO a private network or a credential
  a reader could act on:
      private_ipv4_10          10.0.0.0/8
      private_ipv4_172         172.16.0.0/12
      private_ipv4_192_168     192.168.0.0/16
      private_ipv4_cgnat       100.64.0.0/10 (carrier-grade NAT; this is also
                                the range Tailscale hands out its own
                                addresses from)
      ssh_key_file             id_ed25519[_...], id_rsa[...], id_ecdsa[...],
                                id_dsa[...]
      ssh_key_flag             `-i ~/.ssh/...`
      ssh_proxycommand         `ProxyCommand`
      tailscale_nc             `tailscale nc ...`
      mac_address              a colon- or hyphen-separated MAC address

  Layer 2 -- an OPTIONAL PRIVATE denylist of exact tokens (real hostnames,
  tailnet names, aliases) for a maintainer who wants to additionally flag
  specific known identifiers, loaded from a file OUTSIDE this repository:
      path = $ESHKOL_DISCLOSURE_DENYLIST, else ~/.eshkol/disclosure-denylist.txt
      format = one token per line, `#`-prefixed lines and blank lines ignored
      match = case-insensitive substring
  This file must never be committed here. In CI it is materialized from a
  repository secret (see .github/workflows/ci.yml) when one is configured.
  This layer is off by default and that is a normal, supported mode: if no
  denylist file is found, the gate prints one neutral line noting that Layer
  2 did not run and proceeds with Layer 1 alone.

Allowlist (`scripts/disclosure_allow.txt`, tracked, public-safe): exact
phrases that are permitted to remain even though a piece of them might
otherwise resemble a hit -- reproducible-benchmark hardware names ("Apple M2
Ultra") and CI capability labels ("self-hosted", "Linux", "X64"). A match is
suppressed only when its exact character span sits fully inside an
allowlisted phrase's span on the same line; a hit next to, but not inside,
an allowlisted phrase still fails.

Output never prints a matched token verbatim -- not even for Layer 1 hits
(the CI log this prints into is itself public in a public repo, and a
generic-pattern hit can still be a live secret, e.g. a real private IP).
Every finding is reported as `<location>: <pattern class>: <redacted excerpt>`
with the matched span itself replaced by `[REDACTED]`, keeping only a fixed
window of surrounding context for orientation. Layer 2 hits are never even
associated with which denylist token matched -- only the class name
"denylist" is reported -- so the private list's own contents cannot be
reconstructed from a failing gate's output.

Usage
    python3 scripts/check_disclosure.py --base <ref> --head <ref>
        [--pr-title-file FILE] [--pr-body-file FILE]
        [--denylist-path FILE] [--allowlist-path FILE]
        [--format text|json] [--trace-dir DIR] [--no-trace]
    python3 scripts/check_disclosure.py --noop-pass --reason "..."
    python3 scripts/check_disclosure.py --self-test

PR title/body may also be supplied via the PR_TITLE / PR_BODY environment
variables instead of `--pr-title-file` / `--pr-body-file` (used by CI, which
routes untrusted PR text through env vars and files -- never through direct
shell-command interpolation).

Exit status is 0 on PASS (including `--noop-pass` and a clean `--self-test`)
and 1 on FAIL (any finding, or a self-test that could not distinguish a
broken fixture from a clean one).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import NamedTuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "disclosure_gate.jsonl"
PROBE_ID = "disclosure_clean"

DEFAULT_ALLOWLIST_PATH = os.path.join(REPO_ROOT, "scripts", "disclosure_allow.txt")
DENYLIST_ENV_VAR = "ESHKOL_DISCLOSURE_DENYLIST"
DEFAULT_DENYLIST_PATH = os.path.expanduser("~/.eshkol/disclosure-denylist.txt")

SCRATCH_DIR = os.path.join(REPO_ROOT, ".scratch")

REDACT_CONTEXT = 24  # characters of surrounding, non-secret context to keep


class DisclosureError(Exception):
    """Raised when a required git operation cannot be completed."""


class Finding(NamedTuple):
    location: str
    pattern: str
    excerpt: str


# ───────────────────────── Layer 1: generic patterns ─────────────────────────
#
# Every regex here matches a STRUCTURAL shape, not a specific secret -- the
# pattern names and regex bodies are safe to publish even though what they
# catch, in a real hit, is not.

def _mac_address_pattern() -> "re.Pattern[str]":
    return re.compile(r"\b[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}\b")


GENERIC_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("private_ipv4_10", re.compile(r"\b10\.(?:\d{1,3}\.){2}\d{1,3}\b")),
    ("private_ipv4_172", re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b")),
    ("private_ipv4_192_168", re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")),
    ("private_ipv4_cgnat", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")),
    ("ssh_key_file", re.compile(r"\bid_(?:ed25519|rsa|ecdsa|dsa)(?:_[A-Za-z0-9._-]+)?\b", re.IGNORECASE)),
    ("ssh_key_flag", re.compile(r"-i\s+~/\.ssh/\S+")),
    ("ssh_proxycommand", re.compile(r"\bProxyCommand\b", re.IGNORECASE)),
    ("tailscale_nc", re.compile(r"\btailscale\s+nc\b", re.IGNORECASE)),
    ("mac_address", _mac_address_pattern()),
]


# ───────────────────────────── allowlist ─────────────────────────────

def load_allowlist(path: str) -> list[str]:
    """Load public-safe phrases, one per line. '#'-prefixed and blank lines
    are ignored. Missing file -> empty allowlist (not an error: a repo that
    has never needed one yet is still clean)."""
    if not os.path.isfile(path):
        return []
    phrases = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            phrases.append(line)
    return phrases


def find_allowlist_spans(line: str, allow_phrases: list[str]) -> list[tuple[int, int]]:
    spans = []
    for phrase in allow_phrases:
        if not phrase:
            continue
        for m in re.finditer(re.escape(phrase), line, re.IGNORECASE):
            spans.append((m.start(), m.end()))
    return spans


def is_allowlisted(start: int, end: int, allow_spans: list[tuple[int, int]]) -> bool:
    return any(a_start <= start and end <= a_end for a_start, a_end in allow_spans)


# ───────────────────────────── denylist (Layer 2) ─────────────────────────────

def load_denylist(path: str | None) -> tuple[list[str], bool, str]:
    """Returns (tokens, active, note). Layer 2 is optional: `active` is False
    whenever no denylist file is configured, which is a normal, supported
    mode, not a failure -- `note` is then a single neutral line (never
    warning-toned) recording that fact for the caller to print."""
    resolved = path or os.environ.get(DENYLIST_ENV_VAR) or DEFAULT_DENYLIST_PATH
    if not os.path.isfile(resolved):
        note = (
            f"note: private disclosure-denylist layer not configured (optional; "
            f"set ${DENYLIST_ENV_VAR} or place a file at "
            f"~/.eshkol/disclosure-denylist.txt to enable it) -- running generic patterns only."
        )
        return [], False, note
    tokens = []
    with open(resolved, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens.append(line)
    return tokens, True, ""


def scan_denylist_spans(line: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans = []
    lower_line = line.lower()
    for tok in tokens:
        tok_l = tok.lower()
        if not tok_l:
            continue
        start = 0
        while True:
            idx = lower_line.find(tok_l, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(tok_l)))
            start = idx + 1
    return spans


# ───────────────────────────── redaction ─────────────────────────────

def redact(line: str, start: int, end: int, context: int = REDACT_CONTEXT) -> str:
    """The matched span itself is NEVER included in the result -- only a
    fixed window of surrounding, non-matching context, so a finding is
    orientable without reprinting the secret it names."""
    prefix = line[max(0, start - context):start]
    suffix = line[end:end + context]
    if start - context > 0:
        prefix = "…" + prefix
    if end + context < len(line):
        suffix = suffix + "…"
    return f"{prefix}[REDACTED]{suffix}"


# ───────────────────────────── per-line analysis ─────────────────────────────

def analyze_line(location: str, line: str, allow_phrases: list[str], denylist_tokens: list[str]) -> list[Finding]:
    if not line:
        return []
    allow_spans = find_allowlist_spans(line, allow_phrases)
    findings: list[Finding] = []
    for name, regex in GENERIC_PATTERNS:
        for m in regex.finditer(line):
            if is_allowlisted(m.start(), m.end(), allow_spans):
                continue
            findings.append(Finding(location, name, redact(line, m.start(), m.end())))
    if denylist_tokens:
        for start, end in scan_denylist_spans(line, denylist_tokens):
            if is_allowlisted(start, end, allow_spans):
                continue
            findings.append(Finding(location, "denylist", redact(line, start, end)))
    return findings


# ───────────────────────────── git-backed sources ─────────────────────────────

def _run_git(args: list[str], repo_root: str = REPO_ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise DisclosureError(f"git {' '.join(args)} failed: {exc} {stderr}".strip()) from exc
    return proc.stdout


def get_commit_records(base: str, head: str, repo_root: str = REPO_ROOT) -> list[tuple[str, str]]:
    """Returns [(sha, body), ...] for every commit in base..head."""
    out = _run_git(["log", "--format=%H%x1f%B%x1e", f"{base}..{head}"], repo_root)
    records = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        sha, _, body = rec.partition("\x1f")
        records.append((sha, body))
    return records


def get_diff_text(base: str, head: str, repo_root: str = REPO_ROOT) -> str:
    return _run_git(["diff", "--unified=0", "--no-color", f"{base}..{head}", "--"], repo_root)


_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_unified_diff_added_lines(diff_text: str):
    """Yields (file, line_no, text) for every ADDED line of a `--unified=0`
    diff, skipping binary files. `line_no` is the 1-based line number in the
    NEW version of the file, tracked purely from the hunk header plus the
    order of `+`/`-` lines within it (unified=0 means no context lines
    interleave, so this is exact)."""
    current_file: str | None = None
    is_binary = False
    new_line_no = 0
    for raw_line in diff_text.split("\n"):
        if raw_line.startswith("diff --git"):
            current_file = None
            is_binary = False
            continue
        if raw_line.startswith("Binary files"):
            is_binary = True
            continue
        if raw_line.startswith("+++ "):
            path = raw_line[4:]
            current_file = None if path.strip() == "/dev/null" else (
                path[2:] if path.startswith("b/") else path
            )
            continue
        if raw_line.startswith("@@"):
            m = _HUNK_HEADER_RE.match(raw_line)
            if m:
                new_line_no = int(m.group(1))
            continue
        if current_file is None or is_binary:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            yield (current_file, new_line_no, raw_line[1:])
            new_line_no += 1
        # '-' lines (removed) do not advance new-file line numbering; every
        # other line (file-mode headers, "index ..." lines, etc.) is ignored.


# ───────────────────────────── PR title/body loading ─────────────────────────────

def load_text(file_arg: str | None, env_var: str) -> str:
    if file_arg:
        with open(file_arg, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return os.environ.get(env_var, "")


# ───────────────────────────── trace ─────────────────────────────

def emit_trace(trace_dir: str, status: str, snippet: str) -> str:
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, TRACE_BASENAME)
    event = {
        "kind": "eshkol_smoke",
        "name": PROBE_ID,
        "value": status,
        "snippet": snippet[:2000],
        "confidence": 1.0,
    }
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


# ───────────────────────────── self-test ─────────────────────────────
#
# "A gate that cannot fail is not a gate." Every fixture below is entirely
# synthetic -- no real private identifier appears anywhere in this file or
# in this function.

_GENERIC_FIXTURES: list[tuple[str, str, str]] = [
    # (pattern_name, triggering_text, benign_text)
    ("private_ipv4_10", "reachable at 10.20.30.40 over the tunnel", "built against LLVM 10.0.1"),
    ("private_ipv4_172", "internal box on 172.20.5.9", "measured 172.99.5.9 packets/sec (not a private range)"),
    ("private_ipv4_192_168", "bound to 192.168.1.50 on the LAN", "adjacent, non-private 192.169.1.50"),
    ("private_ipv4_cgnat", "tailnet address 100.96.130.16", "unrelated value 100.1.2.3 (first octet out of CGNAT range)"),
    ("ssh_key_file", "use -i id_ed25519_workstation for auth", "the identity provider handles auth"),
    ("ssh_key_flag", "ssh -i ~/.ssh/id_ed25519_prod host", "the -i flag inverts the sense of the match"),
    ("ssh_proxycommand", "ProxyCommand ssh -W %h:%p jumpbox", "check the ProxyServer settings"),
    ("tailscale_nc", "tailscale nc some-host 22", "tailscale status looked healthy"),
    ("mac_address", "interface at b8:27:eb:12:34:56", "timestamp read 12:34:56 UTC"),
]


def _self_test_generic() -> bool:
    all_ok = True
    print("  generic patterns:")
    for name, trigger_text, benign_text in _GENERIC_FIXTURES:
        trigger_hits = {f.pattern for f in analyze_line("fixture", trigger_text, [], [])}
        benign_hits = {f.pattern for f in analyze_line("fixture", benign_text, [], [])}
        trigger_ok = name in trigger_hits
        benign_ok = name not in benign_hits
        ok = trigger_ok and benign_ok
        all_ok = all_ok and ok
        verdict = "OK" if ok else "GATE IS BROKEN"
        print(f"    [{verdict}] {name}: trigger_flagged={trigger_ok} benign_clean={benign_ok}")
        if not ok:
            print(f"             trigger hits={sorted(trigger_hits)} benign hits={sorted(benign_hits)}")
    return all_ok


def _self_test_allowlist() -> bool:
    # A synthetic allowlist entry that fully contains a text span which would
    # otherwise trigger `ssh_proxycommand` on its own.
    allow_phrases = ["ProxyCommand tunnel-demo"]
    inside_line = "See ProxyCommand tunnel-demo in the docs for the harness contract."
    outside_line = "ProxyCommand appears here with no allowlisted phrase around it."

    suppressed = analyze_line("fixture", inside_line, allow_phrases, [])
    still_flagged = analyze_line("fixture", outside_line, allow_phrases, [])

    suppressed_ok = not any(f.pattern == "ssh_proxycommand" for f in suppressed)
    still_flagged_ok = any(f.pattern == "ssh_proxycommand" for f in still_flagged)
    ok = suppressed_ok and still_flagged_ok

    verdict = "OK" if ok else "GATE IS BROKEN"
    print(f"  allowlist: [{verdict}] suppresses a hit fully inside an allowlisted phrase, "
          f"still flags the same pattern outside one (suppressed_ok={suppressed_ok} still_flagged_ok={still_flagged_ok})")

    # The real, tracked allowlist file must also parse without error.
    real_phrases = load_allowlist(DEFAULT_ALLOWLIST_PATH)
    real_ok = len(real_phrases) > 0
    print(f"  allowlist: [{'OK' if real_ok else 'GATE IS BROKEN'}] "
          f"{DEFAULT_ALLOWLIST_PATH} loads {len(real_phrases)} phrase(s)")
    return ok and real_ok


def _self_test_denylist() -> bool:
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    fixture_path = None
    all_ok = True
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=SCRATCH_DIR,
            prefix=".selftest-disclosure-denylist-",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as handle:
            fixture_path = handle.name
            fake_token = "zzz-selftest-fakehost-4471"
            handle.write(f"# self-test fixture, not a real identifier\n{fake_token}\n")

        tokens, active, note = load_denylist(fixture_path)
        active_ok = active and not note and fake_token in tokens
        print(f"  denylist (active): [{'OK' if active_ok else 'GATE IS BROKEN'}] "
              f"loaded from a repo-local .scratch/ fixture (never /tmp), active={active}")
        all_ok = all_ok and active_ok

        hit_line = f"observed the token {fake_token} in the wild"
        clean_line = "nothing sensitive on this line at all"
        hit_findings = analyze_line("fixture", hit_line, [], tokens)
        clean_findings = analyze_line("fixture", clean_line, [], tokens)

        hit_ok = any(f.pattern == "denylist" for f in hit_findings)
        clean_ok = not any(f.pattern == "denylist" for f in clean_findings)
        # The fake token itself must never appear in the reported excerpt.
        redaction_ok = all(fake_token not in f.excerpt for f in hit_findings)
        ok = hit_ok and clean_ok and redaction_ok
        print(f"    [{'OK' if ok else 'GATE IS BROKEN'}] flags the fixture token (hit_ok={hit_ok}), "
              f"stays clean without it (clean_ok={clean_ok}), never echoes it back (redaction_ok={redaction_ok})")
        all_ok = all_ok and ok
    finally:
        if fixture_path and os.path.exists(fixture_path):
            os.remove(fixture_path)

    # Inactive path: must return an empty token set AND a non-empty warning
    # note -- the thing that keeps "no denylist configured" from ever being
    # silent.
    missing_path = os.path.join(SCRATCH_DIR, ".selftest-nonexistent-denylist-does-not-exist.txt")
    tokens2, active2, note2 = load_denylist(missing_path)
    inactive_ok = (tokens2 == []) and (active2 is False) and bool(note2.strip())
    print(f"  denylist (inactive): [{'OK' if inactive_ok else 'GATE IS BROKEN'}] "
          f"empty token set, active=False, and a non-empty visible NOTE is always produced "
          f"(never a silent fail-open)")
    all_ok = all_ok and inactive_ok

    return all_ok


def _self_test_diff_line_tracking() -> bool:
    # A synthetic unified=0 diff exercising: pure addition, a replace
    # (removal immediately followed by addition), and a second hunk further
    # down the file -- proving new-file line numbers are tracked correctly
    # without relying on context lines (there are none, by construction).
    diff_text = (
        "diff --git a/example.txt b/example.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/example.txt\n"
        "+++ b/example.txt\n"
        "@@ -5,1 +5,1 @@\n"
        "-old line five\n"
        "+new line five\n"
        "@@ -20,0 +21,2 @@\n"
        "+new line twenty-one\n"
        "+new line twenty-two\n"
    )
    got = list(parse_unified_diff_added_lines(diff_text))
    expected = [
        ("example.txt", 5, "new line five"),
        ("example.txt", 21, "new line twenty-one"),
        ("example.txt", 22, "new line twenty-two"),
    ]
    ok = got == expected
    print(f"  diff line-tracking: [{'OK' if ok else 'GATE IS BROKEN'}] new-file line numbers "
          f"recovered exactly from a --unified=0 diff (no context lines)")
    if not ok:
        print(f"    expected {expected}, got {got}")

    binary_diff = (
        "diff --git a/image.png b/image.png\n"
        "index 1111111..2222222 100644\n"
        "Binary files a/image.png and b/image.png differ\n"
    )
    binary_got = list(parse_unified_diff_added_lines(binary_diff))
    binary_ok = binary_got == []
    print(f"  diff line-tracking: [{'OK' if binary_ok else 'GATE IS BROKEN'}] "
          f"binary files are skipped, not scanned as text")
    return ok and binary_ok


def self_test() -> bool:
    print("check_disclosure.py self-test:")
    results = [
        _self_test_generic(),
        _self_test_allowlist(),
        _self_test_denylist(),
        _self_test_diff_line_tracking(),
    ]
    all_ok = all(results)
    if all_ok:
        print("self-test: PASS -- every generic pattern class discriminates its trigger fixture from its "
              "benign fixture, the allowlist suppresses only spans it fully contains, the private denylist "
              "layer is exercised both active (via a .scratch/ fixture, never /tmp) and inactive (with a "
              "guaranteed visible NOTE, never a silent pass), and diff line-number tracking is exact.")
    else:
        print("self-test: FAIL -- the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


# ───────────────────────────── main ─────────────────────────────

def _format_findings(findings: list[Finding]) -> list[str]:
    return [f"{f.location}: {f.pattern}: {f.excerpt}" for f in findings]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="base ref/sha (start of the exclusive commit range)")
    parser.add_argument("--head", default="HEAD", help="head ref/sha (default: HEAD)")
    parser.add_argument("--pr-title-file")
    parser.add_argument("--pr-body-file")
    parser.add_argument("--denylist-path", help=f"default: ${DENYLIST_ENV_VAR} or {DEFAULT_DENYLIST_PATH}")
    parser.add_argument("--allowlist-path", default=DEFAULT_ALLOWLIST_PATH)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--noop-pass", action="store_true",
                         help="emit a PASS with no scanning (for CI events that carry no PR text to check)")
    parser.add_argument("--reason", default="", help="explanation recorded alongside --noop-pass")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if args.noop_pass:
        snippet = args.reason or "no-op pass (nothing to scan for this event)"
        if not args.no_trace:
            emit_trace(args.trace_dir, "PASS", snippet)
        print(f"{PROBE_ID}: PASS (no-op) -- {snippet}")
        return 0

    if not args.base:
        parser.error("--base is required unless --self-test or --noop-pass is given")

    allow_phrases = load_allowlist(args.allowlist_path)
    denylist_tokens, denylist_active, denylist_note = load_denylist(args.denylist_path)
    if not denylist_active:
        print(denylist_note, file=sys.stderr)

    findings: list[Finding] = []

    try:
        for sha, body in get_commit_records(args.base, args.head):
            for line_no, line in enumerate(body.split("\n"), start=1):
                findings.extend(analyze_line(f"commit {sha[:8]}:{line_no}", line, allow_phrases, denylist_tokens))

        title_text = load_text(args.pr_title_file, "PR_TITLE")
        body_text = load_text(args.pr_body_file, "PR_BODY")
        for line_no, line in enumerate(title_text.split("\n"), start=1):
            findings.extend(analyze_line(f"pr-title:{line_no}", line, allow_phrases, denylist_tokens))
        for line_no, line in enumerate(body_text.split("\n"), start=1):
            findings.extend(analyze_line(f"pr-body:{line_no}", line, allow_phrases, denylist_tokens))

        diff_text = get_diff_text(args.base, args.head)
        for file_path, line_no, text in parse_unified_diff_added_lines(diff_text):
            findings.extend(analyze_line(f"{file_path}:{line_no}", text, allow_phrases, denylist_tokens))
    except DisclosureError as exc:
        snippet = f"could not scan {args.base}..{args.head}: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL -- {exc}", file=sys.stderr)
        return 1

    passed = not findings
    status = "PASS" if passed else "FAIL"
    finding_lines = _format_findings(findings)

    if passed:
        layer2 = "active" if denylist_active else "not configured"
        snippet = f"clean over {args.base}..{args.head} (private denylist layer: {layer2})"
    else:
        snippet = f"{len(finding_lines)} finding(s): " + "; ".join(finding_lines[:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({
            "status": status,
            "denylist_active": denylist_active,
            "findings": finding_lines,
        }, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  private denylist layer: {'active' if denylist_active else 'not configured (optional)'}")
        if finding_lines:
            print("  FINDINGS (matched token redacted; fix by removing/rewording the source line, "
                  "then force-push or amend before merge):")
            for line in finding_lines:
                print(f"    {line}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
