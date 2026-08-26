#!/usr/bin/env python3
"""Structural gate: every AD operator's differentiation carrier is declared,
verified against its source, and exact wherever parity is claimed.

Motivating incident: the 2026-08-25 architectural audit found `op:GRADIENT`
promoted from `gap` to `vm-supported` on a VM-only forward-dual path
(`VmDual`, lib/backend/vm_native.c) that shares zero code with the native AD
node model in lib/backend/autodiff_codegen.cpp. Two independent AD
implementations existed and every gate stayed green, because the only check
that compared them compared PROGRAM OUTPUT. Two independently-correct
implementations agree on output, so an output differential is structurally
incapable of seeing a fork. The same blindness let `curl` ship as a CENTRAL
DIFFERENCE at h = 1e-7 -- measured returning
#(1.1102230246251565e-09 -3.3306690738754696e-09 0) for a field whose curl is
exactly zero -- while `INV-ad-exact-no-finite-differences` in
.icc/architecture-model.yaml read PASS. That invariant is
`kind: dependency-presence`: it asks whether an exact constructor is PRESENT
in the native codegen, never whether a finite difference is ABSENT, and its
site set contains no VM file at all. It was green and could not have gone red.

This gate asserts the property BY CONSTRUCTION instead of by comparison. It
grades .icc/ad-carrier-manifest.yaml against tests/vm_parity/PARITY.tsv and
against the source itself:

  C1 MANIFEST     the manifest parses, matches its schema, and names carriers
                  that exist in its own carrier vocabulary.
  C2 COVERAGE     every AD operator row in PARITY.tsv is covered by exactly
                  one manifest operator, and every parity row the manifest
                  names exists in PARITY.tsv. An AD op cannot become
                  `vm-supported` without declaring which carrier answers it.
  C3 VERIFICATION the declared VM carrier is RE-DERIVED from source: the
                  operator's `case <native_call_id>:` block is extracted from
                  its declared file, comments and literals are stripped, and
                  the block is classified by which carrier witnesses it
                  actually touches. Declared must equal observed. The manifest
                  cannot lie, and editing it cannot fix a red gate.
  C4 EXACTNESS    a parity row reading `vm-supported` requires an exact
                  carrier. A finite difference may never back a vm-supported
                  AD op.
  C5 RATCHET      the number of operators whose VM carrier is not the shared
                  node model may not exceed `fork_debt.forked_carrier_budget`.
                  Shrink-only: the fork is capped at its measured size.
  C6 FD LEDGER    every difference-quotient site in `fd_scan.paths` appears in
                  `fd_allowlist` with an owner, the op it serves and a
                  justification. An unledgered finite difference is a FAIL.
  C7 SCAN SCOPE   `fd_scan.paths` covers every file any operator names, so a
                  finding cannot be silenced by deleting a path.

WHAT COUNTS AS A FINITE DIFFERENCE
    The magnitude of a constant proves nothing: lib/backend/autodiff_codegen.cpp
    uses 1e-7 six times as a domain-guard clamp on a Poincare-ball divisor
    (`select(denom < eps, eps, denom)`), and lib/backend/vm_dual.c uses 1e-12
    as an equality tolerance. Neither is a derivative approximation. What makes
    a site a finite difference is the DIFFERENCE-QUOTIENT SHAPE: an
    epsilon-valued name that is added to a base expression, subtracted from the
    same base, and divided into the resulting difference. A clamp or a
    tolerance only ever reaches its epsilon through a comparison. This gate
    detects the shape, not the number.

Grading
    PASS  C1-C7 all hold.
    FAIL  any check fails, or the manifest / PARITY.tsv / a scanned source
          file is absent or unreadable.

The gate FAILS CLOSED. A missing manifest is not evidence that no AD fork
exists, and it is the exact class of false green this gate was added to end.

Usage
    python3 scripts/gate_ad_shared_node_model.py
    python3 scripts/gate_ad_shared_node_model.py --manifest path/to/file.yaml
    python3 scripts/gate_ad_shared_node_model.py --format json
    python3 scripts/gate_ad_shared_node_model.py --self-test

Exit status is 0 on PASS and 1 on FAIL, so the script also works as a plain
CI step without ICC.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, ".icc", "ad-carrier-manifest.yaml")
DEFAULT_PARITY = os.path.join(REPO_ROOT, "tests", "vm_parity", "PARITY.tsv")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "ad_carrier_gate.jsonl"

EXPECTED_SCHEMA = "eshkol.ad_carrier_manifest.v1"
PROBE_ID = "ad_carrier_model_clean"

FD_CARRIER = "finite-difference"
SHARED_CARRIER = "shared-node-model"

# PARITY.tsv statuses that assert the VM answers this op.
VM_SUPPORTED = "vm-supported"


class GateError(Exception):
    """Any condition that makes the gate ungradeable. Always a FAIL."""


# ───────────────────────────── loading ─────────────────────────────

def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment problem
        raise GateError(f"PyYAML is required to grade the AD carrier manifest: {exc}")
    if not os.path.isfile(path):
        raise GateError(f"AD carrier manifest not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        raise GateError(f"AD carrier manifest at {path} is unparseable: {exc}")
    if not isinstance(data, dict):
        raise GateError(f"AD carrier manifest at {path} is not a mapping")
    return data


def load_parity(path: str) -> dict[str, str]:
    """Return {name: status} for every non-comment row of PARITY.tsv."""
    if not os.path.isfile(path):
        raise GateError(f"VM parity manifest not found at {path}")
    rows: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            rows[fields[0].strip()] = fields[1].strip()
    if not rows:
        raise GateError(f"VM parity manifest at {path} yielded no rows")
    return rows


# ───────────────────────── source normalisation ─────────────────────────

_C_LIKE = (".c", ".cpp", ".cc", ".h", ".hpp")


def strip_noise(text: str, path: str) -> str:
    """Remove comments and string/char literals, preserving line structure.

    Prose matters here: lib/backend/vm_native.c documents its own finite
    differences in a comment ("Compute the 3x3 Jacobian via central
    differences"). A scanner that read comments would grade the documentation
    rather than the code, and could be silenced by deleting an honest comment.
    Newlines are preserved so reported line numbers stay true to the file.
    """
    if path.endswith(".esk") or path.endswith(".scm"):
        out = []
        for line in text.split("\n"):
            # A `;` inside a string is not a comment; strings are rare in the
            # AD modules and are removed first.
            line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
            idx = line.find(";")
            out.append(line[:idx] if idx >= 0 else line)
        return "\n".join(out)

    def _blank(match: re.Match) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    text = re.sub(r"/\*.*?\*/", _blank, text, flags=re.S)
    text = re.sub(r"//[^\n]*", _blank, text)
    text = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)
    text = re.sub(r"'(?:[^'\\\n]|\\.)'", "' '", text)
    return text


def read_source(path: str) -> str:
    if not os.path.isfile(path):
        raise GateError(f"declared AD source file is missing: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def extract_case_block(text: str, case_id: int) -> str:
    """Return the brace-balanced body of `case <case_id>:` in normalised text.

    The VM dispatches native calls from one switch, so an operator's whole
    implementation is exactly one case block. Extracting it -- rather than
    grepping the 14k-line file -- is what makes the carrier classification
    per-operator instead of per-file.
    """
    match = re.search(r"(?m)^[ \t]*case\s+%d\s*:" % case_id, text)
    if not match:
        raise GateError(f"no `case {case_id}:` found in the declared VM source")
    start = text.find("{", match.end())
    if start < 0:
        raise GateError(f"`case {case_id}:` has no block body")
    depth = 0
    for pos in range(start, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise GateError(f"`case {case_id}:` block is unbalanced")


def extract_function_block(text: str, name: str) -> str:
    """Return the brace-balanced body of function `name` in normalised text.

    An operator may delegate its whole implementation to a helper -- case 750
    is one line, `vm_push(vm, vm_gradient_compute(vm, f_val, x_val))`. Without
    this, delegation would launder a carrier past the classifier, which is the
    single most obvious way to defeat a case-block gate. A helper must be
    DECLARED in the manifest, so the audit trail names it too.
    """
    pattern = re.compile(
        r"(?m)^[A-Za-z_][\w\s\*&:<>,]*?\b" + re.escape(name) + r"\s*\([^;{)]*\)\s*\{"
    )
    match = pattern.search(text)
    if not match:
        raise GateError(f"declared helper {name!r} has no definition in the declared VM source")
    start = text.rindex("{", match.start(), match.end())
    depth = 0
    for pos in range(start, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise GateError(f"helper {name!r} has an unbalanced body")


# ─────────────────── finite-difference shape detection ───────────────────

# An epsilon-looking numeric literal: scientific notation with a negative
# exponent of 3 or more, or a small plain decimal. Below 1e-3 nothing in this
# codebase is a step size, and above it nothing is a tolerance.
EPS_LITERAL = r"(?<![\w.])\d+(?:\.\d+)?[eE]-0*(?:[3-9]|[1-9]\d)(?![\w.])"
EPS_DECIMAL = r"(?<![\w.])0\.0{2,}\d+(?![\w.])"
EPS_ANY = f"(?:{EPS_LITERAL}|{EPS_DECIMAL})"

# `->`, `++`, `--` and the `e-` of an exponent all contain `+`/`-` characters
# that mean nothing arithmetically. They are neutralised before the shape test
# so a member access next to an epsilon cannot read as a perturbation.
#
# Every substitution is LENGTH-PRESERVING on purpose: bindings are located in
# the original text and the shape test runs over the neutralised copy, so the
# two must agree on offsets. (Neutralising first would also destroy the very
# `1e-7` the binding scan is looking for.)
_NEUTRALISE = [
    (re.compile(r"->"), "_a"),
    (re.compile(r"\+\+"), "_p"),
    (re.compile(r"--"), "_m"),
    (re.compile(r"([0-9])([eE])-([0-9])"), r"\1\2_\3"),
]


def _neutralise(text: str) -> str:
    for pattern, repl in _NEUTRALISE:
        text = pattern.sub(repl, text)
    return text


def _block_around(text: str, pos: int, path: str) -> str:
    """Return the innermost brace block containing offset `pos`.

    Scheme is scoped differently and deliberately handled differently: an
    epsilon in an AD module is a top-level `(define tape-fd-eps 1.0e-6)` whose
    uses are in a sibling procedure, so the enclosing form is the define
    itself and tells us nothing. For .esk the whole module is the scope.
    """
    if path.endswith((".esk", ".scm")):
        return text
    opener, closer = "{", "}"
    depth = 0
    start = 0
    for i in range(pos, -1, -1):
        ch = text[i]
        if ch == closer:
            depth += 1
        elif ch == opener:
            if depth == 0:
                start = i
                break
            depth -= 1
    depth = 0
    end = len(text)
    for i in range(start, len(text)):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return text[start:end]


def _bindings(text: str, path: str) -> list[tuple[str, int]]:
    """Find (name, offset) for every name bound to an epsilon-looking value."""
    found: list[tuple[str, int]] = []
    if path.endswith((".esk", ".scm")):
        for m in re.finditer(r"\(define\s+([A-Za-z_][\w!?*+/<>=-]*)\s+" + EPS_ANY, text):
            found.append((m.group(1), m.start()))
        return found
    # `double h = 1e-7;`, `llvm::Value* eps = ConstantFP::get(ty, 1e-7);`
    for m in re.finditer(r"([A-Za-z_]\w*)\s*=\s*[^;=\n]*?" + EPS_ANY, text):
        found.append((m.group(1), m.start()))
    # `#define AD_EPS 1e-10`
    for m in re.finditer(r"#\s*define\s+([A-Za-z_]\w*)\s+" + EPS_ANY, text):
        found.append((m.group(1), m.start()))
    return found


def _has_shape(block: str, name: str) -> tuple[bool, bool, bool]:
    """(added, subtracted, divided-into) for `name` within `block`."""
    esc = re.escape(name)
    near = r"[^;\n]{0,60}"
    added = bool(re.search(r"\+" + near + r"(?<![\w])" + esc + r"(?![\w])", block))
    subtracted = bool(re.search(r"-" + near + r"(?<![\w])" + esc + r"(?![\w])", block))
    divided = bool(re.search(r"/" + near + r"(?<![\w])" + esc + r"(?![\w])", block))
    return added, subtracted, divided


def find_fd_sites(text: str, path: str) -> list[dict]:
    """Return every difference-quotient site in normalised `text`.

    A site qualifies when an epsilon-valued name is DIVIDED INTO a quantity and
    is also ADDED to a base expression -- the signature of `(f(x+h) - f(x))/h`
    -- and is classified `central` when it is subtracted as well. A name that
    only ever appears in a comparison (a clamp, a tolerance) reaches none of
    these and is correctly not a site.
    """
    flat = _neutralise(text)
    sites: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for name, pos in _bindings(text, path):
        block = _block_around(flat, pos, path)
        added, subtracted, divided = _has_shape(block, name)
        if not (divided and added):
            continue
        line = flat.count("\n", 0, pos) + 1
        key = (name, line)
        if key in seen:
            continue
        seen.add(key)
        sites.append(
            {
                "path": path,
                "symbol": name,
                "line": line,
                "kind": "central" if subtracted else "forward",
            }
        )
    return sites


# ───────────────────── carrier classification ─────────────────────

def classify_carrier(block: str, carriers: dict, path: str) -> str:
    """Return the carrier a block ACTUALLY uses.

    Finite difference wins over everything: a block that seeds duals and then
    falls back to a difference quotient for some shapes is a finite-difference
    block, because the answer a caller receives can be the approximated one.
    That precedence is the whole reason `divergence` was classified honestly.
    """
    if find_fd_sites(block, path):
        return FD_CARRIER
    # Most specific first: a hyper-dual block also mentions dual helpers.
    order = ["vm-hyperdual-forward", "vm-dual-forward", "native-jet", "native-ad-node", SHARED_CARRIER]
    for name in order:
        spec = carriers.get(name)
        if not isinstance(spec, dict):
            continue
        for witness in spec.get("witnesses") or []:
            if re.search(witness, block):
                return name
    return "unclassified"


# ───────────────────────────── the audit ─────────────────────────────

AD_PARITY_ROW = re.compile(
    r"^(?:op:(?:GRADIENT|JACOBIAN|HESSIAN|DIVERGENCE|CURL|LAPLACIAN|DERIVATIVE"
    r"|DERIVATIVE_N|DIFF|DIRECTIONAL_DERIV)"
    r"|gradient|jacobian|hessian|divergence|curl|laplacian|derivative"
    r"|directional-derivative|ad-tape-new|ad-var|ad-const|ad-backward|ad-gradient)$"
)


def audit(manifest: dict, parity: dict[str, str], repo_root: str) -> dict:
    errors: list[str] = []
    findings: list[dict] = []

    # ── C1 manifest well-formed ────────────────────────────────────────
    if manifest.get("schema") != EXPECTED_SCHEMA:
        errors.append(
            f"C1 manifest schema is {manifest.get('schema')!r}, expected {EXPECTED_SCHEMA!r}"
        )
    carriers = manifest.get("carriers")
    if not isinstance(carriers, dict) or not carriers:
        raise GateError("C1 manifest has no `carriers` vocabulary")
    operators = manifest.get("operators")
    if not isinstance(operators, list) or not operators:
        raise GateError("C1 manifest has no `operators` list")
    fork_debt = manifest.get("fork_debt") or {}
    budget = fork_debt.get("forked_carrier_budget")
    if not isinstance(budget, int):
        raise GateError("C1 manifest has no integer `fork_debt.forked_carrier_budget`")
    fd_scan = manifest.get("fd_scan") or {}
    scan_paths = fd_scan.get("paths")
    if not isinstance(scan_paths, list) or not scan_paths:
        raise GateError("C1 manifest has no `fd_scan.paths` list")
    allowlist = manifest.get("fd_allowlist") or []
    if not isinstance(allowlist, list):
        raise GateError("C1 manifest `fd_allowlist` is not a list")

    # ── C2 coverage in both directions ─────────────────────────────────
    covered_rows: dict[str, str] = {}
    for entry in operators:
        op = entry.get("op")
        for row in entry.get("parity_rows") or []:
            if row in covered_rows:
                errors.append(
                    f"C2 parity row {row!r} is claimed by two operators "
                    f"({covered_rows[row]!r} and {op!r})"
                )
            covered_rows[row] = op
            if row not in parity:
                errors.append(
                    f"C2 operator {op!r} names parity row {row!r}, which does not "
                    f"exist in the parity manifest"
                )
    for row in sorted(parity):
        if AD_PARITY_ROW.match(row) and row not in covered_rows:
            errors.append(
                f"C2 AD parity row {row!r} (status {parity[row]!r}) has no operator "
                f"in the carrier manifest -- an AD op cannot claim a VM status "
                f"without declaring which carrier answers it"
            )

    # ── C3 / C4 / C5 per operator ──────────────────────────────────────
    forked = 0
    for entry in operators:
        op = entry.get("op")
        rows = entry.get("parity_rows") or []
        statuses = {parity.get(r) for r in rows if r in parity}
        claims_vm = VM_SUPPORTED in statuses

        for side in ("native", "vm"):
            spec = entry.get(side)
            if spec is None:
                if side == "vm" and claims_vm:
                    errors.append(
                        f"C2 operator {op!r} declares no VM implementation but a "
                        f"parity row reads {VM_SUPPORTED}"
                    )
                continue
            declared = spec.get("carrier")
            if declared not in carriers:
                errors.append(
                    f"C1 operator {op!r} {side} carrier {declared!r} is not in the "
                    f"carrier vocabulary"
                )
                continue
            if side == "vm" and declared != SHARED_CARRIER:
                forked += 1

        vm = entry.get("vm")
        if not isinstance(vm, dict):
            continue
        case_id = vm.get("native_call_id")
        vm_path = vm.get("file")
        declared = vm.get("carrier")
        if case_id is None or not vm_path:
            errors.append(
                f"C3 operator {op!r} VM block needs both `file` and `native_call_id` "
                f"so its carrier can be re-derived from source"
            )
            continue
        abs_path = os.path.join(repo_root, vm_path)
        helpers = vm.get("helpers") or []
        try:
            normalised = strip_noise(read_source(abs_path), vm_path)
            block = extract_case_block(normalised, int(case_id))
            for helper in helpers:
                block += "\n" + extract_function_block(normalised, helper)
        except GateError as exc:
            errors.append(f"C3 operator {op!r}: {exc}")
            continue
        observed = classify_carrier(block, carriers, vm_path)
        findings.append(
            {
                "op": op,
                "native_call_id": case_id,
                "helpers": helpers,
                "declared_carrier": declared,
                "observed_carrier": observed,
                "parity_rows": {r: parity.get(r, "<absent>") for r in rows},
                "claims_vm_supported": claims_vm,
            }
        )
        if observed != declared:
            errors.append(
                f"C3 operator {op!r} (case {case_id} in {vm_path}) declares carrier "
                f"{declared!r} but its source classifies as {observed!r}"
            )
        effective = observed if observed != "unclassified" else declared
        exact = bool((carriers.get(effective) or {}).get("exact"))
        if claims_vm and not exact:
            errors.append(
                f"C4 operator {op!r} has a {VM_SUPPORTED} parity row but its VM "
                f"carrier {effective!r} is not exact -- a finite difference may "
                f"never back a vm-supported AD op"
            )
        if claims_vm and observed == "unclassified":
            errors.append(
                f"C4 operator {op!r} claims {VM_SUPPORTED} but case {case_id} touches "
                f"no known carrier witness -- an unclassifiable AD path cannot be "
                f"graded exact"
            )

    if forked > budget:
        errors.append(
            f"C5 {forked} operators run a VM carrier that is not the shared node "
            f"model, over the shrink-only budget of {budget} -- the AD fork grew"
        )

    # ── C7 scan scope, then C6 FD ledger ───────────────────────────────
    declared_files = set()
    for entry in operators:
        for side in ("native", "vm"):
            spec = entry.get(side)
            if isinstance(spec, dict) and spec.get("file"):
                declared_files.add(spec["file"])
    for path in sorted(declared_files - set(scan_paths)):
        errors.append(
            f"C7 {path} backs an operator but is not in fd_scan.paths -- the scan "
            f"set may not exclude a file the manifest itself relies on"
        )

    ledgered: dict[tuple[str, str], dict] = {}
    for item in allowlist:
        if not isinstance(item, dict):
            errors.append("C6 fd_allowlist contains a non-mapping entry")
            continue
        missing = [f for f in ("path", "symbol", "owner", "op", "justification") if not item.get(f)]
        if missing:
            errors.append(
                f"C6 fd_allowlist entry {item.get('symbol') or '<unnamed>'} is missing "
                f"{', '.join(missing)} -- a ledgered finite difference states who owns "
                f"it, which op it serves and why it is not a defect"
            )
            continue
        ledgered[(item["path"], item["symbol"])] = item

    fd_sites: list[dict] = []
    for rel in scan_paths:
        abs_path = os.path.join(repo_root, rel)
        try:
            normalised = strip_noise(read_source(abs_path), rel)
        except GateError as exc:
            errors.append(f"C6 {exc}")
            continue
        for site in find_fd_sites(normalised, rel):
            site["ledgered"] = (site["path"], site["symbol"]) in ledgered
            fd_sites.append(site)
            if not site["ledgered"]:
                errors.append(
                    f"C6 unledgered {site['kind']} finite difference: {site['path']}:"
                    f"{site['line']} steps by {site['symbol']!r} on an AD path"
                )

    for key in sorted(ledgered):
        if not any(s["path"] == key[0] and s["symbol"] == key[1] for s in fd_sites):
            errors.append(
                f"C6 fd_allowlist entry {key[1]!r} in {key[0]} matches no site the scan "
                f"found -- a stale waiver hides the next real one"
            )

    return {
        "passed": not errors,
        "errors": errors,
        "operators": findings,
        "fd_sites": fd_sites,
        "forked_carriers": forked,
        "forked_carrier_budget": budget,
    }


# ───────────────────────────── reporting ─────────────────────────────

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
# Every fixture below is a RED fixture paired with the GREEN one it was
# derived from. A gate that has only ever been observed green is
# indistinguishable from a gate that cannot go red, which is the defect this
# file exists to end -- so the gate refuses to be trusted until it has
# demonstrated the opposite on each check it claims to enforce.

_GOOD_SOURCE = """
switch (fid) {
case 750: { /* gradient */
    double point[8];
    VM_AD_MAKE_DUAL(vm, point[0], 1.0, dual_arg);
    Value result = vm_call_closure_from_native(vm, f_val, &dual_arg, 1);
    break;
}
case 754: { /* curl */
    VM_AD_MAKE_DUAL(vm, point[j], 1.0, args[j]);
    jac[i][j] = rd->tangent;
    break;
}
}
"""

# The real defect, reduced: a central difference at h = 1e-7 inside the curl
# case, with prose that honestly describes it. This is what shipped.
_FD_SOURCE = """
switch (fid) {
case 750: { /* gradient */
    VM_AD_MAKE_DUAL(vm, point[0], 1.0, dual_arg);
    break;
}
case 754: { /* curl via central differences */
    double h = 1e-7;
    for (int j = 0; j < 3; j++) {
        ap[k] = FLOAT_VAL(point[k] + ((k == j) ? h : 0));
        am[k] = FLOAT_VAL(point[k] - ((k == j) ? h : 0));
        jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
    }
    break;
}
}
"""

# The clamp and the tolerance: the same order of magnitude, used only in a
# comparison. Must NOT be flagged, or the gate is noise and gets disabled.
_CLAMP_SOURCE = """
switch (fid) {
case 750: {
    VM_AD_MAKE_DUAL(vm, point[0], 1.0, dual_arg);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
    llvm::Value* safe = ctx_.builder().CreateSelect(
        ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
    double tol = 1e-12;
    if (fabs(a - b) < tol) { return 1; }
    break;
}
case 754: {
    VM_AD_MAKE_DUAL(vm, point[j], 1.0, args[j]);
    break;
}
}
"""

_BASE_MANIFEST = {
    "schema": EXPECTED_SCHEMA,
    "carriers": {
        SHARED_CARRIER: {"exact": True, "shared": True, "witnesses": [r"\bad_add\s*\("]},
        "vm-dual-forward": {"exact": True, "shared": False, "witnesses": [r"\bVM_AD_MAKE_DUAL\s*\("]},
        "native-jet": {"exact": True, "shared": False, "witnesses": [r"\bseedForwardAndPush\s*\("]},
        FD_CARRIER: {"exact": False, "shared": False, "witnesses": []},
    },
    "fork_debt": {"forked_carrier_budget": 2},
    "fd_scan": {"paths": ["vm.c"]},
    "fd_allowlist": [],
    "operators": [
        {
            "op": "gradient",
            "parity_rows": ["op:GRADIENT"],
            "native": {"file": "vm.c", "carrier": "native-jet"},
            "vm": {"file": "vm.c", "native_call_id": 750, "carrier": "vm-dual-forward"},
        },
        {
            "op": "curl",
            "parity_rows": ["op:CURL"],
            "native": {"file": "vm.c", "carrier": "native-jet"},
            "vm": {"file": "vm.c", "native_call_id": 754, "carrier": "vm-dual-forward"},
        },
    ],
}

_PARITY_GREEN = "# name\tstatus\nop:GRADIENT\tvm-supported\t\nop:CURL\tvm-supported\t\n"


def _deep_copy(obj):
    return json.loads(json.dumps(obj))


def _write_case(tmp_dir: str, name: str, source: str, manifest: dict, parity: str) -> tuple[dict, str, dict]:
    case_dir = os.path.join(tmp_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    with open(os.path.join(case_dir, "vm.c"), "w", encoding="utf-8") as handle:
        handle.write(source)
    parity_path = os.path.join(case_dir, "PARITY.tsv")
    with open(parity_path, "w", encoding="utf-8") as handle:
        handle.write(parity)
    return manifest, case_dir, load_parity(parity_path)


def self_test() -> bool:
    cases: list[tuple[str, str, dict, str, bool, str]] = []

    # GREEN baseline -- exact duals on both ops, nothing to flag.
    cases.append(("green_exact_duals", _GOOD_SOURCE, _deep_copy(_BASE_MANIFEST), _PARITY_GREEN, True,
                  "exact dual carriers, declared == observed"))

    # RED C3/C4/C6 -- the shipped defect: a vm-supported op backed by a
    # central difference while the manifest claims a dual.
    cases.append(("red_central_difference_under_vm_supported", _FD_SOURCE, _deep_copy(_BASE_MANIFEST),
                  _PARITY_GREEN, False,
                  "central difference at h=1e-7 backing a vm-supported op"))

    # GREEN -- a clamp and an equality tolerance at the same magnitude must
    # not be mistaken for a step size.
    cases.append(("green_clamp_and_tolerance_are_not_fd", _CLAMP_SOURCE, _deep_copy(_BASE_MANIFEST),
                  _PARITY_GREEN, True,
                  "1e-7 clamp and 1e-12 tolerance are comparisons, not steps"))

    # GREEN -- the same finite difference, ledgered with an owner and a
    # justification. A declared FD escape hatch is allowed to exist.
    ledgered = _deep_copy(_BASE_MANIFEST)
    ledgered["fd_allowlist"] = [{
        "path": "vm.c", "symbol": "h", "owner": "tsotchke", "op": "curl",
        "justification": "self-test fixture: a deliberately ledgered FD site",
    }]
    ledgered["operators"][1]["vm"]["carrier"] = FD_CARRIER
    ledgered["operators"][1]["parity_rows"] = ["op:CURL"]
    cases.append(("green_ledgered_fd_on_a_gap_row", _FD_SOURCE, ledgered,
                  "# name\tstatus\nop:GRADIENT\tvm-supported\t\nop:CURL\tgap\tledgered FD\n", True,
                  "ledgered FD backing a `gap` row is permitted"))

    # RED C4 -- the same ledgered FD, but the op now claims vm-supported.
    # Ledgering an FD does not license claiming parity on it.
    ledgered_claimed = _deep_copy(ledgered)
    cases.append(("red_ledgered_fd_cannot_claim_vm_supported", _FD_SOURCE, ledgered_claimed,
                  _PARITY_GREEN, False,
                  "a ledgered FD still may not back a vm-supported op"))

    # RED C2 -- a new AD op appears in PARITY.tsv as vm-supported with no
    # manifest row at all. This is the exact shape of the v1.3.4 promotion.
    cases.append(("red_undeclared_vm_supported_ad_op", _GOOD_SOURCE, _deep_copy(_BASE_MANIFEST),
                  _PARITY_GREEN + "op:HESSIAN\tvm-supported\t\n", False,
                  "an AD op promoted to vm-supported with no carrier declaration"))

    # RED C3 -- the manifest is edited to claim the shared node model over a
    # body that plainly uses the private dual carrier. Proves the declaration
    # is verified, not believed.
    lying = _deep_copy(_BASE_MANIFEST)
    lying["operators"][0]["vm"]["carrier"] = SHARED_CARRIER
    cases.append(("red_manifest_claims_shared_model_falsely", _GOOD_SOURCE, lying, _PARITY_GREEN, False,
                  "declaring shared-node-model over a VmDual body"))

    # RED C5 -- the fork grows past its shrink-only budget.
    over = _deep_copy(_BASE_MANIFEST)
    over["operators"].append({
        "op": "laplacian",
        "parity_rows": ["op:LAPLACIAN"],
        "native": {"file": "vm.c", "carrier": "native-jet"},
        "vm": {"file": "vm.c", "native_call_id": 750, "carrier": "vm-dual-forward"},
    })
    cases.append(("red_fork_budget_exceeded", _GOOD_SOURCE, over,
                  _PARITY_GREEN + "op:LAPLACIAN\tvm-supported\t\n", False,
                  "a third forked carrier over a shrink-only budget of two"))

    # RED C7 -- silencing a finding by dropping the file from the scan set.
    narrowed = _deep_copy(_BASE_MANIFEST)
    narrowed["fd_scan"]["paths"] = ["PARITY.tsv"]
    cases.append(("red_scan_set_narrowed_to_hide_a_site", _FD_SOURCE, narrowed, _PARITY_GREEN, False,
                  "removing the scanned file to hide the difference quotient"))

    all_ok = True
    print("gate_ad_shared_node_model.py self-test:")
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-ad-carrier-") as tmp_dir:
        for name, source, manifest, parity, expect_passed, description in cases:
            manifest, case_dir, parity_rows = _write_case(tmp_dir, name, source, manifest, parity)
            try:
                result = audit(manifest, parity_rows, case_dir)
            except GateError as exc:
                result = {"passed": False, "errors": [str(exc)]}
            ok = result["passed"] == expect_passed
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            arrow = "PASS" if expect_passed else "FAIL"
            print(f"  [{verdict}] {name}: expected {arrow} -- {description}")
            if not result["passed"]:
                for err in result["errors"][:3]:
                    print(f"            {err}")

    if all_ok:
        print("self-test: PASS -- the gate goes red on an undeclared vm-supported AD op, "
              "on a false carrier declaration, on an unledgered difference quotient, on a "
              "ledgered one that claims parity, on a grown fork and on a narrowed scan set; "
              "and stays green on exact duals, on clamps and on tolerances")
    else:
        print("self-test: FAIL -- the gate did not discriminate red input from green input",
              file=sys.stderr)
    return all_ok


# ───────────────────────────── entry point ─────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=os.environ.get("ESHKOL_AD_MANIFEST", DEFAULT_MANIFEST))
    parser.add_argument("--parity", default=DEFAULT_PARITY)
    parser.add_argument("--repo-root", default=REPO_ROOT)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        manifest = _load_yaml(args.manifest)
        parity = load_parity(args.parity)
        result = audit(manifest, parity, args.repo_root)
    except GateError as exc:
        snippet = f"AD carrier model ungradeable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL -- {exc}", file=sys.stderr)
        return 1

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = (
            f"{len(result['operators'])} AD operators: every declared carrier re-derived from "
            f"source and matched; {result['forked_carriers']}/{result['forked_carrier_budget']} "
            f"forked carriers; {len(result['fd_sites'])} finite-difference site(s), all ledgered"
        )
    else:
        snippet = f"{len(result['errors'])} finding(s): " + "; ".join(result["errors"][:4])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  manifest    : {args.manifest}")
        print(f"  parity      : {args.parity}")
        print(f"  fork        : {result['forked_carriers']} forked carrier(s), "
              f"budget {result['forked_carrier_budget']} (shrink-only)")
        print()
        header = f"{'operator':<24} {'fid':>5}  {'declared':<22} {'observed':<22} {'vm-supported':<12}"
        print(header)
        print("-" * len(header))
        for f in result["operators"]:
            mark = "" if f["declared_carrier"] == f["observed_carrier"] else "   <-- MISMATCH"
            print(
                f"{f['op']:<24} {str(f['native_call_id']):>5}  {str(f['declared_carrier']):<22} "
                f"{f['observed_carrier']:<22} {str(f['claims_vm_supported']):<12}{mark}"
            )
        if result["fd_sites"]:
            print("\n  finite-difference sites:")
            for s in result["fd_sites"]:
                tag = "ledgered" if s["ledgered"] else "UNLEDGERED"
                print(f"    [{tag}] {s['path']}:{s['line']} {s['kind']} step {s['symbol']!r}")
        if result["errors"]:
            print("\n  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
