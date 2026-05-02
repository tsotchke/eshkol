#!/usr/bin/env python3
"""Generate the JIT/AOT/VM parity ledger by scraping the Eshkol source tree.

Usage:
    python3 tools/icc_extras/generate_parity_ledger.py \
        --repo-root . \
        --out tools/icc_extras/parity_ledger.json \
        [--no-merge]            # discard existing parity statuses, start fresh
        [--missing-only]        # print missing-VM/missing-AOT entries to stdout, write nothing

The generator extracts *structural* facts only — file paths, line numbers, and
presence-or-absence in each backend. It never *infers* a parity status: when it
sees a previously-unseen op or builtin, the entry is born `unverified` and a
human flips it during review. When it re-runs over an existing ledger it
preserves any hand-set status unless the underlying structural facts have
materially changed (e.g. AOT path disappeared) — in which case the status is
demoted to `unverified` and a `notes` annotation explains why.

Sources scraped:
  - lib/frontend/parser.cpp                special-form name → ESHKOL_*_OP table
  - lib/backend/*_codegen.cpp              `case ESHKOL_*_OP:` for AOT entries
  - lib/backend/llvm_codegen.cpp           ditto
  - lib/backend/vm_compiler.c              `OP_NATIVE_CALL, NNN /* name */` mappings
  - lib/backend/vm_parser.c                ditto
  - lib/backend/vm_native.c                `case NNN:` native handlers
  - lib/backend/vm_*.c sub-handlers        any `case NNN:` not in vm_native.c
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------------- #
#  Source-shape regexes
# ----------------------------------------------------------------------------- #

# Parser table: lines like  `if (op == "letrec") return ESHKOL_LETREC_OP;`
PARSER_TABLE_RE = re.compile(
    r'^\s*if\s*\(op\s*==\s*"(?P<name>[^"]+)"\)\s*return\s+(?P<ast_op>ESHKOL_[A-Z0-9_]+_OP)\s*;'
)

# AOT codegen entry: `case ESHKOL_LETREC_OP:` — possibly with body on same line
AOT_CASE_RE = re.compile(r'\bcase\s+(?P<ast_op>ESHKOL_[A-Z0-9_]+_OP)\s*:')

# VM compiler emit:  `chunk_emit(c, OP_NATIVE_CALL, 73); /* append */`
VM_EMIT_RE = re.compile(
    r'\bOP_NATIVE_CALL\s*,\s*(?P<id>\d+)\s*\)\s*;\s*(?:/\*\s*(?P<name>[^*]+?)\s*\*/)?'
)

# VM special-form recognition.  vm_compiler.c handles structural forms
# (letrec, lambda, if, …) by direct AST shape inspection; there is no
# OP_NATIVE_CALL for these.  Two patterns appear:
#     is_sym(head, "letrec")
#     strcmp(head->symbol, "letrec") == 0
VM_SPECIAL_FORM_RE = re.compile(
    r'(?:is_sym\s*\(\s*head\s*,\s*"(?P<n1>[^"]+)"\)|'
    r'strcmp\s*\(\s*head->symbol\s*,\s*"(?P<n2>[^"]+)"\s*\))'
)

# VM native handler:  `case 73: { ... }`  inside vm_*.c
VM_HANDLER_RE = re.compile(r'^\s*case\s+(?P<id>\d+)\s*:')

# WASM-guard: a line containing `#ifndef ESHKOL_VM_WASM` or `#ifdef ESHKOL_VM_WASM`
WASM_GUARD_RE = re.compile(r'^\s*#\s*if(?:n?def|\s+!?defined\b).*\bESHKOL_VM_WASM\b')


# ----------------------------------------------------------------------------- #
#  Repo helpers
# ----------------------------------------------------------------------------- #


def repo_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def git_head(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


# ----------------------------------------------------------------------------- #
#  Scrapers
# ----------------------------------------------------------------------------- #


def scrape_parser_table(parser_cpp: Path) -> dict[str, dict[str, Any]]:
    """Return {ast_op_enum: {name, file, line}}."""
    table: dict[str, dict[str, Any]] = {}
    for lineno, line in enumerate(parser_cpp.read_text().splitlines(), start=1):
        m = PARSER_TABLE_RE.match(line)
        if not m:
            continue
        ast_op = m.group("ast_op")
        # Multiple surface names can map to the same ast_op (include / include-ci).
        # Keep the first occurrence as the canonical entry; record aliases.
        if ast_op not in table:
            table[ast_op] = {
                "name": m.group("name"),
                "file": str(parser_cpp),
                "line": lineno,
                "aliases": [],
            }
        else:
            table[ast_op]["aliases"].append(m.group("name"))
    return table


def scrape_aot_cases(codegen_files: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Return {ast_op_enum: [{file, line}, …]}.  One ast_op may appear in
    multiple files (e.g. ESHKOL_LETREC_OP in both binding_codegen.cpp's
    letrec() and llvm_codegen.cpp's dispatcher)."""
    cases: dict[str, list[dict[str, Any]]] = {}
    for cf in codegen_files:
        text = cf.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = AOT_CASE_RE.search(line)
            if not m:
                continue
            ast_op = m.group("ast_op")
            cases.setdefault(ast_op, []).append({"file": str(cf), "line": lineno})
    return cases


def scrape_vm_emits(vm_compiler_files: list[Path]) -> dict[int, dict[str, Any]]:
    """Return {native_id: {name, file, line}}.  Captures the inline `/* name */`
    comments that vm_compiler.c attaches to each OP_NATIVE_CALL emit."""
    emits: dict[int, dict[str, Any]] = {}
    for vf in vm_compiler_files:
        for lineno, line in enumerate(vf.read_text(errors="replace").splitlines(), start=1):
            m = VM_EMIT_RE.search(line)
            if not m:
                continue
            nid = int(m.group("id"))
            raw = (m.group("name") or "").strip()
            # The /* … */ trailer is a free-form note; we want only the
            # leading identifier (e.g. "apply: takes f and args-list…" → "apply",
            # "open_upvalues(closure, count, base_slot)" → "open_upvalues").
            # Split on the first non-identifier character.
            m_ident = re.match(r"[A-Za-z_][A-Za-z0-9_!?\-+*/<>=]*", raw)
            name = m_ident.group(0) if m_ident else (raw or None)
            # First emit wins; subsequent ones are usually duplicates.
            emits.setdefault(nid, {"name": name, "file": str(vf), "line": lineno})
    return emits


def scrape_vm_special_forms(vm_compiler_files: list[Path]) -> dict[str, dict[str, Any]]:
    """Return {form_name: {file, line}} for special forms that vm_compiler.c
    recognises by direct AST-shape match (no OP_NATIVE_CALL emit)."""
    found: dict[str, dict[str, Any]] = {}
    for vf in vm_compiler_files:
        for lineno, line in enumerate(vf.read_text(errors="replace").splitlines(), start=1):
            for m in VM_SPECIAL_FORM_RE.finditer(line):
                name = m.group("n1") or m.group("n2")
                if not name:
                    continue
                # First sighting wins; later usages are usually re-checks.
                found.setdefault(name, {"file": str(vf), "line": lineno})
    return found


def scrape_vm_handlers(vm_files: list[Path]) -> dict[int, dict[str, Any]]:
    """Return {native_id: {file, line, wasm_guarded}}."""
    handlers: dict[int, dict[str, Any]] = {}
    for vf in vm_files:
        text = vf.read_text(errors="replace")
        lines = text.splitlines()
        wasm_depth = 0
        for lineno, line in enumerate(lines, start=1):
            if WASM_GUARD_RE.match(line):
                wasm_depth += 1
            elif re.match(r"^\s*#\s*endif\b", line) and wasm_depth > 0:
                wasm_depth -= 1
            m = VM_HANDLER_RE.match(line)
            if not m:
                continue
            nid = int(m.group("id"))
            handlers.setdefault(
                nid,
                {
                    "file": str(vf),
                    "line": lineno,
                    "wasm_guarded": wasm_depth > 0,
                },
            )
    return handlers


# ----------------------------------------------------------------------------- #
#  Entry assembly
# ----------------------------------------------------------------------------- #


def _to_relative(rec: dict[str, Any], root: Path) -> dict[str, Any]:
    out = dict(rec)
    if "file" in out:
        try:
            out["file"] = repo_relative(Path(out["file"]), root)
        except ValueError:
            # Path was already relative or outside repo — leave as-is.
            pass
    return out


def assemble_special_forms(
    parser_table: dict[str, dict[str, Any]],
    aot_cases: dict[str, list[dict[str, Any]]],
    vm_emits_by_name: dict[str, dict[str, Any]],
    vm_handlers: dict[int, dict[str, Any]],
    vm_structural_forms: dict[str, dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    forms: list[dict[str, Any]] = []
    for ast_op, prec in sorted(parser_table.items()):
        aot_entries = [_to_relative(c, repo_root) for c in aot_cases.get(ast_op, [])]
        emit = vm_emits_by_name.get(prec["name"])
        vm_block: dict[str, Any] | None = None
        if emit is not None:
            handler = vm_handlers.get(emit.get("native_id"))
            vm_block = {
                "native_id": emit.get("native_id"),
                "compiler": _to_relative({"file": emit["file"], "line": emit["line"]}, repo_root),
                "handler": _to_relative(handler, repo_root) if handler else None,
                "wasm_guarded": handler["wasm_guarded"] if handler else False,
            }
        else:
            # Structural recognition path — vm_compiler.c handles letrec/lambda/if/…
            # by direct AST-shape match, no OP_NATIVE_CALL emitted.  Match on
            # canonical name and on aliases (e.g. "include-ci" → ESHKOL_INCLUDE_OP).
            candidate_names = [prec["name"]] + list(prec.get("aliases", []))
            for candidate in candidate_names:
                struct = vm_structural_forms.get(candidate)
                if struct is not None:
                    vm_block = {
                        "native_id": None,
                        "compiler": _to_relative(struct, repo_root),
                        "handler": None,
                        "wasm_guarded": False,
                    }
                    break
        forms.append(
            {
                "name": prec["name"],
                "ast_op": ast_op,
                "parser": _to_relative(
                    {"file": prec["file"], "line": prec["line"]}, repo_root
                ),
                "aot": aot_entries,
                "vm": vm_block,
                "tests": [],
                "memory_refs": [],
                "parity": "unverified",
                "notes": "",
            }
        )
    return forms


def assemble_builtins(
    vm_emits: dict[int, dict[str, Any]],
    vm_handlers: dict[int, dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Surface every named OP_NATIVE_CALL as a builtin entry. Ones whose name
    we couldn't extract from a `/* … */` comment land with name='<id-NNN>'
    so the human reviewer can fill them in."""
    builtins: list[dict[str, Any]] = []
    for nid in sorted(vm_emits):
        emit = vm_emits[nid]
        handler = vm_handlers.get(nid)
        name = emit.get("name") or f"<id-{nid}>"
        builtins.append(
            {
                "name": name,
                "category": "misc",
                "aot": [],   # populated by hand for known builtins; no good auto-source
                "runtime": [],
                "vm": {
                    "native_id": nid,
                    "compiler": _to_relative(
                        {"file": emit["file"], "line": emit["line"]}, repo_root
                    ),
                    "handler": _to_relative(handler, repo_root) if handler else None,
                    "wasm_guarded": handler["wasm_guarded"] if handler else False,
                },
                "tests": [],
                "memory_refs": [],
                "parity": "unverified",
                "notes": "",
            }
        )
    # Plus: handlers that are never emitted from vm_compiler.c — these are
    # internal helpers (e.g. AD-internal natives invoked via vm_dispatch_native
    # from another native).  Track them separately so reviewers know they're
    # not user-callable.
    unemitted_ids = sorted(set(vm_handlers) - set(vm_emits))
    for nid in unemitted_ids:
        h = vm_handlers[nid]
        builtins.append(
            {
                "name": f"<internal-id-{nid}>",
                "category": "misc",
                "aot": [],
                "runtime": [],
                "vm": {
                    "native_id": nid,
                    "compiler": None,
                    "handler": _to_relative(h, repo_root),
                    "wasm_guarded": h["wasm_guarded"],
                },
                "tests": [],
                "memory_refs": [],
                "parity": "unverified",
                "notes": "Internal native — invoked via vm_dispatch_native from another native, not directly emitted by the compiler.",
            }
        )
    return builtins


# ----------------------------------------------------------------------------- #
#  Merge with prior ledger
# ----------------------------------------------------------------------------- #


def _key_special(entry: dict[str, Any]) -> str:
    return entry["ast_op"]


def _key_builtin(entry: dict[str, Any]) -> str:
    # name + native_id is the stable key (name alone may collide on '<id-NNN>')
    nid = entry.get("vm", {}).get("native_id") if entry.get("vm") else None
    return f"{entry['name']}#{nid}"


def merge(
    new_forms: list[dict[str, Any]],
    new_builtins: list[dict[str, Any]],
    prior: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Carry forward hand-set statuses, tests, memory_refs, notes from prior
    ledger.  If a structural key (file/line) has shifted but the entry exists,
    keep the status.  If an AOT path *disappeared* in the new scrape, demote
    status to 'unverified' with a note."""
    if not prior:
        return new_forms, new_builtins

    prior_forms = {_key_special(e): e for e in prior.get("special_forms", [])}
    prior_builtins = {_key_builtin(e): e for e in prior.get("builtins", [])}

    def _carry(new_entry: dict[str, Any], old_entry: dict[str, Any], aot_field: str) -> None:
        # Preserve human-set fields
        for f in ("tests", "memory_refs", "notes", "category"):
            if old_entry.get(f) and not new_entry.get(f):
                new_entry[f] = old_entry[f]
        old_status = old_entry.get("parity", "unverified")
        new_aot_paths = {(c["file"], c.get("line")) for c in new_entry.get(aot_field, [])}
        old_aot_paths = {(c["file"], c.get("line")) for c in old_entry.get(aot_field, [])}
        if old_status != "unverified" and not new_aot_paths and old_aot_paths:
            new_entry["parity"] = "unverified"
            note = f"Auto-demoted: AOT path(s) {sorted(old_aot_paths)} no longer match. Re-audit."
            new_entry["notes"] = (
                (old_entry.get("notes", "") + " | " if old_entry.get("notes") else "") + note
            )
        else:
            new_entry["parity"] = old_status

    for entry in new_forms:
        old = prior_forms.get(_key_special(entry))
        if old:
            _carry(entry, old, "aot")

    for entry in new_builtins:
        old = prior_builtins.get(_key_builtin(entry))
        if old:
            _carry(entry, old, "aot")

    return new_forms, new_builtins


# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path, default=Path("tools/icc_extras/parity_ledger.json"))
    ap.add_argument(
        "--no-merge",
        action="store_true",
        help="Discard existing ledger statuses, regenerate from scratch.",
    )
    ap.add_argument(
        "--missing-only",
        action="store_true",
        help="Print AOT-only and VM-only entries to stdout; do not write the ledger.",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    if not (root / "lib" / "frontend" / "parser.cpp").exists():
        print(f"error: --repo-root={root} does not look like the eshkol tree", file=sys.stderr)
        return 2

    parser_cpp = root / "lib" / "frontend" / "parser.cpp"
    codegen_files = sorted((root / "lib" / "backend").glob("*_codegen.cpp"))
    codegen_files += [root / "lib" / "backend" / "llvm_codegen.cpp"]
    # Dedupe — llvm_codegen.cpp matches the *_codegen.cpp glob too.
    codegen_files = sorted({p.resolve() for p in codegen_files if p.exists()})

    vm_compiler_files = [
        root / "lib" / "backend" / "vm_compiler.c",
        root / "lib" / "backend" / "vm_parser.c",
    ]
    vm_compiler_files = [p for p in vm_compiler_files if p.exists()]

    vm_handler_files = sorted((root / "lib" / "backend").glob("vm_*.c"))

    parser_table = scrape_parser_table(parser_cpp)
    aot_cases = scrape_aot_cases(codegen_files)
    vm_emits = scrape_vm_emits(vm_compiler_files)
    vm_handlers = scrape_vm_handlers(vm_handler_files)
    vm_structural = scrape_vm_special_forms(vm_compiler_files)

    # Index emits by name for special-form lookup.  Special forms emit with
    # their surface name in the trailing comment, so this works.
    vm_emits_by_name: dict[str, dict[str, Any]] = {}
    for nid, emit in vm_emits.items():
        if emit.get("name"):
            vm_emits_by_name.setdefault(emit["name"], {**emit, "native_id": nid})

    new_forms = assemble_special_forms(
        parser_table, aot_cases, vm_emits_by_name, vm_handlers, vm_structural, root
    )
    new_builtins = assemble_builtins(vm_emits, vm_handlers, root)

    prior: dict[str, Any] | None = None
    if not args.no_merge and args.out.exists():
        try:
            prior = json.loads(args.out.read_text())
        except json.JSONDecodeError:
            print(f"warning: existing {args.out} is not valid JSON; ignoring", file=sys.stderr)

    new_forms, new_builtins = merge(new_forms, new_builtins, prior)

    # AOT-only / VM-only quick report
    aot_only_forms = [
        e["ast_op"] for e in new_forms if e["aot"] and e["vm"] is None
    ]
    vm_only_forms = [
        e["ast_op"] for e in new_forms if not e["aot"] and e["vm"] is not None
    ]
    vm_only_builtins = [
        e["name"] for e in new_builtins if not e["aot"] and e["vm"] is not None and not e["name"].startswith("<")
    ]

    if args.missing_only:
        print("AOT-only special forms (not in VM):")
        for n in sorted(aot_only_forms):
            print(f"  {n}")
        print("\nVM-only special forms (no AOT case):")
        for n in sorted(vm_only_forms):
            print(f"  {n}")
        print(f"\nTotal special forms: {len(new_forms)}  AOT-only: {len(aot_only_forms)}  VM-only: {len(vm_only_forms)}")
        print(f"Total builtins: {len(new_builtins)}  VM-only named: {len(vm_only_builtins)}")
        return 0

    out = {
        "schema_version": 1,
        "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_commit": git_head(root),
        "special_forms": new_forms,
        "builtins": new_builtins,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    print(
        f"wrote {args.out} — {len(new_forms)} special forms, {len(new_builtins)} builtins "
        f"(AOT-only forms: {len(aot_only_forms)}, VM-only forms: {len(vm_only_forms)}, "
        f"VM-only builtins: {len(vm_only_builtins)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
