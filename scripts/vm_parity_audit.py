#!/usr/bin/env python3
"""vm_parity_audit.py — codegen-vs-VM surface parity ratchet (adversarial P5).

Eshkol has two executable surfaces:

  * the NATIVE codegen surface — every builtin name and special-form op the
    LLVM backend (lib/backend/llvm_codegen.cpp) knows how to compile, plus
    the AST op enum (eshkol_op_t in inc/eshkol/eshkol.h);
  * the VM surface — every name the bytecode VM can resolve: the standalone
    driver's BUILTINS[] first-class native table (lib/backend/eshkol_vm.c),
    the VM compiler's inline special-form dispatch (lib/backend/vm_compiler.c
    + lib/backend/vm_parser.c `is_sym(head, ...)` / `strcmp(head->symbol,...)`),
    and the Scheme prelude compiled into every VM
    (lib/backend/vm_prelude_source.h).  vm_native.c / vm_tensor_ops.c /
    vm_logic.c implement numeric fids only — every NAME binding to those fids
    flows through the tables above, so names are the honest VM surface.

The ratchet: every symbol on the codegen surface must be either

  1. present on the VM surface (parity holds), or
  2. covered by an explicit row in tests/vm_parity/PARITY.tsv with status
     `native-only-justified` (a conscious waiver) or `gap` (an acknowledged,
     counted hole).

Any codegen symbol absent from BOTH the VM surface and the manifest fails
the audit (exit 1).  Adding a language feature therefore forces a decision:
teach the VM, or write a justified manifest row.

Also enforced:
  * a manifest row claiming `vm-supported` for a plain builtin name that the
    VM surface does NOT contain fails (stale/false claim);
  * `native-only-justified` and `gap` rows must carry a non-empty
    justification;
  * manifest rows for symbols that vanished from the codegen surface are
    reported as warnings (stale rows; tidy when convenient).

Extraction is regex-on-actual-table-shapes, verified against the tables as
of 2026-07; each extractor asserts a sane minimum count so silent format
drift breaks the audit loudly instead of quietly extracting nothing.

Usage:
  scripts/vm_parity_audit.py                 # audit (exit 0/1)
  scripts/vm_parity_audit.py --dump-codegen  # print codegen surface
  scripts/vm_parity_audit.py --dump-vm       # print VM surface
  scripts/vm_parity_audit.py --seed          # print a fresh TSV skeleton
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "tests", "vm_parity", "PARITY.tsv")

CODEGEN_CPP = os.path.join(REPO, "lib", "backend", "llvm_codegen.cpp")
AST_HEADER = os.path.join(REPO, "inc", "eshkol", "eshkol.h")
VM_DRIVER = os.path.join(REPO, "lib", "backend", "eshkol_vm.c")
VM_COMPILER = os.path.join(REPO, "lib", "backend", "vm_compiler.c")
VM_PARSER = os.path.join(REPO, "lib", "backend", "vm_parser.c")
VM_PRELUDE = os.path.join(REPO, "lib", "backend", "vm_prelude_source.h")

VALID_STATUSES = ("vm-supported", "native-only-justified", "gap")

# Names matched by `func_name == "..."` that are compiler-internal control
# strings, not user-facing builtins.  Keep TIGHT — when in doubt, leave the
# name in and let the manifest carry a row for it.
CODEGEN_NAME_DENYLIST = {
    "main",  # entry-point sentinel comparisons, not a callable builtin
}


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def die(msg):
    sys.stderr.write("vm_parity_audit: %s\n" % msg)
    sys.exit(2)


# ── codegen surface ──────────────────────────────────────────────────────

def extract_op_enum():
    """ESHKOL_*_OP members of eshkol_op_t in inc/eshkol/eshkol.h -> op:NAME."""
    src = read(AST_HEADER)
    m = re.search(
        r"typedef enum \{\s*ESHKOL_INVALID_OP,(.*?)\}\s*eshkol_op_t\s*;",
        src, re.S)
    if not m:
        die("op-enum shape changed in %s (typedef enum { ESHKOL_INVALID_OP "
            "... } eshkol_op_t; not found) — update extract_op_enum()" %
            AST_HEADER)
    names = re.findall(r"\bESHKOL_([A-Z0-9_]+)_OP\b", m.group(1))
    ops = {"op:" + n for n in names}
    if len(ops) < 80:
        die("op-enum extraction found only %d ops (expected 80+) — "
            "format drift?" % len(ops))
    return ops


def extract_codegen_builtins():
    """Builtin names the LLVM backend dispatches on."""
    src = read(CODEGEN_CPP)
    names = set()

    # 1. direct call dispatch:  func_name == "car"
    names.update(re.findall(r'func_name\s*==\s*"([^"]+)"', src))

    # 2. return-type registry:  function_return_types["car"] = ...
    names.update(re.findall(r'function_return_types\["([^"]+)"\]', src))

    # 3. math-builtin first-class sets:
    #    static const std::set<std::string> math_builtins = { "exp", ... };
    for body in re.findall(
            r"std::set<std::string>\s+math_builtins\s*=\s*\{(.*?)\};",
            src, re.S):
        names.update(re.findall(r'"([^"]+)"', body))

    names = {n for n in names if n and n not in CODEGEN_NAME_DENYLIST}
    if len(names) < 300:
        die("codegen builtin extraction found only %d names (expected 300+) "
            "— dispatch shape drift in %s?" % (len(names), CODEGEN_CPP))
    return names


# ── VM surface ───────────────────────────────────────────────────────────

def extract_vm_builtins_table():
    """{"name", fid, arity} rows of BUILTINS[] in eshkol_vm.c."""
    src = read(VM_DRIVER)
    m = re.search(
        r"static const BuiltinDef BUILTINS\[\]\s*=\s*\{(.*?)\n\};",
        src, re.S)
    if not m:
        die("BUILTINS[] table not found in %s — update "
            "extract_vm_builtins_table()" % VM_DRIVER)
    names = set(re.findall(r'\{\s*"([^"]+)"\s*,\s*\d+\s*,\s*\d+\s*\}',
                           m.group(1)))
    if len(names) < 200:
        die("BUILTINS[] extraction found only %d names (expected 200+)" %
            len(names))
    return names


def extract_vm_compiler_dispatch():
    """Special forms / inline builtins the VM compiler matches by name."""
    names = set()
    for path in (VM_COMPILER, VM_PARSER):
        src = read(path)
        names.update(re.findall(r'is_sym\([^,()]*,\s*"([^"]+)"\s*\)', src))
        names.update(re.findall(
            r'strcmp\([^,()]*->symbol,\s*"([^"]+)"\s*\)\s*==\s*0', src))
        names.update(re.findall(
            r'strcmp\([^,()]*symbol,\s*"([^"]+)"\s*\)\s*==\s*0', src))
    # '#t' / '#f' / '.' literals and pure syntax markers are not builtins
    names -= {"#t", "#f", ".", "...", "_", "else", "=>",
              "unquote", "unquote-splicing"}
    if len(names) < 80:
        die("VM compiler dispatch extraction found only %d names "
            "(expected 80+)" % len(names))
    return names


def extract_vm_prelude_defines():
    """(define (name ...) / (define name ...) in the VM Scheme prelude."""
    src = read(VM_PRELUDE)
    # The prelude is a C string: each line is "...\n".  Concatenate literals.
    chunks = re.findall(r'"((?:[^"\\]|\\.)*)"', src)
    text = "".join(chunks).replace("\\n", "\n").replace('\\"', '"')
    names = set(re.findall(r"\(define\s+\(([^\s()]+)", text))
    names.update(re.findall(r"\(define\s+([^\s()]+)\s", text))
    if len(names) < 20:
        die("prelude extraction found only %d defines (expected 20+)" %
            len(names))
    return names


def vm_surface():
    return (extract_vm_builtins_table()
            | extract_vm_compiler_dispatch()
            | extract_vm_prelude_defines())


def codegen_surface():
    return extract_codegen_builtins() | extract_op_enum()


# ── manifest ─────────────────────────────────────────────────────────────

def load_manifest():
    if not os.path.exists(MANIFEST):
        die("manifest missing: %s (run --seed to generate a skeleton)" %
            MANIFEST)
    rows = {}
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                die("%s:%d: expected name<TAB>status[<TAB>justification]" %
                    (MANIFEST, ln))
            name, status = parts[0].strip(), parts[1].strip()
            just = parts[2].strip() if len(parts) > 2 else ""
            if status not in VALID_STATUSES:
                die("%s:%d: bad status %r (want %s)" %
                    (MANIFEST, ln, status, "|".join(VALID_STATUSES)))
            if name in rows:
                die("%s:%d: duplicate row for %r" % (MANIFEST, ln, name))
            rows[name] = (status, just)
    return rows


# ── modes ────────────────────────────────────────────────────────────────

def seed(cg, vm):
    print("# tests/vm_parity/PARITY.tsv — VM parity manifest (seed skeleton)")
    print("# name\tstatus\tjustification")
    for name in sorted(cg):
        status = "vm-supported" if name in vm else "gap"
        print("%s\t%s\t%s" % (name, status,
                              "" if status == "vm-supported"
                              else "TODO: classify"))


def audit(cg, vm):
    manifest = load_manifest()
    failures, warnings = [], []

    unratcheted = sorted(n for n in cg if n not in vm and n not in manifest)
    for n in unratcheted:
        failures.append(
            "RATCHET %s: on the codegen surface but absent from both the VM "
            "surface and tests/vm_parity/PARITY.tsv — add VM support or a "
            "justified manifest row" % n)

    for name, (status, just) in sorted(manifest.items()):
        if status == "vm-supported" and not name.startswith("op:") \
                and name not in vm:
            failures.append(
                "STALE-CLAIM %s: manifest says vm-supported but the VM "
                "surface does not contain it" % name)
        if status in ("native-only-justified", "gap") and not just:
            failures.append(
                "UNJUSTIFIED %s: status %s requires a justification" %
                (name, status))
        if name not in cg:
            warnings.append(
                "stale-row %s: no longer on the codegen surface" % name)

    n_sup = sum(1 for s, _ in manifest.values() if s == "vm-supported")
    n_just = sum(1 for s, _ in manifest.values()
                 if s == "native-only-justified")
    n_gap = sum(1 for s, _ in manifest.values() if s == "gap")
    auto = sorted(n for n in cg if n in vm and n not in manifest)

    print("vm_parity_audit: codegen surface = %d symbols "
          "(%d builtins + %d ops)" %
          (len(cg), len([n for n in cg if not n.startswith("op:")]),
           len([n for n in cg if n.startswith("op:")])))
    print("vm_parity_audit: VM surface      = %d names" % len(vm))
    print("vm_parity_audit: manifest rows   = %d "
          "(vm-supported=%d native-only-justified=%d gap=%d)" %
          (len(manifest), n_sup, n_just, n_gap))
    print("vm_parity_audit: auto-parity (in both surfaces, no manifest row "
          "needed) = %d" % len(auto))

    for w in warnings:
        print("vm_parity_audit: WARN %s" % w)
    for f in failures:
        print("vm_parity_audit: FAIL %s" % f)
    if failures:
        print("vm_parity_audit: %d failure(s) — the parity ratchet is "
              "violated" % len(failures))
        return 1
    print("vm_parity_audit: OK — every codegen symbol is VM-supported or "
          "consciously waived")
    return 0


def main(argv):
    cg, vm = codegen_surface(), vm_surface()
    if "--dump-codegen" in argv:
        for n in sorted(cg):
            print("%s\t%s" % (n, "vm" if n in vm else "-"))
        return 0
    if "--dump-vm" in argv:
        for n in sorted(vm):
            print(n)
        return 0
    if "--seed" in argv:
        seed(cg, vm)
        return 0
    return audit(cg, vm)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
