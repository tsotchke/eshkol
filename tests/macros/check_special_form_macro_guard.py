#!/usr/bin/env python3
"""Keep the shared macro-shadow guard aligned with parser special forms."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[2]
parser = (root / "lib/frontend/parser.cpp").read_text(encoding="utf-8")
mapping = parser[parser.index("static eshkol_op_t get_operator_type"):
                 parser.index("// Forward declarations")]
names = set(re.findall(r'if \(op == "([^"]+)"\)', mapping))
binding = (root / "inc/eshkol/frontend/binding_forms.h").read_text(encoding="utf-8")
names.update(re.findall(r'X\([^,]+,\s*"([^"]+)"', binding))
guard = (root / "inc/eshkol/frontend/macro_binding_guards.h").read_text(encoding="utf-8")
guard_names = set(re.findall(r"\bX\(([^)]+)\)", guard.split("static inline", 1)[0]))

if names != guard_names:
    print("special-form macro guard drift", file=sys.stderr)
    print("missing:", sorted(names - guard_names), file=sys.stderr)
    print("extra:", sorted(guard_names - names), file=sys.stderr)
    sys.exit(1)
print(f"PASS: special-form macro guard matches parser mapping ({len(names)} names)")
