"""Check the class seen by GCC/Clang in each participating translation unit."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

build = Path(sys.argv[1])
units = {"llvm_codegen.cpp", "builtin_factory_codegen.cpp", "module_init_codegen.cpp",
         "repl_resolution_codegen.cpp"}
seen = set()
definitions = set()
for row in json.loads((build / "compile_commands.json").read_text()):
    name = Path(row["file"]).name
    if name not in units:
        continue
    args = row.get("arguments") or shlex.split(row["command"])
    dest = args.index("-o")
    del args[dest:dest + 2]
    args.remove("-c")
    # Identical commands for object/static targets see the same source.
    key = (row["directory"], tuple(args))
    if key in seen:
        continue
    seen.add(key)
    source = subprocess.check_output([*args, "-E", "-P"], cwd=row["directory"])
    matches = list(re.finditer(rb"(?m)^class EshkolLLVMCodeGen\s*\{", source))
    if len(matches) != 1:
        sys.exit(f"{name} contains {len(matches)} definitions")
    end = source.index(b"\n};", matches[0].end()) + 3
    digest = hashlib.sha256(source[matches[0].start():end]).hexdigest()
    print(name, digest)
    definitions.add((name, digest))

if {name for name, _ in definitions} != units or len({digest for _, digest in definitions}) != 1:
    sys.exit("translation units must share the full class definition")
