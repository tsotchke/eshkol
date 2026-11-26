"""Inspect candidate shape writers and replay the native and VM view witness."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIRS = ("lib", "inc", "exe", "bindings", "web")
SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".inc", ".def", ".cu",
            ".cuh", ".m", ".mm", ".js"}
SHIPPED = ("site/static/eshkol-runtime.js", "site/static/eshkol-webgpu.js",
           "site/static/eshkol-vm.js", "site/static/eshkol-vm.wasm")
# Preserve offsets for line numbers. These are lexical review aids, not C++ alias analysis.
NON_CODE = re.compile(r"""//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'""")
ASSIGN = r"\s*(?:=(?!=)|[+*/%&|^-]=|<<=|>>=|\+\+|--)"
SUBSCRIPT = r"(?:\s*\[[^\];]+\])?"
MEMBER = r"(?:\.|->)\s*"
NATIVE = r"(?:num_dimensions|total_elements|dimensions)"
VM = r"(?:n_dims|shape|inline_shape|strides|inline_strides|total)"
RULES = {
    "native-field": re.compile(MEMBER + NATIVE + r"\b" + SUBSCRIPT + ASSIGN),
    "native-bulk-destination": re.compile(r"\b(?:memcpy|memmove|memset)\s*\([^,;]*?" + MEMBER + NATIVE + r"\b[^,;]*,"),
    "vm-field": re.compile(MEMBER + VM + r"\b" + SUBSCRIPT + ASSIGN),
    "vm-bulk-destination": re.compile(r"\b(?:memcpy|memmove|memset)\s*\([^,;]*?" + MEMBER + VM + r"\b[^,;]*,"),
    "runtime-axis-buffer": re.compile(r"\b(?:dims|gdims)\s*\[[^\];]+\]" + ASSIGN),
    "llvm-write-candidate": re.compile(r"\bCreate(?:Store|MemCpy|MemMove|MemSet)\s*\("),
    "wasm-host-write-candidate": re.compile(r"\.set(?:Big)?(?:U?Int|Uint|Float)(?:8|16|32|64)\s*\(", re.I),
}


def inventory(root, listing):
    files = sorted({p for folder in SOURCE_DIRS for p in (root / folder).rglob("*")
                    if p.is_file() and p.suffix in SUFFIXES} | {root / name for name in SHIPPED})
    fingerprint = hashlib.sha256()
    counts = Counter()
    for path in files:
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        fingerprint.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
        if name in SHIPPED[2:]:
            continue
        src = data.decode("utf-8")
        code = NON_CODE.sub(lambda m: " " * len(m[0]), src)
        starts = [m.end() for m in re.finditer("\n", src)]
        for kind, rule in RULES.items():
            if kind.startswith("vm-") and not path.name.startswith("vm_"):
                continue
            if kind == "runtime-axis-buffer" and name != "lib/core/system_builtins.c":
                continue
            if (kind == "wasm-host-write-candidate") != (path.suffix == ".js"):
                continue
            for match in rule.finditer(code):
                counts[kind] += 1
                if listing:
                    line = bisect_right(starts, match.start()) + 1
                    end = src.find("\n", match.start())
                    print(f"{name}:{line} {kind} {src[match.start():end].strip()[:130]}")
    return {"files": len(files), "sha256": fingerprint.hexdigest(), "lexical_counts": dict(sorted(counts.items()))}



def check_shape_transfer(ir):
    # Follow SSA use/def chains inside the witness only. Field 0 is the axis
    # pointer and field 2 is the element buffer in this native tensor layout.
    function = re.search(r"^define [^\n]+ @shape-owner\([^\n]+\{\n(.*?)^}", ir, re.M | re.S)
    if not function:
        raise ValueError("Missing shape-owner function")
    body = function[1]
    ssa = r"(%[-\w.]+)"

    def unique(pattern):
        hits = re.findall("^  " + pattern, body, re.M)
        if len(hits) != 1:
            raise ValueError(f"Expected one IR match, got {len(hits)} for {pattern}")
        return re.escape(hits[0])

    slot = unique(r"store %eshkol_tagged_value %x, ptr " + ssa + ",")
    operand = unique(ssa + r" = call ptr @eshkol_tensor_operand_carrier_checked\(ptr " + slot + ",")
    header = unique(ssa + r" = call ptr @arena_allocate_tensor_with_header\(")
    elements = unique(ssa + r" = call ptr @arena_allocate\(")

    def field(base, index):
        return unique(ssa + r" = getelementptr [^\n]*%tensor, ptr " + base + ", i32 0, i32 " + str(index) + "$")

    axes = unique(ssa + r" = load ptr, ptr " + field(operand, 0) + ",")
    for index, pointer in ((0, axes), (2, elements)):
        destination = field(header, index)
        stores = re.findall(r"^  store ptr " + ssa + ", ptr " + destination + ",", body, re.M)
        if len(stores) != 1 or re.escape(stores[0]) != pointer:
            raise ValueError(f"Unexpected pointer stored in tensor-apply result field {index}")


def replay(compiler, root):
    witness = HERE / "shape-view.esk"
    expected = "((1 3) (3 2) (2 3) #t #t (3 2) #t)"
    js = """
const fs = require('node:fs');
const [modulePath, specimen] = process.argv.slice(1);
(async () => {
  const lines = [];
  const vm = await require(modulePath)({print: s => lines.push(s), printErr: s => {throw Error(s)}});
  vm.ccall('repl_init', null, [], []);
  lines.length = 0;
  const value = vm.cwrap('repl_eval', 'string', ['string'])('(begin\\n' +
      fs.readFileSync(specimen, 'utf8') + '\\n)');
  if (value) throw Error(value);
  process.stdout.write(lines.join('\\n') + '\\n');
})().catch(e => {console.error(e); process.exitCode = 1});
"""
    with tempfile.TemporaryDirectory() as work:
        def observe(label, cmd):
            result = subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=True, timeout=180)
            if result.stdout.rstrip("\n") != expected or result.stderr:
                raise ValueError(f"{label}: unexpected output {result.stdout!r}, {result.stderr!r}")
            print(f"{label} {expected}")

        observe("vm", ["node", "-e", js, root / SHIPPED[2], witness])
        for opt in ("-O0", "-O2"):
            binary = Path(work) / ("view" + opt)
            options = ["--dump-ir"] if opt == "-O0" else []
            subprocess.run([compiler, *options, "-I", root / "lib", "-O", opt[-1], witness, "-o", binary], cwd=work, check=True, timeout=180)
            observe(opt, [binary])
            if opt == "-O0":
                check_shape_transfer(binary.with_suffix(".ll").read_text())
                print("tensor-apply -O0: new header shares input axes; separately allocated elements")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show candidate locations, including non-shape writes")
    parser.add_argument("--compiler", type=Path, help="also run the view witness on the shipped VM and this native compiler")
    args = parser.parse_args()
    expected = json.loads((HERE / "shape-writers.json").read_text())["snapshot"]
    actual = inventory(ROOT, args.list)
    if actual != expected:
        sys.exit("Source inventory changed; review required. Observed " + json.dumps(actual, sort_keys=True))
    print(f"Source inventory matched ({actual['files']} files). Descriptor immutability remains open.")
    if args.compiler:
        replay(args.compiler.resolve(strict=True), ROOT)


if __name__ == "__main__":
    main()
