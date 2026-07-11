#!/usr/bin/env python3
"""Regression guard for the WASM stdlib dead-strip.

Compiles a small program that genuinely uses the standard library
(map / filter / fold-left, plus a first-class function value) to WebAssembly
with `eshkol-run --wasm` and WITHOUT `--no-stdlib`, then asserts:

  1. a .wasm was produced and has the WASM magic header;
  2. its size is under a hard bound (default 2 MiB) — comfortably below the
     5 MB GitHub Pages artifact gate;
  3. the number of defined functions is bounded (unused stdlib was stripped);
  4. (if `node` is available) the module passes WebAssembly.compile and
     WebAssembly.instantiate under the same memory/table/global model the site
     glue uses.

Background — the regression this guards against (issue #240 follow-up):
`eshkol-run` auto-injects `(require stdlib)` into any unit not passing
`--no-stdlib`. Before the fix, the `--wasm` path emitted a single unlinked
object with no `wasm-ld --gc-sections` and eagerly address-took every top-level
function from `main()` (homoiconic display registry), pinning the ENTIRE stdlib
against dead-code elimination — a small program ballooned to tens of MB (or, when
a native stdlib.o was auto-linked, to an un-instantiable pile of `env.*` imports).
The fix skips the homoiconic registry for wasm and runs internalize + globalDCE
before emission, so only reachable code survives.

Exit:
    0  wasm produced, under size/function bounds, (optionally) instantiates
    1  a bound was exceeded, or the module failed to compile/instantiate
    2  toolchain could not produce a .wasm (neutral / environment failure)
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Uses map / filter / fold-left (stdlib) and passes `inc` as a first-class
# function value to `twice`, so the test also proves used first-class functions
# survive the dead-strip.
STDLIB_PROGRAM = """
(define (square x) (* x x))
(define (inc x) (+ x 1))
(define (twice f x) (f (f x)))
(define nums (list 1 2 3 4 5 6))
(display (map square nums)) (newline)
(display (filter even? nums)) (newline)
(display (fold-left + 0 (map square nums))) (newline)
(display (twice inc 40)) (newline)
"""


def count_wasm_functions(data: bytes) -> int:
    """Return the count field of the WASM Function section (section id 10),
    i.e. the number of module-defined functions. 0 if absent."""
    if data[:4] != b"\x00asm":
        raise ValueError("not a WASM module (bad magic)")
    off = 8

    def uleb(o: int) -> tuple[int, int]:
        result = shift = 0
        while True:
            b = data[o]
            o += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result, o
            shift += 7

    while off < len(data):
        section_id = data[off]
        off += 1
        section_size, off = uleb(off)
        if section_id == 3:  # Function section
            count, _ = uleb(off)
            return count
        off += section_size
    return 0


def compile_wasm(eshkol_run: Path, source: str, timeout: int) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "sizecheck.esk"
        out_path = Path(tmp) / "sizecheck.wasm"
        src_path.write_text(source)
        try:
            res = subprocess.run(
                # Deliberately NO --no-stdlib: this must exercise the stdlib path.
                [str(eshkol_run), "--wasm", str(src_path), "-o", str(out_path)],
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"warning: wasm compile timed out after {timeout}s", file=sys.stderr)
            return None
        except FileNotFoundError:
            print(f"warning: {eshkol_run} not found", file=sys.stderr)
            return None
        if res.returncode != 0 or not out_path.exists():
            stderr = res.stderr.decode("utf-8", errors="replace").strip()
            if stderr:
                print(stderr[-4000:], file=sys.stderr)
            print(f"warning: wasm compile failed (exit {res.returncode})", file=sys.stderr)
            return None
        return out_path.read_bytes()


_NODE_HARNESS = r"""
import fs from 'node:fs';
const bytes = fs.readFileSync(process.argv[2]);
const mod = await WebAssembly.compile(bytes);
const memory = new WebAssembly.Memory({ initial: 256, maximum: 1024 });
const table = new WebAssembly.Table({ initial: 256, element: 'anyfunc' });
const env = {
  __linear_memory: memory,
  __stack_pointer: new WebAssembly.Global({ value: 'i32', mutable: true }, 1048576),
  __indirect_function_table: table,
};
for (const imp of WebAssembly.Module.imports(mod)) {
  if (imp.module !== 'env' || imp.name in env) continue;
  if (imp.kind === 'function') env[imp.name] = (...a) => 0;
  else if (imp.kind === 'global') env[imp.name] = new WebAssembly.Global({ value: 'i32', mutable: false }, 0);
  else if (imp.kind === 'memory') env[imp.name] = memory;
  else if (imp.kind === 'table') env[imp.name] = table;
}
await WebAssembly.instantiate(mod, { env });
console.log('instantiate-ok');
"""


def node_instantiates(wasm_bytes: bytes) -> bool | None:
    """Return True/False if node validated the module, or None if node absent."""
    node = shutil.which("node")
    if not node:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        wasm_path = Path(tmp) / "m.wasm"
        harness = Path(tmp) / "h.mjs"
        wasm_path.write_bytes(wasm_bytes)
        harness.write_text(_NODE_HARNESS)
        res = subprocess.run(
            [node, str(harness), str(wasm_path)],
            capture_output=True, timeout=60,
        )
        ok = res.returncode == 0 and b"instantiate-ok" in res.stdout
        if not ok:
            print(res.stderr.decode("utf-8", errors="replace")[-2000:], file=sys.stderr)
        return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    ap.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024,
                    help="Fail if the wasm is at least this many bytes (default 2 MiB).")
    ap.add_argument("--max-functions", type=int, default=200,
                    help="Fail if the module defines at least this many functions.")
    ap.add_argument("--compile-timeout", type=int, default=300)
    args = ap.parse_args()

    eshkol_run = args.build_dir / "eshkol-run"
    if not eshkol_run.exists():
        print(f"error: {eshkol_run} not found. Build first.", file=sys.stderr)
        return 2

    wasm = compile_wasm(eshkol_run, STDLIB_PROGRAM, args.compile_timeout)
    if wasm is None:
        return 2

    size = len(wasm)
    try:
        nfuncs = count_wasm_functions(wasm)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"stdlib-using --wasm build: {size} bytes, {nfuncs} defined functions")

    failed = False
    if size >= args.max_bytes:
        print(f"FAIL: wasm size {size} >= bound {args.max_bytes} "
              f"— unused stdlib is not being dead-stripped", file=sys.stderr)
        failed = True
    if nfuncs >= args.max_functions:
        print(f"FAIL: {nfuncs} defined functions >= bound {args.max_functions} "
              f"— unused stdlib is not being dead-stripped", file=sys.stderr)
        failed = True

    inst = node_instantiates(wasm)
    if inst is None:
        print("note: node not found — skipped WebAssembly.instantiate check")
    elif inst:
        print("WebAssembly.compile + instantiate: OK")
    else:
        print("FAIL: module did not compile/instantiate in node", file=sys.stderr)
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
