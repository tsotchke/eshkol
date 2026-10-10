#!/usr/bin/env python3
"""Focused browser ABI controls for the retained closure arena."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
MINIMAL_WASM = [0, 97, 115, 109, 1, 0, 0, 0]


def main() -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        for bundle, class_name in [
            ("web/eshkol-repl.js", "EshkolRepl"),
            ("site/static/eshkol-runtime.js", "EshkolRuntime"),
        ]:
            page = browser.new_page()
            page.set_content("<!doctype html><body></body>")
            page.add_script_tag(path=str(ROOT / bundle))
            page.evaluate(
                """({className, wasm}) => {
                    const rt = className === 'EshkolRepl' ? new EshkolRepl() : new EshkolRuntime();
                    const layout = rt.prepareWasm(new Uint8Array(wasm));
                    const env = rt.createImports().env;
                    const memory = rt.memory || rt._importedMemory;
                    const dv = new DataView(memory.buffer);
                    const check = (p, captures, packed, subtype) => {
                        if (!p || (p & 7) !== 0 || p + 40 > memory.buffer.byteLength)
                            throw Error('closure pointer alignment/bounds mismatch');
                        if (dv.getUint8(p - 8) !== subtype || dv.getUint32(p - 4, true) !== 40)
                            throw Error('closure object header mismatch');
                        if (dv.getBigUint64(p, true) !== 0x1234n || dv.getBigUint64(p + 16, true) !== 0x5678n)
                            throw Error('closure scalar fields were not initialized');
                        if (dv.getUint8(p + 32) !== 1 || dv.getUint8(p + 33) !== 2 || dv.getUint32(p + 36, true) !== 0x9a)
                            throw Error('closure type metadata was not initialized');
                        const envPtr = dv.getUint32(p + 8, true);
                        if (captures === 0 && envPtr !== 0) throw Error('zero-capture closure has an environment');
                        if (captures > 0) {
                            if (!envPtr || (envPtr & 7) !== 0 || envPtr + 8 + captures * 16 > memory.buffer.byteLength)
                                throw Error('capture environment alignment/bounds mismatch');
                            if (dv.getBigUint64(envPtr, true) !== packed) throw Error('packed capture metadata mismatch');
                        }
                    };
                    const zero = env.arena_allocate_closure_with_header(1, 0x1234, 0, 0x5678, 0x9a0201, 0);
                    check(zero, 0, 0n, 1);
                    const packed = (3n << 32n) | 2n;
                    const two = env.arena_allocate_closure_with_header(1, 0x1234, packed, 0x5678, 0x9a0201, 0);
                    check(two, 2, packed, 0);
                    if (layout.heapFloor >= zero || layout.heapFloor >= two)
                        throw Error('closure allocation crossed prepared heap floor');
                    if (typeof env.arena_allocate_vector_with_header === 'function' &&
                        className === 'EshkolRepl' && env.arena_allocate_vector_with_header(1, 0) !== 0)
                        throw Error('unsupported vector allocation unexpectedly fabricated an object');
                }""",
                {"className": class_name, "wasm": MINIMAL_WASM},
            )
            page.close()
            print(f"{bundle}: closure header/fields, zero-capture, bounds, alignment PASS")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
