#!/usr/bin/env python3
"""Real Chrome value controls of both production LLVM/WASM host bundles.

Compile the checked-in fixture with the supplied compiler, including automatic
stdlib initialization. Python Fraction.from_float and struct provide independent
exact fraction/IEEE bit oracles. Browser errors and unexpected outputs fail.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import os
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    wasm = evidence / "exact-values.wasm"
    source = ROOT / "tests/toolchain/wasm_exact_browser.esk"
    command = [str(args.compiler.resolve()), "--wasm", str(source), "-o", str(wasm)]
    with (evidence / "fixture-compile.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    negative_source = evidence / "complex-refusal.esk"
    negative_source.write_text("(display (expt (sqrt -1) 2))\n")
    negative_wasm = evidence / "complex-refusal.wasm"
    with (evidence / "complex-refusal-compile.log").open("w") as log:
        subprocess.run([str(args.compiler.resolve()), "--wasm", str(negative_source), "-o", str(negative_wasm)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    f = Fraction.from_float
    inputs = [0., -0., .1, -.1, 2.**63, -(2.**63), float.fromhex("0x1.fffffffffffffp1023"),
              1e300, 1e-300, float.fromhex("0x0.0000000000001p-1022"), -float.fromhex("0x0.0000000000001p-1022")]
    exact = list(map(f, inputs)) + [Fraction(f(.1).numerator), Fraction(f(.1).denominator),
            f(.1)+f(.2), f(.1)/f(.2), f(inputs[9])**2, Fraction(2**189)] + [Fraction(1)]*4
    expected = [{"text": str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}",
                 "type": 1 if v.denominator == 1 and -(2**63) <= v.numerator < 2**63 else 8} for v in exact]
    expected += [{"bits": struct.pack("<d", v).hex(), "type": 2} for v in [inputs[9], 1e-300, inputs[6], .1+.2]]
    expected += [{"text": str(n), "type": 1} for n in [-4,-3,-3,-4,2,4]]
    expected += [{"text": str(int(f(1e300))//2), "type": 8},
                 {"bits": struct.pack("<d", .1).hex(), "type": 2}, {"text": "1", "type": 1},
                 {"text": "2/3", "type": 8}, {"bits": struct.pack("<d", 2**.5).hex(), "type": 2},
                 {"text": "4/9", "type": 8}, {"text": "9/4", "type": 8},
                 {"bits": struct.pack("<d", 2**.5).hex(), "type": 2}]
    expected += [{"type":2,"text":text,"bits":struct.pack("<d",value).hex()}
                 for value,text in [(0.,"0"),(-0.,"-0.0"),(float('inf'),"+inf.0"),(-float('inf'),"-inf.0")]]
    expected += [{"type":2,"text":"+nan.0"}]
    expected += [{"type":2,"text":text,"bits":struct.pack("<d",value).hex()}
                 for value,text in [(float('inf'),"+inf.0"),(-float('inf'),"-inf.0"),(0.,"0"),(-0.,"-0.0")]]
    expected_strings = ["0","-0.0","+inf.0","-inf.0","+nan.0","0.1","5e-324",
                        "1.7976931348623157e+308","+inf.0","-inf.0","-0.0"]
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        for bundle, class_name in [("web/eshkol-repl.js", "EshkolRepl"), ("site/static/eshkol-runtime.js", "EshkolRuntime")]:
            page = browser.new_page()
            page.set_content("<!doctype html><body></body>")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            page.add_script_tag(path=str(ROOT / bundle))
            actual = page.evaluate("""async ({bytes, className}) => {
                const rt = className === 'EshkolRepl' ? new EshkolRepl() : new EshkolRuntime();
                const wasm = new Uint8Array(bytes); const layout = rt.prepareWasm(wasm);
                const imports = rt.createImports(); const observations = [];
                const strings = []; let formatCalls = 0;
                imports.env.exact_observe_string = p => strings.push(rt.readString(p));
                const formatter = imports.env.eshkol_format_double;
                imports.env.eshkol_format_double = (...args) => { formatCalls++; return formatter(...args); };
                let complexCalls = 0;
                for (const key of ['eshkol_complex_pow','eshkol_complex_sqrt']) {
                    const complexImport = imports.env[key];
                    imports.env[key] = (...args) => { complexCalls++; return complexImport(...args); };
                }
                const original = imports.env.eshkol_display_value;
                const printed = []; const log = console.log;
                console.log = (...values) => { printed.push(values.join(' ')); log(...values); };
                imports.env.eshkol_display_value = p => {
                    const memory = rt.memory || rt._importedMemory;
                    const dv = new DataView(memory.buffer); const type = dv.getUint8(p) & 15;
                    original(p);
                    const item = { type, text: printed.pop() };
                    if (type === 2) item.bits = Array.from(new Uint8Array(memory.buffer, p + 8, 8)).map(n=>n.toString(16).padStart(2,'0')).join('');
                    observations.push(item);
                };
                const {instance} = await WebAssembly.instantiate(wasm, imports);
                if (rt.setInstance) rt.setInstance(instance);
                instance.exports.main(0,0);
                if (complexCalls !== 0) throw Error('positive real controls reached complex refusal');
                const scalar = rt._bump(16), ok = rt._bump(8);
                imports.env.eshkol_double_to_exact_tagged(1,Number.MIN_VALUE,scalar);
                if (imports.env.eshkol_ad_seed_to_double(scalar,ok) !== Number.MIN_VALUE) throw Error('heap scalar AD extraction mismatch');
                if (imports.env.eshkol_ad_point_is_exact_scalar(scalar) !== 1) throw Error('heap exact classification mismatch');
                let refused = false;
                try { imports.env.eshkol_taylor_seed_tagged(1,scalar,2,scalar); }
                catch(error) { refused = /Exact Taylor differentiation is unsupported/.test(error.message); }
                if (!refused) throw Error('unsupported exact tower silently returned');
                return {observations,strings,formatCalls,layout};
            }""", {"bytes": list(wasm.read_bytes()), "className": class_name})
            if errors:
                raise AssertionError(f"{bundle}: browser errors: {errors}")
            if len(actual["observations"]) != len(expected):
                raise AssertionError(f"{bundle}: expected {len(expected)} values, got {actual}")
            for index, (got, wanted) in enumerate(zip(actual["observations"], expected)):
                if any(got.get(key) != value for key, value in wanted.items()):
                    raise AssertionError(f"{bundle} value {index}: {got} != {wanted}")
            if actual["strings"] != expected_strings or actual["formatCalls"] != len(expected_strings):
                raise AssertionError(f"{bundle}: compiled number->string did not use working native formatter: {actual}")
            results.append({"bundle": bundle, "sha256": hashlib.sha256((ROOT/bundle).read_bytes()).hexdigest(), **actual})
            refused = page.evaluate("""async ({bytes,className}) => {
                const rt = className === 'EshkolRepl' ? new EshkolRepl() : new EshkolRuntime();
                const wasm = new Uint8Array(bytes);rt.prepareWasm(wasm);
                const {instance}=await WebAssembly.instantiate(wasm,rt.createImports());
                if(rt.setInstance)rt.setInstance(instance);
                try {instance.exports.main(0,0);}
                catch(error) {return error.code==='ESH_NUMERIC_UNSUPPORTED' && /Complex exponentiation/.test(error.message);}
                return false;
            }""", {"bytes": list(negative_wasm.read_bytes()), "className": class_name})
            if not refused:
                raise AssertionError(f"{bundle}: compiled complex route did not refuse explicitly")
            print(f"{bundle}: {len(expected)} compiled values + {len(expected_strings)} number->string controls + AD scalar/refusal PASS")
            page.close()
        browser_version = browser.version
        browser.close()
    receipt = {"status": "pass", "browser": browser_version, "playwright": "1.59.0",
               "compiler": str(args.compiler.resolve()), "compiler_sha256": hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
               "compiler_version": subprocess.run([str(args.compiler), "--version"], text=True, capture_output=True, check=True).stdout,
               "compile_command": command, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "wasm_sha256": hashlib.sha256(wasm.read_bytes()).hexdigest(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
               "oracle": "Python Fraction.from_float + struct.pack('<d')", "results": results}
    (evidence / "browser-values.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
