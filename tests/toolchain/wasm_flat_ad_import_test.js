#!/usr/bin/env node
// Instantiate a real WASM module against the generated flat-AD import block.
// This checks callable linkage as well as the deliberate unsupported-path throw.
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const files = [
    path.join(root, 'web/eshkol-repl.js'),
    path.join(root, 'site/static/eshkol-runtime.js'),
];
const begin = '// BEGIN GENERATED FLAT-AD IMPORTS';
const end = '// END GENERATED FLAT-AD IMPORTS';
const blocks = files.map((file) => {
    const text = fs.readFileSync(file, 'utf8');
    const start = text.indexOf(begin);
    const stop = text.indexOf(end, start);
    assert.notEqual(start, -1, `${file} has a generated block`);
    assert.notEqual(stop, -1, `${file} has a generated block end`);
    assert.equal(text.indexOf(begin, start + begin.length), -1, `${file} has one generated block`);
    return text.slice(start + begin.length, stop).trim();
});
assert.equal(blocks[0], blocks[1], 'both bundles embed the same flat-AD imports');

const env = Function(`return ({${blocks[0]}});`)();
assert.equal(env.eshkol_ad_tower_carry_result(), 0);
assert.equal(env.eshkol_ad_jet_extract_tower(), 0);
assert.equal(env.eshkol_ad_tower_enter(), undefined);
assert.equal(env.eshkol_ad_tower_leave(), undefined);
assert.equal(typeof env.eshkol_double_to_exact_tagged, 'function');
assert.throws(
    () => env.eshkol_double_to_exact_tagged(0, 0.1, 0),
    /eshkol_double_to_exact_tagged is unavailable in the browser LLVM\/WASM host glue/,
);

function uleb(value) {
    const result = [];
    do {
        let byte = value & 0x7f;
        value >>>= 7;
        if (value) byte |= 0x80;
        result.push(byte);
    } while (value);
    return result;
}
function wasmString(value) {
    const bytes = [...Buffer.from(value, 'utf8')];
    return [...uleb(bytes.length), ...bytes];
}
function section(id, payload) {
    return [id, ...uleb(payload.length), ...payload];
}

// Types: () -> i32 for extraction imports, () -> () for void hooks, and
// (i32, f64, i32) -> () for exact conversion's arena/double/out ABI.
const typeSection = [
    ...uleb(3),
    0x60, 0x00, 0x01, 0x7f,
    0x60, 0x00, 0x00,
    0x60, 0x03, 0x7f, 0x7c, 0x7f, 0x00,
];
const importEntries = [
    ['eshkol_ad_tower_carry_result', 0],
    ['eshkol_ad_jet_extract_tower', 0],
    ['eshkol_ad_nested_capture_unsupported', 1],
    ['eshkol_ad_tower_enter', 1],
    ['eshkol_ad_tower_leave', 1],
    ['eshkol_double_to_exact_tagged', 2],
].flatMap(([name, type]) => [
    ...wasmString('env'), ...wasmString(name), 0x00, ...uleb(type),
]);
const importSection = [...uleb(6), ...importEntries];
const functionSection = [...uleb(2), ...uleb(1), ...uleb(1)];
const exportSection = [
    ...uleb(2),
    ...wasmString('invokeUnsupported'), 0x00, ...uleb(6),
    ...wasmString('invokeExactUnavailable'), 0x00, ...uleb(7),
];
const body = [0x00, 0x10, 0x02, 0x0b]; // no locals; call imported function 2; end
const exactBody = [
    0x00,             // no locals
    0x41, 0x00,       // arena = 0
    0x44, ...Array(8).fill(0), // double = 0.0
    0x41, 0x00,       // out = 0
    0x10, 0x05,       // call imported function 5
    0x0b,
];
const codeSection = [
    ...uleb(2), ...uleb(body.length), ...body,
    ...uleb(exactBody.length), ...exactBody,
];
const wasm = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...section(1, typeSection),
    ...section(2, importSection),
    ...section(3, functionSection),
    ...section(7, exportSection),
    ...section(10, codeSection),
]);

(async () => {
    const { instance } = await WebAssembly.instantiate(wasm, { env });
    assert.throws(
        () => instance.exports.invokeUnsupported(),
        /Nested autodiff through captured values is unsupported/,
    );
    assert.throws(
        () => instance.exports.invokeExactUnavailable(),
        /eshkol_double_to_exact_tagged is unavailable in the browser LLVM\/WASM host glue/,
    );
    process.stdout.write('OK — flat-AD WASM imports link and unsupported numeric conversions throw.\n');
})().catch((error) => {
    process.stderr.write(`${error.stack || error}\n`);
    process.exitCode = 1;
});
