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

// Types: () -> i32 for the two extraction imports; () -> () for void hooks.
const typeSection = [
    ...uleb(2),
    0x60, 0x00, 0x01, 0x7f,
    0x60, 0x00, 0x00,
];
const importEntries = [
    ['eshkol_ad_tower_carry_result', 0],
    ['eshkol_ad_jet_extract_tower', 0],
    ['eshkol_ad_nested_capture_unsupported', 1],
    ['eshkol_ad_tower_enter', 1],
    ['eshkol_ad_tower_leave', 1],
].flatMap(([name, type]) => [
    ...wasmString('env'), ...wasmString(name), 0x00, ...uleb(type),
]);
const importSection = [...uleb(5), ...importEntries];
const functionSection = [...uleb(1), ...uleb(1)];
const exportSection = [
    ...uleb(1), ...wasmString('invokeUnsupported'), 0x00, ...uleb(5),
];
const body = [0x00, 0x10, 0x02, 0x0b]; // no locals; call imported function 2; end
const codeSection = [...uleb(1), ...uleb(body.length), ...body];
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
    process.stdout.write('OK — flat-AD WASM imports link and unsupported capture throws.\n');
})().catch((error) => {
    process.stderr.write(`${error.stack || error}\n`);
    process.exitCode = 1;
});
