'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function main() {
    const filename = path.resolve(process.argv[2]);
    const vm = await require(filename)({
        wasmBinary: fs.readFileSync(filename.replace(/\.js$/, '.wasm')),
    });
    const evalPointer = vm.cwrap('repl_eval', 'string', ['number']);
    assert.equal(evalPointer(0), 'ERROR: source is required');
    assert.equal(vm.ccall('repl_eval', 'string', ['string'], ['(+ 1 2)']), '');
    assert.equal(evalPointer(0), 'ERROR: source is required');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
