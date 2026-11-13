#!/usr/bin/env node
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const bytes = fs.readFileSync(process.argv[2]);
const expected = ["9", "18", "63", "3", "19", "10", "24", "20", "2080", "2145"];
(async () => {
    for (const [file, name] of [['web/eshkol-repl.js', 'EshkolRepl'],
                               ['site/static/eshkol-runtime.js', 'EshkolRuntime']]) {
        const output = [];
        const context = {console: {...console, log: (...v) => output.push(v.join(' '))},
            WebAssembly, TextEncoder, TextDecoder, Map, Set, Uint8Array, DataView,
            ArrayBuffer, document: {body: {}}, window: {}};
        context.globalThis = context;
        vm.runInNewContext(fs.readFileSync(path.join(root, file), 'utf8') +
                           `\nthis.Runtime=${name};`, context);
        const rt = new context.Runtime();
        rt.prepareWasm(bytes);
        const imports = rt.createImports();
        const counts = new Set();
        const allocateClosure = imports.env.arena_allocate_closure_with_header;
        imports.env.arena_allocate_closure_with_header = (...args) => {
            counts.add(Number(BigInt(args[2]) & 0xffffffffn));
            return allocateClosure(...args);
        };
        const {instance} = await WebAssembly.instantiate(bytes, imports);
        if (rt.setInstance) rt.setInstance(instance);
        instance.exports.main(0, 0);
        assert.deepEqual(output.map(x => x.trim()).filter(Boolean), expected,
            `${file}: fixed, variadic, apply and capture-boundary values`);
        for (const n of [0, 1, 3, 64, 65])
            assert.ok(counts.has(n), `${file}: actual ${n}-capture closure exercised`);
        console.log(`PASS: ${file}: ten compiled WASM closure value checks`);
    }
})().catch(error => {console.error(error); process.exitCode = 1;});
