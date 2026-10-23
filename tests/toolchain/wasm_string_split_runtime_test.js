#!/usr/bin/env node
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const bytes = fs.readFileSync(process.argv[2]);
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
        const {instance} = await WebAssembly.instantiate(bytes, rt.createImports());
        if (rt.setInstance) rt.setInstance(instance);
        instance.exports.main(0, 0);
        assert.deepEqual(output.map(x => x.trim()).filter(Boolean), ['1', '1', '1', '1'],
            `${file}: Unicode, NUL, NUL/Unicode, and empty delimiter results`);
        console.log(`PASS: ${file}: four compiled WASM split value checks`);
    }
})().catch(error => {console.error(error); process.exitCode = 1;});
