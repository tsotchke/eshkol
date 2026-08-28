const fs = require('fs');
const vm = require('vm');

const geometry = [
    1, 4, 8, 4, 8, 0, 1, 2, 4,
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    16, 8, 0, 1, 2, 8, 4,
];

for (const [path, className] of [
    ['web/eshkol-repl.js', 'EshkolRepl'],
    ['site/static/eshkol-runtime.js', 'EshkolRuntime'],
]) {
    const source = fs.readFileSync(path, 'utf8') + `\nthis.__TestClass = ${className};`;
    const context = {
        console, WebAssembly, TextEncoder, TextDecoder, Map, Set, Uint8Array, DataView,
        document: { body: {} }, window: {}, globalThis: null,
    };
    context.globalThis = context;
    vm.runInNewContext(source, context, { filename: path });

    const instance = new context.__TestClass();
    const check = instance.createImports().env.eshkol_wasm_abi_check;
    check(...geometry);

    let fired = false;
    try {
        check(...geometry.slice(0, 2), 9, ...geometry.slice(3));
    } catch (error) {
        fired = /object ABI mismatch/.test(error.message) &&
                /objectHeaderSize/.test(error.message);
    }
    if (!fired) throw new Error(`${path}: wrong geometry did not throw`);
    console.log(`${path}: geometry guard PASS`);
}
