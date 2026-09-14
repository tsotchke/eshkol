const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

for (const [path, name] of [
    ['web/eshkol-repl.js', 'EshkolRepl'],
    ['site/static/eshkol-runtime.js', 'EshkolRuntime'],
]) {
    const context = { console, WebAssembly, TextEncoder, TextDecoder,
        document: { body: {} }, window: {} };
    vm.runInNewContext(fs.readFileSync(path, 'utf8') + `\nthis.Runtime = ${name};`, context);
    const runtime = new context.Runtime();
    runtime.memory = new WebAssembly.Memory({ initial: 1 });
    const shape = runtime.createImports().env.eshkol_broadcast_shape_f64;
    const view = new DataView(runtime.memory.buffer);
    function check(a, b, expected) {
        a.forEach((d, i) => view.setBigInt64(1024 + i * 8, BigInt(d), true));
        b.forEach((d, i) => view.setBigInt64(1280 + i * 8, BigInt(d), true));
        view.setBigInt64(1800, 99n, true);
        view.setBigInt64(1808, 99n, true);
        const rc = shape(1024, BigInt(a.length), 1280, BigInt(b.length), 1536, 1800, 1808);
        assert.strictEqual(rc, expected === null ? -1n : 0n);
        assert.strictEqual(view.getBigInt64(1800, true), BigInt(expected?.length || 0));
        assert.strictEqual(view.getBigInt64(1808, true), expected === null ? 0n :
            expected.reduce((n, d) => n * BigInt(d), 1n));
        expected?.forEach((d, i) => assert.strictEqual(view.getBigInt64(1536 + i * 8, true), BigInt(d)));
    }
    check([2, 1], [1, 3], [2, 3]);
    check([], [3], [3]);
    check([], [], []);
    check([0, 3], [1, 3], [0, 3]);
    check([2], [3], null);
    check([-1], [1], null);
    check([0x7fffffffffffffffn, 2], [1], null);
    check(Array(17).fill(1), [1], null);
    console.log(`${path}: broadcast shape PASS`);
}
