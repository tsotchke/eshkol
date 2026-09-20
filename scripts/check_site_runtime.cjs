// Exercise runtime imports used during homepage execution, not just WASM parsing.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const file = path.join(__dirname, '..', 'site/static/eshkol-runtime.js');
const context = {document: {body: {}}, window: {}, console, WebAssembly,
                 TextEncoder, TextDecoder, DataView, Uint8Array};
vm.createContext(context);
vm.runInContext(fs.readFileSync(file, 'utf8') + '\nthis.Runtime = EshkolRuntime;', context);
const runtime = new context.Runtime();
const predicate = runtime.createImports().env.eshkol_is_i128_tagged;
assert.equal(predicate(0), 0);
runtime.memory = new WebAssembly.Memory({initial: 1});
const data = new DataView(runtime.memory.buffer);
for (let type = 0; type < 16; type++) {
  data.setUint8(32, type);
  if (type !== 8) assert.equal(predicate(32), 0);
}
data.setUint8(32, 8);
data.setBigUint64(40, 128n, true);
data.setUint8(120, 2);
assert.equal(predicate(32), 0, 'ordinary heap values must not abort page rendering');
data.setUint8(120, 25);
assert.equal(predicate(32), 1, 'i128 payload must be recognized');
data.setUint8(32, 0x88);
assert.equal(predicate(32), 1, 'non-type flag bits must be ignored');
data.setBigUint64(40, 0n, true);
assert.equal(predicate(32), 0);
console.log('PASS: site runtime classifies ordinary and i128 values without throwing');
