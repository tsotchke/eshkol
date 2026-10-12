#!/usr/bin/env node
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const program = fs.readFileSync(process.argv[2]);
const abi = fs.readFileSync(process.argv[3]);
function load(file, name, limited = false) {
    const output=[];
    const hostConsole={...console, log: (...values)=>output.push(values.join(' '))};
    const context = {console: hostConsole, WebAssembly, TextEncoder, TextDecoder, Map, Set,
        Uint8Array, DataView, ArrayBuffer, document: {body: {}}, window: {}};
    context.globalThis = context;
    vm.runInNewContext(fs.readFileSync(path.join(root, file), 'utf8') + `\nthis.Runtime=${name};`, context);
    const rt = new context.Runtime();
    rt._testOutput=output;
    if (limited) {
        const pages = name === 'EshkolRepl' ? 3 : 20;
        const memory = new WebAssembly.Memory({initial: pages, maximum: pages});
        if (name === 'EshkolRepl') rt.memory = memory; else rt._importedMemory = memory;
    }
    return rt;
}
const memory = rt => rt.memory || rt._importedMemory;
const view = rt => new DataView(memory(rt).buffer);
async function instance(rt, bytes) {
    rt.prepareWasm(bytes);
    const imports = rt.createImports();
    imports.env.memory = memory(rt); // the independently compiled C ABI caller
    const result = await WebAssembly.instantiate(bytes, imports);
    return {exports: result.instance.exports, env: imports.env};
}
function tensor(rt, shape, slots, dtype = 64, dirtyPadding = false) {
    const exact = rt._numeric(), p = exact.header(40, 3);
    const dims = rt._bump(shape.length * 8), data = rt._bump(Math.max(1, slots.length) * (dtype === 0 ? 8 : 16));
    let v = view(rt);
    v.setUint32(p, dims, true); v.setBigUint64(p + 8, BigInt(shape.length), true);
    v.setUint32(p + 16, data, true); v.setBigUint64(p + 24, BigInt(slots.length), true);
    v.setBigUint64(p + 32, BigInt(dtype), true);
    if (dirtyPadding) { v.setUint32(p + 4, 0xfedcba98, true); v.setUint32(p + 20, 0xabcdef01, true); }
    shape.forEach((d, i) => v.setBigUint64(dims + i * 8, BigInt(d), true));
    slots.forEach((x, i) => {
        if (dtype === 0) { view(rt).setFloat64(data + i * 8, x, true); return; }
        const q = data + i * 16;
        if (Array.isArray(x)) {
            const dual = rt._bump(64); v = view(rt);
            v.setUint8(q, 6); v.setUint8(q + 1, 0x20); v.setBigUint64(q + 8, BigInt(dual), true);
            x.forEach((c, j) => v.setFloat64(dual + j * 8, c, true));
        } else if (typeof x === 'object') {
            const payload = exact.header(64, x.subtype || 23); v = view(rt);
            v.setUint8(q, x.tag || 8); v.setBigUint64(q + 8, BigInt(payload), true);
        } else exact.write(q, typeof x === 'bigint' ? {n: x, d: 1n} : {double: x});
    });
    return p;
}
function readTensor(rt, p) {
    const v = view(rt), rank = Number(v.getBigUint64(p + 8, true));
    const dims = v.getUint32(p, true), data = v.getUint32(p + 16, true), total = Number(v.getBigUint64(p + 24, true));
    assert.equal(v.getUint8(p - 8), 3); assert.equal(v.getUint32(p - 4, true), 40);
    assert.equal(v.getBigUint64(p + 32, true), 64n);
    return {shape: Array.from({length: rank}, (_, i) => Number(v.getBigUint64(dims + i * 8, true))),
        slots: Array.from({length: total}, (_, i) => {
            const q = data + i * 16;
            if (v.getUint8(q) === 6) {
                const d = Number(v.getBigUint64(q + 8, true));
                return Array.from({length: 8}, (_, j) => v.getFloat64(d + j * 8, true));
            }
            return {type: v.getUint8(q), text: rt._numeric().format(rt._numeric().read(q))};
        })};
}
const A = [2,1,2,4,3,5,6,7], B = [5,7,11,17,13,19,23,29];
const PRODUCT = [10,19,32,79,41,97,135,354];
async function run(file, name) {
    let compiledCalls = 0, grownCalls = 0;
    for (const grow of [false, true]) {
        const rt = load(file, name); rt.prepareWasm(program); const imports = rt.createImports();
        const observed = [], display=imports.env.eshkol_display_value;
        imports.env.eshkol_display_value = p => {
            const before=rt._testOutput.length;
            display(p);
            assert.equal(rt._testOutput.length,before+1,'production display emits the computed scalar');
            observed.push(rt._testOutput.at(-1));
        };
        for (const key of ['eshkol_jet_tensor_binary', 'eshkol_jet_tensor_matmul']) {
            const original = imports.env[key]; assert.equal(typeof original, 'function');
            imports.env[key] = (...args) => {
                compiledCalls++;
                if (grow) rt._bump(memory(rt).buffer.byteLength - rt._bumpPtr - 8);
                const before = memory(rt).buffer, out = original(...args);
                if (grow) { assert.notEqual(memory(rt).buffer, before); grownCalls++; }
                return out;
            };
        }
        const {instance: module} = await WebAssembly.instantiate(program, imports);
        if (rt.setInstance) rt.setInstance(module);
        module.exports.main(0, 0);
        assert.deepEqual(observed, ['7','6','6'], `${file}: compiled ordinary matmul and tensor derivatives`);
    }
    assert.ok(compiledCalls >= 4 && grownCalls >= 2, `${file}: both actual compiler routes executed`);
    const rt = load(file, name), {exports: call, env} = await instance(rt, abi);
    const multiple=env.arena_allocate_multi_value(1,3);
    assert.equal(view(rt).getUint8(multiple-8),4);
    assert.equal(view(rt).getBigUint64(multiple,true),3n);
    assert.equal(view(rt).getUint32(multiple-4,true),56);
    const listOut=rt._bump(16), emptyList=rt._bump(16);
    env.eshkol_list_to_vector_sret(listOut,emptyList);
    let listView=view(rt), emptyVector=Number(listView.getBigUint64(listOut+8,true));
    assert.equal(listView.getUint8(listOut),8); assert.equal(listView.getUint8(emptyVector-8),2);
    assert.equal(listView.getBigInt64(emptyVector,true),0n);
    const cons=rt._numeric().header(32,0), list=rt._bump(16);
    rt._numeric().write(cons,{n:9007199254740993n,d:1n});
    listView=view(rt); listView.setUint8(list,8); listView.setBigUint64(list+8,BigInt(cons),true);
    env.eshkol_list_to_vector_sret(listOut,list);
    const vector=Number(view(rt).getBigUint64(listOut+8,true));
    assert.equal(view(rt).getBigInt64(vector,true),1n);
    assert.equal(rt._numeric().format(rt._numeric().read(vector+8)),'9007199254740993');
    listView=view(rt); listView.setUint8(cons+16,8); listView.setBigUint64(cons+24,BigInt(cons),true);
    const originalCeiling=rt._exactMemoryCeiling, cycleCheckpoint=rt._bumpPtr;
    rt._exactMemoryCeiling=rt._bumpPtr+256;
    assert.throws(()=>env.eshkol_list_to_vector_sret(listOut,list),/memory|cycle|limit/i);
    assert.equal(rt._bumpPtr,cycleCheckpoint); rt._exactMemoryCeiling=originalCeiling;
    // The raw LLVM browser lane uses TypeSystem's 144-byte AD node layout;
    // its size_t-shaped IR fields remain i64 even with wasm32 pointers.
    const pack=env.arena_allocate_ad_node_with_header(1), variable=env.arena_allocate_ad_node_with_header(1);
    assert.equal(view(rt).getUint32(pack-4,true),144);
    const packedValue=rt._bump(8), packedShape=rt._bump(8), savedNode=rt._bump(4);
    let nodeView=view(rt); nodeView.setInt32(pack,83,true); // TENSOR_PACK
    nodeView.setFloat64(packedValue,7,true); nodeView.setBigInt64(packedShape,1n,true);
    nodeView.setUint32(pack+40,packedValue,true); nodeView.setUint32(pack+120,packedShape,true);
    nodeView.setBigUint64(pack+128,1n,true); nodeView.setUint32(pack+56,savedNode,true);
    nodeView.setBigUint64(pack+64,1n,true);
    const projected=env.eshkol_ad_dense_node_elements(pack);
    assert.equal(view(rt).getFloat64(view(rt).getUint32(projected+16,true),true),7);
    nodeView=view(rt); nodeView.setInt32(variable,1,true); nodeView.setUint32(savedNode,variable,true);
    const projectionCheckpoint=rt._bumpPtr;
    assert.throws(()=>env.eshkol_ad_dense_node_elements(pack),/reverse|unsupported/i);
    assert.equal(rt._bumpPtr,projectionCheckpoint,'variable-dependent projection refuses without erasing its derivative');
    view(rt).setUint32(savedNode,pack,true);
    assert.throws(()=>env.eshkol_ad_dense_node_elements(pack),/cyclic/i);
    assert.equal(rt._bumpPtr,projectionCheckpoint);
    assert.throws(()=>env.eshkol_ad_dense_node_elements(BigInt(pack)+(1n<<32n)),/wasm32/i);
    const a = tensor(rt, [1,1], [A], 64, true), b = tensor(rt, [1,1], [B], 64, true);
    assert.deepEqual(readTensor(rt, call.binary(a,b,0,0)).slots[0], [7,8,13,21,16,24,29,36]);
    assert.deepEqual(readTensor(rt, call.binary(a,b,1,0)).slots[0], [-3,-6,-9,-13,-10,-14,-17,-22]);
    const product = call.binary(a,b,2,0);
    assert.deepEqual(readTensor(rt, product).slots[0], PRODUCT);
    const quotient = readTensor(rt, call.binary(product,b,3,0)).slots[0];
    quotient.forEach((v,i) => assert.ok(Math.abs(v-A[i]) < 1e-12));
    assert.deepEqual(readTensor(rt, call.matmul(a,b,0)).slots[0], PRODUCT);
    const squareA = tensor(rt,[2,2],[A,A,A,A]), squareB = tensor(rt,[2,2],[B,B,B,B]);
    readTensor(rt,call.matmul(squareA,squareB,0)).slots.forEach(x=>assert.deepEqual(x,PRODUCT.map(c=>2*c)));
    const column = tensor(rt, [2,1], [A,A]), row = tensor(rt, [1,2], [B,B]);
    const broadcast = readTensor(rt, call.binary(column,row,2,0));
    assert.deepEqual(broadcast.shape,[2,2]); broadcast.slots.forEach(x=>assert.deepEqual(x,PRODUCT));
    const numeric = tensor(rt,[1,1],[2],0);
    assert.deepEqual(readTensor(rt,call.binary(a,numeric,2,0)).slots[0],A.map(x=>2*x));
    const exact = tensor(rt,[1],[9007199254740993n],65), one = tensor(rt,[1],[1n]);
    assert.deepEqual(readTensor(rt,call.binary(exact,one,0,0)).slots[0],{type:1,text:'9007199254740994'});
    const fusedA=tensor(rt,[1],[[-1,1+2**-27,0,0,0,0,0,0]]);
    const fusedB=tensor(rt,[1],[[1-2**-27,1,0,0,0,0,0,0]]);
    assert.equal(readTensor(rt,call.binary(fusedA,fusedB,2,0)).slots[0][1],-(2**-54),'single-rounded FMA');
    const tiny=2**-1023+Number.MIN_VALUE;
    const underflowA=tensor(rt,[1],[[-(tiny+Number.MIN_VALUE),1+Number.EPSILON,0,0,0,0,0,0]]);
    const underflowB=tensor(rt,[1],[[tiny,1,0,0,0,0,0,0]]);
    assert.ok(Object.is(readTensor(rt,call.binary(underflowA,underflowB,2,0)).slots[0][1],-0),
        'single rounding retains negative underflow zero');
    const overflowA=tensor(rt,[1],[[-Infinity,1e308,0,0,0,0,0,0]]);
    const overflowB=tensor(rt,[1],[[1e308,1,0,0,0,0,0,0]]);
    assert.equal(readTensor(rt,call.binary(overflowA,overflowB,2,0)).slots[0][1],-Infinity,
        'finite product must not overflow before addition to infinite accumulator');
    const inf=tensor(rt,[1],[[Infinity,0,1,0,0,0,0,0]]), zero=tensor(rt,[1],[[0,0,0,0,0,0,0,0]]);
    const pole=readTensor(rt,call.binary(inf,zero,2,0)).slots[0];
    assert.ok(Number.isNaN(pole[0])); pole.slice(1).forEach(x=>assert.equal(x,0));
    for (const bad of [tensor(rt,[1],[{subtype:23}]),tensor(rt,[1],[{tag:9,subtype:22}]),tensor(rt,[1],[{tag:33}])]) {
        const checkpoint=rt._bumpPtr;
        assert.throws(()=>call.binary(bad,one,0,0),/unsupported|numeric|reverse|Taylor/i);
        assert.equal(rt._bumpPtr,checkpoint,'refusal preserves arena');
    }
    assert.throws(()=>call.binary(a,b,0,1),/reverse/i);
    assert.throws(()=>call.matmul(column,column,0),/shape|2-D|cols/i);
    const badShape=tensor(rt,[3],[1,2,3],0);
    assert.throws(()=>call.binary(row,badShape,0,0),/broadcast|shape/i);
    const limited=load(file,name,true), limitedCall=(await instance(limited,abi)).exports;
    const la=tensor(limited,[1],[A]),lb=tensor(limited,[1],[B]);
    limited._bump(memory(limited).buffer.byteLength-limited._bumpPtr-48);
    const checkpoint=limited._bumpPtr;
    assert.throws(()=>limitedCall.binary(la,lb,2,0),/exhaust|memory|limit/i);
    assert.equal(limited._bumpPtr,checkpoint,'partial allocation rolls back');
    console.log(`${file}: compiled Eshkol routes, C ABI jets, exact slots, FMA, growth and refusal controls PASS`);
}
(async()=>{
    await run('web/eshkol-repl.js','EshkolRepl');
    await run('site/static/eshkol-runtime.js','EshkolRuntime');
})().catch(e=>{console.error(e);process.exitCode=1;});
