#!/usr/bin/env node
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const root = require('node:path').resolve(__dirname, '../..');
const bytes = new DataView(new ArrayBuffer(8));
const bits = d => { bytes.setFloat64(0, d, true); return bytes.getBigUint64(0, true); };
const number = b => { bytes.setBigUint64(0, b, true); return bytes.getFloat64(0, true); };
// Independent unreduced IEEE fraction oracle: retain the hidden significand
// and normalize by Euclid, instead of the runtime's odd-mantissa factoring.
function oracle(d) {
    const b = bits(d), exponent = Number(b >> 52n & 2047n);
    let n = (b & ((1n << 52n) - 1n)) + (exponent ? 1n << 52n : 0n);
    let denominator = 1n;
    const power = exponent ? exponent - 1023 - 52 : -1074;
    if (power < 0) denominator <<= BigInt(-power); else n <<= BigInt(power);
    let x = n, y = denominator;
    while (y) [x, y] = [y, x % y];
    return { n: (b >> 63n ? -n : n) / x, d: denominator / x };
}
function uleb(n) { const a = []; do { let b = n & 127; n >>>= 7; if (n) b |= 128; a.push(b); } while (n); return a; }
const str = s => [...uleb(s.length), ...Buffer.from(s)];
const section = (id, a) => [id, ...uleb(a.length), ...a];
// Real WASM function (f64)->void calls the production (arena,f64,out) import.
// Nonzero out lives below stack reservation; data segments exercise heap floor.
function fixture(extraData = false) {
    const types = [2, 0x60, 3, 0x7f, 0x7c, 0x7f, 0, 0x60, 1, 0x7c, 0];
    const imports = [1, ...str('env'), ...str('eshkol_double_to_exact_tagged'), 0, 0];
    const body = [0, 0x41, 1, 0x20, 0, 0x41, 0x80, 0x20, 0x10, 0, 0x0b]; // out4096
    return new Uint8Array([0,97,115,109,1,0,0,0,
        ...section(1,types), ...section(2,imports), ...section(3,[1,1]),
        ...section(7,[1,...str('convert'),0,1]), ...section(10,[1,...uleb(body.length),...body]),
        ...(extraData ? section(11, [1, 0, 0x41, 0x80, 0x80, 0x80, 0x01, 0x0b, 1, 7]) : [])]);
}
function memcmpFixture() {
    const types = [1, 0x60, 3, 0x7f, 0x7f, 0x7f, 1, 0x7f];
    const imports = [1, ...str('env'), ...str('memcmp'), 0, 0];
    const exports = [1, ...str('compare'), 0, 0];
    return new Uint8Array([0,97,115,109,1,0,0,0,
        ...section(1, types), ...section(2, imports), ...section(7, exports)]);
}
function exactBridgeFixture() {
    const types = [
        3,
        0x60, 3, 0x7f, 0x7f, 0x7f, 0,
        0x60, 3, 0x7f, 0x7f, 0x7f, 1, 0x7f,
        0x60, 1, 0x7f, 1, 0x7e,
    ];
    const imports = [3,
        ...str('env'), ...str('eshkol_list_reverse_tagged'), 0, 0,
        ...str('env'), ...str('eshkol_make_string_checked'), 0, 1,
        ...str('env'), ...str('eshkol_string_byte_length'), 0, 2];
    const exports = [3,
        ...str('reverse'), 0, 0,
        ...str('make_string'), 0, 1,
        ...str('byte_length'), 0, 2];
    return new Uint8Array([0,97,115,109,1,0,0,0,
        ...section(1, types), ...section(2, imports), ...section(7, exports)]);
}
function consBridgeFixture() {
    const types = [
        9,
        0x60, 2, 0x7f, 0x7f, 1, 0x7e, // get int64 / pointer
        0x60, 2, 0x7f, 0x7f, 1, 0x7c, // get double
        0x60, 2, 0x7f, 0x7f, 1, 0x7f, // get type / flags
        0x60, 4, 0x7f, 0x7f, 0x7e, 0x7f, 0, // set int64
        0x60, 4, 0x7f, 0x7f, 0x7c, 0x7f, 0, // set double
        0x60, 2, 0x7f, 0x7f, 0, // set null
        0x60, 3, 0x7f, 0x7f, 0x7f, 0, // set tagged value
        0x60, 4, 0x7f, 0x7f, 0x7e, 0x7f, 0, // set pointer
        0x60, 3, 0x7f, 0x7f, 0x7f, 0, // get tagged value sret
    ];
    const names = [
        'arena_tagged_cons_get_int64', 'arena_tagged_cons_get_double',
        'arena_tagged_cons_get_ptr', 'arena_tagged_cons_get_type',
        'arena_tagged_cons_get_flags', 'arena_tagged_cons_set_int64',
        'arena_tagged_cons_set_double', 'arena_tagged_cons_set_ptr',
        'arena_tagged_cons_set_null', 'arena_tagged_cons_set_tagged_value',
        'arena_tagged_cons_get_tagged_value',
    ];
    const typeIndices = [0, 1, 0, 2, 2, 3, 4, 7, 5, 6, 8];
    const imports = [names.length, ...names.flatMap((name, i) =>
        [...str('env'), ...str(name), 0, typeIndices[i]])];
    const exports = [names.length, ...names.flatMap((name, i) => [...str(name), 0, i])];
    return new Uint8Array([0,97,115,109,1,0,0,0,
        ...section(1, types), ...section(2, imports), ...section(7, exports)]);
}
function load(file, className) {
    const context = { console, WebAssembly, TextEncoder, TextDecoder, Map, Set,
        Uint8Array, DataView, ArrayBuffer, document: { body: {} }, window: {} };
    context.globalThis = context;
    vm.runInNewContext(fs.readFileSync(`${root}/${file}`, 'utf8') + `\nthis.TestClass=${className};`, context);
    return new context.TestClass();
}
async function run(file, className) {
    const rt = load(file, className), wasm = fixture();
    const layout = rt.prepareWasm(wasm), env = rt.createImports().env;
    assert.ok(layout.stackBegin >= layout.staticEnd);
    assert.ok(layout.heapFloor >= layout.stackEnd);
    assert.equal(env.__stack_pointer.value, layout.stackEnd);
    const abi = JSON.parse(fs.readFileSync(`${root}/scripts/wasm_exact_import_abi.json`));
    for (const [name, parameters] of Object.entries(abi)) {
        assert.equal(typeof env[name], 'function', `${name}: production import present`);
        if (name !== 'eshkol_wasm_numeric_abi_check') assert.equal(env[name].length, parameters.length, `${name}: native ABI arity`);
    }
    assert.deepEqual(abi.eshkol_rational_floor_tagged,['arena_ptr','rational_payload_ptr','tagged_out_ptr']);
    assert.deepEqual(abi.eshkol_rational_compare_tagged_ptr,['arena_ptr','tagged_ptr','tagged_ptr','i32','tagged_out_ptr']);
    const memory = rt.memory || rt._importedMemory;
    const bridgeWasm = await WebAssembly.instantiate(exactBridgeFixture(), { env });
    const reverse = bridgeWasm.instance.exports.reverse;
    const makeString = bridgeWasm.instance.exports.make_string;
    const stringByteLength = bridgeWasm.instance.exports.byte_length;
    const consWasm = await WebAssembly.instantiate(consBridgeFixture(), { env });
    const cons = consWasm.instance.exports;
    const allocRaw = size => env.arena_allocate(1, size);
    const taggedBytes = (tag, value = 0n) => {
        const bytes = new Uint8Array(16); bytes[0] = tag;
        new DataView(bytes.buffer).setBigUint64(8, BigInt.asUintN(64, BigInt(value)), true);
        return bytes;
    };
    const copyBytes = (ptr, size) => new Uint8Array(memory.buffer, ptr, size).slice();
    const storeTagged = (ptr, bytes) => new Uint8Array(memory.buffer, ptr, 16).set(bytes);
    const allocateCons = (car, cdr) => {
        const cell = env.arena_allocate_cons_with_header(1);
        storeTagged(cell, car); storeTagged(cell + 16, cdr); return cell;
    };
    const headPointer = (ptr, tag = 8) => {
        const slot = allocRaw(16); storeTagged(slot, taggedBytes(tag, BigInt(ptr))); return slot;
    };
    const readReversedCars = out => {
        const bytes = new Uint8Array(memory.buffer), dv = new DataView(memory.buffer);
        let tag = bytes[out], ptr = Number(dv.getBigUint64(out + 8, true));
        const cars = [], cells = new Set();
        while (tag === 8) {
            assert.ok(ptr && !cells.has(ptr), `${file}: reverse result is an acyclic heap list`);
            cells.add(ptr); cars.push(copyBytes(ptr, 16));
            const cdr = ptr + 16; tag = bytes[cdr]; ptr = Number(dv.getBigUint64(cdr + 8, true));
        }
        assert.equal(tag, 0, `${file}: reverse result ends in null`);
        assert.equal(ptr, 0, `${file}: null result has zero payload`);
        return { cars, cells };
    };
    for (const name of ['eshkol_list_reverse_tagged', 'eshkol_make_string_checked', 'eshkol_string_byte_length'])
        assert.equal(typeof env[name], 'function', `${file}: production bridge import ${name} exists`);
    assert.equal(env.eshkol_list_reverse_tagged.length, 3, `${file}: reverse matches native three-pointer ABI`);
    assert.equal(env.eshkol_make_string_checked.length, 3, `${file}: make-string matches native ABI`);
    assert.equal(env.eshkol_string_byte_length.length, 1, `${file}: byte length matches native ABI`);
    for (const name of ['arena_tagged_cons_get_int64', 'arena_tagged_cons_get_double',
        'arena_tagged_cons_get_ptr', 'arena_tagged_cons_get_type', 'arena_tagged_cons_get_flags',
        'arena_tagged_cons_set_int64', 'arena_tagged_cons_set_double', 'arena_tagged_cons_set_ptr',
        'arena_tagged_cons_set_null', 'arena_tagged_cons_set_tagged_value',
        'arena_tagged_cons_get_tagged_value']) {
        assert.equal(typeof cons[name], 'function', `${file}: real WASM cons import ${name}`);
    }
    const consInt = (cell, isCdr) => cons.arena_tagged_cons_get_int64(cell, isCdr ? 1 : 0);
    const consType = (cell, isCdr) => cons.arena_tagged_cons_get_type(cell, isCdr ? 1 : 0);
    const consTagged = (cell, isCdr, out) => cons.arena_tagged_cons_get_tagged_value(out, cell, isCdr ? 1 : 0);
    const consCell = env.arena_allocate_cons_with_header(1);
    const consCar = taggedBytes(1, -0x123456789abcdefn), consCdr = taggedBytes(2, bits(3.25));
    const consCarPtr = allocRaw(16), consCdrPtr = allocRaw(16), consOut = allocRaw(16);
    storeTagged(consCarPtr, consCar); storeTagged(consCdrPtr, consCdr);
    cons.arena_tagged_cons_set_tagged_value(consCell, 0, consCarPtr);
    cons.arena_tagged_cons_set_tagged_value(consCell, 1, consCdrPtr);
    assert.equal(consInt(consCell, false), -0x123456789abcdefn, `${file}: car int64 keeps signed 64-bit payload`);
    assert.equal(cons.arena_tagged_cons_get_double(consCell, 1), 3.25, `${file}: true selects cdr double`);
    assert.equal(consType(consCell, false), 1, `${file}: false selects car type`);
    assert.equal(consType(consCell, true), 2, `${file}: true selects cdr type`);
    assert.deepEqual([...new Uint8Array(memory.buffer, consCell, 16)], [...consCar], `${file}: cdr write preserves car`);
    consTagged(consCell, false, consOut);
    assert.deepEqual([...copyBytes(consOut, 16)], [...consCar], `${file}: tagged getter sret copies all 16 car bytes`);
    consTagged(consCell, true, consOut);
    assert.deepEqual([...copyBytes(consOut, 16)], [...consCdr], `${file}: tagged getter sret copies all 16 cdr bytes`);
    const flagged = taggedBytes(1, 44n); flagged[1] = 0x10; flagged[2] = 0x34; flagged[3] = 0x12;
    new DataView(flagged.buffer).setUint32(4, 0x89abcdef, true);
    const flaggedPtr = allocRaw(16); storeTagged(flaggedPtr, flagged);
    cons.arena_tagged_cons_set_tagged_value(consCell, 0, flaggedPtr);
    assert.equal(cons.arena_tagged_cons_get_flags(consCell, 0), 0x10, `${file}: flags getter exposes raw tag flags`);
    consTagged(consCell, false, consOut);
    assert.deepEqual([...copyBytes(consOut, 16)], [...flagged],
        `${file}: full tagged getter preserves type, flags, reserved, padding, and payload bytes`);
    cons.arena_tagged_cons_set_int64(consCell, 0, 45n, 1);
    assert.equal(consInt(consCell, false), 45n, `${file}: int64 setter stores value`);
    assert.equal(cons.arena_tagged_cons_get_flags(consCell, 0), 0x10, `${file}: scalar setter preserves native flags behavior`);
    cons.arena_tagged_cons_set_double(consCell, 0, -8.5, 2);
    assert.equal(cons.arena_tagged_cons_get_double(consCell, 0), -8.5, `${file}: double setter stores value`);
    cons.arena_tagged_cons_set_ptr(consCell, 1, 0x123456789abcdef0n, 8);
    assert.equal(cons.arena_tagged_cons_get_ptr(consCell, 1), 0x123456789abcdef0n, `${file}: pointer setter/getter preserve i64`);
    cons.arena_tagged_cons_set_null(consCell, 1);
    assert.equal(cons.arena_tagged_cons_get_ptr(consCell, 1), 0n, `${file}: null pointer reads as zero`);

    // Legacy raw 32-byte cells remain accepted, and a distinct object subtype,
    // malformed bounds, and malformed cons header are refused.
    const legacyCons = allocRaw(32);
    cons.arena_tagged_cons_set_int64(legacyCons, 0, -7n, 1);
    assert.equal(consInt(legacyCons, false), -7n, `${file}: legacy raw cons cell is supported`);
    const nonCons = rt._exact.header(32, 1);
    assert.throws(() => consType(nonCons, false), /non-cons object/, `${file}: non-cons subtype is refused`);
    assert.throws(() => consType(memory.buffer.byteLength - 16, false), /invalid WASM memory span/,
        `${file}: out-of-bounds cell span is refused`);
    const malformedCons = rt._exact.header(16, 0);
    assert.throws(() => consType(malformedCons, false), /invalid WASM cons header size/,
        `${file}: malformed cons header is refused`);
    const growthBuffer = memory.buffer;
    memory.grow(1);
    assert.notEqual(memory.buffer, growthBuffer, `${file}: test grew linear memory`);
    cons.arena_tagged_cons_set_int64(consCell, 0, 99n, 1);
    assert.equal(consInt(consCell, false), 99n, `${file}: cons accesses refresh views after memory growth`);
    assert.throws(() => cons.arena_tagged_cons_set_double(consCell, 0, 1.0, 1), /invalid type/,
        `${file}: invalid typed setter is refused`);

    const integerValue = taggedBytes(1, -17n), characterValue = taggedBytes(4, 0x3bbn);
    const rationalSlot = allocRaw(16); rt._exact.write(rationalSlot, { n: 7n, d: 3n });
    const rationalValue = copyBytes(rationalSlot, 16);
    const nullValue = new Uint8Array(16);
    const tail = allocateCons(characterValue, nullValue);
    const middle = allocateCons(rationalValue, taggedBytes(8, BigInt(tail)));
    const first = allocateCons(integerValue, taggedBytes(8, BigInt(middle)));
    const head = headPointer(first), reverseOut = allocRaw(16);
    const inputSnapshots = [head, first - 8, middle - 8, tail - 8].map(ptr => ({ ptr, bytes: copyBytes(ptr, ptr === head ? 16 : 40) }));
    reverse(1, head, reverseOut);
    const reversed = readReversedCars(reverseOut);
    assert.deepEqual(reversed.cars, [characterValue, rationalValue, integerValue], `${file}: proper list reverses full tagged values`);
    for (const ptr of [first, middle, tail]) assert.ok(!reversed.cells.has(ptr), `${file}: reverse allocates fresh cons cells`);
    for (const snapshot of inputSnapshots)
        assert.deepEqual(copyBytes(snapshot.ptr, snapshot.bytes.length), snapshot.bytes, `${file}: input list remains untouched`);

    const emptyHead = headPointer(0, 0), emptyOut = allocRaw(16);
    new Uint8Array(memory.buffer, emptyOut, 16).fill(0xa5);
    reverse(1, emptyHead, emptyOut);
    assert.deepEqual([...copyBytes(emptyOut, 16)], Array(16).fill(0), `${file}: empty list writes a zeroed null tagged value`);

    const dottedTail = taggedBytes(1, 99n), dottedCar = taggedBytes(1, 88n);
    const dottedCell = allocateCons(dottedCar, dottedTail), dottedHead = headPointer(dottedCell), dottedOut = allocRaw(16);
    const dottedSnapshot = copyBytes(dottedCell - 8, 40);
    reverse(1, dottedHead, dottedOut);
    assert.deepEqual(readReversedCars(dottedOut).cars, [dottedCar], `${file}: reverse drops a dotted tail`);
    assert.deepEqual(copyBytes(dottedCell - 8, 40), dottedSnapshot, `${file}: dotted input remains untouched`);

    const legacyCell = allocRaw(32), legacyCar = taggedBytes(2, bits(2.5));
    storeTagged(legacyCell, legacyCar); storeTagged(legacyCell + 16, nullValue);
    const legacyHead = headPointer(legacyCell, 32), legacyOut = allocRaw(16);
    reverse(1, legacyHead, legacyOut);
    assert.deepEqual(readReversedCars(legacyOut).cars, [legacyCar], `${file}: legacy CONS32 tag is supported`);

    const malformedHead = headPointer(memory.buffer.byteLength + 8), malformedOut = allocRaw(16);
    assert.throws(() => reverse(1, malformedHead, malformedOut), /invalid WASM cons header span/,
        `${file}: malformed cons pointers are refused`);
    const cyclicCell = env.arena_allocate_cons_with_header(1);
    storeTagged(cyclicCell, integerValue); storeTagged(cyclicCell + 16, taggedBytes(8, BigInt(cyclicCell)));
    const cyclicHead = headPointer(cyclicCell), cyclicOut = allocRaw(16), bumpBeforeCycle = rt._bumpPtr;
    assert.throws(() => reverse(1, cyclicHead, cyclicOut), /cyclic WASM cons list/,
        `${file}: cyclic lists are refused`);
    assert.equal(rt._bumpPtr, bumpBeforeCycle, `${file}: cycle rejection performs no allocation`);

    const makeTaggedSlot = bytes => { const ptr = allocRaw(16); storeTagged(ptr, bytes); return ptr; };
    const lengthOne = makeTaggedSlot(taggedBytes(1, 2n));
    const lambdaFill = makeTaggedSlot(characterValue);
    const lambdaString = makeString(1, lengthOne, lambdaFill);
    const lambdaBytes = new TextEncoder().encode('λλ');
    assert.equal(new DataView(memory.buffer).getUint8(lambdaString - 8), 1, `${file}: make-string allocates STRING subtype`);
    assert.equal(new DataView(memory.buffer).getUint32(lambdaString - 4, true), lambdaBytes.length + 1,
        `${file}: string header stores UTF-8 bytes plus terminator`);
    assert.deepEqual([...new Uint8Array(memory.buffer, lambdaString, lambdaBytes.length + 1)], [...lambdaBytes, 0]);
    assert.equal(stringByteLength(lambdaString), 4n, `${file}: byte length counts UTF-8 bytes`);
    const defaultFillString = makeString(1, lengthOne, 0);
    assert.deepEqual([...new Uint8Array(memory.buffer, defaultFillString, 3)], [0x20, 0x20, 0],
        `${file}: omitted fill defaults to spaces`);
    const zeroFill = makeTaggedSlot(taggedBytes(4, 0n));
    const nulString = makeString(1, lengthOne, zeroFill);
    assert.deepEqual([...new Uint8Array(memory.buffer, nulString, 3)], [0, 0, 0], `${file}: NUL fill is repeated and terminated`);
    assert.equal(stringByteLength(nulString), 2n, `${file}: header length preserves embedded NULs`);
    const rawCString = allocRaw(8);
    new Uint8Array(memory.buffer, rawCString, 4).set([0xe2, 0x98, 0x83, 0]);
    assert.equal(stringByteLength(rawCString), 3n, `${file}: raw C-string fallback counts bytes to NUL`);
    assert.equal(stringByteLength(0), 0n, `${file}: null C-string pointer has zero byte length`);
    assert.throws(() => makeString(1, makeTaggedSlot(taggedBytes(2, 2n)), lambdaFill), /non-negative exact integer/,
        `${file}: inexact make-string length is rejected`);
    assert.throws(() => makeString(1, makeTaggedSlot(taggedBytes(1, -1n)), lambdaFill), /non-negative exact integer/,
        `${file}: negative make-string length is rejected`);
    assert.throws(() => makeString(1, lengthOne, makeTaggedSlot(taggedBytes(1, 32n))), /Unicode character/,
        `${file}: non-character fill is rejected`);
    assert.throws(() => makeString(1, lengthOne, makeTaggedSlot(taggedBytes(4, 0xd800n))), /Unicode character/,
        `${file}: surrogate fill is rejected`);
    assert.throws(() => makeString(1, lengthOne, makeTaggedSlot(taggedBytes(4, 0x110000n))), /Unicode character/,
        `${file}: out-of-range fill is rejected`);
    assert.throws(() => makeString(1, makeTaggedSlot(taggedBytes(1, 0x7fffffffffffffffn)), lambdaFill), /header limit/,
        `${file}: oversized string length is rejected without truncation`);

    const isolatedRuntime = async maximum => {
        const isolated = load(file, className), isolatedMemory = new WebAssembly.Memory({ initial: 20, maximum });
        if (className === 'EshkolRepl') isolated.memory = isolatedMemory;
        else isolated._importedMemory = isolatedMemory;
        isolated.prepareWasm(wasm);
        const isolatedEnv = isolated.createImports().env;
        const isolatedModule = await WebAssembly.instantiate(exactBridgeFixture(), { env: isolatedEnv });
        return { runtime: isolated, memory: isolatedMemory, env: isolatedEnv,
            exports: isolatedModule.instance.exports };
    };
    const freshLegacy = await isolatedRuntime(21);
    const freshLegacyCell = freshLegacy.env.arena_allocate(1, 32);
    freshLegacy.env.arena_tagged_cons_set_int64(freshLegacyCell, 0, -12n, 1);
    assert.equal(freshLegacy.env.arena_tagged_cons_get_int64(freshLegacyCell, 0), -12n,
        `${file}: first raw legacy cons is accepted with a zeroed preceding span`);
    const growth = await isolatedRuntime(22), growthMemory = growth.memory;
    const growthLength = growth.env.arena_allocate(1, 16), growthFill = growth.env.arena_allocate(1, 16);
    new Uint8Array(growthMemory.buffer, growthLength, 16).set(taggedBytes(1, 2n));
    new Uint8Array(growthMemory.buffer, growthFill, 16).set(characterValue);
    const growthCar = taggedBytes(1, 123n), growthCell = growth.env.arena_allocate_cons_with_header(1);
    new Uint8Array(growthMemory.buffer, growthCell, 16).set(growthCar);
    new Uint8Array(growthMemory.buffer, growthCell + 16, 16).set(nullValue);
    const growthHead = growth.env.arena_allocate(1, 16), growthOut = growth.env.arena_allocate(1, 16);
    new Uint8Array(growthMemory.buffer, growthHead, 16).set(taggedBytes(8, BigInt(growthCell)));
    growth.runtime._bump(growthMemory.buffer.byteLength - growth.runtime._bumpPtr - 8);
    let isolatedBufferBeforeGrowth = growthMemory.buffer;
    const grownString = growth.exports.make_string(1, growthLength, growthFill);
    assert.notEqual(growthMemory.buffer, isolatedBufferBeforeGrowth, `${file}: make-string handles memory growth`);
    assert.equal(growth.exports.byte_length(grownString), 4n, `${file}: grown string remains readable through its new buffer`);
    growth.runtime._bump(growthMemory.buffer.byteLength - growth.runtime._bumpPtr - 8);
    isolatedBufferBeforeGrowth = growthMemory.buffer;
    growth.exports.reverse(1, growthHead, growthOut);
    assert.notEqual(growthMemory.buffer, isolatedBufferBeforeGrowth, `${file}: list reverse handles memory growth`);
    const grownCell = Number(new DataView(growthMemory.buffer).getBigUint64(growthOut + 8, true));
    assert.equal(new DataView(growthMemory.buffer).getUint8(growthOut), 8);
    assert.deepEqual(new Uint8Array(growthMemory.buffer, grownCell, 16).slice(), growthCar,
        `${file}: grown reverse preserves the car bytes`);
    assert.equal(new DataView(growthMemory.buffer).getUint8(grownCell + 16), 0);

    const exhausted = await isolatedRuntime(20), exhaustedMemory = exhausted.memory;
    const exhaustedCar = taggedBytes(1, 5n), exhaustedCell = exhausted.env.arena_allocate_cons_with_header(1);
    new Uint8Array(exhaustedMemory.buffer, exhaustedCell, 16).set(exhaustedCar);
    new Uint8Array(exhaustedMemory.buffer, exhaustedCell + 16, 16).set(nullValue);
    const exhaustedHead = exhausted.env.arena_allocate(1, 16), exhaustedOut = exhausted.env.arena_allocate(1, 16);
    const exhaustedLength = exhausted.env.arena_allocate(1, 16), exhaustedFill = exhausted.env.arena_allocate(1, 16);
    new Uint8Array(exhaustedMemory.buffer, exhaustedHead, 16).set(taggedBytes(8, BigInt(exhaustedCell)));
    new Uint8Array(exhaustedMemory.buffer, exhaustedLength, 16).set(taggedBytes(1, 1n));
    new Uint8Array(exhaustedMemory.buffer, exhaustedFill, 16).set(characterValue);
    exhausted.runtime._bump(exhaustedMemory.buffer.byteLength - exhausted.runtime._bumpPtr - 8);
    const exhaustedCheckpoint = exhausted.runtime._bumpPtr;
    assert.throws(() => exhausted.exports.reverse(1, exhaustedHead, exhaustedOut), /arena exhausted/,
        `${file}: reverse refuses allocation exhaustion`);
    assert.equal(exhausted.runtime._bumpPtr, exhaustedCheckpoint, `${file}: reverse allocation failure rolls back`);
    assert.deepEqual([...new Uint8Array(exhaustedMemory.buffer, exhaustedOut, 16)], Array(16).fill(0),
        `${file}: failed reverse leaves the output as null`);
    assert.throws(() => exhausted.exports.make_string(1, exhaustedLength, exhaustedFill), /arena exhausted/,
        `${file}: make-string refuses allocation exhaustion`);
    assert.equal(exhausted.runtime._bumpPtr, exhaustedCheckpoint, `${file}: make-string allocation failure rolls back`);

    const memcmpWasm = await WebAssembly.instantiate(memcmpFixture(), { env });
    const compare = memcmpWasm.instance.exports.compare;
    const left = 9000, right = 9016;
    new Uint8Array(memory.buffer).set([0, 255, 128, 0], left);
    new Uint8Array(memory.buffer).set([0, 127, 255, 0], right);
    assert.ok(compare(left, right, 4) > 0, `${file}: memcmp compares unsigned bytes after NUL`);
    assert.ok(compare(right, left, 4) < 0, `${file}: memcmp returns the opposite sign`);
    assert.equal(compare(left, left, 4), 0, `${file}: equal byte ranges compare equal`);
    assert.equal(compare(-1, 0, 0), 0, `${file}: zero-length comparison does not access memory`);
    assert.throws(() => compare(0, right, 1), /memory span/, `${file}: null range is refused`);
    assert.throws(() => compare(memory.buffer.byteLength - 1, right, 2), /memory span/,
        `${file}: out-of-bounds range is refused`);
    assert.throws(() => compare(-1, right, 1), /memory span/,
        `${file}: signed wasm32 pointer is validated as unsigned`);
    const oldMemoryLength = memory.buffer.byteLength;
    memory.grow(1);
    new Uint8Array(memory.buffer).set([0x80, 0x7f], oldMemoryLength);
    assert.ok(compare(oldMemoryLength, oldMemoryLength + 1, 1) > 0,
        `${file}: memcmp reads the current buffer after memory growth`);
    assert.ok(compare(oldMemoryLength + 1, oldMemoryLength, 1) < 0);
    const closure = (imports, funcPtr, packedInfo, sexprPtr, returnTypeInfo, namePtr) =>
        imports.arena_allocate_closure_with_header(1, funcPtr, packedInfo, sexprPtr, returnTypeInfo, namePtr);
    const assertClosure = (p, packedInfo, subtype, namePtr, variadic) => {
        const v = new DataView(memory.buffer);
        assert.ok(p > 0 && (p & 7) === 0 && p + 40 <= memory.buffer.byteLength);
        assert.equal(v.getUint8(p - 8), subtype);
        assert.equal(v.getUint32(p - 4, true), 40);
        assert.equal(v.getBigUint64(p, true), 0x1234n);
        assert.equal(v.getBigUint64(p + 16, true), 0x5678n);
        assert.equal(v.getUint32(p + 24, true), Number(namePtr & 0xffffffffn));
        assert.equal(v.getUint8(p + 32), 1);
        assert.equal(v.getUint8(p + 33), 2);
        assert.equal(v.getUint8(p + 34), (variadic ? 1 : 0) | (namePtr !== 0n ? 2 : 0));
        assert.equal(v.getUint8(p + 35), 0);
        assert.equal(v.getUint32(p + 36, true), 0x9abcdef0);
        const env = v.getUint32(p + 8, true);
        if (packedInfo === 0n) assert.equal(env, 0);
        else {
            assert.ok(env && (env & 7) === 0 && env + 8 + Number(packedInfo & 0xffffffffn) * 16 <= memory.buffer.byteLength);
            assert.equal(new DataView(memory.buffer).getBigUint64(env, true), packedInfo);
        }
    };
    const instance = await WebAssembly.instantiate(wasm, { env });
    const dv = () => new DataView(memory.buffer), out = 4096;
    const readBig = p => {
        const v = dv(), count = v.getUint32(p+4,true);
        assert.equal(v.getUint8(p-8),11); assert.ok(count > 0);
        assert.equal(v.getUint32(p-4,true),8+8*count);
        let n=0n; for(let i=count-1;i>=0;i--) n=(n<<64n)|v.getBigUint64(p+8+i*8,true);
        return v.getInt32(p,true) ? -n : n;
    };
    const decode = p => {
        const v=dv(), type=v.getUint8(p);
        assert.equal(v.getUint16(p+2,true),0);assert.equal(v.getUint32(p+4,true),0);
        if(type===1) {assert.equal(v.getUint8(p+1),0x10);return {n:v.getBigInt64(p+8,true),d:1n};}
        assert.equal(type,8);const q=Number(v.getBigUint64(p+8,true));
        assert.ok(q>=layout.heapFloor);assert.equal(q%8,0);
        const subtype=v.getUint8(q-8);
        if(subtype===11) {assert.equal(v.getUint8(p+1),0x10);return {n:readBig(q),d:1n};}
        assert.equal(subtype,19);assert.equal(v.getUint32(q-4,true),32);
        assert.equal(v.getUint8(p+1),0);assert.equal(v.getInt32(q+20,true),0);
        if(v.getInt32(q+16,true)) {
            assert.equal(v.getBigInt64(q,true),0n);assert.equal(v.getBigInt64(q+8,true),1n);
            return {n:readBig(v.getUint32(q+24,true)),d:readBig(v.getUint32(q+28,true))};
        }
        assert.equal(v.getUint32(q+24,true),0);assert.equal(v.getUint32(q+28,true),0);
        return {n:v.getBigInt64(q,true),d:v.getBigInt64(q+8,true)};
    };
    let conversions=0;
    const formatBuffer=env.arena_allocate_string_with_header(1,48);
    assert.equal(dv().getUint8(formatBuffer-8),1);
    assert.equal(dv().getUint32(formatBuffer-4,true),49);
    // Closure ABI controls: zero captures, metadata/name/variadic boundaries,
    // and a capture allocation that crosses the linear-memory growth boundary.
    const typeInfo = 0x9abcdef00201n;
    const zeroClosure = closure(env, 0x1234n, 0n, 0x5678n, typeInfo, 0x12345678n);
    assertClosure(zeroClosure, 0n, 1, 0x12345678n, false);
    const packedClosure = (0x3n << 32n) | 2n;
    const beforeGrowth = memory.buffer;
    const freeBeforeGrowth = memory.buffer.byteLength - rt._bumpPtr;
    rt._bump(freeBeforeGrowth - 48);
    const grownClosure = closure(env, 0x1234n, packedClosure, 0x5678n, typeInfo, 0x12345678n);
    assert.notEqual(memory.buffer, beforeGrowth, `${file}: closure capture allocation grows memory safely`);
    assertClosure(grownClosure, packedClosure, 0, 0x12345678n, false);
    const variadicPacked = (1n << 63n) | (0xffffn << 32n) | 1n;
    const variadicClosure = closure(env, 0x1234n, variadicPacked, 0x5678n, typeInfo, 0xfeedbeefn);
    assertClosure(variadicClosure, variadicPacked, 0, 0xfeedbeefn, true);
    // A fixed one-page memory cannot grow. The closure header fits exactly,
    // but its capture environment is refused; transaction rollback must leave
    // the bump pointer at the pre-call checkpoint.
    const limitedRuntime = load(file, className);
    if (className === 'EshkolRepl') limitedRuntime.memory = new WebAssembly.Memory({initial: 3, maximum: 3});
    else limitedRuntime._importedMemory = new WebAssembly.Memory({initial: 20, maximum: 20});
    limitedRuntime.prepareWasm(wasm);
    const limitedEnv = limitedRuntime.createImports().env;
    const limitedMemory = limitedRuntime.memory || limitedRuntime._importedMemory;
    limitedRuntime._bump(limitedMemory.buffer.byteLength - limitedRuntime._bumpPtr - 48);
    const checkpointClosure = limitedRuntime._bumpPtr;
    assert.throws(() => closure(limitedEnv, 0x1234n, packedClosure, 0x5678n, typeInfo, 0n), /arena exhausted/);
    assert.equal(limitedRuntime._bumpPtr, checkpointClosure, `${file}: failed closure allocation rolls back`);
    const check = d => {
        assert.equal(bits(Number(rt._exact.format({double:d}))),bits(d),'finite decimal readback preserves IEEE bits');
        assert.equal(env.eshkol_format_double(formatBuffer,48,d),rt._exact.format({double:d}).length);
        assert.equal(bits(Number(rt.readString(formatBuffer))),bits(d),'native buffer formatting readback preserves IEEE bits');
        new Uint8Array(memory.buffer,out-8,32).fill(0xa5);
        instance.instance.exports.convert(d);
        assert.deepEqual(decode(out),oracle(d),`${file} ${d}`);
        assert.equal(dv().getBigUint64(out-8,true),0xa5a5a5a5a5a5a5a5n);
        assert.equal(dv().getBigUint64(out+16,true),0xa5a5a5a5a5a5a5a5n);
        assert.equal(bits(rt._exact.toDouble(rt._exact.read(out))),bits(d===0?0:d));
        conversions++;
    };
    for(const d of [0,-0,1,-1,1.5,-1.5,0.1,-0.1,1e300,-1e300,Number.MAX_VALUE,-Number.MAX_VALUE,
        Number.MIN_VALUE,-Number.MIN_VALUE,1e-300,-1e-300,2**-1022,2**-62,2**-63,
        2**63,-(2**63),number(bits(2**63)-1n),number(bits(-(2**63))+1n),number((1n<<52n)-1n)]) check(d);
    // Every finite exponent class, several significands, both signs.
    for(let e=0;e<2047;e++) for(const fraction of [0n,1n,0x5555555555555n,0xfffffffffffffn])
        for(const sign of [0n,1n<<63n]) check(number(sign|(BigInt(e)<<52n)|fraction));
    let random=0x123456789abcdefn;
    for(let i=0;i<1024;i++) {random=BigInt.asUintN(64,random*6364136223846793005n+1442695040888963407n);if(((random>>52n)&2047n)!==2047n)check(number(random));}
    const slot=8192, other=8208, result=8224;
    env.eshkol_double_to_exact_tagged(1,0.1,slot);
    env.eshkol_double_to_exact_tagged(1,0.2,other);
    env.eshkol_rational_binary_tagged_ptr(1,slot,other,0,result);
    assert.deepEqual(decode(result),{n:10808639105689191n,d:36028797018963968n});
    env.eshkol_rational_binary_tagged_ptr(1,result,slot,1,result);
    assert.deepEqual(decode(result),oracle(0.2));
    env.eshkol_rational_binary_tagged_ptr(1,slot,other,3,result);
    assert.deepEqual(decode(result),{n:1n,d:2n});
    env.eshkol_rational_numerator_tagged(1,slot,result);assert.equal(decode(result).n,3602879701896397n);
    env.eshkol_rational_denominator_tagged(1,slot,result);assert.equal(decode(result).n,36028797018963968n);
    // Raw rational payload ABI, including results too wide for i64.
    for(const [n,d,expected] of [[-7n,2n,[-4n,-3n,-3n,-4n]], [5n,2n,[2n,3n,2n,2n]],
        [7n,2n,[3n,4n,3n,4n]], [(1n<<120n)+1n,2n,[1n<<119n,(1n<<119n)+1n,1n<<119n,1n<<119n]]]) {
        rt._exact.write(slot,{n,d});const raw=Number(dv().getBigUint64(slot+8,true));
        for(const [i,op] of ['floor','ceil','truncate','round'].entries()) {
            assert.equal(env[`eshkol_rational_${op}_tagged`].length,3);
            env[`eshkol_rational_${op}_tagged`](1,raw,result);assert.equal(decode(result).n,expected[i]);
            if(expected[i]>=-(1n<<63n)&&expected[i]<(1n<<63n)) assert.equal(env[`eshkol_rational_${op}`](raw),expected[i]);
        }
    }
    rt._exact.write(slot,{double:0.1});env.eshkol_rational_numerator_tagged(1,slot,result);
    assert.equal(dv().getUint8(result),2);assert.equal(dv().getFloat64(result+8,true),0.1);
    env.eshkol_rational_denominator_tagged(1,slot,result);assert.deepEqual(decode(result),{n:1n,d:1n});
    env.eshkol_double_to_exact_tagged(1,0.1,slot);
    env.eshkol_rational_compare_tagged_ptr(1,slot,other,0,result);assert.equal(dv().getUint8(result),3);assert.equal(dv().getBigInt64(result+8,true),1n);
    assert.equal(env.eshkol_is_rational_tagged_ptr(slot),1);
    assert.equal(env.eshkol_ad_point_is_exact_scalar(slot),1);
    assert.equal(env.eshkol_ad_point_is_exact_number(slot),1);
    assert.equal(env.eshkol_ad_point_to_double(slot),0.1);
    assert.equal(env.eshkol_ad_seed_to_double(slot,8304),0.1);assert.equal(dv().getInt32(8304,true),1);
    env.eshkol_double_to_exact_tagged(1,Number.MIN_VALUE,slot);
    env.eshkol_rational_denominator_tagged(1,slot,result);
    assert.equal(env.eshkol_is_bignum_tagged(result),1);assert.equal(decode(result).n,1n<<1074n);
    const minPtr=Number(dv().getBigUint64(slot+8,true));
    assert.equal(dv().getUint32(dv().getUint32(minPtr+28,true)+4,true),17);
    env.eshkol_rational_binary_tagged_ptr(1,slot,slot,2,result);
    assert.equal(decode(result).d,1n<<2148n);assert.equal(rt._exact.toDouble(rt._exact.read(result)),0);
    // Exact rational -> double ties at normal/subnormal boundaries, including
    // nearest-even and sign-preserving underflow of nonzero negative values.
    for(const [n,d,expected] of [[1n,1n<<1075n,0],[3n,1n<<1075n,number(2n)],[-1n,1n<<1075n,-0],
        [(1n<<53n)+1n,1n<<53n,1],[(1n<<53n)+3n,1n<<53n,number(bits(1)+2n)]]) {
        rt._exact.write(result,{n,d});assert.equal(bits(rt._exact.toDouble(rt._exact.read(result))),bits(expected));
    }
    env.eshkol_double_to_exact_tagged(1,-7,slot);env.eshkol_double_to_exact_tagged(1,3,other);
    for(const [op,n] of [[4,2n],[5,-2n],[6,-1n]]) {env.eshkol_bignum_binary_tagged(1,slot,other,op,result);assert.equal(decode(result).n,n);}
    env.eshkol_double_to_exact_tagged(1,2**63,slot);env.eshkol_double_to_exact_tagged(1,3,other);
    env.eshkol_bignum_pow_tagged(1,slot,other,result);assert.equal(decode(result).n,1n<<189n);
    env.eshkol_bignum_binary_tagged(1,slot,slot,0,result);assert.equal(decode(result).n,1n<<64n);
    env.eshkol_bignum_binary_tagged(1,slot,slot,1,result);assert.equal(decode(result).n,0n);
    rt._exact.write(slot,{n:4n,d:9n});env.eshkol_exact_sqrt_tagged(1,slot,99,result);
    assert.deepEqual(decode(result),{n:2n,d:3n});
    rt._exact.write(slot,{n:2n,d:1n});env.eshkol_exact_sqrt_tagged(1,slot,1.25,result);
    assert.equal(dv().getUint8(result),2);assert.equal(dv().getFloat64(result+8,true),1.25);
    rt._exact.write(slot,{n:8n,d:27n});rt._exact.write(other,{n:2n,d:3n});
    env.eshkol_exact_rational_pow_tagged(1,slot,other,99,result);assert.deepEqual(decode(result),{n:4n,d:9n});
    rt._exact.write(other,{n:-2n,d:3n});env.eshkol_exact_rational_pow_tagged(1,slot,other,99,result);
    assert.deepEqual(decode(result),{n:9n,d:4n});
    rt._exact.write(slot,{n:2n,d:1n});rt._exact.write(other,{n:1n,d:2n});
    env.eshkol_exact_rational_pow_tagged(1,slot,other,1.25,result);assert.equal(dv().getFloat64(result+8,true),1.25);
    rt._exact.write(slot,{double:1e308});rt._exact.write(other,{double:1.7e308});
    env.eshkol_bignum_binary_tagged(1,slot,other,4,result);assert.equal(dv().getFloat64(result+8,true),1e308);
    rt._exact.write(other,{double:0});env.eshkol_bignum_binary_tagged(1,slot,other,3,result);
    assert.equal(dv().getFloat64(result+8,true),Infinity);
    rt._exact.write(other,{n:0n,d:1n});assert.throws(()=>env.eshkol_bignum_binary_tagged(1,slot,other,3,result),/division by zero/);
    env.eshkol_double_to_exact_tagged(1,2**63,slot);
    const port=env.eshkol_open_output_string();env.eshkol_display_value_to_port(slot,port);
    assert.equal(rt.readString(env.eshkol_get_output_string(port)),'9223372036854775808');
    const oldBuffer=memory.buffer;rt._bump(oldBuffer.byteLength);
    assert.notEqual(memory.buffer,oldBuffer);assert.equal(decode(slot).n,1n<<63n);assert.equal(env.eshkol_bignum_to_double(Number(dv().getBigUint64(slot+8,true))),2**63);
    const tinyNegative=rt._exact.toDouble({n:-1n,d:1n<<2148n});
    const hugePositive=rt._exact.toDouble({n:1n<<2048n,d:1n});
    for(const [d,text] of [[0,'0'],[-0,'-0.0'],[3,'3'],[Infinity,'+inf.0'],[-Infinity,'-inf.0'],
        [NaN,'+nan.0'],[hugePositive,'+inf.0'],[-hugePositive,'-inf.0'],[tinyNegative,'-0.0']]) {
        rt._exact.write(result,{double:d});assert.equal(rt._exact.format(rt._exact.read(result)),text);
        assert.equal(rt.readString(rt._exact.string(rt._exact.format({double:d}))),text);
        new Uint8Array(memory.buffer,formatBuffer,48).fill(0xa5);
        assert.equal(env.eshkol_format_double(formatBuffer,48,d),text.length);
        assert.equal(rt.readString(formatBuffer),text);
        assert.equal(dv().getUint8(formatBuffer+text.length),0);
        assert.equal(dv().getUint8(formatBuffer+text.length+1),0xa5);
        const printed=[],original=console.log;
        try {console.log=value=>printed.push(value);env.eshkol_display_value(result);env.eshkol_write_value(result);env.eshkol_fprint_double(0,d);}
        finally {console.log=original;}
        assert.deepEqual(printed,[text,text,text]);
        const outputPort=env.eshkol_open_output_string();
        env.eshkol_display_value_to_port(result,outputPort);env.eshkol_write_value_to_port(result,outputPort);env.eshkol_fprint_double(outputPort,d);
        assert.equal(rt.readString(env.eshkol_get_output_string(outputPort)),text+text+text);
    }
    for(const capacity of [1,2,3,4]) {
        new Uint8Array(memory.buffer,formatBuffer,48).fill(0xa5);
        assert.equal(env.eshkol_format_double(formatBuffer,capacity,-0),4);
        assert.equal(rt.readString(formatBuffer),'-0.0'.slice(0,capacity-1));
        assert.equal(dv().getUint8(formatBuffer+capacity),0xa5);
    }
    assert.equal(env.eshkol_format_double(0,0,Infinity),0);
    assert.throws(()=>env.eshkol_format_double(0,1,0),/memory span/);
    assert.throws(()=>env.eshkol_format_double(formatBuffer,-1,0),/invalid wasm32/);
    assert.throws(()=>env.eshkol_format_double(memory.buffer.byteLength-1,2,0),/memory span/);
    assert.throws(()=>env.eshkol_fprint_double(123,0),/output stream/);
    for(const d of [Infinity,-Infinity,NaN]) {
        new Uint8Array(memory.buffer,result,16).fill(0xa5);const bump=rt._bumpPtr;
        assert.throws(()=>env.eshkol_double_to_exact_tagged(1,d,result),error=>error.code==='ESH_NUMERIC_DOMAIN'&&/no exact representation/.test(error.message));
        assert.equal(rt._bumpPtr,bump);assert.deepEqual([...new Uint8Array(memory.buffer,result,16)],Array(16).fill(0xa5));
    }
    env.eshkol_double_to_exact_tagged(1,0.1,slot);const p=Number(dv().getBigUint64(slot+8,true));
    dv().setInt32(p+16,2,true);assert.throws(()=>env.eshkol_rational_to_double(p),/discriminator/);
    assert.throws(()=>env.eshkol_double_to_exact_tagged(1,1,result+1),/memory span/);
    assert.throws(()=>env.eshkol_wasm_numeric_abi_check(8,0,4,8,40,0,8,16,20,24,32),/numeric ABI mismatch/);
    assert.throws(()=>rt._bump(-1),/invalid wasm32/);
    dv().setUint8(other,33); // Legacy string pointers never become integers.
    assert.equal(env.eshkol_ad_point_is_exact_number(other),0);
    assert.throws(()=>rt._exact.read(other),/real numeric/);
    assert.deepEqual(Object.keys(rt._exact.imports).sort(),Object.keys(abi).sort());
    assert.doesNotThrow(()=>env.eshkol_double_to_exact_tagged(1,NaN,0)); // native null-out no-op
    assert.throws(()=>env.eshkol_taylor_seed_tagged(1,slot,2,result),/Exact Taylor differentiation is unsupported/);
    new Uint8Array(memory.buffer,result,16).fill(0xa5);
    assert.throws(()=>env.eshkol_complex_pow(slot,other,result),e=>e.code==='ESH_NUMERIC_UNSUPPORTED');
    assert.throws(()=>env.eshkol_complex_sqrt(slot,result),e=>e.code==='ESH_NUMERIC_UNSUPPORTED');
    assert.deepEqual([...new Uint8Array(memory.buffer,result,16)],Array(16).fill(0xa5));
    assert.throws(()=>rt.prepareWasm(wasm),/live arena/);
    const fresh=load(file,className);const isolated=new WebAssembly.Memory({initial:32,maximum:32});
    fresh.memory=isolated;fresh.prepareWasm(wasm);const limited=fresh.createImports().env;
    fresh._bump(isolated.buffer.byteLength-fresh._bumpPtr-24);
    new Uint8Array(isolated.buffer,result,16).fill(0xa5);const checkpoint=fresh._bumpPtr;
    assert.throws(()=>limited.eshkol_double_to_exact_tagged(1,Number.MIN_VALUE,result),/arena exhausted/);
    assert.equal(fresh._bumpPtr,checkpoint);assert.deepEqual([...new Uint8Array(isolated.buffer,result,16)],Array(16).fill(0xa5));
    const elevated=load(file,className).prepareWasm(fixture(true));assert.equal(elevated.staticEnd,2097153);assert.ok(elevated.heapFloor>2097153);
    console.log(`${file}: ${conversions} exact finite conversions; list/string/memcmp imports, ABI/consumers/rounding/growth/corruption/exhaustion PASS`);
}
(async()=>{await run('web/eshkol-repl.js','EshkolRepl');await run('site/static/eshkol-runtime.js','EshkolRuntime');})().catch(e=>{console.error(e);process.exitCode=1;});
