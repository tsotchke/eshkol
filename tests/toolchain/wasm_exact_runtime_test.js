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
    console.log(`${file}: ${conversions} exact finite conversions; ABI/consumers/rounding/growth/corruption/exhaustion PASS`);
}
(async()=>{await run('web/eshkol-repl.js','EshkolRepl');await run('site/static/eshkol-runtime.js','EshkolRuntime');})().catch(e=>{console.error(e);process.exitCode=1;});
