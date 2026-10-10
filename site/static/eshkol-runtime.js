/**
 * Eshkol WASM Runtime — DOM bridge for Eshkol programs compiled to WebAssembly.
 *
 * Extracted from web/eshkol-repl.js (the browser REPL runtime).
 * Provides the handle-based DOM API that maps Eshkol's extern declarations
 * to JavaScript DOM operations.
 *
 * Handle system: integer handles → JavaScript objects
 *   0 = null, 1 = document, 2 = window, 3 = document.body
 */
// BEGIN GENERATED EXACT RUNTIME
// Canonical browser numeric/arena implementation. Embedded into both bundles.
// Heap objects use the C WASM32 ABI; BigInt is only arithmetic scratch state.
function createEshkolExactRuntime(memoryRef, owner, stackBytes) {
    const MIN = -(1n << 63n), MAX = (1n << 63n) - 1n;
    const MASK = (1n << 64n) - 1n;
    class NumericError extends RangeError {
        constructor(message, code = 'ESH_NUMERIC_DOMAIN') {
            super(message); this.name = 'EshkolNumericError'; this.code = code;
        }
    }
    const fail = (message, code) => { throw new NumericError(message, code); };
    const mem = () => memoryRef() || fail('missing WASM linear memory', 'ESH_NUMERIC_ABI');
    const view = () => new DataView(mem().buffer);
    const uint = (x) => {
        const n = Number(x);
        if (!Number.isSafeInteger(n) || n < 0 || n > 0xffffffff)
            fail('invalid wasm32 size or pointer', 'ESH_NUMERIC_ABI');
        return n;
    };
    const span = (p, n, alignment = 1) => {
        p = uint(p); n = uint(n);
        if (!p || p % alignment || p + n > mem().buffer.byteLength)
            fail('invalid WASM memory span', 'ESH_NUMERIC_ABI');
        return p;
    };
    const align = n => Math.ceil(n / 8) * 8;
    const ensure = end => {
        if (end > 0xffffffff) fail('WASM arena exhausted', 'ESH_NUMERIC_MEMORY');
        const m = mem();
        if (end > m.buffer.byteLength) {
            try { m.grow(Math.ceil((end - m.buffer.byteLength) / 65536)); }
            catch (_) { fail('WASM arena exhausted', 'ESH_NUMERIC_MEMORY'); }
        }
    };
    // Parse active data offsets before instantiation, before any import can
    // allocate. This raw-object lane has no wasm-ld exported __heap_base.
    function prepare(bytes) {
        const a = new Uint8Array(bytes);
        if (a.length < 8 || a[0] !== 0 || a[1] !== 97 || a[2] !== 115 || a[3] !== 109)
            fail('invalid WASM module header', 'ESH_NUMERIC_ABI');
        let p = 8, staticEnd = 1024;
        const byte = () => { if (p >= a.length) fail('truncated WASM module', 'ESH_NUMERIC_ABI'); return a[p++]; };
        const leb = (signed = false) => {
            let x = 0n, shift = 0n, b;
            do {
                b = byte(); x |= BigInt(b & 127) << shift; shift += 7n;
                if (shift > 35n) fail('invalid WASM offset', 'ESH_NUMERIC_ABI');
            } while (b & 128);
            if (signed && (b & 64)) x -= 1n << shift;
            return Number(x);
        };
        while (p < a.length) {
            const id = byte(), length = leb(), end = p + length;
            if (end > a.length) fail('truncated WASM section', 'ESH_NUMERIC_ABI');
            if (id === 11) {
                const count = leb();
                for (let i = 0; i < count; i++) {
                    const flags = leb();
                    let offset = null;
                    if (flags === 0 || flags === 2) {
                        if (flags === 2 && leb() !== 0) fail('multiple WASM memories unsupported', 'ESH_NUMERIC_ABI');
                        if (byte() !== 0x41) fail('data offset must be a constant', 'ESH_NUMERIC_ABI');
                        offset = uint(leb(true));
                        if (byte() !== 0x0b) fail('invalid data offset expression', 'ESH_NUMERIC_ABI');
                    } else if (flags !== 1) fail('invalid data segment flags', 'ESH_NUMERIC_ABI');
                    const size = leb();
                    if (offset !== null) staticEnd = Math.max(staticEnd, offset + size);
                    p += size;
                    if (p > end) fail('truncated data segment', 'ESH_NUMERIC_ABI');
                }
                if (p !== end) fail('invalid data section length', 'ESH_NUMERIC_ABI');
            }
            p = end;
        }
        const stackBegin = align(staticEnd), stackEnd = stackBegin + stackBytes;
        const floor = align(stackEnd);
        if (owner._exactPrepared && owner._bumpPtr > owner._heapFloor)
            fail('cannot instantiate another raw module over a live arena', 'ESH_NUMERIC_ABI');
        ensure(floor);
        owner._stackTop = stackEnd; owner._heapFloor = floor; owner._bumpPtr = floor; owner._exactPrepared = true;
        return { staticEnd, stackBegin, stackEnd, heapFloor: floor };
    }
    const allocate = size => {
        if (!owner._exactPrepared) fail('prepare WASM memory layout before allocation', 'ESH_NUMERIC_ABI');
        size = uint(size);
        const p = owner._bumpPtr, end = p + align(Math.max(size, 1));
        ensure(end); owner._bumpPtr = end;
        new Uint8Array(mem().buffer, p, end - p).fill(0);
        return p;
    };
    const header = (size, subtype, flags = 0) => {
        size = uint(size);
        const block = allocate(size + 8), v = view();
        v.setUint8(block, subtype); v.setUint8(block + 1, flags);
        v.setUint32(block + 4, size, true);
        return block + 8;
    };
    const transaction = fn => {
        const checkpoint = owner._bumpPtr;
        try { return fn(); }
        catch (error) { owner._bumpPtr = checkpoint; throw error; }
    };
    const payload = (p, subtype, minimum) => {
        p = span(p, minimum, 8); span(p - 8, 8);
        const v = view(), size = v.getUint32(p - 4, true);
        if (v.getUint8(p - 8) !== subtype || size < minimum)
            fail('invalid numeric object header', 'ESH_NUMERIC_ABI');
        span(p, size); return size;
    };
    const abs = x => x < 0n ? -x : x;
    const gcd = (a, b) => { a = abs(a); b = abs(b); while (b) [a, b] = [b, a % b]; return a; };
    const rational = (n, d = 1n) => {
        if (!d) fail('exact division by zero');
        if (d < 0n) { n = -n; d = -d; }
        const g = gcd(n, d); return { n: n / g, d: d / g };
    };
    function readBignum(p) {
        const size = payload(p, 11, 16), v = view();
        const sign = v.getInt32(p, true), count = v.getUint32(p + 4, true);
        if (sign < 0 || sign > 1 || !count || 8 + count * 8 !== size)
            fail('invalid bignum layout', 'ESH_NUMERIC_ABI');
        let n = 0n;
        for (let i = count - 1; i >= 0; i--) n = (n << 64n) | v.getBigUint64(p + 8 + 8 * i, true);
        if ((count > 1 && v.getBigUint64(p + 8 * count, true) === 0n) || (!n && sign))
            fail('noncanonical bignum', 'ESH_NUMERIC_ABI');
        return sign ? -n : n;
    }
    const writeBignum = n => {
        let magnitude = abs(n), limbs = [];
        do { limbs.push(magnitude & MASK); magnitude >>= 64n; } while (magnitude);
        const p = header(8 + limbs.length * 8, 11), v = view();
        v.setInt32(p, n < 0n ? 1 : 0, true); v.setUint32(p + 4, limbs.length, true);
        limbs.forEach((limb, i) => v.setBigUint64(p + 8 + 8 * i, limb, true));
        return p;
    };
    function readRational(p) {
        if (payload(p, 19, 32) !== 32) fail('invalid rational size', 'ESH_NUMERIC_ABI');
        const v = view(), big = v.getInt32(p + 16, true);
        let n, d;
        if (big === 0) {
            n = v.getBigInt64(p, true); d = v.getBigInt64(p + 8, true);
            if (v.getUint32(p + 24, true) || v.getUint32(p + 28, true)) fail('invalid small rational pointers', 'ESH_NUMERIC_ABI');
        } else if (big === 1) {
            if (v.getBigInt64(p, true) !== 0n || v.getBigInt64(p + 8, true) !== 1n)
                fail('invalid big rational inactive fields', 'ESH_NUMERIC_ABI');
            n = readBignum(v.getUint32(p + 24, true)); d = readBignum(v.getUint32(p + 28, true));
            if (n >= MIN && n <= MAX && d <= MAX) fail('noncanonical big rational', 'ESH_NUMERIC_ABI');
        } else fail('invalid rational discriminator', 'ESH_NUMERIC_ABI');
        if (v.getInt32(p + 20, true) !== 0 || d <= 0n || gcd(n, d) !== 1n)
            fail('noncanonical rational', 'ESH_NUMERIC_ABI');
        return { n, d };
    }
    const writeRational = r => {
        const big = r.n < MIN || r.n > MAX || r.d > MAX;
        const num = big ? writeBignum(r.n) : 0, den = big ? writeBignum(r.d) : 0;
        const p = header(32, 19), v = view();
        v.setBigInt64(p, big ? 0n : r.n, true); v.setBigInt64(p + 8, big ? 1n : r.d, true);
        v.setInt32(p + 16, big ? 1 : 0, true);
        v.setUint32(p + 24, num, true); v.setUint32(p + 28, den, true);
        return p;
    };
    const baseType = p => {
        const t = view().getUint8(span(p, 16, 8));
        // Folded numeric flags exist; legacy string tag33 is not an integer.
        return t === 0x11 ? 1 : t === 0x22 ? 2 : t;
    };
    const pointer = p => {
        const x = view().getBigUint64(span(p, 16, 8) + 8, true);
        if (x > 0xffffffffn) fail('numeric pointer exceeds wasm32', 'ESH_NUMERIC_ABI');
        return Number(x);
    };
    const subtype = p => { p = span(p, 1, 8); span(p - 8, 8); return view().getUint8(p - 8); };
    const isSubtype = (p, type) => baseType(p) === 8 && subtype(pointer(p)) === type;
    const read = p => {
        const type = baseType(p), v = view();
        if (type === 1) return { n: v.getBigInt64(Number(p) + 8, true), d: 1n };
        if (type === 2) return { double: v.getFloat64(Number(p) + 8, true) };
        if (type === 8) {
            const q = pointer(p), s = subtype(q);
            if (s === 11) return { n: readBignum(q), d: 1n };
            if (s === 19) return readRational(q);
        }
        fail('expected a real numeric value', 'ESH_NUMERIC_TYPE');
    };
    const write = (p, r) => {
        p = span(p, 16, 8);
        let type, flags, data;
        if ('double' in r) { type = 2; flags = 0x20; data = r.double; }
        else if (r.d === 1n && r.n >= MIN && r.n <= MAX) { type = 1; flags = 0x10; data = r.n; }
        else { type = 8; flags = r.d === 1n ? 0x10 : 0; data = BigInt(r.d === 1n ? writeBignum(r.n) : writeRational(r)); }
        const v = view(); new Uint8Array(mem().buffer, p, 16).fill(0);
        v.setUint8(p, type); v.setUint8(p + 1, flags);
        if (type === 2) v.setFloat64(p + 8, data, true);
        else if (type === 1) v.setBigInt64(p + 8, data, true);
        else v.setBigUint64(p + 8, data, true);
    };
    const output = (p, fn) => { span(p, 16, 8); return transaction(() => write(p, fn())); };
    const bits = new DataView(new ArrayBuffer(8));
    function fromDouble(d) {
        if (!Number.isFinite(d)) fail(`inexact->exact: no exact representation for ${Number.isNaN(d) ? '+nan.0' : d > 0 ? '+inf.0' : '-inf.0'}`);
        bits.setFloat64(0, d, true);
        const raw = bits.getBigUint64(0, true), E = Number((raw >> 52n) & 2047n);
        let m = raw & ((1n << 52n) - 1n), e = E ? E - 1075 : -1074;
        if (E) m |= 1n << 52n;
        if (!m) return { n: 0n, d: 1n };
        while ((m & 1n) === 0n) { m >>= 1n; e++; }
        if (raw >> 63n) m = -m;
        return e >= 0 ? { n: m << BigInt(e), d: 1n } : { n: m, d: 1n << BigInt(-e) };
    }
    const bitlen = n => n.toString(2).length;
    function toDouble(r) {
        if ('double' in r) return r.double;
        if (!r.n) return 0;
        const negative = r.n < 0n, n = abs(r.n), d = r.d;
        let e = bitlen(n) - bitlen(d);
        if (e >= 0 ? n < (d << BigInt(e)) : (n << BigInt(-e)) < d) e--;
        if (e > 1023) return negative ? -Infinity : Infinity;
        const shift = e < -1022 ? 1074 : 52 - e;
        const N = shift >= 0 ? n << BigInt(shift) : n;
        const D = shift >= 0 ? d : d << BigInt(-shift);
        let q = N / D;
        const twice = (N % D) * 2n;
        if (twice > D || (twice === D && (q & 1n))) q++;
        let raw;
        if (e < -1022) raw = q;
        else {
            if (q === (1n << 53n)) { q >>= 1n; e++; }
            raw = e > 1023 ? 0x7ff0000000000000n : (BigInt(e + 1023) << 52n) | (q - (1n << 52n));
        }
        if (negative) raw |= 1n << 63n;
        bits.setBigUint64(0, raw, true); return bits.getFloat64(0, true);
    }
    const integer = r => {
        if ('double' in r || r.d !== 1n) fail('expected an exact integer', 'ESH_NUMERIC_TYPE');
        return r.n;
    };
    const binary = (a, b, op) => {
        if (op === 7) return 'double' in a ? { double: -a.double } : { n: -a.n, d: a.d };
        if ('double' in a || 'double' in b) {
            const x = toDouble(a), y = toDouble(b);
            if ((op === 3 && !('double' in b) && b.n === 0n) || (op >= 4 && op <= 6 && y === 0)) fail('division by zero');
            const modulo = () => {
                let rem = x % y;
                if (rem !== 0 && (rem < 0) !== (y < 0)) rem += y;
                return rem;
            };
            const ops = [() => x + y, () => x - y, () => x * y, () => x / y,
                modulo, () => Math.trunc(x / y), () => x % y];
            if (!ops[op]) fail('invalid numeric operation', 'ESH_NUMERIC_ABI');
            return { double: ops[op]() };
        }
        if (op === 0) return rational(a.n * b.d + b.n * a.d, a.d * b.d);
        if (op === 1) return rational(a.n * b.d - b.n * a.d, a.d * b.d);
        if (op === 2) return rational(a.n * b.n, a.d * b.d);
        if (op === 3) return rational(a.n * b.d, a.d * b.n);
        const x = integer(a), y = integer(b);
        if (!y) fail('exact division by zero');
        if (op === 5) return rational(x / y);
        let r = x % y;
        if (op === 4 && r && (r < 0n) !== (y < 0n)) r += y;
        else if (op !== 4 && op !== 6) fail('invalid numeric operation', 'ESH_NUMERIC_ABI');
        return rational(r);
    };
    const boolean = (p, b) => {
        p = span(p, 16, 8); const v = view();
        new Uint8Array(mem().buffer, p, 16).fill(0); v.setUint8(p, 3); v.setBigInt64(p + 8, b ? 1n : 0n, true);
    };
    const compare = (a, b, op, out) => {
        if (op < 0 || op > 4) fail('invalid comparison operation', 'ESH_NUMERIC_ABI');
        let x, y;
        if ('double' in a || 'double' in b) { x = toDouble(a); y = toDouble(b); }
        else { x = a.n * b.d; y = b.n * a.d; }
        boolean(out, [x < y, x > y, x === y, x <= y, x >= y][op]);
    };
    const round = (r, op) => {
        const q = r.n / r.d, rem = r.n % r.d;
        if (op === 0) return q - (rem < 0n ? 1n : 0n);
        if (op === 1) return q + (rem > 0n ? 1n : 0n);
        if (op === 2) return q;
        const twice = abs(rem) * 2n;
        return q + ((twice > r.d || (twice === r.d && (abs(q) & 1n))) ? r.n < 0n ? -1n : 1n : 0n);
    };
    const pow = (a, e) => {
        if ('double' in a || 'double' in e || e.d !== 1n) return { double: Math.pow(toDouble(a), toDouble(e)) };
        let k = abs(e.n), n = 1n, d = 1n, N = a.n, D = a.d;
        while (k) { if (k & 1n) { n *= N; d *= D; } k >>= 1n; if (k) { N *= N; D *= D; } }
        return e.n < 0n ? rational(d, n) : rational(n, d);
    };
    // Integer root by binary search with bounded exponentiation. Comparison
    // stops once the trial power exceeds n, avoiding enormous dead products.
    const root = (n, degree) => {
        if (n < 0n || degree <= 0n) return null;
        if (n <= 1n || degree === 1n) return n;
        if (degree > BigInt(bitlen(n))) return null;
        let low = 1n, high = 1n << ((BigInt(bitlen(n)) + degree - 1n) / degree);
        const comparePower = base => {
            let value = 1n, k = degree;
            while (k) {
                if (k & 1n) { value *= base; if (value > n) return 1; }
                k >>= 1n;
                if (k) { base *= base; if (base > n) base = n + 1n; }
            }
            return value < n ? -1 : value > n ? 1 : 0;
        };
        while (low <= high) {
            const middle = (low + high) >> 1n, cmp = comparePower(middle);
            if (!cmp) return middle;
            if (cmp < 0) low = middle + 1n; else high = middle - 1n;
        }
        return null;
    };
    const exactRoot = (a, degree, exponent, fallback) => {
        if ('double' in a || a.n < 0n || (a.n === 0n && exponent < 0n)) return { double: fallback };
        const n = root(a.n, degree), d = root(a.d, degree);
        return n === null || d === null ? { double: fallback } : pow({ n, d }, { n: exponent, d: 1n });
    };
    const formatDouble = d => {
        if (Number.isNaN(d)) return '+nan.0';
        if (d === Infinity) return '+inf.0';
        if (d === -Infinity) return '-inf.0';
        if (Object.is(d, -0)) return '-0.0';
        // The engine supplies shortest round-trip finite decimal rendering.
        // Its decimal choice need not be byte-identical to native dtoa.
        return String(d);
    };
    const format = r => 'double' in r ? formatDouble(r.double) : r.d === 1n ? String(r.n) : `${r.n}/${r.d}`;
    const string = text => {
        const bytes = new TextEncoder().encode(text), p = header(bytes.length + 1, 1);
        new Uint8Array(mem().buffer, p, bytes.length).set(bytes); return p;
    };
    const numeric = p => {
        const t = baseType(p);
        return t === 1 || t === 2 || (t === 8 && [11, 19].includes(subtype(pointer(p))));
    };
    const displayAdNodePrimal = p => {
        if (baseType(p) !== 9) return null;
        const node = pointer(p);
        if (subtype(node) !== 2 || !adNodeSet().has(node)) return '#<ad-node>';
        span(node, 144, 8);
        const dense = adNodeShape(node);
        if (dense && dense.value && dense.shape.length === 1) {
            const values = new Array(dense.shape[0]);
            for (let i = 0; i < values.length; i++)
                values[i] = formatDouble(view().getFloat64(dense.value + i * 8, true));
            return `#(${values.join(' ')})`;
        }
        const exact = view().getUint32(node + 136, true);
        if (exact && numeric(exact)) return format(read(exact));
        return formatDouble(view().getFloat64(node + 8, true));
    };
    const display = (p, port) => {
        const isNumber = numeric(p);
        const adText = !isNumber ? displayAdNodePrimal(p) : null;
        if (!isNumber && adText === null && port === undefined) return;
        const text = isNumber ? format(read(p)) : adText === null ? String(p) : adText;
        const chunks = owner._stringPorts && owner._stringPorts.get(port);
        if (chunks) chunks.push(text); else console.log(text);
    };
    const adDouble = p => {
        if (numeric(p)) return toDouble(read(p));
        const type = baseType(p), v = view();
        if (type === 6) return v.getFloat64(span(pointer(p), 8, 8), true);
        if (type === 3 || type === 4) return Number(v.getBigInt64(Number(p) + 8, true));
        fail('AD point is not a numeric scalar', 'ESH_NUMERIC_TYPE');
    };
    // The LLVM tensor runtime stores a dual tensor as 16-byte tagged slots.
    // Its flat forward carrier is the complete eight-double square-free jet
    // (e1,e2,reverse-seed), not just the first two coefficients.
    const jetZero = value => [value, 0, 0, 0, 0, 0, 0, 0];
    const tensorMemoryCeiling = () => Number(owner._exactMemoryCeiling || mem().buffer.byteLength);
    const tensorResultBytes = (rank, total, kind = 'jet') => {
        const arenaBytes = 48 + align(rank * 8) + (kind === 'jet'
            ? align(Math.max(1, total) * 16) + total * 64 : align(total * 8));
        const bytes = arenaBytes + (kind === 'jet' ? total * 128 : 0);
        if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff ||
            !Number.isSafeInteger(owner._bumpPtr) || owner._bumpPtr + bytes > tensorMemoryCeiling())
            fail(kind === 'jet' ? 'tensor jet result exceeds browser memory ceiling' :
                'tensor allocation exceeds browser memory ceiling', 'ESH_NUMERIC_MEMORY');
    };
    const readCString = p => {
        p = uint(p);
        if (!p) return 'tensor arithmetic';
        const bytes = new Uint8Array(mem().buffer), end = Math.min(bytes.length, p + 1024);
        let stop = p;
        while (stop < end && bytes[stop] !== 0) stop++;
        return new TextDecoder().decode(bytes.subarray(p, stop)) || 'tensor arithmetic';
    };
    const jetRead = (p, tagged) => {
        p = span(p, tagged ? 16 : 8, 8);
        const v = view();
        if (!tagged) return { num: { double: v.getFloat64(p, true) } };
        const type = baseType(p);
        if (type === 6) {
            const q = v.getBigUint64(p + 8, true);
            if (!q || q > 0xffffffffn) fail('invalid dual tensor slot pointer', 'ESH_NUMERIC_ABI');
            const base = span(Number(q), 64, 8), out = new Array(8);
            const dv = view();
            for (let i = 0; i < 8; i++) out[i] = dv.getFloat64(base + 8 * i, true);
            return { jet: out };
        }
        if (type === 40) fail('reverse-mode nodes cannot enter forward-mode tensor arithmetic', 'ESH_AD_UNSUPPORTED');
        if (type === 8) {
            const q = v.getBigUint64(p + 8, true);
            if (!q || q > 0xffffffffn) fail('invalid heap tensor slot pointer', 'ESH_NUMERIC_ABI');
            const subtype = view().getUint8(span(Number(q) - 8, 8));
            if (subtype === 23) fail('Taylor towers are unsupported in browser tensor arithmetic', 'ESH_AD_UNSUPPORTED');
            if (subtype === 22) fail('reverse-mode nodes cannot enter forward-mode tensor arithmetic', 'ESH_AD_UNSUPPORTED');
            return { num: read(p) };
        }
        if (type === 1 || type === 2) return { num: read(p) };
        fail('unsupported tagged value in forward-mode tensor arithmetic', 'ESH_AD_UNSUPPORTED');
    };
    const jetFma = (a, ai, b, bi, c) => {
        if ((ai > 0 && a === 0) || (bi > 0 && b === 0)) return c;
        if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(c)) return NaN;
        if (Number.isFinite(a) && Number.isFinite(b) && !Number.isFinite(c)) return c;
        if (!Number.isFinite(a) || !Number.isFinite(b)) {
            if ((a === 0 && !Number.isFinite(b)) || (b === 0 && !Number.isFinite(a))) return NaN;
            const product = a * b;
            if (!Number.isFinite(c) && product === -c) return NaN;
            return product + c;
        }
        const product = binary(fromDouble(a), fromDouble(b), 2);
        const sum = binary(product, fromDouble(c), 0);
        if (sum.n === 0n && Object.is(a * b, -0) && Object.is(c, -0)) return -0;
        return toDouble(sum);
    };
    const jetBinary = (a, b, op) => {
        if (!a.jet && !b.jet) return { num: binary(a.num, b.num, op) };
        a = a.jet || jetZero(toDouble(a.num));
        b = b.jet || jetZero(toDouble(b.num));
        const r = new Array(8);
        if (op === 0 || op === 1) {
            for (let i = 0; i < 8; i++) r[i] = op === 0 ? a[i] + b[i] : a[i] - b[i];
        } else if (op === 2) {
            r[0] = a[0] * b[0];
            for (let s = 1; s < 8; s++) {
                let sum = 0;
                for (let t = 0; t < 8; t++) if ((t & s) === t)
                    sum = jetFma(a[t], t, b[s ^ t], s ^ t, sum);
                r[s] = sum;
            }
        } else if (op === 3) {
            for (let s = 0; s < 8; s++) {
                let sum = a[s];
                for (let t = 1; t < 8; t++) if ((t & s) === t)
                    sum = jetFma(-b[t], t, r[s ^ t], s ^ t, sum);
                r[s] = sum / b[0];
            }
        } else fail(`unsupported tensor jet arithmetic opcode ${op}`, 'ESH_AD_UNSUPPORTED');
        return { jet: r };
    };
    const tensorView = (p, legacy = false) => {
        p = span(p, 40, 8);
        if (!legacy) span(p - 8, 8);
        const v = view();
        if (!legacy && (v.getUint8(p - 8) !== 3 || v.getUint32(p - 4, true) < 40))
            fail('invalid tensor object in forward-mode arithmetic', 'ESH_NUMERIC_ABI');
        const rank64 = v.getBigUint64(p + 8, true), total64 = v.getBigUint64(p + 24, true);
        const dims = BigInt(v.getUint32(p, true)), elements = BigInt(v.getUint32(p + 16, true));
        const dtype = v.getBigUint64(p + 32, true);
        if (rank64 > 16n || total64 > 0xffffffffn || (rank64 && (!dims || dims > 0xffffffffn)) ||
            (total64 && !elements))
            fail('invalid tensor layout in forward-mode arithmetic', 'ESH_NUMERIC_ABI');
        const rank = Number(rank64), total = Number(total64), dv = view();
        const shape = new Array(rank); let product = 1;
        if (rank) span(Number(dims), rank * 8, 8);
        for (let i = 0; i < rank; i++) {
            const d = dv.getBigUint64(Number(dims) + 8 * i, true);
            if (d > 0xffffffffn) fail('tensor dimension exceeds browser limit', 'ESH_NUMERIC_ABI');
            shape[i] = Number(d); product *= shape[i];
            if (!Number.isSafeInteger(product) || product > 0xffffffff) fail('tensor element count exceeds browser limit', 'ESH_NUMERIC_ABI');
        }
        if (product !== total) fail('tensor shape and element count disagree', 'ESH_NUMERIC_ABI');
        const tagged = dtype === 64n || dtype === 65n;
        if (total) span(Number(elements), total * (tagged ? 16 : 8), 8);
        return { p, rank, total, shape, elements: Number(elements), tagged, dtype };
    };
    const tensorSlot = (t, i) => jetRead(t.elements + i * (t.tagged ? 16 : 8), t.tagged);
    const tensorResult = (shape, slots) => {
        const total = slots.length;
        let product = 1;
        for (const d of shape) {
            product *= d;
            if (!Number.isSafeInteger(product) || product > 0xffffffff) fail('tensor result exceeds browser limit', 'ESH_NUMERIC_MEMORY');
        }
        if (product !== total) fail('internal tensor result shape mismatch', 'ESH_NUMERIC_ABI');
        tensorResultBytes(shape.length, total);
        const result = header(40, 3), dims = shape.length ? allocate(shape.length * 8) : 0;
        const elements = allocate(Math.max(1, total) * 16);
        let v = view();
        v.setUint32(result, dims, true); v.setBigUint64(result + 8, BigInt(shape.length), true);
        v.setUint32(result + 16, elements, true); v.setBigUint64(result + 24, BigInt(total), true);
        v.setBigUint64(result + 32, 64n, true);
        for (let i = 0; i < shape.length; i++) { v = view(); v.setBigUint64(dims + 8 * i, BigInt(shape[i]), true); }
        for (let i = 0; i < total; i++) {
            const tagged = elements + i * 16, value = slots[i];
            if (value.jet) {
                const data = allocate(64);
                v = view(); v.setUint8(tagged, 6); v.setUint8(tagged + 1, 0x20); v.setBigUint64(tagged + 8, BigInt(data), true);
                for (let j = 0; j < 8; j++) { v = view(); v.setFloat64(data + j * 8, value.jet[j], true); }
            } else write(tagged, value.num);
        }
        return result;
    };
    const tensorAllocateFull = (rank, total) => transaction(() => {
        rank = uint(rank); total = uint(total);
        if (rank > 16) fail('tensor rank exceeds browser limit', 'ESH_NUMERIC_ABI');
        tensorResultBytes(rank, total, 'full');
        const result = header(40, 3), dims = rank ? allocate(rank * 8) : 0;
        const elements = total ? allocate(total * 8) : 0;
        const v = view();
        v.setUint32(result, dims, true); v.setBigUint64(result + 8, BigInt(rank), true);
        v.setUint32(result + 16, elements, true); v.setBigUint64(result + 24, BigInt(total), true);
        v.setBigUint64(result + 32, 0n, true);
        return result;
    });
    const tensorJetBinary = (arena, aPtr, bPtr, op, reverse, name) => transaction(() => {
        if (Number(reverse)) fail(`${readCString(name)} cannot combine forward and reverse tensor modes`, 'ESH_AD_UNSUPPORTED');
        if (![0, 1, 2, 3].includes(Number(op))) fail(`unsupported tensor jet arithmetic opcode ${op}`, 'ESH_AD_UNSUPPORTED');
        const a = tensorView(aPtr), b = tensorView(bPtr), rank = Math.max(a.rank, b.rank), shape = new Array(rank);
        for (let i = 0; i < rank; i++) {
            const ad = i < rank - a.rank ? 1 : a.shape[i - (rank - a.rank)];
            const bd = i < rank - b.rank ? 1 : b.shape[i - (rank - b.rank)];
            if (ad !== bd && ad !== 1 && bd !== 1) fail(`${readCString(name)}: tensor shapes cannot broadcast`, 'ESH_AD_SHAPE');
            shape[i] = ad === 1 ? bd : bd === 1 ? ad : ad;
        }
        let total = 1;
        for (const d of shape) total *= d;
        if (!shape.length) total = 1;
        if (!Number.isSafeInteger(total) || total > 0xffffffff) fail('tensor result exceeds browser limit', 'ESH_NUMERIC_MEMORY');
        tensorResultBytes(rank, total);
        const slots = new Array(total), outStrides = new Array(rank); let stride = 1;
        for (let i = rank - 1; i >= 0; i--) { outStrides[i] = stride; stride *= shape[i]; }
        for (let flat = 0; flat < total; flat++) {
            let ia = 0, ib = 0, sa = 1, sb = 1;
            for (let k = a.rank - 1, q = rank - 1; k >= 0; k--, q--) {
                const coord = Math.floor(flat / outStrides[q]) % shape[q];
                if (a.shape[k] !== 1) ia += coord * sa;
                sa *= a.shape[k];
            }
            for (let k = b.rank - 1, q = rank - 1; k >= 0; k--, q--) {
                const coord = Math.floor(flat / outStrides[q]) % shape[q];
                if (b.shape[k] !== 1) ib += coord * sb;
                sb *= b.shape[k];
            }
            slots[flat] = jetBinary(tensorSlot(a, ia), tensorSlot(b, ib), Number(op));
        }
        return tensorResult(shape, slots);
    });
    const tensorJetMatmul = (arena, aPtr, bPtr, reverse, name) => transaction(() => {
        if (Number(reverse)) fail(`${readCString(name)} cannot combine forward and reverse tensor modes`, 'ESH_AD_UNSUPPORTED');
        const a = tensorView(aPtr), b = tensorView(bPtr);
        if (a.rank !== 2 || b.rank !== 2 || a.shape[1] !== b.shape[0])
            fail(`${readCString(name)}: forward-mode (jet) matmul requires 2-D operands with A.cols == B.rows`, 'ESH_AD_SHAPE');
        const [m, k] = a.shape, n = b.shape[1], total = m * n;
        if (!Number.isSafeInteger(total) || total > 0xffffffff) fail('matmul result exceeds browser limit', 'ESH_NUMERIC_MEMORY');
        tensorResultBytes(2, total);
        const slots = new Array(total);
        for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
            let acc = { num: { double: 0 } };
            for (let q = 0; q < k; q++) {
                const prod = jetBinary(tensorSlot(a, i * k + q), tensorSlot(b, q * n + j), 2);
                acc = q === 0 ? prod : jetBinary(acc, prod, 0);
            }
            slots[i * n + j] = acc;
        }
        return tensorResult([m, n], slots);
    });
    const listToVectorSret = (outPtr, listPtr) => {
        outPtr = span(outPtr, 16, 8);
        let v = view();
        new Uint8Array(mem().buffer, outPtr, 16).fill(0);
        v.setUint8(outPtr, 0); // ESHKOL_VALUE_NULL until the vector is complete.
        const isCons = taggedPtr => {
            taggedPtr = span(taggedPtr, 16, 8);
            const tag = view().getUint8(taggedPtr);
            if (tag !== 8 && tag !== 32) return 0;
            const raw = view().getBigUint64(taggedPtr + 8, true);
            if (!raw || raw > 0xffffffffn) return 0;
            const cell = Number(raw);
            if (tag === 8 && subtype(cell) !== 0) return 0;
            span(cell, 32, 8);
            return cell;
        };
        return transaction(() => {
            let cur = listPtr ? span(listPtr, 16, 8) : 0;
            const ceiling = tensorMemoryCeiling();
            const available = Math.max(0, ceiling - owner._bumpPtr);
            const maxItems = Math.min((1 << 28) - 1, Math.floor(Math.max(0, available - 32) / 16));
            let capacity = 0;
            while (cur) {
                const cell = isCons(cur);
                if (!cell) break;
                if (capacity >= maxItems) fail('vector exceeds browser memory ceiling', 'ESH_NUMERIC_MEMORY');
                capacity++;
                cur = cell + 16;
            }
            const payloadBytes = 8 + capacity * 16;
            if (payloadBytes > 0xffffffff || capacity >= (1 << 28))
                fail('vector capacity exceeds browser limit', 'ESH_NUMERIC_MEMORY');
            const vector = header(payloadBytes, 2), dv = view();
            dv.setBigInt64(vector, BigInt(capacity), true);
            cur = listPtr ? span(listPtr, 16, 8) : 0;
            for (let i = 0; i < capacity; i++) {
                const source = isCons(cur), src = span(source, 16, 8), dst = vector + 8 + i * 16;
                const bytes = new Uint8Array(mem().buffer);
                bytes.copyWithin(dst, src, src + 16);
                cur = source + 16;
            }
            v = view(); v.setUint8(outPtr, 8); v.setUint8(outPtr + 1, 0);
            v.setBigUint64(outPtr + 8, BigInt(vector), true);
            return undefined;
        });
    };
    const tensorFromCollection = (_arena, inputPtr) => transaction(() => {
        inputPtr = span(inputPtr, 16, 8);
        const rootType = view().getUint8(inputPtr);
        if (rootType === 8) {
            const rootObject = pointer(inputPtr), rootSubtype = subtype(rootObject);
            if (rootSubtype === 3) return rootObject;
        }
        const maxList = Math.min((1 << 28) - 1,
            Math.max(1, Math.floor((tensorMemoryCeiling() - owner._bumpPtr) / 64)));
        const listCell = ref => {
            const tag = view().getUint8(ref);
            if (tag !== 8 && tag !== 32) return 0;
            const q = view().getBigUint64(ref + 8, true);
            if (!q || q > 0xffffffffn) return 0;
            const cell = Number(q);
            if (tag === 8 && subtype(cell) !== 0) return 0;
            span(cell, 32, 8); return cell;
        };
        const listLength = ref => {
            let cur = ref, n = 0;
            const seen = new Set();
            while (cur) {
                const cell = listCell(cur);
                if (!cell) break;
                if (seen.has(cell)) fail('cyclic list cannot form a tensor', 'ESH_AD_SHAPE');
                if (n >= maxList) fail('tensor collection exceeds browser memory ceiling', 'ESH_NUMERIC_MEMORY');
                seen.add(cell); n++; cur = cell + 16;
            }
            return n;
        };
        const kind = ref => {
            const tag = view().getUint8(ref);
            if (tag === 32) return listCell(ref) ? 'list' : 'leaf';
            if (tag !== 8) return 'leaf';
            const object = pointer(ref), sub = subtype(object);
            if (sub === 0) return 'list';
            if (sub === 2) return 'vector';
            if (sub === 3) return 'tensor';
            return 'leaf';
        };
        const vectorInfo = ref => {
            const object = pointer(ref), size = payload(object, 2, 8), v = view();
            const len = v.getBigInt64(object, true);
            if (len < 0n || len > BigInt((size - 8) / 16) || 8n + len * 16n !== BigInt(size))
                fail('invalid vector layout in tensor constructor', 'ESH_NUMERIC_ABI');
            return { object, length: Number(len) };
        };
        const childAt = (ref, index, parentKind) => {
            if (parentKind === 'list') {
                let cur = ref;
                for (let i = 0; i < index; i++) cur = listCell(cur) + 16;
                return listCell(cur);
            }
            const info = vectorInfo(ref);
            return info.object + 8 + index * 16;
        };
        const tensorAt = (ref, index) => {
            const t = tensorView(pointer(ref));
            const p = t.elements + index * (t.tagged ? 16 : 8);
            return t.tagged ? { tagged: p } : { raw: p };
        };
        const tensorShape = ref => tensorView(pointer(ref)).shape;
        const shape = [];
        const discover = (ref, depth) => {
            const k = kind(ref);
            if (k === 'leaf') return;
            if (k === 'tensor') {
                for (const dim of tensorShape(ref)) {
                    if (shape.length >= 8) fail('tensor: nested list/vector nests deeper than 8 dimensions', 'ESH_AD_SHAPE');
                    shape.push(dim);
                }
                return;
            }
            if (shape.length >= 8) fail('tensor: nested list/vector nests deeper than 8 dimensions', 'ESH_AD_SHAPE');
            const len = k === 'list' ? listLength(ref) : vectorInfo(ref).length;
            shape.push(len);
            if (len) discover(childAt(ref, 0, k), depth + 1);
        };
        const readLeaf = ref => {
            if (typeof ref === 'object' && ref.raw !== undefined)
                return { num: { double: view().getFloat64(span(ref.raw, 8, 8), true) } };
            const p = typeof ref === 'object' ? ref.tagged : ref;
            const t = baseType(p);
            if (t === 6) return jetRead(p, true);
            if (t === 40 || t === 9) fail('reverse-mode values are unsupported in browser tensor construction', 'ESH_AD_UNSUPPORTED');
            if (t === 8) {
                const q = pointer(p), sub = subtype(q);
                if (sub === 23) fail('Taylor towers are unsupported in browser tensor construction', 'ESH_AD_UNSUPPORTED');
                if (sub === 22) fail('reverse-mode values are unsupported in browser tensor construction', 'ESH_AD_UNSUPPORTED');
                return { num: read(p) };
            }
            if (t === 1 || t === 2) return { num: read(p) };
            fail('tensor: element is not a number', 'ESH_NUMERIC_TYPE');
        };
        if (kind(inputPtr) === 'leaf') shape.push(1);
        else discover(inputPtr, 0);
        let total = 1;
        for (const d of shape) {
            total *= d;
            if (!Number.isSafeInteger(total) || total > 0xffffffff)
                fail('tensor: nested list/vector shape exceeds browser limits', 'ESH_NUMERIC_MEMORY');
        }
        tensorResultBytes(shape.length, total);
        const values = new Array(total); let pos = 0, hasJet = false;
        const fill = (ref, level) => {
            const k = kind(ref);
            if (level === shape.length) {
                if (k !== 'leaf') fail('tensor: nested collection appears where a number was expected', 'ESH_AD_SHAPE');
                const value = readLeaf(ref); values[pos++] = value; hasJet ||= !!value.jet; return;
            }
            if (k === 'tensor') {
                const t = tensorView(pointer(ref)), remaining = shape.length - level;
                if (t.rank !== remaining || t.shape.some((d, i) => d !== shape[level + i]))
                    fail('tensor: nested tensor element does not have the same shape as its siblings', 'ESH_AD_SHAPE');
                for (let i = 0; i < t.total; i++) {
                    const value = readLeaf(tensorAt(ref, i)); values[pos++] = value; hasJet ||= !!value.jet;
                }
                return;
            }
            if (k !== 'list' && k !== 'vector') fail('tensor: a number appears where a sub-collection was expected', 'ESH_AD_SHAPE');
            const len = k === 'list' ? listLength(ref) : vectorInfo(ref).length;
            if (len !== shape[level]) fail('tensor: nested list/vector is ragged', 'ESH_AD_SHAPE');
            if (k === 'list') {
                let cur = ref;
                for (let i = 0; i < len; i++) {
                    const cell = listCell(cur);
                    fill(cell, level + 1);
                    cur = cell + 16;
                }
            } else {
                for (let i = 0; i < len; i++) fill(childAt(ref, i, k), level + 1);
            }
        };
        if (shape.length === 1 && kind(inputPtr) === 'leaf') {
            const value = readLeaf(inputPtr); values[0] = value; hasJet = !!value.jet;
        } else fill(inputPtr, 0);
        if (pos !== total && !(shape.length === 1 && kind(inputPtr) === 'leaf' && total === 1))
            fail('tensor: collection shape and element count disagree', 'ESH_AD_SHAPE');
        const result = header(40, 3), dims = shape.length ? allocate(shape.length * 8) : 0;
        const tagged = hasJet, elements = total ? allocate(total * (tagged ? 16 : 8)) : 0;
        let v = view(); v.setUint32(result, dims, true); v.setBigUint64(result + 8, BigInt(shape.length), true);
        v.setUint32(result + 16, elements, true); v.setBigUint64(result + 24, BigInt(total), true);
        v.setBigUint64(result + 32, tagged ? 64n : 0n, true);
        for (let i = 0; i < shape.length; i++) { v = view(); v.setBigUint64(dims + i * 8, BigInt(shape[i]), true); }
        for (let i = 0; i < total; i++) {
            const value = values[i];
            if (!tagged) { v = view(); v.setFloat64(elements + i * 8, toDouble(value.num), true); }
            else if (value.jet) {
                const slot = elements + i * 16, data = allocate(64);
                v = view(); v.setUint8(slot, 6); v.setUint8(slot + 1, 0x20); v.setBigUint64(slot + 8, BigInt(data), true);
                for (let j = 0; j < 8; j++) { v = view(); v.setFloat64(data + j * 8, value.jet[j], true); }
            } else write(elements + i * 16, { double: toDouble(value.num) });
        }
        return result;
    });
    const tensorOperandCarrierChecked = (valuePtr, opNamePtr) => {
        const name = readCString(opNamePtr);
        valuePtr = span(valuePtr, 16, 8);
        const tag = baseType(valuePtr);
        if (tag === 8 || tag === 33) {
            const object = pointer(valuePtr);
            const isLegacy = tag === 33;
            if (isLegacy) {
                const t = tensorView(object, true);
                if (t.dtype === 65n) fail(`${name}: expected numeric tensor`, 'ESH_NUMERIC_TYPE');
                return object;
            }
            const sub = subtype(object);
            if (sub === 3) {
                const t = tensorView(object);
                if (t.dtype === 65n) fail(`${name}: expected numeric tensor`, 'ESH_NUMERIC_TYPE');
                return object;
            }
            if (sub === 2 || sub === 0) {
                if (sub === 0) validateProperList(valuePtr);
                return tensorFromCollection(1, valuePtr);
            }
        } else if (tag === 9) {
            const object = pointer(valuePtr);
            if (subtype(object) === 2 && adNodeSet().has(object)) return denseNodeElements(object);
        } else if (tag === 32) {
            validateProperList(valuePtr);
            return tensorFromCollection(1, valuePtr);
        }
        fail(`${name}: expected tensor or numeric collection`, 'ESH_NUMERIC_TYPE');
    };
    const validateProperList = valuePtr => {
        let cur = span(valuePtr, 16, 8), count = 0;
        const seen = new Set();
        const max = Math.min((1 << 28) - 1,
            Math.max(1, Math.floor((tensorMemoryCeiling() - owner._bumpPtr) / 64)));
        while (true) {
            const tag = view().getUint8(cur);
            if (tag !== 8 && tag !== 32) {
                if (tag === 0) return;
                fail('tensor: list operand must be a proper numeric list', 'ESH_NUMERIC_TYPE');
            }
            const q = view().getBigUint64(cur + 8, true);
            if (!q || q > 0xffffffffn) fail('invalid list pointer in tensor operand', 'ESH_NUMERIC_ABI');
            const cell = Number(q);
            if (tag === 8 && subtype(cell) !== 0) fail('tensor: list operand must be a proper numeric list', 'ESH_NUMERIC_TYPE');
            span(cell, 32, 8);
            if (seen.has(cell)) fail('cyclic list cannot be a tensor operand', 'ESH_AD_SHAPE');
            if (count++ >= max) fail('tensor list exceeds browser memory ceiling', 'ESH_NUMERIC_MEMORY');
            seen.add(cell); cur = cell + 16;
        }
    };
    // Match TypeSystem's raw LLVM/WASM layout, whose size_t-shaped fields are
    // emitted as i64: 144-byte payload, tensor_value at40, shape at120, ndim
    // at128, exact_value at136. The separate wasm32 C runtime has a different
    // size_t layout and is not the producer of these browser-hosted nodes.
    const adNodeSet = () => owner._exactAdNodePtrs || (owner._exactAdNodePtrs = new Set());
    const adNodeAllocateRaw = () => {
        const ptr = allocate(144); adNodeSet().add(ptr); return ptr;
    };
    const adNodeAllocateWithHeader = () => {
        const ptr = header(144, 2); adNodeSet().add(ptr); return ptr;
    };
    const adNodeProbe = (_arena, bits, _expectType) => {
        const candidate = BigInt.asUintN(64, BigInt(bits));
        if (!candidate || candidate > 0xffffffffn) return 0;
        const ptr = Number(candidate);
        if (!adNodeSet().has(ptr)) return 0;
        fail('Reverse-mode AD nodes are unsupported in the browser WASM runtime', 'ESH_AD_UNSUPPORTED');
    };
    const adCopyShapeToHome = (shapePtr, ndimValue) => transaction(() => {
        const ndim = Number(BigInt(ndimValue));
        if (!shapePtr || ndim <= 0 || ndim > 16) return 0;
        shapePtr = span(shapePtr, ndim * 8, 8);
        const values = new Array(ndim), src = view();
        for (let i = 0; i < ndim; i++) values[i] = src.getBigInt64(shapePtr + i * 8, true);
        const copy = allocate(ndim * 8);
        for (let i = 0; i < ndim; i++) { const dst = view(); dst.setBigInt64(copy + i * 8, values[i], true); }
        return copy;
    });
    const adNodeShape = p => {
        if (!adNodeSet().has(p)) return null;
        span(p, 144, 8); span(p - 8, 8);
        const v = view();
        if (v.getUint8(p - 8) !== 2 || v.getUint32(p - 4, true) < 144)
            fail('invalid browser AD node object header', 'ESH_NUMERIC_ABI');
        const value = v.getUint32(p + 40, true), shapePtr = v.getUint32(p + 120, true);
        const rank64 = v.getBigUint64(p + 128, true);
        if (!value) return { value: 0, shape: [], total: 0 };
        if (!shapePtr || !rank64 || rank64 > 16n) fail('invalid dense AD tensor shape', 'ESH_NUMERIC_ABI');
        const rank = Number(rank64); span(shapePtr, rank * 8, 8);
        const shape = new Array(rank); let total = 1;
        for (let i = 0; i < rank; i++) {
            const d = view().getBigInt64(shapePtr + i * 8, true);
            if (d < 0n || d > 0xffffffffn) fail('invalid dense AD tensor dimension', 'ESH_NUMERIC_ABI');
            shape[i] = Number(d); total *= shape[i];
            if (!Number.isSafeInteger(total) || total > 0xffffffff) fail('dense AD tensor size overflows', 'ESH_NUMERIC_ABI');
        }
        span(value, total * 8, 8);
        return { value, shape, total };
    };
    const adNodeHasVariable = root => {
        const seen = new Set(), active = new Set(), visit = p => {
            if (!p) return false;
            if (active.has(p)) fail('cyclic AD dependency in browser tensor projection', 'ESH_AD_UNSUPPORTED');
            if (seen.has(p)) return false;
            if (!adNodeSet().has(p)) fail('untracked reverse-mode node in browser tensor path', 'ESH_AD_UNSUPPORTED');
            seen.add(p); span(p, 144, 8);
            const v = view(), type = v.getInt32(p, true);
            if (type === 0) return false;
            if (type === 1) return true;
            if (type !== 24 && type !== 67 && (type < 83 || type > 91))
                fail('unsupported reverse-mode node kind in browser tensor projection', 'ESH_AD_UNSUPPORTED');
            const children = [v.getUint32(p + 24, true), v.getUint32(p + 28, true),
                v.getUint32(p + 48, true), v.getUint32(p + 52, true)];
            // TENSOR_PACK's saved_tensors array is specifically the scalar AD
            // node for each tensor element. Other tensor operators save raw
            // value buffers and scratch tensors there, so treating every saved
            // pointer as an AD node fabricates dependencies.
            if (type === 83) {
                const saved = v.getUint32(p + 56, true), count64 = v.getBigUint64(p + 64, true);
                if (count64 > 0xffffffffn) fail('invalid AD node saved-value count', 'ESH_NUMERIC_ABI');
                const count = Number(count64);
                if (count) span(saved, count * 4, 4);
                for (let i = 0; i < count; i++) children.push(view().getUint32(saved + i * 4, true));
            }
            active.add(p);
            let hasChild = false;
            for (const child of children) if (child) { hasChild = true; if (visit(child)) return true; }
            active.delete(p);
            // TENSOR_PACK is also emitted as a plain primal carrier. With no
            // saved scalar dependencies it is a constant dense value, not a
            // reverse-mode variable. Other childless node kinds are opaque and
            // must not have their primal mistaken for a differentiable result.
            if (!hasChild) return type !== 83;
            return false;
        };
        return visit(root);
    };
    const denseNodeElements = nodeValue => transaction(() => {
        const p = uint(nodeValue);
        const t = adNodeShape(p);
        if (!t || !t.value || !t.shape.length)
            fail('dense reverse-mode tensor values are unavailable in browser WASM', 'ESH_AD_UNSUPPORTED');
        if (adNodeHasVariable(p))
            fail('reverse-mode tensor projection is unsupported in browser WASM', 'ESH_AD_UNSUPPORTED');
        tensorResultBytes(t.shape.length, t.total, 'full');
        const result = header(40, 3), dims = allocate(t.shape.length * 8);
        const elements = t.total ? allocate(t.total * 8) : 0;
        let v = view(); v.setUint32(result, dims, true); v.setBigUint64(result + 8, BigInt(t.shape.length), true);
        v.setUint32(result + 16, elements, true); v.setBigUint64(result + 24, BigInt(t.total), true);
        v.setBigUint64(result + 32, 0n, true);
        for (let i = 0; i < t.shape.length; i++) { v = view(); v.setBigUint64(dims + i * 8, BigInt(t.shape[i]), true); }
        for (let i = 0; i < t.total; i++) {
            const src = view().getFloat64(t.value + i * 8, true);
            view().setFloat64(elements + i * 8, src, true);
        }
        return result;
    });
    const adNodeTotalElements = nodeValue => {
        const p = uint(nodeValue);
        if (!p) return 0n;
        const t = adNodeShape(p);
        return t ? BigInt(t.total) : 0n;
    };
    const tapeSet = () => owner._exactTapes || (owner._exactTapes = new Map());
    const tapeAllocate = (_arena, capacityValue) => {
        let capacity = Number(capacityValue);
        if (!Number.isSafeInteger(capacity) || capacity < 0) fail('invalid WASM tape capacity', 'ESH_NUMERIC_ABI');
        if (!capacity) capacity = 64;
        const ptr = allocate(40);
        tapeSet().set(ptr, { capacity, nodes: [] });
        return ptr;
    };
    const tapeAddNode = (tapeValue, nodeValue) => {
        const tape = uint(tapeValue), node = uint(nodeValue);
        if (!tape || !node) return tapeSet().get(tape)?.nodes.length || 0;
        let state = tapeSet().get(tape);
        // Some lite builds use the nonzero current-tape sentinel before a real
        // arena tape is materialized. Keep its node stream in host-owned storage.
        if (!state) { state = { capacity: 0, nodes: [] }; tapeSet().set(tape, state); }
        state.nodes.push(node);
        return state.nodes.length;
    };
    const tapeReset = tapeValue => {
        const tape = uint(tapeValue), state = tapeSet().get(tape);
        if (state) state.nodes.length = 0;
    };
    const tapeGetNode = (tapeValue, indexValue) => {
        const tape = uint(tapeValue), index = Number(indexValue), state = tapeSet().get(tape);
        if (!state || !Number.isSafeInteger(index) || index < 0 || index >= state.nodes.length) return 0;
        return state.nodes[index];
    };
    const tapeGetNodeCount = tapeValue => tapeSet().get(uint(tapeValue))?.nodes.length || 0;
    const consAllocateWithHeader = (_arena) => header(32, 0);
    const vectorAllocateWithHeader = (_arena, capacityValue) => transaction(() => {
        const capacity = uint(capacityValue);
        if (capacity >= (1 << 28)) fail('vector capacity exceeds browser limit', 'ESH_NUMERIC_MEMORY');
        const bytes = 8 + capacity * 16;
        if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff || owner._bumpPtr + 8 + align(bytes) > tensorMemoryCeiling())
            fail('vector allocation exceeds browser memory ceiling', 'ESH_NUMERIC_MEMORY');
        return header(bytes, 2);
    });
    const multiValueAllocate = (_arena, countValue) => transaction(() => {
        const count = uint(countValue), bytes = 8 + count * 16;
        if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff)
            fail('multiple-values allocation exceeds browser limit', 'ESH_NUMERIC_MEMORY');
        // The raw LLVM lane loads an i64 count followed by tagged slots.
        const result = header(bytes, 4);
        view().setBigUint64(result, BigInt(count), true);
        return result;
    });
    const taggedIndex = p => {
        p = span(p, 16, 8);
        const type = baseType(p), v = view();
        if (type === 2) {
            const n = v.getFloat64(p + 8, true);
            if (!Number.isFinite(n) || n < -9223372036854775808 || n >= 9223372036854775808)
                return -(1n << 63n);
            return BigInt(Math.trunc(n));
        }
        return v.getBigInt64(p + 8, true);
    };
    const consCellForIndex = p => {
        p = span(p, 16, 8);
        if (baseType(p) !== 8) return 0;
        const q = pointer(p);
        return q && subtype(q) === 0 ? q : 0;
    };
    const unwrapListIndex = p => {
        if (!p) return 0n;
        const cell = consCellForIndex(p);
        return taggedIndex(cell || p);
    };
    const tensorLinearIndex = (idxPtr, shape) => {
        if (!idxPtr) return 0n;
        const first = consCellForIndex(idxPtr);
        if (!first) return taggedIndex(idxPtr);
        const I64_MIN = -(1n << 63n), I64_MAX = (1n << 63n) - 1n;
        let linear = 0n, count = 0, cur = idxPtr;
        while (true) {
            const cell = consCellForIndex(cur);
            if (!cell) break;
            if (count >= shape.length) return I64_MIN;
            const value = taggedIndex(cell), dim = BigInt(shape[count]);
            if (value < 0n || value >= dim) return I64_MIN;
            if (count === 0) linear = value;
            else {
                if (dim !== 0n && linear > (I64_MAX - value) / dim) return I64_MIN;
                linear = linear * dim + value;
            }
            count++; cur = cell + 16;
        }
        if (consCellForIndex(cur)) return I64_MIN;
        for (let i = count; i < shape.length; i++) {
            const dim = BigInt(shape[i]);
            if (dim !== 0n && linear > I64_MAX / dim) return I64_MIN;
            linear *= dim;
        }
        return linear;
    };
    const vrefUnwrapIndex = (vecPtr, idxPtr) => {
        if (!vecPtr || !idxPtr) return unwrapListIndex(idxPtr);
        if (baseType(vecPtr) === 8) {
            const object = pointer(vecPtr);
            if (subtype(object) === 3) return tensorLinearIndex(idxPtr, tensorView(object).shape);
        }
        return unwrapListIndex(idxPtr);
    };
    const memoryCopy = (dstValue, srcValue, lengthValue) => {
        const dst = uint(dstValue), src = uint(srcValue), length = uint(lengthValue);
        if (!length) return dst;
        const source = span(src, length), target = span(dst, length);
        new Uint8Array(mem().buffer).copyWithin(target, source, source + length);
        return dst;
    };
    const memorySet = (dstValue, value, lengthValue) => {
        const dst = uint(dstValue), length = uint(lengthValue);
        if (!length) return dst;
        const target = span(dst, length);
        new Uint8Array(mem().buffer).fill(Number(value) & 0xff, target, target + length);
        return dst;
    };
    const numericGeometry = [8, 0, 4, 8, 32, 0, 8, 16, 20, 24, 28];
    const imports = {
        eshkol_jet_tensor_binary: tensorJetBinary,
        eshkol_jet_tensor_matmul: tensorJetMatmul,
        eshkol_ad_node_probe: adNodeProbe,
        eshkol_ad_dense_node_elements: denseNodeElements,
        eshkol_ad_copy_shape_to_home: adCopyShapeToHome,
        eshkol_ad_node_total_elements: adNodeTotalElements,
        eshkol_list_to_vector_sret: listToVectorSret,
        eshkol_tensor_from_collection: tensorFromCollection,
        eshkol_tensor_operand_carrier_checked: tensorOperandCarrierChecked,
        arena_allocate_tape: tapeAllocate,
        arena_allocate_cons_with_header: consAllocateWithHeader,
        arena_allocate_vector_with_header: vectorAllocateWithHeader,
        arena_allocate_multi_value: multiValueAllocate,
        arena_tape_add_node: tapeAddNode,
        arena_tape_reset: tapeReset,
        arena_tape_get_node: tapeGetNode,
        arena_tape_get_node_count: tapeGetNodeCount,
        eshkol_unwrap_list_index: unwrapListIndex,
        eshkol_vref_unwrap_index: vrefUnwrapIndex,
        memcpy: memoryCopy,
        memmove: memoryCopy,
        memset: memorySet,
        eshkol_format_double: (buffer, capacity, d) => {
            capacity = uint(capacity);
            // Native dtoa_shortest returns0 for cap0 without touching buf.
            if (!capacity) return 0;
            buffer = span(buffer, capacity);
            const text = formatDouble(d), bytes = new TextEncoder().encode(text);
            const written = Math.min(bytes.length, capacity - 1);
            const target = new Uint8Array(mem().buffer, buffer, capacity);
            target.set(bytes.subarray(0, written)); target[written] = 0;
            return bytes.length;
        },
        eshkol_fprint_double: (file, d) => {
            file = uint(file);
            const text = formatDouble(d);
            if (!file) { console.log(text); return; }
            const chunks = owner._stringPorts && owner._stringPorts.get(file);
            if (!chunks) fail('invalid or unsupported WASM output stream', 'ESH_NUMERIC_ABI');
            chunks.push(text);
        },
        eshkol_complex_pow: (_a, _b, _out) => fail('Complex exponentiation is unsupported in the browser LLVM/WASM lane', 'ESH_NUMERIC_UNSUPPORTED'),
        eshkol_complex_sqrt: (_in, _out) => fail('Complex square root is unsupported in the browser LLVM/WASM lane', 'ESH_NUMERIC_UNSUPPORTED'),
        eshkol_wasm_numeric_abi_check: (...actual) => {
            if (actual.length !== numericGeometry.length || actual.some((n, i) => Number(n) !== numericGeometry[i]))
                fail('Eshkol WASM numeric ABI mismatch', 'ESH_NUMERIC_ABI');
        },
        eshkol_double_to_exact_tagged: (_arena, d, out) => { if (out) output(out, () => fromDouble(d)); },
        eshkol_double_to_rational: (_arena, d) => transaction(() => writeRational(fromDouble(d))),
        eshkol_bignum_from_int64: (_arena, n) => writeBignum(BigInt(n)),
        eshkol_bignum_from_overflow: (_arena, a, b, op) => {
            a = BigInt(a); b = BigInt(b);
            if (op < 0 || op > 2) fail('invalid overflow operation', 'ESH_NUMERIC_ABI');
            return writeBignum([a + b, a - b, a * b][op]);
        },
        eshkol_bignum_to_double: p => toDouble({ n: readBignum(p), d: 1n }),
        eshkol_bignum_to_string: (_arena, p) => string(String(readBignum(p))),
        eshkol_bignum_is_zero: p => readBignum(p) === 0n ? 1 : 0,
        eshkol_bignum_is_even: p => (readBignum(p) & 1n) === 0n ? 1 : 0,
        eshkol_bignum_is_odd: p => (readBignum(p) & 1n) === 1n ? 1 : 0,
        eshkol_bignum_neg: (_arena, p) => writeBignum(-readBignum(p)),
        eshkol_is_bignum_tagged: p => isSubtype(p, 11) ? 1 : 0,
        eshkol_is_rational_tagged_ptr: p => isSubtype(p, 19) ? 1 : 0,
        eshkol_bignum_binary_tagged: (_arena, a, b, op, out) => output(out, () => binary(read(a), op === 7 ? null : read(b), op)),
        eshkol_bignum_compare_tagged: (a, b, op, out) => compare(read(a), read(b), op, out),
        eshkol_rational_create: (_arena, n, d) => transaction(() => writeRational(rational(BigInt(n), BigInt(d)))),
        eshkol_rational_make_tagged: (_arena, n, d, out) => output(out, () => rational(integer(read(n)), integer(read(d)))),
        eshkol_rational_from_bignums_tagged: (_arena, n, d, out) => output(out, () => rational(readBignum(n), readBignum(d))),
        eshkol_rational_to_double: p => toDouble(readRational(p)),
        eshkol_rational_to_string: (_arena, p) => string(format(readRational(p))),
        eshkol_rational_binary_tagged_ptr: (_arena, a, b, op, out) => output(out, () => binary(read(a), read(b), op)),
        eshkol_rational_compare_tagged_ptr: (_arena, a, b, op, out) => compare(read(a), read(b), op, out),
        eshkol_rational_numerator_tagged: (_arena, p, out) => output(out, () => {
            const r = read(p); return 'double' in r ? r : rational(r.n);
        }),
        eshkol_rational_denominator_tagged: (_arena, p, out) => output(out, () => {
            const r = read(p); return rational('double' in r ? 1n : r.d);
        }),
        eshkol_exact_sqrt_tagged: (_arena, p, fallback, out) => {
            if (out) output(out, () => exactRoot(read(p), 2n, 1n, fallback));
        },
        eshkol_exact_rational_pow_tagged: (_arena, p, e, fallback, out) => {
            if (out) output(out, () => {
                const exponent = read(e);
                return 'double' in exponent || !isSubtype(e, 19) ? { double: fallback }
                    : exactRoot(read(p), exponent.d, exponent.n, fallback);
            });
        },
        eshkol_bignum_pow_tagged: (_arena, a, e, out) => output(out, () => pow(read(a), read(e))),
        eshkol_rational_pow_tagged: (_arena, a, e, out) => output(out, () => pow(read(a), read(e))),
        eshkol_rational_equal: (a, b) => { const x = readRational(a), y = readRational(b); return x.n === y.n && x.d === y.d ? 1 : 0; },
        eshkol_display_value: p => display(p),
        eshkol_write_value: p => display(p),
        eshkol_display_value_to_port: (p, port) => display(p, port),
        eshkol_write_value_to_port: (p, port) => display(p, port),
        eshkol_ad_point_to_double: (p, _what) => adDouble(p),
        eshkol_ad_seed_to_double: (p, ok) => {
            span(ok, 4, 4);
            const t = baseType(p), valid = numeric(p) || [3, 4, 6].includes(t);
            const d = valid ? adDouble(p) : 0;
            view().setInt32(Number(ok), valid ? 1 : 0, true); return d;
        },
        eshkol_ad_point_is_scalar: p => numeric(p) || [3, 4, 6].includes(baseType(p)) ? 1 : 0,
        eshkol_ad_point_is_exact_scalar: p => baseType(p) === 8 && numeric(p) ? 1 : 0,
        eshkol_ad_point_is_exact_number: p => numeric(p) && baseType(p) !== 2 ? 1 : 0,
    };
    for (const [name, op] of [['floor', 0], ['ceil', 1], ['truncate', 2], ['round', 3]]) {
        imports[`eshkol_rational_${name}`] = p => {
            const n = round(readRational(p), op);
            if (n < MIN || n > MAX) fail('rational integer result requires tagged ABI', 'ESH_NUMERIC_ABI');
            return n;
        };
        imports[`eshkol_rational_${name}_tagged`] = (_arena, p, out) => output(out, () => rational(round(readRational(p), op)));
    }
    return { imports, prepare, allocate, header, read, readBignum, readRational,
        fromDouble, toDouble, write: (p, r) => output(p, () => r), format,
        tensorJetBinary, tensorJetMatmul, tensorAllocateFull,
        adNodeAllocateRaw, adNodeAllocateWithHeader,
        NumericError, transaction, string, span };
}
// END GENERATED EXACT RUNTIME

class EshkolRuntime {
    constructor() {
        // Handle system
        this.handles = new Map();
        this.nextHandle = 1;
        this.documentHandle = this.createHandle(document);       // 1
        this.windowHandle = this.createHandle(window);           // 2
        this.bodyHandle = this.createHandle(document.body);      // 3

        // WASM instance (set after instantiation)
        this.instance = null;
        this.memory = null;
    }

    // A header-prefixed string buffer of `len` bytes plus the terminator;
    // the one place the JS string allocation knows the header's size.
    _allocString(len) {
        return this._numeric().header(len + 1, 1);
    }

    _numeric() {
        if (!this.memory && !this._importedMemory)
            this._importedMemory = new WebAssembly.Memory({ initial: 256, maximum: 1024 });
        this._exactMemoryCeiling = 1024 * 65536;
        return this._exact || (this._exact = createEshkolExactRuntime(
            () => this.memory || this._importedMemory, this, 1048576));
    }

    prepareWasm(bytes) { return this._numeric().prepare(bytes); }

    _bump(size) { return this._numeric().allocate(size); }


    setInstance(instance) {
        this.instance = instance;
        this.memory = instance.exports.memory || this._importedMemory;
    }

    wrapInstance(instance) {
        const G = (typeof globalThis !== 'undefined') && globalThis.EshkolWebGPU;
        const exports = G && typeof G.promisingExports === 'function'
            ? G.promisingExports(instance.exports) : instance.exports;
        return { rawInstance: instance, exports };
    }

    // Export facades do not wrap function-table entries. Browser callbacks
    // therefore apply the JSPI promising wrapper at the callback boundary.
    invokeWasmCallback(callbackFuncPtr, ...args) {
        const table = this.instance?.exports?.__indirect_function_table;
        const fn = table && table.get(callbackFuncPtr);
        if (typeof fn !== 'function') throw new Error('missing WASM callback ' + callbackFuncPtr);
        const G = (typeof globalThis !== 'undefined') && globalThis.EshkolWebGPU;
        const entry = G && typeof G.promisingTableEntry === 'function'
            ? G.promisingTableEntry(table, callbackFuncPtr)
            : G && typeof G.promisingEntry === 'function'
            ? G.promisingEntry(fn) : fn;
        return entry(...args);
    }

    // === Handle System ===

    createHandle(obj) {
        if (obj === null || obj === undefined) return 0;
        const handle = this.nextHandle++;
        this.handles.set(handle, obj);
        return handle;
    }

    getHandle(handle) {
        if (handle === 0) return null;
        return this.handles.get(handle) || null;
    }

    releaseHandle(handle) {
        if (handle > 3) this.handles.delete(handle);
    }

    // === String Helpers ===

    readString(ptr) {
        const mem = this.memory || this._importedMemory;
        if (!ptr || !mem) return '';
        const bytes = new Uint8Array(mem.buffer);
        let end = ptr;
        while (end < bytes.length && bytes[end] !== 0) end++;
        return new TextDecoder().decode(bytes.slice(ptr, end));
    }

    writeString(ptr, str, maxLen) {
        const mem = this.memory || this._importedMemory;
        if (!ptr || !mem) return 0;
        const bytes = new TextEncoder().encode(str);
        const view = new Uint8Array(mem.buffer);
        const len = Math.min(bytes.length, maxLen - 1);
        view.set(bytes.subarray(0, len), ptr);
        view[ptr + len] = 0;
        return len;
    }

    // Write a tagged #f into a struct-return out-parameter slot.
    //
    // The tagged value layout is {u8 type, u8 flags, u16 reserved, u32 pad,
    // u64 data} = 16 bytes, and type 3 with data 0 is #f (see
    // SYS_TYPE_BOOL in lib/core/system_builtins.c). Older degradation stubs in
    // this file are written `() => {}`, which leaves the caller's slot holding
    // whatever was on the stack — fine for a value nobody reads, wrong for one
    // the program branches on. Anything that must FAIL CLOSED in the browser
    // writes a real #f instead.
    writeFalse(ptr) {
        const mem = this.memory || this._importedMemory;
        if (!ptr || !mem) return;
        const view = new Uint8Array(mem.buffer, ptr, 16);
        view.fill(0);
        view[0] = 3;
    }

    // === Markdown Renderer ===
    // Lightweight markdown-to-HTML converter (no external dependencies)

    // Single-pass Scheme/Eshkol tokenizer used by the markdown highlighter.
    //
    // History note: an earlier version did syntax highlighting by chaining
    // .replace() calls — first wrapping comments in a span, then strings,
    // then numbers, etc. That was self-corrupting because each later regex
    // happily matched substrings the earlier passes had ALREADY inserted
    // into the output. The string regex matched the literal "color:#606078"
    // attribute the comment pass had just added, and the number regex then
    // matched "606078" inside *that* attribute, producing
    //
    //     <span style=<span ...>"color:#<span ...>606078</span>"</span>>...
    //
    // which the browser then displayed as raw text, leaving things like
    //     "color:#606078">; Integers are exact
    // visible in the rendered docs.
    //
    // The fix is to walk the source ONCE, recognise each token (comment,
    // string, number, hash literal, keyword, identifier, other), and emit
    // its HTML in one shot. Once a region of the source has been emitted
    // as a span, it cannot be re-tokenised, so the cross-corruption is
    // architecturally impossible.
    //
    // Returns an HTML string with all `<`, `>`, `&` properly escaped.
    static highlightScheme(source) {
        const KEYWORDS = new Set([
            'define','lambda','let','let*','letrec','letrec*','if','cond','begin',
            'set!','when','unless','do','case','match','and','or','not','quote',
            'quasiquote','unquote','unquote-splicing','require','provide','extern',
            'gradient','derivative','jacobian','hessian','divergence','curl','laplacian',
        ]);
        const COL_COMMENT = '#606078';
        const COL_KEYWORD = '#c084fc';
        const COL_STRING  = '#a78bfa';
        const COL_NUMBER  = '#00ff88';
        const escape = (s) =>
            s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const span = (color, text) =>
            `<span style="color:${color}">${escape(text)}</span>`;
        const isDigit  = (c) => c >= '0' && c <= '9';
        const isIdent  = (c) =>
            (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            '!$%&*+-./:<=>?@^_~'.indexOf(c) !== -1;
        const isIdentStart = (c) =>
            (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            '!$%&*+-./:<=>?@^_~'.indexOf(c) !== -1;
        let out = '';
        let i = 0;
        while (i < source.length) {
            const c = source[i];

            // Line comment: ; ... \n
            if (c === ';') {
                let j = i;
                while (j < source.length && source[j] !== '\n') j++;
                out += span(COL_COMMENT, source.slice(i, j));
                i = j;
                continue;
            }

            // String literal: "...", honoring \" escapes
            if (c === '"') {
                let j = i + 1;
                while (j < source.length) {
                    if (source[j] === '\\' && j + 1 < source.length) { j += 2; continue; }
                    if (source[j] === '"') { j++; break; }
                    j++;
                }
                out += span(COL_STRING, source.slice(i, j));
                i = j;
                continue;
            }

            // Hash literal: #t, #f, #\char
            if (c === '#' && i + 1 < source.length) {
                const n = source[i + 1];
                if (n === 't' || n === 'f') {
                    out += span(COL_NUMBER, source.slice(i, i + 2));
                    i += 2;
                    continue;
                }
                if (n === '\\' && i + 2 < source.length) {
                    out += span(COL_NUMBER, source.slice(i, i + 3));
                    i += 3;
                    continue;
                }
            }

            // Number literal: digits with optional fraction
            // Treats a leading '-' or '+' as numeric only when followed by a digit
            // and the previous emitted character is whitespace or an opening paren,
            // so things like (- x 1) keep the '-' as an identifier/operator.
            if (isDigit(c) || ((c === '-' || c === '+') && i + 1 < source.length && isDigit(source[i + 1]))) {
                let j = i + 1;
                let sawDot = false;
                while (j < source.length) {
                    const cj = source[j];
                    if (isDigit(cj)) { j++; continue; }
                    if (cj === '.' && !sawDot) { sawDot = true; j++; continue; }
                    break;
                }
                out += span(COL_NUMBER, source.slice(i, j));
                i = j;
                continue;
            }

            // Identifier or keyword
            if (isIdentStart(c)) {
                let j = i + 1;
                while (j < source.length && isIdent(source[j])) j++;
                const word = source.slice(i, j);
                if (KEYWORDS.has(word)) {
                    out += span(COL_KEYWORD, word);
                } else {
                    out += escape(word);
                }
                i = j;
                continue;
            }

            // Anything else (parens, whitespace, punctuation) — just escape
            out += escape(c);
            i++;
        }
        return out;
    }

    // ── Documentation pages ────────────────────────────────────────
    // Published pages are HTML fragments under /content/, generated from the
    // docs tree by scripts/build-site-content.sh (site/pages.json is the
    // declared list). A view names its page in the URL fragment as
    // "#page=<slug>" or "#page=<slug>&<heading-id>", so every page has a
    // shareable address while the pathname router keeps seeing "/docs" or
    // "/tutorials". Any other fragment is a heading id in the default page.
    static contentRequest(hash) {
        const match = /^#page=([a-z0-9][a-z0-9_]*)(?:&(.*))?$/.exec(hash || '');
        if (!match) return { url: null, anchor: decodeURIComponent((hash || '').slice(1)) };
        return { url: '/content/' + match[1] + '.html', anchor: decodeURIComponent(match[2] || '') };
    }

    // Load `url` into `target`. `.html` URLs are generated fragments and are
    // inserted as they are (Scheme code blocks are highlighted here, because
    // the fragments are rendered without a highlighter); anything else is
    // fetched as Markdown and rendered client-side. When `url` is a view's
    // default page and the address names another page, that page is loaded.
    loadContent(url, target, options) {
        const isDefault = !(options && options.explicit);
        const isNav = /\/nav-[a-z0-9_]+\.html$/.test(url);
        const request = EshkolRuntime.contentRequest(window.location.hash);
        if (isDefault && !isNav && request.url) url = request.url;
        if (!isNav) this._contentUrl = url;
        const markActive = () => {
            document.querySelectorAll('.docs-sidebar-item[data-url]').forEach((el) => {
                el.classList.toggle('active', el.getAttribute('data-url') === this._contentUrl);
            });
        };
        target.innerHTML = '<p style="color:#606078;font-style:italic">Loading...</p>';
        return fetch(url).then((r) => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.text();
        }).then((text) => {
            if (!isNav && this._contentUrl !== url) return;  // superseded by a later click
            if (/\.html$/.test(url)) {
                target.innerHTML = text;
                const SCHEME_LANGS = ['scheme', 'eshkol', 'lisp', 'scm'];
                target.querySelectorAll('pre > code').forEach((code) => {
                    const classes = (code.parentElement.className + ' ' + code.className).split(/\s+/);
                    if (classes.some((name) => SCHEME_LANGS.includes(name))) {
                        code.innerHTML = EshkolRuntime.highlightScheme(code.textContent);
                    }
                });
            } else {
                target.innerHTML = this.renderMarkdown(text);
            }
            markActive();
            if (isNav) return;
            // The anchor only exists after this async load, so scroll now.
            const anchor = request.anchor ? document.getElementById(request.anchor) : null;
            if (anchor) anchor.scrollIntoView();
            else if (!isDefault) window.scrollTo(0, 0);
        }).catch((e) => {
            target.innerHTML = '<p style="color:#ff4444">Failed to load: ' + e.message + '</p>';
        });
    }

    renderMarkdown(md) {
        let html = md;
        // Extract code blocks FIRST to protect them from markdown transforms
        const codeBlocks = [];
        const SCHEME_LANGS = new Set(['scheme', 'eshkol', 'lisp', 'scm', '']);
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            let body;
            if (SCHEME_LANGS.has(lang)) {
                // Single-pass tokeniser handles its own escaping.
                body = EshkolRuntime.highlightScheme(code);
            } else {
                // Other languages — just escape, no highlighting.
                body = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            }
            const block = `<pre style="background:#0a0a14;border:1px solid #27272a;border-radius:8px;padding:16px 20px;overflow-x:auto;margin:1em 0"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;line-height:1.6;color:#d4d4d8">${body}</code></pre>`;
            codeBlocks.push(block);
            return `\x00CODEBLOCK${codeBlocks.length - 1}\x00`;
        });
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code style="background:#0d0d15;padding:2px 6px;border-radius:4px;font-size:0.85em;color:#a78bfa">$1</code>');
        // Headers — each carries a GitHub-style slug id so in-document TOC
        // links and /docs#section deep links have anchors to land on.
        const usedSlugs = new Map();
        const headingId = (text) => {
            let slug = text
                .replace(/<[^>]+>/g, '')
                .toLowerCase()
                .replace(/[^a-z0-9\s_-]/g, '')
                .trim()
                .replace(/[\s_]+/g, '-');
            const seen = usedSlugs.get(slug);
            usedSlugs.set(slug, (seen || 0) + 1);
            if (seen) slug = `${slug}-${seen}`;
            return slug;
        };
        const HEADING_STYLES = {
            1: 'margin-top:2em;margin-bottom:0.5em;font-size:2rem;color:#e8e8f0;border-bottom:1px solid #2a2a3a;padding-bottom:0.3em',
            2: 'margin-top:2.5em;margin-bottom:0.5em;font-size:1.5rem;color:#e8e8f0;border-bottom:1px solid #2a2a3a;padding-bottom:0.3em',
            3: 'margin-top:2em;margin-bottom:0.5em;font-size:1.2rem;color:#e8e8f0',
            4: 'margin-top:1.8em;margin-bottom:0.4em;font-size:1rem;color:#a0a0b8',
            5: 'margin-top:1.5em;margin-bottom:0.3em;font-size:0.9rem;color:#a0a0b8',
            6: 'margin-top:1.5em;margin-bottom:0.3em;font-size:0.85rem;color:#a0a0b8',
        };
        for (let level = 6; level >= 1; level--) {
            const re = new RegExp(`^#{${level}}\\s+(.+)$`, 'gm');
            html = html.replace(re, (_, text) =>
                `<h${level} id="${headingId(text)}" style="${HEADING_STYLES[level]}">${text}</h${level}>`);
        }
        // Bold and italic
        html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, href) =>
            href.startsWith('#')
                ? `<a href="${href}" style="color:#a78bfa">${text}</a>`
                : `<a href="${href}" target="_blank" style="color:#a78bfa">${text}</a>`);
        // Images (just show as links)
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<em>[$1]</em>');
        // Blockquotes
        html = html.replace(/^>\s+(.+)$/gm, '<blockquote style="border-left:3px solid #7c3aed;padding-left:16px;margin:1em 0;color:#a0a0b8">$1</blockquote>');
        // Horizontal rules
        html = html.replace(/^---+$/gm, '<hr style="border:none;border-top:1px solid #2a2a3a;margin:2em 0">');
        // Tables
        html = html.replace(/^\|(.+)\|$/gm, (line) => {
            const cells = line.split('|').filter(c => c.trim());
            if (cells.every(c => /^[-:\s]+$/.test(c))) return ''; // separator row
            const tag = 'td';
            const cellHtml = cells.map(c => `<${tag} style="border:1px solid #2a2a3a;padding:8px 12px;color:#a0a0b8">${c.trim()}</${tag}>`).join('');
            return `<tr>${cellHtml}</tr>`;
        });
        html = html.replace(/(<tr>[\s\S]*?<\/tr>\n?)+/g, m => `<table style="width:100%;border-collapse:collapse;margin:1em 0">${m}</table>`);
        // Unordered lists
        html = html.replace(/^[\s]*[-*]\s+(.+)$/gm, '<li style="color:#a0a0b8;margin:0.2em 0">$1</li>');
        html = html.replace(/(<li[\s\S]*?<\/li>\n?)+/g, m => `<ul style="margin:0.8em 0;padding-left:1.5em">${m}</ul>`);
        // Ordered lists
        html = html.replace(/^\d+\.\s+(.+)$/gm, '<li style="color:#a0a0b8;margin:0.2em 0">$1</li>');
        // Paragraphs (lines not already tagged — skip code block placeholders)
        html = html.replace(/^(?!<[huplbtdoa]|<\/|<hr|<code|<str|<em>|\x00|$)(.+)$/gm, '<p style="margin:0.6em 0;color:#a0a0b8;line-height:1.7">$1</p>');
        // Restore code blocks from placeholders
        html = html.replace(/\x00CODEBLOCK(\d+)\x00/g, (_, idx) => codeBlocks[parseInt(idx)]);
        // Clean up double-spaced lines in pre blocks (ASCII art, etc.)
        html = html.replace(/<\/p>\n<p/g, '</p><p');
        return html;
    }

    // === WebGPU compute backend ===

    /**
     * Acquire a WebGPU device, if this browser has one. Must be awaited
     * BEFORE createImports() — the GPU env entries are built once, and
     * whether they are GPU-backed or CPU-backed is decided at that moment.
     *
     * Resolves to a result object rather than throwing: no WebGPU, no JSPI,
     * or no adapter all mean "run on the CPU", never "fail to load".
     */
    async initWebGPU(opts) {
        const G = (typeof globalThis !== 'undefined') && globalThis.EshkolWebGPU;
        if (!G) {
            this._webgpuStatus = { ok: false, reason: 'eshkol-webgpu.js not loaded' };
            return this._webgpuStatus;
        }
        if (!G.jspiAvailable()) {
            // WebGPU readback is async and the wasm runtime is synchronous;
            // without JSPI there is no way to suspend, so the CPU path is the
            // only correct one. Say so rather than pretending the GPU ran.
            this._webgpuStatus = { ok: false, reason: 'JSPI unavailable (WebAssembly.Suspending missing)' };
            return this._webgpuStatus;
        }
        const res = await G.create(opts || {});
        this._webgpuBackend = res.ok ? res.backend : null;
        this.__gpuEnv = null;
        // The Emscripten-built VM shares this backend through
        // EshkolWebGPU.attachVm(module, runtime.webgpuBackend) (ADR-0029).
        this._webgpuStatus = res;
        return res;
    }

    /** The WebGPU backend object, or null when running on the CPU path. */
    get webgpuBackend() { return this._webgpuBackend || null; }

    /**
     * Build (once) the GPU/tensor env entries. Uses the shared WebGPU module
     * when it is present, and a compact CPU implementation otherwise, so the
     * loader keeps working if eshkol-webgpu.js is not on the page.
     */
    _gpuEnv() {
        if (this.__gpuEnv) return this.__gpuEnv;
        const rt = this;
        const memRef = () => rt.memory || rt._importedMemory;
        const G = (typeof globalThis !== 'undefined') && globalThis.EshkolWebGPU;
        this.__gpuEnv = G
            ? G.makeImports(rt._webgpuBackend || null, memRef)
            : EshkolRuntime._cpuOnlyGpuEnv(memRef);
        return this.__gpuEnv;
    }

    /** CPU-only GPU env, used when eshkol-webgpu.js is absent. */
    static _cpuOnlyGpuEnv(memRef) {
        const f64 = (p, n) => new Float64Array(memRef().buffer, p, n);
        return {
            eshkol_matmul_dispatch: (aP, bP, cP, M, K, N) => {
                M = Number(M); K = Number(K); N = Number(N);
                const A = f64(aP, M * K), B = f64(bP, K * N), C = f64(cP, M * N);
                for (let i = 0; i < M; i++)
                    for (let j = 0; j < N; j++) {
                        let s = 0;
                        for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
                        C[i * N + j] = s;
                    }
            },
            eshkol_batch_matmul_dispatch: (aP, bP, cP, batch, M, K, N) => {
                batch = Number(batch); M = Number(M); K = Number(K); N = Number(N);
                const aStride = M * K, bStride = K * N, cStride = M * N;
                for (let q = 0; q < batch; q++) {
                    const A = f64(aP + q * aStride * 8, aStride);
                    const B = f64(bP + q * bStride * 8, bStride);
                    const C = f64(cP + q * cStride * 8, cStride);
                    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
                        let s = 0;
                        for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
                        C[i * N + j] = s;
                    }
                }
            },
            eshkol_gpu_elementwise_f64: () => -1,
            eshkol_gpu_reduce_f64: () => -1,
            eshkol_gpu_init: () => 0,
            eshkol_gpu_shutdown: () => {},
            eshkol_gpu_get_backend: () => 0,
            eshkol_gpu_backend_available: () => 0,
            eshkol_gpu_supports_f64: () => 0,
            eshkol_gpu_has_fp64: () => 0,
            eshkol_gpu_should_use: () => 0,
            eshkol_gpu_set_threshold: () => {},
            eshkol_gpu_get_threshold: () => 100000
        };
    }

    // === WASM Imports ===

    createImports() {
        const rt = this;
        const wasmAbiGeometry = Object.freeze({
            abiVersion: 1,
            pointerWidth: 4,
            objectHeaderSize: 8,
            objectHeaderAlign: 4,
            objectPayloadAlign: 8,
            objectSubtypeOffset: 0,
            objectFlagsOffset: 1,
            objectRefCountOffset: 2,
            objectSizeOffset: 4,
            objectLayoutIdOffset: 0xffffffff,
            objectIdOffset: 0xffffffff,
            objectHomeOffset: 0xffffffff,
            objectAuxOffset: 0xffffffff,
            taggedValueSize: 16,
            taggedValueAlign: 8,
            taggedValueTypeOffset: 0,
            taggedValueFlagsOffset: 1,
            taggedValueReservedOffset: 2,
            taggedValueDataOffset: 8,
            taggedValuePaddingOffset: 4,
        });
        const checkWasmAbiGeometry = (...actual) => {
            const fields = Object.keys(wasmAbiGeometry);
            if (actual.length !== fields.length) {
                throw new Error(`Eshkol WASM object ABI mismatch: expected ${fields.length} geometry values, got ${actual.length}`);
            }
            for (let i = 0; i < fields.length; i++) {
                const expected = wasmAbiGeometry[fields[i]] >>> 0;
                const got = Number(actual[i]) >>> 0;
                if (got !== expected) {
                    throw new Error(`Eshkol WASM object ABI mismatch: ${fields[i]} expected ${expected}, got ${got}`);
                }
            }
        };
        if (!rt._importedMemory) {
            rt._importedMemory = new WebAssembly.Memory({ initial: 256, maximum: 1024 });
        }
        const exact = this._numeric();
        const gpu = rt._gpuEnv();
        return {
            env: {
                // Memory
                __linear_memory: rt._importedMemory,
                // WASM linker globals (required by LLVM static relocation)
                __stack_pointer: new WebAssembly.Global({ value: 'i32', mutable: true }, rt._stackTop || 1048576), // 1MB stack
                __indirect_function_table: new WebAssembly.Table({ initial: 256, element: 'anyfunc' }),

                // Bump allocator (all arena functions use this)
                // All params may be BigInt (i64) — convert with Number()
                arena_create: () => 1,
                arena_create_threadsafe: () => 1,
                get_global_arena: () => 1,
                // OALR memctx accessor (ESH-0001 Phase A / #239): generated code
                // now imports the current thread-arena accessor. No region system
                // in the browser, so it degrades to the same fake arena as
                // get_global_arena. Without this stub, a site WASM rebuilt with a
                // post-#239 compiler fails to instantiate (stuck at "Loading").
                eshkol_current_arena: () => 1,
                // Literal constants materialized at run time live in a dedicated arena on
                // native builds; the lite runtime has one arena, so it is the same handle.
                eshkol_literal_arena: () => 1,
                eshkol_memctx_current: () => 1,
                eshkol_wasm_abi_check: (...geometry) => checkWasmAbiGeometry(...geometry),
                arena_destroy: () => {},
                arena_allocate: (arena, size) => { return rt._bump(Number(size)); },
                arena_allocate_with_header: (_arena, size, subtype, flags) => exact.header(Number(size), subtype, flags),
                arena_push_scope: () => {},
                arena_pop_scope: () => {},
                // Named-let TCO loop per-iteration arena scope reclamation
                // (ESH-0214b / fix/loop-arena-reclamation) -- rt is a bump
                // allocator with no reclamation, so this is a no-op, same
                // as arena_push_scope/arena_pop_scope above. SW-164 added a
                // LOOP scope outside the per-iteration one and a distinct
                // end-of-loop entry point; both are no-ops for the same reason.
                //
                // These three are the only arena imports that may WRITE to the
                // caller's memory: on the native runtime an escaping back edge
                // promotes the loop-carried values out of the span it rewinds,
                // rewrites them in `vals` in place, and the generated code reads
                // the array back and stores what it finds into the loop's
                // parameter slots. Doing nothing is the correct implementation
                // of that contract here, because nothing is ever reclaimed: the
                // caller's own values are still in the array and still live, so
                // the read-back returns exactly what it wrote. A stub that wrote
                // into the array, or that returned a value instead of void,
                // would hand the next iteration a null accumulator.
                eshkol_arena_iter_scope_end: () => {},
                eshkol_arena_iter_scope_finish: () => {},
                eshkol_arena_loop_scope_begin: () => {},
                arena_allocate_cons_cell: () => rt._bump(32),
                arena_allocate_tagged_cons_cell: () => rt._bump(48),
                eshkol_tagged_cons_set_tagged_value: () => {},
                arena_tagged_cons_set_ptr: () => {},
                arena_tagged_cons_set_null: () => {},
                arena_tagged_cons_set_int64: () => {},
                arena_allocate_tensor_with_header: () => exact.header(40, 3),
                arena_allocate_tensor_full: (_arena, ndim, total) => exact.tensorAllocateFull(ndim, total),
                arena_allocate_ad_node: () => exact.adNodeAllocateRaw(),
                arena_allocate_ad_node_with_header: () => exact.adNodeAllocateWithHeader(),
                arena_hash_table_create_with_header: () => rt._bump(32) + 8,
                arena_allocate_string_with_header: (arena, len) => rt._allocString(Number(len)),
                // (make-string k [char]): the rules of the native runtime's
                // eshkol_make_string_checked -- k an exact non-negative
                // integer (tag 1), the fill a character (tag 4) written as
                // UTF-8, a space when the fill pointer is null.
                eshkol_make_string_checked: (arena, kPtr, fillPtr) => {
                    const dv = new DataView(rt.memory.buffer);
                    const base = (t) => (t >= 8 ? t : (t & 0x0F));
                    const kType = base(dv.getUint8(Number(kPtr)));
                    const k = dv.getBigInt64(Number(kPtr) + 8, true);
                    if (kType !== 1 || k < 0n) throw new Error('Type error in make-string: expected non-negative exact integer');
                    let cp = 32;
                    if (Number(fillPtr) !== 0) {
                        const fType = base(dv.getUint8(Number(fillPtr)));
                        const c = Number(dv.getBigInt64(Number(fillPtr) + 8, true));
                        if (fType !== 4 || c < 0 || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF))
                            throw new Error('Type error in make-string: expected character');
                        cp = c;
                    }
                    const enc = new TextEncoder().encode(String.fromCodePoint(cp));
                    const count = Number(k);
                    const len = count * enc.length;
                    const buf = rt._allocString(len);
                    const mem = new Uint8Array(rt.memory.buffer);
                    for (let i = 0; i < count; i++) mem.set(enc, Number(buf) + i * enc.length);
                    mem[Number(buf) + len] = 0;
                    return buf;
                },
                arena_allocate_closure_with_header: (_arena, funcPtr, packedInfo, sexprPtr, returnTypeInfo, namePtr) => {
                    return exact.transaction(() => {
                        const captures = Number(BigInt(packedInfo) & 0xffffffffn);
                        const closure = exact.header(40, captures === 0 ? 1 : 0, 0);
                        let env = 0;
                        if (captures > 0) {
                            env = rt._bump(8 + captures * 16);
                            const envView = new DataView((rt.memory || rt._importedMemory).buffer);
                            envView.setBigUint64(env, BigInt(packedInfo), true);
                        }
                        // _bump may grow linear memory; reacquire the view
                        // after every allocation before touching the object.
                        const dv = new DataView((rt.memory || rt._importedMemory).buffer);
                        dv.setBigUint64(closure, BigInt(funcPtr), true);
                        dv.setUint32(closure + 8, env, true);
                        dv.setBigUint64(closure + 16, BigInt(sexprPtr), true);
                        dv.setUint32(closure + 24, Number(namePtr) >>> 0, true);
                        dv.setUint8(closure + 32, Number(BigInt(returnTypeInfo) & 0xffn));
                        dv.setUint8(closure + 33, Number((BigInt(returnTypeInfo) >> 8n) & 0xffn));
                        dv.setUint8(closure + 34, ((BigInt(packedInfo) >> 63n) ? 1 : 0) | (BigInt(namePtr) !== 0n ? 2 : 0));
                        dv.setUint8(closure + 35, 0);
                        dv.setUint32(closure + 36, Number((BigInt(returnTypeInfo) >> 16n) & 0xffffffffn), true);
                        return closure;
                    });
                },
                arena_tagged_cons_get_int64: () => 0n,
                arena_tagged_cons_get_double: () => 0.0,
                arena_tagged_cons_get_ptr: () => 0n,
                arena_tagged_cons_get_type: () => 0,
                arena_tagged_cons_get_flags: () => 0,
                arena_tagged_cons_set_double: () => {},

                // Runtime functions
                __eshkol_register_parallel_workers: () => {},
                // Quantum-RNG builtins degrade to Math.random in the browser
                // (kept in sync with web/eshkol-repl.js so either glue satisfies
                // a WASM built with the QRNG surface — architecture-model
                // wasm-import-glue-equality invariant).
                eshkol_qrng_double: Math.random,
                eshkol_qrng_uint64: () => BigInt(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)),
                eshkol_qrng_range: (min, max) => BigInt(Math.floor(Math.random() * Number(max - min)) + Number(min)),
                eshkol_init_global_arena: () => {},
                // Tail-transfer record (ESH-0102c). NOT a no-op stub: only the
                // AArch64 backends can lower an aggregate-return `musttail`, so
                // on wasm32 EVERY mutual tail call takes the tail-transfer
                // dispatcher, and the compiled code writes the pending flag, the
                // target and the arguments into this record for its driver loop
                // to read back. One bump allocation, memoised so every call in a
                // module returns the same address; sized for the C struct's
                // widest form (two u32 and a pointer padded to the tagged-value
                // alignment, then ESHKOL_TAIL_TRANSFER_MAX_ARGS 16-byte slots).
                // Linear memory starts zeroed and `_bump` never reuses, so
                // `pending` reads 0 before the first transfer. One record is
                // correct here where native needs a thread_local, because the
                // browser lite runtime is single-threaded.
                eshkol_tail_transfer_slot: (() => {
                    let slot = 0;
                    return () => (slot || (slot = rt._bump(16 + 32 * 16)));
                })(),
                // R7RS parameter-object runtime (make-parameter / parameterize):
                // opaque no-ops in the browser lite runtime (no arena), matching
                // the other hosted-runtime imports. Full fidelity on native + VM.
                eshkol_make_parameter_ptr: () => 0,
                eshkol_parameter_set_ptr: () => {},
                eshkol_parameter_set_converter_ptr: () => {},
                eshkol_parameter_ref_ptr: () => {},
                eshkol_parameter_converter_ref_ptr: () => {},
                eshkol_init_stack_size: () => {},
                eshkol_check_recursion_depth: () => 0,  // returns i32 (size_t)
                eshkol_decrement_recursion_depth: () => {},
                eshkol_make_exception_with_header: () => 0,
                eshkol_raise: (exc) => { console.error('Eshkol exception raised'); },
                // R7RS `write` (write/write-shared/write-simple): same
                // single-pointer-argument shape as eshkol_display_value above
                // (void eshkol_write_value(const tagged_value_t* value)),
                // degraded the same way.
                eshkol_deep_equal: () => 0,
                eshkol_type_error: () => { throw new Error('Eshkol type error (WASM stub)'); },
                eshkol_shape_error: () => { throw new Error('Eshkol shape error (WASM stub)'); },
                eshkol_tensor_result_dtype_binary: (r) => r,
                eshkol_tensor_result_dtype_unary: (r) => r,
                eshkol_type_error_with_operand: () => { throw new Error('Eshkol type error (WASM stub)'); },
                eshkol_procedure_call_error: () => { throw new Error('Eshkol call error: not a procedure or arity mismatch'); },
                eshkol_continuation_transfer_check: () => {},
                eshkol_arity_mismatch_error: () => { throw new Error('Eshkol arity mismatch (WASM stub)'); },
                eshkol_ad_mixed_record: () => 0,
                eshkol_ad_seed_flag: () => 0,
                eshkol_tensor_operand_checked: () => 0,
                // Same lite-glue contract as eshkol_tensor_operand_checked: the
                // browser glue has no tensor runtime (docs/FEATURE_MATRIX.md).
                eshkol_tensor_destination_checked: () => 0,
                eshkol_tensor_matrix_operand_checked: () => 0,
                eshkol_tensor_counts_checked: () => {},
                eshkol_tensor_axis_checked: (axis) => axis,
                // Shape and index helpers with the native contracts
                // (lib/core/tensor_validation.cpp, runtime_tensor_math.cpp,
                // runtime_tensor_index.cpp). i64 results are BigInt.
                eshkol_tensor_shape_total: (dimsPtr, ndim) => {
                    const dv = this.memory ? new DataView(this.memory.buffer) : null;
                    const n = Number(ndim);
                    if (!dv || !dimsPtr || n <= 0) return -1n;
                    const MAX = 0x7fffffffffffffffn;
                    let total = 1n;
                    for (let i = 0; i < n; i++) {
                        const d = dv.getBigInt64(Number(dimsPtr) + i * 8, true);
                        if (d < 0n) return -1n;
                        if (d === 0n) { total = 0n; continue; }
                        if (total > MAX / d) return -1n;
                        total *= d;
                    }
                    return total > MAX / 8n ? -1n : total;
                },
                eshkol_matmul_shape_valid: (M, K, N) => {
                    const MAX = 0x7fffffffffffffffn;
                    const pair = (a, b) => {
                        a = BigInt(a); b = BigInt(b);
                        if (a < 0n || b < 0n) return false;
                        if (a !== 0n && b !== 0n && a > MAX / b) return false;
                        return a * b <= MAX / 8n;
                    };
                    return (pair(M, K) && pair(K, N) && pair(M, N)) ? 1n : 0n;
                },
                // Shape helpers use the same row-major broadcast contract as
                // the native runtime.  These operate on WASM linear-memory
                // int64 arrays and are needed by generated tensor code.
                eshkol_tensor_broadcast_shape: (aPtr, aRank, bPtr, bRank, outPtr, outRankPtr, outTotalPtr) => {
                    const dv = this.memory ? new DataView(this.memory.buffer) : null;
                    if (!dv || aRank <= 0 || bRank <= 0 || aRank > 16 || bRank > 16) return 0;
                    const rank = Math.max(Number(aRank), Number(bRank));
                    const out = [];
                    for (let i = 0; i < rank; i++) {
                        const ai = i < aRank ? dv.getBigInt64(Number(aPtr) + (Number(aRank) - 1 - i) * 8, true) : 1n;
                        const bi = i < bRank ? dv.getBigInt64(Number(bPtr) + (Number(bRank) - 1 - i) * 8, true) : 1n;
                        if (ai < 0n || bi < 0n || (ai !== bi && ai !== 1n && bi !== 1n)) return 0;
                        out[rank - 1 - i] = ai === 1n ? bi : ai;
                    }
                    let total = 1n;
                    for (let i = 0; i < rank; i++) {
                        if (out[i] === 0n) total = 0n;
                        else {
                            if (total > 0x7fffffffffffffffn / out[i]) return 0;
                            total *= out[i];
                        }
                        dv.setBigInt64(Number(outPtr) + i * 8, out[i], true);
                    }
                    dv.setBigInt64(Number(outRankPtr), BigInt(rank), true);
                    dv.setBigInt64(Number(outTotalPtr), total, true);
                    return 1;
                },
                eshkol_broadcast_source_index: (flat, outPtr, outRank, srcPtr, srcRank) => {
                    const dv = this.memory ? new DataView(this.memory.buffer) : null;
                    const rank = Number(outRank), srank = Number(srcRank);
                    if (!dv || Number(flat) < 0 || rank < 0 || srank < 0 || rank > 16 || srank > rank) return -1n;
                    const strides = Array(srank).fill(0n);
                    if (srank) { strides[srank - 1] = 1n; for (let i = srank - 2; i >= 0; i--) { const d = dv.getBigInt64(Number(srcPtr) + (i + 1) * 8, true); if (d <= 0n) return -1n; strides[i] = strides[i + 1] * d; } }
                    let rem = BigInt(flat), result = 0n;
                    for (let i = rank - 1; i >= 0; i--) { const d = dv.getBigInt64(Number(outPtr) + i * 8, true); if (d <= 0n) return -1n; const coord = rem % d; rem /= d; const si = i - (rank - srank); if (si >= 0 && dv.getBigInt64(Number(srcPtr) + si * 8, true) !== 1n) result += coord * strides[si]; }
                    return result;
                },
                // No resource limit is ever active in the browser (limits come
                // from the native environment), so the ceiling check is the
                // native no-op path of lib/core/resource_limits.cpp.
                eshkol_enforce_tensor_elements: () => {},
                // The browser glue never records an AD tape (see
                // arena_allocate_ad_node above), so the home arena is the
                // caller's arena: the native no-tape path of
                // lib/core/runtime_autodiff.cpp.
                eshkol_ad_home_arena: (fallback) => fallback,
                eshkol_ad_node_set_exact_value: () => { throw new Error('exact AD values unsupported in WASM glue'); },
                eshkol_continuation_capture_handlers: () => { throw new Error('continuation handlers unsupported in WASM glue'); },
                eshkol_continuation_restore_handlers: () => { throw new Error('continuation handlers unsupported in WASM glue'); },
                eshkol_i128_binary_tagged: () => { throw new Error('i128 arithmetic unsupported in WASM glue'); },
                eshkol_i128_compare_tagged: () => { throw new Error('i128 comparison unsupported in WASM glue'); },
                eshkol_is_i128_tagged: (v) => {
                    // Match lib/core/i128_runtime.cpp: a HEAP_PTR (8) with
                    // HEAP_SUBTYPE_I128 (25) in its eight-byte object header.
                    // Generic arithmetic asks this of ordinary values too, so
                    // unlike the two operators above (only ever reached once
                    // this predicate has already said "yes"), this one MUST
                    // answer for real rather than throw: a throwing stub here
                    // would abort ordinary generic arithmetic on any heap
                    // operand, not just genuine i128 values.
                    const dv = this.memory ? new DataView(this.memory.buffer) : null;
                    if (!dv || !v) return 0;
                    if ((dv.getUint8(Number(v)) & 0x0F) !== 8) return 0;
                    const p = Number(dv.getBigUint64(Number(v) + 8, true) & 0xFFFFFFFFn);
                    return (p >= 8 && dv.getUint8(p - 8) === 25) ? 1 : 0;
                },

                // GPU compute (WebGPU). These are the ordinary GPU dispatch
                // seam — the same symbols the native Metal/CUDA backends
                // define in lib/backend/gpu/ — not a browser-special path.
                // eshkol_matmul_dispatch is what codegenMatmul emits; the
                // rest mirror the gpu_memory.h surface so a program can query
                // and steer dispatch. Values come from _gpuEnv(): a
                // WebAssembly.Suspending wrapper when WebGPU+JSPI are both
                // present, a plain synchronous CPU function otherwise.
                // Keep these keys IDENTICAL to web/eshkol-repl.js — the
                // INV-wasm-import-glue-equality invariant in
                // .icc/architecture-model.yaml is critical-severity.
                eshkol_matmul_dispatch: gpu.eshkol_matmul_dispatch,
                eshkol_batch_matmul_dispatch: gpu.eshkol_batch_matmul_dispatch,
                eshkol_gpu_elementwise_f64: gpu.eshkol_gpu_elementwise_f64,
                eshkol_gpu_reduce_f64: gpu.eshkol_gpu_reduce_f64,
                eshkol_gpu_init: gpu.eshkol_gpu_init,
                eshkol_gpu_shutdown: gpu.eshkol_gpu_shutdown,
                eshkol_gpu_get_backend: gpu.eshkol_gpu_get_backend,
                eshkol_gpu_backend_available: gpu.eshkol_gpu_backend_available,
                eshkol_gpu_supports_f64: gpu.eshkol_gpu_supports_f64,
                eshkol_gpu_has_fp64: gpu.eshkol_gpu_has_fp64,
                eshkol_gpu_should_use: gpu.eshkol_gpu_should_use,
                eshkol_gpu_set_threshold: gpu.eshkol_gpu_set_threshold,
                eshkol_gpu_get_threshold: gpu.eshkol_gpu_get_threshold,

                eshkol_set_error_location: () => {},
                eshkol_lambda_registry_init: () => {},
                eshkol_lambda_registry_add: () => {},
                eshkol_lambda_registry_lookup: () => 0,
                eshkol_closure_get_arity: () => 0,

                // C library
                strcmp: (a, b) => {
                    const mem = new Uint8Array(rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer);
                    let i = 0;
                    while (true) {
                        const ca = mem[a + i], cb = mem[b + i];
                        if (ca !== cb) return ca < cb ? -1 : 1;
                        if (ca === 0) return 0;
                        i++;
                    }
                },
                strncmp: (a, b, n) => {
                    const av = rt.readString(a).slice(0, Number(n));
                    const bv = rt.readString(b).slice(0, Number(n));
                    return av === bv ? 0 : (av < bv ? -1 : 1);
                },
                strlen: (s) => {
                    const mem = new Uint8Array(rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer);
                    let i = 0;
                    while (mem[s + i] !== 0) i++;
                    return i;
                },

                // I/O
                printf: (fmt, ...args) => { console.log('printf:', rt.readString(fmt)); return 0; },
                fprintf: (stream, fmt, ...args) => { console.log(rt.readString(fmt)); return 0; },
                puts: (s) => { console.log(rt.readString(s)); return 0; },
                putchar: (c) => { const ch = String.fromCharCode(c); process?.stdout?.write?.(ch) ?? console.log(ch); return c; },
                fflush: () => 0,
                snprintf: () => 0,
                abort: () => { throw new Error('abort called'); },
                exit: (code) => { console.log('exit:', code); },
                eshkol_fopen: () => 0,
                fclose: () => 0,
                fgets: () => 0,
                fputs: () => 0,
                fseek: () => -1,
                ftell: () => -1n,
                fread: () => 0n,
                fwrite: () => 0n,
                eshkol_remove: () => -1,
                eshkol_rename: () => -1,
                eshkol_builtin_make_temp_file: () => 0,
                // __arena-used (ESH-0187): void eshkol_builtin_arena_used(sv_t* out)
                // — SystemCodegen's all-pointer calling convention passes only the
                // result out-slot for a zero-arg builtin. No bounded/introspectable
                // arena exists in the browser build (fresh bump allocator per
                // program, nothing to report), so degrade to a no-op like the
                // error-object accessor stubs below rather than fabricate a count.
                eshkol_builtin_arena_used: (out) => {},
                clock_gettime: () => 0,
                pow: Math.pow,
                fmod: (a, b) => a % b,
                remainder: (a, b) => a - Math.round(a / b) * b,
                drand48: Math.random,
                strlen: (ptr) => { let len = 0; const b = new Uint8Array(rt.memory?.buffer || rt.createImports().env.__linear_memory.buffer); while (b[ptr + len]) len++; return len; },

                // Math
                sin: Math.sin, cos: Math.cos, tan: Math.tan,
                asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
                sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
                exp: Math.exp, log: Math.log, log2: Math.log2, log10: Math.log10,
                sqrt: Math.sqrt, cbrt: Math.cbrt, fabs: Math.abs,
                floor: Math.floor, ceil: Math.ceil, round: Math.round, trunc: Math.trunc,
                fmin: Math.min, fmax: Math.max,
                exp2: (x) => Math.pow(2, x),

                // Globals
                __global_arena: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),
                __stdinp: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),
                __stdoutp: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),
                __stderrp: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),

                // ────────────────────────────────────────────────────────────
                // Eshkol runtime helpers — keep this block in sync with the
                // C runtime in lib/core/.  The check is automated via
                // scripts/check_wasm_imports.py: it parses the WASM `import`
                // section and fails CI when any `env.<name>` import the
                // codegen emits is missing here.  Do NOT delete a stub
                // without first removing the matching `extern "C"` from the
                // codegen-emitted side; otherwise the website breaks at
                // WebAssembly.instantiate() with `function import requires a
                // callable`.
                // ────────────────────────────────────────────────────────────

                // Symbol interning — must canonicalise on name so that
                // (eq? 'foo 'foo) returns #t at runtime.  We bump-allocate
                // an 8-byte object header followed by the NUL-terminated
                // name into WASM linear memory, key the JS-side map by the
                // string, and return the data pointer (header lives at
                // ptr-8).  Identical names hit the cached pointer.
                eshkol_intern_symbol_lookup: (namePtr) => {
                    if (!namePtr) return 0;
                    if (!rt._symbolMap) rt._symbolMap = new Map();
                    const name = rt.readString(namePtr);
                    const cached = rt._symbolMap.get(name);
                    if (cached !== undefined) return cached;
                    const headerSize = 8;
                    const encoded = new TextEncoder().encode(name);
                    const block = rt._bump(headerSize + encoded.length + 1);
                    const dataPtr = block + headerSize;
                    const memory = rt._importedMemory || rt.memory;
                    const mem = new Uint8Array(memory.buffer);
                    const header = new DataView(memory.buffer, block, headerSize);
                    header.setUint8(0, 10);                         // HEAP_SUBTYPE_SYMBOL
                    header.setUint8(1, 0);                          // flags
                    header.setUint16(2, 0, true);                   // ref_count
                    header.setUint32(4, encoded.length + 1, true);  // size, including NUL
                    mem.set(encoded, dataPtr);
                    mem[dataPtr + encoded.length] = 0;
                    rt._symbolMap.set(name, dataPtr);
                    return dataPtr;
                },

                // Runtime / lifecycle no-ops — the website uses a fresh
                // bump-allocated arena per program, so init/recursion-depth
                // bookkeeping is unnecessary.
                __eshkol_lib_init__: () => {},
                eshkol_runtime_init: () => {},
                eshkol_init_stack_size: () => {},
                eshkol_check_recursion_depth: () => 0,
                eshkol_decrement_recursion_depth: () => {},
                eshkol_runtime_current_output_fp: () => 0,
                arena_tagged_cons_set_tagged_value: () => {},
                hash_table_set: () => false,
                hash_table_get: () => false,
                hash_table_has_key: () => false,
                hash_table_keys: () => 0,
                eshkol_string_byte_length: (ptr) => BigInt(rt.readString(ptr).length),
                eshkol_utf8_strlen: (ptr) => BigInt(Array.from(rt.readString(ptr)).length),
                eshkol_utf8_ref: () => 0,
                eshkol_utf8_substring: () => 0,
                eshkol_string_from_codepoints: () => 0,
                eshkol_string_to_number_tagged: () => 0n,

                // String ports — bump-allocate a buffer; reads/writes use a
                // JS-side Map keyed by the buffer pointer so we can
                // accumulate output and return it via get-output-string.
                eshkol_open_output_string: () => {
                    if (!rt._stringPorts) rt._stringPorts = new Map();
                    const port = rt._bump(16);  // sentinel block; identity only
                    rt._stringPorts.set(port, []);
                    return port;
                },
                eshkol_get_output_string: (port) => {
                    if (!rt._stringPorts) return 0;
                    const chunks = rt._stringPorts.get(port) || [];
                    const text = chunks.join('');
                    const block = rt._bump(8 + text.length + 1);
                    const dataPtr = block + 8;
                    const mem = new Uint8Array(rt._importedMemory.buffer);
                    for (let i = 0; i < text.length; i++) mem[dataPtr + i] = text.charCodeAt(i);
                    mem[dataPtr + text.length] = 0;
                    return dataPtr;
                },
                // R7RS `write` to an explicit port: same degraded shape as
                // eshkol_display_value_to_port above.
                // Exception handling — Eshkol exceptions in the WASM build
                // are degraded to console.error + bail; setjmp returns 0
                // (no-jump path) and longjmp throws so the JS host can
                // catch it.  This is fine for the website demo surface;
                // full exception support is a v1.4 item.
                eshkol_raise: (excPtr) => {
                    console.error('Eshkol raise (WASM stub): exception at ptr', excPtr);
                    throw new Error('Eshkol exception (WASM stub)');
                },
                eshkol_raise_not_pair: () => {
                    console.error('Eshkol: car/cdr of non-pair');
                    throw new Error('not a pair');
                },
                // Native constructor null checks use a non-returning failure
                // path. Browser exceptions already degrade to a host throw.
                eshkol_raise_allocation_failure: () => {
                    throw new Error('Eshkol allocation failed (WASM runtime)');
                },
                eshkol_push_exception_handler: () => 0,
                eshkol_pop_exception_handler: () => {},
                // SW-58 guard-loop replay. The browser build has no handler
                // chain to keep, so a depth of 0 with a no-op unwind and a
                // "no snapshot" restore degrades to the pre-SW-58 behaviour
                // rather than to a missing-import failure.
                eshkol_exception_handler_depth: () => 0n,
                eshkol_exception_handlers_unwind_to: () => {},
                eshkol_guard_replay_snapshot: () => {},
                eshkol_guard_replay_restore: () => 0,
                eshkol_get_current_exception: () => 0,
                eshkol_clear_current_exception: () => {},
                eshkol_get_raised_value: () => 0,
                eshkol_set_raised_value: () => {},
                // R7RS error-object accessors (llvm_codegen.cpp:
                // codegenErrorObjectPredicate / codegenErrorObjectAccessor).
                // eshkol_error_object_p(tagged*) -> i32; the message/irritants
                // accessors take (tagged* obj, tagged* out) and write the
                // result through the out-param. Degrade to "not an error /
                // empty result" for the browser build.
                eshkol_error_object_p:         () => 0,
                eshkol_error_object_message:   (_obj, _out) => {},
                eshkol_error_object_irritants: (_obj, _out) => {},
                eshkol_unwind_dynamic_wind: () => {},
                // Multi-shot re-entry (eshkol_continuation_resume) and its
                // rerooting companion depend on the same native longjmp this
                // build already can't provide, so they degrade the same way:
                // reroot is a no-op (there is nothing to re-enter) and resume
                // throws like the longjmp stub above it does.
                eshkol_reroot_dynamic_wind: () => {},
                eshkol_continuation_resume: () => { throw new Error('eshkol_continuation_resume (WASM stub) — not supported'); },
                // Promise evaluation rollback accompanies hosted setjmp/
                // longjmp. Browser continuations are deliberately degraded,
                // so these opaque markers are inert like the handler stubs.
                eshkol_promise_eval_mark: () => 0,
                eshkol_promise_eval_begin: () => {},
                eshkol_promise_eval_commit_one: () => {},
                eshkol_promise_eval_commit_to: () => {},
                eshkol_promise_eval_unwind_to: () => {},
                eshkol_jmp_buf_size: () => 64,
                setjmp: () => 0,                  // first-pass through
                longjmp: () => { throw new Error('longjmp (WASM stub) — not supported'); },

                // libc fallbacks the codegen reaches for occasionally
                puts: (s) => { console.log(rt.readString(s)); return 0; },
                fputc: (c, _fp) => { /* drop — we have no fp model */ return c; },
                length: () => 0,                  // Scheme length stub; codegen normally inlines

                // Compiler-rt builtins LLVM emits for 128-bit arithmetic.
                __multi3: (alo, ahi, blo, bhi) => 0n,  // 128×128→128 mul; bignums route through eshkol_bignum_*

                // Exact numeric imports are generated below.
                eshkol_list_reverse_tagged:       (value) => value,

                // Taylor towers remain outside this browser lane. Exact entry
                // refuses at seeding; the ordinary scalar lane uses the shared
                // numeric extraction imports, including exact heap values.
                // ESH-0393/0394 AD point classification + coercion. These used to
                // be inline IR (a bitcast for DOUBLE, SIToFP for everything
                // else); they became runtime calls so an exact rational/bignum
                // point -- which is HEAP-tagged, so its data field is a POINTER
                // -- stops being reinterpreted as a number. They are on the
                // ORDINARY jet path, so a `() => 0` stub would silently
                // differentiate every browser program at 0. Implement the
                // conversion here instead, over the same tagged layout the
                // region helpers above use: [0]=type, [1]=flags, [8..16]=data.
                //
                // Exact heap scalars use the shared numeric imports below.
                eshkol_is_taylor_tagged:        () => 0,
                eshkol_taylor_c0:               () => 0.0,
                eshkol_taylor_binary_tagged:    () => 0,
                eshkol_taylor_unary_tagged:     () => 0,
                eshkol_taylor_seed_tagged:      (_arena, _point, _order, _out) => { throw new exact.NumericError('Exact Taylor differentiation is unsupported in the browser LLVM/WASM lane', 'ESH_NUMERIC_UNSUPPORTED'); },
                eshkol_taylor_extract:          () => 0.0,
                eshkol_taylor_coeffs_list:      () => 0,
                // P5 reverse-over-Taylor helpers (autodiff_codegen.cpp):
                //   i32  eshkol_taylor_has_tangent(tagged*)
                //   f64  eshkol_taylor_extract_tangent(tagged*, i32)
                //   void eshkol_taylor_lift_ad_node(arena*, node*, i32, tagged*)
                eshkol_taylor_has_tangent:      () => 0,
                eshkol_taylor_extract_tangent:  () => 0.0,
                eshkol_taylor_lift_ad_node:     () => {},
                eshkol_taylor_project_forward_tangent: () => 0,
                // ADR-0027 nested levels (runtime_taylor.c):
                //   i32  eshkol_ad_nested_seed(arena*, tagged*, i32, i64, i32, i64, tagged*)
                //   void eshkol_ad_nested_extract(arena*, tagged*, i32, i32, i32, tagged*)
                //   void eshkol_ad_nested_unsupported(i32)
                //   void eshkol_ad_curried_gradient_unsupported()
                // ESH_AD_NEST_NONE (0) keeps the lite lane on the unchanged
                // non-nested seeding, exactly as the sibling stubs degrade.
                eshkol_ad_nested_seed:          () => 0,
                // BEGIN GENERATED EXACT IMPORTS
                eshkol_ad_point_is_exact_number: exact.imports.eshkol_ad_point_is_exact_number,
                eshkol_ad_point_is_exact_scalar: exact.imports.eshkol_ad_point_is_exact_scalar,
                eshkol_ad_point_is_scalar: exact.imports.eshkol_ad_point_is_scalar,
                eshkol_ad_point_to_double: exact.imports.eshkol_ad_point_to_double,
                eshkol_ad_dense_node_elements: exact.imports.eshkol_ad_dense_node_elements,
                eshkol_ad_copy_shape_to_home: exact.imports.eshkol_ad_copy_shape_to_home,
                eshkol_ad_node_probe: exact.imports.eshkol_ad_node_probe,
                eshkol_ad_node_total_elements: exact.imports.eshkol_ad_node_total_elements,
                eshkol_ad_seed_to_double: exact.imports.eshkol_ad_seed_to_double,
                eshkol_bignum_binary_tagged: exact.imports.eshkol_bignum_binary_tagged,
                eshkol_bignum_compare_tagged: exact.imports.eshkol_bignum_compare_tagged,
                eshkol_bignum_from_int64: exact.imports.eshkol_bignum_from_int64,
                eshkol_bignum_from_overflow: exact.imports.eshkol_bignum_from_overflow,
                eshkol_bignum_is_even: exact.imports.eshkol_bignum_is_even,
                eshkol_bignum_is_odd: exact.imports.eshkol_bignum_is_odd,
                eshkol_bignum_is_zero: exact.imports.eshkol_bignum_is_zero,
                eshkol_bignum_neg: exact.imports.eshkol_bignum_neg,
                eshkol_bignum_pow_tagged: exact.imports.eshkol_bignum_pow_tagged,
                eshkol_bignum_to_double: exact.imports.eshkol_bignum_to_double,
                eshkol_bignum_to_string: exact.imports.eshkol_bignum_to_string,
                eshkol_complex_pow: exact.imports.eshkol_complex_pow,
                eshkol_complex_sqrt: exact.imports.eshkol_complex_sqrt,
                eshkol_display_value: exact.imports.eshkol_display_value,
                eshkol_display_value_to_port: exact.imports.eshkol_display_value_to_port,
                eshkol_double_to_exact_tagged: exact.imports.eshkol_double_to_exact_tagged,
                eshkol_double_to_rational: exact.imports.eshkol_double_to_rational,
                eshkol_exact_rational_pow_tagged: exact.imports.eshkol_exact_rational_pow_tagged,
                eshkol_exact_sqrt_tagged: exact.imports.eshkol_exact_sqrt_tagged,
                eshkol_format_double: exact.imports.eshkol_format_double,
                eshkol_fprint_double: exact.imports.eshkol_fprint_double,
                eshkol_is_bignum_tagged: exact.imports.eshkol_is_bignum_tagged,
                // Checked eight-coefficient tensor jets; Taylor and reverse carriers refuse.
                eshkol_jet_tensor_binary: exact.imports.eshkol_jet_tensor_binary,
                eshkol_jet_tensor_matmul: exact.imports.eshkol_jet_tensor_matmul,
                eshkol_list_to_vector_sret: exact.imports.eshkol_list_to_vector_sret,
                eshkol_is_rational_tagged_ptr: exact.imports.eshkol_is_rational_tagged_ptr,
                eshkol_rational_binary_tagged_ptr: exact.imports.eshkol_rational_binary_tagged_ptr,
                eshkol_rational_ceil: exact.imports.eshkol_rational_ceil,
                eshkol_rational_ceil_tagged: exact.imports.eshkol_rational_ceil_tagged,
                eshkol_rational_compare_tagged_ptr: exact.imports.eshkol_rational_compare_tagged_ptr,
                eshkol_rational_create: exact.imports.eshkol_rational_create,
                eshkol_rational_denominator_tagged: exact.imports.eshkol_rational_denominator_tagged,
                eshkol_rational_equal: exact.imports.eshkol_rational_equal,
                eshkol_rational_floor: exact.imports.eshkol_rational_floor,
                eshkol_rational_floor_tagged: exact.imports.eshkol_rational_floor_tagged,
                eshkol_rational_from_bignums_tagged: exact.imports.eshkol_rational_from_bignums_tagged,
                eshkol_rational_make_tagged: exact.imports.eshkol_rational_make_tagged,
                eshkol_rational_numerator_tagged: exact.imports.eshkol_rational_numerator_tagged,
                eshkol_rational_pow_tagged: exact.imports.eshkol_rational_pow_tagged,
                eshkol_rational_round: exact.imports.eshkol_rational_round,
                eshkol_rational_round_tagged: exact.imports.eshkol_rational_round_tagged,
                eshkol_rational_to_double: exact.imports.eshkol_rational_to_double,
                eshkol_rational_to_string: exact.imports.eshkol_rational_to_string,
                eshkol_rational_truncate: exact.imports.eshkol_rational_truncate,
                eshkol_rational_truncate_tagged: exact.imports.eshkol_rational_truncate_tagged,
                eshkol_tensor_from_collection: exact.imports.eshkol_tensor_from_collection,
                eshkol_tensor_operand_carrier_checked: exact.imports.eshkol_tensor_operand_carrier_checked,
                eshkol_unwrap_list_index: exact.imports.eshkol_unwrap_list_index,
                eshkol_wasm_numeric_abi_check: exact.imports.eshkol_wasm_numeric_abi_check,
                eshkol_vref_unwrap_index: exact.imports.eshkol_vref_unwrap_index,
                arena_allocate_tape: exact.imports.arena_allocate_tape,
                arena_allocate_cons_with_header: exact.imports.arena_allocate_cons_with_header,
                arena_allocate_multi_value: exact.imports.arena_allocate_multi_value,
                arena_allocate_vector_with_header: exact.imports.arena_allocate_vector_with_header,
                arena_tape_add_node: exact.imports.arena_tape_add_node,
                arena_tape_get_node: exact.imports.arena_tape_get_node,
                arena_tape_get_node_count: exact.imports.arena_tape_get_node_count,
                arena_tape_reset: exact.imports.arena_tape_reset,
                memcpy: exact.imports.memcpy,
                memmove: exact.imports.memmove,
                memset: exact.imports.memset,
                eshkol_write_value: exact.imports.eshkol_write_value,
                eshkol_write_value_to_port: exact.imports.eshkol_write_value_to_port,
                // END GENERATED EXACT IMPORTS

                // BEGIN GENERATED FLAT-AD IMPORTS
                // Browser WASM has no Taylor tower lane. Keep the base lane's established
                // flat behavior: extraction declines the tower and enter/leave do nothing.
                eshkol_ad_tower_carry_result: () => 0,
                eshkol_ad_jet_extract_tower: () => 0,
                // Captured nested differentiation is explicitly unsupported in this lane.
                // Throwing is required so unsupported semantics cannot silently look valid.
                eshkol_ad_nested_capture_unsupported: () => {
                    throw new Error('Nested autodiff through captured values is unsupported in the browser WASM runtime');
                },
                eshkol_ad_tower_enter: () => {},
                eshkol_ad_tower_leave: () => {},
                // A jet pass's extraction guard (ADR-0027). The lite lane has no Taylor
                // carrier, so no carrier can reach it.
                eshkol_ad_jet_result_check: () => {},
                // END GENERATED FLAT-AD IMPORTS
                eshkol_ad_nested_extract:       () => {},
                eshkol_ad_nested_unsupported:   () => {},
                eshkol_ad_curried_gradient_unsupported: () => {},
                // ESH-0413 nested differentiation levels (runtime_taylor.c):
                //   i32 eshkol_ad_jet_extract_tower(arena*, tagged*, tagged*)
                // Reports "I did not handle this; keep your own extraction" by
                // returning 0, which is exactly what it returns natively for a
                // pass with no enclosing tower level. The whole tower path is
                // opaque in the browser build (eshkol_is_taylor_tagged above is
                // a constant 0), so no result here can ever be a tangent-carrying
                // tower and 0 is the faithful answer, not a degradation.
                eshkol_ad_jet_extract_tower:    () => 0,

                // Newly-surfaced runtime env imports the wasm backend can emit
                // (ESH-0224). Degrade like the sibling stubs above: allocators
                // bump the linear heap, void helpers no-op, and the capability
                // sandbox / file ops are browser no-ops.
                //   void* arena_allocate_multi_value(arena*, size_t count)
                //     -> header(size_t) + count * tagged_value(16B)
                //   void* eshkol_list_to_svec(arena*, tagged* head)
                //   void* eshkol_tensor_map_libm(arena*, tagged* in, i32 op)
                //   int   eshkol_fputs(const char* str, FILE*)
                //   void  eshkol_exception_add_irritant_ptr(exc*, tagged*)
                //   void  eshkol_parallel_map_sret(...)  (struct return)
                //   void  eshkol_builtin_file_rename(sv* out, sv* a, sv* b)
                //   void  eshkol_capability_runtime_{begin_install,allow,clear}
                eshkol_list_to_svec:               () => rt._bump(64),
                eshkol_tensor_map_libm:            () => rt._bump(64),
                eshkol_fputs:                      (s) => { console.log(rt.readString(s)); return 0; },
                eshkol_exception_add_irritant_ptr: () => {},
                eshkol_parallel_map_sret:          () => {},
                eshkol_builtin_file_rename:        () => {},

                // ESH-0011 portable event loop. The browser sandbox has no file
                // descriptors, so there is nothing for a readiness multiplexer
                // to watch and the whole surface fails closed with #f — the same
                // degradation make-pipe / fd-write / fd-close already use here.
                // Native builds get kqueue / epoll / IOCP; see
                // lib/core/event_loop.c. `event-loop-backend` answers "none" so
                // a program can detect the situation instead of guessing.
                eshkol_builtin_make_event_loop:       (out) => rt.writeFalse(out),
                eshkol_builtin_event_loop_add_fd:     (out) => rt.writeFalse(out),
                eshkol_builtin_event_loop_remove_fd:  (out) => rt.writeFalse(out),
                eshkol_builtin_event_loop_poll:       (out) => rt.writeFalse(out),
                eshkol_builtin_event_loop_close:      (out) => rt.writeFalse(out),
                eshkol_builtin_event_loop_backend:    (out) => rt.writeFalse(out),

                eshkol_capability_runtime_begin_install: () => {},
                eshkol_capability_runtime_allow:   () => {},
                eshkol_capability_runtime_clear:   () => {},

                // Lazy futures — no browser worker backing for this WASM glue.
                eshkol_lazy_future_is_ready: () => 1,
                eshkol_lazy_future_is_async: () => 0,
                eshkol_lazy_future_join_async: () => {},
                eshkol_lazy_future_get_thunk_ptr: () => 0,
                eshkol_lazy_future_get_thunk_type: () => 0,
                eshkol_lazy_future_get_thunk_flags: () => 0,
                eshkol_lazy_future_get_result_ptr: () => 0,
                eshkol_lazy_future_get_result_type: () => 0,
                eshkol_lazy_future_get_result_flags: () => 0,
                eshkol_lazy_future_set_result_ptr: () => {},

                // OALR regions — JIT path uses these.  In WASM we don't
                // have a region system; degrade (region_create returns a fake
                // handle, push/pop are no-ops). But escape / write-barrier
                // receive an UNINITIALIZED `out` slot that the caller reads back
                // — the runtime fn is the SOLE writer of `out`, so a no-op would
                // drop the value. With nothing ever freed, the correct
                // degradation is a shallow 16-byte tagged-value copy src -> out.
                region_create: (_name, _size_hint) => 1,
                region_push:   () => {},
                region_pop:    () => {},
                // with-region hijack (thread-safe region scope): JS has no region
                // system, so decline the hijack. enter returns 0 (no displaced arena to
                // restore); leave is a no-op for a declined enter.
                eshkol_region_enter: (_region) => 0,
                eshkol_region_leave: (_saved) => {},
                region_escape_tagged_value_into: (out, val) => {
                    const buf = rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer;
                    if (!buf || !out || !val) return;
                    const o = Number(out), v = Number(val);
                    new Uint8Array(buf).copyWithin(o, v, v + 16);
                },
                // eshkol_region_write_barrier_into(out, dst, value): shallow-copy
                // value -> out (dst is only a region-ownership probe).
                eshkol_region_write_barrier_into: (out, _dst, value) => {
                    const buf = rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer;
                    if (!buf || !out || !value) return;
                    const o = Number(out), v = Number(value);
                    new Uint8Array(buf).copyWithin(o, v, v + 16);
                },
                // Range form (vector-copy!): slots already populated by the
                // preceding memmove; nothing to promote -> genuine no-op.
                eshkol_region_write_barrier_range: () => {},

                // #341 user-reachable region handles + the shared region-teardown
                // primitive. `with-region` lowering now goes through
                // eshkol_region_unwind_to (promote the kept value one level out,
                // restore the allocation slot, pop) instead of open-coding
                // escape + region_pop + region_leave, so these imports appear in
                // ANY module that uses with-region — not only ones using handles.
                //
                // eshkol_region_unwind_to(mark, vals, n): the caller stores the
                // result INTO `vals` and reads the same slot back afterwards, so
                // with no region system to promote out of the correct degradation
                // is a genuine no-op — the value is already in the slot.
                eshkol_region_unwind_to:  (_mark, _vals, _n) => {},
                eshkol_region_mark:       () => 0,
                // Continuation-crossing unwind: nothing to tear down.
                eshkol_region_unwind_for_continuation: (_state) => {},
                // (region-open …) — hand back a nonzero opaque token. Slot 1 /
                // generation 1 in the runtime encoding ((gen << 8) | (slot+1)).
                eshkol_region_open_builtin: (out, _a, _b, _reclaim) => {
                    const buf = rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer;
                    if (!buf || !out) return;
                    const o = Number(out);
                    new Uint8Array(buf).fill(0, o, o + 16);
                    const dv = new DataView(buf);
                    dv.setUint8(o, 1);            // ESHKOL_VALUE_INT64
                    dv.setUint8(o + 1, 1);        // exact flag
                    dv.setBigInt64(o + 8, 257n, true);  // (1 << 8) | 1
                },
                // (region-close handle v …) — same out-slot ABI as the escape
                // helpers above: shallow-copy the first keep into `out` (nothing
                // is freed, so nothing needs promoting), or leave the empty list for none.
                // The n > 1 list form degrades to the first keep in the browser.
                eshkol_region_close_builtin: (out, _handle, vals, n) => {
                    const buf = rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer;
                    if (!buf || !out) return;
                    const o = Number(out);
                    const bytes = new Uint8Array(buf);
                    bytes.fill(0, o, o + 16);     // ESHKOL_VALUE_NULL = the empty list
                    if (Number(n) >= 1 && vals) {
                        const v = Number(vals);
                        bytes.copyWithin(o, v, v + 16);
                    }
                },
                // (region-open? handle) — with no region system a handle is never
                // actually torn down, so report #f rather than claim a liveness
                // the browser cannot track.
                eshkol_region_open_p_builtin: (out, _handle) => {
                    const buf = rt._importedMemory?.buffer || rt.instance?.exports?.memory?.buffer;
                    if (!buf || !out) return;
                    const o = Number(out);
                    new Uint8Array(buf).fill(0, o, o + 16);
                    new DataView(buf).setUint8(o, 3);   // ESHKOL_VALUE_BOOL
                },

                // Tensor runtime helpers
                eshkol_broadcast_elementwise_f64: () => 0,
                eshkol_broadcast_shape_f64: (ap, ar, bp, br, out, rankOut, totalOut) => {
                    const view = new DataView(this.memory.buffer);
                    view.setBigInt64(rankOut, 0n, true);
                    view.setBigInt64(totalOut, 0n, true);
                    ar = Number(ar); br = Number(br);
                    if (ar < 0 || br < 0 || ar > 16 || br > 16) return -1n;
                    const rank = Math.max(ar, br), dims = [];
                    let total = 1n;
                    for (let axis = 0; axis < rank; axis++) {
                        const ai = axis - (rank - ar), bi = axis - (rank - br);
                        const a = ai < 0 ? 1n : view.getBigInt64(ap + 8 * ai, true);
                        const b = bi < 0 ? 1n : view.getBigInt64(bp + 8 * bi, true);
                        if (a < 0n || b < 0n || (a !== b && a !== 1n && b !== 1n)) return -1n;
                        const dim = a === 1n ? b : a;
                        if (dim && total > 0x7fffffffffffffffn / dim) return -1n;
                        total *= dim;
                        dims.push(dim);
                    }
                    dims.forEach((dim, axis) => view.setBigInt64(out + axis * 8, dim, true));
                    view.setBigInt64(rankOut, BigInt(rank), true);
                    view.setBigInt64(totalOut, total, true);
                    return 0n;
                },
                eshkol_shapes_equal: () => 0,

                // Continuations (call/cc) — WASM can't longjmp out of
                // host frames, so we degrade these.  Calling the
                // continuation will throw at runtime via setjmp/longjmp
                // stubs above, which is detectable by the user.
                eshkol_make_continuation_state:   () => 0,
                eshkol_make_continuation_state_flags: () => 0,
                eshkol_make_continuation_closure: () => 0,

                // === DOM API ===

                web_get_body: () => rt.bodyHandle,
                web_get_document: () => rt.documentHandle,
                web_get_window: () => rt.windowHandle,

                web_create_element: (tagPtr) => {
                    const tag = rt.readString(tagPtr);
                    return rt.createHandle(document.createElement(tag));
                },

                web_create_text_node: (textPtr) => {
                    return rt.createHandle(document.createTextNode(rt.readString(textPtr)));
                },

                web_append_child: (parentHandle, childHandle) => {
                    const parent = rt.getHandle(parentHandle);
                    const child = rt.getHandle(childHandle);
                    if (parent && child) parent.appendChild(child);
                    return childHandle;
                },

                web_remove_child: (parentHandle, childHandle) => {
                    const parent = rt.getHandle(parentHandle);
                    const child = rt.getHandle(childHandle);
                    if (parent && child) parent.removeChild(child);
                    return 0;
                },

                web_set_text_content: (elHandle, textPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.textContent = rt.readString(textPtr);
                    return 0;
                },

                web_set_inner_html: (elHandle, htmlPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el && el.innerHTML !== undefined) el.innerHTML = rt.readString(htmlPtr);
                    return 0;
                },

                web_set_attribute: (elHandle, namePtr, valuePtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.setAttribute(rt.readString(namePtr), rt.readString(valuePtr));
                    return 0;
                },

                web_get_attribute: (elHandle, namePtr, bufPtr, bufLen) => {
                    const el = rt.getHandle(elHandle);
                    if (!el) return 0;
                    const val = el.getAttribute(rt.readString(namePtr)) || '';
                    return rt.writeString(bufPtr, val, bufLen);
                },

                web_set_style: (elHandle, propPtr, valuePtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.style[rt.readString(propPtr)] = rt.readString(valuePtr);
                    return 0;
                },

                web_add_class: (elHandle, classPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.classList.add(rt.readString(classPtr));
                    return 0;
                },

                web_remove_class: (elHandle, classPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.classList.remove(rt.readString(classPtr));
                    return 0;
                },

                web_get_element_by_id: (idPtr) => {
                    const el = document.getElementById(rt.readString(idPtr));
                    return el ? rt.createHandle(el) : 0;
                },

                web_query_selector: (selectorPtr) => {
                    const el = document.querySelector(rt.readString(selectorPtr));
                    return el ? rt.createHandle(el) : 0;
                },

                web_query_selector_all: (selectorPtr) => {
                    const els = document.querySelectorAll(rt.readString(selectorPtr));
                    return rt.createHandle(Array.from(els));
                },

                web_add_event_listener: (elHandle, eventPtr, callbackFuncPtr) => {
                    const el = rt.getHandle(elHandle);
                    if (!el || !rt.instance) return 0;
                    const eventName = rt.readString(eventPtr);
                    const handler = (e) => {
                        const eventHandle = rt.createHandle(e);
                        try {
                            const result = rt.invokeWasmCallback(callbackFuncPtr, eventHandle);
                            if (result && typeof result.then === 'function') {
                                result.catch((err) => console.error('Event handler error:', err))
                                    .finally(() => rt.releaseHandle(eventHandle));
                                return;
                            }
                        } catch (err) {
                            console.error('Event handler error:', err);
                        }
                        rt.releaseHandle(eventHandle);
                    };
                    el.addEventListener(eventName, handler);
                    return 1;
                },

                web_prevent_default: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    if (e && e.preventDefault) e.preventDefault();
                    return 0;
                },

                web_console_log: (msgPtr) => {
                    console.log(rt.readString(msgPtr));
                    return 0;
                },

                // Canvas 2D
                web_canvas_get_context: (canvasHandle, ctxTypePtr) => {
                    const canvas = rt.getHandle(canvasHandle);
                    if (!canvas) return 0;
                    return rt.createHandle(canvas.getContext(rt.readString(ctxTypePtr)));
                },

                web_canvas_fill_rect: (ctxHandle, x, y, w, h) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fillRect(x, y, w, h);
                    return 0;
                },

                web_canvas_clear_rect: (ctxHandle, x, y, w, h) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.clearRect(x, y, w, h);
                    return 0;
                },

                web_canvas_set_fill_style: (ctxHandle, colorPtr) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fillStyle = rt.readString(colorPtr);
                    return 0;
                },

                web_canvas_set_stroke_style: (ctxHandle, colorPtr) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.strokeStyle = rt.readString(colorPtr);
                    return 0;
                },

                web_canvas_set_line_width: (ctxHandle, width) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.lineWidth = width;
                    return 0;
                },

                web_canvas_begin_path: (ctxHandle) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.beginPath();
                    return 0;
                },

                web_canvas_move_to: (ctxHandle, x, y) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.moveTo(x, y);
                    return 0;
                },

                web_canvas_line_to: (ctxHandle, x, y) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.lineTo(x, y);
                    return 0;
                },

                web_canvas_stroke: (ctxHandle) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.stroke();
                    return 0;
                },

                web_canvas_fill: (ctxHandle) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fill();
                    return 0;
                },

                web_canvas_fill_text: (ctxHandle, textPtr, x, y) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.fillText(rt.readString(textPtr), x, y);
                    return 0;
                },

                web_canvas_set_font: (ctxHandle, fontPtr) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.font = rt.readString(fontPtr);
                    return 0;
                },

                web_canvas_arc: (ctxHandle, x, y, r, start, end) => {
                    const ctx = rt.getHandle(ctxHandle);
                    if (ctx) ctx.arc(x, y, r, start, end);
                    return 0;
                },

                // Timers & Animation
                web_set_timeout: (callbackFuncPtr, ms) => {
                    if (!rt.instance) return 0;
                    return setTimeout(() => {
                        try {
                            const result = rt.invokeWasmCallback(callbackFuncPtr, 0);
                            if (result && typeof result.then === 'function') {
                                result.catch((e) => console.error('Timeout callback error:', e));
                            }
                        } catch(e) { console.error('Timeout callback error:', e); }
                    }, ms);
                },

                web_set_interval: (callbackFuncPtr, ms) => {
                    if (!rt.instance) return 0;
                    return setInterval(() => {
                        try {
                            const result = rt.invokeWasmCallback(callbackFuncPtr, 0);
                            if (result && typeof result.then === 'function') {
                                result.catch((e) => console.error('Interval callback error:', e));
                            }
                        } catch(e) { console.error('Interval callback error:', e); }
                    }, ms);
                },

                web_clear_interval: (id) => { clearInterval(id); return 0; },
                web_clear_timeout: (id) => { clearTimeout(id); return 0; },

                web_request_animation_frame: (callbackFuncPtr) => {
                    if (!rt.instance) return 0;
                    return requestAnimationFrame((ts) => {
                        try {
                            const result = rt.invokeWasmCallback(callbackFuncPtr, ts | 0);
                            if (result && typeof result.then === 'function') {
                                result.catch((e) => console.error('RAF callback error:', e));
                            }
                        } catch(e) { console.error('RAF callback error:', e); }
                    });
                },

                // Fetch API
                web_fetch: (urlPtr, callbackFuncPtr) => {
                    const url = rt.readString(urlPtr);
                    fetch(url).then(r => r.text()).then(text => {
                        if (rt.instance) {
                            // Write response to WASM memory
                            const ptr = rt.createImports().env.arena_allocate(0, text.length + 1);
                            rt.writeString(ptr, text, text.length + 1);
                            try {
                                const result = rt.invokeWasmCallback(callbackFuncPtr, ptr);
                                if (result && typeof result.then === 'function') {
                                    result.catch((e) => console.error('Fetch callback error:', e));
                                }
                            } catch(e) { console.error('Fetch callback error:', e); }
                        }
                    }).catch(e => console.error('fetch error:', e));
                    return 0;
                },

                // Location (pathname-based routing — clean URLs, no hash)
                web_get_hash: (bufPtr, bufLen) => {
                    var path = window.location.pathname.replace(/\/$/, '') || '/';
                    return rt.writeString(bufPtr, path, bufLen);
                },

                web_set_hash: (hashPtr) => {
                    var path = rt.readString(hashPtr);
                    history.pushState(null, '', path);
                    return 0;
                },

                web_get_href: (bufPtr, bufLen) => {
                    return rt.writeString(bufPtr, window.location.href, bufLen);
                },

                // REPL evaluation (uses eshkol-vm.wasm — the full bytecode VM)
                web_repl_eval: (sourcePtr, resultHandle) => {
                    const source = rt.readString(sourcePtr);
                    const target = rt.handles.get(resultHandle);
                    if (target && rt._eshkolVM) {
                        // The VM may suspend on a WebGPU readback (ADR-0029),
                        // so its result is appended when it resolves.
                        const G = (typeof globalThis !== 'undefined') && globalThis.EshkolWebGPU;
                        const done = G && G.vmCall
                            ? G.vmCall(rt._eshkolVM, 'repl_eval', source)
                            : Promise.resolve().then(() =>
                                rt._eshkolVM.cwrap('repl_eval', 'string', ['string'])(source));
                        done.then((result) => { target.textContent += result; },
                                  (e) => { target.textContent += 'Error: ' + e.message + '\n'; });
                    }
                    return 0;
                },

                // Content loading (fetch URL into target element). The work
                // is in loadContent so the sidebar click handler shares it.
                web_load_content: (urlPtr, targetHandle) => {
                    const target = rt.handles.get(targetHandle);
                    if (target) rt.loadContent(rt.readString(urlPtr), target);
                    return 0;
                },

                // Window
                web_get_window_width: () => window.innerWidth,
                web_get_window_height: () => window.innerHeight,
                web_get_scroll_y: () => window.scrollY | 0,
                web_scroll_to: (x, y) => { window.scrollTo(Number(x), Number(y)); return 0; },

                // Performance
                web_performance_now: () => performance.now() | 0,

                // Storage
                web_local_storage_set: (keyPtr, valuePtr) => {
                    localStorage.setItem(rt.readString(keyPtr), rt.readString(valuePtr));
                    return 0;
                },

                web_local_storage_get: (keyPtr, bufPtr, bufLen) => {
                    const val = localStorage.getItem(rt.readString(keyPtr)) || '';
                    return rt.writeString(bufPtr, val, bufLen);
                },

                // Event properties
                web_event_target: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    return e?.target ? rt.createHandle(e.target) : 0;
                },

                web_event_key: (eventHandle, bufPtr, bufLen) => {
                    const e = rt.getHandle(eventHandle);
                    return rt.writeString(bufPtr, e?.key || '', bufLen);
                },

                web_event_client_x: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    return e?.clientX || 0;
                },

                web_event_client_y: (eventHandle) => {
                    const e = rt.getHandle(eventHandle);
                    return e?.clientY || 0;
                },

                // Input elements
                web_get_value: (elHandle, bufPtr, bufLen) => {
                    const el = rt.getHandle(elHandle);
                    return rt.writeString(bufPtr, el?.value || '', bufLen);
                },

                web_set_value: (elHandle, valuePtr) => {
                    const el = rt.getHandle(elHandle);
                    if (el) el.value = rt.readString(valuePtr);
                    return 0;
                },

                // Runnable code block — code block + Run ▶ button + output area
                web_create_runnable_code: (rawCodePtr, htmlPtr, parentHandle) => {
                    const rawCode = rt.readString(rawCodePtr);
                    const html = rt.readString(htmlPtr);
                    const parent = rt.getHandle(parentHandle);
                    if (!parent) return 0;

                    const wrapper = document.createElement('div');
                    wrapper.className = 'runnable-code';
                    wrapper.style.cssText = 'background:#0d0d15;border:1px solid #2a2a3a;border-radius:12px;overflow:hidden;margin-bottom:0';

                    // Code display
                    const codeDiv = document.createElement('div');
                    codeDiv.style.cssText = 'padding:16px 20px;overflow-x:auto;overflow-y:visible';
                    codeDiv.innerHTML = html;
                    wrapper.appendChild(codeDiv);

                    // Toolbar
                    const toolbar = document.createElement('div');
                    toolbar.style.cssText = 'display:flex;align-items:center;justify-content:flex-end;padding:6px 16px;border-top:1px solid #1a1a28;background:#0a0a0f';

                    const btn = document.createElement('button');
                    btn.textContent = 'Run \u25b6';
                    btn.dataset.runCode = rawCode;
                    btn.style.cssText = 'background:#7c3aed;color:#fff;border:none;padding:4px 14px;border-radius:5px;font-size:0.78rem;font-weight:600;cursor:pointer;font-family:JetBrains Mono,monospace;letter-spacing:0.5px';
                    btn.addEventListener('mouseenter', () => { btn.style.background = '#6d28d9'; });
                    btn.addEventListener('mouseleave', () => { btn.style.background = '#7c3aed'; });
                    toolbar.appendChild(btn);
                    wrapper.appendChild(toolbar);

                    // Output area (hidden until first run)
                    const output = document.createElement('pre');
                    output.setAttribute('data-run-output', '1');
                    output.style.cssText = 'display:none;margin:0;padding:10px 28px;color:#00ff88;font-family:JetBrains Mono,monospace;font-size:0.82rem;line-height:1.5;background:#050510;border-top:1px solid #1a1a28;white-space:pre-wrap';
                    wrapper.appendChild(output);

                    parent.appendChild(wrapper);
                    return rt.createHandle(wrapper);
                },
            }
        };
    }
}
