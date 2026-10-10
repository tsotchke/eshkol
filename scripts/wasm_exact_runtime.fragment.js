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
            const sub = subtype(Number(q));
            if (sub === 23) fail('Taylor towers are unsupported in browser tensor arithmetic', 'ESH_AD_UNSUPPORTED');
            if (sub === 22) fail('reverse-mode nodes cannot enter forward-mode tensor arithmetic', 'ESH_AD_UNSUPPORTED');
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
        if (!legacy) payload(p, 3, 40);
        const v = view();
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
        payload(p, 2, 144);
        const v = view();
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
