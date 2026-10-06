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
    const display = (p, port) => {
        const isNumber = numeric(p);
        if (!isNumber && port === undefined) return;
        const text = isNumber ? format(read(p)) : String(p);
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
    const numericGeometry = [8, 0, 4, 8, 32, 0, 8, 16, 20, 24, 28];
    const imports = {
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
        NumericError, transaction, string, span };
}
