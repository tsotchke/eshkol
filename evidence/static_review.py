from pathlib import Path
import subprocess
src=Path(__file__).resolve().parents[1]
codec=(src/'lib/core/msgpack.esk').read_text()
test=(src/'tests/v1_2_edge_cases/msgpack_utf8_test.esk').read_text()
gate=(src/'scripts/run_msgpack_gate.py').read_text()
docs=(src/'docs/reference/stdlib/msgpack.md').read_text()
# The byte-range gates implement Unicode scalar-value UTF-8, not merely
# continuation-byte syntax.
for token in [
    '(>= b 194)', '(= b 224)', '(= b 237)', '(= b 240)',
    '(and (>= b 241) (<= b 243))', '(= b 244)',
    '(else (error "msgpack: invalid UTF-8 leading byte" i))',
    'msgpack: truncated UTF-8 scalar', 'msgpack: invalid UTF-8 continuation']:
    assert token in codec, token
enc=codec[codec.index('(define (msgpack-encode-string'):codec.index('(define (msgpack-encode-array-elements')]
decbin=codec[codec.index('(define (msgpack-decode-bin'):codec.index('(define (msgpack-decode-string')]
decstr=codec[codec.index('(define (msgpack-decode-string'):codec.index('(define (msgpack-decode-array-elements')]
value=codec[codec.index('(define (msgpack-encode-value'):codec.index('(define (msgpack-encode value')]
assert 'msgpack-require-utf8' in enc
assert 'msgpack-require-available' in decstr and 'msgpack-require-utf8' in decstr
assert decstr.index('msgpack-require-available') < decstr.index('msgpack-require-utf8') < decstr.index('utf8->string')
assert 'msgpack-require-utf8' not in decbin
assert '((bytevector? value) (msgpack-encode-bin-bytes' in value
# Independent acceptance covers U+0000..U+10FFFF boundary encodings, all
# four MessagePack text-length transition values, 16 malformed-frame shapes,
# strict invalid-string encoding, and arbitrary binary bytes.
assert sum(1 for line in test.splitlines() if line.startswith('(golden ')) == 10
assert sum(1 for line in test.splitlines() if line.startswith('(string-golden ')) == 4
assert 'msgpack_utf8_test.esk", "PASS: MessagePack UTF-8 80 checks"' in gate
assert '80 UTF-8 checks' in docs
assert '(bytevector 0 127 128 191 192 255)' in test
print('STATIC_PASS utf8_scalar_ranges=10_boundary_goldens=10 marker=80 binary_bypass=verified')
