/*
 * Golden-vector test for inc/eshkol/core/protobuf_wire.h.
 *
 * Every expected byte string below is derived by hand from the protobuf
 * encoding specification (https://protobuf.dev/programming-guides/encoding/);
 * the spec's own worked examples (150 -> 96 01, Test1/Test2/Test3 and the
 * packed field `d`) are reproduced verbatim. The same vectors are checked
 * against the Eshkol-level encoder in tests/protobuf/protobuf_wire_test.esk.
 */

#include "eshkol/core/protobuf_wire.h"

#include <stdio.h>
#include <string.h>

static int checks = 0;
static int failures = 0;

static void check(int ok, const char* label) {
    ++checks;
    if (!ok) {
        ++failures;
        printf("FAIL: %s\n", label);
    }
}

static int bytes_equal(const uint8_t* got, size_t got_len,
                       const uint8_t* want, size_t want_len) {
    return got_len == want_len && memcmp(got, want, want_len) == 0;
}

static void varint_vector(uint64_t value, const uint8_t* want, size_t want_len,
                          const char* label) {
    uint8_t buf[ESHKOL_PB_VARINT_MAX_BYTES];
    const size_t n = eshkol_pb_write_varint(buf, value);
    check(bytes_equal(buf, n, want, want_len), label);
    check(eshkol_pb_varint_size(value) == want_len, label);
    uint64_t back = 0;
    check(eshkol_pb_read_varint(want, want_len, &back) == want_len && back == value, label);
    /* Every strict prefix is truncated input. */
    for (size_t cut = 0; cut < want_len; ++cut)
        check(eshkol_pb_read_varint(want, cut, &back) == 0, label);
}

#define VARINT(value, label, ...)                                            \
    do {                                                                     \
        static const uint8_t want_[] = {__VA_ARGS__};                        \
        varint_vector((value), want_, sizeof want_, (label));                \
    } while (0)

int main(void) {
    /* Varints: one-byte boundary, the spec's 150, two/three-byte boundaries,
     * 32-bit and 64-bit extremes. */
    VARINT(0u, "varint 0", 0x00);
    VARINT(1u, "varint 1", 0x01);
    VARINT(127u, "varint 127", 0x7F);
    VARINT(128u, "varint 128", 0x80, 0x01);
    VARINT(150u, "varint 150 (spec)", 0x96, 0x01);
    VARINT(300u, "varint 300", 0xAC, 0x02);
    VARINT(16383u, "varint 16383", 0xFF, 0x7F);
    VARINT(16384u, "varint 16384", 0x80, 0x80, 0x01);
    VARINT(4294967295u, "varint 2^32-1", 0xFF, 0xFF, 0xFF, 0xFF, 0x0F);
    VARINT(9223372036854775807u, "varint 2^63-1",
           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F);
    VARINT(18446744073709551615u, "varint 2^64-1 (int32/int64 -1 on the wire)",
           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01);

    /* Zigzag (spec table plus the 32/64-bit extremes). */
    check(eshkol_pb_zigzag_encode64(0) == 0u, "zigzag 0");
    check(eshkol_pb_zigzag_encode64(-1) == 1u, "zigzag -1");
    check(eshkol_pb_zigzag_encode64(1) == 2u, "zigzag 1");
    check(eshkol_pb_zigzag_encode64(-2) == 3u, "zigzag -2");
    check(eshkol_pb_zigzag_encode64(2147483647) == 4294967294u, "zigzag int32 max");
    check(eshkol_pb_zigzag_encode64(-2147483647 - 1) == 4294967295u, "zigzag int32 min");
    check(eshkol_pb_zigzag_encode64(INT64_MAX) == UINT64_MAX - 1u, "zigzag int64 max");
    check(eshkol_pb_zigzag_encode64(INT64_MIN) == UINT64_MAX, "zigzag int64 min");
    {
        static const int64_t samples[] = {0, -1, 1, -2, 63, -64, 64, INT64_MAX, INT64_MIN};
        for (size_t i = 0; i < sizeof samples / sizeof samples[0]; ++i)
            check(eshkol_pb_zigzag_decode64(eshkol_pb_zigzag_encode64(samples[i])) == samples[i],
                  "zigzag round trip");
    }

    /* Tags. */
    {
        uint8_t buf[ESHKOL_PB_VARINT_MAX_BYTES];
        static const uint8_t t1[] = {0x08}, t2[] = {0x12}, t15[] = {0x7D},
                             t16[] = {0x80, 0x01},
                             tmax[] = {0xFD, 0xFF, 0xFF, 0xFF, 0x0F};
        check(bytes_equal(buf, eshkol_pb_write_tag(buf, 1, ESHKOL_PB_WIRE_VARINT), t1, 1),
              "tag 1 VARINT");
        check(bytes_equal(buf, eshkol_pb_write_tag(buf, 2, ESHKOL_PB_WIRE_LEN), t2, 1),
              "tag 2 LEN");
        check(bytes_equal(buf, eshkol_pb_write_tag(buf, 15, ESHKOL_PB_WIRE_FIXED32), t15, 1),
              "tag 15 I32");
        check(bytes_equal(buf, eshkol_pb_write_tag(buf, 16, ESHKOL_PB_WIRE_VARINT), t16, 2),
              "tag 16 VARINT");
        check(bytes_equal(buf, eshkol_pb_write_tag(buf, ESHKOL_PB_MAX_FIELD_NUMBER,
                                                   ESHKOL_PB_WIRE_FIXED32), tmax, 5),
              "tag 2^29-1 I32");
        check(eshkol_pb_tag_size(16, ESHKOL_PB_WIRE_VARINT) == 2, "tag size 16");
        uint32_t field = 0, type = 0;
        check(eshkol_pb_read_tag(tmax, 5, &field, &type) == 5 &&
              field == ESHKOL_PB_MAX_FIELD_NUMBER && type == ESHKOL_PB_WIRE_FIXED32,
              "read tag 2^29-1");
        static const uint8_t zero_field[] = {0x00}, wire6[] = {0x0E}, wire7[] = {0x0F},
                             too_big[] = {0x80, 0x80, 0x80, 0x80, 0x10};
        check(eshkol_pb_read_tag(zero_field, 1, &field, &type) == 0, "field number 0 refused");
        check(eshkol_pb_read_tag(wire6, 1, &field, &type) == 0, "wire type 6 refused");
        check(eshkol_pb_read_tag(wire7, 1, &field, &type) == 0, "wire type 7 refused");
        check(eshkol_pb_read_tag(too_big, 5, &field, &type) == 0, "field number 2^29 refused");
    }

    /* Fixed-width little-endian. */
    {
        uint8_t buf[8];
        static const uint8_t f32[] = {0x78, 0x56, 0x34, 0x12};
        static const uint8_t f64[] = {0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12};
        static const uint8_t one_double[] = {0, 0, 0, 0, 0, 0, 0xF0, 0x3F};
        check(bytes_equal(buf, eshkol_pb_write_fixed32(buf, 0x12345678u), f32, 4), "fixed32");
        check(bytes_equal(buf, eshkol_pb_write_fixed64(buf, 0x1234567890ABCDEFull), f64, 8),
              "fixed64");
        uint64_t one_bits = 0;
        const double one = 1.0;
        memcpy(&one_bits, &one, sizeof one_bits);
        check(bytes_equal(buf, eshkol_pb_write_fixed64(buf, one_bits), one_double, 8),
              "double 1.0 as fixed64");
        uint32_t v32 = 0;
        uint64_t v64 = 0;
        check(eshkol_pb_read_fixed32(f32, 4, &v32) == 4 && v32 == 0x12345678u, "read fixed32");
        check(eshkol_pb_read_fixed64(f64, 8, &v64) == 8 && v64 == 0x1234567890ABCDEFull,
              "read fixed64");
        check(eshkol_pb_read_fixed32(f32, 3, &v32) == 0, "truncated fixed32 refused");
        check(eshkol_pb_read_fixed64(f64, 7, &v64) == 0, "truncated fixed64 refused");
    }

    /* Spec worked examples, assembled from the primitives. */
    {
        uint8_t buf[32];
        size_t n = 0;
        static const uint8_t test1[] = {0x08, 0x96, 0x01};
        n = eshkol_pb_write_tag(buf, 1, ESHKOL_PB_WIRE_VARINT);
        n += eshkol_pb_write_varint(buf + n, 150);
        check(bytes_equal(buf, n, test1, sizeof test1), "spec Test1 a=150");

        static const uint8_t test2[] = {0x12, 0x07, 't', 'e', 's', 't', 'i', 'n', 'g'};
        n = eshkol_pb_write_tag(buf, 2, ESHKOL_PB_WIRE_LEN);
        n += eshkol_pb_write_varint(buf + n, 7);
        memcpy(buf + n, "testing", 7);
        n += 7;
        check(bytes_equal(buf, n, test2, sizeof test2), "spec Test2 b=\"testing\"");

        static const uint8_t test3[] = {0x1A, 0x03, 0x08, 0x96, 0x01};
        n = eshkol_pb_write_tag(buf, 3, ESHKOL_PB_WIRE_LEN);
        n += eshkol_pb_write_varint(buf + n, sizeof test1);
        memcpy(buf + n, test1, sizeof test1);
        n += sizeof test1;
        check(bytes_equal(buf, n, test3, sizeof test3), "spec Test3 c={a=150}");

        static const uint8_t packed[] = {0x32, 0x06, 0x03, 0x8E, 0x02, 0x9E, 0xA7, 0x05};
        static const uint64_t d[] = {3, 270, 86942};
        size_t payload = 0;
        for (size_t i = 0; i < 3; ++i) payload += eshkol_pb_varint_size(d[i]);
        n = eshkol_pb_write_tag(buf, 6, ESHKOL_PB_WIRE_LEN);
        n += eshkol_pb_write_varint(buf + n, payload);
        for (size_t i = 0; i < 3; ++i) n += eshkol_pb_write_varint(buf + n, d[i]);
        check(bytes_equal(buf, n, packed, sizeof packed), "spec packed d=[3,270,86942]");

        size_t offset = 0, len = 0;
        check(eshkol_pb_read_len(test2 + 1, sizeof test2 - 1, &offset, &len) == 8 &&
              offset == 1 && len == 7, "read LEN payload");
        check(eshkol_pb_read_len(test2 + 1, sizeof test2 - 2, &offset, &len) == 0,
              "LEN longer than input refused");
    }

    /* Malformed varints. */
    {
        uint64_t v = 0;
        static const uint8_t eleven[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01};
        static const uint8_t overflow10[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                             0xFF, 0xFF, 0xFF, 0xFF, 0x02};
        static const uint8_t lone_continuation[] = {0x80};
        check(eshkol_pb_read_varint(eleven, sizeof eleven, &v) == 0, "11-byte varint refused");
        check(eshkol_pb_read_varint(overflow10, sizeof overflow10, &v) == 0,
              "varint above 2^64 refused");
        check(eshkol_pb_read_varint(lone_continuation, 1, &v) == 0, "truncated varint refused");
        check(eshkol_pb_read_varint(lone_continuation, 0, &v) == 0, "empty input refused");
    }

    printf("protobuf wire primitives: %d checks, %d failed\n", checks, failures);
    if (failures == 0) printf("PASS: protobuf wire golden vectors\n");
    return failures == 0 ? 0 : 1;
}
