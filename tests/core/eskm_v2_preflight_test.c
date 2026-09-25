/* Bounded, deterministic tests for the private experimental v2 contract. */
#include "eskm_v2_preflight.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAPACITY 70000
#define GUARD UINT64_C(0x9ac31f2476e805bd)
typedef struct bytes { uint8_t data[CAPACITY]; size_t size; } bytes;
static unsigned checks;
static unsigned failures;
static const eskm_v2_limits defaults = ESKM_V2_LIMITS_DEFAULT;

static void check(int condition, const char *name) {
    ++checks;
    if (!condition) { ++failures; fprintf(stderr, "FAIL: %s\n", name); }
}

static void put(bytes *b, uint64_t value, size_t width) {
    if (width > CAPACITY - b->size) { fputs("test buffer capacity\n", stderr); exit(2); }
    for (size_t i = 0; i < width; ++i) { b->data[b->size++] = (uint8_t)value; value >>= 8; }
}

static void patch(bytes *b, size_t offset, uint64_t value, size_t width) {
    if (offset > b->size || width > b->size - offset) { fputs("test patch bounds\n", stderr); exit(2); }
    for (size_t i = 0; i < width; ++i) { b->data[offset + i] = (uint8_t)value; value >>= 8; }
}

static uint64_t get(const uint8_t *data, size_t width) {
    uint64_t value = 0;
    for (size_t i = width; i > 0; --i) value = (value << 8) | data[i - 1];
    return value;
}

static uint32_t checksum(const uint8_t *data, size_t size) {
    uint32_t crc = UINT32_MAX;
    while (size--) {
        crc ^= *data++;
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc & 1) ? (crc >> 1) ^ UINT32_C(0xedb88320) : crc >> 1;
    }
    return crc ^ UINT32_MAX;
}

static bytes header(uint32_t records) {
    bytes b = {{0}, 0};
    put(&b, UINT32_C(0x4d4b5345), 4); put(&b, 2, 4);
    put(&b, records, 4); put(&b, 0, 4); put(&b, 0, 8);
    return b;
}

static void seal(bytes *b) { put(b, checksum(b->data, b->size), 4); }
static void reseal(bytes *b) { patch(b, b->size - 4, checksum(b->data, b->size - 4), 4); }
static void end_extensions(bytes *b) { patch(b, 16, b->size - 24, 8); }
static void tlv(bytes *b, uint32_t type, uint32_t length) { put(b, type, 4); put(b, length, 4); }

static void entry(bytes *b, const uint8_t *key, uint32_t key_size,
                  const uint8_t *value, uint32_t value_size) {
    put(b, key_size, 4); put(b, value_size, 4);
    for (uint32_t i = 0; i < key_size; ++i) put(b, key[i], 1);
    for (uint32_t i = 0; i < value_size; ++i) put(b, value[i], 1);
}

static size_t annotation_start(bytes *b, uint32_t count) {
    size_t start = b->size;
    tlv(b, 1, 0); put(b, count, 4);
    return start;
}
static void annotation_end(bytes *b, size_t start) { patch(b, start + 4, b->size - start - 8, 4); }

static void record(bytes *b, const uint64_t *dimensions, uint32_t rank,
                   const uint64_t *elements, size_t count) {
    put(b, 0, 4); put(b, rank, 4);
    for (uint32_t i = 0; i < rank; ++i) put(b, dimensions[i], 8);
    put(b, 0, 1);
    for (size_t i = 0; i < count; ++i) put(b, elements[i], 8);
}

static int same_result(eskm_v2_result a, eskm_v2_result b) {
    return a.extension_offset == b.extension_offset && a.extension_bytes == b.extension_bytes &&
           a.records_offset == b.records_offset && a.records_bytes == b.records_bytes &&
           a.record_count == b.record_count && a.tlv_count == b.tlv_count &&
           a.annotation_count == b.annotation_count && a.has_annotations == b.has_annotations;
}

static int same_error(eskm_v2_error error, eskm_v2_status status, uint64_t offset) {
    return error.status == status && error.offset == offset;
}

static eskm_v2_result run(const char *name, const bytes *b, const eskm_v2_limits *limits,
                          eskm_v2_status status, uint64_t offset) {
    struct { uint64_t before; eskm_v2_workspace workspace; uint64_t after; } guarded;
    uint8_t snapshot[CAPACITY];
    eskm_v2_result result;
    guarded.before = guarded.after = GUARD;
    memset(&guarded.workspace, 0xa5, sizeof(guarded.workspace));
    memset(&result, 0xa5, sizeof(result));
    memcpy(snapshot, b->data, b->size);
    eskm_v2_error error = eskm_v2_preflight(b->data, b->size, limits, &guarded.workspace, &result);
    if (!same_error(error, status, offset)) {
        fprintf(stderr, "%s: got status %d @%" PRIu64 ", expected %d @%" PRIu64 "\n",
                name, error.status, error.offset, status, offset);
    }
    check(same_error(error, status, offset), name);
    check(guarded.before == GUARD && guarded.after == GUARD, "workspace canaries");
    check(memcmp(snapshot, b->data, b->size) == 0, "input remains immutable");
    if (status != ESKM_V2_OK) {
        const uint8_t zero[sizeof(result)] = {0};
        check(memcmp(&result, zero, sizeof(result)) == 0, "entire failed result cleared");
    }
    return result;
}

static eskm_v2_result accept(const char *name, const bytes *b) { return run(name, b, &defaults, ESKM_V2_OK, 0); }
static void reject(const char *name, const bytes *b, eskm_v2_status status, uint64_t offset) {
    (void)run(name, b, &defaults, status, offset);
}

static bytes fixture(const char *directory, const char *name) {
    char path[4096];
    bytes b = {{0}, 0};
    if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= (int)sizeof(path)) exit(2);
    FILE *file = fopen(path, "rb");
    if (!file) { perror(path); exit(2); }
    b.size = fread(b.data, 1, sizeof(b.data), file);
    if (ferror(file) || !feof(file) || fclose(file)) { fputs("fixture read failed\n", stderr); exit(2); }
    return b;
}

static void goldens(const char *directory) {
    bytes b = fixture(directory, "empty.eskm");
    static const uint8_t empty[] = {
        0x45,0x53,0x4b,0x4d,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0x70,0xe8,0xc3
    };
    check(b.size == sizeof(empty) && memcmp(b.data, empty, sizeof(empty)) == 0, "canonical empty bytes");
    check(checksum(b.data, 24) == UINT32_C(0xc3e87005), "independent canonical CRC");
    eskm_v2_result expected = {24, 0, 24, 0, 0, 0, 0, 0};
    check(same_result(accept("empty golden", &b), expected), "empty view");
    b = fixture(directory, "raw-bits.eskm");
    expected = (eskm_v2_result){24, 0, 24, 76, 1, 0, 0, 0};
    check(same_result(accept("raw-bits golden", &b), expected), "raw-bits view");
    check(get(b.data + 24, 4) == 3 && memcmp(b.data + 28, "w\0\xff", 3) == 0,
          "opaque record name");
    check(get(b.data + 31, 4) == 2 && get(b.data + 35, 8) == 2 &&
          get(b.data + 43, 8) == 3 && b.data[51] == 0, "record metadata");
    const uint64_t bits[] = {0, UINT64_C(0x8000000000000000), UINT64_C(0x3ff0000000000000),
        UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000), UINT64_C(0x7ff8000000001234)};
    for (size_t i = 0; i < 6; ++i) check(get(b.data + 52 + i * 8, 8) == bits[i], "exact raw binary64 bits");
    b = fixture(directory, "annotations-scalar.eskm");
    expected = (eskm_v2_result){24, 62, 86, 17, 1, 3, 3, 1};
    check(same_result(accept("annotations scalar golden", &b), expected), "annotation view");
}

static void arguments(void) {
    bytes b = header(0); seal(&b);
    eskm_v2_workspace workspace;
    eskm_v2_result result;
    const uint8_t zero[sizeof(result)] = {0};
    for (unsigned missing = 0; missing < 4; ++missing) {
        memset(&result, 0xa5, sizeof(result));
        eskm_v2_error error = eskm_v2_preflight(missing == 0 ? NULL : b.data, b.size,
            missing == 1 ? NULL : &defaults, missing == 2 ? NULL : &workspace,
            missing == 3 ? NULL : &result);
        check(error.status == ESKM_V2_INVALID_ARGUMENT && error.offset == 0, "null argument");
        if (missing != 3) check(memcmp(&result, zero, sizeof(result)) == 0, "argument failure clears result");
    }
    for (size_t size = 0; size < 28; ++size) {
        bytes short_file = b; short_file.size = size;
        reject("short complete container", &short_file, ESKM_V2_MALFORMED, 0);
    }
    eskm_v2_limits limits = defaults;
    uint64_t *fields[] = {&limits.buffer_bytes, &limits.extension_bytes, &limits.tlvs,
                         &limits.annotations, &limits.key_bytes, &limits.value_bytes};
    for (size_t i = 0; i < 6; ++i) {
        ++*fields[i];
        run("cannot raise compiled limit", &b, &limits, ESKM_V2_INVALID_LIMITS, 0);
        --*fields[i];
    }
    b.data[0] = 0;
    limits.buffer_bytes = b.size - 1;
    run("buffer cap before bad magic", &b, &limits, ESKM_V2_RESOURCE_LIMIT, 0);
    limits.key_bytes = ESKM_V2_MAX_KEY_BYTES + 1;
    run("invalid limits before buffer cap", &b, &limits, ESKM_V2_INVALID_LIMITS, 0);
}

static void header_failures(void) {
    bytes control = header(0); seal(&control); accept("header failure control", &control);
    const struct mutation { const char *name; size_t field; uint64_t value; size_t width;
                            eskm_v2_status status; uint64_t offset; } cases[] = {
        {"bad magic",0,0,1,ESKM_V2_BAD_MAGIC,0},
        {"old version",4,1,4,ESKM_V2_UNSUPPORTED_VERSION,4},
        {"future version",4,UINT32_MAX,4,ESKM_V2_UNSUPPORTED_VERSION,4},
        {"required feature",12,1,4,ESKM_V2_UNSUPPORTED_FEATURES,12},
        {"high required feature",12,UINT32_C(0x80000000),4,ESKM_V2_UNSUPPORTED_FEATURES,12},
        {"bad checksum",24,0,4,ESKM_V2_CHECKSUM_MISMATCH,24}
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        bytes b = control; patch(&b, cases[i].field, cases[i].value, cases[i].width);
        reject(cases[i].name, &b, cases[i].status, cases[i].offset);
    }
    bytes b = control; patch(&b, 16, 1, 8);
    reject("checksum before extension structure", &b, ESKM_V2_CHECKSUM_MISMATCH, 24);
    reseal(&b); reject("extension beyond container", &b, ESKM_V2_MALFORMED, 16);
    patch(&b, 16, UINT64_MAX, 8); reseal(&b);
    reject("extension cap before range", &b, ESKM_V2_RESOURCE_LIMIT, 16);
    patch(&b, 12, 1, 4);
    reject("features before checksum and extension", &b, ESKM_V2_UNSUPPORTED_FEATURES, 12);
    patch(&b, 4, 1, 4);
    reject("version before features", &b, ESKM_V2_UNSUPPORTED_VERSION, 4);
    patch(&b, 0, 0, 1);
    reject("magic before version", &b, ESKM_V2_BAD_MAGIC, 0);
    b = control; patch(&b, 8, 1, 4); reseal(&b);
    reject("missing declared record", &b, ESKM_V2_MALFORMED, 24);
    b = header(0); put(&b, 0, 1); seal(&b);
    reject("unclaimed trailing byte", &b, ESKM_V2_MALFORMED, 24);
}

static void extensions(void) {
    bytes b = header(0);
    tlv(&b, 2, 0); tlv(&b, UINT32_MAX, 2); put(&b, 0xff00, 2); tlv(&b, 2, 0);
    end_extensions(&b); seal(&b);
    check(accept("repeated opaque unknown TLVs", &b).tlv_count == 3, "unknown TLV count");
    const struct { size_t bytes; uint32_t type; uint32_t length; uint64_t offset; } cases[] = {
        {1,2,0,24}, {3,2,0,24}, {4,2,0,28}, {7,2,0,28}, {8,0,0,24},
        {8,2,1,28}, {8,2,UINT32_MAX,28}, {8,1,0,32}, {11,1,3,32}
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        b = header(0); tlv(&b, cases[i].type, cases[i].length);
        if (cases[i].bytes < 8) b.size = 24 + cases[i].bytes;
        else while (b.size < 24 + cases[i].bytes) put(&b, 0, 1);
        end_extensions(&b); seal(&b);
        reject("TLV framing rejection", &b, ESKM_V2_MALFORMED, cases[i].offset);
    }
    b = header(0); size_t start = annotation_start(&b, 0); annotation_end(&b, start);
    end_extensions(&b); seal(&b);
    eskm_v2_result result = accept("empty annotation map", &b);
    check(result.has_annotations == 1 && result.annotation_count == 0, "empty map presence");
    b.size -= 4; annotation_start(&b, 0); end_extensions(&b); seal(&b);
    reject("repeated annotation map", &b, ESKM_V2_MALFORMED, 36);
    b = header(0); start = annotation_start(&b, 0); put(&b, 0, 1);
    annotation_end(&b, start); end_extensions(&b); seal(&b);
    reject("annotation trailing byte", &b, ESKM_V2_MALFORMED, 36);
}

static void annotations(void) {
    const uint8_t a[] = {'a'}, z[] = {'z'}, prefix[] = {'a',0}, opaque[] = {0xff,0};
    bytes b = header(0); size_t start = annotation_start(&b, 4);
    entry(&b, z, 1, NULL, 0); entry(&b, prefix, 2, opaque, 2);
    entry(&b, a, 1, NULL, 0); entry(&b, opaque, 2, NULL, 0);
    annotation_end(&b, start); end_extensions(&b); seal(&b);
    check(accept("unordered opaque prefix keys", &b).annotation_count == 4, "four annotation entries");
    b = header(0); start = annotation_start(&b, 4);
    entry(&b, z, 1, NULL, 0); entry(&b, z, 1, NULL, 0);
    entry(&b, a, 1, NULL, 0); entry(&b, a, 1, NULL, 0);
    annotation_end(&b, start); end_extensions(&b); seal(&b);
    reject("earliest later duplicate across lexical groups", &b, ESKM_V2_MALFORMED, 45);
    patch(&b, 63, 0, 4); reseal(&b);
    reject("later malformed entry before duplicate scan", &b, ESKM_V2_MALFORMED, 63);
    const struct { uint32_t key; uint32_t value; size_t body; uint64_t offset; } cases[] = {
        {1,0,0,36}, {1,0,3,36}, {1,0,4,40}, {1,0,7,40},
        {0,0,8,36}, {2,0,9,36}, {1,2,10,40},
        {4097,0,8,36}, {1,65537,8,40}
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        b = header(0); start = annotation_start(&b, 1);
        put(&b, cases[i].key, 4); put(&b, cases[i].value, 4);
        b.size = 36 + cases[i].body;
        if (cases[i].body > 8) memset(b.data + 44, 'x', cases[i].body - 8);
        annotation_end(&b, start); end_extensions(&b); seal(&b);
        eskm_v2_status status = i >= 7 ? ESKM_V2_RESOURCE_LIMIT : ESKM_V2_MALFORMED;
        reject("annotation field rejection", &b, status, cases[i].offset);
    }
    b = header(0); start = annotation_start(&b, 1025); annotation_end(&b, start);
    end_extensions(&b); seal(&b);
    reject("annotation compiled count cap", &b, ESKM_V2_RESOURCE_LIMIT, 32);
    b = header(0); start = annotation_start(&b, 1024);
    for (uint32_t i = 1024; i > 0; --i) {
        uint8_t key[] = {(uint8_t)(i >> 8), (uint8_t)i};
        entry(&b, key, 2, NULL, 0);
    }
    annotation_end(&b, start); end_extensions(&b); seal(&b);
    check(accept("all workspace entries reverse order", &b).annotation_count == 1024, "workspace maximum");
    patch(&b, 44 + 1023 * 10, 1024 >> 8, 1);
    patch(&b, 45 + 1023 * 10, 1024 & 255, 1); reseal(&b);
    reject("duplicate at workspace end", &b, ESKM_V2_MALFORMED, 36 + 1023 * 10);
}

static void records(void) {
    const uint64_t scalar = UINT64_C(0x7ff0000000000001);
    const uint64_t zero_first[] = {0, UINT64_MAX, UINT64_MAX};
    const uint64_t zero_second[] = {UINT64_MAX, 0, UINT64_MAX};
    const uint64_t overflow_first[] = {UINT64_MAX, 2, 0};
    const uint64_t payload_overflow[] = {UINT64_MAX};
    bytes b = header(1); record(&b, NULL, 0, &scalar, 1); seal(&b);
    check(accept("rank zero scalar", &b).records_bytes == 17, "scalar size");
    bytes control = b;
    patch(&b, 32, 1, 1); reseal(&b);
    reject("unknown dtype", &b, ESKM_V2_MALFORMED, 32);
    b = control; b.size -= 1; reseal(&b);
    reject("truncated scalar payload", &b, ESKM_V2_MALFORMED, 32);
    b = header(1); record(&b, zero_first, 3, NULL, 0); seal(&b); accept("zero before huge dimensions", &b);
    b = header(1); record(&b, zero_second, 3, NULL, 0); seal(&b); accept("huge dimension before zero without overflow", &b);
    b = header(1); record(&b, overflow_first, 3, NULL, 0); seal(&b);
    reject("ordered overflow before zero", &b, ESKM_V2_MALFORMED, 40);
    b = header(1); record(&b, payload_overflow, 1, NULL, 0); seal(&b);
    reject("binary64 payload multiplication overflow", &b, ESKM_V2_MALFORMED, 40);
    b = header(2); record(&b, NULL, 0, &scalar, 1); record(&b, NULL, 0, &scalar, 1); seal(&b);
    check(accept("duplicate names allowed", &b).record_count == 2, "multiple record count");
    patch(&b, 49, 1, 1); reseal(&b);
    reject("later record failure clears complete output", &b, ESKM_V2_MALFORMED, 49);
    const struct { size_t body; uint32_t name; uint32_t rank; uint64_t offset; } cases[] = {
        {0,0,0,24}, {3,0,0,24}, {4,1,0,24}, {4,UINT32_MAX,0,24},
        {4,0,0,28}, {7,0,0,28}, {8,0,1,28}, {8,0,UINT32_MAX,28}, {8,0,0,32}
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        b = header(1); put(&b, cases[i].name, 4); put(&b, cases[i].rank, 4);
        b.size = 24 + cases[i].body; seal(&b);
        reject("record field rejection", &b, ESKM_V2_MALFORMED, cases[i].offset);
    }
    /* Zero ends multiplication, not consumption: a missing later dimension fails. */
    b = header(1); put(&b, 0, 4); put(&b, 2, 4); put(&b, 0, 8); seal(&b);
    reject("all dimensions required after zero", &b, ESKM_V2_MALFORMED, 28);
}

static void limits(void) {
    const uint8_t key[] = {'a','b'}, value[] = {0,1,2};
    bytes b = header(0); size_t start = annotation_start(&b, 1);
    entry(&b, key, 2, value, 3); annotation_end(&b, start); end_extensions(&b); seal(&b);
    const uint64_t used[] = {b.size, b.size - 28, 1, 1, 2, 3};
    const uint64_t offsets[] = {0,16,24,32,36,40};
    for (size_t resource = 0; resource < 6; ++resource) {
        for (int delta = -1; delta <= 1; ++delta) {
            eskm_v2_limits lower = defaults;
            uint64_t *fields[] = {&lower.buffer_bytes, &lower.extension_bytes, &lower.tlvs,
                                 &lower.annotations, &lower.key_bytes, &lower.value_bytes};
            *fields[resource] = used[resource] + (uint64_t)delta;
            run("lowered inclusive limit boundary", &b, &lower,
                delta < 0 ? ESKM_V2_RESOURCE_LIMIT : ESKM_V2_OK, delta < 0 ? offsets[resource] : 0);
        }
    }
    bytes empty = header(0); seal(&empty);
    eskm_v2_limits zero = {28,0,0,0,0,0};
    run("zero metadata caps permit no metadata", &empty, &zero, ESKM_V2_OK, 0);
    zero.buffer_bytes = 0;
    run("zero buffer cap is not default", &empty, &zero, ESKM_V2_RESOURCE_LIMIT, 0);
    b = header(0); start = annotation_start(&b, 0); annotation_end(&b, start); end_extensions(&b); seal(&b);
    zero = defaults; zero.annotations = zero.key_bytes = zero.value_bytes = 0;
    run("zero entry cap accepts present empty map", &b, &zero, ESKM_V2_OK, 0);
    b = header(0); start = annotation_start(&b, 1); entry(&b, key, 2, NULL, 0);
    annotation_end(&b, start); end_extensions(&b); seal(&b);
    zero = defaults; zero.value_bytes = 0;
    run("zero value cap accepts empty values", &b, &zero, ESKM_V2_OK, 0);
    b = header(0);
    for (uint32_t i = 0; i < 1024; ++i) tlv(&b, 2, 0);
    end_extensions(&b); seal(&b); accept("maximum optional TLVs", &b);
    b.size -= 4; tlv(&b, 2, 0); end_extensions(&b); seal(&b);
    reject("optional TLV compiled cap", &b, ESKM_V2_RESOURCE_LIMIT, 24 + 1024 * 8);
    uint8_t long_key[4096], long_value[65536];
    memset(long_key, 0xff, sizeof(long_key)); memset(long_value, 0, sizeof(long_value));
    b = header(0); start = annotation_start(&b, 1);
    entry(&b, long_key, sizeof(long_key), NULL, 0);
    annotation_end(&b, start); end_extensions(&b); seal(&b); accept("maximum annotation key", &b);
    b = header(0); start = annotation_start(&b, 1);
    entry(&b, key, 2, long_value, sizeof(long_value));
    annotation_end(&b, start); end_extensions(&b); seal(&b); accept("maximum annotation value", &b);
}

static int negative_control(const char *directory, const char *mode) {
    bytes b = fixture(directory, "empty.eskm");
    eskm_v2_workspace workspace;
    eskm_v2_result result;
    eskm_v2_error error = eskm_v2_preflight(b.data, b.size, &defaults, &workspace, &result);
    const eskm_v2_result expected = {24,0,24,0,0,0,0,0};
    if (!same_error(error, ESKM_V2_OK, 0) || !same_result(result, expected)) return 2;
    int rejected = 0;
    if (strcmp(mode, "status") == 0) rejected = !same_error(error, ESKM_V2_MALFORMED, 0);
    else if (strcmp(mode, "offset") == 0) rejected = !same_error(error, ESKM_V2_OK, 1);
    else if (strcmp(mode, "metadata") == 0) {
        eskm_v2_result changed = expected; changed.records_offset = 25;
        rejected = !same_result(result, changed);
    } else return 2;
    if (rejected) { printf("negative control rejected: %s\n", mode); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strncmp(argv[2], "--negative-control=", 19) == 0)
        return negative_control(argv[1], argv[2] + 19);
    if (argc != 2) { fprintf(stderr, "usage: %s FIXTURE_DIRECTORY [--negative-control=MODE]\n", argv[0]); return 2; }
    goldens(argv[1]); arguments(); header_failures(); extensions(); annotations(); records(); limits();
    printf("ESKM v2 preflight: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
