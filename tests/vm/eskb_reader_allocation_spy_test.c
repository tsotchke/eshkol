#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eskb_format.h"
#include "eshkol/backend/vm_limits.h"

static int allocator_spy_active;
static unsigned oversized_attempts;
static size_t largest_attempt;
static unsigned malloc_calls;
static unsigned calloc_calls;
static unsigned realloc_calls;
static int check_allocation_counts;
static unsigned expected_malloc_calls;
static unsigned expected_calloc_calls;
static unsigned expected_realloc_calls;

static void *spy_malloc(size_t size);
static void *spy_calloc(size_t count, size_t size);
static void *spy_realloc(void *ptr, size_t size);

#define malloc spy_malloc
#define calloc spy_calloc
#define realloc spy_realloc
#ifndef ESKB_READER_SOURCE
#define ESKB_READER_SOURCE "../../lib/backend/eskb_reader.c"
#endif
#include ESKB_READER_SOURCE
#undef malloc
#undef calloc
#undef realloc

enum { BUFFER_CAPACITY = 1024 * 1024, ALLOCATION_DENY_AT = 64 * 1024 };

typedef struct {
    uint8_t bytes[BUFFER_CAPACITY];
    size_t len;
} Buffer;

static Buffer entries;
static Buffer const_section;
static Buffer code_section;
static Buffer instructions;
static Buffer payload;
static Buffer module_bytes;

static void buffer_reset(Buffer *buffer) {
    buffer->len = 0;
}

static int buffer_put(Buffer *buffer, const void *data, size_t size) {
    if (size > BUFFER_CAPACITY - buffer->len) return -1;
    memcpy(buffer->bytes + buffer->len, data, size);
    buffer->len += size;
    return 0;
}

static int buffer_u8(Buffer *buffer, uint8_t value) {
    return buffer_put(buffer, &value, 1);
}

static int buffer_leb(Buffer *buffer, uint64_t value) {
    uint8_t encoded[10];
    int count = eskb_write_leb128(encoded, value);
    return buffer_put(buffer, encoded, (size_t)count);
}

static int buffer_instruction(Buffer *buffer, uint8_t opcode, int32_t operand) {
    uint32_t zigzag = ((uint32_t)operand << 1) ^ (uint32_t)-(operand < 0);
    return buffer_u8(buffer, opcode) < 0 || buffer_leb(buffer, zigzag) < 0 ? -1 : 0;
}

static int make_code_section(uint64_t function_count, uint64_t code_len,
                             const Buffer *code_bytes) {
    buffer_reset(&code_section);
    if (buffer_leb(&code_section, function_count) < 0) return -1;
    if (function_count != 1) return 0;
    int minimum_header = code_len == 0 && code_bytes->len == 0;
    if (buffer_leb(&code_section, minimum_header ? 1 : 4) < 0 ||
        buffer_put(&code_section, minimum_header ? "m" : "main",
                   minimum_header ? 1 : 4) < 0 ||
        buffer_u8(&code_section, 0) < 0 ||
        buffer_leb(&code_section, 0) < 0 ||
        buffer_leb(&code_section, 0) < 0 ||
        buffer_leb(&code_section, code_len) < 0 ||
        buffer_put(&code_section, code_bytes->bytes, code_bytes->len) < 0) return -1;
    return 0;
}

static int make_module(uint64_t advertised_constants, const Buffer *constant_entries,
                       uint64_t function_count, uint64_t code_len,
                       const Buffer *code_bytes) {
    buffer_reset(&const_section);
    if (buffer_leb(&const_section, advertised_constants) < 0 ||
        buffer_put(&const_section, constant_entries->bytes, constant_entries->len) < 0 ||
        make_code_section(function_count, code_len, code_bytes) < 0) return -1;

    buffer_reset(&payload);
    if (buffer_leb(&payload, 2) < 0 ||
        buffer_u8(&payload, ESKB_SECTION_CONST) < 0 ||
        buffer_leb(&payload, const_section.len) < 0 ||
        buffer_u8(&payload, ESKB_SECTION_CODE) < 0 ||
        buffer_leb(&payload, code_section.len) < 0 ||
        buffer_put(&payload, const_section.bytes, const_section.len) < 0 ||
        buffer_put(&payload, code_section.bytes, code_section.len) < 0) return -1;

    EskbHeader header = {ESKB_MAGIC, ESKB_VERSION, ESKB_FLAG_LITTLE_ENDIAN,
                         eskb_crc32(payload.bytes, payload.len)};
    buffer_reset(&module_bytes);
    return buffer_put(&module_bytes, &header, sizeof(header)) < 0 ||
           buffer_put(&module_bytes, payload.bytes, payload.len) < 0 ? -1 : 0;
}

static int expect_load(const char *name, int expected_result,
                       unsigned expected_oversized_min,
                       unsigned expected_oversized_max) {
    EskbModule module = {0};
    oversized_attempts = 0;
    largest_attempt = 0;
    malloc_calls = 0;
    calloc_calls = 0;
    realloc_calls = 0;
    allocator_spy_active = 1;
    int result = eskb_load_memory(module_bytes.bytes, module_bytes.len, &module);
    allocator_spy_active = 0;
    eskb_module_free(&module);
    if ((expected_result == 0 && result != 0) ||
        (expected_result != 0 && result == 0) ||
        oversized_attempts < expected_oversized_min ||
        oversized_attempts > expected_oversized_max ||
        (check_allocation_counts &&
         (malloc_calls != expected_malloc_calls ||
          calloc_calls != expected_calloc_calls ||
          realloc_calls != expected_realloc_calls))) {
        fprintf(stderr, "FAIL: %s result=%d oversized=%u largest=%zu malloc=%u calloc=%u realloc=%u\n",
                name, result, oversized_attempts, largest_attempt,
                malloc_calls, calloc_calls, realloc_calls);
        check_allocation_counts = 0;
        return -1;
    }
    if (expected_oversized_min > 0) {
        printf("RED: %s attempted %u oversized allocation(s), largest=%zu bytes; denied\n",
               name, oversized_attempts, largest_attempt);
    }
    check_allocation_counts = 0;
    return 0;
}

static int expect_load_counts(const char *name, int expected_result,
                              unsigned expected_malloc,
                              unsigned expected_calloc,
                              unsigned expected_realloc) {
    check_allocation_counts = 1;
    expected_malloc_calls = expected_malloc;
    expected_calloc_calls = expected_calloc;
    expected_realloc_calls = expected_realloc;
    return expect_load(name, expected_result, 0, 0);
}

static int prepare_halt(void) {
    buffer_reset(&instructions);
    return buffer_instruction(&instructions, 36, 0);
}

static int prepare_large_constant_count(void) {
    buffer_reset(&entries);
    if (prepare_halt() < 0) return -1;
    return make_module((uint64_t)ESHKOL_VM_MAX_CONSTS_CEILING + 1,
                       &entries, 1, 1, &instructions);
}

static int prepare_large_string_length(void) {
    buffer_reset(&entries);
    if (buffer_u8(&entries, ESKB_CONST_STRING) < 0 ||
        buffer_leb(&entries, UINT64_C(1) << 30) < 0 || prepare_halt() < 0) return -1;
    return make_module(1, &entries, 1, 1, &instructions);
}

static int prepare_large_function_count(void) {
    buffer_reset(&entries);
    buffer_reset(&instructions);
    return make_module(0, &entries, 4096, 0, &instructions);
}

static int prepare_large_instruction_count(void) {
    buffer_reset(&entries);
    buffer_reset(&instructions);
    return make_module(0, &entries, 1, 65536, &instructions);
}

static int prepare_short_constant_count(void) {
    buffer_reset(&entries);
    if (buffer_u8(&entries, ESKB_CONST_NIL) < 0 || prepare_halt() < 0) return -1;
    return make_module(2, &entries, 1, 1, &instructions);
}

static int prepare_short_string_length(void) {
    buffer_reset(&entries);
    if (buffer_u8(&entries, ESKB_CONST_STRING) < 0 ||
        buffer_leb(&entries, 5) < 0 || buffer_u8(&entries, 'x') < 0 ||
        prepare_halt() < 0) return -1;
    return make_module(1, &entries, 1, 1, &instructions);
}

static int prepare_short_instruction_count(void) {
    buffer_reset(&entries);
    buffer_reset(&instructions);
    return make_module(0, &entries, 1, 1, &instructions);
}

static int run_adversarial(const char *name, int (*prepare)(void), int expect_red) {
    if (prepare() < 0 || module_bytes.len > BUFFER_CAPACITY) {
        fprintf(stderr, "FAIL: fixture construction: %s\n", name);
        return -1;
    }
    return expect_load(name, -1, expect_red ? 1 : 0, expect_red ? UINT32_MAX : 0);
}

static int run_fixed_positive_cases(void) {
    buffer_reset(&entries);
    if (prepare_halt() < 0 || make_module(0, &entries, 1, 1, &instructions) < 0 ||
        expect_load("valid-empty-pool", 0, 0, 0) < 0) return -1;

    buffer_reset(&entries);
    if (buffer_u8(&entries, ESKB_CONST_STRING) < 0 ||
        buffer_leb(&entries, 1) < 0 || buffer_u8(&entries, 'x') < 0 ||
        prepare_halt() < 0 || make_module(1, &entries, 1, 1, &instructions) < 0 ||
        expect_load("valid-string-at-section-boundary", 0, 0, 0) < 0) return -1;

    if (prepare_short_constant_count() < 0 ||
        expect_load_counts("constant-count-over-remaining", -1, 0, 0, 0) < 0) return -1;
    if (prepare_short_string_length() < 0 ||
        expect_load_counts("string-length-over-remaining", -1, 0, 4, 0) < 0) return -1;
    if (prepare_short_instruction_count() < 0 ||
        expect_load_counts("instruction-count-over-remaining", -1, 1, 5, 0) < 0) return -1;

    /* A six-byte nonempty-name function header is the exact nf*6 lower bound.
     * The reader accepts it; VM profile validation separately rejects empty code. */
    buffer_reset(&entries);
    buffer_reset(&instructions);
    if (make_module(0, &entries, 1, 0, &instructions) < 0 ||
        expect_load("function-count-at-minimum-header", 0, 0, 0) < 0) return -1;

    buffer_reset(&entries);
    for (uint64_t i = 0; i < 4097; i++) {
        if (buffer_u8(&entries, ESKB_CONST_NIL) < 0) return -1;
    }
    buffer_reset(&instructions);
    if (buffer_instruction(&instructions, 1, 4096) < 0 ||
        buffer_instruction(&instructions, 5, 0) < 0 ||
        buffer_instruction(&instructions, 36, 0) < 0 ||
        make_module(4097, &entries, 1, 3, &instructions) < 0 ||
        expect_load("valid-pool-over-4096", 0, 0, 0) < 0) return -1;
    return 0;
}

int main(int argc, char **argv) {
    int expect_red = argc == 2 && strcmp(argv[1], "--expect-red") == 0;
    if (argc > 2 || (argc == 2 && !expect_red && strcmp(argv[1], "--expect-green") != 0)) {
        fprintf(stderr, "usage: %s [--expect-red|--expect-green]\n", argv[0]);
        return 2;
    }
    if (run_adversarial("constant-count-over-ceiling", prepare_large_constant_count,
                        expect_red) < 0 ||
        run_adversarial("string-length-over-remaining", prepare_large_string_length,
                        expect_red) < 0 ||
        run_adversarial("function-count-over-remaining", prepare_large_function_count,
                        expect_red) < 0 ||
        run_adversarial("instruction-count-over-remaining", prepare_large_instruction_count,
                        expect_red) < 0) return 1;

    if (!expect_red && run_fixed_positive_cases() < 0) return 1;
    printf("PASS: allocator-spy %s; oversized allocation attempts %s\n",
           expect_red ? "baseline RED reproduced" : "bounds enforced",
           expect_red ? "observed and denied" : "zero");
    return 0;
}

static void note_attempt(size_t size) {
    if (!allocator_spy_active || size < ALLOCATION_DENY_AT) return;
    oversized_attempts++;
    if (size > largest_attempt) largest_attempt = size;
}

static void *spy_malloc(size_t size) {
    if (allocator_spy_active) malloc_calls++;
    note_attempt(size);
    if (allocator_spy_active && size >= ALLOCATION_DENY_AT) return NULL;
    return malloc(size);
}

static void *spy_calloc(size_t count, size_t size) {
    if (allocator_spy_active) calloc_calls++;
    size_t bytes = count != 0 && size > SIZE_MAX / count ? SIZE_MAX : count * size;
    note_attempt(bytes);
    if (allocator_spy_active && bytes >= ALLOCATION_DENY_AT) return NULL;
    return calloc(count, size);
}

static void *spy_realloc(void *ptr, size_t size) {
    if (allocator_spy_active) realloc_calls++;
    note_attempt(size);
    if (allocator_spy_active && size >= ALLOCATION_DENY_AT) return NULL;
    return realloc(ptr, size);
}
