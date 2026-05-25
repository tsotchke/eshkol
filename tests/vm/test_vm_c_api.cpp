/* test_vm_c_api.cpp
 *
 * Smoke tests for the public bytecode VM C ABI used by embedders that already
 * have an ESKB chunk in memory.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <eshkol/backend/thread_pool.h>
#include <eshkol/backend/vm.h>

extern "C" {
#include "eskb_format.h"
}

namespace {

enum : uint8_t {
    OP_CONST = 1,
    OP_FALSE = 4,
    OP_ADD = 7,
    OP_JUMP_IF_FALSE = 29,
    OP_HALT = 36,
    OP_NATIVE_CALL = 37,
};

struct Instr {
    uint8_t op;
    int32_t operand;
};

int g_total = 0;
int g_failures = 0;

#define CHECK(cond, name) do {                                  \
        ++g_total;                                              \
        if (cond) {                                             \
            std::printf("[PASS] %s\n", name);                  \
        } else {                                                \
            std::printf("[FAIL] %s (%s:%d)\n",                  \
                        name, __FILE__, __LINE__);              \
            ++g_failures;                                       \
        }                                                       \
    } while (0)

void write_instr(EskbBuffer* b, const Instr& instr) {
    eskb_buf_write_u8(b, instr.op);
    int64_t operand = instr.operand;
    uint64_t zigzag = (static_cast<uint64_t>(operand) << 1) ^
                      static_cast<uint64_t>(operand >> 63);
    eskb_buf_write_leb128(b, zigzag);
}

void write_function(EskbBuffer* b, const char* name, const Instr* code, size_t n_code) {
    eskb_buf_write_string(b, name, std::strlen(name));
    eskb_buf_write_u8(b, 0);
    eskb_buf_write_leb128(b, 0);
    eskb_buf_write_u8(b, 0);
    eskb_buf_write_leb128(b, n_code);
    for (size_t i = 0; i < n_code; ++i) {
        write_instr(b, code[i]);
    }
}

void write_int64_const(EskbBuffer* b, int64_t value) {
    eskb_buf_write_u8(b, ESKB_CONST_INT64);
    eskb_buf_write_i64(b, value);
}

void write_f64_const(EskbBuffer* b, double value) {
    eskb_buf_write_u8(b, ESKB_CONST_F64);
    eskb_buf_write_f64(b, value);
}

uint64_t pack_ascii_string(const char* text) {
    uint64_t packed = 0;
    const size_t len = std::strlen(text);
    for (size_t i = 0; i < len && i < 8; ++i) {
        packed |= static_cast<uint64_t>(static_cast<unsigned char>(text[i])) << (i * 8);
    }
    return packed;
}

EskbBuffer make_number_to_string_radix_chunk(int64_t n, int64_t radix, const char* expected) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 4);
    write_int64_const(&const_buf, n);
    write_int64_const(&const_buf, radix);
    write_int64_const(&const_buf, static_cast<int64_t>(std::strlen(expected)));
    write_int64_const(&const_buf, static_cast<int64_t>(pack_ascii_string(expected)));

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_CONST, 1},
        {OP_NATIVE_CALL, 51},
        {OP_CONST, 2},
        {OP_CONST, 3},
        {OP_NATIVE_CALL, 100},
        {OP_NATIVE_CALL, 560},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));

    eskb_buf_write_leb128(&payload, 2);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);

    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    eskb_buf_write(&file, &hdr, sizeof(hdr));
    eskb_buf_write(&file, payload.data, payload.len);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&payload);
    return file;
}

EskbBuffer make_host_native_int64_chunk(int native_fid) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 2);
    write_int64_const(&const_buf, 18);
    write_int64_const(&const_buf, 19);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_CONST, 1},
        {OP_NATIVE_CALL, native_fid},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));

    eskb_buf_write_leb128(&payload, 2);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);

    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    eskb_buf_write(&file, &hdr, sizeof(hdr));
    eskb_buf_write(&file, payload.data, payload.len);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&payload);
    return file;
}

EskbBuffer make_host_native_double_chunk(int native_fid) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 2);
    write_f64_const(&const_buf, 20.5);
    write_f64_const(&const_buf, 21.5);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_CONST, 1},
        {OP_NATIVE_CALL, native_fid},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));

    eskb_buf_write_leb128(&payload, 2);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);

    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    eskb_buf_write(&file, &hdr, sizeof(hdr));
    eskb_buf_write(&file, payload.data, payload.len);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&payload);
    return file;
}

EskbBuffer make_file_mmap_u8_ref_chunk(const char* path, int64_t offset, int64_t len,
                                       int64_t index) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 5);
    write_int64_const(&const_buf, static_cast<int64_t>(std::strlen(path)));
    write_int64_const(&const_buf, static_cast<int64_t>(pack_ascii_string(path)));
    write_int64_const(&const_buf, offset);
    write_int64_const(&const_buf, len);
    write_int64_const(&const_buf, index);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_CONST, 1},
        {OP_NATIVE_CALL, 100},
        {OP_CONST, 2},
        {OP_CONST, 3},
        {OP_NATIVE_CALL, 1758},
        {OP_CONST, 4},
        {OP_NATIVE_CALL, 682},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));

    eskb_buf_write_leb128(&payload, 2);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);

    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    eskb_buf_write(&file, &hdr, sizeof(hdr));
    eskb_buf_write(&file, payload.data, payload.len);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&payload);
    return file;
}

EskbBuffer make_file_mmap_length_chunk(const char* path, int64_t offset, int64_t len) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 4);
    write_int64_const(&const_buf, static_cast<int64_t>(std::strlen(path)));
    write_int64_const(&const_buf, static_cast<int64_t>(pack_ascii_string(path)));
    write_int64_const(&const_buf, offset);
    write_int64_const(&const_buf, len);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_CONST, 1},
        {OP_NATIVE_CALL, 100},
        {OP_CONST, 2},
        {OP_CONST, 3},
        {OP_NATIVE_CALL, 1758},
        {OP_NATIVE_CALL, 681},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));

    eskb_buf_write_leb128(&payload, 2);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);

    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    eskb_buf_write(&file, &hdr, sizeof(hdr));
    eskb_buf_write(&file, payload.data, payload.len);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&payload);
    return file;
}

EskbBuffer make_native_string_compare_chunk(int native_fid, const char* a, const char* b,
                                           const char* expected) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 6);
    write_int64_const(&const_buf, static_cast<int64_t>(std::strlen(a)));
    write_int64_const(&const_buf, static_cast<int64_t>(pack_ascii_string(a)));
    write_int64_const(&const_buf, static_cast<int64_t>(std::strlen(b)));
    write_int64_const(&const_buf, static_cast<int64_t>(pack_ascii_string(b)));
    write_int64_const(&const_buf, static_cast<int64_t>(std::strlen(expected)));
    write_int64_const(&const_buf, static_cast<int64_t>(pack_ascii_string(expected)));

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_CONST, 1},
        {OP_NATIVE_CALL, 100},
        {OP_CONST, 2},
        {OP_CONST, 3},
        {OP_NATIVE_CALL, 100},
        {OP_NATIVE_CALL, native_fid},
        {OP_CONST, 4},
        {OP_CONST, 5},
        {OP_NATIVE_CALL, 100},
        {OP_NATIVE_CALL, 560},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));

    eskb_buf_write_leb128(&payload, 2);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);

    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    eskb_buf_write(&file, &hdr, sizeof(hdr));
    eskb_buf_write(&file, payload.data, payload.len);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&payload);
    return file;
}

EskbBuffer make_test_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    const std::string long_string(320, 'x');

    eskb_buf_write_leb128(&const_buf, 3);
    eskb_buf_write_u8(&const_buf, ESKB_CONST_INT64);
    eskb_buf_write_i64(&const_buf, 41);
    eskb_buf_write_u8(&const_buf, ESKB_CONST_INT64);
    eskb_buf_write_i64(&const_buf, 1);
    eskb_buf_write_u8(&const_buf, ESKB_CONST_STRING);
    eskb_buf_write_string(&const_buf, long_string.c_str(), long_string.size());

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_CONST, 1},
        {OP_ADD, 0},
        {OP_HALT, 0},
    };
    const Instr helper_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };
    const Instr helper_branch_code[] = {
        {OP_FALSE, 0},
        {OP_JUMP_IF_FALSE, 4},
        {OP_CONST, 0},
        {OP_HALT, 0},
        {OP_CONST, 1},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 3);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    write_function(&code_buf, "helper1", helper_code, sizeof(helper_code) / sizeof(helper_code[0]));
    write_function(&code_buf, "helper2", helper_branch_code, sizeof(helper_branch_code) / sizeof(helper_branch_code[0]));

    eskb_buf_write_leb128(&payload, 2);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);

    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    eskb_buf_write(&file, &hdr, sizeof(hdr));
    eskb_buf_write(&file, payload.data, payload.len);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&payload);
    return file;
}

void test_number_to_string_radix_case(const char* label, int64_t n, int64_t radix,
                                      const char* expected) {
    EskbBuffer chunk = make_number_to_string_radix_chunk(n, radix, expected);
    EshkolVmHandle* vm = eshkol_vm_load_chunk(chunk.data, chunk.len);

    std::string load_label = std::string(label) + ": load chunk";
    CHECK(vm != nullptr, load_label.c_str());
    if (vm) {
        std::string run_label = std::string(label) + ": run chunk";
        CHECK(eshkol_vm_run(vm) == 0, run_label.c_str());

        int64_t top = 0;
        std::string read_label = std::string(label) + ": read comparison result";
        CHECK(eshkol_vm_top_int64(vm, &top) == 0, read_label.c_str());

        std::string value_label = std::string(label) + ": expected string";
        CHECK(top == 1, value_label.c_str());
        eshkol_vm_destroy(vm);
    }

    eskb_buf_free(&chunk);
}

void test_number_to_string_radix(void) {
    test_number_to_string_radix_case("number->string base 2", 10, 2, "1010");
    test_number_to_string_radix_case("number->string negative base 16", -255, 16, "-ff");
    test_number_to_string_radix_case("number->string base 36", 35, 36, "z");
    test_number_to_string_radix_case("number->string base 36 carry", 36, 36, "10");
}

int host_add_with_offset(VM* vm) {
    int64_t b = 0;
    int64_t a = 0;
    if (eshkol_vm_host_pop_int64(vm, &b) != 0) return -1;
    if (eshkol_vm_host_pop_int64(vm, &a) != 0) return -1;
    return eshkol_vm_host_push_int64(vm, a + b + 5);
}

int host_add_double(VM* vm) {
    double b = 0.0;
    double a = 0.0;
    if (eshkol_vm_host_pop_double(vm, &b) != 0) return -1;
    if (eshkol_vm_host_pop_double(vm, &a) != 0) return -1;
    return eshkol_vm_host_push_double(vm, a + b);
}

void test_static_host_native_table(void) {
    eshkol_vm_clear_host_natives();
    CHECK(eshkol_vm_host_native_capacity() >= 2,
          "host-native table exposes usable capacity");
    CHECK(eshkol_vm_host_native_count() == 0,
          "host-native table starts empty after clear");

    const EshkolVmHostNative table[] = {
        {"test.static-add-with-offset", host_add_with_offset},
        {"test.static-add-double", host_add_double},
    };
    CHECK(eshkol_vm_install_host_natives(table, 2) == 0,
          "install static host-native table");
    CHECK(eshkol_vm_host_native_count() == 2,
          "static host-native table count is fixed");

    const EshkolVmHostNative duplicate_table[] = {
        {"test.static-duplicate", host_add_with_offset},
        {"test.static-duplicate", host_add_double},
    };
    CHECK(eshkol_vm_install_host_natives(duplicate_table, 2) == -1,
          "reject duplicate static host-native table");
    CHECK(eshkol_vm_host_native_count() == 2,
          "duplicate static table does not mutate installed table");

    CHECK(eshkol_vm_install_host_natives(table, -1) == -1,
          "reject negative static host-native table count");
    CHECK(eshkol_vm_host_native_count() == 2,
          "negative static table count does not mutate installed table");
    CHECK(eshkol_vm_install_host_natives(nullptr, 1) == -1,
          "reject null static host-native table with positive count");
    CHECK(eshkol_vm_host_native_count() == 2,
          "null static table does not mutate installed table");

    const EshkolVmHostNative bad_name_table[] = {
        {"test.static-add-with-offset", host_add_with_offset},
        {"", host_add_double},
    };
    CHECK(eshkol_vm_install_host_natives(bad_name_table, 2) == -1,
          "reject empty static host-native name");
    CHECK(eshkol_vm_host_native_count() == 2,
          "empty static name does not mutate installed table");

    const EshkolVmHostNative null_fn_table[] = {
        {"test.static-add-with-offset", host_add_with_offset},
        {"test.static-null-fn", nullptr},
    };
    CHECK(eshkol_vm_install_host_natives(null_fn_table, 2) == -1,
          "reject null static host-native callback");
    CHECK(eshkol_vm_host_native_count() == 2,
          "null static callback does not mutate installed table");

    std::vector<std::string> overflow_names;
    std::vector<EshkolVmHostNative> overflow_table;
    const int overflow_count = eshkol_vm_host_native_capacity() + 1;
    overflow_names.reserve(static_cast<size_t>(overflow_count));
    overflow_table.reserve(static_cast<size_t>(overflow_count));
    for (int i = 0; i < overflow_count; ++i) {
        overflow_names.push_back("test.static-overflow-" + std::to_string(i));
        overflow_table.push_back({overflow_names.back().c_str(), host_add_with_offset});
    }
    CHECK(eshkol_vm_install_host_natives(overflow_table.data(), overflow_count) == -1,
          "reject over-capacity static host-native table");
    CHECK(eshkol_vm_host_native_count() == 2,
          "over-capacity static table does not mutate installed table");

    CHECK(eshkol_vm_register_host_native("test.static-add-with-offset",
                                         host_add_with_offset) == -1,
          "dynamic host-native register rejects installed static duplicate");
    int dynamic_slot = eshkol_vm_register_host_native("test.dynamic-after-static",
                                                     host_add_with_offset);
    CHECK(dynamic_slot == 2, "dynamic host-native appends after static table");
    if (dynamic_slot >= 0) {
        CHECK(eshkol_vm_unregister_host_native(dynamic_slot) == 0,
              "cleanup dynamic host-native after static table");
    }

    EskbBuffer int_chunk = make_host_native_int64_chunk(ESHKOL_VM_HOST_NATIVE_BASE);
    EshkolVmHandle* int_vm = eshkol_vm_load_chunk(int_chunk.data, int_chunk.len);
    CHECK(int_vm != nullptr, "load static int64 host-native chunk");
    if (int_vm) {
        CHECK(eshkol_vm_run(int_vm) == 0, "run static int64 host-native chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(int_vm, &top) == 0,
              "read static int64 host-native result");
        CHECK(top == 42, "static int64 host-native result == 42");
        eshkol_vm_destroy(int_vm);
    }
    eskb_buf_free(&int_chunk);

    EskbBuffer double_chunk =
        make_host_native_double_chunk(ESHKOL_VM_HOST_NATIVE_BASE + 1);
    EshkolVmHandle* double_vm =
        eshkol_vm_load_chunk(double_chunk.data, double_chunk.len);
    CHECK(double_vm != nullptr, "load static double host-native chunk");
    if (double_vm) {
        CHECK(eshkol_vm_run(double_vm) == 0,
              "run static double host-native chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(double_vm, &top) == 0,
              "read static double host-native result");
        CHECK(top == 42, "static double host-native result coerces to 42");
        eshkol_vm_destroy(double_vm);
    }
    eskb_buf_free(&double_chunk);

    eshkol_vm_clear_host_natives();
    CHECK(eshkol_vm_host_native_count() == 0,
          "clear static host-native table");
    CHECK(eshkol_vm_install_host_natives(nullptr, 0) == 0,
          "install empty static host-native table");
    CHECK(eshkol_vm_host_native_count() == 0,
          "empty static host-native table remains empty");
}

void* no_op_pool_task(void* arg) {
    return arg;
}

struct CrossPoolSubmitArg {
    eshkol_thread_pool_t* target;
    eshkol_future_t* first;
    eshkol_future_t* second;
};

void* submit_to_bounded_target_pool(void* raw) {
    auto* arg = static_cast<CrossPoolSubmitArg*>(raw);
    arg->first = thread_pool_submit(arg->target, no_op_pool_task, nullptr);
    arg->second = thread_pool_submit(arg->target, no_op_pool_task, nullptr);
    return nullptr;
}

eshkol_thread_pool_t* create_paused_bounded_pool(size_t capacity, const char* name) {
    eshkol_thread_pool_config_t config = ESHKOL_THREAD_POOL_DEFAULT_CONFIG;
    config.num_threads = 2;
    config.task_queue_capacity = capacity;
    config.name = name;

    eshkol_thread_pool_t* pool = thread_pool_create(&config);
    if (pool) {
        thread_pool_pause(pool);
    }
    return pool;
}

void test_thread_pool_bounded_external_queue(void) {
    {
        eshkol_thread_pool_t* pool = create_paused_bounded_pool(
            1, "bounded-external-submit-test");
        CHECK(pool != nullptr, "thread-pool bounded external submit: create pool");
        if (pool) {
            eshkol_future_t* first = thread_pool_submit(pool, no_op_pool_task, nullptr);
            eshkol_future_t* second = thread_pool_submit(pool, no_op_pool_task, nullptr);
            CHECK(first != nullptr, "thread-pool bounded external submit: accepts capacity slot");
            CHECK(second == nullptr, "thread-pool bounded external submit: rejects full queue");
            thread_pool_destroy(pool);
            future_release(first);
            future_release(second);
        }
    }

    {
        eshkol_thread_pool_t* pool = create_paused_bounded_pool(
            2, "bounded-external-batch-test");
        CHECK(pool != nullptr, "thread-pool bounded external batch: create pool");
        if (pool) {
            eshkol_task_fn fns[4] = {
                no_op_pool_task, no_op_pool_task, no_op_pool_task, no_op_pool_task
            };
            void* args[4] = {nullptr, nullptr, nullptr, nullptr};
            eshkol_future_t* futures[4] = {nullptr, nullptr, nullptr, nullptr};

            size_t submitted = thread_pool_submit_batch(pool, fns, args, futures, 4);
            CHECK(submitted == 2, "thread-pool bounded external batch: submits to capacity");
            CHECK(futures[0] != nullptr && futures[1] != nullptr &&
                      futures[2] == nullptr && futures[3] == nullptr,
                  "thread-pool bounded external batch: leaves overflow futures empty");
            thread_pool_destroy(pool);
            for (size_t i = 0; i < submitted; ++i) {
                future_release(futures[i]);
            }
        }
    }

    {
        eshkol_thread_pool_t* pool = create_paused_bounded_pool(
            1, "bounded-external-detached-test");
        CHECK(pool != nullptr, "thread-pool bounded external detached: create pool");
        if (pool) {
            bool first = thread_pool_submit_detached(pool, no_op_pool_task, nullptr);
            bool second = thread_pool_submit_detached(pool, no_op_pool_task, nullptr);
            CHECK(first, "thread-pool bounded external detached: accepts capacity slot");
            CHECK(!second, "thread-pool bounded external detached: rejects full queue");
            thread_pool_destroy(pool);
        }
    }

    {
        eshkol_thread_pool_t* target = create_paused_bounded_pool(
            1, "bounded-cross-pool-target-test");
        CHECK(target != nullptr, "thread-pool bounded cross-pool submit: create target");

        eshkol_thread_pool_config_t source_config = ESHKOL_THREAD_POOL_DEFAULT_CONFIG;
        source_config.num_threads = 1;
        source_config.name = "bounded-cross-pool-source-test";
        eshkol_thread_pool_t* source = thread_pool_create(&source_config);
        CHECK(source != nullptr, "thread-pool bounded cross-pool submit: create source");

        if (target && source) {
            CrossPoolSubmitArg arg = {target, nullptr, nullptr};
            eshkol_future_t* submitter =
                thread_pool_submit(source, submit_to_bounded_target_pool, &arg);
            CHECK(submitter != nullptr, "thread-pool bounded cross-pool submit: submit worker task");
            CHECK(future_wait(submitter, 5000),
                  "thread-pool bounded cross-pool submit: worker task completes");
            CHECK(arg.first != nullptr,
                  "thread-pool bounded cross-pool submit: accepts target capacity slot");
            CHECK(arg.second == nullptr,
                  "thread-pool bounded cross-pool submit: rejects target overflow");
            thread_pool_destroy(source);
            future_release(submitter);
            thread_pool_destroy(target);
            future_release(arg.first);
            future_release(arg.second);
        } else {
            if (source) thread_pool_destroy(source);
            if (target) thread_pool_destroy(target);
        }
    }
}

void test_host_native_registry(void) {
    eshkol_vm_clear_host_natives();

    int slot = eshkol_vm_register_host_native("test.add-with-offset", host_add_with_offset);
    CHECK(slot >= 0, "register host native");
    CHECK(eshkol_vm_register_host_native("test.add-with-offset", host_add_with_offset) == -1,
          "reject duplicate host native");
    if (slot < 0) {
        eshkol_vm_clear_host_natives();
        return;
    }

    EskbBuffer chunk = make_host_native_int64_chunk(ESHKOL_VM_HOST_NATIVE_BASE + slot);
    EshkolVmHandle* vm = eshkol_vm_load_chunk(chunk.data, chunk.len);
    CHECK(vm != nullptr, "load host-native ESKB chunk");
    if (vm) {
        CHECK(eshkol_vm_run(vm) == 0, "run host-native ESKB chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(vm, &top) == 0, "read host-native result");
        CHECK(top == 42, "host-native result == 42");
        eshkol_vm_destroy(vm);
    }
    eskb_buf_free(&chunk);

    int double_slot = eshkol_vm_register_host_native("test.add-double", host_add_double);
    CHECK(double_slot >= 0, "register double host native");
    if (double_slot < 0) {
        eshkol_vm_clear_host_natives();
        return;
    }

    EskbBuffer double_chunk = make_host_native_double_chunk(ESHKOL_VM_HOST_NATIVE_BASE + double_slot);
    EshkolVmHandle* double_vm = eshkol_vm_load_chunk(double_chunk.data, double_chunk.len);
    CHECK(double_vm != nullptr, "load double host-native ESKB chunk");
    if (double_vm) {
        CHECK(eshkol_vm_run(double_vm) == 0, "run double host-native ESKB chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(double_vm, &top) == 0, "read double host-native result");
        CHECK(top == 42, "double host-native result coerces to 42");
        eshkol_vm_destroy(double_vm);
    }
    eskb_buf_free(&double_chunk);

    CHECK(eshkol_vm_unregister_host_native(slot) == 0, "unregister host native");
    CHECK(eshkol_vm_unregister_host_native(slot) == -1, "reject duplicate host-native unregister");

    int reused_slot = eshkol_vm_register_host_native("test.add-with-offset", host_add_with_offset);
    CHECK(reused_slot == slot, "reuse tombstoned host-native slot");
    if (reused_slot >= 0) {
        EskbBuffer reused_chunk = make_host_native_int64_chunk(ESHKOL_VM_HOST_NATIVE_BASE + reused_slot);
        EshkolVmHandle* reused_vm = eshkol_vm_load_chunk(reused_chunk.data, reused_chunk.len);
        CHECK(reused_vm != nullptr, "load reused host-native ESKB chunk");
        if (reused_vm) {
            CHECK(eshkol_vm_run(reused_vm) == 0, "run reused host-native ESKB chunk");
            int64_t top = 0;
            CHECK(eshkol_vm_top_int64(reused_vm, &top) == 0, "read reused host-native result");
            CHECK(top == 42, "reused host-native result == 42");
            eshkol_vm_destroy(reused_vm);
        }
        eskb_buf_free(&reused_chunk);
        CHECK(eshkol_vm_unregister_host_native(reused_slot) == 0, "cleanup reused host native");
    }
    CHECK(eshkol_vm_unregister_host_native(double_slot) == 0, "cleanup double host native");
    eshkol_vm_clear_host_natives();
}

void test_file_mmap_native(void) {
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::path original_cwd = fs::current_path(ec);
    CHECK(!ec, "capture current directory for mmap regression");
    if (ec) return;

    fs::path test_dir = fs::temp_directory_path(ec) / "eshkol_vm_mmap_api_test";
    CHECK(!ec, "resolve temp directory for mmap regression");
    if (ec) return;
    fs::create_directories(test_dir, ec);
    CHECK(!ec, "create temp directory for mmap regression");
    if (ec) return;
    fs::current_path(test_dir, ec);
    CHECK(!ec, "enter temp directory for mmap regression");
    if (ec) return;

    const char* path = "mmap.bin";
    FILE* f = std::fopen(path, "wb");
    CHECK(f != nullptr, "create mmap regression fixture");
    if (!f) {
        fs::current_path(original_cwd, ec);
        return;
    }
    const char bytes[] = "abcdef";
    CHECK(std::fwrite(bytes, 1, 6, f) == 6, "write mmap regression fixture");
    std::fclose(f);

    EskbBuffer byte_chunk = make_file_mmap_u8_ref_chunk(path, 1, 3, 0);
    EshkolVmHandle* byte_vm = eshkol_vm_load_chunk(byte_chunk.data, byte_chunk.len);
    CHECK(byte_vm != nullptr, "load file-mmap byte chunk");
    if (byte_vm) {
        CHECK(eshkol_vm_run(byte_vm) == 0, "run file-mmap byte chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(byte_vm, &top) == 0, "read file-mmap byte result");
        CHECK(top == 98, "file-mmap byte slice starts at 'b'");
        eshkol_vm_destroy(byte_vm);
    }
    eskb_buf_free(&byte_chunk);

    EskbBuffer empty_chunk = make_file_mmap_length_chunk(path, 6, 0);
    EshkolVmHandle* empty_vm = eshkol_vm_load_chunk(empty_chunk.data, empty_chunk.len);
    CHECK(empty_vm != nullptr, "load file-mmap empty chunk");
    if (empty_vm) {
        CHECK(eshkol_vm_run(empty_vm) == 0, "run file-mmap empty chunk");
        int64_t top = -1;
        CHECK(eshkol_vm_top_int64(empty_vm, &top) == 0, "read file-mmap empty length");
        CHECK(top == 0, "file-mmap empty EOF slice length is zero");
        eshkol_vm_destroy(empty_vm);
    }
    eskb_buf_free(&empty_chunk);

    CHECK(std::remove(path) == 0, "remove mmap regression fixture");
    fs::current_path(original_cwd, ec);
    CHECK(!ec, "restore current directory after mmap regression");
    fs::remove(test_dir, ec);
}

void test_native_string_case(const char* label, int native_fid, const char* a, const char* b,
                             const char* expected) {
    EskbBuffer chunk = make_native_string_compare_chunk(native_fid, a, b, expected);
    EshkolVmHandle* vm = eshkol_vm_load_chunk(chunk.data, chunk.len);

    std::string load_label = std::string(label) + ": load chunk";
    CHECK(vm != nullptr, load_label.c_str());
    if (vm) {
        std::string run_label = std::string(label) + ": run chunk";
        CHECK(eshkol_vm_run(vm) == 0, run_label.c_str());

        int64_t top = 0;
        std::string read_label = std::string(label) + ": read comparison result";
        CHECK(eshkol_vm_top_int64(vm, &top) == 0, read_label.c_str());

        std::string value_label = std::string(label) + ": expected string";
        CHECK(top == 1, value_label.c_str());
        eshkol_vm_destroy(vm);
    }

    eskb_buf_free(&chunk);
}

void test_path_native_helpers(void) {
#ifdef _WIN32
    const char* relative_expected = "..\\c";
    const char* resolve_expected = "a\\c";
#else
    const char* relative_expected = "../c";
    const char* resolve_expected = "a/c";
#endif
    test_native_string_case("path-relative", 1727, "a/b", "a/c", relative_expected);
    test_native_string_case("path-resolve", 1728, "a/b", "../c", resolve_expected);
}

void test_valid_chunk(void) {
    EskbBuffer chunk = make_test_chunk();
    EshkolVmHandle* vm = eshkol_vm_load_chunk(chunk.data, chunk.len);
    CHECK(vm != nullptr, "load in-memory ESKB chunk");
    if (vm) {
        CHECK(eshkol_vm_has_function(vm, "main") == 1,
              "loaded chunk exposes main function");
        CHECK(eshkol_vm_has_function(vm, "helper2") == 1,
              "loaded chunk exposes helper function");
        CHECK(eshkol_vm_has_function(vm, "missing") == 0,
              "loaded chunk rejects missing function lookup");
        CHECK(eshkol_vm_has_function(vm, nullptr) == -1,
              "function lookup rejects null name");

        CHECK(eshkol_vm_run(vm) == 0, "run loaded chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(vm, &top) == 0, "read top-of-stack int64");
        CHECK(top == 42, "top-of-stack value == 42");

        CHECK(eshkol_vm_call(vm, "helper2") == 0, "run named helper entry");
        CHECK(eshkol_vm_top_int64(vm, &top) == 0, "read named helper result");
        CHECK(top == 1, "named helper branch result == 1");

        CHECK(eshkol_vm_call(vm, "missing") == -1, "reject missing named entry");
        CHECK(eshkol_vm_call(vm, "main") == 0, "rerun named main entry");
        CHECK(eshkol_vm_top_int64(vm, &top) == 0, "read rerun main result");
        CHECK(top == 42, "rerun main result == 42");
        eshkol_vm_destroy(vm);
    }
    eskb_buf_free(&chunk);
}

void test_bad_inputs(void) {
    CHECK(eshkol_vm_load_chunk(nullptr, 0) == nullptr, "reject null chunk");

    EskbBuffer chunk = make_test_chunk();
    if (chunk.len > sizeof(EskbHeader)) {
        chunk.data[chunk.len - 1] ^= 0x01;
    }
    CHECK(eshkol_vm_load_chunk(chunk.data, chunk.len) == nullptr,
          "reject checksum-corrupted chunk");
    eskb_buf_free(&chunk);
}

}  // namespace

int main(void) {
    std::printf("=== VM C API tests ===\n");
    test_valid_chunk();
    test_number_to_string_radix();
    test_static_host_native_table();
    test_host_native_registry();
    test_thread_pool_bounded_external_queue();
    test_file_mmap_native();
    test_path_native_helpers();
    test_bad_inputs();
    std::printf("\nResults: %d/%d checks passed\n", g_total - g_failures, g_total);
    return g_failures == 0 ? 0 : 1;
}
