/* test_vm_c_api.cpp
 *
 * Smoke tests for the public bytecode VM C ABI used by embedders that already
 * have an ESKB chunk in memory.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <eshkol/backend/thread_pool.h>
#include <eshkol/backend/vm.h>

extern "C" {
#include "eskb_format.h"
}

namespace {

enum : uint8_t {
    OP_NOP = 0,
    OP_CONST = 1,
    OP_FALSE = 4,
    OP_ADD = 7,
    OP_JUMP = 28,
    OP_JUMP_IF_FALSE = 29,
    OP_HALT = 36,
    OP_NATIVE_CALL = 37,
    OP_STR_LEN = 44,
    OP_INVALID = 255,
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

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
}

void write_instr(EskbBuffer* b, const Instr& instr) {
    eskb_buf_write_u8(b, instr.op);
    int64_t operand = instr.operand;
    uint64_t zigzag = (static_cast<uint64_t>(operand) << 1) ^
                      static_cast<uint64_t>(operand >> 63);
    eskb_buf_write_leb128(b, zigzag);
}

void write_function_with_metadata(EskbBuffer* b, const char* name,
                                  uint8_t n_params, uint64_t n_locals,
                                  uint8_t n_upvalues, const Instr* code,
                                  size_t n_code) {
    eskb_buf_write_string(b, name, std::strlen(name));
    eskb_buf_write_u8(b, n_params);
    eskb_buf_write_leb128(b, n_locals);
    eskb_buf_write_u8(b, n_upvalues);
    eskb_buf_write_leb128(b, n_code);
    for (size_t i = 0; i < n_code; ++i) {
        write_instr(b, code[i]);
    }
}

void write_function(EskbBuffer* b, const char* name, const Instr* code, size_t n_code) {
    write_function_with_metadata(b, name, 0, 0, 0, code, n_code);
}

void write_function_with_locals(EskbBuffer* b, const char* name, uint64_t n_locals,
                                const Instr* code, size_t n_code) {
    write_function_with_metadata(b, name, 0, n_locals, 0, code, n_code);
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

EskbBuffer make_function_name_validation_chunk(const char* first_name,
                                               const char* second_name) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 2);
    write_int64_const(&const_buf, 1);
    write_int64_const(&const_buf, 2);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };
    const Instr first_tick_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };
    const Instr second_tick_code[] = {
        {OP_CONST, 1},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 3);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    write_function(&code_buf, first_name, first_tick_code,
                   sizeof(first_tick_code) / sizeof(first_tick_code[0]));
    write_function(&code_buf, second_name, second_tick_code,
                   sizeof(second_tick_code) / sizeof(second_tick_code[0]));

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

EskbBuffer make_duplicate_function_name_chunk(void) {
    return make_function_name_validation_chunk("tick", "tick");
}

EskbBuffer make_empty_function_name_chunk(void) {
    return make_function_name_validation_chunk("", "tick");
}

EskbBuffer make_trailing_code_section_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 7);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    eskb_buf_write_u8(&code_buf, 0x7f);

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

EskbBuffer make_trailing_payload_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 11);

    const Instr main_code[] = {
        {OP_CONST, 0},
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
    eskb_buf_write_u8(&payload, 0x7f);

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

EskbBuffer make_uncalled_desktop_native_helper_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 42);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };
    const Instr helper_code[] = {
        {OP_NATIVE_CALL, 100},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 2);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    write_function(&code_buf, "helper", helper_code, sizeof(helper_code) / sizeof(helper_code[0]));

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

EskbBuffer make_cross_function_branch_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 1);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };
    const Instr jumping_helper_code[] = {
        {OP_JUMP, 2},
        {OP_HALT, 0},
    };
    const Instr target_helper_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 3);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    write_function(&code_buf, "jumping-helper", jumping_helper_code,
                   sizeof(jumping_helper_code) / sizeof(jumping_helper_code[0]));
    write_function(&code_buf, "target-helper", target_helper_code,
                   sizeof(target_helper_code) / sizeof(target_helper_code[0]));

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

EskbBuffer make_metadata_test_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 7);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };
    const Instr captured_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 2);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    write_function_with_metadata(&code_buf, "captured", 1, 1, 1, captured_code,
                                 sizeof(captured_code) / sizeof(captured_code[0]));

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

EskbBuffer make_parameterized_main_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 9);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function_with_metadata(&code_buf, "main", 1, 0, 0, main_code,
                                 sizeof(main_code) / sizeof(main_code[0]));

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

EskbBuffer make_upvalue_entry_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 11);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };
    const Instr upvalue_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 2);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    write_function_with_metadata(&code_buf, "closed-over", 0, 0, 1, upvalue_code,
                                 sizeof(upvalue_code) / sizeof(upvalue_code[0]));

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

EskbBuffer make_upvalue_main_chunk(void) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    write_int64_const(&const_buf, 13);

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_HALT, 0},
    };

    eskb_buf_write_leb128(&code_buf, 1);
    write_function_with_metadata(&code_buf, "main", 0, 0, 1, main_code,
                                 sizeof(main_code) / sizeof(main_code[0]));

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

EskbBuffer make_profile_validation_chunk(size_t n_constants,
                                         uint64_t n_locals,
                                         const std::vector<Instr>& code) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, n_constants);
    for (size_t i = 0; i < n_constants; ++i) {
        write_int64_const(&const_buf, static_cast<int64_t>(i));
    }

    eskb_buf_write_leb128(&code_buf, 1);
    write_function_with_locals(&code_buf, "main", n_locals, code.data(), code.size());

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

EskbBuffer make_string_constant_len_chunk(const char* text) {
    EskbBuffer const_buf;
    EskbBuffer code_buf;
    EskbBuffer payload;
    EskbBuffer file;
    eskb_buf_init(&const_buf);
    eskb_buf_init(&code_buf);
    eskb_buf_init(&payload);
    eskb_buf_init(&file);

    eskb_buf_write_leb128(&const_buf, 1);
    eskb_buf_write_u8(&const_buf, ESKB_CONST_STRING);
    eskb_buf_write_string(&const_buf, text, std::strlen(text));

    const Instr main_code[] = {
        {OP_CONST, 0},
        {OP_STR_LEN, 0},
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

void test_host_only_native_policy(void) {
    CHECK(eshkol_vm_set_native_policy(nullptr, ESHKOL_VM_NATIVE_POLICY_HOST_ONLY) == -1,
          "native policy setter rejects null VM");
    CHECK(eshkol_vm_get_native_policy(nullptr) == -1,
          "native policy getter rejects null VM");
    CHECK(eshkol_vm_default_load_options(nullptr) == -1,
          "default load options reject null output");

    eshkol_vm_clear_host_natives();
    const EshkolVmHostNative table[] = {
        {"test.host-only-add", host_add_with_offset},
    };
    CHECK(eshkol_vm_install_host_natives(table, 1) == 0,
          "install host-only native table");

    EskbBuffer host_chunk = make_host_native_int64_chunk(ESHKOL_VM_HOST_NATIVE_BASE);
    EshkolVmLoadOptions host_only_options{};
    CHECK(eshkol_vm_default_load_options(&host_only_options) == 0,
          "initialize default VM load options");
    CHECK(host_only_options.reject_desktop_native_calls == 0,
          "default VM load options allow desktop native calls");
    host_only_options.native_policy = ESHKOL_VM_NATIVE_POLICY_HOST_ONLY;
    host_only_options.reject_desktop_native_calls = 1;
    EshkolVmHandle* host_vm =
        eshkol_vm_load_chunk_with_options(host_chunk.data, host_chunk.len,
                                          &host_only_options);
    CHECK(host_vm != nullptr, "load host-only host-native chunk");
    if (host_vm) {
        CHECK(eshkol_vm_get_native_policy(host_vm) == ESHKOL_VM_NATIVE_POLICY_HOST_ONLY,
              "load options set host-native-only policy");
        CHECK(eshkol_vm_set_native_policy(host_vm, 9999) == -1,
              "native policy setter rejects invalid policy");
        CHECK(eshkol_vm_set_native_policy(host_vm, ESHKOL_VM_NATIVE_POLICY_DESKTOP) == 0,
              "native policy setter can switch back to desktop");
        CHECK(eshkol_vm_set_native_policy(host_vm, ESHKOL_VM_NATIVE_POLICY_HOST_ONLY) == 0,
              "native policy setter can restore host-native-only policy");
        CHECK(eshkol_vm_run(host_vm) == 0,
              "host-native-only policy permits fixed host-native slot");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(host_vm, &top) == 0,
              "read host-native-only result");
        CHECK(top == 42, "host-native-only result == 42");
        eshkol_vm_destroy(host_vm);
    }
    eskb_buf_free(&host_chunk);

    host_only_options.native_policy = 9999;
    EskbBuffer invalid_options_chunk =
        make_host_native_int64_chunk(ESHKOL_VM_HOST_NATIVE_BASE);
    CHECK(eshkol_vm_load_chunk_with_options(invalid_options_chunk.data,
                                            invalid_options_chunk.len,
                                            &host_only_options) == nullptr,
          "load options reject invalid native policy");
    eskb_buf_free(&invalid_options_chunk);

    host_only_options.native_policy = ESHKOL_VM_NATIVE_POLICY_HOST_ONLY;
    host_only_options.reject_desktop_native_calls = 1;
    EskbBuffer rejected_desktop_chunk =
        make_number_to_string_radix_chunk(10, 2, "1010");
    CHECK(eshkol_vm_load_chunk_with_options(rejected_desktop_chunk.data,
                                            rejected_desktop_chunk.len,
                                            &host_only_options) == nullptr,
          "embedded load options reject desktop native fid before run");
    eskb_buf_free(&rejected_desktop_chunk);

    EskbBuffer rejected_helper_chunk =
        make_uncalled_desktop_native_helper_chunk();
    CHECK(eshkol_vm_load_chunk_with_options(rejected_helper_chunk.data,
                                            rejected_helper_chunk.len,
                                            &host_only_options) == nullptr,
          "embedded load options reject desktop native fid in helper function");
    eskb_buf_free(&rejected_helper_chunk);

    EskbBuffer desktop_chunk =
        make_number_to_string_radix_chunk(10, 2, "1010");
    EshkolVmHandle* desktop_vm =
        eshkol_vm_load_chunk(desktop_chunk.data, desktop_chunk.len);
    CHECK(desktop_vm != nullptr, "load desktop-native policy regression chunk");
    if (desktop_vm) {
        CHECK(eshkol_vm_set_native_policy(desktop_vm, ESHKOL_VM_NATIVE_POLICY_HOST_ONLY) == 0,
              "switch desktop-native chunk to host-native-only policy");
        CHECK(eshkol_vm_run(desktop_vm) == -1,
              "host-native-only policy rejects desktop native fid");
        eshkol_vm_destroy(desktop_vm);
    }
    eskb_buf_free(&desktop_chunk);
    eshkol_vm_clear_host_natives();
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
    EshkolVmLoadOptions required_options{};
    CHECK(eshkol_vm_default_load_options(&required_options) == 0,
          "initialize required-entry load options");
    CHECK(required_options.required_functions == nullptr &&
          required_options.required_function_count == 0,
          "default VM load options do not require named entries");

    const char* required_entries[] = {"main", "helper2"};
    required_options.required_functions = required_entries;
    required_options.required_function_count = 2;
    EshkolVmHandle* required_vm =
        eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                          &required_options);
    CHECK(required_vm != nullptr, "load chunk with required named entries");
    if (required_vm) eshkol_vm_destroy(required_vm);

    const char* missing_entries[] = {"main", "tick"};
    required_options.required_functions = missing_entries;
    required_options.required_function_count = 2;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &required_options) == nullptr,
          "load options reject missing required named entry");

    const char* invalid_entries[] = {"main", nullptr};
    required_options.required_functions = invalid_entries;
    required_options.required_function_count = 2;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &required_options) == nullptr,
          "load options reject null required named entry");

    required_options.required_functions = nullptr;
    required_options.required_function_count = 1;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &required_options) == nullptr,
          "load options reject required-entry count without names");

    required_options.required_function_count = -1;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &required_options) == nullptr,
          "load options reject negative required-entry count");

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
        CHECK(eshkol_vm_function_count(nullptr) == -1,
              "function count rejects null handle");
        CHECK(eshkol_vm_function_name(nullptr, 0) == nullptr,
              "function name lookup rejects null handle");
        CHECK(eshkol_vm_function_count(vm) == 3,
              "loaded chunk reports function table size");
        const char* fn0 = eshkol_vm_function_name(vm, 0);
        const char* fn1 = eshkol_vm_function_name(vm, 1);
        const char* fn2 = eshkol_vm_function_name(vm, 2);
        CHECK(fn0 && std::strcmp(fn0, "main") == 0,
              "function table exposes main name");
        CHECK(fn1 && std::strcmp(fn1, "helper1") == 0,
              "function table exposes first helper name");
        CHECK(fn2 && std::strcmp(fn2, "helper2") == 0,
              "function table exposes second helper name");
        CHECK(eshkol_vm_function_name(vm, -1) == nullptr,
              "function name lookup rejects negative index");
        CHECK(eshkol_vm_function_name(vm, 3) == nullptr,
              "function name lookup rejects past-end index");
        EshkolVmFunctionInfo fn_info{};
        CHECK(eshkol_vm_function_info(nullptr, 0, &fn_info) == -1,
              "function info rejects null handle");
        CHECK(eshkol_vm_function_info(vm, -1, &fn_info) == -1,
              "function info rejects negative index");
        CHECK(eshkol_vm_function_info(vm, 3, &fn_info) == -1,
              "function info rejects past-end index");
        CHECK(eshkol_vm_function_info(vm, 0, nullptr) == -1,
              "function info rejects null output");
        CHECK(eshkol_vm_function_info(vm, 0, &fn_info) == 0,
              "function info exposes main metadata");
        CHECK(fn_info.name && std::strcmp(fn_info.name, "main") == 0,
              "function info exposes main name");
        CHECK(fn_info.n_params == 0, "main function has zero params");
        CHECK(fn_info.n_upvalues == 0, "main function has zero upvalues");
        CHECK(fn_info.code_offset == 0, "main function starts at code offset zero");
        CHECK(fn_info.code_len > 0, "main function reports code length");
        CHECK(eshkol_vm_function_info(vm, 2, &fn_info) == 0,
              "function info exposes helper metadata");
        CHECK(fn_info.name && std::strcmp(fn_info.name, "helper2") == 0,
              "function info exposes helper name");
        CHECK(fn_info.n_params == 0, "helper function has zero params");
        CHECK(fn_info.n_upvalues == 0, "helper function has zero upvalues");
        CHECK(fn_info.code_offset > 0, "helper function starts after main");
        CHECK(fn_info.code_len > 0, "helper function reports code length");

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

void test_required_function_metadata_options(void) {
    EskbBuffer chunk = make_metadata_test_chunk();

    EshkolVmLoadOptions options{};
    CHECK(eshkol_vm_default_load_options(&options) == 0,
          "initialize required metadata load options");

    const EshkolVmFunctionRequirement valid_requirements[] = {
        {"main", 0, 0, 2, 1},
        {"captured", 1, 1, 2, 0},
    };
    options.required_function_metadata = valid_requirements;
    options.required_function_metadata_count = 2;
    EshkolVmHandle* valid_vm =
        eshkol_vm_load_chunk_with_options(chunk.data, chunk.len, &options);
    CHECK(valid_vm != nullptr,
          "load options accept matching function metadata requirements");
    if (valid_vm) eshkol_vm_destroy(valid_vm);

    const EshkolVmFunctionRequirement wildcard_requirements[] = {
        {"captured", -1, -1, -1, 0},
    };
    options.required_function_metadata = wildcard_requirements;
    options.required_function_metadata_count = 1;
    EshkolVmHandle* wildcard_vm =
        eshkol_vm_load_chunk_with_options(chunk.data, chunk.len, &options);
    CHECK(wildcard_vm != nullptr,
          "load options accept wildcard function metadata requirements");
    if (wildcard_vm) eshkol_vm_destroy(wildcard_vm);

    const EshkolVmFunctionRequirement arity_mismatch[] = {
        {"captured", 0, -1, -1, 0},
    };
    options.required_function_metadata = arity_mismatch;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject function arity mismatch");

    const EshkolVmFunctionRequirement closure_mismatch[] = {
        {"captured", 1, -1, -1, 1},
    };
    options.required_function_metadata = closure_mismatch;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject required closed entry with upvalues");

    const EshkolVmFunctionRequirement locals_mismatch[] = {
        {"captured", -1, 0, -1, 0},
    };
    options.required_function_metadata = locals_mismatch;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject function local budget overflow");

    const EshkolVmFunctionRequirement code_len_mismatch[] = {
        {"captured", -1, -1, 1, 0},
    };
    options.required_function_metadata = code_len_mismatch;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject function code-length budget overflow");

    const EshkolVmFunctionRequirement missing_function[] = {
        {"render", -1, -1, -1, 0},
    };
    options.required_function_metadata = missing_function;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject missing metadata-required function");

    const EshkolVmFunctionRequirement empty_name[] = {
        {"", -1, -1, -1, 0},
    };
    options.required_function_metadata = empty_name;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject empty metadata-required function name");

    const EshkolVmFunctionRequirement invalid_requirement[] = {
        {"captured", -2, -1, -1, 0},
    };
    options.required_function_metadata = invalid_requirement;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject invalid metadata requirement fields");

    const EshkolVmFunctionRequirement invalid_closedness[] = {
        {"captured", -1, -1, -1, 2},
    };
    options.required_function_metadata = invalid_closedness;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject invalid metadata closedness field");

    options.required_function_metadata = nullptr;
    options.required_function_metadata_count = 1;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject metadata count without requirements");

    options.required_function_metadata_count = -1;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "load options reject negative metadata requirement count");

    eskb_buf_free(&chunk);
}

void test_duplicate_function_names_rejected(void) {
    EskbBuffer chunk = make_duplicate_function_name_chunk();
    CHECK(eshkol_vm_load_chunk(chunk.data, chunk.len) == nullptr,
          "load rejects duplicate function names");

    EshkolVmLoadOptions options{};
    CHECK(eshkol_vm_default_load_options(&options) == 0,
          "initialize duplicate-name admission options");
    const char* required_entries[] = {"tick"};
    options.required_functions = required_entries;
    options.required_function_count = 1;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &options) == nullptr,
          "entry admission rejects duplicate function names before lookup");
    eskb_buf_free(&chunk);

    EskbBuffer empty_name = make_empty_function_name_chunk();
    CHECK(eshkol_vm_load_chunk(empty_name.data, empty_name.len) == nullptr,
          "load rejects empty function name");
    eskb_buf_free(&empty_name);
}

void test_zero_arg_entry_dispatch(void) {
    EskbBuffer chunk = make_metadata_test_chunk();
    EshkolVmHandle* vm = eshkol_vm_load_chunk(chunk.data, chunk.len);
    CHECK(vm != nullptr, "load metadata chunk for zero-arg dispatch checks");
    if (vm) {
        CHECK(eshkol_vm_call(vm, "main") == 0,
              "zero-arg call API accepts zero-parameter entry");
        CHECK(eshkol_vm_call(vm, "captured") == -1,
              "zero-arg call API rejects parameterized named entry");
        eshkol_vm_destroy(vm);
    }
    eskb_buf_free(&chunk);

    EskbBuffer upvalue_entry = make_upvalue_entry_chunk();
    EshkolVmHandle* upvalue_vm =
        eshkol_vm_load_chunk(upvalue_entry.data, upvalue_entry.len);
    CHECK(upvalue_vm != nullptr, "load upvalue named entry chunk");
    if (upvalue_vm) {
        CHECK(eshkol_vm_call(upvalue_vm, "closed-over") == -1,
              "zero-arg call API rejects named entry requiring upvalues");
        eshkol_vm_destroy(upvalue_vm);
    }
    eskb_buf_free(&upvalue_entry);

    EskbBuffer parameterized_main = make_parameterized_main_chunk();
    EshkolVmHandle* parameterized_vm =
        eshkol_vm_load_chunk(parameterized_main.data, parameterized_main.len);
    CHECK(parameterized_vm != nullptr, "load parameterized main entry chunk");
    if (parameterized_vm) {
        CHECK(eshkol_vm_run(parameterized_vm) == -1,
              "zero-arg run API rejects parameterized main entry");
        eshkol_vm_destroy(parameterized_vm);
    }
    eskb_buf_free(&parameterized_main);

    EskbBuffer upvalue_main = make_upvalue_main_chunk();
    EshkolVmHandle* upvalue_main_vm =
        eshkol_vm_load_chunk(upvalue_main.data, upvalue_main.len);
    CHECK(upvalue_main_vm != nullptr, "load upvalue main entry chunk");
    if (upvalue_main_vm) {
        CHECK(eshkol_vm_run(upvalue_main_vm) == -1,
              "zero-arg run API rejects main entry requiring upvalues");
        eshkol_vm_destroy(upvalue_main_vm);
    }
    eskb_buf_free(&upvalue_main);
}

void test_profile_limits(void) {
    EshkolVmProfileLimits limits{};
    CHECK(eshkol_vm_get_profile_limits(&limits) == 0,
          "query compiled VM profile limits");
    CHECK(limits.heap_objects == ESHKOL_VM_HEAP_SIZE,
          "profile exposes compiled heap limit");
    CHECK(limits.stack_slots == ESHKOL_VM_STACK_SIZE,
          "profile exposes compiled stack limit");
    CHECK(limits.max_frames == ESHKOL_VM_MAX_FRAMES,
          "profile exposes compiled frame limit");
    CHECK(limits.max_constants == ESHKOL_VM_MAX_CONSTS,
          "profile exposes compiled constant limit");
    CHECK(limits.max_instructions == ESHKOL_VM_MAX_CODE,
          "profile exposes compiled instruction limit");
    CHECK(eshkol_vm_get_profile_limits(nullptr) == -1,
          "profile limit query rejects null output");

    const std::vector<Instr> halt_code = {{OP_HALT, 0}};
    EskbBuffer too_many_consts =
        make_profile_validation_chunk(static_cast<size_t>(limits.max_constants) + 1,
                                      0,
                                      halt_code);
    CHECK(eshkol_vm_load_chunk(too_many_consts.data, too_many_consts.len) == nullptr,
          "reject ESKB chunk over constant-pool profile limit");
    eskb_buf_free(&too_many_consts);

    const std::vector<Instr> invalid_opcode_code = {{OP_INVALID, 0}};
    EskbBuffer invalid_opcode =
        make_profile_validation_chunk(0, 0, invalid_opcode_code);
    CHECK(eshkol_vm_load_chunk(invalid_opcode.data, invalid_opcode.len) == nullptr,
          "reject ESKB chunk with invalid opcode before dispatch");
    eskb_buf_free(&invalid_opcode);

    std::vector<Instr> too_much_code(static_cast<size_t>(limits.max_instructions) + 1,
                                     {OP_NOP, 0});
    EskbBuffer too_many_instructions =
        make_profile_validation_chunk(0, 0, too_much_code);
    CHECK(eshkol_vm_load_chunk(too_many_instructions.data,
                               too_many_instructions.len) == nullptr,
          "reject ESKB chunk over instruction profile limit");
    eskb_buf_free(&too_many_instructions);

    EskbBuffer too_many_locals =
        make_profile_validation_chunk(0,
                                      static_cast<uint64_t>(limits.stack_slots) + 1,
                                      halt_code);
    CHECK(eshkol_vm_load_chunk(too_many_locals.data, too_many_locals.len) == nullptr,
          "reject ESKB function over stack/local profile limit");
    eskb_buf_free(&too_many_locals);
}

void test_string_constant_materialization(void) {
    EskbBuffer chunk = make_string_constant_len_chunk("pet-script");
    EshkolVmHandle* vm = eshkol_vm_load_chunk(chunk.data, chunk.len);
    CHECK(vm != nullptr, "load ESKB string constant chunk");
    if (vm) {
        CHECK(eshkol_vm_run(vm) == 0,
              "run ESKB string constant chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(vm, &top) == 0,
              "read ESKB string length result");
        CHECK(top == 10,
              "ESKB string constant materializes as VM string");
        eshkol_vm_destroy(vm);
    }

    EshkolVmLoadOptions reject_strings{};
    CHECK(eshkol_vm_default_load_options(&reject_strings) == 0,
          "initialize string-rejecting VM load options");
    reject_strings.reject_string_constants = 1;
    CHECK(eshkol_vm_load_chunk_with_options(chunk.data, chunk.len,
                                            &reject_strings) == nullptr,
          "embedded load options reject ESKB string constants");
    eskb_buf_free(&chunk);
}

void test_embedded_eskb_emission_load_policy(void) {
    namespace fs = std::filesystem;

    std::error_code ec;
    const fs::path test_dir = fs::temp_directory_path(ec) / "eshkol_vm_embedded_emit_test";
    CHECK(!ec, "resolve temp directory for embedded ESKB emission");
    if (ec) return;
    fs::remove_all(test_dir, ec);
    fs::create_directories(test_dir, ec);
    CHECK(!ec, "create temp directory for embedded ESKB emission");
    if (ec) return;

    const fs::path ok_path = test_dir / "ok.eskb";
    CHECK(eshkol_emit_eskb_embedded("(define (tick) (+ 19 23))\n(tick)\n",
                                    ok_path.string().c_str()) == 0,
          "embedded ESKB emitter accepts opcode-only source");
    const std::vector<uint8_t> ok_bytes = read_binary_file(ok_path);
    CHECK(!ok_bytes.empty(), "read embedded ESKB output bytes");

    EshkolVmLoadOptions embedded_options{};
    CHECK(eshkol_vm_default_load_options(&embedded_options) == 0,
          "initialize embedded load options for emitted ESKB");
    embedded_options.native_policy = ESHKOL_VM_NATIVE_POLICY_HOST_ONLY;
    embedded_options.reject_string_constants = 1;
    embedded_options.reject_desktop_native_calls = 1;
    const char* required_entries[] = {"main", "tick"};
    embedded_options.required_functions = required_entries;
    embedded_options.required_function_count = 2;
    EshkolVmHandle* embedded_vm = ok_bytes.empty()
        ? nullptr
        : eshkol_vm_load_chunk_with_options(ok_bytes.data(), ok_bytes.size(),
                                            &embedded_options);
    CHECK(embedded_vm != nullptr,
          "embedded loader accepts compiler-emitted embedded ESKB");
    if (embedded_vm) {
        CHECK(eshkol_vm_has_function(embedded_vm, "tick") == 1,
              "embedded ESKB exposes emitted tick entry");
        CHECK(eshkol_vm_call(embedded_vm, "tick") == 0,
              "embedded ESKB can call emitted tick entry");
        int64_t tick_result = 0;
        CHECK(eshkol_vm_top_int64(embedded_vm, &tick_result) == 0,
              "read emitted tick entry result");
        CHECK(tick_result == 42,
              "emitted tick entry returns expected result");
        eshkol_vm_destroy(embedded_vm);
    }

    const fs::path rejected_path = test_dir / "rejected.eskb";
    CHECK(eshkol_emit_eskb_embedded("(display \"dynamic string\")\n",
                                    rejected_path.string().c_str()) != 0,
          "embedded ESKB emitter rejects desktop-native string construction");
    CHECK(!fs::exists(rejected_path),
          "embedded ESKB emitter does not write rejected output");

    fs::remove_all(test_dir, ec);
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

    EskbBuffer trailing_code_chunk = make_trailing_code_section_chunk();
    CHECK(eshkol_vm_load_chunk(trailing_code_chunk.data,
                               trailing_code_chunk.len) == nullptr,
          "reject CODE section with trailing bytes");
    eskb_buf_free(&trailing_code_chunk);

    EskbBuffer trailing_payload_chunk = make_trailing_payload_chunk();
    CHECK(eshkol_vm_load_chunk(trailing_payload_chunk.data,
                               trailing_payload_chunk.len) == nullptr,
          "reject ESKB payload trailing bytes outside sections");
    eskb_buf_free(&trailing_payload_chunk);

    EskbBuffer cross_branch_chunk = make_cross_function_branch_chunk();
    CHECK(eshkol_vm_load_chunk(cross_branch_chunk.data,
                               cross_branch_chunk.len) == nullptr,
          "reject function-local branch target outside function body");
    eskb_buf_free(&cross_branch_chunk);
}

}  // namespace

int main(void) {
    std::printf("=== VM C API tests ===\n");
    test_valid_chunk();
    test_required_function_metadata_options();
    test_duplicate_function_names_rejected();
    test_zero_arg_entry_dispatch();
    test_profile_limits();
    test_string_constant_materialization();
    test_embedded_eskb_emission_load_policy();
    test_number_to_string_radix();
    test_static_host_native_table();
    test_host_only_native_policy();
    test_host_native_registry();
    test_thread_pool_bounded_external_queue();
    test_file_mmap_native();
    test_path_native_helpers();
    test_bad_inputs();
    std::printf("\nResults: %d/%d checks passed\n", g_total - g_failures, g_total);
    return g_failures == 0 ? 0 : 1;
}
