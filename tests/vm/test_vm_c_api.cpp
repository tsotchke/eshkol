/* test_vm_c_api.cpp
 *
 * Smoke tests for the public bytecode VM C ABI used by embedders that already
 * have an ESKB chunk in memory.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <eshkol/backend/vm.h>

extern "C" {
#include "eskb_format.h"
}

namespace {

enum : uint8_t {
    OP_CONST = 1,
    OP_ADD = 7,
    OP_HALT = 36,
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

    eskb_buf_write_leb128(&code_buf, 3);
    write_function(&code_buf, "main", main_code, sizeof(main_code) / sizeof(main_code[0]));
    write_function(&code_buf, "helper1", helper_code, sizeof(helper_code) / sizeof(helper_code[0]));
    write_function(&code_buf, "helper2", helper_code, sizeof(helper_code) / sizeof(helper_code[0]));

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

void test_valid_chunk(void) {
    EskbBuffer chunk = make_test_chunk();
    EshkolVmHandle* vm = eshkol_vm_load_chunk(chunk.data, chunk.len);
    CHECK(vm != nullptr, "load in-memory ESKB chunk");
    if (vm) {
        CHECK(eshkol_vm_run(vm) == 0, "run loaded chunk");
        int64_t top = 0;
        CHECK(eshkol_vm_top_int64(vm, &top) == 0, "read top-of-stack int64");
        CHECK(top == 42, "top-of-stack value == 42");
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
    test_bad_inputs();
    std::printf("\nResults: %d/%d checks passed\n", g_total - g_failures, g_total);
    return g_failures == 0 ? 0 : 1;
}
