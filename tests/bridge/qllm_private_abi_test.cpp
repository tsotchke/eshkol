#include "eshkol/bridge/qllm_bridge.h"
#include "../../lib/backend/eskb_format.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)
int main(int argc, char**) {
    CHECK(!eshkol_qllm_bridge_ready());
    CHECK(!eshkol_qllm_bridge_init("/does/not/exist"));
    const double source[] = {-1.5, 0.25, 3.75, 2.0, -2.5, 0.0};
    const size_t shape[] = {2, 3};
#if !defined(ESHKOL_HAS_QLLM)
    CHECK(!eshkol_qllm_bridge_available());
    CHECK(!eshkol_qllm_bridge_init(nullptr));
    CHECK(!eshkol_to_qllm_tensor(source, shape, 2));
    double out = 77; size_t count = 1;
    CHECK(!qllm_to_eshkol_tensor(nullptr, &out, &count));
    CHECK(out == 77 && count == 0);
    eshkol_qllm_tensor_destroy(nullptr);
#else
    CHECK(eshkol_qllm_bridge_available());
    if (argc > 1) {
        CHECK(eshkol_qllm_bridge_init(nullptr));
        CHECK(eshkol_qllm_bridge_ready());
        CHECK(eshkol_qllm_bridge_init(nullptr));
        int64_t handle = 0;
        CHECK(qllm_eshkol_eval_to_int64("(qllm-tensor-create-zeros 2 3)", &handle) == QLLM_SUCCESS);
        CHECK(handle != 0);
        char expr[256]; int64_t result = 0;
        std::snprintf(expr, sizeof expr, "(qllm-tensor-shape %lld)", (long long)handle);
        CHECK(qllm_eshkol_eval_to_int64(expr, &result) == QLLM_SUCCESS && result == 2000003);
        std::snprintf(expr, sizeof expr, "(qllm-tensor-destroy %lld)", (long long)handle);
        CHECK(qllm_eshkol_eval_to_int64(expr, &result) == QLLM_SUCCESS && result == 1);
        eshkol_qllm_bridge_shutdown();
        CHECK(!eshkol_qllm_bridge_ready());
        std::puts("PASS real qLLM native registration and tagged-value execution");
        return 0;
    }
    auto* tensor = eshkol_to_qllm_tensor(source, shape, 2);
    CHECK(tensor);
    // Read with qLLM's actual API and layout, then consume with its real matmul.
    CHECK(tensor->dims == 2 && tensor->shape[0] == 2 && tensor->shape[1] == 3);
    CHECK(tensor->options.dtype == QLLM_DTYPE_FLOAT32 && qllm_tensor_get_size(tensor) == 6);
    double out[6] = {77,77,77,77,77,77}; size_t count = 5;
    CHECK(!qllm_to_eshkol_tensor(tensor, out, &count));
    CHECK(count == 6 && out[0] == 77 && out[5] == 77);
    count = 6;
    CHECK(qllm_to_eshkol_tensor(tensor, out, &count));
    for (size_t i = 0; i < 6; ++i) CHECK(out[i] == source[i]);
    size_t bshape[] = {3,1}; double bdata[] = {1,2,3};
    auto* b = eshkol_to_qllm_tensor(bdata, bshape, 2);
    auto* product = qllm_tensor_matmul(tensor, b);
    CHECK(product); count = 6;
    CHECK(qllm_to_eshkol_tensor(product, out, &count));
    CHECK(count == 2 && out[0] == 10.25 && out[1] == -3);
    auto opts = qllm_tensor_options_default(QLLM_DEVICE_CPU);
    opts.dtype = QLLM_DTYPE_INT64;
    auto* integers = qllm_tensor_create(2, shape, &opts);
    CHECK(integers); count = 6; out[0] = 77;
    CHECK(!qllm_to_eshkol_tensor(integers, out, &count) && out[0] == 77);
    const size_t overflow[] = {SIZE_MAX,2};
    CHECK(!eshkol_to_qllm_tensor(source, overflow, 2));
    // Exercise the real typed encoder with mixed constants; no ABI copies.
    qllm_eshkol_instr_t instructions[] = {{1,0},{36,0}};
    auto* program = qllm_eshkol_bytecode_to_tensor(instructions, 2);
    CHECK(program);
    qllm_eshkol_const_entry_t constants[3] = {};
    constants[0].type = ESKB_CONST_INT64; constants[0].value.i64 = 42;
    constants[1].type = ESKB_CONST_F64; constants[1].value.f64 = 3.25;
    constants[2].type = ESKB_CONST_STRING; constants[2].value.str = "typed qLLM";
    uint8_t* bytes = nullptr; size_t length = 0;
    CHECK(qllm_eshkol_tensor_to_eskb_chunk_typed(program, constants, 3, "main", &bytes, &length) == QLLM_SUCCESS);
    CHECK(length > 16);
    EskbHeader header; std::memcpy(&header, bytes, sizeof header);
    CHECK(header.magic == ESKB_MAGIC);
    CHECK(header.checksum == eskb_crc32(bytes + 16, length - 16));
    // Inspect the independent wire contract, including all three const arms.
    const uint8_t* cursor = bytes + 16;
    CHECK(*cursor++ == 2); CHECK(*cursor++ == ESKB_SECTION_CONST);
    uint64_t section_size = 0;
    int consumed = eskb_read_leb128(cursor, length - (cursor - bytes), &section_size);
    CHECK(consumed > 0); cursor += consumed;
    CHECK(*cursor++ == ESKB_SECTION_CODE);
    consumed = eskb_read_leb128(cursor, length - (cursor - bytes), &section_size);
    CHECK(consumed > 0); cursor += consumed;
    CHECK(*cursor++ == 3); CHECK(*cursor++ == ESKB_CONST_INT64);
    int64_t iv; std::memcpy(&iv, cursor, 8); cursor += 8; CHECK(iv == 42);
    CHECK(*cursor++ == ESKB_CONST_F64);
    double fv; std::memcpy(&fv, cursor, 8); cursor += 8; CHECK(fv == 3.25);
    CHECK(*cursor++ == ESKB_CONST_STRING); CHECK(*cursor++ == 10);
    CHECK(std::memcmp(cursor, "typed qLLM", 10) == 0);
    std::free(bytes); bytes = nullptr; length = 0;
    auto status = eshkol_qllm_tensor_to_eskb(program, constants, 3, "main", &bytes, &length);
    if (header.version != ESKB_VERSION) {
        CHECK(status == QLLM_ERROR_INVALID_STATE_CODE && !bytes && !length);
        CHECK(std::strstr(qllm_get_last_error()->message, "ESKB version"));
        std::printf("PASS explicit capability refusal: %s\n", qllm_get_last_error()->message);
    } else {
        CHECK(status == QLLM_SUCCESS && bytes && length > 16);
        std::free(bytes);
    }
    eshkol_qllm_tensor_destroy(program);
    eshkol_qllm_tensor_destroy(integers);
    eshkol_qllm_tensor_destroy(product);
    eshkol_qllm_tensor_destroy(b);
    eshkol_qllm_tensor_destroy(tensor);
#endif
    eshkol_qllm_bridge_shutdown();
    CHECK(!eshkol_qllm_bridge_ready());
    std::puts("PASS qLLM capability, real tensor ABI, and typed ESKB contract");
    return 0;
}
