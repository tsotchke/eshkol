#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/xla_runtime.h"
extern "C" {
#include "pjrt_c_api.h"
}

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
int fail(const std::string& message) { std::cerr << "FAIL: " << message << '\n'; return 1; }
}

int main(int argc, char** argv) {
    if (argc != 3) return fail("usage: pjrt_typed_buffers_test f32|bf16|f32-int32|bf16-uint16|later-transfer-failure|overflow|negative|on-device PLUGIN");
    const std::string mode = argv[1], plugin = argv[2];
    setenv("ESHKOL_XLA_PJRT", "1", 1);
    setenv("ESHKOL_PJRT_PLUGIN_PATH", plugin.c_str(), 1);
    setenv("PJRT_TEST_OUTPUT_MISMATCH", "ok", 1);
    const bool two_outputs = mode == "later-transfer-failure";
    if (two_outputs) {
        setenv("PJRT_TEST_TWO_OUTPUTS", "1", 1);
        setenv("PJRT_TEST_FAIL_SECOND_COPY", "1", 1);
    }
    eshkol::xla::BufferElementType type = eshkol::xla::BufferElementType::kF32;
    const char* device_type = "f32";
    if (mode == "bf16" || mode == "bf16-uint16") {
        type = eshkol::xla::BufferElementType::kBF16;
        device_type = mode == "bf16" ? "bf16" : "u16";
    } else if (mode == "f32-int32") device_type = "s32";
    else if (mode != "f32" && !two_outputs && mode != "overflow" &&
             mode != "negative" && mode != "on-device") return fail("unknown control mode");
    setenv("PJRT_TEST_DEVICE_OUTPUT_TYPE", device_type, 1);

    float f32_input = 2.0f, f32_output = -17.0f;
    uint16_t bf16_input = 0x4000, bf16_output = 0xbeef;
    void* input_data = type == eshkol::xla::BufferElementType::kBF16
        ? static_cast<void*>(&bf16_input) : static_cast<void*>(&f32_input);
    void* output_data = type == eshkol::xla::BufferElementType::kBF16
        ? static_cast<void*>(&bf16_output) : static_cast<void*>(&f32_output);
    std::vector<int64_t> input_shape = mode == "overflow"
        ? std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 2}
        : mode == "negative" ? std::vector<int64_t>{-1} : std::vector<int64_t>{1};
    std::vector<eshkol::xla::TypedBufferDescriptor> inputs = {
        {input_data, input_shape, type, mode == "on-device"}};
    std::vector<eshkol::xla::TypedBufferDescriptor> outputs;
    outputs.push_back({output_data, {1}, type, false});
    if (two_outputs) outputs.push_back({output_data, {1}, type, false});

    eshkol::xla::XLARuntime runtime;
    if (!runtime.initialize(eshkol::xla::Target::CPU)) return fail("offline PJRT initialization failed");
    auto program = runtime.compileStableHLO("module {}");
    if (!program) return fail("offline PJRT compile failed");
    const auto result = runtime.executeCompiledStableHLO(*program, inputs, outputs);
    void* monitor = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!monitor) return fail("could not inspect offline plugin transfer type");
    using LastType = int (*)();
    auto last_type = reinterpret_cast<LastType>(dlsym(monitor, "PjrtTestLastInputElementType"));
    if (!last_type) return fail("offline plugin did not expose input type control");
    const bool rejected_before_transfer = mode == "overflow" || mode == "negative" || mode == "on-device";
    if (rejected_before_transfer) {
        if (last_type() != 0) return fail("invalid input descriptor reached PJRT transfer");
    } else {
        const int expected_type = type == eshkol::xla::BufferElementType::kBF16
            ? static_cast<int>(PJRT_Buffer_Type_BF16) : static_cast<int>(PJRT_Buffer_Type_F32);
        if (last_type() != expected_type) return fail("semantic host type was not passed to PJRT");
    }
    dlclose(monitor);
    const bool mismatch = mode == "f32-int32" || mode == "bf16-uint16" ||
        mode == "overflow" || mode == "negative" || mode == "on-device";
    if (mismatch) {
        if (result.success) return fail("invalid typed host descriptor was accepted");
        if ((mode == "f32-int32" || mode == "bf16-uint16") &&
            result.error_message.find("element type") == std::string::npos)
            return fail("same-width integer element type was accepted as a float");
        if (f32_output != -17.0f || bf16_output != 0xbeef)
            return fail("mismatched metadata changed caller output");
    } else if (two_outputs) {
        if (result.success || result.error_message.find("offline compile failure") == std::string::npos)
            return fail("later host transfer failure did not propagate");
        if (f32_output != -17.0f || bf16_output != 0xbeef)
            return fail("later transfer failure partially committed outputs");
    } else if (!result.success) return fail("matching typed host buffers were refused: " + result.error_message);
    std::cout << "PASS: offline typed PJRT control " << mode << '\n';
    return 0;
}
