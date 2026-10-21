#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/xla_runtime.h"
extern "C" {
#include "pjrt_c_api.h"
}

#include <cstdlib>
#include <dlfcn.h>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
int fail(const std::string& message) { std::cerr << "FAIL: " << message << '\n'; return 1; }
}

int main(int argc, char** argv) {
    if (argc != 3) return fail("usage: pjrt_typed_buffers_test f32|bf16|f32-int32|bf16-uint16|later-transfer-failure|overflow|negative|on-device|output-shape|output-on-device|invalid-output-type PLUGIN");
    const std::string mode = argv[1], plugin = argv[2];
    setenv("ESHKOL_XLA_PJRT", "1", 1);
    setenv("ESHKOL_PJRT_PLUGIN_PATH", plugin.c_str(), 1);
    setenv("PJRT_TEST_OUTPUT_MISMATCH", "ok", 1);
    if (mode == "output-shape") setenv("PJRT_TEST_OUTPUT_MISMATCH", "shape", 1);
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
             mode != "negative" && mode != "on-device" && mode != "output-shape" &&
             mode != "output-on-device" && mode != "invalid-output-type") return fail("unknown control mode");
    setenv("PJRT_TEST_DEVICE_OUTPUT_TYPE", device_type, 1);

    float f32_input = 2.0f, f32_output = -17.0f, f32_output2 = -23.0f;
    uint16_t bf16_input = 0x4000, bf16_output = 0xbeef, bf16_output2 = 0xcafe;
    void* input_data = type == eshkol::xla::BufferElementType::kBF16
        ? static_cast<void*>(&bf16_input) : static_cast<void*>(&f32_input);
    void* output_data = type == eshkol::xla::BufferElementType::kBF16
        ? static_cast<void*>(&bf16_output) : static_cast<void*>(&f32_output);
    std::vector<int64_t> input_shape = mode == "overflow"
        ? std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 2}
        : mode == "negative" ? std::vector<int64_t>{-1} : std::vector<int64_t>{1};
    if (mode == "invalid-output-type") type = static_cast<eshkol::xla::BufferElementType>(99);
    std::vector<eshkol::xla::TypedBufferDescriptor> inputs = {
        {input_data, input_shape, type, mode == "on-device"}};
    std::vector<eshkol::xla::TypedBufferDescriptor> outputs;
    outputs.push_back({output_data, {1}, type, mode == "output-on-device"});
    void* second_output_data = type == eshkol::xla::BufferElementType::kBF16
        ? static_cast<void*>(&bf16_output2) : static_cast<void*>(&f32_output2);
    if (two_outputs) outputs.push_back({second_output_data, {1}, type, false});

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
    using Counter = int (*)();
    auto counter = [&](const char* name) { return reinterpret_cast<Counter>(dlsym(monitor, name)); };
    auto host_copies = counter("PjrtTestHostCopies");
    auto successful_copies = counter("PjrtTestSuccessfulHostCopies");
    auto destroyed_buffers = counter("PjrtTestDestroyedBuffers");
    auto destroyed_executables = counter("PjrtTestDestroyedExecutables");
    if (!host_copies || !successful_copies || !destroyed_buffers || !destroyed_executables)
        return fail("offline plugin omitted transfer or cleanup controls");
    const bool rejected_before_transfer = mode == "overflow" || mode == "negative" ||
        mode == "on-device" || mode == "output-on-device" || mode == "invalid-output-type";
    if (rejected_before_transfer) {
        if (last_type() != 0) return fail("invalid input descriptor reached PJRT transfer");
    } else {
        const int expected_type = type == eshkol::xla::BufferElementType::kBF16
            ? static_cast<int>(PJRT_Buffer_Type_BF16) : static_cast<int>(PJRT_Buffer_Type_F32);
        if (last_type() != expected_type) return fail("semantic host type was not passed to PJRT");
    }
    const bool mismatch = mode == "f32-int32" || mode == "bf16-uint16" ||
        mode == "overflow" || mode == "negative" || mode == "on-device" ||
        mode == "output-shape" || mode == "output-on-device" || mode == "invalid-output-type";
    if (mismatch) {
        if (result.success) return fail("invalid typed host descriptor was accepted");
        if ((mode == "f32-int32" || mode == "bf16-uint16") &&
            result.error_message.find("element type") == std::string::npos)
            return fail("same-width integer element type was accepted as a float");
        if (mode == "output-shape" && result.error_message.find("shape") == std::string::npos)
            return fail("typed output shape mismatch was not diagnosed");
        if (host_copies() != 0 || successful_copies() != 0)
            return fail("invalid typed descriptor reached a host output copy");
        if (f32_output != -17.0f || bf16_output != 0xbeef)
            return fail("mismatched metadata changed caller output");
    } else if (two_outputs) {
        if (result.success || result.error_message.find("offline compile failure") == std::string::npos)
            return fail("later host transfer failure did not propagate");
        if (host_copies() != 2 || successful_copies() != 1)
            return fail("first output transfer did not write its staged payload before the second failed");
        if (f32_output != -17.0f || f32_output2 != -23.0f ||
            bf16_output != 0xbeef || bf16_output2 != 0xcafe)
            return fail("later transfer failure partially committed outputs");
    } else {
        if (!result.success) return fail("matching typed host buffers were refused: " + result.error_message);
        if (type == eshkol::xla::BufferElementType::kF32) {
            const uint32_t expected_bits = 0x3f800001u;
            if (std::memcmp(&f32_output, &expected_bits, sizeof(expected_bits)) != 0)
                return fail("f32 transfer bytes differed from the offline fixture payload");
        } else if (type == eshkol::xla::BufferElementType::kBF16) {
            const uint16_t expected_bits = 0x3f81u;
            if (std::memcmp(&bf16_output, &expected_bits, sizeof(expected_bits)) != 0)
                return fail("bf16 transfer bytes differed from the offline fixture payload");
        }
    }
    program.reset();
    const int expected_destroyed_buffers = two_outputs ? 3
        : (mode == "f32-int32" || mode == "bf16-uint16" || mode == "output-shape") ? 2
        : rejected_before_transfer ? 0 : 2;
    if (destroyed_buffers() != expected_destroyed_buffers)
        return fail("PJRT buffers were not cleaned up after typed execution");
    if (destroyed_executables() != 1)
        return fail("typed program executable was not destroyed on release");
    dlclose(monitor);
    std::cout << "PASS: offline typed PJRT control " << mode << '\n';
    return 0;
}
