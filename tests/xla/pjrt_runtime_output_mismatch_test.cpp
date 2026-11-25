#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/xla_runtime.h"

#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Counter = int (*)();

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

Counter counter(void* library, const char* name) {
    return reinterpret_cast<Counter>(dlsym(library, name));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3)
        return fail("usage: pjrt_runtime_output_mismatch_test dtype|shape|on-device-input|on-device-output PLUGIN");
    const std::string mode = argv[1];
    if (mode != "dtype" && mode != "shape" && mode != "on-device-input" &&
        mode != "on-device-output")
        return fail("unknown control mode: " + mode);
    const std::string plugin_path = argv[2];
    setenv("ESHKOL_XLA_PJRT", "1", 1);
    setenv("ESHKOL_PJRT_PLUGIN_PATH", plugin_path.c_str(), 1);
    setenv("PJRT_TEST_OUTPUT_MISMATCH", mode.c_str(), 1);

    void* monitor = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!monitor) return fail("could not open offline test plugin for counters");
    Counter destroyed_buffers = counter(monitor, "PjrtTestDestroyedBuffers");
    Counter destroyed_executables = counter(monitor, "PjrtTestDestroyedExecutables");
    Counter host_copies = counter(monitor, "PjrtTestHostCopies");
    Counter host_transfers = counter(monitor, "PjrtTestHostTransfers");
    if (!destroyed_buffers || !destroyed_executables || !host_copies || !host_transfers) {
        dlclose(monitor);
        return fail("offline test plugin did not expose its cleanup counters");
    }

    const double input_value = 7.0;
    const std::vector<eshkol::xla::BufferDescriptor> inputs = {
        {const_cast<double*>(&input_value), {1}, sizeof(double), mode == "on-device-input"}
    };
    const std::vector<int64_t> output_shape = mode == "dtype"
        ? std::vector<int64_t>{1} : std::vector<int64_t>{2, 3};
    std::array<double, 6> output;
    output.fill(91.25);
    std::vector<eshkol::xla::BufferDescriptor> outputs = {
        {output.data(), output_shape, sizeof(double), mode == "on-device-output"}
    };

    int result_code = 0;
    {
        eshkol::xla::XLARuntime runtime;
        if (!runtime.initialize(eshkol::xla::Target::CPU))
            return fail("runtime did not initialize with the offline PJRT plugin");
        if (runtime.getDescription().find("PJRT device active for StableHLO") == std::string::npos)
            return fail("runtime did not report the fake PJRT backend as active");

        const auto result = runtime.executeStableHLO("module {}", inputs, outputs);
        if (result.success) {
            result_code = fail("mismatched PJRT output metadata was accepted");
        } else {
            const std::string expected = mode == "dtype" ? "has device element type"
                : mode == "shape" ? "has device shape" : "host-resident";
            if (result.error_message.find(expected) == std::string::npos)
                result_code = fail("wrong metadata diagnostic: " + result.error_message);
        }
        for (double value : output) {
            if (value != 91.25) {
                result_code = fail("output memory changed before metadata validation failed");
                break;
            }
        }
        if (host_copies() != 0)
            result_code = fail("PJRT host copy ran before all outputs were validated");
        const bool descriptor_control = mode == "on-device-input" ||
                                        mode == "on-device-output";
        if (host_transfers() != (descriptor_control ? 0 : 1))
            result_code = fail("PJRT host transfer count was incorrect");
        if (destroyed_buffers() != (descriptor_control ? 0 : 2))
            result_code = fail("PJRT buffer cleanup count was incorrect");
        if (destroyed_executables() != 1)
            result_code = fail("loaded PJRT executable was not destroyed");
    }
    dlclose(monitor);
    if (result_code != 0) return result_code;
    std::cout << "PASS: offline PJRT " << mode
              << " refused before unsafe access; output preserved and cleanup verified\n";
    return 0;
}
