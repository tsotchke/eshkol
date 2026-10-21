#include "eshkol/backend/xla/xla_runtime.h"
#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/xla_types.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void addOne(const void* const* inputs, void* const* outputs) {
    *static_cast<double*>(outputs[0]) = *static_cast<const double*>(inputs[0]) + 1.0;
}

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return fail("usage: pjrt_runtime_control_test off|missing-plugin [plugin-path]");
    const std::string mode = argv[1];
    if ((mode == "off" && argc != 2) || (mode == "missing-plugin" && argc != 3))
        return fail("usage: pjrt_runtime_control_test off|missing-plugin [plugin-path]");
    if (mode != "off" && mode != "missing-plugin") return fail("unknown mode: " + mode);
    setenv("ESHKOL_XLA_PJRT", mode == "off" ? "0" : "1", 1);
    if (mode != "off") {
        setenv("ESHKOL_PJRT_PLUGIN_PATH", argv[2], 1);
    }

    eshkol::xla::XLARuntime runtime;
    if (!runtime.initialize(eshkol::xla::Target::CPU))
        return fail("CPU runtime did not initialize");

    const std::string description = runtime.getDescription();
    if (description.find("CPU, LLVM direct dispatch") == std::string::npos)
        return fail("PJRT opt-out did not report the actual CPU backend: " + description);
    if (description.find("PJRT device execution") != std::string::npos)
        return fail("PJRT opt-out falsely reported device execution: " + description);
    if (mode == "missing-plugin" &&
        (description.find("failed to load PJRT plugin") == std::string::npos ||
         description.find("Falling back to LLVM-direct dispatch") == std::string::npos))
        return fail("failed PJRT request did not report the actual CPU fallback: " + description);

    const double input_value = 41.0;
    double output_value = 0.0;
    const std::vector<eshkol::xla::BufferDescriptor> inputs = {
        {const_cast<double*>(&input_value), {1}, sizeof(double), false}
    };
    std::vector<eshkol::xla::BufferDescriptor> outputs = {
        {&output_value, {1}, sizeof(double), false}
    };
    const auto refusal = runtime.executeStableHLO("module {}", inputs, outputs);
    if (refusal.success)
        return fail("StableHLO execution unexpectedly succeeded without an active PJRT device");
    if (refusal.error_message.empty())
        return fail("inactive PJRT StableHLO request returned no refusal diagnostic");
    if (mode == "missing-plugin" &&
        refusal.error_message.find("failed to load PJRT plugin") == std::string::npos)
        return fail("StableHLO refusal did not report the plugin failure: " + refusal.error_message);

    const auto result = runtime.execute(reinterpret_cast<void*>(&addOne), inputs, outputs);
    if (!result.success) return fail("CPU executable failed: " + result.error_message);
    if (output_value != 42.0) return fail("CPU fallback returned an incorrect result");

    std::cout << "PASS: PJRT " << mode
              << " reports the actual CPU path, refuses device execution, and preserves host execution\n";
    return 0;
}
