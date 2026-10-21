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
Counter counter(void* library, const char* name) {
    return reinterpret_cast<Counter>(dlsym(library, name));
}
int fail(const std::string& message) { std::cerr << "FAIL: " << message << '\n'; return 1; }
}

int main(int argc, char** argv) {
    if (argc != 3) return fail("usage: pjrt_reusable_program_test reuse|baseline|owner|reinit|moved-from|compile-fail|no-active|uninitialized|empty|typed-dtype|typed-shape|typed-on-device-input|typed-on-device-output PLUGIN");
    const std::string mode = argv[1], path = argv[2];
    setenv("ESHKOL_XLA_PJRT", "1", 1);
    setenv("ESHKOL_PJRT_PLUGIN_PATH", path.c_str(), 1);
    setenv("PJRT_TEST_OUTPUT_MISMATCH", "ok", 1);
    if (mode == "compile-fail") setenv("PJRT_TEST_COMPILE_FAIL", "1", 1);
    if (mode == "no-active") setenv("ESHKOL_XLA_PJRT", "0", 1);
    void* monitor = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!monitor) return fail("cannot monitor offline plugin");
    auto compiles = counter(monitor, "PjrtTestCompileCalls");
    auto executions = counter(monitor, "PjrtTestExecuteCalls");
    auto destroyed = counter(monitor, "PjrtTestDestroyedExecutables");
    auto copies = counter(monitor, "PjrtTestHostCopies");
    auto clients = counter(monitor, "PjrtTestDestroyedClients");
    if (!compiles || !executions || !destroyed || !copies || !clients) return fail("missing plugin counters");

    int result = 0;
    const double input_value = 2.0;
    std::vector<eshkol::xla::BufferDescriptor> inputs = {{const_cast<double*>(&input_value), {1}, sizeof(double), false}};
    std::array<double, 1> output{};
    std::vector<eshkol::xla::BufferDescriptor> outputs = {{output.data(), {1}, sizeof(double), false}};
    auto run = [&](eshkol::xla::XLARuntime& runtime) {
        auto r = runtime.executeStableHLO("module {}", inputs, outputs);
        if (!r.success || output[0] != 8.25) result = fail("convenience execution failed or result differed");
    };

    if (mode == "reuse") {
        std::unique_ptr<eshkol::xla::StableHLOProgram> program;
        {
            eshkol::xla::XLARuntime runtime;
            runtime.initialize(eshkol::xla::Target::CPU);
            std::string diagnostic;
            program = runtime.compileStableHLO("module {}", &diagnostic);
            if (!program) return fail("compile failed: " + diagnostic);
            for (int i = 0; i < 3; ++i) {
                auto r = runtime.executeCompiledStableHLO(*program, inputs, outputs);
                if (!r.success || output[0] != 8.25) result = fail("reused result incorrect");
            }
            if (compiles() != 1 || executions() != 3 || destroyed() != 0)
                result = fail("reuse counters were not compile=1, execute=3, destroy=0 before release");
        }
        if (destroyed() != 0 || clients() != 0) result = fail("runtime teardown released a live program session");
        program.reset();
        if (destroyed() != 1 || clients() != 1) result = fail("session resources were not released with the program");
    } else if (mode == "baseline") {
        eshkol::xla::XLARuntime runtime;
        runtime.initialize(eshkol::xla::Target::CPU);
        for (int i = 0; i < 3; ++i) run(runtime);
        if (compiles() != 3 || executions() != 3 || destroyed() != 3) result = fail("baseline counters were not all three");
    } else if (mode == "owner" || mode == "reinit" || mode == "moved-from") {
        eshkol::xla::XLARuntime first, second;
        first.initialize(eshkol::xla::Target::CPU);
        second.initialize(eshkol::xla::Target::CPU);
        auto p = first.compileStableHLO("module {}");
        if (!p) return fail("owner control compile failed");
        if (mode == "owner") {
            auto refused = second.executeCompiledStableHLO(*p, inputs, outputs);
            if (refused.success || executions() != 0 || copies() != 0) result = fail("wrong runtime accepted program or performed work");
        } else if (mode == "reinit") {
            first.initialize(eshkol::xla::Target::CPU);
            auto refused = first.executeCompiledStableHLO(*p, inputs, outputs);
            if (refused.success || executions() != 0 || copies() != 0) result = fail("reinitialized runtime accepted stale program");
        } else {
            eshkol::xla::StableHLOProgram moved(std::move(*p));
            auto refused = first.executeCompiledStableHLO(*p, inputs, outputs);
            if (refused.success || executions() != 0 || copies() != 0) result = fail("moved-from program was accepted");
            auto accepted = first.executeCompiledStableHLO(moved, inputs, outputs);
            if (!accepted.success || executions() != 1 || copies() != 1) result = fail("moved-to program lost ownership");
        }
    } else if (mode == "compile-fail") {
        eshkol::xla::XLARuntime runtime;
        runtime.initialize(eshkol::xla::Target::CPU);
        std::string diagnostic;
        auto p = runtime.compileStableHLO("module {}", &diagnostic);
        if (p || diagnostic.find("offline compile failure") == std::string::npos || executions() != 0)
            result = fail("compile failure did not propagate without execution");
    } else if (mode == "no-active") {
        eshkol::xla::XLARuntime runtime;
        runtime.initialize(eshkol::xla::Target::CPU);
        std::string diagnostic;
        auto p = runtime.compileStableHLO("module {}", &diagnostic);
        auto r = runtime.executeStableHLO("module {}", inputs, outputs);
        if (p || r.success || compiles() != 0 || executions() != 0 || copies() != 0)
            result = fail("inactive PJRT path compiled or executed");
    } else if (mode == "uninitialized") {
        eshkol::xla::XLARuntime runtime;
        std::string diagnostic;
        auto p = runtime.compileStableHLO("module {}", &diagnostic);
        if (p || diagnostic.find("not initialized") == std::string::npos || compiles() != 0)
            result = fail("uninitialized compile was not refused before compilation");
    } else if (mode == "empty") {
        eshkol::xla::XLARuntime runtime;
        runtime.initialize(eshkol::xla::Target::CPU);
        std::string diagnostic;
        auto p = runtime.compileStableHLO("", &diagnostic);
        if (p || diagnostic.find("empty") == std::string::npos || compiles() != 0)
            result = fail("empty module was not refused before compilation");
    } else if (mode == "typed-dtype" || mode == "typed-shape" ||
               mode == "typed-on-device-input" || mode == "typed-on-device-output") {
        const std::string mismatch = mode.substr(6);
        setenv("PJRT_TEST_OUTPUT_MISMATCH", mismatch.c_str(), 1);
        eshkol::xla::XLARuntime runtime;
        runtime.initialize(eshkol::xla::Target::CPU);
        auto p = runtime.compileStableHLO("module {}");
        if (!p) return fail("typed metadata control compile failed");
        if (mode == "typed-on-device-input") inputs[0].on_device = true;
        if (mode == "typed-on-device-output") outputs[0].on_device = true;
        auto r = runtime.executeCompiledStableHLO(*p, inputs, outputs);
        const std::string expected = mode == "typed-dtype" ? "element type"
            : mode == "typed-shape" ? "shape" : "host-resident";
        if (r.success || r.error_message.find(expected) == std::string::npos || copies() != 0)
            result = fail("typed execution did not refuse unsafe descriptor before host copy");
    } else return fail("unknown mode");
    dlclose(monitor);
    if (result) return result;
    std::cout << "PASS: offline PJRT reusable program " << mode << '\n';
    return 0;
}
