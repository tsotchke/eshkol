/*
 * PJRT Round-Trip Test
 *
 * pjrt_smoke_test.cpp proves PJRT connectivity only: discovery, load, client
 * creation, platform name, device enumeration. It never asks the plugin to
 * compile or run anything. This file is the first test in the repo that
 * proves a StableHLO program actually EXECUTES through Eshkol's PJRT client
 * on real hardware:
 *
 *   StableHLOEmitter builds a module -> PjrtClient::compile() lowers it to a
 *   loaded executable -> bufferFromHost() stages real inputs on the device ->
 *   PjrtClient::execute() runs it -> bufferToHost() reads the result back ->
 *   the result is checked element-by-element against a hand-computed answer.
 *
 * Two programs are run this way: an elementwise add (proves the basic
 * compile/execute/transfer plumbing) and a matmul (proves shape plumbing
 * beyond elementwise — DotDimensionNumbers, a rank-2 result type, a
 * non-square contraction).
 *
 * A negative control (unconditional, once a plugin is found) hands compile()
 * a StableHLO module whose function declares one result type but returns a
 * value of a different type. compile() must refuse it with a non-empty
 * error. Without this, a compile() that regressed into a rubber stamp could
 * still let every test above pass by accident — a gate that cannot fail is
 * worthless.
 *
 * Only builds when ESHKOL_XLA_ENABLED=ON (see CMakeLists.txt).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "eshkol/backend/xla/pjrt_client.h"
#include "eshkol/backend/xla/stablehlo_emitter.h"

using eshkol::xla::DotDimensionNumbers;
using eshkol::xla::ElementType;
using eshkol::xla::findPjrtPlugin;
using eshkol::xla::PjrtClient;
using eshkol::xla::PjrtDeviceInfo;
using eshkol::xla::PjrtElementType;
using eshkol::xla::PjrtPlugin;
using eshkol::xla::StableHLOEmitter;

// Test utilities — same idiom as tests/xla/pjrt_smoke_test.cpp.
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

static int tests_passed = 0;
static int tests_failed = 0;

// State shared between steps, mirroring pjrt_smoke_test.cpp's g_plugin/g_client.
static std::string g_plugin_path;
static std::unique_ptr<PjrtPlugin> g_plugin;
static std::unique_ptr<PjrtClient> g_client;
static int g_device_index = -1;   // index into g_client->devices(), addressable
static std::string g_platform_name;
static size_t g_device_count = 0;
static int g_elements_checked = 0;

// The exact search order findPjrtPlugin("tpu") uses (pjrt_client.cpp), kept
// in sync here purely for the SKIP message, exactly as pjrt_smoke_test does.
static std::string searchedLocationsDescription() {
    std::string out;
    out += "  1. $ESHKOL_PJRT_PLUGIN_PATH (not set or empty)\n";
    const char* home = std::getenv("HOME");
    std::string home_str = home ? home : "$HOME (unset)";
    out += "  2. " + home_str + "/.local/lib/python3.{10,11,12}/site-packages/libtpu/libtpu.so\n";
    out += "  3. /usr/lib/libtpu.so\n";
    out += "  4. /usr/local/lib/libtpu.so\n";
    return out;
}

static bool traceModuleEnabled() {
    const char* v = std::getenv("ESHKOL_XLA_TRACE_MODULE");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

// Print the StableHLO module text to stderr under ESHKOL_XLA_TRACE_MODULE=1
// so a compile failure can actually be read (the module never appears
// otherwise, and a compile error message alone rarely says which line of IR
// it is complaining about).
static void maybeTraceModule(const std::string& label, const std::string& text) {
    if (!traceModuleEnabled()) return;
    std::cerr << "----- ESHKOL_XLA_TRACE_MODULE: " << label << " -----" << std::endl;
    std::cerr << text << std::endl;
    std::cerr << "----- end " << label << " -----" << std::endl;
}

// ===== Test 1 (discovery) is handled specially in main(): finding no plugin
// is exit 77, not a FAIL — see tests/xla/pjrt_smoke_test.cpp for the same
// convention. =====

// ===== Test 2: load the plugin and create a client =====
bool test_pjrt_plugin_loads_and_client_creates() {
    std::cout << "Test: PJRT Plugin Load + Client Create... ";

    std::string error;
    g_plugin = PjrtPlugin::load(g_plugin_path, &error);
    TEST_ASSERT(g_plugin != nullptr,
        "PjrtPlugin::load(" + g_plugin_path + ") must succeed for a plugin "
        "findPjrtPlugin() itself reported as present: " + error);

    g_client = PjrtClient::create(g_plugin.get(), &error);
    TEST_ASSERT(g_client != nullptr, "PjrtClient::create() must succeed: " + error);

    g_platform_name = g_client->platformName();
    TEST_ASSERT(!g_platform_name.empty(), "platformName() must be non-empty on a real client");

    const std::vector<PjrtDeviceInfo>& devices = g_client->devices();
    TEST_ASSERT(!devices.empty(), "devices() must be non-empty on a real client with a real plugin");
    g_device_count = devices.size();

    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i].is_addressable) {
            g_device_index = static_cast<int>(i);
            break;
        }
    }
    TEST_ASSERT(g_device_index >= 0,
        "at least one addressable device is required to run anything");

    std::cout << "PASS (platform=\"" << g_platform_name << "\", "
              << g_device_count << " device(s), first addressable index="
              << g_device_index << ")" << std::endl;
    return true;
}

// ===== Test 3: elementwise add, compiled and executed through PJRT =====
bool test_pjrt_add_roundtrip() {
    std::cout << "Test: PJRT Add Round Trip (f32[8] + f32[8])... ";

    StableHLOEmitter emitter;
    TEST_ASSERT(emitter.isAvailable(), "StableHLOEmitter must have MLIR support compiled in");

    std::vector<void*> args = emitter.beginFunction(
        "main", {{{8}, ElementType::F32}, {{8}, ElementType::F32}});
    TEST_ASSERT(args.size() == 2, "beginFunction must return 2 block arguments for 2 params");

    void* sum = emitter.emitAdd(args[0], args[1]);
    TEST_ASSERT(sum != nullptr, "emitAdd must produce a value");
    TEST_ASSERT(emitter.endFunction({sum}), "endFunction must succeed");

    std::string module_text = emitter.serializeToString();
    TEST_ASSERT(!module_text.empty(), "serializeToString must produce non-empty StableHLO text");
    maybeTraceModule("add", module_text);

    std::string error;
    PJRT_LoadedExecutable* executable = g_client->compile(module_text, "mlir", &error);
    TEST_ASSERT(executable != nullptr,
        "compile() of the add module must succeed — this is a genuine FAIL, "
        "never a skip, once a plugin has been found: " + error);

    float a[8], b[8], expected[8];
    for (int i = 0; i < 8; i++) {
        a[i] = static_cast<float>(i) + 1.0f;         // 1..8
        b[i] = static_cast<float>(i) * 2.0f + 0.5f;  // 0.5, 2.5, 4.5, ...
        expected[i] = a[i] + b[i];
    }

    PJRT_Buffer* buf_a = g_client->bufferFromHost(a, PjrtElementType::kF32, {8}, g_device_index, &error);
    TEST_ASSERT(buf_a != nullptr, "bufferFromHost(a) must succeed: " + error);
    PJRT_Buffer* buf_b = g_client->bufferFromHost(b, PjrtElementType::kF32, {8}, g_device_index, &error);
    TEST_ASSERT(buf_b != nullptr, "bufferFromHost(b) must succeed: " + error);

    std::vector<PJRT_Buffer*> outputs;
    auto status = g_client->execute(executable, {buf_a, buf_b}, outputs);
    TEST_ASSERT(status.ok(), "execute() of the add module must succeed: " + status.message());
    TEST_ASSERT(outputs.size() == 1,
        "add module must produce exactly 1 output, got " + std::to_string(outputs.size()));

    float result[8] = {};
    auto to_host_status = g_client->bufferToHost(outputs[0], result, sizeof(result));
    TEST_ASSERT(to_host_status.ok(),
        "bufferToHost of the add result must succeed: " + to_host_status.message());

    bool all_match = true;
    std::string mismatch_detail;
    for (int i = 0; i < 8; i++) {
        if (std::fabs(result[i] - expected[i]) > 1e-4f) {
            all_match = false;
            mismatch_detail += " [" + std::to_string(i) + "]=" + std::to_string(result[i]) +
                                " expected=" + std::to_string(expected[i]);
        }
    }
    TEST_ASSERT(all_match, "add result mismatch, index[]=actual expected=hand-computed:" + mismatch_detail);

    g_client->destroyBuffer(buf_a);
    g_client->destroyBuffer(buf_b);
    g_client->destroyBuffer(outputs[0]);
    g_client->destroyExecutable(executable);

    g_elements_checked += 8;
    std::cout << "PASS (8 elements checked)" << std::endl;
    return true;
}

// ===== Test 4: matmul, exercising shape plumbing beyond elementwise =====
bool test_pjrt_matmul_roundtrip() {
    std::cout << "Test: PJRT Matmul Round Trip (f32[2,3] x f32[3,2])... ";

    StableHLOEmitter emitter;
    TEST_ASSERT(emitter.isAvailable(), "StableHLOEmitter must have MLIR support compiled in");

    std::vector<void*> args = emitter.beginFunction(
        "main", {{{2, 3}, ElementType::F32}, {{3, 2}, ElementType::F32}});
    TEST_ASSERT(args.size() == 2, "beginFunction must return 2 block arguments for 2 params");

    // Canonical matrix-multiply contraction: lhs dim 1 (columns) against rhs
    // dim 0 (rows), no batching. Result is laid out [lhs free dims][rhs free
    // dims] = [2, 2] (see Impl::dotGeneral in stablehlo_emitter.cpp).
    DotDimensionNumbers dims;
    dims.lhs_contracting_dims = {1};
    dims.rhs_contracting_dims = {0};
    void* product = emitter.emitMatmul(args[0], args[1], dims);
    TEST_ASSERT(product != nullptr, "emitMatmul must produce a value");
    TEST_ASSERT(emitter.endFunction({product}), "endFunction must succeed");

    std::string module_text = emitter.serializeToString();
    TEST_ASSERT(!module_text.empty(), "serializeToString must produce non-empty StableHLO text");
    maybeTraceModule("matmul", module_text);

    std::string error;
    PJRT_LoadedExecutable* executable = g_client->compile(module_text, "mlir", &error);
    TEST_ASSERT(executable != nullptr,
        "compile() of the matmul module must succeed — this is a genuine FAIL, "
        "never a skip, once a plugin has been found: " + error);

    // A = [[1,2,3],[4,5,6]] (row-major [2,3]); B = [[1,2],[3,4],[5,6]] (row-major [3,2]).
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {1, 2, 3, 4, 5, 6};
    // A*B = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
    //     = [[22, 28], [49, 64]], hand-computed, row-major [2,2].
    float expected[4] = {22, 28, 49, 64};

    PJRT_Buffer* buf_a = g_client->bufferFromHost(a, PjrtElementType::kF32, {2, 3}, g_device_index, &error);
    TEST_ASSERT(buf_a != nullptr, "bufferFromHost(a) must succeed: " + error);
    PJRT_Buffer* buf_b = g_client->bufferFromHost(b, PjrtElementType::kF32, {3, 2}, g_device_index, &error);
    TEST_ASSERT(buf_b != nullptr, "bufferFromHost(b) must succeed: " + error);

    std::vector<PJRT_Buffer*> outputs;
    auto status = g_client->execute(executable, {buf_a, buf_b}, outputs);
    TEST_ASSERT(status.ok(), "execute() of the matmul module must succeed: " + status.message());
    TEST_ASSERT(outputs.size() == 1,
        "matmul module must produce exactly 1 output, got " + std::to_string(outputs.size()));

    float result[4] = {};
    auto to_host_status = g_client->bufferToHost(outputs[0], result, sizeof(result));
    TEST_ASSERT(to_host_status.ok(),
        "bufferToHost of the matmul result must succeed: " + to_host_status.message());

    bool all_match = true;
    std::string mismatch_detail;
    for (int i = 0; i < 4; i++) {
        if (std::fabs(result[i] - expected[i]) > 1e-3f) {
            all_match = false;
            mismatch_detail += " [" + std::to_string(i) + "]=" + std::to_string(result[i]) +
                                " expected=" + std::to_string(expected[i]);
        }
    }
    TEST_ASSERT(all_match, "matmul result mismatch, index[]=actual expected=hand-computed:" + mismatch_detail);

    g_client->destroyBuffer(buf_a);
    g_client->destroyBuffer(buf_b);
    g_client->destroyBuffer(outputs[0]);
    g_client->destroyExecutable(executable);

    g_elements_checked += 4;
    std::cout << "PASS (4 elements checked)" << std::endl;
    return true;
}

// ===== Test 5: negative control — compile() must REFUSE a module whose
// declared function result type does not match the type of the value it
// actually returns. Runs unconditionally once a client exists, with or
// without either program above having exercised every op path: this is the
// evidence that compile() itself has not regressed into a rubber stamp that
// would let every PASS above happen by accident. =====
bool test_pjrt_negative_control_rejects_malformed_module() {
    std::cout << "Test: PJRT Negative Control (reject malformed StableHLO module)... ";

    // %arg0 is tensor<4xf32> and is returned as-is, but the function signature
    // declares the result type tensor<8xf32>. This is invalid StableHLO/MLIR:
    // a func.return's operand types must match the enclosing function's
    // declared result types.
    static const char* const kMalformedModule = R"mlir(
module @eshkol_negative_control {
  func.func public @main(%arg0: tensor<4xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<4xf32>
  }
}
)mlir";

    std::string error;
    PJRT_LoadedExecutable* executable = g_client->compile(kMalformedModule, "mlir", &error);

    TEST_ASSERT(executable == nullptr,
        "compile() must REFUSE a module whose declared function result type "
        "does not match the type of the value it returns — a gate that "
        "cannot fail is worthless, and this is the check that it can");
    TEST_ASSERT(!error.empty(), "a refused compile() must set a non-empty error message");

    if (executable != nullptr) {
        // Defensive: never leak an executable even if the assertion above
        // somehow did not already return.
        g_client->destroyExecutable(executable);
    }

    std::cout << "PASS (refused: \"" << error << "\")" << std::endl;
    return true;
}

// ===== Main Test Runner =====
int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  PJRT Round-Trip Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    auto run_test = [](bool (*test_func)()) {
        if (test_func()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
    };

    // ----- Discovery: exactly the order pjrt_smoke_test uses. No plugin is
    // an expected outcome (exit 77), never a FAIL. Once a plugin IS found,
    // nothing past this point may skip. -----
    std::cout << "Test: PJRT Plugin Discovery (tpu)... ";
    g_plugin_path = findPjrtPlugin("tpu");
    if (g_plugin_path.empty()) {
        std::cout << "SKIP" << std::endl;
        std::cout << "  No PJRT plugin found on this host. Searched, in order:" << std::endl;
        std::cout << searchedLocationsDescription();
        std::cout << "  This is the correct outcome when no accelerator/plugin is present. "
                     "Set ESHKOL_PJRT_PLUGIN_PATH to force a specific plugin." << std::endl;
        return 77;
    }
    std::cout << "PASS (found " << g_plugin_path << ")" << std::endl;
    tests_passed++;

    run_test(test_pjrt_plugin_loads_and_client_creates);
    if (g_client) {
        run_test(test_pjrt_add_roundtrip);
        run_test(test_pjrt_matmul_roundtrip);
        run_test(test_pjrt_negative_control_rejects_malformed_module);
    } else {
        std::cerr << "FAIL: no client available — add/matmul/negative-control tests cannot run" << std::endl;
        tests_failed += 3;
    }

    // Explicit teardown ordering: client before plugin, plugin last — every
    // PJRT handle the client holds belongs to the plugin's shared object.
    g_client.reset();
    g_plugin.reset();

    // Summary
    std::cout << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "  Test Results" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Passed:  " << tests_passed << std::endl;
    std::cout << "Failed:  " << tests_failed << std::endl;
    std::cout << "SUMMARY: platform=\"" << g_platform_name << "\" devices=" << g_device_count
              << " elements_checked=" << g_elements_checked << std::endl;
    std::cout << std::endl;

    if (tests_failed != 0) {
        return 1;
    }
    return 0;
}
