/*
 * PJRT Smoke Test
 *
 * Proves the PJRT device-runtime path (inc/eshkol/backend/xla/pjrt_client.h,
 * lib/backend/xla/pjrt_client.cpp) is real: it finds a plugin, loads it,
 * creates a client, and enumerates devices on whatever accelerator this host
 * actually has. On a host with no accelerator plugin at all, discovery SKIPs
 * (exit 77) — that is the correct outcome for "no hardware here", never a
 * silent no-op. Once a plugin IS found, nothing past that point may skip:
 * every subsequent step must produce a real PASS or FAIL against real
 * hardware, because "plugin present but broken" is a genuine defect.
 *
 * A negative control (test 6) runs unconditionally, with or without an
 * accelerator present: it feeds the loader something that is definitely not
 * a PJRT plugin and asserts the loader rejects it for the right reason
 * (GetPjrtApi is absent), not some unrelated reason. Without this, a build
 * where PjrtPlugin::load() has quietly regressed into a no-op could still
 * report a clean PASS/SKIP split on every stage above.
 *
 * Only builds when ESHKOL_XLA_ENABLED=ON (see CMakeLists.txt).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

// XLA headers
#include "eshkol/backend/xla/pjrt_client.h"

using eshkol::xla::findPjrtPlugin;
using eshkol::xla::PjrtClient;
using eshkol::xla::PjrtDeviceInfo;
using eshkol::xla::PjrtPlugin;

// Test utilities — same idiom as tests/xla/xla_codegen_test.cpp.
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// Test counters
static int tests_passed = 0;
static int tests_failed = 0;

// State shared between the discovery/load/create/query steps, mirroring
// g_test_arena's role in xla_codegen_test.cpp: one real resource created
// once, used by the steps that depend on it.
static std::string g_plugin_path;
static std::unique_ptr<PjrtPlugin> g_plugin;
static std::unique_ptr<PjrtClient> g_client;

// The exact search order findPjrtPlugin("tpu") uses (pjrt_client.cpp), kept
// in sync here purely for the SKIP message — this test does not re-implement
// discovery, it only needs to tell a human what was searched.
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

// ===== Test 1 (discovery) is handled specially in main(): finding no
// plugin is a SKIP, not a FAIL, and gates whether tests 2-5 run at all. =====

// ===== Test 2: PjrtPlugin::load() succeeds with a sane API version =====
bool test_pjrt_plugin_loads() {
    std::cout << "Test: PJRT Plugin Load... ";

    std::string error;
    g_plugin = PjrtPlugin::load(g_plugin_path, &error);
    TEST_ASSERT(g_plugin != nullptr,
        "PjrtPlugin::load(" + g_plugin_path + ") must succeed for a plugin "
        "findPjrtPlugin() itself reported as present: " + error);
    TEST_ASSERT(g_plugin->api() != nullptr, "loaded plugin must expose a non-null PJRT_Api table");

    // "Sane" means the version PjrtPlugin::load() accepted and recorded is a
    // real, non-negative version pair — load() itself already refused any
    // major-version mismatch against what this build was compiled against
    // (see PjrtPlugin::load in pjrt_client.cpp), so reaching this line is
    // already evidence the version negotiation happened for real.
    TEST_ASSERT(g_plugin->apiMajorVersion() >= 0, "API major version must be non-negative");
    TEST_ASSERT(g_plugin->apiMinorVersion() >= 0, "API minor version must be non-negative");
    TEST_ASSERT(g_plugin->path() == g_plugin_path, "plugin must record the path it was loaded from");

    std::cout << "PASS (path=" << g_plugin->path()
              << ", api=" << g_plugin->apiMajorVersion() << "." << g_plugin->apiMinorVersion()
              << ")" << std::endl;
    return true;
}

// ===== Test 3: PjrtClient::create() succeeds =====
bool test_pjrt_client_creates() {
    std::cout << "Test: PJRT Client Create... ";

    TEST_ASSERT(g_plugin != nullptr, "a loaded plugin is required to create a client");

    std::string error;
    g_client = PjrtClient::create(g_plugin.get(), &error);
    TEST_ASSERT(g_client != nullptr, "PjrtClient::create() must succeed against a loaded plugin: " + error);

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Test 4: platformName() is non-empty =====
bool test_pjrt_platform_name() {
    std::cout << "Test: PJRT Platform Name... ";

    TEST_ASSERT(g_client != nullptr, "a created client is required to query platformName()");

    std::string name = g_client->platformName();
    TEST_ASSERT(!name.empty(), "platformName() must be non-empty on a real client");

    std::cout << "PASS (platform=\"" << name << "\")" << std::endl;
    return true;
}

// ===== Test 5: devices() and addressableDevices() are non-empty =====
bool test_pjrt_devices_enumerate() {
    std::cout << "Test: PJRT Device Enumeration... ";

    TEST_ASSERT(g_client != nullptr, "a created client is required to enumerate devices");

    const std::vector<PjrtDeviceInfo>& devices = g_client->devices();
    TEST_ASSERT(!devices.empty(), "devices() must be non-empty on a real client with a real plugin");

    std::vector<PjrtDeviceInfo> addressable = g_client->addressableDevices();
    TEST_ASSERT(!addressable.empty(),
        "addressableDevices() must be non-empty — a client this process cannot "
        "place work on at all is not a usable PJRT connection");

    std::cout << "PASS (" << devices.size() << " device(s), "
              << addressable.size() << " addressable, first kind=\""
              << (devices.front().kind.empty() ? "<empty>" : devices.front().kind)
              << "\")" << std::endl;
    return true;
}

// ===== Test 6: negative control — loading something that is definitely NOT
// a PJRT plugin must fail, and must fail because GetPjrtApi is absent, not
// for some unrelated reason. Runs unconditionally: with or without real
// accelerator hardware on this host, this is the evidence that the loader
// itself has not regressed into a no-op that would let every stage above
// pass or SKIP vacuously. =====

// A handful of ordinary, near-universally-present system shared libraries
// that are NOT PJRT plugins. Any one of these dlopen()s successfully (unlike
// a plain executable or a file of random bytes, which fail at dlopen() for a
// format reason that has nothing to do with GetPjrtApi), so PjrtPlugin::load
// reaches its dlsym(..., "GetPjrtApi") check and fails specifically there —
// the one failure mode this negative control exists to prove.
static bool findRealNonPluginLibrary(std::string* out_path) {
    static const char* const kCandidates[] = {
        // Linux (glibc), common multiarch and non-multiarch layouts
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/usr/lib/x86_64-linux-gnu/libc.so.6",
        "/lib/aarch64-linux-gnu/libc.so.6",
        "/usr/lib/aarch64-linux-gnu/libc.so.6",
        "/lib64/libc.so.6",
        "/lib/libc.so.6",
        "/usr/lib/libc.so.6",
        "/lib/x86_64-linux-gnu/libm.so.6",
        "/usr/lib/x86_64-linux-gnu/libm.so.6",
        // macOS
        "/usr/lib/libSystem.B.dylib",
        "/usr/lib/libz.1.dylib",
    };
    if (const char* override_path = std::getenv("ESHKOL_XLA_TEST_NOT_A_PLUGIN_PATH")) {
        if (override_path[0] != '\0' && ::access(override_path, R_OK) == 0) {
            *out_path = override_path;
            return true;
        }
    }
    for (const char* candidate : kCandidates) {
        if (::access(candidate, R_OK) == 0) {
            *out_path = candidate;
            return true;
        }
    }
    return false;
}

// Durable fallback location if no real system library can be found at all
// (an exotic host). NEVER /tmp or /private/tmp: derived from this source
// file's own path, which puts it under the repo's tests/xla/ directory tree
// regardless of where the build runs from, or overridden explicitly by
// ESHKOL_XLA_GATE_SCRATCH_DIR (scripts/run_xla_gate.sh sets this to a path
// under <repo>/.scratch/).
static std::string durableScratchDir() {
    if (const char* env_dir = std::getenv("ESHKOL_XLA_GATE_SCRATCH_DIR")) {
        if (env_dir[0] != '\0') return std::string(env_dir);
    }
    std::string here = __FILE__;
    const std::string marker = "/tests/xla/pjrt_smoke_test.cpp";
    auto pos = here.rfind(marker);
    std::string repo_root = (pos != std::string::npos) ? here.substr(0, pos) : ".";
    return repo_root + "/.scratch/pjrt_smoke_test";
}

bool test_pjrt_rejects_non_plugin() {
    std::cout << "Test: PJRT Negative Control (reject non-plugin)... ";

    std::string candidate;
    bool have_real_library = findRealNonPluginLibrary(&candidate);
    std::string cleanup_path;

    if (!have_real_library) {
        // Fall back to a file of random bytes under a durable scratch dir.
        // This still proves load() rejects garbage, but — because such a
        // file does not even dlopen() as a shared object — the failure
        // happens before GetPjrtApi is ever looked up, so the message will
        // not mention it. That is a strictly weaker negative control, which
        // is exactly why it is only the fallback.
        std::string dir = durableScratchDir();
        // mkdir -p equivalent, one level at a time, tolerant of EEXIST.
        std::string accum;
        for (size_t i = 1; i < dir.size(); ++i) {
            if (dir[i] == '/') {
                accum = dir.substr(0, i);
                ::mkdir(accum.c_str(), 0755);
            }
        }
        ::mkdir(dir.c_str(), 0755);
        candidate = dir + "/not_a_pjrt_plugin.bin";
        cleanup_path = candidate;

        std::ofstream out(candidate, std::ios::binary | std::ios::trunc);
        TEST_ASSERT(out.good(), "must be able to create the negative-control scratch file at " + candidate);
        std::mt19937 rng(0xE58C0L);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        for (int i = 0; i < 4096; ++i) {
            char b = static_cast<char>(byte_dist(rng));
            out.write(&b, 1);
        }
        out.close();
        TEST_ASSERT(out.good(), "must be able to write the negative-control scratch file at " + candidate);
        std::cerr << std::endl
                  << "  NOTE: no known system shared library found (searched libc/libz on "
                     "Linux and macOS); falling back to a random-bytes file at "
                  << candidate << ". This still proves load() rejects garbage, but not "
                     "specifically via the GetPjrtApi check." << std::endl;
    }

    std::string error;
    std::unique_ptr<PjrtPlugin> bogus = PjrtPlugin::load(candidate, &error);

    TEST_ASSERT(bogus == nullptr,
        "PjrtPlugin::load(" + candidate + ") must fail — this is not a PJRT plugin");
    TEST_ASSERT(!error.empty(), "a failed load() must set a non-empty error message");

    if (have_real_library) {
        TEST_ASSERT(error.find("GetPjrtApi") != std::string::npos,
            "loading a real (non-PJRT) shared library must fail specifically because "
            "it does not export GetPjrtApi — got instead: \"" + error + "\". If this "
            "assertion fails, the loader is not actually checking for the plugin "
            "symbol and this whole test file could pass on a build where PJRT "
            "loading is broken end-to-end.");
    }

    if (!cleanup_path.empty()) {
        ::unlink(cleanup_path.c_str());
    }

    std::cout << "PASS (" << (have_real_library ? "real library " : "random-bytes fallback ")
              << "rejected: \"" << error << "\")" << std::endl;
    return true;
}

// ===== Main Test Runner =====
int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  PJRT Smoke Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    auto run_test = [](bool (*test_func)()) {
        if (test_func()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
    };

    // ----- Step 1: discovery. SKIP (not FAIL) is correct here, and ONLY
    // here — no accelerator plugin on this host is an expected outcome, not
    // a defect. -----
    bool hardware_skipped = false;
    std::cout << "Test: PJRT Plugin Discovery (tpu)... ";
    g_plugin_path = findPjrtPlugin("tpu");
    if (g_plugin_path.empty()) {
        hardware_skipped = true;
        std::cout << "SKIP" << std::endl;
        std::cout << "  No TPU PJRT plugin found on this host. Searched, in order:" << std::endl;
        std::cout << searchedLocationsDescription();
        std::cout << "  This is the correct outcome when no accelerator/plugin is present. "
                     "Set ESHKOL_PJRT_PLUGIN_PATH to force a specific plugin." << std::endl;
    } else {
        std::cout << "PASS (found " << g_plugin_path << ")" << std::endl;
        tests_passed++;

        // ----- Steps 2-5: a plugin WAS found. From here on, nothing may
        // SKIP — a plugin present but broken is a genuine defect and must
        // FAIL loudly. -----
        run_test(test_pjrt_plugin_loads);
        if (g_plugin) {
            run_test(test_pjrt_client_creates);
            if (g_client) {
                run_test(test_pjrt_platform_name);
                run_test(test_pjrt_devices_enumerate);
            }
        }
    }

    // ----- Step 6: negative control. Always runs. -----
    run_test(test_pjrt_rejects_non_plugin);

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
    std::cout << "Skipped: " << (hardware_skipped ? "hardware discovery (no TPU plugin on this host)" : "none") << std::endl;
    std::cout << std::endl;

    if (tests_failed != 0) {
        return 1;
    }
    if (hardware_skipped) {
        // Convention: 77 signals a deliberate, expected skip (see
        // scripts/run_wasm_differential.sh and scripts/lib/harness_outcome.sh),
        // distinct from both a clean pass (0) and a real failure (1).
        return 77;
    }
    return 0;
}
