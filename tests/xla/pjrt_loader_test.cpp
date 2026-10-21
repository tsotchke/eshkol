#include "eshkol/backend/xla/pjrt_client.h"

extern "C" {
#include "pjrt_c_api.h"
}

#include <iostream>
#include <string>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return fail("usage: pjrt_loader_test CASE LIBRARY");

    const std::string test_case = argv[1];
    const std::string library = argv[2];
    std::string error;
    auto plugin = eshkol::xla::PjrtPlugin::load(library, &error);
    if (test_case == "compatible") {
        if (!plugin) return fail("compatible table rejected: " + error);
        if (plugin->apiMajorVersion() != 0) {
            return fail("compatible table reported an unexpected major version");
        }
        if (plugin->api()->extension_start == nullptr) {
            return fail("PJRT_Plugin_Initialize did not run");
        }
        plugin.reset();  // Also exercises the normal dlclose path.
        std::cout << "PASS: compatible PJRT table initialized, loaded, and unloaded\n";
        return 0;
    }
    if (plugin) return fail(test_case + " unexpectedly loaded a plugin");
    if (error.empty()) return fail(test_case + " returned no diagnostic");

    const std::string expected = test_case == "missing"
        ? "dlopen(" : test_case == "nonplugin"
        ? "does not export GetPjrtApi" : test_case == "incompatible"
        ? "reports PJRT API major version" : test_case == "short"
        ? "PJRT_Api table too short" : "null required function PJRT_Plugin_Initialize";
    if (error.find(expected) == std::string::npos) {
        return fail(test_case + " diagnostic did not contain '" + expected +
                    "': " + error);
    }

    std::cout << "PASS: " << test_case << " rejected with diagnostic: "
              << error << '\n';
    return 0;
}
