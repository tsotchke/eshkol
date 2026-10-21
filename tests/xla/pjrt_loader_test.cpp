#include "eshkol/backend/xla/pjrt_client.h"

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
    if (plugin) return fail(test_case + " unexpectedly loaded a plugin");
    if (error.empty()) return fail(test_case + " returned no diagnostic");

    const std::string expected = test_case == "missing"
        ? "dlopen(" : test_case == "nonplugin"
        ? "does not export GetPjrtApi" : "reports PJRT API major version";
    if (error.find(expected) == std::string::npos) {
        return fail(test_case + " diagnostic did not contain '" + expected +
                    "': " + error);
    }

    std::cout << "PASS: " << test_case << " rejected with diagnostic: "
              << error << '\n';
    return 0;
}
