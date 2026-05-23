#include "eshkol/eshkol.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <setjmp.h>
#include <string>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: forward_ref_scan_boundary_test <build-dir>");
    }

    const std::filesystem::path original_cwd = std::filesystem::current_path();
    const std::filesystem::path fixture_root =
        std::filesystem::path(argv[1]) / "forward_ref_scan_fixture";
    const std::filesystem::path generated_provider =
        fixture_root / "build-codex-verify" / "generated" / "provider.esk";

    std::error_code ec;
    std::filesystem::remove_all(fixture_root, ec);
    std::filesystem::create_directories(generated_provider.parent_path(), ec);
    if (ec) {
        return fail("failed to create scan fixture: " + ec.message());
    }

    {
        std::ofstream output(generated_provider);
        output << "(provide missing-forward-ref)\n"
               << "(define (missing-forward-ref x) x)\n";
        if (!output) {
            return fail("failed to write generated provider fixture");
        }
    }

    std::filesystem::current_path(fixture_root);
    g_current_exception = nullptr;

    jmp_buf jump;
    if (setjmp(jump) == 0) {
        void* unresolved = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
        eshkol_push_exception_handler(&jump);
        (void)eshkol_check_forward_ref(unresolved, unresolved, "missing-forward-ref");
        eshkol_pop_exception_handler();
        std::filesystem::current_path(original_cwd);
        std::filesystem::remove_all(fixture_root, ec);
        return fail("forward-reference check returned without raising");
    }

    eshkol_pop_exception_handler();
    std::filesystem::current_path(original_cwd);
    std::filesystem::remove_all(fixture_root, ec);

    if (!g_current_exception || !g_current_exception->message) {
        return fail("forward-reference check raised without an exception message");
    }

    const std::string message = g_current_exception->message;
    if (message.find("forward-referenced") == std::string::npos) {
        return fail("diagnostic missing forward-reference text: " + message);
    }
    if (message.find("Likely missing") != std::string::npos ||
        message.find("provider.esk") != std::string::npos) {
        return fail("scanner used generated build-tree provider hint: " + message);
    }

    return 0;
}
