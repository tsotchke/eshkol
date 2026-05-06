#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& label) {
    if (haystack.find(needle) == std::string::npos) {
        std::cerr << "missing: " << label << "\nneedle: " << needle << std::endl;
        return false;
    }
    return true;
}

bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& label) {
    if (haystack.find(needle) != std::string::npos) {
        std::cerr << "unexpected: " << label << "\nneedle: " << needle
                  << std::endl;
        return false;
    }
    return true;
}

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: debian_package_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path script_path = source_root / "scripts" / "build-deb.sh";
    if (!fs::exists(script_path)) {
        return fail("build-deb.sh not found under source root");
    }

    const std::string script = read_file(script_path);
    bool ok = true;

    ok = ok &&
         expect_contains(script, "validate_package_version()",
                         "Debian package script declares version validation") &&
         expect_contains(script, "unsafe Debian package version",
                         "Debian package script rejects unsafe version text") &&
         expect_contains(script, "validate_package_version \"$VERSION\"",
                         "Debian package script validates the user-supplied version") &&
         expect_contains(script, "require_regular_executable()",
                         "Debian package script declares executable validation") &&
         expect_contains(script, "if ! test -e \"$ESHKOL_RUN\"; then",
                         "Debian package script only builds when eshkol-run is absent") &&
         expect_contains(script, "require_regular_executable \"eshkol-run\" \"$ESHKOL_RUN\"",
                         "Debian package script validates the built compiler executable") &&
         expect_contains(script, "require_regular_package_file()",
                         "Debian package script declares package artifact validation") &&
         expect_contains(script, "require_regular_package_file \"$DEB_FILE\"",
                         "Debian package script validates the generated package artifact") &&
         expect_contains(script, "mv -- \"$DEB_FILE\" \"../${OUTPUT_NAME}\"",
                         "Debian package script moves the generated package with option termination");

    ok = ok &&
         expect_not_contains(script, "[ ! -f \"${BUILD_DIR}/eshkol-run\" ]",
                             "Debian package script should not use a plain compiler file probe") &&
         expect_not_contains(script, "mv \"$DEB_FILE\"",
                             "Debian package script should not move the package without option termination");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
