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
        return fail("usage: docker_build_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path script_path = source_root / "scripts" / "build-docker.sh";
    if (!fs::exists(script_path)) {
        return fail("build-docker.sh not found under source root");
    }

    const std::string script = read_file(script_path);
    bool ok = true;

    ok = ok &&
         expect_contains(script, "validate_package_version()",
                         "Docker build script declares version validation") &&
         expect_contains(script, "unsafe Docker artifact version",
                         "Docker build script rejects unsafe version text") &&
         expect_contains(script, "validate_package_version \"$VERSION\"",
                         "Docker build script validates the user-supplied version") &&
         expect_contains(script, "require_output_directory()",
                         "Docker build script declares output directory validation") &&
         expect_contains(script, "require_output_directory \"Docker artifact output directory\" \"$OUTPUT_DIR\"",
                         "Docker build script validates the output root") &&
         expect_contains(script, "require_output_directory \"Docker architecture artifact directory\" \"$ARCH_OUTPUT\"",
                         "Docker build script validates the architecture output directory") &&
         expect_contains(script, "ARCH_OUTPUT=\"$(cd \"$ARCH_OUTPUT\" && pwd -P)\"",
                         "Docker build script canonicalizes the architecture output directory") &&
         expect_contains(script, "remove_package_stage()",
                         "Docker build script declares guarded package-stage cleanup") &&
         expect_contains(script, "\"$root\"/.pkg.*",
                         "Docker build script restricts package-stage removal to .pkg staging dirs") &&
         expect_contains(script, "PKG_STAGE=\"$(mktemp -d \"$ARCH_OUTPUT/.pkg.XXXXXX\")\"",
                         "Docker build script creates a unique package staging directory") &&
         expect_contains(script, "tar -czvf \"$ARCH_OUTPUT/$TARBALL_NAME\" -C \"$PKG_STAGE\" .",
                         "Docker build script tars from the validated package staging directory") &&
         expect_contains(script, "remove_package_stage \"$PKG_STAGE\" \"$ARCH_OUTPUT\"",
                         "Docker build script removes only the guarded package staging directory");

    ok = ok &&
         expect_not_contains(script, "mkdir -p pkg/bin pkg/lib pkg/share/eshkol",
                             "Docker build script should not stage into a fixed pkg directory") &&
         expect_not_contains(script, "tar -czvf \"$TARBALL_NAME\" -C pkg .",
                             "Docker build script should not tar from a fixed pkg directory") &&
         expect_not_contains(script, "rm -rf pkg",
                             "Docker build script should not remove a fixed pkg directory");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
