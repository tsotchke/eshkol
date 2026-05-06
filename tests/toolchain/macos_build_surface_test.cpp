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
        return fail("usage: macos_build_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path script_path = source_root / "scripts" / "build-macos.sh";
    if (!fs::exists(script_path)) {
        return fail("build-macos.sh not found under source root");
    }

    const std::string script = read_file(script_path);
    bool ok = true;

    ok = ok &&
         expect_contains(script, "validate_package_version()",
                         "macOS build script declares version validation") &&
         expect_contains(script, "unsafe macOS artifact version",
                         "macOS build script rejects unsafe version text") &&
         expect_contains(script, "validate_target_arch()",
                         "macOS build script declares architecture validation") &&
         expect_contains(script, "validate_target_arch \"$TARGET_ARCH\"",
                         "macOS build script validates the user-supplied architecture") &&
         expect_contains(script, "require_output_directory()",
                         "macOS build script declares output directory validation") &&
         expect_contains(script, "require_output_directory \"macOS artifact output directory\" \"$OUTPUT_DIR\"",
                         "macOS build script validates the output root") &&
         expect_contains(script, "OUTPUT_DIR=\"$(cd \"$OUTPUT_DIR\" && pwd -P)\"",
                         "macOS build script canonicalizes the output directory") &&
         expect_contains(script, "remove_build_directory()",
                         "macOS build script declares guarded clean-build removal") &&
         expect_contains(script, "remove_build_directory \"$build_dir\"",
                         "macOS build script uses the guarded build-directory cleanup") &&
         expect_contains(script, "remove_package_stage()",
                         "macOS build script declares guarded package-stage cleanup") &&
         expect_contains(script, "\"$root\"/.pkg.*",
                         "macOS build script restricts package-stage removal to .pkg staging dirs") &&
         expect_contains(script, "pkg_stage=\"$(mktemp -d \"$pkg_dir/.pkg.XXXXXX\")\"",
                         "macOS build script creates unique per-arch package staging") &&
         expect_contains(script, "pkg_stage=\"$(mktemp -d \"$universal_dir/.pkg.XXXXXX\")\"",
                         "macOS build script creates unique universal package staging") &&
         expect_contains(script, "tar -czvf \"$tarball\" -C \"$pkg_stage\" .",
                         "macOS build script tars from the validated package staging directory") &&
         expect_contains(script, "remove_package_stage \"$pkg_stage\" \"$pkg_dir\"",
                         "macOS build script removes guarded per-arch package staging") &&
         expect_contains(script, "remove_package_stage \"$pkg_stage\" \"$universal_dir\"",
                         "macOS build script removes guarded universal package staging");

    ok = ok &&
         expect_not_contains(script, "rm -rf \"$build_dir\"",
                             "macOS build script should not remove build dirs without the guard helper") &&
         expect_not_contains(script, "tar -czvf \"$tarball\" -C \"$pkg_dir/pkg\" .",
                             "macOS build script should not tar from fixed per-arch pkg staging") &&
         expect_not_contains(script, "tar -czvf \"$tarball\" -C \"$universal_dir/pkg\" .",
                             "macOS build script should not tar from fixed universal pkg staging") &&
         expect_not_contains(script, "rm -rf \"$pkg_dir/pkg\"",
                             "macOS build script should not remove fixed per-arch pkg staging") &&
         expect_not_contains(script, "rm -rf \"$universal_dir/pkg\"",
                             "macOS build script should not remove fixed universal pkg staging");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
