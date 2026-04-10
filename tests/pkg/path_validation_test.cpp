// Regression test for the path-traversal vulnerability that allowed a
// dependency named "../outside_link" in eshkol.toml to escape the project's
// eshkol_deps/ directory. The fix is in tools/pkg/eshkol_pkg.cpp:
// is_valid_dependency_name() — names must be 1–64 chars of [A-Za-z0-9._-],
// must start with an alphanumeric, and must not be '.', '..', or a Windows
// reserved name.
//
// We exercise the binary end-to-end by writing a hostile manifest, calling
// `eshkol-pkg install`, and asserting that (a) the binary exits non-zero
// and (b) no file appears at the would-be-escaped path.

#include <eshkol/pkg/subprocess.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return fail("expected path to eshkol-pkg binary");

    const fs::path pkg_binary = fs::absolute(argv[1]);
    const fs::path temp_root = fs::temp_directory_path() / "eshkol-pkg-path-validation-test";
    const fs::path project_dir = temp_root / "project";
    const fs::path escaped = project_dir / "outside_link";

    std::error_code ec;
    fs::remove_all(temp_root, ec);
    fs::create_directories(project_dir / "src");

    std::ofstream manifest(project_dir / "eshkol.toml");
    manifest << "[package]\n";
    manifest << "name = \"demo\"\n";
    manifest << "version = \"0.1.0\"\n";
    manifest << "entry = \"src/main.esk\"\n\n";
    manifest << "[dependencies]\n";
    manifest << "../outside_link = \"1.0.0\"\n";
    manifest.close();

    std::ofstream source(project_dir / "src" / "main.esk");
    source << ";; dependency path validation regression\n";
    source.close();

    const int exit_code = eshkol::pkg::run_subprocess(
        {pkg_binary.string(), "install"}, &project_dir);
    if (exit_code == 0) {
        return fail("eshkol-pkg install accepted an invalid dependency name");
    }

    if (fs::exists(escaped)) {
        return fail("invalid dependency name escaped the eshkol_deps directory");
    }

    std::cout << "PASS" << std::endl;
    fs::remove_all(temp_root, ec);
    return 0;
}
