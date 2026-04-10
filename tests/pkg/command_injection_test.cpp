// Regression test for the command-injection vulnerability that allowed a
// dependency name like  $(touch /tmp/owned)  in eshkol.toml to execute via
// std::system() during `eshkol-pkg build`. The fix is in
// tools/pkg/eshkol_pkg.cpp / inc/eshkol/pkg/subprocess.h: every subprocess
// is now launched via fork+execvp (POSIX) or CreateProcessW with structured
// argv (Windows), so manifest data is treated literally.
//
// We exercise the binary end-to-end by writing a manifest with shell
// metacharacters in the package name, calling `eshkol-pkg build`, and
// asserting that the marker file the metacharacters would have created
// does NOT appear.

#include <eshkol/pkg/subprocess.h>

#include <cstdlib>
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

std::string fake_compiler_path(const fs::path& dir) {
#ifdef _WIN32
    fs::path compiler = dir / "fake-compiler.bat";
    std::ofstream out(compiler);
    out << "@echo off\r\n";
    out << "exit /b 0\r\n";
    return compiler.string();
#else
    fs::path compiler = dir / "fake-compiler.sh";
    std::ofstream out(compiler);
    out << "#!/bin/sh\n";
    out << "exit 0\n";
    out.close();
    fs::permissions(compiler,
                    fs::perms::owner_exec | fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::replace);
    return compiler.string();
#endif
}

void set_compiler_env(const std::string& compiler) {
#ifdef _WIN32
    _putenv_s("ESHKOL_COMPILER", compiler.c_str());
#else
    setenv("ESHKOL_COMPILER", compiler.c_str(), 1);
#endif
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return fail("expected path to eshkol-pkg binary");

    const fs::path pkg_binary = fs::absolute(argv[1]);
    const fs::path temp_root = fs::temp_directory_path() / "eshkol-pkg-command-injection-test";
    const fs::path project_dir = temp_root / "project";
    const fs::path marker = temp_root / "marker";

    std::error_code ec;
    fs::remove_all(temp_root, ec);
    fs::create_directories(project_dir / "src");

    std::ofstream manifest(project_dir / "eshkol.toml");
    manifest << "[package]\n";
#ifdef _WIN32
    manifest << "name = \"%TEMP%\"\n";
#else
    manifest << "name = \"$(touch " << marker.string() << ")\"\n";
#endif
    manifest << "version = \"0.1.0\"\n";
    manifest << "entry = \"src/main.esk\"\n";
    manifest.close();

    std::ofstream source(project_dir / "src" / "main.esk");
    source << ";; command injection regression\n";
    source.close();

    const std::string compiler = fake_compiler_path(temp_root);
    set_compiler_env(compiler);

    const int exit_code = eshkol::pkg::run_subprocess(
        {pkg_binary.string(), "build"}, &project_dir);
    if (exit_code != 0) {
        return fail("eshkol-pkg build failed with exit code " + std::to_string(exit_code));
    }

#ifndef _WIN32
    if (fs::exists(marker)) {
        return fail("shell metacharacters in manifest data were executed");
    }
#endif

    std::cout << "PASS" << std::endl;
    fs::remove_all(temp_root, ec);
    return 0;
}
