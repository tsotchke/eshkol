#include "eshkol/core/object_limits.h"

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

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool contains(const std::string& source, const char* text) {
    return source.find(text) != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return fail("usage: eshkol_compiler_vector_limit_surface_test <source-root>");

    const size_t limit = ESHKOL_MAX_VECTOR_CAPACITY;
    if (!eshkol_vector_capacity_fits(limit - 1) ||
        eshkol_vector_capacity_fits(limit) ||
        eshkol_vector_capacity_fits(limit + 1)) {
        return fail("hosted vector limit predicate boundary is incorrect");
    }

    const std::string source = read_file(
        fs::path(argv[1]) / "lib" / "backend" / "eshkol_compiler.c");
    if (!contains(source, "#include \"eshkol/core/object_limits.h\"")) {
        return fail("hosted compiler does not include shared object limits");
    }
    if (!contains(source, "!eshkol_vector_capacity_fits((size_t)requested_size)")) {
        return fail("hosted compiler does not use the shared vector predicate");
    }
    if (!contains(source, "make-vector: length is outside the representable vector range")) {
        return fail("hosted compiler does not use the shared vector diagnostic");
    }
    if (contains(source, "if (n > 256) n = 256")) {
        return fail("hosted compiler still clamps vector capacity to 256");
    }

    std::cout << "PASS: hosted vector limit boundary " << (limit - 1)
              << ", " << limit << ", " << (limit + 1) << '\n';
    return 0;
}
