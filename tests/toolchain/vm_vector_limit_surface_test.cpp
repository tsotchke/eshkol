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
    if (argc != 2) return fail("usage: vm_vector_limit_surface_test <source-root>");

    const size_t limit = ESHKOL_MAX_VECTOR_CAPACITY;
    if (!eshkol_vector_capacity_fits(limit - 1) ||
        eshkol_vector_capacity_fits(limit) ||
        eshkol_vector_capacity_fits(limit + 1)) {
        return fail("shared vector limit predicate boundary is incorrect");
    }

    const std::string source = read_file(
        fs::path(argv[1]) / "lib" / "backend" / "vm_native.c");
    if (!contains(source, "requested_size >= (double)ESHKOL_MAX_VECTOR_CAPACITY")) {
        return fail("VM make-vector does not enforce the shared vector capacity limit");
    }
    if (!contains(source, "make-vector: size is outside the representable range")) {
        return fail("VM make-vector does not use the shared vector diagnostic");
    }


    std::cout << "PASS: VM vector limit boundary " << (limit - 1)
              << ", " << limit << ", " << (limit + 1) << '\n';
    return 0;
}
