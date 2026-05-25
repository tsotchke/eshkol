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

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: vm_limits_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path cmake_path = source_root / "CMakeLists.txt";
    const fs::path header_path =
        source_root / "inc" / "eshkol" / "backend" / "vm_limits.h";
    if (!fs::exists(header_path)) {
        return fail("vm_limits.h not found under source root");
    }

    const std::string cmake = read_file(cmake_path);
    const std::string header = read_file(header_path);
    bool ok = true;

    ok = ok &&
         expect_contains(cmake, "set(ESHKOL_VM_HEAP_SIZE \"65536\" CACHE STRING",
                         "CMake exposes bytecode VM heap capacity") &&
         expect_contains(cmake, "set(ESHKOL_VM_STACK_SIZE \"4096\" CACHE STRING",
                         "CMake exposes bytecode VM stack capacity") &&
         expect_contains(cmake, "set(ESHKOL_VM_MAX_FRAMES \"256\" CACHE STRING",
                         "CMake exposes bytecode VM frame capacity") &&
         expect_contains(cmake, "set(ESHKOL_VM_MAX_CONSTS \"1024\" CACHE STRING",
                         "CMake exposes bytecode VM constant-pool capacity") &&
         expect_contains(cmake, "set(ESHKOL_VM_MAX_CODE \"100000\" CACHE STRING",
                         "CMake exposes bytecode VM instruction capacity") &&
         expect_contains(cmake, "MATCHES \"^[1-9][0-9]*$\"",
                         "CMake rejects non-positive VM limits") &&
         expect_contains(cmake, "function(eshkol_apply_vm_limit_definitions target_name)",
                         "CMake centralizes VM limit compile definitions") &&
         expect_contains(cmake, "ESHKOL_VM_HEAP_SIZE=${ESHKOL_VM_HEAP_SIZE}",
                         "CMake propagates heap limit to VM targets") &&
         expect_contains(cmake, "ESHKOL_VM_STACK_SIZE=${ESHKOL_VM_STACK_SIZE}",
                         "CMake propagates stack limit to VM targets") &&
         expect_contains(cmake, "ESHKOL_VM_MAX_FRAMES=${ESHKOL_VM_MAX_FRAMES}",
                         "CMake propagates frame limit to VM targets") &&
         expect_contains(cmake, "ESHKOL_VM_MAX_CONSTS=${ESHKOL_VM_MAX_CONSTS}",
                         "CMake propagates const limit to VM targets") &&
         expect_contains(cmake, "ESHKOL_VM_MAX_CODE=${ESHKOL_VM_MAX_CODE}",
                         "CMake propagates code limit to VM targets") &&
         expect_contains(cmake, "eshkol_apply_vm_limit_definitions(eshkol-vm-unity-obj)",
                         "CMake applies VM limits to the unity VM object") &&
         expect_contains(cmake, "eshkol_apply_vm_limit_definitions(eshkol-vm-standalone-test)",
                         "CMake applies VM limits to the standalone VM test") &&
         expect_contains(cmake, "eshkol_apply_vm_limit_definitions(test_vm_c_api)",
                         "CMake applies VM limits to the public VM C API test");

    ok = ok &&
         expect_contains(header, "#define ESHKOL_VM_HEAP_SIZE 65536",
                         "vm_limits.h preserves the heap default") &&
         expect_contains(header, "#define ESHKOL_VM_STACK_SIZE 4096",
                         "vm_limits.h preserves the stack default") &&
         expect_contains(header, "#define ESHKOL_VM_MAX_FRAMES 256",
                         "vm_limits.h preserves the frame default") &&
         expect_contains(header, "#define ESHKOL_VM_MAX_CONSTS 1024",
                         "vm_limits.h preserves the const-pool default") &&
         expect_contains(header, "#define ESHKOL_VM_MAX_CODE 100000",
                         "vm_limits.h preserves the instruction default") &&
         expect_contains(header, "#error \"ESHKOL_VM_HEAP_SIZE must be positive\"",
                         "vm_limits.h validates heap limit") &&
         expect_contains(header, "#error \"ESHKOL_VM_MAX_CODE must be positive\"",
                         "vm_limits.h validates instruction limit") &&
         expect_contains(header, "#define HEAP_SIZE ESHKOL_VM_HEAP_SIZE",
                         "vm_limits.h keeps the legacy heap alias") &&
         expect_contains(header, "#define STACK_SIZE ESHKOL_VM_STACK_SIZE",
                         "vm_limits.h keeps the legacy stack alias") &&
         expect_contains(header, "#define MAX_FRAMES ESHKOL_VM_MAX_FRAMES",
                         "vm_limits.h keeps the legacy frame alias") &&
         expect_contains(header, "#define MAX_CONSTS ESHKOL_VM_MAX_CONSTS",
                         "vm_limits.h keeps the legacy const alias") &&
         expect_contains(header, "#define MAX_CODE ESHKOL_VM_MAX_CODE",
                         "vm_limits.h keeps the legacy instruction alias");

    const std::string vm_core =
        read_file(source_root / "lib" / "backend" / "vm_core.c");
    const std::string vm_parser =
        read_file(source_root / "lib" / "backend" / "vm_parser.c");
    const std::string eshkol_vm =
        read_file(source_root / "lib" / "backend" / "eshkol_vm.c");
    const std::string eskb_reader =
        read_file(source_root / "lib" / "backend" / "eskb_reader.c");

    ok = ok &&
         expect_contains(vm_core, "#include \"eshkol/backend/vm_limits.h\"",
                         "vm_core.c includes VM limits") &&
         expect_contains(vm_parser, "#include \"eshkol/backend/vm_limits.h\"",
                         "vm_parser.c includes VM limits") &&
         expect_contains(eshkol_vm, "#include \"eshkol/backend/vm_limits.h\"",
                         "eshkol_vm.c includes VM limits for unity builds") &&
         expect_contains(eskb_reader, "#include \"eshkol/backend/vm_limits.h\"",
                         "eskb_reader.c includes VM limits");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
