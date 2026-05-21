#include <eshkol/eshkol.h>
#include <eshkol/llvm_backend.h>
#include <eshkol/types/hott_types.h>
#include <eshkol/types/type_checker.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace eshkol::hott;

namespace {

template <typename T>
bool expect_equal(const T& actual, const T& expected, const std::string& label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& label) {
    if (haystack.find(needle) != std::string::npos) {
        return true;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

eshkol_ast_t parse_single(const std::string& source) {
    std::stringstream stream(source);
    return eshkol_parse_next_ast_from_stream(stream);
}

bool test_compiler_fence_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast = parse_single("(compiler-fence seq-cst)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "compiler-fence type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "compiler-fence returns Null") &&
           expect_equal(checker.hasErrors(), false,
                        "compiler-fence ordering operand is accepted");
}

bool test_memory_fence_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast = parse_single("(memory-fence acquire)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "memory-fence type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "memory-fence returns Null") &&
           expect_equal(checker.hasErrors(), false,
                        "memory-fence ordering operand is accepted");
}

bool test_fence_rejects_unsupported_ordering() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast = parse_single("(memory-fence relaxed)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "memory-fence still synthesizes after ordering issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "memory-fence keeps Null result on ordering issue") &&
           expect_equal(checker.hasErrors(), true,
                        "memory-fence records unsupported ordering error");
}

bool test_fence_ir_lowering() {
    eshkol_set_uses_stdlib(0);
    eshkol_set_target(nullptr);

    eshkol_ast_t asts[4] = {
        parse_single("(define (cf) : null (compiler-fence seq-cst))"),
        parse_single("(define (mf) : null (memory-fence acquire))"),
        parse_single("(define (cr) : null (compiler-fence release))"),
        parse_single("(define (ma) : null (memory-fence acq-rel))"),
    };

    LLVMModuleRef module = eshkol_generate_llvm_ir_library(asts, 4, "fence_ops_test");
    if (!module) {
        std::cerr << "FAIL: fence LLVM module generation" << std::endl;
        return false;
    }

    char ir_path[] = "/tmp/eshkol-fence-ir-XXXXXX";
    const int fd = mkstemp(ir_path);
    if (fd == -1) {
        std::cerr << "FAIL: fence temp IR path" << std::endl;
        eshkol_dispose_llvm_module(module);
        return false;
    }
    close(fd);

    bool ok = false;
    if (eshkol_dump_llvm_ir_to_file(module, ir_path) != 0) {
        std::cerr << "FAIL: fence IR dump" << std::endl;
    } else {
        std::ifstream ir_stream(ir_path);
        std::stringstream ir_buffer;
        ir_buffer << ir_stream.rdbuf();
        const std::string ir = ir_buffer.str();

        ok = expect_contains(ir, "fence syncscope(\"singlethread\") seq_cst",
                             "compiler-fence lowers to singlethread seq_cst fence") &&
             expect_contains(ir, "fence acquire",
                             "memory-fence lowers to acquire fence") &&
             expect_contains(ir, "fence syncscope(\"singlethread\") release",
                             "compiler-fence lowers release ordering") &&
             expect_contains(ir, "fence acq_rel",
                             "memory-fence lowers acq-rel ordering") &&
             expect_contains(ir, "cf",
                             "compiler-fence survives in function IR") &&
             expect_contains(ir, "mf",
                             "memory-fence survives in function IR");
    }

    std::remove(ir_path);
    eshkol_dispose_llvm_module(module);
    return ok;
}

}  // namespace

int main() {
    if (!test_compiler_fence_type_synthesis()) {
        return 1;
    }
    if (!test_memory_fence_type_synthesis()) {
        return 1;
    }
    if (!test_fence_rejects_unsupported_ordering()) {
        return 1;
    }
    if (!test_fence_ir_lowering()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
