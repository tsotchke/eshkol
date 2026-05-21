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

bool test_target_intrinsic_integer_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("value32", BuiltinTypes::UInt32);

    eshkol_ast_t ast =
        parse_single("(target-intrinsic u32 \"llvm.bswap\" u32 value32)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "target-intrinsic integer type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt32,
                        "target-intrinsic returns requested integer type") &&
           expect_equal(checker.hasErrors(), false,
                        "target-intrinsic accepts typed integer arguments");
}

bool test_target_intrinsic_null_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast =
        parse_single("(target-intrinsic null \"llvm.trap\")");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "target-intrinsic null type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "target-intrinsic supports explicit Null returns") &&
           expect_equal(checker.hasErrors(), false,
                        "target-intrinsic accepts zero-argument intrinsic forms");
}

bool test_target_intrinsic_rejects_unsupported_return_type() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("value32", BuiltinTypes::UInt32);

    eshkol_ast_t ast =
        parse_single("(target-intrinsic number \"llvm.bswap\" u32 value32)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "target-intrinsic still synthesizes after return type issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::Value,
                        "target-intrinsic falls back to Value on invalid return type") &&
           expect_equal(checker.hasErrors(), true,
                        "target-intrinsic records unsupported return type errors");
}

bool test_target_intrinsic_rejects_pointer_mismatch() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("value32", BuiltinTypes::UInt32);

    eshkol_ast_t ast =
        parse_single("(target-intrinsic u32 \"llvm.bswap\" ptr value32)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "target-intrinsic still synthesizes after pointer mismatch") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt32,
                        "target-intrinsic keeps explicit return type on arg mismatch") &&
           expect_equal(checker.hasErrors(), true,
                        "target-intrinsic records pointer mismatch errors");
}

bool test_target_intrinsic_rejects_non_string_name() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast =
        parse_single("(target-intrinsic u32 llvm.bswap)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "target-intrinsic still synthesizes after name issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt32,
                        "target-intrinsic keeps explicit return type on name issue") &&
           expect_equal(checker.hasErrors(), true,
                        "target-intrinsic records non-string name errors");
}

bool test_target_intrinsic_rejects_unpaired_argument_type() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast =
        parse_single("(target-intrinsic u32 \"llvm.bswap\" u32)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "target-intrinsic still synthesizes after pair issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt32,
                        "target-intrinsic keeps explicit return type on pair issue") &&
           expect_equal(checker.hasErrors(), true,
                        "target-intrinsic records unpaired argument type errors");
}

bool test_target_intrinsic_ir_lowering() {
    eshkol_set_uses_stdlib(0);
    eshkol_set_target(nullptr);

    eshkol_ast_t asts[3] = {
        parse_single("(define (swap32 (x : u32)) : u32 (target-intrinsic u32 \"llvm.bswap\" u32 x))"),
        parse_single("(define (frame) : ptr (target-intrinsic ptr \"llvm.frameaddress\" i32 0))"),
        parse_single("(define (halt) : null (target-intrinsic null \"llvm.trap\"))"),
    };

    LLVMModuleRef module =
        eshkol_generate_llvm_ir_library(asts, 3, "target_intrinsic_test");
    if (!module) {
        std::cerr << "FAIL: target intrinsic LLVM module generation" << std::endl;
        return false;
    }

    char ir_path[] = "/tmp/eshkol-target-intrinsic-ir-XXXXXX";
    const int fd = mkstemp(ir_path);
    if (fd == -1) {
        std::cerr << "FAIL: target intrinsic temp IR path" << std::endl;
        eshkol_dispose_llvm_module(module);
        return false;
    }
    close(fd);

    bool ok = false;
    if (eshkol_dump_llvm_ir_to_file(module, ir_path) != 0) {
        std::cerr << "FAIL: target intrinsic IR dump" << std::endl;
    } else {
        std::ifstream ir_stream(ir_path);
        std::stringstream ir_buffer;
        ir_buffer << ir_stream.rdbuf();
        const std::string ir = ir_buffer.str();

        ok = expect_contains(ir, "@llvm.bswap.i32",
                             "target-intrinsic lowers integer overloads through LLVM intrinsics") &&
             expect_contains(ir, "call i32 @llvm.bswap.i32",
                             "target-intrinsic emits typed integer intrinsic calls") &&
             expect_contains(ir, "@llvm.frameaddress",
                             "target-intrinsic lowers pointer-return intrinsics") &&
             expect_contains(ir, "@llvm.trap",
                             "target-intrinsic lowers void intrinsics") &&
             expect_contains(ir, "swap32",
                             "target intrinsic integer function survives in IR") &&
             expect_contains(ir, "frame",
                             "target intrinsic pointer function survives in IR") &&
             expect_contains(ir, "halt",
                             "target intrinsic null-return function survives in IR");
    }

    std::remove(ir_path);
    eshkol_dispose_llvm_module(module);
    return ok;
}

}  // namespace

int main() {
    if (!test_target_intrinsic_integer_type_synthesis()) {
        return 1;
    }
    if (!test_target_intrinsic_null_type_synthesis()) {
        return 1;
    }
    if (!test_target_intrinsic_rejects_unsupported_return_type()) {
        return 1;
    }
    if (!test_target_intrinsic_rejects_pointer_mismatch()) {
        return 1;
    }
    if (!test_target_intrinsic_rejects_non_string_name()) {
        return 1;
    }
    if (!test_target_intrinsic_rejects_unpaired_argument_type()) {
        return 1;
    }
    if (!test_target_intrinsic_ir_lowering()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
