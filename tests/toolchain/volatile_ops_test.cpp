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

bool test_volatile_load_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("uart-dr", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(volatile-load u8 uart-dr)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "volatile-load type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt8,
                        "volatile-load returns requested machine type") &&
           expect_equal(checker.hasErrors(), false, "volatile-load pointer operand is accepted");
}

bool test_volatile_load_usize_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("uart-dr", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(volatile-load usize uart-dr)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "volatile-load usize synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::USize,
                        "volatile-load usize returns usize") &&
           expect_equal(checker.hasErrors(), false, "volatile-load usize operand is accepted");
}

bool test_volatile_store_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("uart-dr", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(volatile-store! u16 uart-dr 255)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "volatile-store! type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "volatile-store! returns Null") &&
           expect_equal(checker.hasErrors(), false, "volatile-store! integer value is accepted");
}

bool test_volatile_pointer_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("uart-dr", BuiltinTypes::Pointer);
    checker.context().bind("next-base", BuiltinTypes::Pointer);

    eshkol_ast_t load_ast = parse_single("(volatile-load ptr uart-dr)");
    eshkol_ast_t store_ast = parse_single("(volatile-store! ptr uart-dr next-base)");
    const TypeCheckResult load_result = checker.synthesize(&load_ast);
    const TypeCheckResult store_result = checker.synthesize(&store_ast);

    return expect_equal(load_result.success, true, "volatile-load ptr synthesis succeeds") &&
           expect_equal(load_result.inferred_type, BuiltinTypes::Pointer,
                        "volatile-load ptr returns Ptr") &&
           expect_equal(store_result.success, true, "volatile-store! ptr synthesis succeeds") &&
           expect_equal(store_result.inferred_type, BuiltinTypes::Null,
                        "volatile-store! ptr returns Null") &&
           expect_equal(checker.hasErrors(), false, "volatile ptr operands are accepted");
}

bool test_volatile_load_rejects_unsupported_type() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("uart-dr", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(volatile-load number uart-dr)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "volatile-load still synthesizes after type issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::Value,
                        "volatile-load falls back to Value on invalid machine type") &&
           expect_equal(checker.hasErrors(), true, "volatile-load records unsupported type error");
}

bool test_volatile_store_rejects_non_pointer_target() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("not-a-pointer", BuiltinTypes::UInt64);

    eshkol_ast_t ast = parse_single("(volatile-store! u8 not-a-pointer 1)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "volatile-store! still synthesizes after target issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "volatile-store! keeps Null result on target issue") &&
           expect_equal(checker.hasErrors(), true,
                        "volatile-store! records pointer target error");
}

bool test_volatile_ir_lowering() {
    eshkol_set_uses_stdlib(0);
    eshkol_set_target(nullptr);

    eshkol_ast_t asts[6] = {
        parse_single("(define mmio-base (usize->ptr 4096))"),
        parse_single("(define next-base (usize->ptr 8192))"),
        parse_single("(define (peek) : u8 (volatile-load u8 mmio-base))"),
        parse_single("(define (poke (v : u8)) : null (volatile-store! u8 mmio-base v))"),
        parse_single("(define (peek-next) : ptr (volatile-load ptr mmio-base))"),
        parse_single("(define (poke-next) : null (volatile-store! ptr mmio-base next-base))"),
    };

    LLVMModuleRef module = eshkol_generate_llvm_ir_library(asts, 6, "volatile_ops_test");
    if (!module) {
        std::cerr << "FAIL: volatile LLVM module generation" << std::endl;
        return false;
    }

    char ir_path[] = "/tmp/eshkol-volatile-ir-XXXXXX";
    const int fd = mkstemp(ir_path);
    if (fd == -1) {
        std::cerr << "FAIL: volatile temp IR path" << std::endl;
        eshkol_dispose_llvm_module(module);
        return false;
    }
    close(fd);

    bool ok = false;
    if (eshkol_dump_llvm_ir_to_file(module, ir_path) != 0) {
        std::cerr << "FAIL: volatile IR dump" << std::endl;
    } else {
        std::ifstream ir_stream(ir_path);
        std::stringstream ir_buffer;
        ir_buffer << ir_stream.rdbuf();
        const std::string ir = ir_buffer.str();

        ok = expect_contains(ir, "load volatile i8",
                             "volatile-load lowers to volatile i8 load") &&
             expect_contains(ir, "store volatile i8",
                             "volatile-store! lowers to volatile i8 store") &&
             expect_contains(ir, "load volatile ptr",
                             "volatile-load ptr lowers to volatile ptr load") &&
             expect_contains(ir, "store volatile ptr",
                             "volatile-store! ptr lowers to volatile ptr store") &&
             expect_contains(ir, "mmio-base",
                             "volatile IR retains global pointer binding") &&
             expect_contains(ir, "next-base",
                             "volatile IR retains pointer payload binding") &&
             expect_contains(ir, "peek", "volatile load survives in function IR") &&
             expect_contains(ir, "poke", "volatile store survives in function IR") &&
             expect_contains(ir, "peek-next",
                             "volatile pointer load survives in function IR") &&
             expect_contains(ir, "poke-next",
                             "volatile pointer store survives in function IR");
    }

    std::remove(ir_path);
    eshkol_dispose_llvm_module(module);
    return ok;
}

}  // namespace

int main() {
    if (!test_volatile_load_type_synthesis()) {
        return 1;
    }
    if (!test_volatile_load_usize_type_synthesis()) {
        return 1;
    }
    if (!test_volatile_store_type_synthesis()) {
        return 1;
    }
    if (!test_volatile_pointer_type_synthesis()) {
        return 1;
    }
    if (!test_volatile_load_rejects_unsupported_type()) {
        return 1;
    }
    if (!test_volatile_store_rejects_non_pointer_target()) {
        return 1;
    }
    if (!test_volatile_ir_lowering()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
