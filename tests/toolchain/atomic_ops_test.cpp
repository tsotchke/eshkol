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

bool test_atomic_load_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-load u8 mmio-base acquire)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "atomic-load type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt8,
                        "atomic-load returns requested machine type") &&
           expect_equal(checker.hasErrors(), false, "atomic-load pointer/order operands accepted");
}

bool test_atomic_store_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-store! u16 mmio-base 255 release)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "atomic-store! type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "atomic-store! returns Null") &&
           expect_equal(checker.hasErrors(), false,
                        "atomic-store! pointer/value/order operands accepted");
}

bool test_atomic_exchange_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-exchange! u32 mmio-base 7 acq-rel)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "atomic-exchange! type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt32,
                        "atomic-exchange! returns previous value type") &&
           expect_equal(checker.hasErrors(), false,
                        "atomic-exchange! pointer/value/order operands accepted");
}

bool test_atomic_fetch_add_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-fetch-add! u32 mmio-base 4 acq-rel)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true, "atomic-fetch-add! type synthesis succeeds") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt32,
                        "atomic-fetch-add! returns previous value type") &&
           expect_equal(checker.hasErrors(), false,
                        "atomic-fetch-add! pointer/value/order operands accepted");
}

bool test_atomic_pointer_type_synthesis() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);
    checker.context().bind("next-base", BuiltinTypes::Pointer);

    eshkol_ast_t load_ast = parse_single("(atomic-load ptr mmio-base seq-cst)");
    eshkol_ast_t store_ast = parse_single("(atomic-store! ptr mmio-base next-base seq-cst)");
    const TypeCheckResult load_result = checker.synthesize(&load_ast);
    const TypeCheckResult store_result = checker.synthesize(&store_ast);

    return expect_equal(load_result.success, true, "atomic-load ptr synthesis succeeds") &&
           expect_equal(load_result.inferred_type, BuiltinTypes::Pointer,
                        "atomic-load ptr returns Ptr") &&
           expect_equal(store_result.success, true, "atomic-store! ptr synthesis succeeds") &&
           expect_equal(store_result.inferred_type, BuiltinTypes::Null,
                        "atomic-store! ptr returns Null") &&
           expect_equal(checker.hasErrors(), false, "atomic ptr operands are accepted");
}

bool test_atomic_exchange_rejects_invalid_ordering() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-exchange! u8 mmio-base 1 consume)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "atomic-exchange! still synthesizes after ordering issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt8,
                        "atomic-exchange! keeps requested type after ordering issue") &&
           expect_equal(checker.hasErrors(), true,
                        "atomic-exchange! records invalid ordering");
}

bool test_atomic_fetch_add_rejects_pointer_type() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);
    checker.context().bind("next-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-fetch-add! ptr mmio-base next-base acq-rel)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "atomic-fetch-add! still synthesizes after type issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::Pointer,
                        "atomic-fetch-add! keeps requested type after type issue") &&
           expect_equal(checker.hasErrors(), true,
                        "atomic-fetch-add! records pointer designator error");
}

bool test_atomic_load_rejects_store_only_ordering() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-load u8 mmio-base release)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "atomic-load still synthesizes after ordering issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::UInt8,
                        "atomic-load keeps requested type after ordering issue") &&
           expect_equal(checker.hasErrors(), true,
                        "atomic-load records invalid release ordering");
}

bool test_atomic_store_rejects_load_only_ordering() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("mmio-base", BuiltinTypes::Pointer);

    eshkol_ast_t ast = parse_single("(atomic-store! u8 mmio-base 1 acquire)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "atomic-store! still synthesizes after ordering issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "atomic-store! keeps Null after ordering issue") &&
           expect_equal(checker.hasErrors(), true,
                        "atomic-store! records invalid acquire ordering");
}

bool test_atomic_store_rejects_non_pointer_target() {
    TypeEnvironment env;
    TypeChecker checker(env);
    checker.context().bind("not-a-pointer", BuiltinTypes::UInt64);

    eshkol_ast_t ast = parse_single("(atomic-store! u8 not-a-pointer 1 release)");
    const TypeCheckResult result = checker.synthesize(&ast);

    return expect_equal(result.success, true,
                        "atomic-store! still synthesizes after target issue") &&
           expect_equal(result.inferred_type, BuiltinTypes::Null,
                        "atomic-store! keeps Null result on target issue") &&
           expect_equal(checker.hasErrors(), true,
                        "atomic-store! records pointer target error");
}

bool test_atomic_ir_lowering() {
    eshkol_set_uses_stdlib(0);
    eshkol_set_target(nullptr);

    eshkol_ast_t asts[10] = {
        parse_single("(define mmio-base (usize->ptr 4096))"),
        parse_single("(define next-base (usize->ptr 8192))"),
        parse_single("(define (peek) : u8 (atomic-load u8 mmio-base acquire))"),
        parse_single("(define (peek-relaxed) : u16 (atomic-load u16 mmio-base relaxed))"),
        parse_single("(define (poke (v : u16)) : null (atomic-store! u16 mmio-base v release))"),
        parse_single("(define (peek-next) : ptr (atomic-load ptr mmio-base seq-cst))"),
        parse_single("(define (poke-next) : null (atomic-store! ptr mmio-base next-base seq-cst))"),
        parse_single("(define (swap32 (v : u32)) : u32 (atomic-exchange! u32 mmio-base v acq-rel))"),
        parse_single("(define (add32 (v : u32)) : u32 (atomic-fetch-add! u32 mmio-base v acq-rel))"),
        parse_single("(define (sub16 (v : u16)) : u16 (atomic-fetch-sub! u16 mmio-base v release))"),
    };

    LLVMModuleRef module = eshkol_generate_llvm_ir_library(asts, 10, "atomic_ops_test");
    if (!module) {
        std::cerr << "FAIL: atomic LLVM module generation" << std::endl;
        return false;
    }

    char ir_path[] = "/tmp/eshkol-atomic-ir-XXXXXX";
    const int fd = mkstemp(ir_path);
    if (fd == -1) {
        std::cerr << "FAIL: atomic temp IR path" << std::endl;
        eshkol_dispose_llvm_module(module);
        return false;
    }
    close(fd);

    bool ok = false;
    if (eshkol_dump_llvm_ir_to_file(module, ir_path) != 0) {
        std::cerr << "FAIL: atomic IR dump" << std::endl;
    } else {
        std::ifstream ir_stream(ir_path);
        std::stringstream ir_buffer;
        ir_buffer << ir_stream.rdbuf();
        const std::string ir = ir_buffer.str();

        ok = expect_contains(ir, "load atomic i8",
                             "atomic-load lowers to atomic i8 load") &&
             expect_contains(ir, "acquire",
                             "atomic-load keeps acquire ordering") &&
             expect_contains(ir, "load atomic i16",
                             "atomic-load relaxed lowers to atomic i16 load") &&
             expect_contains(ir, "monotonic",
                             "atomic relaxed maps to LLVM monotonic ordering") &&
             expect_contains(ir, "store atomic i16",
                             "atomic-store! lowers to atomic i16 store") &&
             expect_contains(ir, "release",
                             "atomic-store! keeps release ordering") &&
             expect_contains(ir, "load atomic ptr",
                             "atomic-load ptr lowers to atomic ptr load") &&
             expect_contains(ir, "store atomic ptr",
                             "atomic-store! ptr lowers to atomic ptr store") &&
             expect_contains(ir, "seq_cst",
                             "atomic pointer ops keep seq-cst ordering") &&
             expect_contains(ir, "atomicrmw xchg",
                             "atomic-exchange! lowers to atomicrmw xchg") &&
             expect_contains(ir, "atomicrmw add",
                             "atomic-fetch-add! lowers to atomicrmw add") &&
             expect_contains(ir, "atomicrmw sub",
                             "atomic-fetch-sub! lowers to atomicrmw sub") &&
             expect_contains(ir, "acq_rel",
                             "atomic-exchange! keeps acq-rel ordering") &&
             expect_contains(ir, "peek", "atomic load survives in function IR") &&
             expect_contains(ir, "poke", "atomic store survives in function IR") &&
             expect_contains(ir, "swap32", "atomic exchange survives in function IR") &&
             expect_contains(ir, "add32", "atomic fetch-add survives in function IR") &&
             expect_contains(ir, "sub16", "atomic fetch-sub survives in function IR");
    }

    std::remove(ir_path);
    eshkol_dispose_llvm_module(module);
    return ok;
}

}  // namespace

int main() {
    if (!test_atomic_load_type_synthesis()) {
        return 1;
    }
    if (!test_atomic_store_type_synthesis()) {
        return 1;
    }
    if (!test_atomic_exchange_type_synthesis()) {
        return 1;
    }
    if (!test_atomic_fetch_add_type_synthesis()) {
        return 1;
    }
    if (!test_atomic_pointer_type_synthesis()) {
        return 1;
    }
    if (!test_atomic_exchange_rejects_invalid_ordering()) {
        return 1;
    }
    if (!test_atomic_fetch_add_rejects_pointer_type()) {
        return 1;
    }
    if (!test_atomic_load_rejects_store_only_ordering()) {
        return 1;
    }
    if (!test_atomic_store_rejects_load_only_ordering()) {
        return 1;
    }
    if (!test_atomic_store_rejects_non_pointer_target()) {
        return 1;
    }
    if (!test_atomic_ir_lowering()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
