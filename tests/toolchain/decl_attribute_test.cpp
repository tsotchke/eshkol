#include <eshkol/eshkol.h>
#include <eshkol/llvm_backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

template <typename T>
bool expect_equal(const T& actual, const T& expected, const std::string& label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

bool expect_string(const char* actual, const char* expected, const std::string& label) {
    if (actual && std::strcmp(actual, expected) == 0) {
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

bool expect_line_contains(const std::string& text, const std::string& anchor,
                          const std::string& needle, const std::string& label) {
    std::stringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find(anchor) == std::string::npos) {
            continue;
        }
        if (line.find(needle) != std::string::npos) {
            return true;
        }
        break;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

bool expect_line_not_contains(const std::string& text, const std::string& anchor,
                              const std::string& needle, const std::string& label) {
    std::stringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find(anchor) == std::string::npos) {
            continue;
        }
        if (line.find(needle) == std::string::npos) {
            return true;
        }
        break;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

eshkol_ast_t parse_single(const std::string& source) {
    std::stringstream stream(source);
    return eshkol_parse_next_ast_from_stream(stream);
}

bool test_define_attribute_parse_surface() {
    eshkol_ast_t ast = parse_single(
        "(define boot-flag 1 :link-section \".boot.data\" :align 32 :used :weak :export-symbol boot_flag_symbol)");

    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP) {
        std::cerr << "FAIL: define attribute parse shape" << std::endl;
        return false;
    }

    return expect_string(ast.operation.define_op.link_section, ".boot.data",
                         "define link-section parses") &&
           expect_equal(ast.operation.define_op.alignment, uint64_t{32},
                        "define align parses") &&
           expect_equal(ast.operation.define_op.has_alignment, uint8_t{1},
                        "define align flag parses") &&
           expect_equal(ast.operation.define_op.is_used, uint8_t{1},
                        "define used flag parses") &&
           expect_equal(ast.operation.define_op.is_weak, uint8_t{1},
                        "define weak flag parses") &&
           expect_equal(ast.operation.define_op.export_symbol, uint8_t{1},
                        "define export-symbol flag parses") &&
           expect_string(ast.operation.define_op.export_name, "boot_flag_symbol",
                         "define export-symbol emitted name parses");
}

bool test_define_same_name_export_surface() {
    eshkol_ast_t ast = parse_single("(define keep-name 1 :export-symbol)");

    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP) {
        std::cerr << "FAIL: define same-name export parse shape" << std::endl;
        return false;
    }

    return expect_equal(ast.operation.define_op.export_symbol, uint8_t{1},
                        "define same-name export flag parses") &&
           expect_equal(ast.operation.define_op.export_name == nullptr, true,
                        "define same-name export leaves emitted name empty");
}

bool test_extern_attribute_parse_surface() {
    eshkol_ast_t ast = parse_single(
        "(extern void halt :extern-symbol abort :weak :no-return)");

    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_EXTERN_OP) {
        std::cerr << "FAIL: extern attribute parse shape" << std::endl;
        return false;
    }

    return expect_string(ast.operation.extern_op.real_name, "abort",
                         "extern-symbol parses into real symbol name") &&
           expect_equal(ast.operation.extern_op.is_weak, uint8_t{1},
                        "extern weak flag parses") &&
           expect_equal(ast.operation.extern_op.is_no_return, uint8_t{1},
                        "extern no-return flag parses");
}

bool test_extern_var_attribute_parse_surface() {
    eshkol_ast_t ast = parse_single(
        "(extern-var int errno-slot :extern-symbol errno)");

    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_EXTERN_VAR_OP) {
        std::cerr << "FAIL: extern-var attribute parse shape" << std::endl;
        return false;
    }

    return expect_string(ast.operation.extern_var_op.real_name, "errno",
                         "extern-var real symbol name parses");
}

bool test_declaration_attribute_ir_lowering() {
    eshkol_set_uses_stdlib(0);
    eshkol_set_target("x86_64-unknown-linux-gnu");

    eshkol_ast_t asts[3] = {
        parse_single("(extern void halt :extern-symbol abort :weak :no-return)"),
        parse_single("(define boot-flag 1 :link-section \".boot.data\" :align 32 :used :weak :export-symbol boot_flag_symbol)"),
        parse_single("(define (entry) : null (compiler-fence seq-cst) :link-section \".boot.text\" :align 16 :used :export-symbol entry_symbol :no-return)"),
    };

    LLVMModuleRef module =
        eshkol_generate_llvm_ir_library(asts, 3, "decl_attribute_test");
    if (!module) {
        std::cerr << "FAIL: declaration attribute LLVM module generation" << std::endl;
        eshkol_set_target(nullptr);
        return false;
    }

    char ir_path[] = "/tmp/eshkol-decl-attrs-ir-XXXXXX";
    const int fd = mkstemp(ir_path);
    if (fd == -1) {
        std::cerr << "FAIL: declaration attribute temp IR path" << std::endl;
        eshkol_dispose_llvm_module(module);
        eshkol_set_target(nullptr);
        return false;
    }
    close(fd);

    bool ok = false;
    if (eshkol_dump_llvm_ir_to_file(module, ir_path) != 0) {
        std::cerr << "FAIL: declaration attribute IR dump" << std::endl;
    } else {
        std::ifstream ir_stream(ir_path);
        std::stringstream ir_buffer;
        ir_buffer << ir_stream.rdbuf();
        const std::string ir = ir_buffer.str();

        ok = expect_contains(ir, "target triple = \"x86_64-unknown-linux-gnu\"",
                             "library-mode IR honors explicit target triple") &&
             expect_line_contains(ir, "@boot_flag_symbol", "weak",
                                  "weak global lowering survives in IR") &&
             expect_line_contains(ir, "@boot_flag_symbol", "section \".boot.data\"",
                                  "global link-section lowering survives in IR") &&
             expect_line_contains(ir, "@boot_flag_symbol", "align 32",
                                  "global align lowering survives in IR") &&
             expect_contains(ir, "declare extern_weak void @abort()",
                             "extern weak symbol lowering survives in IR") &&
             expect_line_contains(ir, "@entry_symbol(", "section \".boot.text\"",
                                  "function link-section lowering survives in IR") &&
             expect_line_contains(ir, "@entry_symbol(", "align 16",
                                  "function align lowering survives in IR") &&
             expect_line_not_contains(ir, "@entry_symbol(", "linkonce_odr",
                                      "export-symbol keeps entry out of linkonce_odr linkage") &&
             expect_contains(ir, "@llvm.used",
                             "used declarations lower through llvm.used") &&
             expect_contains(ir, "noreturn",
                             "no-return lowers to LLVM noreturn attributes");
    }

    std::remove(ir_path);
    eshkol_dispose_llvm_module(module);
    eshkol_set_target(nullptr);
    return ok;
}

}  // namespace

int main() {
    if (!test_define_attribute_parse_surface()) {
        return 1;
    }
    if (!test_define_same_name_export_surface()) {
        return 1;
    }
    if (!test_extern_attribute_parse_surface()) {
        return 1;
    }
    if (!test_extern_var_attribute_parse_surface()) {
        return 1;
    }
    if (!test_declaration_attribute_ir_lowering()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
