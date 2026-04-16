#include <eshkol/eshkol.h>
#include <eshkol/types/hott_types.h>
#include <eshkol/types/type_checker.h>

#include <iostream>
#include <sstream>
#include <string>

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

eshkol_ast_t parse_single(const std::string& source) {
    std::stringstream stream(source);
    return eshkol_parse_next_ast_from_stream(stream);
}

bool test_environment_aliases() {
    TypeEnvironment env;

    auto u8 = env.lookupType("u8");
    auto u16 = env.lookupType("u16");
    auto u32 = env.lookupType("u32");
    auto u64 = env.lookupType("u64");
    auto usize = env.lookupType("usize");
    auto i8 = env.lookupType("i8");
    auto i16 = env.lookupType("i16");
    auto i32 = env.lookupType("i32");
    auto isize = env.lookupType("isize");

    return u8 && u16 && u32 && u64 && usize && i8 && i16 && i32 && isize &&
           expect_equal(*u8, BuiltinTypes::UInt8, "u8 alias lookup") &&
           expect_equal(*u16, BuiltinTypes::UInt16, "u16 alias lookup") &&
           expect_equal(*u32, BuiltinTypes::UInt32, "u32 alias lookup") &&
           expect_equal(*u64, BuiltinTypes::UInt64, "u64 alias lookup") &&
           expect_equal(*usize, BuiltinTypes::USize, "usize alias lookup") &&
           expect_equal(*i8, BuiltinTypes::Int8, "i8 alias lookup") &&
           expect_equal(*i16, BuiltinTypes::Int16, "i16 alias lookup") &&
           expect_equal(*i32, BuiltinTypes::Int32, "i32 alias lookup") &&
           expect_equal(*isize, BuiltinTypes::ISize, "isize alias lookup") &&
           expect_equal(env.getRuntimeRep(BuiltinTypes::UInt8), RuntimeRep::UInt8,
                        "UInt8 runtime rep") &&
           expect_equal(env.getRuntimeRep(BuiltinTypes::Int16), RuntimeRep::Int16,
                        "Int16 runtime rep") &&
           expect_equal(env.toRuntimeType(BuiltinTypes::UInt64),
                        static_cast<uint8_t>(ESHKOL_VALUE_INT64),
                        "UInt64 runtime type bridge") &&
           expect_equal(env.promoteForArithmetic(BuiltinTypes::UInt8, BuiltinTypes::UInt16),
                        BuiltinTypes::Int64,
                        "machine integer arithmetic promotion");
}

bool test_type_resolution() {
    TypeEnvironment env;
    TypeChecker checker(env);

    hott_type_expr_t* u8_expr = hott_make_type_var("u8");
    hott_type_expr_t* usize_expr = hott_make_type_var("usize");
    hott_type_expr_t* number_expr = hott_make_type_var("number");

    const bool ok =
        expect_equal(checker.resolveType(u8_expr), BuiltinTypes::UInt8, "resolve u8 builtin") &&
        expect_equal(checker.resolveType(usize_expr), BuiltinTypes::USize, "resolve usize builtin") &&
        expect_equal(checker.resolveType(number_expr), BuiltinTypes::Number, "resolve number builtin");

    hott_free_type_expr(u8_expr);
    hott_free_type_expr(usize_expr);
    hott_free_type_expr(number_expr);
    return ok;
}

bool test_parser_type_annotation_resolution() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast = parse_single("(: io-port u8)");
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_TYPE_ANNOTATION_OP) {
        std::cerr << "FAIL: type annotation parse shape" << std::endl;
        return false;
    }
    if (!ast.operation.type_annotation_op.type_expr) {
        std::cerr << "FAIL: type annotation missing type expression" << std::endl;
        return false;
    }

    return expect_equal(
        checker.resolveType(ast.operation.type_annotation_op.type_expr),
        BuiltinTypes::UInt8,
        "parser + checker resolve (: io-port u8)");
}

bool test_parser_define_signature_resolution() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast = parse_single("(define (offset (n : usize)) : isize n)");
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP) {
        std::cerr << "FAIL: define parse shape" << std::endl;
        return false;
    }
    if (!ast.operation.define_op.param_types || !ast.operation.define_op.param_types[0]) {
        std::cerr << "FAIL: define missing parameter type annotation" << std::endl;
        return false;
    }
    if (!ast.operation.define_op.return_type) {
        std::cerr << "FAIL: define missing return type annotation" << std::endl;
        return false;
    }

    return expect_equal(
               checker.resolveType(ast.operation.define_op.param_types[0]),
               BuiltinTypes::USize,
               "parser + checker resolve define param usize") &&
           expect_equal(
               checker.resolveType(ast.operation.define_op.return_type),
               BuiltinTypes::ISize,
               "parser + checker resolve define return isize");
}

}  // namespace

int main() {
    if (!test_environment_aliases()) {
        return 1;
    }
    if (!test_type_resolution()) {
        return 1;
    }
    if (!test_parser_type_annotation_resolution()) {
        return 1;
    }
    if (!test_parser_define_signature_resolution()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
