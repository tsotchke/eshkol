#include <eshkol/eshkol.h>
#include <eshkol/types/hott_types.h>
#include <eshkol/types/type_checker.h>

#include <cstring>
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

bool expect_string(const char* actual, const char* expected, const std::string& label) {
    if (actual && std::strcmp(actual, expected) == 0) {
        return true;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

eshkol_ast_t parse_single(const std::string& source) {
    std::stringstream stream(source);
    return eshkol_parse_next_ast_from_stream(stream);
}

bool test_environment_surface() {
    TypeEnvironment env;

    auto ptr = env.lookupType("ptr");
    auto pointer = env.lookupType("pointer");
    auto uppercase = env.lookupType("Ptr");

    if (!ptr || !pointer || !uppercase) {
        std::cerr << "FAIL: missing pointer builtin aliases" << std::endl;
        return false;
    }

    ParameterizedType byte_ptr = env.makePointerType(BuiltinTypes::UInt8);

    return expect_equal(*ptr, BuiltinTypes::Pointer, "ptr alias lookup") &&
           expect_equal(*pointer, BuiltinTypes::Pointer, "pointer alias lookup") &&
           expect_equal(*uppercase, BuiltinTypes::Pointer, "Ptr builtin lookup") &&
           expect_equal(env.getRuntimeRep(BuiltinTypes::Pointer), RuntimeRep::Pointer,
                        "Ptr runtime rep") &&
           expect_equal(env.isTypeFamily(BuiltinTypes::Pointer), true,
                        "Ptr is a type family") &&
           expect_equal(byte_ptr.base_type, BuiltinTypes::Pointer,
                        "makePointerType base type") &&
           expect_equal(byte_ptr.elementType(), BuiltinTypes::UInt8,
                        "makePointerType element type");
}

bool test_type_expression_round_trip() {
    hott_type_expr_t* u8 = hott_make_type_var("u8");
    hott_type_expr_t* ptr = hott_make_pointer_type(u8);
    hott_type_expr_t* tensor = hott_make_tensor_type(u8);
    hott_type_expr_t* ptr_copy = hott_copy_type_expr(ptr);

    char* ptr_string = hott_type_to_string(ptr);
    char* tensor_string = hott_type_to_string(tensor);

    const bool ok =
        ptr_copy &&
        ptr_copy->kind == HOTT_TYPE_POINTER &&
        ptr_copy->container.element_type &&
        ptr_copy->container.element_type->kind == HOTT_TYPE_VAR &&
        ptr_copy->container.element_type->var_name &&
        std::strcmp(ptr_copy->container.element_type->var_name, "u8") == 0 &&
        expect_string(ptr_string, "(ptr u8)", "pointer type stringification") &&
        expect_string(tensor_string, "(tensor u8)", "tensor type stringification");

    hott_free_type_expr(ptr_copy);
    hott_free_type_expr(tensor);
    hott_free_type_expr(ptr);
    hott_free_type_expr(u8);
    return ok;
}

bool test_type_resolution() {
    TypeEnvironment env;
    TypeChecker checker(env);

    hott_type_expr_t* raw_ptr = hott_make_type_var("ptr");
    hott_type_expr_t* u8 = hott_make_type_var("u8");
    hott_type_expr_t* ptr_u8 = hott_make_pointer_type(u8);

    const bool ok =
        expect_equal(checker.resolveType(raw_ptr), BuiltinTypes::Pointer,
                     "resolve ptr builtin") &&
        expect_equal(checker.resolveType(ptr_u8), BuiltinTypes::Pointer,
                     "resolve (ptr u8) builtin") &&
        expect_equal(checker.resolveType(ptr_u8->container.element_type), BuiltinTypes::UInt8,
                     "resolve pointer element builtin");

    hott_free_type_expr(u8);
    hott_free_type_expr(ptr_u8);
    hott_free_type_expr(raw_ptr);
    return ok;
}

bool test_parser_type_annotation_resolution() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast = parse_single("(: uart-base (ptr u8))");
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_TYPE_ANNOTATION_OP) {
        std::cerr << "FAIL: pointer type annotation parse shape" << std::endl;
        return false;
    }
    if (!ast.operation.type_annotation_op.type_expr) {
        std::cerr << "FAIL: pointer type annotation missing type expression" << std::endl;
        return false;
    }

    hott_type_expr_t* type_expr = ast.operation.type_annotation_op.type_expr;
    if (type_expr->kind != HOTT_TYPE_POINTER || !type_expr->container.element_type) {
        std::cerr << "FAIL: pointer type annotation kind" << std::endl;
        return false;
    }

    return expect_equal(checker.resolveType(type_expr), BuiltinTypes::Pointer,
                        "parser + checker resolve (: uart-base (ptr u8))") &&
           expect_equal(checker.resolveType(type_expr->container.element_type), BuiltinTypes::UInt8,
                        "parser + checker resolve pointer element type");
}

bool test_parser_define_signature_resolution() {
    TypeEnvironment env;
    TypeChecker checker(env);

    eshkol_ast_t ast = parse_single("(define (peek (base : (ptr u8))) : (ptr u8) base)");
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP) {
        std::cerr << "FAIL: pointer define parse shape" << std::endl;
        return false;
    }
    if (!ast.operation.define_op.param_types || !ast.operation.define_op.param_types[0]) {
        std::cerr << "FAIL: pointer define missing parameter type annotation" << std::endl;
        return false;
    }
    if (!ast.operation.define_op.return_type) {
        std::cerr << "FAIL: pointer define missing return type annotation" << std::endl;
        return false;
    }

    hott_type_expr_t* param_type = ast.operation.define_op.param_types[0];
    hott_type_expr_t* return_type = ast.operation.define_op.return_type;
    if (param_type->kind != HOTT_TYPE_POINTER || return_type->kind != HOTT_TYPE_POINTER) {
        std::cerr << "FAIL: pointer define annotation kinds" << std::endl;
        return false;
    }

    return expect_equal(checker.resolveType(param_type), BuiltinTypes::Pointer,
                        "define param resolves to Ptr") &&
           expect_equal(checker.resolveType(param_type->container.element_type), BuiltinTypes::UInt8,
                        "define param pointer element resolves to UInt8") &&
           expect_equal(checker.resolveType(return_type), BuiltinTypes::Pointer,
                        "define return resolves to Ptr") &&
           expect_equal(checker.resolveType(return_type->container.element_type), BuiltinTypes::UInt8,
                        "define return pointer element resolves to UInt8");
}

}  // namespace

int main() {
    if (!test_environment_surface()) {
        return 1;
    }
    if (!test_type_expression_round_trip()) {
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
