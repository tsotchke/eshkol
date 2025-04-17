/**
 * @file type_inference.c
 * @brief Implementation of the type inference system
 */

#include "frontend/type_inference/type_inference.h"
#include "frontend/binding/binding.h"
#include "frontend/type_inference/context.h"
#include "frontend/type_inference/inference.h"
#include "frontend/type_inference/conversion.h"
#include "core/memory.h"
#include "core/type.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include "core/type_conversion.h"
#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

/**
 * @brief Initialize the type inference system
 */
TypeInferenceContext* type_inference_init(BindingSystem* binding_system, Arena* arena, DiagnosticContext* diagnostics) {
    assert(arena != NULL);
    
    // Create the type inference context
    TypeInferenceContext* context = type_inference_context_create(binding_system, arena, diagnostics);
    if (!context) return NULL;
    
    return context;
}

/**
 * @brief Infer types for an AST
 */
bool type_inference_run(TypeInferenceContext* context, AstNode* ast) {
    assert(context != NULL);
    assert(ast != NULL);
    
    // First, collect explicit types from the AST
    if (!type_inference_collect_explicit_types(context, ast)) {
        return false;
    }
    
    // Then, infer types for the AST
    if (!type_inference_infer(context, ast)) {
        return false;
    }
    
    return true;
}

/**
 * @brief Get the inferred type for an AST node
 */
Type* type_inference_get_node_type(TypeInferenceContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    return type_inference_resolve_type(context, node);
}

/**
 * @brief Check if a type can be implicitly converted to another type
 */
bool type_inference_can_convert(Type* from, Type* to) {
    assert(from != NULL);
    assert(to != NULL);
    
    return type_can_convert(from, to);
}

/**
 * @brief Apply type conversion to an expression
 */
char* type_inference_apply_conversion(Arena* arena, const char* expr, Type* from, Type* to) {
    assert(arena != NULL);
    assert(expr != NULL);
    assert(from != NULL);
    assert(to != NULL);
    
    return type_apply_conversion(arena, expr, from, to);
}

/**
 * @brief Get the common supertype of two types
 */
Type* type_inference_common_supertype(Arena* arena, Type* a, Type* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    
    return type_common_supertype(arena, a, b);
}

/**
 * @brief Check if two types are equal
 */
bool type_inference_types_equal(Type* a, Type* b) {
    assert(a != NULL);
    assert(b != NULL);
    
    return type_equals(a, b);
}

/**
 * @brief Check if a type is a subtype of another
 */
bool type_inference_is_subtype(Type* sub, Type* super) {
    assert(sub != NULL);
    assert(super != NULL);
    
    return type_is_subtype(sub, super);
}

/**
 * @brief Convert a type to a string
 */
char* type_inference_type_to_string(Arena* arena, Type* type) {
    assert(arena != NULL);
    assert(type != NULL);
    
    return type_to_string(arena, type);
}

/**
 * @brief Parse a type from a string
 */
Type* type_inference_type_from_string(Arena* arena, const char* str) {
    assert(arena != NULL);
    assert(str != NULL);
    
    return type_from_string(arena, str);
}

/**
 * @brief Create a vector type
 */
Type* type_inference_create_vector_type(Arena* arena, Type* element_type, size_t size) {
    assert(arena != NULL);
    assert(element_type != NULL);
    
    return type_vector_create(arena, element_type, size);
}

/**
 * @brief Create a float type
 */
Type* type_inference_create_float_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_float_create(arena, FLOAT_SIZE_32);
}

/**
 * @brief Create an integer type
 */
Type* type_inference_create_integer_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_integer_create(arena, INT_SIZE_32);
}

/**
 * @brief Create a boolean type
 */
Type* type_inference_create_boolean_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_boolean_create(arena);
}

/**
 * @brief Create a void type
 */
Type* type_inference_create_void_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_void_create(arena);
}

/**
 * @brief Create an any type
 */
Type* type_inference_create_any_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_any_create(arena);
}

/**
 * @brief Create an unknown type
 */
Type* type_inference_create_unknown_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_unknown_create(arena);
}

/**
 * @brief Create a function type
 */
Type* type_inference_create_function_type(Arena* arena, size_t param_count, Type** params, Type* return_type, bool variadic) {
    assert(arena != NULL);
    assert(param_count == 0 || params != NULL);
    assert(return_type != NULL);
    
    return type_function_create(arena, param_count, params, return_type, variadic);
}

/**
 * @brief Create a pair type
 */
Type* type_inference_create_pair_type(Arena* arena, Type* car_type, Type* cdr_type) {
    assert(arena != NULL);
    assert(car_type != NULL);
    assert(cdr_type != NULL);
    
    return type_pair_create(arena, car_type, cdr_type);
}

/**
 * @brief Create a string type
 */
Type* type_inference_create_string_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_string_create(arena);
}

/**
 * @brief Create a character type
 */
Type* type_inference_create_char_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_char_create(arena);
}

/**
 * @brief Create a symbol type
 */
Type* type_inference_create_symbol_type(Arena* arena) {
    assert(arena != NULL);
    
    return type_symbol_create(arena);
}

/**
 * @brief Get the element type of a vector type
 */
Type* type_inference_get_vector_element_type(Type* vector_type) {
    assert(vector_type != NULL);
    assert(vector_type->kind == TYPE_VECTOR);
    
    return vector_type->vector.element_type;
}

/**
 * @brief Get the size of a vector type
 */
size_t type_inference_get_vector_size(Type* vector_type) {
    assert(vector_type != NULL);
    assert(vector_type->kind == TYPE_VECTOR);
    
    return vector_type->vector.size;
}

/**
 * @brief Get the return type of a function type
 */
Type* type_inference_get_function_return_type(Type* function_type) {
    assert(function_type != NULL);
    assert(function_type->kind == TYPE_FUNCTION);
    
    return function_type->function.return_type;
}

/**
 * @brief Get the parameter count of a function type
 */
size_t type_inference_get_function_param_count(Type* function_type) {
    assert(function_type != NULL);
    assert(function_type->kind == TYPE_FUNCTION);
    
    return function_type->function.param_count;
}

/**
 * @brief Get the parameter types of a function type
 */
Type** type_inference_get_function_param_types(Type* function_type) {
    assert(function_type != NULL);
    assert(function_type->kind == TYPE_FUNCTION);
    
    return function_type->function.params;
}

/**
 * @brief Check if a function type is variadic
 */
bool type_inference_is_function_variadic(Type* function_type) {
    assert(function_type != NULL);
    assert(function_type->kind == TYPE_FUNCTION);
    
    return function_type->function.variadic;
}

/**
 * @brief Get the car type of a pair type
 */
Type* type_inference_get_pair_car_type(Type* pair_type) {
    assert(pair_type != NULL);
    assert(pair_type->kind == TYPE_PAIR);
    
    return pair_type->function.params[0];
}

/**
 * @brief Get the cdr type of a pair type
 */
Type* type_inference_get_pair_cdr_type(Type* pair_type) {
    assert(pair_type != NULL);
    assert(pair_type->kind == TYPE_PAIR);
    
    return pair_type->function.params[1];
}

/**
 * @brief Get the kind of a type
 */
TypeKind type_inference_get_type_kind(Type* type) {
    assert(type != NULL);
    
    return type->kind;
}

/**
 * @brief Check if a type is a vector type
 */
bool type_inference_is_vector_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_VECTOR;
}

/**
 * @brief Check if a type is a function type
 */
bool type_inference_is_function_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_FUNCTION;
}

/**
 * @brief Check if a type is a pair type
 */
bool type_inference_is_pair_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_PAIR;
}

/**
 * @brief Check if a type is a float type
 */
bool type_inference_is_float_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_FLOAT;
}

/**
 * @brief Check if a type is an integer type
 */
bool type_inference_is_integer_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_INTEGER;
}

/**
 * @brief Check if a type is a boolean type
 */
bool type_inference_is_boolean_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_BOOLEAN;
}

/**
 * @brief Check if a type is a void type
 */
bool type_inference_is_void_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_VOID;
}

/**
 * @brief Check if a type is an any type
 */
bool type_inference_is_any_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_ANY;
}

/**
 * @brief Check if a type is an unknown type
 */
bool type_inference_is_unknown_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_UNKNOWN;
}

/**
 * @brief Check if a type is a string type
 */
bool type_inference_is_string_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_STRING;
}

/**
 * @brief Check if a type is a character type
 */
bool type_inference_is_char_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_CHAR;
}

/**
 * @brief Check if a type is a symbol type
 */
bool type_inference_is_symbol_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_SYMBOL;
}

/**
 * @brief Check if a type is a numeric type (integer or float)
 */
bool type_inference_is_numeric_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_INTEGER || type->kind == TYPE_FLOAT;
}

/**
 * @brief Check if a type is a scalar type (numeric, boolean, character)
 */
bool type_inference_is_scalar_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_INTEGER || type->kind == TYPE_FLOAT || 
           type->kind == TYPE_BOOLEAN || type->kind == TYPE_CHAR;
}

/**
 * @brief Check if a type is a compound type (vector, function, pair, struct, union)
 */
bool type_inference_is_compound_type(Type* type) {
    assert(type != NULL);
    
    return type->kind == TYPE_VECTOR || type->kind == TYPE_FUNCTION || 
           type->kind == TYPE_PAIR || type->kind == TYPE_STRUCT || 
           type->kind == TYPE_UNION;
}
