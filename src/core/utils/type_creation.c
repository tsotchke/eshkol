/**
 * @file type_creation.c
 * @brief Implementation of type creation functions
 */

#include "core/type_creation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a void type
 */
Type* type_void_create(Arena* arena) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_VOID;
    
    return type;
}

/**
 * @brief Create a boolean type
 */
Type* type_boolean_create(Arena* arena) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_BOOLEAN;
    
    return type;
}

/**
 * @brief Create an integer type
 */
Type* type_integer_create(Arena* arena, IntSize size) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_INTEGER;
    type->int_size = size;
    
    return type;
}

/**
 * @brief Create a float type
 */
Type* type_float_create(Arena* arena, FloatSize size) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_FLOAT;
    type->float_size = size;
    
    return type;
}

/**
 * @brief Create a character type
 */
Type* type_char_create(Arena* arena) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_CHAR;
    
    return type;
}

/**
 * @brief Create a string type
 */
Type* type_string_create(Arena* arena) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_STRING;
    
    return type;
}

/**
 * @brief Create a symbol type
 */
Type* type_symbol_create(Arena* arena) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_SYMBOL;
    
    return type;
}

/**
 * @brief Create a pair type
 */
Type* type_pair_create(Arena* arena, Type* car_type, Type* cdr_type) {
    assert(car_type != NULL);
    assert(cdr_type != NULL);
    
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_PAIR;
    
    // Allocate memory for parameters
    Type** params = arena_alloc(arena, 2 * sizeof(Type*));
    if (!params) return NULL;
    
    params[0] = car_type;
    params[1] = cdr_type;
    
    type->function.param_count = 2;
    type->function.params = params;
    type->function.return_type = NULL;
    type->function.variadic = false;
    
    return type;
}

/**
 * @brief Create a vector type
 */
Type* type_vector_create(Arena* arena, Type* element_type, size_t size) {
    assert(element_type != NULL);
    
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_VECTOR;
    type->vector.element_type = element_type;
    type->vector.size = size;
    
    return type;
}

/**
 * @brief Create a function type
 */
Type* type_function_create(Arena* arena, size_t param_count, Type** params, Type* return_type, bool variadic) {
    assert(return_type != NULL);
    assert(param_count == 0 || params != NULL);
    
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_FUNCTION;
    type->function.param_count = param_count;
    type->function.params = params;
    type->function.return_type = return_type;
    type->function.variadic = variadic;
    
    return type;
}

/**
 * @brief Create a structure type
 */
Type* type_struct_create(Arena* arena, size_t field_count, StructField* fields) {
    assert(field_count == 0 || fields != NULL);
    
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_STRUCT;
    type->structure.field_count = field_count;
    type->structure.fields = fields;
    
    return type;
}

/**
 * @brief Create a union type
 */
Type* type_union_create(Arena* arena, size_t variant_count, Type** variants) {
    assert(variant_count == 0 || variants != NULL);
    
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_UNION;
    type->union_type.variant_count = variant_count;
    type->union_type.variants = variants;
    
    return type;
}

/**
 * @brief Create an any type
 */
Type* type_any_create(Arena* arena) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_ANY;
    
    return type;
}

/**
 * @brief Create an unknown type
 */
Type* type_unknown_create(Arena* arena) {
    Type* type = arena_alloc(arena, sizeof(Type));
    if (!type) return NULL;
    
    type->kind = TYPE_UNKNOWN;
    
    return type;
}
