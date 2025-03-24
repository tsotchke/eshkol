/**
 * @file type_conversion.c
 * @brief Type conversion utilities for code generation
 */

#include "backend/codegen/type_conversion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Convert an Eshkol type to a C type string
 */
const char* codegen_type_to_c_type(Type* type) {
    // Handle null type gracefully
    if (type == NULL) {
        return "void*";
    }
    
    switch (type->kind) {
        case TYPE_VOID:
            return "void";
        case TYPE_BOOLEAN:
            return "bool";
        case TYPE_INTEGER:
            switch (type->int_size) {
                case INT_SIZE_8:
                    return "int8_t";
                case INT_SIZE_16:
                    return "int16_t";
                case INT_SIZE_32:
                    return "int32_t";
                case INT_SIZE_64:
                    return "int64_t";
                default:
                    return "int";
            }
        case TYPE_FLOAT:
            switch (type->float_size) {
                case FLOAT_SIZE_32:
                    return "float";
                case FLOAT_SIZE_64:
                    return "double";
                default:
                    return "double";
            }
        case TYPE_CHAR:
            return "char";
        case TYPE_STRING:
            return "char*";
        case TYPE_SYMBOL:
            return "const char*";
        case TYPE_PAIR:
            return "void*"; // Pairs are represented as pointers
        case TYPE_VECTOR:
            return "VectorF*"; // Vectors are represented as VectorF*
        case TYPE_FUNCTION:
            return "void*"; // Functions are represented as function pointers
        case TYPE_STRUCT:
            return "void*"; // Structs are represented as pointers
        case TYPE_UNION:
            return "void*"; // Unions are represented as pointers
        case TYPE_ANY:
            return "eshkol_value_t"; // Any type is represented as eshkol_value_t
        case TYPE_UNKNOWN:
            return "void*"; // Unknown type is represented as void*
        default:
            return "int"; // Default to int
    }
}

/**
 * @brief Check if a type is a numeric type
 */
bool codegen_type_is_numeric(Type* type) {
    if (type == NULL) {
        return false;
    }
    
    return type->kind == TYPE_INTEGER || type->kind == TYPE_FLOAT;
}

/**
 * @brief Check if a type is a floating-point type
 */
bool codegen_type_is_float(Type* type) {
    if (type == NULL) {
        return false;
    }
    
    return type->kind == TYPE_FLOAT;
}

/**
 * @brief Check if a type is an integer type
 */
bool codegen_type_is_integer(Type* type) {
    if (type == NULL) {
        return false;
    }
    
    return type->kind == TYPE_INTEGER;
}

/**
 * @brief Check if a type is a boolean type
 */
bool codegen_type_is_boolean(Type* type) {
    if (type == NULL) {
        return false;
    }
    
    return type->kind == TYPE_BOOLEAN;
}

/**
 * @brief Check if a type is a string type
 */
bool codegen_type_is_string(Type* type) {
    if (type == NULL) {
        return false;
    }
    
    return type->kind == TYPE_STRING;
}

/**
 * @brief Check if a type is a vector type
 */
bool codegen_type_is_vector(Type* type) {
    if (type == NULL) {
        return false;
    }
    
    return type->kind == TYPE_VECTOR;
}

/**
 * @brief Check if a type is a function type
 */
bool codegen_type_is_function(Type* type) {
    if (type == NULL) {
        return false;
    }
    
    return type->kind == TYPE_FUNCTION;
}
