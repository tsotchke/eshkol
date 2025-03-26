/**
 * @file type_conversion.c
 * @brief Type conversion for code generation implementation
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
    assert(type != NULL);
    
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
            return "char*";
        case TYPE_PAIR:
            return "Pair*";
        case TYPE_VECTOR:
            if (type->vector.element_type) {
                switch (type->vector.element_type->kind) {
                    case TYPE_BOOLEAN:
                        return "VectorB*";
                    case TYPE_INTEGER:
                        return "VectorI*";
                    case TYPE_FLOAT:
                        return "VectorF*";
                    case TYPE_CHAR:
                        return "VectorC*";
                    case TYPE_STRING:
                        return "VectorS*";
                    default:
                        return "Vector*";
                }
            } else {
                return "Vector*";
            }
        case TYPE_FUNCTION:
            return "void*"; // Function pointers are complex in C
        case TYPE_STRUCT:
            return "void*"; // Structs are complex in C
        case TYPE_UNION:
            return "void*"; // Unions are complex in C
        case TYPE_ANY:
            return "void*";
        case TYPE_UNKNOWN:
            return "void*";
        default:
            return "void*";
    }
}

/**
 * @brief Get the C type string for an AST node
 */
const char* codegen_get_c_type(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Get node type
    Type* type = type_inference_get_type(type_context, node);
    
    // Convert to C type
    if (type) {
        return codegen_type_to_c_type(type);
    } else {
        return "int"; // Default to int
    }
}

/**
 * @brief Check if a type can be implicitly converted to another type
 */
bool codegen_type_can_convert(Type* from, Type* to) {
    assert(from != NULL);
    assert(to != NULL);
    
    // Same type
    if (type_equals(from, to)) {
        return true;
    }
    
    // Numeric conversions
    if ((from->kind == TYPE_INTEGER || from->kind == TYPE_FLOAT) &&
        (to->kind == TYPE_INTEGER || to->kind == TYPE_FLOAT)) {
        return true;
    }
    
    // Char to int
    if (from->kind == TYPE_CHAR && to->kind == TYPE_INTEGER) {
        return true;
    }
    
    // Int to char
    if (from->kind == TYPE_INTEGER && to->kind == TYPE_CHAR) {
        return true;
    }
    
    // Any to void*
    if (to->kind == TYPE_ANY) {
        return true;
    }
    
    // String to void* (for mixed type conditionals)
    if (from->kind == TYPE_STRING && to->kind == TYPE_ANY) {
        return true;
    }
    
    // Numeric to void* (for mixed type conditionals)
    if ((from->kind == TYPE_INTEGER || from->kind == TYPE_FLOAT) && to->kind == TYPE_ANY) {
        return true;
    }
    
    return false;
}

/**
 * @brief Generate code for type conversion
 */
const char* codegen_type_conversion_code(CodegenContext* context, Type* from, Type* to) {
    assert(context != NULL);
    assert(from != NULL);
    assert(to != NULL);
    
    // Same type
    if (type_equals(from, to)) {
        return "";
    }
    
    // Get C types
    const char* from_type = codegen_type_to_c_type(from);
    const char* to_type = codegen_type_to_c_type(to);
    
    // Generate conversion
    if (from->kind == TYPE_INTEGER && to->kind == TYPE_FLOAT) {
        return "(float)";
    } else if (from->kind == TYPE_FLOAT && to->kind == TYPE_INTEGER) {
        return "(int)";
    } else if (from->kind == TYPE_CHAR && to->kind == TYPE_INTEGER) {
        return "(int)";
    } else if (from->kind == TYPE_INTEGER && to->kind == TYPE_CHAR) {
        return "(char)";
    } else if (to->kind == TYPE_ANY) {
        // Special case for string to void*
        if (from->kind == TYPE_STRING) {
            return "(void*)(char*)";
        }
        // Special case for numeric to void*
        else if (from->kind == TYPE_INTEGER || from->kind == TYPE_FLOAT) {
            // For scientific computing, we need to be careful with numeric conversions
            // We'll use a more explicit cast to void*
            Arena* arena = codegen_context_get_arena(context);
            char* result = arena_alloc(arena, strlen(from_type) + 20);
            sprintf(result, "({ %s temp = ", from_type);
            return result;
        }
        return "(void*)";
    } else {
        // Default cast
        char* result = arena_alloc(codegen_context_get_arena(context), strlen(to_type) + 3);
        sprintf(result, "(%s)", to_type);
        return result;
    }
}

/**
 * @brief Apply type conversion to an expression
 */
char* codegen_type_apply_conversion(CodegenContext* context, const char* expr, Type* from, Type* to) {
    assert(context != NULL);
    assert(expr != NULL);
    assert(from != NULL);
    assert(to != NULL);
    
    // Get conversion code
    const char* conversion = codegen_type_conversion_code(context, from, to);
    
    // Apply conversion
    Arena* arena = codegen_context_get_arena(context);
    
    // Special case for numeric to void* (for mixed type conditionals)
    if (to->kind == TYPE_ANY && (from->kind == TYPE_INTEGER || from->kind == TYPE_FLOAT)) {
        char* result = arena_alloc(arena, strlen(conversion) + strlen(expr) + 30);
        sprintf(result, "%s%s; (void*)&temp; })", conversion, expr);
        return result;
    }
    
    char* result = arena_alloc(arena, strlen(conversion) + strlen(expr) + 3);
    sprintf(result, "%s(%s)", conversion, expr);
    return result;
}
