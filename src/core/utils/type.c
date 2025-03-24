/**
 * @file type.c
 * @brief Implementation of the type system
 */

#include "core/type.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

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
    
    // Allocate memory for the parameter types
    Type** params = arena_alloc(arena, 2 * sizeof(Type*));
    if (!params) return NULL;
    
    params[0] = car_type;
    params[1] = cdr_type;
    
    type->function.param_count = 2;
    type->function.params = params;
    
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

/**
 * @brief Check if two types are equal
 */
bool type_equals(Type* a, Type* b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    if (a->kind != b->kind) return false;
    
    switch (a->kind) {
        case TYPE_VOID:
        case TYPE_BOOLEAN:
        case TYPE_CHAR:
        case TYPE_STRING:
        case TYPE_SYMBOL:
        case TYPE_ANY:
        case TYPE_UNKNOWN:
            return true;
            
        case TYPE_INTEGER:
            return a->int_size == b->int_size;
            
        case TYPE_FLOAT:
            return a->float_size == b->float_size;
            
        case TYPE_PAIR:
            return type_equals(a->function.params[0], b->function.params[0]) &&
                   type_equals(a->function.params[1], b->function.params[1]);
            
        case TYPE_VECTOR:
            return type_equals(a->vector.element_type, b->vector.element_type) &&
                   a->vector.size == b->vector.size;
            
        case TYPE_FUNCTION:
            if (a->function.param_count != b->function.param_count ||
                a->function.variadic != b->function.variadic ||
                !type_equals(a->function.return_type, b->function.return_type)) {
                return false;
            }
            
            for (size_t i = 0; i < a->function.param_count; i++) {
                if (!type_equals(a->function.params[i], b->function.params[i])) {
                    return false;
                }
            }
            
            return true;
            
        case TYPE_STRUCT:
            if (a->structure.field_count != b->structure.field_count) {
                return false;
            }
            
            for (size_t i = 0; i < a->structure.field_count; i++) {
                if (strcmp(a->structure.fields[i].name, b->structure.fields[i].name) != 0 ||
                    !type_equals(a->structure.fields[i].type, b->structure.fields[i].type)) {
                    return false;
                }
            }
            
            return true;
            
        case TYPE_UNION:
            if (a->union_type.variant_count != b->union_type.variant_count) {
                return false;
            }
            
            // Check if all variants in a are in b
            for (size_t i = 0; i < a->union_type.variant_count; i++) {
                bool found = false;
                
                for (size_t j = 0; j < b->union_type.variant_count; j++) {
                    if (type_equals(a->union_type.variants[i], b->union_type.variants[j])) {
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    return false;
                }
            }
            
            return true;
    }
    
    return false;
}

/**
 * @brief Check if a type is a subtype of another
 */
bool type_is_subtype(Type* sub, Type* super) {
    if (sub == NULL || super == NULL) return false;
    
    // Any type is a subtype of itself
    if (type_equals(sub, super)) return true;
    
    // Any type is a subtype of TYPE_ANY
    if (super->kind == TYPE_ANY) return true;
    
    // TYPE_UNKNOWN is a subtype of any type
    if (sub->kind == TYPE_UNKNOWN) return true;
    
    switch (sub->kind) {
        case TYPE_INTEGER:
            // Integer widening
            if (super->kind == TYPE_INTEGER) {
                return sub->int_size <= super->int_size;
            }
            
            // Integer to float conversion
            if (super->kind == TYPE_FLOAT) {
                return true;
            }
            
            return false;
            
        case TYPE_FLOAT:
            // Float widening
            if (super->kind == TYPE_FLOAT) {
                return sub->float_size <= super->float_size;
            }
            
            return false;
            
        case TYPE_FUNCTION:
            if (super->kind != TYPE_FUNCTION) {
                return false;
            }
            
            // Contravariant parameter types
            if (sub->function.param_count < super->function.param_count) {
                return false;
            }
            
            for (size_t i = 0; i < super->function.param_count; i++) {
                // For contravariance, the parameter types of the supertype must be subtypes of the parameter types of the subtype
                if (!type_is_subtype(super->function.params[i], sub->function.params[i])) {
                    return false;
                }
            }
            
            // Covariant return type
            // For covariance, the return type of the subtype must be a subtype of the return type of the supertype
            if (!type_is_subtype(sub->function.return_type, super->function.return_type)) {
                return false;
            }
            
            // Variadic functions
            if (super->function.variadic && !sub->function.variadic) {
                return false;
            }
            
            return true;
            
        case TYPE_VECTOR:
            if (super->kind != TYPE_VECTOR) {
                return false;
            }
            
            // Vector element type
            if (!type_is_subtype(sub->vector.element_type, super->vector.element_type)) {
                return false;
            }
            
            // Vector size
            if (super->vector.size != 0 && sub->vector.size != super->vector.size) {
                return false;
            }
            
            return true;
            
        case TYPE_STRUCT:
            if (super->kind != TYPE_STRUCT) {
                return false;
            }
            
            // Structural subtyping: all fields in super must be in sub
            for (size_t i = 0; i < super->structure.field_count; i++) {
                bool found = false;
                
                for (size_t j = 0; j < sub->structure.field_count; j++) {
                    if (strcmp(super->structure.fields[i].name, sub->structure.fields[j].name) == 0) {
                        if (!type_is_subtype(sub->structure.fields[j].type, super->structure.fields[i].type)) {
                            return false;
                        }
                        
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    return false;
                }
            }
            
            return true;
            
        case TYPE_UNION:
            if (super->kind == TYPE_UNION) {
                // All variants in sub must be subtypes of some variant in super
                for (size_t i = 0; i < sub->union_type.variant_count; i++) {
                    bool found = false;
                    
                    for (size_t j = 0; j < super->union_type.variant_count; j++) {
                        if (type_is_subtype(sub->union_type.variants[i], super->union_type.variants[j])) {
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found) {
                        return false;
                    }
                }
                
                return true;
            } else {
                // A union type is a subtype of a non-union type if all its variants are subtypes of that type
                for (size_t i = 0; i < sub->union_type.variant_count; i++) {
                    if (!type_is_subtype(sub->union_type.variants[i], super)) {
                        return false;
                    }
                }
                
                return true;
            }
            
        default:
            return false;
    }
}

/**
 * @brief Get the common supertype of two types
 */
Type* type_common_supertype(Arena* arena, Type* a, Type* b) {
    if (a == NULL || b == NULL) return NULL;
    
    // If one is a subtype of the other, return the supertype
    if (type_is_subtype(a, b)) return b;
    if (type_is_subtype(b, a)) return a;
    
    // If both are integers, return the wider integer type
    if (a->kind == TYPE_INTEGER && b->kind == TYPE_INTEGER) {
        return type_integer_create(arena, a->int_size > b->int_size ? a->int_size : b->int_size);
    }
    
    // If both are floats, return the wider float type
    if (a->kind == TYPE_FLOAT && b->kind == TYPE_FLOAT) {
        return type_float_create(arena, a->float_size > b->float_size ? a->float_size : b->float_size);
    }
    
    // If one is an integer and the other is a float, return the float type
    if (a->kind == TYPE_INTEGER && b->kind == TYPE_FLOAT) {
        return b;
    }
    if (a->kind == TYPE_FLOAT && b->kind == TYPE_INTEGER) {
        return a;
    }
    
    // If both are vectors with the same size, return a vector with the common supertype of the element types
    if (a->kind == TYPE_VECTOR && b->kind == TYPE_VECTOR && a->vector.size == b->vector.size) {
        Type* element_type = type_common_supertype(arena, a->vector.element_type, b->vector.element_type);
        if (element_type) {
            return type_vector_create(arena, element_type, a->vector.size);
        }
    }
    
    // If both are functions with the same parameter count, return a function with the common supertype of the return types
    if (a->kind == TYPE_FUNCTION && b->kind == TYPE_FUNCTION && a->function.param_count == b->function.param_count) {
        // Check if parameter types are compatible
        Type** params = NULL;
        if (a->function.param_count > 0) {
            params = arena_alloc(arena, a->function.param_count * sizeof(Type*));
            if (!params) return NULL;
            
            for (size_t i = 0; i < a->function.param_count; i++) {
                params[i] = type_common_supertype(arena, a->function.params[i], b->function.params[i]);
                if (!params[i]) return NULL;
            }
        }
        
        // Get common supertype of return types
        Type* return_type = type_common_supertype(arena, a->function.return_type, b->function.return_type);
        if (!return_type) return NULL;
        
        return type_function_create(arena, a->function.param_count, params, return_type, a->function.variadic || b->function.variadic);
    }
    
    // If both are structs, return a struct with the common fields
    if (a->kind == TYPE_STRUCT && b->kind == TYPE_STRUCT) {
        // Count common fields
        size_t common_field_count = 0;
        for (size_t i = 0; i < a->structure.field_count; i++) {
            for (size_t j = 0; j < b->structure.field_count; j++) {
                if (strcmp(a->structure.fields[i].name, b->structure.fields[j].name) == 0) {
                    common_field_count++;
                    break;
                }
            }
        }
        
        // If no common fields, return any type
        if (common_field_count == 0) {
            return type_any_create(arena);
        }
        
        // Create common fields
        StructField* fields = arena_alloc(arena, common_field_count * sizeof(StructField));
        if (!fields) return NULL;
        
        size_t field_index = 0;
        for (size_t i = 0; i < a->structure.field_count; i++) {
            for (size_t j = 0; j < b->structure.field_count; j++) {
                if (strcmp(a->structure.fields[i].name, b->structure.fields[j].name) == 0) {
                    Type* field_type = type_common_supertype(arena, a->structure.fields[i].type, b->structure.fields[j].type);
                    if (!field_type) return NULL;
                    
                    fields[field_index].name = a->structure.fields[i].name;
                    fields[field_index].type = field_type;
                    field_index++;
                    break;
                }
            }
        }
        
        return type_struct_create(arena, common_field_count, fields);
    }
    
    // If both are unions, return a union of all variants
    if (a->kind == TYPE_UNION && b->kind == TYPE_UNION) {
        // Count total variants
        size_t total_variant_count = a->union_type.variant_count + b->union_type.variant_count;
        
        // Create variants array
        Type** variants = arena_alloc(arena, total_variant_count * sizeof(Type*));
        if (!variants) return NULL;
        
        // Copy variants from a
        for (size_t i = 0; i < a->union_type.variant_count; i++) {
            variants[i] = a->union_type.variants[i];
        }
        
        // Copy variants from b
        for (size_t i = 0; i < b->union_type.variant_count; i++) {
            variants[a->union_type.variant_count + i] = b->union_type.variants[i];
        }
        
        return type_union_create(arena, total_variant_count, variants);
    }
    
    // If one is a union, return a union of the other type and all variants
    if (a->kind == TYPE_UNION) {
        // Count total variants
        size_t total_variant_count = a->union_type.variant_count + 1;
        
        // Create variants array
        Type** variants = arena_alloc(arena, total_variant_count * sizeof(Type*));
        if (!variants) return NULL;
        
        // Copy variants from a
        for (size_t i = 0; i < a->union_type.variant_count; i++) {
            variants[i] = a->union_type.variants[i];
        }
        
        // Add b
        variants[a->union_type.variant_count] = b;
        
        return type_union_create(arena, total_variant_count, variants);
    }
    
    if (b->kind == TYPE_UNION) {
        // Count total variants
        size_t total_variant_count = b->union_type.variant_count + 1;
        
        // Create variants array
        Type** variants = arena_alloc(arena, total_variant_count * sizeof(Type*));
        if (!variants) return NULL;
        
        // Copy variants from b
        for (size_t i = 0; i < b->union_type.variant_count; i++) {
            variants[i] = b->union_type.variants[i];
        }
        
        // Add a
        variants[b->union_type.variant_count] = a;
        
        return type_union_create(arena, total_variant_count, variants);
    }
    
    // If no common supertype, return any type
    return type_any_create(arena);
}

/**
 * @brief Convert a type to a string
 */
char* type_to_string(Arena* arena, Type* type) {
    if (!type) return NULL;
    
    char buffer[1024];
    
    switch (type->kind) {
        case TYPE_VOID:
            snprintf(buffer, sizeof(buffer), "void");
            break;
            
        case TYPE_BOOLEAN:
            snprintf(buffer, sizeof(buffer), "boolean");
            break;
            
        case TYPE_INTEGER:
            switch (type->int_size) {
                case INT_SIZE_8:
                    snprintf(buffer, sizeof(buffer), "int8");
                    break;
                case INT_SIZE_16:
                    snprintf(buffer, sizeof(buffer), "int16");
                    break;
                case INT_SIZE_32:
                    snprintf(buffer, sizeof(buffer), "int32");
                    break;
                case INT_SIZE_64:
                    snprintf(buffer, sizeof(buffer), "int64");
                    break;
                default:
                    snprintf(buffer, sizeof(buffer), "int");
                    break;
            }
            break;
            
        case TYPE_FLOAT:
            switch (type->float_size) {
                case FLOAT_SIZE_32:
                    snprintf(buffer, sizeof(buffer), "float32");
                    break;
                case FLOAT_SIZE_64:
                    snprintf(buffer, sizeof(buffer), "float64");
                    break;
                default:
                    snprintf(buffer, sizeof(buffer), "float");
                    break;
            }
            break;
            
        case TYPE_CHAR:
            snprintf(buffer, sizeof(buffer), "char");
            break;
            
        case TYPE_STRING:
            snprintf(buffer, sizeof(buffer), "string");
            break;
            
        case TYPE_SYMBOL:
            snprintf(buffer, sizeof(buffer), "symbol");
            break;
            
        case TYPE_PAIR: {
            char* car_str = type_to_string(arena, type->function.params[0]);
            char* cdr_str = type_to_string(arena, type->function.params[1]);
            
            if (!car_str || !cdr_str) {
                return NULL;
            }
            
            snprintf(buffer, sizeof(buffer), "(Pair %s %s)", car_str, cdr_str);
            break;
        }
            
        case TYPE_VECTOR: {
            char* element_str = type_to_string(arena, type->vector.element_type);
            
            if (!element_str) {
                return NULL;
            }
            
            if (type->vector.size == 0) {
                snprintf(buffer, sizeof(buffer), "(Vector %s)", element_str);
            } else {
                snprintf(buffer, sizeof(buffer), "(Vector %s %zu)", element_str, type->vector.size);
            }
            break;
        }
            
        case TYPE_FUNCTION: {
            char* return_str = type_to_string(arena, type->function.return_type);
            
            if (!return_str) {
                return NULL;
            }
            
            char params_buffer[512] = "";
            
            for (size_t i = 0; i < type->function.param_count; i++) {
                char* param_str = type_to_string(arena, type->function.params[i]);
                
                if (!param_str) {
                    return NULL;
                }
                
                if (i > 0) {
                    strcat(params_buffer, " ");
                }
                
                strcat(params_buffer, param_str);
            }
            
            if (type->function.variadic) {
                if (type->function.param_count > 0) {
                    strcat(params_buffer, " ...");
                } else {
                    strcat(params_buffer, "...");
                }
            }
            
            snprintf(buffer, sizeof(buffer), "(-> (%s) %s)", params_buffer, return_str);
            break;
        }
            
        case TYPE_STRUCT: {
            snprintf(buffer, sizeof(buffer), "(Struct");
            
            for (size_t i = 0; i < type->structure.field_count; i++) {
                char* field_str = type_to_string(arena, type->structure.fields[i].type);
                
                if (!field_str) {
                    return NULL;
                }
                
                char field_buffer[256];
                snprintf(field_buffer, sizeof(field_buffer), " [%s : %s]", type->structure.fields[i].name, field_str);
                strcat(buffer, field_buffer);
            }
            
            strcat(buffer, ")");
            break;
        }
            
        case TYPE_UNION: {
            snprintf(buffer, sizeof(buffer), "(Union");
            
            for (size_t i = 0; i < type->union_type.variant_count; i++) {
                char* variant_str = type_to_string(arena, type->union_type.variants[i]);
                
                if (!variant_str) {
                    return NULL;
                }
                
                char variant_buffer[256];
                snprintf(variant_buffer, sizeof(variant_buffer), " %s", variant_str);
                strcat(buffer, variant_buffer);
            }
            
            strcat(buffer, ")");
            break;
        }
            
        case TYPE_ANY:
            snprintf(buffer, sizeof(buffer), "any");
            break;
            
        case TYPE_UNKNOWN:
            snprintf(buffer, sizeof(buffer), "unknown");
            break;
    }
    
    // Allocate memory for the string
    size_t len = strlen(buffer) + 1;
    char* result = arena_alloc(arena, len);
    if (!result) return NULL;
    
    // Copy the string
    memcpy(result, buffer, len);
    
    return result;
}

/**
 * @brief Parse a type from a string
 */
Type* type_from_string(Arena* arena, const char* str) {
    if (!str) return NULL;
    
    // Skip whitespace
    while (*str && isspace(*str)) {
        str++;
    }
    
    // Check for primitive types
    if (strcmp(str, "void") == 0) {
        return type_void_create(arena);
    } else if (strcmp(str, "boolean") == 0) {
        return type_boolean_create(arena);
    } else if (strcmp(str, "int") == 0 || strcmp(str, "int32") == 0) {
        return type_integer_create(arena, INT_SIZE_32);
    } else if (strcmp(str, "int8") == 0) {
        return type_integer_create(arena, INT_SIZE_8);
    } else if (strcmp(str, "int16") == 0) {
        return type_integer_create(arena, INT_SIZE_16);
    } else if (strcmp(str, "int64") == 0) {
        return type_integer_create(arena, INT_SIZE_64);
    } else if (strcmp(str, "float") == 0 || strcmp(str, "float32") == 0) {
        return type_float_create(arena, FLOAT_SIZE_32);
    } else if (strcmp(str, "float64") == 0) {
        return type_float_create(arena, FLOAT_SIZE_64);
    } else if (strcmp(str, "char") == 0) {
        return type_char_create(arena);
    } else if (strcmp(str, "string") == 0) {
        return type_string_create(arena);
    } else if (strcmp(str, "symbol") == 0) {
        return type_symbol_create(arena);
    } else if (strcmp(str, "any") == 0) {
        return type_any_create(arena);
    } else if (strcmp(str, "unknown") == 0) {
        return type_unknown_create(arena);
    }
    
    // Check for compound types
    if (str[0] == '(') {
        // Skip opening parenthesis
        str++;
        
        // Skip whitespace
        while (*str && isspace(*str)) {
            str++;
        }
        
        // Check for vector type
        if (strncmp(str, "Vector", 6) == 0) {
            str += 6;
            
            // Skip whitespace
            while (*str && isspace(*str)) {
                str++;
            }
            
            // Parse element type
            const char* end = strchr(str, ')');
            if (!end) return NULL;
            
            // Make a copy of the content between Vector and the closing parenthesis
            char vector_content[256];
            size_t content_len = end - str;
            if (content_len >= sizeof(vector_content)) return NULL;
            
            strncpy(vector_content, str, content_len);
            vector_content[content_len] = '\0';
            
            // Find the last space in the content
            char* last_space = strrchr(vector_content, ' ');
            
            // If there's a space, the size is after it
            size_t size = 0;
            char* element_type_str = vector_content;
            
            if (last_space) {
                // Null-terminate the element type string
                *last_space = '\0';
                
                // Parse the size
                char* size_str = last_space + 1;
                while (*size_str && isspace(*size_str)) {
                    size_str++;
                }
                
                if (*size_str && isdigit(*size_str)) {
                    size = atoi(size_str);
                }
            }
            
            // Parse element type
            Type* element_type = type_from_string(arena, element_type_str);
            if (!element_type) return NULL;
            
            return type_vector_create(arena, element_type, size);
        }
        
        // Check for function type
        if (strncmp(str, "->", 2) == 0) {
            str += 2;
            
            // Skip whitespace
            while (*str && isspace(*str)) {
                str++;
            }
            
            // Check for opening parenthesis
            if (*str != '(') return NULL;
            str++;
            
            // Parse parameter types
            const char* params_end = strchr(str, ')');
            if (!params_end) return NULL;
            
            // Count parameters
            size_t param_count = 0;
            bool variadic = false;
            
            // Check for variadic
            if (strstr(str, "...") != NULL) {
                variadic = true;
            }
            
            // Parse parameters
            char params_str[512];
            size_t len = params_end - str;
            if (len >= sizeof(params_str)) return NULL;
            
            strncpy(params_str, str, len);
            params_str[len] = '\0';
            
            // Count parameters
            char* token = strtok(params_str, " ");
            while (token) {
                if (strcmp(token, "...") != 0) {
                    param_count++;
                }
                token = strtok(NULL, " ");
            }
            
            // Parse return type
            str = params_end + 1;
            
            // Skip whitespace
            while (*str && isspace(*str)) {
                str++;
            }
            
            // Parse return type
            const char* return_end = strchr(str, ')');
            if (!return_end) return NULL;
            
            char return_str[256];
            len = return_end - str;
            if (len >= sizeof(return_str)) return NULL;
            
            strncpy(return_str, str, len);
            return_str[len] = '\0';
            
            // Parse return type
            Type* return_type = type_from_string(arena, return_str);
            if (!return_type) return NULL;
            
            // Parse parameter types
            Type** params = NULL;
            if (param_count > 0) {
                params = arena_alloc(arena, param_count * sizeof(Type*));
                if (!params) return NULL;
                
                // Reset params_str
                len = params_end - str;
                if (len >= sizeof(params_str)) return NULL;
                
                strncpy(params_str, str, len);
                params_str[len] = '\0';
                
                // Parse parameters
                token = strtok(params_str, " ");
                size_t i = 0;
                while (token && i < param_count) {
                    if (strcmp(token, "...") != 0) {
                        params[i] = type_from_string(arena, token);
                        if (!params[i]) return NULL;
                        i++;
                    }
                    token = strtok(NULL, " ");
                }
            }
            
            return type_function_create(arena, param_count, params, return_type, variadic);
        }
        
        // Check for pair type
        if (strncmp(str, "Pair", 4) == 0) {
            str += 4;
            
            // Skip whitespace
            while (*str && isspace(*str)) {
                str++;
            }
            
            // Parse car type
            const char* car_end = strchr(str, ' ');
            if (!car_end) return NULL;
            
            char car_str[256];
            size_t len = car_end - str;
            if (len >= sizeof(car_str)) return NULL;
            
            strncpy(car_str, str, len);
            car_str[len] = '\0';
            
            // Parse car type
            Type* car_type = type_from_string(arena, car_str);
            if (!car_type) return NULL;
            
            // Parse cdr type
            str = car_end + 1;
            
            // Skip whitespace
            while (*str && isspace(*str)) {
                str++;
            }
            
            // Parse cdr type
            const char* cdr_end = strchr(str, ')');
            if (!cdr_end) return NULL;
            
            char cdr_str[256];
            len = cdr_end - str;
            if (len >= sizeof(cdr_str)) return NULL;
            
            strncpy(cdr_str, str, len);
            cdr_str[len] = '\0';
            
            // Parse cdr type
            Type* cdr_type = type_from_string(arena, cdr_str);
            if (!cdr_type) return NULL;
            
            return type_pair_create(arena, car_type, cdr_type);
        }
    }
    
    // If we get here, we couldn't parse the type
    return type_unknown_create(arena);
}
