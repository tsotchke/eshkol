/**
 * @file type_conversion.c
 * @brief Implementation of type conversion functions
 */

#include "core/type_conversion.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

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

/**
 * @brief Check if a type can be converted to another type
 */
bool type_can_convert(Type* from, Type* to) {
    if (!from || !to) return false;
    
    // Same type
    if (type_equals(from, to)) return true;
    
    // Any type can be converted to void
    if (to->kind == TYPE_VOID) return true;
    
    // Any type can be converted to any
    if (to->kind == TYPE_ANY) return true;
    
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
    
    // String to any
    if (from->kind == TYPE_STRING && to->kind == TYPE_ANY) {
        return true;
    }
    
    // Numeric to any
    if ((from->kind == TYPE_INTEGER || from->kind == TYPE_FLOAT) && to->kind == TYPE_ANY) {
        return true;
    }
    
    // Subtype relationship
    if (type_is_subtype(from, to)) {
        return true;
    }
    
    return false;
}

/**
 * @brief Apply type conversion to a value
 */
char* type_apply_conversion(Arena* arena, const char* expr, Type* from, Type* to) {
    assert(arena != NULL);
    assert(expr != NULL);
    assert(from != NULL);
    assert(to != NULL);
    
    // Same type, no conversion needed
    if (type_equals(from, to)) {
        char* result = arena_alloc(arena, strlen(expr) + 1);
        if (!result) return NULL;
        strcpy(result, expr);
        return result;
    }
    
    // Generate conversion code
    char buffer[1024];
    
    // Integer to float
    if (from->kind == TYPE_INTEGER && to->kind == TYPE_FLOAT) {
        snprintf(buffer, sizeof(buffer), "(float)(%s)", expr);
    }
    // Float to integer
    else if (from->kind == TYPE_FLOAT && to->kind == TYPE_INTEGER) {
        snprintf(buffer, sizeof(buffer), "(int)(%s)", expr);
    }
    // Char to integer
    else if (from->kind == TYPE_CHAR && to->kind == TYPE_INTEGER) {
        snprintf(buffer, sizeof(buffer), "(int)(%s)", expr);
    }
    // Integer to char
    else if (from->kind == TYPE_INTEGER && to->kind == TYPE_CHAR) {
        snprintf(buffer, sizeof(buffer), "(char)(%s)", expr);
    }
    // String to any
    else if (from->kind == TYPE_STRING && to->kind == TYPE_ANY) {
        snprintf(buffer, sizeof(buffer), "(void*)(char*)(%s)", expr);
    }
    // Numeric to any
    else if ((from->kind == TYPE_INTEGER || from->kind == TYPE_FLOAT) && to->kind == TYPE_ANY) {
        const char* type_str = from->kind == TYPE_INTEGER ? "int" : "float";
        snprintf(buffer, sizeof(buffer), "({ %s temp = (%s); (void*)&temp; })", type_str, expr);
    }
    // Any other conversion
    else {
        char* to_str = type_to_string(arena, to);
        if (!to_str) return NULL;
        
        snprintf(buffer, sizeof(buffer), "(%s)(%s)", to_str, expr);
    }
    
    // Allocate memory for the result
    char* result = arena_alloc(arena, strlen(buffer) + 1);
    if (!result) return NULL;
    
    // Copy the result
    strcpy(result, buffer);
    
    return result;
}
