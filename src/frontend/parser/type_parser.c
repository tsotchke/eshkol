/**
 * @file type_parser.c
 * @brief Implementation of the type parser for Eshkol
 */

#include "frontend/parser/parser.h"
#include "frontend/ast/parameter.h"
#include "core/type.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include "core/type_conversion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Forward declarations for parser functions
extern bool match(Parser* parser, TokenType type);
extern bool check(Parser* parser, TokenType type);
extern bool is_at_end(Parser* parser);
extern void consume(Parser* parser, TokenType type, const char* message);
extern void error(Parser* parser, const char* message);

// Forward declarations for type parser functions
static Type* parse_primitive_type(Parser* parser);
static Type* parse_function_type(Parser* parser);
static Type* parse_list_type(Parser* parser);
static Type* parse_vector_type(Parser* parser);

/**
 * @brief Parse a type annotation
 * 
 * @param parser The parser
 * @return The parsed type, or NULL on failure
 */
Type* parse_type(Parser* parser) {
    // Check for different type forms
    if (match(parser, TOKEN_COLON)) {
        // Type annotation starts with a colon
        
        if (match(parser, TOKEN_IDENTIFIER)) {
            StringId type_name = parser->previous.value.string_id;
            
            // Check for primitive types
            if (strcmp(type_name, "number") == 0 ||
                strcmp(type_name, "boolean") == 0 ||
                strcmp(type_name, "string") == 0 ||
                strcmp(type_name, "char") == 0 ||
                strcmp(type_name, "symbol") == 0 ||
                strcmp(type_name, "any") == 0) {
                return parse_primitive_type(parser);
            }
            
            // Check for function type
            if (strcmp(type_name, "function") == 0) {
                return parse_function_type(parser);
            }
            
            // Check for list type
            if (strcmp(type_name, "list") == 0) {
                return parse_list_type(parser);
            }
            
            // Check for vector type
            if (strcmp(type_name, "vector") == 0) {
                return parse_vector_type(parser);
            }
            
            // Unknown type
            error(parser, "Unknown type");
            return NULL;
        }
        
        error(parser, "Expected type name");
        return NULL;
    }
    
    // No type annotation
    return NULL;
}

/**
 * @brief Parse a primitive type
 * 
 * @param parser The parser
 * @return The parsed type, or NULL on failure
 */
static Type* parse_primitive_type(Parser* parser) {
    StringId type_name = parser->previous.value.string_id;
    
    TypeKind kind;
    if (strcmp(type_name, "number") == 0) {
        kind = TYPE_FLOAT;
    } else if (strcmp(type_name, "boolean") == 0) {
        kind = TYPE_BOOLEAN;
    } else if (strcmp(type_name, "string") == 0) {
        kind = TYPE_STRING;
    } else if (strcmp(type_name, "char") == 0) {
        kind = TYPE_CHAR;
    } else if (strcmp(type_name, "symbol") == 0) {
        kind = TYPE_SYMBOL;
    } else if (strcmp(type_name, "any") == 0) {
        kind = TYPE_ANY;
    } else {
        error(parser, "Unknown primitive type");
        return NULL;
    }
    
    switch (kind) {
        case TYPE_BOOLEAN:
            return type_boolean_create(parser->arena);
        case TYPE_FLOAT:
            return type_float_create(parser->arena, FLOAT_SIZE_64);
        case TYPE_CHAR:
            return type_char_create(parser->arena);
        case TYPE_STRING:
            return type_string_create(parser->arena);
        case TYPE_SYMBOL:
            return type_symbol_create(parser->arena);
        case TYPE_ANY:
            return type_any_create(parser->arena);
        default:
            return type_unknown_create(parser->arena);
    }
}

/**
 * @brief Parse a function type
 * 
 * @param parser The parser
 * @return The parsed type, or NULL on failure
 */
static Type* parse_function_type(Parser* parser) {
    // Parse the parameter types
    if (!match(parser, TOKEN_LPAREN)) {
        error(parser, "Expected '(' before parameter types");
        return NULL;
    }
    
    size_t param_count = 0;
    Type** param_types = arena_alloc(parser->arena, sizeof(Type*) * 16); // Arbitrary initial capacity
    if (!param_types) {
        error(parser, "Failed to allocate memory for parameter types");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (param_count >= 16) {
            error(parser, "Too many parameter types");
            return NULL;
        }
        
        Type* param_type = parse_type(parser);
        if (!param_type) {
            error(parser, "Expected type");
            return NULL;
        }
        
        param_types[param_count++] = param_type;
    }
    
    // Consume the closing parenthesis of the parameter types
    consume(parser, TOKEN_RPAREN, "Expected ')' after parameter types");
    
    // Parse the return type
    Type* return_type = parse_type(parser);
    if (!return_type) {
        error(parser, "Expected return type");
        return NULL;
    }
    
    return type_function_create(parser->arena, param_count, param_types, return_type, false);
}

/**
 * @brief Parse a list type
 * 
 * @param parser The parser
 * @return The parsed type, or NULL on failure
 */
static Type* parse_list_type(Parser* parser) {
    // Parse the element type
    Type* element_type = parse_type(parser);
    if (!element_type) {
        error(parser, "Expected element type");
        return NULL;
    }
    
    // Create a pair type with the element type and a nil terminator
    return type_pair_create(parser->arena, element_type, type_void_create(parser->arena));
}

/**
 * @brief Parse a vector type
 * 
 * @param parser The parser
 * @return The parsed type, or NULL on failure
 */
static Type* parse_vector_type(Parser* parser) {
    // Parse the element type
    Type* element_type = parse_type(parser);
    if (!element_type) {
        error(parser, "Expected element type");
        return NULL;
    }
    
    return type_vector_create(parser->arena, element_type, 0);
}

/**
 * @brief Parse a parameter with optional type annotation
 * 
 * @param parser The parser
 * @param arena The arena allocator
 * @return The parsed parameter, or NULL on failure
 */
Parameter* parse_parameter(Parser* parser, Arena* arena) {
    // Parse the parameter name
    if (!match(parser, TOKEN_IDENTIFIER)) {
        error(parser, "Expected parameter name");
        return NULL;
    }
    
    StringId name = parser->previous.value.string_id;
    size_t line = parser->previous.line;
    size_t column = parser->previous.column;
    
    // Parse the optional type annotation
    Type* type = parse_type(parser);
    
    return parameter_create(arena, name, type, line, column);
}

/**
 * @brief Skip a type annotation
 * 
 * @param parser The parser
 */
void skip_type_annotation(Parser* parser) {
    if (match(parser, TOKEN_COLON)) {
        // Skip the type name
        if (match(parser, TOKEN_IDENTIFIER)) {
            StringId type_name = parser->previous.value.string_id;
            
            // Skip function type parameters and return type
            if (strcmp(type_name, "function") == 0) {
                if (match(parser, TOKEN_LPAREN)) {
                    // Skip parameter types
                    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
                        skip_type_annotation(parser);
                    }
                    
                    // Skip the closing parenthesis
                    if (match(parser, TOKEN_RPAREN)) {
                        // Skip the return type
                        skip_type_annotation(parser);
                    }
                }
            }
            
            // Skip list or vector element type
            if (strcmp(type_name, "list") == 0 || strcmp(type_name, "vector") == 0) {
                skip_type_annotation(parser);
            }
        }
    }
}
