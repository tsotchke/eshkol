/**
 * @file lexer.c
 * @brief Implementation of the lexer for Eshkol
 */

#include "frontend/lexer/lexer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

/**
 * @brief Create a new lexer
 * 
 * @param arena Arena allocator
 * @param strings String table
 * @param diag Diagnostic context
 * @param source Source code
 * @return A new lexer, or NULL on failure
 */
Lexer* lexer_create(Arena* arena, StringTable* strings, DiagnosticContext* diag, const char* source) {
    assert(arena != NULL);
    assert(strings != NULL);
    assert(diag != NULL);
    assert(source != NULL);
    
    Lexer* lexer = arena_alloc(arena, sizeof(Lexer));
    if (!lexer) {
        return NULL;
    }
    
    lexer->arena = arena;
    lexer->strings = strings;
    lexer->diag = diag;
    lexer->source = source;
    lexer->start = source;
    lexer->current = source;
    lexer->line = 1;
    lexer->column = 1;
    lexer->had_error = false;
    
    return lexer;
}

/**
 * @brief Check if we've reached the end of the source
 * 
 * @param lexer The lexer
 * @return true if we've reached the end, false otherwise
 */
static bool is_at_end(Lexer* lexer) {
    return *lexer->current == '\0';
}

/**
 * @brief Advance to the next character
 * 
 * @param lexer The lexer
 * @return The character we just consumed
 */
static char advance(Lexer* lexer) {
    lexer->column++;
    return *lexer->current++;
}

/**
 * @brief Peek at the current character
 * 
 * @param lexer The lexer
 * @return The current character
 */
static char peek(Lexer* lexer) {
    return *lexer->current;
}

/**
 * @brief Peek at the next character
 * 
 * @param lexer The lexer
 * @return The next character
 */
static char peek_next(Lexer* lexer) {
    if (is_at_end(lexer)) {
        return '\0';
    }
    return lexer->current[1];
}

/**
 * @brief Check if the current character matches the expected character
 * 
 * @param lexer The lexer
 * @param expected The expected character
 * @return true if the current character matches, false otherwise
 */
static bool match(Lexer* lexer, char expected) {
    if (is_at_end(lexer)) {
        return false;
    }
    if (*lexer->current != expected) {
        return false;
    }
    
    lexer->current++;
    lexer->column++;
    return true;
}

/**
 * @brief Create a token
 * 
 * @param lexer The lexer
 * @param type The token type
 * @return The token
 */
static Token make_token(Lexer* lexer, TokenType type) {
    Token token;
    token.type = type;
    token.lexeme = lexer->start;
    token.length = (size_t)(lexer->current - lexer->start);
    token.line = lexer->line;
    token.column = lexer->column - token.length;
    
    return token;
}

/**
 * @brief Create an error token
 * 
 * @param lexer The lexer
 * @param message The error message
 * @return The error token
 */
static Token error_token(Lexer* lexer, const char* message) {
    Token token;
    token.type = TOKEN_ERROR;
    token.lexeme = message;
    token.length = strlen(message);
    token.line = lexer->line;
    token.column = lexer->column;
    
    lexer->had_error = true;
    
    // Report the error
    diagnostic_error(lexer->diag, token.line, token.column, message);
    
    return token;
}

/**
 * @brief Skip whitespace
 * 
 * @param lexer The lexer
 */
static void skip_whitespace(Lexer* lexer) {
    for (;;) {
        char c = peek(lexer);
        switch (c) {
            case ' ':
            case '\t':
            case '\r':
                advance(lexer);
                break;
            case '\n':
                lexer->line++;
                lexer->column = 0;
                advance(lexer);
                break;
            case ';':
                // Comment goes until the end of the line
                while (peek(lexer) != '\n' && !is_at_end(lexer)) {
                    advance(lexer);
                }
                break;
            default:
                return;
        }
    }
}

/**
 * @brief Check if a character is a digit
 * 
 * @param c The character
 * @return true if the character is a digit, false otherwise
 */
bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

/**
 * @brief Check if a character is a letter
 * 
 * @param c The character
 * @return true if the character is a letter, false otherwise
 */
bool is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

/**
 * @brief Check if a character is alphanumeric
 * 
 * @param c The character
 * @return true if the character is alphanumeric, false otherwise
 */
bool is_alnum(char c) {
    return is_alpha(c) || is_digit(c);
}

/**
 * @brief Check if a character is a valid identifier character
 * 
 * @param c The character
 * @return true if the character is a valid identifier character, false otherwise
 */
bool is_identifier_char(char c) {
    return is_alnum(c) || c == '-' || c == '+' || c == '*' || c == '/' || c == '<' || c == '>' ||
           c == '=' || c == '?' || c == '!' || c == '.' || c == '$' || c == '%' || c == '&' ||
           c == ':' || c == '^' || c == '~' || c == '_' || c == '@';
}

/**
 * @brief Check if a character is whitespace
 * 
 * @param c The character
 * @return true if the character is whitespace, false otherwise
 */
bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/**
 * @brief Parse a number
 * 
 * @param lexer The lexer
 * @return The number token
 */
static Token number(Lexer* lexer) {
    while (is_digit(peek(lexer))) {
        advance(lexer);
    }
    
    // Look for a decimal point
    if (peek(lexer) == '.' && is_digit(peek_next(lexer))) {
        // Consume the '.'
        advance(lexer);
        
        // Consume the fractional part
        while (is_digit(peek(lexer))) {
            advance(lexer);
        }
    }
    
    // Parse the number
    char* end;
    double value = strtod(lexer->start, &end);
    
    Token token = make_token(lexer, TOKEN_NUMBER);
    token.value.number = value;
    
    return token;
}

/**
 * @brief Parse an identifier
 * 
 * @param lexer The lexer
 * @return The identifier token
 */
static Token identifier(Lexer* lexer) {
    while (is_identifier_char(peek(lexer))) {
        advance(lexer);
    }
    
    // Check for boolean literals
    if (lexer->current - lexer->start == 2 && lexer->start[0] == '#') {
        if (lexer->start[1] == 't') {
            Token token = make_token(lexer, TOKEN_BOOLEAN);
            token.value.boolean = true;
            return token;
        } else if (lexer->start[1] == 'f') {
            Token token = make_token(lexer, TOKEN_BOOLEAN);
            token.value.boolean = false;
            return token;
        }
    }
    
    // Intern the identifier
    StringId id = string_table_intern_n(lexer->strings, lexer->start, (size_t)(lexer->current - lexer->start));
    
    Token token = make_token(lexer, TOKEN_IDENTIFIER);
    token.value.string_id = id;
    
    return token;
}

/**
 * @brief Parse a string
 * 
 * @param lexer The lexer
 * @return The string token
 */
static Token string(Lexer* lexer) {
    // Skip the opening quote
    lexer->start++;
    
    // Find the closing quote
    while (peek(lexer) != '"' && !is_at_end(lexer)) {
        if (peek(lexer) == '\n') {
            lexer->line++;
            lexer->column = 0;
        }
        advance(lexer);
    }
    
    if (is_at_end(lexer)) {
        return error_token(lexer, "Unterminated string.");
    }
    
    // Get the string content
    size_t length = (size_t)(lexer->current - lexer->start);
    
    // Intern the string
    StringId id = string_table_intern_n(lexer->strings, lexer->start, length);
    
    // Skip the closing quote
    advance(lexer);
    
    Token token = make_token(lexer, TOKEN_STRING);
    token.value.string_id = id;
    
    return token;
}

/**
 * @brief Parse a character
 * 
 * @param lexer The lexer
 * @return The character token
 */
static Token character(Lexer* lexer) {
    if (is_at_end(lexer)) {
        return error_token(lexer, "Unterminated character literal.");
    }
    
    char c = advance(lexer);
    
    Token token = make_token(lexer, TOKEN_CHARACTER);
    token.value.character = c;
    
    return token;
}

/**
 * @brief Scan the next token
 * 
 * @param lexer The lexer
 * @return The next token
 */
Token lexer_scan_token(Lexer* lexer) {
    skip_whitespace(lexer);
    
    lexer->start = lexer->current;
    
    if (is_at_end(lexer)) {
        return make_token(lexer, TOKEN_EOF);
    }
    
    char c = advance(lexer);
    
    // Check for numbers
    if (is_digit(c)) {
        return number(lexer);
    }
    
    // Check for identifiers
    if (is_alpha(c) || is_identifier_char(c)) {
        return identifier(lexer);
    }
    
    // Check for special characters
    switch (c) {
        case '(': return make_token(lexer, TOKEN_LPAREN);
        case ')': return make_token(lexer, TOKEN_RPAREN);
        case '[': return make_token(lexer, TOKEN_LBRACKET);
        case ']': return make_token(lexer, TOKEN_RBRACKET);
        case '\'': return make_token(lexer, TOKEN_QUOTE);
        case '`': return make_token(lexer, TOKEN_BACKQUOTE);
        case ',':
            if (match(lexer, '@')) {
                return make_token(lexer, TOKEN_COMMA_AT);
            }
            return make_token(lexer, TOKEN_COMMA);
        case '.': return make_token(lexer, TOKEN_DOT);
        case ':': return make_token(lexer, TOKEN_COLON);
        case '"': return string(lexer);
        case '#':
            if (match(lexer, '(')) {
                return make_token(lexer, TOKEN_VECTOR_START);
            } else if (match(lexer, '\\')) {
                return character(lexer);
            } else if (match(lexer, 't')) {
                Token token = make_token(lexer, TOKEN_BOOLEAN);
                token.value.boolean = true;
                return token;
            } else if (match(lexer, 'f')) {
                Token token = make_token(lexer, TOKEN_BOOLEAN);
                token.value.boolean = false;
                return token;
            }
            break;
    }
    
    return error_token(lexer, "Unexpected character.");
}

/**
 * @brief Check if a token is of a specific type
 * 
 * @param token The token
 * @param type The token type
 * @return true if the token is of the specified type, false otherwise
 */
bool token_is(const Token* token, TokenType type) {
    assert(token != NULL);
    return token->type == type;
}

/**
 * @brief Get the string representation of a token type
 * 
 * @param type The token type
 * @return The string representation
 */
const char* token_type_to_string(TokenType type) {
    switch (type) {
        case TOKEN_EOF: return "EOF";
        case TOKEN_LPAREN: return "(";
        case TOKEN_RPAREN: return ")";
        case TOKEN_LBRACKET: return "[";
        case TOKEN_RBRACKET: return "]";
        case TOKEN_QUOTE: return "'";
        case TOKEN_BACKQUOTE: return "`";
        case TOKEN_COMMA: return ",";
        case TOKEN_COMMA_AT: return ",@";
        case TOKEN_DOT: return ".";
        case TOKEN_COLON: return ":";
        case TOKEN_IDENTIFIER: return "IDENTIFIER";
        case TOKEN_BOOLEAN: return "BOOLEAN";
        case TOKEN_NUMBER: return "NUMBER";
        case TOKEN_CHARACTER: return "CHARACTER";
        case TOKEN_STRING: return "STRING";
        case TOKEN_VECTOR_START: return "#(";
        case TOKEN_COMMENT: return "COMMENT";
        case TOKEN_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Get the string representation of a token
 * 
 * @param token The token
 * @return The string representation
 */
const char* token_to_string(const Token* token) {
    assert(token != NULL);
    
    static char buffer[1024];
    
    switch (token->type) {
        case TOKEN_IDENTIFIER:
            snprintf(buffer, sizeof(buffer), "IDENTIFIER(%s)", token->value.string_id);
            break;
        case TOKEN_BOOLEAN:
            snprintf(buffer, sizeof(buffer), "BOOLEAN(%s)", token->value.boolean ? "#t" : "#f");
            break;
        case TOKEN_NUMBER:
            snprintf(buffer, sizeof(buffer), "NUMBER(%g)", token->value.number);
            break;
        case TOKEN_CHARACTER:
            snprintf(buffer, sizeof(buffer), "CHARACTER(%c)", token->value.character);
            break;
        case TOKEN_STRING:
            snprintf(buffer, sizeof(buffer), "STRING(\"%s\")", token->value.string_id);
            break;
        default:
            return token_type_to_string(token->type);
    }
    
    return buffer;
}
