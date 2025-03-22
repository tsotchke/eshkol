/**
 * @file lexer.h
 * @brief Lexer for Eshkol
 * 
 * This file defines the lexer for the Eshkol language, which tokenizes
 * source code into a stream of tokens.
 */

#ifndef ESHKOL_LEXER_H
#define ESHKOL_LEXER_H

#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include <stddef.h>
#include <stdbool.h>

/**
 * @brief String identifier type
 * 
 * This is a pointer to an interned string in the string table.
 */
typedef const char* StringId;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Token types
 */
typedef enum {
    TOKEN_EOF,              /**< End of file */
    TOKEN_LPAREN,           /**< Left parenthesis '(' */
    TOKEN_RPAREN,           /**< Right parenthesis ')' */
    TOKEN_LBRACKET,         /**< Left bracket '[' */
    TOKEN_RBRACKET,         /**< Right bracket ']' */
    TOKEN_QUOTE,            /**< Quote ''' */
    TOKEN_BACKQUOTE,        /**< Backquote '`' */
    TOKEN_COMMA,            /**< Comma ',' */
    TOKEN_COMMA_AT,         /**< Comma-at ',@' */
    TOKEN_DOT,              /**< Dot '.' */
    TOKEN_COLON,            /**< Colon ':' */
    TOKEN_IDENTIFIER,       /**< Identifier */
    TOKEN_BOOLEAN,          /**< Boolean literal (#t or #f) */
    TOKEN_NUMBER,           /**< Number literal */
    TOKEN_CHARACTER,        /**< Character literal */
    TOKEN_STRING,           /**< String literal */
    TOKEN_VECTOR_START,     /**< Vector start '#(' */
    TOKEN_COMMENT,          /**< Comment */
    TOKEN_ERROR,            /**< Error token */
} TokenType;

/**
 * @brief Token structure
 */
typedef struct {
    TokenType type;         /**< Token type */
    const char* lexeme;     /**< Token lexeme */
    size_t length;          /**< Token length */
    size_t line;            /**< Line number */
    size_t column;          /**< Column number */
    union {
        double number;      /**< Number value */
        bool boolean;       /**< Boolean value */
        char character;     /**< Character value */
        StringId string_id; /**< String value */
    } value;                /**< Token value */
} Token;

/**
 * @brief Lexer structure
 */
typedef struct {
    Arena* arena;           /**< Arena allocator */
    StringTable* strings;   /**< String table */
    DiagnosticContext* diag; /**< Diagnostic context */
    const char* source;     /**< Source code */
    const char* start;      /**< Start of current token */
    const char* current;    /**< Current position in source */
    size_t line;            /**< Current line number */
    size_t column;          /**< Current column number */
    bool had_error;         /**< Whether an error occurred */
} Lexer;

/**
 * @brief Create a new lexer
 * 
 * @param arena Arena allocator
 * @param strings String table
 * @param diag Diagnostic context
 * @param source Source code
 * @return A new lexer, or NULL on failure
 */
Lexer* lexer_create(Arena* arena, StringTable* strings, DiagnosticContext* diag, const char* source);

/**
 * @brief Scan the next token
 * 
 * @param lexer The lexer
 * @return The next token
 */
Token lexer_scan_token(Lexer* lexer);

/**
 * @brief Check if a token is of a specific type
 * 
 * @param token The token
 * @param type The token type
 * @return true if the token is of the specified type, false otherwise
 */
bool token_is(const Token* token, TokenType type);

/**
 * @brief Get the string representation of a token type
 * 
 * @param type The token type
 * @return The string representation
 */
const char* token_type_to_string(TokenType type);

/**
 * @brief Get the string representation of a token
 * 
 * @param token The token
 * @return The string representation
 */
const char* token_to_string(const Token* token);

/**
 * @brief Check if a character is a digit
 * 
 * @param c The character
 * @return true if the character is a digit, false otherwise
 */
bool is_digit(char c);

/**
 * @brief Check if a character is a letter
 * 
 * @param c The character
 * @return true if the character is a letter, false otherwise
 */
bool is_alpha(char c);

/**
 * @brief Check if a character is alphanumeric
 * 
 * @param c The character
 * @return true if the character is alphanumeric, false otherwise
 */
bool is_alnum(char c);

/**
 * @brief Check if a character is a valid identifier character
 * 
 * @param c The character
 * @return true if the character is a valid identifier character, false otherwise
 */
bool is_identifier_char(char c);

/**
 * @brief Check if a character is whitespace
 * 
 * @param c The character
 * @return true if the character is whitespace, false otherwise
 */
bool is_whitespace(char c);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_LEXER_H */
