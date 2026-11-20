#ifndef ESHKOL_FRONTEND_TOKEN_TYPE_H
#define ESHKOL_FRONTEND_TOKEN_TYPE_H

enum TokenType {
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_QUOTE,
    TOKEN_BACKQUOTE,     // ` for quasiquote
    TOKEN_COMMA,         // , for unquote
    TOKEN_COMMA_AT,      // ,@ for unquote-splicing
    TOKEN_SYMBOL,
    TOKEN_STRING,
    TOKEN_NUMBER,
    TOKEN_BOOLEAN,
    TOKEN_CHAR,
    TOKEN_VECTOR_START,  // #( for vector literals
    TOKEN_COLON,         // : for type annotations
    TOKEN_ARROW,         // -> for function types
    TOKEN_KEYWORD,       // #:name Racket-style self-quoting keyword
    TOKEN_EOF
};

#endif
