/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#include <string.h>
#include <cctype>
#include <sstream>
#include <vector>
#include <set>

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
    TOKEN_EOF
};

struct Token {
    TokenType type;
    std::string value;
    size_t pos;
    uint32_t line;    // 1-based line number
    uint32_t column;  // 1-based column number
};

class SchemeTokenizer {
private:
    std::string input;
    size_t pos;
    size_t length;
    uint32_t line_;
    uint32_t column_;
    size_t line_start_;  // Position of current line start for column calculation
    std::vector<Token> pushback_buffer;  // Buffer for pushed back tokens

public:
    SchemeTokenizer(const std::string& text)
        : input(text), pos(0), length(text.length()), line_(1), column_(1), line_start_(0) {}

    // Push a token back to be returned by the next nextToken() call
    void pushBack(const Token& token) {
        pushback_buffer.push_back(token);
    }

    // Peek at the next token without consuming it
    Token peekToken() {
        Token t = nextToken();
        pushBack(t);
        return t;
    }

    Token nextToken() {
        // First check pushback buffer
        if (!pushback_buffer.empty()) {
            Token t = pushback_buffer.back();
            pushback_buffer.pop_back();
            return t;
        }

        skipWhitespace();

        if (pos >= length) {
            return {TOKEN_EOF, "", pos, line_, column_};
        }

        // Save current location for the token
        uint32_t tok_line = line_;
        uint32_t tok_col = column_;

        char ch = input[pos];

        switch (ch) {
            case '(':
                pos++;
                column_++;
                return {TOKEN_LPAREN, "(", pos - 1, tok_line, tok_col};
            case ')':
                pos++;
                column_++;
                return {TOKEN_RPAREN, ")", pos - 1, tok_line, tok_col};
            case '\'':
                pos++;
                column_++;
                return {TOKEN_QUOTE, "'", pos - 1, tok_line, tok_col};
            case '`':
                pos++;
                column_++;
                return {TOKEN_BACKQUOTE, "`", pos - 1, tok_line, tok_col};
            case ',':
                pos++;
                column_++;
                // Check for ,@ (unquote-splicing)
                if (pos < length && input[pos] == '@') {
                    pos++;
                    column_++;
                    return {TOKEN_COMMA_AT, ",@", pos - 2, tok_line, tok_col};
                }
                return {TOKEN_COMMA, ",", pos - 1, tok_line, tok_col};
            case ':':
                pos++;
                column_++;
                return {TOKEN_COLON, ":", pos - 1, tok_line, tok_col};
            case '"':
                return readString();
            default:
                // Check for arrow type: ->
                if (ch == '-' && pos + 1 < length && input[pos + 1] == '>') {
                    pos += 2;
                    column_ += 2;
                    return {TOKEN_ARROW, "->", pos - 2, tok_line, tok_col};
                }
                if (std::isdigit(ch) || (ch == '-' && pos + 1 < length && std::isdigit(input[pos + 1]))) {
                    return readNumber();
                } else if (ch == '#') {
                    return readBoolean();
                } else {
                    return readSymbol();
                }
        }
    }
    
private:
    void skipWhitespace() {
        while (pos < length) {
            // Skip whitespace
            if (std::isspace(input[pos])) {
                if (input[pos] == '\n') {
                    line_++;
                    pos++;
                    line_start_ = pos;
                } else {
                    pos++;
                }
                continue;
            }
            // Skip comments (from ; to end of line)
            if (input[pos] == ';') {
                while (pos < length && input[pos] != '\n') {
                    pos++;
                }
                continue;
            }
            // Not whitespace or comment, stop
            break;
        }
        // Update column after whitespace
        column_ = static_cast<uint32_t>(pos - line_start_ + 1);
    }
    
    Token readString() {
        size_t start = pos;
        uint32_t tok_line = line_;
        uint32_t tok_col = column_;
        pos++; // skip opening quote
        column_++;
        std::string value;

        while (pos < length && input[pos] != '"') {
            if (input[pos] == '\\' && pos + 1 < length) {
                pos++;
                column_++;
                switch (input[pos]) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    default: value += input[pos]; break;
                }
            } else if (input[pos] == '\n') {
                value += input[pos];
                line_++;
                line_start_ = pos + 1;
            } else {
                value += input[pos];
            }
            pos++;
            column_ = static_cast<uint32_t>(pos - line_start_ + 1);
        }

        if (pos < length) {
            pos++; // skip closing quote
            column_++;
        }
        return {TOKEN_STRING, value, start, tok_line, tok_col};
    }
    
    Token readNumber() {
        size_t start = pos;
        uint32_t tok_line = line_;
        uint32_t tok_col = column_;
        std::string value;

        if (input[pos] == '-') {
            value += input[pos++];
            column_++;
        }

        // Read integer or decimal part
        while (pos < length && (std::isdigit(input[pos]) || input[pos] == '.')) {
            value += input[pos++];
            column_++;
        }

        // Handle scientific notation (e.g., 1e-7, 2.5E+10, 3e4)
        if (pos < length && (input[pos] == 'e' || input[pos] == 'E')) {
            value += input[pos++];
            column_++;
            // Handle optional sign after 'e'
            if (pos < length && (input[pos] == '+' || input[pos] == '-')) {
                value += input[pos++];
                column_++;
            }
            // Read exponent digits
            while (pos < length && std::isdigit(input[pos])) {
                value += input[pos++];
                column_++;
            }
        }

        return {TOKEN_NUMBER, value, start, tok_line, tok_col};
    }
    
    Token readBoolean() {
        size_t start = pos;
        uint32_t tok_line = line_;
        uint32_t tok_col = column_;
        pos++; // skip #
        column_++;

        // Check for vector literal: #(
        if (pos < length && input[pos] == '(') {
            pos++;  // skip (
            column_++;
            return {TOKEN_VECTOR_START, "#(", start, tok_line, tok_col};
        }

        if (pos < length && (input[pos] == 't' || input[pos] == 'f')) {
            std::string value = std::string("#") + input[pos];
            pos++;
            column_++;
            return {TOKEN_BOOLEAN, value, start, tok_line, tok_col};
        }

        // Check for character literal: #\x
        if (pos < length && input[pos] == '\\') {
            pos++; // skip backslash
            column_++;
            if (pos < length) {
                // Check for named characters like #\space, #\newline, #\tab
                if (pos + 4 < length && input.substr(pos, 5) == "space") {
                    pos += 5;
                    column_ += 5;
                    return {TOKEN_CHAR, " ", start, tok_line, tok_col};
                } else if (pos + 6 < length && input.substr(pos, 7) == "newline") {
                    pos += 7;
                    column_ += 7;
                    return {TOKEN_CHAR, "\n", start, tok_line, tok_col};
                } else if (pos + 2 < length && input.substr(pos, 3) == "tab") {
                    pos += 3;
                    column_ += 3;
                    return {TOKEN_CHAR, "\t", start, tok_line, tok_col};
                } else {
                    // Single character
                    std::string value(1, input[pos]);
                    pos++;
                    column_++;
                    return {TOKEN_CHAR, value, start, tok_line, tok_col};
                }
            }
        }

        // Invalid boolean, treat as symbol
        pos = start;
        column_ = tok_col;
        return readSymbol();
    }
    
    Token readSymbol() {
        size_t start = pos;
        uint32_t tok_line = line_;
        uint32_t tok_col = column_;
        std::string value;

        while (pos < length && !std::isspace(input[pos]) &&
               input[pos] != '(' && input[pos] != ')' &&
               input[pos] != '\'' && input[pos] != '"' &&
               input[pos] != ':') {  // Stop at colon for type annotations
            // Note: We allow '->' within symbols (e.g., number->string)
            // The TOKEN_ARROW is only recognized at the start of a new token
            value += input[pos++];
            column_++;
        }

        return {TOKEN_SYMBOL, value, start, tok_line, tok_col};
    }
};

static eshkol_ast_t parse_atom(const Token& token) {
    eshkol_ast_t ast = {};  // Zero-initialize all fields
    ast.type = ESHKOL_INVALID;
    ast.line = token.line;
    ast.column = token.column;

    switch (token.type) {
        case TOKEN_STRING:
            ast.type = ESHKOL_STRING;
            ast.str_val.size = token.value.length() + 1;
            ast.str_val.ptr = new char[ast.str_val.size];
            if (ast.str_val.ptr) {
                strcpy(ast.str_val.ptr, token.value.c_str());
            }
            break;

        case TOKEN_NUMBER: {
            // Check if it's a floating-point number (has '.' or scientific notation 'e'/'E')
            if (token.value.find('.') != std::string::npos ||
                token.value.find('e') != std::string::npos ||
                token.value.find('E') != std::string::npos) {
                ast.type = ESHKOL_DOUBLE;
                ast.double_val = std::stod(token.value);
            } else {
                ast.type = ESHKOL_INT64;
                ast.int64_val = std::stoll(token.value);
            }
            break;
        }

        case TOKEN_BOOLEAN:
            ast.type = ESHKOL_BOOL;
            ast.int64_val = (token.value == "#t") ? 1 : 0;
            break;

        case TOKEN_CHAR:
            ast.type = ESHKOL_CHAR;
            // Store character as Unicode codepoint (ASCII for single byte)
            ast.int64_val = static_cast<unsigned char>(token.value[0]);
            break;

        case TOKEN_SYMBOL:
            ast.type = ESHKOL_VAR;
            ast.variable.id = new char[token.value.length() + 1];
            if (ast.variable.id) {
                strcpy(ast.variable.id, token.value.c_str());
            }
            ast.variable.data = nullptr;
            break;

        default:
            break;
    }

    return ast;
}

static eshkol_op_t get_operator_type(const std::string& op) {
    if (op == "if") return ESHKOL_IF_OP;
    if (op == "lambda") return ESHKOL_LAMBDA_OP;
    if (op == "let") return ESHKOL_LET_OP;
    if (op == "let*") return ESHKOL_LET_STAR_OP;
    if (op == "letrec") return ESHKOL_LETREC_OP;
    if (op == "and") return ESHKOL_AND_OP;
    if (op == "or") return ESHKOL_OR_OP;
    if (op == "cond") return ESHKOL_COND_OP;
    if (op == "case") return ESHKOL_CASE_OP;
    if (op == "match") return ESHKOL_MATCH_OP;
    if (op == "do") return ESHKOL_DO_OP;
    if (op == "when") return ESHKOL_WHEN_OP;
    if (op == "unless") return ESHKOL_UNLESS_OP;
    if (op == "quote") return ESHKOL_QUOTE_OP;
    if (op == "quasiquote") return ESHKOL_QUASIQUOTE_OP;
    if (op == "unquote") return ESHKOL_UNQUOTE_OP;
    if (op == "unquote-splicing") return ESHKOL_UNQUOTE_SPLICING_OP;
    // Note: "compose" is now a user-definable higher-order function, not a special op
    if (op == "define") return ESHKOL_DEFINE_OP;
    if (op == "define-type") return ESHKOL_DEFINE_TYPE_OP;
    if (op == "define-syntax") return ESHKOL_DEFINE_SYNTAX_OP;
    if (op == "set!") return ESHKOL_SET_OP;
    if (op == "import") return ESHKOL_IMPORT_OP;
    if (op == "require") return ESHKOL_REQUIRE_OP;  // module system: (require module.name ...)
    if (op == "provide") return ESHKOL_PROVIDE_OP;  // module system: (provide name1 name2 ...)
    // Memory management operators (OALR - Ownership-Aware Lexical Regions)
    if (op == "with-region") return ESHKOL_WITH_REGION_OP;
    if (op == "owned") return ESHKOL_OWNED_OP;
    if (op == "move") return ESHKOL_MOVE_OP;
    if (op == "borrow") return ESHKOL_BORROW_OP;
    if (op == "shared") return ESHKOL_SHARED_OP;
    if (op == "weak-ref") return ESHKOL_WEAK_REF_OP;
    if (op == "extern") return ESHKOL_EXTERN_OP;
    if (op == "extern-var") return ESHKOL_EXTERN_VAR_OP;
    if (op == "tensor") return ESHKOL_TENSOR_OP;
    // Note: "vector" is now a function call for Scheme vectors (heterogeneous)
    // Use "tensor" for homogeneous numerical arrays
    if (op == "matrix") return ESHKOL_TENSOR_OP;  // matrix is just 2D tensor
    if (op == "diff") return ESHKOL_DIFF_OP;
    // Automatic differentiation operators
    if (op == "derivative") return ESHKOL_DERIVATIVE_OP;
    if (op == "gradient") return ESHKOL_GRADIENT_OP;
    if (op == "jacobian") return ESHKOL_JACOBIAN_OP;
    if (op == "hessian") return ESHKOL_HESSIAN_OP;
    if (op == "divergence") return ESHKOL_DIVERGENCE_OP;
    if (op == "curl") return ESHKOL_CURL_OP;
    if (op == "laplacian") return ESHKOL_LAPLACIAN_OP;
    if (op == "directional-derivative") return ESHKOL_DIRECTIONAL_DERIV_OP;
    // Exception handling operators
    if (op == "guard") return ESHKOL_GUARD_OP;
    if (op == "raise") return ESHKOL_RAISE_OP;
    // Multiple return values operations
    if (op == "values") return ESHKOL_VALUES_OP;
    if (op == "call-with-values") return ESHKOL_CALL_WITH_VALUES_OP;
    if (op == "let-values") return ESHKOL_LET_VALUES_OP;
    if (op == "let*-values") return ESHKOL_LET_STAR_VALUES_OP;
    // Treat arithmetic operations as special CALL_OPs so they get proper argument handling
    // We'll store the operation type in the function name and handle display in the printer
    return ESHKOL_CALL_OP;
}

// Forward declarations
static eshkol_ast_t parse_quoted_data(SchemeTokenizer& tokenizer);
static eshkol_ast_t parse_quoted_data_with_token(SchemeTokenizer& tokenizer, Token token);
static eshkol_ast_t parse_quoted_list_internal(SchemeTokenizer& tokenizer);
static hott_type_expr_t* parseTypeExpression(SchemeTokenizer& tokenizer);

// ===== HoTT TYPE EXPRESSION PARSING =====
// Parse type expressions for the HoTT type system
// Supports: primitive types, arrow types, container types, forall, etc.

// Parse a primitive type name and return the corresponding type expression
static hott_type_expr_t* parsePrimitiveType(const std::string& name) {
    // Check for primitive types (case-insensitive)
    std::string lower = name;
    for (auto& c : lower) c = std::tolower(c);

    if (lower == "integer" || lower == "int" || lower == "int64") {
        return hott_make_integer_type();
    } else if (lower == "real" || lower == "float" || lower == "double" || lower == "float64") {
        return hott_make_real_type();
    } else if (lower == "boolean" || lower == "bool") {
        return hott_make_boolean_type();
    } else if (lower == "string" || lower == "str") {
        return hott_make_string_type();
    } else if (lower == "char" || lower == "character") {
        return hott_make_char_type();
    } else if (lower == "symbol") {
        return hott_make_symbol_type();
    } else if (lower == "null" || lower == "nil") {
        return hott_make_null_type();
    } else if (lower == "any") {
        return hott_make_any_type();
    } else if (lower == "nothing" || lower == "never") {
        return hott_make_nothing_type();
    } else {
        // Treat as type variable (lowercase letters starting with a-z)
        return hott_make_type_var(name.c_str());
    }
}

// Parse a type expression from the tokenizer
// Handles: primitive types, arrow types, list/vector types, forall, etc.
static hott_type_expr_t* parseTypeExpression(SchemeTokenizer& tokenizer) {
    Token token = tokenizer.nextToken();

    if (token.type == TOKEN_SYMBOL) {
        // Simple type name or type variable
        return parsePrimitiveType(token.value);
    }

    if (token.type == TOKEN_ARROW) {
        // Shorthand: -> without parens is a type constructor
        // This shouldn't happen in well-formed input, treat as error
        eshkol_error("Unexpected -> in type expression");
        return nullptr;
    }

    if (token.type == TOKEN_LPAREN) {
        // Compound type expression
        Token first = tokenizer.nextToken();

        if (first.type == TOKEN_RPAREN) {
            // Empty parens () - treat as null/unit type
            return hott_make_null_type();
        }

        if (first.type == TOKEN_ARROW) {
            // Arrow type: (-> param1 param2 ... return)
            std::vector<hott_type_expr_t*> param_types;

            while (true) {
                Token peek = tokenizer.peekToken();
                if (peek.type == TOKEN_RPAREN) {
                    tokenizer.nextToken();  // consume the )
                    // End of arrow type - last element was return type
                    if (param_types.empty()) {
                        eshkol_error("Arrow type requires at least a return type");
                        return nullptr;
                    }
                    // Pop last as return type
                    hott_type_expr_t* return_type = param_types.back();
                    param_types.pop_back();

                    hott_type_expr_t* result = hott_make_arrow_type(
                        param_types.data(), param_types.size(), return_type);

                    // Free temporary param types and return type
                    for (auto* p : param_types) hott_free_type_expr(p);
                    hott_free_type_expr(return_type);

                    return result;
                }
                if (peek.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in arrow type");
                    for (auto* p : param_types) hott_free_type_expr(p);
                    return nullptr;
                }

                // Parse next type in the arrow (recursive call handles all cases)
                hott_type_expr_t* next_type = parseTypeExpression(tokenizer);
                if (!next_type) {
                    for (auto* p : param_types) hott_free_type_expr(p);
                    return nullptr;
                }
                param_types.push_back(next_type);
            }
        }

        if (first.type == TOKEN_SYMBOL) {
            std::string type_name = first.value;
            std::string lower = type_name;
            for (auto& c : lower) c = std::tolower(c);

            if (lower == "list") {
                // (list element-type)
                hott_type_expr_t* elem = parseTypeExpression(tokenizer);
                Token rparen = tokenizer.nextToken();
                if (rparen.type != TOKEN_RPAREN) {
                    eshkol_error("Expected ) after list element type");
                    hott_free_type_expr(elem);
                    return nullptr;
                }
                hott_type_expr_t* result = hott_make_list_type(elem);
                hott_free_type_expr(elem);
                return result;
            }

            if (lower == "vector") {
                // (vector element-type)
                hott_type_expr_t* elem = parseTypeExpression(tokenizer);
                Token rparen = tokenizer.nextToken();
                if (rparen.type != TOKEN_RPAREN) {
                    eshkol_error("Expected ) after vector element type");
                    hott_free_type_expr(elem);
                    return nullptr;
                }
                hott_type_expr_t* result = hott_make_vector_type(elem);
                hott_free_type_expr(elem);
                return result;
            }

            if (lower == "tensor") {
                // (tensor element-type) - same as vector but for multi-dimensional arrays
                hott_type_expr_t* elem = parseTypeExpression(tokenizer);
                Token rparen = tokenizer.nextToken();
                if (rparen.type != TOKEN_RPAREN) {
                    eshkol_error("Expected ) after tensor element type");
                    hott_free_type_expr(elem);
                    return nullptr;
                }
                hott_type_expr_t* result = hott_make_tensor_type(elem);
                hott_free_type_expr(elem);
                return result;
            }

            if (lower == "pair") {
                // (pair left right)
                hott_type_expr_t* left = parseTypeExpression(tokenizer);
                hott_type_expr_t* right = parseTypeExpression(tokenizer);
                Token rparen = tokenizer.nextToken();
                if (rparen.type != TOKEN_RPAREN) {
                    eshkol_error("Expected ) after pair types");
                    hott_free_type_expr(left);
                    hott_free_type_expr(right);
                    return nullptr;
                }
                hott_type_expr_t* result = hott_make_pair_type(left, right);
                hott_free_type_expr(left);
                hott_free_type_expr(right);
                return result;
            }

            if (lower == "*" || lower == "product") {
                // (* left right)
                hott_type_expr_t* left = parseTypeExpression(tokenizer);
                hott_type_expr_t* right = parseTypeExpression(tokenizer);
                Token rparen = tokenizer.nextToken();
                if (rparen.type != TOKEN_RPAREN) {
                    eshkol_error("Expected ) after product types");
                    hott_free_type_expr(left);
                    hott_free_type_expr(right);
                    return nullptr;
                }
                hott_type_expr_t* result = hott_make_product_type(left, right);
                hott_free_type_expr(left);
                hott_free_type_expr(right);
                return result;
            }

            if (lower == "+" || lower == "sum" || lower == "either") {
                // (+ left right)
                hott_type_expr_t* left = parseTypeExpression(tokenizer);
                hott_type_expr_t* right = parseTypeExpression(tokenizer);
                Token rparen = tokenizer.nextToken();
                if (rparen.type != TOKEN_RPAREN) {
                    eshkol_error("Expected ) after sum types");
                    hott_free_type_expr(left);
                    hott_free_type_expr(right);
                    return nullptr;
                }
                hott_type_expr_t* result = hott_make_sum_type(left, right);
                hott_free_type_expr(left);
                hott_free_type_expr(right);
                return result;
            }

            if (lower == "forall") {
                // (forall (a b ...) body-type)
                Token vars_start = tokenizer.nextToken();
                if (vars_start.type != TOKEN_LPAREN) {
                    eshkol_error("Expected ( after forall");
                    return nullptr;
                }

                std::vector<char*> type_vars;
                while (true) {
                    Token var = tokenizer.nextToken();
                    if (var.type == TOKEN_RPAREN) break;
                    if (var.type != TOKEN_SYMBOL) {
                        eshkol_error("Expected type variable name in forall");
                        for (auto* v : type_vars) free(v);
                        return nullptr;
                    }
                    type_vars.push_back(strdup(var.value.c_str()));
                }

                hott_type_expr_t* body = parseTypeExpression(tokenizer);
                Token rparen = tokenizer.nextToken();
                if (rparen.type != TOKEN_RPAREN) {
                    eshkol_error("Expected ) after forall body");
                    for (auto* v : type_vars) free(v);
                    hott_free_type_expr(body);
                    return nullptr;
                }

                hott_type_expr_t* result = hott_make_forall_type(
                    type_vars.data(), type_vars.size(), body);

                for (auto* v : type_vars) free(v);
                hott_free_type_expr(body);
                return result;
            }

            // Unknown type constructor - treat as type application or just type name
            // Skip remaining tokens until )
            int depth = 1;
            while (depth > 0) {
                Token t = tokenizer.nextToken();
                if (t.type == TOKEN_LPAREN) depth++;
                else if (t.type == TOKEN_RPAREN) depth--;
                else if (t.type == TOKEN_EOF) break;
            }
            return parsePrimitiveType(type_name);
        }

        // Unexpected token in type expression
        eshkol_error("Unexpected token in type expression");
        return nullptr;
    }

    // Unexpected token
    eshkol_error("Expected type expression");
    return nullptr;
}

// ===== END HoTT TYPE EXPRESSION PARSING =====

// Parse quoted data - allows any expression including data lists like (1 2 3)
// This is called AFTER consuming the opening token (QUOTE or first element of quote form)
static eshkol_ast_t parse_quoted_data(SchemeTokenizer& tokenizer) {
    Token token = tokenizer.nextToken();
    return parse_quoted_data_with_token(tokenizer, token);
}

// Parse quoted data when we already have the first token
static eshkol_ast_t parse_quoted_data_with_token(SchemeTokenizer& tokenizer, Token token) {
    if (token.type == TOKEN_LPAREN) {
        // Parse a list without requiring a symbol as first element
        return parse_quoted_list_internal(tokenizer);
    } else if (token.type == TOKEN_QUOTE) {
        // Nested quote - parse recursively
        eshkol_ast_t quoted = parse_quoted_data(tokenizer);
        eshkol_ast_t ast;
        ast.type = ESHKOL_OP;
        ast.operation.op = ESHKOL_QUOTE_OP;
        ast.operation.call_op.func = nullptr;
        ast.operation.call_op.num_vars = 1;
        ast.operation.call_op.variables = new eshkol_ast_t[1];
        ast.operation.call_op.variables[0] = quoted;
        return ast;
    } else {
        // Atom
        return parse_atom(token);
    }
}

// Parse a quoted list - called after consuming '('
static eshkol_ast_t parse_quoted_list_internal(SchemeTokenizer& tokenizer) {
    std::vector<eshkol_ast_t> elements;

    while (true) {
        Token inner_token = tokenizer.nextToken();
        if (inner_token.type == TOKEN_RPAREN) break;
        if (inner_token.type == TOKEN_EOF) {
            eshkol_error("Unexpected end of input in quoted list");
            return {.type = ESHKOL_INVALID};
        }

        // Recursively parse each element (handles arbitrary nesting)
        eshkol_ast_t elem = parse_quoted_data_with_token(tokenizer, inner_token);
        if (elem.type == ESHKOL_INVALID) {
            return elem;
        }
        elements.push_back(elem);
    }

    // Build list as CALL_OP with "list" as function
    // This allows codegenQuotedAST to handle it correctly
    eshkol_ast_t ast;
    ast.type = ESHKOL_OP;
    ast.operation.op = ESHKOL_CALL_OP;
    ast.operation.call_op.func = new eshkol_ast_t;
    ast.operation.call_op.func->type = ESHKOL_VAR;
    ast.operation.call_op.func->variable.id = new char[5];
    strcpy(ast.operation.call_op.func->variable.id, "list");
    ast.operation.call_op.func->variable.data = nullptr;
    ast.operation.call_op.num_vars = elements.size();
    if (elements.size() > 0) {
        ast.operation.call_op.variables = new eshkol_ast_t[elements.size()];
        for (size_t i = 0; i < elements.size(); i++) {
            ast.operation.call_op.variables[i] = elements[i];
        }
    } else {
        ast.operation.call_op.variables = nullptr;
    }
    return ast;
}

// Forward declaration for scope tracking
class ScopeTracker {
private:
    std::vector<std::set<std::string>> scope_stack;
    
public:
    ScopeTracker() {
        // Push global scope
        scope_stack.push_back(std::set<std::string>());
    }
    
    void pushScope() {
        scope_stack.push_back(std::set<std::string>());
    }
    
    void popScope() {
        if (scope_stack.size() > 1) {  // Keep global scope
            scope_stack.pop_back();
        }
    }
    
    void addVariable(const std::string& var) {
        if (!scope_stack.empty()) {
            scope_stack.back().insert(var);
        }
    }
    
    bool isInCurrentScope(const std::string& var) const {
        if (scope_stack.empty()) return false;
        return scope_stack.back().count(var) > 0;
    }
    
    bool isInAnyParentScope(const std::string& var) const {
        // Check all scopes except current
        for (size_t i = 0; i + 1 < scope_stack.size(); i++) {
            if (scope_stack[i].count(var) > 0) {
                return true;
            }
        }
        return false;
    }
    
    std::set<std::string> getAllParentScopeVars() const {
        std::set<std::string> all_vars;
        // Collect from all scopes except current
        for (size_t i = 0; i + 1 < scope_stack.size(); i++) {
            all_vars.insert(scope_stack[i].begin(), scope_stack[i].end());
        }
        return all_vars;
    }
};

// Global scope tracker (will be reset for each file)
static ScopeTracker g_scope_tracker;


// ===== CLOSURE CAPTURE ANALYSIS =====
// Static AST analysis for closure capture detection

// Helper: Collect all defined variables at a given AST level (not recursing into nested functions)
static void collectDefinedVariables(const eshkol_ast_t* ast, std::set<std::string>& defined_vars) {
    if (!ast) return;
    
    // Only look at top-level defines, not nested ones
    if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_DEFINE_OP) {
        if (ast->operation.define_op.name) {
            defined_vars.insert(ast->operation.define_op.name);
        }
    }
}

// Helper: Collect defined variables from function body (for function's local scope)
static void collectBodyDefinedVariables(const eshkol_ast_t* body, std::set<std::string>& defined_vars) {
    if (!body) return;
    
    // Handle sequence of expressions (common in function bodies)
    if (body->type == ESHKOL_OP && body->operation.op == ESHKOL_SEQUENCE_OP) {
        for (uint64_t i = 0; i < body->operation.sequence_op.num_expressions; i++) {
            collectDefinedVariables(&body->operation.sequence_op.expressions[i], defined_vars);
        }
    } else {
        collectDefinedVariables(body, defined_vars);
    }
}

// Recursively collect all variable references in an AST subtree
static void collectVariableReferences(const eshkol_ast_t* ast, std::set<std::string>& refs) {
    if (!ast) return;
    
    switch (ast->type) {
        case ESHKOL_VAR:
            // Found a variable reference
            if (ast->variable.id) {
                refs.insert(ast->variable.id);
            }
            break;
            
        case ESHKOL_OP:
            switch (ast->operation.op) {
                case ESHKOL_CALL_OP:
                    // Collect from function
                    if (ast->operation.call_op.func) {
                        collectVariableReferences(ast->operation.call_op.func, refs);
                    }
                    // Collect from arguments
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        collectVariableReferences(&ast->operation.call_op.variables[i], refs);
                    }
                    break;
                    
                case ESHKOL_LAMBDA_OP:
                    // Don't collect from lambda parameters (they shadow)
                    // But do collect from lambda body
                    if (ast->operation.lambda_op.body) {
                        collectVariableReferences(ast->operation.lambda_op.body, refs);
                    }
                    break;
                    
                case ESHKOL_DEFINE_OP:
                    // Collect from defined value
                    if (ast->operation.define_op.value) {
                        collectVariableReferences(ast->operation.define_op.value, refs);
                    }
                    break;
                    
                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                    // Collect from bindings and body
                    for (uint64_t i = 0; i < ast->operation.let_op.num_bindings; i++) {
                        collectVariableReferences(&ast->operation.let_op.bindings[i], refs);
                    }
                    if (ast->operation.let_op.body) {
                        collectVariableReferences(ast->operation.let_op.body, refs);
                    }
                    break;
                    
                case ESHKOL_SEQUENCE_OP:
                    // Collect from all expressions in sequence
                    for (uint64_t i = 0; i < ast->operation.sequence_op.num_expressions; i++) {
                        collectVariableReferences(&ast->operation.sequence_op.expressions[i], refs);
                    }
                    break;
                    
                case ESHKOL_DERIVATIVE_OP:
                    if (ast->operation.derivative_op.function) {
                        collectVariableReferences(ast->operation.derivative_op.function, refs);
                    }
                    if (ast->operation.derivative_op.point) {
                        collectVariableReferences(ast->operation.derivative_op.point, refs);
                    }
                    break;
                    
                case ESHKOL_GRADIENT_OP:
                    if (ast->operation.gradient_op.function) {
                        collectVariableReferences(ast->operation.gradient_op.function, refs);
                    }
                    if (ast->operation.gradient_op.point) {
                        collectVariableReferences(ast->operation.gradient_op.point, refs);
                    }
                    break;
                    
                default:
                    // For other operations, recursively check if they have nested ASTs
                    break;
            }
            break;
            
        case ESHKOL_CONS:
            if (ast->cons_cell.car) {
                collectVariableReferences(ast->cons_cell.car, refs);
            }
            if (ast->cons_cell.cdr) {
                collectVariableReferences(ast->cons_cell.cdr, refs);
            }
            break;
            
        default:
            // Leaf nodes (numbers, strings, etc.) don't reference variables
            break;
    }
}

// Transform internal defines to letrec
// Takes a vector of body expressions and returns a transformed AST
// ALL internal defines are collected into a letrec, regardless of position
//
// Example:
//   (display "before")
//   (define a 1)
//   (define (helper x) (+ x 1))
//   (+ a (helper 2))
// Becomes:
//   (letrec ((a 1) (helper (lambda (x) (+ x 1))))
//     (begin (display "before") (+ a (helper 2))))
static eshkol_ast_t transformInternalDefinesToLetrec(const std::vector<eshkol_ast_t>& body_expressions) {
    eshkol_ast_t result;

    // PROPER EVALUATION ORDER FIX:
    // - Statements BEFORE defines should execute first
    // - Consecutive defines (starting from first define) are collected into letrec
    // - Statements AFTER the define block become the letrec body
    //
    // Example: (display "before") (define x 1) (define y 2) (display "after") (+ x y)
    // Becomes: (begin (display "before") (letrec ((x 1) (y 2)) (display "after") (+ x y)))

    std::vector<eshkol_ast_t> before_defines;  // Statements before first define
    std::vector<eshkol_ast_t> defines;          // Consecutive defines
    std::vector<eshkol_ast_t> after_defines;    // Everything after defines

    // Find first define
    size_t first_define_idx = body_expressions.size();
    for (size_t i = 0; i < body_expressions.size(); i++) {
        if (body_expressions[i].type == ESHKOL_OP &&
            body_expressions[i].operation.op == ESHKOL_DEFINE_OP) {
            first_define_idx = i;
            break;
        }
    }

    // Collect statements before first define
    for (size_t i = 0; i < first_define_idx; i++) {
        before_defines.push_back(body_expressions[i]);
    }

    // Collect consecutive defines starting from first_define_idx
    size_t define_end_idx = first_define_idx;
    for (size_t i = first_define_idx; i < body_expressions.size(); i++) {
        if (body_expressions[i].type == ESHKOL_OP &&
            body_expressions[i].operation.op == ESHKOL_DEFINE_OP) {
            defines.push_back(body_expressions[i]);
            define_end_idx = i + 1;
        } else {
            // Non-define encountered - stop collecting defines
            // But continue to check for more defines (allow interleaved for now)
            // Actually, for strict behavior we'd break here. For Eshkol extension,
            // collect ALL remaining defines but preserve non-define positions
            after_defines.push_back(body_expressions[i]);
        }
    }

    // Simplified: Collect ALL defines, put all non-defines after
    // But statements before FIRST define execute before letrec
    defines.clear();
    after_defines.clear();
    for (size_t i = first_define_idx; i < body_expressions.size(); i++) {
        const auto& expr = body_expressions[i];
        if (expr.type == ESHKOL_OP && expr.operation.op == ESHKOL_DEFINE_OP) {
            defines.push_back(expr);
        } else {
            after_defines.push_back(expr);
        }
    }

    // If no internal defines, return original body
    if (defines.empty()) {
        if (body_expressions.size() == 1) {
            return body_expressions[0];
        } else {
            // Create sequence
            result.type = ESHKOL_OP;
            result.operation.op = ESHKOL_SEQUENCE_OP;
            result.operation.sequence_op.num_expressions = body_expressions.size();
            result.operation.sequence_op.expressions = new eshkol_ast_t[body_expressions.size()];
            for (size_t i = 0; i < body_expressions.size(); i++) {
                result.operation.sequence_op.expressions[i] = body_expressions[i];
            }
            return result;
        }
    }

    eshkol_debug("Transforming %zu internal defines to letrec", defines.size());

    // Create letrec AST
    result.type = ESHKOL_OP;
    result.operation.op = ESHKOL_LETREC_OP;

    // Convert defines to bindings
    result.operation.let_op.num_bindings = defines.size();
    result.operation.let_op.bindings = new eshkol_ast_t[defines.size()];
    result.operation.let_op.binding_types = nullptr;  // No type annotations for internal defines
    result.operation.let_op.name = nullptr;  // Not a named let

    for (size_t i = 0; i < defines.size(); i++) {
        const auto& def = defines[i];

        // Create variable node for binding
        eshkol_ast_t var_ast = {.type = ESHKOL_VAR};
        var_ast.variable.id = new char[strlen(def.operation.define_op.name) + 1];
        strcpy(var_ast.variable.id, def.operation.define_op.name);
        var_ast.variable.data = nullptr;

        // Get value - if it's a function define, wrap in lambda
        eshkol_ast_t val_ast;
        if (def.operation.define_op.is_function) {
            // Create lambda for function definition
            val_ast.type = ESHKOL_OP;
            val_ast.operation.op = ESHKOL_LAMBDA_OP;
            val_ast.operation.lambda_op.num_params = def.operation.define_op.num_params;

            if (val_ast.operation.lambda_op.num_params > 0) {
                val_ast.operation.lambda_op.parameters = new eshkol_ast_t[val_ast.operation.lambda_op.num_params];
                for (size_t j = 0; j < val_ast.operation.lambda_op.num_params; j++) {
                    val_ast.operation.lambda_op.parameters[j] = def.operation.define_op.parameters[j];
                }
            } else {
                val_ast.operation.lambda_op.parameters = nullptr;
            }

            val_ast.operation.lambda_op.body = def.operation.define_op.value;
            val_ast.operation.lambda_op.captured_vars = nullptr;
            val_ast.operation.lambda_op.num_captured = 0;
            val_ast.operation.lambda_op.is_variadic = def.operation.define_op.is_variadic;
            if (def.operation.define_op.rest_param) {
                val_ast.operation.lambda_op.rest_param = new char[strlen(def.operation.define_op.rest_param) + 1];
                strcpy(val_ast.operation.lambda_op.rest_param, def.operation.define_op.rest_param);
            } else {
                val_ast.operation.lambda_op.rest_param = nullptr;
            }

            // HoTT type annotations - copy from define_op if present
            val_ast.operation.lambda_op.return_type = def.operation.define_op.return_type;
            if (val_ast.operation.lambda_op.num_params > 0 && def.operation.define_op.param_types) {
                val_ast.operation.lambda_op.param_types = new hott_type_expr_t*[val_ast.operation.lambda_op.num_params];
                for (size_t j = 0; j < val_ast.operation.lambda_op.num_params; j++) {
                    val_ast.operation.lambda_op.param_types[j] = def.operation.define_op.param_types[j];
                }
            } else {
                val_ast.operation.lambda_op.param_types = nullptr;
            }
        } else {
            // Simple variable define - use value directly
            val_ast = *def.operation.define_op.value;
        }

        // Create binding as cons cell (var . val)
        result.operation.let_op.bindings[i].type = ESHKOL_CONS;
        result.operation.let_op.bindings[i].cons_cell.car = new eshkol_ast_t;
        *result.operation.let_op.bindings[i].cons_cell.car = var_ast;
        result.operation.let_op.bindings[i].cons_cell.cdr = new eshkol_ast_t;
        *result.operation.let_op.bindings[i].cons_cell.cdr = val_ast;
    }

    // Create body from expressions after defines
    eshkol_ast_t body;

    if (after_defines.empty()) {
        // No body expressions after defines - this is an error
        // But for robustness, return null
        eshkol_error("Internal defines must be followed by at least one expression");
        body.type = ESHKOL_NULL;
    } else if (after_defines.size() == 1) {
        body = after_defines[0];
    } else {
        // Multiple expressions - wrap in sequence
        body.type = ESHKOL_OP;
        body.operation.op = ESHKOL_SEQUENCE_OP;
        body.operation.sequence_op.num_expressions = after_defines.size();
        body.operation.sequence_op.expressions = new eshkol_ast_t[after_defines.size()];
        for (size_t i = 0; i < after_defines.size(); i++) {
            body.operation.sequence_op.expressions[i] = after_defines[i];
        }
    }

    result.operation.let_op.body = new eshkol_ast_t;
    *result.operation.let_op.body = body;

    // If there are statements before the first define, wrap in sequence
    // (begin before1 before2 ... (letrec ...))
    if (!before_defines.empty()) {
        eshkol_ast_t final_result;
        final_result.type = ESHKOL_OP;
        final_result.operation.op = ESHKOL_SEQUENCE_OP;
        final_result.operation.sequence_op.num_expressions = before_defines.size() + 1;
        final_result.operation.sequence_op.expressions = new eshkol_ast_t[before_defines.size() + 1];

        // Copy before_defines statements first
        for (size_t i = 0; i < before_defines.size(); i++) {
            final_result.operation.sequence_op.expressions[i] = before_defines[i];
        }
        // Then the letrec
        final_result.operation.sequence_op.expressions[before_defines.size()] = result;

        return final_result;
    }

    return result;
}

// Analyze lambda for captured variables using STATIC AST ANALYSIS
// Returns a list of variable names that are captured from parent scope
// parent_defined_vars: variables available in the enclosing scope
static std::vector<std::string> analyzeLambdaCaptures(
    const eshkol_ast_t* lambda_body,
    const std::vector<eshkol_ast_t>& params,
    const std::set<std::string>& parent_defined_vars
) {
    // Collect parameter names (these shadow parent scope)
    std::set<std::string> param_names;
    for (const auto& param : params) {
        if (param.type == ESHKOL_VAR && param.variable.id) {
            param_names.insert(param.variable.id);
        }
    }
    
    // Collect variables defined in lambda body (these are local)
    std::set<std::string> local_defined;
    collectBodyDefinedVariables(lambda_body, local_defined);
    
    // Collect all variable references in lambda body
    std::set<std::string> all_refs;
    collectVariableReferences(lambda_body, all_refs);
    
    // Determine captures: referenced, not in params, not locally defined, but in parent scope
    std::vector<std::string> captures;
    for (const auto& var : all_refs) {
        if (param_names.count(var) == 0 &&          // Not a parameter
            local_defined.count(var) == 0 &&         // Not locally defined
            parent_defined_vars.count(var) > 0) {    // Available in parent scope
            captures.push_back(var);
            eshkol_debug("Lambda captures variable: %s", var.c_str());
        }
    }
    
    return captures;
}

// Static analysis helper: Build scope context for analyzing a function body
// Returns set of all variables available in the current scope
static std::set<std::string> buildScopeContext(const eshkol_ast_t* enclosing_func_body) {
    std::set<std::string> available_vars;
    
    if (!enclosing_func_body) return available_vars;
    
    // Collect all defines from enclosing function body
    collectBodyDefinedVariables(enclosing_func_body, available_vars);
    
    return available_vars;
}

// ===== END CLOSURE CAPTURE ANALYSIS =====

static eshkol_ast_t parse_expression(SchemeTokenizer& tokenizer);
static eshkol_ast_t parse_list(SchemeTokenizer& tokenizer);
static eshkol_pattern_t* parse_pattern(SchemeTokenizer& tokenizer);

static eshkol_ast_t parse_function_signature(SchemeTokenizer& tokenizer) {
    // Parse function signature: (func-name param1 param2 ...)
    // Or:                       (func-name param1 . rest)  - with rest parameter
    // Or:                       (func-name (x : int) (y : real))  - with inline type annotations
    // Returns an AST with function name and parameters
    eshkol_ast_t signature = {.type = ESHKOL_FUNC};
    std::vector<eshkol_ast_t> params;
    std::vector<hott_type_expr_t*> param_types;

    // Initialize variadic fields
    signature.eshkol_func.is_variadic = 0;
    signature.eshkol_func.rest_param = nullptr;
    signature.eshkol_func.param_types = nullptr;
    signature.eshkol_func.return_type = nullptr;

    Token token = tokenizer.nextToken();

    // First token should be the function name
    if (token.type != TOKEN_SYMBOL) {
        eshkol_error("Expected function name in define");
        signature.type = ESHKOL_INVALID;
        return signature;
    }

    // Set function name
    signature.eshkol_func.id = new char[token.value.length() + 1];
    strcpy(signature.eshkol_func.id, token.value.c_str());
    signature.eshkol_func.is_lambda = 0;

    // Parse parameters
    while (true) {
        token = tokenizer.nextToken();
        if (token.type == TOKEN_RPAREN) break;
        if (token.type == TOKEN_EOF) {
            eshkol_error("Unexpected end of input in function signature");
            signature.type = ESHKOL_INVALID;
            return signature;
        }

        // Check for dotted rest parameter: (func-name x y . rest)
        if (token.type == TOKEN_SYMBOL && token.value == ".") {
            // Next token should be the rest parameter name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("Expected rest parameter name after '.'");
                signature.type = ESHKOL_INVALID;
                return signature;
            }
            signature.eshkol_func.is_variadic = 1;
            signature.eshkol_func.rest_param = new char[token.value.length() + 1];
            strcpy(signature.eshkol_func.rest_param, token.value.c_str());

            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("Expected ')' after rest parameter");
                signature.type = ESHKOL_INVALID;
                return signature;
            }
            break;
        }

        // Check for inline type annotation: (param-name : type)
        if (token.type == TOKEN_LPAREN) {
            // Parse (param-name : type)
            Token param_token = tokenizer.nextToken();
            if (param_token.type != TOKEN_SYMBOL) {
                eshkol_error("Expected parameter name in typed parameter");
                signature.type = ESHKOL_INVALID;
                return signature;
            }

            // Create parameter AST
            eshkol_ast_t param = {.type = ESHKOL_VAR};
            param.variable.id = new char[param_token.value.length() + 1];
            strcpy(param.variable.id, param_token.value.c_str());
            param.variable.data = nullptr;

            // Expect colon
            Token colon = tokenizer.nextToken();
            if (colon.type != TOKEN_COLON) {
                eshkol_error("Expected ':' after parameter name in typed parameter");
                signature.type = ESHKOL_INVALID;
                return signature;
            }

            // Parse type expression
            hott_type_expr_t* type_expr = parseTypeExpression(tokenizer);
            if (!type_expr) {
                eshkol_error("Failed to parse type in typed parameter");
                signature.type = ESHKOL_INVALID;
                return signature;
            }

            // Expect closing paren
            Token rparen = tokenizer.nextToken();
            if (rparen.type != TOKEN_RPAREN) {
                eshkol_error("Expected ')' after type in typed parameter");
                hott_free_type_expr(type_expr);
                signature.type = ESHKOL_INVALID;
                return signature;
            }

            params.push_back(param);
            param_types.push_back(type_expr);

            eshkol_debug("Parsed typed parameter '%s'", param_token.value.c_str());
            continue;
        }

        if (token.type == TOKEN_SYMBOL) {
            eshkol_ast_t param = {.type = ESHKOL_VAR};
            param.variable.id = new char[token.value.length() + 1];
            strcpy(param.variable.id, token.value.c_str());
            param.variable.data = nullptr;
            params.push_back(param);
            param_types.push_back(nullptr);  // No type annotation
        } else {
            eshkol_error("Expected parameter name in function signature");
            signature.type = ESHKOL_INVALID;
            return signature;
        }
    }

    // Set up parameter array
    signature.eshkol_func.num_variables = params.size();
    if (signature.eshkol_func.num_variables > 0) {
        signature.eshkol_func.variables = new eshkol_ast_t[signature.eshkol_func.num_variables];
        signature.eshkol_func.param_types = new hott_type_expr_t*[signature.eshkol_func.num_variables];
        for (size_t i = 0; i < signature.eshkol_func.num_variables; i++) {
            signature.eshkol_func.variables[i] = params[i];
            signature.eshkol_func.param_types[i] = param_types[i];  // Transfer ownership
        }
    } else {
        signature.eshkol_func.variables = nullptr;
        signature.eshkol_func.param_types = nullptr;
    }

    return signature;
}

// ===== PATTERN MATCHING HELPER =====

// Recursive pattern parser - handles nested patterns
static eshkol_pattern_t* parse_pattern(SchemeTokenizer& tokenizer) {
    Token token = tokenizer.nextToken();

    eshkol_pattern_t* pattern = new eshkol_pattern_t;
    pattern->type = PATTERN_INVALID;

    if (token.type == TOKEN_SYMBOL) {
        // Symbol: wildcard (_) or variable binding
        if (token.value == "_") {
            pattern->type = PATTERN_WILDCARD;
        } else {
            pattern->type = PATTERN_VARIABLE;
            pattern->variable.name = new char[token.value.length() + 1];
            strcpy(pattern->variable.name, token.value.c_str());
        }
    } else if (token.type == TOKEN_NUMBER || token.type == TOKEN_STRING ||
               token.type == TOKEN_BOOLEAN || token.type == TOKEN_CHAR) {
        // Literal pattern
        pattern->type = PATTERN_LITERAL;
        pattern->literal.value = new eshkol_ast_t;
        *pattern->literal.value = parse_atom(token);
    } else if (token.type == TOKEN_QUOTE) {
        // Quoted literal pattern 'x
        pattern->type = PATTERN_LITERAL;
        pattern->literal.value = new eshkol_ast_t;
        *pattern->literal.value = parse_quoted_data(tokenizer);
    } else if (token.type == TOKEN_LPAREN) {
        // Complex pattern: (cons ...), (list ...), (? ...), (or ...), or literal list
        Token peek = tokenizer.nextToken();

        if (peek.type == TOKEN_SYMBOL) {
            if (peek.value == "cons") {
                // Cons pattern: (cons car-pat cdr-pat)
                pattern->type = PATTERN_CONS;
                // Recursively parse car and cdr patterns
                pattern->cons.car_pattern = parse_pattern(tokenizer);
                pattern->cons.cdr_pattern = parse_pattern(tokenizer);
                // Consume closing paren
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing paren after cons pattern");
                    pattern->type = PATTERN_INVALID;
                }
            } else if (peek.value == "list") {
                // List pattern: (list p1 p2 ...)
                pattern->type = PATTERN_LIST;
                std::vector<eshkol_pattern_t*> list_pats;
                while (true) {
                    // Peek to see if we're at the end
                    Token peek_end = tokenizer.nextToken();
                    if (peek_end.type == TOKEN_RPAREN) break;
                    // Put the token back by parsing it as part of pattern
                    tokenizer.pushBack(peek_end);
                    eshkol_pattern_t* elem_pat = parse_pattern(tokenizer);
                    if (elem_pat) {
                        list_pats.push_back(elem_pat);
                    }
                }
                pattern->list.num_patterns = list_pats.size();
                pattern->list.patterns = new eshkol_pattern_t*[list_pats.size()];
                for (size_t i = 0; i < list_pats.size(); i++) {
                    pattern->list.patterns[i] = list_pats[i];
                }
            } else if (peek.value == "?") {
                // Predicate pattern: (? pred)
                pattern->type = PATTERN_PREDICATE;
                Token pred_tok = tokenizer.nextToken();
                pattern->predicate.predicate = new eshkol_ast_t;
                if (pred_tok.type == TOKEN_LPAREN) {
                    *pattern->predicate.predicate = parse_list(tokenizer);
                } else {
                    *pattern->predicate.predicate = parse_atom(pred_tok);
                }
                // Consume closing paren
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing paren after predicate pattern");
                    pattern->type = PATTERN_INVALID;
                }
            } else if (peek.value == "or") {
                // Or pattern: (or p1 p2 ...)
                pattern->type = PATTERN_OR;
                std::vector<eshkol_pattern_t*> or_pats;
                while (true) {
                    Token peek_end = tokenizer.nextToken();
                    if (peek_end.type == TOKEN_RPAREN) break;
                    tokenizer.pushBack(peek_end);
                    eshkol_pattern_t* sub_pat = parse_pattern(tokenizer);
                    if (sub_pat) {
                        or_pats.push_back(sub_pat);
                    }
                }
                pattern->or_pat.num_patterns = or_pats.size();
                pattern->or_pat.patterns = new eshkol_pattern_t*[or_pats.size()];
                for (size_t i = 0; i < or_pats.size(); i++) {
                    pattern->or_pat.patterns[i] = or_pats[i];
                }
            } else {
                // Unknown pattern keyword - treat as literal (quoted list-like)
                pattern->type = PATTERN_LITERAL;
                pattern->literal.value = new eshkol_ast_t;
                eshkol_ast_t list_ast = {.type = ESHKOL_VAR};
                list_ast.variable.id = new char[peek.value.length() + 1];
                strcpy(list_ast.variable.id, peek.value.c_str());
                *pattern->literal.value = list_ast;
                // Skip to end of this list
                int depth = 1;
                while (depth > 0) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_LPAREN) depth++;
                    else if (token.type == TOKEN_RPAREN) depth--;
                }
            }
        } else if (peek.type == TOKEN_RPAREN) {
            // Empty list pattern () - matches null
            pattern->type = PATTERN_LITERAL;
            pattern->literal.value = new eshkol_ast_t;
            pattern->literal.value->type = ESHKOL_NULL;
        } else {
            // Other starting token - probably a quoted list, treat as literal
            pattern->type = PATTERN_LITERAL;
            pattern->literal.value = new eshkol_ast_t;
            pattern->literal.value->type = ESHKOL_INVALID;
            // Skip the rest
            int depth = 1;
            while (depth > 0) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_LPAREN) depth++;
                else if (token.type == TOKEN_RPAREN) depth--;
            }
        }
    } else {
        // Unknown token type - invalid pattern
        pattern->type = PATTERN_INVALID;
    }

    return pattern;
}

// ===== END PATTERN MATCHING HELPER =====

static eshkol_ast_t parse_list(SchemeTokenizer& tokenizer) {
    eshkol_ast_t ast = {};  // Zero-initialize all fields
    ast.type = ESHKOL_OP;
    std::vector<eshkol_ast_t> elements;

    Token token = tokenizer.nextToken();
    // Set source location from first token in the list
    ast.line = token.line;
    ast.column = token.column;
    
    // Empty list
    if (token.type == TOKEN_RPAREN) {
        ast.operation.op = ESHKOL_INVALID_OP;
        return ast;
    }
    
    // Handle case where first element is a lambda or other expression (e.g., ((lambda ...) ...))
    if (token.type == TOKEN_LPAREN) {
        // Parse the first element as an expression (could be lambda, function call, etc.)
        eshkol_ast_t func_expr = parse_list(tokenizer);
        if (func_expr.type == ESHKOL_INVALID) {
            ast.type = ESHKOL_INVALID;
            return ast;
        }
        
        // Parse remaining elements as arguments
        while (true) {
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) break;
            if (token.type == TOKEN_EOF) {
                eshkol_error("Unexpected end of input in list");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t element;
            if (token.type == TOKEN_LPAREN) {
                element = parse_list(tokenizer);
            } else if (token.type == TOKEN_QUOTE) {
                // Handle quoted expressions in nested list arguments
                eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                element.type = ESHKOL_OP;
                element.operation.op = ESHKOL_QUOTE_OP;
                element.operation.call_op.func = nullptr;
                element.operation.call_op.num_vars = 1;
                element.operation.call_op.variables = new eshkol_ast_t[1];
                element.operation.call_op.variables[0] = quoted;
            } else {
                element = parse_atom(token);
            }
            elements.push_back(element);
        }

        // Set up as a function call with the lambda/expression as the function
        ast.operation.op = ESHKOL_CALL_OP;
        ast.operation.call_op.func = new eshkol_ast_t;
        *ast.operation.call_op.func = func_expr;
        
        ast.operation.call_op.num_vars = elements.size();
        if (ast.operation.call_op.num_vars > 0) {
            ast.operation.call_op.variables = new eshkol_ast_t[ast.operation.call_op.num_vars];
            for (size_t i = 0; i < ast.operation.call_op.num_vars; i++) {
                ast.operation.call_op.variables[i] = elements[i];
            }
        } else {
            ast.operation.call_op.variables = nullptr;
        }
        
        return ast;
    }

    // Handle type annotation: (: name type)
    if (token.type == TOKEN_COLON) {
        // Parse: (: name type-expression)
        Token name_token = tokenizer.nextToken();
        if (name_token.type != TOKEN_SYMBOL) {
            eshkol_error("Expected identifier after : in type annotation");
            ast.type = ESHKOL_INVALID;
            return ast;
        }

        hott_type_expr_t* type_expr = parseTypeExpression(tokenizer);
        if (!type_expr) {
            eshkol_error("Failed to parse type expression in type annotation");
            ast.type = ESHKOL_INVALID;
            return ast;
        }

        Token rparen = tokenizer.nextToken();
        if (rparen.type != TOKEN_RPAREN) {
            eshkol_error("Expected ) after type annotation");
            hott_free_type_expr(type_expr);
            ast.type = ESHKOL_INVALID;
            return ast;
        }

        // Set up type annotation operation
        ast.operation.op = ESHKOL_TYPE_ANNOTATION_OP;
        ast.operation.type_annotation_op.name = new char[name_token.value.length() + 1];
        strcpy(ast.operation.type_annotation_op.name, name_token.value.c_str());
        ast.operation.type_annotation_op.type_expr = type_expr;

        eshkol_debug("Parsed type annotation for '%s'", name_token.value.c_str());
        return ast;
    }

    // First element should determine the operation
    if (token.type == TOKEN_SYMBOL) {
        std::string first_symbol = token.value;  // Store the function name
        ast.operation.op = get_operator_type(token.value);

        // Special handling for define - we need to check if next token is LPAREN for function definitions
        if (ast.operation.op == ESHKOL_DEFINE_OP) {
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("define requires arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            if (token.type == TOKEN_LPAREN) {
                // Function definition: (define (name params...) body)
                // Or with return type: (define (name params...) : type body)
                eshkol_ast_t func_signature = parse_function_signature(tokenizer);
                if (func_signature.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Check for optional return type annotation: : type
                Token peek = tokenizer.peekToken();
                if (peek.type == TOKEN_COLON) {
                    tokenizer.nextToken();  // consume ':'
                    func_signature.eshkol_func.return_type = parseTypeExpression(tokenizer);
                    if (!func_signature.eshkol_func.return_type) {
                        eshkol_error("Failed to parse return type annotation in define");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    eshkol_debug("Parsed return type annotation for function '%s'",
                                func_signature.eshkol_func.id);
                }

                // Parse function body (can be multiple expressions)
                std::vector<eshkol_ast_t> body_expressions;
                
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in function body");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    
                    eshkol_ast_t expr;
                    if (token.type == TOKEN_LPAREN) {
                        expr = parse_list(tokenizer);
                    } else {
                        expr = parse_atom(token);
                    }
                    body_expressions.push_back(expr);
                }
                
                // Transform internal defines to letrec (if any)
                // This handles: single expression, sequence, and internal defines
                if (body_expressions.empty()) {
                    eshkol_error("Function body cannot be empty");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t body = transformInternalDefinesToLetrec(body_expressions);
                
                // Set up define operation for function
                ast.operation.define_op.name = new char[strlen(func_signature.eshkol_func.id) + 1];
                strcpy(ast.operation.define_op.name, func_signature.eshkol_func.id);
                
                ast.operation.define_op.is_function = 1;
                ast.operation.define_op.num_params = func_signature.eshkol_func.num_variables;
                
                if (ast.operation.define_op.num_params > 0) {
                    ast.operation.define_op.parameters = new eshkol_ast_t[ast.operation.define_op.num_params];
                    for (size_t i = 0; i < ast.operation.define_op.num_params; i++) {
                        ast.operation.define_op.parameters[i] = func_signature.eshkol_func.variables[i];
                    }
                } else {
                    ast.operation.define_op.parameters = nullptr;
                }
                
                ast.operation.define_op.value = new eshkol_ast_t;
                *ast.operation.define_op.value = body;

                // Copy variadic information from function signature
                ast.operation.define_op.is_variadic = func_signature.eshkol_func.is_variadic;
                if (func_signature.eshkol_func.rest_param) {
                    ast.operation.define_op.rest_param = new char[strlen(func_signature.eshkol_func.rest_param) + 1];
                    strcpy(ast.operation.define_op.rest_param, func_signature.eshkol_func.rest_param);
                } else {
                    ast.operation.define_op.rest_param = nullptr;
                }

                // Copy HoTT type annotations from function signature
                if (func_signature.eshkol_func.param_types && ast.operation.define_op.num_params > 0) {
                    ast.operation.define_op.param_types = new hott_type_expr_t*[ast.operation.define_op.num_params];
                    for (size_t i = 0; i < ast.operation.define_op.num_params; i++) {
                        // Transfer ownership of type expressions
                        ast.operation.define_op.param_types[i] = func_signature.eshkol_func.param_types[i];
                    }
                    delete[] func_signature.eshkol_func.param_types;  // Free the array but not the elements
                } else {
                    ast.operation.define_op.param_types = nullptr;
                }
                ast.operation.define_op.return_type = func_signature.eshkol_func.return_type;

                return ast;
                
            } else if (token.type == TOKEN_SYMBOL) {
                // Variable definition: (define name value)
                eshkol_ast_t name_ast = parse_atom(token);
                
                token = tokenizer.nextToken();
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Expected value after variable name in define");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                eshkol_ast_t value;
                if (token.type == TOKEN_LPAREN) {
                    value = parse_list(tokenizer);
                } else if (token.type == TOKEN_VECTOR_START) {
                    // Handle vector literal #(...)
                    value.type = ESHKOL_OP;
                    value.operation.op = ESHKOL_TENSOR_OP;
                    std::vector<eshkol_ast_t> elements;
                    while (true) {
                        Token elem_token = tokenizer.nextToken();
                        if (elem_token.type == TOKEN_RPAREN) break;
                        if (elem_token.type == TOKEN_EOF) {
                            eshkol_error("Unexpected end of input in vector literal");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        eshkol_ast_t element;
                        if (elem_token.type == TOKEN_LPAREN) {
                            element = parse_list(tokenizer);
                        } else {
                            element = parse_atom(elem_token);
                        }
                        if (element.type == ESHKOL_INVALID) {
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        elements.push_back(element);
                    }
                    value.operation.tensor_op.num_dimensions = 1;
                    value.operation.tensor_op.dimensions = new uint64_t[1];
                    value.operation.tensor_op.dimensions[0] = elements.size();
                    value.operation.tensor_op.total_elements = elements.size();
                    value.operation.tensor_op.elements = new eshkol_ast_t[elements.size()];
                    for (size_t i = 0; i < elements.size(); i++) {
                        value.operation.tensor_op.elements[i] = elements[i];
                    }
                } else if (token.type == TOKEN_QUOTE) {
                    // Handle quoted expressions
                    eshkol_ast_t quoted_expr = parse_quoted_data(tokenizer);
                    value.type = ESHKOL_OP;
                    value.operation.op = ESHKOL_QUOTE_OP;
                    value.operation.call_op.func = nullptr;
                    value.operation.call_op.num_vars = 1;
                    value.operation.call_op.variables = new eshkol_ast_t[1];
                    value.operation.call_op.variables[0] = quoted_expr;
                } else {
                    value = parse_atom(token);
                }
                
                // Check for closing paren
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing parenthesis after variable value");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                // Set up define operation for variable
                ast.operation.define_op.name = new char[strlen(name_ast.variable.id) + 1];
                strcpy(ast.operation.define_op.name, name_ast.variable.id);
                
                ast.operation.define_op.value = new eshkol_ast_t;
                *ast.operation.define_op.value = value;
                
                ast.operation.define_op.is_function = 0;
                ast.operation.define_op.parameters = nullptr;
                ast.operation.define_op.num_params = 0;
                
                return ast;
                
            } else {
                eshkol_error("define first argument must be a symbol or parameter list");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
        }

        // Special handling for set! - variable mutation
        if (ast.operation.op == ESHKOL_SET_OP) {
            // Syntax: (set! varname value)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("set! requires a variable name and value");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("set! first argument must be a variable name");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Store variable name
            ast.operation.set_op.name = new char[strlen(token.value.c_str()) + 1];
            strcpy(ast.operation.set_op.name, token.value.c_str());

            // Parse value
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF || token.type == TOKEN_RPAREN) {
                eshkol_error("set! requires a value");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t value;
            if (token.type == TOKEN_LPAREN) {
                value = parse_list(tokenizer);
            } else {
                value = parse_atom(token);
            }

            ast.operation.set_op.value = new eshkol_ast_t;
            *ast.operation.set_op.value = value;

            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("set! takes exactly 2 arguments: variable name and value");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            return ast;
        }

        // Special handling for define-type - type alias definition
        // Syntax: (define-type Name type-expr)
        //     or: (define-type (Name a b) type-expr)  - parameterized
        if (ast.operation.op == ESHKOL_DEFINE_TYPE_OP) {
            token = tokenizer.nextToken();

            std::string type_name;
            std::vector<std::string> type_params;

            if (token.type == TOKEN_SYMBOL) {
                // Simple type alias: (define-type Name type-expr)
                type_name = token.value;
            } else if (token.type == TOKEN_LPAREN) {
                // Parameterized type: (define-type (Name a b) type-expr)
                token = tokenizer.nextToken();
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("define-type requires type name");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                type_name = token.value;

                // Parse type parameters
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type != TOKEN_SYMBOL) {
                        eshkol_error("define-type parameters must be symbols");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    type_params.push_back(token.value);
                }
            } else {
                eshkol_error("define-type requires type name or (name params...)");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse type expression
            hott_type_expr_t* type_expr = parseTypeExpression(tokenizer);
            if (!type_expr) {
                eshkol_error("Failed to parse type expression in define-type");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("Expected ')' after define-type");
                hott_free_type_expr(type_expr);
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Set up define-type operation
            ast.operation.define_type_op.name = new char[type_name.length() + 1];
            strcpy(ast.operation.define_type_op.name, type_name.c_str());
            ast.operation.define_type_op.type_expr = type_expr;
            ast.operation.define_type_op.num_type_params = type_params.size();

            if (!type_params.empty()) {
                ast.operation.define_type_op.type_params = new char*[type_params.size()];
                for (size_t i = 0; i < type_params.size(); i++) {
                    ast.operation.define_type_op.type_params[i] = new char[type_params[i].length() + 1];
                    strcpy(ast.operation.define_type_op.type_params[i], type_params[i].c_str());
                }
            } else {
                ast.operation.define_type_op.type_params = nullptr;
            }

            eshkol_debug("Parsed define-type '%s' with %zu type parameters",
                        type_name.c_str(), type_params.size());
            return ast;
        }

        // Special handling for if - conditional expression
        if (ast.operation.op == ESHKOL_IF_OP) {
            // Syntax: (if condition then-expr else-expr)
            
            // Parse condition
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("if requires condition as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t condition;
            if (token.type == TOKEN_LPAREN) {
                condition = parse_list(tokenizer);
            } else if (token.type == TOKEN_QUOTE) {
                // Handle quoted expressions in condition
                eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                condition.type = ESHKOL_OP;
                condition.operation.op = ESHKOL_QUOTE_OP;
                condition.operation.call_op.func = nullptr;
                condition.operation.call_op.num_vars = 1;
                condition.operation.call_op.variables = new eshkol_ast_t[1];
                condition.operation.call_op.variables[0] = quoted;
            } else {
                condition = parse_atom(token);
            }
            
            // Parse then expression
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("if requires then-expression as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t then_expr;
            if (token.type == TOKEN_LPAREN) {
                then_expr = parse_list(tokenizer);
            } else if (token.type == TOKEN_QUOTE) {
                // Handle quoted expressions in then-branch
                eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                then_expr.type = ESHKOL_OP;
                then_expr.operation.op = ESHKOL_QUOTE_OP;
                then_expr.operation.call_op.func = nullptr;
                then_expr.operation.call_op.num_vars = 1;
                then_expr.operation.call_op.variables = new eshkol_ast_t[1];
                then_expr.operation.call_op.variables[0] = quoted;
            } else {
                then_expr = parse_atom(token);
            }
            
            // Parse else expression (optional in Scheme)
            token = tokenizer.nextToken();

            eshkol_ast_t else_expr;
            bool has_else = false;

            if (token.type == TOKEN_RPAREN) {
                // No else clause - use null as default (Scheme unspecified value)
                else_expr.type = ESHKOL_NULL;
                has_else = false;
            } else if (token.type == TOKEN_EOF) {
                eshkol_error("Unexpected end of input in if expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            } else if (token.type == TOKEN_LPAREN) {
                else_expr = parse_list(tokenizer);
                has_else = true;
            } else if (token.type == TOKEN_QUOTE) {
                // Handle quoted expressions in else-branch
                eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                else_expr.type = ESHKOL_OP;
                else_expr.operation.op = ESHKOL_QUOTE_OP;
                else_expr.operation.call_op.func = nullptr;
                else_expr.operation.call_op.num_vars = 1;
                else_expr.operation.call_op.variables = new eshkol_ast_t[1];
                else_expr.operation.call_op.variables[0] = quoted;
                has_else = true;
            } else {
                else_expr = parse_atom(token);
                has_else = true;
            }

            // Check for closing paren (only if we had an else clause)
            if (has_else) {
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing parenthesis after if expression");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
            }
            
            // Store the if operation as a call operation with 3 arguments
            // This is a workaround since the if_op structure is incomplete
            ast.operation.op = ESHKOL_CALL_OP;
            
            // Create function name AST node for "if"
            ast.operation.call_op.func = new eshkol_ast_t;
            ast.operation.call_op.func->type = ESHKOL_VAR;
            ast.operation.call_op.func->variable.id = new char[3];
            strcpy(ast.operation.call_op.func->variable.id, "if");
            ast.operation.call_op.func->variable.data = nullptr;
            
            // Set up arguments: condition, then-expr, else-expr
            ast.operation.call_op.num_vars = 3;
            ast.operation.call_op.variables = new eshkol_ast_t[3];
            ast.operation.call_op.variables[0] = condition;
            ast.operation.call_op.variables[1] = then_expr;
            ast.operation.call_op.variables[2] = else_expr;
            
            return ast;
        }
        
        // Special handling for lambda - anonymous function
        if (ast.operation.op == ESHKOL_LAMBDA_OP) {
            // Syntax: (lambda (param1 param2 ...) body)
            // Or:     (lambda (param1 param2 . rest) body)  - with rest parameter
            // Or:     (lambda args body)                     - variadic, all args as list
            // Or:     (lambda ((x : int) (y : real)) body)   - with inline type annotations

            // Initialize variadic fields
            ast.operation.lambda_op.is_variadic = 0;
            ast.operation.lambda_op.rest_param = nullptr;
            ast.operation.lambda_op.param_types = nullptr;
            ast.operation.lambda_op.return_type = nullptr;

            // Parse parameter list or single symbol
            token = tokenizer.nextToken();

            std::vector<eshkol_ast_t> params;
            std::vector<hott_type_expr_t*> param_types;

            if (token.type == TOKEN_SYMBOL) {
                // Variadic lambda: (lambda args body)
                // All arguments are captured as a single list parameter
                ast.operation.lambda_op.is_variadic = 1;
                ast.operation.lambda_op.rest_param = new char[token.value.length() + 1];
                strcpy(ast.operation.lambda_op.rest_param, token.value.c_str());
                // No fixed parameters
            } else if (token.type == TOKEN_LPAREN) {
                // Regular parameter list or mixed with rest parameter
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in lambda parameter list");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    // Check for dotted rest parameter: (x y . rest)
                    if (token.type == TOKEN_SYMBOL && token.value == ".") {
                        // Next token should be the rest parameter name
                        token = tokenizer.nextToken();
                        if (token.type != TOKEN_SYMBOL) {
                            eshkol_error("Expected rest parameter name after '.'");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        ast.operation.lambda_op.is_variadic = 1;
                        ast.operation.lambda_op.rest_param = new char[token.value.length() + 1];
                        strcpy(ast.operation.lambda_op.rest_param, token.value.c_str());

                        // Expect closing paren
                        token = tokenizer.nextToken();
                        if (token.type != TOKEN_RPAREN) {
                            eshkol_error("Expected ')' after rest parameter");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        break;
                    }

                    // Check for inline type annotation: (param-name : type)
                    if (token.type == TOKEN_LPAREN) {
                        Token param_token = tokenizer.nextToken();
                        if (param_token.type != TOKEN_SYMBOL) {
                            eshkol_error("Expected parameter name in typed lambda parameter");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }

                        eshkol_ast_t param = {.type = ESHKOL_VAR};
                        param.variable.id = new char[param_token.value.length() + 1];
                        strcpy(param.variable.id, param_token.value.c_str());
                        param.variable.data = nullptr;

                        Token colon = tokenizer.nextToken();
                        if (colon.type != TOKEN_COLON) {
                            eshkol_error("Expected ':' after parameter name in typed lambda parameter");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }

                        hott_type_expr_t* type_expr = parseTypeExpression(tokenizer);
                        if (!type_expr) {
                            eshkol_error("Failed to parse type in typed lambda parameter");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }

                        Token rparen = tokenizer.nextToken();
                        if (rparen.type != TOKEN_RPAREN) {
                            eshkol_error("Expected ')' after type in typed lambda parameter");
                            hott_free_type_expr(type_expr);
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }

                        params.push_back(param);
                        param_types.push_back(type_expr);
                        continue;
                    }

                    if (token.type == TOKEN_SYMBOL) {
                        eshkol_ast_t param = {.type = ESHKOL_VAR};
                        param.variable.id = new char[token.value.length() + 1];
                        strcpy(param.variable.id, token.value.c_str());
                        param.variable.data = nullptr;
                        params.push_back(param);
                        param_types.push_back(nullptr);  // No type annotation
                    } else {
                        eshkol_error("Expected parameter name in lambda");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                }
            } else {
                eshkol_error("lambda requires parameter list or rest parameter symbol");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Check for optional return type annotation: : type
            // Syntax: (lambda (params...) : type body)
            Token peek = tokenizer.peekToken();
            if (peek.type == TOKEN_COLON) {
                tokenizer.nextToken();  // consume ':'
                ast.operation.lambda_op.return_type = parseTypeExpression(tokenizer);
                if (!ast.operation.lambda_op.return_type) {
                    eshkol_error("Failed to parse return type annotation in lambda");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                eshkol_debug("Parsed return type annotation for lambda");
            }

            // Parse lambda body (can be multiple expressions with internal defines)
            std::vector<eshkol_ast_t> body_expressions;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in lambda body");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }
                body_expressions.push_back(expr);
            }

            if (body_expressions.empty()) {
                eshkol_error("lambda requires body expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Transform internal defines to letrec (same as function define)
            eshkol_ast_t body = transformInternalDefinesToLetrec(body_expressions);
            
            // Set up lambda operation
            ast.operation.lambda_op.num_params = params.size();
            if (ast.operation.lambda_op.num_params > 0) {
                ast.operation.lambda_op.parameters = new eshkol_ast_t[ast.operation.lambda_op.num_params];
                ast.operation.lambda_op.param_types = new hott_type_expr_t*[ast.operation.lambda_op.num_params];
                for (size_t i = 0; i < ast.operation.lambda_op.num_params; i++) {
                    ast.operation.lambda_op.parameters[i] = params[i];
                    ast.operation.lambda_op.param_types[i] = param_types[i];  // Transfer ownership
                }
            } else {
                ast.operation.lambda_op.parameters = nullptr;
                ast.operation.lambda_op.param_types = nullptr;
            }
            
            ast.operation.lambda_op.body = new eshkol_ast_t;
            *ast.operation.lambda_op.body = body;
            
            // Analyze for captured variables
            // NOTE: During parsing we don't have complete scope context yet
            // Real capture analysis happens in codegen via findFreeVariables()
            // Pass empty set for now - this just populates AST structure
            std::set<std::string> empty_parent_scope;
            std::vector<std::string> captures = analyzeLambdaCaptures(&body, params, empty_parent_scope);
            
            if (!captures.empty()) {
                // Lambda has captured variables - populate AST
                ast.operation.lambda_op.num_captured = captures.size();
                ast.operation.lambda_op.captured_vars = new eshkol_ast_t[captures.size()];
                
                for (size_t i = 0; i < captures.size(); i++) {
                    ast.operation.lambda_op.captured_vars[i].type = ESHKOL_VAR;
                    ast.operation.lambda_op.captured_vars[i].variable.id =
                        new char[captures[i].length() + 1];
                    strcpy(ast.operation.lambda_op.captured_vars[i].variable.id,
                           captures[i].c_str());
                    ast.operation.lambda_op.captured_vars[i].variable.data = nullptr;
                }
                
                eshkol_debug("Lambda has %zu captured variables", captures.size());
            } else {
                // No captured variables
                ast.operation.lambda_op.captured_vars = nullptr;
                ast.operation.lambda_op.num_captured = 0;
            }
            
            return ast;
        }
        
        // Special handling for let/let*/letrec - local variable bindings
        // let: bindings can't reference each other
        // let*: sequential (each binding can reference previous ones)
        // letrec: all bindings visible to all values (mutual recursion)
        // We use the same parsing - codegen handles the difference
        if (ast.operation.op == ESHKOL_LET_OP || ast.operation.op == ESHKOL_LET_STAR_OP || ast.operation.op == ESHKOL_LETREC_OP) {
            // Syntax: (let ((var1 val1) (var2 val2) ...) body)
            // Named let: (let name ((var1 init1) ...) body)

            // Parse bindings list (or name for named let)
            token = tokenizer.nextToken();

            // Check for named let: (let name ((var init) ...) body)
            // Named let is only valid for regular let, not let* or letrec
            std::string named_let_name;
            if (ast.operation.op == ESHKOL_LET_OP && token.type == TOKEN_SYMBOL) {
                // This is a named let - save the name and get the bindings
                named_let_name = token.value;
                eshkol_debug("Parsing named let with name '%s'", named_let_name.c_str());
                token = tokenizer.nextToken();
            }

            if (token.type != TOKEN_LPAREN) {
                eshkol_error("let requires bindings list as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            std::vector<eshkol_ast_t> bindings;
            std::vector<hott_type_expr_t*> binding_types;

            // Parse each binding: (variable value) or (variable : type value)
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in let bindings");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("let binding must be a list (variable value) or (variable : type value)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse variable name
                token = tokenizer.nextToken();
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("let binding must start with variable name");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t var_ast = {.type = ESHKOL_VAR};
                var_ast.variable.id = new char[token.value.length() + 1];
                strcpy(var_ast.variable.id, token.value.c_str());
                var_ast.variable.data = nullptr;

                // Check for optional type annotation: (var : type value)
                hott_type_expr_t* binding_type = nullptr;
                Token peek = tokenizer.peekToken();
                if (peek.type == TOKEN_COLON) {
                    tokenizer.nextToken();  // consume ':'
                    binding_type = parseTypeExpression(tokenizer);
                    if (!binding_type) {
                        eshkol_error("Failed to parse type annotation in let binding");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    eshkol_debug("Parsed type annotation for let binding '%s'", var_ast.variable.id);
                }

                // Parse value expression
                token = tokenizer.nextToken();
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Expected value in let binding");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                eshkol_ast_t val_ast;
                if (token.type == TOKEN_LPAREN) {
                    val_ast = parse_list(tokenizer);
                } else if (token.type == TOKEN_VECTOR_START) {
                    // Handle vector literal #(...) in let binding value
                    val_ast.type = ESHKOL_OP;
                    val_ast.operation.op = ESHKOL_TENSOR_OP;
                    std::vector<eshkol_ast_t> elements;
                    while (true) {
                        Token elem_token = tokenizer.nextToken();
                        if (elem_token.type == TOKEN_RPAREN) break;
                        if (elem_token.type == TOKEN_EOF) {
                            eshkol_error("Unexpected end of input in vector literal in let binding");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        eshkol_ast_t element;
                        if (elem_token.type == TOKEN_LPAREN) {
                            element = parse_list(tokenizer);
                        } else {
                            element = parse_atom(elem_token);
                        }
                        if (element.type == ESHKOL_INVALID) {
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        elements.push_back(element);
                    }
                    val_ast.operation.tensor_op.num_dimensions = 1;
                    val_ast.operation.tensor_op.dimensions = new uint64_t[1];
                    val_ast.operation.tensor_op.dimensions[0] = elements.size();
                    val_ast.operation.tensor_op.total_elements = elements.size();
                    val_ast.operation.tensor_op.elements = new eshkol_ast_t[elements.size()];
                    for (size_t i = 0; i < elements.size(); i++) {
                        val_ast.operation.tensor_op.elements[i] = elements[i];
                    }
                } else if (token.type == TOKEN_QUOTE) {
                    // Handle quoted expressions in let binding value: (let ((x '())) ...)
                    eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                    val_ast.type = ESHKOL_OP;
                    val_ast.operation.op = ESHKOL_QUOTE_OP;
                    val_ast.operation.call_op.func = nullptr;
                    val_ast.operation.call_op.num_vars = 1;
                    val_ast.operation.call_op.variables = new eshkol_ast_t[1];
                    val_ast.operation.call_op.variables[0] = quoted;
                } else {
                    val_ast = parse_atom(token);
                }
                
                if (val_ast.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                // Check for closing paren of binding
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing parenthesis after let binding");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                // Store binding as a cons cell (var . val)
                eshkol_ast_t binding = {.type = ESHKOL_CONS};
                binding.cons_cell.car = new eshkol_ast_t;
                *binding.cons_cell.car = var_ast;
                binding.cons_cell.cdr = new eshkol_ast_t;
                *binding.cons_cell.cdr = val_ast;

                bindings.push_back(binding);
                binding_types.push_back(binding_type);  // May be nullptr if no annotation
            }
            
            // Parse body expressions (can be multiple, like in define)
            std::vector<eshkol_ast_t> body_expressions;
            
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in let body");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }
                
                if (expr.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                body_expressions.push_back(expr);
            }
            
            if (body_expressions.empty()) {
                eshkol_error("let body cannot be empty");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Transform internal defines to letrec (same as function and lambda bodies)
            // This handles: single expression, sequence, and internal defines
            eshkol_ast_t body = transformInternalDefinesToLetrec(body_expressions);
            
            // Set up let operation
            ast.operation.let_op.num_bindings = bindings.size();
            if (ast.operation.let_op.num_bindings > 0) {
                ast.operation.let_op.bindings = new eshkol_ast_t[ast.operation.let_op.num_bindings];
                ast.operation.let_op.binding_types = new hott_type_expr_t*[ast.operation.let_op.num_bindings];
                for (size_t i = 0; i < ast.operation.let_op.num_bindings; i++) {
                    ast.operation.let_op.bindings[i] = bindings[i];
                    ast.operation.let_op.binding_types[i] = binding_types[i];  // Transfer ownership
                }
            } else {
                ast.operation.let_op.bindings = nullptr;
                ast.operation.let_op.binding_types = nullptr;
            }
            
            ast.operation.let_op.body = new eshkol_ast_t;
            *ast.operation.let_op.body = body;

            // Set named let name (NULL for regular let)
            if (!named_let_name.empty()) {
                ast.operation.let_op.name = new char[named_let_name.length() + 1];
                strcpy(ast.operation.let_op.name, named_let_name.c_str());
                eshkol_debug("Created named let '%s' with %zu bindings",
                            named_let_name.c_str(), bindings.size());
            } else {
                ast.operation.let_op.name = nullptr;
            }

            return ast;
        }

        // ===== MULTIPLE RETURN VALUES OPERATIONS =====

        // Special handling for values - return multiple values
        // Syntax: (values expr1 expr2 ...)
        if (ast.operation.op == ESHKOL_VALUES_OP) {
            std::vector<eshkol_ast_t> values;

            // Parse all value expressions
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in values");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else if (token.type == TOKEN_QUOTE) {
                    eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                    expr.type = ESHKOL_OP;
                    expr.operation.op = ESHKOL_QUOTE_OP;
                    expr.operation.call_op.func = nullptr;
                    expr.operation.call_op.num_vars = 1;
                    expr.operation.call_op.variables = new eshkol_ast_t[1];
                    expr.operation.call_op.variables[0] = quoted;
                } else {
                    expr = parse_atom(token);
                }

                if (expr.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                values.push_back(expr);
            }

            // Store values
            ast.operation.values_op.num_values = values.size();
            if (values.size() > 0) {
                ast.operation.values_op.expressions = new eshkol_ast_t[values.size()];
                for (size_t i = 0; i < values.size(); i++) {
                    ast.operation.values_op.expressions[i] = values[i];
                }
            } else {
                ast.operation.values_op.expressions = nullptr;
            }

            return ast;
        }

        // Special handling for call-with-values - apply consumer to producer's values
        // Syntax: (call-with-values producer consumer)
        if (ast.operation.op == ESHKOL_CALL_WITH_VALUES_OP) {
            // Parse producer (a thunk that returns multiple values)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("call-with-values requires producer argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t producer;
            if (token.type == TOKEN_LPAREN) {
                producer = parse_list(tokenizer);
            } else {
                producer = parse_atom(token);
            }

            if (producer.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse consumer (a function that takes the multiple values)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("call-with-values requires consumer argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t consumer;
            if (token.type == TOKEN_LPAREN) {
                consumer = parse_list(tokenizer);
            } else {
                consumer = parse_atom(token);
            }

            if (consumer.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Check for closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("call-with-values takes exactly 2 arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.call_with_values_op.producer = new eshkol_ast_t;
            *ast.operation.call_with_values_op.producer = producer;
            ast.operation.call_with_values_op.consumer = new eshkol_ast_t;
            *ast.operation.call_with_values_op.consumer = consumer;

            return ast;
        }

        // Special handling for let-values and let*-values - bind multiple values from producers
        // Syntax: (let-values (((var1 var2 ...) producer1) ...) body ...)
        if (ast.operation.op == ESHKOL_LET_VALUES_OP || ast.operation.op == ESHKOL_LET_STAR_VALUES_OP) {
            // Parse bindings list
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("let-values requires bindings list as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            std::vector<std::vector<std::string>> all_vars;  // Variable names per binding
            std::vector<eshkol_ast_t> producers;             // Producer expressions

            // Parse each binding: ((var1 var2 ...) producer)
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in let-values bindings");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("let-values binding must be a list ((vars ...) producer)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse variable list: (var1 var2 ...)
                token = tokenizer.nextToken();
                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("let-values binding must start with variable list");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                std::vector<std::string> vars;
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in let-values variable list");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    if (token.type != TOKEN_SYMBOL) {
                        eshkol_error("let-values variable list must contain only symbols");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    vars.push_back(token.value);
                }

                // Parse producer expression
                token = tokenizer.nextToken();
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Expected producer expression in let-values binding");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t producer;
                if (token.type == TOKEN_LPAREN) {
                    producer = parse_list(tokenizer);
                } else if (token.type == TOKEN_QUOTE) {
                    eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                    producer.type = ESHKOL_OP;
                    producer.operation.op = ESHKOL_QUOTE_OP;
                    producer.operation.call_op.func = nullptr;
                    producer.operation.call_op.num_vars = 1;
                    producer.operation.call_op.variables = new eshkol_ast_t[1];
                    producer.operation.call_op.variables[0] = quoted;
                } else {
                    producer = parse_atom(token);
                }

                if (producer.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Check for closing paren of binding
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing parenthesis after let-values binding");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                all_vars.push_back(vars);
                producers.push_back(producer);
            }

            // Parse body expressions
            std::vector<eshkol_ast_t> body_expressions;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in let-values body");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }

                if (expr.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                body_expressions.push_back(expr);
            }

            if (body_expressions.empty()) {
                eshkol_error("let-values body cannot be empty");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Transform internal defines to letrec
            eshkol_ast_t body = transformInternalDefinesToLetrec(body_expressions);

            // Set up let-values operation
            ast.operation.let_values_op.num_bindings = all_vars.size();
            if (all_vars.size() > 0) {
                ast.operation.let_values_op.binding_vars = new char**[all_vars.size()];
                ast.operation.let_values_op.binding_var_counts = new uint64_t[all_vars.size()];
                ast.operation.let_values_op.producers = new eshkol_ast_t[all_vars.size()];

                for (size_t i = 0; i < all_vars.size(); i++) {
                    ast.operation.let_values_op.binding_var_counts[i] = all_vars[i].size();
                    ast.operation.let_values_op.binding_vars[i] = new char*[all_vars[i].size()];
                    for (size_t j = 0; j < all_vars[i].size(); j++) {
                        ast.operation.let_values_op.binding_vars[i][j] = new char[all_vars[i][j].length() + 1];
                        strcpy(ast.operation.let_values_op.binding_vars[i][j], all_vars[i][j].c_str());
                    }
                    ast.operation.let_values_op.producers[i] = producers[i];
                }
            } else {
                ast.operation.let_values_op.binding_vars = nullptr;
                ast.operation.let_values_op.binding_var_counts = nullptr;
                ast.operation.let_values_op.producers = nullptr;
            }

            ast.operation.let_values_op.body = new eshkol_ast_t;
            *ast.operation.let_values_op.body = body;

            return ast;
        }

        // ===== END MULTIPLE RETURN VALUES OPERATIONS =====

        // Special handling for guard - exception handler (R7RS)
        // Syntax: (guard (var clause ...) body ...)
        // Where each clause is (test expr ...) or (else expr ...)
        if (ast.operation.op == ESHKOL_GUARD_OP) {
            // Parse handler specification: (var clause ...)
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("guard requires handler specification (var clause ...) as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse exception variable name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("guard handler specification must start with variable name");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Store variable name
            ast.operation.guard_op.var_name = new char[token.value.length() + 1];
            strcpy(ast.operation.guard_op.var_name, token.value.c_str());

            // Parse clauses: ((test expr ...) ...)
            std::vector<eshkol_ast_t> clauses;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;  // End of handler spec
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in guard handler");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("guard clause must be a list (test expr ...)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse clause: (test expr ...) or (else expr ...)
                eshkol_ast_t clause = {.type = ESHKOL_OP};
                clause.operation.op = ESHKOL_CALL_OP;

                // Parse test (first element)
                token = tokenizer.nextToken();
                if (token.type == TOKEN_EOF) {
                    eshkol_error("guard clause cannot be empty");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t test;
                if (token.type == TOKEN_LPAREN) {
                    test = parse_list(tokenizer);
                } else {
                    test = parse_atom(token);
                }

                clause.operation.call_op.func = new eshkol_ast_t;
                *clause.operation.call_op.func = test;

                // Parse body expressions
                std::vector<eshkol_ast_t> body_exprs;
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in guard clause body");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    eshkol_ast_t expr;
                    if (token.type == TOKEN_LPAREN) {
                        expr = parse_list(tokenizer);
                    } else {
                        expr = parse_atom(token);
                    }
                    body_exprs.push_back(expr);
                }

                clause.operation.call_op.num_vars = body_exprs.size();
                if (body_exprs.size() > 0) {
                    clause.operation.call_op.variables = new eshkol_ast_t[body_exprs.size()];
                    for (size_t i = 0; i < body_exprs.size(); i++) {
                        clause.operation.call_op.variables[i] = body_exprs[i];
                    }
                } else {
                    clause.operation.call_op.variables = nullptr;
                }

                clauses.push_back(clause);
            }

            // Store clauses
            ast.operation.guard_op.num_clauses = clauses.size();
            if (clauses.size() > 0) {
                ast.operation.guard_op.clauses = new eshkol_ast_t[clauses.size()];
                for (size_t i = 0; i < clauses.size(); i++) {
                    ast.operation.guard_op.clauses[i] = clauses[i];
                }
            } else {
                ast.operation.guard_op.clauses = nullptr;
            }

            // Parse body expressions
            std::vector<eshkol_ast_t> body_expressions;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in guard body");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }
                body_expressions.push_back(expr);
            }

            if (body_expressions.empty()) {
                eshkol_error("guard body cannot be empty");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Transform internal defines to letrec if needed
            eshkol_ast_t body = transformInternalDefinesToLetrec(body_expressions);

            ast.operation.guard_op.body = new eshkol_ast_t[1];
            ast.operation.guard_op.body[0] = body;
            ast.operation.guard_op.num_body_exprs = 1;

            eshkol_debug("Parsed guard with variable '%s' and %zu clauses",
                        ast.operation.guard_op.var_name, clauses.size());
            return ast;
        }

        // Special handling for raise - raise exception
        // Syntax: (raise exception)
        if (ast.operation.op == ESHKOL_RAISE_OP) {
            // Parse exception expression
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF || token.type == TOKEN_RPAREN) {
                eshkol_error("raise requires an exception expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t exception;
            if (token.type == TOKEN_LPAREN) {
                exception = parse_list(tokenizer);
            } else {
                exception = parse_atom(token);
            }

            if (exception.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.raise_op.exception = new eshkol_ast_t;
            *ast.operation.raise_op.exception = exception;

            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("raise takes exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_debug("Parsed raise expression");
            return ast;
        }

        // Special handling for case - switch on value
        // case: (case key ((datum ...) expr ...) ... (else expr ...))
        if (ast.operation.op == ESHKOL_CASE_OP) {
            // Parse key expression
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF || token.type == TOKEN_RPAREN) {
                eshkol_error("case requires a key expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t key;
            if (token.type == TOKEN_LPAREN) {
                key = parse_list(tokenizer);
            } else {
                key = parse_atom(token);
            }

            if (key.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse clauses
            std::vector<eshkol_ast_t> clauses;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in case expression");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("case clause must be a list");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse clause: ((datum ...) expr ...) or (else expr ...)
                // First, check if it's an else clause
                token = tokenizer.nextToken();

                eshkol_ast_t clause;
                clause.type = ESHKOL_CONS;

                if (token.type == TOKEN_SYMBOL && token.value == "else") {
                    // else clause - create special marker for datums
                    eshkol_ast_t else_marker = {.type = ESHKOL_VAR};
                    else_marker.variable.id = new char[5];
                    strcpy(else_marker.variable.id, "else");
                    else_marker.variable.data = nullptr;

                    clause.cons_cell.car = new eshkol_ast_t;
                    *clause.cons_cell.car = else_marker;
                } else if (token.type == TOKEN_LPAREN) {
                    // Datums list - parse as quoted data
                    // Parse the datums list as quoted data
                    std::vector<eshkol_ast_t> datums;

                    while (true) {
                        Token inner = tokenizer.nextToken();
                        if (inner.type == TOKEN_RPAREN) break;
                        if (inner.type == TOKEN_EOF) {
                            eshkol_error("Unexpected end of input in case datums");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }

                        // Parse each datum as quoted data (not as expression)
                        eshkol_ast_t datum = parse_quoted_data_with_token(tokenizer, inner);
                        if (datum.type == ESHKOL_INVALID) {
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        datums.push_back(datum);
                    }

                    // Store datums as a proper list
                    // Build the list from the datums
                    if (datums.empty()) {
                        eshkol_error("case clause datums list cannot be empty");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    // Create a CALL_OP structure to hold the datums
                    // This way codegen can iterate through them
                    eshkol_ast_t datums_list = {.type = ESHKOL_OP};
                    datums_list.operation.op = ESHKOL_CALL_OP;
                    datums_list.operation.call_op.func = nullptr;
                    datums_list.operation.call_op.num_vars = datums.size();
                    datums_list.operation.call_op.variables = new eshkol_ast_t[datums.size()];
                    for (size_t i = 0; i < datums.size(); i++) {
                        datums_list.operation.call_op.variables[i] = datums[i];
                    }

                    clause.cons_cell.car = new eshkol_ast_t;
                    *clause.cons_cell.car = datums_list;
                } else {
                    eshkol_error("case clause must start with datums list or else");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse body expressions
                std::vector<eshkol_ast_t> body_exprs;

                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in case clause body");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    eshkol_ast_t expr;
                    if (token.type == TOKEN_LPAREN) {
                        expr = parse_list(tokenizer);
                    } else {
                        expr = parse_atom(token);
                    }

                    if (expr.type == ESHKOL_INVALID) {
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    body_exprs.push_back(expr);
                }

                // Store body as CALL_OP for iteration
                eshkol_ast_t body = {.type = ESHKOL_OP};
                body.operation.op = ESHKOL_CALL_OP;
                body.operation.call_op.func = nullptr;
                body.operation.call_op.num_vars = body_exprs.size();
                if (body_exprs.size() > 0) {
                    body.operation.call_op.variables = new eshkol_ast_t[body_exprs.size()];
                    for (size_t i = 0; i < body_exprs.size(); i++) {
                        body.operation.call_op.variables[i] = body_exprs[i];
                    }
                } else {
                    body.operation.call_op.variables = nullptr;
                }

                clause.cons_cell.cdr = new eshkol_ast_t;
                *clause.cons_cell.cdr = body;

                clauses.push_back(clause);
            }

            // Set up case operation
            ast.operation.call_op.func = new eshkol_ast_t;
            *ast.operation.call_op.func = key;
            ast.operation.call_op.num_vars = clauses.size();
            if (clauses.size() > 0) {
                ast.operation.call_op.variables = new eshkol_ast_t[clauses.size()];
                for (size_t i = 0; i < clauses.size(); i++) {
                    ast.operation.call_op.variables[i] = clauses[i];
                }
            } else {
                ast.operation.call_op.variables = nullptr;
            }

            return ast;
        }

        // Special handling for match - pattern matching
        // match: (match expr (pattern body ...) ... (_ default-body ...))
        if (ast.operation.op == ESHKOL_MATCH_OP) {
            // Parse expression to match against
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF || token.type == TOKEN_RPAREN) {
                eshkol_error("match requires an expression to match against");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t expr;
            if (token.type == TOKEN_LPAREN) {
                expr = parse_list(tokenizer);
            } else {
                expr = parse_atom(token);
            }

            if (expr.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse clauses: (pattern body ...)
            std::vector<eshkol_match_clause_t> clauses;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in match expression");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("match clause must be a list (pattern body ...)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_match_clause_t clause;
                clause.guard = nullptr;

                // Parse pattern using the recursive pattern parser
                clause.pattern = parse_pattern(tokenizer);
                if (!clause.pattern || clause.pattern->type == PATTERN_INVALID) {
                    eshkol_error("Invalid pattern in match clause");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse body expressions
                std::vector<eshkol_ast_t> body_exprs;

                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in match clause body");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    eshkol_ast_t body_expr;
                    if (token.type == TOKEN_LPAREN) {
                        body_expr = parse_list(tokenizer);
                    } else {
                        body_expr = parse_atom(token);
                    }

                    if (body_expr.type == ESHKOL_INVALID) {
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    body_exprs.push_back(body_expr);
                }

                // Store body - if multiple, wrap in sequence
                if (body_exprs.size() == 1) {
                    clause.body = new eshkol_ast_t;
                    *clause.body = body_exprs[0];
                } else if (body_exprs.size() > 1) {
                    clause.body = new eshkol_ast_t;
                    clause.body->type = ESHKOL_OP;
                    clause.body->operation.op = ESHKOL_SEQUENCE_OP;
                    clause.body->operation.sequence_op.num_expressions = body_exprs.size();
                    clause.body->operation.sequence_op.expressions = new eshkol_ast_t[body_exprs.size()];
                    for (size_t i = 0; i < body_exprs.size(); i++) {
                        clause.body->operation.sequence_op.expressions[i] = body_exprs[i];
                    }
                } else {
                    clause.body = new eshkol_ast_t;
                    clause.body->type = ESHKOL_NULL;
                }

                clauses.push_back(clause);
            }

            // Set up match operation
            ast.operation.match_op.expr = new eshkol_ast_t;
            *ast.operation.match_op.expr = expr;
            ast.operation.match_op.num_clauses = clauses.size();
            if (clauses.size() > 0) {
                ast.operation.match_op.clauses = new eshkol_match_clause_t[clauses.size()];
                for (size_t i = 0; i < clauses.size(); i++) {
                    ast.operation.match_op.clauses[i] = clauses[i];
                }
            } else {
                ast.operation.match_op.clauses = nullptr;
            }

            return ast;
        }

        // Special handling for define-syntax - macro definition
        // define-syntax: (define-syntax name (syntax-rules (literals...) ((pattern) template) ...))
        if (ast.operation.op == ESHKOL_DEFINE_SYNTAX_OP) {
            // Parse macro name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("define-syntax requires a name");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_macro_def_t *macro = new eshkol_macro_def_t;
            macro->name = strdup(token.value.c_str());
            macro->literals = nullptr;
            macro->num_literals = 0;
            macro->rules = nullptr;
            macro->num_rules = 0;

            // Expect (syntax-rules ...)
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("define-syntax requires (syntax-rules ...) as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Verify "syntax-rules"
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL || token.value != "syntax-rules") {
                eshkol_error("define-syntax currently only supports syntax-rules");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse literals list: (literal1 literal2 ...)
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("syntax-rules requires literals list");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            std::vector<std::string> literals;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in syntax-rules literals");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                if (token.type == TOKEN_SYMBOL) {
                    literals.push_back(token.value);
                }
            }

            if (literals.size() > 0) {
                macro->literals = new char*[literals.size()];
                macro->num_literals = literals.size();
                for (size_t i = 0; i < literals.size(); i++) {
                    macro->literals[i] = strdup(literals[i].c_str());
                }
            }

            // Parse rules: ((pattern) template) ...
            std::vector<eshkol_macro_rule_t> rules;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;  // End of syntax-rules
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in syntax-rules");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("syntax-rules rule must be a list ((pattern) template)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_macro_rule_t rule;
                rule.pattern = nullptr;
                rule.template_ = nullptr;

                // Parse pattern (which is itself a list)
                token = tokenizer.nextToken();
                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("syntax-rules pattern must be a list");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // For now, store the pattern as a simple list structure
                // We'll parse it into eshkol_macro_pattern_t in the macro expander
                // This allows us to store the raw S-expression pattern
                rule.pattern = new eshkol_macro_pattern_t;
                rule.pattern->type = MACRO_PAT_LIST;
                rule.pattern->followed_by_ellipsis = 0;

                // Parse pattern elements
                std::vector<eshkol_macro_pattern_t*> pat_elements;
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in macro pattern");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    eshkol_macro_pattern_t* elem = new eshkol_macro_pattern_t;
                    elem->followed_by_ellipsis = 0;

                    if (token.type == TOKEN_SYMBOL) {
                        if (token.value == "...") {
                            // Mark previous element as followed by ellipsis
                            if (pat_elements.size() > 0) {
                                pat_elements.back()->followed_by_ellipsis = 1;
                            }
                            delete elem;
                            continue;
                        }

                        // Check if it's a literal
                        bool is_literal = false;
                        for (const auto& lit : literals) {
                            if (lit == token.value) {
                                is_literal = true;
                                break;
                            }
                        }

                        elem->type = is_literal ? MACRO_PAT_LITERAL : MACRO_PAT_VARIABLE;
                        elem->identifier = strdup(token.value.c_str());
                    } else if (token.type == TOKEN_LPAREN) {
                        // Nested pattern - parse recursively (simplified for now)
                        elem->type = MACRO_PAT_LIST;
                        elem->list.elements = nullptr;
                        elem->list.num_elements = 0;
                        elem->list.rest = nullptr;
                        // Skip nested list for now - count parens
                        int depth = 1;
                        while (depth > 0) {
                            token = tokenizer.nextToken();
                            if (token.type == TOKEN_LPAREN) depth++;
                            else if (token.type == TOKEN_RPAREN) depth--;
                            else if (token.type == TOKEN_EOF) {
                                eshkol_error("Unexpected end of input in nested pattern");
                                ast.type = ESHKOL_INVALID;
                                return ast;
                            }
                        }
                    } else {
                        // Literal value
                        elem->type = MACRO_PAT_LITERAL;
                        elem->identifier = strdup(token.value.c_str());
                    }

                    pat_elements.push_back(elem);
                }

                if (pat_elements.size() > 0) {
                    rule.pattern->list.elements = new eshkol_macro_pattern_t*[pat_elements.size()];
                    rule.pattern->list.num_elements = pat_elements.size();
                    for (size_t i = 0; i < pat_elements.size(); i++) {
                        rule.pattern->list.elements[i] = pat_elements[i];
                    }
                } else {
                    rule.pattern->list.elements = nullptr;
                    rule.pattern->list.num_elements = 0;
                }
                rule.pattern->list.rest = nullptr;

                // Parse template - store as AST for now
                token = tokenizer.nextToken();
                eshkol_ast_t template_ast;
                if (token.type == TOKEN_LPAREN) {
                    template_ast = parse_list(tokenizer);
                } else {
                    template_ast = parse_atom(token);
                }

                // Convert AST to template structure (simplified)
                rule.template_ = new eshkol_macro_template_t;
                rule.template_->type = MACRO_TPL_LITERAL;
                rule.template_->literal = new eshkol_ast_t;
                *rule.template_->literal = template_ast;
                rule.template_->followed_by_ellipsis = 0;

                // Consume closing paren of rule
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing paren after macro rule template");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                rules.push_back(rule);
            }

            if (rules.size() > 0) {
                macro->rules = new eshkol_macro_rule_t[rules.size()];
                macro->num_rules = rules.size();
                for (size_t i = 0; i < rules.size(); i++) {
                    macro->rules[i] = rules[i];
                }
            }

            // Consume closing paren of define-syntax
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing paren after define-syntax");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.define_syntax_op.macro = macro;
            return ast;
        }

        // Special handling for do - iteration construct
        // do: (do ((var init step) ...) ((test) result ...) body ...)
        if (ast.operation.op == ESHKOL_DO_OP) {
            // Parse bindings list: ((var1 init1 step1) (var2 init2 step2) ...)
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("do requires bindings list as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            std::vector<eshkol_ast_t> bindings;

            // Parse each binding: (variable init [step])
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in do bindings");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("do binding must be a list (variable init [step])");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse variable name
                token = tokenizer.nextToken();
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("do binding must start with variable name");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t var_ast = {.type = ESHKOL_VAR};
                var_ast.variable.id = new char[token.value.length() + 1];
                strcpy(var_ast.variable.id, token.value.c_str());
                var_ast.variable.data = nullptr;

                // Parse init expression
                token = tokenizer.nextToken();
                if (token.type == TOKEN_EOF || token.type == TOKEN_RPAREN) {
                    eshkol_error("do binding requires init expression");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t init_ast;
                if (token.type == TOKEN_LPAREN) {
                    init_ast = parse_list(tokenizer);
                } else {
                    init_ast = parse_atom(token);
                }

                if (init_ast.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse optional step expression
                token = tokenizer.nextToken();
                eshkol_ast_t step_ast;
                bool has_step = false;

                if (token.type != TOKEN_RPAREN) {
                    has_step = true;
                    if (token.type == TOKEN_LPAREN) {
                        step_ast = parse_list(tokenizer);
                    } else {
                        step_ast = parse_atom(token);
                    }

                    if (step_ast.type == ESHKOL_INVALID) {
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    // Expect closing paren
                    token = tokenizer.nextToken();
                    if (token.type != TOKEN_RPAREN) {
                        eshkol_error("Expected closing parenthesis after do binding");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                }

                // Store binding as CONS: car=var, cdr=CONS(init, step_or_var)
                // If no step, step is same as var (no update)
                eshkol_ast_t inner_cons = {.type = ESHKOL_CONS};
                inner_cons.cons_cell.car = new eshkol_ast_t;
                *inner_cons.cons_cell.car = init_ast;
                inner_cons.cons_cell.cdr = new eshkol_ast_t;
                if (has_step) {
                    *inner_cons.cons_cell.cdr = step_ast;
                } else {
                    *inner_cons.cons_cell.cdr = var_ast;  // Step is just the var (no change)
                }

                eshkol_ast_t binding = {.type = ESHKOL_CONS};
                binding.cons_cell.car = new eshkol_ast_t;
                *binding.cons_cell.car = var_ast;
                binding.cons_cell.cdr = new eshkol_ast_t;
                *binding.cons_cell.cdr = inner_cons;

                bindings.push_back(binding);
            }

            // Parse test clause: ((test) result ...)
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("do requires test clause as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse test expression (first element of test clause)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF || token.type == TOKEN_RPAREN) {
                eshkol_error("do test clause requires test expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t test_ast;
            if (token.type == TOKEN_LPAREN) {
                test_ast = parse_list(tokenizer);
            } else {
                test_ast = parse_atom(token);
            }

            if (test_ast.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse result expressions
            std::vector<eshkol_ast_t> result_exprs;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in do test clause");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }

                if (expr.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                result_exprs.push_back(expr);
            }

            // Parse body expressions
            std::vector<eshkol_ast_t> body_exprs;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in do body");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }

                if (expr.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                body_exprs.push_back(expr);
            }

            // Store structure:
            // call_op.func = CONS(bindings-list, test-clause)
            // call_op.variables = body expressions
            // Where bindings-list is a CALL_OP with bindings as variables
            // And test-clause is a CONS(test, results-list)

            // Create bindings list as CALL_OP
            eshkol_ast_t bindings_list = {.type = ESHKOL_OP};
            bindings_list.operation.op = ESHKOL_CALL_OP;
            bindings_list.operation.call_op.func = nullptr;
            bindings_list.operation.call_op.num_vars = bindings.size();
            if (bindings.size() > 0) {
                bindings_list.operation.call_op.variables = new eshkol_ast_t[bindings.size()];
                for (size_t i = 0; i < bindings.size(); i++) {
                    bindings_list.operation.call_op.variables[i] = bindings[i];
                }
            } else {
                bindings_list.operation.call_op.variables = nullptr;
            }

            // Create results list as CALL_OP
            eshkol_ast_t results_list = {.type = ESHKOL_OP};
            results_list.operation.op = ESHKOL_CALL_OP;
            results_list.operation.call_op.func = nullptr;
            results_list.operation.call_op.num_vars = result_exprs.size();
            if (result_exprs.size() > 0) {
                results_list.operation.call_op.variables = new eshkol_ast_t[result_exprs.size()];
                for (size_t i = 0; i < result_exprs.size(); i++) {
                    results_list.operation.call_op.variables[i] = result_exprs[i];
                }
            } else {
                results_list.operation.call_op.variables = nullptr;
            }

            // Create test clause as CONS(test, results)
            eshkol_ast_t test_clause = {.type = ESHKOL_CONS};
            test_clause.cons_cell.car = new eshkol_ast_t;
            *test_clause.cons_cell.car = test_ast;
            test_clause.cons_cell.cdr = new eshkol_ast_t;
            *test_clause.cons_cell.cdr = results_list;

            // Create main structure CONS(bindings-list, test-clause)
            eshkol_ast_t main_cons = {.type = ESHKOL_CONS};
            main_cons.cons_cell.car = new eshkol_ast_t;
            *main_cons.cons_cell.car = bindings_list;
            main_cons.cons_cell.cdr = new eshkol_ast_t;
            *main_cons.cons_cell.cdr = test_clause;

            // Set up do operation
            ast.operation.call_op.func = new eshkol_ast_t;
            *ast.operation.call_op.func = main_cons;
            ast.operation.call_op.num_vars = body_exprs.size();
            if (body_exprs.size() > 0) {
                ast.operation.call_op.variables = new eshkol_ast_t[body_exprs.size()];
                for (size_t i = 0; i < body_exprs.size(); i++) {
                    ast.operation.call_op.variables[i] = body_exprs[i];
                }
            } else {
                ast.operation.call_op.variables = nullptr;
            }

            return ast;
        }

        // Special handling for and - short-circuit logical AND
        if (ast.operation.op == ESHKOL_AND_OP) {
            // Syntax: (and expr1 expr2 ...)
            // Returns first false value or last value if all are true
            std::vector<eshkol_ast_t> exprs;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in and expression");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }

                if (expr.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                exprs.push_back(expr);
            }

            // Store in sequence_op
            ast.operation.sequence_op.num_expressions = exprs.size();
            if (exprs.size() > 0) {
                ast.operation.sequence_op.expressions = new eshkol_ast_t[exprs.size()];
                for (size_t i = 0; i < exprs.size(); i++) {
                    ast.operation.sequence_op.expressions[i] = exprs[i];
                }
            } else {
                ast.operation.sequence_op.expressions = nullptr;
            }

            return ast;
        }

        // Special handling for or - short-circuit logical OR
        if (ast.operation.op == ESHKOL_OR_OP) {
            // Syntax: (or expr1 expr2 ...)
            // Returns first true value or last value if all are false
            std::vector<eshkol_ast_t> exprs;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in or expression");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }

                if (expr.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                exprs.push_back(expr);
            }

            // Store in sequence_op
            ast.operation.sequence_op.num_expressions = exprs.size();
            if (exprs.size() > 0) {
                ast.operation.sequence_op.expressions = new eshkol_ast_t[exprs.size()];
                for (size_t i = 0; i < exprs.size(); i++) {
                    ast.operation.sequence_op.expressions[i] = exprs[i];
                }
            } else {
                ast.operation.sequence_op.expressions = nullptr;
            }

            return ast;
        }

        // Special handling for tensor - create tensor literals
        if (ast.operation.op == ESHKOL_TENSOR_OP) {
            std::string tensor_name = first_symbol;
            
            if (tensor_name == "vector") {
                // Syntax: (vector element1 element2 ...)
                std::vector<eshkol_ast_t> tensor_elements;
                
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in vector literal");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    
                    eshkol_ast_t element;
                    if (token.type == TOKEN_LPAREN) {
                        element = parse_list(tokenizer);
                    } else {
                        element = parse_atom(token);
                    }
                    tensor_elements.push_back(element);
                }
                
                // Set up 1D tensor (vector)
                ast.operation.tensor_op.num_dimensions = 1;
                ast.operation.tensor_op.dimensions = new uint64_t[1];
                ast.operation.tensor_op.dimensions[0] = tensor_elements.size();
                ast.operation.tensor_op.total_elements = tensor_elements.size();
                
                if (ast.operation.tensor_op.total_elements > 0) {
                    ast.operation.tensor_op.elements = new eshkol_ast_t[ast.operation.tensor_op.total_elements];
                    for (size_t i = 0; i < ast.operation.tensor_op.total_elements; i++) {
                        ast.operation.tensor_op.elements[i] = tensor_elements[i];
                    }
                } else {
                    ast.operation.tensor_op.elements = nullptr;
                }
                
                return ast;
                
            } else if (tensor_name == "matrix") {
                // Syntax: (matrix rows cols element1 element2 ...)
                
                // Parse number of rows
                token = tokenizer.nextToken();
                if (token.type != TOKEN_NUMBER) {
                    eshkol_error("matrix requires rows as first argument");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                uint64_t rows = std::stoull(token.value);
                
                // Parse number of columns
                token = tokenizer.nextToken();
                if (token.type != TOKEN_NUMBER) {
                    eshkol_error("matrix requires columns as second argument");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                uint64_t cols = std::stoull(token.value);
                
                // Parse matrix elements
                std::vector<eshkol_ast_t> tensor_elements;
                uint64_t expected_elements = rows * cols;
                
                for (uint64_t i = 0; i < expected_elements; i++) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) {
                        if (i < expected_elements - 1) {
                            eshkol_error("matrix has insufficient elements: expected %llu, got %llu", 
                                       (unsigned long long)expected_elements, (unsigned long long)i);
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        break;
                    }
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in matrix literal");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    
                    eshkol_ast_t element;
                    if (token.type == TOKEN_LPAREN) {
                        element = parse_list(tokenizer);
                    } else {
                        element = parse_atom(token);
                    }
                    tensor_elements.push_back(element);
                }
                
                // Consume closing parenthesis if not already consumed
                if (tensor_elements.size() == expected_elements) {
                    token = tokenizer.nextToken();
                    if (token.type != TOKEN_RPAREN) {
                        eshkol_error("Expected closing parenthesis after matrix elements");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                }
                
                // Set up 2D tensor (matrix)
                ast.operation.tensor_op.num_dimensions = 2;
                ast.operation.tensor_op.dimensions = new uint64_t[2];
                ast.operation.tensor_op.dimensions[0] = rows;
                ast.operation.tensor_op.dimensions[1] = cols;
                ast.operation.tensor_op.total_elements = expected_elements;
                ast.operation.tensor_op.elements = new eshkol_ast_t[expected_elements];
                
                for (size_t i = 0; i < expected_elements; i++) {
                    ast.operation.tensor_op.elements[i] = tensor_elements[i];
                }
                
                return ast;
                
            } else {
                // Generic tensor: (tensor dim1 dim2 ... dimN element1 element2 ...)
                std::vector<uint64_t> dimensions;
                
                // Parse dimensions until we hit a non-number token
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_NUMBER) {
                        dimensions.push_back(std::stoull(token.value));
                    } else {
                        break; // First non-number token starts the elements
                    }
                }
                
                if (dimensions.empty()) {
                    eshkol_error("tensor requires at least one dimension");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                // Calculate total elements
                uint64_t total_elements = 1;
                for (uint64_t dim : dimensions) {
                    total_elements *= dim;
                }
                
                // Parse tensor elements, starting with the token we just read
                std::vector<eshkol_ast_t> tensor_elements;
                
                // Handle the first element we already read
                if (token.type != TOKEN_RPAREN && token.type != TOKEN_EOF) {
                    eshkol_ast_t element;
                    if (token.type == TOKEN_LPAREN) {
                        element = parse_list(tokenizer);
                    } else {
                        element = parse_atom(token);
                    }
                    tensor_elements.push_back(element);
                }
                
                // Parse remaining elements
                for (uint64_t i = tensor_elements.size(); i < total_elements; i++) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) {
                        if (i < total_elements - 1) {
                            eshkol_error("tensor has insufficient elements: expected %llu, got %llu", 
                                       (unsigned long long)total_elements, (unsigned long long)i);
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        break;
                    }
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in tensor literal");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    
                    eshkol_ast_t element;
                    if (token.type == TOKEN_LPAREN) {
                        element = parse_list(tokenizer);
                    } else {
                        element = parse_atom(token);
                    }
                    tensor_elements.push_back(element);
                }
                
                // Consume closing parenthesis if not already consumed
                if (tensor_elements.size() == total_elements) {
                    token = tokenizer.nextToken();
                    if (token.type != TOKEN_RPAREN) {
                        eshkol_error("Expected closing parenthesis after tensor elements");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                }
                
                // Set up generic tensor
                ast.operation.tensor_op.num_dimensions = dimensions.size();
                ast.operation.tensor_op.dimensions = new uint64_t[dimensions.size()];
                for (size_t i = 0; i < dimensions.size(); i++) {
                    ast.operation.tensor_op.dimensions[i] = dimensions[i];
                }
                ast.operation.tensor_op.total_elements = total_elements;
                ast.operation.tensor_op.elements = new eshkol_ast_t[total_elements];
                
                for (size_t i = 0; i < total_elements; i++) {
                    ast.operation.tensor_op.elements[i] = tensor_elements[i];
                }
                
                return ast;
            }
        }

        // Special handling for diff - symbolic differentiation
        if (ast.operation.op == ESHKOL_DIFF_OP) {
            // Syntax: (diff expression variable)
            // Example: (diff (+ (* x x) (* 2 x)) x)
            
            // Parse the expression to differentiate
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("diff requires expression as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t expression;
            if (token.type == TOKEN_LPAREN) {
                expression = parse_list(tokenizer);
            } else {
                expression = parse_atom(token);
            }
            
            if (expression.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the variable to differentiate with respect to
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("diff requires variable name as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after diff arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up diff operation
            ast.operation.diff_op.expression = new eshkol_ast_t;
            *ast.operation.diff_op.expression = expression;
            
            ast.operation.diff_op.variable = new char[token.value.length() + 1];
            strcpy(ast.operation.diff_op.variable, token.value.c_str());
            
            return ast;
        }
        
        // Special handling for derivative - forward-mode automatic differentiation
        if (ast.operation.op == ESHKOL_DERIVATIVE_OP) {
            // Syntax: (derivative function point)
            // Example: (derivative (lambda (x) (* x x)) 5.0)
            
            // Parse the function to differentiate (can be lambda or function reference)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("derivative requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (optional for higher-order usage)
            token = tokenizer.nextToken();

            // Check if this is the 1-argument form (derivative f) - returns a closure
            if (token.type == TOKEN_RPAREN) {
                // Higher-order form: (derivative f) returns a function that computes derivatives
                ast.operation.derivative_op.function = new eshkol_ast_t;
                *ast.operation.derivative_op.function = function;
                ast.operation.derivative_op.point = nullptr;  // No point = higher-order form
                ast.operation.derivative_op.mode = 0; // Forward-mode by default
                return ast;
            }

            if (token.type == TOKEN_EOF) {
                eshkol_error("derivative requires evaluation point as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }

            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after derivative arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Set up derivative operation (2-argument form)
            ast.operation.derivative_op.function = new eshkol_ast_t;
            *ast.operation.derivative_op.function = function;

            ast.operation.derivative_op.point = new eshkol_ast_t;
            *ast.operation.derivative_op.point = point;

            ast.operation.derivative_op.mode = 0; // Forward-mode by default

            return ast;
        }
        
        // Special handling for gradient - reverse-mode automatic differentiation
        if (ast.operation.op == ESHKOL_GRADIENT_OP) {
            // Syntax: (gradient function vector)
            // Example: (gradient (lambda (v) (dot v v)) (vector 1.0 2.0 3.0))
            
            // Parse the function to differentiate (can be lambda or function reference)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("gradient requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (optional for higher-order usage)
            token = tokenizer.nextToken();

            // Check if this is the 1-argument form (gradient f) - returns a closure
            if (token.type == TOKEN_RPAREN) {
                // Higher-order form: (gradient f) returns a function that computes gradients
                ast.operation.gradient_op.function = new eshkol_ast_t;
                *ast.operation.gradient_op.function = function;
                ast.operation.gradient_op.point = nullptr;  // No point = higher-order form
                return ast;
            }

            if (token.type == TOKEN_EOF) {
                eshkol_error("gradient requires evaluation vector as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }

            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after gradient arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Set up gradient operation (2-argument form)
            ast.operation.gradient_op.function = new eshkol_ast_t;
            *ast.operation.gradient_op.function = function;
            
            ast.operation.gradient_op.point = new eshkol_ast_t;
            *ast.operation.gradient_op.point = point;
            
            return ast;
        }
        
        // Special handling for jacobian - matrix of partial derivatives
        if (ast.operation.op == ESHKOL_JACOBIAN_OP) {
            // Syntax: (jacobian function vector)
            // Example: (jacobian (lambda (v) (vector (* (vref v 0) (vref v 1)) (pow (vref v 0) 2))) (vector 2.0 3.0))
            
            // Parse the function to differentiate
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("jacobian requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (vector)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("jacobian requires evaluation vector as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }
            
            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after jacobian arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up jacobian operation
            ast.operation.jacobian_op.function = new eshkol_ast_t;
            *ast.operation.jacobian_op.function = function;
            
            ast.operation.jacobian_op.point = new eshkol_ast_t;
            *ast.operation.jacobian_op.point = point;
            
            return ast;
        }
        
        // Special handling for hessian - matrix of second derivatives
        if (ast.operation.op == ESHKOL_HESSIAN_OP) {
            // Syntax: (hessian function vector)
            // Example: (hessian (lambda (v) (dot v v)) (vector 1.0 2.0 3.0))
            
            // Parse the function to differentiate
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("hessian requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (vector)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("hessian requires evaluation vector as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }
            
            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after hessian arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up hessian operation
            ast.operation.hessian_op.function = new eshkol_ast_t;
            *ast.operation.hessian_op.function = function;
            
            ast.operation.hessian_op.point = new eshkol_ast_t;
            *ast.operation.hessian_op.point = point;
            
            return ast;
        }
        
        // Special handling for divergence - vector field divergence
        if (ast.operation.op == ESHKOL_DIVERGENCE_OP) {
            // Syntax: (divergence function vector)
            // Example: (divergence (lambda (v) v) (vector 1.0 2.0 3.0))
            
            // Parse the function (vector field F: ℝⁿ → ℝⁿ)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("divergence requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (vector)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("divergence requires evaluation vector as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }
            
            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after divergence arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up divergence operation
            ast.operation.divergence_op.function = new eshkol_ast_t;
            *ast.operation.divergence_op.function = function;
            
            ast.operation.divergence_op.point = new eshkol_ast_t;
            *ast.operation.divergence_op.point = point;
            
            return ast;
        }
        
        // Special handling for curl - vector field curl (3D only)
        if (ast.operation.op == ESHKOL_CURL_OP) {
            // Syntax: (curl function vector)
            // Example: (curl (lambda (v) (vector (* (vref v 1) (vref v 2)) ...)) (vector 1.0 2.0 3.0))
            
            // Parse the function (vector field F: ℝ³ → ℝ³)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("curl requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (vector)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("curl requires evaluation vector as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }
            
            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after curl arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up curl operation
            ast.operation.curl_op.function = new eshkol_ast_t;
            *ast.operation.curl_op.function = function;
            
            ast.operation.curl_op.point = new eshkol_ast_t;
            *ast.operation.curl_op.point = point;
            
            return ast;
        }
        
        // Special handling for laplacian - scalar field laplacian
        if (ast.operation.op == ESHKOL_LAPLACIAN_OP) {
            // Syntax: (laplacian function vector)
            // Example: (laplacian (lambda (v) (dot v v)) (vector 1.0 2.0 3.0))
            
            // Parse the function (scalar field f: ℝⁿ → ℝ)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("laplacian requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (vector)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("laplacian requires evaluation vector as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }
            
            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after laplacian arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up laplacian operation
            ast.operation.laplacian_op.function = new eshkol_ast_t;
            *ast.operation.laplacian_op.function = function;
            
            ast.operation.laplacian_op.point = new eshkol_ast_t;
            *ast.operation.laplacian_op.point = point;
            
            return ast;
        }
        
        // Special handling for directional-derivative - derivative in a direction
        if (ast.operation.op == ESHKOL_DIRECTIONAL_DERIV_OP) {
            // Syntax: (directional-derivative function point direction)
            // Example: (directional-derivative f (vector 1.0 2.0) (vector 1.0 0.0))
            
            // Parse the function (scalar field f: ℝⁿ → ℝ)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("directional-derivative requires function as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t function;
            if (token.type == TOKEN_LPAREN) {
                function = parse_list(tokenizer);
            } else {
                function = parse_atom(token);
            }
            
            if (function.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the evaluation point (vector)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("directional-derivative requires evaluation point as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t point;
            if (token.type == TOKEN_LPAREN) {
                point = parse_list(tokenizer);
            } else {
                point = parse_atom(token);
            }
            
            if (point.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Parse the direction vector
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("directional-derivative requires direction vector as third argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t direction;
            if (token.type == TOKEN_LPAREN) {
                direction = parse_list(tokenizer);
            } else {
                direction = parse_atom(token);
            }
            
            if (direction.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Check for closing paren
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after directional-derivative arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up directional-derivative operation
            ast.operation.directional_deriv_op.function = new eshkol_ast_t;
            *ast.operation.directional_deriv_op.function = function;
            
            ast.operation.directional_deriv_op.point = new eshkol_ast_t;
            *ast.operation.directional_deriv_op.point = point;
            
            ast.operation.directional_deriv_op.direction = new eshkol_ast_t;
            *ast.operation.directional_deriv_op.direction = direction;
            
            return ast;
        }

        // Special handling for extern - declare external C variable/function.
        if (ast.operation.op == ESHKOL_EXTERN_VAR_OP) {
            // Syntax: (extern-var type variable name)

            // Parse type
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("extern requires return type as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.extern_var_op.type = new char[strlen(token.value.c_str()) + 1];
            strcpy(ast.operation.extern_var_op.type, token.value.c_str());

            // Parse variable name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("extern requires function name as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.extern_var_op.name = new char[strlen(token.value.c_str()) + 1];
            strcpy(ast.operation.extern_var_op.name, token.value.c_str());
        } else if (ast.operation.op == ESHKOL_EXTERN_OP) {
            // Syntax: (extern return-type function-name param1-type param2-type ...)
            // Example: (extern void print_hello)
            // Example: (extern int add int int)
            
            // Parse return type
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("extern requires return type as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            ast.operation.extern_op.return_type = new char[strlen(token.value.c_str()) + 1];
            strcpy(ast.operation.extern_op.return_type, token.value.c_str());
            
            // Parse function name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("extern requires function name as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            ast.operation.extern_op.name = new char[strlen(token.value.c_str()) + 1];
            strcpy(ast.operation.extern_op.name, token.value.c_str());
            
            // Initialize real_name to nullptr (will be set if :real modifier is used)
            ast.operation.extern_op.real_name = nullptr;
            
            // Check for :real modifier
            token = tokenizer.nextToken();
            if (token.type == TOKEN_SYMBOL && token.value == ":real") {
                // Parse the real function name
                token = tokenizer.nextToken();
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("extern :real requires function name as argument");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                ast.operation.extern_op.real_name = new char[strlen(token.value.c_str()) + 1];
                strcpy(ast.operation.extern_op.real_name, token.value.c_str());
                
                // Get next token for parameter parsing
                token = tokenizer.nextToken();
            }
            
            // Parse parameter types
            std::vector<eshkol_ast_t> param_types;
            while (true) {
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in extern declaration");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("extern parameter types must be symbols");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                // Create parameter type AST node
                eshkol_ast_t param_type = {.type = ESHKOL_STRING};
                param_type.str_val.size = strlen(token.value.c_str());
                param_type.str_val.ptr = new char[param_type.str_val.size + 1];
                strcpy(param_type.str_val.ptr, token.value.c_str());
                
                param_types.push_back(param_type);
                
                // Get next token
                token = tokenizer.nextToken();
            }
            
            // Set up extern operation
            ast.operation.extern_op.num_params = param_types.size();
            if (ast.operation.extern_op.num_params > 0) {
                ast.operation.extern_op.parameters = new eshkol_ast_t[ast.operation.extern_op.num_params];
                for (size_t i = 0; i < ast.operation.extern_op.num_params; i++) {
                    ast.operation.extern_op.parameters[i] = param_types[i];
                }
            } else {
                ast.operation.extern_op.parameters = nullptr;
            }
            
            return ast;
        } else if (ast.operation.op == ESHKOL_IMPORT_OP) {
            // Legacy syntax: (import "path/to/file.esk")
            token = tokenizer.nextToken();
            if (token.type != TOKEN_STRING) {
                eshkol_error("import requires a string path as argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.import_op.path = new char[strlen(token.value.c_str()) + 1];
            strcpy(ast.operation.import_op.path, token.value.c_str());

            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("import takes exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_REQUIRE_OP) {
            // Module system syntax: (require module.name ...)
            // Module names are symbolic (e.g., data.json, core.strings)
            std::vector<std::string> modules;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in require");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("require expects symbolic module names (e.g., data.json)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                modules.push_back(token.value);
            }

            if (modules.empty()) {
                eshkol_error("require expects at least one module name");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Allocate and copy module names
            ast.operation.require_op.num_modules = modules.size();
            ast.operation.require_op.module_names = new char*[modules.size()];
            for (size_t i = 0; i < modules.size(); i++) {
                ast.operation.require_op.module_names[i] = new char[modules[i].length() + 1];
                strcpy(ast.operation.require_op.module_names[i], modules[i].c_str());
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_PROVIDE_OP) {
            // Module system syntax: (provide name1 name2 ...)
            // Export names are symbols
            std::vector<std::string> exports;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in provide");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("provide expects symbol names to export");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                exports.push_back(token.value);
            }

            if (exports.empty()) {
                eshkol_error("provide expects at least one export name");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Allocate and copy export names
            ast.operation.provide_op.num_exports = exports.size();
            ast.operation.provide_op.export_names = new char*[exports.size()];
            for (size_t i = 0; i < exports.size(); i++) {
                ast.operation.provide_op.export_names[i] = new char[exports[i].length() + 1];
                strcpy(ast.operation.provide_op.export_names[i], exports[i].c_str());
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_WITH_REGION_OP) {
            // Memory management syntax: (with-region body ...)
            // or: (with-region 'name body ...)
            // or: (with-region ('name size) body ...)
            ast.operation.with_region_op.name = nullptr;
            ast.operation.with_region_op.size_hint = 0;

            token = tokenizer.nextToken();

            // Check for optional region name or (name size) pair
            if (token.type == TOKEN_QUOTE) {
                // (with-region 'name body ...)
                Token name_token = tokenizer.nextToken();
                if (name_token.type != TOKEN_SYMBOL) {
                    eshkol_error("with-region name must be a symbol");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                ast.operation.with_region_op.name = strdup(name_token.value.c_str());
                token = tokenizer.nextToken();
            } else if (token.type == TOKEN_LPAREN) {
                // Could be ('name size) or a body expression
                Token peek = tokenizer.nextToken();
                if (peek.type == TOKEN_QUOTE) {
                    // ('name size) - named region with size hint
                    Token name_token = tokenizer.nextToken();
                    if (name_token.type != TOKEN_SYMBOL) {
                        eshkol_error("with-region name must be a symbol");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    ast.operation.with_region_op.name = strdup(name_token.value.c_str());

                    Token size_token = tokenizer.nextToken();
                    if (size_token.type == TOKEN_NUMBER) {
                        ast.operation.with_region_op.size_hint = std::stoull(size_token.value);
                    }

                    Token close = tokenizer.nextToken();
                    if (close.type != TOKEN_RPAREN) {
                        eshkol_error("Expected closing paren in with-region name/size spec");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    token = tokenizer.nextToken();
                } else {
                    // It's a body expression starting with (, need to reconstruct the list
                    // We already consumed the opening '(' and got peek as the first element
                    // We need to build a complete list AST: (operator arg1 arg2 ...)

                    // Build the first body expression as a call/list
                    eshkol_ast_t first_body;
                    first_body.type = ESHKOL_OP;
                    first_body.operation.op = get_operator_type(peek.value);

                    // If it's a call, set up the function and arguments
                    if (first_body.operation.op == ESHKOL_CALL_OP) {
                        first_body.operation.call_op.func = new eshkol_ast_t;
                        first_body.operation.call_op.func->type = ESHKOL_VAR;
                        first_body.operation.call_op.func->variable.id = strdup(peek.value.c_str());
                        first_body.operation.call_op.func->variable.data = nullptr;

                        // Parse arguments until we hit the closing paren of this list
                        std::vector<eshkol_ast_t> args;
                        while (true) {
                            Token arg_token = tokenizer.nextToken();
                            if (arg_token.type == TOKEN_RPAREN) break;
                            if (arg_token.type == TOKEN_EOF) {
                                eshkol_error("Unexpected end of input in with-region body list");
                                ast.type = ESHKOL_INVALID;
                                return ast;
                            }
                            eshkol_ast_t arg;
                            if (arg_token.type == TOKEN_LPAREN) {
                                arg = parse_list(tokenizer);
                            } else {
                                arg = parse_atom(arg_token);
                            }
                            args.push_back(arg);
                        }

                        first_body.operation.call_op.num_vars = args.size();
                        if (!args.empty()) {
                            first_body.operation.call_op.variables = new eshkol_ast_t[args.size()];
                            for (size_t i = 0; i < args.size(); i++) {
                                first_body.operation.call_op.variables[i] = args[i];
                            }
                        } else {
                            first_body.operation.call_op.variables = nullptr;
                        }
                    }

                    // Now continue parsing more body expressions for with-region
                    std::vector<eshkol_ast_t> body_elements;
                    body_elements.push_back(first_body);

                    while (true) {
                        token = tokenizer.nextToken();
                        if (token.type == TOKEN_RPAREN) break;  // End of with-region
                        if (token.type == TOKEN_EOF) {
                            eshkol_error("Unexpected end of input in with-region");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        eshkol_ast_t body_expr;
                        if (token.type == TOKEN_LPAREN) {
                            body_expr = parse_list(tokenizer);
                        } else {
                            body_expr = parse_atom(token);
                        }
                        body_elements.push_back(body_expr);
                    }

                    ast.operation.with_region_op.num_body_exprs = body_elements.size();
                    ast.operation.with_region_op.body = new eshkol_ast_t[body_elements.size()];
                    for (size_t i = 0; i < body_elements.size(); i++) {
                        ast.operation.with_region_op.body[i] = body_elements[i];
                    }
                    return ast;
                }
            }

            // Parse body expressions
            std::vector<eshkol_ast_t> body_exprs;
            while (token.type != TOKEN_RPAREN) {
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in with-region body");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                eshkol_ast_t body_expr;
                if (token.type == TOKEN_LPAREN) {
                    body_expr = parse_list(tokenizer);
                } else {
                    body_expr = parse_atom(token);
                }
                body_exprs.push_back(body_expr);
                token = tokenizer.nextToken();
            }

            if (body_exprs.empty()) {
                eshkol_error("with-region requires at least one body expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.with_region_op.num_body_exprs = body_exprs.size();
            ast.operation.with_region_op.body = new eshkol_ast_t[body_exprs.size()];
            for (size_t i = 0; i < body_exprs.size(); i++) {
                ast.operation.with_region_op.body[i] = body_exprs[i];
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_OWNED_OP) {
            // Memory management syntax: (owned expr)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("owned requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t value_expr;
            if (token.type == TOKEN_LPAREN) {
                value_expr = parse_list(tokenizer);
            } else {
                value_expr = parse_atom(token);
            }

            ast.operation.owned_op.value = new eshkol_ast_t;
            *ast.operation.owned_op.value = value_expr;

            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("owned requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_MOVE_OP) {
            // Memory management syntax: (move value)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("move requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t value_expr;
            if (token.type == TOKEN_LPAREN) {
                value_expr = parse_list(tokenizer);
            } else {
                value_expr = parse_atom(token);
            }

            ast.operation.move_op.value = new eshkol_ast_t;
            *ast.operation.move_op.value = value_expr;

            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("move requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_BORROW_OP) {
            // Memory management syntax: (borrow value body ...)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("borrow requires a value and body expressions");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse the value to borrow
            eshkol_ast_t value_expr;
            if (token.type == TOKEN_LPAREN) {
                value_expr = parse_list(tokenizer);
            } else {
                value_expr = parse_atom(token);
            }

            ast.operation.borrow_op.value = new eshkol_ast_t;
            *ast.operation.borrow_op.value = value_expr;

            // Parse body expressions
            std::vector<eshkol_ast_t> body_exprs;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in borrow body");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                eshkol_ast_t body_expr;
                if (token.type == TOKEN_LPAREN) {
                    body_expr = parse_list(tokenizer);
                } else {
                    body_expr = parse_atom(token);
                }
                body_exprs.push_back(body_expr);
            }

            if (body_exprs.empty()) {
                eshkol_error("borrow requires at least one body expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            ast.operation.borrow_op.num_body_exprs = body_exprs.size();
            ast.operation.borrow_op.body = new eshkol_ast_t[body_exprs.size()];
            for (size_t i = 0; i < body_exprs.size(); i++) {
                ast.operation.borrow_op.body[i] = body_exprs[i];
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_SHARED_OP) {
            // Memory management syntax: (shared expr)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("shared requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t value_expr;
            if (token.type == TOKEN_LPAREN) {
                value_expr = parse_list(tokenizer);
            } else {
                value_expr = parse_atom(token);
            }

            ast.operation.shared_op.value = new eshkol_ast_t;
            *ast.operation.shared_op.value = value_expr;

            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("shared requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            return ast;
        } else if (ast.operation.op == ESHKOL_WEAK_REF_OP) {
            // Memory management syntax: (weak-ref shared-value)
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("weak-ref requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            eshkol_ast_t value_expr;
            if (token.type == TOKEN_LPAREN) {
                value_expr = parse_list(tokenizer);
            } else {
                value_expr = parse_atom(token);
            }

            ast.operation.weak_ref_op.value = new eshkol_ast_t;
            *ast.operation.weak_ref_op.value = value_expr;

            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("weak-ref requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            return ast;
        }

        // Parse arguments for non-define operations
        // Special handling for quote - its argument can be any data, including (1 2 3)
        if (ast.operation.op == ESHKOL_QUOTE_OP) {
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("quote requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            if (token.type == TOKEN_EOF) {
                eshkol_error("Unexpected end of input in quote");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse the quoted expression using our special quoted data parser
            eshkol_ast_t quoted = parse_quoted_data_with_token(tokenizer, token);
            if (quoted.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            elements.push_back(quoted);

            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("quote requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
        } else {
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in list");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_ast_t element;
                if (token.type == TOKEN_LPAREN) {
                    element = parse_list(tokenizer);
                } else if (token.type == TOKEN_QUOTE) {
                    // Handle quoted expressions as arguments: (car '(1 2 3))
                    eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                    if (quoted.type == ESHKOL_INVALID) {
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    element.type = ESHKOL_OP;
                    element.operation.op = ESHKOL_QUOTE_OP;
                    element.operation.call_op.func = nullptr;
                    element.operation.call_op.num_vars = 1;
                    element.operation.call_op.variables = new eshkol_ast_t[1];
                    element.operation.call_op.variables[0] = quoted;
                } else {
                    element = parse_atom(token);
                }
                elements.push_back(element);
            }
        }

        // Set up operation based on type and arguments
        if (ast.operation.op == ESHKOL_CALL_OP) {
            // Create function name AST node
            ast.operation.call_op.func = new eshkol_ast_t;
            ast.operation.call_op.func->type = ESHKOL_VAR;
            ast.operation.call_op.func->variable.id = new char[first_symbol.length() + 1];
            strcpy(ast.operation.call_op.func->variable.id, first_symbol.c_str());
            ast.operation.call_op.func->variable.data = nullptr;

            // For function calls, allocate variables array for arguments
            ast.operation.call_op.num_vars = elements.size();
            if (ast.operation.call_op.num_vars > 0) {
                ast.operation.call_op.variables = new eshkol_ast_t[ast.operation.call_op.num_vars];
                for (size_t i = 0; i < ast.operation.call_op.num_vars; i++) {
                    ast.operation.call_op.variables[i] = elements[i];
                }
            } else {
                ast.operation.call_op.variables = nullptr;
            }
        } else if (ast.operation.op == ESHKOL_COND_OP) {
            // cond uses the call_op structure to hold its clauses
            // Each clause is a list (test expr...) parsed into elements
            ast.operation.call_op.func = nullptr;  // cond doesn't have a function
            ast.operation.call_op.num_vars = elements.size();
            if (ast.operation.call_op.num_vars > 0) {
                ast.operation.call_op.variables = new eshkol_ast_t[ast.operation.call_op.num_vars];
                for (size_t i = 0; i < ast.operation.call_op.num_vars; i++) {
                    ast.operation.call_op.variables[i] = elements[i];
                }
            } else {
                ast.operation.call_op.variables = nullptr;
            }
        } else if (ast.operation.op == ESHKOL_WHEN_OP || ast.operation.op == ESHKOL_UNLESS_OP) {
            // when/unless uses call_op structure: test, expr1, expr2, ...
            // All elements (test and body expressions) are stored in variables
            ast.operation.call_op.func = nullptr;
            ast.operation.call_op.num_vars = elements.size();
            if (ast.operation.call_op.num_vars > 0) {
                ast.operation.call_op.variables = new eshkol_ast_t[ast.operation.call_op.num_vars];
                for (size_t i = 0; i < ast.operation.call_op.num_vars; i++) {
                    ast.operation.call_op.variables[i] = elements[i];
                }
            } else {
                ast.operation.call_op.variables = nullptr;
            }
        } else if (ast.operation.op == ESHKOL_QUOTE_OP) {
            // quote takes exactly one argument - the expression to quote
            // Store it in call_op.variables[0] for codegenQuotedAST to access
            if (elements.size() != 1) {
                eshkol_error("quote requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            ast.operation.call_op.func = nullptr;
            ast.operation.call_op.num_vars = 1;
            ast.operation.call_op.variables = new eshkol_ast_t[1];
            ast.operation.call_op.variables[0] = elements[0];
        }
    } else {
        eshkol_error("Expected symbol as first element of list");
        ast.type = ESHKOL_INVALID;
    }
    
    return ast;
}

static eshkol_ast_t parse_expression(SchemeTokenizer& tokenizer) {
    Token token = tokenizer.nextToken();

    switch (token.type) {
        case TOKEN_LPAREN:
            return parse_list(tokenizer);

        case TOKEN_VECTOR_START: {
            // Handle vector literal: #(element1 element2 ...)
            // Creates a 1D tensor (same as (vector ...))
            eshkol_ast_t ast = {};  // Zero-initialize all fields
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_TENSOR_OP;

            std::vector<eshkol_ast_t> elements;

            while (true) {
                Token elem_token = tokenizer.nextToken();
                if (elem_token.type == TOKEN_RPAREN) break;
                if (elem_token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in vector literal #(...)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Put token back and parse as expression
                // Since we can't easily "unget" a token, handle atoms directly here
                eshkol_ast_t element;
                if (elem_token.type == TOKEN_LPAREN) {
                    element = parse_list(tokenizer);
                } else if (elem_token.type == TOKEN_VECTOR_START) {
                    // Nested vector - recursively handle
                    // Put this logic in a helper or handle inline
                    // For simplicity, create a temporary tokenizer state
                    eshkol_error("Nested vector literals not yet supported");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                } else {
                    element = parse_atom(elem_token);
                }

                if (element.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                elements.push_back(element);
            }

            // Set up 1D tensor (vector)
            ast.operation.tensor_op.num_dimensions = 1;
            ast.operation.tensor_op.dimensions = new uint64_t[1];
            ast.operation.tensor_op.dimensions[0] = elements.size();
            ast.operation.tensor_op.total_elements = elements.size();
            ast.operation.tensor_op.elements = new eshkol_ast_t[elements.size()];
            for (size_t i = 0; i < elements.size(); i++) {
                ast.operation.tensor_op.elements[i] = elements[i];
            }

            return ast;
        }

        case TOKEN_QUOTE: {
            // Handle quoted expressions - use parse_quoted_data for proper data list handling
            eshkol_ast_t quoted_expr = parse_quoted_data(tokenizer);
            if (quoted_expr.type == ESHKOL_INVALID) {
                return quoted_expr;
            }

            // Create a quote operation
            eshkol_ast_t ast;
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_QUOTE_OP;
            ast.operation.call_op.func = nullptr;
            ast.operation.call_op.num_vars = 1;
            ast.operation.call_op.variables = new eshkol_ast_t[1];
            ast.operation.call_op.variables[0] = quoted_expr;
            return ast;
        }

        case TOKEN_BACKQUOTE: {
            // Handle quasiquoted expressions - `expr becomes (quasiquote expr)
            eshkol_ast_t inner_expr = parse_expression(tokenizer);
            if (inner_expr.type == ESHKOL_INVALID) {
                return inner_expr;
            }

            // Create a quasiquote operation
            eshkol_ast_t ast;
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_QUASIQUOTE_OP;
            ast.operation.call_op.func = nullptr;
            ast.operation.call_op.num_vars = 1;
            ast.operation.call_op.variables = new eshkol_ast_t[1];
            ast.operation.call_op.variables[0] = inner_expr;
            return ast;
        }

        case TOKEN_COMMA: {
            // Handle unquote - ,expr becomes (unquote expr)
            eshkol_ast_t inner_expr = parse_expression(tokenizer);
            if (inner_expr.type == ESHKOL_INVALID) {
                return inner_expr;
            }

            // Create an unquote operation
            eshkol_ast_t ast;
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_UNQUOTE_OP;
            ast.operation.call_op.func = nullptr;
            ast.operation.call_op.num_vars = 1;
            ast.operation.call_op.variables = new eshkol_ast_t[1];
            ast.operation.call_op.variables[0] = inner_expr;
            return ast;
        }

        case TOKEN_COMMA_AT: {
            // Handle unquote-splicing - ,@expr becomes (unquote-splicing expr)
            eshkol_ast_t inner_expr = parse_expression(tokenizer);
            if (inner_expr.type == ESHKOL_INVALID) {
                return inner_expr;
            }

            // Create an unquote-splicing operation
            eshkol_ast_t ast;
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_UNQUOTE_SPLICING_OP;
            ast.operation.call_op.func = nullptr;
            ast.operation.call_op.num_vars = 1;
            ast.operation.call_op.variables = new eshkol_ast_t[1];
            ast.operation.call_op.variables[0] = inner_expr;
            return ast;
        }

        case TOKEN_SYMBOL:
        case TOKEN_STRING:
        case TOKEN_NUMBER:
        case TOKEN_BOOLEAN:
        case TOKEN_CHAR:
            return parse_atom(token);

        case TOKEN_RPAREN:
            eshkol_error("Unexpected closing parenthesis");
            return {.type = ESHKOL_INVALID};

        case TOKEN_EOF:
        default:
            return {.type = ESHKOL_INVALID};
    }
}

// Generic stream-based parser (works with any std::istream)
eshkol_ast_t eshkol_parse_next_ast_from_stream(std::istream &in_stream)
{
    std::string input;
    std::string line;
    bool in_quote = false;
    int bracket_depth = 0;
    bool found_expression = false;

    // Read characters until we have a complete S-expression
    while (!in_stream.eof()) {
        int c = in_stream.get();
        if (in_stream.eof()) break;

        // Handle comments - skip to end of line
        if (c == ';' && !in_quote) {
            std::getline(in_stream, line); // consume rest of line
            if (bracket_depth == 0 && !input.empty()) {
                input += ' '; // Add space to separate from next token
            }
            continue;
        }

        // Track quotes - a quote is escaped only if preceded by ODD number of backslashes
        if (c == '"') {
            // Count trailing backslashes in input
            size_t backslash_count = 0;
            for (size_t i = input.size(); i > 0 && input[i-1] == '\\'; i--) {
                backslash_count++;
            }
            // Quote is escaped only if odd number of backslashes precede it
            if (backslash_count % 2 == 0) {
                in_quote = !in_quote;
            }
        }

        input += static_cast<char>(c);

        // Track parentheses depth (only outside quotes)
        if (!in_quote) {
            if (c == '(') {
                bracket_depth++;
                found_expression = true;
            } else if (c == ')') {
                bracket_depth--;
                if (bracket_depth == 0 && found_expression) {
                    break; // Complete expression found
                }
            } else if (!std::isspace(c) && bracket_depth == 0) {
                // Found atom at top level - read until whitespace or special char
                while (!in_stream.eof()) {
                    int next_c = in_stream.peek();
                    if (std::isspace(next_c) || next_c == '(' || next_c == ')' || next_c == ';') {
                        break;
                    }
                    c = in_stream.get();
                    if (c == '"') in_quote = !in_quote;
                    input += static_cast<char>(c);
                }
                found_expression = true;
                break;
            }
        }
    }

    // Parse the collected input
    if (!input.empty() && found_expression) {
        // Remove leading/trailing whitespace
        size_t start = input.find_first_not_of(" \t\n\r");
        size_t end = input.find_last_not_of(" \t\n\r");

        if (start != std::string::npos && end != std::string::npos) {
            input = input.substr(start, end - start + 1);

            SchemeTokenizer tokenizer(input);
            return parse_expression(tokenizer);
        }
    }

    return {.type = ESHKOL_INVALID};
}

// File-based parser (wrapper for backwards compatibility)
eshkol_ast_t eshkol_parse_next_ast(std::ifstream &in_file)
{
    return eshkol_parse_next_ast_from_stream(in_file);
}
