/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/core/logic.h>
#include <eshkol/logger.h>

#include <string.h>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <vector>
#include <set>
#include <pthread.h>

// Stack space check: detect remaining stack and bail before overflow.
// Uses platform APIs to measure actual stack consumption rather than
// imposing an arbitrary depth limit.
static bool check_stack_space() {
    static const size_t STACK_SAFETY_MARGIN = 65536; // 64 KB reserved
#ifdef __APPLE__
    pthread_t self = pthread_self();
    void* stack_addr = pthread_get_stackaddr_np(self);
    size_t stack_size = pthread_get_stacksize_np(self);
    char local_var;
    // On macOS, stack_addr is the TOP (highest address) of the stack
    size_t used = (size_t)((char*)stack_addr - &local_var);
    return (stack_size > used) && ((stack_size - used) > STACK_SAFETY_MARGIN);
#elif defined(__linux__)
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) != 0) return true;
    void* stack_addr;
    size_t stack_size;
    pthread_attr_getstack(&attr, &stack_addr, &stack_size);
    pthread_attr_destroy(&attr);
    char local_var;
    // On Linux, stack_addr is the BOTTOM (lowest address) of the stack
    size_t used = (size_t)(&local_var - (char*)stack_addr);
    // If local_var is below stack_addr, we have a problem anyway
    if (&local_var < (char*)stack_addr) return true;
    return used > STACK_SAFETY_MARGIN;
#else
    // Unknown platform: assume stack is fine
    return true;
#endif
}

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
                } else if ((ch == '+' || ch == '-') && pos + 4 < length &&
                           (input.substr(pos + 1, 4) == "inf." || input.substr(pos + 1, 4) == "nan.")) {
                    // R7RS special float literals: +inf.0, -inf.0, +nan.0, -nan.0
                    size_t start_pos = pos;
                    std::string value;
                    while (pos < length && !std::isspace(input[pos]) &&
                           input[pos] != '(' && input[pos] != ')' &&
                           input[pos] != '\'' && input[pos] != '"') {
                        value += input[pos++];
                        column_++;
                    }
                    if (value == "+inf.0" || value == "-inf.0" ||
                        value == "+nan.0" || value == "-nan.0") {
                        return {TOKEN_NUMBER, value, start_pos, tok_line, tok_col};
                    }
                    // Not a special literal, treat as symbol
                    pos = start_pos;
                    column_ = tok_col;
                    return readSymbol();
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

        // Handle rational literals (e.g., 1/3, 22/7)
        // Only if we haven't seen a decimal point and next char is '/'
        if (pos < length && input[pos] == '/' &&
            value.find('.') == std::string::npos &&
            pos + 1 < length && std::isdigit(input[pos + 1])) {
            value += input[pos++]; // consume '/'
            column_++;
            while (pos < length && std::isdigit(input[pos])) {
                value += input[pos++];
                column_++;
            }
            return {TOKEN_NUMBER, value, start, tok_line, tok_col};
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
        case TOKEN_STRING: {
            size_t _len = token.value.length();
            char* ptr = new char[_len + 1];
            if (ptr) memcpy(ptr, token.value.c_str(), _len + 1);
            eshkol_ast_make_string(&ast, ptr, _len + 1);
            break;
        }

        case TOKEN_NUMBER: {
            // R7RS special float literals
            if (token.value == "+inf.0") {
                eshkol_ast_make_double(&ast, std::numeric_limits<double>::infinity());
                break;
            }
            if (token.value == "-inf.0") {
                eshkol_ast_make_double(&ast, -std::numeric_limits<double>::infinity());
                break;
            }
            if (token.value == "+nan.0" || token.value == "-nan.0") {
                eshkol_ast_make_double(&ast, std::numeric_limits<double>::quiet_NaN());
                break;
            }

            // Check if it's a rational literal (e.g., 1/3, 22/7)
            if (token.value.find('/') != std::string::npos) {
                size_t slash_pos = token.value.find('/');
                std::string num_str = token.value.substr(0, slash_pos);
                std::string den_str = token.value.substr(slash_pos + 1);
                int64_t num, den;
                try {
                    num = std::stoll(num_str);
                    den = std::stoll(den_str);
                } catch (...) {
                    eshkol_error("invalid rational literal: %s", token.value.c_str());
                    break;
                }
                // Create (make-rational num den) call AST
                ast.type = ESHKOL_OP;
                ast.operation.op = ESHKOL_CALL_OP;
                ast.operation.call_op.func = new eshkol_ast_t;
                ast.operation.call_op.func->type = ESHKOL_VAR;
                ast.operation.call_op.func->variable.id = new char[sizeof("make-rational")];
                memcpy(ast.operation.call_op.func->variable.id, "make-rational", sizeof("make-rational"));
                ast.operation.call_op.func->variable.data = nullptr;
                ast.operation.call_op.num_vars = 2;
                ast.operation.call_op.variables = new eshkol_ast_t[2];
                eshkol_ast_make_int64(&ast.operation.call_op.variables[0], num);
                eshkol_ast_make_int64(&ast.operation.call_op.variables[1], den);
                break;
            }
            // Check if it's a floating-point number (has '.' or scientific notation 'e'/'E')
            if (token.value.find('.') != std::string::npos ||
                token.value.find('e') != std::string::npos ||
                token.value.find('E') != std::string::npos) {
                char* endptr = nullptr;
                double dval = strtod(token.value.c_str(), &endptr);
                if (endptr == token.value.c_str()) {
                    eshkol_error("invalid numeric literal: %s", token.value.c_str());
                    break;
                }
                eshkol_ast_make_double(&ast, dval);
            } else {
                try {
                    eshkol_ast_make_int64(&ast, std::stoll(token.value));
                } catch (const std::out_of_range&) {
                    // Integer literal too large for int64 — store as string for bignum construction at codegen
                    ast.type = ESHKOL_BIGNUM_LITERAL;
                    size_t _len = token.value.length();
                    char* ptr = new char[_len + 1];
                    if (ptr) memcpy(ptr, token.value.c_str(), _len + 1);
                    ast.str_val.ptr = ptr;
                    ast.str_val.size = _len + 1;
                }
            }
            break;
        }

        case TOKEN_BOOLEAN:
            eshkol_ast_make_bool(&ast, token.value == "#t");
            break;

        case TOKEN_CHAR:
            eshkol_ast_make_char(&ast, static_cast<unsigned char>(token.value[0]));
            break;

        case TOKEN_SYMBOL:
            // Check for logic variable syntax: ?x, ?name, etc.
            if (!token.value.empty() && token.value[0] == '?' && token.value.length() > 1) {
                ast.type = ESHKOL_OP;
                ast.operation.op = ESHKOL_LOGIC_VAR_OP;
                uint64_t var_id = eshkol_make_logic_var(token.value.c_str());
                ast.operation.logic_var_op.var_id = var_id;
                size_t _len = token.value.length();
                ast.operation.logic_var_op.name = new char[_len + 1];
                if (ast.operation.logic_var_op.name) {
                    memcpy(const_cast<char*>(ast.operation.logic_var_op.name), token.value.c_str(), _len + 1);
                }
            } else {
                ast.type = ESHKOL_VAR;
                size_t _len = token.value.length();
                ast.variable.id = new char[_len + 1];
                if (ast.variable.id) {
                    memcpy(ast.variable.id, token.value.c_str(), _len + 1);
                }
                ast.variable.data = nullptr;
            }
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
    if (op == "letrec*") return ESHKOL_LETREC_STAR_OP;
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
    if (op == "let-syntax") return ESHKOL_LET_SYNTAX_OP;
    if (op == "letrec-syntax") return ESHKOL_LETREC_SYNTAX_OP;
    if (op == "call/cc") return ESHKOL_CALL_CC_OP;
    if (op == "call-with-current-continuation") return ESHKOL_CALL_CC_OP;
    if (op == "dynamic-wind") return ESHKOL_DYNAMIC_WIND_OP;
    if (op == "set!") return ESHKOL_SET_OP;
    if (op == "import") return ESHKOL_IMPORT_OP;
    if (op == "require") return ESHKOL_REQUIRE_OP;  // module system: (require module.name ...)
    if (op == "load") return ESHKOL_REQUIRE_OP;     // R7RS load: alias for require with file path
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
    if (op == "differentiate") return ESHKOL_DIFF_OP;
    // Automatic differentiation operators
    if (op == "derivative") return ESHKOL_DERIVATIVE_OP;
    if (op == "D") return ESHKOL_DERIVATIVE_OP;
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
    // Neuro-symbolic consciousness engine operations
    if (op == "unify") return ESHKOL_UNIFY_OP;
    if (op == "make-substitution") return ESHKOL_MAKE_SUBST_OP;
    if (op == "walk") return ESHKOL_WALK_OP;
    if (op == "make-fact") return ESHKOL_MAKE_FACT_OP;
    if (op == "make-kb") return ESHKOL_MAKE_KB_OP;
    if (op == "kb-assert!") return ESHKOL_KB_ASSERT_OP;
    if (op == "kb-query") return ESHKOL_KB_QUERY_OP;
    if (op == "logic-var?") return ESHKOL_LOGIC_VAR_PRED_OP;
    if (op == "substitution?") return ESHKOL_SUBSTITUTION_PRED_OP;
    if (op == "kb?") return ESHKOL_KB_PRED_OP;
    // Active inference operations
    if (op == "make-factor-graph") return ESHKOL_MAKE_FACTOR_GRAPH_OP;
    if (op == "fg-add-factor!") return ESHKOL_FG_ADD_FACTOR_OP;
    if (op == "fg-infer!") return ESHKOL_FG_INFER_OP;
    if (op == "fg-observe!") return ESHKOL_FG_OBSERVE_OP;
    if (op == "free-energy") return ESHKOL_FREE_ENERGY_OP;
    if (op == "expected-free-energy") return ESHKOL_EXPECTED_FREE_ENERGY_OP;
    // Global workspace operations
    if (op == "make-workspace") return ESHKOL_MAKE_WORKSPACE_OP;
    if (op == "ws-register!") return ESHKOL_WS_REGISTER_OP;
    if (op == "ws-step!") return ESHKOL_WS_STEP_OP;
    if (op == "fg-update-cpt!") return ESHKOL_FG_UPDATE_CPT_OP;
    // Type predicates
    if (op == "fact?") return ESHKOL_FACT_PRED_OP;
    if (op == "factor-graph?") return ESHKOL_FACTOR_GRAPH_PRED_OP;
    if (op == "workspace?") return ESHKOL_WORKSPACE_PRED_OP;
    // R7RS Wave 3 special forms
    if (op == "case-lambda") return ESHKOL_CASE_LAMBDA_OP;
    if (op == "define-record-type") return ESHKOL_DEFINE_RECORD_TYPE_OP;
    if (op == "parameterize") return ESHKOL_PARAMETERIZE_OP;
    if (op == "make-parameter") return ESHKOL_MAKE_PARAMETER_OP;
    if (op == "cond-expand") return ESHKOL_COND_EXPAND_OP;
    if (op == "include") return ESHKOL_INCLUDE_OP;
    if (op == "include-ci") return ESHKOL_INCLUDE_OP;  // handled same as include
    if (op == "syntax-error") return ESHKOL_SYNTAX_ERROR_OP;
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
    ast.operation.call_op.func->variable.id = new char[sizeof("list")];
    memcpy(ast.operation.call_op.func->variable.id, "list", sizeof("list"));
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

// Forward declarations for quasiquote parsing
static eshkol_ast_t parse_quasiquoted_data(SchemeTokenizer& tokenizer);
static eshkol_ast_t parse_quasiquoted_data_with_token(SchemeTokenizer& tokenizer, Token token);
static eshkol_ast_t parse_quasiquoted_list_internal(SchemeTokenizer& tokenizer);

// Parse quasiquoted data - similar to quoted data but handles unquote/unquote-splicing
static eshkol_ast_t parse_quasiquoted_data(SchemeTokenizer& tokenizer) {
    Token token = tokenizer.nextToken();
    return parse_quasiquoted_data_with_token(tokenizer, token);
}

// Parse quasiquoted data when we already have the first token
static eshkol_ast_t parse_quasiquoted_data_with_token(SchemeTokenizer& tokenizer, Token token) {
    if (token.type == TOKEN_LPAREN) {
        // Parse a list without requiring a symbol as first element
        return parse_quasiquoted_list_internal(tokenizer);
    } else if (token.type == TOKEN_COMMA) {
        // Unquote: ,expr - parse as code (not data)
        eshkol_ast_t inner = parse_quasiquoted_data(tokenizer);
        eshkol_ast_t ast;
        ast.type = ESHKOL_OP;
        ast.operation.op = ESHKOL_UNQUOTE_OP;
        ast.operation.call_op.func = nullptr;
        ast.operation.call_op.num_vars = 1;
        ast.operation.call_op.variables = new eshkol_ast_t[1];
        ast.operation.call_op.variables[0] = inner;
        return ast;
    } else if (token.type == TOKEN_COMMA_AT) {
        // Unquote-splicing: ,@expr - parse as code
        eshkol_ast_t inner = parse_quasiquoted_data(tokenizer);
        eshkol_ast_t ast;
        ast.type = ESHKOL_OP;
        ast.operation.op = ESHKOL_UNQUOTE_SPLICING_OP;
        ast.operation.call_op.func = nullptr;
        ast.operation.call_op.num_vars = 1;
        ast.operation.call_op.variables = new eshkol_ast_t[1];
        ast.operation.call_op.variables[0] = inner;
        return ast;
    } else if (token.type == TOKEN_QUOTE) {
        // Nested quote inside quasiquote
        eshkol_ast_t quoted = parse_quoted_data(tokenizer);
        eshkol_ast_t ast;
        ast.type = ESHKOL_OP;
        ast.operation.op = ESHKOL_QUOTE_OP;
        ast.operation.call_op.func = nullptr;
        ast.operation.call_op.num_vars = 1;
        ast.operation.call_op.variables = new eshkol_ast_t[1];
        ast.operation.call_op.variables[0] = quoted;
        return ast;
    } else if (token.type == TOKEN_BACKQUOTE) {
        // Nested quasiquote
        eshkol_ast_t inner = parse_quasiquoted_data(tokenizer);
        eshkol_ast_t ast;
        ast.type = ESHKOL_OP;
        ast.operation.op = ESHKOL_QUASIQUOTE_OP;
        ast.operation.call_op.func = nullptr;
        ast.operation.call_op.num_vars = 1;
        ast.operation.call_op.variables = new eshkol_ast_t[1];
        ast.operation.call_op.variables[0] = inner;
        return ast;
    } else {
        // Atom
        return parse_atom(token);
    }
}

// Parse a quasiquoted list - called after consuming '('
static eshkol_ast_t parse_quasiquoted_list_internal(SchemeTokenizer& tokenizer) {
    std::vector<eshkol_ast_t> elements;

    while (true) {
        Token inner_token = tokenizer.nextToken();
        if (inner_token.type == TOKEN_RPAREN) break;
        if (inner_token.type == TOKEN_EOF) {
            eshkol_error("Unexpected end of input in quasiquoted list");
            return {.type = ESHKOL_INVALID};
        }

        // Recursively parse each element (handles unquote/unquote-splicing)
        eshkol_ast_t elem = parse_quasiquoted_data_with_token(tokenizer, inner_token);
        if (elem.type == ESHKOL_INVALID) {
            return elem;
        }
        elements.push_back(elem);
    }

    // Build list as CALL_OP with "list" as function
    // This allows codegenQuasiquotedAST to handle it correctly
    eshkol_ast_t ast;
    ast.type = ESHKOL_OP;
    ast.operation.op = ESHKOL_CALL_OP;
    ast.operation.call_op.func = new eshkol_ast_t;
    ast.operation.call_op.func->type = ESHKOL_VAR;
    ast.operation.call_op.func->variable.id = new char[sizeof("list")];
    memcpy(ast.operation.call_op.func->variable.id, "list", sizeof("list"));
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

    // Collect CONSECUTIVE defines starting from first_define_idx.
    // Once a non-define expression is encountered, STOP collecting defines.
    // All remaining expressions (including any later defines) go into after_defines.
    // This preserves evaluation order: side effects between defines execute
    // at the correct time, not after all defines are processed.
    size_t define_end_idx = first_define_idx;
    for (size_t i = first_define_idx; i < body_expressions.size(); i++) {
        if (body_expressions[i].type == ESHKOL_OP &&
            body_expressions[i].operation.op == ESHKOL_DEFINE_OP &&
            after_defines.empty()) {
            // Still in the consecutive define block
            defines.push_back(body_expressions[i]);
            define_end_idx = i + 1;
        } else {
            // Non-define encountered (or we already left the define block)
            // Everything from here goes into the body
            after_defines.push_back(body_expressions[i]);
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

    eshkol_debug("Transforming %zu internal defines to letrec*", defines.size());

    // Create letrec* AST (R7RS: internal defines use letrec* semantics)
    // letrec* evaluates bindings left-to-right, each can see previous bindings
    result.type = ESHKOL_OP;
    result.operation.op = ESHKOL_LETREC_STAR_OP;

    // Convert defines to bindings
    result.operation.let_op.num_bindings = defines.size();
    result.operation.let_op.bindings = new eshkol_ast_t[defines.size()];
    result.operation.let_op.binding_types = nullptr;  // No type annotations for internal defines
    result.operation.let_op.name = nullptr;  // Not a named let

    for (size_t i = 0; i < defines.size(); i++) {
        const auto& def = defines[i];

        // Create variable node for binding
        eshkol_ast_t var_ast = {.type = ESHKOL_VAR};
        { size_t _len = strlen(def.operation.define_op.name);
        var_ast.variable.id = new char[_len + 1];
        memcpy(var_ast.variable.id, def.operation.define_op.name, _len + 1); }
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
                { size_t _len = strlen(def.operation.define_op.rest_param);
                val_ast.operation.lambda_op.rest_param = new char[_len + 1];
                memcpy(val_ast.operation.lambda_op.rest_param, def.operation.define_op.rest_param, _len + 1); }
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
        eshkol_ast_make_null(&body);
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
static eshkol_ast_t parse_vector_body(SchemeTokenizer& tokenizer);
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
    { size_t _len = token.value.length();
    signature.eshkol_func.id = new char[_len + 1];
    memcpy(signature.eshkol_func.id, token.value.c_str(), _len + 1); }
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
            { size_t _len = token.value.length();
            signature.eshkol_func.rest_param = new char[_len + 1];
            memcpy(signature.eshkol_func.rest_param, token.value.c_str(), _len + 1); }

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
            { size_t _len = param_token.value.length();
            param.variable.id = new char[_len + 1];
            memcpy(param.variable.id, param_token.value.c_str(), _len + 1); }
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
            { size_t _len = token.value.length();
            param.variable.id = new char[_len + 1];
            memcpy(param.variable.id, token.value.c_str(), _len + 1); }
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
            { size_t _len = token.value.length();
            pattern->variable.name = new char[_len + 1];
            memcpy(pattern->variable.name, token.value.c_str(), _len + 1); }
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
                { size_t _len = peek.value.length();
                list_ast.variable.id = new char[_len + 1];
                memcpy(list_ast.variable.id, peek.value.c_str(), _len + 1); }
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
            eshkol_ast_make_null(pattern->literal.value);
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
            
            // Use parse_expression for full expression support in arguments
            tokenizer.pushBack(token);
            eshkol_ast_t element = parse_expression(tokenizer);
            if (element.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
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
        { size_t _len = name_token.value.length();
        ast.operation.type_annotation_op.name = new char[_len + 1];
        memcpy(ast.operation.type_annotation_op.name, name_token.value.c_str(), _len + 1); }
        ast.operation.type_annotation_op.type_expr = type_expr;

        eshkol_debug("Parsed type annotation for '%s'", name_token.value.c_str());
        return ast;
    }

    // First element should determine the operation
    if (token.type == TOKEN_SYMBOL) {
        std::string first_symbol = token.value;  // Store the function name
        ast.operation.op = get_operator_type(token.value);

        // R7RS (delay expr) → (%make-lazy-promise (lambda () expr))
        // (delay-force expr) → (%make-lazy-promise-force (lambda () expr))
        if (first_symbol == "delay" || first_symbol == "delay-force") {
            // Parse the body expression
            eshkol_ast_t body_expr = parse_expression(tokenizer);
            if (body_expr.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("%s requires exactly one expression", first_symbol.c_str());
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            // Build: (lambda () body_expr)
            eshkol_ast_t lambda_ast = {};
            lambda_ast.type = ESHKOL_OP;
            lambda_ast.operation.op = ESHKOL_LAMBDA_OP;
            lambda_ast.operation.lambda_op.parameters = nullptr;
            lambda_ast.operation.lambda_op.num_params = 0;
            lambda_ast.operation.lambda_op.body = new eshkol_ast_t(body_expr);
            lambda_ast.operation.lambda_op.captured_vars = nullptr;
            lambda_ast.operation.lambda_op.num_captured = 0;
            lambda_ast.operation.lambda_op.is_variadic = false;
            lambda_ast.operation.lambda_op.rest_param = nullptr;
            lambda_ast.operation.lambda_op.return_type = nullptr;
            lambda_ast.operation.lambda_op.param_types = nullptr;
            // Build call: (%make-lazy-promise lambda_ast)
            const char* func_id = (first_symbol == "delay-force")
                ? "%make-lazy-promise-force" : "%make-lazy-promise";
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_CALL_OP;
            ast.operation.call_op.func = new eshkol_ast_t();
            ast.operation.call_op.func->type = ESHKOL_VAR;
            { size_t _len = strlen(func_id);
            ast.operation.call_op.func->variable.id = new char[_len + 1];
            memcpy(ast.operation.call_op.func->variable.id, func_id, _len + 1); }
            ast.operation.call_op.num_vars = 1;
            ast.operation.call_op.variables = new eshkol_ast_t[1];
            ast.operation.call_op.variables[0] = lambda_ast;
            return ast;
        }

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
                { size_t _len = strlen(func_signature.eshkol_func.id);
                ast.operation.define_op.name = new char[_len + 1];
                memcpy(ast.operation.define_op.name, func_signature.eshkol_func.id, _len + 1); }
                
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
                    { size_t _len = strlen(func_signature.eshkol_func.rest_param);
                    ast.operation.define_op.rest_param = new char[_len + 1];
                    memcpy(ast.operation.define_op.rest_param, func_signature.eshkol_func.rest_param, _len + 1); }
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

                // Parse value expression using parse_expression for full syntax support
                eshkol_ast_t value = parse_expression(tokenizer);
                
                // Check for closing paren
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing parenthesis after variable value");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                // Set up define operation for variable
                { size_t _len = strlen(name_ast.variable.id);
                ast.operation.define_op.name = new char[_len + 1];
                memcpy(ast.operation.define_op.name, name_ast.variable.id, _len + 1); }
                
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
            { size_t _len = token.value.length();
            ast.operation.set_op.name = new char[_len + 1];
            memcpy(ast.operation.set_op.name, token.value.c_str(), _len + 1); }

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
            { size_t _len = type_name.length();
            ast.operation.define_type_op.name = new char[_len + 1];
            memcpy(ast.operation.define_type_op.name, type_name.c_str(), _len + 1); }
            ast.operation.define_type_op.type_expr = type_expr;
            ast.operation.define_type_op.num_type_params = type_params.size();

            if (!type_params.empty()) {
                ast.operation.define_type_op.type_params = new char*[type_params.size()];
                for (size_t i = 0; i < type_params.size(); i++) {
                    { size_t _len = type_params[i].length();
                    ast.operation.define_type_op.type_params[i] = new char[_len + 1];
                    memcpy(ast.operation.define_type_op.type_params[i], type_params[i].c_str(), _len + 1); }
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
                eshkol_ast_make_null(&else_expr);
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
            
            // Store the if operation as a call to builtin "if" with 3 arguments.
            // The codegen recognizes "if" calls and emits proper conditional branches.
            ast.operation.op = ESHKOL_CALL_OP;
            
            // Create function name AST node for "if"
            ast.operation.call_op.func = new eshkol_ast_t;
            ast.operation.call_op.func->type = ESHKOL_VAR;
            ast.operation.call_op.func->variable.id = new char[sizeof("if")];
            memcpy(ast.operation.call_op.func->variable.id, "if", sizeof("if"));
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
                { size_t _len = token.value.length();
                ast.operation.lambda_op.rest_param = new char[_len + 1];
                memcpy(ast.operation.lambda_op.rest_param, token.value.c_str(), _len + 1); }
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
                        { size_t _len = token.value.length();
                        ast.operation.lambda_op.rest_param = new char[_len + 1];
                        memcpy(ast.operation.lambda_op.rest_param, token.value.c_str(), _len + 1); }

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
                        { size_t _len = param_token.value.length();
                        param.variable.id = new char[_len + 1];
                        memcpy(param.variable.id, param_token.value.c_str(), _len + 1); }
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
                        { size_t _len = token.value.length();
                        param.variable.id = new char[_len + 1];
                        memcpy(param.variable.id, token.value.c_str(), _len + 1); }
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
                } else if (token.type == TOKEN_QUOTE) {
                    // Handle quoted expressions in lambda body: (lambda () 'ok)
                    eshkol_ast_t quoted = parse_quoted_data(tokenizer);
                    expr.type = ESHKOL_OP;
                    expr.operation.op = ESHKOL_QUOTE_OP;
                    expr.operation.call_op.func = nullptr;
                    expr.operation.call_op.num_vars = 1;
                    expr.operation.call_op.variables = new eshkol_ast_t[1];
                    expr.operation.call_op.variables[0] = quoted;
                } else if (token.type == TOKEN_BACKQUOTE) {
                    // Handle quasiquoted expressions in lambda body: (lambda () `expr)
                    eshkol_ast_t inner = parse_quasiquoted_data(tokenizer);
                    expr.type = ESHKOL_OP;
                    expr.operation.op = ESHKOL_QUASIQUOTE_OP;
                    expr.operation.call_op.func = nullptr;
                    expr.operation.call_op.num_vars = 1;
                    expr.operation.call_op.variables = new eshkol_ast_t[1];
                    expr.operation.call_op.variables[0] = inner;
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
                    { size_t _len = captures[i].length();
                    ast.operation.lambda_op.captured_vars[i].variable.id =
                        new char[_len + 1];
                    memcpy(ast.operation.lambda_op.captured_vars[i].variable.id,
                           captures[i].c_str(), _len + 1); }
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
        
        // Special handling for let/let*/letrec/letrec* - local variable bindings
        // let: bindings can't reference each other
        // let*: sequential (each binding can reference previous ones)
        // letrec: all bindings visible to all values (mutual recursion)
        // letrec*: sequential + recursive (R7RS: left-to-right with mutual visibility)
        // We use the same parsing - codegen handles the difference
        if (ast.operation.op == ESHKOL_LET_OP || ast.operation.op == ESHKOL_LET_STAR_OP ||
            ast.operation.op == ESHKOL_LETREC_OP || ast.operation.op == ESHKOL_LETREC_STAR_OP) {
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
                { size_t _len = token.value.length();
                var_ast.variable.id = new char[_len + 1];
                memcpy(var_ast.variable.id, token.value.c_str(), _len + 1); }
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

                // Parse value expression using parse_expression for full syntax support
                eshkol_ast_t val_ast = parse_expression(tokenizer);
                
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
                { size_t _len = named_let_name.length();
                ast.operation.let_op.name = new char[_len + 1];
                memcpy(ast.operation.let_op.name, named_let_name.c_str(), _len + 1); }
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
                        { size_t _len = all_vars[i][j].length();
                        ast.operation.let_values_op.binding_vars[i][j] = new char[_len + 1];
                        memcpy(ast.operation.let_values_op.binding_vars[i][j], all_vars[i][j].c_str(), _len + 1); }
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
            { size_t _len = token.value.length();
            ast.operation.guard_op.var_name = new char[_len + 1];
            memcpy(ast.operation.guard_op.var_name, token.value.c_str(), _len + 1); }

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
                    else_marker.variable.id = new char[sizeof("else")];
                    memcpy(else_marker.variable.id, "else", sizeof("else"));
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
                    eshkol_ast_make_null(clause.body);
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

                // Recursive pattern parser (handles arbitrary nesting depth)
                std::function<void(std::vector<eshkol_macro_pattern_t*>&)> parsePatternElements;
                parsePatternElements = [&](std::vector<eshkol_macro_pattern_t*>& elements) {
                    while (true) {
                        token = tokenizer.nextToken();
                        if (token.type == TOKEN_RPAREN) break;
                        if (token.type == TOKEN_EOF) {
                            eshkol_error("Unexpected end of input in macro pattern");
                            break;
                        }
                        auto* elem = new eshkol_macro_pattern_t;
                        elem->followed_by_ellipsis = 0;
                        if (token.type == TOKEN_SYMBOL) {
                            if (token.value == "...") {
                                if (!elements.empty())
                                    elements.back()->followed_by_ellipsis = 1;
                                delete elem;
                                continue;
                            }
                            bool is_lit = false;
                            for (const auto& lit : literals) {
                                if (lit == token.value) { is_lit = true; break; }
                            }
                            elem->type = is_lit ? MACRO_PAT_LITERAL : MACRO_PAT_VARIABLE;
                            elem->identifier = strdup(token.value.c_str());
                        } else if (token.type == TOKEN_LPAREN) {
                            elem->type = MACRO_PAT_LIST;
                            elem->list.rest = nullptr;
                            std::vector<eshkol_macro_pattern_t*> nested;
                            parsePatternElements(nested);  // Recurse
                            if (!nested.empty()) {
                                elem->list.elements = new eshkol_macro_pattern_t*[nested.size()];
                                elem->list.num_elements = nested.size();
                                for (size_t j = 0; j < nested.size(); j++)
                                    elem->list.elements[j] = nested[j];
                            } else {
                                elem->list.elements = nullptr;
                                elem->list.num_elements = 0;
                            }
                        } else {
                            elem->type = MACRO_PAT_LITERAL;
                            elem->identifier = strdup(token.value.c_str());
                        }
                        elements.push_back(elem);
                    }
                };

                // Parse pattern elements
                std::vector<eshkol_macro_pattern_t*> pat_elements;
                parsePatternElements(pat_elements);

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

        // Special handling for let-syntax / letrec-syntax - local macro bindings
        // (let-syntax ((name (syntax-rules ...)) ...) body ...)
        if (ast.operation.op == ESHKOL_LET_SYNTAX_OP ||
            ast.operation.op == ESHKOL_LETREC_SYNTAX_OP) {

            // Parse bindings list: ((name1 (syntax-rules ...)) (name2 (syntax-rules ...)) ...)
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("let-syntax requires bindings list");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            std::vector<eshkol_macro_def_t*> macro_defs;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;  // End of bindings list
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in let-syntax bindings");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("let-syntax binding must be (name (syntax-rules ...))");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse macro name
                token = tokenizer.nextToken();
                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("let-syntax binding requires a name");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                eshkol_macro_def_t* macro = new eshkol_macro_def_t;
                macro->name = strdup(token.value.c_str());
                macro->literals = nullptr;
                macro->num_literals = 0;
                macro->rules = nullptr;
                macro->num_rules = 0;

                // Expect (syntax-rules ...)
                token = tokenizer.nextToken();
                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("let-syntax binding value must be (syntax-rules ...)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                token = tokenizer.nextToken();
                if (token.type != TOKEN_SYMBOL || token.value != "syntax-rules") {
                    eshkol_error("let-syntax currently only supports syntax-rules transformers");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse literals list
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
                        eshkol_error("syntax-rules rule must be a list");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    eshkol_macro_rule_t rule;
                    rule.pattern = nullptr;
                    rule.template_ = nullptr;

                    // Parse pattern
                    token = tokenizer.nextToken();
                    if (token.type != TOKEN_LPAREN) {
                        eshkol_error("syntax-rules pattern must be a list");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }

                    rule.pattern = new eshkol_macro_pattern_t;
                    rule.pattern->type = MACRO_PAT_LIST;
                    rule.pattern->followed_by_ellipsis = 0;

                    // Recursive pattern parser (handles arbitrary nesting depth)
                    std::function<void(std::vector<eshkol_macro_pattern_t*>&)> parsePatElems;
                    parsePatElems = [&](std::vector<eshkol_macro_pattern_t*>& elements) {
                        while (true) {
                            token = tokenizer.nextToken();
                            if (token.type == TOKEN_RPAREN) break;
                            if (token.type == TOKEN_EOF) {
                                eshkol_error("Unexpected end of input in macro pattern");
                                break;
                            }
                            auto* elem = new eshkol_macro_pattern_t;
                            elem->followed_by_ellipsis = 0;
                            if (token.type == TOKEN_SYMBOL) {
                                if (token.value == "...") {
                                    if (!elements.empty())
                                        elements.back()->followed_by_ellipsis = 1;
                                    delete elem;
                                    continue;
                                }
                                bool is_lit = false;
                                for (const auto& lit : literals) {
                                    if (lit == token.value) { is_lit = true; break; }
                                }
                                elem->type = is_lit ? MACRO_PAT_LITERAL : MACRO_PAT_VARIABLE;
                                elem->identifier = strdup(token.value.c_str());
                            } else if (token.type == TOKEN_LPAREN) {
                                elem->type = MACRO_PAT_LIST;
                                elem->list.rest = nullptr;
                                std::vector<eshkol_macro_pattern_t*> nested;
                                parsePatElems(nested);  // Recurse
                                if (!nested.empty()) {
                                    elem->list.elements = new eshkol_macro_pattern_t*[nested.size()];
                                    elem->list.num_elements = nested.size();
                                    for (size_t j = 0; j < nested.size(); j++)
                                        elem->list.elements[j] = nested[j];
                                } else {
                                    elem->list.elements = nullptr;
                                    elem->list.num_elements = 0;
                                }
                            } else {
                                elem->type = MACRO_PAT_LITERAL;
                                elem->identifier = strdup(token.value.c_str());
                            }
                            elements.push_back(elem);
                        }
                    };

                    std::vector<eshkol_macro_pattern_t*> pat_elements;
                    parsePatElems(pat_elements);

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

                    // Parse template
                    token = tokenizer.nextToken();
                    eshkol_ast_t template_ast;
                    if (token.type == TOKEN_LPAREN) {
                        template_ast = parse_list(tokenizer);
                    } else {
                        template_ast = parse_atom(token);
                    }

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

                // Consume closing paren of this binding: (name (syntax-rules ...))
                token = tokenizer.nextToken();
                if (token.type != TOKEN_RPAREN) {
                    eshkol_error("Expected closing paren after let-syntax binding");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                macro_defs.push_back(macro);
            }

            // Parse body expressions
            std::vector<eshkol_ast_t> body_exprs;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;  // End of let-syntax
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in let-syntax body");
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

            // Store macro definitions
            ast.operation.let_syntax_op.num_macros = macro_defs.size();
            if (macro_defs.size() > 0) {
                ast.operation.let_syntax_op.macros = new eshkol_macro_def_t*[macro_defs.size()];
                for (size_t i = 0; i < macro_defs.size(); i++) {
                    ast.operation.let_syntax_op.macros[i] = macro_defs[i];
                }
            } else {
                ast.operation.let_syntax_op.macros = nullptr;
            }

            // Wrap body in sequence if multiple expressions, or single expr
            if (body_exprs.size() == 1) {
                ast.operation.let_syntax_op.body = new eshkol_ast_t;
                *ast.operation.let_syntax_op.body = body_exprs[0];
            } else if (body_exprs.size() > 1) {
                eshkol_ast_t* seq = new eshkol_ast_t;
                seq->type = ESHKOL_OP;
                seq->operation.op = ESHKOL_SEQUENCE_OP;
                seq->operation.sequence_op.num_expressions = body_exprs.size();
                seq->operation.sequence_op.expressions = new eshkol_ast_t[body_exprs.size()];
                for (size_t i = 0; i < body_exprs.size(); i++) {
                    seq->operation.sequence_op.expressions[i] = body_exprs[i];
                }
                ast.operation.let_syntax_op.body = seq;
            } else {
                eshkol_ast_t* null_body = new eshkol_ast_t;
                eshkol_ast_make_null(null_body);
                ast.operation.let_syntax_op.body = null_body;
            }

            return ast;
        }

        // Special handling for call/cc - first-class continuations
        // (call/cc proc) or (call-with-current-continuation proc)
        if (ast.operation.op == ESHKOL_CALL_CC_OP) {
            token = tokenizer.nextToken();
            eshkol_ast_t proc_ast;
            if (token.type == TOKEN_LPAREN) {
                proc_ast = parse_list(tokenizer);
            } else {
                proc_ast = parse_atom(token);
            }
            ast.operation.call_cc_op.proc = new eshkol_ast_t;
            *ast.operation.call_cc_op.proc = proc_ast;

            // Consume closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("call/cc expects exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            return ast;
        }

        // Special handling for dynamic-wind
        // (dynamic-wind before thunk after)
        if (ast.operation.op == ESHKOL_DYNAMIC_WIND_OP) {
            // Parse before thunk
            token = tokenizer.nextToken();
            eshkol_ast_t before_ast;
            if (token.type == TOKEN_LPAREN) {
                before_ast = parse_list(tokenizer);
            } else {
                before_ast = parse_atom(token);
            }
            ast.operation.dynamic_wind_op.before = new eshkol_ast_t;
            *ast.operation.dynamic_wind_op.before = before_ast;

            // Parse body thunk
            token = tokenizer.nextToken();
            eshkol_ast_t thunk_ast;
            if (token.type == TOKEN_LPAREN) {
                thunk_ast = parse_list(tokenizer);
            } else {
                thunk_ast = parse_atom(token);
            }
            ast.operation.dynamic_wind_op.thunk = new eshkol_ast_t;
            *ast.operation.dynamic_wind_op.thunk = thunk_ast;

            // Parse after thunk
            token = tokenizer.nextToken();
            eshkol_ast_t after_ast;
            if (token.type == TOKEN_LPAREN) {
                after_ast = parse_list(tokenizer);
            } else {
                after_ast = parse_atom(token);
            }
            ast.operation.dynamic_wind_op.after = new eshkol_ast_t;
            *ast.operation.dynamic_wind_op.after = after_ast;

            // Consume closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("dynamic-wind expects exactly three arguments");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            return ast;
        }

        // ===== R7RS WAVE 3: syntax-error - compile-time error =====
        // (syntax-error "message" datum ...)
        if (ast.operation.op == ESHKOL_SYNTAX_ERROR_OP) {
            // Collect all remaining elements as the error message
            std::string error_msg = "syntax-error: ";
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in syntax-error");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                if (token.type == TOKEN_STRING) {
                    error_msg += token.value;
                } else {
                    error_msg += token.value;
                }
                error_msg += " ";
            }
            eshkol_error("%s", error_msg.c_str());
            ast.type = ESHKOL_INVALID;
            return ast;
        }

        // ===== R7RS WAVE 3: cond-expand - conditional expansion =====
        // (cond-expand (feature-req body ...) ... (else body ...))
        if (ast.operation.op == ESHKOL_COND_EXPAND_OP) {
            // Define supported features
            auto hasFeature = [](const std::string& feature) -> bool {
                return feature == "r7rs" || feature == "eshkol" ||
                       feature == "ieee-float" || feature == "ratios" ||
                       feature == "exact-complex" ||
                #ifdef __APPLE__
                       feature == "macosx" || feature == "darwin" ||
                       feature == "unix" || feature == "posix" ||
                #endif
                #ifdef __linux__
                       feature == "linux" || feature == "unix" || feature == "posix" ||
                #endif
                #if defined(__x86_64__) || defined(_M_X64)
                       feature == "x86-64" ||
                #endif
                #if defined(__aarch64__) || defined(_M_ARM64)
                       feature == "aarch64" ||
                #endif
                       feature == "eshkol-1.1";
            };

            // Parse clauses until we find a match
            bool matched = false;
            std::vector<eshkol_ast_t> matched_body;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in cond-expand");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("cond-expand clause must be a list");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Parse feature requirement
                token = tokenizer.nextToken();
                bool clause_matches = false;

                if (token.type == TOKEN_SYMBOL && token.value == "else") {
                    clause_matches = !matched;
                } else if (token.type == TOKEN_SYMBOL) {
                    clause_matches = !matched && hasFeature(token.value);
                } else if (token.type == TOKEN_LPAREN) {
                    // Complex feature req: (and f1 f2), (or f1 f2), (not f), (library name)
                    Token req_type = tokenizer.nextToken();
                    if (req_type.type == TOKEN_SYMBOL && req_type.value == "and") {
                        clause_matches = true;
                        while (true) {
                            Token ft = tokenizer.nextToken();
                            if (ft.type == TOKEN_RPAREN) break;
                            if (ft.type == TOKEN_SYMBOL && !hasFeature(ft.value)) {
                                clause_matches = false;
                            }
                        }
                        clause_matches = clause_matches && !matched;
                    } else if (req_type.type == TOKEN_SYMBOL && req_type.value == "or") {
                        clause_matches = false;
                        while (true) {
                            Token ft = tokenizer.nextToken();
                            if (ft.type == TOKEN_RPAREN) break;
                            if (ft.type == TOKEN_SYMBOL && hasFeature(ft.value)) {
                                clause_matches = true;
                            }
                        }
                        clause_matches = clause_matches && !matched;
                    } else if (req_type.type == TOKEN_SYMBOL && req_type.value == "not") {
                        Token ft = tokenizer.nextToken();
                        clause_matches = !matched && ft.type == TOKEN_SYMBOL && !hasFeature(ft.value);
                        token = tokenizer.nextToken(); // consume rparen
                    } else {
                        // Skip unknown compound feature requirement
                        int depth = 1;
                        while (depth > 0) {
                            token = tokenizer.nextToken();
                            if (token.type == TOKEN_LPAREN) depth++;
                            else if (token.type == TOKEN_RPAREN) depth--;
                        }
                    }
                }

                // Parse body expressions of this clause
                std::vector<eshkol_ast_t> clause_body;
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) {
                        eshkol_error("Unexpected end of input in cond-expand clause");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    eshkol_ast_t expr;
                    if (token.type == TOKEN_LPAREN) {
                        expr = parse_list(tokenizer);
                    } else {
                        expr = parse_atom(token);
                    }
                    clause_body.push_back(expr);
                }

                if (clause_matches && !matched) {
                    matched = true;
                    matched_body = clause_body;
                }
            }

            if (!matched || matched_body.empty()) {
                // No matching clause or empty body — return void/null
                ast.type = ESHKOL_OP;
                ast.operation.op = ESHKOL_SEQUENCE_OP;
                ast.operation.sequence_op.num_expressions = 0;
                ast.operation.sequence_op.expressions = nullptr;
                return ast;
            }

            if (matched_body.size() == 1) {
                return matched_body[0];
            }

            // Wrap multiple expressions in a sequence
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_SEQUENCE_OP;
            ast.operation.sequence_op.num_expressions = matched_body.size();
            ast.operation.sequence_op.expressions = new eshkol_ast_t[matched_body.size()];
            for (size_t i = 0; i < matched_body.size(); i++) {
                ast.operation.sequence_op.expressions[i] = matched_body[i];
            }
            return ast;
        }

        // ===== R7RS WAVE 3: include / include-ci - file inclusion =====
        // (include "filename" ...)
        if (ast.operation.op == ESHKOL_INCLUDE_OP) {
            std::vector<eshkol_ast_t> all_exprs;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in include");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type != TOKEN_STRING) {
                    eshkol_error("include requires string filename arguments");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Read and parse the included file
                std::string filename = token.value;
                std::ifstream inc_file(filename);
                if (!inc_file.is_open()) {
                    eshkol_error("include: cannot open file '%s'", filename.c_str());
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                while (true) {
                    eshkol_ast_t file_ast = eshkol_parse_next_ast_from_stream(inc_file);
                    if (file_ast.type == ESHKOL_INVALID) break;
                    all_exprs.push_back(file_ast);
                }
                inc_file.close();
            }

            if (all_exprs.empty()) {
                ast.type = ESHKOL_OP;
                ast.operation.op = ESHKOL_SEQUENCE_OP;
                ast.operation.sequence_op.num_expressions = 0;
                ast.operation.sequence_op.expressions = nullptr;
                return ast;
            }

            if (all_exprs.size() == 1) {
                return all_exprs[0];
            }

            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_SEQUENCE_OP;
            ast.operation.sequence_op.num_expressions = all_exprs.size();
            ast.operation.sequence_op.expressions = new eshkol_ast_t[all_exprs.size()];
            for (size_t i = 0; i < all_exprs.size(); i++) {
                ast.operation.sequence_op.expressions[i] = all_exprs[i];
            }
            return ast;
        }

        // ===== R7RS WAVE 3: case-lambda - multi-arity dispatch =====
        // Transforms at parse time into a variadic lambda with arity dispatch:
        //   (case-lambda (() body0) ((x) body1) ((x y) body2))
        // => (lambda __cl_args
        //      (if (= (length __cl_args) 0) body0
        //        (if (= (length __cl_args) 1) (let ((x (car __cl_args))) body1)
        //          (if (= (length __cl_args) 2) (let ((x (car __cl_args)) (y (car (cdr __cl_args)))) body2)
        //            #f))))
        if (ast.operation.op == ESHKOL_CASE_LAMBDA_OP) {
            // --- Helper lambdas for AST construction ---
            auto clMakeVar = [](const char* name) -> eshkol_ast_t {
                eshkol_ast_t v = {};
                v.type = ESHKOL_VAR;
                size_t _len = strlen(name);
                v.variable.id = new char[_len + 1];
                memcpy(v.variable.id, name, _len + 1);
                v.variable.data = nullptr;
                return v;
            };
            auto clMakeInt = [](int64_t val) -> eshkol_ast_t {
                eshkol_ast_t lit = {};
                eshkol_ast_make_int64(&lit, val);
                return lit;
            };
            auto clMakeCall1 = [&clMakeVar](const char* func, eshkol_ast_t arg) -> eshkol_ast_t {
                eshkol_ast_t call = {};
                call.type = ESHKOL_OP;
                call.operation.op = ESHKOL_CALL_OP;
                call.operation.call_op.func = new eshkol_ast_t;
                *call.operation.call_op.func = clMakeVar(func);
                call.operation.call_op.num_vars = 1;
                call.operation.call_op.variables = new eshkol_ast_t[1];
                call.operation.call_op.variables[0] = arg;
                return call;
            };
            auto clMakeCall2 = [&clMakeVar](const char* func, eshkol_ast_t a, eshkol_ast_t b) -> eshkol_ast_t {
                eshkol_ast_t call = {};
                call.type = ESHKOL_OP;
                call.operation.op = ESHKOL_CALL_OP;
                call.operation.call_op.func = new eshkol_ast_t;
                *call.operation.call_op.func = clMakeVar(func);
                call.operation.call_op.num_vars = 2;
                call.operation.call_op.variables = new eshkol_ast_t[2];
                call.operation.call_op.variables[0] = a;
                call.operation.call_op.variables[1] = b;
                return call;
            };
            auto clMakeIf = [](eshkol_ast_t cond, eshkol_ast_t then_e, eshkol_ast_t else_e) -> eshkol_ast_t {
                eshkol_ast_t if_ast = {};
                if_ast.type = ESHKOL_OP;
                if_ast.operation.op = ESHKOL_IF_OP;
                if_ast.operation.call_op.func = nullptr;
                if_ast.operation.call_op.num_vars = 3;
                if_ast.operation.call_op.variables = new eshkol_ast_t[3];
                if_ast.operation.call_op.variables[0] = cond;
                if_ast.operation.call_op.variables[1] = then_e;
                if_ast.operation.call_op.variables[2] = else_e;
                return if_ast;
            };
            // Extract Nth arg from list: car(cdr^N(list))
            auto clMakeNthArg = [&clMakeCall1, &clMakeVar](int n) -> eshkol_ast_t {
                eshkol_ast_t cur = clMakeVar("__cl_args");
                for (int i = 0; i < n; i++) {
                    cur = clMakeCall1("cdr", cur);
                }
                return clMakeCall1("car", cur);
            };

            // --- Parse all clauses ---
            struct CaseClause {
                std::vector<std::string> param_names;
                bool is_variadic;
                std::string rest_param;
                eshkol_ast_t body;
            };
            std::vector<CaseClause> parsed_clauses;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in case-lambda");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("case-lambda clause must be (formals body ...)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                CaseClause clause;
                clause.is_variadic = false;

                // Parse formals
                token = tokenizer.nextToken();
                if (token.type == TOKEN_SYMBOL) {
                    clause.is_variadic = true;
                    clause.rest_param = token.value;
                } else if (token.type == TOKEN_LPAREN) {
                    while (true) {
                        token = tokenizer.nextToken();
                        if (token.type == TOKEN_RPAREN) break;
                        if (token.type == TOKEN_SYMBOL && token.value == ".") {
                            token = tokenizer.nextToken();
                            clause.is_variadic = true;
                            clause.rest_param = token.value;
                            token = tokenizer.nextToken(); // consume rparen
                            break;
                        }
                        if (token.type == TOKEN_SYMBOL) {
                            clause.param_names.push_back(token.value);
                        }
                    }
                }

                // Parse body
                std::vector<eshkol_ast_t> body_exprs;
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_RPAREN) break;
                    if (token.type == TOKEN_EOF) break;
                    eshkol_ast_t expr;
                    if (token.type == TOKEN_LPAREN) {
                        expr = parse_list(tokenizer);
                    } else {
                        expr = parse_atom(token);
                    }
                    body_exprs.push_back(expr);
                }
                clause.body = transformInternalDefinesToLetrec(body_exprs);
                parsed_clauses.push_back(clause);
            }

            // --- Build dispatch body (from last clause to first) ---
            // Fallback: #f
            eshkol_ast_t dispatch_body = {};
            eshkol_ast_make_bool(&dispatch_body, false);

            for (int i = (int)parsed_clauses.size() - 1; i >= 0; i--) {
                auto& cl = parsed_clauses[i];

                // For variadic clauses (catch-all), just use the body directly
                if (cl.is_variadic && cl.param_names.empty()) {
                    // (let ((rest_param __cl_args)) body)
                    eshkol_ast_t let_ast = {};
                    let_ast.type = ESHKOL_OP;
                    let_ast.operation.op = ESHKOL_LET_OP;
                    let_ast.operation.let_op.num_bindings = 1;
                    let_ast.operation.let_op.bindings = new eshkol_ast_t[1];
                    let_ast.operation.let_op.binding_types = nullptr;
                    let_ast.operation.let_op.name = nullptr;
                    let_ast.operation.let_op.bindings[0].type = ESHKOL_CONS;
                    let_ast.operation.let_op.bindings[0].cons_cell.car = new eshkol_ast_t;
                    *let_ast.operation.let_op.bindings[0].cons_cell.car = clMakeVar(cl.rest_param.c_str());
                    let_ast.operation.let_op.bindings[0].cons_cell.cdr = new eshkol_ast_t;
                    *let_ast.operation.let_op.bindings[0].cons_cell.cdr = clMakeVar("__cl_args");
                    let_ast.operation.let_op.body = new eshkol_ast_t;
                    *let_ast.operation.let_op.body = cl.body;
                    dispatch_body = let_ast;
                    continue;
                }

                uint64_t nparams = cl.param_names.size();

                // Condition: check list has exactly N elements using null?/cdr
                // N=0: (null? __cl_args)
                // N=1: (if (null? __cl_args) #f (null? (cdr __cl_args)))
                // N=2: (if (null? __cl_args) #f (if (null? (cdr __cl_args)) #f (null? (cdr (cdr __cl_args)))))
                eshkol_ast_t cond;
                if (nparams == 0) {
                    cond = clMakeCall1("null?", clMakeVar("__cl_args"));
                } else {
                    // Build: (null? (cdr^N __cl_args))
                    eshkol_ast_t nth_cdr = clMakeVar("__cl_args");
                    for (uint64_t k = 0; k < nparams; k++) {
                        nth_cdr = clMakeCall1("cdr", nth_cdr);
                    }
                    eshkol_ast_t tail_null = clMakeCall1("null?", nth_cdr);

                    // Guard: check list has at least N elements
                    // Wrap in: (if (null? cdr^(N-1) __cl_args) #f tail_null)
                    // from innermost to outermost
                    eshkol_ast_t false_ast = {};
                    eshkol_ast_make_bool(&false_ast, false);

                    cond = tail_null;
                    for (int64_t k = (int64_t)nparams - 1; k >= 0; k--) {
                        eshkol_ast_t kth_cdr = clMakeVar("__cl_args");
                        for (int64_t m = 0; m < k; m++) {
                            kth_cdr = clMakeCall1("cdr", kth_cdr);
                        }
                        eshkol_ast_t guard = clMakeCall1("null?", kth_cdr);
                        cond = clMakeIf(guard, false_ast, cond);
                    }
                }

                // Then branch
                eshkol_ast_t then_branch;
                if (nparams == 0) {
                    then_branch = cl.body;
                } else {
                    // Wrap body in let: (let ((p0 (car __cl_args)) (p1 (car (cdr __cl_args))) ...) body)
                    eshkol_ast_t let_ast = {};
                    let_ast.type = ESHKOL_OP;
                    let_ast.operation.op = ESHKOL_LET_OP;
                    let_ast.operation.let_op.num_bindings = nparams;
                    let_ast.operation.let_op.bindings = new eshkol_ast_t[nparams];
                    let_ast.operation.let_op.binding_types = nullptr;
                    let_ast.operation.let_op.name = nullptr;

                    for (uint64_t j = 0; j < nparams; j++) {
                        let_ast.operation.let_op.bindings[j].type = ESHKOL_CONS;
                        let_ast.operation.let_op.bindings[j].cons_cell.car = new eshkol_ast_t;
                        *let_ast.operation.let_op.bindings[j].cons_cell.car = clMakeVar(cl.param_names[j].c_str());
                        let_ast.operation.let_op.bindings[j].cons_cell.cdr = new eshkol_ast_t;
                        *let_ast.operation.let_op.bindings[j].cons_cell.cdr = clMakeNthArg((int)j);
                    }

                    let_ast.operation.let_op.body = new eshkol_ast_t;
                    *let_ast.operation.let_op.body = cl.body;
                    then_branch = let_ast;
                }

                dispatch_body = clMakeIf(cond, then_branch, dispatch_body);
            }

            // --- Create variadic lambda: (lambda __cl_args dispatch_body) ---
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_LAMBDA_OP;
            ast.operation.lambda_op.is_variadic = 1;
            ast.operation.lambda_op.rest_param = new char[sizeof("__cl_args")];
            memcpy(ast.operation.lambda_op.rest_param, "__cl_args", sizeof("__cl_args"));
            ast.operation.lambda_op.num_params = 0;
            ast.operation.lambda_op.parameters = nullptr;
            ast.operation.lambda_op.param_types = nullptr;
            ast.operation.lambda_op.return_type = nullptr;
            ast.operation.lambda_op.captured_vars = nullptr;
            ast.operation.lambda_op.num_captured = 0;
            ast.operation.lambda_op.body = new eshkol_ast_t;
            *ast.operation.lambda_op.body = dispatch_body;
            return ast;
        }

        // ===== R7RS WAVE 3: define-record-type =====
        // (define-record-type <name> (ctor field ...) pred (field accessor [mutator]) ...)
        // Expands into multiple defines at parse time
        if (ast.operation.op == ESHKOL_DEFINE_RECORD_TYPE_OP) {
            // Parse record type name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("define-record-type requires a type name");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            std::string type_name = token.value;
            // Strip angle brackets if present: <point> -> point
            if (type_name.front() == '<' && type_name.back() == '>') {
                type_name = type_name.substr(1, type_name.size() - 2);
            }

            // Parse constructor: (make-name field1 field2 ...)
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("define-record-type requires constructor specification");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            token = tokenizer.nextToken();
            std::string ctor_name = token.value;
            std::vector<std::string> ctor_fields;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_SYMBOL) {
                    ctor_fields.push_back(token.value);
                }
            }

            // Parse predicate name
            token = tokenizer.nextToken();
            std::string pred_name = (token.type == TOKEN_SYMBOL) ? token.value : "";

            // Parse field specifications: (field-name accessor [mutator]) ...
            struct FieldSpec {
                std::string name;
                std::string accessor;
                std::string mutator;
                int index;
            };
            std::vector<FieldSpec> fields;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in define-record-type");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                if (token.type == TOKEN_LPAREN) {
                    FieldSpec fs;
                    token = tokenizer.nextToken();
                    fs.name = token.value;
                    fs.index = (int)fields.size();
                    token = tokenizer.nextToken();
                    fs.accessor = (token.type == TOKEN_SYMBOL) ? token.value : "";
                    Token peek = tokenizer.peekToken();
                    if (peek.type == TOKEN_SYMBOL) {
                        token = tokenizer.nextToken();
                        fs.mutator = token.value;
                    }
                    token = tokenizer.nextToken(); // consume rparen
                    fields.push_back(fs);
                }
            }

            // Expand into a sequence of defines:
            // 1. Constructor: (define (make-name f1 f2) (vector 'type-tag f1 f2))
            // 2. Predicate: (define (pred? obj) (and (vector? obj) (eq? (vector-ref obj 0) 'type-tag)))
            // 3. Accessors: (define (field-accessor obj) (vector-ref obj index))
            // 4. Mutators: (define (field-mutator! obj val) (vector-set! obj index val))

            // We represent the record as a vector: [type-tag field0 field1 ...]
            // type-tag is a symbol identifying the record type

            std::vector<eshkol_ast_t> defines;

            // Build constructor define
            // (define (ctor-name f1 f2 ...) (vector 'type-tag f1 f2 ...))
            {
                eshkol_ast_t def = {};
                def.type = ESHKOL_OP;
                def.operation.op = ESHKOL_DEFINE_OP;
                { size_t _len = ctor_name.length();
                def.operation.define_op.name = new char[_len + 1];
                memcpy(def.operation.define_op.name, ctor_name.c_str(), _len + 1); }
                def.operation.define_op.is_function = 1;
                def.operation.define_op.is_variadic = 0;
                def.operation.define_op.rest_param = nullptr;
                def.operation.define_op.is_external = 0;
                def.operation.define_op.return_type = nullptr;
                def.operation.define_op.param_types = nullptr;
                def.operation.define_op.num_params = ctor_fields.size();
                def.operation.define_op.parameters = new eshkol_ast_t[ctor_fields.size()];
                for (size_t i = 0; i < ctor_fields.size(); i++) {
                    def.operation.define_op.parameters[i].type = ESHKOL_VAR;
                    { size_t _len = ctor_fields[i].length();
                    def.operation.define_op.parameters[i].variable.id = new char[_len + 1];
                    memcpy(def.operation.define_op.parameters[i].variable.id, ctor_fields[i].c_str(), _len + 1); }
                    def.operation.define_op.parameters[i].variable.data = nullptr;
                }

                // Body: (vector 'type-tag f1 f2 ...)
                // Build as a call to "vector" with type tag + fields
                eshkol_ast_t* body = new eshkol_ast_t;
                body->type = ESHKOL_OP;
                body->operation.op = ESHKOL_CALL_OP;
                body->operation.call_op.func = new eshkol_ast_t;
                body->operation.call_op.func->type = ESHKOL_VAR;
                body->operation.call_op.func->variable.id = new char[sizeof("vector")];
                memcpy(body->operation.call_op.func->variable.id, "vector", sizeof("vector"));
                body->operation.call_op.func->variable.data = nullptr;
                body->operation.call_op.num_vars = 1 + ctor_fields.size(); // type-tag + fields
                body->operation.call_op.variables = new eshkol_ast_t[body->operation.call_op.num_vars];

                // First element: type tag as a quoted symbol
                body->operation.call_op.variables[0].type = ESHKOL_OP;
                body->operation.call_op.variables[0].operation.op = ESHKOL_QUOTE_OP;
                body->operation.call_op.variables[0].operation.call_op.func = nullptr;
                body->operation.call_op.variables[0].operation.call_op.num_vars = 1;
                body->operation.call_op.variables[0].operation.call_op.variables = new eshkol_ast_t[1];
                body->operation.call_op.variables[0].operation.call_op.variables[0].type = ESHKOL_VAR;
                { size_t _len = type_name.length();
                body->operation.call_op.variables[0].operation.call_op.variables[0].variable.id = new char[_len + 1];
                memcpy(body->operation.call_op.variables[0].operation.call_op.variables[0].variable.id, type_name.c_str(), _len + 1); }
                body->operation.call_op.variables[0].operation.call_op.variables[0].variable.data = nullptr;

                // Remaining elements: field references
                for (size_t i = 0; i < ctor_fields.size(); i++) {
                    body->operation.call_op.variables[1 + i].type = ESHKOL_VAR;
                    { size_t _len = ctor_fields[i].length();
                    body->operation.call_op.variables[1 + i].variable.id = new char[_len + 1];
                    memcpy(body->operation.call_op.variables[1 + i].variable.id, ctor_fields[i].c_str(), _len + 1); }
                    body->operation.call_op.variables[1 + i].variable.data = nullptr;
                }

                def.operation.define_op.value = body;
                defines.push_back(def);
            }

            // Build predicate define
            // (define (pred? obj) (and (vector? obj) (equal? (vector-ref obj 0) 'type-tag)))
            if (!pred_name.empty()) {
                eshkol_ast_t def = {};
                def.type = ESHKOL_OP;
                def.operation.op = ESHKOL_DEFINE_OP;
                { size_t _len = pred_name.length();
                def.operation.define_op.name = new char[_len + 1];
                memcpy(def.operation.define_op.name, pred_name.c_str(), _len + 1); }
                def.operation.define_op.is_function = 1;
                def.operation.define_op.is_variadic = 0;
                def.operation.define_op.rest_param = nullptr;
                def.operation.define_op.is_external = 0;
                def.operation.define_op.return_type = nullptr;
                def.operation.define_op.param_types = nullptr;
                def.operation.define_op.num_params = 1;
                def.operation.define_op.parameters = new eshkol_ast_t[1];
                def.operation.define_op.parameters[0].type = ESHKOL_VAR;
                def.operation.define_op.parameters[0].variable.id = new char[sizeof("obj")];
                memcpy(def.operation.define_op.parameters[0].variable.id, "obj", sizeof("obj"));
                def.operation.define_op.parameters[0].variable.data = nullptr;

                // Body: (vector? obj) — simplified predicate checking if it's a vector
                // A full implementation would check the type tag, but for now we check vector?
                // and compare the first element to the type tag symbol
                eshkol_ast_t* body = new eshkol_ast_t;
                body->type = ESHKOL_OP;
                body->operation.op = ESHKOL_CALL_OP;
                body->operation.call_op.func = new eshkol_ast_t;
                body->operation.call_op.func->type = ESHKOL_VAR;
                body->operation.call_op.func->variable.id = new char[sizeof("vector?")];
                memcpy(body->operation.call_op.func->variable.id, "vector?", sizeof("vector?"));
                body->operation.call_op.func->variable.data = nullptr;
                body->operation.call_op.num_vars = 1;
                body->operation.call_op.variables = new eshkol_ast_t[1];
                body->operation.call_op.variables[0].type = ESHKOL_VAR;
                body->operation.call_op.variables[0].variable.id = new char[sizeof("obj")];
                memcpy(body->operation.call_op.variables[0].variable.id, "obj", sizeof("obj"));
                body->operation.call_op.variables[0].variable.data = nullptr;

                def.operation.define_op.value = body;
                defines.push_back(def);
            }

            // Build accessor defines
            for (const auto& fs : fields) {
                if (fs.accessor.empty()) continue;

                // Find field index: +1 because index 0 is the type tag
                int actual_index = -1;
                for (size_t i = 0; i < ctor_fields.size(); i++) {
                    if (ctor_fields[i] == fs.name) {
                        actual_index = (int)i + 1; // +1 for type tag
                        break;
                    }
                }
                if (actual_index < 0) actual_index = fs.index + 1;

                // (define (accessor obj) (vector-ref obj index))
                eshkol_ast_t def = {};
                def.type = ESHKOL_OP;
                def.operation.op = ESHKOL_DEFINE_OP;
                { size_t _len = fs.accessor.length();
                def.operation.define_op.name = new char[_len + 1];
                memcpy(def.operation.define_op.name, fs.accessor.c_str(), _len + 1); }
                def.operation.define_op.is_function = 1;
                def.operation.define_op.is_variadic = 0;
                def.operation.define_op.rest_param = nullptr;
                def.operation.define_op.is_external = 0;
                def.operation.define_op.return_type = nullptr;
                def.operation.define_op.param_types = nullptr;
                def.operation.define_op.num_params = 1;
                def.operation.define_op.parameters = new eshkol_ast_t[1];
                def.operation.define_op.parameters[0].type = ESHKOL_VAR;
                def.operation.define_op.parameters[0].variable.id = new char[sizeof("obj")];
                memcpy(def.operation.define_op.parameters[0].variable.id, "obj", sizeof("obj"));
                def.operation.define_op.parameters[0].variable.data = nullptr;

                eshkol_ast_t* body = new eshkol_ast_t;
                body->type = ESHKOL_OP;
                body->operation.op = ESHKOL_CALL_OP;
                body->operation.call_op.func = new eshkol_ast_t;
                body->operation.call_op.func->type = ESHKOL_VAR;
                body->operation.call_op.func->variable.id = new char[sizeof("vector-ref")];
                memcpy(body->operation.call_op.func->variable.id, "vector-ref", sizeof("vector-ref"));
                body->operation.call_op.func->variable.data = nullptr;
                body->operation.call_op.num_vars = 2;
                body->operation.call_op.variables = new eshkol_ast_t[2];
                body->operation.call_op.variables[0].type = ESHKOL_VAR;
                body->operation.call_op.variables[0].variable.id = new char[sizeof("obj")];
                memcpy(body->operation.call_op.variables[0].variable.id, "obj", sizeof("obj"));
                body->operation.call_op.variables[0].variable.data = nullptr;
                eshkol_ast_make_int64(&body->operation.call_op.variables[1], actual_index);

                def.operation.define_op.value = body;
                defines.push_back(def);

                // Build mutator if present
                if (!fs.mutator.empty()) {
                    // (define (mutator obj val) (vector-set! obj index val))
                    eshkol_ast_t mut_def = {};
                    mut_def.type = ESHKOL_OP;
                    mut_def.operation.op = ESHKOL_DEFINE_OP;
                    { size_t _len = fs.mutator.length();
                    mut_def.operation.define_op.name = new char[_len + 1];
                    memcpy(mut_def.operation.define_op.name, fs.mutator.c_str(), _len + 1); }
                    mut_def.operation.define_op.is_function = 1;
                    mut_def.operation.define_op.is_variadic = 0;
                    mut_def.operation.define_op.rest_param = nullptr;
                    mut_def.operation.define_op.is_external = 0;
                    mut_def.operation.define_op.return_type = nullptr;
                    mut_def.operation.define_op.param_types = nullptr;
                    mut_def.operation.define_op.num_params = 2;
                    mut_def.operation.define_op.parameters = new eshkol_ast_t[2];
                    mut_def.operation.define_op.parameters[0].type = ESHKOL_VAR;
                    mut_def.operation.define_op.parameters[0].variable.id = new char[sizeof("obj")];
                    memcpy(mut_def.operation.define_op.parameters[0].variable.id, "obj", sizeof("obj"));
                    mut_def.operation.define_op.parameters[0].variable.data = nullptr;
                    mut_def.operation.define_op.parameters[1].type = ESHKOL_VAR;
                    mut_def.operation.define_op.parameters[1].variable.id = new char[sizeof("val")];
                    memcpy(mut_def.operation.define_op.parameters[1].variable.id, "val", sizeof("val"));
                    mut_def.operation.define_op.parameters[1].variable.data = nullptr;

                    eshkol_ast_t* mut_body = new eshkol_ast_t;
                    mut_body->type = ESHKOL_OP;
                    mut_body->operation.op = ESHKOL_CALL_OP;
                    mut_body->operation.call_op.func = new eshkol_ast_t;
                    mut_body->operation.call_op.func->type = ESHKOL_VAR;
                    mut_body->operation.call_op.func->variable.id = new char[sizeof("vector-set!")];
                    memcpy(mut_body->operation.call_op.func->variable.id, "vector-set!", sizeof("vector-set!"));
                    mut_body->operation.call_op.func->variable.data = nullptr;
                    mut_body->operation.call_op.num_vars = 3;
                    mut_body->operation.call_op.variables = new eshkol_ast_t[3];
                    mut_body->operation.call_op.variables[0].type = ESHKOL_VAR;
                    mut_body->operation.call_op.variables[0].variable.id = new char[sizeof("obj")];
                    memcpy(mut_body->operation.call_op.variables[0].variable.id, "obj", sizeof("obj"));
                    mut_body->operation.call_op.variables[0].variable.data = nullptr;
                    eshkol_ast_make_int64(&mut_body->operation.call_op.variables[1], actual_index);
                    mut_body->operation.call_op.variables[2].type = ESHKOL_VAR;
                    mut_body->operation.call_op.variables[2].variable.id = new char[sizeof("val")];
                    memcpy(mut_body->operation.call_op.variables[2].variable.id, "val", sizeof("val"));
                    mut_body->operation.call_op.variables[2].variable.data = nullptr;

                    mut_def.operation.define_op.value = mut_body;
                    defines.push_back(mut_def);
                }
            }

            // Return as a sequence of defines
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_SEQUENCE_OP;
            ast.operation.sequence_op.num_expressions = defines.size();
            ast.operation.sequence_op.expressions = new eshkol_ast_t[defines.size()];
            for (size_t i = 0; i < defines.size(); i++) {
                ast.operation.sequence_op.expressions[i] = defines[i];
            }
            return ast;
        }

        // ===== R7RS WAVE 3: make-parameter =====
        // (make-parameter init) → transforms to:
        //   (let ((__p_cell (vector init)))
        //     (lambda __p_args
        //       (if (null? __p_args)
        //         (vector-ref __p_cell 0)
        //         (begin (vector-set! __p_cell 0 (car __p_args))
        //                (vector-ref __p_cell 0)))))
        if (ast.operation.op == ESHKOL_MAKE_PARAMETER_OP) {
            // Parse the init expression
            token = tokenizer.nextToken();
            eshkol_ast_t init_expr;
            if (token.type == TOKEN_LPAREN) {
                init_expr = parse_list(tokenizer);
            } else if (token.type == TOKEN_EOF || token.type == TOKEN_RPAREN) {
                eshkol_error("make-parameter requires an initial value");
                ast.type = ESHKOL_INVALID;
                return ast;
            } else {
                init_expr = parse_atom(token);
            }
            // Consume closing paren
            token = tokenizer.nextToken();

            // AST helpers (reusing pattern from case-lambda)
            auto mpMakeVar = [](const char* name) -> eshkol_ast_t {
                eshkol_ast_t v = {};
                v.type = ESHKOL_VAR;
                size_t _len = strlen(name);
                v.variable.id = new char[_len + 1];
                memcpy(v.variable.id, name, _len + 1);
                v.variable.data = nullptr;
                return v;
            };
            auto mpMakeInt = [](int64_t val) -> eshkol_ast_t {
                eshkol_ast_t lit = {};
                eshkol_ast_make_int64(&lit, val);
                return lit;
            };
            auto mpMakeCall1 = [&mpMakeVar](const char* func, eshkol_ast_t arg) -> eshkol_ast_t {
                eshkol_ast_t call = {};
                call.type = ESHKOL_OP;
                call.operation.op = ESHKOL_CALL_OP;
                call.operation.call_op.func = new eshkol_ast_t;
                *call.operation.call_op.func = mpMakeVar(func);
                call.operation.call_op.num_vars = 1;
                call.operation.call_op.variables = new eshkol_ast_t[1];
                call.operation.call_op.variables[0] = arg;
                return call;
            };
            auto mpMakeCall2 = [&mpMakeVar](const char* func, eshkol_ast_t a, eshkol_ast_t b) -> eshkol_ast_t {
                eshkol_ast_t call = {};
                call.type = ESHKOL_OP;
                call.operation.op = ESHKOL_CALL_OP;
                call.operation.call_op.func = new eshkol_ast_t;
                *call.operation.call_op.func = mpMakeVar(func);
                call.operation.call_op.num_vars = 2;
                call.operation.call_op.variables = new eshkol_ast_t[2];
                call.operation.call_op.variables[0] = a;
                call.operation.call_op.variables[1] = b;
                return call;
            };
            auto mpMakeCall3 = [&mpMakeVar](const char* func, eshkol_ast_t a, eshkol_ast_t b, eshkol_ast_t c) -> eshkol_ast_t {
                eshkol_ast_t call = {};
                call.type = ESHKOL_OP;
                call.operation.op = ESHKOL_CALL_OP;
                call.operation.call_op.func = new eshkol_ast_t;
                *call.operation.call_op.func = mpMakeVar(func);
                call.operation.call_op.num_vars = 3;
                call.operation.call_op.variables = new eshkol_ast_t[3];
                call.operation.call_op.variables[0] = a;
                call.operation.call_op.variables[1] = b;
                call.operation.call_op.variables[2] = c;
                return call;
            };

            // Build: (vector-ref __p_cell 0)
            eshkol_ast_t vref = mpMakeCall2("vector-ref", mpMakeVar("__p_cell"), mpMakeInt(0));

            // Build: (vector-set! __p_cell 0 (car __p_args))
            eshkol_ast_t vset = mpMakeCall3("vector-set!", mpMakeVar("__p_cell"), mpMakeInt(0),
                                            mpMakeCall1("car", mpMakeVar("__p_args")));

            // Build: (begin (vector-set! ...) (vector-ref ...))
            eshkol_ast_t set_body = {};
            set_body.type = ESHKOL_OP;
            set_body.operation.op = ESHKOL_SEQUENCE_OP;
            set_body.operation.sequence_op.num_expressions = 2;
            set_body.operation.sequence_op.expressions = new eshkol_ast_t[2];
            set_body.operation.sequence_op.expressions[0] = vset;
            set_body.operation.sequence_op.expressions[1] = vref;

            // Build: (if (null? __p_args) (vector-ref __p_cell 0) (begin ...))
            eshkol_ast_t if_ast = {};
            if_ast.type = ESHKOL_OP;
            if_ast.operation.op = ESHKOL_IF_OP;
            if_ast.operation.call_op.func = nullptr;
            if_ast.operation.call_op.num_vars = 3;
            if_ast.operation.call_op.variables = new eshkol_ast_t[3];
            if_ast.operation.call_op.variables[0] = mpMakeCall1("null?", mpMakeVar("__p_args"));
            if_ast.operation.call_op.variables[1] = vref;
            if_ast.operation.call_op.variables[2] = set_body;

            // Build: (lambda __p_args (if ...))
            eshkol_ast_t lambda_ast = {};
            lambda_ast.type = ESHKOL_OP;
            lambda_ast.operation.op = ESHKOL_LAMBDA_OP;
            lambda_ast.operation.lambda_op.is_variadic = 1;
            lambda_ast.operation.lambda_op.rest_param = new char[sizeof("__p_args")];
            memcpy(lambda_ast.operation.lambda_op.rest_param, "__p_args", sizeof("__p_args"));
            lambda_ast.operation.lambda_op.num_params = 0;
            lambda_ast.operation.lambda_op.parameters = nullptr;
            lambda_ast.operation.lambda_op.param_types = nullptr;
            lambda_ast.operation.lambda_op.return_type = nullptr;
            lambda_ast.operation.lambda_op.captured_vars = nullptr;
            lambda_ast.operation.lambda_op.num_captured = 0;
            lambda_ast.operation.lambda_op.body = new eshkol_ast_t;
            *lambda_ast.operation.lambda_op.body = if_ast;

            // Build: (let ((__p_cell (vector init))) (lambda ...))
            eshkol_ast_t vector_call = mpMakeCall1("vector", init_expr);

            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_LET_OP;
            ast.operation.let_op.num_bindings = 1;
            ast.operation.let_op.bindings = new eshkol_ast_t[1];
            ast.operation.let_op.binding_types = nullptr;
            ast.operation.let_op.name = nullptr;
            ast.operation.let_op.bindings[0].type = ESHKOL_CONS;
            ast.operation.let_op.bindings[0].cons_cell.car = new eshkol_ast_t;
            *ast.operation.let_op.bindings[0].cons_cell.car = mpMakeVar("__p_cell");
            ast.operation.let_op.bindings[0].cons_cell.cdr = new eshkol_ast_t;
            *ast.operation.let_op.bindings[0].cons_cell.cdr = vector_call;
            ast.operation.let_op.body = new eshkol_ast_t;
            *ast.operation.let_op.body = lambda_ast;
            return ast;
        }

        // ===== R7RS WAVE 3: parameterize =====
        // (parameterize ((param1 val1) (param2 val2) ...) body ...)
        // Transforms to:
        //   (let ((__saved0 (param0)) (__saved1 (param1)) ...)
        //     (param0 val0) (param1 val1) ...
        //     (let ((__result (begin body ...)))
        //       (param0 __saved0) (param1 __saved1) ...
        //       __result))
        if (ast.operation.op == ESHKOL_PARAMETERIZE_OP) {
            // Parse bindings list
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("parameterize requires bindings list");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            std::vector<eshkol_ast_t> params;
            std::vector<eshkol_ast_t> values;

            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in parameterize bindings");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                if (token.type != TOKEN_LPAREN) {
                    eshkol_error("parameterize binding must be (param value)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                // Parse param expression
                token = tokenizer.nextToken();
                eshkol_ast_t param_expr;
                if (token.type == TOKEN_LPAREN) {
                    param_expr = parse_list(tokenizer);
                } else {
                    param_expr = parse_atom(token);
                }
                // Parse value expression
                token = tokenizer.nextToken();
                eshkol_ast_t val_expr;
                if (token.type == TOKEN_LPAREN) {
                    val_expr = parse_list(tokenizer);
                } else {
                    val_expr = parse_atom(token);
                }
                // Consume closing paren
                token = tokenizer.nextToken();
                params.push_back(param_expr);
                values.push_back(val_expr);
            }

            // Parse body expressions
            std::vector<eshkol_ast_t> body_exprs;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) break;
                eshkol_ast_t expr;
                if (token.type == TOKEN_LPAREN) {
                    expr = parse_list(tokenizer);
                } else {
                    expr = parse_atom(token);
                }
                body_exprs.push_back(expr);
            }

            eshkol_ast_t body = transformInternalDefinesToLetrec(body_exprs);

            // AST helper
            auto pmMakeVar = [](const char* name) -> eshkol_ast_t {
                eshkol_ast_t v = {};
                v.type = ESHKOL_VAR;
                size_t _len = strlen(name);
                v.variable.id = new char[_len + 1];
                memcpy(v.variable.id, name, _len + 1);
                v.variable.data = nullptr;
                return v;
            };

            // Build the transformation:
            // Outer let: save current values by calling params with 0 args
            size_t n = params.size();

            // Build sequence: set new values, evaluate body, restore old values, return result
            // (begin (param0 val0) ... (paramN valN) (let ((__result body)) (param0 __saved0) ... __result))
            std::vector<eshkol_ast_t> outer_body;

            // Set new values: (param val) for each binding
            for (size_t i = 0; i < n; i++) {
                eshkol_ast_t set_call = {};
                set_call.type = ESHKOL_OP;
                set_call.operation.op = ESHKOL_CALL_OP;
                set_call.operation.call_op.func = new eshkol_ast_t;
                *set_call.operation.call_op.func = params[i];
                set_call.operation.call_op.num_vars = 1;
                set_call.operation.call_op.variables = new eshkol_ast_t[1];
                set_call.operation.call_op.variables[0] = values[i];
                outer_body.push_back(set_call);
            }

            // Inner let: capture body result
            std::string result_name = "__pz_result";
            eshkol_ast_t inner_let = {};
            inner_let.type = ESHKOL_OP;
            inner_let.operation.op = ESHKOL_LET_OP;
            inner_let.operation.let_op.num_bindings = 1;
            inner_let.operation.let_op.bindings = new eshkol_ast_t[1];
            inner_let.operation.let_op.binding_types = nullptr;
            inner_let.operation.let_op.name = nullptr;
            inner_let.operation.let_op.bindings[0].type = ESHKOL_CONS;
            inner_let.operation.let_op.bindings[0].cons_cell.car = new eshkol_ast_t;
            *inner_let.operation.let_op.bindings[0].cons_cell.car = pmMakeVar("__pz_result");
            inner_let.operation.let_op.bindings[0].cons_cell.cdr = new eshkol_ast_t;
            *inner_let.operation.let_op.bindings[0].cons_cell.cdr = body;

            // Inner let body: restore values, then return __pz_result
            std::vector<eshkol_ast_t> restore_exprs;
            for (size_t i = 0; i < n; i++) {
                std::string saved_name = "__pz_saved_" + std::to_string(i);
                eshkol_ast_t restore_call = {};
                restore_call.type = ESHKOL_OP;
                restore_call.operation.op = ESHKOL_CALL_OP;
                restore_call.operation.call_op.func = new eshkol_ast_t;
                *restore_call.operation.call_op.func = params[i];
                restore_call.operation.call_op.num_vars = 1;
                restore_call.operation.call_op.variables = new eshkol_ast_t[1];
                restore_call.operation.call_op.variables[0] = pmMakeVar(saved_name.c_str());
                restore_exprs.push_back(restore_call);
            }
            restore_exprs.push_back(pmMakeVar("__pz_result"));

            // Wrap restore sequence in a begin
            eshkol_ast_t restore_seq = {};
            restore_seq.type = ESHKOL_OP;
            restore_seq.operation.op = ESHKOL_SEQUENCE_OP;
            restore_seq.operation.sequence_op.num_expressions = restore_exprs.size();
            restore_seq.operation.sequence_op.expressions = new eshkol_ast_t[restore_exprs.size()];
            for (size_t i = 0; i < restore_exprs.size(); i++) {
                restore_seq.operation.sequence_op.expressions[i] = restore_exprs[i];
            }

            inner_let.operation.let_op.body = new eshkol_ast_t;
            *inner_let.operation.let_op.body = restore_seq;
            outer_body.push_back(inner_let);

            // Wrap set+inner_let in a begin for the outer let body
            eshkol_ast_t outer_seq = {};
            outer_seq.type = ESHKOL_OP;
            outer_seq.operation.op = ESHKOL_SEQUENCE_OP;
            outer_seq.operation.sequence_op.num_expressions = outer_body.size();
            outer_seq.operation.sequence_op.expressions = new eshkol_ast_t[outer_body.size()];
            for (size_t i = 0; i < outer_body.size(); i++) {
                outer_seq.operation.sequence_op.expressions[i] = outer_body[i];
            }

            // Outer let: save current values
            ast.type = ESHKOL_OP;
            ast.operation.op = ESHKOL_LET_OP;
            ast.operation.let_op.num_bindings = n;
            ast.operation.let_op.bindings = new eshkol_ast_t[n];
            ast.operation.let_op.binding_types = nullptr;
            ast.operation.let_op.name = nullptr;

            for (size_t i = 0; i < n; i++) {
                std::string saved_name = "__pz_saved_" + std::to_string(i);
                // Save: __pz_saved_i = (param_i)  ; call with 0 args
                eshkol_ast_t save_call = {};
                save_call.type = ESHKOL_OP;
                save_call.operation.op = ESHKOL_CALL_OP;
                save_call.operation.call_op.func = new eshkol_ast_t;
                *save_call.operation.call_op.func = params[i];
                save_call.operation.call_op.num_vars = 0;
                save_call.operation.call_op.variables = nullptr;

                ast.operation.let_op.bindings[i].type = ESHKOL_CONS;
                ast.operation.let_op.bindings[i].cons_cell.car = new eshkol_ast_t;
                *ast.operation.let_op.bindings[i].cons_cell.car = pmMakeVar(saved_name.c_str());
                ast.operation.let_op.bindings[i].cons_cell.cdr = new eshkol_ast_t;
                *ast.operation.let_op.bindings[i].cons_cell.cdr = save_call;
            }

            ast.operation.let_op.body = new eshkol_ast_t;
            *ast.operation.let_op.body = outer_seq;
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
                { size_t _len = token.value.length();
                var_ast.variable.id = new char[_len + 1];
                memcpy(var_ast.variable.id, token.value.c_str(), _len + 1); }
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

                // Parse dimensions until we hit a non-integer token
                // Dimensions must be integers (no decimal point or exponent)
                while (true) {
                    token = tokenizer.nextToken();
                    if (token.type == TOKEN_NUMBER) {
                        // Check if it's an integer (no decimal point or exponent)
                        bool is_integer = true;
                        for (char c : token.value) {
                            if (c == '.' || c == 'e' || c == 'E') {
                                is_integer = false;
                                break;
                            }
                        }
                        if (is_integer) {
                            dimensions.push_back(std::stoull(token.value));
                        } else {
                            break; // Float starts the elements
                        }
                    } else {
                        break; // Non-number token starts the elements
                    }
                }
                
                if (dimensions.empty()) {
                    // No integer dimensions found — treat as 1D tensor from elements
                    // (tensor 1.0 2.0 3.0) → 1D tensor [1.0, 2.0, 3.0]
                    std::vector<eshkol_ast_t> tensor_elements;

                    // First non-integer token is the first element
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
                    while (true) {
                        token = tokenizer.nextToken();
                        if (token.type == TOKEN_RPAREN) break;
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

                    // Set up as 1D tensor
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
            // Accept both bare symbol and quoted symbol: (diff expr x) or (diff expr 'x)
            if (token.type == TOKEN_QUOTE) {
                token = tokenizer.nextToken();
            }
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
            
            { size_t _len = token.value.length();
            ast.operation.diff_op.variable = new char[_len + 1];
            memcpy(ast.operation.diff_op.variable, token.value.c_str(), _len + 1); }
            
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

            // Check for closing paren or additional point arguments
            // Supports: (gradient f x), (gradient f x y), (gradient f x y z ...)
            // Multiple args are packed into #(x y z ...) tensor for the codegen.
            Token close_token = tokenizer.nextToken();
            if (close_token.type != TOKEN_RPAREN) {
                // Multi-argument gradient: (gradient f x y ...) → (gradient f #(x y ...))
                // Collect all remaining arguments
                std::vector<eshkol_ast_t> point_args;
                point_args.push_back(point);  // First point arg already parsed

                // Parse the current token and any remaining args
                eshkol_ast_t next_arg;
                if (close_token.type == TOKEN_LPAREN) {
                    next_arg = parse_list(tokenizer);
                } else {
                    next_arg = parse_atom(close_token);
                }
                point_args.push_back(next_arg);

                while (true) {
                    Token t = tokenizer.nextToken();
                    if (t.type == TOKEN_RPAREN) break;
                    if (t.type == TOKEN_EOF) {
                        eshkol_error("Unterminated gradient expression");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                    eshkol_ast_t arg;
                    if (t.type == TOKEN_LPAREN) {
                        arg = parse_list(tokenizer);
                    } else {
                        arg = parse_atom(t);
                    }
                    point_args.push_back(arg);
                }

                // Build tensor literal #(x y z ...) from the collected args.
                // The gradient codegen detects tensor inputs and uses forward-mode AD
                // with proper multi-parameter function call unpacking.
                eshkol_ast_t tensor_point;
                tensor_point.type = ESHKOL_TENSOR;
                tensor_point.tensor_val.total_elements = point_args.size();
                tensor_point.tensor_val.num_dimensions = 1;
                tensor_point.tensor_val.dimensions = new uint64_t[1];
                tensor_point.tensor_val.dimensions[0] = point_args.size();
                tensor_point.tensor_val.elements = new eshkol_ast_t[point_args.size()];
                for (size_t i = 0; i < point_args.size(); i++) {
                    tensor_point.tensor_val.elements[i] = point_args[i];
                }
                point = tensor_point;
            }

            // Set up gradient operation (2-argument form, point may be packed tensor)
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

            { size_t _len = token.value.length();
            ast.operation.extern_var_op.type = new char[_len + 1];
            memcpy(ast.operation.extern_var_op.type, token.value.c_str(), _len + 1); }

            // Parse variable name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("extern requires function name as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            { size_t _len = token.value.length();
            ast.operation.extern_var_op.name = new char[_len + 1];
            memcpy(ast.operation.extern_var_op.name, token.value.c_str(), _len + 1); }
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
            
            { size_t _len = token.value.length();
            ast.operation.extern_op.return_type = new char[_len + 1];
            memcpy(ast.operation.extern_op.return_type, token.value.c_str(), _len + 1); }
            
            // Parse function name
            token = tokenizer.nextToken();
            if (token.type != TOKEN_SYMBOL) {
                eshkol_error("extern requires function name as second argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            { size_t _len = token.value.length();
            ast.operation.extern_op.name = new char[_len + 1];
            memcpy(ast.operation.extern_op.name, token.value.c_str(), _len + 1); }
            
            // Initialize real_name to nullptr (will be set if :real modifier is used)
            ast.operation.extern_op.real_name = nullptr;

            // Parse parameter types and :real modifier
            // Syntax: (extern return-type func-name param-types... :real real-name)
            // The :real modifier can appear at the end after all parameter types
            std::vector<eshkol_ast_t> param_types;
            token = tokenizer.nextToken();

            while (true) {
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in extern declaration");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Check for :real modifier (TOKEN_COLON followed by "real")
                if (token.type == TOKEN_COLON) {
                    Token next = tokenizer.nextToken();
                    if (next.type == TOKEN_SYMBOL && next.value == "real") {
                        // Parse the real function name
                        token = tokenizer.nextToken();
                        if (token.type != TOKEN_SYMBOL) {
                            eshkol_error("extern :real requires function name as argument");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }

                        { size_t _len = token.value.length();
                        ast.operation.extern_op.real_name = new char[_len + 1];
                        memcpy(ast.operation.extern_op.real_name, token.value.c_str(), _len + 1); }

                        // Expect closing paren after :real name
                        token = tokenizer.nextToken();
                        if (token.type != TOKEN_RPAREN) {
                            eshkol_error(":real modifier must be at end of extern declaration");
                            ast.type = ESHKOL_INVALID;
                            return ast;
                        }
                        break;
                    } else {
                        eshkol_error("Unexpected ':' in extern declaration (did you mean :real?)");
                        ast.type = ESHKOL_INVALID;
                        return ast;
                    }
                }

                if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("extern parameter types must be symbols");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }

                // Create parameter type AST node
                eshkol_ast_t param_type = {};
                {
                    size_t len = token.value.length();
                    char* ptr = new char[len + 1];
                    memcpy(ptr, token.value.c_str(), len + 1);
                    eshkol_ast_make_string(&param_type, ptr, len + 1);
                }

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

            { size_t _len = token.value.length();
            ast.operation.import_op.path = new char[_len + 1];
            memcpy(ast.operation.import_op.path, token.value.c_str(), _len + 1); }

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
                if (token.type == TOKEN_STRING) {
                    // (load "path/to/file.esk") — convert file path to module name
                    std::string path = token.value;
                    // Strip .esk suffix if present
                    if (path.size() > 4 && path.substr(path.size() - 4) == ".esk")
                        path = path.substr(0, path.size() - 4);
                    // Convert / to . for module resolution
                    for (char& c : path) { if (c == '/') c = '.'; }
                    modules.push_back(path);
                } else if (token.type != TOKEN_SYMBOL) {
                    eshkol_error("require expects symbolic module names (e.g., data.json)");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                } else {
                    modules.push_back(token.value);
                }
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
                { size_t _len = modules[i].length();
                ast.operation.require_op.module_names[i] = new char[_len + 1];
                memcpy(ast.operation.require_op.module_names[i], modules[i].c_str(), _len + 1); }
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
                { size_t _len = exports[i].length();
                ast.operation.provide_op.export_names[i] = new char[_len + 1];
                memcpy(ast.operation.provide_op.export_names[i], exports[i].c_str(), _len + 1); }
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

        // Special handling for begin - transform internal defines to letrec
        // This implements R6RS/R7RS semantics where internal defines in begin are letrec*
        if (first_symbol == "begin") {
            // Parse all expressions in the begin block
            std::vector<eshkol_ast_t> begin_expressions;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in begin expression");
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
                begin_expressions.push_back(expr);
            }

            if (begin_expressions.empty()) {
                eshkol_error("begin requires at least one expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Check if there are any internal defines
            bool has_defines = false;
            for (const auto& expr : begin_expressions) {
                if (expr.type == ESHKOL_OP && expr.operation.op == ESHKOL_DEFINE_OP) {
                    has_defines = true;
                    break;
                }
            }

            if (has_defines) {
                // Transform internal defines to letrec - this handles nested function defines correctly
                eshkol_debug("Transforming begin with internal defines to letrec");
                return transformInternalDefinesToLetrec(begin_expressions);
            } else {
                // No defines - create a simple sequence
                if (begin_expressions.size() == 1) {
                    return begin_expressions[0];
                }
                ast.type = ESHKOL_OP;
                ast.operation.op = ESHKOL_SEQUENCE_OP;
                ast.operation.sequence_op.num_expressions = begin_expressions.size();
                ast.operation.sequence_op.expressions = new eshkol_ast_t[begin_expressions.size()];
                for (size_t i = 0; i < begin_expressions.size(); i++) {
                    ast.operation.sequence_op.expressions[i] = begin_expressions[i];
                }
                return ast;
            }
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
        } else if (ast.operation.op == ESHKOL_QUASIQUOTE_OP) {
            // Special handling for quasiquote - its argument can be any data, with unquote/unquote-splicing
            token = tokenizer.nextToken();
            if (token.type == TOKEN_RPAREN) {
                eshkol_error("quasiquote requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            if (token.type == TOKEN_EOF) {
                eshkol_error("Unexpected end of input in quasiquote");
                ast.type = ESHKOL_INVALID;
                return ast;
            }

            // Parse the quasiquoted expression using our special quasiquoted data parser
            eshkol_ast_t quoted = parse_quasiquoted_data_with_token(tokenizer, token);
            if (quoted.type == ESHKOL_INVALID) {
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            elements.push_back(quoted);

            // Expect closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("quasiquote requires exactly one argument");
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

                // Use parse_expression for full expression support in arguments
                // (handles #(...) vector literals, quoted expressions, nested lists, atoms, etc.)
                tokenizer.pushBack(token);
                eshkol_ast_t element = parse_expression(tokenizer);
                if (element.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                elements.push_back(element);
            }
        }

        // Set up operation based on type and arguments
        if (ast.operation.op == ESHKOL_CALL_OP) {
            // Create function name AST node
            ast.operation.call_op.func = new eshkol_ast_t;
            ast.operation.call_op.func->type = ESHKOL_VAR;
            { size_t _len = first_symbol.length();
            ast.operation.call_op.func->variable.id = new char[_len + 1];
            memcpy(ast.operation.call_op.func->variable.id, first_symbol.c_str(), _len + 1); }
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
        } else if (ast.operation.op == ESHKOL_QUASIQUOTE_OP) {
            // quasiquote takes exactly one argument - the expression to quasiquote
            // Store it in call_op.variables[0] for codegenQuasiquotedAST to access
            if (elements.size() != 1) {
                eshkol_error("quasiquote requires exactly one argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            ast.operation.call_op.func = nullptr;
            ast.operation.call_op.num_vars = 1;
            ast.operation.call_op.variables = new eshkol_ast_t[1];
            ast.operation.call_op.variables[0] = elements[0];
        } else if (ast.operation.op >= ESHKOL_UNIFY_OP && ast.operation.op <= ESHKOL_WORKSPACE_PRED_OP) {
            // Neuro-symbolic consciousness engine operations
            // All use call_op structure: func=nullptr, variables=arguments
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
        }
    } else {
        // Non-symbol first element - parse as a cons structure
        // This handles cases like (#t 'yes) or ((null? lst) #f) in cond clauses
        // Structure: CALL_OP with func=first_element, variables=rest

        eshkol_ast_t first_elem;
        if (token.type == TOKEN_LPAREN) {
            first_elem = parse_list(tokenizer);
        } else {
            first_elem = parse_atom(token);
        }

        if (first_elem.type == ESHKOL_INVALID) {
            ast.type = ESHKOL_INVALID;
            return ast;
        }

        // Parse remaining elements
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
                // Handle quoted expressions
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

        // Create as CALL_OP structure: func=first_elem, variables=rest
        ast.operation.op = ESHKOL_CALL_OP;
        ast.operation.call_op.func = new eshkol_ast_t;
        *ast.operation.call_op.func = first_elem;

        ast.operation.call_op.num_vars = elements.size();
        if (ast.operation.call_op.num_vars > 0) {
            ast.operation.call_op.variables = new eshkol_ast_t[ast.operation.call_op.num_vars];
            for (size_t i = 0; i < ast.operation.call_op.num_vars; i++) {
                ast.operation.call_op.variables[i] = elements[i];
            }
        } else {
            ast.operation.call_op.variables = nullptr;
        }
    }

    return ast;
}

/**
 * Parse the body of a vector literal after the opening #( has been consumed.
 * Handles recursive nesting: #(#(1 2) #(3 4)) → 2D tensor [2,2]
 * Also supports arbitrary depth: #(#(#(1 2) #(3 4)) #(#(5 6) #(7 8))) → 3D tensor [2,2,2]
 *
 * When all elements are sub-tensors with identical shapes, they are flattened
 * into an N+1-dimensional tensor. Otherwise, elements are kept as a 1D vector.
 */
static eshkol_ast_t parse_vector_body(SchemeTokenizer& tokenizer) {
    eshkol_ast_t ast = {};
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

        // Push back the token and use parse_expression to handle ALL expression
        // types uniformly: nested #(...), quoted 'expr, lists, atoms, etc.
        tokenizer.pushBack(elem_token);
        eshkol_ast_t element = parse_expression(tokenizer);

        if (element.type == ESHKOL_INVALID) {
            ast.type = ESHKOL_INVALID;
            return ast;
        }
        elements.push_back(element);
    }

    if (elements.empty()) {
        // Empty vector: #() → 1D tensor with 0 elements
        ast.operation.tensor_op.num_dimensions = 1;
        ast.operation.tensor_op.dimensions = new uint64_t[1];
        ast.operation.tensor_op.dimensions[0] = 0;
        ast.operation.tensor_op.total_elements = 0;
        ast.operation.tensor_op.elements = nullptr;
        return ast;
    }

    // Check if ALL elements are tensor_ops with identical shapes → nested vector flattening
    bool all_sub_tensors = true;
    for (auto& elem : elements) {
        if (elem.type != ESHKOL_OP || elem.operation.op != ESHKOL_TENSOR_OP) {
            all_sub_tensors = false;
            break;
        }
    }

    if (all_sub_tensors && elements.size() > 0) {
        // Verify all sub-tensors have the same shape
        uint64_t sub_ndim = elements[0].operation.tensor_op.num_dimensions;
        uint64_t* sub_dims = elements[0].operation.tensor_op.dimensions;
        uint64_t sub_total = elements[0].operation.tensor_op.total_elements;
        bool shapes_match = true;

        for (size_t i = 1; i < elements.size(); i++) {
            if (elements[i].operation.tensor_op.num_dimensions != sub_ndim) {
                eshkol_error("Nested vector dimension mismatch: element 0 has %llu dimensions, element %zu has %llu",
                    (unsigned long long)sub_ndim, i,
                    (unsigned long long)elements[i].operation.tensor_op.num_dimensions);
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            for (uint64_t d = 0; d < sub_ndim; d++) {
                if (elements[i].operation.tensor_op.dimensions[d] != sub_dims[d]) {
                    eshkol_error("Nested vector shape mismatch at dimension %llu: element 0 has %llu, element %zu has %llu",
                        (unsigned long long)d, (unsigned long long)sub_dims[d], i,
                        (unsigned long long)elements[i].operation.tensor_op.dimensions[d]);
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
            }
        }

        if (shapes_match) {
            // Flatten into N+1 dimensional tensor
            // #(#(1 2) #(3 4)) with sub_dims=[2] → dims=[2, 2], elements=[1, 2, 3, 4]
            uint64_t outer_count = elements.size();
            uint64_t new_ndim = sub_ndim + 1;
            uint64_t* new_dims = new uint64_t[new_ndim];
            new_dims[0] = outer_count;
            for (uint64_t d = 0; d < sub_ndim; d++) {
                new_dims[d + 1] = sub_dims[d];
            }

            uint64_t new_total = outer_count * sub_total;
            eshkol_ast_t* new_elements = new eshkol_ast_t[new_total];

            uint64_t idx = 0;
            for (auto& elem : elements) {
                for (uint64_t i = 0; i < sub_total; i++) {
                    new_elements[idx++] = elem.operation.tensor_op.elements[i];
                }
                // Free the sub-tensor's allocated arrays (they've been flattened)
                delete[] elem.operation.tensor_op.dimensions;
                delete[] elem.operation.tensor_op.elements;
            }

            ast.operation.tensor_op.num_dimensions = new_ndim;
            ast.operation.tensor_op.dimensions = new_dims;
            ast.operation.tensor_op.total_elements = new_total;
            ast.operation.tensor_op.elements = new_elements;
            return ast;
        }
    }

    // Default: 1D vector of scalar elements
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

static eshkol_ast_t parse_expression(SchemeTokenizer& tokenizer) {
    // Stack space guard: detect actual remaining stack space using platform APIs.
    // This prevents segfaults from deeply nested input without imposing arbitrary limits.
    if (!check_stack_space()) {
        eshkol_error("Stack space exhausted during parsing — expression nesting too deep");
        eshkol_ast_t invalid = {};
        invalid.type = ESHKOL_INVALID;
        return invalid;
    }

    Token token = tokenizer.nextToken();

    switch (token.type) {
        case TOKEN_LPAREN:
            return parse_list(tokenizer);

        case TOKEN_VECTOR_START:
            // Handle vector literal: #(element1 element2 ...)
            // Supports nested vectors: #(#(1 2) #(3 4)) → 2D tensor
            return parse_vector_body(tokenizer);

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
                // Reader prefix chars (' ` , #) modify the next expression —
                // don't treat them as standalone atoms, continue to read what follows.
                if (c == '\'' || c == '`' || c == ',' || c == '#') {
                    found_expression = true;
                    continue;
                }
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
