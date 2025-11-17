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

enum TokenType {
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_QUOTE,
    TOKEN_SYMBOL,
    TOKEN_STRING,
    TOKEN_NUMBER,
    TOKEN_BOOLEAN,
    TOKEN_EOF
};

struct Token {
    TokenType type;
    std::string value;
    size_t pos;
};

class SchemeTokenizer {
private:
    std::string input;
    size_t pos;
    size_t length;
    
public:
    SchemeTokenizer(const std::string& text) : input(text), pos(0), length(text.length()) {}
    
    Token nextToken() {
        skipWhitespace();
        
        if (pos >= length) {
            return {TOKEN_EOF, "", pos};
        }
        
        char ch = input[pos];
        
        switch (ch) {
            case '(':
                pos++;
                return {TOKEN_LPAREN, "(", pos - 1};
            case ')':
                pos++;
                return {TOKEN_RPAREN, ")", pos - 1};
            case '\'':
                pos++;
                return {TOKEN_QUOTE, "'", pos - 1};
            case '"':
                return readString();
            default:
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
        while (pos < length && std::isspace(input[pos])) {
            pos++;
        }
    }
    
    Token readString() {
        size_t start = pos;
        pos++; // skip opening quote
        std::string value;
        
        while (pos < length && input[pos] != '"') {
            if (input[pos] == '\\' && pos + 1 < length) {
                pos++;
                switch (input[pos]) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    default: value += input[pos]; break;
                }
            } else {
                value += input[pos];
            }
            pos++;
        }
        
        if (pos < length) pos++; // skip closing quote
        return {TOKEN_STRING, value, start};
    }
    
    Token readNumber() {
        size_t start = pos;
        std::string value;
        
        if (input[pos] == '-') {
            value += input[pos++];
        }
        
        while (pos < length && (std::isdigit(input[pos]) || input[pos] == '.')) {
            value += input[pos++];
        }
        
        return {TOKEN_NUMBER, value, start};
    }
    
    Token readBoolean() {
        size_t start = pos;
        pos++; // skip #
        
        if (pos < length && (input[pos] == 't' || input[pos] == 'f')) {
            std::string value = std::string("#") + input[pos];
            pos++;
            return {TOKEN_BOOLEAN, value, start};
        }
        
        // Invalid boolean, treat as symbol
        pos = start;
        return readSymbol();
    }
    
    Token readSymbol() {
        size_t start = pos;
        std::string value;
        
        while (pos < length && !std::isspace(input[pos]) && 
               input[pos] != '(' && input[pos] != ')' && 
               input[pos] != '\'' && input[pos] != '"') {
            value += input[pos++];
        }
        
        return {TOKEN_SYMBOL, value, start};
    }
};

static eshkol_ast_t parse_atom(const Token& token) {
    eshkol_ast_t ast = {.type = ESHKOL_INVALID};
    
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
            if (token.value.find('.') != std::string::npos) {
                ast.type = ESHKOL_DOUBLE;
                ast.double_val = std::stod(token.value);
            } else {
                ast.type = ESHKOL_INT64;
                ast.int64_val = std::stoll(token.value);
            }
            break;
        }
        
        case TOKEN_BOOLEAN:
            ast.type = ESHKOL_UINT8;
            ast.uint8_val = (token.value == "#t") ? 1 : 0;
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
    if (op == "compose") return ESHKOL_COMPOSE_OP;
    if (op == "define") return ESHKOL_DEFINE_OP;
    if (op == "extern") return ESHKOL_EXTERN_OP;
    if (op == "extern-var") return ESHKOL_EXTERN_VAR_OP;
    if (op == "tensor") return ESHKOL_TENSOR_OP;
    if (op == "vector") return ESHKOL_TENSOR_OP;  // vector is just 1D tensor
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
    // Treat arithmetic operations as special CALL_OPs so they get proper argument handling
    // We'll store the operation type in the function name and handle display in the printer
    return ESHKOL_CALL_OP;
}

static eshkol_ast_t parse_expression(SchemeTokenizer& tokenizer);

static eshkol_ast_t parse_function_signature(SchemeTokenizer& tokenizer) {
    // Parse function signature: (func-name param1 param2 ...)
    // Returns an AST with function name and parameters
    eshkol_ast_t signature = {.type = ESHKOL_FUNC};
    std::vector<eshkol_ast_t> params;
    
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
        
        if (token.type == TOKEN_SYMBOL) {
            eshkol_ast_t param = {.type = ESHKOL_VAR};
            param.variable.id = new char[token.value.length() + 1];
            strcpy(param.variable.id, token.value.c_str());
            param.variable.data = nullptr;
            params.push_back(param);
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
        for (size_t i = 0; i < signature.eshkol_func.num_variables; i++) {
            signature.eshkol_func.variables[i] = params[i];
        }
    } else {
        signature.eshkol_func.variables = nullptr;
    }
    
    return signature;
}

static eshkol_ast_t parse_list(SchemeTokenizer& tokenizer) {
    eshkol_ast_t ast = {.type = ESHKOL_OP};
    std::vector<eshkol_ast_t> elements;
    
    Token token = tokenizer.nextToken();
    
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
                eshkol_ast_t func_signature = parse_function_signature(tokenizer);
                if (func_signature.type == ESHKOL_INVALID) {
                    ast.type = ESHKOL_INVALID;
                    return ast;
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
                
                // Create proper sequence for multiple expressions
                eshkol_ast_t body;
                if (body_expressions.size() == 1) {
                    body = body_expressions[0];
                } else if (body_expressions.size() > 1) {
                    // Create a sequence operation to hold all expressions
                    body.type = ESHKOL_OP;
                    body.operation.op = ESHKOL_SEQUENCE_OP;
                    body.operation.sequence_op.num_expressions = body_expressions.size();
                    body.operation.sequence_op.expressions = new eshkol_ast_t[body_expressions.size()];
                    
                    for (size_t i = 0; i < body_expressions.size(); i++) {
                        body.operation.sequence_op.expressions[i] = body_expressions[i];
                    }
                } else {
                    eshkol_error("Function body cannot be empty");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
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
            } else {
                then_expr = parse_atom(token);
            }
            
            // Parse else expression
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("if requires else-expression as third argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t else_expr;
            if (token.type == TOKEN_LPAREN) {
                else_expr = parse_list(tokenizer);
            } else {
                else_expr = parse_atom(token);
            }
            
            // Check for closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after if expression");
                ast.type = ESHKOL_INVALID;
                return ast;
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
            
            // Parse parameter list
            token = tokenizer.nextToken();
            if (token.type != TOKEN_LPAREN) {
                eshkol_error("lambda requires parameter list as first argument");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            std::vector<eshkol_ast_t> params;
            while (true) {
                token = tokenizer.nextToken();
                if (token.type == TOKEN_RPAREN) break;
                if (token.type == TOKEN_EOF) {
                    eshkol_error("Unexpected end of input in lambda parameter list");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
                
                if (token.type == TOKEN_SYMBOL) {
                    eshkol_ast_t param = {.type = ESHKOL_VAR};
                    param.variable.id = new char[token.value.length() + 1];
                    strcpy(param.variable.id, token.value.c_str());
                    param.variable.data = nullptr;
                    params.push_back(param);
                } else {
                    eshkol_error("Expected parameter name in lambda");
                    ast.type = ESHKOL_INVALID;
                    return ast;
                }
            }
            
            // Parse lambda body
            token = tokenizer.nextToken();
            if (token.type == TOKEN_EOF) {
                eshkol_error("lambda requires body expression");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            eshkol_ast_t body;
            if (token.type == TOKEN_LPAREN) {
                body = parse_list(tokenizer);
            } else {
                body = parse_atom(token);
            }
            
            // Check for closing paren
            token = tokenizer.nextToken();
            if (token.type != TOKEN_RPAREN) {
                eshkol_error("Expected closing parenthesis after lambda body");
                ast.type = ESHKOL_INVALID;
                return ast;
            }
            
            // Set up lambda operation
            ast.operation.lambda_op.num_params = params.size();
            if (ast.operation.lambda_op.num_params > 0) {
                ast.operation.lambda_op.parameters = new eshkol_ast_t[ast.operation.lambda_op.num_params];
                for (size_t i = 0; i < ast.operation.lambda_op.num_params; i++) {
                    ast.operation.lambda_op.parameters[i] = params[i];
                }
            } else {
                ast.operation.lambda_op.parameters = nullptr;
            }
            
            ast.operation.lambda_op.body = new eshkol_ast_t;
            *ast.operation.lambda_op.body = body;
            
            // For now, no captured variables (will implement closure analysis later)
            ast.operation.lambda_op.captured_vars = nullptr;
            ast.operation.lambda_op.num_captured = 0;
            
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
            
            // Parse the evaluation point
            token = tokenizer.nextToken();
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
            
            // Set up derivative operation
            ast.operation.derivative_op.function = new eshkol_ast_t;
            *ast.operation.derivative_op.function = function;
            
            ast.operation.derivative_op.point = new eshkol_ast_t;
            *ast.operation.derivative_op.point = point;
            
            ast.operation.derivative_op.mode = 0; // Forward-mode by default
            
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
        }
        
        // Parse arguments for non-define operations
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
            } else {
                element = parse_atom(token);
            }
            elements.push_back(element);
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
            
        case TOKEN_QUOTE: {
            // Handle quoted expressions - for now, just parse the next expression
            // A proper implementation would wrap it in a quote node
            return parse_expression(tokenizer);
        }
        
        case TOKEN_SYMBOL:
        case TOKEN_STRING:
        case TOKEN_NUMBER:
        case TOKEN_BOOLEAN:
            return parse_atom(token);
            
        case TOKEN_RPAREN:
            eshkol_error("Unexpected closing parenthesis");
            return {.type = ESHKOL_INVALID};
            
        case TOKEN_EOF:
        default:
            return {.type = ESHKOL_INVALID};
    }
}

eshkol_ast_t eshkol_parse_next_ast(std::ifstream &in_file)
{
    std::string input;
    std::string line;
    bool in_quote = false;
    int bracket_depth = 0;
    bool found_expression = false;
    
    // Read characters until we have a complete S-expression
    while (!in_file.eof()) {
        int c = in_file.get();
        if (in_file.eof()) break;
        
        // Handle comments - skip to end of line
        if (c == ';' && !in_quote) {
            std::getline(in_file, line); // consume rest of line
            if (bracket_depth == 0 && !input.empty()) {
                input += ' '; // Add space to separate from next token
            }
            continue;
        }
        
        // Track quotes
        if (c == '"' && (input.empty() || input.back() != '\\')) {
            in_quote = !in_quote;
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
                while (!in_file.eof()) {
                    int next_c = in_file.peek();
                    if (std::isspace(next_c) || next_c == '(' || next_c == ')' || next_c == ';') {
                        break;
                    }
                    c = in_file.get();
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
