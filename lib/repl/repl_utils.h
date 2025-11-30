//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#ifndef ESHKOL_REPL_UTILS_H
#define ESHKOL_REPL_UTILS_H

#include <string>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <unistd.h>
#include <iostream>

namespace eshkol {
namespace repl {

// ============================================================================
// ANSI Color Support
// ============================================================================

// Check if terminal supports colors
inline bool supports_color() {
    static int cached = -1;
    if (cached == -1) {
        const char* term = std::getenv("TERM");
        const char* colorterm = std::getenv("COLORTERM");
        const char* no_color = std::getenv("NO_COLOR");

        if (no_color && no_color[0] != '\0') {
            cached = 0;
        } else if (!isatty(STDOUT_FILENO)) {
            cached = 0;
        } else if (colorterm && colorterm[0] != '\0') {
            cached = 1;
        } else if (term) {
            std::string t(term);
            cached = (t.find("color") != std::string::npos ||
                     t.find("256") != std::string::npos ||
                     t.find("xterm") != std::string::npos ||
                     t.find("screen") != std::string::npos ||
                     t.find("tmux") != std::string::npos ||
                     t.find("vt100") != std::string::npos) ? 1 : 0;
        } else {
            cached = 0;
        }
    }
    return cached == 1;
}

// ANSI Color Codes
namespace color {
    // Reset
    inline const char* reset()    { return supports_color() ? "\033[0m" : ""; }

    // Regular colors
    inline const char* black()    { return supports_color() ? "\033[30m" : ""; }
    inline const char* red()      { return supports_color() ? "\033[31m" : ""; }
    inline const char* green()    { return supports_color() ? "\033[32m" : ""; }
    inline const char* yellow()   { return supports_color() ? "\033[33m" : ""; }
    inline const char* blue()     { return supports_color() ? "\033[34m" : ""; }
    inline const char* magenta()  { return supports_color() ? "\033[35m" : ""; }
    inline const char* cyan()     { return supports_color() ? "\033[36m" : ""; }
    inline const char* white()    { return supports_color() ? "\033[37m" : ""; }

    // Bright colors
    inline const char* bright_black()   { return supports_color() ? "\033[90m" : ""; }
    inline const char* bright_red()     { return supports_color() ? "\033[91m" : ""; }
    inline const char* bright_green()   { return supports_color() ? "\033[92m" : ""; }
    inline const char* bright_yellow()  { return supports_color() ? "\033[93m" : ""; }
    inline const char* bright_blue()    { return supports_color() ? "\033[94m" : ""; }
    inline const char* bright_magenta() { return supports_color() ? "\033[95m" : ""; }
    inline const char* bright_cyan()    { return supports_color() ? "\033[96m" : ""; }
    inline const char* bright_white()   { return supports_color() ? "\033[97m" : ""; }

    // Styles
    inline const char* bold()      { return supports_color() ? "\033[1m" : ""; }
    inline const char* dim()       { return supports_color() ? "\033[2m" : ""; }
    inline const char* italic()    { return supports_color() ? "\033[3m" : ""; }
    inline const char* underline() { return supports_color() ? "\033[4m" : ""; }

    // Semantic colors for Eshkol REPL
    inline const char* number()    { return bright_cyan(); }      // Numbers
    inline const char* string()    { return bright_green(); }     // Strings
    inline const char* symbol()    { return bright_yellow(); }    // Symbols/variables
    inline const char* keyword()   { return bright_magenta(); }   // Keywords (define, lambda, etc.)
    inline const char* function()  { return bright_blue(); }      // Function names
    inline const char* operator_() { return yellow(); }           // Operators (+, -, *, /)
    inline const char* comment()   { return bright_black(); }     // Comments
    inline const char* error()     { return bright_red(); }       // Errors
    inline const char* success()   { return bright_green(); }     // Success messages
    inline const char* info()      { return bright_cyan(); }      // Info messages
    inline const char* prompt()    { return bright_blue(); }      // Prompt
    inline const char* result()    { return cyan(); }             // Result values
    inline const char* type()      { return dim(); }              // Type annotations
    inline const char* boolean()   { return bright_magenta(); }   // Booleans
    inline const char* null()      { return bright_black(); }     // Null/empty list
    inline const char* list()      { return white(); }            // List delimiters
    inline const char* lambda()    { return bright_yellow(); }    // Lambda indicator
}

// ============================================================================
// Built-in Symbols for Tab Completion
// ============================================================================

// All built-in functions and special forms
inline const std::vector<std::string>& get_builtin_symbols() {
    static const std::vector<std::string> builtins = {
        // Special forms
        "define", "lambda", "if", "cond", "case", "let", "let*", "letrec",
        "begin", "do", "and", "or", "not", "when", "unless", "quote", "set!",

        // Arithmetic
        "+", "-", "*", "/", "abs", "quotient", "remainder", "modulo",
        "floor", "ceiling", "truncate", "round",
        "min", "max", "gcd", "lcm",

        // Comparison
        "=", "<", ">", "<=", ">=", "zero?", "positive?", "negative?",
        "odd?", "even?", "exact?", "inexact?",

        // Math functions
        "sin", "cos", "tan", "asin", "acos", "atan",
        "exp", "log", "sqrt", "pow", "expt",

        // List operations
        "cons", "car", "cdr", "list", "append", "reverse", "length",
        "list-ref", "list-tail", "list-head",
        "null?", "pair?", "list?",
        "set-car!", "set-cdr!",
        "caar", "cadr", "cdar", "cddr",
        "caaar", "caadr", "cadar", "caddr", "cdaar", "cdadr", "cddar", "cdddr",
        "map", "filter", "fold", "foldl", "foldr", "for-each",
        "memq", "memv", "member", "assq", "assv", "assoc",

        // Vector operations
        "vector", "make-vector", "vector-length", "vector-ref", "vector-set!",
        "vector->list", "list->vector", "vector-fill!",

        // String operations
        "string", "make-string", "string-length", "string-ref", "string-set!",
        "string-append", "substring", "string->list", "list->string",
        "string=?", "string<?", "string>?", "string<=?", "string>=?",
        "string-ci=?", "string-ci<?", "string-ci>?", "string-ci<=?", "string-ci>=?",

        // Type predicates
        "number?", "integer?", "real?", "complex?", "rational?",
        "string?", "symbol?", "char?", "boolean?", "procedure?",
        "vector?", "port?", "eof-object?",

        // Type conversions
        "number->string", "string->number",
        "symbol->string", "string->symbol",
        "char->integer", "integer->char",
        "exact->inexact", "inexact->exact",

        // I/O
        "display", "newline", "printf", "write", "read",
        "read-char", "write-char", "peek-char",

        // Boolean
        "eq?", "eqv?", "equal?",

        // Control
        "apply", "call/cc", "call-with-current-continuation",
        "values", "call-with-values",
        "dynamic-wind", "error",

        // Automatic differentiation
        "derivative", "gradient", "jacobian", "hessian",
        "divergence", "curl", "laplacian",
        "directional-derivative",

        // Tensor operations
        "tensor", "tensor-ref", "tensor-set!",
        "tensor-shape", "tensor-rank",
        "tensor-add", "tensor-sub", "tensor-mul", "tensor-div",
        "tensor-dot", "tensor-transpose",
        "tensor-map", "tensor-reduce",

        // REPL special
        "exit"
    };
    return builtins;
}

// Check if a string is a builtin
inline bool is_builtin(const std::string& name) {
    static std::unordered_set<std::string> builtin_set;
    if (builtin_set.empty()) {
        for (const auto& b : get_builtin_symbols()) {
            builtin_set.insert(b);
        }
    }
    return builtin_set.count(name) > 0;
}

// ============================================================================
// REPL Commands
// ============================================================================

struct ReplCommand {
    std::string name;
    std::string alias;
    std::string description;
    std::string usage;
};

inline const std::vector<ReplCommand>& get_repl_commands() {
    static const std::vector<ReplCommand> commands = {
        {":help",    ":h",  "Show this help message", ":help"},
        {":quit",    ":q",  "Exit the REPL", ":quit"},
        {":cancel",  ":c",  "Cancel multi-line input", ":cancel"},
        {":clear",   "",    "Clear the screen", ":clear"},
        {":env",     ":e",  "Show defined symbols in environment", ":env"},
        {":type",    ":t",  "Show type of an expression", ":type <expr>"},
        {":doc",     ":d",  "Show documentation for a function", ":doc <name>"},
        {":ast",     "",    "Show AST for an expression", ":ast <expr>"},
        {":time",    "",    "Time execution of an expression", ":time <expr>"},
        {":load",    ":l",  "Load and execute a file", ":load <filename>"},
        {":reload",  ":r",  "Reload the last loaded file", ":reload"},
        {":reset",   "",    "Reset the REPL state", ":reset"},
        {":history", "",    "Show command history", ":history [n]"},
        {":version", ":v",  "Show version information", ":version"},
        {":examples","",    "Show example expressions", ":examples"},
    };
    return commands;
}

// ============================================================================
// History File Path
// ============================================================================

inline std::string get_history_file_path() {
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.eshkol_history";
    }
    return ".eshkol_history";
}

// ============================================================================
// Pretty Printing Utilities
// ============================================================================

// Format a number with appropriate precision
inline std::string format_number(double value) {
    // Check if it's effectively an integer
    if (value == static_cast<int64_t>(value) &&
        value >= -1e15 && value <= 1e15) {
        return std::to_string(static_cast<int64_t>(value));
    }

    // Format with appropriate precision
    char buf[64];
    snprintf(buf, sizeof(buf), "%.15g", value);
    return buf;
}

// Truncate long strings with ellipsis
inline std::string truncate_string(const std::string& str, size_t max_len = 80) {
    if (str.length() <= max_len) {
        return str;
    }
    return str.substr(0, max_len - 3) + "...";
}

// ============================================================================
// Welcome Banner
// ============================================================================

inline void print_welcome_banner() {
    using namespace color;

    std::cout << "\n";
    std::cout << bold() << bright_cyan();
    std::cout << R"(  ╭────────────────────────────────────────────────────────────────╮)" << "\n";
    std::cout << R"(  │                                                                │)" << "\n";
    std::cout << R"(  │)" << bright_yellow() << "        ███████╗███████╗██╗  ██╗██╗  ██╗ ██████╗ ██╗            " << bright_cyan() << R"(│)" << "\n";
    std::cout << R"(  │)" << bright_yellow() << "        ██╔════╝██╔════╝██║  ██║██║ ██╔╝██╔═══██╗██║            " << bright_cyan() << R"(│)" << "\n";
    std::cout << R"(  │)" << bright_yellow() << "        █████╗  ███████╗███████║█████╔╝ ██║   ██║██║            " << bright_cyan() << R"(│)" << "\n";
    std::cout << R"(  │)" << bright_yellow() << "        ██╔══╝  ╚════██║██╔══██║██╔═██╗ ██║   ██║██║            " << bright_cyan() << R"(│)" << "\n";
    std::cout << R"(  │)" << bright_yellow() << "        ███████╗███████║██║  ██║██║  ██╗╚██████╔╝███████╗       " << bright_cyan() << R"(│)" << "\n";
    std::cout << R"(  │)" << bright_yellow() << "        ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝       " << bright_cyan() << R"(│)" << "\n";
    std::cout << R"(  │                                                                │)" << "\n";
    std::cout << R"(  │)" << reset() << dim() << "        A Scheme dialect with automatic differentiation         " << bold() << bright_cyan() << R"(│)" << "\n";
    std::cout << R"(  │                                                                │)" << "\n";
    std::cout << R"(  ╰────────────────────────────────────────────────────────────────╯)" << reset() << "\n";
    std::cout << "\n";

    std::cout << dim() << "  Version " << reset() << "0.1.1" << dim() << " | ";
    std::cout << "Type " << reset() << bright_blue() << ":help" << reset() << dim() << " for commands | ";
    std::cout << "Type " << reset() << bright_blue() << ":examples" << reset() << dim() << " for demos" << reset() << "\n";
    std::cout << dim() << "  Press " << reset() << "Ctrl+D" << dim() << " or type " << reset() << bright_blue() << "(exit)" << reset() << dim() << " to quit" << reset() << "\n";
    std::cout << "\n";
}

// ============================================================================
// Example Expressions
// ============================================================================

inline void print_examples() {
    using namespace color;

    std::cout << "\n" << bold() << bright_cyan() << "Eshkol Examples" << reset() << "\n";
    std::cout << dim() << "───────────────────────────────────────────────────────────" << reset() << "\n\n";

    std::cout << keyword() << "Arithmetic:" << reset() << "\n";
    std::cout << "  " << function() << "(+ 1 2)" << reset() << "               " << dim() << "; => 3" << reset() << "\n";
    std::cout << "  " << function() << "(* 2 (+ 3 4))" << reset() << "         " << dim() << "; => 14" << reset() << "\n";
    std::cout << "  " << function() << "(- 10 3)" << reset() << "              " << dim() << "; => 7" << reset() << "\n";
    std::cout << "\n";

    std::cout << keyword() << "Variables & Functions:" << reset() << "\n";
    std::cout << "  " << function() << "(define x 42)" << reset() << "\n";
    std::cout << "  " << function() << "(define (square n) (* n n))" << reset() << "\n";
    std::cout << "  " << function() << "(square 5)" << reset() << "             " << dim() << "; => 25" << reset() << "\n";
    std::cout << "\n";

    std::cout << keyword() << "Lambda & Closures:" << reset() << "\n";
    std::cout << "  " << function() << "(define f (lambda (x) (+ x 5)))" << reset() << "\n";
    std::cout << "  " << function() << "(f 10)" << reset() << "                " << dim() << "; => 15" << reset() << "\n";
    std::cout << "\n";

    std::cout << keyword() << "Lists:" << reset() << "\n";
    std::cout << "  " << function() << "(list 1 2 3)" << reset() << "           " << dim() << "; => (1 2 3)" << reset() << "\n";
    std::cout << "  " << function() << "(car (list 1 2 3))" << reset() << "     " << dim() << "; => 1" << reset() << "\n";
    std::cout << "  " << function() << "(cons 0 (list 1 2))" << reset() << "    " << dim() << "; => (0 1 2)" << reset() << "\n";
    std::cout << "\n";

    std::cout << keyword() << "Control Flow:" << reset() << "\n";
    std::cout << "  " << function() << "(if (> 5 3) 1 0)" << reset() << "       " << dim() << "; => 1" << reset() << "\n";
    std::cout << "  " << function() << "(let ((a 10)) (+ a 5))" << reset() << "  " << dim() << "; => 15" << reset() << "\n";
    std::cout << "\n";

    std::cout << keyword() << "Automatic Differentiation:" << reset() << dim() << " (unique to Eshkol!)" << reset() << "\n";
    std::cout << "  " << function() << "(derivative (lambda (x) (* x x)) 3.0)" << reset() << "\n";
    std::cout << "    " << dim() << "; => 6.0  (d/dx of x^2 at x=3)" << reset() << "\n";
    std::cout << "\n";
}

// ============================================================================
// Colored Prompt
// ============================================================================

inline std::string make_prompt(bool continuation = false, int line_num = 0, int depth = 0) {
    using namespace color;
    std::string p;

    if (!continuation) {
        p += prompt();
        p += bold();
        p += "eshkol";
        p += reset();
        p += prompt();
        p += "> ";
        p += reset();
    } else {
        p += dim();
        p += "  [";
        p += std::to_string(line_num);
        p += ",";
        p += std::to_string(depth);
        p += "]";
        p += prompt();
        p += "> ";
        p += reset();
    }

    return p;
}

// ============================================================================
// AST Type Names
// ============================================================================

inline const char* get_type_name(eshkol_type_t type) {
    switch (type) {
        case ESHKOL_INVALID: return "invalid";
        case ESHKOL_UNTYPED: return "untyped";
        case ESHKOL_UINT8:   return "uint8";
        case ESHKOL_UINT16:  return "uint16";
        case ESHKOL_UINT32:  return "uint32";
        case ESHKOL_UINT64:  return "uint64";
        case ESHKOL_INT8:    return "int8";
        case ESHKOL_INT16:   return "int16";
        case ESHKOL_INT32:   return "int32";
        case ESHKOL_INT64:   return "integer";
        case ESHKOL_DOUBLE:  return "number";
        case ESHKOL_STRING:  return "string";
        case ESHKOL_FUNC:    return "procedure";
        case ESHKOL_VAR:     return "symbol";
        case ESHKOL_OP:      return "expression";
        case ESHKOL_CONS:    return "pair";
        case ESHKOL_NULL:    return "null";
        case ESHKOL_TENSOR:  return "tensor";
        case ESHKOL_CHAR:    return "char";
        default:             return "unknown";
    }
}

inline const char* get_op_name(eshkol_op_t op) {
    switch (op) {
        case ESHKOL_INVALID_OP:      return "invalid-op";
        case ESHKOL_COMPOSE_OP:      return "compose";
        case ESHKOL_IF_OP:           return "if";
        case ESHKOL_ADD_OP:          return "+";
        case ESHKOL_SUB_OP:          return "-";
        case ESHKOL_MUL_OP:          return "*";
        case ESHKOL_DIV_OP:          return "/";
        case ESHKOL_CALL_OP:         return "application";
        case ESHKOL_DEFINE_OP:       return "define";
        case ESHKOL_SEQUENCE_OP:     return "begin";
        case ESHKOL_EXTERN_OP:       return "extern";
        case ESHKOL_EXTERN_VAR_OP:   return "extern-var";
        case ESHKOL_LAMBDA_OP:       return "lambda";
        case ESHKOL_LET_OP:          return "let";
        case ESHKOL_LET_STAR_OP:     return "let*";
        case ESHKOL_LETREC_OP:       return "letrec";
        case ESHKOL_AND_OP:          return "and";
        case ESHKOL_OR_OP:           return "or";
        case ESHKOL_COND_OP:         return "cond";
        case ESHKOL_CASE_OP:         return "case";
        case ESHKOL_DO_OP:           return "do";
        case ESHKOL_WHEN_OP:         return "when";
        case ESHKOL_UNLESS_OP:       return "unless";
        case ESHKOL_QUOTE_OP:        return "quote";
        case ESHKOL_SET_OP:          return "set!";
        case ESHKOL_TENSOR_OP:       return "tensor";
        case ESHKOL_DIFF_OP:         return "diff";
        case ESHKOL_DERIVATIVE_OP:   return "derivative";
        case ESHKOL_GRADIENT_OP:     return "gradient";
        case ESHKOL_JACOBIAN_OP:     return "jacobian";
        case ESHKOL_HESSIAN_OP:      return "hessian";
        case ESHKOL_DIVERGENCE_OP:   return "divergence";
        case ESHKOL_CURL_OP:         return "curl";
        case ESHKOL_LAPLACIAN_OP:    return "laplacian";
        case ESHKOL_DIRECTIONAL_DERIV_OP: return "directional-derivative";
        default:                     return "unknown-op";
    }
}

// Get detailed type string for an AST node
inline std::string get_ast_type_string(const eshkol_ast_t* ast) {
    if (!ast) return "nil";

    switch (ast->type) {
        case ESHKOL_INT64:
            return "integer";
        case ESHKOL_DOUBLE:
            return "number (inexact)";
        case ESHKOL_STRING:
            return "string";
        case ESHKOL_CHAR:
            return "char";
        case ESHKOL_VAR:
            return std::string("symbol '") + (ast->variable.id ? ast->variable.id : "?") + "'";
        case ESHKOL_CONS:
            return "pair";
        case ESHKOL_NULL:
            return "empty list";
        case ESHKOL_FUNC:
            if (ast->eshkol_func.is_lambda) {
                return "lambda";
            }
            return std::string("procedure '") + (ast->eshkol_func.id ? ast->eshkol_func.id : "?") + "'";
        case ESHKOL_OP: {
            std::string result = "expression (";
            result += get_op_name(ast->operation.op);
            result += ")";
            return result;
        }
        case ESHKOL_TENSOR:
            return "tensor";
        default:
            return get_type_name(ast->type);
    }
}

// ============================================================================
// Built-in Function Documentation
// ============================================================================

struct FunctionDoc {
    std::string name;
    std::string signature;
    std::string description;
    std::string example;
};

inline const std::vector<FunctionDoc>& get_function_docs() {
    static const std::vector<FunctionDoc> docs = {
        // Arithmetic
        {"+", "(+ n1 n2 ...)", "Add numbers together", "(+ 1 2 3) => 6"},
        {"-", "(- n1 n2 ...)", "Subtract numbers (or negate with one arg)", "(- 10 3) => 7"},
        {"*", "(* n1 n2 ...)", "Multiply numbers together", "(* 2 3 4) => 24"},
        {"/", "(/ n1 n2 ...)", "Divide numbers", "(/ 10 2) => 5"},
        {"abs", "(abs n)", "Absolute value", "(abs -5) => 5"},
        {"quotient", "(quotient n1 n2)", "Integer division", "(quotient 10 3) => 3"},
        {"remainder", "(remainder n1 n2)", "Division remainder", "(remainder 10 3) => 1"},
        {"modulo", "(modulo n1 n2)", "Modulo operation", "(modulo -10 3) => 2"},
        {"floor", "(floor n)", "Round down to nearest integer", "(floor 3.7) => 3.0"},
        {"ceiling", "(ceiling n)", "Round up to nearest integer", "(ceiling 3.2) => 4.0"},
        {"min", "(min n1 n2 ...)", "Minimum of numbers", "(min 3 1 4) => 1"},
        {"max", "(max n1 n2 ...)", "Maximum of numbers", "(max 3 1 4) => 4"},

        // Math
        {"sin", "(sin n)", "Sine (radians)", "(sin 0) => 0.0"},
        {"cos", "(cos n)", "Cosine (radians)", "(cos 0) => 1.0"},
        {"tan", "(tan n)", "Tangent (radians)", "(tan 0) => 0.0"},
        {"asin", "(asin n)", "Arcsine (returns radians)", "(asin 0) => 0.0"},
        {"acos", "(acos n)", "Arccosine (returns radians)", "(acos 1) => 0.0"},
        {"atan", "(atan n)", "Arctangent (returns radians)", "(atan 0) => 0.0"},
        {"exp", "(exp n)", "e^n (exponential)", "(exp 0) => 1.0"},
        {"log", "(log n)", "Natural logarithm (base e)", "(log 1) => 0.0"},
        {"sqrt", "(sqrt n)", "Square root", "(sqrt 4) => 2.0"},
        {"expt", "(expt base exp)", "Exponentiation (base^exp)", "(expt 2 10) => 1024"},

        // Comparison
        {"=", "(= n1 n2 ...)", "Numeric equality", "(= 1 1 1) => #t"},
        {"<", "(< n1 n2 ...)", "Less than (monotonically increasing)", "(< 1 2 3) => #t"},
        {">", "(> n1 n2 ...)", "Greater than (monotonically decreasing)", "(> 3 2 1) => #t"},
        {"<=", "(<= n1 n2 ...)", "Less than or equal", "(<= 1 1 2) => #t"},
        {">=", "(>= n1 n2 ...)", "Greater than or equal", "(>= 2 1 1) => #t"},
        {"zero?", "(zero? n)", "Test if n is zero", "(zero? 0) => #t"},
        {"positive?", "(positive? n)", "Test if n is positive", "(positive? 1) => #t"},
        {"negative?", "(negative? n)", "Test if n is negative", "(negative? -1) => #t"},
        {"odd?", "(odd? n)", "Test if integer is odd", "(odd? 3) => #t"},
        {"even?", "(even? n)", "Test if integer is even", "(even? 4) => #t"},

        // Lists
        {"cons", "(cons a b)", "Construct a pair", "(cons 1 2) => (1 . 2)"},
        {"car", "(car pair)", "First element of pair", "(car (cons 1 2)) => 1"},
        {"cdr", "(cdr pair)", "Second element of pair", "(cdr (cons 1 2)) => 2"},
        {"list", "(list e1 e2 ...)", "Create a list", "(list 1 2 3) => (1 2 3)"},
        {"append", "(append l1 l2 ...)", "Concatenate lists", "(append '(1) '(2 3)) => (1 2 3)"},
        {"reverse", "(reverse lst)", "Reverse a list", "(reverse '(1 2 3)) => (3 2 1)"},
        {"length", "(length lst)", "Length of a list", "(length '(1 2 3)) => 3"},
        {"list-ref", "(list-ref lst n)", "Get nth element (0-indexed)", "(list-ref '(a b c) 1) => b"},
        {"null?", "(null? obj)", "Test if obj is empty list", "(null? '()) => #t"},
        {"pair?", "(pair? obj)", "Test if obj is a pair", "(pair? '(1 . 2)) => #t"},
        {"list?", "(list? obj)", "Test if obj is a proper list", "(list? '(1 2 3)) => #t"},
        {"map", "(map proc lst)", "Apply proc to each element", "(map square '(1 2 3)) => (1 4 9)"},
        {"filter", "(filter pred lst)", "Keep elements matching pred", "(filter even? '(1 2 3 4)) => (2 4)"},
        {"fold", "(fold proc init lst)", "Fold list with proc", "(fold + 0 '(1 2 3)) => 6"},

        // Control
        {"if", "(if test then else)", "Conditional expression", "(if (> 2 1) 'yes 'no) => yes"},
        {"cond", "(cond (test1 e1) (test2 e2) ...)", "Multi-way conditional", "(cond ((> 1 2) 'a) (else 'b)) => b"},
        {"and", "(and e1 e2 ...)", "Short-circuit and", "(and #t #t) => #t"},
        {"or", "(or e1 e2 ...)", "Short-circuit or", "(or #f #t) => #t"},
        {"not", "(not x)", "Logical negation", "(not #f) => #t"},
        {"when", "(when test e1 e2 ...)", "Execute if test is true", "(when #t (display \"yes\"))"},
        {"unless", "(unless test e1 e2 ...)", "Execute if test is false", "(unless #f (display \"no\"))"},

        // Binding
        {"define", "(define name value) or (define (name args) body)", "Define a variable or function", "(define x 42)"},
        {"lambda", "(lambda (args) body)", "Create anonymous function", "(lambda (x) (* x x))"},
        {"let", "(let ((var val) ...) body)", "Local bindings (parallel)", "(let ((x 1)) (+ x 1)) => 2"},
        {"let*", "(let* ((var val) ...) body)", "Local bindings (sequential)", "(let* ((x 1) (y x)) y) => 1"},
        {"letrec", "(letrec ((var val) ...) body)", "Local recursive bindings", "(letrec ((f (lambda ...))) ...)"},
        {"set!", "(set! var value)", "Mutate a variable", "(set! x 10)"},

        // I/O
        {"display", "(display obj)", "Output obj to stdout", "(display \"Hello\")"},
        {"newline", "(newline)", "Output a newline", "(newline)"},

        // Type predicates
        {"number?", "(number? obj)", "Test if obj is a number", "(number? 42) => #t"},
        {"integer?", "(integer? obj)", "Test if obj is an integer", "(integer? 42) => #t"},
        {"string?", "(string? obj)", "Test if obj is a string", "(string? \"hi\") => #t"},
        {"symbol?", "(symbol? obj)", "Test if obj is a symbol", "(symbol? 'foo) => #t"},
        {"procedure?", "(procedure? obj)", "Test if obj is a procedure", "(procedure? +) => #t"},
        {"boolean?", "(boolean? obj)", "Test if obj is a boolean", "(boolean? #t) => #t"},

        // Equality
        {"eq?", "(eq? a b)", "Pointer equality", "(eq? 'a 'a) => #t"},
        {"eqv?", "(eqv? a b)", "Equivalence (numbers, chars)", "(eqv? 1 1) => #t"},
        {"equal?", "(equal? a b)", "Structural equality (recursive)", "(equal? '(1 2) '(1 2)) => #t"},

        // Automatic Differentiation (unique to Eshkol!)
        {"derivative", "(derivative f x)", "Compute df/dx at point x", "(derivative (lambda (x) (* x x)) 3) => 6.0"},
        {"gradient", "(gradient f point)", "Compute gradient of f at point", "(gradient f #(1.0 2.0))"},
        {"jacobian", "(jacobian f point)", "Compute Jacobian matrix of f at point", "(jacobian f #(1.0 2.0))"},
        {"hessian", "(hessian f point)", "Compute Hessian matrix of f at point", "(hessian f #(1.0 2.0))"},
        {"divergence", "(divergence f point)", "Compute divergence of vector field f", "(divergence f #(1.0 2.0 3.0))"},
        {"curl", "(curl f point)", "Compute curl of 3D vector field f", "(curl f #(1.0 2.0 3.0))"},
        {"laplacian", "(laplacian f point)", "Compute Laplacian of scalar field f", "(laplacian f #(1.0 2.0))"},
        {"directional-derivative", "(directional-derivative f point dir)", "Compute directional derivative", "(directional-derivative f pt dir)"},
    };
    return docs;
}

// Look up documentation for a function
inline const FunctionDoc* lookup_doc(const std::string& name) {
    const auto& docs = get_function_docs();
    for (const auto& doc : docs) {
        if (doc.name == name) {
            return &doc;
        }
    }
    return nullptr;
}

// Print documentation for a function
inline void print_doc(const std::string& name) {
    using namespace color;

    const FunctionDoc* doc = lookup_doc(name);
    if (!doc) {
        std::cout << error() << "No documentation found for: " << reset() << name << "\n";
        std::cout << dim() << "Try :doc without arguments for a list of documented functions" << reset() << "\n";
        return;
    }

    std::cout << "\n";
    std::cout << bold() << bright_cyan() << doc->name << reset() << "\n";
    std::cout << dim() << "─────────────────────────────────────" << reset() << "\n";
    std::cout << keyword() << "Signature: " << reset() << doc->signature << "\n";
    std::cout << "\n" << doc->description << "\n";
    std::cout << "\n" << dim() << "Example:" << reset() << "\n";
    std::cout << "  " << function() << doc->example << reset() << "\n";
    std::cout << "\n";
}

// Print list of documented topics
inline void print_doc_topics() {
    using namespace color;

    std::cout << "\n" << bold() << bright_cyan() << "Available Documentation Topics" << reset() << "\n";
    std::cout << dim() << "───────────────────────────────────────────────────────────" << reset() << "\n\n";

    std::cout << dim() << "Use " << reset() << bright_blue() << ":doc <name>" << reset() << dim() << " to get details\n\n" << reset();

    // Group by category
    std::cout << keyword() << "Arithmetic: " << reset();
    std::cout << "+ - * / abs quotient remainder modulo floor ceiling min max\n\n";

    std::cout << keyword() << "Math: " << reset();
    std::cout << "sin cos tan asin acos atan exp log sqrt expt\n\n";

    std::cout << keyword() << "Comparison: " << reset();
    std::cout << "= < > <= >= zero? positive? negative? odd? even?\n\n";

    std::cout << keyword() << "Lists: " << reset();
    std::cout << "cons car cdr list append reverse length list-ref null? pair? list? map filter fold\n\n";

    std::cout << keyword() << "Control: " << reset();
    std::cout << "if cond and or not when unless\n\n";

    std::cout << keyword() << "Binding: " << reset();
    std::cout << "define lambda let let* letrec set!\n\n";

    std::cout << keyword() << "I/O: " << reset();
    std::cout << "display newline\n\n";

    std::cout << keyword() << "Types: " << reset();
    std::cout << "number? integer? string? symbol? procedure? boolean?\n\n";

    std::cout << keyword() << "Equality: " << reset();
    std::cout << "eq? eqv? equal?\n\n";

    std::cout << keyword() << "AutoDiff: " << reset() << dim() << "(unique to Eshkol!)" << reset() << "\n";
    std::cout << "derivative gradient jacobian hessian divergence curl laplacian directional-derivative\n\n";
}

// ============================================================================
// Error Formatting
// ============================================================================

inline void print_error(const std::string& message, const std::string& detail = "") {
    using namespace color;
    std::cerr << bold() << error() << "Error: " << reset();
    std::cerr << message;
    if (!detail.empty()) {
        std::cerr << "\n" << dim() << "  " << detail << reset();
    }
    std::cerr << "\n";
}

inline void print_warning(const std::string& message) {
    using namespace color;
    std::cerr << bold() << yellow() << "Warning: " << reset();
    std::cerr << message << "\n";
}

inline void print_info(const std::string& message) {
    using namespace color;
    std::cout << info() << message << reset() << "\n";
}

inline void print_success(const std::string& message) {
    using namespace color;
    std::cout << success() << message << reset() << "\n";
}

} // namespace repl
} // namespace eshkol

#endif // ESHKOL_REPL_UTILS_H
