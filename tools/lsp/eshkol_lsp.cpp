/*
 * Eshkol Language Server Protocol (LSP) Implementation
 *
 * Provides IDE integration via the LSP protocol over stdin/stdout.
 * Features:
 *   - textDocument/publishDiagnostics (parse errors)
 *   - textDocument/completion (keyword + builtin completion)
 *   - textDocument/hover (type and documentation info)
 *   - textDocument/definition (symbol navigation)
 *   - textDocument/didOpen, didChange, didClose (document sync)
 *
 * JSON-RPC 2.0 transport with Content-Length headers.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <regex>

#include "eshkol/eshkol.h"

// ============================================================================
// Minimal JSON Implementation (no external dependencies)
// ============================================================================

class JsonValue {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    JsonValue() : type_(Null) {}
    explicit JsonValue(bool b) : type_(Bool), bool_val_(b) {}
    explicit JsonValue(int64_t n) : type_(Number), num_val_(static_cast<double>(n)) {}
    explicit JsonValue(double n) : type_(Number), num_val_(n) {}
    explicit JsonValue(const std::string& s) : type_(String), str_val_(s) {}
    explicit JsonValue(const char* s) : type_(String), str_val_(s ? s : "") {}

    static JsonValue null() { return JsonValue(); }
    static JsonValue array() { JsonValue v; v.type_ = Array; return v; }
    static JsonValue object() { JsonValue v; v.type_ = Object; return v; }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Null; }
    bool is_string() const { return type_ == String; }
    bool is_number() const { return type_ == Number; }
    bool is_object() const { return type_ == Object; }
    bool is_array() const { return type_ == Array; }

    bool as_bool() const { return bool_val_; }
    double as_number() const { return num_val_; }
    int64_t as_int() const { return static_cast<int64_t>(num_val_); }
    const std::string& as_string() const { return str_val_; }

    // Object access
    JsonValue& operator[](const std::string& key) {
        type_ = Object;
        return obj_val_[key];
    }
    const JsonValue& get(const std::string& key) const {
        static JsonValue null_val;
        auto it = obj_val_.find(key);
        return it != obj_val_.end() ? it->second : null_val;
    }
    bool has(const std::string& key) const {
        return obj_val_.find(key) != obj_val_.end();
    }

    // Array access
    void push(const JsonValue& v) {
        type_ = Array;
        arr_val_.push_back(v);
    }
    size_t size() const {
        return type_ == Array ? arr_val_.size() : obj_val_.size();
    }
    const JsonValue& at(size_t i) const { return arr_val_.at(i); }

    // Serialization
    std::string to_string() const {
        std::ostringstream ss;
        serialize(ss);
        return ss.str();
    }

private:
    void serialize(std::ostringstream& ss) const {
        switch (type_) {
            case Null: ss << "null"; break;
            case Bool: ss << (bool_val_ ? "true" : "false"); break;
            case Number: {
                if (num_val_ == static_cast<int64_t>(num_val_)) {
                    ss << static_cast<int64_t>(num_val_);
                } else {
                    ss << num_val_;
                }
                break;
            }
            case String: {
                ss << '"';
                for (char c : str_val_) {
                    switch (c) {
                        case '"': ss << "\\\""; break;
                        case '\\': ss << "\\\\"; break;
                        case '\n': ss << "\\n"; break;
                        case '\r': ss << "\\r"; break;
                        case '\t': ss << "\\t"; break;
                        default: ss << c;
                    }
                }
                ss << '"';
                break;
            }
            case Array: {
                ss << '[';
                for (size_t i = 0; i < arr_val_.size(); i++) {
                    if (i > 0) ss << ',';
                    arr_val_[i].serialize(ss);
                }
                ss << ']';
                break;
            }
            case Object: {
                ss << '{';
                bool first = true;
                for (const auto& [k, v] : obj_val_) {
                    if (!first) ss << ',';
                    first = false;
                    ss << '"';
                    for (char c : k) {
                        if (c == '"') ss << "\\\"";
                        else if (c == '\\') ss << "\\\\";
                        else ss << c;
                    }
                    ss << "\":";
                    v.serialize(ss);
                }
                ss << '}';
                break;
            }
        }
    }

    Type type_ = Null;
    bool bool_val_ = false;
    double num_val_ = 0;
    std::string str_val_;
    std::vector<JsonValue> arr_val_;
    std::unordered_map<std::string, JsonValue> obj_val_;
};

// Minimal JSON parser
class JsonParser {
public:
    static JsonValue parse(const std::string& input) {
        size_t pos = 0;
        return parse_value(input, pos);
    }

private:
    static void skip_ws(const std::string& s, size_t& pos) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            pos++;
    }

    static JsonValue parse_value(const std::string& s, size_t& pos) {
        skip_ws(s, pos);
        if (pos >= s.size()) return JsonValue::null();

        char c = s[pos];
        if (c == '"') return parse_string(s, pos);
        if (c == '{') return parse_object(s, pos);
        if (c == '[') return parse_array(s, pos);
        if (c == 't' || c == 'f') return parse_bool(s, pos);
        if (c == 'n') return parse_null(s, pos);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(s, pos);
        return JsonValue::null();
    }

    static JsonValue parse_string(const std::string& s, size_t& pos) {
        pos++; // skip opening quote
        std::string result;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                pos++;
                switch (s[pos]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case '/': result += '/'; break;
                    case 'u': {
                        // Basic unicode escape: \uXXXX
                        if (pos + 4 < s.size()) {
                            std::string hex = s.substr(pos + 1, 4);
                            unsigned int codepoint = std::stoul(hex, nullptr, 16);
                            if (codepoint < 0x80) {
                                result += static_cast<char>(codepoint);
                            } else if (codepoint < 0x800) {
                                result += static_cast<char>(0xC0 | (codepoint >> 6));
                                result += static_cast<char>(0x80 | (codepoint & 0x3F));
                            } else {
                                result += static_cast<char>(0xE0 | (codepoint >> 12));
                                result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (codepoint & 0x3F));
                            }
                            pos += 4;
                        }
                        break;
                    }
                    default: result += s[pos];
                }
            } else {
                result += s[pos];
            }
            pos++;
        }
        if (pos < s.size()) pos++; // skip closing quote
        return JsonValue(result);
    }

    static JsonValue parse_number(const std::string& s, size_t& pos) {
        size_t start = pos;
        if (s[pos] == '-') pos++;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
        if (pos < s.size() && s[pos] == '.') {
            pos++;
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            pos++;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++;
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
        }
        return JsonValue(std::stod(s.substr(start, pos - start)));
    }

    static JsonValue parse_bool(const std::string& s, size_t& pos) {
        if (s.substr(pos, 4) == "true") { pos += 4; return JsonValue(true); }
        if (s.substr(pos, 5) == "false") { pos += 5; return JsonValue(false); }
        return JsonValue::null();
    }

    static JsonValue parse_null(const std::string& s, size_t& pos) {
        if (s.substr(pos, 4) == "null") { pos += 4; }
        return JsonValue::null();
    }

    static JsonValue parse_object(const std::string& s, size_t& pos) {
        pos++; // skip {
        JsonValue obj = JsonValue::object();
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == '}') { pos++; return obj; }
        while (pos < s.size()) {
            skip_ws(s, pos);
            JsonValue key = parse_string(s, pos);
            skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ':') pos++;
            obj[key.as_string()] = parse_value(s, pos);
            skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ',') pos++;
            else break;
        }
        if (pos < s.size() && s[pos] == '}') pos++;
        return obj;
    }

    static JsonValue parse_array(const std::string& s, size_t& pos) {
        pos++; // skip [
        JsonValue arr = JsonValue::array();
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') { pos++; return arr; }
        while (pos < s.size()) {
            arr.push(parse_value(s, pos));
            skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ',') pos++;
            else break;
        }
        if (pos < s.size() && s[pos] == ']') pos++;
        return arr;
    }
};

// ============================================================================
// JSON-RPC Transport (Content-Length framing over stdin/stdout)
// ============================================================================

class JsonRpcTransport {
public:
    // Read a JSON-RPC message from stdin
    std::string read_message() {
        // Read headers
        std::string line;
        int content_length = -1;

        while (std::getline(std::cin, line)) {
            // Remove \r if present
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) break;  // Empty line = end of headers

            if (line.substr(0, 16) == "Content-Length: ") {
                content_length = std::stoi(line.substr(16));
            }
        }

        if (content_length <= 0) return "";

        // Read body
        std::string body(content_length, '\0');
        std::cin.read(&body[0], content_length);
        return body;
    }

    // Write a JSON-RPC message to stdout
    void write_message(const std::string& body) {
        std::cout << "Content-Length: " << body.size() << "\r\n\r\n" << body;
        std::cout.flush();
    }

    void send_response(const JsonValue& id, const JsonValue& result) {
        JsonValue response = JsonValue::object();
        response["jsonrpc"] = JsonValue("2.0");
        response["id"] = id;
        response["result"] = result;
        write_message(response.to_string());
    }

    void send_notification(const std::string& method, const JsonValue& params) {
        JsonValue notification = JsonValue::object();
        notification["jsonrpc"] = JsonValue("2.0");
        notification["method"] = JsonValue(method);
        notification["params"] = params;
        write_message(notification.to_string());
    }

    void send_error(const JsonValue& id, int code, const std::string& message) {
        JsonValue response = JsonValue::object();
        response["jsonrpc"] = JsonValue("2.0");
        response["id"] = id;
        JsonValue error = JsonValue::object();
        error["code"] = JsonValue(static_cast<int64_t>(code));
        error["message"] = JsonValue(message);
        response["error"] = error;
        write_message(response.to_string());
    }
};

// ============================================================================
// Document Store (tracks open files and their contents)
// ============================================================================

struct TextDocument {
    std::string uri;
    std::string content;
    int version = 0;
};

class DocumentStore {
public:
    void open(const std::string& uri, const std::string& content, int version) {
        docs_[uri] = {uri, content, version};
    }

    void update(const std::string& uri, const std::string& content, int version) {
        docs_[uri].content = content;
        docs_[uri].version = version;
    }

    void close(const std::string& uri) {
        docs_.erase(uri);
    }

    const TextDocument* get(const std::string& uri) const {
        auto it = docs_.find(uri);
        return it != docs_.end() ? &it->second : nullptr;
    }

    const std::unordered_map<std::string, TextDocument>& all() const { return docs_; }

private:
    std::unordered_map<std::string, TextDocument> docs_;
};

// ============================================================================
// Eshkol Language Server
// ============================================================================

class EshkolLanguageServer {
public:
    void run() {
        while (!shutdown_requested_) {
            std::string body = transport_.read_message();
            if (body.empty()) {
                if (std::cin.eof()) break;
                continue;
            }
            handle_message(body);
        }
    }

private:
    JsonRpcTransport transport_;
    DocumentStore documents_;
    bool initialized_ = false;
    bool shutdown_requested_ = false;

    // Eshkol keywords and special forms
    static const std::vector<std::string>& keywords() {
        static const std::vector<std::string> kw = {
            "define", "lambda", "if", "cond", "else", "case",
            "let", "let*", "letrec", "begin", "do",
            "and", "or", "not", "when", "unless",
            "set!", "quote", "quasiquote", "unquote", "unquote-splicing",
            "define-syntax", "syntax-rules", "let-syntax", "letrec-syntax",
            "import", "export", "library", "require", "provide",
            "define-record-type", "define-type",
            "with-region", "owned", "move", "borrow", "shared", "weak-ref",
            "guard", "raise", "with-exception-handler",
            "values", "call-with-values", "call-with-current-continuation",
            "dynamic-wind", "parameterize", "make-parameter",
        };
        return kw;
    }

    // Eshkol built-in functions
    static const std::vector<std::string>& builtins() {
        static const std::vector<std::string> bi = {
            // Arithmetic
            "+", "-", "*", "/", "=", "<", ">", "<=", ">=",
            "abs", "max", "min", "modulo", "remainder", "quotient",
            "floor", "ceiling", "truncate", "round",
            "sin", "cos", "tan", "exp", "log", "sqrt", "expt",
            "exact->inexact", "inexact->exact",
            "number->string", "string->number",
            // Predicates
            "number?", "integer?", "real?", "complex?",
            "exact?", "inexact?", "zero?", "positive?", "negative?",
            "boolean?", "char?", "string?", "symbol?", "procedure?",
            "null?", "pair?", "list?", "vector?", "tensor?",
            "eq?", "eqv?", "equal?",
            // Pairs/Lists
            "cons", "car", "cdr", "list", "length", "append", "reverse",
            "map", "for-each", "filter", "fold",
            "assoc", "assv", "assq", "member", "memv", "memq",
            "caar", "cadr", "cdar", "cddr",
            // Strings
            "string-length", "string-ref", "string-append", "substring",
            "string-copy", "string->list", "list->string",
            "string=?", "string<?",
            "symbol->string", "string->symbol",
            // Vectors
            "make-vector", "vector", "vector-length", "vector-ref", "vector-set!",
            // I/O
            "display", "newline", "write", "read",
            "open-input-file", "read-line", "close-port", "eof-object?",
            "read-file", "write-file",
            // Tensor operations
            "tensor", "tensor-rank", "tensor-shape", "tensor-ref",
            "tensor-dot", "tensor-sum", "tensor-mean",
            "tensor-map", "tensor-apply", "tensor-reduce",
            "tensor-reshape", "tensor-transpose", "tensor-slice",
            // Autodiff
            "gradient", "jacobian", "dual", "dual-value", "dual-derivative",
            // Parallel
            "parallel-map", "parallel-filter", "parallel-fold",
            "parallel-execute", "future", "force",
            // System
            "system", "sleep", "exit", "command-line",
            "current-seconds", "current-time-ms", "current-time-ns",
            "error", "display", "newline",
            // Environment
            "eval", "apply",
            "interaction-environment", "scheme-report-environment",
            "null-environment", "current-environment",
        };
        return bi;
    }

    // Built-in documentation
    static std::string get_doc(const std::string& name) {
        static const std::unordered_map<std::string, std::string> docs = {
            {"define", "(define name value) or (define (name params...) body...)\nDefine a variable or function."},
            {"lambda", "(lambda (params...) body...)\nCreate an anonymous function."},
            {"if", "(if test consequent alternate)\nConditional expression."},
            {"let", "(let ((var val) ...) body...)\nLocal variable binding."},
            {"cond", "(cond (test expr) ... (else expr))\nMulti-way conditional."},
            {"begin", "(begin expr1 expr2 ...)\nSequential evaluation."},
            {"cons", "(cons a b) -> pair\nCreate a pair (cons cell)."},
            {"car", "(car pair) -> value\nGet the first element of a pair."},
            {"cdr", "(cdr pair) -> value\nGet the second element of a pair."},
            {"map", "(map proc list) -> list\nApply proc to each element."},
            {"filter", "(filter pred list) -> list\nKeep elements matching predicate."},
            {"fold", "(fold proc init list) -> value\nReduce list with binary operation."},
            {"tensor", "(tensor dims... values...) -> tensor\nCreate an N-dimensional tensor."},
            {"tensor-dot", "(tensor-dot a b) -> tensor\nMatrix multiplication."},
            {"gradient", "(gradient f) -> f'\nCompute gradient of f via autodiff."},
            {"parallel-map", "(parallel-map proc list) -> list\nParallel map using thread pool."},
            {"owned", "(owned expr) -> value\nMark value as owned (linear type)."},
            {"move", "(move var) -> value\nTransfer ownership, invalidate source."},
            {"borrow", "(borrow var body...) -> value\nTemporary read-only access."},
            {"shared", "(shared expr) -> value\nCreate reference-counted shared value."},
            {"weak-ref", "(weak-ref val) -> value\nCreate weak reference to shared value."},
            {"with-region", "(with-region body...) -> value\nLexical memory region."},
            {"eval", "(eval expr [env]) -> value\nEvaluate expression at runtime."},
        };
        auto it = docs.find(name);
        return it != docs.end() ? it->second : "";
    }

    void handle_message(const std::string& body) {
        JsonValue msg = JsonParser::parse(body);
        std::string method = msg.get("method").as_string();
        JsonValue id = msg.get("id");
        JsonValue params = msg.get("params");

        if (method == "initialize") handle_initialize(id, params);
        else if (method == "initialized") { /* no-op */ }
        else if (method == "shutdown") handle_shutdown(id);
        else if (method == "exit") handle_exit();
        else if (method == "textDocument/didOpen") handle_did_open(params);
        else if (method == "textDocument/didChange") handle_did_change(params);
        else if (method == "textDocument/didClose") handle_did_close(params);
        else if (method == "textDocument/completion") handle_completion(id, params);
        else if (method == "textDocument/hover") handle_hover(id, params);
        else if (method == "textDocument/definition") handle_definition(id, params);
        else if (!id.is_null()) {
            transport_.send_error(id, -32601, "Method not found: " + method);
        }
    }

    void handle_initialize(const JsonValue& id, const JsonValue& params) {
        (void)params;
        JsonValue result = JsonValue::object();

        // Server capabilities
        JsonValue capabilities = JsonValue::object();

        // Text document sync: full content on change
        capabilities["textDocumentSync"] = JsonValue(int64_t(1));  // Full sync

        // Completion
        JsonValue completion = JsonValue::object();
        JsonValue triggers = JsonValue::array();
        triggers.push(JsonValue("("));
        triggers.push(JsonValue(" "));
        completion["triggerCharacters"] = triggers;
        capabilities["completionProvider"] = completion;

        // Hover
        capabilities["hoverProvider"] = JsonValue(true);

        // Go to definition
        capabilities["definitionProvider"] = JsonValue(true);

        result["capabilities"] = capabilities;

        JsonValue server_info = JsonValue::object();
        server_info["name"] = JsonValue("eshkol-lsp");
        server_info["version"] = JsonValue("1.1.0");
        result["serverInfo"] = server_info;

        transport_.send_response(id, result);
        initialized_ = true;
    }

    void handle_shutdown(const JsonValue& id) {
        transport_.send_response(id, JsonValue::null());
        shutdown_requested_ = true;
    }

    void handle_exit() {
        std::exit(shutdown_requested_ ? 0 : 1);
    }

    void handle_did_open(const JsonValue& params) {
        const JsonValue& doc = params.get("textDocument");
        std::string uri = doc.get("uri").as_string();
        std::string text = doc.get("text").as_string();
        int version = doc.get("version").as_int();
        documents_.open(uri, text, version);
        publish_diagnostics(uri);
    }

    void handle_did_change(const JsonValue& params) {
        const JsonValue& doc = params.get("textDocument");
        std::string uri = doc.get("uri").as_string();
        int version = doc.get("version").as_int();

        // Full sync: take the last content change
        const JsonValue& changes = params.get("contentChanges");
        if (changes.is_array() && changes.size() > 0) {
            std::string text = changes.at(changes.size() - 1).get("text").as_string();
            documents_.update(uri, text, version);
        }
        publish_diagnostics(uri);
    }

    void handle_did_close(const JsonValue& params) {
        std::string uri = params.get("textDocument").get("uri").as_string();
        documents_.close(uri);

        // Clear diagnostics for closed file
        JsonValue diag_params = JsonValue::object();
        diag_params["uri"] = JsonValue(uri);
        diag_params["diagnostics"] = JsonValue::array();
        transport_.send_notification("textDocument/publishDiagnostics", diag_params);
    }

    // Parse document and publish diagnostics
    void publish_diagnostics(const std::string& uri) {
        const TextDocument* doc = documents_.get(uri);
        if (!doc) return;

        JsonValue diagnostics = JsonValue::array();

        // Parse the document for errors using Eshkol's parser
        {
            std::istringstream stream(doc->content);
            try {
                while (stream.good() && !stream.eof()) {
                    // Skip whitespace and comments
                    int c;
                    while ((c = stream.peek()) != EOF) {
                        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                            stream.get();
                        } else if (c == ';') {
                            // Skip comment line
                            std::string comment_line;
                            std::getline(stream, comment_line);
                        } else {
                            break;
                        }
                    }
                    if (stream.eof() || stream.peek() == EOF) break;
                    eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(stream);
                    if (ast.type == ESHKOL_INVALID) break;
                }
            } catch (...) {
                // Parser threw an exception — ignore, rely on paren matching below
            }

            // Check for unbalanced parentheses
            bool in_string = false;
            bool in_comment = false;
            int paren_depth = 0;
            int line = 0, col = 0;
            for (size_t i = 0; i < doc->content.size(); i++) {
                char ch = doc->content[i];
                if (in_comment) {
                    if (ch == '\n') in_comment = false;
                } else if (in_string) {
                    if (ch == '\\' && i + 1 < doc->content.size()) { i++; col++; }
                    else if (ch == '"') in_string = false;
                } else {
                    if (ch == ';') in_comment = true;
                    else if (ch == '"') in_string = true;
                    else if (ch == '(') paren_depth++;
                    else if (ch == ')') {
                        paren_depth--;
                        if (paren_depth < 0) {
                            JsonValue diag = JsonValue::object();
                            JsonValue range = JsonValue::object();
                            JsonValue start = JsonValue::object();
                            start["line"] = JsonValue(int64_t(line));
                            start["character"] = JsonValue(int64_t(col));
                            JsonValue end = JsonValue::object();
                            end["line"] = JsonValue(int64_t(line));
                            end["character"] = JsonValue(int64_t(col + 1));
                            range["start"] = start;
                            range["end"] = end;
                            diag["range"] = range;
                            diag["severity"] = JsonValue(int64_t(1)); // Error
                            diag["source"] = JsonValue("eshkol");
                            diag["message"] = JsonValue("Unmatched closing parenthesis");
                            diagnostics.push(diag);
                            paren_depth = 0;
                        }
                    }
                }
                if (ch == '\n') { line++; col = 0; }
                else col++;
            }

            if (paren_depth > 0) {
                JsonValue diag = JsonValue::object();
                JsonValue range = JsonValue::object();
                JsonValue start = JsonValue::object();
                start["line"] = JsonValue(int64_t(line));
                start["character"] = JsonValue(int64_t(0));
                JsonValue end_pos = JsonValue::object();
                end_pos["line"] = JsonValue(int64_t(line));
                end_pos["character"] = JsonValue(int64_t(col));
                range["start"] = start;
                range["end"] = end_pos;
                diag["range"] = range;
                diag["severity"] = JsonValue(int64_t(1)); // Error
                diag["source"] = JsonValue("eshkol");
                diag["message"] = JsonValue(std::to_string(paren_depth) + " unclosed parenthesis(es)");
                diagnostics.push(diag);
            }
        }

        JsonValue diag_params = JsonValue::object();
        diag_params["uri"] = JsonValue(uri);
        diag_params["diagnostics"] = diagnostics;
        transport_.send_notification("textDocument/publishDiagnostics", diag_params);
    }

    // Completion: keywords, builtins, and document-local symbols
    void handle_completion(const JsonValue& id, const JsonValue& params) {
        std::string uri = params.get("textDocument").get("uri").as_string();
        const TextDocument* doc = documents_.get(uri);

        JsonValue items = JsonValue::array();

        // Get the word being typed at the cursor position
        int line = params.get("position").get("line").as_int();
        int character = params.get("position").get("character").as_int();
        std::string prefix;
        if (doc) {
            prefix = get_word_at(doc->content, line, character);
        }

        // Add keywords (kind=14 = Keyword)
        for (const auto& kw : keywords()) {
            if (!prefix.empty() && kw.find(prefix) == std::string::npos) continue;
            JsonValue item = JsonValue::object();
            item["label"] = JsonValue(kw);
            item["kind"] = JsonValue(int64_t(14));  // Keyword
            item["detail"] = JsonValue("keyword");
            std::string doc_str = get_doc(kw);
            if (!doc_str.empty()) {
                item["documentation"] = JsonValue(doc_str);
            }
            items.push(item);
        }

        // Add builtins (kind=3 = Function)
        for (const auto& bi : builtins()) {
            if (!prefix.empty() && bi.find(prefix) == std::string::npos) continue;
            JsonValue item = JsonValue::object();
            item["label"] = JsonValue(bi);
            item["kind"] = JsonValue(int64_t(3));  // Function
            item["detail"] = JsonValue("builtin");
            std::string doc_str = get_doc(bi);
            if (!doc_str.empty()) {
                item["documentation"] = JsonValue(doc_str);
            }
            items.push(item);
        }

        // Add document-local defines (kind=6 = Variable or 3 = Function)
        if (doc) {
            auto defs = extract_defines(doc->content);
            for (const auto& [name, is_func] : defs) {
                if (!prefix.empty() && name.find(prefix) == std::string::npos) continue;
                JsonValue item = JsonValue::object();
                item["label"] = JsonValue(name);
                item["kind"] = JsonValue(int64_t(is_func ? 3 : 6));
                item["detail"] = JsonValue(is_func ? "function" : "variable");
                items.push(item);
            }
        }

        JsonValue result = JsonValue::object();
        result["isIncomplete"] = JsonValue(false);
        result["items"] = items;
        transport_.send_response(id, result);
    }

    // Hover: show documentation for symbol under cursor
    void handle_hover(const JsonValue& id, const JsonValue& params) {
        std::string uri = params.get("textDocument").get("uri").as_string();
        const TextDocument* doc = documents_.get(uri);
        if (!doc) {
            transport_.send_response(id, JsonValue::null());
            return;
        }

        int line = params.get("position").get("line").as_int();
        int character = params.get("position").get("character").as_int();
        std::string word = get_word_at(doc->content, line, character);

        if (word.empty()) {
            transport_.send_response(id, JsonValue::null());
            return;
        }

        std::string doc_str = get_doc(word);
        if (doc_str.empty()) {
            // Check if it's a keyword or builtin
            auto& kw = keywords();
            auto& bi = builtins();
            if (std::find(kw.begin(), kw.end(), word) != kw.end()) {
                doc_str = "**" + word + "** — Eshkol special form";
            } else if (std::find(bi.begin(), bi.end(), word) != bi.end()) {
                doc_str = "**" + word + "** — Eshkol built-in function";
            } else {
                // Check document-local defines
                auto defs = extract_defines(doc->content);
                for (const auto& [name, is_func] : defs) {
                    if (name == word) {
                        doc_str = "**" + word + "** — " + (is_func ? "function" : "variable") + " (defined in this file)";
                        break;
                    }
                }
            }
        }

        if (doc_str.empty()) {
            transport_.send_response(id, JsonValue::null());
            return;
        }

        JsonValue result = JsonValue::object();
        JsonValue contents = JsonValue::object();
        contents["kind"] = JsonValue("markdown");
        contents["value"] = JsonValue(doc_str);
        result["contents"] = contents;
        transport_.send_response(id, result);
    }

    // Go to definition: find define for symbol
    void handle_definition(const JsonValue& id, const JsonValue& params) {
        std::string uri = params.get("textDocument").get("uri").as_string();
        const TextDocument* doc = documents_.get(uri);
        if (!doc) {
            transport_.send_response(id, JsonValue::null());
            return;
        }

        int line = params.get("position").get("line").as_int();
        int character = params.get("position").get("character").as_int();
        std::string word = get_word_at(doc->content, line, character);

        if (word.empty()) {
            transport_.send_response(id, JsonValue::null());
            return;
        }

        // Search for (define word or (define (word in the document
        auto loc = find_definition(doc->content, word);
        if (loc.first < 0) {
            transport_.send_response(id, JsonValue::null());
            return;
        }

        JsonValue result = JsonValue::object();
        result["uri"] = JsonValue(uri);
        JsonValue range = JsonValue::object();
        JsonValue start = JsonValue::object();
        start["line"] = JsonValue(int64_t(loc.first));
        start["character"] = JsonValue(int64_t(loc.second));
        JsonValue end_pos = JsonValue::object();
        end_pos["line"] = JsonValue(int64_t(loc.first));
        end_pos["character"] = JsonValue(int64_t(loc.second + word.size()));
        range["start"] = start;
        range["end"] = end_pos;
        result["range"] = range;
        transport_.send_response(id, result);
    }

    // ========================================================================
    // Helper functions
    // ========================================================================

    // Get the word at a given position in the text
    std::string get_word_at(const std::string& text, int target_line, int target_col) {
        int line = 0, col = 0;
        size_t pos = 0;

        // Navigate to the target line
        while (pos < text.size() && line < target_line) {
            if (text[pos] == '\n') line++;
            pos++;
        }
        pos += target_col;

        if (pos >= text.size()) return "";

        // Find word boundaries (Scheme identifier characters)
        auto is_ident = [](char c) {
            return c != '(' && c != ')' && c != '"' && c != '\'' &&
                   c != ' ' && c != '\t' && c != '\n' && c != '\r' &&
                   c != ';' && c != ',' && c != '`' && c != '#';
        };

        // Find start of word
        size_t start = pos;
        while (start > 0 && is_ident(text[start - 1])) start--;

        // Find end of word
        size_t end = pos;
        while (end < text.size() && is_ident(text[end])) end++;

        if (start == end) return "";
        return text.substr(start, end - start);
    }

    // Extract define'd names from document content
    std::vector<std::pair<std::string, bool>> extract_defines(const std::string& text) {
        std::vector<std::pair<std::string, bool>> result;

        // Simple regex-based extraction
        // Match (define name or (define (name
        size_t pos = 0;
        while (pos < text.size()) {
            pos = text.find("(define", pos);
            if (pos == std::string::npos) break;

            pos += 7;  // skip "(define"

            // Skip whitespace
            while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n'))
                pos++;

            if (pos >= text.size()) break;

            if (text[pos] == '(') {
                // Function definition: (define (name ...)
                pos++;
                while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t'))
                    pos++;

                size_t name_start = pos;
                while (pos < text.size() && text[pos] != ' ' && text[pos] != ')' &&
                       text[pos] != '\n' && text[pos] != '\t')
                    pos++;

                if (pos > name_start) {
                    result.push_back({text.substr(name_start, pos - name_start), true});
                }
            } else {
                // Variable definition: (define name ...)
                size_t name_start = pos;
                while (pos < text.size() && text[pos] != ' ' && text[pos] != ')' &&
                       text[pos] != '\n' && text[pos] != '\t')
                    pos++;

                if (pos > name_start) {
                    std::string name = text.substr(name_start, pos - name_start);
                    // Check if value is a lambda
                    size_t check = pos;
                    while (check < text.size() && (text[check] == ' ' || text[check] == '\n' || text[check] == '\t'))
                        check++;
                    bool is_func = (check + 7 < text.size() && text.substr(check, 7) == "(lambda");
                    result.push_back({name, is_func});
                }
            }
        }

        return result;
    }

    // Find the definition location of a name
    std::pair<int, int> find_definition(const std::string& text, const std::string& name) {
        // Look for (define name or (define (name
        std::string pattern1 = "(define " + name;
        std::string pattern2 = "(define (" + name;

        auto search = [&](const std::string& pattern, int offset) -> std::pair<int, int> {
            size_t pos = 0;
            while ((pos = text.find(pattern, pos)) != std::string::npos) {
                // Count line and column
                int line = 0, col = 0;
                for (size_t i = 0; i < pos + offset; i++) {
                    if (text[i] == '\n') { line++; col = 0; }
                    else col++;
                }
                return {line, col};
            }
            return {-1, -1};
        };

        auto loc = search(pattern2, 9);  // "(define (" = 9 chars before name
        if (loc.first >= 0) return loc;
        return search(pattern1, 8);  // "(define " = 8 chars before name
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Check for --version flag
    if (argc > 1 && (std::string(argv[1]) == "--version" || std::string(argv[1]) == "-v")) {
        std::cout << "eshkol-lsp 1.1.0" << std::endl;
        return 0;
    }

    // Check for --help flag
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: eshkol-lsp [options]" << std::endl;
        std::cout << "  Eshkol Language Server Protocol implementation" << std::endl;
        std::cout << "  Communicates via JSON-RPC 2.0 over stdin/stdout" << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -v, --version  Print version and exit" << std::endl;
        std::cout << "  -h, --help     Print this help and exit" << std::endl;
        return 0;
    }

    EshkolLanguageServer server;
    server.run();
    return 0;
}
