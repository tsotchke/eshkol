/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/eshkol.h>
#include <pthread.h>
#include <sys/resource.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "../../lib/frontend/parser_task.h"

static constexpr size_t stack_bytes = 8 * 1024 * 1024;
static constexpr size_t nesting = 16000;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static eshkol_ast_t parse(const std::string& source) {
    eshkol_reset_parse_line_counter();
    eshkol_reset_parse_errors();
    eshkol_set_parse_source_context("parser-stack-test.esk");
    std::istringstream input(source);
    auto ast = eshkol_parse_next_ast_from_stream(input);
    require(ast.type != ESHKOL_INVALID && !eshkol_parse_had_error(), "parse failed");
    return ast;
}

static std::string nest(const std::string& open, const std::string& leaf,
                        const std::string& close) {
    std::string text;
    for (size_t i = 0; i < nesting; ++i) text += open;
    text += leaf;
    for (size_t i = 0; i < nesting; ++i) text += close;
    return text;
}

// Exercise exception propagation and frame destruction on the same explicit
// driver used by the grammar, including unoptimized compiler builds.
static size_t live_frames = 0;
struct LiveFrame {
    LiveFrame() { ++live_frames; }
    ~LiveFrame() { --live_frames; }
};
static ParserTask<size_t> failing_parse(size_t depth) {
    LiveFrame frame;
    if (!depth) throw std::runtime_error("expected leaf failure");
    co_return 1 + (co_await failing_parse(depth - 1));
}

static void test() {
    try {
        failing_parse(nesting).run();
        require(false, "missing parser exception");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "expected leaf failure", "wrong exception");
    }
    require(live_frames == 0, "parser continuation leak during exception unwind");

    auto ast = parse(nest("(+ 1 ", "0", ")"));
    auto* node = &ast;
    for (size_t i = 0; i < nesting; ++i) {
        require(node->type == ESHKOL_OP && node->operation.op == ESHKOL_CALL_OP &&
                std::string(node->operation.call_op.func->variable.id) == "+",
                "nested arithmetic operation changed");
        require(node->line == 1 && node->column == 2 + 5 * i && node->node_id != 0,
                "nested source location or node identity changed");
        require(node->operation.call_op.num_vars == 2, "arithmetic arity changed");
        require(node->operation.call_op.variables[0].int64_val == 1,
                "arithmetic operand order changed");
        node = &node->operation.call_op.variables[1];
    }
    require(node->type == ESHKOL_INT64 && node->int64_val == 0, "arithmetic leaf changed");

    ast = parse("'" + nest("(", "7 . 9", ")"));
    require(ast.operation.op == ESHKOL_QUOTE_OP, "reader quote wrapper lost");
    node = &ast.operation.call_op.variables[0];
    for (size_t i = 1; i < nesting; ++i) {
        require(node->operation.op == ESHKOL_CALL_OP &&
                node->operation.call_op.num_vars == 1, "quoted list nesting changed");
        node = &node->operation.call_op.variables[0];
    }
    require(node->operation.op == ESHKOL_CALL_OP &&
            std::string(node->operation.call_op.func->variable.id) == "cons" &&
            node->operation.call_op.num_vars == 2, "dotted tail changed");

    ast = parse(nest("`", "(,7)", ""));
    require(ast.operation.op == ESHKOL_QUASIQUOTE_OP, "quasiquote changed");
    parse("\"~{" + nest("(+ 1 ", "0", ")") + "}\"");
    parse("(lambda () " + nest("(+ 1 ", "0", ")") + ")");
    ast = parse("(define (typed (x : " + nest("(list ", "integer", ")") + ")) x)");
    require(ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP &&
            ast.operation.define_op.param_types, "typed function signature lost");
    auto* type = ast.operation.define_op.param_types[0];
    for (size_t i = 0; i < nesting; ++i) {
        require(type && type->kind == HOTT_TYPE_LIST, "nested type constructor changed");
        type = type->container.element_type;
    }
    require(type && type->kind == HOTT_TYPE_INTEGER, "nested type leaf changed");
    parse("(define (typed (x : " + nest("(-> integer ", "integer", ")") + ")) x)");
    parse("(define (typed (x : " + nest("(forall (a) ", "a", ")") + ")) x)");
    ast = parse(nest("#(", "7", ")"));
    require(ast.operation.op == ESHKOL_TENSOR_OP &&
            ast.operation.tensor_op.num_dimensions == nesting &&
            ast.operation.tensor_op.total_elements == 1 &&
            ast.operation.tensor_op.elements[0].int64_val == 7,
            "nested vector shape or value changed");
    parse(nest("(begin ", "7", ")"));
    parse(nest("(if #t ", "7", " 0)"));
    parse(nest("(let ((x 1)) ", "7", ")"));
    parse("(match 0 (" + nest("(cons _ ", "_", ")") + " 7))");
    parse("(define-syntax m (syntax-rules () ((m " + nest("(", "x", ")") + ") x)))");
    parse("(cond-expand (" + nest("(not ", "unknown", ")") + " 7) (else 9))");
    parse("(import " + nest("(only ", "(scheme base)", " x)") + ")");

    eshkol_reset_parse_errors();
    std::istringstream bad(nest("(+ 1\n", "(if)", "\n)"));
    ast = eshkol_parse_next_ast_from_stream(bad);
    require(ast.type == ESHKOL_INVALID || eshkol_parse_had_error(),
            "deep malformed syntax lost its diagnostic");
    require(parse("42").int64_val == 42, "parser did not recover after malformed input");
    std::cout << "PASS: parser preserves 16000-level syntax on an 8 MiB stack\n";
}

static void* worker(void*) {
    try { test(); return nullptr; }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return reinterpret_cast<void*>(1);
    }
}

int main() {
    // Bound the process on Linux. A dedicated 8 MiB stack also makes the test
    // meaningful on macOS, whose main-stack mapping is fixed by the linker.
    rlimit limit{stack_bytes, stack_bytes};
    if (setrlimit(RLIMIT_STACK, &limit)) return 2;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) return 2;
    if (pthread_attr_setstacksize(&attr, stack_bytes)) return 2;
    pthread_t thread;
    if (pthread_create(&thread, &attr, worker, nullptr)) return 2;
    pthread_attr_destroy(&attr);
    void* result = nullptr;
    if (pthread_join(thread, &result)) return 2;
    return result ? 1 : 0;
}
