/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/differentiation_graph.h>
#include <cstdio>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <vector>
using eshkol::frontend::DifferentiationGraph;
using eshkol::frontend::DifferentiationOperation;
namespace {
int passed = 0, failed = 0;
void check(bool ok, const char* name) { if (ok) ++passed; else { ++failed; std::fprintf(stderr, "FAIL: %s\n", name); } }
struct AstArena {
    std::deque<std::string> strings;
    std::vector<std::unique_ptr<eshkol_ast_t[]>> arrays;
    eshkol_ast_t* array(const std::vector<eshkol_ast_t>& values) {
        auto p = std::make_unique<eshkol_ast_t[]>(values.size());
        for (size_t i = 0; i < values.size(); ++i) p[i] = values[i];
        auto result = p.get(); arrays.push_back(std::move(p)); return result;
    }
    char* text(std::string value) { strings.push_back(std::move(value)); return strings.back().data(); }
    eshkol_ast_t var(const char* name) { eshkol_ast_t a{}; a.type = ESHKOL_VAR; a.variable.id = text(name); return a; }
    eshkol_ast_t integer(int64_t value) { eshkol_ast_t a{}; eshkol_ast_make_int64(&a, value); return a; }
    eshkol_ast_t call(const char* name, std::vector<eshkol_ast_t> args) {
        eshkol_ast_t a{}; a.type = ESHKOL_OP; a.operation.op = ESHKOL_CALL_OP;
        a.operation.call_op.func = array({var(name)}); a.operation.call_op.variables = array(args);
        a.operation.call_op.num_vars = args.size(); a.line = 7; a.column = 9; a.node_id = 17; return a;
    }
    eshkol_ast_t lambda(std::vector<eshkol_ast_t> params, eshkol_ast_t body) {
        eshkol_ast_t a{}; a.type = ESHKOL_OP; a.operation.op = ESHKOL_LAMBDA_OP;
        a.operation.lambda_op.parameters = array(params); a.operation.lambda_op.num_params = params.size();
        a.operation.lambda_op.body = array({body}); return a;
    }
    eshkol_ast_t let(eshkol_op_t op, std::vector<std::pair<std::string, eshkol_ast_t>> bindings, eshkol_ast_t body) {
        std::vector<eshkol_ast_t> pairs;
        for (const auto& [name, value] : bindings) {
            eshkol_ast_t pair{}; pair.type = ESHKOL_CONS; pair.cons_cell.car = array({var(name.c_str())});
            pair.cons_cell.cdr = array({value}); pairs.push_back(pair);
        }
        eshkol_ast_t a{}; a.type = ESHKOL_OP; a.operation.op = op;
        a.operation.let_op.bindings = array(pairs); a.operation.let_op.num_bindings = pairs.size();
        a.operation.let_op.body = array({body}); return a;
    }
};
bool refuses(const eshkol_ast_t& function, const char* message) {
    try { (void)DifferentiationGraph::build(function); return false; }
    catch (const eshkol::frontend::DifferentiationError& error) { return std::string(error.what()).find(message) != std::string::npos; }
}
}
int main() {
    AstArena a;
    auto square = a.call("*", {a.var("x"), a.var("x")});
    auto graph = DifferentiationGraph::build(a.lambda({a.var("x")}, a.call("+", {square, square})));
    check(graph.nodes.size() == 3, "shared arithmetic expression is one graph node");
    check(graph.nodes[1].inputs == std::vector<uint32_t>({0, 0}), "repeated input edges retained");
    check(graph.nodes[2].inputs == std::vector<uint32_t>({1, 1}), "shared result edges retained");
    bool binding_refused = false;
    try { (void)DifferentiationGraph::build(a.lambda({a.var("x")}, square), [](const auto&) { return false; }); }
    catch (const eshkol::frontend::DifferentiationError& error) {
        binding_refused = std::string(error.what()).find("not bound to its primitive") != std::string::npos;
    }
    check(binding_refused, "production admission consults resolved primitive binding");
    check(graph.nodes[1].origin.line == 7 && graph.nodes[1].origin.column == 9 && graph.nodes[1].origin.node_id == 17,
          "original node identity and source coordinates retained");
    auto numeric = graph.numeric_source();
    check(numeric.find("(gradient") != std::string::npos && numeric.find("(* __dg_0 __dg_0)") != std::string::npos,
          "numeric lowering uses the existing gradient engine");
    check(numeric.find("(vector-ref __dg_inputs 0)") != std::string::npos, "source scalar input maps to point slot");
    auto parallel = DifferentiationGraph::build(a.lambda({a.var("x"), a.var("y")},
        a.let(ESHKOL_LET_OP, {{"x", a.var("y")}, {"y", a.var("x")}}, a.call("-", {a.var("x"), a.var("y")}))));
    check(parallel.nodes.back().inputs == std::vector<uint32_t>({1, 0}), "parallel let reads original environment");
    auto sequential = DifferentiationGraph::build(a.lambda({a.var("x"), a.var("y")},
        a.let(ESHKOL_LET_STAR_OP, {{"x", a.var("y")}, {"y", a.var("x")}}, a.call("-", {a.var("x"), a.var("y")}))));
    check(sequential.nodes.back().inputs == std::vector<uint32_t>({1, 1}), "let star observes sequential shadowing");
    check(refuses(a.lambda({a.var("x")}, a.var("foreign")), "free capture"), "unlisted capture rejected");
    check(refuses(a.lambda({a.var("x")}, a.call("opaque", {a.var("x")})), "opaque call"), "opaque callback rejected");
    check(refuses(a.lambda({a.var("x"), a.var("x")}, a.var("x")), "distinct"), "duplicate formal rejected");
    check(refuses(a.lambda({a.var("x")}, a.let(ESHKOL_LET_OP,
        {{"z", a.integer(1)}, {"z", a.integer(2)}}, a.var("z"))), "duplicate parallel"), "duplicate parallel binding rejected");
    check(refuses(a.lambda({a.var("x")}, a.let(ESHKOL_LET_OP,
        {{"unused", a.call("display", {a.var("x")})}}, a.var("x"))), "opaque call"), "unused effectful binding still rejected");
    eshkol_ast_t infinity{}; eshkol_ast_make_double(&infinity, std::numeric_limits<double>::infinity());
    check(refuses(a.lambda({a.var("x")}, infinity), "nonfinite"), "nonfinite literal rejected");
    eshkol_ast_t negative_zero{}; eshkol_ast_make_double(&negative_zero, -0.0);
    eshkol_ast_t positive_zero{}; eshkol_ast_make_double(&positive_zero, 0.0);
    auto signed_zero = DifferentiationGraph::build(a.lambda({a.var("x")}, a.call("+", {negative_zero, positive_zero})));
    check(signed_zero.nodes.size() == 4 && signed_zero.nodes[1].literal == "-0.0", "signed floating literals not conflated");
    auto unary = DifferentiationGraph::build(a.lambda({a.var("x")}, a.call("/", {a.var("x")})));
    check(unary.nodes.back().operation == DifferentiationOperation::Divide && unary.nodes[1].literal == "1", "reciprocal normalizes to division");
    auto rational = DifferentiationGraph::build(a.lambda({a.var("x")}, a.call("make-rational", {a.integer(3), a.integer(7)})));
    check(rational.nodes.back().literal == "3/7", "exact rational literals retained as literals");
    check(refuses(a.lambda({a.var("x")}, a.call("make-rational", {a.integer(1), a.integer(0)})), "zero rational"), "invalid rational rejected");
    auto bignum = a.integer(0); bignum.type = ESHKOL_BIGNUM_LITERAL;
    bignum.str_val.ptr = a.text("10000000000000000000000001"); bignum.str_val.size = 27;
    auto owned = DifferentiationGraph::build(a.lambda({a.var("x")}, bignum));
    bignum.str_val.ptr[0] = '9';
    check(owned.nodes.back().literal == "10000000000000000000000001", "graph owns literal storage");
    auto variadic = a.lambda({a.var("x")}, a.var("x")); variadic.operation.lambda_op.is_variadic = 1;
    check(refuses(variadic, "fixed scalar"), "variadic lambda rejected");
    auto deep = a.var("x"); for (int i = 0; i < 140; ++i) deep = a.call("+", {deep, a.integer(1)});
    check(refuses(a.lambda({a.var("x")}, deep), "depth budget"), "deep source refuses at bounded compiler depth");
    std::vector<std::pair<std::string, eshkol_ast_t>> bindings;
    std::string prior = "x";
    for (int i = 0; i < 1024; ++i) { std::string name = "n" + std::to_string(i);
        bindings.emplace_back(name, a.call("+", {a.var(prior.c_str()), a.integer(1)})); prior = name; }
    check(refuses(a.lambda({a.var("x")}, a.let(ESHKOL_LET_STAR_OP, bindings, a.var(prior.c_str()))), "node budget"), "node budget checked across a shallow let spine");
    for (const auto& node : graph.nodes) check(eshkol::frontend::differentiation_rule(node.operation).numeric,
        "every admitted graph node has a numeric interpretation");
    std::printf("Passed: %d Failed: %d\nRESULT: %s\n", passed, failed, failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
