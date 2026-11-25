/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/differentiation_graph.h>
#include <eshkol/frontend/scalar_lowering.h>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

namespace eshkol::frontend {
namespace {
using Op = DifferentiationOperation;
constexpr std::array<DifferentiationRule, 6> rules{{
    {Op::Input, "input", 0, true, true}, {Op::Constant, "constant", 0, true, true},
    {Op::Add, "+", 2, true, true}, {Op::Subtract, "-", 2, true, true},
    {Op::Multiply, "*", 2, true, true}, {Op::Divide, "/", 2, true, true}
}};
DifferentiationOrigin origin(const eshkol_ast_t& ast) {
    return {ast.node_id, ast.source_file_id, ast.line, ast.column};
}
[[noreturn]] void refuse(const eshkol_ast_t& ast, const std::string& message) {
    throw DifferentiationError("differentiation graph: " + message, origin(ast));
}
std::string integer_literal(const eshkol_ast_t& ast) {
    if (ast.type == ESHKOL_INT64) return std::to_string(ast.int64_val);
    if (ast.type == ESHKOL_BIGNUM_LITERAL && ast.str_val.ptr && ast.str_val.size <= 4097) {
        std::string value(ast.str_val.ptr);
        const size_t start = !value.empty() && (value[0] == '-' || value[0] == '+') ? 1 : 0;
        if (start == value.size() || value.size() > 4096) refuse(ast, "invalid integer literal");
        for (size_t i = start; i < value.size(); ++i)
            if (value[i] < '0' || value[i] > '9') refuse(ast, "invalid integer literal");
        return value;
    }
    refuse(ast, "expected an integer literal");
}
class Builder {
    DifferentiationGraph graph_;
    std::map<std::string, uint32_t> interned_;
    uint32_t visited_ = 0;
    std::function<bool(const eshkol_ast_t&)> primitive_binding_;
    using Environment = std::map<std::string, uint32_t>;
    uint32_t publish(Op op, std::vector<uint32_t> inputs, std::string literal,
                     uint32_t input, const eshkol_ast_t& ast) {
        std::string key = std::to_string(static_cast<unsigned>(op)) + ":" + literal;
        for (uint32_t parent : inputs) key += ":" + std::to_string(parent);
        if (op == Op::Input) key += ":" + std::to_string(input);
        auto known = interned_.find(key);
        if (known != interned_.end()) return known->second;
        if (graph_.nodes.size() >= DifferentiationGraph::max_nodes) refuse(ast, "node budget exceeded (1024)");
        uint32_t id = static_cast<uint32_t>(graph_.nodes.size());
        graph_.nodes.push_back({op, std::move(inputs), std::move(literal), input, origin(ast)});
        interned_.emplace(std::move(key), id);
        return id;
    }
    uint32_t constant(const std::string& value, const eshkol_ast_t& ast) {
        return publish(Op::Constant, {}, value, 0, ast);
    }
    uint32_t expression(const eshkol_ast_t& ast, const Environment& env, uint32_t depth) {
        if (++visited_ > 8192 || depth > DifferentiationGraph::max_depth)
            refuse(ast, "syntax/depth budget exceeded (8192/128)");
        if (ast.type == ESHKOL_VAR) {
            const std::string name = ast.variable.id ? ast.variable.id : "";
            auto value = env.find(name);
            if (value == env.end()) refuse(ast, "free capture '" + name + "' must be an explicit lambda input");
            return value->second;
        }
        if (ast.type == ESHKOL_INT64 || ast.type == ESHKOL_BIGNUM_LITERAL)
            return constant(integer_literal(ast), ast);
        if (ast.type == ESHKOL_DOUBLE) {
            if (!std::isfinite(ast.double_val)) refuse(ast, "nonfinite literal");
            char buffer[128];
            auto result = std::to_chars(buffer, buffer + sizeof(buffer), ast.double_val,
                                        std::chars_format::general, std::numeric_limits<double>::max_digits10);
            if (result.ec != std::errc{}) refuse(ast, "cannot encode floating literal");
            std::string text(buffer, result.ptr);
            if (text.find_first_of(".eE") == std::string::npos) text += ".0";
            return constant(text, ast);
        }
        if (ast.type != ESHKOL_OP) refuse(ast, "expected a scalar arithmetic expression");
        const auto& operation = ast.operation;
        if (operation.op == ESHKOL_LET_OP || operation.op == ESHKOL_LET_STAR_OP) {
            if (operation.let_op.name) refuse(ast, "named let/recursion is outside this graph");
            if (!operation.let_op.body || operation.let_op.num_bindings > 1024)
                refuse(ast, "malformed or over-budget let");
            if (operation.let_op.binding_types) {
                for (uint64_t i = 0; i < operation.let_op.num_bindings; ++i)
                    if (operation.let_op.binding_types[i]) refuse(ast, "typed bindings are outside this scalar graph");
            }
            Environment inner = env;
            std::map<std::string, uint32_t> parallel;
            for (uint64_t i = 0; i < operation.let_op.num_bindings; ++i) {
                const auto& binding = operation.let_op.bindings[i];
                if (binding.type != ESHKOL_CONS || !binding.cons_cell.car || !binding.cons_cell.cdr ||
                    binding.cons_cell.car->type != ESHKOL_VAR || !binding.cons_cell.car->variable.id)
                    refuse(binding, "malformed lexical binding");
                std::string name = binding.cons_cell.car->variable.id;
                if (operation.op == ESHKOL_LET_OP && parallel.count(name)) refuse(binding, "duplicate parallel let binding");
                uint32_t value = expression(*binding.cons_cell.cdr,
                    operation.op == ESHKOL_LET_OP ? env : inner, depth + 1);
                if (operation.op == ESHKOL_LET_OP) parallel[name] = value;
                else inner[name] = value;
            }
            for (const auto& [name, value] : parallel) inner[name] = value;
            return expression(*operation.let_op.body, inner, depth + 1);
        }
        if (operation.op == ESHKOL_SEQUENCE_OP) {
            if (operation.sequence_op.num_expressions != 1 || !operation.sequence_op.expressions)
                refuse(ast, "the graph body must contain one expression");
            return expression(operation.sequence_op.expressions[0], env, depth + 1);
        }
        if (operation.op != ESHKOL_CALL_OP && operation.op != ESHKOL_ADD_OP &&
            operation.op != ESHKOL_SUB_OP && operation.op != ESHKOL_MUL_OP && operation.op != ESHKOL_DIV_OP)
            refuse(ast, "effects, control flow and nested callable forms are unsupported");
        const auto& call = operation.call_op;
        std::string name;
        if (call.func && call.func->type == ESHKOL_VAR && call.func->variable.id) name = call.func->variable.id;
        else if (operation.op == ESHKOL_ADD_OP) name = "+";
        else if (operation.op == ESHKOL_SUB_OP) name = "-";
        else if (operation.op == ESHKOL_MUL_OP) name = "*";
        else if (operation.op == ESHKOL_DIV_OP) name = "/";
        if (env.count(name)) refuse(ast, "a lexical callable shadows an arithmetic operator");
        if (name == "make-rational" && call.num_vars == 2 && call.variables) {
            // Reader-created rational literals have no callable identity;
            // explicitly written constructor calls still require admission.
            if (primitive_binding_ && call.func && call.func->node_id != ESHKOL_NODE_ID_NONE &&
                !primitive_binding_(ast)) refuse(ast, "arithmetic operator is not bound to its primitive");
            std::string numerator = integer_literal(call.variables[0]);
            std::string denominator = integer_literal(call.variables[1]);
            if (denominator == "0" || denominator == "+0" || denominator == "-0") refuse(ast, "zero rational denominator");
            return constant(numerator + "/" + denominator, ast);
        }
        Op kind;
        if (name == "+") kind = Op::Add;
        else if (name == "-") kind = Op::Subtract;
        else if (name == "*") kind = Op::Multiply;
        else if (name == "/") kind = Op::Divide;
        else refuse(ast, "opaque call '" + name + "' is unsupported");
        if (primitive_binding_ && !primitive_binding_(ast)) refuse(ast, "arithmetic operator is not bound to its primitive");
        if (call.num_vars > 1024 || (call.num_vars && !call.variables)) refuse(ast, "invalid arithmetic arity");
        if (call.num_vars == 0) {
            if (kind == Op::Add || kind == Op::Multiply) return constant(kind == Op::Add ? "0" : "1", ast);
            refuse(ast, "subtraction/division need at least one operand");
        }
        std::vector<uint32_t> arguments;
        for (uint64_t i = 0; i < call.num_vars; ++i) arguments.push_back(expression(call.variables[i], env, depth + 1));
        if (arguments.size() == 1) {
            if (kind == Op::Add || kind == Op::Multiply) return arguments[0];
            return publish(kind, {constant(kind == Op::Subtract ? "0" : "1", ast), arguments[0]}, {}, 0, ast);
        }
        uint32_t result = arguments[0];
        for (size_t i = 1; i < arguments.size(); ++i) result = publish(kind, {result, arguments[i]}, {}, 0, ast);
        return result;
    }
public:
    explicit Builder(std::function<bool(const eshkol_ast_t&)> binding) : primitive_binding_(std::move(binding)) {}
    DifferentiationGraph build(const eshkol_ast_t& function) {
        if (function.type != ESHKOL_OP || function.operation.op != ESHKOL_LAMBDA_OP)
            refuse(function, "computation must be an inline lambda");
        const auto& lambda = function.operation.lambda_op;
        if (lambda.is_variadic || lambda.num_params < 1 || lambda.num_params > DifferentiationGraph::max_inputs ||
            !lambda.parameters || !lambda.body) refuse(function, "lambda needs 1..64 fixed scalar parameters");
        if (lambda.return_type) refuse(function, "typed lambdas are outside this scalar graph");
        if (lambda.param_types) for (uint64_t i = 0; i < lambda.num_params; ++i)
            if (lambda.param_types[i]) refuse(function, "typed lambdas are outside this scalar graph");
        Environment env;
        for (uint64_t i = 0; i < lambda.num_params; ++i) {
            const auto& parameter = lambda.parameters[i];
            if (parameter.type != ESHKOL_VAR || !parameter.variable.id || env.count(parameter.variable.id))
                refuse(parameter, "lambda parameters must be distinct identifiers");
            graph_.parameters.emplace_back(parameter.variable.id);
            env[parameter.variable.id] = publish(Op::Input, {}, {}, static_cast<uint32_t>(i), parameter);
        }
        graph_.output = expression(*lambda.body, env, 0);
        return std::move(graph_);
    }
};
}
const DifferentiationRule& differentiation_rule(DifferentiationOperation operation) {
    const size_t index = static_cast<size_t>(operation);
    if (index >= rules.size()) throw std::invalid_argument("unknown differentiation operation");
    return rules[index];
}
DifferentiationGraph DifferentiationGraph::build(const eshkol_ast_t& function,
        const std::function<bool(const eshkol_ast_t&)>& binding) { return Builder{binding}.build(function); }
std::string DifferentiationGraph::numeric_source() const {
    const auto lowered = scalar_lower(*this);
    std::ostringstream out;
    out << "(lambda (__dg_point) (if (not (and (vector? __dg_point) (= (vector-length __dg_point) "
        << parameters.size() << "))) (error \"differentiate numeric point width mismatch\")) "
        << "(let __dg_check ((__dg_i 0)) (if (< __dg_i " << parameters.size()
        << ") (let ((__dg_value (vector-ref __dg_point __dg_i))) "
        << "(if (not (and (real? __dg_value) (or (exact? __dg_value) (finite? __dg_value)))) "
        << "(error \"differentiate numeric requires finite real inputs\")) (__dg_check (+ __dg_i 1))))) "
        << "(gradient (lambda (__dg_inputs) (let* (";
    for (size_t i = 0; i < lowered.bindings.size(); ++i) {
        out << "(__dg_" << i << ' ';
        write_scalar_core(out, lowered.bindings[i], *this, "__dg_inputs", "__dg_");
        out << ')';
    }
    out << ") __dg_" << output << ")) __dg_point))";
    return out.str();
}
}
