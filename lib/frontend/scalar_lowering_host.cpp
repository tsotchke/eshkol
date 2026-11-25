/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/scalar_lowering.h>
#include "compiler_pass_values.h"

extern "C" void eshkol_scalar_lower_call(const eshkol_tagged_value_t*, eshkol_tagged_value_t*);

namespace eshkol::frontend {
namespace {
using namespace compiler_pass;
ScalarCoreTerm term(Value value, unsigned depth = 0) {
    // The language is shallow: one primitive with two leaf operands. Bound
    // decoding before semantic verification, including cyclic vector output.
    if (depth > 1) throw std::logic_error("scalar lowering: core nesting exceeds its grammar");
    const auto fields = elements(value, 3);
    if (fields.size() != 3) throw std::logic_error("scalar lowering: invalid core term shape");
    ScalarCoreTerm result{static_cast<ScalarCoreKind>(index(fields[0], 6)),
        static_cast<uint32_t>(index(fields[1], DifferentiationGraph::max_nodes)), {}};
    for (auto argument : elements(fields[2], 2)) result.arguments.push_back(term(argument, depth + 1));
    return result;
}
ScalarLowering selfhost_lower(const DifferentiationGraph& graph) {
    const auto count = graph.nodes.size(), width = graph.parameters.size();
    if (!width || width > DifferentiationGraph::max_inputs || count < width ||
        count > DifferentiationGraph::max_nodes || graph.output >= count)
        throw std::logic_error("scalar lowering: invalid graph shape");
    PlanningArena scope;
    std::vector<Value> nodes;
    for (const auto& node : graph.nodes) {
        // The public operation table remains authoritative for admission.
        const auto& rule = differentiation_rule(node.operation);
        if (!rule.numeric || !rule.incremental || node.inputs.size() != rule.arity)
            throw std::logic_error("scalar lowering: invalid graph operation or arity");
        std::vector<Value> operands;
        for (auto parent : node.inputs) operands.push_back(integer(parent));
        nodes.push_back(vector(scope.arena.get(), {integer(static_cast<size_t>(node.operation)),
            integer(node.input_index), vector(scope.arena.get(), operands)}));
    }
    auto input = vector(scope.arena.get(), {integer(1), integer(width), integer(graph.output),
        vector(scope.arena.get(), nodes)});
    Value output{};
    eshkol_scalar_lower_call(&input, &output);
    const auto fields = elements(output, 5);
    if (fields.size() != 5 || index(fields[0], 2) != 1)
        throw std::logic_error("scalar lowering: unsupported core schema");
    ScalarLowering result{static_cast<uint32_t>(index(fields[1], 65)),
        static_cast<uint32_t>(index(fields[2], count)), {}, {}};
    for (auto binding : elements(fields[3], count)) result.bindings.push_back(term(binding));
    for (auto kernel : elements(fields[4], count)) result.kernels.push_back(term(kernel));
    return result; // scalar_lower checks the complete owned result after arena release.
}
}
void enable_self_hosted_scalar_lowering() { install_scalar_lowerer(selfhost_lower); }
}
