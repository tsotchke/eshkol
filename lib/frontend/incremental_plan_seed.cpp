/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/incrementalize.h>
#include <atomic>

namespace eshkol::frontend {
namespace {
std::atomic<IncrementalPlanner> installed_planner{nullptr};
IncrementalPlan seed_plan(const DifferentiationGraph& graph) {
    using Op = DifferentiationOperation;
    const size_t count = graph.nodes.size(), width = graph.parameters.size();
    if (!width || width > DifferentiationGraph::max_inputs || count < width ||
        count > DifferentiationGraph::max_nodes || graph.output >= count)
        throw std::logic_error("invalid graph passed to IncrementalizePass");
    IncrementalPlan plan;
    plan.consumers.resize(count);
    while (plan.capacity < count) { plan.capacity *= 2; ++plan.height; }
    for (size_t i = 0; i < count; ++i) {
        const auto& node = graph.nodes[i];
        if ((i < width && (node.operation != Op::Input || node.input_index != i)) ||
            (i >= width && node.operation == Op::Input))
            throw std::logic_error("incremental graph inputs are not a parameter prefix");
        const auto& rule = differentiation_rule(node.operation);
        if (!rule.incremental || node.inputs.size() != rule.arity)
            throw std::logic_error("invalid incremental graph operation or arity");
        if (!node.inputs.empty()) ++plan.kernels;
        for (auto parent : node.inputs) {
            if (parent >= i) throw std::logic_error("incremental graph is not topological");
            ++plan.edges;
            auto& list = plan.consumers[parent];
            if (list.empty() || list.back() != i) list.push_back(i);
        }
    }
    return plan;
}
}
void install_incremental_planner(IncrementalPlanner planner) {
    if (!planner) throw std::logic_error("cannot install a null incremental compiler pass");
    IncrementalPlanner expected = nullptr;
    if (!installed_planner.compare_exchange_strong(expected, planner) && expected != planner)
        throw std::logic_error("incremental compiler pass is already installed");
}
const char* incremental_plan_backend() { return installed_planner.load() ? "eshkol" : "seed"; }
IncrementalPlan incremental_plan(const DifferentiationGraph& graph) {
    const auto planner = installed_planner.load();
    return planner ? planner(graph) : seed_plan(graph);
}
}
