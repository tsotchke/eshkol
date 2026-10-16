/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/incrementalize.h>
#include <algorithm>
#include <sstream>

namespace eshkol::frontend {
void verify_incremental_plan(const DifferentiationGraph& graph, const IncrementalPlan& plan) {
    using Op = DifferentiationOperation;
    const size_t count = graph.nodes.size(), width = graph.parameters.size();
    if (!width || width > DifferentiationGraph::max_inputs || count < width ||
        count > DifferentiationGraph::max_nodes || graph.output >= count ||
        plan.consumers.size() != count || plan.height > 10 || plan.capacity < count ||
        plan.capacity != (size_t{1} << plan.height) ||
        (plan.height && plan.capacity / 2 >= count))
        throw std::logic_error("incremental plan has an invalid shape");
    // Independently verify the returned relation against source operands.
    // This checks completeness as well as legality without constructing a
    // second plan or accepting a sampled equivalence claim.
    std::vector<size_t> cursor(count, 0);
    size_t edges = 0, kernels = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto& node = graph.nodes[i];
        const auto& rule = differentiation_rule(node.operation);
        if (!rule.incremental || node.inputs.size() != rule.arity)
            throw std::logic_error("incremental plan has an invalid operation or arity");
        if ((i < width && (node.operation != Op::Input || node.input_index != i)) ||
            (i >= width && node.operation == Op::Input))
            throw std::logic_error("self-hosted incremental plan has an invalid input prefix");
        if (!node.inputs.empty()) ++kernels;
        for (size_t j = 0; j < node.inputs.size(); ++j) {
            const auto parent = node.inputs[j]; ++edges;
            if (parent >= i) throw std::logic_error("self-hosted incremental plan is not topological");
            if (j && parent == node.inputs[0]) continue;
            if (cursor[parent] >= plan.consumers[parent].size() ||
                plan.consumers[parent][cursor[parent]++] != i)
                throw std::logic_error("self-hosted incremental plan does not match source operands");
        }
    }
    for (size_t i = 0; i < count; ++i)
        if (cursor[i] != plan.consumers[i].size())
            throw std::logic_error("self-hosted incremental plan contains an extra consumer");
    if (edges != plan.edges || kernels != plan.kernels)
        throw std::logic_error("self-hosted incremental plan has invalid work metadata");
}

std::string IncrementalizePass::source(const DifferentiationGraph& graph) {
    using Op = DifferentiationOperation;
    const auto plan = incremental_plan(graph);
    const size_t count = graph.nodes.size(), width = graph.parameters.size();
    const auto& consumers = plan.consumers;
    const auto capacity = plan.capacity, height = plan.height;
    const auto edges = plan.edges, kernels = plan.kernels;
    // Context: baseline, kernels, consumers, capacity, width, output, info,
    //          read, write, minimum, check, factory, enqueue, refresh.
    // Tick: context, persistent values, queue, owned work receipt.
    // Work: kernels, changed values, trie visits, copied trie slots,
    //       enqueue attempts, inspected input slots.
    std::ostringstream out;
    out << "(lambda (__ig_point) "
        << "(if (not (and (vector? __ig_point) (= (vector-length __ig_point) " << width
        << "))) (error \"differentiate incremental point width mismatch\")) "
        << "(let* ((__ig_base (make-vector " << count << " 0)) (__ig_kernels (vector ";
    for (const auto& node : graph.nodes) {
        if (node.inputs.empty()) out << "#f ";
        else out << "(lambda (__ig_get) (" << differentiation_rule(node.operation).spelling
                 << " (__ig_get " << node.inputs[0] << ") (__ig_get " << node.inputs[1] << "))) ";
    }
    out << ")) (__ig_children (vector ";
    for (const auto& list : consumers) {
        out << "(vector ";
        for (size_t child : list) out << child << ' ';
        out << ") ";
    }
    out << ")) (__ig_context (vector __ig_base __ig_kernels __ig_children " << capacity << ' '
        << width << ' ' << graph.output << " (vector " << count << ' ' << edges << ' ' << width << ' '
        << height << ") #f #f #f #f #f #f #f))) ";

    // Initial evaluation is a compiled let spine. Only initialization touches
    // every node; ticks path-copy trie branches and execute queued kernels.
    out << "(let* (";
    for (size_t i = 0; i < count; ++i) {
        const auto& node = graph.nodes[i];
        out << "(__ig_" << i << " (__eshkol_incremental_check ";
        if (node.operation == Op::Input) out << "(vector-ref __ig_point " << node.input_index << ')';
        else if (node.operation == Op::Constant) out << node.literal;
        else out << '(' << differentiation_rule(node.operation).spelling
                 << " __ig_" << node.inputs[0] << " __ig_" << node.inputs[1] << ')';
        out << "))";
    }
    out << ") ";
    for (size_t i = 0; i < count; ++i) out << "(vector-set! __ig_base " << i << " __ig_" << i << ") ";
    out << "(__eshkol_incremental_state __ig_context (vector " << kernels << ' ' << count << " 0 0 0 " << width << ")))))";
    return out.str();
}
}
