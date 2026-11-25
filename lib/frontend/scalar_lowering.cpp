/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/scalar_lowering.h>
#include <atomic>

namespace eshkol::frontend {
namespace {
using Op = DifferentiationOperation;
using Kind = ScalarCoreKind;
std::atomic<ScalarLowerer> installed_lowerer{nullptr};
ScalarCoreTerm leaf(Kind kind, uint32_t slot) { return {kind, slot, {}}; }
void validate_graph(const DifferentiationGraph& graph) {
    const auto width = graph.parameters.size(), count = graph.nodes.size();
    if (!width || width > DifferentiationGraph::max_inputs || count < width ||
        count > DifferentiationGraph::max_nodes || graph.output >= count)
        throw std::logic_error("scalar lowering: invalid graph shape");
    for (size_t i = 0; i < count; ++i) {
        const auto& node = graph.nodes[i];
        const auto& rule = differentiation_rule(node.operation);
        if (!rule.numeric || !rule.incremental || node.inputs.size() != rule.arity ||
            (i < width && (node.operation != Op::Input || node.input_index != i)) ||
            (i >= width && node.operation == Op::Input))
            throw std::logic_error("scalar lowering: invalid operation, arity or input prefix");
        for (auto parent : node.inputs)
            if (parent >= i) throw std::logic_error("scalar lowering: non-topological operand");
    }
}
ScalarLowering seed_lower(const DifferentiationGraph& graph) {
    validate_graph(graph);
    ScalarLowering result{static_cast<uint32_t>(graph.parameters.size()), graph.output, {}, {}};
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        if (node.operation == Op::Input || node.operation == Op::Constant) {
            result.bindings.push_back(leaf(node.operation == Op::Input ? Kind::Input : Kind::Literal,
                node.operation == Op::Input ? node.input_index : static_cast<uint32_t>(i)));
            result.kernels.push_back(leaf(Kind::Absent, 0));
        } else {
            const auto primitive = static_cast<uint32_t>(node.operation) - static_cast<uint32_t>(Op::Add);
            result.bindings.push_back({Kind::Primitive, primitive,
                {leaf(Kind::Binding, node.inputs[0]), leaf(Kind::Binding, node.inputs[1])}});
            result.kernels.push_back({Kind::Primitive, primitive,
                {leaf(Kind::Read, node.inputs[0]), leaf(Kind::Read, node.inputs[1])}});
        }
    }
    return result;
}
bool is_leaf(const ScalarCoreTerm& term, Kind kind, uint32_t payload) {
    return term.kind == kind && term.payload == payload && term.arguments.empty();
}
}

void verify_scalar_lowering(const DifferentiationGraph& graph, const ScalarLowering& lowered) {
    validate_graph(graph);
    if (lowered.width != graph.parameters.size() || lowered.output != graph.output ||
        lowered.bindings.size() != graph.nodes.size() || lowered.kernels.size() != graph.nodes.size())
        throw std::logic_error("scalar lowering: invalid core shape");
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        const auto& binding = lowered.bindings[i];
        const auto& kernel = lowered.kernels[i];
        bool valid = false;
        if (node.operation == Op::Input)
            valid = is_leaf(binding, Kind::Input, node.input_index) && is_leaf(kernel, Kind::Absent, 0);
        else if (node.operation == Op::Constant)
            valid = is_leaf(binding, Kind::Literal, i) && is_leaf(kernel, Kind::Absent, 0);
        else {
            // Check the semantic primitive independently of the seed's enum
            // arithmetic, then check both ordered operand identities and scopes.
            uint32_t primitive;
            switch (node.operation) {
                case Op::Add: primitive = 0; break;
                case Op::Subtract: primitive = 1; break;
                case Op::Multiply: primitive = 2; break;
                case Op::Divide: primitive = 3; break;
                default: throw std::logic_error("scalar lowering: unsupported primitive");
            }
            valid = binding.kind == Kind::Primitive && binding.payload == primitive && binding.arguments.size() == 2 &&
                kernel.kind == Kind::Primitive && kernel.payload == primitive && kernel.arguments.size() == 2;
            for (size_t j = 0; valid && j < 2; ++j)
                valid = is_leaf(binding.arguments[j], Kind::Binding, node.inputs[j]) &&
                        is_leaf(kernel.arguments[j], Kind::Read, node.inputs[j]);
        }
        if (!valid) throw std::logic_error("scalar lowering: core term does not preserve its source node");
    }
}
void install_scalar_lowerer(ScalarLowerer lowerer) {
    if (!lowerer) throw std::logic_error("cannot install a null scalar compiler pass");
    ScalarLowerer expected = nullptr;
    if (!installed_lowerer.compare_exchange_strong(expected, lowerer) && expected != lowerer)
        throw std::logic_error("scalar compiler pass is already installed");
}
const char* scalar_lowering_backend() { return installed_lowerer.load() ? "eshkol" : "seed"; }
ScalarLowering scalar_lower(const DifferentiationGraph& graph) {
    const auto lowerer = installed_lowerer.load();
    auto result = lowerer ? lowerer(graph) : seed_lower(graph);
    // Every caller, including an embedding installing its own pass, receives a
    // fully checked result. An installed pass's refusal never retries the seed.
    verify_scalar_lowering(graph, result);
    return result;
}
void write_scalar_core(std::ostream& out, const ScalarCoreTerm& term,
        const DifferentiationGraph& graph, const char* input, const char* prefix) {
    switch (term.kind) {
        case Kind::Input: out << "(vector-ref " << input << ' ' << term.payload << ')'; break;
        case Kind::Literal: out << graph.nodes.at(term.payload).literal; break;
        case Kind::Binding: out << prefix << term.payload; break;
        case Kind::Read: out << "(__ig_get " << term.payload << ')'; break;
        case Kind::Absent: out << "#f"; break;
        case Kind::Primitive: {
            static constexpr const char* names[] = {"+", "-", "*", "/"};
            if (term.payload >= 4 || term.arguments.size() != 2)
                throw std::logic_error("scalar lowering: invalid primitive term");
            out << '(' << names[term.payload];
            for (const auto& argument : term.arguments) {
                out << ' '; write_scalar_core(out, argument, graph, input, prefix);
            }
            out << ')'; break;
        }
        default: throw std::logic_error("scalar lowering: unknown core term");
    }
}
}
