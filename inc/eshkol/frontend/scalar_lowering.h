/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_FRONTEND_SCALAR_LOWERING_H
#define ESHKOL_FRONTEND_SCALAR_LOWERING_H
#include <eshkol/frontend/differentiation_graph.h>
#include <ostream>

namespace eshkol::frontend {
// A small, closed core language. Slots are compiler-owned identities, never
// source names. Literals refer back to owned graph nodes without conversion.
enum class ScalarCoreKind { Input, Literal, Primitive, Binding, Read, Absent };
struct ScalarCoreTerm {
    ScalarCoreKind kind;
    uint32_t payload = 0; // Input/literal/binding/read slot or primitive 0..3.
    std::vector<ScalarCoreTerm> arguments;
    bool operator==(const ScalarCoreTerm&) const = default;
};
struct ScalarLowering {
    uint32_t width = 0, output = 0;
    // Position i binds slot i and retains graph.nodes[i]'s source origin.
    std::vector<ScalarCoreTerm> bindings, kernels;
    bool operator==(const ScalarLowering&) const = default;
};
using ScalarLowerer = ScalarLowering (*)(const DifferentiationGraph&);
ScalarLowering scalar_lower(const DifferentiationGraph& graph);
void verify_scalar_lowering(const DifferentiationGraph& graph, const ScalarLowering& lowered);
void install_scalar_lowerer(ScalarLowerer lowerer);
const char* scalar_lowering_backend();
void enable_self_hosted_scalar_lowering();
// Render verified core terms; arithmetic selection belongs to the lowering
// pass. These names are supplied only by compiler-owned wrapper emitters.
void write_scalar_core(std::ostream& out, const ScalarCoreTerm& term,
    const DifferentiationGraph& graph, const char* input, const char* binding_prefix);
}
#endif
