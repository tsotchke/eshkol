/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_FRONTEND_DIFFERENTIATION_GRAPH_H
#define ESHKOL_FRONTEND_DIFFERENTIATION_GRAPH_H
#include <eshkol/eshkol.h>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace eshkol::frontend {
// Compiler-owned, backend-neutral data. No AST pointers or runtime values escape.
enum class DifferentiationOperation { Input, Constant, Add, Subtract, Multiply, Divide };
struct DifferentiationRule {
    DifferentiationOperation operation;
    const char* spelling;
    unsigned arity;
    bool numeric;
    bool incremental;
};
const DifferentiationRule& differentiation_rule(DifferentiationOperation operation);
struct DifferentiationOrigin {
    eshkol_node_id_t node_id = ESHKOL_NODE_ID_NONE;
    uint32_t file_id = 0;
    uint32_t line = 0;
    uint32_t column = 0;
};
struct DifferentiationNode {
    DifferentiationOperation operation;
    std::vector<uint32_t> inputs;
    std::string literal; // Canonical, owned finite numeric literal (no source evaluation).
    uint32_t input_index = 0;
    DifferentiationOrigin origin;
};
class DifferentiationError : public std::runtime_error {
public:
    DifferentiationOrigin origin;
    DifferentiationError(const std::string& message, DifferentiationOrigin location)
        : std::runtime_error(message), origin(location) {}
};
struct DifferentiationGraph {
    static constexpr uint32_t max_inputs = 64;
    static constexpr uint32_t max_nodes = 1024;
    static constexpr uint32_t max_depth = 128;
    std::vector<std::string> parameters;
    std::vector<DifferentiationNode> nodes;
    uint32_t output = 0;
    // Validates the entire admitted expression, including unused let bindings.
    // Production callers must provide the post-resolution primitive predicate;
    // omission supports structural tests over hand-constructed ASTs only.
    static DifferentiationGraph build(const eshkol_ast_t& function,
        const std::function<bool(const eshkol_ast_t&)>& primitive_binding = {});
    // A closed point->gradient wrapper; actual AD stays in the existing engine.
    std::string numeric_source() const;
};
// Parser-private marker and generated-source gateway. The marker is emitted
// only by the explicit interpretation syntax; ordinary calls do not mint it.
inline constexpr const char* differentiation_numeric_marker = "__eshkol_graph_numeric";
inline constexpr const char* differentiation_incremental_marker = "__eshkol_graph_incremental";
// Tagged index zero is reserved and never returned by the source allocator,
// even when exhaustion degrades ordinary source identities to NONE.
inline constexpr eshkol_node_id_t differentiation_gateway_node = ESHKOL_NODE_ID_TAG << 24;
inline constexpr eshkol_node_id_t differentiation_incremental_gateway_node = (ESHKOL_NODE_ID_TAG ^ 1u) << 24;
static_assert(differentiation_gateway_node != ESHKOL_NODE_ID_NONE &&
              differentiation_incremental_gateway_node != ESHKOL_NODE_ID_NONE);
eshkol_ast_t parse_generated_differentiation(const std::string& source,
                                            DifferentiationOrigin location);
}
#endif
