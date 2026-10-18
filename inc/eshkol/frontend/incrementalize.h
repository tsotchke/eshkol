/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_FRONTEND_INCREMENTALIZE_H
#define ESHKOL_FRONTEND_INCREMENTALIZE_H
#include <eshkol/frontend/differentiation_graph.h>
namespace eshkol::frontend {
// Compile the admitted graph into a closed point->persistent-state program.
// Node kernels are ordinary native arithmetic closures, not an opcode interpreter.
class IncrementalizePass {
public:
    static std::string source(const DifferentiationGraph& graph);
};
}
#endif
