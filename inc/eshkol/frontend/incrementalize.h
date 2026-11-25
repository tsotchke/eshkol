/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_FRONTEND_INCREMENTALIZE_H
#define ESHKOL_FRONTEND_INCREMENTALIZE_H
#include <eshkol/frontend/differentiation_graph.h>
namespace eshkol::frontend {
// Version-1 scalar binary DAG plan. Source/binding identity remains on the
// validated graph; planning neither evaluates literals nor rewrites arithmetic.
struct IncrementalPlan {
    size_t capacity = 1, height = 0, edges = 0, kernels = 0;
    std::vector<std::vector<size_t>> consumers;
    bool operator==(const IncrementalPlan&) const = default;
};
// The seed library supplies the C++ bootstrap implementation. Final native
// tools explicitly link the Eshkol implementation instead (no runtime fallback).
IncrementalPlan incremental_plan(const DifferentiationGraph& graph);
const char* incremental_plan_backend();
void verify_incremental_plan(const DifferentiationGraph& graph, const IncrementalPlan& plan);
using IncrementalPlanner = IncrementalPlan (*)(const DifferentiationGraph&);
// Explicit, once-only startup registration. The seed never loads or evaluates
// a pass dynamically; an installed pass's failure is propagated without retry.
void install_incremental_planner(IncrementalPlanner planner);
void enable_self_hosted_incremental();
// Compile the admitted graph into a closed point->persistent-state program.
// Node kernels are ordinary native arithmetic closures, not an opcode interpreter.
class IncrementalizePass {
public:
    static std::string source(const DifferentiationGraph& graph);
};
}
#endif
