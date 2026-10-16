/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/incrementalize.h>
#include "../core/arena_memory.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/DynamicLibrary.h>
#include <cstddef>
#include <cstring>
#include <memory>

extern "C" void eshkol_incremental_plan_call(const eshkol_tagged_value_t*, eshkol_tagged_value_t*);
// Address-only declarations. Eshkol-to-Eshkol calls use the internal tagged
// convention, not the platform C aggregate convention; C++ never calls these.
extern "C" void __eshkol_compiler_incremental_state();
extern "C" void __eshkol_compiler_incremental_check();

namespace eshkol::frontend {
namespace {
using Value = eshkol_tagged_value_t;
static_assert(sizeof(Value) == 16 && offsetof(Value, data) == 8,
              "self-hosted planning requires the native 16-byte tagged ABI");
struct PlanningArena {
    std::unique_ptr<arena_t, decltype(&arena_destroy)> arena{arena_create(65536), arena_destroy};
    eshkol_memctx_t* context = eshkol_memctx_current();
    arena_t* previous = context->allocation_domain;
    PlanningArena() {
        if (!arena) throw std::bad_alloc();
        context->allocation_domain = arena.get();
        if (eshkol_current_arena() != arena.get()) {
            context->allocation_domain = previous;
            throw std::logic_error("incremental planning requires an isolated compiler memory context");
        }
    }
    ~PlanningArena() { context->allocation_domain = previous; }
};
Value integer(size_t n) { return eshkol_make_int64(static_cast<int64_t>(n), true); }
Value vector(arena_t* arena, const std::vector<Value>& values) {
    void* data = arena_allocate_vector_with_header(arena, values.size());
    if (!data) throw std::bad_alloc();
    *static_cast<uint64_t*>(data) = values.size();
    if (!values.empty()) std::memcpy(static_cast<char*>(data) + 8, values.data(), values.size() * sizeof(Value));
    Value value{}; value.type = ESHKOL_VALUE_HEAP_PTR;
    value.data.ptr_val = reinterpret_cast<uintptr_t>(data);
    return value;
}
size_t index(Value value, size_t bound) {
    if (value.type != ESHKOL_VALUE_INT64 || value.data.int_val < 0 ||
        static_cast<uint64_t>(value.data.int_val) >= bound)
        throw std::logic_error("self-hosted incremental plan contains an invalid integer");
    return static_cast<size_t>(value.data.int_val);
}
std::vector<Value> elements(Value value, size_t bound) {
    if (value.type != ESHKOL_VALUE_HEAP_PTR || !value.data.ptr_val)
        throw std::logic_error("self-hosted incremental planner refused the graph");
    auto* data = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(value.data.ptr_val));
    const auto* header = reinterpret_cast<const eshkol_object_header_t*>(data) - 1;
    const auto count = *data;
    if (header->subtype != HEAP_SUBTYPE_VECTOR || count > bound ||
        header->size != 8 + count * sizeof(Value))
        throw std::logic_error("self-hosted incremental plan contains an invalid vector");
    auto* begin = reinterpret_cast<const Value*>(data + 1);
    return {begin, begin + count};
}
}

static IncrementalPlan selfhost_plan(const DifferentiationGraph& graph) {
    using Op = DifferentiationOperation;
    const size_t count = graph.nodes.size(), width = graph.parameters.size();
    if (!width || width > DifferentiationGraph::max_inputs || count < width ||
        count > DifferentiationGraph::max_nodes || graph.output >= count)
        throw std::logic_error("invalid graph passed to IncrementalizePass");
    PlanningArena scope;
    std::vector<Value> nodes;
    for (const auto& node : graph.nodes) {
        const auto& rule = differentiation_rule(node.operation);
        if (!rule.incremental || node.inputs.size() != rule.arity)
            throw std::logic_error("invalid incremental graph operation or arity");
        std::vector<Value> operands;
        for (auto parent : node.inputs) operands.push_back(integer(parent));
        const size_t kind = node.operation == Op::Input ? 0 : node.operation == Op::Constant ? 1 : 2;
        nodes.push_back(vector(scope.arena.get(), {integer(kind), integer(node.input_index), vector(scope.arena.get(), operands)}));
    }
    auto input = vector(scope.arena.get(), {integer(1), integer(width), integer(graph.output), vector(scope.arena.get(), nodes)});
    Value output{};
    eshkol_incremental_plan_call(&input, &output);
    const auto fields = elements(output, 6);
    if (fields.size() != 6 || index(fields[0], 2) != 1)
        throw std::logic_error("self-hosted incremental plan has an unsupported schema");
    IncrementalPlan plan;
    plan.capacity = index(fields[1], 1025); plan.height = index(fields[2], 11);
    plan.edges = index(fields[3], 2049); plan.kernels = index(fields[4], 1025);
    const auto consumers = elements(fields[5], count);
    if (consumers.size() != count || plan.capacity < count ||
        plan.capacity != (size_t{1} << plan.height) ||
        (plan.height && plan.capacity / 2 >= count))
        throw std::logic_error("self-hosted incremental plan has an invalid shape");
    for (auto row : consumers) {
        std::vector<size_t> children;
        for (auto child : elements(row, count)) children.push_back(index(child, count));
        plan.consumers.push_back(std::move(children));
    }
    verify_incremental_plan(graph, plan);
    return plan;
}
void enable_self_hosted_incremental() {
    llvm::sys::DynamicLibrary::AddSymbol("__eshkol_compiler_incremental_state", reinterpret_cast<void*>(&__eshkol_compiler_incremental_state));
    llvm::sys::DynamicLibrary::AddSymbol("__eshkol_compiler_incremental_check", reinterpret_cast<void*>(&__eshkol_compiler_incremental_check));
    install_incremental_planner(selfhost_plan);
}
}
