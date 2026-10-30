/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/incrementalize.h>
#include "../../lib/core/arena_memory.h"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
using namespace eshkol::frontend;
using Op = DifferentiationOperation;

static DifferentiationGraph example(size_t count, size_t width, uint32_t seed) {
    DifferentiationGraph graph;
    for (size_t i = 0; i < width; ++i) {
        graph.parameters.push_back("x" + std::to_string(i));
        graph.nodes.push_back({Op::Input, {}, {}, static_cast<uint32_t>(i), {}});
    }
    auto next = [&] { seed = seed * 1664525u + 1013904223u; return seed; };
    while (graph.nodes.size() < count) {
        const auto i = graph.nodes.size();
        if (i % 5 == 0) graph.nodes.push_back({Op::Constant, {}, "7/3", 0, {}});
        else {
            const auto a = next() % i, b = i % 3 ? next() % i : a;
            const auto op = static_cast<Op>(static_cast<unsigned>(Op::Add) + next() % 4);
            graph.nodes.push_back({op, {static_cast<uint32_t>(a), static_cast<uint32_t>(b)}, {}, 0, {}});
        }
    }
    graph.output = static_cast<uint32_t>(count - 1);
    return graph;
}

static IncrementalPlan oracle(const DifferentiationGraph& graph) {
    IncrementalPlan plan;
    plan.consumers.resize(graph.nodes.size());
    while (plan.capacity < graph.nodes.size()) { plan.capacity *= 2; ++plan.height; }
    for (size_t child = 0; child < graph.nodes.size(); ++child) {
        const auto& operands = graph.nodes[child].inputs;
        plan.edges += operands.size();
        if (!operands.empty()) ++plan.kernels;
        for (size_t j = 0; j < operands.size(); ++j)
            if (!j || operands[j] != operands[0]) plan.consumers[operands[j]].push_back(child);
    }
    return plan;
}

int main(int argc, char** argv) {
#ifdef ESHKOL_EXPECT_SELFHOST
    enable_self_hosted_incremental();
    enable_self_hosted_incremental(); // Same explicit installation is idempotent.
    const char* expected_backend = "eshkol";
#else
    const char* expected_backend = "seed";
#endif
    const bool emit = argc == 2 && std::strcmp(argv[1], "--emit") == 0;
    int checks = 0, failures = 0;
    auto check = [&](bool ok, const char* name) {
        ++checks; if (!ok) { ++failures; std::cerr << "FAIL: " << name << '\n'; }
    };
    check(std::strcmp(incremental_plan_backend(), expected_backend) == 0, "selected bootstrap backend");
    auto* context = eshkol_memctx_current();
    auto* previous = context->allocation_domain;
    auto* sentinel = arena_create(4096);
    if (!sentinel) return 2;
    context->allocation_domain = sentinel;
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{64}, size_t{65}, size_t{256}, size_t{1024}}) {
        for (size_t width : {size_t{1}, std::min(n, size_t{2}), std::min(n, size_t{64})}) {
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                auto graph = example(n, width, seed);
                auto plan = incremental_plan(graph);
                check(plan == oracle(graph), "all consumer edges and plan metadata agree independently");
                check(context->allocation_domain == sentinel, "compiler memory context restored after success");
                if (emit) {
                    std::cout << n << ':' << width << ':' << seed << ':' << plan.capacity << ':' << plan.height
                              << ':' << plan.edges << ':' << plan.kernels;
                    for (const auto& row : plan.consumers) {
                        std::cout << '|'; for (auto child : row) std::cout << child << ',';
                    }
                    std::cout << '\n';
                }
                plan.consumers.clear();
                check(incremental_plan(graph) == oracle(graph), "plans own their data after arena release");
            }
        }
    }
    const auto good = example(7, 2, 3);
    auto refuses = [&](DifferentiationGraph graph) {
        bool refused = false;
        try { (void)incremental_plan(graph); } catch (const std::logic_error&) { refused = true; }
        check(refused, "malformed graph is refused");
        check(context->allocation_domain == sentinel, "compiler memory context restored after refusal");
        check(incremental_plan(good) == oracle(good), "successful planning after refusal");
    };
    auto bad = good; bad.output = bad.nodes.size(); refuses(bad);
    bad = good; bad.parameters.clear(); refuses(bad);
    bad = good; bad.parameters.resize(65); refuses(bad);
    bad = good; bad.nodes[0].input_index = 1; refuses(bad);
    bad = good; bad.nodes[2].inputs = {0}; refuses(bad);
    bad = good; bad.nodes[2].inputs = {0, 2}; refuses(bad);
    bad = good; bad.nodes[2] = {Op::Input, {}, {}, 2, {}}; refuses(bad);
    bad = good; bad.nodes.resize(1025); refuses(bad);
    auto rejects_plan = [&](IncrementalPlan plan) {
        bool refused = false;
        try { verify_incremental_plan(good, plan); } catch (const std::logic_error&) { refused = true; }
        check(refused, "corrupted planning certificate is refused");
    };
    auto corrupt = oracle(good); ++corrupt.edges; rejects_plan(corrupt);
    corrupt = oracle(good); ++corrupt.kernels; rejects_plan(corrupt);
    corrupt = oracle(good); corrupt.capacity *= 2; rejects_plan(corrupt);
    corrupt = oracle(good); corrupt.consumers.pop_back(); rejects_plan(corrupt);
    corrupt = oracle(good);
    auto row = std::find_if(corrupt.consumers.begin(), corrupt.consumers.end(), [](const auto& v) { return !v.empty(); });
    row->pop_back(); rejects_plan(corrupt);
    corrupt = oracle(good); corrupt.consumers[0].push_back(6); rejects_plan(corrupt);
    context->allocation_domain = previous;
    arena_destroy(sentinel);
    std::atomic<int> thread_failures{0};
    auto worker = [&] {
        for (uint32_t i = 1; i <= 16; ++i) {
            auto graph = example(128, 4, i);
            auto* before = eshkol_memctx_current()->allocation_domain;
            if (!(incremental_plan(graph) == oracle(graph)) ||
                eshkol_memctx_current()->allocation_domain != before) ++thread_failures;
        }
    };
    std::thread a(worker), b(worker); a.join(); b.join();
    check(thread_failures.load() == 0, "independent compiler-thread planning arenas");
    if (!emit) std::cout << "Backend: " << incremental_plan_backend() << " Passed: " << checks - failures
                         << " Failed: " << failures << "\nRESULT: " << (failures ? "FAIL" : "PASS") << '\n';
    return failures ? 1 : 0;
}
