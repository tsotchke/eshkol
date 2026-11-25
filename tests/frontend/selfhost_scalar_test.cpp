/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/scalar_lowering.h>
#include <eshkol/frontend/incrementalize.h>
#include "../../lib/core/arena_memory.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
using namespace eshkol::frontend;
using Op = DifferentiationOperation;
using Kind = ScalarCoreKind;

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
        graph.nodes.back().origin = {static_cast<eshkol_node_id_t>(i + 1), 3, static_cast<uint32_t>(i + 7), 9};
    }
    graph.output = static_cast<uint32_t>(count - 1);
    return graph;
}
// Independent, direct text oracle uses the source rule table rather than the
// lowering's primitive encoding, and handles binding/read scopes separately.
static std::string oracle(const DifferentiationGraph& graph, size_t i, bool kernel) {
    const auto& node = graph.nodes[i];
    if (kernel && node.inputs.empty()) return "#f";
    if (node.operation == Op::Input) return "(vector-ref point " + std::to_string(node.input_index) + ")";
    if (node.operation == Op::Constant) return node.literal;
    std::string result = "(" + std::string(differentiation_rule(node.operation).spelling);
    for (auto parent : node.inputs)
        result += kernel ? " (__ig_get " + std::to_string(parent) + ")" : " slot" + std::to_string(parent);
    return result + ")";
}
static std::string render(const ScalarCoreTerm& term, const DifferentiationGraph& graph) {
    std::ostringstream out; write_scalar_core(out, term, graph, "point", "slot"); return out.str();
}
static uint64_t fingerprint(const std::string& text) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ull; }
    return hash;
}
static ScalarLowering refusing_pass(const DifferentiationGraph&) { throw std::logic_error("installed refusal"); }

int main(int argc, char** argv) {
#ifdef ESHKOL_EXPECT_SELFHOST
    enable_self_hosted_scalar_lowering();
    enable_self_hosted_scalar_lowering();
    const char* expected = "eshkol";
#else
    const char* expected = "seed";
#endif
    const bool emit = argc == 2 && std::strcmp(argv[1], "--emit") == 0;
    int checks = 0, failures = 0;
    auto check = [&](bool ok, const char* name) {
        ++checks; if (!ok) { ++failures; std::cerr << "FAIL: " << name << '\n'; }
    };
    check(std::strcmp(scalar_lowering_backend(), expected) == 0, "explicit bootstrap backend");
    auto* context = eshkol_memctx_current();
    auto* previous = context->allocation_domain;
    auto* sentinel = arena_create(4096);
    if (!sentinel) return 2;
    context->allocation_domain = sentinel;
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{64}, size_t{65}, size_t{256}, size_t{1024}}) {
        for (size_t width : {size_t{1}, std::min(n, size_t{2}), std::min(n, size_t{64})}) {
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                auto graph = example(n, width, seed);
                auto lowered = scalar_lower(graph);
                bool agrees = lowered.width == width && lowered.output == graph.output;
                std::ostringstream receipt;
                receipt << n << ':' << width << ':' << seed;
                for (size_t i = 0; i < n; ++i) {
                    const auto binding = render(lowered.bindings[i], graph), kernel = render(lowered.kernels[i], graph);
                    agrees = agrees && binding == oracle(graph, i, false) && kernel == oracle(graph, i, true);
                    receipt << '|' << binding << ':' << kernel;
                }
                check(agrees, "all ordered primitive, literal and scoped slot terms agree independently");
                check(context->allocation_domain == sentinel, "memory domain restored after lowering");
                check(scalar_lower(graph) == lowered, "owned terms survive compiler arena release");
                const auto numeric = graph.numeric_source(), incremental = IncrementalizePass::source(graph);
                if (emit) std::cout << receipt.str() << ':' << fingerprint(numeric) << ':' << fingerprint(incremental) << '\n';
            }
        }
    }
    auto good = example(7, 2, 3);
    auto refuses_graph = [&](DifferentiationGraph graph) {
        bool refused = false;
        try { (void)scalar_lower(graph); } catch (const std::logic_error&) { refused = true; }
        check(refused, "malformed source graph refused");
        check(context->allocation_domain == sentinel, "memory domain restored after refusal");
        check(scalar_lower(good).bindings.size() == good.nodes.size(), "lowering recovers after refusal");
    };
    auto bad = good; bad.output = bad.nodes.size(); refuses_graph(bad);
    bad = good; bad.parameters.clear(); refuses_graph(bad);
    bad = good; bad.parameters.resize(65); refuses_graph(bad);
    bad = good; bad.nodes[0].input_index = 1; refuses_graph(bad);
    bad = good; bad.nodes[2].inputs = {0}; refuses_graph(bad);
    bad = good; bad.nodes[2].inputs = {0, 2}; refuses_graph(bad);
    bad = good; bad.nodes[2] = {Op::Input, {}, {}, 2, {}}; refuses_graph(bad);
    bad = good; bad.nodes[2].operation = static_cast<Op>(99); refuses_graph(bad);
    bad = good; bad.nodes.resize(1025); refuses_graph(bad);
    const auto certified = scalar_lower(good);
    auto rejects = [&](ScalarLowering lowered) {
        bool refused = false;
        try { verify_scalar_lowering(good, lowered); } catch (const std::logic_error&) { refused = true; }
        check(refused, "corrupt lowering is refused at the pass boundary");
    };
    auto corrupt = certified; ++corrupt.width; rejects(corrupt);
    corrupt = certified; corrupt.output = 0; rejects(corrupt);
    corrupt = certified; corrupt.bindings.pop_back(); rejects(corrupt);
    corrupt = certified; corrupt.kernels.pop_back(); rejects(corrupt);
    corrupt = certified; corrupt.bindings[2].payload = (corrupt.bindings[2].payload + 1) % 4; rejects(corrupt);
    corrupt = certified; corrupt.bindings[2].arguments[0].payload = 6; rejects(corrupt);
    corrupt = certified; corrupt.bindings[2].arguments[0].kind = Kind::Read; rejects(corrupt);
    corrupt = certified; corrupt.kernels[2].arguments[0].kind = Kind::Binding; rejects(corrupt);
    corrupt = certified; corrupt.bindings[5].payload = 0; rejects(corrupt);
    corrupt = certified; corrupt.kernels[0].arguments.push_back(certified.bindings[0]); rejects(corrupt);
    corrupt = certified; corrupt.bindings[2].arguments.push_back(certified.bindings[0]); rejects(corrupt);
    corrupt = certified; corrupt.kernels[2].kind = Kind::Absent; rejects(corrupt);
    // Deliberately noncommutative graph catches operand reversal, including a
    // graph whose output is an earlier input and whose later nodes stay present.
    auto ordered = example(2, 2, 1);
    ordered.nodes.push_back({Op::Subtract, {1, 0}, {}, 0, {}});
    ordered.nodes.push_back({Op::Divide, {0, 1}, {}, 0, {}});
    ordered.output = 0;
    auto lower_ordered = scalar_lower(ordered);
    check(lower_ordered.output == 0 && lower_ordered.bindings.size() == 4, "unused nodes are retained in source order");
    bool reversed_refused = false;
    std::swap(lower_ordered.bindings[2].arguments[0], lower_ordered.bindings[2].arguments[1]);
    try { verify_scalar_lowering(ordered, lower_ordered); } catch (const std::logic_error&) { reversed_refused = true; }
    check(reversed_refused, "operand reversal cannot pass certification");
    for (const char* literal : {"-0.0", "0.0", "7/3", "10000000000000000000000001"}) {
        auto exact = example(1, 1, 1); exact.nodes.push_back({Op::Constant, {}, literal, 0, {17, 9, 7, 3}}); exact.output = 1;
        const auto lowered = scalar_lower(exact);
        check(render(lowered.bindings[1], exact) == literal && exact.nodes[1].origin.node_id == 17 &&
            exact.nodes[1].origin.line == 7, "literal text and original source ownership are retained");
    }
    context->allocation_domain = previous; arena_destroy(sentinel);
    std::atomic<int> thread_failures{0};
    auto worker = [&] {
        for (uint32_t i = 1; i <= 16; ++i) {
            auto graph = example(128, 4, i);
            auto* before = eshkol_memctx_current()->allocation_domain;
            if (scalar_lower(graph).bindings.size() != 128 || eshkol_memctx_current()->allocation_domain != before)
                ++thread_failures;
        }
    };
    std::thread a(worker), b(worker); a.join(); b.join();
    check(thread_failures.load() == 0, "compiler threads use independent temporary arenas");
    bool null_refused = false;
    try { install_scalar_lowerer(nullptr); } catch (const std::logic_error&) { null_refused = true; }
    check(null_refused, "null installation is refused");
#ifdef ESHKOL_EXPECT_SELFHOST
    bool replacement_refused = false;
    try { install_scalar_lowerer(refusing_pass); } catch (const std::logic_error&) { replacement_refused = true; }
    check(replacement_refused, "a selected compiler pass cannot be replaced");
#else
    install_scalar_lowerer(refusing_pass);
    bool propagated = false;
    try { (void)scalar_lower(good); } catch (const std::logic_error& error) { propagated = std::string(error.what()) == "installed refusal"; }
    check(propagated, "installed refusal is propagated without seed fallback");
#endif
    if (!emit) std::cout << "Backend: " << expected << " Passed: " << checks - failures << " Failed: " << failures
                         << "\nRESULT: " << (failures ? "FAIL" : "PASS") << '\n';
    return failures ? 1 : 0;
}
