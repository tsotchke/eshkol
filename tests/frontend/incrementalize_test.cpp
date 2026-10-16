/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#include <eshkol/frontend/incrementalize.h>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace eshkol::frontend;
int main(int argc, char** argv) {
    using Op = DifferentiationOperation;
    DifferentiationGraph graph;
    graph.parameters = {"x", "y"};
    graph.nodes = {{Op::Input, {}, {}, 0, {}}, {Op::Input, {}, {}, 1, {}},
                   {Op::Multiply, {0, 1}, {}, 0, {}}, {Op::Multiply, {0, 0}, {}, 0, {}},
                   {Op::Add, {2, 3}, {}, 0, {}}};
    graph.output = 4;
    const auto source = IncrementalizePass::source(graph);
    if (argc == 2 && std::strcmp(argv[1], "--emit") == 0) { std::cout << source << '\n'; return 0; }
    int failures = 0, checks = 0;
    auto check = [&](bool ok, const char* name) { ++checks; if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", name); } };
    check(source.find("(lambda (__ig_get) (* (__ig_get 0) (__ig_get 1)))") != std::string::npos,
          "native primitive kernel with compiled parent indices");
    check(source.find("(vector 5 6 2 3)") != std::string::npos, "graph shape receipt");
    check(source.find("(vector 2 3 )") != std::string::npos, "deduplicated repeated-edge consumers");
    check(source.find("eqv? old value") != std::string::npos, "change detection preserves exactness and floating signed zero");
    check(source.find("__ig_minimum") != std::string::npos, "ordered sparse work queue");
    check(source.find("(gradient") == std::string::npos && source.find("eval") == std::string::npos,
          "incremental program contains neither AD invocation nor source interpreter");
    auto invalid = graph; invalid.nodes[2].inputs[0] = 4;
    bool refused = false;
    try { (void)IncrementalizePass::source(invalid); } catch (const std::logic_error&) { refused = true; }
    check(refused, "non-topological graph refused");
    invalid = graph; invalid.nodes[2].inputs.clear(); refused = false;
    try { (void)IncrementalizePass::source(invalid); } catch (const std::logic_error&) { refused = true; }
    check(refused, "malformed operation arity refused");
    invalid = graph; invalid.nodes[0].input_index = 1; refused = false;
    try { (void)IncrementalizePass::source(invalid); } catch (const std::logic_error&) { refused = true; }
    check(refused, "malformed input mapping refused");
    std::printf("Passed: %d Failed: %d\nRESULT: %s\n", checks - failures, failures, failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
