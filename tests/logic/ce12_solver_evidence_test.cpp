/*
 * CDCL proof evidence, ASP enumeration, and NARS premise order
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/asp_solver.h>
#include <eshkol/core/cdcl_sat.h>
#include <eshkol/core/nars_truth.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <set>
#include <vector>

namespace {

struct AllocationControl {
    unsigned malloc_calls = 0;
    unsigned realloc_calls = 0;
    unsigned fail_malloc_at = 0;
    unsigned fail_realloc_at = 0;
};

AllocationControl allocations;
unsigned program_index = 0;

void require(bool condition, const char* expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL at line %d (program %u): %s\n",
                     line, program_index, expression);
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(expr) require((expr), #expr, __LINE__)

using CdclSolver = std::unique_ptr<eshkol_cdcl_solver_t, decltype(&eshkol_cdcl_destroy)>;
using AspSolver = std::unique_ptr<eshkol_asp_solver_t, decltype(&eshkol_asp_destroy)>;

struct Cnf {
    uint32_t vars;
    std::vector<std::vector<int32_t>> clauses;

    std::vector<eshkol_cnf_clause_t> views() const {
        std::vector<eshkol_cnf_clause_t> result;
        for (const auto& clause : clauses) {
            result.push_back({clause.data(), static_cast<uint32_t>(clause.size())});
        }
        return result;
    }

    CdclSolver solver(bool emit_drat = true) const {
        auto config = eshkol_cdcl_default_config();
        config.emit_drat = emit_drat;
        CdclSolver sat(eshkol_cdcl_create(vars, &config), eshkol_cdcl_destroy);
        CHECK(sat != nullptr);
        for (const auto& clause : views()) {
            CHECK(eshkol_cdcl_add_clause(sat.get(), clause.lits, clause.num_lits));
        }
        return sat;
    }
};

Cnf pigeonhole(int32_t pigeons, int32_t holes) {
    Cnf formula{static_cast<uint32_t>(pigeons * holes), {}};
    for (int32_t p = 0; p < pigeons; ++p) {
        std::vector<int32_t> occupied;
        for (int32_t h = 1; h <= holes; ++h) occupied.push_back(p * holes + h);
        formula.clauses.push_back(occupied);
    }
    for (int32_t h = 1; h <= holes; ++h) {
        for (int32_t p = 0; p < pigeons; ++p) {
            for (int32_t q = p + 1; q < pigeons; ++q) {
                formula.clauses.push_back({-(p * holes + h), -(q * holes + h)});
            }
        }
    }
    return formula;
}

void check_cnf_pair(const Cnf& unsat, size_t omitted_clause) {
    auto solver = unsat.solver();
    CHECK(eshkol_cdcl_solve(solver.get()) == ESHKOL_SAT_UNSATISFIABLE);
    eshkol_cdcl_stats_t stats{};
    eshkol_cdcl_get_stats(solver.get(), &stats);
    CHECK(stats.evidence_status == ESHKOL_EVIDENCE_CERTIFIED);
    CHECK(!stats.drat_alloc_failed);
    const auto* proof = eshkol_cdcl_get_proof(solver.get());
    CHECK(proof->num_steps > 1 && stats.drat_steps == proof->num_steps);
    const auto clauses = unsat.views();
    CHECK(eshkol_cdcl_verify_drat(unsat.vars, clauses.data(), clauses.size(), proof));

    Cnf sat = unsat;
    sat.clauses.erase(sat.clauses.begin() + omitted_clause);
    const auto weakened = sat.views();
    auto sat_solver = sat.solver();
    CHECK(eshkol_cdcl_solve(sat_solver.get()) == ESHKOL_SAT_SATISFIABLE);
    eshkol_cdcl_get_stats(sat_solver.get(), &stats);
    CHECK(stats.evidence_status == ESHKOL_EVIDENCE_NONE && !stats.drat_alloc_failed);
    std::vector<int8_t> model(sat.vars + 1);
    CHECK(eshkol_cdcl_copy_model(sat_solver.get(), model.data(), model.size()));
    CHECK(eshkol_cdcl_verify_model(sat.vars, weakened.data(), weakened.size(),
                                   model.data(), model.size()));
    CHECK(!eshkol_cdcl_verify_model(unsat.vars, clauses.data(), clauses.size(),
                                    model.data(), model.size()));
    CHECK(!eshkol_cdcl_verify_drat(sat.vars, weakened.data(), weakened.size(), proof));
}

void test_cdcl_evidence() {
    check_cnf_pair({2, {{1, 2}, {-1, 2}, {1, -2}, {-1, -2}}}, 3);
    Cnf three_sat{3, {}};
    for (unsigned signs = 0; signs < 8; ++signs) {
        three_sat.clauses.push_back({signs & 1 ? -1 : 1, signs & 2 ? -2 : 2,
                                    signs & 4 ? -3 : 3});
    }
    check_cnf_pair(three_sat, 7);
    check_cnf_pair(pigeonhole(4, 3), 3);
    check_cnf_pair(pigeonhole(5, 4), 4);

    const Cnf units{1, {{1}, {-1}}};
    for (bool emit_drat : {false, true}) {
        auto solver = units.solver(emit_drat);
        CHECK(eshkol_cdcl_solve(solver.get()) == ESHKOL_SAT_UNSATISFIABLE);
        eshkol_cdcl_stats_t stats{};
        eshkol_cdcl_get_stats(solver.get(), &stats);
        CHECK(!stats.drat_alloc_failed);
        CHECK(stats.evidence_status == (emit_drat ? ESHKOL_EVIDENCE_CERTIFIED
                                                  : ESHKOL_EVIDENCE_UNCERTIFIED));
        CHECK(stats.drat_steps == (emit_drat ? 1u : 0u));
        const auto clauses = units.views();
        CHECK(eshkol_cdcl_verify_drat(units.vars, clauses.data(), clauses.size(),
                                     eshkol_cdcl_get_proof(solver.get())) == emit_drat);
    }
}

void check_allocation_failure(const Cnf& formula, unsigned malloc_at, unsigned realloc_at,
                              const eshkol_drat_proof_t* baseline = nullptr) {
    auto solver = formula.solver();
    allocations = {0, 0, malloc_at, realloc_at};

    CHECK(eshkol_cdcl_solve(solver.get()) == ESHKOL_SAT_UNSATISFIABLE);
    if (malloc_at) CHECK(allocations.malloc_calls == malloc_at);
    if (realloc_at) CHECK(allocations.realloc_calls == realloc_at);
    allocations = {};

    const auto* proof = eshkol_cdcl_get_proof(solver.get());
    const auto clauses = formula.views();
    const uint32_t recorded = proof->num_steps;
    if (malloc_at) CHECK(recorded == malloc_at - 1);
    if (realloc_at == 1) CHECK(recorded == 0 && proof->steps == nullptr);
    if (realloc_at > 1) CHECK(recorded == proof->capacity && recorded > 0);
    for (uint32_t i = 0; i < recorded; ++i) {
        CHECK(baseline != nullptr && i < baseline->num_steps);
        const auto& step = proof->steps[i];
        const auto& original = baseline->steps[i];
        CHECK(step.num_lits > 0 && step.lits != nullptr);
        CHECK(step.op == original.op && step.num_lits == original.num_lits);
        for (uint32_t j = 0; j < step.num_lits; ++j) CHECK(step.lits[j] == original.lits[j]);
    }
    CHECK(!eshkol_cdcl_verify_drat(formula.vars, clauses.data(), clauses.size(), proof));

    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        eshkol_cdcl_stats_t stats{};
        eshkol_cdcl_get_stats(solver.get(), &stats);
        CHECK(stats.evidence_status == ESHKOL_EVIDENCE_INCOMPLETE_ALLOC);
        CHECK(stats.drat_alloc_failed && stats.drat_steps == recorded);
        CHECK(proof->num_steps == recorded);
        CHECK(eshkol_cdcl_solve(solver.get()) == ESHKOL_SAT_UNSATISFIABLE);
    }
}

void test_drat_allocation_failure() {
    check_allocation_failure({1, {{1}, {-1}}}, 0, 1);

    const auto formula = pigeonhole(6, 5);
    auto solver = formula.solver();
    allocations = {};
    CHECK(eshkol_cdcl_solve(solver.get()) == ESHKOL_SAT_UNSATISFIABLE);
    const unsigned malloc_calls = allocations.malloc_calls;
    const unsigned realloc_calls = allocations.realloc_calls;
    CHECK(malloc_calls > 1 && realloc_calls > 1);
    const auto* proof = eshkol_cdcl_get_proof(solver.get());
    const auto clauses = formula.views();
    CHECK(eshkol_cdcl_verify_drat(formula.vars, clauses.data(), clauses.size(), proof));
    for (unsigned i = 1; i <= malloc_calls; ++i) check_allocation_failure(formula, i, 0, proof);
    for (unsigned i = 1; i <= realloc_calls; ++i) check_allocation_failure(formula, 0, i, proof);
    std::printf("PASS DRAT allocation signalling (%u injected failures)\n",
                1 + malloc_calls + realloc_calls);
}

AspSolver make_asp(uint32_t atoms, const eshkol_asp_config_t* config = nullptr) {
    AspSolver solver(eshkol_asp_create(atoms, config), eshkol_asp_destroy);
    CHECK(solver != nullptr);
    return solver;
}

std::set<uint32_t> model_masks(const AspSolver& solver, uint32_t atoms,
                               const eshkol_asp_stats_t& stats) {
    CHECK(eshkol_asp_num_models(solver.get()) == stats.num_models);
    std::set<uint32_t> result;
    for (uint32_t i = 0; i < stats.num_models; ++i) {
        const auto* model = eshkol_asp_get_model(solver.get(), i);
        CHECK(model != nullptr);
        CHECK(eshkol_asp_verify_stable_model(solver.get(), model->true_atoms,
                                             model->num_true_atoms));
        uint32_t mask = 0;
        uint32_t previous = 0;
        for (uint32_t j = 0; j < model->num_true_atoms; ++j) {
            const uint32_t atom = model->true_atoms[j];
            CHECK(atom > previous && atom <= atoms);
            mask |= 1u << (atom - 1);
            previous = atom;
        }
        CHECK(result.insert(mask).second);
    }
    CHECK(eshkol_asp_get_model(solver.get(), stats.num_models) == nullptr);
    return result;
}

void test_asp_round_limits() {
    auto config = eshkol_asp_default_config();
    config.max_cegar_rounds = 100;
    auto many = make_asp(14, &config);
    const uint32_t heads[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    CHECK(eshkol_asp_add_choice_rule(many.get(), heads, 14, 0, UINT32_MAX,
                                     nullptr, 0, nullptr, 0));
    eshkol_asp_stats_t stats{};
    CHECK(eshkol_asp_solve(many.get(), &stats) == ESHKOL_ASP_PARTIAL);
    CHECK(stats.sat_calls == 100 && stats.evidence_status == ESHKOL_EVIDENCE_NONE);
    CHECK(model_masks(many, 14, stats).size() == 100);

    for (uint32_t rounds : {1u, 10u}) {
        config.max_cegar_rounds = rounds;
        auto unique = make_asp(5, &config);
        const uint32_t a4 = 4;
        const uint32_t a5 = 5;
        const uint32_t absent[]{5, 1};
        const uint32_t loop_body[]{5, 2};
        // The first completion candidate contains the unsupported loop through a5.
        CHECK(eshkol_asp_add_normal_rule(unique.get(), 2, nullptr, 0, nullptr, 0));
        CHECK(eshkol_asp_add_normal_rule(unique.get(), 1, &a4, 1, absent, 2));
        CHECK(eshkol_asp_add_normal_rule(unique.get(), 1, nullptr, 0, &a5, 1));
        CHECK(eshkol_asp_add_choice_rule(unique.get(), heads, 3, 1, 3, &a5, 1, nullptr, 0));
        CHECK(eshkol_asp_add_normal_rule(unique.get(), a5, loop_body, 2, nullptr, 0));
        const auto status = eshkol_asp_solve(unique.get(), &stats);
        const auto models = model_masks(unique, 5, stats);
        CHECK(stats.evidence_status == ESHKOL_EVIDENCE_NONE);
        if (rounds == 1) {
            CHECK(status == ESHKOL_ASP_BUDGET_EXCEEDED && models.empty());
            CHECK(stats.sat_calls == 1 && stats.loop_formulas_added > 0);
        } else {
            CHECK(status == ESHKOL_ASP_SAT && models == std::set<uint32_t>{3});
        }
    }

    config.max_cegar_rounds = 1;
    auto contradictory = make_asp(1, &config);
    CHECK(eshkol_asp_add_constraint(contradictory.get(), nullptr, 0, nullptr, 0));
    CHECK(eshkol_asp_solve(contradictory.get(), &stats) == ESHKOL_ASP_UNSAT);
    CHECK(stats.sat_calls == 1 && stats.evidence_status == ESHKOL_EVIDENCE_UNCERTIFIED);

    config.max_models = 1;
    auto first = make_asp(2, &config);
    CHECK(eshkol_asp_add_choice_rule(first.get(), heads, 2, 0, 2, nullptr, 0, nullptr, 0));
    CHECK(eshkol_asp_solve(first.get(), &stats) == ESHKOL_ASP_SAT);
    CHECK(stats.sat_calls == 1 && model_masks(first, 2, stats).size() == 1);
}

void test_asp_conflict_limits() {
    auto config = eshkol_asp_default_config();
    config.max_sat_conflicts = 1;
    const uint32_t heads[]{1, 2};
    for (bool allow_empty : {false, true}) {
        auto solver = make_asp(2, &config);
        CHECK(eshkol_asp_add_choice_rule(solver.get(), heads, 2, 0, 2,
                                         nullptr, 0, nullptr, 0));
        CHECK(eshkol_asp_add_constraint(solver.get(), heads, 2, nullptr, 0));
        CHECK(eshkol_asp_add_constraint(solver.get(), heads, 1, heads + 1, 1));
        CHECK(eshkol_asp_add_constraint(solver.get(), heads + 1, 1, heads, 1));
        if (!allow_empty) {
            CHECK(eshkol_asp_add_constraint(solver.get(), nullptr, 0, heads, 2));
        }
        const auto status = eshkol_asp_solve(solver.get(), nullptr);
        CHECK(status == (allow_empty ? ESHKOL_ASP_PARTIAL : ESHKOL_ASP_BUDGET_EXCEEDED));
        eshkol_asp_stats_t stats{};
        CHECK(eshkol_asp_solve(solver.get(), &stats) == status);
        CHECK(stats.total_conflicts == 1 && stats.evidence_status == ESHKOL_EVIDENCE_NONE);
        const auto models = model_masks(solver, 2, stats);
        if (allow_empty) {
            CHECK(models == std::set<uint32_t>{0});
        } else {
            CHECK(models.empty());
        }
    }
}

enum class RuleKind { Normal, Choice, Constraint };

struct Rule {
    RuleKind kind;
    uint32_t heads;
    uint32_t positive;
    uint32_t negative;
    uint32_t minimum = 0;
    uint32_t maximum = 0;
};

std::vector<uint32_t> atom_ids(uint32_t mask) {
    std::vector<uint32_t> result;
    for (uint32_t atom = 1; mask; ++atom, mask >>= 1) {
        if (mask & 1u) result.push_back(atom);
    }
    return result;
}

void add_rule(const AspSolver& solver, const Rule& rule) {
    const auto heads = atom_ids(rule.heads);
    const auto pos = atom_ids(rule.positive);
    const auto neg = atom_ids(rule.negative);
    switch (rule.kind) {
    case RuleKind::Normal:
        CHECK(eshkol_asp_add_normal_rule(solver.get(), heads.front(), pos.data(), pos.size(),
                                         neg.data(), neg.size()));
        break;
    case RuleKind::Choice:
        CHECK(eshkol_asp_add_choice_rule(solver.get(), heads.data(), heads.size(),
                                         rule.minimum, rule.maximum, pos.data(), pos.size(),
                                         neg.data(), neg.size()));
        break;
    case RuleKind::Constraint:
        CHECK(eshkol_asp_add_constraint(solver.get(), pos.data(), pos.size(),
                                        neg.data(), neg.size()));
        break;
    }
}

bool stable_by_reduct(const std::vector<Rule>& program, uint32_t candidate) {
    // Each reduct rule is a positive implication from its body to its head set.
    std::vector<Rule> reduct;
    for (const Rule& rule : program) {
        if (rule.negative & candidate) continue;
        if (rule.kind == RuleKind::Choice) {
            const uint32_t selected = rule.heads & candidate;
            if ((candidate & rule.positive) == rule.positive) {
                const size_t count = atom_ids(selected).size();
                if (count < rule.minimum || count > rule.maximum) return false;
            }
            reduct.push_back({RuleKind::Normal, selected, rule.positive, 0});
        } else {
            reduct.push_back({rule.kind, rule.heads, rule.positive, 0});
        }
    }
    const auto models_reduct = [&](uint32_t atoms) {
        for (const Rule& rule : reduct) {
            if ((atoms & rule.positive) != rule.positive) continue;
            if (rule.kind == RuleKind::Constraint || (atoms & rule.heads) != rule.heads) {
                return false;
            }
        }
        return true;
    };
    if (!models_reduct(candidate)) return false;
    // Check minimality by subset enumeration independently of closure propagation.
    for (uint32_t subset = candidate; subset != 0;) {
        subset = (subset - 1) & candidate;
        if (models_reduct(subset)) return false;
    }
    return true;
}

void test_asp_against_reduct() {
    constexpr uint32_t atoms = 5;
    constexpr uint32_t assignments = 1u << atoms;
    constexpr unsigned programs = 512;
    std::mt19937 rng(959);
    const auto random_atom = [&] { return 1u << (rng() % atoms); };
    unsigned with_models = 0;
    unsigned with_multiple = 0;
    unsigned with_loops = 0;
    for (program_index = 1; program_index <= programs; ++program_index) {
        auto solver = make_asp(atoms);
        std::vector<Rule> program;
        const auto num_rules = 1 + rng() % 12;
        for (unsigned r = 0; r < num_rules; ++r) {
            const auto kind = static_cast<RuleKind>(rng() % 3);
            uint32_t positive = 0;
            uint32_t negative = 0;
            for (auto count = rng() % 3; count > 0; --count) positive |= random_atom();
            for (auto count = rng() % 3; count > 0; --count) negative |= random_atom();
            Rule rule{kind, 0, positive, negative};
            if (kind == RuleKind::Normal) {
                rule.heads = random_atom();
            } else if (kind == RuleKind::Choice) {
                rule.heads = 1 + rng() % (assignments - 1);
                const auto heads = atom_ids(rule.heads).size();
                rule.minimum = rng() % (heads + 1);
                rule.maximum = rule.minimum + rng() % (heads - rule.minimum + 1);
            }
            program.push_back(rule);
            add_rule(solver, rule);
        }

        std::set<uint32_t> expected;
        for (uint32_t candidate = 0; candidate < assignments; ++candidate) {
            const bool stable = stable_by_reduct(program, candidate);
            if (stable) expected.insert(candidate);
            const auto ids = atom_ids(candidate);
            CHECK(eshkol_asp_verify_stable_model(solver.get(), ids.data(), ids.size()) == stable);
        }
        eshkol_asp_stats_t stats{};
        CHECK(eshkol_asp_solve(solver.get(), &stats) ==
              (expected.empty() ? ESHKOL_ASP_UNSAT : ESHKOL_ASP_SAT));
        CHECK(stats.evidence_status == (expected.empty() ? ESHKOL_EVIDENCE_UNCERTIFIED
                                                        : ESHKOL_EVIDENCE_NONE));
        CHECK(model_masks(solver, atoms, stats) == expected);
        with_models += !expected.empty();
        with_multiple += expected.size() > 1;
        with_loops += stats.loop_formulas_added > 0;
    }
    program_index = 0;
    CHECK(with_models > 0 && with_models < programs && with_multiple > 0 && with_loops > 0);
    std::printf("PASS ASP reduct comparison (%u programs)\n", programs);
}

void test_nars_premise_order() {
    const eshkol_nars_truth_t m_to_p{0.2, 0.6};
    const eshkol_nars_truth_t s_to_p_or_m_to_s{0.8, 0.7};
    const double k = 2.0;
    const double abduction_weight = 0.084;
    const double induction_weight = 0.336;

    eshkol_nars_truth_t result{};
    CHECK(eshkol_nars_abduction(m_to_p, s_to_p_or_m_to_s, k, &result));
    CHECK(result.frequency == 0.8);
    CHECK(std::abs(result.confidence - abduction_weight / (abduction_weight + k)) < 1e-12);
    CHECK(eshkol_nars_abduction(s_to_p_or_m_to_s, m_to_p, k, &result));
    CHECK(result.frequency == 0.2);
    CHECK(std::abs(result.confidence - induction_weight / (induction_weight + k)) < 1e-12);

    CHECK(eshkol_nars_induction(m_to_p, s_to_p_or_m_to_s, k, &result));
    CHECK(result.frequency == 0.2);
    CHECK(std::abs(result.confidence - induction_weight / (induction_weight + k)) < 1e-12);
    CHECK(eshkol_nars_induction(s_to_p_or_m_to_s, m_to_p, k, &result));
    CHECK(result.frequency == 0.8);
    CHECK(std::abs(result.confidence - abduction_weight / (abduction_weight + k)) < 1e-12);
}

} // namespace

void* eshkol_cdcl_test_malloc(size_t size) {
    if (++allocations.malloc_calls == allocations.fail_malloc_at) return nullptr;
    return std::malloc(size);
}

void* eshkol_cdcl_test_realloc(void* ptr, size_t size) {
    if (++allocations.realloc_calls == allocations.fail_realloc_at) return nullptr;
    return std::realloc(ptr, size);
}

int main() {
    test_cdcl_evidence();
    std::puts("PASS CDCL models, DRAT proofs, and weakened formulas");
    test_drat_allocation_failure();
    test_asp_round_limits();
    test_asp_conflict_limits();
    std::puts("PASS ASP round and conflict budgets");
    test_asp_against_reduct();
    test_nars_premise_order();
    std::puts("PASS NARS premise order");
    return EXIT_SUCCESS;
}
