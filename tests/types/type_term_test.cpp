/**
 * @file type_term_test.cpp
 * @brief The core term language of dependent types (ADR-0042): hash-consing
 *        and alpha-equivalence, normalization and definitional equality,
 *        unification with metavariables, binders, universes, and the bridge
 *        to TypeId and the gradual type relation.
 */
#include "eshkol/types/hott_types.h"
#include "eshkol/types/type_relation.h"
#include "eshkol/types/type_term.h"

#include <cstdio>
#include <string>

using namespace eshkol::hott;

static int g_failures = 0;
static int g_checks = 0;

static void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

static TermEq nominal(uint32_t a, uint32_t b) { return a == b ? TermEq::Equal : TermEq::Unequal; }

int main() {
    TypeEnvironment env;
    TermStore& t = env.terms();
    TypeRelation rel(env);

    const TermRef n = t.freeVar(t.freshStamp(), "n");
    const TermRef m = t.freeVar(t.freshStamp(), "m");
    const TermRef one = t.intLit(1);
    const TermRef real = env.termOf(BuiltinTypes::Float64);

    // ---- normalization: polynomial normal form of index arithmetic ----
    check(t.normalize(t.op("+", {t.intLit(2), t.intLit(3)})) == t.intLit(5), "2 + 3 normalizes to 5");
    check(t.normalize(t.op("+", {n, one})) == t.normalize(t.op("+", {one, n})), "n + 1 = 1 + n");
    check(t.normalize(t.op("*", {t.intLit(2), t.op("+", {n, one})})) ==
              t.normalize(t.op("+", {t.op("*", {t.intLit(2), n}), t.intLit(2)})),
          "2(n + 1) = 2n + 2");
    check(t.normalize(t.op("-", {n, n})) == t.intLit(0), "n - n = 0");
    check(t.normalize(t.op("*", {t.op("+", {n, m}), t.op("+", {n, m})})) ==
              t.normalize(t.op("+", {t.op("*", {n, n}), t.op("*", {t.intLit(2), n, m}),
                                     t.op("*", {m, m})})),
          "(n + m)^2 = n^2 + 2nm + m^2");
    check(t.definitionallyEqual(t.op("+", {n, one}), n) == TermEq::Unequal, "n + 1 != n");
    check(t.definitionallyEqual(n, m) == TermEq::Unequal, "distinct rigid variables differ");

    // ---- definitional equality on types ----
    const TermRef vec_n1 = t.app("vector", {real, t.op("+", {n, one})});
    const TermRef vec_1n = t.app("vector", {real, t.op("+", {one, n})});
    check(t.definitionallyEqual(vec_n1, vec_1n) == TermEq::Equal,
          "(vector real (+ n 1)) = (vector real (+ 1 n))");
    check(t.definitionallyEqual(t.app("vector", {real, t.intLit(2)}),
                                t.app("vector", {real, t.intLit(3)})) == TermEq::Unequal,
          "(vector real 2) != (vector real 3)");

    // ---- delta: a defined variable unfolds ----
    const uint64_t k_stamp = t.freshStamp();
    const TermRef k = t.freeVar(k_stamp, "k");
    t.defineFree(k_stamp, t.intLit(3));
    check(t.definitionallyEqual(t.app("vector", {real, k}), t.app("vector", {real, t.intLit(3)})) ==
              TermEq::Equal,
          "let k = 3: (vector real k) = (vector real 3)");

    // ---- binders: alpha-equivalence and instantiation ----
    const TermRef pi_a = t.pi({env.termOf(BuiltinTypes::Int64)}, t.app("vector", {real, t.bound(0)}), 0,
                              {"n"});
    const TermRef pi_b = t.pi({env.termOf(BuiltinTypes::Int64)}, t.app("vector", {real, t.bound(0)}), 0,
                              {"len"});
    check(pi_a == pi_b, "alpha-equivalent Pi types are one term");
    check(t.instantiate(t.node(pi_a).args.back(), {t.intLit(3)}) ==
              t.app("vector", {real, t.intLit(3)}),
          "(Pi ((n : integer)) (vector real n)) applied to 3 is (vector real 3)");
    check(t.mentionsBound(t.node(pi_a).args.back(), 0), "the Pi body depends on its binder");
    check(!t.hasLooseBound(pi_a), "a closed Pi has no loose binder");
    const uint64_t p_stamp = t.freshStamp();
    const TermRef p = t.freeVar(p_stamp, "p");
    check(t.abstractFrees(t.app("vector", {real, p}), {p_stamp}) ==
              t.app("vector", {real, t.bound(0)}),
          "closing over a program variable makes it the binder");

    // ---- unification with metavariables ----
    const TermRef meta = t.freshMeta();
    check(t.unify(t.op("+", {meta, one}), t.intLit(4), nominal) == TermEq::Equal &&
              t.normalize(meta) == t.intLit(3),
          "?m + 1 = 4 solves ?m = 3");
    const TermRef meta2 = t.freshMeta();
    check(t.unify(t.app("vector", {real, meta2}), t.app("vector", {real, t.intLit(7)}), nominal) ==
              TermEq::Equal && t.normalize(meta2) == t.intLit(7),
          "(vector real ?n) = (vector real 7) solves ?n = 7");
    const TermRef meta3 = t.freshMeta();
    check(t.unify(t.op("*", {t.intLit(2), meta3}), t.intLit(5), nominal) == TermEq::Unknown,
          "2 ?m = 5 has no integer solution and stays unknown");
    check(t.unify(t.app("vector", {real, t.intLit(2)}), t.app("vector", {real, t.intLit(3)}),
                  nominal) == TermEq::Unequal,
          "unification refuses (vector real 2) = (vector real 3)");

    // ---- universes ----
    const auto level = [&](uint32_t packed) -> uint32_t {
        return static_cast<uint8_t>(env.typeFromTerm(t.constant(packed)).level);
    };
    check(t.universeOf(t.universe(0), level) == 1u, "(universe 0) lives in (universe 1)");
    check(t.universeOf(t.universe(3), level) == 4u, "(universe 3) lives in (universe 4)");
    check(t.universeOf(real, level) == 0u, "real lives in (universe 0)");
    check(t.universeOf(t.app("vector", {real, t.intLit(3)}), level) == 0u,
          "(vector real 3) lives in the universe of its elements");
    const TermRef poly = t.pi({t.universe(0)}, t.bound(0), 0, {"A"});
    check(t.universeOf(poly, level) == 1u, "(Pi ((A : (universe 0))) A) lives in (universe 1)");
    check(!t.universeOf(t.intLit(3), level).has_value(), "an index is not a type");

    // ---- TypeId bridge ----
    const TypeId arrow = env.makeFunctionType({BuiltinTypes::Int64}, BuiltinTypes::Float64);
    const TermRef arrow_term =
        t.pi({env.termOf(BuiltinTypes::Int64)}, env.termOf(BuiltinTypes::Float64), 0);
    check(env.typeFromTerm(arrow_term) == arrow, "a non-dependent Pi is the nominal signature");
    check(env.termOf(arrow) == arrow_term, "termOf inverts typeFromTerm on a signature");
    const TypeId v3 = env.typeFromTerm(t.app("vector", {real, t.intLit(3)}));
    check(env.isTermType(v3), "(vector real 3) is a dependent type");
    check(env.typeFromTerm(t.app("vector", {real, t.op("+", {t.intLit(1), t.intLit(2)})})) == v3,
          "(vector real (+ 1 2)) is the same TypeId as (vector real 3)");
    check(env.areEquivalent(v3, env.typeFromTerm(t.app("vector", {real, t.intLit(3)}))),
          "areEquivalent decides definitional equality");
    check(env.erase(v3) == BuiltinTypes::Vector, "(vector real 3) erases to Vector");
    check(env.codegenReprOf(v3) == BuiltinTypes::Vector, "codegen reads the erased type");
    check(env.typeFromTerm(t.universe(0)) == BuiltinTypes::TypeU0, "(universe 0) is Type0");
    check(rel.print(v3) == "(vector Float64 3)", "a dependent type prints as its term");

    // ---- the gradual relation over dependent types ----
    const TypeId v2 = env.typeFromTerm(t.app("vector", {real, t.intLit(2)}));
    check(rel.isSubtype(v3, BuiltinTypes::Vector), "(vector real 3) <: Vector");
    check(!rel.isSubtype(BuiltinTypes::Vector, v3), "Vector is not statically a (vector real 3)");
    check(rel.isConsistentSubtype(BuiltinTypes::Vector, v3),
          "a Vector of unknown length is consistent with (vector real 3)");
    check(!rel.isConsistentSubtype(v2, v3), "(vector real 2) does not flow to (vector real 3)");
    check(!rel.isConsistentSubtype(BuiltinTypes::String, v3), "a String is not a vector");
    const TypeId vany3 = env.typeFromTerm(t.app("vector", {env.termOf(BuiltinTypes::Value), t.intLit(3)}));
    check(rel.isConsistentSubtype(vany3, v3), "(vector Value 3) flows to (vector real 3)");
    check(!rel.castable(v2, v3), "(the (vector real 3) v) of a (vector real 2) is a contradiction");
    check(rel.join(v2, v3) == BuiltinTypes::Vector, "two lengths join at Vector");

    if (g_failures == 0) {
        std::printf("PASS: dependent type terms (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("%d of %d checks failed\n", g_failures, g_checks);
    return 1;
}
