/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Focused tests for the shared gradual type relation.
 */
#include <eshkol/types/type_relation.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace eshkol::hott;

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

int main() {
    TypeEnvironment env;
    TypeRelation relation(env);

    const TypeId accepts_number_returns_int =
        env.makeFunctionType({BuiltinTypes::Number}, BuiltinTypes::Int64);
    const TypeId accepts_int_returns_number =
        env.makeFunctionType({BuiltinTypes::Int64}, BuiltinTypes::Number);
    require(relation.isSubtype(accepts_number_returns_int, BuiltinTypes::Function),
            "a signature is a subtype of Function");
    require(relation.isSubtype(accepts_number_returns_int, BuiltinTypes::Closure),
            "a signature is a subtype of Closure");
    require(relation.isSubtype(accepts_number_returns_int, accepts_int_returns_number),
            "arrow parameters are contravariant and results covariant");
    require(!relation.isSubtype(accepts_int_returns_number, accepts_number_returns_int),
            "the reverse arrow variance is rejected");

    const TypeId gradual_arrow = env.makeFunctionType({BuiltinTypes::Value}, BuiltinTypes::Value);
    require(relation.isConsistentSubtype(gradual_arrow, accepts_number_returns_int),
            "Value remains consistent inside arrow domains and codomains");
    require(!relation.isConsistentSubtype(
                env.makeFunctionType({BuiltinTypes::String}, BuiltinTypes::String),
                accepts_int_returns_number),
            "concrete arrow mismatch remains rejected");
    require(relation.isConsistent(BuiltinTypes::Invalid, BuiltinTypes::String),
            "unresolved types remain gradual");

    const TypeId mixed = relation.join(BuiltinTypes::Int64, BuiltinTypes::String);
    require(mixed == BuiltinTypes::Value, "unrelated branch types join at Value");
    require(relation.join(BuiltinTypes::Never, BuiltinTypes::Int64) == BuiltinTypes::Int64,
            "Never is the join identity");
    require(relation.meet(BuiltinTypes::Int64, BuiltinTypes::String) == BuiltinTypes::Never,
            "disjoint concrete types meet at Never");
    require(relation.meet(BuiltinTypes::Value, BuiltinTypes::String) == BuiltinTypes::String,
            "Value is the meet identity");
    require(relation.narrow(BuiltinTypes::Number, BuiltinTypes::Int64) == BuiltinTypes::Int64,
            "successful narrowing uses the intersection");
    require(relation.narrow(BuiltinTypes::String, BuiltinTypes::Number) == BuiltinTypes::Number,
            "a successful runtime test outranks an empty static intersection");

    const TypeId pair_int_string = env.makePairType(BuiltinTypes::Int64, BuiltinTypes::String);
    const TypeId pair_number_text = env.makePairType(BuiltinTypes::Number, BuiltinTypes::Text);
    require(relation.isSubtype(pair_int_string, pair_number_text), "pair components are covariant");
    require(relation.print(accepts_number_returns_int) == "(-> Number Int64)",
            "signatures print through the relation");
    require(relation.print(env.makePairType(BuiltinTypes::Value, BuiltinTypes::Value)) == "Pair",
            "a fully dynamic pair prints with its canonical generic name");
    require(relation.print(BuiltinTypes::Function) == "Function",
            "the generic procedure type has a readable name");

    const auto refused = relation.widen(BuiltinTypes::Int64, BuiltinTypes::String,
                                        WidenPolicy::InferenceSlot);
    require(refused.conflict && !refused.changed && refused.type == BuiltinTypes::Int64,
            "inference slots report a top join as a conflict");
    const auto do_widen = relation.widen(BuiltinTypes::Int64, BuiltinTypes::String,
                                         WidenPolicy::AdoptTop);
    require(do_widen.changed && do_widen.type == BuiltinTypes::Value,
            "do slots adopt a top join");

    std::cout << "PASS: TypeRelation judgments, lattice, printing, and widening\n";
    return 0;
}
