/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * HoTT Type System Unit Tests
 */

#include <eshkol/types/hott_types.h>
#include <eshkol/eshkol.h>
#include <iostream>
#include <cassert>

using namespace eshkol::hott;

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;
static bool current_test_failed = false;
static const char* current_test_error = nullptr;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    current_test_failed = false; \
    current_test_error = nullptr; \
    test_##name(); \
    if (!current_test_failed) { \
        std::cout << "PASS" << std::endl; \
        tests_passed++; \
    } else { \
        std::cout << "FAIL: " << (current_test_error ? current_test_error : "unknown") << std::endl; \
        tests_failed++; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { current_test_failed = true; current_test_error = "Assertion failed: " #cond; return; } \
} while(0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

// ============================================================================
// TESTS
// ============================================================================

TEST(builtin_types_exist) {
    TypeEnvironment env;

    // Check that all builtin types can be looked up
    ASSERT_TRUE(env.lookupType("Int64").has_value());
    ASSERT_TRUE(env.lookupType("Float64").has_value());
    ASSERT_TRUE(env.lookupType("String").has_value());
    ASSERT_TRUE(env.lookupType("Boolean").has_value());
    ASSERT_TRUE(env.lookupType("List").has_value());
    ASSERT_TRUE(env.lookupType("Vector").has_value());
    ASSERT_TRUE(env.lookupType("Null").has_value());
}

TEST(type_aliases) {
    TypeEnvironment env;

    // Check common aliases
    ASSERT_EQ(env.lookupType("int"), env.lookupType("Int64"));
    ASSERT_EQ(env.lookupType("integer"), env.lookupType("Int64"));
    ASSERT_EQ(env.lookupType("float"), env.lookupType("Float64"));
    ASSERT_EQ(env.lookupType("double"), env.lookupType("Float64"));
    ASSERT_EQ(env.lookupType("string"), env.lookupType("String"));
    ASSERT_EQ(env.lookupType("bool"), env.lookupType("Boolean"));
}

TEST(numeric_tower_subtyping) {
    TypeEnvironment env;

    // Int64 <: Integer <: Number <: Value
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Int64, BuiltinTypes::Integer));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Int64, BuiltinTypes::Number));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Int64, BuiltinTypes::Value));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Integer, BuiltinTypes::Number));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Integer, BuiltinTypes::Value));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Number, BuiltinTypes::Value));

    // Float64 <: Real <: Number <: Value
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Float64, BuiltinTypes::Real));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Float64, BuiltinTypes::Number));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Float64, BuiltinTypes::Value));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Real, BuiltinTypes::Number));

    // Not subtypes
    ASSERT_FALSE(env.isSubtype(BuiltinTypes::Int64, BuiltinTypes::Float64));
    ASSERT_FALSE(env.isSubtype(BuiltinTypes::Float64, BuiltinTypes::Int64));
    ASSERT_FALSE(env.isSubtype(BuiltinTypes::Integer, BuiltinTypes::Real));
    ASSERT_FALSE(env.isSubtype(BuiltinTypes::Real, BuiltinTypes::Integer));
}

TEST(text_subtyping) {
    TypeEnvironment env;

    // String <: Text <: Value
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::String, BuiltinTypes::Text));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::String, BuiltinTypes::Value));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Text, BuiltinTypes::Value));

    // Char <: Text <: Value
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Char, BuiltinTypes::Text));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Char, BuiltinTypes::Value));
}

TEST(reflexivity) {
    TypeEnvironment env;

    // Every type is a subtype of itself
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Int64, BuiltinTypes::Int64));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Float64, BuiltinTypes::Float64));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::String, BuiltinTypes::String));
    ASSERT_TRUE(env.isSubtype(BuiltinTypes::Value, BuiltinTypes::Value));
}

TEST(least_common_supertype) {
    TypeEnvironment env;

    // LCS of Int64 and Float64 should be Number
    auto lcs1 = env.leastCommonSupertype(BuiltinTypes::Int64, BuiltinTypes::Float64);
    ASSERT_TRUE(lcs1.has_value());
    ASSERT_EQ(lcs1->id, BuiltinTypes::Number.id);

    // LCS of String and Char should be Text
    auto lcs2 = env.leastCommonSupertype(BuiltinTypes::String, BuiltinTypes::Char);
    ASSERT_TRUE(lcs2.has_value());
    ASSERT_EQ(lcs2->id, BuiltinTypes::Text.id);

    // LCS of Int64 and String should be Value
    auto lcs3 = env.leastCommonSupertype(BuiltinTypes::Int64, BuiltinTypes::String);
    ASSERT_TRUE(lcs3.has_value());
    ASSERT_EQ(lcs3->id, BuiltinTypes::Value.id);

    // LCS of same type is that type
    auto lcs4 = env.leastCommonSupertype(BuiltinTypes::Int64, BuiltinTypes::Int64);
    ASSERT_TRUE(lcs4.has_value());
    ASSERT_EQ(lcs4->id, BuiltinTypes::Int64.id);
}

TEST(arithmetic_promotion) {
    TypeEnvironment env;

    // Int64 + Int64 -> Int64
    auto p1 = env.promoteForArithmetic(BuiltinTypes::Int64, BuiltinTypes::Int64);
    ASSERT_EQ(p1.id, BuiltinTypes::Int64.id);

    // Float64 + Float64 -> Float64
    auto p2 = env.promoteForArithmetic(BuiltinTypes::Float64, BuiltinTypes::Float64);
    ASSERT_EQ(p2.id, BuiltinTypes::Float64.id);

    // Int64 + Float64 -> Float64 (promotion)
    auto p3 = env.promoteForArithmetic(BuiltinTypes::Int64, BuiltinTypes::Float64);
    ASSERT_EQ(p3.id, BuiltinTypes::Float64.id);

    // Float64 + Int64 -> Float64 (symmetric)
    auto p4 = env.promoteForArithmetic(BuiltinTypes::Float64, BuiltinTypes::Int64);
    ASSERT_EQ(p4.id, BuiltinTypes::Float64.id);

    // Integer + Real -> Float64
    auto p5 = env.promoteForArithmetic(BuiltinTypes::Integer, BuiltinTypes::Real);
    ASSERT_EQ(p5.id, BuiltinTypes::Float64.id);
}

TEST(type_flags) {
    TypeEnvironment env;

    // Int64 should be exact
    ASSERT_TRUE(BuiltinTypes::Int64.isExact());
    ASSERT_FALSE(BuiltinTypes::Int64.isLinear());
    ASSERT_FALSE(BuiltinTypes::Int64.isProof());

    // Float64 should not be exact
    ASSERT_FALSE(BuiltinTypes::Float64.isExact());

    // Handle should be linear
    ASSERT_TRUE(BuiltinTypes::Handle.isLinear());

    // Eq (proposition) should be proof type
    ASSERT_TRUE(BuiltinTypes::Eq.isProof());
}

TEST(universe_levels) {
    // Ground types are in U0
    ASSERT_EQ(static_cast<int>(BuiltinTypes::Int64.level), static_cast<int>(Universe::U0));
    ASSERT_EQ(static_cast<int>(BuiltinTypes::Float64.level), static_cast<int>(Universe::U0));
    ASSERT_EQ(static_cast<int>(BuiltinTypes::String.level), static_cast<int>(Universe::U0));

    // Type constructors are in U1
    ASSERT_EQ(static_cast<int>(BuiltinTypes::List.level), static_cast<int>(Universe::U1));
    ASSERT_EQ(static_cast<int>(BuiltinTypes::Vector.level), static_cast<int>(Universe::U1));
    ASSERT_EQ(static_cast<int>(BuiltinTypes::Function.level), static_cast<int>(Universe::U1));

    // Propositions are in U2
    ASSERT_EQ(static_cast<int>(BuiltinTypes::Eq.level), static_cast<int>(Universe::U2));
    ASSERT_EQ(static_cast<int>(BuiltinTypes::Bounded.level), static_cast<int>(Universe::U2));
}

TEST(runtime_type_mapping) {
    TypeEnvironment env;

    // Map from runtime types
    ASSERT_EQ(env.fromRuntimeType(ESHKOL_VALUE_INT64).id, BuiltinTypes::Int64.id);
    ASSERT_EQ(env.fromRuntimeType(ESHKOL_VALUE_DOUBLE).id, BuiltinTypes::Float64.id);
    ASSERT_EQ(env.fromRuntimeType(ESHKOL_VALUE_STRING_PTR).id, BuiltinTypes::String.id);
    ASSERT_EQ(env.fromRuntimeType(ESHKOL_VALUE_BOOL).id, BuiltinTypes::Boolean.id);
    ASSERT_EQ(env.fromRuntimeType(ESHKOL_VALUE_NULL).id, BuiltinTypes::Null.id);

    // Map to runtime types
    ASSERT_EQ(env.toRuntimeType(BuiltinTypes::Int64), ESHKOL_VALUE_INT64);
    ASSERT_EQ(env.toRuntimeType(BuiltinTypes::Float64), ESHKOL_VALUE_DOUBLE);
    ASSERT_EQ(env.toRuntimeType(BuiltinTypes::String), ESHKOL_VALUE_STRING_PTR);
    ASSERT_EQ(env.toRuntimeType(BuiltinTypes::Boolean), ESHKOL_VALUE_BOOL);
}

TEST(user_type_registration) {
    TypeEnvironment env;

    // Register a user type
    auto my_type = env.registerUserType("MyCustomType", Universe::U0, 0,
                                        BuiltinTypes::Value);

    // Should be able to look it up
    auto lookup = env.lookupType("MyCustomType");
    ASSERT_TRUE(lookup.has_value());
    ASSERT_EQ(lookup->id, my_type.id);

    // Should be a subtype of Value
    ASSERT_TRUE(env.isSubtype(my_type, BuiltinTypes::Value));

    // ID should be >= 1000 (user type range)
    ASSERT_TRUE(my_type.id >= 1000);
}

TEST(supertype_chain) {
    TypeEnvironment env;

    // Get the supertype chain for Int64
    auto chain = env.getSupertypeChain(BuiltinTypes::Int64);

    // Should be: Int64 -> Integer -> Number -> Value
    ASSERT_TRUE(chain.size() >= 4);
    ASSERT_EQ(chain[0].id, BuiltinTypes::Int64.id);
    ASSERT_EQ(chain[1].id, BuiltinTypes::Integer.id);
    ASSERT_EQ(chain[2].id, BuiltinTypes::Number.id);
    ASSERT_EQ(chain[3].id, BuiltinTypes::Value.id);
}

TEST(type_families) {
    TypeEnvironment env;

    // List should be a type family
    const TypeNode* list_node = env.getTypeNode(BuiltinTypes::List);
    ASSERT_TRUE(list_node != nullptr);
    ASSERT_TRUE(list_node->is_type_family);
    ASSERT_EQ(list_node->param_names.size(), 1);
    ASSERT_EQ(list_node->param_names[0], "a");

    // Pair should have two parameters
    const TypeNode* pair_node = env.getTypeNode(BuiltinTypes::Pair);
    ASSERT_TRUE(pair_node != nullptr);
    ASSERT_TRUE(pair_node->is_type_family);
    ASSERT_EQ(pair_node->param_names.size(), 2);
}

TEST(runtime_rep) {
    TypeEnvironment env;

    // Check runtime representations
    ASSERT_EQ(env.getRuntimeRep(BuiltinTypes::Int64), RuntimeRep::Int64);
    ASSERT_EQ(env.getRuntimeRep(BuiltinTypes::Float64), RuntimeRep::Float64);
    ASSERT_EQ(env.getRuntimeRep(BuiltinTypes::String), RuntimeRep::Pointer);
    ASSERT_EQ(env.getRuntimeRep(BuiltinTypes::List), RuntimeRep::Pointer);
    ASSERT_EQ(env.getRuntimeRep(BuiltinTypes::Eq), RuntimeRep::Erased);
}

TEST(utility_functions) {
    TypeEnvironment env;

    // isNumericType
    ASSERT_TRUE(isNumericType(env, BuiltinTypes::Int64));
    ASSERT_TRUE(isNumericType(env, BuiltinTypes::Float64));
    ASSERT_TRUE(isNumericType(env, BuiltinTypes::Integer));
    ASSERT_TRUE(isNumericType(env, BuiltinTypes::Real));
    ASSERT_TRUE(isNumericType(env, BuiltinTypes::Number));
    ASSERT_FALSE(isNumericType(env, BuiltinTypes::String));
    ASSERT_FALSE(isNumericType(env, BuiltinTypes::Boolean));

    // Universe strings
    ASSERT_TRUE(universeToString(Universe::U0) != nullptr);
    ASSERT_TRUE(universeToString(Universe::U1) != nullptr);
    ASSERT_TRUE(universeToString(Universe::U2) != nullptr);

    // RuntimeRep strings
    ASSERT_TRUE(runtimeRepToString(RuntimeRep::Int64) != nullptr);
    ASSERT_TRUE(runtimeRepToString(RuntimeRep::Pointer) != nullptr);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== HoTT Type System Tests ===" << std::endl;
    std::cout << std::endl;

    RUN_TEST(builtin_types_exist);
    RUN_TEST(type_aliases);
    RUN_TEST(numeric_tower_subtyping);
    RUN_TEST(text_subtyping);
    RUN_TEST(reflexivity);
    RUN_TEST(least_common_supertype);
    RUN_TEST(arithmetic_promotion);
    RUN_TEST(type_flags);
    RUN_TEST(universe_levels);
    RUN_TEST(runtime_type_mapping);
    RUN_TEST(user_type_registration);
    RUN_TEST(supertype_chain);
    RUN_TEST(type_families);
    RUN_TEST(runtime_rep);
    RUN_TEST(utility_functions);

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
