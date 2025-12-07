/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Comprehensive Type Checker Tests for HoTT Type System
 * Tests: CTValue, DependentType, DimensionChecker, LinearContext,
 *        BorrowChecker, UnsafeContext, SigmaType, TypeChecker
 */

#include <eshkol/types/type_checker.h>
#include <eshkol/types/dependent.h>
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
// CTValue Tests (Compile-Time Values)
// ============================================================================

TEST(ctvalue_nat_creation) {
    CTValue val = CTValue::makeNat(42);
    ASSERT_TRUE(val.isNat());
    ASSERT_FALSE(val.isBool());
    ASSERT_FALSE(val.isExpr());
    ASSERT_FALSE(val.isUnknown());
    ASSERT_TRUE(val.isKnown());
    ASSERT_EQ(val.natValue(), 42);
}

TEST(ctvalue_bool_creation) {
    CTValue t = CTValue::makeBool(true);
    CTValue f = CTValue::makeBool(false);

    ASSERT_TRUE(t.isBool());
    ASSERT_TRUE(t.boolValue());
    ASSERT_TRUE(f.isBool());
    ASSERT_FALSE(f.boolValue());
}

TEST(ctvalue_unknown_creation) {
    CTValue val = CTValue::makeUnknown();
    ASSERT_TRUE(val.isUnknown());
    ASSERT_FALSE(val.isKnown());
    ASSERT_FALSE(val.isNat());
}

TEST(ctvalue_eval_nat) {
    CTValue known = CTValue::makeNat(100);
    CTValue unknown = CTValue::makeUnknown();

    auto eval1 = known.tryEvalNat();
    ASSERT_TRUE(eval1.has_value());
    ASSERT_EQ(*eval1, 100);

    auto eval2 = unknown.tryEvalNat();
    ASSERT_FALSE(eval2.has_value());
}

TEST(ctvalue_eval_bool) {
    CTValue known = CTValue::makeBool(true);
    CTValue unknown = CTValue::makeUnknown();

    auto eval1 = known.tryEvalBool();
    ASSERT_TRUE(eval1.has_value());
    ASSERT_EQ(*eval1, true);

    auto eval2 = unknown.tryEvalBool();
    ASSERT_FALSE(eval2.has_value());
}

TEST(ctvalue_less_than) {
    CTValue a = CTValue::makeNat(5);
    CTValue b = CTValue::makeNat(10);
    CTValue c = CTValue::makeNat(5);
    CTValue unknown = CTValue::makeUnknown();

    ASSERT_EQ(a.lessThan(b), CTValue::CompareResult::True);
    ASSERT_EQ(b.lessThan(a), CTValue::CompareResult::False);
    ASSERT_EQ(a.lessThan(c), CTValue::CompareResult::False);  // 5 < 5 is false
    ASSERT_EQ(a.lessThan(unknown), CTValue::CompareResult::Unknown);
}

TEST(ctvalue_equals) {
    CTValue a = CTValue::makeNat(5);
    CTValue b = CTValue::makeNat(5);
    CTValue c = CTValue::makeNat(10);
    CTValue t = CTValue::makeBool(true);

    ASSERT_EQ(a.equals(b), CTValue::CompareResult::True);
    ASSERT_EQ(a.equals(c), CTValue::CompareResult::False);
    ASSERT_EQ(a.equals(t), CTValue::CompareResult::Unknown);  // Different kinds
}

TEST(ctvalue_arithmetic) {
    CTValue a = CTValue::makeNat(5);
    CTValue b = CTValue::makeNat(3);

    CTValue sum = a.add(b);
    ASSERT_TRUE(sum.isNat());
    ASSERT_EQ(sum.natValue(), 8);

    CTValue product = a.mul(b);
    ASSERT_TRUE(product.isNat());
    ASSERT_EQ(product.natValue(), 15);

    // Unknown propagation
    CTValue unknown = CTValue::makeUnknown();
    CTValue sum2 = a.add(unknown);
    ASSERT_TRUE(sum2.isUnknown());
}

TEST(ctvalue_to_string) {
    ASSERT_EQ(CTValue::makeNat(42).toString(), "42");
    ASSERT_EQ(CTValue::makeBool(true).toString(), "#t");
    ASSERT_EQ(CTValue::makeBool(false).toString(), "#f");
    ASSERT_EQ(CTValue::makeUnknown().toString(), "?");
}

// ============================================================================
// DependentType Tests
// ============================================================================

TEST(dependent_type_simple) {
    DependentType dt(BuiltinTypes::Int64);
    ASSERT_TRUE(dt.isSimple());
    ASSERT_TRUE(dt.allValuesKnown());
    ASSERT_EQ(dt.base.id, BuiltinTypes::Int64.id);
}

TEST(dependent_type_with_values) {
    DependentType dt;
    dt.base = BuiltinTypes::Vector;
    dt.type_indices.push_back(BuiltinTypes::Float64);
    dt.value_indices.push_back(CTValue::makeNat(100));

    ASSERT_FALSE(dt.isSimple());
    ASSERT_TRUE(dt.allValuesKnown());

    auto dim = dt.getDimension(0);
    ASSERT_TRUE(dim.has_value());
    ASSERT_EQ(*dim, 100);
}

TEST(dependent_type_unknown_dimension) {
    DependentType dt;
    dt.base = BuiltinTypes::Vector;
    dt.value_indices.push_back(CTValue::makeUnknown());

    ASSERT_FALSE(dt.allValuesKnown());
    auto dim = dt.getDimension(0);
    ASSERT_FALSE(dim.has_value());
}

TEST(dependent_type_equality) {
    DependentType a(BuiltinTypes::Vector);
    a.value_indices.push_back(CTValue::makeNat(10));

    DependentType b(BuiltinTypes::Vector);
    b.value_indices.push_back(CTValue::makeNat(10));

    DependentType c(BuiltinTypes::Vector);
    c.value_indices.push_back(CTValue::makeNat(20));

    ASSERT_TRUE(a.equals(b));
    ASSERT_FALSE(a.equals(c));
}

// ============================================================================
// DimensionChecker Tests
// ============================================================================

TEST(dimension_bounds_check_pass) {
    CTValue idx = CTValue::makeNat(5);
    CTValue bound = CTValue::makeNat(10);

    auto result = DimensionChecker::checkBounds(idx, bound, "test");
    ASSERT_TRUE(result.valid);
}

TEST(dimension_bounds_check_fail) {
    CTValue idx = CTValue::makeNat(15);
    CTValue bound = CTValue::makeNat(10);

    auto result = DimensionChecker::checkBounds(idx, bound, "test");
    ASSERT_FALSE(result.valid);
    ASSERT_TRUE(result.error_message.find("out of bounds") != std::string::npos);
}

TEST(dimension_bounds_edge_case) {
    // Edge case: index equals bound (should fail, since bound is exclusive)
    CTValue idx = CTValue::makeNat(10);
    CTValue bound = CTValue::makeNat(10);

    auto result = DimensionChecker::checkBounds(idx, bound, "test");
    ASSERT_FALSE(result.valid);
}

TEST(dimension_equality_check) {
    CTValue a = CTValue::makeNat(100);
    CTValue b = CTValue::makeNat(100);
    CTValue c = CTValue::makeNat(50);

    auto result1 = DimensionChecker::checkDimensionsEqual(a, b, "test");
    ASSERT_TRUE(result1.valid);

    auto result2 = DimensionChecker::checkDimensionsEqual(a, c, "test");
    ASSERT_FALSE(result2.valid);
}

TEST(dimension_dot_product) {
    DependentType vec1;
    vec1.base = BuiltinTypes::Vector;
    vec1.value_indices.push_back(CTValue::makeNat(100));

    DependentType vec2;
    vec2.base = BuiltinTypes::Vector;
    vec2.value_indices.push_back(CTValue::makeNat(100));

    DependentType vec3;
    vec3.base = BuiltinTypes::Vector;
    vec3.value_indices.push_back(CTValue::makeNat(50));

    auto result1 = DimensionChecker::checkDotProductDimensions(vec1, vec2, "test");
    ASSERT_TRUE(result1.valid);

    auto result2 = DimensionChecker::checkDotProductDimensions(vec1, vec3, "test");
    ASSERT_FALSE(result2.valid);
}

TEST(dimension_matrix_multiply) {
    // Matrix A: 3x4
    DependentType matA;
    matA.base = BuiltinTypes::Tensor;
    matA.value_indices.push_back(CTValue::makeNat(3));
    matA.value_indices.push_back(CTValue::makeNat(4));

    // Matrix B: 4x5 (compatible with A)
    DependentType matB;
    matB.base = BuiltinTypes::Tensor;
    matB.value_indices.push_back(CTValue::makeNat(4));
    matB.value_indices.push_back(CTValue::makeNat(5));

    // Matrix C: 5x2 (not compatible with A)
    DependentType matC;
    matC.base = BuiltinTypes::Tensor;
    matC.value_indices.push_back(CTValue::makeNat(5));
    matC.value_indices.push_back(CTValue::makeNat(2));

    auto result1 = DimensionChecker::checkMatMulDimensions(matA, matB, "test");
    ASSERT_TRUE(result1.valid);

    auto result2 = DimensionChecker::checkMatMulDimensions(matA, matC, "test");
    ASSERT_FALSE(result2.valid);
}

// ============================================================================
// SigmaType Tests (Dependent Pairs)
// ============================================================================

TEST(sigma_type_simple_product) {
    SigmaType prod = SigmaType::makeProduct(BuiltinTypes::Int64, BuiltinTypes::String);

    ASSERT_TRUE(prod.isSimpleProduct());
    ASSERT_EQ(prod.firstType().id, BuiltinTypes::Int64.id);
    ASSERT_EQ(prod.secondType().id, BuiltinTypes::String.id);
}

TEST(sigma_type_dependent) {
    SigmaType sigma("n", BuiltinTypes::Int64, BuiltinTypes::Vector, true);

    ASSERT_FALSE(sigma.isSimpleProduct());
    ASSERT_EQ(sigma.witness_name, "n");
    ASSERT_EQ(sigma.witness_type.id, BuiltinTypes::Int64.id);
    ASSERT_TRUE(sigma.is_dependent);
}

TEST(sigma_value_known_witness) {
    SigmaValue val(CTValue::makeNat(10), BuiltinTypes::Int64, BuiltinTypes::Vector);

    ASSERT_TRUE(val.hasKnownWitness());
    ASSERT_EQ(val.witness.natValue(), 10);
}

TEST(sigma_value_unknown_witness) {
    SigmaValue val(CTValue::makeUnknown(), BuiltinTypes::Int64, BuiltinTypes::Vector);

    ASSERT_FALSE(val.hasKnownWitness());
}

// ============================================================================
// LinearContext Tests
// ============================================================================

TEST(linear_context_declare) {
    LinearContext ctx;
    ctx.declareLinear("x");

    ASSERT_TRUE(ctx.isLinear("x"));
    ASSERT_FALSE(ctx.isLinear("y"));
    ASSERT_EQ(ctx.getUsage("x"), LinearContext::Usage::Unused);
}

TEST(linear_context_use_once) {
    LinearContext ctx;
    ctx.declareLinear("x");
    ctx.use("x");

    ASSERT_EQ(ctx.getUsage("x"), LinearContext::Usage::UsedOnce);
    ASSERT_TRUE(ctx.checkAllUsedOnce());
}

TEST(linear_context_use_multiple) {
    LinearContext ctx;
    ctx.declareLinear("x");
    ctx.use("x");
    ctx.use("x");

    ASSERT_EQ(ctx.getUsage("x"), LinearContext::Usage::UsedMultiple);
    ASSERT_FALSE(ctx.checkAllUsedOnce());
}

TEST(linear_context_unused) {
    LinearContext ctx;
    ctx.declareLinear("x");
    ctx.declareLinear("y");
    ctx.use("x");

    ASSERT_FALSE(ctx.checkAllUsedOnce());
    auto unused = ctx.getUnused();
    ASSERT_EQ(unused.size(), 1);
    ASSERT_EQ(unused[0], "y");
}

TEST(linear_context_overused) {
    LinearContext ctx;
    ctx.declareLinear("x");
    ctx.declareLinear("y");
    ctx.use("x");
    ctx.use("x");
    ctx.use("y");

    auto overused = ctx.getOverused();
    ASSERT_EQ(overused.size(), 1);
    ASSERT_EQ(overused[0], "x");
}

// ============================================================================
// BorrowChecker Tests
// ============================================================================

TEST(borrow_declare_owned) {
    BorrowChecker bc;
    bc.declareOwned("x");

    ASSERT_EQ(bc.getState("x"), BorrowState::Owned);
    ASSERT_TRUE(bc.canUse("x"));
    ASSERT_TRUE(bc.canMove("x"));
    ASSERT_TRUE(bc.canBorrowShared("x"));
    ASSERT_TRUE(bc.canBorrowMut("x"));
}

TEST(borrow_move) {
    BorrowChecker bc;
    bc.declareOwned("x");

    ASSERT_TRUE(bc.move("x"));
    ASSERT_EQ(bc.getState("x"), BorrowState::Moved);
    ASSERT_FALSE(bc.canUse("x"));
    ASSERT_FALSE(bc.canMove("x"));
}

TEST(borrow_double_move_fails) {
    BorrowChecker bc;
    bc.declareOwned("x");

    ASSERT_TRUE(bc.move("x"));
    ASSERT_FALSE(bc.move("x"));  // Can't move twice
    ASSERT_TRUE(bc.hasErrors());
}

TEST(borrow_shared) {
    BorrowChecker bc;
    bc.declareOwned("x");

    ASSERT_TRUE(bc.borrowShared("x"));
    ASSERT_EQ(bc.getState("x"), BorrowState::BorrowedShared);

    // Can borrow shared again
    ASSERT_TRUE(bc.borrowShared("x"));

    // Can't move while borrowed
    ASSERT_FALSE(bc.canMove("x"));
}

TEST(borrow_mutable) {
    BorrowChecker bc;
    bc.declareOwned("x");

    ASSERT_TRUE(bc.borrowMut("x"));
    ASSERT_EQ(bc.getState("x"), BorrowState::BorrowedMut);

    // Can't borrow again
    ASSERT_FALSE(bc.canBorrowShared("x"));
    ASSERT_FALSE(bc.canBorrowMut("x"));
}

TEST(borrow_return) {
    BorrowChecker bc;
    bc.declareOwned("x");

    bc.borrowShared("x");
    bc.returnBorrow("x");

    ASSERT_EQ(bc.getState("x"), BorrowState::Owned);
    ASSERT_TRUE(bc.canMove("x"));
}

TEST(borrow_mutable_exclusive) {
    BorrowChecker bc;
    bc.declareOwned("x");

    bc.borrowShared("x");
    ASSERT_FALSE(bc.borrowMut("x"));  // Can't borrow mut when shared exists
    ASSERT_TRUE(bc.hasErrors());
}

TEST(borrow_double_mutable_fails) {
    BorrowChecker bc;
    bc.declareOwned("x");

    bc.borrowMut("x");
    ASSERT_FALSE(bc.borrowMut("x"));  // Can't have two mutable borrows
}

TEST(borrow_drop) {
    BorrowChecker bc;
    bc.declareOwned("x");

    ASSERT_TRUE(bc.drop("x"));
    ASSERT_EQ(bc.getState("x"), BorrowState::Dropped);
    ASSERT_FALSE(bc.canUse("x"));
}

TEST(borrow_use_after_drop) {
    BorrowChecker bc;
    bc.declareOwned("x");
    bc.drop("x");

    ASSERT_FALSE(bc.borrowShared("x"));  // Can't borrow dropped value
    ASSERT_TRUE(bc.hasErrors());
}

TEST(borrow_scope_management) {
    BorrowChecker bc;

    bc.pushScope();
    bc.declareOwned("x");
    ASSERT_EQ(bc.currentScope(), 1);

    bc.popScope();
    ASSERT_EQ(bc.currentScope(), 0);
}

// ============================================================================
// UnsafeContext Tests
// ============================================================================

TEST(unsafe_default_safe) {
    UnsafeContext ctx;
    ASSERT_FALSE(ctx.isUnsafe());
    ASSERT_EQ(ctx.depth(), 0);
}

TEST(unsafe_enter_exit) {
    UnsafeContext ctx;

    ctx.enterUnsafe();
    ASSERT_TRUE(ctx.isUnsafe());
    ASSERT_EQ(ctx.depth(), 1);

    ctx.exitUnsafe();
    ASSERT_FALSE(ctx.isUnsafe());
    ASSERT_EQ(ctx.depth(), 0);
}

TEST(unsafe_nested) {
    UnsafeContext ctx;

    ctx.enterUnsafe();
    ctx.enterUnsafe();
    ASSERT_EQ(ctx.depth(), 2);
    ASSERT_TRUE(ctx.isUnsafe());

    ctx.exitUnsafe();
    ASSERT_EQ(ctx.depth(), 1);
    ASSERT_TRUE(ctx.isUnsafe());

    ctx.exitUnsafe();
    ASSERT_FALSE(ctx.isUnsafe());
}

TEST(unsafe_scoped_raii) {
    UnsafeContext ctx;

    {
        UnsafeContext::ScopedUnsafe guard(ctx);
        ASSERT_TRUE(ctx.isUnsafe());
    }

    ASSERT_FALSE(ctx.isUnsafe());
}

// ============================================================================
// Context Tests (Type Binding Context)
// ============================================================================

TEST(context_bind_lookup) {
    Context ctx;
    ctx.bind("x", BuiltinTypes::Int64);

    auto lookup = ctx.lookup("x");
    ASSERT_TRUE(lookup.has_value());
    ASSERT_EQ(lookup->id, BuiltinTypes::Int64.id);
}

TEST(context_unbound_lookup) {
    Context ctx;
    auto lookup = ctx.lookup("nonexistent");
    ASSERT_FALSE(lookup.has_value());
}

TEST(context_scope_shadowing) {
    Context ctx;
    ctx.bind("x", BuiltinTypes::Int64);

    ctx.pushScope();
    ctx.bind("x", BuiltinTypes::String);

    auto lookup = ctx.lookup("x");
    ASSERT_TRUE(lookup.has_value());
    ASSERT_EQ(lookup->id, BuiltinTypes::String.id);

    ctx.popScope();
    lookup = ctx.lookup("x");
    ASSERT_TRUE(lookup.has_value());
    ASSERT_EQ(lookup->id, BuiltinTypes::Int64.id);
}

TEST(context_linear_binding) {
    Context ctx;
    ctx.bindLinear("q", BuiltinTypes::Int64);

    ASSERT_TRUE(ctx.isLinear("q"));
    ASSERT_FALSE(ctx.isLinearUsed("q"));

    ctx.useLinear("q");
    ASSERT_TRUE(ctx.isLinearUsed("q"));
}

TEST(context_linear_constraints) {
    Context ctx;
    ctx.bindLinear("a", BuiltinTypes::Int64);
    ctx.bindLinear("b", BuiltinTypes::Int64);

    ctx.useLinear("a");
    // 'b' not used - should fail constraints
    ASSERT_FALSE(ctx.checkLinearConstraints());

    ctx.useLinear("b");
    ASSERT_TRUE(ctx.checkLinearConstraints());
}

// ============================================================================
// TypeChecker Tests
// ============================================================================

TEST(type_checker_unsafe_context) {
    TypeEnvironment env;
    TypeChecker checker(env);

    ASSERT_FALSE(checker.isUnsafe());

    checker.enterUnsafe();
    ASSERT_TRUE(checker.isUnsafe());

    checker.exitUnsafe();
    ASSERT_FALSE(checker.isUnsafe());
}

TEST(type_checker_borrow_access) {
    TypeEnvironment env;
    TypeChecker checker(env);

    BorrowChecker& bc = checker.borrowChecker();
    bc.declareOwned("test");
    ASSERT_EQ(bc.getState("test"), BorrowState::Owned);
}

TEST(type_checker_context_access) {
    TypeEnvironment env;
    TypeChecker checker(env);

    Context& ctx = checker.context();
    ctx.bind("x", BuiltinTypes::Float64);

    auto lookup = ctx.lookup("x");
    ASSERT_TRUE(lookup.has_value());
    ASSERT_EQ(lookup->id, BuiltinTypes::Float64.id);
}

TEST(type_checker_linear_type_check) {
    TypeEnvironment env;
    TypeChecker checker(env);

    // Handle type should be linear
    ASSERT_TRUE(checker.isLinearType(BuiltinTypes::Handle));

    // Int64 should not be linear
    ASSERT_FALSE(checker.isLinearType(BuiltinTypes::Int64));
}

TEST(type_checker_check_linear_let_empty) {
    TypeEnvironment env;
    TypeChecker checker(env);

    std::vector<std::string> empty;
    auto result = checker.checkLinearLet(empty);
    ASSERT_TRUE(result.success);
}

TEST(type_checker_check_linear_let_unused) {
    TypeEnvironment env;
    TypeChecker checker(env);

    Context& ctx = checker.context();
    ctx.bindLinear("q", BuiltinTypes::Handle);
    // Don't use 'q'

    std::vector<std::string> bindings = {"q"};
    auto result = checker.checkLinearLet(bindings);
    ASSERT_FALSE(result.success);
}

TEST(type_checker_check_linear_let_unsafe_skips) {
    TypeEnvironment env;
    TypeChecker checker(env);

    Context& ctx = checker.context();
    ctx.bindLinear("q", BuiltinTypes::Handle);
    // Don't use 'q', but in unsafe mode

    checker.enterUnsafe();

    std::vector<std::string> bindings = {"q"};
    auto result = checker.checkLinearLet(bindings);
    ASSERT_TRUE(result.success);  // Should pass in unsafe mode
}

TEST(type_checker_dimension_check) {
    TypeEnvironment env;
    TypeChecker checker(env);

    CTValue idx = CTValue::makeNat(5);
    CTValue bound = CTValue::makeNat(10);

    auto result = checker.checkVectorBounds(idx, bound, "test");
    ASSERT_TRUE(result.success);
}

TEST(type_checker_dot_product_dimensions) {
    TypeEnvironment env;
    TypeChecker checker(env);

    DependentType vec1;
    vec1.base = BuiltinTypes::Vector;
    vec1.value_indices.push_back(CTValue::makeNat(100));

    DependentType vec2;
    vec2.base = BuiltinTypes::Vector;
    vec2.value_indices.push_back(CTValue::makeNat(100));

    auto result = checker.checkDotProductDimensions(vec1, vec2);
    ASSERT_TRUE(result.success);
}

TEST(type_checker_matrix_multiply_dimensions) {
    TypeEnvironment env;
    TypeChecker checker(env);

    DependentType mat1;
    mat1.base = BuiltinTypes::Tensor;
    mat1.value_indices.push_back(CTValue::makeNat(3));
    mat1.value_indices.push_back(CTValue::makeNat(4));

    DependentType mat2;
    mat2.base = BuiltinTypes::Tensor;
    mat2.value_indices.push_back(CTValue::makeNat(4));
    mat2.value_indices.push_back(CTValue::makeNat(5));

    auto result = checker.checkMatrixMultiplyDimensions(mat1, mat2);
    ASSERT_TRUE(result.success);
}

TEST(type_checker_errors_clear) {
    TypeEnvironment env;
    TypeChecker checker(env);

    ASSERT_FALSE(checker.hasErrors());

    // Force an error
    CTValue idx = CTValue::makeNat(100);
    CTValue bound = CTValue::makeNat(10);
    checker.checkVectorBounds(idx, bound, "test");

    ASSERT_TRUE(checker.hasErrors());

    checker.clearErrors();
    ASSERT_FALSE(checker.hasErrors());
}

// ============================================================================
// TypeCheckResult Tests
// ============================================================================

TEST(type_check_result_ok) {
    auto result = TypeCheckResult::ok(BuiltinTypes::Int64);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.inferred_type.id, BuiltinTypes::Int64.id);
    ASSERT_TRUE(result.error_message.empty());
}

TEST(type_check_result_error) {
    auto result = TypeCheckResult::error("test error", 10, 5);
    ASSERT_FALSE(result.success);
    ASSERT_EQ(result.error_message, "test error");
    ASSERT_EQ(result.line, 10);
    ASSERT_EQ(result.column, 5);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== HoTT Type Checker Comprehensive Tests ===" << std::endl;
    std::cout << std::endl;

    std::cout << "--- CTValue Tests ---" << std::endl;
    RUN_TEST(ctvalue_nat_creation);
    RUN_TEST(ctvalue_bool_creation);
    RUN_TEST(ctvalue_unknown_creation);
    RUN_TEST(ctvalue_eval_nat);
    RUN_TEST(ctvalue_eval_bool);
    RUN_TEST(ctvalue_less_than);
    RUN_TEST(ctvalue_equals);
    RUN_TEST(ctvalue_arithmetic);
    RUN_TEST(ctvalue_to_string);

    std::cout << std::endl << "--- DependentType Tests ---" << std::endl;
    RUN_TEST(dependent_type_simple);
    RUN_TEST(dependent_type_with_values);
    RUN_TEST(dependent_type_unknown_dimension);
    RUN_TEST(dependent_type_equality);

    std::cout << std::endl << "--- DimensionChecker Tests ---" << std::endl;
    RUN_TEST(dimension_bounds_check_pass);
    RUN_TEST(dimension_bounds_check_fail);
    RUN_TEST(dimension_bounds_edge_case);
    RUN_TEST(dimension_equality_check);
    RUN_TEST(dimension_dot_product);
    RUN_TEST(dimension_matrix_multiply);

    std::cout << std::endl << "--- SigmaType Tests ---" << std::endl;
    RUN_TEST(sigma_type_simple_product);
    RUN_TEST(sigma_type_dependent);
    RUN_TEST(sigma_value_known_witness);
    RUN_TEST(sigma_value_unknown_witness);

    std::cout << std::endl << "--- LinearContext Tests ---" << std::endl;
    RUN_TEST(linear_context_declare);
    RUN_TEST(linear_context_use_once);
    RUN_TEST(linear_context_use_multiple);
    RUN_TEST(linear_context_unused);
    RUN_TEST(linear_context_overused);

    std::cout << std::endl << "--- BorrowChecker Tests ---" << std::endl;
    RUN_TEST(borrow_declare_owned);
    RUN_TEST(borrow_move);
    RUN_TEST(borrow_double_move_fails);
    RUN_TEST(borrow_shared);
    RUN_TEST(borrow_mutable);
    RUN_TEST(borrow_return);
    RUN_TEST(borrow_mutable_exclusive);
    RUN_TEST(borrow_double_mutable_fails);
    RUN_TEST(borrow_drop);
    RUN_TEST(borrow_use_after_drop);
    RUN_TEST(borrow_scope_management);

    std::cout << std::endl << "--- UnsafeContext Tests ---" << std::endl;
    RUN_TEST(unsafe_default_safe);
    RUN_TEST(unsafe_enter_exit);
    RUN_TEST(unsafe_nested);
    RUN_TEST(unsafe_scoped_raii);

    std::cout << std::endl << "--- Context Tests ---" << std::endl;
    RUN_TEST(context_bind_lookup);
    RUN_TEST(context_unbound_lookup);
    RUN_TEST(context_scope_shadowing);
    RUN_TEST(context_linear_binding);
    RUN_TEST(context_linear_constraints);

    std::cout << std::endl << "--- TypeChecker Tests ---" << std::endl;
    RUN_TEST(type_checker_unsafe_context);
    RUN_TEST(type_checker_borrow_access);
    RUN_TEST(type_checker_context_access);
    RUN_TEST(type_checker_linear_type_check);
    RUN_TEST(type_checker_check_linear_let_empty);
    RUN_TEST(type_checker_check_linear_let_unused);
    RUN_TEST(type_checker_check_linear_let_unsafe_skips);
    RUN_TEST(type_checker_dimension_check);
    RUN_TEST(type_checker_dot_product_dimensions);
    RUN_TEST(type_checker_matrix_multiply_dimensions);
    RUN_TEST(type_checker_errors_clear);

    std::cout << std::endl << "--- TypeCheckResult Tests ---" << std::endl;
    RUN_TEST(type_check_result_ok);
    RUN_TEST(type_check_result_error);

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    std::cout << "Total:  " << (tests_passed + tests_failed) << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
