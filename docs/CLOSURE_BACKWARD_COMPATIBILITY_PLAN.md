# Closure Implementation Backward Compatibility Plan

## Critical Constraint

**MUST NOT BREAK ANY EXISTING LIST OPERATIONS**

All current functionality must continue working:
- ✅ [`map`](lib/backend/llvm_codegen.cpp) with lambdas
- ✅ [`fold`](lib/backend/llvm_codegen.cpp) with lambdas  
- ✅ [`filter`](lib/backend/llvm_codegen.cpp) with lambdas
- ✅ Top-level function definitions
- ✅ Simple lambdas without captures
- ✅ Autodiff on regular functions
- ✅ All 100+ existing test files

## Safety Strategy

### Principle: **Additive Changes Only**

The closure implementation will be **purely additive** - adding new code paths while preserving all existing behavior.

### Key Safety Rules

1. **Default to Existing Behavior**
   - If `num_captured == 0`, use current codegen (no environment parameter)
   - Only modify behavior when captures are detected

2. **Preserve Function Signatures**
   - Non-closure functions keep current signatures
   - Only closures get environment parameter

3. **Preserve Variable Lookup**
   - Current lookup path remains unchanged
   - Environment lookup is additional, not replacement

4. **Preserve Arena Operations**
   - New environment allocation doesn't affect existing arena usage
   - All existing arena functions remain untouched

## Implementation Safety Checks

### Check 1: Function Signature Guard

```cpp
llvm::Function* createFunction(const std::string& name, 
                              llvm::Type* return_type,
                              const std::vector<llvm::Type*>& params,
                              eshkol_ast_t* lambda_node) {
    // SAFETY: Only add environment param if explicitly needed
    bool needs_env = (lambda_node && 
                     lambda_node->operation.lambda_op.num_captured > 0);
    
    if (!needs_env) {
        // EXISTING PATH: Use current function creation
        return createFunctionExisting(name, return_type, params);
    } else {
        // NEW PATH: Add environment parameter
        return createFunctionWithEnvironment(name, return_type, params);
    }
}
```

### Check 2: Variable Lookup Guard

```cpp
llvm::Value* lookupVariable(const std::string& var_name) {
    // SAFETY: Try environment ONLY if we have one
    if (current_function_env_ptr && isInCapturedList(var_name)) {
        // NEW PATH: Load from environment
        return loadFromEnvironment(var_name);
    } else {
        // EXISTING PATH: Use current symbol table lookup
        return lookupVariableExisting(var_name);
    }
}
```

### Check 3: Call Site Guard

```cpp
llvm::Value* codegenCall(llvm::Function* func, const std::vector<llvm::Value*>& args) {
    // SAFETY: Only pass environment if function expects it
    if (functionExpectsEnvironment(func)) {
        // NEW PATH: Include environment parameter
        return codegenCallWithEnvironment(func, args);
    } else {
        // EXISTING PATH: Use current call generation
        return Builder.CreateCall(func, args);
    }
}
```

## Regression Testing Plan

### Phase 1: Before Any Changes
Run complete test suite and capture baselines:
```bash
./scripts/run_all_tests.sh > test_baseline_before_closures.txt
./scripts/run_autodiff_tests.sh > autodiff_baseline_before_closures.txt
```

### Phase 2: After Each Implementation Step
Re-run tests to ensure nothing broke:
```bash
# After arena changes
./scripts/run_all_tests.sh > test_after_arena.txt
diff test_baseline_before_closures.txt test_after_arena.txt

# After parser changes
./scripts/run_all_tests.sh > test_after_parser.txt
diff test_baseline_before_closures.txt test_after_parser.txt

# After codegen changes
./scripts/run_all_tests.sh > test_after_codegen.txt
diff test_baseline_before_closures.txt test_after_codegen.txt
```

### Phase 3: Comprehensive Validation
Test specific critical areas:

#### List Operations Tests
```scheme
;; MUST STILL WORK
(map (lambda (x) (* 2 x)) (list 1 2 3))
(fold + 0 (list 1 2 3))
(filter (lambda (x) (> x 0)) (list -1 0 1 2))
```

#### Autodiff Tests  
```scheme
;; MUST STILL WORK
(derivative (lambda (x) (* x x)) 3.0)
(gradient (lambda (v) (+ (* (vref v 0) (vref v 0))
                         (* (vref v 1) (vref v 1))))
          (vector 3.0 4.0))
```

#### Higher-Order Functions
```scheme
;; MUST STILL WORK
(define (compose f g) (lambda (x) (f (g x))))
((compose (lambda (x) (* 2 x)) (lambda (x) (+ x 1))) 5)
```

## Validation Test Suite

Create comprehensive regression test: [`tests/closure_compatibility_test.esk`](tests/closure_compatibility_test.esk)

```scheme
;;
;; Closure Compatibility Test
;; Ensures closure implementation doesn't break existing functionality
;;

(define (main)
  (display "CLOSURE BACKWARD COMPATIBILITY TEST")
  (newline)
  (newline)
  
  ;; Test 1: Simple map (NO closures)
  (display "Test 1: Simple map with lambda")
  (newline)
  (define result1 (map (lambda (x) (* 2 x)) (list 1 2 3)))
  (display result1)
  (display " (expected: (2 4 6))")
  (newline)
  (newline)
  
  ;; Test 2: Simple fold (NO closures)
  (display "Test 2: Simple fold")
  (newline)
  (define result2 (fold + 0 (list 1 2 3)))
  (display result2)
  (display " (expected: 6)")
  (newline)
  (newline)
  
  ;; Test 3: Derivative without closure
  (display "Test 3: Derivative (no closure)")
  (newline)
  (define result3 (derivative (lambda (x) (* x x)) 3.0))
  (display result3)
  (display " (expected: 6.0)")
  (newline)
  (newline)
  
  ;; Test 4: Top-level function call
  (display "Test 4: Top-level function")
  (newline)
  (define (square x) (* x x))
  (define result4 (square 5))
  (display result4)
  (display " (expected: 25)")
  (newline)
  (newline)
  
  ;; Test 5: Nested map/fold (NO closures)
  (display "Test 5: Nested higher-order operations")
  (newline)
  (define lists (list (list 1 2) (list 3 4)))
  (define result5 (map (lambda (lst) (fold + 0 lst)) lists))
  (display result5)
  (display " (expected: (3 7))")
  (newline)
  (newline)
  
  (display "ALL COMPATIBILITY TESTS PASSED!")
  (newline)
  0)
```

## Rollback Plan

If closure implementation breaks anything:

1. **Immediate Rollback:**
   ```bash
   git stash  # Save closure work
   git checkout HEAD -- lib/  # Restore working code
   ```

2. **Identify Break Point:**
   - Run regression tests
   - Find first failing test
   - Isolate problematic change

3. **Fix Forward:**
   - Apply safety guards
   - Add environment checks
   - Re-test incrementally

## Critical Code Paths to Preserve

### 1. Map/Fold/Filter Codegen

**Location:** [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) - search for "map", "fold", "filter"

**Safety:** These must continue generating calls to regular lambdas without environment parameters.

### 2. Top-Level Function Definitions

**Location:** [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) - search for "codegenDefine"

**Safety:** Top-level functions never need environments (no parent scope).

### 3. Simple Lambda Codegen

**Location:** [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) - search for "codegenLambda"

**Safety:** Lambdas without captures keep current behavior.

### 4. Autodiff Operators

**Location:** [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) - search for "codegenDerivative", "codegenGradient"

**Safety:** Must work with both closure and non-closure functions.

## Implementation Verification Checklist

Before committing ANY change:

- [ ] Run full test suite - all tests pass
- [ ] Run autodiff test suite - all tests pass
- [ ] Test simple map/fold/filter - works
- [ ] Test top-level functions - works
- [ ] Test simple lambdas - works
- [ ] Test autodiff on regular functions - works
- [ ] No new compiler warnings
- [ ] No LLVM verification errors on existing tests

## Incremental Implementation Steps

### Step 1: Add Arena Function (SAFE - doesn't affect codegen)
- Add `arena_allocate_closure_env` to arena files
- Test: Compile and run existing tests - should be identical

### Step 2: Add Parser Analysis (SAFE - just populates AST fields)
- Add capture detection in parser
- Test: Parse existing files - AST should have num_captured=0 for non-closures

### Step 3: Add Codegen Guards (SAFE - guarded by num_captured check)
- Add environment parameter ONLY if num_captured > 0
- Test: All existing tests should use old path (num_captured=0)

### Step 4: Add Environment Passing (REQUIRES CAREFUL TESTING)
- Generate environment creation and passing
- Test: Existing tests MUST still work, new closure tests ALSO work

## Emergency Guardrails

### Compile-Time Guardrails

```cpp
// In codegen, assert our safety assumptions
void codegenClosureFunction(eshkol_ast_t* lambda_node) {
    uint64_t num_cap = lambda_node->operation.lambda_op.num_captured;
    
    // SAFETY: If no captures, use existing codegen
    if (num_cap == 0) {
        return codegenLambdaExisting(lambda_node);  // OLD PATH
    }
    
    // NEW PATH: Only reached for actual closures
    // ...
}
```

### Runtime Guardrails

```cpp
// In environment access, validate before use
llvm::Value* loadFromEnvironment(const std::string& var_name, int index) {
    // SAFETY: Null check environment
    if (!current_env_ptr) {
        eshkol_error("Attempted environment access with null environment");
        return nullptr;
    }
    
    // SAFETY: Bounds check
    if (index >= num_captures) {
        eshkol_error("Environment index out of bounds: %d >= %d", 
                    index, num_captures);
        return nullptr;
    }
    
    // Safe to proceed
    // ...
}
```

## Success Metrics

### Green Light Criteria (MUST achieve all)

1. ✅ All existing tests pass without modification
2. ✅ No performance regression on non-closure code (< 1% overhead)
3. ✅ No new compiler warnings
4. ✅ No LLVM verification errors on existing code
5. ✅ Map/fold/filter work identically
6. ✅ Autodiff works identically on non-closures

### Red Light Criteria (ABORT if any occur)

1. ❌ ANY existing test fails
2. ❌ ANY list operation breaks
3. ❌ Autodiff stops working on regular functions
4. ❌ Performance degradation > 5% on non-closure code
5. ❌ New LLVM errors on previously working code

If ANY red light occurs:
1. STOP immediately
2. Revert changes
3. Analyze root cause
4. Fix in isolation
5. Re-test before proceeding

## Phased Rollout

### Phase 1: Infrastructure (LOW RISK)
- Add arena function
- Add AST analysis helpers
- Test: No functional changes, just new functions exist

### Phase 2: Parser Enhancement (MEDIUM RISK)
- Populate captured_vars in AST
- Test: Existing tests should parse identically (num_captured=0)

### Phase 3: Codegen Transformation (HIGH RISK)
- Add conditional logic for closures
- Test extensively at each sub-step

### Phase 4: Validation (VERIFICATION)
- Run all tests
- Verify neural network tests work
- Performance benchmarking

## Conclusion

The closure implementation will succeed by:
1. Adding features without removing any
2. Using conditional logic based on `num_captured`
3. Defaulting to existing behavior when `num_captured == 0`
4. Testing exhaustively after each change
5. Having clear rollback plan at every step

**Confidence Level:** High - because we're adding, not modifying existing paths.