# Gradient Resolution Systematic Fix Plan

## Executive Summary

**Problem**: `(gradient test-func (vector 1.0 2.0))` fails with "Failed to resolve function for gradient computation"  
**Root Cause**: `resolveLambdaFunction()` returns nullptr for global lambda variable references  
**Timeline**: 4-6 hours systematic fix with comprehensive validation  
**Risk**: Low (targeted debugging with rollback strategy)

---

## Problem Analysis

### Failing Test (Test 8 in test_let_and_lambda.esk)

```scheme
;; This FAILS:
(define test-func (lambda (v) 0.0))
(gradient test-func (vector 1.0 2.0))
```

**Debug Output**:
```
DEBUG: resolveLambdaFunction returned: 0x0
Failed to resolve function for gradient computation
```

### Working Test (Test 7)

```scheme
;; This WORKS:
(gradient (lambda (v) 0.0) (vector 1.0 2.0))
```

### Code Path Analysis

1. **Storage** ([`llvm_codegen.cpp:3703-3711`](lib/backend/llvm_codegen.cpp:3703)):
   ```cpp
   // In codegenVariableDefinition for global lambda:
   symbol_table[std::string(var_name) + "_func"] = func;
   global_symbol_table[std::string(var_name) + "_func"] = func;
   ```
   ✅ **CONFIRMED**: Storage happens correctly

2. **Retrieval** ([`llvm_codegen.cpp:12840-12850`](lib/backend/llvm_codegen.cpp:12840)):
   ```cpp
   // In resolveLambdaFunction:
   func_it = global_symbol_table.find(func_name + "_func");
   if (func_it != global_symbol_table.end() && func_it->second) {
       if (isa<Function>(func_it->second)) {
           return func_it->second;
       }
   }
   ```
   ❌ **SUSPECTED**: Retrieval fails somehow

### Hypothesis

**Possible causes** (in priority order):

1. **Timing Issue**: Global lambda stored in `global_symbol_table` AFTER `gradient` call tries to resolve it
   - Global variables processed in `main()` context
   - Gradient might execute before storage completes
   
2. **Scope Issue**: `test-func_func` stored in different scope/context than where gradient looks
   - Symbol tables might be cleared/restored between storage and use
   
3. **Type Issue**: Value stored is not actually a `Function*`
   - `isa<Function>` check might be failing
   
4. **Key Mismatch**: Storage uses different key than retrieval
   - Unlikely given identical code for key construction

---

## Systematic Debug Strategy

### Phase 1: Enhanced Logging (30 min)

**Location**: [`llvm_codegen.cpp:3621-3756`](lib/backend/llvm_codegen.cpp:3621)

Add comprehensive logging in `codegenVariableDefinition`:

```cpp
// After line 3710:
eshkol_error("STORAGE DEBUG: Stored global lambda %s_func -> %p in both symbol tables",
            var_name, func);
eshkol_error("STORAGE DEBUG: symbol_table size = %zu, global_symbol_table size = %zu",
            symbol_table.size(), global_symbol_table.size());
```

**Location**: [`llvm_codegen.cpp:12795-12937`](lib/backend/llvm_codegen.cpp:12795)

Add comprehensive logging in `resolveLambdaFunction`:

```cpp
// After line 12816:
eshkol_error("RESOLVE DEBUG: Looking for %s", func_name.c_str());
eshkol_error("RESOLVE DEBUG: symbol_table size = %zu, global_symbol_table size = %zu",
            symbol_table.size(), global_symbol_table.size());

// After line 12826 (symbol_table lookup):
if (func_it != symbol_table.end()) {
    eshkol_error("RESOLVE DEBUG: Found %s_func in symbol_table", func_name.c_str());
    eshkol_error("RESOLVE DEBUG: Value pointer = %p", func_it->second);
    eshkol_error("RESOLVE DEBUG: Is Function? %d", isa<Function>(func_it->second));
} else {
    eshkol_error("RESOLVE DEBUG: NOT found %s_func in symbol_table", func_name.c_str());
}

// After line 12840 (global_symbol_table lookup):
if (func_it != global_symbol_table.end()) {
    eshkol_error("RESOLVE DEBUG: Found %s_func in global_symbol_table", func_name.c_str());
    eshkol_error("RESOLVE DEBUG: Value pointer = %p", func_it->second);
    eshkol_error("RESOLVE DEBUG: Is Function? %d", isa<Function>(func_it->second));
} else {
    eshkol_error("RESOLVE DEBUG: NOT found %s_func in global_symbol_table", func_name.c_str());
    // Print all keys in global_symbol_table for debugging
    eshkol_error("RESOLVE DEBUG: global_symbol_table contents:");
    for (auto& entry : global_symbol_table) {
        eshkol_error("  key: %s, value: %p", entry.first.c_str(), entry.second);
    }
}
```

### Phase 2: Root Cause Identification (1 hour)

**Test Execution**:
```bash
cmake --build build
./build/eshkol-run tests/autodiff/test_let_and_lambda.esk 2>&1 | grep -A5 "Test 8"
```

**Expected Debug Output** (will reveal actual issue):
```
STORAGE DEBUG: Stored global lambda test-func_func -> 0x... in both symbol tables
STORAGE DEBUG: symbol_table size = X, global_symbol_table size = Y

... (later when gradient executes) ...

RESOLVE DEBUG: Looking for test-func
RESOLVE DEBUG: symbol_table size = X, global_symbol_table size = Y
RESOLVE DEBUG: NOT found test-func_func in symbol_table  <-- Or found but wrong type
RESOLVE DEBUG: NOT found test-func_func in global_symbol_table  <-- Issue here!
RESOLVE DEBUG: global_symbol_table contents:
  key: ..., value: 0x...
  (test-func_func should be in this list!)
```

**Analysis Questions**:
1. Is `test-func_func` in the printed list?
   - YES → Type/value issue
   - NO → Storage/scope issue

2. When is storage happening vs when is resolution happening?
   - Check timestamps/order of debug messages

3. Are symbol tables being cleared between storage and use?
   - Check for symbol_table restoration code

### Phase 3: Implement Fix (2 hours)

**Fix A: Timing Issue** (if global variable processed too late)

Ensure global lambda definitions are processed BEFORE other expressions in main:

Location: [`llvm_codegen.cpp:272-278`](lib/backend/llvm_codegen.cpp:272)

```cpp
// Current code processes non-function defines
// FIX: Process lambda definitions FIRST, then other globals, then expressions

// Step 1: Process global lambda definitions
for (size_t i = 0; i < num_asts; i++) {
    if (asts[i].type == ESHKOL_OP && asts[i].operation.op == ESHKOL_DEFINE_OP &&
        !asts[i].operation.define_op.is_function &&
        asts[i].operation.define_op.value &&
        asts[i].operation.define_op.value->type == ESHKOL_OP &&
        asts[i].operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP) {
        // Process lambda variable definitions first
        codegenAST(&asts[i]);
    }
}

// Step 2: Process other global variables
for (size_t i = 0; i < num_asts; i++) {
    if (asts[i].type == ESHKOL_OP && asts[i].operation.op == ESHKOL_DEFINE_OP &&
        !asts[i].operation.define_op.is_function &&
        !(asts[i].operation.define_op.value &&
          asts[i].operation.define_op.value->type == ESHKOL_OP &&
          asts[i].operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP)) {
        // Process non-lambda global variables
        codegenAST(&asts[i]);
    }
}
```

**Fix B: Scope Clearing Issue** (if symbol_table gets cleared)

Check if main function's symbol_table scope is being cleared.

Location: Look for `symbol_table = prev_symbols` in main function path

**Fix C: Direct Lookup Enhancement** (defensive fix regardless of root cause)

Location: [`llvm_codegen.cpp:12795`](lib/backend/llvm_codegen.cpp:12795)

Add fallback to search for variable itself (not just _func):

```cpp
// After line 12850 (if global_symbol_table lookup fails):
if (!func_it->second || !isa<Function>(func_it->second)) {
    // Fallback: Look for the variable itself (test-func)
    // It might be an AllocaInst or GlobalVariable containing function pointer
    auto var_it = global_symbol_table.find(func_name);
    if (var_it == global_symbol_table.end()) {
        var_it = symbol_table.find(func_name);
    }
    
    if (var_it != symbol_table.end() && var_it->second) {
        Value* var_value = var_it->second;
        
        // If it's a GlobalVariable containing a function pointer
        if (isa<GlobalVariable>(var_value)) {
            GlobalVariable* global_var = dyn_cast<GlobalVariable>(var_value);
            if (global_var->getValueType()->isIntegerTy(64)) {
                // This is the function pointer as i64
                // Need to find the actual function by searching function_table
                // Look for lambda with matching name in variable definition
                
                // Try _func entry one more time with full debug
                auto retry_func_it = global_symbol_table.find(func_name + "_func");
                if (retry_func_it != global_symbol_table.end() && retry_func_it->second) {
                    return retry_func_it->second;
                }
            }
        }
    }
}
```

### Phase 4: Test Neural Network Updates (1.5 hours)

Update all neural network tests to use `let` + lambda instead of nested `define`:

**Files to Update**:
- `tests/neural/nn_training.esk`
- `tests/neural/nn_computation.esk`  
- `tests/neural/nn_simple.esk` (if needed)

**Pattern Replacement**:

OLD (nested define - NOT SUPPORTED):
```scheme
(define (train-step w b lr x y)
  (define (loss-w w-val)
    (mse w-val b x y))
  (derivative loss-w w))
```

NEW (let + lambda - SUPPORTED):
```scheme
(define (train-step w b lr x y)
  (let ((loss-w (lambda (w-val)
                  (mse w-val b x y))))
    (derivative loss-w w)))
```

### Phase 5: Validation (1 hour)

**Test Sequence**:

1. **Compile**:
   ```bash
   cmake --build build
   ```

2. **Test 8 specifically**:
   ```bash
   ./build/eshkol-run tests/autodiff/test_let_and_lambda.esk 2>&1 | grep -A10 "Test 8"
   ```
   Expected: Test 8 passes, displays `#(0 0)` or correct gradient

3. **All autodiff tests**:
   ```bash
   ./scripts/run_autodiff_tests.sh
   ```
   Expected: All pass

4. **Neural network tests**:
   ```bash
   for f in tests/neural/*.esk; do
       echo "Testing $f"
       ./build/eshkol-run "$f"
       echo "---"
   done
   ```
   Expected: All tests run successfully

5. **Full regression**:
   ```bash
   ./scripts/run_all_tests.sh
   ```
   Expected: 110+ tests pass

---

## Implementation Checklist

### Hour 1: Enhanced Debugging
- [ ] Add storage debug logs in `codegenVariableDefinition`
- [ ] Add retrieval debug logs in `resolveLambdaFunction`
- [ ] Compile and run Test 8
- [ ] Analyze debug output to identify root cause

### Hour 2: Root Cause Fix
- [ ] Implement appropriate fix based on debug findings
- [ ] Add defensive fallback lookup regardless of root cause
- [ ] Compile and verify no LLVM errors
- [ ] Test Test 8 passes

### Hour 3: Neural Network Test Updates
- [ ] Update `nn_training.esk` to use let + lambda
- [ ] Update `nn_computation.esk` to use let + lambda
- [ ] Update any other affected tests
- [ ] Verify all neural tests compile and run

### Hour 4: Validation & Testing
- [ ] Run all autodiff tests
- [ ] Run all neural network tests
- [ ] Run full test suite
- [ ] Fix any regressions

### Hour 5-6: Documentation & Cleanup
- [ ] Remove debug logging (or convert to eshkol_debug)
- [ ] Document the fix in CLOSURE_IMPLEMENTATION_STATUS.md
- [ ] Create neural network status report
- [ ] Update v1.0-foundation status

---

## Success Criteria

**Must Pass**:
- [ ] Test 8 in test_let_and_lambda.esk passes
- [ ] `(gradient test-func (vector 1.0 2.0))` works
- [ ] All autodiff tests continue to pass
- [ ] At least 3/5 neural network tests pass
- [ ] No regressions in existing 110+ tests

**Nice to Have**:
- [ ] All 5 neural network tests pass
- [ ] Performance within 10% of baseline
- [ ] Clean, minimal code changes

---

## Rollback Strategy

If systematic fix takes too long or causes regressions:

1. **Revert debug logging** (remove eshkol_error calls)
2. **Update neural network tests only** (skip gradient fix)
3. **Document limitation**: Gradient with global lambda references not supported in v1.0
4. **Defer fix to v1.1**

---

## Known Limitations (Post-Fix)

Even after fix, these won't work in v1.0:

1. **Nested `define`**: Not supported (architectural decision)
   ```scheme
   (define (f)
     (define (g) ...)  ; ERROR
     ...)
   ```
   **Solution**: Use `let` + lambda

2. **Mutable captures**: Not implemented
   ```scheme
   (define n 5)
   (set! n 10)  ; Not supported
   ```

3. **Runtime function construction**: Limited
   - Most gradient/jacobian operations need compile-time function knowledge

---

## Next Steps After Fix

1. **Test comprehensive neural network examples**
2. **Benchmark gradient performance**
3. **Add more neural network primitives** (activation functions, optimizers)
4. **Consider runtime autodiff** for v1.1 (if needed)

---

## Timeline

| Hour | Task | Deliverable |
|------|------|-------------|
| 0-1 | Enhanced debugging | Debug output reveals root cause |
| 1-2 | Implement fix | Test 8 passes |
| 2-3 | Update NN tests | All tests use proper syntax |
| 3-4 | Validation | All tests pass |
| 4-5 | Documentation | Status report complete |
| 5-6 | Buffer | Handle any issues |

Total: 4-6 hours