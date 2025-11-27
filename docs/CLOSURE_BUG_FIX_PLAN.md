# Closure Bug Fix Plan - Complete Analysis

## Problem Summary

**Failing Test**: [`tests/autodiff/test_let_and_lambda.esk`](tests/autodiff/test_let_and_lambda.esk:46) Test 10 - Nested Lambda

```scheme
;; Test 10: Nested lambda in let (FAILS)
(let ((make-adder (lambda (n) (lambda (x) (+ x n)))))
  (let ((add5 (make-adder 5)))
    (display (add5 10))))
```

**Error Output**:
```
error: DEBUG: add5_func NOT in symbol_table, checking global
error: DEBUG: add5_func NOT in global_symbol_table either!
error: LLVM module verification failed: Basic Block in function 'lambda_4' does not have terminator!
label %add_merge
```

## Root Cause Analysis

### Issue 1: Insertion Point Corruption in `codegenLambda()`

**Location**: [`lib/backend/llvm_codegen.cpp:5765-5960`](lib/backend/llvm_codegen.cpp:5765)

**Problem**: Capture storage code runs AFTER lambda is complete and insertion point is restored.

**Current Flow** (BROKEN):
```cpp
// Line 5765: Start codegenLambda
// Line 5770-5778: Find free variables ✓
// Line 5821: Save insertion point, enter lambda
// Line 5854: Generate lambda body (may create nested lambdas!)
// Line 5877: Add return statement (LAMBDA IS DONE)
// Line 5888: Restore symbol table
// Line 5955: Restore insertion point
// Line 5899-5950: ❌ CAPTURE STORAGE CODE RUNS HERE!
//                    But insertion point is back in parent function
//                    And lambda is already terminated
//                    This corrupts IR generation for nested lambdas
```

**Why This Breaks Nested Lambdas**:
1. Outer lambda (`make-adder`) starts generating
2. Its body contains inner lambda (`(lambda (x) (+ x n))`)
3. Inner lambda generation calls `codegenLambda()` recursively
4. Inner lambda completes and returns
5. Outer lambda continues, adds its return
6. **BUG**: Outer lambda's capture storage code runs NOW
7. But inner lambda's capture storage ALSO needs to run
8. Insertion points get corrupted, creating unterminated blocks

### Issue 2: Symbol Table Lookup Failure

**Location**: [`lib/backend/llvm_codegen.cpp:3970-3990`](lib/backend/llvm_codegen.cpp:3970)

**Problem**: When `add5` (which holds the returned inner lambda) is called, the code can't find `add5_func` in symbol tables.

**Why**:
1. `make-adder` returns `lambda_1` (inner lambda)
2. `let` binds `add5` to this returned lambda value
3. `codegenLet()` stores `add5_func` → `lambda_1` mapping (lines 6009-6014)
4. But when calling `add5`, lookup fails
5. This suggests the `_func` mapping isn't being preserved correctly

## Fix Strategy

### Fix 1: Move Capture Storage to Beginning of `codegenLambda()`

**Goal**: Eliminate insertion point corruption by storing captures BEFORE creating the lambda function.

**New Flow**:
```cpp
Value* codegenLambda(const eshkol_operations_t* op) {
    static int lambda_counter = 0;
    std::string lambda_name = "lambda_" + std::to_string(lambda_counter++);
    
    // STEP 1: Find free variables (stays at line 5770-5778) ✓
    std::vector<std::string> free_vars;
    findFreeVariables(op->lambda_op.body, symbol_table, 
                     op->lambda_op.parameters, op->lambda_op.num_params, free_vars);
    
    // STEP 2: *** NEW *** Store captures IMMEDIATELY in parent function
    // This happens BEFORE lambda creation, so no insertion point issues!
    Function* parent_func = current_function;
    
    for (const std::string& var_name : free_vars) {
        auto var_it = symbol_table.find(var_name);
        if (var_it != symbol_table.end()) {
            Value* captured_val = var_it->second;
            
            // Load if it's an alloca
            if (isa<AllocaInst>(captured_val)) {
                captured_val = builder->CreateLoad(
                    dyn_cast<AllocaInst>(captured_val)->getAllocatedType(),
                    captured_val);
            }
            
            // Create alloca in PARENT function entry (before current insertion point)
            if (parent_func && !parent_func->empty()) {
                IRBuilderBase::InsertPoint saved = builder->saveIP();
                BasicBlock& entry = parent_func->getEntryBlock();
                builder->SetInsertPoint(&entry, entry.begin());
                
                std::string key = lambda_name + "_capture_" + var_name;
                Value* storage = builder->CreateAlloca(
                    captured_val->getType(), nullptr, key.c_str());
                
                builder->restoreIP(saved);
                builder->CreateStore(captured_val, storage);
                
                symbol_table[key] = storage;
                global_symbol_table[key] = storage;
            }
        }
    }
    
    // STEP 3: Create lambda function (lines 5779-5817)
    // ... existing code ...
    
    // STEP 4: Generate lambda body (lines 5820-5877)
    // ... existing code ...
    
    // STEP 5: Restore insertion point (line 5955)
    builder->restoreIP(old_point);
    
    // *** REMOVED *** No more capture code here!
    // Lines 5899-5950 DELETE completely
    
    return lambda_func;
}
```

**Benefits**:
- ✅ Captures stored in parent function BEFORE lambda creation
- ✅ No insertion point manipulation after lambda is complete
- ✅ Nested lambdas work because each level stores its captures independently
- ✅ No terminator errors because we never modify completed basic blocks

### Fix 2: Improve Symbol Table Preservation in `codegenLet()`

**Location**: [`lib/backend/llvm_codegen.cpp:6043-6063`](lib/backend/llvm_codegen.cpp:6043)

**Current Code** (lines 6043-6063):
```cpp
// CRITICAL FIX (Bug #2): Preserve _func entries before restoring symbol table
std::map<std::string, Value*> func_refs_to_preserve;
for (auto& entry : symbol_table) {
    if (entry.first.length() > 5 &&
        entry.first.substr(entry.first.length() - 5) == "_func") {
        func_refs_to_preserve[entry.first] = entry.second;
        global_symbol_table[entry.first] = entry.second;
    }
}

symbol_table = prev_symbols;

for (auto& entry : func_refs_to_preserve) {
    symbol_table[entry.first] = entry.second;
}
```

**This looks correct**, but we need to verify it's actually working for returned lambdas.

### Fix 3: Generic Capture Loading in `codegenCall()`

**Location**: [`lib/backend/llvm_codegen.cpp:4143-4254`](lib/backend/llvm_codegen.cpp:4143)

**Problem**: Current capture loading tries to find captures but uses inefficient search.

**Solution**: Use lambda parameter names to determine capture order.

```cpp
if (is_closure_call) {
    size_t num_captures = func_type->getNumParams() - op->call_op.num_vars;
    std::string lambda_name = callee->getName().str();
    
    // ROBUST: Extract capture names from lambda's parameter names
    auto arg_it = callee->arg_begin();
    
    // Skip explicit parameters
    for (size_t i = 0; i < op->call_op.num_vars && arg_it != callee->arg_end(); i++) {
        arg_it++;
    }
    
    // Load captures using parameter names from lambda itself
    for (size_t i = 0; i < num_captures && arg_it != callee->arg_end(); i++, arg_it++) {
        std::string param_name = arg_it->getName().str();
        
        // Parameter is named "captured_VARNAME", extract VARNAME
        std::string var_name = param_name;
        if (param_name.find("captured_") == 0) {
            var_name = param_name.substr(9);  // Remove "captured_" prefix
        }
        
        // Look up stored capture
        std::string capture_key = lambda_name + "_capture_" + var_name;
        
        Value* captured_val = nullptr;
        auto it = symbol_table.find(capture_key);
        if (it == symbol_table.end()) {
            it = global_symbol_table.find(capture_key);
        }
        
        if (it != symbol_table.end() && it->second) {
            Value* storage = it->second;
            
            if (isa<AllocaInst>(storage)) {
                captured_val = builder->CreateLoad(
                    dyn_cast<AllocaInst>(storage)->getAllocatedType(), storage);
            } else {
                captured_val = storage;
            }
            
            // Pack to tagged_value
            if (captured_val && captured_val->getType() != tagged_value_type) {
                if (captured_val->getType()->isIntegerTy(64)) {
                    captured_val = packInt64ToTaggedValue(captured_val, true);
                } else if (captured_val->getType()->isDoubleTy()) {
                    captured_val = packDoubleToTaggedValue(captured_val);
                } else {
                    TypedValue tv = detectValueType(captured_val);
                    captured_val = typedValueToTaggedValue(tv);
                }
            }
            
            args.push_back(captured_val);
        }
    }
}
```

## Implementation Plan

### Phase 1: Fix Insertion Point Corruption (1.5 hours)

**Files**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

**Changes**:
1. Move lines 5899-5950 (capture storage) to immediately after line 5778
2. Restructure to store captures BEFORE creating lambda function
3. Remove duplicate capture storage code from end
4. Simplify insertion point management

**Test After**:
```bash
cmake --build build && ./build/eshkol-run tests/autodiff/test_let_and_lambda.esk && ./a.out
```
Expected: Tests 1-9 still work, Test 10 should get further (maybe different error)

### Phase 2: Fix Generic Capture Loading (1 hour)

**Files**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

**Changes**:
1. Refactor lines 4143-4254 to use lambda parameter names
2. Remove hardcoded variable name searches
3. Use `callee->arg_begin()` to iterate parameters
4. Extract variable names from `captured_VARNAME` pattern

**Test After**:
```bash
cmake --build build && ./build/eshkol-run tests/autodiff/test_let_and_lambda.esk && ./a.out
```
Expected: Test 10 should work completely

### Phase 3: Verify Symbol Table Preservation (30 minutes)

**Files**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

**Changes**:
1. Add debug logging to `codegenLet()` to trace `_func` preservation
2. Verify returned lambdas get stored correctly
3. Add safety checks for null values

**Test After**:
```bash
cmake --build build && ./build/eshkol-run tests/autodiff/test_let_and_lambda.esk && ./a.out
```
Expected: All tests pass, including nested lambdas

### Phase 4: Integration Testing (1 hour)

**Tests to Run**:
1. Simple closure: `(let ((x 5)) (lambda (y) (+ x y)))`
2. Nested closure: Test 10 from failing test
3. Multiple captures: `(let ((a 1) (b 2)) (lambda (x) (+ x a b)))`
4. Closure with autodiff: `(gradient test-func (vector 1.0 2.0))`

**Validation**:
- All 110 existing tests must pass
- test_let_and_lambda.esk must pass completely
- Neural network tests should work (at least nn_minimal.esk)

## Code Modifications Required

### Modification 1: Restructure `codegenLambda()`

**Location**: Lines 5765-5960

**Before** (simplified):
```cpp
1. Find free vars
2. Create lambda function
3. Enter lambda, save insertion point
4. Generate lambda body
5. Add return
6. Restore symbol table & insertion point
7. ❌ Store captures (TOO LATE!)
```

**After** (fixed):
```cpp
1. Find free vars
2. ✓ Store captures NOW (in parent function)
3. Create lambda function with extra params
4. Enter lambda, save insertion point  
5. Generate lambda body
6. Add return
7. Restore symbol table & insertion point (DONE!)
```

### Modification 2: Simplify Capture Loading

**Location**: Lines 4143-4254

**Replace**: Complex search through candidate variable names

**With**: Generic parameter-based extraction:
- Iterate `callee->arg_begin()` starting from position `num_explicit_params`
- Extract variable name from each parameter's `getName()`
- Look up `lambda_name + "_capture_" + var_name`
- Load and pack to `tagged_value`

### Modification 3: Enhanced Debug Logging

Add comprehensive logging to track:
- When captures are stored
- When captures are loaded
- When `_func` references are created
- Symbol table state at key points

## Error Pattern Analysis

The error shows a specific sequence:

1. ✓ `make-adder` lambda created successfully
2. ✓ Inner lambda (`lambda (x)`) created successfully  
3. ✓ `test-func` global lambda works
4. ✓ Simple `let` bindings work
5. ❌ `add5` binding fails - can't find `add5_func`
6. ❌ This cascades to unterminated basic block in `lambda_4`

**Key Insight**: The problem isn't in creating lambdas, it's in **storing and retrieving returned lambda values**.

## Testing Strategy

### Test Progression

1. **Baseline** (before fix):
   - Tests 1-9: Should pass
   - Test 10: Fails with current error

2. **After Fix 1** (insertion point):
   - Tests 1-9: Must still pass
   - Test 10: Should get further or different error

3. **After Fix 2** (capture loading):
   - Tests 1-10: All should pass

4. **After Fix 3** (debug logging):
   - All tests: Pass with detailed logs

### Regression Prevention

**Before each modification**:
```bash
# Baseline working tests
./build/eshkol-run tests/autodiff/test_let_simple.esk && ./a.out
./build/eshkol-run tests/test_let_lambda_basic.esk && ./a.out
```

**After each modification**:
```bash
# Full test
./build/eshkol-run tests/autodiff/test_let_and_lambda.esk && ./a.out

# If it fails, check which test:
# - Tests 1-9 fail → broke existing functionality, REVERT
# - Test 10 fails → expected, continue
```

## Success Criteria

### Must Achieve

- [x] Tests 1-9 in test_let_and_lambda.esk pass
- [ ] Test 10 (nested lambda) passes
- [ ] No LLVM verification errors
- [ ] All 110 existing tests still pass
- [ ] No insertion point corruption warnings

### Nice to Have

- [ ] Clean debug output showing capture flow
- [ ] Performance metrics (< 5% overhead)
- [ ] Documentation updated

## Risk Assessment

### High Risk Areas

1. **`codegenLambda()` refactoring**: Touches core lambda generation
   - **Mitigation**: Make changes incrementally, test after each
   - **Rollback**: Keep original code commented for easy revert

2. **Symbol table handling**: Complex state management
   - **Mitigation**: Add extensive debug logging
   - **Rollback**: Preserve original logic paths

3. **Nested lambda recursion**: Complex control flow
   - **Mitigation**: Test with simple nested case first
   - **Rollback**: Start with single-level closures

### Low Risk Areas

1. **Debug logging**: No functional impact
2. **Test additions**: Only add, don't modify existing
3. **Documentation**: Can update after code works

## Timeline

| Phase | Duration | Task |
|-------|----------|------|
| 1 | 1.5h | Fix insertion point corruption |
| 2 | 1h | Fix generic capture loading |
| 3 | 0.5h | Verify symbol table preservation |
| 4 | 1h | Integration testing |
| **Total** | **4h** | **Complete fix** |

## Next Steps

1. Review this plan with the user
2. Get approval to proceed
3. Switch to Code mode
4. Implement Fix 1: Insertion point corruption
5. Test incrementally
6. Implement Fix 2: Generic capture loading
7. Test completely
8. Verify all existing tests pass

## References

- **Failing Test**: [`tests/autodiff/test_let_and_lambda.esk:46`](tests/autodiff/test_let_and_lambda.esk:46)
- **Bug Location**: [`lib/backend/llvm_codegen.cpp:5765-5960`](lib/backend/llvm_codegen.cpp:5765)
- **Capture Detection**: [`lib/backend/llvm_codegen.cpp:5700-5763`](lib/backend/llvm_codegen.cpp:5700)
- **Capture Loading**: [`lib/backend/llvm_codegen.cpp:4143-4254`](lib/backend/llvm_codegen.cpp:4143)
- **Design Doc**: [`docs/CLOSURE_IMPLEMENTATION_FINAL_STRATEGY.md`](docs/CLOSURE_IMPLEMENTATION_FINAL_STRATEGY.md)