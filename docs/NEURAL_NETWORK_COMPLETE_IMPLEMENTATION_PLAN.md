# Neural Network Complete Implementation Plan

## Mission: Make ALL Neural Network Tests Pass

**Goal**: All 5 tests in `tests/neural/` run successfully  
**Timeline**: 6-8 hours of focused work  
**Blockers Identified**: 4 critical issues  

---

## Current Status

### ✅ What Works (Foundation Complete)
- Gradient computation with global lambda refs (just fixed!)
- Closures with autodiff
- List operations (map, fold, filter)
- Mixed-type arithmetic
- **nn_minimal.esk passes completely**

### ❌ What's Blocking (Must Fix)

| Issue | Affects | Priority | Time |
|-------|---------|----------|------|
| Nested `define` | 3/5 tests | **CRITICAL** | 2-3h |
| `let*` | 2/5 tests | **HIGH** | 2-3h |
| Derivative type handling | 1/5 tests | **MEDIUM** | 1h |
| Missing stdlib (tanh, range, etc.) | 2/5 tests | **MEDIUM** | 1-2h |
| Segfault in nn_simple.esk | 1/5 tests | **HIGH** | 1h |

---

## PRIORITY 1: Nested `define` Support (2-3 hours)

### Problem

```scheme
(define (train-step w)
  (define (loss-fn w-val)  ; ❌ ERROR: "nested define not supported"
    (* w-val w-val))
  (derivative loss-fn w))
```

### Root Cause

[`llvm_codegen.cpp:3512-3515`](lib/backend/llvm_codegen.cpp:3512):
```cpp
if (!function) {
    eshkol_error("Function %s not found in function_table (nested define not supported - use let+lambda instead)", func_name);
    return nullptr;
}
```

The error happens because nested functions aren't added to `function_table` before their parent function is fully processed.

### Solution Architecture

**Approach**: Allow function definitions inside function bodies by treating them as local functions.

**Implementation Steps**:

#### Step 1: Remove Error Check (5 min)

**File**: `lib/backend/llvm_codegen.cpp` line 3512

**Change**:
```cpp
// OLD:
if (!function) {
    eshkol_error("Function %s not found in function_table (nested define not supported - use let+lambda instead)", func_name);
    return nullptr;
}

// NEW:
if (!function) {
    // Nested function - create it now in parent function context
    eshkol_debug("Creating nested function: %s", func_name);
    
    // Create function declaration
    createFunctionDeclaration(ast);
    
    // Retrieve newly created function
    function = function_table[func_name];
    
    if (!function) {
        eshkol_error("Failed to create nested function: %s", func_name);
        return nullptr;
    }
}
```

#### Step 2: Store Nested Function Reference (10 min)

After function body generation, register the nested function:

```cpp
// After line 3589 (after global_symbol_table registration):
// Also register in parent function's symbol table
if (prev_function) {
    // We're in a nested function context
    symbol_table[std::string(func_name)] = function;
    symbol_table[std::string(func_name) + "_func"] = function;
    eshkol_debug("Registered nested function %s in parent scope", func_name);
}
```

#### Step 3: Test (15 min)

```bash
# Test with simple nested define:
cat > test_nested_define.esk << 'EOF'
(define (outer x)
  (define (inner y)
    (+ x y))
  (inner 10))

(display (outer 5))  ; Should print 15
(newline)
EOF

./build/eshkol-run test_nested_define.esk && ./a.out
```

Expected: `15`

---

## PRIORITY 2: `let*` Implementation (2-3 hours)

### Problem

```scheme
(let* ((w1 0.5)
       (grad (derivative loss w1))   ; Can use w1!
       (w2 (- w1 (* lr grad))))      ; Can use w1 AND grad!
  w2)
```

Currently fails with parse error (operator not recognized).

### Solution Architecture

**`let*`** = Sequential `let` where each binding sees previous bindings.

**Implementation Steps**:

#### Step 1: Add Parser Support (30 min)

**File**: `lib/frontend/parser.cpp`

**Find**: `get_operator_type` function (~line 195-219)

**Add**:
```cpp
if (op == "let*") return ESHKOL_LET_STAR_OP;
```

**File**: `inc/eshkol/eshkol.h`

**Find**: Enum definitions for operators

**Add**:
```cpp
ESHKOL_LET_STAR_OP,  // Add after ESHKOL_LET_OP
```

#### Step 2: Add AST Structure (30 min)

**File**: `inc/eshkol/eshkol.h`

**Add** (similar to `let_op`):
```cpp
typedef struct {
    eshkol_ast_t* bindings;      // Array of (var . value) pairs
    uint64_t num_bindings;
    eshkol_ast_t* body;          // Body expression
} eshkol_let_star_op_t;
```

**Update union**:
```cpp
typedef union {
    // ... existing ops ...
    eshkol_let_star_op_t let_star_op;
} eshkol_operations_union_t;
```

#### Step 3: Add Parser Logic (45 min)

**File**: `lib/frontend/parser.cpp`

**Find**: Where `ESHKOL_LET_OP` is parsed (~line 620-650)

**Add similar code** for `ESHKOL_LET_STAR_OP`:
```cpp
case ESHKOL_LET_STAR_OP: {
    // Parse like let, but bindings are sequential
    // (same structure, different codegen semantics)
    // ... copy let parsing logic ...
}
```

#### Step 4: Add Codegen (60 min)

**File**: `lib/backend/llvm_codegen.cpp`

**Add** to `codegenOperation()`:
```cpp
case ESHKOL_LET_STAR_OP:
    return codegenLetStar(op);
```

**Implement `codegenLetStar()`**:
```cpp
Value* codegenLetStar(const eshkol_operations_t* op) {
    if (!op || !op->let_star_op.body) {
        eshkol_error("Invalid let* expression");
        return nullptr;
    }
    
    // KEY DIFFERENCE from let: DON'T save symbol table at start
    // Each binding updates symbol_table immediately, so next binding sees it
    
    // Process bindings SEQUENTIALLY (not in parallel like let)
    for (uint64_t i = 0; i < op->let_star_op.num_bindings; i++) {
        const eshkol_ast_t* binding = &op->let_star_op.bindings[i];
        
        // Extract variable and value (same as let)
        const eshkol_ast_t* var_ast = binding->cons_cell.car;
        std::string var_name = var_ast->variable.id;
        
        // Evaluate value IN CURRENT CONTEXT (sees previous bindings!)
        Value* val = codegenAST(binding->cons_cell.cdr);
        
        // Store immediately in symbol_table
        if (current_function) {
            Type* storage_type = val->getType();
            
            // Handle lambdas specially (same as let)
            if (val && isa<Function>(val)) {
                Function* func = dyn_cast<Function>(val);
                storage_type = Type::getInt64Ty(*context);
                symbol_table[var_name + "_func"] = func;
                global_symbol_table[var_name + "_func"] = func;
                val = builder->CreatePtrToInt(func, storage_type);
            }
            
            AllocaInst* var_alloca = builder->CreateAlloca(storage_type, nullptr, var_name.c_str());
            if (storage_type->isIntegerTy(64)) {
                var_alloca->setAlignment(Align(8));
            }
            builder->CreateStore(val, var_alloca);
            symbol_table[var_name] = var_alloca;  // IMMEDIATE update!
        }
    }
    
    // Evaluate body with all bindings visible
    Value* body_result = codegenAST(op->let_star_op.body);
    
    // DON'T restore symbol_table (bindings stay visible)
    // Actually, DO restore for proper scoping
    // BUT: Keep _func entries like regular let
    
    return body_result ? body_result : ConstantInt::get(Type::getInt64Ty(*context), 0);
}
```

#### Step 5: Test (15 min)

```scheme
(let* ((a 1)
       (b (+ a 2))    ; Uses a
       (c (* b 3)))   ; Uses b
  c)  ; Should be 9
```

---

## PRIORITY 3: Fix Derivative Type Handling (1 hour)

### Problem

From `nn_computation.esk`:
```
error: derivative point must be numeric (int64 or double)
```

### Root Cause

**File**: `lib/backend/llvm_codegen.cpp` lines 8869-8875

```cpp
// Convert x to double if it's an integer
if (x->getType()->isIntegerTy()) {
    x = builder->CreateSIToFP(x, Type::getDoubleTy(*context));
} else if (!x->getType()->isDoubleTy()) {  // <-- BUG: doesn't handle tagged_value!
    eshkol_error("derivative point must be numeric (int64 or double)");
    return nullptr;
}
```

### Solution

```cpp
// Convert x to double if needed
if (x->getType() == tagged_value_type) {
    // Unpack from tagged_value first
    Value* x_type = getTaggedValueType(x);
    Value* x_base = builder->CreateAnd(x_type,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    Value* x_is_double = builder->CreateICmpEQ(x_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    
    // Branch to unpack correctly
    Function* current_func = builder->GetInsertBlock()->getParent();
    BasicBlock* unpack_double = BasicBlock::Create(*context, "deriv_unpack_double", current_func);
    BasicBlock* unpack_int = BasicBlock::Create(*context, "deriv_unpack_int", current_func);
    BasicBlock* deriv_continue = BasicBlock::Create(*context, "deriv_continue", current_func);
    
    builder->CreateCondBr(x_is_double, unpack_double, unpack_int);
    
    builder->SetInsertPoint(unpack_double);
    Value* x_double = unpackDoubleFromTaggedValue(x);
    builder->CreateBr(deriv_continue);
    
    builder->SetInsertPoint(unpack_int);
    Value* x_int = unpackInt64FromTaggedValue(x);
    Value* x_double_from_int = builder->CreateSIToFP(x_int, Type::getDoubleTy(*context));
    builder->CreateBr(deriv_continue);
    
    builder->SetInsertPoint(deriv_continue);
    PHINode* x_phi = builder->CreatePHI(Type::getDoubleTy(*context), 2);
    x_phi->addIncoming(x_double, unpack_double);
    x_phi->addIncoming(x_double_from_int, unpack_int);
    x = x_phi;
} else if (x->getType()->isIntegerTy()) {
    x = builder->CreateSIToFP(x, Type::getDoubleTy(*context));
} else if (!x->getType()->isDoubleTy()) {
    eshkol_error("derivative point must be numeric (int64 or double)");
    return nullptr;
}
```

---

## PRIORITY 4: Add Missing Stdlib Functions (1-2 hours)

### C Library Interop Strategy

You're absolutely right - we should leverage C interop! Functions like `tanh`, `range`, `zip`, `random` can be:
1. Declared as external C functions (`extern`)
2. Implemented in C and linked
3. OR implemented in Eshkol stdlib

### Implementation

#### Built-in Math Functions (30 min)

**`tanh` is already in libm!** Just declare it:

**File**: `lib/backend/llvm_codegen.cpp` in `createBuiltinFunctions()`

Add after `pow_func`:
```cpp
// tanh function declaration (from libm)
std::vector<Type*> tanh_args;
tanh_args.push_back(Type::getDoubleTy(*context));

FunctionType* tanh_type = FunctionType::get(
    Type::getDoubleTy(*context),
    tanh_args,
    false
);

Function* tanh_func = Function::Create(
    tanh_type,
    Function::ExternalLinkage,
    "tanh",
    module.get()
);

function_table["tanh"] = tanh_func;
```

#### Eshkol Stdlib Functions (60 min)

**File**: Create `lib/stdlib/stdlib.esk` (NEW)

```scheme
;; Eshkol Standard Library
;; Functions that should be available by default

;; Range: Create list from start to end (exclusive)
(define (range start end)
  (define (range-helper current end acc)
    (if (>= current end)
        (reverse acc)
        (range-helper (+ current 1) end (cons current acc))))
  (range-helper start end '()))

;; Zip: Combine two lists into list of pairs
(define (zip list1 list2)
  (if (or (null? list1) (null? list2))
      '()
      (cons (cons (car list1) (car list2))
            (zip (cdr list1) (cdr list2)))))

;; Random: Simple pseudo-random (needs C backend for real RNG)
(define (random)
  0.5)  ; Placeholder - real implementation needs C

;; Fold-left alias (some tests use foldl)
(define foldl fold)

;; Reduce (same as fold but different arg order in some Schemes)
(define (reduce fn init lst)
  (fold fn init lst))
```

**Auto-load**: Modify compiler to preload stdlib before user code.

---

## PRIORITY 5: Debug Segfault in nn_simple.esk (1 hour)

### Investigation Strategy

1. **Add Safety Checks** before the segfault point
2. **Run under debugger** to find exact crash location
3. **Likely causes**:
   - Null pointer dereference in list operations
   - Bad cons cell access
   - Type mismatch in polymorphic operations

### Debug Approach

```bash
# Run with verbose debugging
./build/eshkol-run tests/neural/nn_simple.esk 2>&1 | grep -B5 "Segmentation"

# If that doesn't help, run under lldb:
lldb ./a.out
(lldb) run
(lldb) bt  # When it crashes
```

---

## Detailed Implementation Timeline

### Day 1 Session (6-8 hours)

#### Hour 1: Nested `define` Support
- [ ] Remove error check in `codegenFunctionDefinition`
- [ ] Add logic to create function on-demand
- [ ] Register nested functions in parent scope
- [ ] Test with simple nested function
- [ ] Verify `nn_training.esk` compiles

#### Hour 2: `let*` Parser
- [ ] Add `ESHKOL_LET_STAR_OP` to enums
- [ ] Add AST structure
- [ ] Implement parser logic
- [ ] Test parsing (no codegen yet)

#### Hour 3: `let*` Codegen
- [ ] Implement `codegenLetStar()`
- [ ] Handle sequential binding visibility
- [ ] Test with sequential bindings
- [ ] Verify works with autodiff

#### Hour 4: Derivative Type Handling
- [ ] Fix tagged_value unpacking in `codegenDerivative`
- [ ] Add proper type detection branching
- [ ] Test `nn_computation.esk` derivatives
- [ ] Verify no regressions

#### Hour 5: Stdlib Functions
- [ ] Add `tanh` to builtin math functions
- [ ] Create `lib/stdlib/stdlib.esk`
- [ ] Implement `range`, `zip` in Eshkol
- [ ] Auto-load stdlib before user code

#### Hour 6: Debugging & Testing
- [ ] Debug `nn_simple.esk` segfault
- [ ] Test all 5 neural network files
- [ ] Fix any discovered issues
- [ ] Document working patterns

#### Hour 7-8: Validation & Cleanup
- [ ] Run full test suite (ensure no regressions)
- [ ] Remove debug logging
- [ ] Update documentation
- [ ] Create final status report

---

## Code Modifications Summary

### Files to Modify

1. **`inc/eshkol/eshkol.h`**
   - Add `ESHKOL_LET_STAR_OP` enum
   - Add `eshkol_let_star_op_t` structure

2. **`lib/frontend/parser.cpp`**
   - Add `let*` operator recognition
   - Add `let*` parsing logic

3. **`lib/backend/llvm_codegen.cpp`**
   - Fix nested `define` (remove error, create on-demand)
   - Implement `codegenLetStar()`
   - Fix derivative type handling
   - Add `tanh` to builtins

4. **`lib/stdlib/stdlib.esk`** (NEW)
   - Implement `range`, `zip`, `random` in Eshkol

5. **`exe/eshkol-run.cpp`** (MAYBE)
   - Auto-load stdlib before user code

---

## Testing Strategy

### Unit Tests (After Each Fix)

1. **Nested `define`**:
   ```scheme
   (define (f x) (define (g y) (+ x y)) (g 10))
   (f 5)  ; Should be 15
   ```

2. **`let*`**:
   ```scheme
   (let* ((a 1) (b (+ a 2)) (c (* b 3))) c)  ; Should be 9
   ```

3. **Derivative types**:
   ```scheme
   (define w 1.0)
   (derivative (lambda (x) (* x x)) w)  ; Should work
   ```

4. **Stdlib**:
   ```scheme
   (tanh 1.0)  ; Should return ~0.76
   (range 0 5)  ; Should return (0 1 2 3 4)
   ```

### Integration Tests (Final)

Run all neural network tests:
```bash
for f in tests/neural/*.esk; do
    echo "=== Testing $f ==="
    ./build/eshkol-run "$f" && ./a.out
    echo "Status: $?"
    echo ""
done
```

**Success Criteria**: All 5 tests either pass or run to completion without errors.

---

## Risk Mitigation

### Rollback Points

1. **After each priority**: Commit working state
2. **If nested `define` breaks things**: Can revert and use `let` + lambda workaround
3. **If `let*` is complex**: Can defer to v1.1 and use nested `let`

### Backup Strategy

If full implementation takes too long:
- **Minimum**: Fix nested `define` + derivative types (gets 3/5 tests working)
- **Medium**: Add `let*` too (gets 4/5 tests working)
- **Full**: All fixes (gets 5/5 tests working)

---

## Expected Outcomes

### After All Fixes

**Test Results**:
- ✅ nn_minimal.esk - Already working
- ✅ nn_computation.esk - After derivative fix
- ✅ nn_simple.esk - After segfault fix
- ✅ nn_training.esk - After nested `define` + `let*`
- ✅ nn_working.esk - After all fixes
- ⚠️ nn_complete.esk - May need additional work (complex)

**Capabilities Enabled**:
- Real gradient descent training loops
- Multi-layer perceptron implementation
- XOR problem solution
- Clean, readable neural network code

---

## Post-Implementation

### Documentation Updates

1. Update `V1_0_NEURAL_NETWORK_READY_STATUS.md`
2. Create tutorial for neural network training
3. Add examples to documentation
4. Update `NEURAL_NETWORK_IMPLEMENTATION_STATUS.md`

### Future Enhancements (v1.1+)

1. **Optimizers**: Adam, SGD with momentum
2. **Batch processing**: Mini-batch gradient descent
3. **More activations**: Leaky ReLU, ELU, Swish
4. **Loss functions**: Cross-entropy, Huber
5. **Layers**: Dense, Conv2D, BatchNorm

---

## Ready to Implement?

This plan provides:
- ✅ Clear problem identification
- ✅ Step-by-step solutions
- ✅ Code locations and modifications
- ✅ Test strategy
- ✅ Risk mitigation
- ✅ Timeline (6-8 hours)

**Recommendation**: Start fresh tomorrow, work through priorities in order, commit after each success.

**Estimated completion**: End of tomorrow you'll have ALL neural network tests passing and a robust foundation for AI/ML work in Eshkol!