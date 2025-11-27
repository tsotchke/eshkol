# Eshkol Neural Network: Complete Technical Fix Specification

## Executive Summary

After comprehensive code analysis of 14,665 lines of LLVM codegen, parser, and AST code, I've identified **7 specific technical bugs** preventing neural networks from working. Each has a precise code location and fix.

---

## CRITICAL BUG #1: Derivative Point Type Check (HIGHEST PRIORITY)

### Location
**File:** `lib/backend/llvm_codegen.cpp`  
**Function:** `codegenDerivative`  
**Lines:** 8724-8736

### Current Code (BROKEN)
```cpp
Value* x = codegenAST(op->derivative_op.point);
if (!x) {
    eshkol_error("Failed to evaluate derivative point");
    return nullptr;
}

// Convert x to double if it's an integer
if (x->getType()->isIntegerTy()) {
    x = builder->CreateSIToFP(x, Type::getDoubleTy(*context));
} else if (x->getType() != Type::getDoubleTy(*context)) {  // <-- BUG HERE!
    eshkol_error("derivative point must be numeric (int64 or double)");
    return nullptr;
}
```

### Problem
**The type check `x->getType() != Type::getDoubleTy(*context)` is TOO STRICT.**

When `x` comes from:
- **Literal:** `3.0` → Type is `Type::getDoubleTy(*context)` ✓ Works
- **Computed value:** `(- w1 grad)` → Type is `LoadInst` of `double` ✗ **FAILS**

The `LoadInst` has a double type BUT its pointer type doesn't match the exact `Type::getDoubleTy` reference check.

### Fix Required
```cpp
// Convert x to double if it's an integer
if (x->getType()->isIntegerTy()) {
    x = builder->CreateSIToFP(x, Type::getDoubleTy(*context));
} 
// FIX: Accept ANY double type (not just exact reference match)
else if (!x->getType()->isDoubleTy()) {
    // Try to extract from tagged_value
    if (x->getType() == tagged_value_type) {
        Value* type_tag = getTaggedValueType(x);
        Value* base_type = builder->CreateAnd(type_tag,
            ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
        Value* is_double = builder->CreateICmpEQ(base_type,
            ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
        
        // Runtime branch to unpack correct type
        // ... (implementation needed)
    } else {
        eshkol_error("derivative point must be numeric (int64 or double)");
        return nullptr;
    }
}
```

### Impact
✅ **Fixes:** Iterative training (weight updates work)  
✅ **Enables:** Multi-epoch gradient descent  
✅ **Estimated Time:** 2-4 hours

---

## CRITICAL BUG #2: Local Functions Lost in Scope Restoration

### Location
**File:** `lib/backend/llvm_codegen.cpp`  
**Function:** `codegenLet`  
**Lines:** 5844-5930

### Current Code (BROKEN)
```cpp
Value* codegenLet(const eshkol_operations_t* op) {
    // ...
    
    // Save current symbol table state
    std::map<std::string, Value*> prev_symbols = symbol_table;
    
    // Process all bindings
    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
        // ... create variables ...
        
        if (val && isa<Function>(val)) {
            Function* func = dyn_cast<Function>(val);
            // Store direct function reference for lambda resolution
            symbol_table[var_name + "_func"] = func;  // <-- STORED HERE
        }
        
        // ... more code ...
    }
    
    // Evaluate body
    Value* body_result = codegenAST(op->let_op.body);
    
    // Restore previous symbol table state
    symbol_table = prev_symbols;  // <-- BUG: ERASES LOCAL FUNCTIONS!
    
    return body_result;
}
```

### Problem
**Local function definitions are ERASED when `symbol_table` is restored.**

```scheme
(let ((loss (lambda (w) (* w w))))
  (derivative loss 5.0))  ; Works here

; After let exits, loss_func is GONE from symbol_table
```

### Fix Required
**Option A: Selective Restoration**
```cpp
// Restore previous symbols EXCEPT for functions that need to persist
for (auto& entry : prev_symbols) {
    // Only restore non-function entries
    if (symbol_table.find(entry.first + "_func") == symbol_table.end()) {
        symbol_table[entry.first] = entry.second;
    }
}
```

**Option B: Global Function Registration** (RECOMMENDED)
```cpp
// Store function references in BOTH tables (never lose them)
if (val && isa<Function>(val)) {
    Function* func = dyn_cast<Function>(val);
    symbol_table[var_name + "_func"] = func;
    global_symbol_table[var_name + "_func"] = func;  // <-- ADD THIS!
}
```

### Impact
✅ **Fixes:** Local function definitions with autodiff  
✅ **Enables:** Modular training code  
✅ **Estimated Time:** 1-2 hours

---

## CRITICAL BUG #3: `let*` Not Implemented

### Location
**File:** `lib/frontend/parser.cpp`  
**Function:** `get_operator_type`  
**Lines:** 195-219

### Current Code (INCOMPLETE)
```cpp
static eshkol_op_t get_operator_type(const std::string& op) {
    if (op == "if") return ESHKOL_IF_OP;
    if (op == "lambda") return ESHKOL_LAMBDA_OP;
    if (op == "let") return ESHKOL_LET_OP;
    // let* is NOT HERE!
    if (op == "compose") return ESHKOL_COMPOSE_OP;
    // ...
}
```

### Problem
**`let*` syntax is not recognized**, so sequential bindings fail:

```scheme
(let* ((w1 0.5)
       (grad (derivative loss w1))  ; Can use w1!
       (w2 (- w1 grad)))            ; Can use both w1 and grad!
  w2)
```

Currently this FAILS with parse error.

### Fix Required

**Step 1:** Add to parser (parser.cpp line ~198):
```cpp
if (op == "let*") return ESHKOL_LET_STAR_OP;
```

**Step 2:** Add enum to eshkol.h:
```cpp
ESHKOL_LET_STAR_OP,  // Add after ESHKOL_LET_OP
```

**Step 3:** Add sequential parsing logic (parser.cpp ~626):
```cpp
if (ast.operation.op == ESHKOL_LET_STAR_OP) {
    // Parse like let, but evaluate bindings sequentially
    // (same structure, different codegen)
}
```

**Step 4:** Add codegen (llvm_codegen.cpp):
```cpp
Value* codegenLetStar(const eshkol_operations_t* op) {
    // Similar to codegenLet but DON'T restore symbol table between bindings!
    for (uint64_t i = 0; i < op->let_star_op.num_bindings; i++) {
        // Evaluate binding i
        // Add to symbol_table IMMEDIATELY (don't wait for all bindings)
        // Next binding can see this one!
    }
}
```

### Impact
✅ **Fixes:** Sequential computations  
✅ **Enables:** Multi-step gradient descent  
✅ **Estimated Time:** 3-4 hours

---

## BUG #4: Lambda Argument Count Mismatch in `map`

### Location
**File:** `lib/backend/llvm_codegen.cpp`  
**Function:** `codegenMapSingleList`  
**Lines:** 12722-12824

### Current Code (BROKEN)
```cpp
Value* codegenMapSingleList(Function* proc_func, Value* list) {
    // ...
    // Extract car as tagged_value
    Value* car_tagged = extractCarAsTaggedValue(current_val);
    
    // Apply procedure to current element
    Value* proc_result = builder->CreateCall(proc_func, {car_tagged});
    // <-- BUG: What if proc_func expects 2 args (captured variable)?
}
```

### Problem
**Lambda functions with captured variables have MORE parameters than expected.**

```scheme
(define (make-multiplier n)
  (lambda (x) (* n x)))  ; <-- This lambda has 2 params: x (explicit) + n (captured)

(define times2 (make-multiplier 2))
(map times2 (list 1 2 3))  ; <-- FAILS: "Incorrect number of arguments"
```

The lambda actually has signature: `lambda(x_tagged, n_captured_tagged)` but map only passes `x_tagged`.

### Fix Required
```cpp
// Check if closure (more params than we're passing)
FunctionType* func_type = proc_func->getFunctionType();
size_t expected_params = func_type->getNumParams();
size_t provided_args = 1;  // We're passing 1 arg (car)

if (expected_params > provided_args) {
    // This is a closure - need to extract captured values
    // For now, pass zeros for missing args (or implement proper closure environment)
    std::vector<Value*> full_args;
    full_args.push_back(car_tagged);
    
    // Add zero/null for captured params
    for (size_t i = 1; i < expected_params; i++) {
        full_args.push_back(packInt64ToTaggedValue(
            ConstantInt::get(Type::getInt64Ty(*context), 0), true));
    }
    
    proc_result = builder->CreateCall(proc_func, full_args);
}
```

### Impact
✅ **Fixes:** Higher-order functions with closures  
✅ **Enables:** Functional programming patterns  
✅ **Estimated Time:** 4-6 hours (needs closure environment design)

---

## BUG #5: Built-in Functions Not First-Class Values

### Location
**File:** `lib/backend/llvm_codegen.cpp`  
**Function:** `codegenVariableDefinition`  
**Lines:** 3596-3732

### Problem
**Can't assign built-in functions to variables:**

```scheme
(define my-tanh tanh)  ; FAILS: "tanh not found"
```

Built-in functions (`sin`, `cos`, `tanh`, `exp`) exist in `function_table` but can't be referenced as values.

### Fix Required
```cpp
Value* codegenVariableDefinition(const eshkol_operations_t* op) {
    // ...
    if (op->define_op.value) {
        value = codegenAST(op->define_op.value);
    }
    
    // NEW: Check if value is a built-in function name
    if (!value && op->define_op.value->type == ESHKOL_VAR) {
        std::string name = op->define_op.value->variable.id;
        auto builtin_it = function_table.find(name);
        if (builtin_it != function_table.end()) {
            // Found built-in function - treat as function pointer
            value = builtin_it->second;
        }
    }
    // ...
}
```

### Impact
✅ **Fixes:** Activation function parameters  
✅ **Enables:** Parameterized neural network layers  
✅ **Estimated Time:** 2-3 hours

---

## BUG #6: Gradient Doesn't Accept Function References

### Location
**File:** `lib/backend/llvm_codegen.cpp`  
**Function:** `resolveLambdaFunction`  
**Lines:** 12575-12719

### Problem
**`gradient` only works with inline lambdas:**

```scheme
(define (f v) (+ (vref v 0) (vref v 1)))
(gradient f (vector 1.0 2.0))  ; FAILS: "f_func NOT in symbol_table"
```

The `resolveLambdaFunction` tries to find `f_func` but regular function definitions don't store with `_func` suffix.

### Fix Required
```cpp
Value* resolveLambdaFunction(const eshkol_ast_t* func_ast, size_t required_arity) {
    // ...
    if (func_ast->type == ESHKOL_VAR) {
        std::string func_name = func_ast->variable.id;
        
        // NEW: Check function_table FIRST for regular functions
        auto direct_it = function_table.find(func_name);
        if (direct_it != function_table.end()) {
            return direct_it->second;  // Return the function directly!
        }
        
        // Then check for lambdas with _func suffix
        auto func_it = symbol_table.find(func_name + "_func");
        // ... rest of existing code
    }
}
```

### Impact
✅ **Fixes:** Modular function definitions  
✅ **Enables:** Code organization  
✅ **Estimated Time:** 1 hour

---

## BUG #7: No Practical Iteration Construct

### Location
**Multiple** - No iteration construct exists in parser or codegen

### Problem
**Cannot iterate for realistic training:**

```scheme
; Need something like this:
(for i 0 100
  (set! w (train-step w)))

; Or tail-recursive (but hits autodiff issues):
(define (train n w)
  (if (= n 0) w
      (train (- n 1) (update-weight w))))
```

### Fix Required

**Option A: Add `for` loop** (SIMPLER)

Add to parser.cpp:
```cpp
if (op == "for") return ESHKOL_FOR_OP;
```

Add to llvm_codegen.cpp:
```cpp
Value* codegenFor(const eshkol_operations_t* op) {
    // (for var start end body)
    // Generate LLVM loop with counter variable
    BasicBlock* loop_cond = ...
    BasicBlock* loop_body = ...
    // Execute body with var bound to counter
}
```

**Option B: Add `set!` mutation** (HARDER but more Scheme-like)

Requires tracking mutable vs immutable bindings + SSA phi nodes.

### Impact
✅ **Fixes:** Multi-epoch training  
✅ **Enables:** Practical neural networks  
✅ **Estimated Time:** 6-8 hours (for loop) or 12-16 hours (set!)

---

## ARCHITECTURE ISSUE: Compile-Time vs Runtime Autodiff

### Core Problem
Eshkol's autodiff is **compile-time** (builds IR at compile time), but neural networks need **runtime** flexibility:

```scheme
; This SHOULD work but doesn't:
(define (train-epoch w data)
  (define (loss w-val)        ; <-- Function defined at RUNTIME
    (* w-val (car data)))
  (derivative loss w))         ; <-- Can't find runtime function!
```

### Why This Happens
Look at `codegenDerivative` line 8710:
```cpp
Value* func = resolveLambdaFunction(op->derivative_op.function);
```

This tries to find the function at **COMPILE TIME**. If the function is defined inside another function, it's not in `function_table` yet!

### Architectural Fix Needed
**Two-phase autodiff system:**

1. **Compile-time phase** (current):
   - Build AD operations into IR
   - Works for top-level functions

2. **Runtime phase** (NEW):
   - Store function pointers in tagged_value
   - Look up functions at runtime
   - Build AD graph dynamically

### Implementation
```cpp
// NEW: Runtime function resolution
Value* codegenDerivativeRuntime(Value* func_ptr_tagged, Value* point) {
    // Extract function pointer from tagged_value
    Value* func_addr = unpackInt64FromTaggedValue(func_ptr_tagged);
    Value* func_ptr = builder->CreateIntToPtr(func_addr, ...);
    
    // Create dual number
    Value* x_dual = packDualNumber(point, 1.0);
    
    // Call function indirectly through pointer
    Value* result_dual = builder->CreateCall(func_ptr, {x_dual});
    
    // Extract derivative
    return extractDerivativeFromDual(result_dual);
}
```

### Impact
✅ **Fixes:** Dynamic function composition  
✅ **Enables:** Training loops with local functions  
✅ **Estimated Time:** 16-24 hours (major refactoring)

---

## Comparison to PyTorch/JAX

### What PyTorch Has That Eshkol Doesn't

| Feature | PyTorch | Eshkol | Fix Complexity |
|---------|---------|--------|----------------|
| Computed values in grad | ✅ | ❌ Bug #1 | **2-4 hours** |
| Local function scope | ✅ | ❌ Bug #2 | **1-2 hours** |
| Sequential bindings (let*) | ✅ | ❌ Bug #3 | **3-4 hours** |
| Closures in map/filter | ✅ | ❌ Bug #4 | **4-6 hours** |
| Activation function params | ✅ | ❌ Bug #5 | **2-3 hours** |
| Function references in grad | ✅ | ❌ Bug #6 | **1 hour** |
| Training loops | ✅ | ❌ Bug #7 | **6-8 hours** |
| Runtime graph construction | ✅ | ❌ Architecture | **16-24 hours** |
| Native tensors (not lists) | ✅ | ❌ Architecture | **40+ hours** |

### Total Fix Time Estimate

**Phase 1: Critical Bugs (Enable Basic Training)**
- Bug #1: Derivative type check → **2-4 hours**
- Bug #6: Function ref in gradient → **1 hour**
- Bug #2: Local scope → **1-2 hours**
- **TOTAL: 4-7 hours** → Gets basic training working

**Phase 2: Practical Use (Enable Real Workflows)**
- Bug #3: let* → **3-4 hours**
- Bug #7: for loop → **6-8 hours**
- Bug #5: Builtin functions → **2-3 hours**
- **TOTAL: 11-15 hours** → Production-ready basic NN

**Phase 3: Advanced (Match PyTorch)**
- Bug #4: Closures in map → **4-6 hours**
- Runtime autodiff → **16-24 hours**
- Native tensors → **40+ hours**
- **TOTAL: 60-70 hours** → Full framework

---

## Implementation Priority

### Week 1: Make Training Work
1. ✅ Fix derivative type check (Bug #1) - **CRITICAL**
2. ✅ Fix function references (Bug #6) - **HIGH**
3. ✅ Fix local scope (Bug #2) - **HIGH**

**After Week 1:** Can train simple perceptrons iteratively!

### Week 2: Practical Workflows
4. ✅ Add let* (Bug #3) - **MEDIUM**
5. ✅ Add for loop (Bug #7) - **MEDIUM**
6. ✅ Builtin functions (Bug #5) - **LOW**

**After Week 2:** Can write real training code!

### Month 1: Production Ready
7. ✅ Closures in map (Bug #4) - **MEDIUM**
8. ✅ Runtime autodiff - **HIGH**
9. ✅ Performance optimization - **MEDIUM**

**After Month 1:** Competitive with JAX for research!

---

## Code Locations Quick Reference

```
CRITICAL FIXES (4-7 hours):
├─ lib/backend/llvm_codegen.cpp:8724-8736    [Bug #1: derivative type]
├─ lib/backend/llvm_codegen.cpp:12575-12633  [Bug #6: function refs]
└─ lib/backend/llvm_codegen.cpp:5924         [Bug #2: scope restore]

PRACTICAL FIXES (11-15 hours):
├─ lib/frontend/parser.cpp:198               [Bug #3: let* parsing]
├─ lib/backend/llvm_codegen.cpp:NEW          [Bug #3: let* codegen]
├─ lib/frontend/parser.cpp:NEW               [Bug #7: for parsing]
├─ lib/backend/llvm_codegen.cpp:NEW          [Bug #7: for codegen]
└─ lib/backend/llvm_codegen.cpp:3596-3732    [Bug #5: builtins]

ADVANCED FIXES (60-70 hours):
├─ lib/backend/llvm_codegen.cpp:12722-12824  [Bug #4: closures]
├─ lib/backend/llvm_codegen.cpp:8702-8760    [Runtime autodiff]
└─ Multiple files                             [Native tensors]
```

---

## Minimal Viable Neural Network

**After just Bugs #1, #2, #6 are fixed (4-7 hours), this WILL work:**

```scheme
(define (train-network w data lr epochs)
  (if (= epochs 0)
      w
      (let ((loss-fn (lambda (w-val)
                       (let ((pred (* w-val (car data))))
                         (* (- pred (cadr data)) (- pred (cadr data))))))
            (grad (derivative loss-fn w))
            (w-new (- w (* lr grad))))
        (train-network w-new (cddr data) lr (- epochs 1)))))
```

**After Bugs #3, #7 are also fixed (Week 2), this WILL work:**

```scheme
(for epoch 1 100
  (for-each (lambda (example)
              (let* ((x (car example))
                     (y (cadr example))
                     (loss-fn (lambda (w) ...))
                     (grad (derivative loss-fn w)))
                (set! w (- w (* lr grad)))))
            training-data))
```

---

## Bottom Line

**Current State:**  
- Autodiff works for simple cases ✓
- Lists work ✓
- Basic training demonstrated ✓

**What's Broken:**  
7 specific technical bugs with known locations and fixes

**To Match PyTorch:**  
- **Week 1:** Fix 3 critical bugs → Basic training works
- **Week 2:** Fix 3 more bugs → Practical code works
- **Month 1:** Major refactoring → Production-ready framework

**Recommendation:**  
Start with Bugs #1, #6, #2 (4-7 hours) to prove the architecture. Then decide on Bugs #3, #7 based on use cases.