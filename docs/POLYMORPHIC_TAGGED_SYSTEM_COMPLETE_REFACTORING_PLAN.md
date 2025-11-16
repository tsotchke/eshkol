# Complete Polymorphic Tagged Type System Refactoring Plan
## Eshkol v1.0-architecture - Foundation for HoTT

**Document Version:** 1.0  
**Created:** 2025-11-16  
**Status:** ARCHITECTURAL DESIGN COMPLETE - READY FOR IMPLEMENTATION  
**Scope:** Complete LLVM codegen refactoring for proper tagged type support  
**Timeline:** 3-4 weeks (60-80 hours)  
**Risk Level:** HIGH (fundamental system redesign)  
**Reward:** Clean, type-safe foundation for HoTT integration

---

## 1. PROBLEM ANALYSIS - ROOT CAUSE IDENTIFICATION

### 1.1 The Three-Layer Architecture Problem

```
┌─────────────────────────────────────────────────┐
│ Layer 1: STORAGE (C Runtime)                   │
│ ✅ Status: COMPLETE                             │
│ - 24-byte tagged cons cells                     │
│ - arena_tagged_cons_cell_t with type tags       │
│ - Helper functions: get/set int64/double/ptr    │
│ Location: lib/core/arena_memory.cpp            │
└─────────────────────────────────────────────────┘
           ↓ Type information flows down
┌─────────────────────────────────────────────────┐
│ Layer 2: ACCESS (LLVM Codegen)                  │
│ ⚠️ Status: PARTIALLY MIGRATED                   │
│ - car/cdr: ✅ Use tagged helpers (lines 2424-2580)│
│ - Higher-order: ❌ 40 CreateStructGEP sites      │
│ - Compound car/cdr: ✅ Using tagged helpers      │
│ Location: lib/backend/llvm_codegen.cpp         │
└─────────────────────────────────────────────────┘
           ↓ Values extracted with types
┌─────────────────────────────────────────────────┐
│ Layer 3: FUNCTION CALLS (Type Propagation)      │
│ ❌ Status: FUNDAMENTALLY BROKEN                  │
│ - ALL functions: int64 params (line 694-700)    │
│ - Builtin arithmetic: int64 only (line 6952)    │
│ - Higher-order map/filter: Bitcast corruption!  │
│ Location: lib/backend/llvm_codegen.cpp         │
└─────────────────────────────────────────────────┘
```

### 1.2 The Critical Bug - Bitcast Corruption

**Location:** [`lib/backend/llvm_codegen.cpp:5470-5472`](../lib/backend/llvm_codegen.cpp:5470-5472)

```cpp
// THE SMOKING GUN - THIS IS WHY EVERYTHING BREAKS:
builder->SetInsertPoint(extract_double);
Value* element_double = unpackDoubleFromTaggedValue(car_tagged);
Value* element_double_as_int = builder->CreateBitCast(element_double, Type::getInt64Ty(*context));
//                              ^^^^^^^^^^^^^^^^^^^ THIS CORRUPTS THE VALUE!
builder->CreateBr(merge_extract);
```

**What Happens:**
- `3.14` (double) → IEEE-754 bit pattern `0x400921FB54442D18`
- Treated as int64 → `4614256650576692846` (GARBAGE!)
- Passed to arithmetic function expecting int64
- Result is meaningless

**Why This Exists:**
- Functions expect `int64` parameters (line 694-700)
- Map extracts `double` from tagged cons cell
- Bitcast is used to "convert" double→int64 to match signature
- **But bitcast doesn't convert, it reinterprets the bits!**

### 1.3 Comprehensive Issue Inventory

#### Issue 1: Function Signature Mismatch (CRITICAL - 694-700)
```cpp
// ALL user-defined functions expect int64
std::vector<Type*> param_types(num_params, Type::getInt64Ty(*context));
```
**Impact:** Cannot pass doubles to functions at all

#### Issue 2: Builtin Arithmetic Functions (CRITICAL - 6952)
```cpp
// Builtin +, *, -, / all expect int64
std::vector<Type*> param_types(arity, Type::getInt64Ty(*context));
```
**Impact:** Mixed-type arithmetic fundamentally broken

#### Issue 3: CreateStructGEP Direct Access (HIGH - 40 instances)
**Found in 14 functions:**
- `codegenAppend`: 4 sites (lines 4813, 4839, 4846, 4864)
- `codegenReverse`: 2 sites (lines 4919, 4928)
- `codegenFilter`: 3 sites (lines 5616, 5647, 5654)
- `codegenFold`: 2 sites (lines 5724, 5735)
- `codegenForEachSingleList`: 2 sites (lines 5947, 5954)
- `codegenMember`: 2 sites (lines 5850, 5871)
- `codegenAssoc`: 4 sites (lines 6011, 6023, 6044, 6067)
- `codegenTake`: 3 sites (lines 6190, 6214, 6221)
- `codegenFind`: 3 sites (lines 6347, 6363, 6384)
- `codegenPartition`: 4 sites (lines 6466, 6497, 6523, 6530)
- `codegenSplitAt`: 3 sites (lines 6603, 6627, 6634)
- `codegenRemove`: 3 sites (lines 6705, 6742, 6749)
- `codegenSetCar`: 1 site (line 5110)
- `codegenSetCdr`: 1 site (line 5131)

**Impact:** Bypasses tagged type system, assumes 16-byte int64-only cons cells

---

## 2. THE POLYMORPHIC SOLUTION

### 2.1 Design Philosophy

**DECISION:** Make the ENTIRE system polymorphic - NO int64-only legacy code

**Rationale:**
1. v1.0-architecture should be architecturally sound
2. Backward compatibility not needed (pre-1.0 is development)
3. Clean foundation required for HoTT integration
4. Half-measures create technical debt

### 2.2 Unified Architecture

```
┌──────────────────────────────────────────────────┐
│ AST Domain (Compile Time)                       │
│ - TypedValue used temporarily during parsing     │
│ - Immediately converted to tagged_value          │
└─────────────────┬────────────────────────────────┘
                  ↓ typedValueToTaggedValue()
┌──────────────────────────────────────────────────┐
│ LLVM IR Domain (Runtime) - ALL TAGGED VALUES    │
│                                                   │
│ Functions: (tagged_value, ...) → tagged_value    │
│ Cons Cells: Store tagged values                  │
│ Arithmetic: Unpack, promote, compute, repack     │
│ Lists: Extract/store tagged values               │
└──────────────────────────────────────────────────┘
```

### 2.3 Key Components

**Already Implemented:**
- ✅ [`eshkol_tagged_value_t`](../inc/eshkol/eshkol.h:68-77) - C struct definition
- ✅ [`tagged_value_type`](../lib/backend/llvm_codegen.cpp:130-136) - LLVM struct type
- ✅ [`pack/unpack helpers`](../lib/backend/llvm_codegen.cpp:1097-1167) - Convert to/from tagged_value
- ✅ [`extractCarAsTaggedValue()`](../lib/backend/llvm_codegen.cpp:1168-1205) - Type-aware car extraction
- ✅ [`extractCdrAsTaggedValue()`](../lib/backend/llvm_codegen.cpp:1207-1244) - Type-aware cdr extraction
- ✅ [`codegenTaggedArenaConsCell()`](../lib/backend/llvm_codegen.cpp:1012-1092) - Create typed cons cells

**Needs Implementation:**
- ❌ `typedValueToTaggedValue()` - Boundary conversion helper
- ❌ `createPolymorphicArithmeticFunction()` - Type-aware arithmetic
- ❌ `unpackTaggedValueForCall()` - Conditional unpacking based on callee needs
- ❌ Polymorphic function signatures (ALL functions)
- ❌ CreateStructGEP elimination (40 sites)

---

## 3. DETAILED IMPLEMENTATION PLAN

### PHASE 1: Core Infrastructure (Week 1 - Days 1-5)

#### Day 1: Add Boundary Conversion Helper

**Location:** Add after [`detectValueType()`](../lib/backend/llvm_codegen.cpp:1246-1269)

```cpp
// Convert TypedValue to tagged_value (AST→IR boundary)
Value* typedValueToTaggedValue(const TypedValue& tv) {
    if (tv.isInt64()) {
        return packInt64ToTaggedValue(tv.llvm_value, tv.is_exact);
    } else if (tv.isDouble()) {
        return packDoubleToTaggedValue(tv.llvm_value);
    } else if (tv.type == ESHKOL_VALUE_CONS_PTR) {
        return packPtrToTaggedValue(tv.llvm_value, ESHKOL_VALUE_CONS_PTR);
    } else if (tv.isNull()) {
        return packInt64ToTaggedValue(
            ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    }
    
    // Fallback: null
    return packInt64ToTaggedValue(
        ConstantInt::get(Type::getInt64Ty(*context), 0), true);
}
```

**Lines to add:** ~20  
**Test:** Convert TypedValue to tagged_value and verify structure

#### Day 2: Replace codegenTypedAST with Tagged Version

**Location:** Modify [`codegenTypedAST()`](../lib/backend/llvm_codegen.cpp:904-941)

**Option A:** Keep codegenTypedAST, add wrapper
```cpp
Value* codegenTaggedValueAST(const eshkol_ast_t* ast) {
    TypedValue tv = codegenTypedAST(ast);
    return typedValueToTaggedValue(tv);
}
```

**Option B:** Replace codegenTypedAST entirely (RECOMMENDED)
```cpp
Value* codegenTypedAST(const eshkol_ast_t* ast) {
    if (!ast) return packInt64ToTaggedValue(
        ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    
    switch (ast->type) {
        case ESHKOL_INT64:
            return packInt64ToTaggedValue(
                ConstantInt::get(Type::getInt64Ty(*context), ast->int64_val), 
                true);
        
        case ESHKOL_DOUBLE:
            return packDoubleToTaggedValue(
                ConstantFP::get(Type::getDoubleTy(*context), ast->double_val));
        
        case ESHKOL_VAR:
        case ESHKOL_OP:
        default: {
            // Generate value and wrap in tagged_value
            Value* val = codegenAST(ast);
            if (!val) return packInt64ToTaggedValue(
                ConstantInt::get(Type::getInt64Ty(*context), 0), true);
            
            // Detect type and pack
            Type* llvm_type = val->getType();
            if (llvm_type->isIntegerTy(64)) {
                return packInt64ToTaggedValue(val, true);
            } else if (llvm_type->isDoubleTy()) {
                return packDoubleToTaggedValue(val);
            } else if (llvm_type == tagged_value_type) {
                return val; // Already tagged
            } else {
                // Pointer or other type
                return packPtrToTaggedValue(val, ESHKOL_VALUE_CONS_PTR);
            }
        }
    }
}
```

**Lines to change:** ~40  
**Test:** AST literals properly packaged as tagged_value

#### Day 3: Polymorphic Arithmetic Foundation

**Location:** Add new function before [`createBuiltinArithmeticFunction()`](../lib/backend/llvm_codegen.cpp:6931)

```cpp
// Create polymorphic arithmetic function that handles mixed types
Function* createPolymorphicArithmeticFunction(const std::string& operation, size_t arity) {
    if (arity == 0) {
        eshkol_error("Cannot create arithmetic function with 0 arguments");
        return nullptr;
    }
    
    std::string func_name = "poly_" + operation + "_" + std::to_string(arity);
    
    // Check if already exists
    auto it = function_table.find(func_name);
    if (it != function_table.end()) {
        return it->second;
    }
    
    // Create function: (tagged_value, ...) → tagged_value
    std::vector<Type*> param_types(arity, tagged_value_type);
    FunctionType* func_type = FunctionType::get(tagged_value_type, param_types, false);
    
    Function* func = Function::Create(
        func_type, Function::ExternalLinkage, func_name, module.get());
    
    BasicBlock* entry = BasicBlock::Create(*context, "entry", func);
    IRBuilderBase::InsertPoint old_point = builder->saveIP();
    builder->SetInsertPoint(entry);
    
    // STEP 1: Detect if any argument is double
    Value* any_double = ConstantInt::get(Type::getInt1Ty(*context), 0);
    
    auto arg_it = func->arg_begin();
    for (size_t i = 0; i < arity; ++i, ++arg_it) {
        Value* arg_type = getTaggedValueType(&*arg_it);
        Value* base_type = builder->CreateAnd(arg_type,
            ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
        Value* is_double = builder->CreateICmpEQ(base_type,
            ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
        any_double = builder->CreateOr(any_double, is_double);
    }
    
    // STEP 2: Branch based on type
    BasicBlock* int_path = BasicBlock::Create(*context, "int_arithmetic", func);
    BasicBlock* double_path = BasicBlock::Create(*context, "double_arithmetic", func);
    BasicBlock* pack_result = BasicBlock::Create(*context, "pack_result", func);
    
    builder->CreateCondBr(any_double, double_path, int_path);
    
    // INT PATH: All int64 arithmetic
    builder->SetInsertPoint(int_path);
    arg_it = func->arg_begin();
    Value* int_result = unpackInt64FromTaggedValue(&*arg_it++);
    
    for (size_t i = 1; i < arity && arg_it != func->arg_end(); ++i, ++arg_it) {
        Value* operand = unpackInt64FromTaggedValue(&*arg_it);
        
        if (operation == "+") {
            int_result = builder->CreateAdd(int_result, operand);
        } else if (operation == "-") {
            int_result = builder->CreateSub(int_result, operand);
        } else if (operation == "*") {
            int_result = builder->CreateMul(int_result, operand);
        } else if (operation == "/") {
            int_result = builder->CreateSDiv(int_result, operand);
        }
    }
    Value* int_tagged = packInt64ToTaggedValue(int_result, true);
    builder->CreateBr(pack_result);
    
    // DOUBLE PATH: Mixed type arithmetic with promotion
    builder->SetInsertPoint(double_path);
    arg_it = func->arg_begin();
    
    // Unpack first argument with type-aware extraction
    Value* first_arg_type = getTaggedValueType(&*arg_it);
    Value* first_is_int = builder->CreateICmpEQ(
        builder->CreateAnd(first_arg_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F)),
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_INT64));
    
    Value* first_int = unpackInt64FromTaggedValue(&*arg_it);
    Value* first_double = unpackDoubleFromTaggedValue(&*arg_it);
    Value* double_result = builder->CreateSelect(first_is_int,
        builder->CreateSIToFP(first_int, Type::getDoubleTy(*context)),
        first_double);
    
    ++arg_it;
    
    // Process remaining arguments
    for (size_t i = 1; i < arity && arg_it != func->arg_end(); ++i, ++arg_it) {
        Value* arg_type = getTaggedValueType(&*arg_it);
        Value* is_int = builder->CreateICmpEQ(
            builder->CreateAnd(arg_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F)),
            ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_INT64));
        
        Value* operand_int = unpackInt64FromTaggedValue(&*arg_it);
        Value* operand_double = unpackDoubleFromTaggedValue(&*arg_it);
        Value* operand = builder->CreateSelect(is_int,
            builder->CreateSIToFP(operand_int, Type::getDoubleTy(*context)),
            operand_double);
        
        if (operation == "+") {
            double_result = builder->CreateFAdd(double_result, operand);
        } else if (operation == "-") {
            double_result = builder->CreateFSub(double_result, operand);
        } else if (operation == "*") {
            double_result = builder->CreateFMul(double_result, operand);
        } else if (operation == "/") {
            double_result = builder->CreateFDiv(double_result, operand);
        }
    }
    Value* double_tagged = packDoubleToTaggedValue(double_result);
    builder->CreateBr(pack_result);
    
    // Return result
    builder->SetInsertPoint(pack_result);
    PHINode* result = builder->CreatePHI(tagged_value_type, 2);
    result->addIncoming(int_tagged, int_path);
    result->addIncoming(double_tagged, double_path);
    builder->CreateRet(result);
    
    builder->restoreIP(old_point);
    registerContextFunction(func_name, func);
    
    eshkol_debug("Created polymorphic arithmetic: %s with arity %zu", func_name.c_str(), arity);
    return func;
}
```

**Lines to add:** ~120-150  
**Test:** `(+ 1 2.0)` should return tagged double 3.0

#### Days 4-5: Test Phase 1 Infrastructure

**Create test file:** `tests/phase1_polymorphic_infrastructure_test.esk`
```scheme
;; Test polymorphic arithmetic
(display "=== PHASE 1 INFRASTRUCTURE TESTS ===")
(newline)

;; Pure integer
(display "1. Pure integer: (+ 1 2 3) = ")
(display (+ 1 2 3))
(newline)

;; Pure double
(display "2. Pure double: (+ 1.0 2.0 3.0) = ")
(display (+ 1.0 2.0 3.0))
(newline)

;; Mixed types - THE CRITICAL TEST
(display "3. Mixed types: (+ 1 2.0 3) = ")
(display (+ 1 2.0 3))
(newline)

;; Multiplication
(display "4. Mixed mult: (* 3 2.5) = ")
(display (* 3 2.5))
(newline)

(display "=== ALL TESTS SHOULD SHOW CORRECT VALUES ===")
(newline)
```

**Expected Output:**
```
=== PHASE 1 INFRASTRUCTURE TESTS ===
1. Pure integer: (+ 1 2 3) = 6
2. Pure double: (+ 1.0 2.0 3.0) = 6.0
3. Mixed types: (+ 1 2.0 3) = 6.0
4. Mixed mult: (* 3 2.5) = 7.5
=== ALL TESTS SHOULD SHOW CORRECT VALUES ===
```

---

### PHASE 2: Function System Overhaul (Week 2 - Days 1-10)

#### 2.1 Update Function Declarations (Days 1-2)

**File:** [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:685-723)

**Line 694-700 - COMPLETE REPLACEMENT:**
```cpp
void createFunctionDeclaration(const eshkol_ast_t* ast) {
    if (ast->type != ESHKOL_OP || ast->operation.op != ESHKOL_DEFINE_OP || 
        !ast->operation.define_op.is_function) {
        return;
    }
    
    const char* func_name = ast->operation.define_op.name;
    uint64_t num_params = ast->operation.define_op.num_params;
    
    // NEW: ALL parameters and return are tagged_value
    std::vector<Type*> param_types(num_params, tagged_value_type);
    FunctionType* func_type = FunctionType::get(
        tagged_value_type, // Return tagged_value
        param_types,
        false
    );
    
    Function* function = Function::Create(
        func_type,
        Function::ExternalLinkage,
        func_name,
        module.get()
    );
    
    // Set parameter names (unchanged)
    if (ast->operation.define_op.parameters) {
        auto arg_it = function->arg_begin();
        for (uint64_t i = 0; i < num_params && arg_it != function->arg_end(); ++i, ++arg_it) {
            if (ast->operation.define_op.parameters[i].type == ESHKOL_VAR &&
                ast->operation.define_op.parameters[i].variable.id) {
                arg_it->setName(ast->operation.define_op.parameters[i].variable.id);
            }
        }
    }
    
    registerContextFunction(func_name, function);
    eshkol_debug("Created POLYMORPHIC function declaration: %s with %llu tagged_value parameters", 
                func_name, (unsigned long long)num_params);
}
```

**Lines changed:** 7 (695, 696, 697, 722)

#### 2.2 Update Function Body Generation (Days 3-4)

**File:** [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:1405-1475)

**Changes Required:**

1. **Line 1429:** Parameters are already tagged_value, store directly
2. **Lines 1449-1466:** Pack return value if not already tagged

```cpp
// Return the result - ensure we always have a terminator
if (body_result) {
    // Check if body_result is already a tagged_value
    if (body_result->getType() == tagged_value_type) {
        builder->CreateRet(body_result);
    } else if (isa<Function>(body_result)) {
        // Lambda return - pack function pointer
        Function* lambda_func = dyn_cast<Function>(body_result);
        Value* func_addr = builder->CreatePtrToInt(lambda_func, Type::getInt64Ty(*context));
        Value* func_tagged = packPtrToTaggedValue(
            builder->CreateIntToPtr(func_addr, builder->getPtrTy()),
            ESHKOL_VALUE_CONS_PTR); // Use CONS_PTR for function pointers
        builder->CreateRet(func_tagged);
    } else {
        // Detect type and pack
        TypedValue typed = detectValueType(body_result);
        Value* tagged = typedValueToTaggedValue(typed);
        builder->CreateRet(tagged);
    }
} else {
    // Return null tagged_value
    Value* null_tagged = packInt64ToTaggedValue(
        ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    builder->CreateRet(null_tagged);
}
```

**Lines changed:** ~20

#### 2.3 Update Lambda Generation (Days 5-6)

**File:** [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:2743-2862)

**Lines 2758-2764, 2767:** Change to tagged_value parameters

```cpp
// Create function type - ALL parameters are tagged_value
std::vector<Type*> param_types;
for (uint64_t i = 0; i < op->lambda_op.num_params; i++) {
    param_types.push_back(tagged_value_type);
}
for (size_t i = 0; i < free_vars.size(); i++) {
    param_types.push_back(tagged_value_type);
}

FunctionType* func_type = FunctionType::get(
    tagged_value_type, // Return tagged_value
    param_types,
    false
);
```

**Lines 2832-2837:** Pack return value

```cpp
// Ensure we always have a terminator
if (body_result) {
    if (body_result->getType() == tagged_value_type) {
        builder->CreateRet(body_result);
    } else {
        TypedValue typed = detectValueType(body_result);
        Value* tagged = typedValueToTaggedValue(typed);
        builder->CreateRet(tagged);
    }
} else {
    Value* null_tagged = packInt64ToTaggedValue(
        ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    builder->CreateRet(null_tagged);
}
```

**Lines changed:** 12

#### 2.4 Update Builtin Function Resolution (Day 7)

**File:** [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:5195-5284)

**Lines 5229-5240:** Replace with polymorphic versions

```cpp
// Handle builtin functions with polymorphic implementation
if (func_name == "+") {
    return createPolymorphicArithmeticFunction("+", required_arity);
}
if (func_name == "*") {
    return createPolymorphicArithmeticFunction("*", required_arity);
}
if (func_name == "-") {
    return createPolymorphicArithmeticFunction("-", required_arity);
}
if (func_name == "/") {
    return createPolymorphicArithmeticFunction("/", required_arity);
}
```

**Lines changed:** 4

#### 2.5 Update ALL Function Call Sites (Days 8-10)

**Strategy:** Search and replace pattern

**Search for:** `builder->CreateCall`  
**Count:** ~200 instances

**For each instance:**
1. Check if arguments are tagged_value
2. If not, wrap using `codegenTypedAST()` or `typedValueToTaggedValue()`
3. Ensure return value is tagged_value

**Example - Line 1932:**
```cpp
// OLD:
return builder->CreateCall(callee, args);

// NEW:
Value* result = builder->CreateCall(callee, args);
// result is now tagged_value (no change needed if args were tagged)
return result;
```

**Critical call sites:**
- [`codegenArithmetic()`](../lib/backend/llvm_codegen.cpp:1935-1955) - Already uses TypedValue, needs update
- [`codegenCall()`](../lib/backend/llvm_codegen.cpp:1563-1933) - Multiple call sites
- All higher-order functions (map, filter, fold) - Procedure calls

---

### PHASE 3: Higher-Order Function Migration (Week 3 - Days 1-10)

#### 3.1 Map Functions (Days 1-2) - HIGHEST PRIORITY

##### `codegenMapSingleList` (Lines 5287-5392)

**CRITICAL FIX - Line 5326-5327:**
```cpp
// OLD (BROKEN):
Value* car_tagged = extractCarAsTaggedValue(current_val);
Value* input_element = unpackInt64FromTaggedValue(car_tagged);
// Then bitcast if double! ^^^

// NEW (CORRECT):
Value* car_tagged = extractCarAsTaggedValue(current_val);
// Pass DIRECTLY to procedure - it accepts tagged_value now!
Value* proc_result = builder->CreateCall(proc_func, {car_tagged});
```

**Lines to delete:** 5327 (unpacking step)  
**Lines to change:** 5330 (procedure call)

**Complete updated loop body:**
```cpp
// Loop body: apply procedure and build result
builder->SetInsertPoint(loop_body);

// Extract car as tagged_value
Value* car_tagged = extractCarAsTaggedValue(current_val);

// Call procedure with tagged_value (proc accepts tagged_value now!)
Value* proc_result = builder->CreateCall(proc_func, {car_tagged});
// proc_result is ALSO tagged_value

// Create new cons cell - need to convert tagged_value to TypedValue
// Extract type from tagged_value
TypedValue proc_result_typed = taggedValueToTypedValue(proc_result);
TypedValue cdr_null(ConstantInt::get(Type::getInt64Ty(*context), 0), ESHKOL_VALUE_NULL);
Value* new_result_cons = codegenTaggedArenaConsCell(proc_result_typed, cdr_null);

// Rest unchanged (tail updates already use tagged helpers)
```

##### `codegenMapMultiList` (Lines 5395-5552)

**CRITICAL FIX - Lines 5447-5485:**
```cpp
// OLD (BROKEN):
// Extract each list's car, unpack, bitcast if double, create PHI to merge

// NEW (CORRECT):
std::vector<Value*> proc_args;
for (size_t i = 0; i < current_ptrs.size(); i++) {
    Value* current_val = builder->CreateLoad(Type::getInt64Ty(*context), current_ptrs[i]);
    
    // Extract as tagged_value
    Value* car_tagged = extractCarAsTaggedValue(current_val);
    
    // Pass directly - NO UNPACKING, NO BITCASTING!
    proc_args.push_back(car_tagged);
}

// Call procedure with tagged_value arguments
Value* proc_result = builder->CreateCall(proc_func, proc_args);
// proc_result is tagged_value

// Create cons cell from tagged result
TypedValue proc_result_typed = taggedValueToTypedValue(proc_result);
TypedValue cdr_null(ConstantInt::get(Type::getInt64Ty(*context), 0), ESHKOL_VALUE_NULL);
Value* new_result_cons = codegenTaggedArenaConsCell(proc_result_typed, cdr_null);
```

**Lines to delete:** 5454-5485 (complex type branching and bitcast logic)  
**Lines to add:** ~15 (simpler tagged value handling)

#### 3.2 Filter/Fold/Find (Days 3-4)

**Pattern applies to all three:**

```cpp
// Extract car as tagged_value
Value* car_tagged = extractCarAsTaggedValue(current_val);

// Pass to predicate/procedure AS-IS
Value* result = builder->CreateCall(pred_func, {car_tagged});

// For fold: accumulator is also tagged_value
Value* acc_tagged = builder->CreateLoad(tagged_value_type, accumulator);
Value* new_acc = builder->CreateCall(proc_func, {acc_tagged, car_tagged});
builder->CreateStore(new_acc, accumulator);
```

**Functions to update:**
1. [`codegenFilter`](../lib/backend/llvm_codegen.cpp:5555) - Lines 5616, 5647, 5654
2. [`codegenFold`](../lib/backend/llvm_codegen.cpp:5672) - Lines 5724, 5735
3. [`codegenFind`](../lib/backend/llvm_codegen.cpp:6297) - Lines 6347, 6363, 6384

**Estimated changes:** 30-40 lines per function

#### 3.3 Member/Assoc/Search Functions (Days 5-6)

**These need type-aware comparison:**

```cpp
// OLD: Assumes int64 comparison
Value* is_match = builder->CreateICmpEQ(current_element, item);

// NEW: Type-aware comparison
Value* is_match = createTaggedValueComparison(current_element_tagged, item_tagged, "equal");
```

**New helper needed:**
```cpp
Value* createTaggedValueComparison(Value* left_tagged, Value* right_tagged, 
                                    const std::string& comparison_type) {
    // Unpack both values
    // If types differ, convert to common type
    // Perform appropriate comparison (int or float)
    // Return i1 result
}
```

**Functions to update:**
1. [`codegenMember`](../lib/backend/llvm_codegen.cpp:5812) - Lines 5850, 5871
2. [`codegenAssoc`](../lib/backend/llvm_codegen.cpp:5972) - Lines 6011, 6023, 6044, 6067

#### 3.4 List Manipulation Functions (Days 7-9)

**Direct CreateStructGEP elimination - mechanical replacement:**

1. **`codegenAppend`** (Lines 4737-4879) - 4 sites
   - Line 4813: Use `extractCarAsNativeValue()` or just work with tagged values
   - Lines 4839, 4846, 4864: Use `arena_tagged_cons_get/set_ptr_func`

2. **`codegenReverse`** (Lines 4882-4938) - 2 sites
   - Line 4919: Extract car with tagged helper
   - Line 4928: Use tagged cons get_ptr

3. **`codegenTake`** (Lines 6138-6238) - 3 sites
4. **`codegenSplitAt`** (Lines 6551-6655) - 3 sites
5. **`codegenPartition`** (Lines 6402-6548) - 4 sites
6. **`codegenRemove`** (Lines 6658-6763) - 3 sites

**Pattern for all:**
```cpp
// CAR extraction:
// OLD: CreateStructGEP(arena_cons_type, cons_ptr, 0) + Load
// NEW: extractCarAsTaggedValue(cons_ptr_int)

// CDR iteration:
// OLD: CreateStructGEP(arena_cons_type, cons_ptr, 1) + Load
// NEW: arena_tagged_cons_get_ptr_func(cons_ptr, is_cdr=1)

// CDR update:
// OLD: CreateStructGEP(arena_cons_type, cons_ptr, 1) + Store
// NEW: arena_tagged_cons_set_ptr_func(cons_ptr, is_cdr=1, value, type)
```

#### 3.5 Mutable Operations (Day 10)

**`codegenSetCar`** (Lines 5097-5115) - Line 5110:
```cpp
// OLD:
Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);
builder->CreateStore(new_value, car_ptr);

// NEW:
// Detect type of new_value
TypedValue new_typed = detectValueType(new_value);
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);

if (new_typed.isInt64()) {
    builder->CreateCall(arena_tagged_cons_set_int64_func,
        {cons_ptr, is_car, new_typed.llvm_value,
         ConstantInt::get(Type::getInt8Ty(*context), new_typed.type)});
} else if (new_typed.isDouble()) {
    builder->CreateCall(arena_tagged_cons_set_double_func,
        {cons_ptr, is_car, new_typed.llvm_value,
         ConstantInt::get(Type::getInt8Ty(*context), new_typed.type)});
}
```

**`codegenSetCdr`** (Lines 5118-5136) - Line 5131: Same pattern

---

### PHASE 4: Validation & Testing (Week 4 - Days 1-5)

#### Day 1-2: Comprehensive Test Suite

**Create:** `tests/complete_polymorphic_validation.esk`

```scheme
(display "=== COMPLETE POLYMORPHIC SYSTEM VALIDATION ===")
(newline)

;; Section 1: Basic Operations
(display "1. BASIC OPERATIONS")
(newline)

(define int-list (list 1 2 3))
(define double-list (list 1.0 2.0 3.0))
(define mixed-list (list 1 2.0 3 4.0 5))

(display "   int-list: ") (display int-list) (newline)
(display "   double-list: ") (display double-list) (newline)
(display "   mixed-list: ") (display mixed-list) (newline)

;; Section 2: Map Operations
(display "2. MAP OPERATIONS")
(newline)

(display "   map + on mixed: ")
(display (map + (list 1 2.0 3) (list 4 5.0 6)))
(newline)

(display "   map * on mixed: ")
(display (map * (list 2 3.0) (list 4.0 5)))
(newline)

;; Section 3: Filter Operations
(display "3. FILTER OPERATIONS")
(newline)

(display "   filter >5 on mixed: ")
(display (filter (lambda (x) (> x 5)) (list 1 2.0 10 3.5 20)))
(newline)

;; Section 4: Fold Operations
(display "4. FOLD OPERATIONS")
(newline)

(display "   fold + on mixed: ")
(display (fold + 0 (list 1 2.0 3)))
(newline)

(display "   fold * on mixed: ")
(display (fold * 1 (list 2 3.0 4)))
(newline)

;; Section 5: Complex Chaining
(display "5. COMPLEX CHAINING")
(newline)

(display "   map + filter + fold: ")
(display (fold + 0 (map (lambda (x) (* x 2)) 
                         (filter (lambda (x) (> x 0)) 
                                 (list -1 2 -3 4.0 5)))))
(newline)

(display "=== ALL TESTS SHOULD SHOW CORRECT TYPED VALUES ===")
(newline)
```

**Expected Output:**
```
=== COMPLETE POLYMORPHIC SYSTEM VALIDATION ===
1. BASIC OPERATIONS
   int-list: (1 2 3)
   double-list: (1.0 2.0 3.0)
   mixed-list: (1 2.0 3 4.0 5)
2. MAP OPERATIONS
   map + on mixed: (5.0 7.0 9.0)
   map * on mixed: (8.0 15.0)
3. FILTER OPERATIONS
   filter >5 on mixed: (10 3.5 20)
4. FOLD OPERATIONS
   fold + on mixed: 6.0
   fold * on mixed: 24.0
5. COMPLEX CHAINING
   map + filter + fold: 22.0
=== ALL TESTS SHOULD SHOW CORRECT TYPED VALUES ===
```

#### Days 3-4: Regression Testing

**Run ALL existing tests:**
- [`tests/integer_only_test.esk`](../tests/integer_only_test.esk) - Should still pass
- [`tests/mixed_type_lists_basic_test.esk`](../tests/mixed_type_lists_basic_test.esk) - Should pass now
- [`tests/phase_2a_multilist_map_test.esk`](../tests/phase_2a_multilist_map_test.esk) - Should pass
- [`tests/session_006_multilist_map_test.esk`](../tests/session_006_multilist_map_test.esk) - Should pass

#### Day 5: Performance & Memory Validation

**Benchmarks:**
1. Integer-only operations (baseline - should be similar to before)
2. Mixed-type operations (new capability)
3. Memory usage (tagged_value is 16 bytes vs 8 bytes int64)

**Memory Safety:**
```bash
valgrind ./build/eshkol-run tests/complete_polymorphic_validation.esk
# Should show: No memory leaks, no invalid accesses
```

---

## 4. IMPLEMENTATION GUIDELINES

### 4.1 Coding Standards

**Type Conversion Rules:**
1. ✅ **DO:** Use `packInt64ToTaggedValue()` for int64→tagged_value
2. ✅ **DO:** Use `packDoubleToTaggedValue()` for double→tagged_value
3. ✅ **DO:** Use `extractCarAsTaggedValue()` for cons car extraction
4. ❌ **DON'T:** Use `CreateBitCast(double, i64)` - EVER!
5. ❌ **DON'T:** Use `CreateStructGEP` on arena_cons_type

**Function Signature Rules:**
1. ✅ **ALL** user functions: `(tagged_value, ...) → tagged_value`
2. ✅ **ALL** lambdas: `(tagged_value, ...) → tagged_value`
3. ✅ **ALL** builtins: Polymorphic versions

**Naming Conventions:**
- Polymorphic functions: `poly_{operation}_{arity}`
- Tagged value helpers: `*TaggedValue*`
- Native value helpers: `*Native*`

### 4.2 Migration Safety Protocol

**For Each Function:**
1. **Document current behavior** - Write test showing current output
2. **Make changes** - Update function code
3. **Test immediately** - Run specific test for that function
4. **Validate no regression** - Run full test suite
5. **Commit** - Git checkpoint

**Git Branch Strategy:**
```bash
git checkout -b feature/polymorphic-tagged-system
git commit -m "Phase 1.1: Add boundary conversion helper"
git commit -m "Phase 1.2: Update AST to IR conversion"
git commit -m "Phase 1.3: Add polymorphic arithmetic"
# etc.
```

### 4.3 Debugging Tips

**When map/filter fails:**
1. Check if procedure accepts tagged_value
2. Verify no bitcasting happening
3. Print type tags before/after operations
4. Check PHI nodes merge same types

**When arithmetic gives wrong results:**
1. Verify type promotion logic
2. Check if using correct arithmetic (int vs float)
3. Ensure exactness flags set correctly

---

## 5. DETAILED FUNCTION-BY-FUNCTION SPECS

### 5.1 Map Functions

#### `codegenMapSingleList` - COMPLETE SPEC

**Current State:** Lines 5287-5392  
**CreateStructGEP Count:** 0 (already migrated to tagged helpers for cdr iteration)  
**Critical Bug:** Line 5327 unpacking + line 5472 bitcasting in multi-list version  

**Changes Required:**

**Line 5326-5330 (BEFORE):**
```cpp
Value* car_tagged = extractCarAsTaggedValue(current_val);
Value* input_element = unpackInt64FromTaggedValue(car_tagged);

// Apply procedure to current element
Value* proc_result = builder->CreateCall(proc_func, {input_element});
```

**Line 5326-5330 (AFTER):**
```cpp
// Extract car as tagged_value
Value* car_tagged = extractCarAsTaggedValue(current_val);

// Apply procedure - it now accepts tagged_value!
Value* proc_result = builder->CreateCall(proc_func, {car_tagged});
// proc_result is also tagged_value
```

**Line 5334-5336 (BEFORE):**
```cpp
TypedValue proc_result_typed = detectValueType(proc_result);
TypedValue cdr_null = TypedValue(ConstantInt::get(Type::getInt64Ty(*context), 0), ESHKOL_VALUE_NULL);
Value* new_result_cons = codegenTaggedArenaConsCell(proc_result_typed, cdr_null);
```

**Line 5334-5336 (AFTER):**
```cpp
// Convert tagged_value result to TypedValue for cons creation
TypedValue proc_result_typed = taggedValueToTypedValue(proc_result);
TypedValue cdr_null = TypedValue(ConstantInt::get(Type::getInt64Ty(*context), 0), ESHKOL_VALUE_NULL);
Value* new_result_cons = codegenTaggedArenaConsCell(proc_result_typed, cdr_null);
```

**New helper needed:**
```cpp
TypedValue taggedValueToTypedValue(Value* tagged_val) {
    Value* type_field = getTaggedValueType(tagged_val);
    Value* base_type = builder->CreateAnd(type_field,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    Value* is_int = builder->CreateICmpEQ(base_type,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_INT64));
    Value* is_double = builder->CreateICmpEQ(base_type,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    
    // Conditional extraction
    Function* current_func = builder->GetInsertBlock()->getParent();
    BasicBlock* int_block = BasicBlock::Create(*context, "tv_int", current_func);
    BasicBlock* double_block = BasicBlock::Create(*context, "tv_double", current_func);
    BasicBlock* merge_block = BasicBlock::Create(*context, "tv_merge", current_func);
    
    // Create storage for result type
    Value* result_type_storage = builder->CreateAlloca(Type::getInt8Ty(*context));
    Value* result_value_storage = builder->CreateAlloca(Type::getInt64Ty(*context));
    
    builder->CreateCondBr(is_int, int_block, double_block);
    
    builder->SetInsertPoint(int_block);
    Value* int_val = unpackInt64FromTaggedValue(tagged_val);
    builder->CreateStore(ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_INT64), result_type_storage);
    builder->CreateStore(int_val, result_value_storage);
    builder->CreateBr(merge_block);
    
    builder->SetInsertPoint(double_block);
    Value* double_val = unpackDoubleFromTaggedValue(tagged_val);
    builder->CreateStore(ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE), result_type_storage);
    Value* double_as_int64 = builder->CreateBitCast(double_val, Type::getInt64Ty(*context)); // Only for storage!
    builder->CreateStore(double_as_int64, result_value_storage);
    builder->CreateBr(merge_block);
    
    builder->SetInsertPoint(merge_block);
    Value* final_type = builder->CreateLoad(Type::getInt8Ty(*context), result_type_storage);
    Value* final_value_bits = builder->CreateLoad(Type::getInt64Ty(*context), result_value_storage);
    
    // Reconstruct proper typed value
    Value* is_final_double = builder->CreateICmpEQ(final_type,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    
    BasicBlock* reconstruct_double = BasicBlock::Create(*context, "recon_double", current_func);
    BasicBlock* reconstruct_int = BasicBlock::Create(*context, "recon_int", current_func);
    BasicBlock* final_merge = BasicBlock::Create(*context, "final_merge", current_func);
    
    builder->CreateCondBr(is_final_double, reconstruct_double, reconstruct_int);
    
    builder->SetInsertPoint(reconstruct_double);
    Value* final_double = builder->CreateBitCast(final_value_bits, Type::getDoubleTy(*context));
    builder->CreateBr(final_merge);
    
    builder->SetInsertPoint(reconstruct_int);
    Value* final_int = final_value_bits;
    builder->CreateBr(final_merge);
    
    builder->SetInsertPoint(final_merge);
    // Can't use PHI for different types
    // Return TypedValue with type flag and handle separately in caller
    
    // ACTUALLY, this is getting complex. Better solution:
    // Just keep TypedValue as C++ struct, extract in calling code
}
```

**Wait, this is getting too complex. Let me reconsider...**

**SIMPLER APPROACH:** Keep TypedValue as intermediate, only use tagged_value for function parameters/returns.

---

## 6. REVISED SIMPLER ARCHITECTURE

### 6.1 Keep TypedValue for Internal Use

```cpp
// AST → TypedValue (C++ struct, compile time)
TypedValue codegenTypedAST(const eshkol_ast_t* ast);

// TypedValue → tagged_value (for function calls)
Value* typedValueToTaggedValue(const TypedValue& tv);

// tagged_value → TypedValue (for cons cell creation)
TypedValue unpackToTypedValue(Value* tagged_val) {
    // Extract type
    Value* type_field = getTaggedValueType(tagged_val);
    Value* base = builder->CreateAnd(type_field, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    // CANNOT use PHI for different types
    // SOLUTION: Return different Value* based on runtime type
    // But we need compile-time TypedValue...
    
    // ACTUALLY: Just make two versions of cons cell creation:
    // 1. codegenTaggedArenaConsCell(TypedValue, TypedValue) - Current
    // 2. codegenTaggedArenaCons Cellfromtagged_values(Value*, Value*) - NEW
}
```

**OR EVEN SIMPLER:**

### 6.2 The ACTUAL Simple Solution

**Keep everything as-is for cons cell creation, ONLY fix function signatures and calls:**

1. **Functions accept/return `tagged_value`**
2. **Cons cell creation uses `TypedValue`** (current `codegenTaggedArenaConsCell`)
3. **Conversion happens at call boundaries:**
   - Before calling function: Pack TypedValue → tagged_value
   - After function returns: Detect type → TypedValue
   - For cons creation: Use TypedValue as before

**This means:**
```cpp
// In map:
TypedValue car_typed = extractCarAsTypedValue(current_val); // NEW helper
Value* car_tagged = typedValueToTaggedValue(car_typed);
Value* result_tagged = builder->CreateCall(proc_func, {car_tagged});
TypedValue result_typed = taggedValueToTypedValue(result_tagged); // NEW helper
Value* new_cons = codegenTaggedArenaConsCell(result_typed, cdr_null);
```

**New helpers needed:**
```cpp
// Extract car as TypedValue (not tagged_value struct)
TypedValue extractCarAsTypedValue(Value* cons_ptr_int) {
    // Call arena_tagged_cons_get_type
    // If INT64: call get_int64, return TypedValue(val, INT64, true)
    // If DOUBLE: call get_double, return TypedValue(val, DOUBLE, false)
    // No PHI nodes needed - return different types in different branches
    // Caller handles based on TypedValue.type field
}

// Convert tagged_value struct to TypedValue (reverse of typedValueToTaggedValue)
TypedValue taggedValueToTypedValue(Value* tagged_val) {
    // Similar to extractCarAsTypedValue but operates on tagged_value struct
    Value* type_field = getTaggedValueType(tagged_val);
    // Branch and extract in native type
    // Return TypedValue with correct type and value
}
```

---

## 7. FINAL SIMPLIFIED IMPLEMENTATION PLAN

### PHASE 1: Infrastructure (Week 1)

#### 1.1 Add Conversion Helpers (Days 1-2)
- [ ] `typedValueToTaggedValue(const TypedValue&)` - TypedValue → tagged_value struct
- [ ] `taggedValueToTypedValue(Value* tagged_val)` - tagged_value struct → TypedValue  
- [ ] `extractCarAsTypedValue(Value* cons_ptr)` - Cons car → TypedValue
- [ ] `extractCdrAsTypedValue(Value* cons_ptr)` - Cons cdr → TypedValue

#### 1.2 Polymorphic Arithmetic (Days 3-4)
- [ ] `createPolymorphicArithmeticFunction(op, arity)` - Tagged params, type promotion
- [ ] Test: `(+ 1 2.0)` → 3.0

#### 1.3 Update codegenTypedAST (Day 5)
- [ ] Make it return tagged_value for literals directly
- [ ] Or keep returning TypedValue (cleaner separation)

**DECISION:** Keep TypedValue for internal use, only use tagged_value at function boundaries.

---

### PHASE 2: Function Signatures (Week 2)

#### 2.1 Update All Function Declarations (Days 1-2)
- [ ] [`createFunctionDeclaration()`](../lib/backend/llvm_codegen.cpp:685) - Lines 695-700
- [ ] [`codegenLambda()`](../lib/backend/llvm_codegen.cpp:2743) - Lines 2758-2770
- [ ] Test: Define function with mixed-type params

#### 2.2 Update Function Body Generation (Days 3-4)
- [ ] [`codegenFunctionDefinition()`](../lib/backend/llvm_codegen.cpp:1405) - Lines 1449-1466
- [ ] Handle tagged_value parameters (unpack as needed in body)
- [ ] Pack return value to tagged_value

#### 2.3 Update Builtin Resolution (Day 5)
- [ ] [`resolveLambdaFunction()`](../lib/backend/llvm_codegen.cpp:5195) - Lines 5229-5240
- [ ] Use polymorphic arithmetic functions

#### 2.4 Update Function Call Sites (Days 6-10)
- [ ] Update ~200 `builder->CreateCall` sites
- [ ] Ensure arguments are tagged_value
- [ ] Handle tagged_value returns

---

### PHASE 3: Higher-Order Migration (Week 3)

#### 3.1 Map Functions (Days 1-2) - CRITICAL
- [ ] `codegenMapSingleList` - Remove line 5327 unpacking
- [ ] `codegenMapMultiList` - Remove lines 5454-5485 bitcast logic
- [ ] Test: `(map + (list 1 2.0 3) (list 4 5.0 6))` → `(5.0 7.0 9.0)`

#### 3.2 Filter/Fold/Find (Days 3-4)
- [ ] `codegenFilter` - 3 CreateStructGEP sites → tagged helpers
- [ ] `codegenFold` - 2 sites + accumulator handling
- [ ] `codegenFind` - 3 sites
- [ ] `codegenForEachSingleList` - 2 sites

#### 3.3 Member/Assoc (Days 5-6)
- [ ] `codegenMember` - 2 sites + type-aware comparison
- [ ] `codegenAssoc` - 4 sites + nested cons handling
- [ ] Add `createTaggedValueComparison()` helper

#### 3.4 List Manipulation (Days 7-9)
- [ ] `codegenAppend` - 4 sites
- [ ] `codegenReverse` - 2 sites
- [ ] `codegenTake` - 3 sites
- [ ] `codegenSplitAt` - 3 sites
- [ ] `codegenPartition` - 4 sites
- [ ] `codegenRemove` - 3 sites

#### 3.5 Mutable Operations (Day 10)
- [ ] `codegenSetCar` - 1 site → tagged setter
- [ ] `codegenSetCdr` - 1 site → tagged setter

---

### PHASE 4: Validation (Week 4)

#### 4.1 Test Coverage (Days 1-2)
- [ ] Unit tests for all helpers
- [ ] Function tests for each operation
- [ ] Integration tests for chained operations

#### 4.2 Regression Testing (Day 3)
- [ ] Run all existing tests
- [ ] Verify no performance regression

#### 4.3 Documentation (Day 4)
- [ ] Update [`TAGGED_VALUE_SYSTEM_IMPLEMENTATION.md`](../docs/TAGGED_VALUE_SYSTEM_IMPLEMENTATION.md)
- [ ] Update [`BUILD_STATUS.md`](../docs/BUILD_STATUS.md)
- [ ] Add migration notes for future HoTT integration

#### 4.4 Final Validation (Day 5)
- [ ] Memory safety (valgrind)
- [ ] Performance benchmarks
- [ ] Code review
- [ ] Merge to main

---

## 8. MIGRATION CHECKLIST BY FILE

### [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h)
- ✅ No changes needed - tagged_value_t already defined

### [`lib/core/arena_memory.h`](../lib/core/arena_memory.h) & [`lib/core/arena_memory.cpp`](../lib/core/arena_memory.cpp)
- ✅ No changes needed - tagged cons cell helpers complete

### [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) - MAJOR CHANGES

**New Code to Add (after line 1269):**
```cpp
// ===== BOUNDARY CONVERSION HELPERS =====

Value* typedValueToTaggedValue(const TypedValue& tv) { /* 20 lines */ }
TypedValue taggedValueToTypedValue(Value* tagged_val) { /* 80 lines */ }
TypedValue extractCarAsTypedValue(Value* cons_ptr_int) { /* 60 lines */ }
TypedValue extractCdrAsTypedValue(Value* cons_ptr_int) { /* 60 lines */ }
Function* createPolymorphicArithmeticFunction(const std::string& op, size_t arity) { /* 150 lines */ }
Value* createTaggedValueComparison(Value* left, Value* right, const std::string& type) { /* 80 lines */ }

// Total new code: ~450 lines
```

**Existing Code to Modify:**

| Function | Lines | Changes | Type |
|----------|-------|---------|------|
| `createFunctionDeclaration` | 685-723 | 4 lines | Signature |
| `codegenFunctionDefinition` | 1405-1475 | 20 lines | Pack/unpack |
| `codegenLambda` | 2743-2862 | 15 lines | Signature |
| `resolveLambdaFunction` | 5195-5284 | 12 lines | Use poly |
| `codegenMapSingleList` | 5287-5392 | 10 lines | Remove unpack |
| `codegenMapMultiList` | 5395-5552 | 40 lines | Remove bitcast |
| `codegenFilter` | 5555-5669 | 15 lines | Tagged extract |
| `codegenFold` | 5672-5744 | 12 lines | Tagged extract |
| `codegenForEachSingleList` | 5918-5963 | 10 lines | Tagged extract |
| `codegenMember` | 5812-5882 | 15 lines | Tagged compare |
| `codegenAssoc` | 5972-6082 | 20 lines | Nested tagged |
| `codegenTake` | 6138-6238 | 18 lines | Tagged extract |
| `codegenFind` | 6297-6399 | 18 lines | Tagged extract |
| `codegenPartition` | 6402-6548 | 25 lines | Dual tagged |
| `codegenSplitAt` | 6551-6655 | 20 lines | Tagged extract |
| `codegenRemove` | 6658-6763 | 18 lines | Tagged compare |
| `codegenAppend` | 4737-4879 | 25 lines | Tagged extract |
| `codegenReverse` | 4882-4938 | 12 lines | Tagged extract |
| `codegenSetCar` | 5097-5115 | 10 lines | Tagged setter |
| `codegenSetCdr` | 5118-5136 | 10 lines | Tagged setter |
| ALL `CreateCall` sites | Various | 200+ | Wrap args |

**Total lines to modify:** ~500  
**Total new lines:** ~450  
**Total effort:** ~950 lines of changes

---

## 9. SUCCESS CRITERIA

### Code Quality
- ✅ Zero `CreateBitCast(double, i64)` operations
- ✅ Zero `CreateStructGEP` on `arena_cons_type`
- ✅ All functions use `tagged_value` parameters
- ✅ All arithmetic handles mixed types correctly

### Functional Requirements
- ✅ Mixed-type lists work correctly
- ✅ Map/filter/fold preserve types
- ✅ Arithmetic promotes int64→double automatically
- ✅ Type information never lost

### Performance
- ✅ Integer-only code: <10% slower than before (acceptable)
- ✅ Mixed-type code: Correct results (new capability)
- ✅ Memory: No leaks, no corruption

### HoTT Readiness
- ✅ Clean tagged value system
- ✅ Type information preserved at runtime
- ✅ Extensible for dependent types
- ✅ No technical debt from half-measures

---

## 10. RISK ASSESSMENT

### High Risks
1. **Breaking all existing code**
   - Mitigation: Test each phase incrementally
   - Rollback: Git checkpoints at each phase

2. **Performance degradation**
   - Mitigation: Optimize hot paths, benchmark early
   - Acceptable: <10% slowdown for correctness

3. **Introducing new bugs**
   - Mitigation: Comprehensive test suite
   - Detection: Automated testing after each change

### Medium Risks
4. **Complex conditional logic**
   - Issue: Type branching makes code complex
   - Mitigation: Extract to helper functions

5. **PHI node type mismatches**
   - Issue: Can't merge i64 and double in PHI
   - Solution: Keep as TypedValue, only use tagged_value at boundaries

---

## 11. IMPLEMENTATION WORKFLOW

### Session Structure (Each Session = 2-4 hours)

**Week 1: Foundation**
- Session 1: Add typedValueToTaggedValue helper + tests
- Session 2: Add taggedValueToTypedValue helper + tests
- Session 3: Create polymorphic arithmetic foundation
- Session 4: Test polymorphic arithmetic with mixed types
- Session 5: Validate Phase 1 complete

**Week 2: Function System**
- Session 6: Update createFunctionDeclaration
- Session 7: Update codegenFunctionDefinition
- Session 8: Update codegenLambda
- Session 9: Update resolveLambdaFunction (use polymorphic builtins)
- Session 10: Test user functions with mixed types
- Session 11-15: Update function call sites (40 sites per session)

**Week 3: Higher-Order Migration**
- Session 16: Fix codegenMapSingleList (remove bitcast)
- Session 17: Fix codegenMapMultiList (full polymorphic)
- Session 18: Fix codegenFilter + codegenFold
- Session 19: Fix codegenFind + codegenForEachSingleList
- Session 20: Fix codegenMember + codegenAssoc
- Session 21: Fix list manipulation (Append, Reverse, Take)
- Session 22: Fix remaining (SplitAt, Partition, Remove)
- Session 23: Fix mutable operations (SetCar, SetCdr)
- Session 24: Test all higher-order functions
- Session 25: Integration testing

**Week 4: Validation**
- Session 26-27: Comprehensive test suite
- Session 28: Regression testing
- Session 29: Performance optimization
- Session 30: Documentation and final validation

---

## 12. NEXT STEPS

### Immediate Actions
1. ✅ **This plan is now documented** - Can be used across sessions
2. **Review and approve** - Ensure architectural direction is correct
3. **Create feature branch** - `feature/polymorphic-tagged-system`
4. **Begin Phase 1.1** - Add boundary conversion helper

### Starting Implementation

**Switch to Code Mode with:**
```
Implement Phase 1.1 of the polymorphic tagged type system refactoring plan:

Reference: docs/POLYMORPHIC_TAGGED_SYSTEM_COMPLETE_REFACTORING_PLAN.md

Tasks:
1. Add typedValueToTaggedValue() helper function after line 1269
2. Add taggedValueToTypedValue() helper function  
3. Test boundary conversion with simple examples

Start with the first helper and test before proceeding.
```

### Documentation Cross-References
- Main Plan: [`POLYMORPHIC_TAGGED_SYSTEM_COMPLETE_REFACTORING_PLAN.md`](../docs/POLYMORPHIC_TAGGED_SYSTEM_COMPLETE_REFACTORING_PLAN.md) (THIS FILE)
- Original Analysis: [`HIGHER_ORDER_REWRITE_PLAN.md`](../docs/HIGHER_ORDER_REWRITE_PLAN.md)
- Architecture Clarity: [`ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md`](../docs/ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md)
- Tagged System: [`TAGGED_VALUE_SYSTEM_IMPLEMENTATION.md`](../docs/TAGGED_VALUE_SYSTEM_IMPLEMENTATION.md)
- Build Status: [`BUILD_STATUS.md`](../docs/BUILD_STATUS.md)

---

## 13. CONCLUSION

This plan provides a complete roadmap to refactor the Eshkol codegen to a fully polymorphic tagged value system. The approach:

**✅ Makes the entire system type-aware**  
**✅ Eliminates bitcast corruption**  
**✅ Prepares for HoTT integration**  
**✅ Provides clear phase-by-phase implementation path**

The estimated effort is 60-80 hours over 3-4 weeks, but results in a robust v1.0-architecture without technical debt.

**Key Principle:** No half-measures. We're building the CORRECT type system for v1.0-architecture, with a solid foundation for future HoTT dependent types.

---

**Document Status:** COMPLETE - Ready for implementation  
**Next Action:** Review plan, then begin Phase 1.1 in Code mode  
**Success Metric:** All tests pass with proper mixed-type support