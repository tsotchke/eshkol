# Phase 3 Comparison Fix Plan

## Executive Summary

**Problem**: ICmp assertion failure when comparing values from mixed-type lists  
**Root Cause**: Architectural boundary violation - trying to determine runtime types at C++ compile time  
**Solution**: Create `polymorphicComparison()` following the established pattern from `polymorphicAdd()`

## The Architectural Confusion

### What Went Wrong

The error occurred because we tried to fix comparison operations by modifying [`codegenTypedAST()`](lib/backend/llvm_codegen.cpp:966) to handle runtime type detection. This is **fundamentally wrong** according to our architecture.

### The Critical Distinction

From [ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md](docs/ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md):

```
C++ Domain (Compile Time)          LLVM Domain (Runtime)
=====================              ====================
TypedValue struct                  tagged_value struct
- type: enum (INT64/DOUBLE)        - type: i8 (runtime value)
- llvm_value: Value*               - data: union{i64/double}
- is_exact: bool                   - flags: i8

Known at C++ time              →   Known at IR runtime
```

**Key Principle:**
- **TypedValue** = C++ struct for tracking types WE KNOW at compile time (from AST)
- **tagged_value** = LLVM struct for types DETERMINED at runtime (from variables, function results)

### Why codegenTypedAST Cannot Handle Runtime Types

**Current Bad Code** (lines 993-1033 in [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:993)):

```cpp
// WRONG: Trying to detect type at runtime and return it in TypedValue
if (llvm_type == tagged_value_type) {
    // Extract type tag from tagged_value
    Value* type_tag = getTaggedValueType(val);
    Value* base_type = builder->CreateAnd(type_tag, ...);
    
    // Check if it's a double
    Value* is_double = builder->CreateICmpEQ(base_type, ...);
    
    // Create complex branching with PHI nodes...
    // Then at line 1033:
    return TypedValue(val, ESHKOL_VALUE_INT64, true);  // ❌ GUESS!
}
```

**Why This Fails:**
1. We create PHI nodes that determine the type at **IR runtime**
2. But TypedValue.type must be set at **C++ compile time**
3. Line 1033 just guesses INT64, which is WRONG if it's actually DOUBLE
4. When [`codegenComparison()`](lib/backend/llvm_codegen.cpp:2608) uses this TypedValue, it calls ICmp on doubles → assertion failure

### The Correct Architecture

**From AST (Compile-Time Known):**
```cpp
TypedValue codegenTypedAST(ESHKOL_INT64 ast) {
    return TypedValue(
        ConstantInt::get(..., ast->int64_val),
        ESHKOL_VALUE_INT64,  // ✅ We KNOW this at C++ time
        true
    );
}
```

**From Variables/Operations (Runtime Types):**
```cpp
Value* codegenAST(ESHKOL_VAR ast) {
    Value* val = symbol_table[name];  // Might be tagged_value
    return val;  // ❌ DON'T try to wrap in TypedValue!
}
```

## The Correct Fix Pattern

### How Arithmetic Was Fixed (Phase 1.3)

**Before (Wrong):**
```cpp
Value* codegenArithmetic(...) {
    TypedValue left = codegenTypedAST(...);   // ❌ Guesses type
    TypedValue right = codegenTypedAST(...);  // ❌ Guesses type
    auto [promoted_left, promoted_right] = promoteToCommonType(left, right);
    // ICmp/FCmp based on TypedValue.type → FAILS when guess is wrong
}
```

**After (Correct):**
```cpp
Value* codegenArithmetic(...) {
    TypedValue left_tv = codegenTypedAST(...);   // Get value with AST type
    TypedValue right_tv = codegenTypedAST(...);  // Get value with AST type
    
    // Convert to tagged_value for runtime polymorphism
    Value* left_tagged = typedValueToTaggedValue(left_tv);
    Value* right_tagged = typedValueToTaggedValue(right_tv);
    
    // Call polymorphic helper that does runtime type detection
    return polymorphicAdd(left_tagged, right_tagged);  // ✅ Handles runtime types
}
```

**The Pattern** (from [`polymorphicAdd`](lib/backend/llvm_codegen.cpp:1550)):
```cpp
Value* polymorphicAdd(Value* left_tagged, Value* right_tagged) {
    // Extract type tags AT RUNTIME
    Value* left_type = getTaggedValueType(left_tagged);
    Value* right_type = getTaggedValueType(right_tagged);
    
    // Check if either is double AT RUNTIME
    Value* any_double = builder->CreateOr(left_is_double, right_is_double);
    
    // Branch to appropriate code path
    builder->CreateCondBr(any_double, double_path, int_path);
    
    // Double path: promote and FAdd
    // Int path: Add
    // Merge with PHI: returns tagged_value
}
```

## The Complete Fix

### Step 1: Revert Bad Changes to codegenTypedAST

**What to Revert**: Lines 993-1033 in [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:993)

**Current Bad Code:**
```cpp
if (llvm_type == tagged_value_type) {
    // Complex runtime branching...
    // PHI nodes for type detection...
    // Line 1033: return TypedValue(val, ESHKOL_VALUE_INT64, true);
}
```

**Correct Replacement:**
```cpp
if (llvm_type == tagged_value_type) {
    // Don't try to detect type - just return the tagged_value as-is
    // The caller should handle it properly
    return TypedValue(val, ESHKOL_VALUE_INT64, true);
}
```

**Even Better - Just Remove the Whole Branch:**

`codegenTypedAST` should ONLY be used for AST nodes where type is known:
- ESHKOL_INT64 → TypedValue with INT64
- ESHKOL_DOUBLE → TypedValue with DOUBLE
- Variables/operations → DON'T use codegenTypedAST!

Actually, looking at the code more carefully, the issue is in the `default` case (lines 984-1043). Let me reconsider...

The cleanest fix: **Remove lines 993-1042** (the entire complex branching). Replace with:

```cpp
case ESHKOL_VAR:
case ESHKOL_OP:
default: {
    Value* val = codegenAST(ast);
    if (!val) return TypedValue();
    
    // For variables and operations, we don't know the type at compile time
    // If it's a tagged_value, return it wrapped (caller will handle properly)
    if (val->getType() == tagged_value_type) {
        return TypedValue(val, ESHKOL_VALUE_INT64, true);  // Type info is in tagged_value itself
    }
    
    // For primitive types, we can detect from LLVM type
    Type* llvm_type = val->getType();
    if (llvm_type->isIntegerTy(64)) {
        return TypedValue(val, ESHKOL_VALUE_INT64, true);
    } else if (llvm_type->isDoubleTy()) {
        return TypedValue(val, ESHKOL_VALUE_DOUBLE, false);
    } else if (llvm_type->isPointerTy()) {
        Value* as_int = builder->CreatePtrToInt(val, Type::getInt64Ty(*context));
        return TypedValue(as_int, ESHKOL_VALUE_CONS_PTR, true);
    }
    
    return TypedValue(
        ConstantInt::get(Type::getInt64Ty(*context), 0),
        ESHKOL_VALUE_NULL,
        true
    );
}
```

### Step 2: Create polymorphicComparison Helper

Following the exact pattern of [`polymorphicAdd`](lib/backend/llvm_codegen.cpp:1550), create comparison helpers:

```cpp
// Add after polymorphicDiv (around line 1740)

Value* polymorphicCompare(Value* left_tagged, Value* right_tagged, 
                         const std::string& operation) {
    if (!left_tagged || !right_tagged) {
        return packInt64ToTaggedValue(ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    }
    
    // Extract type tags
    Value* left_type = getTaggedValueType(left_tagged);
    Value* right_type = getTaggedValueType(right_tagged);
    
    Value* left_base = builder->CreateAnd(left_type,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    Value* right_base = builder->CreateAnd(right_type,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    // Check if either operand is double
    Value* left_is_double = builder->CreateICmpEQ(left_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    Value* right_is_double = builder->CreateICmpEQ(right_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    Value* any_double = builder->CreateOr(left_is_double, right_is_double);
    
    Function* current_func = builder->GetInsertBlock()->getParent();
    BasicBlock* double_path = BasicBlock::Create(*context, "cmp_double_path", current_func);
    BasicBlock* int_path = BasicBlock::Create(*context, "cmp_int_path", current_func);
    BasicBlock* merge = BasicBlock::Create(*context, "cmp_merge", current_func);
    
    builder->CreateCondBr(any_double, double_path, int_path);
    
    // Double path: promote both to double and compare
    builder->SetInsertPoint(double_path);
    Value* left_double = builder->CreateSelect(left_is_double,
        unpackDoubleFromTaggedValue(left_tagged),
        builder->CreateSIToFP(unpackInt64FromTaggedValue(left_tagged), Type::getDoubleTy(*context)));
    Value* right_double = builder->CreateSelect(right_is_double,
        unpackDoubleFromTaggedValue(right_tagged),
        builder->CreateSIToFP(unpackInt64FromTaggedValue(right_tagged), Type::getDoubleTy(*context)));
    
    Value* double_result = nullptr;
    if (operation == "lt") {
        double_result = builder->CreateFCmpOLT(left_double, right_double);
    } else if (operation == "gt") {
        double_result = builder->CreateFCmpOGT(left_double, right_double);
    } else if (operation == "eq") {
        double_result = builder->CreateFCmpOEQ(left_double, right_double);
    } else if (operation == "le") {
        double_result = builder->CreateFCmpOLE(left_double, right_double);
    } else if (operation == "ge") {
        double_result = builder->CreateFCmpOGE(left_double, right_double);
    }
    
    Value* double_result_int = builder->CreateZExt(double_result, Type::getInt64Ty(*context));
    Value* tagged_double_result = packInt64ToTaggedValue(double_result_int, true);
    builder->CreateBr(merge);
    
    // Int path: compare as int64
    builder->SetInsertPoint(int_path);
    Value* left_int = unpackInt64FromTaggedValue(left_tagged);
    Value* right_int = unpackInt64FromTaggedValue(right_tagged);
    
    Value* int_result = nullptr;
    if (operation == "lt") {
        int_result = builder->CreateICmpSLT(left_int, right_int);
    } else if (operation == "gt") {
        int_result = builder->CreateICmpSGT(left_int, right_int);
    } else if (operation == "eq") {
        int_result = builder->CreateICmpEQ(left_int, right_int);
    } else if (operation == "le") {
        int_result = builder->CreateICmpSLE(left_int, right_int);
    } else if (operation == "ge") {
        int_result = builder->CreateICmpSGE(left_int, right_int);
    }
    
    Value* int_result_extended = builder->CreateZExt(int_result, Type::getInt64Ty(*context));
    Value* tagged_int_result = packInt64ToTaggedValue(int_result_extended, true);
    builder->CreateBr(merge);
    
    // Merge results
    builder->SetInsertPoint(merge);
    PHINode* result_phi = builder->CreatePHI(tagged_value_type, 2);
    result_phi->addIncoming(tagged_double_result, double_path);
    result_phi->addIncoming(tagged_int_result, int_path);
    
    return result_phi;
}
```

### Step 3: Update codegenComparison

**Current Code** (lines 2608-2663):
```cpp
Value* codegenComparison(const eshkol_operations_t* op, const std::string& operation) {
    // Generate operands with type information
    TypedValue left_tv = codegenTypedAST(&op->call_op.variables[0]);    // ❌ WRONG
    TypedValue right_tv = codegenTypedAST(&op->call_op.variables[1]);   // ❌ WRONG
    
    // Promote to common type
    auto [left_promoted, right_promoted] = promoteToCommonType(left_tv, right_tv);
    
    // Compare based on TypedValue.type
    if (left_promoted.isInt64() && right_promoted.isInt64()) {
        result = builder->CreateICmpSLT(...);  // ❌ FAILS if actually double!
    }
}
```

**Correct Replacement:**
```cpp
Value* codegenComparison(const eshkol_operations_t* op, const std::string& operation) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("Comparison operation requires exactly 2 arguments");
        return nullptr;
    }
    
    // Generate operands with type information (for AST literals)
    TypedValue left_tv = codegenTypedAST(&op->call_op.variables[0]);
    TypedValue right_tv = codegenTypedAST(&op->call_op.variables[1]);
    
    if (!left_tv.llvm_value || !right_tv.llvm_value) return nullptr;
    
    // Convert to tagged_value for runtime polymorphism
    Value* left_tagged = typedValueToTaggedValue(left_tv);
    Value* right_tagged = typedValueToTaggedValue(right_tv);
    
    // Call polymorphic comparison that handles runtime type detection
    return polymorphicCompare(left_tagged, right_tagged, operation);
}
```

## Why This Is The Correct Approach

### HoTT Principles Applied

From [hott_runtime_representation.md](docs/hott_runtime_representation.md):

> **Proof Erasure at Runtime:**
> - All proof terms are `constexpr` and exist only during compilation
> - Runtime structures contain only essential data
> - Type information encoded in template specializations, not runtime tags

In our current Phase 3 implementation:
- **TypedValue** = Our "proof" that we know the type (compile-time)
- **tagged_value** = Runtime data with type tag (runtime polymorphism)
- **Polymorphic functions** = Handle runtime type dispatch

### The Boundary Crossing Pattern

From [ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md](docs/ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md:459):

```
AST Domain → typedValueToTaggedValue() → IR Domain
              (Cross boundary once)
```

**For comparison operations:**

```
1. AST nodes (+ 10 20) or variables (+ x y)
   ↓
2. codegenTypedAST() - Get TypedValue (compile-time type info)
   ↓
3. typedValueToTaggedValue() - Convert to LLVM tagged_value
   ↓
4. polymorphicCompare() - Runtime type detection and comparison
   ↓
5. Result: tagged_value with int64 result (0 or 1)
```

### Why polymorphicAdd Works

Looking at [`polymorphicAdd`](lib/backend/llvm_codegen.cpp:1550-1605):

1. **Accepts tagged_value** - No compile-time type assumptions
2. **Extracts types at runtime** - Uses IR instructions to check type tags
3. **Branches appropriately** - Double path or int path
4. **Promotes as needed** - SIToFP for int→double
5. **Returns tagged_value** - Preserves type information

**Comparison must follow the EXACT same pattern!**

## Implementation Details

### Code Locations

**File**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

**Changes Required:**

1. **Lines 993-1042**: DELETE complex runtime type detection in `codegenTypedAST`
2. **After line 1739**: ADD `polymorphicCompare()` helper
3. **Lines 2608-2663**: REPLACE `codegenComparison` implementation

### The Reverted codegenTypedAST

**Remove Complex Branching** (lines 993-1042):

The default case should be simple:

```cpp
case ESHKOL_VAR:
case ESHKOL_OP:
default: {
    Value* val = codegenAST(ast);
    if (!val) return TypedValue();
    
    // Simple type detection from LLVM type (no complex branching!)
    Type* llvm_type = val->getType();
    
    if (llvm_type == tagged_value_type) {
        // It's a tagged_value - type is embedded, return as-is
        // Caller must use polymorphic operations on it
        return TypedValue(val, ESHKOL_VALUE_INT64, true);
    } else if (llvm_type->isIntegerTy(64)) {
        return TypedValue(val, ESHKOL_VALUE_INT64, true);
    } else if (llvm_type->isDoubleTy()) {
        return TypedValue(val, ESHKOL_VALUE_DOUBLE, false);
    } else if (llvm_type->isPointerTy()) {
        Value* as_int = builder->CreatePtrToInt(val, Type::getInt64Ty(*context));
        return TypedValue(as_int, ESHKOL_VALUE_CONS_PTR, true);
    }
    
    return TypedValue(
        ConstantInt::get(Type::getInt64Ty(*context), 0),
        ESHKOL_VALUE_NULL,
        true
    );
}
```

**Key Point**: When `llvm_type == tagged_value_type`, we **don't try to detect** the actual type. We just return the tagged_value wrapped in TypedValue. The caller (like `codegenComparison`) must then:
1. Convert it to tagged_value using `typedValueToTaggedValue`
2. Pass to polymorphic function that does runtime type detection

### The New polymorphicCompare Function

**Location**: After [`polymorphicDiv`](lib/backend/llvm_codegen.cpp:1711-1739) (around line 1740)

**Full Implementation**:

```cpp
// ===== POLYMORPHIC COMPARISON FUNCTIONS (Phase 3 Fix) =====
// Handle mixed-type comparisons with runtime type detection

Value* polymorphicCompare(Value* left_tagged, Value* right_tagged, 
                         const std::string& operation) {
    if (!left_tagged || !right_tagged) {
        return packInt64ToTaggedValue(ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    }
    
    // Extract type tags
    Value* left_type = getTaggedValueType(left_tagged);
    Value* right_type = getTaggedValueType(right_tagged);
    
    Value* left_base = builder->CreateAnd(left_type,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    Value* right_base = builder->CreateAnd(right_type,
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    // Check if either operand is double
    Value* left_is_double = builder->CreateICmpEQ(left_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    Value* right_is_double = builder->CreateICmpEQ(right_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    Value* any_double = builder->CreateOr(left_is_double, right_is_double);
    
    Function* current_func = builder->GetInsertBlock()->getParent();
    BasicBlock* double_path = BasicBlock::Create(*context, "cmp_double_path", current_func);
    BasicBlock* int_path = BasicBlock::Create(*context, "cmp_int_path", current_func);
    BasicBlock* merge = BasicBlock::Create(*context, "cmp_merge", current_func);
    
    builder->CreateCondBr(any_double, double_path, int_path);
    
    // Double path: promote both to double and compare
    builder->SetInsertPoint(double_path);
    Value* left_double = builder->CreateSelect(left_is_double,
        unpackDoubleFromTaggedValue(left_tagged),
        builder->CreateSIToFP(unpackInt64FromTaggedValue(left_tagged), Type::getDoubleTy(*context)));
    Value* right_double = builder->CreateSelect(right_is_double,
        unpackDoubleFromTaggedValue(right_tagged),
        builder->CreateSIToFP(unpackInt64FromTaggedValue(right_tagged), Type::getDoubleTy(*context)));
    
    Value* double_cmp = nullptr;
    if (operation == "lt") {
        double_cmp = builder->CreateFCmpOLT(left_double, right_double);
    } else if (operation == "gt") {
        double_cmp = builder->CreateFCmpOGT(left_double, right_double);
    } else if (operation == "eq") {
        double_cmp = builder->CreateFCmpOEQ(left_double, right_double);
    } else if (operation == "le") {
        double_cmp = builder->CreateFCmpOLE(left_double, right_double);
    } else if (operation == "ge") {
        double_cmp = builder->CreateFCmpOGE(left_double, right_double);
    }
    Value* double_result_int = builder->CreateZExt(double_cmp, Type::getInt64Ty(*context));
    Value* tagged_double_result = packInt64ToTaggedValue(double_result_int, true);
    builder->CreateBr(merge);
    
    // Int path: compare as int64
    builder->SetInsertPoint(int_path);
    Value* left_int = unpackInt64FromTaggedValue(left_tagged);
    Value* right_int = unpackInt64FromTaggedValue(right_tagged);
    
    Value* int_cmp = nullptr;
    if (operation == "lt") {
        int_cmp = builder->CreateICmpSLT(left_int, right_int);
    } else if (operation == "gt") {
        int_cmp = builder->CreateICmpSGT(left_int, right_int);
    } else if (operation == "eq") {
        int_cmp = builder->CreateICmpEQ(left_int, right_int);
    } else if (operation == "le") {
        int_cmp = builder->CreateICmpSLE(left_int, right_int);
    } else if (operation == "ge") {
        int_cmp = builder->CreateICmpSGE(left_int, right_int);
    }
    Value* int_result_extended = builder->CreateZExt(int_cmp, Type::getInt64Ty(*context));
    Value* tagged_int_result = packInt64ToTaggedValue(int_result_extended, true);
    builder->CreateBr(merge);
    
    // Merge results
    builder->SetInsertPoint(merge);
    PHINode* result_phi = builder->CreatePHI(tagged_value_type, 2);
    result_phi->addIncoming(tagged_double_result, double_path);
    result_phi->addIncoming(tagged_int_result, int_path);
    
    return result_phi;
}
```

### The Updated codegenComparison

**Location**: Lines 2608-2663

**New Implementation**:

```cpp
Value* codegenComparison(const eshkol_operations_t* op, const std::string& operation) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("Comparison operation requires exactly 2 arguments");
        return nullptr;
    }
    
    // Generate operands with type information
    TypedValue left_tv = codegenTypedAST(&op->call_op.variables[0]);
    TypedValue right_tv = codegenTypedAST(&op->call_op.variables[1]);
    
    if (!left_tv.llvm_value || !right_tv.llvm_value) return nullptr;
    
    // Convert to tagged_value for runtime polymorphism
    Value* left_tagged = typedValueToTaggedValue(left_tv);
    Value* right_tagged = typedValueToTaggedValue(right_tv);
    
    // Call polymorphic comparison that handles runtime type detection
    return polymorphicCompare(left_tagged, right_tagged, operation);
}
```

## Why This Follows HoTT Principles

### Type Preservation Through Boundaries

From [hott_llvm_integration.md](docs/hott_llvm_integration.md:9):

> **Enhanced TypedValue with Proof Information:**
> - Compile-time proof information (erased at runtime)
> - Runtime type tag (for compatibility)

Our fix maintains this separation:

1. **Compile-Time (TypedValue)**:
   - Used for AST nodes where type is KNOWN
   - Quick path for literal values
   - Type safety where possible

2. **Runtime (tagged_value)**:
   - Used for variables and operation results
   - Dynamic type dispatch
   - Full polymorphism

### The Clean Architecture

```
┌─────────────────────────────────────────────────────────┐
│ C++ Compile Time (Type Proofs)                          │
├─────────────────────────────────────────────────────────┤
│ AST Nodes (ESHKOL_INT64, ESHKOL_DOUBLE)                │
│   ↓                                                      │
│ codegenTypedAST() - Creates TypedValue                  │
│   ↓                                                      │
│ TypedValue {                                            │
│   llvm_value: Value*,                                   │
│   type: KNOWN at C++ compile time,                     │
│   is_exact: bool                                        │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                         ↓
           typedValueToTaggedValue()
                  (BOUNDARY)
                         ↓
┌─────────────────────────────────────────────────────────┐
│ LLVM IR Runtime (Dynamic Types)                         │
├─────────────────────────────────────────────────────────┤
│ tagged_value {                                          │
│   type: i8 (runtime check),                            │
│   flags: i8,                                            │
│   data: union                                           │
│ }                                                        │
│   ↓                                                      │
│ polymorphicAdd(tagged, tagged)                          │
│ polymorphicCompare(tagged, tagged)                      │
│   ↓                                                      │
│ Runtime type detection via IR instructions              │
│ Branch to int_path or double_path                       │
│   ↓                                                      │
│ PHI merge → tagged_value result                         │
└─────────────────────────────────────────────────────────┘
```

### Why Previous Arithmetic Works

[`codegenArithmetic`](lib/backend/llvm_codegen.cpp:2572) already follows this pattern:

```cpp
// Line 2578-2585: Convert to tagged_value
for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
    TypedValue tv = codegenTypedAST(&op->call_op.variables[i]);
    if (!tv.llvm_value) continue;
    Value* tagged = typedValueToTaggedValue(tv);  // ✅ Convert once
    tagged_operands.push_back(tagged);
}

// Line 2592-2600: Use polymorphic operations
if (operation == "add") {
    result = polymorphicAdd(result, tagged_operands[i]);  // ✅ Runtime dispatch
}
```

**Comparison must use the SAME pattern!**

## Verification Strategy

### Test Cases

1. **Integer-only comparison** (should still work):
   ```scheme
   (< 5 10)  ; Both int64, known at compile time
   ```

2. **Double-only comparison** (should still work):
   ```scheme
   (< 2.5 3.7)  ; Both double, known at compile time
   ```

3. **Mixed literal comparison** (already works):
   ```scheme
   (< 5 2.5)  ; Mixed types, promotes to double
   ```

4. **Variable comparison** (THIS is what fails now):
   ```scheme
   (define x 10)
   (define y 2.5)
   (< x y)  ; x is tagged_value(int64), y is tagged_value(double)
   ```

5. **List element comparison** (critical test):
   ```scheme
   (define list1 (list 5 10 15))
   (define list2 (list 2.5 3.7 8.9))
   (map < list1 list2)  ; Comparing elements from mixed-type lists
   ```

### Expected Behavior After Fix

**Test 4 Should Work:**
```
1. x = tagged_value{type=INT64, data=10}
2. y = tagged_value{type=DOUBLE, data=2.5}
3. polymorphicCompare extracts types at runtime
4. Sees INT64 and DOUBLE → takes double_path
5. Promotes 10 to 10.0
6. FCmpOLT(10.0, 2.5) → false (0)
7. Returns tagged_value{type=INT64, data=0}
```

**Test 5 Should Work:**
```
map extracts: car(list1) → tagged_value{INT64, 5}
              car(list2) → tagged_value{DOUBLE, 2.5}
Passes both to polymorphicCompare (via < wrapper function)
Runtime detection handles the type difference correctly
```

## Architectural Lessons Learned

### The Core Mistake

**What We Did Wrong:**
> "Let's make codegenTypedAST smarter to detect runtime types!"

**Why It Failed:**
> TypedValue is a C++ struct. Its `type` field must be set at C++ compile time. 
> We can't set it based on runtime PHI node results.

**The Correct Thinking:**
> "TypedValue is for compile-time. For runtime types, use tagged_value and polymorphic helpers."

### The Pattern to Remember

**When do we use TypedValue?**
- ✅ AST literals (type known from AST node type)
- ✅ Quick boundary crossing to tagged_value
- ❌ Variables (type determined at runtime)
- ❌ Operation results (type determined at runtime)

**When do we use polymorphic functions?**
- ✅ Arithmetic on variables
- ✅ Arithmetic on mixed types
- ✅ **Comparisons** (the fix we're applying)
- ✅ Any operation where types are runtime-determined

### Future-Proofing for Full HoTT

From [hott_llvm_integration.md](docs/hott_llvm_integration.md:16):

> **ProofCarryingValue** - Enhanced TypedValue that carries proof information
> - Compile-time proof information (erased at runtime)
> - Runtime type tag (for compatibility)

Our fix maintains compatibility:
- TypedValue → Will become ProofCarryingValue
- tagged_value → Runtime representation
- Polymorphic functions → Runtime dispatch (proof-optimized later)

## Implementation Checklist

- [ ] Revert lines 993-1042 in `codegenTypedAST` (remove runtime type detection)
- [ ] Add `polymorphicCompare()` after line 1739 (following polymorphicDiv pattern)
- [ ] Replace `codegenComparison()` implementation (lines 2608-2663)
- [ ] Test with phase3b_debug.esk (simple mixed arithmetic)
- [ ] Test with comparison operations on variables
- [ ] Test with map operations that include comparisons
- [ ] Verify no ICmp assertion failures

## Success Criteria

1. ✅ No ICmp/FCmp type assertion failures
2. ✅ Correct results for mixed-type comparisons
3. ✅ Variables with tagged_value types work correctly
4. ✅ List operations with comparisons work
5. ✅ Code follows the established polymorphic pattern
6. ✅ Clear architectural boundary maintained

## Related Documentation

- [ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md](docs/ARCHITECTURAL_CLARITY_CPP_LLVM_BOUNDARY.md) - The TypedValue/tagged_value distinction
- [hott_runtime_representation.md](docs/hott_runtime_representation.md) - Runtime type erasure principles
- [hott_llvm_integration.md](docs/hott_llvm_integration.md) - Proof-carrying values and optimization

---

**Status**: Plan complete, ready for code mode implementation  
**Estimated LOC Changes**: ~150 lines (40 removed, 110 added)  
**Risk Level**: LOW (following established pattern)  
**Testing Required**: Comprehensive (all comparison operations)