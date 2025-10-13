# Architectural Clarity: Resolving the C++/LLVM Type Boundary

## The Fundamental Mismatch - Root Cause Analysis

### Origin of the Problem

**Historical Context:**
The higher-order functions (map, filter, fold) were written BEFORE the type system existed. At that time:
- Everything was `int64_t` / `Type::getInt64Ty(*context)`
- No type tags, no unions, no TypedValue
- Simple 16-byte cons cells: `{int64 car, int64 cdr}`

**Then we added:**
1. TypedValue C++ struct (for compile-time type tracking during codegen)
2. eshkol_tagged_value LLVM struct (for runtime type preservation)
3. 24-byte tagged cons cells (for type-safe storage)

**But we never fully updated the higher-order functions!**

### The Mismatch Explained

```
C++ Domain (Compile Time)          LLVM Domain (Runtime)
=====================              ====================
TypedValue struct                  tagged_value struct
- type: enum (INT64/DOUBLE)        - type: i8 (runtime value)
- llvm_value: Value*               - data: union{i64/double}
- is_exact: bool                   - flags: i8

Known at C++ time              →   Known at IR runtime
```

**When does this cause issues?**

**AST → IR (Works Fine):**
```cpp
// We know type from AST at C++ compile time
eshkol_ast_t ast;  // type = ESHKOL_DOUBLE
TypedValue tv(codegen(ast), ESHKOL_VALUE_DOUBLE, false);  // ✅ Type known
```

**IR → IR (Problem!):**
```cpp
// Extracting from cons cell - type determined at RUNTIME
Value* cons_cell = ...;  // Points to memory
Value* car_type = getType(cons_cell);  // i8 value, determined at runtime!

// Can't create TypedValue because we don't know type at C++ compile time!
TypedValue tv(car_val, ???, ???);  // ❌ Type unknown at C++ time
```

## Why This Matters

### Functions Affected

**AST→IR Generation (No Problem):**
- `codegenAST()` - Knows types from AST
- `codegenTypedAST()` - Creates TypedValue from AST
- `codegenTaggedArenaConsCell()` - Takes TypedValue (AST-derived)

**IR→IR Operations (Problematic):**
- Extracting from cons cells (runtime types)
- Higher-order function results (runtime types)
- List operations that copy elements (runtime types)

### The Design Flaw

**TypedValue was designed for AST→IR** but we're trying to use it for IR→IR!

## Three Architectural Solutions

### Solution 1: Eliminate TypedValue (Radical Simplification)

**Principle:** Work exclusively in LLVM domain, abandon C++ type tracking

**Changes:**
```cpp
// REMOVE TypedValue completely
// REMOVE codegenTypedAST()
// CHANGE codegenTaggedArenaConsCell to:

Value* codegenTaggedArenaConsCell(Value* car_val, Value* cdr_val) {
    // Detect types from LLVM Value* types
    Type* car_llvm_type = car_val->getType();
    Type* cdr_llvm_type = cdr_val->getType();
    
    eshkol_value_type_t car_type = detectTypeFromLLVM(car_llvm_type);
    eshkol_value_type_t cdr_type = detectTypeFromLLVM(cdr_llvm_type);
    
    // Create cell with detected types
    // Use C helpers to set values
    ...
}
```

**Pros:**
- ✅ Single source of truth (LLVM IR)
- ✅ No mismatch possible
- ✅ Simpler conceptually
- ✅ Works for both AST→IR and IR→IR

**Cons:**
- ⚠️ Loses compile-time type safety in C++ code
- ⚠️ Can't track exactness as easily
- ⚠️ Bigger initial refactor

### Solution 2: Clear Boundary Separation (Current Hybrid)

**Principle:** TypedValue for AST→IR, tagged_value for IR→IR

**Design:**
```cpp
// AST→IR: Use TypedValue
TypedValue codegenTypedAST(const eshkol_ast_t* ast) {
    // Returns TypedValue with compile-time known type
}

Value* codegenTaggedArenaConsCell(const TypedValue& car, const TypedValue& cdr) {
    // Accepts TypedValue (from AST)
}

// IR→IR: Use tagged_value structs
Value* extractCarAsTaggedValue(Value* cons_ptr) {
    // Returns LLVM tagged_value struct (runtime type)
}

Value* createConsFromTaggedValues(Value* car_tagged, Value* cdr_tagged) {
    // Accepts LLVM tagged_value structs
}
```

**Usage:**
```cpp
// AST generation:
TypedValue elem = codegenTypedAST(ast);  // C++ domain
Value* cons = codegenTaggedArenaConsCell(elem, null_tv);  // C++→LLVM

// List manipulation:
Value* car_tagged = extractCarAsTaggedValue(cons);  // LLVM→LLVM
Value* new_cons = createConsFromTaggedValues(car_tagged, null_tagged);  // LLVM→LLVM
```

**Pros:**
- ✅ Clear separation of concerns
- ✅ Type safety where it matters (AST processing)
- ✅ Flexibility for runtime operations
- ✅ Both APIs available for appropriate use cases

**Cons:**
- ⚠️ Two parallel APIs (could be confusing)
- ⚠️ Need both builders

### Solution 3: Unified Tagged Value API (Best of Both Worlds)

**Principle:** Always work with tagged values, convert TypedValue to tagged_value early

**Design:**
```cpp
// Convert TypedValue to tagged_value early in pipeline
Value* typedValueToTaggedValue(const TypedValue& tv) {
    if (tv.isInt64()) {
        return packInt64ToTaggedValue(tv.llvm_value, tv.is_exact);
    } else if (tv.isDouble()) {
        return packDoubleToTaggedValue(tv.llvm_value);
    }
    // etc.
}

// Single unified cons cell builder
Value* codegenTaggedConsCell(Value* car_tagged, Value* cdr_tagged) {
    // Accepts tagged_value structs (LLVM IR)
    // Works for both AST-derived and extracted values
}

// AST generation becomes:
TypedValue elem_tv = codegenTypedAST(ast);  // C++ compile time
Value* elem_tagged = typedValueToTaggedValue(elem_tv);  // Convert once
Value* cons = codegenTaggedConsCell(elem_tagged, null_tagged);  // LLVM only

// List operations become:
Value* car_tagged = extractCarAsTaggedValue(cons);  // LLVM
Value* new_cons = codegenTaggedConsCell(car_tagged, null_tagged);  // LLVM
```

**Pros:**
- ✅ Single API (codegenTaggedConsCell)
- ✅ Clear conversion point (AST→tagged_value)
- ✅ All IR operations use same type
- ✅ Easier to understand

**Cons:**
- ⚠️ Lose TypedValue benefits for AST processing
- ⚠️ Slightly more verbose AST generation

## Recommended Solution: Solution 3 (Unified Tagged Value)

### Why This Is Best

1. **Eliminates the mismatch** - TypedValue used only temporarily during AST processing
2. **Single cons cell builder** - No confusion about which to use
3. **Consistent IR operations** - Everything works with tagged_value structs
4. **Future-proof** - Clean for HoTT where all types are runtime-determined

### Implementation Strategy

#### Step 1: Create Conversion Function

```cpp
Value* typedValueToTaggedValue(const TypedValue& tv) {
    if (tv.isInt64()) {
        return packInt64ToTaggedValue(tv.llvm_value, tv.is_exact);
    } else if (tv.isDouble()) {
        return packDoubleToTaggedValue(tv.llvm_value);
    } else if (tv.isNull()) {
        return packInt64ToTaggedValue(
            ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    } else {
        // CONS_PTR
        return packPtrToTaggedValue(tv.llvm_value, tv.type);
    }
}
```

#### Step 2: Create Unified Cons Cell Builder

```cpp
Value* codegenTaggedConsCell(Value* car_tagged, Value* cdr_tagged) {
    Value* arena_ptr = getArenaPtr();
    Value* cons_ptr = builder->CreateCall(arena_allocate_tagged_cons_cell_func, {arena_ptr});
    
    // Extract type and value from car_tagged, call appropriate setter
    Value* car_type = getTaggedValueType(car_tagged);
    Value* car_base_type = builder->CreateAnd(car_type, 
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
    
    // Branch and set based on type
    // (Similar to current codegenTaggedArenaConsCell but from tagged_value input)
    
    return builder->CreatePtrToInt(cons_ptr, Type::getInt64Ty(*context));
}
```

#### Step 3: Update Existing Functions

**codegenCons() becomes:**
```cpp
Value* codegenCons(const eshkol_operations_t* op) {
    TypedValue car_tv = codegenTypedAST(&op->call_op.variables[0]);
    TypedValue cdr_tv = codegenTypedAST(&op->call_op.variables[1]);
    
    Value* car_tagged = typedValueToTaggedValue(car_tv);
    Value* cdr_tagged = typedValueToTaggedValue(cdr_tv);
    
    return codegenTaggedConsCell(car_tagged, cdr_tagged);
}
```

**codegenList() becomes:**
```cpp
Value* codegenList(const eshkol_operations_t* op) {
    Value* null_tagged = packInt64ToTaggedValue(ConstantInt::get(...), true);
    Value* result = null_tagged;
    
    for (int64_t i = op->call_op.num_vars - 1; i >= 0; i--) {
        TypedValue elem_tv = codegenTypedAST(&op->call_op.variables[i]);
        Value* elem_tagged = typedValueToTaggedValue(elem_tv);
        result = codegenTaggedConsCell(elem_tagged, result);
    }
    
    // Return pointer, not tagged_value
    return unpackInt64FromTaggedValue(result);
}
```

**Wait, that's wrong!** `codegenTaggedConsCell` should return pointer, not tagged_value.

Let me reconsider...

#### Step 3 REVISED: Proper Return Types

```cpp
Value* codegenTaggedConsCell(Value* car_tagged, Value* cdr_tagged) {
    // ... sets values in cell ...
    return builder->CreatePtrToInt(cons_ptr, Type::getInt64Ty(*context));
    // Returns int64 pointer, NOT tagged_value
}
```

So the flow is:
```
tagged_value → createConsFromTagged() → int64 pointer → extractCarAsTaggedValue() → tagged_value
```

This makes sense! The cons cell is stored in memory, pointer is int64. Extraction gives you tagged_value.

## Complete Architecture (Solution 3 Refined)

### Type Flow Diagram

```
                   C++ Compile Time          |          LLVM IR Runtime
                                             |
AST                                          |
 ↓                                           |
codegenTypedAST()                            |
 ↓                                           |
TypedValue (C++ struct)                      |
 ↓                                           |
typedValueToTaggedValue() ----------------→ | tagged_value struct (LLVM)
                                             |  ↓
                                             | codegenTaggedConsCell()
                                             |  ↓
                                             | int64 pointer to cons cell
                                             |  ↓
                                             | extractCarAsTaggedValue()
                                             |  ↓
                                             | tagged_value struct (LLVM)
                                             |  ↓
                                             | (back to codegenTaggedConsCell)
```

### API Design

**AST Domain (C++):**
```cpp
TypedValue codegenTypedAST(const eshkol_ast_t* ast);
Value* typedValueToTaggedValue(const TypedValue& tv);
```

**IR Domain (LLVM):**
```cpp
Value* codegenTaggedConsCell(Value* car_tagged, Value* cdr_tagged);  
Value* extractCarAsTaggedValue(Value* cons_ptr_int);
Value* extractCdrAsTaggedValue(Value* cons_ptr_int);
```

**Boundary Crossing:**
```cpp
// AST → IR: Convert TypedValue to tagged_value
Value* tagged = typedValueToTaggedValue(typed_val);

// IR Display: Unpack tagged_value for printf
Value* int_val = unpackInt64FromTaggedValue(tagged_val);
Value* double_val = unpackDoubleFromTaggedValue(tagged_val);
```

### Clean Function Signatures

```cpp
class EshkolLLVMCodeGen {
private:
    // ===== AST DOMAIN (Compile-time types) =====
    
    TypedValue codegenTypedAST(const eshkol_ast_t* ast);
    
    // ===== BOUNDARY CONVERSION =====
    
    Value* typedValueToTaggedValue(const TypedValue& tv);
    
    // ===== IR DOMAIN (Runtime types) =====
    
    // Cons cell construction (from tagged values)
    Value* codegenTaggedConsCell(Value* car_tagged, Value* cdr_tagged);
    
    // Cons cell extraction (to tagged values)
    Value* extractCarAsTaggedValue(Value* cons_ptr_int);
    Value* extractCdrAsTaggedValue(Value* cons_ptr_int);
    
    // Tagged value packing (raw values → tagged values)
    Value* packInt64ToTaggedValue(Value* int64_val, bool is_exact = true);
    Value* packDoubleToTaggedValue(Value* double_val);
    Value* packPtrToTaggedValue(Value* ptr_val, eshkol_value_type_t type);
    
    // Tagged value unpacking (tagged values → raw values)
    Value* getTaggedValueType(Value* tagged_val);
    Value* unpackInt64FromTaggedValue(Value* tagged_val);
    Value* unpackDoubleFromTaggedValue(Value* tagged_val);
    Value* unpackPtrFromTaggedValue(Value* tagged_val);
};
```

## Migration Path

### Phase 2A: Add Missing Pieces (NEW)

**Add:**
1. `typedValueToTaggedValue()` - Convert boundary
2. `codegenTaggedConsCell(tagged, tagged)` - Unified builder
3. `extractCarAsTaggedValue()` - Factor from codegenCar
4. `extractCdrAsTaggedValue()` - Factor from codegenCdr

### Phase 2B: Update AST Generation

**Update these to use new API:**
- `codegenCons()` - Use typedValueToTaggedValue + codegenTaggedConsCell
- `codegenList()` - Same pattern

**KEEP:**
- `codegenTypedAST()` - Still useful for AST processing
- `codegenTaggedArenaConsCell()` - Keep as convenience wrapper

**OR Deprecate:**
- `codegenTaggedArenaConsCell()` → Call new `codegenTaggedConsCell` internally

### Phase 2C: Migrate All List Functions

Now it's mechanical! Everything uses the same 4 IR-domain functions:
```cpp
Value* car = extractCarAsTaggedValue(cons);
Value* cdr = extractCdrAsTaggedValue(cons);
Value* new_cons = codegenTaggedConsCell(car, null_tagged);
```

## Complete Rewrite Consideration

### Should We Rewrite Higher-Order Functions?

**Current State:**
- Written for pre-type system
- Assume all values are int64
- Use old 16-byte cells
- ~500 lines of complex logic

**Option A: Minimal Migration**
- Just fix struct accesses
- Keep existing logic
- Use new helpers for extraction/construction

**Option B: Complete Rewrite**
- Redesign with types from ground up
- Simplify using new helpers
- Clean code, but risky

### Recommendation: Minimal Migration First

**Why:**
1. Higher-order functions have complex logic (loops, PHI nodes, etc.)
2. That logic is correct, just uses wrong cell layout
3. Rewriting risks introducing bugs in complex logic
4. Can optimize/simplify AFTER it works

**Approach:**
- Mechanically replace old struct accesses with helper calls
- Keep all existing control flow
- Test thoroughly
- THEN consider refactoring for clarity

## Final Recommendation

### Unified Architecture Design

**1. Clear Domains:**
- **C++ Domain:** AST processing, using TypedValue temporarily
- **LLVM Domain:** IR operations, using tagged_value exclusively

**2. Single Crossing Point:**
- `typedValueToTaggedValue()` - Converts C++ → LLVM once

**3. Consistent IR Operations:**
- All cons cell work uses tagged_value structs
- Single builder: `codegenTaggedConsCell(tagged, tagged)`
- Extraction returns tagged_value

**4. Migration Strategy:**
- Add 4 core functions
- Update AST generation to use them
- Mechanically migrate list functions
- Test at each step

### Why This Solves Everything

✅ **Eliminates mismatch** - Clear C++/LLVM boundary  
✅ **Maintainable** - Single API for IR operations  
✅ **Type-safe** - Types tracked through tagged_value  
✅ **Forward-compatible** - Ready for HoTT runtime types  
✅ **Low risk** - Mechanical migration, no logic changes

## Implementation Plan

### Session 1: Core Infrastructure

1. Add `typedValueToTaggedValue()`
2. Add `codegenTaggedConsCell(tagged, tagged)` - NEW unified builder
3. Add `extractCarAsTaggedValue()` - Factor from codegenCar
4. Add `extractCdrAsTaggedValue()` - Factor from codegenCdr

### Session 2: Update Existing

5. Update `codegenCons()` to use new API
6. Update `codegenList()` to use new API
7. Test basic operations still work

### Session 3-5: Migrate Everything Else

8. Migrate length, append, reverse (proof of concept)
9. Migrate remaining 21 functions mechanically
10. Remove `codegenArenaConsCell()`

**Total Effort:** ~6-8 hours, LOW RISK

---

**Conclusion:** Solution 3 (Unified Tagged Value API) is the ideal architecture. It's clean, maintainable, and solves the mismatch at its root.

**Ready:** Detailed plan complete, ready for implementation approval