# Autodiff Type System Interaction Analysis
**Date**: November 27, 2025  
**Purpose**: Deep dive into how LLVM type system conflicts manifest  
**Goal**: Understand exact failure mechanisms before fixing

---

## LLVM Type System Fundamentals

### How LLVM Handles Named Struct Types

**LLVM's Type Uniquing System**:

```cpp
// First creation:
StructType* t1 = StructType::create(ctx, fields, "tensor");
// Creates: %tensor = type { ptr, i64, ptr, i64 }

// Second creation with SAME name:
StructType* t2 = StructType::create(ctx, fields, "tensor");  
// Creates: %tensor.0 = type { ptr, i64, ptr, i64 }
//          ^^^^^^^
//          Auto-renamed to avoid conflict!

// Third creation:
StructType* t3 = StructType::create(ctx, fields, "tensor");
// Creates: %tensor.1 = type { ptr, i64, ptr, i64 }
```

**Critical Points**:
1. `t1`, `t2`, `t3` are DIFFERENT C++ pointers
2. They point to DIFFERENT LLVM Type objects  
3. Even though fields are identical, they are **not interchangeable**
4. LLVM type system is **structural** but **name-based for debugging**

### Type Equivalence Rules

**LLVM considers types equivalent if**:
- Same StructType* pointer (identity equality)
- NOT if same field layout but different names

**Example**:
```llvm
%tensor = type { ptr, i64, ptr, i64 }
%tensor.0 = type { ptr, i64, ptr, i64 }

; These are DIFFERENT types to LLVM!
; Cannot use %tensor pointer with %tensor.0 GEP
```

---

## How Shadowing Creates Corruption

### Execution Flow in codegenJacobian

**Without shadowing (correct)**:
```cpp
// Class member (line 468):
tensor_type = StructType::create(*context, tensor_fields, "tensor");
// Creates: %tensor in IR

// In codegenJacobian (line 8100):
Value* test_output_ptr = builder->CreateIntToPtr(test_output_int, builder->getPtrTy());
// test_output_ptr has type: ptr

Value* output_dims_field = builder->CreateStructGEP(tensor_type, test_output_ptr, 0);
// Uses class member tensor_type → %tensor
// IR: %1 = getelementptr inbounds %tensor, ptr %test_output_ptr, i32 0, i32 0
// ✓ Correct: Consistent type usage
```

**With shadowing (broken)**:
```cpp
// Class member (line 468):
tensor_type = StructType::create(*context, tensor_fields, "tensor");
// Creates: %tensor in IR

// codegenJacobian (line 8024):
StructType* tensor_type = StructType::create(*context, tensor_fields, "tensor");
// Creates: %tensor.0 in IR (auto-renamed)
// Shadows class member!

// Line 8100 - uses LOCAL variable:
Value* output_dims_field = builder->CreateStructGEP(tensor_type, test_output_ptr, 0);
// Uses local tensor_type → %tensor.0
// IR: %1 = getelementptr inbounds %tensor.0, ptr %test_output_ptr, i32 0, i32 0

// But test_output_ptr was created by another function that used %tensor!
// Type mismatch: %tensor.0 GEP on %tensor pointer
// ✗ UNDEFINED BEHAVIOR
```

### Why This Causes Segfaults

**LLVM Optimization Behavior**:

1. **Type-based alias analysis**: Assumes different types don't alias
2. **Pointer arithmetic optimization**: Calculates offsets based on type
3. **Dead store elimination**: Removes "redundant" stores based on type
4. **Load forwarding**: Forwards loads based on type matching

**When types mismatch**:
```llvm
; Store using %tensor:
%ptr1 = getelementptr %tensor, ptr %base, i32 0, i32 0
store ptr %dims, ptr %ptr1

; Load using %tensor.0:
%ptr2 = getelementptr %tensor.0, ptr %base, i32 0, i32 0
%val = load ptr, ptr %ptr2

; Optimizer thinks:
; - %ptr1 and %ptr2 are different (different types)
; - Load doesn't depend on store
; - Can reorder/eliminate operations
; → Loads garbage!
```

**Result**:
- Loads return arbitrary values (uninitialized memory)
- Stores go to wrong offsets
- Pointer arithmetic produces invalid addresses
- **Segfault when dereferencing invalid pointer**

---

## Specific Jacobian Crash Analysis

### The Crash Location

**Runtime output shows**:
```
JACOBIAN: Stored n to dims, now setting tensor fields
RUNTIME: malloc returned 0x137704290 (size=32)
zsh: segmentation fault  ./a.out
```

**This happens at line 8416-8421**:
```cpp
builder->CreateStore(typed_jac_ad_dims,
    builder->CreateStructGEP(tensor_type, typed_jac_ad_tensor, 0));
//                       ^^^^^^^^^^^ ← Using LOCAL tensor_type (%tensor.0)

builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 1),
    builder->CreateStructGEP(tensor_type, typed_jac_ad_tensor, 1));
//                       ^^^^^^^^^^^ ← Still using LOCAL %tensor.0

builder->CreateStore(n,
    builder->CreateStructGEP(tensor_type, typed_jac_ad_tensor, 3));
//                       ^^^^^^^^^^^ ← Still using LOCAL %tensor.0
```

### Why The Crash Happens Here

**Sequence**:
1. Line 8366: Malloc creates tensor (pointer has type `ptr`)
2. Line 8376: Cast to typed pointer (still just `ptr` in LLVM)
3. Line 8417-8421: Three StructGEP calls with LOCAL tensor_type

**But earlier**, test_output was created at line 8079:
```cpp
Value* test_output_tagged = builder->CreateCall(func_ptr, {vector_tagged});
```

The function returns a tensor created with a DIFFERENT tensor_type instance!

**Result**:
- `typed_jac_ad_tensor` expects %tensor.0 layout
- But actual memory might have %tensor or %tensor.1 layout (from lambda call)
- StructGEP calculates WRONG offset
- Store writes to invalid address
- Next malloc tries to allocate from corrupted heap metadata
- **SEGFAULT**

---

## Display Type Confusion Mechanism

### The Tagged Cons Cell Structure

**File**: [`arena_memory.cpp`](../lib/core/arena_memory.cpp)
```c
struct arena_tagged_cons_cell {
    eshkol_tagged_value_t car;  // 16 bytes: {type, flags, reserved, data}
    eshkol_tagged_value_t cdr;  // 16 bytes: {type, flags, reserved, data}
};
// Total: 32 bytes
```

### The Tensor Structure

**From IR (same total size!)**:
```c
struct tensor {
    uint64_t* dimensions;   // 8 bytes: pointer
    uint64_t num_dimensions; // 8 bytes: value
    double* elements;        // 8 bytes: pointer  
    uint64_t total_elements; // 8 bytes: value
};
// Total: 32 bytes
```

**Both are 32 bytes! Coincidence creates more confusion.**

### What Happens When Display Calls Cons Functions

**Display receives**: `int64_t = 0x1376060a0` (tensor pointer)

**Line 3874 calls**:
```cpp
Value* car_type_tag = builder->CreateCall(arena_tagged_cons_get_type_func,
    {cons_check_ptr, is_car_check});
```

**Runtime execution**:
```c
// arena_tagged_cons_get_type_func expects:
struct arena_tagged_cons_cell {
    uint8_t car.type;     // Offset 0
    // ...
};

// But pointer points to:
struct tensor {
    uint64_t* dimensions; // Offset 0: e.g., 0x137606090
    // ...
};

// Reads car.type from offset 0:
// Gets: LOW BYTE of dimensions pointer
// Example: 0x137606090 → byte 0 = 0x90 = 144
// Interprets as type: 144 & 0x0F = 0 (NULL type)
//                     (144 & 0xF0) >> 4 = 9 (garbage flags)
```

**But the error says "type=32"**, so let me recalculate...

Actually, the arena functions might be doing pointer arithmetic differently. Let me check the actual structure layout:

```c
// eshkol_tagged_value_t layout:
struct eshkol_tagged_value {
    uint8_t type;        // Offset 0
    uint8_t flags;       // Offset 1
    uint16_t reserved;   // Offset 2-3
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
    } data;              // Offset 4-11 (but union, so offset 4-11 or 8-15 depending on alignment)
};
```

Wait, with proper alignment, it's likely:
```
Offset 0-1: type (uint8_t)
Offset 1-2: flags (uint8_t)
Offset 2-4: reserved (uint16_t)
Offset 4-7: padding for alignment
Offset 8-15: data (uint64_t)
```

So total is 16 bytes with alignment.

Now for cons cell:
```
Offset 0-15: car (eshkol_tagged_value_t)
Offset 16-31: cdr (eshkol_tagged_value_t)
```

When we read tensor.dimensions pointer location (offset 0-7) as cons_cell.car.type (offset 0):
- Byte 0 of pointer: varies by address
- If address is 0x137606090:
  - Byte 0: 0x90 = 144
  - As type tag: 144 & 0x0F = 0 (NULL)
  - As flags: (144 >> 4) & 0x0F = 9

But error says type=32. Let me think differently...

Type 32 = 0x20 = ESHKOL_VALUE_INEXACT_FLAG exactly. This is strange.

Maybe the error comes from a DIFFERENT location? Let me check where "Attempted to get int64 from non-int64 cell" is printed...

This is from arena_memory.cpp. When display tries to call `arena_tagged_cons_get_int64_func`, it checks the type and throws this error if type isn't INT64.

So the type=32 is what it READ from memory interpreted as a cons cell, which happens to be INEXACT_FLAG value. This could be from anywhere in the tensor structure.

The key point: **calling cons cell functions on tensor memory reads garbage**.

---

## Diagnostic Test Design

### Goal

Create a minimal test that:
1. Proves type shadowing causes LLVM IR conflicts
2. Shows exactly where segfault occurs
3. Validates fix eliminates the problem

### Test 1: LLVM IR Type Inspection

**File**: `tests/autodiff/diagnostic_type_shadowing.esk`

```scheme
;; Minimal test to expose LLVM IR type conflicts
(extern void printf char* ...)

(define (simple-jacobian)
  ;; This should trigger tensor_type creation in jacobian
  (jacobian (lambda (v) (vector 1.0 2.0)) (vector 1.0 2.0)))

(define (main)
  (display "Testing jacobian type shadowing...\n")
  (let ((result (simple-jacobian)))
    (display "Jacobian computed\n"))
  0)
```

**How to validate**:
1. Compile: `./build/eshkol-run tests/autodiff/diagnostic_type_shadowing.esk`
2. Dump IR: `./build/eshkol-run --emit-llvm tests/autodiff/diagnostic_type_shadowing.esk > jacobian.ll`
3. **Check IR for type conflicts**:
   ```bash
   grep "type.*tensor" jacobian.ll
   ```
4. **Expected BEFORE fix**:
   ```llvm
   %tensor = type { ptr, i64, ptr, i64 }
   %tensor.0 = type { ptr, i64, ptr, i64 }  ← BAD: Conflict!
   %tensor.1 = type { ptr, i64, ptr, i64 }  ← BAD: More conflicts!
   ```
5. **Expected AFTER fix**:
   ```llvm
   %tensor = type { ptr, i64, ptr, i64 }  ← ONLY one!
   ; No .0, .1, .2 suffixes
   ```

### Test 2: Gradient Display Type Check

**File**: `tests/autodiff/diagnostic_display_type.esk`

```scheme
;; Test gradient result display to expose type confusion
(extern void printf char* ...)

(define (main)
  (display "Creating gradient result...\n")
  (let ((grad (gradient (lambda (v) 0) (vector 1.0 2.0))))
    (display "Gradient result: ")
    (display grad)  ;; This triggers type confusion!
    (newline))
  0)
```

**Expected behavior**:
- **BEFORE fix**: Prints `(` with type=32 error, then crashes or garbage
- **AFTER fix**: Prints `#(0.0 0.0)` correctly

### Test 3: Jacobian Minimal Crash Reproduction

**File**: `tests/autodiff/diagnostic_jacobian_minimal.esk`

```scheme
;; Absolute minimal jacobian to trigger segfault
(extern void printf char* ...)

(define test-fn (lambda (v) (vector 1.0 2.0)))

(define (main)
  (display "Allocating input vector...\n")
  (let ((input (vector 1.0 2.0)))
    (display "Calling jacobian...\n")
    (let ((jac (jacobian test-fn input)))
      (display "Jacobian succeeded!\n")))
  0)
```

**Expected behavior**:
- **BEFORE fix**: Segfaults after "Calling jacobian..."
- **AFTER fix**: Prints "Jacobian succeeded!"

---

## Type System Interaction Patterns

### Pattern 1: Cross-Function Type Usage

**Problematic code path**:
```
1. codegenGradient creates result tensor (line 7729)
   → Uses LOCAL tensor_type → Creates %tensor.0

2. Returns tensor pointer as int64

3. Later, codegenDisplay receives this pointer

4. codegenDisplay creates ANOTHER local tensor_type (line 3899)
   → Creates %tensor.1

5. Tries to read tensor created with %tensor.0
   → Using %tensor.1 GEP instructions
   → TYPE MISMATCH
```

### Pattern 2: Nested Call Type Conflicts

**In codegenJacobian**:
```
1. Jacobian creates local tensor_type (line 8024)
   → %tensor.0

2. Calls lambda function (line 8470)
   → Lambda might use codegenTensorOperation
   → codegenTensorOperation creates local tensor_type (line 5195)
   → %tensor.1

3. Lambda returns tensor with %tensor.1 type

4. Jacobian tries to read it with %tensor.0 GEP
   → TYPE MISMATCH → Garbage read
```

### Pattern 3: Loop Iteration Type Evolution

**In jacobian double loop**:
```
Outer loop iteration 0:
  - Creates AD tensor with %tensor.0
  
Outer loop iteration 1:
  - Optimizer might have eliminated %tensor.0 reference
  - Creates NEW tensor_type? Or reuses?
  - Depends on optimizer decisions
  - **Non-deterministic behavior**
```

This explains why crash location varies!

---

## Memory Corruption Forensics

### What "type=32" Actually Means

**Error**: `Attempted to get int64 from non-int64 cell (type=32)`

**Decoding type=32**:
```c
32 decimal = 0x20 hex = 0b00100000 binary

Bit layout:
- Bits 0-3 (base type): 0000 = ESHKOL_VALUE_NULL
- Bit 4 (exact flag): 0
- Bit 5 (inexact flag): 1  ← ESHKOL_VALUE_INEXACT_FLAG
- Bits 6-7: 00
```

**This is impossible**! Type should be 0-6, and inexact flag should only be set for DOUBLE type.

**This means**: We read a random byte from tensor memory that happened to be 0x20.

**Hypothesis**: 
- Gradient result tensor at address 0x1376060a0
- Display tries to read cons_cell.car.type at offset 0
- Reads byte 0 of tensor.dimensions pointer
- If pointer is 0xXXXXXX20, byte 0 = 0x20 = 32

### Proof Experiment

Add debug output in display():
```cpp
// Before calling arena_tagged_cons_get_type_func:
if (printf_func) {
    builder->CreateCall(printf_func, {
        codegenString("DEBUG: Reading type from address %p, first 4 bytes as hex: %08x\n"),
        cons_check_ptr,
        builder->CreateLoad(Type::getInt32Ty(*context), cons_check_ptr)
    });
}
```

This would show us EXACTLY what bytes display() is reading!

---

## Validation Strategy

### Step 1: Confirm Type Shadowing via IR

**Command**:
```bash
./build/eshkol-run --emit-llvm tests/autodiff/debug_operators.esk > debug_operators.ll
grep "tensor" debug_operators.ll | grep "type.*{"
```

**Expected output BEFORE fix**:
```llvm
%tensor = type { ptr, i64, ptr, i64 }
%tensor.0 = type { ptr, i64, ptr, i64 }
%tensor.1 = type { ptr, i64, ptr, i64 }
; Multiple definitions!
```

**Expected output AFTER fix**:
```llvm
%tensor = type { ptr, i64, ptr, i64 }
; Only one definition!
```

### Step 2: Confirm Crash Location via GDB

**Commands**:
```bash
# Compile with debug info:
cmake --build build -DCMAKE_BUILD_TYPE=Debug

# Run under debugger:
gdb ./a.out
(gdb) run
# Wait for segfault...
(gdb) backtrace
(gdb) info registers
(gdb) x/32xb $rdi  # Examine memory at faulting address
```

**Expected backtrace**:
```
#0  malloc() 
#1  jacobian_lambda_...()
#2  scheme_main()
#3  main()
```

Crash is in malloc because PREVIOUS operation corrupted heap metadata!

### Step 3: Valgrind Memory Analysis

**Command**:
```bash
valgrind --leak-check=full --track-origins=yes ./a.out
```

**Expected output BEFORE fix**:
```
==12345== Invalid write of size 8
==12345==    at 0x...: jacobian_lambda_...
==12345==  Address 0x... is 16 bytes before a block of size 32 alloc'd
==12345==  
==12345== HEAP SUMMARY:
==12345==     in use at exit: 128 bytes in 4 blocks
==12345==   total heap usage: 10 allocs, 6 frees, 320 bytes allocated
```

**After fix**: No invalid writes reported.

---

## Implementation Risk Analysis

### Risk 1: Incomplete Shadow Elimination

**Scenario**: Miss one local tensor_type declaration

**Impact**: Partial type conflicts remain, intermittent crashes

**Detection**:
```bash
# After fixes, search for any remaining local declarations:
grep -n "StructType.*tensor_type.*create" lib/backend/llvm_codegen.cpp

# Should return ONLY line 468 (class init)
```

**Mitigation**: Systematic checklist of all 28 locations

### Risk 2: Breaking Non-Autodiff Code

**Scenario**: Tensor operations outside autodiff also used local types

**Impact**: Tests that worked before now fail

**Detection**: Run full test suite, compare before/after

**Mitigation**: 
- Test incrementally
- Fix critical autodiff functions first
- Validate each fix before next

### Risk 3: Type Tag Propagation Incomplete

**Scenario**: Some tensor returns don't use TENSOR_PTR tag

**Impact**: Display still can't distinguish some tensors

**Detection**: Search for all `CreatePtrToInt` on tensor pointers

**Mitigation**: Comprehensive audit of tensor return statements

---

## Fix Validation Checklist

### Pre-Fix Baseline

- [ ] Run `debug_operators.esk`, document crash location
- [ ] Save LLVM IR with type conflicts: `debug_operators_BEFORE.ll`
- [ ] Note exact error messages
- [ ] Run valgrind, save report

### Post-Fix Validation  

- [ ] Compile succeeds without LLVM errors
- [ ] LLVM IR has only ONE %tensor type
- [ ] `debug_operators.esk` runs without segfault
- [ ] Gradient displays correctly (no type=32)
- [ ] Jacobian computes matrix
- [ ] Valgrind shows no memory errors
- [ ] All 40 autodiff tests pass

---

## Next Steps

### Option A: Create Diagnostic Tests First (Recommended)

**Timeline**: 1 hour
**Benefit**: Proves root cause conclusively
**Output**: Failing tests that will pass after fix

**Steps**:
1. Create three diagnostic tests above
2. Run and document failures
3. Implement fixes
4. Re-run tests to confirm success

### Option B: Implement Fixes Immediately

**Timeline**: 6-8 hours
**Risk**: If wrong root cause, waste time
**Benefit**: Faster if root cause is correct

**Steps**:
1. Delete all local tensor_type declarations
2. Add TENSOR_PTR type tag
3. Fix display()
4. Test and validate

---

**Status**: Deep type system analysis complete  
**Recommendation**: Create diagnostic tests to validate analysis, then implement fixes  
**Next Action**: Awaiting user decision on approach