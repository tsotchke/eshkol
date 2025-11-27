# Autodiff System: Release-Ready Implementation Strategy
**Date**: November 26, 2025  
**Status**: RESEARCH COMPLETE - Comprehensive Fix Strategy  
**Goal**: Systematically fix autodiff system for Eshkol v1.0 release

---

## Table of Contents
1. [Type System Architecture Analysis](#type-system-architecture-analysis)
2. [Data Flow Analysis](#data-flow-analysis)
3. [Root Cause Identification](#root-cause-identification)
4. [Systematic Fix Strategy](#systematic-fix-strategy)
5. [Testing & Validation](#testing--validation)
6. [Implementation Checklist](#implementation-checklist)

---

## Type System Architecture Analysis

### Value Type Hierarchy

```
eshkol_value_type_t (enum, 4 bits)
├─ 0: ESHKOL_VALUE_NULL       - Empty/null value
├─ 1: ESHKOL_VALUE_INT64      - 64-bit signed integer
├─ 2: ESHKOL_VALUE_DOUBLE     - Double-precision floating point  
├─ 3: ESHKOL_VALUE_CONS_PTR   - Pointer to cons cell (for lists)
├─ 4: ESHKOL_VALUE_DUAL_NUMBER - Dual number pointer (forward-mode AD)
└─ 5: ESHKOL_VALUE_AD_NODE_PTR - AD node pointer (reverse-mode AD)
```

### Storage Mechanisms

**Tagged Values** (16 bytes):
```c
struct eshkol_tagged_value {
    uint8_t type;        // Type tag (0-5)
    uint8_t flags;       // Exactness flags
    uint16_t reserved;
    union data {
        int64_t int_val;
        double double_val;  // Stored as bitcasted int64
        uint64_t ptr_val;   // For pointers
    };
};
```

**Tensors** (32 bytes + arrays):
```c
struct tensor {
    uint64_t* dimensions;    // Field 0: Dimension sizes
    uint64_t num_dimensions; // Field 1: Number of dimensions
    int64_t* elements;       // Field 2: Element data (RAW int64 array)
    uint64_t total_elements; // Field 3: Total element count
};
```

**Key Insight**: Tensor elements are **raw int64 arrays** with **NO per-element type metadata**.

---

## Data Flow Analysis

### Normal Tensor Flow (Non-AD)
```
Parser: (vector 1.0 2.0 3.0)
  ↓
codegenTensorOperation: Creates tensor struct
  ├─ Allocates dimensions array: [3]
  ├─ Allocates elements array: [double→int64, double→int64, double→int64]
  └─ Uses BitCast to preserve double bit patterns (line 5056)
  ↓
Returns: tensor_ptr as int64
  ↓
vref: Loads element[i] as int64
  ├─ Uses IEEE754 heuristic to detect doubles
  ├─ BitCasts back to double if detected
  └─ Returns tagged_value{type=DOUBLE, data=double}
```

### AD Tensor Flow (Gradient Mode)
```
gradient operator called
  ↓
codegenGradient: Creates AD variable nodes
  ├─ Loads input vector elements as int64
  ├─ PROBLEM: Uses SIToFP instead of BitCast (line 7684)
  ├─ Creates AD nodes with WRONG values
  └─ Stores AD node pointers in new tensor elements
  ↓
Builds AD tensor: elements = [ad_node_ptr, ad_node_ptr, ...]
  ↓
Calls lambda with AD tensor
  ↓
Lambda uses vref to access elements
  ├─ vref loads int64 (which is AD node pointer)
  ├─ Heuristic tries to detect: int? double? pointer?
  ├─ PROBLEM: Can misidentify AD node pointers
  └─ Should return tagged_value{type=AD_NODE_PTR, data=ptr}
  ↓
polymorphicAdd/Mul/etc detect AD_NODE_PTR type
  ├─ Call recordADNodeBinary to build computation graph
  └─ Return new AD node wrapped in tagged_value
  ↓
Lambda returns: AD node (function output)
  ↓
codegenBackward: Runs backpropagation
  └─ Computes gradients
```

---

## Root Cause Identification

### CRITICAL ISSUE 1: SIToFP vs BitCast for Double Conversion

**Location**: [`llvm_codegen.cpp:7684`](../lib/backend/llvm_codegen.cpp:7684)

**What Happens**:
1. Tensor stores `1.0` as `0x3FF0000000000000` (IEEE754 double bits)
2. Gradient loads this as int64: `4607182418800017408`  
3. **WRONG**: `SIToFP` treats this as signed integer → creates huge floating point value
4. **RIGHT**: `BitCast` reinterprets bits as double → gets `1.0` back

**Impact**: AD variables initialized with garbage, entire computation is corrupted.

**Evidence from test**:
```
DEBUG 8: LOADED n = 1 (should be 3 for vector(1.0 2.0 3.0))
Gradient computed (may be zero vector due to implementation issues)
```

The "n = 1" is misleading - actual issue is that AD variables have wrong values.

### CRITICAL ISSUE 2: Tagged Value vs Raw Value Mixing

**Location**: [`llvm_codegen.cpp:9087-9312`](../lib/backend/llvm_codegen.cpp:9087)

**What Happens**:
1. Symbolic diff calls `codegenAST(x)` → returns tagged_value struct
2. Then calls `createTypedMul(f, g, ...)` → expects raw i64/double
3. LLVM sees: `mul i64 1, %eshkol_tagged_value %178` → TYPE MISMATCH

**Why**: `codegenAST` now returns polymorphic tagged_value (for AD support), but symbolic diff wasn't updated.

**Impact**: Compilation fails with LLVM verification errors.

### MODERATE ISSUE 3: vref Type Detection Heuristic

**Location**: [`llvm_codegen.cpp:5286-5334`](../lib/backend/llvm_codegen.cpp:5286)

**Current Logic**:
```cpp
if (value < 1000) → int
else if (has_ieee754_exponent_bits) → double  
else → AD node pointer
```

**Problem**: Some AD node pointers might have exponent-like bits, causing misidentification.

**Impact**: vref returns wrong type, polymorphic operations don't see AD nodes, graph not built.

### MINOR ISSUE 4: Lambda Vector Return = 0

**Location**: Test `debug_lambda_vector_return.esk`

**Symptom**: Lambda returns 0 instead of vector pointer

**Likely Cause**: Lambda body generation or return value packing issue, needs investigation.

---

## Systematic Fix Strategy

### Architecture Decision: Tensor Element Type Metadata

**Question**: Should tensors store typed elements or raw data?

**Current**: Raw int64 array (no per-element types)
**Pro**: Simple, memory efficient, matches C arrays
**Con**: Cannot distinguish doubles from AD node pointers at runtime

**Option A**: Add type array to tensor struct
```c
struct tensor {
    uint64_t* dimensions;
    uint64_t num_dimensions;
    int64_t* elements;          // Raw data
    uint8_t* element_types;     // NEW: Type tag per element
    uint64_t total_elements;
};
```
**Pro**: Precise type information
**Con**: Breaks ABI, requires updating all tensor operations

**Option B**: Use tagged_value array for tensors
```c
struct tensor {
    uint64_t* dimensions;
    uint64_t num_dimensions;
    eshkol_tagged_value_t* elements;  // CHANGED: Tagged elements
    uint64_t total_elements;
};
```
**Pro**: Complete type safety, works with existing tagged system
**Con**: 2x memory usage (16 bytes vs 8 bytes per element)

**Option C**: Context-aware type detection (CHOSEN)
```cpp
// Use current_tape_ptr as signal for AD mode
// In AD mode: large values are AD node pointers
// In normal mode: use IEEE754 heuristic for doubles
```
**Pro**: No ABI changes, minimal code changes
**Con**: Still heuristic-based, not 100% reliable

**Decision**: Implement Option C for v1.0, consider Option B for v1.1

---

## Systematic Fix Strategy (Detailed)

### FIX 1: Tensor Double BitCast Correction

**Scope**: All places that load double values from tensor elements

**Changes Required**:

**1a. codegenGradient - Line 7678-7684**
```cpp
// BEFORE (WRONG):
Value* elem_val_int64 = builder->CreateLoad(Type::getInt64Ty(*context), elem_ptr);
Value* elem_val = builder->CreateSIToFP(elem_val_int64, Type::getDoubleTy(*context));

// AFTER (CORRECT):
Value* elem_val_int64 = builder->CreateLoad(Type::getInt64Ty(*context), elem_ptr);
// CRITICAL: BitCast preserves IEEE754 bits, SIToFP corrupts them
Value* elem_val = builder->CreateBitCast(elem_val_int64, Type::getDoubleTy(*context));
```

**1b. codegenJacobian - Line 8062-8067**
Same fix pattern as 1a.

**1c. codegenHessian - Line 8358-8363**
Same fix pattern as 1a.

**Rationale**: Tensors store doubles using BitCast (line 5056), so reading must also use BitCast to get original value back. Using SIToFP treats the IEEE754 bit pattern as a signed integer, producing astronomically wrong values.

**Testing**: After this fix, run:
```bash
./build/eshkol-run tests/autodiff/test_gradient_minimal.esk && ./a.out
```
Expected: Non-zero gradient values (may still be wrong if other bugs exist).

---

### FIX 2: Symbolic Differentiation Type Unpacking

**Scope**: All arithmetic operations in `differentiateOperation`

**Root Cause**: After polymorphic arithmetic was added, `codegenAST` returns tagged_value structs, but symbolic diff still expects raw values.

**Pattern to Apply**: Before any arithmetic on AST-generated values, unpack them:
```cpp
// General pattern:
Value* f = codegenAST(&ast);
f = safeExtractInt64(f);  // Unpack if tagged_value, convert if needed
```

**Specific Locations**:

**2a. Product Rule (Lines 9094-9116)**
```cpp
// After line 9096:
Value* f = codegenAST(&op->call_op.variables[0]);
Value* g = codegenAST(&op->call_op.variables[1]);

// ADD:
if (f->getType() == tagged_value_type) f = safeExtractInt64(f);
if (g->getType() == tagged_value_type) g = safeExtractInt64(g);

// Then continue with f_prime, g_prime, etc.
```

**2b. Division Rule (Lines 9122-9138)**
Same pattern after line 9124.

**2c. Sin Rule (Lines 9143-9158)**  
Same pattern after line 9144.

**2d. Cos Rule (Lines 9163-9179)**
Same pattern after line 9164.

**2e. Exp Rule (Lines 9184-9209)**
Same pattern after line 9185.

**2f. Log Rule (Lines 9214-9238)**
Same pattern after line 9215.

**2g. Pow Rule (Lines 9243-9280)**
Same pattern after line 9248.

**2h. Sqrt Rule (Lines 9285-9306)**
Same pattern after line 9286.

**Alternative**: Modify `codegenAST` to return raw values for symbolic diff context, but this would require context-awareness which is more complex.

**Testing**: After this fix:
```bash
./build/eshkol-run tests/autodiff/phase0_diff_fixes.esk && ./a.out
```
Expected: Compiles without LLVM verification errors.

---

### FIX 3: vref AD Node Detection Enhancement

**Scope**: `codegenVectorRef` (lines 5245-5335)

**Current Heuristic** (lines 5286-5312):
```cpp
if (value < 1000) → int
else if (exponent_bits != 0) → double
else → AD node
```

**Problem**: AD node pointers can have any bit pattern, may match exponent pattern.

**Enhanced Detection** (Context-Aware):
```cpp
// NEW: Check if we're in AD mode by testing current_tape_ptr
Value* tape_ptr_int = builder->CreatePtrToInt(current_tape_ptr, Type::getInt64Ty(*context));
Value* in_ad_mode = builder->CreateICmpNE(tape_ptr_int, ConstantInt::get(Type::getInt64Ty(*context), 0));

// Branching logic:
if (in_ad_mode) {
    // AD mode: prioritize AD node interpretation for large values
    if (value < 1000) → int
    else → AD node (skip exponent check)
} else {
    // Normal mode: use existing heuristic
    if (value < 1000) → int  
    else if (exponent_bits != 0) → double
    else → int/pointer
}
```

**Implementation**:
Add new basic blocks in vref after line 5301:
- `ad_mode_check` - Branch on current_tape_ptr != null
- `ad_mode_large` - For AD mode, large values are AD nodes
- `normal_mode_check` - Existing IEEE754 heuristic

**Testing**: Verify gradient computation still works, potentially more reliable.

---

### FIX 4: Lambda Vector Return Investigation

**Scope**: Lambda body generation and return value handling

**Test Case**: `debug_lambda_vector_return.esk`
```scheme
(define make-vec (lambda (v) (vector 1.0 2.0)))
(define result (make-vec 0))
; result should be vector pointer, but shows 0
```

**Investigation Steps**:
1. Add debug output in `codegenLambda` (line 4854) to track body_result value
2. Verify `codegenTensorOperation` (line 5182) PtrToInt is preserved
3. Check if `detectValueType` (line 1803) correctly identifies tensor pointers
4. Trace through lambda return packing (lines 4865-4879)

**Hypothesis**: Tensor pointer is being lost during lambda return packaging.

**Potential Fixes**:
- Ensure lambda returns preserve PtrToInt results
- Verify `typedValueToTaggedValue` handles CONS_PTR correctly for tensors
- May need to distinguish tensor pointers from cons cell pointers

**Testing**: After investigation/fix:
```bash
./build/eshkol-run tests/autodiff/debug_lambda_vector_return.esk && ./a.out
```
Expected: Shows vector pointer value, not 0.

---

## Testing & Validation

### Test Categories

**Category 1: Compilation Tests**
- Goal: All tests compile without LLVM verification errors
- Tests: `phase0_diff_fixes.esk`, all phase2/3/4 tests
- Success: No "Both operands" type mismatch errors

**Category 2: Numerical Correctness Tests**  
- Goal: Gradients compute mathematically correct values
- Tests: `validation_01`, `test_gradient_minimal.esk`
- Success: Gradient of x² at x=5 returns 10, not 0

**Category 3: Segfault Prevention Tests**
- Goal: No crashes during gradient/jacobian computation
- Tests: `debug_operators.esk`, `phase2_forward_test.esk`, `phase3_complete_test.esk`, `phase4_vector_calculus_test.esk`
- Success: All execute without SIGSEGV

**Category 4: Integration Tests**
- Goal: AD works with lambdas, vectors, and tensors
- Tests: All 38 autodiff tests
- Success: 38/38 passing

### Validation Methodology

**After Each Fix**:
1. Run affected test suite
2. Check for improvements (tests passing, errors disappearing)
3. Document which tests start working
4. Verify no regressions in previously passing tests

**Final Validation**:
```bash
./scripts/run_autodiff_tests.sh
```
Expected output:
```
Total Tests:    38
Passed:         38  ← UP from 33
Failed:         0   ← DOWN from 5
Pass Rate: 100%
```

---

## Implementation Checklist

### Pre-Implementation
- [ ] Review this strategy document
- [ ] Understand type system and AD node flow
- [ ] Clarify any remaining questions
- [ ] Get approval to proceed

### Phase 1: Critical Fixes (FIX 1 & 2)
- [ ] Apply FIX 1a: codegenGradient BitCast (line 7684)
- [ ] Apply FIX 1b: codegenJacobian BitCast (line 8066)
- [ ] Apply FIX 1c: codegenHessian BitCast (line 8362)
- [ ] Test: `test_gradient_minimal.esk` shows non-zero values
- [ ] Apply FIX 2: All symbolic diff unpacking (8 locations)
- [ ] Test: `phase0_diff_fixes.esk` compiles cleanly
- [ ] Run full autodiff test suite, document improvements

### Phase 2: Enhanced Reliability (FIX 3)
- [ ] Implement context-aware vref detection
- [ ] Test: Verify gradient still works
- [ ] Test: Check for improved AD node detection
- [ ] Run full test suite, check for new passing tests

### Phase 3: Investigation & Polish (FIX 4)
- [ ] Add debug output to lambda return path
- [ ] Identify why vector return shows 0
- [ ] Implement fix if needed
- [ ] Test: `debug_lambda_vector_return.esk` shows pointer
- [ ] Final full test suite run

### Post-Implementation
- [ ] Document all changes made
- [ ] Update test expectations if needed
- [ ] Verify 100% test pass rate
- [ ] Create release notes for autodiff fixes

---

## Risk Assessment

### Fix 1: BitCast for Doubles
**Risk**: VERY LOW  
**Mitigation**: BitCast is mathematically correct for IEEE754 reinterpretation  
**Rollback**: Trivial (revert 3 lines)

### Fix 2: Tagged Value Unpacking
**Risk**: LOW  
**Mitigation**: `safeExtractInt64` is battle-tested, handles all cases  
**Rollback**: Simple (remove unpacking calls)

### Fix 3: Context-Aware vref
**Risk**: LOW  
**Mitigation**: Falls back to existing heuristic if tape is null  
**Rollback**: Moderate (revert control flow changes)

### Fix 4: Lambda Return Investigation
**Risk**: LOW (investigation only)  
**Mitigation**: No changes until root cause identified  
**Rollback**: N/A

---

## Success Criteria

### Must Have ✅
1. All 38 autodiff tests compile without errors
2. Gradient operator returns non-zero values
3. Gradient numerical values are correct (e.g., ∇(x²) = 2x)
4. No segmentation faults in any test

### Should Have ✅
5. 38/38 test pass rate (up from 33/38)
6. All compile failures resolved
7. All runtime errors resolved

### Nice to Have ⭐
8. Lambda vector returns work correctly
9. Perfect AD node type detection
10. Clean, minimal debug output

---

## Timeline

**Total Effort**: 2-3 hours

**Breakdown**:
- Fix 1 (BitCast): 20 minutes
- Fix 2 (Unpacking): 30 minutes  
- Testing & Verification: 30 minutes
- Fix 3 (vref): 30 minutes
- Fix 4 (Investigation): 20 minutes
- Final validation: 20 minutes

---

## Next Steps

1. **Review this strategy** - Confirm understanding and approach
2. **Get approval** - Ensure this comprehensive plan is correct
3. **Switch to code mode** - Begin systematic implementation
4. **Test incrementally** - Verify each fix independently
5. **Document results** - Track improvements and final state

**Questions Before Implementation**:
1. Is the type system analysis correct?
2. Is the Fix 1 (BitCast) approach sound?
3. Should we consider more fundamental changes (Option A/B) instead of Option C?
4. Any other concerns about the implementation strategy?

---

**READY FOR REVIEW AND APPROVAL**

---

**END OF IMPLEMENTATION STRATEGY**