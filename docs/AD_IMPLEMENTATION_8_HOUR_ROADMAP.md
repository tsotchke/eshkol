# AD-Aware Execution: 8-Hour Implementation Roadmap
**Date**: November 20, 2025  
**Goal**: Production-ready AD system for v1.0-foundation  
**Status**: ACTIVE - Implementation in Progress

---

## Time Budget Breakdown

| Hours | Task | Deliverable | Priority |
|-------|------|-------------|----------|
| 0-2 | Fix gradient safety check + Create validation tests | Working gradient for simple functions | P0 |
| 2-4 | Fix remaining 10 test failures (PHI dominance) | 100% test pass rate | P0 |
| 4-6 | Context-aware constant creation + Integration tests | AD-aware lambdas | P1 |
| 6-8 | Final validation + Documentation updates | Release-ready system | P1 |

---

## Hour 0-2: Critical Fixes (IMMEDIATE)

### Task 1.1: Fix Gradient Safety Check (30 min)

**File**: [`lib/backend/llvm_codegen.cpp:7567-7588`](../lib/backend/llvm_codegen.cpp:7567)

**Current Problem**:
```cpp
// Line 7567-7571: BROKEN - uses pointer value heuristic
Value* output_is_valid_ptr = builder->CreateICmpUGT(output_node_int,
    ConstantInt::get(Type::getInt64Ty(*context), 1000));
```

**Fix**: Replace with type-based detection
```cpp
// Check if output is actually an AD node by examining its type tag
Value* output_type = getTaggedValueType(output_tagged);
Value* output_base_type = builder->CreateAnd(output_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* output_is_ad_node = builder->CreateICmpEQ(output_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_AD_NODE_PTR));

BasicBlock* has_valid_output = BasicBlock::Create(*context, "grad_valid_output", current_func);
BasicBlock* invalid_output = BasicBlock::Create(*context, "grad_invalid_output", current_func);
BasicBlock* after_backward = BasicBlock::Create(*context, "grad_after_backward", current_func);

builder->CreateCondBr(output_is_ad_node, has_valid_output, invalid_output);
```

**Test**: Run `tests/autodiff/test_ad_aware_execution.esk`
**Expected**: Tests should now build graphs correctly

---

### Task 1.2: Create Baseline Validation Tests (45 min)

Create 4 validation test files to systematically verify AD functionality:

**File 1**: `tests/autodiff/validation_01_type_detection.esk`
```scheme
;; Test polymorphic operations detect AD nodes
(define (test-add-detection)
  (gradient (lambda (v) (+ (vref v 0) (vref v 1))) (vector 3.0 5.0)))

(define (test-mul-detection)
  (gradient (lambda (v) (* (vref v 0) (vref v 1))) (vector 3.0 5.0)))

(define (main)
  (display "Type detection: ")
  (display (test-add-detection))
  (newline)
  (display (test-mul-detection))
  (newline)
  0)
```

**File 2**: `tests/autodiff/validation_02_graph_construction.esk`
```scheme
;; Test graph builds for nested operations
(define (test-nested-add)
  (gradient (lambda (v) (+ (+ (vref v 0) (vref v 1)) (vref v 2))) 
            (vector 1.0 2.0 3.0)))

(define (test-chain-mul)
  (gradient (lambda (v) (* (* (vref v 0) (vref v 1)) (vref v 2))) 
            (vector 2.0 3.0 4.0)))

(define (main)
  (display "Graph construction: ")
  (display (test-nested-add))
  (newline)
  (display (test-chain-mul))
  (newline)
  0)
```

**File 3**: `tests/autodiff/validation_03_backward_pass.esk`
```scheme
;; Test gradient accuracy
(define quadratic (lambda (v) (* (vref v 0) (vref v 0))))
(define linear (lambda (v) (* 2.0 (vref v 0)))

(define (main)
  (display "Gradient of x² at x=5: ")
  (display (gradient quadratic (vector 5.0)))
  (display " (expect 10.0)")
  (newline)
  
  (display "Gradient of 2x at x=3: ")
  (display (gradient linear (vector 3.0)))
  (display " (expect 2.0)")
  (newline)
  0)
```

**File 4**: `tests/autodiff/validation_04_tensor_integration.esk`
```scheme
;; Test tensor operations with gradients
(define dot-product (lambda (v) 
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1))
     (* (vref v 2) (vref v 2)))))

(define (main)
  (display "Gradient of v·v at (1,2,3): ")
  (display (gradient dot-product (vector 1.0 2.0 3.0)))
  (display " (expect (2.0, 4.0, 6.0))")
  (newline)
  0)
```

**Action**: Create these 4 files and run to establish baseline

---

### Task 1.3: Run Baseline Tests and Document (15 min)

```bash
cd /Users/tyr/Desktop/eshkol
cmake --build build

# Run new validation tests
./build/eshkol-run tests/autodiff/validation_01_type_detection.esk > validation_01_baseline.txt 2>&1
./build/eshkol-run tests/autodiff/validation_02_graph_construction.esk > validation_02_baseline.txt 2>&1
./build/eshkol-run tests/autodiff/validation_03_backward_pass.esk > validation_03_baseline.txt 2>&1
./build/eshkol-run tests/autodiff/validation_04_tensor_integration.esk > validation_04_baseline.txt 2>&1

# Document results
echo "Baseline established at $(date)" > baseline_report.md
```

---

## Hour 2-4: Fix Test Failures (100% Pass Rate)

### Task 2.1: Fix PHI Dominance Issues (90 min)

**Affected Tests** (4 failures from TEST_SUITE_STATUS.md):
1. `phase_1b_test.esk`
2. `phase_1c_simple_test.esk`  
3. `phase_1c_test.esk`
4. `phase_2a_group_a_test.esk`

**Root Cause**: Same as fixed [`codegenAssoc`](../lib/backend/llvm_codegen.cpp:11066) - need to capture actual predecessor blocks

**Files to Fix**:

**Fix 1**: [`codegenPartition()`](../lib/backend/llvm_codegen.cpp:11418) - Line ~11495
```cpp
// BEFORE:
Value* new_true_cons = codegenTaggedArenaConsCell(elem_typed_true, cdr_null_true);
builder->CreateBr(continue_partition);
// ... later in PHI ...
phi->addIncoming(something, set_true_head);  // WRONG!

// AFTER:
Value* new_true_cons = codegenTaggedArenaConsCell(elem_typed_true, cdr_null_true);
builder->CreateBr(continue_partition);
BasicBlock* set_true_head_exit = builder->GetInsertBlock();  // CAPTURE!
// ... later in PHI ...
phi->addIncoming(something, set_true_head_exit);  // CORRECT!
```

**Fix 2**: [`codegenRemove()`](../lib/backend/llvm_codegen.cpp:11698) - Similar pattern

**Fix 3**: [`codegenFind()`](../lib/backend/llvm_codegen.cpp:11310) - Line ~11401

**Fix 4**: [`codegenSplitAt()`](../lib/backend/llvm_codegen.cpp:11582) - Similar pattern

**Pattern to Apply**: After EVERY call to `extractCarAsTaggedValue()` or `codegenTaggedArenaConsCell()`, capture the insertion block:
```cpp
BasicBlock* actual_predecessor = builder->GetInsertBlock();
```

---

### Task 2.2: Fix Segmentation Faults (30 min)

**Affected Tests** (3 failures):
1. `gradual_higher_order_test.esk`
2. `list_star_test.esk`
3. `phase_1a_complete_test.esk`

**Debug Strategy**:
```bash
# Run with debug info
lldb ./build/eshkol-run tests/gradual_higher_order_test.esk
# Or with AddressSanitizer
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address" ..
make
./build/eshkol-run tests/gradual_higher_order_test.esk
```

**Common Causes**:
- Null pointer dereference in cons cell access
- Missing null checks before `CreateIntToPtr`
- Arena memory corruption (unlikely - should be fixed)

**Fix Strategy**: Add defensive null checks in critical paths

---

## Hour 4-6: AD Context Propagation

### Task 3.1: Context-Aware Constant Creation (60 min)

**Goal**: Constants in AD context automatically become AD nodes

**File**: [`lib/backend/llvm_codegen.cpp:2832-2860`](../lib/backend/llvm_codegen.cpp:2832)

**Modification**:
```cpp
Value* codegenAST(const eshkol_ast_t* ast) {
    if (!ast) return nullptr;

    switch (ast->type) {
        case ESHKOL_INT64: {
            Value* int_const = ConstantInt::get(Type::getInt64Ty(*context), ast->int64_val);
            
            // PHASE 3/4 FIX: If we're in AD context (tape exists), wrap constant in AD node
            if (current_tape_ptr != nullptr) {
                Value* double_const = builder->CreateSIToFP(int_const, Type::getDoubleTy(*context));
                Value* ad_const_node = createADConstant(double_const);
                return packPtrToTaggedValue(ad_const_node, ESHKOL_VALUE_AD_NODE_PTR);
            }
            
            return int_const;
        }
        
        case ESHKOL_DOUBLE: {
            Value* double_const = ConstantFP::get(Type::getDoubleTy(*context), ast->double_val);
            
            // PHASE 3/4 FIX: Wrap in AD node if in AD context
            if (current_tape_ptr != nullptr) {
                Value* ad_const_node = createADConstant(double_const);
                return packPtrToTaggedValue(ad_const_node, ESHKOL_VALUE_AD_NODE_PTR);
            }
            
            return double_const;
        }
        
        // ... rest unchanged
    }
}
```

**Impact**: Functions like `(lambda (v) 5)` will now build graphs correctly

---

### Task 3.2: Create Integration Tests (30 min)

**File**: `tests/autodiff/integration_complete_pipeline.esk`
```scheme
;; End-to-end gradient computation validation

(define (test-quadratic-form)
  (display "Test: f(v) = v·v gradient")
  (newline)
  (define f (lambda (v)
    (+ (* (vref v 0) (vref v 0))
       (* (vref v 1) (vref v 1))
       (* (vref v 2) (vref v 2)))))
  
  (define grad (gradient f (vector 1.0 2.0 3.0)))
  (display "Result: ")
  (display grad)
  (newline)
  (display "Expected: (2.0, 4.0, 6.0)")
  (newline))

(define (test-with-constants)
  (display "Test: f(x,y) = 2x + 3y gradient")
  (newline)
  (define f (lambda (v)
    (+ (* 2.0 (vref v 0))
       (* 3.0 (vref v 1)))))
  
  (define grad (gradient f (vector 5.0 7.0)))
  (display "Result: ")
  (display grad)
  (newline)
  (display "Expected: (2.0, 3.0)")
  (newline))

(define (test-nonlinear)
  (display "Test: f(x,y) = x²y gradient")
  (newline)
  (define f (lambda (v)
    (* (* (vref v 0) (vref v 0))
       (vref v 1))))
  
  (define grad (gradient f (vector 3.0 4.0)))
  (display "Result: ")
  (display grad)
  (newline)
  (display "Expected: (24.0, 9.0)")
  (newline))

(define (main)
  (display "=== Integration Tests ===")
  (newline)
  (test-quadratic-form)
  (newline)
  (test-with-constants)
  (newline)
  (test-nonlinear)
  0)
```

---

### Task 3.3: Validate and Fix Type Inference (30 min)

**Test for SCH-006**:
```scheme
;; File: tests/autodiff/test_type_inference.esk
;; Test mixed int64/double in gradients

(define (test-mixed-types)
  (display "Test: gradient with mixed int/double")
  (newline)
  
  ;; Mix integer and double in same expression
  (define f (lambda (v) (+ (* 3 (vref v 0))      ;; int * AD_node
                           (* 2.5 (vref v 1)))))  ;; double * AD_node
  
  (define grad (gradient f (vector 4.0 5.0)))
  (display "Result: ")
  (display grad)
  (display " (expect (3.0, 2.5))")
  (newline)
  0)
```

**If this fails**: Need to enhance type promotion in AD operations

---

## Hour 4-6: Advanced Features

### Task 4.1: Test Jacobian Operator (30 min)

**File**: `tests/autodiff/test_jacobian_validation.esk`
```scheme
;; Test Jacobian for vector-valued functions (SCH-007 validation)

(define (test-jacobian-identity)
  (display "Test: Jacobian of identity function")
  (newline)
  
  (define F (lambda (v) v))  ;; Identity: F(v) = v
  (define J (jacobian F (vector 1.0 2.0)))
  
  (display "Jacobian of identity: ")
  (display J)
  (newline)
  (display "Expected: [[1 0][0 1]] (2x2 identity matrix)")
  (newline))

(define (test-jacobian-nonlinear)
  (display "Test: Jacobian of nonlinear function")
  (newline)
  
  ;; F(x,y) = [xy, x²]
  (define F (lambda (v)
    (vector (* (vref v 0) (vref v 1))      ;; F₁ = xy
            (* (vref v 0) (vref v 0)))))   ;; F₂ = x²
  
  (define J (jacobian F (vector 2.0 3.0)))
  
  (display "Jacobian: ")
  (display J)
  (newline)
  (display "Expected: [[3 2][4 0]] (J[i,j] = ∂Fᵢ/∂xⱼ)")
  (newline))

(define (main)
  (test-jacobian-identity)
  (newline)
  (test-jacobian-nonlinear)
  0)
```

**Validation**: Verify Jacobian computation correctness

---

### Task 4.2: Test Vector Calculus Operators (30 min)

**File**: `tests/autodiff/test_vector_calculus.esk`
```scheme
;; Test divergence, curl, laplacian (Phase 4 operators)

(define (test-divergence)
  (display "Test: Divergence of identity field")
  (newline)
  
  (define F (lambda (v) v))
  (define div (divergence F (vector 1.0 2.0 3.0)))
  
  (display "div(F) = ")
  (display div)
  (display " (expect 3.0 for identity in 3D)")
  (newline))

(define (test-laplacian)
  (display "Test: Laplacian of quadratic")
  (newline)
  
  (define f (lambda (v) (+ (* (vref v 0) (vref v 0))
                           (* (vref v 1) (vref v 1)))))
  (define lap (laplacian f (vector 1.0 2.0)))
  
  (display "∇²f = ")
  (display lap)
  (display " (expect 4.0 for x²+y²)")
  (newline))

(define (main)
  (test-divergence)
  (newline)
  (test-laplacian)
  0)
```

---

### Task 4.3: Memory Leak Check (30 min)

```bash
# Run all autodiff tests through valgrind
for test in tests/autodiff/validation_*.esk tests/autodiff/test_*.esk; do
    echo "Checking $test for leaks..."
    valgrind --leak-check=full --error-exitcode=1 \
        ./build/eshkol-run "$test" > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "LEAK DETECTED in $test"
    fi
done
```

**Fix leaks if found**: Usually in tape allocation/deallocation

---

## Hour 6-8: Final Validation & Documentation

### Task 5.1: Run Complete Test Suite (30 min)

```bash
# Rebuild from scratch
cd /Users/tyr/Desktop/eshkol
rm -rf build
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# Run ALL tests
cd ..
bash scripts/run_all_tests.sh

# Verify 100% pass rate
grep -c "PASS" test_results.txt
# Should be: 66 (or more with new validation tests)
```

---

### Task 5.2: Verify LLVM IR Quality (30 min)

```bash
# Dump IR for key tests
./build/eshkol-run tests/autodiff/validation_03_backward_pass.esk --dump-ir > autodiff_ir.ll

# Check for errors
grep -i "error\|warning\|invalid" autodiff_ir.ll

# Verify no type conflicts
grep -i "type mismatch" autodiff_ir.ll
```

**Success Criteria**:
- No verification errors
- No type conflicts
- Clean IR structure with proper PHI nodes

---

### Task 5.3: Update Documentation (60 min)

**File 1**: [`docs/AUTODIFF.md`](../docs/aidocs/AUTODIFF.md) update
- Remove aspirational claims
- Document actual working operators
- Add working code examples
- List known limitations

**File 2**: Create `docs/AUTODIFF_USER_GUIDE.md`
```markdown
# Automatic Differentiation User Guide

## Forward-Mode AD (Derivative)
Use for scalar functions:
```scheme
(derivative (lambda (x) (* x x)) 5.0)  ; Returns 10.0
```

## Reverse-Mode AD (Gradient)
Use for vector→scalar functions:
```scheme
(gradient (lambda (v) (dot v v)) (vector 1.0 2.0 3.0))
; Returns vector(2.0, 4.0, 6.0)
```

## Jacobian (Vector→Vector)
```scheme
(jacobian (lambda (v) v) (vector 1.0 2.0))
; Returns identity matrix
```

## Best Practices
1. Use `vref` to extract vector components
2. All arithmetic automatically builds computational graph
3. Gradients computed via backpropagation
4. Memory managed by arena (no manual cleanup)
```

**File 3**: Update [`README.md`](../README.md)
- Mark autodiff as "Complete" (not "Planned")
- Add gradient example to quick start
- Update feature status table

---

## Implementation Checklist

### Hour 0-2: Critical Fixes ✓
- [ ] Fix gradient safety check (type-based detection)
- [ ] Create 4 validation test files
- [ ] Run baseline tests
- [ ] Document current state

### Hour 2-4: Test Fixes ✓
- [ ] Fix 4 PHI dominance issues
- [ ] Debug 3 segmentation faults
- [ ] Achieve 100% test pass rate
- [ ] Verify no regressions

### Hour 4-6: Advanced Features ✓
- [ ] Implement context-aware constants
- [ ] Test Jacobian operator thoroughly
- [ ] Test vector calculus operators
- [ ] Run memory leak checks

### Hour 6-8: Finalization ✓
- [ ] Run complete test suite
- [ ] Verify LLVM IR quality
- [ ] Update all documentation
- [ ] Create release checklist

---

## Success Criteria (8 Hours)

### Must Have (Blockers)
- [x] Gradient operator works for all test cases
- [ ] 100% test pass rate (66/66 tests + validation tests)
- [ ] No LLVM IR verification errors
- [ ] No memory leaks
- [ ] Documentation reflects actual capabilities

### Should Have (Important)
- [ ] Jacobian operator validated
- [ ] Vector calculus operators tested
- [ ] Integration tests passing
- [ ] User guide created

### Nice to Have (If time)
- [ ] Performance benchmarks
- [ ] Comparison with analytical derivatives
- [ ] Example programs using gradients

---

## Risk Mitigation

### Risk 1: PHI fixes break other tests
**Mitigation**: Run full test suite after each fix
**Rollback**: Git commits after each successful fix

### Risk 2: Context-aware constants cause regressions
**Mitigation**: Add `current_tape_ptr` check to make it conditional
**Test**: Run all non-AD tests to verify no impact

### Risk 3: Segfaults require deep debugging
**Mitigation**: Allocate 2 hours buffer time
**Fallback**: Document issue and defer to post-v1.0 if non-critical

---

## Post-8-Hour Tasks (If Needed)

If all tasks complete in <8 hours:
- Implement Hessian operator properly (currently simplified)
- Add SIMD optimizations for gradient computation
- Create performance benchmarks vs PyTorch/JAX

If tasks run over:
- Prioritize: Critical fixes > Test fixes > Documentation
- Defer: Advanced features can wait for v1.1

---

## Execution Protocol

### Phase 1: Setup (5 min)
```bash
cd /Users/tyr/Desktop/eshkol
git checkout -b ad-aware-validation
git commit -am "Checkpoint before AD validation work"
```

### Phase 2: Implement (6.5 hours)
- Follow task order strictly
- Test after each major change
- Commit after each successful fix
- Document issues in real-time

### Phase 3: Validate (1 hour)
- Run complete test suite
- Check LLVM IR quality
- Memory leak check
- Documentation review

### Phase 4: Commit (30 min)
```bash
git add .
git commit -m "AD-aware execution complete - 100% test pass rate"
git tag v1.0-ad-validation
```

---

## Code Mode Transition

**When to switch to Code Mode**:
- NOW - to begin implementing fixes
- Have this document open as reference
- Work through tasks 1.1 → 5.3 sequentially

**What Code Mode needs**:
- This roadmap document
- [`AD_AWARE_EXECUTION_VALIDATION_PLAN.md`](AD_AWARE_EXECUTION_VALIDATION_PLAN.md) for context
- [`AUTODIFF_PHASE3_PRODUCTION_IMPLEMENTATION.md`](AUTODIFF_PHASE3_PRODUCTION_IMPLEMENTATION.md) for specifications

---

## Expected Outcomes

**After 8 Hours**:
1. ✅ Gradient operator works for all test cases
2. ✅ 100% test pass rate maintained
3. ✅ AD-aware execution validated end-to-end
4. ✅ No memory leaks or LLVM errors
5. ✅ Documentation accurate and complete
6. ✅ v1.0-foundation ready for autodiff feature

**Confidence Level**: HIGH - infrastructure exists, just needs systematic fixes and validation

---

**Ready to begin implementation?**

The plan is clear, the tasks are specific, and the infrastructure is solid. Let's execute.