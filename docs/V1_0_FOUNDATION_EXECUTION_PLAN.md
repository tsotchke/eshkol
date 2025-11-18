# Eshkol v1.0-Foundation Execution Plan

**Status**: ACTIVE - Ready for Session 21  
**Created**: November 17, 2025  
**Target Completion**: Q1 2026 (Sessions 21-60)  
**Previous Milestone**: v1.0-architecture COMPLETE ✅

---

## Executive Summary

### Current State
- **v1.0-architecture COMPLETE** (Month 1 finished, Sessions 1-20)
- **Test Pass Rate**: 100% (66/66 tests passing)
- **Memory Safety**: Zero unsafe operations, all via type-safe helpers
- **Higher-Order Functions**: 17/17 migrated to tagged value system
- **Achievement**: Type-safe polymorphic foundation established

### v1.0-Foundation Vision
Production-ready release with:
1. **Full Autodiff Functionality** - All critical bugs resolved (SCH-006/007/008)
2. **Automated Infrastructure** - CI/CD catching regressions automatically
3. **Production Examples** - 30 curated, working examples
4. **Distribution Ready** - Packages for Ubuntu/Debian + macOS
5. **Accurate Documentation** - Reality-based, no aspirational claims

### Strategic Decision: Option A - Full Autodiff Fixes
We've chosen to include complete autodiff bug fixes for maximum value in v1.0-foundation. This provides:
- Reliable scientific computing capabilities
- Confidence in the type system integration
- Strong foundation for future tensor operations (Phase 2)

---

## Month 2: Autodiff & Examples (Sessions 21-40)

### Week 3: Autodiff Type System (Sessions 21-30)

#### **Session 21-22: Investigation & Root Cause Analysis**

**Objective**: Deep understanding of autodiff type system issues

**Tasks**:
1. **Read autodiff implementation** in [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)
   - Search for "autodiff", "gradient", "derivative" functions
   - Identify type inference code paths
   - Map LLVM IR generation for autodiff

2. **Create test cases** for each bug:
   - SCH-006: Type inference incomplete
     ```scheme
     ;; Should infer types correctly
     (define f (lambda (x) (* x x)))
     (d/dx f 5)  ; Should work without explicit types
     ```
   - SCH-007: Vector return types not handled
     ```scheme
     (gradient (lambda (v) (dot v v)) #(1 2 3))  ; Should return vector
     ```
   - SCH-008: Type conflicts in generated code
     ```scheme
     ;; Complex autodiff should compile without IR errors
     (define g (lambda (x y) (+ (* x x) (* y y))))
     (gradient-2d g 1 2)
     ```

3. **Document findings** in `docs/AUTODIFF_TYPE_ANALYSIS.md`:
   - Type flow through autodiff system
   - Where type inference fails
   - LLVM IR generation issues
   - Proposed fix strategies

**Deliverables**:
- `docs/AUTODIFF_TYPE_ANALYSIS.md` - Root cause analysis
- `tests/autodiff_debug/` - Minimal reproducible tests
- Clear understanding of fix requirements

**Success Criteria**:
- ✅ All three bugs fully understood
- ✅ Fix strategies documented
- ✅ Test cases reproduce issues reliably

---

#### **Session 23-24: SCH-006 Fix (Type Inference)**

**Objective**: Complete type inference for autodiff functions

**Background**: 
Current issue is that autodiff functions don't properly infer return types, especially for:
- Scalar → scalar derivatives (should be scalar)
- Scalar → vector derivatives (Jacobian, should be vector)
- Vector → scalar derivatives (gradient, should be vector)

**Implementation**:

1. **Locate type inference code** (likely in `lib/backend/llvm_codegen.cpp` around autodiff functions)

2. **Implement type inference rules**:
   ```cpp
   // Pseudocode for type inference
   Type* inferAutodiffReturnType(FunctionType* inputFunc, Mode mode) {
       if (mode == FORWARD_MODE) {
           // Forward mode: d/dx f returns same type as f
           return inputFunc->getReturnType();
       } else if (mode == REVERSE_MODE) {
           // Reverse mode (gradient): always returns vector
           Type* inputType = inputFunc->getParamType(0);
           if (inputType->isVector()) {
               return inputType;  // Vector input → vector gradient
           } else {
               // Scalar input → should error or return scalar
           }
       }
   }
   ```

3. **Handle edge cases**:
   - Multiple input variables
   - Composition of autodiff operations
   - Mixed scalar/vector operations

4. **Create tests**:
   ```scheme
   ;; Test scalar derivative
   (define f1 (lambda (x) (* x x)))
   (display (d/dx f1 5))  ; => 10 (type: scalar)
   
   ;; Test gradient
   (define f2 (lambda (v) (dot v v)))
   (display (gradient f2 #(1 2 3)))  ; => #(2 4 6) (type: vector)
   
   ;; Test composition
   (define g (lambda (x) (* x x)))
   (define h (lambda (x) (+ x 1)))
   (display (d/dx (compose g h) 2))  ; => 6
   ```

**Files to Modify**:
- `lib/backend/llvm_codegen.cpp` - Type inference implementation
- `tests/autodiff_type_inference_test.esk` - Comprehensive tests

**Deliverables**:
- SCH-006 resolved
- Type inference working for all autodiff modes
- Comprehensive test coverage

**Success Criteria**:
- ✅ All type inference tests pass
- ✅ No manual type annotations needed for simple cases
- ✅ Composition works correctly

---

#### **Session 25-26: SCH-007 Fix (Vector Returns)**

**Objective**: Proper LLVM vector type returns for gradient functions

**Background**:
Vector return types not correctly handled in LLVM IR generation. This affects:
- `gradient` function (should return vector)
- Jacobian computations
- Any autodiff operation returning multiple values

**Implementation**:

1. **Analyze current vector return handling**:
   - Find where vector returns are generated
   - Identify LLVM IR type mismatches
   - Document current vs. correct approach

2. **Implement proper vector returns**:
   ```cpp
   // Pseudocode for vector return
   Value* codegenGradient(const eshkol_operations_t* op) {
       // Get function to differentiate
       Function* func = ...;
       
       // Compute gradient (returns array of derivatives)
       std::vector<Value*> derivatives;
       for (size_t i = 0; i < numInputs; i++) {
           derivatives.push_back(computePartialDerivative(func, i));
       }
       
       // Pack into LLVM vector type
       Type* elementType = derivatives[0]->getType();
       VectorType* vecType = VectorType::get(elementType, numInputs, false);
       
       Value* result = UndefValue::get(vecType);
       for (size_t i = 0; i < numInputs; i++) {
           result = builder->CreateInsertElement(result, derivatives[i], i);
       }
       
       return result;
   }
   ```

3. **Handle return value storage**:
   - Vectors may need special allocation
   - Return via pointer if too large
   - Ensure proper calling convention

4. **Create tests**:
   ```scheme
   ;; Test gradient return
   (define f (lambda (v) (dot v v)))
   (define grad (gradient f #(1 2 3)))
   (display (vector-ref grad 0))  ; => 2
   (display (vector-ref grad 1))  ; => 4
   (display (vector-ref grad 2))  ; => 6
   
   ;; Test Jacobian (if implemented)
   (define f2 (lambda (v) (make-vector (+ (vector-ref v 0) (vector-ref v 1))
                                       (* (vector-ref v 0) (vector-ref v 1)))))
   (define jac (jacobian f2 #(2 3)))
   ;; Should return 2x2 matrix
   ```

**Files to Modify**:
- `lib/backend/llvm_codegen.cpp` - Vector return implementation
- `inc/eshkol/eshkol.h` - Vector type definitions (if needed)
- `tests/autodiff_vector_returns_test.esk` - Vector return tests

**Deliverables**:
- SCH-007 resolved
- Vector returns working correctly
- Gradient function fully functional

**Success Criteria**:
- ✅ Gradient returns vector correctly
- ✅ Vector elements accessible via vector-ref
- ✅ LLVM IR passes verification

---

#### **Session 27-28: SCH-008 Fix (Type Conflicts)**

**Objective**: Resolve type conflicts in generated LLVM IR

**Background**:
Type conflicts occur when:
- Same value has different LLVM types in different basic blocks
- Type conversions not properly inserted
- PHI nodes have mismatched incoming types

**Implementation**:

1. **Identify conflict locations**:
   - Run LLVM verifier on generated IR
   - Parse error messages
   - Locate conflicting type declarations

2. **Common conflict patterns**:
   ```cpp
   // Pattern 1: Scalar/Vector mismatch
   // FIX: Insert proper vector constructors
   if (needsVectorization) {
       scalar = builder->CreateInsertElement(
           UndefValue::get(VectorType::get(scalar->getType(), 1, false)),
           scalar, 0);
   }
   
   // Pattern 2: Integer/Float mismatch
   // FIX: Insert explicit conversions
   if (srcType->isIntegerTy() && destType->isFloatingPointTy()) {
       value = builder->CreateSIToFP(value, destType);
   }
   
   // Pattern 3: PHI node type mismatch
   // FIX: Ensure all incoming values have same type
   PHINode* phi = builder->CreatePHI(targetType, numIncoming);
   for (auto& incoming : incomingValues) {
       if (incoming.value->getType() != targetType) {
           incoming.value = convertType(incoming.value, targetType);
       }
       phi->addIncoming(incoming.value, incoming.block);
   }
   ```

3. **Implement type unification**:
   - Create helper function to find common type
   - Insert conversions automatically
   - Maintain type consistency across basic blocks

4. **Verify fixes**:
   ```bash
   # Should compile without errors
   ./build/eshkol-run tests/complex_autodiff.esk
   
   # Verify IR
   ./build/eshkol-run --emit-llvm tests/complex_autodiff.esk
   llvm-as generated.ll && echo "Valid LLVM IR"
   ```

**Files to Modify**:
- `lib/backend/llvm_codegen.cpp` - Type unification
- `tests/autodiff_type_conflicts_test.esk` - Conflict tests

**Deliverables**:
- SCH-008 resolved
- Type conflicts eliminated
- Clean LLVM IR generation

**Success Criteria**:
- ✅ All autodiff programs compile
- ✅ LLVM IR passes verification
- ✅ No type errors in generated code

---

#### **Session 29-30: Comprehensive Autodiff Testing**

**Objective**: Exhaustive testing of autodiff system

**Test Suite Structure**:

```scheme
;; tests/autodiff_comprehensive.esk

;; ============================================
;; PART 1: Forward Mode Derivatives
;; ============================================

;; Simple derivatives
(define test-forward-simple
  (lambda ()
    (display "Testing forward mode - simple functions")
    
    ;; d/dx(x²) = 2x
    (define f1 (lambda (x) (* x x)))
    (assert-equal (d/dx f1 5) 10)
    
    ;; d/dx(sin(x))
    (define f2 (lambda (x) (sin x)))
    (assert-approx (d/dx f2 0) 1.0 0.001)
    
    ;; d/dx(e^x) = e^x
    (define f3 (lambda (x) (exp x)))
    (assert-approx (d/dx f3 0) 1.0 0.001)))

;; Composition
(define test-forward-composition
  (lambda ()
    (display "Testing forward mode - composition")
    
    ;; d/dx(sin(x²))
    (define f (lambda (x) (* x x)))
    (define g (lambda (x) (sin x)))
    (define h (compose g f))
    (display (d/dx h 1.0))  ; Should be 2x*cos(x²) at x=1
    ))

;; Multi-variable
(define test-forward-multi-var
  (lambda ()
    (display "Testing forward mode - multi-variable")
    
    ;; ∂/∂x(x²y) = 2xy
    (define f (lambda (x y) (* (* x x) y)))
    (assert-equal (partial-x f 3 4) 24)))

;; ============================================
;; PART 2: Reverse Mode (Gradient)
;; ============================================

(define test-reverse-gradient
  (lambda ()
    (display "Testing reverse mode - gradients")
    
    ;; ∇(v·v) = 2v
    (define f1 (lambda (v) (dot v v)))
    (define g1 (gradient f1 #(1 2 3)))
    (assert-vector-equal g1 #(2 4 6))
    
    ;; ∇(x² + y² + z²) = (2x, 2y, 2z)
    (define f2 (lambda (v) 
      (+ (+ (* (vector-ref v 0) (vector-ref v 0))
            (* (vector-ref v 1) (vector-ref v 1)))
         (* (vector-ref v 2) (vector-ref v 2)))))
    (define g2 (gradient f2 #(1 2 3)))
    (assert-vector-equal g2 #(2 4 6))))

;; ============================================
;; PART 3: Higher-Order Derivatives
;; ============================================

(define test-higher-order
  (lambda ()
    (display "Testing higher-order derivatives")
    
    ;; d²/dx²(x⁴) = 12x²
    (define f (lambda (x) (* (* x x) (* x x))))
    (define df (lambda (x) (d/dx f x)))
    (assert-equal (d/dx df 2) 48)))

;; ============================================
;; PART 4: Edge Cases
;; ============================================

(define test-edge-cases
  (lambda ()
    (display "Testing edge cases")
    
    ;; Constant function
    (define f1 (lambda (x) 5))
    (assert-equal (d/dx f1 10) 0)
    
    ;; Linear function
    (define f2 (lambda (x) (+ (* 3 x) 5)))
    (assert-equal (d/dx f2 10) 3)
    
    ;; Zero gradient
    (define f3 (lambda (v) 0))
    (define g3 (gradient f3 #(1 2 3)))
    (assert-vector-equal g3 #(0 0 0))))

;; ============================================
;; PART 5: Performance Benchmarks
;; ============================================

(define test-performance
  (lambda ()
    (display "Testing autodiff performance")
    
    ;; Benchmark against hand-coded derivative
    (define f (lambda (x) (* x x)))
    (define manual-derivative (lambda (x) (* 2 x)))
    
    ;; Time autodiff
    (define start-auto (current-milliseconds))
    (define result-auto
      (fold + 0
        (map (lambda (x) (d/dx f x))
             (range 0 10000))))
    (define end-auto (current-milliseconds))
    (define time-auto (- end-auto start-auto))
    
    ;; Time manual
    (define start-manual (current-milliseconds))
    (define result-manual
      (fold + 0
        (map manual-derivative (range 0 10000))))
    (define end-manual (current-milliseconds))
    (define time-manual (- end-manual start-manual))
    
    ;; Check overhead
    (define overhead (/ time-auto time-manual))
    (display (string-append "Autodiff overhead: " (number->string overhead) "x"))
    (assert-less-than overhead 3.0)))  ; Must be < 3x

;; ============================================
;; RUN ALL TESTS
;; ============================================

(define main
  (lambda ()
    (test-forward-simple)
    (test-forward-composition)
    (test-forward-multi-var)
    (test-reverse-gradient)
    (test-higher-order)
    (test-edge-cases)
    (test-performance)
    (display "All autodiff tests passed!")))

(main)
```

**Additional Test Files**:
- `tests/autodiff_forward_mode_test.esk` - Forward mode exhaustive
- `tests/autodiff_reverse_mode_test.esk` - Reverse mode exhaustive
- `tests/autodiff_composition_test.esk` - Function composition
- `tests/autodiff_performance_test.esk` - Performance benchmarks

**Deliverables**:
- Comprehensive test suite (100+ test cases)
- Performance benchmarks
- Documentation of autodiff capabilities

**Success Criteria**:
- ✅ All tests pass
- ✅ Autodiff overhead < 3x (verified)
- ✅ No LLVM IR errors
- ✅ Memory clean (Valgrind)

---

### Week 4: Examples & Documentation (Sessions 31-40)

#### **Session 31-32: Example Curation & Categorization**

**Objective**: Identify and categorize 30 production-quality examples

**Process**:

1. **Audit all examples** in `examples/` directory (~100 files)

2. **Selection criteria**:
   - ✅ Demonstrates core functionality
   - ✅ Educational value
   - ✅ Currently works (or can be easily fixed)
   - ✅ Represents best practices
   - ❌ Reject: Duplicates, experiments, broken code

3. **Categorize into 6 groups** (5 examples each):

**Category 1: Basics** (5 examples)
```
examples/01-basics/
├── hello.esk                 - Basic program structure
├── arithmetic.esk            - Numeric operations
├── lists.esk                 - List creation and access
├── conditionals.esk          - if/cond/case
└── functions.esk             - Lambda and define
```

**Category 2: List Operations** (5 examples)
```
examples/02-list-ops/
├── map_filter.esk            - Basic higher-order functions
├── fold.esk                  - Reduction operations
├── list_utilities.esk        - length, append, reverse, etc.
├── mixed_types.esk           - int64 + double in lists
└── nested_lists.esk          - List manipulation
```

**Category 3: Higher-Order Functions** (5 examples)
```
examples/03-higher-order/
├── composition.esk           - Function composition
├── closures.esk              - Lexical closures
├── currying.esk              - Partial application
├── recursion.esk             - Recursive patterns
└── mutual_recursion.esk      - Mutual recursion
```

**Category 4: Numerical Computing** (5 examples)
```
examples/04-numerical/
├── vector_ops.esk            - Vector operations
├── matrix_ops.esk            - Matrix operations
├── numeric_integration.esk   - Simpson's rule, etc.
├── optimization.esk          - Gradient descent
└── statistics.esk            - Mean, variance, etc.
```

**Category 5: Automatic Differentiation** (5 examples)
```
examples/05-autodiff/
├── forward_mode.esk          - d/dx examples
├── reverse_mode.esk          - Gradient examples
├── gradient_descent.esk      - Optimization with autodiff
├── neural_network.esk        - Simple NN training
└── optimization_demo.esk     - Finding minima/maxima
```

**Category 6: Advanced Features** (5 examples)
```
examples/06-advanced/
├── type_system.esk           - Type annotations
├── performance.esk           - Optimization techniques
├── memory_management.esk     - Arena usage patterns
├── interop.esk               - C interoperability
└── scientific_computing.esk  - Complex computation
```

4. **Document selection** in `docs/EXAMPLE_CATALOG.md`:
   - List of selected examples
   - Educational objectives for each
   - Prerequisites
   - Estimated time to understand
   - Related examples

**Deliverables**:
- `docs/EXAMPLE_CATALOG.md` - Complete catalog
- Reorganized `examples/` directory structure
- README in each category directory

**Success Criteria**:
- ✅ 30 examples selected
- ✅ Categorized into 6 groups
- ✅ Each example has clear purpose
- ✅ No duplicates or broken examples

---

#### **Session 33-34: Update Example Syntax (Batch 1)**

**Objective**: Update first 15 examples to current syntax

**Migration Pattern**:

```scheme
# ========================================
# OLD SYNTAX (Don't use)
# ========================================

(define main 
  (lambda () 
    (display "Hello")))

# ========================================
# NEW SYNTAX (Current standard)
# ========================================

(display "Hello")

# OR with explicit main function:

(define (main)
  (display "Hello"))

(main)
```

**Process for Each Example**:

1. **Read existing example**
2. **Update syntax**:
   - Remove old `main` lambda wrapper
   - Use direct expressions at top level
   - Or use `(define (main) ...)` + `(main)` pattern
3. **Add comprehensive comments**:
   ```scheme
   ;; Example: List Operations with map
   ;;
   ;; This example demonstrates:
   ;; 1. Creating lists with mixed types (int64 + double)
   ;; 2. Using map to transform elements
   ;; 3. Type preservation through operations
   ;;
   ;; Expected output:
   ;; (2 5.0 6 9.5 10)
   
   (define data (list 1 2.5 3 4.75 5))
   (display (map (lambda (x) (* x 2)) data))
   ```
4. **Test example runs correctly**
5. **Include expected output in comments**

**Examples to Update** (Batch 1):

**Basics (5)**:
- `01-basics/hello.esk`
- `01-basics/arithmetic.esk`
- `01-basics/lists.esk`
- `01-basics/conditionals.esk`
- `01-basics/functions.esk`

**List Operations (5)**:
- `02-list-ops/map_filter.esk`
- `02-list-ops/fold.esk`
- `02-list-ops/list_utilities.esk`
- `02-list-ops/mixed_types.esk`
- `02-list-ops/nested_lists.esk`

**Higher-Order (5)**:
- `03-higher-order/composition.esk`
- `03-higher-order/closures.esk`
- `03-higher-order/currying.esk`
- `03-higher-order/recursion.esk`
- `03-higher-order/mutual_recursion.esk`

**Deliverables**:
- 15 updated examples
- All examples tested and working
- `docs/SYNTAX_MIGRATION.md` - Migration guide

**Success Criteria**:
- ✅ All 15 examples use current syntax
- ✅ All examples run without errors
- ✅ Comments explain functionality clearly
- ✅ Expected output documented

---

#### **Session 35-36: Update Example Syntax (Batch 2)**

**Objective**: Update remaining 15 examples

**Examples to Update** (Batch 2):

**Numerical (5)**:
- `04-numerical/vector_ops.esk`
- `04-numerical/matrix_ops.esk`
- `04-numerical/numeric_integration.esk`
- `04-numerical/optimization.esk`
- `04-numerical/statistics.esk`

**Autodiff (5)**:
- `05-autodiff/forward_mode.esk`
- `05-autodiff/reverse_mode.esk`
- `05-autodiff/gradient_descent.esk`
- `05-autodiff/neural_network.esk`
- `05-autodiff/optimization_demo.esk`

**Advanced (5)**:
- `06-advanced/type_system.esk`
- `06-advanced/performance.esk`
- `06-advanced/memory_management.esk`
- `06-advanced/interop.esk`
- `06-advanced/scientific_computing.esk`

**Same process as Batch 1**, plus:
- **Special attention to autodiff examples** (verify with fixes from Sessions 23-28)
- **Performance examples** should include benchmarks
- **Advanced examples** should be thoroughly documented

**Deliverables**:
- 15 updated examples
- All examples tested and working
- Batch 1 + Batch 2 = 30 total examples complete

**Success Criteria**:
- ✅ All 30 examples use current syntax
- ✅ All 30 examples run without errors
- ✅ Comprehensive comments throughout
- ✅ Educational value verified

---

#### **Session 37-38: Create Showcase Examples**

**Objective**: Create 4 publication-quality showcase examples

**Example 1: Mixed Types Demo**

File: `examples/showcase/mixed_types_demo.esk`

```scheme
;; ============================================================
;; MIXED-TYPE LIST OPERATIONS - SHOWCASE
;; ============================================================
;;
;; This example demonstrates Eshkol's unique capability to
;; seamlessly mix int64 and double values in lists while
;; preserving type information through all operations.
;;
;; Key features demonstrated:
;; 1. Tagged value system (int64 + double coexistence)
;; 2. Type preservation through map/filter/fold
;; 3. Automatic type promotion in arithmetic
;; 4. Zero-copy polymorphic operations
;;
;; Expected output: See comments after each operation
;; ============================================================

;; Create a mixed-type list: 1, 2.5, 3, 4.75, 5
(define mixed-list (list 1 2.5 3 4.75 5))

(display "Original list: ")
(display mixed-list)
;; Output: (1 2.5 3 4.75 5)

;; Transform with map - doubles each element, preserves types
(display "\nDoubled: ")
(display (map (lambda (x) (* x 2)) mixed-list))
;; Output: (2 5.0 6 9.5 10)
;; Note: 1*2=2 (int64), 2.5*2=5.0 (double)

;; Filter - keeps only values > 3
(display "\nValues > 3: ")
(display (filter (lambda (x) (> x 3)) mixed-list))
;; Output: (4.75 5)

;; Fold - sum all values (automatic promotion to double)
(display "\nSum: ")
(display (fold + 0 mixed-list))
;; Output: 16.25
;; Note: Result is double due to 2.5 + 4.75

;; Demonstrate type preservation
(display "\n\nType demonstration:")
(display "\nFirst element (int64): ")
(display (car mixed-list))
(display "\nSecond element (double): ")
(display (car (cdr mixed-list)))

;; Complex operation: square even integers, double odd ones
(display "\n\nComplex transformation:")
(display (map 
  (lambda (x) 
    (if (= (remainder (truncate x) 2) 0)
        (* x x)      ; Square evens
        (* x 2)))    ; Double odds
  mixed-list))
;; Output: (2 5.0 9 9.5 10)

(display "\n\nShowcase complete!")
```

**Example 2: Higher-Order Functions Demo**

File: `examples/showcase/higher_order_demo.esk`

```scheme
;; ============================================================
;; HIGHER-ORDER FUNCTIONS - COMPLETE GUIDE
;; ============================================================
;;
;; Comprehensive demonstration of map, filter, fold, and
;; composition in Eshkol.
;; ============================================================

(define data (list 1 2 3 4 5 6 7 8 9 10))

;; ========== MAP ==========
(display "=== MAP ===\n")

;; Square each element
(display "Squares: ")
(display (map (lambda (x) (* x x)) data))
;; Output: (1 4 9 16 25 36 49 64 81 100)

;; ========== FILTER ==========
(display "\n\n=== FILTER ===\n")

;; Keep only even numbers
(display "Evens: ")
(display (filter (lambda (x) (= (remainder x 2) 0)) data))
;; Output: (2 4 6 8 10)

;; Keep only numbers > 5
(display "\nGreater than 5: ")
(display (filter (lambda (x) (> x 5)) data))
;; Output: (6 7 8 9 10)

;; ========== FOLD ==========
(display "\n\n=== FOLD ===\n")

;; Sum
(display "Sum: ")
(display (fold + 0 data))
;; Output: 55

;; Product
(display "\nProduct: ")
(display (fold * 1 data))
;; Output: 3628800

;; Maximum
(display "\nMaximum: ")
(display (fold (lambda (a b) (if (> a b) a b)) 0 data))
;; Output: 10

;; ========== COMPOSITION ==========
(display "\n\n=== COMPOSITION ===\n")

;; Combine operations: sum of squares of evens
(display "Sum of squares of evens: ")
(display 
  (fold + 0
    (map (lambda (x) (* x x))
      (filter (lambda (x) (= (remainder x 2) 0)) data))))
;; Output: 220 (2² + 4² + 6² + 8² + 10² = 4+16+36+64+100)

(display "\n\nDemo complete!")
```

**Example 3: Autodiff Tutorial**

File: `examples/showcase/autodiff_tutorial.esk`

```scheme
;; ============================================================
;; AUTOMATIC DIFFERENTIATION - TUTORIAL
;; ============================================================
;;
;; Complete guide to automatic differentiation in Eshkol,
;; from basic derivatives to optimization.
;; ============================================================

;; ========== LESSON 1: Basic Derivatives ==========
(display "=== LESSON 1: Basic Derivatives ===\n")

;; d/dx(x²) = 2x
(define f1 (lambda (x) (* x x)))
(display "d/dx(x²) at x=5: ")
(display (d/dx f1 5))
;; Output: 10

;; d/dx(sin(x)) = cos(x)
(define f2 (lambda (x) (sin x)))
(display "\nd/dx(sin(x)) at x=0: ")
(display (d/dx f2 0))
;; Output: 1.0

;; ========== LESSON 2: Composition ==========
(display "\n\n=== LESSON 2: Composition ===\n")

;; Chain rule: d/dx(sin(x²)) = 2x*cos(x²)
(define g (lambda (x) (* x x)))
(define h (lambda (x) (sin x)))
(display "d/dx(sin(x²)) at x=1: ")
(display (d/dx (compose h g) 1))
;; Output: ~1.08 (2*1*cos(1))

;; ========== LESSON 3: Gradients ==========
(display "\n\n=== LESSON 3: Gradients ===\n")

;; ∇(v·v) = 2v
(define quadratic (lambda (v) (dot v v)))
(display "∇(v·v) at v=(1,2,3): ")
(display (gradient quadratic #(1 2 3)))
;; Output: #(2 4 6)

;; ========== LESSON 4: Optimization ==========
(display "\n\n=== LESSON 4: Optimization ===\n")

;; Find minimum of (x-3)² using gradient descent
(define objective (lambda (x) (* (- x 3) (- x 3))))

(define gradient-descent
  (lambda (f x0 learning-rate iterations)
    (if (= iterations 0)
        x0
        (gradient-descent 
          f
          (- x0 (* learning-rate (d/dx f x0)))
          learning-rate
          (- iterations 1)))))

(display "Finding minimum of (x-3)²:")
(display "\nStarting at x=0")
(display "\nMinimum found at x=")
(display (gradient-descent objective 0 0.1 100))
;; Output: ~3.0

(display "\n\nTutorial complete!")
```

**Example 4: Vector Operations**

File: `examples/showcase/vector_operations.esk`

```scheme
;; ============================================================
;; VECTOR CALCULUS OPERATIONS - SHOWCASE
;; ============================================================
;;
;; Demonstration of vector calculus capabilities.
;; ============================================================

;; ========== Gradient ==========
(display "=== GRADIENT ===\n")

;; ∇(x² + y²) = (2x, 2y)
(define f (lambda (x y) (+ (* x x) (* y y))))
(display "∇(x² + y²) at (1,2): ")
(display (gradient-2d f 1 2))
;; Output: (2 4)

;; ========== Divergence ==========
(display "\n\n=== DIVERGENCE ===\n")

;; ∇·F where F = (x, y)
(define fx (lambda (x y) x))
(define fy (lambda (x y) y))
(display "∇·(x,y) at (1,1): ")
(display (divergence fx fy 1 1))
;; Output: 2

;; ========== Laplacian ==========
(display "\n\n=== LAPLACIAN ===\n")

;; ∇²(x² + y²) = 4
(display "∇²(x² + y²) at (1,2): ")
(display (laplacian f 1 2))
;; Output: 4

(display "\n\nShowcase complete!")
```

**Deliverables**:
- 4 showcase examples (mixed-types, higher-order, autodiff, vectors)
- Publication-quality code and comments
- Comprehensive expected output documentation

**Success Criteria**:
- ✅ Each example is self-contained
- ✅ Educational progression is clear
- ✅ Code demonstrates best practices
- ✅ Could be featured in academic paper or blog post

---

#### **Session 39-40: Documentation Accuracy Pass**

**Objective**: Remove aspirational claims, ensure accuracy

**Files to Update**:

**1. README.md**

Changes needed:
- ✅ **Remove**: Claims about debugger (doesn't exist)
- ✅ **Remove**: Claims about profiler (doesn't exist)
- ✅ **Remove**: "Production-ready" claims (v1.0-foundation is early adopter release)
- ✅ **Update**: Feature matrix to show actual status
- ✅ **Add**: "Early Adopter Release" disclaimer
- ✅ **Add**: Link to example catalog
- ✅ **Add**: Link to quick start guide

**2. GETTING_STARTED.md**

Changes needed:
- ✅ **Remove**: Lines 253-272 (debugger references)
- ✅ **Remove**: Profiler references
- ✅ **Update**: Installation instructions for v1.0-foundation
- ✅ **Add**: Troubleshooting section
- ✅ **Add**: "What Works Now" section
- ✅ **Add**: "Known Limitations" section

**3. Create New Files**

**`docs/QUICK_START.md`**:
```markdown
# Eshkol Quick Start (5 minutes)

## Installation
...

## Your First Program
...

## Key Concepts
...

## Next Steps
...
```

**`docs/EXAMPLES_GUIDE.md`**:
```markdown
# Eshkol Examples Guide

## Example Categories

### 1. Basics
- `01-basics/hello.esk` - Your first program
- ...

### 2. List Operations
...

## Learning Path

Recommended order:
1. Start with basics/hello.esk
2. ...

## Running Examples

```bash
./build/eshkol-run examples/01-basics/hello.esk
```
```

**`docs/TROUBLESHOOTING.md`**:
```markdown
# Troubleshooting Guide

## Installation Issues

### LLVM Not Found
...

### Build Failures
...

## Runtime Issues

### Memory Errors
...

### Type Errors
...

## Getting Help
...
```

**4. Update Feature Matrix**

**Before** (aspirational):
```markdown
| Feature | Status |
|---------|--------|
| Core Language | 100% |
| Autodiff | 100% |
| Debugger | Available |
```

**After** (reality):
```markdown
| Feature | Status | Notes |
|---------|--------|-------|
| Core Language | 90% | Basic Scheme + mixed types |
| Higher-Order Functions | 100% | map, filter, fold, etc. |
| Autodiff | 95% | Forward + reverse mode (SCH-006/007/008 fixed) |
| Type System | 70% | Gradual typing, partial inference |
| Debugger | ❌ Planned | Not in v1.0-foundation |
| Profiler | ❌ Planned | Not in v1.0-foundation |
```

**Deliverables**:
- Updated README.md (accurate claims only)
- Updated GETTING_STARTED.md (remove false features)
- New: QUICK_START.md
- New: EXAMPLES_GUIDE.md
- New: TROUBLESHOOTING.md
- Accurate feature matrix

**Success Criteria**:
- ✅ No aspirational claims remain
- ✅ All documented features actually work
- ✅ Known limitations clearly stated
- ✅ Troubleshooting guide helpful
- ✅ Quick start enables success in 5 minutes

---

## Month 3: Infrastructure (Sessions 41-60)

### Week 5: CI/CD & Packaging (Sessions 41-50)

#### **Session 41-42: GitHub Actions CI (Ubuntu)**

**Objective**: Automated builds and tests on Ubuntu

**File**: `.github/workflows/ci.yml` (new)

```yaml
name: Eshkol CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build-ubuntu:
    runs-on: ubuntu-22.04
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Install LLVM
        run: |
          sudo apt-get update
          sudo apt-get install -y llvm-14-dev clang-14 cmake ninja-build
      
      - name: Build
        run: |
          mkdir build
          cd build
          cmake -G Ninja ..
          ninja -j$(nproc)
      
      - name: Test Core
        run: |
          cd build
          echo "Testing mixed-type lists..."
          ./eshkol-run ../tests/mixed_type_lists_basic_test.esk
          
          echo "Testing higher-order functions..."
          ./eshkol-run ../tests/phase3_basic.esk
          
          echo "Testing autodiff..."
          ./eshkol-run ../tests/autodiff_comprehensive.esk
      
      - name: Run Full Test Suite
        run: |
          bash scripts/run_all_tests.sh
      
      - name: Verify 100% Pass Rate
        run: |
          # Fail if any test failed
          if grep -q "FAIL" test_results.txt; then
            echo "❌ Test failures detected"
            exit 1
          fi
          
          # Verify 66/66 tests passed
          pass_count=$(grep -c "PASS" test_results.txt)
          if [ "$pass_count" -ne 66 ]; then
            echo "❌ Expected 66 tests, got $pass_count"
            exit 1
          fi
          
          echo "✅ All 66 tests passed"
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-ubuntu
          path: test_results.txt
```

**Tasks**:
1. Create `.github/workflows/` directory
2. Write `ci.yml` with above content
3. Push to GitHub and verify CI runs
4. Fix any issues that arise
5. Add CI badge to README.md

**Deliverables**:
- `.github/workflows/ci.yml` - Working CI configuration
- Green CI badge in README
- Automated testing on every push

**Success Criteria**:
- ✅ CI runs on every push
- ✅ All 66 tests run automatically
- ✅ Build fails if any test fails
- ✅ Results visible in GitHub Actions tab

---

#### **Session 43-44: GitHub Actions CI (macOS)**

**Objective**: Cross-platform CI for macOS

**Addition to** `.github/workflows/ci.yml`:

```yaml
  build-macos:
    runs-on: macos-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Install LLVM
        run: |
          brew install llvm cmake ninja
      
      - name: Build
        env:
          LLVM_DIR: /opt/homebrew/opt/llvm
        run: |
          mkdir build && cd build
          cmake -G Ninja \
                -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm \
                ..
          ninja -j$(sysctl -n hw.ncpu)
      
      - name: Test
        run: |
          cd build
          ./eshkol-run ../tests/mixed_type_lists_basic_test.esk
          bash ../scripts/run_all_tests.sh
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-macos
          path: test_results.txt
```

**Platform-Specific Considerations**:
- macOS uses Homebrew for LLVM
- ARM64 (Apple Silicon) vs x86_64 support
- Different default paths
- Different core count detection

**Deliverables**:
- macOS CI workflow added
- Tests passing on both ARM64 and x86_64
- Cross-platform compatibility verified

**Success Criteria**:
- ✅ CI runs on macOS (latest)
- ✅ All tests pass on both platforms
- ✅ No platform-specific issues
- ✅ Build artifacts available for both

---

#### **Session 45-46: CMake Install Targets**

**Objective**: System-wide installation support

**File**: `CMakeLists.txt`

**Additions**:

```cmake
# ============================================================
# INSTALL TARGETS
# ============================================================

# Install executable
install(TARGETS eshkol-run
        RUNTIME DESTINATION bin
        COMPONENT runtime)

# Install library
install(TARGETS eshkol-static
        ARCHIVE DESTINATION lib
        COMPONENT development)

# Install headers
install(DIRECTORY inc/eshkol
        DESTINATION include
        COMPONENT development
        FILES_MATCHING PATTERN "*.h")

# Install examples
install(DIRECTORY examples/
        DESTINATION share/eshkol/examples
        COMPONENT examples
        FILES_MATCHING PATTERN "*.esk")

# Install documentation
install(DIRECTORY docs/
        DESTINATION share/doc/eshkol
        COMPONENT documentation
        FILES_MATCHING PATTERN "*.md")

# Install standard library (when created)
# install(DIRECTORY stdlib/
#         DESTINATION share/eshkol/stdlib
#         COMPONENT runtime
#         FILES_MATCHING PATTERN "*.esk")

# ============================================================
# UNINSTALL TARGET
# ============================================================

if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# ============================================================
# COMPONENT INSTALLATION
# ============================================================

# Allow selective installation
# cmake --install build --component runtime     # Just executable
# cmake --install build --component development # Headers + library
# cmake --install build --component examples    # Examples
# cmake --install build                         # Everything
```

**Create** `cmake/cmake_uninstall.cmake.in`:

```cmake
if(NOT EXISTS "@CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: @CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt")
endif()

file(READ "@CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")

foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    exec_program(
      "@CMAKE_COMMAND@" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
      )
    if(NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    endif()
  else()
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif()
endforeach()
```

**Testing**:

```bash
# Build
cmake -B build
cmake --build build

# Install to /usr/local (requires sudo)
sudo cmake --install build

# Verify installation
which eshkol-run
# Should output: /usr/local/bin/eshkol-run

ls /usr/local/share/eshkol/examples
# Should list example files

# Test installed version
eshkol-run /usr/local/share/eshkol/examples/01-basics/hello.esk

# Uninstall
sudo cmake --build build --target uninstall
```

**Deliverables**:
- Install targets in CMakeLists.txt
- Uninstall target
- Tested installation to /usr/local
- Documentation of install process

**Success Criteria**:
- ✅ `sudo make install` works
- ✅ Executable in /usr/local/bin
- ✅ Examples in /usr/local/share
- ✅ Headers in /usr/local/include
- ✅ Uninstall works correctly

---

#### **Session 47-48: CPack for Debian**

**Objective**: Create `.deb` packages for Ubuntu/Debian

**File**: `CMakeLists.txt`

**Additions**:

```cmake
# ============================================================
# CPACK CONFIGURATION
# ============================================================

include(CPack)

# Basic package information
set(CPACK_PACKAGE_NAME "eshkol")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_RELEASE "1")
set(CPACK_PACKAGE_VENDOR "Eshkol Project")
set(CPACK_PACKAGE_CONTACT "eshkol-dev@example.com")

# Description
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "High-performance Scheme with scientific computing and autodiff")
set(CPACK_PACKAGE_DESCRIPTION
    "Eshkol is a LISP-like language combining Scheme's elegance with 
     C's performance. Features include:
     - Native LLVM backend for optimal performance
     - Mixed-type lists (int64 + double seamlessly)
     - Automatic differentiation for ML/scientific computing
     - Type-safe polymorphic operations
     - Arena-based memory management
     
     v1.0-foundation includes complete autodiff, CI/CD infrastructure,
     and 30 curated examples.")

# License and readme
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Debian-specific settings
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Eshkol Team <eshkol-dev@example.com>")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "llvm-14, libc6 (>= 2.34)")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/tsotchke/eshkol")

# Component descriptions
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Eshkol Runtime")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Eshkol compiler and runtime")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Eshkol Development Files")
set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Headers and libraries for Eshkol development")
set(CPACK_COMPONENT_EXAMPLES_DISPLAY_NAME "Eshkol Examples")
set(CPACK_COMPONENT_EXAMPLES_DESCRIPTION "Example programs demonstrating Eshkol features")

# Generator
set(CPACK_GENERATOR "DEB")

# Architecture
execute_process(COMMAND dpkg --print-architecture
                OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Package filename
set(CPACK_PACKAGE_FILE_NAME 
    "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
```

**Building and Testing**:

```bash
# Build the package
cd build
cmake ..
make
cpack

# This creates: eshkol_1.0.0-1_amd64.deb

# Install the package
sudo dpkg -i eshkol_1.0.0-1_amd64.deb

# Verify installation
which eshkol-run
dpkg -L eshkol  # List installed files

# Test installed package
eshkol-run /usr/share/eshkol/examples/01-basics/hello.esk

# Uninstall
sudo dpkg -r eshkol
```

**Deliverables**:
- CPack configuration in CMakeLists.txt
- Working .deb package
- Tested installation on Ubuntu 22.04
- Documentation of packaging process

**Success Criteria**:
- ✅ `cpack` creates .deb file
- ✅ Package installs with `sudo dpkg -i`
- ✅ All dependencies correctly specified
- ✅ Package can be uninstalled cleanly
- ✅ Package metadata correct

---

#### **Session 49-50: Docker Build System**

**Objective**: Reproducible builds in containerized environment

**File**: `docker/ubuntu/release/Dockerfile`

```dockerfile
# ============================================================
# Eshkol v1.0-foundation Release Build
# ============================================================

FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    llvm-14-dev \
    clang-14 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /eshkol

# Copy source code
COPY . .

# Build
RUN mkdir build && \
    cd build && \
    cmake -G Ninja .. && \
    ninja -j$(nproc)

# Run tests
RUN cd build && \
    ./eshkol-run ../tests/mixed_type_lists_basic_test.esk && \
    ./eshkol-run ../tests/phase3_basic.esk && \
    bash ../scripts/run_all_tests.sh

# Build package
RUN cd build && cpack

# Create output directory
RUN mkdir -p /output

# Copy artifacts
RUN cp build/*.deb /output/

# Default command
CMD ["/bin/bash"]
```

**Usage Scripts**:

**`scripts/docker_build.sh`**:
```bash
#!/bin/bash
# Build Eshkol in Docker

set -e

echo "Building Eshkol in Docker..."
docker build -t eshkol:1.0-foundation -f docker/ubuntu/release/Dockerfile .

echo "Extracting artifacts..."
docker create --name eshkol-artifacts eshkol:1.0-foundation
docker cp eshkol-artifacts:/output/ ./docker-artifacts/
docker rm eshkol-artifacts

echo "Artifacts available in ./docker-artifacts/"
ls -lh ./docker-artifacts/
```

**`scripts/docker_test.sh`**:
```bash
#!/bin/bash
# Test Eshkol package in clean Ubuntu container

set -e

echo "Testing Eshkol package in clean Ubuntu..."

docker run --rm -v $(pwd)/docker-artifacts:/artifacts ubuntu:22.04 bash -c "
    apt-get update && apt-get install -y llvm-14 && \
    dpkg -i /artifacts/eshkol_*.deb && \
    eshkol-run /usr/share/eshkol/examples/01-basics/hello.esk
"

echo "Package test successful!"
```

**Deliverables**:
- `docker/ubuntu/release/Dockerfile`
- `scripts/docker_build.sh`
- `scripts/docker_test.sh`
- Working Docker-based builds
- Clean container testing

**Success Criteria**:
- ✅ Docker builds complete successfully
- ✅ All tests pass in container
- ✅ Package can be extracted
- ✅ Package installs in clean Ubuntu
- ✅ Reproducible builds

---

### Week 6: Release Preparation (Sessions 51-60)

#### **Session 51-52: Homebrew Formula**

**Objective**: macOS package distribution via Homebrew

**Repository**: Create new GitHub repo `homebrew-eshkol`

**File**: `Formula/eshkol.rb`

```ruby
class Eshkol < Formula
  desc "High-performance Scheme with scientific computing and autodiff"
  homepage "https://github.com/tsotchke/eshkol"
  url "https://github.com/tsotchke/eshkol/archive/v1.0.0.tar.gz"
  sha256 "..." # Will be computed from release tarball
  license "MIT"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "llvm"

  def install
    system "cmake", "-G", "Ninja", ".", *std_cmake_args
    system "ninja"
    system "ninja", "install"
  end

  test do
    (testpath/"test.esk").write "(display (+ 2 3))"
    assert_equal "5", shell_output("#{bin}/eshkol-run test.esk").strip
  end
end
```

**Setup Instructions**:

1. **Create Homebrew tap repository**:
   ```bash
   # On GitHub, create new repo: homebrew-eshkol
   git clone https://github.com/tsotchke/homebrew-eshkol
   cd homebrew-eshkol
   mkdir Formula
   # Create Formula/eshkol.rb with content above
   git add Formula/eshkol.rb
   git commit -m "Add Eshkol formula"
   git push
   ```

2. **Test locally**:
   ```bash
   # Add tap
   brew tap tsotchke/eshkol
   
   # Install from local formula
   brew install --build-from-source Formula/eshkol.rb
   
   # Test
   eshkol-run -h
   eshkol-run examples/01-basics/hello.esk
   ```

3. **Prepare for release**:
   - Formula will be updated with correct SHA256 after v1.0.0 release
   - Instructions for users will be:
     ```bash
     brew tap tsotchke/eshkol
     brew install eshkol
     ```

**Deliverables**:
- `homebrew-eshkol` repository created
- `Formula/eshkol.rb` written and tested
- Local installation tested
- Documentation for Homebrew users

**Success Criteria**:
- ✅ Tap repository created
- ✅ Formula syntax valid
- ✅ Local installation works
- ✅ Test passes
- ✅ Ready for public release

---

#### **Session 53-54: Integration Testing**

**Objective**: End-to-end validation of complete system

**Directory**: `tests/integration/`

**Test 1: Mixed-Type Comprehensive**

File: `tests/integration/mixed_type_comprehensive.esk`

```scheme
;; Comprehensive mixed-type list operations test
;; Tests all combinations of int64/double operations

(define (test-basic-operations)
  (display "Testing basic mixed-type operations...")
  
  ;; Creation
  (define list1 (list 1 2.5 3 4.75 5))
  (assert-equal (length list1) 5)
  
  ;; Access
  (assert-equal (car list1) 1)
  (assert-equal (cadr list1) 2.5)
  
  ;; Modification
  (set-car! list1 10)
  (assert-equal (car list1) 10)
  
  (display "✓"))

(define (test-higher-order)
  (display "Testing higher-order functions...")
  
  (define data (list 1 2.5 3 4.75 5))
  
  ;; Map
  (define doubled (map (lambda (x) (* x 2)) data))
  (assert-equal (car doubled) 2)
  (assert-equal (cadr doubled) 5.0)
  
  ;; Filter
  (define evens (filter (lambda (x) (= (remainder (truncate x) 2) 0)) data))
  (assert-equal (length evens) 1)
  
  ;; Fold
  (define sum (fold + 0 data))
  (assert-equal sum 16.25)
  
  (display "✓"))

(define (test-edge-cases)
  (display "Testing edge cases...")
  
  ;; Empty list
  (assert-equal (length '()) 0)
  
  ;; Single element
  (define single (list 42))
  (assert-equal (length single) 1)
  (assert-equal (car single) 42)
  
  ;; All integers
  (define ints (list 1 2 3 4 5))
  (assert-equal (fold * 1 ints) 120)
  
  ;; All floats
  (define floats (list 1.0 2.0 3.0))
  (assert-equal (fold + 0.0 floats) 6.0)
  
  (display "✓"))

(define (main)
  (display "\n=== Mixed-Type Comprehensive Integration Test ===\n")
  (test-basic-operations)
  (test-higher-order)
  (test-edge-cases)
  (display "\n✅ All integration tests passed!\n"))

(main)
```

**Test 2: Higher-Order Comprehensive**

File: `tests/integration/higher_order_comprehensive.esk`

```scheme
;; Comprehensive higher-order functions test
;; Tests all combinations and edge cases

;; ... (similar comprehensive structure)
```

**Test 3: Complex Computation**

File: `tests/integration/complex_computation.esk`

```scheme
;; Real-world computation example
;; Demonstrates practical use of mixed features

(define (matrix-multiply A B)
  ;; Matrix multiplication using map/fold
  ;; ...)

(define (main)
  ;; ... realistic computation
  )
```

**Test 4: Autodiff Integration**

File: `tests/integration/autodiff_integration.esk`

```scheme
;; Integration of autodiff with other features

(define (gradient-descent-demo)
  ;; Realistic optimization problem
  ;; Using autodiff with higher-order functions
  ;; ...)
```

**Deliverables**:
- 4 integration test files
- Each covering different aspect
- Real-world usage patterns
- All tests passing

**Success Criteria**:
- ✅ All integration tests pass
- ✅ Tests cover realistic usage
- ✅ Edge cases handled
- ✅ Performance acceptable

---

#### **Session 55-56: Memory & Performance Testing**

**Objective**: Ensure memory safety and performance targets

**Memory Testing**:

**Script**: `scripts/memory_test.sh`

```bash
#!/bin/bash
# Comprehensive memory testing with Valgrind

set -e

echo "=== Eshkol Memory Testing ==="

# Check if Valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "❌ Valgrind not found. Installing..."
    sudo apt-get install -y valgrind
fi

FAILED=0
TOTAL=0

# Test each file
for test in tests/*.esk tests/integration/*.esk; do
    echo "Testing: $test"
    TOTAL=$((TOTAL + 1))
    
    # Run with Valgrind
    if valgrind --leak-check=full \
                --error-exitcode=1 \
                --quiet \
                build/eshkol-run "$test" > /dev/null 2>&1; then
        echo "  ✓ Memory clean"
    else
        echo "  ❌ MEMORY LEAK DETECTED"
        FAILED=$((FAILED + 1))
        
        # Detailed report
        valgrind --leak-check=full \
                 --show-leak-kinds=all \
                 build/eshkol-run "$test" 2>&1 | tee "leak_report_$(basename $test).txt"
    fi
done

# Summary
echo ""
echo "=== Memory Test Summary ==="
echo "Total: $TOTAL"
echo "Passed: $((TOTAL - FAILED))"
echo "Failed: $FAILED"

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests memory-clean!"
    exit 0
else
    echo "❌ $FAILED tests have memory issues"
    echo "See leak_report_*.txt for details"
    exit 1
fi
```

**Performance Testing**:

**Script**: `scripts/performance_test.sh`

```bash
#!/bin/bash
# Performance benchmarking

set -e

echo "=== Eshkol Performance Benchmarks ==="

# Autodiff overhead test
echo "Testing autodiff overhead..."
./build/eshkol-run tests/autodiff_performance_test.esk

# Expected output:
# Autodiff overhead: 2.1x ✓ (< 3x target)

# Compilation speed test
echo "Testing compilation speed..."
start=$(date +%s%N)
./build/eshkol-run tests/integration/complex_computation.esk > /dev/null
end=$(date +%s%N)
duration=$(( (end - start) / 1000000 ))  # Convert to ms

echo "Compilation time: ${duration}ms"

if [ $duration -lt 10000 ]; then
    echo "✓ Compilation speed acceptable"
else
    echo "❌ Compilation too slow (> 10s)"
fi

# Memory efficiency test
echo "Testing memory efficiency..."
/usr/bin/time -v ./build/eshkol-run tests/integration/complex_computation.esk 2>&1 | grep "Maximum resident"

echo ""
echo "=== Performance Test Complete ==="
```

**Create**: `docs/TESTING.md`

```markdown
# Eshkol Testing Guide

## Test Categories

### Unit Tests
- Located in `tests/`
- 66 tests covering core functionality
- Run with: `bash scripts/run_all_tests.sh`

### Integration Tests
- Located in `tests/integration/`
- Real-world usage patterns
- Run with: `bash scripts/run_integration_tests.sh`

### Memory Tests
- Valgrind-based leak detection
- Run with: `bash scripts/memory_test.sh`
- All tests must pass before release

### Performance Tests
- Autodiff overhead benchmarks
- Compilation speed tests
- Run with: `bash scripts/performance_test.sh`

## Running Tests

```bash
# All tests
make test

# Specific category
bash scripts/run_all_tests.sh          # Unit tests
bash scripts/memory_test.sh            # Memory tests
bash scripts/performance_test.sh       # Performance tests
```

## Continuous Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- GitHub Actions CI

See `.github/workflows/ci.yml` for details.
```

**Deliverables**:
- `scripts/memory_test.sh` - Valgrind testing
- `scripts/performance_test.sh` - Performance benchmarks
- `docs/TESTING.md` - Testing documentation
- All tests passing
- Memory clean (zero leaks)
- Performance targets met

**Success Criteria**:
- ✅ Valgrind reports zero leaks
- ✅ Autodiff overhead < 3x
- ✅ Compilation < 10s for 10K LOC
- ✅ Memory usage reasonable
- ✅ All tests pass in CI

---

#### **Session 57-58: Documentation Final Pass**

**Objective**: Publication-ready documentation

**Files to Review and Update**:

**1. GETTING_STARTED.md**
- ✅ Remove debugger references (lines 253-272)
- ✅ Remove profiler references
- ✅ Update installation instructions
- ✅ Add troubleshooting section
- ✅ Test all examples in document

**2. COMPILATION_GUIDE.md**
- ✅ Update LLVM version requirements (14+)
- ✅ Add platform-specific notes
- ✅ Include Docker build instructions
- ✅ Add CI/CD information

**3. Create SECURITY.md**

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please report security vulnerabilities to: eshkol-security@example.com

**Do not** open public GitHub issues for security vulnerabilities.

### What to include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response timeline:
- Acknowledgment: Within 48 hours
- Initial assessment: Within 7 days
- Fix timeline: Depends on severity

## Known Limitations

See [KNOWN_ISSUES.md](docs/scheme_compatibility/KNOWN_ISSUES.md) for current limitations.
```

**4. Create CODE_OF_CONDUCT.md**

```markdown
# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone...

[Standard Contributor Covenant text]
```

**5. Review and Update**:
- `README.md` - Verify all claims accurate
- `ROADMAP.md` - Update with v1.0-foundation status
- `CONTRIBUTING.md` - Add v1.0 contribution guidelines
- All `docs/*.md` files - Check for outdated information

**Documentation Checklist**:

```
Documentation Accuracy Checklist:

Core Documentation:
[ ] README.md - No aspirational claims
[ ] GETTING_STARTED.md - All examples work
[ ] COMPILATION_GUIDE.md - Instructions current
[ ] SECURITY.md - Created
[ ] CODE_OF_CONDUCT.md - Created
[ ] CONTRIBUTING.md - Updated for v1.0

Technical Documentation:
[ ] docs/V1_0_FOUNDATION_EXECUTION_PLAN.md - This file
[ ] docs/AUTODIFF_TYPE_ANALYSIS.md - Created in Session 21
[ ] docs/EXAMPLE_CATALOG.md - Created in Session 31
[ ] docs/TESTING.md - Created in Session 55
[ ] docs/QUICK_START.md - Created in Session 39
[ ] docs/EXAMPLES_GUIDE.md - Created in Session 39
[ ] docs/TROUBLESHOOTING.md - Created in Session 39

Examples Documentation:
[ ] Each example has comments explaining functionality
[ ] Expected output documented
[ ] Prerequisites stated
[ ] README in each example category

Links and References:
[ ] All internal links work
[ ] External links current
[ ] No broken references
[ ] API documentation accurate
```

**Deliverables**:
- All documentation updated and accurate
- New files created (SECURITY, CODE_OF_CONDUCT)
- Documentation checklist completed
- No aspirational claims remain

**Success Criteria**:
- ✅ All documentation accurate
- ✅ No broken links
- ✅ Examples all work
- ✅ Installation instructions tested
- ✅ Troubleshooting helpful

---

#### **Session 59-60: v1.0-foundation Release**

**Objective**: PUBLIC RELEASE of v1.0-foundation

**Pre-Release Checklist**:

```
v1.0-foundation Release Checklist:

Code Quality:
[ ] All 66 unit tests passing
[ ] All 4 integration tests passing
[ ] Memory tests clean (Valgrind)
[ ] Performance targets met
[ ] CI green on Ubuntu + macOS
[ ] No compiler warnings
[ ] LLVM IR validation passes

Documentation:
[ ] README.md accurate
[ ] CHANGELOG.md complete
[ ] RELEASE_NOTES_v1.0.md written
[ ] All examples working
[ ] Quick start guide tested
[ ] Troubleshooting guide complete

Infrastructure:
[ ] CI/CD working
[ ] .deb package builds
[ ] Homebrew formula ready
[ ] Docker builds working
[ ] Installation tested on clean systems

Distribution:
[ ] Git tag created: v1.0-foundation
[ ] GitHub release created
[ ] Binaries uploaded
[ ] Checksums computed
[ ] Release notes published

Communication:
[ ] Blog post drafted
[ ] Social media prepared
[ ] Mailing list notified
[ ] Documentation site updated
```

**Release Process**:

**1. Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0-foundation] - 2026-01-XX

### Added

#### Core Features
- Complete mixed-type list operations (int64 + double seamlessly)
- 17 polymorphic higher-order functions (map, filter, fold, etc.)
- Type-safe tagged value system
- Arena-based memory management

#### Automatic Differentiation
- Forward-mode autodiff (d/dx)
- Reverse-mode autodiff (gradient)
- Composition support
- Vector return types
- Fixed SCH-006 (type inference)
- Fixed SCH-007 (vector returns)
- Fixed SCH-008 (type conflicts)

#### Infrastructure
- GitHub Actions CI (Ubuntu + macOS)
- Debian package (.deb)
- Homebrew formula
- Docker build system
- CMake install targets

#### Examples & Documentation
- 30 curated production-quality examples
- 4 showcase examples (mixed-types, higher-order, autodiff, vectors)
- Comprehensive documentation
- Quick start guide
- Troubleshooting guide

#### Testing
- 66 unit tests (100% pass rate)
- 4 integration tests
- Comprehensive autodiff test suite
- Memory testing (Valgrind clean)
- Performance benchmarks

### Fixed
- PHI node ordering violations
- Instruction dominance violations
- Arena memory scope issues
- Type preservation in all operations
- Autodiff type inference (SCH-006)
- Vector return types (SCH-007)
- Type conflicts in generated IR (SCH-008)

### Changed
- All higher-order functions now use type-safe interfaces
- Arena memory management simplified
- Test infrastructure fully automated
- Documentation now reality-based (no aspirational claims)

### Performance
- Autodiff overhead: < 3x vs hand-written derivatives ✓
- Compilation time: < 10s for typical programs ✓
- Memory efficiency: Arena allocation maintains O(1) ✓

## [0.1.0-architecture] - 2025-11-17

### Added
- Initial v1.0-architecture milestone
- Type-safe polymorphic foundation
- 100% test pass rate achieved

---

For detailed changes, see [RELEASE_NOTES_v1.0.md](RELEASE_NOTES_v1.0.md)
```

**2. Create RELEASE_NOTES_v1.0.md**

```markdown
# Eshkol v1.0-foundation Release Notes

**Release Date**: 2026-01-XX  
**Codename**: Foundation

## Overview

v1.0-foundation is the first production-ready release of Eshkol, providing:
- Stable mixed-type list operations
- Complete automatic differentiation
- Production infrastructure (CI/CD, packaging)
- 30 curated examples
- Comprehensive documentation

## Highlights

### 🎉 Complete Autodiff System
All three critical autodiff bugs resolved:
- **SCH-006**: Type inference now complete
- **SCH-007**: Vector return types working
- **SCH-008**: Type conflicts eliminated

Autodiff now reliably supports:
- Forward-mode derivatives
- Reverse-mode gradients
- Function composition
- Vector operations
- < 3x overhead (verified)

### 🚀 Production Infrastructure
- **CI/CD**: Automated testing on every commit
- **Packages**: .deb for Ubuntu/Debian, Homebrew for macOS
- **Docker**: Reproducible builds
- **Testing**: 100% pass rate maintained automatically

### 📚 Example Library
30 production-quality examples covering:
- Basic syntax and operations
- List manipulation (mixed types)
- Higher-order functions
- Numerical computing
- Automatic differentiation
- Advanced features

### 💪 Rock-Solid Foundation
- **Zero unsafe operations**: All memory access type-safe
- **100% test pass rate**: 66/66 tests passing
- **Memory clean**: Valgrind verified, zero leaks
- **Cross-platform**: Ubuntu + macOS supported

## Breaking Changes

None - v1.0-foundation is the first stable release.

## Known Limitations

See [KNOWN_ISSUES.md](docs/scheme_compatibility/KNOWN_ISSUES.md) for current limitations.

Notable limitations in v1.0-foundation:
- No eval/apply (planned for v1.1)
- No macro system (planned for v1.1)
- No module system (planned for v1.2)
- No REPL (planned for v1.2)
- Tail call optimization not implemented (future)
- Continuations not supported (future)

## Installation

### Ubuntu/Debian

```bash
# Download .deb package
wget https://github.com/tsotchke/eshkol/releases/download/v1.0.0/eshkol_1.0.0-1_amd64.deb

# Install
sudo dpkg -i eshkol_1.0.0-1_amd64.deb

# Test
eshkol-run /usr/share/eshkol/examples/01-basics/hello.esk
```

### macOS (Homebrew)

```bash
brew tap tsotchke/eshkol
brew install eshkol
```

### From Source

```bash
git clone https://github.com/tsotchke/eshkol
cd eshkol
git checkout v1.0-foundation
mkdir build && cd build
cmake ..
make -j$(nproc)
./eshkol-run ../examples/01-basics/hello.esk
```

## Migration Guide

If upgrading from v0.1.0-architecture, note:
- Example syntax updated (no `main` wrapper needed)
- Some autodiff functions may have improved type inference
- Test suite expanded

See [SYNTAX_MIGRATION.md](docs/SYNTAX_MIGRATION.md) for details.

## What's Next

### v1.1-metaprogramming (Q1 2026)
- Eval/apply system
- Macro system (syntax-rules)
- Quasiquotation

### v1.2-infrastructure (Q2 2026)
- File I/O
- Module system
- REPL

See [ROADMAP.md](ROADMAP.md) for complete roadmap.

## Contributors

This release was made possible by:
- Core development team
- Community contributors
- Early adopters and testers

Thank you! 🙏

## Support

- **Documentation**: https://github.com/tsotchke/eshkol/tree/main/docs
- **Issues**: https://github.com/tsotchke/eshkol/issues
- **Discussions**: https://github.com/tsotchke/eshkol/discussions

---

Happy coding with Eshkol! 🚀
```

**3. Git Tag and Release**

```bash
# Ensure everything committed
git status

# Create annotated tag
git tag -a v1.0-foundation -m "v1.0-foundation: Production-ready release

Complete autodiff, CI/CD infrastructure, 30 examples, 100% test pass rate.

Fixes:
- SCH-006: Autodiff type inference
- SCH-007: Vector return types
- SCH-008: Type conflicts

See RELEASE_NOTES_v1.0.md for details."

# Push tag
git push origin v1.0-foundation

# Verify tag
git show v1.0-foundation
```

**4. Build Release Artifacts**

```bash
# Debian package
cd build
cpack
# Creates: eshkol_1.0.0-1_amd64.deb

# Source tarball
git archive --format=tar.gz --prefix=eshkol-1.0.0/ v1.0-foundation > eshkol-1.0.0.tar.gz

# Compute checksums
sha256sum eshkol_1.0.0-1_amd64.deb > eshkol_1.0.0-1_amd64.deb.sha256
sha256sum eshkol-1.0.0.tar.gz > eshkol-1.0.0.tar.gz.sha256
```

**5. Create GitHub Release**

1. Go to: https://github.com/tsotchke/eshkol/releases
2. Click "Draft a new release"
3. Choose tag: `v1.0-foundation`
4. Release title: "v1.0-foundation: Production-Ready Release"
5. Description: Copy from RELEASE_NOTES_v1.0.md
6. Upload artifacts:
   - `eshkol_1.0.0-1_amd64.deb`
   - `eshkol_1.0.0-1_amd64.deb.sha256`
   - `eshkol-1.0.0.tar.gz`
   - `eshkol-1.0.0.tar.gz.sha256`
7. Click "Publish release"

**6. Update Homebrew Formula**

```bash
cd homebrew-eshkol
# Update Formula/eshkol.rb with correct SHA256
# Commit and push
git add Formula/eshkol.rb
git commit -m "Release v1.0.0"
git push
```

**7. Communication**

**Blog Post** (Draft):
```markdown
# Announcing Eshkol v1.0-foundation

We're thrilled to announce v1.0-foundation, the first production-ready 
release of Eshkol!

[Details about release, features, getting started]

[Link to GitHub release]
```

**Social Media**:
```
🚀 Eshkol v1.0-foundation is here!

✨ Complete autodiff
🔒 100% test pass rate
📦 Easy installation (.deb + Homebrew)
📚 30 production-quality examples

Try it now: https://github.com/tsotchke/eshkol/releases/v1.0-foundation

#eshkol #scheme #autodiff #scientific computing
```

**Deliverables**:
- CHANGELOG.md
- RELEASE_NOTES_v1.0.md
- Git tag: v1.0-foundation
- GitHub release created
- Binaries uploaded
- Homebrew formula updated
- Blog post published
- Announcements made

**Success Criteria**:
- ✅ All release artifacts available
- ✅ Installation instructions work
- ✅ GitHub release published
- ✅ Homebrew formula updated
- ✅ Documentation live
- ✅ Community notified

---

## Post-Release Actions

After v1.0-foundation is released:

**Immediate (Day 1)**:
- Monitor GitHub issues for bug reports
- Monitor social media for feedback
- Respond to community questions
- Fix any critical issues immediately

**Week 1**:
- Collect user feedback
- Document common issues
- Update troubleshooting guide
- Plan v1.0.1 patch if needed

**Week 2**:
- Review adoption metrics
- Analyze performance feedback
- Begin planning v1.1-metaprogramming

---

## Success Metrics (Post-Release)

**Week 1 Targets**:
- 50+ downloads
- 100+ GitHub stars
- 10+ community issues filed
- 0 critical bugs

**Month 1 Targets**:
- 100+ downloads
- 150+ GitHub stars
- 2+ external contributors
- 5+ success stories

**Quarter 1 Targets**:
- 500+ downloads
- 300+ GitHub stars
- 5+ external contributors
- 10+ production deployments
- 1+ academic paper citing Eshkol

---

## Alignment with 24-Month Master Plan

v1.0-foundation completes:
- ✅ **Phase 1, Months 1-3**: Foundation & Core Language

Next up:
- **Months 4-5**: Eval/apply + Macros → v1.1-metaprogramming
- **Month 6**: I/O + Modules → v1.2-infrastructure
- **Months 7-12**: Native Scientific Computing (Phase 2)
- **Months 13-18**: Symbolic & Neural DSL (Phase 3)
- **Months 19-24**: Formal Verification (Phase 4)

---

## Appendix: Quick Reference

### Session Overview

| Sessions | Week | Focus | Deliverable |
|----------|------|-------|-------------|
| 21-30 | Week 3 | Autodiff fixes | SCH-006/007/008 resolved |
| 31-40 | Week 4 | Examples & docs | 30 examples + accurate docs |
| 41-50 | Week 5 | CI/CD | Automated builds + packaging |
| 51-60 | Week 6 | Release prep | v1.0-foundation PUBLIC |

### Critical Path Items

1. **Autodiff fixes** (Sessions 21-30) - BLOCKING
2. **CI/CD setup** (Sessions 41-50) - BLOCKING
3. **Release artifacts** (Session 59-60) - BLOCKING

Everything else can be parallelized.

### Quality Gates

Before each milestone:
- ✅ All tests passing
- ✅ Valgrind clean
- ✅ CI green
- ✅ Documentation accurate

### Key Commands

```bash
# Build
cmake -B build && cmake --build build

# Test
bash scripts/run_all_tests.sh

# Memory test
bash scripts/memory_test.sh

# Performance test
bash scripts/performance_test.sh

# Package
cd build && cpack

# Release
git tag -a v1.0-foundation -m "..."
git push origin v1.0-foundation
```

---

**Document Status**: EXECUTION PLAN - READY  
**Created**: November 17, 2025  
**Updated**: November 17, 2025  
**Version**: 1.0  
**Next Review**: After Session 30 (Autodiff fixes complete)

---

**END OF EXECUTION PLAN**