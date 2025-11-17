# Mixed Type Lists: Type Promotion and LLVM Code Generation Refinement

## Enhanced Type Promotion Strategy

### R7RS Scheme Numeric Tower Compatibility

Our implementation must follow the Scheme numeric tower strictly for maximum compatibility:

```
Complex Numbers
    ↑
Real Numbers (double in our implementation)
    ↑  
Rational Numbers (represented as double for simplicity)
    ↑
Integer Numbers (int64 in our implementation)
```

### Detailed Promotion Rules

#### Arithmetic Operations (`+`, `-`, `*`)
| Operation | Left Type | Right Type | Result Type | Conversion Strategy |
|-----------|-----------|------------|-------------|-------------------|
| `+` | int64 | int64 | int64 | Direct integer arithmetic |
| `+` | int64 | double | double | Convert int64 to double, then add |
| `+` | double | int64 | double | Convert int64 to double, then add |
| `+` | double | double | double | Direct double arithmetic |

#### Division Operation (`/`)
**Special Scheme Rule**: Division always produces an exact result if possible, otherwise inexact (double)
| Operation | Left Type | Right Type | Result Type | Notes |
|-----------|-----------|------------|-------------|-------|
| `/` | int64 | int64 | double | Always promotes to double (Scheme exact→inexact) |
| `/` | int64 | double | double | Standard promotion |  
| `/` | double | int64 | double | Standard promotion |
| `/` | double | double | double | Direct double arithmetic |

#### Comparison Operations (`=`, `<`, `>`, `<=`, `>=`, `<>`)
- **Principle**: Compare values numerically, not representations
- **Strategy**: Always promote to common type for comparison, but preserve original types
- **Special case**: `=` must handle exact/inexact distinction per Scheme semantics

#### Modulo/Remainder Operations (`remainder`, `modulo`)
- **Constraint**: Only defined for integer operands in our implementation
- **Error handling**: Throw runtime error if either operand is double

### LLVM Code Generation Patterns

#### Optimized Type Dispatch Strategy

Instead of naive switch statements, use jump tables and prediction:

```llvm
; Efficient type dispatch using computed goto pattern
define i64 @mixed_arithmetic_add(%tagged_cons_cell* %left_cell, %tagged_cons_cell* %right_cell) {
entry:
    %left_type = load i8, i8* %left_cell.car_type
    %right_type = load i8, i8* %right_cell.car_type
    
    ; Create combined type index: (left_type << 2) | right_type
    %left_shifted = shl i8 %left_type, 2
    %type_combination = or i8 %left_shifted, %right_type
    
    ; Jump table for 4x4 type combinations
    switch i8 %type_combination, label %error [
        i8 5,  label %int64_int64     ; (1<<2)|1 = 5: int64 + int64
        i8 6,  label %int64_double    ; (1<<2)|2 = 6: int64 + double  
        i8 9,  label %double_int64    ; (2<<2)|1 = 9: double + int64
        i8 10, label %double_double   ; (2<<2)|2 = 10: double + double
    ]

int64_int64:
    ; Fast path: pure integer arithmetic with overflow checking
    %left_val = load i64, i64* %left_cell.car_data.int_val
    %right_val = load i64, i64* %right_cell.car_data.int_val
    %result = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %left_val, i64 %right_val)
    %sum = extractvalue {i64, i1} %result, 0
    %overflow = extractvalue {i64, i1} %result, 1
    br i1 %overflow, label %promote_to_double, label %return_int64

promote_to_double:
    ; Overflow detected, promote to double
    %left_double = sitofp i64 %left_val to double
    %right_double = sitofp i64 %right_val to double
    %double_sum = fadd double %left_double, %right_double
    ; Create tagged result with double type
    ret i64 %packed_double_result

return_int64:
    ; Pack int64 result with type tag
    ret i64 %packed_int64_result
```

#### Memory Layout Optimization

```llvm
; Define struct type for better LLVM optimization
%tagged_cons_cell = type {
    i8,     ; car_type
    i8,     ; cdr_type  
    i16,    ; padding (reserved)
    %data_union,  ; car_data
    %data_union   ; cdr_data
}

%data_union = type {
    i64     ; Largest member size, used for all access
}

; Efficient data access with bitcasting
define double @extract_double_value(%data_union* %data_ptr) {
    %raw_ptr = bitcast %data_union* %data_ptr to double*
    %value = load double, double* %raw_ptr
    ret double %value
}

define i64 @extract_int64_value(%data_union* %data_ptr) {
    %raw_ptr = bitcast %data_union* %data_ptr to i64*
    %value = load i64, i64* %raw_ptr
    ret i64 %value
}
```

#### Branch Prediction Optimization

```llvm
; Use LLVM branch weight metadata for common cases
define i64 @optimized_car_access(%tagged_cons_cell* %cell) {
    %car_type = load i8, i8* %cell.car_type
    %is_int64 = icmp eq i8 %car_type, 1
    
    ; Predict int64 is most common (weight 90/10)
    br i1 %is_int64, label %handle_int64, label %handle_other, !prof !{!"branch_weights", i32 90, i32 10}

handle_int64:
    ; Fast path for most common case
    %int_val = call i64 @extract_int64_value(%cell.car_data)
    ret i64 %packed_int64_result

handle_other:
    ; Handle double and other types
    %is_double = icmp eq i8 %car_type, 2
    br i1 %is_double, label %handle_double, label %handle_error
    
handle_double:
    %double_val = call double @extract_double_value(%cell.car_data)
    ret i64 %packed_double_result
}
```

### Advanced Type Coercion Strategy

#### Lazy Type Conversion
Instead of eager conversion, defer type conversion until necessary:

```c
// Pseudo-C for LLVM generation concept
typedef struct mixed_value {
    uint8_t type;
    union {
        int64_t int_val;
        double double_val;
    } data;
    bool converted_to_double;  // Memoization flag
    double cached_double;      // Cached conversion
} mixed_value_t;
```

#### Smart Arithmetic Dispatch
Use LLVM's function attributes for optimization:

```llvm
; Mark arithmetic functions as pure for better optimization
define fastcc i64 @int64_add(i64 %a, i64 %b) #0 {
    %result = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %a, i64 %b)
    ; ... handle overflow
}

define fastcc double @double_add(double %a, double %b) #0 {
    %result = fadd fast double %a, %b
    ret double %result
}

attributes #0 = { nounwind readnone speculatable }
```

### Integration with Existing Eshkol Patterns

#### Maintain Arena Allocation Efficiency
```llvm
; Batch allocate tagged cons cells for list operations
define %tagged_cons_cell* @arena_allocate_tagged_cons_batch(%arena_t* %arena, i32 %count) {
    %total_size = mul i32 %count, 24  ; sizeof(tagged_cons_cell)
    %memory = call i8* @arena_allocate(%arena, i32 %total_size)
    %typed_memory = bitcast i8* %memory to %tagged_cons_cell*
    ret %tagged_cons_cell* %typed_memory
}
```

#### Preserve Existing Function Signatures
```llvm
; Wrapper functions to maintain compatibility
define i64 @eshkol_cons(i64 %car, i64 %cdr) {
    ; Detect types of car and cdr values
    %car_type = call i8 @detect_value_type(i64 %car)
    %cdr_type = call i8 @detect_value_type(i64 %cdr)
    
    ; Create tagged cons cell
    %result = call i64 @create_tagged_cons(i8 %car_type, i64 %car, i8 %cdr_type, i64 %cdr)
    ret i64 %result
}
```

### Error Handling and Type Safety

#### Runtime Type Checking
```llvm
define void @validate_arithmetic_operands(i8 %left_type, i8 %right_type, i8* %op_name) {
    ; Check for invalid type combinations
    %left_valid = and i8 %left_type, 3  ; Only types 1,2 valid for arithmetic
    %right_valid = and i8 %right_type, 3
    %both_valid = and i8 %left_valid, %right_valid
    %is_valid = icmp ne i8 %both_valid, 0
    
    br i1 %is_valid, label %continue, label %error
    
error:
    call void @eshkol_runtime_error(i8* @"Invalid operand types for arithmetic operation")
    unreachable
    
continue:
    ret void
}
```

#### Scheme Exactness Tracking
```c
// Additional type bits for Scheme exactness
#define ESHKOL_TYPE_EXACT_FLAG   0x10
#define ESHKOL_TYPE_INEXACT_FLAG 0x20

// Combined type constants
#define ESHKOL_TYPE_EXACT_INT64     (ESHKOL_TYPE_INT64 | ESHKOL_TYPE_EXACT_FLAG)
#define ESHKOL_TYPE_INEXACT_DOUBLE  (ESHKOL_TYPE_DOUBLE | ESHKOL_TYPE_INEXACT_FLAG)
```

### Performance Optimization Strategies

#### SIMD Vectorization for List Operations
```llvm
; Vectorized type checking for large lists
define void @batch_type_check(%tagged_cons_cell* %cells, i32 %count, i8* %results) {
    ; Use LLVM's vectorization for type extraction
    ; Process 4 cons cells at once using SIMD instructions
    %vec_size = 4
    ; ... SIMD implementation for type batch processing
}
```

#### Cache-Friendly Memory Layout
- **Principle**: Keep hot data together
- **Strategy**: Reorder struct fields by access frequency
- **Implementation**: Place type fields at beginning for faster access

```c
// Optimized layout based on access patterns
typedef struct arena_tagged_cons_cell_optimized {
    uint8_t car_type;     // Most frequently accessed
    uint8_t cdr_type;     // Second most frequent
    uint16_t flags;       // Reserved for exactness/immutability flags
    // Data fields follow, maintaining 8-byte alignment
    union { int64_t int_val; double double_val; uint64_t ptr_val; } car_data;
    union { int64_t int_val; double double_val; uint64_t ptr_val; } cdr_data;
} arena_tagged_cons_cell_optimized_t;
```

### Testing and Validation Strategy

#### Comprehensive Type Promotion Test Matrix
```scheme
;; Test cases covering all type combinations
(define mixed-type-test-cases
  '((+ 1 2)           ; int64 + int64 → int64
    (+ 1 2.0)         ; int64 + double → double
    (+ 1.0 2)         ; double + int64 → double
    (+ 1.0 2.0)       ; double + double → double
    (/ 1 2)           ; int64 / int64 → double (exact→inexact)
    (= 1 1.0)         ; int64 = double → #t (numeric equality)
    (eqv? 1 1.0)))    ; int64 eqv? double → #f (type-strict)
```

#### Performance Benchmarking Framework
- **Baseline**: Current int64-only implementation
- **Target**: Mixed type operations within 20% performance overhead
- **Methodology**: Micro-benchmarks for each operation type
- **Metrics**: Operations per second, memory usage, cache misses

This refined approach ensures maximum compatibility with Scheme semantics while maintaining the performance characteristics of our existing arena-based system.