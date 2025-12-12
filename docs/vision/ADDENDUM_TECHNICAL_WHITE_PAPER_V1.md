# Eshkol v1.0-architecture: Technical Implementation Deep Dive

**A Production Compiler with Integrated Automatic Differentiation**

This document provides a detailed technical analysis of the Eshkol v1.0-architecture implementation - the actual production compiler, not aspirational features. For the broader vision, see [TECHNICAL_WHITE_PAPER.md](TECHNICAL_WHITE_PAPER.md).

## Abstract

Eshkol v1.0-architecture is a production-ready Scheme dialect compiler built on LLVM infrastructure, featuring compiler-integrated automatic differentiation, deterministic arena memory management, and a HoTT-inspired gradual type system. This paper examines the actual implementation architecture, focusing on the modular LLVM backend design, the tagged value runtime system, the automatic differentiation implementation with nested gradient support, and the ownership-aware lexical region memory model. We present the technical innovations that enable Eshkol to combine Scheme's homoiconicity with native code performance while maintaining deterministic memory behavior suitable for real-time applications.

## 1. Compiler Architecture

### 1.1 Compilation Pipeline

```
Source (.esk)
    ↓
Parser → AST (with source locations)
    ↓
Macro Expansion → Transformed AST
    ↓
Type Checker (HoTT) → Annotated AST
    ↓
Ownership Analysis → Allocation Strategy Annotations
    ↓
LLVM Backend → LLVM IR
    ↓
LLVM Optimizer → Optimized IR
    ↓
Code Generation → Object File (.o)
    ↓
Linker → Executable or Library
```

### 1.2 Parser Implementation

**File:** `lib/frontend/parser.cpp`

**Tokenizer:**
```cpp
enum TokenType {
    TOKEN_LPAREN, TOKEN_RPAREN,
    TOKEN_QUOTE,               // '
    TOKEN_BACKQUOTE,           // `
    TOKEN_COMMA,               // ,
    TOKEN_COMMA_AT,            // ,@
    TOKEN_SYMBOL,
    TOKEN_STRING,
    TOKEN_NUMBER,              // Supports scientific notation
    TOKEN_BOOLEAN,             // #t #f
    TOKEN_CHAR,                // #\a, #\space, #\newline
    TOKEN_VECTOR_START,        // #(
    TOKEN_COLON,               // : for type annotations
    TOKEN_ARROW,               // -> for function types
    TOKEN_EOF
};
```

**AST Structure:**
```c
struct eshkol_ast {
    eshkol_type_t type;        // AST node type
    uint32_t line;             // Source location
    uint32_t column;
    uint32_t inferred_hott_type;  // Type checker result
    
    union {
        int64_t int64_val;
        double double_val;
        struct { char* ptr; uint64_t size; } str_val;
        struct { char* id; void* data; } variable;
        eshkol_operations_t operation;
        // ... other node types
    };
}
```

**Parsing Features:**
- Recursive descent with operator precedence
- HoTT type expression parsing (arrow, forall, list, vector, tensor, pair, product, sum)
- Pattern matching support (literal, variable, wildcard, cons, list, predicate, or)
- Internal defines transformed to letrec automatically
- Closure capture analysis via static AST scanning

**Type Annotation Parsing:**
```scheme
; Inline parameter types
(define (f (x : integer) (y : real)) body)

; Return type annotation  
(define (g x y) : real body)

; Type expressions
(-> integer real)              ; Function type
(list real)                    ; List of reals
(forall (a) (-> a a))         ; Polymorphic identity
```

### 1.3 Type Checker Implementation

**File:** `lib/types/type_checker.cpp`

**Bidirectional Type Checking:**

**Synthesis Mode (⇒):** Infer type from expression
```cpp
TypeCheckResult synthesize(eshkol_ast_t* expr) {
    switch (expr->type) {
        case ESHKOL_INT64:   return Int64;
        case ESHKOL_DOUBLE:  return Float64;
        case ESHKOL_VAR:     return lookup_in_context(expr->variable.id);
        case ESHKOL_OP:      return synthesizeOperation(expr);
        // ...
    }
}
```

**Checking Mode (⇐):** Verify expression has expected type
```cpp
TypeCheckResult check(eshkol_ast_t* expr, TypeId expected) {
    auto result = synthesize(expr);
    if (isSubtype(result.type, expected)) {
        return TypeCheckResult::ok(expected);
    }
    return TypeCheckResult::error("Type mismatch");
}
```

**Type Hierarchy:**
```
Value (root)
├── Number
│   ├── Integer → Int64, Natural
│   └── Real → Float64
├── Text
│   ├── String
│   └── Char
├── Boolean
├── Null
└── Symbol
```

**Subtyping with Promotion:**
```cpp
TypeId promoteForArithmetic(TypeId left, TypeId right) {
    if (left == Int64 && right == Int64) return Int64;
    if (left == Float64 || right == Float64) return Float64;
    // ... additional rules
}
```

### 1.4 Module System Implementation

**Module Resolution:**
1. Current directory (relative to source file)
2. `lib/` directory
3. `$ESHKOL_PATH` (colon-separated paths)
4. System library paths

**Dependency Graph:**
```cpp
struct ModuleNode {
    std::string name;
    std::vector<std::string> dependencies;
    eshkol_ast_t* ast;
    enum { UNVISITED, VISITING, VISITED } color;
};
```

**Cycle Detection:**
- Depth-first search with coloring
- VISITING color detects cycles
- Reports complete dependency chain on error

**Symbol Visibility:**
```scheme
(provide add-squared multiply-squared)  ; Public
(define helper ...)                      ; Private → __module__helper
```

**Topological Sort:**
- Modules compiled in dependency order
- Dependencies before dependents
- Enables forward references

## 2. LLVM Backend Architecture

### 2.1 Modular Codegen Design

**File:** `lib/backend/llvm_codegen.cpp`

The backend is organized into 15 specialized modules rather than monolithic generation:

```cpp
class EshkolLLVMCodeGen {
    // Core infrastructure
    CodegenContext context;
    TypeSystem typeSystem;
    
    // Specialized modules
    TaggedValueCodegen taggedValues;
    ArithmeticCodegen arithmetic;
    AutodiffCodegen autodiff;
    BindingCodegen bindings;
    CollectionCodegen collections;
    ControlFlowCodegen controlFlow;
    FunctionCodegen functions;
    TensorCodegen tensors;
    HomoiconicCodegen homoiconic;
    CallApplyCodegen callApply;
    MapCodegen map;
    TailCallCodegen tailCall;
    SystemCodegen system;
    HashCodegen hash;
    StringIOCodegen stringIO;
}
```

**Benefits:**
- Clear separation of concerns
- Independent module development
- Easier testing and debugging
- Modular optimization passes

### 2.2 Tagged Value System

**File:** `lib/backend/tagged_value_codegen.cpp`

**Runtime Representation (16 bytes):**
```c
struct eshkol_tagged_value {
    uint8_t type;        // Type tag (0-255)
    uint8_t flags;       // Exactness, special flags
    uint16_t reserved;   // Future use
    uint32_t padding;    // Alignment
    union {
        int64_t int_val;     // INT64, BOOL, CHAR, SYMBOL
        double double_val;   // DOUBLE
        uint64_t ptr_val;    // Heap pointers
        uint64_t raw_val;    // Raw manipulation
    } data;
}
```

**Type Encoding:**

**Immediate Types (0-7):** Store data directly
```
0: NULL          - Empty list
1: INT64         - 64-bit integer
2: DOUBLE        - IEEE 754 double
3: BOOL          - Boolean (#t/#f)
4: CHAR          - Unicode codepoint
5: SYMBOL        - Interned symbol
6: DUAL_NUMBER   - Forward-mode AD
```

**Consolidated Types (8-9):** Subtype in object header
```
8: HEAP_PTR      - Cons, string, vector, tensor, hash, exception
9: CALLABLE      - Closure, lambda-sexpr, AD node, primitive, continuation
```

**Legacy Types (32+):** Backward compatibility
```
32: CONS_PTR     - Use HEAP_PTR + HEAP_SUBTYPE_CONS
33: STRING_PTR   - Use HEAP_PTR + HEAP_SUBTYPE_STRING
34-40: Other legacy pointer types
```

**Pack/Unpack Operations:**
```cpp
// From tagged_value_codegen.cpp
llvm::Value* packInt64(llvm::Value* int64_val)
llvm::Value* packDouble(llvm::Value* double_val)
llvm::Value* packPtr(llvm::Value* ptr_val, uint8_t type)
llvm::Value* packBool(llvm::Value* bool_val)

llvm::Value* unpackInt64(llvm::Value* tagged_val)
llvm::Value* unpackDouble(llvm::Value* tagged_val)
llvm::Value* unpackPtr(llvm::Value* tagged_val)
```

### 2.3 Object Header System

**Structure (8 bytes):**
```c
struct eshkol_object_header {
    uint8_t subtype;      // Distinguishes cons, string, vector, etc.
    uint8_t flags;        // GC marks, linear status, etc.
    uint16_t ref_count;   // Reference count (0 = not ref-counted)
    uint32_t size;        // Object size in bytes
}
```

**Header Placement:**
- Prepended to heap objects (at offset -8)
- Data pointer points AFTER header
- Macro: `ESHKOL_GET_HEADER(data_ptr)` subtracts 8 bytes

**Heap Subtypes:**
```c
// HEAP_PTR subtypes
HEAP_SUBTYPE_CONS        = 0
HEAP_SUBTYPE_STRING      = 1
HEAP_SUBTYPE_VECTOR      = 2
HEAP_SUBTYPE_TENSOR      = 3
HEAP_SUBTYPE_MULTI_VALUE = 4  // Multiple return values
HEAP_SUBTYPE_HASH        = 5
HEAP_SUBTYPE_EXCEPTION   = 6
HEAP_SUBTYPE_RECORD      = 7
HEAP_SUBTYPE_BYTEVECTOR  = 8
HEAP_SUBTYPE_PORT        = 9
```

**Callable Subtypes:**
```c
// CALLABLE subtypes
CALLABLE_SUBTYPE_CLOSURE      = 0
CALLABLE_SUBTYPE_LAMBDA_SEXPR = 1
CALLABLE_SUBTYPE_AD_NODE      = 2
CALLABLE_SUBTYPE_PRIMITIVE    = 3
CALLABLE_SUBTYPE_CONTINUATION = 4
```

### 2.4 TypedValue in Code Generation

**Compiler Internal Structure:**
```cpp
struct TypedValue {
    llvm::Value* llvm_value;        // LLVM IR value
    eshkol_value_type_t type;       // Runtime type
    bool is_exact;                  // Numeric exactness
    uint32_t flags;                 // Additional metadata
    hott_type_expr_t* hott_type;    // HoTT type (optional)
    TypeId param_type;              // Parameterized type (List<T>)
}
```

**Purpose:**
- Carries both LLVM value and type information
- Enables type-directed code generation
- Facilitates polymorphic dispatch
- Tracks exactness for Scheme numeric tower

**Usage in Codegen:**
```cpp
TypedValue codegenExpression(eshkol_ast_t* ast) {
    TypedValue result;
    result.llvm_value = /* generate IR */;
    result.type = /* infer or annotate */;
    result.is_exact = /* determine exactness */;
    return result;
}
```

## 3. Automatic Differentiation Implementation

### 3.1 Dual Number Codegen

**File:** `lib/backend/autodiff_codegen.cpp`

**Dual Number Arithmetic:**
```cpp
// (a, a') + (b, b') = (a+b, a'+b')
TypedValue dualAdd(TypedValue left, TypedValue right) {
    llvm::Value* dual_a = unpackDual(left.llvm_value);
    llvm::Value* dual_b = unpackDual(right.llvm_value);
    
    llvm::Value* val_a = extractDualValue(dual_a);
    llvm::Value* der_a = extractDualDerivative(dual_a);
    llvm::Value* val_b = extractDualValue(dual_b);
    llvm::Value* der_b = extractDualDerivative(dual_b);
    
    llvm::Value* sum_val = builder.CreateFAdd(val_a, val_b);
    llvm::Value* sum_der = builder.CreateFAdd(der_a, der_b);
    
    return packDual(sum_val, sum_der);
}
```

**Math Function Derivatives:**
```cpp
// sin: (sin(x), cos(x)×x')
TypedValue dualSin(TypedValue dual_arg) {
    llvm::Value* x = extractDualValue(dual_arg);
    llvm::Value* x_prime = extractDualDerivative(dual_arg);
    
    llvm::Value* sin_x = callMathFunction("sin", x);
    llvm::Value* cos_x = callMathFunction("cos", x);
    llvm::Value* derivative = builder.CreateFMul(cos_x, x_prime);
    
    return packDual(sin_x, derivative);
}

// Similar for: dualCos, dualExp, dualLog, dualSqrt, dualTan
```

### 3.2 Computational Graph Implementation

**AD Node Structure:**
```c
struct ad_node {
    ad_node_type_t type;   // Operation type
    double value;          // Forward pass result
    double gradient;       // Backward pass gradient
    ad_node_t* input1;     // First parent node
    ad_node_t* input2;     // Second parent node (binary ops)
    size_t id;             // Topological sort ID
}
```

**Node Types:**
```c
AD_NODE_CONSTANT   // Leaf: constant value
AD_NODE_VARIABLE   // Leaf: input variable
AD_NODE_ADD        // a + b
AD_NODE_SUB        // a - b
AD_NODE_MUL        // a × b
AD_NODE_DIV        // a ÷ b
AD_NODE_SIN        // sin(a)
AD_NODE_COS        // cos(a)
AD_NODE_EXP        // exp(a)
AD_NODE_LOG        // log(a)
AD_NODE_POW        // a^b
AD_NODE_NEG        // -a
```

**Tape Structure:**
```c
struct ad_tape {
    ad_node_t** nodes;         // Array of nodes
    size_t num_nodes;          // Current count
    size_t capacity;           // Allocated capacity
    ad_node_t** variables;     // Input variable nodes
    size_t num_variables;      // Number of inputs
}
```

**Graph Recording:**
```cpp
TypedValue createADNodeBinary(ad_node_type_t op,
                              TypedValue left,
                              TypedValue right) {
    // Allocate AD node
    ad_node_t* node = arena_allocate_ad_node(__global_arena);
    node->type = op;
    node->input1 = extractADNode(left);
    node->input2 = extractADNode(right);
    
    // Compute forward value
    node->value = compute_operation(op, 
                                    node->input1->value,
                                    node->input2->value);
    node->gradient = 0.0;
    node->id = __current_ad_tape->num_nodes;
    
    // Add to tape
    arena_tape_add_node(__current_ad_tape, node);
    
    return packADNode(node);
}
```

**Backpropagation Algorithm:**
```cpp
void backpropagate(ad_tape_t* tape) {
    // Seed output gradient
    tape->nodes[tape->num_nodes - 1]->gradient = 1.0;
    
    // Traverse in reverse topological order
    for (int64_t i = tape->num_nodes - 1; i >= 0; i--) {
        ad_node_t* node = tape->nodes[i];
        
        switch (node->type) {
            case AD_NODE_ADD:
                node->input1->gradient += node->gradient;
                node->input2->gradient += node->gradient;
                break;
            
            case AD_NODE_MUL:
                node->input1->gradient += node->gradient * node->input2->value;
                node->input2->gradient += node->gradient * node->input1->value;
                break;
            
            case AD_NODE_SIN:
                double cos_val = cos(node->input1->value);
                node->input1->gradient += node->gradient * cos_val;
                break;
            
            // ... all other operations
        }
    }
}
```

### 3.3 Nested Gradient Implementation

**Tape Stack (Global State):**
```c
#define MAX_TAPE_DEPTH 32
ad_tape_t* __ad_tape_stack[MAX_TAPE_DEPTH];
uint64_t __ad_tape_depth = 0;
```

**Push/Pop Operations:**
```cpp
void pushTapeContext() {
    if (__ad_tape_depth >= MAX_TAPE_DEPTH) {
        eshkol_error("Tape stack overflow");
        return;
    }
    
    // Create new tape for inner gradient
    ad_tape_t* new_tape = arena_allocate_tape(__global_arena, 64);
    __ad_tape_stack[__ad_tape_depth++] = __current_ad_tape;
    __current_ad_tape = new_tape;
}

void popTapeContext() {
    if (__ad_tape_depth == 0) {
        eshkol_error("Tape stack underflow");
        return;
    }
    
    // Restore outer tape
    __current_ad_tape = __ad_tape_stack[--__ad_tape_depth];
}
```

**Nested Gradient Execution:**
```scheme
; Outer gradient pushes tape 0
(gradient 
  (lambda (x)
    ; Inner gradient pushes tape 1
    (gradient 
      (lambda (y) (* x y y))
      (vector 1.0)))
  (vector 2.0))

; Tape 1 computes ∂(xy²)/∂y
; Tape 0 computes ∂(result)/∂x
```

## 4. Memory Management Implementation

### 4.1 Arena Allocation

**File:** `lib/core/arena_memory.cpp`

**Arena Structure:**
```c
struct arena {
    arena_block_t* current_block;   // Current block
    arena_scope_t* current_scope;   // Lexical scope
    size_t default_block_size;      // Default 64KB
    size_t total_allocated;         // Statistics
    size_t alignment;               // 8 bytes
}
```

**Block Structure:**
```c
struct arena_block {
    uint8_t* memory;    // Allocated memory
    size_t size;        // Block size
    size_t used;        // Bytes used
    arena_block_t* next;  // Next block in chain
}
```

**Allocation Algorithm:**
```c
void* arena_allocate(arena_t* arena, size_t size) {
    size_t aligned_size = (size + 7) & ~7;  // 8-byte align
    
    arena_block_t* block = arena->current_block;
    size_t current_used = (block->used + 7) & ~7;
    
    if (current_used + aligned_size > block->size) {
        // Need new block
        size_t new_size = max(aligned_size, arena->default_block_size);
        arena_block_t* new_block = create_arena_block(new_size);
        new_block->next = arena->current_block;
        arena->current_block = new_block;
        block = new_block;
        current_used = 0;
    }
    
    void* ptr = block->memory + current_used;
    block->used = current_used + aligned_size;
    return ptr;
}
```

**Scope Management:**
```c
void arena_push_scope(arena_t* arena) {
    arena_scope_t* scope = malloc(sizeof(arena_scope_t));
    scope->block = arena->current_block;
    scope->used = arena->current_block->used;
    scope->parent = arena->current_scope;
    arena->current_scope = scope;
}

void arena_pop_scope(arena_t* arena) {
    arena_scope_t* scope = arena->current_scope;
    
    // Free blocks allocated after scope start
    // ... block deallocation logic
    
    // Restore arena state
    arena->current_block = scope->block;
    arena->current_block->used = scope->used;
    arena->current_scope = scope->parent;
    free(scope);
}
```

### 4.2 Ownership Analysis

**File:** `exe/eshkol-run.cpp`

**Ownership States:**
```cpp
enum OwnershipState {
    UNOWNED,     // Not yet owned
    OWNED,       // Exclusively owned
    MOVED,       // Ownership transferred
    BORROWED     // Temporarily accessed
};
```

**Escape Analysis:**
```cpp
enum EscapeResult {
    NO_ESCAPE,         // Stack allocation
    RETURN_ESCAPE,     // Region allocation
    CLOSURE_ESCAPE,    // Shared allocation (captured)
    GLOBAL_ESCAPE      // Shared allocation (global)
};
```

**Analysis Algorithm:**
1. Scan AST for variable definitions and uses
2. Track data flow through expressions
3. Detect captures in closures
4. Determine if value escapes function/region
5. Annotate AST with allocation strategy

**Decision Tree:**
```
If value only used within function:
    → NO_ESCAPE → Stack allocation
Else if value returned but not captured:
    → RETURN_ESCAPE → Region allocation
Else if value captured in closure OR stored globally:
    → CLOSURE_ESCAPE/GLOBAL_ESCAPE → Shared allocation
```

### 4.3 Region-Based Memory

**Region Structure:**
```c
struct eshkol_region {
    arena_t* arena;            // Region's arena
    const char* name;          // Optional name
    eshkol_region_t* parent;   // Enclosing region
    size_t size_hint;          // Size hint
    uint64_t escape_count;     // Escape tracking
    uint8_t is_active;         // Currently active?
}
```

**Region Stack (Global):**
```c
#define MAX_REGION_DEPTH 16
eshkol_region_t* __region_stack[MAX_REGION_DEPTH];
uint64_t __region_stack_depth = 0;
```

**with-region Implementation:**
```cpp
// Generated code for (with-region 'name body)
region = region_create("name", size_hint);
region_push(region);

// Execute body expressions
result = eval_body();

region_pop();  // Destroys region and frees arena
return result;
```

## 5. Closure System Implementation

### 5.1 Closure Structure

**Runtime Layout (24 bytes + environment):**
```c
struct eshkol_closure {
    uint64_t func_ptr;              // Function pointer
    eshkol_closure_env_t* env;      // Captured environment
    uint64_t sexpr_ptr;             // S-expression for display
    uint8_t return_type;            // Return type category
    uint8_t input_arity;            // Expected arguments
    uint8_t flags;                  // Variadic, etc.
    uint8_t reserved;
    uint32_t hott_type_id;          // HoTT type ID
}
```

**Environment Structure:**
```c
struct eshkol_closure_env {
    size_t num_captures;  // Packed field (see below)
    eshkol_tagged_value_t captures[];  // Flexible array
}
```

**Packed Info Encoding:**
```
num_captures field (64 bits):
├─ Bits 0-15:  actual capture count
├─ Bits 16-31: fixed parameter count
└─ Bit 63:     is_variadic flag
```

**Macros for Unpacking:**
```c
#define CLOSURE_ENV_GET_NUM_CAPTURES(packed) ((packed) & 0xFFFF)
#define CLOSURE_ENV_GET_FIXED_PARAMS(packed) (((packed) >> 16) & 0xFFFF)
#define CLOSURE_ENV_IS_VARIADIC(packed) (((packed) >> 63) & 1)
```

### 5.2 Closure Compilation

**Static Capture Analysis (Parser):**
```cpp
std::vector<std::string> analyzeLambdaCaptures(
    const eshkol_ast_t* lambda_body,
    const std::vector<eshkol_ast_t>& params,
    const std::set<std::string>& parent_defined_vars
) {
    // Collect parameter names (shadow parent scope)
    std::set<std::string> param_names = extract_params(params);
    
    // Collect locally defined variables
    std::set<std::string> local_defined = find_defines(lambda_body);
    
    // Collect all variable references
    std::set<std::string> all_refs = find_var_refs(lambda_body);
    
    // Captures = referenced && not param && not local && in parent
    std::vector<std::string> captures;
    for (const auto& var : all_refs) {
        if (!param_names.count(var) &&
            !local_defined.count(var) &&
            parent_defined_vars.count(var)) {
            captures.push_back(var);
        }
    }
    return captures;
}
```

**Closure Allocation (Codegen):**
```cpp
TypedValue createClosure(eshkol_ast_t* lambda_ast,
                         std::vector<std::string> captures) {
    // Generate lambda function with extra capture parameters
    llvm::Function* lambda_func = generateLambdaFunction(
        lambda_ast->operation.lambda_op.parameters,
        lambda_ast->operation.lambda_op.body,
        captures  // Added as trailing parameters
    );
    
    // Pack closure info
    size_t num_captures = captures.size();
    size_t fixed_params = lambda_ast->operation.lambda_op.num_params;
    bool is_variadic = lambda_ast->operation.lambda_op.is_variadic;
    
    size_t packed_info = num_captures |
                         (fixed_params << 16) |
                         (is_variadic ? (1ULL << 63) : 0);
    
    // Allocate closure
    llvm::Value* closure = call_arena_allocate_closure(
        func_ptr,
        packed_info,
        sexpr_ptr,
        return_type_info
    );
    
    // Store captured values in environment
    for (size_t i = 0; i < num_captures; i++) {
        llvm::Value* capture_val = lookup_value(captures[i]);
        store_capture(closure, i, capture_val);
    }
    
    return packCallable(closure, CALLABLE_SUBTYPE_CLOSURE);
}
```

**Calling Convention:**
```c
// Lambda signature:
eshkol_tagged_value lambda_func(
    param1, param2, ..., paramN,      // Regular parameters
    capture1, capture2, ..., captureM  // Captured variables
)

// Closure call unpacks environment:
eshkol_tagged_value closure_call(closure_ptr, arg1, arg2, ...) {
    eshkol_closure_t* closure = (eshkol_closure_t*)closure_ptr;
    
    // Extract captures from environment
    capture1 = closure->env->captures[0];
    capture2 = closure->env->captures[1];
    // ...
    
    // Call function with args + captures
    return ((closure_func_t)closure->func_ptr)(
        arg1, arg2, ..., argN,
        capture1, capture2, ..., captureM
    );
}
```

### 5.3 Homoiconic Display

**Lambda Registry (Global):**
```c
struct eshkol_lambda_registry {
    eshkol_lambda_entry_t* entries;  // Array of entries
    size_t count;                     // Current count
    size_t capacity;                  // Allocated capacity
}

struct eshkol_lambda_entry {
    uint64_t func_ptr;    // Function pointer
    uint64_t sexpr_ptr;   // S-expression cons chain
    const char* name;     // Optional name
}
```

**Registration (During Compilation):**
```cpp
// When compiling lambda, store S-expression
llvm::Value* sexpr = codegenQuotedAST(lambda_ast);
call_lambda_registry_add(func_ptr, sexpr, name);

// Also store in closure->sexpr_ptr for direct access
```

**Display (Runtime):**
```c
void eshkol_display_closure(uint64_t closure_ptr) {
    eshkol_closure_t* closure = (eshkol_closure_t*)closure_ptr;
    
    if (closure->sexpr_ptr != 0) {
        // Display embedded S-expression
        eshkol_display_list(closure->sexpr_ptr, opts);
    } else {
        // Fallback to registry lookup
        uint64_t sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
        if (sexpr != 0) {
            eshkol_display_list(sexpr, opts);
        } else {
            printf("#<closure>");
        }
    }
}
```

## 6. Cons Cell Implementation

### 6.1 Tagged Cons Structure

**Memory Layout (32 bytes):**
```c
struct arena_tagged_cons_cell {
    eshkol_tagged_value_t car;  // 16 bytes
    eshkol_tagged_value_t cdr;  // 16 bytes
}
```

**With Object Header (40 bytes total):**
```
Offset -8:  eshkol_object_header_t (8 bytes)
Offset 0:   car (16 bytes)
Offset 16:  cdr (16 bytes)
```

**Allocation:**
```cpp
arena_tagged_cons_cell_t* arena_allocate_cons_with_header(arena_t* arena) {
    // Allocate header + cons cell
    size_t total = sizeof(eshkol_object_header_t) + 
                   sizeof(arena_tagged_cons_cell_t);
    uint8_t* mem = arena_allocate_aligned(arena, total, 16);
    
    // Initialize header
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_CONS;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = sizeof(arena_tagged_cons_cell_t);
    
    // Initialize cons cell
    arena_tagged_cons_cell_t* cell = 
        (arena_tagged_cons_cell_t*)(mem + sizeof(eshkol_object_header_t));
    cell->car.type = ESHKOL_VALUE_NULL;
    cell->cdr.type = ESHKOL_VALUE_NULL;
    // ... zero remaining fields
    
    return cell;
}
```

### 6.2 Mixed-Type List Support

**Type Information in Each Element:**
```scheme
; Each element fully typed
(define mixed (list 42              ; INT64
                    3.14            ; DOUBLE
                    "hello"         ; STRING (via HEAP_PTR)
                    #t              ; BOOL
                    '(1 2)          ; CONS (via HEAP_PTR)
                    (lambda (x) x)  ; CLOSURE (via CALLABLE)
                    ))

; No boxing - direct storage in 16-byte slots
```

**Access Pattern:**
```c
// Get car type
uint8_t car_type = cell->car.type;

// Dispatch based on type
if (car_type == ESHKOL_VALUE_INT64) {
    int64_t val = cell->car.data.int_val;
} else if (car_type == ESHKOL_VALUE_DOUBLE) {
    double val = cell->car.data.double_val;
} else if (car_type == ESHKOL_VALUE_HEAP_PTR) {
    uint8_t subtype = ESHKOL_GET_SUBTYPE((void*)cell->car.data.ptr_val);
    // Handle based on subtype (CONS, STRING, VECTOR, etc.)
}
```

## 7. Hash Table Implementation

### 7.1 Hash Table Structure

**File:** `lib/core/arena_memory.cpp`

```c
struct eshkol_hash_table {
    size_t capacity;                // Bucket count
    size_t size;                    // Entry count
    size_t tombstones;              // Deleted entries
    eshkol_tagged_value_t* keys;    // Key array
    eshkol_tagged_value_t* values;  // Value array
    uint8_t* status;                // Entry status
}
```

**Entry Status:**
```c
#define HASH_ENTRY_EMPTY    0
#define HASH_ENTRY_OCCUPIED 1
#define HASH_ENTRY_DELETED  2  // Tombstone for open addressing
```

### 7.2 Hashing Algorithm

**FNV-1a Hash:**
```c
uint64_t hash_tagged_value(const eshkol_tagged_value_t* value) {
    uint64_t hash = FNV_OFFSET_BASIS;  // 14695981039346656037
    
    // Mix in type
    hash ^= value->type;
    hash *= FNV_PRIME;  // 1099511628211
    
    switch (value->type) {
        case ESHKOL_VALUE_INT64:
            hash ^= fnv1a_hash_u64(value->data.int_val);
            break;
        
        case ESHKOL_VALUE_STRING_PTR:
            hash ^= fnv1a_hash_string((char*)value->data.ptr_val);
            break;
        
        case ESHKOL_VALUE_HEAP_PTR:
            // Check subtype for string vs pointer hash
            if (subtype == HEAP_SUBTYPE_STRING) {
                hash ^= fnv1a_hash_string((char*)value->data.ptr_val);
            } else {
                hash ^= fnv1a_hash_u64(value->data.ptr_val);
            }
            break;
        
        // ... other types
    }
    return hash;
}
```

**Collision Resolution:**
- Open addressing with linear probing
- Load factor threshold: 0.75
- Automatic rehashing when exceeded
- Initial capacity: 16 buckets

## 8. Exception Handling

### 8.1 Exception Structure

```c
struct eshkol_exception {
    eshkol_exception_type_t type;   // Exception category
    char* message;                   // Error message
    eshkol_tagged_value_t* irritants;  // Irritant values
    uint32_t num_irritants;
    uint32_t line;                   // Source location
    uint32_t column;
    char* filename;
}
```

**Exception Types:**
```c
ESHKOL_EXCEPTION_ERROR
ESHKOL_EXCEPTION_TYPE_ERROR
ESHKOL_EXCEPTION_FILE_ERROR
ESHKOL_EXCEPTION_READ_ERROR
ESHKOL_EXCEPTION_SYNTAX_ERROR
ESHKOL_EXCEPTION_RANGE_ERROR
ESHKOL_EXCEPTION_ARITY_ERROR
ESHKOL_EXCEPTION_DIVIDE_BY_ZERO
ESHKOL_EXCEPTION_USER_DEFINED
```

### 8.2 guard/raise Implementation

**Mechanism:**
- C `setjmp`/`longjmp` for non-local exits
- Global exception handler stack
- Global `g_current_exception` pointer

**guard Compilation:**
```cpp
// guard generates:
jmp_buf handler_buf;
if (setjmp(handler_buf) == 0) {
    eshkol_push_exception_handler(&handler_buf);
    result = eval_body();
    eshkol_pop_exception_handler();
} else {
    // Exception caught
    exc = eshkol_get_current_exception();
    result = eval_handler_clauses(exc);
}
```

**raise Execution:**
```c
void eshkol_raise(eshkol_exception_t* exc) {
    g_current_exception = exc;
    
    if (g_exception_handler_stack && 
        g_exception_handler_stack->jmp_buf_ptr) {
        longjmp(*jmp_buf_ptr, 1);  // Jump to handler
    } else {
        // Unhandled - print and abort
        fprintf(stderr, "Unhandled exception: %s\n", exc->message);
        abort();
    }
}
```

## 9. REPL/JIT Implementation

### 9.1 LLVM ORC JIT

**File:** `lib/repl/repl_jit.cpp`

**JIT Engine:**
```cpp
std::unique_ptr<llvm::orc::LLJIT> jit;

// Initialize JIT
jit = llvm::orc::LLJITBuilder().create();

// Add module
jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(module), 
    std::move(context)
));

// Lookup symbol
auto symbol = jit->lookup("__eshkol_expression_0");
auto func_ptr = (expr_func_t)symbol->getAddress();

// Execute
eshkol_tagged_value_t result = func_ptr();
```

**Symbol Resolution:**
- Runtime functions exported from `lib/core/arena_memory.cpp`
- Previous REPL definitions registered as external symbols
- Cross-evaluation function calls supported

### 9.2 Shared Arena for Persistence

**Global REPL Arena:**
```c
arena_t* __repl_shared_arena;  // Shared across evaluations
```

**Persistence Mechanism:**
```scheme
> (define data (iota 1000))  ; Allocated in shared arena
> (define squared (map (lambda (x) (* x x)) data))
> ; Both persist across evaluations
> (length squared)
1000
```

**Lifetime:**
- Shared arena created at REPL startup
- Survives across all evaluations
- Only freed on REPL exit
- Enables true interactive development

## 10. Performance Characteristics

### 10.1 Memory Performance

**Allocation Speed:**
- Arena allocation: O(1) bump-pointer
- No free-list traversal
- No per-object metadata (except headers)
- Cache-friendly sequential layout

**Deallocation Speed:**
- Scope exit: O(1) bulk free
- No individual object tracking
- No reference counting overhead (except shared)

### 10.2 Function Call Performance

**Direct Calls:**
- Native LLVM function calls
- No indirection
- Inlining opportunities

**Closure Calls:**
- One indirect call through func_ptr
- Environment lookup: indexed array access
- Minimal overhead vs direct calls

### 10.3 Type Dispatch Performance

**Polymorphic Arithmetic:**
- Runtime type check (1 compare)
- Branch to specialized implementation
- Each path fully optimized

**Monomorphic Fast Paths:**
- Compiler recognizes constant types
- Generates direct calls (no dispatch)
- LLVM optimizes away branches

## 11. Module Compilation Modes

### 11.1 Standalone Executable

**Generated Code:**
```cpp
int main(int argc, char** argv) {
    // Store command-line args
    __eshkol_argc = argc;
    __eshkol_argv = argv;
    
    // Initialize global arena
    __global_arena = arena_create(65536);
    
    // Initialize modules (in dependency order)
    __eshkol_mod_utils_init();
    __eshkol_mod_main_init();
    
    // Execute main expression
    eshkol_tagged_value_t result = __eshkol_main_expr();
    
    // Cleanup
    arena_destroy(__global_arena);
    
    return 0;
}
```

### 11.2 Shared Library Mode

**Generated Code:**
```cpp
extern "C" void __eshkol_lib_init__() {
    // Initialize global arena
    if (!__global_arena) {
        __global_arena = arena_create(65536);
    }
    
    // Initialize exported functions
    // ... function definitions with external linkage
}
```

**Usage:**
```c
// From C/C++ code
__eshkol_lib_init__();
eshkol_tagged_value_t result = exported_function(arg1, arg2);
```

### 11.3 Object File Compilation

**Command:**
```bash
eshkol-run --object-only input.esk -o output.o
```

**Linking:**
```bash
eshkol-run main.esk stdlib.o utils.o -o executable
```

## 12. Quantum RNG Implementation

### 12.1 What It Actually Is

**8-qubit quantum circuit simulation** for **classical randomness** - NOT quantum computing.

**File:** `lib/quantum/quantum_rng.c`

**Context:**
```c
struct qrng_ctx {
    uint64_t phase[8];              // 8 qubit phases
    double quantum_state[8];        // Quantum state simulation
    uint64_t entangle[8];           // Entanglement correlations
    uint64_t last_measurement[8];   // Previous measurements
    double entropy_pool[16];        // Entropy accumulation
    uint64_t pool_mixer;            // Mixing state
    uint64_t system_entropy;        // From system sources
    uint64_t runtime_entropy;       // Continuous injection
    struct timeval init_time;       // Initialization time
    pid_t pid;                      // Process ID
    uint64_t unique_id;             // Unique context ID
    uint64_t counter;               // Operation counter
    // ... buffer state
}
```

**Quantum Operations (Simulated):**
```c
hadamard_gate(x)        // Superposition simulation
phase_gate(x, angle)    // Phase rotation
measure_state(ctx, state, last)  // Collapse to classical
quantum_noise(x)        // Quantum uncertainty simulation
```

**Entropy Sources:**
- System time (microsecond precision)
- Process ID
- CPU cycle counter (if available)
- Stack address entropy
- Runtime entropy (continuously updated)

**Use Case:**
- High-quality random numbers for Monte Carlo methods
- Stochastic optimization
- Random sampling in scientific computing
- **Not for:** Quantum algorithms, qubits, quantum gates

## 13. v1.0 Implementation Summary

### Actual Capabilities

**Compiler:**
- ✅ LLVM-based modular backend
- ✅ Bidirectional type checking (HoTT-inspired)
- ✅ Ownership and escape analysis
- ✅ Module system with dependency resolution
- ✅ Macro expansion (syntax-rules)
- ✅ Source location tracking

**Runtime:**
- ✅ 16-byte tagged values
- ✅ 8-byte object headers
- ✅ Arena memory with scopes
- ✅ Closure system with captures
- ✅ Exception handling (guard/raise)
- ✅ Hash tables (FNV-1a)
- ✅ Lambda registry for homoiconicity

**Automatic Differentiation:**
- ✅ Forward-mode (dual numbers)
- ✅ Reverse-mode (computational graphs)
- ✅ Nested gradients (32-level stack)
- ✅ Vector calculus operators (8 total)
- ✅ Polymorphic arithmetic

**Data Structures:**
- ✅ Mixed-type cons cells
- ✅ Heterogeneous vectors
- ✅ N-dimensional tensors
- ✅ Hash tables
- ✅ Strings with UTF-8

**Standard Library:**
- ✅ List operations (60+ functions)
- ✅ String utilities (30+ functions)
- ✅ Functional programming (compose, curry, flip)
- ✅ JSON parsing/serialization
- ✅ CSV parsing/generation
- ✅ Base64 encoding/decoding
- ✅ Math library (linear algebra, integration, root finding, statistics)

**Development Tools:**
- ✅ Interactive REPL with JIT
- ✅ Standalone compiler
- ✅ Library compilation mode
- ✅ Comprehensive test suite

### Not in v1.0

- ❌ GPU acceleration
- ❌ Multi-threading
- ❌ Distributed computing
- ❌ Quantum computing
- ❌ JIT in standalone mode
- ❌ Symbolic mathematics beyond basic differentiation
- ❌ Built-in visualization
- ❌ Profiling tools
- ❌ Debugger integration

## Conclusion

Eshkol v1.0-architecture represents a substantial implementation achievement: a production compiler combining Scheme semantics, LLVM performance, compiler-integrated automatic differentiation, and deterministic memory management. The modular backend architecture, sophisticated AD system with nested gradient support, and arena-based memory model provide a solid foundation for scientific computing and machine learning applications.

The technical innovations - particularly the integrated AD system operating at compiler, runtime, and IR levels simultaneously, and the ownership-aware lexical region memory model - demonstrate novel approaches to long-standing challenges in language implementation.

While v1.0 focuses on core capabilities (gradient-based optimization, neural network fundamentals, numerical computing), it establishes the architectural patterns and implementation quality necessary for future expansions including GPU acceleration, distributed training, and quantum computing integration.

---

*This document describes the actual v1.0-architecture implementation as it exists in the production codebase. For aspirational features and long-term vision, see [TECHNICAL_WHITE_PAPER.md](TECHNICAL_WHITE_PAPER.md) and [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md).*