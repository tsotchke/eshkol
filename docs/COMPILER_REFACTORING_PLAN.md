# Eshkol Compiler Refactoring Plan

**Version**: 1.0
**Branch**: `feat/extensions`
**Date**: December 2024
**Status**: Planning Phase

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Target Architecture](#3-target-architecture)
4. [Module Specifications](#4-module-specifications)
5. [Dependency Graph](#5-dependency-graph)
6. [Implementation Phases](#6-implementation-phases)
7. [Testing Strategy](#7-testing-strategy)
8. [Migration Guidelines](#8-migration-guidelines)
9. [Future Extensibility](#9-future-extensibility)
10. [Risk Assessment](#10-risk-assessment)
11. [Appendix](#11-appendix)

---

## 1. Executive Summary

### 1.1 Goal

Transform the monolithic `llvm_codegen.cpp` (27,758 lines) into a modular, maintainable, and extensible compiler backend while:

- **Preserving 100% backward compatibility**
- **Maintaining all existing functionality**
- **Enabling future language extensions**
- **Improving code organization and testability**

### 1.2 Principles

1. **Incremental Extraction**: One module at a time, tests pass after each step
2. **Dependency Inversion**: Modules depend on abstractions, not concrete implementations
3. **Single Responsibility**: Each module has one clear purpose
4. **Composition Over Inheritance**: Modules composed via constructor injection
5. **Zero Breaking Changes**: Existing code paths work identically

### 1.3 Success Criteria

- [ ] All 180+ tests pass after each extraction phase
- [ ] No functionality regression
- [ ] Clear module boundaries with documented interfaces
- [ ] Each module independently testable
- [ ] Main codegen file reduced to <3000 lines (orchestration only)

---

## 2. Current State Analysis

### 2.1 Completed Refactoring

| Module | File | Lines | Status |
|--------|------|-------|--------|
| TypeSystem | `type_system.h/cpp` | 210 | ✅ Complete |
| FunctionCache | `function_cache.h/cpp` | 240 | ✅ Complete |
| MemoryCodegen | `memory_codegen.h/cpp` | 314 | ✅ Complete |

**Total extracted**: ~764 lines

### 2.2 Remaining in llvm_codegen.cpp

| Subsystem | Line Range | Approx Lines | Complexity |
|-----------|------------|--------------|------------|
| Class declaration & state | 1-320 | 320 | Medium |
| generateIR entry point | 322-930 | 608 | High |
| createBuiltinFunctions | 944-1380 | 436 | Low |
| Display tensor recursive | 1421-1635 | 214 | Low |
| Function declaration | 1636-1837 | 201 | Medium |
| Library init function | 1838-2130 | 292 | Medium |
| Main wrapper creation | 1947-2340 | 393 | High |
| Type inference helpers | 2342-2750 | 408 | Medium |
| Cons cell codegen | 2753-3005 | 252 | Medium |
| **Tagged value system** | 3006-4029 | 1023 | **Critical** |
| **Polymorphic arithmetic** | 4030-5170 | 1140 | **Critical** |
| **Main AST dispatcher** | 5171-6820 | 1649 | **Critical** |
| Function call codegen | 6821-7937 | 1116 | High |
| Math functions | 8096-8557 | 461 | Low |
| Control flow | 7938-9342 | 1404 | Medium |
| String operations | 9343-10357 | 1014 | Medium |
| Vector operations | 10358-10654 | 296 | Medium |
| Equality predicates | 10655-10926 | 271 | Low |
| I/O operations | 10927-11642 | 715 | Medium |
| **List operations** | 11643-13200 | 1557 | **High** |
| **Lambda/Closure** | 13201-15709 | 2508 | **Critical** |
| **Tensor operations** | 15710-17149 | 1439 | High |
| **Autodiff helpers** | 17150-18458 | 1308 | High |
| **Dual number system** | 18043-18458 | 415 | Medium |
| Nested gradient support | 18459-18665 | 206 | Medium |
| **AD node system** | 18666-19315 | 649 | High |
| **Gradient operators** | 19316-22279 | 2963 | **Critical** |
| Vector calculus | 22280-22959 | 679 | Medium |
| OALR codegen | 22960-23128 | 168 | Low |
| Symbolic diff | 23129-27758 | 4629 | High |

**Total remaining**: ~26,994 lines

### 2.3 Critical Interdependencies

```
                    ┌─────────────────┐
                    │   TypeSystem    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌───────────┐  ┌─────────────┐  ┌──────────────┐
      │ FuncCache │  │ MemoryCodeg │  │ TaggedValue  │
      └─────┬─────┘  └──────┬──────┘  └───────┬──────┘
            │               │                 │
            └───────────────┼─────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │   Arithmetic    │◄─────────────────┐
                  └────────┬────────┘                  │
                           │                           │
         ┌─────────────────┼─────────────────┐         │
         │                 │                 │         │
         ▼                 ▼                 ▼         │
   ┌──────────┐     ┌───────────┐     ┌──────────┐    │
   │  String  │     │   List    │     │ Control  │    │
   └──────────┘     └─────┬─────┘     └──────────┘    │
                          │                           │
                          ▼                           │
                   ┌─────────────┐                    │
                   │   Lambda    │────────────────────┤
                   └──────┬──────┘                    │
                          │                           │
         ┌────────────────┼────────────────┐          │
         │                │                │          │
         ▼                ▼                ▼          │
   ┌──────────┐    ┌───────────┐    ┌──────────┐     │
   │  Tensor  │    │ Autodiff  │    │   I/O    │     │
   └──────────┘    └───────────┘    └──────────┘     │
                          │                           │
                          ▼                           │
                   ┌─────────────┐                    │
                   │  Gradient   │────────────────────┘
                   │  Operators  │
                   └─────────────┘
```

---

## 3. Target Architecture

### 3.1 Directory Structure

```
inc/eshkol/
├── eshkol.h                      # Public API (unchanged)
├── llvm_backend.h                # LLVM backend API (unchanged)
├── logger.h                      # Logging utilities (unchanged)
│
└── backend/
    ├── codegen_fwd.h             # Forward declarations for all codegen types
    ├── codegen_context.h         # Shared state and symbol tables
    │
    ├── core/
    │   ├── type_system.h         # ✅ Done - LLVM type management
    │   ├── function_cache.h      # ✅ Done - C library functions
    │   ├── memory_codegen.h      # ✅ Done - Arena functions
    │   ├── tagged_value.h        # Tagged value pack/unpack
    │   └── builtin_decls.h       # Builtin function declarations
    │
    ├── ops/
    │   ├── arithmetic.h          # Polymorphic arithmetic
    │   ├── comparison.h          # Polymorphic comparisons
    │   ├── control_flow.h        # if/cond/case/and/or
    │   ├── string_ops.h          # String operations
    │   ├── list_ops.h            # List operations
    │   └── io_ops.h              # I/O operations
    │
    ├── functions/
    │   ├── lambda_codegen.h      # Lambda expression codegen
    │   ├── closure_codegen.h     # Closure creation and calling
    │   ├── function_codegen.h    # Function definition codegen
    │   └── call_codegen.h        # Function call dispatch
    │
    ├── collections/
    │   ├── vector_codegen.h      # Vector operations
    │   └── tensor_codegen.h      # Tensor/matrix operations
    │
    ├── autodiff/
    │   ├── dual_number.h         # Forward-mode AD
    │   ├── ad_node.h             # Reverse-mode graph nodes
    │   ├── ad_tape.h             # Computation tape management
    │   ├── backward_pass.h       # Backpropagation implementation
    │   ├── gradient_ops.h        # gradient/jacobian/hessian
    │   ├── vector_calculus.h     # div/curl/laplacian
    │   └── symbolic_diff.h       # Symbolic differentiation
    │
    └── llvm_codegen.h            # Main orchestrator (reduced)

lib/backend/
├── core/
│   ├── type_system.cpp           # ✅ Done
│   ├── function_cache.cpp        # ✅ Done
│   ├── memory_codegen.cpp        # ✅ Done
│   ├── tagged_value.cpp          # NEW
│   └── builtin_decls.cpp         # NEW
│
├── ops/
│   ├── arithmetic.cpp            # NEW
│   ├── comparison.cpp            # NEW
│   ├── control_flow.cpp          # NEW
│   ├── string_ops.cpp            # NEW
│   ├── list_ops.cpp              # NEW
│   └── io_ops.cpp                # NEW
│
├── functions/
│   ├── lambda_codegen.cpp        # NEW
│   ├── closure_codegen.cpp       # NEW
│   ├── function_codegen.cpp      # NEW
│   └── call_codegen.cpp          # NEW
│
├── collections/
│   ├── vector_codegen.cpp        # NEW
│   └── tensor_codegen.cpp        # NEW
│
├── autodiff/
│   ├── dual_number.cpp           # NEW
│   ├── ad_node.cpp               # NEW
│   ├── ad_tape.cpp               # NEW
│   ├── backward_pass.cpp         # NEW
│   ├── gradient_ops.cpp          # NEW
│   ├── vector_calculus.cpp       # NEW
│   └── symbolic_diff.cpp         # NEW
│
└── llvm_codegen.cpp              # Reduced to orchestration (~2500 lines)
```

### 3.2 Module Dependency Layers

```
Layer 0 (Foundation):
┌──────────────────────────────────────────────────────────┐
│  TypeSystem  │  FunctionCache  │  MemoryCodegen          │
└──────────────────────────────────────────────────────────┘

Layer 1 (Core Operations):
┌──────────────────────────────────────────────────────────┐
│  TaggedValue  │  BuiltinDecls  │  CodegenContext         │
└──────────────────────────────────────────────────────────┘

Layer 2 (Basic Operations):
┌──────────────────────────────────────────────────────────┐
│  Arithmetic  │  Comparison  │  ControlFlow  │  StringOps │
└──────────────────────────────────────────────────────────┘

Layer 3 (Complex Operations):
┌──────────────────────────────────────────────────────────┐
│  ListOps  │  VectorCodegen  │  IOOps                     │
└──────────────────────────────────────────────────────────┘

Layer 4 (Functions):
┌──────────────────────────────────────────────────────────┐
│  LambdaCodegen  │  ClosureCodegen  │  FunctionCodegen    │
└──────────────────────────────────────────────────────────┘

Layer 5 (Advanced):
┌──────────────────────────────────────────────────────────┐
│  TensorCodegen  │  DualNumber  │  ADNode  │  ADTape      │
└──────────────────────────────────────────────────────────┘

Layer 6 (Operators):
┌──────────────────────────────────────────────────────────┐
│  BackwardPass  │  GradientOps  │  VectorCalculus         │
└──────────────────────────────────────────────────────────┘

Layer 7 (Orchestration):
┌──────────────────────────────────────────────────────────┐
│                    LLVMCodegen                           │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Module Specifications

### 4.1 CodegenContext (NEW - Shared State)

**Purpose**: Centralize all shared state that modules need access to.

**File**: `inc/eshkol/backend/codegen_context.h`

```cpp
namespace eshkol {

class CodegenContext {
public:
    CodegenContext(llvm::LLVMContext& ctx, llvm::Module& mod);

    // LLVM Infrastructure
    llvm::LLVMContext& context();
    llvm::Module& module();
    llvm::IRBuilder<>& builder();

    // Type System Access
    TypeSystem& types();
    FunctionCache& funcs();
    MemoryCodegen& memory();

    // Symbol Tables
    void pushScope();
    void popScope();
    llvm::Value* lookupSymbol(const std::string& name);
    void defineSymbol(const std::string& name, llvm::Value* val);

    llvm::Function* lookupFunction(const std::string& name);
    void defineFunction(const std::string& name, llvm::Function* func);

    // Global State
    llvm::GlobalVariable* globalArena();
    llvm::GlobalVariable* adModeActive();
    llvm::GlobalVariable* currentAdTape();

    // REPL Support
    bool isReplMode() const;
    void setReplMode(bool enabled);

    // Current Function Context
    llvm::Function* currentFunction();
    void setCurrentFunction(llvm::Function* func);

    // Variadic Function Info
    void registerVariadicFunction(const std::string& name,
                                   size_t fixedParams,
                                   bool isVariadic);
    std::pair<size_t, bool> getVariadicInfo(const std::string& name);

    // String Interning
    llvm::GlobalVariable* internString(const std::string& str);

private:
    // ... implementation details
};

} // namespace eshkol
```

### 4.2 TaggedValue (NEW - Core Type System)

**Purpose**: All tagged value packing/unpacking operations.

**File**: `inc/eshkol/backend/core/tagged_value.h`

```cpp
namespace eshkol {

class TaggedValueCodegen {
public:
    TaggedValueCodegen(CodegenContext& ctx);

    // === Packing Functions ===
    llvm::Value* packInt64(llvm::Value* val, bool isExact = true);
    llvm::Value* packDouble(llvm::Value* val);
    llvm::Value* packBool(llvm::Value* val);
    llvm::Value* packChar(llvm::Value* val);
    llvm::Value* packNull();
    llvm::Value* packPtr(llvm::Value* ptr, eshkol_value_type_t type,
                         uint8_t flags = 0);
    llvm::Value* packPtrWithFlags(llvm::Value* ptr,
                                   llvm::Value* type,
                                   llvm::Value* flags);

    // === Unpacking Functions ===
    llvm::Value* unpackInt64(llvm::Value* tagged);
    llvm::Value* unpackDouble(llvm::Value* tagged);
    llvm::Value* unpackBool(llvm::Value* tagged);
    llvm::Value* unpackPtr(llvm::Value* tagged);

    // === Type Inspection ===
    llvm::Value* getType(llvm::Value* tagged);
    llvm::Value* getFlags(llvm::Value* tagged);
    llvm::Value* isType(llvm::Value* tagged, eshkol_value_type_t type);

    // === Cons Cell Operations ===
    llvm::Value* extractCar(llvm::Value* consCell);
    llvm::Value* extractCdr(llvm::Value* consCell);
    llvm::Value* extractCarAsTaggedValue(llvm::Value* consCell);
    llvm::Value* extractCdrAsTaggedValue(llvm::Value* consCell);

    // === Dual Number Support ===
    llvm::Value* packDual(llvm::Value* value, llvm::Value* derivative);
    llvm::Value* unpackDualValue(llvm::Value* dual);
    llvm::Value* unpackDualDerivative(llvm::Value* dual);
    llvm::Value* packDualToTaggedValue(llvm::Value* dual);
    llvm::Value* unpackDualFromTaggedValue(llvm::Value* tagged);

    // === Conversion ===
    llvm::Value* typedValueToTagged(const TypedValue& tv);
    TypedValue taggedToTypedValue(llvm::Value* tagged,
                                   eshkol_value_type_t expectedType);

    // === Safe Extraction (with type checking) ===
    llvm::Value* safeExtractInt64(llvm::Value* tagged,
                                   llvm::BasicBlock* errorBlock);
    llvm::Value* safeExtractDouble(llvm::Value* tagged,
                                    llvm::BasicBlock* errorBlock);

private:
    CodegenContext& ctx_;

    // Field index constants from TypeSystem
    static constexpr unsigned TYPE_IDX = 0;
    static constexpr unsigned FLAGS_IDX = 1;
    static constexpr unsigned RESERVED_IDX = 2;
    static constexpr unsigned PADDING_IDX = 3;
    static constexpr unsigned DATA_IDX = 4;
};

} // namespace eshkol
```

### 4.3 Arithmetic (NEW - Polymorphic Math)

**Purpose**: Type-polymorphic arithmetic operations with AD support.

**File**: `inc/eshkol/backend/ops/arithmetic.h`

```cpp
namespace eshkol {

class ArithmeticCodegen {
public:
    ArithmeticCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === Core Polymorphic Operations ===
    // Handle: int64, double, dual numbers, AD nodes, tensors
    llvm::Value* add(llvm::Value* left, llvm::Value* right);
    llvm::Value* sub(llvm::Value* left, llvm::Value* right);
    llvm::Value* mul(llvm::Value* left, llvm::Value* right);
    llvm::Value* div(llvm::Value* left, llvm::Value* right);

    // === Unary Operations ===
    llvm::Value* neg(llvm::Value* operand);
    llvm::Value* abs(llvm::Value* operand);

    // === Integer-Specific ===
    llvm::Value* modulo(llvm::Value* left, llvm::Value* right);
    llvm::Value* quotient(llvm::Value* left, llvm::Value* right);
    llvm::Value* remainder(llvm::Value* left, llvm::Value* right);
    llvm::Value* gcd(llvm::Value* left, llvm::Value* right);
    llvm::Value* lcm(llvm::Value* left, llvm::Value* right);

    // === N-ary Operations ===
    llvm::Value* addN(const std::vector<llvm::Value*>& operands);
    llvm::Value* mulN(const std::vector<llvm::Value*>& operands);
    llvm::Value* min(const std::vector<llvm::Value*>& operands);
    llvm::Value* max(const std::vector<llvm::Value*>& operands);

    // === Math Functions (AD-aware) ===
    llvm::Value* sin(llvm::Value* operand);
    llvm::Value* cos(llvm::Value* operand);
    llvm::Value* tan(llvm::Value* operand);
    llvm::Value* exp(llvm::Value* operand);
    llvm::Value* log(llvm::Value* operand);
    llvm::Value* sqrt(llvm::Value* operand);
    llvm::Value* pow(llvm::Value* base, llvm::Value* exponent);

    // === Rounding ===
    llvm::Value* floor(llvm::Value* operand);
    llvm::Value* ceiling(llvm::Value* operand);
    llvm::Value* truncate(llvm::Value* operand);
    llvm::Value* round(llvm::Value* operand);

    // === Type Coercion ===
    llvm::Value* toDouble(llvm::Value* tagged);
    llvm::Value* toExact(llvm::Value* tagged);
    llvm::Value* toInexact(llvm::Value* tagged);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;

    // Internal helpers
    llvm::Value* binaryOp(llvm::Value* left, llvm::Value* right,
                          const std::string& opName,
                          std::function<llvm::Value*(llvm::Value*, llvm::Value*)> intOp,
                          std::function<llvm::Value*(llvm::Value*, llvm::Value*)> doubleOp,
                          std::function<llvm::Value*(llvm::Value*, llvm::Value*)> dualOp,
                          std::function<llvm::Value*(llvm::Value*, llvm::Value*)> adNodeOp);

    llvm::Value* tensorArithmetic(llvm::Value* left, llvm::Value* right,
                                   const std::string& op);
};

} // namespace eshkol
```

### 4.4 ListOps (NEW - List Operations)

**Purpose**: All Scheme list operations.

**File**: `inc/eshkol/backend/ops/list_ops.h`

```cpp
namespace eshkol {

class ListOpsCodegen {
public:
    ListOpsCodegen(CodegenContext& ctx,
                   TaggedValueCodegen& tagged,
                   ClosureCodegen& closures);

    // === Construction ===
    llvm::Value* cons(llvm::Value* car, llvm::Value* cdr);
    llvm::Value* list(const std::vector<llvm::Value*>& elements);
    llvm::Value* listStar(const std::vector<llvm::Value*>& elements);
    llvm::Value* makeList(llvm::Value* count, llvm::Value* fill);

    // === Access ===
    llvm::Value* car(llvm::Value* pair);
    llvm::Value* cdr(llvm::Value* pair);
    llvm::Value* caar(llvm::Value* pair);
    llvm::Value* cadr(llvm::Value* pair);
    llvm::Value* cdar(llvm::Value* pair);
    llvm::Value* cddr(llvm::Value* pair);
    // ... all c[ad]+r combinations

    llvm::Value* listRef(llvm::Value* list, llvm::Value* index);
    llvm::Value* listTail(llvm::Value* list, llvm::Value* count);

    // === Mutation ===
    llvm::Value* setCar(llvm::Value* pair, llvm::Value* value);
    llvm::Value* setCdr(llvm::Value* pair, llvm::Value* value);

    // === Properties ===
    llvm::Value* length(llvm::Value* list);
    llvm::Value* isNull(llvm::Value* value);
    llvm::Value* isPair(llvm::Value* value);
    llvm::Value* isList(llvm::Value* value);

    // === Transformations ===
    llvm::Value* append(const std::vector<llvm::Value*>& lists);
    llvm::Value* reverse(llvm::Value* list);
    llvm::Value* listCopy(llvm::Value* list);

    // === Higher-Order ===
    llvm::Value* map(llvm::Value* func, llvm::Value* list);
    llvm::Value* mapN(llvm::Value* func, const std::vector<llvm::Value*>& lists);
    llvm::Value* filter(llvm::Value* pred, llvm::Value* list);
    llvm::Value* foldLeft(llvm::Value* func, llvm::Value* init, llvm::Value* list);
    llvm::Value* foldRight(llvm::Value* func, llvm::Value* init, llvm::Value* list);
    llvm::Value* forEach(llvm::Value* func, llvm::Value* list);

    // === Search ===
    llvm::Value* member(llvm::Value* obj, llvm::Value* list);
    llvm::Value* memv(llvm::Value* obj, llvm::Value* list);
    llvm::Value* memq(llvm::Value* obj, llvm::Value* list);
    llvm::Value* assoc(llvm::Value* key, llvm::Value* alist);
    llvm::Value* assv(llvm::Value* key, llvm::Value* alist);
    llvm::Value* assq(llvm::Value* key, llvm::Value* alist);

    // === Conversion ===
    llvm::Value* listToVector(llvm::Value* list);
    llvm::Value* vectorToList(llvm::Value* vector);
    llvm::Value* listToString(llvm::Value* list);
    llvm::Value* stringToList(llvm::Value* string);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    ClosureCodegen& closures_;
};

} // namespace eshkol
```

### 4.5 LambdaCodegen (NEW - Lambda & Closure)

**Purpose**: Lambda expression and closure generation.

**File**: `inc/eshkol/backend/functions/lambda_codegen.h`

```cpp
namespace eshkol {

struct CaptureInfo {
    std::string name;
    llvm::Value* value;
    eshkol_value_type_t type;
};

class LambdaCodegen {
public:
    LambdaCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === Lambda Generation ===
    llvm::Value* generateLambda(const eshkol_operations_t* lambdaOp);
    llvm::Value* generateNamedLambda(const std::string& name,
                                      const eshkol_operations_t* lambdaOp);

    // === Closure Support ===
    std::vector<CaptureInfo> analyzeFreeVariables(const eshkol_ast_t* body,
                                                   const std::set<std::string>& params);
    llvm::Value* createClosure(llvm::Function* func,
                               const std::vector<CaptureInfo>& captures,
                               llvm::Value* sexprPtr);
    llvm::Value* callClosure(llvm::Value* closure,
                              const std::vector<llvm::Value*>& args);

    // === Homoiconicity ===
    llvm::Value* generateSExpr(const eshkol_operations_t* lambdaOp);
    void registerLambdaSExpr(llvm::Function* func, llvm::Value* sexpr,
                              const std::string& name);
    llvm::Value* lookupLambdaSExpr(llvm::Function* func);

    // === Nested Functions ===
    llvm::Function* generateNestedFunction(const std::string& name,
                                            const eshkol_operations_t* defineOp,
                                            const std::vector<CaptureInfo>& outerCaptures);

    // === Letrec Support ===
    void generateLetrecBindings(const std::vector<std::pair<std::string,
                                                            const eshkol_ast_t*>>& bindings);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;

    // Lambda naming
    size_t lambdaCounter_;
    std::string generateLambdaName();

    // Capture management
    std::unordered_map<std::string, std::vector<std::string>> functionCaptures_;

    // S-expression registry
    std::unordered_map<llvm::Function*, std::string> lambdaSExprMap_;
};

} // namespace eshkol
```

### 4.6 AutodiffCodegen (NEW - Automatic Differentiation)

**Purpose**: All automatic differentiation functionality.

**File**: `inc/eshkol/backend/autodiff/gradient_ops.h`

```cpp
namespace eshkol {

class GradientOpsCodegen {
public:
    GradientOpsCodegen(CodegenContext& ctx,
                       TaggedValueCodegen& tagged,
                       LambdaCodegen& lambdas,
                       TensorCodegen& tensors);

    // === Derivative Operators ===
    llvm::Value* derivative(llvm::Value* func, llvm::Value* point);
    llvm::Value* partialDerivative(llvm::Value* func, size_t varIndex,
                                    llvm::Value* point);

    // === Gradient (Reverse-Mode) ===
    llvm::Value* gradient(llvm::Value* func, llvm::Value* point);

    // === Higher-Order Derivatives ===
    llvm::Value* jacobian(llvm::Value* func, llvm::Value* point);
    llvm::Value* hessian(llvm::Value* func, llvm::Value* point);

    // === Vector Calculus ===
    llvm::Value* divergence(llvm::Value* vectorField, llvm::Value* point);
    llvm::Value* curl(llvm::Value* vectorField, llvm::Value* point);
    llvm::Value* laplacian(llvm::Value* scalarField, llvm::Value* point);

    // === Tape Management ===
    void pushTape();
    void popTape();
    llvm::Value* getCurrentTape();

    // === AD Mode Control ===
    void enterADMode();
    void exitADMode();
    llvm::Value* isADModeActive();

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    LambdaCodegen& lambdas_;
    TensorCodegen& tensors_;

    // Tape stack for nested gradients
    std::vector<llvm::Value*> tapeStack_;

    // Backward pass implementation
    llvm::Value* backwardPass(llvm::Value* tape, llvm::Value* outputNode);

    // Node creation
    llvm::Value* createADVariable(llvm::Value* value);
    llvm::Value* createADConstant(llvm::Value* value);
    llvm::Value* createADOperation(ad_node_type_t type,
                                    llvm::Value* input1,
                                    llvm::Value* input2,
                                    llvm::Value* result);
};

} // namespace eshkol
```

---

## 5. Dependency Graph

### 5.1 Build Order

Modules must be built in this order to satisfy dependencies:

```
Phase 1: Foundation (already done)
  1. TypeSystem
  2. FunctionCache
  3. MemoryCodegen

Phase 2: Core Infrastructure
  4. CodegenContext (depends on 1, 2, 3)
  5. TaggedValueCodegen (depends on 4)
  6. BuiltinDeclarations (depends on 4)

Phase 3: Basic Operations
  7. ArithmeticCodegen (depends on 5)
  8. ComparisonCodegen (depends on 5, 7)
  9. ControlFlowCodegen (depends on 5)
  10. StringOpsCodegen (depends on 5)

Phase 4: Collections
  11. VectorCodegen (depends on 5, 7)
  12. ListOpsCodegen (depends on 5, 7, 11)

Phase 5: Functions
  13. LambdaCodegen (depends on 5, 12)
  14. ClosureCodegen (depends on 5, 13)
  15. FunctionCodegen (depends on 13, 14)
  16. CallCodegen (depends on 14, 15)

Phase 6: Advanced Collections
  17. TensorCodegen (depends on 5, 7, 11)

Phase 7: Autodiff
  18. DualNumberCodegen (depends on 5, 7)
  19. ADNodeCodegen (depends on 5)
  20. ADTapeCodegen (depends on 19)
  21. BackwardPassCodegen (depends on 19, 20)
  22. GradientOpsCodegen (depends on 13, 17, 18, 19, 20, 21)
  23. VectorCalculusCodegen (depends on 22)
  24. SymbolicDiffCodegen (depends on 5)

Phase 8: I/O
  25. IOOpsCodegen (depends on 5, 13)

Phase 9: Orchestration
  26. LLVMCodegen (depends on all above)
```

### 5.2 Circular Dependency Prevention

**Problem**: Lambda depends on List (for captures), List depends on Lambda (for map/filter).

**Solution**: Use forward declarations and interfaces:

```cpp
// In codegen_fwd.h
namespace eshkol {
    class CodegenContext;
    class TaggedValueCodegen;
    class ArithmeticCodegen;
    class ListOpsCodegen;
    class LambdaCodegen;
    class ClosureCodegen;
    // ... etc
}

// In list_ops.h
class ListOpsCodegen {
public:
    // Closure parameter is a function pointer, not ClosureCodegen
    llvm::Value* map(llvm::Value* closurePtr, llvm::Value* list);
};

// The actual closure call is delegated to CallCodegen
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Create CodegenContext
**Priority**: Critical
**Effort**: 2 days
**Risk**: Low

```
Tasks:
[ ] Create inc/eshkol/backend/codegen_context.h
[ ] Create lib/backend/codegen_context.cpp
[ ] Move symbol_table, function_table from EshkolLLVMCodeGen
[ ] Move global variables (arena, ad_mode, tape)
[ ] Add scope management (push/pop)
[ ] Update EshkolLLVMCodeGen to use CodegenContext
[ ] Run all tests
```

#### 1.2 Extract TaggedValueCodegen
**Priority**: Critical
**Effort**: 3 days
**Risk**: Medium (heavily used)

```
Tasks:
[ ] Create inc/eshkol/backend/core/tagged_value.h
[ ] Create lib/backend/core/tagged_value.cpp
[ ] Extract all pack* functions
[ ] Extract all unpack* functions
[ ] Extract extractCar/Cdr functions
[ ] Extract dual number pack/unpack
[ ] Add forwarding methods in EshkolLLVMCodeGen
[ ] Run all tests
[ ] Verify REPL functionality
```

#### 1.3 Extract BuiltinDeclarations
**Priority**: Medium
**Effort**: 1 day
**Risk**: Low

```
Tasks:
[ ] Create inc/eshkol/backend/core/builtin_decls.h
[ ] Create lib/backend/core/builtin_decls.cpp
[ ] Move createBuiltinFunctions content
[ ] Organize by category (math, io, string, etc.)
[ ] Update EshkolLLVMCodeGen to use BuiltinDecls
[ ] Run all tests
```

### Phase 2: Basic Operations (Week 2)

#### 2.1 Extract ArithmeticCodegen
**Priority**: Critical
**Effort**: 3 days
**Risk**: High (AD integration)

```
Tasks:
[ ] Create inc/eshkol/backend/ops/arithmetic.h
[ ] Create lib/backend/ops/arithmetic.cpp
[ ] Extract polymorphicAdd/Sub/Mul/Div
[ ] Extract math functions (sin, cos, exp, etc.)
[ ] Preserve AD node creation paths
[ ] Preserve dual number paths
[ ] Preserve tensor arithmetic paths
[ ] Run all autodiff tests
[ ] Run all arithmetic tests
```

#### 2.2 Extract ComparisonCodegen
**Priority**: Medium
**Effort**: 1 day
**Risk**: Low

```
Tasks:
[ ] Create inc/eshkol/backend/ops/comparison.h
[ ] Create lib/backend/ops/comparison.cpp
[ ] Extract polymorphicCompare
[ ] Extract equality predicates (eq?, eqv?, equal?)
[ ] Run all tests
```

#### 2.3 Extract ControlFlowCodegen
**Priority**: Medium
**Effort**: 2 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/ops/control_flow.h
[ ] Create lib/backend/ops/control_flow.cpp
[ ] Extract codegenIf
[ ] Extract codegenCond
[ ] Extract codegenCase
[ ] Extract codegenAnd/Or
[ ] Extract codegenWhen/Unless
[ ] Extract codegenDo
[ ] Run all tests
```

### Phase 3: String & I/O (Week 3)

#### 3.1 Extract StringOpsCodegen
**Priority**: Medium
**Effort**: 2 days
**Risk**: Low

```
Tasks:
[ ] Create inc/eshkol/backend/ops/string_ops.h
[ ] Create lib/backend/ops/string_ops.cpp
[ ] Extract all string operations
[ ] Extract string/number conversions
[ ] Run string tests
```

#### 3.2 Extract IOOpsCodegen
**Priority**: Medium
**Effort**: 1 day
**Risk**: Low

```
Tasks:
[ ] Create inc/eshkol/backend/ops/io_ops.h
[ ] Create lib/backend/ops/io_ops.cpp
[ ] Extract display/write operations
[ ] Extract file I/O operations
[ ] Extract printf-style formatting
[ ] Run I/O tests
```

### Phase 4: Collections (Week 4)

#### 4.1 Extract VectorCodegen
**Priority**: Medium
**Effort**: 1 day
**Risk**: Low

```
Tasks:
[ ] Create inc/eshkol/backend/collections/vector_codegen.h
[ ] Create lib/backend/collections/vector_codegen.cpp
[ ] Extract vector operations
[ ] Run vector tests
```

#### 4.2 Extract ListOpsCodegen
**Priority**: High
**Effort**: 3 days
**Risk**: Medium (closure integration)

```
Tasks:
[ ] Create inc/eshkol/backend/ops/list_ops.h
[ ] Create lib/backend/ops/list_ops.cpp
[ ] Extract cons/car/cdr operations
[ ] Extract list construction
[ ] Extract list transformations
[ ] Extract higher-order operations (map, filter, fold)
[ ] Handle closure calls properly
[ ] Run all list tests
```

### Phase 5: Functions (Week 5-6)

#### 5.1 Extract LambdaCodegen
**Priority**: Critical
**Effort**: 4 days
**Risk**: High (most complex)

```
Tasks:
[ ] Create inc/eshkol/backend/functions/lambda_codegen.h
[ ] Create lib/backend/functions/lambda_codegen.cpp
[ ] Extract codegenLambda
[ ] Extract free variable analysis
[ ] Extract capture mechanism
[ ] Extract S-expression generation
[ ] Extract lambda registry
[ ] Run closure tests
[ ] Run lambda tests
[ ] Verify REPL lambda support
```

#### 5.2 Extract ClosureCodegen
**Priority**: High
**Effort**: 2 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/functions/closure_codegen.h
[ ] Create lib/backend/functions/closure_codegen.cpp
[ ] Extract closure creation
[ ] Extract codegenClosureCall
[ ] Extract environment capture
[ ] Run all closure tests
```

#### 5.3 Extract FunctionCodegen
**Priority**: High
**Effort**: 2 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/functions/function_codegen.h
[ ] Create lib/backend/functions/function_codegen.cpp
[ ] Extract codegenDefine (function case)
[ ] Extract codegenNestedFunctionDefinition
[ ] Extract function declaration creation
[ ] Run function definition tests
```

#### 5.4 Extract CallCodegen
**Priority**: High
**Effort**: 2 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/functions/call_codegen.h
[ ] Create lib/backend/functions/call_codegen.cpp
[ ] Extract codegenCall
[ ] Extract variadic argument handling
[ ] Extract builtin call dispatch
[ ] Run all call tests
```

### Phase 6: Tensors (Week 7)

#### 6.1 Extract TensorCodegen
**Priority**: Medium
**Effort**: 3 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/collections/tensor_codegen.h
[ ] Create lib/backend/collections/tensor_codegen.cpp
[ ] Extract tensor creation
[ ] Extract tensor operations
[ ] Extract tensor arithmetic
[ ] Run tensor tests
[ ] Run ML tests
```

### Phase 7: Autodiff (Week 8-9)

#### 7.1 Extract DualNumberCodegen
**Priority**: High
**Effort**: 2 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/autodiff/dual_number.h
[ ] Create lib/backend/autodiff/dual_number.cpp
[ ] Extract dual number creation
[ ] Extract dual arithmetic
[ ] Extract dual math functions
[ ] Run forward-mode AD tests
```

#### 7.2 Extract ADNodeCodegen
**Priority**: High
**Effort**: 2 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/autodiff/ad_node.h
[ ] Create lib/backend/autodiff/ad_node.cpp
[ ] Extract AD node creation
[ ] Extract AD operation nodes
[ ] Extract AD variable nodes
[ ] Run reverse-mode AD tests
```

#### 7.3 Extract ADTapeCodegen
**Priority**: High
**Effort**: 1 day
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/autodiff/ad_tape.h
[ ] Create lib/backend/autodiff/ad_tape.cpp
[ ] Extract tape creation
[ ] Extract tape push/pop
[ ] Extract node recording
```

#### 7.4 Extract BackwardPassCodegen
**Priority**: High
**Effort**: 2 days
**Risk**: High

```
Tasks:
[ ] Create inc/eshkol/backend/autodiff/backward_pass.h
[ ] Create lib/backend/autodiff/backward_pass.cpp
[ ] Extract backward pass implementation
[ ] Extract gradient accumulation
[ ] Run gradient tests
```

#### 7.5 Extract GradientOpsCodegen
**Priority**: High
**Effort**: 3 days
**Risk**: High

```
Tasks:
[ ] Create inc/eshkol/backend/autodiff/gradient_ops.h
[ ] Create lib/backend/autodiff/gradient_ops.cpp
[ ] Extract derivative operator
[ ] Extract gradient operator
[ ] Extract jacobian operator
[ ] Extract hessian operator
[ ] Run all autodiff tests
```

#### 7.6 Extract VectorCalculusCodegen
**Priority**: Medium
**Effort**: 2 days
**Risk**: Medium

```
Tasks:
[ ] Create inc/eshkol/backend/autodiff/vector_calculus.h
[ ] Create lib/backend/autodiff/vector_calculus.cpp
[ ] Extract divergence operator
[ ] Extract curl operator
[ ] Extract laplacian operator
[ ] Run vector calculus tests
```

#### 7.7 Extract SymbolicDiffCodegen
**Priority**: Low
**Effort**: 2 days
**Risk**: Low

```
Tasks:
[ ] Create inc/eshkol/backend/autodiff/symbolic_diff.h
[ ] Create lib/backend/autodiff/symbolic_diff.cpp
[ ] Extract symbolic differentiation
[ ] Extract AST transformation
[ ] Run symbolic diff tests
```

### Phase 8: Final Integration (Week 10)

#### 8.1 Refactor Main Dispatcher
**Priority**: Critical
**Effort**: 3 days
**Risk**: High

```
Tasks:
[ ] Refactor codegenAST to delegate to modules
[ ] Refactor codegenOperation to delegate
[ ] Remove all duplicate code
[ ] Simplify generateIR
[ ] Run ALL tests
```

#### 8.2 Remove Backward Compatibility Shims
**Priority**: Low
**Effort**: 2 days
**Risk**: Low

```
Tasks:
[ ] Remove forwarding methods
[ ] Update all internal callers
[ ] Clean up unused code
[ ] Final test run
```

#### 8.3 Documentation & Cleanup
**Priority**: Medium
**Effort**: 2 days
**Risk**: Low

```
Tasks:
[ ] Document all public APIs
[ ] Add usage examples
[ ] Update build documentation
[ ] Create architecture diagram
[ ] Final code review
```

---

## 7. Testing Strategy

### 7.1 Test Categories

| Category | Location | Count | Coverage |
|----------|----------|-------|----------|
| List Operations | tests/lists/ | 129 | Core list functionality |
| Autodiff | tests/autodiff_debug/ | 51 | AD operators |
| Memory | tests/memory/ | ~20 | Arena, closures |
| ML/Tensor | tests/ml/ | ~30 | Tensor operations |
| Neural | tests/neural/ | ~20 | Neural network ops |
| Stdlib | tests/stdlib/ | ~30 | Standard library |

### 7.2 Test Protocol

**After EVERY extraction:**

1. **Build Check**
   ```bash
   cmake --build build -j8
   ```
   Must compile with no errors, warnings acceptable.

2. **Quick Smoke Test**
   ```bash
   ./build/eshkol-run examples/factorial.esk
   ./build/eshkol-run examples/lambda_closure_test.esk
   ```

3. **Full Test Suite**
   ```bash
   ./scripts/run_list_tests.sh
   ./scripts/run_autodiff_tests.sh
   ./scripts/run_ml_tests.sh
   ```
   Must maintain 100% pass rate.

4. **REPL Test**
   ```bash
   echo "(define (f x) (* x x)) (f 5)" | ./build/eshkol-repl
   ```
   Must work correctly.

### 7.3 Regression Prevention

Create a test manifest that tracks:
- Test name
- Expected output
- Actual output
- Pass/fail status

Any regression must be fixed before proceeding.

---

## 8. Migration Guidelines

### 8.1 Code Extraction Template

When extracting a module, follow this template:

**Header File** (`inc/eshkol/backend/[category]/[module].h`):
```cpp
/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * [ModuleName] - [Brief description]
 */
#ifndef ESHKOL_BACKEND_[CATEGORY]_[MODULE]_H
#define ESHKOL_BACKEND_[CATEGORY]_[MODULE]_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
// Other includes...

namespace eshkol {

/**
 * [ModuleName] handles [detailed description].
 *
 * Dependencies:
 *   - CodegenContext: [why needed]
 *   - [Other deps]: [why needed]
 *
 * Usage:
 *   [ModuleName] mod(ctx, deps...);
 *   auto result = mod.someOperation(args...);
 */
class [ModuleName] {
public:
    explicit [ModuleName](CodegenContext& ctx, [other deps]);

    // === [Category 1] ===
    llvm::Value* operation1(...);

    // === [Category 2] ===
    llvm::Value* operation2(...);

private:
    CodegenContext& ctx_;
    // Other members...
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_[CATEGORY]_[MODULE]_H
```

**Implementation File** (`lib/backend/[category]/[module].cpp`):
```cpp
/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * [ModuleName] implementation
 */

#include <eshkol/backend/[category]/[module].h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

namespace eshkol {

[ModuleName]::[ModuleName](CodegenContext& ctx, [deps])
    : ctx_(ctx), [other inits] {
    // Initialization
}

llvm::Value* [ModuleName]::operation1(...) {
    // Implementation
    // Use ctx_.builder() for IR building
    // Use ctx_.types() for type access
    // etc.
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
```

### 8.2 Backward Compatibility

During migration, add forwarding methods:

```cpp
// In EshkolLLVMCodeGen (temporary, remove in Phase 8)
Value* packInt64ToTaggedValue(Value* val, bool exact = true) {
    return tagged_->packInt64(val, exact);
}
```

### 8.3 Commit Guidelines

Each extraction should be a single commit:

```
refactor(backend): extract [ModuleName] from llvm_codegen

- Create inc/eshkol/backend/[category]/[module].h
- Create lib/backend/[category]/[module].cpp
- Move [N] functions from llvm_codegen.cpp
- Add forwarding methods for backward compatibility
- All tests passing

Lines moved: ~[N]
Lines remaining in llvm_codegen.cpp: ~[M]
```

---

## 9. Future Extensibility

### 9.1 New Language Features

The modular architecture enables easy addition of:

**New Types**:
1. Add type constant to `eshkol.h`
2. Add LLVM type to `TypeSystem`
3. Add pack/unpack to `TaggedValueCodegen`
4. Add operations to relevant modules

**New Operators**:
1. Add operator enum to `eshkol.h`
2. Add parsing in `parser.cpp`
3. Add codegen in appropriate module
4. Add to dispatcher in `LLVMCodegen`

**New AD Operations**:
1. Add to `GradientOpsCodegen`
2. Add node type to `ADNodeCodegen`
3. Add backward rule to `BackwardPassCodegen`

### 9.2 Backend Alternatives

The architecture allows for alternative backends:

```
inc/eshkol/backend/
├── interface/
│   ├── codegen_interface.h    # Abstract interface
│   └── value_interface.h      # Value representation interface
│
├── llvm/                      # LLVM backend (current)
│   ├── llvm_codegen.h
│   └── ...
│
├── c/                         # Future: C code generation
│   ├── c_codegen.h
│   └── ...
│
└── wasm/                      # Future: WebAssembly
    ├── wasm_codegen.h
    └── ...
```

### 9.3 Optimization Passes

Modular structure enables optimization:

```cpp
class OptimizationPipeline {
public:
    void addPass(std::unique_ptr<OptimizationPass> pass);
    void run(llvm::Module& module);
};

// Example passes:
// - ConstantFolding
// - DeadCodeElimination
// - InlineExpansion
// - TailCallOptimization
// - ADGraphSimplification
```

### 9.4 Plugin System

Future plugin architecture:

```cpp
class CodegenPlugin {
public:
    virtual ~CodegenPlugin() = default;
    virtual std::string name() const = 0;
    virtual void initialize(CodegenContext& ctx) = 0;
    virtual bool canHandle(const eshkol_operations_t* op) = 0;
    virtual llvm::Value* codegen(const eshkol_operations_t* op) = 0;
};

// Register custom operations:
// codegen.registerPlugin(std::make_unique<MyCustomPlugin>());
```

---

## 10. Risk Assessment

### 10.1 High-Risk Areas

| Area | Risk | Mitigation |
|------|------|------------|
| TaggedValue extraction | May break all operations | Extensive testing, phased rollout |
| Lambda/Closure | Complex state management | Careful capture analysis |
| AD integration | Multiple interacting systems | Dedicated AD test suite |
| REPL symbol resolution | Cross-module complexity | REPL-specific tests |

### 10.2 Rollback Strategy

If a phase causes issues:

1. **Immediate**: Revert commit, analyze failure
2. **Short-term**: Fix in isolation, re-test
3. **Long-term**: Adjust extraction strategy if needed

### 10.3 Performance Considerations

Monitor for performance regression:

```bash
# Benchmark compile time
time ./build/eshkol-run examples/large_program.esk

# Benchmark runtime
./build/eshkol-run benchmarks/fibonacci_35.esk
```

Acceptable overhead: <5% compile time, 0% runtime.

---

## 11. Appendix

### 11.1 File Line Counts (Current)

```
lib/backend/llvm_codegen.cpp    27,758
lib/frontend/parser.cpp          3,685
lib/core/arena_memory.cpp        1,778
lib/backend/memory_codegen.cpp     177
lib/backend/function_cache.cpp     151
lib/backend/type_system.cpp         96
```

### 11.2 Key Function Signatures

```cpp
// Main entry point
std::pair<std::unique_ptr<Module>, std::unique_ptr<LLVMContext>>
generateIR(const eshkol_ast_t* asts, size_t num_asts);

// AST codegen
Value* codegenAST(const eshkol_ast_t* ast);
TypedValue codegenTypedAST(const eshkol_ast_t* ast);

// Operation dispatch
Value* codegenOperation(const eshkol_operations_t* op);

// Polymorphic operations
Value* polymorphicAdd(Value* left, Value* right);
Value* polymorphicSub(Value* left, Value* right);
Value* polymorphicMul(Value* left, Value* right);
Value* polymorphicDiv(Value* left, Value* right);

// Tagged value operations
Value* packInt64ToTaggedValue(Value* val, bool exact);
Value* unpackInt64FromTaggedValue(Value* tagged);
```

### 11.3 CMakeLists.txt Updates

After refactoring, update CMakeLists.txt:

```cmake
set(BACKEND_SOURCES
    # Core
    lib/backend/core/type_system.cpp
    lib/backend/core/function_cache.cpp
    lib/backend/core/memory_codegen.cpp
    lib/backend/core/tagged_value.cpp
    lib/backend/core/builtin_decls.cpp
    lib/backend/codegen_context.cpp

    # Operations
    lib/backend/ops/arithmetic.cpp
    lib/backend/ops/comparison.cpp
    lib/backend/ops/control_flow.cpp
    lib/backend/ops/string_ops.cpp
    lib/backend/ops/list_ops.cpp
    lib/backend/ops/io_ops.cpp

    # Functions
    lib/backend/functions/lambda_codegen.cpp
    lib/backend/functions/closure_codegen.cpp
    lib/backend/functions/function_codegen.cpp
    lib/backend/functions/call_codegen.cpp

    # Collections
    lib/backend/collections/vector_codegen.cpp
    lib/backend/collections/tensor_codegen.cpp

    # Autodiff
    lib/backend/autodiff/dual_number.cpp
    lib/backend/autodiff/ad_node.cpp
    lib/backend/autodiff/ad_tape.cpp
    lib/backend/autodiff/backward_pass.cpp
    lib/backend/autodiff/gradient_ops.cpp
    lib/backend/autodiff/vector_calculus.cpp
    lib/backend/autodiff/symbolic_diff.cpp

    # Main orchestrator
    lib/backend/llvm_codegen.cpp
)
```

### 11.4 References

- LLVM Documentation: https://llvm.org/docs/
- Scheme R7RS-small: https://small.r7rs.org/
- Automatic Differentiation: https://arxiv.org/abs/1502.05767

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2024 | Claude | Initial comprehensive plan |

---

**Next Step**: Begin Phase 1.1 - Create CodegenContext
