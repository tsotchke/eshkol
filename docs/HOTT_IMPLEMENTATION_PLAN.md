# Eshkol HoTT Type System Implementation Plan

## Detailed Architectural Specification

**Version:** 1.0
**Date:** 2025-12-01
**Status:** Implementation Blueprint
**Scope:** Complete HoTT type system integration with existing compiler

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Implementation Phases](#3-implementation-phases)
4. [Phase 1: Type Infrastructure](#4-phase-1-type-infrastructure)
5. [Phase 2: Parser Extensions](#5-phase-2-parser-extensions)
6. [Phase 3: Type Checker](#6-phase-3-type-checker)
7. [Phase 4: Codegen Integration](#7-phase-4-codegen-integration)
8. [Phase 5: Dependent Types](#8-phase-5-dependent-types)
9. [Phase 6: Linear Types](#9-phase-6-linear-types)
10. [File-by-File Changes](#10-file-by-file-changes)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Current Architecture Analysis

### 1.1 Compiler Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────▶│   Parser    │────▶│   Codegen   │────▶│   Runtime   │
│   (.esk)    │     │ parser.cpp  │     │llvm_codegen │     │arena_memory │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    eshkol_ast_t         TypedValue
                    (eshkol_type_t)      (eshkol_value_type_t)
```

### 1.2 Key Source Files

| File | Purpose | Lines | Key Structures |
|------|---------|-------|----------------|
| `inc/eshkol/eshkol.h` | Core types and AST | ~485 | `eshkol_ast_t`, `eshkol_type_t`, `eshkol_tagged_value_t` |
| `lib/frontend/parser.cpp` | Tokenizer and parser | ~2000+ | `SchemeTokenizer`, `parse_*` functions |
| `lib/backend/llvm_codegen.cpp` | LLVM IR generation | ~22000+ | `TypedValue`, `EshkolLLVMCodeGen` |
| `lib/core/arena_memory.h` | Runtime memory mgmt | ~245 | `arena_t`, `arena_tagged_cons_cell_t` |
| `inc/eshkol/llvm_backend.h` | Backend interface | ~166 | Function declarations |

### 1.3 Type Information Flow

```cpp
// Parse-time types (flat enum, no hierarchy)
typedef enum {
    ESHKOL_INVALID, ESHKOL_UNTYPED, ESHKOL_UINT8, ... ESHKOL_INT64,
    ESHKOL_DOUBLE, ESHKOL_STRING, ESHKOL_FUNC, ESHKOL_VAR, ESHKOL_OP,
    ESHKOL_CONS, ESHKOL_NULL, ESHKOL_TENSOR, ESHKOL_CHAR
} eshkol_type_t;  // in eshkol.h:19-39

// Runtime value types (4-bit tag + flags)
typedef enum {
    ESHKOL_VALUE_NULL = 0, ESHKOL_VALUE_INT64 = 1, ESHKOL_VALUE_DOUBLE = 2,
    ESHKOL_VALUE_CONS_PTR = 3, ESHKOL_VALUE_DUAL_NUMBER = 4, ...
    ESHKOL_VALUE_MAX = 15  // 4-bit limit
} eshkol_value_type_t;  // in eshkol.h:42-58

// Codegen type tracking
struct TypedValue {
    Value* llvm_value;
    eshkol_value_type_t type;
    bool is_exact;
    uint8_t flags;
};  // in llvm_codegen.cpp:86-104
```

### 1.4 Limitations to Address

1. **No type annotations**: Types inferred from literals only
2. **Flat type hierarchy**: No subtyping relationships
3. **No compile-time checking**: All errors are runtime
4. **No dependent types**: Cannot express dimension constraints
5. **No linear types**: Cannot track resource usage
6. **No proof system**: Cannot express type-level propositions

---

## 2. Target Architecture

### 2.1 Extended Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────▶│   Parser    │────▶│    Type     │────▶│   Codegen   │────▶│   Runtime   │
│   (.esk)    │     │ + Type Ann  │     │   Checker   │     │ + Erasure   │     │  (unchanged)│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                    eshkol_ast_t        StaticType           TypedValue
                    + TypeExpr*         (HoTT types)         + StaticType
```

### 2.2 New Components

| Component | File | Purpose |
|-----------|------|---------|
| Type representation | `lib/types/type_system.h/cpp` | TypeId, TypeNode, Universe levels |
| Type environment | `lib/types/type_env.h/cpp` | Type context, subtyping |
| Type checker | `lib/types/type_checker.h/cpp` | Bidirectional type checking |
| Linear checker | `lib/types/linear_checker.h/cpp` | Resource usage tracking |

---

## 3. Implementation Phases

```
Phase 1: Type Infrastructure (Foundation)
├── TypeId structure with universe levels
├── TypeNode for type hierarchy
├── TypeEnvironment for type context
└── Subtype checking

Phase 2: Parser Extensions (Syntax)
├── Type annotation tokens (: name type)
├── Type expression parsing
├── AST extensions for type fields
└── Backward compatibility

Phase 3: Type Checker (Core HoTT)
├── Bidirectional type checking
├── Type inference algorithm
├── Subtype verification
└── Error reporting

Phase 4: Codegen Integration (Proof Erasure)
├── Extended TypedValue
├── Type-directed code generation
├── Proof erasure implementation
└── Optimization passes

Phase 5: Dependent Types (Advanced)
├── Dependent function types (Π-types)
├── Dependent pair types (Σ-types)
├── Dimension tracking
└── Compile-time evaluation

Phase 6: Linear Types (Resources)
├── Linear modality
├── Usage tracking
├── Borrow checking
└── Escape hatches
```

---

## 4. Phase 1: Type Infrastructure

### 4.1 New Files to Create

**File: `lib/types/type_system.h`**

```cpp
#ifndef ESHKOL_TYPE_SYSTEM_H
#define ESHKOL_TYPE_SYSTEM_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <memory>

namespace eshkol::types {

// ============================================================================
// UNIVERSE LEVELS - HoTT Foundation
// ============================================================================

enum class Universe : uint8_t {
    U0 = 0,      // Ground types: integer, float64, string, char, boolean
    U1 = 1,      // Type constructors: list, vector, ->, ×, Handle, Buffer
    U2 = 2,      // Propositions: Eq, <:, Bounded, Linear
    UOmega = 255 // Universe polymorphic
};

// ============================================================================
// TYPE FLAGS
// ============================================================================

enum TypeFlags : uint8_t {
    NONE     = 0,
    EXACT    = 1 << 0,  // Scheme exactness (integer vs inexact)
    LINEAR   = 1 << 1,  // Must use exactly once
    PROOF    = 1 << 2,  // Compile-time only (erased at runtime)
    ABSTRACT = 1 << 3   // Cannot be instantiated directly
};

// ============================================================================
// TYPE IDENTIFIER
// ============================================================================

struct TypeId {
    uint16_t id;          // Unique identifier
    Universe level;       // Universe membership
    uint8_t flags;        // Type flags

    bool operator==(const TypeId& other) const { return id == other.id; }
    bool operator<(const TypeId& other) const { return id < other.id; }

    // Flag queries
    bool isExact() const { return flags & TypeFlags::EXACT; }
    bool isLinear() const { return flags & TypeFlags::LINEAR; }
    bool isProof() const { return flags & TypeFlags::PROOF; }
    bool isAbstract() const { return flags & TypeFlags::ABSTRACT; }
};

// ============================================================================
// RUNTIME REPRESENTATION MAPPING
// ============================================================================

enum class RuntimeRep : uint8_t {
    Int64,       // 64-bit integer
    Float64,     // IEEE 754 double
    Pointer,     // Pointer to heap object
    TaggedValue, // eshkol_tagged_value_t
    Struct,      // LLVM struct (for dependent pairs, etc.)
    Erased       // No runtime representation (proofs)
};

// ============================================================================
// TYPE NODE - Full type definition
// ============================================================================

struct TypeNode {
    TypeId id;
    std::string name;

    // Hierarchy
    std::optional<TypeId> supertype;
    std::vector<TypeId> subtypes;

    // Parameterization (for List<A>, Buffer<A,n>, etc.)
    std::vector<TypeId> parameters;  // Parameter types (or universe levels)
    std::vector<std::string> param_names;  // Parameter names for display

    // Runtime mapping
    RuntimeRep runtime_rep;

    // For dependent types: is this a type family (takes value args)?
    bool is_type_family = false;

    // Source location for error messages
    enum class Origin { Builtin, UserDefined } origin = Origin::Builtin;
};

// ============================================================================
// TYPE EXPRESSION - AST for type annotations
// ============================================================================

struct TypeExpr {
    enum class Kind {
        Name,       // integer, float64, string
        Apply,      // (list integer), (vector float64 3)
        Arrow,      // (-> A B)
        Product,    // (* A B)
        Forall,     // (forall (a) (-> a a))
        Dependent,  // (forall (n : natural) (-> (vector float64 n) float64))
        Refine,     // (refine integer (lambda (x) (>= x 0)))
        Linear,     // (Linear A)
        Universe    // Type₀, Type₁
    };

    Kind kind;
    std::string name;  // For Name, Universe

    // For compound types
    std::vector<std::unique_ptr<TypeExpr>> args;  // For Apply
    std::unique_ptr<TypeExpr> left;   // For Arrow, Product
    std::unique_ptr<TypeExpr> right;  // For Arrow, Product

    // For quantified types
    std::vector<std::string> binder_names;  // For Forall, Dependent
    std::vector<std::unique_ptr<TypeExpr>> binder_types;  // For Dependent

    // For refinement types
    std::string predicate_source;  // Original source for error messages
};

// ============================================================================
// TYPE ENVIRONMENT
// ============================================================================

class TypeEnvironment {
private:
    std::map<TypeId, TypeNode> types_;
    std::map<std::string, TypeId> name_to_id_;
    uint16_t next_user_id_ = 1000;  // User-defined types start at 1000

    // Subtype cache for performance
    mutable std::map<std::pair<uint16_t, uint16_t>, bool> subtype_cache_;

public:
    TypeEnvironment();

    // ========== Type Registration ==========

    TypeId registerBuiltinType(const std::string& name, Universe level,
                               uint8_t flags, RuntimeRep rep,
                               std::optional<TypeId> supertype = std::nullopt);

    TypeId registerTypeFamily(const std::string& name, Universe level,
                              const std::vector<std::string>& param_names,
                              RuntimeRep rep);

    TypeId registerUserType(const std::string& name, Universe level,
                            uint8_t flags, std::optional<TypeId> supertype);

    // ========== Type Lookup ==========

    std::optional<TypeId> lookupType(const std::string& name) const;
    const TypeNode* getTypeNode(TypeId id) const;

    // ========== Subtyping ==========

    bool isSubtype(TypeId sub, TypeId super) const;
    std::optional<TypeId> leastCommonSupertype(TypeId a, TypeId b) const;
    std::vector<TypeId> getSupertypeChain(TypeId type) const;

    // ========== Arithmetic Promotion ==========

    TypeId promoteForArithmetic(TypeId a, TypeId b) const;

    // ========== Type Application ==========

    // Apply type constructor to arguments: (list integer) -> List<integer>
    TypeId applyType(TypeId constructor, const std::vector<TypeId>& args);

    // ========== Type Equivalence ==========

    bool areEquivalent(TypeId a, TypeId b) const;

private:
    void initializeBuiltinTypes();
    bool isSubtypeUncached(TypeId sub, TypeId super) const;
};

// ============================================================================
// BUILTIN TYPE IDS (Compile-time constants)
// ============================================================================

namespace BuiltinTypes {
    // Universes
    inline constexpr TypeId TypeU0{0, Universe::U0, 0};
    inline constexpr TypeId TypeU1{1, Universe::U1, 0};
    inline constexpr TypeId TypeU2{2, Universe::U2, 0};

    // Root supertypes
    inline constexpr TypeId Value{10, Universe::U0, 0};

    // Numeric tower
    inline constexpr TypeId Number{11, Universe::U0, 0};
    inline constexpr TypeId Integer{12, Universe::U0, TypeFlags::EXACT};
    inline constexpr TypeId Int64{13, Universe::U0, TypeFlags::EXACT};
    inline constexpr TypeId Natural{14, Universe::U0, TypeFlags::EXACT};
    inline constexpr TypeId Real{15, Universe::U0, 0};
    inline constexpr TypeId Float64{16, Universe::U0, 0};

    // Text types
    inline constexpr TypeId Text{20, Universe::U0, 0};
    inline constexpr TypeId String{21, Universe::U0, 0};
    inline constexpr TypeId Char{22, Universe::U0, TypeFlags::EXACT};

    // Other ground types
    inline constexpr TypeId Boolean{25, Universe::U0, TypeFlags::EXACT};
    inline constexpr TypeId Null{26, Universe::U0, 0};
    inline constexpr TypeId Symbol{27, Universe::U0, 0};

    // Type constructors (U1)
    inline constexpr TypeId List{100, Universe::U1, 0};
    inline constexpr TypeId Vector{101, Universe::U1, 0};
    inline constexpr TypeId Pair{102, Universe::U1, 0};
    inline constexpr TypeId Function{103, Universe::U1, 0};
    inline constexpr TypeId Tensor{104, Universe::U1, 0};

    // Resource types (U1, Linear)
    inline constexpr TypeId Handle{110, Universe::U1, TypeFlags::LINEAR};
    inline constexpr TypeId Buffer{111, Universe::U1, 0};
    inline constexpr TypeId Stream{112, Universe::U1, TypeFlags::LINEAR};

    // Proposition types (U2, Proof - erased at runtime)
    inline constexpr TypeId Eq{200, Universe::U2, TypeFlags::PROOF};
    inline constexpr TypeId LessThan{201, Universe::U2, TypeFlags::PROOF};
    inline constexpr TypeId Bounded{202, Universe::U2, TypeFlags::PROOF};
}

} // namespace eshkol::types

#endif // ESHKOL_TYPE_SYSTEM_H
```

**File: `lib/types/type_system.cpp`**

```cpp
#include "type_system.h"

namespace eshkol::types {

TypeEnvironment::TypeEnvironment() {
    initializeBuiltinTypes();
}

void TypeEnvironment::initializeBuiltinTypes() {
    using namespace BuiltinTypes;

    // Register universes (not really types, but useful for lookup)
    types_[TypeU0] = {TypeU0, "Type₀", std::nullopt, {}, {}, {}, RuntimeRep::Erased, false};
    types_[TypeU1] = {TypeU1, "Type₁", std::nullopt, {}, {}, {}, RuntimeRep::Erased, false};
    types_[TypeU2] = {TypeU2, "Type₂", std::nullopt, {}, {}, {}, RuntimeRep::Erased, false};

    // Root: Value supertype
    types_[Value] = {Value, "Value", std::nullopt,
                     {Number, Text, Boolean, Null, Symbol}, {}, {}, RuntimeRep::TaggedValue, false};

    // Numeric tower
    types_[Number] = {Number, "Number", Value, {Integer, Real}, {}, {}, RuntimeRep::TaggedValue, false};
    types_[Integer] = {Integer, "Integer", Number, {Int64, Natural}, {}, {}, RuntimeRep::Int64, false};
    types_[Int64] = {Int64, "Int64", Integer, {}, {}, {}, RuntimeRep::Int64, false};
    types_[Natural] = {Natural, "Natural", Integer, {}, {}, {}, RuntimeRep::Int64, false};
    types_[Real] = {Real, "Real", Number, {Float64}, {}, {}, RuntimeRep::Float64, false};
    types_[Float64] = {Float64, "Float64", Real, {}, {}, {}, RuntimeRep::Float64, false};

    // Text types
    types_[Text] = {Text, "Text", Value, {String, Char}, {}, {}, RuntimeRep::Pointer, false};
    types_[String] = {String, "String", Text, {}, {}, {}, RuntimeRep::Pointer, false};
    types_[Char] = {Char, "Char", Text, {}, {}, {}, RuntimeRep::Int64, false};

    // Other ground types
    types_[Boolean] = {Boolean, "Boolean", Value, {}, {}, {}, RuntimeRep::Int64, false};
    types_[Null] = {Null, "Null", Value, {}, {}, {}, RuntimeRep::Int64, false};
    types_[Symbol] = {Symbol, "Symbol", Value, {}, {}, {}, RuntimeRep::Pointer, false};

    // Type constructors (parameterized)
    types_[List] = {List, "List", std::nullopt, {}, {"a"}, {}, RuntimeRep::Pointer, true};
    types_[Vector] = {Vector, "Vector", std::nullopt, {}, {"a"}, {}, RuntimeRep::Pointer, true};
    types_[Pair] = {Pair, "Pair", std::nullopt, {}, {"a", "b"}, {}, RuntimeRep::Pointer, true};
    types_[Function] = {Function, "->", std::nullopt, {}, {"a", "b"}, {}, RuntimeRep::Pointer, true};
    types_[Tensor] = {Tensor, "Tensor", std::nullopt, {}, {"a", "shape"}, {}, RuntimeRep::Pointer, true};

    // Resource types
    types_[Handle] = {Handle, "Handle", std::nullopt, {}, {"k"}, {}, RuntimeRep::Pointer, true};
    types_[Buffer] = {Buffer, "Buffer", std::nullopt, {}, {"a", "n"}, {}, RuntimeRep::Pointer, true};
    types_[Stream] = {Stream, "Stream", std::nullopt, {}, {"a"}, {}, RuntimeRep::Pointer, true};

    // Proposition types (erased)
    types_[Eq] = {Eq, "Eq", std::nullopt, {}, {"a", "b"}, {}, RuntimeRep::Erased, true};
    types_[LessThan] = {LessThan, "<", std::nullopt, {}, {"a", "b"}, {}, RuntimeRep::Erased, true};
    types_[Bounded] = {Bounded, "Bounded", std::nullopt, {}, {"a", "lo", "hi"}, {}, RuntimeRep::Erased, true};

    // Build name lookup table
    for (const auto& [id, node] : types_) {
        name_to_id_[node.name] = id;
        // Add lowercase aliases
        std::string lower = node.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower != node.name) {
            name_to_id_[lower] = id;
        }
    }

    // Add common aliases
    name_to_id_["int"] = Int64;
    name_to_id_["integer"] = Int64;
    name_to_id_["float"] = Float64;
    name_to_id_["float64"] = Float64;
    name_to_id_["double"] = Float64;
    name_to_id_["string"] = String;
    name_to_id_["bool"] = Boolean;
    name_to_id_["boolean"] = Boolean;
    name_to_id_["number"] = Number;
    name_to_id_["natural"] = Natural;
}

bool TypeEnvironment::isSubtype(TypeId sub, TypeId super) const {
    // Check cache
    auto key = std::make_pair(sub.id, super.id);
    auto it = subtype_cache_.find(key);
    if (it != subtype_cache_.end()) return it->second;

    bool result = isSubtypeUncached(sub, super);
    subtype_cache_[key] = result;
    return result;
}

bool TypeEnvironment::isSubtypeUncached(TypeId sub, TypeId super) const {
    // Reflexivity
    if (sub == super) return true;

    // Walk supertype chain
    const TypeNode* node = getTypeNode(sub);
    if (!node) return false;

    while (node->supertype.has_value()) {
        if (node->supertype.value() == super) return true;
        node = getTypeNode(node->supertype.value());
        if (!node) return false;
    }

    return false;
}

std::vector<TypeId> TypeEnvironment::getSupertypeChain(TypeId type) const {
    std::vector<TypeId> chain;
    chain.push_back(type);

    const TypeNode* node = getTypeNode(type);
    while (node && node->supertype.has_value()) {
        chain.push_back(node->supertype.value());
        node = getTypeNode(node->supertype.value());
    }

    return chain;
}

std::optional<TypeId> TypeEnvironment::leastCommonSupertype(TypeId a, TypeId b) const {
    auto chain_a = getSupertypeChain(a);
    auto chain_b = getSupertypeChain(b);

    // Find first common element
    for (const auto& t : chain_a) {
        for (const auto& u : chain_b) {
            if (t == u) return t;
        }
    }

    return std::nullopt;
}

TypeId TypeEnvironment::promoteForArithmetic(TypeId a, TypeId b) const {
    using namespace BuiltinTypes;

    if (a == b) return a;

    // Integer + Real -> Real (Float64)
    if ((isSubtype(a, Integer) && isSubtype(b, Real)) ||
        (isSubtype(a, Real) && isSubtype(b, Integer))) {
        return Float64;
    }

    // Both integers -> Int64
    if (isSubtype(a, Integer) && isSubtype(b, Integer)) {
        return Int64;
    }

    // Both reals -> Float64
    if (isSubtype(a, Real) && isSubtype(b, Real)) {
        return Float64;
    }

    // Fallback
    return leastCommonSupertype(a, b).value_or(Value);
}

std::optional<TypeId> TypeEnvironment::lookupType(const std::string& name) const {
    auto it = name_to_id_.find(name);
    return (it != name_to_id_.end()) ? std::optional{it->second} : std::nullopt;
}

const TypeNode* TypeEnvironment::getTypeNode(TypeId id) const {
    auto it = types_.find(id);
    return (it != types_.end()) ? &it->second : nullptr;
}

} // namespace eshkol::types
```

### 4.2 Integration Points

| Existing File | Changes Required |
|---------------|------------------|
| `CMakeLists.txt` | Add `lib/types/` subdirectory |
| `inc/eshkol/eshkol.h` | Include forward declaration for `TypeExpr` |

---

## 5. Phase 2: Parser Extensions

### 5.1 Token Extensions

**In `lib/frontend/parser.cpp`**, extend TokenType:

```cpp
enum TokenType {
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_QUOTE,
    TOKEN_SYMBOL,
    TOKEN_STRING,
    TOKEN_NUMBER,
    TOKEN_BOOLEAN,
    TOKEN_CHAR,
    TOKEN_VECTOR_START,
    TOKEN_COLON,         // NEW: :
    TOKEN_ARROW,         // NEW: ->
    TOKEN_FORALL,        // NEW: forall (keyword in symbol)
    TOKEN_EOF
};
```

### 5.2 Type Expression Parser

**New functions in `lib/frontend/parser.cpp`:**

```cpp
// Forward declarations
static TypeExpr* parseTypeExpr(SchemeTokenizer& tokenizer);
static TypeExpr* parseTypeDeclaration(SchemeTokenizer& tokenizer);

// Parse (: name type) form
static TypeExpr* parseTypeDeclaration(SchemeTokenizer& tokenizer) {
    // Called after consuming '(' and seeing ':' as first symbol
    Token name_token = tokenizer.nextToken();
    if (name_token.type != TOKEN_SYMBOL) {
        eshkol_error("Expected identifier in type declaration");
        return nullptr;
    }

    TypeExpr* type = parseTypeExpr(tokenizer);
    if (!type) return nullptr;

    Token rparen = tokenizer.nextToken();
    if (rparen.type != TOKEN_RPAREN) {
        eshkol_error("Expected ')' in type declaration");
        delete type;
        return nullptr;
    }

    // Store binding: name_token.value -> type
    return type;
}

// Parse inline type annotation: identifier : type
static TypeExpr* parseInlineTypeAnnotation(SchemeTokenizer& tokenizer,
                                           std::string& out_name) {
    Token name = tokenizer.nextToken();
    if (name.type != TOKEN_SYMBOL) {
        tokenizer.pushBack(name);
        return nullptr;
    }

    Token colon = tokenizer.nextToken();
    if (colon.type != TOKEN_SYMBOL || colon.value != ":") {
        tokenizer.pushBack(colon);
        tokenizer.pushBack(name);
        return nullptr;
    }

    out_name = name.value;
    return parseTypeExpr(tokenizer);
}

// Parse type expression
static TypeExpr* parseTypeExpr(SchemeTokenizer& tokenizer) {
    Token token = tokenizer.nextToken();

    // Simple type name
    if (token.type == TOKEN_SYMBOL) {
        // Check for built-in types
        TypeExpr* expr = new TypeExpr();
        expr->kind = TypeExpr::Kind::Name;
        expr->name = token.value;
        return expr;
    }

    // Compound type
    if (token.type == TOKEN_LPAREN) {
        Token first = tokenizer.nextToken();

        // Arrow type: (-> A B)
        if (first.type == TOKEN_SYMBOL && first.value == "->") {
            return parseArrowType(tokenizer);
        }

        // Forall type: (forall (a b) type)
        if (first.type == TOKEN_SYMBOL && first.value == "forall") {
            return parseForallType(tokenizer);
        }

        // Type application: (list integer)
        TypeExpr* expr = new TypeExpr();
        expr->kind = TypeExpr::Kind::Apply;
        expr->name = first.value;

        while (true) {
            Token next = tokenizer.nextToken();
            if (next.type == TOKEN_RPAREN) break;
            tokenizer.pushBack(next);
            expr->args.push_back(std::unique_ptr<TypeExpr>(parseTypeExpr(tokenizer)));
        }

        return expr;
    }

    eshkol_error("Unexpected token in type expression");
    return nullptr;
}
```

### 5.3 AST Extensions

**In `inc/eshkol/eshkol.h`**, add to `eshkol_ast_t`:

```cpp
struct eshkol_ast {
    eshkol_type_t type;
    // ... existing union ...

    // NEW: Optional type annotation (for explicit typing)
    void* type_annotation;  // Points to TypeExpr* (opaque to C code)
};

// Extend define_op
struct {
    char *name;
    struct eshkol_ast *value;
    uint8_t is_function;
    struct eshkol_ast *parameters;
    uint64_t num_params;
    uint8_t is_variadic;
    char *rest_param;

    // NEW: Type annotations
    void* return_type;      // TypeExpr* for return type
    void** param_types;     // Array of TypeExpr* for parameter types
} define_op;
```

---

## 6. Phase 3: Type Checker

### 6.1 Core Type Checker

**File: `lib/types/type_checker.h`**

```cpp
#ifndef ESHKOL_TYPE_CHECKER_H
#define ESHKOL_TYPE_CHECKER_H

#include "type_system.h"
#include "../inc/eshkol/eshkol.h"
#include <vector>
#include <string>

namespace eshkol::types {

// ============================================================================
// TYPE ERRORS
// ============================================================================

struct TypeError {
    enum class Kind {
        TypeMismatch,
        NotASubtype,
        UndefinedType,
        UndefinedVariable,
        ArityMismatch,
        LinearityViolation,
        UniverseMismatch,
        ProofRequired
    };

    Kind kind;
    std::string message;
    std::string expected;
    std::string actual;
    size_t line = 0;
    size_t column = 0;
};

// ============================================================================
// TYPING CONTEXT (Γ)
// ============================================================================

class Context {
    std::map<std::string, TypeId> term_bindings_;      // Variables
    std::map<std::string, TypeId> type_bindings_;      // Type variables
    std::map<std::string, bool> linear_used_;          // Linear variable usage
    Context* parent_ = nullptr;

public:
    explicit Context(Context* parent = nullptr);

    // Variable binding
    void bindVariable(const std::string& name, TypeId type);
    std::optional<TypeId> lookupVariable(const std::string& name) const;

    // Type variable binding (for polymorphism)
    void bindTypeVariable(const std::string& name, TypeId kind);
    std::optional<TypeId> lookupTypeVariable(const std::string& name) const;

    // Linear tracking
    void markLinearUsed(const std::string& name);
    bool isLinearUsed(const std::string& name) const;
    std::vector<std::string> getUnusedLinearVars() const;

    // Scope management
    Context extend() const;
};

// ============================================================================
// BIDIRECTIONAL TYPE CHECKER
// ============================================================================

class TypeChecker {
    TypeEnvironment& env_;
    std::vector<TypeError> errors_;

public:
    explicit TypeChecker(TypeEnvironment& env);

    // ========== Main Entry Points ==========

    // Check mode: verify expr has expected type
    bool check(const eshkol_ast_t* expr, TypeId expected, Context& ctx);

    // Synth mode: infer type of expr
    std::optional<TypeId> synth(const eshkol_ast_t* expr, Context& ctx);

    // ========== Error Handling ==========

    const std::vector<TypeError>& errors() const { return errors_; }
    bool hasErrors() const { return !errors_.empty(); }
    void clearErrors() { errors_.clear(); }

private:
    // ========== Check Mode Implementations ==========

    bool checkLiteral(const eshkol_ast_t* ast, TypeId expected);
    bool checkVariable(const eshkol_ast_t* ast, TypeId expected, Context& ctx);
    bool checkLambda(const eshkol_ast_t* ast, TypeId expected, Context& ctx);
    bool checkApplication(const eshkol_ast_t* ast, TypeId expected, Context& ctx);
    bool checkLet(const eshkol_ast_t* ast, TypeId expected, Context& ctx);
    bool checkIf(const eshkol_ast_t* ast, TypeId expected, Context& ctx);

    // ========== Synth Mode Implementations ==========

    std::optional<TypeId> synthLiteral(const eshkol_ast_t* ast);
    std::optional<TypeId> synthVariable(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> synthLambda(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> synthApplication(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> synthArithmetic(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> synthListOp(const eshkol_ast_t* ast, Context& ctx);

    // ========== Subtyping ==========

    bool checkSubtype(TypeId actual, TypeId expected, const std::string& context);

    // ========== Error Reporting ==========

    void reportError(TypeError::Kind kind, const std::string& msg,
                     const std::string& expected = "",
                     const std::string& actual = "");
};

} // namespace eshkol::types

#endif // ESHKOL_TYPE_CHECKER_H
```

### 6.2 Integration with Compilation

**In main compilation function** (e.g., `eshkol_generate_llvm_ir`):

```cpp
LLVMModuleRef eshkol_generate_llvm_ir(const eshkol_ast_t* asts, size_t num_asts,
                                       const char* module_name) {
    // NEW: Type checking phase
    eshkol::types::TypeEnvironment type_env;
    eshkol::types::TypeChecker checker(type_env);
    eshkol::types::Context global_ctx;

    // Check all top-level definitions
    for (size_t i = 0; i < num_asts; i++) {
        if (asts[i].type == ESHKOL_OP &&
            asts[i].operation.op == ESHKOL_DEFINE_OP) {
            // Type check definition
            auto type = checker.synth(&asts[i], global_ctx);
            if (!type) {
                // Type error - report and optionally abort
                for (const auto& err : checker.errors()) {
                    eshkol_error("Type error: %s", err.message.c_str());
                }
                // Continue for gradual typing, or return nullptr for strict
            }
        }
    }

    // Existing codegen...
    EshkolLLVMCodeGen codegen(module_name);
    return codegen.generateIR(asts, num_asts);
}
```

---

## 7. Phase 4: Codegen Integration

### 7.1 Extended TypedValue

**In `lib/backend/llvm_codegen.cpp`:**

```cpp
struct TypedValue {
    Value* llvm_value;
    eshkol_value_type_t runtime_type;  // Runtime tag

    // NEW: Static type information from type checker
    std::optional<eshkol::types::TypeId> static_type;

    bool is_exact;
    uint8_t flags;

    // Flag constants
    static constexpr uint8_t FLAG_INDIRECT = 0x01;
    static constexpr uint8_t FLAG_LINEAR   = 0x02;  // NEW: Must consume
    static constexpr uint8_t FLAG_PROOF    = 0x04;  // NEW: Erase at codegen

    // Constructors
    TypedValue() : llvm_value(nullptr), runtime_type(ESHKOL_VALUE_NULL),
                   static_type(std::nullopt), is_exact(true), flags(0) {}

    TypedValue(Value* val, eshkol_value_type_t rt,
               std::optional<eshkol::types::TypeId> st = std::nullopt,
               bool exact = true, uint8_t f = 0)
        : llvm_value(val), runtime_type(rt), static_type(st),
          is_exact(exact), flags(f) {}

    // Query methods
    bool hasStaticType() const { return static_type.has_value(); }
    bool shouldErase() const { return flags & FLAG_PROOF; }
    bool isLinear() const { return flags & FLAG_LINEAR; }
};
```

### 7.2 Proof Erasure

```cpp
// In codegenTypedAST:
TypedValue codegenTypedAST(const eshkol_ast_t* ast) {
    // ...existing code...

    // NEW: Check for proof type (should be erased)
    if (ast->type_annotation) {
        auto* type_expr = static_cast<TypeExpr*>(ast->type_annotation);
        // If this is a proof type, return unit/null
        TypeId resolved = resolveTypeExpr(type_expr);
        if (type_env_.getTypeNode(resolved)->runtime_rep == RuntimeRep::Erased) {
            // Proof erasure: generate nothing
            return TypedValue(
                ConstantInt::get(Type::getInt64Ty(*context), 0),
                ESHKOL_VALUE_NULL,
                resolved,
                true,
                TypedValue::FLAG_PROOF
            );
        }
    }

    // ...existing code...
}
```

### 7.3 Type-Directed Arithmetic

```cpp
TypedValue codegenArithmetic(const eshkol_operations_t* op, const std::string& func_name) {
    TypedValue left = codegenTypedAST(&op->call_op.variables[0]);
    TypedValue right = codegenTypedAST(&op->call_op.variables[1]);

    // NEW: Use static types if available
    TypeId left_type = left.hasStaticType() ? *left.static_type
                                            : runtimeToStaticType(left.runtime_type);
    TypeId right_type = right.hasStaticType() ? *right.static_type
                                              : runtimeToStaticType(right.runtime_type);

    // Determine result type via promotion
    TypeId result_type = type_env_.promoteForArithmetic(left_type, right_type);

    // Generate promotion code if needed
    left = promoteToType(left, result_type);
    right = promoteToType(right, result_type);

    // Generate operation
    Value* result;
    if (type_env_.isSubtype(result_type, BuiltinTypes::Float64)) {
        if (func_name == "+") result = builder->CreateFAdd(left.llvm_value, right.llvm_value);
        else if (func_name == "-") result = builder->CreateFSub(left.llvm_value, right.llvm_value);
        else if (func_name == "*") result = builder->CreateFMul(left.llvm_value, right.llvm_value);
        else if (func_name == "/") result = builder->CreateFDiv(left.llvm_value, right.llvm_value);
    } else {
        if (func_name == "+") result = builder->CreateAdd(left.llvm_value, right.llvm_value);
        else if (func_name == "-") result = builder->CreateSub(left.llvm_value, right.llvm_value);
        else if (func_name == "*") result = builder->CreateMul(left.llvm_value, right.llvm_value);
        else if (func_name == "/") result = builder->CreateSDiv(left.llvm_value, right.llvm_value);
    }

    return TypedValue(
        result,
        staticToRuntimeType(result_type),
        result_type,
        type_env_.isSubtype(result_type, BuiltinTypes::Integer),
        0
    );
}
```

---

## 8. Phase 5: Dependent Types

### 8.1 Compile-Time Value Representation

```cpp
// In lib/types/dependent.h

// Compile-time values for type indices
struct CTValue {
    enum class Kind { Nat, Bool, Expr } kind;

    union {
        uint64_t nat_val;
        bool bool_val;
    };

    // For symbolic expressions
    std::unique_ptr<eshkol_ast_t> expr;

    // Constructors
    static CTValue makeNat(uint64_t n) {
        CTValue v; v.kind = Kind::Nat; v.nat_val = n; return v;
    }

    // Evaluate at compile time (if possible)
    std::optional<uint64_t> tryEvalNat() const;
};

// Dependent type representation
struct DependentType {
    TypeId base;                      // e.g., Buffer
    std::vector<CTValue> indices;     // e.g., [Float64, 100]

    std::string toString() const;
};
```

### 8.2 Dimension Checking

```cpp
// In type_checker.cpp

bool TypeChecker::checkDimensionProof(const CTValue& idx, const CTValue& bound,
                                       const std::string& context) {
    // If both are concrete naturals, check statically
    auto idx_val = idx.tryEvalNat();
    auto bound_val = bound.tryEvalNat();

    if (idx_val && bound_val) {
        if (*idx_val >= *bound_val) {
            reportError(TypeError::Kind::ProofRequired,
                       "Index " + std::to_string(*idx_val) +
                       " out of bounds (< " + std::to_string(*bound_val) + ")",
                       "< " + std::to_string(*bound_val),
                       std::to_string(*idx_val));
            return false;
        }
        return true;  // Proof succeeds
    }

    // For symbolic indices, require explicit proof
    reportError(TypeError::Kind::ProofRequired,
               "Cannot statically verify index bound: " + context);
    return false;
}
```

---

## 9. Phase 6: Linear Types

### 9.1 Linear Context Tracking

```cpp
// In lib/types/linear_checker.h

class LinearContext {
    // Track variable usage: name -> usage count
    std::map<std::string, size_t> usage_counts_;
    // Track which variables are linear
    std::set<std::string> linear_vars_;

public:
    void declareLinear(const std::string& name);
    void use(const std::string& name);
    void consume(const std::string& name);  // Mark as fully used

    // Verification
    bool checkAllConsumed() const;
    std::vector<std::string> getUnconsumed() const;
    std::vector<std::string> getOverused() const;
};
```

### 9.2 Linear Type Checking Rules

```cpp
// In type_checker.cpp

bool TypeChecker::checkLinearVariable(const eshkol_ast_t* ast, Context& ctx) {
    std::string name = ast->variable.id;
    auto type = ctx.lookupVariable(name);

    if (!type) {
        reportError(TypeError::Kind::UndefinedVariable, name);
        return false;
    }

    const TypeNode* node = env_.getTypeNode(*type);
    if (node && (node->id.flags & TypeFlags::LINEAR)) {
        // Check not already used
        if (ctx.isLinearUsed(name)) {
            reportError(TypeError::Kind::LinearityViolation,
                       "Linear variable '" + name + "' used more than once");
            return false;
        }
        ctx.markLinearUsed(name);
    }

    return true;
}

bool TypeChecker::checkLinearLet(const eshkol_ast_t* ast, Context& ctx) {
    // After checking let body, verify all linear bindings consumed
    auto unused = ctx.getUnusedLinearVars();
    for (const auto& name : unused) {
        reportError(TypeError::Kind::LinearityViolation,
                   "Linear variable '" + name + "' not consumed");
    }
    return unused.empty();
}
```

---

## 10. File-by-File Changes

### 10.1 New Files

| File | Lines (est.) | Description |
|------|--------------|-------------|
| `lib/types/type_system.h` | ~250 | Type representation, universe levels |
| `lib/types/type_system.cpp` | ~300 | Type environment, subtyping |
| `lib/types/type_checker.h` | ~150 | Type checker interface |
| `lib/types/type_checker.cpp` | ~500 | Bidirectional type checking |
| `lib/types/linear_checker.h` | ~80 | Linear type tracking |
| `lib/types/linear_checker.cpp` | ~200 | Linear usage verification |
| `lib/types/dependent.h` | ~100 | Compile-time values |
| `lib/types/dependent.cpp` | ~150 | Dimension checking |

### 10.2 Modified Files

| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Add `type_annotation` to AST, extend `define_op` |
| `lib/frontend/parser.cpp` | Add type expression parsing, inline annotations |
| `lib/backend/llvm_codegen.cpp` | Extend `TypedValue`, add proof erasure |
| `inc/eshkol/llvm_backend.h` | Add type environment parameter |
| `CMakeLists.txt` | Add `lib/types/` subdirectory |

### 10.3 Unchanged Files

| File | Reason |
|------|--------|
| `lib/core/arena_memory.h/cpp` | Runtime unchanged, proof erasure |
| `lib/core/printer.cpp` | Display unchanged |
| `lib/core/ast.cpp` | Minimal changes |

---

## 11. Testing Strategy

### 11.1 Unit Tests

```scheme
;; tests/types/type_inference_test.esk

;; Test 1: Integer literals
(: x integer)
(define x 42)
(assert-type x integer)

;; Test 2: Float literals
(: y float64)
(define y 3.14)
(assert-type y float64)

;; Test 3: Function types
(: square (-> integer integer))
(define (square n : integer) : integer
  (* n n))
(assert-type (square 5) integer)

;; Test 4: Polymorphic functions
(: id (forall (a) (-> a a)))
(define (id x) x)
(assert-type (id 42) integer)
(assert-type (id "hello") string)

;; Test 5: List types
(: nums (list number))
(define nums (list 1 2.5 3))
(assert-type nums (list number))

;; Test 6: Subtyping
(: process (-> number float64))
(define (process x : number) : float64
  (* x 2.0))
(assert-type (process 42) float64)  ; integer <: number
```

### 11.2 Error Tests

```scheme
;; tests/types/type_errors_test.esk

;; Test: Type mismatch
(: x integer)
(define x "hello")  ; ERROR: expected integer, got string

;; Test: Arity mismatch
(: f (-> integer integer))
(define (f x y) (+ x y))  ; ERROR: expected 1 param, got 2

;; Test: Linear violation
(: handle (Linear (Handle Window)))
(define handle (window-create "Test" 800 600))
(use handle)
(use handle)  ; ERROR: linear value used twice
```

### 11.3 Integration Tests

```scheme
;; tests/types/dependent_test.esk

;; Test: Buffer dimensions
(: buf (Buffer float64 100))
(define buf (make-buffer 100))

(buffer-ref buf 50)   ; OK: 50 < 100
(buffer-ref buf 100)  ; ERROR: 100 >= 100

;; Test: Vector operations
(: v1 (vector float64 3))
(: v2 (vector float64 3))
(define v1 (vector 1.0 2.0 3.0))
(define v2 (vector 4.0 5.0 6.0))
(vec-dot v1 v2)  ; OK: same dimensions

(: v3 (vector float64 4))
(define v3 (vector 1.0 2.0 3.0 4.0))
(vec-dot v1 v3)  ; ERROR: dimension mismatch (3 vs 4)
```

---

## Summary

This implementation plan provides a complete roadmap for adding HoTT-based typing to Eshkol:

1. **Phase 1**: Foundation - Type infrastructure with universes and supertypes
2. **Phase 2**: Syntax - Parser extensions for type annotations
3. **Phase 3**: Core - Bidirectional type checker
4. **Phase 4**: Integration - Codegen with proof erasure
5. **Phase 5**: Advanced - Dependent types for dimension safety
6. **Phase 6**: Resources - Linear types for resource management

The implementation maintains backward compatibility (untyped code continues to work) while enabling gradual adoption of explicit typing.
