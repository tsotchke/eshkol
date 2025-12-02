# Extending Eshkol's Type System for HoTT with Supertypes

## Technical Specification

**Version:** 1.0
**Date:** 2025-12-01
**Status:** Design Specification

---

## Table of Contents

1. [Current Type System Analysis](#1-current-type-system-analysis)
2. [HoTT Type System Design](#2-hott-type-system-design)
3. [Supertype Hierarchy](#3-supertype-hierarchy)
4. [Universe Levels](#4-universe-levels)
5. [Type Representation](#5-type-representation)
6. [Implementation Strategy](#6-implementation-strategy)
7. [Parser Extensions](#7-parser-extensions)
8. [Codegen Extensions](#8-codegen-extensions)
9. [Runtime Representation](#9-runtime-representation)
10. [Migration Path](#10-migration-path)

---

## 1. Current Type System Analysis

### 1.1 Existing Architecture

Eshkol currently uses a **runtime-tagged value system** with no explicit type annotations:

```c
// Parse-time types (AST level)
typedef enum {
    ESHKOL_INVALID, ESHKOL_UNTYPED, ESHKOL_UINT8, ESHKOL_UINT16,
    ESHKOL_UINT32, ESHKOL_UINT64, ESHKOL_INT8, ESHKOL_INT16,
    ESHKOL_INT32, ESHKOL_INT64, ESHKOL_DOUBLE, ESHKOL_STRING,
    ESHKOL_FUNC, ESHKOL_VAR, ESHKOL_OP, ESHKOL_CONS,
    ESHKOL_NULL, ESHKOL_TENSOR, ESHKOL_CHAR
} eshkol_type_t;

// Runtime value types (tagged value level)
typedef enum {
    ESHKOL_VALUE_NULL = 0,
    ESHKOL_VALUE_INT64 = 1,
    ESHKOL_VALUE_DOUBLE = 2,
    ESHKOL_VALUE_CONS_PTR = 3,
    ESHKOL_VALUE_DUAL_NUMBER = 4,
    ESHKOL_VALUE_AD_NODE_PTR = 5,
    ESHKOL_VALUE_TENSOR_PTR = 6,
    ESHKOL_VALUE_LAMBDA_SEXPR = 7,
    ESHKOL_VALUE_STRING_PTR = 8,
    ESHKOL_VALUE_CHAR = 9,
    ESHKOL_VALUE_VECTOR_PTR = 10,
    ESHKOL_VALUE_SYMBOL = 11,
    ESHKOL_VALUE_CLOSURE_PTR = 12,
    ESHKOL_VALUE_MAX = 15
} eshkol_value_type_t;
```

### 1.2 Type Inference Flow

```
Source Code → Parser → AST (eshkol_type_t) → Codegen (TypedValue) → Runtime (eshkol_tagged_value_t)
```

Types are currently:
- **Inferred from literals**: `42` → INT64, `3.14` → DOUBLE, `"hello"` → STRING
- **Propagated through operations**: arithmetic, list operations, function calls
- **Stored at runtime**: 16-byte tagged values with 4-bit type tag

### 1.3 Limitations

1. **Flat hierarchy**: No subtyping relationships
2. **No type annotations**: Cannot express programmer intent
3. **Runtime-only checking**: No compile-time type errors
4. **No dependent types**: Cannot express dimension constraints
5. **No proof system**: Cannot verify type-level properties

---

## 2. HoTT Type System Design

### 2.1 Core Principles

The HoTT extension follows these principles:

1. **Types are first-class**: Types themselves have types (universes)
2. **Hierarchy is explicit**: Subtype relationships are defined, not inferred
3. **Proofs erase**: Type-level information vanishes at runtime
4. **Backward compatible**: Existing code continues to work

### 2.2 Type Judgment Forms

We extend Eshkol with these judgment forms:

```
Γ ⊢ A : 𝒰ₙ          A is a type at universe level n
Γ ⊢ a : A           Term a has type A
Γ ⊢ A <: B          A is a subtype of B
Γ ⊢ A ≡ B : 𝒰ₙ      A and B are definitionally equal types
```

### 2.3 Type Formation Rules

```
──────────────────── (Universe)
    𝒰ₙ : 𝒰ₙ₊₁

Γ ⊢ A : 𝒰ₙ    Γ ⊢ B : 𝒰ₙ
────────────────────────── (Function Type)
    Γ ⊢ A → B : 𝒰ₙ

Γ ⊢ A : 𝒰ₙ    Γ, x : A ⊢ B : 𝒰ₙ
────────────────────────────────── (Dependent Function)
       Γ ⊢ (x : A) → B : 𝒰ₙ

Γ ⊢ A : 𝒰ₙ    Γ, x : A ⊢ B : 𝒰ₙ
────────────────────────────────── (Dependent Pair)
       Γ ⊢ Σ(x : A).B : 𝒰ₙ
```

---

## 3. Supertype Hierarchy

### 3.1 Base Supertype Design

The key insight is organizing types into a hierarchy with **supertypes** that subsume multiple concrete types:

```
                         𝒰₂ (Universe of Propositions)
                            │
                            ▼
                         𝒰₁ (Universe of Types)
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
         Value          Resource        Proposition
            │               │               │
     ┌──────┼──────┐       │           ┌───┴───┐
     │      │      │       │           │       │
     ▼      ▼      ▼       ▼           ▼       ▼
  Number  Text  Collection Handle    Proof   Witness
     │      │      │       │
  ┌──┴──┐   │   ┌──┴──┐    │
  │     │   │   │     │    │
  ▼     ▼   ▼   ▼     ▼    ▼
Int64 Float64 String List Vector Window
```

### 3.2 Supertype Definitions

Using Eshkol's type declaration syntax (see ESHKOL_COMPLETE_LANGUAGE_SPECIFICATION.md):

```scheme
;; Value supertype - all runtime values
;; Separate type declaration form: (: name type)
(define-type Value
  (supertype-of Number Text Collection Procedure)
  (universe Type₁))

;; Number supertype - Scheme numeric tower
(define-type Number
  (subtype-of Value)
  (supertype-of Integer Rational Real Complex)
  (universe Type₀)
  (operations + - * / < > = <= >=))

;; Integer type
(define-type Integer
  (subtype-of Number)
  (supertype-of Natural Positive Zero)
  (universe Type₀)
  (exact #t)
  (runtime-rep int64))

;; Real numbers (inexact floating-point)
(define-type Real
  (subtype-of Number)
  (universe Type₀)
  (exact #f)
  (runtime-rep float64))

;; Collection supertype - parameterized by element type
(define-type Collection
  (subtype-of Value)
  (supertype-of (List a) (Vector a))
  (universe Type₁)
  (operations length empty? map fold filter))

;; List type family - using Eshkol parametric type syntax
;; Type annotation: (: my-list (list integer))
(define-type-family (List a)
  (subtype-of Collection)
  (universe Type₁)
  (element-type a)
  (runtime-rep cons-chain))
```

### 3.2.1 Type Declarations in Eshkol Syntax

Following the established Eshkol type annotation conventions:

```scheme
;; Separate type declarations (preferred for complex types)
(: x integer)
(: nums (list number))
(: transform (-> (list a) (list b)))

;; Inline type annotations in function definitions
(define (add-integers x : integer y : integer) : integer
  (+ x y))

;; Polymorphic functions using forall
(: id (forall (a) (-> a a)))
(define (id x) x)

;; Type class constraints
(: sum (forall (a) (-> (vector a) a) (where (Numeric a))))
```

### 3.3 Subtyping Rules

```
─────────────────────── (Reflexivity)
       A <: A

A <: B    B <: C
──────────────────── (Transitivity)
      A <: C

─────────────────────── (Integer-Number)
    Integer <: Number

─────────────────────── (Real-Number)
     Real <: Number

     A <: A'
─────────────────────── (List Covariance)
  List A <: List A'

────────────────────────────────── (Function Contravariance)
A' <: A    B <: B'
─────────────────────────────────
  (A → B) <: (A' → B')
```

### 3.4 Type Promotion

Arithmetic operations use the numeric tower for automatic promotion:

```scheme
;; Type promotion rules
(define-promotion-rule
  :from Integer :to Real
  :when (one-operand-is Real)
  :method exact->inexact)

;; Example: integer + real → real
(+ 1 2.5)  ; Result type: Real
```

---

## 4. Universe Levels

### 4.1 Universe Hierarchy

```
𝒰₀ : 𝒰₁ : 𝒰₂ : 𝒰₃ : ...

𝒰₀  =  { Int64, Float64, Char, String, Bool }           ; Ground types
𝒰₁  =  { List, Vector, →, ×, Handle, Buffer }           ; Type constructors
𝒰₂  =  { Proof, Witness, Eq, <:, Bounded }              ; Propositions
```

### 4.2 Universe Polymorphism

Using Eshkol's type annotation syntax:

```scheme
;; Identity function polymorphic over universes
;; Type declaration using (: name type) form
(: id (forall (U) (forall (A : U) (-> A A))))
(define (id x) x)

;; Usage at different universe levels
(id 42)                    ; A instantiated at Type₀ (integer)
(id '(1 2 3))              ; A instantiated at Type₁ (list integer)
```

### 4.3 Universe Constraints

```scheme
;; Type family constrained to ground types (Type₀)
;; Buffer with element type A and length n (dependent type)
(: Buffer (forall (A : Type₀) (forall (n : natural) Type₁)))

(define-type-family (Buffer A n)
  (universe Type₁)
  (element-type A)
  (length n)
  (constraint (Ground A)))  ; A must be ground type

;; Using dependent types for dimension safety
(: make-buffer (forall (n : natural) (-> (Buffer float64 n))))
(: buffer-ref (forall (A : Type₀) (forall (n : natural)
                (-> (Buffer A n) (refine natural (lambda (i) (< i n))) A))))
```

---

## 5. Type Representation

### 5.1 Compile-Time Type Structure

```cpp
// Extended type representation for compile-time analysis
enum class UniverseLevel : uint8_t {
    U0 = 0,  // Ground types
    U1 = 1,  // Type constructors
    U2 = 2,  // Propositions
    UOmega = 255  // Universe polymorphic
};

struct TypeId {
    uint16_t id;           // Unique type identifier
    UniverseLevel level;   // Universe membership
    uint8_t flags;         // Type flags (exact, linear, etc.)
};

// Type node in the type graph
struct TypeNode {
    TypeId id;
    std::string name;

    // Supertype chain
    TypeId supertype;
    std::vector<TypeId> subtypes;

    // For parameterized types
    std::vector<TypeId> parameters;

    // Runtime representation
    RuntimeRep runtime_rep;

    // Operations defined on this type
    std::vector<OperationId> operations;
};

// Complete type environment
class TypeEnvironment {
    std::map<TypeId, TypeNode> types;
    std::map<std::string, TypeId> name_to_id;

public:
    // Subtype checking
    bool isSubtype(TypeId a, TypeId b) const;

    // Find common supertype
    TypeId leastCommonSupertype(TypeId a, TypeId b) const;

    // Type promotion for arithmetic
    TypeId promoteForArithmetic(TypeId a, TypeId b) const;
};
```

### 5.2 Type IDs for Eshkol

```cpp
// Extended type ID system (compile-time)
namespace TypeIds {
    // Universe markers
    constexpr TypeId U0{0, UniverseLevel::U0, 0};
    constexpr TypeId U1{1, UniverseLevel::U1, 0};
    constexpr TypeId U2{2, UniverseLevel::U2, 0};

    // Ground types (𝒰₀)
    constexpr TypeId Value{10, UniverseLevel::U0, 0};
    constexpr TypeId Number{11, UniverseLevel::U0, 0};
    constexpr TypeId Integer{12, UniverseLevel::U0, TYPE_EXACT};
    constexpr TypeId Int64{13, UniverseLevel::U0, TYPE_EXACT};
    constexpr TypeId Natural{14, UniverseLevel::U0, TYPE_EXACT};
    constexpr TypeId Real{15, UniverseLevel::U0, 0};
    constexpr TypeId Float64{16, UniverseLevel::U0, 0};
    constexpr TypeId Text{17, UniverseLevel::U0, 0};
    constexpr TypeId String{18, UniverseLevel::U0, 0};
    constexpr TypeId Char{19, UniverseLevel::U0, 0};
    constexpr TypeId Bool{20, UniverseLevel::U0, 0};

    // Type constructors (𝒰₁)
    constexpr TypeId Collection{100, UniverseLevel::U1, 0};
    constexpr TypeId List{101, UniverseLevel::U1, 0};
    constexpr TypeId Vector{102, UniverseLevel::U1, 0};
    constexpr TypeId Function{103, UniverseLevel::U1, 0};
    constexpr TypeId Pair{104, UniverseLevel::U1, 0};
    constexpr TypeId Handle{105, UniverseLevel::U1, TYPE_LINEAR};
    constexpr TypeId Buffer{106, UniverseLevel::U1, 0};
    constexpr TypeId Stream{107, UniverseLevel::U1, TYPE_LINEAR};

    // Propositions (𝒰₂)
    constexpr TypeId Eq{200, UniverseLevel::U2, TYPE_PROOF};
    constexpr TypeId Subtype{201, UniverseLevel::U2, TYPE_PROOF};
    constexpr TypeId Bounded{202, UniverseLevel::U2, TYPE_PROOF};
}
```

### 5.3 Runtime Type Tags (Extended)

```c
// Extended runtime type tags (still fits in 4 bits for basic types)
// Supertypes are represented by their most specific subtype at runtime
typedef enum {
    ESHKOL_RT_NULL = 0,
    ESHKOL_RT_INT64 = 1,
    ESHKOL_RT_FLOAT64 = 2,
    ESHKOL_RT_CONS = 3,
    ESHKOL_RT_STRING = 4,
    ESHKOL_RT_CHAR = 5,
    ESHKOL_RT_BOOL = 6,
    ESHKOL_RT_VECTOR = 7,
    ESHKOL_RT_CLOSURE = 8,
    ESHKOL_RT_HANDLE = 9,
    ESHKOL_RT_BUFFER = 10,
    ESHKOL_RT_TENSOR = 11,
    ESHKOL_RT_DUAL = 12,
    ESHKOL_RT_SYMBOL = 13,
    ESHKOL_RT_RESERVED1 = 14,
    ESHKOL_RT_RESERVED2 = 15
} eshkol_runtime_type_t;

// Extended flags byte for supertype information
// Lower 4 bits: exactness, linearity
// Upper 4 bits: supertype category
#define ESHKOL_FLAG_EXACT      0x01
#define ESHKOL_FLAG_LINEAR     0x02
#define ESHKOL_FLAG_BORROWED   0x04
#define ESHKOL_FLAG_OWNED      0x08

#define ESHKOL_SUPER_VALUE     0x00  // Value supertype
#define ESHKOL_SUPER_NUMBER    0x10  // Number supertype
#define ESHKOL_SUPER_TEXT      0x20  // Text supertype
#define ESHKOL_SUPER_COLLECT   0x30  // Collection supertype
#define ESHKOL_SUPER_RESOURCE  0x40  // Resource supertype
```

---

## 6. Implementation Strategy

### 6.1 Phase Overview

```
Phase 1: Type Infrastructure
├── TypeId system
├── TypeEnvironment class
├── Supertype relationships
└── Basic subtype checking

Phase 2: Parser Extensions
├── Type annotation syntax
├── Type expression parsing
├── Constraint parsing
└── AST type field extensions

Phase 3: Type Checker
├── Type inference engine
├── Subtype verification
├── Error reporting
└── Proof generation

Phase 4: Codegen Extensions
├── TypedValue enhancements
├── Promotion code generation
├── Type-directed dispatch
└── Proof erasure

Phase 5: Integration
├── Standard library types
├── Backward compatibility
├── Performance optimization
└── Documentation
```

### 6.2 Phase 1: Type Infrastructure (C++)

**File: `lib/types/type_system.h`**

```cpp
#pragma once

#include <cstdint>
#include <map>
#include <vector>
#include <string>
#include <optional>

namespace eshkol::types {

// Forward declarations
struct TypeId;
struct TypeNode;
class TypeEnvironment;

// Universe levels
enum class Universe : uint8_t {
    U0 = 0,      // Ground types
    U1 = 1,      // Type constructors
    U2 = 2,      // Propositions
    UOmega = 255 // Universe polymorphic
};

// Type flags
enum TypeFlags : uint8_t {
    NONE = 0,
    EXACT = 1 << 0,
    LINEAR = 1 << 1,
    PROOF = 1 << 2,
    ABSTRACT = 1 << 3
};

// Unique type identifier
struct TypeId {
    uint16_t id;
    Universe level;
    uint8_t flags;

    bool operator==(const TypeId& other) const {
        return id == other.id;
    }
    bool operator<(const TypeId& other) const {
        return id < other.id;
    }

    bool isExact() const { return flags & TypeFlags::EXACT; }
    bool isLinear() const { return flags & TypeFlags::LINEAR; }
    bool isProof() const { return flags & TypeFlags::PROOF; }
};

// Runtime representation mapping
enum class RuntimeRep : uint8_t {
    Int64,
    Float64,
    Pointer,
    TaggedValue,
    Erased  // For proofs
};

// Type node in the type graph
struct TypeNode {
    TypeId id;
    std::string name;

    // Type hierarchy
    std::optional<TypeId> supertype;
    std::vector<TypeId> subtypes;

    // For parameterized types (List A, Buffer A n)
    std::vector<TypeId> parameters;

    // Runtime mapping
    RuntimeRep runtime_rep;

    // Source of definition
    enum class Origin { Builtin, UserDefined };
    Origin origin;
};

// Type environment maintaining all type information
class TypeEnvironment {
private:
    std::map<TypeId, TypeNode> types_;
    std::map<std::string, TypeId> name_to_id_;
    uint16_t next_id_ = 1000;  // User types start at 1000

    // Subtype cache for performance
    mutable std::map<std::pair<TypeId, TypeId>, bool> subtype_cache_;

public:
    TypeEnvironment();

    // Registration
    TypeId registerType(const std::string& name, Universe level,
                        uint8_t flags, std::optional<TypeId> supertype);

    // Lookup
    std::optional<TypeId> lookupType(const std::string& name) const;
    const TypeNode* getTypeNode(TypeId id) const;

    // Hierarchy queries
    bool isSubtype(TypeId sub, TypeId super) const;
    std::optional<TypeId> leastCommonSupertype(TypeId a, TypeId b) const;
    std::vector<TypeId> getSupertypeChain(TypeId type) const;

    // Arithmetic promotion
    TypeId promoteForArithmetic(TypeId a, TypeId b) const;

    // Type application (for parameterized types)
    TypeId applyType(TypeId constructor, const std::vector<TypeId>& args);

private:
    void initializeBuiltinTypes();
    bool isSubtypeUncached(TypeId sub, TypeId super) const;
};

// Built-in type IDs (compile-time constants)
namespace BuiltinTypes {
    // Universes
    inline constexpr TypeId U0{0, Universe::U0, 0};
    inline constexpr TypeId U1{1, Universe::U1, 0};
    inline constexpr TypeId U2{2, Universe::U2, 0};

    // Ground types
    inline constexpr TypeId Value{10, Universe::U0, 0};
    inline constexpr TypeId Number{11, Universe::U0, 0};
    inline constexpr TypeId Integer{12, Universe::U0, TypeFlags::EXACT};
    inline constexpr TypeId Int64{13, Universe::U0, TypeFlags::EXACT};
    inline constexpr TypeId Real{14, Universe::U0, 0};
    inline constexpr TypeId Float64{15, Universe::U0, 0};
    inline constexpr TypeId Text{16, Universe::U0, 0};
    inline constexpr TypeId String{17, Universe::U0, 0};
    inline constexpr TypeId Char{18, Universe::U0, 0};
    inline constexpr TypeId Bool{19, Universe::U0, 0};
    inline constexpr TypeId Null{20, Universe::U0, 0};

    // Type constructors
    inline constexpr TypeId List{100, Universe::U1, 0};
    inline constexpr TypeId Vector{101, Universe::U1, 0};
    inline constexpr TypeId Function{102, Universe::U1, 0};
    inline constexpr TypeId Pair{103, Universe::U1, 0};
    inline constexpr TypeId Handle{104, Universe::U1, TypeFlags::LINEAR};
    inline constexpr TypeId Buffer{105, Universe::U1, 0};
}

} // namespace eshkol::types
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

    // Register universes
    types_[U0] = {"U0", "𝒰₀", std::nullopt, {}, {}, RuntimeRep::Erased, TypeNode::Origin::Builtin};
    types_[U1] = {"U1", "𝒰₁", std::nullopt, {}, {}, RuntimeRep::Erased, TypeNode::Origin::Builtin};
    types_[U2] = {"U2", "𝒰₂", std::nullopt, {}, {}, RuntimeRep::Erased, TypeNode::Origin::Builtin};

    // Register Value hierarchy
    types_[Value] = {Value, "Value", std::nullopt, {Number, Text, Bool, Null},
                     {}, RuntimeRep::TaggedValue, TypeNode::Origin::Builtin};

    // Number hierarchy
    types_[Number] = {Number, "Number", Value, {Integer, Real},
                      {}, RuntimeRep::TaggedValue, TypeNode::Origin::Builtin};
    types_[Integer] = {Integer, "Integer", Number, {Int64},
                       {}, RuntimeRep::Int64, TypeNode::Origin::Builtin};
    types_[Int64] = {Int64, "Int64", Integer, {},
                     {}, RuntimeRep::Int64, TypeNode::Origin::Builtin};
    types_[Real] = {Real, "Real", Number, {Float64},
                    {}, RuntimeRep::Float64, TypeNode::Origin::Builtin};
    types_[Float64] = {Float64, "Float64", Real, {},
                       {}, RuntimeRep::Float64, TypeNode::Origin::Builtin};

    // Text hierarchy
    types_[Text] = {Text, "Text", Value, {String, Char},
                    {}, RuntimeRep::Pointer, TypeNode::Origin::Builtin};
    types_[String] = {String, "String", Text, {},
                      {}, RuntimeRep::Pointer, TypeNode::Origin::Builtin};
    types_[Char] = {Char, "Char", Text, {},
                    {}, RuntimeRep::Int64, TypeNode::Origin::Builtin};

    // Other ground types
    types_[Bool] = {Bool, "Bool", Value, {},
                    {}, RuntimeRep::Int64, TypeNode::Origin::Builtin};
    types_[Null] = {Null, "Null", Value, {},
                    {}, RuntimeRep::Int64, TypeNode::Origin::Builtin};

    // Type constructors
    types_[List] = {List, "List", std::nullopt, {},
                    {BuiltinTypes::Value}, RuntimeRep::Pointer, TypeNode::Origin::Builtin};
    types_[Vector] = {Vector, "Vector", std::nullopt, {},
                      {BuiltinTypes::Value}, RuntimeRep::Pointer, TypeNode::Origin::Builtin};
    types_[Handle] = {Handle, "Handle", std::nullopt, {},
                      {}, RuntimeRep::Pointer, TypeNode::Origin::Builtin};
    types_[Buffer] = {Buffer, "Buffer", std::nullopt, {},
                      {BuiltinTypes::Value}, RuntimeRep::Pointer, TypeNode::Origin::Builtin};

    // Build name lookup
    for (const auto& [id, node] : types_) {
        name_to_id_[node.name] = id;
    }
}

bool TypeEnvironment::isSubtype(TypeId sub, TypeId super) const {
    // Check cache first
    auto key = std::make_pair(sub, super);
    auto it = subtype_cache_.find(key);
    if (it != subtype_cache_.end()) {
        return it->second;
    }

    bool result = isSubtypeUncached(sub, super);
    subtype_cache_[key] = result;
    return result;
}

bool TypeEnvironment::isSubtypeUncached(TypeId sub, TypeId super) const {
    // Reflexivity
    if (sub == super) return true;

    // Walk supertype chain
    auto node = getTypeNode(sub);
    if (!node) return false;

    while (node->supertype.has_value()) {
        if (node->supertype.value() == super) return true;
        node = getTypeNode(node->supertype.value());
        if (!node) return false;
    }

    return false;
}

std::optional<TypeId> TypeEnvironment::leastCommonSupertype(TypeId a, TypeId b) const {
    // Get supertype chains
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

std::vector<TypeId> TypeEnvironment::getSupertypeChain(TypeId type) const {
    std::vector<TypeId> chain;
    chain.push_back(type);

    auto node = getTypeNode(type);
    while (node && node->supertype.has_value()) {
        chain.push_back(node->supertype.value());
        node = getTypeNode(node->supertype.value());
    }

    return chain;
}

TypeId TypeEnvironment::promoteForArithmetic(TypeId a, TypeId b) const {
    using namespace BuiltinTypes;

    // If both are the same, return that type
    if (a == b) return a;

    // Integer + Real -> Real
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

    // Fallback to common supertype
    return leastCommonSupertype(a, b).value_or(Value);
}

const TypeNode* TypeEnvironment::getTypeNode(TypeId id) const {
    auto it = types_.find(id);
    return (it != types_.end()) ? &it->second : nullptr;
}

std::optional<TypeId> TypeEnvironment::lookupType(const std::string& name) const {
    auto it = name_to_id_.find(name);
    return (it != name_to_id_.end()) ? std::optional{it->second} : std::nullopt;
}

} // namespace eshkol::types
```

### 6.3 Phase 2: Parser Extensions

**Extended AST with type annotations:**

```cpp
// Type expression AST node
struct TypeExpr {
    enum class Kind {
        Name,       // Int64, String, etc.
        Apply,      // List Int64, Buffer Float64 100
        Arrow,      // A → B
        Product,    // A × B
        Dependent,  // (x : A) → B(x)
        Universe    // 𝒰₀, 𝒰₁
    };

    Kind kind;
    std::string name;                    // For Name, Universe
    std::vector<TypeExpr> arguments;     // For Apply
    std::unique_ptr<TypeExpr> left;      // For Arrow, Product
    std::unique_ptr<TypeExpr> right;     // For Arrow, Product
    std::string binder_name;             // For Dependent
};

// Extended define operation with type annotation
struct {
    char *name;
    struct eshkol_ast *value;
    uint8_t is_function;
    struct eshkol_ast *parameters;
    uint64_t num_params;

    // NEW: Type annotation
    TypeExpr* type_annotation;      // Optional explicit type
    TypeExpr* return_type;          // For functions
    TypeExpr** param_types;         // For function parameters
} define_op;
```

**Type annotation syntax** (following ESHKOL_COMPLETE_LANGUAGE_SPECIFICATION.md):

```scheme
;; Separate type declaration (preferred for complex types)
(: x integer)
(define x 42)

;; Inline type annotations in function definitions
(define (square x : integer) : integer
  (* x x))

;; Separate declaration for complex function types
(: transform (-> (list a) (list b)))

;; Explicit type for complex expressions
(: result (list number))
(define result (list 1 2.5 3 4.5))

;; Generic function with type parameter (forall syntax)
(: id (forall (a) (-> a a)))
(define (id x) x)

;; Dependent type (buffer with length)
(: make-buffer (forall (n : natural) (-> (Buffer float64 n))))
(define (make-buffer n) (buffer-allocate n))
```

### 6.4 Phase 3: Type Checker

**File: `lib/types/type_checker.h`**

```cpp
#pragma once

#include "type_system.h"
#include "../frontend/ast.h"
#include <vector>
#include <string>

namespace eshkol::types {

// Type checking error
struct TypeError {
    enum class Kind {
        TypeMismatch,
        NotASubtype,
        UndefinedType,
        UndefinedVariable,
        ArityMismatch,
        LinearityViolation,
        UniverseMismatch
    };

    Kind kind;
    std::string message;
    size_t line;
    size_t column;
};

// Typing context (Γ)
class Context {
    std::map<std::string, TypeId> variables_;
    std::map<std::string, TypeId> type_variables_;
    Context* parent_;

public:
    explicit Context(Context* parent = nullptr);

    void bindVariable(const std::string& name, TypeId type);
    void bindTypeVariable(const std::string& name, TypeId type);
    std::optional<TypeId> lookupVariable(const std::string& name) const;
    std::optional<TypeId> lookupTypeVariable(const std::string& name) const;

    Context extend() const;
};

// Type checker
class TypeChecker {
    TypeEnvironment& env_;
    std::vector<TypeError> errors_;

public:
    explicit TypeChecker(TypeEnvironment& env);

    // Main entry point
    std::optional<TypeId> check(const eshkol_ast_t* ast, Context& ctx);

    // Get accumulated errors
    const std::vector<TypeError>& errors() const { return errors_; }
    bool hasErrors() const { return !errors_.empty(); }
    void clearErrors() { errors_.clear(); }

private:
    // Type inference for different AST nodes
    std::optional<TypeId> inferLiteral(const eshkol_ast_t* ast);
    std::optional<TypeId> inferVariable(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> inferApplication(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> inferLambda(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> inferLet(const eshkol_ast_t* ast, Context& ctx);
    std::optional<TypeId> inferIf(const eshkol_ast_t* ast, Context& ctx);

    // Subtype checking with error reporting
    bool checkSubtype(TypeId actual, TypeId expected, const std::string& context);

    // Arithmetic type promotion
    TypeId checkArithmetic(TypeId left, TypeId right, const std::string& op);

    // Error helpers
    void reportError(TypeError::Kind kind, const std::string& msg,
                     size_t line = 0, size_t column = 0);
};

} // namespace eshkol::types
```

---

## 7. Parser Extensions

### 7.1 Alignment with Eshkol Language Specification

The type annotation syntax follows the established Eshkol convention from
ESHKOL_COMPLETE_LANGUAGE_SPECIFICATION.md:

**Separate Type Declaration Form:**
```scheme
(: identifier type)
```

**Inline Type Annotation (in function definitions):**
```scheme
(define (function-name param1 : type1 param2 : type2) : return-type
  body)
```

### 7.2 Type Expression Grammar

```bnf
type_expr     ::= type_name
                | type_apply
                | type_arrow
                | type_product
                | type_forall
                | '(' type_expr ')'

type_name     ::= IDENTIFIER                              ; integer, float64, string

type_apply    ::= '(' type_name type_expr+ ')'            ; (list integer), (vector float64 3)

type_arrow    ::= '(' '->' type_expr+ ')'                 ; (-> integer integer)

type_product  ::= '(' '×' type_expr type_expr ')'         ; (× integer string)

type_forall   ::= '(' 'forall' '(' IDENTIFIER+ ')' type_expr ')'  ; (forall (a) (-> a a))

type_decl     ::= '(' ':' IDENTIFIER type_expr ')'        ; (: x integer)

inline_type   ::= IDENTIFIER ':' type_expr                ; x : integer (in function params)
```

### 7.3 Parser Modifications

```cpp
// In parser.cpp - new token types
enum TokenType {
    // ... existing tokens ...
    TOKEN_COLON,       // :
    TOKEN_ARROW,       // ->
    TOKEN_FORALL,      // forall keyword
    TOKEN_TIMES,       // × (product type)
};

// Parse separate type declaration: (: identifier type)
TypeExpr* parseTypeDeclaration(SchemeTokenizer& tokenizer) {
    // Already consumed '(' and ':' tokens
    Token name = tokenizer.nextToken();
    if (name.type != TOKEN_SYMBOL) {
        eshkol_error("Expected identifier in type declaration");
        return nullptr;
    }

    TypeExpr* type = parseTypeExpr(tokenizer);

    Token rparen = tokenizer.nextToken();
    if (rparen.type != TOKEN_RPAREN) {
        eshkol_error("Expected ')' in type declaration");
        return nullptr;
    }

    return type;
}

// Parse inline type annotation: param : type
// Used within function parameter lists
TypeExpr* parseInlineTypeAnnotation(SchemeTokenizer& tokenizer,
                                     std::string& param_name) {
    Token name = tokenizer.nextToken();
    if (name.type != TOKEN_SYMBOL) {
        tokenizer.pushBack(name);
        return nullptr;
    }

    Token colon = tokenizer.nextToken();
    if (colon.type != TOKEN_COLON) {
        tokenizer.pushBack(colon);
        tokenizer.pushBack(name);
        return nullptr;
    }

    param_name = name.value;
    return parseTypeExpr(tokenizer);
}

// Parse type expression
TypeExpr* parseTypeExpr(SchemeTokenizer& tokenizer) {
    Token token = tokenizer.nextToken();

    if (token.type == TOKEN_SYMBOL) {
        // Simple type name: integer, float64, string, etc.
        return new TypeExpr{TypeExpr::Kind::Name, token.value, {}, nullptr, nullptr, ""};
    }

    if (token.type == TOKEN_LPAREN) {
        Token first = tokenizer.nextToken();

        // Check for forall
        if (first.value == "forall") {
            return parseForallType(tokenizer);
        }

        // Check for arrow type: (-> param-type return-type)
        if (first.value == "->") {
            return parseArrowType(tokenizer);
        }

        // Type application: (list integer), (vector float64 3)
        std::vector<TypeExpr> args;
        tokenizer.pushBack(first);
        while (true) {
            Token next = tokenizer.nextToken();
            if (next.type == TOKEN_RPAREN) break;
            tokenizer.pushBack(next);
            args.push_back(*parseTypeExpr(tokenizer));
        }
        return new TypeExpr{TypeExpr::Kind::Apply, first.value, args,
                           nullptr, nullptr, ""};
    }

    eshkol_error("Unexpected token in type expression");
    return nullptr;
}
```

---

## 8. Codegen Extensions

### 8.1 Enhanced TypedValue

```cpp
// Extended TypedValue in llvm_codegen.cpp
struct TypedValue {
    Value* llvm_value;
    eshkol_value_type_t runtime_type;  // Runtime tag
    types::TypeId static_type;          // NEW: Static type from checker
    bool is_exact;
    uint8_t flags;

    // Check if static type is available
    bool hasStaticType() const {
        return static_type.id != 0;
    }

    // Get the most precise type information
    types::TypeId effectiveType() const {
        if (hasStaticType()) return static_type;
        // Fall back to runtime type mapping
        return runtimeToStaticType(runtime_type);
    }
};
```

### 8.2 Type-Directed Codegen

```cpp
// In EshkolLLVMCodeGen class

// Generate code with type awareness
TypedValue codegenWithType(const eshkol_ast_t* ast,
                           const types::Context& ctx) {
    // Type-directed generation
    if (ast->type_annotation) {
        // Use explicit annotation
        types::TypeId expected = resolveTypeExpr(ast->type_annotation);
        TypedValue result = codegenAST(ast);

        // Insert coercion if needed
        if (!env_.isSubtype(result.static_type, expected)) {
            result = insertCoercion(result, expected);
        }

        return result;
    }

    // Fall back to inference
    return codegenAST(ast);
}

// Generate type promotion code
TypedValue promoteToType(TypedValue value, types::TypeId target) {
    using namespace types::BuiltinTypes;

    // Int64 to Float64
    if (env_.isSubtype(value.static_type, Int64) &&
        env_.isSubtype(target, Float64)) {
        Value* promoted = builder->CreateSIToFP(
            value.llvm_value,
            Type::getDoubleTy(*context),
            "int_to_float");
        return TypedValue{promoted, ESHKOL_VALUE_DOUBLE, Float64, false, 0};
    }

    // ... other promotions

    return value;
}

// Generate arithmetic with promotion
TypedValue codegenArithmetic(const eshkol_operations_t* op) {
    TypedValue left = codegenAST(op->call_op.variables);
    TypedValue right = codegenAST(op->call_op.variables + 1);

    // Determine result type
    types::TypeId result_type = env_.promoteForArithmetic(
        left.static_type, right.static_type);

    // Promote operands if needed
    left = promoteToType(left, result_type);
    right = promoteToType(right, result_type);

    // Generate operation
    Value* result;
    if (env_.isSubtype(result_type, types::BuiltinTypes::Float64)) {
        result = builder->CreateFAdd(left.llvm_value, right.llvm_value, "fadd");
    } else {
        result = builder->CreateAdd(left.llvm_value, right.llvm_value, "add");
    }

    return TypedValue{result,
                      result_type == types::BuiltinTypes::Float64
                          ? ESHKOL_VALUE_DOUBLE : ESHKOL_VALUE_INT64,
                      result_type,
                      env_.isSubtype(result_type, types::BuiltinTypes::Integer),
                      0};
}
```

### 8.3 Proof Erasure

```cpp
// Proof terms generate no runtime code
TypedValue codegenProof(const eshkol_ast_t* proof_ast) {
    // Proofs are erased - return unit value
    return TypedValue{
        ConstantInt::get(Type::getInt64Ty(*context), 0),
        ESHKOL_VALUE_NULL,
        types::BuiltinTypes::Null,
        true,
        0
    };
}

// Compile-time assertions become no-ops
void codegenStaticAssert(const eshkol_ast_t* assert_ast) {
    // Type checker has already verified
    // Generate nothing at runtime
}
```

---

## 9. Runtime Representation

### 9.1 Tagged Value Compatibility

The runtime representation remains compatible with the existing system:

```c
// No changes to tagged value structure
typedef struct eshkol_tagged_value {
    uint8_t type;        // Runtime type tag (4 bits used)
    uint8_t flags;       // Exactness + supertype category
    uint16_t reserved;
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;
    } data;
} eshkol_tagged_value_t;
```

### 9.2 Supertype Query Functions

```c
// Runtime supertype checking (when needed)
static inline bool eshkol_is_number(const eshkol_tagged_value_t* val) {
    uint8_t base = ESHKOL_GET_BASE_TYPE(val->type);
    return base == ESHKOL_VALUE_INT64 ||
           base == ESHKOL_VALUE_DOUBLE ||
           base == ESHKOL_VALUE_DUAL_NUMBER;
}

static inline bool eshkol_is_collection(const eshkol_tagged_value_t* val) {
    uint8_t base = ESHKOL_GET_BASE_TYPE(val->type);
    return base == ESHKOL_VALUE_CONS_PTR ||
           base == ESHKOL_VALUE_VECTOR_PTR ||
           base == ESHKOL_VALUE_TENSOR_PTR;
}

static inline bool eshkol_is_text(const eshkol_tagged_value_t* val) {
    uint8_t base = ESHKOL_GET_BASE_TYPE(val->type);
    return base == ESHKOL_VALUE_STRING_PTR ||
           base == ESHKOL_VALUE_CHAR;
}

// Get supertype category from flags
static inline uint8_t eshkol_get_supertype(const eshkol_tagged_value_t* val) {
    return val->flags & 0xF0;
}
```

---

## 10. Migration Path

### 10.1 Backward Compatibility

All existing Eshkol code continues to work:

```scheme
;; This still works (type inferred as Int64)
(define x 42)

;; This still works (type inferred as Float64)
(define y 3.14)

;; This still works (type inferred as (List Number))
(define nums (list 1 2.5 3))
```

### 10.2 Gradual Adoption

Type annotations are optional and can be added incrementally, following the
Eshkol type annotation conventions from ESHKOL_COMPLETE_LANGUAGE_SPECIFICATION.md:

```scheme
;; Step 1: No annotations (existing code - fully supported)
(define (add a b) (+ a b))

;; Step 2: Add separate type declaration
(: add (-> number number number))
(define (add a b) (+ a b))

;; Step 3: Add inline parameter types
(define (add a : number b : number) : number
  (+ a b))

;; Step 4: Constrain to specific types with polymorphism
(: add-int (-> integer integer integer))
(define (add-int a : integer b : integer) : integer
  (+ a b))

;; Step 5: Use type classes for generic operations
(: generic-add (forall (a) (-> a a a) (where (Numeric a))))
(define (generic-add x y) (+ x y))
```

### 10.3 Implementation Phases

```
Phase 1: Type Infrastructure
├── Implement TypeId and TypeNode
├── Implement TypeEnvironment
├── Implement subtype checking
└── Unit tests for type system

Phase 2: Parser Extensions
├── Add type annotation tokens (: name type)
├── Implement type expression parsing
├── Extend AST with type fields
└── Parser tests

Phase 3: HoTT Type Checker
├── Implement Context (typing environment Γ)
├── Implement bidirectional type checking
├── Implement dependent type verification
├── Implement identity types and path induction
└── Error reporting with proof obligations

Phase 4: Codegen Extensions
├── Extend TypedValue with static type info
├── Implement type-directed codegen
├── Implement numeric tower promotion
└── Proof erasure (zero runtime overhead)

Phase 5: Integration
├── Standard library types
├── Performance optimization
└── Documentation
```

---

## Summary

This specification extends Eshkol's type system with HoTT (Homotopy Type Theory)
foundations, moving beyond Hindley-Milner to support:

1. **Supertype hierarchy**: Value > Number > Integer > Int64
2. **Universe levels**: Type₀ (ground), Type₁ (constructors), Type₂ (propositions)
3. **Dependent types**: Types that depend on values (Π-types, Σ-types)
4. **Identity types**: HoTT-style equality with path induction
5. **Proof erasure**: Zero runtime overhead for type-level proofs
6. **Full backward compatibility**: Existing untyped code continues to work

The implementation follows Eshkol's established architecture:
- Types flow from parser through codegen
- Runtime uses tagged values (unchanged)
- Type annotations use Eshkol syntax: `(: name type)` and `param : type`
- Compatible with LLVM backend

Key HoTT features enabled:
- Dependent function types: `(forall (n : natural) (-> (vector float64 n) float64))`
- Refinement types: `(refine integer (lambda (x) (>= x 0)))`
- Linear types for resource management: `(Linear (Handle Window))`
- Path types for type equivalence proofs
