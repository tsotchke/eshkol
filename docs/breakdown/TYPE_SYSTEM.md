# Type System in Eshkol

## Table of Contents

- [Overview](#overview)
- [Runtime Type System](#runtime-type-system)
  - [Tagged Value Representation](#tagged-value-representation)
  - [Immediate Types](#immediate-types)
  - [Consolidated Pointer Types](#consolidated-pointer-types)
  - [Object Header System](#object-header-system)
  - [Exactness and Numeric Flags](#exactness-and-numeric-flags)
- [HoTT Compile-Time Type System](#hott-compile-time-type-system)
  - [Theoretical Foundations](#theoretical-foundations)
  - [Universe Hierarchy](#universe-hierarchy)
  - [Dependent Function Types (Pi-Types)](#dependent-function-types-pi-types)
  - [Dependent Pair Types (Sigma-Types)](#dependent-pair-types-sigma-types)
  - [Identity and Path Types](#identity-and-path-types)
  - [Sum Types (Coproducts)](#sum-types-coproducts)
  - [Type Expression Syntax](#type-expression-syntax)
- [Bidirectional Type Checking](#bidirectional-type-checking)
  - [Synthesis Mode](#synthesis-mode)
  - [Checking Mode](#checking-mode)
  - [Constraint Generation and Unification](#constraint-generation-and-unification)
- [Gradual Typing Semantics](#gradual-typing-semantics)
  - [Consistency Relation](#consistency-relation)
  - [Cast Insertion](#cast-insertion)
  - [Practical Implications](#practical-implications)
- [Numeric Type Hierarchy](#numeric-type-hierarchy)
  - [Scheme Numeric Tower](#scheme-numeric-tower)
  - [Arithmetic Promotion Rules](#arithmetic-promotion-rules)
- [Ownership and Linearity](#ownership-and-linearity)
  - [Linear Types](#linear-types)
  - [Borrow Checking](#borrow-checking)
- [Dependent Dimension Tracking](#dependent-dimension-tracking)
- [Type Annotations](#type-annotations)
- [Implementation Details](#implementation-details)
- [See Also](#see-also)

---

## Overview

Eshkol features a **triple-layer type system** that unifies runtime efficiency, compile-time
safety, and mathematical expressiveness:

1. **Runtime layer** — 16-byte tagged values with 8-bit type discriminators and 8-byte object
   headers for heap-allocated data. Every value is self-describing at runtime, enabling
   polymorphic dispatch without separate type metadata.

2. **Compile-time layer** — A Homotopy Type Theory (HoTT)-inspired type system with a universe
   hierarchy, dependent function types (Pi-types), dependent pair types (Sigma-types), and
   identity/path types. Type checking is bidirectional (synthesis and checking modes) and
   gradual (type errors produce warnings, not compilation failures).

3. **Dependent layer** — Compile-time value tracking for array dimensions, tensor shapes, and
   bounds checking. Values known at compile time (CTValues) flow through the type system to
   enable static verification of matrix multiplication dimensions, tensor reshapes, and
   index bounds.

The three layers interact coherently: the HoTT type system informs codegen optimization
(type-directed specialization), while the runtime tagged values provide a safety net for
gradual typing (any value can be inspected at runtime). Dependent dimension tracking bridges
compile-time guarantees with runtime tensor operations.

**Implementation:**

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Runtime Types | [eshkol.h](../../inc/eshkol/eshkol.h) | ~550 | Type tags, struct definitions, macros |
| HoTT Environment | [hott_types.cpp](../../lib/types/hott_types.cpp) | 752 | Universe hierarchy, subtyping, type nodes |
| Type Checker | [type_checker.cpp](../../lib/types/type_checker.cpp) | 1,999 | Bidirectional checking, constraint solving |
| Dependent Types | [dependent.cpp](../../lib/types/dependent.cpp) | 440 | Compile-time value tracking, dimension checks |
| LLVM Type Gen | [type_system.cpp](../../lib/backend/type_system.cpp) | 130 | Type-to-LLVM-type mapping |

---

## Runtime Type System

### Tagged Value Representation

Every Eshkol value at runtime is a **16-byte tagged value** (`eshkol_tagged_value_t`). The
structure packs a type discriminator, flags, and an 8-byte data payload into a cache-friendly
layout:

```c
typedef struct eshkol_tagged_value {
    uint8_t  type;       // Value type discriminator (0-255)
    uint8_t  flags;      // Exactness, port direction, etc.
    uint16_t reserved;   // Reserved for future use
    uint32_t padding;    // Alignment padding
    union {
        int64_t  int_val;    // Integer data
        double   double_val; // Floating-point data
        uint64_t ptr_val;    // Pointer to heap object
        uint64_t raw_val;    // Raw bit manipulation
    } data;
} eshkol_tagged_value_t;  // 16 bytes total
```

**LLVM struct type:** `{i8, i8, i16, i32, i64}` — the data field is at **index 4** (not 3).
This is critical for correct `ExtractValue` / `InsertValue` operations in codegen.

**Memory layout:**

```
Offset  Size  Field         Description
------  ----  -----         -----------
0       1     type          Type discriminator (eshkol_value_type_t)
1       1     flags         Exactness, port flags, etc.
2       2     reserved      Reserved for future use
4       4     padding       Alignment to 8-byte boundary
8       8     data          Union: int64 | double | pointer
------  ----
Total:  16 bytes
```

### Immediate Types

Types 0-7 store data directly in the tagged value (no heap allocation required):

| Tag | Constant | Description | Data Field |
|-----|----------|-------------|------------|
| 0 | `ESHKOL_VALUE_NULL` | Empty list `()` / null | Unused |
| 1 | `ESHKOL_VALUE_INT64` | 64-bit signed integer | `data.int_val` |
| 2 | `ESHKOL_VALUE_DOUBLE` | IEEE 754 double-precision float | `data.double_val` |
| 3 | `ESHKOL_VALUE_BOOL` | Boolean `#t` / `#f` | `data.int_val` (0 or 1) |
| 4 | `ESHKOL_VALUE_CHAR` | Unicode character | `data.int_val` (codepoint) |
| 5 | `ESHKOL_VALUE_SYMBOL` | Interned symbol | `data.int_val` (symbol ID) |
| 6 | `ESHKOL_VALUE_DUAL_NUMBER` | Forward-mode AD dual number | `data.ptr_val` -> `{value, derivative}` |
| 7 | `ESHKOL_VALUE_COMPLEX` | Complex number | `data.ptr_val` -> `{real, imag}` |
| 10 | `ESHKOL_VALUE_LOGIC_VAR` | Logic variable (`?x`) | `data.int_val` (variable ID) |

### Consolidated Pointer Types

Types 8-9 are **consolidated pointer types**. The `data.ptr_val` field points to a
heap-allocated object with an 8-byte header prepended at offset -8 from the data pointer.
The header's subtype field distinguishes the specific data structure.

#### HEAP_PTR (Type 8) — Data Structures

| Subtype | Constant | Description | Element Size |
|---------|----------|-------------|--------------|
| 0 | `HEAP_SUBTYPE_CONS` | Cons cell (pair) | 32 bytes (two tagged values) |
| 1 | `HEAP_SUBTYPE_STRING` | UTF-8 string | Variable (length-prefixed) |
| 2 | `HEAP_SUBTYPE_VECTOR` | Heterogeneous vector | 16 bytes per element (tagged values) |
| 3 | `HEAP_SUBTYPE_TENSOR` | N-dimensional numeric tensor | 8 bytes per element (int64 bitpatterns of doubles) |
| 4 | `HEAP_SUBTYPE_MULTI_VALUE` | Multiple return values | Variable |
| 5 | `HEAP_SUBTYPE_HASH` | Hash table (FNV-1a) | Variable |
| 6 | `HEAP_SUBTYPE_EXCEPTION` | Exception object | Variable |
| 7 | `HEAP_SUBTYPE_RECORD` | User-defined record | Variable |
| 8 | `HEAP_SUBTYPE_BYTEVECTOR` | Raw byte vector (R7RS) | 1 byte per element |
| 9 | `HEAP_SUBTYPE_PORT` | I/O port | Variable |
| 10 | `HEAP_SUBTYPE_SYMBOL` | Interned symbol (heap) | Variable |
| 11 | `HEAP_SUBTYPE_BIGNUM` | Arbitrary-precision integer | Variable (limb array) |
| 12 | `HEAP_SUBTYPE_SUBSTITUTION` | Logic substitution map | Variable |
| 13 | `HEAP_SUBTYPE_FACT` | Logic fact (predicate + args) | Variable |
| 15 | `HEAP_SUBTYPE_KNOWLEDGE_BASE` | Knowledge base (fact collection) | Variable |
| 16 | `HEAP_SUBTYPE_FACTOR_GRAPH` | Factor graph (probabilistic inference) | Variable |
| 17 | `HEAP_SUBTYPE_WORKSPACE` | Global workspace (cognitive competition) | Variable |
| 18 | `HEAP_SUBTYPE_PROMISE` | Lazy promise (delay/force) | Variable |
| 19 | `HEAP_SUBTYPE_RATIONAL` | Exact rational (numerator/denominator) | Variable |

#### CALLABLE (Type 9) — Function-Like Objects

| Subtype | Constant | Description |
|---------|----------|-------------|
| 0 | `CALLABLE_SUBTYPE_CLOSURE` | Compiled closure (function pointer + captured environment) |
| 1 | `CALLABLE_SUBTYPE_LAMBDA_SEXPR` | Lambda as data (homoiconic S-expression) |
| 2 | `CALLABLE_SUBTYPE_AD_NODE` | Autodiff computational graph node |
| 3 | `CALLABLE_SUBTYPE_PRIMITIVE` | Built-in primitive function |
| 4 | `CALLABLE_SUBTYPE_CONTINUATION` | First-class continuation (call/cc) |

### Object Header System

Every heap-allocated object has an **8-byte header at offset -8** from the data pointer:

```c
typedef struct eshkol_object_header {
    uint8_t  subtype;    // Type-specific subtype (0-255)
    uint8_t  flags;      // GC mark, linear, borrowed, etc.
    uint16_t ref_count;  // Reference count (0 = not ref-counted)
    uint32_t size;       // Object size in bytes (excluding header)
} eshkol_object_header_t;  // 8 bytes
```

**Object flags:**

| Flag | Constant | Purpose |
|------|----------|---------|
| `0x01` | `ESHKOL_OBJ_FLAG_MARKED` | GC mark bit (arena cleanup) |
| `0x02` | `ESHKOL_OBJ_FLAG_LINEAR` | Linear type (consume exactly once) |
| `0x04` | `ESHKOL_OBJ_FLAG_BORROWED` | Currently borrowed (temporary access) |
| `0x08` | `ESHKOL_OBJ_FLAG_CONSUMED` | Linear value has been consumed |
| `0x10` | `ESHKOL_OBJ_FLAG_SHARED` | Reference-counted shared object |
| `0x20` | `ESHKOL_OBJ_FLAG_WEAK` | Weak reference |
| `0x40` | `ESHKOL_OBJ_FLAG_PINNED` | Pinned in memory (no relocation) |
| `0x80` | `ESHKOL_OBJ_FLAG_EXTERNAL` | External resource (needs cleanup) |

### Exactness and Numeric Flags

Scheme's R7RS standard distinguishes **exact** (integer, rational, bignum) from **inexact**
(floating-point) numbers. Eshkol encodes this in the `flags` byte:

```c
#define ESHKOL_VALUE_EXACT_FLAG    0x10  // Exact number
#define ESHKOL_VALUE_INEXACT_FLAG  0x20  // Inexact number
```

Port types reuse the flags byte for direction:

```c
#define ESHKOL_PORT_INPUT_FLAG     0x10  // Input port
#define ESHKOL_PORT_OUTPUT_FLAG    0x40  // Output port
#define ESHKOL_PORT_BINARY_FLAG    0x04  // Binary (vs. textual) port
```

Port type checks use bitwise AND: `(type_tag & 0x50) != 0` to detect any port flag bits,
because ports have type `ESHKOL_VALUE_HEAP_PTR | flag_bits`.

---

## HoTT Compile-Time Type System

### Theoretical Foundations

Eshkol's compile-time type system draws from **Homotopy Type Theory** (HoTT), a foundation
of mathematics that unifies type theory with homotopy theory. In HoTT, types are interpreted
as spaces, terms as points in those spaces, and equality proofs as paths between points.
This foundation provides:

- **Universe hierarchy** — a stratified tower of type universes avoiding Russell's paradox
- **Dependent types** — types that depend on values, enabling dimension-indexed tensors
- **Path types** — proofs of equality that can be transported along type families
- **Proof erasure** — propositions (proofs) carry no runtime cost

Eshkol implements a **practical subset** of HoTT suitable for scientific computing: the
universe hierarchy organizes types, dependent function types (Pi-types) model dimension-polymorphic
operations, and path types provide a foundation for type equality reasoning. Proofs are
erased at compile time — they inform optimization but incur zero runtime overhead.

**Key references:**
- The HoTT Book (Univalent Foundations Program, 2013) — formal foundations
- Siek & Taha (2006) — gradual typing integration
- Pierce (2002) — bidirectional type checking

### Universe Hierarchy

Types in Eshkol are organized into a cumulative hierarchy of **universes**:

```
U₀ : U₁ : U₂ : ... : Uω
```

**Cumulativity:** If `Γ ⊢ A : Uₙ` then `Γ ⊢ A : Uₙ₊₁`. Every type at level n is also
a type at level n+1, ensuring that functions operating on types can accept types from any
lower universe.

**Universe contents:**

| Universe | Contents | Examples |
|----------|----------|----------|
| U₀ | Ground types (base values) | `Integer`, `Real`, `Boolean`, `String`, `Char`, `Symbol`, `Null`, `Complex`, `BigInt`, `Natural` |
| U₁ | Type constructors (parameterized types) | `List<T>`, `Vector<T>`, `Tensor<T>`, `Pair<A,B>`, `(-> A B)`, `DualNumber`, `ADNode`, `HashTable` |
| U₂ | Propositions (proof types, erased at runtime) | `Eq<A,a,b>`, `LessThan<a,b>`, `Bounded<n,lo,hi>`, `Subtype<A,B>` |
| Uω | Universe-polymorphic operations | Polymorphic combinators that work across universe levels |

**Formation rule:**

```
            Γ ⊢ A : Uᵢ
    ────────────────────── (Cumulativity)
         Γ ⊢ A : Uᵢ₊₁
```

**Implementation:** Each type is represented as a `TypeNode` with a `TypeId` containing a
universe level. The type environment ([hott_types.cpp](../../lib/types/hott_types.cpp))
registers ~35 builtin types across U₀-U₂, with IDs 0-999 reserved for builtins and 1000+
for user-defined types.

### Dependent Function Types (Pi-Types)

A **dependent function type** `Π(x:A).B(x)` represents functions whose return type depends
on the input value. When B does not depend on x, this reduces to the ordinary function
type `A → B`.

**Formation:**

```
    Γ ⊢ A : Uᵢ      Γ, x:A ⊢ B : Uⱼ
    ─────────────────────────────────── (Π-formation)
         Γ ⊢ Π(x:A).B : U_max(i,j)
```

**Introduction (lambda abstraction):**

```
         Γ, x:A ⊢ b : B
    ──────────────────────── (Π-introduction)
     Γ ⊢ λx.b : Π(x:A).B
```

**Elimination (function application):**

```
    Γ ⊢ f : Π(x:A).B      Γ ⊢ a : A
    ──────────────────────────────────── (Π-elimination)
            Γ ⊢ f(a) : B[a/x]
```

**Computation (beta-reduction):**

```
    (λx.b)(a) ≡ b[a/x]
```

**Eshkol implementation:** Pi-types are represented as `PiType` structs in
[hott_types.h](../../inc/eshkol/types/hott_types.h) with parameters that can be either
type parameters or value parameters (for dimensions). The `is_dependent` flag distinguishes
true dependent types from simple function types.

**Example — dimension-dependent function:**

```scheme
;; Type: Π(n:Nat). Vector<Float64, n> → Vector<Float64, n>
(define (normalize (v : (vector real)))
  (let ((n (norm v)))
    (tensor-apply (lambda (x) (/ x n)) v)))
```

### Dependent Pair Types (Sigma-Types)

A **dependent pair type** `Σ(x:A).B(x)` represents pairs where the type of the second
component depends on the value of the first. When B does not depend on x, this reduces
to the ordinary product type `A × B`.

**Formation:**

```
    Γ ⊢ A : Uᵢ      Γ, x:A ⊢ B : Uⱼ
    ─────────────────────────────────── (Σ-formation)
         Γ ⊢ Σ(x:A).B : U_max(i,j)
```

**Introduction (pair construction):**

```
    Γ ⊢ a : A      Γ ⊢ b : B[a/x]
    ──────────────────────────────── (Σ-introduction)
       Γ ⊢ (a, b) : Σ(x:A).B
```

**Elimination (projections):**

```
         Γ ⊢ p : Σ(x:A).B
    ────────────────────────── (Σ-elim-1)
          Γ ⊢ π₁(p) : A

         Γ ⊢ p : Σ(x:A).B
    ────────────────────────────── (Σ-elim-2)
       Γ ⊢ π₂(p) : B[π₁(p)/x]
```

**Eshkol implementation:** Sigma-types are represented as `SigmaType` structs in
[dependent.h](../../inc/eshkol/types/dependent.h) with a witness name, witness type, body
type, and dependency flag.

**Example — existential type (length-indexed vector):**

```scheme
;; Type: Σ(n:Nat). Vector<Float64, n>
;; "A vector of some length n, bundled with that length"
(define (read-data filename)
  ;; Returns a pair: (length . data-vector)
  (let ((data (parse-csv filename)))
    (cons (length data) (list->tensor data))))
```

### Identity and Path Types

In HoTT, an **identity type** (or **path type**) `Id_A(a, b)` represents a proof that two
terms `a` and `b` of type `A` are equal. A term `p : Id_A(a, b)` is a "path" from `a` to `b`
in the space `A`.

**Formation:**

```
    Γ ⊢ A : Uᵢ      Γ ⊢ a : A      Γ ⊢ b : A
    ─────────────────────────────────────────────── (Id-formation)
                Γ ⊢ Id_A(a, b) : Uᵢ
```

**Introduction (reflexivity):**

```
          Γ ⊢ a : A
    ───────────────────── (reflexivity)
     Γ ⊢ refl_a : Id_A(a, a)
```

**Elimination (transport / path induction):**

Given `p : Id_A(a, b)` and a type family `P : A → U`, we obtain:

```
    Γ ⊢ p : Id_A(a, b)      Γ ⊢ P : A → Uⱼ
    ────────────────────────────────────────── (transport)
           Γ ⊢ p* : P(a) → P(b)
```

Transport allows "moving" a proof of `P(a)` to a proof of `P(b)` along the path `p`.

**Eshkol implementation:** Path types are represented with `HOTT_TYPE_PATH` kind in the
type expression AST. They are used at the U₂ (proposition) level and are **erased at runtime**
— they exist only to guide type checking and optimization.

**Example — type-safe vector operations:**

```scheme
;; The type system tracks that reshape preserves total element count:
;; If v has shape (m, n), then reshape(v, (m*n,)) has shape (m*n,)
;; Path type: Id_Nat(m*n, length(reshape(v, (m*n,))))
```

### Sum Types (Coproducts)

A **sum type** (coproduct) `A + B` represents a value that is **either** of type `A` or
type `B`, with injection functions distinguishing the variants.

**Formation:**

```
    Γ ⊢ A : Uᵢ      Γ ⊢ B : Uⱼ
    ───────────────────────────── (+-formation)
       Γ ⊢ A + B : U_max(i,j)
```

**Introduction:**

```
        Γ ⊢ a : A                    Γ ⊢ b : B
    ─────────────────── (inl)    ─────────────────── (inr)
     Γ ⊢ inl(a) : A + B          Γ ⊢ inr(b) : A + B
```

**Elimination (case analysis):**

```
    Γ ⊢ s : A + B    Γ, x:A ⊢ c₁ : C    Γ, y:B ⊢ c₂ : C
    ──────────────────────────────────────────────────────── (case)
                      Γ ⊢ case(s, c₁, c₂) : C
```

**Eshkol builtins:** `inject-left`, `inject-right`, `sum-tag`, `sum-value`, `left?`, `right?`

```scheme
(define result (inject-left 42))   ; Left variant
(sum-tag result)                    ; => 0
(sum-value result)                  ; => 42
(left? result)                      ; => #t
```

### Type Expression Syntax

Eshkol's type expression grammar covers all HoTT constructs:

```scheme
;; Primitive types (U₀)
integer        ; 64-bit signed integer (exact)
real           ; double-precision float (inexact)
number         ; supertype of integer and real
boolean        ; #t or #f
string         ; UTF-8 string
char           ; Unicode character
symbol         ; Interned symbol
null           ; Empty list / null value

;; Compound types (U₁)
(-> A B ... R)           ; Function type (last type is return)
(list T)                 ; Homogeneous list
(vector T)               ; Heterogeneous Scheme vector
(tensor T)               ; N-dimensional numeric tensor
(pair A B)               ; Cons pair
(* A B)                  ; Product type
(+ A B)                  ; Sum type (coproduct / either)

;; Polymorphic types
(forall (a b ...) T)     ; Universal quantification

;; Special types
any                      ; Top type (accepts anything)
nothing                  ; Bottom type (uninhabited)

;; Type-level constructs (U₂ — erased at runtime)
(path A a b)             ; Identity/path type: proof that a ≡ b in A
(universe n)             ; Universe at level n
```

**Implementation:** Type expressions are parsed into `hott_type_expr_t` AST nodes
([eshkol.h](../../inc/eshkol/eshkol.h)) with kind tags (`hott_type_kind_t`) for each
construct: `HOTT_TYPE_ARROW`, `HOTT_TYPE_LIST`, `HOTT_TYPE_FORALL`, `HOTT_TYPE_DEPENDENT`,
`HOTT_TYPE_PATH`, `HOTT_TYPE_UNIVERSE`, `HOTT_TYPE_SUM`, `HOTT_TYPE_PRODUCT`, etc.

---

## Bidirectional Type Checking

Eshkol's type checker operates in two complementary modes, following the bidirectional
type checking methodology (Pierce & Turner, 2000):

### Synthesis Mode

**Synthesis** (⇒) infers a type from the expression's structure (bottom-up). Used when the
type is determinable from the expression alone.

```
    ─────────────── (Syn-Int)        ─────────────────── (Syn-Bool)
    Γ ⊢ n ⇒ Integer                 Γ ⊢ #t ⇒ Boolean

    x:A ∈ Γ
    ────────────── (Syn-Var)
    Γ ⊢ x ⇒ A

    Γ ⊢ f ⇒ (→ A₁...Aₙ R)      Γ ⊢ aᵢ ⇐ Aᵢ  (for each i)
    ──────────────────────────────────────────────────────── (Syn-App)
                         Γ ⊢ (f a₁...aₙ) ⇒ R
```

Synthesis is used for: literals, variable references, function applications (when the
function's type is known), and arithmetic operations (via the numeric tower).

### Checking Mode

**Checking** (⇐) verifies that an expression matches an expected type (top-down). Used when
the expected type is known from context (e.g., function parameter types, explicit annotations).

```
    Γ, x:A ⊢ e ⇐ B
    ──────────────────────── (Chk-Lambda)
    Γ ⊢ (lambda (x) e) ⇐ (→ A B)

    Γ ⊢ e ⇒ A      A ~ B
    ─────────────────────── (Chk-Subsumption)
         Γ ⊢ e ⇐ B
```

The subsumption rule uses the **consistency relation** `~` from gradual typing (see below),
not strict equality. This allows typed and untyped code to interoperate seamlessly.

### Constraint Generation and Unification

The type checker processes the AST in a single pass, generating constraints and solving them
via **unification**:

1. **Constraint generation** — Recursive AST traversal produces a set of type equality
   constraints `τ₁ = τ₂` and subtype constraints `τ₁ <: τ₂`.

2. **Unification** — Constraints are solved by finding a substitution σ mapping type
   variables to concrete types such that all constraints are satisfied.

3. **Subtype resolution** — Subtype constraints use the type hierarchy: `Integer <: Number`,
   `Number <: Value`, etc. The type environment caches subtype judgments in a
   `map<(sub_id, super_id), bool>` for O(1) amortized lookup.

4. **Error reporting** — When unification fails, the checker emits a **warning** (not an
   error) with diagnostic context: expected type, actual type, source location, and a
   suggested fix. The program compiles regardless.

**Implementation:** The `Context` class ([type_checker.cpp](../../lib/types/type_checker.cpp))
maintains a scope stack for lexical scoping, type aliases from `define-type`, and tracks
the current checking mode. Each AST node is processed by `checkExpression` (checking mode)
or `synthesizeExpression` (synthesis mode).

---

## Gradual Typing Semantics

Eshkol implements **gradual typing** (Siek & Taha, 2006), where type annotations are
optional and type mismatches produce warnings rather than errors. This design enables
rapid prototyping with progressive type safety.

### Consistency Relation

Two types `τ` and `σ` are **consistent** (written `τ ~ σ`) if and only if:

1. `τ = ?` (the unknown/dynamic type), or
2. `σ = ?`, or
3. `τ` and `σ` are structurally compatible:
   - Both are the same base type
   - Both are function types with pairwise consistent parameter and return types
   - Both are container types with consistent element types
   - `τ <: σ` in the type hierarchy (subtype consistency)

Consistency is **reflexive** and **symmetric** but **not transitive** — this is the key
difference from subtyping that makes gradual typing work.

### Cast Insertion

At type boundaries (where typed code meets untyped code), the compiler conceptually inserts
**runtime casts**. In Eshkol, these casts are trivially satisfied by the tagged value system:
every value already carries its type tag at runtime, so "casting" is simply a runtime type check.

```scheme
;; Typed function
(define (square (x : real)) : real
  (* x x))

;; Untyped call — compiler inserts implicit cast (real? check at runtime)
(define y (square some-untyped-value))
```

### Practical Implications

- **All Scheme code compiles** — untyped programs have type `?` everywhere, and `? ~ τ` for
  all `τ`, so no type errors are possible.
- **Type annotations are optimization hints** — when the compiler knows a value is `Int64`,
  it can emit direct machine instructions instead of polymorphic dispatch.
- **Progressive typing** — developers can add annotations incrementally, each one providing
  more compile-time safety and better performance.
- **Warnings guide improvement** — type warnings identify potential bugs without blocking
  development.

---

## Numeric Type Hierarchy

### Scheme Numeric Tower

Eshkol implements the R7RS numeric tower with exact/inexact distinction:

```
                    Number
                   /      \
              Exact        Inexact
             /    \            \
         Integer  Rational    Real (Float64)
        /     \                     \
    Int64   BigInt               Complex
```

| Type | Exactness | Representation | Range |
|------|-----------|----------------|-------|
| Int64 | Exact | 64-bit signed | -2^63 to 2^63 - 1 |
| BigInt | Exact | Limb array (arbitrary) | Unbounded |
| Rational | Exact | Numerator/denominator (always reduced) | Exact fractions |
| Float64 | Inexact | IEEE 754 double | ~15 decimal digits |
| Complex | Inexact | `{real: f64, imag: f64}` | Complex plane |

### Arithmetic Promotion Rules

When operands have different numeric types, the compiler promotes according to R7RS semantics:

| Left | Right | Result | Rule |
|------|-------|--------|------|
| Int64 | Int64 | Int64 | Same type |
| Int64 | Float64 | Float64 | Exact + inexact = inexact |
| Int64 | BigInt | BigInt | Promote to wider exact |
| BigInt | Float64 | Float64 | Exact + inexact = inexact |
| BigInt | Rational | Rational | Both exact, rational is more general |
| Any numeric | Complex | Complex | Complex absorbs all |
| Int64 overflow | — | BigInt | Automatic promotion |
| BigInt fits int64 | — | Int64 | Automatic demotion |

**Implementation:** The `promoteForArithmetic` method in the type environment walks the
supertype chains of both operands to find the least common supertype. The result is cached
for performance.

---

## Ownership and Linearity

### Linear Types

Eshkol supports **linear types** for resource management. A linear value must be used
**exactly once** — it cannot be duplicated or silently discarded.

```
    Γ, x:!A ⊢ e : B      x used exactly once in e
    ──────────────────────────────────────────────── (Linear-Intro)
              Γ ⊢ (lambda (x) e) : !A → B
```

The `LinearContext` class tracks usage counts for linear variables:
- `Unused` — declared but not yet consumed
- `UsedOnce` — consumed exactly once (valid)
- `UsedMultiple` — consumed more than once (error: duplication)

Linear types are used for:
- File handles (must be closed)
- Mutable references (single-owner mutation)
- Arena-allocated objects (tracked lifetime)

### Borrow Checking

The `BorrowChecker` enforces Rust-inspired ownership rules:

| State | Can Move? | Can Borrow Shared? | Can Borrow Mutable? |
|-------|-----------|--------------------|---------------------|
| `Owned` | Yes | Yes | Yes |
| `Moved` | No | No | No |
| `BorrowedShared` | No | Yes (multiple) | No |
| `BorrowedMut` | No | No | No (exclusive) |
| `Dropped` | No | No | No |

**Rules enforced:**
1. A value can be moved at most once
2. While borrowed, the value cannot be moved
3. Mutable borrows are exclusive (no concurrent borrows)
4. Shared borrows allow multiple simultaneous readers
5. Borrows must not outlive the borrowed value

An `UnsafeContext` provides an escape hatch for FFI and low-level memory operations,
bypassing linear type and borrow checking within explicitly marked unsafe blocks.

---

## Dependent Dimension Tracking

The dependent type layer tracks **compile-time values** (CTValues) that flow through tensor
operations, enabling static verification of dimension constraints.

```cpp
// CTValue can be:
// - Literal natural: CTValue::makeNat(100)
// - Boolean: CTValue::makeBool(true)
// - Symbolic expression: CTValue::makeExpr(ast_ptr)
// - Unknown: CTValue::makeUnknown()
```

**Dimension checking rules** (from [dependent.cpp](../../lib/types/dependent.cpp)):

| Operation | Constraint | Error if violated |
|-----------|-----------|-------------------|
| `(vref v i)` | `0 <= i < length(v)` | Index out of bounds |
| `(matmul A B)` | `cols(A) = rows(B)` | Dimension mismatch |
| `(tensor-dot a b)` | `length(a) = length(b)` | Vector length mismatch |
| `(reshape v dims)` | `product(dims) = length(v)` | Element count mismatch |

When dimensions are known at compile time, the `DimensionChecker` verifies constraints
statically. When dimensions are unknown (runtime-determined), checks are deferred to runtime.

**Example:**

```scheme
(define (safe-matmul (A : (tensor real)) (B : (tensor real)))
  ;; Compiler tracks: A is m×n, B is n×p → result is m×p
  ;; If A is 3×4 and B is 5×2, compile-time warning:
  ;; "matmul dimension mismatch: inner dimensions 4 ≠ 5"
  (matmul A B))
```

---

## Type Annotations

Eshkol supports several annotation forms:

**Inline parameter types:**

```scheme
(define (add-ints (x : integer) (y : integer)) : integer
  (+ x y))
```

**Separate type declarations:**

```scheme
(: factorial (-> integer integer))
(define (factorial n)
  (if (<= n 1) 1 (* n (factorial (- n 1)))))
```

**Lambda type annotations:**

```scheme
(lambda ((x : real)) : real
  (* x x))
```

**Type aliases:**

```scheme
(define-type Point (pair real real))
(define-type Matrix (tensor real))
(define-type Predicate (-> any boolean))
```

**Parameterized type aliases:**

```scheme
(define-type (Maybe a) (+ a null))
(define-type (Either a b) (+ a b))
```

---

## Implementation Details

### Subtype Caching

The type environment caches subtype judgments for O(1) amortized lookup:

```cpp
mutable map<pair<uint16_t, uint16_t>, bool> subtype_cache_;
mutable map<uint16_t, PiType> function_type_cache_;
mutable map<uint16_t, pair<TypeId, TypeId>> pair_element_cache_;
mutable map<uint16_t, vector<CTValueSimple>> dimension_cache_;
```

### Type ID Allocation

| ID Range | Purpose |
|----------|---------|
| 0-9 | Reserved (universes, special) |
| 10-99 | Ground types (Integer, Real, Boolean, etc.) |
| 100-199 | Collection types (List, Vector, Tensor, etc.) |
| 200-299 | Proposition types (Eq, LessThan, etc.) |
| 300-499 | Dynamically allocated pair types |
| 500-999 | Dynamically allocated function types |
| 1000+ | User-defined types |

### Runtime Representation Mapping

Each compile-time type maps to a runtime representation for codegen:

| RuntimeRep | Description | Used by |
|------------|-------------|---------|
| `Int64` | 64-bit signed integer | Integer, Natural, Boolean, Char |
| `Float64` | IEEE 754 double | Real, Float64 |
| `Pointer` | Pointer to heap object | String, List, Vector, Tensor |
| `TaggedValue` | Full 16-byte tagged value | Polymorphic contexts |
| `Struct` | LLVM struct type | Closures, dependent pairs |
| `Erased` | No runtime representation | Propositions, universe types |

---

## v1.1 Type Extensions

Eshkol v1.1 ("accelerate") extends the core type system with five categories of new types: logic variables for symbolic reasoning, consciousness engine heap subtypes for active inference, exact arithmetic types for R7RS compliance, port type flag encoding for I/O, and promises for lazy evaluation.

All v1.1 types follow the same architectural pattern established in v1.0: immediate values occupy a single `eshkol_tagged_value_t` (16 bytes), while heap-allocated types use `ESHKOL_VALUE_HEAP_PTR` (type tag 8) with an `eshkol_object_header_t` prefix (8 bytes) that carries the subtype discriminator.

### 1. Logic Variable Type (ESHKOL_VALUE_LOGIC_VAR)

Logic variables are the only new *immediate* type added in v1.1. They occupy type tag `10` in the `eshkol_value_type_t` enum, placed directly after the consolidated pointer types (8-9) and before the reserved multimedia range (16-19).

**Tagged value encoding:**

| Field      | Type     | Value                          |
|------------|----------|--------------------------------|
| `type`     | `uint8_t`  | `10` (`ESHKOL_VALUE_LOGIC_VAR`) |
| `flags`    | `uint8_t`  | `0`                            |
| `reserved` | `uint16_t` | `0`                            |
| `data.int_val` | `int64_t` | Variable ID (monotonic, 0-based) |

**Parser integration.** The parser recognizes the `?x` syntax (where `?` is followed by one or more identifier characters) as a logic variable. This is R7RS-compatible because `?` is a valid identifier-start character in Scheme. When the lexer produces a `TOKEN_SYMBOL` whose value begins with `?` and has length > 1, the parser sets the AST node type to `ESHKOL_OP` with operation `ESHKOL_LOGIC_VAR_OP`. It calls `eshkol_make_logic_var(name)` to obtain the variable ID, which is stored in `ast.operation.logic_var_op.var_id`. The variable name string is also copied into the AST node for display purposes.

**Variable ID allocation.** The global variable registry in `logic.cpp` maintains:
- `g_var_names[LOGIC_VAR_MAX]` -- static array of interned name pointers (max 65,536 variables)
- `g_var_count` -- `std::atomic<uint64_t>` counter for ID allocation
- `g_var_mutex` -- `std::mutex` protecting the name array

`eshkol_make_logic_var(name)` acquires the mutex, performs a linear scan of `g_var_names` to check for an existing registration (same name always returns the same ID), and if not found, atomically increments `g_var_count` via `fetch_add(1)` to allocate a new slot. Name strings are interned in a separate `g_var_name_pool` (lock-free atomic offset bump allocator) to avoid per-variable heap allocation. The mutex serializes the compound find-then-register operation, while the atomic counter provides the unique ID guarantee even under concurrent registration attempts that race past the limit check.

**Unification semantics.** The `eshkol_unify()` function implements Robinson's unification algorithm with occurs check:

1. **Walk** both terms through the substitution to resolve variable chains.
2. If the walked terms are **identical** (same type and value, including same logic variable ID), return the substitution unchanged.
3. If either walked term is a **logic variable**, perform the **occurs check** (recursive descent into fact arguments with a depth limit of 1,000) to prevent circular bindings, then return an extended substitution binding that variable to the other term.
4. If both walked terms are **HEAP_PTR** values with **HEAP_SUBTYPE_FACT** headers, perform **structural unification**: check predicate pointer equality (interned symbols), check arity equality, then recursively unify each argument pair.
5. Otherwise, unification **fails** (returns `NULL`).

Access macros for type checking:
```c
#define ESHKOL_IS_LOGIC_VAR(tv)  ((tv).type == ESHKOL_VALUE_LOGIC_VAR)
#define ESHKOL_LOGIC_VAR_ID(tv)  ((tv).data.int_val)
```

### 2. Consciousness Engine Heap Subtypes

The consciousness engine introduces five new heap subtypes (12, 13, 15, 16, 17) under the existing `ESHKOL_VALUE_HEAP_PTR` type tag. Subtype 14 is reserved for rules (planned for v1.2 backward chaining). All consciousness engine objects are arena-allocated with the standard `eshkol_object_header_t` prefix.

#### HEAP_SUBTYPE_SUBSTITUTION (12)

Substitutions are immutable mappings from logic variable IDs to terms, used by the unification engine. They follow **copy-on-extend semantics**: each `eshkol_extend_subst()` call allocates a new substitution containing all old bindings plus the new one. The old substitution is never modified (arena-allocated, never freed), which enables backtracking by simply discarding the extended substitution.

**Memory layout:** `[eshkol_object_header_t (8 bytes)][eshkol_substitution_t][var_ids...][terms...]`

**Struct definition** (`inc/eshkol/core/logic.h`):

```c
typedef struct eshkol_substitution {
    uint32_t num_bindings;   // Current number of bound variables
    uint32_t capacity;       // Allocated capacity for parallel arrays
    /* Followed by: uint64_t var_ids[capacity] */
    /* Followed by: eshkol_tagged_value_t terms[capacity] */
} eshkol_substitution_t;
```

| Field           | Type       | Size   | Description                              |
|-----------------|------------|--------|------------------------------------------|
| `num_bindings`  | `uint32_t` | 4 bytes | Number of active bindings                |
| `capacity`      | `uint32_t` | 4 bytes | Array capacity (grows on extend)         |
| `var_ids[]`     | `uint64_t[]` | 8 * capacity bytes | Variable ID parallel array |
| `terms[]`       | `eshkol_tagged_value_t[]` | 16 * capacity bytes | Bound term parallel array |

Total allocation: `8 (header) + 8 (struct) + capacity * (8 + 16)` bytes.

Lookup is linear scan over the `var_ids` array (sufficient for v1.1 workloads). Access macros:
```c
#define SUBST_VAR_IDS(s) ((uint64_t*)((uint8_t*)(s) + sizeof(eshkol_substitution_t)))
#define SUBST_TERMS(s)   ((eshkol_tagged_value_t*)((uint8_t*)(s) + sizeof(eshkol_substitution_t) + (s)->capacity * sizeof(uint64_t)))
```

#### HEAP_SUBTYPE_FACT (13)

Facts represent predicate-argument structures of the form `(predicate arg1 arg2 ...)`. The predicate is stored as a pointer to an interned symbol (`HEAP_SUBTYPE_SYMBOL`), enabling O(1) predicate matching via pointer equality.

**Memory layout:** `[eshkol_object_header_t (8 bytes)][eshkol_fact_t][args...]`

**Struct definition** (`inc/eshkol/core/logic.h`):

```c
typedef struct eshkol_fact {
    uint64_t predicate;    // Pointer to interned symbol (HEAP_SUBTYPE_SYMBOL)
    uint32_t arity;        // Number of arguments
    uint32_t _pad;         // Padding for alignment
    /* Followed by: eshkol_tagged_value_t args[arity] */
} eshkol_fact_t;
```

| Field        | Type       | Size   | Description                              |
|--------------|------------|--------|------------------------------------------|
| `predicate`  | `uint64_t` | 8 bytes | Interned symbol pointer                 |
| `arity`      | `uint32_t` | 4 bytes | Number of arguments                     |
| `_pad`       | `uint32_t` | 4 bytes | Alignment padding                       |
| `args[]`     | `eshkol_tagged_value_t[]` | 16 * arity bytes | Argument values |

Total allocation: `8 (header) + 16 (struct) + arity * 16` bytes.

Access macro:
```c
#define FACT_ARGS(f) ((eshkol_tagged_value_t*)((uint8_t*)(f) + sizeof(eshkol_fact_t)))
```

#### HEAP_SUBTYPE_KNOWLEDGE_BASE (15)

The knowledge base is a growable collection of fact pointers with query support. It uses arena allocation -- when the facts array is full, a new larger array is allocated and the old pointers are copied. v1.2 will add predicate indexing for O(1) lookup.

**Memory layout:** `[eshkol_object_header_t (8 bytes)][eshkol_knowledge_base_t]`

**Struct definition** (`inc/eshkol/core/logic.h`):

```c
typedef struct eshkol_knowledge_base {
    uint32_t num_facts;              // Number of asserted facts
    uint32_t capacity;               // Array capacity
    eshkol_fact_t** facts;           // Arena-allocated array of fact pointers
} eshkol_knowledge_base_t;
```

| Field        | Type              | Size    | Description                              |
|--------------|-------------------|---------|------------------------------------------|
| `num_facts`  | `uint32_t`        | 4 bytes  | Current fact count                      |
| `capacity`   | `uint32_t`        | 4 bytes  | Allocated array slots                   |
| `facts`      | `eshkol_fact_t**` | 8 bytes  | Pointer to arena-allocated fact pointer array |

Total struct size: 16 bytes (plus 8-byte header). The facts array itself is a separately arena-allocated block of `capacity * 8` bytes.

Query (`eshkol_kb_query`) returns a cons list of substitutions by attempting unification of the query pattern against each fact in the knowledge base. An optional initial substitution may be provided to constrain the query.

#### HEAP_SUBTYPE_FACTOR_GRAPH (16)

Factor graphs support probabilistic inference via the sum-product algorithm (loopy belief propagation) in log-space. Variables are discrete random variables, factors connect variable subsets via conditional probability tables (CPTs), and beliefs are maintained as log-probability vectors.

**Memory layout:** `[eshkol_object_header_t (8 bytes)][eshkol_factor_graph_t]`

**Factor struct** (`inc/eshkol/core/inference.h`):

```c
typedef struct eshkol_factor {
    uint32_t num_vars;         // Number of connected variables
    uint32_t cpt_size;         // Total entries in CPT (product of connected variable dims)
    double*  cpt;              // Pointer to log-probability tensor (arena-allocated)
    uint32_t* dims;            // Dimension of each connected variable's state space
    /* Followed by: uint32_t var_indices[num_vars] */
} eshkol_factor_t;
```

| Field        | Type         | Size    | Description                              |
|--------------|--------------|---------|------------------------------------------|
| `num_vars`   | `uint32_t`   | 4 bytes  | Number of variables this factor connects |
| `cpt_size`   | `uint32_t`   | 4 bytes  | Flat CPT array length (d0 * d1 * ... * dn) |
| `cpt`        | `double*`    | 8 bytes  | Log-probability tensor data              |
| `dims`       | `uint32_t*`  | 8 bytes  | Per-variable state space dimensions      |
| `var_indices[]` | `uint32_t[]` | 4 * num_vars bytes | Indices into factor graph's variable array |

**Factor graph struct** (`inc/eshkol/core/inference.h`):

```c
typedef struct eshkol_factor_graph {
    uint32_t num_vars;         // Total number of random variables
    uint32_t num_factors;      // Number of factors currently added
    uint32_t max_factors;      // Capacity of factors array
    uint32_t _pad;             // Padding for alignment
    double** beliefs;          // beliefs[i] = log-probability vector for variable i
    uint32_t* var_dims;        // var_dims[i] = number of states for variable i
    eshkol_factor_t** factors; // Array of factor pointers
    double** msg_fv;           // Messages from factors to variables
    double** msg_vf;           // Messages from variables to factors
    uint32_t total_messages;   // Total number of factor-variable edges
} eshkol_factor_graph_t;
```

| Field             | Type                | Size    | Description                              |
|-------------------|---------------------|---------|------------------------------------------|
| `num_vars`        | `uint32_t`          | 4 bytes  | Variable count                          |
| `num_factors`     | `uint32_t`          | 4 bytes  | Current factor count                    |
| `max_factors`     | `uint32_t`          | 4 bytes  | Factor array capacity                   |
| `_pad`            | `uint32_t`          | 4 bytes  | Alignment padding                       |
| `beliefs`         | `double**`          | 8 bytes  | Array of per-variable belief vectors    |
| `var_dims`        | `uint32_t*`         | 8 bytes  | State space dimension per variable      |
| `factors`         | `eshkol_factor_t**` | 8 bytes  | Factor pointer array                    |
| `msg_fv`          | `double**`          | 8 bytes  | Factor-to-variable message arrays       |
| `msg_vf`          | `double**`          | 8 bytes  | Variable-to-factor message arrays       |
| `total_messages`  | `uint32_t`          | 4 bytes  | Edge count (for message array sizing)   |

Total struct size: 60 bytes (with 4 bytes trailing padding to 64 for alignment, plus 8-byte header).

Beliefs are initialized to uniform (`log(1/dim)`) on construction. The `eshkol_fg_infer()` function performs loopy BP with configurable iteration limit and convergence tolerance. `fg-update-cpt!` enables learning by mutating a factor's CPT and resetting all messages, causing beliefs to reconverge on the next inference pass.

Free energy computation follows the variational formulation: `F = E_q[ln q(s)] - E_q[ln p(o,s)]`. Observations are passed as `#(var_index observed_state)` pairs, not full state vectors. Expected free energy decomposes into pragmatic value (goal achievement) and epistemic value (uncertainty reduction).

#### HEAP_SUBTYPE_WORKSPACE (17)

The global workspace implements Global Workspace Theory (Baars 1988, Bengio 2017). Cognitive modules (Eshkol closures) compete for attention via softmax over salience scores. The winning module's proposal becomes the new workspace content, broadcast to all modules in the next cycle.

**Memory layout:** `[eshkol_object_header_t (8 bytes)][eshkol_workspace_t][modules...]`

**Module struct** (`inc/eshkol/core/workspace.h`):

```c
typedef struct eshkol_workspace_module {
    char* name;                        // Module name (arena-allocated string)
    eshkol_tagged_value_t process_fn;  // Closure tagged value
    double salience;                   // Last computed salience score
} eshkol_workspace_module_t;
```

| Field        | Type                      | Size     | Description                              |
|--------------|---------------------------|----------|------------------------------------------|
| `name`       | `char*`                   | 8 bytes   | Arena-allocated name string             |
| `process_fn` | `eshkol_tagged_value_t`   | 16 bytes  | Closure: `(tensor -> (cons double tensor))` |
| `salience`   | `double`                  | 8 bytes   | Last salience score from competition    |

Module struct size: 32 bytes.

**Workspace struct** (`inc/eshkol/core/workspace.h`):

```c
typedef struct eshkol_workspace {
    uint32_t num_modules;              // Current registered module count
    uint32_t max_modules;              // Module array capacity
    uint32_t dim;                      // Workspace vector dimension
    uint32_t step_count;               // Cognitive cycle counter
    double* content;                   // Current workspace content (arena tensor data)
    /* Followed by: eshkol_workspace_module_t modules[max_modules] */
} eshkol_workspace_t;
```

| Field         | Type       | Size    | Description                              |
|---------------|------------|---------|------------------------------------------|
| `num_modules` | `uint32_t` | 4 bytes  | Registered module count                 |
| `max_modules` | `uint32_t` | 4 bytes  | Module array capacity                   |
| `dim`         | `uint32_t` | 4 bytes  | Content vector dimension                |
| `step_count`  | `uint32_t` | 4 bytes  | Cycles completed                        |
| `content`     | `double*`  | 8 bytes  | Pointer to workspace content doubles    |

Total struct size: 24 bytes (plus 8-byte header, plus `max_modules * 32` bytes for the trailing module array).

The `ws-step!` cognitive cycle is implemented in LLVM codegen (not the C runtime) because it must invoke Eshkol closures via `codegenClosureCall`. C runtime helpers `eshkol_ws_make_content_tensor` (wraps content doubles into a tensor tagged value for passing to closures) and `eshkol_ws_step_finalize` (performs softmax over salience scores, copies winner's proposal to workspace content) handle the non-closure parts.

Access macro:
```c
#define WS_MODULES(ws) ((eshkol_workspace_module_t*)((uint8_t*)(ws) + sizeof(eshkol_workspace_t)))
```

### 3. Exact Arithmetic Types

#### Bignum (HEAP_SUBTYPE_BIGNUM = 11)

Bignums provide arbitrary-precision exact integers as required by R7RS. They use a sign-magnitude representation with an array of 64-bit "limbs" stored little-endian (least significant limb first).

**Memory layout:** `[eshkol_object_header_t (8 bytes)][eshkol_bignum_t][limbs...]`

**Struct definition** (`inc/eshkol/core/bignum.h`):

```c
typedef struct eshkol_bignum {
    int32_t  sign;       // 0 = non-negative, 1 = negative
    uint32_t num_limbs;  // Number of 64-bit limbs
    /* uint64_t limbs[] follows in memory */
} eshkol_bignum_t;
```

| Field       | Type       | Size   | Description                              |
|-------------|------------|--------|------------------------------------------|
| `sign`      | `int32_t`  | 4 bytes | 0 = non-negative, 1 = negative          |
| `num_limbs` | `uint32_t` | 4 bytes | Limb count                              |
| `limbs[]`   | `uint64_t[]` | 8 * num_limbs bytes | Little-endian limb array |

The numeric value is: `sign_factor * sum(limbs[i] * 2^(64*i), i=0..num_limbs-1)`, where `sign_factor` is 1 if `sign == 0`, and -1 if `sign == 1`. Zero is represented as `num_limbs=1, limbs[0]=0, sign=0`.

**`__uint128_t` arithmetic.** The implementation uses GCC/Clang's 128-bit integer extension (`__uint128_t`) extensively for carry propagation in the core algorithms:
- **Addition**: `__uint128_t sum = (__uint128_t)av + (__uint128_t)bv + carry` -- the upper 64 bits become the next carry.
- **Multiplication**: `__uint128_t prod = (__uint128_t)a[i] * (__uint128_t)b_limb + (__uint128_t)result[i + offset] + carry` -- schoolbook multiply with 128-bit intermediate.
- **Division**: Knuth's Algorithm D with 128-bit trial quotient estimation (`(__uint128_t)un[j+n] << 64 | un[j+n-1]`).
- **String conversion**: repeated division by 10 with 128-bit remainder accumulation.

Bignums are immutable (arena-allocated). All arithmetic operations return new bignums. The tagged value stores the data pointer (after the 8-byte header) with type tag `ESHKOL_VALUE_HEAP_PTR` (8). Detection uses the header subtype:

```c
#define ESHKOL_IS_BIGNUM(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR && \
     (val).data.ptr_val != 0 && \
     ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_BIGNUM)
```

Codegen dispatch uses runtime functions (`eshkol_bignum_binary_tagged`, `eshkol_bignum_compare_tagged`, `eshkol_is_bignum_tagged`) via the alloca/store/call/load pattern. Operation codes for `eshkol_bignum_binary_tagged`: 0=add, 1=sub, 2=mul, 3=div, 4=mod, 5=quotient, 6=remainder, 7=neg. Exponentiation (`eshkol_bignum_pow`) uses repeated squaring for O(log n) multiplications. Bitwise operations follow R7RS two's complement semantics for negative bignums.

**Overflow promotion.** When int64 arithmetic overflows, `eshkol_bignum_from_overflow(arena, a, b, op)` computes the correct bignum result. The codegen detects overflow via LLVM intrinsics (e.g., `llvm.sadd.with.overflow.i64`) and branches to bignum promotion.

#### Rational (HEAP_SUBTYPE_RATIONAL = 19)

Rationals provide exact rational numbers as R7RS `numerator/denominator` pairs. They are always stored in reduced form with a positive denominator.

**Memory layout:** `[eshkol_object_header_t (8 bytes)][eshkol_rational_t]`

**Struct definition** (`inc/eshkol/core/rational.h`):

```c
typedef struct {
    int64_t numerator;
    int64_t denominator;
} eshkol_rational_t;
```

| Field         | Type      | Size   | Description                              |
|---------------|-----------|--------|------------------------------------------|
| `numerator`   | `int64_t` | 8 bytes | Numerator (may be negative)             |
| `denominator` | `int64_t` | 8 bytes | Denominator (always > 0)                |

Total allocation: `8 (header) + 16 (struct) = 24` bytes.

**GCD reduction invariant.** Every `eshkol_rational_create()` call normalizes the result:
1. If `denominator == 0`, fall back to `0/1`.
2. If `denominator < 0`, negate both numerator and denominator.
3. Compute `g = gcd(|numerator|, denominator)` and divide both by `g`.

Arithmetic operations (`add`, `sub`, `mul`, `div`) use 128-bit intermediate arithmetic (`__int128_t`) to avoid overflow during cross-multiplication, with a corresponding `gcd128()` for safe reduction. If the result does not fit in `int64_t` after GCD reduction, the operation falls back to double-precision floating-point.

**R7RS exactness rule.** The tagged dispatch function `eshkol_rational_binary_tagged` returns a rational (exact) if both operands are exact, and a double (inexact) if either operand is inexact. Mixed int/rational operands are handled by promoting the integer to rational form (`n/1`).

Rounding operations (`floor`, `ceil`, `truncate`, `round`) return exact `int64_t` results. The `rationalize` function implements R7RS `(rationalize x epsilon)` to find the simplest rational within epsilon of x.

### 4. Port Type Flag Encoding

Port types deviate from the standard type tag model. Rather than using a dedicated type tag value or a heap subtype, ports encode their directionality as **flag bits OR'd into the type byte** alongside `ESHKOL_VALUE_HEAP_PTR` (8).

**Flag definitions** (`inc/eshkol/eshkol.h`):

```c
#define ESHKOL_PORT_INPUT_FLAG    0x10   // Input port
#define ESHKOL_PORT_OUTPUT_FLAG   0x40   // Output port
#define ESHKOL_PORT_BINARY_FLAG   0x04   // Binary port (vs textual)
#define ESHKOL_PORT_ANY_FLAG      0x50   // Mask: input (0x10) | output (0x40)
```

**Type byte values:**

| Port Kind              | Type Byte Value                            | Hex  |
|------------------------|--------------------------------------------|------|
| Input port             | `ESHKOL_VALUE_HEAP_PTR \| ESHKOL_PORT_INPUT_FLAG`  | `0x18` |
| Output port            | `ESHKOL_VALUE_HEAP_PTR \| ESHKOL_PORT_OUTPUT_FLAG` | `0x48` |
| Input binary port      | `ESHKOL_VALUE_HEAP_PTR \| 0x10 \| 0x04`            | `0x1C` |
| Output binary port     | `ESHKOL_VALUE_HEAP_PTR \| 0x40 \| 0x04`            | `0x4C` |
| Input-output port      | `ESHKOL_VALUE_HEAP_PTR \| 0x10 \| 0x40`            | `0x58` |

**Detection pattern.** Because the type byte is not a simple equality check, port detection must use bitwise AND:

```c
// CORRECT: detect any port (input or output)
bool is_port = (type_tag & ESHKOL_PORT_ANY_FLAG) != 0;

// INCORRECT: this always returns false for ports
bool is_port = (type_tag == ESHKOL_VALUE_HEAP_PTR);  // BUG
```

This encoding was the root cause of a hanging-stdin bug in `readChar`/`peekChar`: the old code checked `type_tag == ESHKOL_VALUE_HEAP_PTR` (0x08), which never matched a port type byte like `0x18`, causing the fallback to stdin. The fix was `AND(type_tag, 0x50) != 0`.

The heap object itself still has `HEAP_SUBTYPE_PORT` (9) in its `eshkol_object_header_t`, so subtype-based dispatch (e.g., in the display system) works normally. The flag-bit encoding is specific to the *tagged value's type byte* and exists to enable fast port-direction checks without dereferencing the heap pointer.

### 5. Promise Type (HEAP_SUBTYPE_PROMISE = 18)

Promises implement R7RS `delay`/`force` with memoization. The parser desugars `(delay expr)` into a call to `%make-lazy-promise` with a thunk `(lambda () expr)` as its argument.

**Memory layout:** `[eshkol_object_header_t (8 bytes)][forced (8 bytes)][thunk (16 bytes)][cached (16 bytes)]`

| Offset | Field    | Type                     | Size     | Description                              |
|--------|----------|--------------------------|----------|------------------------------------------|
| 0      | `forced` | `int64_t`                | 8 bytes  | 0 = not yet forced, 1 = already forced  |
| 8      | `thunk`  | `eshkol_tagged_value_t`  | 16 bytes | Closure to evaluate on first force      |
| 24     | `cached` | `eshkol_tagged_value_t`  | 16 bytes | Memoized result (valid when forced = 1) |

Total allocation: `8 (header) + 40 (data) = 48` bytes.

Note that unlike the other consciousness engine types, the promise does not have a named C struct. Its layout is defined implicitly by the LLVM codegen in `llvm_codegen.cpp`, which uses GEP (GetElementPtr) with byte offsets to access the three fields.

**Force semantics.** When `(force promise)` is called:
1. Check that the value is `ESHKOL_VALUE_HEAP_PTR` with `HEAP_SUBTYPE_PROMISE` header.
2. Load the `forced` flag at offset 0.
3. If `forced != 0`: return the `cached` value at offset 24.
4. If `forced == 0`: invoke the thunk closure at offset 8, store the result into `cached` at offset 24, set `forced = 1`, and return the result.
5. If the argument is not a promise, return it as-is (R7RS: `force` on a non-promise is identity).

`(make-promise val)` creates an already-forced promise (`forced = 1`, `cached = val`), used for wrapping concrete values in the promise protocol.

---

## Implementation Source Files

The type system is implemented across several files:

| File | Purpose |
|------|---------|
| `inc/eshkol/types/hott_types.h` | HoTT foundation: universe levels (U0-UOmega), type flags (exact, linear, proof), supertype hierarchy, subtype checking with caching, type promotion rules |
| `lib/types/hott_types.cpp` | HoTT type system implementation |
| `inc/eshkol/types/dependent.h` | Dependent type support: CTValue (compile-time values for array dimensions, type-level naturals, booleans, symbolic expressions), DependentType for types parameterized by compile-time values |
| `lib/types/dependent.cpp` | Dependent type checking implementation (Phase 5 of HoTT) |
| `inc/eshkol/types/type_checker.h` | Bidirectional type checker interface |
| `lib/types/type_checker.cpp` | Type checker implementation: synthesis mode, checking mode, constraint generation, unification |

### Key Type System Concepts in Code

**Universe levels** (`hott_types.h`): The type hierarchy is organized into four universe levels:

- `U0` — Ground types: integer, float64, string, char, boolean
- `U1` — Type constructors: list, vector, function (->), pair (*), tensor, handle, buffer
- `U2` — Propositions: Eq, <:, Bounded, Linear (proof types, erased at runtime)
- `UOmega` — Universe polymorphic (for generic functions)

**Type flags** (`hott_types.h`): Additional properties carried on types:

- `TYPE_FLAG_EXACT` — Scheme exactness (integer vs inexact)
- `TYPE_FLAG_LINEAR` — Must use exactly once (linear types)
- `TYPE_FLAG_PROOF` — Compile-time only, erased at runtime

**Compile-time values** (`dependent.h`): The `CTValue` class represents values known at compile time, used for:

- Array dimension indices (e.g., the 100 in `Vector<Float64, 100>`)
- Type-level naturals for dependent types
- Compile-time boolean flags
- Symbolic expressions that may be reducible at compile time

CTValue kinds: `Nat` (uint64_t), `Bool`, `Expr` (AST reference), `Unknown` (runtime-only).

---

## See Also

- [Memory Management (OALR System)](MEMORY_MANAGEMENT.md) — Arena allocation, object headers, lifetimes
- [Vector Operations](VECTOR_OPERATIONS.md) — Scheme vectors vs. tensors, heterogeneous vs. homogeneous
- [Automatic Differentiation](AUTODIFF.md) — Dual numbers, AD nodes, computational graphs
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) — Type checking pipeline, LLVM codegen
- [Exact Arithmetic](EXACT_ARITHMETIC.md) — Bignums, rationals, numeric tower implementation
- [API Reference](../API_REFERENCE.md) — Complete function reference with types
