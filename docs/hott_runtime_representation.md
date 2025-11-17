# HoTT Runtime Representation with Type Erasure Optimization

## Overview

This document specifies the runtime representation for Eshkol's HoTT-based mixed type lists. The key innovation is complete **type erasure** at runtime while maintaining zero-cost abstractions and optimal memory layouts. All type safety is guaranteed at compile time through proof obligations.

## Core Principles

### 1. Proof Erasure at Runtime
- All proof terms are `constexpr` and exist only during compilation
- Runtime structures contain only essential data
- Type information encoded in template specializations, not runtime tags

### 2. Memory Layout Optimization
- Maintain compatibility with existing 16-byte cons cell layout for common cases
- Use template specialization for optimal layouts
- Arena allocation remains unchanged

### 3. Zero-Cost Abstractions
- No runtime overhead for type safety
- Template metaprogramming resolves all type operations at compile time
- LLVM IR generation uses specialized paths based on compile-time proofs

## Runtime Data Structures

### Erased Cons Cell Base

```cpp
// Minimal runtime representation - no type tags needed
namespace eshkol::hott::runtime {

// Base erased storage for cons cells
struct ErasedConsCell {
    // Raw storage - interpretation depends on compile-time type info
    alignas(8) uint64_t data[2]; // 16 bytes, same as current system
    
    static constexpr size_t size = 16;
    static constexpr size_t alignment = 8;
    
    // Default constructor
    constexpr ErasedConsCell() : data{0, 0} {}
    
    // Raw data access (used by template specializations)
    template<typename T>
    T* as() { 
        static_assert(sizeof(T) <= sizeof(data));
        return reinterpret_cast<T*>(data); 
    }
    
    template<typename T>
    const T* as() const { 
        static_assert(sizeof(T) <= sizeof(data));
        return reinterpret_cast<const T*>(data); 
    }
};

} // namespace eshkol::hott::runtime
```

### Template Specializations for Common Cases

```cpp
// Optimized specializations for homogeneous types
namespace eshkol::hott {

// Homogeneous integer cons cell - identical to current representation
template<>
struct HottConsCell<TypeCode::Int64, TypeCode::Int64> {
    int64_t car_data;
    int64_t cdr_data;
    
    // Compile-time metadata
    static constexpr TypeCode car_code = TypeCode::Int64;
    static constexpr TypeCode cdr_code = TypeCode::Int64;
    static constexpr bool is_homogeneous = true;
    static constexpr bool is_optimized = true;
    static constexpr size_t size = 16;
    
    // Constructor
    constexpr HottConsCell(int64_t car, int64_t cdr) 
        : car_data(car), cdr_data(cdr) {}
    
    // Runtime conversion to/from erased form
    runtime::ErasedConsCell to_erased() const {
        runtime::ErasedConsCell result;
        *result.as<HottConsCell>() = *this;
        return result;
    }
    
    static HottConsCell from_erased(const runtime::ErasedConsCell& erased) {
        return *erased.as<HottConsCell>();
    }
};

// Homogeneous double cons cell
template<>
struct HottConsCell<TypeCode::Double, TypeCode::Double> {
    double car_data;
    double cdr_data;
    
    static constexpr TypeCode car_code = TypeCode::Double;
    static constexpr TypeCode cdr_code = TypeCode::Double;
    static constexpr bool is_homogeneous = true;
    static constexpr bool is_optimized = true;
    static constexpr size_t size = 16;
    
    constexpr HottConsCell(double car, double cdr) 
        : car_data(car), cdr_data(cdr) {}
    
    runtime::ErasedConsCell to_erased() const {
        runtime::ErasedConsCell result;
        *result.as<HottConsCell>() = *this;
        return result;
    }
    
    static HottConsCell from_erased(const runtime::ErasedConsCell& erased) {
        return *erased.as<HottConsCell>();
    }
};

// Mixed type cons cell (int64, double)
template<>
struct HottConsCell<TypeCode::Int64, TypeCode::Double> {
    int64_t car_data;
    double cdr_data;
    
    static constexpr TypeCode car_code = TypeCode::Int64;
    static constexpr TypeCode cdr_code = TypeCode::Double;
    static constexpr bool is_homogeneous = false;
    static constexpr bool is_optimized = true;
    static constexpr size_t size = 16;
    
    constexpr HottConsCell(int64_t car, double cdr) 
        : car_data(car), cdr_data(cdr) {}
    
    runtime::ErasedConsCell to_erased() const {
        runtime::ErasedConsCell result;
        *result.as<HottConsCell>() = *this;
        return result;
    }
    
    static HottConsCell from_erased(const runtime::ErasedConsCell& erased) {
        return *erased.as<HottConsCell>();
    }
};

} // namespace eshkol::hott
```

### Dynamic Type Handling for General Cases

```cpp
// For cases that don't fit in optimized layouts
namespace eshkol::hott {

// General cons cell with indirection for large types
template<TypeCode CarCode, TypeCode CdrCode>
struct HottConsCell {
private:
    using CarType = Interpret<CarCode>;
    using CdrType = Interpret<CdrCode>;
    
    static constexpr bool car_fits = sizeof(CarType) <= 8;
    static constexpr bool cdr_fits = sizeof(CdrType) <= 8;
    static constexpr bool both_fit = car_fits && cdr_fits;

public:
    static constexpr TypeCode car_code = CarCode;
    static constexpr TypeCode cdr_code = CdrCode;
    static constexpr bool is_optimized = both_fit;
    
    // Storage strategy based on size
    std::conditional_t<both_fit, 
        DirectStorage<CarType, CdrType>,
        IndirectStorage<CarType, CdrType>
    > storage;
    
    // Unified interface regardless of storage strategy
    CarType car() const { return storage.get_car(); }
    CdrType cdr() const { return storage.get_cdr(); }
    
    void set_car(const CarType& val) { storage.set_car(val); }
    void set_cdr(const CdrType& val) { storage.set_cdr(val); }
};

// Direct storage for small types (fits in 16 bytes)
template<typename CarType, typename CdrType>
struct DirectStorage {
    CarType car_data;
    CdrType cdr_data;
    
    CarType get_car() const { return car_data; }
    CdrType get_cdr() const { return cdr_data; }
    
    void set_car(const CarType& val) { car_data = val; }
    void set_cdr(const CdrType& val) { cdr_data = val; }
};

// Indirect storage for large types (uses arena allocation)
template<typename CarType, typename CdrType>
struct IndirectStorage {
    CarType* car_ptr;
    CdrType* cdr_ptr;
    
    CarType get_car() const { return *car_ptr; }
    CdrType get_cdr() const { return *cdr_ptr; }
    
    void set_car(const CarType& val) { *car_ptr = val; }
    void set_cdr(const CdrType& val) { *cdr_ptr = val; }
};

} // namespace eshkol::hott
```

## Arena Integration

### Type-Aware Arena Allocation

```cpp
namespace eshkol::hott {

// Enhanced arena with type-aware allocation
class TypeAwareArena {
private:
    arena_t* base_arena; // Existing arena implementation
    
public:
    explicit TypeAwareArena(size_t default_block_size = 8192) 
        : base_arena(arena_create(default_block_size)) {}
    
    ~TypeAwareArena() {
        if (base_arena) arena_destroy(base_arena);
    }
    
    // Template-based allocation with compile-time size determination
    template<TypeCode CarCode, TypeCode CdrCode>
    HottConsCell<CarCode, CdrCode>* allocate_cons() {
        using ConsType = HottConsCell<CarCode, CdrCode>;
        
        if constexpr (ConsType::is_optimized) {
            // Use existing fast path for 16-byte cells
            static_assert(sizeof(ConsType) == 16);
            return static_cast<ConsType*>(
                arena_allocate_aligned(base_arena, sizeof(ConsType), alignof(ConsType))
            );
        } else {
            // Handle larger cells with potential indirection
            return allocate_general_cons<CarCode, CdrCode>();
        }
    }
    
    // Batch allocation for homogeneous lists
    template<TypeCode Code>
    HottConsCell<Code, TypeCode::Null>* allocate_homogeneous_batch(size_t count) {
        using ConsType = HottConsCell<Code, TypeCode::Null>;
        size_t total_size = sizeof(ConsType) * count;
        
        return static_cast<ConsType*>(
            arena_allocate_aligned(base_arena, total_size, alignof(ConsType))
        );
    }

private:
    template<TypeCode CarCode, TypeCode CdrCode>
    HottConsCell<CarCode, CdrCode>* allocate_general_cons() {
        using ConsType = HottConsCell<CarCode, CdrCode>;
        using CarType = Interpret<CarCode>;
        using CdrType = Interpret<CdrCode>;
        
        if constexpr (sizeof(CarType) + sizeof(CdrType) <= 16) {
            // Fits in direct storage
            return static_cast<ConsType*>(
                arena_allocate_aligned(base_arena, sizeof(ConsType), alignof(ConsType))
            );
        } else {
            // Requires indirect storage
            auto* cons = static_cast<ConsType*>(
                arena_allocate_aligned(base_arena, sizeof(ConsType), alignof(ConsType))
            );
            
            // Allocate space for large car/cdr values
            if constexpr (sizeof(CarType) > 8) {
                cons->storage.car_ptr = static_cast<CarType*>(
                    arena_allocate_aligned(base_arena, sizeof(CarType), alignof(CarType))
                );
            }
            
            if constexpr (sizeof(CdrType) > 8) {
                cons->storage.cdr_ptr = static_cast<CdrType*>(
                    arena_allocate_aligned(base_arena, sizeof(CdrType), alignof(CdrType))
                );
            }
            
            return cons;
        }
    }
};

} // namespace eshkol::hott
```

### Allocation Strategy Selection

```cpp
// Compile-time allocation strategy selection
namespace eshkol::hott {

template<TypeCode CarCode, TypeCode CdrCode>
struct AllocationStrategy {
private:
    using CarType = Interpret<CarCode>;
    using CdrType = Interpret<CdrCode>;
    using ConsType = HottConsCell<CarCode, CdrCode>;
    
    static constexpr bool is_small = sizeof(ConsType) <= 16;
    static constexpr bool is_aligned = alignof(ConsType) <= 8;
    static constexpr bool is_pod = std::is_trivially_copyable_v<ConsType>;

public:
    enum class Strategy {
        OPTIMIZED,    // 16-byte aligned, existing arena path
        STANDARD,     // Standard arena allocation
        INDIRECT      // Requires indirection for large types
    };
    
    static constexpr Strategy strategy = 
        (is_small && is_aligned && is_pod) ? Strategy::OPTIMIZED :
        (is_pod) ? Strategy::STANDARD :
        Strategy::INDIRECT;
    
    // Allocation function selection
    template<typename Arena>
    static ConsType* allocate(Arena& arena) {
        if constexpr (strategy == Strategy::OPTIMIZED) {
            return allocate_optimized(arena);
        } else if constexpr (strategy == Strategy::STANDARD) {
            return allocate_standard(arena);
        } else {
            return allocate_indirect(arena);
        }
    }

private:
    template<typename Arena>
    static ConsType* allocate_optimized(Arena& arena) {
        // Fast path - reuse existing arena_allocate_cons_cell
        static_assert(sizeof(ConsType) == 16);
        return reinterpret_cast<ConsType*>(arena_allocate_cons_cell(arena.get_arena()));
    }
    
    template<typename Arena>
    static ConsType* allocate_standard(Arena& arena) {
        return arena.template allocate<ConsType>();
    }
    
    template<typename Arena>
    static ConsType* allocate_indirect(Arena& arena) {
        return arena.template allocate_general_cons<CarCode, CdrCode>();
    }
};

} // namespace eshkol::hott
```

## Performance Optimizations

### Template Specialization for LLVM Types

```cpp
// Map HoTT types to LLVM types at compile time
namespace eshkol::hott::llvm_integration {

template<TypeCode Code>
struct LLVMTypeMapping;

template<>
struct LLVMTypeMapping<TypeCode::Int64> {
    static llvm::Type* get(llvm::LLVMContext& ctx) {
        return llvm::Type::getInt64Ty(ctx);
    }
    
    static llvm::Value* create_constant(llvm::LLVMContext& ctx, int64_t value) {
        return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), value);
    }
    
    static constexpr bool is_primitive = true;
    static constexpr bool needs_boxing = false;
};

template<>
struct LLVMTypeMapping<TypeCode::Double> {
    static llvm::Type* get(llvm::LLVMContext& ctx) {
        return llvm::Type::getDoubleTy(ctx);
    }
    
    static llvm::Value* create_constant(llvm::LLVMContext& ctx, double value) {
        return llvm::ConstantFP::get(llvm::Type::getDoubleTy(ctx), value);
    }
    
    static constexpr bool is_primitive = true;
    static constexpr bool needs_boxing = false;
};

// Cons cell LLVM type generation
template<TypeCode CarCode, TypeCode CdrCode>
struct ConsLLVMType {
    static llvm::Type* get(llvm::LLVMContext& ctx) {
        using ConsType = HottConsCell<CarCode, CdrCode>;
        
        if constexpr (ConsType::is_optimized) {
            // Direct struct type
            return llvm::StructType::get(ctx, {
                LLVMTypeMapping<CarCode>::get(ctx),
                LLVMTypeMapping<CdrCode>::get(ctx)
            });
        } else {
            // Indirect representation
            return llvm::StructType::get(ctx, {
                LLVMTypeMapping<CarCode>::get(ctx)->getPointerTo(),
                LLVMTypeMapping<CdrCode>::get(ctx)->getPointerTo()
            });
        }
    }
    
    static constexpr bool is_optimized = HottConsCell<CarCode, CdrCode>::is_optimized;
};

} // namespace eshkol::hott::llvm_integration
```

### Memory Access Optimization

```cpp
// Optimized memory access patterns
namespace eshkol::hott {

// Cache-friendly list traversal
template<TypeCode... Codes>
class OptimizedListTraversal {
private:
    static constexpr bool all_homogeneous = sizeof...(Codes) > 0 && 
        std::conjunction_v<std::is_same<
            std::tuple_element_t<0, std::tuple<std::integral_constant<TypeCode, Codes>...>>,
            std::integral_constant<TypeCode, Codes>
        >...>;
        
    static constexpr TypeCode uniform_code = 
        all_homogeneous ? std::get<0>(std::tuple<std::integral_constant<TypeCode, Codes>...>{}).value : TypeCode::Mixed;

public:
    // Specialized traversal for homogeneous lists
    template<typename Function>
    static void traverse(const HList<Codes...>& list, Function&& f) {
        if constexpr (all_homogeneous && sizeof...(Codes) > 4) {
            // Use vectorized traversal for large homogeneous lists
            traverse_vectorized(list, std::forward<Function>(f));
        } else {
            // Standard traversal
            traverse_standard(list, std::forward<Function>(f));
        }
    }

private:
    template<typename Function>
    static void traverse_vectorized(const HList<Codes...>& list, Function&& f) {
        // Implement SIMD-friendly traversal for homogeneous numeric types
        static_assert(all_homogeneous);
        
        if constexpr (uniform_code == TypeCode::Int64 || uniform_code == TypeCode::Double) {
            // Process in chunks for better cache utilization
            constexpr size_t chunk_size = 64 / sizeof(Interpret<uniform_code>);
            // ... vectorized implementation
        }
    }
    
    template<typename Function>
    static void traverse_standard(const HList<Codes...>& list, Function&& f) {
        // Standard element-by-element traversal
        std::apply([&f](const auto&... elements) {
            (f(elements), ...);
        }, list.data);
    }
};

} // namespace eshkol::hott
```

### Zero-Copy Conversions

```cpp
// Zero-copy conversions between compatible representations
namespace eshkol::hott {

template<typename From, typename To>
struct ZeroCopyConverter {
    static constexpr bool is_convertible = 
        sizeof(From) == sizeof(To) && 
        alignof(From) == alignof(To) &&
        std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>;
    
    static To convert(const From& from) {
        if constexpr (is_convertible) {
            // Bit-cast conversion without copying
            return std::bit_cast<To>(from);
        } else {
            // Fallback to explicit conversion
            return To{from};
        }
    }
};

// Specialized conversions for common cases
template<>
struct ZeroCopyConverter<runtime::ErasedConsCell, HottConsCell<TypeCode::Int64, TypeCode::Int64>> {
    static constexpr bool is_convertible = true;
    
    static HottConsCell<TypeCode::Int64, TypeCode::Int64> convert(const runtime::ErasedConsCell& erased) {
        return HottConsCell<TypeCode::Int64, TypeCode::Int64>::from_erased(erased);
    }
};

} // namespace eshkol::hott
```

## Integration with Existing System

### Compatibility Layer

```cpp
// Compatibility with existing tagged union system
namespace eshkol::hott::compat {

// Convert from old tagged cons cell to HoTT representation
template<TypeCode CarCode, TypeCode CdrCode>
HottConsCell<CarCode, CdrCode> from_tagged_cons(const arena_tagged_cons_cell_t& tagged) {
    using CarType = Interpret<CarCode>;
    using CdrType = Interpret<CdrCode>;
    
    // Runtime type checking for safety during migration
    assert(tagged.car_type == static_cast<uint8_t>(CarCode));
    assert(tagged.cdr_type == static_cast<uint8_t>(CdrCode));
    
    CarType car_val;
    CdrType cdr_val;
    
    // Extract values based on type codes
    if constexpr (CarCode == TypeCode::Int64) {
        car_val = tagged.car_data.int_val;
    } else if constexpr (CarCode == TypeCode::Double) {
        car_val = tagged.car_data.double_val;
    }
    
    if constexpr (CdrCode == TypeCode::Int64) {
        cdr_val = tagged.cdr_data.int_val;
    } else if constexpr (CdrCode == TypeCode::Double) {
        cdr_val = tagged.cdr_data.double_val;
    }
    
    return HottConsCell<CarCode, CdrCode>{car_val, cdr_val};
}

// Convert from HoTT representation to old tagged cons cell
template<TypeCode CarCode, TypeCode CdrCode>
arena_tagged_cons_cell_t to_tagged_cons(const HottConsCell<CarCode, CdrCode>& hott_cons) {
    arena_tagged_cons_cell_t tagged{};
    
    tagged.car_type = static_cast<uint8_t>(CarCode);
    tagged.cdr_type = static_cast<uint8_t>(CdrCode);
    
    // Store values in union based on type codes
    if constexpr (CarCode == TypeCode::Int64) {
        tagged.car_data.int_val = hott_cons.car();
    } else if constexpr (CarCode == TypeCode::Double) {
        tagged.car_data.double_val = hott_cons.car();
    }
    
    if constexpr (CdrCode == TypeCode::Int64) {
        tagged.cdr_data.int_val = hott_cons.cdr();
    } else if constexpr (CdrCode == TypeCode::Double) {
        tagged.cdr_data.double_val = hott_cons.cdr();
    }
    
    return tagged;
}

} // namespace eshkol::hott::compat
```

### Migration Strategy

```cpp
// Gradual migration from tagged unions to HoTT representation
namespace eshkol::hott::migration {

// Feature flag for enabling HoTT representation
#ifndef ESHKOL_ENABLE_HOTT
#define ESHKOL_ENABLE_HOTT 0
#endif

// Conditional type aliases
#if ESHKOL_ENABLE_HOTT
template<TypeCode CarCode, TypeCode CdrCode>
using MigrationConsCell = HottConsCell<CarCode, CdrCode>;
#else
using MigrationConsCell = arena_tagged_cons_cell_t;
#endif

// Runtime detection of representation type
enum class RepresentationType {
    TAGGED_UNION,
    HOTT_TYPED
};

struct RuntimeTypeInfo {
    RepresentationType representation;
    TypeCode car_code;
    TypeCode cdr_code;
    
    // Factory function for creating appropriate representation
    template<typename CarType, typename CdrType>
    static RuntimeTypeInfo create() {
        constexpr TypeCode car_code = type_code_for<CarType>();
        constexpr TypeCode cdr_code = type_code_for<CdrType>();
        
        return RuntimeTypeInfo{
            ESHKOL_ENABLE_HOTT ? RepresentationType::HOTT_TYPED : RepresentationType::TAGGED_UNION,
            car_code,
            cdr_code
        };
    }
};

} // namespace eshkol::hott::migration
```

## Performance Characteristics

### Memory Usage

| Representation | Homogeneous Int64 | Homogeneous Double | Mixed (Int64, Double) | General Case |
|----------------|-------------------|--------------------|-----------------------|--------------|
| Current Tagged | 24 bytes          | 24 bytes           | 24 bytes              | 24 bytes     |
| HoTT Optimized | 16 bytes          | 16 bytes           | 16 bytes              | Variable     |
| Memory Savings | 33%               | 33%                | 33%                   | Varies       |

### Runtime Performance

- **Type Checking**: Zero cost (compile-time only)
- **Memory Access**: Direct pointer dereference (no indirection for optimized cases)
- **Cache Performance**: Better due to smaller memory footprint
- **Vectorization**: Enabled for homogeneous lists

### Compile-time Performance

- **Template Instantiation**: O(1) per unique type combination
- **Proof Checking**: O(n) in AST depth
- **Code Generation**: Specialized functions reduce binary size

This runtime representation achieves the goals of:
1. **Zero runtime type checking overhead**
2. **Optimal memory layouts for common cases** 
3. **Compatibility with existing arena system**
4. **Performance improvements through specialization**

The system provides a smooth migration path while enabling significant performance optimizations through compile-time type information and proof erasure.