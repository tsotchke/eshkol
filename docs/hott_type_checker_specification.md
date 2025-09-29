# HoTT Type Checker with Proof Obligation Generation

## Overview

This document specifies the compile-time type checker for Eshkol's HoTT-based mixed type lists. The type checker generates and validates proof obligations at compile time, ensuring type safety while enabling aggressive optimizations through proof erasure at runtime.

## Core Architecture

### Template Metaprogramming Foundation

```cpp
// Core template metaprogramming infrastructure
namespace eshkol::hott {

// Type-level natural numbers for universe levels
template<unsigned N>
struct Nat {
    static constexpr unsigned value = N;
    using type = Nat<N>;
};

// Type-level lists for type sequences
template<typename...>
struct TypeList {};

template<typename T, typename... Ts>
struct TypeList<T, Ts...> {
    using head = T;
    using tail = TypeList<Ts...>;
    static constexpr size_t size = 1 + sizeof...(Ts);
};

// Type-level booleans for proof validation
struct True {
    static constexpr bool value = true;
    using type = True;
};

struct False {
    static constexpr bool value = false;
    using type = False;
};

} // namespace eshkol::hott
```

### Universe Level System

```cpp
// Universe hierarchy with compile-time validation
template<unsigned Level>
struct Universe {
    static constexpr unsigned level = Level;
    
    template<typename T>
    static constexpr bool contains() {
        return T::universe_level <= Level;
    }
};

// Type codes with universe membership proofs
enum class TypeCode : uint8_t {
    Int64 = 0,
    Double = 1,
    String = 2,
    Null = 3,
    List = 4,
    Mixed = 5
};

template<TypeCode Code>
struct TypeInfo;

template<>
struct TypeInfo<TypeCode::Int64> {
    using cpp_type = int64_t;
    static constexpr unsigned universe_level = 0;
    static constexpr bool is_numeric = true;
    static constexpr bool is_exact = true;
    static constexpr size_t size = sizeof(int64_t);
    static constexpr size_t alignment = alignof(int64_t);
};

template<>
struct TypeInfo<TypeCode::Double> {
    using cpp_type = double;
    static constexpr unsigned universe_level = 0;
    static constexpr bool is_numeric = true;
    static constexpr bool is_exact = false;
    static constexpr size_t size = sizeof(double);
    static constexpr size_t alignment = alignof(double);
};

// Type interpretation function ⟦_⟧
template<TypeCode Code>
using Interpret = typename TypeInfo<Code>::cpp_type;
```

### Dependent Type Encoding

```cpp
// Heterogeneous list with type-level length tracking
template<TypeCode... Codes>
struct HList {
    static constexpr size_t length = sizeof...(Codes);
    using type_sequence = TypeList<Interpret<Codes>...>;
    
    // Storage with perfect forwarding
    std::tuple<Interpret<Codes>...> data;
    
    template<size_t N>
    auto get() const -> std::tuple_element_t<N, std::tuple<Interpret<Codes>...>> {
        return std::get<N>(data);
    }
    
    template<size_t N>
    auto get() -> std::tuple_element_t<N, std::tuple<Interpret<Codes>...>>& {
        return std::get<N>(data);
    }
};

// Cons cell with dependent typing
template<TypeCode CarCode, TypeCode CdrCode>
struct HottConsCell {
    using car_type = Interpret<CarCode>;
    using cdr_type = Interpret<CdrCode>;
    
    // Type-level metadata
    static constexpr TypeCode car_code = CarCode;
    static constexpr TypeCode cdr_code = CdrCode;
    static constexpr unsigned universe_level = 
        std::max(TypeInfo<CarCode>::universe_level, 
                 TypeInfo<CdrCode>::universe_level) + 1;
    
    // Runtime data
    car_type car_data;
    cdr_type cdr_data;
    
    // Constructors with perfect forwarding
    template<typename CarArg, typename CdrArg>
    HottConsCell(CarArg&& car, CdrArg&& cdr) 
        : car_data(std::forward<CarArg>(car))
        , cdr_data(std::forward<CdrArg>(cdr)) {}
};
```

### Proof Term Infrastructure

```cpp
// Base proof term structure
template<typename ProofType, bool Valid>
struct ProofTerm {
    using proof_type = ProofType;
    static constexpr bool is_valid = Valid;
    static constexpr bool is_erased = true; // Proofs erased at runtime
};

// Type safety proof for cons cells
template<TypeCode CarCode, TypeCode CdrCode>
struct TypeSafetyProof {
private:
    static constexpr bool car_valid = TypeInfo<CarCode>::universe_level >= 0;
    static constexpr bool cdr_valid = TypeInfo<CdrCode>::universe_level >= 0;
    static constexpr bool size_valid = 
        TypeInfo<CarCode>::size + TypeInfo<CdrCode>::size <= 24; // Arena limit
    
public:
    static constexpr bool is_valid = car_valid && cdr_valid && size_valid;
    
    using car_type = Interpret<CarCode>;
    using cdr_type = Interpret<CdrCode>;
    
    // Proof obligations
    static_assert(car_valid, "Car type must be in valid universe");
    static_assert(cdr_valid, "Cdr type must be in valid universe");
    static_assert(size_valid, "Cons cell size exceeds arena allocation limit");
};

// Arithmetic operation proof
template<TypeCode Left, TypeCode Right>
struct ArithmeticProof {
private:
    static constexpr bool both_numeric = 
        TypeInfo<Left>::is_numeric && TypeInfo<Right>::is_numeric;
    
    // Result type calculation following Scheme numeric tower
    static constexpr TypeCode result_code = 
        (Left == TypeCode::Double || Right == TypeCode::Double) 
            ? TypeCode::Double 
            : TypeCode::Int64;
    
    static constexpr bool result_exact = 
        TypeInfo<Left>::is_exact && TypeInfo<Right>::is_exact && 
        result_code == TypeCode::Int64;

public:
    static constexpr bool is_valid = both_numeric;
    static constexpr TypeCode result_type = result_code;
    static constexpr bool is_exact = result_exact;
    
    using left_type = Interpret<Left>;
    using right_type = Interpret<Right>;
    using result_cpp_type = Interpret<result_code>;
    
    static_assert(is_valid, "Arithmetic requires numeric types");
};
```

### Type Inference Engine

```cpp
// Forward declaration for AST types
struct ASTNode;
struct ConsExpr;
struct ArithmeticExpr;
struct SymbolRef;
struct Literal;

// Type inference results
template<typename T>
struct InferenceResult {
    using type = T;
    static constexpr TypeCode type_code = /* computed based on T */;
    static constexpr bool is_inferrable = true;
};

// Main type inference engine
template<typename ASTNodeType>
struct TypeInferenceEngine;

template<>
struct TypeInferenceEngine<Literal<int64_t>> {
    using result_type = int64_t;
    static constexpr TypeCode inferred_code = TypeCode::Int64;
    static constexpr bool is_exact = true;
    
    using proof = ProofTerm<TypeSafetyProof<TypeCode::Int64, TypeCode::Null>, true>;
};

template<>
struct TypeInferenceEngine<Literal<double>> {
    using result_type = double;
    static constexpr TypeCode inferred_code = TypeCode::Double;
    static constexpr bool is_exact = false;
    
    using proof = ProofTerm<TypeSafetyProof<TypeCode::Double, TypeCode::Null>, true>;
};

template<typename CarAST, typename CdrAST>
struct TypeInferenceEngine<ConsExpr<CarAST, CdrAST>> {
private:
    using car_inference = TypeInferenceEngine<CarAST>;
    using cdr_inference = TypeInferenceEngine<CdrAST>;
    
    static constexpr TypeCode car_code = car_inference::inferred_code;
    static constexpr TypeCode cdr_code = cdr_inference::inferred_code;

public:
    using result_type = HottConsCell<car_code, cdr_code>;
    static constexpr TypeCode inferred_code = TypeCode::Mixed;
    
    using safety_proof = TypeSafetyProof<car_code, cdr_code>;
    using proof = ProofTerm<safety_proof, safety_proof::is_valid>;
    
    static_assert(proof::is_valid, "Cons expression type safety proof failed");
};

template<typename LeftAST, typename RightAST, typename Op>
struct TypeInferenceEngine<ArithmeticExpr<LeftAST, RightAST, Op>> {
private:
    using left_inference = TypeInferenceEngine<LeftAST>;
    using right_inference = TypeInferenceEngine<RightAST>;
    
    static constexpr TypeCode left_code = left_inference::inferred_code;
    static constexpr TypeCode right_code = right_inference::inferred_code;

public:
    using arith_proof = ArithmeticProof<left_code, right_code>;
    using result_type = typename arith_proof::result_cpp_type;
    static constexpr TypeCode inferred_code = arith_proof::result_type;
    
    using proof = ProofTerm<arith_proof, arith_proof::is_valid>;
    
    static_assert(proof::is_valid, "Arithmetic expression type proof failed");
};
```

### Proof Obligation Generator

```cpp
// Proof obligation collection and validation
template<typename AST>
struct ProofObligationCollector {
    using type = typename TypeInferenceEngine<AST>::proof;
    
    static constexpr bool all_obligations_satisfied = type::is_valid;
    
    // Recursive collection for compound expressions
    template<typename... SubASTs>
    struct CollectMultiple {
        using obligations = TypeList<typename TypeInferenceEngine<SubASTs>::proof...>;
        static constexpr bool all_valid = (TypeInferenceEngine<SubASTs>::proof::is_valid && ...);
    };
};

// Proof discharge mechanism
template<typename ProofObligation>
struct ProofDischarge {
    static constexpr bool can_discharge = ProofObligation::is_valid;
    
    // Generate compile-time error if proof cannot be discharged
    template<bool = can_discharge>
    struct Validate {
        static_assert(can_discharge, "Cannot discharge proof obligation");
        using type = ProofObligation;
    };
    
    using discharged_proof = typename Validate<>::type;
};

// Main type checking interface
template<typename AST>
class HottTypeChecker {
private:
    using inference_result = TypeInferenceEngine<AST>;
    using proof_obligations = ProofObligationCollector<AST>;
    using discharged = ProofDischarge<typename proof_obligations::type>;

public:
    using inferred_type = typename inference_result::result_type;
    static constexpr TypeCode inferred_code = inference_result::inferred_code;
    
    static constexpr bool is_well_typed = proof_obligations::all_obligations_satisfied;
    
    // Compile-time type checking
    static constexpr void check() {
        static_assert(is_well_typed, "Type checking failed - proof obligations not satisfied");
    }
    
    // Generate optimized code based on proof properties
    template<typename CodeGen>
    static auto generate_code(CodeGen& generator) {
        if constexpr (is_well_typed) {
            return generator.template generate_optimized<inferred_type, 
                                                        typename discharged::discharged_proof>();
        } else {
            return generator.template generate_with_checks<AST>();
        }
    }
};
```

### Integration with Parser

```cpp
// Enhanced AST nodes with type information
template<typename ValueType>
struct TypedLiteral {
    ValueType value;
    static constexpr TypeCode type_code = /* derive from ValueType */;
    
    using type_info = TypeInfo<type_code>;
    using inference = TypeInferenceEngine<TypedLiteral<ValueType>>;
};

template<typename CarExpr, typename CdrExpr>
struct TypedConsExpr {
    CarExpr car_expr;
    CdrExpr cdr_expr;
    
    using car_inference = TypeInferenceEngine<CarExpr>;
    using cdr_inference = TypeInferenceEngine<CdrExpr>;
    using inference = TypeInferenceEngine<TypedConsExpr<CarExpr, CdrExpr>>;
    
    static constexpr TypeCode car_code = car_inference::inferred_code;
    static constexpr TypeCode cdr_code = cdr_inference::inferred_code;
};

// Enhanced parser that builds typed AST
class HottAwareParser {
private:
    template<typename TokenStream>
    auto parse_literal(TokenStream& tokens) -> auto {
        auto token = tokens.current();
        
        if (token.type == TokenType::NUMBER) {
            if (token.value.find('.') != std::string::npos) {
                double value = std::stod(token.value);
                return TypedLiteral<double>{value};
            } else {
                int64_t value = std::stoll(token.value);
                return TypedLiteral<int64_t>{value};
            }
        }
        
        // Handle other literal types...
    }
    
    template<typename TokenStream>
    auto parse_cons(TokenStream& tokens) -> auto {
        tokens.consume("(");
        tokens.consume("cons");
        
        auto car_expr = parse_expression(tokens);
        auto cdr_expr = parse_expression(tokens);
        
        tokens.consume(")");
        
        using CarType = decltype(car_expr);
        using CdrType = decltype(cdr_expr);
        
        return TypedConsExpr<CarType, CdrType>{
            std::move(car_expr), 
            std::move(cdr_expr)
        };
    }

public:
    template<typename TokenStream>
    auto parse_expression(TokenStream& tokens) -> auto {
        // Dispatch based on token type to build typed AST
        // Each parse method returns a typed AST node
    }
};
```

### Proof-Directed Optimization

```cpp
// Optimization strategies based on proof properties
template<typename AST, typename Proof>
struct OptimizationStrategy {
    static constexpr bool can_eliminate_checks = Proof::is_valid;
    static constexpr bool can_specialize = /* based on proof properties */;
    static constexpr bool can_inline = /* based on proof properties */;
};

// Proof-aware code generator
template<typename IRBuilder>
class ProofDirectedCodeGen {
private:
    IRBuilder& builder;
    
public:
    explicit ProofDirectedCodeGen(IRBuilder& b) : builder(b) {}
    
    // Generate cons allocation with proof-based optimization
    template<typename CarType, typename CdrType, typename SafetyProof>
    auto generate_cons_alloc() -> llvm::Value* {
        using cons_type = HottConsCell</* derive from types */>;
        using optimization = OptimizationStrategy<cons_type, SafetyProof>;
        
        if constexpr (SafetyProof::is_valid && optimization::can_specialize) {
            // Generate specialized allocation without runtime checks
            return generate_specialized_cons<CarType, CdrType>();
        } else {
            // Generate with runtime safety checks
            return generate_checked_cons<CarType, CdrType>();
        }
    }
    
    // Generate arithmetic with type promotion proofs
    template<typename LeftType, typename RightType, typename ArithProof>
    auto generate_arithmetic(llvm::Value* lhs, llvm::Value* rhs) -> llvm::Value* {
        if constexpr (ArithProof::is_valid) {
            using result_type = typename ArithProof::result_cpp_type;
            
            if constexpr (std::is_same_v<result_type, int64_t>) {
                return builder.CreateAdd(lhs, rhs);
            } else if constexpr (std::is_same_v<result_type, double>) {
                // Handle promotion if needed
                auto lhs_promoted = promote_if_needed<LeftType, double>(lhs);
                auto rhs_promoted = promote_if_needed<RightType, double>(rhs);
                return builder.CreateFAdd(lhs_promoted, rhs_promoted);
            }
        } else {
            // Generate with runtime type checking
            return generate_runtime_checked_arithmetic(lhs, rhs);
        }
    }

private:
    template<typename CarType, typename CdrType>
    auto generate_specialized_cons() -> llvm::Value* {
        // Specialized cons cell allocation without type tags
        auto* cons_type = llvm::StructType::get(
            builder.getContext(),
            {llvm_type_for<CarType>(), llvm_type_for<CdrType>()}
        );
        
        return builder.CreateAlloca(cons_type);
    }
    
    template<typename From, typename To>
    auto promote_if_needed(llvm::Value* value) -> llvm::Value* {
        if constexpr (!std::is_same_v<From, To>) {
            if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, double>) {
                return builder.CreateSIToFP(value, builder.getDoubleTy());
            }
        }
        return value;
    }
};
```

### Type Checking Workflow

```cpp
// Main type checking and code generation workflow
template<typename SourceAST>
class HottCompilationPipeline {
public:
    static auto compile() {
        // Phase 1: Type inference and proof generation
        using type_checker = HottTypeChecker<SourceAST>;
        type_checker::check(); // Compile-time validation
        
        // Phase 2: Proof discharge
        constexpr bool well_typed = type_checker::is_well_typed;
        static_assert(well_typed, "Compilation failed: type checking errors");
        
        // Phase 3: Proof-directed optimization
        using optimization = OptimizationStrategy<SourceAST, 
                                                  typename type_checker::discharged>;
        
        // Phase 4: Code generation
        return [](auto& code_generator) {
            return type_checker::generate_code(code_generator);
        };
    }
    
    // Integration with existing LLVM pipeline
    static void integrate_with_llvm(llvm::Module* module) {
        auto compilation_result = compile();
        
        llvm::IRBuilder<> builder(module->getContext());
        ProofDirectedCodeGen codegen(builder);
        
        compilation_result(codegen);
    }
};
```

## Usage Examples

### Basic Type Checking

```cpp
// Example: Type checking a simple cons expression
auto cons_expr = TypedConsExpr{
    TypedLiteral<int64_t>{42},
    TypedLiteral<double>{3.14}
};

using checker = HottTypeChecker<decltype(cons_expr)>;
checker::check(); // Validates at compile time

static_assert(checker::is_well_typed);
static_assert(checker::inferred_code == TypeCode::Mixed);
```

### Arithmetic with Proofs

```cpp
// Example: Arithmetic operation with type promotion
auto arith_expr = TypedArithmeticExpr<AddOp>{
    TypedLiteral<int64_t>{10},
    TypedLiteral<double>{2.5}
};

using checker = HottTypeChecker<decltype(arith_expr)>;
using proof = typename checker::discharged_proof;

static_assert(proof::result_type == TypeCode::Double);
static_assert(!proof::is_exact); // Result is inexact due to double
```

## Integration Points

### Parser Integration

The type checker integrates with the existing parser by:
1. Building typed AST nodes during parsing
2. Performing type inference on-the-fly
3. Generating proof obligations for each construct
4. Validating proofs before proceeding to code generation

### LLVM Integration

The type checker integrates with LLVM codegen by:
1. Providing type information for specialized code generation
2. Enabling proof-directed optimizations
3. Eliminating runtime type checks where proofs guarantee safety
4. Maintaining compatibility with existing LLVM IR patterns

### Performance Characteristics

- **Compile-time Overhead**: O(n) in AST size, dominated by template instantiation
- **Runtime Overhead**: Zero for well-typed programs (complete proof erasure)
- **Memory Usage**: No additional runtime memory for proof terms
- **Optimization Potential**: Better than current system due to proof-guaranteed properties

This specification provides the foundation for implementing a mathematically rigorous type checker that maintains Eshkol's performance characteristics while providing strong type safety guarantees through HoTT principles.