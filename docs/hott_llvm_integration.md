# HoTT-LLVM Integration Layer Specification

## Overview

This document specifies the integration layer between Eshkol's HoTT-based mixed type lists and the existing LLVM code generation infrastructure. The integration leverages compile-time proof information to generate highly optimized LLVM IR while maintaining compatibility with the existing codegen architecture.

## Core Integration Architecture

### Enhanced TypedValue with Proof Information

```cpp
// Enhanced TypedValue that carries proof information
namespace eshkol::hott::codegen {

// Proof-carrying typed value for LLVM IR generation
template<typename ProofType = void>
struct ProofCarryingValue {
    llvm::Value* llvm_value;              // LLVM IR value
    eshkol_value_type_t runtime_type;     // Runtime type tag (for compatibility)
    bool is_exact;                        // Scheme exactness tracking
    
    // Compile-time proof information (erased at runtime)
    using proof_type = ProofType;
    static constexpr bool has_proof = !std::is_void_v<ProofType>;
    static constexpr bool is_proven_safe = has_proof && ProofType::is_valid;
    
    // Constructor for proven values
    template<typename P = ProofType>
    ProofCarryingValue(llvm::Value* val, eshkol_value_type_t type, bool exact = true)
        requires (!std::is_void_v<P> && P::is_valid)
        : llvm_value(val), runtime_type(type), is_exact(exact) {
        static_assert(P::is_valid, "Proof must be valid for proven value construction");
    }
    
    // Constructor for unproven values (legacy compatibility)
    ProofCarryingValue(llvm::Value* val, eshkol_value_type_t type, bool exact = true)
        requires std::is_void_v<ProofType>
        : llvm_value(val), runtime_type(type), is_exact(exact) {}
    
    // Type checking helpers with proof awareness
    constexpr bool isInt64() const { return runtime_type == ESHKOL_VALUE_INT64; }
    constexpr bool isDouble() const { return runtime_type == ESHKOL_VALUE_DOUBLE; }
    constexpr bool isNull() const { return runtime_type == ESHKOL_VALUE_NULL; }
    
    // Proof-based optimization queries
    static constexpr bool can_eliminate_runtime_checks() { return is_proven_safe; }
    static constexpr bool can_specialize() { return has_proof && ProofType::allows_specialization; }
    static constexpr bool can_vectorize() { return has_proof && ProofType::allows_vectorization; }
};

// Type aliases for common cases
using UnprovenValue = ProofCarryingValue<void>;
template<typename P> using ProvenValue = ProofCarryingValue<P>;

} // namespace eshkol::hott::codegen
```

### Proof-Directed Code Generator

```cpp
// Enhanced LLVM code generator with proof integration
class HottEnabledLLVMCodeGen : public EshkolLLVMCodeGen {
private:
    // Proof registry for tracking active proofs
    std::unordered_map<llvm::Value*, std::unique_ptr<ProofMetadata>> proof_registry;
    
    // Optimization strategy selection based on proofs
    enum class CodeGenStrategy {
        LEGACY_COMPATIBLE,  // No proofs, use existing paths
        PROOF_OPTIMIZED,    // Proofs available, use optimized paths
        PROOF_SPECIALIZED   // Strong proofs, generate specialized code
    };

public:
    // Proof-aware cons cell generation
    template<typename CarProof, typename CdrProof>
    ProvenValue<ConsCreationProof<CarProof, CdrProof>> 
    generateHottCons(const ProvenValue<CarProof>& car_val, 
                     const ProvenValue<CdrProof>& cdr_val) {
        
        using cons_proof = ConsCreationProof<CarProof, CdrProof>;
        static_assert(cons_proof::is_valid, "Cons creation proof failed");
        
        auto strategy = selectStrategy<cons_proof>();
        
        switch (strategy) {
            case CodeGenStrategy::PROOF_SPECIALIZED:
                return generateSpecializedCons<CarProof, CdrProof>(car_val, cdr_val);
            case CodeGenStrategy::PROOF_OPTIMIZED:
                return generateOptimizedCons<CarProof, CdrProof>(car_val, cdr_val);
            default:
                return generateLegacyCons(car_val.llvm_value, cdr_val.llvm_value);
        }
    }
    
    // Proof-aware arithmetic generation
    template<typename LeftProof, typename RightProof>
    auto generateHottArithmetic(ArithmeticOp op,
                                const ProvenValue<LeftProof>& left_val,
                                const ProvenValue<RightProof>& right_val) {
        
        using arith_proof = ArithmeticOperationProof<LeftProof, RightProof>;
        static_assert(arith_proof::is_valid, "Arithmetic operation proof failed");
        
        if constexpr (arith_proof::requires_promotion) {
            return generateWithPromotion<arith_proof>(op, left_val, right_val);
        } else if constexpr (arith_proof::is_homogeneous) {
            return generateHomogeneousArithmetic<arith_proof>(op, left_val, right_val);
        } else {
            return generateGeneralArithmetic<arith_proof>(op, left_val, right_val);
        }
    }
    
    // Proof-aware list operation generation
    template<typename ListProof>
    auto generateHottListOp(ListOperation op, const ProvenValue<ListProof>& list_val) {
        using list_op_proof = ListOperationProof<ListProof>;
        static_assert(list_op_proof::is_valid, "List operation proof failed");
        
        if constexpr (list_op_proof::can_vectorize) {
            return generateVectorizedListOp<list_op_proof>(op, list_val);
        } else if constexpr (list_op_proof::can_unroll) {
            return generateUnrolledListOp<list_op_proof>(op, list_val);
        } else {
            return generateStandardListOp<list_op_proof>(op, list_val);
        }
    }

private:
    template<typename ProofType>
    CodeGenStrategy selectStrategy() {
        if constexpr (!ProofType::is_valid) {
            return CodeGenStrategy::LEGACY_COMPATIBLE;
        } else if constexpr (ProofType::enables_specialization) {
            return CodeGenStrategy::PROOF_SPECIALIZED;
        } else {
            return CodeGenStrategy::PROOF_OPTIMIZED;
        }
    }
};
```

### Specialized Code Generation Patterns

```cpp
// Specialized code generators for proven safe operations
namespace eshkol::hott::codegen::specialized {

// Homogeneous integer list operations
template<size_t Length>
class HomogeneousInt64Generator {
public:
    // Generate optimized cons cell allocation for int64 pairs
    static llvm::Value* generateConsAlloc(llvm::IRBuilder<>& builder, 
                                           TypeAwareArena& arena) {
        // Direct 16-byte allocation, no type tags needed
        auto* i64_type = builder.getInt64Ty();
        auto* cons_type = llvm::StructType::get(builder.getContext(), {i64_type, i64_type});
        
        // Use existing fast arena allocation path
        auto* arena_func = getArenaAllocConsFunction();
        return builder.CreateCall(arena_func, {arena.getLLVMValue()});
    }
    
    // Generate SIMD-optimized arithmetic for homogeneous lists
    static llvm::Value* generateVectorizedAdd(llvm::IRBuilder<>& builder,
                                               llvm::Value* list1_ptr,
                                               llvm::Value* list2_ptr) {
        if constexpr (Length >= 4) {
            // Use LLVM vector types for SIMD operations
            auto* vec_type = llvm::FixedVectorType::get(builder.getInt64Ty(), 4);
            
            // Load vectors
            auto* vec1 = builder.CreateLoad(vec_type, 
                builder.CreateBitCast(list1_ptr, vec_type->getPointerTo()));
            auto* vec2 = builder.CreateLoad(vec_type,
                builder.CreateBitCast(list2_ptr, vec_type->getPointerTo()));
            
            // Vector addition
            return builder.CreateAdd(vec1, vec2);
        } else {
            // Fall back to scalar operations for small lists
            return generateScalarAdd(builder, list1_ptr, list2_ptr);
        }
    }
    
    // Generate loop-free operations for small known-size lists
    template<size_t N>
    static llvm::Value* generateUnrolledMap(llvm::IRBuilder<>& builder,
                                             llvm::Function* map_func,
                                             llvm::Value* list_ptr) {
        static_assert(N <= 16, "Unrolling only beneficial for small lists");
        
        // Generate N separate function calls
        std::array<llvm::Value*, N> results;
        for (size_t i = 0; i < N; ++i) {
            auto* element_ptr = builder.CreateGEP(builder.getInt64Ty(), list_ptr, 
                                                   builder.getInt64(i));
            auto* element = builder.CreateLoad(builder.getInt64Ty(), element_ptr);
            results[i] = builder.CreateCall(map_func, {element});
        }
        
        return createResultList(builder, results);
    }
};

// Mixed type operations with compile-time type information
template<TypeCode CarCode, TypeCode CdrCode>
class MixedTypeGenerator {
private:
    using CarType = Interpret<CarCode>;
    using CdrType = Interpret<CdrCode>;
    
public:
    // Generate type-specific access patterns
    static llvm::Value* generateCarAccess(llvm::IRBuilder<>& builder,
                                           llvm::Value* cons_ptr) {
        auto* car_type = getLLVMType<CarType>(builder.getContext());
        auto* cons_type = llvm::StructType::get(builder.getContext(), 
                                                {car_type, getLLVMType<CdrType>(builder.getContext())});
        
        // Direct GEP access, no runtime type checking
        auto* car_ptr = builder.CreateStructGEP(cons_type, cons_ptr, 0);
        return builder.CreateLoad(car_type, car_ptr);
    }
    
    static llvm::Value* generateCdrAccess(llvm::IRBuilder<>& builder,
                                           llvm::Value* cons_ptr) {
        auto* cdr_type = getLLVMType<CdrType>(builder.getContext());
        auto* cons_type = llvm::StructType::get(builder.getContext(), 
                                                {getLLVMType<CarType>(builder.getContext()), cdr_type});
        
        auto* cdr_ptr = builder.CreateStructGEP(cons_type, cons_ptr, 1);
        return builder.CreateLoad(cdr_type, cdr_ptr);
    }
    
    // Generate type-safe arithmetic with automatic promotion
    template<ArithmeticOp Op>
    static llvm::Value* generateTypedArithmetic(llvm::IRBuilder<>& builder,
                                                 llvm::Value* left_val,
                                                 llvm::Value* right_val) {
        if constexpr (CarCode == TypeCode::Int64 && CdrCode == TypeCode::Int64) {
            // Pure integer arithmetic
            return generateIntArithmetic<Op>(builder, left_val, right_val);
        } else if constexpr (CarCode == TypeCode::Double || CdrCode == TypeCode::Double) {
            // Promote to double
            auto* left_double = promoteToDouble<CarCode>(builder, left_val);
            auto* right_double = promoteToDouble<CdrCode>(builder, right_val);
            return generateDoubleArithmetic<Op>(builder, left_double, right_double);
        } else {
            static_assert(false, "Unsupported type combination for arithmetic");
        }
    }
};

} // namespace eshkol::hott::codegen::specialized
```

### Proof-Directed Optimization Passes

```cpp
// LLVM optimization passes that leverage proof information
namespace eshkol::hott::optimization {

class HottOptimizationPass : public llvm::FunctionPass {
public:
    static char ID;
    HottOptimizationPass() : FunctionPass(ID) {}
    
    bool runOnFunction(llvm::Function& F) override {
        bool changed = false;
        
        // Pass 1: Eliminate redundant type checks based on proofs
        changed |= eliminateProvenTypeChecks(F);
        
        // Pass 2: Specialize generic operations based on proof information
        changed |= specializeProvenOperations(F);
        
        // Pass 3: Enable vectorization for proven homogeneous operations
        changed |= enableProofBasedVectorization(F);
        
        // Pass 4: Inline small proven-safe operations
        changed |= inlineProvenSafeOperations(F);
        
        return changed;
    }

private:
    bool eliminateProvenTypeChecks(llvm::Function& F) {
        bool changed = false;
        
        for (auto& BB : F) {
            for (auto I = BB.begin(), E = BB.end(); I != E;) {
                llvm::Instruction* inst = &*I++;
                
                // Look for type checking patterns
                if (isTypeCheckInstruction(inst)) {
                    auto proof_info = getProofInfo(inst);
                    if (proof_info && proof_info->guarantees_type_safety) {
                        // Replace type check with constant true/false
                        auto* constant = llvm::ConstantInt::get(inst->getType(), 
                                                               proof_info->check_result);
                        inst->replaceAllUsesWith(constant);
                        inst->eraseFromParent();
                        changed = true;
                    }
                }
            }
        }
        
        return changed;
    }
    
    bool specializeProvenOperations(llvm::Function& F) {
        bool changed = false;
        
        for (auto& BB : F) {
            for (auto& I : BB) {
                if (auto* call = llvm::dyn_cast<llvm::CallInst>(&I)) {
                    auto* callee = call->getCalledFunction();
                    if (!callee) continue;
                    
                    // Check if this is a generic operation that can be specialized
                    if (isGenericOperation(callee)) {
                        auto proof_info = getCallProofInfo(call);
                        if (proof_info && proof_info->enables_specialization) {
                            auto* specialized_func = getOrCreateSpecializedFunction(
                                callee, proof_info->type_signature);
                            
                            // Replace call with specialized version
                            auto* new_call = llvm::CallInst::Create(specialized_func,
                                                                   call->args(),
                                                                   call->getName(),
                                                                   call);
                            call->replaceAllUsesWith(new_call);
                            call->eraseFromParent();
                            changed = true;
                        }
                    }
                }
            }
        }
        
        return changed;
    }
    
    bool enableProofBasedVectorization(llvm::Function& F) {
        bool changed = false;
        
        // Look for loops operating on proven homogeneous data
        for (auto& BB : F) {
            if (auto* loop = detectSimpleLoop(&BB)) {
                auto vectorization_info = analyzeVectorizationOpportunity(loop);
                if (vectorization_info.is_vectorizable && 
                    vectorization_info.has_homogeneity_proof) {
                    
                    // Add vectorization hints for LLVM's auto-vectorizer
                    addVectorizationHints(loop, vectorization_info);
                    changed = true;
                }
            }
        }
        
        return changed;
    }
};

// Register the optimization pass
char HottOptimizationPass::ID = 0;
static llvm::RegisterPass<HottOptimizationPass> 
    X("hott-opt", "HoTT-based optimizations", false, false);

} // namespace eshkol::hott::optimization
```

### Integration with Existing Arena System

```cpp
// Enhanced arena integration with proof-aware allocation
namespace eshkol::hott::arena {

class ProofAwareArenaManager {
private:
    TypeAwareArena* base_arena;
    llvm::IRBuilder<>* builder;
    llvm::Module* module;
    
    // Cached specialized allocation functions
    std::unordered_map<std::string, llvm::Function*> specialized_allocators;

public:
    ProofAwareArenaManager(TypeAwareArena* arena, llvm::IRBuilder<>* b, llvm::Module* m)
        : base_arena(arena), builder(b), module(m) {}
    
    // Generate allocation code based on proof information
    template<typename AllocationProof>
    llvm::Value* generateAllocation() {
        using proof = AllocationProof;
        static_assert(proof::is_valid, "Allocation proof must be valid");
        
        if constexpr (proof::is_standard_layout) {
            return generateStandardAllocation<proof>();
        } else if constexpr (proof::requires_custom_layout) {
            return generateCustomAllocation<proof>();
        } else {
            return generateFallbackAllocation<proof>();
        }
    }
    
    // Batch allocation for proven homogeneous lists
    template<typename BatchProof>
    llvm::Value* generateBatchAllocation(llvm::Value* count) {
        using proof = BatchProof;
        static_assert(proof::enables_batch_allocation, "Proof must enable batch allocation");
        
        // Generate optimized batch allocation
        auto* element_size = llvm::ConstantInt::get(builder->getInt64Ty(), 
                                                    proof::element_size);
        auto* total_size = builder->CreateMul(count, element_size);
        
        // Call arena batch allocation
        auto* batch_alloc_func = getOrCreateBatchAllocator<proof>();
        return builder->CreateCall(batch_alloc_func, 
                                   {base_arena->getLLVMValue(), total_size});
    }

private:
    template<typename Proof>
    llvm::Value* generateStandardAllocation() {
        // Use existing arena_allocate_cons_cell for standard 16-byte allocations
        auto* arena_val = base_arena->getLLVMValue();
        auto* alloc_func = module->getFunction("arena_allocate_cons_cell");
        
        if (!alloc_func) {
            // Declare the function if not already present
            auto* func_type = llvm::FunctionType::get(
                builder->getInt8PtrTy(),
                {builder->getInt8PtrTy()}, // arena pointer
                false
            );
            alloc_func = llvm::Function::Create(func_type, 
                                                llvm::Function::ExternalLinkage,
                                                "arena_allocate_cons_cell", module);
        }
        
        return builder->CreateCall(alloc_func, {arena_val});
    }
    
    template<typename Proof>
    llvm::Function* getOrCreateBatchAllocator() {
        std::string func_name = "hott_batch_alloc_" + std::to_string(Proof::element_size);
        
        auto it = specialized_allocators.find(func_name);
        if (it != specialized_allocators.end()) {
            return it->second;
        }
        
        // Create specialized batch allocator function
        auto* func_type = llvm::FunctionType::get(
            builder->getInt8PtrTy(),
            {builder->getInt8PtrTy(), builder->getInt64Ty()}, // arena, count
            false
        );
        
        auto* func = llvm::Function::Create(func_type,
                                            llvm::Function::InternalLinkage,
                                            func_name, module);
        
        // Generate function body
        generateBatchAllocatorBody<Proof>(func);
        
        specialized_allocators[func_name] = func;
        return func;
    }
    
    template<typename Proof>
    void generateBatchAllocatorBody(llvm::Function* func) {
        auto* entry = llvm::BasicBlock::Create(builder->getContext(), "entry", func);
        auto* old_bb = builder->GetInsertBlock();
        builder->SetInsertPoint(entry);
        
        auto args = func->arg_begin();
        llvm::Value* arena_arg = &*args++;
        llvm::Value* count_arg = &*args++;
        
        // Calculate total allocation size
        auto* element_size = llvm::ConstantInt::get(builder->getInt64Ty(), Proof::element_size);
        auto* total_size = builder->CreateMul(count_arg, element_size);
        
        // Call arena_allocate with calculated size
        auto* arena_alloc_func = module->getFunction("arena_allocate_aligned");
        auto* alignment = llvm::ConstantInt::get(builder->getInt64Ty(), Proof::alignment);
        auto* result = builder->CreateCall(arena_alloc_func, 
                                           {arena_arg, total_size, alignment});
        
        builder->CreateRet(result);
        builder->SetInsertPoint(old_bb);
    }
};

} // namespace eshkol::hott::arena
```

### Compatibility Layer with Current System

```cpp
// Bridge between HoTT codegen and existing LLVM infrastructure
namespace eshkol::hott::bridge {

class LLVMBridge {
private:
    EshkolLLVMCodeGen* legacy_codegen;
    HottEnabledLLVMCodeGen* hott_codegen;
    bool hott_enabled;

public:
    LLVMBridge(EshkolLLVMCodeGen* legacy, bool enable_hott = false) 
        : legacy_codegen(legacy), hott_enabled(enable_hott) {
        if (hott_enabled) {
            hott_codegen = new HottEnabledLLVMCodeGen(*legacy);
        }
    }
    
    // Unified interface for cons cell generation
    template<typename CarValue, typename CdrValue>
    auto generateCons(const CarValue& car, const CdrValue& cdr) {
        if constexpr (isProvenValue<CarValue> && isProvenValue<CdrValue>) {
            if (hott_enabled) {
                return hott_codegen->generateHottCons(car, cdr);
            }
        }
        
        // Fall back to legacy implementation
        return legacy_codegen->codegenCons(car.llvm_value, cdr.llvm_value);
    }
    
    // Unified interface for arithmetic operations
    template<typename LeftValue, typename RightValue>
    auto generateArithmetic(ArithmeticOp op, const LeftValue& left, const RightValue& right) {
        if constexpr (isProvenValue<LeftValue> && isProvenValue<RightValue>) {
            if (hott_enabled) {
                return hott_codegen->generateHottArithmetic(op, left, right);
            }
        }
        
        // Fall back to legacy implementation
        switch (op) {
            case ArithmeticOp::ADD:
                return legacy_codegen->codegenAdd(left.llvm_value, right.llvm_value);
            case ArithmeticOp::SUB:
                return legacy_codegen->codegenSub(left.llvm_value, right.llvm_value);
            case ArithmeticOp::MUL:
                return legacy_codegen->codegenMul(left.llvm_value, right.llvm_value);
            case ArithmeticOp::DIV:
                return legacy_codegen->codegenDiv(left.llvm_value, right.llvm_value);
            default:
                throw std::runtime_error("Unsupported arithmetic operation");
        }
    }
    
    // Progressive migration support
    void enableHottOptimizations() {
        if (!hott_codegen) {
            hott_codegen = new HottEnabledLLVMCodeGen(*legacy_codegen);
        }
        hott_enabled = true;
    }
    
    void disableHottOptimizations() {
        hott_enabled = false;
    }
};

// Global bridge instance for gradual migration
extern LLVMBridge* g_llvm_bridge;

// Macro for feature-flag controlled code generation
#define ESHKOL_CODEGEN(legacy_call, hott_call) \
    (g_llvm_bridge ? g_llvm_bridge->hott_call : legacy_call)

} // namespace eshkol::hott::bridge
```

### Performance Monitoring and Analysis

```cpp
// Performance analysis for HoTT-optimized code
namespace eshkol::hott::profiling {

struct OptimizationMetrics {
    size_t eliminated_type_checks = 0;
    size_t specialized_functions = 0;
    size_t vectorized_loops = 0;
    size_t inlined_operations = 0;
    double compile_time_overhead = 0.0;
    
    void print() const {
        eshkol_info("HoTT Optimization Metrics:");
        eshkol_info("  Eliminated type checks: %zu", eliminated_type_checks);
        eshkol_info("  Specialized functions: %zu", specialized_functions);
        eshkol_info("  Vectorized loops: %zu", vectorized_loops);
        eshkol_info("  Inlined operations: %zu", inlined_operations);
        eshkol_info("  Compile time overhead: %.2f%%", compile_time_overhead * 100);
    }
};

class PerformanceProfiler {
private:
    OptimizationMetrics metrics;
    std::chrono::steady_clock::time_point compile_start;
    
public:
    void startCompilation() {
        compile_start = std::chrono::steady_clock::now();
    }
    
    void recordOptimization(OptimizationType type) {
        switch (type) {
            case OptimizationType::TYPE_CHECK_ELIMINATION:
                metrics.eliminated_type_checks++;
                break;
            case OptimizationType::FUNCTION_SPECIALIZATION:
                metrics.specialized_functions++;
                break;
            case OptimizationType::LOOP_VECTORIZATION:
                metrics.vectorized_loops++;
                break;
            case OptimizationType::OPERATION_INLINING:
                metrics.inlined_operations++;
                break;
        }
    }
    
    void finishCompilation() {
        auto compile_end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            compile_end - compile_start);
        metrics.compile_time_overhead = duration.count() / 1000.0;
    }
    
    const OptimizationMetrics& getMetrics() const { return metrics; }
};

} // namespace eshkol::hott::profiling
```

## Integration Workflow

### Compilation Pipeline Integration

```cpp
// Enhanced compilation pipeline with HoTT integration
class HottCompilationPipeline {
public:
    enum class Mode {
        LEGACY_ONLY,        // Use only existing codegen
        HOTT_COMPATIBLE,    // Use HoTT where proofs available, legacy otherwise
        HOTT_AGGRESSIVE     // Require proofs for all operations
    };
    
    static void compileFunctionWithHott(eshkol_ast* ast, 
                                        llvm::Module* module,
                                        Mode mode = Mode::HOTT_COMPATIBLE) {
        
        // Phase 1: Parse and build typed AST
        auto typed_ast = buildTypedAST(ast);
        
        // Phase 2: Generate and validate proofs
        auto proof_context = generateProofContext(typed_ast);
        
        // Phase 3: Select codegen strategy based on available proofs
        auto strategy = selectCodegenStrategy(proof_context, mode);
        
        // Phase 4: Generate LLVM IR with proof-directed optimizations
        generateOptimizedIR(typed_ast, proof_context, strategy, module);
        
        // Phase 5: Apply HoTT-specific optimization passes
        applyHottOptimizations(module);
        
        // Phase 6: Validate generated IR and collect metrics
        validateAndProfile(module);
    }

private:
    static CodegenStrategy selectCodegenStrategy(const ProofContext& proofs, Mode mode) {
        switch (mode) {
            case Mode::LEGACY_ONLY:
                return CodegenStrategy::LEGACY_FALLBACK;
            
            case Mode::HOTT_COMPATIBLE:
                if (proofs.hasValidProofs()) {
                    return proofs.enablesSpecialization() ? 
                           CodegenStrategy::HOTT_SPECIALIZED :
                           CodegenStrategy::HOTT_OPTIMIZED;
                } else {
                    return CodegenStrategy::LEGACY_FALLBACK;
                }
            
            case Mode::HOTT_AGGRESSIVE:
                if (!proofs.hasCompleteProofs()) {
                    throw CompilationError("HoTT aggressive mode requires complete proofs");
                }
                return CodegenStrategy::HOTT_SPECIALIZED;
        }
    }
};
```

This integration specification provides:

1. **Seamless Bridge**: Between HoTT proofs and LLVM IR generation
2. **Performance Optimization**: Proof-directed code generation and optimization passes
3. **Compatibility**: Gradual migration path from existing system
4. **Extensibility**: Framework for adding new proof-based optimizations
5. **Monitoring**: Performance analysis and metrics collection

The integration maintains the existing LLVM codegen architecture while enabling significant optimizations through compile-time proof information, achieving the goal of zero-cost abstractions with mathematical rigor.