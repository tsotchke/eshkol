#ifndef ESHKOL_BACKEND_LLVM_CODEGEN_H
#define ESHKOL_BACKEND_LLVM_CODEGEN_H
#include <eshkol/eshkol.h>
#include <eshkol/backend/arithmetic_codegen.h>
#include <eshkol/backend/autodiff_codegen.h>
#include <eshkol/backend/binding_codegen.h>
#include <eshkol/backend/builtin_declarations.h>
#include <eshkol/backend/call_apply_codegen.h>
#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/ir_builder.h>
#include <eshkol/backend/collection_codegen.h>
#include <eshkol/backend/complex_codegen.h>
#include <eshkol/backend/control_flow_codegen.h>
#include <eshkol/backend/function_cache.h>
#include <eshkol/backend/function_codegen.h>
#include <eshkol/backend/hash_codegen.h>
#include <eshkol/backend/homoiconic_codegen.h>
#include <eshkol/backend/logic_workspace_codegen.h>
#include <eshkol/backend/map_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/backend/parallel_codegen.h>
#include <eshkol/backend/string_io_codegen.h>
#include <eshkol/backend/system_codegen.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/tail_call_codegen.h>
#include <eshkol/backend/tensor_codegen.h>
#include <eshkol/backend/type_system.h>
#include <eshkol/types/type_checker.h>
#include <eshkol/logger.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/Support/AtomicOrdering.h>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef ESHKOL_LLVM_BACKEND_ENABLED
using namespace llvm;

template<class T> class ContinuationTask;

struct TypedValue {
    Value* llvm_value;              // LLVM IR value
    eshkol_value_type_t type;       // Our type tag from eshkol.h
    bool is_exact;                  // Scheme exactness tracking
    uint8_t flags;                  // Additional flags (e.g., indirect reference flag)
    eshkol::hott::TypeId hott_type; // HoTT compile-time type (for gradual typing)

    // HoTT PARAMETERIZED TYPE: For tracking List<Int64>, Vector<Float64>, etc.
    // This is optional - only set for collection types with known element types
    std::optional<eshkol::hott::ParameterizedType> param_type;

    // Flag constants
    static constexpr uint8_t FLAG_INDIRECT = 0x01;  // Value is address of global to load from
    static constexpr uint8_t FLAG_LINEAR   = 0x02;  // Must be consumed exactly once (linear type)
    static constexpr uint8_t FLAG_PROOF    = 0x04;  // Compile-time only, erased at runtime

    TypedValue()
        : llvm_value(nullptr), type(ESHKOL_VALUE_NULL), is_exact(true), flags(0),
          hott_type(eshkol::hott::BuiltinTypes::Null), param_type(std::nullopt) {}

    TypedValue(Value* val, eshkol_value_type_t t, bool exact = true, uint8_t f = 0)
        : llvm_value(val), type(t), is_exact(exact), flags(f),
          hott_type(eshkol::hott::BuiltinTypes::Value), param_type(std::nullopt) {}

    // Constructor with explicit HoTT type
    TypedValue(Value* val, eshkol_value_type_t t, eshkol::hott::TypeId hott_t, bool exact = true, uint8_t f = 0)
        : llvm_value(val), type(t), is_exact(exact), flags(f), hott_type(hott_t), param_type(std::nullopt) {}

    // Constructor with parameterized type (for List<T>, Vector<T>, etc.)
    TypedValue(Value* val, eshkol_value_type_t t, eshkol::hott::TypeId hott_t,
               const eshkol::hott::ParameterizedType& ptype, bool exact = true, uint8_t f = 0)
        : llvm_value(val), type(t), is_exact(exact), flags(f), hott_type(hott_t), param_type(ptype) {}

    // Helper methods
    bool isInt64() const { return type == ESHKOL_VALUE_INT64; }
    bool isDouble() const { return type == ESHKOL_VALUE_DOUBLE; }
    bool isNull() const { return type == ESHKOL_VALUE_NULL; }
    bool isIndirect() const { return (flags & FLAG_INDIRECT) != 0; }

    // HoTT type helpers
    bool hasKnownType() const { return hott_type != eshkol::hott::BuiltinTypes::Value; }
    bool isHottInt64() const { return hott_type == eshkol::hott::BuiltinTypes::Int64; }
    bool isHottFloat64() const { return hott_type == eshkol::hott::BuiltinTypes::Float64; }

    // HoTT parameterized type helpers
    bool hasParameterizedType() const { return param_type.has_value(); }
    eshkol::hott::TypeId elementType() const {
        if (param_type.has_value()) {
            return param_type->elementType();
        }
        return eshkol::hott::BuiltinTypes::Value;
    }
    // M1 Migration: Check both legacy and consolidated formats
    bool isList() const { return type == ESHKOL_VALUE_HEAP_PTR || type == ESHKOL_VALUE_HEAP_PTR; }
    bool isVector() const { return type == ESHKOL_VALUE_HEAP_PTR || type == ESHKOL_VALUE_HEAP_PTR; }

    // Linear and proof type helpers (HoTT Phase 4)
    bool isLinear() const { return (flags & FLAG_LINEAR) != 0; }
    bool isProof() const { return (flags & FLAG_PROOF) != 0; }
    bool shouldErase() const { return isProof(); }  // Proofs are erased at runtime

    // Create an erased value with a unit placeholder (for contexts that need a valid LLVM value)
    static TypedValue makeErasedWithPlaceholder(llvm::Value* unit_value, eshkol::hott::TypeId proof_type) {
        return TypedValue(unit_value, ESHKOL_VALUE_NULL, proof_type, true, FLAG_PROOF);
    }
};


struct LambdaSExprMetadata {
    const eshkol_operations_t* lambda_ast;
    std::string lambda_name;
};

namespace ControlFlowCallbacks {
    // Close the active per-iteration arena + loop scopes immediately before a
    // validated terminal process exit.
    void finishIterScopeBeforeExitWrapper(void* context);
    // Wrapper for codegenAST - returns LLVM Value*
    llvm::Value* codegenASTWrapper(const void* ast, void* context);
    // Wrapper for codegenTypedAST - returns pointer to TypedValue (caller owns)
    void* codegenTypedASTWrapper(const void* ast, void* context);
    // Wrapper for typedValueToTaggedValue
    llvm::Value* typedToTaggedWrapper(void* typed_value, void* context);
    // Wrapper for codegenNestedFunctionDefinition
    void codegenFuncDefineWrapper(const void* op, void* context);
    // Wrapper for codegenVariableDefinition
    void codegenVarDefineWrapper(const void* op, void* context);
    // Wrapper for callBuiltinEqv
    llvm::Value* eqvCompareWrapper(llvm::Value* a, llvm::Value* b, void* context);
    // Wrapper for detectValueType + typedValueToTaggedValue
    llvm::Value* detectAndPackWrapper(llvm::Value* val, void* context);
    // Wrapper for codegenTaggedArenaConsCellFromTaggedValue
    llvm::Value* consCreateWrapper(llvm::Value* car, llvm::Value* cdr, void* context);
    // Wrapper to get TypedValue type
    int getTypedValueTypeWrapper(void* typed_value, void* context);
    // Wrapper to register function binding
    void registerFuncBindingWrapper(const char* var_name, void* typed_value, void* context);
    // Wrapper for extractConsCarAsTaggedValue (for CallApplyCodegen)
    llvm::Value* extractConsCarWrapper(llvm::Value* cons_ptr, void* context);
    // Wrapper for getTaggedConsGetPtrFunc (for CallApplyCodegen)
    llvm::Function* getConsAccessorWrapper(void* context);
    // Wrapper for codegenAST with typed signature (for CallApplyCodegen)
    llvm::Value* codegenASTTypedWrapper(const eshkol_ast_t* ast, void* context);
    // Wrappers for MapCodegen
    llvm::Value* codegenLambdaWrapper(const eshkol_operations_t* op, void* context);
    llvm::Value* closureCallWrapper(llvm::Value* closure, const std::vector<llvm::Value*>& args, void* context);
    llvm::Value* closureSpreadCallWrapper(llvm::Value*, llvm::Value*, llvm::Value*, int, void*);
    llvm::Value* closureListCallWrapper(llvm::Value*, llvm::Value*, void*);
    llvm::Value* closureCallWithInfoWrapper(llvm::Value* closure, const std::vector<llvm::Value*>& args, const char* info, void* context);
    llvm::Value* gradientSpreadCallWrapper(llvm::Value* closure, llvm::Value* point_vector,
                                                  llvm::Value* dual_elems, llvm::Value* declared_arity,
                                                  void* context);
    llvm::Function* getClosureAllocWrapper(void* context);
    llvm::Function* getConsSetPtrWrapper(void* context);
    llvm::Value* resolveLambdaWrapper(const eshkol_ast_t* ast, size_t arity, void* context);
    bool variadicLookupWrapper(const char* name, void* context);
    bool applyVariadicInfoWrapper(const char* name, uint64_t* fixed_params, void* context);
    llvm::Value* indirectCallWrapper(llvm::Value* arg, size_t arity, void* context);
    void pushFunctionContextWrapper(void* context);
    void popFunctionContextWrapper(void* context);
    // TCO callback for checking self-tail-recursion
    bool isSelfTailRecursiveWrapper(const void* lambda_op, const char* func_name, void* context);
    // Binding callback for assignment conversion of lexical locals.
    bool isVarSetWrapper(const void* ast, const char* name, void* context);
    bool isReassignedTopLevelNameWrapper(const char* name, void* context);
    // Companion assignment-conversion queries (SW-62): whether the mutated
    // location can still be read after the mutation through a context that
    // outlives the native frame, and whether an escaping continuation may
    // restore control into this scope after the frame is gone.
    bool isVarObservedWrapper(const void* ast, const char* name, void* context);
    bool continuationEscapeWrapper(const void* ast, void* context);
    // Wrapper for getting builtin arithmetic functions (for CallApplyCodegen)
    llvm::Function* getBuiltinArithmeticWrapper(const std::string& op, void* context);
    // Wrapper for resolving comparison/equality/predicate builtins (for apply)
    llvm::Function* getBuiltinPredicateWrapper(const std::string& name, void* context);
    // Wrapper for applying tensor/vector builtin functions
    llvm::Value* applyBuiltinWrapper(const std::string& func_name, const std::vector<llvm::Value*>& args, llvm::Value* arg_count, void* context);
    // Bug P (2026-04-23): wrapper for forward-ref apply (cross-file user defines)
    llvm::Value* applyForwardRefWrapper(const std::string& func_name, llvm::Value* list_int, void* context);
}

namespace eshkol::llvm_codegen_detail {
bool& replModeEnabled();
std::mutex& replMutex();
std::unordered_map<std::string, uint64_t>& replFunctionAddresses();
std::unordered_map<std::string, size_t>& replFunctionArities();
std::unordered_map<std::string, std::string>& replLambdaNames();
std::unordered_set<std::string>& replPrivateSymbols();
std::unordered_set<std::string>& replNativeCFunctions();
std::unordered_map<std::string, std::vector<std::string>>& replLambdaCaptures();
std::unordered_map<std::string, uint64_t>& replSymbolAddresses();
std::unordered_map<std::string, std::pair<size_t, bool>>& replVariadicFunctions();
std::unordered_set<std::string>& replUserVariableNames();
std::string& sourceFilepath();
std::string& sourceText();
std::string& lastGeneratedLambdaName();
std::vector<LambdaSExprMetadata>& pendingLambdaSExprs();
}

class EshkolLLVMCodeGen {
    /* The first-class builtin table's row type and its variadic-rest
     * strategy are DEFINED next to the table itself (see
     * lookupInlineBuiltin), because the table is the thing a reader
     * needs them for. The value-position sites that consult the table
     * appear textually before it, so the names are introduced here. */
    enum class VariadicRest : uint8_t;
    struct InlineBuiltinSpec;

    // Friend declarations for ControlFlowCodegen callbacks
    friend llvm::Value* ControlFlowCallbacks::codegenASTWrapper(const void* ast, void* context);
    friend void* ControlFlowCallbacks::codegenTypedASTWrapper(const void* ast, void* context);
    friend llvm::Value* ControlFlowCallbacks::typedToTaggedWrapper(void* typed_value, void* context);
    friend void ControlFlowCallbacks::codegenFuncDefineWrapper(const void* op, void* context);
    friend void ControlFlowCallbacks::codegenVarDefineWrapper(const void* op, void* context);
    friend llvm::Value* ControlFlowCallbacks::eqvCompareWrapper(llvm::Value* a, llvm::Value* b, void* context);
    friend llvm::Value* ControlFlowCallbacks::detectAndPackWrapper(llvm::Value* val, void* context);
    friend bool ControlFlowCallbacks::isVarSetWrapper(const void* ast, const char* name, void* context);
    friend bool ControlFlowCallbacks::isReassignedTopLevelNameWrapper(const char* name, void* context);
    friend bool ControlFlowCallbacks::isVarObservedWrapper(const void* ast, const char* name, void* context);
    friend bool ControlFlowCallbacks::continuationEscapeWrapper(const void* ast, void* context);
    friend llvm::Value* ControlFlowCallbacks::consCreateWrapper(llvm::Value* car, llvm::Value* cdr, void* context);
    friend int ControlFlowCallbacks::getTypedValueTypeWrapper(void* typed_value, void* context);
    friend void ControlFlowCallbacks::registerFuncBindingWrapper(const char* var_name, void* typed_value, void* context);
    // Friend declarations for CallApplyCodegen callbacks
    friend llvm::Value* ControlFlowCallbacks::extractConsCarWrapper(llvm::Value* cons_ptr, void* context);
    friend llvm::Function* ControlFlowCallbacks::getConsAccessorWrapper(void* context);
    friend llvm::Value* ControlFlowCallbacks::codegenASTTypedWrapper(const eshkol_ast_t* ast, void* context);
    // Friend declarations for MapCodegen callbacks
    friend llvm::Value* ControlFlowCallbacks::codegenLambdaWrapper(const eshkol_operations_t* op, void* context);
    friend llvm::Value* ControlFlowCallbacks::closureCallWrapper(llvm::Value* closure, const std::vector<llvm::Value*>& args, void* context);
    friend llvm::Function* ControlFlowCallbacks::getConsSetPtrWrapper(void* context);
    friend llvm::Value* ControlFlowCallbacks::resolveLambdaWrapper(const eshkol_ast_t* ast, size_t arity, void* context);
    friend bool ControlFlowCallbacks::variadicLookupWrapper(const char* name, void* context);
    friend bool ControlFlowCallbacks::applyVariadicInfoWrapper(const char* name, uint64_t* fixed_params, void* context);
    friend llvm::Value* ControlFlowCallbacks::indirectCallWrapper(llvm::Value* arg, size_t arity, void* context);
    friend void ControlFlowCallbacks::pushFunctionContextWrapper(void* context);
    friend void ControlFlowCallbacks::popFunctionContextWrapper(void* context);
    friend bool ControlFlowCallbacks::isSelfTailRecursiveWrapper(const void* lambda_op, const char* func_name, void* context);
    friend llvm::Function* ControlFlowCallbacks::getBuiltinArithmeticWrapper(const std::string& op, void* context);
    friend llvm::Function* ControlFlowCallbacks::getBuiltinPredicateWrapper(const std::string& name, void* context);
    friend llvm::Value* ControlFlowCallbacks::applyBuiltinWrapper(const std::string& func_name, const std::vector<llvm::Value*>& args, llvm::Value* arg_count, void* context);
    friend llvm::Value* ControlFlowCallbacks::applyForwardRefWrapper(const std::string& func_name, llvm::Value* list_int, void* context);
    friend void ControlFlowCallbacks::finishIterScopeBeforeExitWrapper(void* context);
    friend llvm::Value* ControlFlowCallbacks::closureSpreadCallWrapper(llvm::Value*, llvm::Value*, llvm::Value*, int, void*);
    friend llvm::Value* ControlFlowCallbacks::closureListCallWrapper(llvm::Value*, llvm::Value*, void*);
    friend llvm::Value* ControlFlowCallbacks::closureCallWithInfoWrapper(llvm::Value* closure, const std::vector<llvm::Value*>& args, const char* info, void* context);
    friend llvm::Value* ControlFlowCallbacks::gradientSpreadCallWrapper(llvm::Value* closure, llvm::Value* point_vector,
                                                                        llvm::Value* dual_elems, llvm::Value* declared_arity,
                                                                        void* context);
    friend llvm::Function* ControlFlowCallbacks::getClosureAllocWrapper(void* context);

private:
    std::unique_ptr<LLVMContext> context;
    std::unique_ptr<Module> module;
    std::unique_ptr<eshkol::CodegenIRBuilder> builder;

    // Monotonic counter used by codegen sites that need a unique-but-stable
    // suffix in IR variable names (e.g. pattern-match argument slots). We
    // previously used `reinterpret_cast<uintptr_t>(some_value)` for this,
    // which made IR non-deterministic across runs because heap addresses
    // change. Replacing with a counter restores reproducibility.
    // Reset at the start of every generateIR() / library compilation so two
    // compilations of the same source always produce identical IR.
    uint64_t name_uniquifier_ = 0;

    // Interned plain C strings used only by opt-in runtime language coverage.
    // Keeping one global per distinct spelling avoids bloating instrumented
    // modules with a copy of the source path for every AST node.
    std::unordered_map<std::string, llvm::Value*> coverage_string_cache_;
    const bool language_coverage_enabled_ = [] {
        const char* dir = std::getenv("ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR");
        return dir && *dir;
    }();

    // Type system (manages all LLVM types)
    std::unique_ptr<eshkol::TypeSystem> types;
    std::vector<std::unique_ptr<eshkol_ast_t>> differentiation_asts_;

    // Function cache (lazy-loaded C library functions)
    std::unique_ptr<eshkol::FunctionCache> funcs;

    // Memory codegen (arena function declarations)
    std::unique_ptr<eshkol::MemoryCodegen> mem;

    // CodegenContext - Shared state for extracted modules
    // This provides clean access to LLVM infrastructure, symbol tables, and function caches
    // for modules like TaggedValueCodegen, ArithmeticCodegen, etc.
    std::unique_ptr<eshkol::CodegenContext> ctx_;

    // TaggedValueCodegen - Pack/unpack operations for tagged values
    // This module handles the runtime type system's tagged value representation
    std::unique_ptr<eshkol::TaggedValueCodegen> tagged_;

    // BuiltinDeclarations - External runtime function declarations
    // Handles deep_equal, display_value, lambda_registry functions
    std::unique_ptr<eshkol::BuiltinDeclarations> builtins_;

    // ComplexCodegen - Complex number arithmetic operations
    std::unique_ptr<eshkol::ComplexCodegen> complex_;

    // ArithmeticCodegen - Polymorphic arithmetic operations
    // Note: Main polymorphic implementations still in this file; module provides interface
    std::unique_ptr<eshkol::ArithmeticCodegen> arith_;

    // CallApplyCodegen - Function call and apply operations
    // Handles Scheme's apply and closure dispatch
    std::unique_ptr<eshkol::CallApplyCodegen> call_apply_;

    // MapCodegen - Higher-order list mapping operations
    // Handles (map proc list1 ...) with closure and capture support
    std::unique_ptr<eshkol::MapCodegen> map_;

    // ControlFlowCodegen - Control flow operations (and, or, if, cond, begin)
    // Note: Main implementations still in this file; module provides isTruthy helper
    std::unique_ptr<eshkol::ControlFlowCodegen> flow_;

    // StringIOCodegen - String and I/O operations
    // Note: Main implementations still in this file; module provides string creation and printf
    std::unique_ptr<eshkol::StringIOCodegen> strio_;

    // CollectionCodegen - List and vector operations
    // Note: Main implementations still in this file; module provides allocConsCell helper
    std::unique_ptr<eshkol::CollectionCodegen> coll_;

    // FunctionCodegen - Lambda and closure operations
    // Note: Main implementations still in this file; module provides createClosure helper
    std::unique_ptr<eshkol::FunctionCodegen> func_;

    // TensorCodegen - Tensor operations
    // Note: Main implementations still in this file; module provides interface
    std::unique_ptr<eshkol::TensorCodegen> tensor_;

    // AutodiffCodegen - Automatic differentiation operations
    // Note: Main implementations still in this file; module provides interface
    std::unique_ptr<eshkol::AutodiffCodegen> autodiff_;

    // BindingCodegen - Variable binding operations (define, let, letrec, set!)
    std::unique_ptr<eshkol::BindingCodegen> binding_;

    // HomoiconicCodegen - Quote and S-expression operations
    std::unique_ptr<eshkol::HomoiconicCodegen> homoiconic_;

    // TailCallCodegen - Tail call optimization support
    std::unique_ptr<eshkol::TailCallCodegen> tailcall_;

    // SystemCodegen - System, environment, and file operations
    std::unique_ptr<eshkol::SystemCodegen> system_;

    // HashCodegen - Hash table operations
    std::unique_ptr<eshkol::HashCodegen> hash_;

    // LogicWorkspaceCodegen - Consciousness engine: logic vars, KB, factor
    // graphs, active inference, global workspace, tensor/model serialization
    std::unique_ptr<eshkol::LogicWorkspaceCodegen> logic_workspace_;

    // ParallelCodegen - Parallel execution primitives (parallel-map, parallel-fold, etc.)
    std::unique_ptr<eshkol::ParallelCodegen> parallel_;

    // Local type pointers (initialized from TypeSystem for backward compatibility)
    StructType* tagged_value_type;
    StructType* dual_number_type;
    StructType* ad_node_type;
    StructType* tensor_type;

    // PHASE 3: Current tape for reverse-mode AD
    Value* current_tape_ptr;
    size_t next_node_id;

    std::unordered_map<std::string, Value*> symbol_table;
    std::unordered_map<std::string, Value*> global_symbol_table; // Persistent global symbols
    std::unordered_map<std::string, Function*> function_table;
    std::unordered_map<const eshkol_ast_t*, Function*> declared_functions_by_ast;
    std::unordered_map<std::string, std::vector<std::string>> nested_function_captures; // Free vars for nested defines
    std::unordered_map<std::string, std::string> functions_returning_lambda; // Maps function name -> lambda name it returns

    // NAMED-LET CAPTURE METADATA (#224 architectural fix):
    // Per-loop list of captured free-var NAMES, keyed by the loop Function*.
    // Loop functions take captures as EXTRA pointer args after the regular
    // params.  codegenCall consults this when emitting a non-tail recursive
    // call to a loop function and appends the corresponding pointers from
    // the current symbol_table.  Replaces the old GlobalVariable-shuttle
    // approach which was not thread-safe (two threads invoking the same
    // closure raced on the global, both reading the second store).
    // Storage is per-call: the pointer is either the outer's alloca, the
    // outer's GlobalVariable, the outer's pointer-typed Argument (already a
    // capture-slot), or — for raw-value Arguments — a freshly alloca'd cell
    // seeded with the outer's value at the call site.
    std::unordered_map<llvm::Function*, std::vector<std::string>> named_let_captures;

    // ESCAPED NAMED-LET LOOP PROCEDURE.  A named-let loop function's LLVM
    // signature is (params..., one capture POINTER per captured free var), so
    // it can only be entered by a caller that knows the capture list.  When the
    // loop procedure is used as a first-class VALUE — `(set! g loop)`,
    // `(cons loop '())`, `(map loop xs)` — codegenVariable used to fall through
    // to the generic `function_table` path and hand back the bare Function*,
    // whose capture parameters the closure dispatcher then filled with whatever
    // happened to be in the argument registers.  Calling the leaked procedure
    // dereferenced that garbage as a capture cell (SIGSEGV).  These two maps let
    // codegenVariable build a REAL closure instead: the env holds one tagged
    // int64 per capture (the address of the shared cell, exactly the convention
    // `capturePointerTagFromCurrentFunction` already uses), and a per-loop
    // trampoline re-derefs those slots back into the pointer arguments the loop
    // function expects.
    struct NamedLetEscapeInfo {
        llvm::Function* loop_func = nullptr;
        std::vector<std::string> captures;
        uint64_t arity = 0;
    };
    std::unordered_map<std::string, NamedLetEscapeInfo> named_let_escapes;
    std::unordered_map<llvm::Function*, llvm::Function*> named_let_escape_thunks;

    // HoTT TYPE TRACKING: Maps variable names to their compile-time HoTT types
    // This enables type-directed optimizations when both operand types are known
    std::unordered_map<std::string, eshkol::hott::TypeId> symbol_hott_types;
    std::unordered_map<std::string, eshkol::hott::TypeId> global_symbol_hott_types;

    // HoTT PARAMETERIZED TYPE TRACKING: Maps variable names to parameterized types (List<T>, Vector<T>)
    // This enables element type propagation through car/cdr, vector-ref, etc.
    std::unordered_map<std::string, eshkol::hott::ParameterizedType> symbol_param_types;
    std::unordered_map<std::string, eshkol::hott::ParameterizedType> global_symbol_param_types;

    // HoTT FUNCTION RETURN TYPE TRACKING: Maps function names to their return types
    // Enables type-directed optimization for function call results
    std::unordered_map<std::string, eshkol::hott::TypeId> function_return_types;

    // LETREC REFACTOR: Set of names to exclude from free variable capture (letrec-bound names)
    // This is set by BindingCodegen::letrec before generating lambda bindings and cleared after
    std::set<std::string> letrec_excluded_capture_names;

    // CLOSURE-OVER-NAMED-LET-LOOPVAR FIX: Set of free vars that codegenNamedLet
    // arena-moved because they are set!-mutated inside the loop body. When
    // codegenLambda later captures one of these (it sees the arena pointer as a
    // pointer-typed Argument `..._cap`), it must POINTER-PASS rather than
    // by-value load so the mutation is shared.
    std::set<std::string> mutable_loop_captures_;

    // VARIADIC FUNCTION TRACKING: Maps function name to (fixed_param_count, is_variadic)
    // For variadic functions, when calling, extra args beyond fixed_param_count are packaged into a list
    std::unordered_map<std::string, std::pair<uint64_t, bool>> variadic_function_info;

    // Per-AST mutation summaries. Binding decisions ask the same lexical body
    // once per binding; retaining the flat set! targets turns the common
    // generated N-binding case from repeated whole-body walks into one pass.
    std::unordered_map<const eshkol_ast_t*, std::unordered_set<std::string>>
        flat_mutation_targets_;
    std::unordered_set<const eshkol_ast_t*> flat_mutation_ineligible_;

    // FUNCTION-AS-VALUE FIX: Maps function name to user-facing arity (excludes captures)
    // Used when functions are referenced as values (first-class functions) to wrap them in closures
    std::unordered_map<std::string, uint64_t> function_arity_table;

    // FFI POINTER-ARG GUARD (ESH-0363): the declared parameter type KEYWORDS of
    // every `extern`, keyed by both the Eshkol-visible name and the real C
    // symbol (call sites resolve by either).
    //
    // The LLVM signature alone cannot tell an FFI pointer parameter from an
    // internal one — `ptr`, `string` and `char*` all collapse to the same
    // opaque pointer type, and internal Eshkol functions also take pointer
    // parameters (closure environments, named-let capture slots). Recording
    // the declared keyword at `extern` codegen time is what lets the call site
    // guard exactly the FFI boundary and nothing else.
    std::unordered_map<std::string, std::vector<std::string>> extern_param_type_names_;

    // True when the emitted module can raise into the hosted error runtime.
    // Standalone freestanding and wasm32 objects have no hosted exception path
    // (eshkol_type_error and friends live in the hosted runtime source set), so
    // the FFI pointer-argument guard — which raises a catchable type error — is
    // only emitted for hosted native codegen.
    bool ffiPointerArgGuardEnabled() const;

    /* R7RS §5.3.1 TOP-LEVEL REDEFINITION.
     *
     * "At the top level of a program, a definition
     *      (define <variable> <expression>)
     *  has essentially the same effect as the assignment expression
     *      (set! <variable> <expression>)
     *  if <variable> is bound to a non-syntax value.  However, if
     *  <variable> is not bound [...] the definition will bind <variable>
     *  to a new location before performing the assignment."
     *
     * So a top-level name has exactly ONE location, every definition of it
     * assigns to that location in program order, and every reference reads
     * the location when it is evaluated — not when it is compiled.
     *
     * Eshkol's fast path deliberately violates that in exchange for direct
     * calls: a `(define (f ...) ...)` becomes an LLVM Function resolved at
     * the call site through function_table / `<name>_func`, while a
     * `(define f <expr>)` becomes a tagged_value GlobalVariable resolved
     * through symbol_table.  Two definitions of one name therefore produced
     * two independent bindings, and the call site picked between them by
     * namespace priority and by codegen order rather than by program order.
     *
     * This set holds the names that are defined more than once at the top
     * level of the current compilation unit.  For exactly those names the
     * fast path is switched off and the R7RS single-location model is used:
     * the location is a tagged_value global (created in Step 1.5), every
     * definition — value or procedure — stores into it at its own position
     * in program order (Step 3), and every reference loads from it.  Single
     * definitions, which is every name in practice, keep the direct call.
     */
    std::unordered_set<std::string> redefined_toplevel_names;
    std::unordered_set<std::string> toplevel_callee_names;

    /* Names whose top-level binding is reassigned: defined more than once, or
     * the target of a set! at top-level scope. Such a binding never gets a
     * static `<name>_func` alias (static_callee_binding.h). Answered lazily
     * per name from the unit's top-level forms and memoised, so only names
     * that are actually bound to lambdas pay for the scan. */
    const eshkol_ast_t* toplevel_asts_for_reassignment = nullptr;
    size_t num_toplevel_asts_for_reassignment = 0;
    std::unordered_map<std::string, bool> reassigned_toplevel_memo;

    // ESH-0078: Maps a defined function name to its source body AST, so an AD
    // operator applied to a NAMED function (via var) can run the same
    // source-level tensor-flow analysis (adAstUsesTensorOps) that inline
    // lambdas get. Populated at define codegen; consumed by AutodiffCodegen.
    std::unordered_map<std::string, const eshkol_ast_t*> function_body_ast;

    // ESH-0187: full DEFINE node per function name (parameters + body), so a
    // NAMED single-arg function passed to (derivative-n f x K)/(taylor f x K)
    // can be resolved to (param, body) for P2 compile-time-K monomorphization.
    std::unordered_map<std::string, const eshkol_ast_t*> function_def_ast;

    // MIGRATED: String interning moved to CodegenContext::interned_strings_
    // StringIOCodegen::createString() handles string interning now

    // Current function being generated
    Function* current_function;
    BasicBlock* main_entry;

    // Arena management for list operations - GLOBAL ARENA ARCHITECTURE
    GlobalVariable* global_arena; // Global arena pointer (shared across all scopes)
    size_t arena_scope_depth; // Track nested arena scopes

    // PHASE 1 AUTODIFF FIX: Global AD mode flag for runtime context detection
    GlobalVariable* ad_mode_active; // Global flag: true when executing in AD context

    // PHASE 1 AUTODIFF FIX: Global tape pointer for runtime graph recording
    GlobalVariable* current_ad_tape; // Global tape pointer: set by gradient/jacobian/etc operators

    // NESTED GRADIENT FIX: Tape stack for arbitrary-depth nested gradients
    // Allows inner gradients to save/restore outer gradient context
    static const size_t MAX_TAPE_DEPTH = 32; // Support up to 32 levels of nesting
    GlobalVariable* ad_tape_stack;  // Array of tape pointers [MAX_TAPE_DEPTH]
    GlobalVariable* ad_tape_depth;  // Current stack depth (0 = no active gradient)
    GlobalVariable* ad_pert_level;  // ESH-0070 forward-mode perturbation level (runtime)
    GlobalVariable* ad_tower_active; // ESH-0190 P5: Taylor-tower diff depth (>0 while active)
    GlobalVariable* ad_tower_order;  // ESH-0190 P5: current innermost tower order

    // DOUBLE BACKWARD: Storage for outer AD node when in nested gradient
    // Used to connect inner gradient's result to outer's computation graph
    GlobalVariable* outer_ad_node_storage;  // Pointer to outer AD node (or null if not nested)
    GlobalVariable* outer_ad_node_to_inner; // Maps outer AD node to inner variable node
    GlobalVariable* outer_grad_accumulator; // AD node accumulating gradient on outer tape
    GlobalVariable* inner_var_node_ptr;     // Pointer to the inner variable node (for matching)
    GlobalVariable* gradient_x_degree;      // Polynomial degree of gradient in x (for double backward)

    // N-DIMENSIONAL DERIVATIVES: Stack of outer AD nodes for arbitrary depth nesting
    GlobalVariable* outer_ad_node_stack;    // Array of outer AD node pointers [MAX_TAPE_DEPTH]
    GlobalVariable* outer_ad_node_depth;    // Current depth in the outer AD node stack

    // Note: Arena functions are now in MemoryCodegen (mem->)
    // Forwarding accessors are provided below for backward compatibility

    // Deep equality comparison for nested lists
    Function* eshkol_deep_equal_func;

    // Unified display system (Phase 4 - homoiconic display)
    Function* eshkol_display_value_func;
    Function* eshkol_lambda_registry_init_func;
    Function* eshkol_lambda_registry_add_func;
    Function* eshkol_lambda_registry_lookup_func;

    // Recursive tensor display helper (N-dimensional nested structure)
    Function* display_tensor_recursive_func;

    // Cached LLVM types (avoid repeated lookups - massive performance win)
    // Note: C library functions are now in FunctionCache (funcs->)
    IntegerType* int64_type;
    IntegerType* int32_type;
    IntegerType* int16_type;
    IntegerType* int8_type;
    IntegerType* int1_type;
    IntegerType* size_type;    // i32 on wasm32, i64 on native — for size_t params
    IntegerType* intptr_type;  // i32 on wasm32, i64 on native — for intptr_t (PtrToInt, closure func_ptr)
    Type* double_type;
    Type* void_type;
    Type* ptr_type;

    // Helper: create a size_t constant (target-dependent width)
    Value* sizeConst(uint64_t n);

    // Helper: create an intptr_t constant (target-dependent width)
    Value* intPtrConst(uint64_t n);

    // Helper: truncate or extend a value to intptr_t width
    Value* toIntPtr(Value* v);

    void markGlobalValueUsed(GlobalValue* value);

    bool applyExportedSymbolName(const char* export_name,
                                 const char* source_name,
                                 GlobalValue* value);

    void applyDefineObjectAttributes(const decltype(((eshkol_operations_t*)nullptr)->define_op)& def,
                                     GlobalObject* object);

    void applyDefineVariableAttributes(const decltype(((eshkol_operations_t*)nullptr)->define_op)& def);

    void applyDefineFunctionAttributes(const decltype(((eshkol_operations_t*)nullptr)->define_op)& def,
                                       Function* function);

    void applyExternFunctionAttributes(const decltype(((eshkol_operations_t*)nullptr)->extern_op)& ext,
                                       Function* function);

    // LIBRARY MODE: When true, skip main function creation and export all symbols
    bool library_mode;
    bool freestanding_codegen_;
    // SHARED-LIBRARY EXPORT ABI: true only for the linked --shared-lib
    // flavour.  See g_shared_library_exports and
    // emitSharedLibraryExportWrappers().
    bool shared_library_exports_ = false;
    // WASM MODE: True when the module targets a standalone
    // wasm32-unknown-unknown object. Like freestanding native, a standalone
    // wasm module has no hosted REPL to introspect function sources, so the
    // homoiconic display registry is skipped. Skipping it is also what lets
    // the wasm dead-strip (internalize + globalDCE) remove unused stdlib:
    // the registry would otherwise address-take (ptrtoint) every top-level
    // function from main(), pinning the entire stdlib against DCE.
    bool wasm_codegen_ = false;
    bool fatal_codegen_error_;

    // The homoiconic display registry eagerly registers every top-level
    // function (name + source S-expression + function pointer) so a hosted
    // program can `(display <fn>)` and see its source. Standalone freestanding
    // and wasm objects have no such host, and the eager registration both
    // bloats the module and pins every function against dead-code elimination.
    bool homoiconicRegistryEnabled() const;

    // SW-10: the cooperative execution-timeout poll on the tail-call back-edge
    // only exists where something can request an interrupt. The requester is
    // the hosted watchdog thread in lib/core/resource_limits.cpp, which is not
    // in the freestanding source set and is not linked into a standalone wasm
    // module at all — so in those profiles the poll can never observe an
    // interrupt, and emitting it only creates a dependency on a symbol the
    // profile does not have. On wasm32 that dependency is an `env` import the
    // JS glue would have to stub, and the stub would then be called across the
    // JS boundary on every iteration of every loop in the program.
    //
    // Same direction as the VM's limit installer (see
    // eshkol_vm_install_limits): a build with no hosted runtime to push the
    // configuration in keeps the compiled-in default and links nothing extra.
    bool timeoutInterruptPollEnabled() const;

    // Module prefix for unique lambda naming (prevents symbol collision when linking)
    std::string module_prefix;

    void markFatalCodegenError() __attribute__((noinline, used));

    // DWARF DEBUG INFO: DIBuilder and metadata for source-level debugging
    std::unique_ptr<DIBuilder> di_builder_;
    DICompileUnit* di_cu_ = nullptr;
    DIFile* di_file_ = nullptr;
    std::vector<DIScope*> di_scope_stack_;
    bool emit_debug_info_ = false;
    std::string source_filename_;

public:
    EshkolLLVMCodeGen(const char* module_name, bool is_library_mode = false,
                       const char* target_triple = nullptr,
                       bool is_freestanding_codegen = false);

    // Register return types for builtin functions
    void registerBuiltinReturnTypes();

    std::pair<std::unique_ptr<Module>, std::unique_ptr<LLVMContext>> generateIR(const eshkol_ast_t* asts, size_t num_asts);

private:
    // C library function getters (forwarding to FunctionCache)
    Function* getStrlenFunc();
    Function* getMallocFunc();
    Function* getMemcpyFunc();
    Function* getMemsetFunc();
    Function* getStrcmpFunc();
    Function* getStrcpyFunc();
    Function* getStrcatFunc();
    Function* getStrstrFunc();
    Function* getSnprintfFunc();
    Function* getStrtodFunc();

    void createBuiltinFunctions();

    void registerArenaFunctions() __attribute__((noinline, used));

    // Note: Old createArenaFunctions was ~600 lines - now in MemoryCodegen

    // Arena function accessors (forwarding to MemoryCodegen)
    Function* getArenaCreateFunc();
    Function* getGlobalArenaFunc();
    Function* getArenaDestroyFunc();
    Function* getArenaAllocateFunc();
    Function* getArenaPushScopeFunc();
    Function* getArenaPopScopeFunc();
    Function* getArenaAllocateConsCellFunc();
    Function* getArenaAllocateClosureFunc();
    Function* getArenaAllocateClosureWithHeaderFunc();
    Function* getArenaAllocateTaggedConsCellFunc();
    Function* getArenaAllocateConsWithHeaderFunc();
    Function* getTaggedConsGetInt64Func();
    Function* getTaggedConsGetDoubleFunc();
    Function* getTaggedConsGetPtrFunc();
    Function* getTaggedConsSetInt64Func();
    Function* getTaggedConsSetDoubleFunc();
    Function* getTaggedConsSetPtrFunc();
    Function* getTaggedConsSetNullFunc();
    Function* getTaggedConsGetTypeFunc();
    Function* getTaggedConsGetFlagsFunc();
    Function* getTaggedConsSetTaggedValueFunc();
    Function* getTaggedConsGetTaggedValueFunc();
    Function* getArenaAllocateTapeFunc();
    Function* getArenaTapeAddNodeFunc();
    Function* getArenaTapeResetFunc();
    Function* getArenaTapeGetNodeFunc();
    Function* getArenaTapeGetNodeCountFunc();
    Function* getArenaAllocateAdNodeFunc();

    Value* unavailableParallelBuiltin(const std::string& func_name);

    // Legacy code removed - arena functions are now in MemoryCodegen (~600 lines saved)
    // The old createArenaFunctions() method has been replaced by:
    //   1. MemoryCodegen construction in generateIR()
    //   2. registerArenaFunctions() for function_table population
    //   3. Forwarding accessors above for direct function pointer access

    // ===== OLD ARENA FUNCTION DECLARATIONS REMOVED (see memory_codegen.cpp) =====
    // Approximately 560 lines of arena function declarations were moved to MemoryCodegen

    void createDisplayTensorRecursiveFunction();

    /* R7RS §5.3.1: find the top-level names that are defined more than once
     * in this compilation unit.  Those are the names whose binding must be a
     * single mutable location assigned in program order (see the
     * redefined_toplevel_names comment) instead of a compile-time-resolved
     * direct call.
     *
     * `:external` defines are skipped: they are the precompiled stdlib's
     * bindings, re-materialised into this AST by process_requires with their
     * bodies living in the linked object.  A user define of the same name is
     * a *shadow* of a separately compiled unit, not an in-unit redefinition,
     * and is already handled by the user-shadows-builtin path (audit Bug G).
     * Counting them here would drag every stdlib-shadowing call onto the
     * indirect path for no semantic gain.
     */
    void collectRedefinedTopLevelNames(const eshkol_ast_t* asts, size_t num_asts);

    bool isRedefinedTopLevelName(const char* name) const;

    /* Is the top-level binding of `name` reassigned in this compilation
     * unit: defined more than once, or the target of a set! anywhere at
     * top-level scope (inside any top-level form, unless a nearer binding of
     * the same name shadows it)? A top-level lambda binding of such a name
     * must not be resolved statically: the lambda it was created with is not
     * what the name denotes after the assignment. */
    bool isReassignedTopLevelName(const char* name);

    /* Wrap a top-level LLVM function in a zero-capture arena closure and
     * return it as a CALLABLE tagged_value — the first-class value of a
     * procedure definition.  Shared by the function-as-value path in
     * codegenVariable and by the R7RS §5.3.1 redefinition stores, so both
     * produce byte-identical closures (same packed_info / return_type_info
     * layout the closure-call path decodes). */
    /* Is `name` a REST-ARG (variadic) procedure, and if so how many FIXED
     * parameters does it take? Consults both registries, exactly as the call
     * path in codegenCall does: `variadic_function_info` for the ordinary
     * compile, `g_repl_variadic_functions` for REPL/`-e` batches. Both are
     * keyed by the user-visible name ("append"), never by the mangled LLVM
     * symbol REPL hot-reload produces ("append__rv0"), so callers pass the
     * user name first and the mangled form only as a fallback. */
    bool lookupVariadicProcedure(const std::string& name,
                                 uint64_t* fixed_params_out);

    /* Turn an LLVM function into a first-class Eshkol callable.
     *
     * SW-27 — the variadic arguments are not decoration. A rest-arg procedure
     * `(define (append . lists) …)` is called through a DIFFERENT ABI from a
     * fixed-arity one: the closure dispatcher (codegenClosureCall) conses the
     * caller's surplus arguments into a list and passes `fixed_params` values
     * plus that one list. It selects that path from the closure metadata —
     * bit 63 of `packed_info`, which the allocator turns into
     * CLOSURE_FLAG_VARIADIC, and, for the 0-capture case where `env` is null,
     * the `input_arity` byte, which it reads back as `fixed_params`.
     *
     * Before this, every function-as-value site packed a plain arity and no
     * variadic bit, so a rest-arg procedure referenced as a value was called
     * as if fixed-arity: its rest parameter received a bare argument instead
     * of a list. `(h append '(1) '(2))` answered `1` — the first argument,
     * silently — and `(h string-copy "abc")` SIGSEGV'd under AOT walking a
     * non-list as a list. Call position was always correct; only the value
     * representation lied about the procedure's shape.
     *
     * NOTE the asymmetry in what `input_arity` must hold: for a fixed-arity
     * closure it is the parameter count, but for a variadic one the dispatcher
     * reads it as the FIXED count (the rest parameter is not among them). A
     * rest-arg procedure with one fixed parameter stores 1, not 2.
     */
    Value* emitFunctionAsCallableValue(Function* func, uint64_t num_params,
                                       bool is_variadic = false,
                                       uint64_t fixed_params = 0,
                                       uint8_t return_category = CLOSURE_RETURN_UNKNOWN);

    /* emitFunctionAsCallableValue for a NAMED procedure: resolves the
     * rest-arg shape from the registries so callers cannot forget to. Every
     * "this name is being read as a value" site should use this rather than
     * packing arity by hand. */
    Value* emitNamedFunctionAsCallableValue(Function* func,
                                            const std::string& user_name,
                                            uint64_t num_params);

    /* Store a top-level PROCEDURE definition whose binding is mutable.
     * Immutable procedure definitions need no store — their call sites
     * resolve the LLVM function directly — but a reassigned name is bound
     * through a location, so each definition assigns its procedure there at
     * its own point in program order. Emitted from the
     * main/global-init sequence, where `builder` is already positioned at
     * the definition's place in the top-level order. */
    void emitMutableProcedureStoreForFunctionDefine(const eshkol_ast_t* ast);


    // DWARF DEBUG INFO: build the DISubprogram for an Eshkol `define`d function.
    //
    // `is_definition` decides the *flavour* of the node, and that distinction is
    // load-bearing for LLVM's verifier, not cosmetic:
    //
    //   * a definition subprogram is a *distinct* MDNode owned by the compile
    //     unit -- it is what produces a DW_TAG_subprogram with code ranges, and
    //     it is the only kind an LLVM Function *with a body* may carry;
    //   * a declaration subprogram is *uniqued* and unit-less -- the only kind a
    //     bodyless Function may carry ("function declaration may only have a
    //     unique !dbg attachment", Verifier::visitFunction).
    //
    // Attaching a definition subprogram to a function that never receives a body
    // is therefore a hard IR verification error, which is exactly what broke
    // every `-g` build: `createFunctionDeclaration` runs over *all* top-level
    // defines, including the `:external` ones that a `(require <stdlib module>)`
    // produces, whose bodies live in the pre-linked stdlib object and are skipped
    // by codegenFunctionDefinition. So `@caar`, `@cadr`, `@caadr`, `@cadar`,
    // `@caddr`, `@cddr`, ... all ended up as declarations wearing a definition
    // subprogram and the module failed to verify before a single byte of output
    // was written.
    //
    // The invariant is now structural rather than predicted: declarations get the
    // declaration flavour here, and codegenFunctionDefinition upgrades to the
    // definition flavour at the point it actually creates the entry block. A
    // function can only ever carry a definition subprogram if it has a body.
    DISubprogram* createDefineSubprogram(Function* function,
                                        const char* func_name,
                                        unsigned line_no,
                                        uint64_t num_params,
                                        bool is_variadic,
                                        bool is_definition);

    // DWARF DEBUG INFO: re-anchor the builder's current debug location to the
    // function we are emitting into *right now*.
    //
    // IRBuilder's current debug location is sticky, but codegen moves the
    // insertion point across function boundaries constantly (lambda bodies,
    // nested defines, pre-generated helpers, AD/tensor thunks). A DILocation is
    // only legal inside the function whose DISubprogram scopes it, so an
    // inherited location is an IR verification error, not a cosmetic slip:
    //
    //   * a location scoped to another function's subprogram gives
    //     "!dbg attachment points at wrong subprogram for function" -- e.g. the
    //     entry-block allocas of `make-counter` were still scoped to `fact`,
    //     simply because `fact`'s body was emitted just before it;
    //   * no location at all, in a function that has debug info, gives
    //     "inlinable function call in a function with debug info must have a
    //     !dbg location" for every call in the entry scaffolding emitted before
    //     the first AST node was visited.
    //
    // Both are the same defect -- trusting a sticky location across a scope
    // change -- so the location is derived from the current insertion point
    // instead of being trusted to still apply.
    //
    // `line == 0` means "no source position of its own": keep the current line
    // if it is already scoped to this function, otherwise fall back to the
    // function's scope line.
    void anchorDebugLocation(unsigned line, unsigned column);

    // Anchor to the start of whatever function is now being emitted into. Called
    // right after a function body's entry block becomes the insertion point, and
    // after an insertion-point restore that crosses a function boundary.
    void anchorDebugLocationToCurrentFunction();

    void createFunctionDeclaration(const eshkol_ast_t* ast);

    // Recursively traverse AST to find nested function definitions
    // Pre-generate top-level lambdas so user-defined main can reference them
    // This is needed because user's main is compiled in Step 2, before Step 3 processes global defines
    void preGenerateTopLevelLambdas(const eshkol_ast_t* asts, size_t num_asts);

    // Note: We don't pre-declare nested functions anymore - they're generated at codegen time
    // with proper closure support (like lambdas)
    void declareNestedFunctions(const eshkol_ast_t* ast);

    bool isLibraryInitAST(const eshkol_ast_t& ast) const __attribute__((noinline, used));

    void codegenLibraryInitAST(const eshkol_ast_t& ast) __attribute__((noinline, used));

    Function* createLibraryInitChunkFunction(
        FunctionType* init_type,
        size_t chunk_index,
        const eshkol_ast_t* asts,
        const std::vector<size_t>& init_indices,
        size_t begin_index,
        size_t end_index
    ) __attribute__((noinline, used));

    void emitLambdaSExprRegistration(const LambdaSExprMetadata& meta) __attribute__((noinline, used));

    Function* createLibraryLambdaSExprChunkFunction(
        FunctionType* init_type,
        size_t chunk_index,
        size_t begin_index,
        size_t end_index
    ) __attribute__((noinline, used));

    void finalizeLibrarySymbols(const eshkol_ast_t* asts, size_t num_asts) __attribute__((noinline, used));

    /* ── SHARED-LIBRARY EXPORT ABI ───────────────────────────────────────────
     *
     * THE DEFECT this exists to fix: LLVM's calling convention for a
     * first-class-struct return is NOT the platform C calling convention for
     * the same struct, and `--shared-lib` was exporting the former while
     * promising the latter.
     *
     * `eshkol_tagged_value_type` is the raw 5-field struct
     * `{i8, i8, i16, i32, i64}` (type, flags, reserved, explicit padding,
     * data) — a field-for-field mirror of the C `eshkol_tagged_value_t`, which
     * is exactly right for the LAYOUT of loads and stores.  It is not a
     * description of how the value travels in registers.  Given a
     * `define {i8,i8,i16,i32,i64} @f()`, the backend flattens the aggregate
     * into one return value PER FIELD and assigns them to consecutive return
     * registers.  On AArch64 that is:
     *
     *     _f:  mov w0, #1    ; type
     *          mov w1, #0x10 ; flags       <- C caller reads this as data
     *          mov w2, #0    ; reserved
     *          mov w3, #0    ; padding
     *          mov w4, #42   ; data        <- the payload, in x4
     *          ret
     *
     * A C caller compiled against `eshkol_tagged_value_t` follows AAPCS: a
     * 16-byte non-HFA composite comes back in x0:x1, so it reads
     * `{type = 1, flags = 0, data = 16}` — the flags byte, zero-extended,
     * masquerading as the payload.  `42` is stranded in x4.  Arguments were
     * corrupted the same way and worse: with two tagged parameters the
     * flattened fields overflow the 8 argument registers and the callee reads
     * its second argument off the stack, while the C caller passed it in
     * x2:x3.
     *
     * Both directions are affected, so the wrapper coerces both.
     *
     * THE FIX: leave the internal convention alone — every Eshkol-to-Eshkol
     * call in the module is consistent with itself, and the struct type is
     * what makes GEP-based field access correct — and coerce only at the
     * export boundary.  The user's function keeps its body under a decorated
     * symbol and the exported name becomes a thin thunk with the platform C
     * signature.  This is the same coercion the REPL already performs in the
     * opposite direction when it calls a native C function that returns a
     * tagged value (see tryResolveReplFunction / `__cabi_thunk`); it was
     * simply never wired for shared-library exports.
     *
     * PER-PLATFORM COERCION
     *
     *   AArch64 (AAPCS64) and x86-64 SysV, and every other 64-bit ELF/Mach-O
     *   target we emit for (riscv64 LP64D, ppc64le ELFv2, loongarch64):
     *   a 16-byte composite of two integer eightbytes is returned in the
     *   first two GPRs (x0:x1 / rax:rdx) and passed in two consecutive GPRs.
     *   `[2 x i64]` is precisely what Clang coerces such a struct to, in both
     *   directions, so the thunk's signature is
     *   `[2 x i64] @f([2 x i64], …)`.  Windows on ARM64 follows AAPCS64 here
     *   and is covered by the same case.
     *
     *   Windows x64 (Microsoft ABI) is different and needs the memory form:
     *   an aggregate whose size is not exactly 1, 2, 4 or 8 bytes is returned
     *   through a hidden pointer supplied by the caller in RCX, and passed as
     *   a pointer to a caller-owned temporary.  A 16-byte tagged value hits
     *   both rules, so the thunk there is
     *   `void @f(ptr sret(%tv) noalias, ptr, …)` — which is byte-identical to
     *   what MSVC/clang-cl generate for
     *   `eshkol_tagged_value_t f(eshkol_tagged_value_t …)`.
     *
     *   Anything else (32-bit targets, where the return is via memory but the
     *   arguments are by-value on the stack, so neither shape above applies)
     *   is refused rather than silently mis-exported.
     */
    enum class SharedLibraryExportAbi {
        RegisterPair,  // [2 x i64] in and out (AAPCS64, SysV, RISC-V, …)
        MemorySret,    // sret return + by-pointer arguments (Windows x64)
        Unsupported
    };

    SharedLibraryExportAbi sharedLibraryExportAbi() const;

    // Symbol suffix carrying the unwrapped, internal-convention definition of
    // an exported function.  Kept exported (not internalized) so a linker map
    // or a debugger still names the real body, and so a future Eshkol-to-
    // Eshkol dynamic link has a struct-convention entry point to bind to.
    static std::string sharedLibraryImplSymbolName(const std::string& name);

    /* ── FREESTANDING C-ABI EXPORTS ──────────────────────────────────────────
     *
     * Inside a module every Eshkol function speaks the internal tagged-value
     * convention.  A freestanding image is entered from code that does not:
     * a reset handler, a vector table, a C board-support layer.  So in a
     * freestanding profile an `:export-symbol` function whose parameters and
     * return are all annotated with machine types (u8..u64, i8..i64, usize,
     * isize, ptr; `null` return) is exported under its symbol name with the
     * platform C ABI for those scalars: the tagged body moves to an internal
     * `<symbol>.eshkol` definition and a thunk under the exported name boxes
     * the native arguments, calls it, and unboxes the result.  In-module
     * Eshkol callers hold the body by pointer and keep the internal
     * convention.  `:section`, `:used`, `:weak` and `:no-return` apply to
     * the exported entry.
     */
    Type* freestandingExportScalarType(eshkol::hott::TypeId t, bool allow_void);

    void emitFreestandingScalarExportWrappers(const eshkol_ast_t* asts, size_t num_asts);

    void emitSharedLibraryExportWrappers(const eshkol_ast_t* asts, size_t num_asts);

    // LIBRARY MODE: Create initialization function instead of main
    // This function initializes global state but doesn't create an entry point
    void createLibraryInitFunction(const eshkol_ast_t* asts, size_t num_asts);

    /* A freestanding image has no unwinder and no exception runtime, so no
     * function in it can unwind. Saying so keeps targets with table-driven
     * unwinding (ARM EHABI) from referencing personality routines such as
     * __aeabi_unwind_cpp_pr0 that only a hosted C++ runtime provides. */
    void markFreestandingNoUnwind();

    void pruneUnusedFreestandingDeclarations();

    // ESH-0216: pair the eshkol_runtime_init() call emitted at C main()
    // entry with a matching eshkol_runtime_shutdown() right before main
    // returns. Without this, a standalone AOT-compiled Eshkol binary never
    // goes through the runtime's own ordered teardown at all: worker-thread
    // pools (parallel-map/parallel-execute) never get stopped/joined ahead
    // of shutdown hooks, and the fatal-signal handler / altstack installed
    // by eshkol_runtime_init_signals() is never uninstalled — process exit
    // relies solely on the unordered libc atexit/static-destructor chain.
    // That is the SIGSEGV-after-"graceful shutdown" race fixed in ESH-0216:
    // a still-running worker thread dereferences shared state a shutdown
    // hook (or other teardown) frees out from under it, well after the
    // program logged its own graceful-shutdown message. Call this at every
    // site that emits eshkol_runtime_init(), immediately before the
    // corresponding CreateRet. Uses the same REPL/WASM guard as the init
    // call: the REPL keeps the process alive across evaluations (it calls
    // eshkol_runtime_shutdown() itself once, from exe/eshkol-repl.cpp, at
    // real process exit), and WASM has no runtime_init call to pair with.
    /**
     * ESH-0101 / SW-81: emit the native stack-headroom check at the entry of a
     * user function body.
     *
     * Plain non-tail user recursion had NO guard at all. The frame-counting
     * `eshkol_check_recursion_depth` is emitted only at lambda entry, and its
     * ceiling (ESHKOL_MAX_STACK, default 100000) is a proxy for the wrong
     * resource: raising it to cover legitimate 250k-frame recursion would make
     * it useless, and lowering it to fire before overflow would break programs
     * that fit. So the stack ran into its guard page and the process died with
     * a bare SIGSEGV/SIGILL and no message.
     *
     * This call watches the actual resource — bytes left on this thread's
     * stack — and is stateless, so unlike the depth counter it needs no
     * matching decrement before returns and therefore cannot interfere with
     * tail-call optimization.
     *
     * Not emitted for freestanding native or wasm objects: those profiles do
     * not link the hosted stack runtime. A freestanding object must remain
     * free of hosted imports, and wasm has no RLIMIT_STACK to read.
     */
    void emitStackGuardCheck();

    void emitRuntimeShutdownBeforeMainReturn();

    void createMainWrapper();

    void initializeArena();

    Value* getArenaPtr();

    // Mixed type arithmetic helper functions
    TypedValue promoteInt64ToDouble(const TypedValue& int64_val);

    std::pair<TypedValue, TypedValue> promoteToCommonType(const TypedValue& left, const TypedValue& right);

    // Read the HoTT type inferred for an AST node during the type-checking phase
    // WITHOUT emitting any LLVM IR.
    //
    // SINGLE-EVAL FIX (ESH-0098): The typed-builtin branches below (list, vector,
    // cons, car, cdr, vector-ref, addr-of) used to call codegenTypedAST() on their
    // operand ASTs purely to recover an element/pointee type, then call
    // codegenAST(ast) to actually build the value — which re-generated those same
    // operands. That double-generated any side-effecting operand (e.g.
    // (list (bump!)) called bump! twice; (vector ...) twice per element). The value
    // builders (CollectionCodegen::list, codegenVector, ...) already evaluate each
    // operand exactly once, so for type recovery we must NOT emit IR again. The
    // type checker (run over every top-level form before codegen) annotates each
    // sub-expression's inferred_hott_type, so we read that directly. Falls back to
    // the generic Value type when a node was not type-checked (0), which is safe.
    eshkol::hott::TypeId inferredHottType(const eshkol_ast_t* ast) const;

    // Recover the element type of a collection expression WITHOUT emitting IR.
    // The type checker records only the base List/Vector type in inferred_hott_type
    // (element precision is not packed), so for accessors like car/cdr/vector-ref we
    // recover the element type directly from a nested constructor's operand types.
    // Returns the generic Value type when it cannot be determined cheaply. Used by
    // the SINGLE-EVAL FIX (ESH-0098) to avoid re-generating the collection operand.
    eshkol::hott::TypeId inferredElementTypeOf(const eshkol_ast_t* coll_ast) const;

    // Create TypedValue from AST node
    // The central expression/call/arithmetic descent runs on heap-owned
    // continuations. Keep these synchronous entry points for backend callbacks;
    // recursive edges within that descent must await the Task variants so a
    // source nesting level never adds another native dispatch frame. Locals
    // (including provenance guards and earlier operands) live until the parent
    // resumes, preserving the existing evaluation and IR insertion order.
    TypedValue codegenTypedAST(const eshkol_ast_t* ast);

    ContinuationTask<TypedValue> codegenTypedASTTask(const eshkol_ast_t* ast);

private:
    // Function context management for isolation
    struct FunctionContext {
        std::unordered_map<std::string, Function*> local_functions;
        std::vector<std::string> created_functions;
    };

    std::stack<FunctionContext> function_contexts;

    void pushFunctionContext();

    void popFunctionContext();

    void registerContextFunction(const std::string& name, Function* func);

    Function* resolveFunctionByLogicalName(const std::string& name) __attribute__((noinline, used));

    // REPL MODE: Check if a function exists in the global REPL context and create external declaration
    Function* tryResolveReplFunction(const std::string& func_name);

    Value* codegenArenaConsCell(Value* car_val, Value* cdr_val);
    // Phase 3B: Simplified tagged cons cell allocation - direct tagged_value storage!
    Value* codegenTaggedArenaConsCell(const TypedValue& car_val, const TypedValue& cdr_val);

    // Allocate a cons cell whose car and cdr are COMPLETE copies of two
    // tagged values: type byte, flags and payload. A list builtin that moves
    // an existing element into a new list must use this (or
    // codegenTaggedArenaConsCellFromTaggedValue); reducing the element to its
    // 64-bit payload first loses its type, which is how remove and split-at
    // returned the raw bits of inexact elements.
    Value* codegenTaggedArenaConsCellCopying(Value* car_tagged, Value* cdr_tagged);

    // ROBUST SOLUTION: Create cons cell directly from tagged_value with type preservation
    // This stores the VALUE from tagged_value into the cons cell car, preserving the type
    Value* codegenTaggedArenaConsCellFromTaggedValue(Value* car_tagged, Value* cdr_tagged);

    // ===== TAGGED VALUE HELPER FUNCTIONS =====
    // Pack/unpack values to/from eshkol_tagged_value_t structs

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* packInt64ToTaggedValue(Value* int64_val, bool is_exact = true);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* packBoolToTaggedValue(Value* bool_val);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* packDoubleToTaggedValue(Value* double_val);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* packPtrToTaggedValue(Value* ptr_val, eshkol_value_type_t type, uint8_t flags = 0);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* packPtrToTaggedValueWithFlags(Value* ptr_val, Value* type_val, Value* flags_val);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* packNullToTaggedValue();

    /** The unspecified value (ADR-0024). */
    Value* packUnspecifiedToTaggedValue();

    // Ensure a value is in tagged format
    // codegenAST returns raw i64/double for primitives, but tagged structs for complex types
    // This function checks the LLVM type and packs raw values into tagged format
    Value* ensureTaggedValue(Value* val);

    // Apply an optional Scheme converter exactly once.  Converter execution is
    // generated here (rather than in the C parameter runtime) because a
    // converter is an arbitrary Eshkol closure.
    Value* codegenOptionalParameterConverter(Value* converter, Value* value,
                                             const char* context_name);

    Value* codegenParameterConverterFor(Value* param_ptr, Value* value,
                                        const char* context_name);

    Value* codegenParameterRef(Value* param_ptr);

    Value* codegenParameterSet(Value* param_ptr, Value* value);

    Value* codegenParameterPush(Value* parameter, Value* value);

    Value* codegenParameterPop(Value* parameter);

    Value* codegenMakeParameter(const eshkol_operations_t* op);

    /* Argument source for a RUNTIME-COUNTED closure call (see codegenClosureCall).
     *
     * Normally a call site knows statically how many arguments it passes, so
     * `call_args` carries them. A caller that only learns the count at run time
     * — the AD gradient spreading a point into a runtime closure's declared
     * number of scalar parameters — instead hands over a staging array plus the
     * runtime count. The dispatcher already switches on a runtime argument
     * count (it has to: `fixed_params` comes from the closure), so this mode
     * reuses that one switch instead of making the caller emit a separate,
     * fully expanded closure call per possible arity.
     *
     * Contract for the caller:
     *   - `args_ptr` points to `width` contiguous tagged_value slots,
     *   - slots [0, count) hold the live arguments,
     *   - slots [count, width) are initialised to tagged null (they are loaded
     *     unconditionally, and are what the arity-mismatch padding reads),
     *   - `count` is already clamped to [0, width].
     */
    struct ClosureSpreadArgs {
        Value* args_ptr = nullptr;
        Value* count = nullptr;
        int width = 0;
        // When set, variadic dispatch forwards this original proper list
        // after splitting the runtime fixed prefix; no argument cap applies
        // to the rest tail.
        Value* list_int = nullptr;
        Value* full_count = nullptr;
    };

    // A resolved ABI signature (e.g. a checked REPL forward reference) already
    // has its fixed/rest arguments marshalled. Invocation still belongs here.
    struct KnownCallableTarget {
        FunctionType* signature;
        Value* address;
    };

    // WASM call_indirect requires the callee's exact capture tail. Cache one
    // non-inline dispatcher per user-argument arity in the module: placing the
    // 65-way capture switch at every call site would multiply compiler work.
    // Native caller-cleanup calls keep their existing padded fast path.
    Value* codegenWasmSmallClosureCall(Value* function_ptr, Value* captures,
                                     Value* capture_count,
                                     const std::vector<Value*>& arguments);

    // Runtime closure call dispatcher - supports variadic closures with up to 16 captures
    // This is essential for N-dimensional lambda calculus and AD operations
    // void eshkol_procedure_call_error(const eshkol_tagged_value_t* callee,
    //     int64_t expected, int64_t got, int64_t variadic)  -- never returns.
    Function* getProcedureCallErrorFunc();

    Value* codegenClosureCall(Value* func_result, const std::vector<Value*>& call_args,
                              const char* caller_info = "unknown",
                              bool parameter_dispatch = true,
                              const ClosureSpreadArgs* spread = nullptr,
                              const KnownCallableTarget* known = nullptr);

    Value* codegenClosureSpreadCall(Value* closure, Value* slots, Value* count, int width);

    // Apply a tagged list without materialising its unbounded tail.  The
    // staging array contains only the fixed-prefix window; variadic dispatch
    // receives the original list and forwards its complete remaining tail.
    Value* codegenClosureListCall(Value* closure, Value* list_int);

    /* ================= runtime-closure arity spread (AD gradient) =============
     *
     * A gradient of a RUNTIME closure has to call that closure with its own
     * declared number of scalar arguments, and that number is only known at run
     * time. GRAD_MAX_ARITY is the supported ceiling; above it the call raises a
     * named error instead of quietly taking the single-vector path (which would
     * leave the loss's remaining parameters uninitialised — the silent wrong
     * answer that raising the ceiling from 8 fixed).
     *
     * The dispatch is emitted ONCE PER MODULE, out of line, and every gradient
     * site calls it. It used to be inlined at each site as a switch with one
     * fully expanded closure call per arity, whose per-arity cost grows with the
     * arity (n-ary arithmetic fold, unrolled variadic rest-list chain). With the
     * ceiling at 32 and six gradient sites in the standard library that came to
     * ~1.03M lines of IR — 2.7x the entire stdlib (585,967 -> 1,561,902 lines,
     * stdlib.bc 6.65MB -> 19.76MB). Platforms that emit stdlib definitions
     * `linkonce_odr` discard the unused ones early; Windows/COFF emits them
     * `weak_any` (see sexprGlobalLinkage), which cannot be discarded, so every
     * user compile ran opt and llc over all of it: measured LLVM work per
     * compile went 25.4s -> 117.7s and every GPU/XLA test hit the harness's 120s
     * compile budget.
     *
     * Out of line the arity dispatch happens once, and it is the closure
     * dispatcher's OWN runtime argument-count switch (ClosureSpreadArgs) rather
     * than a second, per-arity one layered on top. Raising the ceiling now costs
     * one more arm in one function instead of 24 fully expanded calls at every
     * gradient site.
     */
    static constexpr int GRAD_MAX_ARITY = 32;
    /* Sentinel the closure ABI uses for a variadic callable; it legitimately
     * wants the vectorized single-argument form. */
    static constexpr uint64_t GRAD_VARIADIC_ARITY = 255;

    Function* getOrCreateGradSpreadHelper();

    Value* codegenGradientSpreadCall(Value* closure_val, Value* point_vector,
                                     Value* dual_elems, Value* declared_arity);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* getTaggedValueType(Value* tagged_val);

    // Get base type from type tag, handling exactness flags correctly
    // For immediate types (0-7): mask with 0x0F to strip exactness flags
    // For types >= 8 (consolidated, multimedia, legacy): use directly
    Value* getBaseType(Value* type_tag);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* unpackInt64FromTaggedValue(Value* tagged_val);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* unpackDoubleFromTaggedValue(Value* tagged_val);

    // MIGRATED: Delegates to TaggedValueCodegen
    Value* unpackPtrFromTaggedValue(Value* tagged_val);

    // MIGRATION HELPER: Get subtype from object header for HEAP_PTR/CALLABLE values
    // Object header is 8 bytes before the data pointer:
    //   [subtype(1)][flags(1)][ref_count(2)][size(4)][data...]
    // Returns the subtype byte (0-255)
    Value* getObjectSubtype(Value* ptr_val);

    // MIGRATION HELPER: Check if a tagged value has a specific HEAP_PTR subtype
    Value* isHeapSubtype(Value* tagged_val, uint8_t expected_subtype);

    // MIGRATION HELPER: Check if a tagged value has a specific CALLABLE subtype
    Value* isCallableSubtype(Value* tagged_val, uint8_t expected_subtype);

    Value* extractCarAsTaggedValue(Value* cons_ptr_int);

    Value* extractCdrAsTaggedValue(Value* cons_ptr_int);

    // MIGRATED: Delegates to TaggedValueCodegen
    // Helper to safely extract i64 from possibly-tagged values for ICmp operations
    // CRITICAL: This prevents ICmp type mismatch assertions
    Value* safeExtractInt64(Value* val);

    /**
     * @brief R7RS 6.2.6 `floor-quotient`: floor(a/b), for any numeric
     *        representation.
     *
     * Derived from the floored remainder rather than re-implemented:
     * `(a - (modulo a b)) / b`. The numerator is an exact multiple of `b` by
     * construction, so exact operands stay exact (the division never produces a
     * rational) and fixnum / bignum / flonum operands all flow through the
     * polymorphic primitives. Also the sole place `floor/` gets its quotient,
     * so the two can never disagree.
     */
    Value* emitFloorQuotient(Value* a, Value* b);

    // Helper: Extract car element from cons cell as tagged value (type-safe approach)
    // This avoids ABI issues with returning 16-byte structs from C functions
    Value* extractConsCarAsTaggedValue(Value* cons_ptr);

    // Create a unit value for erased types (when we need a valid LLVM value but the type is erased)
    // Returns null tagged_value which is our "unit" type
    Value* createErasedPlaceholder();

    bool isImmediateIntegerHottType(eshkol::hott::TypeId type_id) const;

    struct LowLevelValueTypeInfo {
        eshkol::hott::TypeId hott_type;
        Type* llvm_type;
        bool is_signed_integer;
        bool is_pointer;
        bool is_null;
    };

    struct TargetIntrinsicCallInfo {
        LowLevelValueTypeInfo return_type;
        std::vector<LowLevelValueTypeInfo> arg_types;
        Function* declaration;
    };

    std::optional<LowLevelValueTypeInfo> resolveLowLevelTypeInfo(const eshkol_ast_t* type_ast,
                                                                 const char* builtin_name,
                                                                 const char* role,
                                                                 bool allow_null) const;

    std::optional<LowLevelValueTypeInfo> resolveMemoryAccessTypeInfo(
        const eshkol_ast_t* type_ast, const char* builtin_name) const;

    TypedValue makeLowLevelTypedValue(Value* raw_value, const LowLevelValueTypeInfo& type_info);

    Value* coerceValueToLowLevelScalar(const TypedValue& tv,
                                       const LowLevelValueTypeInfo& type_info,
                                       const char* builtin_name);

    std::optional<TargetIntrinsicCallInfo> resolveTargetIntrinsicCall(
        const eshkol_operations_t* op, const char* builtin_name);

    std::optional<AtomicOrdering> resolveFenceOrdering(const eshkol_ast_t* ordering_ast,
                                                       const char* builtin_name) const;

    std::optional<AtomicOrdering> resolveAtomicOrdering(const eshkol_ast_t* ordering_ast,
                                                        const char* builtin_name,
                                                        bool for_store) const;

    std::optional<AtomicOrdering> resolveAtomicRMWOrdering(const eshkol_ast_t* ordering_ast,
                                                           const char* builtin_name) const;

    std::optional<AtomicOrdering> resolveAtomicCmpXchgFailureOrdering(
        const eshkol_ast_t* ordering_ast,
        const char* builtin_name) const;

    bool isAtomicCmpXchgFailureOrderingAllowed(AtomicOrdering success_ordering,
                                               AtomicOrdering failure_ordering) const;

    Align lowLevelABIAlignment(Type* type) const;

    eshkol::hott::TypeId resolveDeclaredHottTypeId(const hott_type_expr_t* type_expr) const;

    uint8_t closureReturnCategoryForHottType(eshkol::hott::TypeId hott_type) const;

    // Robust helper to convert tagged_value to TypedValue with proper runtime type detection
    // This preserves type information through PHI nodes
    TypedValue detectValueType(Value* llvm_val);
    // Convert TypedValue to tagged_value (AST→IR boundary crossing)
    Value* typedValueToTaggedValue(const TypedValue& tv);

    // Simple helper to wrap tagged_value in TypedValue (for cons cell creation)
    // This avoids complex control flow by just storing the tagged_value as-is
    // ===== POLYMORPHIC ARITHMETIC FUNCTIONS (Phase 1.3 + Phase 2 Dual Number Support) =====
    // These operate on tagged_value parameters and handle mixed types + dual numbers

    // MIGRATED: Polymorphic addition - delegates to ArithmeticCodegen
    Value* polymorphicAdd(Value* left_tagged, Value* right_tagged);

    // MIGRATED: Polymorphic subtraction - delegates to ArithmeticCodegen
    Value* polymorphicSub(Value* left_tagged, Value* right_tagged);

    // MIGRATED: Polymorphic multiplication - delegates to ArithmeticCodegen
    Value* polymorphicMul(Value* left_tagged, Value* right_tagged);

    // MIGRATED: Polymorphic division - delegates to ArithmeticCodegen
    Value* polymorphicDiv(Value* left_tagged, Value* right_tagged);

    // MIGRATED: Polymorphic comparison - delegates to ArithmeticCodegen
    Value* polymorphicCompare(Value* left_tagged, Value* right_tagged,
                             const std::string& operation);



    // ===== POLYMORPHIC FUNCTION WRAPPERS (Phase 2.4) =====
    // Create Function* objects that wrap polymorphic arithmetic for use in higher-order functions

    Function* polymorphicAdd();

    Function* polymorphicSub();

    Function* polymorphicMul();

    Function* polymorphicDiv();


    // Track current source location for error reporting
    uint32_t current_source_line = 0;
    uint32_t current_source_column = 0;

    Value* languageCoverageString(const std::string& value);

    void emitLanguageCoverage(const eshkol_ast_t* ast);

    Value* codegenAST(const eshkol_ast_t* ast);

    ContinuationTask<Value*> codegenASTTask(const eshkol_ast_t* ast);

    // MIGRATED: Delegates to StringIOCodegen
    Value* codegenString(const char* str);

    // OWNERSHIP: Eshkol uses arena allocation with deterministic lifetimes.
    // Values are valid for the lifetime of their arena region. No move
    // semantics — the arena model guarantees validity within scope.
    // This function is a no-op passthrough.
    Value* emitUseAfterMoveCheck(Value* loaded_val, const std::string& var_name);

    // OWNERSHIP ENFORCEMENT: Check if a value has the BORROWED flag set
    // Used by codegenSet to prevent mutation of borrowed references
    Value* emitBorrowMutationCheck(Value* var_val, const std::string& var_name);

    Value* codegenVariable(const eshkol_ast_t* ast);

    // Create a first-class wrapper for `list` / `values` — variadic builtins
    // that the codegen normally inlines (coll_->list / codegenValues). When
    // the user references the bare name `list` as a value (e.g. as a
    // call-with-values consumer or a map argument), no inline expansion
    // applies, so we have to synthesize an honest runtime function with the
    // variadic-closure ABI. The wrapper takes a single tagged_value rest
    // parameter (the list the closure dispatcher builds from the caller's
    // args) and just returns it. For `list` that's the whole semantic;
    // `values` similarly returns its args as a list in this path (used as a
    // first-class function; the multi-value return path goes through a
    // different code path).
    //
    // The closure metadata uses CLOSURE_FLAG_VARIADIC (bit 0 of flags byte at
    // offset 34), arity=0, which signals codegenClosureCall to cons all
    // caller args into a list and pass that as the single argument. See the
    // "Variadic closure" branch around line ~5440.
    Function* getOrCreateVariadicIdentityWrapper(const std::string& wrapper_name);

    Value* makeVariadicIdentityClosureValue(const std::string& wrapper_name);

    /* Wrap a `(rest_list) -> tagged_value` function as a VARIADIC closure
     * value: arity 0 with the variadic bit set, which is what makes the
     * closure dispatcher cons the caller's arguments into the single rest
     * list the wrapper expects. */
    Value* makeVariadicClosureValueFor(Function* wrapper);

    Function* getOrCreateBytevectorRestWrapper();


    /* SW-230: a builtin that accepts a small RANGE of argument counts (`atan`
     * of 1 or 2, `round` with an optional precision) as a value. The closure
     * is variadic and dispatches on the length of its rest list to the
     * wrapper built at that arity, each generated from the call-position
     * lowering; any other count raises an arity error. A fixed-arity value
     * silently dropped the second argument: `((lambda (f) (f 1.0 2.0)) atan)`
     * answered atan(1). */
    Value* makeByCountBuiltinClosureValue(const std::string& name,
                                          const InlineBuiltinSpec& spec);

    /* SW-173: materialise a VARIADIC builtin as a genuine variadic closure.
     *
     * Returns nullptr when the row declares no rest form, so the caller
     * falls back to the row's fixed-arity materialisation — the pre-existing
     * compromise, now reached only for rows that say so. */
    Value* makeVariadicBuiltinClosureValue(const std::string& name,
                                           const InlineBuiltinSpec& spec);

    /* `(f a b c …)` == `(f (f (f a b) c) …)` for the associative variadic
     * builtins. The BINARY form is the authoritative call-position lowering,
     * so the fold cannot disagree with `f` on any pair; the only thing
     * asserted here is associativity, and it is asserted only for the rows
     * that declare LeftFold. There is no argument-count ceiling — unlike a
     * fixed-arity wrapper, or a switch over unrolled arities. */
    void emitVariadicBuiltinLeftFold(Function* wrap_fn, const std::string& name,
                                     Value* rest, Function* binary_fn,
                                     const std::function<Value*(Value*)>& unary = nullptr,
                                     const std::function<Value*()>& nullary = nullptr);

    /* `(f a b c …)` == `(and (f a b) (f b c) …)` for the R7RS comparison
     * chains. Fewer than two arguments is an arity error. */
    void emitVariadicBuiltinChain(Function* wrap_fn, const std::string& name,
                                  Value* rest, Function* binary_fn);

    /* SW-241: the arithmetic operators and numeric comparisons as VALUES are
     * the variadic R7RS procedures, not their 2-argument lowering. Built from
     * the same polymorphic operations the call position uses. */
    Value* makeVariadicOperatorClosureValue(const std::string& op);

    void raiseVariadicBuiltinZeroArgError(const std::string& name);

    void raiseBuiltinArityError(const std::string& message);

    /* User-shadowable builtin OPs (audit Bug G).
     *
     * The parser maps a handful of builtin names to dedicated OP
     * tags (e.g. ESHKOL_MAKE_WORKSPACE_OP) so the codegen can emit
     * a direct C-runtime call without a string-match dispatch step.
     * That was a premature binding decision: R7RS §5.3.1 allows any
     * top-level identifier to be redefined, and Noesis (among others)
     * legitimately defines `make-workspace`, `make-kb`, `make-fact`,
     * etc. at user scope. With the parser mapping in place those
     * defines were silently ignored at every call site.
     *
     * The architecturally correct place to resolve this is codegen,
     * where user scope is available (`function_table`). For every OP
     * that corresponds to a user-overridable builtin, check whether
     * the user has a same-named define and — if so — route the call
     * through codegenCall with a synthesised CALL_OP/VAR header.
     * codegenCall owns the full call-dispatch pipeline (TCO,
     * closures, arity checks, variadic packing); synthesising the
     * header is the minimal vehicle to reuse it without duplicating
     * 150 lines. */
    static const std::unordered_map<eshkol_op_t, const char*>& userShadowableOps();

    /* Does `name` resolve to a user-defined binding in a scope that
     * is unambiguously visible from the current call site?
     *
     * Intended solely for the userShadowableOps redirect in
     * codegenOperation — not a general-purpose shadow check. The
     * callers are the consciousness-engine / workspace OP tags
     * (make-workspace, make-kb, make-fact, kb-*, fg-*, ws-*, …),
     * none of which collide with the C math stubs registered in
     * function_table during init (sin, cos, exp, sqrt, pow, exit,
     * printf).
     *
     * Intentionally narrower than codegenCall's cascade: we only
     * look at function_table (canonical top-level defines) and the
     * scoped `<func>.<name>_func` key (true inner defines from the
     * currently-enclosing function). The UNSCOPED `_func` key in
     * symbol_table / global_symbol_table is NOT consulted here
     * because inner-function codegen has historically registered
     * nested defines there (llvm_codegen.cpp:8517 and friends) —
     * so an unscoped hit can mean "an inner define from a sibling
     * function that happens to have compiled first", which from the
     * current call site is NOT actually in scope. A top-level
     * `(define k1 (make-kb))` after a `(define (other) (define
     * (make-kb) ...))` would otherwise divert to codegenCall and
     * find the wrong (inner, non-visible) lambda via the leaked
     * unscoped key. Fix: only accept unambiguous scope matches. */
    bool hasUserShadow(const std::string& name);

    Value* codegenOperation(const eshkol_operations_t* op);

    ContinuationTask<Value*> codegenOperationTask(const eshkol_operations_t* op);

    Value* codegenDefine(const eshkol_ast_t* ast);

    Value* codegenDefine(const eshkol_operations_t* op);

    Value* codegenFunctionDefinition(const eshkol_ast_t* ast);

    bool isLocalCaptureStorage(Value* value) const;

    bool shouldDropReplTopLevelCapture(const std::string& name) const;

    void filterReplTopLevelCaptures(std::vector<std::string>& free_vars);

    // Generate a nested function definition as a closure (like a lambda)
    Value* codegenNestedFunctionDefinition(const eshkol_operations_t* op);


    // set! - mutate an existing variable
    Value* codegenSet(const eshkol_operations_t* op);

    // ─────────────────────────────────────────────────────────────────────
    // codegenCall pre-dispatch helpers (mechanical extraction; IR-identical)
    //
    // These three handlers cover call expressions whose head is itself an
    // operation rather than a variable reference:
    //   - ((lambda (x) ...) arg)        → codegenCallInlineLambda
    //   - ((f x) y)                     → codegenCallResultAsFunc
    //   - ((derivative f) x), etc.      → codegenCallOperationResultAsFunc
    // Each evaluates the head into a closure and dispatches via
    // codegenClosureCall after packing args to tagged_value.
    // ─────────────────────────────────────────────────────────────────────

    Value* codegenCallInlineLambda(const eshkol_operations_t* op);

    Value* codegenCallResultAsFunc(const eshkol_operations_t* op);

    // Plain-language name for a non-procedure literal in operator position,
    // used by the call diagnostic in codegenCallTask.
    static const char* describeLiteralOperator(const eshkol_ast_t* head);

    Value* codegenCallOperationResultAsFunc(const eshkol_operations_t* op);

    Value* codegenFormatBuiltin(const eshkol_operations_t* op);

    // ESH-0362 (JIT/REPL half) ────────────────────────────────────────────────
    //
    // Authoritative parameter count of a REPL-registered callee, or -1 when it
    // cannot be established and no arity conclusion may be drawn.
    //
    // The two REPL slot-call paths (a `__repl_fwd_<name>` indirect call through
    // a JIT-resolved function pointer) SYNTHESISE the callee's FunctionType from
    // the CALL's argument count. An arity mismatch there is therefore not even a
    // mismatch — it is a silent ABI disagreement: the callee reads its missing
    // parameter out of whatever the register happened to hold. Under `-r` this
    // is the path a `(require …)`d module's functions are called through, so
    // `(process-spawn-argv argv)` against the two-parameter definition handed
    // the C spawn shim an uninitialised `cwd`, with no diagnostic at all. (The
    // AOT/direct-call arity check does report it, which is why the same file
    // compiled with `-o` names the error and this one did not.)
    //
    // The registry's arity is F.arg_size() at registration time, i.e. the ABI
    // parameter count — it counts capture slots and the variadic rest slot too.
    // So it is only usable as a user-visible arity when the callee has neither,
    // and every other case returns -1 rather than risk rejecting a legal call.
    int64_t replEnforceableArity(const std::string& func_name);

    // Report a REPL slot-call arity mismatch with the same wording the
    // AOT/direct-call path uses, and fail the compilation. Returns true when a
    // mismatch was found (caller must abort codegen of the call).
    bool replSlotArityMismatch(const std::string& func_name, uint64_t num_call_args);

    // FFI POINTER-ARG GUARD (ESH-0363) ────────────────────────────────────────
    //
    // An `extern` parameter declared `ptr` / `string` / `char*` is passed to C
    // by unpacking the tagged value's 64-bit payload and IntToPtr'ing it. That
    // conversion is unconditional, so a NUMBER lands in the callee as an
    // address: `(run-argv-capture argv 5000)` — 5000 mistaken for the positional
    // `cwd` string — reached execvp's C shim as `const char* 0x1388` and died
    // with SIGSEGV at address 0x1388. No diagnostic, no exit code, just a fault
    // at a numerically suspicious address, and only because the value happened
    // to be small; a large fixnum can hit mapped memory and corrupt instead.
    //
    // This emits a branch on the argument's tagged TYPE BYTE that rejects the
    // values which provably cannot be an address, and raises a catchable type
    // error naming the extern, the argument position, and the declared type.
    //
    // The predicate is a DENYLIST of immediate tags, deliberately not an
    // allowlist of pointer tags. Eshkol's type byte is a crowded encoding: the
    // multimedia tags (HANDLE/BUFFER/STREAM/EVENT = 16..19), the deprecated
    // pointer aliases (CONS_PTR/STRING_PTR/… = 32..40) and the port flag bits
    // OR'd onto HEAP_PTR all denote real addresses, and some codegen paths fold
    // the exact/inexact flag into the type byte (see i128_runtime.cpp's
    // TYPE_FLAG_MASK note). An allowlist would have to enumerate all of those
    // correctly or it would reject a legitimate pointer — a guard that breaks
    // working programs. Matching only the seven unflagged immediate tags means
    // the worst case is a MISSED catch on an exotic encoding, never a false
    // rejection.
    //
    // `#f` is exempt: it is how Eshkol spells a NULL pointer argument at this
    // boundary (`(process-spawn-raw command cwd #f 0)` passes a NULL envp,
    // `(process-read-all-stdout-raw proc n #f)` a NULL out-param), so BOOL is
    // rejected only when its payload is non-zero, i.e. `#t`.
    void emitFfiPointerArgGuard(Value* tagged_arg,
                                const std::string& extern_name,
                                const std::string& real_symbol,
                                uint64_t param_index,
                                const std::string& declared_type);

    /**
     * @brief Hoist a tagged-value alloca (optionally an array) to the current
     *        function's entry block.
     *
     * The region-handle builtins are designed to be called once per iteration of
     * a long-running loop, and a TCO'd named-let/do loop is an in-function branch
     * back to a loop header — an alloca in the body therefore re-adjusts the
     * stack pointer on every pass and is only reclaimed when the *function*
     * returns. That is the ESH-0214 leak that presented as a spurious "stack
     * overflow"; `with-region` fixes it the same way for its own slots.
     */
    AllocaInst* entryTaggedAlloca(uint64_t count, const char* name);

    /**
     * @brief Lower `(region-open …)`, `(region-close …)` and `(region-open? …)` —
     *        the user-reachable, non-lexical region-handle surface (#341).
     *
     * All three are thin shims over the shared C entry points in
     * runtime_regions.cpp, which the bytecode VM calls too, so the handle
     * protocol and every error message are identical across substrates by
     * construction rather than by convention.
     *
     * `region-close` passes its keep-list as one contiguous array of tagged
     * values: eshkol_region_unwind_to promotes that array IN PLACE, level by
     * level, through the same escape evacuator `with-region` uses for its result,
     * so a kept value's whole reachable subgraph (interior pointers included) is
     * deep-promoted out before the region arena is freed.
     */
    Value* codegenRegionHandleBuiltin(const eshkol_operations_t* op,
                                      const std::string& func_name);

    /**
     * @brief Emit the uniform out-of-range guard shared by every indexed
     *        accessor: when @p bad is true, raise a *catchable*
     *        ESHKOL_EXCEPTION_ERROR carrying @p msg; otherwise fall through
     *        with the builder positioned in the in-range continuation.
     *
     * R7RS 6.9 makes an out-of-range index "an error", and the house parity
     * contract requires every substrate (AOT/JIT codegen, the C runtime
     * helpers in lib/core/runtime_bytevector.cpp, and the bytecode VM) to
     * signal it identically — same catchable condition, same message text —
     * so a `guard` around the access behaves byte-for-byte the same
     * everywhere. This mirrors the pre-existing vector-ref / vector-set! /
     * string-ref / tensor-ref guards (see collection_codegen.cpp) so the
     * whole accessor family shares one contract.
     */
    void emitBoundsCheckRaise(Value* bad, const char* msg);

    // Declared type keyword of parameter `param_index` of an `extern`, or an
    // empty string when `callee_name` is not an extern (or the position is
    // beyond its declared parameters — a varargs tail). Tries the Eshkol name
    // first, then the real C symbol, because call sites reach externs by either.
    std::string externDeclaredParamType(const std::string& eshkol_name,
                                        const std::string& real_symbol,
                                        uint64_t param_index) const;

    static bool externTypeIsPointerLike(const std::string& declared);

    // Bytevector constructors accept only exact integers in [0,255].  Keep
    // the original tagged representation through the exactness/range checks;
    // conversion to a byte happens only in the success block.  In particular,
    // bignums are compared by the existing exact runtime comparator.
    Value* emitCheckedByte(Value* value, const std::string& who);

    Value* codegenCall(const eshkol_operations_t* op);

    ContinuationTask<Value*> codegenCallTask(const eshkol_operations_t* op);

    // HoTT-optimized binary arithmetic: when both types are known, skip runtime dispatch
    // Returns nullptr if optimization not possible (fall back to polymorphic path)
    // ── FREESTANDING MACHINE-INTEGER ARITHMETIC ─────────────────────────────
    //
    // The numeric tower (bignums, rationals, exact/inexact dispatch, AD
    // carriers) is a hosted runtime facility.  A freestanding object has no
    // runtime to call, so in a freestanding profile the integer operators
    // lower to native modular machine arithmetic instead.  One rule covers
    // every operator below:
    //
    //   * The operation width and signedness come from the operands that
    //     carry a declared machine-integer type (u8..u64, i8..i64, usize,
    //     isize: annotated parameters, low-level memory/atomic/intrinsic
    //     results, and the results of these operators).  The widest such
    //     type wins; on a width tie an unsigned type wins (C's usual
    //     arithmetic conversions).  usize/isize are pointer-sized for the
    //     module's data layout.
    //   * Integer literals and values without a declared type adopt that
    //     machine type; with no declared operand the operation is i64.
    //   * Results wrap modulo 2^width; division and remainder by zero trap
    //     (llvm.trap).  Comparisons use unsigned predicates for unsigned types.
    //
    // Non-integer operands (reals, strings, pairs) have no freestanding
    // meaning and are rejected at compile time.
    static const char* freestandingMachineIntegerOp(const std::string& name);

    struct MachineIntegerShape {
        eshkol::hott::TypeId type;
        unsigned bits;
        bool is_signed;
    };

    std::optional<MachineIntegerShape> machineIntegerShape(eshkol::hott::TypeId t) const;

    // Integer payload of an operand, or nullptr when the operand is not an
    // integer value in the freestanding sense.
    Value* machineIntegerPayload(const TypedValue& tv);

    void emitTrapUnless(Value* ok, const char* label);

    TypedValue emitMachineIntegerOp(const char* operation, const char* source_name,
                                    const std::vector<TypedValue>& operands,
                                    const std::vector<bool>& is_literal);

    ContinuationTask<TypedValue> codegenMachineIntegerOpTask(const eshkol_operations_t* op,
                                                             const char* operation,
                                                             const char* source_name);

    // Freestanding builtin integer operator call that the program has not
    // shadowed with its own definition.
    const char* freestandingMachineIntegerCall(const std::string& func_name);

    Value* hottOptimizedBinaryArith(const TypedValue& left, const TypedValue& right,
                                     const std::string& operation);

    Value* codegenArithmetic(const eshkol_operations_t* op, const std::string& operation);

    ContinuationTask<Value*> codegenArithmeticTask(const eshkol_operations_t* op, const std::string& operation);

    Value* codegenComparison(const eshkol_operations_t* op, const std::string& operation);

    // HoTT type-directed comparison optimization
    // Returns optimized result if both types are known, nullptr to fall back to polymorphic
    Value* hottOptimizedComparison(const TypedValue& left, const TypedValue& right,
                                    const std::string& operation);


    /** The spelling the PROGRAMMER wrote for this call, for diagnostics.
     *
     *  Lowering names and public names are not the same word: `(ceiling)` is
     *  lowered by codegenMathFunction(op, "ceil") and `(truncate)` by
     *  codegenMathFunction(op, "trunc"). Reporting the lowering name told the
     *  reader about an intrinsic they never wrote. Falls back to the lowering
     *  name for a synthesized call with no variable callee. */
    static std::string publicCalleeName(const eshkol_operations_t* op,
                                        const std::string& lowering_name);

    // stop-gradient (ADR-0049). One runtime entry point,
    // eshkol_stop_gradient_tagged, reads the primal of any carrier -- forward
    // jet, Taylor tower or level, reverse-mode node -- exactly as the
    // comparisons do, so `(stop-gradient x)` is a constant to every enclosing
    // `derivative`/`gradient`/`derivative-n` pass while its value equals x.
    // Non-carriers pass through unchanged. This is the primitive behind the
    // straight-through estimators: soft + (stop-gradient (- hard soft)) has
    // the hard value and the soft derivative.
    Value* codegenStopGradient(const eshkol_operations_t* op);

    // Directed rounding: fl-next-up / fl-next-down. Unary Scheme surface
    // (the direction is fixed by which name was called), lowered to the
    // libm binary `nextafter(x, direction)` with `direction` a compile-time
    // +INFINITY/-INFINITY constant -- nextafter saturates at the argument's
    // own value once x has no representable neighbour in that direction
    // (e.g. nextafter(+inf, +inf) == +inf, nextafter(DBL_MAX, +inf) == +inf,
    // per C99 7.12.11.3), so no separate overflow handling is needed here.
    // Deliberately real-scalar-only: no complex/tensor/dual/AD-node/tower
    // fast path (see call-site comment) -- an operand in any of those
    // representations reads as a tagged DOUBLE by extractDoubleFromTagged,
    // which is correct only for the plain-real case, so this must never be
    // reached for those tags. Guard here rather than trust every call site.
    Value* codegenNextafter(const eshkol_operations_t* op, bool up);

    /**
     * @brief The libm function backing a scalar math builtin, guaranteed
     *        non-null and of the expected `double(double[, double])` type.
     *
     * `function_table` is an `unordered_map<string, Function*>`, so reading it
     * with `operator[]` on a MISSING key silently inserts a NULL Function* and
     * hands that straight to IRBuilder::CreateCall — a null dereference inside
     * CallInst::Create, i.e. the compiler dies with SIGSEGV instead of
     * producing a diagnostic. That is exactly how
     *
     *     (display (tensor-ref (elu t 2) 0))  ; puts a bare @exp in the module
     *     (display (exp -2.0))                ; SIGSEGV
     *
     * crashed: the activation lowering materialised `exp` by bare name,
     * AutodiffCodegen::getMathFunc then found that module symbol and returned
     * it WITHOUT recording it in function_table, and the scalar path's
     * `function_table["exp"]` inserted null. `exp` was the only exposed name
     * because it was the one libm entry BuiltinFactoryCodegen never
     * pre-declared; every other scalar math name already had a row.
     *
     * Routing every scalar-math lookup through here removes the whole class:
     * a missing or wrongly-typed row is repaired from libm_codegen (intrinsic
     * first, so it can never bind to an unrelated symbol of the same name)
     * rather than crashed on.
     */
    Function* mathFunc(const std::string& name);

    Value* codegenMathFunction(const eshkol_operations_t* op, const std::string& func_name);

    // The value-level core of the procedure above: everything after its
    // arguments are evaluated. Complex component formulas (ADR-0025) call it
    // on a component, so a derivative carrier gets the same rule it gets here.
    // `operand_is_real` is set by the complex component formulas, whose operands
    // are components and so never complex: the complex-operand branch is not
    // emitted for them. It must not be, because that branch emits those same
    // formulas, and the emission would not terminate.
    Value* codegenMathFunctionOnTagged(Value* arg_tagged, const std::string& func_name,
                                       bool operand_is_real = false);

    Value* codegenActivationFunction(const eshkol_operations_t* op, bool relu);

    // ===== ML activation family: one dispatch point =====
    //
    // Every activation takes a number or a tensor and returns the same kind
    // (docs/tutorials/01_AUTODIFF_AND_ML.md, docs/API_REFERENCE.md). This table
    // and codegenActivationFamily() are the only place an activation name is
    // lowered. The operand is evaluated ONCE, bound to a generated name, and
    // classified at run time:
    //   - a tensor, or a reverse-mode node that carries a tensor, goes to the
    //     TensorCodegen lowering, which owns tensor layout and tensor AD;
    //   - a scalar number or scalar differentiation carrier (fixnum, flonum,
    //     bignum, rational, dual, Taylor tower, scalar AD node) goes to the
    //     activation's scalar definition. That definition is a composition of
    //     the scalar primitives (+ - * / exp log tanh min max and `>` on the
    //     primal), each of which already carries every differentiation mode,
    //     so the activation is differentiable without a rule of its own. It is
    //     the same formula, constants and threshold as the tensor kernel;
    //   - anything else goes to the tensor lowering, whose operand check
    //     coerces a numeric collection or raises a catchable type error.
    // relu and sigmoid enter through the same table but keep their dedicated
    // scalar lowering (codegenActivationFunction), which records dedicated AD
    // node and Taylor operations.
    static eshkol_ast_t* actCall(const char* fn, std::initializer_list<eshkol_ast_t*> args);
    static eshkol_ast_t* actNum(double v);

    // An activation's definition as an AST over the operand x and the optional
    // parameter p, and the TensorCodegen kernel it pairs with.
    using ActivationFormula = eshkol_ast_t* (*)(eshkol_ast_t* x, eshkol_ast_t* p);
    using ActivationKernel = Value* (eshkol::TensorCodegen::*)(const eshkol_operations_t*);
    struct ActivationSpec {
        const char* name;
        uint64_t min_args;
        uint64_t max_args;
        double param_default;  // value of the optional second argument
        ActivationKernel tensor_fn;
        // Scalar definition; null for the activations with a dedicated scalar
        // lowering (relu, sigmoid).
        ActivationFormula scalar;
        // Tensor definition used when the optional parameter is supplied and
        // the tensor kernel has no parameter of its own (swish's beta).
        ActivationFormula tensor_with_param;
    };

    static const ActivationSpec* findActivation(const std::string& name);

    // i1: the operand belongs to the scalar definition (a number or a scalar
    // differentiation carrier). Everything else -- a tensor, a reverse-mode node
    // that carries a tensor, or a non-number -- belongs to the tensor lowering.
    Value* emitActivationOperandIsScalar(Value* arg);

    Value* codegenActivationFamily(const eshkol_operations_t* op, const ActivationSpec& spec);

    // One element loader for the multi-sequence iterators (vector-map,
    // vector-for-each): the i-th element of a source as a tagged value. A
    // Scheme vector slot is the value; a tensor slot is a double bit pattern,
    // unless the carrier was promoted by a non-numeric store or is a dual
    // tensor (ADR-0020), where the slot is already a tagged value.
    Value* loadSequenceElement(Value* is_tensor, Value* tagged_slots,
                               Value* elems, Value* base, Value* idx);

    // Polymorphic abs - handles AD/dual, then delegates to ArithmeticCodegen::abs
    // for numeric types (int64, double, bignum)
    Value* codegenAbs(const eshkol_operations_t* op);

    Value* codegenRound(const eshkol_operations_t* op);

    // Binary math function codegen (for atan2, fmod, fmin, fmax, remainder, etc.)
    Value* codegenBinaryMathFunction(const eshkol_operations_t* op, const std::string& func_name);

    // The value-level core of the procedure above: everything after its
    // arguments are evaluated. Complex component formulas (ADR-0025) call it
    // on a component, so a derivative carrier gets the same rule it gets here.
    Value* codegenBinaryMathFunctionOnTagged(Value* arg1, Value* arg2, const std::string& func_name);

    // atan2 of operands that are neither Taylor carriers nor forward jets: a
    // reverse-tape node (recorded on the tape) or plain numbers.
    Value* codegenAtan2NonTaylor(Value* arg1, Value* arg2);

    // Modulo operation - Scheme semantics (result has same sign as divisor)
    Value* codegenModulo(const eshkol_operations_t* op);

    // Remainder operation - handles both integer and floating point
    // Uses truncated division semantics (sign of result matches dividend)
    Value* codegenRemainder(const eshkol_operations_t* op);

    // Integer quotient (truncated division)
    Value* codegenQuotient(const eshkol_operations_t* op);

    // GCD (Greatest Common Divisor) using Euclidean algorithm
    // Helper: convert a typed value to absolute int64 for GCD/LCM
    Value* toAbsInt64(Value* val);

    // Helper: emit inline Euclidean GCD loop for two int64 values
    // Returns gcd(a, b) as raw int64. Creates fresh basic blocks.
    Value* emitGCDPair(Value* a, Value* b);

    // R7RS §6.2.6: Variadic GCD via fold of Euclidean algorithm
    // (gcd) → 0, (gcd n) → |n|, (gcd a b ...) → gcd(gcd(a,b), ...)
    Value* codegenGCD(const eshkol_operations_t* op);

    // Helper: emit inline LCM for two absolute int64 values
    // lcm(a, b) = a * (b / gcd(a, b)), with zero short-circuit
    // Returns raw int64. Creates fresh basic blocks.
    Value* emitLCMPair(Value* abs_a, Value* abs_b);

    // R7RS §6.2.6: Variadic LCM via fold
    // (lcm) → 1, (lcm n) → |n|, (lcm a b ...) → lcm(lcm(a,b), ...)
    Value* codegenLCM(const eshkol_operations_t* op);

    // MIGRATED: Delegates to ArithmeticCodegen
    // Helper to convert any value to double
    Value* toDouble(Value* val);

    // Min/Max - variadic, handles mixed types
    Value* codegenMinMax(const eshkol_operations_t* op, bool is_min);

    // R7RS exactness probe: a value is inexact iff it is a flonum (DOUBLE) or
    // a complex number (always inexact in Eshkol). int64 / bignum / rational
    // are exact. Returns an i1.
    Value* isInexactTagged(Value* tagged);

    // Conditionally coerce an EXACT tagged value to its inexact (double)
    // representation. When `cond` is false the value is returned unchanged;
    // when true an exact int64/bignum/rational is converted via extractAsDouble
    // (the single exact->inexact path).
    //
    // Crucially the coercion only fires for *genuinely exact* results
    // (INT64 / HEAP_PTR bignum / rational). A DUAL number (forward-mode AD)
    // is already an inexact float carrying a tangent — routing it through
    // extractAsDouble would strip the derivative and break AD through min/max.
    // DOUBLE and COMPLEX are already inexact, so leaving them untouched is
    // both correct and avoids needless work.
    Value* coerceToInexactIf(Value* result, Value* cond);

    // MIGRATED: Delegates to ArithmeticCodegen
    // Helper to extract double from tagged value (handles both int and double)
    Value* extractDoubleFromTagged(Value* tagged);

    // MIGRATED: Delegates to ControlFlowCodegen
    // Helper to check if a tagged value is "truthy" (non-false, non-null, non-zero)
    Value* isTruthy(Value* val);
    // MIGRATED: Short-circuit AND - delegates to ControlFlowCodegen
    Value* codegenAnd(const eshkol_operations_t* op);

    // MIGRATED: Short-circuit OR - delegates to ControlFlowCodegen
    Value* codegenOr(const eshkol_operations_t* op);

    // MIGRATED: Cond expression - delegates to ControlFlowCodegen
    Value* codegenCond(const eshkol_operations_t* op);

    // SW-229: the forward-mode AD dynamic state a non-local exit must put back.
    //
    // A derivative pass raises __ad_pert_level (and a Taylor pass
    // __ad_tower_active/__ad_tower_order) on entry and lowers it on its normal
    // exit. A raise or an escaping continuation from inside the pass longjmps
    // past that exit, so the level stayed raised: every later `derivative`
    // believed it was nested, declined the exact tier and answered inexact
    // (exact 6 became 6.0), and a later derivative through exact division by
    // exact 0 answered -inf.0 instead of raising. The runtime's unwinder
    // (eshkol_ad_state_restore) cannot reach these: they are module globals,
    // internal to an AOT module. So every site that installs a longjmp target
    // snapshots them into entry-block slots before setjmp and stores them back
    // first thing on the path a longjmp arrives by.
    using AdForwardSnapshot = std::vector<std::pair<GlobalVariable*, Value*>>;
    AdForwardSnapshot snapshotAdForwardState();
    // Emit the restores at the start of `landing`, the block a longjmp reaches.
    void restoreAdForwardStateAt(BasicBlock* landing, const AdForwardSnapshot& snap);

    // The recursion-depth counter is dynamic state a non-local exit must put
    // back too: the guarded returns a longjmp skips never decrement it, so a
    // loop that keeps leaving deep code through raise or a continuation would
    // reach ESHKOL_MAX_STACK without ever being deep. Same discipline as the
    // AD state above: an entry-block slot written before setjmp, read first
    // thing on the landing path.
    // An alloca for one tagged value in the entry block of the current function.
    Value* createEntryBlockTaggedSlot(const char* name);

    Value* snapshotRecursionDepth();
    void restoreRecursionDepthAt(BasicBlock* landing, Value* slot);

    // Exception handling: guard expression
    // Syntax: (guard (var clause ...) body ...)
    // Sets up setjmp handler, evaluates body, handles exceptions via clauses
    Value* codegenGuard(const eshkol_operations_t* op);

    // Exception handling: raise expression
    // Syntax: (raise exception)
    // Simplified: Always create a new exception from the given value
    Value* codegenRaise(const eshkol_operations_t* op);

    // ===== CALL/CC — First-class continuations =====
    // Syntax: (call/cc proc) or (call-with-current-continuation proc)
    // Uses setjmp/longjmp: setjmp captures the return point, longjmp invokes the continuation
    // True when this call/cc's continuation provably cannot outlive the
    // capturing frame, so no stack image is needed. Requires proc to be a
    // literal 1-parameter lambda whose body only ever calls the parameter;
    // any other shape (a named procedure, a variadic lambda, a stored or
    // returned reference) is treated as escaping.
    bool callCCContinuationStaysLocal(const eshkol_operations_t* op);

    bool astHasEscapingCallCC(const eshkol_ast_t* ast);

    Value* codegenCallCC(const eshkol_operations_t* op);

    // ===== DYNAMIC-WIND =====
    // Syntax: (dynamic-wind before thunk after)
    // Calls before(), then thunk(), then after(), returns thunk's result
    // before/after are also called during continuation jumps across dynamic-wind boundaries
    // Borrow scopes close their port on every exit: a native cleanup frame on
    // the dynamic-wind stack closes it when a raise or a continuation escape
    // unwinds the scope, and normal return pops the frame and closes it here.
    void pushPortCleanupWind(Value* close_fp);

    void popPortCleanupWindAndClose(Value* close_fp);

    Value* codegenDynamicWind(const eshkol_operations_t* op);

    // ===== WITH-EXCEPTION-HANDLER (R7RS) =====
    // Syntax: (with-exception-handler handler thunk)
    //   handler: a 1-arg procedure called with the raised value on exception
    //   thunk: a 0-arg procedure whose body is protected
    // Uses setjmp/longjmp, same pattern as codegenGuard
    Value* codegenWithExceptionHandler(const eshkol_operations_t* op);

    // ===== RAISE-CONTINUABLE (R7RS 6.11) =====
    // (raise-continuable obj): call the current with-exception-handler's
    // procedure with obj in the dynamic environment of the raise, the outer
    // handlers installed, and return its value. With a guard (or no handler)
    // as the current handler, the runtime raises obj as `raise` does.
    Value* codegenRaiseContinuable(const eshkol_operations_t* op);

    // ===== MULTIPLE RETURN VALUES OPERATIONS =====

    // Emit a diagnosed runtime error from a multiple-values dispatch path.
    // The caller must have positioned the builder in a dedicated failure block;
    // this helper terminates that block with `unreachable`.
    void emitMultipleValuesRaise(const char* message);

    // (values expr1 expr2 ...) - Return multiple values
    // Creates a multi-value object that packages multiple values together
    Value* codegenValues(const eshkol_operations_t* op);

    // (call-with-values producer consumer) - Apply consumer to producer's values
    // producer: thunk that returns multiple values
    // consumer: function that accepts those values
    Value* codegenCallWithValues(const eshkol_operations_t* op);

    // Helper: Call consumer with unpacked multi-value
    // Uses runtime dispatch to handle the closure call system's supported
    // dynamic arities (0..16).  Counts above the documented closure dispatch
    // ceiling are rejected explicitly rather than silently calling the
    // consumer with zero arguments.
    Value* callConsumerWithMultiValue(Value* consumer, Value* multi_val);

    // Helper: Call closure with dynamic argument count
    // Uses codegenClosureCall which handles all closure calling conventions
    Value* callClosureWithArgs(Value* closure, const std::vector<Value*>& args, Value* actual_count);

    // Helper: Call consumer with single value
    Value* callConsumerWithSingleValue(Value* consumer, Value* val);

    // (let-values (((var1 var2) producer1) ...) body) - Bind multiple values
    Value* codegenLetValues(const eshkol_operations_t* op);

    // ===== END MULTIPLE RETURN VALUES OPERATIONS =====

    // ===== QUASIQUOTATION OPERATIONS =====

    // Codegen for quasiquote - process unquotes within quoted structure.
    //
    // Parser shape (lib/frontend/parser.cpp: parse_quasiquoted_list_internal):
    //   `(a ,x b)           => CALL_OP(list, [VAR(a), UNQUOTE(x), VAR(b)])
    //   `(1 ,@xs 5)         => CALL_OP(list, [INT(1), UNQUOTE_SPLICING(xs), INT(5)])
    //   `atom               => the bare atom AST
    //   ,expr               => UNQUOTE_OP(expr)
    //   ,@expr              => UNQUOTE_SPLICING_OP(expr)
    //
    // So the job here is: for CALL_OP(list, [...]), iterate args right-to-left,
    // evaluating UNQUOTE bodies, splicing UNQUOTE_SPLICING bodies, and
    // quoting everything else. Anything that isn't a recognised list shape
    // falls back to codegenQuotedAST, which treats the AST as data.
    Value* codegenQuasiquoteEscape(const eshkol_ast_t* escape);

    Value* codegenQuasiquote(const eshkol_ast_t* ast);

    // ===== END QUASIQUOTATION OPERATIONS =====

    // ===== PATTERN MATCHING OPERATIONS =====

    // Helper: Compare two tagged values for equality (eqv? semantics)
    Value* matchCompareValues(Value* val1, Value* val2);

    // Helper: Check if a value is a pair (cons cell). HEAP_PTR alone also
    // covers strings, vectors, records, ...; the object header's subtype is
    // what makes it a pair, as for `pair?`. The header is read only for a
    // non-null HEAP_PTR.
    Value* matchIsPair(Value* val);

    // Helper: Check if a value is null (empty list)
    Value* matchIsNull(Value* val);

    // Helper: Get car of a pair (assumes val is already known to be a pair)
    Value* matchGetCar(Value* val);

    // Helper: Get cdr of a pair (assumes val is already known to be a pair)
    Value* matchGetCdr(Value* val);

    // Recursive pattern matching - returns i1 indicating if pattern matches
    // If match succeeds, binds pattern variables in symbol_table
    // fail_block: where to branch on match failure
    // continue_block: where to branch on match success (after bindings)
    Value* compilePatternMatch(const eshkol_pattern_t* pattern, Value* val,
                               BasicBlock* fail_block,
                               std::vector<std::pair<std::string, Value*>>& bindings);

    // (match expr (pattern body) ...) - Pattern matching expression
    Value* codegenMatch(const eshkol_operations_t* op);

    // ===== END PATTERN MATCHING OPERATIONS =====

    // Helper function to compare two tagged values using eqv? semantics
    // Returns an i1 (boolean) value
    Value* callBuiltinEqv(Value* arg1, Value* arg2);

    // MIGRATED: Case expression - delegates to ControlFlowCodegen
    Value* codegenCase(const eshkol_operations_t* op);

    // Do loop: (do ((var init step) ...) ((test) result ...) body ...)
    // Parser structure: call_op.func = CONS(bindings-list, test-clause)
    //                   call_op.variables = body expressions
    // Where bindings-list is CALL_OP with CONS bindings (var, CONS(init, step))
    // And test-clause is CONS(test, results-list)
    // ESH-0074c: does any closure created ANYWHERE in this `do` form capture
    // `var`? `main_cons` carries the bindings (inits and steps), the test and the
    // result expressions; `op->call_op.variables[]` carries the body commands.
    // Every one of them can build a closure, and any of them doing so forces the
    // loop variable onto a shared cell — see codegenDo's storage-class comment.
    bool doFormCapturesVar(const eshkol_operations_t* op,
                           const eshkol_ast_t* main_cons,
                           const std::string& var);

    // Assignment conversion is a storage decision, not only a capture
    // decision. A do variable that is set! from its own body must use one
    // durable cell when an escaping continuation can re-enter the loop; a
    // captured variable must use that same cell even without call/cc. Keeping
    // this predicate beside doFormCapturesVar prevents the step/header/body
    // paths from ever selecting different locations.
    bool doFormSetsVar(const eshkol_operations_t* op,
                       const eshkol_ast_t* main_cons,
                       const std::string& var);

    Value* codegenDo(const eshkol_operations_t* op);

    // MIGRATED: Logical NOT - delegates to ControlFlowCodegen
    Value* codegenNot(const eshkol_operations_t* op);

    // MIGRATED: When conditional - delegates to ControlFlowCodegen
    Value* codegenWhen(const eshkol_operations_t* op);

    // MIGRATED: Unless conditional - delegates to ControlFlowCodegen
    Value* codegenUnless(const eshkol_operations_t* op);

    // ============================================================
    // Bitwise Operations (Phase 8)
    // ============================================================

    // bitwise-and: (bitwise-and a b) -> integer AND
    // Helper: emit bignum bitwise dispatch for binary ops
    Value* emitBitwiseBignumDispatch(Value* a, Value* b, int op_code, const char* int_op_name);

    Value* codegenBitwiseAnd(const eshkol_operations_t* op);

    // bitwise-or: (bitwise-or a b) -> integer OR
    Value* codegenBitwiseOr(const eshkol_operations_t* op);

    // bitwise-xor: (bitwise-xor a b) -> integer XOR
    Value* codegenBitwiseXor(const eshkol_operations_t* op);

    // bitwise-not: (bitwise-not a) -> integer NOT (one's complement)
    Value* codegenBitwiseNot(const eshkol_operations_t* op);

    // arithmetic-shift: (arithmetic-shift n count) -> shift n by count bits
    // Positive count = left shift, negative count = right shift (arithmetic)
    Value* codegenArithmeticShift(const eshkol_operations_t* op);

    /**
     * @brief `(bit-shift-left n k)` / `(bit-shift-right n k)`.
     *
     * Documented in docs/tutorials/20_BITWISE_AND_SYSTEM.md:
     *   (bit-shift-left 1 8)    => 256
     *   (bit-shift-right 256 4) => 16
     *
     * These are the directional spellings of `arithmetic-shift` and share its
     * exact semantics — including the limb-aware bignum path and the
     * sign-propagating (arithmetic, to -inf) right shift — so
     *   (bit-shift-left  n k) === (arithmetic-shift n k)
     *   (bit-shift-right n k) === (arithmetic-shift n (- k))
     * A negative count shifts the other way rather than being undefined.
     */
    Value* codegenBitShift(const eshkol_operations_t* op, bool shift_right);

    /**
     * @brief `(popcount n)` / `(bit-count n)` — population count.
     *
     * Documented in docs/tutorials/20_BITWISE_AND_SYSTEM.md as
     * `(popcount 255) => 8` and in docs/API_REFERENCE.md as `(bit-count n)`;
     * the two names are exact synonyms.
     *
     * Contract, defined for every exact integer (fixnum and bignum alike):
     *   n >= 0 : the number of 1 bits in n's binary representation.
     *   n <  0 : Eshkol models a negative integer as an *infinite*
     *            two's-complement bit string (see
     *            docs/breakdown/EXACT_ARITHMETIC.md), which has infinitely
     *            many 1 bits, so the width-independent R6RS
     *            `bitwise-bit-count` convention is used:
     *              (popcount n) = (bitwise-not (popcount (bitwise-not n)))
     *                           = -1 - (popcount (bitwise-not n))
     *            e.g. (popcount -1) => -1, (popcount -256) => -9.
     * Because the rule never mentions a word width, the int64 fast path and
     * the bignum path agree on every value they both represent.
     */
    Value* codegenPopcount(const eshkol_operations_t* op);

    /**
     * @brief Shared body of `arithmetic-shift` / `bit-shift-left` /
     *        `bit-shift-right`.
     *
     * Dispatches to the exact bignum runtime (`eshkol_bignum_bitwise_tagged`
     * op=4) when the operand is already a bignum, *or* when a left shift of an
     * int64 operand would overflow int64.  Without the overflow test the
     * documented `(arithmetic-shift 1 64) => 18446744073709551616`
     * (docs/internal/ESHKOL_V1_LANGUAGE_REFERENCE.md) silently wrapped to
     * -2^63, and `(bit-shift-left 1 100)` would do the same.  Right shifts of
     * an int64 can never overflow, so they stay on the fast path.
     */
    Value* emitArithmeticShiftTagged(Value* n, Value* count);

    // Type predicates - MIGRATED to consolidated type system
    // For immediate types: checks type directly
    Value* codegenTypePredicate(const eshkol_operations_t* op, uint8_t expected_type);

    // Consolidated type predicates - check HEAP_PTR/CALLABLE and subtype in header
    Value* codegenHeapSubtypePredicate(const eshkol_operations_t* op, uint8_t expected_subtype);

    // vector? predicate - returns #t for both Scheme vectors (HEAP_SUBTYPE_VECTOR)
    // and tensor literals created with #(...) syntax (HEAP_SUBTYPE_TENSOR)
    Value* codegenVectorPredicate(const eshkol_operations_t* op);

    // ============================================================================
    // STRING FUNCTIONS
    // ============================================================================

    // Helper: Extract string pointer from tagged value
    Value* extractStringPtr(Value* tagged_val);

    // symbol->string: Convert a symbol to its string representation
    // Symbols and strings share the same data layout, just differ in header subtype
    // Layout: header at offset -8, string data starts at offset 0 (null-terminated)
    // Header has: subtype(i8), flags(i8), ref_count(i16), size(i32)
    // Size field at offset -4 contains string length + 1 (for null terminator)
    Value* codegenSymbolToString(const eshkol_operations_t* op);

    // string->symbol: Convert a string to a symbol.
    //
    // Previously this allocated a fresh symbol per call, which meant two
    // (string->symbol "foo") calls returned distinct objects and broke eq?
    // — the same identity bug as R7RS-1 but on the runtime-constructed
    // side. Fix: route through eshkol_intern_symbol_lookup (the same
    // runtime helper that codegenQuote uses for literal symbols), so
    //   (eq? (string->symbol "foo") (string->symbol "foo")) ⇒ #t
    //   (eq? (string->symbol "foo") 'foo)                   ⇒ #t
    // which is what R7RS §6.5 mandates.
    //
    // The input must be a C-terminated string; Eshkol strings carry their
    // char data directly at the tagged-value's ptr, so we pass it straight
    // to the helper — the helper computes strlen and does its own copy
    // into a properly-headered symbol allocation.
    Value* codegenStringToSymbol(const eshkol_operations_t* op);

    // gensym: generate a fresh, uninterned symbol ("G<counter>", process-wide
    // monotonically increasing counter). Was implemented in
    // lib/core/introspection.cpp (eshkol_gensym / eshkol_gensym_prefix) but
    // never wired into any dispatch table, so `(gensym)` failed with
    // "Unknown function: gensym" on this backend. Wired the same way as the
    // string->symbol sibling above: call the runtime helper for the raw
    // symbol pointer and pack it as a HEAP_PTR tagged value (the header
    // written by arena_allocate_symbol_with_header, inside
    // eshkol_gensym_prefix, is what makes ESHKOL_GET_HEADER report
    // HEAP_SUBTYPE_SYMBOL for it).
    Value* codegenGensym(const eshkol_operations_t* op);

    // ptr->string: Convert a raw C char* pointer (from FFI extern) to an Eshkol string.
    // The FFI extern returns a ptr (i64 holding a char* address). This function:
    //   1. Calls strlen() on the raw pointer to get length
    //   2. Allocates a proper Eshkol string with object header
    //   3. Copies the C string data
    //   4. Returns a tagged HEAP_PTR value
    // This enables: (display (ptr->string (system-capture "echo hello")))
    Value* codegenPtrToString(const eshkol_operations_t* op);

    // ptr->string-n: Copy exactly N bytes from a raw C pointer into an Eshkol
    // string. Unlike ptr->string, this does not call strlen(), so embedded NUL
    // bytes are preserved. This is the required boundary for binary-safe FFI
    // APIs such as HTTP responses that return a pointer and an explicit length.
    Value* codegenPtrToStringN(const eshkol_operations_t* op);

    // ============================================================================
    // CHARACTER FUNCTIONS
    // ============================================================================

    // Helper: Pack a character (as integer codepoint) into tagged value
    Value* packCharToTaggedValue(Value* char_val);

    // ============================================================================
    // SCHEME VECTOR FUNCTIONS (heterogeneous arrays)
    // ============================================================================

    // Vector structure: [i64 length][tagged_value_t elements...]
    // Each element size comes from LLVM's DataLayout for proper alignment


    // vector: Create a vector from given elements
    Value* codegenVector(const eshkol_operations_t* op);

    // Numeric predicates
    Value* codegenNumericPredicate(const eshkol_operations_t* op, const std::string& pred);

    // eq? - Identity comparison (pointer equality for lists, value equality for primitives)
    Value* codegenEq(const eshkol_operations_t* op);

    // eq? logic operating on two already-tagged values (shared by codegenEq and
    // the first-class / apply'd eq? wrapper). Returns a packed boolean.
    Value* emitEqTagged(Value* arg1, Value* arg2);

    // eqv? - Value equality for numbers (including bignums), identity for everything else
    Value* codegenEqv(const eshkol_operations_t* op);

    // eqv? logic operating on two already-tagged values (shared by codegenEqv and
    // the first-class / apply'd eqv? wrapper). Returns a packed boolean.
    Value* emitEqvTagged(Value* arg1, Value* arg2);

    // equal? - Deep structural equality using runtime helper
    Value* codegenEqual(const eshkol_operations_t* op);

    // equal? logic operating on two already-tagged values (shared by codegenEqual
    // and the first-class / apply'd equal? wrapper). Returns a packed boolean.
    Value* emitEqualTagged(Value* arg1, Value* arg2);

    // (type-name v) on an already-tagged value: the runtime classifies it
    // through the shared type-name vocabulary and returns an interned symbol.
    Value* emitTypeNameTagged(Value* tagged);

    // NOTE: codegenNewline has been migrated to StringIOCodegen (strio_->newline)

    // (error-object? obj) — #t iff obj is an error object created by `error`.
    Value* codegenErrorObjectPredicate(const eshkol_operations_t* op);

    // (error-object-message obj) / (error-object-irritants obj) — extract the
    // stored message string or irritant list via an out-param runtime call.
    Value* codegenErrorObjectAccessor(const eshkol_operations_t* op,
                                      const char* runtime_fn, const char* scheme_name);

    Value* codegenError(const eshkol_operations_t* op);

    // =========================================================================
    // SYSTEM & ENVIRONMENT / FILE SYSTEM FUNCTIONS
    // Moved to system_codegen.cpp - delegated via system_-> in dispatch
    // =========================================================================

    Value* codegenSequence(const eshkol_operations_t* op);

    Value* codegenExternVar(const eshkol_operations_t* op);

    Value* codegenExtern(const eshkol_operations_t* op);

    // =========================================================================
    // R7RS ENVIRONMENT PRIMITIVES IMPLEMENTATION
    // =========================================================================

    // (eval expr) or (eval expr env) → Evaluates S-expression via JIT
    Value* codegenEval(const eshkol_operations_t* op);

    // (null-environment 7) → Returns empty environment (only syntax keywords)
    Value* codegenNullEnvironment(const eshkol_operations_t* op);

    // (scheme-report-environment 7) → Returns R7RS standard environment
    // Builds an alist of standard procedure names as symbols.
    // Since all standard bindings are always available through builtin dispatch,
    // the alist serves as introspection data for the R7RS standard set.
    Value* codegenSchemeReportEnvironment(const eshkol_operations_t* op);

    // (interaction-environment) / (current-environment) → Returns current bindings
    // Builds an alist of all globally defined symbol names at compile time.
    Value* codegenInteractionEnvironment(const eshkol_operations_t* op);

    // If expression - delegate to control flow module
    Value* codegenIfCall(const eshkol_operations_t* op);

    // Begin sequence - delegate to control flow module
    Value* codegenBegin(const eshkol_operations_t* op);

    // NOTE: codegenCons, codegenCar, codegenCdr, codegenList, codegenNullCheck, codegenPairCheck
    // have been migrated to CollectionCodegen. See coll_->cons(), coll_->car(), etc.
    // ~1100 lines of old implementations removed (now in collection_codegen.cpp)

    Value* codegenConsCell(const eshkol_ast_t* ast);

    // ===== TAIL CALL OPTIMIZATION SUPPORT =====
    // These functions detect and handle tail-recursive patterns to prevent stack overflow
    // NOTE: The active TailCallContext is in BindingCodegen (binding_codegen.h:283).
    // TCO setup is in letrec()/letrecStar()/codegenFunctionDefinition.
    // TCO interception is in codegenCall (line ~8640) via binding_->isTCOActive().

    // Mutual TCO: set of operation nodes that are non-self tail calls in the current function
    std::unordered_set<const eshkol_operations_t*> mutual_tail_call_sites_;

    // ===== TAIL-TRANSFER DISPATCHER (ADR-0006 §3) ==============================
    //
    // `musttail` is only one of the two lowerings a proper tail call may take.
    // It requires byte-identical LLVM signatures, no argument pointing into the
    // caller's frame, and a backend able to lower an aggregate return that way.
    // The shapes that fail one of those conditions -- different arities between
    // the two procedures, a `guard` frame that owes a handler-stack pop, and
    // every non-AArch64 target -- take the transfer lowering instead.
    //
    // A transfer does not call the callee. It copies the evaluated arguments
    // into the thread's eshkol_tail_transfer_t, records the callee's uniform
    // entry, sets `pending`, and lets the value flow to the function's ordinary
    // return. Returning normally is exactly what makes `guard` sound here where
    // `musttail` is not: leaving the guard still runs its
    // eshkol_pop_exception_handler(), because nothing about the epilogue is
    // skipped. The driver loop then runs the transfer in the caller's stead.
    //
    // Three symbols implement it for a participating procedure F:
    //
    //   F                    public entry, unchanged type and linkage. Its body
    //                        becomes the DRIVER: call F<BODY>, then loop while a
    //                        transfer is pending. Every ordinary caller -- Eshkol
    //                        code, the FFI, a closure value, an AOT export --
    //                        keeps calling this symbol and is unaffected.
    //   F<BODY>              internal, same type. The real body, with transfer
    //                        sites in it. It never drives, so a transfer chain
    //                        entered from one driver reuses that one frame.
    //   F<UNIFORM>           internal, (const tagged*, i64) -> tagged. Loads the
    //                        record's arguments into SSA values and calls
    //                        F<BODY>. This is where differing arities stop
    //                        mattering: every transfer target has this one shape.
    //
    // musttail sites target F<BODY> rather than F, because musttail into a
    // driver would leave that driver's frame live for the rest of the chain --
    // one frame every two hops, which is the growth this work exists to remove.
    // A callee that was never split still needs the symbol, so
    // finalizeTailTransferThunks() gives every remaining F<BODY> declaration a
    // one-instruction forwarder.
    static constexpr const char* kTailBodySuffix = "__eshkol_tail_body";
    static constexpr const char* kTailUniformSuffix = "__eshkol_tail_uniform";

    // Set for the duration of one define body that is allowed to emit transfer
    // sites; null everywhere else. Guards against a nested lambda or a stale
    // AST-node match emitting a transfer into a function that has no driver.
    Function* tail_transfer_home_ = nullptr;
    // True once tail_transfer_home_ has emitted EITHER lowering. Both need the
    // driver: a transfer queues one directly, and a `musttail` into a callee's
    // body hands this frame to code that may queue one, with no other frame left
    // to catch it. A function with no mutual tail call at all is untouched.
    bool tail_transfer_emitted_ = false;
    // Every F<BODY> this module has referred to, so finalization can tell a real
    // split body from a declaration that still needs a forwarder.
    std::vector<Function*> tail_body_decls_;

    /**
     * @brief Declare eshkol_tail_transfer_slot(), the per-thread record accessor.
     */
    Function* getTailTransferSlotFunc();

    /**
     * @brief The (const tagged*, i64) -> tagged shape every uniform entry has.
     */
    FunctionType* tailUniformEntryType();

    /**
     * @brief The F<BODY> symbol for @p target, declaring it if needed.
     *
     * Returns null when the name is already taken by something of another type,
     * in which case the caller must keep the ordinary lowering rather than emit
     * a call it cannot type-check.
     */
    Function* getOrDeclareTailBody(Function* target);

    /**
     * @brief The F<UNIFORM> entry for @p target, generating it on first use.
     *
     * Loads exactly the callee's declared number of arguments out of the
     * record-owned buffer -- by value, before calling the body, so a transfer
     * the body performs may overwrite that buffer -- and calls F<BODY>. Any
     * argument the transfer did not supply arrives as tagged null, matching
     * what the ordinary arity-mismatch path would have produced.
     */
    Function* getOrCreateTailUniformEntry(Function* target);

    /**
     * @brief Field address inside the thread's transfer record.
     */
    Value* tailTransferField(Value* slot, size_t offset, const char* name);

    /**
     * @brief Emit a tail TRANSFER to @p callee in place of a call.
     *
     * Stores the arguments into the thread's record, names the callee's uniform
     * entry as the target, raises `pending`, and returns the value the enclosing
     * expression should yield -- tagged null, which no epilogue interprets. The
     * caller must NOT emit a `ret`: the whole point is that control leaves this
     * frame through the ordinary return path, running every pop, scope end and
     * handler unwind that path owes.
     *
     * @return the placeholder value, or null if this site cannot be transferred
     *         (in which case the caller keeps the ordinary, bounded lowering).
     */
    Value* emitTailTransfer(Function* callee, const std::vector<Value*>& args);

    /**
     * @brief Turn @p f into F (driver) + F<BODY> once its body is complete.
     *
     * The body is MOVED, not copied: `f` keeps its name, type, linkage and
     * Function* identity, so every table, export wrapper and closure value that
     * already refers to it stays correct, while the code that contains transfer
     * sites moves into an internal symbol that never drives.
     *
     * @return true if the split happened.
     */
    bool splitFunctionForTailDriver(Function* f);

    /**
     * @brief Give every F<BODY> that was referenced but never split a forwarder.
     *
     * A musttail site names the callee's F<BODY> without knowing whether that
     * callee will contain transfer sites of its own. When it does not, the
     * symbol is still owed a definition: a single forwarding call to the public
     * entry. It is a `musttail` forward wherever the target can lower one, so
     * the forwarder's frame is replaced rather than stacked and the chain stays
     * flat; where it cannot, transfers are the only lowering in use and the
     * forwarder is entered once per hop from a driver that reclaims it.
     */
    void finalizeTailTransferThunks();

    // ═══════════ SW-58: a `guard` that carries a self-recursive TCO loop ═══════════
    //
    // ESH-0222 made a `guard` body a tail position for the self-call it
    // contains and had the back edge DRAIN the handler chain. That is what the
    // resident tick loop needs — its error boundary is re-entered once per
    // tick, forever, so the chain must not grow — and it is exactly wrong for a
    // handler that RE-RAISES. R7RS 7.3 derives `guard` from
    // `with-exception-handler` wrapping the body, so every activation installs
    // its own handler and they NEST; a re-raise out of the innermost one has to
    // find the PREVIOUS activation's guard, which answers with THAT
    // activation's variables. Draining them sent the re-raise to whatever stood
    // outside the loop, and a different handler answered — silently. That is
    // SW-58, whose fixture is
    // tests/tco/guard_tail_context/04_reraise_reaches_enclosing_guard.esk.
    //
    // Reverting ESH-0222 would trade a rare wrong answer for a common crash, so
    // it is not the fix. Instead every guard that carries a loop is classified
    // into one of three lowerings. ALL THREE ARE SEMANTICALLY EXACT — they
    // differ only in what they cost:
    //
    //   Collapse  the enclosing activations are PROVABLY unobservable, so
    //             draining them changes nothing a program can see. Constant
    //             stack and flat RSS. This is the resident tick-loop shape.
    //   Replay    keep the handler frames standing and attach the departing
    //             activation's loop parameters to each. Constant stack; one
    //             handler frame per LIVE guard — the space R7RS's own semantics
    //             require, moved off the native stack onto the heap.
    //   Forbid    the clauses read a binding the loop rebinds every iteration,
    //             which no snapshot of the loop PARAMETERS can restore. Leave
    //             the call a real call: R7RS's own stack cost, never a guess.
    enum class GuardLoopClass { Collapse, Replay, Forbid };

    // Names bound inside the innermost TCO loop currently being generated,
    // minus its own parameters. Set by setupGuardReplayForLoop, consulted by
    // codegenGuard, saved/restored alongside the rest of the TCO context by
    // codegenNamedLet.
    std::set<std::string> tco_loop_bound_names_;
    bool in_do_loop_codegen_ = false;

    // Operators whose Eshkol lowering has no path to `eshkol_raise`, so an
    // expression built from them cannot transfer control to an ENCLOSING
    // handler while a guard's clauses are running (the guard's own handler is
    // already removed at that point, so such a transfer is precisely what makes
    // the enclosing activations observable).
    //
    // Deliberately small and total. `/`, `modulo`, `quotient` and `remainder`
    // are absent because a zero divisor raises; `car`/`cdr`/`vector-ref` are
    // absent because a domain violation raises; anything user-defined is absent
    // because its body is unbounded. A name missing from this set costs a
    // Replay lowering — correct, and one handler frame per iteration — never a
    // wrong answer, so the set may be grown by measurement but never guessed.
    static const std::set<std::string>& guardNonRaisingOperators();

    // Can evaluating `e` inside a guard clause reach an enclosing handler?
    // Conservative: anything not modelled answers "yes".
    //
    // A self tail call to `loop_name` counts as non-raising because it is not a
    // call at all in this lowering — it is the loop's back edge, and whatever
    // the next activation raises is caught by the guard that activation
    // re-establishes. (That is an induction over the loop, and it is sound
    // exactly when this predicate holds for every clause, which is the
    // condition Collapse is granted under.)
    bool guardClauseExprCannotRaise(const eshkol_ast_t* e,
                                    const std::string& loop_name);

    // Does this guard have a clause that always matches — `else`, or a literal
    // true test?  Without one, a raise the clauses do not match is re-raised
    // into the enclosing handler by R7RS, which is the enclosing ACTIVATION's
    // guard, which makes the collapsed activations observable.
    bool guardHasCatchAllClause(const eshkol_operations_t* g);

    // Names bound INSIDE a loop body — let/let*/letrec/letrec*/named let/do
    // variables and lambda parameters. A guard clause that reads one of these
    // cannot be replayed from a snapshot of the loop PARAMETERS, because that
    // binding's storage is reused by the next iteration.
    void collectLoopBoundNames(const eshkol_ast_t* ast, std::set<std::string>& out);

    // Is there a call to `loop_name` in TAIL position of `body`?  Mirrors the
    // tail-context rules isInTailPosition() encodes, walking DOWN from a known
    // tail expression rather than up from a known call.
    bool tailPositionHasSelfCall(const eshkol_ast_t* body,
                                 const std::string& loop_name);

    // Does this guard's BODY carry the loop — i.e. does a back edge get taken
    // from inside its dynamic extent?  Only those guards constrain the
    // lowering; a guard whose self-call sits in a CLAUSE has already had its
    // own handler removed when that call runs, so it leaves nothing behind.
    bool guardCarriesLoop(const eshkol_operations_t* g, const std::string& loop_name);

    GuardLoopClass classifyGuardForLoop(const eshkol_operations_t* g,
                                        const std::string& loop_name,
                                        const std::set<std::string>& loop_bound);

    // Whole-loop verdict: does ANY guard that carries this loop need the replay
    // lowering?  One does ⇒ every guard back edge in the loop takes it, which
    // keeps a single rule for the whole loop and is always exact (Collapse is
    // only ever an optimisation over Replay).
    bool loopNeedsGuardReplay(const eshkol_ast_t* ast, const std::string& loop_name,
                              const std::set<std::string>& loop_bound);

    // One-shot setup for a loop that is about to be lowered: decides the guard
    // lowering and, when it is Replay, allocates the two entry-block slots the
    // mechanism needs. `arity` is the loop's parameter count.
    void setupGuardReplayForLoop(eshkol::BindingCodegen::TailCallContext& tco_ctx,
                                 const eshkol_ast_t* loop_body,
                                 const std::string& loop_name);

    // Declarations for the four SW-58 runtime entry points.
    enum class GuardReplayFn { Depth, UnwindTo, Snapshot, Restore };
    Function* getGuardReplayFn(GuardReplayFn which);

    // Emitted on every path that leaves a replaying guard loop: drop whatever
    // replay frames the loop's back edges left standing. Without it a loop that
    // exits WITHOUT passing through the guard (`(if done acc (guard ...))`)
    // would leave live handler frames whose landing pad is a frame the function
    // is about to return from.
    void emitGuardReplayUnwindAt(Value* mark_slot);

    // Check if an AST node is in tail position within its parent
    // Note: IF_OP uses call_op structure with variables[0]=cond, [1]=then, [2]=else
    bool isInTailPosition(const eshkol_ast_t* expr, const eshkol_ast_t* body);

    // Count ALL recursive calls (both tail and non-tail) to a specific function name
    size_t countAllRecursiveCalls(const eshkol_ast_t* ast, const std::string& func_name);

    // Find all tail calls to a specific function name in an AST
    // Note: IF_OP uses call_op structure with variables[0]=cond, [1]=then, [2]=else
    void findTailCalls(const eshkol_ast_t* ast, const eshkol_ast_t* body,
                       const std::string& func_name,
                       std::vector<const eshkol_operations_t*>& tail_calls);

    // Collect ALL non-self tail call sites in a function body for mutual TCO.
    // Walks the AST to find function calls in tail position that are NOT self-recursive
    // (self-recursive calls are handled by the loop-based TCO above).
    void collectMutualTailCallSites(const eshkol_ast_t* ast, const eshkol_ast_t* body,
                                     const std::string& self_name);

    // Check if a lambda/function is self-tail-recursive (calls itself ONLY in tail position)
    // Returns true only if ALL recursive calls are in tail position
    bool isSelfTailRecursive(const eshkol_operations_t* op, const std::string& func_name);

    // ═════════════════════════════════════════════════════════════════════
    // ESH-0214b: AUTOMATIC PER-ITERATION ARENA SCOPE RECLAMATION
    //
    // When a named-let loop compiles to the branch-based TCO transform, its
    // body's arena allocations normally accumulate in the enclosing
    // (typically process-lifetime) arena forever -- the unbounded-RSS
    // daemon-loop failure mode of ESH-0214. Here we make the loop reclaim
    // its own per-iteration garbage automatically:
    //
    //   * codegenNamedLet pushes an arena scope at the top of the TCO loop
    //     header block (once per iteration),
    //   * every TCO back edge and the loop's exit path end the scope via
    //     eshkol_arena_iter_scope_end(arena, outflowing_values, n), which
    //     POPS the scope (rewinds -- reclaims the whole iteration) when no
    //     out-flowing value points into the iteration's allocation span, and
    //     COMMITS it (keeps the memory, drops the mark -- the pre-existing
    //     behavior) otherwise.
    //
    // Correctness rests on two pillars:
    //   1. A conservative STATIC analysis (below) that only enables the
    //      mechanism when the loop body cannot leak an iteration-allocated
    //      value through any channel other than the loop-carried arguments
    //      or the loop result: no set!, no mutating builtins (any name
    //      ending in '!', plus an explicit blacklist), no unknown callees
    //      (only whitelisted allocation-escape-free builtins, local named
    //      lets, inline lambdas, and recursively-analyzed user defines), no
    //      exception/continuation control flow that could skip the scope
    //      end, no parallel constructs (worker arenas interleave with the
    //      scope span).
    //   2. A DYNAMIC per-value check at each scope end (the runtime helper)
    //      for the two channels that remain: loop args and loop result. A
    //      heap value allocated in the ending iteration flips that edge to
    //      commit -- so e.g. an accumulator loop keeps today's keep-
    //      everything behavior, while a loop whose carried state is numeric
    //      (or pre-loop-allocated, like a port) reclaims every iteration.
    //
    // Anything the analysis cannot prove safe silently disables the
    // mechanism (the loop just behaves exactly as before this feature).
    // ESHKOL_NO_ITER_SCOPE=1 disables it globally for debugging.
    // ═════════════════════════════════════════════════════════════════════

    // Memo for per-user-function analysis results (name -> safe?)
    std::unordered_map<std::string, bool> iter_scope_fn_memo;

    // ESH-0214e: parallel memo recording whether an escape-safe user function's
    // body contains a barriered structural mutation, so a loop that calls it is
    // lowered with a nursery region even on a memo hit (a mutation reached only
    // through a called define must still flip the whole loop to nursery mode, or
    // the arena-scope path would rewind a slot the mutation stored into
    // persistent state). Keyed identically to iter_scope_fn_memo.
    std::unordered_map<std::string, bool> iter_scope_fn_nursery_memo;

    // ESH-0214e: set by iterScopeSafeExpr while analyzing ONE loop body when it
    // admits a barriered structural mutation (see iterScopeNurseryMutators).
    // loopBodyIterScopeSafe resets it before each analysis and reports it out.
    // Mutation channels admitted here MUST be exactly the ones whose codegen
    // unconditionally emits eshkol_region_write_barrier_into on the mutated
    // structure pointer, so a nursery-allocated value stored into persistent
    // state is deep-promoted out of the nursery at the store.
    bool iter_scope_needs_nursery_ = false;
    bool iter_scope_direct_exit_allowed_ = false;
    bool iter_scope_has_terminal_exit_ = false;
    bool iter_scope_tensor_dot_allowed_ = false;

    class IterScopeExitAdmissionGuard {
    public:
        IterScopeExitAdmissionGuard(bool& flag, bool value)
            : flag_(flag), saved_(flag) { flag_ = value; }
        ~IterScopeExitAdmissionGuard() { flag_ = saved_; }
    private:
        bool& flag_;
        bool saved_;
    };

    // The structural mutators admitted into iter-scope under ESH-0214e. Each is
    // barriered UNCONDITIONALLY at its codegen site on the mutated structure's
    // pointer (collection_codegen.cpp vector-set!/vector-fill!, hash_codegen.cpp
    // hash-table-set!, llvm_codegen.cpp set-car!/set-cdr!), so the write barrier
    // promotes any nursery value they store into an outer/persistent structure.
    // set! is deliberately EXCLUDED: its barrier fires only for GlobalVariable
    // targets, and proving a set! target is global (not a shadowing enclosing-
    // scope local, whose alloca is NOT barriered) needs full lexical resolution
    // this downward-only analysis does not have — admitting it unsoundly would
    // reintroduce exactly the dangling-pointer corruption this series closes.
    static const std::set<std::string>& iterScopeNurseryMutators();

    // ═════════════════════════════════════════════════════════════════════
    // ESH-0214c: PARALLEL-WORKER REACHABILITY (whole-program pre-pass)
    //
    // iterScopeSafeExpr above proves a loop body cannot LEAK a value through
    // any channel other than loop args / loop result. That is necessary but
    // not sufficient: __global_arena's scope stack (arena_commit_scope /
    // eshkol_arena_iter_scope_end, see arena_memory.h + runtime_arena_core.cpp)
    // is a single, un-synchronized linked list shared by every OS thread --
    // "thread-safe" there only covers the bump-pointer allocation path
    // (arena_create_threadsafe), not scope push/commit/pop. If a named-let's
    // *enclosing function* is ever invoked on a parallel-map/-fold/-filter/
    // -for-each/-execute or future worker thread, that thread's per-iteration
    // scope push/pop races against every other worker doing the same thing
    // concurrently and corrupts the shared scope list -> SIGSEGV. This is
    // exactly what tests/parallel/parallel_flags_byte_regression.esk hits:
    // `work` (run on worker threads by parallel-map) calls `busy-int64`,
    // whose named-let loop body is otherwise perfectly escape-safe.
    //
    // The hazard lives in the *caller* context, which iterScopeSafeExpr
    // cannot see (it only walks downward from the loop body). So: once per
    // compilation, before any function is codegen'd, walk the whole program
    // for calls into a parallel/future constructor, and mark every AST node
    // transitively reachable from each callback argument (through ordinary
    // calls) as parallel-worker code. namedLetIterScopeSafe then hard-rejects
    // any loop whose body lands in that set, before running the local
    // analysis. Conservative by construction: a function that merely *can*
    // run on a worker thread is excluded everywhere it is defined, even at
    // call sites that never go through parallel-map. Known precision loss:
    // limited to the single generateIR() invocation's AST (a REPL line
    // referencing a not-yet-compiled future parallel-map user is not
    // caught -- consistent with the pre-existing per-function analysis,
    // which has the same single-compilation-unit horizon).
    // ═════════════════════════════════════════════════════════════════════

    std::unordered_set<const void*> parallel_worker_ast_nodes;

    // Builtins whose callback argument(s) run on a worker thread (or, for
    // `future`, a continuation not provably the calling thread). Everything
    // but parallel-execute takes the callback as argument 0; parallel-execute
    // treats every argument as an independently-scheduled thunk.
    static bool iterScopeIsParallelCallSite(const std::string& name, bool& all_args_are_callbacks);

    // Mark `name`'s body (if known) as parallel-worker-reachable. Guarded by
    // `visiting_fns` so mutually recursive helpers terminate.
    void iterScopeMarkFnUnsafe(const std::string& name,
                                const std::unordered_map<std::string, const eshkol_ast_t*>& all_fn_bodies,
                                std::set<std::string>& visiting_fns);

    // Single recursive walk used both to scan the whole program for calls
    // into a parallel/future constructor (mark=false: nothing is inserted,
    // we are only looking for such call sites) and to flood-fill a reachable
    // subtree once one is found (mark=true: every node visited is recorded
    // in parallel_worker_ast_nodes). Either way, any call into a parallel
    // constructor found along the walk switches its callback argument(s) to
    // mark=true regardless of the walk's own mode.
    void iterScopeWalkParallelReach(const eshkol_ast_t* node, bool mark,
                                     const std::unordered_map<std::string, const eshkol_ast_t*>& all_fn_bodies,
                                     std::set<std::string>& visiting_fns);

    // Entry point: call once, early in generateIR(), on the whole flattened
    // program before any function body is codegen'd.
    void computeParallelWorkerReachability(const eshkol_ast_t* asts, size_t num_asts);

    // First-order builtins that cannot leak an argument or an allocation
    // into a structure that outlives the call: pure computation, fresh-
    // allocation constructors, read-side I/O, and output of DATA (display
    // writes bytes, it does not retain pointers). Anything absent → unsafe.
    static const std::set<std::string>& iterScopeSafeBuiltins();

    // Higher-order builtins that CALL one of their arguments but do not
    // retain pointers beyond their own return value. Maps builtin name to
    // the index of the function-valued argument, which must itself be
    // analyzable (inline lambda, local loop, or known user define).
    static const std::map<std::string, int>& iterScopeHigherOrderBuiltins();

    // Explicit blacklist: names without a trailing '!' that still leak
    // pointers into global/runtime state, run arbitrary code, or introduce
    // control flow that could skip the balanced scope end.
    static const std::set<std::string>& iterScopeBlacklist();

    static bool iterScopeNumericTensorElement(const eshkol_ast_t* expr);

    bool iterScopeSafeTensorLiteral(const eshkol_ast_t* elements,
                                   const uint64_t* dimensions,
                                   uint64_t num_dimensions,
                                   uint64_t total_elements,
                                   std::set<std::string>& local_fns,
                                   std::set<std::string>& analyzing,
                                   int depth);

    bool iterScopeSafeTensorDotOperand(const eshkol_ast_t* expr,
                                       uint64_t* vector_length,
                                       std::set<std::string>& local_fns,
                                       std::set<std::string>& analyzing,
                                       int depth);

    // Is `expr` free of iteration-escape channels? local_fns holds names
    // callable as plain loops from this position (the enclosing loop name +
    // any locally nested named-let names); analyzing holds user-define names
    // currently on the recursion stack (cycles are resolved inductively).
    bool iterScopeSafeExpr(const eshkol_ast_t* expr,
                           std::set<std::string>& local_fns,
                           std::set<std::string>& analyzing,
                           int depth);

    // Analyze a user-defined function's body by name (define-level map).
    // Memoized; a name already on the analysis stack is resolved inductively
    // (recursive helpers are safe iff every other path through them is).
    bool iterScopeSafeUserFn(const std::string& name,
                             std::set<std::string>& analyzing,
                             int depth);

    // Gate for codegenNamedLet AND the self-tail-recursive define TCO path
    // (ESH-0214b Bug 1): is this loop body eligible for automatic
    // per-iteration scope reclamation? `loop_name` is the name whose
    // self-tail-calls are the loop's back edges (the named-let name, or the
    // define's own function name).
    bool loopBodyIterScopeSafe(const eshkol_ast_t* body, const std::string& loop_name,
                               bool* out_needs_nursery = nullptr,
                               bool loop_shadows_exit = false,
                               bool loop_shadows_tensor_dot = false);

    // Emit the end-of-iteration scope release: store the out-flowing tagged
    // values into an entry-hoisted scratch array and call the runtime helper,
    // which pops the scope when none of them escape it and commits otherwise.
    // SW-164: open the loop's LOOP scope once, in the setup block that
    // dominates the loop header. It sits OUTSIDE the per-iteration scope and
    // is what gives an escaping back edge somewhere to rewind to: without it
    // the runtime can only retain the iteration (the pre-SW-164 behavior),
    // because the previous iteration's promoted accumulator lives below the
    // iteration mark and nothing can reclaim it. Balanced by
    // emitIterScopeFinish on every exit path.
    void emitLoopScopeBegin();

    // SW-164: end an iteration scope and RETURN the out-values as they stand
    // afterwards. The runtime promotes survivors out of the span it rewinds and
    // rewrites them in the scratch array in place, so the values that go in are
    // not necessarily the values that come out — the caller must use these,
    // exactly as the ESH-0214e nursery recycle already required.
    //
    // @param finish  false at a tail-call back edge (the loop scope is reopened
    //                for the next iteration); true at the loop's exit (both the
    //                iteration scope and the loop scope are closed for good).
    std::vector<Value*> emitIterScopeEndImpl(const std::vector<Value*>& out_values,
                                             bool finish);

    std::vector<Value*> emitIterScopeEnd(const std::vector<Value*>& out_values);
    Value* emitIterScopeFinish(Value* out_value);
    void finishIterScopeBeforeExit();
    // ═══════════════════ END ESH-0214b ═══════════════════

    // ═══════════════════ ESH-0214e: nursery lowering for mutating loops ═══════
    // A loop whose body is escape-safe AND contains a barriered structural
    // mutation is lowered with a per-loop NURSERY REGION instead of the
    // arena-scope path. These three emitters reuse the EXACT runtime entry points
    // codegenWithRegion emits (region_create/push/enter/escape/pop/leave), so the
    // nursery converges on the 48-h-validated with-region deep-promotion path
    // rather than forking a second evacuator. See runtime_regions.cpp
    // (eshkol_iter_nursery_recycle) for the runtime side.
    struct IterNurseryFns {
        FunctionCallee create, push, pop, enter, leave, recycle, escape_into;
    };
    IterNurseryFns getIterNurseryFns();

    // Open the loop's nursery region in the current (setup) block, ONCE per loop
    // activation. Stashes the region ptr and the eshkol_region_enter displaced-
    // arena token into tco_ctx — both SSA values in the setup block, which
    // dominates every back edge and the loop exit.
    void emitIterNurseryOpen(eshkol::BindingCodegen::TailCallContext& tco_ctx);

    // End-of-iteration recycle at a TCO back edge: store the loop-carried
    // out-values into an entry-hoisted scratch array, call the runtime recycle
    // (promote them out of the nursery + reset the nursery), then load the
    // PROMOTED values back so the caller stores those into the loop's parameter
    // allocas (the originals are about to be reclaimed).
    std::vector<Value*> emitIterNurseryRecycle(const std::vector<Value*>& out_values,
                                               Value* region);

    // Loop exit: escape the result out of the nursery (same as with-region's
    // result escape), then tear the nursery down — region_pop frees its arena,
    // eshkol_region_leave restores the displaced allocation arena (@p saved_arena
    // is the eshkol_region_enter token from emitIterNurseryOpen). Returns the
    // escaped result value to return from the loop. Takes @p saved_arena
    // explicitly so the caller may hold it in a local past any tco_ctx reset.
    Value* emitIterNurseryClose(Value* result, Value* saved_arena);
    // ═══════════════════ END ESH-0214e ═══════════════════

    // Generate a tail call as a jump back to loop header (TCO transformation)
    // Uses the binding module's TCO context
    Value* codegenTailCallFromContext(const eshkol_operations_t* call_op,
                                       eshkol::BindingCodegen::TailCallContext& tco_ctx);

    // Shared core of a TCO loop back-edge: evaluate `arg_nodes` (with TCO
    // suppressed during the evaluation, since arguments are never in tail
    // position), store the resulting values into the loop's parameter allocas,
    // and branch to the loop header. Returns a null sentinel (the block is
    // terminated by the branch and the caller never observes a real value), or
    // nullptr if TCO is inactive or the argument count does not match the
    // loop's arity, in which case the caller must fall back to a normal call.
    Value* emitTCOBackEdge(const std::vector<const eshkol_ast_t*>& arg_nodes,
                           eshkol::BindingCodegen::TailCallContext& tco_ctx);
    // ===== END TAIL CALL OPTIMIZATION SUPPORT =====

    // ESH-0074c: which node test astScanVar() applies while walking. See the
    // comment on astScanVar for the contract.
    enum class VarScanMode { SetTarget, ClosureCapture, ObservationContext };

    // True iff `var` is bound by this parameter list (so an inner reference to it
    // is the PARAMETER, not a capture of the enclosing binding).
    bool paramListShadows(const eshkol_ast_t* params, uint64_t num_params,
                          const char* rest_param, const std::string& var);

    // Same question for a let/named-let binding list, whose entries are
    // (variable value) pairs.
    bool bindingListShadows(const eshkol_ast_t* bindings, uint64_t num_bindings,
                            const std::string& var);

    // Helper function to find free variables in a lambda body
    // bound_vars tracks all variable names bound by enclosing lambdas (for nested closure support)
    // CLOSURE-OVER-NAMED-LET-LOOPVAR FIX: Returns true iff `var` is the target of
    // a SET_OP anywhere in the AST subtree `ast`. Mirrors the recursion structure
    // of findFreeVariablesImpl.
    //
    // COMPLETENESS MATTERS (ESH-0214f / vm-parity `do_set_param_native`): this
    // predicate also drives ASSIGNMENT CONVERSION — boxing a set!-mutated
    // function/lambda parameter into an alloca (codegenFunctionDefinition ~11252,
    // codegenLambda ~27380). There, a MISSED set! is not merely "incomplete": the
    // parameter stays a by-value LLVM Argument, so codegenSet finds no storage
    // location and the assignment is dropped with a "not mutable" diagnostic. That
    // is exactly how
    //     (define (f n) (do ((i 0 (+ i 1))) ((= i 3)) (set! n (+ n 1))) n)
    // returned 10 instead of 13: `do` lowers to the call_op layout but had no case
    // here, and nothing walked into ESHKOL_CONS subtrees (which is where `do`
    // keeps its bindings/steps/test/result forms), so the set! was invisible.
    //
    // The safe direction of error is to over-approximate — a spurious `true` only
    // costs one unused alloca / one arena-promoted capture, whereas a spurious
    // `false` silently miscompiles a mutation. So every structural container the
    // parser can produce is walked, and `default: return false` is now reached
    // only by genuine leaves (literals, quote, type annotations, macro defs).
    //
    // ESH-0074c: that completeness is scarce, so the traversal is shared rather
    // than copied. `VarScanMode` selects which NODE TEST is applied while the
    // walk itself stays byte-identical:
    //   SetTarget      — `var` is the target of a set! (the original predicate).
    //   ClosureCapture — `var` is referenced from inside a form whose capture
    //                    machinery SHARES the enclosing cell by pointer, and which
    //                    arena-moves a stack alloca to do it: a lambda or a named
    //                    let. Internal function defines capture through
    //                    module-level storage instead and are excluded on purpose
    //                    (see the DEFINE_OP case).
    // The second mode drives codegenDo's storage-class decision; see
    // doFormCapturesVar().
    bool astScanVar(const eshkol_ast_t* ast, const std::string& var,
                    VarScanMode mode, unsigned scope_depth = 0);

    // Collect a mutation summary for ASTs that contain no nested binder. Such
    // bodies are the hot path for large generated lets/letrecs; binder-shaped
    // ASTs deliberately fall back to astScanVar so shadowing remains exact.
    bool collectFlatMutationTargets(const eshkol_ast_t* ast,
                                    std::unordered_set<std::string>& targets);

    // Returns true iff `var` is the target of a set! anywhere in `ast`.
    bool astSetsVar(const eshkol_ast_t* ast, const std::string& var);

    // ESH-0074c: returns true iff some closure created inside `ast` captures
    // `var` THROUGH THE POINTER-SHARING PATH — i.e. `var` is referenced from the
    // body of a lambda or a named let and is not shadowed by that form's own
    // bindings. Drives codegenDo's storage-class decision.
    bool astVarCapturedByNestedClosure(const eshkol_ast_t* ast, const std::string& var);

    // Shared assignment-conversion observation query used by BindingCodegen.
    // This includes compiler-generated contexts such as guard handlers, not
    // only source-level lambda nodes.
    bool astMayBeObservedAfterMutation(const eshkol_ast_t* ast,
                                       const std::string& var);

    // ── Escape analysis for a captured continuation ───────────────────────
    //
    // Snapshotting the C stack is what makes a continuation survive its
    // frame's return, but it costs an arena copy proportional to the live
    // stack at every capture. The overwhelmingly common use of call/cc —
    // early return / exception-style unwinding, `(call/cc (lambda (k) … (k v)
    // …))` — never needs it: such a continuation can only be invoked while
    // proc is running, hence while the capturing frame is still live, so the
    // plain setjmp/longjmp path is already correct and free. Measured before
    // this analysis: an escape-only call/cc at depth 200 burned ~270KB of
    // arena per capture and drove a 3000-iteration loop into the 1GB heap
    // limit, where it had previously used none.
    //
    // Returns true only when EVERY use of `name` is the operator of a direct
    // call and none appears inside a nested lambda or named let (which could
    // outlive the frame). Anything this does not model falls through to
    // astReferencesVar(), so an unrecognised form is treated as escaping. The
    // safe direction of error is `false`: a spurious false only pays for a
    // stack copy that was not needed, whereas a spurious true would drop the
    // image a later re-entry depends on.
    bool continuationUseStaysLocal(const eshkol_ast_t* ast, const std::string& name);

    bool astReferencesVarScoped(const eshkol_ast_t* ast, const std::string& var,
                                bool shadowed);

    bool astReferencesVar(const eshkol_ast_t* ast, const std::string& var);

    /**
     * @brief True iff @p name is used as a VALUE (not merely called) anywhere in
     *        @p ast — `(set! g loop)`, `(list loop)`, `(map loop xs)`, a return
     *        position — as opposed to the direct call `(loop …)`.
     *
     * codegenNamedLet uses this to decide whether the loop procedure can outlive
     * the frame that created it. If it can, every capture cell must live in the
     * arena rather than on the caller's stack: a leaked procedure reads its
     * captures through the cell pointer, and a stack cell is dangling as soon as
     * the enclosing function returns. That was a real crash, not a theoretical
     * one — a read-only captured PARAMETER (materialised into an entry-block
     * alloca) came back as a pair once the frame had been reused, so
     * `(< i n)` inside a leaked loop body reported "expected number, got pair".
     *
     * Only the direct-callee position is excluded, and only for CALL_OP. IF_OP /
     * COND_OP share the call_op layout but their `func` field is the condition,
     * so it is inspected normally. Unhandled op kinds fall back to
     * astReferencesVar, which counts a direct call too: that OVER-approximates
     * escape, which costs one arena cell and can never be incorrect.
     */
    bool astUsesNameAsValue(const eshkol_ast_t* ast, const std::string& name);

    void findFreeVariablesImpl(const eshkol_ast_t* ast,
                          const std::unordered_map<std::string, Value*>& current_scope,
                          const eshkol_ast_t* parameters, uint64_t num_params,
                          std::vector<std::string>& free_vars,
                          std::unordered_set<std::string>& bound_vars);

    // Wrapper function for findFreeVariablesImpl - initializes empty bound_vars set
    void findFreeVariables(const eshkol_ast_t* ast,
                          const std::unordered_map<std::string, Value*>& current_scope,
                          const eshkol_ast_t* parameters, uint64_t num_params,
                          std::vector<std::string>& free_vars);

    Value* codegenLambda(const eshkol_operations_t* op);


    /**
     * @brief Build (once per loop function) the trampoline that lets an escaped
     *        named-let loop procedure be entered through the ordinary closure ABI.
     *
     * A named-let loop function is `tagged f(tagged p0..pk-1, ptr c0..cm-1)`,
     * where each `ci` points DIRECTLY at the shared cell holding capture `i`.
     * The closure dispatcher, by contrast, passes `&env.captures[i]` — a slot
     * holding a tagged value. The trampoline bridges exactly that one level:
     * each slot carries the cell address as a tagged INT64 (the same convention
     * `capturePointerTagFromCurrentFunction` uses when a body lambda captures a
     * `_cap` argument), so the thunk loads it, turns it back into a pointer, and
     * forwards it. Without the trampoline the dispatcher would hand the loop
     * function pointers to the env slots themselves and every read of a captured
     * variable would return the tag bytes instead of the value.
     *
     * Emitted lazily: only a named let whose loop procedure is actually used as
     * a value pays for it, so the stdlib's many ordinary loops emit no extra IR.
     */
    Function* getNamedLetEscapeThunk(const NamedLetEscapeInfo& info);

    /**
     * @brief Materialise a first-class closure for a named-let loop procedure
     *        used as a VALUE, or nullptr when the capture cells are not
     *        reachable from the function currently being emitted.
     *
     * Each capture's shared cell is located the same three ways the rest of the
     * backend locates one: the loop function's own `<fv>_cap` pointer argument,
     * an enclosing lambda's `captured_<fv>` slot (which holds the cell address
     * as a tagged int64), or the symbol table when the cell is a plain alloca /
     * global / pointer instruction of the active function. Returning nullptr
     * (rather than guessing) keeps the old behaviour for any shape not covered,
     * so this can never make a working program worse.
     */
    Value* codegenNamedLetEscapeClosure(const std::string& loop_name,
                                        const NamedLetEscapeInfo& info);

    // Named let: (let loop ((var init) ...) body)
    // Transforms to: (letrec ((loop (lambda (var ...) body))) (loop init ...))
    Value* codegenNamedLet(const eshkol_operations_t* op);


    // NOTE: codegenLetrec was removed (was ~480 lines of dead code).
    // Letrec is fully handled by binding_codegen.cpp:letrec() (dispatched at
    // codegenOperation). The dead code used a separate TailCallContext in
    // LLVMCodegen that did not integrate with the active TCO interception in
    // codegenCall (which checks binding_->isTCOActive()).

    Value* codegenTensor(const eshkol_ast_t* ast);

    // Emit an out-of-bounds raise for the vref/tensor-ref access paths and
    // terminate the current block with `unreachable`. The caller is expected to
    // have positioned the builder in a dedicated failure block. Mirrors the
    // eshkol_raise / eshkol_make_exception_with_header idiom used by the
    // collection and string bounds-check paths.
    void emitVrefBoundsRaise(const char* message);

    Value* codegenTensorVectorRef(const eshkol_operations_t* op);


    // matmul: (matmul A B) - Matrix multiplication [M x K] @ [K x N] -> [M x N]
    // Uses BLAS (Accelerate/OpenBLAS) for large matrices via runtime dispatch
    Value* codegenMatmul(const eshkol_operations_t* op);


    // trace: (trace matrix) - Sum of diagonal elements
    Value* codegenTrace(const eshkol_operations_t* op);

    // det: Now implemented in lib/math.esk using pure Eshkol
    // The compiler-level implementation was removed due to complexity.

    // norm: (norm vector) - Euclidean norm (L2 norm)
    Value* codegenNorm(const eshkol_operations_t* op);

    // outer: (outer v1 v2) - Outer product of two vectors
    Value* codegenOuterProduct(const eshkol_operations_t* op);

    // Symbolic differentiation function
    // Returns S-expression (list) representing symbolic derivative formula
    Value* codegenDiff(const eshkol_operations_t* op);

    // ===== PHASE 0: AUTODIFF TYPE-AWARE HELPERS =====

    // Helper: Detect if an expression evaluates to double type
    bool isDoubleExpression(const eshkol_ast_t* expr);

    // Helper: Type-aware multiplication for derivatives
    Value* createTypedMul(Value* a, Value* b, const eshkol_ast_t* reference_expr);

    // Helper: Type-aware addition for derivatives
    Value* createTypedAdd(Value* a, Value* b, const eshkol_ast_t* reference_expr);

    // Helper: Type-aware subtraction for derivatives
    Value* createTypedSub(Value* a, Value* b, const eshkol_ast_t* reference_expr);

    // Helper: Type-aware division for derivatives
    Value* createTypedDiv(Value* a, Value* b, const eshkol_ast_t* reference_expr);
    // ===== SYMBOLIC DIFFERENTIATION HELPER FUNCTIONS =====
    // AST-based symbolic derivative builder (compile-time transformation)

    // Helper: Check if AST is a constant (number)
    bool isConstant(const eshkol_ast_t* ast);

    // Helper: Check if AST is specific variable
    bool isVariable(const eshkol_ast_t* ast, const char* var_name);

    // Helper: Check if constant equals specific value
    bool isConstantValue(const eshkol_ast_t* ast, double value);

    // Helper: Check if constant equals 0
    bool isConstantZero(const eshkol_ast_t* ast);

    // Helper: Check if constant equals 1
    bool isConstantOne(const eshkol_ast_t* ast);

    // Helper: Get numeric value from constant AST node
    double getConstantValue(const eshkol_ast_t* ast);

    // Helper: Check if two ASTs are structurally identical
    bool astEqual(const eshkol_ast_t* a, const eshkol_ast_t* b);

    // Helper: Check if AST is a call to a specific function
    bool isCallTo(const eshkol_ast_t* ast, const char* func_name);

    // Helper: Check if AST contains the given variable
    bool containsVariable(const eshkol_ast_t* ast, const char* var);

    // ===== EXPRESSION SIMPLIFICATION ENGINE =====
    // Post-differentiation simplification: algebraic identities, constant folding,
    // log/exp cancellation, term collection. Called after each differentiation step.

    eshkol_ast_t* simplifySymbolicAST(eshkol_ast_t* ast);

    // ===== HIGHER-ORDER SYMBOLIC DERIVATIVES =====
    // Apply differentiation n times with simplification after each step

    // Core symbolic differentiation function (AST → AST transformation)
    eshkol_ast_t* buildSymbolicDerivative(const eshkol_ast_t* expr, const char* var);

    // Differentiate operations (symbolic, AST-based)
    eshkol_ast_t* differentiateOperationSymbolic(const eshkol_operations_t* op, const char* var);

    // Convert AST to runtime S-expression (quoted list). Identifiers carry
    // their source spelling: this is also the source form of a procedure,
    // whose binders the expander renamed (syntax_color.h).
    Value* codegenQuotedAST(const eshkol_ast_t* ast);

    // Handle operation AST nodes for S-expression generation
    Value* codegenQuotedOperation(const eshkol_operations_t* op);

    // Helper to build (op arg1 arg2 ...) for n-ary operations
    Value* codegenQuotedNaryOp(const char* op_name, const eshkol_ast_t* args, uint64_t num_args);

    // True when `op` is exactly the `n/d` -> `(make-rational n d)` desugar
    // the parser synthesizes for a rational-literal token (parser.cpp,
    // parse_atom's TOKEN_NUMBER '/' branch): a call to "make-rational" with
    // precisely two operands, each an int64 or bignum-magnitude LITERAL.
    //
    // This is deliberately the same whitelist autodiff_codegen.cpp's
    // towerSafeExpr uses to accept a rational literal into the exact AD
    // tier, and for the same reason: `(make-rational n d)` written directly
    // by a user is indistinguishable in the AST from the literal's desugar,
    // so only the shape a literal can actually produce is treated as a
    // literal. `(make-rational x 3)` (a variable operand) is a genuine call
    // and must still be quoted as list data.
    static bool isQuotedRationalLiteralDesugar(const eshkol_operations_t* op);

    // True when `op` is exactly the complex-literal desugar the parser
    // synthesizes (parser.cpp, parse_atom's TOKEN_NUMBER branch): a call to
    // "make-rectangular" with two DOUBLE literal operands. A complex literal's
    // parts are always inexact, so a variable or exact operand is a genuine
    // call and stays list data under quote.
    static bool isQuotedComplexLiteralDesugar(const eshkol_operations_t* op);

    // Construct the actual rational VALUE for a quoted `n/d` literal
    // (isQuotedRationalLiteralDesugar already verified the shape). Routes
    // through the same eshkol_rational_make_tagged runtime entry point the
    // evaluated `(make-rational n d)` call uses (see the
    // func_name == "make-rational" arm above), so a quoted bignum-rational
    // literal and its evaluated twin are the same runtime value (SW-163).
    Value* codegenQuotedRationalLiteral(const eshkol_operations_t* op);

    // Build runtime S-expression list from call operation
    Value* codegenQuotedList(const eshkol_operations_t* op);

    // ===== LAMBDA S-EXPRESSION HOMOICONIC DISPLAY =====
    // Convert lambda AST to runtime S-expression for code-as-data display

    // Helper: Build parameter list as cons chain: (param1 param2 ...)
    Value* buildParameterList(const eshkol_ast_t* params, uint64_t num_params);

    // Convert lambda or function definition AST to runtime S-expression for homoiconic display
    // Returns cons list pointer (int64): (lambda (param1 param2 ...) body)
    // Handles both LAMBDA_OP and DEFINE_OP (for named function definitions)
    Value* codegenLambdaToSExpr(const eshkol_operations_t* op);

    // ===== END LAMBDA S-EXPRESSION HOMOICONIC DISPLAY =====

    // ===== DUAL NUMBER LLVM IR HELPER FUNCTIONS =====

    // MIGRATED: Pack value and derivative into dual number struct - delegates to AutodiffCodegen
    Value* packDualNumber(Value* value, Value* derivative);

    // MIGRATED: Unpack dual number into value and derivative components - uses AutodiffCodegen
    std::pair<Value*, Value*> unpackDualNumber(Value* dual);

    // MIGRATED: Pack dual number into tagged value for storage - delegates to AutodiffCodegen
    Value* packDualToTaggedValue(Value* dual);

    // MIGRATED: Unpack dual number from tagged value - delegates to AutodiffCodegen
    Value* unpackDualFromTaggedValue(Value* tagged);

    // ===== END DUAL NUMBER HELPERS =====

    // ===== PHASE 2: DUAL NUMBER ARITHMETIC OPERATIONS =====
    // MIGRATED: These all delegate to AutodiffCodegen

    // MIGRATED: Addition: (a, a') + (b, b') = (a+b, a'+b')
    Value* dualAdd(Value* dual_a, Value* dual_b);

    // MIGRATED: Subtraction: (a, a') - (b, b') = (a-b, a'-b')
    Value* dualSub(Value* dual_a, Value* dual_b);

    // MIGRATED: Multiplication: (a, a') * (b, b') = (a*b, a'*b + a*b')
    Value* dualMul(Value* dual_a, Value* dual_b);

    // MIGRATED: Division: (a, a') / (b, b') = (a/b, (a'*b - a*b')/b²)
    Value* dualDiv(Value* dual_a, Value* dual_b);

    // MIGRATED: Dual number math operations - now delegate to AutodiffCodegen

    // Sine: sin(a, a') = (sin(a), a' * cos(a))
    Value* dualSin(Value* dual_a);

    // Cosine: cos(a, a') = (cos(a), -a' * sin(a))
    Value* dualCos(Value* dual_a);

    // Exponential: exp(a, a') = (exp(a), a' * exp(a))
    Value* dualExp(Value* dual_a);

    // Logarithm: log(a, a') = (log(a), a'/a)
    Value* dualLog(Value* dual_a);

    // Tangent: tan(a, a') = (tan(a), a' * sec²(a))
    Value* dualTan(Value* dual_a);

    // Hyperbolic sine: sinh(a, a') = (sinh(a), a' * cosh(a))
    Value* dualSinh(Value* dual_a);

    // Hyperbolic cosine: cosh(a, a') = (cosh(a), a' * sinh(a))
    Value* dualCosh(Value* dual_a);

    // Hyperbolic tangent: tanh(a, a') = (tanh(a), a' * sech²(a))
    Value* dualTanh(Value* dual_a);

    // Absolute value: abs(a, a') = (|a|, a' * sign(a))
    Value* dualAbs(Value* dual_a);

    // Square root: sqrt(a, a') = (sqrt(a), a' / (2 * sqrt(a)))
    Value* dualSqrt(Value* dual_a);

    // Power: (a, a')^(b, b') = (a^b, a^b * (b' * log(a) + b * a'/a))
    Value* dualPow(Value* dual_a, Value* dual_b);

    // Negation: -(a, a') = (-a, -a')
    Value* dualNeg(Value* dual_a);

    // Arc sine: asin(a, a') = (asin(a), a' / sqrt(1 - a²))
    Value* dualAsin(Value* dual_a);

    // Arc cosine: acos(a, a') = (acos(a), -a' / sqrt(1 - a²))
    Value* dualAcos(Value* dual_a);

    // Arc tangent: atan(a, a') = (atan(a), a' / (1 + a²))
    Value* dualAtan(Value* dual_a);

    // Inverse hyperbolic sine: asinh(a, a') = (asinh(a), a' / sqrt(1 + a²))
    Value* dualAsinh(Value* dual_a);

    // Inverse hyperbolic cosine: acosh(a, a') = (acosh(a), a' / sqrt(a² - 1))
    Value* dualAcosh(Value* dual_a);

    // Inverse hyperbolic tangent: atanh(a, a') = (atanh(a), a' / (1 - a²))
    Value* dualAtanh(Value* dual_a);

    // Base-10 logarithm: log10(a, a') = (log10(a), a' / (a * ln(10)))
    Value* dualLog10(Value* dual_a);

    // Base-2 logarithm: log2(a, a') = (log2(a), a' / (a * ln(2)))
    Value* dualLog2(Value* dual_a);

    // Base-2 exponential: exp2(a, a') = (2^a, a' * 2^a * ln(2))
    Value* dualExp2(Value* dual_a);

    // Cube root: cbrt(a, a') = (cbrt(a), a' / (3 * cbrt(a)²))
    Value* dualCbrt(Value* dual_a);

    // ===== END DUAL NUMBER ARITHMETIC =====

    // ===== NESTED GRADIENT SUPPORT: TAPE STACK OPERATIONS =====
    // MIGRATED: These delegate to AutodiffCodegen

    // Push current tape context onto stack and activate new tape
    void pushTapeContext(Value* new_tape);

    // Pop tape context from stack, restoring previous tape
    void popTapeContext();

    // ===== DOUBLE BACKWARD HELPER FUNCTIONS =====
    // MIGRATED: These delegate to AutodiffCodegen

    // Get the outer tape (from stack[depth-1])
    Value* getOuterTape();

    // Check if currently nested (tape_depth > 0)
    Value* isNested();

    // MIGRATED: Create AD constant node on a specific tape - delegates to AutodiffCodegen
    Value* createADConstantOnTape(Value* tape_ptr, Value* value);

    // MIGRATED: Record binary operation on a specific tape - delegates to AutodiffCodegen
    Value* recordADNodeBinaryOnTape(Value* tape_ptr, uint32_t op_type, Value* left_node, Value* right_node);

    // ===== PHASE 3: AD NODE HELPER FUNCTIONS =====
    // Computational graph construction for reverse-mode automatic differentiation

    // MIGRATED: Create AD node for a constant value - delegates to AutodiffCodegen
    Value* createADConstant(Value* value);

    // MIGRATED: Create AD variable node - delegates to AutodiffCodegen
    Value* createADVariable(Value* value, size_t var_index);

    // MIGRATED: Record binary operation node - delegates to AutodiffCodegen
    Value* recordADNodeBinary(uint32_t op_type, Value* left_node, Value* right_node);

    // MIGRATED: Record unary operation node - delegates to AutodiffCodegen
    Value* recordADNodeUnary(uint32_t op_type, Value* input_node);

    // MIGRATED: AD node helpers delegate to AutodiffCodegen
    Value* loadNodeValue(Value* node_ptr);

    Value* loadNodeGradient(Value* node_ptr);

    void storeNodeGradient(Value* node_ptr, Value* gradient);

    void accumulateGradient(Value* node_ptr, Value* gradient_to_add);

    // MIGRATED: Load input node pointers - delegates to AutodiffCodegen
    Value* loadNodeInput1(Value* node_ptr);

    Value* loadNodeInput2(Value* node_ptr);

    // ===== END AD NODE HELPERS =====
    // ===== PHASE 3: BACKWARD PASS IMPLEMENTATION =====
    // Backpropagation through computational graph (delegated to AutodiffCodegen)

    // ===== END BACKWARD PASS =====


    // ===== PHASE 2: DERIVATIVE OPERATOR IMPLEMENTATION =====
    // Runtime derivative computation using dual numbers

    // Helper function to load captures for an autodiff function call
    // Returns a vector of captured values that should be appended to call arguments

    // ===== CALCULUS OPERATORS — All in AutodiffCodegen =====

    // ===== OALR (Ownership-Aware Lexical Regions) CODEGEN =====

    Value* codegenWithRegion(const eshkol_operations_t* op);

    // Codegen for (owned expr) - marks a value as owned (linear type)
    // Sets ESHKOL_OBJ_FLAG_LINEAR on the object header so the value must be consumed exactly once
    Value* codegenOwned(const eshkol_operations_t* op);

    // Apply the ownership marker to a value that has already been evaluated.
    Value* codegenOwnedValue(Value* value);

    // Codegen for (move value) - transfers ownership
    // Sets CONSUMED flag on source object and nulls out the source binding
    Value* codegenMove(const eshkol_operations_t* op);

    // Apply the move effects to an already evaluated tagged value.
    Value* codegenMoveValue(const eshkol_operations_t* op, Value* value);

    // Codegen for (borrow value body ...) - temporary read-only access with scope guard
    // Sets BORROWED flag before body, clears it after body completes
    Value* codegenBorrow(const eshkol_operations_t* op);

    // Codegen for (shared expr) - creates a reference-counted value
    // Sets SHARED flag and initializes ref_count to 1 in the object header
    Value* codegenShared(const eshkol_operations_t* op);

    // Apply the shared marker to a value that has already been evaluated.
    Value* codegenSharedValue(Value* value);

    // Codegen for (weak-ref value) - creates a weak reference to a shared value
    // Sets WEAK flag on the object header without incrementing ref_count
    Value* codegenWeakRef(const eshkol_operations_t* op);

    // ===== END OALR CODEGEN =====


    // Core symbolic differentiation function
    // Now works within lambda context - variable comes from lambda parameter
    Value* differentiate(const eshkol_ast_t* expr, const char* var);

    // Differentiate operations (arithmetic, functions, etc.)
    Value* differentiateOperation(const eshkol_operations_t* op, const char* var);

    Value* codegenVectorToString(const eshkol_operations_t* op);

    Value* codegenMatrixToString(const eshkol_operations_t* op);

    // Production implementation: Compound car/cdr operations using TAGGED cons cells
    Value* codegenCompoundCarCdr(const eshkol_operations_t* op, const std::string& pattern);

    // Production implementation: List length
    // codegenLength removed - now in stdlib.esk (core/list/query.esk)

    // Random number generation: (random) returns double in [0.0, 1.0)
    Value* codegenRandom(const eshkol_operations_t* op);

    // ─── PRNG: deterministic-replay seed + isolated per-task generators ───
    //
    // (set-random-seed! N)        ─ explicit seed for the global PRNG
    // (make-prng N)               ─ allocate isolated PRNG state (lock-free)
    // (prng? x)                   ─ heap-subtype check for HEAP_SUBTYPE_PRNG
    // (prng-random p)             ─ next double in [0.0, 1.0) on the given PRNG
    // (prng-random-integer p n)   ─ next int64 in [0, n) on the given PRNG
    //
    // The runtime functions live in lib/core/prng.cpp. The global PRNG path
    // remains the existing drand48-with-mutex; the per-PRNG path is mutex-
    // free because each state is independent.

    Value* codegenSetRandomSeed(const eshkol_operations_t* op);

    Value* codegenMakePrng(const eshkol_operations_t* op);

    Value* codegenPrngP(const eshkol_operations_t* op);

    Value* codegenPrngRandom(const eshkol_operations_t* op);

    Value* codegenPrngRandomInteger(const eshkol_operations_t* op);

    // Quantum random number generation: (quantum-random) returns double in [0.0, 1.0)
    // Uses quantum-inspired RNG for higher quality randomness
    Value* codegenQuantumRandom(const eshkol_operations_t* op);

    /**
     * Lower (vqe-energy-primitive ham params) to the Moonlab energy bridge.
     *
     * The AD branch receives the parameter tensor produced by gradient(): its
     * element slots hold ad_node_t* bit patterns. The C preparation helper
     * snapshots those primal values, creates an arena-owned exact Moonlab VJP,
     * and returns its descriptor plus the input-node array. recordADNodeCustom
     * then appends the opaque scalar to the active Eshkol tape.
     */
    Value* codegenVqeEnergyPrimitive(const eshkol_operations_t* op);

    // Quantum random integer: (quantum-random-int bound) -> integer in [0, bound).
    // Mirrors the VM path (vm_native.c case 1861): bound <= 1 yields 0, otherwise
    // the unsigned remainder of a raw 64-bit draw by bound. Previously this ignored
    // its argument and returned the raw uint64, diverging from the VM and violating
    // the documented [0, bound) contract.
    Value* codegenQuantumRandomInt(const eshkol_operations_t* op);

    // Quantum random range: (quantum-random-range min max) returns int in [min, max]
    Value* codegenQuantumRandomRange(const eshkol_operations_t* op);

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPLEX NUMBER OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    // Helper: Create complex number from real and imaginary parts
    Value* createComplexNumber(Value* real_val, Value* imag_val);

    // Helper: Pack complex struct to tagged value (heap allocate)
    Value* packComplexToTagged(Value* complex_struct);

    // Helper: Unpack complex struct from tagged value
    Value* unpackComplexFromTagged(Value* tagged_val);

    // Helper: Extract real part from complex struct
    Value* getComplexReal(Value* complex_struct);

    // Helper: Extract imaginary part from complex struct
    Value* getComplexImag(Value* complex_struct);

    // ── Native 128-bit integer (i128) builtins ──────────────────────────────
    //
    // i128 is a distinct fixed-width, wrapping type that lives OFF the numeric
    // tower. Every builtin lowers to a runtime call in lib/core/i128_runtime.cpp
    // (which delegates the actual arithmetic to the pure header core shared with
    // the VM). Following the bignum tagged-in/tagged-out dispatch pattern, all
    // args and the result are marshalled through entry-block allocas so the call
    // never grows the stack inside a loop (preserving TCO), mirroring
    // ArithmeticCodegen::emitBignumBinaryCall.
    //
    // Correctness first: this PR intentionally boxes on the arena and does NOT
    // attempt an unboxed LLVM-i128 hot path (noted as future work in the docs).
    Value* codegenI128Runtime(const eshkol_operations_t* op, const char* fn_name,
                              int expected_args, bool needs_arena,
                              bool has_op, int op_code);

    // (make-rectangular real imag) - Create complex from rectangular coordinates
    // === Complex values that carry a derivative (ADR-0025) ===
    //
    // Every complex builtin reaches its carrier rule here and nowhere else. A
    // carrier complex is read through its two tagged components and the result
    // is the textbook component formula evaluated by the generic arithmetic
    // and the value-level math functions, so a jet, a tape node or a Taylor
    // tower flows through with the rule it already has. A REAL carrier is a
    // complex number with no imaginary part, and gets the same treatment
    // instead of being flattened to its primal.
    enum ComplexUnaryKind { CPX_REAL_PART, CPX_IMAG_PART, CPX_MAGNITUDE, CPX_ANGLE, CPX_CONJUGATE };

    // Raises unless the complex value is away from the origin, where neither
    // the magnitude nor the angle has a derivative. Never answers 0 there.
    void guardComplexAwayFromOrigin(Value* re, Value* im, const char* who);

    // 0 - v through the binary operator, which has a rule for every carrier;
    // the unary negation has one for tape nodes only.
    Value* carrierNegate(Value* v);

    Value* carrierMagnitude(Value* re, Value* im);

    Value* carrierAngle(Value* re, Value* im);

    Value* complexUnaryWithCarriers(Value* z_tagged, ComplexUnaryKind kind,
                                    const std::function<Value*()>& plain);

    // A complex number as its two real components, each any real number: a
    // plain number or a derivative carrier. The carrier formulas below are
    // written on these pairs with the real arithmetic and the value-level math
    // functions, which have a rule for every carrier, and mirror the plain
    // kernels in <eshkol/core/complex_math.h> formula for formula.
    struct CPair { Value* re; Value* im; };
    Value* cpxDouble(double v);
    Value* cpxExact(int64_t v);
    CPair cpxOf(Value* z);
    Value* cpxPack(const CPair& p);
    Value* cpxReal(Value* v, const char* name);
    CPair cpxAdd(const CPair& a, const CPair& b);
    CPair cpxSub(const CPair& a, const CPair& b);
    CPair cpxMul(const CPair& a, const CPair& b);
    CPair cpxDiv(const CPair& a, const CPair& b);
    CPair cpxScale(const CPair& a, Value* k);
    CPair cpxMulI(const CPair& a);   // i a
    CPair cpxDivI(const CPair& a);   // a / i
    CPair cpxConst(int64_t re);
    CPair cpxExp(const CPair& a);
    CPair cpxLog(const CPair& a);
    CPair cpxSqrt(const CPair& a);
    CPair cpxSin(const CPair& a);
    CPair cpxCos(const CPair& a);
    CPair cpxSinh(const CPair& a);
    CPair cpxCosh(const CPair& a);
    CPair cpxAsin(const CPair& z);

    // Every complex procedure with a plain complex kernel, on a carrier complex,
    // by its component formula (ADR-0025). nullptr for a procedure with no
    // formula here; the caller raises.
    Value* complexCarrierMath(Value* z_tagged, const std::string& func_name);

    // (expt a b) when a derivative is in play and either operand is complex
    // (SW-211). The plain kernel eshkol_complex_pow works on the primals and
    // drops the derivative. An exact integer exponent is repeated complex
    // multiplication (binary exponentiation, a reciprocal for a negative one),
    // which is exact where the components are and defined at a zero base;
    // any other exponent is exp(b log a), as eshkol_cpx_pow computes it.
    Value* complexCarrierPow(Value* base, Value* expo);

    Value* codegenMakeRectangular(const eshkol_operations_t* op);

    // (make-polar magnitude angle) - Create complex from polar coordinates
    Value* codegenMakePolar(const eshkol_operations_t* op);

    // (real-part z) - Extract real component
    Value* codegenRealPart(const eshkol_operations_t* op);

    // (imag-part z) - Extract imaginary component
    Value* codegenImagPart(const eshkol_operations_t* op);

    // (magnitude z) - |z| = sqrt(real² + imag²)
    Value* codegenMagnitude(const eshkol_operations_t* op);

    // (angle z) - arg(z) = atan2(imag, real)
    Value* codegenAngle(const eshkol_operations_t* op);

    // (complex? x) - Type predicate
    Value* codegenComplexPredicate(const eshkol_operations_t* op);

    // (conjugate z) - Complex conjugate: conj(a+bi) = a-bi
    Value* codegenConjugate(const eshkol_operations_t* op);

    // ═══════════════════════════════════════════════════════════════════════════
    // HoTT SUM TYPE OPERATIONS (Discriminated Unions)
    // Sum types are represented as cons pairs: (tag . value)
    // where tag is 0 (left) or 1 (right).
    // ═══════════════════════════════════════════════════════════════════════════

    // (inject-left value) or (inject-right value) — construct sum type variant
    Value* codegenSumInject(const eshkol_operations_t* op, int tag);

    // (sum-tag sum-val) — extract tag from sum type (0 = left, 1 = right)
    Value* codegenSumTag(const eshkol_operations_t* op);

    // (sum-value sum-val) — extract inner value from sum type
    Value* codegenSumValue(const eshkol_operations_t* op);

    // (left? sum-val) or (right? sum-val) — check sum variant
    Value* codegenSumPredicate(const eshkol_operations_t* op, int expected_tag);

    // ═══════════════════════════════════════════════════════════════════════════
    // FFT/IFFT Implementation (Cooley-Tukey Radix-2 DIT)
    // ═══════════════════════════════════════════════════════════════════════════

    // Helper: Check if n is a power of 2
    Value* isPowerOfTwo(Value* n);

    // Helper: Bit-reverse an index for FFT
    Value* bitReverse(Value* idx, Value* log2n);

    // Helper: Compute log2(n) for power-of-2 n
    Value* computeLog2(Value* n);

    // (fft vec) or (ifft vec) - Cooley-Tukey radix-2 DIT FFT
    // Input: vector of real or complex numbers (length must be power of 2)
    // Output: vector of complex numbers
    Value* codegenFFT(const eshkol_operations_t* op, bool inverse);

    // codegenRange removed - now in stdlib.esk (core/list/generate.esk)
    // codegenZip removed - now in stdlib.esk (core/list/generate.esk)

    // sort is now implemented in stdlib.esk




    // Reduce/Fold: Left fold with explicit initial value
    // (reduce f init list) => fold-left style
    // Also supports (reduce f list) which uses first element as initial value
    Value* codegenReduce(const eshkol_operations_t* op);

    // Boolean predicate: checks if value is #t or #f
    Value* codegenBooleanPredicate(const eshkol_operations_t* op);

    // List predicate: checks if value is a proper list (null-terminated chain of cons cells)
    // A proper list is either:
    //   - null (empty list)
    //   - a cons cell whose cdr is also a proper list
    Value* codegenListPredicate(const eshkol_operations_t* op);

    // Procedure predicate: checks if value is a function
    Value* codegenProcedurePredicate(const eshkol_operations_t* op);

    // codegenAppend removed - now in stdlib.esk (core/list/transform.esk)

    // codegenIterativeAppend removed - now in stdlib.esk (core/list/transform.esk)

    // codegenReverse removed - now in stdlib.esk (core/list/transform.esk)

    // codegenListRef removed - now in stdlib.esk (core/list/search.esk)
    // codegenListTail removed - now in stdlib.esk (core/list/search.esk)

    // ─── Procedure reflection ──────────────────────────────────────────
    //
    //   (procedure-arity proc)      → fixed param count (int), or 0 if not a
    //                                 closure. For variadic procedures this
    //                                 is the minimum-arity (the leading fixed
    //                                 parameters); use (procedure-variadic? p)
    //                                 to distinguish.
    //   (procedure-name proc)       → the bound name from `(define name …)`
    //                                 or "" for anonymous lambdas.
    //   (procedure-variadic? proc)  → #t if the procedure accepts a rest
    //                                 argument, else #f.
    //
    // Each takes a CALLABLE tagged value, unpacks the pointer, and reads
    // from the closure struct (eshkol_closure_t). For non-closures (e.g.
    // builtins exposed as CALLABLE_SUBTYPE_PRIMITIVE without a closure
    // struct), we currently return 0/""/#f — a future refinement could
    // consult a builtin arity table.

    Value* codegenProcedureArity(const eshkol_operations_t* op);

    // Production implementation: Set car (mutable)
    // Shared implementation: mutate car (is_cdr=0) or cdr (is_cdr=1) of a
    // pair. Previously the typed setters (SetInt64 / SetDouble / SetPtr)
    // forced us to compress every replacement value into a single
    // {value,type-tag} slot, which lost the type byte whenever the new
    // value was already a full tagged_value struct (e.g. the result of
    // `(list ...)` or `(cons ...)`). detectValueType was then returning
    // INT64 for the struct's payload, so the setter wrote the raw heap
    // address with an INT64 tag — and all subsequent cdr walks saw an
    // integer instead of a pair (Noesis Bug E, 2026-04-19).
    //
    // Fix: when the replacement is already a tagged_value struct, spill
    // it to a stack slot and call arena_tagged_cons_set_tagged_value,
    // which copies the full 16-byte struct (including type byte) into
    // the cons cell. This is the same path used by cons / allocConsCell,
    // so behaviour is now uniform across cons, set-car!, and set-cdr!.
    Value* codegenSetPairField(const eshkol_operations_t* op, bool is_cdr_bit);

    Value* codegenSetCar(const eshkol_operations_t* op);

    Value* codegenSetCdr(const eshkol_operations_t* op);

    // Create a wrapper function for indirect calls through function parameters
    // This enables higher-order functions where functions are passed as arguments
    // The wrapper takes the function pointer as its first argument (captured from outer scope)
    // followed by the actual arguments for the function call
    Value* codegenIndirectFunctionCall(Argument* func_arg, size_t arity);

    // Production implementation: Map function
    // REFACTORED: Delegates to MapCodegen module
    Value* codegenMap(const eshkol_operations_t* op);


    // Helper function to resolve lambda/function from AST with arity-specific builtin handling
    /**
     * @brief True when @p name denotes a runtime binding of the current
     *        function that no static alias describes (ESH-0070 class).
     *
     * A function parameter or local storage cell shadows any same-named
     * top-level function, and any scoped alias an earlier binding of the name
     * left in this function. Static procedure resolution MUST decline when
     * this returns true; the only resolution that agrees with lexical scope is
     * runtime dispatch on the local value. The exception is a scoped alias
     * recorded by the binding the name denotes now: that binding is a
     * never-reassigned lambda and static resolution is exact. The rule lives
     * in static_callee_binding.h so every static fast path applies it.
     */
    bool isShadowedByLocalRuntimeBinding(const std::string& name);

    Value* resolveLambdaFunction(const eshkol_ast_t* func_ast, size_t required_arity = 0);

    // Production implementation: Apply function
    // (apply fn args-list) - calls fn with arguments from args-list
    // REFACTORED: Delegates to CallApplyCodegen module
    Value* codegenApply(const eshkol_operations_t* op);

    // Recognize and lower a statically-spelled apply-self-call in tail position
    // to a TCO loop back-edge (ESH-0227). Returns nullptr (no interception) for
    // anything that is not provably `(apply <this-loop> leading... (list ...))`
    // with a total argument count equal to the loop's arity, so the caller
    // falls back to the ordinary apply lowering.
    Value* tryApplyTailCall(const eshkol_operations_t* op);

    /* Bug P (2026-04-23): apply on a cross-file user-defined function
     * (REPL forward-reference). Direct calls already handle this via
     * the `__repl_fwd_<name>` indirect-call shape; apply needs the
     * same. Returns nullptr if `func_name` is not a known REPL
     * forward-ref, so the caller's existing "Unknown function"
     * fallback still fires for genuine unknowns.
     *
     * This mirrors the direct-call indirect-emit at ~line 13520+ but
     * sources arguments from a runtime cons list (`list_int`) instead
     * of compile-time AST nodes. Variadic functions are handled via
     * g_repl_variadic_functions: the first fixed_params elements are
     * extracted positionally from the list, the remaining tail is
     * passed as the rest argument unchanged. */
    Value* emitApplyForwardRef(const std::string& func_name, Value* list_int);

    // codegenAssoc removed - now in stdlib.esk (core/list/search.esk)

    // Production implementation: List* (improper list constructor)
    Value* codegenListStar(const eshkol_operations_t* op);

    // Production implementation: Acons (association constructor)
    Value* codegenAcons(const eshkol_operations_t* op);

    // codegenTake removed - now in stdlib.esk (core/list/query.esk)
    // codegenDrop removed - now in stdlib.esk (core/list/query.esk)
    // codegenFind removed - now in stdlib.esk (core/list/query.esk)


    // Production implementation: Split-at function (split list at index)
    Value* codegenSplitAt(const eshkol_operations_t* op);

    // Production implementation: Remove function family (remove elements that match)
    // Supports both element-based removal: (remove 2 '(1 2 3)) => (1 3)
    // and predicate-based removal: (remove even? '(1 2 3 4)) => (1 3)
    Value* codegenRemove(const eshkol_operations_t* op, const std::string& comparison_type);

    // Production implementation: Last function (return last element)
    Value* codegenLast(const eshkol_operations_t* op);

    // Production implementation: Last-pair function (return last cons cell)
    Value* codegenLastPair(const eshkol_operations_t* op);

    // Production implementation: Create arity-specific builtin arithmetic functions (POLYMORPHIC)
    Function* createBuiltinArithmeticFunction(const std::string& operation, size_t arity);

    // Create builtin comparison function for use as first-class value (sort, etc.)
    // Comparison operators take 2 tagged values and return a tagged boolean
    Function* createBuiltinComparisonFunction(const std::string& operation);

    // Create a (tagged_value, tagged_value) -> tagged_value wrapper for an
    // identity/equality predicate (eq?, eqv?, equal?) so it can be used as a
    // first-class value or via apply. These are normally codegen-inline
    // (codegenEq/Eqv/Equal); without a wrapper, `(car (list eq?))` and
    // `(apply eq? …)` had no callable closure. The body delegates to the same
    // emitEq*Tagged helpers the inline path uses, so semantics stay identical.
    Function* createBuiltinEqualityFunction(const std::string& pred_name);

    // Create wrapper function for builtin predicates (even?, odd?, zero?, etc.)
    Function* createBuiltinPredicateFunction(const std::string& pred_name);

    // Create a tagged_value -> tagged_value wrapper for an ASCII character
    // case-conversion builtin (char-upcase / char-downcase / char-foldcase)
    // so it can be used as a first-class procedure. The body mirrors the
    // inline codegen in codegenAST's char-upcase/char-downcase dispatch.
    Function* createBuiltinCharFunction(const std::string& op_name);

    // Create a closure-ABI wrapper around a C runtime sret builtin.
    //
    // Many faculty-level builtins (ad-mul, ad-add, ad-sin, hash-table-ref, …)
    // are exposed to Scheme only at the call site — `codegenCall` recognises
    // their name and dispatches directly. That made them fail to evaluate
    // when passed as a value: `(define op ad-mul)` or
    // `(apply-op ad-mul a b)` would hit "Undefined variable" because
    // codegenVariable had no synthesised closure for them.
    //
    // This helper produces an LLVM function with the closure ABI
    //   tagged_value(tagged_value…)    (by-value args, by-value return)
    // that internally calls the sret-style C helper
    //   void c_name(tagged_value* out, const tagged_value* a, …)
    // — the same sret functions `codegenCall` already emits for call-site
    // dispatch. Wrapping it into a closure makes the builtin a first-class
    // value without duplicating per-builtin logic.
    //
    // Returns nullptr if the builtin name isn't known. Result is cached so
    // repeated (define x ad-mul) uses share one LLVM function.
    Function* createBuiltinSretWrapper(const std::string& c_name,
                                       size_t arity);

    // Map from Scheme name → (C sret symbol, arity) for faculty-level
    // builtins that aren't yet first-class. Expand this table as we
    // extend coverage. Hot-path builtins (math ops, comparisons,
    // arithmetic) go through their dedicated factories above.
    const std::pair<std::string, size_t>*
    lookupSretBuiltin(const std::string& var_name) const;

    /* ------------------------------------------------------------------
     * FIRST-CLASS REFERENCES TO CALL-POSITION-ONLY BUILTINS  (LE-01)
     * ------------------------------------------------------------------
     *
     * Most of Eshkol's builtin surface is lowered INLINE at the call site:
     * `codegenCall` string-matches the head name and emits the operation
     * directly (`string<?` -> StringIOCodegen::stringCompare, `vector-ref`
     * -> the tensor/vector path, `expt`/`min`/`max` -> ArithmeticCodegen,
     * and so on). That is the right lowering for `(string<? a b)`, but it
     * means the NAME by itself never denotes a value: there is no LLVM
     * function to point a closure at.
     *
     * Before this, each builtin that someone needed as a value got its own
     * hand-written factory (createBuiltinComparisonFunction,
     * createBuiltinPredicateFunction, createBuiltinCharFunction, …), each
     * re-implementing the operation's semantics a SECOND time in IR. Two
     * consequences, both observed:
     *
     *   1. Coverage was whatever had been asked for. Everything else fell
     *      off the end of codegenVariable into "Undefined variable: <name>"
     *      — or, worse, into the raw-Function* fallback below, which hands
     *      the closure dispatcher a pointer to a function with a FOREIGN
     *      ABI. Calling that is undefined behaviour: `(h2 remainder 7 3)`
     *      SIGSEGV'd and `(h2 append '(1) '(2))` returned `1`.
     *   2. The second implementation could drift from the call-site one
     *      with nothing to catch it.
     *
     * The fix generalises the wrapper-closure IDIOM but not the duplicated
     * bodies: we synthesise `builtin_fc_<name>_a<arity>` with the closure ABI
     * (tagged_value…)->tagged_value whose body is generated by re-entering
     * `codegenCall` on a synthetic `(name p0 … pN-1)` AST whose arguments
     * are the wrapper's own parameters. The authoritative call-site lowering
     * IS the wrapper body, so a first-class reference cannot disagree with a
     * direct call, and adding a builtin to the first-class surface is one
     * row of data rather than a new IR implementation.
     *
     * Placement: this is consulted only where the old code would otherwise
     * have produced a garbage callable or an "Undefined variable" error, so
     * it never changes which binding a name resolves to. Shadowing order —
     * user definitions, locals, captures, REPL namespaces — is decided
     * strictly before we get here and is untouched.
     */
    /* How a VARIADIC builtin computes its result from the REST LIST.
     *
     * A first-class reference to a variadic builtin is called through the
     * closure ABI, and that ABI has exactly one shape for "any number of
     * arguments": the dispatcher conses the caller's arguments into a list
     * and passes that single list (see the variadic branch of the closure
     * dispatcher, `CLOSURE_FLAG_VARIADIC`). So the wrapper body for such a
     * reference is not "call the builtin with N arguments" — it is
     * "compute the builtin's answer from this list", and each variadic
     * builtin says here how:
     *
     *   Identity   the rest list IS the answer  (`list`; `values` in this
     *              path, which yields its arguments as a list)
     *   RestUnary  a UNARY builtin is definitionally `(apply <name> lst)`
     *              (`vector` is `list->vector`, `string` is `list->string`)
     *   LeftFold   left-fold the list with the builtin's BINARY form
     *              (`string-append`, `max`, `gcd`, `vector-append`, …)
     *   ByCount    the builtin takes `arity` required arguments and up to
     *              `max_arity` in all (`atan`, `round`, and `display`/`write`
     *              with their optional port): the rest list is counted and
     *              the call-position lowering at exactly that count is
     *              called; any other count raises an arity error
     *   None       the row is variadic but has no rest form, so a value
     *              reference still materialises at the row's fixed arity —
     *              the pre-existing compromise, recorded rather than
     *              silently assumed.
     *
     * Every one of these is built from the AUTHORITATIVE call-position
     * lowering (createInlineBuiltinWrapper re-enters codegenCall), so none
     * of them is a second implementation that can drift. */
    enum class VariadicRest : uint8_t { None, Identity, RestUnary, LeftFold, ByCount, Chain, Bytevector };

    /* A row of the first-class builtin table.
     *
     * `arity` is the arity a wrapper is built at when the USE SITE names
     * none. For a `variadic` row that number is a default, never a claim
     * about the procedure: a use site that knows its own arity — `map` over
     * k lists passes k as required_arity — must get a wrapper built at THAT
     * arity, or the surplus arguments are silently dropped (SW-173). */
    struct InlineBuiltinSpec {
        size_t arity;
        bool variadic = false;
        VariadicRest rest = VariadicRest::None;
        const char* rest_unary = nullptr;   // RestUnary only
        size_t max_arity = 0;               // ByCount only: arities arity..max_arity
    };

    /* A row's `arity` is what a use site gets when it names no arity of its
     * own. It was once the WHOLE story, described here as a compromise:
     * R7RS `min`/`max`/`string-append` accept any number of arguments and a
     * value reference yielded the binary form. That compromise was not a
     * compromise, it was a silently wrong answer — `(map vector xs ys)`
     * dropped the second list, `(apply vector (list 1 2 3))` answered
     * `#(1)`, `(define f string-append) (f "a" "b" "c")` answered `"ab"`,
     * none of them with a diagnostic (SW-173). So a variadic row now SAYS
     * it is variadic, and says how it computes its answer from a rest list
     * (see VariadicRest above): a use site that knows its arity gets a
     * wrapper built at that arity, and a value reference gets a genuine
     * variadic closure. */
    const InlineBuiltinSpec* lookupInlineBuiltin(const std::string& name) const;

    // Names currently having a wrapper body generated, so a builtin whose
    // call-site lowering ends up resolving its own head as a variable cannot
    // recurse into this factory forever.
    std::set<std::string> inline_builtin_wrapper_in_progress_;
    // Stable storage for the synthetic parameter identifiers handed to the
    // AST (`variable.id` is a char*; a std::deque never invalidates).
    std::deque<std::string> inline_builtin_synthetic_names_;

    /* Symbol-safe spelling of a builtin's name. Shared by every wrapper
     * factory so one builtin can never acquire two wrapper symbols. */
    static std::string inlineBuiltinSymbolSuffix(const std::string& name);

    Function* createInlineBuiltinWrapper(const std::string& name, size_t arity);

    /* Give a call-position-only builtin an honest first-class value, or
     * nullptr if the name is not one we can wrap. */
    Value* codegenInlineBuiltinAsValue(const std::string& name);

    // SW-230: a math builtin of the math dispatch (a row or an alias of the
    // Taylor recurrence table, which the route guard keeps complete) that has
    // a first-class table row. The activations have their own value route.
    bool isFirstClassMathBuiltin(const std::string& name) const;

    /* Quirk 11: create a unary (or nullary for `newline`) closure wrapper
     * around an I/O runtime helper so the builtin can be passed as a
     * first-class value. ABI: tagged_value -> tagged_value (returns null).
     * Reusable from codegenVariable; memoised in function_table so each
     * distinct call-site shares one wrapper. */
    Function* createBuiltinIOFunction(const std::string& io_name);

    // ═════════════════════════════════════════════════════════════════════
    // NEURO-SYMBOLIC CONSCIOUSNESS ENGINE CODEGEN
    // ═════════════════════════════════════════════════════════════════════

    // Helper: get-or-declare a C runtime function
    Function* getOrDeclareRuntimeFunc(const char* name, FunctionType* fn_type);

    // Helper: load arena pointer from global
    Value* loadArenaPtr();

    // Helper: alloca + store a tagged value, return the alloca ptr
    Value* allocaAndStore(Value* tagged_val, const char* name);

    // Helper: alloca a tagged value result slot
    Value* allocaResult(const char* name);

    Value* allocaJmpBuf(const char* name);

    Function* getOrDeclareSetjmpFunc();

    std::vector<Value*> makeSetjmpArgs(Value* jmp_buf_alloc);

    // Helper: load tagged value from alloca
    Value* loadResult(Value* result_alloca, const char* name);

    // ─── Logic / KB primitives ──────────────────────────────────────────
    // codegenLogicVar / codegenUnify / codegenMakeSubst / codegenWalk
    // codegenMakeFact / codegenMakeKB / codegenKBAssert / codegenKBQuery
    // codegenKBQueryPrefix — extracted to LogicWorkspaceCodegen.

    // ─── Type Predicates ────────────────────────────────────────────────

    Value* codegenLogicVarPred(const eshkol_operations_t* op);

    Value* codegenSubstPred(const eshkol_operations_t* op);

    Value* codegenKBPred(const eshkol_operations_t* op);

    // ─── Tensor/model serialization + factor-graph + active inference ────
    // codegenTensorSave / codegenTensorLoad / codegenModelSave / codegenModelLoad
    // codegenMakeFactorGraph / codegenFGAddFactor / codegenFGInfer
    // codegenFreeEnergy / codegenEFE / codegenFGUpdateCPT / codegenFGObserve
    // — extracted to LogicWorkspaceCodegen.

    // ─── Additional Type Predicates ─────────────────────────────────────

    Value* codegenFactPred(const eshkol_operations_t* op);

    Value* codegenFactorGraphPred(const eshkol_operations_t* op);

    Value* codegenWorkspacePred(const eshkol_operations_t* op);

    // ─── Global Workspace ───────────────────────────────────────────────
    // codegenMakeWorkspace / codegenWSRegister / codegenWSStep — extracted to
    // LogicWorkspaceCodegen.
 // codegenMakeWorkspace / codegenWSRegister / codegenWSStep stubs (extracted)
};

#endif
#endif
