/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>
#include <eshkol/core/runtime.h>
#include <eshkol/core/resource_limits.h>

#include <eshkol/llvm_backend.h>
#include "../lib/repl/repl_jit.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <errno.h>
#include <filesystem>
#ifdef __APPLE__
#include <mach-o/dyld.h>  // For _NSGetExecutablePath on macOS
#endif
#ifdef __linux__
#include <linux/limits.h>  // For PATH_MAX on Linux
#endif

#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <cstdlib>

static struct option long_options[] = {
    {"help", no_argument, nullptr, 'h'},
    {"debug", no_argument, nullptr, 'd'},
    {"dump-ast", no_argument, nullptr, 'a'},
    {"dump-ir", no_argument, nullptr, 'i'},
    {"output", required_argument, nullptr, 'o'},
    {"compile-only", no_argument, nullptr, 'c'},
    {"shared-lib", no_argument, nullptr, 's'},
    {"wasm", no_argument, nullptr, 'w'},
    {"lib", required_argument, nullptr, 'l'},
    {"lib-path", required_argument, nullptr, 'L'},
    {"no-stdlib", no_argument, nullptr, 'n'},
    {"eval", required_argument, nullptr, 'e'},
    {"run", no_argument, nullptr, 'r'},
    {"strict-types", no_argument, nullptr, 256},
    {"unsafe", no_argument, nullptr, 257},
    {"debug-info", no_argument, nullptr, 'g'},
    {"optimize", required_argument, nullptr, 'O'},
    {"emit-eskb", required_argument, nullptr, 'B'},
    {0, 0, 0, 0}
};

/* Bytecode VM compiler — emit ESKB format */
extern "C" int eshkol_emit_eskb(const char* source, const char* output_path);

// Set to track imported files (prevent circular imports)
static std::set<std::string> imported_files;

// Set to track pre-compiled modules (when linking with .o files like stdlib.o)
// When a module is in this set, process_requires() will skip loading it
static std::set<std::string> precompiled_modules;

// ===== MODULE DEPENDENCY RESOLVER =====
// Provides cycle detection and topological sorting for module loading

class ModuleDependencyResolver {
public:
    // Module states for cycle detection (DFS coloring)
    enum class ModuleState { UNVISITED, VISITING, VISITED };

    // Add a module and its dependencies
    void addModule(const std::string& module_name, const std::vector<std::string>& dependencies) {
        if (dependency_graph.find(module_name) == dependency_graph.end()) {
            dependency_graph[module_name] = dependencies;
            module_states[module_name] = ModuleState::UNVISITED;
        }
    }

    // Check for cycles and return topologically sorted module order
    // Returns empty vector if cycle detected
    std::vector<std::string> resolve() {
        std::vector<std::string> sorted_modules;
        std::vector<std::string> cycle_path;

        for (const auto& [module_name, _] : dependency_graph) {
            if (module_states[module_name] == ModuleState::UNVISITED) {
                if (!dfs(module_name, sorted_modules, cycle_path)) {
                    // Cycle detected
                    std::string cycle_str = formatCycle(cycle_path);
                    eshkol_error("Circular dependency detected: %s", cycle_str.c_str());
                    return {};
                }
            }
        }

        // Reverse for topological order (dependencies first)
        std::reverse(sorted_modules.begin(), sorted_modules.end());
        return sorted_modules;
    }

    // Clear the resolver state
    void clear() {
        dependency_graph.clear();
        module_states.clear();
    }

private:
    std::map<std::string, std::vector<std::string>> dependency_graph;
    std::map<std::string, ModuleState> module_states;

    // DFS with cycle detection
    bool dfs(const std::string& module, std::vector<std::string>& sorted,
             std::vector<std::string>& path) {
        module_states[module] = ModuleState::VISITING;
        path.push_back(module);

        auto it = dependency_graph.find(module);
        if (it != dependency_graph.end()) {
            for (const auto& dep : it->second) {
                // Ensure dependency is in the graph
                if (dependency_graph.find(dep) == dependency_graph.end()) {
                    dependency_graph[dep] = {};
                    module_states[dep] = ModuleState::UNVISITED;
                }

                if (module_states[dep] == ModuleState::VISITING) {
                    // Cycle detected - add the closing node
                    path.push_back(dep);
                    return false;
                }

                if (module_states[dep] == ModuleState::UNVISITED) {
                    if (!dfs(dep, sorted, path)) {
                        return false;
                    }
                }
            }
        }

        module_states[module] = ModuleState::VISITED;
        sorted.push_back(module);
        path.pop_back();
        return true;
    }

    std::string formatCycle(const std::vector<std::string>& path) {
        if (path.empty()) return "(empty)";

        std::string result;
        // Find where cycle starts (last element appears earlier)
        size_t cycle_start = 0;
        for (size_t i = 0; i < path.size() - 1; i++) {
            if (path[i] == path.back()) {
                cycle_start = i;
                break;
            }
        }

        for (size_t i = cycle_start; i < path.size(); i++) {
            if (i > cycle_start) result += " -> ";
            result += path[i];
        }
        return result;
    }
};

// Global module resolver instance
static ModuleDependencyResolver g_module_resolver;

// Set of modules currently being loaded (for cycle detection during recursive require)
static std::set<std::string> g_loading_modules;

// ===== END MODULE DEPENDENCY RESOLVER =====

// ===== MODULE SYMBOL TABLE =====
// Tracks symbol visibility (exports) per module

class ModuleSymbolTable {
public:
    // Register a module's exported symbols
    void registerModuleExports(const std::string& module_name, const std::set<std::string>& exports) {
        module_exports[module_name] = exports;
    }

    // Check if a symbol is exported by a module
    bool isExported(const std::string& module_name, const std::string& symbol) const {
        auto it = module_exports.find(module_name);
        if (it == module_exports.end()) {
            // No exports registered = everything is visible (backward compatibility)
            return true;
        }
        if (it->second.empty()) {
            // Empty export list = nothing explicitly provided = all visible
            return true;
        }
        return it->second.count(symbol) > 0;
    }

    // Get the private (mangled) name for a symbol
    static std::string getPrivateName(const std::string& module_name, const std::string& symbol) {
        // Convert module name to safe identifier: test.modules.mod_a -> __test_modules_mod_a__
        std::string mangled = "__";
        for (char c : module_name) {
            mangled += (c == '.') ? '_' : c;
        }
        mangled += "__";
        mangled += symbol;
        return mangled;
    }

    // Clear all tracked modules
    void clear() {
        module_exports.clear();
    }

    // Debug: print all registered exports
    void dump() const {
        for (const auto& [module, exports] : module_exports) {
            std::string exp_str;
            for (const auto& e : exports) {
                if (!exp_str.empty()) exp_str += ", ";
                exp_str += e;
            }
            eshkol_debug("Module '%s' exports: [%s]", module.c_str(), exp_str.c_str());
        }
    }

private:
    std::map<std::string, std::set<std::string>> module_exports;
};

// Global symbol table instance
static ModuleSymbolTable g_symbol_table;

// ===== END MODULE SYMBOL TABLE =====

// ===== OWNERSHIP ANALYSIS =====
// Compile-time tracking of owned values for OALR memory management

class OwnershipAnalyzer {
public:
    // Ownership state for a variable
    enum class State {
        UNOWNED,    // Normal value, no ownership tracking
        OWNED,      // Owned value, must be consumed before scope exit
        MOVED,      // Has been moved, cannot be used
        BORROWED    // Currently borrowed by a borrow expression
    };

    struct VariableInfo {
        State state;
        std::string defined_at;  // Location info for error messages
        bool is_owned_binding;   // Was bound with (owned ...)
    };

    // Ownership scope (lexical block)
    struct Scope {
        std::map<std::string, VariableInfo> variables;
        std::string name;  // For debugging
        std::set<std::string> borrowed_vars;  // Variables currently borrowed in this scope
    };

    OwnershipAnalyzer() : has_errors_(false) {}

    // Run analysis on all ASTs
    bool analyze(const std::vector<eshkol_ast_t>& asts) {
        has_errors_ = false;
        errors_.clear();
        scope_stack_.clear();

        // Push global scope
        pushScope("global");

        for (const auto& ast : asts) {
            analyzeAST(&ast);
        }

        // Check global scope exit
        checkScopeExit();
        popScope();

        return !has_errors_;
    }

    // Get error messages
    const std::vector<std::string>& getErrors() const { return errors_; }
    bool hasErrors() const { return has_errors_; }

    // Print all errors
    void printErrors() const {
        for (const auto& err : errors_) {
            eshkol_error("%s", err.c_str());
        }
    }

private:
    std::vector<Scope> scope_stack_;
    std::vector<std::string> errors_;
    bool has_errors_;

    void pushScope(const std::string& name = "") {
        scope_stack_.push_back(Scope{{}, name, {}});
    }

    void popScope() {
        if (!scope_stack_.empty()) {
            scope_stack_.pop_back();
        }
    }

    Scope& currentScope() {
        return scope_stack_.back();
    }

    // Look up variable in all scopes
    VariableInfo* lookupVariable(const std::string& name) {
        for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
            auto var_it = it->variables.find(name);
            if (var_it != it->variables.end()) {
                return &var_it->second;
            }
        }
        return nullptr;
    }

    // Check if variable is currently borrowed anywhere
    bool isBorrowed(const std::string& name) {
        for (const auto& scope : scope_stack_) {
            if (scope.borrowed_vars.count(name) > 0) {
                return true;
            }
        }
        return false;
    }

    void reportError(const std::string& msg) {
        errors_.push_back(msg);
        has_errors_ = true;
    }

    // Check scope exit for unconsumed owned values
    void checkScopeExit() {
        if (scope_stack_.empty()) return;

        const auto& scope = currentScope();

        // Global scope owned values don't require consumption (they live until program exit)
        if (scope.name == "global") return;

        for (const auto& [name, info] : scope.variables) {
            if (info.state == State::OWNED && info.is_owned_binding) {
                reportError("Owned value '" + name + "' not consumed before scope exit" +
                           (info.defined_at.empty() ? "" : " (defined at " + info.defined_at + ")"));
            }
        }
    }

    // Get variable name from AST
    std::string getVarName(const eshkol_ast_t* ast) {
        if (ast && ast->type == ESHKOL_VAR && ast->variable.id) {
            return ast->variable.id;
        }
        return "";
    }

    void analyzeAST(const eshkol_ast_t* ast) {
        if (!ast) return;

        switch (ast->type) {
            case ESHKOL_VAR: {
                // Variable use - check if it's been moved
                std::string name = getVarName(ast);
                if (!name.empty()) {
                    VariableInfo* info = lookupVariable(name);
                    if (info && info->state == State::MOVED) {
                        reportError("Use of moved value '" + name + "'");
                    }
                }
                break;
            }

            case ESHKOL_OP:
                analyzeOperation(&ast->operation);
                break;

            case ESHKOL_CONS:
                analyzeAST(ast->cons_cell.car);
                analyzeAST(ast->cons_cell.cdr);
                break;

            default:
                // Literals etc - nothing to analyze
                break;
        }
    }

    void analyzeOperation(const eshkol_operations_t* op) {
        if (!op) return;

        switch (op->op) {
            case ESHKOL_DEFINE_OP: {
                // Track defined variable
                std::string name = op->define_op.name ? op->define_op.name : "";
                if (!name.empty()) {
                    // Check if the value transfers ownership (owned or move)
                    bool is_owned = transfersOwnership(op->define_op.value);
                    currentScope().variables[name] = {
                        is_owned ? State::OWNED : State::UNOWNED,
                        name,  // Use variable name as location identifier
                        is_owned
                    };
                    // Analyze the value (this handles the move marking)
                    analyzeAST(op->define_op.value);
                }
                break;
            }

            case ESHKOL_OWNED_OP: {
                // (owned expr) - the result should be tracked as owned
                // The actual tracking happens when bound to a variable
                analyzeAST(op->owned_op.value);
                break;
            }

            case ESHKOL_MOVE_OP: {
                // (move var) - transfers ownership, marks var as moved
                std::string var_name = getVarName(op->move_op.value);
                if (!var_name.empty()) {
                    VariableInfo* info = lookupVariable(var_name);
                    if (info) {
                        if (info->state == State::MOVED) {
                            reportError("Double move of '" + var_name + "' - value already moved");
                        } else if (isBorrowed(var_name)) {
                            reportError("Cannot move '" + var_name + "' while it is borrowed");
                        } else {
                            info->state = State::MOVED;
                        }
                    }
                } else {
                    // Moving a non-variable expression - just analyze it
                    analyzeAST(op->move_op.value);
                }
                break;
            }

            case ESHKOL_BORROW_OP: {
                // (borrow var body...) - marks var as borrowed during body
                std::string var_name = getVarName(op->borrow_op.value);
                if (!var_name.empty()) {
                    VariableInfo* info = lookupVariable(var_name);
                    if (info && info->state == State::MOVED) {
                        reportError("Cannot borrow moved value '" + var_name + "'");
                    } else {
                        // Mark as borrowed for this scope
                        currentScope().borrowed_vars.insert(var_name);
                    }
                }
                // Analyze body
                for (uint64_t i = 0; i < op->borrow_op.num_body_exprs; i++) {
                    analyzeAST(&op->borrow_op.body[i]);
                }
                // Unborrow at end
                if (!var_name.empty()) {
                    currentScope().borrowed_vars.erase(var_name);
                }
                break;
            }

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                pushScope("let");
                // Analyze bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        bool is_owned = transfersOwnership(binding->cons_cell.cdr);
                        if (!var_name.empty()) {
                            currentScope().variables[var_name] = {
                                is_owned ? State::OWNED : State::UNOWNED,
                                "",
                                is_owned
                            };
                        }
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }
                // Analyze body
                analyzeAST(op->let_op.body);
                // Check scope exit
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_LETREC_OP: {
                pushScope("letrec");
                // First pass: register all bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        bool is_owned = transfersOwnership(binding->cons_cell.cdr);
                        if (!var_name.empty()) {
                            currentScope().variables[var_name] = {
                                is_owned ? State::OWNED : State::UNOWNED,
                                "",
                                is_owned
                            };
                        }
                    }
                }
                // Second pass: analyze values
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }
                // Analyze body
                analyzeAST(op->let_op.body);
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_LAMBDA_OP: {
                pushScope("lambda");
                // Add parameters to scope
                for (uint64_t i = 0; i < op->lambda_op.num_params; i++) {
                    std::string param_name = getVarName(&op->lambda_op.parameters[i]);
                    if (!param_name.empty()) {
                        currentScope().variables[param_name] = {State::UNOWNED, "", false};
                    }
                }
                // Analyze body
                analyzeAST(op->lambda_op.body);
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_CALL_OP: {
                // Analyze function and arguments
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP: {
                // These use call_op structure
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_SEQUENCE_OP: {
                for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                    analyzeAST(&op->sequence_op.expressions[i]);
                }
                break;
            }

            case ESHKOL_WITH_REGION_OP: {
                pushScope("region");
                for (uint64_t i = 0; i < op->with_region_op.num_body_exprs; i++) {
                    analyzeAST(&op->with_region_op.body[i]);
                }
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_SHARED_OP:
                analyzeAST(op->shared_op.value);
                break;

            case ESHKOL_WEAK_REF_OP:
                analyzeAST(op->weak_ref_op.value);
                break;

            case ESHKOL_SET_OP:
                analyzeAST(op->set_op.value);
                break;

            case ESHKOL_QUOTE_OP:
                // Quoted data - no ownership analysis needed
                break;

            default:
                // Other ops - nothing special to do
                break;
        }
    }

    // Check if expression is (owned ...)
    bool isOwnedExpr(const eshkol_ast_t* ast) {
        if (!ast) return false;
        if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_OWNED_OP) {
            return true;
        }
        return false;
    }

    // Check if expression is (move ...)
    bool isMoveExpr(const eshkol_ast_t* ast) {
        if (!ast) return false;
        if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_MOVE_OP) {
            return true;
        }
        return false;
    }

    // Check if expression transfers ownership (owned or move)
    bool transfersOwnership(const eshkol_ast_t* ast) {
        return isOwnedExpr(ast) || isMoveExpr(ast);
    }
};

// Global ownership analyzer
static OwnershipAnalyzer g_ownership_analyzer;

// ===== END OWNERSHIP ANALYSIS =====

// ===== ESCAPE ANALYSIS =====
// Compile-time tracking of value flow for allocation decisions

class EscapeAnalyzer {
public:
    // Escape classification - determines allocation strategy
    enum class EscapeKind {
        NO_ESCAPE,      // Value stays in scope → Stack allocation
        RETURN_ESCAPE,  // Value returned from function → Caller's region
        CLOSURE_ESCAPE, // Value captured by closure → Shared (ref-counted)
        GLOBAL_ESCAPE   // Value stored in global/mutable → Shared (ref-counted)
    };

    // Allocation strategy based on escape analysis
    enum class AllocationStrategy {
        STACK,          // Fast bump-pointer, freed on scope exit
        REGION,         // Arena allocation, freed with region
        SHARED          // Reference-counted, freed when count hits zero
    };

    // Node in the escape graph representing a value
    struct EscapeNode {
        std::string name;           // Variable/value name (for debugging)
        EscapeKind escape_kind;     // Current escape classification
        AllocationStrategy strategy;// Determined allocation strategy
        std::set<std::string> flows_to;  // Values this flows into
        std::set<std::string> flows_from;// Values that flow into this
        bool is_return_value;       // Is this a return value?
        bool is_closure_captured;   // Captured by a closure?
        bool is_globally_stored;    // Stored in global/mutable location?
        int scope_depth;            // Lexical scope depth
    };

    // Scope for tracking values
    struct EscapeScope {
        std::string name;
        int depth;
        std::set<std::string> local_values;  // Values defined in this scope
        std::string return_target;           // What variable receives our return?
    };

    EscapeAnalyzer() : current_depth_(0), in_lambda_(false) {}

    // Run escape analysis on all ASTs
    bool analyze(const std::vector<eshkol_ast_t>& asts) {
        escape_graph_.clear();
        scope_stack_.clear();
        current_depth_ = 0;
        in_lambda_ = false;

        // Push global scope
        pushScope("global");

        for (const auto& ast : asts) {
            analyzeAST(&ast);
        }

        popScope();

        // Compute final escape classifications
        computeEscapeKinds();

        // Determine allocation strategies
        determineAllocationStrategies();

        return true;  // Escape analysis is informational, doesn't fail
    }

    // Get escape info for a variable
    const EscapeNode* getEscapeInfo(const std::string& name) const {
        auto it = escape_graph_.find(name);
        return (it != escape_graph_.end()) ? &it->second : nullptr;
    }

    // Get allocation strategy for a variable
    AllocationStrategy getAllocationStrategy(const std::string& name) const {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            return it->second.strategy;
        }
        return AllocationStrategy::STACK;  // Default to stack
    }

    // Print escape analysis results (for debugging)
    void printAnalysis() const {
        eshkol_debug("=== Escape Analysis Results ===");
        for (const auto& [name, node] : escape_graph_) {
            const char* escape_str = "unknown";
            switch (node.escape_kind) {
                case EscapeKind::NO_ESCAPE: escape_str = "no-escape"; break;
                case EscapeKind::RETURN_ESCAPE: escape_str = "return"; break;
                case EscapeKind::CLOSURE_ESCAPE: escape_str = "closure"; break;
                case EscapeKind::GLOBAL_ESCAPE: escape_str = "global"; break;
            }
            const char* alloc_str = "unknown";
            switch (node.strategy) {
                case AllocationStrategy::STACK: alloc_str = "stack"; break;
                case AllocationStrategy::REGION: alloc_str = "region"; break;
                case AllocationStrategy::SHARED: alloc_str = "shared"; break;
            }
            eshkol_debug("  %s: escape=%s, alloc=%s",
                        name.c_str(), escape_str, alloc_str);
        }
    }

private:
    std::map<std::string, EscapeNode> escape_graph_;
    std::vector<EscapeScope> scope_stack_;
    int current_depth_;
    bool in_lambda_;
    std::set<std::string> captured_by_current_lambda_;

    void pushScope(const std::string& name) {
        scope_stack_.push_back({name, current_depth_, {}, ""});
        current_depth_++;
    }

    void popScope() {
        if (!scope_stack_.empty()) {
            scope_stack_.pop_back();
            current_depth_--;
        }
    }

    EscapeScope& currentScope() {
        return scope_stack_.back();
    }

    // Register a new value in the escape graph
    void registerValue(const std::string& name, bool is_return = false) {
        if (escape_graph_.find(name) == escape_graph_.end()) {
            EscapeNode node;
            node.name = name;
            node.escape_kind = EscapeKind::NO_ESCAPE;
            node.strategy = AllocationStrategy::STACK;
            node.is_return_value = is_return;
            node.is_closure_captured = false;
            node.is_globally_stored = false;
            node.scope_depth = current_depth_;
            escape_graph_[name] = node;
        }
        currentScope().local_values.insert(name);
    }

    // Record that value 'from' flows into value 'to'
    void recordFlow(const std::string& from, const std::string& to) {
        if (escape_graph_.find(from) != escape_graph_.end()) {
            escape_graph_[from].flows_to.insert(to);
        }
        if (escape_graph_.find(to) != escape_graph_.end()) {
            escape_graph_[to].flows_from.insert(from);
        }
    }

    // Mark a value as captured by a closure
    void markClosureCaptured(const std::string& name) {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            it->second.is_closure_captured = true;
        }
    }

    // Mark a value as stored globally
    void markGloballyStored(const std::string& name) {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            it->second.is_globally_stored = true;
        }
    }

    // Mark a value as being returned
    void markAsReturn(const std::string& name) {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            it->second.is_return_value = true;
        }
    }

    // Get variable name from AST
    std::string getVarName(const eshkol_ast_t* ast) {
        if (ast && ast->type == ESHKOL_VAR && ast->variable.id) {
            return ast->variable.id;
        }
        return "";
    }

    // Generate unique name for anonymous values
    int anon_counter_ = 0;
    std::string genAnonName() {
        return "$anon" + std::to_string(anon_counter_++);
    }

    void analyzeAST(const eshkol_ast_t* ast) {
        if (!ast) return;

        switch (ast->type) {
            case ESHKOL_VAR: {
                std::string name = getVarName(ast);
                // Check if this variable is from an outer scope (closure capture)
                if (in_lambda_ && !name.empty()) {
                    // Check if it's defined in an outer scope
                    for (int i = scope_stack_.size() - 2; i >= 0; i--) {
                        if (scope_stack_[i].local_values.count(name) > 0) {
                            // This is a capture!
                            markClosureCaptured(name);
                            captured_by_current_lambda_.insert(name);
                            break;
                        }
                    }
                }
                break;
            }

            case ESHKOL_OP:
                analyzeOperation(&ast->operation);
                break;

            case ESHKOL_CONS:
                analyzeAST(ast->cons_cell.car);
                analyzeAST(ast->cons_cell.cdr);
                break;

            default:
                break;
        }
    }

    void analyzeOperation(const eshkol_operations_t* op) {
        if (!op) return;

        switch (op->op) {
            case ESHKOL_DEFINE_OP: {
                std::string name = op->define_op.name ? op->define_op.name : "";
                if (!name.empty()) {
                    registerValue(name);

                    // If at global scope, mark as globally stored
                    if (current_depth_ == 1) {  // depth 1 is global scope
                        markGloballyStored(name);
                    }

                    // Track flow from value to variable
                    std::string value_name = getVarName(op->define_op.value);
                    if (!value_name.empty()) {
                        recordFlow(value_name, name);
                    }

                    analyzeAST(op->define_op.value);
                }
                break;
            }

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                pushScope("let");

                // Analyze bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        if (!var_name.empty()) {
                            registerValue(var_name);

                            std::string value_name = getVarName(binding->cons_cell.cdr);
                            if (!value_name.empty()) {
                                recordFlow(value_name, var_name);
                            }
                        }
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }

                // Analyze body - the result escapes if this is the return value
                analyzeAST(op->let_op.body);

                // Check if body returns a local value (return escape)
                std::string body_name = getVarName(op->let_op.body);
                if (!body_name.empty() && currentScope().local_values.count(body_name) > 0) {
                    // This local value is returned from the let
                    markAsReturn(body_name);
                }

                popScope();
                break;
            }

            case ESHKOL_LETREC_OP: {
                pushScope("letrec");

                // First pass: register all bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        if (!var_name.empty()) {
                            registerValue(var_name);
                        }
                    }
                }

                // Second pass: analyze values
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }

                analyzeAST(op->let_op.body);
                popScope();
                break;
            }

            case ESHKOL_LAMBDA_OP: {
                bool was_in_lambda = in_lambda_;
                in_lambda_ = true;
                std::set<std::string> prev_captured = captured_by_current_lambda_;
                captured_by_current_lambda_.clear();

                pushScope("lambda");

                // Register parameters
                for (uint64_t i = 0; i < op->lambda_op.num_params; i++) {
                    std::string param_name = getVarName(&op->lambda_op.parameters[i]);
                    if (!param_name.empty()) {
                        registerValue(param_name);
                    }
                }

                // Analyze body
                analyzeAST(op->lambda_op.body);

                // Check if body returns a local value
                std::string body_name = getVarName(op->lambda_op.body);
                if (!body_name.empty() && currentScope().local_values.count(body_name) > 0) {
                    markAsReturn(body_name);
                }

                popScope();

                captured_by_current_lambda_ = prev_captured;
                in_lambda_ = was_in_lambda;
                break;
            }

            case ESHKOL_CALL_OP: {
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_SET_OP: {
                // set! to a variable - if the target is in an outer scope,
                // the value escapes
                std::string target = op->set_op.name ? op->set_op.name : "";

                if (!target.empty()) {
                    // Check if target is in a parent scope
                    bool in_outer_scope = false;
                    for (int i = scope_stack_.size() - 2; i >= 0; i--) {
                        if (scope_stack_[i].local_values.count(target) > 0) {
                            in_outer_scope = true;
                            break;
                        }
                    }

                    if (in_outer_scope || current_depth_ == 1) {
                        // Value escapes via mutation
                        std::string value_name = getVarName(op->set_op.value);
                        if (!value_name.empty()) {
                            markGloballyStored(value_name);
                        }
                    }

                    std::string value_name = getVarName(op->set_op.value);
                    if (!value_name.empty()) {
                        recordFlow(value_name, target);
                    }
                }

                analyzeAST(op->set_op.value);
                break;
            }

            case ESHKOL_SEQUENCE_OP: {
                for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                    analyzeAST(&op->sequence_op.expressions[i]);
                }
                break;
            }

            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP: {
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_WITH_REGION_OP: {
                pushScope("region");
                for (uint64_t i = 0; i < op->with_region_op.num_body_exprs; i++) {
                    analyzeAST(&op->with_region_op.body[i]);
                }
                popScope();
                break;
            }

            case ESHKOL_OWNED_OP:
                analyzeAST(op->owned_op.value);
                break;

            case ESHKOL_MOVE_OP:
                analyzeAST(op->move_op.value);
                break;

            case ESHKOL_BORROW_OP: {
                analyzeAST(op->borrow_op.value);
                for (uint64_t i = 0; i < op->borrow_op.num_body_exprs; i++) {
                    analyzeAST(&op->borrow_op.body[i]);
                }
                break;
            }

            case ESHKOL_SHARED_OP: {
                // (shared expr) - explicitly requests shared allocation
                std::string value_name = getVarName(op->shared_op.value);
                if (!value_name.empty()) {
                    // Mark as globally stored to force shared allocation
                    markGloballyStored(value_name);
                }
                analyzeAST(op->shared_op.value);
                break;
            }

            case ESHKOL_WEAK_REF_OP:
                analyzeAST(op->weak_ref_op.value);
                break;

            default:
                break;
        }
    }

    // Propagate escape information through the flow graph
    void computeEscapeKinds() {
        // Fixed-point iteration until no changes
        bool changed = true;
        int iterations = 0;
        const int max_iterations = 100;  // Safety limit

        while (changed && iterations < max_iterations) {
            changed = false;
            iterations++;

            for (auto& [name, node] : escape_graph_) {
                EscapeKind new_kind = node.escape_kind;

                // Check direct escape markers
                if (node.is_globally_stored) {
                    new_kind = EscapeKind::GLOBAL_ESCAPE;
                } else if (node.is_closure_captured) {
                    new_kind = std::max(new_kind, EscapeKind::CLOSURE_ESCAPE);
                } else if (node.is_return_value) {
                    new_kind = std::max(new_kind, EscapeKind::RETURN_ESCAPE);
                }

                // Propagate from flows_to (if we flow into an escaping value, we escape)
                for (const auto& target : node.flows_to) {
                    auto it = escape_graph_.find(target);
                    if (it != escape_graph_.end()) {
                        new_kind = std::max(new_kind, it->second.escape_kind);
                    }
                }

                if (new_kind != node.escape_kind) {
                    node.escape_kind = new_kind;
                    changed = true;
                }
            }
        }
    }

    // Determine allocation strategy based on escape kind
    void determineAllocationStrategies() {
        for (auto& [name, node] : escape_graph_) {
            switch (node.escape_kind) {
                case EscapeKind::NO_ESCAPE:
                    node.strategy = AllocationStrategy::STACK;
                    break;
                case EscapeKind::RETURN_ESCAPE:
                    node.strategy = AllocationStrategy::REGION;
                    break;
                case EscapeKind::CLOSURE_ESCAPE:
                case EscapeKind::GLOBAL_ESCAPE:
                    node.strategy = AllocationStrategy::SHARED;
                    break;
            }
        }
    }
};

// Global escape analyzer
static EscapeAnalyzer g_escape_analyzer;

// ===== END ESCAPE ANALYSIS =====

// Forward declarations
static void process_imports(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode);
static void load_file_asts(const std::string& filepath, std::vector<eshkol_ast_t>& asts, bool debug_mode);

static void print_help(int x = 0)
{
    printf(
        "Usage: eshkol-run [options] <input.esk|input.o> [input.esk|input.o]\n"
        "       eshkol-run -e '<expression>'   (JIT evaluate expression)\n"
        "       eshkol-run -r <file.esk>       (JIT run file)\n\n"
        "\t--help:[-h] = Print this help message.\n"
        "\t--debug:[-d] = Debugging information added inside the program.\n"
        "\t--dump-ast:[-a] = Dumps the AST into a .ast file.\n"
        "\t--dump-ir:[-i] = Dumps the IR into a .ll file.\n"
        "\t--output:[-o] = Outputs into a binary file.\n"
        "\t--compile-only:[-c] = Compiles into an intermediate object file.\n"
        "\t--shared-lib:[-s] = Compiles it into a shared library.\n"
        "\t--wasm:[-w] = Compiles to WebAssembly (.wasm) format.\n"
        "\t--lib:[-l] = Links a shared library to the resulting executable.\n"
        "\t--lib-path:[-L] = Adds a directory to the library search path.\n"
        "\t--no-stdlib:[-n] = Do not auto-load the standard library.\n"
        "\t--eval:[-e] = JIT evaluate an expression and print the result.\n"
        "\t--run:[-r] = JIT run a file (interpret without compiling).\n"
        "\t--debug-info:[-g] = Emit DWARF debug info (enables lldb/gdb source-level debugging).\n"
        "\t--optimize:[-O] N = Set LLVM optimization level (0=none, 1=basic, 2=full, 3=aggressive).\n"
        "\t--strict-types = Type errors are fatal (default: gradual/warnings).\n"
        "\t--unsafe = Skip all type checks.\n\n"
        "Eshkol Compiler v%s\n",
        ESHKOL_VER
    );
    exit(x);
}

// Load ASTs from a file
static void load_file_asts(const std::string& filepath, std::vector<eshkol_ast_t>& asts, bool debug_mode)
{
    // Check file exists first (required before calling canonical)
    if (!std::filesystem::exists(filepath)) {
        eshkol_error("File not found: %s", filepath.c_str());
        return;
    }

    // Normalize path to prevent duplicate imports
    std::string normalized_path = std::filesystem::canonical(filepath).string();

    // Skip if already imported
    if (imported_files.count(normalized_path)) {
        if (debug_mode) {
            eshkol_debug("Skipping already imported file: %s", normalized_path.c_str());
        }
        return;
    }
    imported_files.insert(normalized_path);

    std::ifstream read_file(filepath);
    if (!read_file.is_open()) {
        eshkol_error("Failed to open file: %s", filepath.c_str());
        return;
    }

    if (debug_mode) {
        eshkol_info("Loading file: %s", filepath.c_str());
    }

    eshkol_ast_t ast = eshkol_parse_next_ast(read_file);
    while (ast.type != ESHKOL_INVALID) {
        if (debug_mode) {
            printf("\n=== AST Debug Output ===\n");
            eshkol_ast_pretty_print(&ast, 0);
            printf("========================\n\n");
        }
        asts.push_back(ast);
        ast = eshkol_parse_next_ast(read_file);
    }

    read_file.close();
}

// Process import statements in ASTs and load referenced files
static void process_imports(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode)
{
    std::vector<eshkol_ast_t> new_asts;
    std::vector<eshkol_ast_t> imported_asts;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_IMPORT_OP) {
            // This is an import statement
            std::string import_path = ast.operation.import_op.path;

            // Resolve relative paths
            std::filesystem::path resolved_path;
            if (import_path[0] == '/') {
                // Absolute path
                resolved_path = import_path;
            } else {
                // Relative to base directory
                resolved_path = std::filesystem::path(base_dir) / import_path;
            }

            if (!std::filesystem::exists(resolved_path)) {
                eshkol_error("Import file not found: %s", resolved_path.c_str());
                continue;
            }

            // Load the imported file
            std::vector<eshkol_ast_t> file_asts;
            load_file_asts(resolved_path.string(), file_asts, debug_mode);

            // Recursively process imports in the loaded file
            std::string import_dir = resolved_path.parent_path().string();
            process_imports(file_asts, import_dir, debug_mode);

            // Add imported ASTs (definitions from the imported file)
            for (auto& imported_ast : file_asts) {
                imported_asts.push_back(imported_ast);
            }

            // Don't add the import statement itself to new_asts
        } else {
            // Not an import, keep the AST
            new_asts.push_back(ast);
        }
    }

    // Prepend imported ASTs before the current file's ASTs
    asts.clear();
    for (auto& ast : imported_asts) {
        asts.push_back(ast);
    }
    for (auto& ast : new_asts) {
        asts.push_back(ast);
    }
}

// Find the stdlib.esk file
static std::string find_stdlib()
{
    // Check common locations
    std::vector<std::string> stdlib_paths = {
        // Relative to current directory (development)
        "lib/stdlib.esk",
        "../lib/stdlib.esk",
        // Installation paths
        "/usr/local/share/eshkol/stdlib.esk",
        "/usr/share/eshkol/stdlib.esk",
    };

    // Also check relative to the executable
    char exe_path[4096];
    bool got_exe_path = false;

#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        got_exe_path = true;
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        got_exe_path = true;
    }
#endif

    if (got_exe_path) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "../lib/stdlib.esk").string());
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "stdlib.esk").string());
    }

    for (const auto& path : stdlib_paths) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::canonical(path).string();
        }
    }

    return "";  // Not found
}

// Find the pre-compiled stdlib.o file
static std::string find_stdlib_object()
{
    std::vector<std::string> stdlib_paths = {
        // Relative to current directory (development)
        "stdlib.o",
        "build/stdlib.o",
        "../build/stdlib.o",
        // Installation paths
        "/usr/local/lib/eshkol/stdlib.o",
        "/usr/local/lib/stdlib.o",
        "/usr/lib/eshkol/stdlib.o",
        // Homebrew paths
        "/opt/homebrew/lib/eshkol/stdlib.o",
        "/opt/homebrew/lib/stdlib.o",
    };

    // Also check relative to the executable
    char exe_path[4096];
    bool got_exe_path = false;

#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        got_exe_path = true;
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        got_exe_path = true;
    }
#endif

    if (got_exe_path) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "stdlib.o").string());
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "../lib/stdlib.o").string());
        stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "../lib/eshkol/stdlib.o").string());
    }

    for (const auto& path : stdlib_paths) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::canonical(path).string();
        }
    }

    return "";  // Not found
}

// Find the runtime library (libeshkol-static.a)
static std::string find_runtime_library()
{
    std::vector<std::string> lib_paths = {
        // Relative to current directory (development)
        "libeshkol-static.a",
        "build/libeshkol-static.a",
        "../build/libeshkol-static.a",
        // Installation paths (Linux)
        "/usr/local/lib/libeshkol-static.a",
        "/usr/local/lib/eshkol/libeshkol-static.a",
        "/usr/lib/libeshkol-static.a",
        "/usr/lib/eshkol/libeshkol-static.a",
        // Homebrew paths (macOS)
        "/opt/homebrew/lib/libeshkol-static.a",
        "/opt/homebrew/lib/eshkol/libeshkol-static.a",
    };

    // Also check relative to the executable
    char exe_path[4096];
    bool got_exe_path = false;

#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        got_exe_path = true;
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        got_exe_path = true;
    }
#endif

    if (got_exe_path) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        // Check relative to executable
        lib_paths.insert(lib_paths.begin(), (exe_dir / "libeshkol-static.a").string());
        lib_paths.insert(lib_paths.begin(), (exe_dir / "../lib/libeshkol-static.a").string());
        lib_paths.insert(lib_paths.begin(), (exe_dir / "../lib/eshkol/libeshkol-static.a").string());
    }

    for (const auto& path : lib_paths) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::canonical(path).string();
        }
    }

    return "";  // Not found
}

// Check if any AST contains a require for stdlib or core.* modules
static bool requires_stdlib(const std::vector<eshkol_ast_t>& asts)
{
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string module_name = ast.operation.require_op.module_names[i];
                if (module_name == "stdlib" || module_name.find("core.") == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Find the library base directory (lib/)
static std::string find_lib_dir()
{
    // Check common locations for the lib directory
    std::vector<std::string> lib_dirs = {
        // Relative to current directory (development)
        "lib",
        "../lib",
        // Installation paths
        "/usr/local/share/eshkol/lib",
        "/usr/share/eshkol/lib",
    };

    // Also check relative to the executable
    char exe_path[4096];
    bool got_exe_path = false;

#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        got_exe_path = true;
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        got_exe_path = true;
    }
#endif

    if (got_exe_path) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        lib_dirs.insert(lib_dirs.begin(), (exe_dir / "../lib").string());
        lib_dirs.insert(lib_dirs.begin(), (exe_dir / "lib").string());
    }

    for (const auto& dir : lib_dirs) {
        if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
            return std::filesystem::canonical(dir).string();
        }
    }

    return "";  // Not found
}

// Convert symbolic module name to file path
// e.g., "data.json" -> "lib/data/json.esk"
//       "core.strings" -> "lib/core/strings.esk"
static std::string resolve_module_path(const std::string& module_name, const std::string& base_dir, const std::string& lib_dir)
{
    // Convert dots to path separators
    std::string path_part = module_name;
    for (char& c : path_part) {
        if (c == '.') c = '/';
    }
    path_part += ".esk";

    // Search order:
    // 1. Current directory (relative to base_dir)
    // 2. Library path (lib/)
    // 3. Environment variable $ESHKOL_PATH (colon-separated)

    // Try current directory first
    std::filesystem::path current_path = std::filesystem::path(base_dir) / path_part;
    if (std::filesystem::exists(current_path)) {
        return std::filesystem::canonical(current_path).string();
    }

    // Try library directory
    if (!lib_dir.empty()) {
        std::filesystem::path lib_path = std::filesystem::path(lib_dir) / path_part;
        if (std::filesystem::exists(lib_path)) {
            return std::filesystem::canonical(lib_path).string();
        }
        // Directory-as-module: if lib/web.esk doesn't exist, try lib/web/web.esk
        // This allows (require web) to find lib/web/web.esk as the package entry point
        std::filesystem::path dir_module = std::filesystem::path(lib_dir) / module_name;
        if (std::filesystem::is_directory(dir_module)) {
            // Try same-name entry point: lib/web/web.esk
            std::filesystem::path entry = dir_module / (module_name + ".esk");
            if (std::filesystem::exists(entry)) {
                return std::filesystem::canonical(entry).string();
            }
            // Try index.esk: lib/web/index.esk
            std::filesystem::path index = dir_module / "index.esk";
            if (std::filesystem::exists(index)) {
                return std::filesystem::canonical(index).string();
            }
        }
    }

    // Try $ESHKOL_PATH
    const char* eshkol_path = std::getenv("ESHKOL_PATH");
    if (eshkol_path) {
        std::stringstream ss(eshkol_path);
        std::string search_dir;
        while (std::getline(ss, search_dir, ':')) {
            std::filesystem::path env_path = std::filesystem::path(search_dir) / path_part;
            if (std::filesystem::exists(env_path)) {
                return std::filesystem::canonical(env_path).string();
            }
        }
    }

    return "";  // Not found
}

// Global library directory (cached)
static std::string g_lib_dir;

// Recursively discover all modules that a library requires.
// Used to mark all sub-modules of a pre-compiled library as pre-compiled.
// NOTE: Does NOT use load_file_asts to avoid polluting imported_files.
static void collect_all_submodules(const std::string& module_name,
                                   std::set<std::string>& out,
                                   const std::string& lib_dir) {
    if (out.count(module_name)) return;  // already visited
    out.insert(module_name);

    // Find and parse the module source to discover its requires
    std::string module_path = resolve_module_path(module_name, ".", lib_dir);
    if (module_path.empty()) return;

    // Parse directly — do NOT use load_file_asts (it tracks imported_files
    // and would prevent process_requires from loading these modules later)
    std::ifstream file(module_path);
    if (!file.is_open()) return;

    eshkol_ast_t ast = eshkol_parse_next_ast(file);
    while (ast.type != ESHKOL_INVALID) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string sub = ast.operation.require_op.module_names[i];
                collect_all_submodules(sub, out, lib_dir);
            }
        }
        ast = eshkol_parse_next_ast(file);
    }
}

// ===== SYMBOL VISIBILITY HELPERS =====

// Collect exported symbols from provide declarations in ASTs
static std::set<std::string> collect_module_exports(const std::vector<eshkol_ast_t>& asts) {
    std::set<std::string> exports;
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_PROVIDE_OP) {
            for (uint64_t i = 0; i < ast.operation.provide_op.num_exports; i++) {
                exports.insert(ast.operation.provide_op.export_names[i]);
            }
        }
    }
    return exports;
}

// Collect all defined symbols from ASTs
static std::set<std::string> collect_defined_symbols(const std::vector<eshkol_ast_t>& asts) {
    std::set<std::string> defined;
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP) {
            if (ast.operation.define_op.name) {
                defined.insert(ast.operation.define_op.name);
            }
        }
    }
    return defined;
}

// Forward declaration for recursive reference updating
static void update_ast_references(eshkol_ast_t* ast,
                                  const std::map<std::string, std::string>& rename_map);

// Update references in an AST node recursively
static void update_ast_references(eshkol_ast_t* ast,
                                  const std::map<std::string, std::string>& rename_map) {
    if (!ast) return;

    switch (ast->type) {
        case ESHKOL_VAR: {
            if (ast->variable.id) {
                auto it = rename_map.find(ast->variable.id);
                if (it != rename_map.end()) {
                    delete[] ast->variable.id;
                    ast->variable.id = strdup(it->second.c_str());
                }
            }
            break;
        }

        case ESHKOL_OP: {
            switch (ast->operation.op) {
                case ESHKOL_CALL_OP:
                case ESHKOL_COND_OP:  // cond uses call_op structure
                case ESHKOL_IF_OP:    // if uses call_op structure
                    if (ast->operation.call_op.func) {
                        update_ast_references(ast->operation.call_op.func, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        update_ast_references(&ast->operation.call_op.variables[i], rename_map);
                    }
                    break;

                case ESHKOL_AND_OP:
                case ESHKOL_OR_OP:
                    // NOTE: AND_OP/OR_OP use sequence_op structure, NOT call_op!
                    for (uint64_t i = 0; i < ast->operation.sequence_op.num_expressions; i++) {
                        update_ast_references(&ast->operation.sequence_op.expressions[i], rename_map);
                    }
                    break;

                case ESHKOL_DEFINE_OP:
                    // Don't rename the definition name itself - that's handled separately
                    // But do update references in the body
                    update_ast_references(ast->operation.define_op.value, rename_map);
                    break;

                case ESHKOL_LAMBDA_OP:
                    update_ast_references(ast->operation.lambda_op.body, rename_map);
                    break;

                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:  // R7RS letrec* - used for internal defines
                    // Each binding is a CONS cell: (var . value)
                    for (uint64_t i = 0; i < ast->operation.let_op.num_bindings; i++) {
                        eshkol_ast_t* binding = &ast->operation.let_op.bindings[i];
                        if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr) {
                            // Update references in the value part
                            update_ast_references(binding->cons_cell.cdr, rename_map);
                        } else {
                            // Fallback: treat entire binding as expression
                            update_ast_references(binding, rename_map);
                        }
                    }
                    if (ast->operation.let_op.body) {
                        update_ast_references(ast->operation.let_op.body, rename_map);
                    }
                    break;

                case ESHKOL_SEQUENCE_OP:
                    for (uint64_t i = 0; i < ast->operation.sequence_op.num_expressions; i++) {
                        update_ast_references(&ast->operation.sequence_op.expressions[i], rename_map);
                    }
                    break;

                case ESHKOL_SET_OP:
                    // Update the target variable name
                    if (ast->operation.set_op.name) {
                        auto it = rename_map.find(ast->operation.set_op.name);
                        if (it != rename_map.end()) {
                            delete[] ast->operation.set_op.name;
                            ast->operation.set_op.name = strdup(it->second.c_str());
                        }
                    }
                    update_ast_references(ast->operation.set_op.value, rename_map);
                    break;

                // Calculus operations - function + point (or expression)
                case ESHKOL_GRADIENT_OP:
                    if (ast->operation.gradient_op.function) {
                        update_ast_references(ast->operation.gradient_op.function, rename_map);
                    }
                    if (ast->operation.gradient_op.point) {
                        update_ast_references(ast->operation.gradient_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DERIVATIVE_OP:
                    if (ast->operation.derivative_op.function) {
                        update_ast_references(ast->operation.derivative_op.function, rename_map);
                    }
                    if (ast->operation.derivative_op.point) {
                        update_ast_references(ast->operation.derivative_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DIRECTIONAL_DERIV_OP:
                    if (ast->operation.directional_deriv_op.function) {
                        update_ast_references(ast->operation.directional_deriv_op.function, rename_map);
                    }
                    if (ast->operation.directional_deriv_op.point) {
                        update_ast_references(ast->operation.directional_deriv_op.point, rename_map);
                    }
                    if (ast->operation.directional_deriv_op.direction) {
                        update_ast_references(ast->operation.directional_deriv_op.direction, rename_map);
                    }
                    break;

                case ESHKOL_JACOBIAN_OP:
                    if (ast->operation.jacobian_op.function) {
                        update_ast_references(ast->operation.jacobian_op.function, rename_map);
                    }
                    if (ast->operation.jacobian_op.point) {
                        update_ast_references(ast->operation.jacobian_op.point, rename_map);
                    }
                    break;

                case ESHKOL_HESSIAN_OP:
                    if (ast->operation.hessian_op.function) {
                        update_ast_references(ast->operation.hessian_op.function, rename_map);
                    }
                    if (ast->operation.hessian_op.point) {
                        update_ast_references(ast->operation.hessian_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DIVERGENCE_OP:
                    if (ast->operation.divergence_op.function) {
                        update_ast_references(ast->operation.divergence_op.function, rename_map);
                    }
                    if (ast->operation.divergence_op.point) {
                        update_ast_references(ast->operation.divergence_op.point, rename_map);
                    }
                    break;

                case ESHKOL_CURL_OP:
                    if (ast->operation.curl_op.function) {
                        update_ast_references(ast->operation.curl_op.function, rename_map);
                    }
                    if (ast->operation.curl_op.point) {
                        update_ast_references(ast->operation.curl_op.point, rename_map);
                    }
                    break;

                case ESHKOL_LAPLACIAN_OP:
                    if (ast->operation.laplacian_op.function) {
                        update_ast_references(ast->operation.laplacian_op.function, rename_map);
                    }
                    if (ast->operation.laplacian_op.point) {
                        update_ast_references(ast->operation.laplacian_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DIFF_OP:
                    if (ast->operation.diff_op.expression) {
                        update_ast_references(ast->operation.diff_op.expression, rename_map);
                    }
                    break;

                // call_op structure operations (same as CALL_OP handler)
                case ESHKOL_WHEN_OP:
                case ESHKOL_UNLESS_OP:
                case ESHKOL_DO_OP:
                case ESHKOL_CASE_OP:
                case ESHKOL_QUOTE_OP:
                case ESHKOL_QUASIQUOTE_OP:
                case ESHKOL_UNQUOTE_OP:
                case ESHKOL_UNQUOTE_SPLICING_OP:
                    if (ast->operation.call_op.func) {
                        update_ast_references(ast->operation.call_op.func, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        update_ast_references(&ast->operation.call_op.variables[i], rename_map);
                    }
                    break;

                // Control flow operations
                case ESHKOL_DYNAMIC_WIND_OP:
                    if (ast->operation.dynamic_wind_op.before) {
                        update_ast_references(ast->operation.dynamic_wind_op.before, rename_map);
                    }
                    if (ast->operation.dynamic_wind_op.thunk) {
                        update_ast_references(ast->operation.dynamic_wind_op.thunk, rename_map);
                    }
                    if (ast->operation.dynamic_wind_op.after) {
                        update_ast_references(ast->operation.dynamic_wind_op.after, rename_map);
                    }
                    break;

                case ESHKOL_CALL_CC_OP:
                    if (ast->operation.call_cc_op.proc) {
                        update_ast_references(ast->operation.call_cc_op.proc, rename_map);
                    }
                    break;

                case ESHKOL_GUARD_OP: {
                    for (uint64_t i = 0; i < ast->operation.guard_op.num_body_exprs; i++) {
                        update_ast_references(&ast->operation.guard_op.body[i], rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.guard_op.num_clauses; i++) {
                        update_ast_references(&ast->operation.guard_op.clauses[i], rename_map);
                    }
                    break;
                }

                case ESHKOL_RAISE_OP:
                    if (ast->operation.raise_op.exception) {
                        update_ast_references(ast->operation.raise_op.exception, rename_map);
                    }
                    break;

                case ESHKOL_VALUES_OP:
                    for (uint64_t i = 0; i < ast->operation.values_op.num_values; i++) {
                        update_ast_references(&ast->operation.values_op.expressions[i], rename_map);
                    }
                    break;

                case ESHKOL_CALL_WITH_VALUES_OP:
                    if (ast->operation.call_with_values_op.producer) {
                        update_ast_references(ast->operation.call_with_values_op.producer, rename_map);
                    }
                    if (ast->operation.call_with_values_op.consumer) {
                        update_ast_references(ast->operation.call_with_values_op.consumer, rename_map);
                    }
                    break;

                case ESHKOL_MATCH_OP: {
                    if (ast->operation.match_op.expr) {
                        update_ast_references(ast->operation.match_op.expr, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.match_op.num_clauses; i++) {
                        if (ast->operation.match_op.clauses[i].guard) {
                            update_ast_references(ast->operation.match_op.clauses[i].guard, rename_map);
                        }
                        if (ast->operation.match_op.clauses[i].body) {
                            update_ast_references(ast->operation.match_op.clauses[i].body, rename_map);
                        }
                    }
                    break;
                }

                // Memory management operations - FIXED: with_region_op.body is an array
                case ESHKOL_WITH_REGION_OP:
                    for (uint64_t i = 0; i < ast->operation.with_region_op.num_body_exprs; i++) {
                        update_ast_references(&ast->operation.with_region_op.body[i], rename_map);
                    }
                    break;

                case ESHKOL_OWNED_OP:
                    update_ast_references(ast->operation.owned_op.value, rename_map);
                    break;

                case ESHKOL_MOVE_OP:
                    update_ast_references(ast->operation.move_op.value, rename_map);
                    break;

                // FIXED: borrow_op.body is an array
                case ESHKOL_BORROW_OP:
                    if (ast->operation.borrow_op.value) {
                        update_ast_references(ast->operation.borrow_op.value, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.borrow_op.num_body_exprs; i++) {
                        update_ast_references(&ast->operation.borrow_op.body[i], rename_map);
                    }
                    break;

                case ESHKOL_SHARED_OP:
                    update_ast_references(ast->operation.shared_op.value, rename_map);
                    break;

                case ESHKOL_WEAK_REF_OP:
                    update_ast_references(ast->operation.weak_ref_op.value, rename_map);
                    break;

                // Logic/consciousness operations use call_op structure
                case ESHKOL_UNIFY_OP:
                case ESHKOL_MAKE_SUBST_OP:
                case ESHKOL_WALK_OP:
                case ESHKOL_MAKE_FACT_OP:
                case ESHKOL_MAKE_KB_OP:
                case ESHKOL_KB_ASSERT_OP:
                case ESHKOL_KB_QUERY_OP:
                case ESHKOL_LOGIC_VAR_PRED_OP:
                case ESHKOL_SUBSTITUTION_PRED_OP:
                case ESHKOL_KB_PRED_OP:
                case ESHKOL_FACT_PRED_OP:
                case ESHKOL_FACTOR_GRAPH_PRED_OP:
                case ESHKOL_WORKSPACE_PRED_OP:
                case ESHKOL_MAKE_FACTOR_GRAPH_OP:
                case ESHKOL_FG_ADD_FACTOR_OP:
                case ESHKOL_FG_INFER_OP:
                case ESHKOL_FG_UPDATE_CPT_OP:
                case ESHKOL_FREE_ENERGY_OP:
                case ESHKOL_EXPECTED_FREE_ENERGY_OP:
                case ESHKOL_MAKE_WORKSPACE_OP:
                case ESHKOL_WS_REGISTER_OP:
                case ESHKOL_WS_STEP_OP:
                case ESHKOL_EXTERN_OP:
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        update_ast_references(&ast->operation.call_op.variables[i], rename_map);
                    }
                    break;

                default:
                    eshkol_debug("update_ast_references: unhandled op type %d", ast->operation.op);
                    break;
            }
            break;
        }

        case ESHKOL_CONS:
            // Traverse cons cells
            if (ast->cons_cell.car) {
                update_ast_references(ast->cons_cell.car, rename_map);
            }
            if (ast->cons_cell.cdr) {
                update_ast_references(ast->cons_cell.cdr, rename_map);
            }
            break;

        default:
            // Literals, etc. - no update needed
            break;
    }
}

// Rename private (non-exported) symbols in module ASTs
static void rename_private_symbols(std::vector<eshkol_ast_t>& asts,
                                   const std::string& module_name,
                                   const std::set<std::string>& exports,
                                   bool debug_mode) {
    // Build rename map: private_name -> mangled_name
    std::map<std::string, std::string> rename_map;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP) {
            if (ast.operation.define_op.name) {
                std::string name = ast.operation.define_op.name;
                // Only rename if not exported
                if (exports.count(name) == 0) {
                    std::string mangled = ModuleSymbolTable::getPrivateName(module_name, name);
                    rename_map[name] = mangled;

                    if (debug_mode) {
                        eshkol_debug("  Renaming private symbol '%s' -> '%s'",
                                    name.c_str(), mangled.c_str());
                    }
                }
            }
        }
    }

    if (rename_map.empty()) {
        return;  // No private symbols to rename
    }

    // Apply renames: first rename definitions, then update all references
    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP) {
            if (ast.operation.define_op.name) {
                auto it = rename_map.find(ast.operation.define_op.name);
                if (it != rename_map.end()) {
                    delete[] ast.operation.define_op.name;
                    ast.operation.define_op.name = strdup(it->second.c_str());
                }
            }
        }
        // Update all references in this AST
        update_ast_references(&ast, rename_map);
    }
}

// ===== END SYMBOL VISIBILITY HELPERS =====

// Process require statements (new module system)
static void process_requires(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode)
{
    if (g_lib_dir.empty()) {
        g_lib_dir = find_lib_dir();
    }

    std::vector<eshkol_ast_t> new_asts;
    std::vector<eshkol_ast_t> required_asts;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            // Process each required module
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string module_name = ast.operation.require_op.module_names[i];

                // Check if this module is pre-compiled (e.g., from stdlib.o)
                bool is_precompiled = precompiled_modules.count(module_name) > 0;

                // All sub-modules of pre-compiled libraries are already in
                // precompiled_modules (populated by collect_all_submodules),
                // so the direct lookup above is sufficient.

                if (is_precompiled) {
                    // For pre-compiled modules, we parse to get function declarations
                    // but strip the bodies. The actual code comes from the .o file.
                    std::string module_path = resolve_module_path(module_name, base_dir, g_lib_dir);
                    if (module_path.empty()) {
                        if (debug_mode) {
                            eshkol_warn("Pre-compiled module %s source not found - skipping declarations", module_name.c_str());
                        }
                        continue;
                    }

                    if (debug_mode) {
                        eshkol_info("Module %s is pre-compiled - loading declarations from %s", module_name.c_str(), module_path.c_str());
                    }

                    // Load the module to get function declarations
                    std::vector<eshkol_ast_t> module_asts;
                    load_file_asts(module_path, module_asts, debug_mode);

                    if (module_asts.empty()) {
                        continue;
                    }

                    // Collect module exports BEFORE processing requires (process_requires may modify ASTs)
                    std::set<std::string> module_exports = collect_module_exports(module_asts);
                    if (debug_mode) {
                        std::string exp_str;
                        for (const auto& e : module_exports) {
                            if (!exp_str.empty()) exp_str += ", ";
                            exp_str += e;
                        }
                        eshkol_debug("Module provides: %s (count=%zu)", exp_str.c_str(), module_exports.size());
                    }

                    // Recursively process requires in the module (they're also pre-compiled)
                    std::string module_dir = std::filesystem::path(module_path).parent_path().string();
                    process_requires(module_asts, module_dir, debug_mode);

                    // Extract function/variable definitions for external linkage
                    // After process_requires, module_asts contains:
                    //   - ASTs from sub-required modules (already marked external)
                    //   - This module's ASTs (need to check against module_exports)
                    // The actual code/data will come from stdlib.o at link time
                    for (auto& module_ast : module_asts) {
                        if (module_ast.type == ESHKOL_OP &&
                            module_ast.operation.op == ESHKOL_DEFINE_OP) {
                            const char* sym_name = module_ast.operation.define_op.name;
                            if (!sym_name) continue;

                            // Only export if symbol is public (in provides list) OR already external
                            // Already-external symbols come from sub-required modules
                            // Private symbols have been/will be renamed with module prefix
                            bool already_external = module_ast.operation.define_op.is_external;
                            bool is_public = already_external || module_exports.empty() || module_exports.count(sym_name) > 0;
                            if (debug_mode) {
                                eshkol_debug("  Check symbol %s: exports_empty=%d, in_exports=%d, is_public=%d",
                                            sym_name, (int)module_exports.empty(),
                                            (int)(module_exports.count(sym_name) > 0), (int)is_public);
                            }
                            if (!is_public) {
                                if (debug_mode) {
                                    eshkol_debug("  Skipping private symbol: %s", sym_name);
                                }
                                continue;
                            }

                            if (module_ast.operation.define_op.is_function) {
                                // Mark as external function (body will come from .o file)
                                module_ast.operation.define_op.is_external = 1;
                                required_asts.push_back(module_ast);
                                if (debug_mode) {
                                    eshkol_info("  External function: %s", sym_name);
                                }
                            } else {
                                // Export PUBLIC variable as external
                                // Only public variables (in provides) need external declarations
                                module_ast.operation.define_op.is_external = 1;
                                required_asts.push_back(module_ast);
                                if (debug_mode) {
                                    eshkol_info("  External variable: %s", sym_name);
                                }
                            }
                        }
                    }
                    continue;
                }

                std::string module_path = resolve_module_path(module_name, base_dir, g_lib_dir);

                if (module_path.empty()) {
                    eshkol_error("Module '%s' not found", module_name.c_str());
                    eshkol_error("  Searched:");
                    eshkol_error("    - %s/%s.esk", base_dir.c_str(), module_name.c_str());
                    if (!g_lib_dir.empty()) {
                        // Convert dots to slashes for display
                        std::string path_display = module_name;
                        for (char& c : path_display) if (c == '.') c = '/';
                        eshkol_error("    - %s/%s.esk", g_lib_dir.c_str(), path_display.c_str());
                    }
                    eshkol_error("    - $ESHKOL_PATH entries");
                    continue;
                }

                if (debug_mode) {
                    eshkol_info("Requiring module '%s' from: %s", module_name.c_str(), module_path.c_str());
                }

                // Normalize path for cycle detection
                std::string norm_path = std::filesystem::canonical(module_path).string();

                // Circular dependency detection: if this module is currently being loaded
                // higher up the call stack AND hasn't been fully loaded yet, we have a cycle.
                // If it's already in imported_files, it's fully loaded — not a cycle.
                if (g_loading_modules.count(norm_path) && !imported_files.count(norm_path)) {
                    eshkol_error("Circular dependency detected: module '%s' requires itself (directly or indirectly)",
                                module_name.c_str());
                    continue;
                }
                g_loading_modules.insert(norm_path);

                // Load the module file
                std::vector<eshkol_ast_t> module_asts;
                load_file_asts(module_path, module_asts, debug_mode);

                // Skip if module was already loaded (detected by load_file_asts)
                if (module_asts.empty()) {
                    continue;
                }

                // Collect exports from this module BEFORE processing sub-modules
                std::set<std::string> exports = collect_module_exports(module_asts);

                // Register exports in the symbol table
                g_symbol_table.registerModuleExports(module_name, exports);

                if (debug_mode && !exports.empty()) {
                    std::string exp_str;
                    for (const auto& e : exports) {
                        if (!exp_str.empty()) exp_str += ", ";
                        exp_str += e;
                    }
                    eshkol_info("Module '%s' exports: [%s]", module_name.c_str(), exp_str.c_str());
                }

                // Rename private (non-exported) symbols to avoid collisions
                if (!exports.empty()) {
                    rename_private_symbols(module_asts, module_name, exports, debug_mode);
                }

                // Recursively process requires in the loaded module
                std::string module_dir = std::filesystem::path(module_path).parent_path().string();
                process_requires(module_asts, module_dir, debug_mode);

                // Also process legacy imports
                process_imports(module_asts, module_dir, debug_mode);

                // Add module ASTs
                for (auto& module_ast : module_asts) {
                    required_asts.push_back(module_ast);
                }

                // Remove from loading stack (module fully processed)
                g_loading_modules.erase(norm_path);
            }
            // Don't add the require statement itself to new_asts
        } else if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_PROVIDE_OP) {
            // For now, we just skip provide statements
            // Full symbol visibility will be implemented later
            if (debug_mode) {
                std::string exports;
                for (uint64_t i = 0; i < ast.operation.provide_op.num_exports; i++) {
                    if (i > 0) exports += ", ";
                    exports += ast.operation.provide_op.export_names[i];
                }
                eshkol_debug("Module provides: %s", exports.c_str());
            }
            // Don't add provide to final ASTs
        } else {
            new_asts.push_back(ast);
        }
    }

    // Prepend required modules before the current file's ASTs
    asts.clear();
    for (auto& ast : required_asts) {
        asts.push_back(ast);
    }
    for (auto& ast : new_asts) {
        asts.push_back(ast);
    }
}

int main(int argc, char **argv)
{
    int ch = 0;

    uint8_t debug_mode = 0;
    uint8_t dump_ast = 0;
    uint8_t dump_ir = 0;
    uint8_t compile_only = 0;
    uint8_t shared_lib = 0;
    uint8_t wasm_output = 0;
    uint8_t no_stdlib = 0;
    uint8_t strict_types = 0;
    uint8_t unsafe_mode = 0;
    uint8_t debug_info = 0;  // -g flag: emit DWARF debug info for lldb/gdb
    int opt_level = 0;       // -O flag: LLVM optimization level (0-3)

    std::vector<char*> source_files;
    std::vector<char*> compiled_files;
    std::vector<char*> linked_libs;
    std::vector<char*> lib_paths;

    std::vector<eshkol_ast_t> asts;

    char *output = nullptr;
    char *eval_expr = nullptr;  // For -e/--eval flag
    uint8_t run_mode = 0;       // For -r/--run flag (JIT run file)

    if (argc == 1) print_help(1);

    const char* eskb_output_path = nullptr;
    while ((ch = getopt_long(argc, argv, "hdaio:cswl:L:ne:rgO:B:", long_options, nullptr)) != -1) {
        switch (ch) {
        case 'h':
            print_help(0);
            break;
        case 'd':
            debug_mode = 1;
            eshkol_set_logger_level(ESHKOL_DEBUG);
            break;
        case 'a':
            dump_ast = 1;
            break;
        case 'i':
            dump_ir = 1;
            break;
        case 'o':
            output = optarg;
            break;
        case 'c':
            compile_only = 1;
            break;
        case 's':
            shared_lib = 1;
            compile_only = 1;  // Library mode implies compile to object file
            break;
        case 'w':
            wasm_output = 1;
            eshkol_set_target("wasm32-unknown-unknown");
            break;
        case 'l':
            linked_libs.push_back(optarg);
            break;
        case 'L':
            lib_paths.push_back(optarg);
            break;
        case 'n':
            no_stdlib = 1;
            break;
        case 'e':
            eval_expr = optarg;
            break;
        case 'r':
            run_mode = 1;
            break;
        case 'g':
            debug_info = 1;
            break;
        case 'O':
            opt_level = atoi(optarg);
            if (opt_level < 0 || opt_level > 3) {
                fprintf(stderr, "Invalid optimization level: %s (must be 0-3)\n", optarg);
                return 1;
            }
            break;
        case 256:
            strict_types = 1;
            break;
        case 257:
            unsafe_mode = 1;
            break;
        case 'B':
            eskb_output_path = optarg;
            break;
        default:
            print_help(1);
        }
    }

    // If we have an eval expression, use JIT mode
    if (eval_expr) {
        // Initialize runtime system
        if (eshkol_runtime_init() != 0) {
            eshkol_error("Failed to initialize runtime system");
            return 1;
        }
        eshkol_init_limits_from_env();

        // Create JIT context
        eshkol::ReplJITContext jit_ctx;

        // Load stdlib for eval mode
        if (!no_stdlib) {
            jit_ctx.loadStdlib();
        }

        // Parse the expression using istringstream (no temp file needed)
        std::string eval_input = std::string(eval_expr) + "\n";
        std::istringstream eval_stream(eval_input);

        // Parse and execute ALL expressions from the input (not just one)
        eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(eval_stream);
        bool parsed_any = false;

        while (ast.type != ESHKOL_INVALID) {
            parsed_any = true;

            if (debug_mode) {
                printf("=== AST ===\n");
                eshkol_ast_pretty_print(&ast, 0);
                printf("===========\n");
            }

            // Execute the AST (definitions don't print, expressions do via display)
            bool is_define = (ast.type == ESHKOL_OP &&
                (ast.operation.op == ESHKOL_DEFINE_OP ||
                 ast.operation.op == ESHKOL_REQUIRE_OP ||
                 ast.operation.op == ESHKOL_IMPORT_OP));

            // Check if it's a display/print call (don't wrap these)
            bool is_output_call = false;
            if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_CALL_OP &&
                ast.operation.call_op.func && ast.operation.call_op.func->type == ESHKOL_VAR) {
                const char* name = ast.operation.call_op.func->variable.id;
                if (name && (strcmp(name, "display") == 0 || strcmp(name, "newline") == 0 ||
                             strcmp(name, "print") == 0 || strcmp(name, "write") == 0)) {
                    is_output_call = true;
                }
            }

            // Execute directly (don't auto-wrap with display - let user control output)
            jit_ctx.execute(&ast);

            // Parse next expression
            ast = eshkol_parse_next_ast_from_stream(eval_stream);
        }

        if (!parsed_any) {
            eshkol_error("Failed to parse expression: %s", eval_expr);
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }

        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
        return 0;
    }

    // If run mode, JIT execute all expressions from file(s)
    if (run_mode) {
        if (optind == argc) {
            eshkol_error("No input file specified for --run mode");
            print_help(1);
        }

        // Initialize runtime system
        if (eshkol_runtime_init() != 0) {
            eshkol_error("Failed to initialize runtime system");
            return 1;
        }
        eshkol_init_limits_from_env();

        // Create JIT context
        eshkol::ReplJITContext jit_ctx;

        // Load stdlib for run mode
        if (!no_stdlib) {
            jit_ctx.loadStdlib();
        }

        // Process each input file
        for (int i = optind; i < argc; i++) {
            std::string filepath = argv[i];

            if (!std::filesystem::exists(filepath)) {
                eshkol_error("File not found: %s", filepath.c_str());
                continue;
            }

            std::ifstream file(filepath);
            if (!file.is_open()) {
                eshkol_error("Failed to open file: %s", filepath.c_str());
                continue;
            }

            if (debug_mode) {
                eshkol_info("JIT running: %s", filepath.c_str());
            }

            // Parse and execute all expressions in the file
            eshkol_ast_t ast = eshkol_parse_next_ast(file);
            while (ast.type != ESHKOL_INVALID) {
                if (debug_mode) {
                    printf("=== AST ===\n");
                    eshkol_ast_pretty_print(&ast, 0);
                    printf("===========\n");
                }

                // Execute the AST (definitions don't print, expressions do via display)
                bool is_define = (ast.type == ESHKOL_OP &&
                    (ast.operation.op == ESHKOL_DEFINE_OP ||
                     ast.operation.op == ESHKOL_REQUIRE_OP ||
                     ast.operation.op == ESHKOL_IMPORT_OP));

                // Check if it's a display/print call
                bool is_output_call = false;
                if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_CALL_OP &&
                    ast.operation.call_op.func && ast.operation.call_op.func->type == ESHKOL_VAR) {
                    const char* name = ast.operation.call_op.func->variable.id;
                    if (name && (strcmp(name, "display") == 0 || strcmp(name, "newline") == 0 ||
                                 strcmp(name, "print") == 0 || strcmp(name, "write") == 0)) {
                        is_output_call = true;
                    }
                }

                jit_ctx.execute(&ast);

                // Parse next expression
                ast = eshkol_parse_next_ast(file);
            }
            file.close();
        }

        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
        return 0;
    }

    if (optind == argc) print_help(1);

    // Initialize runtime system (signal handlers, shutdown hooks)
    if (eshkol_runtime_init() != 0) {
        eshkol_error("Failed to initialize runtime system");
        return 1;
    }

    // Initialize resource limits from environment variables
    // Supported: ESHKOL_MAX_HEAP, ESHKOL_TIMEOUT_MS, ESHKOL_MAX_STACK, etc.
    eshkol_init_limits_from_env();

    // Helper function to check string suffix (C++17 compatible)
    auto ends_with = [](const std::string& str, const std::string& suffix) {
        if (suffix.size() > str.size()) return false;
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    for (; optind < argc; ++optind) {
        std::string tmp = (const char*) argv[optind];
        if (ends_with(tmp, ".esk"))
            source_files.push_back(argv[optind]);
        else if (ends_with(tmp, ".o"))
            compiled_files.push_back(argv[optind]);
    }

    // NOTE: stdlib.esk auto-loading is now DEPRECATED.
    // The new module system uses (require ...) statements which are processed
    // by process_requires(). User code should explicitly use:
    //   (require stdlib)      - for all standard library
    //   (require core.functional.compose)  - for specific modules
    //
    // The old no_stdlib flag is kept for backwards compatibility but does nothing.
    (void)no_stdlib;  // Suppress unused variable warning

    // First pass: Load source files to check for stdlib requirements
    // (We'll process requires after potentially adding stdlib.o)
    for (const auto &source_file : source_files) {
        load_file_asts(source_file, asts, debug_mode);
    }

    // Auto-link stdlib.o if the source requires stdlib and we're not in library mode
    if (!shared_lib && requires_stdlib(asts)) {
        // Check if stdlib.o is already in compiled_files
        bool has_stdlib = false;
        for (const auto& obj_file : compiled_files) {
            std::string filename = std::filesystem::path(obj_file).filename().string();
            if (filename == "stdlib.o" || filename == "libstdlib.o") {
                has_stdlib = true;
                break;
            }
        }

        if (!has_stdlib) {
            // Try to find stdlib.o automatically
            std::string stdlib_path = find_stdlib_object();
            if (!stdlib_path.empty()) {
                eshkol_info("Auto-linking pre-compiled stdlib: %s", stdlib_path.c_str());
                compiled_files.push_back(strdup(stdlib_path.c_str()));
            }
        }
    }

    // Detect pre-compiled libraries and discover ALL their sub-modules.
    // Every module recursively required by a pre-compiled library is also pre-compiled.
    if (g_lib_dir.empty()) g_lib_dir = find_lib_dir();
    for (const auto& obj_file : compiled_files) {
        std::string filename = std::filesystem::path(obj_file).filename().string();
        if (filename == "stdlib.o" || filename == "libstdlib.o") {
            eshkol_info("Detected pre-compiled stdlib: %s", obj_file);
            // Recursively discover all modules included in stdlib
            collect_all_submodules("stdlib", precompiled_modules, g_lib_dir);
            eshkol_info("Pre-compiled modules: %zu total", precompiled_modules.size());
            // Tell codegen that stdlib is being used (for homoiconic display support)
            eshkol_set_uses_stdlib(1);
        }
        // Future: detect other pre-compiled libraries by naming convention
    }

    // Second pass: Process require statements now that we know about precompiled modules
    for (const auto &source_file : source_files) {
        std::filesystem::path source_path(source_file);
        std::string base_dir = source_path.parent_path().string();
        if (base_dir.empty()) base_dir = ".";

        // Process require statements (new module system)
        process_requires(asts, base_dir, debug_mode);

        // Process imports in the loaded ASTs (legacy)
        process_imports(asts, base_dir, debug_mode);
    }

    // Handle AST dumping if requested
    if (dump_ast && !source_files.empty()) {
        std::string ast_filename;
        if (output) {
            ast_filename = std::string(output) + ".ast";
        } else {
            std::string first_source = source_files[0];
            size_t last_slash = first_source.find_last_of("/\\");
            size_t last_dot = first_source.find_last_of('.');
            std::string base_name;
            if (last_slash != std::string::npos) {
                if (last_dot != std::string::npos && last_dot > last_slash) {
                    base_name = first_source.substr(last_slash + 1, last_dot - last_slash - 1);
                } else {
                    base_name = first_source.substr(last_slash + 1);
                }
            } else {
                if (last_dot != std::string::npos) {
                    base_name = first_source.substr(0, last_dot);
                } else {
                    base_name = first_source;
                }
            }
            ast_filename = base_name + ".ast";
        }

        std::ofstream ast_file(ast_filename);
        if (ast_file.is_open()) {
            for (const auto& ast : asts) {
                ast_file << "=== AST Node ===\n";
                eshkol_ast_pretty_print(&ast, 0);
                ast_file << "=================\n\n";
            }
            ast_file.close();
            eshkol_info("AST dumped to: %s", ast_filename.c_str());
        } else {
            eshkol_error("Failed to open AST file: %s", ast_filename.c_str());
        }
    }

    // Run ownership analysis before code generation
    if (!asts.empty()) {
        eshkol_info("Running ownership analysis...");
        if (!g_ownership_analyzer.analyze(asts)) {
            eshkol_error("Ownership analysis failed:");
            g_ownership_analyzer.printErrors();
            return 1;
        }
        if (debug_mode) {
            eshkol_info("Ownership analysis passed");
        }
    }

    // Run escape analysis for allocation decisions
    if (!asts.empty()) {
        eshkol_info("Running escape analysis...");
        g_escape_analyzer.analyze(asts);
        if (debug_mode) {
            g_escape_analyzer.printAnalysis();
        }
    }

    // Generate LLVM IR if we have ASTs and need compilation or IR output
    // Default behavior is to compile to executable unless only AST dump is requested
    if (!asts.empty()) {
        // Determine module name from first source file or use default
        std::string module_name = "eshkol_module";
        if (!source_files.empty()) {
            std::string source_file = source_files[0];
            size_t last_slash = source_file.find_last_of("/\\");
            size_t last_dot = source_file.find_last_of('.');
            if (last_slash != std::string::npos) {
                if (last_dot != std::string::npos && last_dot > last_slash) {
                    module_name = source_file.substr(last_slash + 1, last_dot - last_slash - 1);
                } else {
                    module_name = source_file.substr(last_slash + 1);
                }
            } else {
                if (last_dot != std::string::npos) {
                    module_name = source_file.substr(0, last_dot);
                } else {
                    module_name = source_file;
                }
            }
        }
        
        // Apply type system flags to global config
        if (strict_types || unsafe_mode) {
            eshkol_config_t cfg = *eshkol_config_get();
            cfg.strict_types = strict_types;
            cfg.unsafe_mode = unsafe_mode;
            eshkol_config_set(&cfg);
        }

        eshkol_info("Generating LLVM IR for module: %s", module_name.c_str());

        // LLVM OPTIMIZATION: Set optimization level before compilation
        if (opt_level > 0) {
            eshkol_set_optimization_level(opt_level);
        }

        // DWARF DEBUG INFO: Enable debug info before IR generation if -g flag was set
        if (debug_info && !source_files.empty()) {
            // Resolve the source file to an absolute path for DWARF
            std::filesystem::path abs_source = std::filesystem::absolute(source_files[0]);
            eshkol_enable_debug_info(abs_source.string().c_str());
        }

        // Generate LLVM IR (use library mode if --shared-lib flag is set)
        LLVMModuleRef llvm_module;
        if (shared_lib) {
            eshkol_info("Using library mode (no main function)");
            llvm_module = eshkol_generate_llvm_ir_library(
                asts.data(),
                asts.size(),
                module_name.c_str()
            );
        } else {
            llvm_module = eshkol_generate_llvm_ir(
                asts.data(),
                asts.size(),
                module_name.c_str()
            );
        }
        
        if (!llvm_module) {
            eshkol_error("Failed to generate LLVM IR");
            return 1;
        }
        
        // Handle different output modes
        if (dump_ir) {
            // Dump IR to file
            std::string ir_filename;
            if (output) {
                ir_filename = std::string(output) + ".ll";
            } else {
                ir_filename = module_name + ".ll";
            }
            
            eshkol_info("Dumping LLVM IR to: %s", ir_filename.c_str());
            if (eshkol_dump_llvm_ir_to_file(llvm_module, ir_filename.c_str()) != 0) {
                eshkol_error("Failed to dump LLVM IR to file");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        }
        
        if (debug_mode) {
            eshkol_info("Generated LLVM IR:");
            eshkol_print_llvm_ir(llvm_module);
        }
        
        /* Emit ESKB bytecode if requested */
        if (eskb_output_path) {
            /* Read source file for bytecode compilation */
            FILE* eskb_src_f = source_files.empty() ? NULL : fopen(source_files[0], "r");
            if (eskb_src_f) {
                fseek(eskb_src_f, 0, SEEK_END);
                long eskb_len = ftell(eskb_src_f);
                fseek(eskb_src_f, 0, SEEK_SET);
                char* eskb_source = (char*)malloc(eskb_len + 1);
                if (eskb_source) {
                    fread(eskb_source, 1, eskb_len, eskb_src_f);
                    eskb_source[eskb_len] = 0;
                    fclose(eskb_src_f);
                    int eskb_result = eshkol_emit_eskb(eskb_source, eskb_output_path);
                    if (eskb_result == 0) {
                        printf("[ESKB] Emitted bytecode to %s\n", eskb_output_path);
                    } else {
                        fprintf(stderr, "WARNING: ESKB emission failed\n");
                    }
                    free(eskb_source);
                } else {
                    fclose(eskb_src_f);
                }
            }
        }

        if (compile_only) {
            // Compile to object file
            std::string obj_filename;
            if (output) {
                obj_filename = std::string(output) + ".o";
            } else {
                obj_filename = module_name + ".o";
            }

            eshkol_info("Compiling to object file: %s", obj_filename.c_str());
            if (eshkol_compile_llvm_ir_to_object(llvm_module, obj_filename.c_str()) != 0) {
                eshkol_error("Object file compilation failed");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }

            // Also emit bitcode for REPL JIT loading (avoids ABI mismatch with addObjectFile)
            std::string bc_filename;
            if (output) {
                bc_filename = std::string(output) + ".bc";
            } else {
                bc_filename = module_name + ".bc";
            }
            eshkol_compile_llvm_ir_to_bitcode(llvm_module, bc_filename.c_str());
        } else if (wasm_output) {
            // Compile to WebAssembly
            std::string wasm_filename;
            if (output) {
                wasm_filename = std::string(output);
                // Add .wasm extension if not present
                if (wasm_filename.size() < 5 || wasm_filename.substr(wasm_filename.size() - 5) != ".wasm") {
                    wasm_filename += ".wasm";
                }
            } else {
                wasm_filename = module_name + ".wasm";
            }

            eshkol_info("Compiling to WebAssembly: %s", wasm_filename.c_str());
            if (eshkol_compile_llvm_ir_to_wasm_file(llvm_module, wasm_filename.c_str()) != 0) {
                eshkol_error("WebAssembly compilation failed");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        } else if (!compile_only) {
            // Default behavior: compile to executable
            // If we have object files to link (like stdlib.o), compile to temp .o first
            // then link everything together
            std::string exe_name = output ? std::string(output) : "a.out";

            if (!compiled_files.empty()) {
                // Compile main program to temp .o, then link with stdlib.o etc.
                // Use mkstemp for safe temp file creation (avoids symlink attacks)
                const char* tmpdir = getenv("TMPDIR");
                if (!tmpdir) tmpdir = P_tmpdir;  // POSIX fallback (usually /tmp)
                std::string temp_template = std::string(tmpdir) + "/eshkol_XXXXXX.o";
                std::vector<char> temp_buf(temp_template.begin(), temp_template.end());
                temp_buf.push_back('\0');
                int tmp_fd = mkstemps(temp_buf.data(), 2);  // 2 = strlen(".o")
                if (tmp_fd < 0) {
                    eshkol_error("Failed to create temp file: %s", strerror(errno));
                    eshkol_dispose_llvm_module(llvm_module);
                    return 1;
                }
                close(tmp_fd);  // eshkol_compile_llvm_ir_to_object opens by name
                std::string temp_obj(temp_buf.data());
                eshkol_info("Compiling to temp object: %s", temp_obj.c_str());
                if (eshkol_compile_llvm_ir_to_object(llvm_module, temp_obj.c_str()) != 0) {
                    eshkol_error("Object file compilation failed");
                    eshkol_dispose_llvm_module(llvm_module);
                    return 1;
                }
                compiled_files.push_back(strdup(temp_obj.c_str()));
                // Set output so the linking section runs
                if (!output) output = strdup(exe_name.c_str());
            } else {
                // No object files to link, compile directly to executable
                eshkol_info("Compiling to executable: %s", exe_name.c_str());

                // Prepare C-style arrays for library paths and libraries
                const char** lib_path_ptrs = nullptr;
                const char** linked_lib_ptrs = nullptr;

                if (!lib_paths.empty()) {
                    lib_path_ptrs = const_cast<const char**>(lib_paths.data());
                }
                if (!linked_libs.empty()) {
                    linked_lib_ptrs = const_cast<const char**>(linked_libs.data());
                }

                if (eshkol_compile_llvm_ir_to_executable(llvm_module, exe_name.c_str(),
                                                       lib_path_ptrs, lib_paths.size(),
                                                       linked_lib_ptrs, linked_libs.size()) != 0) {
                    eshkol_error("Executable compilation failed");
                    eshkol_dispose_llvm_module(llvm_module);
                    return 1;
                }
            }
        }
        
        // Clean up
        eshkol_dispose_llvm_module(llvm_module);
    }

    // Process compiled object files if we have them and an output target
    // Don't link in compile-only mode (-c flag) - user can link manually
    if (!compiled_files.empty() && output && !compile_only) {
        std::string link_cmd = "c++ -fPIE";

        // Add all object files
        for (const auto &compiled_file : compiled_files) {
            link_cmd += " " + std::string(compiled_file);
        }

        // Add library search paths
        for (const auto &lib_path : lib_paths) {
            link_cmd += " -L" + std::string(lib_path);
        }

        // Add libeshkol-static.a (needed for arena functions)
        std::string runtime_lib = find_runtime_library();
        if (!runtime_lib.empty()) {
            link_cmd += " " + runtime_lib;
        } else {
            eshkol_error("Could not find libeshkol-static.a");
            eshkol_error("Searched: ./build/, /usr/local/lib/, /opt/homebrew/lib/, and relative to executable");
            eshkol_error("Please install Eshkol properly or build from source");
            return 1;
        }

        // Add linked libraries
        for (const auto &linked_lib : linked_libs) {
            link_cmd += " -l" + std::string(linked_lib);
        }

        // Add BLAS framework/library (required for libeshkol-static.a BLAS functions)
#ifdef __APPLE__
        // Link with Accelerate framework for BLAS (Apple Silicon optimized)
        link_cmd += " -framework Accelerate";
#elif defined(__linux__)
        // Link with OpenBLAS on Linux
        link_cmd += " -lopenblas";
#endif

        // Add GPU frameworks/libraries (for GPU-accelerated tensor operations)
        // Metal: System framework on macOS, always safe to link (like Accelerate)
        // CUDA: Not a system library, requires explicit detection (handled by cmake)
#ifdef __APPLE__
        // Metal and MetalPerformanceShaders for GPU on macOS
        link_cmd += " -framework Metal -framework MetalPerformanceShaders -framework Foundation";
        link_cmd += " -lobjc";  // Objective-C runtime
#endif

        // Add output
        link_cmd += " -o " + std::string(output) + " -lm";
        
        eshkol_info("Linking object files: %s", link_cmd.c_str());
        int result = system(link_cmd.c_str());

        // Clean up temp object files (mkstemp-created files in TMPDIR)
        for (const auto &compiled_file : compiled_files) {
            std::string file(compiled_file);
            if (file.find("eshkol_") != std::string::npos && file.find(".o") != std::string::npos) {
                std::remove(file.c_str());
            }
        }

        if (result != 0) {
            eshkol_error("Linking failed with exit code %d", result);
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }

        eshkol_info("Successfully created executable: %s", output);
    } else if (!compiled_files.empty() && !compile_only) {
        // Only warn about unused object files if we're not in compile-only mode
        // In compile-only mode, we intentionally don't link
        eshkol_warn("Object files provided but no output specified. Use -o to specify output executable.");
        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
        return 1;
    }

    // Graceful shutdown - calls all registered shutdown hooks
    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
    return 0;
}
