/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#include <eshkol/llvm_backend.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <filesystem>
#include <mach-o/dyld.h>  // For _NSGetExecutablePath on macOS

#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <sstream>

static struct option long_options[] = {
    {"help", no_argument, nullptr, 'h'},
    {"debug", no_argument, nullptr, 'd'},
    {"dump-ast", no_argument, nullptr, 'a'},
    {"dump-ir", no_argument, nullptr, 'i'},
    {"output", required_argument, nullptr, 'o'},
    {"compile-only", no_argument, nullptr, 'c'},
    {"shared-lib", no_argument, nullptr, 's'},
    {"lib", required_argument, nullptr, 'l'},
    {"lib-path", required_argument, nullptr, 'L'},
    {"no-stdlib", no_argument, nullptr, 'n'},
    {0, 0, 0, 0}
};

// Set to track imported files (prevent circular imports)
static std::set<std::string> imported_files;

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
                        "",  // TODO: add location tracking
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
        "Usage: eshkol-run [options] <input.esk|input.o> [input.esk|input.o]\n\n"
        "\t--help:[-h] = Print this help message.\n"
        "\t--debug:[-d] = Debugging information added inside the program.\n"
        "\t--dump-ast:[-a] = Dumps the AST into a .ast file.\n"
        "\t--dump-ir:[-i] = Dumps the IR into a .ll file.\n"
        "\t--output:[-o] = Outputs into a binary file.\n"
        "\t--compile-only:[-c] = Compiles into an intermediate object file.\n"
        "\t--shared-lib:[-s] = Compiles it into a shared library.\n"
        "\t--lib:[-l] = Links a shared library to the resulting executable.\n"
        "\t--lib-path:[-L] = Adds a directory to the library search path.\n"
        "\t--no-stdlib:[-n] = Do not auto-load the standard library.\n\n"
        "This is an early developer release (%s) of the Eshkol Compiler/Interpreter.\n",
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
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        // Try macOS method
        uint32_t size = sizeof(exe_path);
        if (_NSGetExecutablePath(exe_path, &size) == 0) {
            std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
            stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "../lib/stdlib.esk").string());
            stdlib_paths.insert(stdlib_paths.begin(), (exe_dir / "stdlib.esk").string());
        }
    } else {
        exe_path[len] = '\0';
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
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        // Try macOS method
        uint32_t size = sizeof(exe_path);
        if (_NSGetExecutablePath(exe_path, &size) == 0) {
            std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
            lib_dirs.insert(lib_dirs.begin(), (exe_dir / "../lib").string());
            lib_dirs.insert(lib_dirs.begin(), (exe_dir / "lib").string());
        }
    } else {
        exe_path[len] = '\0';
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
                    free(ast->variable.id);
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
                case ESHKOL_AND_OP:   // and uses call_op/sequence structure
                case ESHKOL_OR_OP:    // or uses call_op/sequence structure
                    if (ast->operation.call_op.func) {
                        update_ast_references(ast->operation.call_op.func, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        update_ast_references(&ast->operation.call_op.variables[i], rename_map);
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
                            free(ast->operation.set_op.name);
                            ast->operation.set_op.name = strdup(it->second.c_str());
                        }
                    }
                    update_ast_references(ast->operation.set_op.value, rename_map);
                    break;

                case ESHKOL_WITH_REGION_OP:
                    // Update references in with-region body
                    if (ast->operation.with_region_op.body) {
                        update_ast_references(ast->operation.with_region_op.body, rename_map);
                    }
                    break;

                case ESHKOL_OWNED_OP:
                    update_ast_references(ast->operation.owned_op.value, rename_map);
                    break;

                case ESHKOL_MOVE_OP:
                    update_ast_references(ast->operation.move_op.value, rename_map);
                    break;

                case ESHKOL_BORROW_OP:
                    update_ast_references(ast->operation.borrow_op.value, rename_map);
                    if (ast->operation.borrow_op.body) {
                        update_ast_references(ast->operation.borrow_op.body, rename_map);
                    }
                    break;

                case ESHKOL_SHARED_OP:
                    update_ast_references(ast->operation.shared_op.value, rename_map);
                    break;

                case ESHKOL_WEAK_REF_OP:
                    update_ast_references(ast->operation.weak_ref_op.value, rename_map);
                    break;

                default:
                    // Other operations - no recursive update needed or not applicable
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
                    free(ast.operation.define_op.name);
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
    uint8_t no_stdlib = 0;

    std::vector<char*> source_files;
    std::vector<char*> compiled_files;
    std::vector<char*> linked_libs;
    std::vector<char*> lib_paths;

    std::vector<eshkol_ast_t> asts;

    char *output = nullptr;

    if (argc == 1) print_help(1);

    while ((ch = getopt_long(argc, argv, "hdaio:csl:L:n", long_options, nullptr)) != -1) {
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
            // TODO: Implement shared library support
            eshkol_warn("Shared library support not yet implemented");
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
        default:
            print_help(1);
        }
    }

    if (optind == argc) print_help(1);

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

    // Load user source files
    for (const auto &source_file : source_files) {
        // Get base directory for resolving imports
        std::filesystem::path source_path(source_file);
        std::string base_dir = source_path.parent_path().string();
        if (base_dir.empty()) base_dir = ".";

        // Load the file
        load_file_asts(source_file, asts, debug_mode);

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
        
        eshkol_info("Generating LLVM IR for module: %s", module_name.c_str());
        
        // Generate LLVM IR
        LLVMModuleRef llvm_module = eshkol_generate_llvm_ir(
            asts.data(), 
            asts.size(), 
            module_name.c_str()
        );
        
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
        } else if (!compile_only && !dump_ir && !dump_ast) {
            // Default behavior: compile to executable
            // If we have object files to link (like stdlib.o), compile to temp .o first
            // then link everything together
            std::string exe_name = output ? std::string(output) : "a.out";

            if (!compiled_files.empty()) {
                // Compile main program to temp .o, then link with stdlib.o etc.
                std::string temp_obj = exe_name + ".tmp.o";
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
    if (!compiled_files.empty() && output) {
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
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            std::string cwd_str = std::string(cwd);
            std::string lib_path;
            if (cwd_str.length() >= 5 && cwd_str.substr(cwd_str.length() - 5) == "build") {
                lib_path = cwd_str + "/libeshkol-static.a";
            } else {
                lib_path = cwd_str + "/build/libeshkol-static.a";
            }
            link_cmd += " " + lib_path;
        }

        // Add linked libraries
        for (const auto &linked_lib : linked_libs) {
            link_cmd += " -l" + std::string(linked_lib);
        }

        // Add output
        link_cmd += " -o " + std::string(output) + " -lm";
        
        eshkol_info("Linking object files: %s", link_cmd.c_str());
        int result = system(link_cmd.c_str());

        // Clean up temp object files
        for (const auto &compiled_file : compiled_files) {
            std::string file(compiled_file);
            if (file.find(".tmp.o") != std::string::npos) {
                std::remove(file.c_str());
            }
        }

        if (result != 0) {
            eshkol_error("Linking failed with exit code %d", result);
            return 1;
        }

        eshkol_info("Successfully created executable: %s", output);
    } else if (!compiled_files.empty()) {
        eshkol_warn("Object files provided but no output specified. Use -o to specify output executable.");
        return 1;
    }

    return 0;
}
