#ifndef ESHKOL_FRONTEND_WORKSPACE_H
#define ESHKOL_FRONTEND_WORKSPACE_H

#include <eshkol/frontend/semantic_identity.h>
#include <string>
#include <vector>

namespace eshkol::frontend {

struct WorkspaceModule {
    ModuleId id = ESHKOL_MODULE_ID_NONE;
    std::string name;
    std::string path;
    std::vector<ModuleId> dependencies;
};

struct WorkspaceCheckResult {
    std::vector<WorkspaceModule> modules;
    ResolutionResult bindings;
    std::vector<std::string> diagnostics;

    bool ok() const { return diagnostics.empty() && bindings.ok(); }
    std::string json() const;
    std::string markdown() const;
};

/*
 * LLVM-free source analysis shared by `eshkol check`, LSP, documentation
 * tooling, and the VM's C resolver adapter. It parses source and constructs a
 * deterministic module graph but never evaluates project code or initializes
 * an LLVM target.
 */
class WorkspaceResolver {
public:
    explicit WorkspaceResolver(std::string module_root = {});

    std::string resolve_path(const std::string& reference,
                             const std::string& requiring_file = {}) const;
    WorkspaceCheckResult check(const std::string& entry_path) const;
    WorkspaceCheckResult check_source(const std::string& entry_path,
                                       const std::string& source) const;

    static ModuleId stable_module_id(const std::string& module_name);
    static SymbolId stable_symbol_id(const std::string& module_name,
                                     const std::string& symbol_name);

private:
    std::string module_root_;
};

}  // namespace eshkol::frontend

#endif
