#include <eshkol/frontend/workspace.h>

#include <eshkol/eshkol.h>
#include <eshkol/module_resolver.h>
#include <eshkol/platform_runtime.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <set>

namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(c) << std::dec;
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

std::string canonical_path(const std::string& path) {
    std::error_code error;
    auto result = std::filesystem::weakly_canonical(path, error);
    return error ? path : result.string();
}

std::string module_name_from_path(const std::string& path) {
    std::filesystem::path p(path);
    std::string name = p.stem().string();
    std::vector<std::string> parts;
    const auto parent = p.parent_path();
    auto anchor = parent.end();
    for (auto it = parent.begin(); it != parent.end(); ++it) {
        if (it->string() == "lib" || it->string() == "tests" ||
            it->string() == "examples") anchor = it;
    }
    const auto start = anchor == parent.end() ? parent.begin() :
        (anchor->string() == "lib" ? std::next(anchor) : anchor);
    for (auto it = start; it != parent.end(); ++it) {
        if (it->string() != ".") parts.push_back(it->string());
    }
    if (!parts.empty()) {
        std::string joined;
        for (const auto& part : parts) {
            if (!joined.empty()) joined += ".";
            joined += part;
        }
        name = joined + "." + name;
    }
    return name;
}

struct GraphBuilder {
    const eshkol::frontend::WorkspaceResolver& resolver;
    std::map<std::string, eshkol::frontend::WorkspaceModule> modules;
    std::set<std::string> visiting;
    std::vector<std::string> diagnostics;

    void add(const std::string& path) {
        const std::string canonical = canonical_path(path);
        if (modules.find(canonical) != modules.end()) return;
        if (!visiting.insert(canonical).second) {
            diagnostics.push_back("module cycle: " + canonical);
            return;
        }

        std::ifstream input(canonical, std::ios::binary);
        if (!input.is_open()) {
            diagnostics.push_back("cannot open module: " + canonical);
            visiting.erase(canonical);
            return;
        }
        std::string source((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
        eshkol::platform::ScopedRequiringFile scope(canonical);
        eshkol_reset_parse_errors();
        eshkol_set_parse_source_context(canonical.c_str());
        eshkol_reset_parse_line_counter();
        std::istringstream stream(source);
        std::vector<eshkol_ast_t> forms;
        while (true) {
            eshkol_ast_t form = eshkol_parse_next_ast_from_stream(stream);
            if (form.type == ESHKOL_INVALID) break;
            forms.push_back(form);
        }
        if (eshkol_parse_had_error()) {
            diagnostics.push_back("syntax error in module: " + canonical);
        }

        eshkol::frontend::WorkspaceModule module;
        module.name = module_name_from_path(canonical);
        module.id = eshkol::frontend::WorkspaceResolver::stable_module_id(module.name);
        module.path = canonical;
        std::set<std::string> seen_dependencies;
        std::function<void(const eshkol_ast_t&)> visit = [&](const eshkol_ast_t& form) {
            if (form.type != ESHKOL_OP) return;
            if (form.operation.op == ESHKOL_SEQUENCE_OP) {
                for (uint64_t i = 0; i < form.operation.sequence_op.num_expressions; ++i)
                    visit(form.operation.sequence_op.expressions[i]);
                return;
            }
            if (form.operation.op == ESHKOL_REQUIRE_OP) {
                for (uint64_t i = 0; i < form.operation.require_op.num_modules; ++i) {
                    const char* reference = form.operation.require_op.module_names[i];
                    if (!reference) continue;
                    const std::string dependency = resolver.resolve_path(
                        reference, canonical);
                    if (dependency.empty()) {
                        diagnostics.push_back("module '" + std::string(reference) +
                                              "' not found from " + canonical);
                        continue;
                    }
                    add(dependency);
                    const std::string dependency_name = module_name_from_path(dependency);
                    if (seen_dependencies.insert(dependency_name).second) {
                        module.dependencies.push_back(
                            eshkol::frontend::WorkspaceResolver::stable_module_id(dependency_name));
                    }
                }
            } else if (form.operation.op == ESHKOL_IMPORT_OP &&
                       form.operation.import_op.path) {
                const std::string dependency = resolver.resolve_path(
                    form.operation.import_op.path, canonical);
                if (dependency.empty()) {
                    diagnostics.push_back("module '" +
                                          std::string(form.operation.import_op.path) +
                                          "' not found from " + canonical);
                    return;
                }
                add(dependency);
                const std::string dependency_name = module_name_from_path(dependency);
                if (seen_dependencies.insert(dependency_name).second) {
                    module.dependencies.push_back(
                        eshkol::frontend::WorkspaceResolver::stable_module_id(dependency_name));
                }
            }
        };
        for (const auto& form : forms) visit(form);
        std::sort(module.dependencies.begin(), module.dependencies.end());
        modules.emplace(canonical, std::move(module));
        visiting.erase(canonical);
    }
};

}  // namespace

namespace eshkol::frontend {

WorkspaceResolver::WorkspaceResolver(std::string module_root)
    : module_root_(std::move(module_root)) {
    if (module_root_.empty()) module_root_ = platform::module_source_root().path.string();
}

std::string WorkspaceResolver::resolve_path(const std::string& reference,
                                            const std::string& requiring_file) const {
    std::string output(4096, '\0');
    std::filesystem::path base = requiring_file.empty()
        ? std::filesystem::path(".") : std::filesystem::path(requiring_file).parent_path();
    if (base.empty()) base = ".";
    if (!eshkol_resolve_module_source_path_c(reference.c_str(), base.string().c_str(),
                                             module_root_.c_str(), output.data(), output.size())) {
        return {};
    }
    output.resize(std::strlen(output.c_str()));
    return output;
}

ModuleId WorkspaceResolver::stable_module_id(const std::string& module_name) {
    uint32_t hash = 2166136261u;
    for (unsigned char c : module_name) hash = (hash ^ c) * 16777619u;
    return hash == ESHKOL_MODULE_ID_NONE ? 1 : hash;
}

SymbolId WorkspaceResolver::stable_symbol_id(const std::string& module_name,
                                             const std::string& symbol_name) {
    return stable_module_id(module_name + "\0" + symbol_name);
}

WorkspaceCheckResult WorkspaceResolver::check(const std::string& entry_path) const {
    WorkspaceCheckResult result;
    const std::string entry = resolve_path(entry_path, {});
    const std::string actual = entry.empty() ? canonical_path(entry_path) : entry;
    GraphBuilder builder{*this};
    builder.add(actual);
    result.diagnostics = std::move(builder.diagnostics);
    for (auto& item : builder.modules) result.modules.push_back(std::move(item.second));

    std::ifstream input(actual, std::ios::binary);
    if (!input.is_open()) {
        result.diagnostics.push_back("cannot open entry source: " + actual);
        return result;
    }
    std::string source((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
    platform::ScopedRequiringFile scope(actual);
    eshkol_reset_parse_errors();
    eshkol_set_parse_source_context(actual.c_str());
    eshkol_reset_parse_line_counter();
    std::istringstream stream(source);
    std::vector<eshkol_ast_t> forms;
    while (true) {
        eshkol_ast_t form = eshkol_parse_next_ast_from_stream(stream);
        if (form.type == ESHKOL_INVALID) break;
        forms.push_back(form);
    }
    frontend::BindingResolver bindings(
        result.modules.empty() ? ESHKOL_MODULE_ID_NONE : result.modules.front().id);
    result.bindings = bindings.resolve(forms);
    if (eshkol_parse_had_error()) result.diagnostics.push_back("entry source did not parse");
    for (const auto& diagnostic : result.bindings.diagnostics) {
        result.diagnostics.push_back(diagnostic.message);
    }
    return result;
}

WorkspaceCheckResult WorkspaceResolver::check_source(const std::string& entry_path,
                                                     const std::string& source) const {
    WorkspaceCheckResult result;
    WorkspaceModule module;
    module.name = module_name_from_path(canonical_path(entry_path));
    module.id = stable_module_id(module.name);
    module.path = canonical_path(entry_path);
    result.modules.push_back(module);

    platform::ScopedRequiringFile scope(entry_path);
    eshkol_reset_parse_errors();
    eshkol_set_parse_source_context(entry_path.c_str());
    eshkol_reset_parse_line_counter();
    std::istringstream stream(source);
    std::vector<eshkol_ast_t> forms;
    while (true) {
        eshkol_ast_t form = eshkol_parse_next_ast_from_stream(stream);
        if (form.type == ESHKOL_INVALID) break;
        forms.push_back(form);
    }
    if (eshkol_parse_had_error()) result.diagnostics.push_back("source did not parse");

    BindingResolver bindings(module.id);
    result.bindings = bindings.resolve(forms);
    for (const auto& form : forms) {
        if (form.type != ESHKOL_OP || form.operation.op != ESHKOL_REQUIRE_OP) continue;
        for (uint64_t i = 0; i < form.operation.require_op.num_modules; ++i) {
            const char* reference = form.operation.require_op.module_names[i];
            if (reference && resolve_path(reference, entry_path).empty())
                result.diagnostics.push_back("module '" + std::string(reference) + "' not found");
        }
    }
    for (const auto& diagnostic : result.bindings.diagnostics)
        result.diagnostics.push_back(diagnostic.message);
    return result;
}

std::string WorkspaceCheckResult::json() const {
    std::ostringstream out;
    out << "{\"schema\":\"eshkol.workspace-check.v1\",\"ok\":"
        << (ok() ? "true" : "false") << ",\"modules\":[";
    for (size_t i = 0; i < modules.size(); ++i) {
        if (i) out << ',';
        out << "{\"id\":" << modules[i].id << ",\"name\":\""
            << json_escape(modules[i].name) << "\",\"dependencies\":[";
        for (size_t j = 0; j < modules[i].dependencies.size(); ++j) {
            if (j) out << ',';
            out << modules[i].dependencies[j];
        }
        out << "]}";
    }
    out << "],\"diagnostics\":[";
    for (size_t i = 0; i < diagnostics.size(); ++i) {
        if (i) out << ',';
        out << "\"" << json_escape(diagnostics[i]) << "\"";
    }
    out << "],\"identifiers\":" << bindings.identifiers
        << ",\"resolved\":" << bindings.resolved << "}\n";
    return out.str();
}

std::string WorkspaceCheckResult::markdown() const {
    std::ostringstream out;
    out << "Workspace check: " << (ok() ? "PASS" : "FAIL") << "\n\n";
    out << "Modules: " << modules.size() << "\n";
    for (const auto& module : modules) {
        out << "- " << module.name << " (ModuleId " << module.id << ")";
        if (!module.dependencies.empty()) {
            out << " ->";
            for (ModuleId dependency : module.dependencies) out << ' ' << dependency;
        }
        out << '\n';
    }
    out << "Identifiers: " << bindings.identifiers << "\n"
        << "Resolved: " << bindings.resolved << "\n";
    for (const auto& diagnostic : diagnostics) out << "ERROR: " << diagnostic << '\n';
    return out.str();
}

}  // namespace eshkol::frontend

extern "C" int eshkol_resolve_module_source_path_c(const char* module_name,
                                                    const char* base_dir,
                                                    const char* lib_dir,
                                                    char* output,
                                                    size_t output_size) {
    if (!module_name || !output || output_size == 0) return 0;
    const std::string resolved = eshkol::platform::resolve_module_source_path(
        module_name, base_dir ? base_dir : ".", lib_dir ? lib_dir : "");
    if (resolved.empty() || resolved.size() + 1 > output_size) return 0;
    std::copy(resolved.begin(), resolved.end(), output);
    output[resolved.size()] = '\0';
    return 1;
}
