#include <cstdio>
#include <map>
#include <vector>

#include <eshkol/frontend/diagnostic.h>
#include <eshkol/frontend/semantic_identity.h>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

void collect_diagnostic(const eshkol_diagnostic_v1_t* diagnostic, void* userdata) {
    if (!diagnostic || !userdata) return;
    int* count = static_cast<int*>(userdata);
    if (diagnostic->schema == ESHKOL_DIAGNOSTIC_V1_SCHEMA &&
        diagnostic->severity == ESHKOL_DIAGNOSTIC_ERROR) {
        ++*count;
    }
}

}  // namespace

int main() {
    const eshkol_node_id_t declaration = eshkol_node_id_new(1, 3, 1);
    const eshkol_node_id_t use = eshkol_node_id_new(1, 4, 5);

    eshkol_ast_t define{};
    define.type = ESHKOL_OP;
    define.node_id = declaration;
    define.operation.op = ESHKOL_DEFINE_OP;
    define.operation.define_op.name = const_cast<char*>("answer");
    define.operation.define_op.value = nullptr;
    define.operation.define_op.is_function = 0;

    eshkol_ast_t reference{};
    reference.type = ESHKOL_VAR;
    reference.node_id = use;
    reference.variable.id = const_cast<char*>("answer");

    eshkol::frontend::BindingResolver resolver(7);
    const auto result = resolver.resolve({define, reference});
    check(result.ok(), "resolver accepts a declared binding");
    check(result.identifiers >= 1, "resolver counts identifier nodes");

    const eshkol_binding_id_t declaration_id =
        eshkol_binding_id_for_node(declaration);
    const eshkol_binding_id_t use_id = eshkol_binding_id_for_node(use);
    check(declaration_id != ESHKOL_BINDING_ID_NONE, "definition has BindingId");
    check(use_id == declaration_id, "use and definition share BindingId");

    eshkol_binding_info_t info{};
    check(eshkol_binding_info(use_id, &info), "BindingId resolves to metadata");
    check(info.module == 7 && info.kind == ESHKOL_BINDING_VALUE,
          "binding metadata preserves module and kind");

    eshkol_typed_expr_info_t typed{};
    check(eshkol_typed_expr_info(use, &typed), "typed side table uses NodeId");
    check(typed.type.kind == ESHKOL_TYPE_DYN,
          "unannotated expression is Dyn, not Value");
    check(eshkol_nominal_type_intern("Int") == eshkol_nominal_type_intern("Int"),
          "nominal types are interned");

    eshkol::frontend::ImportResolver imports;
    eshkol::frontend::LibraryInterface interface;
    interface.module = 11;
    interface.name = "demo.math";
    interface.exports.emplace("x", declaration_id);
    interface.exports.emplace("y", use_id);
    imports.define(interface);
    std::map<std::string, eshkol_binding_id_t> view;
    std::string import_error;
    const auto transformed = eshkol::frontend::ImportSet::prefixing(
        eshkol::frontend::ImportSet::renaming(
            eshkol::frontend::ImportSet::library_named("demo.math"),
            {{"x", "z"}}), "p-");
    check(imports.evaluate(transformed, &view, &import_error),
          "recursive import-set algebra resolves");
    check(view["p-z"] == declaration_id && view["p-y"] == use_id,
          "prefix and rename preserve BindingId identity");

    int diagnostic_count = 0;
    eshkol_diagnostic_reset_v1();
    eshkol_diagnostic_set_sink_v1(collect_diagnostic, &diagnostic_count);
    eshkol_diagnostic_emit_v1(ESHKOL_DIAGNOSTIC_ERROR, use, "E-BIND", "test diagnostic");
    check(diagnostic_count == 1 && eshkol_diagnostic_count_v1() == 1,
          "Diagnostic v1 reaches the shared sink");
    eshkol_diagnostic_set_sink_v1(nullptr, nullptr);

    if (failures != 0) return 1;
    std::puts("PASS: semantic identity and Diagnostic v1");
    return 0;
}
