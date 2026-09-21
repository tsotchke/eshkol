#include <eshkol/frontend/workspace.h>

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    eshkol::frontend::WorkspaceResolver resolver;
    const auto result = resolver.check(argv[1]);
    if (!result.ok()) {
        std::cerr << result.markdown();
        return 1;
    }
    const auto module = eshkol::frontend::WorkspaceResolver::stable_module_id("demo");
    const auto symbol = eshkol::frontend::WorkspaceResolver::stable_symbol_id("demo", "value");
    if (module == ESHKOL_MODULE_ID_NONE || symbol == ESHKOL_SYMBOL_ID_NONE) return 1;
    std::cout << "PASS: LLVM-free workspace resolver\n";
    std::cout << "modules=" << result.modules.size()
              << " identifiers=" << result.bindings.identifiers << "\n";
    return 0;
}
