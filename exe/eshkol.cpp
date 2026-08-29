#include <eshkol/frontend/workspace.h>
#include <eshkol/platform_runtime.h>

#include <iostream>
#include <string>
#include <cstdlib>

static void print_help(int status) {
    std::cout <<
        "Usage: eshkol <command> [options] [path]\n\n"
        "Commands:\n"
        "  check [path]             Parse and resolve a workspace without executing it\n"
        "                           (use --format json for the machine contract)\n"
        "  doc modules [path]      Print the resolver-owned module graph\n"
        "  help                    Print this help message\n"
        "\n"
        "Options for check:\n"
        "  --format human|json     Select deterministic output (default: human)\n"
        "  -h, --help              Print this help message\n";
    std::exit(status);
}

int main(int argc, char** argv) {
    if (argc < 2) print_help(1);
    const std::string command = argv[1];
    if (command == "help" || command == "--help" || command == "-h") print_help(0);
    bool documentation = command == "doc";
    if (documentation && argc >= 3 && std::string(argv[2]) == "modules") {
    } else if (command != "check") {
        std::cerr << "eshkol: unknown command '" << command << "'\n";
        print_help(1);
    }

    std::string path = ".";
    std::string format = "human";
    const int first_option = documentation ? 3 : 2;
    for (int i = first_option; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") print_help(0);
        if (arg == "--format" && i + 1 < argc) {
            format = argv[++i];
        } else if (arg.rfind("--format=", 0) == 0) {
            format = arg.substr(9);
        } else if (!arg.empty() && arg[0] != '-') {
            path = arg;
        } else {
            std::cerr << "eshkol check: unknown option '" << arg << "'\n";
            return 2;
        }
    }
    if (format != "human" && format != "json") {
        std::cerr << "eshkol check: --format must be human or json\n";
        return 2;
    }

    eshkol::frontend::WorkspaceResolver resolver;
    const auto result = resolver.check(path);
    std::cout << (format == "json" ? result.json() : result.markdown());
    return result.ok() ? 0 : 1;
}
