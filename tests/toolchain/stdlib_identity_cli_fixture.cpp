// Standalone driver of the production metadata helpers, not an Eshkol runtime.
#include <eshkol/stdlib_identity_cli.h>
#include <cstring>
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--stdlib-identity") == 0)
        return eshkol::stdlib_identity::print_expected_identity(stdout) ? 0 : 1;
    if (argc == 3 && std::strcmp(argv[1], "--check-stdlib-identity") == 0)
        return eshkol::stdlib_identity::check_file_identity(argv[2], stdout, stderr);
    return 2;
}
