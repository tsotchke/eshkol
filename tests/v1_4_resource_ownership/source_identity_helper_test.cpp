#include <eshkol/strict_source_identity.h>

#include <map>
#include <string>

int main() {
    std::map<std::string, std::string> identities;
    const std::string path = "/virtual/core.net.esk";
    const std::string parsed_bytes = "(define parsed-version 1)";
    const std::string later_disk_bytes = "(define disk-version 2)";
    auto digest = [](const std::string& bytes) { return "digest:" + bytes; };

    if (!eshkol::strict_source_identity::remember(identities, path, digest(parsed_bytes))) return 1;
    // A same-path second observation with different content cannot replace the
    // identity bound to the bytes originally passed to the parser.
    if (eshkol::strict_source_identity::remember(identities, path, digest(later_disk_bytes))) return 2;
    if (identities.at(path) != digest(parsed_bytes)) return 3;

    std::string changed_path;
    if (eshkol::strict_source_identity::unchanged(identities,
        [&](const std::string&) { return digest(later_disk_bytes); }, &changed_path)) return 4;
    if (changed_path != path) return 5;
    if (!eshkol::strict_source_identity::unchanged(identities,
        [&](const std::string&) { return digest(parsed_bytes); })) return 6;
}
