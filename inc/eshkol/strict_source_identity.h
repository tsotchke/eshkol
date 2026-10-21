#pragma once

#include <map>
#include <string>

namespace eshkol::strict_source_identity {

inline bool remember(std::map<std::string, std::string>& identities,
                     const std::string& path, const std::string& digest) {
    auto [entry, inserted] = identities.emplace(path, digest);
    return inserted || entry->second == digest;
}

template <class DigestReader>
inline bool unchanged(const std::map<std::string, std::string>& identities,
                      DigestReader read_digest, std::string* changed_path = nullptr) {
    for (const auto& [path, digest] : identities) {
        if (read_digest(path) != digest) {
            if (changed_path) *changed_path = path;
            return false;
        }
    }
    return true;
}

}  // namespace eshkol::strict_source_identity
