#pragma once

#include <cctype>
#include <limits>
#include <string>

inline bool eshkol_parse_content_length(const std::string& raw_value, size_t& content_length) {
    size_t start = 0;
    while (start < raw_value.size() && std::isspace(static_cast<unsigned char>(raw_value[start]))) {
        ++start;
    }

    size_t end = raw_value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(raw_value[end - 1]))) {
        --end;
    }

    if (start == end) {
        return false;
    }

    const std::string trimmed = raw_value.substr(start, end - start);
    for (char ch : trimmed) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }

    try {
        const unsigned long long parsed = std::stoull(trimmed);
        if (parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        content_length = static_cast<size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}
