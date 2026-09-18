/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file source_paths.cpp
 * @brief Display-path normalization for recorded source locations (ADR-0020).
 *
 * See inc/eshkol/frontend/source_paths.h for the rule set. The table is
 * process-lifetime for the same reason the interned file table is: an id
 * stamped during parsing must still resolve at codegen time.
 */

#include <eshkol/frontend/source_paths.h>

#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::mutex g_mutex;

std::deque<std::string>& storage() {
    static std::deque<std::string> s;  /* never invalidates c_str() */
    return s;
}

std::unordered_map<std::string, const char*>& memo() {
    static std::unordered_map<std::string, const char*> m;
    return m;
}

/** Module roots from ESHKOL_PATH, resolved once. */
const std::vector<fs::path>& module_roots() {
    static const std::vector<fs::path> roots = [] {
        std::vector<fs::path> out;
        const char* raw = std::getenv("ESHKOL_PATH");
        if (!raw || !*raw) return out;
        std::string value(raw);
#ifdef _WIN32
        const char separator = ';';
#else
        const char separator = ':';
#endif
        size_t start = 0;
        while (start <= value.size()) {
            size_t end = value.find(separator, start);
            if (end == std::string::npos) end = value.size();
            std::string entry = value.substr(start, end - start);
            if (!entry.empty()) {
                std::error_code ec;
                fs::path canonical = fs::weakly_canonical(fs::path(entry), ec);
                out.push_back(ec ? fs::path(entry).lexically_normal() : canonical);
            }
            start = end + 1;
        }
        return out;
    }();
    return roots;
}

/** Relative spelling of @p path under @p root, or empty when it is not under it. */
std::string relative_to(const fs::path& path, const fs::path& root) {
    if (root.empty()) return {};
    const fs::path relative = path.lexically_relative(root);
    if (relative.empty()) return {};
    const std::string text = relative.generic_string();
    if (text == "." || text.rfind("..", 0) == 0) return {};  /* not under root */
    return text;
}

/** Nearest ancestor of @p path that looks like a project root. */
fs::path project_root_of(const fs::path& path) {
    std::error_code ec;
    for (fs::path dir = path.parent_path(); !dir.empty(); dir = dir.parent_path()) {
        if (fs::exists(dir / ".git", ec) || fs::exists(dir / "CMakeLists.txt", ec)) {
            return dir;
        }
        if (!dir.has_relative_path()) break;  /* reached the filesystem root */
    }
    return {};
}

std::string compute_display(const std::string& input) {
    /* Pseudo-names ("<unknown>", "<stdin>", "<string>") name no file. */
    if (input.front() == '<') return input;

    const fs::path raw(input);
    const fs::path normal = raw.lexically_normal();
    if (!normal.is_absolute()) {
        const std::string text = normal.generic_string();
        return text.rfind("./", 0) == 0 ? text.substr(2) : text;
    }

    std::error_code ec;
    fs::path canonical = fs::weakly_canonical(normal, ec);
    if (ec) canonical = normal;

    for (const fs::path& root : module_roots()) {
        std::string relative = relative_to(canonical, root);
        if (!relative.empty()) return relative;
    }

    std::string relative = relative_to(canonical, project_root_of(canonical));
    if (!relative.empty()) return relative;

    const fs::path cwd = fs::current_path(ec);
    if (!ec) {
        relative = relative_to(canonical, fs::weakly_canonical(cwd, ec));
        if (!relative.empty()) return relative;
    }

    /* Outside everything we can name relative to: the file name alone. It is
     * still a location a reader can act on, and it carries no host layout. */
    return canonical.filename().generic_string();
}

}  // namespace

extern "C" const char* eshkol_source_path_display(const char* path) {
    if (!path || !*path) return path;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto& table = memo();
    const std::string key(path);
    auto it = table.find(key);
    if (it != table.end()) return it->second;

    std::string display;
    try {
        display = compute_display(key);
    } catch (const std::exception&) {
        display = key;  /* a filesystem error must never lose the location */
    }
    if (display.empty()) display = key;

    storage().push_back(std::move(display));
    const char* stable = storage().back().c_str();
    table.emplace(key, stable);
    return stable;
}

extern "C" bool eshkol_source_path_is_host_absolute(const char* path) {
    if (!path || !*path) return false;
    if (path[0] == '<') return false;
    return fs::path(path).is_absolute();
}
