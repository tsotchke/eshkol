/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file source_paths_test.cpp
 * @brief Contract tests for recorded source-path normalization (ADR-0021).
 *
 * The claim under test is the one the shipped artifacts depend on: a path
 * that is RECORDED — interned into the `NodeId` substrate's file table, put
 * in the parse context that diagnostics print, or embedded by the backend as
 * a runtime location — is never the absolute host path, while the host path
 * is still available for reading the file behind a caret.
 */

#include <eshkol/eshkol.h>
#include <eshkol/frontend/source_paths.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

static int g_failures = 0;

static void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

int main() {
    namespace fs = std::filesystem;

    /* A path inside this repository records as a repository-relative one. */
    const fs::path source_file = fs::path(__FILE__);
    const std::string host = fs::weakly_canonical(source_file).string();
    const char* display = eshkol_source_path_display(host.c_str());
    check(display != nullptr, "a host path has a display spelling");
    check(display && !eshkol_source_path_is_host_absolute(display),
          "the display spelling is not an absolute host path");
    check(display && std::strstr(display, "source_paths_test.cpp") != nullptr,
          "the display spelling still names the file");
    check(display && std::strcmp(display, "tests/frontend/source_paths_test.cpp") == 0,
          "a file in the repository records as its repository-relative path");

    /* Stability: the same input gives the same pointer, so a recorded path
     * can be held for the process. */
    check(eshkol_source_path_display(host.c_str()) == display,
          "the display spelling is memoized and stable");

    /* A path outside anything nameable keeps only its file name. */
    const char* outside = eshkol_source_path_display("/nonexistent-root-xyz/someone/work/a.esk");
    check(outside && std::strcmp(outside, "a.esk") == 0,
          "a path outside every root records as its file name alone");

    /* Pseudo-names and relative paths pass through. */
    check(std::strcmp(eshkol_source_path_display("<unknown>"), "<unknown>") == 0,
          "a pseudo-name is unchanged");
    check(std::strcmp(eshkol_source_path_display("./examples/hello.esk"),
                      "examples/hello.esk") == 0,
          "a relative path keeps its spelling, lexically normalized");
    check(eshkol_source_path_display(nullptr) == nullptr, "NULL stays NULL");

    /* The interned table records the display path and keeps the host path. */
    const uint32_t id = eshkol_intern_source_file(host.c_str());
    check(id != 0, "the host path interns");
    const char* interned = eshkol_source_file_name(id);
    check(interned && std::strcmp(interned, display) == 0,
          "eshkol_source_file_name returns the display path");
    const char* interned_host = eshkol_source_file_host_path(id);
    check(interned_host && std::strcmp(interned_host, host.c_str()) == 0,
          "eshkol_source_file_host_path returns the path the file is read from");
    check(eshkol_intern_source_file(host.c_str()) == id, "interning is stable");

    /* The parse context — what diagnostics print — is the display path. */
    eshkol_set_parse_source_context(host.c_str());
    const char* context = eshkol_get_parse_source_context();
    check(context && !eshkol_source_path_is_host_absolute(context),
          "the parse context is never an absolute host path");
    check(context && std::strcmp(context, display) == 0,
          "the parse context is the display path");

    if (g_failures) {
        std::printf("FAIL: source path normalization (%d failures)\n", g_failures);
        return 1;
    }
    std::printf("PASS: source path normalization\n");
    return 0;
}
