/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef ESHKOL_FRONTEND_SOURCE_PATHS_H
#define ESHKOL_FRONTEND_SOURCE_PATHS_H

/**
 * @file source_paths.h
 * @brief The one place a source path is turned into the spelling that may be
 * recorded or emitted (ADR-0021).
 *
 * A compiler reads files by their host path and *records* where code came
 * from in several places: the `NodeId -> SourceSpan` substrate's interned
 * file table, diagnostics, coverage records, and string constants the backend
 * embeds in generated objects and WebAssembly modules so the runtime can name
 * a location at error time. The host path is right for reading and wrong for
 * recording: it carries the build machine's directory layout (a home
 * directory, a user name, a worktree name) into a shipped artifact, and it
 * makes the artifact differ between two builds of the same source.
 *
 * The rule is therefore: **read by host path, record the display path.**
 * eshkol_source_path_display() computes the display path, once per distinct
 * input, and every recording site uses it:
 *
 *   - `eshkol_intern_source_file()` interns both columns, so every
 *     `source_file_id` resolves to a display name (and, separately, to the
 *     host path for reading the text behind a caret);
 *   - `eshkol_set_parse_source_context()` and the backend's ambient source
 *     context hold the display path, which is what diagnostics print and what
 *     the backend embeds for `eshkol_set_error_location`.
 *
 * ## How a display path is chosen
 *
 * In order, the first rule that applies:
 *
 *   1. A pseudo-name (`<unknown>`, `<stdin>`, `<string>`) or an already
 *      relative path is kept as it is, lexically normalized.
 *   2. Under a directory on `ESHKOL_PATH`: the module name as resolved there
 *      (the path relative to that root).
 *   3. Under a project root (the nearest ancestor holding `.git` or
 *      `CMakeLists.txt`): the repository-relative path.
 *   4. Under the current working directory: the path relative to it.
 *   5. Otherwise: the file name alone. Never an absolute host path.
 *
 * Results are memoized; the returned pointer is stable for the life of the
 * process.
 */

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The spelling of @p path that may be recorded or embedded.
 *
 * @param path A host path, a relative path, or a `<pseudo-name>`.
 * @return A stable, process-lifetime string. NULL and empty inputs come back
 *         unchanged, so a caller that already handles "unknown" keeps doing so.
 */
const char* eshkol_source_path_display(const char* path);

/**
 * @brief True when @p path would be recorded verbatim as an absolute host path.
 *
 * Always false for the result of eshkol_source_path_display(); exposed so a
 * test can assert that, rather than asserting the absence of one spelling.
 */
bool eshkol_source_path_is_host_absolute(const char* path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ESHKOL_FRONTEND_SOURCE_PATHS_H */
