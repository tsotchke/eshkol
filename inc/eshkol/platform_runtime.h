/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef ESHKOL_PLATFORM_RUNTIME_H
#define ESHKOL_PLATFORM_RUNTIME_H

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace eshkol::platform {

std::filesystem::path executable_path();
std::filesystem::path executable_directory();
std::filesystem::path current_directory();
std::string find_first_existing(const std::vector<std::filesystem::path>& candidates);
std::string home_directory();
bool stdin_isatty();
bool stdout_isatty();
bool initialize_interactive_console();
bool stdout_supports_utf8();
std::filesystem::path make_temp_path(std::string_view stem, std::string_view extension = ".tmp");
std::string cxx_compiler();
std::string llc_executable();
std::string executable_suffix();
std::string static_library_name(std::string_view stem);
std::vector<std::string> host_runtime_link_args();
std::filesystem::path with_executable_suffix(const std::filesystem::path& path);
std::string shell_quote(std::string_view argument);
int run_command(const std::vector<std::string>& arguments);
std::filesystem::path resolve_executable_output(const std::filesystem::path& base_path);

} // namespace eshkol::platform

#endif // ESHKOL_PLATFORM_RUNTIME_H
