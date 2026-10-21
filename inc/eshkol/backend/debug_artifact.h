/* Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_BACKEND_DEBUG_ARTIFACT_H
#define ESHKOL_BACKEND_DEBUG_ARTIFACT_H

#include <eshkol/pkg/subprocess.h>
#include <eshkol/platform_runtime.h>
#include <cstdlib>
#include <array>
#include <fstream>
#include <filesystem>
#include <optional>
#include <string>
#ifdef __APPLE__
#include <mach-o/loader.h>
#endif

namespace eshkol::backend {

#ifdef __APPLE__
inline std::optional<std::array<unsigned char, 16>> nativeDebugUUID(
        const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    mach_header_64 header{};
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        header.magic != MH_MAGIC_64) return std::nullopt;
    uint32_t remaining = header.sizeofcmds;
    for (uint32_t i = 0; i < header.ncmds; ++i) {
        load_command command{};
        if (remaining < sizeof(command) ||
            !file.read(reinterpret_cast<char*>(&command), sizeof(command)) ||
            command.cmdsize < sizeof(command) || command.cmdsize > remaining)
            return std::nullopt;
        if (command.cmd == LC_UUID) {
            std::array<unsigned char, 16> uuid{};
            if (command.cmdsize != sizeof(uuid_command) ||
                !file.read(reinterpret_cast<char*>(uuid.data()), uuid.size()))
                return std::nullopt;
            return uuid;
        }
        file.seekg(command.cmdsize - sizeof(command), std::ios::cur);
        if (!file) return std::nullopt;
        remaining -= command.cmdsize;
    }
    return std::nullopt;
}
#endif

// Mach-O executables carry a debug map referring to the input objects. Collect
// their DWARF before deleting owned linker temporaries, so -g remains usable
// after the compiler exits. ELF/COFF linkers retain their own debug information.
inline int finishNativeDebugArtifact(const std::filesystem::path& output,
                                     unsigned timeout_seconds) {
#ifdef __APPLE__
    std::string tool;
    if (const char* override_tool = std::getenv("ESHKOL_DSYMUTIL")) {
        tool = override_tool;
    }
    if (tool.empty()) {
        auto sibling = std::filesystem::path(platform::cxx_compiler()).parent_path()
                       / "dsymutil";
        std::error_code ec;
        tool = std::filesystem::is_regular_file(sibling, ec)
             ? sibling.string() : "dsymutil";
    }
    const std::filesystem::path bundle(output.string() + ".dSYM");
    int status = pkg::run_subprocess(
        {tool, "--num-threads", "1", output.string(), "-o", bundle.string()},
        nullptr, timeout_seconds);
    if (status != 0) return status;
    const auto dwarf = bundle / "Contents" / "Resources" / "DWARF"
                       / output.filename();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(dwarf, ec) || ec) return 1;
    // A successful tool exit cannot certify an old bundle left beside a new
    // executable. LLDB associates a dSYM with its image by this UUID.
    const auto image_uuid = nativeDebugUUID(output);
    const auto dwarf_uuid = nativeDebugUUID(dwarf);
    return image_uuid && dwarf_uuid && *image_uuid == *dwarf_uuid ? 0 : 1;
#else
    (void)output;
    (void)timeout_seconds;
    return 0;
#endif
}

} // namespace eshkol::backend
#endif
