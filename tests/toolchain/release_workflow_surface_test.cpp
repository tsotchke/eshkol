#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    std::string contents = buffer.str();
    std::string normalized;
    normalized.reserve(contents.size());
    for (char ch : contents) {
        if (ch != '\r') {
            normalized.push_back(ch);
        }
    }
    return normalized;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& label) {
    if (haystack.find(needle) == std::string::npos) {
        std::cerr << "missing: " << label << "\nneedle: " << needle << std::endl;
        return false;
    }
    return true;
}

bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& label) {
    if (haystack.find(needle) != std::string::npos) {
        std::cerr << "unexpected: " << label << "\nneedle: " << needle
                  << std::endl;
        return false;
    }
    return true;
}

std::size_t count_occurrences(const std::string& haystack,
                              const std::string& needle) {
    if (needle.empty()) return 0;
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

struct ReleaseAsset {
    const char* name;
    const char* extension;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: release_workflow_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path workflow_path =
        source_root / ".github" / "workflows" / "release.yml";
    if (!fs::exists(workflow_path)) {
        return fail("release.yml not found under source root");
    }

    const std::string workflow = read_file(workflow_path);
    bool ok = true;

    ok = ok &&
         expect_contains(workflow, "name: Release",
                         "release workflow is named") &&
         expect_contains(workflow, "tags:\n      - 'v*'",
                         "release workflow only runs for version tags") &&
         expect_contains(workflow, "permissions:\n  contents: write",
                         "release workflow can publish GitHub releases") &&
         expect_contains(workflow, "cancel-in-progress: false",
                         "release workflow does not cancel an in-flight tag build") &&
         expect_contains(workflow, "unix-release-matrix:",
                         "release workflow has Unix asset matrix") &&
         expect_contains(workflow, "windows-release-matrix:",
                         "release workflow has Windows asset matrix") &&
         expect_contains(workflow, "publish-release:",
                         "release workflow has publish job") &&
         expect_contains(workflow, "needs:\n      - unix-release-matrix\n      - windows-release-matrix",
                         "publish job waits for both platform matrices") &&
         expect_contains(workflow, "pattern: release-asset-*",
                         "publish job downloads all packaged assets by prefix") &&
         expect_contains(workflow, "merge-multiple: true",
                         "publish job flattens downloaded assets") &&
         expect_contains(workflow, "Validate Release Asset Set",
                         "release workflow validates the final asset set") &&
         expect_contains(workflow,
                         "asset_count=\"$(find . -maxdepth 1 -type f ! -name SHA256SUMS.txt | wc -l | tr -d ' ')\"",
                         "release workflow counts non-checksum assets") &&
         expect_contains(workflow, "Expected ${#expected[@]} release assets",
                         "release workflow rejects missing or extra assets") &&
         expect_contains(workflow, "sha256sum * > SHA256SUMS.txt",
                         "release workflow writes checksums for all assets") &&
         expect_contains(workflow, "gh release view \"$GITHUB_REF_NAME\"",
                         "release workflow checks for existing release") &&
         expect_contains(workflow,
                         "refusing to overwrite or append assets",
                         "release workflow refuses to append to existing releases") &&
         expect_contains(workflow, "gh release create \"$GITHUB_REF_NAME\"",
                         "release workflow creates a tag-named GitHub release") &&
         expect_contains(workflow, "--verify-tag",
                         "release workflow verifies that the release tag exists") &&
         expect_contains(workflow, "dist/*",
                         "release workflow uploads the complete dist directory");

    ok = ok &&
         expect_contains(workflow,
                         "archive_root=\"eshkol-${{ github.ref_name }}-${{ matrix.name }}\"",
                         "Unix release archives include the tag and platform name") &&
         expect_contains(workflow,
                         "tar -czf \"$RUNNER_TEMP/$archive_root.tar.gz\"",
                         "Unix release matrix produces tar.gz assets") &&
         expect_contains(workflow,
                         "path: ${{ runner.temp }}/eshkol-${{ github.ref_name }}-${{ matrix.name }}.tar.gz",
                         "Unix upload path includes the tag and platform name") &&
         expect_contains(workflow,
                         "$archiveRoot = \"eshkol-${{ github.ref_name }}-${{ matrix.name }}\"",
                         "Windows release archives include the tag and platform name") &&
         expect_contains(workflow,
                         "$archivePath = Join-Path $env:RUNNER_TEMP \"$archiveRoot.zip\"",
                         "Windows release matrix produces zip assets") &&
         expect_contains(workflow,
                         "path: ${{ runner.temp }}\\eshkol-${{ github.ref_name }}-${{ matrix.name }}.zip",
                         "Windows upload path includes the tag and platform name");

    const std::vector<ReleaseAsset> expected_assets = {
        {"linux-x64-lite", "tar.gz"},
        {"linux-arm64-lite", "tar.gz"},
        {"linux-x64-xla", "tar.gz"},
        {"linux-arm64-xla", "tar.gz"},
        {"linux-x64-cuda", "tar.gz"},
        {"linux-arm64-cuda", "tar.gz"},
        {"macos-arm64-lite", "tar.gz"},
        {"macos-x64-lite", "tar.gz"},
        {"macos-arm64-xla", "tar.gz"},
        {"macos-x64-xla", "tar.gz"},
        {"windows-x64-lite", "zip"},
        {"windows-arm64-lite", "zip"},
        {"windows-x64-xla", "zip"},
        {"windows-arm64-xla", "zip"},
        {"windows-x64-cuda", "zip"},
        {"windows-arm64-cuda", "zip"},
    };

    for (const ReleaseAsset& asset : expected_assets) {
        const std::string matrix_name = std::string("- name: ") + asset.name;
        const std::string expected_filename =
            std::string("\"eshkol-${GITHUB_REF_NAME}-") + asset.name + "." +
            asset.extension + "\"";
        ok = ok &&
             expect_contains(workflow, matrix_name,
                             std::string("release matrix includes ") + asset.name) &&
             expect_contains(workflow, expected_filename,
                             std::string("validated asset list includes ") + asset.name);
    }

    const std::size_t validated_asset_count =
        count_occurrences(workflow, "\"eshkol-${GITHUB_REF_NAME}-");
    if (validated_asset_count != expected_assets.size()) {
        std::cerr << "expected " << expected_assets.size()
                  << " validated release asset names, found "
                  << validated_asset_count << std::endl;
        ok = false;
    }

    ok = ok &&
         expect_not_contains(workflow, "gh release upload",
                             "release workflow should not append assets to an existing release") &&
         expect_not_contains(workflow, "--clobber",
                             "release workflow should not overwrite release assets") &&
         expect_not_contains(workflow, "cancel-in-progress: true",
                             "release workflow should not cancel active tag builds");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
