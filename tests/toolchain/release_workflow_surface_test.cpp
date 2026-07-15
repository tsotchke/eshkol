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
         expect_contains(workflow, "workflow_dispatch:",
                         "release workflow supports a manual dry run") &&
         expect_contains(workflow, "candidate_tag:",
                         "manual dry run requires an explicit candidate tag label") &&
         expect_contains(workflow,
                         "RELEASE_TAG: ${{ github.event_name == 'workflow_dispatch' && inputs.candidate_tag || github.ref_name }}",
                         "tag and dry-run asset naming share one release label") &&
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
         expect_contains(workflow, "gh release view \"$RELEASE_TAG\"",
                         "release workflow checks for existing release") &&
         expect_contains(workflow,
                         "refusing to overwrite or append assets",
                         "release workflow refuses to append to existing releases") &&
         expect_contains(workflow, "gh release create \"$RELEASE_TAG\"",
                         "release workflow creates a tag-named GitHub release") &&
         expect_contains(workflow, "--verify-tag",
                         "release workflow verifies that the release tag exists") &&
         expect_contains(workflow, "Prepare Curated Release Notes",
                         "release workflow extracts the current curated notes") &&
         expect_contains(workflow,
                         "--notes-file \"$RUNNER_TEMP/release-notes.md\"",
                         "GitHub release body uses curated release notes") &&
         expect_contains(workflow, "Upload Validated Dry-Run Assets",
                         "manual dry run preserves its validated asset set") &&
         expect_contains(workflow,
                         "if: github.event_name == 'workflow_dispatch'",
                         "dry-run artifact upload is dispatch-only") &&
         expect_contains(workflow, "dist/*",
                         "release workflow uploads the complete dist directory");

    ok = ok &&
         expect_contains(workflow,
                         "archive_root=\"eshkol-${RELEASE_TAG}-${{ matrix.name }}\"",
                         "Unix release archives include the tag and platform name") &&
         expect_contains(workflow,
                         "tar -czf \"$RUNNER_TEMP/$archive_root.tar.gz\"",
                         "Unix release matrix produces tar.gz assets") &&
         expect_contains(workflow,
                         "path: ${{ runner.temp }}/eshkol-${{ env.RELEASE_TAG }}-${{ matrix.name }}.tar.gz",
                         "Unix upload path includes the tag and platform name") &&
         expect_contains(workflow,
                         "$archiveRoot = \"eshkol-$env:RELEASE_TAG-${{ matrix.name }}\"",
                         "Windows release archives include the tag and platform name") &&
         expect_contains(workflow,
                         "$archivePath = Join-Path $env:RUNNER_TEMP \"$archiveRoot.zip\"",
                         "Windows release matrix produces zip assets") &&
         expect_contains(workflow,
                         "path: ${{ runner.temp }}\\eshkol-${{ env.RELEASE_TAG }}-${{ matrix.name }}.zip",
                         "Windows upload path includes the tag and platform name") &&
         expect_contains(workflow,
                         "for doc in README.md LICENSE CHANGELOG.md RELEASE_NOTES.md; do",
                         "Unix archives include release notes") &&
         expect_contains(workflow,
                         "foreach ($doc in @('README.md', 'LICENSE', 'CHANGELOG.md', 'RELEASE_NOTES.md'))",
                         "Windows archives include release notes") &&
         expect_contains(workflow,
                         "python3 scripts/stage_third_party_licenses.py",
                         "Unix archives stage pinned third-party licenses") &&
         expect_contains(workflow,
                         "python scripts/stage_third_party_licenses.py",
                         "Windows archives stage pinned third-party licenses");

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
            std::string("\"eshkol-${RELEASE_TAG}-") + asset.name + "." +
            asset.extension + "\"";
        ok = ok &&
             expect_contains(workflow, matrix_name,
                             std::string("release matrix includes ") + asset.name) &&
             expect_contains(workflow, expected_filename,
                             std::string("validated asset list includes ") + asset.name);
    }

    const std::size_t expected_block_begin = workflow.find("          expected=(\n");
    const std::size_t expected_block_end =
        expected_block_begin == std::string::npos
            ? std::string::npos
            : workflow.find("\n          )", expected_block_begin);
    const std::string expected_block =
        expected_block_begin == std::string::npos ||
                expected_block_end == std::string::npos
            ? std::string()
            : workflow.substr(expected_block_begin,
                              expected_block_end - expected_block_begin);
    const std::size_t validated_asset_count =
        count_occurrences(expected_block, "\"eshkol-${RELEASE_TAG}-");
    if (validated_asset_count != expected_assets.size()) {
        std::cerr << "expected " << expected_assets.size()
                  << " validated release asset names, found "
                  << validated_asset_count << std::endl;
        ok = false;
    }

    ok = ok &&
         expect_not_contains(workflow, "gh release upload",
                             "release workflow should not append assets to an existing release") &&
         expect_not_contains(workflow, "--generate-notes",
                             "release workflow should not replace curated notes with generated notes") &&
         expect_not_contains(workflow, "--clobber",
                             "release workflow should not overwrite release assets") &&
         expect_not_contains(workflow, "cancel-in-progress: true",
                             "release workflow should not cancel active tag builds");

    if (count_occurrences(workflow, "if: github.event_name == 'push'") < 2) {
        std::cerr << "publish and Homebrew jobs must both be disabled during manual dry runs"
                  << std::endl;
        ok = false;
    }

    if (count_occurrences(workflow, "uses: actions/checkout@v6") < 3) {
        std::cerr << "all build matrices and the publish job must checkout the tagged source"
                  << std::endl;
        ok = false;
    }

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
