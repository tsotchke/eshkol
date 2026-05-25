#!/usr/bin/env bash
# Run the standard Eshkol native Windows verification flow on a remote SSH host.

set -euo pipefail

HOST=""
REPO_DIR="${REPO_DIR:-}"
BUILD_DIR="${BUILD_DIR:-build-windows-x64}"
GENERATOR="${GENERATOR:-Visual Studio 17 2022}"
ARCH="${ARCH:-x64}"
TOOLSET="${TOOLSET:-ClangCL}"
CONFIG="${CONFIG:-Release}"
LLVM_VERSION="${LLVM_VERSION:-21}"
LLVM_DIR="${LLVM_DIR:-}"
FETCH_REF="${FETCH_REF:-origin/master}"
UPDATE=1
RUN_CTEST=1
RUN_WINDOWS_LITE=1

usage() {
    cat <<'USAGE'
Usage: scripts/remote_windows_verify.sh HOST [options]

Options:
  --repo DIR             Remote repo path (default: %USERPROFILE%\projects\eshkol)
  --build-dir DIR        Remote build directory (default: build-windows-x64)
  --generator NAME       CMake generator (default: Visual Studio 17 2022)
  --arch ARCH            Visual Studio target architecture (default: x64)
  --toolset TOOLSET      Visual Studio toolset (default: ClangCL)
  --config CONFIG        CMake build config (default: Release)
  --llvm-version N       Required LLVM major version (default: 21)
  --llvm-dir DIR         Remote LLVM CMake package directory
  --fetch-ref REF        Remote ref to fast-forward to (default: origin/master)
  --no-update            Do not fetch/fast-forward the remote checkout
  --no-ctest             Skip the Windows CTest surface guard
  --no-windows-lite      Skip scripts/run_all_tests.ps1 -Mode windows-lite
  -h, --help             Show this help
USAGE
}

require_option_value() {
    local option="$1"
    if [ "$#" -lt 2 ] || [[ "$2" == -* ]]; then
        echo "missing value for option: $option" >&2
        usage >&2
        exit 2
    fi
}

ps_quote() {
    local value=${1//\'/\'\'}
    printf "'%s'" "$value"
}

ps_bool() {
    if [ "$1" -eq 1 ]; then
        printf '$true'
    else
        printf '$false'
    fi
}

emit_ps_arg() {
    local name="$1"
    local value="$2"
    printf '$%s = %s\n' "$name" "$(ps_quote "$value")"
}

emit_ps_bool() {
    local name="$1"
    local value="$2"
    printf '$%s = %s\n' "$name" "$(ps_bool "$value")"
}

remote_script() {
    emit_ps_arg RepoDirArg "$REPO_DIR"
    emit_ps_arg BuildDirArg "$BUILD_DIR"
    emit_ps_arg GeneratorArg "$GENERATOR"
    emit_ps_arg ArchArg "$ARCH"
    emit_ps_arg ToolsetArg "$TOOLSET"
    emit_ps_arg ConfigArg "$CONFIG"
    emit_ps_arg LLVMVersionArg "$LLVM_VERSION"
    emit_ps_arg LLVMDirArg "$LLVM_DIR"
    emit_ps_arg FetchRefArg "$FETCH_REF"
    emit_ps_bool UpdateArg "$UPDATE"
    emit_ps_bool RunCtestArg "$RUN_CTEST"
    emit_ps_bool RunWindowsLiteArg "$RUN_WINDOWS_LITE"

    cat <<'PS_SCRIPT'
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Resolve-RepoDir {
    if (-not [string]::IsNullOrWhiteSpace($RepoDirArg)) {
        return $RepoDirArg
    }
    if (-not [string]::IsNullOrWhiteSpace($env:ESHKOL_REMOTE_REPO)) {
        return $env:ESHKOL_REMOTE_REPO
    }
    return (Join-Path $env:USERPROFILE "projects\eshkol")
}

function Resolve-BuildDir {
    param([string]$RepoDir)

    if ([string]::IsNullOrWhiteSpace($BuildDirArg)) {
        return (Join-Path $RepoDir "build-windows-x64")
    }
    if ([System.IO.Path]::IsPathRooted($BuildDirArg)) {
        return $BuildDirArg
    }
    return (Join-Path $RepoDir $BuildDirArg)
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    Write-Host ("+ {0} {1}" -f $FilePath, ($Arguments -join " "))
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("{0} failed with exit code {1}" -f $FilePath, $LASTEXITCODE)
    }
}

$RepoDir = Resolve-RepoDir
$BuildDir = Resolve-BuildDir -RepoDir $RepoDir

if (-not (Test-Path -LiteralPath $RepoDir -PathType Container)) {
    throw "Remote Eshkol checkout not found: $RepoDir"
}

Set-Location -LiteralPath $RepoDir

Invoke-External git @("status", "--short", "--branch")

if ($UpdateArg) {
    Invoke-External git @("fetch", "origin", "master")
    Invoke-External git @("merge", "--ff-only", $FetchRefArg)
}

$Head = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "git rev-parse failed"
}
Write-Host ("HEAD: {0}" -f $Head)

$cmakeArgs = @()
if (-not [string]::IsNullOrWhiteSpace($GeneratorArg)) {
    $cmakeArgs += @("-G", $GeneratorArg)
    if ($GeneratorArg -like "Visual Studio*") {
        if (-not [string]::IsNullOrWhiteSpace($ArchArg)) {
            $cmakeArgs += @("-A", $ArchArg)
        }
        if (-not [string]::IsNullOrWhiteSpace($ToolsetArg)) {
            $cmakeArgs += @("-T", $ToolsetArg)
        }
    }
}
$cmakeArgs += @(
    "-S", $RepoDir,
    "-B", $BuildDir,
    "-DESHKOL_BUILD_TESTS=ON",
    "-DESHKOL_REQUIRED_LLVM_MAJOR=$LLVMVersionArg"
)
if (-not [string]::IsNullOrWhiteSpace($LLVMDirArg)) {
    $cmakeArgs += "-DLLVM_DIR=$LLVMDirArg"
}

Invoke-External cmake $cmakeArgs

$buildArgs = @(
    "--build", $BuildDir,
    "--config", $ConfigArg,
    "--target", "eshkol-run", "eshkol-repl", "stdlib", "windows_suite_surface_test",
    "--parallel"
)
Invoke-External cmake $buildArgs

if ($RunCtestArg) {
    $ctestArgs = @(
        "--test-dir", $BuildDir,
        "-C", $ConfigArg,
        "-R", "windows_suite_surface_test",
        "--output-on-failure"
    )
    Invoke-External ctest $ctestArgs
}

if ($RunWindowsLiteArg) {
    $suitePath = Join-Path $RepoDir "scripts\run_all_tests.ps1"
    Invoke-External powershell.exe @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $suitePath,
        "-BuildDir", $BuildDir,
        "-Mode", "windows-lite"
    )
}

Invoke-External git @("diff", "--check")
Invoke-External git @("status", "--short", "--branch")
PS_SCRIPT
}

run_remote() {
    remote_script | ssh "$HOST" \
        "powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command -"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo)
            require_option_value "$@"
            REPO_DIR="$2"
            shift 2
            ;;
        --build-dir)
            require_option_value "$@"
            BUILD_DIR="$2"
            shift 2
            ;;
        --generator)
            require_option_value "$@"
            GENERATOR="$2"
            shift 2
            ;;
        --arch)
            require_option_value "$@"
            ARCH="$2"
            shift 2
            ;;
        --toolset)
            require_option_value "$@"
            TOOLSET="$2"
            shift 2
            ;;
        --config)
            require_option_value "$@"
            CONFIG="$2"
            shift 2
            ;;
        --llvm-version)
            require_option_value "$@"
            LLVM_VERSION="$2"
            shift 2
            ;;
        --llvm-dir)
            require_option_value "$@"
            LLVM_DIR="$2"
            shift 2
            ;;
        --fetch-ref)
            require_option_value "$@"
            FETCH_REF="$2"
            shift 2
            ;;
        --no-update)
            UPDATE=0
            shift
            ;;
        --no-ctest)
            RUN_CTEST=0
            shift
            ;;
        --no-windows-lite)
            RUN_WINDOWS_LITE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [ -n "$HOST" ]; then
                echo "host already set: $HOST" >&2
                exit 2
            fi
            HOST="$1"
            shift
            ;;
    esac
done

if [ -z "$HOST" ]; then
    usage >&2
    exit 2
fi

run_remote
