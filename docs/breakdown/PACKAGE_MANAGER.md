# Eshkol Package Manager (eshkol-pkg)

**Status:** Production — v1.1.11
**Source:** `tools/pkg/eshkol_pkg.cpp` (722 lines)
**Binary:** `eshkol-pkg`

---

## Overview

`eshkol-pkg` is the official package manager for Eshkol. It manages project manifests, fetches
dependencies from git-based registries, invokes `eshkol-run` to compile projects, and provides
a publishing workflow for sharing packages. The tool is self-contained: the only external
dependency is a POSIX shell and `git` on `PATH`. No network daemon, no lock-file resolver, no
virtual environment — each project has a flat `eshkol_deps/` directory with symlinks into a
shared user-level cache at `~/.eshkol/cache/packages/`.

The manifest format is TOML, parsed by a minimal custom implementation (~170 lines) with no
third-party dependencies. It supports tables, key-value pairs, quoted strings, integers, floats,
booleans, and single-line arrays. Multi-line strings, dotted keys, and inline tables are not
supported.

---

## Quick Start

```sh
# 1. Create a new project
mkdir my-project && cd my-project
eshkol-pkg init

# 2. Edit src/main.esk, then compile
eshkol-pkg build

# 3. Compile and run in one step
eshkol-pkg run
```

`init` writes `eshkol.toml` and creates `src/main.esk` with a hello-world template. The project
name is taken from the current directory name. No further configuration is needed to build a
single-file project.

---

## eshkol.toml Manifest Format

Every Eshkol project is described by a single `eshkol.toml` at the project root. The file has
two sections: `[package]` (required) and `[dependencies]` (optional).

### [package]

```toml
[package]
name        = "my-project"      # Package name; used as the output binary name.
version     = "0.1.0"           # Semver string. Displayed in diagnostics and publish output.
description = "A short blurb"   # Optional free-form description shown by eshkol-pkg search.
author      = "Alice <a@b.com>" # Optional author string. Not interpreted by the tool.
license     = "MIT"             # SPDX license identifier. Defaults to "MIT" on init.
entry       = "src/main.esk"    # Relative path to the main compilation unit (required).
sources     = ["src/*.esk"]     # Glob patterns for additional source files (informational).
```

All fields except `entry` are optional but recommended. The `name` field is used directly as the
output binary filename under `build/`. The `sources` array is recorded in the manifest for
documentation and tooling purposes; the compiler invocation uses only `entry`.

### [dependencies]

```toml
[dependencies]
# Registry package: looks up https://github.com/tsotchke/<name>.git
numerics  = "1.2.0"

# Direct git URL: cloned verbatim, version string is the URL
my-lib    = "https://github.com/example/my-lib.git"
```

Each dependency is a key-value pair where the key is the package name and the value is either a
version string (for registry packages) or a full `https://` git URL (for direct git packages).
The distinction is made at install time: if the version value contains `://`, it is treated as a
git URL; otherwise, the registry URL
`https://github.com/tsotchke/<name>.git` is used.

The wildcard version `*` is the default when a version is omitted with `eshkol-pkg add`. It
instructs the installer to clone the default branch without any version pinning.

### Complete Example

```toml
[package]
name        = "signal-proc"
version     = "0.3.1"
description = "Digital signal processing utilities for Eshkol"
author      = "Bob <bob@example.com>"
license     = "Apache-2.0"
entry       = "src/main.esk"
sources     = ["src/*.esk", "src/filters/*.esk"]

[dependencies]
numerics   = "1.0.0"
fft-lib    = "https://github.com/example/eshkol-fft.git"
```

---

## Command Reference

All commands operate on the `eshkol.toml` in the current working directory. Commands that
require a manifest will exit with an error if none is found.

### `eshkol-pkg init`

**Synopsis:** `eshkol-pkg init`

Creates `eshkol.toml` and `src/main.esk` in the current directory. Fails if `eshkol.toml`
already exists. The project name is derived from `basename(cwd)`. Generated defaults:

| Field     | Default value          |
|-----------|------------------------|
| version   | `0.1.0`                |
| license   | `MIT`                  |
| entry     | `src/main.esk`         |
| sources   | `["src/*.esk"]`        |

The `src/main.esk` template contains a single `(display ...)` / `(newline)` example and
build instructions in comments. The file is not overwritten if it already exists.

**Example:**
```sh
mkdir my-project && cd my-project
eshkol-pkg init
# => Created eshkol.toml for project 'my-project'
# => src/main.esk created
```

---

### `eshkol-pkg build`

**Synopsis:** `eshkol-pkg build`

Compiles the project by invoking:

```
<compiler> "<entry>" -o "build/<name>" [--link "<dep>/<file>.o" ...]
```

where `<compiler>` is the value of `ESHKOL_COMPILER` or `eshkol-run` by default. The `build/`
directory is created if it does not exist.

If `eshkol_deps/` is present, the build command scans every immediate subdirectory for `.o`
files and passes each as `--link <path>` to link compiled dependency object files into the
output binary.

The command prints the resolved compiler invocation, the build status, and on success the path
to the output binary.

**Example:**
```sh
eshkol-pkg build
# Building signal-proc v0.3.1...
# Built: /home/bob/signal-proc/build/signal-proc
```

---

### `eshkol-pkg run`

**Synopsis:** `eshkol-pkg run`

Equivalent to `eshkol-pkg build && ./build/<name>`. The build step is always performed;
there is no incremental build cache. If the build fails the binary is not executed.

**Example:**
```sh
eshkol-pkg run
# Building signal-proc v0.3.1...
# Built: /home/bob/signal-proc/build/signal-proc
#
# --- Running signal-proc ---
# <program output>
```

---

### `eshkol-pkg install`

**Synopsis:** `eshkol-pkg install`

Reads the `[dependencies]` table from `eshkol.toml` and installs each dependency:

1. Checks whether `~/.eshkol/cache/packages/<name>/` already exists.
2. If not cached, runs `git clone --depth 1 <url> ~/.eshkol/cache/packages/<name>/`. The URL
   is derived from the version field (see Dependency Resolution below).
3. Creates a symlink `./eshkol_deps/<name> -> ~/.eshkol/cache/packages/<name>/` if not already
   present. The symlink is not updated on repeat installs.

A shallow clone (`--depth 1`) is always used to minimise download size. Stderr from git is
redirected to `/dev/null`; only warnings from `eshkol-pkg` itself appear in the terminal.

**Example:**
```sh
eshkol-pkg install
# Installing numerics 1.0.0...
#   Installed numerics
# Installing fft-lib https://github.com/example/eshkol-fft.git...
#   Installed fft-lib
# All dependencies installed.
```

---

### `eshkol-pkg add`

**Synopsis:** `eshkol-pkg add <package> [version]`

Adds or updates a dependency entry in `eshkol.toml`. If `<package>` is already listed, its
version is updated in-place. Otherwise a new entry is appended. The default version when
omitted is `*`.

Does not install the package; run `eshkol-pkg install` afterward.

**Examples:**
```sh
eshkol-pkg add numerics 1.2.0
# Added numerics 1.2.0 to dependencies

eshkol-pkg add my-lib https://github.com/example/my-lib.git
# Added my-lib https://github.com/example/my-lib.git to dependencies

eshkol-pkg add experimental-pkg
# Added experimental-pkg * to dependencies
```

---

### `eshkol-pkg remove`

**Synopsis:** `eshkol-pkg remove <package>`

Removes the named dependency from `eshkol.toml` and deletes `./eshkol_deps/<package>` (the
local symlink or directory). The shared cache at `~/.eshkol/cache/packages/<package>/` is not
deleted; use `eshkol-pkg clean` or remove the cache directory manually if disk reclamation is
needed.

Returns exit code 1 if the package is not present in `eshkol.toml`.

**Example:**
```sh
eshkol-pkg remove fft-lib
# Removed fft-lib from dependencies
```

---

### `eshkol-pkg search`

**Synopsis:** `eshkol-pkg search <query>`

Searches the local package cache at `~/.eshkol/cache/packages/` for directories whose name
contains `<query>` as a substring (case-sensitive). For each match, reads `eshkol.toml` from
the cached package directory and prints the name, version, and description.

**Note:** The search is entirely local; it does not query any remote registry. Packages that
have not yet been installed or cloned will not appear. To discover packages not yet in the
cache, browse the registry directly at `ESHKOL_REGISTRY` (default:
https://github.com/tsotchke/eshkol-registry.git).

**Example:**
```sh
eshkol-pkg search num
#   numerics v1.2.0 - Numeric algorithms for Eshkol
#   bignum-ext v0.4.0 - Arbitrary precision extension
```

If no packages match:
```sh
eshkol-pkg search foobar
# No packages found matching 'foobar'
# Registry: https://github.com/tsotchke/eshkol-registry.git
```

---

### `eshkol-pkg publish`

**Synopsis:** `eshkol-pkg publish`

Prints step-by-step instructions for submitting a package to the community registry. The tool
does not push to git or open a pull request automatically; the workflow is manual.

**Output:**
```
Publishing signal-proc v0.3.1...

To publish to the Eshkol registry:
  1. Push your code to a git repository
  2. Tag the release: git tag v0.3.1
  3. Submit a PR to https://github.com/tsotchke/eshkol-registry.git
     adding your package to the registry index
```

See the Publishing section below for the full workflow.

---

### `eshkol-pkg clean`

**Synopsis:** `eshkol-pkg clean`

Deletes the `build/` directory using `fs::remove_all`. Does not touch `eshkol_deps/` or the
global cache.

**Example:**
```sh
eshkol-pkg clean
# Removed build/
```

---

## Dependency Resolution

### Registry vs. Git URL

When `eshkol-pkg install` processes a dependency, the URL is chosen as follows:

```
if version contains "://"
    repo_url = version          # treat as direct git URL
else
    repo_url = "https://github.com/tsotchke/" + name + ".git"
```

There is no version constraint resolution or semver comparison. The version string is stored in
`eshkol.toml` as metadata and is printed in diagnostic messages, but the installer always clones
the default branch of the resolved URL. Pinning to a specific commit or tag requires updating
the URL to include a ref, or cloning and archiving the package manually in the cache.

### Cache Structure

All packages share a single user-level cache:

```
~/.eshkol/cache/
└── packages/
    ├── numerics/          # git clone of tsotchke/numerics
    ├── fft-lib/           # git clone of example/eshkol-fft
    └── my-lib/            # git clone of example/my-lib
```

Once a package directory exists in the cache, `eshkol-pkg install` will not re-clone it. To
force a fresh clone, delete the cache entry manually:

```sh
rm -rf ~/.eshkol/cache/packages/numerics
eshkol-pkg install
```

### Local Symlinks

`eshkol_deps/` holds one symlink per installed dependency:

```
eshkol_deps/
├── numerics -> ~/.eshkol/cache/packages/numerics
└── fft-lib  -> ~/.eshkol/cache/packages/fft-lib
```

The symlink is created with `fs::create_directory_symlink`. If the symlink already exists (even
if broken), it is not recreated. The `build` command scans `eshkol_deps/` for `.o` files in
immediate subdirectories and passes them as `--link` arguments to `eshkol-run`.

---

## Environment Variables

| Variable           | Default                                                    | Description                                                                                        |
|--------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `ESHKOL_COMPILER`  | `eshkol-run`                                               | Full path or name of the Eshkol compiler binary. Override when `eshkol-run` is not on `PATH`.     |
| `ESHKOL_REGISTRY`  | `https://github.com/tsotchke/eshkol-registry.git`         | Registry URL shown in `search` output and `publish` instructions. Does not affect install URLs.   |
| `HOME`             | System home directory                                      | Base for `~/.eshkol/cache/`. Falls back to `USERPROFILE` on Windows-like environments.            |
| `USERPROFILE`      | (Windows fallback)                                         | Used if `HOME` is unset. Required on environments where `HOME` is not set by default.             |

---

## Directory Layout

### Project Directory

```
my-project/
├── eshkol.toml          # Project manifest (required)
├── src/
│   └── main.esk         # Entry point (matches entry = in eshkol.toml)
├── eshkol_deps/         # Created by eshkol-pkg install
│   └── numerics -> ~/.eshkol/cache/packages/numerics
└── build/               # Created by eshkol-pkg build
    └── my-project       # Compiled binary
```

### Global Cache

```
~/.eshkol/
└── cache/
    └── packages/
        └── <package-name>/    # Shallow git clone of each dependency
            ├── eshkol.toml
            └── src/
```

The global cache is shared across all projects on the same machine. Packages in the cache are
never updated automatically; to update a dependency, delete its cache directory and re-run
`eshkol-pkg install`.

---

## Publishing

Publishing a package to the Eshkol community registry is a four-step manual process:

1. **Host the source.** Push the package to a publicly accessible git repository
   (e.g. GitHub, GitLab, Sourcehut). The repository must contain an `eshkol.toml` at the root.

2. **Tag the release.**
   ```sh
   git tag v0.3.1
   git push origin v0.3.1
   ```
   The tag should match the `version` field in `eshkol.toml`.

3. **Submit a pull request** to the registry repository at
   `https://github.com/tsotchke/eshkol-registry.git`, adding the package name and repository
   URL to the registry index.

4. **Notify users.** Once the PR is merged, users can install the package by name:
   ```sh
   eshkol-pkg add my-package 0.3.1
   eshkol-pkg install
   ```

Before publishing, verify the package builds cleanly from a fresh directory:
```sh
eshkol-pkg clean
eshkol-pkg install
eshkol-pkg build
```

---

## Complete Example: Creating a Project from Scratch

The following walkthrough creates a numeric utility library, adds a dependency, and builds a
working binary.

```sh
# 1. Create the project directory and initialise the manifest
mkdir vec-utils && cd vec-utils
eshkol-pkg init
# => Created eshkol.toml for project 'vec-utils'
# => src/main.esk created

# 2. Inspect the generated manifest
cat eshkol.toml
# [package]
# name = "vec-utils"
# version = "0.1.0"
# description = ""
# author = ""
# license = "MIT"
# entry = "src/main.esk"
# sources = ["src/*.esk"]

# 3. Add a dependency on the numerics package (registry lookup)
eshkol-pkg add numerics 1.0.0
# Added numerics 1.0.0 to dependencies

# 4. Install dependencies (clones to ~/.eshkol/cache/, creates eshkol_deps/ symlink)
eshkol-pkg install
# Installing numerics 1.0.0...
#   Installed numerics
# All dependencies installed.

# 5. Write the main source file
cat > src/main.esk << 'EOF'
;; vec-utils — dot product example
(define (dot-product a b)
  (apply + (map * a b)))

(display (dot-product '(1 2 3) '(4 5 6)))
(newline)
EOF

# 6. Build the project
eshkol-pkg build
# Building vec-utils v0.1.0...
# Built: /home/alice/vec-utils/build/vec-utils

# 7. Run the binary directly
./build/vec-utils
# 32

# 8. Or build and run in one command
eshkol-pkg run
# Building vec-utils v0.1.0...
# Built: /home/alice/vec-utils/build/vec-utils
#
# --- Running vec-utils ---
# 32

# 9. Clean build artifacts
eshkol-pkg clean
# Removed build/

# 10. Remove a dependency no longer needed
eshkol-pkg remove numerics
# Removed numerics from dependencies
```

To use a custom compiler binary:
```sh
ESHKOL_COMPILER=/opt/eshkol/bin/eshkol-run eshkol-pkg build
```

To use a private registry:
```sh
ESHKOL_REGISTRY=https://gitlab.example.com/internal/eshkol-registry.git eshkol-pkg search math
```

---

## See Also

- `eshkol-run(1)` — the Eshkol compiler and execution driver
- `tools/pkg/eshkol_pkg.cpp` — complete package manager source (722 lines)
- `docs/breakdown/README.md` — Eshkol architecture overview
- `CONTRIBUTING.md` — guidelines for contributing packages and compiler changes
- Registry: https://github.com/tsotchke/eshkol-registry.git
