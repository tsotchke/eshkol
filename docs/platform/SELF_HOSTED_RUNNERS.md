# Self-hosted runners (the mesh)

How to attach the maintainer's own machines to this repository as GitHub Actions
runners, what each label means, and what changes once a runner is online.

Everything here is something **only the repository owner can do** — registering a
runner requires a registration token, which requires repo-admin rights. Nothing in
this document is automatable from a pull request, by design.

Related: [CI lanes](CI_LANES.md) · `.github/workflows/ci-mesh.yml` ·
`.github/workflows/gpu-execution-gate.yml`

---

## 1. What this buys, and what it does not

`ci.yml` remains the single source of required check names. With the repository
variable `ESHKOL_MESH_PRIMARY=on`, its required Linux x64 matrix build/test
lanes and build-free guard jobs execute on the owned mesh, while ARM64, macOS,
and Windows remain hosted until their label variables are provisioned. With the
variable unset (the default), the existing hosted routing is unchanged.

The primary switch is explicit and fail-closed: a fork pull request is never
selected for a self-hosted runner. A fork PR uses the hosted fallback, and the
Actions setting requiring approval for all outside-collaborator workflows remains
mandatory. `ci-mesh.yml` is only the legacy advisory matrix; it suppresses its
lanes when primary routing is on so the same work is not run twice.

---

## 2. Label taxonomy

GitHub applies three labels automatically at registration and they cannot be
removed: `self-hosted`, an OS label (`Linux` / `macOS` / `Windows`), and an
architecture label (`X64` / `ARM64` / `ARM`). Label matching is case-insensitive,
which is why the workflows spell them lowercase.

On top of those, this repository defines exactly five custom labels. Keep the set
small: a label is a contract a lane relies on, and an unmet contract is a lane that
either queues forever or lies.

| label | meaning — assign it only if this is true | consumed by |
|---|---|---|
| `eshkol` | This runner is provisioned for **this repository's** build: LLVM 21, cmake, ninja, python3, jq, and (Linux) `ld.lld`. Every mesh lane requires it. | `ci.yml`, `ci-mesh.yml`, `mesh-preflight` |
| `linux-mesh` | Linux x64 primary-build capacity. It is present on the four primary x64 runners. | `ci.yml` Linux x64, WASM, guard, assurance, and surface lanes |
| `gpu` | The host has a **real, addressable GPU device** — not merely a GPU toolchain. `nvidia-smi` lists a device, or the host is Apple Silicon with a working Metal device. | `gpu-execution-gate.yml` (`runs-on: [self-hosted, gpu]`) |
| `cuda` | The host has a CUDA device *and* `nvcc` on `PATH`; the primary CUDA lane also verifies `nvidia-smi`. | `ci.yml` `linux-x64-cuda`, `mesh-linux-x64-cuda-exec` |
| `metal` | Apple Silicon host with a working Metal device. Implies `gpu`; assign both. Reserved for a future Metal execution lane. | (none yet) |

So a full label set looks like `--labels eshkol,linux-mesh` for a plain Linux
x64 build node, or `--labels eshkol,cuda` for the CUDA execution node.

**Do not** assign `gpu` to a host that only has the CUDA *toolkit* installed. That
is precisely the failure the GPU execution gate exists to expose, and a mislabelled
runner would recreate it while looking green. The `Toolchain preflight` step in
`ci-mesh.yml` fails a `cuda`-labelled runner that has no `nvidia-smi` device for
this reason.

---

## 3. Which machines

Measured against the mesh registry (`computer_mesh/nodes.json`) on 2026-08-25 by
SSH probe, not assumed. Full survey and the raw per-node output live with the PR
that introduced this document.

| node | OS / arch | capacity | GPU | registered labels | provisioning still needed |
|---|---|---|---|---|---|
| `mesh-linux-x64-01..04` | Linux x64 | 88 cores shared across four runner instances; about 450 GB free | None usable for CUDA CI | `self-hosted`, `Linux`, `X64`, `eshkol`, `linux-mesh` | LLVM 21, cmake, ninja, python3, jq, `ld.lld`; Node.js for WASM; emsdk for `emcc` |
| `mesh-cuda-01` | Linux x64 | Shared; one runner/job at a time | CUDA 12.4 device | `self-hosted`, `Linux`, `X64`, `eshkol`, `cuda` | LLVM 21, cmake, ninja, python3, jq, `ld.lld`, CUDA toolkit/driver and `nvidia-smi` |

The four `mesh-linux-x64-*` registrations are the primary Linux x64 capacity;
GitHub can run at most four such jobs in parallel. `mesh-cuda-01` is deliberately
single-job capacity. Do not assume the CUDA node can satisfy the Linux x64
`linux-mesh` label: its separate `cuda` label is what keeps ordinary builds off
the shared GPU host.

---

## 4. Provisioning a node

Do this **before** registering the runner. The mesh lanes deliberately do not
install anything: CI has no business mutating a machine you own, and self-installing
would require handing the runner passwordless `sudo`.

### Linux (Debian / Ubuntu)

```bash
# LLVM 21 from apt.llvm.org (adjust the distro codename)
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-21 main" \
  | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt-get update
sudo apt-get install -y \
  cmake ninja-build git python3 jq nodejs pkg-config \
  llvm-21 llvm-21-dev lld-21 \
  libreadline-dev libssl-dev libncurses-dev \
  libpcre2-dev libsqlite3-dev libpng-dev libjpeg-dev libwebp-dev
```

`-fuse-ld=lld` looks for an unversioned `ld.lld`. `ci-mesh.yml` shims it into the
runner's own `PATH` rather than symlinking into `/usr/bin` with sudo, so nothing
further is needed — but a system-wide `sudo ln -sf /usr/bin/ld.lld-21 /usr/bin/ld.lld`
is harmless if you prefer it.

For a `cuda` node, additionally confirm **both** of these answer:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version
```

For the four `linux-mesh` nodes, provision Emscripten without `sudo` and make
the active SDK available to the runner service account. The WASM lane fails
loudly if `emcc` is absent; it does not install or activate an SDK in CI:

```bash
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
cd ~/emsdk
./emsdk install 4.0.22
./emsdk activate 4.0.22
echo 'source "$HOME/emsdk/emsdk_env.sh" >/dev/null' >> ~/.bashrc
```

The service environment must also contain the activated SDK's `emcc` and
`node` on `PATH`; verify with `command -v emcc`, `emcc --version`, and
`node --version` from the account that runs the GitHub Actions service.

### macOS

```bash
brew install llvm@21 cmake ninja readline pcre2 sqlite
```

### Verify the toolchain the way CI will

```bash
git clone https://github.com/tsotchke/eshkol && cd eshkol
cmake -S . -B build-check -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DESHKOL_REQUIRED_LLVM_MAJOR=21 \
  -DLLVM_CONFIG_EXECUTABLE="$(command -v llvm-config-21 || echo /opt/homebrew/opt/llvm@21/bin/llvm-config)"
cmake --build build-check --parallel
./scripts/run_all_tests.sh
```

The primary Linux x64 CI configure contract is `RelWithDebInfo`, tests enabled,
`-DESHKOL_XLA_ENABLED=OFF`, `-DESHKOL_GPU_ENABLED=OFF`, and
`-DESHKOL_REQUIRE_GPU_BACKEND=OFF` for the lite lane. XLA and CUDA lanes retain
their matrix-specific feature flags. CI only discovers the provisioned tools;
it never runs `apt-get`, `sudo`, or a package manager on self-hosted runners.
If the checks above fail by hand, fix the node before registering it: a broken
toolchain produces red lanes that look like code regressions.

---

## 5. Registering the runner

### 5.1 Get a registration token

Registration tokens are **single-use and expire in one hour**. Get a fresh one per
node.

- Web: **Settings → Actions → Runners → New self-hosted runner**, and copy the
  token out of the `./config.sh` line the page shows.
- CLI (needs repo-admin scope):
  ```bash
  gh api -X POST repos/tsotchke/eshkol/actions/runners/registration-token --jq .token
  ```

### 5.2 Install and configure

Pick a directory that is **not** inside a checkout of this repo and has room for
the build trees (25 GiB minimum, 30 GiB for the CUDA lane).

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner

# Linux x64 — swap the asset for linux-arm64 / osx-arm64 / win-x64 as needed.
# Check https://github.com/actions/runner/releases for the current version.
RUNNER_VERSION=2.330.0
curl -o runner.tar.gz -L \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf runner.tar.gz && rm runner.tar.gz

./config.sh \
  --url https://github.com/tsotchke/eshkol \
  --token <REGISTRATION_TOKEN> \
  --name mesh-linux-x64-01 \
  --labels eshkol,linux-mesh \
  --work _work \
  --unattended \
  --replace
```

Use names `mesh-linux-x64-01` through `mesh-linux-x64-04` with
`--labels eshkol,linux-mesh`. Register the CUDA host as `mesh-cuda-01` with
`--labels eshkol,cuda`; the automatic `self-hosted`, `Linux`, and `X64` labels
complete the exact `runs-on` selector used by `linux-x64-cuda`.

- `--name` — use the mesh node name. It is how you will tell runners apart in the
  Settings UI and in `gh api .../actions/runners`.
- `--labels` — from §2. Custom labels only; `self-hosted`, the OS and the arch are
  added for you.
- `--replace` — makes re-registering the same node idempotent.
- `--unattended` — no interactive prompts.

macOS uses the identical `./config.sh`; Windows uses `./config.cmd` with the same
flags.

### 5.3 Install as a service

A runner started from a shell dies with the shell. Install it as a service so it
survives reboots:

```bash
# Linux (systemd)
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status

# macOS (launchd) — no sudo
./svc.sh install
./svc.sh start
./svc.sh status
```

On Windows, `./config.cmd` offers to install the service during configuration;
answer yes.

> `nix-shell nix/jetson/shell.nix` or it will have no CUDA device. Either wrap the

### 5.4 Turn primary routing on

After the four Linux x64 runners and the CUDA runner are online and their
preflight commands pass, the maintainer flips the primary executor with:

Before enabling lanes, seed the shared FetchContent source cache on each Linux
runner with `scripts/mesh/seed_fetchcontent_cache.sh`. The script populates
`$HOME/lanes/_deps`; subsequent lane configurations use
`-DFETCHCONTENT_BASE_DIR="$HOME/lanes/_deps"` and
`-DFETCHCONTENT_FULLY_DISCONNECTED=ON` when the cache marker is present. If
the marker is absent, CI retains its normal network-fetch fallback.

```bash
gh variable set ESHKOL_MESH_PRIMARY --repo tsotchke/eshkol --body on
gh variable set ESHKOL_MESH_CI --repo tsotchke/eshkol --body off
```

The second command explicitly disables the old advisory switch. It is safe if
that variable is already unset. To return to hosted routing:

```bash
gh variable set ESHKOL_MESH_PRIMARY --repo tsotchke/eshkol --body off
```

Future platform flips use JSON label variables. Linux ARM64 takes one full
label array; macOS and Windows take an object with `arm64` and `x64` arrays:

```bash
gh variable set ESHKOL_MESH_ARM64_LABELS --repo tsotchke/eshkol --body '["self-hosted","Linux","ARM64","eshkol","linux-arm64-mesh"]'
gh variable set ESHKOL_MESH_MACOS_LABELS --repo tsotchke/eshkol --body '{"arm64":["self-hosted","macOS","ARM64","eshkol","macos-mesh"],"x64":["self-hosted","macOS","X64","eshkol","macos-mesh"]}'
gh variable set ESHKOL_MESH_WINDOWS_LABELS --repo tsotchke/eshkol --body '{"arm64":["self-hosted","Windows","ARM64","eshkol","windows-mesh"],"x64":["self-hosted","Windows","X64","eshkol","windows-mesh"]}'
```

Those variables are ignored unless `ESHKOL_MESH_PRIMARY=on`. Each value must
name labels actually present on a runner; malformed JSON fails runner
selection rather than silently using a different platform.

---

## 6. Primary routing and required contexts

The 16 required check names do not change and are not renamed by the routing
switch:

`guard`, `assurance-gates`, `surface-manifest`, `linux-x64-xla`,
`linux-arm64-xla`, `linux-x64-cuda`, `linux-arm64-cuda`,
`linux-x64-asan-ubsan`, `wasm-execute-diff`, `windows-arm64-xla`,
`windows-x64-cuda`, `windows-arm64-lite`, `macos-arm64-xla`, `macos-x64-xla`,
`macos-arm64-lite`, `macos-x64-lite`.

With `ESHKOL_MESH_PRIMARY=on`, the routing table is:

| context or lane | `runs-on` labels |
|---|---|
| `guard` | `self-hosted`, `Linux`, `X64`, `eshkol`, `linux-mesh` |
| `assurance-gates`, `surface-manifest` | `self-hosted`, `Linux`, `X64`, `eshkol`, `linux-mesh` |
| `linux-x64-lite`, `linux-x64-xla`, `linux-x64-asan-ubsan` | `self-hosted`, `Linux`, `X64`, `eshkol`, `linux-mesh` |
| `linux-x64-cuda` | `self-hosted`, `Linux`, `X64`, `eshkol`, `cuda` |
| `wasm-execute-diff` | `self-hosted`, `Linux`, `X64`, `eshkol`, `linux-mesh` |
| `linux-arm64-*` | hosted until `ESHKOL_MESH_ARM64_LABELS` is set |
| macOS lanes | hosted until `ESHKOL_MESH_MACOS_LABELS` is set |
| Windows lanes | hosted until `ESHKOL_MESH_WINDOWS_LABELS` is set |

The four Linux x64 runner registrations permit up to four concurrent jobs;
`unix-matrix` advertises `max-parallel: 4`. The CUDA registration is one
runner/job at a time. GitHub's runner matching is not failover: if a selected
label set has no online runner, the required context queues. Keep the primary
variable off until the selected fleet is online and provisioned.

---

## 7. Security

A self-hosted runner executes whatever a workflow tells it to, on a machine you own,
on your network, **with a filesystem that persists between jobs**. On a public
repository that is a real exposure: a pull request from a fork can modify the very
workflow that tests it, so a fork PR reaching a self-hosted runner is arbitrary code
execution on your hardware, and anything it leaves on disk is visible to the next
job.

Controls, in order of load-bearing-ness:

1. **Fork PRs never reach the mesh.** Every job in `ci-mesh.yml` carries
   `if: github.event.pull_request.head.repo.full_name == github.repository`
   (via the `mesh-preflight` gate that all lanes depend on). This is implemented and
   is the control that actually matters — it does not depend on anyone remembering a
   setting.
2. **Require approval for all outside collaborators.** Settings → Actions → General →
   *Fork pull request workflows from outside collaborators*. GitHub's default is
   "require approval for first-time contributors", which still lets a contributor's
   *second* PR run without review. Set it to **all outside collaborators**. This
   repository setting must be verified by an owner before enabling primary routing;
   do not change it from a pull request. The workflow-permission API currently
   reports `default_workflow_permissions` as `read`:

   ```bash
   gh api repos/tsotchke/eshkol/actions/permissions/workflow
   # {"default_workflow_permissions":"read","can_approve_pull_request_reviews":false}
   ```

   That API result does not prove the fork-approval policy. The required policy is
   still **Require approval for all outside collaborators**, plus the workflow's
   same-repository runner-selection guard.
3. **Never put secrets on a mesh lane.** None of the mesh lanes consume repository
   secrets, and none should; a persistent filesystem plus a long-lived machine is a
   poor place for them.
4. **Ephemeral-ish workspace.** Each lane's final step removes its build tree, and
   the disk-budget step fails early rather than filling the node. For stronger
   isolation, register with `--ephemeral` so the runner deregisters after one job and
   is re-registered by a supervising script — at the cost of losing warm build state.

---

## 8. Verifying, and backing out

```bash
# Is it registered and online, and with which labels?
gh api repos/tsotchke/eshkol/actions/runners \
  --jq '.runners[] | {name, status, labels: [.labels[].name]}'

# Force one run without waiting for a PR
gh workflow run ci-mesh.yml --repo tsotchke/eshkol
gh run list --workflow ci-mesh.yml --repo tsotchke/eshkol --limit 5

# The GPU gate specifically (this is what closes ADR-0010 A13)
gh workflow run gpu-execution-gate.yml --repo tsotchke/eshkol
```

A healthy first run shows `mesh-preflight` reporting the online runner count, then
each mesh lane picking up on the node you expect. If a lane sits queued, its labels
match no online runner — compare the job's `runs-on` list against the `labels` in
the `gh api` output above.

### Turning it off

```bash
# Softest: stop dispatching, leave the runners registered.
gh variable set ESHKOL_MESH_CI --repo tsotchke/eshkol --body off
```

```bash
# Full deregistration, on the node:
cd ~/actions-runner
sudo ./svc.sh stop && sudo ./svc.sh uninstall     # omit sudo on macOS
./config.sh remove --token <REMOVAL_TOKEN>
```

Get the removal token from **Settings → Actions → Runners → (runner) → Remove**, or:

```bash
gh api -X POST repos/tsotchke/eshkol/actions/runners/remove-token --jq .token
```

If the machine is gone and cannot be cleaned up locally, delete the registration
from the Settings → Actions → Runners page instead. Because nothing in the required
set depends on a mesh runner, removing one at any time is safe: the mesh lanes stop
reporting and no PR is affected.
