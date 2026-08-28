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

Two distinct wins, worth keeping separate because they have different risk profiles.

**Capacity.** `ci.yml` runs a 14-lane matrix on GitHub-hosted runners. Moving the
Linux and macOS *lite* lanes onto owned hardware returns those minutes to the pool
for the lanes that genuinely need a clean disposable image (Windows SDK downloads,
the ASan lane, the WASM diff).

**Coverage that hosted runners structurally cannot provide.** The `linux-x64-cuda`,
`linux-arm64-cuda` and `windows-x64-cuda` lanes in `ci.yml` run on hosted runners,
and **hosted runners have no GPU**. `nvcc` compiles the kernels, `eshkol_gpu_init()`
reports zero devices, and every GPU tensor op falls through to the CPU path without
saying so. Those lanes are *compilation* gates. A registered GPU runner turns
`gpu-execution-gate.yml` and `mesh-linux-x64-cuda-exec` into real *execution* gates
— which is also how ADR-0010 gap **A13** closes: that workflow currently emits a
`::warning` on every run saying it produced no GPU evidence, because no
`[self-hosted, gpu]` runner has ever existed for this repo.

**What it does not buy: a merge gate.** No job in `ci-mesh.yml` is, or may become,
a required status check. See §6.

---

## 2. Label taxonomy

GitHub applies three labels automatically at registration and they cannot be
removed: `self-hosted`, an OS label (`Linux` / `macOS` / `Windows`), and an
architecture label (`X64` / `ARM64` / `ARM`). Label matching is case-insensitive,
which is why the workflows spell them lowercase.

On top of those, this repository defines exactly four custom labels. Keep the set
small: a label is a contract a lane relies on, and an unmet contract is a lane that
either queues forever or lies.

| label | meaning — assign it only if this is true | consumed by |
|---|---|---|
| `eshkol` | This runner is provisioned for **this repository's** build: LLVM 21, cmake, ninja, python3, and (Linux) `ld.lld`. Every mesh lane requires it. | `ci-mesh.yml` (all lanes), `mesh-preflight`'s online-runner probe |
| `gpu` | The host has a **real, addressable GPU device** — not merely a GPU toolchain. `nvidia-smi` lists a device, or the host is Apple Silicon with a working Metal device. | `gpu-execution-gate.yml` (`runs-on: [self-hosted, gpu]`) |
| `cuda` | The host has a CUDA device *and* `nvcc` on `PATH`. Implies `gpu`; assign both. | `mesh-linux-x64-cuda-exec` |
| `metal` | Apple Silicon host with a working Metal device. Implies `gpu`; assign both. Reserved for a future Metal execution lane. | (none yet) |

So a full label set looks like `--labels eshkol` for a plain build node, or
`--labels eshkol,gpu,cuda` for the CUDA execution node.

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

| node | OS / arch | cores / RAM / free disk | GPU | suggested labels | provisioning still needed |
|---|---|---|---|---|---|

Every other node in the registry was unreachable at survey time: the GCP
Blackwell/RTX-Pro nodes are stopped or spot-preempted, the on-prem boxes are
this machine. That volatility is the reason for §6.

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
  cmake ninja-build git python3 pkg-config \
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

If that passes by hand, the mesh lane will pass. If it does not, fix it here — a
runner registered against a broken toolchain produces red lanes that look like code
regressions.

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
  --labels eshkol,gpu,cuda \
  --work _work \
  --unattended \
  --replace
```

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

### 5.4 Turn the mesh lanes on

`ci-mesh.yml` ships **off**. After at least one runner is online:

Before enabling lanes, seed the shared FetchContent source cache on each Linux
runner with `scripts/mesh/seed_fetchcontent_cache.sh`. The script populates
`$HOME/lanes/_deps`; subsequent lane configurations use
`-DFETCHCONTENT_BASE_DIR="$HOME/lanes/_deps"` and
`-DFETCHCONTENT_FULLY_DISCONNECTED=ON` when the cache marker is present. If
the marker is absent, CI retains its normal network-fetch fallback.

```bash
gh variable set ESHKOL_MESH_CI --repo tsotchke/eshkol --body on
```

`gpu-execution-gate.yml` needs no variable — it dispatches to `[self-hosted, gpu]`
as soon as such a runner exists.

---

## 6. What must NOT change

**No job in `ci-mesh.yml` may be added to branch protection's required contexts.**

The required set is, and stays: `guard`, `linux-x64-xla`, `linux-arm64-xla`,
`linux-x64-cuda`, `linux-arm64-cuda`, `linux-x64-asan-ubsan`, `wasm-execute-diff`,
`windows-arm64-xla`, `windows-x64-cuda` — all hosted.

The reason is mechanical, not conservative. Branch protection has no timeout: a
required context that never reports leaves the PR blocked forever, and the only
recovery is an admin editing the protection rule. This repository has already paid
that price once, on PR #444. A required context whose runner is one physical machine
inherits that machine's availability — and §3 measured this fleet directly: most of
it is off at any given moment, and every datacenter-GPU node in it is a preemptible
spot instance.

This is also why the mesh lanes live in their own workflow rather than as a
"prefer self-hosted, fall back to hosted" matrix inside `ci.yml`. **That fallback
does not exist in GitHub Actions.** `runs-on` is resolved at schedule time; a job
whose labels match no online runner does not fail over to a hosted runner, it
queues. Making a required lane's `runs-on` depend on fleet state would convert every
mesh outage into a permanently pending required check. Separate job names in a
separate file make that mistake structurally impossible.

Promoting a mesh lane to required is a deliberate maintainer decision that requires,
at minimum: the runner demonstrably online across a sustained period, and a plan for
what happens to open PRs when it is not.

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
   *second* PR run without review. Set it to **all outside collaborators**.
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
