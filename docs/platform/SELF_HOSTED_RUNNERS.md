# Self-hosted runners (the mesh)

How to attach the maintainer's own machines to this repository as safe, ephemeral
GitHub Actions runners, what each label means, and what changes once a runner is
online.

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

On top of those, this repository defines exactly five custom labels. Keep the set
small: a label is a contract a lane relies on, and an unmet contract is a lane that
either queues forever or lies.

| label | meaning — assign it only if this is true | consumed by |
|---|---|---|
| `eshkol` | This runner is provisioned for **this repository's** build: LLVM 21, cmake, ninja, python3, and (Linux) `ld.lld`. Every mesh lane requires it. | `ci-mesh.yml` (all lanes), `mesh-preflight`'s online-runner probe |
| `linux-mesh` | The runner is the pinned Ubuntu CI image launched by `launch_ephemeral_runners.sh`; it is not a persistent host runner. | `ci-mesh.yml`'s containerized Linux lane and `mesh-preflight` |
| `gpu` | The host has a **real, addressable GPU device** — not merely a GPU toolchain. `nvidia-smi` lists a device, or the host is Apple Silicon with a working Metal device. | `gpu-execution-gate.yml` (`runs-on: [self-hosted, gpu]`) |
| `cuda` | The host has a CUDA device *and* `nvcc` on `PATH`. Implies `gpu`; assign both. | `mesh-linux-x64-cuda-exec`, `mesh-gpu-gate` |
| `metal` | Apple Silicon host with a working Metal device. Implies `gpu`; assign both. Reserved for a future Metal execution lane. | (none yet) |

The supplied launcher registers `eshkol,linux-mesh`. GPU labels are intentionally
not added by that launcher: a GPU execution image needs a separate device-aware
deployment and must not be implied by a CPU-only container.

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

## 4. Provisioning the container runner

The host only runs Docker and the launcher. The compiler toolchain, Python gate
dependencies, pinned Actions runner, and pinned Emscripten SDK are inside
`scripts/mesh/runner/Dockerfile`. The image uses a non-root `runner` account and
contains no SSH client or cloud credentials.

Build and inspect it from a checkout:

```bash
docker build --tag eshkol-ci-runner:local scripts/mesh/runner
docker run --rm --entrypoint /usr/local/bin/eshkol-toolchain-self-check \
  eshkol-ci-runner:local
```

Seed the shared FetchContent cache on the host before starting jobs. The seed
script uses the lane's `_deps` area, and the launcher exposes it read-only as
`/deps`; the workflow sets `-DFETCHCONTENT_BASE_DIR=/deps` and refuses to run
without the cache marker.

```bash
scripts/mesh/seed_fetchcontent_cache.sh
```

The image is Linux x64. It registers only the `eshkol,linux-mesh` capability
labels, so it cannot accidentally claim an ARM, macOS, or GPU lane.

## 5. Registering ephemeral containers

### 5.1 Create the registration credential

Create a fine-grained personal access token at **Profile picture → Settings →
Developer settings → Fine-grained personal access tokens → Generate new token**.
Choose only this repository, grant **Administration: Read and write**, and set the
expiration to **90 days**. This is the long-lived credential used only to mint
one-shot runner registration tokens; it is not placed in the image or workflow.

### 5.2 Store the credential and configure the launcher

Use a host-only file with mode `600`, and a separate mode-`600` container
environment file copied from `scripts/mesh/runner/runner.env.example`:

```bash
export ESHKOL_RUNNER_TOKEN_FILE="$HOME/.config/eshkol/runner-token"
export ESHKOL_RUNNER_ENV_FILE="$HOME/.config/eshkol/runner-container.env"
export ESHKOL_FETCHCONTENT_CACHE="$HOME/lanes/_deps"
export ESHKOL_RUNNER_REPOSITORY='OWNER/REPOSITORY'
export ESHKOL_RUNNER_URL='https://github.com/OWNER/REPOSITORY'
export ESHKOL_RUNNER_API_URL='https://api.github.com'
export ESHKOL_RUNNER_IMAGE='eshkol-ci-runner:local'
export ESHKOL_RUNNER_PREFIX='eshkol-mesh'
export ESHKOL_RUNNER_LOG_FILE="$HOME/.local/state/eshkol/runner.log"

install -D -m 600 /dev/null "$ESHKOL_RUNNER_TOKEN_FILE"
IFS= read -r -s RUNNER_PAT
printf '%s' "$RUNNER_PAT" > "$ESHKOL_RUNNER_TOKEN_FILE"
unset RUNNER_PAT
install -D -m 600 scripts/mesh/runner/runner.env.example "$ESHKOL_RUNNER_ENV_FILE"
```

For the user service, put the same non-secret launcher settings in its host-only
environment file (including the absolute paths selected above):

```bash
export ESHKOL_RUNNER_LAUNCHER_ENV="$HOME/.config/eshkol/runner-launcher.env"
install -D -m 600 /dev/null "$ESHKOL_RUNNER_LAUNCHER_ENV"
{
  printf '%s\n' "ESHKOL_RUNNER_TOKEN_FILE=$ESHKOL_RUNNER_TOKEN_FILE"
  printf '%s\n' "ESHKOL_RUNNER_ENV_FILE=$ESHKOL_RUNNER_ENV_FILE"
  printf '%s\n' "ESHKOL_FETCHCONTENT_CACHE=$ESHKOL_FETCHCONTENT_CACHE"
  printf '%s\n' "ESHKOL_RUNNER_REPOSITORY=$ESHKOL_RUNNER_REPOSITORY"
  printf '%s\n' "ESHKOL_RUNNER_URL=$ESHKOL_RUNNER_URL"
  printf '%s\n' "ESHKOL_RUNNER_API_URL=$ESHKOL_RUNNER_API_URL"
  printf '%s\n' "ESHKOL_RUNNER_IMAGE=$ESHKOL_RUNNER_IMAGE"
  printf '%s\n' "ESHKOL_RUNNER_PREFIX=$ESHKOL_RUNNER_PREFIX"
  printf '%s\n' "ESHKOL_RUNNER_LOG_FILE=$ESHKOL_RUNNER_LOG_FILE"
} > "$ESHKOL_RUNNER_LAUNCHER_ENV"
chmod 600 "$ESHKOL_RUNNER_LAUNCHER_ENV"
```

The launcher reads the credential only on the host, uses it to mint a fresh
registration token for each slot, and passes that short-lived token only as the
`--token` argument to `config.sh` at container start. It never mounts the
credential file.

### 5.3 Start and keep the launcher alive

Start one or more slots from the checkout:

```bash
scripts/mesh/runner/launch_ephemeral_runners.sh N
```

Each slot uses `--ephemeral`, `--rm`, a read-only image root, resource caps, no
privileges or Docker socket, the default Docker network, a read-only `/deps` bind,
and a per-slot `/work` volume. The volume is removed whenever that container
exits. The default cap is 16 CPUs and 32 GiB of memory per slot; operators may
lower either cap with `ESHKOL_RUNNER_CPUS` and `ESHKOL_RUNNER_MEMORY`. The launcher
prints the exact systemd --user commands for restart-on-login:

```bash
install -D -m 600 scripts/mesh/runner/eshkol-mesh-runner.service \
  "$HOME/.config/systemd/user/eshkol-mesh-runner.service"
systemctl --user daemon-reload
systemctl --user enable --now eshkol-mesh-runner.service
```

The service reads the launcher variables from
`$HOME/.config/eshkol/runner-launcher.env`; keep that file host-only and do not
put credentials other than the token-file path in it.

### 5.4 Turn the mesh lanes on

`ci-mesh.yml` ships **off**. After at least one runner is online, confirm the
container self-check passed, the cache marker exists, and the runner list shows
only ephemeral containers carrying both `eshkol` and `linux-mesh`.

```bash
gh variable set ESHKOL_MESH_CI --repo tsotchke/eshkol --body on
```

`gpu-execution-gate.yml` needs no variable — it dispatches to `[self-hosted, gpu]`
as soon as such a runner exists. The advisory `mesh-gpu-gate` lane in
`ci-mesh.yml` requires `[self-hosted, Linux, X64, eshkol, cuda]` and runs
`tests/gpu/gpu_correctness_gate.sh`; its exit code is the primary verdict.
The lane remains outside branch protection and is controlled by the mesh
preflight dispatch guard.

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

An untrusted pull request can modify the workflow that tests it. If that workflow
reaches a self-hosted runner, it has arbitrary code execution inside the runner's
security boundary. The assets at risk are the host filesystem, Docker control,
registration credentials, network access, and data left by an earlier job.

Controls, in order of importance:

1. **Fork PRs never reach the mesh.** The hosted `mesh-preflight` gate rejects fork
   pull requests before any self-hosted job can be dispatched. Every self-hosted job
   also checks the container markers before checkout.
2. **Require approval for all outside collaborators.** In the repository UI, go to
   **Settings → Actions → General**. Under **Approval for running fork pull request
   workflows from contributors**, select **Require approval for all external
   contributors**, then click **Save**. Review proposed workflow changes before
   approving a fork workflow.
3. **Ephemeral containers and no host mounts.** The launcher uses `--rm`, a read-only
   root, dropped capabilities, no privileged mode, no Docker socket, resource caps,
   the default network, and only a read-only `/deps` cache plus a per-slot `/work`
   volume. The work volume is removed after every container exit.
4. **Least-privilege registration credential.** The token file is host-only, mode
   `600`, and never mounted into the container. The launcher uses it only to mint a
   one-shot registration token and redacts that token from its log.
5. **No secrets on self-hosted jobs.** No self-hosted job references repository
   secrets. Workflow permissions default to `contents: read` and `id-token: none`.

### Operator checklist

Before enabling `ESHKOL_MESH_CI`:

- Build the pinned image and run `eshkol-toolchain-self-check`.
- Seed the shared FetchContent cache with
  `scripts/mesh/seed_fetchcontent_cache.sh`; expose that directory read-only as
  `/deps`.
- Copy `runner.env.example` to a host-only runner environment file. Keep
  `FETCHCONTENT_BASE_DIR=/deps`; do not put credentials in this file.
- Create a fine-grained personal access token through **Profile picture → Settings
  → Developer settings → Fine-grained personal access tokens → Generate new token**.
  Select only this repository, **Administration: Read and write**, and a **90-day
  expiration**.
- Store only that token in `ESHKOL_RUNNER_TOKEN_FILE`, set its mode to `600`, and
  confirm the file is not in the checkout or the cache.
- Install the supplied user unit, start the launcher, and confirm the GitHub runner
  list shows only ephemeral `eshkol,linux-mesh` runners.
- Enable the repository variable `ESHKOL_MESH_CI=on` only after the checks above pass.

The launcher prints the exact `systemctl --user` commands for persistent startup.
Install `eshkol-mesh-runner.service` as a user unit before running those commands.

### Credential rotation

1. Turn `ESHKOL_MESH_CI` off and stop the user service. Existing ephemeral
   containers finish or can be allowed to exit; do not reuse their registration
   tokens.
2. In **Settings → Actions → Runners**, remove any stale runner registrations.
3. Revoke the old token from the account's fine-grained token settings.
4. Create a replacement with the same repository-only scope, **Administration: Read
   and write** permission, and a new 90-day expiration.
5. Replace the contents of the mode-`600` token file without adding it to the
   repository, run the image self-check, and restart the launcher.
6. Confirm that a fresh job uses a new ephemeral registration, then set
   `ESHKOL_MESH_CI=on` again.

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
# Stop the supervisor. Each container is --rm and each runner is --ephemeral.
systemctl --user disable --now eshkol-mesh-runner.service
```

If a container was interrupted before Docker could remove it, the launcher cleanup
removes only the named containers and per-slot volumes it created. If a stale
registration remains in **Settings → Actions → Runners**, remove that registration
there. Because nothing in the required set depends on a mesh runner, stopping it is
safe: the mesh lanes stop reporting and no PR is affected.
