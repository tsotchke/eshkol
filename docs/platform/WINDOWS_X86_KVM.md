# Windows 11 x64 test VM on Linux KVM (old-donkey / cosbox)

Status as of 2026-07-03: **scaffolding provisioned on old-donkey; awaiting a
Windows 11 installer ISO** (manual download step below). No VM has been booted
yet.

The mesh has no native Windows x64 CI runner. jack-blupc covers Windows x64
bare-metal, but a KVM guest on the Linux x64 nodes gives us a reproducible,
snapshotable Windows environment for release testing.

## Host inventory

| Host | KVM | QEMU | OVMF (secure boot) | swtpm | Free disk | Fit |
|------|-----|------|--------------------|-------|-----------|-----|
| old-donkey | yes (user in `kvm`,`libvirt`) | 8.2.2 | yes (`OVMF_*_4M.ms.fd`) | yes | 358 G | **primary** |
| cosbox | yes (device present, user NOT in `kvm` group) | 8.2.2 + virt-install | yes | untested | 21 G (96% full) | fallback only |

old-donkey note: outbound HTTPS to `github.com` and `vlscppe.microsoft.com`
is blocked from this host; general internet (e.g. `microsoft.com`) works.
Transfer files over SSH/scp from another mesh node when needed.

## What is already provisioned (old-donkey)

```
~/vms/win11-x64/
  win11.qcow2        # 80 G thin qcow2 system disk (empty)
  OVMF_VARS.fd       # writable UEFI vars, copied from OVMF_VARS_4M.ms.fd
  start-win11.sh     # launch script (same as scripts/platform/start-win11-kvm.sh)
```

The launch script (`scripts/platform/start-win11-kvm.sh`) boots a q35 machine
with KVM acceleration, secure-boot OVMF, an emulated TPM 2.0 (swtpm) — the
Windows 11 hardware requirements — plus an AHCI disk and e1000 NIC so the
stock Windows installer needs no extra drivers. It forwards guest SSH to host
port 22222 and serves the console on VNC display `:11` (localhost only).

## Remaining manual steps

1. **Download the installer ISO** (requires an interactive browser; Microsoft's
   download API rejects non-browser sessions):
   - Visit <https://www.microsoft.com/en-us/software-download/windows11>
   - Select "Windows 11 (multi-edition ISO for x64 devices)", language
     "English (United States)", and download (~6.5 GB).
   - Copy it to the host: `scp Win11_*.iso old-donkey:~/vms/win11-x64/`
2. **First boot / install**:
   ```
   ssh old-donkey
   cd ~/vms/win11-x64
   ./start-win11.sh ~/vms/win11-x64/Win11_*.iso
   ```
   Tunnel VNC from your workstation: `ssh -L 5911:localhost:5911 old-donkey`,
   then connect a VNC client to `localhost:5911`. Press a key at the "Press any
   key to boot from CD" prompt.
3. **Inside Windows**: complete setup (a local account is fine), then
   - `Settings > System > Optional features > OpenSSH Server` (or
     `Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0`), start the
     `sshd` service and set it to automatic. The guest is then reachable with
     `ssh -p 22222 <user>@localhost` from old-donkey.
4. **Toolchain for Eshkol CI-equivalent runs** (mirrors `.github/workflows/ci.yml`):
   - Visual Studio 2022 Build Tools with the "Desktop development with C++"
     workload, CMake, Git.
   - LLVM SDK: `clang+llvm-21.1.8-x86_64-pc-windows-msvc.tar.xz` from
     `https://github.com/llvm/llvm-project/releases/tag/llvmorg-21.1.8`,
     extracted to `C:\src\` (exactly what CI's "Install LLVM SDK" step does).
   - Configure/build/test with the same commands as the `windows-*` CI lanes
     (`-G 'Visual Studio 17 2022' -A x64 -T ClangCL`, then
     `scripts\run_all_tests.ps1 -BuildDir .\build -SkipConfigureBuild`).
5. **Optional performance**: install the virtio-win driver ISO
   (<https://fedorapeople.org/groups/virt/virtio-win/direct-downloads/>) and
   switch the disk to `if=virtio` / NIC to `virtio-net-pci` in the launch
   script after drivers are in.
6. **Snapshot before experiments**: `qemu-img snapshot -c clean-install
   ~/vms/win11-x64/win11.qcow2` (with the VM shut down).

## cosbox caveats

cosbox has the QEMU/OVMF stack but `tyr` is not in the `kvm` group (needs
`sudo usermod -aG kvm tyr` + re-login) and the disk is 96% full — clear space
before hosting a VM there.
