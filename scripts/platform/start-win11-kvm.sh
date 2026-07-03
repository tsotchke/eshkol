#!/usr/bin/env bash
# Launch the Windows 11 x64 KVM guest on old-donkey (or any Linux KVM host
# with OVMF + swtpm installed). First boot: pass the installer ISO as $1.
#
#   ./start-win11.sh ~/vms/win11-x64/Win11_25H2_English_x64.iso   # install
#   ./start-win11.sh                                              # normal boot
#
# Console: VNC on localhost:5911 (tunnel with `ssh -L 5911:localhost:5911 old-donkey`).
# After install, enable OpenSSH Server inside Windows for headless use.
set -euo pipefail

VMDIR="$HOME/vms/win11-x64"
ISO="${1:-}"

# Software TPM 2.0 (Windows 11 hard requirement)
mkdir -p "$VMDIR/tpm"
if ! pgrep -f "swtpm socket.*$VMDIR/tpm" >/dev/null; then
    swtpm socket --tpm2 \
        --tpmstate dir="$VMDIR/tpm" \
        --ctrl type=unixio,path="$VMDIR/tpm/swtpm-sock" \
        --daemon
fi

ISO_ARGS=()
if [ -n "$ISO" ]; then
    ISO_ARGS+=(-drive "file=$ISO,media=cdrom,if=none,id=cd0" -device "usb-storage,drive=cd0")
fi

exec qemu-system-x86_64 \
    -machine q35,accel=kvm,smm=on \
    -cpu host \
    -smp 8 -m 16G \
    -global driver=cfi.pflash01,property=secure,value=on \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.ms.fd \
    -drive if=pflash,format=raw,file="$VMDIR/OVMF_VARS.fd" \
    -chardev socket,id=chrtpm,path="$VMDIR/tpm/swtpm-sock" \
    -tpmdev emulator,id=tpm0,chardev=chrtpm \
    -device tpm-tis,tpmdev=tpm0 \
    -drive "file=$VMDIR/win11.qcow2,if=ide,format=qcow2,discard=unmap" \
    -device usb-ehci -device usb-tablet \
    -nic user,model=e1000,hostfwd=tcp::22222-:22 \
    -vga std \
    -vnc 127.0.0.1:11 \
    "${ISO_ARGS[@]}"
