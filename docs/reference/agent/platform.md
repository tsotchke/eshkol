# Platform filesystem ABI

The public `eshkol/agent_platform.h` header exposes file metadata for hosted
filesystem clients. `eshkol_file_stat_fields` remains the compatibility ABI;
`eshkol_file_stat_fields_v2` adds epoch-nanosecond modification time and a
stable identity.

The v2 identity is `(device, inode)` on POSIX and `(volume serial, file index)`
on Windows. Both ABIs use `lstat`-equivalent semantics, so a symbolic link is
identified as the directory entry itself. The original epoch-seconds and file
type outputs remain in the same positions for compatibility.
