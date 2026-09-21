/**
 * @file link_probe.h
 * @brief Host linker probes shared by the AOT driver and the JIT link path.
 */
#ifndef ESHKOL_BACKEND_LINK_PROBE_H
#define ESHKOL_BACKEND_LINK_PROBE_H

/**
 * @brief True when `ld.lld` is on the PATH. Cached after the first call.
 *
 * AArch64 Linux links through lld when it is present (GNU ld 2.38 mishandles
 * large user binaries). Passing -fuse-ld=lld on a host without lld fails every
 * link with "invalid linker name", so callers probe first and fall back to the
 * default linker, warning once.
 */
bool eshkol_lld_on_path();

#endif  // ESHKOL_BACKEND_LINK_PROBE_H
