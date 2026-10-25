/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_RUNTIME_WINDOWS_X64_CONTINUATION_H
#define ESHKOL_RUNTIME_WINDOWS_X64_CONTINUATION_H

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
#include <windows.h>
#include <cstdlib>

namespace eshkol {

// Only for transfer AFTER restoring a saved stack image and Eshkol dynamic
// state. Ordinary longjmp would unwind frames whose bytes were overwritten.
// STATUS_LONGJUMP copies nonvolatile state from the jump buffer, while Rax
// supplies setjmp's return value. Keep Windows APIs out of freestanding core.
// https://learn.microsoft.com/en-us/windows/win32/api/winnt/nf-winnt-rtlrestorecontext
[[noreturn]] inline void restoreWindowsX64Continuation(void* jump_buffer, int value) {
    CONTEXT context{};
    RtlCaptureContext(&context);
    context.Rax = static_cast<DWORD64>(value == 0 ? 1 : value);
    EXCEPTION_RECORD exception{};
    exception.ExceptionCode = STATUS_LONGJUMP;
    exception.NumberParameters = 1;
    exception.ExceptionInformation[0] = reinterpret_cast<ULONG_PTR>(jump_buffer);
    RtlRestoreContext(&context, &exception);
    std::abort();
}

} // namespace eshkol
#endif
#endif
