// Linux linker probe for references from objects included in this executable.
// --wrap does NOT intercept hidden/internal allocation within shared libc or
// libstdc++; counters are deliberately not presented as whole-process evidence.
#include "runtime_promotion_allocation_probe.h"
#include <cerrno>
#include <cstddef>
#include <new>
static bool armed=false;
static PromotionAllocationProbe counts{};
void promotion_allocation_probe_arm() noexcept {counts={};armed=true;}
PromotionAllocationProbe promotion_allocation_probe_disarm() noexcept {armed=false;return counts;}
extern "C" {
void* __real_malloc(size_t);void* __real_calloc(size_t,size_t);void* __real_realloc(void*,size_t);
void* __real_aligned_alloc(size_t,size_t);int __real_posix_memalign(void**,size_t,size_t);
void* __real__Znwm(size_t);void* __real__Znam(size_t);
void* __real___cxa_allocate_exception(size_t) noexcept;
void* __wrap_malloc(size_t n){if(armed){++counts.malloc_calls;errno=ENOMEM;return nullptr;}return __real_malloc(n);}
void* __wrap_calloc(size_t n,size_t size){if(armed){++counts.calloc_calls;errno=ENOMEM;return nullptr;}return __real_calloc(n,size);}
void* __wrap_realloc(void* p,size_t n){if(armed){++counts.realloc_calls;errno=ENOMEM;return nullptr;}return __real_realloc(p,n);}
void* __wrap_aligned_alloc(size_t align,size_t n){if(armed){++counts.aligned_alloc_calls;errno=ENOMEM;return nullptr;}return __real_aligned_alloc(align,n);}
int __wrap_posix_memalign(void** p,size_t align,size_t n){if(armed){++counts.posix_memalign_calls;return ENOMEM;}return __real_posix_memalign(p,align,n);}
void* __wrap__Znwm(size_t n){if(armed){++counts.new_calls;throw std::bad_alloc();}return __real__Znwm(n);}
void* __wrap__Znam(size_t n){if(armed){++counts.new_array_calls;throw std::bad_alloc();}return __real__Znam(n);}
// Observe exception allocation separately; never make __cxa_allocate_exception
// return null or conflate its implementation-dependent emergency pool with new.
void* __wrap___cxa_allocate_exception(size_t n) noexcept {if(armed)++counts.exception_allocations;return __real___cxa_allocate_exception(n);}
}
