# Focused checked-promotion acceptance suite. Include after the runtime targets
# and their public system dependencies. The parent owns the default-OFF option.
include_guard(GLOBAL)

if(NOT ESHKOL_PROMOTION_TESTING)
    return()
endif()
if(NOT ESHKOL_BUILD_TESTS)
    message(FATAL_ERROR
        "ESHKOL_PROMOTION_TESTING requires ESHKOL_BUILD_TESTS=ON; "
        "instrumenting a runtime without its acceptance tests is not this mode.")
endif()
foreach(_promotion_runtime_target eshkol-runtime eshkol-runtime-core-obj eshkol-runtime-hosted-obj)
    if(NOT TARGET ${_promotion_runtime_target})
        message(FATAL_ERROR "CheckedPromotionTests must be included after ${_promotion_runtime_target}")
    endif()
endforeach()

# Transaction/unwind/identity tests do not need linker interception. The full
# acceptance mode also requires two --wrap fixtures and Itanium new/new[] names;
# do not silently report a portable subset as successful full-suite coverage.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
   NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    message(FATAL_ERROR
        "The complete ESHKOL_PROMOTION_TESTING suite is not available on this "
        "platform: external-caller and no-allocation probes require 64-bit Linux with "
        "GNU/Clang and a --wrap-capable linker. Disable this explicit test mode "
        "for ordinary builds; no substitute allocation proof is provided.")
endif()

target_compile_definitions(eshkol-runtime-core-obj PRIVATE ESHKOL_PROMOTION_TESTING=1)
target_compile_definitions(eshkol-runtime-hosted-obj PRIVATE ESHKOL_PROMOTION_TESTING=1)
find_package(Threads REQUIRED)

function(eshkol_add_checked_promotion_test target)
    add_executable(${target} ${ARGN})
    target_compile_features(${target} PRIVATE cxx_std_20)
    eshkol_apply_common_compile_settings(${target})
    target_compile_definitions(${target} PRIVATE ESHKOL_PROMOTION_TESTING=1)
    target_link_libraries(${target} PRIVATE
        eshkol-runtime ${ESHKOL_EXTRA_LINK_LIBS} Threads::Threads ${CMAKE_DL_LIBS} m)
    # Directory sanitizer compile/link options from CMakeLists apply normally;
    # no fixture bypasses them with a separate compiler invocation.
    add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
    set_tests_properties(${target} PROPERTIES
        LABELS "checked-promotion;native"
        ENVIRONMENT "ESHKOL_ARENA_POISON=1"
        TIMEOUT 60)
endfunction()

eshkol_add_checked_promotion_test(runtime_promotion_transaction_test
    tests/core/runtime_promotion_transaction_test.cpp)
eshkol_add_checked_promotion_test(runtime_promotion_unwind_test
    tests/core/runtime_promotion_unwind_test.cpp)
eshkol_add_checked_promotion_test(runtime_emergency_semantics_test
    tests/core/runtime_emergency_semantics_test.cpp)
eshkol_add_checked_promotion_test(checked_promotion_external_callers_test
    tests/core/checked_promotion_external_callers_test.cpp)
target_link_options(checked_promotion_external_callers_test PRIVATE
    -Wl,--wrap=malloc -Wl,--wrap=realloc -Wl,--wrap=free)
eshkol_add_checked_promotion_test(runtime_promotion_noalloc_transfer_test
    tests/core/runtime_promotion_noalloc_transfer_test.cpp
    tests/core/runtime_promotion_allocation_probe.cpp)
foreach(_promotion_wrapped_symbol malloc calloc realloc aligned_alloc posix_memalign
                                  _Znwm _Znam __cxa_allocate_exception)
    target_link_options(runtime_promotion_noalloc_transfer_test PRIVATE
        "-Wl,--wrap=${_promotion_wrapped_symbol}")
endforeach()

eshkol_add_checked_promotion_test(runtime_promotion_layout_lifetime_test
    tests/core/runtime_promotion_layout_lifetime_test.cpp)
eshkol_add_checked_promotion_test(runtime_root_arena_failure_test
    tests/core/runtime_root_arena_failure_test.cpp)
target_link_options(runtime_root_arena_failure_test PRIVATE
    -Wl,--wrap=arena_create_threadsafe)

# Generated objects use the just-built compiler. Native shims/runtime and final
# linkage inherit the ordinary platform and sanitizer settings; no fixed host
# compiler path or ad-hoc system-library list is used.
function(eshkol_add_promotion_aot target source shim)
    set(object "${CMAKE_CURRENT_BINARY_DIR}/${target}.o")
    add_custom_command(OUTPUT "${object}"
        BYPRODUCTS "${object}.ll"
        COMMAND $<TARGET_FILE:eshkol-run> --no-stdlib -O 2 --dump-ir --compile-only
            -o "${object}" "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        DEPENDS eshkol-run "${source}"
        VERBATIM)
    set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    eshkol_add_checked_promotion_test(${target} "${shim}" "${object}")
endfunction()
eshkol_add_promotion_aot(constructor_emergency_aot
    tests/core/constructor_emergency_test.esk tests/core/constructor_emergency_shim.cpp)
target_link_options(constructor_emergency_aot PRIVATE
    -Wl,--wrap=arena_allocate_vector_with_header
    -Wl,--wrap=arena_allocate_cons_with_header -Wl,--wrap=malloc)
eshkol_add_promotion_aot(checked_barrier_aot
    tests/core/checked_barrier_aot_test.esk tests/core/checked_barrier_aot_shim.cpp)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME checked_promotion_ir_dominance
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/check_checked_barrier_ir.py"
        "${CMAKE_CURRENT_BINARY_DIR}/checked_barrier_aot.o.ll")
add_test(NAME checked_constructor_ir_dominance
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/check_constructor_emergency_ir.py"
        "${CMAKE_CURRENT_BINARY_DIR}/constructor_emergency_aot.o.ll")
set_tests_properties(checked_constructor_ir_dominance PROPERTIES
    LABELS "checked-promotion;ir" TIMEOUT 60)
set_tests_properties(checked_promotion_ir_dominance PROPERTIES
    LABELS "checked-promotion;ir" TIMEOUT 60)

add_custom_target(checked-promotion-tests DEPENDS
    runtime_root_arena_failure_test
    runtime_promotion_layout_lifetime_test constructor_emergency_aot checked_barrier_aot
    runtime_promotion_transaction_test
    runtime_promotion_unwind_test
    runtime_emergency_semantics_test
    checked_promotion_external_callers_test
    runtime_promotion_noalloc_transfer_test)
