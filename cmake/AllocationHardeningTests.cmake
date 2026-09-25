# Linux linker wrappers provide deterministic allocator failures without
# introducing production failpoints. This suite is explicit and opt-in.
option(ESHKOL_ALLOCATION_TESTING "Build constructor/handler allocation failure tests" OFF)
if(NOT ESHKOL_ALLOCATION_TESTING)
    return()
endif()
if(NOT ESHKOL_BUILD_TESTS OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    message(FATAL_ERROR "ESHKOL_ALLOCATION_TESTING requires tests on Linux with GNU/Clang and --wrap")
endif()
target_compile_definitions(eshkol-runtime-hosted-obj PRIVATE ESHKOL_ALLOCATION_TESTING=1)
find_package(Threads REQUIRED)
function(eshkol_allocation_test target)
    add_executable(${target} ${ARGN})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    target_link_libraries(${target} PRIVATE eshkol-runtime ${ESHKOL_EXTRA_LINK_LIBS}
        Threads::Threads ${CMAKE_DL_LIBS} m)
    add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
    set_tests_properties(${target} PROPERTIES LABELS "allocation-hardening" TIMEOUT 60)
endfunction()
eshkol_allocation_test(runtime_allocation_hardening_test tests/core/runtime_allocation_hardening_test.cpp)
target_link_options(runtime_allocation_hardening_test PRIVATE -Wl,--wrap=malloc -Wl,--wrap=calloc
    -Wl,--wrap=arena_allocate_aligned)
function(eshkol_allocation_aot target source shim)
    set(object "${CMAKE_CURRENT_BINARY_DIR}/${target}.o")
    add_custom_command(OUTPUT "${object}" BYPRODUCTS "${object}.ll"
        COMMAND $<TARGET_FILE:eshkol-run> --no-stdlib -O 2 --dump-ir --compile-only
            -o "${object}" "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        DEPENDS eshkol-run "${source}" VERBATIM)
    set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    eshkol_allocation_test(${target} "${shim}" "${object}")
endfunction()
eshkol_allocation_aot(constructor_allocation_aot tests/core/constructor_allocation_test.esk
    tests/core/constructor_allocation_shim.cpp)
target_link_options(constructor_allocation_aot PRIVATE -Wl,--wrap=arena_allocate_vector_with_header
    -Wl,--wrap=arena_allocate_cons_with_header -Wl,--wrap=arena_allocate_closure_with_header
    -Wl,--wrap=malloc -Wl,--wrap=arena_allocate_aligned)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME checked_constructor_ir_dominance COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/check_constructor_allocation_ir.py"
    "${CMAKE_CURRENT_BINARY_DIR}/constructor_allocation_aot.o.ll")
set_tests_properties(checked_constructor_ir_dominance PROPERTIES
    LABELS "allocation-hardening" TIMEOUT 60)
add_custom_target(allocation-hardening-tests DEPENDS runtime_allocation_hardening_test
    constructor_allocation_aot)
