# EshkolCudaArchitectures.cmake -- the CUDA architecture policy (#606).
#
# The portable default list (ESHKOL_CUDA_ARCHITECTURES) names the GPU
# generations Eshkol is built for. Which of them a given build can target is a
# property of the installed toolkit, not of Eshkol: CUDA 13 removed offline
# compilation below compute capability 7.5, older toolkits cannot compile for
# the newest generations. A hard-coded list therefore breaks configure on every
# toolkit it was not written against -- `nvcc fatal: Unsupported gpu
# architecture 'compute_72'` on CUDA 13 -- before the compiler check can run.
#
# So the list is resolved against the toolkit in hand:
#
#   1. Ask nvcc. `nvcc --list-gpu-arch` prints every virtual architecture the
#      compiler accepts (compute_75, compute_80, ...). That is the authority.
#   2. When nvcc cannot answer (a toolkit without the flag, or a wrapper that
#      does not forward it), fall back to the documented range for the
#      toolkit version: the lowest and highest compute capability each CUDA
#      release compiles for.
#
# Entries the toolkit does not support are dropped with a status message; if
# nothing is left the configure stops and says why. An explicit
# CMAKE_CUDA_ARCHITECTURES stays authoritative and is never filtered: a user
# who names an architecture gets exactly that, and nvcc's own diagnostic if it
# is wrong.

# Numeric compute capability of one CMake CUDA architecture entry ("90a",
# "86-real", "75-virtual" -> 90, 86, 75), or "" for a non-numeric entry such as
# "all", "all-major" or "native".
function(eshkol_cuda_arch_number entry out_var)
    if(entry MATCHES "^([0-9]+)")
        set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

# eshkol_cuda_supported_architectures(<nvcc> <toolkit-version> <out-list> <out-source>)
#
# <out-list>   numeric compute capabilities the toolkit compiles for; for the
#              version fallback, the bounds "<min>;<max>" prefixed by "RANGE"
# <out-source> "nvcc" or "version"
function(eshkol_cuda_supported_architectures nvcc version out_list out_source)
    set(supported "")
    if(nvcc AND EXISTS "${nvcc}")
        execute_process(
            COMMAND "${nvcc}" --list-gpu-arch
            RESULT_VARIABLE rc
            OUTPUT_VARIABLE listing
            ERROR_QUIET
            TIMEOUT 60)
        if(rc EQUAL 0)
            string(REGEX MATCHALL "compute_[0-9]+" arches "${listing}")
            foreach(arch IN LISTS arches)
                string(REPLACE "compute_" "" number "${arch}")
                if(NOT number IN_LIST supported)
                    list(APPEND supported "${number}")
                endif()
            endforeach()
        endif()
    endif()
    if(supported)
        set(${out_list} "${supported}" PARENT_SCOPE)
        set(${out_source} "nvcc" PARENT_SCOPE)
        return()
    endif()

    # Documented offline-compilation range of each CUDA release.
    if(version VERSION_GREATER_EQUAL 13.0)
        set(low 75)
    elseif(version VERSION_GREATER_EQUAL 12.0)
        set(low 50)
    elseif(version VERSION_GREATER_EQUAL 11.0)
        set(low 35)
    else()
        set(low 30)
    endif()
    if(version VERSION_GREATER_EQUAL 12.8)
        set(high 999)
    elseif(version VERSION_GREATER_EQUAL 11.8)
        set(high 90)
    elseif(version VERSION_GREATER_EQUAL 11.1)
        set(high 86)
    elseif(version VERSION_GREATER_EQUAL 11.0)
        set(high 80)
    else()
        set(high 75)
    endif()
    set(${out_list} "RANGE;${low};${high}" PARENT_SCOPE)
    set(${out_source} "version" PARENT_SCOPE)
endfunction()

# eshkol_cuda_filter_architectures(<requested> <supported> <out-kept> <out-dropped>)
#
# Keep the entries of <requested> that <supported> (as returned above) allows.
# Non-numeric entries ("all", "native", ...) are kept: CMake and nvcc resolve
# them against the toolkit themselves.
function(eshkol_cuda_filter_architectures requested supported out_kept out_dropped)
    set(kept "")
    set(dropped "")
    set(range_mode FALSE)
    if(supported MATCHES "^RANGE;")
        set(range_mode TRUE)
        list(GET supported 1 low)
        list(GET supported 2 high)
    endif()
    foreach(entry IN LISTS requested)
        eshkol_cuda_arch_number("${entry}" number)
        if(number STREQUAL "")
            list(APPEND kept "${entry}")
        elseif(range_mode)
            if(number LESS low OR number GREATER high)
                list(APPEND dropped "${entry}")
            else()
                list(APPEND kept "${entry}")
            endif()
        elseif(number IN_LIST supported)
            list(APPEND kept "${entry}")
        else()
            list(APPEND dropped "${entry}")
        endif()
    endforeach()
    set(${out_kept} "${kept}" PARENT_SCOPE)
    set(${out_dropped} "${dropped}" PARENT_SCOPE)
endfunction()

# eshkol_resolve_cuda_architectures(<requested> <nvcc> <toolkit-version> <out-var> <out-supported>)
#
# The whole policy: resolve <requested> against the toolkit, report what was
# dropped, and stop the configure when nothing remains. <out-supported>
# receives the toolkit's supported set, which the caller records for the
# build-graph verifier (scripts/verify_gpu_backend.py).
function(eshkol_resolve_cuda_architectures requested nvcc version out_var out_supported)
    eshkol_cuda_supported_architectures("${nvcc}" "${version}" supported source)
    eshkol_cuda_filter_architectures("${requested}" "${supported}" kept dropped)
    if(source STREQUAL "nvcc")
        set(how "nvcc --list-gpu-arch")
    else()
        set(how "the CUDA ${version} documented range")
    endif()
    if(dropped)
        string(REPLACE ";" ", " dropped_text "${dropped}")
        message(STATUS
            "CUDA ${version}: dropping unsupported architectures ${dropped_text} "
            "from the portable default (per ${how})")
    endif()
    if(NOT kept)
        message(FATAL_ERROR
            "None of the requested CUDA architectures (${requested}) are supported "
            "by CUDA ${version} (per ${how}). Set ESHKOL_CUDA_ARCHITECTURES or "
            "CMAKE_CUDA_ARCHITECTURES to architectures this toolkit supports.")
    endif()
    set(${out_var} "${kept}" PARENT_SCOPE)
    set(${out_supported} "${supported}" PARENT_SCOPE)
endfunction()
