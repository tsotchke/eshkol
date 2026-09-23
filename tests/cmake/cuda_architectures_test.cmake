# tests/cmake/cuda_architectures_test.cmake -- #606.
#
# Exercises cmake/EshkolCudaArchitectures.cmake against mocked nvcc
# executables, so the CUDA architecture policy is verified on hosts without a
# CUDA toolkit. Run as:
#   cmake -DSOURCE_DIR=<repo> -DWORK_DIR=<scratch dir> -P tests/cmake/cuda_architectures_test.cmake

if(NOT SOURCE_DIR OR NOT WORK_DIR)
    message(FATAL_ERROR "SOURCE_DIR and WORK_DIR are required")
endif()
include("${SOURCE_DIR}/cmake/EshkolCudaArchitectures.cmake")

set(DEFAULT_LIST "72;75;80;86;89;90")
set(failures 0)

macro(expect name got want)
    if(NOT "${got}" STREQUAL "${want}")
        message(SEND_ERROR "FAIL: ${name}: got '${got}', want '${want}'")
        math(EXPR failures "${failures} + 1")
    else()
        message(STATUS "ok   ${name}")
    endif()
endmacro()

file(MAKE_DIRECTORY "${WORK_DIR}")

# A mock nvcc that answers --list-gpu-arch with the given virtual architectures.
function(write_mock_nvcc path arches)
    set(lines "")
    foreach(a IN LISTS arches)
        string(APPEND lines "compute_${a}\n")
    endforeach()
    file(WRITE "${path}"
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--list-gpu-arch\" ]; then\n"
        "cat <<'LIST'\n${lines}LIST\n"
        "exit 0\nfi\nexit 1\n")
    execute_process(COMMAND chmod +x "${path}")
endfunction()

# The CUDA 13.0 listing (the toolkit from the issue: SM 7.5 and newer).
write_mock_nvcc("${WORK_DIR}/nvcc13" "75;80;86;87;88;89;90;100;103;110;120;121")
# The CUDA 12.4 listing (SM 5.0 through 9.0).
write_mock_nvcc("${WORK_DIR}/nvcc12" "50;52;53;60;61;62;70;72;75;80;86;87;89;90")
# An nvcc that cannot list its architectures.
file(WRITE "${WORK_DIR}/nvcc_nolist" "#!/bin/sh\nexit 1\n")
execute_process(COMMAND chmod +x "${WORK_DIR}/nvcc_nolist")

# 1. CUDA 13: the issue. compute_72 is gone, everything else stays.
eshkol_resolve_cuda_architectures("${DEFAULT_LIST}" "${WORK_DIR}/nvcc13" "13.0.88" kept supported)
expect("CUDA 13 drops SM72" "${kept}" "75;80;86;89;90")
expect("CUDA 13 supported set comes from nvcc"
       "${supported}" "75;80;86;87;88;89;90;100;103;110;120;121")

# 2. CUDA 12: the portable default is untouched, SM72 included.
eshkol_resolve_cuda_architectures("${DEFAULT_LIST}" "${WORK_DIR}/nvcc12" "12.4" kept supported)
expect("CUDA 12 keeps the portable default" "${kept}" "${DEFAULT_LIST}")

# 3. No listing: the documented version range decides.
eshkol_resolve_cuda_architectures("${DEFAULT_LIST}" "${WORK_DIR}/nvcc_nolist" "13.0" kept supported)
expect("CUDA 13 by version drops SM72" "${kept}" "75;80;86;89;90")
expect("CUDA 13 by version records its range" "${supported}" "RANGE;75;999")
eshkol_resolve_cuda_architectures("${DEFAULT_LIST}" "${WORK_DIR}/nvcc_nolist" "12.4" kept supported)
expect("CUDA 12 by version keeps the default" "${kept}" "${DEFAULT_LIST}")
eshkol_resolve_cuda_architectures("${DEFAULT_LIST}" "" "11.4" kept supported)
expect("CUDA 11.4 (embedded path) keeps SM72, drops 89 and 90" "${kept}" "72;75;80;86")

# 4. Entry forms: suffixes are matched by their number, keywords pass through.
eshkol_cuda_filter_architectures("72-real;75-virtual;90a;native;all-major"
    "75;80;86;90" kept dropped)
expect("suffixed and keyword entries" "${kept}" "75-virtual;90a;native;all-major")
expect("suffixed entry dropped by number" "${dropped}" "72-real")

# 5. Nothing supported: the filter leaves nothing (the resolver then stops the
#    configure with a diagnostic, which script mode cannot catch).
eshkol_cuda_filter_architectures("60;61" "RANGE;75;999" kept dropped)
expect("no supported entry leaves nothing" "${kept}" "")

if(failures GREATER 0)
    message(FATAL_ERROR "cuda_architectures_test: ${failures} failure(s)")
endif()
message(STATUS "PASS: CUDA architecture policy")
