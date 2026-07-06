# ─────────────────────────────────────────────────────────────────────
# EshkolCompile.cmake  (ESH-0215)
#
# Canonical CMake integration for compiling .esk sources into a CMake
# library/executable target with REAL incremental-build dependency
# tracking: editing a (load …)ed / (import …)ed / (require …)d Eshkol
# source now reliably triggers a recompile of the object that transitively
# reaches it, in a single ninja invocation — no more "rm the object and
# run ninja 2-3 times" (Noesis BUGS-2026-07-04 #3/#5, Selene PRIORITY 2).
#
# Provides:
#   eshkol_compile_library(NAME
#     SOURCES        <.esk files…>
#     [INCLUDE_DIRS   <dirs…>]
#     [LINK_LIBRARIES <targets…>]
#     [DEFINES        <name=value…>]
#     [DEPENDS        <extra files / targets…>])
#
#   eshkol_compile_executable(NAME
#     SOURCES        <.esk files…>
#     [INCLUDE_DIRS   <dirs…>]
#     [LINK_LIBRARIES <targets…>]
#     [DEFINES        <name=value…>]
#     [DEPENDS        <extra files / targets…>])
#
# Each .esk source is compiled to a .o by invoking eshkol-run with
# `--emit-object -o <obj>`, then linked into a SHARED (library) or
# EXECUTABLE target via the C++ compiler.  The older
# `--compile-only --output <stem>` compatibility path remains available
# through ESHKOL_OBJECT_MODE=COMPILE_ONLY for older Eshkol binaries that
# predate --emit-object.
#
# --- Dependency tracking (ESH-0215) ---------------------------------
# eshkol-run supports `--emit-depfile <path>`, which walks the SAME
# (load …)/(import …)/(require …) graph the compiler itself follows and
# writes a Makefile-format depfile:
#
#   out.o: main.esk dep1.esk dep2.esk ...
#
# This module always passes `--emit-depfile <obj>.d` and attaches it to
# the custom command via CMake's DEPFILE argument to add_custom_command,
# so Ninja (and, on CMake >= 3.20, the Makefiles generator) re-stats every
# transitively-loaded source on each build and reruns the compile the
# moment any one of them changes — not just the entry file. Generators
# without DEPFILE support (Xcode, Visual Studio) fall back to depending on
# the entry file alone, matching the previous behavior; multi-config /
# IDE generators are out of scope for this ticket.
#
# Eshkol_COMPILER must be defined (by find_package(Eshkol), a vendored
# build directory that exports an `eshkol-run` target, or by the caller).
# ─────────────────────────────────────────────────────────────────────

if(POLICY CMP0116)
  # Ninja DEPFILE paths are resolved relative to CMAKE_CURRENT_BINARY_DIR
  # under the NEW behavior; this module always writes absolute depfile
  # paths, so NEW vs. OLD is a no-op here except for silencing the CMake
  # developer warning that fires whenever DEPFILE is used with policy
  # CMP0116 unset (introduced 3.20; harmless when the active generator
  # predates DEPFILE support entirely).
  cmake_policy(SET CMP0116 NEW)
endif()

if(NOT Eshkol_COMPILER)
  if(TARGET eshkol::eshkol-run)
    get_target_property(Eshkol_COMPILER eshkol::eshkol-run LOCATION)
  elseif(TARGET eshkol-run)
    set(Eshkol_COMPILER "$<TARGET_FILE:eshkol-run>")
  endif()
endif()

if(NOT Eshkol_COMPILER)
  message(FATAL_ERROR
    "EshkolCompile: Eshkol_COMPILER not set. "
    "Either find_package(Eshkol) must succeed, or the vendored Eshkol "
    "build must export an `eshkol-run` target, or set Eshkol_COMPILER "
    "to the eshkol-run binary explicitly.")
endif()

if(NOT DEFINED ESHKOL_OBJECT_MODE)
  set(ESHKOL_OBJECT_MODE "AUTO" CACHE STRING
    "Eshkol object emission mode: AUTO, EMIT_OBJECT, or COMPILE_ONLY")
endif()

string(TOUPPER "${ESHKOL_OBJECT_MODE}" _eshkol_object_mode)
if(NOT _eshkol_object_mode MATCHES "^(AUTO|EMIT_OBJECT|COMPILE_ONLY)$")
  message(FATAL_ERROR
    "ESHKOL_OBJECT_MODE must be AUTO, EMIT_OBJECT, or COMPILE_ONLY "
    "(got: ${ESHKOL_OBJECT_MODE})")
endif()

set(ESHKOL_EFFECTIVE_OBJECT_MODE "${_eshkol_object_mode}")
if(_eshkol_object_mode STREQUAL "AUTO")
  if(Eshkol_COMPILER MATCHES "^\\$<")
    # Vendored compiler targets are not executable at configure time — the
    # pinned Eshkol contract is the direct object path, so use it by default.
    set(ESHKOL_EFFECTIVE_OBJECT_MODE "EMIT_OBJECT")
  else()
    set(_eshkol_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/eshkol-object-mode")
    set(_eshkol_probe_src "${_eshkol_probe_dir}/probe.esk")
    set(_eshkol_probe_obj "${_eshkol_probe_dir}/probe.o")
    file(MAKE_DIRECTORY "${_eshkol_probe_dir}")
    file(WRITE "${_eshkol_probe_src}" "(define (main) 0)\n")
    execute_process(
      COMMAND "${Eshkol_COMPILER}"
              --emit-object -o "${_eshkol_probe_obj}"
              --shared-lib -fPIC
              "${_eshkol_probe_src}"
      RESULT_VARIABLE _eshkol_probe_result
      OUTPUT_VARIABLE _eshkol_probe_stdout
      ERROR_VARIABLE  _eshkol_probe_stderr
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      TIMEOUT 15)
    if(_eshkol_probe_result EQUAL 0 AND EXISTS "${_eshkol_probe_obj}")
      set(ESHKOL_EFFECTIVE_OBJECT_MODE "EMIT_OBJECT")
    else()
      set(ESHKOL_EFFECTIVE_OBJECT_MODE "COMPILE_ONLY")
      message(WARNING
        "Eshkol --emit-object probe failed; falling back to "
        "ESHKOL_OBJECT_MODE=COMPILE_ONLY. Probe output: "
        "${_eshkol_probe_stdout} ${_eshkol_probe_stderr}")
    endif()
  endif()
endif()

message(STATUS
  "Eshkol object mode: ${ESHKOL_EFFECTIVE_OBJECT_MODE} "
  "(requested ${ESHKOL_OBJECT_MODE})")

# Does the active generator honor add_custom_command(... DEPFILE ...)?
# Ninja has supported it since CMake 3.7; the Makefiles family gained
# support in CMake 3.20. Multi-config/IDE generators (Xcode, Visual
# Studio) do not support it at all as of CMake 4.x.
set(_eshkol_depfile_supported FALSE)
if(CMAKE_GENERATOR MATCHES "Ninja")
  set(_eshkol_depfile_supported TRUE)
elseif(CMAKE_GENERATOR MATCHES "Makefiles" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
  set(_eshkol_depfile_supported TRUE)
endif()

# ── Internal helper — compile a single .esk → .o ──────────────────
function(_eshkol_compile_one ESK_FILE OBJ_FILE OUT_DEPS_VAR
                             INCLUDE_DIRS DEFINES DEPENDS_LIST SHARED)
  get_filename_component(_abs "${ESK_FILE}" ABSOLUTE)
  get_filename_component(_dir "${OBJ_FILE}" DIRECTORY)

  set(_eshkol_path_entries
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src"
    ${INCLUDE_DIRS})

  if(NOT Eshkol_COMPILER MATCHES "^\\$<")
    get_filename_component(_eshkol_compiler_dir "${Eshkol_COMPILER}" DIRECTORY)
    get_filename_component(_eshkol_root "${_eshkol_compiler_dir}/.." ABSOLUTE)
    list(APPEND _eshkol_path_entries
      "${_eshkol_root}"
      "${_eshkol_root}/lib")
  endif()

  set(_include_flag_entries)
  list(APPEND _include_flag_entries "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/src")
  foreach(_inc IN LISTS INCLUDE_DIRS)
    if(NOT IS_ABSOLUTE "${_inc}")
      get_filename_component(_inc_abs "${_inc}" ABSOLUTE)
      list(APPEND _eshkol_path_entries "${_inc_abs}")
      list(APPEND _include_flag_entries "${_inc_abs}")
    else()
      list(APPEND _include_flag_entries "${_inc}")
    endif()
  endforeach()
  if(DEFINED ENV{ESHKOL_PATH} AND NOT "$ENV{ESHKOL_PATH}" STREQUAL "")
    list(APPEND _eshkol_path_entries "$ENV{ESHKOL_PATH}")
  endif()

  list(REMOVE_DUPLICATES _eshkol_path_entries)
  if(WIN32)
    string(JOIN "\;" _eshkol_path ${_eshkol_path_entries})
  else()
    string(JOIN ":" _eshkol_path ${_eshkol_path_entries})
  endif()

  set(_flags)
  set(_byproducts)
  if(ESHKOL_EFFECTIVE_OBJECT_MODE STREQUAL "EMIT_OBJECT")
    set(_flags --emit-object -o "${OBJ_FILE}")
    if(SHARED)
      list(APPEND _flags --shared-lib -fPIC)
    endif()
    list(REMOVE_DUPLICATES _include_flag_entries)
    foreach(_inc IN LISTS _include_flag_entries)
      list(APPEND _flags -I "${_inc}")
    endforeach()
    foreach(_define IN LISTS DEFINES)
      list(APPEND _flags -D "${_define}")
    endforeach()
    list(APPEND _byproducts "${OBJ_FILE}.bc")
  else()
    if(DEFINES)
      message(FATAL_ERROR
        "EshkolCompile: DEFINES were requested for ${ESK_FILE}, but "
        "ESHKOL_OBJECT_MODE=COMPILE_ONLY uses the older compatibility "
        "path that does not pass -D/--define.")
    endif()
    set(_obj_stem "${OBJ_FILE}")
    if(_obj_stem MATCHES "\\.o$")
      string(REGEX REPLACE "\\.o$" "" _obj_stem "${_obj_stem}")
    endif()
    set(_flags --compile-only --output "${_obj_stem}")
    if(SHARED)
      list(APPEND _flags --shared-lib)
    endif()
    list(APPEND _byproducts "${_obj_stem}.bc")
  endif()

  # ESH-0215: always ask eshkol-run for a depfile — it walks the same
  # load/import/require graph the compile itself follows, so it is exactly
  # the prerequisite list this build edge needs.  --emit-depfile is
  # supported regardless of EMIT_OBJECT vs. COMPILE_ONLY (both paths route
  # through eshkol-run's shared object-emission code).
  set(_depfile "${OBJ_FILE}.d")
  list(APPEND _flags --emit-depfile "${_depfile}")
  list(APPEND _byproducts "${_depfile}")

  set(_depfile_args)
  if(_eshkol_depfile_supported)
    set(_depfile_args DEPFILE "${_depfile}")
  endif()

  add_custom_command(
    OUTPUT  "${OBJ_FILE}"
    BYPRODUCTS ${_byproducts}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_dir}"
    COMMAND "${CMAKE_COMMAND}" -E env "ESHKOL_PATH=${_eshkol_path}"
            "${Eshkol_COMPILER}" ${_flags} "${_abs}"
    DEPENDS "${_abs}" ${DEPENDS_LIST}
    ${_depfile_args}
    COMMENT "Eshkol → ${OBJ_FILE}"
    VERBATIM)

  set(${OUT_DEPS_VAR} "${OBJ_FILE}" PARENT_SCOPE)
endfunction()

# ── Internal — common machinery for both libraries + executables ──
function(_eshkol_compile_target NAME KIND)
  cmake_parse_arguments(ARG
    ""
    ""
    "SOURCES;INCLUDE_DIRS;LINK_LIBRARIES;DEFINES;DEPENDS"
    ${ARGN})

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "${NAME}: SOURCES is required")
  endif()

  set(_objects)
  set(_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${NAME}.esk.dir")
  set(_shared FALSE)
  if(KIND STREQUAL "LIBRARY")
    set(_shared TRUE)
  endif()

  foreach(_src IN LISTS ARG_SOURCES)
    get_filename_component(_abs "${_src}" ABSOLUTE)
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_abs}")
    string(REPLACE ".." "_up" _rel "${_rel}")
    string(REPLACE "/"  "__" _rel "${_rel}")
    set(_obj "${_obj_dir}/${_rel}.o")

    _eshkol_compile_one("${_abs}" "${_obj}" _obj_out
      "${ARG_INCLUDE_DIRS}" "${ARG_DEFINES}" "${ARG_DEPENDS}" ${_shared})
    list(APPEND _objects "${_obj_out}")
  endforeach()

  if(KIND STREQUAL "LIBRARY")
    add_library(${NAME} SHARED ${_objects})
  else()
    add_executable(${NAME} ${_objects})
  endif()

  # Tell CMake the .o files come from an external producer
  set_source_files_properties(${_objects} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED       TRUE)

  # The runtime is C++ (and Objective-C++ on Apple platforms), so link
  # even a target whose direct inputs are externally generated .o files
  # with the C++ driver.
  set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX)

  if(ARG_LINK_LIBRARIES)
    target_link_libraries(${NAME} PRIVATE ${ARG_LINK_LIBRARIES})
  endif()

  # Always link against the Eshkol runtime, if a target for it exists.
  if(TARGET Eshkol::eshkol)
    target_link_libraries(${NAME} PRIVATE Eshkol::eshkol)
  elseif(TARGET eshkol::eshkol)
    target_link_libraries(${NAME} PRIVATE eshkol::eshkol)
  endif()
endfunction()

# ── Public API ─────────────────────────────────────────────────────
function(eshkol_compile_library NAME)
  _eshkol_compile_target(${NAME} LIBRARY ${ARGN})
endfunction()

function(eshkol_compile_executable NAME)
  _eshkol_compile_target(${NAME} EXECUTABLE ${ARGN})
endfunction()
