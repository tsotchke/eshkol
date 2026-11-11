# Literal scopes are also read by scripts/check_stdlib_source_identity.py.
# Preserve their simple syntax; changing the collector requires checker review.
function(eshkol_collect_stdlib_identity_inputs SOURCE_ROOT OUT_VAR)
  set(_stdlib_identity_patterns
    "inc/eshkol/*"
    "lib/backend/*"
    "lib/core/*"
    "lib/repl/*"
    "cmake/*"
    "lib/math/*.esk"
    "lib/signal/*.esk"
    "lib/random/*.esk"
    "lib/web/*.esk"
    "lib/tensor/*.esk"
    "lib/quantum/*.esk"
    "lib/ml/*.esk")
  set(_stdlib_identity_files
    "CMakeLists.txt"
    "exe/eshkol-run.cpp"
    "lib/stdlib.esk"
    "lib/math.esk")
  set(_inputs "")
  foreach(_pattern IN LISTS _stdlib_identity_patterns)
    file(GLOB_RECURSE _matches CONFIGURE_DEPENDS "${SOURCE_ROOT}/${_pattern}")
    list(APPEND _inputs ${_matches})
  endforeach()
  foreach(_file IN LISTS _stdlib_identity_files)
    list(APPEND _inputs "${SOURCE_ROOT}/${_file}")
  endforeach()
  list(REMOVE_DUPLICATES _inputs)
  list(SORT _inputs)
  set(${OUT_VAR} "${_inputs}" PARENT_SCOPE)
endfunction()
