# embed_metal_shader.cmake
# Reads a Metal shader header file and wraps it in an NSString literal
# for runtime compilation via [MTLDevice newLibraryWithSource:].
#
# Usage: cmake -DINPUT_FILE=<path> -DOUTPUT_FILE=<path> -P embed_metal_shader.cmake

if(NOT INPUT_FILE OR NOT OUTPUT_FILE)
    message(FATAL_ERROR "INPUT_FILE and OUTPUT_FILE must be specified")
endif()

file(READ "${INPUT_FILE}" SHADER_CONTENT)

# Write the NSString wrapper
# Use @R"METAL(...)METAL" raw string literal to avoid escaping issues
file(WRITE "${OUTPUT_FILE}"
    "// Auto-generated from ${INPUT_FILE} — do not edit directly\n"
    "// Modify lib/backend/gpu/metal_softfloat.h and rebuild\n"
    "static NSString* g_matmul_sf64_source = @R\"METAL(\n"
    "${SHADER_CONTENT}"
    ")METAL\";\n"
)
