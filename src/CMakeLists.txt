# Add subdirectories
add_subdirectory(core)
add_subdirectory(frontend)
add_subdirectory(backend)

# Main executable
add_executable(eshkol main.c)
target_link_libraries(eshkol
    PRIVATE
        eshkol_core
        eshkol_frontend
        eshkol_backend
)

# Install
install(TARGETS eshkol
    RUNTIME DESTINATION bin
)
