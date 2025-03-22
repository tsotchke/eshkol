# Add subdirectories
add_subdirectory(core)
add_subdirectory(frontend)

if(ESHKOL_BUILD_BACKEND)
    add_subdirectory(backend)
endif()

# Main executable
add_executable(eshkol main.c)
target_link_libraries(eshkol
    PRIVATE
        eshkol_core
        eshkol_frontend
)

if(ESHKOL_BUILD_BACKEND)
    target_link_libraries(eshkol
        PRIVATE
            eshkol_backend
    )
endif()

# Install
install(TARGETS eshkol
    RUNTIME DESTINATION bin
)
