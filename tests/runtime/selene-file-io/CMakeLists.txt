add_library(selene_file_io_faults STATIC file_io_faults.c)
target_compile_features(selene_file_io_faults PRIVATE c_std_99)

add_test(NAME selene_file_io_aot
    COMMAND ${CMAKE_COMMAND}
        -DESHKOL_RUN=$<TARGET_FILE:eshkol-run>
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/file_io_test.esk
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/file_io_aot
        -DLIBDIR=${PROJECT_BINARY_DIR}
        -DFIXTURE_LIBDIR=$<TARGET_FILE_DIR:selene_file_io_faults>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_aot.cmake)
set_tests_properties(selene_file_io_aot PROPERTIES
    TIMEOUT 0
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ENVIRONMENT "ESHKOL_PATH=${PROJECT_SOURCE_DIR}/lib;ESHKOL_JIT_CACHE=0"
    FAIL_REGULAR_EXPRESSION "FAIL:|fatal signal|LLVM module verification failed")
