# begin /* Add unit tests */
# Include GoogleTest module for test discovery
include(GoogleTest)

set(UNITTEST_SOURCES
  unittests.cu
)

foreach(SOURCE IN LISTS UNITTEST_SOURCES)
  get_filename_component(TEST_NAME ${SOURCE} NAME_WLE)
  add_executable(${TEST_NAME} ${SOURCE})
  target_link_libraries(${TEST_NAME} 
    PRIVATE essentials
    PRIVATE gtest::main
  )
  
  if(${ESSENTIALS_NVIDIA_BACKEND})
    get_target_property(ESSENTIALS_ARCHITECTURES 
      essentials CUDA_ARCHITECTURES
    )
    set_target_properties(${TEST_NAME} 
      PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
    )
  else()
    # HIP_ARCHITECTURES is not a valid target property for executables
    # Architecture is set globally via CMAKE_HIP_ARCHITECTURES
    set_source_files_properties(${SOURCE}
      PROPERTIES
        LANGUAGE HIP
    )
    # Set rpath for AMD builds to find ROCm libraries at runtime
    if(DEFINED ROCM_PATH)
      set_target_properties(${TEST_NAME}
        PROPERTIES
          INSTALL_RPATH "${ROCM_PATH}/lib"
          BUILD_WITH_INSTALL_RPATH TRUE
      )
    endif()
  endif()
  
  # Use gtest_discover_tests to register individual tests instead of add_test
  # Note: gtest_discover_tests runs the executable during CMake configuration
  # For AMD builds, LD_LIBRARY_PATH must be set before running CMake, not here
  # The ENVIRONMENT property here is for test execution, not discovery
  gtest_discover_tests(${TEST_NAME})
endforeach()
# end /* Add unit tests */