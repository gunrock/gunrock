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
    get_target_property(ESSENTIALS_ARCHITECTURES 
      essentials HIP_ARCHITECTURES
    )
    set_target_properties(${TEST_NAME} 
      PROPERTIES 
        HIP_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
    )
    set_source_files_properties(${SOURCE}
      PROPERTIES
        LANGUAGE HIP
    )
  endif()
  
  # Use gtest_discover_tests to register individual tests instead of add_test
  gtest_discover_tests(${TEST_NAME})
endforeach()
# end /* Add unit tests */