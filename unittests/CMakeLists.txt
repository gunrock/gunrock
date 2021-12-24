# begin /* Add unit tests */
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
  get_target_property(ESSENTIALS_ARCHITECTURES 
    essentials CUDA_ARCHITECTURES
  )
  set_target_properties(${TEST_NAME} 
    PROPERTIES 
      CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
  )
  add_test(NAME test_${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
# end /* Add unit tests */