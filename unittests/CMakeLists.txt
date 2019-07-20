
project(unittests)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

### tests using googletest and googlemock
add_test(NAME UNITTESTS COMMAND unittests)
target_link_libraries(unittests gunrock gtest_main)
set_tests_properties(UNITTESTS PROPERTIES PASS_REGULAR_EXPRESSION "PASSED")
