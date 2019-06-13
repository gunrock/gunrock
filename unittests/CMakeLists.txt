
project(unittests)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

### tests using googletest and googlemock
add_test(NAME UNITTESTS COMMAND unittests)
# add_executable(unit test.cpp)
target_link_libraries(unittests gunrock gtest gmock)
