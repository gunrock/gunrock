# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Breadth First Search
# ------------------------------------------------------------------------
project(grmat)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_GRMAT COMMAND grmat rmat)
set_tests_properties(TEST_GRMAT PROPERTIES FAIL_REGULAR_EXPRESSION "Error")
