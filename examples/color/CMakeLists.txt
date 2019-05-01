# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Graph Coloring
# ------------------------------------------------------------------------
project(color)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_COLOR COMMAND color market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --undirected)
set_tests_properties(TEST_COLOR PROPERTIES PASS_REGULAR_EXPRESSION "0 errors occurred.")
