# ------------------------------------------------------------------------
#  Gunrock: Sub-Project k-Core Vertex Decomposition
# ------------------------------------------------------------------------
project(kcore)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_KCORE COMMAND kcore market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx)
set_tests_properties(TEST_KCORE PROPERTIES FAIL_REGULAR_EXPRESSION "[1-9]+[0-9]* errors occurred;FAIL;INCORRECT")
