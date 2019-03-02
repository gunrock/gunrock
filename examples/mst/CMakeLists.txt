# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Minimum Spanning Tree
# ------------------------------------------------------------------------
project(mst)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_MST COMMAND mst market
  ${gunrock_INCLUDE_DIRS}/dataset/small/test_mst.mtx)
set_tests_properties(TEST_MST PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
