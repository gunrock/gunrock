# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Degree Centrality
# ------------------------------------------------------------------------
project(topk)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_TOPK COMMAND topk market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected)
set_tests_properties(TEST_TOPK PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
