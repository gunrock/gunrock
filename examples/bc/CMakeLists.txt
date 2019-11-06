# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Betweenness Centrality
# ------------------------------------------------------------------------
project(bc)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_BC COMMAND bc market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
