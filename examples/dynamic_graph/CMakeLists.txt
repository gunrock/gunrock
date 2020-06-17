# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Dynamic Graph Examples
# ------------------------------------------------------------------------
project(dynamic_graph)
message("-- Project Added: ${PROJECT_NAME}")

add_test(NAME TEST_DYNAMIC_GRAPH_BULK_BUILD COMMAND dynamic_graph market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected)
set_tests_properties(TEST_DYNAMIC_GRAPH_BULK_BUILD PROPERTIES PASS_REGULAR_EXPRESSION "PASSED")


include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)