# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Breadth First Search
# ------------------------------------------------------------------------
project(bfs)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_BFS COMMAND bfs market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BFS PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

if(GUNROCK_MGPU_TESTS)
  add_test(NAME TEST_MGPU_BFS COMMAND bfs market
    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES} --undirected --src=0)
  set_tests_properties(TEST_BFS PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
endif(GUNROCK_MGPU_TESTS)
