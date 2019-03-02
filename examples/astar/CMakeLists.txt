# ------------------------------------------------------------------------
#  Gunrock: Sub-Project A* Search
# ------------------------------------------------------------------------
project(astar)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_ASTAR_DIRECTED COMMAND astar market
  ${gunrock_INCLUDE_DIRS}/dataset/small/map/cities.mtx
  --mapfile=${gunrock_INCLUDE_DIRS}/dataset/small/map/cities.map)
set_tests_properties(TEST_ASTAR_DIRECTED PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_ASTAR_UNDIRECTED COMMAND astar market
  ${gunrock_INCLUDE_DIRS}/dataset/small/map/cities.mtx
  --mapfile=${gunrock_INCLUDE_DIRS}/dataset/small/map/cities.map --undirected)
set_tests_properties(TEST_ASTAR_UNDIRECTED PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
