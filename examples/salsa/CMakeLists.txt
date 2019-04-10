# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Stochastic Approach for Link-Structure Analysis
# ------------------------------------------------------------------------
project(salsa)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_SALSA COMMAND salsa market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx)
set_tests_properties(TEST_SALSA PROPERTIES PASS_REGULAR_EXPRESSION "Vertex ID: 38, Hub Rank: 0.09705")
