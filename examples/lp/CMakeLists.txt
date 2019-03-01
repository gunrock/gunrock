# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Label Propagation
# ------------------------------------------------------------------------
project(lp)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_LP COMMAND lp market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx)
set_tests_properties(TEST_LP PROPERTIES PASS_REGULAR_EXPRESSION "Component Count: 14")
