# ------------------------------------------------------------------------
#  Gunrock: Sub-Project HITS
# ------------------------------------------------------------------------
project(hits)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_HITS COMMAND hits market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx)
set_tests_properties(TEST_HITS PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
