# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Single Source Shortest Path
# ------------------------------------------------------------------------
project(sssp)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_SSSP COMMAND sssp market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --undirected --src=0)
set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "errors")

# Test the combination of --quick and --validation=each on the command line
# See Github Issue #773
add_test(NAME TEST_SSSP_QUICK_VALIDATION COMMAND sssp market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --validation=each --quick)
set_tests_properties(TEST_SSSP_QUICK_VALIDATION PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(TEST_SSSP_QUICK_VALIDATION PROPERTIES FAIL_REGULAR_EXPRESSION "errors")

if(GUNROCK_MGPU_TESTS)
  add_test(NAME TEST_MGPU_SSSP COMMAND sssp market
    ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx ${DEVICES} --undirected --src=0)
  set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
endif(GUNROCK_MGPU_TESTS)
