# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Connected Component
# ------------------------------------------------------------------------
project(cc)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_CC COMMAND cc market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx)
set_tests_properties(TEST_CC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

if(GUNROCK_MGPU_TESTS)
  add_test(NAME TEST_MGPU_CC COMMAND cc market
    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES})
  set_tests_properties(TEST_CC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
endif(GUNROCK_MGPU_TESTS)
