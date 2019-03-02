# ------------------------------------------------------------------------
#  Gunrock: Sub-Project PageRank
# ------------------------------------------------------------------------
project(pr)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_PR COMMAND pr market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --normalized --compensate --undirected)
set_tests_properties(TEST_PR PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

if(GUNROCK_MGPU_TESTS)
  add_test(NAME TEST_MGPU_PR COMMAND pr market
    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES} 
    --normalized --compensate --undirected)
  set_tests_properties(TEST_MGPU_PR PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
endif(GUNROCK_MGPU_TESTS)
