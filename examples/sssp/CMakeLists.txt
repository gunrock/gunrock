# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Single Source Shortest Path
# ------------------------------------------------------------------------
project(sssp)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_SSSP COMMAND sssp market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --undirected --src=0)
set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

if(GUNROCK_MGPU_TESTS)
  add_test(NAME TEST_MGPU_SSSP COMMAND sssp market
    ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx ${DEVICES} --undirected --src=0)
  set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
endif(GUNROCK_MGPU_TESTS)
