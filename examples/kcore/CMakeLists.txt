# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Graph K-Core
# ------------------------------------------------------------------------
project(kcore)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_KCORE COMMAND kcore market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --undirected)
set_tests_properties(TEST_KCORE PROPERTIES PASS_REGULAR_EXPRESSION "0 errors occurred.")
