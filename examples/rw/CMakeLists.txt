# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Random Walk
# ------------------------------------------------------------------------
project(rw)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_RW COMMAND rw market
   ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --walk_length=10 --undirected)
set_tests_properties(TEST_RW PROPERTIES PASS_REGULAR_EXPRESSION "==> random walk output paths:")
