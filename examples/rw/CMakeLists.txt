# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Random Walk
# ------------------------------------------------------------------------
project(rw)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

# TODO: Add the correct expression output in rw's app.
# add_test(NAME TEST_RW COMMAND rw market
#   ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected --walk_length=5)
# set_tests_properties(TEST_RW PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
