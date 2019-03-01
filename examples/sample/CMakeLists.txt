# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Sample Project (A template for develop new graph
#  primitive)
# ------------------------------------------------------------------------
project(sample)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

# add_test(NAME TEST_SAMPLE COMMAND sample market
#    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx)
# set_tests_properties(TEST_SAMPLE PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
