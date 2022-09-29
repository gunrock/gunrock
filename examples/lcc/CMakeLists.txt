# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Triangle Counting
# ------------------------------------------------------------------------
project(lcc)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_LCC COMMAND lcc --undirected=true market
	${gunrock_INCLUDE_DIRS}/dataset/small/test_lcc.mtx --sort-csr) 
set_tests_properties(TEST_LCC PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
