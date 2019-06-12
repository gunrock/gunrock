# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Triangle Counting
# ------------------------------------------------------------------------
project(tc)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_TC COMMAND tc market
	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --sort-csr) 
set_tests_properties(TEST_TC PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
