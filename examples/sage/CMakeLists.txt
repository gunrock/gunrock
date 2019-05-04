# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Graph Sage
# ------------------------------------------------------------------------
project(sage)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_SAGE COMMAND sage market
	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx) 
set_tests_properties(TEST_SAGE PROPERTIES PASS_REGULAR_EXPRESSION "Embedding validation: PASS")
