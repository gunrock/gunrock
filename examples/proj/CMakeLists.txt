# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Graph Projection
# ------------------------------------------------------------------------
project(proj)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_PROJ COMMAND proj market
	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx) 
set_tests_properties(TEST_PROJ PROPERTIES PASS_REGULAR_EXPRESSION "PASSED")
