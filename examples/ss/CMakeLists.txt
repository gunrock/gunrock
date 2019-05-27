# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Scan Statistics
# ------------------------------------------------------------------------
project(ss)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_SS COMMAND ss market
	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx) 
set_tests_properties(TEST_SS PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
