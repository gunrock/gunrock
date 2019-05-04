# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Vertex Nomination
# ------------------------------------------------------------------------
project(vn)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

# add_test(NAME TEST_VN COMMAND vn market
#	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx) 
# set_tests_properties(TEST_VN PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
