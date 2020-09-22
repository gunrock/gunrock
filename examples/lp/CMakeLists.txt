# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Label Propagation Community Detection
# ------------------------------------------------------------------------
project(lp)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_LP COMMAND lp market
	${gunrock_INCLUDE_DIRS}/dataset/small/test_lp_1.mtx
     --test=1 --remove-self-loops=false)

# TODO: Add another test
#add_test(NAME TEST_LP COMMAND lp market
#	${gunrock_INCLUDE_DIRS}/dataset/small/lp_test_2.mtx
#     --test=2 --remove-self-loops=false)

#set_tests_properties(TEST_LP PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

