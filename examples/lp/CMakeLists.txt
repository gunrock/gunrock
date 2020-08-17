# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Label Propagation Community Detection
# ------------------------------------------------------------------------
project(lp)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

#add_test(NAME TEST_LP COMMAND lp market
#	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx 
#	--omp-threads=32 --advance-mode=ALL_EDGES --unify-segments=true)
#set_tests_properties(TEST_LP PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

