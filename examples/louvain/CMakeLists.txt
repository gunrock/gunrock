# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Louvain Community Detection
# ------------------------------------------------------------------------
project(louvain)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_LOUVAIN COMMAND louvain market
	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx 
	--omp-threads=32 --advance-mode=ALL_EDGES --unify-segments=true)
set_tests_properties(TEST_LOUVAIN PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

