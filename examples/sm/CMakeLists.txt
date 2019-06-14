# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Sub-Graph Matching
# ------------------------------------------------------------------------
project(sm)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_SM COMMAND sm market
	${gunrock_INCLUDE_DIRS}/dataset/small/tree.mtx --pattern-graph-type=market
	--pattern-graph-file=${gunrock_INCLUDE_DIRS}/dataset/small/query_sm.mtx 
	--undirected=1 --pattern-undirected=1 --num-runs=1 --quiet=false) 
set_tests_properties(TEST_SM PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
