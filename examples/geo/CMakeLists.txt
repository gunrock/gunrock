# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Geolocation
# ------------------------------------------------------------------------
project(geo)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_GEO COMMAND geo  market
	${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx 
	--labels-file=${gunrock_INCLUDE_DIRS}/examples/geo/_data/samples/sample.labels)
set_tests_properties(TEST_GEO PROPERTIES PASS_REGULAR_EXPRESSION "0 errors occurred")
