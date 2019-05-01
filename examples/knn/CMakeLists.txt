# ------------------------------------------------------------------------
#  Gunrock: Sub-Project k-Nearest Neighbor (& Shared Nearest Neighbor)
# ------------------------------------------------------------------------
project(knn)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_KNN COMMAND knn market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --k=5)
set_tests_properties(TEST_KNN PROPERTIES PASS_REGULAR_EXPRESSION "PASSED KNN")
