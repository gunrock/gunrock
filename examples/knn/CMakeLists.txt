# ------------------------------------------------------------------------
#  Gunrock: Sub-Project k-Nearest Neighbor
# ------------------------------------------------------------------------
project(knn)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_KNN COMMAND knn market --labels-file
  ${gunrock_INCLUDE_DIRS}/dataset/small/stars_2total_separate --k=2)
set_tests_properties(TEST_KNN PROPERTIES PASS_REGULAR_EXPRESSION "PASSED KNN")
set_tests_properties(TEST_KNN PROPERTIES FAIL_REGULAR_EXPRESSION "[1-9]+[0-9]* errors occurred;FAIL;INCORRECT")

add_test(NAME TEST_KNN_MULTIPLE_RUNS COMMAND knn market --labels-file
  ${gunrock_INCLUDE_DIRS}/dataset/small/stars_2total_separate --k=3 --quick --num-runs=10)
set_tests_properties(TEST_KNN_MULTIPLE_RUNS PROPERTIES FAIL_REGULAR_EXPRESSION "[1-9]+[0-9]* errors occurred;FAIL;INCORRECT")
