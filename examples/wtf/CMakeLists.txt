# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Who To Follow
# ------------------------------------------------------------------------
project(wtf)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

add_test(NAME TEST_WTF COMMAND wtf market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx)
set_tests_properties(TEST_WTF PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")
