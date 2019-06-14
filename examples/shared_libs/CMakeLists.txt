# ------------------------------------------------------------------------
#  Gunrock: Sub-Project C Shared Libraries
# ------------------------------------------------------------------------
add_executable(shared_lib_pr shared_lib_pr.c)
target_link_libraries(shared_lib_pr gunrock)

add_test(NAME SHARED_LIB_TEST_PAGERANK COMMAND shared_lib_pr)
set_tests_properties(SHARED_LIB_TEST_PAGERANK
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Score.*1.2*")
