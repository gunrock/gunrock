# ------------------------------------------------------------------------
#  Gunrock: Sub-Project C Shared Libraries
# ------------------------------------------------------------------------
add_executable(shared_lib_pr shared_lib_pr.c)
set_target_properties(shared_lib_pr PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(shared_lib_pr gunrock)

add_executable(shared_lib_sm shared_lib_sm.c)
set_target_properties(shared_lib_sm PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(shared_lib_sm gunrock)

CUDA_ADD_EXECUTABLE(shared_lib_hits shared_lib_hits.cu OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
target_link_libraries(shared_lib_hits gunrock)



add_test(NAME SHARED_LIB_TEST_PAGERANK COMMAND shared_lib_pr)
set_tests_properties(SHARED_LIB_TEST_PAGERANK
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Score.*1.2*")

add_test(NAME SHARED_LIB_TEST_HITS COMMAND shared_lib_hits)
set_tests_properties(SHARED_LIB_TEST_HITS
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID: \\[.\\], Hub Score: \\[.\.......\\], Auth Score: \\[.\.......\\]")
