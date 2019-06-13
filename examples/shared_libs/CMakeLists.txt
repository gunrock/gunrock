# gunrock test rig cmake file
# include_directories(${gunrock_INCLUDE_DIRS}/gunrock)

# add_executable(shared_lib_bfs shared_lib_bfs.c)
# target_link_libraries(shared_lib_bfs gunrock)

# add_executable(shared_lib_bc shared_lib_bc.c)
# target_link_libraries(shared_lib_bc gunrock)

# add_executable(shared_lib_cc shared_lib_cc.c)
# target_link_libraries(shared_lib_cc gunrock)

add_executable(shared_lib_pr shared_lib_pr.c)
target_link_libraries(shared_lib_pr gunrock)

# add_executable(shared_lib_sssp shared_lib_sssp.c)
# target_link_libraries(shared_lib_sssp gunrock)

# add_executable(shared_lib_example simple_example.c)
# target_link_libraries(shared_lib_example gunrock)

### shared library application interface tests
# add_test(NAME SHARED_LIB_TEST_BFS COMMAND shared_lib_bfs)
# set_tests_properties(SHARED_LIB_TEST_BFS
#   PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Label.*1")

# add_test(NAME SHARED_LIB_TEST_BC COMMAND shared_lib_bc)
# set_tests_properties(SHARED_LIB_TEST_BC
#   PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*0.*: Score.*0.5000")

# add_test(NAME SHARED_LIB_TEST_CC COMMAND shared_lib_cc)
# set_tests_properties(SHARED_LIB_TEST_CC
#   PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*1.*: Component.*0")

# add_test(NAME SHARED_LIB_TEST_SSSP COMMAND shared_lib_sssp)
# set_tests_properties(SHARED_LIB_TEST_SSSP
#   PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*1.*: Label.*39.*")

add_test(NAME SHARED_LIB_TEST_PAGERANK COMMAND shared_lib_pr)
set_tests_properties(SHARED_LIB_TEST_PAGERANK
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Score.*1.2*")
