# Gunrock's Testing CMake File

# begin /* Add premitives' subdirectories */
if(GUNROCK_BUILD_APPLICATIONS)
  add_subdirectory(shared_libs)
  #add_subdirectory(simple_example)
  add_subdirectory(bc)
  add_subdirectory(bfs)
  add_subdirectory(cc)
  add_subdirectory(pr)
  add_subdirectory(sssp)
  add_subdirectory(hits)
  add_subdirectory(salsa)
  add_subdirectory(mst)
  add_subdirectory(wtf)
  add_subdirectory(topk)
  add_subdirectory(rw)
  #add_subdirectory(mf)
  #add_subdirectory(template)
  #add_subdirectory(vis)
  #add_subdirectory(mis)

# Individual options to build specific applications
else(GUNROCK_BUILD_APPLICATIONS)
  if(GUNROCK_APP_BC)
    add_subdirectory(bc)
  endif(GUNROCK_APP_BC)

  if(GUNROCK_APP_BFS)
    add_subdirectory(bfs)
  endif(GUNROCK_APP_BFS)

  if(GUNROCK_APP_CC)
    add_subdirectory(cc)
  endif(GUNROCK_APP_CC)

  if(GUNROCK_APP_PR)
    add_subdirectory(pr)
  endif(GUNROCK_APP_PR)

  if(GUNROCK_APP_SSSP)
    add_subdirectory(sssp)
  endif(GUNROCK_APP_SSSP)

  if(GUNROCK_APP_HITS)
    add_subdirectory(hits)
  endif(GUNROCK_APP_HITS)

  if(GUNROCK_APP_SALSA)
    add_subdirectory(salsa)
  endif(GUNROCK_APP_SALSA)

  if(GUNROCK_APP_MST)
    add_subdirectory(mst)
  endif(GUNROCK_APP_MST)

  if(GUNROCK_APP_WTF)
    add_subdirectory(wtf)
  endif(GUNROCK_APP_WTF)

  if(GUNROCK_APP_TOPK)
    add_subdirectory(topk)
  endif(GUNROCK_APP_TOPK)

  if(GUNROCK_APP_RW)
    add_subdirectory(rw)
  endif(GUNROCK_APP_RW)

  # if(GUNROCK_APP_MF)
  #  add_subdirectory(mf)
  # endif(GUNROCK_APP_MF)

  # if(GUNROCK_APP_SAMPLE)
  #   add_subdirectory(sample)
  # endif(GUNROCK_APP_SAMPLE)

endif(GUNROCK_BUILD_APPLICATIONS)
# end /* Add premitives' subdirectories */

# begin /* Enable Testing for `ctest` */
enable_testing()

### primitive tests with bips98_606 graph
add_test(NAME TEST_BFS COMMAND bfs market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BFS PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_BC COMMAND bc market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_CC COMMAND cc market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx)
set_tests_properties(TEST_CC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_SSSP COMMAND sssp market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --undirected --src=0)
set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_PAGERANK COMMAND pr market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --normalized --compensate --undirected)
set_tests_properties(TEST_PAGERANK PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_TOPK COMMAND topk market
  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx --undirected)
set_tests_properties(TEST_TOPK PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

### shared library application interface tests
add_test(NAME SHARED_LIB_TEST_BFS COMMAND shared_lib_bfs)
set_tests_properties(SHARED_LIB_TEST_BFS
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Label.*1")

add_test(NAME SHARED_LIB_TEST_BC COMMAND shared_lib_bc)
set_tests_properties(SHARED_LIB_TEST_BC
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*0.*: Score.*0.5000")

add_test(NAME SHARED_LIB_TEST_CC COMMAND shared_lib_cc)
set_tests_properties(SHARED_LIB_TEST_CC
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*1.*: Component.*0")

add_test(NAME SHARED_LIB_TEST_SSSP COMMAND shared_lib_sssp)
set_tests_properties(SHARED_LIB_TEST_SSSP
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*1.*: Label.*39.*")

add_test(NAME SHARED_LIB_TEST_PAGERANK COMMAND shared_lib_pr)
set_tests_properties(SHARED_LIB_TEST_PAGERANK
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Score.*1.2*")

# add_test(NAME SHARED_LIB_TEST_MF COMMAND shared_lib_mf)
# set_tests_properties(SHARED_LIB_TEST_MF
#  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*1.*: Label.*39.*")

# note: Some premitives are not added as test because they don't have
#	cpu reference code.

# add_test(NAME SIMPLE_EXAMPLE COMMAND simple_example market
#    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx)
# set_tests_properties(SIMPLE_EXAMPLE PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

if(GUNROCK_MGPU_TESTS)
  if(DEFINED DEVICES)
 	   set(DEVICES "--device=${DEVICES}")
  else(DEFINED DEVICES)
	   message(WARNING "GUNROCK_MGPU_TESTS was set ON, but devices were not specified. Using `-DDEVICES=0,0` (default).")
  	 set(DEVICES "--device=0,0")
  endif(DEFINED DEVICES)

  add_test(NAME TEST_MGPU_BFS COMMAND bfs market
    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES} --undirected --src=0)
  set_tests_properties(TEST_BFS PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  # add_test(NAME TEST_MGPU_BC COMMAND bc market
  #  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES} --undirected --src=0)
  # set_tests_properties(TEST_BC PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

  add_test(NAME TEST_MGPU_CC COMMAND cc market
    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES})
  set_tests_properties(TEST_CC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  add_test(NAME TEST_MGPU_SSSP COMMAND sssp market
    ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx ${DEVICES} --undirected --src=0)
  set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  add_test(NAME TEST_MGPU_PR COMMAND pr market
    ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES} --undirected)
  set_tests_properties(TEST_PAGERANK PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  # add_test(NAME TEST_MGPU_TOPK COMMAND topk market
  #  ${gunrock_INCLUDE_DIRS}/dataset/small/bips98_606.mtx ${DEVICES} --undirected)
  # set_tests_properties(TEST_TOPK PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

 endif(GUNROCK_MGPU_TESTS)

# end /* Enable Testing for `ctest` */
