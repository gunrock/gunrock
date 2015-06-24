# gunrock test rig cmake file
# include_directories(${gunrock_INCLUDE_DIRS}/gunrock)

add_executable(simple_interface_test simple_interface_test.c)
target_link_libraries(simple_interface_test gunrock)

add_executable(test_topk test_topk.c)
target_link_libraries(test_topk gunrock)

add_executable(test_bfs test_bfs.c)
target_link_libraries(test_bfs gunrock)

add_executable(test_bc test_bc.c)
target_link_libraries(test_bc gunrock)

add_executable(test_cc test_cc.c)
target_link_libraries(test_cc gunrock)

add_executable(test_sssp test_sssp.c)
target_link_libraries(test_sssp gunrock)

add_executable(test_pr test_pr.c)
target_link_libraries(test_pr gunrock)

add_executable(test_mst test_mst.c)
target_link_libraries(test_mst gunrock)